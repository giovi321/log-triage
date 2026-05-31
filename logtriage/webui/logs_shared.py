"""Shared helpers for the regex-lab and logs explorer routers.

Regex-lab session state + wizard metadata, regex evaluation (with a
backtracking timeout), and log sample-line loading (tail / identified-errors /
on-disk sample). All read live config from STATE; no app.py import.
"""
from __future__ import annotations

import datetime
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Request

from ..notifications import add_notification
from .db import get_recent_findings_for_module
from .state import STATE
from .shared import (
    normalize_sample_source,
    available_sample_logs,
    format_local_timestamp,
)

REGEX_WIZARD_STEPS = [
    ("pick", "Pick sample lines"),
    ("draft", "Draft regex"),
    ("test", "Run tests"),
    ("save", "Save to pipeline"),
]

_REGEX_TIMEOUT_SECONDS = 5.0


def _get_regex_state(request: Request) -> Dict[str, Any]:
    state = request.session.get("regex_lab_state") or {}
    normalized = {
        "module": state.get("module"),
        "sample_source": state.get("sample_source", "tail"),
        "regex_value": state.get("regex_value", ""),
        "regex_kind": state.get("regex_kind", "error"),
        "matches": state.get("matches", []),
        "step": state.get("step", "pick"),
    }
    return normalized


def _update_regex_state(request: Request, **updates: Any) -> Dict[str, Any]:
    state = _get_regex_state(request)
    if "module" in updates and updates["module"] != state.get("module"):
        state["matches"] = []
        state["step"] = "pick"

    for key, value in updates.items():
        if value is not None:
            state[key] = value

    request.session["regex_lab_state"] = state
    return state


def _regex_wizard_metadata(current_step: str) -> Dict[str, Any]:
    steps = []
    seen_current = False
    for step_id, label in REGEX_WIZARD_STEPS:
        if step_id == current_step:
            status = "active"
            seen_current = True
        elif not seen_current:
            status = "complete"
        else:
            status = "upcoming"
        steps.append({"id": step_id, "label": label, "status": status})
    return {
        "steps": steps,
        "active_label": next((label for sid, label in REGEX_WIZARD_STEPS if sid == current_step), REGEX_WIZARD_STEPS[0][1]),
    }


def _regex_step_hints(step: str) -> List[Dict[str, str]]:
    def _hint(key: str, fallback: str) -> str:
        return STATE.context_hints.get(key, fallback)

    mapping = {
        "pick": [
            {
                "title": "Module context",
                "body": _hint(
                    "modules_path",
                    "Each module points at a path. Switch modules to pull sample lines from different log files.",
                ),
            },
        ],
        "draft": [
            {
                "title": "Capture the right severity",
                "body": _hint("classifier_error_regexes", "Use classifier.error_regexes to flag error patterns."),
            },
            {
                "title": "Ignore the noise",
                "body": _hint("classifier_ignore_regexes", "Add noisy patterns to classifier.ignore_regexes so they are skipped."),
            },
        ],
        "test": [
            {
                "title": "Context windows",
                "body": _hint(
                    "modules_llm_context_prefix_lines",
                    "Context prefix lines influence what the LLM sees alongside a match when enabled.",
                ),
            },
        ],
        "save": [
            {
                "title": "Pipelines own the rules",
                "body": _hint("pipelines_classifier", "Pipelines carry classifier regex lists shared across modules."),
            },
            {
                "title": "Name ties it together",
                "body": _hint(
                    "pipelines_name",
                    "modules.pipeline points at the pipeline name that will receive the saved regex.",
                ),
            },
        ],
    }

    return mapping.get(step, mapping["pick"])


def _build_all_regex_hints() -> Dict[str, List[Dict[str, str]]]:
    return {step: _regex_step_hints(step) for step, _ in REGEX_WIZARD_STEPS}


def _evaluate_regex_against_lines(
    regex_value: str, sample_lines: List[str], first_line_number: int = 0
) -> tuple[list[int], Optional[str]]:
    matches: List[int] = []
    error_msg: Optional[str] = None
    if not regex_value:
        return matches, error_msg

    result: Dict[str, Any] = {}

    def _run() -> None:
        try:
            pattern = re.compile(regex_value)
            result["matches"] = [
                idx + first_line_number
                for idx, line in enumerate(sample_lines)
                if pattern.search(line)
            ]
        except re.error as e:
            result["error"] = f"Regex error: {e}"

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=_REGEX_TIMEOUT_SECONDS)

    if t.is_alive():
        return [], "Regex timed out — pattern may cause catastrophic backtracking."

    if "error" in result:
        error_msg = result["error"]
    else:
        matches = result.get("matches", [])

    return matches, error_msg


def _tail_lines(
    path: Path, max_lines: int = 500, *, max_chars_per_line: int = 4000, offset: int = 0
) -> tuple[List[str], int, int]:
    """Return the tail of a file and the 1-based line number for its first entry.
    
    Args:
        path: Path to the log file
        max_lines: Maximum number of lines to return (default 500)
        max_chars_per_line: Maximum characters per line before truncation
        offset: Number of lines to skip from the end (for pagination).
                offset=0 returns the last max_lines lines.
                offset=500 returns lines before those last 500 lines.
    
    Returns:
        Tuple of (lines, start_line_number, total_lines_in_file)
        - lines: List of log line strings
        - start_line_number: 1-based line number of the first returned line
        - total_lines_in_file: Total number of lines in the file
    """
    if not path.exists() or not path.is_file():
        return [], 1, 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
            total_lines = len(all_lines)
            
            if total_lines == 0:
                return [], 1, 0
            
            # Calculate the range of lines to return
            # offset=0 means get the last max_lines
            # offset=500 means skip the last 500 and get the max_lines before that
            end_idx = total_lines - offset
            start_idx = max(0, end_idx - max_lines)
            
            if end_idx <= 0:
                # Offset is beyond the file, no lines to return
                return [], 1, total_lines
            
            selected_lines = all_lines[start_idx:end_idx]
            
            trimmed: List[str] = []
            for ln in selected_lines:
                normalized = ln.rstrip("\n")
                trimmed.append(normalized[:max_chars_per_line])

            # 1-based line number for the first returned line
            start_line = start_idx + 1
            return trimmed, start_line, total_lines
    except Exception:
        return [], 1, 0


def _error_lines_from_findings(module_obj, max_lines: int = 200) -> List[str]:
    findings = get_recent_findings_for_module(module_obj.name, limit=max_lines)
    if not findings:
        return []

    lines: List[str] = []
    for finding in findings:
        created_at = format_local_timestamp(getattr(finding, "created_at", None))
        header = f"[{getattr(finding, 'severity', '')}] finding #{getattr(finding, 'finding_index', '?')} @ {created_at}"
        lines.append(header)
        
        # Get excerpt lines from the finding
        excerpt = getattr(finding, "excerpt", None) or ""
        excerpt_lines = excerpt.splitlines() if isinstance(excerpt, str) else list(excerpt)
        for line_text in excerpt_lines:
            lines.append(line_text)
            if len(lines) >= max_lines:
                return lines[:max_lines]

    return lines[:max_lines]


def _get_sample_lines_for_module(
    module_obj, sample_source: str, max_lines: int = 500, offset: int = 0
) -> tuple[List[str], int, int, Optional[str]]:
    """Get sample lines from a module's log file with pagination support.
    
    Args:
        module_obj: The module configuration object
        sample_source: Source type ('tail', 'errors', or 'sample:...')
        max_lines: Maximum number of lines to return (default 500)
        offset: Number of lines to skip from the end for pagination
    
    Returns:
        Tuple of (lines, start_line_number, total_lines, error_message)
    """
    if module_obj is None:
        return [], 1, 0, None

    source = normalize_sample_source(sample_source)
    if source == "errors":
        if not STATE.db_status.get("connected"):
            return [], 1, 0, "Database not connected; cannot load identified errors."
        lines = _error_lines_from_findings(module_obj, max_lines=max_lines)
        return lines, 1, len(lines), None

    if source.startswith("sample:"):
        sample_logs = available_sample_logs()
        entry = next((item for item in sample_logs if item.get("value") == source), None)
        if entry is None:
            return [], 1, 0, "Sample log not found on disk."
        lines, start_line, total = _tail_lines(entry["path"], max_lines=max_lines, offset=offset)
        if not lines:
            return [], 1, total, "Sample log is empty or unreadable."
        return lines, start_line, total, None

    lines, start_line, total = _tail_lines(Path(module_obj.path), max_lines=max_lines, offset=offset)
    return lines, start_line, total, None
