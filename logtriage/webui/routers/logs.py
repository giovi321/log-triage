"""Logs explorer + finding-management routes (the unified AI-logs view)."""
from __future__ import annotations

import datetime
import re
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ...models import Finding, Severity
from ...notifications import add_notification
from ..auth import get_current_user
from ..db import (
    delete_all_findings,
    delete_finding_by_id,
    delete_findings_by_ids,
    delete_findings_for_module,
    delete_findings_matching_regex,
    get_finding_by_id,
    get_module_stats,
    get_next_finding_index,
    get_recent_findings_for_module,
    store_finding,
    update_finding_llm_data,
    update_finding_severity,
)
from ..ingestion_status import _derive_ingestion_status
from ..regex_utils import _lint_regex_input
from ..state import STATE
from ..shared import (
    templates,
    build_modules_from_config,
    format_local_timestamp as _format_local_timestamp,
    normalize_sample_source as _normalize_sample_source,
    sample_source_options as _sample_source_options,
    sample_source_label as _sample_source_label,
    select_provider_name as _select_provider_name,
    suggest_regex_from_line as _suggest_regex_from_line,
    finding_excerpt_preview as _finding_excerpt_preview,
)
from ..logs_shared import _tail_lines, _get_sample_lines_for_module
from .. import config_io

router = APIRouter()

# Severities that count as "findings" in the logs explorer (matches the triage set).
SEVERITY_CHOICES = ["CRITICAL", "ERROR", "WARNING"]


def _logs_redirect(
    module: Optional[str],
    message: Optional[str] = None,
    error: Optional[str] = None,
    tail_filter: Optional[str] = None,
    issue_filter: Optional[str] = None,
    sample_source: Optional[str] = None,
):
    params = {}
    if module:
        params["module"] = module
    if tail_filter:
        params["tail_filter"] = tail_filter
    if issue_filter:
        params["issue_filter"] = issue_filter
    if sample_source:
        params["sample_source"] = sample_source
    if message:
        params["message"] = message
    if error:
        params["error"] = error

    query = urllib.parse.urlencode(params)
    url = "/ai-logs"
    if query:
        url = f"{url}?{query}"
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)


def _build_log_view_state(
    module_obj,
    tail_filter: str,
    issue_filter: str,
    *,
    sample_lines_override: Optional[List[str]] = None,
    sample_start_line: int = 0,
    total_lines: int = 0,
):
    sample_lines: List[str] = []
    recent_findings = []
    finding_tail: List[Dict[str, Any]] = []
    available_severities: List[str] = []
    tail_filter_normalized = (tail_filter or "all").upper()
    issue_filter_normalized = (issue_filter or "all").upper()
    severity_choices = list(SEVERITY_CHOICES)
    had_finding_tail = False
    had_recent_findings = False
    if module_obj is not None:
        if sample_lines_override is not None:
            sample_lines = list(sample_lines_override)
        else:
            sample_lines, sample_start_line, total_lines = _tail_lines(
                Path(module_obj.path), max_lines=500
            )
        recent_findings = get_recent_findings_for_module(module_obj.name, limit=50)
        had_recent_findings = bool(recent_findings)
        finding_tail = _build_finding_tail(
            sample_lines,
            recent_findings,
            first_line_number=sample_start_line,
            module_path=Path(module_obj.path),
        )
        finding_line_examples = _finding_line_examples(finding_tail)
        had_finding_tail = bool(finding_tail)
        if issue_filter_normalized == "ACTIVE":
            recent_findings = [
                c
                for c in recent_findings
                if str(getattr(c, "severity", "")).upper() in SEVERITY_CHOICES
            ]
            finding_tail = _filter_finding_tail(
                finding_tail,
                tail_filter="ALL",
                extra_predicate=lambda section: str(
                    getattr(section.get("finding"), "severity", "")
                ).upper()
                in SEVERITY_CHOICES,
            )

        # Extract available severities from findings (handle unified view)
        for section in finding_tail:
            if section.get("unified_view"):
                # Unified view: findings are attached to individual lines
                for line_entry in section.get("lines", []):
                    finding = line_entry.get("finding")
                    if not finding:
                        continue
                    sev = str(getattr(finding, "severity", "")).upper()
                    if sev and sev not in available_severities:
                        available_severities.append(sev)
                    if sev and sev not in severity_choices:
                        severity_choices.append(sev)
            else:
                # Legacy: finding attached to section
                finding = section.get("finding")
                if not finding:
                    continue
                sev = str(getattr(finding, "severity", "")).upper()
                if sev and sev not in available_severities:
                    available_severities.append(sev)
                if sev and sev not in severity_choices:
                    severity_choices.append(sev)

        finding_tail = _filter_finding_tail(
            finding_tail, tail_filter_normalized, extra_predicate=None
        )

    else:
        finding_line_examples = {}

    return {
        "sample_lines": sample_lines,
        "recent_findings": recent_findings,
        "had_recent_findings": had_recent_findings,
        "finding_tail": finding_tail,
        "had_finding_tail": had_finding_tail,
        "tail_filter": tail_filter_normalized,
        "issue_filter": issue_filter_normalized,
        "tail_severities": available_severities,
        "severity_choices": severity_choices,
        "finding_examples": finding_line_examples,
        "sample_start_line": sample_start_line,
        "total_lines": total_lines,
    }


def _build_finding_tail(
    sample_lines: List[str],
    recent_findings: List,
    *,
    first_line_number: int = 0,
    module_path: Optional[Path] = None,
    context_window: int = 2,
) -> List[Dict[str, Any]]:
    """Build a unified log view with findings mapped to single lines.
    
    Returns a single section containing all log lines, with each line optionally
    associated with a finding. This allows the UI to show findings in context.
    """
    indexed_lines = [
        {"index": idx + first_line_number, "text": line, "finding": None}
        for idx, line in enumerate(sample_lines)
    ]
    
    # Create a lookup by line index for quick access
    line_by_index: Dict[int, Dict[str, Any]] = {
        entry["index"]: entry for entry in indexed_lines
    }

    sorted_findings = sorted(
        recent_findings,
        key=lambda c: getattr(c, "created_at", datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)),
        reverse=True,
    )

    def _load_from_file(start: int, end: int) -> List[Dict[str, Any]]:
        if not module_path or not module_path.exists():
            return []

        loaded: List[Dict[str, Any]] = []
        try:
            with module_path.open("r", encoding="utf-8", errors="ignore") as fh:
                for idx, raw in enumerate(fh, start=1):
                    if idx < start:
                        continue
                    if idx > end:
                        break
                    loaded.append({"index": idx, "text": raw.rstrip("\n")})
        except Exception:
            return []

        return loaded

    def _primary_line_for_finding(finding) -> Optional[Dict[str, Any]]:
        """Get the single primary line that represents this finding.
        
        The line_start attribute is the authoritative source for which line
        matched the regex. The excerpt may contain context lines before/after
        the match, so we should NOT use the first excerpt line as the primary.
        """
        start = getattr(finding, "line_start", None)
        excerpt = getattr(finding, "excerpt", None) or ""
        excerpt_lines = excerpt.splitlines() if isinstance(excerpt, str) else list(excerpt)
        
        # line_start is the authoritative matching line number
        if not isinstance(start, int):
            return None
        
        # Try to find the line in our indexed lines first
        if start in line_by_index:
            return {"index": start, "text": line_by_index[start]["text"]}
        
        # Try to load the actual line from file
        loaded = _load_from_file(start, start)
        if loaded:
            return loaded[0]
        
        # Last resort: if we have an excerpt, try to find the matching line
        # by looking for a line that matches the rule_id pattern
        rule_id = getattr(finding, "rule_id", None)
        if rule_id and excerpt_lines:
            try:
                pattern = re.compile(rule_id)
                for line_text in excerpt_lines:
                    if pattern.search(line_text):
                        return {"index": start, "text": line_text}
            except re.error:
                pass
            # If no pattern match, use the middle line of excerpt as best guess
            # (since context is added before and after the match)
            if excerpt_lines:
                middle_idx = len(excerpt_lines) // 2
                return {"index": start, "text": excerpt_lines[middle_idx]}
        
        return None

    # Map findings to their primary lines
    for finding in sorted_findings:
        primary = _primary_line_for_finding(finding)
        if not primary:
            continue
        
        line_index = primary.get("index")
        if line_index is not None and line_index in line_by_index:
            # Associate finding with this line (only if not already associated)
            if line_by_index[line_index]["finding"] is None:
                line_by_index[line_index]["finding"] = finding
        else:
            # Line not in current view - add it if we have an index
            if line_index is not None:
                new_entry = {
                    "index": line_index,
                    "text": primary.get("text", ""),
                    "finding": finding,
                }
                # Insert in correct position
                inserted = False
                for i, entry in enumerate(indexed_lines):
                    if entry["index"] > line_index:
                        indexed_lines.insert(i, new_entry)
                        line_by_index[line_index] = new_entry
                        inserted = True
                        break
                if not inserted:
                    indexed_lines.append(new_entry)
                    line_by_index[line_index] = new_entry

    # Return a single section with all lines
    return [{"finding": None, "lines": indexed_lines, "unified_view": True}]


def _finding_line_examples(sections: List[Dict[str, Any]]) -> Dict[int, str]:
    """Extract example lines for each finding from the unified view."""
    examples: Dict[int, str] = {}
    for section in sections:
        # Handle unified view where findings are attached to individual lines
        if section.get("unified_view"):
            for line_entry in section.get("lines", []):
                finding = line_entry.get("finding")
                if not finding:
                    continue
                finding_id = getattr(finding, "id", None)
                if finding_id is None or finding_id in examples:
                    continue
                text = line_entry.get("text", "")
                if text:
                    examples[int(finding_id)] = text
        else:
            # Legacy handling for non-unified sections
            finding = section.get("finding")
            if not finding:
                continue
            finding_id = getattr(finding, "id", None)
            if finding_id is None or finding_id in examples:
                continue
            lines = section.get("lines") or []
            texts: List[str] = []
            for entry in lines:
                text = entry.get("text") if isinstance(entry, dict) else None
                if text:
                    texts.append(text)
            if texts:
                examples[int(finding_id)] = "\n".join(texts)
    return examples


def _filter_finding_tail(
    sections: List[Dict[str, Any]],
    tail_filter: str,
    extra_predicate=None,
) -> List[Dict[str, Any]]:
    """Filter sections/lines based on severity filter.
    
    For unified view, this filters the lines within the section rather than
    filtering entire sections. When a severity filter is active, only lines
    with findings matching that severity are shown (not all lines for context).
    """
    if not sections:
        return []

    normalized = (tail_filter or "ALL").upper()
    if normalized in {"", "ALL"} and extra_predicate is None:
        return sections

    filtered: List[Dict[str, Any]] = []
    for section in sections:
        # Handle unified view - filter lines within the section
        if section.get("unified_view"):
            # For unified view, we need to filter the lines based on their findings
            filtered_lines = []
            for line_entry in section.get("lines", []):
                finding = line_entry.get("finding")
                if finding:
                    severity = str(getattr(finding, "severity", "")).upper()
                    severity_match = normalized in {"", "ALL"} or severity == normalized
                    extra_match = extra_predicate(line_entry) if extra_predicate else True
                    if severity_match and extra_match:
                        filtered_lines.append(line_entry)
                elif normalized in {"", "ALL"}:
                    # Only keep non-finding lines when showing all severities
                    filtered_lines.append(line_entry)
                # When filtering by specific severity, skip non-finding lines
            
            # Only add the section if we have any lines that match
            if filtered_lines:
                filtered_section = section.copy()
                filtered_section["lines"] = filtered_lines
                filtered.append(filtered_section)
        else:
            # Legacy handling for non-unified sections
            finding = section.get("finding")
            severity = str(getattr(finding, "severity", "")).upper() if finding else ""
            severity_match = normalized in {"", "ALL"} or severity == normalized
            extra_match = extra_predicate(section) if extra_predicate else True
            if severity_match and extra_match:
                filtered.append(section)

    return filtered


@router.get("/ai-logs", name="ai_logs")
async def ai_logs(
    request: Request,
    module: Optional[str] = None,
    tail_filter: str = "all",
    issue_filter: str = "all",
    sample_source: str = "tail",
    message: Optional[str] = None,
    error: Optional[str] = None,
):
    """Render the unified AI logs explorer that replaces the legacy logs/LLM pages."""
    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    modules = build_modules_from_config()
    stats = get_module_stats(modules)
    ingestion_status = (
        _derive_ingestion_status(modules, freshness_minutes=STATE.settings.staleness_minutes)
        if modules else None
    )
    safe_sample_source = _normalize_sample_source(sample_source)
    module_obj = None
    open_findings_count = None
    if modules:
        if module:
            module_obj = next((m for m in modules if m.name == module), None)
        if module_obj is None:
            # Default to first enabled module alphabetically
            enabled_modules = sorted(
                [m for m in modules if m.enabled], key=lambda m: m.name
            )
            module_obj = enabled_modules[0] if enabled_modules else None

    sample_lines, sample_start_line, total_lines, sample_error = _get_sample_lines_for_module(
        module_obj, safe_sample_source, max_lines=500
    )
    log_state = _build_log_view_state(
        module_obj,
        tail_filter,
        issue_filter,
        sample_lines_override=sample_lines,
        sample_start_line=sample_start_line,
        total_lines=total_lines,
    )

    # Count findings displayed in current view (from finding_tail)
    if module_obj and STATE.db_status.get("connected"):
        displayed_finding_ids = set()
        for section in log_state.get("finding_tail", []):
            if section.get("unified_view"):
                for line_entry in section.get("lines", []):
                    finding = line_entry.get("finding")
                    if finding:
                        fid = getattr(finding, "id", None)
                        if fid is not None:
                            displayed_finding_ids.add(fid)
            else:
                finding = section.get("finding")
                if finding:
                    fid = getattr(finding, "id", None)
                    if fid is not None:
                        displayed_finding_ids.add(fid)
        open_findings_count = len(displayed_finding_ids)

    llm_defaults = STATE.llm_defaults
    provider_name = _select_provider_name(module_obj)
    provider_cfg = llm_defaults.providers.get(provider_name) if provider_name else None
    provider_settings = {
        name: {
            "temperature": p.temperature,
            "top_p": p.top_p,
            "max_output_tokens": p.max_output_tokens,
            "max_excerpt_lines": p.max_excerpt_lines,
            "model": p.model,
        }
        for name, p in llm_defaults.providers.items()
    }

    seed_prompt = None
    module_prompt_template = None
    if module_obj is not None and log_state["recent_findings"]:
        max_lines = provider_cfg.max_excerpt_lines if provider_cfg else 20
        seed_prompt = _finding_excerpt_preview(log_state["recent_findings"][0], max_lines)
    if module_obj and getattr(module_obj, "llm", None):
        template_path = getattr(module_obj.llm, "prompt_template_path", None)
        if template_path:
            try:
                module_prompt_template = Path(template_path).read_text()
            except Exception:
                module_prompt_template = None

    regex_presets: List[Dict[str, str]] = []

    return templates.TemplateResponse(
        "ai_logs.html",
        {
            "request": request,
            "username": username,
            "modules": modules,
            "current_module": module_obj,
            **log_state,
            "db_status": STATE.db_status,
            "message": message,
            "error": error or sample_error,
            "providers": list(llm_defaults.providers.values()),
            "provider_settings": provider_settings,
            "selected_provider": provider_name,
            "seed_prompt": seed_prompt,
            "module_prompt_template": module_prompt_template,
            "regex_presets": regex_presets,
            "sample_source": safe_sample_source,
            "stats": stats,
            "ingestion_status": ingestion_status,
            "open_findings_count": open_findings_count,
        },
    )


@router.get("/api/log-lines", name="api_log_lines")
async def api_log_lines(
    request: Request,
    module: str,
    offset: int = 0,
    limit: int = 500,
    sample_source: str = "tail",
):
    """API endpoint to load more log lines with pagination.
    
    Returns JSON with lines and pagination info. Used by the "Load more" button.
    """
    username = get_current_user(request, STATE.settings)
    if not username:
        return JSONResponse(
            {"error": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED
        )

    modules = build_modules_from_config()
    module_obj = next((m for m in modules if m.name == module), None)
    if module_obj is None:
        return JSONResponse({"error": "Module not found"}, status_code=404)

    safe_sample_source = _normalize_sample_source(sample_source)
    lines, start_line, total_lines, error = _get_sample_lines_for_module(
        module_obj, safe_sample_source, max_lines=limit, offset=offset
    )

    if error:
        return JSONResponse({"error": error}, status_code=400)

    # Get findings for these lines
    recent_findings = get_recent_findings_for_module(module_obj.name, limit=100)
    finding_tail = _build_finding_tail(
        lines,
        recent_findings,
        first_line_number=start_line,
        module_path=Path(module_obj.path),
    )

    # Convert finding_tail to JSON-serializable format
    lines_data = []
    for section in finding_tail:
        if section.get("unified_view"):
            for line_entry in section.get("lines", []):
                finding = line_entry.get("finding")
                line_data = {
                    "index": line_entry.get("index"),
                    "text": line_entry.get("text", ""),
                    "finding": None,
                }
                if finding:
                    line_data["finding"] = {
                        "id": getattr(finding, "id", None),
                        "finding_index": getattr(finding, "finding_index", None),
                        "severity": str(getattr(finding, "severity", "")),
                        "reason": getattr(finding, "reason", ""),
                        "rule_id": getattr(finding, "rule_id", None),
                        "created_at": _format_local_timestamp(getattr(finding, "created_at", None)),
                        "llm_response_content": getattr(finding, "llm_response_content", None),
                        "llm_error": getattr(finding, "llm_error", None),
                        "llm_provider": getattr(finding, "llm_provider", None),
                        "llm_model": getattr(finding, "llm_model", None),
                    }
                lines_data.append(line_data)

    # Calculate if there are more lines to load
    has_more = start_line > 1

    return JSONResponse({
        "lines": lines_data,
        "start_line": start_line,
        "total_lines": total_lines,
        "has_more": has_more,
        "next_offset": offset + len(lines),
    })


@router.post("/logs/db/flush", name="flush_logs_db")
async def flush_logs_db(
    request: Request,
    module: Optional[str] = Form(None),
    tail_filter: str = Form("all"),
    issue_filter: str = Form("all"),
    sample_source: str = Form("tail"),
):
    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not STATE.db_status.get("connected"):
        return _logs_redirect(
            module,
            error="Database not connected.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    try:
        deleted = delete_all_findings()
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Failed to flush database: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    msg = "Database already empty." if deleted == 0 else f"Deleted {deleted} stored finding(s)."
    return _logs_redirect(
        module,
        message=msg,
        tail_filter=tail_filter,
        issue_filter=issue_filter,
        sample_source=sample_source,
    )


@router.post("/logs/db/flush-module", name="flush_module_logs")
async def flush_module_logs(
    request: Request,
    module: str = Form(...),
    tail_filter: str = Form("all"),
    issue_filter: str = Form("all"),
    sample_source: str = Form("tail"),
    confirm_token: str = Form(""),
):
    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not module:
        return _logs_redirect(
            module,
            error="Select a module before flushing findings.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    if (confirm_token or "").strip().lower() != "confirm":
        return _logs_redirect(
            module,
            error="Enter 'confirm' to proceed with flushing findings.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    if not STATE.db_status.get("connected"):
        return _logs_redirect(
            module,
            error="Database not connected.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    modules = {m.name for m in build_modules_from_config()}
    if module not in modules:
        return _logs_redirect(
            None,
            error=f"Unknown module '{module}'.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    try:
        deleted = delete_findings_for_module(module)
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Failed to flush findings: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    msg = (
        f"No stored findings for {module}."
        if deleted == 0
        else f"Deleted {deleted} stored finding(s) for {module}."
    )
    return _logs_redirect(
        module,
        message=msg,
        tail_filter=tail_filter,
        issue_filter=issue_filter,
        sample_source=sample_source,
    )


@router.post("/logs/finding/delete-multiple", name="delete_selected_findings")
async def delete_selected_findings(
    request: Request,
    module: Optional[str] = Form(None),
    tail_filter: str = Form("all"),
    issue_filter: str = Form("all"),
    sample_source: str = Form("tail"),
    finding_ids: str = Form(""),
):
    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not STATE.db_status.get("connected"):
        return _logs_redirect(
            module,
            error="Database not connected.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    raw_ids = [part.strip() for part in (finding_ids or "").split(",") if part.strip()]
    parsed_ids = []
    for raw in raw_ids:
        try:
            parsed_ids.append(int(raw))
        except ValueError:
            continue

    if not parsed_ids:
        return _logs_redirect(
            module,
            error="Select at least one finding to delete.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    try:
        deleted = delete_findings_by_ids(parsed_ids)
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Failed to delete entries: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    if not deleted:
        return _logs_redirect(
            module,
            error="No matching findings were removed.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    return _logs_redirect(
        module,
        message=f"Deleted {deleted} finding(s).",
        tail_filter=tail_filter,
        issue_filter=issue_filter,
        sample_source=sample_source,
    )


@router.post("/logs/finding/delete", name="delete_finding")
async def delete_finding(
    request: Request,
    finding_id: int = Form(...),
    module: Optional[str] = Form(None),
    tail_filter: str = Form("all"),
    issue_filter: str = Form("all"),
    sample_source: str = Form("tail"),
):
    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not STATE.db_status.get("connected"):
        return _logs_redirect(
            module,
            error="Database not connected.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    try:
        deleted = delete_finding_by_id(finding_id)
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Failed to delete entry: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    if not deleted:
        return _logs_redirect(
            module,
            error="Entry not found.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    return _logs_redirect(
        module,
        message="Finding removed.",
        tail_filter=tail_filter,
        issue_filter=issue_filter,
        sample_source=sample_source,
    )


@router.post("/logs/finding/severity", name="change_finding_severity")
async def change_finding_severity(
    request: Request,
    finding_id: int = Form(...),
    severity: str = Form(...),
    module: Optional[str] = Form(None),
    tail_filter: str = Form("all"),
    issue_filter: str = Form("all"),
    sample_source: str = Form("tail"),
):
    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not STATE.db_status.get("connected"):
        return _logs_redirect(
            module,
            error="Database not connected.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    normalized = (severity or "").upper()
    if normalized not in SEVERITY_CHOICES:
        return _logs_redirect(
            module,
            error="Invalid severity provided.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    try:
        updated = update_finding_severity(finding_id, normalized)
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Failed to update severity: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    if not updated:
        return _logs_redirect(
            module,
            error="Entry not found.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    return _logs_redirect(
        module,
        message=f"Finding severity updated to {normalized}.",
        tail_filter=tail_filter,
        issue_filter=issue_filter,
        sample_source=sample_source,
    )


@router.post("/logs/finding/manual", name="create_manual_finding")
async def create_manual_finding(
    request: Request,
    module: Optional[str] = Form(None),
    severity: str = Form("ERROR"),
    message: Optional[str] = Form(None),
    lines: str = Form(""),
    line_indexes: str = Form(""),
    tail_filter: str = Form("all"),
    issue_filter: str = Form("all"),
    sample_source: str = Form("tail"),
):
    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not STATE.db_status.get("connected"):
        return _logs_redirect(
            module,
            error="Database not connected.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    modules = {m.name: m for m in build_modules_from_config()}
    module_obj = modules.get(module or "")
    if module_obj is None:
        return _logs_redirect(
            None,
            error="Select a module before creating a finding.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    normalized = (severity or "").upper()
    if normalized not in SEVERITY_CHOICES:
        return _logs_redirect(
            module_obj.name,
            error="Invalid severity provided.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    selected_lines = [ln for ln in (lines.split("\n") if lines else []) if ln.strip()]
    if not selected_lines:
        return _logs_redirect(
            module_obj.name,
            error="Select at least one log line to create a finding.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    try:
        severity_value = Severity.from_string(normalized)
    except Exception:
        severity_value = Severity.WARNING

    message_text = (message or "").strip() or selected_lines[0]

    parsed_indexes = []
    if line_indexes:
        for raw in line_indexes.split(","):
            try:
                parsed_indexes.append(int(raw))
            except ValueError:
                continue
    line_start = min(parsed_indexes) if parsed_indexes else 0
    line_end = max(parsed_indexes) if parsed_indexes else line_start + len(selected_lines) - 1

    finding_index = get_next_finding_index(module_obj.name)
    finding = Finding(
        file_path=Path(module_obj.path),
        pipeline_name=getattr(module_obj, "pipeline_name", None) or "",
        finding_index=finding_index,
        severity=severity_value,
        message=message_text,
        line_start=line_start,
        line_end=line_end,
        rule_id=None,
        excerpt=selected_lines,
        needs_llm=False,
    )

    try:
        store_finding(module_obj.name, finding)
    except Exception as exc:
        return _logs_redirect(
            module_obj.name,
            error=f"Failed to store finding: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    return _logs_redirect(
        module_obj.name,
        message="Manual finding recorded.",
        tail_filter=tail_filter,
        issue_filter=issue_filter,
        sample_source=sample_source,
    )


@router.post("/logs/finding/opinion", name="record_finding_opinion")
async def record_finding_opinion(
    request: Request,
    finding_id: int = Form(...),
    provider: str = Form(""),
    model: str = Form(""),
    content: str = Form(""),
    prompt_tokens: Optional[int] = Form(None),
    completion_tokens: Optional[int] = Form(None),
):
    username = get_current_user(request, STATE.settings)
    if not username:
        return JSONResponse({"error": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED)

    if not STATE.db_status.get("connected"):
        return JSONResponse({"error": "Database not connected."}, status_code=status.HTTP_400_BAD_REQUEST)

    finding = get_finding_by_id(finding_id)
    if finding is None:
        return JSONResponse({"error": "Finding not found."}, status_code=status.HTTP_404_NOT_FOUND)

    def _to_int(value):
        if value is None:
            return None
        try:
            stripped = str(value).strip()
            return int(stripped) if stripped else None
        except Exception:
            return None

    try:
        ok = update_finding_llm_data(
            finding_id,
            provider=provider or None,
            model=model or None,
            content=content or None,
            prompt_tokens=_to_int(prompt_tokens),
            completion_tokens=_to_int(completion_tokens),
        )
    except Exception as exc:
        return JSONResponse({"error": f"Failed to store AI opinion: {exc}"}, status_code=500)

    if not ok:
        return JSONResponse({"error": "Failed to store AI opinion."}, status_code=400)

    return JSONResponse({"message": "AI opinion stored."})


@router.post("/logs/finding/false_positive", name="mark_false_positive")
async def mark_false_positive(
    request: Request,
    finding_id: int = Form(...),
    module: Optional[str] = Form(None),
    tail_filter: str = Form("all"),
    issue_filter: str = Form("all"),
    sample_source: str = Form("tail"),
    sample_line: Optional[str] = Form(None),
    regex_value: Optional[str] = Form(None),
):

    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not STATE.db_status.get("connected"):
        return _logs_redirect(
            module,
            error="Database not connected.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    finding = get_finding_by_id(finding_id)
    if finding is None:
        return _logs_redirect(
            module,
            error="Entry not found.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    regex_source = sample_line or getattr(finding, "reason", "") or ""
    regex_value = (regex_value or "").strip() or (
        _suggest_regex_from_line(regex_source) if regex_source else None
    )

    if not regex_value:
        return _logs_redirect(
            module,
            error="No sample available to build an ignore rule.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    regex_issues = _lint_regex_input(regex_value)
    if regex_issues:
        return _logs_redirect(
            module,
            error=" ".join(regex_issues),
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    try:
        cfg_dict = yaml.safe_load(STATE.config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Failed to read config: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    pipeline_name = getattr(finding, "pipeline_name", None)
    if not pipeline_name:
        modules = build_modules_from_config()
        mod_obj = next((m for m in modules if m.name == getattr(finding, "module_name", None)), None)
        pipeline_name = getattr(mod_obj, "pipeline_name", None)

    pipelines = cfg_dict.get("pipelines", []) or []
    pipeline_entry = next((p for p in pipelines if p.get("name") == pipeline_name), None)
    if pipeline_entry is None:
        return _logs_redirect(
            module,
            error="Pipeline not found in config; cannot add ignore rule.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    classifier = pipeline_entry.setdefault("classifier", {})
    ignore_list = classifier.get("ignore_regexes")
    if ignore_list is None or not isinstance(ignore_list, list):
        ignore_list = []
        classifier["ignore_regexes"] = ignore_list

    if regex_value and regex_value not in ignore_list:
        ignore_list.append(regex_value)

    try:
        config_io.save_config_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Failed to write config: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    (STATE.reload_callback or config_io.reload_from_disk)()

    try:
        removed_count = delete_findings_matching_regex(regex_value, pipeline_name=pipeline_name)
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Ignore rule saved, but failed to remove matching findings: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    if not removed_count:
        try:
            deleted = delete_finding_by_id(finding_id)
        except Exception as exc:
            return _logs_redirect(
                module,
                error=f"Ignore rule saved, but failed to remove finding: {exc}",
                tail_filter=tail_filter,
                issue_filter=issue_filter,
                sample_source=sample_source,
            )

        if not deleted:
            return _logs_redirect(
                module,
                error="Ignore rule saved, but finding was not removed.",
                tail_filter=tail_filter,
                issue_filter=issue_filter,
                sample_source=sample_source,
            )

    success_message = (
        "Marked as false positive, removed from findings, and added to ignore rules."
    )
    if removed_count:
        success_message = (
            f"Marked as false positive, removed {removed_count} finding(s) matching the "
            "ignore rule, and added it to ignore rules."
        )

    return _logs_redirect(
        module,
        message=success_message,
        tail_filter=tail_filter,
        issue_filter=issue_filter,
        sample_source=sample_source,
    )
