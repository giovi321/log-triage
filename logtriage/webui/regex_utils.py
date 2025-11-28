from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _lint_regex_input(regex_value: str) -> List[str]:
    issues: List[str] = []
    raw = regex_value or ""
    if not raw.strip():
        issues.append("Pattern cannot be empty.")

    if raw.rstrip().endswith("/"):
        issues.append(r"Trailing slash detected; remove it or escape as \\/.")

    pairs = {"[": "]", "(": ")", "{": "}"}
    opening = set(pairs)
    closing = {v: k for k, v in pairs.items()}
    stack: List[str] = []
    escaped = False
    for ch in raw:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch in opening:
            stack.append(ch)
        elif ch in closing:
            if not stack or stack[-1] != closing[ch]:
                issues.append(f"Unbalanced bracket near '{ch}'.")
                break
            stack.pop()
    if stack:
        issues.append("Unbalanced brackets detected.")

    return issues


def _format_regex_error(pattern: str, err: re.error) -> str:
    position = getattr(err, "pos", None)
    token_hint = ""
    if position is not None and position < len(pattern):
        token = pattern[position]
        token_hint = f" Problem near '{token}' (position {position})."
    escape_hint = " Consider escaping literal characters with \\\\ if needed."
    return f"Regex error: {err.msg}.{token_hint}{escape_hint}"


def _compile_regex_with_feedback(regex_value: str) -> tuple[Optional[re.Pattern[str]], Optional[str]]:
    try:
        return re.compile(regex_value), None
    except re.error as exc:
        return None, _format_regex_error(regex_value, exc)


def _prepare_sample_lines(
    sample_lines: List[str],
    *,
    first_line_number: int = 0,
    max_preview_chars: int = 240,
    max_full_chars: int = 2000,
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for idx, raw_line in enumerate(sample_lines):
        line = raw_line.rstrip("\n") if isinstance(raw_line, str) else str(raw_line)
        clipped_line = line[:max_full_chars]
        was_clipped = len(clipped_line) < len(line)
        preview = clipped_line
        preview_truncated = False
        if len(preview) > max_preview_chars:
            preview = preview[:max_preview_chars] + "…"
            preview_truncated = True

        prepared.append(
            {
                "index": idx + first_line_number,
                "preview": preview,
                "full": clipped_line,
                "is_preview_truncated": preview_truncated,
                "was_clipped": was_clipped,
            }
        )

    return prepared


def _filter_finding_intro_lines(sample_lines: List[str]) -> List[str]:
    """Drop lines that are auto-generated finding headers.

    Regex lab users usually want to target the actual log lines rather than
    the synthetic "[ERROR] finding #123 @ …" headers. Filtering them out keeps
    the sample list focused on actionable content.
    """

    finding_header = re.compile(r"^\[[A-Z]+\]\s+finding #\d+\s+@", re.IGNORECASE)
    return [line for line in sample_lines if not finding_header.match(str(line).strip())]
