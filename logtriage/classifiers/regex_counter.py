import re
from typing import List

from ..models import PipelineConfig, Severity, Finding


def _format_match_text(match_text: str, max_len: int = 120) -> str:
    """Return a single-line, human-friendly snippet of a regex match."""

    sanitized = match_text.replace("\n", "\\n")
    if len(sanitized) > max_len:
        return sanitized[: max_len - 3] + "..."
    return sanitized


def classify_regex_counter(
    pcfg: PipelineConfig,
    file_path,
    pipeline_name: str,
    lines: List[str],
    start_line: int = 1,
    excerpt_limit: int = 20,
    context_prefix_lines: int = 0,
) -> List[Finding]:
    """Emit one finding per matching rule line.

    Ignore patterns are applied before checking error/warning patterns.
    """

    findings: List[Finding] = []
    ignore_res = pcfg.classifier_ignore_regexes or []

    for offset, line in enumerate(lines):
        current_line = start_line + offset
        if any(r.search(line) for r in ignore_res):
            continue

        for r in pcfg.classifier_error_regexes:
            if not r:
                continue
            for match in r.finditer(line):
                excerpt_start = max(0, offset - context_prefix_lines)
                excerpt = lines[excerpt_start : offset + 1]
                if len(excerpt) > excerpt_limit:
                    excerpt = excerpt[-excerpt_limit:]
                match_text = _format_match_text(match.group(0))
                findings.append(
                    Finding(
                        file_path=file_path,
                        pipeline_name=pipeline_name,
                        finding_index=len(findings),
                        severity=Severity.ERROR,
                        message=f"Matched error pattern /{r.pattern}/ on \"{match_text}\"",
                        line_start=current_line,
                        line_end=current_line,
                        rule_id=r.pattern,
                        excerpt=excerpt,
                    )
                )

        for r in pcfg.classifier_warning_regexes:
            if not r:
                continue
            for match in r.finditer(line):
                excerpt_start = max(0, offset - context_prefix_lines)
                excerpt = lines[excerpt_start : offset + 1]
                if len(excerpt) > excerpt_limit:
                    excerpt = excerpt[-excerpt_limit:]
                match_text = _format_match_text(match.group(0))
                findings.append(
                    Finding(
                        file_path=file_path,
                        pipeline_name=pipeline_name,
                        finding_index=len(findings),
                        severity=Severity.WARNING,
                        message=f"Matched warning pattern /{r.pattern}/ on \"{match_text}\"",
                        line_start=current_line,
                        line_end=current_line,
                        rule_id=r.pattern,
                        excerpt=excerpt,
                    )
                )

    return findings
