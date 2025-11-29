import re
from typing import List

from ..models import PipelineConfig, Severity, Finding


def _build_excerpt(
    lines: List[str],
    offset: int,
    context_prefix_lines: int,
    context_suffix_lines: int,
    excerpt_limit: int,
    prefix_lines: List[str],
) -> List[str]:
    """Build context excerpt around a matching line.
    
    Includes prefix and suffix context lines while respecting the excerpt limit.
    Ensures the matching line is included in the excerpt.
    
    Args:
        lines: All lines in the current chunk
        offset: Index of the matching line
        context_prefix_lines: Number of lines to include before the match
        context_suffix_lines: Number of lines to include after the match
        excerpt_limit: Maximum total lines to include in excerpt
        prefix_lines: Lines from previous chunks for context
        
    Returns:
        List of lines forming the excerpt
    """
    excerpt_limit = max(1, excerpt_limit)
    prefix_lines = prefix_lines or []

    before_current = lines[max(0, offset - context_prefix_lines) : offset]
    missing_prefix = max(0, context_prefix_lines - len(before_current))
    before_context = prefix_lines[-missing_prefix:] + before_current

    match_line = lines[offset]
    after_context = lines[offset + 1 : offset + 1 + context_suffix_lines]

    excerpt = before_context + [match_line] + after_context

    if len(excerpt) > excerpt_limit:
        match_idx = len(before_context)
        start = max(0, match_idx - excerpt_limit // 2)
        end = start + excerpt_limit

        if end > len(excerpt):
            end = len(excerpt)
            start = max(0, end - excerpt_limit)

        if not (start <= match_idx < end):
            start = max(0, match_idx - excerpt_limit + 1)
            end = start + excerpt_limit

        excerpt = excerpt[start:end]

    return excerpt


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
    context_suffix_lines: int = 0,
    prefix_lines: List[str] | None = None,
) -> List[Finding]:
    """Classify lines using regex patterns for errors and warnings.

    Creates a separate finding for each line that matches error or warning patterns.
    Ignore patterns are applied before checking error/warning patterns.
    
    Args:
        pcfg: Pipeline configuration with regex patterns
        file_path: Path to the file being processed
        pipeline_name: Name of the processing pipeline
        lines: Lines to analyze
        start_line: Line number of the first line in the chunk
        excerpt_limit: Maximum lines to include in finding excerpts
        context_prefix_lines: Number of context lines before matches
        context_suffix_lines: Number of context lines after matches
        prefix_lines: Previous lines for additional context
        
    Returns:
        List of findings for matched patterns
    """

    findings: List[Finding] = []
    ignore_res = pcfg.classifier_ignore_regexes or []

    prefix_lines = prefix_lines or []

    for offset, line in enumerate(lines):
        current_line = start_line + offset
        if any(r.search(line) for r in ignore_res):
            continue

        for r in pcfg.classifier_error_regexes:
            if not r:
                continue
            for match in r.finditer(line):
                excerpt = _build_excerpt(
                    lines,
                    offset,
                    context_prefix_lines,
                    context_suffix_lines,
                    excerpt_limit,
                    prefix_lines,
                )
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
                excerpt = _build_excerpt(
                    lines,
                    offset,
                    context_prefix_lines,
                    context_suffix_lines,
                    excerpt_limit,
                    prefix_lines,
                )
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
