import re
from typing import List, Sequence, Tuple

from ..models import PipelineConfig, Severity, Finding
from .regex_counter import _build_excerpt


def _split_runs(
    lines: List[str],
    markers: Sequence[re.Pattern],
) -> List[Tuple[int, List[str]]]:
    """Split rsnapshot logs into runs using marker regexes.

    Returns a list of (start_offset, chunk_lines) tuples.
    """

    if not markers:
        return [(0, lines)] if lines else []

    runs: List[Tuple[int, List[str]]] = []
    current: List[str] = []
    current_start = 0

    def _flush():
        nonlocal current
        if current:
            runs.append((current_start, current))
        current = []

    for idx, line in enumerate(lines):
        if any(r.search(line) for r in markers):
            _flush()
            current_start = idx
            current = [line]
        else:
            if not current:
                current_start = idx
            current.append(line)

    _flush()
    return runs


def classify_rsnapshot_basic(
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
    """Heuristic classifier for rsnapshot runs that emits per-line findings."""
    joined_all = "\n".join(lines)

    default_markers = [
        pcfg.grouping_start_regex,
        pcfg.grouping_end_regex,
        re.compile(r"^={5,}$"),
        re.compile(r"^-{5,}$"),
        re.compile(r"rsnapshot\\s+\\w+:\\s+started", re.IGNORECASE),
    ]
    marker_res: List[re.Pattern] = [r for r in default_markers if r is not None]
    runs = _split_runs(lines, marker_res)
    if runs:
        start_offset, lines = runs[-1]
        start_line += start_offset

    error_patterns = [
        re.compile(r"rsync error", re.IGNORECASE),
        re.compile(r"\bERROR:", re.IGNORECASE),
        re.compile(r"FATAL", re.IGNORECASE),
        re.compile(r"Backup FAILED", re.IGNORECASE),
        re.compile(r"partial transfer", re.IGNORECASE),
    ] + list(pcfg.classifier_error_regexes)
    warning_patterns = [re.compile(r"WARNING", re.IGNORECASE)] + list(
        pcfg.classifier_warning_regexes
    )

    ignore_res = pcfg.classifier_ignore_regexes or []
    findings: List[Finding] = []

    prefix_lines = prefix_lines or []

    for offset, line in enumerate(lines):
        current_line = start_line + offset
        if any(r.search(line) for r in ignore_res):
            continue

        for r in error_patterns:
            for _ in r.finditer(line):
                excerpt = _build_excerpt(
                    lines,
                    offset,
                    context_prefix_lines,
                    context_suffix_lines,
                    excerpt_limit,
                    prefix_lines,
                )
                findings.append(
                    Finding(
                        file_path=file_path,
                        pipeline_name=pipeline_name,
                        finding_index=len(findings),
                        severity=Severity.ERROR,
                        message=f"rsnapshot error pattern /{r.pattern}/",
                        line_start=current_line,
                        line_end=current_line,
                        rule_id=r.pattern,
                        excerpt=excerpt,
                    )
                )

        for r in warning_patterns:
            for _ in r.finditer(line):
                excerpt = _build_excerpt(
                    lines,
                    offset,
                    context_prefix_lines,
                    context_suffix_lines,
                    excerpt_limit,
                    prefix_lines,
                )
                findings.append(
                    Finding(
                        file_path=file_path,
                        pipeline_name=pipeline_name,
                        finding_index=len(findings),
                        severity=Severity.WARNING,
                        message=f"rsnapshot warning pattern /{r.pattern}/",
                        line_start=current_line,
                        line_end=current_line,
                        rule_id=r.pattern,
                        excerpt=excerpt,
                    )
                )

    return findings
