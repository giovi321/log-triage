import re
from typing import List

from ..models import PipelineConfig, Severity, Finding


def classify_regex_counter(
    pcfg: PipelineConfig,
    file_path,
    pipeline_name: str,
    lines: List[str],
    start_line: int = 1,
    excerpt_limit: int = 20,
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
            for _ in r.finditer(line):
                findings.append(
                    Finding(
                        file_path=file_path,
                        pipeline_name=pipeline_name,
                        finding_index=len(findings),
                        severity=Severity.ERROR,
                        message=f"Matched error pattern /{r.pattern}/",
                        line_start=current_line,
                        line_end=current_line,
                        rule_id=r.pattern,
                        excerpt=[line],
                    )
                )

        for r in pcfg.classifier_warning_regexes:
            if not r:
                continue
            for _ in r.finditer(line):
                findings.append(
                    Finding(
                        file_path=file_path,
                        pipeline_name=pipeline_name,
                        finding_index=len(findings),
                        severity=Severity.WARNING,
                        message=f"Matched warning pattern /{r.pattern}/",
                        line_start=current_line,
                        line_end=current_line,
                        rule_id=r.pattern,
                        excerpt=[line],
                    )
                )

    if not findings and any(ln.strip() for ln in lines):
        findings.append(
            Finding(
                file_path=file_path,
                pipeline_name=pipeline_name,
                finding_index=0,
                severity=Severity.OK,
                message="No error/warning matches",
                line_start=start_line,
                line_end=start_line + len(lines) - 1,
                rule_id=None,
                excerpt=lines[:excerpt_limit],
            )
        )

    return findings
