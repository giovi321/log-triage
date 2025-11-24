import re
from typing import List

from ..models import PipelineConfig, Severity, Finding


def classify_rsnapshot_basic(
    pcfg: PipelineConfig,
    file_path,
    pipeline_name: str,
    lines: List[str],
    start_line: int = 1,
    excerpt_limit: int = 20,
    context_prefix_lines: int = 0,
) -> List[Finding]:
    """Heuristic classifier for rsnapshot runs that emits per-line findings."""
    joined_all = "\n".join(lines)

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

    for offset, line in enumerate(lines):
        current_line = start_line + offset
        if any(r.search(line) for r in ignore_res):
            continue

        for r in error_patterns:
            for _ in r.finditer(line):
                excerpt_start = max(0, offset - context_prefix_lines)
                excerpt = lines[excerpt_start : offset + 1]
                if len(excerpt) > excerpt_limit:
                    excerpt = excerpt[-excerpt_limit:]
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
                excerpt_start = max(0, offset - context_prefix_lines)
                excerpt = lines[excerpt_start : offset + 1]
                if len(excerpt) > excerpt_limit:
                    excerpt = excerpt[-excerpt_limit:]
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

    m = re.search(r"exit\s+code\s*=\s*(\d+)", joined_all)
    if m:
        exit_code = int(m.group(1))
        severity = Severity.ERROR if exit_code != 0 else Severity.OK
        findings.append(
            Finding(
                file_path=file_path,
                pipeline_name=pipeline_name,
                finding_index=len(findings),
                severity=severity,
                message=f"rsnapshot exit code {exit_code}",
                line_start=start_line,
                line_end=start_line + len(lines) - 1,
                rule_id="exit_code",
                excerpt=lines[:excerpt_limit],
            )
        )
    elif not findings and joined_all.strip():
        findings.append(
            Finding(
                file_path=file_path,
                pipeline_name=pipeline_name,
                finding_index=0,
                severity=Severity.OK,
                message="rsnapshot run looks normal",
                line_start=start_line,
                line_end=start_line + len(lines) - 1,
                rule_id=None,
                excerpt=lines[:excerpt_limit],
            )
        )

    return findings
