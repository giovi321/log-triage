import re
from typing import List, Tuple

from ..models import PipelineConfig, Severity
from .regex_counter import count_matches


def classify_rsnapshot_basic(
    pcfg: PipelineConfig,
    chunk_lines: List[str],
) -> Tuple[Severity, str, int, int]:
    """Heuristic classifier for rsnapshot runs.

    - Applies ignore_regexes for counting error/warning-like patterns.
    - Still scans the full chunk for explicit exit codes and rsnapshot messages.
    """
    joined_all = "\n".join(chunk_lines)

    extra_error_patterns = [
        re.compile(r"rsync error", re.IGNORECASE),
        re.compile(r"\bERROR:", re.IGNORECASE),
        re.compile(r"FATAL", re.IGNORECASE),
        re.compile(r"Backup FAILED", re.IGNORECASE),
        re.compile(r"partial transfer", re.IGNORECASE),
    ]
    extra_warning_patterns = [
        re.compile(r"WARNING", re.IGNORECASE),
    ]

    # Apply ignore filters for counting only
    ignore_res = pcfg.classifier_ignore_regexes or []
    if ignore_res:
        filtered_lines: List[str] = []
        for line in chunk_lines:
            if any(r.search(line) for r in ignore_res):
                continue
            filtered_lines.append(line)
    else:
        filtered_lines = list(chunk_lines)

    joined = "\n".join(filtered_lines)

    error_count = count_matches(pcfg.classifier_error_regexes, joined)
    warning_count = count_matches(pcfg.classifier_warning_regexes, joined)

    error_count += count_matches(extra_error_patterns, joined)
    warning_count += count_matches(extra_warning_patterns, joined)

    # Try to extract exit code from the full, unfiltered chunk.
    exit_code = None
    m = re.search(r"exit\s+code\s*=\s*(\d+)", joined_all)
    if m:
        exit_code = int(m.group(1))

    if exit_code is not None and exit_code != 0:
        severity = Severity.ERROR
        reason = f"rsnapshot exit code {exit_code}"
    elif "completed successfully" in joined_all:
        # Completed successfully but maybe with warnings
        if error_count > 0:
            severity = Severity.ERROR
            reason = f"{error_count} error-like pattern(s) in rsnapshot chunk"
        elif warning_count > 0:
            severity = Severity.WARNING
            reason = f"{warning_count} warning-like pattern(s) in rsnapshot chunk"
        else:
            severity = Severity.OK
            reason = "rsnapshot completed successfully"
    elif error_count > 0:
        severity = Severity.ERROR
        reason = f"{error_count} error-like pattern(s) in rsnapshot chunk"
    elif warning_count > 0:
        severity = Severity.WARNING
        reason = f"{warning_count} warning-like pattern(s) in rsnapshot chunk"
    elif "".join(chunk_lines).strip():
        severity = Severity.OK
        reason = "rsnapshot chunk looks normal"
    else:
        severity = Severity.UNKNOWN
        reason = "empty rsnapshot chunk or only ignored lines"

    return severity, reason, error_count, warning_count
