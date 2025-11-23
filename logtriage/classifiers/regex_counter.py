import re
from typing import List, Tuple

from ..models import PipelineConfig, Severity


def count_matches(regexes: List[re.Pattern], text: str) -> int:
    total = 0
    for r in regexes:
        if not r:
            continue
        total += len(r.findall(text))
    return total


def classify_regex_counter(
    pcfg: PipelineConfig,
    chunk_lines: List[str],
) -> Tuple[Severity, str, int, int]:
    """Generic regex-based classifier with optional ignore rules.

    - Lines matching any `classifier_ignore_regexes` are removed before counting.
    - Error regexes are evaluated first, then warning regexes.
    """
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

    if error_count > 0:
        severity = Severity.ERROR
        reason = f"{error_count} error match(es) found"
    elif warning_count > 0:
        severity = Severity.WARNING
        reason = f"{warning_count} warning match(es) found"
    elif joined.strip():
        severity = Severity.OK
        reason = "no error/warning matches"
    else:
        severity = Severity.UNKNOWN
        reason = "empty chunk or only ignored lines"

    return severity, reason, error_count, warning_count
