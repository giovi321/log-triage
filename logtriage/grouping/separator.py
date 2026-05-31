import re
from typing import List, Optional


def group_by_separator(
    lines: List[str],
    separator_re: Optional[re.Pattern],
    only_last: bool = False,
) -> List[List[str]]:
    """Group log lines using separator patterns for run boundaries.

    A separator regex identifies the start of each run. This is commonly
    used for logs where timestamps or specific headers mark different
    execution runs (e.g., rsnapshot, cron jobs, service restarts).

    When only_last is True, only the final run (after the last separator)
    is processed, which is useful for focusing on the most recent activity
    in historical log files.

    Args:
        lines: Log lines to group
        separator_re: Pattern that identifies run boundaries
        only_last: If True, only return the last run chunk
        
    Returns:
        List of line chunks grouped by separators, or single chunk
        if no separators are found
    """
    if not separator_re:
        return [lines] if lines else []

    # Single pass: accumulate the current run, flushing (if non-empty) each time
    # a separator line is hit. The separator line itself is not part of any run.
    # `current` always holds the run since the most recent separator, so at the
    # end it is exactly the tail after the last separator.
    chunks: List[List[str]] = []
    current: List[str] = []
    saw_separator = False

    for line in lines:
        if separator_re.search(line):
            saw_separator = True
            if current:
                chunks.append(current)
            current = []
        else:
            current.append(line)

    if not saw_separator:
        # No separators found, treat everything as one run.
        return [lines] if lines else []

    if only_last:
        # The run after the final separator (empty if the separator is last).
        return [current] if current else []

    if current:
        chunks.append(current)
    return chunks
