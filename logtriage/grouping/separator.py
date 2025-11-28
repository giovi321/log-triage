import re
from typing import List, Optional


def group_by_separator(
    lines: List[str],
    separator_re: Optional[re.Pattern],
    only_last: bool = False,
) -> List[List[str]]:
    """Separator-based grouping for run boundaries.

    A separator regex identifies the start of each run. When only_last is True,
    only the last run (after the final separator) is processed.
    """
    if not separator_re:
        return [lines] if lines else []

    # Find all separator line indices
    separator_indices = []
    for i, line in enumerate(lines):
        if separator_re.search(line):
            separator_indices.append(i)

    if not separator_indices:
        # No separators found, treat everything as one run
        return [lines] if lines else []

    if only_last:
        # Only process the last run (after the final separator)
        last_separator_idx = separator_indices[-1]
        if last_separator_idx + 1 < len(lines):
            return [lines[last_separator_idx + 1:]]
        else:
            # Separator is the last line, return empty
            return []

    # Process all runs
    chunks: List[List[str]] = []
    
    # First run (before first separator)
    if separator_indices[0] > 0:
        chunks.append(lines[:separator_indices[0]])
    
    # Middle runs (between separators)
    for i in range(len(separator_indices) - 1):
        start_idx = separator_indices[i] + 1
        end_idx = separator_indices[i + 1]
        if start_idx < end_idx:
            chunks.append(lines[start_idx:end_idx])
    
    # Last run (after last separator)
    last_separator_idx = separator_indices[-1]
    if last_separator_idx + 1 < len(lines):
        chunks.append(lines[last_separator_idx + 1:])

    return chunks
