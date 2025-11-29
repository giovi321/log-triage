import re
from typing import List, Optional


def group_by_marker(
    lines: List[str],
    start_re: Optional[re.Pattern],
    end_re: Optional[re.Pattern],
) -> List[List[str]]:
    """Group log lines using start and end markers.

    Creates chunks based on marker patterns:
    - New chunk starts when start_re matches a line
    - Chunk ends when end_re matches (if provided) or when next start_re appears
    - If no start_re is provided, everything is treated as a single chunk
    
    Useful for logs that have clear delimiters between logical sections
    like backup runs, service restarts, or transaction blocks.

    Args:
        lines: Log lines to group
        start_re: Pattern that marks the beginning of a new chunk
        end_re: Optional pattern that marks the end of a chunk
        
    Returns:
        List of line chunks grouped by markers
    """
    if not start_re:
        return [lines] if lines else []

    chunks: List[List[str]] = []
    current: List[str] = []

    for line in lines:
        if start_re.search(line):
            if current:
                chunks.append(current)
            current = [line]
        else:
            current.append(line)
            if end_re and end_re.search(line):
                chunks.append(current)
                current = []

    if current:
        chunks.append(current)

    return chunks
