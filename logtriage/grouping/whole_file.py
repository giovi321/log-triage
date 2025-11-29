from typing import List


def group_whole_file(lines: List[str]) -> List[List[str]]:
    """Group entire file as a single chunk.
    
    Simplest grouping strategy that treats the entire log file
    as one logical unit. Useful for logs that represent single
    runs or when no natural grouping boundaries exist.
    
    Args:
        lines: All lines in the log file
        
    Returns:
        List containing one chunk with all lines, or empty list if no lines
    """
    return [lines] if lines else []
