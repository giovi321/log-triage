import os
from pathlib import Path
from typing import List

from .models import PipelineConfig


def iter_log_files(path: Path) -> List[Path]:
    """Iterate over log files in a directory or return single file.
    
    If path is a file, returns list containing that file.
    If path is a directory, recursively walks the directory
    and returns sorted list of all files found.
    
    Args:
        path: File or directory path to iterate
        
    Returns:
        Sorted list of file paths
    """
    if path.is_file():
        return [path]
    files: List[Path] = []
    for root, dirs, fnames in os.walk(path):
        for name in fnames:
            full = Path(root) / name
            files.append(full)
    return sorted(files)


def select_pipeline(pipelines: List[PipelineConfig], file_path: Path) -> PipelineConfig:
    """Select the appropriate pipeline for a file based on filename regex.
    
    Matches the file path against each pipeline's filename_regex pattern.
    Returns the first matching pipeline, or the first pipeline in the
    list if none match (fallback behavior).
    
    Args:
        pipelines: List of available pipelines
        file_path: Path to the file being processed
        
    Returns:
        Selected pipeline configuration
    """
    fname = str(file_path)
    for p in pipelines:
        if p.match_filename_regex and p.match_filename_regex.search(fname):
            return p
    return pipelines[0]
