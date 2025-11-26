import os
from pathlib import Path
from typing import List

from .models import PipelineConfig


def iter_log_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files: List[Path] = []
    for root, dirs, fnames in os.walk(path):
        for name in fnames:
            full = Path(root) / name
            files.append(full)
    return sorted(files)


def select_pipeline(pipelines: List[PipelineConfig], file_path: Path) -> PipelineConfig:
    fname = str(file_path)
    for p in pipelines:
        if p.match_filename_regex and p.match_filename_regex.search(fname):
            return p
    return pipelines[0]
