from typing import List

from .whole_file import group_whole_file
from .marker import group_by_marker
from ..models import PipelineConfig


def group_lines(pcfg: PipelineConfig, lines: List[str]) -> List[List[str]]:
    if pcfg.grouping_type == "whole_file":
        return group_whole_file(lines)
    if pcfg.grouping_type == "marker":
        return group_by_marker(lines, pcfg.grouping_start_regex, pcfg.grouping_end_regex)
    return group_whole_file(lines)
