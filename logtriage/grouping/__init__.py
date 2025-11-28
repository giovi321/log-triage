from typing import List

from .whole_file import group_whole_file
from .marker import group_by_marker
from .separator import group_by_separator
from ..models import PipelineConfig


def group_lines(pcfg: PipelineConfig, lines: List[str]) -> List[List[str]]:
    if pcfg.grouping_type == "whole_file":
        return group_whole_file(lines)
    if pcfg.grouping_type == "marker":
        return group_by_marker(lines, pcfg.grouping_start_regex, pcfg.grouping_end_regex)
    if pcfg.grouping_type == "separator":
        return group_by_separator(lines, pcfg.grouping_separator_regex, pcfg.grouping_only_last)
    return group_whole_file(lines)
