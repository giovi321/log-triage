from typing import List

from ..models import PipelineConfig, Finding
from .regex_counter import classify_regex_counter
from .rsnapshot_basic import classify_rsnapshot_basic


def classify_lines(
    pcfg: PipelineConfig,
    file_path,
    pipeline_name: str,
    lines: List[str],
    start_line: int = 1,
    excerpt_limit: int = 20,
    context_prefix_lines: int = 0,
    context_suffix_lines: int = 0,
    prefix_lines: List[str] | None = None,
) -> List[Finding]:
    if pcfg.classifier_type == "rsnapshot_basic":
        return classify_rsnapshot_basic(
            pcfg,
            file_path,
            pipeline_name,
            lines,
            start_line,
            excerpt_limit,
            context_prefix_lines,
            context_suffix_lines,
            prefix_lines,
        )
    return classify_regex_counter(
        pcfg,
        file_path,
        pipeline_name,
        lines,
        start_line,
        excerpt_limit,
        context_prefix_lines,
        context_suffix_lines,
        prefix_lines,
    )
