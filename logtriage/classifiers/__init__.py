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
) -> List[Finding]:
    if pcfg.classifier_type == "rsnapshot_basic":
        return classify_rsnapshot_basic(pcfg, file_path, pipeline_name, lines, start_line)
    return classify_regex_counter(pcfg, file_path, pipeline_name, lines, start_line)
