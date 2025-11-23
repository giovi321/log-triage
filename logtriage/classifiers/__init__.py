from typing import List, Tuple

from ..models import PipelineConfig, Severity
from .regex_counter import classify_regex_counter
from .rsnapshot_basic import classify_rsnapshot_basic


def classify_chunk(pcfg: PipelineConfig, chunk_lines: List[str]) -> Tuple[Severity, str, int, int]:
    if pcfg.classifier_type == "rsnapshot_basic":
        return classify_rsnapshot_basic(pcfg, chunk_lines)
    return classify_regex_counter(pcfg, chunk_lines)
