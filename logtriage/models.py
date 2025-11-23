import enum
import dataclasses
import re
from pathlib import Path
from typing import List, Optional, Dict


class Severity(enum.IntEnum):
    UNKNOWN = 0
    OK = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5

    @classmethod
    def from_string(cls, value: str) -> "Severity":
        value = value.upper()
        if value not in cls.__members__:
            raise ValueError(f"Unknown severity: {value}")
        return cls[value]


@dataclasses.dataclass
class PipelineLLMConfig:
    enabled: bool
    min_severity: Severity
    max_chunk_lines: int
    prompt_template_path: Optional[Path] = None


@dataclasses.dataclass
class PipelineConfig:
    name: str
    match_filename_regex: re.Pattern
    grouping_type: str
    grouping_start_regex: Optional[re.Pattern]
    grouping_end_regex: Optional[re.Pattern]
    classifier_type: str
    classifier_error_regexes: List[re.Pattern]
    classifier_warning_regexes: List[re.Pattern]
    classifier_ignore_regexes: List[re.Pattern]
    llm_cfg: PipelineLLMConfig


@dataclasses.dataclass
class LogChunk:
    file_path: Path
    pipeline_name: str
    chunk_index: int
    lines: List[str]
    severity: Severity
    reason: str
    error_count: int
    warning_count: int
    needs_llm: bool


@dataclasses.dataclass
class AlertMQTTConfig:
    enabled: bool
    host: str
    port: int
    topic: str
    username: Optional[str]
    password: Optional[str]
    min_severity: Severity


@dataclasses.dataclass
class AlertWebhookConfig:
    enabled: bool
    url: str
    method: str
    min_severity: Severity
    headers: Dict[str, str]


@dataclasses.dataclass
class BaselineConfig:
    enabled: bool
    state_file: Path
    window: int
    error_multiplier: float
    warning_multiplier: float
    severity_on_anomaly: Severity


@dataclasses.dataclass
class ModuleConfig:
    name: str
    path: Path
    mode: str  # "batch" or "follow"
    pipeline_name: Optional[str]
    output_format: str  # "text" or "json"
    min_print_severity: Severity
    emit_llm_payloads_dir: Optional[Path]
    stream_from_beginning: bool
    stream_interval: float
    llm_payload_mode: str = "full"  # "full" or "errors_only"
    only_last_chunk: bool = False   # if true, only last chunk per file is considered
    exit_code_by_severity: Optional[Dict[Severity, int]] = None
    alert_mqtt: Optional[AlertMQTTConfig] = None
    alert_webhook: Optional[AlertWebhookConfig] = None
    baseline: Optional[BaselineConfig] = None
    enabled: bool = True
