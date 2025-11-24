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
class PipelineConfig:
    name: str
    match_filename_regex: re.Pattern
    classifier_type: str
    classifier_error_regexes: List[re.Pattern]
    classifier_warning_regexes: List[re.Pattern]
    classifier_ignore_regexes: List[re.Pattern]


@dataclasses.dataclass
class Finding:
    file_path: Path
    pipeline_name: str
    finding_index: int
    severity: Severity
    message: str
    line_start: int
    line_end: int
    rule_id: Optional[str]
    excerpt: List[str]
    needs_llm: bool = False
    llm_response: Optional["LLMResponse"] = None


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
    llm: "ModuleLLMConfig"
    stream_from_beginning: bool
    stream_interval: float
    exit_code_by_severity: Optional[Dict[Severity, int]] = None
    alert_mqtt: Optional[AlertMQTTConfig] = None
    alert_webhook: Optional[AlertWebhookConfig] = None
    baseline: Optional[BaselineConfig] = None
    enabled: bool = True


@dataclasses.dataclass
class LLMProviderConfig:
    name: str
    api_base: str
    api_key_env: str
    model: str
    organization: Optional[str] = None
    api_version: Optional[str] = None
    request_timeout: float = 30.0
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 512


@dataclasses.dataclass
class GlobalLLMConfig:
    enabled: bool
    min_severity: Severity
    max_excerpt_lines: int
    max_output_tokens: int
    request_timeout: float
    temperature: float
    top_p: float
    default_provider: Optional[str]
    providers: Dict[str, LLMProviderConfig]


@dataclasses.dataclass
class ModuleLLMConfig:
    enabled: bool
    min_severity: Severity
    max_excerpt_lines: int
    prompt_template_path: Optional[Path]
    provider_name: Optional[str]
    emit_llm_payloads_dir: Optional[Path]
    llm_payload_mode: str = "full"  # "full" or "errors_only"
    only_last_chunk: bool = False   # legacy; kept for config compatibility in findings mode
    max_output_tokens: Optional[int] = None


@dataclasses.dataclass
class LLMResponse:
    provider: str
    model: str
    content: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
