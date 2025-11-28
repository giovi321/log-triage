import dataclasses
import datetime
import enum
import re
from pathlib import Path
from typing import Dict, List, Optional


class Severity(enum.IntEnum):
    WARNING = 1
    ERROR = 2
    CRITICAL = 3

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
    grouping_type: str
    grouping_start_regex: Optional[re.Pattern]
    grouping_end_regex: Optional[re.Pattern]
    grouping_separator_regex: Optional[re.Pattern]
    grouping_only_last: bool


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
    created_at: Optional[datetime.datetime] = None


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
    stale_after_minutes: Optional[int] = None
    alert_mqtt: Optional[AlertMQTTConfig] = None
    alert_webhook: Optional[AlertWebhookConfig] = None
    enabled: bool = True


@dataclasses.dataclass
class LLMProviderConfig:
    name: str
    api_base: str
    api_key_env: Optional[str]
    model: str
    organization: Optional[str] = None
    api_version: Optional[str] = None
    max_excerpt_lines: int = 20
    request_timeout: float = 30.0
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 512


@dataclasses.dataclass
class GlobalLLMConfig:
    enabled: bool
    min_severity: Severity
    default_provider: Optional[str]
    providers: Dict[str, LLMProviderConfig]
    context_prefix_lines: int = 0
    context_suffix_lines: int = 0
    summary_prompt_path: Optional[Path] = None


@dataclasses.dataclass
class ModuleLLMConfig:
    enabled: bool
    min_severity: Severity
    max_excerpt_lines: int
    provider_name: Optional[str] = None
    prompt_template_path: Optional[Path] = None
    emit_llm_payloads_dir: Optional[Path] = None
    context_prefix_lines: int = 0
    context_suffix_lines: int = 0
    max_output_tokens: Optional[int] = None


@dataclasses.dataclass
class LLMResponse:
    provider: str
    model: str
    content: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
