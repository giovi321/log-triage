import dataclasses
import datetime
import enum
import re
from pathlib import Path
from typing import Dict, List, Optional, Any


class Severity(enum.IntEnum):
    """Severity levels for log findings.
    
    Higher values indicate more severe issues.
    Used for filtering and alerting decisions.
    """
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
    """Configuration for a log processing pipeline.
    
    Defines how logs are grouped, classified, and filtered.
    Each pipeline can be matched against files by regex pattern.
    """
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
    """Represents a classified log issue.
    
    Contains the location, severity, and context of a problem
    detected in log files by the pipeline classifiers.
    """
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
    """Configuration for a runtime module.
    
    Connects pipelines to specific file paths and defines
    execution mode (batch/follow) and output options.
    """
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
    rag: Optional["RAGModuleConfig"] = None


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
    """Global LLM provider configuration.
    
    Defines available providers and default settings
    for LLM payload generation across all modules.
    """
    enabled: bool
    default_provider: Optional[str]
    providers: Dict[str, LLMProviderConfig]
    context_prefix_lines: int = 0
    context_suffix_lines: int = 0
    summary_prompt_path: Optional[Path] = None


@dataclasses.dataclass
class ModuleLLMConfig:
    """Module-specific LLM configuration.
    
    Overrides global settings for individual modules,
    including prompt templates and output directories.
    """
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
    """Response from an LLM provider.
    
    Contains the generated content and usage metadata
    for a completed LLM request.
    """
    provider: str
    model: str
    content: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    citations: Optional[List[str]] = None


@dataclasses.dataclass
class RAGGlobalConfig:
    """Global RAG configuration."""
    enabled: bool
    cache_dir: Path
    vector_store_dir: Path
    embedding_model: str
    device: str
    batch_size: int
    top_k: int
    similarity_threshold: float
    max_chunks: int
    # Hidden advanced settings with sensible defaults
    embedding_batch_size: int = 32  # Keep for backward compatibility
    vector_store_type: str = "chroma"


@dataclasses.dataclass
class KnowledgeSourceConfig:
    """Configuration for a single knowledge source (git repository)."""
    repo_url: str
    branch: str = "main"
    include_paths: List[str] = None
    
    def __post_init__(self):
        if self.include_paths is None:
            self.include_paths = ["**/*.md", "**/*.rst", "**/*.txt"]


@dataclasses.dataclass
class RAGModuleConfig:
    """Module-specific RAG configuration."""
    enabled: bool
    knowledge_sources: List[KnowledgeSourceConfig]


@dataclasses.dataclass
class DocumentChunk:
    """A chunk of indexed documentation."""
    chunk_id: str
    repo_id: str
    file_path: str
    heading: str
    content: str
    commit_hash: str
    metadata: Dict[str, Any]


@dataclasses.dataclass
class RetrievalResult:
    """Result from RAG retrieval."""
    chunks: List[DocumentChunk]
    query: str
    total_retrieved: int
