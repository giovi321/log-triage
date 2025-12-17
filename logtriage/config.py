import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

try:
    import yaml  # PyYAML
except ImportError:
    yaml = None

from .models import (
    AlertMQTTConfig,
    AlertWebhookConfig,
    GlobalLLMConfig,
    LLMProviderConfig,
    ModuleConfig,
    ModuleLLMConfig,
    PipelineConfig,
    Severity,
    RAGGlobalConfig,
    RAGModuleConfig,
    KnowledgeSourceConfig,
)


def load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        print("PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return data


def _normalize_regex_pattern(pattern: str) -> str:
    """Decode common escape sequences so YAML-friendly patterns behave as intended."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            normalized = pattern.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        # Fall back to the original pattern if decoding fails so we don't block config loading
        return pattern

    # If YAML or the decoder converted word-boundary escapes (\b) into literal backspace
    # characters, restore them so regex word boundaries work as expected.
    return normalized.replace("\b", "\\b")


def _compile_regex(pattern: Optional[str], flags: int = 0) -> Optional[re.Pattern]:
    if not pattern:
        return None
    normalized = _normalize_regex_pattern(pattern)
    return re.compile(normalized, flags)


def build_pipelines(cfg: Dict[str, Any]) -> List[PipelineConfig]:
    pipelines_cfg = cfg.get("pipelines", []) or []
    pipelines: List[PipelineConfig] = []

    for item in pipelines_cfg:
        name = item["name"]

        match_cfg = item.get("match", {}) or {}
        filename_regex = match_cfg.get("filename_regex", ".*")
        try:
            match_filename_regex = re.compile(filename_regex)
        except re.error as exc:  # pragma: no cover - simple input validation
            raise ValueError(
                f"Pipeline '{name}' has invalid filename_regex '{filename_regex}': {exc}"
            ) from exc

        grouping_cfg = item.get("grouping", {}) or {}
        grouping_type = "whole_file"
        grouping_start_re = None
        grouping_end_re = None
        grouping_only_last = False
        grouping_separator_re = None
        if isinstance(grouping_cfg, str):
            grouping_type = grouping_cfg.lower()
        elif isinstance(grouping_cfg, dict):
            grouping_type = str(grouping_cfg.get("type", "whole_file")).lower()
            grouping_start_re = _compile_regex(grouping_cfg.get("start_regex"))
            grouping_end_re = _compile_regex(grouping_cfg.get("end_regex"))
            grouping_separator_re = _compile_regex(grouping_cfg.get("separator_regex"))
            grouping_only_last = bool(grouping_cfg.get("only_last", False))
        else:
            raise ValueError(
                f"Pipeline '{name}': grouping must be string or mapping (got {type(grouping_cfg)})"
            )

        if grouping_type in ("marker_based", "marker"):
            grouping_type = "marker"
        elif grouping_type in ("separator_based", "separator"):
            grouping_type = "separator"
        elif grouping_type != "whole_file":
            raise ValueError(
                f"Pipeline '{name}': unknown grouping type '{grouping_type}' (expected whole_file, marker, or separator)"
            )

        classifier_cfg = item.get("classifier", {}) or {}
        classifier_type = classifier_cfg.get("type", "regex_counter")

        def _compile_list(key: str) -> List[re.Pattern]:
            patterns = classifier_cfg.get(key, []) or []
            compiled: List[re.Pattern] = []
            for pat in patterns:
                try:
                    compiled_regex = _compile_regex(pat, re.IGNORECASE)
                    if compiled_regex:
                        compiled.append(compiled_regex)
                except re.error as exc:  # pragma: no cover - simple input validation
                    raise ValueError(
                        f"Pipeline '{name}' has invalid {key.rstrip('es')} regex '{pat}': {exc}"
                    ) from exc
            return compiled

        error_regexes = _compile_list("error_regexes")
        warning_regexes = _compile_list("warning_regexes")
        ignore_regexes = _compile_list("ignore_regexes")

        pipelines.append(
            PipelineConfig(
                name=name,
                match_filename_regex=match_filename_regex,
                classifier_type=classifier_type,
                classifier_error_regexes=error_regexes,
                classifier_warning_regexes=warning_regexes,
                classifier_ignore_regexes=ignore_regexes,
                grouping_type=grouping_type,
                grouping_start_regex=grouping_start_re,
                grouping_end_regex=grouping_end_re,
                grouping_separator_regex=grouping_separator_re,
                grouping_only_last=grouping_only_last,
            )
        )

    if not pipelines:
        raise ValueError("No pipelines defined in configuration")

    return pipelines


def build_llm_config(cfg: Dict[str, Any]) -> GlobalLLMConfig:
    llm_cfg = cfg.get("llm", {}) or {}
    defaults = cfg.get("defaults", {}) or {}

    enabled = bool(llm_cfg.get("enabled", defaults.get("llm_enabled", False)))
    min_severity = Severity.from_string(
        llm_cfg.get("min_severity", defaults.get("llm_min_severity", "WARNING"))
    )
    default_provider = llm_cfg.get("default_provider")

    base_max_excerpt_lines = int(
        llm_cfg.get(
            "max_excerpt_lines",
            llm_cfg.get("max_chunk_lines", defaults.get("max_excerpt_lines", 20)),
        )
    )
    base_context_prefix_lines = int(llm_cfg.get("context_prefix_lines", 0))
    base_context_suffix_lines = int(llm_cfg.get("context_suffix_lines", 0))
    base_max_output_tokens = int(
        llm_cfg.get("max_output_tokens", llm_cfg.get("max_tokens", 512))
    )
    base_request_timeout = float(llm_cfg.get("request_timeout", 30.0))
    base_temperature = float(llm_cfg.get("temperature", 0.0))
    base_top_p = float(llm_cfg.get("top_p", 1.0))
    summary_prompt_raw = llm_cfg.get("summary_prompt_path")
    summary_prompt_path = Path(summary_prompt_raw) if summary_prompt_raw else None

    providers: Dict[str, LLMProviderConfig] = {}
    providers_raw = llm_cfg.get("providers", {}) or {}
    for name, pdata in providers_raw.items():
        api_base = str(pdata.get("api_base") or pdata.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        model = pdata.get("model")
        if not model:
            raise ValueError(f"LLM provider {name} must define a model")
        api_key_env_raw = pdata.get("api_key_env")
        api_key_env = str(api_key_env_raw) if api_key_env_raw else None
        providers[name] = LLMProviderConfig(
            name=name,
            api_base=api_base,
            api_key_env=api_key_env,
            model=str(model),
            organization=pdata.get("organization"),
            api_version=pdata.get("api_version"),
            max_excerpt_lines=int(pdata.get("max_excerpt_lines", base_max_excerpt_lines)),
            request_timeout=float(pdata.get("request_timeout", base_request_timeout)),
            temperature=float(pdata.get("temperature", base_temperature)),
            top_p=float(pdata.get("top_p", base_top_p)),
            max_output_tokens=int(pdata.get("max_output_tokens", base_max_output_tokens)),
        )

    if default_provider and default_provider not in providers:
        raise ValueError(
            f"Default LLM provider '{default_provider}' is not defined under llm.providers"
        )

    return GlobalLLMConfig(
        enabled=enabled,
        min_severity=min_severity,
        default_provider=default_provider,
        providers=providers,
        context_prefix_lines=base_context_prefix_lines,
        context_suffix_lines=base_context_suffix_lines,
        summary_prompt_path=summary_prompt_path,
    )


def build_modules(cfg: Dict[str, Any], llm_defaults: GlobalLLMConfig) -> List[ModuleConfig]:
    modules_cfg = cfg.get("modules", []) or []
    modules: List[ModuleConfig] = []

    base_context_prefix_lines = llm_defaults.context_prefix_lines
    base_context_suffix_lines = llm_defaults.context_suffix_lines

    for item in modules_cfg:
        name = item["name"]
        path_str = item["path"]
        mode = item.get("mode", "batch").lower()
        if mode not in ("batch", "follow"):
            raise ValueError(f"Module {name}: invalid mode {mode} (expected 'batch' or 'follow')")

        pipeline_name = item.get("pipeline")
        output_format = item.get("output_format", "text")
        if output_format not in ("text", "json"):
            raise ValueError(f"Module {name}: invalid output_format {output_format}")

        min_sev = Severity.from_string(item.get("min_print_severity", "WARNING"))

        stale_after_raw = item.get("stale_after_minutes")
        stale_after_minutes = None
        if stale_after_raw is not None:
            try:
                stale_after_minutes = int(stale_after_raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Module {name}: invalid stale_after_minutes {stale_after_raw} (expected integer)"
                )
            if stale_after_minutes <= 0:
                raise ValueError(
                    f"Module {name}: stale_after_minutes must be positive (got {stale_after_minutes})"
                )

        llm_cfg = item.get("llm", {}) or {}
        llm_enabled = bool(llm_cfg.get("enabled", llm_defaults.enabled))
        llm_min_sev = Severity.from_string(
            llm_cfg.get("min_severity", llm_defaults.min_severity.name)
        )
        provider_name = llm_cfg.get("provider")
        provider = None
        if provider_name:
            provider = llm_defaults.providers.get(provider_name)
        elif llm_defaults.default_provider:
            provider = llm_defaults.providers.get(llm_defaults.default_provider)

        llm_max_excerpt = llm_cfg.get(
            "max_excerpt_lines", llm_cfg.get("max_chunk_lines")
        )
        if llm_max_excerpt is None:
            if provider is not None:
                llm_max_excerpt = provider.max_excerpt_lines
            else:
                llm_max_excerpt = 20
        llm_max_excerpt = int(llm_max_excerpt)
        prompt_template_raw = llm_cfg.get("prompt_template")
        prompt_template_path = Path(prompt_template_raw) if prompt_template_raw else None
        emit_dir_raw = llm_cfg.get("emit_llm_payloads_dir", item.get("emit_llm_payloads_dir"))
        emit_dir = Path(emit_dir_raw) if emit_dir_raw else None

        context_prefix_lines_raw = llm_cfg.get("context_prefix_lines", base_context_prefix_lines)
        try:
            context_prefix_lines = int(context_prefix_lines_raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"Module {name}: invalid context_prefix_lines {context_prefix_lines_raw} (expected integer)"
            )
        if context_prefix_lines < 0:
            raise ValueError(
                f"Module {name}: context_prefix_lines must be non-negative (got {context_prefix_lines})"
            )

        context_suffix_lines_raw = llm_cfg.get("context_suffix_lines", base_context_suffix_lines)
        try:
            context_suffix_lines = int(context_suffix_lines_raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"Module {name}: invalid context_suffix_lines {context_suffix_lines_raw} (expected integer)"
            )
        if context_suffix_lines < 0:
            raise ValueError(
                f"Module {name}: context_suffix_lines must be non-negative (got {context_suffix_lines})"
            )

        max_output_tokens_raw = llm_cfg.get("max_output_tokens", llm_cfg.get("max_tokens"))
        max_output_tokens = int(max_output_tokens_raw) if max_output_tokens_raw else None

        module_llm_cfg = ModuleLLMConfig(
            enabled=llm_enabled,
            min_severity=llm_min_sev,
            max_excerpt_lines=llm_max_excerpt,
            context_prefix_lines=context_prefix_lines,
            context_suffix_lines=context_suffix_lines,
            prompt_template_path=prompt_template_path,
            provider_name=provider_name,
            emit_llm_payloads_dir=emit_dir,
            max_output_tokens=max_output_tokens,
        )

        stream_cfg = item.get("stream", {}) or {}
        stream_from_beginning = bool(stream_cfg.get("from_beginning", False))
        stream_interval = float(stream_cfg.get("interval", 1.0))

        alerts_cfg = item.get("alerts", {}) or {}

        alert_mqtt_cfg = None
        mqtt_raw = alerts_cfg.get("mqtt")
        if mqtt_raw:
            mqtt_enabled = bool(mqtt_raw.get("enabled", True))
            mqtt_host = mqtt_raw.get("host")
            mqtt_topic = mqtt_raw.get("topic")
            if mqtt_host and mqtt_topic and mqtt_enabled:
                mqtt_port = int(mqtt_raw.get("port", 1883))
                mqtt_username = mqtt_raw.get("username")
                mqtt_password = mqtt_raw.get("password")
                mqtt_min_sev = Severity.from_string(mqtt_raw.get("min_severity", "ERROR"))
                alert_mqtt_cfg = AlertMQTTConfig(
                    enabled=mqtt_enabled,
                    host=mqtt_host,
                    port=mqtt_port,
                    topic=mqtt_topic,
                    username=mqtt_username,
                    password=mqtt_password,
                    min_severity=mqtt_min_sev,
                )

        alert_webhook_cfg = None
        webhook_raw = alerts_cfg.get("webhook")
        if webhook_raw:
            wh_enabled = bool(webhook_raw.get("enabled", True))
            wh_url = webhook_raw.get("url")
            if wh_url and wh_enabled:
                wh_method = str(webhook_raw.get("method", "POST")).upper()
                wh_min_sev = Severity.from_string(webhook_raw.get("min_severity", "ERROR"))
                wh_headers = webhook_raw.get("headers", {}) or {}
                alert_webhook_cfg = AlertWebhookConfig(
                    enabled=wh_enabled,
                    url=wh_url,
                    method=wh_method,
                    min_severity=wh_min_sev,
                    headers={str(k): str(v) for k, v in wh_headers.items()},
                )

        enabled = bool(item.get("enabled", True))

        # Build RAG configuration
        module_rag_cfg = build_module_rag_config(item)

        modules.append(
            ModuleConfig(
                name=name,
                path=Path(path_str),
                mode=mode,
                pipeline_name=pipeline_name,
                output_format=output_format,
                min_print_severity=min_sev,
                llm=module_llm_cfg,
                stream_from_beginning=stream_from_beginning,
                stream_interval=stream_interval,
                stale_after_minutes=stale_after_minutes,
                alert_mqtt=alert_mqtt_cfg,
                alert_webhook=alert_webhook_cfg,
                enabled=enabled,
                rag=module_rag_cfg,
            )
        )

    return modules


def build_rag_config(cfg: Dict[str, Any]) -> Optional[RAGGlobalConfig]:
    """Build global RAG configuration."""
    rag_cfg = cfg.get("rag", {})
    if not rag_cfg or not rag_cfg.get("enabled", False):
        return None
    
    memory_cfg = rag_cfg.get("memory", {})
    
    return RAGGlobalConfig(
        enabled=True,
        cache_dir=Path(rag_cfg.get("cache_dir", "./rag_cache")),
        embedding_model=rag_cfg.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_device=rag_cfg.get("embedding", {}).get("device", "cpu"),
        embedding_batch_size=int(rag_cfg.get("embedding", {}).get("batch_size", 32)),
        vector_store_type=rag_cfg.get("vector_store", {}).get("type", "chroma"),
        vector_store_dir=Path(rag_cfg.get("vector_store", {}).get("persist_directory", "./rag_db")),
        retrieval_top_k=int(rag_cfg.get("retrieval", {}).get("top_k", 5)),
        similarity_threshold=float(rag_cfg.get("retrieval", {}).get("similarity_threshold", 0.7)),
        max_chunks=int(rag_cfg.get("retrieval", {}).get("max_chunks", 10)),
        # Memory management settings
        max_memory_gb=float(memory_cfg.get("max_memory_gb", 3.0)),
        warning_memory_gb=float(memory_cfg.get("warning_memory_gb", 2.0)),
        embedding_max_memory_gb=float(memory_cfg.get("embedding_max_memory_gb", 2.5)),
        max_files_per_repo=int(memory_cfg.get("max_files_per_repo", 5)),
        max_chunks_per_file=int(memory_cfg.get("max_chunks_per_file", 3)),
        max_texts_per_batch=int(memory_cfg.get("max_texts_per_batch", 10)),
    )


def build_module_rag_config(module_cfg: Dict[str, Any]) -> Optional[RAGModuleConfig]:
    """Build module-specific RAG configuration."""
    rag_cfg = module_cfg.get("rag", {})
    if not rag_cfg or not rag_cfg.get("enabled", False):
        return None
    
    knowledge_sources = []
    for source_cfg in rag_cfg.get("knowledge_sources", []):
        knowledge_sources.append(KnowledgeSourceConfig(
            repo_url=source_cfg["repo_url"],
            branch=source_cfg.get("branch", "main"),
            include_paths=source_cfg.get("include_paths", [])
        ))
    
    return RAGModuleConfig(
        enabled=True,
        knowledge_sources=knowledge_sources
    )
