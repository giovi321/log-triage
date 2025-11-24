import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

try:
    import yaml  # PyYAML
except ImportError:
    yaml = None

from .models import (
    Severity,
    PipelineConfig,
    PipelineLLMConfig,
    ModuleConfig,
    AlertMQTTConfig,
    AlertWebhookConfig,
    BaselineConfig,
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


def _compile_regex(pattern: Optional[str]) -> Optional[re.Pattern]:
    if not pattern:
        return None
    return re.compile(pattern)


def build_pipelines(cfg: Dict[str, Any]) -> List[PipelineConfig]:
    defaults = cfg.get("defaults", {}) or {}
    default_llm_enabled = bool(defaults.get("llm_enabled", False))
    default_llm_min_sev = Severity.from_string(defaults.get("llm_min_severity", "WARNING"))
    default_max_excerpt_lines = int(
        defaults.get("max_excerpt_lines", defaults.get("max_chunk_lines", 20))
    )

    pipelines_cfg = cfg.get("pipelines", []) or []
    pipelines: List[PipelineConfig] = []

    for item in pipelines_cfg:
        name = item["name"]

        match_cfg = item.get("match", {}) or {}
        filename_regex = match_cfg.get("filename_regex", ".*")
        match_filename_regex = re.compile(filename_regex)

        classifier_cfg = item.get("classifier", {}) or {}
        classifier_type = classifier_cfg.get("type", "regex_counter")

        def _compile_list(key: str) -> List[re.Pattern]:
            patterns = classifier_cfg.get(key, []) or []
            return [re.compile(pat, re.IGNORECASE) for pat in patterns]

        error_regexes = _compile_list("error_regexes")
        warning_regexes = _compile_list("warning_regexes")
        ignore_regexes = _compile_list("ignore_regexes")

        llm_cfg_data = item.get("llm", {}) or {}
        llm_enabled = bool(llm_cfg_data.get("enabled", default_llm_enabled))
        llm_min_sev = Severity.from_string(
            llm_cfg_data.get("min_severity", default_llm_min_sev.name)
        )
        max_excerpt_lines = int(
            llm_cfg_data.get(
                "max_excerpt_lines", llm_cfg_data.get("max_chunk_lines", default_max_excerpt_lines)
            )
        )
        prompt_template_raw = llm_cfg_data.get("prompt_template")
        prompt_template_path = Path(prompt_template_raw) if prompt_template_raw else None

        llm_cfg = PipelineLLMConfig(
            enabled=llm_enabled,
            min_severity=llm_min_sev,
            max_excerpt_lines=max_excerpt_lines,
            prompt_template_path=prompt_template_path,
        )

        pipelines.append(
            PipelineConfig(
                name=name,
                match_filename_regex=match_filename_regex,
                classifier_type=classifier_type,
                classifier_error_regexes=error_regexes,
                classifier_warning_regexes=warning_regexes,
                classifier_ignore_regexes=ignore_regexes,
                llm_cfg=llm_cfg,
            )
        )

    if not pipelines:
        raise ValueError("No pipelines defined in configuration")

    return pipelines


def build_modules(cfg: Dict[str, Any]) -> List[ModuleConfig]:
    modules_cfg = cfg.get("modules", []) or []
    modules: List[ModuleConfig] = []

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

        min_sev = Severity.from_string(item.get("min_print_severity", "INFO"))

        llm_cfg = item.get("llm", {}) or {}

        emit_dir_raw = llm_cfg.get("emit_llm_payloads_dir", item.get("emit_llm_payloads_dir"))
        emit_dir = Path(emit_dir_raw) if emit_dir_raw else None

        stream_cfg = item.get("stream", {}) or {}
        stream_from_beginning = bool(stream_cfg.get("from_beginning", False))
        stream_interval = float(stream_cfg.get("interval", 1.0))

        llm_mode_raw = llm_cfg.get("llm_payload_mode", item.get("llm_payload_mode", "full"))
        llm_mode = str(llm_mode_raw).lower()
        if llm_mode not in ("full", "errors_only"):
            raise ValueError(
                f"Module {name}: invalid llm_payload_mode {llm_mode} (expected 'full' or 'errors_only')"
            )

        only_last_chunk = bool(llm_cfg.get("only_last_chunk", item.get("only_last_chunk", False)))

        exit_map_raw = item.get("exit_code_by_severity")
        exit_map = None
        if exit_map_raw:
            exit_map = {}
            for sev_name, code in exit_map_raw.items():
                sev = Severity.from_string(str(sev_name))
                exit_map[sev] = int(code)

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

        baseline_cfg = None
        baseline_raw = item.get("baseline")
        if baseline_raw:
            bl_enabled = bool(baseline_raw.get("enabled", True))
            state_file_raw = baseline_raw.get("state_file") or f".logtriage_{name}_baseline.json"
            state_file = Path(state_file_raw)
            window = int(baseline_raw.get("window", 20))
            error_mult = float(baseline_raw.get("error_multiplier", 3.0))
            warn_mult = float(baseline_raw.get("warning_multiplier", 3.0))
            sev_on_anom = Severity.from_string(baseline_raw.get("severity_on_anomaly", "ERROR"))
            baseline_cfg = BaselineConfig(
                enabled=bl_enabled,
                state_file=state_file,
                window=window,
                error_multiplier=error_mult,
                warning_multiplier=warn_mult,
                severity_on_anomaly=sev_on_anom,
            )

        enabled = bool(item.get("enabled", True))

        modules.append(
            ModuleConfig(
                name=name,
                path=Path(path_str),
                mode=mode,
                pipeline_name=pipeline_name,
                output_format=output_format,
                min_print_severity=min_sev,
                emit_llm_payloads_dir=emit_dir,
                stream_from_beginning=stream_from_beginning,
                stream_interval=stream_interval,
                llm_payload_mode=llm_mode,
                only_last_chunk=only_last_chunk,
                exit_code_by_severity=exit_map,
                alert_mqtt=alert_mqtt_cfg,
                alert_webhook=alert_webhook_cfg,
                baseline=baseline_cfg,
                enabled=enabled,
            )
        )

    return modules
