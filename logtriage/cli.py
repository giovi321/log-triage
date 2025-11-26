import argparse
import json
import sys
from pathlib import Path
from threading import Event
from typing import Dict, List, Optional

from .models import GlobalLLMConfig, Severity, Finding, ModuleConfig, PipelineConfig
from .config import build_llm_config, build_modules, build_pipelines, load_config
from .engine import analyze_path
from .llm_client import analyze_findings_with_llm
from .llm_payload import write_llm_payloads, should_send_to_llm
from .utils import select_pipeline
from .stream import stream_file
from .baseline import apply_baseline
from .alerts import send_alerts
from .webui.db import setup_database, cleanup_old_findings, store_finding
from .version import __version__


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="log-triage: rule-based log triage and LLM payload generator."
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    p.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML configuration file.",
    )
    p.add_argument(
        "--module",
        "-m",
        help="Name of a single module to run (from config.yaml). "
             "If omitted, all enabled follow-mode modules are run.",
    )
    p.add_argument(
        "--reload-on-change",
        action="store_true",
        help="Automatically reload when the config file mtime changes (handy when saving via the Web UI).",
    )
    return p.parse_args(argv)


def print_text_summary(findings: List[Finding], min_sev: Severity) -> None:
    for f in findings:
        if f.severity < min_sev:
            continue
        print(f"{f.file_path} [{f.pipeline_name}] finding={f.finding_index}")
        print(f"  severity: {f.severity.name}")
        print(f"  reason:   {f.message}")
        print(f"  lines:    {f.line_start}-{f.line_end}")
        print(f"  needs_llm: {f.needs_llm}")
        print()


def print_json_summary(findings: List[Finding]) -> None:
    out = []
    for f in findings:
        out.append(
            {
                "file_path": str(f.file_path),
                "pipeline": f.pipeline_name,
                "finding_index": f.finding_index,
                "severity": f.severity.name,
                "reason": f.message,
                "needs_llm": f.needs_llm,
                "line_start": f.line_start,
                "line_end": f.line_end,
            }
        )
    json.dump(out, sys.stdout, indent=2)
    print()


def run_module_batch(
    mod: ModuleConfig, pipelines: List[PipelineConfig], llm_defaults: "GlobalLLMConfig"
) -> List[Finding]:
    pipeline_map: Dict[str, PipelineConfig] = {p.name: p for p in pipelines}
    findings = analyze_path(
        mod.path,
        pipelines,
        mod.llm,
        mod.llm.max_excerpt_lines,
        context_prefix_lines=mod.llm.context_prefix_lines,
        pipeline_override=mod.pipeline_name,
    )

    if mod.baseline is not None:
        findings = apply_baseline(mod.baseline, findings)

    for f in findings:
        f.needs_llm = should_send_to_llm(mod.llm, f.severity, f.excerpt)

    analyze_findings_with_llm(findings, llm_defaults, mod.llm)

    for f in findings:
        if mod.alert_mqtt or mod.alert_webhook:
            send_alerts(mod, f)
        try:
            store_finding(mod.name, f, anomaly_flag=f.rule_id == "baseline_anomaly")
        except Exception:
            pass

    if mod.llm.emit_llm_payloads_dir:
        write_llm_payloads(
            findings,
            mod.llm,
            mod.llm.emit_llm_payloads_dir,
        )

    if mod.output_format == "json":
        print_json_summary(findings)
    else:
        print_text_summary(findings, mod.min_print_severity)

    return findings

def run_module_follow(
    mod: ModuleConfig,
    pipelines: List[PipelineConfig],
    llm_defaults: "GlobalLLMConfig",
    should_reload=None,
) -> None:
    if not mod.path.is_file():
        print(f"Module {mod.name}: follow mode requires a file path, got {mod.path}", file=sys.stderr)
        sys.exit(1)

    if mod.pipeline_name:
        pipeline = next((p for p in pipelines if p.name == mod.pipeline_name), None)
        if pipeline is None:
            raise ValueError(f"Module {mod.name}: unknown pipeline {mod.pipeline_name}")
    else:
        pipeline = select_pipeline(pipelines, mod.path)

    stream_file(mod, pipeline, llm_defaults, should_reload=should_reload)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    cfg_path = Path(args.config)
    reload_event = Event()
    last_cfg_mtime_ns: Optional[int] = None

    def _config_changed() -> bool:
        nonlocal last_cfg_mtime_ns
        if not args.reload_on_change:
            return False
        try:
            current = cfg_path.stat().st_mtime_ns
        except FileNotFoundError:
            return False
        if last_cfg_mtime_ns is None:
            return False
        if current != last_cfg_mtime_ns:
            reload_event.set()
            return True
        return False

    def _should_reload() -> bool:
        return reload_event.is_set() or _config_changed()

    while True:
        cfg = load_config(cfg_path)
        pipelines = build_pipelines(cfg)
        llm_defaults = build_llm_config(cfg)
        modules = build_modules(cfg, llm_defaults)

        try:
            last_cfg_mtime_ns = cfg_path.stat().st_mtime_ns
        except FileNotFoundError:
            last_cfg_mtime_ns = None

        # Optional database initialisation
        db_cfg = {}
        if isinstance(cfg, dict):
            db_cfg = cfg.get("database") or {}
        db_url = db_cfg.get("url")
        retention_days = int(db_cfg.get("retention_days", 0) or 0)
        if db_url:
            setup_database(db_url)
            if retention_days > 0:
                try:
                    cleanup_old_findings(retention_days)
                except Exception:
                    # do not abort if cleanup fails
                    pass

        if args.module:
            modules_to_run = [m for m in modules if m.name == args.module]
            if not modules_to_run:
                print(f"No module named {args.module} found in config.", file=sys.stderr)
                sys.exit(1)
        else:
            modules_to_run = [m for m in modules if m.enabled and m.mode == "follow"]

        if not modules_to_run:
            print("No enabled follow-mode modules found in config.", file=sys.stderr)
            sys.exit(1)

        has_follow_module = any(mod.mode == "follow" for mod in modules_to_run)

        for mod in modules_to_run:
            if mod.mode == "follow":
                run_module_follow(
                    mod,
                    pipelines,
                    llm_defaults,
                    should_reload=_should_reload if args.reload_on_change else None,
                )
            else:
                run_module_batch(mod, pipelines, llm_defaults)

            if args.reload_on_change and reload_event.is_set():
                print("Reload requested; reloading configuration...", file=sys.stderr)
                break

        if not (args.reload_on_change and reload_event.is_set() and has_follow_module):
            break

        reload_event.clear()

