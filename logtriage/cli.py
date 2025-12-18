import argparse
import json
import logging
import sys
from threading import Event
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_config, build_pipelines, build_rag_config, build_llm_config, build_modules
from .models import GlobalLLMConfig, ModuleConfig, Severity, Finding, PipelineConfig
from .logging_setup import configure_logging_from_dict
from .engine import analyze_path
from .llm_client import analyze_findings_with_llm
from .llm_payload import write_llm_payloads, should_send_to_llm
from .utils import select_pipeline
from .stream import stream_file
from .alerts import send_alerts
from .webui.db import setup_database, cleanup_old_findings, store_finding, get_next_finding_index
from .version import __version__
from .rag.monitor import RAGServiceMonitor

# Import RAG client (optional import to avoid circular dependencies)
try:
    from .rag.service_client import create_rag_client
except ImportError:
    create_rag_client = None

# Global RAG monitoring state for CLI
rag_client = None
rag_monitor_status = {
    "last_check": None,
    "rag_available": False,
    "rag_ready": False,
    "check_interval": 15,
}


def _cli_get_rag_service_url(cfg_path: Path) -> Optional[str]:
    try:
        cfg = load_config(cfg_path)
        rag_cfg = build_rag_config(cfg)
    except Exception:
        return None
    if not rag_cfg or not getattr(rag_cfg, "enabled", False):
        return None
    return getattr(rag_cfg, "service_url", None) or "http://127.0.0.1:8091"


def _cli_create_rag_client(service_url: str):
    if create_rag_client is None:
        raise RuntimeError("RAG client factory not available")
    return create_rag_client(service_url, fallback=False)


_rag_monitor: Optional[RAGServiceMonitor] = None


def start_rag_monitor(config_path: Path):
    """Start the RAG monitoring background thread for CLI."""
    global _rag_monitor
    logger = logging.getLogger(__name__)

    if _rag_monitor is None:
        _rag_monitor = RAGServiceMonitor(
            status=rag_monitor_status,
            get_service_url=lambda: _cli_get_rag_service_url(config_path),
            create_client=_cli_create_rag_client,
            get_client=lambda: rag_client,
            set_client=lambda c: _set_cli_rag_client(c),
            timestamp_mode="unix",
            include_detailed_status=False,
            logger=logger,
        )

    _rag_monitor.start()
    logger.info("RAG monitoring thread started")


def _set_cli_rag_client(client) -> None:
    global rag_client
    rag_client = client


def stop_rag_monitor():
    """Stop the RAG monitoring background thread for CLI."""
    global _rag_monitor
    logger = logging.getLogger(__name__)

    if _rag_monitor is not None:
        _rag_monitor.stop()
    logger.info("RAG monitoring thread stopped")


def get_rag_client() -> Optional:
    """Get current RAG client status for CLI."""
    if _rag_monitor is None:
        return None
    return _rag_monitor.get_ready_client()


def configure_logging(cfg: Dict) -> None:
    """Configure logging based on configuration.

    Args:
        cfg: Configuration dictionary containing logging settings
    """
    level, log_file = configure_logging_from_dict(cfg)
 
    # Log that configuration is complete
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at level {level}")
    if log_file:
        logger.info(f"Also logging to file: {log_file}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for logtriage.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
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
    """Print findings in human-readable text format.
    
    Args:
        findings: List of findings to print
        min_sev: Minimum severity to include in output
    """
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
    """Print findings in JSON format.
    
    Args:
        findings: List of findings to serialize and print
    """
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


def _modules_to_run(modules: List[ModuleConfig], selected_name: Optional[str]) -> List[ModuleConfig]:
    """Filter modules based on CLI selection.

    - When a specific module is requested, return that module even if it is disabled
      so the user can explicitly run it.
    - Otherwise, include only enabled follow modules (batch modules must be explicitly requested).
    
    Args:
        modules: All available modules from config
        selected_name: Specific module name requested by user
        
    Returns:
        List of modules that should be executed
    """

    if selected_name:
        return [m for m in modules if m.name == selected_name]
    return [m for m in modules if m.enabled and m.mode == "follow"]


def run_module_batch(
    mod: ModuleConfig, pipelines: List[PipelineConfig], llm_defaults: "GlobalLLMConfig"
) -> List[Finding]:
    """Execute a module in batch mode (scan once and exit).
    
    Args:
        mod: Module configuration to execute
        pipelines: Available pipelines for processing
        llm_defaults: Global LLM configuration
        
    Returns:
        List of findings discovered during processing
    """
    pipeline_map: Dict[str, PipelineConfig] = {p.name: p for p in pipelines}
    
    # Get RAG client from monitor (will be None if not ready)
    rag_client = get_rag_client()
    if rag_client:
        logger.info(f"Using RAG service client for module {mod.name}")
    else:
        logger.info(f"RAG service not ready for module {mod.name}, proceeding without RAG")
    
    findings = analyze_path(
        mod.path,
        pipelines,
        mod.llm,
        mod.llm.max_excerpt_lines,
        context_prefix_lines=mod.llm.context_prefix_lines,
        context_suffix_lines=mod.llm.context_suffix_lines,
        pipeline_override=mod.pipeline_name,
        rag_client=rag_client,
    )
    
    # Filter findings based on module configuration
    findings = [
        f
        for f in findings
        if f.severity in mod.severities and (not f.ignored or mod.include_ignored)
    ]
    
    # Analyze findings with LLM if needed
    if mod.llm.enabled and findings:
        logger.info(f"Analyzing {len(findings)} findings with LLM for module {mod.name}")
        findings = analyze_findings_with_llm(
            findings,
            mod.llm,
            llm_defaults,
            rag_client=rag_client,
        )
    
    for f in findings:
        f.needs_llm = should_send_to_llm(mod.llm, f.severity, f.excerpt)

    for f in findings:
        if mod.alert_mqtt or mod.alert_webhook:
            send_alerts(mod, f)
        try:
            store_finding(mod.name, f)
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
    """Execute a module in follow mode (continuous tailing).
    
    This function runs indefinitely until interrupted or reloaded.
    
    Args:
        mod: Module configuration to execute
        pipelines: Available pipelines for processing
        llm_defaults: Global LLM configuration
        should_reload: Optional callback to check for reload requests
    """
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
    """Main entry point for logtriage CLI.
    
    Loads configuration, sets up modules, and executes them according
    to their mode (batch or follow). Handles configuration reloading
    when requested.
    
    Args:
        argv: Optional command line arguments (for testing)
    """
    args = parse_args(argv)

    cfg_path = Path(args.config)
    
    # Initial configuration load
    cfg = load_config(cfg_path)
    
    # Configure logging based on config
    configure_logging(cfg)
    logger = logging.getLogger(__name__)
    logger.info(f"logtriage starting with config: {cfg_path}")
    
    # Start RAG monitoring
    start_rag_monitor(cfg_path)
    
    try:
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
            
            # Reconfigure logging when config is reloaded
            configure_logging(cfg)
            logger = logging.getLogger(__name__)
            logger.info("Configuration reloaded")
            
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

            modules_to_run = _modules_to_run(modules, args.module)

            logger.info(f"Found {len(modules)} total modules, {len(modules_to_run)} to run")
            for mod in modules_to_run:
                logger.info(f"Module: {mod.name} (mode: {mod.mode}, enabled: {mod.enabled})")

            if args.module and not modules_to_run:
                logger.error(f"No module named {args.module} found in config")
                print(f"No module named {args.module} found in config.", file=sys.stderr)
                sys.exit(1)

            if not args.module and not modules_to_run:
                logger.error("No enabled modules found in config")
                print("No enabled modules found in config.", file=sys.stderr)
                sys.exit(1)

            has_follow_module = any(mod.mode == "follow" for mod in modules_to_run)

            for mod in modules_to_run:
                if mod.mode == "follow":
                    logger.info(f"Starting follow mode for module: {mod.name}")
                    run_module_follow(
                        mod,
                        pipelines,
                        llm_defaults,
                        should_reload=_should_reload if args.reload_on_change else None,
                    )
                else:
                    logger.info(f"Running batch mode for module: {mod.name}")
                    run_module_batch(mod, pipelines, llm_defaults)

                if args.reload_on_change and reload_event.is_set():
                    logger.info("Reload requested; reloading configuration...")
                    print("Reload requested; reloading configuration...", file=sys.stderr)
                    break

            if not (args.reload_on_change and reload_event.is_set() and has_follow_module):
                break

    finally:
        # Stop RAG monitoring on exit
        stop_rag_monitor()
