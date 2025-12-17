import argparse
import json
import logging
import os
import sys
import threading
from threading import Event
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_config, build_pipelines, build_rag_config, build_llm_config, build_modules
from .models import GlobalLLMConfig, ModuleConfig, Severity, Finding, PipelineConfig
from .engine import analyze_path
from .llm_client import analyze_findings_with_llm
from .llm_payload import write_llm_payloads, should_send_to_llm
from .utils import select_pipeline
from .stream import stream_file
from .alerts import send_alerts
from .webui.db import setup_database, cleanup_old_findings, store_finding, get_next_finding_index
from .version import __version__

# Import RAG client (optional import to avoid circular dependencies)
try:
    from .rag.service_client import create_rag_client
except ImportError:
    create_rag_client = None

# Global RAG monitoring state for CLI
rag_monitor_thread: Optional[threading.Thread] = None
rag_monitor_stop_event = threading.Event()
rag_monitor_lock = threading.Lock()
rag_client = None
rag_monitor_status = {
    "last_check": None,
    "rag_available": False,
    "rag_ready": False,
    "check_interval": 15  # seconds
}


def rag_monitor_worker(config_path: Path):
    """Background worker that periodically checks RAG service status for CLI."""
    global rag_client, rag_monitor_status
    logger = logging.getLogger(__name__)
    
    while not rag_monitor_stop_event.wait(rag_monitor_status["check_interval"]):
        try:
            # Check if we should try to initialize/reconnect to RAG service
            if rag_client is None or not hasattr(rag_client, 'is_ready'):
                # Try to create a new RAG client
                if create_rag_client is not None:
                    try:
                        cfg = load_config(config_path)
                        rag_config = build_rag_config(cfg)
                        if rag_config and rag_config.enabled:
                            rag_service_url = rag_config.service_url if hasattr(rag_config, 'service_url') else "http://127.0.0.1:8091"
                            new_client = create_rag_client(rag_service_url, fallback=False)  # Don't fallback, we want to know actual status
                            
                            if new_client and hasattr(new_client, 'is_ready') and new_client.is_ready():
                                with rag_monitor_lock:
                                    rag_client = new_client
                                    rag_monitor_status["rag_available"] = True
                                    rag_monitor_status["rag_ready"] = True
                                    logger.info("RAG service became available and ready")
                            elif new_client and hasattr(new_client, 'is_healthy') and new_client.is_healthy():
                                # Service is up but not ready yet
                                with rag_monitor_lock:
                                    rag_client = new_client
                                    rag_monitor_status["rag_available"] = True
                                    rag_monitor_status["rag_ready"] = False
                                    logger.info("RAG service is available but still initializing")
                    except Exception as e:
                        logger.debug(f"RAG service not yet available: {e}")
                        with rag_monitor_lock:
                            rag_monitor_status["rag_available"] = False
                            rag_monitor_status["rag_ready"] = False
            else:
                # Check existing client status
                try:
                    if rag_client and hasattr(rag_client, 'is_ready'):
                        is_ready = rag_client.is_ready()
                        is_healthy = rag_client.is_healthy() if hasattr(rag_client, 'is_healthy') else is_ready
                        
                        with rag_monitor_lock:
                            rag_monitor_status["rag_available"] = is_healthy
                            rag_monitor_status["rag_ready"] = is_ready
                                    
                except Exception as e:
                    logger.debug(f"RAG service check failed: {e}")
                    with rag_monitor_lock:
                        rag_monitor_status["rag_available"] = False
                        rag_monitor_status["rag_ready"] = False
                        rag_client = None  # Reset client
            
            with rag_monitor_lock:
                rag_monitor_status["last_check"] = time.time()
                
        except Exception as e:
            logger.error(f"RAG monitor worker error: {e}")
            with rag_monitor_lock:
                rag_monitor_status["rag_available"] = False
                rag_monitor_status["rag_ready"] = False

def start_rag_monitor(config_path: Path):
    """Start the RAG monitoring background thread for CLI."""
    global rag_monitor_thread
    logger = logging.getLogger(__name__)
    
    with rag_monitor_lock:
        if rag_monitor_thread is None or not rag_monitor_thread.is_alive():
            rag_monitor_stop_event.clear()
            rag_monitor_thread = threading.Thread(target=rag_monitor_worker, args=(config_path,), daemon=True)
            rag_monitor_thread.start()
            logger.info("RAG monitoring thread started")

def stop_rag_monitor():
    """Stop the RAG monitoring background thread for CLI."""
    global rag_monitor_thread
    logger = logging.getLogger(__name__)
    
    rag_monitor_stop_event.set()
    if rag_monitor_thread and rag_monitor_thread.is_alive():
        rag_monitor_thread.join(timeout=5)
    logger.info("RAG monitoring thread stopped")

def get_rag_client() -> Optional:
    """Get current RAG client status for CLI."""
    with rag_monitor_lock:
        return rag_client if rag_monitor_status["rag_ready"] else None

def configure_logging(cfg: Dict) -> None:
    """Configure logging based on configuration.
    
    Args:
        cfg: Configuration dictionary containing logging settings
    """
    logging_config = cfg.get("logging", {})
    
    # Default logging settings
    level = logging_config.get("level", "INFO")
    format_str = logging_config.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    log_file = logging_config.get("file")
    logger_levels = logging_config.get("loggers", {})
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure handlers
    handlers = []
    
    # Console handler (always included)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_str))
    handlers.append(console_handler)
    
    # File handler (optional)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format_str))
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file handler: {e}", file=sys.stderr)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format=format_str,
        force=True  # Override any existing configuration
    )
    
    # Configure specific loggers
    for logger_name, logger_level in logger_levels.items():
        try:
            logger = logging.getLogger(logger_name)
            logger_numeric_level = getattr(logging, logger_level.upper(), logging.INFO)
            logger.setLevel(logger_numeric_level)
        except Exception as e:
            print(f"Warning: Could not configure logger {logger_name}: {e}", file=sys.stderr)
    
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
