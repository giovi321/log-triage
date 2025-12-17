from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import threading
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Import FastAPI and related dependencies
try:
    from fastapi import FastAPI, Request, Response, HTTPException, status, Form
    from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
    from fastapi.templating import Jinja2Templates
    from fastapi.staticfiles import StaticFiles
    from fastapi.security import HTTPBasic, HTTPBasicCredentials
    from fastapi.middleware import SessionMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
    from starlette.responses import PlainTextResponse
except ImportError:
    print("FastAPI dependencies are missing. Install with: pip install fastapi uvicorn jinja2 python-multipart", file=sys.stderr)
    sys.exit(1)

# Import LogTriage components
from ..models import GlobalLLMConfig, Severity, Finding, ModuleConfig, PipelineConfig
from ..config import build_llm_config, build_modules, build_pipelines, load_config, build_rag_config
from ..engine import analyze_path
from ..llm_client import analyze_findings_with_llm
from ..llm_payload import write_llm_payloads, should_send_to_llm
from ..utils import select_pipeline
from ..stream import stream_file
from ..alerts import send_alerts
from ..webui.db import setup_database, cleanup_old_findings, store_finding, get_next_finding_index
from ..version import __version__
from .ingestion_status import INGESTION_STALENESS_MINUTES, _derive_ingestion_status

# Import RAG client (optional import to avoid circular dependencies)
try:
    from ..rag import RAGClient
    from ..rag.service_client import create_rag_client
except ImportError:
    RAGClient = None
    create_rag_client = None
from .config import load_full_config, parse_webui_settings, WebUISettings, get_client_ip
from .auth import authenticate_user, create_session_token, get_current_user, pwd_context

logger = logging.getLogger(__name__)

# Global RAG monitoring state
rag_monitor_thread: Optional[threading.Thread] = None
rag_monitor_stop_event = threading.Event()
rag_monitor_lock = threading.Lock()
rag_monitor_status = {
    "last_check": None,
    "rag_available": False,
    "rag_ready": False,
    "detailed_status": None,
    "check_interval": 10  # seconds
}
from .db import (
    delete_all_findings,
    delete_findings_for_module,
    delete_findings_by_ids,
    delete_findings_matching_regex,
    delete_finding_by_id,
    get_finding_by_id,
    get_module_stats,
    count_open_findings_for_module,
    get_next_finding_index,
    setup_database,
    get_recent_findings_for_module,
    update_finding_llm_data,
    update_finding_severity,
    store_finding,
)
from .regex_utils import (
    _compile_regex_with_feedback,
    _filter_finding_intro_lines,
    _lint_regex_input,
    _prepare_sample_lines,
)


app = FastAPI(title="log-triage Web UI")

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
SAMPLE_LOG_DIR = ROOT_DIR / "samples"
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.globals.update({"app_version": __version__})
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


def _format_local_timestamp(value: Optional[datetime.datetime]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    ts = value
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=datetime.timezone.utc)
    return ts.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


templates.env.filters["localtime"] = _format_local_timestamp


db_status: Dict[str, Any] = {
    "configured": False,
    "connected": False,
    "error": None,
    "url": None,
}

SEVERITY_CHOICES = ["CRITICAL", "ERROR", "WARNING"]


def _init_database(raw: Dict[str, Any]):
    db_cfg = raw.get("database") or {}
    url = db_cfg.get("url")
    db_status.update({"configured": bool(url), "connected": False, "error": None, "url": url})
    if not url:
        return
    try:
        setup_database(url)
        db_status["connected"] = True
    except Exception as exc:
        db_status["error"] = str(exc)


def _load_context_hints() -> Dict[str, str]:
    """Load context hints for the config editor.
    We try a couple of locations and also repair common escape issues
    (like unescaped `\\.` in regex examples inside JSON strings).
    """
    candidates = [
        BASE_DIR / "context_hints.json",
        ASSETS_DIR / "context_hints.json",
    ]

    for path in candidates:
        try:
            if not path.exists():
                continue
            raw = path.read_text(encoding="utf-8")
            # The JSON file should be valid as-is; only apply escape fix if needed
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Fix common invalid escape in JSON like `\.` (used in regex examples)
                # which otherwise triggers `Invalid \escape` errors.
                raw_fixed = raw.replace("\\.", "\\\\.")
                data = json.loads(raw_fixed)
            if isinstance(data, dict):
                data.setdefault(
                    "root",
                    "Top-level sections mirror the README. Move the cursor to a section to see details.",
                )
                return data
        except Exception:
            continue

    # Fallback: generic root hint only
    return {
        "root": "Top-level sections mirror the README. Move the cursor to a section to see details."
    }


def _available_sample_logs() -> List[Dict[str, Any]]:
    if not SAMPLE_LOG_DIR.exists():
        return []

    entries: List[Dict[str, Any]] = []
    for path in sorted(SAMPLE_LOG_DIR.iterdir()):
        if not path.is_file():
            continue
        label = path.stem.replace("_", " ").title()
        entries.append(
            {"value": f"sample:{path.stem}", "label": label, "path": path}
        )
    return entries


def _sample_source_options() -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = [
        {"value": "tail", "label": "Log tail (live)"},
        {"value": "errors", "label": "Identified errors"},
    ]
    for entry in _available_sample_logs():
        options.append(
            {
                "value": entry.get("value"),
                "label": f"Sample log: {entry.get('label', 'unknown')}",
            }
        )
    return options


def _normalize_sample_source(value: str) -> str:
    allowed = {opt.get("value") for opt in _sample_source_options()}
    return value if value in allowed else "tail"


def _sample_source_label(value: str) -> str:
    for opt in _sample_source_options():
        if opt.get("value") == value:
            return opt.get("label", value)
    return "Log tail (live)"


def _load_summary_prompt_template() -> Optional[str]:
    path = getattr(llm_defaults, "summary_prompt_path", None)
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return None


def _load_settings_and_config() -> tuple[WebUISettings, Dict[str, Any], Path]:
    cfg_path_str = os.environ.get("LOGTRIAGE_CONFIG", "config.yaml")
    cfg_path = Path(cfg_path_str).resolve()
    try:
        raw = load_full_config(cfg_path)
    except SystemExit as exc:
        add_notification("error", "Configuration load failed", str(exc))
        raw = {}
    except Exception as exc:  # pragma: no cover - defensive
        add_notification("error", "Configuration load failed", str(exc))
        raw = {}

    web_settings = parse_webui_settings(raw)
    _init_database(raw)
    return web_settings, raw, cfg_path


settings, raw_config, CONFIG_PATH = _load_settings_and_config()
llm_defaults: GlobalLLMConfig = build_llm_config(raw_config)
rag_client: Optional[RAGClient] = None
context_hints = _load_context_hints()
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, session_cookie=settings.session_cookie_name)

REGEX_WIZARD_STEPS = [
    ("pick", "Pick sample lines"),
    ("draft", "Draft regex"),
    ("test", "Run tests"),
    ("save", "Save to pipeline"),
]


def _get_regex_state(request: Request) -> Dict[str, Any]:
    state = request.session.get("regex_lab_state") or {}
    normalized = {
        "module": state.get("module"),
        "sample_source": state.get("sample_source", "tail"),
        "regex_value": state.get("regex_value", ""),
        "regex_kind": state.get("regex_kind", "error"),
        "matches": state.get("matches", []),
        "step": state.get("step", "pick"),
    }
    return normalized


def _update_regex_state(request: Request, **updates: Any) -> Dict[str, Any]:
    state = _get_regex_state(request)
    if "module" in updates and updates["module"] != state.get("module"):
        state["matches"] = []
        state["step"] = "pick"

    for key, value in updates.items():
        if value is not None:
            state[key] = value

    request.session["regex_lab_state"] = state
    return state


def _regex_wizard_metadata(current_step: str) -> Dict[str, Any]:
    steps = []
    seen_current = False
    for step_id, label in REGEX_WIZARD_STEPS:
        if step_id == current_step:
            status = "active"
            seen_current = True
        elif not seen_current:
            status = "complete"
        else:
            status = "upcoming"
        steps.append({"id": step_id, "label": label, "status": status})
    return {
        "steps": steps,
        "active_label": next((label for sid, label in REGEX_WIZARD_STEPS if sid == current_step), REGEX_WIZARD_STEPS[0][1]),
    }


def _regex_step_hints(step: str) -> List[Dict[str, str]]:
    def _hint(key: str, fallback: str) -> str:
        return context_hints.get(key, fallback)

    mapping = {
        "pick": [
            {
                "title": "Module context",
                "body": _hint(
                    "modules_path",
                    "Each module points at a path. Switch modules to pull sample lines from different log files.",
                ),
            },
        ],
        "draft": [
            {
                "title": "Capture the right severity",
                "body": _hint("classifier_error_regexes", "Use classifier.error_regexes to flag error patterns."),
            },
            {
                "title": "Ignore the noise",
                "body": _hint("classifier_ignore_regexes", "Add noisy patterns to classifier.ignore_regexes so they are skipped."),
            },
        ],
        "test": [
            {
                "title": "Context windows",
                "body": _hint(
                    "modules_llm_context_prefix_lines",
                    "Context prefix lines influence what the LLM sees alongside a match when enabled.",
                ),
            },
        ],
        "save": [
            {
                "title": "Pipelines own the rules",
                "body": _hint("pipelines_classifier", "Pipelines carry classifier regex lists shared across modules."),
            },
            {
                "title": "Name ties it together",
                "body": _hint(
                    "pipelines_name",
                    "modules.pipeline points at the pipeline name that will receive the saved regex.",
                ),
            },
        ],
    }

    return mapping.get(step, mapping["pick"])


def _build_all_regex_hints() -> Dict[str, List[Dict[str, str]]]:
    return {step: _regex_step_hints(step) for step, _ in REGEX_WIZARD_STEPS}


def _evaluate_regex_against_lines(
    regex_value: str, sample_lines: List[str], first_line_number: int = 0
) -> tuple[list[int], Optional[str]]:
    matches: List[int] = []
    error_msg: Optional[str] = None
    if not regex_value:
        return matches, error_msg

    try:
        pattern = re.compile(regex_value)
        matches = [idx + first_line_number for idx, line in enumerate(sample_lines) if pattern.search(line)]
    except re.error as e:
        error_msg = f"Regex error: {e}"

    return matches, error_msg


def rag_monitor_worker():
    """Background worker that periodically checks RAG service status."""
    global rag_client, rag_monitor_status
    
    while not rag_monitor_stop_event.wait(rag_monitor_status["check_interval"]):
        try:
            # Check if we should try to initialize/reconnect to RAG service
            if rag_client is None or not hasattr(rag_client, 'is_ready'):
                # Try to create a new RAG client
                if create_rag_client is not None:
                    try:
                        rag_config = build_rag_config(raw_config)
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
                            
                            # Get detailed status if available
                            if hasattr(rag_client, '_make_request'):
                                detailed_status = rag_client._make_request("GET", "/health")
                                if detailed_status:
                                    rag_monitor_status["detailed_status"] = detailed_status
                                    
                except Exception as e:
                    logger.debug(f"RAG service check failed: {e}")
                    with rag_monitor_lock:
                        rag_monitor_status["rag_available"] = False
                        rag_monitor_status["rag_ready"] = False
                        rag_client = None  # Reset client
            
            with rag_monitor_lock:
                rag_monitor_status["last_check"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                
        except Exception as e:
            logger.error(f"RAG monitor worker error: {e}")
            with rag_monitor_lock:
                rag_monitor_status["rag_available"] = False
                rag_monitor_status["rag_ready"] = False

def start_rag_monitor():
    """Start the RAG monitoring background thread."""
    global rag_monitor_thread
    
    with rag_monitor_lock:
        if rag_monitor_thread is None or not rag_monitor_thread.is_alive():
            rag_monitor_stop_event.clear()
            rag_monitor_thread = threading.Thread(target=rag_monitor_worker, daemon=True)
            rag_monitor_thread.start()
            logger.info("RAG monitoring thread started")

def stop_rag_monitor():
    """Stop the RAG monitoring background thread."""
    global rag_monitor_thread
    
    rag_monitor_stop_event.set()
    if rag_monitor_thread and rag_monitor_thread.is_alive():
        rag_monitor_thread.join(timeout=5)
    logger.info("RAG monitoring thread stopped")

def get_rag_monitor_status() -> Dict[str, Any]:
    """Get current RAG monitor status."""
    with rag_monitor_lock:
        return rag_monitor_status.copy()

def get_settings() -> WebUISettings:
    return settings


def _refresh_llm_defaults() -> None:
    global llm_defaults
    try:
        llm_defaults = build_llm_config(raw_config)
    except Exception as exc:
        add_notification("error", "LLM defaults error", str(exc))
        llm_defaults = GlobalLLMConfig(
            enabled=False,
            min_severity=Severity.ERROR,
            providers={},
            default_provider=None,
            context_prefix_lines=0,
            context_suffix_lines=0,
            summary_prompt_path=None,
        )


def _refresh_rag_client() -> None:
    """Initialize or update the RAG client based on configuration."""
    global rag_client
    
    # Try to use RAG service client
    if create_rag_client is not None:
        try:
            logger.info("Initializing RAG service client...")
            rag_config = build_rag_config(raw_config)
            if rag_config and rag_config.enabled:
                logger.info(f"RAG config found and enabled, using service client")
                # Get RAG service URL from config or use default
                rag_service_url = rag_config.service_url if hasattr(rag_config, 'service_url') else "http://127.0.0.1:8091"
                rag_client = create_rag_client(rag_service_url, fallback=True)
                
                if rag_client.is_healthy():
                    logger.info("RAG service client is healthy")
                    # Add module configurations to RAG client
                    modules = _build_modules_from_config()
                    logger.info(f"Adding {len(modules)} modules to RAG service")
                    for module in modules:
                        if module.rag and module.rag.enabled:
                            logger.info(f"Adding RAG config for module: {module.name}")
                            rag_client.add_module_config(module.name, module.rag)
                    # Update knowledge base
                    logger.info("Updating RAG service knowledge base...")
                    rag_client.update_knowledge_base()
                    logger.info("RAG service client initialization completed")
                else:
                    logger.warning("RAG service is not available, RAG functionality will be disabled")
                    # Keep rag_client as NoOp (already returned by create_rag_client)
            else:
                logger.info("RAG disabled in configuration")
                rag_client = None
        except Exception as exc:
            logger.error(f"RAG service client initialization failed: {exc}", exc_info=True)
            add_notification("warning", "RAG service unavailable", "RAG functionality will be disabled")
            rag_client = None
    else:
        logger.info("RAG service client not available, RAG functionality disabled")
        rag_client = None

def _build_modules_from_config() -> List[ModuleConfig]:
    try:
        return build_modules(raw_config, llm_defaults)
    except Exception as exc:
        add_notification("error", "Module configuration error", str(exc))
        return []


# Initialize RAG client after function definition
_refresh_rag_client()


from .ingestion_status import INGESTION_STALENESS_MINUTES, _derive_ingestion_status


def _render_config_editor(
    request: Request,
    username: str,
    config_text: str,
    *,
    error: Optional[str] = None,
    message: Optional[str] = None,
    status_code: int = status.HTTP_200_OK,
):
    # Reload hints to be safe
    current_hints = _load_context_hints()
    return templates.TemplateResponse(
        "config_edit.html",
        {
            "request": request,
            "username": username,
            "config_text": config_text,
            "error": error,
            "message": message,
            "context_hints": current_hints,
        },
        status_code=status_code,
    )


def _regex_context(
    request: Request,
    username: str,
    modules: List[ModuleConfig],
    module_obj,
    *,
    sample_lines: List[str],
    regex_value: str,
    regex_kind: str,
    matches: List[int],
    error: Optional[str],
    message: Optional[str],
    sample_source: str,
    regex_issues: Optional[List[str]] = None,
    active_step: str = "pick",
    wizard: Optional[Dict[str, Any]] = None,
    step_hints: Optional[Dict[str, List[Dict[str, str]]]] = None,
    recent_findings: Optional[List[Any]] = None,
    open_findings_count: Optional[int] = None,
):
    normalized_source = _normalize_sample_source(sample_source)
    return {
        "request": request,
        "username": username,
        "modules": modules,
        "current_module": module_obj,
        "sample_lines": sample_lines,
        "regex_value": regex_value,
        "regex_kind": regex_kind,
        "matches": matches,
        "error": error,
        "message": message,
        "sample_source": normalized_source,
        "sample_source_label": _sample_source_label(normalized_source),
        "sample_options": _sample_source_options(),
        "regex_issues": regex_issues,
        "wizard": wizard or _regex_wizard_metadata(active_step),
        "step_hints": step_hints or _build_all_regex_hints(),
        "active_step": active_step,
        "recent_findings": recent_findings or [],
        "open_findings_count": open_findings_count,
        "db_status": db_status,
    }


@app.middleware("http")
async def ip_allowlist_middleware(request: Request, call_next):
    s = settings
    if s.allowed_ips:
        ip = get_client_ip(request)
        if ip not in s.allowed_ips:
            return HTMLResponse("Access denied", status_code=status.HTTP_403_FORBIDDEN)
    response = await call_next(request)
    return response


@app.get("/login", name="login_form")
async def login_form(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": None, "username": get_current_user(request, settings)},
    )


@app.post("/login", name="login_form_post")
async def login_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    user = authenticate_user(settings, username, password)
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Invalid credentials",
                "username": None,
            },
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    token = create_session_token(username, settings.secret_key)
    request.session["session_token"] = token
    return RedirectResponse(url=app.url_path_for("dashboard"), status_code=status.HTTP_303_SEE_OTHER)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)


@app.get("/", name="dashboard")
async def dashboard(request: Request):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    modules = sorted(
        _build_modules_from_config(),
        key=lambda m: (not m.enabled, m.name.lower()),
    )
    stats = get_module_stats(modules)
    page_rendered_at = datetime.datetime.now(datetime.timezone.utc)
    ingestion_status = _derive_ingestion_status(modules, now=page_rendered_at)
    notif_summary = notification_summary()
    
    # Get RAG status from monitor
    rag_monitor_data = get_rag_monitor_status()
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "username": username,
            "modules": modules,
            "stats": stats,
            "db_status": db_status,
            "page_rendered_at": page_rendered_at,
            "ingestion_status": ingestion_status,
            "notif_summary": notif_summary,
            "rag_status": rag_monitor_data["detailed_status"],
            "rag_service_available": rag_monitor_data["rag_available"],
            "rag_service_ready": rag_monitor_data["rag_ready"],
            "rag_monitor": rag_monitor_data,
        },
    )




@app.get("/api/rag/status")
async def get_rag_status():
    """Get RAG service status for AJAX calls."""
    monitor_data = get_rag_monitor_status()
    
    if not monitor_data["rag_available"]:
        return {
            "enabled": False,
            "message": "RAG service unavailable",
            "service_available": False,
            "service_ready": False,
            "monitor": monitor_data
        }
    
    if not monitor_data["rag_ready"]:
        return {
            "enabled": False,
            "message": "RAG service initializing",
            "service_available": True,
            "service_ready": False,
            "monitor": monitor_data,
            "detailed_status": monitor_data["detailed_status"]
        }
    
    # RAG is ready
    try:
        if rag_client:
            status = rag_client.get_status()
            return {
                "enabled": status.get("enabled", False),
                "service_available": True,
                "service_ready": True,
                "total_repositories": status.get("total_repositories", 0),
                "vector_store_stats": status.get("vector_store_stats", {}),
                "repositories": status.get("repositories", []),
                "monitor": monitor_data
            }
        else:
            return {
                "enabled": False,
                "message": "RAG client not initialized",
                "service_available": True,
                "service_ready": False,
                "monitor": monitor_data
            }
    except Exception as e:
        return {
            "enabled": False,
            "message": f"Error getting RAG status: {str(e)}",
            "service_available": False,
            "service_ready": False,
            "monitor": monitor_data
        }

@app.get("/config/edit", name="edit_config")
async def edit_config(request: Request):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
    except Exception as e:
        text = f"Error reading {CONFIG_PATH}: {e}"

    return _render_config_editor(request, username, text)


@app.post("/config/edit", name="edit_config_post")
async def edit_config_post(
    request: Request,
    config_text: str = Form(...),
):
    global settings, raw_config, llm_defaults, rag_client

    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    try:
        parsed = yaml.safe_load(config_text) or {}
    except Exception as e:
        add_notification("error", "Configuration validation failed", str(e))
        return _render_config_editor(
            request,
            username,
            config_text,
            error=f"YAML error: {e}",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    tmp_path = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".tmp")
    backup_path = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".bak")
    try:
        tmp_path.write_text(config_text, encoding="utf-8")
        if CONFIG_PATH.exists():
            CONFIG_PATH.replace(backup_path)
        tmp_path.replace(CONFIG_PATH)
    except Exception as e:
        return _render_config_editor(
            request,
            username,
            config_text,
            error=f"Write error: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    try:
        raw_config = load_config(CONFIG_PATH)
        settings = parse_webui_settings(raw_config)
        _init_database(raw_config)
        _refresh_llm_defaults()
        _refresh_rag_client()
    except Exception as exc:
        add_notification("error", "Configuration reload failed", str(exc))
        return _render_config_editor(
            request,
            username,
            config_text,
            error=f"Reload failed: {exc}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return _render_config_editor(
        request,
        username,
        config_text,
        message="Configuration saved.",
    )


@app.post("/config/reload", name="reload_config")
async def reload_config(request: Request):
    global settings, raw_config, llm_defaults, rag_client

    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
    except Exception as e:
        text = f"Error reading {CONFIG_PATH}: {e}"
        add_notification("error", "Config reload failed", str(e))
        return _render_config_editor(request, username, text, error=f"Reload failed: {e}")

    try:
        new_raw = load_config(CONFIG_PATH)
        raw_config = new_raw
        settings = parse_webui_settings(new_raw)
        _init_database(new_raw)
        _refresh_llm_defaults()
        _refresh_rag_client()
    except Exception as e:
        add_notification("error", "Config reload failed", str(e))
        return _render_config_editor(
            request,
            username,
            text,
            error=f"Reload failed: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return _render_config_editor(
        request,
        username,
        text,
        message="Configuration reloaded from disk.",
    )


@app.get("/regex", name="regex_lab")
async def regex_lab(
    request: Request,
    module: Optional[str] = None,
    sample_source: str = "tail",
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    stored_state = _get_regex_state(request)
    module = module or stored_state.get("module")
    safe_sample_source = sample_source if sample_source in {"errors", "tail"} else stored_state.get("sample_source", "tail")
    safe_sample_source = safe_sample_source if safe_sample_source in {"errors", "tail"} else "tail"

    modules = _build_modules_from_config()
    stats = get_module_stats(modules)
    ingestion_status = _derive_ingestion_status(modules) if modules else None
    module_obj = None
    if modules:
        if module:
            module_obj = next((m for m in modules if m.name == module), None)
        if module_obj is None:
            module_obj = modules[0]

    raw_sample_lines: List[str] = []
    sample_start_line: int = 1
    sample_error: Optional[str] = None
    safe_sample_source = _normalize_sample_source(sample_source)
    if module_obj is not None:
        raw_sample_lines, sample_start_line, _, sample_error = _get_sample_lines_for_module(
            module_obj, safe_sample_source, max_lines=200
        )

    filtered_sample_lines = _filter_finding_intro_lines(raw_sample_lines)
    prepared_lines = _prepare_sample_lines(filtered_sample_lines, first_line_number=sample_start_line)
    matches, evaluation_error = _evaluate_regex_against_lines(
        stored_state.get("regex_value", ""), filtered_sample_lines, first_line_number=sample_start_line
    )
    active_step = stored_state.get("step", "pick")
    _update_regex_state(
        request,
        module=module_obj.name if module_obj else None,
        sample_source=safe_sample_source,
        matches=matches,
        step=active_step,
        regex_value=stored_state.get("regex_value", ""),
        regex_kind=stored_state.get("regex_kind", "error"),
    )

    # Fetch findings for the module
    recent_findings = []
    open_findings_count = None
    if module_obj and db_status.get("connected"):
        recent_findings = get_recent_findings_for_module(module_obj.name, limit=50)
        open_findings_count = count_open_findings_for_module(
            module_obj.name, severities=SEVERITY_CHOICES
        )

    wizard = _regex_wizard_metadata(active_step)
    step_hints = _build_all_regex_hints()
    return templates.TemplateResponse(
        "regex.html",
        _regex_context(
            request,
            username,
            modules,
            module_obj,
            sample_lines=prepared_lines,
            regex_value=stored_state.get("regex_value", ""),
            regex_kind=stored_state.get("regex_kind", "error"),
            matches=matches,
            error=sample_error or evaluation_error,
            message=None,
            sample_source=safe_sample_source,
            wizard=wizard,
            step_hints=step_hints,
            active_step=active_step,
            recent_findings=recent_findings,
            open_findings_count=open_findings_count,
        ),
    )


def _logs_redirect(
    module: Optional[str],
    message: Optional[str] = None,
    error: Optional[str] = None,
    tail_filter: Optional[str] = None,
    issue_filter: Optional[str] = None,
    sample_source: Optional[str] = None,
):
    params = {}
    if module:
        params["module"] = module
    if tail_filter:
        params["tail_filter"] = tail_filter
    if issue_filter:
        params["issue_filter"] = issue_filter
    if sample_source:
        params["sample_source"] = sample_source
    if message:
        params["message"] = message
    if error:
        params["error"] = error

    query = urllib.parse.urlencode(params)
    url = app.url_path_for("ai_logs")
    if query:
        url = f"{url}?{query}"
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)


def _build_log_view_state(
    module_obj,
    tail_filter: str,
    issue_filter: str,
    *,
    sample_lines_override: Optional[List[str]] = None,
    sample_start_line: int = 0,
    total_lines: int = 0,
):
    sample_lines: List[str] = []
    recent_findings = []
    finding_tail: List[Dict[str, Any]] = []
    available_severities: List[str] = []
    tail_filter_normalized = (tail_filter or "all").upper()
    issue_filter_normalized = (issue_filter or "all").upper()
    severity_choices = list(SEVERITY_CHOICES)
    had_finding_tail = False
    had_recent_findings = False
    if module_obj is not None:
        if sample_lines_override is not None:
            sample_lines = list(sample_lines_override)
        else:
            sample_lines, sample_start_line, total_lines = _tail_lines(
                Path(module_obj.path), max_lines=500
            )
        recent_findings = get_recent_findings_for_module(module_obj.name, limit=50)
        had_recent_findings = bool(recent_findings)
        finding_tail = _build_finding_tail(
            sample_lines,
            recent_findings,
            first_line_number=sample_start_line,
            module_path=Path(module_obj.path),
        )
        finding_line_examples = _finding_line_examples(finding_tail)
        had_finding_tail = bool(finding_tail)
        if issue_filter_normalized == "ACTIVE":
            recent_findings = [
                c
                for c in recent_findings
                if str(getattr(c, "severity", "")).upper() in SEVERITY_CHOICES
            ]
            finding_tail = _filter_finding_tail(
                finding_tail,
                tail_filter="ALL",
                extra_predicate=lambda section: str(
                    getattr(section.get("finding"), "severity", "")
                ).upper()
                in SEVERITY_CHOICES,
            )

        # Extract available severities from findings (handle unified view)
        for section in finding_tail:
            if section.get("unified_view"):
                # Unified view: findings are attached to individual lines
                for line_entry in section.get("lines", []):
                    finding = line_entry.get("finding")
                    if not finding:
                        continue
                    sev = str(getattr(finding, "severity", "")).upper()
                    if sev and sev not in available_severities:
                        available_severities.append(sev)
                    if sev and sev not in severity_choices:
                        severity_choices.append(sev)
            else:
                # Legacy: finding attached to section
                finding = section.get("finding")
                if not finding:
                    continue
                sev = str(getattr(finding, "severity", "")).upper()
                if sev and sev not in available_severities:
                    available_severities.append(sev)
                if sev and sev not in severity_choices:
                    severity_choices.append(sev)

        finding_tail = _filter_finding_tail(
            finding_tail, tail_filter_normalized, extra_predicate=None
        )

    else:
        finding_line_examples = {}

    return {
        "sample_lines": sample_lines,
        "recent_findings": recent_findings,
        "had_recent_findings": had_recent_findings,
        "finding_tail": finding_tail,
        "had_finding_tail": had_finding_tail,
        "tail_filter": tail_filter_normalized,
        "issue_filter": issue_filter_normalized,
        "tail_severities": available_severities,
        "severity_choices": severity_choices,
        "finding_examples": finding_line_examples,
        "sample_start_line": sample_start_line,
        "total_lines": total_lines,
    }


@app.get("/ai-logs", name="ai_logs")
async def ai_logs(
    request: Request,
    module: Optional[str] = None,
    tail_filter: str = "all",
    issue_filter: str = "all",
    sample_source: str = "tail",
    message: Optional[str] = None,
    error: Optional[str] = None,
):
    """Render the unified AI logs explorer that replaces the legacy logs/LLM pages."""
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    modules = _build_modules_from_config()
    stats = get_module_stats(modules)
    ingestion_status = _derive_ingestion_status(modules) if modules else None
    safe_sample_source = _normalize_sample_source(sample_source)
    module_obj = None
    open_findings_count = None
    if modules:
        if module:
            module_obj = next((m for m in modules if m.name == module), None)
        if module_obj is None:
            # Default to first enabled module alphabetically
            enabled_modules = sorted(
                [m for m in modules if m.enabled], key=lambda m: m.name
            )
            module_obj = enabled_modules[0] if enabled_modules else None

    sample_lines, sample_start_line, total_lines, sample_error = _get_sample_lines_for_module(
        module_obj, safe_sample_source, max_lines=500
    )
    log_state = _build_log_view_state(
        module_obj,
        tail_filter,
        issue_filter,
        sample_lines_override=sample_lines,
        sample_start_line=sample_start_line,
        total_lines=total_lines,
    )

    # Count findings displayed in current view (from finding_tail)
    if module_obj and db_status.get("connected"):
        displayed_finding_ids = set()
        for section in log_state.get("finding_tail", []):
            if section.get("unified_view"):
                for line_entry in section.get("lines", []):
                    finding = line_entry.get("finding")
                    if finding:
                        fid = getattr(finding, "id", None)
                        if fid is not None:
                            displayed_finding_ids.add(fid)
            else:
                finding = section.get("finding")
                if finding:
                    fid = getattr(finding, "id", None)
                    if fid is not None:
                        displayed_finding_ids.add(fid)
        open_findings_count = len(displayed_finding_ids)

    provider_name = _select_provider_name(module_obj)
    provider_cfg = llm_defaults.providers.get(provider_name) if provider_name else None
    provider_settings = {
        name: {
            "temperature": p.temperature,
            "top_p": p.top_p,
            "max_output_tokens": p.max_output_tokens,
            "max_excerpt_lines": p.max_excerpt_lines,
            "model": p.model,
        }
        for name, p in llm_defaults.providers.items()
    }

    seed_prompt = None
    module_prompt_template = None
    summary_prompt_template = _load_summary_prompt_template()
    if module_obj is not None and log_state["recent_findings"]:
        max_lines = provider_cfg.max_excerpt_lines if provider_cfg else 20
        seed_prompt = _finding_excerpt_preview(log_state["recent_findings"][0], max_lines)
    if module_obj and getattr(module_obj, "llm", None):
        template_path = getattr(module_obj.llm, "prompt_template_path", None)
        if template_path:
            try:
                module_prompt_template = Path(template_path).read_text()
            except Exception:
                module_prompt_template = None

    regex_presets: List[Dict[str, str]] = []

    return templates.TemplateResponse(
        "ai_logs.html",
        {
            "request": request,
            "username": username,
            "modules": modules,
            "current_module": module_obj,
            **log_state,
            "db_status": db_status,
            "message": message,
            "error": error or sample_error,
            "providers": list(llm_defaults.providers.values()),
            "provider_settings": provider_settings,
            "selected_provider": provider_name,
            "seed_prompt": seed_prompt,
            "module_prompt_template": module_prompt_template,
            "summary_prompt_template": summary_prompt_template,
            "regex_presets": regex_presets,
            "sample_source": safe_sample_source,
            "stats": stats,
            "ingestion_status": ingestion_status,
            "open_findings_count": open_findings_count,
        },
    )


@app.get("/api/log-lines", name="api_log_lines")
async def api_log_lines(
    request: Request,
    module: str,
    offset: int = 0,
    limit: int = 500,
    sample_source: str = "tail",
):
    """API endpoint to load more log lines with pagination.
    
    Returns JSON with lines and pagination info. Used by the "Load more" button.
    """
    username = get_current_user(request, settings)
    if not username:
        return JSONResponse(
            {"error": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED
        )

    modules = _build_modules_from_config()
    module_obj = next((m for m in modules if m.name == module), None)
    if module_obj is None:
        return JSONResponse({"error": "Module not found"}, status_code=404)

    safe_sample_source = _normalize_sample_source(sample_source)
    lines, start_line, total_lines, error = _get_sample_lines_for_module(
        module_obj, safe_sample_source, max_lines=limit, offset=offset
    )

    if error:
        return JSONResponse({"error": error}, status_code=400)

    # Get findings for these lines
    recent_findings = get_recent_findings_for_module(module_obj.name, limit=100)
    finding_tail = _build_finding_tail(
        lines,
        recent_findings,
        first_line_number=start_line,
        module_path=Path(module_obj.path),
    )

    # Convert finding_tail to JSON-serializable format
    lines_data = []
    for section in finding_tail:
        if section.get("unified_view"):
            for line_entry in section.get("lines", []):
                finding = line_entry.get("finding")
                line_data = {
                    "index": line_entry.get("index"),
                    "text": line_entry.get("text", ""),
                    "finding": None,
                }
                if finding:
                    line_data["finding"] = {
                        "id": getattr(finding, "id", None),
                        "finding_index": getattr(finding, "finding_index", None),
                        "severity": str(getattr(finding, "severity", "")),
                        "reason": getattr(finding, "reason", ""),
                        "rule_id": getattr(finding, "rule_id", None),
                        "created_at": _format_local_timestamp(getattr(finding, "created_at", None)),
                        "llm_response_content": getattr(finding, "llm_response_content", None),
                        "llm_provider": getattr(finding, "llm_provider", None),
                        "llm_model": getattr(finding, "llm_model", None),
                    }
                lines_data.append(line_data)

    # Calculate if there are more lines to load
    has_more = start_line > 1

    return JSONResponse({
        "lines": lines_data,
        "start_line": start_line,
        "total_lines": total_lines,
        "has_more": has_more,
        "next_offset": offset + len(lines),
    })


def _select_provider_name(module_obj: Optional[ModuleConfig]) -> Optional[str]:
    if module_obj and getattr(module_obj, "llm", None):
        if module_obj.llm.provider_name:
            return module_obj.llm.provider_name
    if llm_defaults.default_provider:
        return llm_defaults.default_provider
    if llm_defaults.providers:
        return next(iter(llm_defaults.providers.keys()))
    return None


def _finding_excerpt_preview(finding, max_lines: int) -> str:
    excerpt_lines = (getattr(finding, "excerpt", "") or "").splitlines()
    if max_lines > 0:
        excerpt_lines = excerpt_lines[:max_lines]
    return "\n".join(excerpt_lines)


@app.post("/llm/query_finding", name="llm_query_finding")
async def llm_query_finding(
    request: Request,
    finding_id: str = Form(...),
    provider: str = Form(...),
):
    """Query LLM for a specific finding with RAG context."""
    username = get_current_user(request, settings)
    if not username:
        return JSONResponse(
            {"error": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED
        )

    try:
        # Get the finding from database
        finding = get_finding_by_id(finding_id)
        if not finding:
            return JSONResponse({"error": "Finding not found"}, status_code=404)
        
        # Get module name from finding
        module_name = getattr(finding, "module_name", None)
        if not module_name:
            return JSONResponse({"error": "Module name not found for finding"}, status_code=400)
        
        # Get provider configuration
        provider_config = llm_defaults.providers.get(provider)
        if not provider_config:
            return JSONResponse({"error": f"Provider '{provider}' not found"}, status_code=400)
        
        # Create a temporary module config for LLM analysis
        temp_module_llm = ModuleLLMConfig(
            enabled=True,
            provider_name=provider,
            emit_llm_payloads_dir=None,
        )
        
        # Analyze finding with LLM (including RAG context)
        analyze_findings_with_llm(
            [finding], 
            llm_defaults, 
            temp_module_llm,
            rag_client=rag_client,
            module_name=module_name
        )
        
        # Return the LLM response
        if finding.llm_response:
            response_data = {
                "provider": finding.llm_response.provider,
                "model": finding.llm_response.model,
                "content": finding.llm_response.content,
                "usage": {
                    "prompt_tokens": finding.llm_response.prompt_tokens,
                    "completion_tokens": finding.llm_response.completion_tokens,
                },
                "citations": finding.llm_response.citations,
            }
            return JSONResponse(response_data)
        else:
            return JSONResponse({"error": "No LLM response generated"}, status_code=500)
            
    except Exception as e:
        add_notification("error", "LLM query failed", str(e))
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/llm/query", name="llm_query")
async def llm_query(
    request: Request,
    provider: str = Form(...),
    prompt: str = Form(...),
):
    username = get_current_user(request, settings)
    if not username:
        return JSONResponse(
            {"error": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED
        )

    provider_cfg = llm_defaults.providers.get(provider)
    if provider_cfg is None:
        return JSONResponse({"error": f"Unknown provider '{provider}'"}, status_code=400)

    prompt_text = (prompt or "").strip()
    if not prompt_text:
        return JSONResponse({"error": "Prompt cannot be empty."}, status_code=400)

    chat_payload = {
        "model": provider_cfg.model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are a log triage assistant that summarizes log snippets succinctly and "
                    "suggests follow-up actions when appropriate. Respond to the following prompt:\n\n"
                    f"{prompt_text}"
                ),
            }
        ],
        "temperature": provider_cfg.temperature,
        "top_p": provider_cfg.top_p,
        "max_tokens": provider_cfg.max_output_tokens,
    }

    try:
        response_data = _call_chat_completion(provider_cfg, chat_payload)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    message = response_data.get("choices", [{}])[0].get("message", {})
    usage = response_data.get("usage", {}) or {}
    content = message.get("content", "").strip()

    return JSONResponse(
        {
            "provider": provider_cfg.name,
            "model": response_data.get("model", provider_cfg.model),
            "content": content,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            },
        }
    )


@app.post("/logs/db/flush", name="flush_logs_db")
async def flush_logs_db(
    request: Request,
    module: Optional[str] = Form(None),
    tail_filter: str = Form("all"),
    issue_filter: str = Form("all"),
    sample_source: str = Form("tail"),
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not db_status.get("connected"):
        return _logs_redirect(
            module,
            error="Database not connected.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    try:
        deleted = delete_all_findings()
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Failed to flush database: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    msg = "Database already empty." if deleted == 0 else f"Deleted {deleted} stored finding(s)."
    return _logs_redirect(
        module,
        message=msg,
        tail_filter=tail_filter,
        issue_filter=issue_filter,
        sample_source=sample_source,
    )


@app.post("/logs/db/flush-module", name="flush_module_logs")
async def flush_module_logs(
    request: Request,
    module: str = Form(...),
    tail_filter: str = Form("all"),
    issue_filter: str = Form("all"),
    sample_source: str = Form("tail"),
    confirm_token: str = Form(""),
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not module:
        return _logs_redirect(
            module,
            error="Select a module before flushing findings.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    if (confirm_token or "").strip().lower() != "confirm":
        return _logs_redirect(
            module,
            error="Enter 'confirm' to proceed with flushing findings.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    if not db_status.get("connected"):
        return _logs_redirect(
            module,
            error="Database not connected.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    modules = {m.name for m in _build_modules_from_config()}
    if module not in modules:
        return _logs_redirect(
            None,
            error=f"Unknown module '{module}'.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    try:
        deleted = delete_findings_for_module(module)
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Failed to flush findings: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    msg = (
        f"No stored findings for {module}."
        if deleted == 0
        else f"Deleted {deleted} stored finding(s) for {module}."
    )
    return _logs_redirect(
        module,
        message=msg,
        tail_filter=tail_filter,
        issue_filter=issue_filter,
        sample_source=sample_source,
    )


@app.post("/logs/finding/delete-multiple", name="delete_selected_findings")
async def delete_selected_findings(
    request: Request,
    module: Optional[str] = Form(None),
    tail_filter: str = Form("all"),
    issue_filter: str = Form("all"),
    sample_source: str = Form("tail"),
    finding_ids: str = Form(""),
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not db_status.get("connected"):
        return _logs_redirect(
            module,
            error="Database not connected.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    raw_ids = [part.strip() for part in (finding_ids or "").split(",") if part.strip()]
    parsed_ids = []
    for raw in raw_ids:
        try:
            parsed_ids.append(int(raw))
        except ValueError:
            continue

    if not parsed_ids:
        return _logs_redirect(
            module,
            error="Select at least one finding to delete.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    try:
        deleted = delete_findings_by_ids(parsed_ids)
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Failed to delete entries: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    if not deleted:
        return _logs_redirect(
            module,
            error="No matching findings were removed.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    return _logs_redirect(
        module,
        message=f"Deleted {deleted} finding(s).",
        tail_filter=tail_filter,
        issue_filter=issue_filter,
        sample_source=sample_source,
    )


@app.post("/logs/finding/delete", name="delete_finding")
async def delete_finding(
    request: Request,
    finding_id: int = Form(...),
    module: Optional[str] = Form(None),
    tail_filter: str = Form("all"),
    issue_filter: str = Form("all"),
    sample_source: str = Form("tail"),
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not db_status.get("connected"):
        return _logs_redirect(
            module,
            error="Database not connected.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    try:
        deleted = delete_finding_by_id(finding_id)
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Failed to delete entry: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    if not deleted:
        return _logs_redirect(
            module,
            error="Entry not found.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    return _logs_redirect(
        module,
        message="Finding removed.",
        tail_filter=tail_filter,
        issue_filter=issue_filter,
        sample_source=sample_source,
    )


@app.post("/logs/finding/severity", name="change_finding_severity")
async def change_finding_severity(
    request: Request,
    finding_id: int = Form(...),
    severity: str = Form(...),
    module: Optional[str] = Form(None),
    tail_filter: str = Form("all"),
    issue_filter: str = Form("all"),
    sample_source: str = Form("tail"),
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not db_status.get("connected"):
        return _logs_redirect(
            module,
            error="Database not connected.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    normalized = (severity or "").upper()
    if normalized not in SEVERITY_CHOICES:
        return _logs_redirect(
            module,
            error="Invalid severity provided.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    try:
        updated = update_finding_severity(finding_id, normalized)
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Failed to update severity: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    if not updated:
        return _logs_redirect(
            module,
            error="Entry not found.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    return _logs_redirect(
        module,
        message=f"Finding severity updated to {normalized}.",
        tail_filter=tail_filter,
        issue_filter=issue_filter,
        sample_source=sample_source,
    )


@app.post("/logs/finding/manual", name="create_manual_finding")
async def create_manual_finding(
    request: Request,
    module: Optional[str] = Form(None),
    severity: str = Form("ERROR"),
    message: Optional[str] = Form(None),
    lines: str = Form(""),
    line_indexes: str = Form(""),
    tail_filter: str = Form("all"),
    issue_filter: str = Form("all"),
    sample_source: str = Form("tail"),
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not db_status.get("connected"):
        return _logs_redirect(
            module,
            error="Database not connected.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    modules = {m.name: m for m in _build_modules_from_config()}
    module_obj = modules.get(module or "")
    if module_obj is None:
        return _logs_redirect(
            None,
            error="Select a module before creating a finding.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    normalized = (severity or "").upper()
    if normalized not in SEVERITY_CHOICES:
        return _logs_redirect(
            module_obj.name,
            error="Invalid severity provided.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    selected_lines = [ln for ln in (lines.split("\n") if lines else []) if ln.strip()]
    if not selected_lines:
        return _logs_redirect(
            module_obj.name,
            error="Select at least one log line to create a finding.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    try:
        severity_value = Severity.from_string(normalized)
    except Exception:
        severity_value = Severity.WARNING

    message_text = (message or "").strip() or selected_lines[0]

    parsed_indexes = []
    if line_indexes:
        for raw in line_indexes.split(","):
            try:
                parsed_indexes.append(int(raw))
            except ValueError:
                continue
    line_start = min(parsed_indexes) if parsed_indexes else 0
    line_end = max(parsed_indexes) if parsed_indexes else line_start + len(selected_lines) - 1

    finding_index = get_next_finding_index(module_obj.name)
    finding = Finding(
        file_path=Path(module_obj.path),
        pipeline_name=getattr(module_obj, "pipeline_name", None) or "",
        finding_index=finding_index,
        severity=severity_value,
        message=message_text,
        line_start=line_start,
        line_end=line_end,
        rule_id=None,
        excerpt=selected_lines,
        needs_llm=False,
    )

    try:
        store_finding(module_obj.name, finding)
    except Exception as exc:
        return _logs_redirect(
            module_obj.name,
            error=f"Failed to store finding: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    return _logs_redirect(
        module_obj.name,
        message="Manual finding recorded.",
        tail_filter=tail_filter,
        issue_filter=issue_filter,
        sample_source=sample_source,
    )


@app.post("/logs/finding/opinion", name="record_finding_opinion")
async def record_finding_opinion(
    request: Request,
    finding_id: int = Form(...),
    provider: str = Form(""),
    model: str = Form(""),
    content: str = Form(""),
    prompt_tokens: Optional[int] = Form(None),
    completion_tokens: Optional[int] = Form(None),
):
    username = get_current_user(request, settings)
    if not username:
        return JSONResponse({"error": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED)

    if not db_status.get("connected"):
        return JSONResponse({"error": "Database not connected."}, status_code=status.HTTP_400_BAD_REQUEST)

    finding = get_finding_by_id(finding_id)
    if finding is None:
        return JSONResponse({"error": "Finding not found."}, status_code=status.HTTP_404_NOT_FOUND)

    def _to_int(value):
        if value is None:
            return None
        try:
            stripped = str(value).strip()
            return int(stripped) if stripped else None
        except Exception:
            return None

    try:
        ok = update_finding_llm_data(
            finding_id,
            provider=provider or None,
            model=model or None,
            content=content or None,
            prompt_tokens=_to_int(prompt_tokens),
            completion_tokens=_to_int(completion_tokens),
        )
    except Exception as exc:
        return JSONResponse({"error": f"Failed to store AI opinion: {exc}"}, status_code=500)

    if not ok:
        return JSONResponse({"error": "Failed to store AI opinion."}, status_code=400)

    return JSONResponse({"message": "AI opinion stored."})


@app.post("/logs/finding/false_positive", name="mark_false_positive")
async def mark_false_positive(
    request: Request,
    finding_id: int = Form(...),
    module: Optional[str] = Form(None),
    tail_filter: str = Form("all"),
    issue_filter: str = Form("all"),
    sample_source: str = Form("tail"),
    sample_line: Optional[str] = Form(None),
    regex_value: Optional[str] = Form(None),
):
    global raw_config, settings, llm_defaults

    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not db_status.get("connected"):
        return _logs_redirect(
            module,
            error="Database not connected.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    finding = get_finding_by_id(finding_id)
    if finding is None:
        return _logs_redirect(
            module,
            error="Entry not found.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    regex_source = sample_line or getattr(finding, "reason", "") or ""
    regex_value = (regex_value or "").strip() or (
        _suggest_regex_from_line(regex_source) if regex_source else None
    )

    if not regex_value:
        return _logs_redirect(
            module,
            error="No sample available to build an ignore rule.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    regex_issues = _lint_regex_input(regex_value)
    if regex_issues:
        return _logs_redirect(
            module,
            error=" ".join(regex_issues),
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    try:
        cfg_dict = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Failed to read config: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    pipeline_name = getattr(finding, "pipeline_name", None)
    if not pipeline_name:
        modules = _build_modules_from_config()
        mod_obj = next((m for m in modules if m.name == getattr(finding, "module_name", None)), None)
        pipeline_name = getattr(mod_obj, "pipeline_name", None)

    pipelines = cfg_dict.get("pipelines", []) or []
    pipeline_entry = next((p for p in pipelines if p.get("name") == pipeline_name), None)
    if pipeline_entry is None:
        return _logs_redirect(
            module,
            error="Pipeline not found in config; cannot add ignore rule.",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    classifier = pipeline_entry.setdefault("classifier", {})
    ignore_list = classifier.get("ignore_regexes")
    if ignore_list is None or not isinstance(ignore_list, list):
        ignore_list = []
        classifier["ignore_regexes"] = ignore_list

    if regex_value and regex_value not in ignore_list:
        ignore_list.append(regex_value)

    try:
        new_text = yaml.safe_dump(cfg_dict, sort_keys=False)
        tmp_path = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".tmp")
        backup_path = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".bak")
        tmp_path.write_text(new_text, encoding="utf-8")
        if CONFIG_PATH.exists():
            CONFIG_PATH.replace(backup_path)
        tmp_path.replace(CONFIG_PATH)
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Failed to write config: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    from .config import parse_webui_settings  # avoid cycle

    raw_config = load_config(CONFIG_PATH)
    settings = parse_webui_settings(raw_config)
    _refresh_llm_defaults()

    try:
        removed_count = delete_findings_matching_regex(regex_value, pipeline_name=pipeline_name)
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Ignore rule saved, but failed to remove matching findings: {exc}",
            tail_filter=tail_filter,
            issue_filter=issue_filter,
            sample_source=sample_source,
        )

    if not removed_count:
        try:
            deleted = delete_finding_by_id(finding_id)
        except Exception as exc:
            return _logs_redirect(
                module,
                error=f"Ignore rule saved, but failed to remove finding: {exc}",
                tail_filter=tail_filter,
                issue_filter=issue_filter,
                sample_source=sample_source,
            )

        if not deleted:
            return _logs_redirect(
                module,
                error="Ignore rule saved, but finding was not removed.",
                tail_filter=tail_filter,
                issue_filter=issue_filter,
                sample_source=sample_source,
            )

    success_message = (
        "Marked as false positive, removed from findings, and added to ignore rules."
    )
    if removed_count:
        success_message = (
            f"Marked as false positive, removed {removed_count} finding(s) matching the "
            "ignore rule, and added it to ignore rules."
        )

    return _logs_redirect(
        module,
        message=success_message,
        tail_filter=tail_filter,
        issue_filter=issue_filter,
        sample_source=sample_source,
    )


@app.get("/account", name="account")
async def account(request: Request):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    return templates.TemplateResponse(
        "account.html",
        {"request": request, "username": username, "error": None, "message": None},
    )


@app.post("/account/password", name="change_password")
async def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
):
    global settings, raw_config, llm_defaults, rag_client

    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    user = authenticate_user(settings, username, current_password)
    if not user:
        return templates.TemplateResponse(
            "account.html",
            {
                "request": request,
                "username": username,
                "error": "Current password is incorrect.",
                "message": None,
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if new_password != confirm_password:
        return templates.TemplateResponse(
            "account.html",
            {
                "request": request,
                "username": username,
                "error": "New passwords do not match.",
                "message": None,
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if len(new_password) < 8:
        return templates.TemplateResponse(
            "account.html",
            {
                "request": request,
                "username": username,
                "error": "Use at least 8 characters for the new password.",
                "message": None,
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    try:
        cfg_dict = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return templates.TemplateResponse(
            "account.html",
            {
                "request": request,
                "username": username,
                "error": f"Failed to read config: {e}",
                "message": None,
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    webui_cfg = cfg_dict.setdefault("webui", {})
    admins = webui_cfg.get("admin_users") or []
    target = None
    for entry in admins:
        if isinstance(entry, dict) and entry.get("username") == username:
            target = entry
            break

    if target is None:
        return templates.TemplateResponse(
            "account.html",
            {
                "request": request,
                "username": username,
                "error": "Your account is not present in webui.admin_users. Update it via the config editor.",
                "message": None,
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    target["password_hash"] = pwd_context.hash(new_password)

    try:
        new_text = yaml.safe_dump(cfg_dict, sort_keys=False)
        tmp_path = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".tmp")
        backup_path = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".bak")
        tmp_path.write_text(new_text, encoding="utf-8")
        if CONFIG_PATH.exists():
            CONFIG_PATH.replace(backup_path)
        tmp_path.replace(CONFIG_PATH)
    except Exception as e:
        return templates.TemplateResponse(
            "account.html",
            {
                "request": request,
                "username": username,
                "error": f"Failed to write config: {e}",
                "message": None,
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    from .config import parse_webui_settings  # avoid cycle

    raw_config = load_config(CONFIG_PATH)
    settings = parse_webui_settings(raw_config)
    _refresh_llm_defaults()
    _refresh_rag_client()

    return templates.TemplateResponse(
        "account.html",
        {
            "request": request,
            "username": username,
            "error": None,
            "message": "Password updated. Existing sessions stay active until their cookies expire.",
        },
    )


def _tail_lines(
    path: Path, max_lines: int = 500, *, max_chars_per_line: int = 4000, offset: int = 0
) -> tuple[List[str], int, int]:
    """Return the tail of a file and the 1-based line number for its first entry.
    
    Args:
        path: Path to the log file
        max_lines: Maximum number of lines to return (default 500)
        max_chars_per_line: Maximum characters per line before truncation
        offset: Number of lines to skip from the end (for pagination).
                offset=0 returns the last max_lines lines.
                offset=500 returns lines before those last 500 lines.
    
    Returns:
        Tuple of (lines, start_line_number, total_lines_in_file)
        - lines: List of log line strings
        - start_line_number: 1-based line number of the first returned line
        - total_lines_in_file: Total number of lines in the file
    """
    if not path.exists() or not path.is_file():
        return [], 1, 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
            total_lines = len(all_lines)
            
            if total_lines == 0:
                return [], 1, 0
            
            # Calculate the range of lines to return
            # offset=0 means get the last max_lines
            # offset=500 means skip the last 500 and get the max_lines before that
            end_idx = total_lines - offset
            start_idx = max(0, end_idx - max_lines)
            
            if end_idx <= 0:
                # Offset is beyond the file, no lines to return
                return [], 1, total_lines
            
            selected_lines = all_lines[start_idx:end_idx]
            
            trimmed: List[str] = []
            for ln in selected_lines:
                normalized = ln.rstrip("\n")
                trimmed.append(normalized[:max_chars_per_line])

            # 1-based line number for the first returned line
            start_line = start_idx + 1
            return trimmed, start_line, total_lines
    except Exception:
        return [], 1, 0


def _error_lines_from_findings(module_obj, max_lines: int = 200) -> List[str]:
    findings = get_recent_findings_for_module(module_obj.name, limit=max_lines)
    if not findings:
        return []

    lines: List[str] = []
    for finding in findings:
        created_at = _format_local_timestamp(getattr(finding, "created_at", None))
        header = f"[{getattr(finding, 'severity', '')}] finding #{getattr(finding, 'finding_index', '?')} @ {created_at}"
        lines.append(header)
        
        # Get excerpt lines from the finding
        excerpt = getattr(finding, "excerpt", None) or ""
        excerpt_lines = excerpt.splitlines() if isinstance(excerpt, str) else list(excerpt)
        for line_text in excerpt_lines:
            lines.append(line_text)
            if len(lines) >= max_lines:
                return lines[:max_lines]

    return lines[:max_lines]


def _get_sample_lines_for_module(
    module_obj, sample_source: str, max_lines: int = 500, offset: int = 0
) -> tuple[List[str], int, int, Optional[str]]:
    """Get sample lines from a module's log file with pagination support.
    
    Args:
        module_obj: The module configuration object
        sample_source: Source type ('tail', 'errors', or 'sample:...')
        max_lines: Maximum number of lines to return (default 500)
        offset: Number of lines to skip from the end for pagination
    
    Returns:
        Tuple of (lines, start_line_number, total_lines, error_message)
    """
    if module_obj is None:
        return [], 1, 0, None

    source = _normalize_sample_source(sample_source)
    if source == "errors":
        if not db_status.get("connected"):
            return [], 1, 0, "Database not connected; cannot load identified errors."
        lines = _error_lines_from_findings(module_obj, max_lines=max_lines)
        return lines, 1, len(lines), None

    if source.startswith("sample:"):
        sample_logs = _available_sample_logs()
        entry = next((item for item in sample_logs if item.get("value") == source), None)
        if entry is None:
            return [], 1, 0, "Sample log not found on disk."
        lines, start_line, total = _tail_lines(entry["path"], max_lines=max_lines, offset=offset)
        if not lines:
            return [], 1, total, "Sample log is empty or unreadable."
        return lines, start_line, total, None

    lines, start_line, total = _tail_lines(Path(module_obj.path), max_lines=max_lines, offset=offset)
    return lines, start_line, total, None


def _build_finding_tail(
    sample_lines: List[str],
    recent_findings: List,
    *,
    first_line_number: int = 0,
    module_path: Optional[Path] = None,
    context_window: int = 2,
) -> List[Dict[str, Any]]:
    """Build a unified log view with findings mapped to single lines.
    
    Returns a single section containing all log lines, with each line optionally
    associated with a finding. This allows the UI to show findings in context.
    """
    indexed_lines = [
        {"index": idx + first_line_number, "text": line, "finding": None}
        for idx, line in enumerate(sample_lines)
    ]
    
    # Create a lookup by line index for quick access
    line_by_index: Dict[int, Dict[str, Any]] = {
        entry["index"]: entry for entry in indexed_lines
    }

    sorted_findings = sorted(
        recent_findings,
        key=lambda c: getattr(c, "created_at", datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)),
        reverse=True,
    )

    def _load_from_file(start: int, end: int) -> List[Dict[str, Any]]:
        if not module_path or not module_path.exists():
            return []

        loaded: List[Dict[str, Any]] = []
        try:
            with module_path.open("r", encoding="utf-8", errors="ignore") as fh:
                for idx, raw in enumerate(fh, start=1):
                    if idx < start:
                        continue
                    if idx > end:
                        break
                    loaded.append({"index": idx, "text": raw.rstrip("\n")})
        except Exception:
            return []

        return loaded

    def _primary_line_for_finding(finding) -> Optional[Dict[str, Any]]:
        """Get the single primary line that represents this finding.
        
        The line_start attribute is the authoritative source for which line
        matched the regex. The excerpt may contain context lines before/after
        the match, so we should NOT use the first excerpt line as the primary.
        """
        start = getattr(finding, "line_start", None)
        excerpt = getattr(finding, "excerpt", None) or ""
        excerpt_lines = excerpt.splitlines() if isinstance(excerpt, str) else list(excerpt)
        
        # line_start is the authoritative matching line number
        if not isinstance(start, int):
            return None
        
        # Try to find the line in our indexed lines first
        if start in line_by_index:
            return {"index": start, "text": line_by_index[start]["text"]}
        
        # Try to load the actual line from file
        loaded = _load_from_file(start, start)
        if loaded:
            return loaded[0]
        
        # Last resort: if we have an excerpt, try to find the matching line
        # by looking for a line that matches the rule_id pattern
        rule_id = getattr(finding, "rule_id", None)
        if rule_id and excerpt_lines:
            try:
                pattern = re.compile(rule_id)
                for line_text in excerpt_lines:
                    if pattern.search(line_text):
                        return {"index": start, "text": line_text}
            except re.error:
                pass
            # If no pattern match, use the middle line of excerpt as best guess
            # (since context is added before and after the match)
            if excerpt_lines:
                middle_idx = len(excerpt_lines) // 2
                return {"index": start, "text": excerpt_lines[middle_idx]}
        
        return None

    # Map findings to their primary lines
    for finding in sorted_findings:
        primary = _primary_line_for_finding(finding)
        if not primary:
            continue
        
        line_index = primary.get("index")
        if line_index is not None and line_index in line_by_index:
            # Associate finding with this line (only if not already associated)
            if line_by_index[line_index]["finding"] is None:
                line_by_index[line_index]["finding"] = finding
        else:
            # Line not in current view - add it if we have an index
            if line_index is not None:
                new_entry = {
                    "index": line_index,
                    "text": primary.get("text", ""),
                    "finding": finding,
                }
                # Insert in correct position
                inserted = False
                for i, entry in enumerate(indexed_lines):
                    if entry["index"] > line_index:
                        indexed_lines.insert(i, new_entry)
                        line_by_index[line_index] = new_entry
                        inserted = True
                        break
                if not inserted:
                    indexed_lines.append(new_entry)
                    line_by_index[line_index] = new_entry

    # Return a single section with all lines
    return [{"finding": None, "lines": indexed_lines, "unified_view": True}]


def _finding_line_examples(sections: List[Dict[str, Any]]) -> Dict[int, str]:
    """Extract example lines for each finding from the unified view."""
    examples: Dict[int, str] = {}
    for section in sections:
        # Handle unified view where findings are attached to individual lines
        if section.get("unified_view"):
            for line_entry in section.get("lines", []):
                finding = line_entry.get("finding")
                if not finding:
                    continue
                finding_id = getattr(finding, "id", None)
                if finding_id is None or finding_id in examples:
                    continue
                text = line_entry.get("text", "")
                if text:
                    examples[int(finding_id)] = text
        else:
            # Legacy handling for non-unified sections
            finding = section.get("finding")
            if not finding:
                continue
            finding_id = getattr(finding, "id", None)
            if finding_id is None or finding_id in examples:
                continue
            lines = section.get("lines") or []
            texts: List[str] = []
            for entry in lines:
                text = entry.get("text") if isinstance(entry, dict) else None
                if text:
                    texts.append(text)
            if texts:
                examples[int(finding_id)] = "\n".join(texts)
    return examples


def _filter_finding_tail(
    sections: List[Dict[str, Any]],
    tail_filter: str,
    extra_predicate=None,
) -> List[Dict[str, Any]]:
    """Filter sections/lines based on severity filter.
    
    For unified view, this filters the lines within the section rather than
    filtering entire sections. When a severity filter is active, only lines
    with findings matching that severity are shown (not all lines for context).
    """
    if not sections:
        return []

    normalized = (tail_filter or "ALL").upper()
    if normalized in {"", "ALL"} and extra_predicate is None:
        return sections

    filtered: List[Dict[str, Any]] = []
    for section in sections:
        # Handle unified view - filter lines within the section
        if section.get("unified_view"):
            # For unified view, we need to filter the lines based on their findings
            filtered_lines = []
            for line_entry in section.get("lines", []):
                finding = line_entry.get("finding")
                if finding:
                    severity = str(getattr(finding, "severity", "")).upper()
                    severity_match = normalized in {"", "ALL"} or severity == normalized
                    extra_match = extra_predicate(line_entry) if extra_predicate else True
                    if severity_match and extra_match:
                        filtered_lines.append(line_entry)
                elif normalized in {"", "ALL"}:
                    # Only keep non-finding lines when showing all severities
                    filtered_lines.append(line_entry)
                # When filtering by specific severity, skip non-finding lines
            
            # Only add the section if we have any lines that match
            if filtered_lines:
                filtered_section = section.copy()
                filtered_section["lines"] = filtered_lines
                filtered.append(filtered_section)
        else:
            # Legacy handling for non-unified sections
            finding = section.get("finding")
            severity = str(getattr(finding, "severity", "")).upper() if finding else ""
            severity_match = normalized in {"", "ALL"} or severity == normalized
            extra_match = extra_predicate(section) if extra_predicate else True
            if severity_match and extra_match:
                filtered.append(section)

    return filtered


def _suggest_regex_from_line(line: str) -> str:
    # Naive regex suggestion: escape special chars, generalize digits/hex blocks
    escaped = re.escape(line.strip())
    escaped = re.sub(r"\d+", r"\\d+", escaped)
    escaped = re.sub(r"[A-Fa-f0-9]{6,}", r"[A-Fa-f0-9]+", escaped)
    return escaped


@app.post("/regex/test", name="regex_test")
async def regex_test(
    request: Request,
    module: str = Form(...),
    regex_value: str = Form(...),
    regex_kind: str = Form("error"),
    sample_source: str = Form("tail"),
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    modules = _build_modules_from_config()
    safe_sample_source = sample_source if sample_source in {"errors", "tail"} else "tail"
    module_obj = next((m for m in modules if m.name == module), None)
    raw_sample_lines: List[str] = []
    sample_start_line: int = 1
    sample_error: Optional[str] = None
    safe_sample_source = _normalize_sample_source(sample_source)
    if module_obj is not None:
        raw_sample_lines, sample_start_line, _, sample_error = _get_sample_lines_for_module(
            module_obj, safe_sample_source, max_lines=200
        )
    filtered_sample_lines = _filter_finding_intro_lines(raw_sample_lines)
    prepared_lines = _prepare_sample_lines(filtered_sample_lines, first_line_number=sample_start_line)

    regex_issues = _lint_regex_input(regex_value)
    matches: List[int] = []
    compiled, compile_error = (None, None)
    if not regex_issues:
        compiled, compile_error = _compile_regex_with_feedback(regex_value)
        if compile_error:
            regex_issues.append(compile_error)

    error_msg: Optional[str] = None
    if compiled:
        for entry in prepared_lines:
            if compiled.search(entry.get("full", "")):
                matches.append(entry.get("index", 0))
    elif regex_issues:
        error_msg = "Resolve the regex issues before testing."

    wizard = _regex_wizard_metadata("test")
    step_hints = _build_all_regex_hints()
    _update_regex_state(
        request,
        module=module_obj.name if module_obj else None,
        sample_source=safe_sample_source,
        regex_value=regex_value,
        regex_kind=regex_kind,
        matches=matches,
        step="test",
    )
    return templates.TemplateResponse(
        "regex.html",
        _regex_context(
            request,
            username,
            modules,
            module_obj,
            sample_lines=prepared_lines,
            regex_value=regex_value,
            regex_kind=regex_kind,
            matches=matches,
            error=error_msg or sample_error,
            message=None,
            sample_source=safe_sample_source,
            regex_issues=regex_issues,
            wizard=wizard,
            step_hints=step_hints,
            active_step="test",
        ),
    )


@app.post("/regex/suggest", name="regex_suggest")
async def regex_suggest(
    request: Request,
    module: str = Form(...),
    sample_selection: str = Form(""),
    regex_kind: str = Form("error"),
    sample_source: str = Form("tail"),
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    modules = _build_modules_from_config()
    safe_sample_source = sample_source if sample_source in {"errors", "tail"} else "tail"
    module_obj = next((m for m in modules if m.name == module), None)
    raw_sample_lines: List[str] = []
    sample_start_line: int = 1
    sample_error: Optional[str] = None
    safe_sample_source = _normalize_sample_source(sample_source)
    if module_obj is not None:
        raw_sample_lines, sample_start_line, _, sample_error = _get_sample_lines_for_module(
            module_obj, safe_sample_source, max_lines=200
        )

    filtered_sample_lines = _filter_finding_intro_lines(raw_sample_lines)
    prepared_lines = _prepare_sample_lines(filtered_sample_lines, first_line_number=sample_start_line)
    selection = (sample_selection or "").strip()

    if not selection:
        wizard = _regex_wizard_metadata("pick")
        step_hints = _build_all_regex_hints()
        _update_regex_state(
            request,
            module=module_obj.name if module_obj else None,
            sample_source=safe_sample_source,
            regex_value=sample_selection,
            regex_kind=regex_kind,
            matches=[],
            step="pick",
        )

        return templates.TemplateResponse(
            "regex.html",
            _regex_context(
                request,
                username,
                modules,
                module_obj,
                sample_lines=prepared_lines,
                regex_value=sample_selection,
                regex_kind=regex_kind,
                matches=[],
                error=sample_error or "Highlight sample text to draft a regex.",
                message=None,
                sample_source=safe_sample_source,
                wizard=wizard,
                step_hints=step_hints,
                active_step="pick",
            ),
        )

    suggestion = _suggest_regex_from_line(selection)
    wizard = _regex_wizard_metadata("draft")
    step_hints = _build_all_regex_hints()
    _update_regex_state(
        request,
        module=module_obj.name if module_obj else None,
        sample_source=safe_sample_source,
        regex_value=suggestion,
        regex_kind=regex_kind,
        matches=[],
        step="draft",
    )

    return templates.TemplateResponse(
        "regex.html",
        _regex_context(
            request,
            username,
            modules,
            module_obj,
            sample_lines=prepared_lines,
            regex_value=suggestion,
            regex_kind=regex_kind,
            matches=[],
            error=sample_error,
            message="Suggested regex generated from selected text.",
            sample_source=safe_sample_source,
            wizard=wizard,
            step_hints=step_hints,
            active_step="draft",
        ),
    )


@app.post("/regex/save", name="regex_save")
async def regex_save(
    request: Request,
    module: str = Form(...),
    regex_value: str = Form(...),
    regex_kind: str = Form("error"),
    sample_source: str = Form("tail"),
):
    global raw_config, settings, llm_defaults

    safe_sample_source = _normalize_sample_source(sample_source)

    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    modules = _build_modules_from_config()
    module_obj = next((m for m in modules if m.name == module), None)
    if module_obj is None:
        return RedirectResponse(url=app.url_path_for("regex_lab"), status_code=status.HTTP_303_SEE_OTHER)

    tail_lines_result, sample_start_line, _ = _tail_lines(Path(module_obj.path), max_lines=200)
    prepared_lines = _prepare_sample_lines(
        _filter_finding_intro_lines(tail_lines_result),
        first_line_number=sample_start_line
    )
    wizard = _regex_wizard_metadata("save")
    step_hints = _build_all_regex_hints()
    _update_regex_state(
        request,
        module=module_obj.name if module_obj else None,
        sample_source=safe_sample_source,
        regex_value=regex_value,
        regex_kind=regex_kind,
        matches=[],
        step="save",
    )
    regex_issues = _lint_regex_input(regex_value)
    if regex_issues:
        return templates.TemplateResponse(
            "regex.html",
            _regex_context(
                request,
                username,
                modules,
                module_obj,
                sample_lines=prepared_lines,
                regex_value=regex_value,
                regex_kind=regex_kind,
                matches=[],
                error="Resolve the regex issues before saving.",
                message=None,
                sample_source=safe_sample_source,
                regex_issues=regex_issues,
                wizard=wizard,
                step_hints=step_hints,
                active_step="save",
            ),
        )

    if not getattr(module_obj, "pipeline_name", None):
        return templates.TemplateResponse(
            "regex.html",
            _regex_context(
                request,
                username,
                modules,
                module_obj,
                sample_lines=prepared_lines,
                regex_value=regex_value,
                regex_kind=regex_kind,
                matches=[],
                error="Module has no explicit pipeline; cannot save regex automatically.",
                message=None,
                sample_source=safe_sample_source,
                wizard=wizard,
                step_hints=step_hints,
                active_step="save",
            ),
        )

    try:
        cfg_dict = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return templates.TemplateResponse(
            "regex.html",
            _regex_context(
                request,
                username,
                modules,
                module_obj,
                sample_lines=prepared_lines,
                regex_value=regex_value,
                regex_kind=regex_kind,
                matches=[],
                error=f"Failed to read config: {e}",
                message=None,
                sample_source=safe_sample_source,
                wizard=wizard,
                step_hints=step_hints,
                active_step="save",
            ),
        )

    pipelines = cfg_dict.get("pipelines", []) or []
    pipeline_dict = None
    for p in pipelines:
        if p.get("name") == module_obj.pipeline_name:
            pipeline_dict = p
            break

    if pipeline_dict is None:
        return templates.TemplateResponse(
            "regex.html",
            _regex_context(
                request,
                username,
                modules,
                module_obj,
                sample_lines=prepared_lines,
                regex_value=regex_value,
                regex_kind=regex_kind,
                matches=[],
                error=f"Pipeline {module_obj.pipeline_name} not found in config.",
                message=None,
                sample_source=safe_sample_source,
                wizard=wizard,
                step_hints=step_hints,
                active_step="save",
            ),
        )

    classifier = pipeline_dict.setdefault("classifier", {})
    key_map = {
        "error": "error_regexes",
        "warning": "warning_regexes",
        "ignore": "ignore_regexes",
    }
    key = key_map.get(regex_kind, "error_regexes")
    lst = classifier.get(key)
    if lst is None or not isinstance(lst, list):
        lst = []
        classifier[key] = lst

    if regex_value not in lst:
        lst.append(regex_value)

    try:
        new_text = yaml.safe_dump(cfg_dict, sort_keys=False)
        tmp_path = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".tmp")
        backup_path = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".bak")
        tmp_path.write_text(new_text, encoding="utf-8")
        if CONFIG_PATH.exists():
            CONFIG_PATH.replace(backup_path)
        tmp_path.replace(CONFIG_PATH)
    except Exception as e:
        return templates.TemplateResponse(
            "regex.html",
            _regex_context(
                request,
                username,
                modules,
                module_obj,
                sample_lines=prepared_lines,
                regex_value=regex_value,
                regex_kind=regex_kind,
                matches=[],
                error=f"Failed to write config: {e}",
                message=None,
                sample_source=safe_sample_source,
                wizard=wizard,
                step_hints=step_hints,
                active_step="save",
            ),
        )

    from .config import parse_webui_settings  # avoid cycle

    raw_config = load_config(CONFIG_PATH)
    settings = parse_webui_settings(raw_config)
    _refresh_llm_defaults()

    return templates.TemplateResponse(
        "regex.html",
        _regex_context(
            request,
            username,
            modules,
            module_obj,
            sample_lines=prepared_lines,
            regex_value=regex_value,
            regex_kind=regex_kind,
            matches=[],
            error=None,
            message=f"Regex added to classifier.{key} for pipeline {module_obj.pipeline_name}.",
            sample_source=safe_sample_source,
            wizard=wizard,
            step_hints=step_hints,
            active_step="save",
        ),
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Start RAG monitoring when WebUI starts."""
    start_rag_monitor()

@app.on_event("shutdown")
async def shutdown_event():
    """Stop RAG monitoring when WebUI shuts down."""
    stop_rag_monitor()

