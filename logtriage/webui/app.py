
import asyncio
import json
import logging
import os
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Any

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

# Import FastAPI and related dependencies
try:
    from fastapi import FastAPI, Request, status
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    try:
        from fastapi.middleware import SessionMiddleware
    except ImportError:
        # Fallback for older FastAPI versions
        from starlette.middleware.sessions import SessionMiddleware
except ImportError as e:
    print(f"FastAPI dependencies are missing. Error: {e}", file=sys.stderr)
    print("Install with: pip install fastapi uvicorn jinja2 python-multipart", file=sys.stderr)
    sys.exit(1)

# Import LogTriage components
from ..models import GlobalLLMConfig
from ..config import build_llm_config, build_rag_config
from ..notifications import add_notification
from ..rag.monitor import RAGServiceMonitor

# Import RAG client (optional import to avoid circular dependencies)
try:
    from ..rag import RAGClient
    from ..rag.service_client import create_rag_client
except ImportError:
    RAGClient = None
    create_rag_client = None
from .config import (
    load_full_config,
    parse_webui_settings,
    WebUISettings,
    get_client_ip,
)
from .events import EventHub
from ..worker import EnrichmentWorker
from .state import STATE
from . import config_io

logger = logging.getLogger(__name__)

# Global RAG monitoring state
rag_monitor_status = {
    "last_check": None,
    "rag_available": False,
    "rag_ready": False,
    "detailed_status": None,
    "check_interval": 10  # seconds
}

_rag_monitor: Optional[RAGServiceMonitor] = None


def _get_webui_rag_service_url() -> Optional[str]:
    try:
        rag_config = build_rag_config(raw_config)
    except Exception:
        return None
    if not rag_config or not getattr(rag_config, "enabled", False):
        return None
    return getattr(rag_config, "service_url", None) or "http://127.0.0.1:8091"


def _webui_create_rag_client(service_url: str):
    if create_rag_client is None:
        raise RuntimeError("RAG client factory not available")
    return create_rag_client(service_url, fallback=False)


def _set_webui_rag_client(client) -> None:
    global rag_client
    rag_client = client
    _sync_state()

from .db import setup_database


app = FastAPI(title="log-triage Web UI")

# Cross-cutting helpers + Jinja env now live in shared.py (read STATE, reload-safe).
# Keep the old private names as aliases so app.py's many in-file references and
# the existing tests that patch them keep working during the router split.
from .shared import (
    ASSETS_DIR,
    ensure_csrf_token as _ensure_csrf_token,
    load_context_hints as _load_context_hints,
    build_modules_from_config as _build_modules_from_config,
)

app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


db_status: Dict[str, Any] = {
    "configured": False,
    "connected": False,
    "error": None,
    "url": None,
}

SEVERITY_CHOICES = ["CRITICAL", "ERROR", "WARNING"]


def _init_database(raw: Dict[str, Any], web_settings: "Optional[WebUISettings]" = None):
    db_cfg = raw.get("database") or {}
    url = db_cfg.get("url")
    db_status.update({"configured": bool(url), "connected": False, "error": None, "url": url})
    if not url:
        return
    try:
        setup_database(url)
        db_status["connected"] = True
        # One-time migration of legacy config users into the DB user table.
        admin_users = getattr(web_settings, "admin_users", None)
        if admin_users:
            try:
                from .users import seed_users_from_config
                seeded = seed_users_from_config(admin_users)
                if seeded:
                    logger.info("Seeded %d local user(s) from config into the database", seeded)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not seed users from config: %s", exc)
    except Exception as exc:
        db_status["error"] = str(exc)


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
    _init_database(raw, web_settings)
    return web_settings, raw, cfg_path


settings, raw_config, CONFIG_PATH = _load_settings_and_config()
llm_defaults: GlobalLLMConfig = build_llm_config(raw_config)
rag_client: Optional[RAGClient] = None
context_hints = _load_context_hints()

# Seed STATE before anything reads it (e.g. shared.build_modules_from_config,
# called from the initial _refresh_rag_client() below). _sync_state() is defined
# later, so do the initial seed inline here.
STATE.settings = settings
STATE.raw_config = raw_config
STATE.config_path = CONFIG_PATH
STATE.llm_defaults = llm_defaults
STATE.rag_client = rag_client
STATE.context_hints = context_hints
STATE.db_status = db_status
STATE.rag_monitor_status = rag_monitor_status

from . import oidc as oidc_mod
try:
    oidc_mod.configure(settings)
except Exception as exc:  # pragma: no cover - defensive
    logger.warning("OIDC configuration failed: %s", exc)

if not getattr(settings, "secret_key", None) or settings.secret_key == "CHANGE_ME":
    logger.warning("WebUI secret_key is not set (or still CHANGE_ME). Please set webui.secret_key in config.yaml to a strong random value.")


def start_rag_monitor():
    """Start the RAG monitoring background thread."""
    global _rag_monitor

    if _rag_monitor is None:
        _rag_monitor = RAGServiceMonitor(
            status=rag_monitor_status,
            get_service_url=_get_webui_rag_service_url,
            create_client=_webui_create_rag_client,
            get_client=lambda: rag_client,
            set_client=_set_webui_rag_client,
            timestamp_mode="iso",
            include_detailed_status=True,
            logger=logger,
        )

    _rag_monitor.start()
    _sync_state()
    logger.info("RAG monitoring thread started")


def stop_rag_monitor():
    """Stop the RAG monitoring background thread."""
    global _rag_monitor

    if _rag_monitor is not None:
        _rag_monitor.stop()
    logger.info("RAG monitoring thread stopped")


from .live import build_live_snapshot as _build_live_snapshot


def get_settings() -> WebUISettings:
    return settings


def _sync_state() -> None:
    """Mirror app.py's module-globals into the shared STATE singleton.

    Transition shim for the router split: app.py still owns the globals (and all
    its in-file reads keep working), but extracted routers read ``STATE`` only.
    Called once at startup and after every reload/mutation so STATE never goes
    stale. STATE itself is never rebound — only its attributes are reassigned.
    """
    STATE.settings = settings
    STATE.raw_config = raw_config
    STATE.config_path = CONFIG_PATH
    STATE.llm_defaults = llm_defaults
    STATE.rag_client = rag_client
    STATE.context_hints = context_hints
    STATE.event_hub = globals().get("event_hub")
    STATE.enrichment_worker = globals().get("enrichment_worker")
    STATE.rag_monitor = globals().get("_rag_monitor")
    STATE.rag_monitor_status = rag_monitor_status
    STATE.db_status = db_status


def _mirror_state_to_globals() -> None:
    """Copy the reloadable fields back from STATE into app.py's module globals.

    The inverse of _sync_state(): after config_io.reload_from_disk() mutates
    STATE, app.py's own (not-yet-extracted) routes still read these globals, so
    keep them in lock-step. Extracted routers read STATE directly and don't need
    this. Goes away once every route is moved out.
    """
    global settings, raw_config, llm_defaults, rag_client
    settings = STATE.settings
    raw_config = STATE.raw_config
    llm_defaults = STATE.llm_defaults
    rag_client = STATE.rag_client


def _reload_from_disk() -> None:
    """Reload config from disk through the config_io service (STATE-only), then
    mirror the result into app.py's globals so in-file routes stay fresh."""
    config_io.reload_from_disk(
        init_database=_init_database,
        configure_oidc=oidc_mod.configure,
    )
    _mirror_state_to_globals()


# Let extracted routers trigger the app-level reload (which mirrors globals)
# without importing app.
STATE.reload_callback = _reload_from_disk


def _refresh_llm_defaults() -> None:
    """Rebuild llm_defaults via the config_io service, then mirror back."""
    global llm_defaults
    config_io.refresh_llm_defaults()
    llm_defaults = STATE.llm_defaults


def _refresh_rag_client() -> None:
    """(Re)initialise the RAG client via the config_io service, then mirror back."""
    global rag_client
    config_io.refresh_rag_client()
    rag_client = STATE.rag_client


# Initialize RAG client after function definition
_refresh_rag_client()


# ---------------------------------------------------------------------------
# Live updates (SSE) + background enrichment worker
# ---------------------------------------------------------------------------
event_hub = EventHub()
enrichment_worker: Optional[EnrichmentWorker] = None
_sync_state()  # capture event_hub now that it exists (initial RAG sync ran earlier)


def _worker_deps():
    """Provide (modules_by_name, llm_defaults, rag_client) to the worker, or None."""
    if not getattr(llm_defaults, "enabled", False):
        return None
    try:
        modules = {m.name: m for m in _build_modules_from_config()}
    except Exception:
        return None
    return (modules, llm_defaults, rag_client)


def _maybe_start_worker() -> None:
    global enrichment_worker
    wcfg = (raw_config.get("worker") or {}) if isinstance(raw_config, dict) else {}
    if not wcfg.get("enabled", True) or not wcfg.get("run_in_webui", True):
        return
    if not getattr(llm_defaults, "enabled", False):
        logger.info("Enrichment worker not started (LLM disabled)")
        return
    if enrichment_worker is not None:
        return
    enrichment_worker = EnrichmentWorker(
        _worker_deps,
        interval=float(wcfg.get("interval_seconds", 60)),
        batch=int(wcfg.get("batch", 25)),
        logger_=logger,
    )
    enrichment_worker.start()
    _sync_state()


async def _events_poll_loop():
    """Poll the DB/RAG for changes and broadcast a snapshot when it changes."""
    loop = asyncio.get_event_loop()
    last_key = None
    while True:
        # Nobody watching → don't poll the DB/RAG at all.
        if event_hub.subscriber_count() == 0:
            try:
                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                break
            continue
        try:
            snap = await loop.run_in_executor(None, _build_live_snapshot)
            key = json.dumps(snap, sort_keys=True, default=str)
            if key != last_key:
                snap["type"] = "snapshot"
                event_hub.publish(snap)
                last_key = key
        except asyncio.CancelledError:
            break
        except Exception:
            pass
        try:
            await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            break


@app.on_event("startup")
async def _on_startup():
    try:
        app.state._events_task = asyncio.get_event_loop().create_task(_events_poll_loop())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not start SSE poller: %s", exc)
    try:
        _maybe_start_worker()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not start enrichment worker: %s", exc)


@app.on_event("shutdown")
async def _on_shutdown():
    task = getattr(app.state, "_events_task", None)
    if task is not None:
        task.cancel()
    global enrichment_worker
    if enrichment_worker is not None:
        enrichment_worker.stop()


@app.middleware("http")
async def ip_allowlist_middleware(request: Request, call_next):
    s = settings
    if s.allowed_ips:
        ip = get_client_ip(request, trusted_proxies=s.trusted_proxies)
        if ip not in s.allowed_ips:
            return HTMLResponse("Access denied", status_code=status.HTTP_403_FORBIDDEN)
    response = await call_next(request)
    return response


@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    if not getattr(settings, "csrf_enabled", True):
        return await call_next(request)

    # Ensure every browser session has a CSRF token available for templates
    _ensure_csrf_token(request)

    if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
        return await call_next(request)

    content_type = (request.headers.get("content-type") or "").lower()
    # Only enforce CSRF on form posts. JSON API requests are intentionally exempt.
    if content_type.startswith("application/x-www-form-urlencoded"):
        body = await request.body()
        try:
            parsed = urllib.parse.parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True)
        except Exception:
            parsed = {}
        provided = (parsed.get("csrf_token") or [None])[0]
        expected = request.session.get("csrf_token")
        if not provided or not expected or not secrets.compare_digest(str(provided), str(expected)):
            return HTMLResponse("CSRF validation failed", status_code=status.HTTP_400_BAD_REQUEST)

    elif content_type.startswith("multipart/form-data"):
        try:
            form = await request.form()
        except Exception:
            form = {}
        provided = form.get("csrf_token") if hasattr(form, "get") else None
        expected = request.session.get("csrf_token")
        if not provided or not expected or not secrets.compare_digest(str(provided), str(expected)):
            return HTMLResponse("CSRF validation failed", status_code=status.HTTP_400_BAD_REQUEST)

    return await call_next(request)


app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, session_cookie=settings.session_cookie_name)

# Extracted route modules (auth/account/SSO). More groups will move here as the
# router split proceeds; each reads STATE/shared, never app.py.
from .routers import auth as auth_router
from .routers import system as system_router
from .routers import issues as issues_router
from .routers import rag_api as rag_api_router
from .routers import llm_api as llm_api_router
from .routers import dashboard as dashboard_router
from .routers import config_editor as config_editor_router
from .routers import regex as regex_router
from .routers import logs as logs_router
app.include_router(auth_router.router)
app.include_router(system_router.router)
app.include_router(issues_router.router)
app.include_router(rag_api_router.router)
app.include_router(llm_api_router.router)
app.include_router(dashboard_router.router)
app.include_router(config_editor_router.router)
app.include_router(regex_router.router)
app.include_router(logs_router.router)


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Start RAG monitoring when WebUI starts."""
    start_rag_monitor()

@app.on_event("shutdown")
async def shutdown_event():
    """Stop RAG monitoring when WebUI shuts down."""
    stop_rag_monitor()

