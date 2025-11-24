from __future__ import annotations

import datetime
import json
import os
import re
import urllib.parse
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml
from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette import status
from fastapi.staticfiles import StaticFiles

from ..config import build_modules, load_config
from .config import load_full_config, parse_webui_settings, WebUISettings, get_client_ip
from .auth import authenticate_user, create_session_token, get_current_user, pwd_context
from .db import (
    delete_all_chunks,
    delete_chunk_by_id,
    get_module_stats,
    setup_database,
    get_latest_chunk_time,
    get_recent_chunks_for_module,
    update_chunk_severity,
    update_chunk_flags,
    get_chunk_by_id,
)


app = FastAPI(title="log-triage Web UI")

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
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

SEVERITY_CHOICES = ["CRITICAL", "ERROR", "WARNING", "INFO", "OK"]


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
    (like unescaped `\.` in regex examples inside JSON strings).
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


def _load_settings_and_config() -> tuple[WebUISettings, Dict[str, Any], Path]:
    cfg_path_str = os.environ.get("LOGTRIAGE_CONFIG", "config.yaml")
    cfg_path = Path(cfg_path_str).resolve()
    raw = load_full_config(cfg_path)
    web_settings = parse_webui_settings(raw)
    _init_database(raw)
    return web_settings, raw, cfg_path


settings, raw_config, CONFIG_PATH = _load_settings_and_config()
context_hints = _load_context_hints()
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, session_cookie=settings.session_cookie_name)


def get_settings() -> WebUISettings:
    return settings


def _render_config_editor(
    request: Request,
    username: str,
    config_text: str,
    *,
    error: Optional[str] = None,
    message: Optional[str] = None,
    status_code: int = status.HTTP_200_OK,
):
    return templates.TemplateResponse(
        "config_edit.html",
        {
            "request": request,
            "username": username,
            "config_text": config_text,
            "error": error,
            "message": message,
            "context_hints": context_hints,
        },
        status_code=status_code,
    )


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

    modules = build_modules(raw_config)
    stats = get_module_stats()
    latest_chunk_at = get_latest_chunk_time()
    page_rendered_at = datetime.datetime.now(datetime.timezone.utc)
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "username": username,
            "modules": modules,
            "stats": stats,
            "db_status": db_status,
            "latest_chunk_at": latest_chunk_at,
            "page_rendered_at": page_rendered_at,
        },
    )


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
    global settings, raw_config

    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    try:
        parsed = yaml.safe_load(config_text) or {}
    except Exception as e:
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

    raw_config = load_config(CONFIG_PATH)
    settings = parse_webui_settings(raw_config)
    _init_database(raw_config)

    return _render_config_editor(
        request,
        username,
        config_text,
        message="Configuration saved.",
    )


@app.post("/config/reload", name="reload_config")
async def reload_config(request: Request):
    global settings, raw_config

    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
    except Exception as e:
        text = f"Error reading {CONFIG_PATH}: {e}"
        return _render_config_editor(request, username, text, error=f"Reload failed: {e}")

    try:
        new_raw = load_config(CONFIG_PATH)
        raw_config = new_raw
        settings = parse_webui_settings(new_raw)
        _init_database(new_raw)
    except Exception as e:
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

    modules = build_modules(raw_config)
    module_obj = None
    if modules:
        if module:
            module_obj = next((m for m in modules if m.name == module), None)
        if module_obj is None:
            module_obj = modules[0]

    sample_lines: List[str] = []
    sample_error: Optional[str] = None
    if module_obj is not None:
        sample_lines, sample_error = _get_sample_lines_for_module(module_obj, sample_source, max_lines=200)

    return templates.TemplateResponse(
        "regex.html",
        {
            "request": request,
            "username": username,
            "modules": modules,
            "current_module": module_obj,
            "sample_lines": sample_lines,
            "regex_value": "",
            "regex_kind": "error",
            "matches": [],
            "error": sample_error,
            "message": None,
            "sample_source": sample_source if sample_source in {"errors", "tail"} else "tail",
        },
    )


def _logs_redirect(
    module: Optional[str],
    message: Optional[str] = None,
    error: Optional[str] = None,
    tail_filter: Optional[str] = None,
):
    params = {}
    if module:
        params["module"] = module
    if tail_filter:
        params["tail_filter"] = tail_filter
    if message:
        params["message"] = message
    if error:
        params["error"] = error

    query = urllib.parse.urlencode(params)
    url = app.url_path_for("module_logs")
    if query:
        url = f"{url}?{query}"
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)


def _add_ignore_rule_for_pipeline(pipeline_name: Optional[str], regex_value: str) -> tuple[bool, Optional[str]]:
    """Append an ignore regex to the given pipeline in the persisted config."""
    global raw_config, settings

    if not pipeline_name:
        return False, "Chunk has no pipeline associated; cannot add ignore rule."
    if not regex_value:
        return False, "Empty regex value provided; nothing to add."

    try:
        cfg_dict = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return False, f"Failed to read config: {exc}"

    pipelines = cfg_dict.get("pipelines", []) or []
    pipeline_dict = next((p for p in pipelines if p.get("name") == pipeline_name), None)
    if pipeline_dict is None:
        return False, f"Pipeline {pipeline_name} not found in config."

    classifier = pipeline_dict.setdefault("classifier", {})
    ignore_list = classifier.get("ignore_regexes")
    if ignore_list is None or not isinstance(ignore_list, list):
        ignore_list = []
        classifier["ignore_regexes"] = ignore_list

    if regex_value not in ignore_list:
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
        return False, f"Failed to write config: {exc}"

    from .config import parse_webui_settings  # avoid cycle

    raw_config = load_config(CONFIG_PATH)
    settings = parse_webui_settings(raw_config)
    return True, None


@app.get("/logs", name="module_logs")
async def module_logs(
    request: Request,
    module: Optional[str] = None,
    tail_filter: str = "all",
    message: Optional[str] = None,
    error: Optional[str] = None,
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    modules = build_modules(raw_config)
    module_obj = None
    if modules:
        if module:
            module_obj = next((m for m in modules if m.name == module), None)
        if module_obj is None:
            module_obj = modules[0]

    sample_lines: List[str] = []
    recent_chunks = []
    chunked_tail: List[Dict[str, Any]] = []
    available_severities: List[str] = []
    tail_filter_normalized = (tail_filter or "all").upper()
    severity_choices = list(SEVERITY_CHOICES)
    had_chunked_tail = False
    if module_obj is not None:
        sample_lines = _tail_lines(Path(module_obj.path), max_lines=400)
        recent_chunks = get_recent_chunks_for_module(module_obj.name, limit=50)
        chunked_tail = _build_chunked_tail(sample_lines, recent_chunks)
        had_chunked_tail = bool(chunked_tail)
        for section in chunked_tail:
            chunk = section.get("chunk")
            if not chunk:
                continue
            sev = str(getattr(chunk, "severity", "")).upper()
            if sev and sev not in available_severities:
                available_severities.append(sev)
            if sev and sev not in severity_choices:
                severity_choices.append(sev)

        chunked_tail = _filter_chunked_tail(chunked_tail, tail_filter_normalized)

    return templates.TemplateResponse(
        "logs.html",
        {
            "request": request,
            "username": username,
            "modules": modules,
            "current_module": module_obj,
            "sample_lines": sample_lines,
            "recent_chunks": recent_chunks,
            "db_status": db_status,
            "chunked_tail": chunked_tail,
            "had_chunked_tail": had_chunked_tail,
            "tail_filter": tail_filter_normalized,
            "tail_severities": available_severities,
            "severity_choices": severity_choices,
            "message": message,
            "error": error,
        },
    )


@app.post("/logs/db/flush", name="flush_logs_db")
async def flush_logs_db(
    request: Request,
    module: Optional[str] = Form(None),
    tail_filter: str = Form("all"),
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not db_status.get("connected"):
        return _logs_redirect(module, error="Database not connected.", tail_filter=tail_filter)

    try:
        deleted = delete_all_chunks()
    except Exception as exc:
        return _logs_redirect(
            module, error=f"Failed to flush database: {exc}", tail_filter=tail_filter
        )

    msg = "Database already empty." if deleted == 0 else f"Deleted {deleted} stored chunk(s)."
    return _logs_redirect(module, message=msg, tail_filter=tail_filter)


@app.post("/logs/chunk/delete", name="delete_chunk")
async def delete_chunk(
    request: Request,
    chunk_id: int = Form(...),
    module: Optional[str] = Form(None),
    tail_filter: str = Form("all"),
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not db_status.get("connected"):
        return _logs_redirect(module, error="Database not connected.", tail_filter=tail_filter)

    try:
        deleted = delete_chunk_by_id(chunk_id)
    except Exception as exc:
        return _logs_redirect(
            module, error=f"Failed to delete entry: {exc}", tail_filter=tail_filter
        )

    if not deleted:
        return _logs_redirect(module, error="Entry not found.", tail_filter=tail_filter)

    return _logs_redirect(module, message="Log entry removed.", tail_filter=tail_filter)


@app.post("/logs/chunk/severity", name="change_chunk_severity")
async def change_chunk_severity(
    request: Request,
    chunk_id: int = Form(...),
    severity: str = Form(...),
    module: Optional[str] = Form(None),
    tail_filter: str = Form("all"),
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not db_status.get("connected"):
        return _logs_redirect(module, error="Database not connected.", tail_filter=tail_filter)

    normalized = (severity or "").upper()
    if normalized not in SEVERITY_CHOICES:
        return _logs_redirect(
            module, error="Invalid severity provided.", tail_filter=tail_filter
        )

    try:
        updated = update_chunk_severity(chunk_id, normalized)
    except Exception as exc:
        return _logs_redirect(
            module, error=f"Failed to update severity: {exc}", tail_filter=tail_filter
        )

    if not updated:
        return _logs_redirect(module, error="Entry not found.", tail_filter=tail_filter)

    return _logs_redirect(
        module, message=f"Severity updated to {normalized}.", tail_filter=tail_filter
    )


@app.post("/logs/chunk/addressed", name="mark_chunk_addressed")
async def mark_chunk_addressed(
    request: Request,
    chunk_id: int = Form(...),
    module: Optional[str] = Form(None),
    tail_filter: str = Form("all"),
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not db_status.get("connected"):
        return _logs_redirect(module, error="Database not connected.", tail_filter=tail_filter)

    try:
        updated = update_chunk_flags(chunk_id, addressed=True)
        if updated:
            update_chunk_severity(chunk_id, "OK")
    except Exception as exc:
        return _logs_redirect(
            module, error=f"Failed to mark as addressed: {exc}", tail_filter=tail_filter
        )

    if not updated:
        return _logs_redirect(module, error="Entry not found.", tail_filter=tail_filter)

    return _logs_redirect(module, message="Chunk marked as addressed.", tail_filter=tail_filter)


@app.post("/logs/chunk/false_positive", name="mark_chunk_false_positive")
async def mark_chunk_false_positive(
    request: Request,
    chunk_id: int = Form(...),
    module: Optional[str] = Form(None),
    tail_filter: str = Form("all"),
    sample_line: str = Form(""),
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if not db_status.get("connected"):
        return _logs_redirect(module, error="Database not connected.", tail_filter=tail_filter)

    try:
        chunk = get_chunk_by_id(chunk_id)
    except Exception as exc:
        return _logs_redirect(
            module, error=f"Failed to load entry: {exc}", tail_filter=tail_filter
        )

    if chunk is None:
        return _logs_redirect(module, error="Entry not found.", tail_filter=tail_filter)

    pattern_source = sample_line or str(getattr(chunk, "reason", ""))
    suggested_regex = _suggest_regex_from_line(pattern_source) if pattern_source else ""

    try:
        updated = update_chunk_flags(chunk_id, false_positive=True, addressed=True)
        if updated:
            update_chunk_severity(chunk_id, "OK")
    except Exception as exc:
        return _logs_redirect(
            module, error=f"Failed to mark false positive: {exc}", tail_filter=tail_filter
        )

    if not updated:
        return _logs_redirect(module, error="Entry not found.", tail_filter=tail_filter)

    if suggested_regex:
        ok, err = _add_ignore_rule_for_pipeline(getattr(chunk, "pipeline_name", None), suggested_regex)
        if not ok:
            return _logs_redirect(
                module,
                error=f"Marked false positive, but failed to update config: {err}",
                tail_filter=tail_filter,
            )

        msg = (
            "Chunk marked as false positive. Added ignore rule to pipeline "
            f"{getattr(chunk, 'pipeline_name', '')}."
        )
    else:
        msg = "Chunk marked as false positive."

    return _logs_redirect(module, message=msg, tail_filter=tail_filter)


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
    global settings, raw_config

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

    return templates.TemplateResponse(
        "account.html",
        {
            "request": request,
            "username": username,
            "error": None,
            "message": "Password updated. Existing sessions stay active until their cookies expire.",
        },
    )


def _tail_lines(path: Path, max_lines: int = 200) -> List[str]:
    if not path.exists() or not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            return [ln.rstrip("\n") for ln in lines[-max_lines:]]
    except Exception:
        return []


def _error_lines_from_chunks(module_obj, max_lines: int = 200) -> List[str]:
    chunks = get_recent_chunks_for_module(module_obj.name, limit=max_lines)
    if not chunks:
        return []

    tail_lines = _tail_lines(Path(module_obj.path), max_lines=max_lines)
    if not tail_lines:
        return []

    chunked_tail = _build_chunked_tail(tail_lines, chunks)

    lines: List[str] = []
    for section in chunked_tail:
        chunk = section.get("chunk")
        if chunk is None:
            continue

        chunk_lines = section.get("lines", [])
        created_at = _format_local_timestamp(getattr(chunk, "created_at", None))
        header = f"[{getattr(chunk, 'severity', '')}] chunk #{getattr(chunk, 'chunk_index', '?')} @ {created_at}"
        lines.append(header)
        for entry in chunk_lines:
            lines.append(entry.get("text", ""))
            if len(lines) >= max_lines:
                return lines[:max_lines]

    return lines[:max_lines]


def _get_sample_lines_for_module(module_obj, sample_source: str, max_lines: int = 200) -> tuple[List[str], Optional[str]]:
    if module_obj is None:
        return [], None

    source = sample_source if sample_source in {"errors", "tail"} else "tail"
    if source == "errors":
        if not db_status.get("connected"):
            return [], "Database not connected; cannot load identified errors."
        return _error_lines_from_chunks(module_obj, max_lines=max_lines), None

    return _tail_lines(Path(module_obj.path), max_lines=max_lines), None


def _build_chunked_tail(sample_lines: List[str], recent_chunks: List) -> List[Dict[str, Any]]:
    if not sample_lines:
        return []

    indexed_lines = [{"index": idx, "text": line} for idx, line in enumerate(sample_lines)]
    remaining = list(indexed_lines)
    sections: List[Dict[str, Any]] = []

    sorted_chunks = sorted(
        recent_chunks,
        key=lambda c: getattr(c, "created_at", datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)),
        reverse=True,
    )

    for chunk in sorted_chunks:
        if not remaining:
            break
        count = int(getattr(chunk, "line_count", 0) or 0)
        if count <= 0:
            continue
        take = min(count, len(remaining))
        chunk_lines = remaining[-take:]
        remaining = remaining[:-take]
        sections.append({"chunk": chunk, "lines": chunk_lines})

    if remaining:
        sections.append({"chunk": None, "lines": remaining})

    return sections


def _filter_chunked_tail(sections: List[Dict[str, Any]], tail_filter: str) -> List[Dict[str, Any]]:
    if not sections:
        return []

    normalized = (tail_filter or "ALL").upper()
    if normalized in {"", "ALL"}:
        return sections

    filtered: List[Dict[str, Any]] = []
    for section in sections:
        chunk = section.get("chunk")
        severity = str(getattr(chunk, "severity", "")).upper() if chunk else ""
        if severity == normalized:
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

    modules = build_modules(raw_config)
    module_obj = next((m for m in modules if m.name == module), None)
    sample_lines: List[str] = []
    sample_error: Optional[str] = None
    if module_obj is not None:
        sample_lines, sample_error = _get_sample_lines_for_module(module_obj, sample_source, max_lines=200)

    error_msg = None
    matches: List[int] = []
    try:
        pattern = re.compile(regex_value)
        for idx, line in enumerate(sample_lines):
            if pattern.search(line):
                matches.append(idx)
    except re.error as e:
        error_msg = f"Regex error: {e}"

    return templates.TemplateResponse(
        "regex.html",
        {
            "request": request,
            "username": username,
            "modules": modules,
            "current_module": module_obj,
            "sample_lines": sample_lines,
            "regex_value": regex_value,
            "regex_kind": regex_kind,
            "matches": matches,
            "error": error_msg or sample_error,
            "message": None,
            "sample_source": sample_source if sample_source in {"errors", "tail"} else "tail",
        },
    )


@app.post("/regex/suggest", name="regex_suggest")
async def regex_suggest(
    request: Request,
    module: str = Form(...),
    sample_line: str = Form(...),
    regex_kind: str = Form("error"),
    sample_source: str = Form("tail"),
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    modules = build_modules(raw_config)
    module_obj = next((m for m in modules if m.name == module), None)
    sample_lines: List[str] = []
    sample_error: Optional[str] = None
    if module_obj is not None:
        sample_lines, sample_error = _get_sample_lines_for_module(module_obj, sample_source, max_lines=200)

    suggestion = _suggest_regex_from_line(sample_line)

    return templates.TemplateResponse(
        "regex.html",
        {
            "request": request,
            "username": username,
            "modules": modules,
            "current_module": module_obj,
            "sample_lines": sample_lines,
            "regex_value": suggestion,
            "regex_kind": regex_kind,
            "matches": [],
            "error": sample_error,
            "message": "Suggested regex generated from selected line.",
            "sample_source": sample_source if sample_source in {"errors", "tail"} else "tail",
        },
    )


@app.post("/regex/save", name="regex_save")
async def regex_save(
    request: Request,
    module: str = Form(...),
    regex_value: str = Form(...),
    regex_kind: str = Form("error"),
    sample_source: str = Form("tail"),
):
    global raw_config, settings

    safe_sample_source = sample_source if sample_source in {"errors", "tail"} else "tail"

    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    modules = build_modules(raw_config)
    module_obj = next((m for m in modules if m.name == module), None)
    if module_obj is None:
        return RedirectResponse(url=app.url_path_for("regex_lab"), status_code=status.HTTP_303_SEE_OTHER)

    if not getattr(module_obj, "pipeline_name", None):
        return templates.TemplateResponse(
            "regex.html",
            {
                "request": request,
                "username": username,
                "modules": modules,
                "current_module": module_obj,
                "sample_lines": _tail_lines(Path(module_obj.path), max_lines=200),
                "regex_value": regex_value,
                "regex_kind": regex_kind,
                "matches": [],
                "error": "Module has no explicit pipeline; cannot save regex automatically.",
                "message": None,
                "sample_source": safe_sample_source,
            },
        )

    try:
        cfg_dict = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return templates.TemplateResponse(
            "regex.html",
            {
                "request": request,
                "username": username,
                "modules": modules,
                "current_module": module_obj,
                "sample_lines": _tail_lines(Path(module_obj.path), max_lines=200),
                "regex_value": regex_value,
                "regex_kind": regex_kind,
                "matches": [],
                "error": f"Failed to read config: {e}",
                "message": None,
                "sample_source": safe_sample_source,
            },
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
            {
                "request": request,
                "username": username,
                "modules": modules,
                "current_module": module_obj,
                "sample_lines": _tail_lines(Path(module_obj.path), max_lines=200),
                "regex_value": regex_value,
                "regex_kind": regex_kind,
                "matches": [],
                "error": f"Pipeline {module_obj.pipeline_name} not found in config.",
                "message": None,
                "sample_source": safe_sample_source,
            },
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
            {
                "request": request,
                "username": username,
                "modules": modules,
                "current_module": module_obj,
                "sample_lines": _tail_lines(Path(module_obj.path), max_lines=200),
                "regex_value": regex_value,
                "regex_kind": regex_kind,
                "matches": [],
                "error": f"Failed to write config: {e}",
                "message": None,
                "sample_source": safe_sample_source,
            },
        )

    from .config import parse_webui_settings  # avoid cycle

    raw_config = load_config(CONFIG_PATH)
    settings = parse_webui_settings(raw_config)

    sample_lines = _tail_lines(Path(module_obj.path), max_lines=200)

    return templates.TemplateResponse(
        "regex.html",
        {
            "request": request,
            "username": username,
            "modules": modules,
            "current_module": module_obj,
            "sample_lines": sample_lines,
            "regex_value": regex_value,
            "regex_kind": regex_kind,
            "matches": [],
            "error": None,
            "message": f"Regex added to classifier.{key} for pipeline {module_obj.pipeline_name}.",
            "sample_source": safe_sample_source,
        },
    )

