from __future__ import annotations

import datetime
import os
import re
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
from .db import get_module_stats, setup_database, get_latest_chunk_time, get_recent_chunks_for_module


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


def _load_settings_and_config() -> tuple[WebUISettings, Dict[str, Any], Path]:
    cfg_path_str = os.environ.get("LOGTRIAGE_CONFIG", "config.yaml")
    cfg_path = Path(cfg_path_str).resolve()
    raw = load_full_config(cfg_path)
    web_settings = parse_webui_settings(raw)
    _init_database(raw)
    return web_settings, raw, cfg_path


settings, raw_config, CONFIG_PATH = _load_settings_and_config()
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, session_cookie=settings.session_cookie_name)


def get_settings() -> WebUISettings:
    return settings


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


@app.get("/users", name="user_admin")
async def user_admin(request: Request):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    return templates.TemplateResponse(
        "users.html",
        {
            "request": request,
            "username": username,
            "admin_users": settings.admin_users,
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

    return templates.TemplateResponse(
        "config_edit.html",
        {
            "request": request,
            "username": username,
            "config_text": text,
            "error": None,
            "message": None,
        },
    )


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
        return templates.TemplateResponse(
            "config_edit.html",
            {
                "request": request,
                "username": username,
                "config_text": config_text,
                "error": f"YAML error: {e}",
                "message": None,
            },
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
        return templates.TemplateResponse(
            "config_edit.html",
            {
                "request": request,
                "username": username,
                "config_text": config_text,
                "error": f"Write error: {e}",
                "message": None,
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    from .config import parse_webui_settings  # avoid cycle

    raw_config = load_config(CONFIG_PATH)
    settings = parse_webui_settings(raw_config)
    _init_database(raw_config)

    return templates.TemplateResponse(
        "config_edit.html",
        {
            "request": request,
            "username": username,
            "config_text": config_text,
            "error": None,
            "message": "Configuration saved.",
        },
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


@app.get("/logs", name="module_logs")
async def module_logs(request: Request, module: Optional[str] = None):
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
    if module_obj is not None:
        sample_lines = _tail_lines(Path(module_obj.path), max_lines=400)
        recent_chunks = get_recent_chunks_for_module(module_obj.name, limit=50)
        chunked_tail = _build_chunked_tail(sample_lines, recent_chunks)

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
        },
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


def _error_lines_from_chunks(module_name: str, max_lines: int = 200) -> List[str]:
    chunks = get_recent_chunks_for_module(module_name, limit=max_lines)
    lines: List[str] = []
    for chunk in chunks:
        sev = (chunk.severity or "").upper()
        if sev in {"OK", "INFO"} and not (chunk.error_count or chunk.warning_count):
            continue
        reason = (chunk.reason or "").strip() or "-"
        created_at = _format_local_timestamp(getattr(chunk, "created_at", None))
        chunk_header = (
            f"[{chunk.severity}] chunk #{getattr(chunk, 'chunk_index', '?')} @ {created_at}"
        )
        counts = (
            f"errors: {getattr(chunk, 'error_count', 0)}, warnings: {getattr(chunk, 'warning_count', 0)}, "
            f"lines: {getattr(chunk, 'line_count', 0)}"
        )
        lines.append(f"{chunk_header}\n{counts}\nreason: {reason}")
        if len(lines) >= max_lines:
            break
    return lines[:max_lines]


def _get_sample_lines_for_module(module_obj, sample_source: str, max_lines: int = 200) -> tuple[List[str], Optional[str]]:
    if module_obj is None:
        return [], None

    source = sample_source if sample_source in {"errors", "tail"} else "tail"
    if source == "errors":
        if not db_status.get("connected"):
            return [], "Database not connected; cannot load identified errors."
        return _error_lines_from_chunks(module_obj.name, max_lines=max_lines), None

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

    return list(reversed(sections))


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

