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
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette import status
from fastapi.staticfiles import StaticFiles

from ..config import build_llm_config, build_modules, load_config
from ..models import GlobalLLMConfig, ModuleConfig
from ..llm_client import _call_chat_completion
from .config import load_full_config, parse_webui_settings, WebUISettings, get_client_ip
from .auth import authenticate_user, create_session_token, get_current_user, pwd_context
from .db import (
    delete_all_findings,
    delete_finding_by_id,
    get_finding_by_id,
    get_module_stats,
    setup_database,
    get_latest_finding_time,
    get_recent_findings_for_module,
    update_finding_severity,
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
llm_defaults: GlobalLLMConfig = build_llm_config(raw_config)
context_hints = _load_context_hints()
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, session_cookie=settings.session_cookie_name)


def get_settings() -> WebUISettings:
    return settings


def _refresh_llm_defaults() -> None:
    global llm_defaults
    llm_defaults = build_llm_config(raw_config)


def _build_modules_from_config() -> List[ModuleConfig]:
    return build_modules(raw_config, llm_defaults)


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


def _collect_baseline_files(modules: List[ModuleConfig]) -> List[Dict[str, Any]]:
    files: Dict[str, Dict[str, Any]] = {}
    for module in modules:
        baseline_cfg = getattr(module, "baseline", None)
        if not baseline_cfg:
            continue
        path = baseline_cfg.state_file
        path_str = str(path)
        if path_str not in files:
            files[path_str] = {"path": path, "path_str": path_str, "modules": []}
        files[path_str]["modules"].append(module.name)
    return list(files.values())


def _render_severity_editor(
    request: Request,
    username: str,
    *,
    baseline_files: List[Dict[str, Any]],
    selected_path: Optional[str],
    file_text: str,
    selected_modules: List[str],
    error: Optional[str] = None,
    message: Optional[str] = None,
    status_code: int = status.HTTP_200_OK,
):
    return templates.TemplateResponse(
        "severity_files.html",
        {
            "request": request,
            "username": username,
            "baseline_files": baseline_files,
            "selected_path": selected_path,
            "file_text": file_text,
            "error": error,
            "message": message,
            "selected_modules": selected_modules,
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

    modules = _build_modules_from_config()
    stats = get_module_stats()
    latest_finding_at = get_latest_finding_time()
    page_rendered_at = datetime.datetime.now(datetime.timezone.utc)
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "username": username,
            "modules": modules,
            "stats": stats,
            "db_status": db_status,
            "latest_finding_at": latest_finding_at,
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
    global settings, raw_config, llm_defaults

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
    _refresh_llm_defaults()

    return _render_config_editor(
        request,
        username,
        config_text,
        message="Configuration saved.",
    )


@app.post("/config/reload", name="reload_config")
async def reload_config(request: Request):
    global settings, raw_config, llm_defaults

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
        _refresh_llm_defaults()
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


@app.get("/severity-files", name="severity_files")
async def severity_files(request: Request, path: Optional[str] = None):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    try:
        modules = _build_modules_from_config()
    except Exception as exc:
        return _render_severity_editor(
            request,
            username,
            baseline_files=[],
            selected_path=None,
            file_text="",
            selected_modules=[],
            error=f"Failed to load modules from config: {exc}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    baseline_files = _collect_baseline_files(modules)
    if not baseline_files:
        return _render_severity_editor(
            request,
            username,
            baseline_files=[],
            selected_path=None,
            file_text="",
            selected_modules=[],
            message="No modules define a baseline.state_file. Add one in the config to manage baseline severity files here.",
        )

    selected_entry = next((f for f in baseline_files if f["path_str"] == path), baseline_files[0])
    file_path: Path = selected_entry["path"]
    message: Optional[str] = None
    error: Optional[str] = None

    try:
        raw_text = file_path.read_text(encoding="utf-8")
        try:
            parsed = json.loads(raw_text)
            file_text = json.dumps(parsed, indent=2)
        except Exception:
            file_text = raw_text
            error = "File contains invalid JSON; fix the content before saving."
    except FileNotFoundError:
        file_text = json.dumps({"history": []}, indent=2)
        message = f"{file_path} not found. Save to create it."
    except Exception as exc:
        file_text = ""
        error = f"Failed to read {file_path}: {exc}"

    return _render_severity_editor(
        request,
        username,
        baseline_files=baseline_files,
        selected_path=str(file_path),
        file_text=file_text,
        selected_modules=selected_entry.get("modules", []),
        error=error,
        message=message,
    )


@app.post("/severity-files", name="severity_files_post")
async def severity_files_post(
    request: Request,
    path: str = Form(...),
    file_text: str = Form(...),
):
    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    try:
        modules = _build_modules_from_config()
    except Exception as exc:
        return _render_severity_editor(
            request,
            username,
            baseline_files=[],
            selected_path=None,
            file_text=file_text,
            selected_modules=[],
            error=f"Failed to load modules from config: {exc}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    baseline_files = _collect_baseline_files(modules)
    selected_entry = next((f for f in baseline_files if f["path_str"] == path), None)
    if selected_entry is None:
        return _render_severity_editor(
            request,
            username,
            baseline_files=baseline_files,
            selected_path=None,
            file_text=file_text,
            selected_modules=[],
            error="Selected file is not part of configured baselines.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    target_path: Path = selected_entry["path"]

    try:
        parsed = json.loads(file_text)
    except Exception as exc:
        return _render_severity_editor(
            request,
            username,
            baseline_files=baseline_files,
            selected_path=str(target_path),
            file_text=file_text,
            selected_modules=selected_entry.get("modules", []),
            error=f"JSON error: {exc}",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if not isinstance(parsed, dict) or parsed is None:
        return _render_severity_editor(
            request,
            username,
            baseline_files=baseline_files,
            selected_path=str(target_path),
            file_text=file_text,
            selected_modules=selected_entry.get("modules", []),
            error="Top-level JSON must be an object (e.g., {\"history\": []}).",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    history = parsed.get("history")
    if history is not None and not isinstance(history, list):
        return _render_severity_editor(
            request,
            username,
            baseline_files=baseline_files,
            selected_path=str(target_path),
            file_text=file_text,
            selected_modules=selected_entry.get("modules", []),
            error="The 'history' key must be a list if present.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    normalized_text = json.dumps(parsed, indent=2)

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        tmp_path.write_text(normalized_text, encoding="utf-8")
        tmp_path.replace(target_path)
    except Exception as exc:
        return _render_severity_editor(
            request,
            username,
            baseline_files=baseline_files,
            selected_path=str(target_path),
            file_text=file_text,
            selected_modules=selected_entry.get("modules", []),
            error=f"Failed to write {target_path}: {exc}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return _render_severity_editor(
        request,
        username,
        baseline_files=baseline_files,
        selected_path=str(target_path),
        file_text=normalized_text,
        selected_modules=selected_entry.get("modules", []),
        message=f"Saved {target_path}.",
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

    modules = _build_modules_from_config()
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
    module_obj, tail_filter: str, issue_filter: str, *, sample_lines_override: Optional[List[str]] = None
):
    sample_lines: List[str] = []
    recent_findings = []
    finding_tail: List[Dict[str, Any]] = []
    available_severities: List[str] = []
    tail_filter_normalized = (tail_filter or "all").upper()
    issue_filter_normalized = (issue_filter or "all").upper()
    severity_choices = [sev for sev in SEVERITY_CHOICES if sev != "OK"]
    had_finding_tail = False
    had_recent_findings = False
    if module_obj is not None:
        sample_lines = (
            list(sample_lines_override)
            if sample_lines_override is not None
            else _tail_lines(Path(module_obj.path), max_lines=400)
        )
        recent_findings = get_recent_findings_for_module(module_obj.name, limit=50)
        had_recent_findings = bool(recent_findings)
        finding_tail = _build_finding_tail(sample_lines, recent_findings)
        finding_line_examples = _finding_line_examples(finding_tail)
        had_finding_tail = bool(finding_tail)
        if issue_filter_normalized == "ADDRESSED":
            recent_findings = [
                c for c in recent_findings if str(getattr(c, "severity", "")).upper() == "OK"
            ]
            finding_tail = _filter_finding_tail(
                finding_tail,
                tail_filter="ALL",
                extra_predicate=lambda section: str(
                    getattr(section.get("finding"), "severity", "")
                ).upper()
                == "OK",
            )
        elif issue_filter_normalized == "ACTIVE":
            recent_findings = [
                c for c in recent_findings if str(getattr(c, "severity", "")).upper() != "OK"
            ]
            finding_tail = _filter_finding_tail(
                finding_tail,
                tail_filter="ALL",
                extra_predicate=lambda section: str(
                    getattr(section.get("finding"), "severity", "")
                ).upper()
                != "OK",
            )
        for section in finding_tail:
            finding = section.get("finding")
            if not finding:
                continue
            sev = str(getattr(finding, "severity", "")).upper()
            if sev and sev not in available_severities:
                available_severities.append(sev)
            if sev and sev != "OK" and sev not in severity_choices:
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
    module_obj = None
    if modules:
        if module:
            module_obj = next((m for m in modules if m.name == module), None)
        if module_obj is None:
            module_obj = modules[0]

    sample_lines, sample_error = _get_sample_lines_for_module(
        module_obj, sample_source, max_lines=400
    )
    log_state = _build_log_view_state(
        module_obj, tail_filter, issue_filter, sample_lines_override=sample_lines
    )

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
            "sample_source": sample_source if sample_source in {"errors", "tail"} else "tail",
        },
    )


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


@app.post("/logs/finding/address", name="mark_finding_addressed")
async def mark_finding_addressed(
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
        updated = update_finding_severity(finding_id, "OK")
    except Exception as exc:
        return _logs_redirect(
            module,
            error=f"Failed to mark as addressed: {exc}",
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
        message="Finding marked as addressed.",
        tail_filter=tail_filter,
        issue_filter=issue_filter,
        sample_source=sample_source,
    )


@app.post("/logs/finding/false_positive", name="mark_false_positive")
async def mark_false_positive(
    request: Request,
    finding_id: int = Form(...),
    module: Optional[str] = Form(None),
    tail_filter: str = Form("all"),
    issue_filter: str = Form("all"),
    sample_source: str = Form("tail"),
    sample_line: Optional[str] = Form(None),
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
    regex_value = _suggest_regex_from_line(regex_source) if regex_source else None

    if not regex_value:
        return _logs_redirect(
            module,
            error="No sample available to build an ignore rule.",
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
        update_finding_severity(finding_id, "OK")
    except Exception:
        pass

    return _logs_redirect(
        module,
        message="Marked as false positive and added to ignore rules.",
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
    global settings, raw_config, llm_defaults

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


def _error_lines_from_findings(module_obj, max_lines: int = 200) -> List[str]:
    findings = get_recent_findings_for_module(module_obj.name, limit=max_lines)
    if not findings:
        return []

    tail_lines = _tail_lines(Path(module_obj.path), max_lines=max_lines)
    if not tail_lines:
        return []

    finding_tail = _build_finding_tail(tail_lines, findings)

    lines: List[str] = []
    for section in finding_tail:
        finding = section.get("finding")
        if finding is None:
            continue

        finding_lines = section.get("lines", [])
        created_at = _format_local_timestamp(getattr(finding, "created_at", None))
        header = f"[{getattr(finding, 'severity', '')}] finding #{getattr(finding, 'finding_index', '?')} @ {created_at}"
        lines.append(header)
        for entry in finding_lines:
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
        return _error_lines_from_findings(module_obj, max_lines=max_lines), None

    return _tail_lines(Path(module_obj.path), max_lines=max_lines), None


def _build_finding_tail(sample_lines: List[str], recent_findings: List) -> List[Dict[str, Any]]:
    if not sample_lines:
        return []

    indexed_lines = [{"index": idx, "text": line} for idx, line in enumerate(sample_lines)]
    remaining = list(indexed_lines)
    sections: List[Dict[str, Any]] = []

    sorted_findings = sorted(
        recent_findings,
        key=lambda c: getattr(c, "created_at", datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)),
        reverse=True,
    )

    for finding in sorted_findings:
        if not remaining:
            break
        count = int(getattr(finding, "line_count", 0) or 0)
        if count <= 0:
            continue
        take = min(count, len(remaining))
        finding_lines = remaining[-take:]
        remaining = remaining[:-take]
        sections.append({"finding": finding, "lines": finding_lines})

    if remaining:
        sections.append({"finding": None, "lines": remaining})

    return sections


def _finding_line_examples(sections: List[Dict[str, Any]]) -> Dict[int, str]:
    examples: Dict[int, str] = {}
    for section in sections:
        finding = section.get("finding")
        if not finding:
            continue
        finding_id = getattr(finding, "id", None)
        if finding_id is None or finding_id in examples:
            continue
        lines = section.get("lines") or []
        for entry in lines:
            text = entry.get("text") if isinstance(entry, dict) else None
            if text:
                examples[int(finding_id)] = text
                break
    return examples


def _filter_finding_tail(
    sections: List[Dict[str, Any]],
    tail_filter: str,
    extra_predicate=None,
) -> List[Dict[str, Any]]:
    if not sections:
        return []

    normalized = (tail_filter or "ALL").upper()
    if normalized in {"", "ALL"} and extra_predicate is None:
        return sections

    filtered: List[Dict[str, Any]] = []
    for section in sections:
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

    modules = _build_modules_from_config()
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
    global raw_config, settings, llm_defaults

    safe_sample_source = sample_source if sample_source in {"errors", "tail"} else "tail"

    username = get_current_user(request, settings)
    if not username:
        return RedirectResponse(url=app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    modules = _build_modules_from_config()
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
    _refresh_llm_defaults()

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

