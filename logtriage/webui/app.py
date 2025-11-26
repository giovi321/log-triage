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
from ..models import GlobalLLMConfig, ModuleConfig, Severity, Finding
from ..llm_client import _call_chat_completion
from .config import load_full_config, parse_webui_settings, WebUISettings, get_client_ip
from .auth import authenticate_user, create_session_token, get_current_user, pwd_context
from .db import (
    delete_all_findings,
    delete_findings_for_module,
    delete_findings_by_ids,
    delete_finding_by_id,
    get_finding_by_id,
    get_module_stats,
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
SAMPLE_LOG_DIR = ROOT_DIR / "baseline" / "samples"
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
    raw = load_full_config(cfg_path)
    web_settings = parse_webui_settings(raw)
    _init_database(raw)
    return web_settings, raw, cfg_path


settings, raw_config, CONFIG_PATH = _load_settings_and_config()
llm_defaults: GlobalLLMConfig = build_llm_config(raw_config)
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


def _evaluate_regex_against_lines(regex_value: str, sample_lines: List[str]) -> tuple[list[int], Optional[str]]:
    matches: List[int] = []
    error_msg: Optional[str] = None
    if not regex_value:
        return matches, error_msg

    try:
        pattern = re.compile(regex_value)
        matches = [idx for idx, line in enumerate(sample_lines) if pattern.search(line)]
    except re.error as e:
        error_msg = f"Regex error: {e}"

    return matches, error_msg


def get_settings() -> WebUISettings:
    return settings


def _refresh_llm_defaults() -> None:
    global llm_defaults
    llm_defaults = build_llm_config(raw_config)


def _build_modules_from_config() -> List[ModuleConfig]:
    return build_modules(raw_config, llm_defaults)


INGESTION_STALENESS_MINUTES = int(os.getenv("LOGTRIAGE_INGESTION_STALENESS_MINUTES", "60"))


def _derive_ingestion_status(
    modules: List[ModuleConfig],
    now: Optional[datetime.datetime] = None,
    freshness_minutes: int = INGESTION_STALENESS_MINUTES,
) -> Dict[str, Any]:
    """Compute a traffic-light style indicator for log ingestion freshness.

    The indicator is based on the last modification time of each module's log
    file. If a module's log file is missing or stale beyond the configured
    window (per-module `stale_after_minutes` or the default), it is marked as
    stale.
    """

    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)

    enabled_modules = [m.name for m in modules if m.enabled]
    if not enabled_modules:
        return {
            "state_class": "warn",
            "message": "No enabled modules",
            "stale_modules": [],
            "latest_log_update": None,
        }

    latest_update: Optional[datetime.datetime] = None
    stale_modules: List[str] = []
    freshness_windows: set[int] = set()

    for mod in modules:
        if not mod.enabled:
            continue
        mod_freshness_minutes = getattr(mod, "stale_after_minutes", None) or freshness_minutes
        freshness_windows.add(mod_freshness_minutes)
        freshness_window = datetime.timedelta(minutes=mod_freshness_minutes)
        try:
            mtime = datetime.datetime.fromtimestamp(mod.path.stat().st_mtime, tz=datetime.timezone.utc)
        except FileNotFoundError:
            mtime = None

        if mtime and (latest_update is None or mtime > latest_update):
            latest_update = mtime

        if mtime is None or (now - mtime) > freshness_window:
            stale_modules.append(mod.name)

    if not freshness_windows:
        freshness_windows.add(freshness_minutes)

    window_hint = (
        f"{next(iter(freshness_windows))}m"
        if len(freshness_windows) == 1
        else "their configured windows"
    )

    if latest_update is None:
        return {
            "state_class": "error",
            "message": "No log file activity detected",
            "stale_modules": enabled_modules,
            "latest_log_update": None,
        }

    if not stale_modules:
        state_class = "ok"
        message = f"Logs updating (within {window_hint})"
    elif len(stale_modules) == len(enabled_modules):
        state_class = "error"
        message = f"No recent log updates (> {window_hint})"
    else:
        state_class = "warn"
        message = f"Some modules stale (> {window_hint})"

    return {
        "state_class": state_class,
        "message": message,
        "stale_modules": stale_modules,
        "latest_log_update": latest_update,
    }


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
    sample_error: Optional[str] = None
    safe_sample_source = _normalize_sample_source(sample_source)
    if module_obj is not None:
        raw_sample_lines, sample_error = _get_sample_lines_for_module(
            module_obj, safe_sample_source, max_lines=200
        )

    filtered_sample_lines = _filter_finding_intro_lines(raw_sample_lines)
    prepared_lines = _prepare_sample_lines(filtered_sample_lines)
    matches, evaluation_error = _evaluate_regex_against_lines(
        stored_state.get("regex_value", ""), filtered_sample_lines
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
    module_obj, tail_filter: str, issue_filter: str, *, sample_lines_override: Optional[List[str]] = None
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

        for section in finding_tail:
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
    if modules:
        if module:
            module_obj = next((m for m in modules if m.name == module), None)
        if module_obj is None:
            module_obj = modules[0]

    sample_lines, sample_error = _get_sample_lines_for_module(
        module_obj, safe_sample_source, max_lines=400
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
            "sample_source": safe_sample_source,
            "stats": stats,
            "ingestion_status": ingestion_status,
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

    return _logs_redirect(
        module,
        message="Marked as false positive, removed from findings, and added to ignore rules.",
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


def _tail_lines(path: Path, max_lines: int = 200, *, max_chars_per_line: int = 4000) -> List[str]:
    if not path.exists() or not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            trimmed = []
            for ln in lines[-max_lines:]:
                normalized = ln.rstrip("\n")
                trimmed.append(normalized[:max_chars_per_line])
            return trimmed
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

    source = _normalize_sample_source(sample_source)
    if source == "errors":
        if not db_status.get("connected"):
            return [], "Database not connected; cannot load identified errors."
        return _error_lines_from_findings(module_obj, max_lines=max_lines), None

    if source.startswith("sample:"):
        sample_logs = _available_sample_logs()
        entry = next((item for item in sample_logs if item.get("value") == source), None)
        if entry is None:
            return [], "Sample log not found on disk."
        lines = _tail_lines(entry["path"], max_lines=max_lines)
        if not lines:
            return [], "Sample log is empty or unreadable."
        return lines, None

    return _tail_lines(Path(module_obj.path), max_lines=max_lines), None


def _build_finding_tail(sample_lines: List[str], recent_findings: List) -> List[Dict[str, Any]]:
    indexed_lines = [{"index": idx, "text": line} for idx, line in enumerate(sample_lines)]
    used_indexes: set[int] = set()
    sections: List[Dict[str, Any]] = []

    sorted_findings = sorted(
        recent_findings,
        key=lambda c: getattr(c, "created_at", datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)),
        reverse=True,
    )

    def _lines_for_finding(finding):
        start = getattr(finding, "line_start", None)
        end = getattr(finding, "line_end", None)
        excerpt = getattr(finding, "excerpt", None) or ""
        excerpt_lines = excerpt.splitlines() if isinstance(excerpt, str) else list(excerpt)

        if start is not None and end is not None and start >= 0 and end >= start:
            matching = [entry for entry in indexed_lines if start <= entry.get("index", -1) <= end]
            if matching:
                return matching
            if excerpt_lines:
                return [
                    {"index": start + offset, "text": text}
                    for offset, text in enumerate(excerpt_lines)
                ]

        return [
            {"index": (start + idx) if start is not None else None, "text": text}
            for idx, text in enumerate(excerpt_lines)
        ]

    for finding in sorted_findings:
        lines = _lines_for_finding(finding)
        if not lines:
            continue
        for entry in lines:
            idx = entry.get("index")
            if isinstance(idx, int):
                used_indexes.add(idx)
        sections.append({"finding": finding, "lines": lines})

    if indexed_lines:
        remaining = [ln for ln in indexed_lines if ln.get("index") not in used_indexes]
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
    safe_sample_source = sample_source if sample_source in {"errors", "tail"} else "tail"
    module_obj = next((m for m in modules if m.name == module), None)
    raw_sample_lines: List[str] = []
    sample_error: Optional[str] = None
    safe_sample_source = _normalize_sample_source(sample_source)
    if module_obj is not None:
        raw_sample_lines, sample_error = _get_sample_lines_for_module(
            module_obj, safe_sample_source, max_lines=200
        )
    filtered_sample_lines = _filter_finding_intro_lines(raw_sample_lines)
    prepared_lines = _prepare_sample_lines(filtered_sample_lines)

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
    sample_error: Optional[str] = None
    safe_sample_source = _normalize_sample_source(sample_source)
    if module_obj is not None:
        raw_sample_lines, sample_error = _get_sample_lines_for_module(
            module_obj, safe_sample_source, max_lines=200
        )

    filtered_sample_lines = _filter_finding_intro_lines(raw_sample_lines)
    prepared_lines = _prepare_sample_lines(filtered_sample_lines)
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

    prepared_lines = _prepare_sample_lines(
        _filter_finding_intro_lines(_tail_lines(Path(module_obj.path), max_lines=200))
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

