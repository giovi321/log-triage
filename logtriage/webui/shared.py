"""Cross-cutting Web UI helpers shared by app.py and the route modules.

Everything here is import-safe (no side effects beyond building the Jinja
environment) and reads live runtime config from ``STATE`` rather than capturing
module globals, so it stays correct across config reloads. app.py imports these
back so its own in-file references keep working during the router split.
"""
from __future__ import annotations

import datetime
import json
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Request
from fastapi.templating import Jinja2Templates

from ..version import __version__
from ..config import build_modules
from ..models import ModuleConfig
from ..notifications import add_notification
from .state import STATE

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
SAMPLE_LOG_DIR = ROOT_DIR / "samples"

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.globals.update({"app_version": __version__})


def _tmpl_is_admin(request) -> bool:
    """Jinja helper: is the current request's user an admin? Used to gate
    admin-only nav links and controls in templates."""
    try:
        from .auth import current_user_is_admin
        return current_user_is_admin(request, STATE.settings)
    except Exception:
        return False


# Available in every template as ``user_is_admin(request)`` (named to avoid
# colliding with per-route context keys like the account page's ``is_admin``).
templates.env.globals.update({"user_is_admin": _tmpl_is_admin})


def format_local_timestamp(value: Optional[datetime.datetime]) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        try:
            ts = datetime.datetime.fromtimestamp(float(value), tz=datetime.timezone.utc)
            return ts.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            return str(value)
    if isinstance(value, str):
        try:
            raw = value.strip()
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            ts = datetime.datetime.fromisoformat(raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=datetime.timezone.utc)
            return ts.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            return value
    ts = value
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=datetime.timezone.utc)
    return ts.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


templates.env.filters["localtime"] = format_local_timestamp


def ensure_csrf_token(request: Request) -> str:
    token = request.session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = token
    return str(token)


def load_context_hints() -> Dict[str, str]:
    """Load config-editor context hints, repairing common JSON escape issues."""
    fallback = {
        "root": "Top-level sections mirror the README. Move the cursor to a section to see details."
    }
    for path in (BASE_DIR / "context_hints.json", ASSETS_DIR / "context_hints.json"):
        try:
            if not path.exists():
                continue
            raw = path.read_text(encoding="utf-8")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = json.loads(raw.replace("\\.", "\\\\."))
            if isinstance(data, dict):
                data.setdefault("root", fallback["root"])
                return data
        except Exception:
            continue
    return fallback


def available_sample_logs() -> List[Dict[str, Any]]:
    if not SAMPLE_LOG_DIR.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for path in sorted(SAMPLE_LOG_DIR.iterdir()):
        if not path.is_file():
            continue
        label = path.stem.replace("_", " ").title()
        entries.append({"value": f"sample:{path.stem}", "label": label, "path": path})
    return entries


def sample_source_options() -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = [
        {"value": "tail", "label": "Log tail (live)"},
        {"value": "errors", "label": "Identified errors"},
    ]
    for entry in available_sample_logs():
        options.append({
            "value": entry.get("value"),
            "label": f"Sample log: {entry.get('label', 'unknown')}",
        })
    return options


def normalize_sample_source(value: str) -> str:
    allowed = {opt.get("value") for opt in sample_source_options()}
    return value if value in allowed else "tail"


def sample_source_label(value: str) -> str:
    for opt in sample_source_options():
        if opt.get("value") == value:
            return opt.get("label", value)
    return "Log tail (live)"


def build_modules_from_config() -> List[ModuleConfig]:
    """Build ModuleConfig objects from the live config in STATE."""
    try:
        return build_modules(STATE.raw_config, STATE.llm_defaults)
    except Exception as exc:
        add_notification("error", "Module configuration error", str(exc))
        return []


def select_provider_name(module_obj: Optional[ModuleConfig]) -> Optional[str]:
    """Resolve the LLM provider for a module: its own, else the global default."""
    llm_defaults = STATE.llm_defaults
    if module_obj and getattr(module_obj, "llm", None) and module_obj.llm.provider_name:
        return module_obj.llm.provider_name
    if llm_defaults and llm_defaults.default_provider:
        return llm_defaults.default_provider
    if llm_defaults and llm_defaults.providers:
        return next(iter(llm_defaults.providers.keys()))
    return None


def render_config_editor(request, username, config_text, *, error=None, message=None, status_code=200):
    """Render the structured config editor, seeding the form from parsed YAML."""
    import json as _json
    try:
        import yaml as _yaml
    except ImportError:
        _yaml = None
    current_hints = load_context_hints()
    parsed_obj = None
    if _yaml is not None:
        try:
            parsed_obj = _yaml.safe_load(config_text)
        except Exception:
            parsed_obj = None
    config_json = _json.dumps(parsed_obj if isinstance(parsed_obj, dict) else {})
    # The Account tab embeds local-user management, so the editor needs the user
    # list (admins only — Settings is already admin-gated).
    user_list = []
    try:
        from .auth import current_user_is_admin
        from . import users as users_mod
        if current_user_is_admin(request, STATE.settings):
            user_list = users_mod.list_users()
    except Exception:
        user_list = []
    return templates.TemplateResponse(
        "config_edit.html",
        {
            "request": request,
            "username": username,
            "config_text": config_text,
            "config_json": config_json,
            "error": error,
            "message": message,
            "context_hints": current_hints,
            "users": user_list,
        },
        status_code=status_code,
    )


def finding_excerpt_preview(finding, max_lines: int) -> str:
    excerpt_lines = (getattr(finding, "excerpt", "") or "").splitlines()
    if max_lines > 0:
        excerpt_lines = excerpt_lines[:max_lines]
    return "\n".join(excerpt_lines)


def suggest_regex_from_line(line: str) -> str:
    """Naive ignore-regex suggestion: escape, then generalize digits/hex runs."""
    import re
    escaped = re.escape(line.strip())
    escaped = re.sub(r"\d+", r"\\d+", escaped)
    escaped = re.sub(r"[A-Fa-f0-9]{6,}", r"[A-Fa-f0-9]+", escaped)
    return escaped
