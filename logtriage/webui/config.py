from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import Request

from ..config import load_config


@dataclass
class WebUser:
    username: str
    password_hash: str  # bcrypt or similar


@dataclass
class WebUISettings:
    enabled: bool
    host: str
    port: int
    base_path: str
    secret_key: str
    session_cookie_name: str
    dark_mode_default: bool
    csrf_enabled: bool
    allowed_ips: List[str]
    admin_users: List[WebUser]


def load_full_config(config_path: Path) -> Dict[str, Any]:
    return load_config(config_path)


def parse_webui_settings(raw: Dict[str, Any]) -> WebUISettings:
    web = raw.get("webui", {}) or {}

    admins_raw = web.get("admin_users", []) or []
    admins: List[WebUser] = []
    for item in admins_raw:
        if not item:
            continue
        username = str(item.get("username", "")).strip()
        pw_hash = str(item.get("password_hash", "")).strip()
        if username and pw_hash:
            admins.append(WebUser(username=username, password_hash=pw_hash))

    return WebUISettings(
        enabled=bool(web.get("enabled", False)),
        host=str(web.get("host", "127.0.0.1")),
        port=int(web.get("port", 8090)),
        base_path=str(web.get("base_path", "/")) or "/",
        secret_key=str(web.get("secret_key", "CHANGE_ME")),
        session_cookie_name=str(web.get("session_cookie_name", "logtriage_session")),
        dark_mode_default=bool(web.get("dark_mode_default", True)),
        csrf_enabled=bool(web.get("csrf_enabled", True)),
        allowed_ips=[str(ip) for ip in (web.get("allowed_ips") or [])],
        admin_users=admins,
    )


def get_client_ip(request: Request) -> str:
    # Basic IP extraction; you can refine this depending on your proxy setup.
    client_host = request.client.host if request.client else "unknown"
    return client_host
