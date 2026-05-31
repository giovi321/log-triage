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
    # Users defined in the legacy ``webui.admin_users`` config list are admins by
    # definition. This flag lets the config-fallback login path (used when the
    # user has no row in the DB user table yet) grant admin, matching the
    # pre-DB behaviour instead of silently demoting them.
    is_admin: bool = True


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
    trusted_proxies: List[str]
    session_max_age_hours: int
    # Default minutes after which a module's log file is considered stale on the
    # dashboard (per-module `stale_after_minutes` overrides this).
    staleness_minutes: int = 60
    # Prometheus /metrics endpoint (still subject to allowed_ips). Default on.
    metrics_enabled: bool = True
    # Authentik (or any reverse proxy) forward-auth: trust an identity header,
    # but ONLY when the direct peer is one of trusted_proxies.
    forward_auth_enabled: bool = False
    forward_auth_header: str = "X-authentik-username"
    forward_auth_logout_url: Optional[str] = None
    # OIDC Authorization Code + PKCE (via authlib). When enabled, the login page
    # offers "Sign in with SSO"; local password login stays available as a
    # break-glass path unless oidc_exclusive is set.
    oidc_enabled: bool = False
    oidc_issuer: Optional[str] = None            # discovery base (…/.well-known/openid-configuration)
    oidc_client_id: Optional[str] = None
    oidc_client_secret: Optional[str] = None
    oidc_scopes: str = "openid email profile"
    oidc_username_claim: str = "preferred_username"
    oidc_logout_url: Optional[str] = None
    oidc_exclusive: bool = False                 # hide local password login when True
    # Group-based admin: the token claim carrying the user's groups, and the set
    # of group names that grant admin. Empty admin_groups → no OIDC user is admin.
    oidc_groups_claim: str = "groups"
    oidc_admin_groups: List[str] = None          # set in __post_init__ to []

    def __post_init__(self):
        if self.oidc_admin_groups is None:
            self.oidc_admin_groups = []


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

    metrics = web.get("metrics", {}) or {}
    fwd = web.get("forward_auth", {}) or {}
    oidc = web.get("oidc", {}) or {}

    # Default staleness window: webui.staleness_minutes, falling back to the
    # LOGTRIAGE_INGESTION_STALENESS_MINUTES env var, then 60.
    import os
    default_staleness = int(os.getenv("LOGTRIAGE_INGESTION_STALENESS_MINUTES", "60"))
    try:
        staleness_minutes = int(web.get("staleness_minutes", default_staleness))
    except (TypeError, ValueError):
        staleness_minutes = default_staleness
    if staleness_minutes <= 0:
        staleness_minutes = default_staleness

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
        trusted_proxies=[str(ip) for ip in (web.get("trusted_proxies") or [])],
        session_max_age_hours=int(web.get("session_max_age_hours", 24)),
        staleness_minutes=staleness_minutes,
        metrics_enabled=bool(metrics.get("enabled", True)),
        forward_auth_enabled=bool(fwd.get("enabled", False)),
        forward_auth_header=str(fwd.get("username_header", "X-authentik-username")),
        forward_auth_logout_url=(str(fwd["logout_url"]) if fwd.get("logout_url") else None),
        oidc_enabled=bool(oidc.get("enabled", False)),
        oidc_issuer=(str(oidc["issuer"]) if oidc.get("issuer") else None),
        oidc_client_id=(str(oidc["client_id"]) if oidc.get("client_id") else None),
        oidc_client_secret=(str(oidc["client_secret"]) if oidc.get("client_secret") else None),
        oidc_scopes=str(oidc.get("scopes", "openid email profile")),
        oidc_username_claim=str(oidc.get("username_claim", "preferred_username")),
        oidc_logout_url=(str(oidc["logout_url"]) if oidc.get("logout_url") else None),
        oidc_exclusive=bool(oidc.get("exclusive", False)),
        oidc_groups_claim=str(oidc.get("groups_claim", "groups")),
        oidc_admin_groups=[str(g).strip() for g in (oidc.get("admin_groups") or []) if str(g).strip()],
    )


def get_client_ip(request: Request, trusted_proxies: Optional[List[str]] = None) -> str:
    client_host = request.client.host if request.client else "unknown"
    if trusted_proxies and client_host in trusted_proxies:
        # Only trust X-Forwarded-For when the direct peer is a known proxy
        forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
        if forwarded_for:
            # The leftmost IP is the original client
            return forwarded_for.split(",")[0].strip()
    return client_host
