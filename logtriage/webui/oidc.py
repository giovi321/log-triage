"""OIDC Authorization Code + PKCE login via authlib.

authlib is an optional dependency (the ``[oidc]`` extra). This module imports it
lazily so the Web UI still runs when it is absent — OIDC simply stays disabled.

The IdP owns identity here: a successful login establishes the same session
token as local login, keyed on the username claim. We do NOT create a local
password for OIDC users; local accounts remain a separate break-glass path.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Built lazily by configure(); a tuple (OAuth, remote_app) or None.
_oauth = None
_client = None
_configured_key = None


def oidc_available() -> bool:
    """True when authlib is importable."""
    try:
        import authlib.integrations.starlette_client  # noqa: F401
        return True
    except Exception:
        return False


def _config_key(settings) -> tuple:
    return (
        getattr(settings, "oidc_issuer", None),
        getattr(settings, "oidc_client_id", None),
        getattr(settings, "oidc_scopes", None),
    )


def configure(settings) -> bool:
    """(Re)build the OIDC client from settings. Returns True if usable.

    Safe to call repeatedly (e.g. after a config reload): it rebuilds only when
    the relevant settings changed. Returns False — without raising — when OIDC
    is disabled, misconfigured, or authlib is missing.
    """
    global _oauth, _client, _configured_key

    if not getattr(settings, "oidc_enabled", False):
        _oauth = _client = _configured_key = None
        return False

    issuer = getattr(settings, "oidc_issuer", None)
    client_id = getattr(settings, "oidc_client_id", None)
    client_secret = getattr(settings, "oidc_client_secret", None)
    if not issuer or not client_id:
        logger.warning("OIDC enabled but issuer/client_id missing; OIDC disabled.")
        _oauth = _client = _configured_key = None
        return False

    if not oidc_available():
        logger.warning(
            "OIDC enabled but 'authlib' is not installed. "
            "Install with: pip install '.[oidc]'. OIDC disabled."
        )
        _oauth = _client = _configured_key = None
        return False

    key = _config_key(settings)
    if _client is not None and key == _configured_key:
        return True

    from authlib.integrations.starlette_client import OAuth

    # authlib appends /.well-known/openid-configuration to server_metadata_url.
    metadata_url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    oauth = OAuth()
    oauth.register(
        name="oidc",
        client_id=client_id,
        client_secret=client_secret,
        server_metadata_url=metadata_url,
        client_kwargs={
            "scope": getattr(settings, "oidc_scopes", "openid email profile"),
            "code_challenge_method": "S256",  # PKCE
        },
    )
    _oauth = oauth
    _client = oauth.create_client("oidc")
    _configured_key = key
    logger.info("OIDC client configured for issuer %s", issuer)
    return True


def is_configured() -> bool:
    return _client is not None


def get_client():
    return _client


async def authorize_redirect(request, redirect_uri: str):
    """Kick off the login: redirect the browser to the IdP authorize endpoint."""
    if _client is None:
        raise RuntimeError("OIDC is not configured")
    return await _client.authorize_redirect(request, redirect_uri)


async def fetch_identity(request, settings) -> Optional[str]:
    """Complete the callback and return the username, or None on failure.

    Validates the authorization code → token exchange and the ID token (authlib
    verifies the signature against the IdP JWKS and checks nonce/state), then
    pulls the configured username claim.
    """
    if _client is None:
        raise RuntimeError("OIDC is not configured")
    token = await _client.authorize_access_token(request)
    userinfo = token.get("userinfo") if isinstance(token, dict) else None
    if not userinfo:
        try:
            userinfo = await _client.userinfo(token=token)
        except Exception:
            userinfo = None
    if not userinfo:
        return None

    claim = getattr(settings, "oidc_username_claim", "preferred_username")
    username = userinfo.get(claim) or userinfo.get("email") or userinfo.get("sub")
    return str(username) if username else None
