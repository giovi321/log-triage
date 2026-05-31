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

    # Strip stray whitespace/newlines that creep in when client_id/issuer are
    # pasted from the IdP console — a trailing space makes the IdP reject the
    # client_id ("missing or invalid") for a value that looks correct.
    issuer = (getattr(settings, "oidc_issuer", None) or "").strip() or None
    client_id = (getattr(settings, "oidc_client_id", None) or "").strip() or None
    client_secret = getattr(settings, "oidc_client_secret", None)
    if isinstance(client_secret, str):
        client_secret = client_secret.strip()
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
    # Log the client_id (not secret) so a mismatch with the IdP is diagnosable
    # from the service logs without guesswork.
    logger.info("OIDC client configured for issuer %s (client_id=%s)", issuer, client_id)
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


def _is_admin_from_groups(userinfo, settings) -> bool:
    """True iff the user's groups claim intersects the configured admin groups.

    Returns False when no admin groups are configured, the claim is absent, or
    the claim isn't a list/space-delimited string of group names.
    """
    admin_groups = set(getattr(settings, "oidc_admin_groups", None) or [])
    if not admin_groups:
        return False
    claim = getattr(settings, "oidc_groups_claim", "groups")
    raw = userinfo.get(claim)
    if isinstance(raw, str):
        groups = set(raw.split())
    elif isinstance(raw, (list, tuple)):
        groups = {str(g) for g in raw}
    else:
        return False
    return bool(groups & admin_groups)


async def fetch_identity(request, settings):
    """Complete the callback and return ``(username, is_admin)`` or ``(None, False)``.

    Validates the authorization code → token exchange and the ID token (authlib
    verifies the signature against the IdP JWKS and checks nonce/state), then
    pulls the configured username claim and resolves admin from the groups claim.

    Logs at each decision point so a failed SSO login is diagnosable from the
    service logs (which claims the IdP returned, which username claim was sought)
    without exposing secrets.
    """
    if _client is None:
        raise RuntimeError("OIDC is not configured")
    token = await _client.authorize_access_token(request)

    # authlib puts the id_token claims under "userinfo" when openid scope + nonce
    # were used; otherwise hit the userinfo endpoint explicitly.
    userinfo = token.get("userinfo") if isinstance(token, dict) else None
    if not userinfo:
        try:
            userinfo = await _client.userinfo(token=token)
        except Exception as exc:
            logger.warning("OIDC userinfo endpoint call failed: %s", exc)
            userinfo = None
    if not userinfo:
        logger.warning(
            "OIDC login produced no userinfo (token keys=%s). Check that the "
            "'openid' scope is granted and the provider returns an ID token.",
            sorted(token.keys()) if isinstance(token, dict) else type(token).__name__,
        )
        return None, False

    claim = getattr(settings, "oidc_username_claim", "preferred_username")
    username = userinfo.get(claim) or userinfo.get("email") or userinfo.get("sub")
    if not username:
        logger.warning(
            "OIDC login: no username found. Configured username_claim=%r, "
            "claims present=%s. Set webui.oidc.username_claim to one of these, or "
            "grant the scope that carries it (e.g. 'profile' for preferred_username, "
            "'email' for email).",
            claim, sorted(userinfo.keys()),
        )
        return None, False

    is_admin = _is_admin_from_groups(userinfo, settings)
    logger.info("OIDC login resolved username=%r admin=%s", str(username), is_admin)
    return str(username), is_admin
