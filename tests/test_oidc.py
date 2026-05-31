"""Tests for OIDC config parsing and client configuration.

The full Authorization Code flow needs a live IdP and is verified manually on
the deployment runtime. Here we cover settings parsing and the configure()
state machine (disabled / misconfigured / configured), which is where the
logic bugs would hide.
"""
import pytest

from logtriage.webui.config import parse_webui_settings
from logtriage.webui import oidc as oidc_mod


def _settings(**oidc):
    return parse_webui_settings({"webui": {"secret_key": "k", "oidc": oidc}})


def test_oidc_settings_parse_defaults():
    s = parse_webui_settings({"webui": {"secret_key": "k"}})
    assert s.oidc_enabled is False
    assert s.oidc_scopes == "openid email profile"
    assert s.oidc_username_claim == "preferred_username"
    assert s.oidc_exclusive is False


def test_oidc_settings_parse_values():
    s = _settings(
        enabled=True, issuer="https://idp.example/app/o/lt/", client_id="lt",
        client_secret="sec", scopes="openid", username_claim="email", exclusive=True,
    )
    assert s.oidc_enabled is True
    assert s.oidc_issuer == "https://idp.example/app/o/lt/"
    assert s.oidc_client_id == "lt"
    assert s.oidc_client_secret == "sec"
    assert s.oidc_scopes == "openid"
    assert s.oidc_username_claim == "email"
    assert s.oidc_exclusive is True


def test_configure_disabled_returns_false():
    s = _settings(enabled=False)
    assert oidc_mod.configure(s) is False
    assert oidc_mod.is_configured() is False


def test_configure_missing_fields_returns_false():
    s = _settings(enabled=True)  # no issuer/client_id
    assert oidc_mod.configure(s) is False
    assert oidc_mod.is_configured() is False


def test_configure_builds_client_when_authlib_present():
    if not oidc_mod.oidc_available():
        pytest.skip("authlib not installed")
    s = _settings(
        enabled=True, issuer="https://idp.example/app/o/lt/",
        client_id="lt", client_secret="sec",
    )
    assert oidc_mod.configure(s) is True
    assert oidc_mod.is_configured() is True
    # Disabling tears the client down again.
    assert oidc_mod.configure(_settings(enabled=False)) is False
    assert oidc_mod.is_configured() is False
