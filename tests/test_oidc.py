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


def test_configure_strips_whitespace_from_client_id_and_issuer():
    """A trailing space pasted from the IdP console must not reach the client.

    Authentik rejects a client_id with stray whitespace as "missing or invalid",
    so configure() trims issuer/client_id/secret before building the client.
    """
    if not oidc_mod.oidc_available():
        pytest.skip("authlib not installed")
    s = _settings(
        enabled=True, issuer="  https://idp.example/app/o/lt/  ",
        client_id="  lt\n", client_secret="  sec  ",
    )
    assert oidc_mod.configure(s) is True
    client = oidc_mod.get_client()
    assert client.client_id == "lt"
    assert client.client_secret == "sec"
    # Tear down so global client state doesn't leak into other tests.
    oidc_mod.configure(_settings(enabled=False))


# ---- group → admin mapping ------------------------------------------------

def test_admin_groups_parse():
    s = _settings(groups_claim="roles", admin_groups=["lt-admins", "ops"])
    assert s.oidc_groups_claim == "roles"
    assert s.oidc_admin_groups == ["lt-admins", "ops"]


def test_admin_groups_default_empty():
    s = parse_webui_settings({"webui": {"secret_key": "k"}})
    assert s.oidc_groups_claim == "groups"
    assert s.oidc_admin_groups == []


def test_is_admin_from_groups_list_claim():
    s = _settings(groups_claim="groups", admin_groups=["lt-admins"])
    assert oidc_mod._is_admin_from_groups({"groups": ["users", "lt-admins"]}, s) is True
    assert oidc_mod._is_admin_from_groups({"groups": ["users"]}, s) is False


def test_is_admin_from_groups_space_string_claim():
    s = _settings(groups_claim="groups", admin_groups=["lt-admins"])
    assert oidc_mod._is_admin_from_groups({"groups": "users lt-admins"}, s) is True


def test_is_admin_from_groups_no_admin_groups_configured():
    s = _settings(admin_groups=[])
    assert oidc_mod._is_admin_from_groups({"groups": ["anything"]}, s) is False


def test_is_admin_from_groups_missing_claim():
    s = _settings(admin_groups=["lt-admins"])
    assert oidc_mod._is_admin_from_groups({"sub": "x"}, s) is False
