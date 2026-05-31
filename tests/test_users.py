"""Tests for DB-backed local user management."""
import types

import pytest

from logtriage.webui import db, users
from logtriage.webui.auth import authenticate_user


def test_create_and_verify(database):
    u = users.create_user("alice", "supersecret", is_admin=True)
    assert u.username == "alice"
    assert u.is_admin is True
    assert users.count_users() == 1

    assert users.verify_credentials("alice", "supersecret") is not None
    assert users.verify_credentials("alice", "wrong") is None
    assert users.verify_credentials("nobody", "supersecret") is None


def test_create_rejects_short_password_and_duplicates(database):
    users.create_user("alice", "supersecret")
    with pytest.raises(ValueError):
        users.create_user("bob", "short")          # < 8 chars
    with pytest.raises(ValueError):
        users.create_user("", "supersecret")       # empty username
    with pytest.raises(ValueError):
        users.create_user("alice", "anotherone")   # duplicate


def test_set_password(database):
    users.create_user("alice", "supersecret")
    assert users.set_password("alice", "newsecret1") is True
    assert users.verify_credentials("alice", "supersecret") is None
    assert users.verify_credentials("alice", "newsecret1") is not None


def test_delete_user_guards_last_account(database):
    users.create_user("alice", "supersecret")
    users.create_user("bob", "supersecret")
    assert users.delete_user("bob") is True
    assert users.count_users() == 1
    # Deleting the final user is refused (lockout guard).
    with pytest.raises(ValueError):
        users.delete_user("alice")
    assert users.count_users() == 1


def test_seed_from_config_only_when_empty(database):
    cfg_users = [
        types.SimpleNamespace(username="admin", password_hash="$2b$12$abc"),
        types.SimpleNamespace(username="ops", password_hash="$2b$12$def"),
    ]
    assert users.seed_users_from_config(cfg_users) == 2
    assert {u.username for u in users.list_users()} == {"admin", "ops"}
    # Idempotent: a second seed with the table populated inserts nothing.
    assert users.seed_users_from_config(cfg_users) == 0
    assert users.count_users() == 2


def test_authenticate_user_prefers_db(database):
    users.create_user("alice", "supersecret")
    settings = types.SimpleNamespace(admin_users=[], secret_key="k")
    assert authenticate_user(settings, "alice", "supersecret") is not None
    assert authenticate_user(settings, "alice", "nope") is None


def test_first_user_is_forced_admin(database):
    # Even when asked for a non-admin, the very first account must be admin so a
    # fresh install isn't locked out of admin-only surfaces.
    u = users.create_user("alice", "supersecret", is_admin=False)
    assert u.is_admin is True
    # Subsequent users honour the requested flag.
    v = users.create_user("bob", "supersecret", is_admin=False)
    assert v.is_admin is False
    assert users.count_admins() == 1


def test_cannot_delete_last_admin(database):
    users.create_user("admin", "supersecret", is_admin=True)   # forced admin (first)
    users.create_user("bob", "supersecret", is_admin=False)
    assert users.count_admins() == 1
    with pytest.raises(ValueError, match="last remaining admin"):
        users.delete_user("admin")
    # A non-admin can still be deleted.
    assert users.delete_user("bob") is True
