"""Local Web UI user management (DB-backed).

Local users live in the ``webui_users`` table (see :class:`db.UserRecord`),
not in ``config.yaml``. This module is the only place that touches that table;
auth and the user-management UI go through these helpers.

When OIDC or reverse-proxy forward-auth is configured, the IdP owns identity and
these local users are typically unused — but a local admin remains a useful
break-glass account, so the table is always available.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import List, Optional

from . import db
from .auth import pwd_context


@dataclass
class LocalUser:
    """A plain view of a user row (decoupled from the ORM session)."""
    id: int
    username: str
    password_hash: str
    is_admin: bool


def _row_to_user(row) -> LocalUser:
    return LocalUser(
        id=int(row.id),
        username=row.username,
        password_hash=row.password_hash,
        is_admin=bool(row.is_admin),
    )


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def list_users() -> List[LocalUser]:
    sess = db.get_session()
    if sess is None:
        return []
    try:
        rows = sess.query(db.UserRecord).order_by(db.UserRecord.username.asc()).all()
        return [_row_to_user(r) for r in rows]
    except Exception:
        return []
    finally:
        sess.close()


def count_users() -> int:
    sess = db.get_session()
    if sess is None:
        return 0
    try:
        from sqlalchemy import func
        return int(sess.query(func.count(db.UserRecord.id)).scalar() or 0)
    except Exception:
        return 0
    finally:
        sess.close()


def get_user(username: str) -> Optional[LocalUser]:
    if not username:
        return None
    sess = db.get_session()
    if sess is None:
        return None
    try:
        row = (
            sess.query(db.UserRecord)
            .filter(db.UserRecord.username == username)
            .one_or_none()
        )
        return _row_to_user(row) if row is not None else None
    except Exception:
        return None
    finally:
        sess.close()


def verify_credentials(username: str, password: str) -> Optional[LocalUser]:
    """Return the user iff the password matches its stored hash."""
    user = get_user(username)
    if user is None:
        return None
    try:
        if pwd_context.verify(password, user.password_hash):
            return user
    except Exception:
        return None
    return None


def create_user(username: str, password: str, is_admin: bool = True) -> LocalUser:
    """Create a user. Raises ValueError on bad input or a duplicate username."""
    username = (username or "").strip()
    if not username:
        raise ValueError("Username is required.")
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if get_user(username) is not None:
        raise ValueError(f"User '{username}' already exists.")

    sess = db.get_session()
    if sess is None:
        raise RuntimeError("Database is not configured.")
    try:
        row = db.UserRecord(
            username=username,
            password_hash=hash_password(password),
            is_admin=bool(is_admin),
        )
        sess.add(row)
        sess.commit()
        sess.refresh(row)
        return _row_to_user(row)
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def set_password(username: str, new_password: str) -> bool:
    if not new_password or len(new_password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    sess = db.get_session()
    if sess is None:
        return False
    try:
        updated = (
            sess.query(db.UserRecord)
            .filter(db.UserRecord.username == username)
            .update(
                {
                    "password_hash": hash_password(new_password),
                    "updated_at": datetime.datetime.now(datetime.timezone.utc),
                },
                synchronize_session=False,
            )
        )
        sess.commit()
        return bool(updated)
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def delete_user(username: str) -> bool:
    """Delete a user. Refuses to remove the last remaining user (lockout guard)."""
    if count_users() <= 1:
        raise ValueError("Cannot delete the last remaining user.")
    sess = db.get_session()
    if sess is None:
        return False
    try:
        deleted = (
            sess.query(db.UserRecord)
            .filter(db.UserRecord.username == username)
            .delete(synchronize_session=False)
        )
        sess.commit()
        return bool(deleted)
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def seed_users_from_config(admin_users) -> int:
    """One-time seed of config ``webui.admin_users`` into the table.

    Only runs when the table is empty, so it never clobbers users created or
    edited through the UI. ``admin_users`` is the parsed list of WebUser
    (objects with ``.username`` / ``.password_hash``). Returns rows inserted.
    """
    if count_users() > 0:
        return 0
    if not admin_users:
        return 0
    sess = db.get_session()
    if sess is None:
        return 0
    inserted = 0
    try:
        for u in admin_users:
            username = getattr(u, "username", None)
            pw_hash = getattr(u, "password_hash", None)
            if not username or not pw_hash:
                continue
            sess.add(db.UserRecord(username=username, password_hash=pw_hash, is_admin=True))
            inserted += 1
        sess.commit()
        return inserted
    except Exception:
        sess.rollback()
        return 0
    finally:
        sess.close()
