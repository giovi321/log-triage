from __future__ import annotations

import hashlib
import hmac
import time
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette import status
from passlib.context import CryptContext

from .config import WebUISettings, get_client_ip

# Default session lifetime in seconds (24 hours).  Override via webui.session_max_age_hours.
_DEFAULT_SESSION_MAX_AGE = 86400


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBasic()


def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        return pwd_context.verify(plain_password, password_hash)
    except Exception:
        return False


def get_user(settings: WebUISettings, username: str):
    for user in settings.admin_users:
        if user.username == username:
            return user
    return None


def authenticate_user(settings: WebUISettings, username: str, password: str):
    user = get_user(settings, username)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def create_session_token(username: str, secret_key: str) -> str:
    # Token format: username|issued_at_int|hmac_sha256(username|issued_at_int)
    issued_at = str(int(time.time()))
    payload = f"{username}|{issued_at}"
    key = secret_key.encode("utf-8")
    sig = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}|{sig}"


def validate_session_token(token: str, secret_key: str, max_age: int = _DEFAULT_SESSION_MAX_AGE) -> Optional[str]:
    try:
        username, issued_at_str, sig = token.split("|", 2)
    except ValueError:
        return None
    payload = f"{username}|{issued_at_str}"
    key = secret_key.encode("utf-8")
    expected_sig = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return None
    try:
        issued_at = int(issued_at_str)
    except ValueError:
        return None
    if time.time() - issued_at > max_age:
        return None
    return username


def get_current_user(
    request: Request,
    settings: WebUISettings,
) -> Optional[str]:
    token = request.session.get("session_token")
    if not token:
        return None
    max_age = int(getattr(settings, "session_max_age_hours", 24)) * 3600
    username = validate_session_token(token, settings.secret_key, max_age=max_age)
    return username


def require_login(
    request: Request,
    settings: WebUISettings,
) -> str:
    username = get_current_user(request, settings)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return username
