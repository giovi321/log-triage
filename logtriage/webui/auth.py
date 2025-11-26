from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette import status
from passlib.context import CryptContext

from .config import WebUISettings, get_client_ip


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
    # Simple HMAC-based token: username|signature
    import hashlib
    msg = username.encode("utf-8")
    key = secret_key.encode("utf-8")
    sig = hmac.new(key, msg, hashlib.sha256).hexdigest()
    return f"{username}|{sig}"


def validate_session_token(token: str, secret_key: str) -> Optional[str]:
    import hashlib
    try:
        username, sig = token.split("|", 1)
    except ValueError:
        return None
    msg = username.encode("utf-8")
    key = secret_key.encode("utf-8")
    expected_sig = hmac.new(key, msg, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return None
    return username


def get_current_user(
    request: Request,
    settings: WebUISettings,
) -> Optional[str]:
    token = request.session.get("session_token")
    if not token:
        return None
    username = validate_session_token(token, settings.secret_key)
    return username


def require_login(
    request: Request,
    settings: WebUISettings,
) -> str:
    username = get_current_user(request, settings)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return username
