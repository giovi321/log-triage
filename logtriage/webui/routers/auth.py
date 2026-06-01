"""Authentication & account routes: local login, OIDC SSO, user management."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ..auth import authenticate_user, create_session_token, get_current_user, current_user_is_admin
from ...notifications import add_notification
from ..state import STATE
from ..shared import templates, wants_json, user_has_local_password
from .. import oidc as oidc_mod
from .. import users as users_mod

logger = logging.getLogger(__name__)

router = APIRouter()


# ---- login / SSO ----------------------------------------------------------

def _login_context(request: Request, error=None):
    s = STATE.settings
    return {
        "request": request,
        "error": error,
        "username": get_current_user(request, s),
        "oidc_enabled": bool(getattr(s, "oidc_enabled", False) and oidc_mod.is_configured()),
        "oidc_exclusive": bool(getattr(s, "oidc_exclusive", False)),
    }


@router.get("/login", name="login_form")
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", _login_context(request))


@router.get("/login/oidc", name="login_oidc")
async def login_oidc(request: Request):
    """Begin the OIDC Authorization Code + PKCE flow."""
    login_url = request.app.url_path_for("login_form")
    if not oidc_mod.is_configured():
        return RedirectResponse(
            url=login_url + "?error=OIDC+is+not+configured",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    redirect_uri = str(request.url_for("oidc_callback"))
    try:
        return await oidc_mod.authorize_redirect(request, redirect_uri)
    except Exception as exc:
        add_notification("error", "OIDC login failed", str(exc))
        return RedirectResponse(
            url=login_url + "?error=OIDC+login+could+not+start",
            status_code=status.HTTP_303_SEE_OTHER,
        )


def _establish_session(request: Request, username: str, is_admin: bool) -> None:
    """Sign the user in: set the session token + the admin flag (paired to the
    username so it can't be reused for a different identity)."""
    request.session["session_token"] = create_session_token(username, STATE.settings.secret_key)
    request.session["is_admin"] = bool(is_admin)
    request.session["is_admin_user"] = username


@router.get("/auth/callback", name="oidc_callback")
async def oidc_callback(request: Request):
    """Complete the OIDC flow: validate the token and establish a session."""
    if not oidc_mod.is_configured():
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)
    try:
        username, is_admin = await oidc_mod.fetch_identity(request, STATE.settings)
    except Exception as exc:
        # Log the full traceback so a failed callback is diagnosable from the
        # service logs (the notification alone isn't visible there).
        logger.warning("OIDC callback failed: %s", exc, exc_info=True)
        add_notification("error", "OIDC callback failed", str(exc))
        username, is_admin = None, False
    if not username:
        return templates.TemplateResponse(
            "login.html",
            _login_context(request, error="SSO sign-in failed."),
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    _establish_session(request, username, is_admin)
    return RedirectResponse(url=request.app.url_path_for("dashboard"), status_code=status.HTTP_303_SEE_OTHER)


@router.post("/login", name="login_form_post")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    user = authenticate_user(STATE.settings, username, password)
    if not user:
        return templates.TemplateResponse(
            "login.html",
            _login_context(request, error="Invalid credentials"),
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    _establish_session(request, username, bool(getattr(user, "is_admin", False)))
    return RedirectResponse(url=request.app.url_path_for("dashboard"), status_code=status.HTTP_303_SEE_OTHER)


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)


# ---- account / user management --------------------------------------------

def _account_context(request, username, *, error=None, message=None):
    is_admin = current_user_is_admin(request, STATE.settings)
    user_list = []
    if is_admin:
        try:
            user_list = users_mod.list_users()
        except Exception:
            user_list = []
    return {
        "request": request,
        "username": username,
        "error": error,
        "message": message,
        "users": user_list,
        "is_admin": is_admin,
        # SSO-provided accounts have no local password row to change.
        "can_change_password": user_has_local_password(username),
        "oidc_enabled": bool(getattr(STATE.settings, "oidc_enabled", False)),
        "db_status": STATE.db_status,
    }


def _users_payload():
    """JSON-serialisable local-user list for AJAX responses (Account tab table)."""
    try:
        return [{"username": u.username, "is_admin": bool(u.is_admin)} for u in users_mod.list_users()]
    except Exception:
        return []


@router.get("/account", name="account")
async def account(request: Request, error: Optional[str] = None, message: Optional[str] = None):
    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(
        "account.html", _account_context(request, username, error=error, message=message)
    )


@router.post("/account/password", name="change_password")
async def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
):
    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)
    ajax = wants_json(request)

    def _err(msg, code=status.HTTP_400_BAD_REQUEST):
        if ajax:
            return JSONResponse({"ok": False, "error": msg}, status_code=code)
        return templates.TemplateResponse(
            "account.html", _account_context(request, username, error=msg), status_code=code
        )

    if authenticate_user(STATE.settings, username, current_password) is None:
        return _err("Current password is incorrect.")
    if new_password != confirm_password:
        return _err("New passwords do not match.")
    if len(new_password) < 8:
        return _err("Use at least 8 characters for the new password.")

    try:
        if not users_mod.set_password(username, new_password):
            return _err("Your account has no local password to change (it may be managed by your SSO provider).")
    except ValueError as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"Failed to update password: {exc}", code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    msg = "Password updated. Existing sessions stay active until their cookies expire."
    if ajax:
        return JSONResponse({"ok": True, "message": msg})
    return templates.TemplateResponse(
        "account.html", _account_context(request, username, message=msg)
    )


def _admin_gate(request):
    """Admin check for user-management endpoints. Returns a Response (JSON for
    AJAX, redirect otherwise) when denied, else None."""
    if not get_current_user(request, STATE.settings):
        if wants_json(request):
            return JSONResponse({"ok": False, "error": "Not signed in."}, status_code=status.HTTP_401_UNAUTHORIZED)
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)
    if not current_user_is_admin(request, STATE.settings):
        if wants_json(request):
            return JSONResponse({"ok": False, "error": "Admin access required."}, status_code=status.HTTP_403_FORBIDDEN)
        return RedirectResponse(
            url=request.app.url_path_for("account") + "?error=Admin+access+required",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    return None


def _account_redirect(request, *, message=None, error=None):
    """Redirect back to the Settings → Account tab with a flash query param."""
    account_url = request.app.url_path_for("edit_config")
    if message:
        return RedirectResponse(account_url + f"?message={message.replace(' ', '+')}#account", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(account_url + f"?error={(error or '').replace(' ', '+')}#account", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/account/users/create", name="user_create")
async def user_create(
    request: Request,
    new_username: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    is_admin: str = Form(""),
):
    denied = _admin_gate(request)
    if denied is not None:
        return denied
    ajax = wants_json(request)

    def _done(*, message=None, error=None, code=status.HTTP_400_BAD_REQUEST):
        if ajax:
            if error:
                return JSONResponse({"ok": False, "error": error}, status_code=code)
            return JSONResponse({"ok": True, "message": message, "users": _users_payload()})
        return _account_redirect(request, message=message, error=error)

    if new_password != confirm_password:
        return _done(error="Passwords do not match.")
    try:
        users_mod.create_user(new_username, new_password, is_admin=bool(is_admin))
    except ValueError as exc:
        return _done(error=str(exc))
    except Exception as exc:
        add_notification("error", "User creation failed", str(exc))
        return _done(error="Could not create user.", code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return _done(message="User created.")


@router.post("/account/users/reset", name="user_reset_password")
async def user_reset_password(request: Request, target_username: str = Form(...), new_password: str = Form(...)):
    denied = _admin_gate(request)
    if denied is not None:
        return denied
    ajax = wants_json(request)
    try:
        ok = users_mod.set_password(target_username, new_password)
        error = None
    except ValueError as exc:
        ok, error = False, str(exc)
    except Exception:
        ok, error = False, "Could not reset password."
    if ajax:
        if ok:
            return JSONResponse({"ok": True, "message": "Password reset."})
        return JSONResponse({"ok": False, "error": error or "User not found."}, status_code=status.HTTP_400_BAD_REQUEST)
    if ok:
        return _account_redirect(request, message="Password reset")
    return _account_redirect(request, error=error or "User not found")


@router.post("/account/users/delete", name="user_delete")
async def user_delete(request: Request, target_username: str = Form(...)):
    denied = _admin_gate(request)
    if denied is not None:
        return denied
    ajax = wants_json(request)
    username = get_current_user(request, STATE.settings)
    if target_username == username:
        msg = "You cannot delete your own account."
        if ajax:
            return JSONResponse({"ok": False, "error": msg}, status_code=status.HTTP_400_BAD_REQUEST)
        return _account_redirect(request, error="You cannot delete your own account")
    try:
        ok = users_mod.delete_user(target_username)
        error = None
    except ValueError as exc:
        ok, error = False, str(exc)
    except Exception:
        ok, error = False, "User not found."
    if ajax:
        if ok:
            return JSONResponse({"ok": True, "message": "User deleted.", "users": _users_payload()})
        return JSONResponse({"ok": False, "error": error or "User not found."}, status_code=status.HTTP_400_BAD_REQUEST)
    if ok:
        return _account_redirect(request, message="User deleted")
    return _account_redirect(request, error=error or "User not found")
