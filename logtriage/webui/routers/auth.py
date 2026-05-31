"""Authentication & account routes: local login, OIDC SSO, user management."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from ..auth import authenticate_user, create_session_token, get_current_user, current_user_is_admin
from ...notifications import add_notification
from ..state import STATE
from ..shared import templates
from .. import oidc as oidc_mod
from .. import users as users_mod

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
        add_notification("error", "OIDC callback failed", str(exc))
        username, is_admin = None, False
    if not username:
        return templates.TemplateResponse(
            "login.html",
            _login_context(request, error="SSO sign-in failed."),
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    _establish_session(request, username, is_admin)
    return RedirectResponse(url=request.app.url_path_for("issues"), status_code=status.HTTP_303_SEE_OTHER)


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
    return RedirectResponse(url=request.app.url_path_for("issues"), status_code=status.HTTP_303_SEE_OTHER)


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
        "oidc_enabled": bool(getattr(STATE.settings, "oidc_enabled", False)),
        "db_status": STATE.db_status,
    }


def _require_admin_or_redirect(request):
    """Return a redirect Response if the caller isn't an admin, else None."""
    if not get_current_user(request, STATE.settings):
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)
    if not current_user_is_admin(request, STATE.settings):
        return RedirectResponse(
            url=request.app.url_path_for("account") + "?error=Admin+access+required",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    return None


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

    def _err(msg, code=status.HTTP_400_BAD_REQUEST):
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

    return templates.TemplateResponse(
        "account.html",
        _account_context(
            request, username,
            message="Password updated. Existing sessions stay active until their cookies expire.",
        ),
    )


@router.post("/account/users/create", name="user_create")
async def user_create(
    request: Request,
    new_username: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    is_admin: str = Form(""),
):
    denied = _require_admin_or_redirect(request)
    if denied is not None:
        return denied
    account_url = request.app.url_path_for("account")
    if new_password != confirm_password:
        return RedirectResponse(account_url + "?error=Passwords+do+not+match", status_code=status.HTTP_303_SEE_OTHER)
    try:
        users_mod.create_user(new_username, new_password, is_admin=bool(is_admin))
    except ValueError as exc:
        return RedirectResponse(account_url + f"?error={str(exc).replace(' ', '+')}", status_code=status.HTTP_303_SEE_OTHER)
    except Exception as exc:
        add_notification("error", "User creation failed", str(exc))
        return RedirectResponse(account_url + "?error=Could+not+create+user", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(account_url + "?message=User+created", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/account/users/reset", name="user_reset_password")
async def user_reset_password(request: Request, target_username: str = Form(...), new_password: str = Form(...)):
    denied = _require_admin_or_redirect(request)
    if denied is not None:
        return denied
    account_url = request.app.url_path_for("account")
    try:
        ok = users_mod.set_password(target_username, new_password)
    except ValueError as exc:
        return RedirectResponse(account_url + f"?error={str(exc).replace(' ', '+')}", status_code=status.HTTP_303_SEE_OTHER)
    except Exception:
        ok = False
    msg = "?message=Password+reset" if ok else "?error=User+not+found"
    return RedirectResponse(account_url + msg, status_code=status.HTTP_303_SEE_OTHER)


@router.post("/account/users/delete", name="user_delete")
async def user_delete(request: Request, target_username: str = Form(...)):
    denied = _require_admin_or_redirect(request)
    if denied is not None:
        return denied
    username = get_current_user(request, STATE.settings)
    account_url = request.app.url_path_for("account")
    if target_username == username:
        return RedirectResponse(account_url + "?error=You+cannot+delete+your+own+account", status_code=status.HTTP_303_SEE_OTHER)
    try:
        ok = users_mod.delete_user(target_username)
    except ValueError as exc:
        return RedirectResponse(account_url + f"?error={str(exc).replace(' ', '+')}", status_code=status.HTTP_303_SEE_OTHER)
    except Exception:
        ok = False
    msg = "?message=User+deleted" if ok else "?error=User+not+found"
    return RedirectResponse(account_url + msg, status_code=status.HTTP_303_SEE_OTHER)
