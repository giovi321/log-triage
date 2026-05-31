"""Settings editor routes: render the structured form, validate + save YAML."""
from __future__ import annotations

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import RedirectResponse

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from ...notifications import add_notification
from ..auth import get_current_user
from ..state import STATE
from ..shared import render_config_editor
from .. import config_io

router = APIRouter()


@router.get("/config/edit", name="edit_config")
async def edit_config(request: Request):
    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)
    try:
        text = STATE.config_path.read_text(encoding="utf-8")
    except Exception as e:
        text = f"Error reading {STATE.config_path}: {e}"
    return render_config_editor(request, username, text)


@router.post("/config/edit", name="edit_config_post")
async def edit_config_post(request: Request, config_text: str = Form(...)):
    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    if yaml is None:
        return render_config_editor(
            request, username, config_text,
            error="YAML support is not available (missing PyYAML dependency).",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    try:
        yaml.safe_load(config_text)
    except Exception as e:
        add_notification("error", "Configuration validation failed", str(e))
        return render_config_editor(
            request, username, config_text,
            error=f"YAML error: {e}", status_code=status.HTTP_400_BAD_REQUEST,
        )

    try:
        config_io.save_config_text(config_text)
    except Exception as e:
        return render_config_editor(
            request, username, config_text,
            error=f"Write error: {e}", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    try:
        (STATE.reload_callback or config_io.reload_from_disk)()
    except Exception as exc:
        add_notification("error", "Configuration reload failed", str(exc))
        return render_config_editor(
            request, username, config_text,
            error=f"Reload failed: {exc}", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return render_config_editor(request, username, config_text, message="Configuration saved.")
