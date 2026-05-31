"""The overview / operations-console dashboard."""
from __future__ import annotations

import datetime
import logging

from fastapi import APIRouter, Request, status
from fastapi.responses import RedirectResponse

from ...notifications import notification_summary
from ..auth import get_current_user
from ..db import get_module_stats
from ..ingestion_status import _derive_ingestion_status
from ..live import get_rag_monitor_status
from ..state import STATE
from ..shared import templates, build_modules_from_config

logger = logging.getLogger(__name__)

router = APIRouter()

_DEFAULT_INIT = {
    "started": False, "completed": False, "updating": False, "error": None,
    "current_phase": "unavailable",
    "progress": {"current_step": 0, "total_steps": 5,
                 "step_description": "RAG service not available", "percentage": 0.0},
    "repository_updates": {"current_repo": None, "total_repos": 0,
                           "completed_repos": 0, "current_progress": 0.0},
}
_INITIALIZING = {
    "started": True, "completed": False, "updating": True, "error": None,
    "current_phase": "initializing",
    "progress": {"current_step": 0, "total_steps": 5,
                 "step_description": "RAG service initializing...", "percentage": 0.0},
    "repository_updates": {"current_repo": None, "total_repos": 0,
                           "completed_repos": 0, "current_progress": 0.0},
}


@router.get("/", name="dashboard")
async def dashboard(request: Request):
    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    modules = sorted(build_modules_from_config(), key=lambda m: (not m.enabled, m.name.lower()))
    stats = get_module_stats(modules)
    page_rendered_at = datetime.datetime.now(datetime.timezone.utc)
    ingestion_status = _derive_ingestion_status(
        modules, now=page_rendered_at, freshness_minutes=STATE.settings.staleness_minutes
    )
    notif_summary = notification_summary()

    rag_monitor_data = get_rag_monitor_status()
    if rag_monitor_data["detailed_status"] is None:
        rag_monitor_data["detailed_status"] = {"initialization": dict(_DEFAULT_INIT)}
    elif rag_monitor_data["rag_available"] and not rag_monitor_data["rag_ready"]:
        if rag_monitor_data["detailed_status"].get("initialization") is None:
            rag_monitor_data["detailed_status"]["initialization"] = dict(_INITIALIZING)

    normalized_rag_status = {
        "enabled": rag_monitor_data["rag_ready"],
        "total_repositories": 0,
        "vector_store_stats": {"total_chunks": 0, "persist_directory": "N/A"},
        "repositories": [],
        "detailed_status": rag_monitor_data["detailed_status"],
    }

    rag_client = STATE.rag_client
    if rag_monitor_data["rag_ready"] and rag_client:
        try:
            real_status = rag_client.get_status()
            if real_status:
                if real_status.get("vector_store_stats"):
                    normalized_rag_status["vector_store_stats"] = real_status["vector_store_stats"]
                else:
                    normalized_rag_status["vector_store_stats"] = {
                        "total_chunks": 0, "persist_directory": "Service running but no data"
                    }
                for key, value in real_status.items():
                    if key != "vector_store_stats":
                        normalized_rag_status[key] = value
            else:
                logger.warning("RAG client get_status() returned None")
        except Exception as e:
            logger.warning("Failed to get real RAG status: %s", e)
            normalized_rag_status["vector_store_stats"]["persist_directory"] = "Service error"

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "username": username,
            "modules": modules,
            "stats": stats,
            "db_status": STATE.db_status,
            "page_rendered_at": page_rendered_at,
            "ingestion_status": ingestion_status,
            "notif_summary": notif_summary,
            "rag_status": normalized_rag_status,
            "rag_service_available": rag_monitor_data["rag_available"],
            "rag_service_ready": rag_monitor_data["rag_ready"],
            "rag_monitor": rag_monitor_data,
        },
    )
