"""Live-update helpers for the SSE stream (reload-safe, read STATE).

Extracted from app.py so the events router and the background poll loop share
one implementation without importing app. All runtime state (rag_client,
enrichment_worker, rag_monitor) is read from STATE at call time.
"""
from __future__ import annotations

from typing import Any, Dict

from .events import db_snapshot
from .state import STATE


def get_rag_monitor_status() -> Dict[str, Any]:
    """Current RAG monitor status (falls back to the shared status dict)."""
    monitor = STATE.rag_monitor
    if monitor is None:
        return dict(STATE.rag_monitor_status)
    return monitor.get_status()


def fetch_rag_progress():
    client = STATE.rag_client
    if client is None or not hasattr(client, "_make_request"):
        return None
    try:
        return client._make_request("GET", "/progress", max_retries=0)
    except Exception:
        return None


def build_live_snapshot() -> Dict[str, Any]:
    """Snapshot of changing state for the SSE stream (run in a thread executor)."""
    snap = db_snapshot()
    monitor = get_rag_monitor_status()
    snap["rag"] = {
        "available": bool(monitor.get("rag_available")),
        "ready": bool(monitor.get("rag_ready")),
        "progress": fetch_rag_progress() if monitor.get("rag_available") else None,
    }
    worker = STATE.enrichment_worker
    wstatus = worker.status if worker is not None else {"running": False, "total_analyzed": 0}
    snap["worker"] = {
        "running": bool(wstatus.get("running")),
        "total_analyzed": wstatus.get("total_analyzed", 0),
    }
    return snap
