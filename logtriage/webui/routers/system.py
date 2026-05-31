"""System routes: SSE live-update stream and the Prometheus metrics endpoint."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import get_current_user
from ..events import sse_format
from ..live import build_live_snapshot
from ..state import STATE

router = APIRouter()


@router.get("/events", name="events")
async def events_stream(request: Request):
    """Server-Sent Events stream of live snapshots (issues, findings, RAG, worker)."""
    if not get_current_user(request, STATE.settings):
        return JSONResponse({"error": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED)

    event_hub = STATE.event_hub
    queue = event_hub.subscribe()

    async def gen():
        loop = asyncio.get_event_loop()
        try:
            snap = await loop.run_in_executor(None, build_live_snapshot)
            snap["type"] = "snapshot"
            yield sse_format(snap)
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield sse_format(ev)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            event_hub.unsubscribe(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("/metrics", name="metrics")
async def metrics_endpoint(request: Request):
    """Prometheus metrics. Unauthenticated (for scraping) but still behind the
    allowed_ips middleware; disable via webui.metrics.enabled: false."""
    if not getattr(STATE.settings, "metrics_enabled", True):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    from ..metrics import render_metrics

    worker = STATE.enrichment_worker
    wstatus = worker.status if worker is not None else {}
    text = render_metrics(worker_status=wstatus)
    return Response(content=text, media_type="text/plain; version=0.0.4; charset=utf-8")
