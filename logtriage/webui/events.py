"""Server-Sent Events hub for live Web UI updates.

A single in-process background poller builds a small snapshot of changing state
(issue counts, latest finding id, RAG indexing progress) every couple of seconds
and publishes it to all connected browsers. This replaces per-client polling:
N browsers cost one server-side DB poll, and each gets a push stream instead of
hammering ``/api/rag/progress`` on a timer.

Cross-process note: findings are written by a separate CLI process, so the
poller detects changes by reading the shared database — there is no in-process
event from the ingester. That is fine; the poll is cheap (COUNT + MAX(id)).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Set


class EventHub:
    """Fan-out of events to subscribed SSE connections via asyncio queues."""

    def __init__(self) -> None:
        self._subscribers: Set[asyncio.Queue] = set()

    def subscribe(self, maxsize: int = 20) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish(self, event: Dict[str, Any]) -> None:
        """Push an event to every subscriber, dropping the oldest for slow ones."""
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except Exception:
                    pass
                try:
                    q.put_nowait(event)
                except Exception:
                    pass


def sse_format(event: Dict[str, Any]) -> str:
    """Encode an event dict as an SSE frame."""
    etype = event.get("type", "message")
    return f"event: {etype}\ndata: {json.dumps(event, default=str)}\n\n"


def db_snapshot() -> Dict[str, Any]:
    """Cheap database stats for the live snapshot."""
    from .db import issue_status_counts, get_max_finding_id

    try:
        counts = issue_status_counts()
    except Exception:
        counts = {}
    open_issues = (counts.get("open", 0) + counts.get("acknowledged", 0)) if counts else 0
    return {
        "issue_counts": counts,
        "open_issues": open_issues,
        "latest_finding_id": get_max_finding_id(),
    }
