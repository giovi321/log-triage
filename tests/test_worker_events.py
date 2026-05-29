"""Tests for the enrichment worker loop and the SSE event hub."""
import asyncio

from logtriage import worker as worker_mod
from logtriage.worker import EnrichmentWorker
from logtriage.webui.events import EventHub, sse_format


# ---- worker ---------------------------------------------------------------

def test_run_once_calls_analyze_and_accumulates(monkeypatch):
    calls = {}

    def fake(modules, llm_defaults, rag_client=None, limit=25):
        calls["modules"] = modules
        calls["limit"] = limit
        return 3

    monkeypatch.setattr(worker_mod, "analyze_pending_issues", fake)
    w = EnrichmentWorker(lambda: ({"m": object()}, object(), None), interval=10, batch=7)

    assert w.run_once() == 3
    assert calls["limit"] == 7
    assert w.status["last_count"] == 3
    assert w.status["total_analyzed"] == 3
    assert w.run_once() == 3
    assert w.status["total_analyzed"] == 6


def test_run_once_skips_when_no_deps(monkeypatch):
    called = {"n": 0}

    def fake(*a, **k):
        called["n"] += 1
        return 1

    monkeypatch.setattr(worker_mod, "analyze_pending_issues", fake)
    w = EnrichmentWorker(lambda: None, interval=10)
    assert w.run_once() == 0
    assert called["n"] == 0


def test_run_once_handles_analyze_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(worker_mod, "analyze_pending_issues", boom)
    w = EnrichmentWorker(lambda: ({"m": object()}, object(), None))
    assert w.run_once() == 0
    assert "provider down" in (w.status["error"] or "")


# ---- event hub ------------------------------------------------------------

def test_eventhub_publish_subscribe():
    async def run():
        hub = EventHub()
        q = hub.subscribe()
        assert hub.subscriber_count() == 1
        hub.publish({"type": "snapshot", "x": 1})
        ev = await asyncio.wait_for(q.get(), timeout=1)
        assert ev["x"] == 1
        hub.unsubscribe(q)
        assert hub.subscriber_count() == 0

    asyncio.run(run())


def test_eventhub_drops_oldest_when_full():
    async def run():
        hub = EventHub()
        q = hub.subscribe(maxsize=2)
        hub.publish({"n": 1})
        hub.publish({"n": 2})
        hub.publish({"n": 3})  # full → oldest (1) dropped
        got = sorted([(await q.get())["n"], (await q.get())["n"]])
        assert got == [2, 3]

    asyncio.run(run())


def test_sse_format():
    out = sse_format({"type": "snapshot", "a": 1})
    assert out.startswith("event: snapshot\n")
    assert '"a": 1' in out
    assert out.endswith("\n\n")
