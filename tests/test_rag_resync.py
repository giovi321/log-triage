"""Tests for RAG auto-resync: registering a repo added while the service was busy.

Covers the monitor's rising-edge on_ready callback and config_io's
"only re-register when a configured repo is actually missing" diff.
"""
from __future__ import annotations

import types

from logtriage.rag.monitor import RAGServiceMonitor
from logtriage.webui import config_io
from logtriage.webui.state import STATE


def _monitor(on_ready):
    return RAGServiceMonitor(
        status={},
        get_service_url=lambda: "http://x",
        create_client=lambda url: None,
        get_client=lambda: None,
        set_client=lambda c: None,
        on_ready=on_ready,
    )


def test_on_ready_fires_only_on_rising_edge():
    fired = []
    mon = _monitor(lambda: fired.append(1))

    mon._maybe_fire_on_ready(False)   # still not ready -> no fire
    mon._maybe_fire_on_ready(True)    # False -> True: fire
    mon._maybe_fire_on_ready(True)    # stays ready: no fire
    mon._maybe_fire_on_ready(False)   # dropped (e.g. indexing): no fire
    mon._maybe_fire_on_ready(True)    # rose again: fire

    assert len(fired) == 2


def test_on_ready_callback_errors_are_swallowed():
    def boom():
        raise RuntimeError("nope")
    mon = _monitor(boom)
    mon._maybe_fire_on_ready(True)  # must not raise
    assert mon._prev_ready is True


# ---- config_io.resync_rag_if_repos_missing --------------------------------

def _src(repo_url, branch="main"):
    return types.SimpleNamespace(repo_url=repo_url, branch=branch, include_paths=[])


def _module(*sources):
    rag = types.SimpleNamespace(enabled=True, knowledge_sources=list(sources))
    return types.SimpleNamespace(name="svc", rag=rag)


class _FakeProbe:
    def __init__(self, repo_ids):
        self._repo_ids = repo_ids
    def is_ready(self):
        return True
    def get_status(self):
        return {"repositories": [{"repo_id": rid} for rid in self._repo_ids]}


def _patch_common(monkeypatch, modules, service_repo_ids):
    monkeypatch.setattr(config_io, "build_rag_config",
                        lambda raw: types.SimpleNamespace(enabled=True, service_url="http://x"))
    monkeypatch.setattr(config_io, "build_modules_safe", lambda: modules)
    monkeypatch.setattr(config_io, "create_rag_client", lambda url, fallback=True: _FakeProbe(service_repo_ids))
    called = {"refresh": 0}
    monkeypatch.setattr(config_io, "refresh_rag_client", lambda: called.__setitem__("refresh", called["refresh"] + 1))
    return called


def test_resync_triggers_when_repo_missing(monkeypatch):
    src = _src("https://github.com/acme/docs")
    module = _module(src)
    # The service has nothing registered yet -> the configured repo is missing.
    called = _patch_common(monkeypatch, [module], service_repo_ids=set())
    assert config_io.resync_rag_if_repos_missing() is True
    assert called["refresh"] == 1


def test_resync_noop_when_all_present(monkeypatch):
    src = _src("https://github.com/acme/docs")
    module = _module(src)
    present = {config_io._rag_repo_id(src)}
    called = _patch_common(monkeypatch, [module], service_repo_ids=present)
    assert config_io.resync_rag_if_repos_missing() is False
    assert called["refresh"] == 0


def test_resync_noop_when_no_configured_repos(monkeypatch):
    called = _patch_common(monkeypatch, [], service_repo_ids=set())
    assert config_io.resync_rag_if_repos_missing() is False
    assert called["refresh"] == 0
