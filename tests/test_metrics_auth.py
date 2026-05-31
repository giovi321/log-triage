"""Tests for the Prometheus /metrics builder and reverse-proxy forward-auth."""
import datetime
import types
from pathlib import Path

import pytest

from logtriage.models import Finding, Severity
from logtriage.webui import db
from logtriage.webui.metrics import render_metrics
from logtriage.webui.config import parse_webui_settings
from logtriage.webui.auth import resolve_proxy_user, get_current_user


def _finding(text, line):
    return Finding(
        file_path=Path("/var/log/x.log"), pipeline_name="ha", finding_index=0,
        severity=Severity.ERROR, message='Matched error pattern /ERROR/ on "ERROR"',
        line_start=line, line_end=line, rule_id="ERROR", excerpt=[text],
        needs_llm=False, created_at=datetime.datetime.now(datetime.timezone.utc),
    )


# ---- metrics --------------------------------------------------------------

def test_render_metrics_contains_expected_series(database):
    db.store_finding("ha", _finding("ERROR a", 1))
    db.store_finding("ha", _finding("ERROR b", 2))
    out = render_metrics(worker_status={"running": True, "total_analyzed": 5})
    assert "# TYPE logtriage_issues gauge" in out
    assert 'logtriage_issues{status="open"} 2' in out
    assert "logtriage_issues_active 2" in out
    assert "logtriage_findings_total 2" in out
    assert "logtriage_worker_running 1" in out
    assert "logtriage_worker_analyzed_total 5" in out


def test_render_metrics_safe_without_db():
    db._engine = None
    db._db_url = None
    out = render_metrics(worker_status={})
    assert "logtriage_findings_total 0" in out
    assert "logtriage_worker_running 0" in out


# ---- forward-auth ---------------------------------------------------------

def _settings(enabled=True, trusted=("10.0.0.1",), header="X-authentik-username"):
    raw = {"webui": {
        "secret_key": "k", "trusted_proxies": list(trusted),
        "forward_auth": {"enabled": enabled, "username_header": header},
    }}
    return parse_webui_settings(raw)


def _request(host, headers, session=None):
    return types.SimpleNamespace(
        client=types.SimpleNamespace(host=host),
        headers=headers,
        session=session if session is not None else {},
    )


def test_proxy_user_trusted_peer_with_header():
    s = _settings()
    req = _request("10.0.0.1", {"X-authentik-username": "alice"})
    assert resolve_proxy_user(req, s) == "alice"


def test_proxy_user_rejected_from_untrusted_peer():
    s = _settings()
    req = _request("1.2.3.4", {"X-authentik-username": "attacker"})
    assert resolve_proxy_user(req, s) is None


def test_proxy_user_none_when_disabled():
    s = _settings(enabled=False)
    req = _request("10.0.0.1", {"X-authentik-username": "alice"})
    assert resolve_proxy_user(req, s) is None


def test_proxy_user_none_without_header():
    s = _settings()
    req = _request("10.0.0.1", {})
    assert resolve_proxy_user(req, s) is None


def test_get_current_user_falls_back_to_forward_auth():
    s = _settings()
    req = _request("10.0.0.1", {"X-authentik-username": "alice"}, session={})
    assert get_current_user(req, s) == "alice"


def test_get_current_user_none_when_no_session_and_no_proxy():
    s = _settings(enabled=False)
    req = _request("10.0.0.1", {}, session={})
    assert get_current_user(req, s) is None
