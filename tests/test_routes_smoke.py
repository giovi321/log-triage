"""End-to-end smoke tests for non-template routes via FastAPI TestClient.

Template-rendering routes are covered elsewhere / on the deployment runtime;
here we exercise the JSON/text/redirect routes that verify middleware, auth
gating, and route wiring end-to-end. Skipped when the webui extras are absent.
"""
import os

os.environ.setdefault("LOGTRIAGE_CONFIG", "this-config-does-not-exist.yaml")

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from logtriage.webui import app as appmod


@pytest.fixture(scope="module")
def client():
    with TestClient(appmod.app) as c:
        yield c


def test_metrics_endpoint_serves_prometheus_text(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "logtriage_findings_total" in r.text
    assert "logtriage_worker_running" in r.text


def test_root_redirects_to_login_when_unauthenticated(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303


def test_events_requires_authentication(client):
    r = client.get("/events")
    assert r.status_code == 401


def test_issues_requires_auth_redirect(client):
    r = client.get("/issues", follow_redirects=False)
    assert r.status_code == 303


def test_issue_detail_requires_auth_redirect(client):
    r = client.get("/issues/1", follow_redirects=False)
    assert r.status_code == 303
