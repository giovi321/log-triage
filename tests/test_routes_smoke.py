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


def test_account_redirects_when_unauthenticated(client):
    r = client.get("/account", follow_redirects=False)
    assert r.status_code == 303


def test_user_management_routes_are_protected(client):
    # The new user-CRUD POSTs must never act for an unauthenticated request:
    # CSRF rejects them (400) before the handler, and even past CSRF the
    # handler redirects to login (303). Anything but 2xx means "did not act".
    for path, data in (
        ("/account/users/create", {"new_username": "x", "new_password": "y" * 8, "confirm_password": "y" * 8}),
        ("/account/users/reset", {"target_username": "x", "new_password": "y" * 8}),
        ("/account/users/delete", {"target_username": "x"}),
    ):
        r = client.post(path, data=data, follow_redirects=False)
        assert r.status_code in (303, 400, 401, 403)


def test_oidc_login_redirects_when_not_configured(client):
    r = client.get("/login/oidc", follow_redirects=False)
    assert r.status_code == 303
    assert "/login" in r.headers["location"]
