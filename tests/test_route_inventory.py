"""Route-inventory guard for the app.py → routers refactor.

Pins the exact set of (path, methods, name) tuples so the router split (and any
later route change) can't silently drop, rename, or duplicate a route. If you
intentionally add/remove a route, update EXPECTED in the same commit.
"""
import pytest

pytest.importorskip("fastapi")

from logtriage.webui import app as appmod


EXPECTED = {
    ("/", ("GET",), "dashboard"),
    ("/account", ("GET",), "account"),
    ("/account/password", ("POST",), "change_password"),
    ("/account/users/create", ("POST",), "user_create"),
    ("/account/users/delete", ("POST",), "user_delete"),
    ("/account/users/reset", ("POST",), "user_reset_password"),
    ("/ai-logs", ("GET",), "ai_logs"),
    ("/api/llm/test", ("POST",), "api_llm_test"),
    ("/api/log-lines", ("GET",), "api_log_lines"),
    ("/api/rag/progress", ("GET",), "rag_progress"),
    ("/api/rag/reindex/{repo_id}", ("POST",), "reindex_rag_repo"),
    ("/api/rag/scan-docs", ("POST",), "api_scan_docs"),
    ("/api/rag/status", ("GET",), "get_rag_status"),
    ("/auth/callback", ("GET",), "oidc_callback"),
    ("/config/edit", ("GET",), "edit_config"),
    ("/config/edit", ("POST",), "edit_config_post"),
    ("/events", ("GET",), "events"),
    ("/issues", ("GET",), "issues"),
    ("/issues/{issue_id}", ("GET",), "issue_detail"),
    ("/issues/{issue_id}/analyze", ("POST",), "issue_analyze"),
    ("/issues/{issue_id}/ignore", ("POST",), "issue_ignore"),
    ("/issues/{issue_id}/status", ("POST",), "issue_set_status"),
    ("/llm/query", ("POST",), "llm_query"),
    ("/llm/query_finding", ("POST",), "llm_query_finding"),
    ("/login", ("GET",), "login_form"),
    ("/login", ("POST",), "login_form_post"),
    ("/login/oidc", ("GET",), "login_oidc"),
    ("/logout", ("GET",), "logout"),
    ("/logs/db/flush", ("POST",), "flush_logs_db"),
    ("/logs/db/flush-module", ("POST",), "flush_module_logs"),
    ("/logs/finding/delete", ("POST",), "delete_finding"),
    ("/logs/finding/delete-multiple", ("POST",), "delete_selected_findings"),
    ("/logs/finding/false_positive", ("POST",), "mark_false_positive"),
    ("/logs/finding/manual", ("POST",), "create_manual_finding"),
    ("/logs/finding/opinion", ("POST",), "record_finding_opinion"),
    ("/logs/finding/severity", ("POST",), "change_finding_severity"),
    ("/metrics", ("GET",), "metrics"),
    ("/regex", ("GET",), "regex_lab"),
    ("/regex/save", ("POST",), "regex_save"),
    ("/regex/suggest", ("POST",), "regex_suggest"),
    ("/regex/test", ("POST",), "regex_test"),
}


def _current():
    out = set()
    for r in appmod.app.routes:
        methods = getattr(r, "methods", None)
        if not methods:
            continue  # skip mounts / static
        verbs = tuple(sorted(m for m in methods if m not in ("HEAD", "OPTIONS")))
        if not verbs:
            continue
        out.add((r.path, verbs, r.name))
    # Ignore FastAPI's built-in docs routes (not part of our app surface).
    return {t for t in out if t[2] not in ("swagger_ui_html", "swagger_ui_redirect", "redoc_html", "openapi")}


def test_route_inventory_matches_expected():
    current = _current()
    missing = EXPECTED - current
    added = current - EXPECTED
    assert not missing, f"routes disappeared: {sorted(missing)}"
    assert not added, f"unexpected new routes: {sorted(added)}"
