"""Triage routes: the issue queue, issue detail, and workflow actions."""
from __future__ import annotations

import datetime
import urllib.parse
from typing import Optional

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import RedirectResponse

from ...models import ModuleLLMConfig, Severity
from ...notifications import add_notification
from ..auth import get_current_user
from ..db import (
    delete_findings_matching_regex,
    get_findings_for_issue,
    get_issue_by_id,
    get_issue_sparkline,
    get_issue_sparklines,
    get_issues,
    issue_status_counts,
    issue_category_counts,
    update_issue_status,
    ISSUE_STATUSES,
    ISSUE_ACTIVE_STATUSES,
)
from ..state import STATE
from ..shared import (
    templates,
    build_modules_from_config,
    select_provider_name,
    suggest_regex_from_line,
)
from .. import config_io

router = APIRouter()

SEVERITY_CHOICES = ["CRITICAL", "ERROR", "WARNING"]


def _login_redirect(request):
    return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)


@router.get("/issues", name="issues")
async def issues_list(
    request: Request,
    module: Optional[str] = None,
    status_filter: str = "active",
    severity: Optional[str] = None,
    q: Optional[str] = None,
    category: Optional[str] = None,
    group_by_category: int = 0,
    message: Optional[str] = None,
    error: Optional[str] = None,
):
    if not get_current_user(request, STATE.settings):
        return _login_redirect(request)
    username = get_current_user(request, STATE.settings)

    if status_filter == "active":
        statuses = list(ISSUE_ACTIVE_STATUSES)
    elif status_filter in ("all", "", None):
        statuses = None
    else:
        statuses = [status_filter]

    severities = [severity] if severity else None
    now = datetime.datetime.now(datetime.timezone.utc)
    issues = get_issues(
        module_name=module or None, statuses=statuses, severities=severities,
        search=(q or None), limit=150, now=now, category=(category or None),
    )
    sparklines = get_issue_sparklines(
        [iss.id for iss in issues], buckets=24, bucket_seconds=3600, now=now
    )
    counts = issue_status_counts(module or None)
    category_counts = issue_category_counts(module or None, statuses=statuses)
    modules = sorted(build_modules_from_config(), key=lambda m: m.name.lower())

    # Optional grouping: order the already-fetched (priority-sorted) list by
    # category so the template can render section headers without losing rank.
    grouped = None
    if group_by_category:
        grouped = {}
        for iss in issues:
            raw = getattr(iss, "llm_category", None)
            key = (raw.split(":", 1)[0].strip() if raw else "uncategorized") or "other"
            grouped.setdefault(key, []).append(iss)

    return templates.TemplateResponse(
        "issues.html",
        {
            "request": request,
            "username": username,
            "issues": issues,
            "grouped": grouped,
            "sparklines": sparklines,
            "counts": counts,
            "category_counts": category_counts,
            "category_filter": category or "",
            "group_by_category": bool(group_by_category),
            "modules": modules,
            "current_module": module or "",
            "status_filter": status_filter or "active",
            "severity_filter": severity or "",
            "search": q or "",
            "severity_choices": SEVERITY_CHOICES,
            "status_choices": list(ISSUE_STATUSES),
            "db_status": STATE.db_status,
            "message": message,
            "error": error,
        },
    )


@router.get("/issues/{issue_id}", name="issue_detail")
async def issue_detail(request: Request, issue_id: int, message: Optional[str] = None, error: Optional[str] = None):
    username = get_current_user(request, STATE.settings)
    if not username:
        return _login_redirect(request)

    issue = get_issue_by_id(issue_id)
    if issue is None:
        return RedirectResponse(
            url=request.app.url_path_for("issues") + "?error=Issue+not+found",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    now = datetime.datetime.now(datetime.timezone.utc)
    spark_24h = get_issue_sparkline(issue_id, buckets=24, bucket_seconds=3600, now=now)
    spark_14d = get_issue_sparkline(issue_id, buckets=14, bucket_seconds=86400, now=now)
    occurrences = get_findings_for_issue(issue_id, limit=50)

    provider_name = select_provider_name(None)
    sample_first = next((ln for ln in (issue.sample_excerpt or "").splitlines() if ln.strip()), "")
    suggested_ignore = suggest_regex_from_line(sample_first) if sample_first else ""

    return templates.TemplateResponse(
        "issue_detail.html",
        {
            "request": request,
            "username": username,
            "issue": issue,
            "spark_24h": spark_24h,
            "spark_14d": spark_14d,
            "occurrences": occurrences,
            "status_choices": list(ISSUE_STATUSES),
            "db_status": STATE.db_status,
            "can_analyze": bool(getattr(STATE.llm_defaults, "enabled", False) and provider_name),
            "suggested_ignore": suggested_ignore,
            "message": message,
            "error": error,
        },
    )


@router.post("/issues/{issue_id}/status", name="issue_set_status")
async def issue_set_status(request: Request, issue_id: int, new_status: str = Form(...), redirect_to: str = Form("detail")):
    if not get_current_user(request, STATE.settings):
        return _login_redirect(request)

    ok = False
    try:
        ok = update_issue_status(issue_id, new_status)
    except Exception as exc:
        add_notification("error", "Issue status update failed", str(exc))

    suffix = "?message=Status+updated" if ok else "?error=Update+failed"
    if redirect_to == "list":
        url = request.app.url_path_for("issues") + suffix
    else:
        url = request.app.url_path_for("issue_detail", issue_id=issue_id) + suffix
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)


@router.post("/issues/{issue_id}/analyze", name="issue_analyze")
async def issue_analyze(request: Request, issue_id: int):
    if not get_current_user(request, STATE.settings):
        return _login_redirect(request)

    issue = get_issue_by_id(issue_id)
    if issue is None:
        return RedirectResponse(url=request.app.url_path_for("issues") + "?error=Issue+not+found", status_code=status.HTTP_303_SEE_OTHER)

    detail_url = request.app.url_path_for("issue_detail", issue_id=issue_id)

    modules = {m.name: m for m in build_modules_from_config()}
    module = modules.get(issue.module_name)
    provider_name = None
    if module is not None and getattr(module, "llm", None) is not None:
        provider_name = module.llm.provider_name
    provider_name = provider_name or select_provider_name(None)
    provider_cfg = STATE.llm_defaults.providers.get(provider_name) if provider_name else None
    if provider_cfg is None:
        return RedirectResponse(url=detail_url + "?error=No+LLM+provider+configured", status_code=status.HTTP_303_SEE_OTHER)

    from ...enrichment import analyze_issue

    temp_module_llm = ModuleLLMConfig(
        enabled=True, provider_name=provider_name,
        min_severity=Severity.WARNING, max_excerpt_lines=provider_cfg.max_excerpt_lines,
    )
    try:
        wrote = analyze_issue(
            issue, STATE.llm_defaults, temp_module_llm,
            rag_client=STATE.rag_client, module_name=issue.module_name, force=True,
        )
        msg = "Analysis updated" if wrote else "No analysis produced"
        return RedirectResponse(url=detail_url + f"?message={msg.replace(' ', '+')}", status_code=status.HTTP_303_SEE_OTHER)
    except Exception as exc:
        add_notification("error", "Issue analysis failed", str(exc))
        return RedirectResponse(url=detail_url + "?error=Analysis+failed", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/issues/{issue_id}/ignore", name="issue_ignore")
async def issue_ignore(request: Request, issue_id: int, regex_value: str = Form("")):
    if not get_current_user(request, STATE.settings):
        return _login_redirect(request)

    issue = get_issue_by_id(issue_id)
    if issue is None:
        return RedirectResponse(url=request.app.url_path_for("issues") + "?error=Issue+not+found", status_code=status.HTTP_303_SEE_OTHER)
    detail_url = request.app.url_path_for("issue_detail", issue_id=issue_id)

    if not STATE.db_status.get("connected"):
        return RedirectResponse(url=detail_url + "?error=Database+not+connected", status_code=status.HTTP_303_SEE_OTHER)

    regex_value = (regex_value or "").strip()
    if not regex_value:
        sample_first = next((ln for ln in (issue.sample_excerpt or "").splitlines() if ln.strip()), "")
        regex_value = suggest_regex_from_line(sample_first) if sample_first else ""
    if not regex_value:
        return RedirectResponse(url=detail_url + "?error=No+sample+to+build+an+ignore+rule", status_code=status.HTTP_303_SEE_OTHER)

    # app.py registers the reload callback so its globals stay mirrored.
    err = config_io.add_ignore_regex_to_pipeline(
        issue.pipeline_name, regex_value, reload=STATE.reload_callback
    )
    if err:
        return RedirectResponse(url=detail_url + "?error=" + urllib.parse.quote(err), status_code=status.HTTP_303_SEE_OTHER)

    try:
        update_issue_status(issue_id, "false_positive")
        delete_findings_matching_regex(regex_value, pipeline_name=issue.pipeline_name)
    except Exception as exc:
        add_notification("warning", "Ignore rule saved", f"but cleanup failed: {exc}")

    return RedirectResponse(
        url=request.app.url_path_for("issues") + "?message=Ignore+rule+added%2C+issue+suppressed",
        status_code=status.HTTP_303_SEE_OTHER,
    )
