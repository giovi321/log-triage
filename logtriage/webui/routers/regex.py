"""Regex lab routes: interactive pattern testing and save-to-pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import RedirectResponse

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from ..auth import get_current_user
from ..db import get_module_stats, get_recent_findings_for_module
from ..ingestion_status import _derive_ingestion_status
from ..regex_utils import (
    _compile_regex_with_feedback,
    _filter_finding_intro_lines,
    _lint_regex_input,
    _prepare_sample_lines,
)
from ..state import STATE
from ..shared import (
    templates,
    build_modules_from_config,
    normalize_sample_source as _normalize_sample_source,
    sample_source_label,
    sample_source_options,
    suggest_regex_from_line as _suggest_regex_from_line,
)
from ..logs_shared import (
    REGEX_WIZARD_STEPS,
    _get_regex_state,
    _update_regex_state,
    _regex_wizard_metadata,
    _regex_step_hints,
    _build_all_regex_hints,
    _evaluate_regex_against_lines,
    _tail_lines,
    _get_sample_lines_for_module,
)
from .. import config_io

router = APIRouter()


def _regex_context(
    request: Request,
    username: str,
    modules: List[ModuleConfig],
    module_obj,
    *,
    sample_lines: List[str],
    regex_value: str,
    regex_kind: str,
    matches: List[int],
    error: Optional[str],
    message: Optional[str],
    sample_source: str,
    regex_issues: Optional[List[str]] = None,
    active_step: str = "pick",
    wizard: Optional[Dict[str, Any]] = None,
    step_hints: Optional[Dict[str, List[Dict[str, str]]]] = None,
    recent_findings: Optional[List[Any]] = None,
    open_findings_count: Optional[int] = None,
):
    normalized_source = _normalize_sample_source(sample_source)
    return {
        "request": request,
        "username": username,
        "modules": modules,
        "current_module": module_obj,
        "sample_lines": sample_lines,
        "regex_value": regex_value,
        "regex_kind": regex_kind,
        "matches": matches,
        "error": error,
        "message": message,
        "sample_source": normalized_source,
        "sample_source_label": sample_source_label(normalized_source),
        "sample_options": sample_source_options(),
        "regex_issues": regex_issues,
        "wizard": wizard or _regex_wizard_metadata(active_step),
        "step_hints": step_hints or _build_all_regex_hints(),
        "active_step": active_step,
        "recent_findings": recent_findings or [],
        "open_findings_count": open_findings_count,
        "STATE.db_status": STATE.db_status,
    }


@router.get("/regex", name="regex_lab")
async def regex_lab(
    request: Request,
    module: Optional[str] = None,
    sample_source: str = "tail",
):
    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    stored_state = _get_regex_state(request)
    module = module or stored_state.get("module")
    safe_sample_source = sample_source if sample_source in {"errors", "tail"} else stored_state.get("sample_source", "tail")
    safe_sample_source = safe_sample_source if safe_sample_source in {"errors", "tail"} else "tail"

    modules = build_modules_from_config()
    stats = get_module_stats(modules)
    ingestion_status = (
        _derive_ingestion_status(modules, freshness_minutes=STATE.settings.staleness_minutes)
        if modules else None
    )
    module_obj = None
    if modules:
        if module:
            module_obj = next((m for m in modules if m.name == module), None)
        if module_obj is None:
            module_obj = modules[0]

    raw_sample_lines: List[str] = []
    sample_start_line: int = 1
    sample_error: Optional[str] = None
    safe_sample_source = _normalize_sample_source(sample_source)
    if module_obj is not None:
        raw_sample_lines, sample_start_line, _, sample_error = _get_sample_lines_for_module(
            module_obj, safe_sample_source, max_lines=200
        )

    filtered_sample_lines = _filter_finding_intro_lines(raw_sample_lines)
    prepared_lines = _prepare_sample_lines(filtered_sample_lines, first_line_number=sample_start_line)
    matches, evaluation_error = _evaluate_regex_against_lines(
        stored_state.get("regex_value", ""), filtered_sample_lines, first_line_number=sample_start_line
    )
    active_step = stored_state.get("step", "pick")
    _update_regex_state(
        request,
        module=module_obj.name if module_obj else None,
        sample_source=safe_sample_source,
        matches=matches,
        step=active_step,
        regex_value=stored_state.get("regex_value", ""),
        regex_kind=stored_state.get("regex_kind", "error"),
    )

    # Fetch findings for the module
    recent_findings = []
    if module_obj and STATE.db_status.get("connected"):
        recent_findings = get_recent_findings_for_module(module_obj.name, limit=50)

    wizard = _regex_wizard_metadata(active_step)
    step_hints = _build_all_regex_hints()
    return templates.TemplateResponse(
        "regex.html",
        _regex_context(
            request,
            username,
            modules,
            module_obj,
            sample_lines=prepared_lines,
            regex_value=stored_state.get("regex_value", ""),
            regex_kind=stored_state.get("regex_kind", "error"),
            matches=matches,
            error=sample_error or evaluation_error,
            message=None,
            sample_source=safe_sample_source,
            wizard=wizard,
            step_hints=step_hints,
            active_step=active_step,
            recent_findings=recent_findings,
        ),
    )


@router.post("/regex/test", name="regex_test")
async def regex_test(
    request: Request,
    module: str = Form(...),
    regex_value: str = Form(...),
    regex_kind: str = Form("error"),
    sample_source: str = Form("tail"),
):
    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    modules = build_modules_from_config()
    safe_sample_source = sample_source if sample_source in {"errors", "tail"} else "tail"
    module_obj = next((m for m in modules if m.name == module), None)
    raw_sample_lines: List[str] = []
    sample_start_line: int = 1
    sample_error: Optional[str] = None
    safe_sample_source = _normalize_sample_source(sample_source)
    if module_obj is not None:
        raw_sample_lines, sample_start_line, _, sample_error = _get_sample_lines_for_module(
            module_obj, safe_sample_source, max_lines=200
        )
    filtered_sample_lines = _filter_finding_intro_lines(raw_sample_lines)
    prepared_lines = _prepare_sample_lines(filtered_sample_lines, first_line_number=sample_start_line)

    regex_issues = _lint_regex_input(regex_value)
    matches: List[int] = []
    compiled, compile_error = (None, None)
    if not regex_issues:
        compiled, compile_error = _compile_regex_with_feedback(regex_value)
        if compile_error:
            regex_issues.append(compile_error)

    error_msg: Optional[str] = None
    if compiled:
        for entry in prepared_lines:
            if compiled.search(entry.get("full", "")):
                matches.append(entry.get("index", 0))
    elif regex_issues:
        error_msg = "Resolve the regex issues before testing."

    wizard = _regex_wizard_metadata("test")
    step_hints = _build_all_regex_hints()
    _update_regex_state(
        request,
        module=module_obj.name if module_obj else None,
        sample_source=safe_sample_source,
        regex_value=regex_value,
        regex_kind=regex_kind,
        matches=matches,
        step="test",
    )
    return templates.TemplateResponse(
        "regex.html",
        _regex_context(
            request,
            username,
            modules,
            module_obj,
            sample_lines=prepared_lines,
            regex_value=regex_value,
            regex_kind=regex_kind,
            matches=matches,
            error=error_msg or sample_error,
            message=None,
            sample_source=safe_sample_source,
            regex_issues=regex_issues,
            wizard=wizard,
            step_hints=step_hints,
            active_step="test",
        ),
    )


@router.post("/regex/suggest", name="regex_suggest")
async def regex_suggest(
    request: Request,
    module: str = Form(...),
    sample_selection: str = Form(""),
    regex_kind: str = Form("error"),
    sample_source: str = Form("tail"),
):
    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    modules = build_modules_from_config()
    safe_sample_source = sample_source if sample_source in {"errors", "tail"} else "tail"
    module_obj = next((m for m in modules if m.name == module), None)
    raw_sample_lines: List[str] = []
    sample_start_line: int = 1
    sample_error: Optional[str] = None
    safe_sample_source = _normalize_sample_source(sample_source)
    if module_obj is not None:
        raw_sample_lines, sample_start_line, _, sample_error = _get_sample_lines_for_module(
            module_obj, safe_sample_source, max_lines=200
        )

    filtered_sample_lines = _filter_finding_intro_lines(raw_sample_lines)
    prepared_lines = _prepare_sample_lines(filtered_sample_lines, first_line_number=sample_start_line)
    selection = (sample_selection or "").strip()

    if not selection:
        wizard = _regex_wizard_metadata("pick")
        step_hints = _build_all_regex_hints()
        _update_regex_state(
            request,
            module=module_obj.name if module_obj else None,
            sample_source=safe_sample_source,
            regex_value=sample_selection,
            regex_kind=regex_kind,
            matches=[],
            step="pick",
        )

        return templates.TemplateResponse(
            "regex.html",
            _regex_context(
                request,
                username,
                modules,
                module_obj,
                sample_lines=prepared_lines,
                regex_value=sample_selection,
                regex_kind=regex_kind,
                matches=[],
                error=sample_error or "Highlight sample text to draft a regex.",
                message=None,
                sample_source=safe_sample_source,
                wizard=wizard,
                step_hints=step_hints,
                active_step="pick",
            ),
        )

    suggestion = _suggest_regex_from_line(selection)
    wizard = _regex_wizard_metadata("draft")
    step_hints = _build_all_regex_hints()
    _update_regex_state(
        request,
        module=module_obj.name if module_obj else None,
        sample_source=safe_sample_source,
        regex_value=suggestion,
        regex_kind=regex_kind,
        matches=[],
        step="draft",
    )

    return templates.TemplateResponse(
        "regex.html",
        _regex_context(
            request,
            username,
            modules,
            module_obj,
            sample_lines=prepared_lines,
            regex_value=suggestion,
            regex_kind=regex_kind,
            matches=[],
            error=sample_error,
            message="Suggested regex generated from selected text.",
            sample_source=safe_sample_source,
            wizard=wizard,
            step_hints=step_hints,
            active_step="draft",
        ),
    )


@router.post("/regex/save", name="regex_save")
async def regex_save(
    request: Request,
    module: str = Form(...),
    regex_value: str = Form(...),
    regex_kind: str = Form("error"),
    sample_source: str = Form("tail"),
):

    safe_sample_source = _normalize_sample_source(sample_source)

    username = get_current_user(request, STATE.settings)
    if not username:
        return RedirectResponse(url=request.app.url_path_for("login_form"), status_code=status.HTTP_303_SEE_OTHER)

    modules = build_modules_from_config()
    module_obj = next((m for m in modules if m.name == module), None)
    if module_obj is None:
        return RedirectResponse(url=request.app.url_path_for("regex_lab"), status_code=status.HTTP_303_SEE_OTHER)

    tail_lines_result, sample_start_line, _ = _tail_lines(Path(module_obj.path), max_lines=200)
    prepared_lines = _prepare_sample_lines(
        _filter_finding_intro_lines(tail_lines_result),
        first_line_number=sample_start_line
    )
    wizard = _regex_wizard_metadata("save")
    step_hints = _build_all_regex_hints()
    _update_regex_state(
        request,
        module=module_obj.name if module_obj else None,
        sample_source=safe_sample_source,
        regex_value=regex_value,
        regex_kind=regex_kind,
        matches=[],
        step="save",
    )
    regex_issues = _lint_regex_input(regex_value)
    if regex_issues:
        return templates.TemplateResponse(
            "regex.html",
            _regex_context(
                request,
                username,
                modules,
                module_obj,
                sample_lines=prepared_lines,
                regex_value=regex_value,
                regex_kind=regex_kind,
                matches=[],
                error="Resolve the regex issues before saving.",
                message=None,
                sample_source=safe_sample_source,
                regex_issues=regex_issues,
                wizard=wizard,
                step_hints=step_hints,
                active_step="save",
            ),
        )

    if not getattr(module_obj, "pipeline_name", None):
        return templates.TemplateResponse(
            "regex.html",
            _regex_context(
                request,
                username,
                modules,
                module_obj,
                sample_lines=prepared_lines,
                regex_value=regex_value,
                regex_kind=regex_kind,
                matches=[],
                error="Module has no explicit pipeline; cannot save regex automatically.",
                message=None,
                sample_source=safe_sample_source,
                wizard=wizard,
                step_hints=step_hints,
                active_step="save",
            ),
        )

    try:
        cfg_dict = yaml.safe_load(STATE.config_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return templates.TemplateResponse(
            "regex.html",
            _regex_context(
                request,
                username,
                modules,
                module_obj,
                sample_lines=prepared_lines,
                regex_value=regex_value,
                regex_kind=regex_kind,
                matches=[],
                error=f"Failed to read config: {e}",
                message=None,
                sample_source=safe_sample_source,
                wizard=wizard,
                step_hints=step_hints,
                active_step="save",
            ),
        )

    pipelines = cfg_dict.get("pipelines", []) or []
    pipeline_dict = None
    for p in pipelines:
        if p.get("name") == module_obj.pipeline_name:
            pipeline_dict = p
            break

    if pipeline_dict is None:
        return templates.TemplateResponse(
            "regex.html",
            _regex_context(
                request,
                username,
                modules,
                module_obj,
                sample_lines=prepared_lines,
                regex_value=regex_value,
                regex_kind=regex_kind,
                matches=[],
                error=f"Pipeline {module_obj.pipeline_name} not found in config.",
                message=None,
                sample_source=safe_sample_source,
                wizard=wizard,
                step_hints=step_hints,
                active_step="save",
            ),
        )

    classifier = pipeline_dict.setdefault("classifier", {})
    key_map = {
        "error": "error_regexes",
        "warning": "warning_regexes",
        "ignore": "ignore_regexes",
    }
    key = key_map.get(regex_kind, "error_regexes")
    lst = classifier.get(key)
    if lst is None or not isinstance(lst, list):
        lst = []
        classifier[key] = lst

    if regex_value not in lst:
        lst.append(regex_value)

    try:
        config_io.save_config_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    except Exception as e:
        return templates.TemplateResponse(
            "regex.html",
            _regex_context(
                request,
                username,
                modules,
                module_obj,
                sample_lines=prepared_lines,
                regex_value=regex_value,
                regex_kind=regex_kind,
                matches=[],
                error=f"Failed to write config: {e}",
                message=None,
                sample_source=safe_sample_source,
                wizard=wizard,
                step_hints=step_hints,
                active_step="save",
            ),
        )

    (STATE.reload_callback or config_io.reload_from_disk)()

    return templates.TemplateResponse(
        "regex.html",
        _regex_context(
            request,
            username,
            modules,
            module_obj,
            sample_lines=prepared_lines,
            regex_value=regex_value,
            regex_kind=regex_kind,
            matches=[],
            error=None,
            message=f"Regex added to classifier.{key} for pipeline {module_obj.pipeline_name}.",
            sample_source=safe_sample_source,
            wizard=wizard,
            step_hints=step_hints,
            active_step="save",
        ),
    )
