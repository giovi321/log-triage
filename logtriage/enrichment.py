"""Per-issue LLM analysis with caching.

Each de-duplicated issue is analyzed by the LLM **once per signature**: the
result (summary + citations) is cached on the issue row keyed by its
fingerprint. Browsing thousands of occurrences therefore costs zero extra LLM
calls, and the summary stays consistent everywhere it is shown. When the
signature changes (e.g. the rule is edited) the cache is considered stale and
the issue is re-analyzed.

This is the cost-saving counterpart to the old per-finding enrichment: instead
of one LLM call per matching log line, we make one call per unique problem.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

from .models import Finding, GlobalLLMConfig, ModuleConfig, ModuleLLMConfig
from .llm_client import resolve_provider, _select_max_tokens, _call_llm
from .webui.db import (
    get_issues_needing_llm,
    update_issue_llm,
    update_issue_llm_error,
)

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are a log-triage assistant. You are given a recurring log issue (a group of "
    "identical-shape log lines). In 2-4 sentences explain what it most likely means and "
    "the single most useful next action to investigate or resolve it. Be concrete and avoid "
    "filler. If documentation context is provided below, ground your answer in it and cite "
    "sources using their bracketed reference numbers."
)


def build_finding_from_issue(issue) -> Finding:
    """Adapt an IssueRecord into a Finding for RAG retrieval / payload building."""
    return Finding(
        file_path=Path("issue"),
        pipeline_name=getattr(issue, "pipeline_name", None) or "",
        finding_index=0,
        severity=issue.severity_enum,
        message=getattr(issue, "title", "") or "",
        line_start=0,
        line_end=0,
        rule_id=getattr(issue, "rule_id", None),
        excerpt=list(issue.sample_excerpt_as_list),
        needs_llm=True,
    )


def _retrieve_rag(rag_client, finding: Finding, module_name: Optional[str]):
    """Return (context_text, citations) from the RAG service, best-effort."""
    if not rag_client or not module_name:
        return "", []
    try:
        retrieval = rag_client.retrieve_for_finding(finding, module_name)
    except Exception as exc:  # pragma: no cover - network/IO dependent
        logger.debug("RAG retrieval failed for issue analysis: %s", exc)
        return "", []
    if not retrieval or not getattr(retrieval, "chunks", None):
        return "", []

    parts = ["\n\n--- Relevant documentation ---"]
    citations = []
    for i, chunk in enumerate(retrieval.chunks, 1):
        parts.append(f"\n[{i}] {chunk.heading}\n{chunk.content}")
        citations.append(f"[{i}] {chunk.heading} ({Path(chunk.file_path).name})")
    return "\n".join(parts), citations


def analyze_issue(
    issue,
    llm_defaults: GlobalLLMConfig,
    module_llm: ModuleLLMConfig,
    rag_client=None,
    module_name: Optional[str] = None,
    force: bool = False,
) -> bool:
    """Analyze a single issue with the LLM and cache the result.

    Returns True if a fresh analysis was written. Skips (returns False) when the
    issue already has a cached analysis for its current fingerprint unless
    ``force`` is set.
    """
    if not force and getattr(issue, "has_llm_analysis", False):
        return False

    provider = resolve_provider(llm_defaults, module_llm)
    if provider is None:
        return False

    finding = build_finding_from_issue(issue)
    rag_context, citations = _retrieve_rag(rag_client, finding, module_name)

    title = getattr(issue, "title", "") or ""
    excerpt = "\n".join(issue.sample_excerpt_as_list) or title
    user_text = (
        f"Issue signature: {title}\n"
        f"Severity: {getattr(issue, 'severity', 'UNKNOWN')}\n"
        f"Occurrences: {getattr(issue, 'occurrence_count', 0)}\n\n"
        f"Representative log excerpt:\n{excerpt}"
    )
    # The large, stable prefix (instructions + docs) goes in the system block so
    # it can be prompt-cached across analyses that share documentation.
    system_text = SYSTEM_INSTRUCTION + (rag_context if rag_context else "")
    cache_system = bool(rag_context)

    max_tokens = _select_max_tokens(module_llm, provider, llm_defaults)

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

    chat_payload = {
        "model": provider.model,
        "messages": messages,
        "temperature": provider.temperature,
        "top_p": provider.top_p,
        "max_tokens": max_tokens,
        "cache_system": cache_system,
    }

    try:
        response = _call_llm(provider, chat_payload)
    except Exception as exc:
        update_issue_llm_error(issue.id, str(exc), provider=provider.name, model=provider.model)
        return False

    message = (response.get("choices") or [{}])[0].get("message", {})
    content = (message.get("content") or "").strip()
    usage = response.get("usage") or {}

    if not content:
        update_issue_llm_error(
            issue.id, "Empty LLM response", provider=provider.name, model=provider.model
        )
        return False

    update_issue_llm(
        issue.id,
        provider=provider.name,
        model=response.get("model", provider.model),
        content=content,
        citations=citations or None,
        fingerprint=getattr(issue, "fingerprint", None),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )
    return True


def analyze_pending_issues(
    modules_by_name: Dict[str, ModuleConfig],
    llm_defaults: GlobalLLMConfig,
    rag_client=None,
    limit: int = 50,
    force: bool = False,
) -> int:
    """Analyze active issues whose cached analysis is missing or stale.

    Only issues belonging to a module with LLM enabled are auto-analyzed; the
    count of freshly-written analyses is returned. This is the function the
    Phase 4 worker drives on a schedule.
    """
    if not getattr(llm_defaults, "enabled", False):
        return 0

    pending = get_issues_needing_llm(module_names=list(modules_by_name.keys()), limit=limit)
    written = 0
    for issue in pending:
        module = modules_by_name.get(issue.module_name)
        if module is None:
            continue
        module_llm = getattr(module, "llm", None)
        if module_llm is None or not getattr(module_llm, "enabled", False):
            continue
        try:
            if analyze_issue(
                issue,
                llm_defaults,
                module_llm,
                rag_client=rag_client,
                module_name=issue.module_name,
                force=force,
            ):
                written += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Issue analysis failed for issue %s: %s", getattr(issue, "id", "?"), exc)
    return written
