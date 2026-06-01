"""LLM-assisted regex generation from historical, de-duplicated issues.

Authoring classifier patterns by hand is the slow part of running log-triage.
This module learns candidate patterns from the logs already collected, for both
*ignore* (noise suppression) and *error*/*warning* (failure detection).

The key cost-saver: we never feed raw logs to the LLM. Fingerprint
de-duplication (see :mod:`logtriage.fingerprint`) has already collapsed the logs
into a handful of distinct *issue signatures*, each with a representative
excerpt, a severity and an occurrence count. We hand the model those signatures
and ask for a small set of regexes. Every candidate is then compiled and
**back-tested against the full issue corpus** to measure coverage and, crucially,
how many *protected* issues it would wrongly catch (an ignore rule that silences
real errors, or an error rule that fires on noise). Nothing is ever auto-saved;
the route layer presents ranked candidates for human review.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .llm_client import _call_llm

logger = logging.getLogger(__name__)

VALID_KINDS = ("ignore", "error", "warning")
DEFAULT_MAX_SIGNATURES = 40
DEFAULT_CORPUS_LIMIT = 2000

_ERROR_SEVERITIES = {"ERROR", "CRITICAL"}


@dataclass
class RegexCandidate:
    """One proposed pattern plus its validity and back-test statistics."""

    pattern: str
    kind: str
    rationale: str = ""
    covers: List[int] = field(default_factory=list)  # issue ids the LLM says it covers
    valid: bool = True
    error: Optional[str] = None
    # Back-test stats (computed against the historical issue corpus):
    distinct_issues: int = 0      # how many distinct issues this pattern matches
    match_count: int = 0          # weighted by occurrence_count (volume it touches)
    over_match: int = 0           # protected issues it would wrongly catch (DANGER)

    @property
    def safe(self) -> bool:
        """A candidate is safe to apply when it is valid and catches nothing protected."""
        return self.valid and self.over_match == 0


@dataclass
class GenerationResult:
    candidates: List[RegexCandidate]
    signatures_used: int
    provider: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None


def _norm_severity(value) -> str:
    return (value or "").upper()


def _first_excerpt_line(issue) -> str:
    for line in (getattr(issue, "sample_excerpt", "") or "").splitlines():
        if line.strip():
            return line.strip()[:300]
    return ""


def gather_signatures(
    module_name: str,
    kind: str,
    *,
    limit: int = DEFAULT_MAX_SIGNATURES,
) -> Tuple[list, list]:
    """Return ``(selected, corpus)`` issue lists for a module.

    ``selected`` are the representatives shown to the LLM (intent-filtered, ranked
    by occurrence count). ``corpus`` is the full set of issues for the module,
    used as the back-test ground truth. DB access is imported lazily so this
    module stays importable without sqlalchemy.
    """
    from .webui.db import get_issues, ISSUE_ACTIVE_STATUSES

    corpus = get_issues(module_name=module_name, limit=DEFAULT_CORPUS_LIMIT)

    if kind == "ignore":
        # Noise we want to suppress: explicitly muted/false-positive issues first,
        # then the loudest non-error issues (likely benign chatter).
        muted = [i for i in corpus if i.status in ("muted", "false_positive")]
        muted_ids = {i.id for i in muted}
        chatter = [
            i for i in corpus
            if i.id not in muted_ids and _norm_severity(i.severity) not in _ERROR_SEVERITIES
        ]
        chatter.sort(key=lambda i: i.occurrence_count or 0, reverse=True)
        selected = muted + chatter
    else:
        target = _ERROR_SEVERITIES if kind == "error" else {"WARNING"}
        active = [
            i for i in corpus
            if i.status in ISSUE_ACTIVE_STATUSES and _norm_severity(i.severity) in target
        ]
        active.sort(key=lambda i: i.occurrence_count or 0, reverse=True)
        selected = active

    return selected[: max(1, limit)], corpus


def _build_messages(signatures: List[dict], kind: str) -> List[dict]:
    intent = {
        "ignore": "benign, recurring NOISE lines that should be ignored/suppressed",
        "error": "ERROR-level failures that should be detected",
        "warning": "WARNING-level conditions that should be detected",
    }[kind]

    lines: List[str] = []
    for idx, sig in enumerate(signatures, 1):
        lines.append(
            f"[{idx}] (severity={sig['severity']}, count={sig['count']}) {sig['signature']}"
        )
        if sig.get("excerpt"):
            lines.append(f"      e.g. {sig['excerpt']}")
    listing = "\n".join(lines)

    system = (
        "You write Python `re` regular expressions (used with re.IGNORECASE) that match "
        "FAMILIES of log lines. Generalize volatile tokens (numbers, ids, UUIDs, IPs, "
        "timestamps, hex digests, file paths) using character classes or \\S+ / \\d+, but "
        "KEEP the stable, distinguishing words so each pattern stays specific. Prefer ONE "
        "pattern per distinct family; never write an over-broad catch-all such as `.*`. "
        f"Generate patterns to match {intent}.\n"
        "Return ONLY a JSON array, no prose. Each element must be an object: "
        '{"pattern": "<regex>", "rationale": "<short why>", "covers": [<signature numbers>]}'
    )
    user = (
        "Here are de-duplicated log signatures from history, each prefixed with its "
        "number, severity and occurrence count:\n\n" + listing
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _extract_json_array(content: str) -> Optional[list]:
    """Best-effort extraction of a JSON array from an LLM response.

    Tolerates ```code fences``` and leading/trailing prose by slicing from the
    first ``[`` to the last ``]``.
    """
    text = (content or "").strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, list) else None


def _protected_predicate(kind: str) -> Callable[[object], bool]:
    """Return a predicate marking issues a candidate of ``kind`` must NOT match."""
    from .webui.db import ISSUE_ACTIVE_STATUSES

    if kind == "ignore":
        # An ignore rule must never silence a still-active error/warning issue.
        protected = _ERROR_SEVERITIES | {"WARNING"}
        return lambda i: (
            i.status in ISSUE_ACTIVE_STATUSES and _norm_severity(i.severity) in protected
        )
    if kind == "error":
        return lambda i: _norm_severity(i.severity) not in _ERROR_SEVERITIES
    if kind == "warning":
        return lambda i: _norm_severity(i.severity) != "WARNING"
    return lambda i: False


def backtest(candidates: List[RegexCandidate], corpus: list, kind: str) -> None:
    """Populate coverage / over-match stats on each valid candidate, in place."""
    is_protected = _protected_predicate(kind)
    compiled = []
    for issue in corpus:
        text = f"{getattr(issue, 'signature', '') or ''}\n{getattr(issue, 'sample_excerpt', '') or ''}"
        compiled.append((issue, text))

    for cand in candidates:
        if not cand.valid:
            continue
        try:
            rx = re.compile(cand.pattern, re.IGNORECASE)
        except re.error:
            cand.valid = False
            continue
        matched_ids = set()
        occurrences = 0
        over = 0
        for issue, text in compiled:
            if rx.search(text):
                matched_ids.add(issue.id)
                occurrences += issue.occurrence_count or 0
                if is_protected(issue):
                    over += 1
        cand.distinct_issues = len(matched_ids)
        cand.match_count = occurrences
        cand.over_match = over


def generate_regex_candidates(
    module_name: str,
    kind: str,
    provider,
    *,
    max_signatures: int = DEFAULT_MAX_SIGNATURES,
) -> GenerationResult:
    """Generate, validate and back-test regex candidates for one module/kind.

    ``provider`` is a resolved ``LLMProviderConfig`` (the caller picks it from
    config — e.g. a local Ollama provider). The LLM call goes through the shared
    :func:`logtriage.llm_client._call_llm`, so all provider types and the token
    budget apply. Never saves anything.
    """
    kind = kind if kind in VALID_KINDS else "error"
    selected, corpus = gather_signatures(module_name, kind, limit=max_signatures)
    if not selected:
        return GenerationResult(
            candidates=[],
            signatures_used=0,
            provider=getattr(provider, "name", None),
            error="No historical issues to learn from for this module/kind.",
        )

    signatures = [
        {
            "id": i.id,
            "signature": (getattr(i, "signature", None) or getattr(i, "title", "") or "").strip(),
            "excerpt": _first_excerpt_line(i),
            "severity": _norm_severity(i.severity),
            "count": i.occurrence_count or 0,
        }
        for i in selected
    ]

    payload = {
        "model": provider.model,
        "messages": _build_messages(signatures, kind),
        "temperature": getattr(provider, "temperature", 0.0) or 0.0,
        "top_p": getattr(provider, "top_p", 1.0),
        "max_tokens": getattr(provider, "max_output_tokens", None) or 800,
    }

    try:
        response = _call_llm(provider, payload)
    except Exception as exc:
        return GenerationResult(
            candidates=[],
            signatures_used=len(signatures),
            provider=provider.name,
            model=provider.model,
            error=str(exc),
        )

    message = (response.get("choices") or [{}])[0].get("message", {})
    content = (message.get("content") or "").strip()
    model = response.get("model", provider.model)

    items = _extract_json_array(content)
    if items is None:
        return GenerationResult(
            candidates=[],
            signatures_used=len(signatures),
            provider=provider.name,
            model=model,
            error="LLM did not return a parseable JSON array of patterns.",
        )

    idmap = {idx: s["id"] for idx, s in enumerate(signatures, 1)}
    candidates: List[RegexCandidate] = []
    seen: set = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        pattern = (item.get("pattern") or "").strip()
        if not pattern or pattern in seen:
            continue
        seen.add(pattern)
        cand = RegexCandidate(
            pattern=pattern,
            kind=kind,
            rationale=(item.get("rationale") or "").strip()[:300],
        )
        covers = item.get("covers") or []
        cand.covers = [idmap[c] for c in covers if isinstance(c, int) and c in idmap]
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            cand.valid = False
            cand.error = str(exc)
        candidates.append(cand)

    backtest(candidates, corpus, kind)

    # Rank: valid first, then safest (fewest protected matches), then widest coverage.
    candidates.sort(
        key=lambda c: (0 if c.valid else 1, c.over_match, -c.distinct_issues, -c.match_count)
    )

    return GenerationResult(
        candidates=candidates,
        signatures_used=len(signatures),
        provider=provider.name,
        model=model,
    )
