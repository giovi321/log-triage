"""LLM-assisted regex discovery from raw log lines.

Authoring classifier patterns by hand is the slow part of running log-triage.
This module reads a sample of *raw* log lines for a module and asks an LLM which
lines represent errors/warnings worth capturing (or benign noise worth ignoring),
then proposes a regex for each family.

Crucially it works from the **raw log**, not from already-classified issues: an
issue only exists because an existing rule already matched it, so learning from
issues can never surface a problem you are not already catching. Working from the
log lets the model find *uncaptured* errors.

Cost control: near-identical lines are collapsed into *families* via
:func:`logtriage.fingerprint.normalize` (the same volatility-stripping used for
de-duplication), so even a 2000-line sample becomes a few dozen representatives.
Every proposed pattern is then compiled and back-tested against the full sample:
how many lines it would match, how many of those are *new* (not already caught by
an existing error/warning rule), and — for ignore rules — how many are lines that
are currently classified as real problems (the dangerous over-match). Nothing is
ever auto-saved.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .fingerprint import normalize
from .llm_client import _call_llm

logger = logging.getLogger(__name__)

VALID_KINDS = ("ignore", "error", "warning")
DEFAULT_FAMILY_CAP = 80          # max distinct line-families sent to the LLM
DEFAULT_EXAMPLE_LINES = 3        # matched examples shown per candidate
_MAX_LINE_CHARS = 500


@dataclass
class RegexCandidate:
    """One proposed pattern plus its validity and back-test statistics."""

    pattern: str
    kind: str
    rationale: str = ""
    valid: bool = True
    error: Optional[str] = None
    # Back-test stats against the full raw sample:
    match_count: int = 0     # total sample lines this pattern matches
    new_matches: int = 0     # matched lines NOT already caught by an existing error/warning rule
    over_match: int = 0      # ignore-kind only: matched lines currently classified as a real problem
    examples: List[str] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        """Safe to apply: valid and (for ignore) silences nothing currently flagged as a problem."""
        return self.valid and self.over_match == 0


@dataclass
class GenerationResult:
    candidates: List[RegexCandidate]
    lines_sampled: int
    families: int                  # distinct line-families found in the sample
    families_omitted: int = 0      # families dropped from the prompt when over the cap (disclosed, not silent)
    provider: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None


def _clean(line: str) -> str:
    return (line or "").rstrip("\n").rstrip("\r")[:_MAX_LINE_CHARS]


def _compile_all(patterns) -> List["re.Pattern[str]"]:
    compiled = []
    for p in patterns or []:
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error:
            continue
    return compiled


def _dedupe_families(
    lines: List[str],
    problem_rx: List["re.Pattern[str]"],
    *,
    cap: int,
) -> Tuple[List[dict], int, int]:
    """Collapse raw lines into normalized families.

    Returns ``(representatives, total_families, omitted)``. Representatives are
    sorted so families *not* already matched by an existing error/warning rule
    come first (the discovery value), then by frequency. When there are more
    families than ``cap`` the tail is dropped and counted, never silently.
    """
    families: Dict[str, dict] = {}
    for raw in lines:
        line = _clean(raw)
        if not line.strip():
            continue
        key = normalize(line) or line
        fam = families.get(key)
        if fam is None:
            known = any(rx.search(line) for rx in problem_rx)
            families[key] = {"example": line, "count": 1, "known": known}
        else:
            fam["count"] += 1

    ordered = sorted(
        families.values(),
        key=lambda f: (f["known"], -f["count"]),  # unknown-first, then most frequent
    )
    total = len(ordered)
    selected = ordered[: max(1, cap)]
    omitted = max(0, total - len(selected))
    return selected, total, omitted


def _build_messages(representatives: List[dict], kind: str) -> List[dict]:
    intent = {
        "error": (
            "identify which lines indicate ERRORS or failures that should be detected, and "
            "propose a regex to capture each distinct family. Ignore routine INFO/DEBUG and "
            "clearly benign lines."
        ),
        "warning": (
            "identify which lines indicate WARNING conditions worth flagging, and propose a "
            "regex to capture each distinct family. Ignore routine INFO/DEBUG and benign lines."
        ),
        "ignore": (
            "identify benign, recurring NOISE lines that are safe to ignore, and propose a regex "
            "to suppress each family. Do NOT propose any pattern that could match a genuine error "
            "or failure."
        ),
    }[kind]

    listing = "\n".join(
        f"[{i}] (x{rep['count']}) {rep['example']}"
        for i, rep in enumerate(representatives, 1)
    )

    system = (
        "You write Python `re` regular expressions (used with re.IGNORECASE) that match log lines. "
        "Generalize volatile tokens (numbers, ids, UUIDs, IPs, timestamps, hex digests, file paths) "
        "using character classes or \\S+ / \\d+, but KEEP the stable, distinguishing words so each "
        "pattern stays specific. One pattern per distinct family; never an over-broad catch-all such "
        f"as `.*`. Given the log lines below, {intent}\n\n"
        "OUTPUT FORMAT — follow exactly:\n"
        "- Write each pattern on its OWN line, prefixed literally with `REGEX: `.\n"
        "- After the pattern you MAY add ` # ` followed by a short reason.\n"
        "- Write backslashes literally (e.g. \\d+, \\S+) — do NOT JSON-escape them.\n"
        "- Output nothing else: no JSON, no markdown fences, no commentary.\n"
        "Example:\n"
        "REGEX: connection refused to \\S+ # cannot reach a dependency\n"
        "REGEX: \\bFATAL\\b .* unhandled # fatal crash"
    )
    user = (
        "Here is a sample of log lines. Each is prefixed with a number and how many times its "
        "family occurred in the sample:\n\n" + listing
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


_REGEX_LINE = re.compile(r"^\s*(?:[-*]\s*)?REGEX:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _strip_wrappers(value: str) -> str:
    s = (value or "").strip()
    for quote in ("`", '"', "'"):
        if len(s) >= 2 and s[0] == quote and s[-1] == quote:
            s = s[1:-1].strip()
    return s


def _sanitize_json(snippet: str) -> str:
    """Repair the two ways an LLM most often breaks JSON containing regexes.

    Regex bodies are full of ``\\d`` / ``\\S`` / ``\\.`` which are *invalid* JSON
    escapes, so a single bad backslash rejects the whole document. Double every
    backslash that is not already a valid JSON escape, and drop trailing commas.
    """
    snippet = re.sub(r'\\(?![\\"/bfnrtu])', r"\\\\", snippet)
    snippet = re.sub(r",(\s*[\]}])", r"\1", snippet)
    return snippet


def _extract_json_array(content: str) -> Optional[list]:
    """Best-effort extraction of a JSON array from an LLM response.

    Tolerates ```code fences```/prose by slicing from the first ``[`` to the last
    ``]``, and retries after repairing invalid regex backslash escapes and
    trailing commas.
    """
    text = (content or "").strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    for candidate in (snippet, _sanitize_json(snippet)):
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, list):
            return data
    return None


def _parse_candidates(content: str) -> List[dict]:
    """Parse LLM output into ``[{pattern, rationale}, ...]``, robustly.

    Primary path is the line-oriented ``REGEX: <pattern> # <reason>`` format the
    prompt asks for — regexes are written literally, so backslashes can't break
    it and surrounding prose is simply ignored. Falls back to a (repaired) JSON
    array for models that answer in JSON anyway, and salvages partial/truncated
    output line-by-line.
    """
    text = content or ""
    out: List[dict] = []
    seen: set = set()

    def _add(pattern: str, rationale: str = "") -> None:
        pat = _strip_wrappers(pattern)
        if pat and pat not in seen:
            seen.add(pat)
            out.append({"pattern": pat, "rationale": (rationale or "").strip()})

    # 1) Preferred: explicit REGEX: lines (survives backslashes, prose, fences).
    for match in _REGEX_LINE.finditer(text):
        raw = match.group(1).strip()
        rationale = ""
        if " # " in raw:
            raw, rationale = raw.split(" # ", 1)
        _add(raw, rationale)
    if out:
        return out

    # 2) Fallback: a JSON array (with backslash/trailing-comma repair).
    items = _extract_json_array(text)
    for item in items or []:
        if isinstance(item, dict):
            _add(item.get("pattern") or "", item.get("rationale") or "")
        elif isinstance(item, str):
            _add(item, "")
    return out


def _looks_like_empty_result(content: str) -> bool:
    """True when the model legitimately proposed nothing (vs unparseable junk)."""
    text = (content or "").strip().strip("`").strip()
    return text in ("", "[]", "{}")


def backtest(
    candidates: List[RegexCandidate],
    lines: List[str],
    problem_rx: List["re.Pattern[str]"],
) -> None:
    """Populate match / new / over-match stats and example lines, in place.

    ``problem_rx`` are the module's existing error+warning rules. A candidate's
    "new" matches are those an existing problem rule does not already catch; for
    ignore candidates, lines an existing problem rule *does* catch are counted as
    over-match (an ignore rule that would silence a real finding).
    """
    cleaned = [_clean(ln) for ln in lines]
    cleaned = [ln for ln in cleaned if ln.strip()]

    for cand in candidates:
        if not cand.valid:
            continue
        try:
            rx = re.compile(cand.pattern, re.IGNORECASE)
        except re.error:
            cand.valid = False
            continue
        matched = [ln for ln in cleaned if rx.search(ln)]
        already = sum(1 for ln in matched if any(p.search(ln) for p in problem_rx))
        cand.match_count = len(matched)
        cand.new_matches = len(matched) - already
        if cand.kind == "ignore":
            cand.over_match = already
        cand.examples = matched[:DEFAULT_EXAMPLE_LINES]


def backtest_pattern(
    pattern: str,
    kind: str,
    lines: List[str],
    *,
    existing_patterns: Optional[Dict[str, List[str]]] = None,
) -> RegexCandidate:
    """Validate and back-test a single (possibly user-edited) pattern.

    Mirrors what :func:`generate_from_loglines` computes per candidate, so the UI
    can refresh Matches/New/Risk after a pattern is edited. Returns a
    :class:`RegexCandidate` with ``valid``/``error`` and the stats populated.
    """
    kind = kind if kind in VALID_KINDS else "error"
    cand = RegexCandidate(pattern=(pattern or "").strip(), kind=kind)
    if not cand.pattern:
        cand.valid = False
        cand.error = "Pattern is empty."
        return cand
    try:
        re.compile(cand.pattern, re.IGNORECASE)
    except re.error as exc:
        cand.valid = False
        cand.error = str(exc)
        return cand

    existing_patterns = existing_patterns or {}
    problem_rx = _compile_all(
        list(existing_patterns.get("error") or []) + list(existing_patterns.get("warning") or [])
    )
    clean_lines = [ln for ln in (_clean(x) for x in lines) if ln.strip()]
    backtest([cand], clean_lines, problem_rx)
    return cand


def generate_from_loglines(
    lines: List[str],
    kind: str,
    provider,
    *,
    existing_patterns: Optional[Dict[str, List[str]]] = None,
    family_cap: int = DEFAULT_FAMILY_CAP,
) -> GenerationResult:
    """Discover and back-test regex candidates from raw ``lines`` for one kind.

    ``existing_patterns`` is ``{"error": [...], "warning": [...], "ignore": [...]}``
    raw pattern strings from the module's pipeline (used to measure redundancy and
    ignore-rule danger). ``provider`` is a resolved ``LLMProviderConfig`` (e.g. a
    local Ollama provider). Never saves anything.
    """
    kind = kind if kind in VALID_KINDS else "error"
    existing_patterns = existing_patterns or {}
    problem_rx = _compile_all(
        list(existing_patterns.get("error") or []) + list(existing_patterns.get("warning") or [])
    )

    clean_lines = [ln for ln in (_clean(x) for x in lines) if ln.strip()]
    if not clean_lines:
        return GenerationResult(
            candidates=[], lines_sampled=0, families=0,
            provider=getattr(provider, "name", None),
            error="No log lines to analyze for this module.",
        )

    representatives, total_families, omitted = _dedupe_families(clean_lines, problem_rx, cap=family_cap)

    payload = {
        "model": provider.model,
        "messages": _build_messages(representatives, kind),
        "temperature": getattr(provider, "temperature", 0.0) or 0.0,
        "top_p": getattr(provider, "top_p", 1.0),
        # Floor the budget so a low provider default can't truncate the list
        # mid-answer (a common cause of unparseable output on chatty modules).
        "max_tokens": max(getattr(provider, "max_output_tokens", None) or 0, 2000),
    }

    try:
        response = _call_llm(provider, payload)
    except Exception as exc:
        return GenerationResult(
            candidates=[], lines_sampled=len(clean_lines), families=total_families,
            families_omitted=omitted, provider=provider.name, model=provider.model, error=str(exc),
        )

    message = (response.get("choices") or [{}])[0].get("message", {})
    content = (message.get("content") or "").strip()
    model = response.get("model", provider.model)

    parsed = _parse_candidates(content)
    if not parsed:
        # Distinguish "the model proposed nothing" (fine) from "we couldn't parse
        # the reply" (actionable) so the UI message is honest.
        err = None if _looks_like_empty_result(content) else (
            "Could not parse any regex patterns from the model's reply. "
            "Try a smaller sample size or a more capable model."
        )
        return GenerationResult(
            candidates=[], lines_sampled=len(clean_lines), families=total_families,
            families_omitted=omitted, provider=provider.name, model=model, error=err,
        )

    candidates: List[RegexCandidate] = []
    for item in parsed:
        pattern = item["pattern"]
        cand = RegexCandidate(
            pattern=pattern,
            kind=kind,
            rationale=(item.get("rationale") or "")[:300],
        )
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            cand.valid = False
            cand.error = str(exc)
        candidates.append(cand)

    backtest(candidates, clean_lines, problem_rx)

    # Rank: valid first, safest (no over-match), then most *new* coverage, then total.
    candidates.sort(
        key=lambda c: (0 if c.valid else 1, c.over_match, -c.new_matches, -c.match_count)
    )

    return GenerationResult(
        candidates=candidates,
        lines_sampled=len(clean_lines),
        families=total_families,
        families_omitted=omitted,
        provider=provider.name,
        model=model,
    )
