"""Signature / fingerprinting for findings.

Turns the stream of individual findings into stable *issue signatures* by
normalising the volatile parts of a log line (timestamps, IDs, IPs, numbers,
hex blobs, …) and hashing the result together with the pipeline and severity.

Two findings that describe "the same problem" — e.g. the same error recurring
with different timestamps and request IDs — collapse to one fingerprint, which
is what lets the Web UI aggregate them into a single de-duplicated issue with an
occurrence count and a first/last-seen window.
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
from typing import Any, List, Optional, Sequence

# Ordered list of (compiled pattern, placeholder). Order matters: more specific
# patterns (timestamps, UUIDs) must run before generic ones (numbers).
_NORMALISERS: List[tuple[re.Pattern, str]] = [
    # ISO-8601 timestamps: 2026-05-29T14:31:02.123+02:00 / 2026-05-29 14:31:02
    (re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "<TS>"),
    # Bare ISO date
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "<DATE>"),
    # Syslog timestamp: "May 29 14:31:02"
    (re.compile(r"\b[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b"), "<TS>"),
    # Time of day (with optional fractional seconds)
    (re.compile(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), "<TS>"),
    # UUIDs
    (re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"), "<UUID>"),
    # MAC addresses (before IPv6 / hex)
    (re.compile(r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b"), "<MAC>"),
    # URLs
    (re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s'\"<>]+"), "<URL>"),
    # Email addresses
    (re.compile(r"\b[\w.\-+]+@[\w.\-]+\.\w+\b"), "<EMAIL>"),
    # IPv4 with optional port
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),
    # IPv6 (rough — runs after MAC and time so it doesn't clobber them)
    (re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{0,4}\b"), "<IP>"),
    # 0x-prefixed hex
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    # Long hex blobs (commit hashes, tokens, object ids)
    (re.compile(r"\b[0-9a-fA-F]{12,}\b"), "<HEX>"),
    # Floats / versions-with-many-dots handled as numbers below
    (re.compile(r"\b\d+\.\d+\b"), "<N>"),
    # Standalone integers
    (re.compile(r"\b\d+\b"), "<N>"),
]

_WS = re.compile(r"\s+")
_MAX_TITLE = 200


def normalize(text: str) -> str:
    """Collapse the volatile parts of a log line into placeholders.

    Stable across runs and machines, so the same logical error always yields
    the same normalised string regardless of timestamps, IDs, counts, etc.
    """
    if not text:
        return ""
    out = text.strip()
    for pattern, repl in _NORMALISERS:
        out = pattern.sub(repl, out)
    out = _WS.sub(" ", out).strip()
    return out


def _excerpt_lines(excerpt: Any) -> List[str]:
    if excerpt is None:
        return []
    if isinstance(excerpt, str):
        return excerpt.splitlines()
    if isinstance(excerpt, (list, tuple)):
        return [str(line) for line in excerpt]
    return [str(excerpt)]


def representative_line(rule_id: Optional[str], excerpt: Any, message: str = "") -> str:
    """Pick the log line that best represents a finding.

    Prefers the excerpt line that the rule's own regex matches (that is the
    line the finding was raised on). Falls back to the longest excerpt line,
    then to the finding message.
    """
    lines = [ln for ln in _excerpt_lines(excerpt) if ln.strip()]

    if rule_id:
        try:
            pattern = re.compile(rule_id)
            for line in lines:
                if pattern.search(line):
                    return line.strip()
        except re.error:
            pass

    if lines:
        # Longest non-trivial line tends to be the substantive one.
        return max(lines, key=len).strip()

    return (message or "").strip()


@dataclasses.dataclass(frozen=True)
class Signature:
    fingerprint: str  # 16-hex stable id
    signature: str    # normalised grouping key (volatile parts removed)
    title: str        # human-readable representative line (raw, truncated)


def _coerce_severity(severity: Any) -> str:
    name = getattr(severity, "name", None)
    if name:
        return str(name).upper()
    return str(severity or "UNKNOWN").upper()


def compute(
    pipeline_name: Optional[str],
    severity: Any,
    rule_id: Optional[str],
    excerpt: Any,
    message: str = "",
) -> Signature:
    """Compute the stable signature for a finding's identifying inputs."""
    sev = _coerce_severity(severity)
    rep = representative_line(rule_id, excerpt, message)
    normalised = normalize(rep) or normalize(message) or (rule_id or "unknown")

    basis = f"{pipeline_name or ''}\x1f{sev}\x1f{normalised}"
    digest = hashlib.sha1(basis.encode("utf-8", errors="replace")).hexdigest()[:16]

    title = rep or normalised or (message or "").strip()
    if len(title) > _MAX_TITLE:
        title = title[: _MAX_TITLE - 1] + "…"

    return Signature(fingerprint=digest, signature=normalised, title=title)


def signature_for_finding(finding: Any) -> Signature:
    """Convenience wrapper that reads the inputs off a Finding/FindingRecord.

    Tolerates both the in-memory ``Finding`` dataclass (excerpt = list) and the
    persisted ``FindingRecord`` (excerpt = newline-joined string).
    """
    return compute(
        pipeline_name=getattr(finding, "pipeline_name", None),
        severity=getattr(finding, "severity", None),
        rule_id=getattr(finding, "rule_id", None),
        excerpt=getattr(finding, "excerpt", None),
        message=getattr(finding, "message", "") or "",
    )
