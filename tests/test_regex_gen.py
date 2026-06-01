"""Tests for LLM-assisted regex generation (logtriage.regex_gen).

The LLM call and the DB-backed signature gathering are stubbed so these run
without a model or a database; we exercise JSON parsing, validation, back-test
statistics (coverage + over-match), and ranking.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from logtriage import regex_gen


@dataclass
class FakeIssue:
    id: int
    signature: str
    sample_excerpt: str = ""
    severity: str = "ERROR"
    status: str = "open"
    occurrence_count: int = 1
    title: str = ""


@dataclass
class FakeProvider:
    name: str = "ollama-local"
    model: str = "qwen2.5"
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 800


def _stub_llm(content: str):
    def _call(provider, payload):
        return {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "model": "stub-model",
            "usage": {},
        }
    return _call


def _patch(monkeypatch, selected, corpus, content):
    monkeypatch.setattr(regex_gen, "gather_signatures", lambda *a, **k: (selected, corpus))
    monkeypatch.setattr(regex_gen, "_call_llm", _stub_llm(content))


def test_extract_json_array_tolerates_fences():
    raw = "```json\n[{\"pattern\": \"foo\"}]\n```"
    assert regex_gen._extract_json_array(raw) == [{"pattern": "foo"}]


def test_extract_json_array_rejects_garbage():
    assert regex_gen._extract_json_array("no json here") is None


def test_valid_and_invalid_candidates(monkeypatch):
    err = FakeIssue(id=1, signature="Connection refused to <NUM>",
                    sample_excerpt="Connection refused to 5432",
                    severity="ERROR", occurrence_count=10)
    warn = FakeIssue(id=2, signature="slow query <NUM>ms",
                     sample_excerpt="slow query 1200ms",
                     severity="WARNING", occurrence_count=3)
    content = json.dumps([
        {"pattern": "Connection refused", "rationale": "db down", "covers": [1]},
        {"pattern": "[unclosed", "rationale": "bad", "covers": []},
    ])
    _patch(monkeypatch, selected=[err], corpus=[err, warn], content=content)

    result = regex_gen.generate_regex_candidates("svc", "error", FakeProvider())

    assert result.error is None
    assert result.signatures_used == 1
    assert len(result.candidates) == 2
    # Valid candidate ranks first.
    top = result.candidates[0]
    assert top.pattern == "Connection refused"
    assert top.valid is True
    assert top.distinct_issues == 1
    assert top.match_count == 10
    assert top.over_match == 0
    assert top.safe is True
    assert top.covers == [1]
    # Invalid regex is flagged, not dropped.
    bad = result.candidates[1]
    assert bad.valid is False
    assert bad.error


def test_over_match_flags_protected_issue(monkeypatch):
    err = FakeIssue(id=1, signature="connection refused",
                    sample_excerpt="connection refused", severity="ERROR", occurrence_count=10)
    warn = FakeIssue(id=2, signature="slow query slow",
                     sample_excerpt="slow query slow", severity="WARNING", occurrence_count=3)
    # Pattern matches both an ERROR (target) and a WARNING (protected for kind=error).
    content = json.dumps([{"pattern": "refused|slow query", "covers": [1]}])
    _patch(monkeypatch, selected=[err], corpus=[err, warn], content=content)

    result = regex_gen.generate_regex_candidates("svc", "error", FakeProvider())
    cand = result.candidates[0]
    assert cand.distinct_issues == 2
    assert cand.match_count == 13
    assert cand.over_match == 1
    assert cand.safe is False


def test_ignore_kind_protects_active_errors(monkeypatch):
    # An ignore rule that would silence an active ERROR issue is unsafe.
    noise = FakeIssue(id=1, signature="heartbeat ok", sample_excerpt="heartbeat ok",
                      severity="INFO", status="muted", occurrence_count=99)
    real = FakeIssue(id=2, signature="heartbeat failed", sample_excerpt="heartbeat failed",
                     severity="ERROR", status="open", occurrence_count=5)
    content = json.dumps([{"pattern": "heartbeat", "covers": [1]}])
    _patch(monkeypatch, selected=[noise], corpus=[noise, real], content=content)

    result = regex_gen.generate_regex_candidates("svc", "ignore", FakeProvider())
    cand = result.candidates[0]
    assert cand.over_match == 1  # the active ERROR issue
    assert cand.safe is False


def test_no_signatures_returns_helpful_error(monkeypatch):
    monkeypatch.setattr(regex_gen, "gather_signatures", lambda *a, **k: ([], []))
    result = regex_gen.generate_regex_candidates("svc", "error", FakeProvider())
    assert result.candidates == []
    assert "No historical issues" in (result.error or "")


def test_non_json_response_is_reported(monkeypatch):
    issue = FakeIssue(id=1, signature="boom", sample_excerpt="boom")
    _patch(monkeypatch, selected=[issue], corpus=[issue], content="sorry, no JSON")
    result = regex_gen.generate_regex_candidates("svc", "error", FakeProvider())
    assert result.candidates == []
    assert "JSON" in (result.error or "")
