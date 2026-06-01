"""Tests for LLM-assisted regex discovery from raw log lines (logtriage.regex_gen).

The LLM call is stubbed so these run without a model; we exercise family
de-duplication, JSON parsing, validation, and the back-test statistics
(total matches, new-vs-already-covered, and ignore-rule over-match).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from logtriage import regex_gen


@dataclass
class FakeProvider:
    name: str = "ollama-local"
    model: str = "qwen2.5"
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 1000


def _stub_llm(content: str):
    def _call(provider, payload):
        return {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "model": "stub-model",
            "usage": {},
        }
    return _call


def _patch_llm(monkeypatch, content: str):
    monkeypatch.setattr(regex_gen, "_call_llm", _stub_llm(content))


def test_extract_json_array_tolerates_fences():
    raw = "```json\n[{\"pattern\": \"foo\"}]\n```"
    assert regex_gen._extract_json_array(raw) == [{"pattern": "foo"}]


def test_extract_json_array_rejects_garbage():
    assert regex_gen._extract_json_array("no json here") is None


def test_extract_json_array_repairs_regex_backslashes():
    # Unescaped \d / \. are invalid JSON; the repair pass must recover them.
    # This is what a model actually emits: literal backslashes, not JSON-escaped.
    raw = r'[{"pattern": "refused to \d+\.\d+", "rationale": "x"}]'
    data = regex_gen._extract_json_array(raw)
    assert data and data[0]["pattern"] == r"refused to \d+\.\d+"


def test_extract_json_array_tolerates_trailing_comma():
    assert regex_gen._extract_json_array('[{"pattern": "a"},]') == [{"pattern": "a"}]


def test_parse_candidates_marker_format_ignores_prose():
    content = (
        "Here are the patterns I found:\n"
        r"REGEX: connection refused to \S+ # cannot reach dependency" "\n"
        "- REGEX: `\\bFATAL\\b`\n"
        "Let me know if you need more!"
    )
    parsed = regex_gen._parse_candidates(content)
    pats = [p["pattern"] for p in parsed]
    assert pats == [r"connection refused to \S+", r"\bFATAL\b"]
    assert parsed[0]["rationale"] == "cannot reach dependency"


def test_parse_candidates_falls_back_to_json():
    parsed = regex_gen._parse_candidates('[{"pattern": "boom", "rationale": "x"}]')
    assert parsed == [{"pattern": "boom", "rationale": "x"}]


def test_discovers_and_backtests_from_raw_lines(monkeypatch):
    lines = [
        "2026-01-01 INFO service started ok",
        "2026-01-01 ERROR db connection refused to 5432",
        "2026-01-01 ERROR db connection refused to 5599",  # same family (digits normalized)
        "2026-01-01 WARN slow query 1200ms",
    ]
    _patch_llm(monkeypatch, json.dumps([{"pattern": "connection refused", "rationale": "db down"}]))

    result = regex_gen.generate_from_loglines(lines, "error", FakeProvider(), existing_patterns={})

    assert result.error is None
    assert result.lines_sampled == 4
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand.valid is True
    assert cand.match_count == 2          # both ERROR lines
    assert cand.new_matches == 2          # nothing pre-existing to cover them
    assert cand.over_match == 0           # not an ignore rule
    assert len(cand.examples) == 2


def test_new_matches_discounts_already_covered(monkeypatch):
    lines = [
        "ERROR db connection refused to 5432",
        "ERROR db connection refused to 5599",
    ]
    _patch_llm(monkeypatch, json.dumps([{"pattern": "connection refused"}]))

    # An existing error rule already catches these lines → zero NEW value.
    result = regex_gen.generate_from_loglines(
        lines, "error", FakeProvider(),
        existing_patterns={"error": ["connection refused"], "warning": [], "ignore": []},
    )
    cand = result.candidates[0]
    assert cand.match_count == 2
    assert cand.new_matches == 0


def test_ignore_rule_overmatch_flags_real_problems(monkeypatch):
    lines = [
        "heartbeat ok 1",
        "heartbeat ok 2",
        "heartbeat failed boom",  # a real problem, caught by the existing error rule below
    ]
    _patch_llm(monkeypatch, json.dumps([{"pattern": "heartbeat", "rationale": "noise"}]))

    result = regex_gen.generate_from_loglines(
        lines, "ignore", FakeProvider(),
        existing_patterns={"error": ["failed"], "warning": [], "ignore": []},
    )
    cand = result.candidates[0]
    assert cand.match_count == 3
    assert cand.over_match == 1     # the "heartbeat failed" line is currently a real problem
    assert cand.safe is False


def test_families_collapse_and_omission_is_disclosed(monkeypatch):
    lines = [
        "user 1 logged in", "user 2 logged in", "user 3 logged in",  # one family
        "disk full on /dev/sda",                                     # second family
        "cache miss for key abc",                                    # third family
    ]
    _patch_llm(monkeypatch, "[]")  # valid empty array; we only inspect family accounting

    result = regex_gen.generate_from_loglines(
        lines, "error", FakeProvider(), existing_patterns={}, family_cap=2,
    )
    assert result.error is None
    assert result.lines_sampled == 5
    assert result.families == 3
    assert result.families_omitted == 1   # capped to 2, one dropped — disclosed, not silent


def test_invalid_regex_is_flagged_not_dropped(monkeypatch):
    lines = ["ERROR something broke"]
    _patch_llm(monkeypatch, json.dumps([
        {"pattern": "something", "rationale": "ok"},
        {"pattern": "[unclosed", "rationale": "bad"},
    ]))
    result = regex_gen.generate_from_loglines(lines, "error", FakeProvider(), existing_patterns={})
    by_pattern = {c.pattern: c for c in result.candidates}
    assert by_pattern["something"].valid is True
    assert by_pattern["[unclosed"].valid is False
    assert by_pattern["[unclosed"].error


def test_backtest_pattern_recomputes_stats():
    lines = [
        "ERROR db connection refused to 5432",
        "ERROR db connection refused to 5599",
        "INFO all good",
    ]
    cand = regex_gen.backtest_pattern("connection refused", "error", lines, existing_patterns={})
    assert cand.valid is True
    assert cand.match_count == 2
    assert cand.new_matches == 2
    assert len(cand.examples) == 2


def test_backtest_pattern_flags_invalid():
    cand = regex_gen.backtest_pattern("[unclosed", "error", ["whatever"], existing_patterns={})
    assert cand.valid is False
    assert cand.error
    assert cand.match_count == 0


def test_backtest_pattern_ignore_overmatch():
    lines = ["heartbeat ok", "heartbeat failed boom"]
    cand = regex_gen.backtest_pattern(
        "heartbeat", "ignore", lines,
        existing_patterns={"error": ["failed"], "warning": [], "ignore": []},
    )
    assert cand.match_count == 2
    assert cand.over_match == 1
    assert cand.safe is False


def test_no_lines_returns_helpful_error(monkeypatch):
    _patch_llm(monkeypatch, "[]")
    result = regex_gen.generate_from_loglines([], "error", FakeProvider(), existing_patterns={})
    assert result.candidates == []
    assert "No log lines" in (result.error or "")


def test_unparseable_response_is_reported(monkeypatch):
    _patch_llm(monkeypatch, "sorry, I cannot help with that")
    result = regex_gen.generate_from_loglines(["ERROR boom"], "error", FakeProvider(), existing_patterns={})
    assert result.candidates == []
    assert "parse" in (result.error or "").lower()
    # The raw reply is captured so the UI can show it for diagnosis.
    assert result.raw_response == "sorry, I cannot help with that"


def test_empty_result_is_not_an_error(monkeypatch):
    # An explicit empty answer means "no patterns", not a parse failure.
    _patch_llm(monkeypatch, "[]")
    result = regex_gen.generate_from_loglines(["ERROR boom"], "error", FakeProvider(), existing_patterns={})
    assert result.candidates == []
    assert result.error is None


def test_generate_parses_marker_format_with_backslashes(monkeypatch):
    lines = [
        "2026 ERROR db connection refused to 5432",
        "2026 ERROR db connection refused to 5599",
        "2026 INFO fine",
    ]
    # The model answers in the REGEX: line format with literal backslashes —
    # the exact case that broke JSON parsing for jellyfin.
    _patch_llm(monkeypatch, r"REGEX: connection refused to \d+ # db unreachable")
    result = regex_gen.generate_from_loglines(lines, "error", FakeProvider(), existing_patterns={})
    assert result.error is None
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand.pattern == r"connection refused to \d+"
    assert cand.valid is True
    assert cand.match_count == 2
    assert cand.rationale == "db unreachable"
