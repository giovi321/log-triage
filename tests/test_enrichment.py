"""Tests for per-issue LLM analysis + caching."""
import datetime
import types
from pathlib import Path

import pytest

from logtriage.models import Finding, Severity, GlobalLLMConfig, LLMProviderConfig, ModuleLLMConfig
from logtriage.webui import db
from logtriage import enrichment
from logtriage.llm_client import _anthropic_system_field


@pytest.fixture()
def database(tmp_path):
    url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    db._engine = None
    db._db_url = None
    db.setup_database(url)
    yield url
    db._engine = None
    db._db_url = None


def _store_issue(database):
    finding = Finding(
        file_path=Path("/var/log/x.log"),
        pipeline_name="ha",
        finding_index=0,
        severity=Severity.ERROR,
        message='Matched error pattern /ERROR/ on "ERROR"',
        line_start=1,
        line_end=1,
        rule_id="ERROR",
        excerpt=["2026-05-29 14:00:00 ERROR mqtt cannot reach broker 10.0.0.5"],
        needs_llm=False,
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )
    db.store_finding("ha", finding)
    return db.get_issues(module_name="ha")[0]


def _llm_defaults():
    provider = LLMProviderConfig(
        name="test",
        api_base="http://test.local/v1",
        api_key_env=None,
        model="test-model",
        provider_type="openai",
        max_output_tokens=256,
    )
    return GlobalLLMConfig(enabled=True, default_provider="test", providers={"test": provider})


def _module_llm():
    return ModuleLLMConfig(enabled=True, min_severity=Severity.WARNING, max_excerpt_lines=50)


# ---- prompt-cache helper --------------------------------------------------

def test_anthropic_system_field_plain_when_not_cached():
    assert _anthropic_system_field("hello", False) == "hello"


def test_anthropic_system_field_block_when_cached():
    out = _anthropic_system_field("docs here", True)
    assert isinstance(out, list)
    assert out[0]["cache_control"] == {"type": "ephemeral"}
    assert out[0]["text"] == "docs here"


def test_anthropic_system_field_empty_stays_string():
    assert _anthropic_system_field("", True) == ""


# ---- analyze_issue caching ------------------------------------------------

def test_analyze_issue_caches_and_skips_second_time(database, monkeypatch):
    issue = _store_issue(database)
    fingerprint = issue.fingerprint

    calls = []

    def fake_call(provider, payload):
        calls.append(payload)
        return {
            "model": payload["model"],
            "choices": [{"message": {"content": "MQTT broker unreachable. Check network/credentials."}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        }

    monkeypatch.setattr(enrichment, "_call_llm", fake_call)

    wrote = enrichment.analyze_issue(issue, _llm_defaults(), _module_llm())
    assert wrote is True
    assert len(calls) == 1

    refreshed = db.get_issue_by_id(issue.id)
    assert refreshed.llm_content.startswith("MQTT broker unreachable")
    assert refreshed.llm_analyzed_fingerprint == fingerprint
    assert refreshed.has_llm_analysis is True

    # Second analysis is skipped — no extra LLM call.
    wrote_again = enrichment.analyze_issue(refreshed, _llm_defaults(), _module_llm())
    assert wrote_again is False
    assert len(calls) == 1


def test_analyze_issue_records_error_on_empty_response(database, monkeypatch):
    issue = _store_issue(database)
    monkeypatch.setattr(
        enrichment, "_call_llm",
        lambda provider, payload: {"choices": [{"message": {"content": ""}}], "usage": {}},
    )
    assert enrichment.analyze_issue(issue, _llm_defaults(), _module_llm()) is False
    refreshed = db.get_issue_by_id(issue.id)
    assert refreshed.llm_error
    assert refreshed.has_llm_analysis is False


def test_analyze_pending_issues_respects_module_llm_enabled(database, monkeypatch):
    issue = _store_issue(database)
    monkeypatch.setattr(
        enrichment, "_call_llm",
        lambda provider, payload: {
            "model": "test-model",
            "choices": [{"message": {"content": "summary"}}],
            "usage": {},
        },
    )

    # module with LLM enabled → analyzed
    modules = {"ha": types.SimpleNamespace(name="ha", llm=_module_llm())}
    assert enrichment.analyze_pending_issues(modules, _llm_defaults()) == 1
    # already cached → nothing pending
    assert enrichment.analyze_pending_issues(modules, _llm_defaults()) == 0


def test_analyze_pending_issues_skips_disabled_modules(database, monkeypatch):
    _store_issue(database)
    monkeypatch.setattr(enrichment, "_call_llm", lambda p, q: {"choices": [{"message": {"content": "x"}}]})
    disabled = ModuleLLMConfig(enabled=False, min_severity=Severity.WARNING, max_excerpt_lines=50)
    modules = {"ha": types.SimpleNamespace(name="ha", llm=disabled)}
    assert enrichment.analyze_pending_issues(modules, _llm_defaults()) == 0
