"""Tests for LLM provider backends: Ollama, routing, and auto-detection."""
import json
import urllib.request

import pytest

from logtriage import llm_client
from logtriage.llm_client import _ollama_chat_url, _call_ollama, _call_llm
from logtriage.models import LLMProviderConfig
from logtriage.config import build_llm_config


def _provider(provider_type="ollama", api_base="http://127.0.0.1:11434"):
    return LLMProviderConfig(
        name="local", api_base=api_base, api_key_env=None, model="llama3.1",
        provider_type=provider_type, temperature=0.1, top_p=1.0, max_output_tokens=256,
        request_timeout=30.0,
    )


# ---- URL helper -----------------------------------------------------------

def test_ollama_chat_url_from_root():
    assert _ollama_chat_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434/api/chat"


def test_ollama_chat_url_strips_v1_and_api():
    assert _ollama_chat_url("http://host:11434/v1") == "http://host:11434/api/chat"
    assert _ollama_chat_url("http://host:11434/api/") == "http://host:11434/api/chat"


# ---- _call_ollama normalization ------------------------------------------

class _FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode("utf-8")
    def read(self):
        return self._b
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_call_ollama_normalizes_response(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp({
            "model": "llama3.1",
            "message": {"role": "assistant", "content": "Disk is full on /var."},
            "prompt_eval_count": 31,
            "eval_count": 12,
        })

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    payload = {"model": "llama3.1", "messages": [{"role": "user", "content": "hi"}],
               "temperature": 0.1, "top_p": 1.0, "max_tokens": 256}
    out = _call_ollama(_provider(), payload)

    assert captured["url"].endswith("/api/chat")
    assert captured["body"]["stream"] is False
    assert captured["body"]["options"]["num_predict"] == 256
    assert out["choices"][0]["message"]["content"].startswith("Disk is full")
    assert out["usage"] == {"prompt_tokens": 31, "completion_tokens": 12}


# ---- _call_llm routing ----------------------------------------------------

@pytest.mark.parametrize("ptype,target", [
    ("ollama", "_call_ollama"),
    ("anthropic", "_call_anthropic"),
    ("openai", "_call_chat_completion"),
])
def test_call_llm_routes_by_provider_type(monkeypatch, ptype, target):
    calls = []
    for name in ("_call_ollama", "_call_anthropic", "_call_chat_completion"):
        monkeypatch.setattr(
            llm_client, name,
            (lambda n: (lambda p, pl: (calls.append(n) or {"usage": {}, "choices": []})))(name),
        )
    _call_llm(_provider(provider_type=ptype),
              {"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 8})
    assert calls == [target]


# ---- provider_type auto-detection ----------------------------------------

def test_provider_type_autodetected():
    raw = {"llm": {"enabled": True, "providers": {
        "o": {"api_base": "http://127.0.0.1:11434", "model": "llama3.1"},
        "a": {"api_base": "https://api.anthropic.com/v1", "model": "claude-sonnet-4-6"},
        "x": {"api_base": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    }}}
    cfg = build_llm_config(raw)
    assert cfg.providers["o"].provider_type == "ollama"
    assert cfg.providers["a"].provider_type == "anthropic"
    assert cfg.providers["x"].provider_type == "openai"
