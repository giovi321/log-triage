import json
import sys
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from logtriage.llm_client import _call_chat_completion
from logtriage.models import LLMProviderConfig


def _make_provider(api_base: str) -> LLMProviderConfig:
    return LLMProviderConfig(
        name="test-provider",
        api_base=api_base,
        api_key_env=None,
        model="demo-model",
    )


def _stubbed_response():
    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    "model": "demo-model",
                }
            ).encode("utf-8")

    return _FakeResponse()


def test_call_chat_completion_handles_versioned_and_plain_bases(monkeypatch):
    observed_urls = []

    def fake_urlopen(req, timeout):
        observed_urls.append((req.full_url, timeout))
        return _stubbed_response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    _call_chat_completion(_make_provider("https://api.example.com"), payload={"messages": []})
    _call_chat_completion(_make_provider("https://api.example.com/v1/"), payload={"messages": []})

    assert observed_urls == [
        ("https://api.example.com/v1/chat/completions", 30.0),
        ("https://api.example.com/v1/chat/completions", 30.0),
    ]
