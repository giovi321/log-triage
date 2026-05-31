"""Tests for WebUISettings parsing (the bits not covered elsewhere)."""
from logtriage.webui.config import parse_webui_settings


def test_staleness_minutes_default_is_60():
    s = parse_webui_settings({"webui": {"secret_key": "k"}})
    assert s.staleness_minutes == 60


def test_staleness_minutes_from_config():
    s = parse_webui_settings({"webui": {"secret_key": "k", "staleness_minutes": 15}})
    assert s.staleness_minutes == 15


def test_staleness_minutes_invalid_falls_back_to_default():
    for bad in (0, -5, "nope", None):
        s = parse_webui_settings({"webui": {"secret_key": "k", "staleness_minutes": bad}})
        assert s.staleness_minutes == 60


def test_staleness_minutes_env_default(monkeypatch):
    monkeypatch.setenv("LOGTRIAGE_INGESTION_STALENESS_MINUTES", "90")
    s = parse_webui_settings({"webui": {"secret_key": "k"}})
    assert s.staleness_minutes == 90
    # explicit config still wins over the env default
    s2 = parse_webui_settings({"webui": {"secret_key": "k", "staleness_minutes": 30}})
    assert s2.staleness_minutes == 30
