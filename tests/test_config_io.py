"""Tests for the STATE-only config reload/write service."""
from pathlib import Path

import pytest

pytest.importorskip("yaml")
import yaml

from logtriage.webui import config_io
from logtriage.webui.state import STATE


def _write(path: Path, data: dict):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_reload_from_disk_updates_state(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    _write(cfg, {"webui": {"secret_key": "k", "staleness_minutes": 15}, "llm": {"enabled": False}})

    # Point STATE at the file and reload.
    STATE.config_path = cfg
    config_io.reload_from_disk()
    assert STATE.settings.staleness_minutes == 15
    assert STATE.llm_defaults.enabled is False

    # Change the file on disk; a reload must reflect it (no stale binding).
    _write(cfg, {"webui": {"secret_key": "k", "staleness_minutes": 45}, "llm": {"enabled": False}})
    config_io.reload_from_disk()
    assert STATE.settings.staleness_minutes == 45


def test_save_config_text_atomic_with_backup(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("webui:\n  secret_key: old\n", encoding="utf-8")
    STATE.config_path = cfg

    config_io.save_config_text("webui:\n  secret_key: new\n")
    assert "secret_key: new" in cfg.read_text(encoding="utf-8")
    # Prior contents preserved in the .bak sidecar.
    bak = cfg.with_suffix(cfg.suffix + ".bak")
    assert bak.exists() and "secret_key: old" in bak.read_text(encoding="utf-8")


def test_add_regex_to_pipeline_appends_to_correct_kind(tmp_path):
    cfg = tmp_path / "config.yaml"
    _write(cfg, {"pipelines": [{"name": "svc", "classifier": {"error_regexes": ["boom"]}}]})
    STATE.config_path = cfg
    reloads = {"n": 0}

    err = config_io.add_regex_to_pipeline(
        "svc", r"Connection refused", "error", reload=lambda: reloads.__setitem__("n", reloads["n"] + 1)
    )
    assert err is None
    assert reloads["n"] == 1
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    classifier = data["pipelines"][0]["classifier"]
    assert classifier["error_regexes"] == ["boom", "Connection refused"]

    # A different kind creates its own list without touching the first.
    err = config_io.add_regex_to_pipeline("svc", r"deprecated", "warning", reload=lambda: None)
    assert err is None
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["pipelines"][0]["classifier"]["warning_regexes"] == ["deprecated"]


def test_add_regex_to_pipeline_dedupes_and_validates(tmp_path):
    cfg = tmp_path / "config.yaml"
    _write(cfg, {"pipelines": [{"name": "svc", "classifier": {"ignore_regexes": ["noise"]}}]})
    STATE.config_path = cfg

    # Re-adding the same pattern is a no-op (no duplicate).
    assert config_io.add_regex_to_pipeline("svc", "noise", "ignore", reload=lambda: None) is None
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["pipelines"][0]["classifier"]["ignore_regexes"] == ["noise"]

    # Unknown pipeline and empty pattern are reported, not written.
    assert config_io.add_regex_to_pipeline("nope", "x", "error", reload=lambda: None)
    assert config_io.add_regex_to_pipeline("svc", "   ", "error", reload=lambda: None)


def test_reload_callbacks_invoked(tmp_path):
    cfg = tmp_path / "config.yaml"
    _write(cfg, {"webui": {"secret_key": "k"}, "llm": {"enabled": False}})
    STATE.config_path = cfg
    calls = {"db": 0, "oidc": 0}
    config_io.reload_from_disk(
        init_database=lambda raw, s: calls.__setitem__("db", calls["db"] + 1),
        configure_oidc=lambda s: calls.__setitem__("oidc", calls["oidc"] + 1),
    )
    assert calls == {"db": 1, "oidc": 1}
