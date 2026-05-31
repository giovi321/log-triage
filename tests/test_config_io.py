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
