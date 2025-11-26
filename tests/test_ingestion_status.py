import datetime
import os
from pathlib import Path

from logtriage.models import ModuleConfig, ModuleLLMConfig, Severity
from logtriage.webui.ingestion_status import INGESTION_STALENESS_MINUTES, _derive_ingestion_status


def _module(
    name: str,
    mode: str,
    path: Path,
    *,
    enabled: bool = True,
    stale_after_minutes: int | None = None,
) -> ModuleConfig:
    return ModuleConfig(
        name=name,
        path=path,
        mode=mode,
        pipeline_name=None,
        output_format="text",
        min_print_severity=Severity.WARNING,
        llm=ModuleLLMConfig(
            enabled=False,
            min_severity=Severity.ERROR,
            max_excerpt_lines=5,
            context_prefix_lines=0,
        ),
        stream_from_beginning=False,
        stream_interval=1.0,
        stale_after_minutes=stale_after_minutes,
        alert_mqtt=None,
        alert_webhook=None,
        baseline=None,
        enabled=enabled,
    )


def test_batch_modules_not_marked_stale(tmp_path: Path) -> None:
    follow_path = tmp_path / "follow.log"
    follow_path.write_text("old follow log")

    batch_path = tmp_path / "batch.log"
    batch_path.write_text("old batch log")

    old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        minutes=INGESTION_STALENESS_MINUTES + 10
    )
    os.utime(follow_path, (old.timestamp(), old.timestamp()))
    os.utime(batch_path, (old.timestamp(), old.timestamp()))

    modules = [
        _module("batch_mod", "batch", batch_path),
        _module("follow_mod", "follow", follow_path),
    ]

    check_time = old + datetime.timedelta(minutes=INGESTION_STALENESS_MINUTES + 1)

    status = _derive_ingestion_status(
        modules,
        now=check_time,
        freshness_minutes=INGESTION_STALENESS_MINUTES,
    )

    assert "batch_mod" not in status["stale_modules"]
    assert "follow_mod" in status["stale_modules"]


def test_all_batch_modules_skip_staleness(tmp_path: Path) -> None:
    batch_path = tmp_path / "batch.log"
    batch_path.write_text("batch contents")

    modules = [_module("batch_only", "batch", batch_path)]

    status = _derive_ingestion_status(modules)

    assert status["stale_modules"] == []
    assert status["state_class"] == "ok"
