import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logtriage.cli import _modules_to_run
from logtriage.models import ModuleConfig, ModuleLLMConfig, Severity


def _module(name: str, *, mode: str = "batch", enabled: bool = True) -> ModuleConfig:
    llm_cfg = ModuleLLMConfig(
        enabled=False,
        min_severity=Severity.ERROR,
        max_excerpt_lines=20,
    )

    return ModuleConfig(
        name=name,
        path=pathlib.Path(f"/tmp/{name}.log"),
        mode=mode,
        pipeline_name=None,
        output_format="text",
        min_print_severity=Severity.WARNING,
        llm=llm_cfg,
        stream_from_beginning=False,
        stream_interval=1.0,
        stale_after_minutes=None,
        alert_mqtt=None,
        alert_webhook=None,
        baseline=None,
        enabled=enabled,
    )


def test_modules_to_run_includes_enabled_batch_and_follow():
    modules = [_module("batch", mode="batch"), _module("follow", mode="follow")]

    selected = _modules_to_run(modules, selected_name=None)

    assert selected == modules


def test_modules_to_run_skips_disabled_when_not_specified():
    modules = [_module("enabled"), _module("disabled", enabled=False)]

    selected = _modules_to_run(modules, selected_name=None)

    assert selected == [modules[0]]


def test_modules_to_run_allows_explicit_disabled_selection():
    modules = [_module("target", enabled=False), _module("other")]

    selected = _modules_to_run(modules, selected_name="target")

    assert selected == [modules[0]]
