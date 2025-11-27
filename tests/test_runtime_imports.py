from pathlib import Path

from logtriage.baseline import apply_baseline
from logtriage.models import BaselineConfig, Finding, Severity
from logtriage.stream import follow_file


def test_follow_file_reads_from_beginning(tmp_path):
    log_path = tmp_path / "sample.log"
    log_path.write_text("first\nsecond\n", encoding="utf-8")

    gen = follow_file(log_path, from_beginning=True, interval=0.01, should_stop=lambda: False)
    try:
        start_line, lines = next(gen)
    finally:
        gen.close()

    assert start_line == 1
    assert lines == ["first", "second"]


def test_apply_baseline_executes_with_state(tmp_path):
    cfg = BaselineConfig(
        enabled=True,
        state_file=tmp_path / "baseline_state.json",
        window=5,
        error_multiplier=2.0,
        warning_multiplier=2.0,
        severity_on_anomaly=Severity.ERROR,
    )

    finding = Finding(
        file_path=tmp_path / "log.txt",
        pipeline_name="test",
        finding_index=0,
        severity=Severity.ERROR,
        message="error line",
        line_start=1,
        line_end=1,
        rule_id="pattern",
        excerpt=["error line"],
    )

    output = apply_baseline(cfg, [finding])

    assert output
    assert cfg.state_file.exists()
