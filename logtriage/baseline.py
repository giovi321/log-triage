import json
import time
from pathlib import Path
from typing import Any, Dict, List

from .models import BaselineConfig, Finding, Severity


def _load_state(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"history": []}
    except Exception:
        return {"history": []}
    try:
        data = json.loads(text)
    except Exception:
        return {"history": []}
    if "history" not in data or not isinstance(data["history"], list):
        data["history"] = []
    return data


def _save_state(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def apply_baseline(cfg: BaselineConfig, findings: List[Finding]) -> List[Finding]:
    """Update baseline state and emit an anomaly finding when needed."""
    if not cfg.enabled:
        return findings

    state = _load_state(cfg.state_file)
    history = state.get("history", [])

    # compute baseline averages
    avg_err = 0.0
    avg_warn = 0.0
    if history:
        tot_err = sum(int(h.get("error_count", 0)) for h in history)
        tot_warn = sum(int(h.get("warning_count", 0)) for h in history)
        n = len(history)
        if n > 0:
            avg_err = tot_err / n
            avg_warn = tot_warn / n

    # detect anomaly
    anomaly_parts = []
    error_count = sum(1 for f in findings if f.severity >= Severity.ERROR)
    warning_count = sum(1 for f in findings if f.severity == Severity.WARNING)

    if avg_err > 0 and error_count >= avg_err * cfg.error_multiplier:
        factor = error_count / avg_err if avg_err > 0 else 0
        anomaly_parts.append(
            f"errors {error_count} >= {cfg.error_multiplier}x baseline ({avg_err:.2f}, factor {factor:.2f})"
        )
    if avg_warn > 0 and warning_count >= avg_warn * cfg.warning_multiplier:
        factor = warning_count / avg_warn if avg_warn > 0 else 0
        anomaly_parts.append(
            f"warnings {warning_count} >= {cfg.warning_multiplier}x baseline ({avg_warn:.2f}, factor {factor:.2f})"
        )

    if anomaly_parts:
        prefix = "ANOMALY: " + "; ".join(anomaly_parts)
        findings = list(findings)
        findings.append(
            Finding(
                file_path=findings[0].file_path if findings else Path(""),
                pipeline_name=findings[0].pipeline_name if findings else "",
                finding_index=len(findings),
                severity=max(cfg.severity_on_anomaly, Severity.ERROR),
                message=prefix,
                line_start=findings[0].line_start if findings else 1,
                line_end=findings[-1].line_end if findings else 1,
                rule_id="baseline_anomaly",
                excerpt=[prefix],
                needs_llm=False,
                created_at=findings[0].created_at if findings else None,
            )
        )

    # update history with this batch of findings
    entry = {
        "ts": time.time(),
        "error_count": int(error_count),
        "warning_count": int(warning_count),
    }
    history.append(entry)
    max_n = max(1, cfg.window)
    if len(history) > max_n:
        history = history[-max_n:]
    state["history"] = history
    _save_state(cfg.state_file, state)
    return findings
