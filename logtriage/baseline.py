import json
import time
from pathlib import Path
from typing import Any, Dict

from .models import BaselineConfig, LogChunk, Severity


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


def apply_baseline(cfg: BaselineConfig, chunk: LogChunk) -> None:
    """Update baseline state and adjust chunk severity/reason if anomaly detected."""
    if not cfg.enabled:
        return

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
    if avg_err > 0 and chunk.error_count >= avg_err * cfg.error_multiplier:
        factor = chunk.error_count / avg_err if avg_err > 0 else 0
        anomaly_parts.append(
            f"errors {chunk.error_count} >= {cfg.error_multiplier}x baseline ({avg_err:.2f}, factor {factor:.2f})"
        )
    if avg_warn > 0 and chunk.warning_count >= avg_warn * cfg.warning_multiplier:
        factor = chunk.warning_count / avg_warn if avg_warn > 0 else 0
        anomaly_parts.append(
            f"warnings {chunk.warning_count} >= {cfg.warning_multiplier}x baseline ({avg_warn:.2f}, factor {factor:.2f})"
        )

    if anomaly_parts:
        prefix = "ANOMALY: " + "; ".join(anomaly_parts)
        if chunk.reason:
            chunk.reason = prefix + " | " + chunk.reason
        else:
            chunk.reason = prefix
        if chunk.severity < cfg.severity_on_anomaly:
            chunk.severity = cfg.severity_on_anomaly

    # update history with this chunk
    entry = {
        "ts": time.time(),
        "error_count": int(chunk.error_count),
        "warning_count": int(chunk.warning_count),
    }
    history.append(entry)
    max_n = max(1, cfg.window)
    if len(history) > max_n:
        history = history[-max_n:]
    state["history"] = history
    _save_state(cfg.state_file, state)
