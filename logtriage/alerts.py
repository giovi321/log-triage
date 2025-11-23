import json
import sys
import urllib.request
from typing import Optional

from .models import ModuleConfig, LogChunk, Severity

try:
    import paho.mqtt.client as mqtt  # type: ignore
except ImportError:  # pragma: no cover
    mqtt = None

_mqtt_warned = False


def _send_webhook(mod: ModuleConfig, chunk: LogChunk) -> None:
    cfg = mod.alert_webhook
    if cfg is None or not cfg.enabled:
        return
    if chunk.severity < cfg.min_severity:
        return

    payload = {
        "module": mod.name,
        "file_path": str(chunk.file_path),
        "pipeline": chunk.pipeline_name,
        "severity": chunk.severity.name,
        "reason": chunk.reason,
        "error_count": chunk.error_count,
        "warning_count": chunk.warning_count,
        "line_count": len(chunk.lines),
    }
    data = json.dumps(payload).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    headers.update(cfg.headers or {})

    req = urllib.request.Request(cfg.url, data=data, method=cfg.method)
    for k, v in headers.items():
        req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception as e:  # pragma: no cover
        print(f"[logtriage] webhook alert failed for module {mod.name}: {e}", file=sys.stderr)


def _send_mqtt(mod: ModuleConfig, chunk: LogChunk) -> None:
    global _mqtt_warned
    cfg = mod.alert_mqtt
    if cfg is None or not cfg.enabled:
        return
    if chunk.severity < cfg.min_severity:
        return
    if mqtt is None:
        if not _mqtt_warned:
            print(
                "[logtriage] MQTT alert configured but paho-mqtt is not installed. "
                "Install with: pip install paho-mqtt",
                file=sys.stderr,
            )
            _mqtt_warned = True
        return

    payload = {
        "module": mod.name,
        "file_path": str(chunk.file_path),
        "pipeline": chunk.pipeline_name,
        "severity": chunk.severity.name,
        "reason": chunk.reason,
        "error_count": chunk.error_count,
        "warning_count": chunk.warning_count,
        "line_count": len(chunk.lines),
    }
    text = json.dumps(payload)

    try:
        client = mqtt.Client()
        if cfg.username is not None:
            client.username_pw_set(cfg.username, cfg.password or "")
        client.connect(cfg.host, cfg.port, 60)
        client.loop_start()
        client.publish(cfg.topic, text)
        client.loop_stop()
        client.disconnect()
    except Exception as e:  # pragma: no cover
        print(f"[logtriage] MQTT alert failed for module {mod.name}: {e}", file=sys.stderr)


def send_alerts(mod: ModuleConfig, chunk: LogChunk) -> None:
    """Send alerts for a classified chunk according to module settings."""
    _send_webhook(mod, chunk)
    _send_mqtt(mod, chunk)
