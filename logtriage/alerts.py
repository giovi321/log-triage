import json
import sys
import urllib.request
from typing import Optional

from .models import ModuleConfig, Finding, Severity

try:
    import paho.mqtt.client as mqtt  # type: ignore
except ImportError:  # pragma: no cover
    mqtt = None

_mqtt_warned = False


def _send_webhook(mod: ModuleConfig, finding: Finding) -> None:
    """Send webhook alert for a finding.
    
    Args:
        mod: Module configuration with webhook settings
        finding: The finding to send as alert
    """
    cfg = mod.alert_webhook
    if cfg is None or not cfg.enabled:
        return
    if finding.severity < cfg.min_severity:
        return

    payload = {
        "module": mod.name,
        "file_path": str(finding.file_path),
        "pipeline": finding.pipeline_name,
        "severity": finding.severity.name,
        "reason": finding.message,
        "line_start": finding.line_start,
        "line_end": finding.line_end,
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


def _send_mqtt(mod: ModuleConfig, finding: Finding) -> None:
    """Send MQTT alert for a finding.
    
    Args:
        mod: Module configuration with MQTT settings
        finding: The finding to send as alert
    """
    global _mqtt_warned
    cfg = mod.alert_mqtt
    if cfg is None or not cfg.enabled:
        return
    if finding.severity < cfg.min_severity:
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
        "file_path": str(finding.file_path),
        "pipeline": finding.pipeline_name,
        "severity": finding.severity.name,
        "reason": finding.message,
        "line_start": finding.line_start,
        "line_end": finding.line_end,
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


def send_alerts(mod: ModuleConfig, finding: Finding) -> None:
    """Send alerts for a classified finding according to module settings.
    
    Dispatches the finding to all configured alert channels (webhook, MQTT)
    if the finding severity meets the minimum requirements.
    
    Args:
        mod: Module configuration with alert settings
        finding: The classified finding to alert on
    """
    _send_webhook(mod, finding)
    _send_mqtt(mod, finding)
