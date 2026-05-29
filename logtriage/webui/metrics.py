"""Prometheus metrics exposition for the Web UI.

Hand-rendered text exposition (no prometheus_client dependency) so the homelab
observability stack (Prometheus → Grafana, alongside Wazuh) can scrape triage
state: issue counts by status, total findings, and worker activity.
"""
from __future__ import annotations

from typing import Dict, Optional

from .db import issue_status_counts, count_findings, get_max_finding_id


def _metric(lines: list, name: str, help_text: str, mtype: str) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {mtype}")


def render_metrics(worker_status: Optional[Dict] = None) -> str:
    """Return the Prometheus exposition text for current triage state."""
    lines: list = []

    try:
        counts = issue_status_counts()
    except Exception:
        counts = {}

    _metric(lines, "logtriage_issues", "Number of de-duplicated issues by status", "gauge")
    for status_name, n in (counts or {}).items():
        lines.append(f'logtriage_issues{{status="{status_name}"}} {int(n)}')

    active = (counts.get("open", 0) + counts.get("acknowledged", 0)) if counts else 0
    _metric(lines, "logtriage_issues_active", "Active (open + acknowledged) issues", "gauge")
    lines.append(f"logtriage_issues_active {int(active)}")

    _metric(lines, "logtriage_findings_total", "Total findings stored", "counter")
    lines.append(f"logtriage_findings_total {count_findings()}")

    _metric(lines, "logtriage_findings_max_id", "Highest finding id (ingest cursor)", "counter")
    lines.append(f"logtriage_findings_max_id {get_max_finding_id()}")

    ws = worker_status or {}
    _metric(lines, "logtriage_worker_running", "Enrichment worker running (1/0)", "gauge")
    lines.append(f"logtriage_worker_running {1 if ws.get('running') else 0}")

    _metric(lines, "logtriage_worker_analyzed_total", "Issues analyzed by the worker since start", "counter")
    lines.append(f"logtriage_worker_analyzed_total {int(ws.get('total_analyzed', 0) or 0)}")

    return "\n".join(lines) + "\n"
