import datetime
import os
from typing import Any, Dict, List, Optional

from ..models import ModuleConfig


INGESTION_STALENESS_MINUTES = int(os.getenv("LOGTRIAGE_INGESTION_STALENESS_MINUTES", "60"))


def _derive_ingestion_status(
    modules: List[ModuleConfig],
    now: Optional[datetime.datetime] = None,
    freshness_minutes: int = INGESTION_STALENESS_MINUTES,
) -> Dict[str, Any]:
    """Compute a traffic-light style indicator for log ingestion freshness.

    The indicator is based on the last modification time of each module's log
    file. If a module's log file is missing or stale beyond the configured
    window (per-module `stale_after_minutes` or the default), it is marked as
    stale.
    """

    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)

    monitored_modules = [m for m in modules if m.enabled and m.mode != "batch"]
    monitored_module_names = [m.name for m in monitored_modules]

    if not monitored_modules:
        return {
            "state_class": "ok",
            "message": "Staleness tracking skipped for batch modules",
            "stale_modules": [],
            "latest_log_update": None,
        }

    latest_update: Optional[datetime.datetime] = None
    stale_modules: List[str] = []
    freshness_windows: set[int] = set()

    for mod in monitored_modules:
        mod_freshness_minutes = getattr(mod, "stale_after_minutes", None) or freshness_minutes
        freshness_windows.add(mod_freshness_minutes)
        freshness_window = datetime.timedelta(minutes=mod_freshness_minutes)
        try:
            mtime = datetime.datetime.fromtimestamp(mod.path.stat().st_mtime, tz=datetime.timezone.utc)
        except FileNotFoundError:
            mtime = None

        if mtime and (latest_update is None or mtime > latest_update):
            latest_update = mtime

        if mtime is None or (now - mtime) > freshness_window:
            stale_modules.append(mod.name)

    if not freshness_windows:
        freshness_windows.add(freshness_minutes)

    window_hint = (
        f"{next(iter(freshness_windows))}m"
        if len(freshness_windows) == 1
        else "their configured windows"
    )

    if latest_update is None:
        return {
            "state_class": "error",
            "message": "No log file activity detected",
            "stale_modules": monitored_module_names,
            "latest_log_update": None,
        }

    if not stale_modules:
        state_class = "ok"
        message = f"Logs updating (within {window_hint})"
    elif len(stale_modules) == len(monitored_module_names):
        state_class = "error"
        message = f"No recent log updates (> {window_hint})"
    else:
        state_class = "warn"
        message = f"Some modules stale (> {window_hint})"

    return {
        "state_class": state_class,
        "message": message,
        "stale_modules": stale_modules,
        "latest_log_update": latest_update,
    }
