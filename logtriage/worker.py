"""Background enrichment worker.

Periodically analyzes issues whose cached LLM summary is missing or stale
(see :func:`logtriage.enrichment.analyze_pending_issues`). It can run as a
daemon thread inside the Web UI process (the default) or as a standalone
``logtriage-worker`` process for decoupled deployments.

Design note: rather than a separate jobs table, the worker simply scans for
issues needing analysis via ``get_issues_needing_llm``. That query *is* the
queue — it returns active issues whose ``llm_analyzed_fingerprint`` no longer
matches their fingerprint. This is idempotent and good enough for a single
worker; a duplicate analysis (if two workers race) only wastes one LLM call.
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from .enrichment import analyze_pending_issues

logger = logging.getLogger(__name__)

# get_deps() -> (modules_by_name, llm_defaults, rag_client) or None to skip a cycle
DepsProvider = Callable[[], Optional[Tuple[Dict, object, object]]]


class EnrichmentWorker:
    """Runs issue enrichment on an interval in a background daemon thread."""

    def __init__(
        self,
        get_deps: DepsProvider,
        interval: float = 60.0,
        batch: int = 25,
        logger_: Optional[logging.Logger] = None,
    ):
        self._get_deps = get_deps
        self._interval = max(5.0, float(interval))
        self._batch = max(1, int(batch))
        self._log = logger_ or logger
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.status: Dict[str, object] = {
            "running": False,
            "last_run": None,
            "last_count": 0,
            "total_analyzed": 0,
            "error": None,
        }

    def run_once(self) -> int:
        """Run a single enrichment cycle; returns the number of issues analyzed."""
        deps = None
        try:
            deps = self._get_deps()
        except Exception as exc:  # pragma: no cover - defensive
            self.status["error"] = f"deps: {exc}"
            return 0
        if not deps:
            return 0

        modules_by_name, llm_defaults, rag_client = deps
        try:
            count = analyze_pending_issues(
                modules_by_name, llm_defaults, rag_client=rag_client, limit=self._batch
            )
        except Exception as exc:
            self._log.warning("Enrichment cycle failed: %s", exc)
            self.status["error"] = str(exc)
            return 0

        self.status["error"] = None
        self.status["last_count"] = count
        self.status["total_analyzed"] = int(self.status.get("total_analyzed", 0)) + count
        if count:
            self._log.info("Enriched %d issue(s)", count)
        return count

    def _loop(self) -> None:
        self.status["running"] = True
        # Small initial delay so app startup isn't competing with the first cycle.
        if self._stop.wait(timeout=5.0):
            self.status["running"] = False
            return
        while not self._stop.is_set():
            self.run_once()
            self.status["last_run"] = time.time()
            self._stop.wait(timeout=self._interval)
        self.status["running"] = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="enrichment-worker", daemon=True)
        self._thread.start()
        self._log.info("Enrichment worker started (interval=%ss, batch=%s)", self._interval, self._batch)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self.status["running"] = False
        self._log.info("Enrichment worker stopped")


# ---------------------------------------------------------------------------
# Standalone entry point: ``logtriage-worker``
# ---------------------------------------------------------------------------

def _build_deps_from_config(cfg_path: Path):
    """Build (modules_by_name, llm_defaults, rag_client) from a config file."""
    from .config import load_config, build_llm_config, build_modules, build_rag_config
    from .webui.db import setup_database

    cfg = load_config(cfg_path)
    db_cfg = (cfg.get("database") or {}) if isinstance(cfg, dict) else {}
    db_url = db_cfg.get("url")
    if not db_url:
        raise SystemExit("No database.url configured; the worker needs a database.")
    setup_database(db_url)

    llm_defaults = build_llm_config(cfg)
    modules_by_name = {m.name: m for m in build_modules(cfg, llm_defaults)}

    rag_client = None
    try:
        from .rag.service_client import create_rag_client
        rag_cfg = build_rag_config(cfg)
        if rag_cfg and getattr(rag_cfg, "enabled", False) and create_rag_client is not None:
            url = getattr(rag_cfg, "service_url", None) or "http://127.0.0.1:8091"
            candidate = create_rag_client(url, fallback=True)
            rag_client = candidate if candidate.is_healthy() else None
    except Exception:
        rag_client = None

    return modules_by_name, llm_defaults, rag_client


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="log-triage enrichment worker")
    parser.add_argument("--config", "-c", required=True, help="Path to config.yaml")
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds between cycles (default 60)")
    parser.add_argument("--batch", type=int, default=25, help="Max issues analyzed per cycle (default 25)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg_path = Path(args.config)

    def get_deps():
        # Rebuild each cycle so config edits (saved via the Web UI) are picked up.
        try:
            return _build_deps_from_config(cfg_path)
        except SystemExit:
            raise
        except Exception as exc:
            logger.warning("Could not build worker deps: %s", exc)
            return None

    worker = EnrichmentWorker(get_deps, interval=args.interval, batch=args.batch)

    if args.once:
        print(f"Analyzed {worker.run_once()} issue(s).")
        return

    try:
        worker.start()
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        worker.stop()


if __name__ == "__main__":
    main()
