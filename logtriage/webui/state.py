"""Single source of truth for the Web UI's mutable runtime state.

Every piece of state that used to be a reassigned module-global in app.py lives
here as an attribute of one ``STATE`` object. The object is created once and
**never rebound** — config reload mutates its attributes in place. That is the
whole point: routers, middleware, Jinja filters, and background threads (the SSE
poller, the enrichment worker, the RAG monitor callbacks) all read ``STATE.x``
at call time, so they always observe the latest value instead of freezing a
pre-reload snapshot captured at import via ``from .app import settings``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class AppState:
    # Config + parsed settings (reassigned-in-place on reload)
    settings: Any = None                      # WebUISettings
    raw_config: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[Path] = None        # was CONFIG_PATH
    llm_defaults: Any = None                  # GlobalLLMConfig
    rag_client: Any = None
    context_hints: Dict[str, str] = field(default_factory=dict)

    # Long-lived singletons
    event_hub: Any = None
    enrichment_worker: Any = None
    rag_monitor: Any = None                   # was _rag_monitor

    # Status dicts (mutated in place; readers may hold the reference)
    rag_monitor_status: Dict[str, Any] = field(default_factory=lambda: {
        "last_check": None,
        "rag_available": False,
        "rag_ready": False,
        "detailed_status": None,
        "check_interval": 10,
    })
    db_status: Dict[str, Any] = field(default_factory=lambda: {
        "configured": False,
        "connected": False,
        "error": None,
        "url": None,
    })


# The one instance. Import this, read/assign its attributes — never rebind it.
STATE = AppState()
