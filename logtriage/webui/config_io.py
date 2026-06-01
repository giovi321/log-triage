"""Config reload + atomic-write service (STATE-only, reload-safe).

Centralizes the "re-parse config → rebuild llm/rag/oidc → publish to STATE"
sequence that used to be scattered across app.py with `global` rebinds, plus
the atomic config-file writes. Everything here mutates ``STATE`` in place and
touches no module globals, so routers can call it directly and every reader
sees the new values immediately.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from ..config import build_llm_config, build_rag_config, build_modules, load_config
from ..models import GlobalLLMConfig
from ..notifications import add_notification
from .config import parse_webui_settings
from .state import STATE

logger = logging.getLogger(__name__)

try:
    from ..rag.service_client import create_rag_client
except ImportError:  # pragma: no cover - optional dependency
    create_rag_client = None


def refresh_llm_defaults() -> None:
    """Rebuild STATE.llm_defaults from STATE.raw_config."""
    try:
        STATE.llm_defaults = build_llm_config(STATE.raw_config)
    except Exception as exc:
        add_notification("error", "LLM defaults error", str(exc))
        STATE.llm_defaults = GlobalLLMConfig(
            enabled=False, providers={}, default_provider=None,
            context_prefix_lines=0, context_suffix_lines=0,
        )


def _rag_repo_id(source) -> str:
    """Stable repo id for a knowledge source (matches the RAG service's hashing)."""
    import hashlib

    content = f"{source.repo_url}#{source.branch}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _desired_rag_repo_ids(modules) -> set:
    """Repo ids that *should* be registered, per the current config."""
    ids = set()
    for module in modules:
        if module.rag and module.rag.enabled:
            for source in module.rag.knowledge_sources:
                ids.add(_rag_repo_id(source))
    return ids


def resync_rag_if_repos_missing() -> bool:
    """Re-register RAG modules iff a configured repo is missing from the service.

    Registration is rejected while the service is indexing (HTTP 509) and only
    re-attempted on a config reload, so a repo added mid-index is silently
    dropped. The RAG monitor calls this on the rising edge of readiness: once the
    service is idle, we diff the configured repo set against what the service
    actually has and re-run registration only when something is missing (so it
    converges and doesn't churn). Returns True if a resync was triggered.
    """
    if create_rag_client is None:
        return False
    try:
        rag_config = build_rag_config(STATE.raw_config)
        if not rag_config or not rag_config.enabled:
            return False
        desired = _desired_rag_repo_ids(build_modules_safe())
        if not desired:
            return False
        url = getattr(rag_config, "service_url", None) or "http://127.0.0.1:8091"
        probe = create_rag_client(url, fallback=False)
        if not getattr(probe, "is_ready", None) or not probe.is_ready():
            return False
        current = {r.get("repo_id") for r in (probe.get_status() or {}).get("repositories", [])}
        missing = desired - current
        if not missing:
            return False
        logger.info(
            "RAG resync: %d configured repo(s) missing from the service (%s); re-registering.",
            len(missing), ", ".join(sorted(missing)),
        )
        refresh_rag_client()
        return True
    except Exception as exc:
        logger.warning("RAG resync check failed: %s", exc)
        return False


def refresh_rag_client() -> None:
    """(Re)initialise STATE.rag_client from STATE.raw_config."""
    if create_rag_client is None:
        STATE.rag_client = None
        return
    try:
        rag_config = build_rag_config(STATE.raw_config)
        if rag_config and rag_config.enabled:
            url = getattr(rag_config, "service_url", None) or "http://127.0.0.1:8091"
            client = create_rag_client(url, fallback=True)
            STATE.rag_client = client
            if client.is_healthy():
                modules = build_modules_safe()
                keep_repo_ids = []
                for module in modules:
                    if module.rag and module.rag.enabled:
                        client.add_module_config(module.name, module.rag)
                        for source in module.rag.knowledge_sources:
                            keep_repo_ids.append(_rag_repo_id(source))
                # Tear down repos for knowledge sources that were removed from
                # config so they stop showing (and indexing) on the dashboard.
                try:
                    client.reconcile_repos(keep_repo_ids)
                except Exception as exc:
                    logger.warning("RAG repo reconcile failed: %s", exc)
                client.update_knowledge_base()
            else:
                logger.warning("RAG service is not available; RAG disabled")
        else:
            STATE.rag_client = None
    except Exception as exc:
        logger.error("RAG service client initialization failed: %s", exc, exc_info=True)
        add_notification("warning", "RAG service unavailable", "RAG functionality will be disabled")
        STATE.rag_client = None


def build_modules_safe():
    try:
        return build_modules(STATE.raw_config, STATE.llm_defaults)
    except Exception:
        return []


def reload_from_disk(*, init_database=None, configure_oidc=None) -> None:
    """Re-read the config file and republish everything to STATE.

    Optional callbacks (``init_database``, ``configure_oidc``) let app.py inject
    its DB-init and OIDC-configure steps without this module importing them.
    """
    STATE.raw_config = load_config(STATE.config_path)
    STATE.settings = parse_webui_settings(STATE.raw_config)
    if init_database is not None:
        init_database(STATE.raw_config, STATE.settings)
    refresh_llm_defaults()
    refresh_rag_client()
    if configure_oidc is not None:
        configure_oidc(STATE.settings)


def _atomic_write(path: Path, text: str) -> None:
    """Write text to path atomically, keeping a .bak of the prior file."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    backup_path = path.with_suffix(path.suffix + ".bak")
    if path.exists():
        shutil.copy2(path, backup_path)
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except (OSError, AttributeError):
        pass  # directory fsync unsupported on some platforms (e.g. Windows)


def save_config_text(text: str) -> None:
    """Atomically persist new config YAML to the configured path."""
    _atomic_write(STATE.config_path, text)


_CLASSIFIER_KEYS = {"error": "error_regexes", "warning": "warning_regexes", "ignore": "ignore_regexes"}


def add_regex_to_pipeline(pipeline_name, regex_value, kind="ignore", *, reload) -> Optional[str]:
    """Append a regex to a pipeline's classifier list for ``kind``, write, reload.

    ``kind`` is one of error/warning/ignore (unknown values fall back to ignore).
    ``reload`` is the no-arg reload callback (app.py passes _reload_from_disk so
    its globals stay mirrored). Returns an error message, or None on success.
    """
    import yaml
    from .regex_utils import _lint_regex_input

    key = _CLASSIFIER_KEYS.get(kind, "ignore_regexes")
    if not pipeline_name:
        return "Module has no pipeline; cannot add a rule."
    lint = _lint_regex_input(regex_value)
    if lint:
        return " ".join(lint)
    try:
        cfg_dict = yaml.safe_load(Path(STATE.config_path).read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return f"Failed to read config: {exc}"

    entry = next((p for p in (cfg_dict.get("pipelines") or []) if p.get("name") == pipeline_name), None)
    if entry is None:
        return "Pipeline not found in config; cannot add a rule."

    classifier = entry.setdefault("classifier", {})
    target_list = classifier.get(key)
    if not isinstance(target_list, list):
        target_list = []
        classifier[key] = target_list
    if regex_value not in target_list:
        target_list.append(regex_value)

    try:
        save_config_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    except Exception as exc:
        return f"Failed to write config: {exc}"

    (reload or reload_from_disk)()
    return None


def add_ignore_regex_to_pipeline(pipeline_name, regex_value, *, reload) -> Optional[str]:
    """Backwards-compatible wrapper: append an ignore regex to a pipeline."""
    return add_regex_to_pipeline(pipeline_name, regex_value, "ignore", reload=reload)
