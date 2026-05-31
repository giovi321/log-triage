from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
import importlib.util

if importlib.util.find_spec("uvicorn") is None:
    raise SystemExit(
        "Web UI dependencies are missing. Install with `pip install '.[webui]'` or "
        "`pip install fastapi uvicorn jinja2 python-multipart passlib[bcrypt] sqlalchemy itsdangerous`"
    )

import uvicorn

from .app import app, settings
from ..config import load_config
from ..logging_setup import configure_logging_from_dict


def _grant_admin(username: str, *, revoke: bool = False) -> int:
    """Break-glass admin toggle: promote/demote a local user from the shell.

    Uses the same database the Web UI does (resolved from the config file). This
    is the recovery path when nobody can reach the admin-only user-management UI.
    Returns a process exit code.
    """
    config_path = Path(os.environ.get("LOGTRIAGE_CONFIG", "./config.yaml"))
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        print(f"Could not load config from {config_path}: {exc}", file=sys.stderr)
        return 2

    db_url = ((cfg.get("database") or {}) if isinstance(cfg, dict) else {}).get("url")
    if not db_url:
        print("No database.url configured; local users live in the database.", file=sys.stderr)
        return 2

    from .db import setup_database
    from . import users as users_mod

    setup_database(db_url)
    try:
        ok = users_mod.set_admin(username, not revoke)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not ok:
        existing = [u.username for u in users_mod.list_users()]
        print(f"No local user '{username}'. Known users: {existing or '(none)'}", file=sys.stderr)
        return 1
    print(f"User '{username}' is now {'a regular user' if revoke else 'an admin'}. "
          "They must sign out and back in for it to take effect.")
    return 0


def configure_logging_from_config(cfg: dict) -> None:
    """Configure logging based on configuration dictionary.
    
    Args:
        cfg: Configuration dictionary containing logging settings
    """
    configure_logging_from_dict(cfg)


def main():
    """Entry point for the logtriage-webui command.

    Starts the FastAPI web server using the configured host and port from the
    settings. With ``--grant-admin``/``--revoke-admin`` it instead performs a
    one-shot break-glass admin change and exits without starting the server.
    """
    parser = argparse.ArgumentParser(prog="logtriage-webui", description="log-triage Web UI")
    parser.add_argument(
        "--grant-admin", metavar="USERNAME",
        help="Grant admin to a local user, then exit (break-glass recovery).",
    )
    parser.add_argument(
        "--revoke-admin", metavar="USERNAME",
        help="Revoke admin from a local user, then exit.",
    )
    args = parser.parse_args()

    if args.grant_admin or args.revoke_admin:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
        if args.grant_admin:
            raise SystemExit(_grant_admin(args.grant_admin))
        raise SystemExit(_grant_admin(args.revoke_admin, revoke=True))

    # Load configuration and set up logging
    config_path = Path(os.environ.get("LOGTRIAGE_CONFIG", "./config.yaml"))
    try:
        cfg = load_config(config_path)
        configure_logging_from_config(cfg)
        logger = logging.getLogger(__name__)
        logger.info(f"WebUI logging configured from {config_path}")
    except Exception as e:
        # Fallback to basic logging if config fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            force=True
        )
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load config from {config_path}, using default logging: {e}")
    
    host = settings.host
    port = settings.port
    # root_path is only meaningful for a real reverse-proxy sub-path (e.g.
    # "/logtriage"). A bare "/" must become "" — otherwise Starlette prefixes
    # every generated URL with it, producing doubled-slash paths like "//issues".
    root_path = (settings.base_path or "").rstrip("/")
    logger.info(f"Starting WebUI on {host}:{port}")

    uvicorn.run(app, host=host, port=port, root_path=root_path)


if __name__ == "__main__":
    main()
