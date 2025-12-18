from __future__ import annotations

import logging
import os
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


def configure_logging_from_config(cfg: dict) -> None:
    """Configure logging based on configuration dictionary.
    
    Args:
        cfg: Configuration dictionary containing logging settings
    """
    configure_logging_from_dict(cfg)


def main():
    """Entry point for the logtriage-webui command.
    
    Starts the FastAPI web server using the configured host and port
    from the settings. This provides the dashboard interface for
    viewing findings and managing configuration.
    """
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
    base_path = settings.base_path
    logger.info(f"Starting WebUI on {host}:{port}")
    
    uvicorn.run(app, host=host, port=port, root_path=base_path)


if __name__ == "__main__":
    main()
