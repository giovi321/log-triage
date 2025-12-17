from __future__ import annotations

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


def configure_logging_from_config(cfg: dict) -> None:
    """Configure logging based on configuration dictionary.
    
    Args:
        cfg: Configuration dictionary containing logging settings
    """
    logging_config = cfg.get("logging", {})
    
    # Default logging settings
    level = logging_config.get("level", "INFO")
    format_str = logging_config.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    log_file = logging_config.get("file")
    logger_levels = logging_config.get("loggers", {})
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure handlers
    handlers = []
    
    # Console handler (always included)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_str))
    handlers.append(console_handler)
    
    # File handler (optional)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format_str))
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file handler: {e}", file=sys.stderr)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format=format_str,
        force=True  # Override any existing configuration
    )
    
    # Configure specific loggers
    for logger_name, logger_level in logger_levels.items():
        try:
            logger = logging.getLogger(logger_name)
            logger_numeric_level = getattr(logging, logger_level.upper(), logging.INFO)
            logger.setLevel(logger_numeric_level)
        except Exception as e:
            print(f"Warning: Could not configure logger {logger_name}: {e}", file=sys.stderr)


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
    logger.info(f"Starting WebUI on {host}:{port}")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
