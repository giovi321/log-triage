from __future__ import annotations

import logging
import sys
from typing import Any, Mapping, Optional, Tuple


def configure_logging_from_dict(cfg: Mapping[str, Any]) -> Tuple[str, Optional[str]]:
    logging_config = {}
    if isinstance(cfg, Mapping):
        logging_config = cfg.get("logging", {}) or {}

    level = logging_config.get("level", "INFO")
    format_str = logging_config.get(
        "format", "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log_file = logging_config.get("file")
    logger_levels = logging_config.get("loggers", {}) or {}

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)

    handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_str))
    handlers.append(console_handler)

    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format_str))
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file handler: {e}", file=sys.stderr)

    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format=format_str,
        force=True,
    )

    if isinstance(logger_levels, dict):
        for logger_name, logger_level in logger_levels.items():
            try:
                logger = logging.getLogger(str(logger_name))
                logger_numeric_level = getattr(
                    logging, str(logger_level).upper(), logging.INFO
                )
                logger.setLevel(logger_numeric_level)
            except Exception as e:
                print(
                    f"Warning: Could not configure logger {logger_name}: {e}",
                    file=sys.stderr,
                )

    return str(level), str(log_file) if log_file else None
