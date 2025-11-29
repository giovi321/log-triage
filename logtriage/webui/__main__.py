from __future__ import annotations

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


def main():
    """Entry point for the logtriage-webui command.
    
    Starts the FastAPI web server using the configured host and port
    from the settings. This provides the dashboard interface for
    viewing findings and managing configuration.
    """
    host = settings.host
    port = settings.port
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
