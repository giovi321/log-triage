from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from .app import app, settings


def main():
    host = settings.host
    port = settings.port
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
