"""FastAPI route modules for the Web UI.

Each module exposes a ``router`` (an ``APIRouter``) that app.py mounts via
``include_router``. Routers read live runtime state from ``..state.STATE`` and
shared helpers from ``..shared`` — never from app.py — so there is no circular
import and no stale-after-reload binding.
"""
