"""Shared pytest fixtures.

The ``database`` fixture (a throwaway SQLite DB wired into ``logtriage.webui.db``'s
module-global engine) was duplicated verbatim across test_issues_db.py,
test_enrichment.py and test_metrics_auth.py; it lives here once now.
"""
import pytest

from logtriage.webui import db as _db


@pytest.fixture()
def database(tmp_path):
    url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    # reset any prior global engine, then set up a fresh one
    _db._engine = None
    _db._db_url = None
    _db.setup_database(url)
    yield url
    _db._engine = None
    _db._db_url = None
