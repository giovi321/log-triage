from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False)


def init_engine(database_url: str):
    global SessionLocal
    engine = create_engine(database_url, future=True)
    SessionLocal.configure(bind=engine)
    return engine
