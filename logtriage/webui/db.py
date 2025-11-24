from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Text,
)
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False)

_engine = None
_db_url: Optional[str] = None


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    module_name = Column(String(128), index=True, nullable=False)
    pipeline_name = Column(String(128), nullable=True)
    file_path = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    severity = Column(String(16), index=True, nullable=False)
    reason = Column(Text, nullable=False)
    error_count = Column(Integer, nullable=False, default=0)
    warning_count = Column(Integer, nullable=False, default=0)
    line_count = Column(Integer, nullable=False, default=0)
    anomaly_flag = Column(Boolean, nullable=False, default=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        index=True,
    )


@dataclass
class ModuleStats:
    module_name: str
    last_severity: Optional[str]
    last_reason: Optional[str]
    last_seen: Optional[datetime.datetime]
    errors_24h: int
    warnings_24h: int
    chunks_24h: int


def setup_database(database_url: str):
    """Initialise engine + metadata for the given database URL.

    Safe to call multiple times; it will only re-initialise if URL changes.
    """
    global _engine, _db_url, SessionLocal
    if _engine is not None and _db_url == database_url:
        return

    engine = create_engine(database_url, future=True)
    Base.metadata.create_all(engine)
    SessionLocal.configure(bind=engine)
    _engine = engine
    _db_url = database_url


def get_session():
    if _engine is None:
        return None
    return SessionLocal()


def store_chunk(module_name: str, chunk, anomaly_flag: bool = False):
    """Persist a single LogChunk into the database.

    `chunk` is expected to be a logtriage.models.LogChunk instance.
    """
    sess = get_session()
    if sess is None:
        return
    from logtriage.models import LogChunk as LTChunk  # type: ignore

    _ = isinstance(chunk, LTChunk)

    obj = Chunk(
        module_name=module_name,
        pipeline_name=getattr(chunk, "pipeline_name", None),
        file_path=str(getattr(chunk, "file_path", "")),
        chunk_index=int(getattr(chunk, "chunk_index", 0)),
        severity=str(getattr(getattr(chunk, "severity", None), "name", getattr(chunk, "severity", "UNKNOWN"))),
        reason=str(getattr(chunk, "reason", "")),
        error_count=int(getattr(chunk, "error_count", 0)),
        warning_count=int(getattr(chunk, "warning_count", 0)),
        line_count=len(getattr(chunk, "lines", []) or []),
        anomaly_flag=bool(anomaly_flag),
    )
    try:
        sess.add(obj)
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def cleanup_old_chunks(retention_days: int):
    """Delete chunks older than retention_days.

    This is intended to be called at startup of CLI processes.
    """
    if retention_days <= 0:
        return
    sess = get_session()
    if sess is None:
        return
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=retention_days)
    try:
        sess.query(Chunk).filter(Chunk.created_at < cutoff).delete(synchronize_session=False)
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def get_module_stats() -> Dict[str, ModuleStats]:
    """Return basic stats per module for the last 24h and last chunk.

    Used by the dashboard. If DB is not initialised, returns {}.
    """
    sess = get_session()
    if sess is None:
        return {}

    now = datetime.datetime.now(datetime.timezone.utc)
    window_start = now - datetime.timedelta(days=1)
    stats: Dict[str, ModuleStats] = {}

    try:
        rows = (
            sess.query(Chunk)
            .filter(Chunk.created_at >= window_start)
            .order_by(Chunk.module_name, Chunk.created_at.asc())
            .all()
        )
        for row in rows:
            s = stats.get(row.module_name)
            if s is None:
                s = ModuleStats(
                    module_name=row.module_name,
                    last_severity=None,
                    last_reason=None,
                    last_seen=None,
                    errors_24h=0,
                    warnings_24h=0,
                    chunks_24h=0,
                )
                stats[row.module_name] = s
            s.chunks_24h += 1
            s.errors_24h += row.error_count or 0
            s.warnings_24h += row.warning_count or 0
            s.last_severity = row.severity
            s.last_reason = row.reason
            s.last_seen = row.created_at
    except Exception:
        return {}
    finally:
        sess.close()

    return stats


def get_latest_chunk_time():
    sess = get_session()
    if sess is None:
        return None
    try:
        row = sess.query(Chunk).order_by(Chunk.created_at.desc()).first()
        return row.created_at if row else None
    except Exception:
        return None
    finally:
        sess.close()


def get_recent_chunks_for_module(module_name: str, limit: int = 50) -> List[Chunk]:
    sess = get_session()
    if sess is None:
        return []
    try:
        return (
            sess.query(Chunk)
            .filter(Chunk.module_name == module_name)
            .order_by(Chunk.created_at.desc())
            .limit(limit)
            .all()
        )
    except Exception:
        return []
    finally:
        sess.close()


def delete_all_chunks() -> int:
    sess = get_session()
    if sess is None:
        return 0

    try:
        deleted = sess.query(Chunk).delete(synchronize_session=False)
        sess.commit()
        return deleted or 0
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def delete_chunk_by_id(chunk_id: int) -> bool:
    sess = get_session()
    if sess is None:
        return False

    try:
        deleted = (
            sess.query(Chunk)
            .filter(Chunk.id == chunk_id)
            .delete(synchronize_session=False)
        )
        sess.commit()
        return bool(deleted)
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def get_chunk_by_id(chunk_id: int) -> Optional[Chunk]:
    sess = get_session()
    if sess is None:
        return None

    try:
        obj = sess.query(Chunk).filter(Chunk.id == chunk_id).one_or_none()
        return obj
    except Exception:
        return None
    finally:
        sess.close()


def update_chunk_severity(chunk_id: int, severity: str) -> bool:
    sess = get_session()
    if sess is None:
        return False

    try:
        updated = (
            sess.query(Chunk)
            .filter(Chunk.id == chunk_id)
            .update({"severity": severity}, synchronize_session=False)
        )
        sess.commit()
        return bool(updated)
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()
