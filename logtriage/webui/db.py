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


class FindingRecord(Base):
    __tablename__ = "findings"

    id = Column(Integer, primary_key=True)
    module_name = Column(String(128), index=True, nullable=False)
    pipeline_name = Column(String(128), nullable=True)
    file_path = Column(Text, nullable=False)
    finding_index = Column(Integer, nullable=False)
    severity = Column(String(16), index=True, nullable=False)
    message = Column(Text, nullable=False)
    line_start = Column(Integer, nullable=False, default=0)
    line_end = Column(Integer, nullable=False, default=0)
    rule_id = Column(String(256), nullable=True)
    excerpt = Column(Text, nullable=True)
    anomaly_flag = Column(Boolean, nullable=False, default=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        index=True,
    )

    # compatibility helpers for legacy templates
    @property
    def chunk_index(self):
        return self.finding_index

    @property
    def reason(self):
        return self.message

    @property
    def line_count(self):
        return len((self.excerpt or "").splitlines())

    @property
    def error_count(self):
        sev = (self.severity or "").upper()
        return 1 if sev in ("ERROR", "CRITICAL") else 0

    @property
    def warning_count(self):
        sev = (self.severity or "").upper()
        return 1 if sev == "WARNING" else 0


@dataclass
class ModuleStats:
    module_name: str
    last_severity: Optional[str]
    last_reason: Optional[str]
    last_seen: Optional[datetime.datetime]
    errors_24h: int
    warnings_24h: int
    findings_24h: int

    # compatibility alias for legacy templates
    @property
    def chunks_24h(self) -> int:
        return self.findings_24h


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


def store_finding(module_name: str, finding, anomaly_flag: bool = False):
    """Persist a single Finding into the database."""
    sess = get_session()
    if sess is None:
        return
    obj = FindingRecord(
        module_name=module_name,
        pipeline_name=getattr(finding, "pipeline_name", None),
        file_path=str(getattr(finding, "file_path", "")),
        finding_index=int(getattr(finding, "finding_index", 0)),
        severity=str(
            getattr(getattr(finding, "severity", None), "name", getattr(finding, "severity", "UNKNOWN"))
        ),
        message=str(getattr(finding, "message", "")),
        line_start=int(getattr(finding, "line_start", 0)),
        line_end=int(getattr(finding, "line_end", 0)),
        rule_id=getattr(finding, "rule_id", None),
        excerpt="\n".join(getattr(finding, "excerpt", []) or []),
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


def cleanup_old_findings(retention_days: int):
    """Delete findings older than retention_days."""
    if retention_days <= 0:
        return
    sess = get_session()
    if sess is None:
        return
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=retention_days)
    try:
        sess.query(FindingRecord).filter(FindingRecord.created_at < cutoff).delete(synchronize_session=False)
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def get_module_stats() -> Dict[str, ModuleStats]:
    """Return basic stats per module for the last 24h and last finding."""
    sess = get_session()
    if sess is None:
        return {}

    now = datetime.datetime.now(datetime.timezone.utc)
    window_start = now - datetime.timedelta(days=1)
    stats: Dict[str, ModuleStats] = {}

    try:
        rows = (
            sess.query(FindingRecord)
            .filter(FindingRecord.created_at >= window_start)
            .order_by(FindingRecord.module_name, FindingRecord.created_at.asc())
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
                    findings_24h=0,
                )
                stats[row.module_name] = s
            s.findings_24h += 1
            sev = (row.severity or "").upper()
            if sev in ("ERROR", "CRITICAL"):
                s.errors_24h += 1
            elif sev == "WARNING":
                s.warnings_24h += 1
            s.last_severity = row.severity
            s.last_reason = row.message
            s.last_seen = row.created_at
    except Exception:
        return {}
    finally:
        sess.close()

    return stats


def get_latest_finding_time():
    sess = get_session()
    if sess is None:
        return None
    try:
        row = sess.query(FindingRecord).order_by(FindingRecord.created_at.desc()).first()
        return row.created_at if row else None
    except Exception:
        return None
    finally:
        sess.close()


def get_recent_findings_for_module(module_name: str, limit: int = 50) -> List[FindingRecord]:
    sess = get_session()
    if sess is None:
        return []
    try:
        return (
            sess.query(FindingRecord)
            .filter(FindingRecord.module_name == module_name)
            .order_by(FindingRecord.created_at.desc())
            .limit(limit)
            .all()
        )
    except Exception:
        return []
    finally:
        sess.close()


def delete_all_findings() -> int:
    sess = get_session()
    if sess is None:
        return 0

    try:
        deleted = sess.query(FindingRecord).delete(synchronize_session=False)
        sess.commit()
        return deleted or 0
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def delete_finding_by_id(finding_id: int) -> bool:
    sess = get_session()
    if sess is None:
        return False

    try:
        deleted = (
            sess.query(FindingRecord)
            .filter(FindingRecord.id == finding_id)
            .delete(synchronize_session=False)
        )
        sess.commit()
        return bool(deleted)
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def get_finding_by_id(finding_id: int) -> Optional[FindingRecord]:
    sess = get_session()
    if sess is None:
        return None

    try:
        obj = sess.query(FindingRecord).filter(FindingRecord.id == finding_id).one_or_none()
        return obj
    except Exception:
        return None
    finally:
        sess.close()


def update_finding_severity(finding_id: int, severity: str) -> bool:
    sess = get_session()
    if sess is None:
        return False

    try:
        updated = (
            sess.query(FindingRecord)
            .filter(FindingRecord.id == finding_id)
            .update({"severity": severity}, synchronize_session=False)
        )
        sess.commit()
        return bool(updated)
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


# Backward compatibility wrappers for existing UI code
def store_chunk(module_name: str, chunk, anomaly_flag: bool = False):
    return store_finding(module_name, chunk, anomaly_flag=anomaly_flag)


def cleanup_old_chunks(retention_days: int):
    return cleanup_old_findings(retention_days)


def get_latest_chunk_time():
    return get_latest_finding_time()


def get_recent_chunks_for_module(module_name: str, limit: int = 50):
    return get_recent_findings_for_module(module_name, limit=limit)


def delete_all_chunks() -> int:
    return delete_all_findings()


def delete_chunk_by_id(chunk_id: int) -> bool:
    return delete_finding_by_id(chunk_id)


def get_chunk_by_id(chunk_id: int):
    return get_finding_by_id(chunk_id)


def update_chunk_severity(chunk_id: int, severity: str) -> bool:
    return update_finding_severity(chunk_id, severity)
