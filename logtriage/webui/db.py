from __future__ import annotations

import datetime
import importlib.util
from dataclasses import dataclass
from typing import Optional, Dict, List, Iterable

_sqlalchemy_spec = importlib.util.find_spec("sqlalchemy")
if _sqlalchemy_spec is None:
    _sqlalchemy_import_error = ModuleNotFoundError("No module named 'sqlalchemy'")
    create_engine = Column = Integer = String = Boolean = DateTime = Text = None  # type: ignore
    sessionmaker = None  # type: ignore
    declarative_base = lambda: None  # type: ignore
else:
    from sqlalchemy import (
        Boolean,
        Column,
        DateTime,
        Integer,
        String,
        Text,
        create_engine,
        inspect,
        func,
        text,
    )
    from sqlalchemy.orm import declarative_base, sessionmaker
    _sqlalchemy_import_error = None

Base = declarative_base() if declarative_base else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False) if sessionmaker else None

_engine = None
_db_url: Optional[str] = None


def _ensure_llm_columns(engine):
    """Add LLM response columns to the findings table when missing.

    Existing deployments may have been created before these columns existed; we
    issue lightweight ALTER TABLE statements to keep them in sync without
    requiring an external migration step.
    """

    if Base is None:
        return

    inspector = inspect(engine)
    try:
        existing = {col["name"] for col in inspector.get_columns("findings")}
    except Exception:
        return

    ddl_statements = [
        ("llm_provider", "VARCHAR(128)"),
        ("llm_model", "VARCHAR(128)"),
        ("llm_response_content", "TEXT"),
        ("llm_prompt_tokens", "INTEGER"),
        ("llm_completion_tokens", "INTEGER"),
    ]

    with engine.begin() as conn:
        for col_name, col_type in ddl_statements:
            if col_name in existing:
                continue
            try:
                conn.execute(text(f"ALTER TABLE findings ADD COLUMN {col_name} {col_type}"))
            except Exception:
                continue


if Base is not None:
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
        llm_provider = Column(String(128), nullable=True)
        llm_model = Column(String(128), nullable=True)
        llm_response_content = Column(Text, nullable=True)
        llm_prompt_tokens = Column(Integer, nullable=True)
        llm_completion_tokens = Column(Integer, nullable=True)
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


    class ModuleActivity(Base):
        __tablename__ = "module_activity"

        module_name = Column(String(128), primary_key=True)
        last_seen = Column(
            DateTime(timezone=True),
            nullable=False,
            default=lambda: datetime.datetime.now(datetime.timezone.utc),
            index=True,
        )

        @property
        def llm_response(self):
            try:
                from ..models import LLMResponse

                if not self.llm_response_content and not self.llm_provider:
                    return None
                return LLMResponse(
                    provider=self.llm_provider or "",
                    model=self.llm_model or "",
                    content=self.llm_response_content or "",
                    prompt_tokens=self.llm_prompt_tokens,
                    completion_tokens=self.llm_completion_tokens,
                )
            except Exception:
                return None
else:  # pragma: no cover - used when sqlalchemy is absent
    class FindingRecord:
        pass


@dataclass
class ModuleStats:
    module_name: str
    last_severity: Optional[str]
    last_seen: Optional[datetime.datetime]
    errors_24h: int
    warnings_24h: int


def setup_database(database_url: str):
    """Initialise engine + metadata for the given database URL.

    Safe to call multiple times; it will only re-initialise if URL changes.
    """
    global _engine, _db_url, SessionLocal
    if _engine is not None and _db_url == database_url:
        return

    if _sqlalchemy_import_error is not None:
        raise ModuleNotFoundError(
            "sqlalchemy is required for database/Web UI features. "
            "Install with `pip install '.[webui]'` or `pip install fastapi uvicorn jinja2 "
            "python-multipart passlib[bcrypt] sqlalchemy itsdangerous`."
        ) from _sqlalchemy_import_error

    engine = create_engine(database_url, future=True)
    Base.metadata.create_all(engine)
    _ensure_llm_columns(engine)
    SessionLocal.configure(bind=engine)
    _engine = engine
    _db_url = database_url


def get_session():
    if _engine is None or SessionLocal is None:
        return None
    return SessionLocal()


def get_next_finding_index(module_name: str) -> int:
    sess = get_session()
    if sess is None:
        return 1

    try:
        max_idx = (
            sess.query(func.max(FindingRecord.finding_index))
            .filter(FindingRecord.module_name == module_name)
            .scalar()
        )
        return (max_idx or 0) + 1
    except Exception:
        return 1
    finally:
        sess.close()


def store_finding(module_name: str, finding, anomaly_flag: bool = False):
    """Persist a single Finding into the database."""
    sess = get_session()
    if sess is None:
        return

    llm_response = getattr(finding, "llm_response", None)
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
        llm_provider=getattr(llm_response, "provider", None),
        llm_model=getattr(llm_response, "model", None),
        llm_response_content=getattr(llm_response, "content", None),
        llm_prompt_tokens=getattr(llm_response, "prompt_tokens", None),
        llm_completion_tokens=getattr(llm_response, "completion_tokens", None),
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
        activity_rows = sess.query(ModuleActivity).all()
        activities = {row.module_name: row.last_seen for row in activity_rows}
    except Exception:
        activities = {}

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
                    last_seen=None,
                    errors_24h=0,
                    warnings_24h=0,
                )
                stats[row.module_name] = s
            sev = (row.severity or "").upper()
            if sev in ("ERROR", "CRITICAL"):
                s.errors_24h += 1
            elif sev == "WARNING":
                s.warnings_24h += 1
            s.last_severity = row.severity

        last_seen_rows = (
            sess.query(FindingRecord.module_name, func.max(FindingRecord.created_at))
            .group_by(FindingRecord.module_name)
            .all()
        )
        last_seen_by_module = {name: seen_at for name, seen_at in last_seen_rows}

        for mod_name, seen_at in activities.items():
            s = stats.get(mod_name)
            if s is None:
                s = ModuleStats(
                    module_name=mod_name,
                    last_severity=None,
                    last_seen=seen_at,
                    errors_24h=0,
                    warnings_24h=0,
                )
                stats[mod_name] = s
            else:
                if s.last_seen is None or (seen_at and seen_at > s.last_seen):
                    s.last_seen = seen_at

        for mod_name, seen_at in last_seen_by_module.items():
            s = stats.get(mod_name)
            if s is None:
                s = ModuleStats(
                    module_name=mod_name,
                    last_severity=None,
                    last_seen=seen_at,
                    errors_24h=0,
                    warnings_24h=0,
                )
                stats[mod_name] = s
            else:
                if s.last_seen is None or (seen_at and seen_at > s.last_seen):
                    s.last_seen = seen_at
    except Exception:
        return {}
    finally:
        sess.close()

    return stats


def update_module_last_seen(module_name: str, seen_at: Optional[datetime.datetime] = None):
    sess = get_session()
    if sess is None:
        return

    seen_at = seen_at or datetime.datetime.now(datetime.timezone.utc)

    try:
        row = sess.query(ModuleActivity).filter(ModuleActivity.module_name == module_name).first()
        if row is None:
            row = ModuleActivity(module_name=module_name, last_seen=seen_at)
            sess.add(row)
        else:
            if row.last_seen is None or seen_at > row.last_seen:
                row.last_seen = seen_at
        sess.commit()
    except Exception:
        sess.rollback()
    finally:
        sess.close()


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


def delete_findings_for_module(module_name: str) -> int:
    sess = get_session()
    if sess is None:
        return 0

    try:
        deleted = (
            sess.query(FindingRecord)
            .filter(FindingRecord.module_name == module_name)
            .delete(synchronize_session=False)
        )
        sess.commit()
        return deleted or 0
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def delete_findings_by_ids(finding_ids: Iterable[int]) -> int:
    sess = get_session()
    if sess is None:
        return 0

    try:
        ids = [fid for fid in finding_ids if isinstance(fid, int)]
        if not ids:
            return 0
        deleted = (
            sess.query(FindingRecord)
            .filter(FindingRecord.id.in_(ids))
            .delete(synchronize_session=False)
        )
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


def update_finding_llm_data(
    finding_id: int,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    content: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> bool:
    sess = get_session()
    if sess is None:
        return False

    try:
        updated = (
            sess.query(FindingRecord)
            .filter(FindingRecord.id == finding_id)
            .update(
                {
                    "llm_provider": provider,
                    "llm_model": model,
                    "llm_response_content": content,
                    "llm_prompt_tokens": prompt_tokens,
                    "llm_completion_tokens": completion_tokens,
                },
                synchronize_session=False,
            )
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
