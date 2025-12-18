from __future__ import annotations

import datetime
import importlib.util
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Dict, List, TYPE_CHECKING

from ..models import Severity, LLMResponse

logger = logging.getLogger(__name__)

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
SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False) if sessionmaker else None

_engine = None
_db_url: Optional[str] = None

if TYPE_CHECKING:  # pragma: no cover
    from ..models import ModuleConfig


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
        ("needs_llm", "BOOLEAN"),
        ("llm_provider", "VARCHAR(128)"),
        ("llm_model", "VARCHAR(128)"),
        ("llm_response_content", "TEXT"),
        ("llm_error", "TEXT"),
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
        needs_llm = Column(Boolean, nullable=False, default=False)
        llm_provider = Column(String(128), nullable=True)
        llm_model = Column(String(128), nullable=True)
        llm_response_content = Column(Text, nullable=True)
        llm_error = Column(Text, nullable=True)
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

        @property
        def severity_enum(self):
            """Convert string severity back to Severity enum for compatibility."""
            try:
                severity_value = self.severity or "WARNING"
                if not isinstance(severity_value, str):
                    severity_value = str(severity_value)
                return Severity.from_string(severity_value)
            except (ValueError, AttributeError, TypeError) as e:
                # Log the error for debugging
                logger.warning(f"Invalid severity value '{self.severity}' in finding {self.id}: {e}")
                return Severity.WARNING

        @property
        def llm_response(self):
            """Reconstruct LLMResponse object from database columns."""
            if not self.llm_provider:
                return None
            return LLMResponse(
                provider=self.llm_provider,
                model=self.llm_model or "unknown",
                content=self.llm_response_content or "",
                prompt_tokens=self.llm_prompt_tokens,
                completion_tokens=self.llm_completion_tokens,
            )

        @llm_response.setter
        def llm_response(self, value):
            """Set LLMResponse object by updating individual database columns."""
            if value is None:
                self.llm_provider = None
                self.llm_model = None
                self.llm_response_content = None
                self.llm_prompt_tokens = None
                self.llm_completion_tokens = None
            else:
                self.llm_provider = value.provider
                self.llm_model = value.model
                self.llm_response_content = value.content
                self.llm_prompt_tokens = value.prompt_tokens
                self.llm_completion_tokens = value.completion_tokens

        @property
        def excerpt_as_list(self):
            """Convert excerpt string to list for compatibility with LLM functions."""
            if not self.excerpt:
                return []
            return self.excerpt.splitlines()

        @property
        def file_path_obj(self):
            """Convert file_path string to Path object for compatibility."""
            return Path(self.file_path)


else:  # pragma: no cover - used when sqlalchemy is absent
    class FindingRecord:
        pass


@dataclass
class ModuleStats:
    module_name: str
    last_severity: Optional[str]
    last_log_update: Optional[datetime.datetime]
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


def update_finding_llm_error(
    finding_id: int,
    *,
    error: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
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
                    "llm_response_content": None,
                    "llm_error": error,
                    "llm_prompt_tokens": None,
                    "llm_completion_tokens": None,
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


def _normalize_created_at(value: Optional[datetime.datetime]) -> Optional[datetime.datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.timezone.utc)
    return value


def store_finding(module_name: str, finding, anomaly_flag: bool = False):
    """Persist a single Finding into the database."""
    sess = get_session()
    if sess is None:
        return

    llm_response = getattr(finding, "llm_response", None)
    llm_error = getattr(finding, "llm_error", None)
    created_at = _normalize_created_at(getattr(finding, "created_at", None))

    # Check for duplicate finding to prevent re-inserting the same issue
    try:
        existing = (
            sess.query(FindingRecord)
            .filter(FindingRecord.module_name == module_name)
            .filter(FindingRecord.file_path == str(getattr(finding, "file_path", "")))
            .filter(FindingRecord.line_start == int(getattr(finding, "line_start", 0)))
            .filter(FindingRecord.line_end == int(getattr(finding, "line_end", 0)))
            .filter(FindingRecord.message == str(getattr(finding, "message", "")))
            .filter(FindingRecord.severity == str(
                getattr(getattr(finding, "severity", None), "name", getattr(finding, "severity", "UNKNOWN"))
            ))
            .first()
        )
        if existing:
            # Duplicate found, don't store again
            sess.close()
            return
    except Exception:
        # If duplicate check fails, continue with storing
        pass

    record_kwargs = dict(
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
        needs_llm=bool(getattr(finding, "needs_llm", False)),
        llm_provider=getattr(llm_response, "provider", None),
        llm_model=getattr(llm_response, "model", None),
        llm_response_content=getattr(llm_response, "content", None),
        llm_error=str(llm_error) if llm_error else None,
        llm_prompt_tokens=getattr(llm_response, "prompt_tokens", None),
        llm_completion_tokens=getattr(llm_response, "completion_tokens", None),
    )
    if created_at is not None:
        record_kwargs["created_at"] = created_at

    obj = FindingRecord(**record_kwargs)
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


def _collect_file_activity(modules: Optional[Iterable["ModuleConfig"]]) -> Dict[str, ModuleStats]:
    """Build initial stats from module file mtimes."""

    stats: Dict[str, ModuleStats] = {}
    if not modules:
        return stats

    for mod in modules:
        mtime: Optional[datetime.datetime] = None
        try:
            st = mod.path.stat()
            mtime = datetime.datetime.fromtimestamp(st.st_mtime, tz=datetime.timezone.utc)
        except FileNotFoundError:
            mtime = None
        stats[mod.name] = ModuleStats(
            module_name=mod.name,
            last_severity=None,
            last_log_update=mtime,
            errors_24h=0,
            warnings_24h=0,
        )

    return stats


def _ensure_tzaware(dt: datetime.datetime) -> datetime.datetime:
    """Guarantee a datetime has timezone info, assuming UTC when missing."""

    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def get_module_stats(modules: Optional[Iterable["ModuleConfig"]] = None) -> Dict[str, ModuleStats]:
    """Return basic stats per module for the last 24h and last log update."""

    stats = _collect_file_activity(modules)

    sess = get_session()
    if sess is None:
        return stats

    now = datetime.datetime.now(datetime.timezone.utc)
    window_start = now - datetime.timedelta(days=1)

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
                    last_log_update=None,
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
            if row.created_at:
                row_ts = _ensure_tzaware(row.created_at)
                last_ts = _ensure_tzaware(s.last_log_update) if s.last_log_update else None
                if last_ts is None or row_ts > last_ts:
                    s.last_log_update = row_ts
    finally:
        sess.close()

    return stats


def count_open_findings_for_module(
    module_name: str, *, severities: Optional[Iterable[str]] = None
) -> int:
    """Count findings for a module, optionally constrained to severities.

    Returns 0 if the database is unavailable or the query fails.
    """

    sess = get_session()
    if sess is None or not module_name:
        return 0

    try:
        severity_values = [s.upper() for s in (severities or []) if s]
        query = sess.query(func.count(FindingRecord.id)).filter(
            FindingRecord.module_name == module_name
        )
        if severity_values:
            query = query.filter(func.upper(FindingRecord.severity).in_(severity_values))
        return int(query.scalar() or 0)
    except Exception:
        return 0
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


def delete_findings_matching_regex(pattern: str, pipeline_name: Optional[str] = None) -> int:
    """Delete all findings whose message or excerpt matches the given regex.

    When ``pipeline_name`` is provided, the match is limited to that pipeline.
    Returns the number of deleted rows.
    """

    sess = get_session()
    if sess is None:
        return 0

    try:
        regex = re.compile(pattern)
    except re.error:
        return 0

    try:
        query = sess.query(FindingRecord)
        if pipeline_name:
            query = query.filter(FindingRecord.pipeline_name == pipeline_name)

        ids = [
            row.id
            for row in query.all()
            if regex.search(row.message or "") or regex.search(row.excerpt or "")
        ]
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
                    "llm_error": None,
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
