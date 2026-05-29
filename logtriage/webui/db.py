from __future__ import annotations

import datetime
import importlib.util
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Dict, List, Tuple, TYPE_CHECKING

from ..models import Severity, LLMResponse
from ..fingerprint import signature_for_finding, Signature

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
        UniqueConstraint,
        create_engine,
        inspect,
        func,
        or_,
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
        # Phase 2: signature-based de-duplication into issues
        ("fingerprint", "VARCHAR(32)"),
        ("issue_id", "INTEGER"),
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
        # Phase 2: signature-based de-duplication
        fingerprint = Column(String(32), index=True, nullable=True)
        issue_id = Column(Integer, index=True, nullable=True)

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


    class IssueRecord(Base):
        """A de-duplicated issue: many findings sharing one signature.

        Aggregates recurring findings into a single triage unit with an
        occurrence count, a first/last-seen window, the highest severity seen,
        a workflow status, and (Phase 3) a cached LLM analysis keyed on the
        fingerprint so each unique signature is analyzed at most once.
        """
        __tablename__ = "issues"
        __table_args__ = (
            UniqueConstraint("module_name", "fingerprint", name="uq_issue_module_fingerprint"),
        )

        id = Column(Integer, primary_key=True)
        module_name = Column(String(128), index=True, nullable=False)
        pipeline_name = Column(String(128), nullable=True)
        fingerprint = Column(String(32), index=True, nullable=False)
        signature = Column(Text, nullable=False)
        title = Column(Text, nullable=False)
        severity = Column(String(16), index=True, nullable=False, default="WARNING")
        status = Column(String(24), index=True, nullable=False, default="open")
        occurrence_count = Column(Integer, nullable=False, default=0)
        first_seen = Column(DateTime(timezone=True), nullable=True)
        last_seen = Column(DateTime(timezone=True), index=True, nullable=True)
        rule_id = Column(String(256), nullable=True)
        sample_excerpt = Column(Text, nullable=True)
        # Cached per-issue LLM analysis (populated by the Phase 3/4 worker)
        llm_provider = Column(String(128), nullable=True)
        llm_model = Column(String(128), nullable=True)
        llm_content = Column(Text, nullable=True)
        llm_citations = Column(Text, nullable=True)  # JSON-encoded list[str]
        llm_analyzed_fingerprint = Column(String(32), nullable=True)
        llm_error = Column(Text, nullable=True)
        llm_prompt_tokens = Column(Integer, nullable=True)
        llm_completion_tokens = Column(Integer, nullable=True)
        llm_updated_at = Column(DateTime(timezone=True), nullable=True)
        created_at = Column(
            DateTime(timezone=True),
            nullable=False,
            default=lambda: datetime.datetime.now(datetime.timezone.utc),
        )
        updated_at = Column(
            DateTime(timezone=True),
            nullable=True,
            default=lambda: datetime.datetime.now(datetime.timezone.utc),
        )

        @property
        def severity_enum(self):
            try:
                return Severity.from_string(self.severity or "WARNING")
            except (ValueError, AttributeError, TypeError):
                return Severity.WARNING

        @property
        def citations(self) -> List[str]:
            if not self.llm_citations:
                return []
            try:
                data = json.loads(self.llm_citations)
                return list(data) if isinstance(data, list) else []
            except (ValueError, TypeError):
                return []

        @property
        def has_llm_analysis(self) -> bool:
            return bool(self.llm_content) and self.llm_analyzed_fingerprint == self.fingerprint

        @property
        def sample_excerpt_as_list(self) -> List[str]:
            return self.sample_excerpt.splitlines() if self.sample_excerpt else []


else:  # pragma: no cover - used when sqlalchemy is absent
    class FindingRecord:
        pass

    class IssueRecord:
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


def _severity_rank(name: Optional[str]) -> int:
    """Numeric ordering for a severity name (higher = more severe)."""
    try:
        return int(Severity.from_string(str(name)))
    except (ValueError, AttributeError, TypeError):
        return 0


def _finding_severity_name(finding) -> str:
    return str(
        getattr(getattr(finding, "severity", None), "name", getattr(finding, "severity", "UNKNOWN"))
    )


def _finding_excerpt_text(finding) -> str:
    exc = getattr(finding, "excerpt", None)
    if isinstance(exc, str):
        return exc
    if isinstance(exc, (list, tuple)):
        return "\n".join(str(x) for x in exc)
    return ""


def _upsert_issue(
    sess, module_name: str, finding, sig: "Signature", ts: datetime.datetime
) -> Optional[int]:
    """Find-or-create the issue for a finding's signature and fold the finding in.

    Returns the issue id. Operates within the caller's session/transaction
    (the caller commits), so the issue and the finding row land together.
    """
    sev = _finding_severity_name(finding)
    issue = (
        sess.query(IssueRecord)
        .filter(IssueRecord.module_name == module_name)
        .filter(IssueRecord.fingerprint == sig.fingerprint)
        .one_or_none()
    )

    if issue is None:
        issue = IssueRecord(
            module_name=module_name,
            pipeline_name=getattr(finding, "pipeline_name", None),
            fingerprint=sig.fingerprint,
            signature=sig.signature,
            title=sig.title,
            severity=sev,
            status="open",
            occurrence_count=1,
            first_seen=ts,
            last_seen=ts,
            rule_id=getattr(finding, "rule_id", None),
            sample_excerpt=_finding_excerpt_text(finding),
            created_at=ts,
            updated_at=ts,
        )
        sess.add(issue)
        sess.flush()  # assign primary key
        return issue.id

    issue.occurrence_count = (issue.occurrence_count or 0) + 1
    if ts is not None:
        ts_aware = _ensure_tzaware(ts)
        if issue.last_seen is None or ts_aware > _ensure_tzaware(issue.last_seen):
            issue.last_seen = ts
        if issue.first_seen is None or ts_aware < _ensure_tzaware(issue.first_seen):
            issue.first_seen = ts
    if _severity_rank(sev) > _severity_rank(issue.severity):
        issue.severity = sev
    if not issue.sample_excerpt:
        issue.sample_excerpt = _finding_excerpt_text(finding)
    # A recurrence re-opens an issue that was marked resolved.
    if issue.status == "resolved":
        issue.status = "open"
    issue.updated_at = ts
    return issue.id


def store_finding(module_name: str, finding, anomaly_flag: bool = False):
    """Persist a single Finding and fold it into its de-duplicated issue."""
    sess = get_session()
    if sess is None:
        return

    llm_response = getattr(finding, "llm_response", None)
    llm_error = getattr(finding, "llm_error", None)
    created_at = _normalize_created_at(getattr(finding, "created_at", None))
    ts = created_at or datetime.datetime.now(datetime.timezone.utc)

    # Check for duplicate finding to prevent re-inserting the same occurrence
    try:
        existing = (
            sess.query(FindingRecord)
            .filter(FindingRecord.module_name == module_name)
            .filter(FindingRecord.file_path == str(getattr(finding, "file_path", "")))
            .filter(FindingRecord.line_start == int(getattr(finding, "line_start", 0)))
            .filter(FindingRecord.line_end == int(getattr(finding, "line_end", 0)))
            .filter(FindingRecord.message == str(getattr(finding, "message", "")))
            .filter(FindingRecord.severity == _finding_severity_name(finding))
            .first()
        )
        if existing:
            # Duplicate found, don't store again (and don't double-count the issue)
            sess.close()
            return
    except Exception:
        # If duplicate check fails, continue with storing
        pass

    # Signature + issue aggregation
    fingerprint: Optional[str] = None
    issue_id: Optional[int] = None
    try:
        sig = signature_for_finding(finding)
        fingerprint = sig.fingerprint
        issue_id = _upsert_issue(sess, module_name, finding, sig, ts)
    except Exception:
        # Never let fingerprinting break finding persistence.
        sess.rollback()
        fingerprint = None
        issue_id = None

    record_kwargs = dict(
        module_name=module_name,
        pipeline_name=getattr(finding, "pipeline_name", None),
        file_path=str(getattr(finding, "file_path", "")),
        finding_index=int(getattr(finding, "finding_index", 0)),
        severity=_finding_severity_name(finding),
        message=str(getattr(finding, "message", "")),
        line_start=int(getattr(finding, "line_start", 0)),
        line_end=int(getattr(finding, "line_end", 0)),
        rule_id=getattr(finding, "rule_id", None),
        excerpt="\n".join(getattr(finding, "excerpt", []) or []),
        anomaly_flag=bool(anomaly_flag),
        needs_llm=bool(getattr(finding, "needs_llm", False)),
        fingerprint=fingerprint,
        issue_id=issue_id,
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


def get_max_finding_id() -> int:
    """Highest finding id (used as a cheap change cursor for live updates)."""
    sess = get_session()
    if sess is None:
        return 0
    try:
        return int(sess.query(func.max(FindingRecord.id)).scalar() or 0)
    except Exception:
        return 0
    finally:
        sess.close()


def count_findings() -> int:
    """Total number of stored findings."""
    sess = get_session()
    if sess is None:
        return 0
    try:
        return int(sess.query(func.count(FindingRecord.id)).scalar() or 0)
    except Exception:
        return 0
    finally:
        sess.close()


def get_findings_for_issue(issue_id: int, limit: int = 50) -> List[FindingRecord]:
    """Recent occurrences (findings) belonging to one issue, newest first."""
    sess = get_session()
    if sess is None:
        return []
    try:
        return (
            sess.query(FindingRecord)
            .filter(FindingRecord.issue_id == issue_id)
            .order_by(FindingRecord.created_at.desc())
            .limit(limit)
            .all()
        )
    except Exception:
        return []
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


# ---------------------------------------------------------------------------
# Issues (de-duplicated findings)
# ---------------------------------------------------------------------------

ISSUE_STATUSES = ("open", "acknowledged", "resolved", "muted", "false_positive")
ISSUE_ACTIVE_STATUSES = ("open", "acknowledged")


def issue_priority(issue, now: Optional[datetime.datetime] = None) -> float:
    """Score an issue for triage ordering.

    Combines severity, recency (how recently it last fired), rate (occurrences
    per day over its lifetime), and novelty (a bump for brand-new issues).
    Acknowledged issues are de-prioritised; resolved/muted/false-positive sink.
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)

    base = {3: 100.0, 2: 60.0, 1: 25.0}.get(_severity_rank(getattr(issue, "severity", "WARNING")), 10.0)

    last = getattr(issue, "last_seen", None)
    first = getattr(issue, "first_seen", None)
    count = getattr(issue, "occurrence_count", 0) or 0

    recency = 0.0
    if last is not None:
        age_h = max(0.0, (now - _ensure_tzaware(last)).total_seconds() / 3600.0)
        recency = max(0.0, 40.0 * (1.0 - age_h / 72.0))  # decays to 0 over ~3 days

    rate_score = 0.0
    if last is not None and first is not None:
        span_h = max(1.0, (_ensure_tzaware(last) - _ensure_tzaware(first)).total_seconds() / 3600.0)
        per_day = count / (span_h / 24.0)
        rate_score = min(35.0, 12.0 * math.log10(per_day + 1.0))
    elif count > 1:
        rate_score = min(35.0, 12.0 * math.log10(count + 1.0))

    novelty = 0.0
    if first is not None and (now - _ensure_tzaware(first)).total_seconds() <= 3600.0:
        novelty = 15.0

    status_factor = {"open": 0.0, "acknowledged": -40.0}.get(getattr(issue, "status", "open"), -1000.0)

    return round(base + recency + rate_score + novelty + status_factor, 2)


def get_issues(
    module_name: Optional[str] = None,
    statuses: Optional[Iterable[str]] = None,
    severities: Optional[Iterable[str]] = None,
    search: Optional[str] = None,
    limit: int = 300,
    now: Optional[datetime.datetime] = None,
) -> List["IssueRecord"]:
    """Return issues (optionally filtered), ordered by triage priority desc."""
    sess = get_session()
    if sess is None:
        return []
    try:
        q = sess.query(IssueRecord)
        if module_name:
            q = q.filter(IssueRecord.module_name == module_name)
        status_list = [s for s in (statuses or []) if s]
        if status_list:
            q = q.filter(IssueRecord.status.in_(status_list))
        sev_list = [s.upper() for s in (severities or []) if s]
        if sev_list:
            q = q.filter(func.upper(IssueRecord.severity).in_(sev_list))
        if search:
            like = f"%{search}%"
            q = q.filter(
                or_(
                    IssueRecord.title.ilike(like),
                    IssueRecord.signature.ilike(like),
                    IssueRecord.rule_id.ilike(like),
                )
            )
        rows = q.order_by(IssueRecord.last_seen.desc()).limit(max(1, limit)).all()
    except Exception:
        return []
    finally:
        sess.close()

    now = now or datetime.datetime.now(datetime.timezone.utc)
    for r in rows:
        try:
            r.priority_score = issue_priority(r, now)
        except Exception:
            r.priority_score = 0.0
    rows.sort(key=lambda r: getattr(r, "priority_score", 0.0), reverse=True)
    return rows


def get_issue_by_id(issue_id: int) -> Optional["IssueRecord"]:
    sess = get_session()
    if sess is None:
        return None
    try:
        issue = sess.query(IssueRecord).filter(IssueRecord.id == issue_id).one_or_none()
        if issue is not None:
            issue.priority_score = issue_priority(issue)
        return issue
    except Exception:
        return None
    finally:
        sess.close()


def update_issue_status(issue_id: int, status: str) -> bool:
    if status not in ISSUE_STATUSES:
        return False
    sess = get_session()
    if sess is None:
        return False
    try:
        updated = (
            sess.query(IssueRecord)
            .filter(IssueRecord.id == issue_id)
            .update(
                {"status": status, "updated_at": datetime.datetime.now(datetime.timezone.utc)},
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


def issue_status_counts(module_name: Optional[str] = None) -> Dict[str, int]:
    """Count issues by status (for the triage dashboard tiles)."""
    counts = {s: 0 for s in ISSUE_STATUSES}
    sess = get_session()
    if sess is None:
        return counts
    try:
        q = sess.query(IssueRecord.status, func.count(IssueRecord.id))
        if module_name:
            q = q.filter(IssueRecord.module_name == module_name)
        for status, n in q.group_by(IssueRecord.status).all():
            counts[str(status)] = int(n or 0)
    except Exception:
        pass
    finally:
        sess.close()
    return counts


def get_issue_sparkline(
    issue_id: int,
    buckets: int = 24,
    bucket_seconds: int = 3600,
    now: Optional[datetime.datetime] = None,
) -> List[int]:
    """Occurrence counts per time bucket (oldest→newest) for a sparkline."""
    out = [0] * buckets
    sess = get_session()
    if sess is None:
        return out
    now = now or datetime.datetime.now(datetime.timezone.utc)
    start = now - datetime.timedelta(seconds=buckets * bucket_seconds)
    try:
        rows = (
            sess.query(FindingRecord.created_at)
            .filter(FindingRecord.issue_id == issue_id)
            .filter(FindingRecord.created_at >= start)
            .all()
        )
    except Exception:
        return out
    finally:
        sess.close()

    for (ts,) in rows:
        if ts is None:
            continue
        idx = int((_ensure_tzaware(ts) - start).total_seconds() // bucket_seconds)
        if 0 <= idx < buckets:
            out[idx] += 1
    return out


def backfill_issues(batch_size: int = 500) -> int:
    """Assign fingerprints/issues to findings created before de-dup existed.

    Idempotent: only touches findings whose ``issue_id`` is NULL. Uses an
    id-cursor so it always makes forward progress even if a row fails.
    """
    sess = get_session()
    if sess is None:
        return 0

    processed = 0
    last_id = 0
    try:
        while True:
            rows = (
                sess.query(FindingRecord)
                .filter(FindingRecord.issue_id.is_(None))
                .filter(FindingRecord.id > last_id)
                .order_by(FindingRecord.id.asc())
                .limit(batch_size)
                .all()
            )
            if not rows:
                break
            for rec in rows:
                last_id = max(last_id, int(rec.id))
                try:
                    sig = signature_for_finding(rec)
                    ts = _ensure_tzaware(rec.created_at) if rec.created_at else datetime.datetime.now(datetime.timezone.utc)
                    rec.issue_id = _upsert_issue(sess, rec.module_name, rec, sig, ts)
                    rec.fingerprint = sig.fingerprint
                    processed += 1
                except Exception:
                    continue
            sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()
    return processed


# ---- per-issue LLM cache (used by Phase 3/4 enrichment) -------------------

def update_issue_llm(
    issue_id: int,
    *,
    provider: Optional[str],
    model: Optional[str],
    content: Optional[str],
    citations: Optional[List[str]] = None,
    fingerprint: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> bool:
    sess = get_session()
    if sess is None:
        return False
    try:
        updated = (
            sess.query(IssueRecord)
            .filter(IssueRecord.id == issue_id)
            .update(
                {
                    "llm_provider": provider,
                    "llm_model": model,
                    "llm_content": content,
                    "llm_citations": json.dumps(citations or []),
                    "llm_analyzed_fingerprint": fingerprint,
                    "llm_error": None,
                    "llm_prompt_tokens": prompt_tokens,
                    "llm_completion_tokens": completion_tokens,
                    "llm_updated_at": datetime.datetime.now(datetime.timezone.utc),
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


def update_issue_llm_error(issue_id: int, error: str, *, provider: Optional[str] = None, model: Optional[str] = None) -> bool:
    sess = get_session()
    if sess is None:
        return False
    try:
        updated = (
            sess.query(IssueRecord)
            .filter(IssueRecord.id == issue_id)
            .update(
                {
                    "llm_provider": provider,
                    "llm_model": model,
                    "llm_error": error,
                    "llm_updated_at": datetime.datetime.now(datetime.timezone.utc),
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


def get_issues_needing_llm(
    module_names: Optional[Iterable[str]] = None,
    limit: int = 20,
) -> List["IssueRecord"]:
    """Active issues whose cached analysis is missing or stale (signature changed)."""
    sess = get_session()
    if sess is None:
        return []
    try:
        q = sess.query(IssueRecord).filter(IssueRecord.status.in_(ISSUE_ACTIVE_STATUSES))
        names = [n for n in (module_names or []) if n]
        if names:
            q = q.filter(IssueRecord.module_name.in_(names))
        q = q.filter(
            or_(
                IssueRecord.llm_content.is_(None),
                IssueRecord.llm_analyzed_fingerprint.is_(None),
                IssueRecord.llm_analyzed_fingerprint != IssueRecord.fingerprint,
            )
        )
        return q.order_by(IssueRecord.last_seen.desc()).limit(max(1, limit)).all()
    except Exception:
        return []
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
