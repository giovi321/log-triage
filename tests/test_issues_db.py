"""Tests for de-duplication into issues at the DB layer."""
import datetime
from pathlib import Path

import pytest

from logtriage.models import Finding, Severity
from logtriage.webui import db


@pytest.fixture()
def database(tmp_path):
    url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    # reset any prior global engine, then set up a fresh one
    db._engine = None
    db._db_url = None
    db.setup_database(url)
    yield url
    db._engine = None
    db._db_url = None


def _finding(line_text, *, sev=Severity.ERROR, line=1, rule=r"\bERROR\b", ts=None, pipeline="ha"):
    return Finding(
        file_path=Path("/var/log/x.log"),
        pipeline_name=pipeline,
        finding_index=0,
        severity=sev,
        message=f'Matched error pattern /{rule}/ on "ERROR"',
        line_start=line,
        line_end=line,
        rule_id=rule,
        excerpt=[line_text],
        needs_llm=False,
        created_at=ts or datetime.datetime.now(datetime.timezone.utc),
    )


def test_recurring_findings_collapse_to_one_issue(database):
    db.store_finding("ha", _finding("2026-05-29 14:00:00 ERROR mqtt cannot reach 10.0.0.5", line=1))
    db.store_finding("ha", _finding("2026-05-29 15:30:00 ERROR mqtt cannot reach 10.0.0.9", line=2))
    db.store_finding("ha", _finding("2026-05-29 16:45:00 ERROR mqtt cannot reach 10.0.0.1", line=3))
    issues = db.get_issues(module_name="ha")
    assert len(issues) == 1
    assert issues[0].occurrence_count == 3
    assert issues[0].status == "open"


def test_distinct_errors_make_distinct_issues(database):
    db.store_finding("ha", _finding("ERROR disk full on /dev/sda", line=1, rule="ERROR"))
    db.store_finding("ha", _finding("ERROR network unreachable", line=2, rule="ERROR"))
    issues = db.get_issues(module_name="ha")
    assert len(issues) == 2


def test_max_severity_is_tracked(database):
    db.store_finding("ha", _finding("WARNING slow response", sev=Severity.WARNING, line=1, rule="WARNING"))
    # same signature shape but escalated severity should bump the issue
    db.store_finding("ha", _finding("ERROR slow response", sev=Severity.ERROR, line=2, rule="ERROR"))
    # these are different signatures (different rule/text); check severities recorded
    sevs = sorted(i.severity for i in db.get_issues(module_name="ha"))
    assert "ERROR" in sevs and "WARNING" in sevs


def test_status_update_and_reopen(database):
    db.store_finding("ha", _finding("ERROR boom", line=1, rule="ERROR"))
    issue = db.get_issues(module_name="ha")[0]
    assert db.update_issue_status(issue.id, "resolved")
    assert db.get_issue_by_id(issue.id).status == "resolved"
    # a recurrence re-opens it
    db.store_finding("ha", _finding("ERROR boom", line=2, rule="ERROR"))
    reopened = db.get_issue_by_id(issue.id)
    assert reopened.status == "open"
    assert reopened.occurrence_count == 2


def test_priority_orders_critical_above_warning(database):
    now = datetime.datetime.now(datetime.timezone.utc)
    db.store_finding("ha", _finding("WARNING minor", sev=Severity.WARNING, line=1, rule="WARNING", ts=now))
    db.store_finding("ha", _finding("CRITICAL meltdown", sev=Severity.CRITICAL, line=2, rule="CRITICAL", ts=now))
    issues = db.get_issues(module_name="ha", now=now)
    assert issues[0].severity == "CRITICAL"  # highest priority first


def test_backfill_assigns_issues_to_legacy_findings(database):
    # Simulate pre-dedup rows: insert FindingRecords with NULL issue_id directly.
    sess = db.get_session()
    for i in range(3):
        sess.add(db.FindingRecord(
            module_name="legacy",
            pipeline_name="ha",
            file_path="/var/log/x.log",
            finding_index=i,
            severity="ERROR",
            message="legacy",
            line_start=i + 1,
            line_end=i + 1,
            rule_id="ERROR",
            excerpt=f"2026-01-0{i+1} 00:00:00 ERROR recurring boom {i}",
            created_at=datetime.datetime.now(datetime.timezone.utc),
        ))
    sess.commit()
    sess.close()

    processed = db.backfill_issues()
    assert processed == 3
    issues = db.get_issues(module_name="legacy")
    assert len(issues) == 1
    assert issues[0].occurrence_count == 3


def test_sparkline_buckets_occurrences(database):
    now = datetime.datetime.now(datetime.timezone.utc)
    db.store_finding("ha", _finding("ERROR x", line=1, rule="ERROR", ts=now))
    db.store_finding("ha", _finding("ERROR x", line=2, rule="ERROR", ts=now))
    issue = db.get_issues(module_name="ha")[0]
    spark = db.get_issue_sparkline(issue.id, buckets=24, bucket_seconds=3600, now=now + datetime.timedelta(seconds=1))
    assert sum(spark) == 2
    assert len(spark) == 24


def test_store_findings_batch_matches_per_finding(database):
    """store_findings (one transaction) equals N store_finding calls."""
    now = datetime.datetime.now(datetime.timezone.utc)
    batch = [
        _finding("2026-05-29 14:00:00 ERROR mqtt cannot reach 10.0.0.5", line=1, ts=now),
        _finding("2026-05-29 15:00:00 ERROR mqtt cannot reach 10.0.0.9", line=2, ts=now),
        _finding("ERROR disk full on /dev/sda", line=3, rule="DISK", ts=now),
    ]
    stored = db.store_findings("ha", batch)
    assert stored == 3
    issues = db.get_issues(module_name="ha")
    # Two distinct signatures: the recurring mqtt error (count 2) + the disk one.
    assert sorted(i.occurrence_count for i in issues) == [1, 2]
    assert db.count_findings() == 3

    # A duplicate occurrence in a later batch is skipped (0 inserted).
    again = db.store_findings("ha", [batch[0]])
    assert again == 0
    assert db.count_findings() == 3


def test_batch_sparklines_match_per_issue(database):
    """get_issue_sparklines (one query) must equal per-issue get_issue_sparkline."""
    now = datetime.datetime.now(datetime.timezone.utc)
    db.store_finding("ha", _finding("ERROR aaa", line=1, rule="ERROR", ts=now))
    db.store_finding("ha", _finding("ERROR aaa", line=2, rule="ERROR", ts=now))
    db.store_finding("ha", _finding("ERROR bbb", line=3, rule="BERROR", ts=now))
    at = now + datetime.timedelta(seconds=1)
    issues = db.get_issues(module_name="ha")
    ids = [i.id for i in issues]

    batch = db.get_issue_sparklines(ids, buckets=24, bucket_seconds=3600, now=at)
    assert set(batch) == set(ids)
    for iid in ids:
        per = db.get_issue_sparkline(iid, buckets=24, bucket_seconds=3600, now=at)
        assert batch[iid] == per

    # An id with no occurrences still yields an all-zero series of correct length.
    empty = db.get_issue_sparklines([999999], buckets=24, bucket_seconds=3600, now=at)
    assert empty == {999999: [0] * 24}
    assert db.get_issue_sparklines([], now=at) == {}


def test_module_stats_counts_and_latest(database):
    """SQL-aggregated get_module_stats: counts by severity + latest finding."""
    base = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2)
    db.store_finding("ha", _finding("ERROR one", line=1, rule="ERR1", ts=base))
    db.store_finding("ha", _finding("WARNING two", sev=Severity.WARNING, line=2, rule="WARN2", ts=base + datetime.timedelta(minutes=1)))
    db.store_finding("ha", _finding("CRITICAL three", sev=Severity.CRITICAL, line=3, rule="CRIT3", ts=base + datetime.timedelta(minutes=2)))
    # An old finding outside the 24h window must not be counted.
    old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=3)
    db.store_finding("ha", _finding("ERROR ancient", line=4, rule="OLD4", ts=old))

    stats = db.get_module_stats()
    s = stats["ha"]
    assert s.errors_24h == 2  # ERROR + CRITICAL, old one excluded
    assert s.warnings_24h == 1
    assert s.last_severity == "CRITICAL"  # most recent in-window finding
