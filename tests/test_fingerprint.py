"""Tests for signature/fingerprint normalisation and grouping."""
from logtriage.fingerprint import compute, normalize, representative_line


def test_normalize_strips_timestamps_ips_and_numbers():
    a = normalize("2026-05-29T14:31:02 ERROR mqtt: cannot connect to 10.0.0.5:1883 after 3 tries")
    b = normalize("2026-05-29T15:00:59 ERROR mqtt: cannot connect to 10.0.0.9:1883 after 17 tries")
    assert a == b
    assert "<TS>" in a and "<IP>" in a and "<N>" in a


def test_normalize_handles_uuid_and_hex():
    a = normalize("request 1b4e28ba-2fa1-11d2-883f-0016d3cca427 failed token deadbeefcafe1234")
    assert "<UUID>" in a
    assert "<HEX>" in a


def test_compute_groups_recurring_lines():
    s1 = compute("ha", "ERROR", r"\bERROR\b",
                 ["2026-05-29 14:00:00 ERROR worker failed after 3 retries"], "")
    s2 = compute("ha", "ERROR", r"\bERROR\b",
                 ["2026-05-29 19:22:11 ERROR worker failed after 99 retries"], "")
    assert s1.fingerprint == s2.fingerprint
    assert s1.signature == s2.signature


def test_compute_distinguishes_different_errors():
    s1 = compute("ha", "ERROR", "ERROR", ["ERROR disk full on /dev/sda"], "")
    s2 = compute("ha", "ERROR", "ERROR", ["ERROR network unreachable"], "")
    assert s1.fingerprint != s2.fingerprint


def test_severity_and_pipeline_affect_fingerprint():
    base = dict(rule_id="x", excerpt=["x happened here"], message="")
    assert compute("ha", "ERROR", **base).fingerprint != compute("ha", "WARNING", **base).fingerprint
    assert compute("ha", "ERROR", **base).fingerprint != compute("nextcloud", "ERROR", **base).fingerprint


def test_representative_line_prefers_rule_match():
    rep = representative_line(r"Traceback",
                              ["context above", "Traceback (most recent call last):", "context below"],
                              "Matched error pattern")
    assert rep.startswith("Traceback")


def test_representative_line_falls_back_to_longest():
    rep = representative_line(None, ["short", "a much longer substantive line here"], "msg")
    assert rep == "a much longer substantive line here"
