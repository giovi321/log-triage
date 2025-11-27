import pytest

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logtriage.classifiers.regex_counter import _build_excerpt


def test_excerpt_always_includes_match_when_prefix_overflows_limit():
    lines = ["matched line"]
    prefix_lines = [f"prefix {i}" for i in range(6)]

    excerpt = _build_excerpt(
        lines=lines,
        offset=0,
        context_prefix_lines=5,
        excerpt_limit=2,
        prefix_lines=prefix_lines,
    )

    assert "matched line" in excerpt
    assert len(excerpt) == 2


def test_excerpt_prefers_nearby_context_before_and_after():
    lines = ["alpha", "beta", "match here", "delta", "epsilon", "zeta"]

    excerpt = _build_excerpt(
        lines=lines,
        offset=2,
        context_prefix_lines=1,
        excerpt_limit=3,
        prefix_lines=[],
    )

    assert excerpt == ["beta", "match here", "delta"]


def test_excerpt_uses_prefix_buffer_when_not_enough_history():
    lines = ["match here"]
    prefix_lines = ["older 1", "older 2", "older 3"]

    excerpt = _build_excerpt(
        lines=lines,
        offset=0,
        context_prefix_lines=3,
        excerpt_limit=3,
        prefix_lines=prefix_lines,
    )

    assert excerpt == ["older 2", "older 3", "match here"]
