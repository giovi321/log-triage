import pytest

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logtriage.classifiers.regex_counter import _build_excerpt
from logtriage.config import _compile_regex
from logtriage.models import PipelineConfig
from logtriage.classifiers import classify_lines


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


def test_classify_detects_patterns_with_double_escaped_config_entries():
    pcfg = PipelineConfig(
        name="homeassistant",
        match_filename_regex=_compile_regex("homeassistant.*\\.log"),
        classifier_type="regex_counter",
        classifier_error_regexes=[_compile_regex(r"\\berror\\b", flags=re.IGNORECASE)],
        classifier_warning_regexes=[],
        classifier_ignore_regexes=[],
        grouping_type="whole_file",
        grouping_start_regex=None,
        grouping_end_regex=None,
        grouping_only_last=False,
    )

    lines = [
        "2025-11-27 12:54:37.917 ERROR (MainThread) [frontend.js] Uncaught error from Firefox 145.0 on Ubuntu",
        "Error: Failed to execute 'define' on 'CustomElementRegistry': the name \"mushroom-select\" has already been used with this registry",
    ]

    findings = classify_lines(
        pcfg,
        file_path="homeassistant.log",
        pipeline_name="homeassistant",
        lines=lines,
    )

    assert len(findings) == 3
    assert findings[0].excerpt[0] == lines[0]
    assert findings[1].excerpt[0] == lines[0]
    assert findings[2].excerpt[0] == lines[1]
