"""Behavior-locking tests for the single-pass separator grouping."""
import re

from logtriage.grouping.separator import group_by_separator


SEP = re.compile(r"^=== RUN")


def test_no_separator_pattern_returns_single_chunk():
    assert group_by_separator(["a", "b"], None) == [["a", "b"]]
    assert group_by_separator([], None) == []


def test_no_matches_treats_all_as_one_run():
    assert group_by_separator(["a", "b", "c"], SEP) == [["a", "b", "c"]]


def test_splits_on_separators_excluding_separator_lines():
    lines = ["pre1", "=== RUN 1", "a", "b", "=== RUN 2", "c"]
    # text before the first separator is its own run; separator lines dropped.
    assert group_by_separator(lines, SEP) == [["pre1"], ["a", "b"], ["c"]]


def test_empty_runs_between_adjacent_separators_skipped():
    lines = ["=== RUN 1", "=== RUN 2", "x"]
    assert group_by_separator(lines, SEP) == [["x"]]


def test_only_last_returns_tail_after_final_separator():
    lines = ["a", "=== RUN 1", "b", "=== RUN 2", "c", "d"]
    assert group_by_separator(lines, SEP, only_last=True) == [["c", "d"]]


def test_only_last_empty_when_separator_is_final_line():
    lines = ["a", "=== RUN 1", "b", "=== RUN 2"]
    assert group_by_separator(lines, SEP, only_last=True) == []
