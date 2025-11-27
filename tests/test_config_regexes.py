import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logtriage.config import _compile_regex


def test_compile_regex_unescapes_word_boundaries():
    pattern = _compile_regex(r"\\berror\\b", flags=re.IGNORECASE)

    assert pattern is not None
    assert pattern.search("ERROR logged")
    assert pattern.search("some error happened")


def test_compile_regex_keeps_escaped_quotes_and_whitespace():
    pattern = _compile_regex(r'"level"\\s*:\\s*(3|4)', flags=0)

    assert pattern is not None
    assert pattern.search('"level": 3')
    assert pattern.search('"level"   :   4')


def test_compile_regex_restores_word_boundaries_from_backspace():
    pattern = _compile_regex("\berror\b", flags=re.IGNORECASE)

    assert pattern is not None
    assert pattern.search("error message present")
