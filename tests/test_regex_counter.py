import re
import unittest
from pathlib import Path

from logtriage.classifiers.regex_counter import classify_regex_counter
from logtriage.models import PipelineConfig


def _make_pipeline(error_patterns, warning_patterns):
    return PipelineConfig(
        name="test",
        match_filename_regex=re.compile(r".*"),
        classifier_type="regex_counter",
        classifier_error_regexes=[re.compile(p, re.IGNORECASE) for p in error_patterns],
        classifier_warning_regexes=[re.compile(p, re.IGNORECASE) for p in warning_patterns],
        classifier_ignore_regexes=[],
    )


class RegexCounterTests(unittest.TestCase):
    def test_error_message_includes_matched_text(self):
        pcfg = _make_pipeline([r"Traceback"], [])
        lines = ["prefix Traceback suffix"]

        findings = classify_regex_counter(pcfg, Path("/tmp/test.log"), "pipe", lines)

        self.assertTrue(findings[0].message.endswith('on "Traceback"'))

    def test_warning_message_truncates_long_match(self):
        pcfg = _make_pipeline([], [r"warning.+"])
        long_line = "warning " + "x" * 150

        findings = classify_regex_counter(pcfg, Path("/tmp/test.log"), "pipe", [long_line])

        self.assertIn("...", findings[0].message)


if __name__ == "__main__":
    unittest.main()
