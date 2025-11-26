import unittest

from logtriage.webui.regex_utils import (
    _compile_regex_with_feedback,
    _lint_regex_input,
    _prepare_sample_lines,
)


class RegexLabValidationTests(unittest.TestCase):
    def test_lint_empty_pattern(self):
        issues = _lint_regex_input("")
        self.assertIn("Pattern cannot be empty.", issues)

    def test_lint_trailing_slash(self):
        issues = _lint_regex_input("abc/")
        self.assertTrue(any("Trailing slash" in issue for issue in issues))

    def test_lint_unbalanced_brackets(self):
        issues = _lint_regex_input("[abc")
        self.assertTrue(any("Unbalanced" in issue for issue in issues))

    def test_compile_error_highlights_position(self):
        pattern, error = _compile_regex_with_feedback("(abc")
        self.assertIsNone(pattern)
        self.assertIsNotNone(error)
        self.assertIn("Problem near", error)

    def test_prepare_sample_lines_truncates_preview(self):
        long_line = "x" * 300
        prepared = _prepare_sample_lines([long_line], max_preview_chars=120)
        self.assertTrue(prepared[0]["is_preview_truncated"])
        self.assertTrue(prepared[0]["preview"].endswith("…"))
        self.assertEqual(len(prepared[0]["full"]), len(long_line))

    def test_prepare_sample_lines_clips_full_text(self):
        very_long_line = "y" * 2500
        prepared = _prepare_sample_lines([very_long_line], max_full_chars=1000)
        self.assertTrue(prepared[0]["was_clipped"])
        self.assertEqual(len(prepared[0]["full"]), 1000)


if __name__ == "__main__":
    unittest.main()
