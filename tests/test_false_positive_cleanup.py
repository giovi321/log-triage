import tempfile
import unittest
from pathlib import Path

from logtriage.models import Finding, Severity
import logtriage.webui.db as db


@unittest.skipIf(db._sqlalchemy_import_error is not None, "sqlalchemy not available")
class DeleteFindingsMatchingRegexTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(self.tmpdir.name) / "db.sqlite"
        db.setup_database(f"sqlite:///{db_path}")
        self._counter = 1

    def tearDown(self):
        self.tmpdir.cleanup()

    def _store(self, message: str, pipeline: str, module: str, excerpt=None):
        finding = Finding(
            file_path=Path("/tmp/test.log"),
            pipeline_name=pipeline,
            finding_index=self._counter,
            severity=Severity.ERROR,
            message=message,
            line_start=1,
            line_end=1,
            rule_id=None,
            excerpt=excerpt or [],
            needs_llm=False,
        )
        db.store_finding(module, finding)
        self._counter += 1

    def test_removes_all_matching_findings_for_pipeline(self):
        self._store("Noise one", "pipe_a", "mod", excerpt=["Noise one"])
        self._store("Noise two", "pipe_a", "mod", excerpt=["Noise two"])
        self._store("Noise other pipeline", "pipe_b", "mod", excerpt=["Noise other pipeline"])
        self._store("Unrelated", "pipe_a", "mod", excerpt=["Different"])

        removed = db.delete_findings_matching_regex(r"Noise", pipeline_name="pipe_a")

        self.assertEqual(removed, 2)
        remaining = db.get_recent_findings_for_module("mod", limit=10)
        remaining_messages = sorted(row.message for row in remaining)
        self.assertEqual(remaining_messages, ["Noise other pipeline", "Unrelated"])


if __name__ == "__main__":
    unittest.main()
