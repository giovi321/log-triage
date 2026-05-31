"""Static guard against undefined-name bugs in the web layer.

The router split moved handlers between modules; a missed import or a bare
module-global reference (e.g. `llm_defaults` instead of `STATE.llm_defaults`)
becomes a NameError that only surfaces when the route runs — and several of
those routes render Jinja templates, which the Py3.14 dev box can't exercise.
This test runs pyflakes over the webui package and fails on any "undefined
name", catching that whole class of regression at unit-test time.
"""
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyflakes")

WEBUI = Path(__file__).resolve().parent.parent / "logtriage" / "webui"


def test_webui_has_no_undefined_names():
    targets = [str(p) for p in WEBUI.rglob("*.py")]
    # enrichment.py is webui-adjacent and was touched by the same work.
    targets.append(str(WEBUI.parent / "enrichment.py"))
    proc = subprocess.run(
        [sys.executable, "-m", "pyflakes", *targets],
        capture_output=True, text=True,
    )
    offending = [
        line for line in (proc.stdout or "").splitlines()
        if "undefined name" in line
    ]
    assert not offending, "pyflakes found undefined names:\n" + "\n".join(offending)
