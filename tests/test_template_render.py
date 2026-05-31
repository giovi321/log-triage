"""Template smoke test: render every Web UI page through the REAL Jinja env.

Why this exists: the dev box runs Python 3.14, where the FastAPI TestClient
can't drive template routes, so our route tests never executed Jinja. Two
production 500s slipped through that gap — both in-template type errors that only
surface when a template actually renders:

  * ``url_for('issues') + '?...'`` → ``URL + str`` TypeError (issues.html)
  * (and the kind of mistake that would recur without coverage here)

The crucial fidelity point: production ``url_for`` returns a Starlette ``URL``
object, not a string. The old ``_preview_render.py`` stub returned a string, so
its render "passed" while production failed. This test uses a ``url_for`` that
returns a real ``URL`` and renders through the application's own
``templates.env`` (real filters/globals), so a URL/str mix-up fails the test.

Page contexts are imported from ``_preview_render.build_pages()`` so the test
data stays in lockstep with the preview scaffolding and the real routes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# starlette is part of the [webui] extra; skip cleanly if it's absent.
pytest.importorskip("starlette")
from starlette.datastructures import URL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _faithful_url_for(name, **kw):
    """Mimic Starlette's url_for: returns a URL object, NOT a string.

    This is the whole point — concatenating this with a str (``url_for(...) +
    '?x'``) raises TypeError exactly as it does in production, so the test
    catches it instead of a user.
    """
    if name == "assets":
        return URL(f"/assets/{kw.get('path', '')}")
    if kw:
        suffix = "/".join(str(v) for v in kw.values())
        return URL(f"/{name}/{suffix}")
    return URL(f"/{name}")


class _App:
    def url_path_for(self, name, **kw):
        # Starlette returns a str here (not a URL); base.html relies on that.
        return "/" + name

    routes: list = []


class _Session(dict):
    def get(self, k, d=None):
        return super().get(k, d if d is not None else "")


class _Request:
    def __init__(self, path="/"):
        self.url = URL(path)
        self.app = _App()
        self.session = _Session()


def _load_pages():
    import _preview_render as preview
    return preview.build_pages()


def _page_ids():
    try:
        return [out for _tpl, out, _ctx in _load_pages()]
    except Exception:  # pragma: no cover - import-time issues surface in the test
        return []


@pytest.fixture(scope="module")
def real_env():
    """The application's own Jinja environment (real filters + globals)."""
    from logtriage.webui.shared import templates
    return templates.env


@pytest.mark.parametrize("idx", range(len(_page_ids())), ids=_page_ids() or None)
def test_template_renders_with_real_url_for(real_env, idx):
    template_name, _out, ctx = _load_pages()[idx]
    ctx = dict(ctx)
    ctx.setdefault("url_for", _faithful_url_for)
    ctx.setdefault("request", _Request(ctx.get("_path", "/")))
    # Should not raise. A URL/str concat, a missing attribute, a bad filter call,
    # etc. all raise here — which is exactly what we want to catch pre-deploy.
    html = real_env.get_template(template_name).render(**ctx)
    assert html.strip(), f"{template_name} rendered empty"
    assert "<" in html, f"{template_name} produced no markup"


def test_every_page_template_is_covered():
    """Guard: every .html page template (except partials) has a render context.

    If a new page template is added without a preview/test context, this fails —
    forcing the smoke test to keep full coverage. base.html is a layout (covered
    transitively via extends); ai_logs.html has no preview context yet and is
    listed as a known gap so this guard stays honest.
    """
    from logtriage.webui.shared import templates
    all_templates = {
        n for n in templates.env.list_templates() if n.endswith(".html")
    }
    covered = {out for _tpl, out, _ctx in _load_pages()}
    # Layouts/partials rendered only via {% extends %}/{% include %}.
    layouts = {"base.html"}
    # Known gaps: pages without a preview context yet (tracked, not silent).
    known_gaps = {"ai_logs.html"}
    missing = all_templates - covered - layouts - known_gaps
    assert not missing, (
        f"Page templates with no render context (add one to _preview_render.py): {missing}"
    )
