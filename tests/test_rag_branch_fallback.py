"""KnowledgeManager clones the remote's default branch when the configured
branch is absent.

The config defaults a knowledge source's branch to ``main``, but plenty of
repos still default to ``master`` (e.g. jellyfin/jellyfin.org). A hardcoded
``--branch=main`` clone then dies with "Remote branch main not found in
upstream origin". These tests drive the fallback with a stubbed ``git`` module
(GitPython isn't a test dependency).
"""
from __future__ import annotations

import importlib
import sys
import types

import pytest


def _make_fake_git():
    """A minimal stand-in for the parts of GitPython the manager touches."""
    mod = types.ModuleType("git")

    class GitCommandError(Exception):
        def __init__(self, command=None, status=None, stderr=None, stdout=None):
            self.command, self.status, self.stderr = command, status, stderr
            super().__init__(f"Cmd({command}) failed: status={status}\n  stderr: {stderr}")

    class Repo:
        # clone_from / __init__ are replaced per-test via monkeypatch.
        @classmethod
        def clone_from(cls, url, to_path, **kwargs):  # pragma: no cover - patched
            raise NotImplementedError

        def __init__(self, path):  # pragma: no cover - patched
            raise NotImplementedError

    class Git:
        def ls_remote(self, *args):  # pragma: no cover - patched
            return ""

    mod.exc = types.SimpleNamespace(GitCommandError=GitCommandError)
    mod.Repo = Repo
    mod.Git = Git
    mod.GitCommandError = GitCommandError
    return mod


def _fake_repo(branch_name, sha="a1b2c3d4e5f6a7b8"):
    commit = types.SimpleNamespace(
        hexsha=sha,
        committed_datetime=types.SimpleNamespace(isoformat=lambda: "2026-06-01T00:00:00"),
    )
    return types.SimpleNamespace(
        head=types.SimpleNamespace(commit=commit),
        active_branch=types.SimpleNamespace(name=branch_name),
    )


@pytest.fixture
def km(monkeypatch):
    """Import knowledge_manager bound to a fake ``git`` module."""
    fake = _make_fake_git()
    monkeypatch.setitem(sys.modules, "git", fake)
    sys.modules.pop("logtriage.rag.knowledge_manager", None)
    mod = importlib.import_module("logtriage.rag.knowledge_manager")
    try:
        yield mod
    finally:
        sys.modules.pop("logtriage.rag.knowledge_manager", None)


def _config(branch="main"):
    from logtriage.models import KnowledgeSourceConfig
    return KnowledgeSourceConfig(
        repo_url="https://github.com/jellyfin/jellyfin.org", branch=branch, include_paths=[]
    )


def test_clones_requested_branch_when_present(km, tmp_path):
    manager = km.KnowledgeManager(tmp_path)
    seen = {}

    def clone_from(url, to_path, **kwargs):
        seen["branch"] = kwargs.get("branch")
        return _fake_repo(kwargs.get("branch") or "master")

    monkeypatch_clone(km, clone_from)

    repo_id = manager.add_knowledge_source(_config(branch="main"))
    state = manager.get_repo_state(repo_id)
    assert seen["branch"] == "main"          # used as-is, no fallback
    assert state.branch == "main"


def test_falls_back_to_default_branch_when_missing(km, tmp_path):
    manager = km.KnowledgeManager(tmp_path)
    attempts = []

    def clone_from(url, to_path, **kwargs):
        branch = kwargs.get("branch")
        attempts.append(branch)
        if branch == "main":
            raise km.git.exc.GitCommandError(
                ["git", "clone", "--branch=main"], 128,
                stderr="fatal: Remote branch main not found in upstream origin",
            )
        return _fake_repo(branch or "master")

    monkeypatch_clone(km, clone_from)
    # ls-remote --symref HEAD resolves the real default branch.
    monkeypatch_lsremote(km, "ref: refs/heads/master\tHEAD\n0000000\tHEAD\n")

    cfg = _config(branch="main")
    repo_id = manager.add_knowledge_source(cfg)
    state = manager.get_repo_state(repo_id)

    assert attempts == ["main", "master"]                 # tried main, fell back to master
    assert state.branch == "master"                       # actual checked-out branch
    # repo_id stays keyed on the configured branch so config mapping is stable.
    assert repo_id == manager._get_repo_id(cfg.repo_url, "main")


def test_last_resort_clones_default_head_when_symref_unknown(km, tmp_path):
    manager = km.KnowledgeManager(tmp_path)
    attempts = []

    def clone_from(url, to_path, **kwargs):
        branch = kwargs.get("branch")
        attempts.append(branch)
        if branch == "main":
            raise km.git.exc.GitCommandError(
                ["git", "clone"], 128,
                stderr="fatal: Remote branch main not found in upstream origin",
            )
        return _fake_repo("trunk")   # whatever HEAD points at

    monkeypatch_clone(km, clone_from)
    monkeypatch_lsremote(km, "")     # default branch undeterminable

    repo_id = manager.add_knowledge_source(_config(branch="main"))
    assert attempts == ["main", None]                     # retried with no explicit branch
    assert manager.get_repo_state(repo_id).branch == "trunk"


def test_non_branch_git_errors_propagate(km, tmp_path):
    manager = km.KnowledgeManager(tmp_path)

    def clone_from(url, to_path, **kwargs):
        raise km.git.exc.GitCommandError(
            ["git", "clone"], 128, stderr="fatal: could not read Username: terminal prompts disabled",
        )

    monkeypatch_clone(km, clone_from)
    with pytest.raises(km.git.exc.GitCommandError):
        manager.add_knowledge_source(_config(branch="main"))


def test_missing_branch_error_detection(km):
    err = km.git.exc.GitCommandError(["git"], 128, stderr="Remote branch main NOT FOUND in upstream origin")
    assert km.KnowledgeManager._is_missing_branch_error(err) is True
    other = km.git.exc.GitCommandError(["git"], 128, stderr="Permission denied (publickey)")
    assert km.KnowledgeManager._is_missing_branch_error(other) is False


# ---- helpers --------------------------------------------------------------

def monkeypatch_clone(km, fn):
    km.git.Repo.clone_from = staticmethod(fn)


def monkeypatch_lsremote(km, output):
    km.git.Git.ls_remote = lambda self, *args: output
