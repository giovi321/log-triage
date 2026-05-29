"""log-triage package.

``main`` is exposed lazily so that ``import logtriage`` (and importing light
submodules such as ``logtriage.fingerprint``) does not pull in the CLI and its
optional heavy dependencies (GitPython, embeddings, FAISS) unless actually used.
"""
from .version import __version__

__all__ = ["main", "__version__"]


def __getattr__(name):  # PEP 562
    if name == "main":
        from .cli import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
