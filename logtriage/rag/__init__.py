"""RAG (Retrieval-Augmented Generation) package for log-triage.

The heavy submodules (GitPython, sentence-transformers, FAISS, …) are imported
lazily via PEP 562 so that importing a light member such as the service monitor
— or merely importing a module that lives in this package — does not require the
full RAG dependency stack to be installed.
"""
from typing import TYPE_CHECKING

__all__ = [
    "KnowledgeManager",
    "DocumentProcessor",
    "EmbeddingService",
    "VectorStore",
    "RetrievalEngine",
    "RAGClient",
]

_LAZY = {
    "KnowledgeManager": ".knowledge_manager",
    "DocumentProcessor": ".document_processor",
    "EmbeddingService": ".embeddings",
    "VectorStore": ".vector_store",
    "RetrievalEngine": ".retrieval",
    "RAGClient": ".rag_client",
}


def __getattr__(name):  # PEP 562
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target, __name__)
    return getattr(module, name)


if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .knowledge_manager import KnowledgeManager
    from .document_processor import DocumentProcessor
    from .embeddings import EmbeddingService
    from .vector_store import VectorStore
    from .retrieval import RetrievalEngine
    from .rag_client import RAGClient
