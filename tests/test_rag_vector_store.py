"""Regression test for the FAISS rebuild data-loss bug.

Before the fix, VectorStore._rebuild_faiss_index() (invoked by delete_by_repo)
created an EMPTY index, silently dropping the vectors of every *surviving*
repo. Embeddings are now persisted as BLOBs so the index can be rebuilt
faithfully. These tests need faiss + numpy, so they are skipped on the
template/dev box and run on a real RAG runtime.
"""
import pytest

np = pytest.importorskip("numpy")
faiss = pytest.importorskip("faiss")

from logtriage.models import DocumentChunk
from logtriage.rag.vector_store import VectorStore


def _chunk(chunk_id, repo_id):
    return DocumentChunk(
        chunk_id=chunk_id,
        repo_id=repo_id,
        file_path=f"{repo_id}.md",
        heading="h",
        content=f"content {chunk_id}",
        commit_hash="abc123",
        metadata={},
    )


def test_delete_repo_preserves_other_repo(tmp_path):
    store = VectorStore(tmp_path, embedding_dimension=4)
    store.add_chunks([_chunk("a1", "repoA")], np.array([[1, 0, 0, 0]], dtype=np.float32))
    store.add_chunks([_chunk("b1", "repoB")], np.array([[0, 1, 0, 0]], dtype=np.float32))

    # Both repos are queryable to start.
    chunks, _ = store.query(np.array([[0, 1, 0, 0]], dtype=np.float32), n_results=5)
    assert any(c.chunk_id == "b1" for c in chunks)

    # Deleting repoA must NOT wipe repoB's vector (the original bug).
    store.delete_by_repo("repoA")
    assert store._index.ntotal == 1

    chunks, _ = store.query(np.array([[0, 1, 0, 0]], dtype=np.float32), n_results=5)
    assert any(c.chunk_id == "b1" for c in chunks)
    assert not any(c.repo_id == "repoA" for c in chunks)


def test_rebuild_reassigns_contiguous_indices(tmp_path):
    store = VectorStore(tmp_path, embedding_dimension=4)
    store.add_chunks(
        [_chunk("a1", "repoA"), _chunk("a2", "repoA")],
        np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
    )
    store.add_chunks([_chunk("b1", "repoB")], np.array([[0, 1, 0, 0]], dtype=np.float32))

    store.delete_by_repo("repoA")

    # Surviving rows must have contiguous faiss_index values starting at 0 so
    # SQLite lookups line up with FAISS search results.
    rows = store.conn.execute(
        "SELECT faiss_index FROM chunks ORDER BY faiss_index"
    ).fetchall()
    assert [r["faiss_index"] for r in rows] == [0]
    assert store._index.ntotal == 1
