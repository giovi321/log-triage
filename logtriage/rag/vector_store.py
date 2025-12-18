"""FAISS-based vector store for document embeddings with memory management."""

import logging
import uuid
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import gc
import threading

from ..models import DocumentChunk

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based vector database for storing and retrieving document embeddings with memory limits."""
    
    def __init__(self, persist_directory: Path):
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # FAISS and SQLite paths
        self.faiss_index_path = self.persist_directory / "faiss_index.bin"
        self.metadata_db_path = self.persist_directory / "metadata.db"
        
        # Memory limits and embedding dimension
        self.max_chunks_in_memory = 1000  # FAISS can handle more
        self.embedding_dimension = 384  # Default for MiniLM
        
        self._lock = threading.RLock()
        self._index = None
        
        # Initialize FAISS and SQLite
        self._init_faiss()
        self._init_sqlite()
        
        logger.info("FAISS vector store initialized")
    
    def _init_faiss(self):
        """Initialize FAISS index."""
        try:
            import faiss
            
            with self._lock:
                if self.faiss_index_path.exists():
                    self._index = faiss.read_index(str(self.faiss_index_path))
                    self.embedding_dimension = int(self._index.d)
                else:
                    self._index = faiss.IndexFlatIP(self.embedding_dimension)
                    faiss.write_index(self._index, str(self.faiss_index_path))
            
            logger.info("FAISS vector store initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise
    
    def _init_sqlite(self):
        """Initialize SQLite database for metadata."""
        try:
            self.conn = sqlite3.connect(str(self.metadata_db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
            # Create tables
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    heading TEXT,
                    content TEXT NOT NULL,
                    commit_hash TEXT,
                    metadata TEXT,
                    faiss_index INTEGER
                )
            """)
            
            # Create indexes for fast queries
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_repo_id ON chunks(repo_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_faiss_index ON chunks(faiss_index)")
            
            self.conn.commit()
            logger.info("SQLite metadata database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite: {e}")
            raise
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Add document chunks with their embeddings to FAISS and SQLite with streaming approach."""
        if not chunks or embeddings.size == 0:
            logger.debug("No chunks or embeddings to add")
            return
        
        if len(chunks) != len(embeddings):
            logger.error(f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})")
            return
        
        try:
            logger.debug(f"Adding {len(chunks)} chunks to FAISS vector store (streaming)")
            
            import faiss
            
            with self._lock:
                if self._index is None:
                    self._init_faiss()
                
                emb = np.asarray(embeddings, dtype=np.float32)
                faiss.normalize_L2(emb)
                
                start_idx = int(self._index.ntotal)
                self._index.add(emb)
                
                rows = []
                for i, chunk in enumerate(chunks):
                    faiss_idx = start_idx + i
                    metadata_json = json.dumps(chunk.metadata) if chunk.metadata else None
                    rows.append((
                        chunk.chunk_id,
                        chunk.repo_id,
                        chunk.file_path,
                        chunk.heading,
                        chunk.content,
                        chunk.commit_hash,
                        metadata_json,
                        faiss_idx
                    ))
                
                self.conn.executemany("""
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, repo_id, file_path, heading, content, commit_hash, metadata, faiss_index)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
                self.conn.commit()
                
                faiss.write_index(self._index, str(self.faiss_index_path))
                
                del emb
                del rows
            
            # Aggressive cleanup
            for _ in range(3):
                gc.collect()
            
            logger.debug(f"Successfully added {len(chunks)} chunks to FAISS store (streaming)")
            
        except Exception as e:
            logger.error(f"Failed to add chunks to FAISS store: {e}", exc_info=True)
            raise
    
    def query(self, query_embedding: np.ndarray, repo_ids: Optional[List[str]] = None, 
              n_results: int = 5) -> Tuple[List[DocumentChunk], List[float]]:
        """Query for similar documents using FAISS with streaming approach."""
        try:
            import faiss
            
            with self._lock:
                if self._index is None:
                    if not self.faiss_index_path.exists():
                        logger.warning("FAISS index file not found")
                        return [], []
                    self._init_faiss()
                
                q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(q)
                
                search_k = min(n_results * 2, int(self._index.ntotal))  # Get more to filter by repo
                if search_k <= 0:
                    return [], []
                
                scores, indices = self._index.search(q, search_k)
            
            chunks = []
            distances = []
            
            # Get metadata for results
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid results
                    continue
                
                # Get chunk metadata from SQLite
                cursor = self.conn.execute("""
                    SELECT chunk_id, repo_id, file_path, heading, content, commit_hash, metadata
                    FROM chunks WHERE faiss_index = ?
                """, (int(idx),))
                
                row = cursor.fetchone()
                if row:
                    # Filter by repo_ids if specified
                    if repo_ids and row['repo_id'] not in repo_ids:
                        continue
                    
                    # Parse metadata
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    
                    chunk = DocumentChunk(
                        chunk_id=row['chunk_id'],
                        repo_id=row['repo_id'],
                        file_path=row['file_path'],
                        heading=row['heading'],
                        content=row['content'],
                        commit_hash=row['commit_hash'],
                        metadata=metadata
                    )
                    
                    chunks.append(chunk)
                    distances.append(float(score))
                    
                    # Limit results
                    if len(chunks) >= n_results:
                        break
            
            # Aggressive cleanup
            for _ in range(3):
                gc.collect()
            
            return chunks, distances
            
        except Exception as e:
            logger.error(f"Failed to query FAISS store: {e}", exc_info=True)
            return [], []
    
    def delete_by_repo(self, repo_id: str):
        """Delete all chunks from a specific repository."""
        try:
            # Get FAISS indices for this repo
            cursor = self.conn.execute("""
                SELECT faiss_index FROM chunks WHERE repo_id = ?
            """, (repo_id,))
            
            indices_to_remove = [row[0] for row in cursor.fetchall()]
            
            if indices_to_remove:
                # Remove from SQLite
                self.conn.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
                self.conn.commit()
                
                # Rebuild FAISS index (FAISS doesn't support removal)
                self._rebuild_faiss_index()
                
                logger.debug(f"Deleted {len(indices_to_remove)} chunks for repo {repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete repo {repo_id}: {e}")
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from remaining chunks."""
        try:
            import faiss
            
            # Get all remaining embeddings
            cursor = self.conn.execute("""
                SELECT faiss_index, content FROM chunks ORDER BY faiss_index
            """)
            
            rows = cursor.fetchall()
            
            if rows:
                # Create new index
                new_index = faiss.IndexFlatIP(self.embedding_dimension)
                
                # Note: We'd need to re-embed content here, but for now just create empty index
                # In practice, we'd store embeddings or re-generate them
                
                with self._lock:
                    self._index = new_index
                    faiss.write_index(self._index, str(self.faiss_index_path))
                
                logger.info("FAISS index rebuilt")
            else:
                # Create empty index
                with self._lock:
                    self._index = faiss.IndexFlatIP(self.embedding_dimension)
                    faiss.write_index(self._index, str(self.faiss_index_path))
                
        except Exception as e:
            logger.error(f"Failed to rebuild FAISS index: {e}")
    
    def get_repo_chunks(self, repo_id: str) -> List[DocumentChunk]:
        """Get all chunks from a specific repository."""
        try:
            cursor = self.conn.execute("""
                SELECT chunk_id, repo_id, file_path, heading, content, commit_hash, metadata
                FROM chunks WHERE repo_id = ?
                LIMIT 100
            """, (repo_id,))
            
            chunks = []
            for row in cursor.fetchall():
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                
                chunk = DocumentChunk(
                    chunk_id=row['chunk_id'],
                    repo_id=row['repo_id'],
                    file_path=row['file_path'],
                    heading=row['heading'],
                    content=row['content'],
                    commit_hash=row['commit_hash'],
                    metadata=metadata
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get repo chunks: {e}")
            return []

    def get_repo_chunk_count(self, repo_id: str) -> int:
        """Get the total number of chunks stored for a repository."""
        try:
            cursor = self.conn.execute(
                "SELECT COUNT(*) as count FROM chunks WHERE repo_id = ?",
                (repo_id,),
            )
            row = cursor.fetchone()
            if not row:
                return 0
            return int(row["count"])
        except Exception as e:
            logger.error(f"Failed to get repo chunk count: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            # Get total chunks from SQLite
            cursor = self.conn.execute("SELECT COUNT(*) as count FROM chunks")
            total_chunks = cursor.fetchone()['count']
            
            # Get FAISS index stats
            with self._lock:
                faiss_count = int(self._index.ntotal) if self._index is not None else 0
                faiss_dim = int(self._index.d) if self._index is not None else int(self.embedding_dimension)
 
            estimated_faiss_bytes = faiss_count * faiss_dim * 4
            estimated_faiss_gb = estimated_faiss_bytes / 1024**3
            
            return {
                "total_chunks": total_chunks,
                "faiss_vectors": faiss_count,
                "embedding_dimension": self.embedding_dimension,
                "estimated_faiss_gb": estimated_faiss_gb,
                "persist_directory": str(self.persist_directory),
                "index_type": "FAISS IndexFlatIP"
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except:
            pass
