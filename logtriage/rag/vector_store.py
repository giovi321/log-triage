"""Vector store for document embeddings with memory management."""

import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import gc

from ..models import DocumentChunk

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector database for storing and retrieving document embeddings with memory limits."""
    
    def __init__(self, persist_directory: Path):
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Memory limits
        self.max_chunks_in_memory = 200  # Aggressive limit
        self.current_chunks = 0
        
        # Initialize ChromaDB with memory settings
        self._init_chromadb()
    
    def _init_chromadb(self):
        """Initialize ChromaDB with memory-efficient settings."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True  # Allow reset to clear memory
                )
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="documentation",
                metadata={"description": "Documentation chunks for RAG"}
            )
            
            logger.info("ChromaDB initialized with memory-efficient settings")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Add document chunks with their embeddings - memory efficient version."""
        if not chunks or embeddings.size == 0:
            logger.debug("No chunks or embeddings to add")
            return
        
        if len(chunks) != len(embeddings):
            logger.error(f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})")
            return
        
        # Very aggressive memory limit
        max_chunks = 50  # Reduced from 500
        if len(chunks) > max_chunks:
            logger.warning(f"Too many chunks ({len(chunks)}), limiting to {max_chunks}")
            chunks = chunks[:max_chunks]
            embeddings = embeddings[:max_chunks]
        
        try:
            logger.debug(f"Adding {len(chunks)} chunks to vector store")
            
            # Process in very small batches
            batch_size = 10  # Reduced from 100
            for i in range(0, len(chunks), batch_size):
                batch_end = min(i + batch_size, len(chunks))
                batch_chunks = chunks[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                
                # Prepare data for this batch
                ids = [chunk.chunk_id for chunk in batch_chunks]
                documents = [chunk.content for chunk in batch_chunks]
                metadatas = []
                
                for chunk in batch_chunks:
                    metadata = {
                        "repo_id": chunk.repo_id,
                        "file_path": chunk.file_path,
                        "heading": chunk.heading,
                        "commit_hash": chunk.commit_hash,
                        **chunk.metadata
                    }
                    metadatas.append(metadata)
                
                # Add this batch to collection
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=batch_embeddings.tolist(),
                    metadatas=metadatas
                )
                
                # Track memory usage
                self.current_chunks += len(batch_chunks)
                
                # Aggressive cleanup after each batch
                del ids, documents, metadatas, batch_embeddings
                gc.collect()
                
                # Check if we need to reset to free memory
                if self.current_chunks > self.max_chunks_in_memory:
                    logger.warning(f"Reached memory limit ({self.current_chunks} chunks), resetting collection")
                    self._reset_collection()
                    self.current_chunks = 0
                
                logger.debug(f"Processed batch {i//batch_size + 1}: {len(batch_chunks)} chunks")
            
            logger.debug(f"Successfully added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add chunks to vector store: {e}", exc_info=True)
            raise
    
    def _reset_collection(self):
        """Reset collection to free memory."""
        try:
            logger.info("Resetting vector store collection to free memory")
            self.client.delete_collection("documentation")
            self.collection = self.client.get_or_create_collection(
                name="documentation",
                metadata={"description": "Documentation chunks for RAG"}
            )
            gc.collect()
            logger.info("Vector store collection reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
    
    def query(self, query_embedding: np.ndarray, repo_ids: Optional[List[str]] = None, 
              n_results: int = 5) -> Tuple[List[DocumentChunk], List[float]]:  # Reduced default results
        """Query for similar documents with memory management."""
        try:
            query_dict = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": min(n_results, 5)  # Limit results to save memory
            }
            
            # Add filter for specific repositories if provided
            if repo_ids:
                where_clause = {"repo_id": {"$in": repo_ids}}
                query_dict["where"] = where_clause
            
            results = self.collection.query(**query_dict)
            
            chunks = []
            distances = []
            
            if results['ids'] and results['ids'][0]:
                for i, chunk_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]
                    content = results['documents'][0][i]
                    distance = results['distances'][0][i]
                    
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        repo_id=metadata['repo_id'],
                        file_path=metadata['file_path'],
                        heading=metadata['heading'],
                        content=content,
                        commit_hash=metadata['commit_hash'],
                        metadata={k: v for k, v in metadata.items() 
                                 if k not in ['repo_id', 'file_path', 'heading', 'commit_hash']}
                    )
                    
                    chunks.append(chunk)
                    distances.append(distance)
            
            return chunks, distances
            
        except Exception as e:
            logger.error(f"Failed to query vector store: {e}")
            return [], []
    
    def delete_by_repo(self, repo_id: str):
        """Delete all chunks from a specific repository."""
        try:
            self.collection.delete(where={"repo_id": repo_id})
            logger.debug(f"Deleted chunks for repo {repo_id}")
        except Exception as e:
            logger.error(f"Failed to delete repo {repo_id}: {e}")
    
    def get_repo_chunks(self, repo_id: str) -> List[DocumentChunk]:
        """Get all chunks from a specific repository - memory limited."""
        try:
            results = self.collection.get(
                where={"repo_id": repo_id},
                include=["documents", "metadatas"],
                limit=100  # Limit to prevent memory issues
            )
            
            chunks = []
            if results['ids']:
                for i, chunk_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    content = results['documents'][i]
                    
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        repo_id=metadata['repo_id'],
                        file_path=metadata['file_path'],
                        heading=metadata['heading'],
                        content=content,
                        commit_hash=metadata['commit_hash'],
                        metadata={k: v for k, v in metadata.items() 
                                 if k not in ['repo_id', 'file_path', 'heading', 'commit_hash']}
                    )
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get repo chunks: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection.name,
                "persist_directory": str(self.persist_directory),
                "current_memory_chunks": self.current_chunks
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
