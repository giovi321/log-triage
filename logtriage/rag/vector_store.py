"""Vector store for document embeddings."""

import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings

from ..models import DocumentChunk

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector database for storing and retrieving document embeddings."""
    
    def __init__(self, persist_directory: Path):
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="documentation",
            metadata={"description": "Documentation chunks for RAG"}
        )
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Add document chunks with their embeddings."""
        if not chunks or embeddings.size == 0:
            logger.debug("No chunks or embeddings to add")
            return
        
        if len(chunks) != len(embeddings):
            logger.error(f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})")
            return
        
        # Memory limit: refuse to add too many chunks at once
        max_chunks = 500  # Limit to prevent OOM
        if len(chunks) > max_chunks:
            logger.warning(f"Too many chunks ({len(chunks)}), limiting to {max_chunks}")
            chunks = chunks[:max_chunks]
            embeddings = embeddings[:max_chunks]
        
        try:
            logger.debug(f"Adding {len(chunks)} chunks to vector store")
            
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = {
                    "repo_id": chunk.repo_id,
                    "file_path": chunk.file_path,
                    "heading": chunk.heading,
                    "commit_hash": chunk.commit_hash,
                    **chunk.metadata
                }
                metadatas.append(metadata)
            
            # Add to collection in smaller batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_docs = documents[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size].tolist()
                batch_metas = metadatas[i:i+batch_size]
                
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metas
                )
                
                # Force garbage collection after each batch
                import gc
                gc.collect()
            
            logger.debug(f"Successfully added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add chunks to vector store: {e}", exc_info=True)
            raise
    
    def query(self, query_embedding: np.ndarray, repo_ids: Optional[List[str]] = None, 
              n_results: int = 10) -> Tuple[List[DocumentChunk], List[float]]:
        """Query for similar documents."""
        query_dict = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": n_results
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
    
    def delete_by_repo(self, repo_id: str):
        """Delete all chunks from a specific repository."""
        self.collection.delete(where={"repo_id": repo_id})
    
    def get_repo_chunks(self, repo_id: str) -> List[DocumentChunk]:
        """Get all chunks from a specific repository."""
        results = self.collection.get(
            where={"repo_id": repo_id},
            include=["documents", "metadatas"]
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name,
            "persist_directory": str(self.persist_directory)
        }
