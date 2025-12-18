"""Retrieval engine for finding relevant documentation."""

import logging
import re
from typing import List, Dict, Any, Optional
import numpy as np

from ..models import Finding, DocumentChunk, RetrievalResult
from .embeddings import EmbeddingService
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

class RetrievalEngine:
    """Retrieves relevant documentation chunks based on log findings."""
    
    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore,
                 top_k: int = 5, similarity_threshold: float = 0.7, max_chunks: int = 10):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.max_chunks = max_chunks
    
    def retrieve_for_finding(self, finding: Finding, repo_ids: Optional[List[str]] = None) -> RetrievalResult:
        """Retrieve relevant documentation for a log finding."""
        try:
            logger.debug(f"Retrieving documentation for finding: {finding.message}")
            
            # Build query from finding
            query = self._build_query(finding)
            logger.debug(f"Built retrieval query: {query}")
            
            # Generate query embedding
            query_embedding = self.embedding_service.embed_single(query)
            if query_embedding.size == 0:
                logger.warning("Failed to generate query embedding")
                return RetrievalResult(chunks=[], query=query, total_retrieved=0)
            
            # Search vector store
            chunks, distances = self.vector_store.query(
                query_embedding, repo_ids=repo_ids, n_results=self.max_chunks
            )
            
            logger.debug(f"Retrieved {len(chunks)} chunks from vector store")
            
            # Filter by similarity threshold and deduplicate
            filtered_chunks = []
            seen_content = set()
            
            for chunk, distance in zip(chunks, distances):
                # Convert distance to similarity (ChromaDB uses L2 distance)
                similarity = 1 / (1 + distance)
                
                if similarity < self.similarity_threshold:
                    logger.debug(f"Chunk filtered by similarity threshold: {similarity} < {self.similarity_threshold}")
                    continue
                
                # Deduplicate by content hash
                content_hash = hash(chunk.content)
                if content_hash in seen_content:
                    logger.debug("Chunk filtered as duplicate")
                    continue
                
                seen_content.add(content_hash)
                filtered_chunks.append(chunk)
                
                if len(filtered_chunks) >= self.top_k:
                    break
            
            logger.info(f"Returning {len(filtered_chunks)} relevant chunks for finding")
            return RetrievalResult(
                chunks=filtered_chunks,
                query=query,
                total_retrieved=len(chunks)
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve documentation for finding: {e}", exc_info=True)
            return RetrievalResult(chunks=[], query="", total_retrieved=0)
    
    def _build_query(self, finding: Finding) -> str:
        """Build retrieval query from log finding."""
        query_parts = []
        
        # Add error type and message
        if finding.message:
            query_parts.append(finding.message)
        
        # Extract key information from excerpt
        excerpt_text = " ".join(finding.excerpt)
        
        # Look for exception types
        exception_patterns = [
            r'\b\w+Exception\b',
            r'\b\w+Error\b',
            r'\bTypeError\b',
            r'\bValueError\b',
            r'\bKeyError\b',
            r'\bAttributeError\b',
            r'\bImportError\b',
            r'\bModuleNotFoundError\b',
            r'\bConnectionError\b',
            r'\bTimeoutError\b'
        ]
        
        for pattern in exception_patterns:
            matches = re.findall(pattern, excerpt_text, re.IGNORECASE)
            query_parts.extend(matches)
        
        # Look for file/module names
        file_patterns = [
            r'\b\w+\.py\b',
            r'\b\w+\.js\b',
            r'\b\w+\.php\b',
            r'\bmodules/\w+\b',
            r'\bcomponents/\w+\b',
            r'\bservices/\w+\b'
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, excerpt_text)
            query_parts.extend(matches)
        
        # Look for integration/domain names
        domain_patterns = [
            r'\b\w+integration\b',
            r'\b\w+service\b',
            r'\b\w+plugin\b',
            r'\b\w+module\b',
            r'\b\w+component\b'
        ]
        
        for pattern in domain_patterns:
            matches = re.findall(pattern, excerpt_text, re.IGNORECASE)
            query_parts.extend(matches)
        
        # Look for specific error keywords
        error_keywords = [
            'connection', 'timeout', 'failed', 'denied', 'forbidden', 
            'unauthorized', 'certificate', 'ssl', 'tls', 'database',
            'authentication', 'permission', 'configuration', 'dependency'
        ]
        
        for keyword in error_keywords:
            if keyword.lower() in excerpt_text.lower():
                query_parts.append(keyword)
        
        # Add pipeline context
        if finding.pipeline_name:
            query_parts.append(finding.pipeline_name)
        
        # Combine and deduplicate
        unique_parts = list(dict.fromkeys(query_parts))  # Preserve order, remove duplicates
        
        return " ".join(unique_parts)
