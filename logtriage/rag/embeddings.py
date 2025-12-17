"""Embedding service for document chunks."""

import logging
import os
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Provides embedding generation for document chunks."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: str = "cpu", batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model: Optional[SentenceTransformer] = None
        
    def _load_model(self):
        """Load the embedding model on demand."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            logger.debug("No texts provided for embedding")
            return np.array([])
        
        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts using model {self.model_name}")
            self._load_model()
            
            # Process in batches to avoid memory issues
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                logger.debug(f"Processing batch {i//self.batch_size + 1} with {len(batch_texts)} texts")
                
                try:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    all_embeddings.append(batch_embeddings)
                except Exception as e:
                    logger.error(f"Failed to encode batch {i//self.batch_size + 1}: {e}", exc_info=True)
                    # Return empty array to indicate failure
                    return np.array([])
            
            if all_embeddings:
                result = np.vstack(all_embeddings)
                logger.debug(f"Successfully generated embeddings with shape {result.shape}")
                return result
            else:
                logger.warning("No embeddings were generated")
                return np.array([])
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
            return np.array([])
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        try:
            result = self.embed_texts([text])
            return result[0] if result.size > 0 else np.array([])
        except Exception as e:
            logger.error(f"Failed to generate embedding for single text: {e}", exc_info=True)
            return np.array([])
