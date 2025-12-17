"""Embedding service for document chunks."""

import logging
import os
from typing import List, Optional
import numpy as np
import gc

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Provides embedding generation for document chunks."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: str = "cpu", batch_size: int = 8):  # Reduced batch size
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model: Optional = None
        self._model_loaded = False
        
    def _load_model(self):
        """Load the embedding model on demand with memory management."""
        if not self._model_loaded:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self._model_loaded = True
                logger.info("Embedding model loaded successfully")
                
                # Force cleanup after model loading
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
    def _unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            logger.info("Unloading embedding model to free memory")
            del self.model
            self.model = None
            self._model_loaded = False
            gc.collect()
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts with aggressive memory management."""
        if not texts:
            logger.debug("No texts provided for embedding")
            return np.array([])
        
        # Very aggressive memory limit
        max_texts = 100  # Reduced from 1000
        if len(texts) > max_texts:
            logger.warning(f"Too many texts ({len(texts)}), limiting to {max_texts}")
            texts = texts[:max_texts]
        
        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            self._load_model()
            
            # Process one batch at a time with immediate cleanup
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                logger.debug(f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
                
                try:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True  # Normalize to reduce memory
                    )
                    all_embeddings.append(batch_embeddings)
                    
                    # Aggressive cleanup after each batch
                    del batch_embeddings
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Failed to encode batch: {e}")
                    # Unload model on error
                    self._unload_model()
                    return np.array([])
            
            if all_embeddings:
                result = np.vstack(all_embeddings)
                logger.debug(f"Generated embeddings: {result.shape}")
                
                # Cleanup intermediate arrays
                del all_embeddings
                gc.collect()
                
                # Unload model to free memory between operations
                self._unload_model()
                
                return result
            else:
                logger.warning("No embeddings generated")
                return np.array([])
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            self._unload_model()
            return np.array([])
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        try:
            result = self.embed_texts([text])
            return result[0] if result.size > 0 else np.array([])
        except Exception as e:
            logger.error(f"Failed to generate embedding for single text: {e}")
            return np.array([])
