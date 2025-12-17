"""Embedding service for generating text embeddings."""

import logging
from typing import List, Optional
import numpy as np
import gc

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Provides embedding generation."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: str = "cpu", batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model: Optional = None
        self._model_loaded = False
        
        logger.info(f"EmbeddingService initialized with model: {model_name}")
        
    def _load_model(self):
        """Load the embedding model on demand."""
        if not self._model_loaded:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self._model_loaded = True
                logger.info("Embedding model loaded successfully")
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self._unload_model()
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
        """Generate embeddings."""
        if not texts:
            logger.debug("No texts provided for embedding")
            return np.array([])
        
        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            self._load_model()
            
            # Process texts in batches
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                try:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    all_embeddings.append(batch_embeddings)
                    
                except Exception as e:
                    logger.error(f"Failed to encode batch {i//self.batch_size}: {e}")
                    continue
            
            if all_embeddings:
                result = np.vstack(all_embeddings)
                logger.debug(f"Generated embeddings for {len(result)} texts")
                return result
            else:
                logger.warning("No embeddings were generated")
                return np.array([])
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return np.array([])
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        try:
            result = self.embed_texts([text])
            return result[0] if result.size > 0 else np.array([])
        except Exception as e:
            logger.error(f"Failed to generate embedding for single text: {e}")
            return np.array([])
