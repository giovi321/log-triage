"""Ultra-memory-efficient embedding service with hard limits."""

import logging
import os
import psutil
import signal
from typing import List, Optional
import numpy as np
import gc

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Provides embedding generation with aggressive memory limits."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: str = "cpu", batch_size: int = 2):  # Ultra-small batch
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model: Optional = None
        self._model_loaded = False
        
        # Hard memory limits
        self.max_memory_gb = 3.0  # Kill process if exceeds this
        self.warning_memory_gb = 2.0  # Warning at this level
        
        logger.info(f"EmbeddingService initialized with max memory limit: {self.max_memory_gb}GB")
        
    def _check_memory_usage(self):
        """Check memory usage and kill process if needed."""
        try:
            process = psutil.Process()
            memory_gb = process.memory_info().rss / 1024**3
            
            if memory_gb > self.max_memory_gb:
                logger.error(f"CRITICAL: Memory usage {memory_gb:.2f}GB exceeds limit {self.max_memory_gb}GB - terminating process")
                os.kill(os.getpid(), signal.SIGTERM)
                return False
            elif memory_gb > self.warning_memory_gb:
                logger.warning(f"High memory usage: {memory_gb:.2f}GB - forcing cleanup")
                self._force_cleanup()
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to check memory usage: {e}")
            return True
    
    def _force_cleanup(self):
        """Force aggressive memory cleanup."""
        try:
            # Unload model
            if self.model is not None:
                del self.model
                self.model = None
                self._model_loaded = False
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
                
            logger.info("Forced aggressive memory cleanup")
            
        except Exception as e:
            logger.error(f"Failed to force cleanup: {e}")
        
    def _load_model(self):
        """Load the embedding model on demand with memory management."""
        if not self._model_loaded:
            # Check memory before loading model
            if not self._check_memory_usage():
                raise MemoryError("Memory limit exceeded before model loading")
            
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self._model_loaded = True
                logger.info("Embedding model loaded successfully")
                
                # Check memory after model loading
                if not self._check_memory_usage():
                    self._unload_model()
                    raise MemoryError("Memory limit exceeded after model loading")
                
                # Force cleanup after model loading
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
        """Generate embeddings with ultra-aggressive memory management."""
        if not texts:
            logger.debug("No texts provided for embedding")
            return np.array([])
        
        # Ultra-aggressive text limit
        max_texts = 10  # Reduced from 100
        if len(texts) > max_texts:
            logger.warning(f"Too many texts ({len(texts)}), limiting to {max_texts}")
            texts = texts[:max_texts]
        
        # Check memory before processing
        if not self._check_memory_usage():
            return np.array([])
        
        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            self._load_model()
            
            # Process one text at a time to minimize memory
            all_embeddings = []
            
            for i, text in enumerate(texts):
                # Check memory before each text
                if not self._check_memory_usage():
                    break
                
                try:
                    # Process single text
                    embedding = self.model.encode(
                        [text],
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    all_embeddings.append(embedding[0])
                    
                    # Force cleanup after each text
                    gc.collect()
                    
                    logger.debug(f"Processed text {i+1}/{len(texts)}")
                    
                except Exception as e:
                    logger.error(f"Failed to encode text {i+1}: {e}")
                    # Unload model on error
                    self._unload_model()
                    return np.array([])
            
            if all_embeddings:
                result = np.array(all_embeddings)
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
