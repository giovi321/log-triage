"""Embedding services with CPU and GPU optimized strategies."""

import logging
from typing import List, Optional
import numpy as np
import gc
import os
import psutil

logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in GB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3
    except:
        return 0.0

def force_cleanup():
    """Force aggressive memory cleanup."""
    # Multiple garbage collection passes
    for _ in range(3):
        gc.collect()
    
    # Clear any cached data
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

class CPUEmbeddingService:
    """CPU-optimized embedding service for ultra memory-efficient processing."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.device = "cpu"
        self.model: Optional = None
        self._model_loaded = False
        
        logger.info(f"CPUEmbeddingService initialized with model: {model_name}")
        
    def _load_model(self):
        """Load the embedding model on demand with memory monitoring."""
        if not self._model_loaded:
            memory_before = get_memory_usage()
            logger.info(f"Loading CPU embedding model: {self.model_name} (memory: {memory_before:.2f}GB)")
            
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self._model_loaded = True
                
                memory_after = get_memory_usage()
                logger.info(f"CPU embedding model loaded (memory: {memory_after:.2f}GB, delta: {memory_after - memory_before:.2f}GB)")
                
                # Force cleanup after loading
                force_cleanup()
                
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
            force_cleanup()
    
    def embed_texts_streaming(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings one at a time for minimal memory usage."""
        if not texts:
            logger.debug("No texts provided for embedding")
            return np.array([])
        
        try:
            memory_before = get_memory_usage()
            logger.debug(f"Generating embeddings for {len(texts)} texts (CPU streaming, memory: {memory_before:.2f}GB)")
            
            self._load_model()
            
            # Process one text at a time for minimal memory usage
            all_embeddings = []
            
            for i, text in enumerate(texts):
                try:
                    # Monitor memory before processing
                    if i % 5 == 0:
                        current_memory = get_memory_usage()
                        logger.debug(f"Processing text {i+1}/{len(texts)} (memory: {current_memory:.2f}GB)")
                        
                        # Force cleanup if memory is growing too much
                        if current_memory > memory_before + 1.0:  # 1GB increase threshold
                            logger.warning(f"Memory usage high ({current_memory:.2f}GB), forcing cleanup")
                            force_cleanup()
                    
                    # Process single text
                    embedding = self.model.encode(
                        [text],
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    
                    if embedding.size > 0:
                        all_embeddings.append(embedding[0])
                    
                    # Immediate cleanup after each text
                    del embedding
                    force_cleanup()
                    
                except Exception as e:
                    logger.error(f"Failed to encode text {i+1}: {e}")
                    continue
            
            if all_embeddings:
                result = np.array(all_embeddings)
                logger.debug(f"Generated embeddings for {len(result)} texts")
                
                # Final cleanup
                del all_embeddings
                force_cleanup()
                
                memory_after = get_memory_usage()
                logger.debug(f"Completed embedding generation (memory: {memory_after:.2f}GB)")
                
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
            result = self.embed_texts_streaming([text])
            return result[0] if result.size > 0 else np.array([])
        except Exception as e:
            logger.error(f"Failed to generate embedding for single text: {e}")
            return np.array([])

class GPUEmbeddingService:
    """GPU-optimized embedding service for batch processing."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 16):  # Reduced batch size
        self.model_name = model_name
        self.device = "cuda"
        self.batch_size = batch_size
        self.model: Optional = None
        self._model_loaded = False
        
        logger.info(f"GPUEmbeddingService initialized with model: {model_name}, batch_size: {batch_size}")
        
    def _load_model(self):
        """Load the embedding model on demand with memory monitoring."""
        if not self._model_loaded:
            memory_before = get_memory_usage()
            logger.info(f"Loading GPU embedding model: {self.model_name} (memory: {memory_before:.2f}GB)")
            
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self._model_loaded = True
                
                memory_after = get_memory_usage()
                logger.info(f"GPU embedding model loaded (memory: {memory_after:.2f}GB, delta: {memory_after - memory_before:.2f}GB)")
                
                # Clear CUDA cache after loading
                import torch
                torch.cuda.empty_cache()
                
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
            
            # Clear CUDA cache
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
    
    def embed_texts_batched(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings in optimized batches for GPU with memory monitoring."""
        if not texts:
            logger.debug("No texts provided for embedding")
            return np.array([])
        
        try:
            memory_before = get_memory_usage()
            logger.debug(f"Generating embeddings for {len(texts)} texts (GPU batched, memory: {memory_before:.2f}GB)")
            
            self._load_model()
            
            # Process texts in smaller batches for memory efficiency
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                try:
                    # Monitor memory before batch
                    batch_memory = get_memory_usage()
                    logger.debug(f"Processing batch {i//self.batch_size + 1} (memory: {batch_memory:.2f}GB)")
                    
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    all_embeddings.append(batch_embeddings)
                    
                    # Clear CUDA cache between batches
                    import torch
                    torch.cuda.empty_cache()
                    
                    # Force cleanup
                    force_cleanup()
                    
                except Exception as e:
                    logger.error(f"Failed to encode batch {i//self.batch_size}: {e}")
                    continue
            
            if all_embeddings:
                result = np.vstack(all_embeddings)
                logger.debug(f"Generated embeddings for {len(result)} texts")
                
                # Final cleanup
                del all_embeddings
                import torch
                torch.cuda.empty_cache()
                force_cleanup()
                
                memory_after = get_memory_usage()
                logger.debug(f"Completed embedding generation (memory: {memory_after:.2f}GB)")
                
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
            result = self.embed_texts_batched([text])
            return result[0] if result.size > 0 else np.array([])
        except Exception as e:
            logger.error(f"Failed to generate embedding for single text: {e}")
            return np.array([])

class EmbeddingService:
    """Factory class that creates appropriate embedding service based on device."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: str = "cpu", batch_size: int = 16):  # Reduced default batch size
        self.device = device.lower()
        self.model_name = model_name
        
        if self.device == "cuda":
            # Check if CUDA is actually available
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    self.device = "cpu"
            except ImportError:
                logger.warning("PyTorch not available, falling back to CPU")
                self.device = "cpu"
        
        # Create appropriate service
        if self.device == "cuda":
            self.service = GPUEmbeddingService(model_name, batch_size)
        else:
            self.service = CPUEmbeddingService(model_name)
        
        logger.info(f"Created {self.device}-optimized embedding service")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using the appropriate strategy."""
        if isinstance(self.service, CPUEmbeddingService):
            return self.service.embed_texts_streaming(texts)
        else:
            return self.service.embed_texts_batched(texts)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.service.embed_single(text)
