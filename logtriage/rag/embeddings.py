"""Embedding services with CPU and GPU optimized strategies."""

import logging
import subprocess
import json
import sys
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

class SubprocessEmbeddingService:
    """Subprocess-based embedding service for complete memory isolation."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu", batch_size: int = 4):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.subprocess_script = os.path.join(os.path.dirname(__file__), "subprocess_embeddings.py")
        
        logger.info(f"SubprocessEmbeddingService initialized: model={model_name}, device={device}, batch_size={batch_size}")
    
    def embed_texts_streaming(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using subprocess isolation with batch processing."""
        if not texts:
            logger.debug("No texts provided for embedding")
            return np.array([])
        
        try:
            memory_before = get_memory_usage()
            logger.debug(f"Generating embeddings for {len(texts)} texts (subprocess batched, memory: {memory_before:.2f}GB)")
            
            # Process in batches using subprocesses
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                try:
                    # Monitor memory before processing
                    if i % (self.batch_size * 2) == 0:
                        current_memory = get_memory_usage()
                        logger.debug(f"Processing batch {i//self.batch_size + 1} (memory: {current_memory:.2f}GB)")
                    
                    # Run batch embedding in subprocess
                    result = subprocess.run([
                        sys.executable, self.subprocess_script, 
                        json.dumps(batch_texts), self.model_name, self.device, str(self.batch_size)
                    ], capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        try:
                            response = json.loads(result.stdout)
                            if response.get("success") and "embeddings" in response:
                                batch_embeddings = [np.array(emb) for emb in response["embeddings"]]
                                all_embeddings.extend(batch_embeddings)
                            else:
                                logger.error(f"Subprocess embedding failed: {response.get('error', 'Unknown error')}")
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse subprocess response: {e}")
                    else:
                        logger.error(f"Subprocess failed with return code {result.returncode}: {result.stderr}")
                    
                    # Force cleanup after each subprocess
                    force_cleanup()
                    
                except subprocess.TimeoutExpired:
                    logger.error(f"Subprocess embedding timed out for batch {i//self.batch_size + 1}")
                    continue
                except Exception as e:
                    logger.error(f"Failed to process batch {i//self.batch_size + 1}: {e}")
                    continue
            
            if all_embeddings:
                result = np.array(all_embeddings)
                logger.debug(f"Generated embeddings for {len(result)} texts")
                
                # Final cleanup
                del all_embeddings
                force_cleanup()
                
                memory_after = get_memory_usage()
                logger.debug(f"Completed batched subprocess embedding (memory: {memory_after:.2f}GB)")
                
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
    
    def _load_model(self):
        """No-op for subprocess service - model is loaded per subprocess."""
        pass
    
    def _unload_model(self):
        """No-op for subprocess service - model is unloaded when subprocess exits."""
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

class HybridEmbeddingService:
    """Hybrid embedding service balancing memory and performance (4-8GB usage)."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: str = "cpu", batch_size: int = 8, memory_limit_gb: float = 6.0):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.memory_limit_gb = memory_limit_gb
        self.model: Optional = None
        self._model_loaded = False
        self.chunks_since_reload = 0
        self.reload_threshold = 50  # Reload model every 50 chunks
        
        logger.info(f"HybridEmbeddingService initialized: model={model_name}, batch_size={batch_size}, memory_limit={memory_limit_gb}GB")
        
    def _load_model(self):
        """Load the embedding model on demand."""
        if not self._model_loaded:
            memory_before = get_memory_usage()
            logger.info(f"Loading hybrid embedding model (memory: {memory_before:.2f}GB)")
            
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self._model_loaded = True
                
                memory_after = get_memory_usage()
                logger.info(f"Hybrid model loaded (memory: {memory_after:.2f}GB, delta: {memory_after - memory_before:.2f}GB)")
                
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self._unload_model()
                raise
    
    def _unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            logger.info("Unloading hybrid embedding model")
            del self.model
            self.model = None
            self._model_loaded = False
            self.chunks_since_reload = 0
            force_cleanup()
    
    def _should_reload_model(self):
        """Check if model should be reloaded based on memory usage or chunk count."""
        current_memory = get_memory_usage()
        
        # Reload if memory exceeds limit
        if current_memory > self.memory_limit_gb:
            logger.warning(f"Memory {current_memory:.2f}GB exceeds limit {self.memory_limit_gb}GB, reloading model")
            return True
        
        # Reload after threshold chunks
        if self.chunks_since_reload >= self.reload_threshold:
            logger.info(f"Reloading model after {self.chunks_since_reload} chunks")
            return True
        
        return False
    
    def embed_texts_balanced(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings with balanced memory management."""
        if not texts:
            logger.debug("No texts provided for embedding")
            return np.array([])
        
        try:
            memory_before = get_memory_usage()
            logger.debug(f"Generating embeddings for {len(texts)} texts (hybrid, memory: {memory_before:.2f}GB)")
            
            # Process in small batches with periodic model reloading
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Check if we need to reload model
                if self._should_reload_model():
                    self._unload_model()
                    force_cleanup()
                
                # Load model if needed
                if not self._model_loaded:
                    memory_before = get_memory_usage()
                    logger.info(f"Loading hybrid embedding model (memory: {memory_before:.2f}GB)")
                    
                    try:
                        from sentence_transformers import SentenceTransformer
                        self.model = SentenceTransformer(self.model_name, device=self.device)
                        self._model_loaded = True
                        
                        memory_after = get_memory_usage()
                        logger.info(f"Hybrid model loaded (memory: {memory_after:.2f}GB, delta: {memory_after - memory_before:.2f}GB)")
                        
                    except Exception as e:
                        logger.error(f"Failed to load embedding model: {e}")
                        self._unload_model()
                        raise
                
                try:
                    # Process batch
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    
                    if batch_embeddings.size > 0:
                        all_embeddings.append(batch_embeddings)
                        self.chunks_since_reload += len(batch_texts)
                    
                    # Moderate cleanup after each batch
                    del batch_embeddings
                    gc.collect()
                    
                    # Monitor memory
                    if i % (self.batch_size * 5) == 0:  # Every 5 batches
                        current_memory = get_memory_usage()
                        logger.debug(f"Processed {i+len(batch_texts)}/{len(texts)} texts (memory: {current_memory:.2f}GB)")
                
                except Exception as e:
                    logger.error(f"Failed to encode batch {i//self.batch_size}: {e}")
                    continue
            
            if all_embeddings:
                result = np.vstack(all_embeddings)
                logger.debug(f"Generated embeddings for {len(result)} texts")
                
                # Final cleanup
                del all_embeddings
                force_cleanup()
                
                memory_after = get_memory_usage()
                logger.debug(f"Completed hybrid embedding (memory: {memory_after:.2f}GB)")
                
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
            result = self.embed_texts_balanced([text])
            return result[0] if result.size > 0 else np.array([])
        except Exception as e:
            logger.error(f"Failed to generate embedding for single text: {e}")
            return np.array([])

class EmbeddingService:
    """Factory class that creates appropriate embedding service based on device."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: str = "cpu", batch_size: int = 16, use_subprocess: bool = True, 
                 memory_limit_gb: float = 6.0):  # Default to subprocess for memory safety
        self.device = device.lower()
        self.model_name = model_name
        self.use_subprocess = use_subprocess
        self.memory_limit_gb = memory_limit_gb
        
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
        
        # Create appropriate service - default to subprocess for memory safety
        if self.use_subprocess or memory_limit_gb < 10.0:
            # Use subprocess with appropriate batch size
            subprocess_batch_size = min(batch_size, 8)  # Limit batch size for subprocess efficiency
            self.service = SubprocessEmbeddingService(model_name, self.device, subprocess_batch_size)
            logger.info(f"Created subprocess embedding service (batch_size={subprocess_batch_size}) for memory safety")
        elif memory_limit_gb > 15.0:
            # High memory limit - use standard GPU/CPU service
            if self.device == "cuda":
                self.service = GPUEmbeddingService(model_name, batch_size)
            else:
                self.service = CPUEmbeddingService(model_name)
            logger.info("Created standard embedding service for high memory configuration")
        else:
            # Medium memory limit - still use subprocess to be safe
            subprocess_batch_size = min(batch_size, 6)
            self.service = SubprocessEmbeddingService(model_name, self.device, subprocess_batch_size)
            logger.info(f"Created subprocess embedding service (batch_size={subprocess_batch_size}) for medium memory configuration")
        
        logger.info(f"Created {self.device}-optimized embedding service (subprocess={self.use_subprocess}, memory_limit={memory_limit_gb}GB)")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using the appropriate strategy."""
        if isinstance(self.service, SubprocessEmbeddingService):
            return self.service.embed_texts_streaming(texts)
        elif isinstance(self.service, HybridEmbeddingService):
            return self.service.embed_texts_balanced(texts)
        elif isinstance(self.service, CPUEmbeddingService):
            return self.service.embed_texts_streaming(texts)
        else:
            return self.service.embed_texts_batched(texts)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.service.embed_single(text)
    
    def _load_model(self):
        """Load model on the underlying service."""
        if hasattr(self.service, '_load_model'):
            self.service._load_model()
    
    def _unload_model(self):
        """Unload model on the underlying service."""
        if hasattr(self.service, '_unload_model'):
            self.service._unload_model()
