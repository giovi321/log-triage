"""Main RAG client that coordinates all components."""

import gc
import logging
import os
try:
    import psutil
except ImportError:
    psutil = None
import threading
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..models import Finding, RetrievalResult, RAGGlobalConfig, RAGModuleConfig
from ..notifications import add_notification
from .knowledge_manager import KnowledgeManager
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingService
from .vector_store import VectorStore
from .retrieval import RetrievalEngine

logger = logging.getLogger(__name__)

class RAGClient:
    """Main RAG client that coordinates all RAG components."""
    
    def __init__(self, global_config: RAGGlobalConfig):
        self.global_config = global_config
        self.module_configs: Dict[str, RAGModuleConfig] = {}
        
        # Initialize components
        self.knowledge_manager = KnowledgeManager(global_config.cache_dir)
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService(
            model_name=global_config.embedding_model,
            device=global_config.device,
            batch_size=global_config.embedding_batch_size
        )
        self.vector_store = VectorStore(global_config.vector_store_dir)
        self.retrieval_engine = RetrievalEngine(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            top_k=global_config.top_k,
            similarity_threshold=global_config.similarity_threshold,
            max_chunks=global_config.max_chunks
        )
        
        # Track initialized repositories
        self.initialized_repos = set()

        self._progress_lock = threading.Lock()
        self._indexing_progress: Dict[str, Any] = {
            "updating": False,
            "current_repo_id": None,
            "repos_total": 0,
            "repos_done": 0,
            "started_at": None,
            "updated_at": None,
            "finished_at": None,
        }
        self._repo_progress: Dict[str, Dict[str, Any]] = {}

    def get_indexing_progress(self) -> Dict[str, Any]:
        with self._progress_lock:
            return {
                "indexing": dict(self._indexing_progress),
                "repositories": {k: dict(v) for k, v in self._repo_progress.items()},
            }
    
    def add_module_config(self, module_name: str, config: RAGModuleConfig):
        """Add RAG configuration for a module."""
        try:
            logger.info(f"Adding RAG configuration for module: {module_name}")
            self.module_configs[module_name] = config
            
            # Initialize knowledge sources for this module
            if config.enabled:
                logger.info(f"Initializing {len(config.knowledge_sources)} knowledge sources for module {module_name}")
                for source in config.knowledge_sources:
                    try:
                        repo_id = self.knowledge_manager.add_knowledge_source(source)
                        self.initialized_repos.add(repo_id)
                        logger.debug(f"Successfully initialized knowledge source {repo_id} for module {module_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize knowledge source for module {module_name}: {e}", exc_info=True)
                        add_notification(
                            "error",
                            "Failed to initialize knowledge source",
                            f"Module {module_name}, repo {source.repo_url}: {e}"
                        )
            else:
                logger.debug(f"RAG disabled for module {module_name}")
                
        except Exception as e:
            logger.error(f"Failed to add RAG configuration for module {module_name}: {e}", exc_info=True)
            raise
    
    def update_knowledge_base(self):
        """Update knowledge base if needed."""
        try:
            logger.info("Updating knowledge base...")
            repos_updated = 0

            repos_to_update = []
            for repo_id in self.initialized_repos:
                try:
                    needs_reindex = self.knowledge_manager.needs_reindexing(repo_id)
                    chunk_count = self.vector_store.get_repo_chunk_count(repo_id)
                    # If we have no stored chunks, we must reindex regardless of state.
                    if needs_reindex or chunk_count == 0:
                        repos_to_update.append(repo_id)
                except Exception:
                    # Conservative default: if we can't determine status, reindex.
                    repos_to_update.append(repo_id)

            now = time.time()
            with self._progress_lock:
                self._indexing_progress["updating"] = True
                self._indexing_progress["current_repo_id"] = None
                self._indexing_progress["repos_total"] = len(repos_to_update)
                self._indexing_progress["repos_done"] = 0
                self._indexing_progress["started_at"] = now
                self._indexing_progress["updated_at"] = now
                self._indexing_progress["finished_at"] = None

            for repo_id in repos_to_update:
                with self._progress_lock:
                    self._indexing_progress["current_repo_id"] = repo_id
                    self._indexing_progress["updated_at"] = time.time()

                try:
                    logger.debug(f"Reindexing repository: {repo_id}")
                    self._reindex_repository(repo_id)
                    repos_updated += 1
                except Exception as e:
                    logger.error(f"Failed to reindex repository {repo_id}: {e}", exc_info=True)
                    with self._progress_lock:
                        prev = self._repo_progress.get(repo_id, {})
                        repo_state = dict(prev)
                        repo_state.update(
                            {
                                "state": "error",
                                "error": str(e),
                                "updated_at": time.time(),
                            }
                        )
                        self._repo_progress[repo_id] = repo_state

                    add_notification(
                        "error",
                        "Failed to reindex repository",
                        f"Repository {repo_id}: {e}",
                    )
                finally:
                    with self._progress_lock:
                        self._indexing_progress["repos_done"] = repos_updated
                        self._indexing_progress["updated_at"] = time.time()
            
            if repos_updated > 0:
                logger.info(f"Successfully updated {repos_updated} repositories in knowledge base")
                add_notification(
                    "info",
                    "Knowledge base updated",
                    f"Updated {repos_updated} repositories"
                )
            else:
                logger.debug("No repositories needed updating")

            with self._progress_lock:
                self._indexing_progress["updating"] = False
                self._indexing_progress["current_repo_id"] = None
                self._indexing_progress["updated_at"] = time.time()
                self._indexing_progress["finished_at"] = time.time()
                
        except Exception as e:
            logger.error(f"Failed to update knowledge base: {e}", exc_info=True)
            add_notification("error", "Failed to update knowledge base", str(e))

            with self._progress_lock:
                self._indexing_progress["updating"] = False
                self._indexing_progress["updated_at"] = time.time()
                self._indexing_progress["finished_at"] = time.time()
    
    def retrieve_for_finding(self, finding: Finding, module_name: str) -> Optional[RetrievalResult]:
        """Retrieve relevant documentation for a finding."""
        if not self.global_config.enabled:
            return None
        
        module_config = self.module_configs.get(module_name)
        if not module_config or not module_config.enabled:
            return None
        
        # Get repo IDs for this module
        repo_ids = []
        for source in module_config.knowledge_sources:
            repo_id = self.knowledge_manager._get_repo_id(source.repo_url, source.branch)
            repo_ids.append(repo_id)
        
        if not repo_ids:
            return None
        
        try:
            result = self.retrieval_engine.retrieve_for_finding(finding, repo_ids)
            
            if not result.chunks:
                logger.debug(f"No relevant documentation found for finding in {module_name}")
            
            return result
            
        except Exception as e:
            add_notification(
                "error",
                "RAG retrieval failed",
                f"Module {module_name}: {e}"
            )
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get RAG system status for dashboard."""
        status = {
            "enabled": self.global_config.enabled,
            "total_repositories": len(self.initialized_repos),
            "vector_store_stats": self.vector_store.get_stats(),
            "repositories": []
        }
        
        for repo_id in self.initialized_repos:
            repo_state = self.knowledge_manager.get_repo_state(repo_id)
            if repo_state:
                chunks = self.vector_store.get_repo_chunks(repo_id)
                status["repositories"].append({
                    "repo_id": repo_id,
                    "url": repo_state.url,
                    "branch": repo_state.branch,
                    "last_commit_hash": repo_state.last_commit_hash,
                    "last_indexed_hash": repo_state.last_indexed_hash,
                    "needs_reindexing": self.knowledge_manager.needs_reindexing(repo_id),
                    "chunk_count": len(chunks),
                    "local_path": str(repo_state.local_path)
                })
        
        return status
    
    def _reindex_repository(self, repo_id: str):
        """Reindex a repository with ultra-aggressive memory management."""
        
        def get_memory_usage():
            """Get current memory usage in GB."""
            try:
                if psutil is None:
                    return 0.0
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
        
        # Find which module(s) this repo belongs to
        modules_for_repo = []
        for module_name, config in self.module_configs.items():
            if not config.enabled:
                continue
            for source in config.knowledge_sources:
                source_repo_id = self.knowledge_manager._get_repo_id(source.repo_url, source.branch)
                if source_repo_id == repo_id:
                    modules_for_repo.append((module_name, source))
        
        if not modules_for_repo:
            return
        
        # Get repository state
        repo_state = self.knowledge_manager.get_repo_state(repo_id)
        if not repo_state:
            return
        
        # Delete old chunks for this repo
        self.vector_store.delete_by_repo(repo_id)
        
        total_chunks_processed = 0
        total_files_processed = 0
        
        memory_start = get_memory_usage()
        logger.info(f"Starting ultra-aggressive reindex for {repo_id} (memory: {memory_start:.2f}GB)")

        with self._progress_lock:
            prev = self._repo_progress.get(repo_id, {})
            repo_progress = dict(prev)
            repo_progress.update(
                {
                    "repo_id": repo_id,
                    "state": "indexing",
                    "file_current": 0,
                    "file_total": 0,
                    "current_file": None,
                    "chunks_processed": 0,
                    "memory_gb": memory_start,
                    "errors": repo_progress.get("errors", 0),
                    "error": None,
                    "started_at": time.time(),
                    "updated_at": time.time(),
                    "finished_at": None,
                }
            )
            self._repo_progress[repo_id] = repo_progress
        
        all_files: List[Path] = []
        seen_files = set()
        for module_name, source in modules_for_repo:
            files = self.knowledge_manager.get_repo_files(repo_id, source.include_paths)
            for file_path in files:
                key = str(file_path)
                if key in seen_files:
                    continue
                seen_files.add(key)
                all_files.append(file_path)

        with self._progress_lock:
            repo_progress = dict(self._repo_progress.get(repo_id, {}))
            repo_progress["file_total"] = len(all_files)
            repo_progress["updated_at"] = time.time()
            self._repo_progress[repo_id] = repo_progress

        for file_idx, file_path in enumerate(all_files):
            try:
                # Monitor memory before each file
                if file_idx % 3 == 0:  # Monitor more frequently
                    current_memory = get_memory_usage()
                    logger.info(f"Processing file {file_idx+1}/{len(all_files)}: {file_path.name} (memory: {current_memory:.2f}GB)")

                    with self._progress_lock:
                        repo_progress = dict(self._repo_progress.get(repo_id, {}))
                        repo_progress["file_current"] = file_idx + 1
                        repo_progress["file_total"] = len(all_files)
                        repo_progress["current_file"] = file_path.name
                        repo_progress["memory_gb"] = current_memory
                        repo_progress["updated_at"] = time.time()
                        self._repo_progress[repo_id] = repo_progress
                    
                    # Force cleanup if memory is growing too much
                    if current_memory > memory_start + 3.0:  # 3GB threshold for more aggressive cleanup
                        logger.warning(f"Memory usage high ({current_memory:.2f}GB), forcing ultra-aggressive cleanup")
                        # Ultra-aggressive cleanup
                        for _ in range(5):
                            gc.collect()
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except ImportError:
                            pass
                
                logger.debug(f"Processing file: {file_path.name}")
                try:
                    file_size_mb = file_path.stat().st_size / 1024**2
                    logger.info(f"File size: {file_path.name} ({file_size_mb:.2f}MB)")
                except Exception:
                    pass
                
                # Stream chunks from the document processor so we never retain the full list
                batch_size = 10
                chunk_batch = []
                chunk_texts = []

                for chunk in self.document_processor.process_file_iter(
                    file_path, repo_id, repo_state.last_commit_hash
                ):
                    chunk_batch.append(chunk)
                    chunk_texts.append(chunk.content)

                    if len(chunk_batch) >= batch_size:
                        embeddings = self.embedding_service.embed_texts(chunk_texts)
                        if embeddings.size > 0 and len(embeddings) == len(chunk_batch):
                            self.vector_store.add_chunks(chunk_batch, embeddings)
                            total_chunks_processed += len(chunk_batch)

                        with self._progress_lock:
                            repo_progress = dict(self._repo_progress.get(repo_id, {}))
                            repo_progress["chunks_processed"] = total_chunks_processed
                            repo_progress["updated_at"] = time.time()
                            self._repo_progress[repo_id] = repo_progress

                        del embeddings
                        chunk_batch.clear()
                        chunk_texts.clear()
                        for _ in range(2):
                            gc.collect()

                # Flush final partial batch
                if chunk_batch:
                    embeddings = self.embedding_service.embed_texts(chunk_texts)
                    if embeddings.size > 0 and len(embeddings) == len(chunk_batch):
                        self.vector_store.add_chunks(chunk_batch, embeddings)
                        total_chunks_processed += len(chunk_batch)

                    with self._progress_lock:
                        repo_progress = dict(self._repo_progress.get(repo_id, {}))
                        repo_progress["chunks_processed"] = total_chunks_processed
                        repo_progress["updated_at"] = time.time()
                        self._repo_progress[repo_id] = repo_progress

                    del embeddings
                    chunk_batch.clear()
                    chunk_texts.clear()
                    for _ in range(2):
                        gc.collect()
                
                total_files_processed += 1

                with self._progress_lock:
                    repo_progress = dict(self._repo_progress.get(repo_id, {}))
                    repo_progress["file_current"] = total_files_processed
                    repo_progress["file_total"] = len(all_files)
                    repo_progress["current_file"] = file_path.name
                    repo_progress["chunks_processed"] = total_chunks_processed
                    repo_progress["updated_at"] = time.time()
                    self._repo_progress[repo_id] = repo_progress
                
                # Ultra-aggressive cleanup after each file
                for _ in range(3):
                    gc.collect()
                
                # More frequent progress reporting
                if total_files_processed % 5 == 0:
                    current_memory = get_memory_usage()
                    logger.info(f"Progress: {total_files_processed} files, {total_chunks_processed} chunks (memory: {current_memory:.2f}GB)")
                    
                    # Force cleanup if memory is growing too much
                    if current_memory > memory_start + 3.0:
                        logger.warning(f"Memory usage high ({current_memory:.2f}GB), forcing ultra-aggressive cleanup")
                        for _ in range(5):
                            gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")

                with self._progress_lock:
                    repo_progress = dict(self._repo_progress.get(repo_id, {}))
                    repo_progress["errors"] = int(repo_progress.get("errors", 0)) + 1
                    repo_progress["error"] = str(e)
                    repo_progress["updated_at"] = time.time()
                    self._repo_progress[repo_id] = repo_progress

                # Ultra-aggressive cleanup after error
                for _ in range(5):
                    gc.collect()
                continue
        
        # Final cleanup and mark as indexed
        force_cleanup()
        self.knowledge_manager.mark_indexed(repo_id)
        
        memory_end = get_memory_usage()
        logger.info(f"Completed reindex for {repo_id}: {total_chunks_processed} chunks from {total_files_processed} files (memory: {memory_end:.2f}GB, delta: {memory_end - memory_start:.2f}GB)")

        with self._progress_lock:
            repo_progress = dict(self._repo_progress.get(repo_id, {}))
            repo_progress.update(
                {
                    "state": "done",
                    "file_current": total_files_processed,
                    "file_total": len(all_files),
                    "current_file": None,
                    "chunks_processed": total_chunks_processed,
                    "memory_gb": memory_end,
                    "updated_at": time.time(),
                    "finished_at": time.time(),
                }
            )
            self._repo_progress[repo_id] = repo_progress
        
        if total_chunks_processed > 0:
            add_notification(
                "info",
                "Repository reindexed (ultra-aggressive)",
                f"Repository {repo_id}: {total_chunks_processed} chunks indexed from {total_files_processed} files"
            )
        else:
            logger.warning(f"No chunks processed for repository {repo_id}")
