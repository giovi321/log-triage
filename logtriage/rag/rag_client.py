"""Main RAG client that coordinates all components."""

import logging
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
            
            for repo_id in self.initialized_repos:
                if self.knowledge_manager.needs_reindexing(repo_id):
                    try:
                        logger.debug(f"Reindexing repository: {repo_id}")
                        self._reindex_repository(repo_id)
                        repos_updated += 1
                    except Exception as e:
                        logger.error(f"Failed to reindex repository {repo_id}: {e}", exc_info=True)
                        add_notification(
                            "error",
                            "Failed to reindex repository",
                            f"Repository {repo_id}: {e}"
                        )
            
            if repos_updated > 0:
                logger.info(f"Successfully updated {repos_updated} repositories in knowledge base")
                add_notification(
                    "info",
                    "Knowledge base updated",
                    f"Updated {repos_updated} repositories"
                )
            else:
                logger.debug("No repositories needed updating")
                
        except Exception as e:
            logger.error(f"Failed to update knowledge base: {e}", exc_info=True)
            add_notification("error", "Failed to update knowledge base", str(e))
    
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
        # Import memory monitoring functions
        import os
        import psutil
        
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
                import gc
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
        
        for module_name, source in modules_for_repo:
            files = self.knowledge_manager.get_repo_files(repo_id, source.include_paths)
            
            # Process files one by one with immediate storage
            for file_idx, file_path in enumerate(files):
                try:
                    # Monitor memory before each file
                    if file_idx % 5 == 0:
                        current_memory = get_memory_usage()
                        logger.info(f"Processing file {file_idx+1}/{len(files)}: {file_path.name} (memory: {current_memory:.2f}GB)")
                        
                        # Force cleanup if memory is growing too much
                        if current_memory > memory_start + 2.0:  # 2GB increase threshold
                            logger.warning(f"Memory usage high ({current_memory:.2f}GB), forcing aggressive cleanup")
                            force_cleanup()
                    
                    logger.debug(f"Processing file: {file_path.name}")
                    
                    # Process single file
                    chunks = self.document_processor.process_file(
                        file_path, repo_id, repo_state.last_commit_hash
                    )
                    
                    # Force cleanup after document processing
                    force_cleanup()
                    
                    if chunks:
                        # Process this file incrementally - one chunk at a time
                        for chunk_idx, chunk in enumerate(chunks):
                            try:
                                # Monitor memory before each chunk
                                if chunk_idx % 10 == 0:
                                    chunk_memory = get_memory_usage()
                                    logger.debug(f"Processing chunk {chunk_idx+1}/{len(chunks)} from {file_path.name} (memory: {chunk_memory:.2f}GB)")
                                
                                # Generate embedding for single chunk
                                embedding = self.embedding_service.embed_texts([chunk.content])
                                
                                if embedding.size > 0:
                                    # Store immediately in vector store
                                    self.vector_store.add_chunks([chunk], embedding)
                                    total_chunks_processed += 1
                                    
                                    # Ultra-aggressive cleanup after each chunk
                                    del embedding
                                    force_cleanup()
                                    
                            except Exception as e:
                                logger.error(f"Failed to process chunk {chunk_idx} from {file_path.name}: {e}")
                                continue
                    
                    total_files_processed += 1
                    
                    # Ultra-aggressive cleanup after each file
                    force_cleanup()
                    
                    if total_files_processed % 10 == 0:
                        current_memory = get_memory_usage()
                        logger.info(f"Progress: {total_files_processed} files, {total_chunks_processed} chunks (memory: {current_memory:.2f}GB)")
                        
                        # Force cleanup if memory is growing too much
                        if current_memory > memory_start + 2.0:
                            logger.warning(f"Memory usage high ({current_memory:.2f}GB), forcing aggressive cleanup")
                            force_cleanup()
                    
                except Exception as e:
                    logger.error(f"Failed to process file {file_path}: {e}")
                    # Force cleanup after error
                    force_cleanup()
                    continue
        
        # Final cleanup and mark as indexed
        force_cleanup()
        self.knowledge_manager.mark_indexed(repo_id)
        
        memory_end = get_memory_usage()
        logger.info(f"Completed reindex for {repo_id}: {total_chunks_processed} chunks from {total_files_processed} files (memory: {memory_end:.2f}GB, delta: {memory_end - memory_start:.2f}GB)")
        
        if total_chunks_processed > 0:
            add_notification(
                "info",
                "Repository reindexed (ultra-aggressive)",
                f"Repository {repo_id}: {total_chunks_processed} chunks indexed from {total_files_processed} files"
            )
        else:
            logger.warning(f"No chunks processed for repository {repo_id}")
