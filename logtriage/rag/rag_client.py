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
        
        # Prepare memory configuration
        memory_config = {
            'max_memory_gb': global_config.max_memory_gb,
            'warning_memory_gb': global_config.warning_memory_gb,
            'embedding_max_memory_gb': global_config.embedding_max_memory_gb,
            'max_texts_per_batch': global_config.max_texts_per_batch
        }
        
        # Initialize components
        self.knowledge_manager = KnowledgeManager(global_config.cache_dir)
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService(
            model_name=global_config.embedding_model,
            device=global_config.device,
            batch_size=global_config.embedding_batch_size,
            memory_config=memory_config
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
        """Reindex a repository with configurable memory management."""
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
        
        # Use configurable limits
        max_files_to_process = self.global_config.max_files_per_repo
        max_chunks_per_file = self.global_config.max_chunks_per_file
        total_chunks_processed = 0
        total_files_processed = 0
        
        logger.info(f"Starting memory-efficient reindex for {repo_id} (limits: {max_files_to_process} files, {max_chunks_per_file} chunks/file)")
        
        # Check memory before starting
        try:
            import psutil
            memory_gb = psutil.Process().memory_info().rss / 1024**3
            if memory_gb > self.global_config.warning_memory_gb:
                logger.warning(f"High memory usage before reindex: {memory_gb:.2f}GB - aborting")
                return
        except:
            pass
        
        for module_name, source in modules_for_repo:
            files = self.knowledge_manager.get_repo_files(repo_id, source.include_paths)
            
            # Limit total files processed
            if total_files_processed >= max_files_to_process:
                logger.warning(f"Reached file limit ({max_files_to_process}), stopping reindex")
                break
            
            # Process files one by one to minimize memory
            for file_path in files[:max_files_to_process - total_files_processed]:
                # Check memory before each file
                try:
                    import psutil
                    memory_gb = psutil.Process().memory_info().rss / 1024**3
                    if memory_gb > self.global_config.max_memory_gb:
                        logger.warning(f"Memory threshold reached ({memory_gb:.2f}GB), stopping reindex")
                        break
                except:
                    pass
                
                try:
                    # Process single file
                    chunks = self.document_processor.process_file(
                        file_path, repo_id, repo_state.last_commit_hash
                    )
                    
                    # Use configurable chunk limiting per file
                    if len(chunks) > max_chunks_per_file:
                        logger.warning(f"Too many chunks in {file_path.name} ({len(chunks)}), limiting to {max_chunks_per_file}")
                        chunks = chunks[:max_chunks_per_file]
                    
                    if chunks:
                        # Process this file immediately
                        texts = [chunk.content for chunk in chunks]
                        embeddings = self.embedding_service.embed_texts(texts)
                        
                        if embeddings.size > 0:
                            # Add to vector store right away
                            self.vector_store.add_chunks(chunks, embeddings)
                            total_chunks_processed += len(chunks)
                            
                            # Force cleanup after each file
                            import gc
                            gc.collect()
                            
                            logger.debug(f"Processed {file_path.name}: {len(chunks)} chunks, total: {total_chunks_processed}")
                    
                    total_files_processed += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process file {file_path}: {e}")
                    continue
        
        # Mark as indexed even if truncated
        self.knowledge_manager.mark_indexed(repo_id)
        
        if total_chunks_processed > 0:
            add_notification(
                "info",
                "Repository reindexed (memory-limited)",
                f"Repository {repo_id}: {total_chunks_processed} chunks indexed from {total_files_processed} files (memory-efficient mode)"
            )
        else:
            logger.warning(f"No chunks processed for repository {repo_id}")
