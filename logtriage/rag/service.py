"""Standalone RAG service that runs as a separate FastAPI process."""

import logging
import os
import sys
import threading
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional
import importlib.util

if importlib.util.find_spec("fastapi") is None:
    raise SystemExit(
        "FastAPI dependencies are missing. Install with `pip install fastapi uvicorn`"
    )

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ..config import load_config, build_rag_config, build_modules, build_llm_config
from ..models import GlobalLLMConfig, ModuleConfig
from ..notifications import add_notification, list_notifications
from .rag_client import RAGClient

logger = logging.getLogger(__name__)

# Global state
rag_client: Optional[RAGClient] = None
llm_defaults: Optional[GlobalLLMConfig] = None
modules: List[ModuleConfig] = []
initialization_status = {
    "started": False,
    "completed": False,
    "error": None,
    "updating": False,
    "last_update": None,
    "current_phase": "starting",  # "starting", "loading_config", "initializing_client", "adding_modules", "updating_knowledge", "completed", "error"
    "progress": {
        "current_step": 0,
        "total_steps": 5,
        "step_description": "Starting...",
        "percentage": 0.0
    },
    "repository_updates": {
        "current_repo": None,
        "total_repos": 0,
        "completed_repos": 0,
        "current_progress": 0.0,
        "repo_details": {}
    }
}
init_lock = threading.Lock()

# Pydantic models for API
class FindingRequest(BaseModel):
    file_path: str
    pipeline_name: str
    finding_index: int
    severity: str
    message: str
    line_start: int
    line_end: int
    excerpt: List[str]

class RetrievalResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    context: str
    citations: List[Dict[str, Any]]

class StatusResponse(BaseModel):
    enabled: bool
    total_repositories: int
    vector_store_stats: Dict[str, Any]
    repositories: List[Dict[str, Any]]

class ModuleConfigRequest(BaseModel):
    module_name: str
    enabled: bool
    knowledge_sources: List[Dict[str, Any]]

def configure_logging_from_config(cfg: dict) -> None:
    """Configure logging based on configuration dictionary."""
    logging_config = cfg.get("logging", {})
    
    level = logging_config.get("level", "INFO")
    format_str = logging_config.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    log_file = logging_config.get("file")
    logger_levels = logging_config.get("loggers", {})
    
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = []
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_str))
    handlers.append(console_handler)
    
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format_str))
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file handler: {e}", file=sys.stderr)
    
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format=format_str,
        force=True
    )
    
    for logger_name, logger_level in logger_levels.items():
        try:
            logger = logging.getLogger(logger_name)
            logger_numeric_level = getattr(logging, logger_level.upper(), logging.INFO)
            logger.setLevel(logger_numeric_level)
        except Exception as e:
            print(f"Warning: Could not configure logger {logger_name}: {e}", file=sys.stderr)

def background_initialize_rag_client(config_path: Path) -> None:
    """Background thread to initialize the RAG client with configurable memory management."""
    global rag_client, llm_defaults, modules, initialization_status, rag_config
    
    def check_memory_usage():
        """Check memory usage and force cleanup if needed."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_gb = memory_info.rss / 1024**3
            
            # Use configurable memory limits from raw config
            raw_config = load_config(config_path)
            memory_config = raw_config.get("rag", {}).get("memory", {})
            max_memory_gb = memory_config.get("max_memory_gb", 3.0)
            warning_memory_gb = memory_config.get("warning_memory_gb", 2.0)
            
            # Detailed memory breakdown
            memory_percent = process.memory_percent()
            
            logger.info(f"Memory check: {memory_gb:.2f}GB ({memory_percent:.1f}%) - limits: warning={warning_memory_gb}GB, max={max_memory_gb}GB")
            
            if memory_gb > max_memory_gb:
                logger.error(f"CRITICAL memory usage: {memory_gb:.2f}GB ({memory_percent:.1f}%) exceeds limit {max_memory_gb}GB - KILLING PROCESS")
                
                # Force aggressive cleanup
                gc.collect()
                
                # Try to clear SentenceTransformer cache if it exists
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("Cleared CUDA cache")
                except ImportError:
                    pass
                
                # Kill the process to prevent OOM
                import os
                import signal
                logger.error("Terminating process to prevent OOM kill")
                os.kill(os.getpid(), signal.SIGTERM)
                return False
            
            elif memory_gb > warning_memory_gb:
                logger.warning(f"High memory usage: {memory_gb:.2f}GB ({memory_percent:.1f}%) exceeds warning threshold {warning_memory_gb}GB")
                
                # Force aggressive cleanup
                gc.collect()
                
                # Try to clear SentenceTransformer cache if it exists
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("Cleared CUDA cache")
                except ImportError:
                    pass
                
                # Check memory after cleanup
                memory_gb_after = process.memory_info().rss / 1024**3
                logger.warning(f"Memory after cleanup: {memory_gb_after:.2f}GB")
                
                if memory_gb_after > max_memory_gb:
                    logger.error(f"CRITICAL memory usage: {memory_gb_after:.2f}GB exceeds limit {max_memory_gb}GB - KILLING PROCESS")
                    import os
                    import signal
                    os.kill(os.getpid(), signal.SIGTERM)
                    return False
            
            return True
        except Exception as e:
            logger.warning(f"Failed to check memory usage: {e}")
            return True
    
    def update_progress(phase: str, step: int, total: int, description: str):
        """Update progress information."""
        with init_lock:
            initialization_status["current_phase"] = phase
            initialization_status["progress"]["current_step"] = step
            initialization_status["progress"]["total_steps"] = total
            initialization_status["progress"]["step_description"] = description
            initialization_status["progress"]["percentage"] = (step / total) * 100 if total > 0 else 0
    
    def update_repo_progress(repo_name: str, completed: int, total: int, repo_progress: float):
        """Update repository progress information."""
        with init_lock:
            initialization_status["repository_updates"]["current_repo"] = repo_name
            initialization_status["repository_updates"]["completed_repos"] = completed
            initialization_status["repository_updates"]["total_repos"] = total
            initialization_status["repository_updates"]["current_progress"] = repo_progress
    
    with init_lock:
        initialization_status["started"] = True
        initialization_status["completed"] = False
        initialization_status["error"] = None
        initialization_status["updating"] = True
    
    try:
        # Check memory before starting
        if not check_memory_usage():
            return
        
        # Phase 1: Loading configuration
        update_progress("loading_config", 1, 5, "Loading configuration...")
        logger.info(f"Loading configuration from {config_path}")
        raw_config = load_config(config_path)
        
        # Build RAG configuration
        rag_config = build_rag_config(raw_config)
        if not rag_config or not rag_config.enabled:
            logger.info("RAG disabled in configuration")
            with init_lock:
                initialization_status["completed"] = True
                initialization_status["updating"] = False
                update_progress("completed", 5, 5, "RAG disabled")
            return
        
        # Phase 2: Initializing RAG client
        update_progress("initializing_client", 2, 5, "Initializing RAG client...")
        logger.info("Initializing RAG client...")
        rag_client = RAGClient(rag_config)
        
        # Check memory after client initialization
        if not check_memory_usage():
            return
        
        # Phase 3: Building modules
        update_progress("adding_modules", 3, 5, "Building module configurations...")
        llm_defaults = build_llm_config(raw_config)
        modules = build_modules(raw_config, llm_defaults)
        
        # Count RAG-enabled modules for progress tracking
        rag_modules = [m for m in modules if m.rag and m.rag.enabled]
        total_repos = sum(len(m.rag.knowledge_sources) for m in rag_modules)
        
        with init_lock:
            initialization_status["repository_updates"]["total_repos"] = total_repos
        
        # Phase 4: Adding module configurations
        update_progress("adding_modules", 4, 5, f"Adding {len(rag_modules)} modules to RAG client...")
        completed_repos = 0
        for i, module in enumerate(rag_modules):
            if module.rag and module.rag.enabled:
                logger.info(f"Adding RAG config for module: {module.name}")
                rag_client.add_module_config(module.name, module.rag)
                
                # Update progress for each knowledge source
                for source in module.rag.knowledge_sources:
                    completed_repos += 1
                    repo_progress = (completed_repos / total_repos) * 100 if total_repos > 0 else 0
                    update_repo_progress(
                        f"{module.name}:{source.repo_url.split('/')[-1]}",
                        completed_repos,
                        total_repos,
                        repo_progress
                    )
                
                # Check memory after each module
                if not check_memory_usage():
                    return
        
        # Phase 5: Updating knowledge base
        update_progress("updating_knowledge", 5, 5, "Updating knowledge base...")
        logger.info("Updating RAG knowledge base...")
        
        # Simulate progress during knowledge base update (this is the heavy operation)
        update_repo_progress("Knowledge base indexing", completed_repos, total_repos + 1, 0.0)
        rag_client.update_knowledge_base()
        update_repo_progress("Knowledge base indexing", completed_repos, total_repos + 1, 100.0)
        
        # Final memory check
        check_memory_usage()
        
        # Mark as completed
        update_progress("completed", 5, 5, "RAG service ready")
        logger.info("RAG service initialization completed")
        
        with init_lock:
            initialization_status["completed"] = True
            initialization_status["updating"] = False
            initialization_status["last_update"] = time.time()
        
    except Exception as exc:
        logger.error(f"RAG service initialization failed: {exc}", exc_info=True)
        with init_lock:
            initialization_status["error"] = str(exc)
            initialization_status["updating"] = False
            initialization_status["current_phase"] = "error"
            initialization_status["progress"]["step_description"] = f"Error: {str(exc)}"

def initialize_rag_client(config_path: Path) -> None:
    """Initialize the RAG client from configuration in background thread."""
    # Start background initialization
    init_thread = threading.Thread(target=background_initialize_rag_client, args=(config_path,))
    init_thread.daemon = True
    init_thread.start()

# Create FastAPI app
app = FastAPI(
    title="LogTriage RAG Service",
    description="Standalone RAG service for LogTriage",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize RAG service on startup."""
    config_path = Path(os.environ.get("LOGTRIAGE_CONFIG", "./config.yaml"))
    
    # Configure logging first
    try:
        raw_config = load_config(config_path)
        configure_logging_from_config(raw_config)
    except Exception as e:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        logger.warning(f"Failed to load config for logging setup: {e}")
    
    # Start background initialization
    initialize_rag_client(config_path)

@app.get("/health")
async def health_check():
    """Health check endpoint - always responds quickly with detailed status."""
    with init_lock:
        return {
            "status": "healthy", 
            "rag_enabled": rag_client is not None,
            "initialization": {
                "started": initialization_status["started"],
                "completed": initialization_status["completed"],
                "updating": initialization_status["updating"],
                "error": initialization_status["error"],
                "current_phase": initialization_status["current_phase"],
                "progress": initialization_status["progress"].copy(),
                "repository_updates": initialization_status["repository_updates"].copy()
            }
        }

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get RAG system status."""
    with init_lock:
        if not initialization_status["started"]:
            return StatusResponse(
                enabled=False,
                total_repositories=0,
                vector_store_stats={},
                repositories=[]
            )
        
        if initialization_status["updating"]:
            return StatusResponse(
                enabled=True,
                total_repositories=0,
                vector_store_stats={"status": "initializing"},
                repositories=[]
            )
        
        if initialization_status["error"]:
            raise HTTPException(status_code=503, detail=f"RAG initialization failed: {initialization_status['error']}")
    
    if rag_client is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    status = rag_client.get_status()
    return StatusResponse(**status)

@app.post("/retrieve/{module_name}", response_model=RetrievalResponse)
async def retrieve_for_finding(module_name: str, finding: FindingRequest):
    """Retrieve relevant documentation for a finding."""
    with init_lock:
        if initialization_status["updating"]:
            # Return empty result during initialization instead of blocking
            return RetrievalResponse(chunks=[], context="", citations=[])
        
        if initialization_status["error"]:
            raise HTTPException(status_code=503, detail=f"RAG initialization failed: {initialization_status['error']}")
    
    if rag_client is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    # Convert finding request to Finding object
    from ..models import Finding, Severity
    try:
        severity = Severity[finding.severity.upper()]
    except (KeyError, AttributeError):
        # Default to WARNING if severity is invalid
        severity = Severity.WARNING
    
    finding_obj = Finding(
        file_path=Path(finding.file_path),
        pipeline_name=finding.pipeline_name,
        finding_index=finding.finding_index,
        severity=severity,
        message=finding.message,
        line_start=finding.line_start,
        line_end=finding.line_end,
        excerpt=finding.excerpt
    )
    
    result = rag_client.retrieve_for_finding(finding_obj, module_name)
    if result is None:
        return RetrievalResponse(chunks=[], context="", citations=[])
    
    return RetrievalResponse(
        chunks=[{
            "content": chunk.content,
            "source": chunk.source,
            "repo_id": chunk.repo_id,
            "file_path": chunk.file_path,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "score": chunk.score
        } for chunk in result.chunks],
        context=result.context,
        citations=[{
            "content": citation.content,
            "source": citation.source
        } for citation in result.citations]
    )

@app.post("/update-knowledge")
async def update_knowledge_base(background_tasks: BackgroundTasks):
    """Update the knowledge base in background."""
    if rag_client is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    # Run update in background to avoid blocking
    background_tasks.add_task(rag_client.update_knowledge_base)
    return {"message": "Knowledge base update initiated in background"}

@app.post("/module/{module_name}/config")
async def update_module_config(module_name: str, config: ModuleConfigRequest):
    """Update RAG configuration for a module."""
    with init_lock:
        if initialization_status["updating"]:
            raise HTTPException(status_code=509, detail="RAG service is currently initializing, please try again later")
        
        if initialization_status["error"]:
            raise HTTPException(status_code=503, detail=f"RAG initialization failed: {initialization_status['error']}")
    
    if rag_client is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    from ..models import RAGModuleConfig, KnowledgeSourceConfig
    
    knowledge_sources = []
    for source in config.knowledge_sources:
        knowledge_sources.append(KnowledgeSourceConfig(
            repo_url=source["repo_url"],
            branch=source["branch"],
            include_paths=source["include_paths"]
        ))
    
    module_config = RAGModuleConfig(
        enabled=config.enabled,
        knowledge_sources=knowledge_sources
    )
    
    rag_client.add_module_config(module_name, module_config)
    return {"message": f"Configuration updated for module {module_name}"}

@app.get("/notifications")
async def get_notifications():
    """Get system notifications."""
    return {"notifications": list_notifications()}

def main():
    """Entry point for the RAG service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LogTriage RAG Service")
    parser.add_argument(
        "--config",
        "-c",
        default="./config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8091,
        help="Port to bind to"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    # Set config path environment variable
    os.environ["LOGTRIAGE_CONFIG"] = args.config
    
    # Initialize logging before starting the app
    try:
        config_path = Path(args.config)
        if config_path.exists():
            raw_config = load_config(config_path)
            configure_logging_from_config(raw_config)
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s"
            )
            logger.warning(f"Config file {args.config} not found, using default logging")
    except Exception as e:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        logger.warning(f"Failed to load config for logging setup: {e}")
    
    logger.info(f"Starting RAG service on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
