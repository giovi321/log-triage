"""Standalone RAG service that runs as a separate FastAPI process."""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import importlib.util

if importlib.util.find_spec("fastapi") is None:
    raise SystemExit(
        "FastAPI dependencies are missing. Install with `pip install fastapi uvicorn`"
    )

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..config import load_config, build_rag_config, build_modules
from ..models import GlobalLLMConfig, ModuleConfig
from ..notifications import add_notification, list_notifications
from .rag_client import RAGClient

logger = logging.getLogger(__name__)

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

# Global RAG client instance
rag_client: Optional[RAGClient] = None
llm_defaults: Optional[GlobalLLMConfig] = None
modules: List[ModuleConfig] = []

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

def initialize_rag_client(config_path: Path) -> None:
    """Initialize the RAG client from configuration."""
    global rag_client, llm_defaults, modules
    
    try:
        logger.info(f"Loading configuration from {config_path}")
        raw_config = load_config(config_path)
        
        # Configure logging
        configure_logging_from_config(raw_config)
        
        # Build RAG configuration
        rag_config = build_rag_config(raw_config)
        if not rag_config or not rag_config.enabled:
            logger.info("RAG disabled in configuration")
            rag_client = None
            return
        
        logger.info("Initializing RAG client...")
        rag_client = RAGClient(rag_config)
        
        # Build modules configuration
        llm_defaults = build_llm_config(raw_config)
        modules = build_modules(raw_config, llm_defaults)
        
        # Add module configurations to RAG client
        logger.info(f"Adding {len(modules)} modules to RAG client")
        for module in modules:
            if module.rag and module.rag.enabled:
                logger.info(f"Adding RAG config for module: {module.name}")
                rag_client.add_module_config(module.name, module.rag)
        
        # Update knowledge base
        logger.info("Updating RAG knowledge base...")
        rag_client.update_knowledge_base()
        logger.info("RAG service initialization completed")
        
    except Exception as exc:
        logger.error(f"RAG service initialization failed: {exc}", exc_info=True)
        rag_client = None
        raise

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
    initialize_rag_client(config_path)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "rag_enabled": rag_client is not None}

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get RAG system status."""
    if rag_client is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    status = rag_client.get_status()
    return StatusResponse(**status)

@app.post("/retrieve/{module_name}", response_model=RetrievalResponse)
async def retrieve_for_finding(module_name: str, finding: FindingRequest):
    """Retrieve relevant documentation for a finding."""
    if rag_client is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    # Convert finding request to Finding object
    from ..models import Finding, Severity
    finding_obj = Finding(
        file_path=Path(finding.file_path),
        pipeline_name=finding.pipeline_name,
        finding_index=finding.finding_index,
        severity=Severity[finding.severity.upper()],
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
async def update_knowledge_base():
    """Update the knowledge base."""
    if rag_client is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    rag_client.update_knowledge_base()
    return {"message": "Knowledge base update initiated"}

@app.post("/module/{module_name}/config")
async def update_module_config(module_name: str, config: ModuleConfigRequest):
    """Update RAG configuration for a module."""
    if rag_client is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    
    from ..models import RAGModuleConfig, KnowledgeSource
    
    knowledge_sources = []
    for source in config.knowledge_sources:
        knowledge_sources.append(KnowledgeSource(
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
