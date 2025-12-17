"""Client wrapper for communicating with the standalone RAG service."""

import logging
import requests
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..models import Finding, RetrievalResult, RAGGlobalConfig, RAGModuleConfig
from ..notifications import add_notification

logger = logging.getLogger(__name__)

class RAGServiceClient:
    """Client for communicating with the standalone RAG service."""
    
    def __init__(self, service_url: str = "http://127.0.0.1:8091", timeout: int = 10):
        self.service_url = service_url.rstrip('/')
        self.timeout = timeout  # Shorter timeout for better responsiveness
        self.session = requests.Session()
        # Configure session for better performance
        self.session.headers.update({
            'User-Agent': 'logtriage-rag-client/1.0'
        })
        
    def _make_request(self, method: str, endpoint: str, max_retries: int = 2, **kwargs) -> Optional[Dict[str, Any]]:
        """Make a request to the RAG service with retry logic."""
        url = f"{self.service_url}{endpoint}"
        
        for attempt in range(max_retries + 1):
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.ConnectionError:
                if attempt == max_retries:
                    logger.error(f"Cannot connect to RAG service at {self.service_url} after {max_retries + 1} attempts")
                    return None
                logger.debug(f"Connection failed to RAG service, retry {attempt + 1}/{max_retries + 1}")
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
            except requests.exceptions.Timeout:
                if attempt == max_retries:
                    logger.error(f"RAG service request timeout to {self.service_url} after {max_retries + 1} attempts")
                    return None
                logger.debug(f"Request timeout to RAG service, retry {attempt + 1}/{max_retries + 1}")
                time.sleep(0.5 * (attempt + 1))
            except requests.exceptions.RequestException as e:
                logger.error(f"RAG service request failed: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error communicating with RAG service: {e}")
                return None
        
        return None
    
    def is_healthy(self) -> bool:
        """Check if the RAG service is healthy."""
        result = self._make_request("GET", "/health")
        if result is None:
            return False
        
        # Check if initialization is still in progress
        init_status = result.get("initialization", {})
        if init_status.get("updating", False):
            logger.info("RAG service is still initializing")
            return True  # Service is healthy, just initializing
        
        return result.get("status") == "healthy"
    
    def is_ready(self) -> bool:
        """Check if the RAG service is ready to serve requests."""
        result = self._make_request("GET", "/health")
        if result is None:
            return False
        
        init_status = result.get("initialization", {})
        return (
            result.get("status") == "healthy" and 
            init_status.get("started", False) and 
            not init_status.get("updating", False) and
            not init_status.get("error")
        )
    
    def add_module_config(self, module_name: str, config: RAGModuleConfig):
        """Add RAG configuration for a module."""
        data = {
            "module_name": module_name,
            "enabled": config.enabled,
            "knowledge_sources": [
                {
                    "repo_url": source.repo_url,
                    "branch": source.branch,
                    "include_paths": source.include_paths
                }
                for source in config.knowledge_sources
            ]
        }
        
        result = self._make_request("POST", f"/module/{module_name}/config", json=data)
        if result is None:
            raise Exception(f"Failed to add module config for {module_name}")
        
        logger.info(f"Added RAG configuration for module: {module_name}")
    
    def update_knowledge_base(self):
        """Update knowledge base if needed."""
        result = self._make_request("POST", "/update-knowledge")
        if result is None:
            raise Exception("Failed to update knowledge base")
        
        logger.info("Knowledge base update initiated")
    
    def retrieve_for_finding(self, finding: Finding, module_name: str) -> Optional[RetrievalResult]:
        """Retrieve relevant documentation for a finding."""
        data = {
            "file_path": str(finding.file_path),
            "pipeline_name": finding.pipeline_name,
            "finding_index": finding.finding_index,
            "severity": finding.severity.name,
            "message": finding.message,
            "line_start": finding.line_start,
            "line_end": finding.line_end,
            "excerpt": finding.excerpt
        }
        
        result = self._make_request("POST", f"/retrieve/{module_name}", json=data, max_retries=1)
        if result is None:
            return None
        
        # Convert response back to RetrievalResult
        from ..models import DocumentChunk, Citation
        
        chunks = []
        for chunk_data in result.get("chunks", []):
            chunks.append(DocumentChunk(
                content=chunk_data["content"],
                source=chunk_data["source"],
                repo_id=chunk_data["repo_id"],
                file_path=chunk_data["file_path"],
                line_start=chunk_data["line_start"],
                line_end=chunk_data["line_end"],
                score=chunk_data["score"]
            ))
        
        citations = []
        for citation_data in result.get("citations", []):
            citations.append(Citation(
                content=citation_data["content"],
                source=citation_data["source"]
            ))
        
        return RetrievalResult(
            chunks=chunks,
            context=result.get("context", ""),
            citations=citations
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get RAG system status for dashboard."""
        result = self._make_request("GET", "/status")
        if result is None:
            return {
                "enabled": False,
                "total_repositories": 0,
                "vector_store_stats": {},
                "repositories": []
            }
        
        return result

# Fallback client that mimics the interface but does nothing
class NoOpRAGClient:
    """Fallback client that does nothing when RAG service is unavailable."""
    
    def __init__(self, *args, **kwargs):
        logger.info("Using NoOpRAGClient - RAG functionality disabled")
    
    def is_healthy(self) -> bool:
        return False
    
    def add_module_config(self, module_name: str, config: RAGModuleConfig):
        pass
    
    def update_knowledge_base(self):
        pass
    
    def retrieve_for_finding(self, finding: Finding, module_name: str) -> Optional[RetrievalResult]:
        return None
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "total_repositories": 0,
            "vector_store_stats": {},
            "repositories": []
        }

def create_rag_client(service_url: str = "http://127.0.0.1:8091", timeout: int = 10, fallback: bool = True):
    """Create a RAG client, optionally with fallback to NoOp client."""
    client = RAGServiceClient(service_url, timeout)
    
    if fallback:
        # Test if service is ready, otherwise return NoOp client
        if not client.is_ready():
            if client.is_healthy():
                logger.warning(f"RAG service at {service_url} is still initializing, using NoOp client")
            else:
                logger.warning(f"RAG service at {service_url} is not available, using NoOp client")
            return NoOpRAGClient()
    
    return client
