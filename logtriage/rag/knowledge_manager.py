"""Manages git repository knowledge sources."""

import hashlib
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import git
from dataclasses import dataclass

from ..models import KnowledgeSourceConfig
from ..notifications import add_notification

logger = logging.getLogger(__name__)

@dataclass
class RepoState:
    """Tracks repository state for incremental updates."""
    repo_id: str
    url: str
    branch: str
    local_path: Path
    last_commit_hash: str
    last_indexed_hash: Optional[str] = None

class KnowledgeManager:
    """Manages git repositories as knowledge sources."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.repositories: Dict[str, RepoState] = {}
        
    def _get_repo_id(self, repo_url: str, branch: str) -> str:
        """Generate unique repository ID."""
        content = f"{repo_url}#{branch}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def add_knowledge_source(self, config: KnowledgeSourceConfig) -> str:
        """Add a knowledge source and return repo ID."""
        repo_id = self._get_repo_id(config.repo_url, config.branch)
        local_path = self.cache_dir / repo_id
        
        try:
            logger.info(f"Adding knowledge source: {config.repo_url} (branch: {config.branch})")
            
            if local_path.exists():
                logger.debug(f"Repository already exists at {local_path}, updating...")
                # Update existing repository
                repo = git.Repo(local_path)
                repo.remotes.origin.fetch()
                repo.git.checkout(config.branch)
                repo.remotes.origin.pull()
            else:
                logger.debug(f"Cloning new repository to {local_path}")
                # Clone new repository
                repo = git.Repo.clone_from(
                    config.repo_url, 
                    local_path,
                    branch=config.branch,
                    depth=1
                )
                # Disable git hooks for security
                hooks_dir = local_path / ".git" / "hooks"
                if hooks_dir.exists():
                    for hook_file in hooks_dir.glob("*"):
                        hook_file.unlink()
            
            # Get current commit hash
            commit_hash = repo.head.commit.hexsha
            
            state = RepoState(
                repo_id=repo_id,
                url=config.repo_url,
                branch=config.branch,
                local_path=local_path,
                last_commit_hash=commit_hash
            )
            
            self.repositories[repo_id] = state
            
            logger.info(f"Successfully added knowledge source {repo_id} at commit {commit_hash[:8]}")
            add_notification(
                "info",
                "Knowledge source updated",
                f"Repository {config.repo_url} at commit {commit_hash[:8]}"
            )
            
            return repo_id
            
        except git.exc.GitCommandError as e:
            error_msg = f"Git command failed for {config.repo_url}: {e}"
            logger.error(error_msg)
            add_notification("error", "Failed to update knowledge source", error_msg)
            raise
        except Exception as e:
            error_msg = f"Failed to update knowledge source {config.repo_url}: {e}"
            logger.error(error_msg, exc_info=True)
            add_notification("error", "Failed to update knowledge source", error_msg)
            raise
    
    def needs_reindexing(self, repo_id: str) -> bool:
        """Check if repository needs reindexing."""
        if repo_id not in self.repositories:
            return True
        
        state = self.repositories[repo_id]
        return state.last_indexed_hash != state.last_commit_hash
    
    def mark_indexed(self, repo_id: str):
        """Mark repository as indexed."""
        if repo_id in self.repositories:
            self.repositories[repo_id].last_indexed_hash = self.repositories[repo_id].last_commit_hash
    
    def get_repo_files(self, repo_id: str, include_paths: List[str]) -> List[Path]:
        """Get all documentation files from repository using glob patterns."""
        if repo_id not in self.repositories:
            return []
        
        state = self.repositories[repo_id]
        repo_path = state.local_path
        
        files = []
        for include_path in include_paths or ["**/*.md", "**/*.rst", "**/*.txt"]:
            try:
                # Use glob patterns - handle both directory patterns and file patterns
                for file_path in repo_path.glob(include_path):
                    if (file_path.is_file() and 
                        not any(part.startswith('.') for part in file_path.parts)):
                        files.append(file_path)
            except Exception as e:
                logger.warning(f"Failed to glob pattern '{include_path}' in {repo_path}: {e}")
                continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for file_path in files:
            if file_path not in seen:
                seen.add(file_path)
                unique_files.append(file_path)
        
        return unique_files
    
    def get_repo_state(self, repo_id: str) -> Optional[RepoState]:
        """Get repository state."""
        return self.repositories.get(repo_id)
