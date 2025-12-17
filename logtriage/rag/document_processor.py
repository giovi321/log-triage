"""Processes documentation files into chunks."""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import markdown
from dataclasses import dataclass

from ..models import DocumentChunk

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes documentation files into indexed chunks."""
    
    def __init__(self, target_chunk_size: int = 400, overlap_ratio: float = 0.1):
        self.target_chunk_size = target_chunk_size
        self.overlap_size = int(target_chunk_size * overlap_ratio)
        self.allowed_extensions = {'.md', '.rst', '.txt'}
    
    def process_file(self, file_path: Path, repo_id: str, commit_hash: str) -> List[DocumentChunk]:
        """Process a single documentation file into chunks."""
        if file_path.suffix.lower() not in self.allowed_extensions:
            logger.debug(f"Skipping file with unsupported extension: {file_path}")
            return []
        
        try:
            logger.debug(f"Processing file: {file_path}")
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError as e:
            logger.warning(f"Failed to read {file_path} due to encoding error: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}", exc_info=True)
            return []
        
        if not content.strip():
            logger.debug(f"Skipping empty file: {file_path}")
            return []
        
        try:
            if file_path.suffix.lower() == '.md':
                chunks = self._process_markdown(content, file_path, repo_id, commit_hash)
            else:
                chunks = self._process_plain_text(content, file_path, repo_id, commit_hash)
            
            logger.debug(f"Generated {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
            return []
    
    def _process_markdown(self, content: str, file_path: Path, repo_id: str, commit_hash: str) -> List[DocumentChunk]:
        """Process markdown file by splitting on headings."""
        # Split on headings (# ## ### ####)
        heading_pattern = r'^(#{1,4})\s+(.+)$'
        lines = content.split('\n')
        
        chunks = []
        current_heading = "Introduction"
        current_content = []
        
        for line in lines:
            match = re.match(heading_pattern, line)
            if match:
                # Save previous chunk if it has content
                if current_content:
                    chunk_text = '\n'.join(current_content).strip()
                    if chunk_text:
                        chunk = self._create_chunk(
                            chunk_text, file_path, repo_id, commit_hash, current_heading
                        )
                        chunks.extend(chunk)
                
                # Start new chunk
                current_heading = match.group(2).strip()
                current_content = [line]
            else:
                current_content.append(line)
        
        # Don't forget the last chunk
        if current_content:
            chunk_text = '\n'.join(current_content).strip()
            if chunk_text:
                chunk = self._create_chunk(
                    chunk_text, file_path, repo_id, commit_hash, current_heading
                )
                chunks.extend(chunk)
        
        return chunks
    
    def _process_plain_text(self, content: str, file_path: Path, repo_id: str, commit_hash: str) -> List[DocumentChunk]:
        """Process plain text file by paragraph."""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        current_content = []
        current_length = 0
        
        for paragraph in paragraphs:
            para_length = len(paragraph.split())
            
            if current_length + para_length > self.target_chunk_size and current_content:
                # Create chunk
                chunk_text = '\n\n'.join(current_content)
                chunk = self._create_chunk(
                    chunk_text, file_path, repo_id, commit_hash, "Documentation"
                )
                chunks.extend(chunk)
                
                # Start new chunk with overlap
                if self.overlap_size > 0 and len(current_content) > 1:
                    overlap_content = []
                    overlap_length = 0
                    for para in reversed(current_content):
                        para_len = len(para.split())
                        if overlap_length + para_len <= self.overlap_size:
                            overlap_content.insert(0, para)
                            overlap_length += para_len
                        else:
                            break
                    current_content = overlap_content
                    current_length = overlap_length
                else:
                    current_content = []
                    current_length = 0
            
            current_content.append(paragraph)
            current_length += para_length
        
        # Final chunk
        if current_content:
            chunk_text = '\n\n'.join(current_content)
            chunk = self._create_chunk(
                chunk_text, file_path, repo_id, commit_hash, "Documentation"
            )
            chunks.extend(chunk)
        
        return chunks
    
    def _create_chunk(self, content: str, file_path: Path, repo_id: str, commit_hash: str, heading: str) -> List[DocumentChunk]:
        """Create document chunks, splitting if too large."""
        # If content is small enough, create single chunk
        if len(content.split()) <= self.target_chunk_size * 1.5:
            chunk_id = f"{repo_id}:{file_path.relative_to(Path.cwd())}:{hash(content) % 10000}"
            return [DocumentChunk(
                chunk_id=chunk_id,
                repo_id=repo_id,
                file_path=str(file_path),
                heading=heading,
                content=content,
                commit_hash=commit_hash,
                metadata={
                    "file_extension": file_path.suffix,
                    "word_count": len(content.split()),
                    "char_count": len(content)
                }
            )]
        
        # Split large content
        words = content.split()
        chunks = []
        
        start_idx = 0
        chunk_num = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.target_chunk_size, len(words))
            chunk_content = ' '.join(words[start_idx:end_idx])
            
            chunk_id = f"{repo_id}:{file_path.relative_to(Path.cwd())}:{hash(chunk_content) % 10000}_{chunk_num}"
            chunk_heading = f"{heading} (part {chunk_num + 1})" if chunk_num > 0 else heading
            
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                repo_id=repo_id,
                file_path=str(file_path),
                heading=chunk_heading,
                content=chunk_content,
                commit_hash=commit_hash,
                metadata={
                    "file_extension": file_path.suffix,
                    "word_count": len(chunk_content.split()),
                    "char_count": len(chunk_content),
                    "part": chunk_num + 1
                }
            ))
            
            start_idx = end_idx - self.overlap_size
            chunk_num += 1
        
        return chunks
