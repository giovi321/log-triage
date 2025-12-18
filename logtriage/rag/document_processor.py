"""Processes documentation files into chunks with aggressive memory management."""

import logging
import re
import gc
import os
import psutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import markdown
from dataclasses import dataclass

from ..models import DocumentChunk

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
    for _ in range(3):
        gc.collect()

class DocumentProcessor:
    """Processes documentation files into indexed chunks with memory management."""
    
    def __init__(self, target_chunk_size: int = 400, overlap_ratio: float = 0.1):
        self.target_chunk_size = target_chunk_size
        self.overlap_size = int(target_chunk_size * overlap_ratio)
        self.chunks_processed = 0
        
        logger.info(f"DocumentProcessor initialized: chunk_size={target_chunk_size}, overlap={overlap_ratio}")
    
    def process_file(self, file_path: Path, repo_id: str, commit_hash: str) -> List[DocumentChunk]:
        """Process a single documentation file into chunks with memory management."""
        return list(self.process_file_iter(file_path, repo_id, commit_hash))

    def process_file_iter(self, file_path: Path, repo_id: str, commit_hash: str) -> Iterator[DocumentChunk]:
        try:
            logger.debug(f"Processing file: {file_path}")

            if file_path.suffix.lower() == '.md':
                yield from self._iter_process_markdown_file(file_path, repo_id, commit_hash)
            else:
                yield from self._iter_process_plain_text_file(file_path, repo_id, commit_hash)

        except UnicodeDecodeError as e:
            logger.warning(f"Failed to read {file_path} due to encoding error: {e}")
            return
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}", exc_info=True)

            return

    def _iter_process_markdown_file(self, file_path: Path, repo_id: str, commit_hash: str) -> Iterator[DocumentChunk]:
        heading_pattern = re.compile(r'^(#{1,4})\s+(.+)$')

        current_heading = "Introduction"
        current_lines: List[str] = []

        with file_path.open('r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                match = heading_pattern.match(line)
                if match:
                    if current_lines:
                        chunk_text = ''.join(current_lines).strip()
                        if chunk_text:
                            for chunk in self._iter_create_chunk(chunk_text, file_path, repo_id, commit_hash, current_heading):
                                yield chunk
                                self.chunks_processed += 1
                        current_lines = []

                    current_heading = match.group(2).strip() or current_heading
                    current_lines.append(line)
                else:
                    current_lines.append(line)

                if i % 200 == 0:
                    force_cleanup()

        if current_lines:
            chunk_text = ''.join(current_lines).strip()
            if chunk_text:
                for chunk in self._iter_create_chunk(chunk_text, file_path, repo_id, commit_hash, current_heading):
                    yield chunk
                    self.chunks_processed += 1

        if self.chunks_processed % 100 == 0 and self.chunks_processed > 0:
            current_memory = get_memory_usage()
            logger.info(f"Processed {self.chunks_processed} chunks total (memory: {current_memory:.2f}GB)")
            force_cleanup()

    def _iter_process_plain_text_file(self, file_path: Path, repo_id: str, commit_hash: str) -> Iterator[DocumentChunk]:
        def paragraph_iter() -> Iterator[str]:
            buf: List[str] = []
            with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip() == "":
                        if buf:
                            yield ''.join(buf).strip()
                            buf = []
                    else:
                        buf.append(line)
                if buf:
                    yield ''.join(buf).strip()

        chunks: List[str] = []
        current_length = 0

        for i, paragraph in enumerate(paragraph_iter()):
            if not paragraph:
                continue

            para_length = len(paragraph.split())

            if current_length + para_length > self.target_chunk_size and chunks:
                chunk_text = '\n\n'.join(chunks)
                for chunk in self._iter_create_chunk(chunk_text, file_path, repo_id, commit_hash, "Documentation"):
                    yield chunk
                    self.chunks_processed += 1

                if self.overlap_size > 0 and len(chunks) > 1:
                    overlap_content: List[str] = []
                    overlap_length = 0
                    for para in reversed(chunks):
                        para_len = len(para.split())
                        if overlap_length + para_len <= self.overlap_size:
                            overlap_content.insert(0, para)
                            overlap_length += para_len
                        else:
                            break
                    chunks = overlap_content
                    current_length = overlap_length
                else:
                    chunks = []
                    current_length = 0

            chunks.append(paragraph)
            current_length += para_length

            if i % 50 == 0:
                force_cleanup()

        if chunks:
            chunk_text = '\n\n'.join(chunks)
            for chunk in self._iter_create_chunk(chunk_text, file_path, repo_id, commit_hash, "Documentation"):
                yield chunk
                self.chunks_processed += 1

        if self.chunks_processed % 100 == 0 and self.chunks_processed > 0:
            current_memory = get_memory_usage()
            logger.info(f"Processed {self.chunks_processed} chunks total (memory: {current_memory:.2f}GB)")
            force_cleanup()
    
    def _process_markdown(self, content: str, file_path: Path, repo_id: str, commit_hash: str) -> List[DocumentChunk]:
        """Process markdown file by splitting on headings with memory management."""
        # Split on headings (# ## ### ####)
        heading_pattern = r'^(#{1,4})\s+(.+)$'
        lines = content.split('\n')
        
        chunks = []
        current_heading = "Introduction"
        current_content = []
        
        for i, line in enumerate(lines):
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
                        
                        # Cleanup after creating chunk
                        del chunk_text
                        force_cleanup()
                
                # Start new chunk
                current_heading = match.group(2).strip()
                current_content = [line]
            else:
                current_content.append(line)
            
            # Periodic cleanup every 50 lines
            if i % 50 == 0:
                force_cleanup()
        
        # Don't forget the last chunk
        if current_content:
            chunk_text = '\n'.join(current_content).strip()
            if chunk_text:
                chunk = self._create_chunk(
                    chunk_text, file_path, repo_id, commit_hash, current_heading
                )
                chunks.extend(chunk)
                
                # Cleanup
                del chunk_text
                force_cleanup()
        
        # Final cleanup
        del lines
        del current_content
        force_cleanup()
        
        return chunks
    
    def _process_plain_text(self, content: str, file_path: Path, repo_id: str, commit_hash: str) -> List[DocumentChunk]:
        """Process plain text file by paragraph with memory management."""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        current_content = []
        current_length = 0
        
        for i, paragraph in enumerate(paragraphs):
            para_length = len(paragraph.split())
            
            if current_length + para_length > self.target_chunk_size and current_content:
                # Create chunk
                chunk_text = '\n\n'.join(current_content)
                chunk = self._create_chunk(
                    chunk_text, file_path, repo_id, commit_hash, "Documentation"
                )
                chunks.extend(chunk)
                
                # Cleanup after creating chunk
                del chunk_text
                force_cleanup()
                
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
            
            # Periodic cleanup every 20 paragraphs
            if i % 20 == 0:
                force_cleanup()
        
        # Final chunk
        if current_content:
            chunk_text = '\n\n'.join(current_content)
            chunk = self._create_chunk(
                chunk_text, file_path, repo_id, commit_hash, "Documentation"
            )
            chunks.extend(chunk)
            
            # Cleanup
            del chunk_text
            force_cleanup()
        
        # Final cleanup
        del paragraphs
        del current_content
        force_cleanup()
        
        return chunks
    
    def _create_chunk(self, content: str, file_path: Path, repo_id: str, commit_hash: str, heading: str) -> List[DocumentChunk]:
        """Create document chunks, splitting if too large."""
        # If content is small enough, create single chunk
        if len(content.split()) <= self.target_chunk_size * 1.5:
            # Use file_path.name instead of relative_to to avoid path issues
            chunk_id = f"{repo_id}:{file_path.name}:{hash(content) % 10000}"
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
            
            # Use file_path.name instead of relative_to to avoid path issues
            chunk_id = f"{repo_id}:{file_path.name}:{hash(chunk_content) % 10000}_{chunk_num}"
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

            if end_idx >= len(words):
                break

            start_idx = max(0, end_idx - self.overlap_size)
            chunk_num += 1
        
        return chunks

    def _iter_create_chunk(self, content: str, file_path: Path, repo_id: str, commit_hash: str, heading: str) -> Iterator[DocumentChunk]:
        if len(content.split()) <= self.target_chunk_size * 1.5:
            chunk_id = f"{repo_id}:{file_path.name}:{hash(content) % 10000}"
            yield DocumentChunk(
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
            )
            return

        words = content.split()
        start_idx = 0
        chunk_num = 0

        while start_idx < len(words):
            end_idx = min(start_idx + self.target_chunk_size, len(words))
            chunk_content = ' '.join(words[start_idx:end_idx])
            chunk_id = f"{repo_id}:{file_path.name}:{hash(chunk_content) % 10000}_{chunk_num}"
            chunk_heading = f"{heading} (part {chunk_num + 1})" if chunk_num > 0 else heading

            yield DocumentChunk(
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
            )

            if end_idx >= len(words):
                break

            start_idx = max(0, end_idx - self.overlap_size)
            chunk_num += 1
