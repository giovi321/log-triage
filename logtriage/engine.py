from pathlib import Path
from typing import List, Optional

from .models import LogChunk, Severity, PipelineConfig
from .grouping import group_lines
from .classifiers import classify_chunk
from .llm_payload import should_send_to_llm
from .utils import iter_log_files, select_pipeline


def analyze_file(file_path: Path, pcfg: PipelineConfig) -> List[LogChunk]:
    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip("\n") for ln in f]
    except Exception as e:
        chunk = LogChunk(
            file_path=file_path,
            pipeline_name=pcfg.name,
            chunk_index=0,
            lines=[],
            severity=Severity.UNKNOWN,
            reason=f"failed to read file: {e}",
            error_count=0,
            warning_count=0,
            needs_llm=False,
        )
        return [chunk]

    raw_chunks = group_lines(pcfg, lines)
    chunks: List[LogChunk] = []

    for idx, c_lines in enumerate(raw_chunks):
        severity, reason, err_cnt, warn_cnt = classify_chunk(pcfg, c_lines)
        needs_llm = should_send_to_llm(pcfg, severity, c_lines)
        chunk = LogChunk(
            file_path=file_path,
            pipeline_name=pcfg.name,
            chunk_index=idx,
            lines=c_lines,
            severity=severity,
            reason=reason,
            error_count=err_cnt,
            warning_count=warn_cnt,
            needs_llm=needs_llm,
        )
        chunks.append(chunk)

    return chunks


def analyze_path(root: Path, pipelines: List[PipelineConfig], pipeline_override: Optional[str] = None) -> List[LogChunk]:
    if root.is_file() and pipeline_override:
        pipeline_map = {p.name: p for p in pipelines}
        if pipeline_override not in pipeline_map:
            raise ValueError(f"Unknown pipeline {pipeline_override}")
        pcfg = pipeline_map[pipeline_override]
        return analyze_file(root, pcfg)

    files = iter_log_files(root)
    all_chunks: List[LogChunk] = []
    pipeline_map = {p.name: p for p in pipelines}

    for f in files:
        if pipeline_override:
            pcfg = pipeline_map.get(pipeline_override)
            if pcfg is None:
                raise ValueError(f"Unknown pipeline {pipeline_override}")
        else:
            pcfg = select_pipeline(pipelines, f)
        chunks = analyze_file(f, pcfg)
        all_chunks.extend(chunks)
    return all_chunks
