import datetime
from pathlib import Path
from typing import List, Optional

from .models import Finding, Severity, PipelineConfig, ModuleLLMConfig
from .classifiers import classify_lines
from .grouping import group_lines
from .llm_payload import should_send_to_llm
from .notifications import add_notification
from .utils import iter_log_files, select_pipeline


def analyze_file(
    file_path: Path,
    pcfg: PipelineConfig,
    llm_cfg: ModuleLLMConfig,
    excerpt_limit: int,
    context_prefix_lines: int,
    context_suffix_lines: int = 0,
) -> List[Finding]:
    """Analyze a single log file using the specified pipeline.
    
    Reads the file, applies grouping and classification, and returns
    a list of findings with context excerpts and LLM readiness flags.
    
    Args:
        file_path: Path to the log file
        pcfg: Pipeline configuration for processing
        llm_cfg: LLM configuration for payload generation
        excerpt_limit: Maximum lines in finding excerpts
        context_prefix_lines: Context lines before matches
        context_suffix_lines: Context lines after matches
        
    Returns:
        List of findings discovered in the file
    """
    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip("\n") for ln in f]
    except Exception as e:
        return [
            Finding(
                file_path=file_path,
                pipeline_name=pcfg.name,
                finding_index=0,
                severity=Severity.ERROR,
                message=f"failed to read file: {e}",
                line_start=1,
                line_end=1,
                rule_id=None,
                excerpt=[],
                needs_llm=False,
            )
        ]

    groups = group_lines(pcfg, lines)
    if not groups:
        return []

    groups_with_offsets = []
    offset = 0
    for chunk in groups:
        groups_with_offsets.append((offset, chunk))
        offset += len(chunk)

    if pcfg.grouping_only_last and groups_with_offsets:
        groups_with_offsets = [groups_with_offsets[-1]]

    try:
        created_at = datetime.datetime.fromtimestamp(
            file_path.stat().st_mtime, tz=datetime.timezone.utc
        )
    except Exception:
        created_at = None

    findings: List[Finding] = []

    for offset, chunk in groups_with_offsets:
        start_line = 1 + offset
        prefix_lines = []
        if context_prefix_lines > 0:
            prefix_lines = lines[max(0, offset - context_prefix_lines) : offset]

        chunk_findings = classify_lines(
            pcfg,
            file_path,
            pcfg.name,
            chunk,
            start_line,
            excerpt_limit,
            context_prefix_lines,
            context_suffix_lines,
            prefix_lines,
        )
        for f in chunk_findings:
            if getattr(f, "created_at", None) is None and created_at is not None:
                f.created_at = created_at
            f.finding_index = len(findings)
            f.needs_llm = should_send_to_llm(llm_cfg, f.severity, f.excerpt)
            findings.append(f)

    return findings


def analyze_path(
    root: Path,
    pipelines: List[PipelineConfig],
    llm_cfg: ModuleLLMConfig,
    excerpt_limit: int,
    context_prefix_lines: int = 0,
    context_suffix_lines: int = 0,
    pipeline_override: Optional[str] = None,
) -> List[Finding]:
    """Analyze a path (file or directory) for log findings.
    
    Processes either a single file or all files in a directory tree,
    applying appropriate pipelines and collecting all findings.
    
    Args:
        root: Path to analyze (file or directory)
        pipelines: Available pipeline configurations
        llm_cfg: LLM configuration for payload generation
        excerpt_limit: Maximum lines in finding excerpts
        context_prefix_lines: Context lines before matches
        context_suffix_lines: Context lines after matches
        pipeline_override: Force use of specific pipeline
        
    Returns:
        List of all findings discovered
    """
    if not root.exists():
        add_notification(
            "error",
            "Source path missing",
            f"Path {root} does not exist",
        )
        return []

    if root.is_file() and pipeline_override:
        pipeline_map = {p.name: p for p in pipelines}
        if pipeline_override not in pipeline_map:
            raise ValueError(f"Unknown pipeline {pipeline_override}")
        pcfg = pipeline_map[pipeline_override]
        return analyze_file(
            root,
            pcfg,
            llm_cfg,
            excerpt_limit,
            context_prefix_lines,
            context_suffix_lines,
        )

    files = iter_log_files(root)
    if not files:
        add_notification(
            "warning",
            "No log files found",
            f"No files discovered under {root}",
        )
    all_findings: List[Finding] = []
    pipeline_map = {p.name: p for p in pipelines}

    for f in files:
        if pipeline_override:
            pcfg = pipeline_map.get(pipeline_override)
            if pcfg is None:
                raise ValueError(f"Unknown pipeline {pipeline_override}")
        else:
            pcfg = select_pipeline(pipelines, f)
        findings = analyze_file(
            f, pcfg, llm_cfg, excerpt_limit, context_prefix_lines, context_suffix_lines
        )
        all_findings.extend(findings)
    return all_findings
