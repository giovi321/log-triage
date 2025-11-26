from pathlib import Path
from typing import List, Optional

from .models import Finding, Severity, PipelineConfig, ModuleLLMConfig
from .classifiers import classify_lines
from .llm_payload import should_send_to_llm
from .utils import iter_log_files, select_pipeline


def analyze_file(
    file_path: Path,
    pcfg: PipelineConfig,
    llm_cfg: ModuleLLMConfig,
    excerpt_limit: int,
    context_prefix_lines: int,
) -> List[Finding]:
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

    findings = classify_lines(
        pcfg,
        file_path,
        pcfg.name,
        lines,
        1,
        excerpt_limit,
        context_prefix_lines,
    )
    for f in findings:
        f.needs_llm = should_send_to_llm(llm_cfg, f.severity, f.excerpt)
    return findings


def analyze_path(
    root: Path,
    pipelines: List[PipelineConfig],
    llm_cfg: ModuleLLMConfig,
    excerpt_limit: int,
    context_prefix_lines: int = 0,
    pipeline_override: Optional[str] = None,
) -> List[Finding]:
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
        )

    files = iter_log_files(root)
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
            f, pcfg, llm_cfg, excerpt_limit, context_prefix_lines
        )
        all_findings.extend(findings)
    return all_findings
