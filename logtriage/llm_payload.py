import json
from pathlib import Path
from typing import Dict, List, Optional

from .models import Finding, Severity, ModuleLLMConfig


def should_send_to_llm(llm_cfg: ModuleLLMConfig, severity: Severity, excerpt_lines: List[str]) -> bool:
    if not llm_cfg.enabled:
        return False
    if severity < llm_cfg.min_severity:
        return False
    if len(excerpt_lines) == 0:
        return False
    if len(excerpt_lines) > llm_cfg.max_excerpt_lines:
        return False
    return True


def _filter_error_like_lines(lines: List[str]) -> List[str]:
    tokens = [
        "error",
        "failed",
        "failure",
        "exception",
        "traceback",
        "fatal",
        "critical",
        "warning",
        "warn",
    ]
    out: List[str] = []
    for ln in lines:
        low = ln.lower()
        if any(tok in low for tok in tokens):
            out.append(ln)
    return out


_PROMPT_CACHE: Dict[Path, str] = {}


def _load_prompt_template(path: Path) -> Optional[str]:
    if not path:
        return None
    if path in _PROMPT_CACHE:
        return _PROMPT_CACHE[path]
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    _PROMPT_CACHE[path] = text
    return text


def render_llm_payload(
    finding: Finding, llm_cfg: ModuleLLMConfig, mode: str = "errors_only", rag_context: str = ""
) -> Optional[str]:
    if not finding.needs_llm:
        return None

    if mode == "errors_only":
        filtered = _filter_error_like_lines(finding.excerpt)
        payload_lines = filtered if filtered else finding.excerpt
    else:
        payload_lines = finding.excerpt

    return build_llm_payload(llm_cfg, finding, payload_lines, rag_context)


def build_llm_payload(llm_cfg: ModuleLLMConfig, finding: Finding, payload_lines: List[str], rag_context: str = "") -> str:
    template_text = None
    if llm_cfg.prompt_template_path:
        template_text = _load_prompt_template(llm_cfg.prompt_template_path)

    if template_text:
        try:
            header = template_text.format(
                severity=finding.severity.name,
                reason=finding.message,
                file_path=str(finding.file_path),
                pipeline=finding.pipeline_name,
                line_count=len(payload_lines),
            )
        except Exception:
            header = (
                f"You are analyzing log output from pipeline '{finding.pipeline_name}'.\n"
                f"Rule-based severity: {finding.severity.name}\n"
                f"Reason: {finding.message}\n"
                f"File: {finding.file_path}\n"
                f"Line span: {finding.line_start}-{finding.line_end}\n\n"
                "Write a concise, human-readable briefing. Begin with the severity and a short headline. "
                "Explain the main issues in 2–4 sentences, reference the most relevant log lines (up to 5), "
                "and list a few action items the operator should try next. Do not use JSON or code blocks.\n"
            )
    else:
        header = (
            f"You are analyzing log output from pipeline '{finding.pipeline_name}'.\n"
            f"Rule-based severity: {finding.severity.name}\n"
            f"Reason: {finding.message}\n"
            f"File: {finding.file_path}\n"
            f"Line span: {finding.line_start}-{finding.line_end}\n\n"
            "Provide an operator-ready summary: state the severity and headline, describe the main issues in a few sentences, "
            "cite the most important log lines, and recommend concrete follow-up steps. Keep everything as plain text (no JSON "
            "or code blocks).\n"
        )

    body = "\n".join(payload_lines)
    payload = (
        header
        + "\n----- BEGIN LOG CHUNK -----\n"
        + body
        + "\n----- END LOG CHUNK -----\n"
    )
    
    # Add RAG context if provided
    if rag_context:
        payload += rag_context
    
    return payload


def write_llm_payloads(
    findings: List[Finding],
    llm_cfg: ModuleLLMConfig,
    out_dir: Path,
    mode: str = "errors_only",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in findings:
        if not f.needs_llm:
            continue

        fname = f"{f.pipeline_name}_{f.severity.name}_finding{f.finding_index}.txt"
        fpath = out_dir / fname
        payload = render_llm_payload(f, llm_cfg, mode=mode)
        if payload is None:
            continue
        with fpath.open("w", encoding="utf-8") as f:
            f.write(payload)
