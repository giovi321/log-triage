import json
from pathlib import Path
from typing import Dict, List, Optional

from .models import Finding, Severity, PipelineConfig


def should_send_to_llm(pcfg: PipelineConfig, severity: Severity, excerpt_lines: List[str]) -> bool:
    if not pcfg.llm_cfg.enabled:
        return False
    if severity < pcfg.llm_cfg.min_severity:
        return False
    if len(excerpt_lines) == 0:
        return False
    if len(excerpt_lines) > pcfg.llm_cfg.max_excerpt_lines:
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


def build_llm_payload(pcfg: PipelineConfig, finding: Finding, payload_lines: List[str]) -> str:
    template_text = None
    if pcfg.llm_cfg.prompt_template_path:
        template_text = _load_prompt_template(pcfg.llm_cfg.prompt_template_path)

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
                "Return a single JSON object with keys: severity, reason, key_lines, action_items.\n"
                "Do not include any extra text outside the JSON.\n"
            )
    else:
        header = (
            f"You are analyzing log output from pipeline '{finding.pipeline_name}'.\n"
            f"Rule-based severity: {finding.severity.name}\n"
            f"Reason: {finding.message}\n"
            f"File: {finding.file_path}\n"
            f"Line span: {finding.line_start}-{finding.line_end}\n\n"
            "Return a single JSON object with the following keys:\n"
            "  severity: one of ['OK','INFO','WARNING','ERROR','CRITICAL']\n"
            "  reason: short human-readable explanation\n"
            "  key_lines: list of log lines to highlight\n"
            "  action_items: list of suggested actions or checks\n\n"
            "Do not include any text before or after the JSON.\n"
        )

    body = "\n".join(payload_lines)
    payload = (
        header
        + "\n----- BEGIN LOG CHUNK -----\n"
        + body
        + "\n----- END LOG CHUNK -----\n"
    )
    return payload


def write_llm_payloads(
    findings: List[Finding],
    out_dir: Path,
    mode: str = "full",
    pipeline_map: Optional[Dict[str, PipelineConfig]] = None,
    default_pcfg: Optional[PipelineConfig] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in findings:
        if not f.needs_llm:
            continue

        if mode == "errors_only":
            filtered = _filter_error_like_lines(f.excerpt)
            payload_lines = filtered if filtered else f.excerpt
        else:
            payload_lines = f.excerpt

        if pipeline_map and f.pipeline_name in pipeline_map:
            pcfg = pipeline_map[f.pipeline_name]
        else:
            pcfg = default_pcfg

        if pcfg is None:
            # No pipeline config available; skip payload
            continue

        fname = f"{f.pipeline_name}_{f.severity.name}_finding{f.finding_index}.txt"
        fpath = out_dir / fname
        payload = build_llm_payload(pcfg, f, payload_lines)
        with fpath.open("w", encoding="utf-8") as f:
            f.write(payload)
