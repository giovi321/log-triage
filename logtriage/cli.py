import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict

from .models import Severity, LogChunk, ModuleConfig, PipelineConfig
from .config import load_config, build_pipelines, build_modules
from .engine import analyze_path
from .llm_payload import write_llm_payloads, should_send_to_llm
from .utils import select_pipeline
from .stream import stream_file
from .baseline import apply_baseline
from .alerts import send_alerts


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="log-triage: rule-based log triage and LLM payload generator."
    )
    p.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML configuration file.",
    )
    p.add_argument(
        "--module",
        "-m",
        help="Name of a single module to run (from config.yaml). "
             "If omitted, all enabled modules are run.",
    )
    p.add_argument(
        "--inspect-chunks",
        action="store_true",
        help="Debug mode: for the selected module, print chunk boundaries and line counts instead of triage/LLM.",
    )
    return p.parse_args(argv)


def print_text_summary(chunks: List[LogChunk], min_sev: Severity) -> None:
    for ch in chunks:
        if ch.severity < min_sev:
            continue
        print(f"{ch.file_path} [{ch.pipeline_name}] chunk={ch.chunk_index}")
        print(f"  severity: {ch.severity.name}")
        print(f"  reason:   {ch.reason}")
        print(f"  errors:   {ch.error_count}  warnings: {ch.warning_count}")
        print(f"  needs_llm: {ch.needs_llm}")
        print()


def print_json_summary(chunks: List[LogChunk]) -> None:
    out = []
    for ch in chunks:
        out.append(
            {
                "file_path": str(ch.file_path),
                "pipeline": ch.pipeline_name,
                "chunk_index": ch.chunk_index,
                "severity": ch.severity.name,
                "reason": ch.reason,
                "error_count": ch.error_count,
                "warning_count": ch.warning_count,
                "needs_llm": ch.needs_llm,
                "line_count": len(ch.lines),
            }
        )
    json.dump(out, sys.stdout, indent=2)
    print()


def _filter_only_last_chunk(chunks: List[LogChunk]) -> List[LogChunk]:
    """Return only the last chunk per file_path."""
    if not chunks:
        return chunks

    by_file_last = {}
    for ch in chunks:
        key = ch.file_path
        prev = by_file_last.get(key)
        if prev is None or ch.chunk_index > prev.chunk_index:
            by_file_last[key] = ch

    result = [by_file_last[k] for k in sorted(by_file_last.keys())]
    return result


def inspect_chunks(mod: ModuleConfig, pipelines: List[PipelineConfig]) -> None:
    """Debug helper: show how the selected module's pipeline splits files into chunks."""
    if mod.mode != "batch":
        print(f"Module {mod.name}: --inspect-chunks is only supported for batch modules.", file=sys.stderr)
        sys.exit(1)

    pipeline_map: Dict[str, PipelineConfig] = {p.name: p for p in pipelines}
    if mod.pipeline_name:
        pcfg = pipeline_map.get(mod.pipeline_name)
        if pcfg is None:
            raise ValueError(f"Module {mod.name}: unknown pipeline {mod.pipeline_name}")
    else:
        pcfg = None

    chunks = analyze_path(mod.path, pipelines, pipeline_override=mod.pipeline_name)

    if not chunks:
        print(f"Module {mod.name}: no chunks produced.")
        return

    print(f"Module: {mod.name}")
    print(f"Path:   {mod.path}")
    print(f"Chunks: {len(chunks)}")
    print()

    for ch in chunks:
        line_count = len(ch.lines)
        first_line = ch.lines[0] if ch.lines else ""
        last_line = ch.lines[-1] if ch.lines else ""
        print(f"file={ch.file_path}")
        print(f"  chunk_index: {ch.chunk_index}")
        print(f"  line_count:  {line_count}")
        if first_line:
            print(f"  first: {first_line[:200]}")
        if last_line and last_line != first_line:
            print(f"  last:  {last_line[:200]}")
        print()


def run_module_batch(mod: ModuleConfig, pipelines: List[PipelineConfig]) -> List[LogChunk]:
    pipeline_map: Dict[str, PipelineConfig] = {p.name: p for p in pipelines}
    chunks = analyze_path(mod.path, pipelines, pipeline_override=mod.pipeline_name)

    # Apply only_last_chunk filtering
    if mod.only_last_chunk:
        chunks = _filter_only_last_chunk(chunks)

    # Apply baseline anomaly detection and recompute LLM gating, then alerts
    for ch in chunks:
        pcfg = pipeline_map.get(ch.pipeline_name)
        if pcfg is None:
            continue
        if mod.baseline is not None:
            apply_baseline(mod.baseline, ch)
        ch.needs_llm = should_send_to_llm(pcfg, ch.severity, ch.lines)
        if mod.alert_mqtt or mod.alert_webhook:
            send_alerts(mod, ch)

    # Write LLM payloads if requested
    if mod.emit_llm_payloads_dir:
        write_llm_payloads(
            chunks,
            mod.emit_llm_payloads_dir,
            mode=mod.llm_payload_mode,
            pipeline_map=pipeline_map,
        )

    # Print summary
    if mod.output_format == "json":
        print_json_summary(chunks)
    else:
        print_text_summary(chunks, mod.min_print_severity)

    return chunks


def run_module_follow(mod: ModuleConfig, pipelines: List[PipelineConfig]) -> None:
    if not mod.path.is_file():
        print(f"Module {mod.name}: follow mode requires a file path, got {mod.path}", file=sys.stderr)
        sys.exit(1)

    if mod.pipeline_name:
        pipeline = next((p for p in pipelines if p.name == mod.pipeline_name), None)
        if pipeline is None:
            raise ValueError(f"Module {mod.name}: unknown pipeline {mod.pipeline_name}")
    else:
        pipeline = select_pipeline(pipelines, mod.path)

    stream_file(mod, pipeline)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    cfg_path = Path(args.config)

    cfg = load_config(cfg_path)
    pipelines = build_pipelines(cfg)
    modules = build_modules(cfg)

    if args.module:
        modules_to_run = [m for m in modules if m.name == args.module]
        if not modules_to_run:
            print(f"No module named {args.module} found in config.", file=sys.stderr)
            sys.exit(1)
    else:
        modules_to_run = [m for m in modules if m.enabled]

    if not modules_to_run:
        print("No enabled modules found in config.", file=sys.stderr)
        sys.exit(1)

    # In inspect mode we require exactly one module
    if args.inspect_chunks:
        if len(modules_to_run) != 1:
            print("--inspect-chunks requires exactly one module (use --module).", file=sys.stderr)
            sys.exit(1)
        mod = modules_to_run[0]
        inspect_chunks(mod, pipelines)
        return

    # If a single batch module has an exit_code_by_severity mapping,
    # run it and exit with the mapped code based on highest severity.
    if len(modules_to_run) == 1:
        mod = modules_to_run[0]
        if mod.mode == "batch" and mod.exit_code_by_severity:
            chunks = run_module_batch(mod, pipelines)
            if chunks:
                highest = max((c.severity for c in chunks), default=Severity.UNKNOWN)
            else:
                highest = Severity.UNKNOWN
            exit_code = mod.exit_code_by_severity.get(highest, 0)
            sys.exit(exit_code)
        # else: fall through to normal behavior

    for mod in modules_to_run:
        if mod.mode == "follow":
            run_module_follow(mod, pipelines)
        else:
            run_module_batch(mod, pipelines)
