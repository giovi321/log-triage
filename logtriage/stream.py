import datetime
import os
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

from .models import Finding, Severity, PipelineConfig, ModuleConfig
from .classifiers import classify_lines
from .llm_client import analyze_findings_with_llm
from .llm_payload import should_send_to_llm, write_llm_payloads
from .alerts import send_alerts
from .webui.db import store_finding, get_next_finding_index

# Import RAG service client (optional import to avoid circular dependencies)
try:
    from .rag.service_client import create_rag_client
    from .config import load_config, build_rag_config
except ImportError:
    create_rag_client = None
    load_config = None
    build_rag_config = None


def _stat_inode(path: Path) -> Optional[Tuple[int, int, int]]:
    """Return (inode, device, size) tuple or None if path is missing."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return (int(st.st_ino), int(st.st_dev), int(st.st_size))


def follow_file(path: Path, from_beginning: bool, interval: float, should_stop=None):
    """Yield (start_line, lines) batches appended to the file, handling rotation.

    Behavior is similar to `tail -F`:

    - On first open:
      - if from_beginning is True, read from start
      - otherwise seek to end
    - On rotation (inode change) or truncation (size shrinks):
      - reopen the file
      - start reading from the end of the new file (ignore historical content)
    - Lines are accumulated until there is a pause (no new data) of at least
      `interval` seconds, then yielded as a batch.
    """
    f = None
    inode_info: Optional[Tuple[int, int, int]] = None
    buffer: List[str] = []
    current_line_number = 1
    first_open = True

    while True:
        if should_stop is not None and should_stop():
            return
        if f is None:
            stat_now = _stat_inode(path)
            if stat_now is None:
                time.sleep(interval)
                continue

            f = path.open("r", encoding="utf-8", errors="replace")
            inode_info = stat_now

            if first_open and from_beginning:
                f.seek(0, 0)
                current_line_number = 1
            else:
                # Count actual lines in the file so line_start is absolute
                f.seek(0, 0)
                current_line_number = sum(1 for _ in f) + 1
                # Now seek to end to follow new lines
                f.seek(0, 2)
            first_open = False

        line = f.readline()
        if line:
            buffer.append(line.rstrip("\n"))
            current_line_number += 1
            continue

        if buffer:
            start = current_line_number - len(buffer)
            yield start, buffer
            buffer = []

        time.sleep(interval)

        if should_stop is not None and should_stop():
            return

        stat_now = _stat_inode(path)
        if stat_now is None:
            try:
                f.close()
            except Exception:
                pass
            f = None
            inode_info = None
            current_line_number = 1
            continue

        ino, dev, size = stat_now
        old_ino = None
        old_dev = None
        old_size = None
        if inode_info is not None:
            old_ino, old_dev, old_size = inode_info

        rotated = old_ino is not None and (ino != old_ino or dev != old_dev)
        truncated = old_size is not None and size < old_size

        if rotated or truncated:
            try:
                f.close()
            except Exception:
                pass
            f = None
            inode_info = None
            # Reset line number tracking - we'll seek to end on next open
            current_line_number = 1
            first_open = False  # Ensure we seek to end, not beginning
            continue


def stream_file(
    mod: ModuleConfig,
    pcfg: PipelineConfig,
    llm_defaults,
    should_reload=None,
) -> None:
    """Continuously follow a single log file and classify new findings."""
    file_path = mod.path
    min_severity = mod.min_print_severity
    emit_llm_dir = mod.llm.emit_llm_payloads_dir
    from_beginning = mod.stream_from_beginning
    interval = mod.stream_interval
    context_prefix_lines = mod.llm.context_prefix_lines
    context_suffix_lines = mod.llm.context_suffix_lines
    prefix_buffer: Deque[str] = (
        deque(maxlen=context_prefix_lines) if context_prefix_lines > 0 else deque()
    )

    # Create RAG client if available
    rag_client = None
    if create_rag_client is not None and load_config is not None and build_rag_config is not None:
        try:
            # Load config to get RAG settings
            config_path = Path(os.environ.get("LOGTRIAGE_CONFIG", "./config.yaml"))
            cfg = load_config(config_path)
            rag_config = build_rag_config(cfg)
            if rag_config and rag_config.enabled:
                rag_service_url = rag_config.service_url if hasattr(rag_config, 'service_url') else "http://127.0.0.1:8091"
                rag_client = create_rag_client(rag_service_url, fallback=True)
                print(f"RAG service client configured for module {mod.name}")
        except Exception as e:
            print(f"Warning: Failed to initialize RAG service client for {mod.name}: {e}")

    try:
        finding_index = get_next_finding_index(mod.name) - 1
    except Exception:
        # Fallback to 0 if database is not available
        finding_index = 0

    for start_line, lines in follow_file(
        file_path,
        from_beginning=from_beginning,
        interval=interval,
        should_stop=should_reload,
    ):
        if not lines:
            continue

        if should_reload is not None and should_reload():
            break

        findings = classify_lines(
            pcfg,
            file_path,
            pcfg.name,
            lines,
            start_line,
            mod.llm.max_excerpt_lines,
            context_prefix_lines,
            context_suffix_lines,
            list(prefix_buffer),
        )
        now = datetime.datetime.now(datetime.timezone.utc)
        for f in findings:
            if getattr(f, "created_at", None) is None:
                f.created_at = now
        for f in findings:
            f.finding_index = finding_index
            finding_index += 1

        if context_prefix_lines > 0:
            for ln in lines[-context_prefix_lines:]:
                prefix_buffer.append(ln)

        for f in findings:
            f.needs_llm = should_send_to_llm(mod.llm, f.severity, f.excerpt)

        for f in findings:
            if f.needs_llm:
                analyze_findings_with_llm([f], llm_defaults, mod.llm, rag_client=rag_client, module_name=mod.name)
            if emit_llm_dir is not None and f.needs_llm:
                write_llm_payloads([f], mod.llm, emit_llm_dir)

            if f.severity >= min_severity:
                print(f"{f.file_path} [{f.pipeline_name}] finding={f.finding_index}")
                print(f"  severity: {f.severity.name}")
                print(f"  reason:   {f.message}")
                print(f"  lines:    {f.line_start}-{f.line_end}")
                print(f"  needs_llm: {f.needs_llm}")
                print()

            try:
                store_finding(mod.name, f)
            except Exception:
                pass

            if mod.alert_mqtt or mod.alert_webhook:
                send_alerts(mod, f)

