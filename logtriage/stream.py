import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .models import LogChunk, Severity, PipelineConfig, ModuleConfig
from .classifiers import classify_chunk
from .llm_payload import should_send_to_llm, write_llm_payloads
from .alerts import send_alerts
from .baseline import apply_baseline


def _stat_inode(path: Path) -> Optional[Tuple[int, int, int]]:
    """Return (inode, device, size) tuple or None if path is missing."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return (int(st.st_ino), int(st.st_dev), int(st.st_size))


def follow_file(path: Path, from_beginning: bool, interval: float):
    """Yield lists of new lines appended to the file, handling rotation.

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
    first_open = True

    while True:
        if f is None:
            stat_now = _stat_inode(path)
            if stat_now is None:
                time.sleep(interval)
                continue

            f = path.open("r", encoding="utf-8", errors="replace")
            inode_info = stat_now

            if first_open and from_beginning:
                f.seek(0, 0)
            else:
                f.seek(0, 2)
            first_open = False

        line = f.readline()
        if line:
            buffer.append(line.rstrip("\n"))
            continue

        if buffer:
            yield buffer
            buffer = []

        time.sleep(interval)

        stat_now = _stat_inode(path)
        if stat_now is None:
            try:
                f.close()
            except Exception:
                pass
            f = None
            inode_info = None
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
            continue


def stream_file(mod: ModuleConfig, pcfg: PipelineConfig) -> None:
    """Continuously follow a single log file and classify new chunks."""
    file_path = mod.path
    min_severity = mod.min_print_severity
    emit_llm_dir = mod.emit_llm_payloads_dir
    from_beginning = mod.stream_from_beginning
    interval = mod.stream_interval
    llm_payload_mode = mod.llm_payload_mode

    chunk_index = 0

    for lines in follow_file(file_path, from_beginning=from_beginning, interval=interval):
        if not lines:
            continue

        severity, reason, err_cnt, warn_cnt = classify_chunk(pcfg, lines)
        chunk = LogChunk(
            file_path=file_path,
            pipeline_name=pcfg.name,
            chunk_index=chunk_index,
            lines=lines,
            severity=severity,
            reason=reason,
            error_count=err_cnt,
            warning_count=warn_cnt,
            needs_llm=False,
        )
        chunk_index += 1

        if mod.baseline is not None:
            apply_baseline(mod.baseline, chunk)

        chunk.needs_llm = should_send_to_llm(pcfg, chunk.severity, chunk.lines)

        if chunk.severity >= min_severity:
            print(f"{chunk.file_path} [{chunk.pipeline_name}] chunk={chunk.chunk_index}")
            print(f"  severity: {chunk.severity.name}")
            print(f"  reason:   {chunk.reason}")
            print(f"  errors:   {chunk.error_count}  warnings: {chunk.warning_count}")
            print(f"  needs_llm: {chunk.needs_llm}")
            print()

        if emit_llm_dir is not None and chunk.needs_llm:
            write_llm_payloads([chunk], emit_llm_dir, mode=llm_payload_mode, default_pcfg=pcfg)

        if mod.alert_mqtt or mod.alert_webhook:
            send_alerts(mod, chunk)
