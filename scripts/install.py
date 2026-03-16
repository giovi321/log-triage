#!/usr/bin/env python3
"""Interactive installer for log-triage.

Asks which optional extras to install and, if the RAG extra is selected,
whether to use a CPU-only or GPU (CUDA) build of PyTorch.  The CPU-only
build is several gigabytes smaller and is the right choice for machines
that do not have an NVIDIA GPU.
"""

from __future__ import annotations

import subprocess
import sys


def _ask_yes_no(question: str, default: bool = False) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{question} {hint}: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please answer y or n.")


def _ask_choice(question: str, choices: list[str], default: str) -> str:
    choices_str = "/".join(
        c.upper() if c == default else c for c in choices
    )
    while True:
        raw = input(f"{question} [{choices_str}]: ").strip().lower()
        if raw == "":
            return default
        if raw in choices:
            return raw
        print(f"  Please enter one of: {', '.join(choices)}")


def main() -> None:
    print()
    print("=== log-triage interactive installer ===")
    print()

    # --- Extras selection ---
    install_webui = _ask_yes_no("Install Web UI (dashboard)?", default=True)
    install_alerts = _ask_yes_no("Install alert extras (MQTT)?", default=False)
    install_rag = _ask_yes_no(
        "Install RAG support (sentence-transformers + FAISS)?", default=False
    )

    # --- GPU vs CPU ---
    gpu_mode = False
    if install_rag:
        print()
        print("RAG requires PyTorch (via sentence-transformers).")
        print("  cpu  — CPU-only build (~200 MB), works on any machine")
        print("  gpu  — CUDA build (~2+ GB), required for GPU inference")
        print()
        device = _ask_choice(
            "Which PyTorch variant do you want?",
            choices=["cpu", "gpu"],
            default="cpu",
        )
        gpu_mode = device == "gpu"

    # --- Build extras list ---
    extras: list[str] = []
    if install_webui:
        extras.append("webui")
    if install_alerts:
        extras.append("alerts")
    if install_rag:
        extras.append("rag")

    # --- Compose pip command ---
    package_spec = ".[" + ",".join(extras) + "]" if extras else "."

    cmd: list[str] = [sys.executable, "-m", "pip", "install", "--upgrade", package_spec]

    if install_rag and not gpu_mode:
        cmd += ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]

    print()
    print("Running:")
    print("  " + " ".join(cmd))
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print()
        print("Installation failed (see output above).", file=sys.stderr)
        sys.exit(result.returncode)

    print()
    print("Installation complete.")
    if install_rag and not gpu_mode:
        print(
            "Note: CPU-only PyTorch was installed. "
            "If you ever need GPU support, reinstall with:"
        )
        print(f'  pip install "{package_spec}"')


if __name__ == "__main__":
    main()
