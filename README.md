[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/github/license/giovi321/log-triage)](https://github.com/giovi321/log-triage/blob/main/LICENSE)

<img width="317" height="55" alt="log-triage" src="https://github.com/user-attachments/assets/1da91cb3-fe19-408a-80c8-e894f57542a8" />

`log-triage` is a Python tool that sits between your log collector (for example Fluent Bit) and an LLM. It filters noisy logs, detects problems, and gives you a dashboard and API-ready payloads so you can triage faster.

In addition to raw LLM prompting, `log-triage` can run in **RAG (Retrieval-Augmented Generation)** mode: it indexes your documentation repositories and automatically retrieves relevant snippets to ground AI responses with context and citations.

## Overview

### Key concepts
- **Pipelines:** Reusable recipes that define how to group log lines, which regexes to ignore or count, and which prompt template to use for LLM payloads.
- **Modules:** Runtime bindings that attach a pipeline to a file path and decide whether to scan once (batch) or tail continuously (follow) with rotation awareness.
- **Findings:** Structured outputs for each grouped chunk, including severity (WARNING/ERROR/CRITICAL), counts, and optional LLM payloads.
- **Issues:** Recurring findings collapsed by a *signature* (the log line with timestamps/IDs/IPs/numbers normalized away). Each issue tracks an occurrence count, first/last-seen window, the highest severity seen, a workflow status, and a **cached LLM summary** generated once per signature. This is what the **Triage** queue ranks and shows.
- **Addressed & false positives:** Workflow flags in the dashboard; marking a false positive also writes an ignore regex back to the pipeline to prevent repeats.

### How it works

<img src="docs/assets/diagram-dataflow.svg" alt="log-triage data flow: findings collapse into one cached issue" width="100%">

`log-triage` watches your logs, passes them through a configured pipeline, and surfaces only the important pieces:

1. **Collect:** Point a module at a log file (or directory) to read entries once or continuously with rotation handling.
2. **Group:** Apply the pipeline's grouping strategy (whole-file or marker-based) to carve the stream into logical chunks.
3. **Classify:** Count warnings and errors with regex rules, ignore known-noise patterns, and assign a severity.
4. **Deduplicate:** Fold recurring findings into a single **issue** keyed by a normalized signature, with an occurrence count and a first/last-seen window.
5. **Enrich:** Analyze each issue with the LLM **once per signature** (grounded with RAG when configured) and cache the summary — so a problem that occurs 10,000 times costs one LLM call, and the explanation is consistent everywhere.
6. **Deliver:** Triage from a prioritized queue (severity × recency × rate × novelty), send alerts (webhook/MQTT), expose Prometheus `/metrics`, and acknowledge/resolve/mute issues from the dashboard.

### Getting started
1. **Install the package:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   # Interactive installer — prompts for extras and CPU vs GPU PyTorch
   python scripts/install.py
   ```
   The installer asks which optional extras you want (`webui`, `alerts`, `rag`) and,
   if RAG is selected, whether to use a **CPU-only** (~200 MB) or **GPU/CUDA** (~2+ GB)
   build of PyTorch.  Choose CPU unless you are running GPU inference on this machine.

   <details>
   <summary>Manual install (skip the prompt)</summary>

   ```bash
   # CPU-only RAG build (recommended for most machines)
   pip install ".[webui,alerts,rag]" --extra-index-url https://download.pytorch.org/whl/cpu

   # GPU/CUDA RAG build
   pip install ".[webui,alerts,rag]"
   ```
   </details>
2. **Configure:** Copy `config.yaml` and edit pipelines/modules to point at your log files.
3. **Run a module** (omit `--module` to run all enabled follow-mode modules):
   ```bash
   logtriage --config ./config.yaml --module <module-name>
   # add --reload-on-change to auto-reload when the config is saved from the Web UI
   ```
4. **Open the dashboard (optional):**
   ```bash
   export LOGTRIAGE_CONFIG=./config.yaml
   logtriage-webui
   ```
   Visit `http://127.0.0.1:8090` to triage **issues** (deduplicated findings with AI summaries), review modules, edit the config, or tune regexes.
   If you already have findings from a previous version, build issues for them once with:
   ```bash
   logtriage --config ./config.yaml --backfill-issues
   ```
5. **Start RAG service (optional, for improved performance):**
   ```bash
   logtriage-rag --config ./config.yaml
   ```
   The RAG service runs on port 8091 and provides documentation retrieval capabilities. When running, WebUI and CLI will automatically use it for better performance.

## Documentation

See here the [full documentation](https://giovi321.github.io/log-triage/)

- **Architecture:** https://giovi321.github.io/log-triage/architecture/
- **Security considerations:** https://giovi321.github.io/log-triage/security/
- **RAG:**
  - https://giovi321.github.io/log-triage/RAG/
  - https://giovi321.github.io/log-triage/RAG-QuickStart/
  - https://giovi321.github.io/log-triage/RAG-Service/

> **Security note:** The Web UI is intended for trusted, internal networks. It now ships with CSRF protection on form posts, session expiry, an IP allowlist, and optional reverse-proxy forward-auth (e.g. Authentik) — but it is still **not** hardened for direct public-internet exposure. Run it behind TLS + network controls and set a strong `webui.secret_key`. See the [security documentation](https://giovi321.github.io/log-triage/security/) for the full assessment.

## Features

- YAML configuration for both pipelines and modules
- Multiple pipelines, selected by name or filename regex
- Grouping strategies (each in its own module):
  - whole-file grouping
  - marker-based grouping (for example per rsnapshot run)
- Classifiers (each in its own module):
  - generic regex counter
  - rsnapshot-specific heuristic
- Per-pipeline ignore rules (`ignore_regexes`) to drop known-noise lines before counting
- Severity levels:
  - WARNING, ERROR, CRITICAL
- Batch mode (scan file or directory once)
- Follow mode (continuous tail of a single log file), rotation-aware (`tail -F` style)
- Optional config change detection for follow-mode modules to auto-reload after saving via the Web UI (`--reload-on-change`)
- Optional LLM payload generation with conservative gating and per-pipeline prompt templates
- Multiple LLM provider support:
  - **OpenAI** and any OpenAI-compatible API (Azure OpenAI, LM Studio, LiteLLM, etc.)
  - **Anthropic Claude** (native Messages API: claude-sonnet-4-6, claude-opus-4-8, claude-haiku-4-5, etc., with prompt caching)
  - **Ollama** (native `/api/chat`, for local models — no API key needed)
  - Provider auto-detection from `api_base`: `anthropic.com` → Claude, a `:11434` host → Ollama, otherwise OpenAI-compatible
- Per-module options for:
  - context lines included ahead of each finding (`llm.context_prefix_lines`)
  - alert hooks (`alerts.mqtt`, `alerts.webhook`)
- Optional SQL database integration for storing per-finding records (SQLite or Postgres)
- **Signature-based de-duplication** that collapses recurring findings into **issues** with occurrence counts, first/last-seen, severity escalation, and priority scoring
- **Per-issue LLM analysis with caching** (analyze once per signature; Anthropic **prompt caching** for the documentation context) — drastically cheaper than per-finding enrichment
- **Background enrichment worker** (in the Web UI by default, or standalone `logtriage-worker`) that keeps issue summaries fresh
- **Live updates over Server-Sent Events** (no client polling) and a Prometheus **`/metrics`** endpoint
- Web UI (FastAPI), a mission-control interface with light & dark themes (light default, one-click toggle), to:
  - log in with username/password (bcrypt), or **Authentik / reverse-proxy forward-auth** (trusted-header)
  - **Triage queue**: prioritized issues with sparklines and cached AI summaries; per-issue detail with timeline, citations, and acknowledge/resolve/mute/false-positive actions
  - view modules and per-module stats (last severity, 24h error/warning counts, RAG status)
  - edit `config.yaml` via **structured forms** (with a raw-YAML "Advanced" tab), atomic writes with backup
  - experiment with regexes (regex lab) and save them to classifiers

## License
This project is licensed under the GNU GPL v3.0 license. See [LICENSE](LICENSE) for details.
