[![Version](https://img.shields.io/badge/version-v0.1.0-blue.svg)](https://github.com/giovi321/log-triage)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/github/license/giovi321/log-triage)](https://github.com/giovi321/log-triage/blob/main/LICENSE)

<img width="317" height="55" alt="log-triage" src="https://github.com/user-attachments/assets/1da91cb3-fe19-408a-80c8-e894f57542a8" />

`log-triage` is a Python tool that sits between your log collector (for example Fluent Bit) and an LLM. It filters noisy logs, detects problems, and gives you a dashboard and API-ready payloads so you can triage faster.

## Overview

### Key concepts
- **Pipelines:** Reusable recipes that define how to group log lines, which regexes to ignore or count, and which prompt template to use for LLM payloads.
- **Modules:** Runtime bindings that attach a pipeline to a file path and decide whether to scan once (batch) or tail continuously (follow) with rotation awareness.
- **Findings:** Structured outputs for each grouped chunk, including severity (WARNING/ERROR/CRITICAL), counts, and optional LLM payloads.
- **Baseline:** Optional anomaly detection that compares current counts to a rolling window and can bump severity when spikes occur.
- **Addressed & false positives:** Workflow flags in the dashboard; marking a false positive also writes an ignore regex back to the pipeline to prevent repeats.

### How it works (process)
`log-triage` watches your logs, passes them through a configured pipeline, and surfaces only the important pieces:

1. **Collect:** Point a module at a log file (or directory) to read entries once or continuously with rotation handling.
2. **Group:** Apply the pipeline's grouping strategy (whole-file or marker-based) to carve the stream into logical chunks.
3. **Classify:** Count warnings and errors with regex rules, ignore known-noise patterns, and assign a severity.
4. **Baseline:** Optionally compare the run to historical averages and flag anomalies, optionally elevating severity.
5. **Enrich:** Generate an LLM payload per finding using your prompt template and context lines.
6. **Deliver:** Print findings, send alerts (webhook/MQTT), store them for the Web UI, and use the dashboard to reclassify, mark false positives, or update severity.

### Getting started
1. **Install the package:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   # Install core plus optional Web UI + MQTT extras
   pip install ".[webui,alerts]"
   ```
2. **Configure:** Copy `config.yaml` and edit pipelines/modules to point at your log files.
3. **Run a module:**
   ```bash
   logtriage --config ./config.yaml run --module <module-name>
   ```
4. **Open the dashboard (optional):**
   ```bash
   export LOGTRIAGE_CONFIG=./config.yaml
   logtriage-webui
   ```
   Visit `http://127.0.0.1:8090` to review findings, adjust severity, or mark false positives.

## Documentation

See here the [full documentation](https://giovi321.github.io/log-triage/)

> **Security note:** The Web UI is not designed to be exposed to the public internet due to missing CSRF protections, weak sess
ion cookies, and other controls. Run it only on trusted networks and see the documentation for the full disclaimer.

## How does it work

See the “How it works (at a glance)” section above for a quick process overview, or read the full documentation for in-depth
details on pipelines, modules, and the Web UI.

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
- Per-module options for:
  - context lines included ahead of each finding (`llm.context_prefix_lines`)
  - alert hooks (`alerts.mqtt`, `alerts.webhook`)
  - baseline / anomaly detection (`baseline` block)
- Optional SQL database integration for storing per-finding records (SQLite or Postgres)
- Web UI (FastAPI) to:
  - log in with username/password (bcrypt)
  - view modules and per-module stats (last severity, 24h error/warning counts, etc.)
  - inspect and edit `config.yaml` (atomic writes, with backup)
  - experiment with regexes (regex lab) and save them to classifiers
  - run on a dark-mode layout

## License
This project is licensed under the GNU GPL v3.0 license. See [LICENSE](LICENSE) for details.
