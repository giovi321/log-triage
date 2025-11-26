[![Version](https://img.shields.io/badge/version-v0.1.0-blue.svg)](https://github.com/giovi321/log-triage)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/github/license/giovi321/log-triage)](https://github.com/giovi321/log-triage/blob/main/LICENSE)

<img width="317" height="55" alt="log-triage" src="https://github.com/user-attachments/assets/1da91cb3-fe19-408a-80c8-e894f57542a8" />

`log-triage` is a Python tool that sits between your log collector (for example Fluent Bit) and an LLM.
It does three things:
- Filter and sort log lines by error severity
- Send them to a LLM to get an opinion on the error
- Provide a web user interace to manage all findings and configure the software

## Documentation

See here the [full documentation](https://giovi321.github.io/log-triage/)

## How does it work

[WIP]

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
