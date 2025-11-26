# log-triage

`log-triage` is a Python toolkit that sits between your log collector and an LLM. It reads log files, applies configurable rules to group and classify entries, and emits structured findings and payloads you can forward to a model. The CLI and Web UI share the same YAML configuration so you can run batch jobs, follow live streams, or explore results in a dashboard.

## Why use log-triage?

- **Pluggable pipelines.** Define multiple pipelines, each with its own grouping strategies and classifiers.
- **Flexible modules.** Decide whether a module runs once (batch) or tails a file with rotation awareness (follow).
- **Ignore noisy lines.** Drop known noise via `ignore_regexes` before counting errors and warnings.
- **Severity-aware.** Findings are labeled `WARNING`, `ERROR`, or `CRITICAL`, and can be escalated when anomalies are detected.
- **LLM-ready payloads.** Generate concise payloads with prompt templates, without coupling to a specific provider.
- **Web dashboard.** Explore findings, edit configuration, and tune regexes in a dark-mode UI.
- **Alerts and storage.** Send webhook or MQTT alerts, and persist findings in SQLite or Postgres for the Web UI.

## What you need

- Python **3.10+**
- Access to the log files you want to analyze, or a collector such as Fluent Bit writing to disk
- Optional: credentials for your preferred LLM provider and database

## Installation

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install '.[webui,mqtt]'
```

For a manual, explicit install, you can install the core packages and add the optional extras individually:

```bash
pip install pyyaml fastapi uvicorn jinja2 python-multipart passlib[bcrypt] sqlalchemy itsdangerous paho-mqtt
pip install --upgrade --force-reinstall "bcrypt>=4.0,<4.1"
```

## Next steps

- Read the [Getting started](getting-started.md) guide to run your first module.
- Explore [Configuration](configuration.md) to see how pipelines, modules, and prompts fit together.
- Visit [Web UI](web-ui.md) to learn how to browse findings and edit the config from the dashboard.
- Check out [Deployment](deployment.md) for GitHub Pages and MkDocs details.
