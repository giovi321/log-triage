# log-triage

`log-triage` is a Python tool that sits between your log collector (for example Fluent Bit) and an LLM.

It works per *module* defined in a YAML config file. Each module describes:

- which log file or directory to read
- whether to run in batch mode or follow mode
- which pipeline (rules) to apply
- when to prepare LLM payloads
- how to print the summary
- optionally, which exit code to use for automation
- optionally, how to send alerts (webhook / MQTT)
- optionally, how to use a baseline to detect anomalies
- optionally, how to store summaries in a database and show them in a Web UI

Pipelines define how to group and classify logs; modules define how and when to run them.

The actual LLM call is out of scope. `log-triage` only decides what is interesting and produces payload files you can feed to a local or remote LLM. A companion Web UI gives you a dark-mode dashboard, config editor, and regex lab on top.

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
  - UNKNOWN, OK, INFO, WARNING, ERROR, CRITICAL
- Batch mode (scan file or directory once)
- Follow mode (continuous tail of a single log file), rotation-aware (`tail -F` style)
- Optional LLM payload generation with conservative gating and per-pipeline prompt templates
- Per-module options for:
  - full-chunk vs error-only LLM payloads (`llm_payload_mode`)
  - analyze whole history vs only last chunk per file (`only_last_chunk`)
  - mapping highest severity to process exit code (`exit_code_by_severity`)
  - alert hooks (`alerts.mqtt`, `alerts.webhook`)
  - baseline / anomaly detection (`baseline` block)
- Optional SQL database integration for storing chunk summaries (SQLite or Postgres)
- Web UI (FastAPI) to:
  - log in with username/password (bcrypt)
  - view modules and per-module stats (last severity, 24h error/warning counts, etc.)
  - inspect and edit `config.yaml` (atomic writes, with backup)
  - experiment with regexes (regex lab) and save them to classifiers
  - run on a dark-mode layout

## Install

Clone the repo and install dependencies in a virtualenv:

```bash
git clone https://github.com/giovi321/log-triage.git
cd log-triage

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# core CLI
pip install pyyaml

# optional MQTT alerts
pip install paho-mqtt

# Web UI + DB
pip install fastapi uvicorn jinja2 python-multipart passlib[bcrypt] sqlalchemy
```

Package layout (relevant parts):

```text
logtriage/
  __init__.py
  __main__.py
  cli.py
  config.py
  models.py
  utils.py
  stream.py
  engine.py
  grouping/
    __init__.py
    whole_file.py
    marker.py
  classifiers/
    __init__.py
    regex_counter.py
    rsnapshot_basic.py
  alerts.py
  baseline.py
  webui/
    __init__.py
    __main__.py
    app.py
    auth.py
    config.py
    db.py
    templates/
      base.html
      login.html
      dashboard.html
      config.html
      config_edit.html
      regex.html
config.example.yaml
```

Run the CLI:

```bash
python -m logtriage --help
```

Run the Web UI (reads the same `config.yaml`):

```bash
export LOGTRIAGE_CONFIG=/path/to/config.yaml  # optional, default ./config.yaml
python -m logtriage.webui
```

Then open `http://127.0.0.1:8090/login` (or whatever host/port you configured).

## Configuration

Copy the example config and edit it:

```bash
cp config.example.yaml config.yaml
```

The config has five main parts:

- `defaults`: global pipeline defaults
- `pipelines`: reusable rules for grouping and classification
- `modules`: concrete tasks, one per software you want to analyze
- `database`: DB connection and retention
- `webui`: Web UI settings and admin users

Example:

```yaml
defaults:
  llm_enabled: false
  llm_min_severity: WARNING
  max_chunk_lines: 500

pipelines:
  - name: rsnapshot
    match:
      filename_regex: "rsnapshot.*\\.log"
    grouping:
      type: "marker"
      start_regex: "^\\[\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\].*rsnapshot .*: started$"
      end_regex: ""
    classifier:
      type: "rsnapshot_basic"
      error_regexes:
        - "ERROR:"
        - "rsync error"
      warning_regexes:
        - "WARNING"
      ignore_regexes: []
    llm:
      enabled: true
      min_severity: ERROR
      max_chunk_lines: 5000
      prompt_template: "./prompts/rsnapshot.txt"  # optional custom prompt

  - name: homeassistant
    match:
      filename_regex: "homeassistant.*\\.log"
    grouping:
      type: "whole_file"
    classifier:
      type: "regex_counter"
      error_regexes:
        - "\\berror\\b"
        - "Traceback"
      warning_regexes:
        - "\\bwarning\\b"
        - "\\bfailed to\\b"
      ignore_regexes:
        - "Some noisy integration .* does not support.*"
    llm:
      enabled: true
      min_severity: WARNING
      max_chunk_lines: 1000
      prompt_template: "./prompts/homeassistant.txt"

  - name: generic_default
    match:
      filename_regex: ".*"
    grouping:
      type: "whole_file"
    classifier:
      type: "regex_counter"
      error_regexes:
        - "\\berror\\b"
        - "\\bfailed\\b"
        - "exception"
      warning_regexes:
        - "\\bwarn\\b"
        - "\\bdegraded\\b"
      ignore_regexes: []
    llm:
      enabled: true
      min_severity: WARNING
      max_chunk_lines: 300

modules:
  - name: rsnapshot_daily
    enabled: true
    path: "/var/log/rsnapshot.log"
    mode: "batch"
    pipeline: "rsnapshot"
    output_format: "json"
    min_print_severity: "INFO"
    emit_llm_payloads_dir: "./rsnapshot_payloads"
    llm_payload_mode: "errors_only"   # "full" or "errors_only"
    only_last_chunk: true             # only last rsnapshot run
    exit_code_by_severity:
      OK: 0
      INFO: 0
      WARNING: 1
      ERROR: 2
      CRITICAL: 3
    alerts:
      webhook:
        enabled: true
        url: "https://example/rsnapshot-hook"
        method: "POST"
        min_severity: "ERROR"
        headers:
          X-Source: "logtriage"
      mqtt:
        enabled: false
        host: "localhost"
        port: 1883
        topic: "logtriage/rsnapshot"
        username: ""
        password: ""
        min_severity: "WARNING"
    baseline:
      enabled: true
      state_file: "/var/lib/logtriage/rsnapshot_baseline.json"
      window: 20
      error_multiplier: 3.0
      warning_multiplier: 3.0
      severity_on_anomaly: "ERROR"

  - name: homeassistant_follow
    enabled: true
    path: "/var/log/fluent-bit/homeassistant.log"
    mode: "follow"
    pipeline: "homeassistant"
    output_format: "text"
    min_print_severity: "WARNING"
    emit_llm_payloads_dir: "./ha_llm_payloads"
    llm_payload_mode: "errors_only"
    only_last_chunk: false
    alerts:
      mqtt:
        enabled: true
        host: "localhost"
        port: 1883
        topic: "logtriage/homeassistant"
        min_severity: "ERROR"
    baseline:
      enabled: false
    stream:
      from_beginning: false
      interval: 1.0

  - name: all_logs_batch
    enabled: false
    path: "/var/log/fluent-bit"
    mode: "batch"
    output_format: "json"
    min_print_severity: "INFO"
    emit_llm_payloads_dir: "./llm_payloads"
    llm_payload_mode: "full"
    only_last_chunk: false

database:
  url: "sqlite:////var/lib/logtriage/logtriage.db"
  retention_days: 30

webui:
  enabled: true
  host: "127.0.0.1"
  port: 8090
  base_path: "/"
  secret_key: "CHANGE_THIS_TO_A_LONG_RANDOM_STRING"
  session_cookie_name: "logtriage_session"
  dark_mode_default: true
  csrf_enabled: true     # reserved; not fully wired yet
  allowed_ips: ["127.0.0.1"]
  admin_users:
    - username: "admin"
      password_hash: "bcrypt:$2b$12$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Web UI notes:

- Web UI reads this same `config.yaml` using `LOGTRIAGE_CONFIG` or default `./config.yaml`.
- Editing via `/config/edit` performs YAML validation and atomic update with `.bak` backup.
- Regex lab at `/regex` lets you:
  - pick a module, tail its log file
  - generate a regex from a selected line
  - test a regex against the sample
  - save the regex into the matching pipeline classifier.

## Usage

Run all enabled modules defined in `config.yaml`:

```bash
python -m logtriage --config config.yaml
```

Run a single module (useful for one systemd service per module):

```bash
python -m logtriage --config config.yaml --module homeassistant_follow
```

Inspect chunk boundaries for a batch module:

```bash
python -m logtriage --config config.yaml --module rsnapshot_daily --inspect-chunks
```

Run the Web UI:

```bash
export LOGTRIAGE_CONFIG=/path/to/config.yaml
python -m logtriage.webui
```
