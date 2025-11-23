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

Pipelines define how to group and classify logs; modules define how and when to run them.

The actual LLM call is out of scope. `log-triage` only decides what is interesting and produces payload files you can feed to a local or remote LLM.

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
- Debug mode to inspect chunk boundaries (`--inspect-chunks`)

## Install

Clone the repo and install dependencies in a virtualenv:

```bash
git clone https://github.com/youruser/log-triage.git
cd log-triage

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pyyaml
# optional for MQTT alerts
pip install paho-mqtt
```

Package layout:

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
config.example.yaml
```

Run the CLI:

```bash
python -m logtriage --help
```

## Configuration

Copy the example config and edit it:

```bash
cp config.example.yaml config.yaml
```

The config has three main parts:

- `defaults`: global pipeline defaults
- `pipelines`: reusable rules for grouping and classification
- `modules`: concrete tasks, one per software you want to analyze

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
      ignore_regexes: []         # optional: patterns to ignore when counting
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
    llm_payload_mode: "errors_only"
    only_last_chunk: true
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
```

## Pipelines

A pipeline describes *how* to process logs, independent of any concrete file.

### Grouping

Grouping strategies are implemented under `logtriage/grouping/` and selected by `grouping.type`:

- `whole_file`: entire file (or batch of lines) is one chunk.
- `marker`: new chunk starts when `start_regex` matches. Chunk ends when `end_regex` matches (if provided) or when the next `start_regex` appears.

### Classifiers

Classification strategies are in `logtriage/classifiers/` and selected by `classifier.type`.

- `regex_counter`:
  - counts matches of `error_regexes` and `warning_regexes` after removing lines matching `ignore_regexes`.
  - `ERROR` if any error matches, `WARNING` if any warnings, `OK` if non-empty, else `UNKNOWN`.

- `rsnapshot_basic`:
  - applies `error_regexes`, `warning_regexes` and some hard-coded rsnapshot patterns.
  - checks for `exit code = N` (non-zero ⇒ `ERROR`).
  - looks for `completed successfully`.
  - uses `ignore_regexes` only for counting; full text is used to detect exit code and completion.

## Modules

Modules describe actual tasks, one per software or log source. They combine:

- a path (file or directory)
- a pipeline
- the mode (batch or follow)
- printing and LLM options
- optional severity→exit-code mapping
- optional alerts
- optional baseline/anomaly detection

## LLM prompt templates

Each pipeline can point to a prompt template file with `llm.prompt_template`. The file is read once and formatted with:

- `{severity}`
- `{reason}`
- `{file_path}`
- `{pipeline}`
- `{error_count}`
- `{warning_count}`
- `{line_count}`

Example `prompts/rsnapshot.txt`:

```text
You are an assistant analyzing rsnapshot backup runs.

Pipeline: {pipeline}
File: {file_path}
Rule-based severity: {severity}
Rule-based reason: {reason}
Error lines: {error_count}
Warning lines: {warning_count}
Lines in this chunk: {line_count}

Return a single JSON object with keys: severity, reason, key_lines, action_items.
Do not include any text before or after the JSON.
```

If no template is provided, a built-in default prompt is used.

## LLM gating and payloads

For each pipeline, LLM gating is controlled by the `llm` section.

For each chunk in a module, `needs_llm` is set based on:

- `llm.enabled`
- `llm.min_severity`
- `llm.max_chunk_lines`
- chunk severity and length (after baseline adjustments, if any)

If a module has `emit_llm_payloads_dir` set, each chunk with `needs_llm: true` produces a payload file with:

- the prompt header (from template or default)
- the log chunk between `BEGIN/END` markers

Payload content respects `llm_payload_mode`:

- `"full"`: all lines in the chunk.
- `"errors_only"`: only lines containing typical error/warning tokens, with fallback to full chunk when nothing matches.

## Baseline / anomaly detection

If a module has a `baseline` block with `enabled: true`, log-triage maintains a rolling history of error/warning counts in `baseline.state_file`. For each new chunk:

- compute average error and warning counts over the last `window` entries.
- if current errors >= `error_multiplier` × average, mark an error anomaly.
- if current warnings >= `warning_multiplier` × average, mark a warning anomaly.
- on anomaly, prefix the reason text and raise severity to at least `severity_on_anomaly`.

The updated severity is then used for printing, alerts, exit code mapping, and LLM gating.

## Alerts

If `alerts.webhook` is configured for a module, chunks with severity ≥ `min_severity` send a JSON payload to the given URL with:

- module, file_path, pipeline
- severity, reason
- error_count, warning_count, line_count

If `alerts.mqtt` is configured, the same payload is published as a JSON string to the configured MQTT topic. `paho-mqtt` is optional; if not installed, MQTT alerts are skipped with a warning.

## Follow mode and log rotation

Follow mode works like a basic `tail -F`:

- On first open:
  - if `stream.from_beginning: true`, read from the beginning;
  - otherwise start at the end of the file.
- On rotation (inode changes) or truncation (file shrinks), the file is reopened and reading resumes from the end of the new file.
- Lines are buffered and emitted as a chunk after a pause of `interval` seconds with no new data.

This lets you run long-lived modules on logs managed by logrotate or other rotation tools.

## Usage

Run all enabled modules defined in `config.yaml`:

```bash
python -m logtriage --config config.yaml
```

Run a single module (useful for one systemd service per module):

```bash
python -m logtriage --config config.yaml --module homeassistant_follow
```

Inspect how a batch module splits logs into chunks (debugging grouping rules):

```bash
python -m logtriage --config config.yaml --module rsnapshot_daily --inspect-chunks
```

Exit codes for automation (only when running a single batch module with `exit_code_by_severity` set):

```bash
python -m logtriage --config config.yaml --module rsnapshot_daily
echo $?   # reflects highest severity based on exit_code_by_severity mapping
```

Notes:

- `--inspect-chunks` is only supported for batch modules (`mode: "batch"`).
- Modules with `mode: "batch"` run once on their configured path and exit.
- Modules with `mode: "follow"` tail their log file indefinitely until interrupted.

To disable a module in the default run, set `enabled: false` in the `modules` section.
