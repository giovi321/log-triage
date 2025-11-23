# log-triage

`log-triage` is a Python tool that sits between your log collector (for example Fluent Bit) and an LLM.

It works per *module* defined in a YAML config file. Each module describes:

- which log file or directory to read
- whether to run in batch mode or follow mode
- which pipeline (rules) to apply
- when to prepare LLM payloads
- how to print the summary
- optionally, which exit code to use for automation

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
- Follow mode (continuous tail of a single log file), rotation-aware (tail -F style)
- Optional LLM payload generation with conservative gating
- Per-module options for:
  - full-chunk vs error-only LLM payloads (`llm_payload_mode`)
  - analyze whole history vs only last chunk per file (`only_last_chunk`)
  - mapping highest severity to process exit code (`exit_code_by_severity`)
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
      filename_regex: "rsnapshot.*\.log"
    grouping:
      type: "marker"
      start_regex: "^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\].*rsnapshot .*: started$"
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

  - name: homeassistant
    match:
      filename_regex: "homeassistant.*\.log"
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
  # rsnapshot: analyze only the last run in the file, with exit codes
  - name: rsnapshot_daily
    enabled: true
    path: "/var/log/rsnapshot.log"
    mode: "batch"              # process file or directory once
    pipeline: "rsnapshot"
    output_format: "json"      # "text" or "json"
    min_print_severity: "INFO" # filter for console output
    emit_llm_payloads_dir: "./rsnapshot_payloads"
    llm_payload_mode: "errors_only"
    only_last_chunk: true      # consider only the last rsnapshot run in the file
    exit_code_by_severity:
      OK: 0
      INFO: 0
      WARNING: 1
      ERROR: 2
      CRITICAL: 3

  # Home Assistant: continuous follow + error-only payloads
  - name: homeassistant_follow
    enabled: true
    path: "/var/log/fluent-bit/homeassistant.log"
    mode: "follow"               # continuous tail
    pipeline: "homeassistant"
    output_format: "text"
    min_print_severity: "WARNING"
    emit_llm_payloads_dir: "./ha_llm_payloads"
    llm_payload_mode: "errors_only"   # only error/warning-like lines in payload
    only_last_chunk: false
    stream:
      from_beginning: false      # like tail -F: start at end
      interval: 1.0              # seconds between polls

  # Generic batch over a directory, full payloads
  - name: all_logs_batch
    enabled: false
    path: "/var/log/fluent-bit"
    mode: "batch"
    # no pipeline: selection based on pipelines[*].match.filename_regex
    output_format: "json"
    min_print_severity: "INFO"
    emit_llm_payloads_dir: "./llm_payloads"
    llm_payload_mode: "full"
    only_last_chunk: false
```

Notes:

- `pipelines[*].name` is referenced by `modules[*].pipeline`.
- `match.filename_regex` is used when scanning directories or when a module does not specify a pipeline explicitly.
- `modules[*].mode` controls whether the module uses batch or follow behavior.

## Pipelines

A pipeline describes *how* to process logs, independent of any concrete file.

### Grouping

Grouping strategies are implemented under `logtriage/grouping/` and selected by `grouping.type`:

- `whole_file`

  Entire file (or batch of lines) is one chunk.

- `marker`

  New chunk starts when `start_regex` matches. Chunk ends when `end_regex` matches (if provided) or when the next `start_regex` appears.

  Example for rsnapshot logs:

  ```yaml
  grouping:
    type: "marker"
    start_regex: "^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\].*rsnapshot .*: started$"
    end_regex: ""
  ```

### Classifiers

Classification strategies are in `logtriage/classifiers/` and selected by `classifier.type`.

- `regex_counter`

  ```yaml
  classifier:
    type: "regex_counter"
    error_regexes:
      - "\\berror\\b"
      - "\\bfailed\\b"
    warning_regexes:
      - "\\bwarn\\b"
    ignore_regexes:
      - "some known benign message"
  ```

  Logic:

  - Lines matching `ignore_regexes` are dropped before counting.
  - If any error regex matches: `ERROR`
  - Else if any warning regex matches: `WARNING`
  - Else if chunk is not empty: `OK`
  - Else: `UNKNOWN`

- `rsnapshot_basic`

  Adds backup-specific patterns on top of configured regexes:

  - `rsync error`
  - `ERROR:`
  - `FATAL`
  - `Backup FAILED`
  - `WARNING`
  - `partial transfer`

  It also tries to read `exit code = N` from the chunk and looks for
  `completed successfully`. Ignore rules are applied to counting, but full
  text is used for exit code and success detection.

## Modules

Modules describe actual tasks, one per software or log source. They combine:

- a path (file or directory)
- a pipeline
- the mode (batch or follow)
- printing and LLM options
- optional severity→exit-code mapping

Fields:

- `name`: identifier used with `--module`.
- `enabled`: if false, module is ignored unless explicitly selected.
- `path`: file or directory to analyze.
- `mode`:
  - `batch`: run once on current contents.
  - `follow`: tail a single file continuously.
- `pipeline`: optional pipeline name. If omitted and `path` is a directory, pipelines are selected by `match.filename_regex`.
- `output_format`: `text` or `json` for console output.
- `min_print_severity`: minimum severity to print to console.
- `emit_llm_payloads_dir`: directory where LLM payloads are written for chunks that pass LLM gating. Empty or missing means no payloads.
- `llm_payload_mode`: `"full"` (default) to include the entire chunk in the payload, or `"errors_only"` to include only lines that look like errors/warnings.
- `only_last_chunk`: if `true`, only the last chunk per file is kept for summary and payloads (useful for logs grouped by runs, such as backup jobs).
- `exit_code_by_severity`: optional mapping `{"OK": 0, "ERROR": 2, ...}`. If set and you run exactly one batch module, the process will exit with the code corresponding to the highest severity in that module.
- `stream` (only relevant for `mode: follow`):
  - `from_beginning`: if true, start reading from byte 0 instead of end of file.
  - `interval`: polling interval in seconds.

## LLM gating and payloads

For each pipeline, LLM gating is controlled by the `llm` section:

```yaml
llm:
  enabled: true
  min_severity: WARNING
  max_chunk_lines: 300
```

For each chunk in a module:

- if LLM is disabled for the pipeline: `needs_llm` is false
- if chunk severity is below `min_severity`: `needs_llm` is false
- if chunk is empty or exceeds `max_chunk_lines`: `needs_llm` is false
- otherwise `needs_llm` is true

When a module has `emit_llm_payloads_dir` set, each chunk with `needs_llm: true` produces a payload file containing:

- a short instruction header
- pipeline and rule-based severity context
- the log lines between markers (possibly filtered)

The header asks the LLM to respond with a strict JSON object containing:

- `severity`
- `reason`
- `key_lines`
- `action_items`

Payload content is controlled by `llm_payload_mode`:

- `"full"`: payload includes all lines in the chunk.
- `"errors_only"`: payload is filtered to lines containing typical error/warning tokens (`error`, `failed`, `failure`, `exception`, `traceback`, `fatal`, `critical`, `warning`, `warn`). If no lines match, the full chunk is used as a fallback.

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

This prints, for each chunk:

- file path
- chunk index
- line count
- first and last line (truncated)

Exit codes for automation (only when running a single batch module):

```bash
python -m logtriage --config config.yaml --module rsnapshot_daily
echo $?   # reflects highest severity based on exit_code_by_severity mapping
```

Notes:

- `--inspect-chunks` is only supported for batch modules (`mode: "batch"`).
- When `--inspect-chunks` is used, triage/LLM logic is skipped and only chunk boundaries are shown.
- Modules with `mode: "batch"` run once on their configured path and exit.
- Modules with `mode: "follow"` tail their log file indefinitely until interrupted.

To disable a module in the default run, set `enabled: false` in the `modules` section.

## Extending

To add a new grouping strategy:

- create a new file under `logtriage/grouping/`,
- implement a function that takes `lines: List[str]` and returns `List[List[str]]`,
- update `grouping/__init__.py` to dispatch on a new `grouping.type`.

To add a new classifier:

- create a new file under `logtriage/classifiers/`,
- implement a function returning `(Severity, reason, error_count, warning_count)`,
- update `classifiers/__init__.py` to dispatch on a new `classifier.type`.

Most adjustments for new software can be done by editing `config.yaml` only.
