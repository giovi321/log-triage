# Configuration

`log-triage` is a Python toolkit that sits between your log collector and an LLM. It reads log files, applies configurable rules to group and classify entries, and emits structured findings and payloads you can forward to a model. The CLI and Web UI share the same YAML configuration so you can run batch jobs, follow live streams, or explore results in a dashboard.

## Core concepts

- **Pipelines:** Reusable processing recipes. Each pipeline defines how to group log lines, how to classify them, which regexes to ignore, and which prompts to use.
- **Modules:** Runtime bindings that attach a pipeline to a specific file path, execution mode (batch/follow), and output options.
- **Findings:** Structured results produced by a classifier. They include severity, reason, counts, context, and optional LLM payloads for downstream tooling.
- **Severity:** One of `WARNING`, `ERROR`, or `CRITICAL`. Severity can be escalated by anomaly detection or manually adjusted in the Web UI.
- **Regex filters:**
  - `ignore_regexes` drop lines before counting.
  - `warning_regexes` and `error_regexes` count matched lines to set severity.
  - The regex lab in the UI helps experiment with these patterns.
- **Baseline:** A rolling window of historical counts for a module. When current counts exceed configured multipliers, the finding is marked anomalous and its severity can be bumped.
- **Addressed & false positives:** Addressed findings track items you reviewed. Marking a finding as a false positive also adds the pattern to the pipeline's ignore list, preventing repeats.

Keep these terms in mind while reading the configuration sections below.

## File structure at a glance

```yaml
pipelines: []
modules: []
llm:
  default_provider: null
  providers: {}
database:
alerts: {}
baseline: {}
```

Each top-level key controls a specific area of the system. The example below expands all of them in context.

## Example configuration

```yaml
pipelines:
  - name: homeassistant
    grouping: whole_file
    classifier: regex_counter
    ignore_regexes:
      - "^DEBUG"
    warning_regexes:
      - "warning|timeout"
    error_regexes:
      - "(error|failed)"
    prompt_template: './prompts/homeassistant.txt'

modules:
  - name: homeassistant_follow
    enabled: true
    path: '/var/log/fluent-bit/homeassistant.log'
    mode: 'follow'
    pipeline: 'homeassistant'
    output_format: 'text'
    min_print_severity: 'WARNING'
    stale_after_minutes: 60
    from_beginning: false
    interval: 5
    llm:
      enabled: true
      provider: 'openai'
      emit_llm_payloads_dir: './ha_llm_payloads'
      context_prefix_lines: 2
    alerts:
      webhook:
        enabled: true
        url: 'https://hooks.example.local/logs'

llm:
  default_provider: openai
  providers:
    openai:
      base_url: 'https://api.openai.com'
      model: 'gpt-4o-mini'
      api_key_env: 'OPENAI_API_KEY'

database:
  url: 'sqlite:///./logtriage.db'
  echo: false
```

> Tip: The Web UI config editor provides inline hints and validation so you can tweak values safely before saving.

## Pipelines

Pipelines describe how to interpret a log stream.

```yaml
pipelines:
  - name: homeassistant
    grouping: whole_file  # or marker_based
    classifier: regex_counter  # or rsnapshot_heuristic
    ignore_regexes:
      - "^DEBUG"
    warning_regexes:
      - "warning|timeout"
    error_regexes:
      - "(error|failed)"
```

- **grouping** controls how log lines are chunked before classification.
- **classifier** picks the rule set used to score each chunk.
- **ignore_regexes** lets you drop known-noise lines before counting errors and warnings.
- **warning_regexes / error_regexes** count matches separately so the classifier can choose the highest severity observed.

## Modules

Modules connect pipelines to real log files and decide when to run.

```yaml
modules:
  - name: homeassistant_follow
    enabled: true
    path: '/var/log/fluent-bit/homeassistant.log'
    mode: 'follow'           # 'batch' or 'follow'
    pipeline: 'homeassistant'
    output_format: 'text'    # 'text' or 'json'
    min_print_severity: 'WARNING'
    stale_after_minutes: 60  # used by the Web UI for activity indicators
```

### Follow mode options

- `from_beginning`: whether to start at the start of the file or tail new lines only.
- `interval`: how frequently to poll for new lines and detect rotations.
- `reload_on_change`: when set on the CLI, reloads the config if `config.yaml` changes.

### LLM options per module

```yaml
modules:
  - name: homeassistant_follow
    llm:
      enabled: true
      provider: 'openai'
      prompt_template: './prompts/homeassistant.txt'
      emit_llm_payloads_dir: './ha_llm_payloads'
      context_prefix_lines: 2
```

Each module can choose whether to generate payloads. The prompt template is formatted with placeholders such as `{pipeline}`, `{file_path}`, `{severity}`, `{reason}`, `{error_count}`, `{warning_count}`, and `{line_count}`.

### Alerts

Modules support multiple alert channels:

- **Webhook:**

  ```yaml
  alerts:
    webhook:
      enabled: true
      url: 'https://hooks.example.local/logs'
      headers:
        Authorization: 'Bearer <token>'
  ```

- **MQTT:**

  ```yaml
  alerts:
    mqtt:
      enabled: true
      host: 'mqtt.example.local'
      port: 1883
      topic: 'alerts/logs'
      tls: false
  ```

### Baseline and anomaly detection

When `baseline.enabled` is true, the CLI tracks historical error and warning counts and labels runs as anomalies if they exceed configured multipliers.

```yaml
baseline:
  enabled: true
  state_file: './state/homeassistant.json'
  window: 10
  error_multiplier: 2.0
  warning_multiplier: 2.0
  severity_on_anomaly: 'CRITICAL'
```

- The baseline window stores the last `window` runs and computes moving averages for warnings and errors.
- When current counts exceed the average times the configured multipliers, the run is flagged as anomalous.
- If `severity_on_anomaly` is set, the finding severity is elevated (for example to `CRITICAL`). Otherwise, the original severity is preserved but the anomaly flag is stored.
- Baseline state is persisted to the `state_file` per module. The Web UI offers an editor for these JSON files.

## LLM providers

Providers are defined globally and referenced by name from modules.

```yaml
llm:
  default_provider: openai
  providers:
    openai:
      base_url: 'https://api.openai.com'
      model: 'gpt-4o-mini'
      api_key_env: 'OPENAI_API_KEY'
```

When only one provider is defined, modules inherit it automatically. The CLI reads the API key from the named environment variable.

## Database

To persist findings for the Web UI, configure a database connection:

```yaml
database:
  url: 'sqlite:///./logtriage.db'
  echo: false
```

SQLite and Postgres URLs are both supported. When omitted, the Web UI stores data in memory and only reflects the current session.
