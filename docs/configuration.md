# Configuration

The CLI and Web UI share a single YAML configuration file. It defines pipelines, modules, alert channels, LLM providers, and database settings.

## Pipelines

Pipelines describe how to interpret a log stream.

```yaml
pipelines:
  - name: homeassistant
    grouping: whole_file  # or marker_based
    classifier: regex_counter  # or rsnapshot_heuristic
    ignore_regexes:
      - "^DEBUG"
```

- **grouping** controls how log lines are chunked before classification.
- **classifier** picks the rule set used to score each chunk.
- **ignore_regexes** lets you drop known-noise lines before counting errors and warnings.

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
