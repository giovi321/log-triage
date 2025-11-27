# Getting started

This guide walks through running `log-triage` locally, either once in batch mode or continuously in follow mode.

## Create or reuse a config file

`log-triage` looks for `config.yaml` in the current directory by default. You can point to a different file with `LOGTRIAGE_CONFIG` or the `--config` CLI flag.

A minimal config declares one pipeline and one module:

```yaml
pipelines:
  - name: homeassistant
    grouping: whole_file
    classifier: regex_counter
    ignore_regexes:
      - "^DEBUG"

modules:
  - name: homeassistant_batch
    enabled: true
    path: '/var/log/fluent-bit/homeassistant.log'
    mode: 'batch'
    pipeline: 'homeassistant'
    output_format: 'text'
    min_print_severity: 'WARNING'
```

- **Pipelines** describe how to group lines and which classifier to apply.
- **Modules** describe when and how to run a pipeline against a specific file.

You can create or refine these settings in the Web UI's config editor and regex lab instead of editing YAML by hand, which speeds up iteration.

## Run the CLI

Activate your virtual environment and call the CLI module. The examples assume your `config.yaml` sits in the project root.

```bash
source .venv/bin/activate
python -m logtriage --config config.yaml
```

Run only one module:

```bash
python -m logtriage --config config.yaml --module homeassistant_batch
```

Follow a log file in real time, reloading when the config changes:

```bash
python -m logtriage --config config.yaml --module homeassistant_follow --reload-on-change
```

## Enable LLM payloads

Add an LLM provider in the `llm.providers` map and refer to it from a module. Each module can specify its own prompt template and output directory for generated payloads.

```yaml
llm:
  default_provider: openai
  providers:
    openai:
      base_url: 'https://api.openai.com'
      model: 'gpt-4o-mini'
      api_key_env: 'OPENAI_API_KEY'

modules:
  - name: homeassistant_follow
    llm:
      enabled: true
      provider: 'openai'
      prompt_template: './prompts/homeassistant.txt'
      emit_llm_payloads_dir: './ha_llm_payloads'
      context_prefix_lines: 2
```

When `llm.enabled` is true and a provider is available, the CLI writes payload and response files next to the configured directory. You can forward these JSON payloads to your own LLM gateway.

## Alerts and storage

Modules can send alerts or persist findings for the Web UI:

- **Webhook alerts:** configure `alerts.webhook` with a URL and optional headers.
- **MQTT alerts:** configure `alerts.mqtt` with broker details and topic.
- **Database storage:** set the `database` block to use SQLite or Postgres so the Web UI can query historical findings.

See [Configuration](configuration.md) for the full schema.
