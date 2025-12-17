# CLI

The CLI reads the same YAML configuration as the Web UI and provides batch and follow runs for each module. You can point it at any config file, or use the Web UI's config editor to build and edit modules and regexes without touching YAML manually.

## Commands

```bash
logtriage --config ./config.yaml run --module <module-name>
logtriage --config ./config.yaml run-all
logtriage --config ./config.yaml list-modules
```

- `--config`: path to the YAML configuration (defaults to `./config.yaml`).
- `run --module`: execute a single module in `batch` or `follow` mode depending on its config.
- `run-all`: execute every enabled module.
- `list-modules`: print module names, modes, and whether LLM payloads are enabled.

When running in follow mode, use `--reload-on-change` to pick up edits from the Web UI config editor automatically.

## Output formats

Modules can emit findings as plaintext or JSON.

```bash
logtriage run --module homeassistant_follow --output-format text
logtriage run --module homeassistant_follow --output-format json
```

Use `min_print_severity` in the module configuration to suppress lower-importance findings (for example, only `ERROR` and `CRITICAL`).

## Batch vs follow

- **Batch:** scans a file or directory once and exits. Useful for cron jobs.
- **Follow:** tails a file with rotation awareness. Configure `interval` and `from_beginning` in the module to control polling and start position.

Follow mode options map to configuration values explained in [Configuration](configuration.md#follow-mode-options). Edits made in the Web UI immediately shape follow behavior when `--reload-on-change` is used.

## Using alerts and LLM payloads

CLI runs honor module-level alert and LLM settings:

- Alerts fire webhook or MQTT messages based on the `alerts` block.
- LLM payloads are written to `emit_llm_payloads_dir` when `llm.enabled` is true. Prompt templates can be edited from the Web UI for faster iteration.

## Troubleshooting tips

- Run `logtriage --help` for the full list of global and subcommand flags.
- If regexes seem off, open the Web UI regex lab to refine them and re-run the CLI with `--reload-on-change` enabled.
