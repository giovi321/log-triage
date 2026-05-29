# CLI

The CLI reads the same YAML configuration as the Web UI and provides batch and follow runs for each module. You can point it at any config file, or use the Web UI's config editor to build and edit modules and regexes without touching YAML manually.

## Commands

```bash
# Run a single module (batch or follow, per its config)
logtriage --config ./config.yaml --module <module-name>

# Run all enabled follow-mode modules (omit --module)
logtriage --config ./config.yaml

# Auto-reload when config.yaml changes (e.g. saved from the Web UI)
logtriage --config ./config.yaml --reload-on-change

# Build de-duplicated issues for findings created before issues existed, then exit
logtriage --config ./config.yaml --backfill-issues

# Analyze issues whose cached LLM summary is missing or stale, then exit
logtriage --config ./config.yaml --analyze-issues
```

- `--config` / `-c`: path to the YAML configuration (**required**).
- `--module` / `-m`: run a single named module (even if disabled). Omit to run all enabled **follow** modules.
- `--reload-on-change`: reload configuration when the file's mtime changes, so follow-mode modules pick up edits without a restart.
- `--backfill-issues`: one-time, idempotent migration that assigns fingerprints and groups existing findings into issues. The CLI also runs this automatically on startup (cheap once done).
- `--analyze-issues`: run a single enrichment pass over issues needing analysis, then exit. For continuous enrichment use the [`logtriage-worker`](#enrichment-worker) service or the in-Web-UI worker.

Run `logtriage --help` for the full flag list.

## Enrichment worker

`logtriage-worker` continuously analyzes issues whose cached summary is missing or stale (one LLM call per unique signature). By default the **Web UI runs this in-process**, so you only need the standalone process if you set `worker.run_in_webui: false` or run without the Web UI.

```bash
logtriage-worker --config ./config.yaml --interval 60   # loop every 60s
logtriage-worker --config ./config.yaml --once          # single pass, then exit
```

## Output formats

Modules emit findings as plaintext or JSON, set per module via `output_format: text | json` in the configuration (there is no CLI override). Use `min_print_severity` in the module configuration to suppress lower-importance findings (for example, only `ERROR` and `CRITICAL`).

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
