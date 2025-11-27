# Web UI

The Web UI runs on FastAPI and shares the same configuration file as the CLI. It provides login, configuration editing, and a dashboard for findings.

## Start the server

```bash
export LOGTRIAGE_CONFIG=/path/to/config.yaml  # optional; defaults to ./config.yaml
python -m logtriage.webui
```

By default the UI listens on `http://127.0.0.1:8090`. Use `LOGTRIAGE_HOST` and `LOGTRIAGE_PORT` to override the bind address.

## Dashboard overview

The dashboard is the landing page once you log in:

- **Module cards** show the module name, current severity, and last activity time. A stale indicator is based on `stale_after_minutes` from the config.
- **24h counts** display warning and error totals to help spot spikes.
- **Quick links** jump to the logs explorer, config editor, regex lab, or baseline files for the selected module.
- **Status badges** reflect whether modules are enabled and whether baseline or LLM payload generation is active.

Use these cards to prioritize which module to investigate first.

## Working with findings

Use severity updates to reflect the current state of each finding. Marking a finding as a false positive also adds the sample to `classifier.ignore_regexes` in the configuration and reloads it, preventing future matches from producing the same finding.

Findings can move through three related states:
- **Severity:** The current impact level (`WARNING`, `ERROR`, `CRITICAL`). Baseline anomalies can raise this automatically.
- **Addressed:** A checkbox that marks the item as reviewed without hiding it.
- **False positive:** Marks the pattern as noise and adds an ignore regex so future matches are suppressed.

## AI Logs Explorer

The AI Logs Explorer page walks you through triaging findings with LLM context:

1. **Pick a module:** Use the module selector to focus on a single pipeline.
2. **Filter and sort:** Narrow by severity, addressed/false-positive status, or timestamp to focus on recent or critical items.
3. **Open a finding:** Click a row to view the grouped log context, counts, and the reason the classifier chose the severity.
4. **Generate or view LLM payloads:** If LLM is enabled for the module, view the prebuilt payload and send it to your provider. Adjust the prompt template in the config editor if needed.
5. **Update status:** Set the appropriate severity, mark as addressed, or flag as false positive. False positives write the ignore regex back to the pipeline automatically.
6. **Iterate:** Move to the next finding using the navigator controls and repeat until the queue is clear.

This flow ensures each finding is reviewed, classified, and either escalated or suppressed with minimal context switching.

## Features

- **Authentication:** username/password with bcrypt hashing.
- **Modules overview:** see enabled modules, last severity, and 24h error/warning counts.
- **Logs explorer:** browse findings, update severity, or mark false positives.
- **Config editor:** edit `config.yaml` with backups and atomic writes, then reload the running configuration.
- **Regex lab:** experiment with regexes and save them to classifiers.
- **Baseline files:** edit baseline state JSON files used for anomaly detection.

## Database support

Set the `database.url` to SQLite or Postgres to persist findings between restarts. Without a database configured, the UI shows data for the current session only.
