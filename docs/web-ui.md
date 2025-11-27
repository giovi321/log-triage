# Web UI

The Web UI runs on FastAPI and shares the same configuration file as the CLI. It provides login, configuration editing, and a dashboard for findings.

> Configuration and regexes can be built and edited entirely in the Web UI. The config editor saves YAML with backups, while the regex lab lets you test patterns live before adding them to a pipeline.

## Start the server

```bash
export LOGTRIAGE_CONFIG=/path/to/config.yaml  # optional; defaults to ./config.yaml
python -m logtriage.webui
```

By default the UI listens on `http://127.0.0.1:8090`. Use `LOGTRIAGE_HOST` and `LOGTRIAGE_PORT` to override the bind address.

## Quick start

!!! info "Login essentials"
    Open **http://127.0.0.1:8090** in your browser, sign in with **admin / admin123**, and change the password after first logi
n.

1. Start the server (see above) and wait for the startup log line that shows the bind address.
2. Navigate to the Web UI, log in with the default credentials, and confirm you land on the dashboard.
3. Pick a module card to jump into the logs explorer, config editor, regex lab, or baseline editor.
4. Use the config editor or regex lab to adjust pipelines, then rerun follow-mode modules with `--reload-on-change` for immedia
te effect.

## Dashboard overview

The dashboard is the landing page once you log in:

- **Module cards** show the module name, current severity, and last activity time. A stale indicator is based on `stale_after_minutes` from the config for **follow** modules; batch modules finish immediately and never become stale.
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

## Config editor and regex lab

Use these tools to evolve your configuration without leaving the browser:

1. **Open the config editor** from a module card or the navigation sidebar. Edit `pipelines`, `modules`, LLM settings, or alerts inline with schema hints, then save to write `config.yaml` atomically.
2. **Jump to the regex lab** from the same module. Paste sample log lines, try new ignore/warning/error patterns, and save them back to the selected pipeline when they behave as expected.
3. **Reload running modules** if you are tailing with the CLI by starting it with `--reload-on-change`, so changes take effect immediately.

## Baseline editor

The baseline editor helps you maintain anomaly-detection state files without hand-editing JSON:

1. From a module card, click **Baseline file** to open the current state (or create it if missing).
2. Review stored windows and multipliers, update values, and save to persist back to the `state_file` defined in `config.yaml`.
3. Return to the dashboard to see the updated anomaly indicators on the module cards.

## Features

- **Authentication:** username/password with bcrypt hashing.
- **Modules overview:** see enabled modules, last severity, and 24h error/warning counts.
- **Logs explorer:** browse findings, update severity, or mark false positives.
- **Config editor:** edit `config.yaml` with backups and atomic writes, then reload the running configuration.
- **Regex lab:** experiment with regexes and save them to classifiers.
- **Baseline files:** edit baseline state JSON files used for anomaly detection.

## Database support

Set the `database.url` to SQLite or Postgres to persist findings between restarts. Without a database configured, the UI shows data for the current session only.
