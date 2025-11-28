# Web UI

The Web UI runs on FastAPI and shares the same configuration file as the CLI. It provides login, configuration editing, and a dashboard for findings.

> Configuration and regexes can be built and edited entirely in the Web UI. The config editor saves YAML with backups, while the regex lab lets you test patterns live before adding them to a pipeline.

## Start the server

```bash
export LOGTRIAGE_CONFIG=/path/to/config.yaml  # optional; defaults to ./config.yaml
logtriage-webui
```

By default the UI listens on `http://127.0.0.1:8090`. Use `LOGTRIAGE_HOST` and `LOGTRIAGE_PORT` to override the bind address.

## Quick start

!!! info "Login essentials"
    Open **http://127.0.0.1:8090** in your browser, sign in with **admin / admin123**, and change the password after first login.

1. Start the server (see above) and wait for the startup log line that shows the bind address.
2. Navigate to the Web UI, log in with the default credentials, and confirm you land on the dashboard.
3. Pick a module card to jump into the logs explorer, config editor, or regex lab.
4. Use the config editor or regex lab to adjust pipelines, then rerun follow-mode modules with `--reload-on-change` for immediate effect.

## Dashboard overview

The dashboard is the landing page once you log in:

- **Module cards** show the module name, current severity, and last activity time. A stale indicator is based on `stale_after_minutes` from the config for **follow** modules; batch modules finish immediately and never become stale.
- **24h counts** display warning and error totals to help spot spikes.
- **Quick links** jump to the logs explorer, config editor, or regex lab for the selected module.
- **Status badges** reflect whether modules are enabled and whether LLM payload generation is active.

Use these cards to prioritize which module to investigate first.

## Working with findings

Use severity updates to reflect the current state of each finding. Marking a finding as a false positive also adds the sample to `classifier.ignore_regexes` in the configuration and reloads it, preventing future matches from producing the same finding.

Findings can move through three related states:
- **Severity:** The current impact level (`WARNING`, `ERROR`, `CRITICAL`).
- **Addressed:** A checkbox that marks the item as reviewed without hiding it.
- **False positive:** Marks the pattern as noise and adds an ignore regex so future matches are suppressed.

## AI Logs Explorer

The AI Logs Explorer page displays log findings in context, allowing you to see the surrounding log lines and select any lines to send to an LLM for analysis:

1. **Pick a module:** Use the module selector to focus on a single pipeline.
2. **View findings in context:** Findings are highlighted inline within the full log view, with severity badges indicating their importance. Click the expand button on a finding line to see details and AI opinions.
3. **Select lines for analysis:** Check any log lines you want to analyze - including lines before and after findings to provide context.
4. **Send to LLM:** Click "Add to prompt" to add selected lines to the prompt builder, then send to your configured LLM provider.
5. **Update status:** Set the appropriate severity, mark as addressed, or flag as false positive. False positives write the ignore regex back to the pipeline automatically.

This unified view ensures you can see findings in their full context and select exactly the lines needed for analysis.

## Config editor and regex lab

Use these tools to evolve your configuration without leaving the browser:

1. **Open the config editor** from a module card or the navigation sidebar. Edit `pipelines`, `modules`, LLM settings, or alerts inline with schema hints, then save to write `config.yaml` atomically.
2. **Jump to the regex lab** from the same module. Paste sample log lines, try new ignore/warning/error patterns, and save them back to the selected pipeline when they behave as expected.
3. **Reload running modules** if you are tailing with the CLI by starting it with `--reload-on-change`, so changes take effect immediately.

## Features

- **Authentication:** username/password with bcrypt hashing.
- **Modules overview:** see enabled modules, last severity, and 24h error/warning counts.
- **Logs explorer:** browse findings in context, update severity, or mark false positives.
- **Config editor:** edit `config.yaml` with backups and atomic writes, then reload the running configuration.
- **Regex lab:** experiment with regexes and save them to classifiers.

## Database support

Set the `database.url` to SQLite or Postgres to persist findings between restarts. Without a database configured, the UI shows data for the current session only.
