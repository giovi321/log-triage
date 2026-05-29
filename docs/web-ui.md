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

The dashboard also updates **live over Server-Sent Events** (RAG indexing progress, counts) — no page refresh or client-side polling.

## Triage queue (issues)

The **Triage** page is the primary workflow when a database is configured. Recurring findings are collapsed into **issues** by a normalized signature (timestamps, IDs, IPs and numbers are stripped), so you review each distinct problem once instead of scrolling thousands of near-identical lines.

- **Prioritized list:** issues are ranked by `severity × recency × rate × novelty`. Each row shows the severity, signature, **occurrence count**, a 24-hour **sparkline**, first/last-seen, status, and the **cached AI summary**.
- **Status tiles & filters:** filter by status (Active / Open / Acknowledged / Resolved / Muted / False-positive / All), module, severity, or a free-text search.
- **Issue detail:** opens the full AI analysis with documentation **citations**, 14-day and 24-hour occurrence timelines, the representative excerpt, and the recent occurrences table.
- **Workflow actions:** Acknowledge, Resolve, Mute, Mark false-positive, or Reopen. A resolved issue automatically reopens if it recurs.
- **On-demand analysis:** **Analyze now / Re-analyze** runs the LLM for that issue immediately; otherwise the [enrichment worker](cli.md#enrichment-worker) keeps summaries fresh in the background. Analysis is cached per signature, so it runs once and is reused everywhere.

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

1. **Open the config editor** from the navigation. It has two tabs:
   - **Forms** — structured editors for every section (General/Database/Logging, LLM + providers, RAG, Pipelines, Modules, Web UI), with add/remove rows for list items (providers, pipelines, regex lists, modules, knowledge sources, admin users, allowed IPs). Best for everyday changes.
   - **Advanced** — the raw-YAML editor (CodeMirror) with find/replace and context hints, as the fallback and safety net.
   Either way, **Save** validates the YAML and writes `config.yaml` atomically with a `.bak` backup. (Saving from the Forms tab reformats the YAML and does not preserve comments; use Advanced to keep them.)
2. **Jump to the regex lab** from the navigation. Paste sample log lines, try new ignore/warning/error patterns, and save them back to the selected pipeline when they behave as expected.
3. **Reload running modules** if you are tailing with the CLI by starting it with `--reload-on-change`, so changes take effect immediately.

## Features

- **Authentication:** username/password with bcrypt hashing; optional **reverse-proxy forward-auth** (e.g. Authentik) via a trusted identity header — see [Configuration](configuration.md#web-ui-metrics-and-forward-authentication).
- **Triage queue:** prioritized, de-duplicated issues with sparklines, cached AI summaries, and acknowledge/resolve/mute/false-positive actions.
- **Modules overview:** enabled modules, last severity, 24h error/warning counts, and RAG status — updating live over SSE.
- **Logs explorer:** browse findings in context, update severity, or mark false positives.
- **Config editor:** structured **forms** plus a raw-YAML **Advanced** tab, with backups and atomic writes.
- **Regex lab:** experiment with regexes and save them to classifiers.
- **Metrics:** Prometheus exposition at `/metrics` (issue counts, findings, worker activity), subject to the IP allowlist; toggle with `webui.metrics.enabled`.

## Database support

Set the `database.url` to SQLite or Postgres to persist findings between restarts and enable the **issues / Triage** workflow. Without a database configured, the UI shows data for the current session only and the Triage queue is unavailable.
