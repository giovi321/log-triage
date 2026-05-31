# Web UI

The Web UI runs on FastAPI and shares the same configuration file as the CLI. It provides login, configuration editing, and a dashboard for findings.

> Configuration and regexes can be built and edited entirely in the Web UI (admin only). The config editor saves YAML with backups, while the regex lab lets you test patterns live before adding them to a pipeline.

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
2. Navigate to the Web UI, log in with the default credentials, and confirm you land on the **Triage** queue (the primary workflow).
3. Use the top nav to move between Triage, Overview (dashboard), and Logs; admins also see Settings and the Regex Lab.
4. Use the settings editor or regex lab to adjust pipelines, then rerun follow-mode modules with `--reload-on-change` for immediate effect.

![How findings become one cached issue](assets/diagram-dataflow.svg)

## Roles and admin access

Access is role-based. **Admins** can edit settings, use the regex lab, and
manage users; **non-admins** get the read/triage surfaces (Triage, Overview,
Logs, their own Account password). The Settings and Regex Lab nav links, the
user-management panel, and destructive log actions are hidden and
server-enforced for non-admins. Admin status comes from a local user's
`is_admin` flag or, for SSO users, membership in a configured OIDC
`admin_groups` (see [Configuration](configuration.md#oidc-single-sign-on)).

## Overview (dashboard)

The Overview page summarizes module health:

- **Module cards** show the module name, current severity, and last activity time. A stale indicator is based on `stale_after_minutes` from the config for **follow** modules; batch modules finish immediately and never become stale.
- **24h counts** display warning and error totals to help spot spikes.
- **Quick links** jump to the logs explorer, config editor, or regex lab for the selected module.
- **Status badges** reflect whether modules are enabled and whether LLM payload generation is active.

Use these cards to prioritize which module to investigate first.

The dashboard also updates **live over Server-Sent Events** (RAG indexing progress, counts) — no page refresh or client-side polling.

## Triage queue (issues)

The **Triage** page is the login landing and primary workflow when a database is configured. Recurring findings are collapsed into **issues** by a normalized signature (timestamps, IDs, IPs and numbers are stripped), so you review each distinct problem once instead of scrolling thousands of near-identical lines. The LLM analyzes each signature **once** and the summary is cached — so a problem that occurs 10,000 times costs one LLM call.

- **Prioritized list:** issues are ranked by `severity × recency × rate × novelty`. Each row shows the severity, signature, **occurrence count**, a 24-hour **sparkline**, first/last-seen, status, and the **cached AI summary**.
- **Navigate by interpretation:** each analyzed issue carries an LLM-assigned **category** (auth, network, storage, config, dependency, performance, or a free-form `other`). Category **filter chips** (with counts) narrow the queue, and a **Group by category** toggle sections the list by problem theme — so you can browse by *what the problem is*, not just by signature.
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

The AI Logs Explorer page displays log findings in context, allowing you to see the surrounding log lines and select any lines to send to an LLM for analysis. A **Grouped ⇄ Raw lines** toggle switches between the deduplicated, one-row-per-problem view (with occurrence count and cached AI summary, like Triage) and the full line-in-context view; a **Full triage →** link jumps to the global queue.

1. **Pick a module:** Use the module selector to focus on a single pipeline.
2. **View findings in context:** Findings are highlighted inline within the full log view, with severity badges indicating their importance. Click the expand button on a finding line to see details and AI opinions.
3. **Select lines for analysis:** Check any log lines you want to analyze - including lines before and after findings to provide context.
4. **Send to LLM:** Click "Add to prompt" to add selected lines to the prompt builder, then send to your configured LLM provider.
5. **Update status:** Set the appropriate severity, mark as addressed, or flag as false positive. False positives write the ignore regex back to the pipeline automatically.

This unified view ensures you can see findings in their full context and select exactly the lines needed for analysis.

## Settings editor and regex lab

These admin-only tools let you evolve your configuration without leaving the browser:

1. **Open Settings** from the navigation. It is a **structured forms** editor with a section sub-nav (General/Database/Logging, LLM + providers, RAG, Pipelines, Modules, Web UI) and add/remove rows for list items (providers, pipelines, regex lists, modules, knowledge sources, allowed IPs, OIDC admin groups). The Web UI section includes SSO/OIDC (with a copyable redirect URI), forward-auth, and the staleness window. **Save** validates the config and writes `config.yaml` atomically with a `.bak` backup, then reloads it in-process. (Users are managed on the Account page, not here; the raw-YAML editor and the separate "reload from disk" button were removed — saving the forms reloads automatically.)
2. **Jump to the Regex Lab** from the navigation. Paste sample log lines, try new ignore/warning/error patterns, and save them back to the selected pipeline when they behave as expected.
3. **Reload running modules** if you are tailing with the CLI by starting it with `--reload-on-change`, so changes take effect immediately.

## Account and user management

The **Account** page lets any signed-in user change their own password. Admins additionally get a **user-management** panel: list local users (with role), add a user (optionally admin), reset a password, or delete a user. Guardrails prevent deleting your own account or the last remaining admin. When SSO is enabled, local accounts are a break-glass path; most users sign in through the IdP and get admin via group membership.

## Features

- **Authentication:** local username/password (bcrypt, DB-backed), **OIDC SSO** (Authorization Code + PKCE), or **reverse-proxy forward-auth** — see [Configuration](configuration.md#web-ui-users-authentication-and-metrics).
- **Role-based access:** admins manage settings/users; non-admins get read/triage. Admin via local `is_admin` or an OIDC group.
- **Triage queue:** prioritized, de-duplicated issues with sparklines, cached AI summaries, **LLM-category navigation**, and acknowledge/resolve/mute/false-positive actions.
- **Modules overview:** enabled modules, last severity, 24h error/warning counts, and RAG status — updating live over SSE.
- **Logs explorer:** grouped (one row per problem) or raw line-in-context view; update severity or mark false positives.
- **Settings editor:** structured forms with backups and atomic in-process reload.
- **Regex lab:** experiment with regexes and save them to classifiers.
- **Metrics:** Prometheus exposition at `/metrics` (issue counts, findings, worker activity), subject to the IP allowlist; toggle with `webui.metrics.enabled`.
- **Themes:** light by default, with a one-click light/dark toggle in the header (remembered per browser).

## Database support

Set the `database.url` to SQLite or Postgres to persist findings between restarts and enable the **issues / Triage** workflow. Without a database configured, the UI shows data for the current session only and the Triage queue is unavailable.
