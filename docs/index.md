# log-triage

`log-triage` is a Python toolkit that sits between your log collector and an LLM. It reads log files, applies configurable rules to group and classify entries, and emits structured findings and payloads you can forward to a model. The CLI and Web UI share the same YAML configuration so you can run batch jobs, follow live streams, or explore results in a dashboard.

## Why use log-triage?

- **Pluggable pipelines.** Define multiple pipelines, each with its own grouping strategies and classifiers.
- **Flexible modules.** Decide whether a module runs once (batch) or tails a file with rotation awareness (follow).
- **Ignore noisy lines.** Drop known noise via `ignore_regexes` before counting errors and warnings.
- **Pluggable classifiers.** Swap between built-in regex or rsnapshot heuristics, or register your own (see [Classifiers](classifiers.md)).
- **Severity-aware.** Findings are labeled `WARNING`, `ERROR`, or `CRITICAL`, and can be escalated when anomalies are detected.
- **De-duplication into issues.** Recurring findings collapse by signature into ranked **issues** with occurrence counts and first/last-seen — triage each distinct problem once, not every line.
- **LLM analysis, cached per issue.** Analyze each signature once (RAG-grounded, with Anthropic prompt caching) and reuse the summary everywhere; a background worker keeps them fresh.
- **RAG-enhanced analysis.** Automatically retrieve relevant documentation from knowledge bases to provide more accurate, context-aware AI responses with citations.
- **Mission-control web UI.** A prioritized Triage queue, structured settings editor, live updates over SSE, and a Prometheus `/metrics` endpoint.
- **Alerts and storage.** Send webhook or MQTT alerts, and persist findings in SQLite or Postgres for the Web UI.

## What you need

- Python **3.10+**
- Access to the log files you want to analyze, or a collector such as Fluent Bit writing to disk
- Optional: credentials for your preferred LLM provider and database

## Installation

Create a virtual environment and install the package (including the Web UI, alerts, and RAG extras):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ".[webui,alerts,rag]"
```

### RAG Dependencies

The `rag` extra includes dependencies for Retrieval-Augmented Generation:
- `sentence-transformers` for document embeddings
- `chromadb` for vector storage and semantic search
- `GitPython` for repository cloning and management
- `markdown` for document processing

If you don't need RAG functionality, you can install without it:
```bash
pip install ".[webui,alerts]"
```

## Next steps

- Read the [Getting started](getting-started.md) guide to run your first module.
- Explore [Configuration](configuration.md) to see how pipelines, modules, and prompts fit together.
- Visit [Web UI](web-ui.md) to learn how to browse findings and edit the config from the dashboard.
- Check out the [RAG Quick Start Guide](RAG-QuickStart.md) to enable AI-powered analysis with documentation context.
- See the full [RAG documentation](RAG.md) for advanced configuration and troubleshooting.

## Security disclaimer concerning the Web UI

`log-triage` is intended for trusted, internal environments and should not be exposed directly to the public internet. It now ships with several controls, but important residual risks remain:

- **CSRF protection** is enforced on form posts via a per-session token (`webui.csrf_enabled`, on by default). JSON API calls are intentionally exempt.
- **Sessions** are HMAC-signed and now expire (`webui.session_max_age_hours`), but the scheme has no server-side revocation, and the default `secret_key` of `CHANGE_ME` makes forgery trivial — **always set a strong `secret_key`**.
- **Cookie attributes** (Secure/HttpOnly/SameSite) are not explicitly enforced in code; terminate TLS and set policies at a reverse proxy.
- **IP allowlisting** compares `request.client.host` (exact match, no CIDR); behind a proxy use `trusted_proxies` and enforce access at the proxy.
- **OIDC SSO** (Authorization Code + PKCE) and reverse-proxy **forward-auth** (e.g. Authentik) are both supported for putting the UI behind SSO — see [Configuration](configuration.md#web-ui-users-authentication-and-metrics).
- **Config write access is admin-only and root-equivalent:** the settings editor (gated to admins) can repoint log ingestion, webhooks, LLM endpoints, and RAG repos. Admin comes from a local user's flag or an OIDC group.

Run the UI only on private networks, behind strong network controls (TLS + SSO/forward-auth), with a unique `secret_key`. See the [security assessment](security.md) for the full picture.

