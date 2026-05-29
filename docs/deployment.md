# Deployment

A typical production deployment runs as a few long-lived services managed by systemd, behind a reverse proxy on a trusted network.

## Components

| Service | Entry point | Purpose | Needed when |
|---|---|---|---|
| Ingest | `logtriage` | Tail/scan logs, classify, store findings, de-duplicate into issues | always |
| Web UI | `logtriage-webui` | Dashboard, Triage queue, config editor, `/metrics` | to use the UI |
| Worker | `logtriage-worker` | Per-issue LLM enrichment on an interval | only if `worker.run_in_webui: false` (the Web UI runs it in-process by default) |
| RAG | `logtriage-rag` | Documentation retrieval service | only if RAG is enabled |

## Install

```bash
python3 -m venv /opt/logtriage/.venv
/opt/logtriage/.venv/bin/pip install --upgrade pip
/opt/logtriage/.venv/bin/pip install ".[webui,alerts,rag]"   # drop extras you don't need
```

Place your configuration at `/etc/logtriage/config.yaml` (copy from `config.example.yaml`) and set a strong `webui.secret_key`. Restrict permissions: `chmod 600 /etc/logtriage/config.yaml`.

## systemd

Unit files live in [`systemd/`](https://github.com/giovi321/log-triage/tree/main/systemd):

- `logtriage.service` — a follow-mode ingest run
- `logtriage-webui.service` — the Web UI
- `logtriage-rag.service` — the RAG service
- `logtriage-worker.service` — the standalone enrichment worker (only if not running it inside the Web UI)

```bash
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now logtriage-webui.service
# enable logtriage.service (ingest) and, if used, logtriage-rag.service / logtriage-worker.service
```

Run services as a dedicated non-root user and apply systemd hardening where possible.

## First run with an existing database

If you are upgrading a database that predates issues, build them once:

```bash
/opt/logtriage/.venv/bin/logtriage --config /etc/logtriage/config.yaml --backfill-issues
```

(The ingest service also does this automatically on startup.)

## Reverse proxy, TLS, and SSO

The Web UI is for trusted networks — front it with a reverse proxy that terminates TLS and enforces authentication. Two common patterns:

- **Network-only:** bind `logtriage-webui` to `127.0.0.1`, proxy with TLS, and restrict by firewall / `webui.allowed_ips`.
- **SSO via Authentik:** put an Authentik **proxy provider/outpost** in front and set `webui.forward_auth.enabled: true` with `trusted_proxies` pointing at the outpost. log-triage then trusts the `X-authentik-username` header for identity. See [Configuration](configuration.md#web-ui-metrics-and-forward-authentication) and [Security](security.md).

## Monitoring

Scrape `GET /metrics` (Prometheus text) for issue counts, findings totals, and worker activity. Keep it behind `allowed_ips` or the proxy, or disable with `webui.metrics.enabled: false`.
