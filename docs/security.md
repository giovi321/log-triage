# Security considerations and assessment

This page documents the current security posture of `log-triage` (CLI, Web UI, and RAG components), the main threat scenarios, and recommended hardening steps.

`log-triage` is primarily designed for trusted environments (single-tenant, internal networks). If you plan to run it in multi-tenant or internet-exposed contexts, treat this as an **insecure-by-default** application and apply compensating controls.

## Security scope

The software consists of:

- **CLI** (`logtriage`): reads log files, produces findings, optionally calls an LLM provider, optionally sends alerts, optionally writes findings to a DB.
- **Web UI** (`logtriage-webui`): FastAPI app for viewing findings, editing config, managing regexes, and triggering LLM analysis.
- **RAG service** (`logtriage-rag`): FastAPI app that clones documentation repositories, builds embeddings, and serves retrieval results to the CLI/Web UI.

## Threat model (high level)

### Assets

- **Log contents** (may include credentials, tokens, IP addresses, user identifiers, internal URLs).
- **Configuration** (`config.yaml`) including:
  - file paths and glob patterns
  - alert endpoints (webhooks / MQTT)
  - LLM provider endpoints
  - Web UI admin user hashes
  - RAG repo URLs and branches
- **Session state** for Web UI.
- **RAG cache + vector store** contents (cloned repositories and indexed chunk text).
- **Database** records of findings and LLM outputs.

### Trust boundaries

- **Between browser and Web UI** (HTTP).
- **Between CLI/Web UI and external LLM provider** (HTTP).
- **Between CLI/Web UI and RAG service** (HTTP, local network).
- **Between RAG service and remote Git repositories** (network + filesystem writes).

### Primary attacker profiles

- **Network attacker** on the same network segment (sniffing/mitm if HTTP is used).
- **Untrusted user on the same host** (reads config, cache dirs, DB files).
- **Web attacker** able to get a logged-in admin to open a malicious page (CSRF).
- **Repository attacker** controlling a configured knowledge source repo content.

## Current security assessment (as implemented)

### Web UI authentication and session handling

Relevant code:

- `logtriage/webui/auth.py`
- `logtriage/webui/config.py`
- `logtriage/webui/app.py`

Observed behavior:

- **Local auth** is username/password with bcrypt verification; users live in the `webui_users` DB table (`logtriage/webui/users.py`), not in `config.yaml`.
- **OIDC SSO** (Authorization Code + PKCE, `logtriage/webui/oidc.py`) is supported; the IdP validates identity and the username comes from `username_claim`.
- **Role-based access control:** an `is_admin` flag is set at login (local `UserRecord.is_admin`, or OIDC group membership intersected with `webui.oidc.admin_groups`) and stored in the session **paired to the username**, so a flag minted for one identity can't elevate another. Admin is **server-enforced** on the settings editor, regex lab, and user-management routes (`require_admin` / `current_user_is_admin`); non-admins are redirected.
- **Session token** is an HMAC over `username|issued_at` (`create_session_token`), validated with a configurable max age (`webui.session_max_age_hours`, default 24h).
- **Server-side session storage** uses `SessionMiddleware` with a `secret_key` from config. `WebUISettings.secret_key` defaults to **`CHANGE_ME`**.
- **Optional forward-auth**: when `webui.forward_auth.enabled`, the identity is taken from a header (e.g. `X-authentik-username`) **only** if the direct peer is in `trusted_proxies` (`resolve_proxy_user`). Forward-auth users are non-admin by default.

Risks:

- **Weak/forgable sessions if `secret_key` is not changed**.
- **No server-side session revocation** — tokens are valid until they expire by age.
- **Cookie attributes** (Secure/HttpOnly/SameSite) are not explicitly enforced in code; behavior depends on Starlette defaults and deployment.

Recommendations:

- **Always set a strong `webui.secret_key`** (32+ random bytes) in production.
- Run the Web UI only behind:
  - **TLS termination** (reverse proxy), and
  - **network access controls** (VPN / firewall / allowlist).

### CSRF protections

Relevant code:

- `logtriage/webui/config.py` has a `csrf_enabled` setting.

Observed behavior:

- A CSRF middleware (`webui.csrf_enabled`, on by default) issues a per-session token and **validates it on form posts** (`application/x-www-form-urlencoded` and `multipart/form-data`) using a constant-time comparison.
- JSON API requests are **intentionally exempt** from CSRF checks (they are not auto-sent cross-site with credentials in the same way as forms).

Risks:

- The exemption means any future browser-triggered JSON state change would not be CSRF-protected; current state-changing actions use form posts.

Recommendations:

- Keep `csrf_enabled: true`.
- Do not expose the Web UI to untrusted origins; put a reverse proxy in front enforcing SSO and strict origin checks.

### Configuration editor and regex editor

Risk summary:

- The config editor modifies `config.yaml`, which controls:
  - log file locations
  - outbound webhook destinations
  - LLM endpoints
  - RAG repository URLs

Impact:

- If an attacker can write the config they can:
  - exfiltrate findings via webhooks
  - point log ingestion to sensitive files
  - point LLM to an attacker-controlled endpoint
  - cause the RAG service to clone attacker repos / grow disk usage

Recommendations:

- Treat config write access as **admin/root-equivalent**.
- Store `config.yaml` with strict permissions (`chmod 600`).

### IP allowlisting

Relevant code:

- `logtriage/webui/config.py:get_client_ip` uses `request.client.host`.

Risk:

- IP allowlisting is fragile behind reverse proxies (client IP may be the proxy).

Recommendation:

- If you deploy behind a proxy, enforce access control at the proxy layer.

### Reverse-proxy forward authentication

Relevant code:

- `logtriage/webui/auth.py:resolve_proxy_user`

Observed behavior:

- When `webui.forward_auth.enabled` is set, the username is read from `forward_auth.username_header` (default `X-authentik-username`) **only when the request's direct peer is one of `trusted_proxies`**. Otherwise the header is ignored.

Risks:

- If `trusted_proxies` is set too broadly (or to an address an attacker can originate from), the identity header can be spoofed. Keep `trusted_proxies` limited to the actual proxy/outpost address.

Recommendations:

- Terminate forward-auth at an Authentik proxy provider/outpost (or equivalent) and point `trusted_proxies` only at it.
- Ensure the proxy strips inbound `X-authentik-*` headers from clients so only the proxy can set them.

### Metrics endpoint

Relevant code:

- `logtriage/webui/metrics.py`, `/metrics` route in `app.py`.

Observed behavior:

- `/metrics` returns Prometheus text (issue counts, findings totals, worker activity) **without session auth** (so Prometheus can scrape it), but it is still subject to the `allowed_ips` middleware and can be disabled with `webui.metrics.enabled: false`.

Recommendations:

- Restrict scrape access via `allowed_ips` or the reverse proxy; disable it if unused. The exposed values are aggregate counts, not log contents.

### LLM provider calls (data exfiltration)

Relevant code:

- `logtriage/llm_client.py`
- `logtriage/llm_payload.py`

Risks:

- Sending excerpts to an LLM is a form of **intentional exfiltration**.
- LLM payloads may include secrets present in logs.
- API keys are sourced from environment variables (`provider.api_key_env`), which is good, but the operator must ensure they are not logged or exposed.

Recommendations:

- Redact secrets before ingestion (collector-side) or before LLM payload generation (application-side).
- Use a self-hosted LLM endpoint if logs are sensitive.
- Ensure outbound network egress is controlled.

### Alerts (webhook / MQTT)

Risks:

- Webhooks can exfiltrate metadata about incidents.
- Misconfiguration can cause the service to send data to untrusted endpoints.

Recommendations:

- Use TLS-only endpoints.
- Use allowlists or egress policies.

### Database and stored findings

Relevant code:

- `logtriage/webui/db.py`

Risks:

- Findings and LLM responses may store sensitive text.
- SQLite DB files are local files; permissions matter.

Recommendations:

- Restrict file permissions.
- For Postgres, use least-privilege DB accounts and TLS.

### RAG service security

Relevant code:

- `logtriage/rag/service.py`
- `logtriage/rag/knowledge_manager.py`
- `logtriage/rag/vector_store.py`

Observed behavior:

- The service clones repositories via GitPython.
- Git hooks are deleted after clone (`.git/hooks/*`).
- It persists a FAISS index plus a SQLite metadata DB.
- The Web UI's RAG status endpoints (`/api/rag/status`, `/api/rag/progress`) **require a valid session** — they expose operator-facing infra detail (repo URLs, commit hashes, indexing progress, memory) and are no longer reachable unauthenticated.
- The **standalone RAG service** (`logtriage-rag`, default `:8091`) exposes its own `/status` / `/progress` / `/health` **without auth** and is intended to be reached only by the Web UI/worker on the local host.

Risks:

- **Supply chain / content attacks**: repository content can contain prompt-injection text that will be retrieved and appended to prompts.
- **Disk exhaustion**: many repos or large repos can fill cache/vector-store directories.
- **Outbound network access**: clones from configured URLs.
- **Standalone RAG service info exposure** if its port is reachable beyond localhost.

Recommendations:

- Only add trusted documentation repositories.
- **Bind the standalone RAG service to localhost** (or a private interface) and never expose `:8091` publicly; it has no authentication of its own.
- Run the RAG service with:
  - a dedicated system user
  - restrictive filesystem permissions
  - disk quotas / separate partition for cache
- Consider isolating it (container) and restricting outbound network destinations.

## Deployment hardening checklist

- **Network**
  - Keep `logtriage-webui` and `logtriage-rag` bound to `127.0.0.1` when possible.
  - Put a reverse proxy in front for TLS and auth.
  - Firewall: allow only trusted admin subnets.

- **Secrets**
  - Set `webui.secret_key` to a strong random value.
  - Store LLM API keys only in environment variables.

- **Filesystem permissions**
  - Restrict `config.yaml` permissions.
  - Restrict RAG cache/vector store permissions.
  - Restrict DB credentials and DB files.

- **Process isolation**
  - Run services as non-root.
  - Use systemd hardening where possible (see `systemd/`).

- **Monitoring**
  - Enable logging to a secure location.
  - Monitor outbound connections.

## Known limitations / non-goals

- The Web UI is not currently designed for internet exposure.
- CSRF/session hardening is not comprehensive.
- The RAG component can amplify prompt-injection risks if untrusted documentation is indexed.
