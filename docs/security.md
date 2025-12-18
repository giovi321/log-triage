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

- **Admin auth** is username/password with bcrypt verification.
- **Session token** is a simple HMAC: `username|signature` (`create_session_token`).
- **Server-side session storage** uses `SessionMiddleware` with a `secret_key` from config.
- `WebUISettings.secret_key` defaults to **`CHANGE_ME`**.

Risks:

- **Weak/forgable sessions if `secret_key` is not changed**.
- **No explicit session expiry/rotation** in the custom session token.
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

- The docs and codebase indicate that the Web UI currently lacks robust CSRF protections for state-changing requests.

Risks:

- An attacker can attempt to trigger config changes or state updates if an admin is logged in and visits a malicious site.

Recommendations:

- Do not expose the Web UI to untrusted origins.
- Put a reverse proxy in front that enforces:
  - authentication (SSO) and
  - anti-CSRF (or strict origin checks).

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

Risks:

- **Supply chain / content attacks**: repository content can contain prompt-injection text that will be retrieved and appended to prompts.
- **Disk exhaustion**: many repos or large repos can fill cache/vector-store directories.
- **Outbound network access**: clones from configured URLs.

Recommendations:

- Only add trusted documentation repositories.
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
