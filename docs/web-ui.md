# Web UI

The Web UI runs on FastAPI and shares the same configuration file as the CLI. It provides login, configuration editing, and a dashboard for findings.

## Start the server

```bash
export LOGTRIAGE_CONFIG=/path/to/config.yaml  # optional; defaults to ./config.yaml
python -m logtriage.webui
```

By default the UI listens on `http://127.0.0.1:8090`. Use `LOGTRIAGE_HOST` and `LOGTRIAGE_PORT` to override the bind address.

## Features

- **Authentication:** username/password with bcrypt hashing.
- **Modules overview:** see enabled modules, last severity, and 24h error/warning counts.
- **Logs explorer:** browse findings, update severity, or mark false positives.
- **Config editor:** edit `config.yaml` with backups and atomic writes, then reload the running configuration.
- **Regex lab:** experiment with regexes and save them to classifiers.
- **Baseline files:** edit baseline state JSON files used for anomaly detection.

## Working with findings

Use severity updates to reflect the current state of each finding. Marking a finding as a false positive also adds the sample to `classifier.ignore_regexes` in the configuration and reloads it, preventing future matches from producing the same finding.

## Database support

Set the `database.url` to SQLite or Postgres to persist findings between restarts. Without a database configured, the UI shows data for the current session only.
