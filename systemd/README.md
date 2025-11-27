# systemd units for log-triage

Two services are provided so you can manage the main log processing loop and the Web UI independently:

- `logtriage.service`: runs the `logtriage` console script against `/etc/logtriage/config.yaml`.
- `logtriage-webui.service`: runs the `logtriage-webui` console script and reads the same configuration via `LOGTRIAGE_CONFIG`.

Both units assume the project lives in `/opt/logtriage` with a virtual environment in `/opt/logtriage/.venv/`. Adjust `User`, `Group`, `WorkingDirectory`, and the `PATH` environment override if your layout differs.

## Installation

```bash
sudo install -o root -g root -m 644 systemd/logtriage.service /etc/systemd/system/
sudo install -o root -g root -m 644 systemd/logtriage-webui.service /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable --now logtriage.service
sudo systemctl enable --now logtriage-webui.service
```

`StateDirectory=logtriage` and `RuntimeDirectory=logtriage` ensure `/var/lib/logtriage` and `/run/logtriage` exist for payloads, baselines, and SQLite databases configured there.
