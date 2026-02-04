# CommandCenter — Production Deployment

Deployed: **2026-01-31** | Host: **192.168.1.20** | Service prefix: `command-center`

## Services

| Service | Tech | Port | Protocol | Status |
|---------|------|------|----------|--------|
| Frontend | Next.js 14 (custom HTTPS server) | 9210 | HTTPS | `systemctl --user status command-center-frontend` |
| Backend | Django 5 + Gunicorn (3 workers) | 9211 | HTTP | `systemctl --user status command-center-backend` |
| STT | FastAPI (Parakeet/Whisper) | 9212 | HTTP | `systemctl --user status command-center-stt` |

## URLs

### Network (LAN)
- **Frontend**: https://192.168.1.20:9210
- **Backend API**: http://192.168.1.20:9211/api/layer2/
- **Django Admin**: http://192.168.1.20:9211/admin/
- **STT Health**: http://192.168.1.20:9212/v1/stt/health
- **STT Docs**: http://192.168.1.20:9212/docs

### Localhost
- Frontend: https://localhost:9210
- Backend: http://localhost:9211
- STT: http://localhost:9212

## Directory Structure

```
prodo/
  backend/         Django app + STT server (copied, no venv)
  frontend/        Next.js build + node_modules + server.js
  certs/           SSL certificates (lan.key, lan.crt)
  config/
    ports.env      Port assignments for all services
    deployment.env Full deployment metadata
  logs/
    frontend.log   Next.js HTTPS server logs
    backend.log    Gunicorn/Django logs
    stt.log        STT server logs
  README.md        This file
```

## Service Management

```bash
# Status
systemctl --user status command-center-frontend
systemctl --user status command-center-backend
systemctl --user status command-center-stt

# Restart
systemctl --user restart command-center-frontend
systemctl --user restart command-center-backend
systemctl --user restart command-center-stt

# Stop
systemctl --user stop command-center-frontend
systemctl --user stop command-center-backend
systemctl --user stop command-center-stt

# Disable (prevent auto-start on login)
systemctl --user disable command-center-frontend
systemctl --user disable command-center-backend
systemctl --user disable command-center-stt
```

## Logs

```bash
# Live logs
tail -f /home/rohith/desktop/CommandCenter/prodo/logs/frontend.log
tail -f /home/rohith/desktop/CommandCenter/prodo/logs/backend.log
tail -f /home/rohith/desktop/CommandCenter/prodo/logs/stt.log

# Last 50 lines
journalctl --user -u command-center-frontend -n 50 --no-pager
journalctl --user -u command-center-backend -n 50 --no-pager
journalctl --user -u command-center-stt -n 50 --no-pager
```

## Health Checks

```bash
# Frontend
curl -sk https://localhost:9210/ -o /dev/null -w "%{http_code}\n"

# Backend
curl -s http://localhost:9211/api/layer2/ -o /dev/null -w "%{http_code}\n"

# STT
curl -s http://localhost:9212/v1/stt/health
```

## Configuration

- **Port config**: `prodo/config/ports.env`
- **Frontend env**: `prodo/frontend/.env.production.local`
- **SSL certs**: `prodo/certs/lan.key`, `prodo/certs/lan.crt`
- **Django settings**: `prodo/backend/command_center/settings.py`

After changing ports, update `ports.env`, rebuild frontend with new `.env.production.local`, and restart services.

## Database

SQLite database at `prodo/backend/db.sqlite3`.

```bash
# Backup
cp prodo/backend/db.sqlite3 prodo/backend/db.sqlite3.bak

# Restore
cp prodo/backend/db.sqlite3.bak prodo/backend/db.sqlite3
systemctl --user restart command-center-backend
```

## Troubleshooting

1. **Service won't start**: Check logs with `journalctl --user -u command-center-<service> -n 100`
2. **Port conflict**: Check `ss -tuln | grep 921` and update `ports.env`
3. **SSL errors**: Verify certs exist in `prodo/certs/` and are readable
4. **Frontend 502**: Ensure backend is running on port 9211
5. **STT import errors**: STT requires NVIDIA NeMo or faster-whisper in the backend venv

## Redeployment

```bash
# 1. Stop services
systemctl --user stop command-center-frontend command-center-backend command-center-stt

# 2. Rebuild frontend (from source)
cd /home/rohith/desktop/CommandCenter/frontend
npx next build

# 3. Copy new build
rm -rf /home/rohith/desktop/CommandCenter/prodo/frontend/.next
cp -r .next /home/rohith/desktop/CommandCenter/prodo/frontend/.next

# 4. Restart
systemctl --user restart command-center-frontend command-center-backend command-center-stt
```

## Uninstall

```bash
systemctl --user stop command-center-frontend command-center-backend command-center-stt
systemctl --user disable command-center-frontend command-center-backend command-center-stt
rm ~/.config/systemd/user/command-center-*.service
systemctl --user daemon-reload
rm -rf /home/rohith/desktop/CommandCenter/prodo
```
