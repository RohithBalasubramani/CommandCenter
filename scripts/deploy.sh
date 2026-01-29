#!/usr/bin/env bash
# ==============================================================================
# CommandCenter — LAN Deployment Script
#
# Builds frontend, installs systemd user services, enables auto-start.
# Access at http://192.168.1.20:3100 from any device on the network.
#
# Usage:
#   ./scripts/deploy.sh          # Full deploy (build + install + start)
#   ./scripts/deploy.sh --stop   # Stop all services
#   ./scripts/deploy.sh --status # Check service status
#   ./scripts/deploy.sh --logs   # Tail all logs
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICES=(cc-backend cc-frontend cc-stt cc-tts)
SYSTEMD_DIR="$HOME/.config/systemd/user"
LOG_DIR="$PROJECT_DIR/logs"
LAN_IP="192.168.1.20"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
DIM='\033[0;90m'
RESET='\033[0m'

log()   { echo -e "${GREEN}[deploy]${RESET} $1"; }
warn()  { echo -e "${YELLOW}[deploy]${RESET} $1"; }
error() { echo -e "${RED}[deploy]${RESET} $1"; }

# ---------- Subcommands -------------------------------------------------------

if [[ "${1:-}" == "--stop" ]]; then
    log "Stopping all CommandCenter services..."
    systemctl --user stop "${SERVICES[@]}" 2>/dev/null || true
    log "All services stopped."
    exit 0
fi

if [[ "${1:-}" == "--status" ]]; then
    echo ""
    for svc in "${SERVICES[@]}"; do
        status=$(systemctl --user is-active "$svc" 2>/dev/null || echo "inactive")
        if [[ "$status" == "active" ]]; then
            echo -e "  ${GREEN}●${RESET} $svc — ${GREEN}active${RESET}"
        else
            echo -e "  ${RED}●${RESET} $svc — ${RED}$status${RESET}"
        fi
    done
    echo ""
    exit 0
fi

if [[ "${1:-}" == "--logs" ]]; then
    log "Tailing all logs (Ctrl+C to stop)..."
    tail -f "$LOG_DIR"/*.log
    exit 0
fi

# ---------- Full Deploy -------------------------------------------------------

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║      CommandCenter — LAN Deployment              ║${RESET}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${RESET}"
echo ""

# 1. Create logs directory
mkdir -p "$LOG_DIR"
log "Logs directory: $LOG_DIR"

# 2. Stop existing services (if running)
log "Stopping existing services..."
systemctl --user stop "${SERVICES[@]}" 2>/dev/null || true

# 3. Kill any dev server processes on our ports
log "Clearing ports..."
for port in 8100 3100 8890; do
    fuser -k "$port/tcp" 2>/dev/null || true
done

# 4. Install gunicorn if missing
if ! "$PROJECT_DIR/backend/venv/bin/pip" show gunicorn &>/dev/null; then
    log "Installing gunicorn..."
    "$PROJECT_DIR/backend/venv/bin/pip" install gunicorn
else
    log "Gunicorn already installed"
fi

# 5. Build frontend
log "Building Next.js frontend (production)..."
cd "$PROJECT_DIR/frontend"
npm run build
cd "$PROJECT_DIR"
log "Frontend build complete"

# 6. Install systemd service files
log "Installing systemd service files..."
mkdir -p "$SYSTEMD_DIR"
cp "$PROJECT_DIR/scripts/systemd/"*.service "$SYSTEMD_DIR/"
log "Copied services to $SYSTEMD_DIR"

# 7. Reload systemd
systemctl --user daemon-reload
log "systemd daemon reloaded"

# 8. Enable services (auto-start on login)
systemctl --user enable "${SERVICES[@]}"
log "Services enabled for auto-start"

# 9. Enable linger (services persist without active login session)
loginctl enable-linger "$(whoami)"
log "Linger enabled — services survive logout"

# 10. Start all services
log "Starting all services..."
systemctl --user start cc-stt
sleep 2
systemctl --user start cc-tts
sleep 2
systemctl --user start cc-backend
sleep 2
systemctl --user start cc-frontend

# 11. Wait for services to come up
log "Waiting for services to initialize..."
sleep 5

# 12. Status check
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║      Service Status                              ║${RESET}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${RESET}"
echo ""

all_ok=true
for svc in "${SERVICES[@]}"; do
    status=$(systemctl --user is-active "$svc" 2>/dev/null || echo "inactive")
    if [[ "$status" == "active" ]]; then
        echo -e "  ${GREEN}●${RESET} $svc — ${GREEN}active${RESET}"
    else
        echo -e "  ${RED}●${RESET} $svc — ${RED}$status${RESET}"
        all_ok=false
    fi
done

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║      Access URLs                                 ║${RESET}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  Frontend:  ${GREEN}http://${LAN_IP}:3100${RESET}"
echo -e "  Backend:   ${GREEN}http://${LAN_IP}:8100${RESET}"
echo -e "  STT:       ${GREEN}http://${LAN_IP}:8890${RESET}"
echo -e "  TTS:       ${GREEN}http://${LAN_IP}:8880${RESET}"
echo ""
echo -e "  ${DIM}Also accessible at http://localhost on this machine${RESET}"
echo ""

if $all_ok; then
    log "Deployment complete! All services running."
else
    warn "Some services failed to start. Check logs:"
    echo -e "  ${DIM}./scripts/deploy.sh --logs${RESET}"
    echo -e "  ${DIM}journalctl --user -u cc-backend -f${RESET}"
fi

echo ""
echo -e "${DIM}Management commands:${RESET}"
echo -e "  ${DIM}./scripts/deploy.sh --status   # Check status${RESET}"
echo -e "  ${DIM}./scripts/deploy.sh --stop     # Stop all${RESET}"
echo -e "  ${DIM}./scripts/deploy.sh --logs     # Tail logs${RESET}"
echo -e "  ${DIM}systemctl --user restart cc-backend  # Restart one${RESET}"
echo ""
