#!/bin/bash
#
# Command Center - Unified Development Server
#
# V2 Pipeline: STT (Parakeet/Whisper) + TTS (Kokoro) + Django + Next.js
# V1 Pipeline: PersonaPlex (set USE_V2_PIPELINE=false)
#
# Usage:
#   ./scripts/dev.sh                  # Start servers
#   ./scripts/dev.sh --setup          # Run setup first, then start
#   ./scripts/dev.sh --full           # Full reset + setup + start
#   ./scripts/dev.sh --v1             # Use V1 (PersonaPlex) pipeline
#

set -uo pipefail
# Not using set -e: many commands (curl, fuser, docker) are expected to fail gracefully

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝

PROJECT_NAME="Command Center"
LOG_RETENTION_DAYS=5
BACKEND_PORT=8100
FRONTEND_PORT=3100
PERSONAPLEX_PORT=8998
STT_PORT=8890
TTS_PORT=8880
BACKEND_TYPE="django"
FRONTEND_TYPE="nextjs"
USE_V2_PIPELINE=true
RUN_SETUP=false
RUN_FULL_RESET=false

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PATHS                                                                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
LOG_DIR="$SCRIPT_DIR/dev-helpers/logs"
SESSION_ID=$(date +%Y%m%d-%H%M%S)
SESSION_LOG="$LOG_DIR/session-$SESSION_ID.log"

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  COLORS                                                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
DIM='\033[0;90m'
NC='\033[0m'
BOLD='\033[1m'

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PROCESS TRACKING                                                         ║
# ╚════════════════════════════════════════════════════════════════════════════╝

BACKEND_PID=""
FRONTEND_PID=""
PERSONAPLEX_PID=""
STT_PID=""

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  LOGGING                                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝

log_line() {
    local source="$1"
    local message="$2"
    local color="${3:-$NC}"
    local time
    time=$(date '+%H:%M:%S')
    local full_time
    full_time=$(date '+%Y-%m-%d %H:%M:%S')

    local source_color source_tag
    case "$source" in
        SYSTEM)      source_tag="[SYSTEM] "; source_color="$MAGENTA" ;;
        BACKEND)     source_tag="[BACKEND]"; source_color="$YELLOW" ;;
        FRONTEND)    source_tag="[FRONT] "; source_color="$CYAN" ;;
        STT)         source_tag="[STT]   "; source_color="$GREEN" ;;
        TTS)         source_tag="[TTS]   "; source_color="$BLUE" ;;
        PERSONAPLEX) source_tag="[PERSONA]"; source_color="$GREEN" ;;
        *)           source_tag="         "; source_color="$WHITE" ;;
    esac

    echo -e "${DIM}[$time]${NC} ${source_color}${source_tag}${NC} ${color}${message}${NC}"
    echo "[$full_time] $source_tag $message" >> "$SESSION_LOG" 2>/dev/null || true
}

prefix_output() {
    local source="$1"
    local color="$2"
    while IFS= read -r line || [ -n "$line" ]; do
        if [ -n "$line" ]; then
            local time
            time=$(date '+%H:%M:%S')
            local full_time
            full_time=$(date '+%Y-%m-%d %H:%M:%S')
            local line_color="$color"
            if echo "$line" | grep -qiE "error|exception|traceback|failed"; then
                line_color="$RED"
            elif echo "$line" | grep -qiE "warning|warn"; then
                line_color="$YELLOW"
            elif echo "$line" | grep -qiE "ready|compiled|started|listening"; then
                line_color="$GREEN"
            fi
            echo -e "${DIM}[$time]${NC} ${color}[$source]${NC} ${line_color}${line}${NC}"
            echo "[$full_time] [$source] $line" >> "$SESSION_LOG" 2>/dev/null || true
        fi
    done
}

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CLEANUP                                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝

cleanup() {
    echo ""
    log_line "SYSTEM" "Shutting down..." "$YELLOW"

    for pid_var in BACKEND_PID FRONTEND_PID PERSONAPLEX_PID STT_PID; do
        local pid="${!pid_var}"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            log_line "SYSTEM" "  Stopped $pid_var ($pid)" "$DIM"
        fi
    done

    # Kill anything left on our ports (except TTS Docker which persists)
    fuser -k $BACKEND_PORT/tcp 2>/dev/null || true
    fuser -k $FRONTEND_PORT/tcp 2>/dev/null || true
    fuser -k $STT_PORT/tcp 2>/dev/null || true

    echo ""
    log_line "SYSTEM" "Session ended: $SESSION_ID" "$DIM"
    log_line "SYSTEM" "Log: $SESSION_LOG" "$DIM"
    echo -e "    ${MAGENTA}Goodbye!${NC}"
    echo ""
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS                                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝

check_port() {
    local port=$1
    if fuser "$port/tcp" 2>/dev/null; then
        log_line "SYSTEM" "Port $port in use — freeing..." "$YELLOW"
        fuser -k "$port/tcp" 2>/dev/null || true
        sleep 1
    fi
}

wait_for_url() {
    local url="$1"
    local label="$2"
    local timeout="${3:-30}"
    local i=0
    while [ $i -lt "$timeout" ]; do
        if curl -s --connect-timeout 2 "$url" > /dev/null 2>&1; then
            return 0
        fi
        echo -n "."
        sleep 1
        i=$((i + 1))
    done
    echo ""
    return 1
}

activate_venv() {
    local dir="$1"
    if [ -d "$dir/venv" ]; then
        # shellcheck disable=SC1091
        source "$dir/venv/bin/activate"
    elif [ -d "$dir/.venv" ]; then
        # shellcheck disable=SC1091
        source "$dir/.venv/bin/activate"
    fi
}

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PARSE ARGS                                                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

while [[ $# -gt 0 ]]; do
    case $1 in
        --v1)             USE_V2_PIPELINE=false; shift ;;
        --backend-port)   BACKEND_PORT="$2"; shift 2 ;;
        --frontend-port)  FRONTEND_PORT="$2"; shift 2 ;;
        --setup)          RUN_SETUP=true; shift ;;
        --full)           RUN_FULL_RESET=true; RUN_SETUP=true; shift ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --v1              Use V1 pipeline (PersonaPlex) instead of V2"
            echo "  --setup           Run setup before starting"
            echo "  --full            Full reset + setup + start"
            echo "  --backend-port N  Override backend port (default: 8100)"
            echo "  --frontend-port N Override frontend port (default: 3100)"
            echo ""
            exit 0
            ;;
        *) shift ;;
    esac
done

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  BANNER                                                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

mkdir -p "$LOG_DIR"
clear
echo -e "${BOLD}${MAGENTA}"
echo "    +================================================================+"
echo "    |                                                                |"
printf "    |      %-50s      |\n" "${PROJECT_NAME^^}"
if [ "$USE_V2_PIPELINE" = true ]; then
echo "    |      V2 Pipeline: STT + TTS + Django + Next.js                 |"
else
echo "    |      V1 Pipeline: PersonaPlex + Django + Next.js               |"
fi
echo "    |                                                                |"
printf "    |      Session: %-43s |\n" "$SESSION_ID"
echo "    |                                                                |"
echo "    +================================================================+"
echo -e "${NC}"

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PREREQUISITES                                                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

echo -e "  ${DIM}--- PREREQUISITES ------------------------------------------${NC}"
echo ""

if [ -d "$BACKEND_DIR/venv" ] || [ -d "$BACKEND_DIR/.venv" ]; then
    log_line "SYSTEM" "Python venv: OK" "$GREEN"
else
    log_line "SYSTEM" "Python venv not found — will try system Python" "$YELLOW"
fi

if command -v npm &> /dev/null; then
    log_line "SYSTEM" "npm: v$(npm --version)" "$GREEN"
else
    log_line "SYSTEM" "npm not found!" "$RED"
    exit 1
fi

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SETUP (optional)                                                         ║
# ╚════════════════════════════════════════════════════════════════════════════╝

if [ "$RUN_FULL_RESET" = true ]; then
    log_line "SYSTEM" "Full reset + setup..." "$MAGENTA"
    "$SCRIPT_DIR/setup.sh" --reset
elif [ "$RUN_SETUP" = true ]; then
    log_line "SYSTEM" "Running setup..." "$MAGENTA"
    "$SCRIPT_DIR/setup.sh"
fi

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CLEANUP OLD LOGS + PORTS                                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

echo ""
echo -e "  ${DIM}--- CLEANUP ------------------------------------------------${NC}"
echo ""

# Old logs
if [ -d "$LOG_DIR" ]; then
    old_count=$(find "$LOG_DIR" -name "session-*.log" -mtime +$LOG_RETENTION_DAYS 2>/dev/null | wc -l)
    if [ "$old_count" -gt 0 ]; then
        log_line "SYSTEM" "Cleaning $old_count old log(s)" "$DIM"
        find "$LOG_DIR" -name "session-*.log" -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
    fi
fi

# Stop systemd production services (they have Restart=always, so fuser -k alone won't work)
if systemctl --user is-active cc-backend cc-frontend cc-stt cc-tts 2>/dev/null | grep -q "^active"; then
    log_line "SYSTEM" "Stopping production services (systemd)..." "$YELLOW"
    systemctl --user stop cc-backend cc-frontend cc-stt cc-tts 2>/dev/null || true
    sleep 1
fi

# Free ports
check_port $BACKEND_PORT
check_port $FRONTEND_PORT
if [ "$USE_V2_PIPELINE" = true ]; then
    check_port $STT_PORT
fi

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  OLLAMA                                                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

echo ""
echo -e "  ${DIM}--- OLLAMA (LLM) -------------------------------------------${NC}"
echo ""

if command -v ollama &> /dev/null; then
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log_line "SYSTEM" "Ollama: running" "$GREEN"
        models=$(curl -s http://localhost:11434/api/tags 2>/dev/null | grep -o '"name":"[^"]*"' | head -3 | sed 's/"name":"//g;s/"//g' | tr '\n' ', ')
        [ -n "$models" ] && log_line "SYSTEM" "  Models: $models" "$DIM"
    else
        log_line "SYSTEM" "Ollama: starting..." "$YELLOW"
        ollama serve > /dev/null 2>&1 &
        sleep 3
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            log_line "SYSTEM" "  Ollama started" "$GREEN"
        else
            log_line "SYSTEM" "  Ollama failed to start (RAG will use fallbacks)" "$YELLOW"
        fi
    fi
else
    log_line "SYSTEM" "Ollama: not installed (RAG will use fallbacks)" "$YELLOW"
fi

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  START SERVICES                                                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

echo ""
echo -e "  ${DIM}--- STARTING SERVICES --------------------------------------${NC}"
echo ""

# ---------- V2: STT + TTS --------------------------------------------------

if [ "$USE_V2_PIPELINE" = true ]; then

    # --- STT Server ---
    if curl -s --connect-timeout 2 "http://localhost:$STT_PORT/v1/stt/health" > /dev/null 2>&1; then
        log_line "STT" "Already running on :$STT_PORT" "$GREEN"
    else
        log_line "STT" "Starting (Parakeet + Whisper fallback)..." "$GREEN"
        (
            cd "$PROJECT_DIR/backend/stt"
            activate_venv "$BACKEND_DIR"
            exec python3 server.py 2>&1
        ) | prefix_output "STT" "$GREEN" &
        STT_PID=$!
        log_line "STT" "  PID: $STT_PID" "$DIM"

        # Non-blocking: just wait 10s, report status, continue
        log_line "STT" "  Loading model (continuing in background)..." "$DIM"
        if wait_for_url "http://localhost:$STT_PORT/v1/stt/health" "STT" 10; then
            log_line "STT" "Ready on :$STT_PORT" "$GREEN"
        else
            log_line "STT" "Still loading (will be ready soon — first run downloads model)" "$YELLOW"
        fi
    fi

    # --- TTS Server (Kokoro Native GPU) ---
    if curl -s --connect-timeout 2 "http://localhost:$TTS_PORT/v1/models" > /dev/null 2>&1; then
        log_line "TTS" "Kokoro already running on :$TTS_PORT" "$GREEN"
    else
        TTS_DIR="$PROJECT_DIR/backend/tts/Kokoro-FastAPI"
        TTS_VENV="$PROJECT_DIR/backend/tts/venv"
        if [ -d "$TTS_VENV" ] && [ -d "$TTS_DIR" ]; then
            log_line "TTS" "Starting Kokoro (native GPU)..." "$BLUE"
            (
                cd "$TTS_DIR"
                export USE_GPU=true
                export USE_ONNX=false
                export PYTHONPATH="$TTS_DIR/api"
                export MODEL_DIR=src/models
                export VOICES_DIR=src/voices/v1_0
                export WEB_PLAYER_PATH="$TTS_DIR/web"
                exec "$TTS_VENV/bin/uvicorn" api.src.main:app --host 0.0.0.0 --port "$TTS_PORT" 2>&1
            ) | prefix_output "TTS" "$BLUE" &

            if wait_for_url "http://localhost:$TTS_PORT/v1/models" "TTS" 30; then
                log_line "TTS" "Kokoro ready on :$TTS_PORT (GPU)" "$GREEN"
            else
                log_line "TTS" "Kokoro still loading (browser TTS fallback active)" "$YELLOW"
            fi
        else
            log_line "TTS" "Kokoro not installed — run: git clone https://github.com/remsky/Kokoro-FastAPI backend/tts/Kokoro-FastAPI" "$RED"
            log_line "TTS" "Browser TTS fallback active" "$YELLOW"
        fi
    fi

# ---------- V1: PersonaPlex ------------------------------------------------

else
    USE_SSL="${PERSONAPLEX_SSL:-false}"
    SSL_DIR="$PROJECT_DIR/personaplex/ssl"
    PERSONAPLEX_PROTO="http"
    [ "$USE_SSL" = "true" ] && PERSONAPLEX_PROTO="https"

    if curl -k -s --connect-timeout 2 "http://localhost:$PERSONAPLEX_PORT/" > /dev/null 2>&1 ||
       curl -k -s --connect-timeout 2 "https://localhost:$PERSONAPLEX_PORT/" > /dev/null 2>&1; then
        log_line "PERSONAPLEX" "Already running on :$PERSONAPLEX_PORT" "$GREEN"
    else
        log_line "PERSONAPLEX" "Starting PersonaPlex-7B..." "$GREEN"
        SSL_ARGS=""
        if [ "$USE_SSL" = "true" ]; then
            if [ ! -f "$SSL_DIR/cert.pem" ]; then
                mkdir -p "$SSL_DIR"
                openssl req -x509 -newkey rsa:2048 -keyout "$SSL_DIR/key.pem" \
                    -out "$SSL_DIR/cert.pem" -days 365 -nodes \
                    -subj "/CN=localhost" 2>/dev/null
            fi
            SSL_ARGS="--ssl $SSL_DIR"
        fi

        (
            cd "$PROJECT_DIR/personaplex"
            activate_venv "$PROJECT_DIR/personaplex"
            exec python3 -m moshi.server \
                --host "0.0.0.0" --port "$PERSONAPLEX_PORT" $SSL_ARGS 2>&1
        ) | prefix_output "PERSONAPLEX" "$GREEN" &
        PERSONAPLEX_PID=$!

        log_line "PERSONAPLEX" "  Loading models (30-60s)..." "$DIM"
        if wait_for_url "${PERSONAPLEX_PROTO}://localhost:$PERSONAPLEX_PORT/" "PersonaPlex" 90; then
            log_line "PERSONAPLEX" "Ready on :$PERSONAPLEX_PORT" "$GREEN"
        else
            log_line "PERSONAPLEX" "Timeout — may still be loading" "$YELLOW"
        fi
    fi
fi

# ---------- Backend (Django) ------------------------------------------------

log_line "BACKEND" "Starting Django on :$BACKEND_PORT..." "$YELLOW"
(
    cd "$BACKEND_DIR"
    activate_venv "$BACKEND_DIR"
    exec python manage.py runserver "0.0.0.0:$BACKEND_PORT" 2>&1
) | prefix_output "BACKEND" "$YELLOW" &
BACKEND_PID=$!

if wait_for_url "http://localhost:$BACKEND_PORT/" "Backend" 15; then
    log_line "BACKEND" "Ready on :$BACKEND_PORT" "$GREEN"
else
    log_line "BACKEND" "Not ready yet (may still be starting)" "$YELLOW"
fi

# ---------- Frontend (Next.js) ----------------------------------------------

log_line "FRONTEND" "Starting Next.js on :$FRONTEND_PORT..." "$CYAN"
(
    cd "$FRONTEND_DIR"
    exec npm run dev -- -p "$FRONTEND_PORT" 2>&1
) | prefix_output "FRONTEND" "$CYAN" &
FRONTEND_PID=$!

sleep 3

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  STATUS BANNER                                                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

echo ""
echo -e "${GREEN}"
echo "    +------------------------------------------------------------------+"
echo "    |                                                                  |"
echo "    |    COMMAND CENTER - RUNNING                                      |"
echo "    |                                                                  |"
if [ "$USE_V2_PIPELINE" = true ]; then
echo "    |    V2 Voice Pipeline:                                            |"
printf "    |      STT Server:   http://localhost:%-27s|\n" "$STT_PORT"
printf "    |      TTS Server:   http://localhost:%-27s|\n" "$TTS_PORT"
else
echo "    |    V1 Voice Pipeline:                                            |"
printf "    |      PersonaPlex:  wss://localhost:%s/api/chat%-14s|\n" "$PERSONAPLEX_PORT" ""
fi
echo "    |                                                                  |"
echo "    |    Layer 2 (AI + RAG):                                           |"
printf "    |      Backend API:  http://localhost:%-27s|\n" "$BACKEND_PORT"
printf "    |      Ollama LLM:   http://localhost:%-27s|\n" "11434"
echo "    |                                                                  |"
echo "    |    Frontend:                                                     |"
printf "    |      App:          http://localhost:%-27s|\n" "$FRONTEND_PORT"
echo "    |                                                                  |"
echo "    |    Press Ctrl+C to stop all services                             |"
echo "    |                                                                  |"
echo "    +------------------------------------------------------------------+"
echo -e "${NC}"

echo -e "  ${DIM}--- LIVE LOGS (Ctrl+C to stop) ---${NC}"
echo ""

# Wait for all background processes
wait
