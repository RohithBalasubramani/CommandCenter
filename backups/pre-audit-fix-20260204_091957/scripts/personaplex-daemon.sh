#!/bin/bash
#
# PersonaPlex Daemon - Runs PersonaPlex as a persistent background service
#
# Usage:
#   ./scripts/personaplex-daemon.sh start   # Start PersonaPlex daemon
#   ./scripts/personaplex-daemon.sh stop    # Stop PersonaPlex daemon
#   ./scripts/personaplex-daemon.sh status  # Check if running
#   ./scripts/personaplex-daemon.sh restart # Restart daemon
#   ./scripts/personaplex-daemon.sh logs    # Tail logs
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PERSONAPLEX_DIR="$PROJECT_DIR/personaplex"
SSL_DIR="$PERSONAPLEX_DIR/ssl"
PORT=8998
PID_FILE="/tmp/personaplex.pid"
LOG_FILE="/tmp/personaplex.log"
USE_SSL="${PERSONAPLEX_SSL:-false}"  # Set PERSONAPLEX_SSL=true to enable SSL

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

is_running() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    # Also check by port
    if ss -tlnp 2>/dev/null | grep -q ":$PORT"; then
        return 0
    fi
    return 1
}

get_proto() {
    if [ "$USE_SSL" = "true" ]; then
        echo "https"
    else
        echo "http"
    fi
}

wait_for_ready() {
    local proto
    proto=$(get_proto)
    echo -n "Waiting for PersonaPlex to be ready"
    for i in {1..120}; do
        if curl -k -s --connect-timeout 2 "${proto}://localhost:$PORT/" > /dev/null 2>&1; then
            echo ""
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo ""
    return 1
}

start_daemon() {
    if is_running; then
        echo -e "${YELLOW}PersonaPlex is already running on port $PORT${NC}"
        exit 0
    fi

    echo -e "${GREEN}Starting PersonaPlex daemon...${NC}"

    # Start PersonaPlex
    cd "$PERSONAPLEX_DIR"
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    fi

    SSL_ARGS=""
    if [ "$USE_SSL" = "true" ]; then
        # Generate SSL certs if needed
        if [ ! -f "$SSL_DIR/cert.pem" ]; then
            echo "Generating SSL certificates..."
            mkdir -p "$SSL_DIR"
            openssl req -x509 -newkey rsa:2048 -keyout "$SSL_DIR/key.pem" \
                -out "$SSL_DIR/cert.pem" -days 365 -nodes \
                -subj "/CN=localhost" 2>/dev/null
        fi
        SSL_ARGS="--ssl $SSL_DIR"
        echo "SSL: enabled (certs in $SSL_DIR)"
    else
        echo "SSL: disabled (use PERSONAPLEX_SSL=true to enable)"
    fi

    nohup python3 -m moshi.server \
        --host "0.0.0.0" \
        --port "$PORT" \
        $SSL_ARGS \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "PID: $(cat "$PID_FILE")"
    echo "Log: $LOG_FILE"

    # Wait for it to be ready
    if wait_for_ready; then
        echo -e "${GREEN}PersonaPlex is ready on $(get_proto)://localhost:$PORT${NC}"
    else
        echo -e "${YELLOW}PersonaPlex may still be loading. Check: tail -f $LOG_FILE${NC}"
    fi
}

stop_daemon() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping PersonaPlex (PID: $pid)..."
            kill "$pid"
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid"
            fi
            rm -f "$PID_FILE"
            echo -e "${GREEN}PersonaPlex stopped${NC}"
            return
        fi
    fi

    # Try to kill by port
    fuser -k $PORT/tcp 2>/dev/null
    rm -f "$PID_FILE"
    echo "PersonaPlex stopped"
}

status_daemon() {
    if is_running; then
        echo -e "${GREEN}PersonaPlex is RUNNING on port $PORT${NC}"
        if [ -f "$PID_FILE" ]; then
            echo "PID: $(cat "$PID_FILE")"
        fi
        # Test if actually responding
        if curl -k -s --connect-timeout 2 "$(get_proto)://localhost:$PORT/" > /dev/null 2>&1; then
            echo "Status: Responding to requests"
        else
            echo "Status: Port bound but not yet responding (still loading?)"
        fi
    else
        echo -e "${RED}PersonaPlex is NOT running${NC}"
    fi
}

show_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo "No log file found at $LOG_FILE"
    fi
}

case "$1" in
    start)
        start_daemon
        ;;
    stop)
        stop_daemon
        ;;
    restart)
        stop_daemon
        sleep 2
        start_daemon
        ;;
    status)
        status_daemon
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
