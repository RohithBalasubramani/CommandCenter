#!/bin/bash
#
# BMC Metrics Logger
# Continuously logs system metrics including sensors, GPU, CPU, and RAM
#

# Configuration
LOG_DIR="${LOG_DIR:-/var/log/bmc-metrics}"
LOG_FILE="${LOG_DIR}/metrics.log"
INTERVAL="${INTERVAL:-30}"  # Logging interval in seconds
MAX_LOG_SIZE=$((100 * 1024 * 1024))  # 100MB

# Ensure log directory exists
mkdir -p "$LOG_DIR" 2>/dev/null || {
    # Fallback to user directory if /var/log is not writable
    LOG_DIR="$HOME/.local/share/bmc-metrics"
    LOG_FILE="${LOG_DIR}/metrics.log"
    mkdir -p "$LOG_DIR"
}

# Log rotation function
rotate_logs() {
    if [ -f "$LOG_FILE" ]; then
        local size=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null)
        if [ "$size" -gt "$MAX_LOG_SIZE" ]; then
            mv "$LOG_FILE" "${LOG_FILE}.$(date +%Y%m%d-%H%M%S)"
            # Keep only last 5 rotated logs
            ls -t "${LOG_FILE}".* 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null
        fi
    fi
}

# Function to log with timestamp
log_metric() {
    echo "$@" >> "$LOG_FILE"
}

# Function to collect all metrics
collect_metrics() {
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")

    log_metric ""
    log_metric "=========================================="
    log_metric "Timestamp: $timestamp"
    log_metric "=========================================="

    # CPU and RAM usage
    log_metric ""
    log_metric "--- CPU & RAM Usage ---"
    top -bn1 | grep "Cpu(s)" >> "$LOG_FILE"
    free -h | grep -E "Mem|Swap" >> "$LOG_FILE"

    # CPU temperature and details
    if command -v sensors &> /dev/null; then
        log_metric ""
        log_metric "--- Hardware Sensors ---"
        sensors >> "$LOG_FILE" 2>&1
    fi

    # GPU metrics
    if command -v nvidia-smi &> /dev/null; then
        log_metric ""
        log_metric "--- GPU Metrics ---"
        nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,fan.speed \
            --format=csv >> "$LOG_FILE" 2>&1
    fi

    # Disk usage
    log_metric ""
    log_metric "--- Disk Usage ---"
    df -h | grep -E "^/dev/" >> "$LOG_FILE"

    # Load average
    log_metric ""
    log_metric "--- Load Average ---"
    uptime >> "$LOG_FILE"

    # Process count
    log_metric ""
    log_metric "--- Process Count ---"
    echo "Total processes: $(ps aux | wc -l)" >> "$LOG_FILE"
}

# Trap signals for clean shutdown
trap 'echo "$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ") - BMC Metrics Logger stopped" >> "$LOG_FILE"; exit 0' SIGTERM SIGINT

# Main logging loop
echo "$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ") - BMC Metrics Logger started (interval: ${INTERVAL}s, log: $LOG_FILE)" >> "$LOG_FILE"
echo "BMC Metrics Logger started. Logging to: $LOG_FILE"
echo "Logging interval: ${INTERVAL} seconds"

while true; do
    rotate_logs
    collect_metrics
    sleep "$INTERVAL"
done
