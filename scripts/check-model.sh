#!/bin/bash
# Check which model is currently active
# Usage: ./scripts/check-model.sh

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

echo ""
echo "=========================================="
echo "  Current Model Configuration"
echo "=========================================="
echo ""

# Check environment variables
FAST_MODEL="${OLLAMA_MODEL_FAST:-not set}"
QUALITY_MODEL="${OLLAMA_MODEL_QUALITY:-not set}"

echo -e "${BLUE}Environment Variables:${NC}"
echo "  OLLAMA_MODEL_FAST:    $FAST_MODEL"
echo "  OLLAMA_MODEL_QUALITY: $QUALITY_MODEL"
echo ""

# Check .env file
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
    echo -e "${BLUE}.env File:${NC}"
    grep "OLLAMA_MODEL" "$ENV_FILE" 2>/dev/null || echo "  No OLLAMA_MODEL entries found"
    echo ""
fi

# Check running backend process
echo -e "${BLUE}Backend Process:${NC}"
GUNICORN_PIDS=$(pgrep -f "gunicorn.*8100" 2>/dev/null || true)
if [ ! -z "$GUNICORN_PIDS" ]; then
    echo -e "  ${GREEN}✓${NC} Gunicorn running (PIDs: $GUNICORN_PIDS)"

    # Try to get env vars from process
    for pid in $GUNICORN_PIDS; do
        PROC_ENV=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep "OLLAMA_MODEL" || true)
        if [ ! -z "$PROC_ENV" ]; then
            echo "  Process env:"
            echo "$PROC_ENV" | sed 's/^/    /'
            break
        fi
    done
else
    echo "  ✗ No gunicorn process found on port 8100"
fi
echo ""

# Check Ollama models available
echo -e "${BLUE}Available Ollama Models:${NC}"
ollama list 2>/dev/null | grep -E "(llama3|glm-4)" || echo "  No relevant models found"
echo ""

# Determine current model
if [[ "$FAST_MODEL" == *"glm"* ]]; then
    echo "Current model: GLM-4.7-Flash"
elif [[ "$FAST_MODEL" == *"llama"* ]]; then
    echo "Current model: Llama 3.1 8B"
else
    echo "Current model: Unknown or not set"
fi
echo ""
