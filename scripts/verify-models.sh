#!/bin/bash
# Verify both Llama and GLM work after model-agnostic changes
# Usage: ./scripts/verify-models.sh

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

function log_info() { echo -e "${BLUE}[TEST]${NC} $1"; }
function log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
function log_error() { echo -e "${RED}[FAIL]${NC} $1"; }

BACKEND_URL="http://127.0.0.1:8100"
TEST_QUERY='{"transcript":"show pump 1 vibration"}'

echo ""
echo "=========================================="
echo "  Model Verification Tests"
echo "=========================================="
echo ""

# Check if backend is running
log_info "Checking if backend is running..."
if ! curl -s "$BACKEND_URL/api/layer2/health/" > /dev/null 2>&1; then
    log_error "Backend not responding. Start it first:"
    echo "  cd backend && python manage.py runserver 8100"
    exit 1
fi
log_success "Backend is running"

# Test Llama
echo ""
log_info "Testing Llama 3.1 8B..."
./scripts/swap-model.sh llama > /dev/null 2>&1
sleep 2

LLAMA_RESPONSE=$(curl -s -X POST "$BACKEND_URL/api/layer2/orchestrate/" \
    -H "Content-Type: application/json" \
    -d "$TEST_QUERY" 2>/dev/null || echo "")

if echo "$LLAMA_RESPONSE" | grep -q "widgets"; then
    log_success "Llama returned valid widget selection"
    LLAMA_TIME=$(echo "$LLAMA_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('response_time_ms', 0))" 2>/dev/null || echo "unknown")
    echo "         Response time: ${LLAMA_TIME}ms"
else
    log_error "Llama failed to return widgets"
    echo "$LLAMA_RESPONSE" | head -5
fi

# Test GLM
echo ""
log_info "Testing GLM-4.7-Flash..."
./scripts/swap-model.sh glm > /dev/null 2>&1
sleep 2

GLM_RESPONSE=$(curl -s -X POST "$BACKEND_URL/api/layer2/orchestrate/" \
    -H "Content-Type: application/json" \
    -d "$TEST_QUERY" 2>/dev/null || echo "")

if echo "$GLM_RESPONSE" | grep -q "widgets"; then
    log_success "GLM returned valid widget selection"
    GLM_TIME=$(echo "$GLM_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('response_time_ms', 0))" 2>/dev/null || echo "unknown")
    echo "         Response time: ${GLM_TIME}ms"
else
    log_error "GLM failed to return widgets"
    echo "$GLM_RESPONSE" | head -5
fi

# Summary
echo ""
echo "=========================================="
echo "  Verification Complete"
echo "=========================================="
echo ""

if echo "$LLAMA_RESPONSE" | grep -q "widgets" && echo "$GLM_RESPONSE" | grep -q "widgets"; then
    log_success "Both models working! Hot-swap ready."
    echo ""
    echo "You can now switch models anytime:"
    echo "  ./scripts/swap-model.sh llama  # Fast (sub-second)"
    echo "  ./scripts/swap-model.sh glm    # Accurate (5-8 seconds)"
else
    log_error "One or both models failed. Check logs:"
    echo "  tail -f /tmp/gunicorn-error.log"
fi
echo ""
