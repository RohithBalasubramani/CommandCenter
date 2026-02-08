#!/bin/bash
# Deployment Readiness Checker
# Verifies all prerequisites for deployment and operation

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  🔍 Deployment Readiness Check                                     ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

READY=true

# 1. Check exported GGUF model
echo "1. Checking GGUF Model Export..."
if [ -f "data/exports/cc-claude-agent.gguf" ]; then
    SIZE=$(du -h data/exports/cc-claude-agent.gguf | cut -f1)
    echo "   ✅ GGUF model found: $SIZE"
else
    echo "   ❌ GGUF model not found"
    echo "      → Run: python merge_and_export.py"
    READY=false
fi
echo ""

# 2. Check Ollama installation
echo "2. Checking Ollama..."
if command -v ollama &> /dev/null; then
    VERSION=$(ollama --version 2>&1 | head -n1)
    echo "   ✅ Ollama installed: $VERSION"
else
    echo "   ❌ Ollama not installed"
    echo "      → Install: curl -fsSL https://ollama.com/install.sh | sh"
    READY=false
fi
echo ""

# 3. Check Ollama service
echo "3. Checking Ollama Service..."
if systemctl is-active --quiet ollama 2>/dev/null; then
    echo "   ✅ Ollama service running"
elif pgrep -x ollama > /dev/null; then
    echo "   ✅ Ollama process running"
else
    echo "   ⚠️  Ollama not running"
    echo "      → Start: sudo systemctl start ollama"
    echo "      → Or: ollama serve &"
fi
echo ""

# 4. Check GPU availability
echo "4. Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
    echo "   ✅ GPU: $GPU_NAME"
    echo "      Free Memory: ${GPU_MEM} MB"

    if [ "$GPU_MEM" -lt 8000 ]; then
        echo "      ⚠️  Low GPU memory (<8GB free)"
        echo "      → Free memory: pkill ollama; sleep 2"
    fi
else
    echo "   ⚠️  nvidia-smi not found (CPU mode only)"
fi
echo ""

# 5. Check Python dependencies
echo "5. Checking Python Dependencies..."
MISSING=()

if ! python -c "import unsloth" 2>/dev/null; then
    MISSING+=("unsloth")
fi

if ! python -c "import ollama" 2>/dev/null; then
    MISSING+=("ollama")
fi

if ! python -c "import scipy" 2>/dev/null; then
    MISSING+=("scipy")
fi

if [ ${#MISSING[@]} -eq 0 ]; then
    echo "   ✅ All Python dependencies installed"
else
    echo "   ❌ Missing dependencies: ${MISSING[*]}"
    echo "      → Install: pip install ${MISSING[*]}"
    READY=false
fi
echo ""

# 6. Check data directories
echo "6. Checking Data Directories..."
DIRS=("data/exports" "data/dpo_pairs" "data/traces" "data/models")
ALL_EXIST=true

for DIR in "${DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        echo "   ✅ $DIR exists"
    else
        echo "   ⚠️  $DIR missing (will be created)"
        mkdir -p "$DIR"
    fi
done
echo ""

# 7. Check test prompts
echo "7. Checking Test Configuration..."
if [ -f "test_prompts.json" ]; then
    COUNT=$(jq '.test_prompts | length' test_prompts.json 2>/dev/null || echo "?")
    echo "   ✅ Test prompts: $COUNT prompts"
else
    echo "   ⚠️  test_prompts.json not found"
fi
echo ""

# 8. Check scripts executable
echo "8. Checking Scripts..."
SCRIPTS=("auto_deploy.sh" "run_comparisons.sh" "dashboard.sh" "validate_model.py" "analyze_comparisons.py")

for SCRIPT in "${SCRIPTS[@]}"; do
    if [ -f "$SCRIPT" ]; then
        if [ -x "$SCRIPT" ]; then
            echo "   ✅ $SCRIPT (executable)"
        else
            echo "   ⚠️  $SCRIPT (not executable)"
            chmod +x "$SCRIPT"
            echo "      → Fixed: chmod +x $SCRIPT"
        fi
    else
        echo "   ❌ $SCRIPT not found"
        READY=false
    fi
done
echo ""

# 9. Check llama.cpp
echo "9. Checking llama.cpp..."
if [ -d "$HOME/llama.cpp" ]; then
    echo "   ✅ llama.cpp found: $HOME/llama.cpp"

    if [ -f "$HOME/llama.cpp/build/bin/llama-quantize" ]; then
        echo "   ✅ llama-quantize binary found"
    else
        echo "   ⚠️  llama-quantize not found (needed for manual export)"
    fi
else
    echo "   ⚠️  llama.cpp not found (needed for manual export)"
fi
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  READINESS SUMMARY                                                 ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

if [ "$READY" = true ]; then
    echo "✅ ALL CRITICAL CHECKS PASSED"
    echo ""
    echo "Next Steps:"
    echo "  1. Deploy model: ./auto_deploy.sh"
    echo "  2. Validate model: python validate_model.py"
    echo "  3. Run comparisons: ./run_comparisons.sh 10"
    echo "  4. Analyze results: python analyze_comparisons.py"
    echo "  5. Monitor system: ./dashboard.sh"
    echo ""
    exit 0
else
    echo "❌ SOME CHECKS FAILED"
    echo ""
    echo "Please fix the issues above before deployment."
    echo ""
    exit 1
fi
