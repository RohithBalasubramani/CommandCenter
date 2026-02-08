#!/bin/bash
# Manual GGUF Export
# Uses existing llama.cpp installation

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  📦 Manual GGUF Export (using existing llama.cpp)                  ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

MODEL_PATH="data/models/sft_checkpoints/claude-bc-20260208_055059/final"
OUTPUT_DIR="data/exports"
LLAMA_CPP="$HOME/llama.cpp"

# Check inputs
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: SFT model not found at $MODEL_PATH"
    exit 1
fi

if [ ! -d "$LLAMA_CPP" ]; then
    echo "❌ Error: llama.cpp not found at $LLAMA_CPP"
    exit 1
fi

echo "✅ Found SFT model: $MODEL_PATH"
echo "✅ Found llama.cpp: $LLAMA_CPP"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Convert to GGUF f16
echo "Step 1/2: Converting to GGUF f16..."
python "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$MODEL_PATH" \
    --outfile "$OUTPUT_DIR/cc-claude-agent-f16.gguf" \
    --outtype f16

if [ $? -ne 0 ]; then
    echo "❌ Conversion failed!"
    exit 1
fi

echo "✅ F16 GGUF created"
ls -lh "$OUTPUT_DIR/cc-claude-agent-f16.gguf"
echo ""

# Step 2: Quantize to q4_k_m
echo "Step 2/2: Quantizing to q4_k_m..."
"$LLAMA_CPP/build/bin/llama-quantize" \
    "$OUTPUT_DIR/cc-claude-agent-f16.gguf" \
    "$OUTPUT_DIR/cc-claude-agent.gguf" \
    q4_k_m

if [ $? -ne 0 ]; then
    echo "❌ Quantization failed!"
    exit 1
fi

echo "✅ Quantized GGUF created"
ls -lh "$OUTPUT_DIR/cc-claude-agent.gguf"
echo ""

# Cleanup f16 file (optional)
echo "Removing intermediate f16 file..."
rm "$OUTPUT_DIR/cc-claude-agent-f16.gguf"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ EXPORT COMPLETE!                                               ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Output: $OUTPUT_DIR/cc-claude-agent.gguf"
ls -lh "$OUTPUT_DIR/cc-claude-agent.gguf"
echo ""
echo "Next steps:"
echo "  1. Deploy to Ollama: ./auto_deploy.sh"
echo "  2. Test model: ollama run cc-claude-agent '<query>'"
echo "  3. Run comparisons: ./run_comparisons.sh 10"
echo ""
