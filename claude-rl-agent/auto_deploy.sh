#!/bin/bash
# Automatic Deployment Script
# Runs after GGUF export completes

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  🚀 Automatic Deployment to Ollama                                ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if export file exists
EXPORT_FILE="./data/exports/cc-claude-agent.gguf"
if [ ! -f "$EXPORT_FILE" ]; then
    echo "❌ Error: Export file not found: $EXPORT_FILE"
    echo "Please wait for export to complete first."
    exit 1
fi

echo "✅ Found export file: $EXPORT_FILE"
ls -lh "$EXPORT_FILE"
echo ""

# Create Modelfile
echo "📝 Creating Modelfile..."
cat > Modelfile <<'EOF'
FROM ./data/exports/cc-claude-agent.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM """You are a helpful AI assistant specializing in Command Center industrial equipment monitoring. You have deep knowledge of:
- Industrial equipment (chillers, transformers, DG sets, motors, pumps, AHUs, cooling towers)
- Database schemas for equipment monitoring (357 tables)
- SQL queries for equipment data analysis
- Anomaly detection and predictive maintenance
- Energy optimization and efficiency analysis

When answering questions:
1. Explore the database schema when needed
2. Write appropriate SQL queries
3. Provide clear, actionable insights
4. Consider constraints and edge cases
5. Cite your sources (files, queries, etc.)
"""
EOF

echo "✅ Modelfile created"
echo ""

# Remove old model if exists
echo "🗑️  Removing old model (if exists)..."
ollama rm cc-claude-agent:latest 2>/dev/null || echo "No old model to remove"
echo ""

# Create model in Ollama
echo "📦 Creating model in Ollama..."
ollama create cc-claude-agent:latest -f Modelfile

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Model created successfully!"
    echo ""

    # Verify
    echo "🔍 Verifying model..."
    ollama list | grep cc-claude-agent
    echo ""

    # Quick test
    echo "🧪 Running quick test..."
    echo ""
    echo "Prompt: 'What tables store equipment data?'"
    echo "Response:"
    echo "─────────────────────────────────────────────────────────────────────"
    ollama run cc-claude-agent "What tables store equipment data in the Command Center database?" --verbose 2>&1 | head -n 20
    echo "─────────────────────────────────────────────────────────────────────"
    echo ""

    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  ✅ DEPLOYMENT COMPLETE!                                           ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Model: cc-claude-agent:latest"
    echo "Status: Ready for use"
    echo ""
    echo "Next steps:"
    echo "  1. Run test: ollama run cc-claude-agent '<your question>'"
    echo "  2. Run comparisons: cd src && python automated_runner.py --batch 10"
    echo "  3. Check DPO pairs: cat data/dpo_pairs.jsonl"
    echo ""
else
    echo "❌ Model creation failed!"
    exit 1
fi
