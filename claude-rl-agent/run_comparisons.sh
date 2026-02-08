#!/bin/bash
# Automated Comparison Runner
# Runs Claude vs LLaMA comparisons with 35-dimensional analysis

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  📊 Automated Claude vs LLaMA Comparison System                   ║"
echo "║  35-Dimensional Behavioral Analysis                               ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if model exists
if ! ollama list | grep -q "cc-claude-agent"; then
    echo "❌ Error: Model 'cc-claude-agent:latest' not found in Ollama"
    echo "Please deploy the model first: ./auto_deploy.sh"
    exit 1
fi

echo "✅ Model 'cc-claude-agent:latest' found"
echo ""

# Get number of comparisons (default: 10)
BATCH_SIZE=${1:-10}

echo "Configuration:"
echo "  Batch size: $BATCH_SIZE comparisons"
echo "  Model: cc-claude-agent:latest"
echo "  Output: data/comparison_log.jsonl"
echo "  DPO pairs: data/dpo_pairs.jsonl"
echo ""

# Run comparisons
echo "🚀 Starting automated comparisons..."
echo ""

cd src
python automated_runner.py --batch $BATCH_SIZE

if [ $? -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  ✅ COMPARISONS COMPLETE!                                          ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""

    # Show statistics
    echo "📊 Results Summary:"
    echo ""

    COMPARISON_COUNT=$(wc -l < ../data/comparison_log.jsonl 2>/dev/null || echo "0")
    DPO_COUNT=$(wc -l < ../data/dpo_pairs.jsonl 2>/dev/null || echo "0")

    echo "  Total comparisons: $COMPARISON_COUNT"
    echo "  DPO pairs generated: $DPO_COUNT"

    if [ $COMPARISON_COUNT -gt 0 ]; then
        echo ""
        echo "  Average behavioral similarity:"
        python3 -c "
import json
try:
    sims = []
    with open('../data/comparison_log.jsonl', 'r') as f:
        for line in f:
            comp = json.loads(line)
            sims.append(comp['behavioral_comparison']['overall_similarity'])
    if sims:
        print(f'    {sum(sims)/len(sims):.1%} (min: {min(sims):.1%}, max: {max(sims):.1%})')
except Exception as e:
    print(f'    Error: {e}')
"
    fi

    echo ""
    echo "📁 Output files:"
    echo "  - Comparisons: data/comparison_log.jsonl"
    echo "  - DPO pairs: data/dpo_pairs.jsonl"
    echo ""

    if [ $DPO_COUNT -ge 50 ]; then
        echo "🎯 You have $DPO_COUNT DPO pairs - ready for DPO training!"
        echo "   Run: ./run.sh train --phase dpo --pairs 50"
        echo ""
    fi

    echo "Next steps:"
    echo "  - View comparison: tail -n 1 data/comparison_log.jsonl | jq ."
    echo "  - View DPO pair: tail -n 1 data/dpo_pairs.jsonl | jq ."
    echo "  - Run more comparisons: ./run_comparisons.sh <batch_size>"
    echo ""
else
    echo "❌ Comparisons failed!"
    exit 1
fi
