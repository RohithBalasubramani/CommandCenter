# 🚀 Deployment & Testing Guide

**Date**: 2026-02-08
**Status**: Export in progress → Deploy → Test → Compare

---

## Current Status

### ✅ Step 1: Export SFT Model (IN PROGRESS)
**Command**: `python src/agent.py export`
**Status**: Loading model (15-20 minutes expected)
**Output**: `/tmp/claude-1000/-home-rohith-desktop-CommandCenter/tasks/bc91f72.output`

**What it does**:
1. Loads SFT model: `data/models/sft_checkpoints/claude-bc-20260208_055059/final/`
2. Merges LoRA adapters (161MB) with base LLaMA 3.1 8B
3. Converts to GGUF format (f16)
4. Quantizes to q4_k_m (~4.7GB)
5. Saves to: `data/exports/cc-claude-agent.gguf`

---

## ⏳ Step 2: Deploy to Ollama (READY)

Once export completes, deploy:

```bash
cd /home/rohith/desktop/CommandCenter/claude-rl-agent

# Create Modelfile
cat > Modelfile <<EOF
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
EOF

# Create model in Ollama
ollama create cc-claude-agent:latest -f Modelfile

# Verify
ollama list | grep cc-claude-agent
```

**Expected output**:
```
cc-claude-agent:latest    4.7 GB    2 minutes ago
```

---

## 🧪 Step 3: Test Deployed Model (READY)

### Quick Test:
```bash
ollama run cc-claude-agent "What's the average power consumption of chiller_001?"
```

**Expected behavior**:
- Should understand Command Center database context
- Should suggest appropriate SQL queries
- Should mention equipment tables (chillers, transformers, etc.)
- Response time: ~2-5 seconds

### Comprehensive Test:
```bash
cd /home/rohith/desktop/CommandCenter/claude-rl-agent/src

# Test enhanced extraction on deployed model
python -c "
from automated_runner import AutomatedRunner

runner = AutomatedRunner(llama_model='cc-claude-agent:latest')

# Test LLaMA response
prompt = 'What tables store chiller data in the Command Center database?'
response, duration = runner.run_llama(prompt)

print(f'Prompt: {prompt}')
print(f'Duration: {duration:.2f}s')
print(f'Response: {response[:500]}...')
"
```

---

## 📊 Step 4: Run Automated Comparison (READY)

### Single Comparison Test:
```bash
cd /home/rohith/desktop/CommandCenter/claude-rl-agent/src

python automated_runner.py --batch 1 --prompt "What's the difference between chiller_001 and chiller_002?"
```

**What happens**:
1. Runs prompt through Claude CLI (automated)
2. Runs same prompt through cc-claude-agent (Ollama)
3. Extracts 35-dimensional reasoning vectors from both
4. Compares:
   - Tool sequences
   - Reasoning steps
   - Assumptions
   - Validations
   - Counterfactuals
   - Provenance
   - Safety signals
   - Self-critique
   - 35-dim behavioral vectors
5. Generates DPO pair if divergent
6. Saves comparison to `data/comparison_log.jsonl`
7. Saves DPO pair to `data/dpo_pairs.jsonl`

**Expected output**:
```
╔════════════════════════════════════════════════════════════════════╗
║  🔥 Automated Claude vs LLaMA Comparison                          ║
╚════════════════════════════════════════════════════════════════════╝

Prompt: "What's the difference between chiller_001 and chiller_002?"

⏱  Execution Time:
  Claude: 12.45s | LLaMA: 3.21s

🔧 Tool Sequence:
  Claude: Bash → Read(models.py) → Bash(psql query) → Response
  LLaMA:  Bash → Response
  Match:  DIFFERENT

🧠 Reasoning Depth:
  Claude: 5 steps
  LLaMA:  2 steps
  Status: DIFFERENT

🚧 Constraint Detection:
  Claude: 2 constraints
  LLaMA:  0 constraints
  Status: MISSING

🔄 Self-Correction:
  Claude: 1 corrections
  LLaMA:  0 corrections

🔍 Exploration Depth:
  Claude: thorough
  LLaMA:  minimal
  Status: DIFFERENT

📊 ENHANCED SIGNALS:

Assumptions:
  Claude: 3 assumptions (clarity: 0.8)
  LLaMA:  1 assumptions (clarity: 0.3)
  Status: MISSING

Validation Checks:
  Claude: 2 validations (completeness: 0.9)
  LLaMA:  0 validations (completeness: 0.0)
  Status: MISSING

Counterfactual Paths:
  Claude: 1 alternative approaches
  LLaMA:  0 alternative approaches
  Status: MISSING

Provenance (Source Citations):
  Claude: 4 sources (Read: models.py, Bash: psql, ...)
  LLaMA:  1 sources
  Status: MISSING

Safety Signals:
  Claude: 0 concerns
  LLaMA:  0 concerns
  Status: ALIGNED

Self-Critique:
  Claude confidence: 0.85
  LLaMA confidence: 0.50
  Status: DIFFERENT

📐 REASONING VECTOR COMPARISON (35 dimensions):
  Cosine similarity: 0.62
  Euclidean similarity: 0.58
  Average difference: 0.21
  Max difference: 0.65

  Critical Divergences (> 0.3):
    - num_assumptions_made: Claude=3.0, LLaMA=1.0, diff=2.0
    - validation_completeness_score: Claude=0.9, LLaMA=0.0, diff=0.9
    - provenance_citations: Claude=4.0, LLaMA=1.0, diff=3.0
    - reasoning_depth_score: Claude=0.85, LLaMA=0.40, diff=0.45

📈 OVERALL SIMILARITY:
  Base signals (40%): 0.55
  Enhanced signals (60%): 0.48
  TOTAL: 0.51 (51%)

────────────────────────────────────────────────────────────────────
🎯 TRAINING NEEDED
   Reasons:
   • Tool sequence mismatch
   • Reasoning depth differs
   • Assumption clarity differs
   • Validation completeness differs
   • Counterfactual consideration missing
   • Provenance tracking differs
   • Reasoning vector divergence (similarity: 0.62)
────────────────────────────────────────────────────────────────────

✅ DPO pair saved for training

Comparison saved to: data/comparison_log.jsonl
DPO pair saved to: data/dpo_pairs.jsonl
```

---

## 🔄 Step 5: Continuous Improvement Loop (ACTIVATE)

### Run Batch Comparisons:
```bash
cd /home/rohith/desktop/CommandCenter/claude-rl-agent/src

# Run 10 comparisons with Command Center prompts
python automated_runner.py --batch 10
```

**Command Center prompts** (examples):
1. "What's the average power consumption of all chillers?"
2. "Find equipment with temperature anomalies in the last 24 hours"
3. "Compare maintenance costs between transformer_001 and transformer_002"
4. "Which DG set has the highest runtime this month?"
5. "Show me the schema for the equipment table"
6. "What's the relationship between chillers and cooling_towers tables?"
7. "Find all pumps with efficiency below 85%"
8. "Calculate total energy consumption for February"
9. "List equipment due for maintenance this week"
10. "What's the uptime percentage for critical equipment?"

**Results**:
- 10 comparisons → `data/comparison_log.jsonl`
- N DPO pairs (where divergent) → `data/dpo_pairs.jsonl`
- Each with full 35-dimensional analysis

### Check Results:
```bash
# Count total comparisons
wc -l data/comparison_log.jsonl

# Count DPO pairs generated
wc -l data/dpo_pairs.jsonl

# View sample DPO pair
head -n 1 data/dpo_pairs.jsonl | jq .
```

---

## 📈 Step 6: Analyze & Iterate (CONTINUOUS)

### View Comparison Statistics:
```bash
cd /home/rohith/desktop/CommandCenter/claude-rl-agent/src

python -c "
import json

# Load comparisons
comparisons = []
with open('../data/comparison_log.jsonl', 'r') as f:
    for line in f:
        comparisons.append(json.loads(line))

# Compute statistics
similarities = [c['behavioral_comparison']['overall_similarity'] for c in comparisons]
avg_similarity = sum(similarities) / len(similarities)

print(f'Total comparisons: {len(comparisons)}')
print(f'Average behavioral similarity: {avg_similarity:.2%}')
print(f'Min similarity: {min(similarities):.2%}')
print(f'Max similarity: {max(similarities):.2%}')
print(f'DPO pairs needed: {sum(1 for c in comparisons if c[\"should_train\"])}')
"
```

### When to Retrain:
```bash
# When you have 50+ DPO pairs
wc -l data/dpo_pairs.jsonl

# If >= 50, run DPO training
cd /home/rohith/desktop/CommandCenter/claude-rl-agent
./run.sh train --phase dpo --pairs 50
```

**DPO Training** (when ready):
- Uses divergent pairs to improve LLaMA
- Trains to prefer Claude's behavioral patterns
- Takes ~1-2 hours
- Re-export and re-deploy after training
- Similarity should increase (60% → 70% → 80%+)

---

## 🎯 Success Metrics

### After First Deployment:
- ✅ Model deploys successfully to Ollama
- ✅ Responds to Command Center queries
- ✅ Automated comparison runs
- ✅ DPO pairs are generated
- **Expected similarity**: 50-60% (SFT baseline)

### After Week 1 (50+ comparisons):
- ✅ 50+ DPO pairs collected
- ✅ Run DPO training
- ✅ Re-deploy improved model
- **Expected similarity**: 65-75%

### After Week 2 (100+ comparisons):
- ✅ 100+ total comparisons
- ✅ Second DPO training round
- **Expected similarity**: 75-85%

### After Month 1 (500+ comparisons):
- ✅ Continuous DPO training
- ✅ Multiple iterations
- **Expected similarity**: 85-95% (near Claude-level)

---

## 📝 Quick Reference Commands

```bash
# Check export status
tail -f /tmp/claude-1000/-home-rohith-desktop-CommandCenter/tasks/bc91f72.output

# Deploy to Ollama (after export completes)
cd /home/rohith/desktop/CommandCenter/claude-rl-agent
ollama create cc-claude-agent:latest -f Modelfile

# Test model
ollama run cc-claude-agent "Test query"

# Run 1 comparison
cd src && python automated_runner.py --batch 1

# Run 10 comparisons
cd src && python automated_runner.py --batch 10

# Check DPO pairs
wc -l data/dpo_pairs.jsonl
head -n 1 data/dpo_pairs.jsonl | jq .

# When ready for DPO training (50+ pairs)
./run.sh train --phase dpo --pairs 50
```

---

**Current Step**: Waiting for export to complete (~15-20 minutes total)
**Next**: Deploy to Ollama → Test → Run comparisons → Collect DPO pairs
**Status**: ✅ ON TRACK FOR COMPLETE DEPLOYMENT
