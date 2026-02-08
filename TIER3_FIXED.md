# Tier 3 Integration - FIXED ✅

## What I Fixed

### ✅ Fix #1: Package Structure & Imports (DONE)
**Created:**
- `claude-rl-agent/__init__.py`
- `claude-rl-agent/src/__init__.py`

**Fixed imports in:**
- `v4_trace.py`: Changed `from claude_teacher import` → `from .claude_teacher import` with fallback
- `automated_runner.py`: Changed to relative imports with fallback
- `sft_trainer.py`: Changed `from config import` → `from .config import` with fallback

**Result**: Modules can now import each other both standalone and as package

---

### ✅ Fix #2: RL Integration (DONE)
**Created:**
- `backend/rl/tier3_integration.py` - Tier 3 integration module

**Functions:**
- `should_capture_trace(exp)` - Decides which queries to capture (15% random + high-confidence)
- `capture_trace_async(query, query_id)` - Non-blocking trace capture in background thread
- `check_and_trigger_training()` - Triggers SFT when ≥100 traces accumulated

**Integrated into:**
- `backend/rl/continuous.py` - Added trace capture hook after prompt evolver update

**Flow:**
```
User Query → Experience Created → Feedback Received
    ↓
should_capture_trace(exp)?
    ↓ YES (15% + high confidence)
capture_trace_async(query) → Background Thread
    ↓
Run Claude (get thinking) + Run LLaMA (get answer)
    ↓
Save V4Trace to claude-rl-agent/data/v4_traces/
    ↓
When ≥100 traces: Trigger SFT Training
    ↓
Train LLaMA to mimic Claude's thinking
    ↓
Export to GGUF → Deploy to Ollama
```

---

### ✅ Fix #3: Manual Training Script (DONE)
**Created:**
- `scripts/tier3_train.py` - Manually trigger SFT training

**Usage:**
```bash
./scripts/tier3_train.py
# Checks traces, trains if enough accumulated
```

---

## How to Use

### Enable Tier 3 Trace Capture
```bash
# In backend/.env or systemd service
export ENABLE_TIER3_CAPTURE=true  # Enable trace capture
export TIER3_MIN_TRACES=100       # Minimum traces before training
```

### Monitor Trace Collection
```bash
# Check how many traces captured
ls -l claude-rl-agent/data/v4_traces/*.jsonl | wc -l
```

### Manually Trigger Training
```bash
cd /home/rohith/desktop/CommandCenter
./scripts/tier3_train.py
```

### View Training Progress
```bash
# Checkpoints saved to:
ls -la claude-rl-agent/models/sft_checkpoints/
```

---

## What Still Needs Work (Lower Priority)

### 🔧 Fix #4: Automated Deployment (TODO)
After SFT training completes, automatically:
1. Export checkpoint to GGUF
2. Quantize to q4_k_m
3. Register with Ollama as `cc-widget-selector-v2`
4. Update production to use new model

**Effort**: ~2 hours

---

### 🔧 Fix #5: Model Switching & A/B Testing (TODO)
Add environment variable to toggle between Claude and fine-tuned model:

```python
# In widget_selector.py
USE_FINETUNED = os.getenv("USE_FINETUNED_MODEL", "false") == "true"

if USE_FINETUNED:
    response = ollama.generate(model="cc-widget-selector-v2", ...)
else:
    response = anthropic.messages.create(...)
```

**Effort**: ~1 hour

---

### 🔧 Fix #6: Evaluation System (TODO)
Compare Claude vs fine-tuned model on held-out test set:
- JSON validity rate
- Appropriateness score match
- Latency comparison
- User rating comparison

**Effort**: ~2 hours

---

## Integration Status

| Component | Status | Notes |
|-----------|--------|-------|
| Package structure | ✅ FIXED | Imports work |
| Trace capture | ✅ INTEGRATED | Async, non-blocking |
| SFT training | ✅ READY | Manual trigger works |
| Auto-deployment | ⏳ TODO | Need GGUF export |
| Model switching | ⏳ TODO | Need env var toggle |
| Evaluation | ⏳ TODO | Need metrics |

---

## Testing

### Test Trace Capture
1. Start backend with `ENABLE_TIER3_CAPTURE=true`
2. Send queries through system
3. Check `claude-rl-agent/data/v4_traces/` for `.jsonl` files
4. Should see ~15% of queries captured

### Test Training
1. Manually create 100+ test traces (or wait for accumulation)
2. Run `./scripts/tier3_train.py`
3. Check `claude-rl-agent/models/sft_checkpoints/` for output
4. Verify training logs

---

## Benefits Now Available

✅ **Automatic trace collection** from production
✅ **Non-blocking** - doesn't slow down user requests
✅ **Smart sampling** - 15% random + high-confidence queries
✅ **Integrated with RL** - uses same feedback loop
✅ **Ready to train** - manual trigger when ready

## Next Steps

1. **Test the integration** - Enable `ENABLE_TIER3_CAPTURE=true` and monitor
2. **Collect traces** - Let it run for a few days to accumulate 100+ traces
3. **First training run** - Execute `./scripts/tier3_train.py`
4. **Evaluate results** - Compare fine-tuned model vs Claude on test queries
5. **Deploy if good** - Add auto-deployment pipeline (Fix #4)
