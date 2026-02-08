# Tier 3 (V4 Reasoning Distillation) - Integration Audit & Plan

## Current State

### ✅ What Exists (Well-Implemented)
1. **`claude_teacher.py`** - Captures Claude thinking via CLI
2. **`v4_trace.py`** - Stores traces with 3 SFT samples per trace
3. **`sft_trainer.py`** - Curriculum phased training (answer → thinking → consistency)
4. **`automated_runner.py`** - Runs prompts through both Claude and LLaMA

### ❌ What's Missing (Not Integrated)
1. **No automatic trace capture** - Must run manually, not triggered by production queries
2. **No connection to RL feedback loop** - Tier 3 is completely standalone
3. **No deployment pipeline** - Trained models don't replace Claude automatically
4. **No evaluation** - Can't measure if SFT actually improves performance
5. **Import issues** - Modules need proper package structure

##Issues Found

### Bug #1: Module Import Structure
```python
# v4_trace.py tries:
from claude_teacher import ClaudeThinking

# But when run from project root, this fails
# Need: from src.claude_teacher import ClaudeThinking
```

### Bug #2: No Automatic Triggering
- `automated_runner.py` must be run manually
- Production queries don't trigger trace capture
- No connection to `continuous.py` feedback loop

### Bug #3: No Model Deployment
- SFT training produces checkpoints
- But nothing deploys them to replace Claude
- No A/B testing between Claude vs fine-tuned model

### Bug #4: Prompt Parity Risk
- `build_full_prompt()` in `claude_teacher.py` tries to import from backend
- Falls back to file read if import fails
- Could drift from production prompts over time

## Integration Plan

### Phase 1: Fix Imports & Package Structure
```bash
# Add __init__.py
touch claude-rl-agent/src/__init__.py
touch claude-rl-agent/__init__.py

# Fix imports to be package-relative
# v4_trace.py: from .claude_teacher import ClaudeThinking
# automated_runner.py: from .v4_trace import V4TraceStore
```

### Phase 2: Connect to RL Feedback Loop
Add trace capture hook in `backend/rl/continuous.py`:

```python
# After experience is saved, optionally capture trace
if should_capture_trace(exp):  # Random sampling or quality threshold
    from claude_rl_agent.src.automated_runner import AutomatedRunner
    runner = AutomatedRunner()
    runner.run_parallel_comparison(exp.original_query)
```

### Phase 3: Automated SFT Training
Add background trainer in `backend/rl/`:

```python
# tier3_sft.py
class Tier3SFTTrainer:
    def check_and_train(self):
        """Trigger SFT when enough traces accumulated."""
        store = V4TraceStore()
        traces = store.load_all()

        if len(traces) >= MIN_TRACES_FOR_TRAINING:
            sft_trainer = SFTTrainer(config)
            sft_trainer.train(dataset_path=...)

            # Export to GGUF
            export_to_ollama(checkpoint_path, model_name="cc-widget-selector-v2")
```

### Phase 4: Model Deployment & Evaluation
```python
# Switch between Claude and fine-tuned model
USE_FINETUNED_MODEL = os.getenv("USE_FINETUNED_MODEL", "false") == "true"

if USE_FINETUNED_MODEL:
    # Use Ollama with fine-tuned model
    response = ollama_client.generate(model="cc-widget-selector-v2", ...)
else:
    # Use Claude
    response = anthropic_client.messages.create(...)
```

## Recommended Fixes (Priority Order)

### 1. Fix Package Structure (30 min)
- Add `__init__.py` files
- Fix all imports to be package-relative
- Test that modules can import each other

### 2. Add Trace Capture Hook (1 hour)
- Modify `continuous.py` to trigger trace capture
- Sample 10-20% of production queries
- Save traces to `claude-rl-agent/data/v4_traces/`

### 3. Connect SFT Training (2 hours)
- Create `backend/rl/tier3_sft.py`
- Check for new traces periodically (e.g., daily cron)
- Trigger training when ≥100 new traces
- Export to GGUF automatically

### 4. Add Model Switching (1 hour)
- Environment variable to toggle Claude vs fine-tuned
- A/B test: 50% Claude, 50% fine-tuned
- Compare performance metrics

### 5. Add Evaluation (2 hours)
- Compare fine-tuned model vs Claude on held-out test set
- Measure: JSON validity, appropriateness scores, latency
- Dashboard showing Tier 3 training progress

## Expected Benefits After Integration

✅ Traces captured automatically from production
✅ SFT training triggers when enough data accumulated
✅ Fine-tuned models deployed automatically
✅ Can gradually replace Claude with local model
✅ Cost reduction: $0.03/1M tokens (Claude) → free (local)
✅ Latency improvement: ~2s (Claude API) → ~500ms (local GPU)

## Risk Mitigation

- **Quality drops**: Keep Claude as fallback, switch back if scores drop
- **Prompt drift**: Validate prompt parity in CI/CD
- **Training instability**: Checkpoint frequently, validate before deployment
- **Resource usage**: Only train when GPU idle, limit to off-peak hours

---

**Next Step**: Decide which fixes to prioritize. I recommend starting with #1 (package structure) and #2 (trace capture hook) to get the integration flowing.
