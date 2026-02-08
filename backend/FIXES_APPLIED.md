# RL Training Fixes - Successfully Applied
**Date**: 2026-02-08
**Status**: ✅ ALL FIXES VERIFIED AND WORKING

---

## Summary

All critical bugs identified in the audit have been fixed and verified working:

### ✅ Fix 1: Tier 1 Rank Reading from Config

**Problem**: Scorer trained with rank-8 instead of configured rank-32

**Fix Applied**: [lora_scorer.py:726-738](/home/rohith/desktop/CommandCenter/backend/rl/lora_scorer.py#L726-L738)
```python
def get_scorer() -> ContinuousLowRankTrainer:
    """Get the global low-rank scorer instance."""
    global _scorer
    if _scorer is None:
        with _scorer_lock:
            if _scorer is None:
                from .config import CONTINUOUS_RL_CONFIG
                _scorer = ContinuousLowRankTrainer(
                    rank=CONTINUOUS_RL_CONFIG.get("scorer_rank", 8),
                    lr=CONTINUOUS_RL_CONFIG.get("scorer_lr", 1e-3),
                    checkpoint_every=CONTINUOUS_RL_CONFIG.get("scorer_checkpoint_every", 100),
                )
    return _scorer
```

**Verification**:
```
✅ Config scorer_rank: 32
✅ Scorer actual rank: 32
✅ Scorer params: 28,193 (was 7,145)
✅ Checkpoint: 8.2MB at rl_checkpoints/scorer/scorer_latest.pt
```

**Impact**: Tier 1 now has full rank-32 capacity for better query-specificity

---

### ✅ Fix 2: Tier 3 Trace File Reading

**Problem**: Training script looked for `*.json` files but traces stored in `traces.jsonl`

**Fix Applied**: [train_all_tiers.py:147-166](/home/rohith/desktop/CommandCenter/backend/train_all_tiers.py#L147-L166)
```python
# Read traces from traces.jsonl file
trace_file = trace_dir / 'traces.jsonl'
traces = []
if trace_file.exists():
    import json
    with open(trace_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in traces.jsonl")
```

**Verification**:
```
Before: Found 0 thinking traces  ❌
After:  Found 2 thinking traces  ✅
  Trace 1: Has thinking block ✅
  Trace 2: Direct answer only
```

**Impact**: Tier 3 can now detect existing traces for SFT training

**Note**: BackgroundTrainer production code already uses `V4TraceStore.load_all()` which reads correctly from `traces.jsonl`, so no production code changes needed.

---

## Training Results (Post-Fix)

### Tier 1: ✅ COMPLETED with Rank-32
- Architecture: 32-rank, 28,193 parameters
- Trained: 730 experiences with feedback
- Training steps: 12,738
- Checkpoint: rl_checkpoints/scorer/scorer_latest.pt (8.2MB)

### Tier 2: ⏸ INSUFFICIENT DATA (Working as Designed)
- DPO pairs: 15/80 (13 widget, 2 voice)
- Bottleneck: Strict intent similarity (prevents noisy pairs)
- Status: Accumulating more diverse queries

### Tier 3: ⏸ INSUFFICIENT DATA (Now Properly Detecting)
- Traces: 2/100
- Before fix: 0 detected ❌
- After fix: 2 detected ✅
- Status: Enable ENABLE_TIER3_CAPTURE=true for production

---

## Verification Commands

```bash
# Verify Tier 1 rank
cd backend
./venv/bin/python3 -c "
from rl.lora_scorer import get_scorer
import rl.lora_scorer as lora_module
lora_module._scorer = None
scorer = get_scorer()
print(f'Rank: {scorer.scorer.rank}')
print(f'Params: {scorer.scorer.num_parameters:,}')
"

# Verify Tier 3 trace detection
./venv/bin/python3 -c "
import json
from pathlib import Path
trace_file = Path('../claude-rl-agent/data/v4_traces/traces.jsonl')
if trace_file.exists():
    traces = [json.loads(line) for line in open(trace_file) if line.strip()]
    print(f'Traces found: {len(traces)}')
"
```

---

## Next Steps

1. **✅ DONE**: Fix config reading bugs
2. **✅ DONE**: Retrain Tier 1 with rank-32
3. **TODO**: Continue accumulating Tier 2 DPO pairs (need 65 more)
4. **TODO**: Enable Tier 3 capture: `ENABLE_TIER3_CAPTURE=true`
5. **TODO**: Restart backend to apply Tier 3 capture

---

## Files Modified

1. [backend/rl/lora_scorer.py](backend/rl/lora_scorer.py#L726-L738)
   - Fixed `get_scorer()` to read rank from config

2. [backend/train_all_tiers.py](backend/train_all_tiers.py#L147-L166)
   - Fixed trace reading to use `traces.jsonl` file

3. [backend/RL_TRAINING_AUDIT_REPORT.md](backend/RL_TRAINING_AUDIT_REPORT.md)
   - Comprehensive audit documenting all issues

4. [backend/FIXES_APPLIED.md](backend/FIXES_APPLIED.md) *(this file)*
   - Summary of fixes and verification

---

**All critical bugs fixed and verified working!** 🎉

The RL system is now properly configured and ready to scale with more data.
