# RL SYSTEM MONITORING & AUDIT GUIDE

**Last Updated**: 2026-02-08

---

## 📊 REAL-TIME STATUS MONITORING

### Quick Status Check
```bash
curl -s http://127.0.0.1:8100/api/layer2/rl-status/ | python3 -m json.tool
```

### Watch Training Progress (Live Updates)
```bash
watch -n 5 'curl -s http://127.0.0.1:8100/api/layer2/rl-status/ | python3 -c "import sys, json; data = json.load(sys.stdin); t1 = data[\"trainer\"][\"tier1_scorer\"]; t2 = data[\"trainer\"][\"tier2_lora\"]; print(f\"Tier 1: {t1[\"training_steps\"]:,} steps | Loss: {t1[\"avg_loss\"]:.6f}\"); print(f\"Tier 2: {\"TRAINING\" if t2[\"training_in_progress\"] else \"READY\"} | Pairs: {t2[\"pending_pairs\"]} | Trainings: {t2[\"total_trainings\"]}\")"'
```

---

## 📁 LOG FILE LOCATIONS

### 1. Backend Application Logs
```bash
# Real-time logs
tail -f /home/rohith/desktop/CommandCenter/logs/backend.log | grep -E "rl|training|scorer"

# Recent RL activity
tail -100 /home/rohith/desktop/CommandCenter/logs/backend.log | grep -i "tier\|scorer\|dpo"
```

### 2. Gunicorn Process Logs
```bash
# Check systemd service logs
journalctl --user -u cc-backend.service -f

# Last 50 lines
journalctl --user -u cc-backend.service -n 50
```

### 3. RL Training Data Directory
```bash
ls -lh /home/rohith/desktop/CommandCenter/rl_training_data/

# Experience buffer
du -h /home/rohith/desktop/CommandCenter/rl_training_data/experience_buffer.json

# Detailed evaluations
ls -lt /home/rohith/desktop/CommandCenter/rl_training_data/detailed_evaluations/ | head -10
```

---

## 🔍 TIER 1 MONITORING (Low-Rank Scorer)

### Training Progress
```python
import requests
status = requests.get('http://127.0.0.1:8100/api/layer2/rl-status/').json()
tier1 = status['trainer']['tier1_scorer']

print(f"Training Steps: {tier1['training_steps']:,}")
print(f"Average Loss: {tier1['avg_loss']:.6f}")
print(f"Recent Losses: {tier1['recent_losses'][-10:]}")
```

### Checkpoint Status
```bash
# Check if checkpoint exists
ls -lh /home/rohith/desktop/CommandCenter/rl_checkpoints/lora_scorer_checkpoint.pt

# Size indicates training progress
du -h /home/rohith/desktop/CommandCenter/rl_checkpoints/lora_scorer_checkpoint.pt
```

### Performance Metrics
- **Training steps**: Should increment with each feedback event
- **Loss trend**: Should decrease over time (converging)
- **Target loss**: < 0.1 indicates good convergence

---

## 🤖 TIER 2 MONITORING (LoRA DPO)

### Training Status
```python
import requests
status = requests.get('http://127.0.0.1:8100/api/layer2/rl-status/').json()
tier2 = status['trainer']['tier2_lora']

print(f"Training: {tier2['training_in_progress']}")
print(f"Pending Pairs: {tier2['pending_pairs']}")
print(f"Completed Trainings: {tier2['total_trainings']}")
print(f"Current Version: v{tier2['current_version']}")
```

### Checkpoint Inspection
```bash
# List all LoRA versions
ls -d /home/rohith/desktop/CommandCenter/rl_checkpoints/lora_v*

# Check latest checkpoint
ls -lh /home/rohith/desktop/CommandCenter/rl_checkpoints/lora_v1/final/

# Adapter size (should be ~160MB)
du -h /home/rohith/desktop/CommandCenter/rl_checkpoints/lora_v1/final/adapter_model.safetensors
```

### Training Trigger
```bash
# Manually trigger Tier 2 training
curl -X POST http://127.0.0.1:8100/api/layer2/approve-training/

# Check if approval file exists
ls -l /home/rohith/desktop/CommandCenter/rl_training_data/approve_lora_training
```

---

## 📈 EXPERIENCE BUFFER ANALYSIS

### Buffer Statistics
```python
import json
from pathlib import Path

buffer_path = Path('/home/rohith/desktop/CommandCenter/rl_training_data/experience_buffer.json')
with open(buffer_path) as f:
    data = json.load(f)

experiences = data['experiences']
print(f"Total Experiences: {len(experiences)}")
print(f"With Feedback: {sum(1 for e in experiences if e.get('user_rating'))}")
print(f"Up Ratings: {sum(1 for e in experiences if e.get('user_rating') == 'up')}")
print(f"Down Ratings: {sum(1 for e in experiences if e.get('user_rating') == 'down')}")

# Rich evaluation data
rich = sum(1 for e in experiences if e.get('evaluation_confidence'))
print(f"With Rich Evaluations: {rich} ({rich/len(experiences)*100:.1f}%)")
```

### Rich Evaluation Coverage
```bash
# Count detailed evaluations
ls /home/rohith/desktop/CommandCenter/rl_training_data/detailed_evaluations/*.json | wc -l

# View latest evaluation
ls -t /home/rohith/desktop/CommandCenter/rl_training_data/detailed_evaluations/*.json | head -1 | xargs cat | python3 -m json.tool
```

---

## 🎯 PERFORMANCE BENCHMARKS

### Tier 1 Benchmarks
| Metric | Good | Acceptable | Needs Attention |
|--------|------|------------|-----------------|
| Training Steps | > 100K | > 10K | < 10K |
| Average Loss | < 0.1 | 0.1 - 0.5 | > 0.5 |
| Loss Trend | Decreasing | Stable | Increasing |

### Tier 2 Benchmarks
| Metric | Good | Acceptable | Needs Attention |
|--------|------|------------|-----------------|
| DPO Pairs | > 500 | 50 - 500 | < 50 |
| Trainings | > 5 | 1 - 5 | 0 |
| Checkpoint Size | ~160MB | 100-200MB | < 100MB |

### Data Collection Benchmarks
| Metric | Good | Acceptable | Needs Attention |
|--------|------|------------|-----------------|
| Total Experiences | > 1000 | 100 - 1000 | < 100 |
| Feedback Rate | > 50% | 20 - 50% | < 20% |
| Rich Evaluations | > 10% | 1 - 10% | < 1% |

---

## 🔧 DIAGNOSTIC COMMANDS

### Check RL System Health
```bash
python3 <<'EOF'
import requests
import json

try:
    resp = requests.get('http://127.0.0.1:8100/api/layer2/rl-status/', timeout=5)
    if resp.status_code == 200:
        status = resp.json()
        if status.get('running'):
            print("✓ RL System: OPERATIONAL")
            tier1_steps = status['trainer']['tier1_scorer']['training_steps']
            tier2_pairs = status['trainer']['tier2_lora']['pending_pairs']
            print(f"✓ Tier 1: {tier1_steps:,} steps")
            print(f"✓ Tier 2: {tier2_pairs} pairs ready")
        else:
            print("✗ RL System: NOT RUNNING")
    else:
        print(f"✗ API Error: {resp.status_code}")
except Exception as e:
    print(f"✗ Connection Error: {e}")
EOF
```

### Verify Rich Data Flow
```bash
python3 <<'EOF'
import json
from pathlib import Path

buffer_path = Path('/home/rohith/desktop/CommandCenter/rl_training_data/experience_buffer.json')
with open(buffer_path) as f:
    data = json.load(f)

for exp in reversed(data['experiences'][-10:]):
    query_id = exp.get('query_id', 'unknown')[:8]
    conf = exp.get('evaluation_confidence')
    per_widget = len(exp.get('per_widget_feedback', []))

    if conf or per_widget:
        print(f"✓ {query_id}... - Confidence: {conf}, Widgets: {per_widget}")
    else:
        print(f"  {query_id}... - No rich data")
EOF
```

### Monitor Training in Real-Time
```bash
# Watch Tier 1 steps increment
watch -n 2 "curl -s http://127.0.0.1:8100/api/layer2/rl-status/ | python3 -c 'import sys, json; print(json.load(sys.stdin)[\"trainer\"][\"tier1_scorer\"][\"training_steps\"])'"

# Watch Tier 2 status
watch -n 5 "curl -s http://127.0.0.1:8100/api/layer2/rl-status/ | python3 -c 'import sys, json; t2 = json.load(sys.stdin)[\"trainer\"][\"tier2_lora\"]; print(f\"Training: {t2[\"training_in_progress\"]} | Pairs: {t2[\"pending_pairs\"]}\")'
"
```

---

## 📊 CURRENT STATUS (2026-02-08)

### Tier 1: Low-Rank Scorer
- **Status**: ✅ OPERATIONAL
- **Training Steps**: 574,354+
- **Average Loss**: 0.070 (GOOD - converging)
- **Performance**: Training on every feedback event in milliseconds

### Tier 2: LoRA DPO
- **Status**: ✅ COMPLETED TRAINING
- **Checkpoint**: lora_v1/final/ (161MB adapter)
- **DPO Pairs**: 841 available
- **Ready For**: Next training cycle when triggered

### Experience Buffer
- **Total**: 470 experiences
- **With Feedback**: 463 (98.5%)
- **Rich Evaluations**: 3 (0.6%, growing)
- **Ratings**: 7 up, 94 down

### Rich Data Collection
- **Claude Sonnet 4.5**: Auto-evaluator operational
- **Detailed Evaluations**: 3 files saved
- **Per-Widget Feedback**: Working
- **API Integration**: ✅ Fully functional

---

## 🚀 MONITORING BEST PRACTICES

### Daily Checks
1. **RL Status**: `curl http://127.0.0.1:8100/api/layer2/rl-status/`
2. **Tier 1 Progress**: Check training steps incrementing
3. **Buffer Growth**: Verify new experiences being added
4. **Rich Data**: Check detailed_evaluations/ directory

### Weekly Reviews
1. **Tier 1 Loss**: Should show decreasing trend
2. **Tier 2 Checkpoints**: New versions created
3. **Feedback Coverage**: Aim for >50% rated
4. **Rich Evaluation Rate**: Aim for >10%

### Monthly Audits
1. **System Performance**: Widget selection quality
2. **Training Effectiveness**: Loss convergence analysis
3. **Data Quality**: Review rich evaluations
4. **Checkpoint Management**: Archive old versions

---

## 🔔 ALERTS & NOTIFICATIONS

### Warning Signs
- ✗ Tier 1 loss not decreasing after 100K steps
- ✗ Tier 2 not training despite >50 DPO pairs
- ✗ Experience buffer not growing
- ✗ Rich evaluation rate < 1%
- ✗ API returning errors (check logs)

### Critical Issues
- 🚨 RL system not running (`running: false`)
- 🚨 Backend service crashed (check systemd)
- 🚨 GPU unavailable for Tier 2
- 🚨 Experience buffer corrupted
- 🚨 Multi-worker data loss (404 errors)

---

## 📞 TROUBLESHOOTING COMMANDS

### Restart RL System
```bash
# Restart backend (will reinitialize RL)
systemctl --user restart cc-backend.service

# Check if it started
systemctl --user status cc-backend.service
```

### Reset Training (Use with Caution)
```bash
# Backup first!
cp -r /home/rohith/desktop/CommandCenter/rl_checkpoints /tmp/rl_checkpoints_backup
cp /home/rohith/desktop/CommandCenter/rl_training_data/experience_buffer.json /tmp/experience_buffer_backup.json

# Clear checkpoints (Tier 1 will retrain)
rm /home/rohith/desktop/CommandCenter/rl_checkpoints/lora_scorer_checkpoint.pt
```

### Force Tier 2 Training
```bash
# Create approval file
curl -X POST http://127.0.0.1:8100/api/layer2/approve-training/

# Monitor for start (wait ~60s for background cycle)
watch -n 5 'curl -s http://127.0.0.1:8100/api/layer2/rl-status/ | python3 -c "import sys, json; print(json.load(sys.stdin)[\"trainer\"][\"tier2_lora\"][\"training_in_progress\"])"'
```

---

**Prepared by**: Claude Sonnet 4.5
**Last Audit**: 2026-02-08 08:59 UTC
**Status**: ✅ Both tiers operational and training
