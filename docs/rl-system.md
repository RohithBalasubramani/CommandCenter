# Continuous RL System — Three-Tier Architecture

## The Simple Version

Imagine a new employee at a control room. Every time an operator asks a question ("How's pump 001 doing?"), the employee picks dashboard widgets to show and says something out loud. At first, the employee might pick bad widgets or say unhelpful things. But every time the operator gives a thumbs up or thumbs down, the employee learns.

**That's exactly what our RL system does.** It learns from every interaction — automatically, continuously, without any human retraining.

When an operator asks "Show me pump 001 vibration", the AI does two things:

1. **Picks dashboard widgets** — Which charts, gauges, KPIs, and tables to show? How big should each one be?
2. **Speaks a voice response** — A short sentence read aloud through text-to-speech, like "Pump 001 vibration is at 2.4 mm/s, within normal range."

Both can be good or bad. The RL system makes both better over time, using three learning tiers.

---

## The Three Tiers

Think of it like three different learning speeds:
- Notice something immediately and adjust (Tier 1)
- Reflect on patterns over days and retrain the brain (Tier 2)
- Study how an expert thinks and copy their reasoning (Tier 3)

```
User Query → Orchestrator → Widget Selector → Dashboard + Voice Response
                                  ↑
                    ┌─────────────┴──────────────────┐
                    │   Continuous RL System          │
                    │                                 │
                    │  Tier 1: Low-Rank Scorer        │  ← ms latency, CPU
                    │           + Prompt Evolver      │
                    │  Tier 2: Unified DPO Trainer    │  ← periodic, GPU
                    │           (widget + voice)      │
                    │  Tier 3: Reasoning Distillation │  ← periodic, GPU
                    │           (SFT from Claude)     │
                    └─────────────────────────────────┘
                                  ↑
               User Feedback (thumbs up/down, interactions)
               + Claude Auto-Evaluator (widget + voice analysis)
```

### Tier 1 — The Quick Learner (Milliseconds)

**Two components that learn instantly from every feedback:**

#### 1.1: Low-Rank Widget Scorer

A tiny PyTorch network (**28,193 parameters**, rank-32) that makes instant score adjustments for widget selection. Runs on CPU with negligible overhead.

**Architecture:**
```
Input (813-dim):
  ├─ Sentence embedding (768)  × 0.3 scaling
  ├─ Scenario one-hot (19)     × 3.0 scaling
  ├─ Query features (9)        × 3.0 scaling
  ├─ Query type (7)            × 3.0 scaling
  ├─ Widget context (5)        × 3.0 scaling
  └─ Primary characteristic (5) × 3.0 scaling
      ↓
Low-Rank Layer: W = A @ B (rank-32)
  A: (813, 32)    → 26,016 params
  B: (32, 64)     →  2,048 params
      ↓ ReLU
Output Head: (64, 1) → 129 params
      ↓ Tanh
Score Adjustment: [-1.0, +1.0]
```

**Feature scaling prevents embedding dominance** — embeddings are scaled down (×0.3) while structured features are scaled up (×3.0).

**Training target**: Claude's per-widget `appropriateness_score` (0.0-1.0) from `per_widget_feedback`.

**Pairwise ranking loss**: Uses `MarginRankingLoss` for query-specific widget ordering, training the scorer to rank widgets correctly for each specific query.

**How it works:**
- When someone gives a thumbs up, Tier 1 learns "showing a trend chart for pump questions is good"
- When someone gives a thumbs down, it learns "showing a generic table for vibration questions is bad"
- It adjusts widget scores immediately — by the very next question, it's already slightly smarter

**Training:** Online learning via Adam optimizer (LR=1e-3), trains on every feedback event with rich evaluation. Checkpoints every 50 updates to `rl_checkpoints/scorer/scorer_latest.pt`.

**Checkpoint migration**: Handles both dimensionality and rank changes. Rank changes trigger fresh start.

#### 1.2: Prompt Evolver

A **UCB1 multi-armed bandit** that evolves the system prompt to improve widget selection quality.

**7 mutation types** (preserves logic, more impactful than cosmetic):

**Language mutations** (4):
- `rephrase_rule` — Reword rules for clarity
- `simplify_affinity` — Make domain affinity descriptions clearer
- `tighten_wording` — Remove verbosity while keeping meaning
- `improve_clarity` — Reduce ambiguity in instructions

**Structure mutations** (3):
- `reorder_rules` — Change rule order (LLMs are order-sensitive)
- `add_examples` — Add concrete clarifications where vague
- `rephrase_conditional` — Same logic, different framing

**Reward signal**: Claude's `evaluation_confidence` (0.0-1.0 → -1.0 to 1.0 reward range).

**Evolution strategy**:
- Evolves every **25 trials** (was 50, lowered for faster iteration)
- Max **10 variants** at a time
- Epsilon-greedy (ε=0.1) for exploration
- **Validation holdout**: Tests new variants against last 50 experiences before adding
- Rejects variants that perform worse than baseline on validation set

**State persistence**: `rl_training_data/prompt_evolver_state.json`

**Integration**: `widget_selector.py` uses `get_prompt_evolver().get_prompt()` instead of static prompt. Reward flows back via `Experience.prompt_version`.

---

### Tier 2 — The Deep Learner (Periodic, ~Daily)

**Unified LoRA DPO fine-tuning on `unsloth/Meta-Llama-3.1-8B-Instruct`**. Trains the LLM to select better widgets **AND** generate better voice responses in a single training run.

**Key insight**: Widget selection and voice generation share the same base model. Training them together is more efficient than separate training runs.

**How it works:**
1. Accumulates preference pairs from **BOTH** widget selection quality AND voice response quality into a **single pool**
2. Each pair is tagged with `pair_type` field: `"widget"` or `"voice"`
3. When >=**80 pairs** accumulated (mixed types), triggers unified DPO training
4. After training, the improved model is exported to GGUF, quantized, and hot-swapped into Ollama
5. The next question uses the new model for **both** widget selection **and** voice generation

**Training example — Widget pair:**
```
Question: "Show pump 001 vibration"

CHOSEN:  {
  "widgets": [
    {"scenario": "trend", "size": "hero"},
    {"scenario": "kpi", "size": "compact"},
    {"scenario": "alerts", "size": "compact"}
  ]
}

REJECTED: {
  "widgets": [
    {"scenario": "table", "size": "hero"},
    {"scenario": "pie-chart", "size": "small"}
  ]
}
```

**Training example — Voice pair:**
```
Question: "What's pump 001 power consumption?"

CHOSEN:  "Pump 001 is consuming 45 kilowatts, 12 percent above daily average."

REJECTED: "Well, let me think about that. Based on the available data, I'd be happy to
           help you understand the power situation. The pump seems to be running."
```

**Training pipeline:**
```
1. Accumulate DPO pairs (widget + voice) into single pool with pair_type tags
2. Trigger at ≥80 pairs (mixed types OK)
3. Quality filtering:
   - Rich evaluation exists (Claude evaluated)
   - Clear preference signal (min reward gap 0.3)
   - Intent matching (chosen/rejected for similar queries)
   - Deduplication
4. Pre-flight checks: CUDA available, ≥4GB GPU free
5. Train with unified DPO_CONFIG (same config for both types)
6. Evaluation gate: reject if loss > 0.7
7. Export: Merge LoRA → GGUF (q4_k_m) → Register with Ollama
8. Hot-swap: Next inference uses cc-widget-selector:latest
9. Audit log entry persisted to dpo_training_audit.jsonl
```

**DPO Configuration** (single config for both widget + voice):

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Meta-Llama-3.1-8B-Instruct` (ungated mirror) |
| LoRA rank (`lora_r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q,k,v,o,gate,up,down_proj |
| Load in 4-bit | true (NF4) |
| Learning rate | 5e-5 |
| Beta | 0.1 |
| Optimizer | paged_adamw_8bit |
| Max sequence length | 2048 |
| Max prompt length | 1024 |
| Batch size | 4 |
| Gradient accumulation | 4 (effective batch=16) |
| Epochs | 3 |
| Cooldown | 30 min |

**Why unified training works**:
- Widget selection and voice generation use the same base LLM
- Both need to understand industrial domain knowledge
- Training together prevents model drift between widget selector and voice generator
- Single checkpoint to manage and deploy

---

### Tier 3 — The Reasoning Learner (Periodic, ~Weekly)

**Supervised Fine-Tuning (SFT) on Claude's thinking traces.** Teaches LLaMA 8B to think **AND** answer like Claude by behavioral cloning from real reasoning chains.

**The insight**: Claude doesn't just give good answers — it has a reasoning process. By capturing Claude's internal thinking, we can teach LLaMA to reason the same way, not just mimic outputs.

**How it works:**
1. **Trace capture** (fully automated):
   - Auto-captures **15% of queries** randomly + **high-confidence queries** (evaluation_confidence > 0.8)
   - Runs query through **Claude CLI** with thinking enabled
   - Mines thinking blocks from session transcripts at `~/.claude/projects/*/session.jsonl`
   - Saves as V4Trace with `thinking_text` + `answer_text` + metadata
   - Queue-based background processing (non-blocking, zero latency impact)

2. **Training trigger** (fully automated):
   - BackgroundTrainer checks every **30 minutes** for accumulated traces
   - When **>=100 traces** found, automatically triggers SFT training in background thread
   - No manual intervention needed — runs like Tier 1 and Tier 2

3. **Dataset building**:
   - Each trace with thinking produces **THREE SFT samples**:
     - **Sample 1 (type=thinking)**: `prompt → reasoning` — teaches HOW to think
     - **Sample 2 (type=answer)**: `prompt → answer` — teaches WHAT to output
     - **Sample 3 (type=consistency)**: `prompt + thinking → answer` — teaches answer must follow reasoning
   - Traces without thinking produce one sample: `prompt → answer`
   - Different system prompts per type for curriculum learning

4. **Curriculum phased training**:
   - **Phase 1 (answer)**: Learn correct JSON output format (schema discipline)
   - **Phase 2 (thinking)**: Learn how to reason about queries (reasoning skill)
   - **Phase 3 (consistency)**: Learn that answers must follow from reasoning (bridge)
   - Each phase = separate SFTTrainer run, same model, weights accumulate
   - `epochs_per_phase = max(1, total_epochs // num_phases)`

5. **Prompt parity**:
   - Uses `build_full_prompt()` to construct EXACT same prompt Ollama receives
   - Tries import from `backend.layer2.widget_selector.build_production_prompt()`
   - Falls back to file read if running standalone
   - **Critical**: Claude must see same prompt as LLaMA for effective distillation

**Training example:**
```
Input Prompt: "What is pump 001 vibration?"

Claude Thinking (captured from session transcript):
"Check pump 001 data. Vibration sensor is vibration_de_mm_s column.
Last reading: 2.4 mm/s. Normal range for this pump: 0-4.5 mm/s.
2.4 is within normal range. Should mention both value and context."

Claude Answer:
"Pump 001 vibration is currently 2.4 mm/s (normal range: 0-4.5 mm/s)"

SFT Sample 1 (thinking): prompt → thinking
SFT Sample 2 (answer): prompt → answer
SFT Sample 3 (consistency): prompt + thinking → answer
```

**SFT Configuration**:

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Meta-Llama-3.1-8B-Instruct` |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Batch size | 2 |
| Gradient accumulation | 4 (effective=8) |
| Learning rate | 2e-4 |
| Num epochs | 3 (1 per phase) |
| Max sequence length | 4096 |
| Completion-only loss | true (only train on assistant response) |

**Fully automated integration**:
- Trace capture: `rl/tier3_integration.py` hooks into `continuous.py` feedback loop
- Training trigger: `background_trainer.py` checks every 30 min, auto-trains when ready
- Status monitoring: `/api/layer2/rl-status/` → `trainer.tier3_sft`
- Optional manual trigger: `./scripts/tier3_train.py`

**Environment variables**:
- `ENABLE_TIER3_CAPTURE=true` — Enable trace capture
- `TIER3_MIN_TRACES=100` — Minimum traces before training

**Production ready** (verified 2026-02-08):
- All bugs fixed (path issues, parameter bugs, import bugs, threading bugs)
- Comprehensive E2E tests: `python3 backend/rl/test_tier3_e2e.py`
- Non-blocking background execution
- Graceful error handling
- See `TIER3_PRODUCTION_READY.md` for full audit report

---

## Why Three Tiers?

**Tier 1 alone isn't enough** — It can only adjust scores slightly. It can't fundamentally change how the AI thinks about widget selection or voice generation.

**Tier 2 alone isn't enough** — It only improves selection quality. It doesn't teach the model to *reason* about why certain widgets or responses are better.

**Tier 3 alone isn't enough** — Without Tier 1's instant feedback and Tier 2's pattern learning, improvement would be slow and disconnected from user needs.

**Together:**

| Tier | What it improves | How fast | When it runs | Hardware |
|------|-----------------|----------|-------------|----------|
| **Tier 1** | Widget scores + Prompt variants | Instant (ms) | Every feedback | CPU |
| **Tier 2** | Widget + Voice selection model | ~15-25 min | Every >=80 pairs | GPU |
| **Tier 3** | Reasoning ability (thinks like Claude) | ~5 min | Every >=100 traces (auto-check 30min) | GPU |

---

## The Complete Feedback Loop

```
     Operator asks: "Show pump 001 vibration"
                    |
                    v
     AI picks widgets (trend + KPI + alerts)
     AI generates voice: "Pump 001 at 2.4 mm/s, normal range"
                    |
                    v
     Dashboard appears, voice plays
                    |
                    v
     Operator gives thumbs up (or thumbs down)
                    |
                    v
     Claude AI judges the response (automatic, async):
       - Per-widget appropriateness scores
       - Missing widget analysis
       - Voice quality evaluation (5 dimensions)
                    |
                    v
     Reward calculated (combines all signals):
       - User rating: +1.0
       - Widget engagement: +0.2
       - Claude evaluation confidence: +0.4
       - Per-widget appropriateness: +0.6
       - Voice quality: +0.15
       - Total: +2.35
                    |
                    v
     FOUR things happen simultaneously:
       |
       ├─ Tier 1.1 (INSTANT): Low-rank scorer learns
       |   "pump + trend widget = good" — adjusts scores for next query
       |
       ├─ Tier 1.2 (ACCUMULATES): Prompt evolver tracks
       |   Prompt variant performance, evolves every 25 trials
       |
       ├─ Tier 2 (ACCUMULATES): Saves to unified DPO pool
       |   Widget pair: {trend+KPI} > {table+gauge}
       |   Voice pair: specific response > vague filler
       |   When >=80 pairs: unified training → deploy
       |
       └─ Tier 3 (ACCUMULATES): May capture Claude trace (15% + high-conf)
           Background: Query → Claude CLI → mines thinking → saves V4Trace
           BackgroundTrainer checks every 30 min
           When >=100 traces: SFT training → teaches LLaMA to think like Claude
```

---

## Voice Quality Scoring

Every voice response is evaluated on **5 dimensions** by **two independent judges**:

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Groundedness | 0.25 | References equipment IDs, metrics, units ("2.4 mm/s" vs "seems fine") |
| Conciseness | 0.20 | 1-3 sentences, suitable for TTS |
| Directness | 0.20 | Leads with the answer, no filler preamble |
| Specificity | 0.20 | Contains numeric values, equipment names, units |
| TTS-Friendliness | 0.15 | No markdown, abbreviations, URLs, or symbols |

**Two judges, blended:**
- **Rule-based scorer** (`rl/text_quality_scorer.py`) — Checks structural quality (sentence count, filler detection, markdown detection). Fast and consistent.
- **Claude AI evaluator** — Reads the actual response and judges if it's genuinely helpful. Understands meaning.
- **Final score: 60% Claude + 40% rule-based** (Claude is better at semantics, rules are better at format)

**Voice rating derivation** (priority order):
1. **Claude voice evaluation** — If `voice_evaluation_confidence ≥ 0.5` → "good", else "bad" (preferred)
2. **User rating fallback** — thumbs up → "good" (blunt proxy, only when Claude hasn't evaluated)

This prevents a user upvoting great widgets from falsely labeling a mediocre voice response as "good."

---

## Safety Mechanisms

The system has multiple safeguards to prevent degradation:

1. **Evaluation gate** — If training produces a model with loss > 0.7, it's rejected and the old model stays
2. **Relevance floors** — Critical widgets (alerts, KPI) can never be ranked below position 3
3. **Reward clipping** — No single interaction can have outsized influence (rewards capped at [-2, +2])
4. **GPU lock** — Tier 2 DPO uses `lora_training.lock` — only one worker trains at a time
5. **Stale lock detection** — If a training job crashes, the lock is automatically cleared (PID check)
6. **Backup on training** — DPO pairs are backed up before training; restored if training fails
7. **Cooldown** — Tier 2 waits 30 min between trainings to prevent thrashing
8. **CUDA pre-flight** — Before acquiring GPU lock, verifies CUDA available and ≥4GB free
9. **Training audit log** — Every training run logged to `dpo_training_audit.jsonl`
10. **DPO quality gates** — Min reward gap of 0.3, intent matching, dedup, 5,000 pair memory cap
11. **Prompt validation holdout** — New prompt variants tested on last 50 experiences before adoption

---

## The Claude Supervisor

The system uses Claude (Sonnet 4.5) as an automated supervisor. Instead of requiring a human to review every interaction, Claude:

1. **Evaluates widget selection** — Per-widget appropriateness scores, identifies missing widgets
2. **Evaluates voice quality** — 5-dimension scoring (groundedness, conciseness, directness, specificity, TTS-friendliness)
3. **Provides confidence scores** — How sure is Claude about its evaluation?
4. **Suggests improvements** — "Missing a gauge widget for current value", "Voice should mention specific units"

This means the system learns 24/7 without human supervision. Claude acts like a senior operator who reviews every interaction and provides constructive feedback.

### What Claude returns

```json
{
  "overall_rating": "GOOD",
  "confidence": 0.85,
  "reasoning": "The KPI widget immediately shows pump status...",
  "query_understanding": "User wants to monitor pump vibration levels",
  "widget_feedback": [
    {
      "widget_index": 0,
      "widget_type": "trend",
      "appropriateness_score": 0.95,
      "size_appropriate": true,
      "issues": [],
      "strengths": ["Shows vibration trend over time"]
    }
  ],
  "missing_widgets": ["gauge"],
  "suggested_improvements": ["Add gauge for current value"],
  "voice_evaluation": {
    "quality_rating": "GOOD",
    "confidence": 0.92,
    "reasoning": "Cites specific RPM and load, leads with key data",
    "dimension_scores": {
      "groundedness": 0.95,
      "conciseness": 0.90,
      "directness": 0.95,
      "specificity": 0.90,
      "tts_friendliness": 0.95
    }
  }
}
```

### Running the evaluator

```bash
# Single batch of 20
python backend/auto_evaluate_responses.py --batch-size 20

# Continuous mode (evaluates every 5 minutes)
python backend/auto_evaluate_responses.py --continuous
```

---

## Reward Signal Aggregation

Rewards are computed from multiple signal sources with configurable weights:

| Signal | Weight | Range | Description |
|--------|--------|-------|-------------|
| `explicit_rating` | 1.0 | [-1, +1] | Thumbs up (+1) or down (-1) |
| `follow_up_type` | 0.5 | [-1, +1] | satisfied=+1, refinement=-0.3, repeat=-1 |
| `widget_engagement` | 0.3 | [0, +1] | User expanded/clicked widgets |
| `response_latency` | 0.1 | [-1, +1] | Faster responses score higher |
| `intent_confidence` | 0.1 | [0, +1] | Higher parser confidence = more reliable |
| `eval_confidence` | 0.2 | [0, +1] | Claude evaluation confidence boost |
| `per_widget_scores` | 0.4 | [0, +1] | Average per-widget appropriateness from Claude |
| `missing_widgets` | -0.3 | [-1, 0] | Penalty for each missing widget type |
| `size_appropriateness` | 0.2 | [0, +1] | Widget size correctness |
| `text_quality` | 0.4 | [-0.5, +0.5] | Voice quality (blended 60% Claude + 40% rule-based) |

Final reward is clipped to `[-2.0, +2.0]`.

---

## Experience Buffer

Every orchestration result is stored with optional feedback.

### Experience Schema

```python
@dataclass
class Experience:
    # Identity
    query_id: str                    # UUID for feedback tracking
    timestamp: float                 # Unix timestamp

    # Context
    transcript: str                  # User's spoken/typed query
    user_id: str                     # For per-user learning
    parsed_intent: dict              # {type, domains, entities, confidence}

    # Action taken
    widget_plan: dict                # Selected widgets with sizes/relevance
    fixtures: dict                   # Data fixtures used
    voice_response: Optional[str]    # Generated voice text

    # Immediate signals
    intent_confidence: float         # Parser confidence [0, 1]
    processing_time_ms: int          # End-to-end latency

    # Delayed feedback (updated later)
    user_rating: Optional[str]       # "up" | "down"
    follow_up_type: Optional[str]    # "satisfied" | "refinement" | "repeat"
    widget_interactions: list        # [{widget_index, action, duration_ms}]
    correction_text: Optional[str]   # User's correction if any

    # Rich widget evaluation (from Claude auto-evaluator)
    evaluation_confidence: Optional[float]
    evaluation_reasoning: Optional[str]
    query_understanding: Optional[str]
    per_widget_feedback: list        # Per-widget appropriateness scores
    missing_widgets: list
    suggested_improvements: list

    # Voice evaluation
    voice_response_rating: Optional[str]       # "good" | "bad" (derived)
    voice_evaluation_confidence: Optional[float]    # Claude voice confidence
    voice_evaluation_reasoning: Optional[str]       # Why Claude rated it
    voice_dimension_scores_claude: dict             # 5-dimension scores

    # RL metadata
    prompt_version: str              # Which prompt variant was used
    computed_reward: Optional[float] # Aggregated reward [-2, +2]
```

### Storage

- **In-memory**: Thread-safe deque (maxlen=10,000)
- **Disk**: `rl_training_data/experience_buffer.json` — persisted on every update
- **Multi-worker safe**: Reads from disk before adding to handle gunicorn workers

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/layer2/feedback/` | Submit feedback (rating, interactions, Claude eval) |
| `GET` | `/api/layer2/rl-status/` | Current RL system status (all tiers, cooldowns) |
| `GET` | `/api/layer2/rl-history/` | Historical data (reward timeline, loss curves) |

### Status Response Structure

```json
{
  "running": true,
  "buffer": {
    "total_experiences": 1234,
    "with_feedback": 567,
    "ratings": {"up": 400, "down": 167}
  },
  "trainer": {
    "training_steps": 45,
    "tier1_scorer": {
      "training_steps": 567,
      "current_rank": 32,
      "input_dim": 813
    },
    "tier1b_composition": {
      "training_steps": 23
    },
    "dpo": {
      "training_in_progress": false,
      "pending_pairs": 45,
      "pending_widget_pairs": 30,
      "pending_voice_pairs": 15,
      "min_pairs_for_training": 80,
      "cooldown_remaining_s": 1245,
      "ready_to_train": false
    },
    "tier3_sft": {
      "training_in_progress": false,
      "last_check_time": 1707428900,
      "check_interval_s": 1800,
      "next_check_in_s": 450
    }
  }
}
```

---

## Configuration

All RL parameters live in `backend/rl/config.py`.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_CONTINUOUS_RL` | `true` | Enable/disable the RL system |
| `RL_BASE_MODEL` | `unsloth/Meta-Llama-3.1-8B-Instruct` | Base model for Tier 2/3 |
| `RL_OUTPUT_MODEL` | `cc-widget-selector` | Ollama model name |
| `RL_GGUF_QUANT` | `q4_k_m` | Quantization level |
| `GUNICORN_WORKER` | `true` | Required for RL init in gunicorn |
| `HF_HUB_DISABLE_XET` | `1` | Use fast HTTP downloads |
| `ENABLE_TIER3_CAPTURE` | `false` | Enable Tier 3 trace capture |
| `TIER3_MIN_TRACES` | `100` | Min traces before SFT training |

### Key Thresholds

| Parameter | Value |
|-----------|-------|
| Tier 1 scorer rank | 32 |
| Tier 1 scorer params | 28,193 |
| Tier 1 scorer input dim | 813 |
| Tier 1 prompt evolution interval | 25 trials |
| Tier 2 min DPO pairs | 80 (mixed widget + voice) |
| Tier 2 cooldown | 30 min |
| Tier 3 min traces | 100 |
| Tier 3 check interval | 30 min |
| Buffer max size | 10,000 |
| Max acceptable loss (eval gate) | 0.7 |
| Max reward | 2.0 |
| Min reward gap for DPO | 0.3 |

---

## Testing

```bash
cd backend

# Tier 1 unit tests
./venv/bin/python -m rl.test_continuous

# Tier 2 DPO tests
./venv/bin/python -m rl.test_tier2_hardening  # 33 tests
./venv/bin/python -m rl.test_tier2_behavioral # 3 behavioral cases
./venv/bin/python -m rl.test_tier2_stress     # 14 edge cases

# Tier 3 SFT tests
./venv/bin/python rl/test_tier3_e2e.py

# Live E2E against running backend
./venv/bin/python -m rl.e2e_test http://127.0.0.1:8100

# All tiers unified test
./venv/bin/python rl/test_all_tiers.py
```

---

## Monitoring & Operations

### Check RL Status

```bash
curl -s http://localhost:8100/api/layer2/rl-status/ | python3 -m json.tool
```

Includes:
- `trainer.tier1_scorer` — Real-time learning stats
- `trainer.dpo` — Unified DPO (widget + voice) stats
- `trainer.tier3_sft` — SFT reasoning distillation stats

### Monitor Training Progress

```bash
tail -f logs/gunicorn-error.log | grep -E "rl\.|Training|LoRA|scorer|Tier"
```

Look for:
- `Tier 1: Updated scorer` — Instant learning
- `Tier 2: DPO training started` — Unified widget+voice training
- `Tier 3: Checking for SFT training trigger` — Auto-check every 30 min
- `Tier 3: SFT training completed successfully` — Reasoning distillation done

### Training Audit Log

Every training run is appended to `rl_training_data/dpo_training_audit.jsonl`:
```json
{
  "timestamp": 1707400000,
  "tier": 2,
  "pair_type": "unified",
  "status": "success",
  "num_pairs": 85,
  "widget_pairs": 55,
  "voice_pairs": 30,
  "final_loss": 0.42,
  "duration_s": 890,
  "version": 1
}
```

### GPU Memory Requirements

| Component | VRAM |
|-----------|------|
| Tier 1 scorer | ~0 (CPU) |
| Tier 2 DPO training (peak) | ~60 GB |
| Tier 3 SFT training (peak) | ~50 GB |
| Ollama models | ~5-10 GB each |
| Embedding model | ~7 GB |

**GPU lock shared**: Tier 2 and Tier 3 share a GPU lock — only one trains at a time.

---

## File Reference

### Core RL System

| File | Purpose |
|------|---------|
| `backend/rl/config.py` | All configuration (unified DPO_CONFIG, CONTINUOUS_RL_CONFIG) |
| `backend/rl/experience_buffer.py` | Experience storage, disk persistence |
| `backend/rl/reward_signals.py` | Multi-signal reward computation |
| `backend/rl/background_trainer.py` | **Main coordinator**: Tier 1/2/3 training loops, safety nets |
| `backend/rl/continuous.py` | ContinuousRL coordinator (init, record, feedback) |

### Tier 1 (Real-time Online Learning)

| File | Purpose |
|------|---------|
| `backend/rl/lora_scorer.py` | Low-rank widget scorer (28,193 params, rank-32) |
| `backend/rl/prompt_evolver.py` | UCB1 bandit for prompt evolution |

### Tier 2 (Unified DPO)

| File | Purpose |
|------|---------|
| `backend/rl/trainer.py` | Unified DPO trainer (widget + voice) |
| `backend/rl/data_formatter.py` | DPO pair formatting with pair_type tags |
| `backend/rl/text_quality_scorer.py` | Rule-based 5-dimension voice scorer |
| `backend/rl/export.py` | GGUF conversion & Ollama deployment |

### Tier 3 (Reasoning Distillation)

| File | Purpose |
|------|---------|
| `backend/rl/tier3_integration.py` | Automated trace capture & SFT trigger |
| `claude-rl-agent/src/claude_teacher.py` | Runs Claude CLI, mines thinking from transcripts |
| `claude-rl-agent/src/v4_trace.py` | V4Trace storage, SFT dataset builder (3 samples per trace) |
| `claude-rl-agent/src/sft_trainer.py` | ClaudeSFTTrainer with curriculum phased training |
| `claude-rl-agent/src/automated_runner.py` | Runs prompts through Claude and LLaMA |
| `scripts/tier3_train.py` | Manual SFT training trigger (optional) |

### Integration & APIs

| File | Purpose |
|------|---------|
| `backend/layer2/widget_selector.py` | RL reranking integration, prompt evolver usage |
| `backend/layer2/orchestrator.py` | Experience recording + voice response generation |
| `backend/layer2/apps.py` | Django app init, RL startup |
| `backend/layer2/views.py` | Feedback & status API endpoints |
| `backend/auto_evaluate_responses.py` | Claude auto-evaluator (widget + voice) |

### Data Directories

| Path | Contents |
|------|----------|
| `rl_training_data/experience_buffer.json` | Live experience buffer |
| `rl_training_data/pending_dpo_pairs.json` | Accumulated unified DPO pairs |
| `rl_training_data/dpo_training_audit.jsonl` | Training run audit trail |
| `rl_training_data/prompt_evolver_state.json` | Prompt variant performance tracking |
| `rl_checkpoints/scorer/` | Tier 1 scorer checkpoints |
| `rl_checkpoints/dpo_vN/` | Tier 2 unified DPO checkpoints |
| `rl_checkpoints/export/` | GGUF export output |
| `claude-rl-agent/data/v4_traces/` | Tier 3 Claude thinking traces |
| `claude-rl-agent/models/sft_checkpoints/` | Tier 3 SFT training checkpoints |

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                  THREE-TIER RL ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────┘

User Query
    ↓
Orchestrator → Widget Selector (with RL reranking)
    ↓              ↓
Dashboard    Voice Response
    ↓              ↓
User Feedback (thumbs up/down, interactions)
    ↓
Claude Auto-Evaluator (per-widget + voice analysis)
    ↓
Experience Recorded
    ↓
┌───────────────────────────────────────────────────────────────┐
│                    CONTINUOUS RL SYSTEM                        │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  TIER 1 — Real-time Online Learning (CPU, ms latency)        │
│  ├─ Low-Rank Scorer (28,193 params, rank-32)                 │
│  │  Trains on every feedback, adjusts widget scores           │
│  └─ Prompt Evolver (UCB1 bandit)                             │
│     Evolves prompt every 25 trials with validation            │
│                                                                │
│  TIER 2 — Unified DPO (GPU, periodic ~daily)                 │
│  ├─ Accumulates widget + voice pairs (single pool)            │
│  ├─ Triggers at ≥80 pairs (mixed types)                       │
│  ├─ Trains with unified DPO_CONFIG                            │
│  └─ Exports to Ollama: cc-widget-selector:latest              │
│                                                                │
│  TIER 3 — Reasoning Distillation (GPU, periodic ~weekly)     │
│  ├─ Auto-captures 15% queries + high-confidence               │
│  ├─ Runs through Claude CLI, mines thinking                   │
│  ├─ BackgroundTrainer checks every 30 min                     │
│  ├─ Triggers SFT at ≥100 traces                               │
│  └─ Curriculum training: answer → thinking → consistency      │
│                                                                │
└───────────────────────────────────────────────────────────────┘
    ↓
Improved Model → Next Query Uses Better Selection & Reasoning
```

**All three tiers run fully automatically in production.** No manual intervention required.
