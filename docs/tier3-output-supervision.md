# Tier 3: Text Output Quality RL + Claude Output Supervision Framework

> **Status**: Architecture spec + training methodology
> **Scope**: Extends the two-tier RL system with text quality training and defines the complete framework for using Claude's output as training signal for local LLaMA models.

---

## Table of Contents

- [Part 1: Tier 3 Architecture](#part-1-tier-3-architecture)
  - [Three-Tier Overview](#three-tier-overview)
  - [The Gap Tier 3 Closes](#the-gap-tier-3-closes)
  - [Experience Schema Changes](#experience-schema-changes)
  - [Text Quality Scorer](#text-quality-scorer)
  - [Text DPO Pair Generation](#text-dpo-pair-generation)
  - [Training and Deployment](#training-and-deployment)
  - [GPU Coordination](#gpu-coordination)
  - [Status Reporting](#status-reporting)
  - [Data Flow](#data-flow)
  - [Implementation File List](#implementation-file-list)
- [Part 2: Claude Output Supervision Framework](#part-2-claude-output-supervision-framework)
  - [0. Ground Rule](#0-ground-rule)
  - [1. Direct Supervision](#1-direct-supervision-controlled-non-naive)
  - [2. Structural Decomposition](#2-structural-decomposition-of-output)
  - [3. Semantic Content Supervision](#3-semantic-content-supervision)
  - [4. Comparative Output Teaching](#4-comparative-output-teaching-claude-vs-llama)
  - [5. Output-Driven Error Modeling](#5-output-driven-error-modeling)
  - [6. Style-Independent Compression](#6-style-independent-compression-learning)
  - [7. Code-Specific Supervision](#7-code-specific-output-supervision)
  - [8. Output as Oracle for Tests](#8-output-as-oracle-for-automatic-tests)
  - [9. Output-Conditioned Prompting](#9-output-conditioned-prompting-bootstrapping)
  - [10. Output-Anchored Acceptance Gates](#10-output-anchored-acceptance-gates-production)

---

# Part 1: Tier 3 Architecture

## Three-Tier Overview

The Command Center RL system trains local models to improve through continuous user interaction:

| Tier | What it trains | Speed | Hardware | Model |
|------|---------------|-------|----------|-------|
| **Tier 1** | Widget ranking per query | Milliseconds | CPU | Low-rank scorer (6,937 params) |
| **Tier 2** | Widget selection preferences | Periodic (when 200+ DPO pairs) | GPU (QLoRA) | LLaMA 3.1 8B LoRA adapter |
| **Tier 3** | Voice/text response quality | Periodic (when 100+ DPO pairs) | GPU (QLoRA) | LLaMA 3.1 8B LoRA adapter (separate) |

All three tiers run inside the same `BackgroundTrainer` daemon thread. They share the same experience buffer and reward signals but produce different training artifacts.

### Key Files

```
backend/rl/
├── background_trainer.py   # Training loop (all 3 tiers)
├── experience_buffer.py    # Shared experience storage
├── reward_signals.py       # Reward computation (all signals)
├── text_quality_scorer.py  # NEW: Text quality dimensions
├── lora_scorer.py          # Tier 1: online scorer
├── trainer.py              # Tier 2+3: DPO training
├── export.py               # Tier 2+3: GGUF export
├── continuous.py            # Coordinator singleton
└── config.py               # All tier configs
```

---

## The Gap Tier 3 Closes

Before Tier 3, the voice response was a fire-and-forget operation:

```
User query → orchestrator → LLM generates 2-3 sentence voice response → returned to user
                                                                         ↓
                                                              NEVER evaluated
                                                              NEVER improved
                                                              NEVER trained on
```

User feedback (thumbs up/down) only flowed into widget selection scoring. If the widgets were right but the explanation was bad, or vice versa, the system couldn't tell the difference.

**Tier 3 closes this gap** by:
1. Capturing the voice response text in every Experience
2. Scoring text quality independently (5 dimensions)
3. Building DPO preference pairs from good vs bad responses
4. Training a dedicated LoRA adapter for voice response generation
5. Deploying as a separate Ollama model (`cc-voice-response`)

---

## Experience Schema Changes

Three new fields added to the `Experience` dataclass in `experience_buffer.py`:

```python
# Voice response data (Tier 3: text quality)
voice_response: Optional[str] = None           # The actual text generated
voice_response_rating: Optional[str] = None    # "good"/"bad" (derived from user feedback)
voice_quality_scores: dict = field(default_factory=dict)  # Per-dimension scores
```

**Capture point**: The orchestrator passes `voice_response` when recording the experience at the end of each query processing cycle. The variable is already in scope — it's generated at line 862 and the `record_experience()` call is at line 926 of `orchestrator.py`.

**Feedback derivation**: When user submits thumbs up/down, `voice_response_rating` is set:
- `user_rating == "up"` → `voice_response_rating = "good"`
- `user_rating == "down"` → `voice_response_rating = "bad"`

**Backward compatibility**: All new fields have defaults (`None`, `{}`). The existing `from_dict()` method handles missing keys gracefully.

---

## Text Quality Scorer

A lightweight, rule-based scorer in `text_quality_scorer.py`. No ML, runs on CPU in <1ms.

### 5 Dimensions

| Dimension | What it checks | Score |
|-----------|---------------|-------|
| **Groundedness** | Does the response reference specific equipment IDs, metric values, units from the query context? Penalizes vague/generic answers. | 0.0–1.0 |
| **Conciseness** | Is it 2-3 sentences as the TTS spec requires? Penalizes too long (>4 sentences) or too short (0-1). | 0.0–1.0 |
| **Directness** | Does the first sentence directly answer the question? Penalizes filler ("I'd be happy to help", "Let me explain", "Based on..."). | 0.0–1.0 |
| **Specificity** | Contains numeric values and units rather than vague claims? ("37.5 PSI" vs "the pressure is normal") | 0.0–1.0 |
| **TTS-Friendliness** | Is it natural spoken language? Penalizes markdown (`**bold**`, `#`), bullet lists, code blocks, URLs, long parentheticals. | 0.0–1.0 |

### Output

```python
{
    "groundedness": 0.8,
    "conciseness": 1.0,
    "directness": 0.9,
    "specificity": 0.7,
    "tts_friendliness": 1.0,
    "total": 0.88  # Weighted average
}
```

### Integration into Rewards

Added as a new signal in `RewardSignalAggregator.compute_reward()` with weight `0.4`:

```python
self.weights.setdefault("text_quality", 0.4)
```

The signal only fires when `experience.voice_response` is non-empty. For experiences without a voice response, it contributes 0.0 (no penalty, no bonus).

---

## Text DPO Pair Generation

Tier 3 builds preference pairs from voice responses using the same quality filters as Tier 2:

### Filters

```
Chosen (positive side):
  - reward > 0.1
  - evaluation_confidence >= 0.5 (rich eval confirmed)
  - voice_response exists and non-empty

Rejected (negative side):
  - reward < -0.1
  - voice_response exists and non-empty

Pairing rules:
  - Same intent type (status, trend, comparison, etc.)
  - Overlapping domains
  - Similar entity types (both about pumps, or both about cooling towers)
  - Minimum reward gap of 0.3
  - Dedup by (prompt[:200], chosen[:200])
```

### Prompt Format

Mirrors the actual voice generation prompt from `orchestrator.py`:

```
You are Command Center, an industrial operations voice assistant.
Dashboard built: "{heading}" with {N} widgets showing: {widget_summary}
User question: {transcript}
Response:
```

This ensures the DPO training teaches the model to produce better responses for the exact prompt format it sees in production.

---

## Training and Deployment

### Training Pipeline

Same infrastructure as Tier 2 (reuses `CommandCenterDPOTrainer` and `export_to_ollama`):

1. Accumulate text DPO pairs until threshold (100 pairs)
2. Snapshot pairs, save backup to disk
3. Build HuggingFace Dataset, split 90/10 train/eval
4. Run DPO training with QLoRA (4-bit, rank-16)
5. Evaluation gate: reject if final loss > 0.7
6. Export to GGUF (f16 → q4_k_m quantize)
7. Register in Ollama as `cc-voice-response-v{N}`
8. Clean up old versions (keep 2)

### Config

```python
# In CONTINUOUS_RL_CONFIG
"tier3_min_pairs": 100,
"tier3_train_cooldown": 3600,    # 1 hour between trainings
"tier3_auto_deploy": True,
"tier3_model_name": "cc-voice-response",

# Separate DPO config for Tier 3
TIER3_DPO_CONFIG = {
    "lora_r": 16,
    "lora_alpha": 32,
    "learning_rate": 5e-5,
    "beta": 0.1,
    "max_length": 1024,        # Voice responses are short
    "max_prompt_length": 512,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
}
```

### Deployment

Deploys as a **separate model** from Tier 2:
- Tier 2 → `cc-widget-selector-v{N}` (used by `OLLAMA_MODEL_FAST`)
- Tier 3 → `cc-voice-response-v{N}` (future: used by `OLLAMA_MODEL_VOICE`)

---

## GPU Coordination

Tier 2 and Tier 3 share the same file-based lock (`lora_training.lock`). Only one GPU training job runs at a time:

```
Training loop iteration:
  1. Tier 1 update (CPU, always runs unless GPU training active)
  2. Tier 2 accumulate + maybe_train (GPU, first priority)
  3. Tier 3 accumulate + maybe_train (GPU, second priority)

Guard: if self._lora_training OR self._text_training:
  → Skip Tier 1 (avoid CUDA conflicts)
  → Skip the other GPU tier (lock prevents concurrent GPU jobs)
```

Tier 2 runs first in the loop, so it has natural priority. Tier 3 only triggers when Tier 2 is idle.

---

## Status Reporting

Tier 3 metrics are added to `BackgroundTrainer.get_stats()` and flow automatically through the existing `/api/layer2/rl-status/` endpoint:

```json
{
  "tier3_text": {
    "training_in_progress": false,
    "pending_pairs": 42,
    "min_pairs_for_training": 100,
    "total_trainings": 0,
    "total_pairs_trained": 0,
    "last_loss": null,
    "current_version": 0,
    "last_training_time": null
  }
}
```

---

## Data Flow

```
User query
  → orchestrator generates voice_response via quality LLM
  → record_experience(voice_response=voice_response)
  → Experience stored in buffer with voice text

User feedback (thumbs up/down)
  → update_feedback(rating="up")
  → voice_response_rating = "good"
  → text quality scorer populates voice_quality_scores

Background training loop (every 60s):
  → compute_reward() includes text_quality signal (weight 0.4)
  → _tier3_accumulate() builds text DPO pairs from good/bad responses
  → _tier3_maybe_train() triggers at 100 pairs
  → DPO training → GGUF export → Ollama deploy
  → cc-voice-response-v{N} model available

Future: orchestrator uses cc-voice-response instead of generic quality LLM
```

---

## Implementation File List

| File | Action | Changes |
|------|--------|---------|
| `backend/rl/experience_buffer.py` | MODIFY | Add `voice_response`, `voice_response_rating`, `voice_quality_scores` fields. Update `has_feedback()`. |
| `backend/rl/text_quality_scorer.py` | CREATE | `TextQualityScorer` with 5 dimensions. Pure rule-based, CPU, <1ms. |
| `backend/rl/reward_signals.py` | MODIFY | Add `_text_quality_reward()`. Import scorer. Add weight. |
| `backend/rl/config.py` | MODIFY | Add Tier 3 config to `CONTINUOUS_RL_CONFIG`. Add `TIER3_DPO_CONFIG`. |
| `backend/rl/continuous.py` | MODIFY | Add `voice_response` param to `record_experience()`. Derive `voice_response_rating` in `update_feedback()`. |
| `backend/layer2/orchestrator.py` | MODIFY | Pass `voice_response=voice_response` in `record_experience()` call. |
| `backend/rl/background_trainer.py` | MODIFY | Add Tier 3 state, `_tier3_accumulate()`, `_tier3_maybe_train()`, `_run_text_lora_training()`, `_format_voice_prompt()`, persistence, stats. |

**Unchanged**: `trainer.py`, `export.py`, `views.py`, `apps.py` — all reused as-is.

---
---

# Part 2: Claude Output Supervision Framework

> How to use Claude's output as training signal for local LLaMA models.

---

## 0. Ground Rule

**This is the most important section.**

You do **not** treat Claude's output as "text to copy."

You treat it as:

- **Target distribution** — the shape of correct answers
- **Constraint-satisfying solution** — proof that the constraints can be met
- **Gold decision artifact** — the reference against which all local model outputs are measured

Everything below assumes that framing. The goal is never to make LLaMA *sound like* Claude. The goal is to make LLaMA *solve problems like* Claude — producing outputs that satisfy the same constraints, cover the same requirements, and achieve the same functional outcomes.

---

## 1. Direct Supervision (Controlled, Non-Naive)

### 1.1 Final Answer Supervision

| | |
|---|---|
| **What** | Claude's final answer (code / text / JSON) |
| **Use** | Supervised fine-tuning (SFT) only on final answers |
| **Method** | No chain-of-thought. Deterministic decoding. |
| **Purpose** | Teach what a correct terminal answer looks like. Anchor LLaMA's solution manifold. |

**Critical constraints:**
- Use high-confidence tasks only
- Do not blanket SFT across ambiguous prompts
- Filter for tasks where Claude's answer is verifiably correct

### 1.2 Canonical Solution Mapping

| | |
|---|---|
| **What** | Claude output = canonical solution |
| **Use** | Map multiple LLaMA variants → closest canonical form |
| **Method** | Normalize whitespace, variable names, formatting. Compare semantic equivalence, not tokens. |
| **Purpose** | Collapse solution variance. Reduce "almost right but wrong" outputs. |

---

## 2. Structural Decomposition of Output

### 2.1 Output Segmentation

**Extract from Claude output:**
- Sections
- Headings
- Code blocks
- Ordered steps
- Bullet logic

**Use:** Train LLaMA to emit structurally identical outputs. Penalize missing or reordered sections.

This is huge for Claude-like reliability. Claude's outputs have consistent structure — sections appear in predictable order, steps are numbered when appropriate, code is properly fenced. Teaching this structure is more valuable than teaching the exact words.

### 2.2 Output Grammar / Schema

**Extract from Claude output:**
- JSON schemas
- Function signatures
- CLI command structure
- API response shapes

**Use:** Hard schema validation during training. Binary reward: schema valid / invalid.

Claude is excellent at schema discipline. This is one of the biggest gaps to close. When Claude outputs JSON, it's valid JSON. When it writes a function signature, the types match. LLaMA often gets "close enough" — Tier 3 should penalize that gap hard.

---

## 3. Semantic Content Supervision

### 3.1 Claim Extraction

**Extract from Claude output:**
- Atomic claims
- Assertions
- Guarantees
- Preconditions
- Postconditions

**Use:**
- Teach LLaMA to include the same claims
- Penalize hallucinated extra claims
- Penalize missing mandatory claims

Think: **logical coverage, not wording**. If Claude says "pump-002 is running at 37.5 PSI, which is within normal range", the claims are: (1) pump-002 exists, (2) it's running, (3) pressure is 37.5 PSI, (4) normal range includes 37.5. LLaMA must include all four claims. The phrasing doesn't matter.

### 3.2 Constraint Satisfaction Labels

**Extract from Claude output:**
- Which constraints were satisfied
- Which constraints were explicitly refused
- Which constraints dominated tradeoffs

**Use:** Train LLaMA to produce outputs that satisfy the same constraint set. Reward matching constraint resolution, not phrasing.

Example: If the user asks "show me everything about pump 1" and Claude responds with 3 widgets (not 10), that's a constraint decision — Claude chose relevance over completeness. LLaMA should learn to make the same tradeoff.

### 3.3 Answer Completeness Vector

**Compute from Claude output:**
- Required elements present (yes/no)
- Optional elements included
- Edge cases addressed

**Use:**
- Completeness loss during training
- "Partial answer" penalty

Claude rarely forgets required elements. LLaMA often does. This is one of the highest-leverage training signals — a simple binary checklist of "did you include X?" catches most quality gaps.

---

## 4. Comparative Output Teaching (Claude vs LLaMA)

### 4.1 Output Diffing (Semantic, Not Textual)

**Compute between Claude and LLaMA outputs:**
- Missing components
- Extra components
- Incorrect components

**Use:** Targeted correction training. Generate delta-prompts:

> "Fix only what differs from the reference."

Very effective for convergence. Instead of retraining on the full output, train only on the delta. This prevents catastrophic forgetting of things LLaMA already gets right.

### 4.2 Pairwise Preference Training

**Setup:**
- Same prompt
- Claude output vs LLaMA output
- Label: Claude preferred

**Use:** Preference datasets for RLHF / DPO-style training. No need to expose Claude internals — the output alone is the signal.

This is exactly what Tier 3's text DPO pairs do: for the same (or similar) query, the better voice response is "chosen" and the worse one is "rejected." The DPO training teaches LLaMA to prefer the better response pattern.

---

## 5. Output-Driven Error Modeling

### 5.1 Error Taxonomy from Claude Output

**Extract:**
- What Claude explicitly avoids
- What Claude refuses to do
- What Claude flags as unsafe / invalid

**Use:** Negative training signals. Teach LLaMA what **not** to output.

Example: Claude never speculates about equipment status when data is missing. It says "data unavailable." If LLaMA hallucinates a value, that's a hard negative signal.

### 5.2 Edge-Case Coverage

**Extract:**
- Edge cases mentioned
- Boundary conditions
- Failure modes included

**Use:** Penalize LLaMA outputs that omit these. Strong signal for "senior-level" reasoning.

Claude includes edge cases proactively: "Note: pump-002 was offline for maintenance on Jan 15, which may affect the trend." LLaMA should learn to include the same caveats.

---

## 6. Style-Independent Compression Learning

### 6.1 Minimal Sufficient Answer

**Measure:**
- Claude output length vs information density
- Bits of information per token

**Use:** Train LLaMA to achieve the same information coverage with minimal tokens. Penalize verbosity without new information.

This avoids copying Claude's style while matching its efficiency. A 2-sentence answer that covers 5 data points is better than a 5-sentence answer that covers 3 data points.

**Directly relevant to Tier 3**: The voice response spec is "2-3 sentences for TTS." This is a hard compression constraint. The text quality scorer's "conciseness" dimension enforces this.

---

## 7. Code-Specific Output Supervision

### 7.1 Executability

| | |
|---|---|
| **Check** | Claude code runs. Tests pass. Lint clean. |
| **Use** | Binary reward on execution success. LLaMA trained against executable gold. |

### 7.2 API / Tool Correctness

| | |
|---|---|
| **Extract** | Correct imports, correct API usage, correct flags/parameters |
| **Use** | Penalize "almost correct" code. Hard match on critical calls. |

### 7.3 Refactor Patterns

| | |
|---|---|
| **Extract** | Idiomatic refactors, abstractions, naming consistency |
| **Use** | Teach LLaMA patterns, not exact code. Pattern library distilled from Claude outputs. |

**Relevance to Command Center**: The `claude-rl-agent/` system already captures tool call sequences and can evaluate whether LLaMA's generated code/commands are functionally equivalent to Claude's. This section extends that to the full code quality spectrum.

---

## 8. Output as Oracle for Automatic Tests

### 8.1 Test Case Generation

**Use Claude output to:**
- Generate unit tests
- Generate expected outputs
- Generate invariants

**Then:** Test LLaMA outputs against those tests.

Claude becomes a **test generator**, not just an answerer. This is extremely powerful — Claude writes the acceptance criteria, and LLaMA must pass them.

**Command Center application**: Claude's voice response for "what's the status of pump 1?" implies certain facts must be present. Extract those facts as test assertions. LLaMA's response must pass the same assertions.

---

## 9. Output-Conditioned Prompting (Bootstrapping)

### 9.1 Reverse Prompting

**Use Claude output to infer:**
- Implicit sub-tasks
- Hidden questions answered
- Decomposition strategy

**Use:** Teach LLaMA to internally ask the same sub-questions. Improves decomposition without chain-of-thought leakage.

Example: Claude's response about "transformer efficiency" implicitly answers:
1. What is the current efficiency?
2. What is the baseline efficiency?
3. What's the delta?
4. Is the delta concerning?

LLaMA should learn to decompose the same way, even without explicit prompting.

---

## 10. Output-Anchored Acceptance Gates (Production)

### Gate Definition

**Define from Claude output:**
- "Claude-equivalent acceptance checks"
- Hard constraints derived from reference outputs

### Gate Examples

| Gate | Check | Action |
|------|-------|--------|
| Must include X | Equipment IDs, metric values referenced in query | Block if missing |
| Must not include Y | Hallucinated values, speculative claims, markdown formatting | Block if present |
| Must satisfy Z | Schema validation, sentence count, specificity threshold | Retry if failed |

### Deployment

- LLaMA output is **blocked or retried** if gate fails
- Gates are derived from Claude's output patterns, not hardcoded
- Progressive: start strict, relax as LLaMA improves

**Tier 3 integration**: The `TextQualityScorer` dimensions (groundedness, conciseness, directness, specificity, TTS-friendliness) can serve as acceptance gates at inference time, not just training time. If a generated response scores below threshold, retry with higher temperature or fall back to the quality LLM.

---

## Mapping Framework to Tier 3

| Framework Section | Tier 3 Implementation |
|---|---|
| **0. Ground Rule** | Voice responses treated as functional artifacts, not text to copy |
| **1. Direct Supervision** | SFT baseline from `claude-rl-agent/` SFT checkpoint (optional warm-start) |
| **2. Structural Decomposition** | Text quality scorer enforces 2-3 sentence structure |
| **3. Semantic Content** | Groundedness + specificity dimensions check claim coverage |
| **4. Comparative Teaching** | Text DPO pairs = Claude-preferred vs LLaMA-rejected responses |
| **5. Error Modeling** | Directness dimension penalizes filler; groundedness penalizes hallucination |
| **6. Compression** | Conciseness dimension enforces minimal sufficient answer |
| **7. Code-Specific** | N/A for voice responses (covered by `claude-rl-agent/` for code tasks) |
| **8. Oracle Tests** | Future: extract factual claims from chosen responses as test assertions |
| **9. Reverse Prompting** | Future: infer implicit sub-questions from good voice responses |
| **10. Acceptance Gates** | TextQualityScorer as inference-time gate with retry/fallback |
