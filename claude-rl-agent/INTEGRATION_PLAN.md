# 🔧 Maximum Extraction Integration Plan

**Date**: 2026-02-08
**Goal**: Integrate ALL features from MAXIMUM_EXTRACTION.md into the training pipeline

---

## Feature Catalog (35 Total Features)

### ✅ Section A: Explicit Response Content (4 features)
| # | Feature | Status | Location | Integration Needed |
|---|---------|--------|----------|-------------------|
| 1 | Final text answer | ✅ USED | ClaudeTrace.claude_response | None |
| 2 | Structured/JSON outputs | ✅ USED | ReasoningSignals.response_format | None |
| 3 | Multiple-choice/ranked options | ✅ IMPL | PreferenceRanking | automated_runner, reward_model |
| 4 | Tool calls & actions | ✅ USED | ToolCall list | None |

### ✅ Section B: Behavioral/Reasoning Signals (7 features)
| # | Feature | Status | Location | Integration Needed |
|---|---------|--------|----------|-------------------|
| 5 | Step decomposition | ✅ USED | ReasoningSignals.reasoning_steps | None |
| 6 | Self-corrections | ✅ USED | ReasoningSignals.self_corrections | None |
| 7 | Assumption statements | ✅ IMPL | AssumptionStatement | automated_runner, reward_model, builder |
| 8 | Confidence signals | ✅ IMPL | SelfCritique.confidence_level | automated_runner, reward_model |
| 9 | Constraint extraction | ✅ USED | ReasoningSignals.constraints_detected | None |
| 10 | Error validation checks | ✅ IMPL | ErrorValidationCheck | automated_runner, reward_model, builder |
| 11 | Counterfactual paths | ✅ IMPL | CounterfactualPath | automated_runner, reward_model, builder |

### 🔄 Section C: Structural/Token-level (4 features)
| # | Feature | Status | Location | Integration Needed |
|---|---------|--------|----------|-------------------|
| 12 | Token-level logprobs | ⏳ API | TokenConfidence (stub) | Requires Anthropic API - skip for now |
| 13 | Response length/time | ✅ USED | ClaudeTrace.response_time_ms | None |
| 14 | Special channels (code/markdown) | ✅ USED | Parsed in extractor | None |
| 15 | Edit history | ✅ IMPL | EditHistoryEntry | automated_runner, builder |

### ✅ Section D: Interaction & Tooling (3 features)
| # | Feature | Status | Location | Integration Needed |
|---|---------|--------|----------|-------------------|
| 16 | Tool sequence/outputs | ✅ USED | ReasoningSignals.tool_sequence | None |
| 17 | Provenance tracking | ✅ IMPL | ProvenanceRecord | automated_runner, reward_model, builder |
| 18 | Agent chain-of-actions | ✅ USED | Tool parallel execution | None |

### ✅ Section E: Evaluative/Meta (3 features)
| # | Feature | Status | Location | Integration Needed |
|---|---------|--------|----------|-------------------|
| 19 | Self-critique | ✅ IMPL | SelfCritique | automated_runner, reward_model, builder |
| 20 | Preference rankings | ✅ IMPL | PreferenceRanking | automated_runner, reward_model |
| 21 | Safety/refusal signals | ✅ IMPL | SafetySignal | automated_runner, reward_model, builder |

### 🎯 Section F: Derived Numeric Features (2 features)
| # | Feature | Status | Location | Integration Needed |
|---|---------|--------|----------|-------------------|
| 22 | **Reasoning Vector (35-dim)** | ✅ IMPL | ReasoningVector | **CRITICAL**: automated_runner, reward_model |
| 23 | Outcome metrics | ✅ USED | ClaudeTrace.task_completed | None |

### 📚 Section G: Additional Cognitive Patterns (12 features)
| # | Feature | Status | Extraction Method | Integration Needed |
|---|---------|--------|------------------|-------------------|
| 24 | Planning strategies | 🔄 PARTIAL | Extracted in reasoning_steps | Enhance detection |
| 25 | Pattern recognition | 🔄 PARTIAL | Code blocks analysis | Add explicit extraction |
| 26 | Decision criteria & trade-offs | ✅ IMPL | CounterfactualPath | automated_runner, builder |
| 27 | Domain knowledge application | ✅ IMPL | ProvenanceRecord | automated_runner, builder |
| 28 | Error handling approaches | ✅ IMPL | ErrorValidationCheck, self_corrections | automated_runner, builder |
| 29 | Performance considerations | 🔄 PARTIAL | Response time | Add complexity analysis |
| 30 | Communication patterns | 🔄 PARTIAL | Text analysis | Add explicit extraction |
| 31 | Code quality awareness | 🔄 PARTIAL | Code blocks | Add quality scoring |
| 32 | Testing & verification | ✅ IMPL | ErrorValidationCheck | automated_runner, builder |
| 33 | Architecture & design | ✅ IMPL | CounterfactualPath (alternatives) | automated_runner, builder |
| 34 | Context management | ✅ USED | working_directory, tool_sequence | None |
| 35 | Iterative improvement | ✅ IMPL | self_corrections, edit_history | automated_runner, builder |

---

## Integration Tasks

### Phase 1: Core Integration (Now)
**Goal**: Get enhanced extraction working in all pipelines

#### 1.1 Update automated_runner.py
- ✅ Replace `ReasoningSignalExtractor` with `EnhancedSignalExtractor`
- ✅ Add enhanced signal comparison methods
- ✅ Use 35-dim reasoning vector for similarity scoring
- ✅ Compare assumptions, validations, counterfactuals, provenance, safety
- ✅ Add enhanced signals to DPO pair generation

#### 1.2 Update behavioral_cloning_builder.py
- ✅ Import enhanced_extractor
- ✅ Apply enhanced extraction to all traces
- ✅ Include enhanced signals in training samples
- ✅ Store reasoning vectors for analysis

#### 1.3 Update reward_model.py
- ✅ Import ReasoningVector
- ✅ Use `reasoning_vector.to_numpy()` for reward computation
- ✅ Add reward components for:
  - Assumption clarity (0-1)
  - Validation completeness (0-1)
  - Counterfactual consideration (0-1)
  - Provenance citations (count)
  - Safety alignment (0-1)
- ✅ Update multi-objective reward formula

#### 1.4 Update fast_track_bootstrap.py (Optional Enhancement)
- ✅ Add enhanced signals to synthetic traces
- ✅ Generate realistic assumptions
- ✅ Generate validation checks
- ✅ Generate counterfactual paths
- ✅ Generate provenance records

### Phase 2: Enhanced Comparison Logic (Now)
**Goal**: Deep behavioral comparison using all 35 dimensions

#### 2.1 Reasoning Vector Comparison
```python
def compare_reasoning_vectors(claude_vec, llama_vec):
    """
    Compare 35-dimensional behavioral vectors.

    Returns:
        - Overall distance (0-1)
        - Per-dimension scores
        - Critical divergences
    """
```

#### 2.2 Enhanced Signal Comparison
```python
def compare_assumptions(claude_assumptions, llama_assumptions):
    """Compare assumption clarity and count."""

def compare_validations(claude_checks, llama_checks):
    """Compare validation completeness."""

def compare_counterfactuals(claude_paths, llama_paths):
    """Compare alternative consideration."""

def compare_provenance(claude_prov, llama_prov):
    """Compare source citation."""

def compare_safety(claude_safety, llama_safety):
    """Compare safety signal alignment."""
```

### Phase 3: Testing & Validation
- ✅ Test enhanced extraction on sample traces
- ✅ Verify 35-dim vectors are generated correctly
- ✅ Test comparison logic
- ✅ Verify DPO pairs include enhanced signals
- ✅ Run end-to-end pipeline test

---

## Expected Improvements

### Before Enhanced Integration:
- **Extraction Coverage**: ~50% (11/22 extractable features)
- **Behavioral Similarity**: 50-60% (SFT only, base signals)
- **Reward Model Dimensions**: ~11 behavioral signals
- **Comparison Granularity**: Coarse (tool sequence, steps, constraints)

### After Enhanced Integration:
- **Extraction Coverage**: **95%** (21/22 - only token logprobs missing)
- **Behavioral Similarity**: **65-75%** (SFT + enhanced signals)
- **Reward Model Dimensions**: **35 dimensions** (comprehensive behavioral profile)
- **Comparison Granularity**: **Fine-grained** (assumptions, validations, counterfactuals, provenance, safety, critique)

### After PPO with Enhanced Signals:
- **Behavioral Similarity**: **75-85%** (RL alignment on 35 dimensions)

### After DPO on Real Comparisons:
- **Behavioral Similarity**: **85-95%** (continuous improvement)

---

## Critical Files to Modify

### 1. automated_runner.py (Priority: CRITICAL)
**Lines to change**:
- Line 2-10: Imports
  ```python
  from enhanced_extractor import EnhancedSignalExtractor, enhance_trace_with_maximum_extraction
  from enhanced_extraction import (
      EnhancedReasoningSignals, ReasoningVector,
      AssumptionStatement, ErrorValidationCheck, CounterfactualPath,
      ProvenanceRecord, SafetySignal, SelfCritique
  )
  ```
- Line 116: Constructor
  ```python
  self.extractor = EnhancedSignalExtractor()  # Use enhanced!
  ```
- Line 145-271: Add enhanced comparison methods
- Line 340+: Enhance DPO pair creation with enhanced signals

### 2. behavioral_cloning_builder.py (Priority: HIGH)
**Additions needed**:
- Import enhanced_extractor
- Apply enhanced extraction to all loaded traces
- Include enhanced signals in formatted training samples
- Store reasoning vectors for analysis

### 3. reward_model.py (Priority: HIGH)
**Additions needed**:
- Import ReasoningVector
- Update compute_behavioral_reward() to use 35-dim vectors
- Add new reward components for enhanced signals
- Update multi-objective formula

### 4. fast_track_bootstrap.py (Priority: MEDIUM)
**Enhancements**:
- Generate realistic assumptions per workflow
- Generate validation checks
- Generate counterfactual paths
- Generate provenance records

---

## Implementation Timeline

### Now (Next 30-60 minutes):
1. ✅ Integrate EnhancedSignalExtractor into automated_runner.py
2. ✅ Add enhanced comparison logic
3. ✅ Update behavioral_cloning_builder.py
4. ✅ Update reward_model.py
5. ✅ Test integration

### After Integration Complete:
- Option 1: Export SFT model directly (current training)
- Option 2: Run PPO with enhanced reward model (better quality)

---

## Success Criteria

✅ **Integration Complete When**:
1. automated_runner.py uses EnhancedSignalExtractor
2. Reasoning vectors (35-dim) are generated and compared
3. All enhanced signals (assumptions, validations, counterfactuals, provenance, safety) are extracted
4. reward_model.py uses 35-dim vectors for rewards
5. behavioral_cloning_builder.py includes enhanced signals
6. End-to-end test passes

✅ **Validation**:
- Run automated_runner on 1 sample prompt
- Verify 35-dim vector is generated
- Verify enhanced comparison output shows all signals
- Verify DPO pairs include enhanced data

---

**Status**: Ready to begin integration
**Next Step**: Start with automated_runner.py integration
