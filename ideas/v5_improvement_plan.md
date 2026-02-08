# Command Center V5 Dashboard Generation: Comprehensive Improvement Plan

**Status:** Draft for Review
**Date:** 2026-02-08
**Author:** Analysis from widget-test analytics system
**Goal:** Fix V5 orchestrator pipeline to achieve V2/V3 quality (9.7/10 → 9.7/10)

---

## Executive Summary

**Current State:**
- V5 composite score: **3.1/10** (FAILING)
- 89/100 dashboards exceed 12-row viewport (avg 25 rows, max 32)
- 3,094 data schema errors (1,025 critical crashes)
- 11 complete failures (voice-only, no widgets)
- 15 widgets/dashboard vs ideal 8-12

**Target State:**
- V5 composite score: **9.5+/10** (PASSING)
- 100% viewport fit (max 12 rows)
- Zero data errors (validated schemas)
- Zero failures (graceful degradation)
- 8-12 widgets/dashboard (optimal density)

**Required Changes:** 6 fixes across 4 components, estimated **20 hours total** engineering effort

---

## Root Cause Analysis

### 1. **No Widget Count Constraint (CRITICAL)** 🔴

**What:** Orchestrator selects 15-19 widgets per dashboard with no upper limit
**Why:** `backend/layer2/widget_selector.py:select_widgets()` has no `max_widgets` parameter
**Impact:** Viewport overflow (89/100), poor storytelling (density 4.9/10)
**Score Impact:** -3.9 points (viewport: 1.1/10, density: 4.9/10)

**Technical Details:**
```python
# Current (broken)
def select_widgets(intent, available_widgets, rl_scorer):
    llm_proposals = llm_select(intent, available_widgets)  # Returns 15-20
    scores = rl_scorer.score(llm_proposals)
    return sorted(llm_proposals, key=lambda w: scores[w], reverse=True)
    # ❌ No truncation to max_widgets

# Fixed
def select_widgets(intent, available_widgets, rl_scorer, max_widgets=12):
    llm_proposals = llm_select(intent, available_widgets)
    scores = rl_scorer.score(llm_proposals)
    ranked = sorted(llm_proposals, key=lambda w: scores[w], reverse=True)
    return ranked[:max_widgets]  # ✅ Enforce limit
```

**Why LLM Over-Selects:**
- Llama 3.1 8B is trained to be "helpful" → tends to include more information when uncertain
- RL scorer optimizes for "completeness" (include all relevant widgets) not "conciseness"
- No negative reward for widget count in reward_signals.py

---

### 2. **No Viewport Constraint in Grid Packing (CRITICAL)** 🔴

**What:** Grid packing algorithm has no max_row=12 budget check
**Why:** `generate_dashboards_v5.py:compute_grid_positions()` just keeps adding rows
**Impact:** Avg 25 rows (2.1× viewport), 89/100 exceed limit
**Score Impact:** -8.9 points (viewport: 1.1/10)

**Technical Details:**
```python
# Current (broken)
def compute_grid_positions(widgets):
    current_row = 1
    current_col = 1
    positions = []

    for w in widgets:
        cols = SIZE_TO_COLS[w["size"]]
        rows = SIZE_TO_ROWS[w["size"]]

        if current_col + cols > 13:  # Overflow column
            current_row += rows  # ❌ No check if current_row > 12
            current_col = 1

        positions.append({"col": f"{current_col}/{current_col+cols}",
                          "row": f"{current_row}/{current_row+rows}"})
        current_col += cols

    return positions  # ❌ May return 25+ rows

# Fixed
def compute_grid_positions(widgets, max_row=12):
    current_row = 1
    current_col = 1
    positions = []
    budget_used = 0

    for w in widgets:
        cols = SIZE_TO_COLS[w["size"]]
        rows = SIZE_TO_ROWS[w["size"]]

        if current_col + cols > 13:
            current_row += rows
            current_col = 1

        # ✅ Check budget before placing
        if current_row + rows > max_row:
            print(f"Budget exceeded, dropping widget {w['id']}")
            break  # Drop remaining widgets

        positions.append({"col": f"{current_col}/{current_col+cols}",
                          "row": f"{current_row}/{current_row+rows}"})
        current_col += cols
        budget_used = current_row + rows

    return positions  # ✅ Guaranteed ≤ 12 rows
```

**Interaction with Fix #1:**
- If fix #1 limits widgets to 12, AND those 12 widgets fit in 12 rows → perfect
- If 12 widgets don't fit (e.g., all "x-tall"), fix #2 drops the overflow
- Best: fix #1 (limit count) + fix #2 (enforce rows) + fix #4 (resize)

---

### 3. **demoData Schema Mismatch (HIGH)** 🟠

**What:** Orchestrator wraps data in `{ demoData: {...} }` but widgets read top-level fields
**Why:** `backend/layer2/data_collector.py` uses internal format, V5 generator doesn't normalize
**Impact:** 2,069 "missing field" errors, 1,025 crash risks
**Score Impact:** -10.0 points (data: 0.0/10)

**Schema Mismatch Example:**
```javascript
// What orchestrator returns (wrong)
widget.data_override = {
  demoData: {
    label: "Main Incomer Power",
    value: 450.3,
    unit: "kW"
  }
}

// What kpi.tsx expects (correct)
widget.data = {
  layout: {...},
  visual: {...},
  demoData: {
    label: "Main Incomer Power",
    value: 450.3,
    unit: "kW"
  }
}

// Widget code reads: data.demoData.label ✅ BUT
// Orchestrator provides: data_override.demoData.label ❌
```

**Why This Happens:**
1. Orchestrator's `data_collector.py:build_data_override()` returns `{ demoData: {...} }`
2. V5 generator (`generate_dashboards_v5.py:convert_widget()`) directly copies this to `widget.data`
3. But widget components were written for V4 format which has `layout`, `visual`, `demoData` sibling keys
4. Result: widgets crash because `layout` and `visual` are missing

**Fix Location:**
```python
# generate_dashboards_v5.py:convert_widget()
def convert_widget(orch_widget):
    widget_type = orch_widget["scenario"]
    data_override = orch_widget.get("data_override", {})

    # ❌ Current (wrong)
    widget_data = data_override

    # ✅ Fixed (normalize schema)
    if "demoData" in data_override and "layout" not in data_override:
        # Orchestrator format: wrap in proper structure
        widget_data = {
            "layout": DEFAULT_LAYOUTS[widget_type],
            "visual": DEFAULT_VISUALS[widget_type],
            "demoData": data_override["demoData"],
            "variant": data_override.get("variant", DEFAULT_VARIANTS[widget_type])
        }
    else:
        # Already normalized (shouldn't happen but be safe)
        widget_data = data_override

    return {
        "id": orch_widget["widget_id"],
        "type": widget_type,
        "data": widget_data,
        ...
    }
```

**Alternative Fix (better long-term):**
Change orchestrator's `data_collector.py` to return full widget schema:
```python
# backend/layer2/data_collector.py:build_data_override()
def build_data_override(widget_scenario, query_results):
    return {
        "layout": get_layout_for_scenario(widget_scenario),
        "visual": get_visual_theme_for_scenario(widget_scenario),
        "variant": infer_variant(widget_scenario, query_results),
        "demoData": {
            "label": query_results.get("label"),
            "value": query_results.get("value"),
            # ... etc
        }
    }
```

---

### 4. **L-Shaped Grid Gaps from Height Mismatch (MEDIUM)** 🟡

**What:** Side-by-side widgets have different row spans → gaps below shorter widget
**Why:** Simple left-to-right packing, no height equalization
**Impact:** 15% grid waste, 89/100 V5 dashboards have gaps
**Score Impact:** -4.8 points (grid: 5.0/10, avg coverage 89.8%)

**Visual Example:**
```
Row 1: [flow-sankey (6 cols, 4 rows)][composition (6 cols, 3 rows)]
Row 2: [flow-sankey                 ][composition              ]
Row 3: [flow-sankey                 ][composition              ]
Row 4: [flow-sankey                 ][ ❌ GAP (1 row)          ]
Row 5: [next widget starts here]
```

**Fix: Equalize Heights**
```python
def compute_grid_positions(widgets, max_row=12):
    positions = []
    current_row = 1
    i = 0

    while i < len(widgets) and current_row <= max_row:
        w1 = widgets[i]
        cols1 = SIZE_TO_COLS[w1["size"]]
        rows1 = SIZE_TO_ROWS[w1["size"]]

        # Check if next widget fits side-by-side
        if i + 1 < len(widgets):
            w2 = widgets[i + 1]
            cols2 = SIZE_TO_COLS[w2["size"]]
            rows2 = SIZE_TO_ROWS[w2["size"]]

            if cols1 + cols2 <= 12:  # Can fit side-by-side
                # ✅ Equalize heights
                max_rows = max(rows1, rows2)

                positions.append({
                    "col": f"1/{1+cols1}",
                    "row": f"{current_row}/{current_row+max_rows}"
                })
                positions.append({
                    "col": f"{1+cols1}/{1+cols1+cols2}",
                    "row": f"{current_row}/{current_row+max_rows}"
                })

                current_row += max_rows
                i += 2  # Consumed both widgets
                continue

        # Single widget (full width or couldn't pair)
        positions.append({
            "col": f"1/{1+cols1}",
            "row": f"{current_row}/{current_row+rows1}"
        })
        current_row += rows1
        i += 1

    return positions
```

**Trade-off:**
- **Pro:** Eliminates L-shaped gaps, improves grid coverage to 97%+
- **Con:** May make some widgets taller than needed (visual balance issue)
- **Alternative:** Insert filler widgets (small narratives/KPIs) to fill gaps

---

### 5. **Missing Cost/Environmental Tables (LOW)** 🔵

**What:** No PostgreSQL tables for billing, demand, CO2, VFD monitoring
**Why:** Database schema focused on equipment telemetry only
**Impact:** 11/100 queries fail (voice-only response)
**Score Impact:** -0.0 points (doesn't affect score, but 11% failure rate)

**Missing Query Types:**
- Cost: "Are we at risk of exceeding contract demand?" → needs `billing_demand` table
- Cost: "Which areas contribute most to demand charges?" → needs `demand_breakdown` table
- Environmental: "How does environmental compliance track?" → needs `environmental_sensors` table
- Quality: "What is the frequency stability?" → needs `frequency_monitoring` table
- HVAC: "Which zones have highest CO2?" → needs `air_quality` table
- Backup: "How much DG fuel do we have?" → needs `fuel_inventory` table
- Exec: "What is the VFD fleet performance?" → needs `vfd_monitoring` table

**Fix Options:**

**Option A: Add Synthetic Tables (Short-term, 6 hours)**
```python
# rl_training_data/generate_schema.py - add new tables
COST_TABLES = [
    """
    CREATE TABLE billing_demand (
        timestamp TIMESTAMPTZ,
        total_demand_kw FLOAT,
        contract_demand_kw FLOAT,
        demand_charge_inr FLOAT,
        utilization_pct FLOAT,
        penalty_risk VARCHAR(10)
    );
    """,
    """
    CREATE TABLE demand_breakdown (
        timestamp TIMESTAMPTZ,
        area VARCHAR(50),
        demand_kw FLOAT,
        contribution_pct FLOAT
    );
    """
]

# generate_data_gpu.py - generate realistic data
def generate_billing_demand(timestamps):
    contract_demand = 5000  # kW
    demand = contract_demand * (0.7 + 0.25 * np.sin(...))  # Daily pattern
    utilization = demand / contract_demand * 100
    penalty_risk = "HIGH" if utilization > 95 else "LOW"
    ...
```

**Option B: Graceful Degradation (Long-term, 8 hours)**
```python
# backend/layer2/orchestrator.py:handle_missing_tables()
def orchestrate(request):
    ...
    widgets = select_widgets(intent, max_widgets=12)

    widgets_with_data = []
    for w in widgets:
        try:
            data = collect_data(w, intent)
            widgets_with_data.append({...w, data_override: data})
        except TableNotFoundError:
            # Graceful fallback: show partial dashboard
            print(f"Table missing for {w}, skipping widget")
            continue  # ✅ Don't fail entire request

    if len(widgets_with_data) >= 4:  # At least 4 widgets succeeded
        return build_layout(widgets_with_data) + voice_response
    else:
        return voice_only_response  # Only fail if too few widgets
```

**Recommendation:** Start with Option A (add synthetic tables) for immediate fix, then implement Option B for robustness.

---

### 6. **Poor Storytelling Structure (MEDIUM)** 🟡

**What:** Widgets selected independently, no enforced narrative flow
**Why:** `widget_selector.py` has no storytelling template
**Impact:** Storytelling score 0.45 vs V3's 0.87
**Score Impact:** -4.2 points (story: 4.5/10)

**V3's 4-Layer Template (What Works):**
```
┌────────────────────────────────────┐
│ LAYER 1: Header (rows 1-2)        │  ← 2-3 KPIs + opening narrative
├────────────────────────────────────┤
│ LAYER 2: Evidence (rows 3-6)      │  ← Main analytical charts (trend, distribution)
├────────────────────────────────────┤
│ LAYER 3: Breakdown (rows 7-10)    │  ← Detailed views (composition, heatmap, sankey)
├────────────────────────────────────┤
│ LAYER 4: Temporal (rows 11-12)    │  ← Trends + closing narrative
└────────────────────────────────────┘
```

**Fix: Template-Based Selection**
```python
# backend/layer2/widget_selector.py
STORYTELLING_TEMPLATE = {
    "layer1_header": {
        "types": ["kpi", "narrative"],
        "count": (2, 3),  # 2-3 widgets
        "rows": (1, 2),
        "priority": "critical"
    },
    "layer2_evidence": {
        "types": ["trend", "distribution", "timeline"],
        "count": (2, 3),
        "rows": (3, 6),
        "priority": "high"
    },
    "layer3_breakdown": {
        "types": ["composition", "flow-sankey", "matrix-heatmap", "comparison"],
        "count": (3, 4),
        "rows": (7, 10),
        "priority": "medium"
    },
    "layer4_temporal": {
        "types": ["trend-multi-line", "trends-cumulative", "narrative"],
        "count": (2, 3),
        "rows": (11, 12),
        "priority": "medium"
    }
}

def select_widgets_with_template(intent, available_widgets, rl_scorer):
    selected = []

    for layer_name, layer_spec in STORYTELLING_TEMPLATE.items():
        # Filter widgets matching this layer's types
        candidates = [w for w in available_widgets if w["type"] in layer_spec["types"]]

        # LLM + RL score candidates
        scores = rl_scorer.score_for_intent(candidates, intent)
        ranked = sorted(candidates, key=lambda w: scores[w], reverse=True)

        # Select top N for this layer
        min_count, max_count = layer_spec["count"]
        layer_widgets = ranked[:max_count]

        # Assign to layer's row band
        for w in layer_widgets:
            w["layer"] = layer_name
            w["row_band"] = layer_spec["rows"]

        selected.extend(layer_widgets)

    return selected[:12]  # Enforce max 12 total
```

**RL Reward Update:**
```python
# backend/rl/reward_signals.py
def compute_storytelling_reward(widget_layout):
    # Check if KPIs are in top 2 rows
    kpi_header_score = sum(1 for w in widget_layout
                           if w["type"] == "kpi" and parse_row(w["row"])[0] <= 2)

    # Check narrative bookending
    narratives = [w for w in widget_layout if w["type"] == "narrative"]
    has_opening = any(parse_row(n["row"])[0] <= 2 for n in narratives)
    has_closing = any(parse_row(n["row"])[1] >= 11 for n in narratives)
    bookend_score = (has_opening + has_closing) / 2

    # Progressive disclosure: header → evidence → breakdown → temporal
    type_order = [w["type"] for w in sorted(widget_layout, key=lambda w: parse_row(w["row"])[0])]
    disclosure_score = compute_order_score(type_order)

    return (kpi_header_score + bookend_score + disclosure_score) / 3
```

---

## Implementation Roadmap

### **Phase 1: Critical Fixes (Week 1, 8 hours)**

**Priority 1.1: Widget Count Constraint (2 hours)**
- **File:** `backend/layer2/widget_selector.py`
- **Change:** Add `max_widgets=12` parameter, truncate ranked list
- **Testing:** Run 100 V4 questions, verify all ≤12 widgets
- **Success:** Avg widget count drops from 15.0 to 10.5

**Priority 1.2: Viewport Budget in Grid Packing (3 hours)**
- **File:** `generate_dashboards_v5.py`
- **Change:** Add `max_row=12` check in `compute_grid_positions()`
- **Testing:** Visual check 10 dashboards, verify all ≤12 rows
- **Success:** 0/100 exceed viewport (vs current 89/100)

**Priority 1.3: Schema Normalization (3 hours)**
- **File:** `generate_dashboards_v5.py:convert_widget()`
- **Change:** Unwrap `demoData` and add `layout`/`visual` wrappers
- **Testing:** Run data validation test, verify 0 critical errors
- **Success:** Data score 10.0/10 (vs current 0.0/10)

**Expected Outcome:** V5 score improves from **3.1/10 → 8.5/10**

---

### **Phase 2: Quality Improvements (Week 2, 8 hours)**

**Priority 2.1: Height Equalization (4 hours)**
- **File:** `generate_dashboards_v5.py:compute_grid_positions()`
- **Change:** Detect side-by-side pairs, equalize row spans
- **Testing:** Grid analysis, verify coverage >95%
- **Success:** Grid score 9.5+/10 (vs current 5.0/10)

**Priority 2.2: Storytelling Template (4 hours)**
- **File:** `backend/layer2/widget_selector.py`
- **Change:** Add 4-layer template constraint
- **File:** `backend/rl/reward_signals.py`
- **Change:** Add storytelling reward signal
- **Testing:** Manual review of 10 dashboards, verify narrative flow
- **Success:** Storytelling score 0.80+ (vs current 0.45)

**Expected Outcome:** V5 score improves from **8.5/10 → 9.5/10**

---

### **Phase 3: Robustness (Week 3-4, 6 hours)**

**Priority 3.1: Add Cost/Environmental Tables (6 hours)**
- **Files:** `rl_training_data/generate_schema.py`, `generate_data_gpu.py`
- **Change:** Add 7 new tables (billing, demand, env sensors, VFD, fuel, etc.)
- **Testing:** Run 11 failing questions, verify widget generation
- **Success:** 0/100 voice-only failures (vs current 11/100)

**Expected Outcome:** V5 achieves **100% success rate** on all query types

---

## Testing & Validation

### **Unit Tests**
```bash
# Test 1: Widget count constraint
python3 -m pytest backend/layer2/tests/test_widget_selector.py::test_max_widgets

# Test 2: Viewport budget
python3 -m pytest generate_dashboards_v5_test.py::test_viewport_fit

# Test 3: Schema normalization
python3 -m pytest generate_dashboards_v5_test.py::test_data_schema

# Test 4: Height equalization
python3 -m pytest generate_dashboards_v5_test.py::test_grid_packing
```

### **Integration Test**
```bash
# Run full V5 generation with fixes
cd /home/rohith/desktop/widget-test
python3 generate_dashboards_v5_fixed.py --limit 100

# Verify results
python3 analyze_dashboards.py --versions v5_fixed
# Expected: composite score 9.5+/10
```

### **Visual Review**
- Open 10 random V5 dashboards in `http://localhost:5173/dashboards/v5_fixed/:id`
- Check: viewport fit (no scroll), grid coverage (no gaps), data rendering (no errors)
- Compare side-by-side with V3 equivalent

### **A/B Test (Future)**
- Deploy V5-fixed in production
- Route 50% traffic to V5, 50% to V4
- Collect user feedback + RL experiences for 1 week
- Measure: avg session time, interactions per dashboard, positive feedback %

---

## Technical Debt & Long-Term Improvements

### **1. Viewport-Aware Widget Selection**
Currently: Selector picks widgets, packing drops overflow
Better: Selector knows row budget, picks widgets that fit

**Implementation:**
```python
def select_widgets(intent, available_widgets, rl_scorer, max_row=12):
    ranked = rank_by_score(available_widgets, intent, rl_scorer)

    selected = []
    row_budget = 0

    for w in ranked:
        estimated_rows = estimate_row_span(w)
        if row_budget + estimated_rows <= max_row:
            selected.append(w)
            row_budget += estimated_rows
        else:
            break  # Budget exhausted

    return selected
```

---

### **2. Constraint-Based Grid Packing (Advanced)**
Current: Greedy left-to-right packing
Better: Constraint solver (OR-Tools) for optimal layout

**Problem Formulation:**
- Variables: `x[i,j]` = 1 if widget i placed in cell (row, col) j
- Objective: Maximize coverage % + minimize gaps
- Constraints:
  - Each widget placed exactly once
  - No overlaps
  - Max row ≤ 12
  - Adjacent widgets have equal heights (optional)

**Library:** Google OR-Tools (Python CP-SAT solver)
**Complexity:** 50-100ms solve time for 12 widgets, acceptable latency

---

### **3. RL Reward Tuning for Viewport Fit**
Current RL reward doesn't penalize widget count or height
Better: Add negative reward for exceeding budget

```python
# backend/rl/reward_signals.py
def compute_viewport_fit_reward(widget_layout):
    total_rows = max(parse_row(w["row"])[1] for w in widget_layout)

    if total_rows <= 12:
        return 1.0  # Perfect fit
    else:
        overflow = total_rows - 12
        return max(0, 1.0 - overflow * 0.1)  # -0.1 per overflow row

def compute_density_reward(widget_layout):
    count = len(widget_layout)
    if 8 <= count <= 12:
        return 1.0
    elif count < 8:
        return count / 8  # Penalize too sparse
    else:
        return max(0, 1.0 - (count - 12) * 0.1)  # Penalize too dense
```

---

### **4. Unified Schema Definition (pydantic)**
Current: Schema defined implicitly in widget components
Better: Explicit schema validation with pydantic

```python
# backend/layer2/schemas.py
from pydantic import BaseModel, Field

class KpiData(BaseModel):
    layout: dict
    visual: dict
    variant: str
    demoData: dict = Field(..., description="Contains label, value, unit, state")

class WidgetSchema(BaseModel):
    id: str
    type: str
    col: str  # Format: "start/end"
    row: str
    data: Union[KpiData, TrendData, ...]  # Type-safe union

# Usage in data_collector.py
def build_data_override(widget_scenario, query_results) -> KpiData:
    return KpiData(
        layout=get_layout_for_scenario(widget_scenario),
        visual=get_visual_theme(widget_scenario),
        variant=infer_variant(widget_scenario),
        demoData={
            "label": query_results["label"],
            "value": query_results["value"],
            "unit": query_results["unit"],
            "state": query_results.get("state", "normal")
        }
    )
```

**Benefits:**
- Runtime validation (catches schema errors before rendering)
- Auto-generated OpenAPI docs
- IDE autocomplete for data fields

---

## Risk Analysis

### **High Risk:**
- **Schema normalization breaking existing V4 dashboards:** Mitigation: Test V4 after changes, ensure backward compatibility
- **Widget count limit too restrictive for complex queries:** Mitigation: Make max_widgets configurable (8-15 range)

### **Medium Risk:**
- **Height equalization making widgets look stretched:** Mitigation: Add max stretch ratio (e.g., don't stretch 1-row KPI to 4 rows)
- **Storytelling template too rigid:** Mitigation: Use soft constraints (prefer template but allow deviation)

### **Low Risk:**
- **Grid packing performance:** Greedy algorithm is O(n), fast enough
- **Database table generation:** Synthetic data is acceptable for demo

---

## Success Metrics

### **Quantitative (from analyze_dashboards.py)**
| Metric | Current (V5) | Target | V2/V3 Baseline |
|--------|--------------|--------|----------------|
| Composite Score | 3.1/10 | 9.5+/10 | 9.7/10 |
| Grid Score | 5.0/10 | 9.5+/10 | 9.8/10 |
| Data Score | 0.0/10 | 10.0/10 | 10.0/10 |
| Story Score | 4.5/10 | 8.5+/10 | 8.7/10 |
| Viewport Fit % | 11% | 100% | 100% |
| Avg Widget Count | 15.0 | 10.5 | 10.5 |
| Data Error Rate | 2.07 errors/widget | 0.0 | 0.0 |
| Voice-Only Failures | 11/100 | 0/100 | 0/10 |

### **Qualitative (Manual Review)**
- No visible gaps in grid layout
- Narrative flow is logical (KPI → Evidence → Breakdown → Trends)
- Data renders correctly (no crashes or "undefined")
- Dashboard fits single viewport (no scrolling)
- Widget density feels balanced (not too sparse or crowded)

---

## Estimated Timeline

**Total: 3-4 weeks (20-24 hours engineering)**

| Week | Phase | Tasks | Hours | Outcome |
|------|-------|-------|-------|---------|
| 1 | Critical Fixes | max_widgets, viewport budget, schema norm | 8h | V5: 8.5/10 |
| 2 | Quality | Height equalization, storytelling template | 8h | V5: 9.5/10 |
| 3 | Robustness | Cost/env tables | 6h | V5: 100% success |
| 4 | Testing & Docs | Unit tests, integration tests, docs | 4h | Production-ready |

**Dependencies:**
- Week 2 depends on Week 1 (schema must be fixed before grid packing)
- Week 3 can run in parallel with Week 2
- Week 4 depends on Weeks 1-3 completion

---

## Appendix A: File Modification Summary

### **Backend Changes**
| File | Lines Changed | Change Type | Risk |
|------|---------------|-------------|------|
| `backend/layer2/widget_selector.py` | ~30 | Feature (add param) | Low |
| `backend/layer2/data_collector.py` | ~50 | Refactor (schema) | Medium |
| `backend/rl/reward_signals.py` | ~40 | Feature (new signals) | Low |
| `rl_training_data/generate_schema.py` | ~200 | Feature (new tables) | Low |
| `rl_training_data/generate_data_gpu.py` | ~300 | Feature (data gen) | Low |

### **Generator Changes**
| File | Lines Changed | Change Type | Risk |
|------|---------------|-------------|------|
| `generate_dashboards_v5.py:convert_widget()` | ~40 | Refactor (normalize) | Medium |
| `generate_dashboards_v5.py:compute_grid_positions()` | ~80 | Refactor (packing) | High |

### **Test Changes**
| File | Lines Changed | Change Type | Risk |
|------|---------------|-------------|------|
| `backend/layer2/tests/test_widget_selector.py` | ~60 | New tests | Low |
| `generate_dashboards_v5_test.py` | ~150 | New tests | Low |

**Total:** ~950 lines changed across 9 files

---

## Appendix B: Alternative Approaches Considered

### **Alternative 1: Fixed-Height Canvas (Frontend Fix)**
**Idea:** Set dashboard container to `height: 100vh; overflow: hidden` in CSS
**Pro:** No backend changes, immediate fix
**Con:** Widgets get cut off if they overflow, looks broken
**Verdict:** ❌ Not viable, breaks user experience

### **Alternative 2: Dynamic Viewport (No Limit)**
**Idea:** Allow scrolling, don't enforce 12-row limit
**Pro:** Fits user's original requirement ("show all info")
**Con:** User explicitly said "100vh × 100vw, no scroll" — violates requirement
**Verdict:** ❌ Doesn't meet user's stated preference (V3-style single viewport)

### **Alternative 3: Widget Resizing (Shrink to Fit)**
**Idea:** If widgets overflow 12 rows, shrink all to fit (e.g., tall → medium)
**Pro:** Guarantees viewport fit
**Con:** May make charts unreadable, poor UX
**Verdict:** ⚠️ Possible fallback if widget dropping isn't acceptable

### **Alternative 4: Pagination (Multi-Page Dashboard)**
**Idea:** Split 15 widgets across 2 pages, user clicks "Next"
**Pro:** Fits all widgets, no dropping
**Con:** Breaks single-viewport requirement, adds navigation complexity
**Verdict:** ❌ Over-engineered for this use case

**Chosen Approach:** Fix #1 (limit widgets) + Fix #2 (enforce rows) + Fix #4 (smart packing)
**Rationale:** Meets user requirement (100vh fit), proven to work (V2/V3 already do this), minimal code changes

---

## Conclusion

V5's failure (3.1/10) is **not a fundamental flaw in the AI approach**, but rather **missing constraints** that manual generation (V2/V3/V4) had implicitly:

1. ✅ Intent parsing works (95% accuracy)
2. ✅ Widget selection works (types are correct)
3. ✅ Data collection works (queries return correct results)
4. ❌ **Widget count unconstrained** → too many widgets
5. ❌ **Grid packing unconstrained** → viewport overflow
6. ❌ **Schema normalization missing** → data errors

**Key Insight:** V2/V3 succeed because **you manually enforced viewport fit**. V5 needs those same constraints **programmatically encoded**.

**Recommendation:** Implement Phase 1 (8 hours) first. This alone will bring V5 from 3.1/10 to ~8.5/10, making it production-viable. Phase 2 and 3 are polish.

**Next Steps:**
1. Review this plan with team
2. Create GitHub issues for each fix (prioritized 1-6)
3. Start with Fix #1 (widget count) — fastest win
4. Run incremental testing after each fix
5. Deploy V5-fixed when composite score ≥ 9.5/10
