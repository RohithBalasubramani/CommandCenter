# RAG & Widget Selection Redesign Proposal

**Date**: 2026-02-08
**Scope**: Retrieval intelligence and widget selection judgment
**Status**: Engineering specification — ready for implementation

---

## Executive Diagnosis

After reading every line of the retrieval and widget selection pipeline, the system has strong engineering foundations: source verification gates, traversal enforcement, provenance stamping, RL reranking, schema-driven data collection. What it lacks is **reasoning depth** between these stages.

The core problem is architectural: the pipeline is a **serial chain of isolated decisions** where each stage has minimal awareness of what the other stages know, need, or failed to find. The intent parser classifies a query and forgets. The prefetcher scans what exists and reports. The widget selector picks templates from a catalog. The data collector fills templates. Each stage is locally correct but globally naive.

This creates the characteristic failure modes the problem statement describes: technically relevant retrieval that misses the point, widget selections that reflect surface keywords instead of operational judgment, and inconsistent quality across semantically similar prompts.

---

## Part 1: How Retrieval Should Actually Work

### 1.1 Current State

Retrieval currently operates in three disconnected modes:

1. **ChromaDB vector search** (`rag_pipeline.py:225-257`) — cosine similarity over equipment metadata documents. Hybrid BM25+vector fusion via RRF with `alpha=0.7` favoring vectors. Documents are flat text strings like `"Transformer TRF-001 | Type: Transformer | Location: Substation A | Status: running | Health: 87%"`.

2. **PostgreSQL timeseries** (`data_collector.py`) — direct SQL queries against 357 equipment tables, 564M rows. Entity resolution via keyword matching (`ENTITY_TABLE_PREFIX_MAP`), metric resolution via alias lookup (`EQUIPMENT_METRIC_MAP`).

3. **Data prefetcher** (`data_prefetcher.py`) — parallel ChromaDB + PG scan producing ~300-500 token text summaries injected into the widget selection prompt.

**What's missing:**

- **No query reformulation.** The raw transcript goes directly into ChromaDB. "Why is transformer 1 running hot?" searches for that literal phrase in equipment metadata. The metadata doesn't contain the word "hot" — it contains `oil_temperature_top_c: 78.3`. The semantic gap between natural language operational questions and structured industrial data is not bridged.

- **No temporal awareness.** The prefetcher grabs `last_24h_points` but doesn't reason about whether 24 hours is the right window. "What happened to pump 3 during the night shift?" needs a specific 8-hour window. "Has transformer 1's load been increasing?" needs a multi-day trend with slope detection. The retrieval system treats time as a fixed parameter, not a dimension of relevance.

- **No cross-entity reasoning.** If an operator asks "Why is chiller 2 underperforming?", good retrieval would also pull: (a) chiller 2's maintenance history, (b) the condenser water loop shared with cooling tower 4, (c) alerts on the chilled water supply temperature, (d) whether other chillers in the same plant are showing similar patterns. Current retrieval searches each collection independently with the same query string.

- **No data sufficiency assessment.** The prefetcher reports what exists but never reports what's missing. If the operator asks about vibration on a transformer and the transformer table has no vibration column, this isn't surfaced until data collection fails silently and returns demo data.

- **No anomaly-aware retrieval.** Retrieval returns the same data regardless of whether values are normal or alarming. An experienced engineer would instinctively focus on the anomalous readings.

### 1.2 Redesign: Situational Retrieval Engine

Replace the current flat search with a three-phase retrieval process:

#### Phase A: Query Decomposition

Before any search executes, decompose the transcript into **retrieval sub-problems**:

```python
@dataclass
class RetrievalPlan:
    """What data do we actually need to answer this question?"""
    primary_question: str          # "What is the current power output of TRF-1?"
    supporting_questions: list[str] # ["Are there active alerts on TRF-1?", "What is TRF-1's normal operating range?"]
    temporal_scope: TemporalScope  # {start, end, granularity, rationale}
    entity_graph: list[EntityLink] # [{source: "TRF-1", relation: "feeds", target: "LT-PCC-001"}]
    required_metrics: list[str]    # ["active_power_kw", "load_percent"]
    nice_to_have_metrics: list[str] # ["oil_temperature_top_c", "power_factor"]
    causal_hypothesis: str | None  # "Operator may be investigating overload risk"
```

This is an LLM call, but a cheap one — it runs on the 8B model with a structured prompt that maps operator language to data needs. The key insight is that this call produces a **retrieval plan**, not a search query. The plan knows what data types are needed, what temporal scope matters, and what related entities might be relevant.

Implementation approach:
- Add a `QueryDecomposer` class to `layer2/` that takes a `ParsedIntent` and produces a `RetrievalPlan`
- Use the existing 8B LLM with `format: "json"` for structured output
- The prompt includes the equipment type's available metrics (from `EQUIPMENT_METRIC_MAP`) so the decomposer knows what's queryable
- Budget: 200-400ms (runs in parallel with source resolution)

#### Phase B: Targeted Parallel Retrieval

Execute retrieval sub-problems in parallel, each with an appropriate strategy:

| Sub-problem type | Strategy | Source |
|---|---|---|
| Current metric value | Direct PG query, latest row | `data_collector.resolve_entity_to_table()` |
| Time-series trend | PG query with temporal scope from Phase A | PG timeseries tables |
| Historical alerts | ChromaDB `industrial_alerts` + PG `industrial_alert` filtered by entity + time | Hybrid |
| Maintenance context | ChromaDB `maintenance_records` filtered by entity | Vector search |
| Related equipment | Entity graph traversal from Phase A | PG schema join or registry |
| Operating norms | ChromaDB `operational_documents` (SOPs, specs) | Vector search |

The critical change from current behavior: **each sub-problem gets a tailored query**, not the same raw transcript broadcast to every collection. The alert search for "Why is chiller 2 underperforming?" becomes `"chiller 2 temperature pressure alarm warning"` not `"Why is chiller 2 underperforming?"`.

Implementation approach:
- Extend `SchemaDataCollector` with a `collect_for_plan(retrieval_plan)` method
- Each sub-problem maps to a concrete function: `_fetch_current_value()`, `_fetch_timeseries()`, `_fetch_alerts_for_entity()`, `_fetch_maintenance_history()`, `_fetch_related_entities()`
- All execute in the existing `ThreadPoolExecutor(max_workers=8)`
- Budget: 500-1500ms total (parallel execution)

#### Phase C: Retrieval Quality Assessment

After results return, assess data sufficiency before proceeding to widget selection:

```python
@dataclass
class RetrievalAssessment:
    """How good is the data we retrieved?"""
    completeness: float           # 0-1: fraction of required metrics found
    freshness: DataFreshness      # {newest_timestamp, staleness_seconds, is_stale}
    anomalies_detected: list[Anomaly]  # [{metric, value, threshold, severity}]
    gaps: list[str]               # ["No vibration data available for TRF-1", "Maintenance records older than 6 months"]
    confidence: float             # 0-1: overall confidence in retrieved data
    warnings: list[str]           # Surfaced to operator via voice response
```

This is **not** an LLM call — it's deterministic computation over retrieved results:
- Completeness: count of `required_metrics` found vs. planned
- Freshness: compare latest timestamp against current time; flag if > 1 hour stale for live metrics
- Anomalies: compare values against known thresholds from `operational_documents` or equipment specs; use simple z-score over recent window
- Gaps: enumerate planned sub-problems that returned empty results

The assessment feeds into both widget selection (so it knows what data is actually available) and the voice response (so the operator knows when data is partial or stale).

Implementation approach:
- Add `RetrievalAssessor` class that takes the `RetrievalPlan` and retrieval results
- Threshold data comes from alert thresholds already in the `industrial_alert` table
- Freshness thresholds are equipment-type-specific (real-time sensors: 5 min; energy meters: 15 min; maintenance records: 30 days)
- The assessment is passed to `WidgetSelector.select()` as a structured object, replacing the current flat `data_summary` string

### 1.3 Specific Failure Modes This Eliminates

| Current failure | How it's fixed |
|---|---|
| "Show chiller 2 temperature" retrieves chiller metadata instead of temperature timeseries | Query decomposition identifies `required_metrics: ["chw_supply_temp_c"]` and routes to PG timeseries, not ChromaDB |
| "What happened last night?" retrieves nothing useful because no temporal context | Temporal scope resolver maps "last night" to a specific 8-hour window based on shift schedule |
| Stale data served without warning | Freshness check flags timestamps > threshold; voice response includes "Note: data is from 3 hours ago" |
| Missing metric served as demo data | Gap detection reports "No vibration data available for this transformer" before widget selection, preventing a vibration widget from being selected |
| Operator asks about root cause, gets surface-level reading | Causal hypothesis in retrieval plan triggers related-entity search and maintenance history |

---

## Part 2: How Widget Selection Should Actually Work

### 2.1 Current State

Widget selection is a single LLM call (`widget_selector.py:362-436`) where the 8B model receives:
- A widget catalog (19 active scenarios with sizes)
- The parsed intent (type, domains, entities, characteristics)
- A flat text data summary from the prefetcher
- Hard-coded domain affinity rules in the prompt

The LLM returns a JSON array of widgets. Post-processing validates scenarios, enforces height budgets, caps KPIs, and applies RL reranking.

**What's missing:**

- **No dashboard coherence model.** The LLM picks widgets independently — there's no explicit model of how widgets relate to each other or form a narrative. The FAST_SELECT_PROMPT has domain affinity hints, but these are keyword-level associations, not reasoning about what information **should appear together** for operational decision-making.

- **No awareness of data quality per widget.** The widget selector doesn't know whether the data for a particular widget will be real timeseries, stale ChromaDB metadata, or demo data. It selects widgets optimistically and the data collector quietly fills gaps with synthetic data.

- **No redundancy detection.** Two KPIs showing "active_power_kw" for the same entity can be selected if the LLM assigns them slightly different `data_request.metric` strings ("power" vs "active power"). The same metric appears twice with different labels.

- **No contradiction awareness.** If alerts say a device is critical but the KPI shows normal values (because the KPI pulled the latest reading after the alert was resolved), the dashboard contradicts itself. There's no mechanism to detect or resolve this.

- **The RL scorer adjusts relevance but not composition.** The scorer (`lora_scorer.py`) learns per-scenario adjustments for a given transcript. It can learn that "alerts" should be ranked higher for health queries. But it cannot learn that "for comparison queries with 2 entities, the ideal layout is: comparison(hero) + trend-multi-line(expanded) + kpi(compact) + kpi(compact)". The RL signal is per-widget, not per-dashboard.

- **Static sizing rules.** Widget sizes are determined by the LLM based on prompt rules, then post-processed with hard constraints. There's no awareness of what size makes sense given the **amount of data available**. A trend widget with 5 data points should be compact, not hero. A comparison with 6 entities should be hero, not normal.

### 2.2 Redesign: Widget Selection as Dashboard Planning

Replace the single LLM call with a **two-stage decision process**: narrative planning followed by widget allocation.

#### Stage 1: Dashboard Narrative

Before selecting any widgets, determine what the dashboard should **communicate**:

```python
@dataclass
class DashboardNarrative:
    """What story should this dashboard tell?"""
    primary_answer: str           # "TRF-1 is currently at 82% load, above the 75% warning threshold"
    supporting_context: list[str] # ["Load has been increasing over the past 6 hours", "2 active alerts on TRF-1"]
    risk_assessment: str | None   # "TRF-1 may be approaching overload. Monitor closely."
    data_gaps: list[str]          # ["Oil temperature data is 2 hours stale"]
    operator_next_action: str     # "Consider checking oil temperature and cooling system"
    information_slots: list[InformationSlot]  # Typed slots that need to be filled
```

```python
@dataclass
class InformationSlot:
    """A semantic slot in the dashboard that needs filling."""
    role: str                    # "primary_answer" | "trend_context" | "alert_surface" | "comparison_axis" | "detail_panel"
    priority: float              # 0-1
    data_available: bool         # From retrieval assessment
    data_quality: str            # "real_time" | "recent" | "stale" | "missing"
    suggested_scenario: str | None  # Best widget type for this slot, or None if uncertain
    required_entities: list[str]
    required_metrics: list[str]
```

This is a structured LLM call (8B model) that takes the retrieval assessment as input. The key difference from the current approach: **the LLM reasons about what information is important before choosing any widget types**. The `InformationSlot` abstraction separates "what needs to be shown" from "what widget shows it".

Implementation approach:
- Add `DashboardPlanner` class
- Input: `ParsedIntent`, `RetrievalAssessment`, `RetrievalPlan`, `user_context`
- Output: `DashboardNarrative`
- The prompt gives the LLM the retrieval assessment (what data was found, what's missing, what's anomalous) and asks it to plan the dashboard story
- Budget: 300-600ms

#### Stage 2: Widget Allocation

Given the narrative and its information slots, allocate widgets:

```python
def allocate_widgets(narrative: DashboardNarrative, retrieval_results: dict) -> WidgetPlan:
    """Deterministic widget allocation from narrative slots."""
    widgets = []

    for slot in narrative.information_slots:
        if not slot.data_available and slot.data_quality == "missing":
            # Don't allocate a widget for missing data
            continue

        scenario = resolve_scenario(slot)
        size = resolve_size(slot, retrieval_results)

        widgets.append(WidgetPlanItem(
            scenario=scenario,
            size=size,
            relevance=slot.priority,
            why=slot_to_description(slot, narrative),
            data_request=slot_to_data_request(slot),
        ))

    # Coherence pass: detect and resolve conflicts
    widgets = resolve_redundancies(widgets)
    widgets = resolve_contradictions(widgets, retrieval_results)
    widgets = enforce_budget(widgets)

    return WidgetPlan(heading=narrative_to_heading(narrative), widgets=widgets)
```

This is **mostly deterministic** — the LLM's creative work happened in Stage 1. Widget allocation maps information slots to widget types using a rule table (extended from the current domain affinity rules, but keyed on `slot.role` rather than keywords):

| Slot Role | Primary Scenario | Fallback | Preferred Size |
|---|---|---|---|
| `primary_answer` (single metric) | `kpi` | — | hero if only answer, compact if supporting |
| `primary_answer` (comparison) | `comparison` | `category-bar` | hero |
| `primary_answer` (time-series) | `trend` | `trend-multi-line` | hero |
| `trend_context` | `trend` or `trend-multi-line` | `trends-cumulative` | expanded |
| `alert_surface` | `alerts` | — | normal (or expanded if >5 alerts) |
| `breakdown` | `distribution` | `composition` | normal |
| `detail_panel` | `edgedevicepanel` | — | hero or expanded |
| `ranking` | `category-bar` | `matrix-heatmap` | hero or expanded |
| `event_history` | `timeline` | `eventlogstream` | expanded |
| `flow_visualization` | `flow-sankey` | — | hero |

Size resolution is **data-aware**:
- `trend` with < 10 data points → compact (not worth hero)
- `comparison` with > 4 entities → hero (needs space)
- `alerts` with 0 alerts → omit entirely (no information value)
- `kpi` with stale data → mark as stale, demote priority

#### Coherence Rules (deterministic post-processing)

1. **Redundancy elimination**: If two widgets would show the same `(entity, metric, time_range)` triple, keep the higher-priority one and drop the other. This is computed by hashing the resolved data request, not by comparing LLM-generated strings.

2. **Contradiction detection**: If an alert widget shows a critical alert on entity X but a KPI widget shows entity X as "normal", inject a `_conflict` flag. The voice response should acknowledge the discrepancy ("TRF-1 currently reads normal but had a critical alert 20 minutes ago").

3. **Information density guard**: If all widgets would show variants of the same metric (e.g., 4 KPIs all showing power for different entities), replace 2 of them with a single comparison or category-bar widget. This implements the principle that **a dashboard should synthesize, not enumerate**.

4. **Empty-state handling**: If a widget would render with no data (0 alerts, no timeseries, no maintenance records), remove it and redistribute its screen budget. Never show an empty widget — it destroys operator trust.

### 2.3 RL Scorer Upgrade

The current low-rank scorer (`lora_scorer.py`) adjusts per-scenario relevance. Extend it to score **dashboard compositions**:

```python
class CompositionScorer(nn.Module):
    """Score entire dashboard compositions, not just individual widgets."""

    def __init__(self, embed_dim=768, n_scenarios=19, composition_dim=64):
        # Encode the full widget plan as a fixed-size vector
        self.scenario_embed = nn.Embedding(n_scenarios, 16)
        self.composition_encoder = nn.Sequential(
            nn.Linear(n_scenarios * 16 + embed_dim, composition_dim),
            nn.ReLU(),
            nn.Linear(composition_dim, 1),
            nn.Tanh(),
        )

    def forward(self, intent_embedding, scenario_ids):
        # scenario_ids: tensor of shape (n_widgets,) — the selected scenarios
        # Create a bag-of-scenarios representation
        scenario_embeds = self.scenario_embed(scenario_ids)
        composition_vec = scenario_embeds.mean(dim=0)  # Pool over widgets
        combined = torch.cat([intent_embedding, composition_vec])
        return self.composition_encoder(combined)
```

This scores the **full set of selected widgets** as a composition against the intent. The feedback signal is the same (user engagement, follow-up patterns, explicit ratings), but the learning target is whether the dashboard as a whole was useful, not whether individual widgets were relevant.

Keep the existing per-widget scorer as Tier 1; add the composition scorer as Tier 1b. Both are tiny models (< 10K params) that train on CPU in milliseconds.

---

## Part 3: How Retrieval and Widget Selection Interact

### 3.1 Current State

The interaction is one-directional and text-mediated:
1. Prefetcher produces a text summary → injected into widget selector prompt
2. Widget selector outputs `data_request` per widget → data collector queries PG/ChromaDB

The widget selector cannot influence what gets retrieved. The data collector cannot influence which widgets are selected. If the data collector finds nothing for a widget, it silently produces demo data.

### 3.2 Redesign: Bidirectional Data-Widget Contract

```
                    ┌────────────────────────┐
                    │   Query Decomposer     │
                    │  (RetrievalPlan)       │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │  Targeted Retrieval    │
                    │  (parallel PG+Chroma)  │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │  Retrieval Assessment  │
                    │  (completeness, gaps)  │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │  Dashboard Planner     │
                    │  (narrative + slots)   │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │  Widget Allocation     │
                    │  (data-aware sizing)   │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │  Targeted Data Fill    │
                    │  (per-widget queries)  │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │  Reconciliation Loop   │◄─── DROP widgets with no data
                    │  (validate + coherence)│     DEMOTE widgets with partial data
                    └──────────┬─────────────┘     PROMOTE widgets with anomalies
                               │
                    ┌──────────▼─────────────┐
                    │  Final Layout          │
                    └────────────────────────┘
```

**Key interaction points:**

1. **Retrieval assessment informs slot creation.** If retrieval found anomalous values, the planner creates an `alert_surface` slot even if the operator didn't ask about alerts. If retrieval found no timeseries data, the planner doesn't create a `trend_context` slot.

2. **Widget allocation checks data availability per slot.** A `trend` widget is only allocated if the retrieval assessment confirms timeseries data exists with sufficient density (> 10 points in the requested window).

3. **Post-fill reconciliation can drop widgets.** After the data collector runs, any widget where `data_override` contains only demo/synthetic data and the retrieval assessment flagged that metric as `missing` gets dropped from the layout. The height budget is redistributed to remaining widgets.

4. **Anomaly promotion.** If the data fill reveals that a metric value exceeds its alert threshold (detected by comparing against `industrial_alert` thresholds), that widget gets its relevance boosted and potentially promoted in position — even if the operator didn't ask about it. This is the "experienced engineer" behavior: surfacing what matters, not just what was asked.

### 3.3 Latency Budget

The redesign adds one LLM call (query decomposition) and one deterministic computation (retrieval assessment + widget allocation). Net impact on the 8-second total pipeline budget:

| Stage | Current | Redesigned | Delta |
|---|---|---|---|
| Intent Parse | 500ms | 500ms | 0 |
| Source Resolution + Traversal | 200ms | 200ms | 0 |
| Query Decomposition | — | 400ms | +400ms |
| Data Prefetch | 500ms | 800ms (more targeted) | +300ms |
| Retrieval Assessment | — | 50ms (deterministic) | +50ms |
| Dashboard Planning (narrative) | — | 500ms | +500ms |
| Widget Selection (LLM) | 2000ms | 0 (replaced by deterministic allocation) | -2000ms |
| Widget Allocation (deterministic) | — | 50ms | +50ms |
| Data Collection | 1500ms | 1200ms (more targeted, less wasted) | -300ms |
| Reconciliation + Coherence | 100ms | 150ms | +50ms |
| Fixture Selection | 800ms | 800ms | 0 |
| Voice Response | 2000ms (concurrent) | 2000ms (concurrent) | 0 |
| **Total** | **~7600ms** | **~6650ms** | **-950ms** |

The net savings come from eliminating the widget selection LLM call (currently ~2000ms) and replacing it with a faster dashboard planning call (~500ms) plus deterministic allocation (~50ms). More targeted data collection also saves time by not fetching data for widgets that will be dropped.

Query decomposition (400ms) and dashboard planning (500ms) can be **pipelined**: query decomposition starts after intent parsing; retrieval starts after decomposition; assessment + planning run after retrieval returns. The dashboard planning LLM call receives the retrieval assessment directly — no intermediate serialization to text.

---

## Part 4: Failure Awareness

### 4.1 Current Failure Handling

The system has strong hard-stop gates:
- Source resolution gate: refuses if no authoritative source found
- Traversal enforcement: blocks if zero traversal actions executed
- Provenance stamping: rejects data without `_data_source`

But these are **binary gates** — they pass or block. There's no middle ground for "I can partially answer this" or "I have data but it's contradictory."

### 4.2 Redesign: Graduated Confidence System

Introduce a **confidence envelope** that flows through the entire pipeline:

```python
@dataclass
class ConfidenceEnvelope:
    """Graduated confidence that flows through every pipeline stage."""

    # Per-stage confidence (0-1)
    intent_confidence: float       # How sure are we about what the user wants?
    retrieval_completeness: float  # Did we find everything we need?
    data_freshness: float          # How current is the data? (1.0 = real-time, 0.0 = stale)
    widget_fit: float              # How well do the widgets match the information needs?
    data_fill_quality: float       # How much of the widget data is real vs. synthetic?

    # Aggregate
    @property
    def overall(self) -> float:
        """Weighted harmonic mean — dominated by the weakest link."""
        weights = [0.15, 0.30, 0.20, 0.15, 0.20]
        values = [self.intent_confidence, self.retrieval_completeness,
                  self.data_freshness, self.widget_fit, self.data_fill_quality]
        weighted_sum = sum(w / max(v, 0.01) for w, v in zip(weights, values))
        return len(weights) / weighted_sum  # Harmonic mean

    # Action thresholds
    @property
    def action(self) -> str:
        if self.overall >= 0.8:
            return "full_dashboard"       # Normal response
        elif self.overall >= 0.6:
            return "partial_with_caveats"  # Show dashboard but voice caveats
        elif self.overall >= 0.4:
            return "reduced_dashboard"     # Fewer widgets, explicit gaps
        else:
            return "ask_clarification"     # Don't guess — ask the operator
```

**How each stage contributes:**

- **Intent parse**: Current `confidence` field (already exists in `ParsedIntent`). Below 0.5 → trigger clarification.
- **Retrieval completeness**: `len(found_metrics) / len(required_metrics)` from retrieval plan. Below 0.3 → "I don't have enough data to answer this comprehensively."
- **Data freshness**: `min(freshness_scores)` across all retrieved metrics. If any critical metric is stale, overall freshness drops.
- **Widget fit**: `1.0 - (empty_widgets / total_widgets)` after data fill. If half the widgets would be empty, widget fit is 0.5.
- **Data fill quality**: `real_data_widgets / total_widgets`. Tracks how much is real PG timeseries vs. ChromaDB metadata vs. demo data.

### 4.3 What the System Does Instead of Guessing

| Confidence range | System behavior |
|---|---|
| **0.8-1.0** | Full dashboard with voice summary. No caveats needed. |
| **0.6-0.79** | Dashboard renders but voice response includes explicit caveats: "I'm showing you transformer 1's power trend, but oil temperature data is not available right now." Stale metrics get a visual indicator. |
| **0.4-0.59** | Reduced dashboard: only widgets with real data are shown. Voice says: "I have partial data for this query. Here's what I can show you. For [missing aspect], you may want to check [specific system/panel]." |
| **0.2-0.39** | Minimal response: 1-2 high-confidence widgets only. Voice says: "I don't have enough data to build a full dashboard. Here's what I do know: [specific fact]. Would you like me to look into [specific alternative]?" |
| **<0.2** | No dashboard. Voice asks a clarifying question or explains what data sources are needed. This replaces the current behavior of showing a full dashboard of demo data. |

### 4.4 Uncertainty Surfacing

Three channels for surfacing uncertainty:

1. **Voice response**: The 70B voice model receives the confidence envelope and is instructed to include caveats proportional to uncertainty. This already runs concurrently — no latency cost.

2. **Widget-level indicators**: Each widget already has an `isDemo` flag that shows an amber badge. Extend this with:
   - `isStale: boolean` — grey overlay with "Data from X hours ago"
   - `isPartial: boolean` — dashed border indicating incomplete data
   - `conflictFlag: string | null` — red indicator when this widget's data contradicts another widget

3. **Dashboard-level confidence bar**: A thin bar at the top of the BlobGrid showing overall confidence. Green (>0.8), yellow (0.6-0.8), orange (0.4-0.6), red (<0.4). Clicking it expands to show which stages contributed uncertainty. This gives operators a quick trust signal without overwhelming them.

---

## Part 5: Testability

### 5.1 Failure Classes This Redesign Should Eliminate

| # | Failure class | Current behavior | Expected behavior after redesign |
|---|---|---|---|
| F1 | **Empty widget syndrome** | Widget selected but data collection returns nothing → demo data shown | Retrieval assessment flags missing data → widget not allocated → no empty widgets |
| F2 | **Metric mismatch** | "Show chiller temperature" → widget shows power (LLM defaults to power in data_request) | Query decomposition explicitly resolves "temperature" → `chw_supply_temp_c` before widget selection |
| F3 | **Temporal blindness** | "What happened last night?" → 24-hour window | Temporal scope resolver → shift-aware time windows |
| F4 | **Redundant dashboard** | Two KPIs both showing power_kw for same entity | Redundancy elimination in widget allocation |
| F5 | **Contradictory dashboard** | Alert says critical + KPI says normal | Contradiction detection flags conflict in reconciliation |
| F6 | **Oversized empty widget** | Hero-sized trend with 3 data points | Data-aware sizing: < 10 points → compact |
| F7 | **Surface keyword matching** | "Why is X failing?" → shows status KPI instead of root cause analysis | Causal hypothesis in retrieval plan → maintenance history + related entities retrieved |
| F8 | **Stale data as current** | 3-hour-old reading shown without indicator | Freshness check → stale visual indicator |
| F9 | **Demo data confidence** | Full dashboard of synthetic data with no warning | Confidence envelope → reduced dashboard or explicit refusal |
| F10 | **Inconsistent similar prompts** | "Show TRF-1 status" vs "How is transformer 1 doing?" → different dashboards | Narrative planning normalizes intent → consistent slot allocation |

### 5.2 Test Strategy

#### Unit tests (fast, per-component)

```
test_query_decomposer/
  test_temperature_query_extracts_correct_metric.py
  test_comparison_query_identifies_all_entities.py
  test_temporal_scope_resolves_shift_names.py
  test_causal_query_includes_maintenance_context.py

test_retrieval_assessor/
  test_completeness_calculation.py
  test_freshness_detection_thresholds.py
  test_anomaly_detection_against_known_thresholds.py
  test_gap_identification.py

test_dashboard_planner/
  test_missing_data_suppresses_slots.py
  test_anomaly_promotes_alert_slot.py
  test_comparison_query_creates_comparison_slot.py

test_widget_allocator/
  test_redundancy_elimination.py
  test_contradiction_detection.py
  test_data_aware_sizing.py
  test_empty_state_removal.py
  test_budget_enforcement.py

test_confidence_envelope/
  test_low_retrieval_triggers_caveats.py
  test_very_low_confidence_blocks_dashboard.py
  test_stale_data_reduces_freshness.py
```

#### Integration tests (medium, end-to-end pipeline with mocked LLM)

For each of the 10 failure classes above, create a test that:
1. Inputs a specific transcript known to trigger the failure
2. Mocks the LLM to return a controlled response
3. Seeds PG/ChromaDB with specific data states (e.g., stale timestamps, missing columns)
4. Asserts the output dashboard does NOT exhibit the failure

Example test for F1 (empty widget syndrome):

```python
def test_no_empty_widgets_when_metric_unavailable(mock_llm, pg_with_no_vibration):
    """Requesting vibration for a transformer that has no vibration data
    should NOT produce a vibration trend widget with demo data."""
    response = orchestrator.process_transcript("Show me transformer 1 vibration")

    for widget in response.layout_json["widgets"]:
        # No widget should have demo/synthetic data for the primary metric
        data_override = widget.get("data_override", {})
        assert not data_override.get("_synthetic", False), \
            f"Widget '{widget['scenario']}' has synthetic data — should have been dropped"
```

#### Golden dataset tests (slow, require LLM, run nightly)

Maintain a dataset of 50-100 (transcript, expected_dashboard_properties) pairs:

```json
{
  "transcript": "Compare transformer 1 and transformer 2 power consumption",
  "expected": {
    "min_widgets": 3,
    "max_widgets": 7,
    "must_contain_scenarios": ["comparison"],
    "must_not_contain_scenarios": ["helpview", "pulseview"],
    "hero_must_be": ["comparison", "trend-multi-line"],
    "entities_in_data_requests": ["transformer 1", "transformer 2"],
    "metric_in_hero": "power",
    "no_redundant_metric_entity_pairs": true,
    "confidence_above": 0.6
  }
}
```

Run these nightly with the real LLM. Track pass rate over time. Any regression below 90% pass rate blocks deployment.

#### Consistency tests (detect F10)

For each semantic cluster of equivalent queries, verify that dashboard compositions are similar:

```python
equivalent_queries = [
    "Show me transformer 1 status",
    "How is transformer 1 doing?",
    "What's the status of TRF-1?",
    "Transformer one overview",
    "Check transformer 1",
]

dashboards = [orchestrator.process_transcript(q).layout_json for q in equivalent_queries]

# All should have the same set of scenario types (order may differ)
scenario_sets = [frozenset(w["scenario"] for w in d["widgets"]) for d in dashboards]
assert len(set(scenario_sets)) <= 2, \
    f"Equivalent queries produced {len(set(scenario_sets))} different dashboard compositions"
```

### 5.3 Regression Detection

1. **RL feedback correlation**: Track the `overall_confidence` from the confidence envelope against RL reward signals. If confidence is high (>0.8) but reward is low (operator dismissed widgets or asked follow-up clarifications), that's a regression in the narrative planning stage.

2. **Widget drop rate**: Track `widgets_planned / widgets_rendered` ratio. A healthy system should have > 0.85 (most planned widgets have data). If this drops below 0.7, retrieval quality has degraded.

3. **Demo data fraction**: Track `synthetic_widgets / total_widgets` across all responses. This should trend toward 0 as the system matures. Any sustained increase indicates a data availability regression.

4. **Consistency score**: For each pair of semantically similar queries (detected via embedding similarity > 0.9), compute the Jaccard similarity of their scenario sets. Track the average. A drop indicates the system is becoming less deterministic.

---

## Part 6: Implementation Priority

Ordered by impact-to-effort ratio:

### Phase 1: Immediate wins (1-2 weeks)

1. **Retrieval assessment + widget dropping** — Add the `RetrievalAssessor` class and post-fill reconciliation loop that drops widgets with no real data. This alone eliminates F1, F8, and F9.

2. **Data-aware widget sizing** — After data collection, resize widgets based on actual data density. Trend with < 10 points → compact. Alerts with 0 items → drop. This eliminates F6.

3. **Redundancy detection** — After widget allocation, hash each widget's `(entity, resolved_metric_column, time_range)` tuple and deduplicate. This eliminates F4.

### Phase 2: Core redesign (2-4 weeks)

4. **Query decomposer** — The LLM call that produces a `RetrievalPlan` with explicit metrics, temporal scope, and causal hypothesis. This eliminates F2, F3, and F7.

5. **Dashboard narrative planner** — Replace the current widget selection LLM call with narrative planning + deterministic allocation. This eliminates F5 and F10.

6. **Confidence envelope** — Thread confidence through the pipeline and expose it in voice responses and UI indicators.

### Phase 3: Learning improvements (2-3 weeks)

7. **Composition scorer** — Extend RL to score dashboard compositions, not just individual widgets.

8. **Temporal scope learning** — Learn optimal time windows from feedback (e.g., operators consistently expand 24h trends to 7d → learn to default to 7d for that query type).

9. **Consistency regularization** — Add a loss term to the composition scorer that penalizes divergent dashboard compositions for semantically similar queries.

---

## Appendix: Files to Modify

| File | Change |
|---|---|
| `backend/layer2/query_decomposer.py` | **NEW** — Query decomposition LLM call producing `RetrievalPlan` |
| `backend/layer2/retrieval_assessor.py` | **NEW** — Deterministic assessment of retrieval quality |
| `backend/layer2/dashboard_planner.py` | **NEW** — Narrative planning + slot allocation |
| `backend/layer2/confidence.py` | **NEW** — `ConfidenceEnvelope` dataclass and computation |
| `backend/layer2/orchestrator.py` | **MODIFY** — Wire new stages into `_process_transcript_v2` pipeline |
| `backend/layer2/data_prefetcher.py` | **MODIFY** — Accept `RetrievalPlan` for targeted prefetch |
| `backend/layer2/data_collector.py` | **MODIFY** — Accept `RetrievalPlan` for targeted collection; post-fill quality reporting |
| `backend/layer2/widget_selector.py` | **MODIFY** — Replace LLM selection with deterministic allocation from slots |
| `backend/rl/lora_scorer.py` | **MODIFY** — Add `CompositionScorer` alongside existing per-widget scorer |
| `backend/rl/config.py` | **MODIFY** — Add composition scorer hyperparameters |
| `frontend/src/components/layer3/WidgetSlot.tsx` | **MODIFY** — Add stale/partial/conflict visual indicators |
| `frontend/src/components/layer3/BlobGrid.tsx` | **MODIFY** — Add confidence bar at top |
| `frontend/src/types/index.ts` | **MODIFY** — Add `isStale`, `isPartial`, `conflictFlag` to widget instruction type |
