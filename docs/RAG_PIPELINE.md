# Command Center RAG Pipeline v2 — How It Works

## Overview

User speaks a question → 8 stages process it → Dashboard appears with widgets + voice answer.

```
Voice Input → Transcription → Intent Parse → Widget Select → Data Collect → Fixture Pick → Layout Pack → Voice Response
```

---

## Stage 1: Intent Parsing

**Model:** 8B (llama3.1:8b) | **Fallback:** Regex

Takes the raw transcript and classifies it:

| Field | Example |
|-------|---------|
| type | `query`, `action_reminder`, `conversation`, `out_of_scope` |
| domains | `["industrial"]`, `["supply", "people"]` |
| entities | `{devices: ["Transformer 1"], numbers: ["24"], time: ["today"]}` |
| primary_characteristic | `comparison`, `trend`, `energy`, `maintenance`, etc. |
| secondary_characteristics | `["alerts", "trend"]` |
| confidence | `0.92` |

The **primary_characteristic** is the most important output — it drives what kind of dashboard gets built. There are 18 possible values: comparison, trend, distribution, maintenance, shift, work_orders, energy, health_status, flow_sankey, cumulative, multi_source, power_quality, hvac, ups_dg, top_consumers, alerts, people, supply_chain.

---

## Stage 2: Short-Circuit

Before building a dashboard, the orchestrator checks:

- **out_of_scope** → "I can only help with industrial operations" (no dashboard)
- **conversation/greeting** → "Hello! How can I help?" (no dashboard)
- **action_*** → Delegates to action handlers (set reminder, send message, etc.)
- **query** → Continues to build dashboard

---

## Stage 3: Widget Selection

**Model:** 8B or 70B (set `WIDGET_SELECT_QUALITY=1` for 70B) | **Fallback:** Rule-based

The LLM sees the **full widget catalog** (19 scenarios with descriptions) plus the parsed intent, and outputs a JSON widget plan:

```
User: "Show me energy consumption trends"
  → hero: trend (primary answer)
  → expanded: trend-multi-line (correlation)
  → expanded: distribution (breakdown)
  → compact: kpi (quick metric)
  → expanded: category-bar (ranking)
  → expanded: flow-sankey (energy flow)
```

**What the LLM decides:**
- Which scenarios to use (from 19 available)
- What size each should be (hero/expanded/normal/compact)
- A relevance score (0-1) for ordering
- A "why" description shown to the user on each card
- A data_request telling Stage 4 what to fetch

**Constraints enforced after LLM output:**
- Max 10 widgets, max 18 height units
- Max 4 KPIs, max 2 of same scenario
- Banned: helpview, pulseview
- Only 1 hero (extras demoted to expanded)

**Domain affinity hints in the prompt:**
- Energy queries → flow-sankey, distribution, trend, trends-cumulative
- Maintenance → timeline, eventlogstream, alerts, category-bar
- Comparison → comparison, trend-multi-line, matrix-heatmap
- Health/status → matrix-heatmap, edgedevicepanel, kpi, alerts

---

## Stage 4: Data Collection (Parallel RAG)

**5 parallel workers** fetch data for each widget simultaneously.

Each widget has a **schema** that defines its RAG strategy. The strategy determines how to query ChromaDB:

| Strategy | Used By | What It Fetches |
|----------|---------|-----------------|
| `single_metric` | kpi | One number + unit + state from equipment collection |
| `alert_query` | alerts | Active alerts with severity, source, evidence |
| `multi_entity_metric` | comparison | Two entities' values + delta calculation |
| `time_series` | trend, trends-cumulative | SQL energy readings, or synthetic 48-point 24h data |
| `multi_time_series` | trend-multi-line | Multiple entities' time series overlaid |
| `aggregation` | distribution, composition, category-bar | Group-by equipment type, sum/avg values |
| `events_in_range` | timeline, eventlogstream | Chronological events from maintenance/shifts/work orders |
| `single_entity_deep` | edgedevicepanel | Full device: readings + alerts + maintenance history |
| `cross_tabulation` | matrix-heatmap | Equipment x Parameter matrix with health scores |
| `flow_analysis` | flow-sankey | Source → destination nodes + link values |
| `people_query` | peopleview, peoplehexgrid | Shift roster, supervisor info |
| `supply_query` | supplychainglobe | Vendor locations, shipment routes |

**Vector Store:** ChromaDB with `BAAI/bge-base-en-v1.5` embeddings (768-dim, cosine distance)

**6 Collections indexed:**
1. `industrial_equipment` — 439 pieces of equipment (transformers, pumps, chillers, etc.)
2. `industrial_alerts` — 675 alerts with severity and evidence
3. `maintenance_records` — 3963 maintenance logs
4. `operational_documents` — SOPs, inspection reports
5. `shift_logs` — shift handover records
6. `work_orders` — task assignments

---

## Stage 5: Fixture Selection

**Logic:** Rule-based matching on data context + diversity scoring

Each scenario has multiple **visual variants** (80 total across 19 scenarios). The fixture selector picks which variant to render based on the data:

**Example — KPI (9 variants):**

```
Data has state="critical"  → kpi_alert-critical-state (red alert card)
Data has state="warning"   → kpi_alert-warning-state (amber warning)
Data has state="offline"   → kpi_status-offline (grey offline)
Label has "voltage/power"  → kpi_live-high-contrast (electrical reading)
Label has "status/mode"    → kpi_status-badge (status badge)
Label has "load/usage"     → kpi_lifecycle-dark-mode-gauge (gauge dial)
Label has "daily/total"    → kpi_accumulated-daily-total (counter)
Has progress key + % unit  → kpi_lifecycle-progress-bar (progress bar)
Fallback                   → kpi_live-standard (default number)
```

**Context matching:** Rules check both the `label` field (from RAG data) AND `_query_context` (from the user's question + LLM widget plan). This prevents all widgets from falling to the default fixture.

**Diversity scoring:** If a fixture was already used in this dashboard, it gets a -5 penalty. Unused fixtures get a +2 bonus. This forces visual variety.

---

## Stage 6: Layout Packing

**Logic:** Deterministic row-based algorithm

The 12-column CSS grid can have gaps when widget sizes don't sum to 12. The packer fixes this:

```
Before packing:
  Row 1: hero(12)                          = 12/12 OK
  Row 2: expanded(6) + normal(4)           = 10/12 GAP=2
  Row 3: compact(3) + compact(3)           =  6/12 GAP=6

After packing (upsize smallest to fill):
  Row 1: hero(12)                          = 12/12 OK
  Row 2: expanded(6) + expanded(6)         = 12/12 OK  (normal→expanded)
  Row 3: normal(4) + normal(4) + normal(4) = 12/12 OK  (compact→normal x3)
```

**Rules:**
- Heroes always take full row (12 cols)
- Non-hero widgets pack left-to-right until row full
- Upsize smallest widget to fill remaining columns
- Never downsize
- Size-height coherence: x-tall widgets get at least expanded; tall chart types get expanded

---

## Stage 7: Voice Response

**Model:** 70B (llama3.3)

Generates a 2-3 sentence spoken response grounded in:
- The RAG data collected
- The widgets selected
- The user's original question

```
"Our Energy Consumption Trends dashboard shows the current total at 2977 kW,
with peak usage between 10am and 2pm. Chillers are the biggest consumers,
with Chiller 8 at 383 kW."
```

---

## Stage 8: Final Output

```json
{
  "voice_response": "spoken text for TTS",
  "filler_text": "Fetching the latest metrics.",
  "layout_json": {
    "heading": "Energy Consumption Trends",
    "widgets": [
      {
        "scenario": "trend",
        "size": "hero",
        "fixture": "trend_live-area",
        "heightHint": "tall",
        "description": "Shows energy consumption over 24 hours",
        "relevance": 1.0,
        "data_override": { "demoData": { "timeSeries": [...] } }
      }
    ]
  },
  "processing_time_ms": 41000
}
```

The frontend (Layer 3 Blob) receives `layout_json`, resolves each widget to a React component, merges `data_override` with fixture defaults, and renders the dashboard in a 12-column scrollable grid.

---

## Available Widget Scenarios (19 total)

| Scenario | Height | Sizes | Fixture Variants |
|----------|--------|-------|-----------------|
| kpi | 1 (short) | compact, normal | 9 |
| alerts | 2 (medium) | normal, expanded | 5 |
| comparison | 2 (medium) | expanded, hero | 6 |
| trend | 3 (tall) | expanded, hero | 6 |
| trend-multi-line | 3 (tall) | expanded, hero | 6 |
| trends-cumulative | 3 (tall) | expanded, hero | 6 |
| distribution | 3 (tall) | normal, expanded | 6 |
| composition | 3 (tall) | expanded, hero | 5 |
| category-bar | 3 (tall) | expanded, hero | 5 |
| timeline | 3 (tall) | expanded, hero | 5 |
| peopleview | 3 (tall) | expanded, hero | 1 |
| peoplehexgrid | 3 (tall) | expanded, hero | 1 |
| peoplenetwork | 3 (tall) | expanded, hero | 1 |
| supplychainglobe | 3 (tall) | hero | 1 |
| flow-sankey | 4 (x-tall) | hero | 5 |
| matrix-heatmap | 4 (x-tall) | hero | 5 |
| eventlogstream | 4 (x-tall) | expanded, hero | 5 |
| edgedevicepanel | 4 (x-tall) | hero | 1 |
| chatstream | 4 (x-tall) | expanded | 1 |

**Total: 19 scenarios, 80 fixture variants**

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PIPELINE_V2` | `1` | Enable v2 LLM pipeline (vs v1 regex) |
| `WIDGET_SELECT_QUALITY` | `0` | Use 70B for widget selection (`1` = yes) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL_FAST` | `llama3.1:8b` | Fast model (intent + widget select) |
| `OLLAMA_MODEL_QUALITY` | `llama3.3` | Quality model (voice response) |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model for vector search |

---

## Decision Flow Summary

```
User says "Compare transformer 1 and 2"
  │
  ├─ Stage 1: Intent → type=query, primary=comparison, entities={devices:[T1,T2]}
  ├─ Stage 2: query → continue to dashboard
  ├─ Stage 3: LLM picks → comparison(hero), trend-multi-line, distribution, kpi x2, timeline
  ├─ Stage 4: RAG fetches → T1 health=94%, T2 health=87%, delta=-7%, alerts, maintenance
  ├─ Stage 5: Fixtures → delta_bar_deviation, hvac-performance, energy-donut, kpi variants
  ├─ Stage 6: Pack → 3 rows, all 12/12, zero gaps
  ├─ Stage 7: Voice → "Transformer 1 is at 94% health vs T2 at 87%, a 7% difference..."
  └─ Stage 8: Output → layout_json + voice_response → Frontend renders dashboard
```
