# Command Center — System Upgrade Specification v1.0

**Date**: 2026-02-08
**Status**: Design Complete — Ready for Implementation
**Scope**: 10 major upgrades across all 4 layers

---

## Table of Contents

1. [Persistent Semantic Focus Graph](#1-persistent-semantic-focus-graph)
2. [Cross-Widget Reasoning Engine](#2-cross-widget-reasoning-engine)
3. [Time-Travel + Comparison Focus](#3-time-travel--comparison-focus)
4. [Intent → Plan → Execute Orchestration](#4-intent--plan--execute-orchestration)
5. [Constraint-Aware Decision Making](#5-constraint-aware-decision-making)
6. [RL Upgrade: From Widgets → Decisions](#6-rl-upgrade-from-widgets--decisions)
7. [Memory Stratification](#7-memory-stratification)
8. [RAG → Causal Knowledge Engine](#8-rag--causal-knowledge-engine)
9. [Voice as Control Surface](#9-voice-as-control-surface)
10. [Explicit Failure-Mode UX](#10-explicit-failure-mode-ux)
11. [E2E Test Specifications](#11-e2e-test-specifications)

---

## 1. Persistent Semantic Focus Graph

### Problem

Current `interactiveCtx` is flat: `{key, scenario, label, equipment, metric}`. It cannot represent relationships between entities, track causal chains, or survive complex multi-turn investigations. When a user says "now compare that with pump 5", the system has no structured way to know "that" means the vibration anomaly on pump 4 that was discussed 3 turns ago, or that pump 4 and pump 5 share the same cooling loop.

### Data Structures

**Location**: `backend/layer2/focus_graph.py` (new file)

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import time
import hashlib


class NodeType(Enum):
    EQUIPMENT = "equipment"        # pump_004, transformer_001
    METRIC = "metric"              # vibration_de_mm_s, oil_temperature_top_c
    ANOMALY = "anomaly"            # threshold_breach, trend_deviation
    TIME_RANGE = "time_range"      # last_24h, 2026-02-07T14:00/2026-02-07T18:00
    TASK = "task"                   # work_order_1234, inspection_pump_004
    DOCUMENT = "document"          # maintenance_manual_pump, sop_vibration_analysis


class EdgeType(Enum):
    CAUSED_BY = "caused_by"             # anomaly → root cause
    CORRELATED_WITH = "correlated_with" # metric A moves with metric B
    DERIVED_FROM = "derived_from"       # calculated metric → source metrics
    PRECEDES = "precedes"               # event A happened before event B
    EXPLAINS = "explains"               # document/task explains anomaly
    MEASURED_BY = "measured_by"         # equipment → metric
    PART_OF = "part_of"                 # pump_004 → cooling_loop_A
    AFFECTS = "affects"                 # anomaly → downstream equipment
    COMPARED_WITH = "compared_with"     # pump_004 vs pump_005


@dataclass
class FocusNode:
    id: str                              # "equipment:pump_004"
    type: NodeType
    label: str                           # "Pump 4"
    properties: dict = field(default_factory=dict)
    confidence: float = 1.0              # 0.0-1.0, how certain we are this is relevant
    source: str = "user_query"           # user_query | rag_inferred | system_detected
    created_at: float = field(default_factory=time.time)
    last_referenced: float = field(default_factory=time.time)
    reference_count: int = 1


@dataclass
class FocusEdge:
    id: str                              # hash of source+target+type
    source: str                          # node id
    target: str                          # node id
    type: EdgeType
    confidence: float = 0.8
    evidence: str = ""                   # why this edge exists
    created_at: float = field(default_factory=time.time)


@dataclass
class SemanticFocusGraph:
    session_id: str
    nodes: dict[str, FocusNode] = field(default_factory=dict)
    edges: list[FocusEdge] = field(default_factory=list)
    root_node_id: Optional[str] = None   # primary focus equipment
    version: int = 0                      # incremented on every mutation
    created_at: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)

    def add_node(self, node: FocusNode) -> str:
        """Add node, return id. If node exists, increment reference_count."""
        if node.id in self.nodes:
            self.nodes[node.id].reference_count += 1
            self.nodes[node.id].last_referenced = time.time()
            return node.id
        self.nodes[node.id] = node
        if self.root_node_id is None and node.type == NodeType.EQUIPMENT:
            self.root_node_id = node.id
        self.version += 1
        self.last_modified = time.time()
        return node.id

    def add_edge(self, source: str, target: str, edge_type: EdgeType,
                 confidence: float = 0.8, evidence: str = "") -> str:
        """Add edge between existing nodes. Returns edge id."""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Both nodes must exist: {source}, {target}")
        edge_id = hashlib.sha256(f"{source}:{target}:{edge_type.value}".encode()).hexdigest()[:16]
        # Deduplicate
        for e in self.edges:
            if e.id == edge_id:
                e.confidence = max(e.confidence, confidence)
                return edge_id
        self.edges.append(FocusEdge(
            id=edge_id, source=source, target=target,
            type=edge_type, confidence=confidence, evidence=evidence,
        ))
        self.version += 1
        self.last_modified = time.time()
        return edge_id

    def get_neighbors(self, node_id: str, edge_type: EdgeType = None) -> list[FocusNode]:
        """Get all nodes connected to node_id, optionally filtered by edge type."""
        neighbor_ids = set()
        for edge in self.edges:
            if edge_type and edge.type != edge_type:
                continue
            if edge.source == node_id:
                neighbor_ids.add(edge.target)
            elif edge.target == node_id:
                neighbor_ids.add(edge.source)
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]

    def get_root_equipment(self) -> Optional[FocusNode]:
        """Return the primary equipment node."""
        if self.root_node_id and self.root_node_id in self.nodes:
            return self.nodes[self.root_node_id]
        return None

    def get_all_equipment(self) -> list[FocusNode]:
        """Return all equipment nodes, ordered by reference_count desc."""
        equip = [n for n in self.nodes.values() if n.type == NodeType.EQUIPMENT]
        return sorted(equip, key=lambda n: n.reference_count, reverse=True)

    def get_active_anomalies(self) -> list[FocusNode]:
        """Return anomaly nodes with confidence > 0.5."""
        return [n for n in self.nodes.values()
                if n.type == NodeType.ANOMALY and n.confidence > 0.5]

    def get_time_context(self) -> Optional[FocusNode]:
        """Return the most recently referenced time_range node."""
        times = [n for n in self.nodes.values() if n.type == NodeType.TIME_RANGE]
        if not times:
            return None
        return max(times, key=lambda n: n.last_referenced)

    def to_prompt_context(self, max_nodes: int = 15) -> str:
        """Serialize graph to text for LLM prompt injection."""
        lines = []
        # Top nodes by reference count
        ranked = sorted(self.nodes.values(), key=lambda n: n.reference_count, reverse=True)[:max_nodes]
        for node in ranked:
            lines.append(f"- [{node.type.value}] {node.label} (refs={node.reference_count}, conf={node.confidence:.2f})")
        if self.edges:
            lines.append("Relationships:")
            for edge in self.edges[:20]:
                src = self.nodes.get(edge.source)
                tgt = self.nodes.get(edge.target)
                if src and tgt:
                    lines.append(f"  {src.label} --{edge.type.value}--> {tgt.label} (conf={edge.confidence:.2f})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for JSON transport."""
        return {
            "session_id": self.session_id,
            "version": self.version,
            "root_node_id": self.root_node_id,
            "nodes": {nid: {
                "id": n.id, "type": n.type.value, "label": n.label,
                "properties": n.properties, "confidence": n.confidence,
                "source": n.source, "reference_count": n.reference_count,
            } for nid, n in self.nodes.items()},
            "edges": [{
                "id": e.id, "source": e.source, "target": e.target,
                "type": e.type.value, "confidence": e.confidence,
                "evidence": e.evidence,
            } for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticFocusGraph":
        """Deserialize from JSON."""
        graph = cls(session_id=data["session_id"])
        graph.version = data.get("version", 0)
        graph.root_node_id = data.get("root_node_id")
        for nid, nd in data.get("nodes", {}).items():
            graph.nodes[nid] = FocusNode(
                id=nd["id"], type=NodeType(nd["type"]), label=nd["label"],
                properties=nd.get("properties", {}),
                confidence=nd.get("confidence", 1.0),
                source=nd.get("source", "unknown"),
                reference_count=nd.get("reference_count", 1),
            )
        for ed in data.get("edges", []):
            graph.edges.append(FocusEdge(
                id=ed["id"], source=ed["source"], target=ed["target"],
                type=EdgeType(ed["type"]),
                confidence=ed.get("confidence", 0.8),
                evidence=ed.get("evidence", ""),
            ))
        return graph
```

**Graph Builder** — `backend/layer2/focus_graph_builder.py` (new file):

```python
class FocusGraphBuilder:
    """Builds and evolves the semantic focus graph from parsed intents and RAG results."""

    def __init__(self, graph: SemanticFocusGraph):
        self.graph = graph

    def ingest_intent(self, intent: ParsedIntent) -> list[str]:
        """
        Extract entities from ParsedIntent and add as nodes.
        Returns list of newly created node IDs.

        Rules:
        1. Each device in intent.entities["devices"] → NodeType.EQUIPMENT
        2. Detected primary_characteristic → infer metric nodes
        3. intent.entities["time"] → NodeType.TIME_RANGE
        4. If characteristic is "maintenance" → add TASK node
        5. If characteristic is "alerts" → add ANOMALY placeholder
        """

    def ingest_rag_results(self, rag_results: list[dict]) -> list[str]:
        """
        Extract relationships from RAG results.

        Rules:
        1. Alert data → ANOMALY nodes with caused_by edges
        2. Maintenance records → TASK nodes with explains edges
        3. Correlated metrics → CORRELATED_WITH edges
        4. Equipment hierarchy → PART_OF edges
        """

    def resolve_pronoun(self, text: str) -> Optional[FocusNode]:
        """
        Resolve pronouns ("it", "this", "the same one") against the graph.

        Resolution order:
        1. root_node (primary focus equipment)
        2. Most recently referenced equipment node
        3. Most recently referenced anomaly
        4. None (unresolvable — system must ask)
        """

    def merge_comparison_target(self, new_equipment: str) -> str:
        """
        When user says "compare with pump 5", add pump_005 node
        and COMPARED_WITH edge to current root.
        """
```

### State Ownership

| State | Owner | Storage | Lifetime |
|-------|-------|---------|----------|
| `SemanticFocusGraph` instance | `Layer2Orchestrator.context["focus_graph"]` | In-memory dict, serialized to JSON | Session (cleared on WIDGET_INTERACTIVE_EXIT) |
| Graph version counter | Backend orchestrator | In-memory | Session |
| Graph serialization for frontend | Sent in `context_update` of `OrchestratorResponse` | JSON over HTTP | Per-response |

### Execution Flow

```
1. User enters interactive mode on Pump 4 KPI widget
   → Frontend emits WIDGET_INTERACTIVE_ENTER {equipment: "pump_004", metric: "vibration_de_mm_s"}
   → useVoicePipeline sets widget_context on Layer2Service
   → Layer2Service.context.widget_context = {...}

2. Auto-query: "Tell me more about Pump 4 Vibration"
   → POST /api/layer2/orchestrate/ {transcript, context: {widget_context}}
   → Orchestrator: widget_context detected, no existing focus_graph
   → FocusGraphBuilder creates new graph:
       Node: equipment:pump_004 (root)
       Node: metric:vibration_de_mm_s
       Edge: pump_004 --measured_by--> vibration_de_mm_s
   → Graph stored in self.context["focus_graph"]
   → Graph serialized to context_update.focus_graph in response
   → Frontend receives focus_graph, stores in Layer2Service.context

3. Follow-up: "is the temperature also high?"
   → POST /api/layer2/orchestrate/ {transcript, context: {widget_context, focus_graph}}
   → IntentParser sees "temperature" but no explicit device
   → FocusGraphBuilder.resolve_pronoun("it") → pump_004
   → Builder adds: metric:oil_temperature_top_c node
   → Builder adds: pump_004 --measured_by--> oil_temperature_top_c edge
   → Intent entities now: {devices: ["pump_004"], metrics: ["vibration_de_mm_s", "oil_temperature_top_c"]}
   → Widget selector builds comparison dashboard

4. Follow-up: "compare it with pump 5"
   → Builder.merge_comparison_target("pump_005")
   → Adds: equipment:pump_005 node
   → Adds: pump_004 --compared_with--> pump_005 edge
   → Widget selector sees 2 equipment → comparison + trend-multi-line

5. Follow-up: "why is the vibration high but temperature normal?"
   → Cross-widget reasoning engine triggered (see Upgrade 2)
   → Builder adds: anomaly:vibration_threshold_breach node
   → Builder adds: vibration_threshold_breach --affects--> pump_004
   → Causal knowledge engine returns hypotheses (see Upgrade 8)
```

### Integration Points

**Orchestrator** (`orchestrator.py` line ~330):
```python
# After extracting widget_context
focus_graph_data = self.context.get("focus_graph")
if focus_graph_data:
    focus_graph = SemanticFocusGraph.from_dict(focus_graph_data)
elif widget_context:
    focus_graph = SemanticFocusGraph(session_id=query_id)
else:
    focus_graph = None

if focus_graph:
    builder = FocusGraphBuilder(focus_graph)
    builder.ingest_intent(parsed)
    # After RAG:
    builder.ingest_rag_results(rag_results)
    self.context["focus_graph"] = focus_graph.to_dict()
```

**Intent Parser** — `_merge_interactive_context()` upgraded:
```python
def _merge_interactive_context(self, result, widget_context, focus_graph):
    if focus_graph:
        # Use graph for pronoun resolution instead of flat equipment string
        resolved = FocusGraphBuilder(focus_graph).resolve_pronoun(result.raw_text)
        if resolved and not result.entities.get("devices"):
            result.entities.setdefault("devices", []).append(resolved.label)
    elif widget_context:
        # Fallback to flat context (backward compat)
        ...
```

**Widget Selector** — `_build_interactive_prompt_section()` upgraded:
```python
if focus_graph:
    prompt += f"\n\n## SEMANTIC FOCUS\n{focus_graph.to_prompt_context()}"
```

**Frontend** — `Layer2Service.context.focus_graph` is passed through on every orchestrate call. No frontend graph manipulation — the backend is the single source of truth.

### Failure Modes

| Failure | Detection | Handling |
|---------|-----------|----------|
| Graph exceeds 50 nodes (runaway accumulation) | `len(graph.nodes) > 50` | Prune nodes with `reference_count == 1` and `last_referenced > 5min ago` |
| Pronoun resolution ambiguous (2+ equally-ranked candidates) | `resolve_pronoun` returns None | System asks: "Do you mean {candidate_1} or {candidate_2}?" |
| Graph deserialization fails (corrupt JSON from frontend) | `from_dict` raises | Log warning, create fresh graph from widget_context |
| Edge creates cycle (A caused_by B caused_by A) | Cycle detection in `add_edge` | Reject edge, log warning |
| Session ID mismatch (graph from different session) | Compare `graph.session_id != current_session` | Discard stale graph, create fresh |

---

## 2. Cross-Widget Reasoning Engine

### Problem

Widgets are currently rendered independently. When pump vibration is high but temperature is normal, the system cannot reason about why — it just shows both widgets. An operator asking "why is vibration high but temperature normal?" gets a generic LLM response without structured diagnostic reasoning.

### Data Structures

**Location**: `backend/layer2/reasoning_engine.py` (new file)

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ReasoningType(Enum):
    CAUSAL = "causal"                # Why is X happening?
    COMPARATIVE = "comparative"      # Why is A high but B normal?
    DIAGNOSTIC = "diagnostic"        # What could explain this pattern?
    PREDICTIVE = "predictive"        # What will happen if this continues?
    CORRELATIVE = "correlative"      # Are these related?


@dataclass
class DataPoint:
    """A concrete data observation from a widget or RAG."""
    equipment: str
    metric: str
    value: float
    unit: str
    timestamp: float
    status: str                      # "normal", "warning", "critical"
    threshold: Optional[float] = None
    source: str = ""                 # widget scenario or RAG collection


@dataclass
class ReasoningQuery:
    """Structured input to the reasoning engine."""
    type: ReasoningType
    question: str                    # Original user question
    observations: list[DataPoint] = field(default_factory=list)
    focus_graph: Optional[dict] = None
    constraints: list = field(default_factory=list)


@dataclass
class Hypothesis:
    """A possible explanation with evidence."""
    id: str
    statement: str                   # "Bearing wear causes increased vibration without temperature rise"
    confidence: float                # 0.0-1.0
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    check_steps: list[str] = field(default_factory=list)  # What to verify next
    source: str = "knowledge_base"   # knowledge_base | inferred | historical_match


@dataclass
class ReasoningResult:
    """Output of the reasoning engine."""
    query_type: ReasoningType
    hypotheses: list[Hypothesis] = field(default_factory=list)
    known_facts: list[str] = field(default_factory=list)
    unknown_factors: list[str] = field(default_factory=list)
    recommended_checks: list[str] = field(default_factory=list)
    confidence: float = 0.0          # Overall reasoning confidence
    reasoning_chain: list[str] = field(default_factory=list)  # Step-by-step logic
    execution_time_ms: int = 0
```

### Reasoning Engine

```python
class CrossWidgetReasoningEngine:
    """
    Structured diagnostic reasoning across multiple widget data sources.

    NOT a pure LLM text generator. Uses:
    1. Rule-based pattern matching for known failure modes
    2. Statistical correlation from historical data
    3. Knowledge base lookup for equipment-specific diagnostics
    4. LLM synthesis ONLY for natural language output
    """

    # Known diagnostic patterns (industrial domain knowledge)
    DIAGNOSTIC_PATTERNS = [
        {
            "pattern": "vibration_high_temperature_normal",
            "conditions": {
                "vibration": {"status": "critical", "threshold_exceeded": True},
                "temperature": {"status": "normal", "threshold_exceeded": False},
            },
            "hypotheses": [
                {
                    "statement": "Bearing wear or misalignment — mechanical vibration "
                                 "increases before thermal effects manifest",
                    "confidence": 0.72,
                    "check_steps": [
                        "Check vibration frequency spectrum for bearing defect frequencies",
                        "Review last alignment report",
                        "Check bearing replacement date",
                    ],
                },
                {
                    "statement": "Loose mounting or foundation issue — causes vibration "
                                 "without thermal impact",
                    "confidence": 0.45,
                    "check_steps": [
                        "Inspect mounting bolts",
                        "Check foundation for cracks",
                    ],
                },
            ],
        },
        # ... 20+ patterns for common industrial failure modes
    ]

    def reason(self, query: ReasoningQuery) -> ReasoningResult:
        """
        Execute reasoning pipeline:
        1. Collect data points from current widget data + RAG
        2. Match against known diagnostic patterns
        3. If no pattern match, run statistical correlation
        4. If still no match, use LLM with structured prompt
        5. Score and rank hypotheses
        6. Identify unknowns and recommended checks
        """

    def _match_diagnostic_patterns(self, observations: list[DataPoint]) -> list[Hypothesis]:
        """Rule-based pattern matching against DIAGNOSTIC_PATTERNS."""

    def _run_statistical_correlation(self, observations: list[DataPoint],
                                      equipment: str) -> list[Hypothesis]:
        """
        Query PostgreSQL timeseries for historical correlation:
        - Pearson correlation between metric pairs over last 30 days
        - Granger causality test (does A predict B with lag?)
        - Anomaly co-occurrence frequency
        """

    def _synthesize_with_llm(self, query: ReasoningQuery,
                              pattern_results: list[Hypothesis],
                              stat_results: list[Hypothesis]) -> ReasoningResult:
        """
        LLM prompt (structured, not open-ended):

        Given these observations:
        {observations as structured list}

        Known diagnostic matches:
        {pattern_results}

        Statistical correlations:
        {stat_results}

        Equipment context from focus graph:
        {focus_graph.to_prompt_context()}

        Output JSON:
        {
          "hypotheses": [...],
          "known_facts": [...],
          "unknown_factors": [...],
          "recommended_checks": [...],
          "reasoning_chain": ["Step 1: ...", "Step 2: ...", ...]
        }
        """
```

### Execution Flow

```
1. User asks: "why is vibration high but temperature normal on pump 4?"

2. Intent parser detects:
   type: "query"
   primary_characteristic: "comparison" (two metrics being contrasted)
   entities: {devices: ["pump_004"]}
   → Flags: reasoning_required = True (detected "why" + contrast pattern)

3. Plan compiler (Upgrade 4) generates:
   Step 1: RETRIEVE vibration data for pump_004 (last 24h)
   Step 2: RETRIEVE temperature data for pump_004 (last 24h)
   Step 3: REASON across both data points
   Step 4: VISUALIZE findings

4. Data collector fetches:
   DataPoint(equipment="pump_004", metric="vibration_de_mm_s", value=3.2, status="critical", threshold=2.5)
   DataPoint(equipment="pump_004", metric="oil_temperature_top_c", value=42.0, status="normal", threshold=85.0)

5. Reasoning engine:
   a. Pattern match: "vibration_high_temperature_normal" → 2 hypotheses
   b. Statistical: Query pg for 30-day correlation → Pearson r=0.12 (weak) between vibration and temp
   c. LLM synthesis: Combine into ranked hypotheses with evidence

6. ReasoningResult:
   hypotheses: [
     {statement: "Bearing wear...", confidence: 0.72, check_steps: [...]},
     {statement: "Loose mounting...", confidence: 0.45, check_steps: [...]},
   ]
   known_facts: ["Vibration at 3.2mm/s exceeds threshold of 2.5mm/s", "Temperature normal at 42°C"]
   unknown_factors: ["Vibration frequency spectrum not available", "Last bearing replacement date unknown"]
   recommended_checks: ["Request vibration spectrum analysis", "Check maintenance log for bearing replacement"]

7. Widget selector builds dashboard:
   - Hero: Comparison widget (vibration vs temperature trend overlay)
   - Expanded: "Diagnostic Panel" widget (new — shows hypotheses, evidence, checks)
   - Normal: KPI widgets for each metric
   - Normal: Timeline showing related maintenance events

8. Voice response:
   "Pump 4 vibration is at 3.2 millimeters per second, above the 2.5 threshold,
    but temperature is normal at 42 degrees. The most likely explanation is bearing
    wear or misalignment — vibration typically rises before temperature effects show.
    I'd recommend checking the vibration frequency spectrum and reviewing the last
    alignment report. There are two things I don't know yet: when the bearings were
    last replaced, and what the frequency spectrum looks like."
```

### New Widget: Diagnostic Panel

**Location**: `frontend/src/components/layer4/widgets/DiagnosticPanel.tsx`

```typescript
interface DiagnosticPanelProps {
  demoData: {
    hypotheses: Array<{
      statement: string;
      confidence: number;
      supporting_evidence: string[];
      contradicting_evidence: string[];
      check_steps: string[];
    }>;
    known_facts: string[];
    unknown_factors: string[];
    recommended_checks: string[];
    reasoning_chain: string[];
  };
}
```

Renders:
- Ranked hypothesis cards with confidence bars
- Evidence sections (supporting/contradicting)
- "What we don't know" section
- "Recommended next steps" checklist

### Failure Modes

| Failure | Detection | Handling |
|---------|-----------|----------|
| No diagnostic pattern matches | `_match_diagnostic_patterns` returns empty | Fall through to statistical correlation, then LLM |
| Statistical query exceeds 2s budget | Timeout in ThreadPoolExecutor | Skip statistical step, use pattern + LLM only |
| LLM returns malformed reasoning JSON | JSON parse failure in `_synthesize_with_llm` | Return pattern-match results only, flag "reasoning incomplete" |
| Insufficient data points (only 1 metric) | `len(observations) < 2` | Skip reasoning, return single-metric analysis |
| Contradictory evidence exceeds supporting | `contradicting > supporting` for top hypothesis | Demote hypothesis confidence, surface contradiction explicitly |

---

## 3. Time-Travel + Comparison Focus

### Problem

Time is implicit in the current system. Users cannot say "show me what the dashboard looked like before the anomaly" or "compare last week's performance with this week". The system has no concept of temporal snapshots or delta calculations.

### Data Structures

**Location**: `backend/layer2/time_context.py` (new file)

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime, timedelta


class TimeReference(Enum):
    ABSOLUTE = "absolute"            # "2026-02-07 14:00"
    RELATIVE = "relative"            # "last 24 hours"
    EVENT_ANCHORED = "event_anchored"  # "before the anomaly", "after maintenance"
    COMPARATIVE = "comparative"      # "this week vs last week"


@dataclass
class TimeWindow:
    start: datetime
    end: datetime
    reference_type: TimeReference
    label: str                       # "Last 24 hours", "Pre-anomaly (Feb 7 14:00-18:00)"
    anchor_event: Optional[str] = None  # Event ID that anchors this window
    confidence: float = 1.0          # How certain we are about the time resolution


@dataclass
class TimeComparison:
    """Two time windows being compared."""
    baseline: TimeWindow             # "before" or "normal" period
    target: TimeWindow               # "after" or "current" period
    delta_type: str                  # "pre_post_anomaly", "week_over_week", "shift_comparison"
    alignment: str                   # "time_of_day" | "sequential" | "event_relative"


@dataclass
class TemporalSnapshot:
    """A captured state of equipment metrics at a specific time."""
    equipment_id: str
    timestamp: datetime
    metrics: dict[str, float]        # metric_name → value
    alerts_active: list[str]
    status: str                      # "normal", "degraded", "critical"


@dataclass
class TemporalDelta:
    """Computed difference between two snapshots."""
    equipment_id: str
    baseline_time: datetime
    target_time: datetime
    metric_deltas: dict[str, dict]   # metric → {baseline, target, delta, percent_change, significance}
    alert_changes: dict              # {new: [...], resolved: [...], persistent: [...]}
    status_change: Optional[str]     # "normal→critical" or None
```

### Time Resolution Engine

```python
class TimeResolver:
    """Resolves natural language time references to concrete TimeWindows."""

    # Event anchors (detected from alerts/maintenance DB)
    def resolve(self, text: str, focus_graph: SemanticFocusGraph = None) -> Optional[TimeWindow]:
        """
        Resolution cascade:
        1. Explicit datetime parsing ("February 7th at 2pm")
        2. Relative expressions ("last 24 hours", "yesterday", "past week")
        3. Event-anchored ("before the anomaly", "after the last maintenance")
           → Requires focus_graph to identify which anomaly/event
           → Queries alerts table for anomaly timestamp
           → Creates window: [anomaly_time - 6h, anomaly_time]
        4. Comparative ("this week vs last week")
           → Returns TimeComparison with two windows
        """

    def resolve_event_anchor(self, anchor_text: str,
                              focus_graph: SemanticFocusGraph) -> Optional[datetime]:
        """
        Find the event being referenced:
        1. Look for ANOMALY nodes in focus graph → get timestamp from properties
        2. Look for TASK nodes → get timestamp
        3. Query alerts DB for most recent alert on root equipment
        4. Query maintenance DB for most recent work order
        """

    def build_comparison(self, text: str, focus_graph: SemanticFocusGraph) -> Optional[TimeComparison]:
        """
        Parse comparative time references:
        - "before and after" → pre/post event windows
        - "this week vs last week" → week_over_week
        - "morning shift vs night shift" → shift_comparison
        - "compare with January" → month_over_month
        """
```

### Snapshot Engine

```python
class SnapshotEngine:
    """Captures and compares equipment state at different times."""

    def capture_snapshot(self, equipment_id: str, timestamp: datetime) -> TemporalSnapshot:
        """
        Query PostgreSQL timeseries for metric values at/near timestamp.
        Query alerts table for active alerts at timestamp.
        """

    def compute_delta(self, baseline: TemporalSnapshot,
                       target: TemporalSnapshot) -> TemporalDelta:
        """
        For each metric present in both snapshots:
        - Compute absolute delta
        - Compute percent change
        - Compute statistical significance (z-score against 30-day distribution)
        Mark significance: trivial (<0.5σ), notable (0.5-2σ), significant (>2σ)
        """
```

### Integration with Widget Selector

When `TimeComparison` is present, the widget selector receives:
```python
## TIME COMPARISON CONTEXT
Comparing: "Pre-anomaly (Feb 7 14:00-18:00)" vs "Post-anomaly (Feb 7 18:00-22:00)"
Type: pre_post_anomaly

Delta summary:
- vibration_de_mm_s: 1.8 → 3.2 (+78%, SIGNIFICANT)
- oil_temperature_top_c: 41 → 42 (+2%, trivial)
- motor_power_kw: 15.2 → 16.8 (+11%, notable)

Build dashboard showing:
1. Overlay trends (both windows on same chart)
2. Delta KPIs (showing change with significance indicators)
3. Snapshot comparison (side-by-side metrics)
```

This drives the selector to choose:
- Hero: `trend-multi-line` with both time windows overlaid
- Expanded: New `snapshot-comparison` widget (split-screen before/after)
- Normal: KPI widgets with delta indicators
- Normal: `alerts` showing alert state changes

### New Widget: Snapshot Comparison

**Location**: `frontend/src/components/layer4/widgets/SnapshotComparison.tsx`

```typescript
interface SnapshotComparisonProps {
  demoData: {
    baseline: {
      label: string;               // "Pre-anomaly"
      timestamp: string;
      metrics: Record<string, { value: number; unit: string; status: string }>;
    };
    target: {
      label: string;               // "Post-anomaly"
      timestamp: string;
      metrics: Record<string, { value: number; unit: string; status: string }>;
    };
    deltas: Record<string, {
      absolute: number;
      percent: number;
      significance: "trivial" | "notable" | "significant";
    }>;
  };
}
```

### Failure Modes

| Failure | Detection | Handling |
|---------|-----------|----------|
| Event anchor not found (no anomaly in focus graph or DB) | `resolve_event_anchor` returns None | Ask user: "Which event? I see alerts on {dates}" |
| Time window has no data (equipment was offline) | PostgreSQL returns 0 rows | Show "No data available for this period" in widget |
| Time comparison windows overlap | `baseline.end > target.start` | Adjust windows to not overlap, warn user |
| Historical data older than retention period | PostgreSQL partitions pruned | Return available data with caveat: "Data only available from {date}" |

---

## 4. Intent → Plan → Execute Orchestration

### Problem

Current flow: `IntentParser.parse()` → `WidgetSelector.select()` → widgets. This is a direct mapping with no intermediate representation. The system cannot:
- Show the user what it's about to do before doing it
- Detect under-specified requests
- Retry individual steps on failure
- Cancel mid-execution

### Data Structures

**Location**: `backend/layer2/plan_compiler.py` (new file)

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import uuid
import time


class StepType(Enum):
    PARSE = "parse"                  # Analyze user intent
    RETRIEVE = "retrieve"            # Fetch data from RAG/PG
    VALIDATE = "validate"            # Check data freshness, completeness
    CORRELATE = "correlate"          # Cross-reference multiple data sources
    REASON = "reason"                # Diagnostic reasoning (Upgrade 2)
    RESOLVE_TIME = "resolve_time"    # Time window resolution (Upgrade 3)
    SNAPSHOT = "snapshot"            # Capture temporal snapshot
    SELECT_WIDGETS = "select_widgets"  # Choose widget layout
    COLLECT_DATA = "collect_data"    # Schema-driven data collection
    GENERATE_RESPONSE = "generate_response"  # Voice response
    ASK_USER = "ask_user"            # Clarification needed


class StepStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class PlanStatus(Enum):
    COMPILING = "compiling"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"              # Waiting for user clarification


@dataclass
class PlanStep:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: StepType = StepType.RETRIEVE
    description: str = ""            # Human-readable: "Fetch vibration data for pump_004"
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)  # Step IDs this depends on
    constraints: list = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    execution_time_ms: Optional[int] = None
    error: Optional[str] = None
    budget_ms: int = 2000            # Maximum allowed execution time


@dataclass
class ExecutionPlan:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    steps: list[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.COMPILING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    total_budget_ms: int = 8000
    elapsed_ms: int = 0
    blocked_reason: Optional[str] = None  # Why we're asking the user

    def get_ready_steps(self) -> list[PlanStep]:
        """Return steps whose dependencies are all completed."""
        completed_ids = {s.id for s in self.steps if s.status == StepStatus.COMPLETED}
        return [s for s in self.steps
                if s.status == StepStatus.PENDING
                and all(dep in completed_ids for dep in s.dependencies)]

    def is_blocked(self) -> bool:
        return any(s.type == StepType.ASK_USER and s.status == StepStatus.PENDING for s in self.steps)
```

### Plan Compiler

```python
class PlanCompiler:
    """
    Compiles ParsedIntent + SemanticFocusGraph into an ExecutionPlan.

    Compilation rules:
    1. Every plan starts with PARSE (already done — intent is input)
    2. For each entity × metric combination: add RETRIEVE step
    3. If data freshness constraint exists: add VALIDATE step after each RETRIEVE
    4. If reasoning_required flag set: add REASON step (depends on all RETRIEVEs)
    5. If time comparison detected: add RESOLVE_TIME + SNAPSHOT steps
    6. Always ends with SELECT_WIDGETS → COLLECT_DATA → GENERATE_RESPONSE
    7. If intent is under-specified: insert ASK_USER step and set status=BLOCKED
    """

    def compile(self, intent: ParsedIntent, focus_graph: SemanticFocusGraph = None,
                constraints: list = None) -> ExecutionPlan:
        plan = ExecutionPlan()

        # Detect under-specification
        ambiguities = self._detect_ambiguities(intent, focus_graph)
        if ambiguities:
            plan.steps.append(PlanStep(
                type=StepType.ASK_USER,
                description=f"Clarification needed: {ambiguities[0]}",
                inputs={"ambiguities": ambiguities},
            ))
            plan.status = PlanStatus.BLOCKED
            plan.blocked_reason = ambiguities[0]
            return plan

        # Build retrieval steps
        entities = intent.entities.get("devices", [])
        if not entities and focus_graph:
            root = focus_graph.get_root_equipment()
            if root:
                entities = [root.properties.get("equipment_id", root.label)]

        retrieve_ids = []
        for entity in entities:
            step = PlanStep(
                type=StepType.RETRIEVE,
                description=f"Fetch data for {entity}",
                inputs={"equipment": entity, "domains": intent.domains},
                budget_ms=2000,
            )
            plan.steps.append(step)
            retrieve_ids.append(step.id)

        # Validation steps
        validate_ids = []
        if constraints:
            for rid in retrieve_ids:
                vstep = PlanStep(
                    type=StepType.VALIDATE,
                    description="Validate data freshness and completeness",
                    dependencies=[rid],
                    inputs={"constraints": [c.__dict__ for c in constraints]},
                    budget_ms=100,
                )
                plan.steps.append(vstep)
                validate_ids.append(vstep.id)

        # Reasoning step (if causal/diagnostic query)
        reason_deps = validate_ids if validate_ids else retrieve_ids
        reasoning_required = self._needs_reasoning(intent)
        reason_id = None
        if reasoning_required:
            rstep = PlanStep(
                type=StepType.REASON,
                description="Cross-widget diagnostic reasoning",
                dependencies=reason_deps,
                budget_ms=3000,
            )
            plan.steps.append(rstep)
            reason_id = rstep.id

        # Time resolution (if temporal query)
        time_id = None
        if self._needs_time_resolution(intent):
            tstep = PlanStep(
                type=StepType.RESOLVE_TIME,
                description="Resolve time windows for comparison",
                inputs={"time_refs": intent.entities.get("time", [])},
                budget_ms=500,
            )
            plan.steps.append(tstep)
            time_id = tstep.id

        # Widget selection (depends on all prior steps)
        all_deps = reason_deps + ([reason_id] if reason_id else []) + ([time_id] if time_id else [])
        select_step = PlanStep(
            type=StepType.SELECT_WIDGETS,
            description="Select widget layout",
            dependencies=all_deps,
            budget_ms=3000,
        )
        plan.steps.append(select_step)

        # Data collection
        collect_step = PlanStep(
            type=StepType.COLLECT_DATA,
            description="Collect data for each widget",
            dependencies=[select_step.id],
            budget_ms=2000,
        )
        plan.steps.append(collect_step)

        # Voice response
        response_step = PlanStep(
            type=StepType.GENERATE_RESPONSE,
            description="Generate voice response",
            dependencies=[collect_step.id],
            budget_ms=3000,
        )
        plan.steps.append(response_step)

        plan.status = PlanStatus.READY
        return plan

    def _detect_ambiguities(self, intent: ParsedIntent,
                             focus_graph: SemanticFocusGraph = None) -> list[str]:
        """
        Detect when a query is too vague to execute:
        1. No entities and no focus graph → "Which equipment?"
        2. "Compare" intent with only 1 entity and no graph → "Compare with what?"
        3. "Show trend" with no time reference → use default (24h), no ambiguity
        4. Action intent with no target → "What should I start/stop?"
        """
        ambiguities = []
        entities = intent.entities.get("devices", [])
        has_context = focus_graph and focus_graph.get_root_equipment()

        if not entities and not has_context:
            if intent.type == "query" and intent.primary_characteristic not in ("health_status", "top_consumers", "alerts"):
                ambiguities.append("Which equipment are you asking about?")

        if intent.primary_characteristic == "comparison":
            total_entities = len(entities)
            if has_context:
                total_entities += len(focus_graph.get_all_equipment())
            if total_entities < 2:
                ambiguities.append("Compare with which other equipment?")

        if intent.type in ("action_control",) and not entities and not has_context:
            ambiguities.append("Which equipment should I control?")

        return ambiguities

    def _needs_reasoning(self, intent: ParsedIntent) -> bool:
        """Detect queries that need cross-widget reasoning."""
        why_patterns = ["why", "explain", "cause", "reason", "but", "although", "despite", "however"]
        text = intent.raw_text.lower()
        return any(p in text for p in why_patterns)

    def _needs_time_resolution(self, intent: ParsedIntent) -> bool:
        """Detect queries that need temporal resolution."""
        time_patterns = ["before", "after", "compare.*week", "vs.*last", "trend.*since",
                          "pre.*anomaly", "post.*maintenance", "yesterday.*vs"]
        text = intent.raw_text.lower()
        import re
        return any(re.search(p, text) for p in time_patterns)
```

### Plan Executor

```python
class PlanExecutor:
    """
    Executes an ExecutionPlan step by step.

    Key behaviors:
    - Parallel execution of independent steps (no dependency conflicts)
    - Budget enforcement: kills steps that exceed their budget_ms
    - Propagates outputs from completed steps to dependent steps' inputs
    - On step failure: marks as FAILED, checks if downstream can proceed
    - On plan cancellation: marks all PENDING steps as CANCELLED
    """

    def __init__(self, orchestrator: "Layer2Orchestrator"):
        self.orchestrator = orchestrator
        self.executor = ThreadPoolExecutor(max_workers=5)

    async def execute(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Execute plan until completion, failure, or cancellation."""
        plan.status = PlanStatus.EXECUTING
        start_time = time.time()

        while True:
            ready = plan.get_ready_steps()
            if not ready:
                # Check if we're done or stuck
                pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
                if not pending:
                    plan.status = PlanStatus.COMPLETED
                    break
                elif plan.is_blocked():
                    plan.status = PlanStatus.BLOCKED
                    break
                else:
                    # Stuck: pending steps with failed dependencies
                    plan.status = PlanStatus.FAILED
                    break

            # Execute ready steps in parallel
            futures = {}
            for step in ready:
                step.status = StepStatus.EXECUTING
                future = self.executor.submit(self._execute_step, step, plan)
                futures[future] = step

            # Wait for all with timeout
            remaining_budget = plan.total_budget_ms - int((time.time() - start_time) * 1000)
            if remaining_budget <= 0:
                for step in ready:
                    step.status = StepStatus.FAILED
                    step.error = "Plan budget exceeded"
                plan.status = PlanStatus.FAILED
                break

            for future in as_completed(futures, timeout=remaining_budget / 1000):
                step = futures[future]
                try:
                    result = future.result(timeout=step.budget_ms / 1000)
                    step.outputs = result
                    step.status = StepStatus.COMPLETED
                except Exception as e:
                    step.status = StepStatus.FAILED
                    step.error = str(e)

        plan.elapsed_ms = int((time.time() - start_time) * 1000)
        plan.completed_at = time.time()
        return plan

    def cancel(self, plan: ExecutionPlan):
        """Cancel all pending/executing steps."""
        for step in plan.steps:
            if step.status in (StepStatus.PENDING, StepStatus.EXECUTING):
                step.status = StepStatus.CANCELLED
        plan.status = PlanStatus.CANCELLED

    def _execute_step(self, step: PlanStep, plan: ExecutionPlan) -> dict:
        """Route step to appropriate handler."""
        handlers = {
            StepType.RETRIEVE: self._handle_retrieve,
            StepType.VALIDATE: self._handle_validate,
            StepType.CORRELATE: self._handle_correlate,
            StepType.REASON: self._handle_reason,
            StepType.RESOLVE_TIME: self._handle_resolve_time,
            StepType.SNAPSHOT: self._handle_snapshot,
            StepType.SELECT_WIDGETS: self._handle_select_widgets,
            StepType.COLLECT_DATA: self._handle_collect_data,
            StepType.GENERATE_RESPONSE: self._handle_generate_response,
        }
        handler = handlers.get(step.type)
        if not handler:
            raise ValueError(f"No handler for step type: {step.type}")

        start = time.time()
        result = handler(step, plan)
        step.execution_time_ms = int((time.time() - start) * 1000)
        return result
```

### Integration with Orchestrator

Replace the current monolithic `_process_v2()` method:

```python
# Current (monolithic):
def _process_v2(self, transcript, context):
    parsed = self._intent_parser.parse(transcript)
    # ... 200 lines of sequential steps ...

# New (plan-based):
def _process_v2(self, transcript, context):
    parsed = self._intent_parser.parse(transcript, widget_context=widget_context)
    focus_graph = self._get_or_create_focus_graph(context, parsed)
    constraints = self._build_constraints(context)

    # Compile plan
    compiler = PlanCompiler()
    plan = compiler.compile(parsed, focus_graph, constraints)

    # Check if blocked (needs user clarification)
    if plan.status == PlanStatus.BLOCKED:
        return self._build_clarification_response(plan)

    # Execute plan
    executor = PlanExecutor(self)
    plan = executor.execute(plan)

    # Build response from plan outputs
    return self._build_response_from_plan(plan, parsed, focus_graph)
```

### Failure Modes

| Failure | Detection | Handling |
|---------|-----------|----------|
| Plan compilation takes >500ms | Timer in `compile()` | Fallback to direct intent→widget mapping (current behavior) |
| Step exceeds budget | `execution_time_ms > budget_ms` | Kill step, mark FAILED, try to proceed without it |
| All RETRIEVE steps fail | No completed RETRIEVE steps | Return voice response explaining failure, no widgets |
| Plan total budget exceeded (8s) | `elapsed_ms > total_budget_ms` | Cancel remaining steps, return partial results |
| ASK_USER step generated but voice pipeline can't handle | Blocked plan returned to frontend | Frontend shows clarification prompt in context bar |
| Circular dependency in plan | Detected during `get_ready_steps()` (infinite loop) | Break cycle by skipping lowest-priority step |

---

## 5. Constraint-Aware Decision Making

### Problem

The system currently treats all data as equally trustworthy. If sensor data is 6 hours stale, if confidence is 30%, if latency is 15s — it still renders widgets with full confidence. False certainty is a safety hazard in industrial operations.

### Data Structures

**Location**: `backend/layer2/constraints.py` (new file)

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ConstraintType(Enum):
    SAFETY = "safety"                    # Equipment safety limits
    DATA_FRESHNESS = "data_freshness"    # How recent is the data
    CONFIDENCE = "confidence"            # How certain is the answer
    LATENCY = "latency"                  # Response time budget
    COVERAGE = "coverage"               # Fraction of requested data available
    AUTHORITY = "authority"              # Does user have permission for this action


class ViolationAction(Enum):
    REFUSE = "refuse"                    # Don't answer at all
    QUALIFY = "qualify"                  # Answer with explicit caveats
    WARN = "warn"                        # Answer but show warning
    DEGRADE = "degrade"                  # Show partial/lower-quality answer


@dataclass
class Constraint:
    type: ConstraintType
    description: str                     # "Data must be < 5 minutes old"
    threshold: float                     # The required value
    actual: Optional[float] = None       # The measured value
    satisfied: bool = True
    action_on_violation: ViolationAction = ViolationAction.QUALIFY
    message: str = ""                    # Human-readable violation message

    def evaluate(self, actual: float) -> bool:
        self.actual = actual
        if self.type == ConstraintType.DATA_FRESHNESS:
            self.satisfied = actual <= self.threshold
        elif self.type == ConstraintType.CONFIDENCE:
            self.satisfied = actual >= self.threshold
        elif self.type == ConstraintType.LATENCY:
            self.satisfied = actual <= self.threshold
        elif self.type == ConstraintType.COVERAGE:
            self.satisfied = actual >= self.threshold
        else:
            self.satisfied = actual >= self.threshold

        if not self.satisfied:
            self.message = self._build_violation_message()
        return self.satisfied

    def _build_violation_message(self) -> str:
        messages = {
            ConstraintType.DATA_FRESHNESS: f"Data is {self.actual:.0f}s old (limit: {self.threshold:.0f}s)",
            ConstraintType.CONFIDENCE: f"Confidence is {self.actual:.0%} (minimum: {self.threshold:.0%})",
            ConstraintType.LATENCY: f"Response took {self.actual:.0f}ms (budget: {self.threshold:.0f}ms)",
            ConstraintType.COVERAGE: f"Only {self.actual:.0%} of requested data available (need: {self.threshold:.0%})",
            ConstraintType.SAFETY: f"Safety constraint violated: threshold={self.threshold}, actual={self.actual}",
        }
        return messages.get(self.type, f"Constraint violated: {self.type.value}")


@dataclass
class ConstraintEvaluation:
    """Result of evaluating all constraints for a response."""
    plan_id: str
    constraints: list[Constraint] = field(default_factory=list)
    all_satisfied: bool = True
    violations: list[Constraint] = field(default_factory=list)
    action: ViolationAction = ViolationAction.WARN
    qualification_text: str = ""     # Caveat text to prepend to voice response

    def evaluate_all(self):
        self.violations = [c for c in self.constraints if not c.satisfied]
        self.all_satisfied = len(self.violations) == 0
        if self.violations:
            # Highest severity violation determines action
            severity_order = [ViolationAction.REFUSE, ViolationAction.QUALIFY,
                              ViolationAction.WARN, ViolationAction.DEGRADE]
            self.action = min(
                (v.action_on_violation for v in self.violations),
                key=lambda a: severity_order.index(a)
            )
            self._build_qualification()

    def _build_qualification(self):
        if self.action == ViolationAction.REFUSE:
            self.qualification_text = (
                "I can't provide a reliable answer right now. "
                + " ".join(v.message for v in self.violations)
            )
        elif self.action == ViolationAction.QUALIFY:
            caveats = [v.message for v in self.violations]
            self.qualification_text = "Note: " + "; ".join(caveats) + ". "
        elif self.action == ViolationAction.WARN:
            self.qualification_text = "Warning: " + self.violations[0].message + ". "
```

### Default Constraint Profiles

```python
# Applied to every query
DEFAULT_CONSTRAINTS = [
    Constraint(
        type=ConstraintType.DATA_FRESHNESS,
        description="Timeseries data must be < 5 minutes old",
        threshold=300,  # seconds
        action_on_violation=ViolationAction.QUALIFY,
    ),
    Constraint(
        type=ConstraintType.CONFIDENCE,
        description="Intent confidence must be > 40%",
        threshold=0.4,
        action_on_violation=ViolationAction.QUALIFY,
    ),
    Constraint(
        type=ConstraintType.LATENCY,
        description="Total response time must be < 8 seconds",
        threshold=8000,  # ms
        action_on_violation=ViolationAction.DEGRADE,
    ),
    Constraint(
        type=ConstraintType.COVERAGE,
        description="At least 50% of requested data must be available",
        threshold=0.5,
        action_on_violation=ViolationAction.QUALIFY,
    ),
]

# Applied only to safety-critical queries (equipment control, alarms)
SAFETY_CONSTRAINTS = [
    Constraint(
        type=ConstraintType.SAFETY,
        description="Safety-critical data must be < 30 seconds old",
        threshold=30,
        action_on_violation=ViolationAction.REFUSE,
    ),
    Constraint(
        type=ConstraintType.CONFIDENCE,
        description="Safety actions require > 90% confidence",
        threshold=0.9,
        action_on_violation=ViolationAction.REFUSE,
    ),
]
```

### Integration with Voice Response

In the voice response generation step, if `ConstraintEvaluation.qualification_text` is non-empty, it is prepended:

```
"Note: The vibration data is 8 minutes old, which exceeds the 5-minute freshness limit.
With that caveat — pump 4 vibration was at 3.2 millimeters per second, above the 2.5 threshold."
```

### Integration with Widget Rendering (Frontend)

The `context_update` in `OrchestratorResponse` carries constraint violations:
```json
{
  "constraint_violations": [
    {"type": "data_freshness", "message": "Data is 480s old (limit: 300s)", "action": "qualify"}
  ]
}
```

The `WidgetSlot` component renders a constraint badge when violations exist:
- Yellow border + "Stale data" badge for freshness violations
- Red border + "Low confidence" badge for confidence violations
- No rendering for REFUSE actions (voice explains why)

### Failure Modes

| Failure | Detection | Handling |
|---------|-----------|----------|
| All data sources stale (>5min) | All RETRIEVE steps report age > threshold | QUALIFY: Prepend "data may be outdated" to response |
| Safety constraint violated on action intent | REFUSE evaluation | Return only voice: "I can't execute that action — {reason}" |
| Confidence < 20% (effectively random) | Intent parser reports < 0.2 | REFUSE: "I'm not confident I understood your request. Could you rephrase?" |
| Mixed violations (some qualify, some warn) | Multiple violations with different actions | Use most severe action (REFUSE > QUALIFY > WARN > DEGRADE) |

---

## 6. RL Upgrade: From Widgets → Decisions

### Problem

Current RL rewards widget clicks (`user_rating`, `widget_interactions`). This misses the full decision chain: Did the intent parse correctly? Was the plan good? Did the reasoning make sense? Were constraints properly applied? Negative signals (ignored widgets, quick exits, re-asks) are underweighted.

### Enhanced Experience Structure

```python
@dataclass
class DecisionExperience(Experience):
    """Extends Experience with full decision chain tracking."""

    # Decision chain (new)
    plan_steps_executed: list[dict] = field(default_factory=list)
    plan_steps_failed: list[dict] = field(default_factory=list)
    constraints_evaluated: list[dict] = field(default_factory=list)
    constraints_violated: list[dict] = field(default_factory=list)
    reasoning_hypotheses: list[dict] = field(default_factory=list)
    focus_graph_version: int = 0
    focus_graph_node_count: int = 0

    # Negative signals (new)
    widgets_ignored: list[str] = field(default_factory=list)       # Scenarios never interacted with
    time_to_exit_ms: Optional[int] = None                          # How fast user left
    re_ask_within_30s: bool = False                                # User repeated similar query
    user_interrupted: bool = False                                  # User said "stop" / "no"
    plan_was_cancelled: bool = False
    clarification_was_needed: bool = False

    # Operator action tracking (new)
    operator_created_work_order: bool = False                       # After seeing dashboard
    operator_escalated: bool = False                                # Sent to supervisor
    operator_overrode_suggestion: bool = False                      # Manually changed something
    operator_acknowledged_alert: bool = False
```

### Enhanced Reward Computation

```python
class DecisionRewardAggregator(RewardSignalAggregator):
    """Rewards the full decision chain, not just widget clicks."""

    DECISION_WEIGHTS = {
        # Positive signals
        "explicit_rating": 1.0,
        "widget_engagement": 0.3,
        "intent_confidence": 0.1,
        "plan_completion": 0.3,          # Did all plan steps succeed?
        "constraint_compliance": 0.2,     # Were constraints respected?
        "reasoning_used": 0.15,          # Did reasoning produce actionable hypotheses?
        "operator_action_taken": 0.5,    # Operator acted on the information

        # Negative signals
        "widgets_ignored_penalty": -0.2,  # Per ignored widget
        "quick_exit_penalty": -0.4,       # Exited within 5s
        "re_ask_penalty": -0.6,           # Had to ask again within 30s
        "interruption_penalty": -0.3,     # User interrupted the plan
        "override_penalty": -0.2,         # Operator overrode suggestion
        "constraint_violation_penalty": -0.5,  # System violated its own constraints
    }

    def compute_reward(self, exp: DecisionExperience) -> float:
        reward = 0.0

        # Positive signals
        if exp.user_rating == "up":
            reward += self.DECISION_WEIGHTS["explicit_rating"]
        elif exp.user_rating == "down":
            reward -= self.DECISION_WEIGHTS["explicit_rating"]

        reward += exp.intent_confidence * self.DECISION_WEIGHTS["intent_confidence"]

        # Plan completion reward
        if exp.plan_steps_executed:
            total = len(exp.plan_steps_executed) + len(exp.plan_steps_failed)
            completion_rate = len(exp.plan_steps_executed) / max(total, 1)
            reward += completion_rate * self.DECISION_WEIGHTS["plan_completion"]

        # Constraint compliance
        if exp.constraints_evaluated:
            violated = len(exp.constraints_violated)
            compliance = 1.0 - (violated / len(exp.constraints_evaluated))
            reward += compliance * self.DECISION_WEIGHTS["constraint_compliance"]

        # Operator actions (strong positive signal)
        if exp.operator_created_work_order or exp.operator_acknowledged_alert:
            reward += self.DECISION_WEIGHTS["operator_action_taken"]

        # Negative signals
        if exp.widgets_ignored:
            penalty = len(exp.widgets_ignored) * self.DECISION_WEIGHTS["widgets_ignored_penalty"]
            reward += max(penalty, -0.8)  # Cap penalty

        if exp.time_to_exit_ms is not None and exp.time_to_exit_ms < 5000:
            reward += self.DECISION_WEIGHTS["quick_exit_penalty"]

        if exp.re_ask_within_30s:
            reward += self.DECISION_WEIGHTS["re_ask_penalty"]

        if exp.user_interrupted:
            reward += self.DECISION_WEIGHTS["interruption_penalty"]

        return max(-2.0, min(2.0, reward))
```

### Enhanced Low-Rank Scorer

Current scorer input: 787-dim (768 intent embedding + 19 widget one-hot).

New scorer input: 819-dim:
```
768-dim intent embedding
+ 19-dim widget one-hot
+ 8-dim plan features:
    plan_step_count (normalized 0-1)
    reasoning_was_used (0/1)
    constraints_violated_count (normalized 0-1)
    time_comparison_active (0/1)
    focus_graph_depth (normalized 0-1)
    memory_tier_used (0=none, 0.33=session, 0.66=operator, 1.0=site)
    data_freshness_score (0-1)
    confidence_score (0-1)
+ 24-dim negative signal features:
    per-scenario ignore flags (19 scenarios × 0/1, padded to 24)
```

This means `INPUT_DIM` changes from 787 to 819. The low-rank factorization stays rank-8, so parameter count goes from ~6,872 to ~7,128 — still trains in milliseconds.

### Frontend Negative Signal Collection

```typescript
// In useVoicePipeline.ts — track widget ignore and quick exit

// On LAYOUT_UPDATE: record which widgets were delivered
const deliveredWidgets = useRef<Set<string>>(new Set());

// On WIDGET_INTERACTIVE_EXIT: compute time_to_exit
const interactiveEnteredAt = useRef<number>(0);

// Track which widgets got any interaction (hover > 2s, click, scroll)
const interactedWidgets = useRef<Set<string>>(new Set());

// On exit or new query: compute ignored = delivered - interacted
// Submit via feedback endpoint with negative signals
```

### Failure Modes

| Failure | Detection | Handling |
|---------|-----------|----------|
| Negative signals dominate (reward always < -1) | Running average of last 100 rewards < -0.5 | Reduce negative weights by 50%, log alert for review |
| Quick exit false positive (user just wanted KPI) | Simple KPI queries always exit fast | Exempt single-widget queries from quick_exit_penalty |
| Scorer diverges after input dim change | Loss > 2× baseline after retraining | Rollback to previous scorer checkpoint |
| Missing operator action tracking (no frontend integration) | `operator_*` fields always False | Degrade gracefully — these fields are additive, not required |

---

## 7. Memory Stratification

### Problem

Current system has only session-level context (`Layer2Service.context`). When an operator logs out and back in, everything is lost. When a different operator on the same facility asks about the same equipment, they start from zero. When the RL system learns preferences, it doesn't know whose preferences it learned.

### Data Structures

**Location**: `backend/layer2/memory.py` (new file)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time
import json


class MemoryTier(Enum):
    SESSION = "session"        # Current conversation, cleared on exit. TTL: session duration
    OPERATOR = "operator"      # Per-operator preferences/history. TTL: 90 days
    SITE = "site"              # Facility knowledge, shared. TTL: permanent


@dataclass
class MemoryEntry:
    id: str
    tier: MemoryTier
    key: str                           # Semantic key: "pump_004:last_investigation"
    value: dict                        # Structured content
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    created_by: str = ""               # user_id or "system"
    source: str = "explicit"           # explicit | inferred | rl_learned
    ttl: Optional[float] = None        # Seconds, None = use tier default

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl


class MemoryStore:
    """
    Three-tier memory store backed by SQLite (operator/site) and in-memory dict (session).

    Session tier: dict keyed by session_id, lives in RAM.
    Operator tier: SQLite table `operator_memory`, keyed by (user_id, key).
    Site tier: SQLite table `site_memory`, keyed by (site_id, key).
    """

    TIER_TTL_DEFAULTS = {
        MemoryTier.SESSION: None,      # Cleared explicitly on session end
        MemoryTier.OPERATOR: 90 * 86400,  # 90 days
        MemoryTier.SITE: None,         # Permanent
    }

    def __init__(self, db_path: str = "memory.sqlite3"):
        self._session_store: dict[str, dict[str, MemoryEntry]] = {}
        self._db_path = db_path
        self._init_db()

    def remember(self, entry: MemoryEntry, session_id: str = None,
                 user_id: str = None, site_id: str = "default"):
        """Store a memory entry in the appropriate tier."""

    def recall(self, key: str, tier: MemoryTier = None,
               session_id: str = None, user_id: str = None,
               site_id: str = "default") -> Optional[MemoryEntry]:
        """
        Recall a memory entry. Search order:
        1. Session (if session_id provided)
        2. Operator (if user_id provided)
        3. Site
        Returns first match. Updates access_count and accessed_at.
        """

    def recall_relevant(self, query: str, tier: MemoryTier = None,
                         limit: int = 5) -> list[MemoryEntry]:
        """
        Semantic search across memory entries.
        Uses BM25 on the `key` and `value` fields.
        Returns top-k relevant entries.
        """

    def forget_session(self, session_id: str):
        """Clear all session-tier memory for a session."""

    def promote(self, entry_id: str, from_tier: MemoryTier, to_tier: MemoryTier):
        """
        Promote a memory from lower to higher tier.
        E.g., Session → Operator (user found this useful across sessions).
        """
```

### What Gets Remembered

| Event | Memory Key | Tier | Content |
|-------|-----------|------|---------|
| User asks about pump_004 | `pump_004:last_query` | Session | `{query, timestamp, widgets_shown}` |
| User rates response up | `pump_004:preferred_widgets` | Operator | `{scenarios: ["trend", "kpi"], rating: "up"}` |
| System detects recurring alert | `pump_004:recurring_alert` | Site | `{alert_type, frequency, last_occurrence}` |
| Operator always ignores alerts widget | `operator:widget_preference` | Operator | `{suppressed: ["alerts"], preferred_sizes: {"trend": "hero"}}` |
| Bearing replacement recorded | `pump_004:bearing_replaced` | Site | `{date, technician, part_number, next_inspection}` |

### Integration with Focus Graph

The focus graph builder checks memory before creating nodes:
```python
def ingest_intent(self, intent, memory_store, session_id, user_id):
    # Check if we have prior knowledge about this equipment
    for device in intent.entities.get("devices", []):
        site_memory = memory_store.recall(f"{device}:known_issues", MemoryTier.SITE)
        if site_memory:
            # Add known anomaly nodes from site memory
            for issue in site_memory.value.get("issues", []):
                self.graph.add_node(FocusNode(
                    id=f"anomaly:{issue['id']}",
                    type=NodeType.ANOMALY,
                    label=issue["description"],
                    confidence=issue.get("confidence", 0.5),
                    source="site_memory",
                ))
```

### Integration with RL Scorer

The scorer's 819-dim input vector includes `memory_tier_used`:
```python
# In lora_scorer.py, when building input features:
memory_features = [0.0]  # Default: no memory
if memory_tier_used == MemoryTier.SESSION:
    memory_features = [0.33]
elif memory_tier_used == MemoryTier.OPERATOR:
    memory_features = [0.66]
elif memory_tier_used == MemoryTier.SITE:
    memory_features = [1.0]
```

This lets the scorer learn that site memory should weight equipment-specific widgets higher, while session memory should weight conversation-continuation widgets.

### Failure Modes

| Failure | Detection | Handling |
|---------|-----------|----------|
| SQLite DB locked (concurrent writes) | sqlite3.OperationalError | Retry with 100ms backoff, max 3 attempts |
| Memory store exceeds 100MB | `os.path.getsize(db_path) > 100MB` | Prune entries with `access_count == 0` and `accessed_at > 30 days` |
| Stale operator memory (preference changed) | Entry `accessed_at > 30 days` | Reduce confidence by 50% on recall |
| Conflicting site memories (two operators disagree) | Same key, different values | Keep both, use most recently accessed |
| Session store grows unbounded | `len(session_store[sid]) > 500` | Prune oldest entries by `created_at` |

---

## 8. RAG → Causal Knowledge Engine

### Problem

Current RAG is keyword-based retrieval: "pump vibration" → search ChromaDB → return documents. It cannot answer "what could cause high vibration on a centrifugal pump?" without having a document that literally says those words. No symptom matching, no failure mode reasoning, no self-healing when knowledge is missing.

### Data Structures

**Location**: `backend/layer2/causal_knowledge.py` (new file)

```python
@dataclass
class FailureMode:
    id: str
    equipment_type: str              # "centrifugal_pump", "transformer", "motor"
    symptom_signature: dict          # {"vibration": "high", "temperature": "normal", "noise": "increased"}
    root_cause: str                  # "bearing_wear"
    description: str
    confidence_base: float           # Base confidence for this failure mode
    resolution_steps: list[str]
    prevention_steps: list[str]
    mtbf_hours: Optional[float]      # Mean time between failures
    severity: str                    # "low", "medium", "high", "critical"
    sources: list[str]               # Document IDs or "domain_expert"


@dataclass
class SymptomMatch:
    failure_mode: FailureMode
    match_score: float               # 0.0-1.0, how well symptoms match
    matched_symptoms: list[str]      # Which symptoms matched
    missing_symptoms: list[str]      # Which symptoms we couldn't check
    contradicting_symptoms: list[str]  # Which symptoms contradict this mode


@dataclass
class KnowledgeGap:
    """A piece of missing knowledge the system detected."""
    id: str
    equipment_type: str
    query: str                       # What was asked
    expected_knowledge_type: str     # "failure_mode", "maintenance_procedure", "threshold_value"
    timestamp: float
    resolved: bool = False
    resolution: Optional[str] = None


class CausalKnowledgeEngine:
    """
    Replaces simple RAG with symptom-based failure mode matching.

    Knowledge sources (priority order):
    1. Structured failure mode database (FailureMode entries)
    2. Maintenance manual embeddings (ChromaDB operational_documents)
    3. Historical incident reports (ChromaDB maintenance_records)
    4. LLM general knowledge (last resort, lowest confidence)
    """

    def __init__(self, rag_pipeline, failure_mode_db_path: str = "failure_modes.json"):
        self.rag = rag_pipeline
        self.failure_modes = self._load_failure_modes(failure_mode_db_path)
        self.knowledge_gaps: list[KnowledgeGap] = []

    def diagnose(self, equipment_type: str, symptoms: dict,
                  focus_graph: SemanticFocusGraph = None) -> list[SymptomMatch]:
        """
        Match observed symptoms against known failure modes.

        Steps:
        1. Filter failure modes by equipment_type
        2. Score each against observed symptoms (Jaccard similarity)
        3. Boost scores with RAG-retrieved evidence
        4. Penalize for contradicting symptoms
        5. Log knowledge gaps for unmatched symptom combinations
        """

    def retrieve_by_similarity(self, query: str, equipment_type: str = None,
                                limit: int = 5) -> list[dict]:
        """
        Hybrid retrieval: ChromaDB embeddings + BM25 + failure mode matching.
        Returns results with confidence scores and source attribution.
        """

    def log_knowledge_gap(self, equipment_type: str, query: str,
                           expected_type: str):
        """
        Record missing knowledge for later ingestion.
        Stored in knowledge_gaps table, reviewed by engineers.
        """

    def get_pending_gaps(self) -> list[KnowledgeGap]:
        """Return unresolved knowledge gaps, ordered by frequency."""

    def _score_symptom_match(self, mode: FailureMode, symptoms: dict) -> SymptomMatch:
        """
        Scoring:
        - Each matching symptom: +1/total_symptoms
        - Each contradicting symptom: -0.5/total_symptoms
        - Missing symptoms: 0 (neutral, but tracked)
        - Final score = sum * confidence_base
        """
```

### Failure Mode Database

Seeded from industrial knowledge, stored as JSON, extensible by operators:

```json
[
  {
    "id": "fm_pump_bearing_wear",
    "equipment_type": "centrifugal_pump",
    "symptom_signature": {
      "vibration": "high",
      "temperature": "normal_or_slightly_elevated",
      "noise": "increased",
      "flow_rate": "normal",
      "power_consumption": "slightly_increased"
    },
    "root_cause": "bearing_wear",
    "description": "Progressive bearing degradation causes increased vibration at specific frequencies (BPFO/BPFI) before thermal effects manifest",
    "confidence_base": 0.75,
    "resolution_steps": [
      "Perform vibration spectrum analysis to identify bearing defect frequencies",
      "Check bearing lubrication condition",
      "Schedule bearing replacement if defect frequencies confirmed",
      "Verify shaft alignment after replacement"
    ],
    "prevention_steps": [
      "Implement vibration monitoring schedule (monthly)",
      "Follow lubrication intervals per manufacturer spec",
      "Track bearing runtime hours"
    ],
    "mtbf_hours": 25000,
    "severity": "high",
    "sources": ["ISO 10816-3", "SKF Bearing Handbook", "site_maintenance_records"]
  }
]
```

### Self-Healing Knowledge

When the engine encounters a symptom combination it can't match:

```python
# In diagnose():
if not matches or max(m.match_score for m in matches) < 0.3:
    self.log_knowledge_gap(
        equipment_type=equipment_type,
        query=f"Symptoms: {symptoms}",
        expected_type="failure_mode",
    )
    # Fall through to LLM general knowledge (with low confidence flag)
```

Knowledge gaps are exposed via API:
```
GET /api/layer2/knowledge-gaps/
→ [{"equipment_type": "centrifugal_pump", "query": "Symptoms: {vibration: high, pressure: low, temperature: normal}", "frequency": 3, "first_seen": "2026-02-05"}]
```

Engineers can resolve gaps by adding new `FailureMode` entries or uploading documents.

### Failure Modes

| Failure | Detection | Handling |
|---------|-----------|----------|
| Failure mode DB not found | FileNotFoundError on init | Log warning, operate in RAG-only mode (no symptom matching) |
| Symptom key mismatch (metric name drift) | No matching symptoms for known failure modes | Fuzzy match metric names, log mapping gap |
| Too many matches (>10 with score >0.3) | `len(high_score_matches) > 10` | Return top 5, log "overly broad symptom signature" |
| Knowledge gap table grows unbounded | `len(knowledge_gaps) > 1000` | Deduplicate by symptom signature, cap at 500 |

---

## 9. Voice as Control Surface

### Problem

Voice is currently fire-and-forget: user speaks → system processes → system responds. The user cannot interrupt, redirect, or cancel. If the system starts executing a wrong plan, the user must wait for it to finish.

### New Voice Events

**Location**: Add to `frontend/src/types/index.ts`:

```typescript
| { type: "VOICE_INTERRUPT"; reason: "stop" | "redirect" | "cancel" }
| { type: "PLAN_CANCEL"; plan_id: string }
| { type: "PLAN_REDIRECT"; plan_id: string; new_intent: string }
```

### Interrupt Detection

**Location**: `frontend/src/components/layer1/useVoicePipeline.ts`

```typescript
// Interrupt keywords detected during processing phase
const INTERRUPT_PATTERNS = {
  stop: /\b(stop|halt|never ?mind|cancel|forget it)\b/i,
  redirect: /\b(no,|wait,|actually,|instead,|not that|I meant)\b/i,
};

// In the VAD/transcription callback, during PROCESSING state:
function checkForInterrupt(partialTranscript: string): "stop" | "redirect" | null {
  if (INTERRUPT_PATTERNS.stop.test(partialTranscript)) return "stop";
  if (INTERRUPT_PATTERNS.redirect.test(partialTranscript)) return "redirect";
  return null;
}
```

### Interrupt Flow

```
1. User says: "show me pump 4 vibration"
2. System starts processing (PROCESSING state)
3. Plan compiled, execution begins

4a. User says "stop" during processing:
    → Frontend detects interrupt keyword in partial transcript
    → Emits VOICE_INTERRUPT {reason: "stop"}
    → useVoicePipeline aborts the pending fetch() to /api/layer2/orchestrate/
    → Sends POST /api/layer2/cancel/ {plan_id}
    → Backend PlanExecutor.cancel() marks all pending steps CANCELLED
    → Voice: "Okay, cancelled."

4b. User says "no, I meant pump 5" during processing:
    → Frontend detects "redirect" pattern
    → Emits VOICE_INTERRUPT {reason: "redirect"}
    → Waits for full redirect transcript
    → Cancels current plan
    → Submits new query: "pump 5 vibration"
    → Voice: "Got it, switching to pump 5."

4c. User says "no, compare instead" after response arrives:
    → Not an interrupt (response already delivered)
    → Normal follow-up query, processed through focus graph
    → Focus graph already has pump_004, user wants comparison
    → Builder.merge_comparison_target inferred from "compare"
```

### Backend Cancellation Endpoint

```python
@api_view(["POST"])
def cancel_plan(request):
    """
    POST /api/layer2/cancel/
    Body: {"plan_id": "abc123"}

    Cancels a running plan. The PlanExecutor checks a cancellation flag
    before starting each step. Already-running steps complete but their
    results are discarded.
    """
    plan_id = request.data.get("plan_id")
    orchestrator = get_orchestrator()
    cancelled = orchestrator.cancel_active_plan(plan_id)
    return Response({"cancelled": cancelled})
```

### Prosody Signals (Future-Compatible)

When PersonaPlex supports prosody extraction:

```python
@dataclass
class ProsodySignals:
    urgency: float         # 0.0-1.0, from pitch/rate
    hesitation: float      # 0.0-1.0, from pauses/fillers
    confidence: float      # 0.0-1.0, from volume/steadiness

# Integration: urgency > 0.7 → set intent.urgency = "high"
# Integration: hesitation > 0.5 → reduce voice response verbosity
# Integration: confidence < 0.3 → prompt for clarification
```

Until PersonaPlex supports prosody, these signals default to neutral (0.5).

### Failure Modes

| Failure | Detection | Handling |
|---------|-----------|----------|
| False interrupt (user said "stop" in a sentence) | "stop the vibration from increasing" | Use 2-word window: only trigger on "stop" when preceded by silence or sentence boundary |
| Cancel arrives after plan completed | `plan.status == COMPLETED` when cancel received | No-op, return `{"cancelled": false, "reason": "already_completed"}` |
| Redirect transcript is incomplete | User said "no, I meant—" and stopped | Wait 3s for completion, then ask "What would you like instead?" |
| Rapid fire interrupts | 3+ interrupts within 10s | "I'm having trouble understanding. Could you wait a moment and try again?" |

---

## 10. Explicit Failure-Mode UX

### Problem

When the system is uncertain, it either guesses (dangerous) or returns nothing (useless). An industrial operator needs to know: What do you know? What don't you know? What would you check next?

### New Widget: Uncertainty Panel

**Location**: `frontend/src/components/layer4/widgets/UncertaintyPanel.tsx`

```typescript
interface UncertaintyPanelProps {
  demoData: {
    overall_confidence: number;        // 0.0-1.0
    known_facts: Array<{
      statement: string;
      source: string;                  // "timeseries", "rag", "site_memory"
      freshness: string;               // "live", "5min", "1hr", "stale"
      confidence: number;
    }>;
    unknown_factors: Array<{
      description: string;
      why_unknown: string;             // "no_sensor", "data_stale", "no_knowledge"
      impact: "low" | "medium" | "high";
      check_action: string;            // What the operator could do
    }>;
    next_steps: Array<{
      action: string;
      automated: boolean;              // Can the system do this?
      priority: "low" | "medium" | "high";
    }>;
    constraint_violations: Array<{
      type: string;
      message: string;
      severity: "warning" | "error";
    }>;
  };
}
```

### Rendering Logic

The uncertainty panel renders three sections:

1. **Known** (green section): Facts with sources and freshness indicators
2. **Unknown** (amber section): Missing data with impact assessment
3. **Next Steps** (blue section): Actionable recommendations, some executable by the system

### Integration with Orchestrator

At the end of every query, the orchestrator builds an uncertainty assessment:

```python
def _build_uncertainty_assessment(self, plan: ExecutionPlan,
                                    constraint_eval: ConstraintEvaluation,
                                    reasoning_result: Optional[ReasoningResult]) -> dict:
    known_facts = []
    unknown_factors = []
    next_steps = []

    # Extract known facts from completed RETRIEVE steps
    for step in plan.steps:
        if step.type == StepType.RETRIEVE and step.status == StepStatus.COMPLETED:
            data = step.outputs
            known_facts.append({
                "statement": f"{data.get('metric', 'Value')} for {data.get('equipment', 'equipment')}: {data.get('value', 'N/A')} {data.get('unit', '')}",
                "source": data.get("source", "unknown"),
                "freshness": self._compute_freshness_label(data.get("timestamp")),
                "confidence": data.get("confidence", 0.5),
            })

    # Extract unknowns from failed RETRIEVE steps
    for step in plan.steps:
        if step.type == StepType.RETRIEVE and step.status == StepStatus.FAILED:
            unknown_factors.append({
                "description": step.description,
                "why_unknown": step.error or "Data retrieval failed",
                "impact": "medium",
                "check_action": f"Verify sensor connectivity for {step.inputs.get('equipment', 'equipment')}",
            })

    # Extract from reasoning result
    if reasoning_result:
        for fact in reasoning_result.known_facts:
            known_facts.append({"statement": fact, "source": "reasoning", "freshness": "computed", "confidence": reasoning_result.confidence})
        for unknown in reasoning_result.unknown_factors:
            unknown_factors.append({"description": unknown, "why_unknown": "Not enough data to determine", "impact": "high", "check_action": ""})
        for check in reasoning_result.recommended_checks:
            next_steps.append({"action": check, "automated": False, "priority": "medium"})

    # Add constraint violations
    constraint_violations = []
    for v in constraint_eval.violations:
        constraint_violations.append({"type": v.type.value, "message": v.message, "severity": "warning" if v.action_on_violation != ViolationAction.REFUSE else "error"})

    overall_confidence = self._compute_overall_confidence(known_facts, unknown_factors, constraint_eval)

    return {
        "overall_confidence": overall_confidence,
        "known_facts": known_facts,
        "unknown_factors": unknown_factors,
        "next_steps": next_steps,
        "constraint_violations": constraint_violations,
    }
```

### When to Show

The uncertainty panel is added to the widget plan automatically when:
1. `overall_confidence < 0.6` — always show
2. `len(unknown_factors) >= 2` — multiple unknowns warrant visibility
3. `len(constraint_violations) >= 1` — constraint violated
4. User explicitly asks "how confident are you?" or "what don't you know?"

It is NOT shown when:
1. `overall_confidence > 0.85` and no constraint violations — high confidence, clean data
2. Query is conversational (greeting, small talk)
3. Query is out_of_scope

### Voice Response Integration

When uncertainty is high, the voice response reflects it:

```python
if overall_confidence < 0.4:
    voice_prefix = "I'm not very confident in this answer. "
elif overall_confidence < 0.6:
    voice_prefix = "Based on what I can see, but with some gaps — "
else:
    voice_prefix = ""

# Always mention unknowns if present
if unknown_factors:
    voice_suffix = f" I should note that I don't have data on: {', '.join(u['description'] for u in unknown_factors[:2])}."
else:
    voice_suffix = ""
```

### Failure Modes

| Failure | Detection | Handling |
|---------|-----------|----------|
| Uncertainty assessment takes >500ms | Timer | Skip assessment, don't add panel (better to be fast than meta-uncertain) |
| All facts have low confidence (<0.3) | `all(f['confidence'] < 0.3 for f in known_facts)` | Overall confidence → 0.1, voice says "I don't have reliable data for this right now" |
| Unknown factors > 10 | `len(unknown_factors) > 10` | Truncate to top 5 by impact, note "and {N} more unknowns" |

---

## 11. E2E Test Specifications

### Test Infrastructure

All tests live in `frontend/e2e/tests/` (Playwright) and `backend/tests/` (pytest).

**Test Naming Convention**: `{upgrade_number}_{feature}_{scenario}.spec.ts`

**Base URL**: `http://localhost:3100` (frontend), `http://localhost:8100` (backend)

**Mock Policy**:
- Frontend E2E: Live frontend, mocked backend responses (via page.route interceptors)
- Backend integration: Live backend, live PostgreSQL, live Ollama (if available, else skip)
- RL regression: Live scorer, synthetic experiences

---

### Upgrade 1: Semantic Focus Graph Tests

**File**: `frontend/e2e/tests/01-focus-graph.spec.ts`

| Test ID | Name | Intent | Pass Criteria | Mock |
|---------|------|--------|--------------|------|
| FG-01 | Graph created on interactive enter | Enter interactive mode on pump_004 KPI | `context_update.focus_graph` contains node `equipment:pump_004` as root | Backend response mocked with focus_graph in context_update |
| FG-02 | Graph persists across follow-ups | Send 3 follow-up queries in interactive mode | Each response's `focus_graph.version` increments | Sequential mocked responses with incrementing version |
| FG-03 | Pronoun resolves to root equipment | Query "is it running?" with active graph | Backend receives focus_graph, response references pump_004 | Intercept POST to verify focus_graph in request body |
| FG-04 | Comparison adds second equipment node | Query "compare with pump 5" | Graph has 2 equipment nodes + COMPARED_WITH edge | Mocked response with 2 nodes |
| FG-05 | Graph cleared on interactive exit | Exit interactive mode | Next query's request body has no focus_graph | Intercept next POST, verify no focus_graph |
| FG-06 | Graph survives empty widget response | Backend returns 0 widgets but valid graph | Graph still present in Layer2Service context | Mocked empty-widget response with focus_graph |
| FG-07 | Corrupt graph JSON handled gracefully | Inject malformed focus_graph in context | System creates fresh graph, no crash | Modify context before next query |

**File**: `backend/tests/test_focus_graph.py`

| Test ID | Name | Pass Criteria |
|---------|------|--------------|
| FG-B01 | SemanticFocusGraph.add_node creates node | Node appears in graph.nodes dict |
| FG-B02 | Duplicate node increments reference_count | reference_count == 2 after second add |
| FG-B03 | add_edge rejects missing nodes | Raises ValueError |
| FG-B04 | to_dict/from_dict round-trips | Deserialized graph equals original |
| FG-B05 | to_prompt_context truncates to max_nodes | Output has ≤ max_nodes lines |
| FG-B06 | get_neighbors returns correct nodes | Correct set for known graph topology |
| FG-B07 | FocusGraphBuilder.resolve_pronoun returns root | With graph containing root, returns it |
| FG-B08 | FocusGraphBuilder.resolve_pronoun returns None when ambiguous | Two equipment nodes, same reference_count |

---

### Upgrade 2: Cross-Widget Reasoning Tests

**File**: `frontend/e2e/tests/02-reasoning.spec.ts`

| Test ID | Name | Pass Criteria | Mock |
|---------|------|--------------|------|
| RX-01 | "Why" query renders diagnostic panel | Dashboard includes `diagnostic-panel` widget scenario | Mocked response with diagnostic panel widget |
| RX-02 | Hypotheses displayed with confidence bars | Panel shows ≥1 hypothesis with score | DOM check for confidence bar elements |
| RX-03 | "Unknown factors" section renders | When unknowns present, amber section visible | Mocked data with unknown_factors |
| RX-04 | Non-causal query does NOT show diagnostic panel | "show pump 4 power" → no diagnostic panel | Mocked response without diagnostic widgets |

**File**: `backend/tests/test_reasoning_engine.py`

| Test ID | Name | Pass Criteria |
|---------|------|--------------|
| RX-B01 | Pattern match: vibration high + temp normal | Returns ≥1 hypothesis with "bearing" in statement |
| RX-B02 | No pattern match → empty list | Novel symptom combination → [] |
| RX-B03 | Contradicting symptoms reduce confidence | Hypothesis confidence < base confidence |
| RX-B04 | Missing symptoms tracked correctly | SymptomMatch.missing_symptoms populated |
| RX-B05 | LLM synthesis fallback works | When patterns + stats empty, LLM produces result |
| RX-B06 | LLM returns malformed JSON → graceful fallback | Returns pattern results only |

---

### Upgrade 3: Time-Travel Tests

**File**: `frontend/e2e/tests/03-time-travel.spec.ts`

| Test ID | Name | Pass Criteria | Mock |
|---------|------|--------------|------|
| TT-01 | "Before the anomaly" resolves to time window | Response contains snapshot-comparison widget | Mocked response with temporal data |
| TT-02 | Snapshot comparison renders two columns | Widget shows baseline and target side by side | DOM check for split-panel layout |
| TT-03 | Delta indicators show significance | "SIGNIFICANT" badge on changed metrics | DOM check for significance badges |
| TT-04 | "Last week vs this week" creates comparison | Two time windows in response | Mocked response with week-over-week data |

**File**: `backend/tests/test_time_context.py`

| Test ID | Name | Pass Criteria |
|---------|------|--------------|
| TT-B01 | TimeResolver parses "last 24 hours" | Returns TimeWindow with correct start/end |
| TT-B02 | TimeResolver parses "before the anomaly" with focus graph | Returns window ending at anomaly timestamp |
| TT-B03 | TimeResolver returns None for unresolvable | "before the thing" with empty graph → None |
| TT-B04 | SnapshotEngine computes delta correctly | 78% change computed for 1.8→3.2 |
| TT-B05 | Statistical significance computed | z-score > 2 → "significant" |

---

### Upgrade 4: Plan Compiler Tests

**File**: `backend/tests/test_plan_compiler.py`

| Test ID | Name | Pass Criteria |
|---------|------|--------------|
| PC-01 | Simple query compiles to RETRIEVE → SELECT → COLLECT → RESPOND | 4 steps, correct types |
| PC-02 | "Why" query includes REASON step | REASON step present with correct dependencies |
| PC-03 | Time comparison includes RESOLVE_TIME step | RESOLVE_TIME step present |
| PC-04 | Under-specified query creates ASK_USER step | Plan status == BLOCKED |
| PC-05 | "Compare" with 1 entity and no context → ASK_USER | Blocked reason mentions "compare with what" |
| PC-06 | Dependencies form DAG (no cycles) | Topological sort succeeds |
| PC-07 | Budget exceeded → plan fails gracefully | Status == FAILED after timeout |
| PC-08 | Cancel stops pending steps | All PENDING → CANCELLED |
| PC-09 | Step failure doesn't block independent siblings | Parallel steps execute despite one failure |
| PC-10 | Clarification query (no entities, no context) | Detects ambiguity, returns ASK_USER |

**File**: `frontend/e2e/tests/04-plan-execution.spec.ts`

| Test ID | Name | Pass Criteria | Mock |
|---------|------|--------------|------|
| PE-01 | Clarification prompt shown when plan blocked | Context bar shows "Which equipment?" | Mock blocked response |
| PE-02 | After clarification, plan proceeds | User answers, widgets appear | Sequential mocked responses |
| PE-03 | Cancel via voice interrupt cancels plan | Interrupt "stop" during processing → plan cancelled | Intercept cancel endpoint |

---

### Upgrade 5: Constraint Tests

**File**: `backend/tests/test_constraints.py`

| Test ID | Name | Pass Criteria |
|---------|------|--------------|
| CS-01 | Fresh data satisfies freshness constraint | constraint.satisfied == True |
| CS-02 | Stale data violates freshness constraint | constraint.satisfied == False, message populated |
| CS-03 | Safety constraint refuses on violation | evaluation.action == REFUSE |
| CS-04 | Multiple violations use most severe action | REFUSE > QUALIFY > WARN |
| CS-05 | Qualification text prepended to voice response | voice_response starts with "Note:" |
| CS-06 | Low confidence triggers QUALIFY | confidence=0.3 → qualification |
| CS-07 | Very low confidence triggers REFUSE | confidence=0.15 → refusal |

**File**: `frontend/e2e/tests/05-constraints.spec.ts`

| Test ID | Name | Pass Criteria | Mock |
|---------|------|--------------|------|
| CS-F01 | Stale data badge shown on widget | Yellow "Stale data" badge visible | Mocked response with constraint_violations |
| CS-F02 | Refusal shows explanation instead of widgets | No widgets, voice explains why | Mocked refusal response |
| CS-F03 | Warning badge visible on affected widgets | Widget border changes to yellow | DOM class check |

---

### Upgrade 6: RL Decision Rewards Tests

**File**: `backend/tests/test_rl_decisions.py`

| Test ID | Name | Pass Criteria |
|---------|------|--------------|
| RL-01 | Quick exit produces negative reward | time_to_exit_ms=2000 → reward < 0 |
| RL-02 | Re-ask produces negative reward | re_ask_within_30s=True → reward < -0.3 |
| RL-03 | Operator action produces positive reward | operator_created_work_order=True → reward > 0.3 |
| RL-04 | Ignored widgets penalized | 3 widgets ignored → penalty applied |
| RL-05 | Constraint violation penalized | constraints_violated=2 → penalty |
| RL-06 | Scorer input dim is 819 | LowRankScorer.input_dim == 819 |
| RL-07 | Scorer trains without error on new dim | Train step completes, loss decreases |
| RL-08 | Scorer rollback on divergence | Loss > 2× baseline → reverts to checkpoint |

---

### Upgrade 7: Memory Tests

**File**: `backend/tests/test_memory.py`

| Test ID | Name | Pass Criteria |
|---------|------|--------------|
| MEM-01 | Session memory stores and recalls | recall() returns stored entry |
| MEM-02 | Session memory cleared on forget_session | recall() returns None after forget |
| MEM-03 | Operator memory persists across sessions | Store in session A, recall in session B |
| MEM-04 | Site memory shared across operators | Store as user A, recall as user B |
| MEM-05 | Recall searches tiers in order | Session → Operator → Site priority |
| MEM-06 | Expired entries not returned | Entry with TTL expired → recall() returns None |
| MEM-07 | Promote moves entry between tiers | Session entry promoted to Operator |
| MEM-08 | Memory influences focus graph | Site memory for pump_004 → anomaly nodes added |
| MEM-09 | Memory size bounded | >100MB → pruning triggered |

---

### Upgrade 8: Causal Knowledge Tests

**File**: `backend/tests/test_causal_knowledge.py`

| Test ID | Name | Pass Criteria |
|---------|------|--------------|
| CK-01 | Diagnose with matching symptoms | Returns ≥1 SymptomMatch with score > 0.5 |
| CK-02 | No matching symptoms → knowledge gap logged | KnowledgeGap added to engine |
| CK-03 | Contradicting symptom reduces score | Score lower than without contradiction |
| CK-04 | Missing failure mode DB → RAG-only mode | No crash, results from RAG |
| CK-05 | Knowledge gap deduplication | Same symptoms twice → 1 gap with frequency=2 |
| CK-06 | get_pending_gaps returns ordered by frequency | Most frequent first |

---

### Upgrade 9: Voice Control Tests

**File**: `frontend/e2e/tests/09-voice-control.spec.ts`

| Test ID | Name | Pass Criteria | Mock |
|---------|------|--------------|------|
| VC-01 | "stop" during processing cancels | Processing state → cancel sent → "Cancelled" spoken | Intercept cancel endpoint |
| VC-02 | "no, I meant pump 5" redirects | First query cancelled, second query sent with "pump 5" | Sequential intercepted requests |
| VC-03 | "stop" in sentence not falsely triggered | "stop the motor" → not interrupted | Verify no cancel request |
| VC-04 | Rapid interrupts handled gracefully | 3 interrupts in 10s → "try again" message | Track interrupt count |

---

### Upgrade 10: Failure UX Tests

**File**: `frontend/e2e/tests/10-failure-ux.spec.ts`

| Test ID | Name | Pass Criteria | Mock |
|---------|------|--------------|------|
| FU-01 | Uncertainty panel renders when confidence < 0.6 | `uncertainty-panel` widget present in DOM | Mocked response with low confidence |
| FU-02 | Known facts section shows green indicators | Green badges visible | DOM check |
| FU-03 | Unknown factors section shows amber indicators | Amber section visible | DOM check with unknowns |
| FU-04 | Next steps rendered as checklist | Checklist items visible | DOM check |
| FU-05 | High confidence → no uncertainty panel | No uncertainty panel in DOM | Mocked response with confidence=0.9 |
| FU-06 | Constraint violation shown in panel | Red badge for violated constraint | Mocked response with violations |
| FU-07 | "How confident are you?" forces panel | Uncertainty panel appears regardless of score | Mocked response triggered by specific query |

---

### Regression Tests

**File**: `frontend/e2e/tests/regression-interactive-mode.spec.ts`

These ensure the existing 16 interactive mode tests still pass after all upgrades:

| Test ID | Name | Pass Criteria |
|---------|------|--------------|
| REG-01 | All 13 passing interactive mode tests still pass | 13/13 pass |
| REG-02 | Default layout renders without upgrades active | Widgets appear, no focus graph, no constraints |
| REG-03 | Non-interactive queries bypass plan compiler | Direct intent→widget mapping, <3s latency |
| REG-04 | RL scorer still functions with old 787-dim input | Backward compat: pad 787→819 with zeros |
| REG-05 | Empty focus_graph in request handled gracefully | No crash, normal processing |

---

## Implementation Priority

| Phase | Upgrades | Rationale |
|-------|----------|-----------|
| **Phase 1** (Foundation) | 1 (Focus Graph), 4 (Plan Compiler), 5 (Constraints) | Core infrastructure that everything else builds on |
| **Phase 2** (Intelligence) | 2 (Reasoning), 8 (Causal Knowledge), 10 (Failure UX) | Diagnostic capability + honest uncertainty |
| **Phase 3** (Time + Memory) | 3 (Time-Travel), 7 (Memory Stratification) | Temporal reasoning + persistent knowledge |
| **Phase 4** (Control + RL) | 6 (RL Decisions), 9 (Voice Control) | Feedback loop + real-time control |

Each phase is independently deployable. Phase 1 produces immediate value. Phase 4 requires all prior phases.

---

## File Manifest

### New Backend Files (10)
```
backend/layer2/focus_graph.py            # Upgrade 1: SemanticFocusGraph + FocusNode + FocusEdge
backend/layer2/focus_graph_builder.py    # Upgrade 1: FocusGraphBuilder
backend/layer2/reasoning_engine.py       # Upgrade 2: CrossWidgetReasoningEngine
backend/layer2/time_context.py           # Upgrade 3: TimeResolver + SnapshotEngine
backend/layer2/plan_compiler.py          # Upgrade 4: PlanCompiler + PlanExecutor
backend/layer2/constraints.py            # Upgrade 5: Constraint + ConstraintEvaluation
backend/layer2/memory.py                 # Upgrade 7: MemoryStore + MemoryEntry
backend/layer2/causal_knowledge.py       # Upgrade 8: CausalKnowledgeEngine + FailureMode
backend/data/failure_modes.json          # Upgrade 8: Seeded failure mode database
backend/tests/test_*.py                  # All backend tests (8 files)
```

### Modified Backend Files (5)
```
backend/layer2/orchestrator.py           # Upgrades 1,4,5,10: Focus graph, plan compiler, constraints, uncertainty
backend/layer2/intent_parser.py          # Upgrade 1: Graph-based pronoun resolution
backend/layer2/widget_selector.py        # Upgrades 1,3: Graph context, time comparison
backend/rl/experience_buffer.py          # Upgrade 6: DecisionExperience fields
backend/rl/reward_signals.py             # Upgrade 6: DecisionRewardAggregator
backend/rl/lora_scorer.py               # Upgrade 6: Input dim 787→819
backend/layer2/views.py                  # Upgrades 4,9: Cancel endpoint, knowledge gaps endpoint
```

### New Frontend Files (3)
```
frontend/src/components/layer4/widgets/DiagnosticPanel.tsx      # Upgrade 2
frontend/src/components/layer4/widgets/SnapshotComparison.tsx   # Upgrade 3
frontend/src/components/layer4/widgets/UncertaintyPanel.tsx     # Upgrade 10
frontend/e2e/tests/01-focus-graph.spec.ts                       # Tests
frontend/e2e/tests/02-reasoning.spec.ts
frontend/e2e/tests/03-time-travel.spec.ts
frontend/e2e/tests/04-plan-execution.spec.ts
frontend/e2e/tests/05-constraints.spec.ts
frontend/e2e/tests/09-voice-control.spec.ts
frontend/e2e/tests/10-failure-ux.spec.ts
frontend/e2e/tests/regression-interactive-mode.spec.ts
```

### Modified Frontend Files (3)
```
frontend/src/types/index.ts                             # Upgrades 1,9: New event types
frontend/src/components/layer1/useVoicePipeline.ts      # Upgrade 9: Interrupt detection
frontend/src/components/layer4/widgetRegistry.ts        # Upgrades 2,3,10: Register 3 new widgets
```

---

**Total**: 10 new backend files, 5 modified backend files, 11 new frontend files, 3 modified frontend files, 83 tests across 16 test files.
