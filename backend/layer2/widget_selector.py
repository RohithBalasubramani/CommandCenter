"""
LLM-based Widget Selector for Pipeline v2.

Replaces the 15-branch if-elif chain in orchestrator.py (~1000 lines) with a
single LLM call that sees the full widget catalog and selects appropriate
widgets based on the parsed intent.

The LLM outputs a structured widget plan including:
- Which widgets to show
- What size each should be
- Why each is relevant
- What data each widget needs (drives Stage 3 data collection)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from layer2.rag_pipeline import get_rag_pipeline
from layer2.widget_catalog import (
    WIDGET_CATALOG,
    CATALOG_BY_SCENARIO,
    VALID_SCENARIOS,
    get_catalog_prompt_text,
)
from layer2.intent_parser import ParsedIntent
from rl.prompt_evolver import get_prompt_evolver

logger = logging.getLogger(__name__)

MAX_HEIGHT_UNITS = 72  # scrollable dashboard budget — generous for 20+ widget dashboards
MAX_WIDGETS = 24
MAX_KPIS = 6

# Scenarios that should NEVER be selected by the LLM
BANNED_SCENARIOS = {"helpview", "pulseview"}
# Maximum instances of the same scenario type
MAX_SAME_SCENARIO = 3

# Safety-critical scenarios that must maintain minimum visibility
# These should never be suppressed by RL reranking (e.g., alerts, safety KPIs)
SAFETY_CRITICAL_SCENARIOS = {"alerts", "kpi"}
SAFETY_RELEVANCE_FLOOR = 0.7   # Minimum relevance for safety widgets
SAFETY_MAX_POSITION = 3         # Safety widgets must appear within top N (after hero)


SYSTEM_PROMPT = """You are a dashboard widget selector for an industrial operations command center.
You select the best combination of widgets to build an informative dashboard for the user's query.
You MUST respond with ONLY valid JSON, no explanation or markdown."""

SELECT_PROMPT_TEMPLATE = """Select widgets for an industrial operations dashboard.

RULES:
1. The FIRST widget must directly answer the user's question (the "hero" — use hero size)
2. Subsequent widgets provide supporting context with progressively more detail
3. Order widgets by RELEVANCE to the query — most important first. KPIs do NOT need to be first.
4. Aim for 6-8 widgets to create a full, information-dense dashboard. Use varied sizes.
5. Maximum {max_widgets} widgets total, maximum total height: {max_height} units
6. Maximum {max_kpis} KPI widgets
7. NEVER use "helpview" or "pulseview" scenarios
8. Use at LEAST 3 different scenario types. Maximum 2 widgets of the same scenario.
9. Mix sizes: use hero (100% width), expanded (50%), normal (33%), compact (25%)
10. Only use scenarios from the catalog below
11. Use ONLY ONE hero-sized widget (the primary answer). Extra heroes will be demoted to expanded.
12. flow-sankey, matrix-heatmap, edgedevicepanel work best at hero or expanded size — do NOT use compact/normal for them

DOMAIN AFFINITY (prefer these widget types for these topics):
- Energy queries → flow-sankey, distribution, trend, trends-cumulative
- Maintenance queries → timeline, eventlogstream, alerts, category-bar
- Comparison queries → comparison, trend-multi-line, matrix-heatmap
- Health/status queries → matrix-heatmap, edgedevicepanel, kpi, alerts
- Production queries → category-bar, composition, trend, kpi
- Power quality → trend-multi-line, distribution, matrix-heatmap

The "why" field is shown to the user as a description on each widget card. Write it as a clear
1-2 sentence explanation of what the widget shows and why it matters for their question.
Example: "Shows the energy consumption trend over the last 24 hours, highlighting peak usage periods."

Available widgets:
{catalog}

DATA AVAILABLE FOR MENTIONED ENTITIES:
{data_summary}

USER HISTORY:
{user_memory}

DASHBOARD STORY:
Think step by step about what this dashboard should communicate:
1. What is the user REALLY trying to understand — not just the literal question?
2. Given the data that EXISTS above, which metrics show meaningful insights?
3. If entities have alerts or maintenance issues, should those be surfaced?
4. What would a plant operator/engineer want to see alongside the primary answer?
5. Consider what the user has asked before — are they investigating a pattern?

For each widget, write a specific "why" explaining its role in the story.
Do NOT add widgets just to fill space — every widget should serve the narrative.

User query: "{query}"
Intent type: {intent_type}
Domains: {domains}
Entities: {entities}
Primary characteristic: {primary_char}
Secondary characteristics: {secondary_chars}

Select widgets as JSON:
{{
  "heading": "<short dashboard title>",
  "widgets": [
    {{
      "scenario": "<scenario name from catalog>",
      "size": "compact|normal|expanded|hero",
      "relevance": 0.0-1.0,
      "why": "<1-2 sentence user-facing description>",
      "data_request": {{
        "query": "<natural language description of what data this widget needs>",
        "entities": ["<entity1>", ...],
        "metric": "<what metric to show>",
        "collections": ["equipment", "alerts", "maintenance", "operational_docs", "work_orders", "shift_logs"]
      }}
    }}
  ]
}}

JSON:"""

# ══════════════════════════════════════════════════════════════════════════════
# FAST PROMPT - Optimized for speed (2s vs 15s) while maintaining 100% accuracy
# Key optimization: No "why" field generation - use templates instead
# ══════════════════════════════════════════════════════════════════════════════

FAST_SELECT_PROMPT = '''Select widgets for this industrial operations query.

## WIDGET CATALOG
{catalog}

## QUERY
"{query}"
Intent: {intent_type} | Domains: {domains} | Entities: {entities}
Primary focus: {primary_char} | Also relevant: {secondary_chars}

## DATA AVAILABLE
{data_summary}

## DOMAIN AFFINITY (prefer these widget types):
- Comparison queries → comparison (hero), trend-multi-line, kpi for each entity
- Energy/power queries → trend (hero), distribution, kpi, trends-cumulative
- Alerts queries → alerts (hero), kpi, trend, timeline
- Maintenance queries → timeline (hero), eventlogstream, alerts, category-bar
- Health/status single device → edgedevicepanel (hero), trend, alerts, kpi
- Health/status overview → matrix-heatmap (hero), category-bar, alerts, kpi
- HVAC/chiller queries → trend (hero), comparison, kpi, alerts
- Top consumers → category-bar (hero), distribution, kpi, trend

## SIZING RULES
hero: First widget only (primary answer). Use: trend, trend-multi-line, comparison, category-bar, matrix-heatmap, edgedevicepanel, timeline, flow-sankey, eventlogstream
expanded: Secondary detail widgets. Use: trend, trend-multi-line, comparison, distribution, alerts, timeline, composition
normal: Supporting context. Use: alerts, distribution, kpi
compact: Quick-glance metrics only. Use: kpi

## RULES
1. Select EXACTLY {widget_count} widgets — use ALL {widget_count} slots with DIVERSE widget types
2. First widget MUST be hero size — pick the scenario that BEST answers the query
3. Each widget MUST have a relevance score (0.0-1.0) reflecting how useful it is for THIS specific query
   - Hero: 0.90-0.98 | Direct supporting: 0.75-0.89 | Context: 0.55-0.74 | Nice-to-have: 0.40-0.54
4. Use EXACT scenario names from catalog
5. Max 2 KPI widgets per entity, each showing a DIFFERENT metric
6. Each widget MUST include data_request with query, metric, and entities from the query
7. MAXIMIZE widget type diversity — use DIFFERENT scenario types (trend, alerts, distribution, category-bar, composition, timeline, eventlogstream, etc.)
8. Operators need comprehensive dashboards — show KPIs AND trends AND alerts AND distributions, not just one type

## OUTPUT (JSON only, no explanation)
{{"heading": "<short dashboard title>", "widgets": [
  {{"scenario": "<name>", "size": "<size>", "relevance": <0.0-1.0>, "data_request": {{"query": "<what data>", "metric": "<metric_name>", "entities": ["{entity_hint}"]}}}}
]}}'''

# Template-based "why" descriptions - avoids slow LLM text generation
WHY_TEMPLATES = {
    "kpi": "Shows the current value and status of the key metric at a glance.",
    "trend": "Displays how this metric has changed over time to identify patterns.",
    "trend-multi-line": "Compares multiple metrics over time on the same chart.",
    "trends-cumulative": "Shows the running total accumulating over the time period.",
    "distribution": "Breaks down the data by category to show proportions.",
    "category-bar": "Ranks items by performance or consumption for quick comparison.",
    "comparison": "Side-by-side comparison of the requested entities.",
    "composition": "Shows what components make up the total value.",
    "flow-sankey": "Visualizes how energy or resources flow through the system.",
    "alerts": "Lists active alerts and warnings that need attention.",
    "timeline": "Shows maintenance events and scheduled activities over time.",
    "eventlogstream": "Live feed of recent events and system activity.",
    "matrix-heatmap": "Heat map showing status across multiple assets at once.",
    "edgedevicepanel": "Detailed panel showing device health and metrics.",
    "peopleview": "Overview of personnel assignments and workforce status.",
    "supplychainglobe": "Global view of supply chain and logistics status.",
}


def build_production_prompt(
    query: str,
    intent_type: str = "query",
    domains: str = "industrial",
    entities: str = "none",
    entity_hint: str = "",
    primary_char: str = "general query",
    secondary_chars: str = "none",
    data_summary: str = "No pre-fetched data available.",
    widget_count: int = 8,
    use_evolver: bool = True,
) -> tuple:
    """Build the EXACT (system_prompt, user_prompt, prompt_version) that the LLM receives.

    This is the single source of truth for prompt construction.
    Used by:
      - widget_selector._select_with_llm() during production inference
      - claude_teacher.py during trace capture for distillation
      - automated_runner.py for Claude vs Ollama comparison

    All three MUST produce identical prompts for the training data to be valid.

    Returns:
        (system_prompt, formatted_user_prompt, prompt_version)
    """
    catalog_text = get_catalog_prompt_text()

    # Select prompt template (evolver variant or static)
    prompt_version = "static"
    prompt_template = FAST_SELECT_PROMPT
    if use_evolver:
        try:
            evolver = get_prompt_evolver()
            prompt_template, prompt_version = evolver.get_prompt()
        except Exception:
            pass

    user_prompt = prompt_template.format(
        catalog=catalog_text,
        query=query,
        intent_type=intent_type,
        domains=domains,
        entities=entities,
        primary_char=primary_char,
        secondary_chars=secondary_chars,
        data_summary=data_summary,
        widget_count=widget_count,
        entity_hint=entity_hint,
    )

    return SYSTEM_PROMPT, user_prompt, prompt_version


@dataclass
class WidgetPlanItem:
    """Single widget in the plan."""
    scenario: str
    size: str = "normal"
    relevance: float = 0.8
    why: str = ""
    data_request: dict = field(default_factory=dict)
    height_units: int = 2


@dataclass
class WidgetPlan:
    """Complete widget plan from the selector."""
    heading: str = ""
    widgets: list = field(default_factory=list)  # list[WidgetPlanItem]
    total_height_units: int = 0
    select_method: str = "llm"  # "llm" or "fallback"
    prompt_version: str = ""    # Which prompt variant was used (for RL prompt optimization)


class WidgetSelector:
    """LLM-based widget selector with RL-based reranking."""

    def __init__(self):
        self._pipeline = None
        self._rl_scorer = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = get_rag_pipeline()
        return self._pipeline

    @property
    def rl_scorer(self):
        """Lazy-load the RL low-rank scorer for widget reranking."""
        if self._rl_scorer is None:
            try:
                if os.getenv("ENABLE_CONTINUOUS_RL", "true").lower() == "true":
                    from rl.lora_scorer import get_scorer
                    self._rl_scorer = get_scorer()
                    logger.info("RL low-rank scorer loaded for widget reranking")
            except Exception as e:
                logger.debug(f"RL scorer not available: {e}")
                self._rl_scorer = False  # Sentinel to avoid retrying
        return self._rl_scorer if self._rl_scorer is not False else None

    def select(self, intent: ParsedIntent, data_summary: str = "",
               user_context: str = "", widget_context: dict = None,
               focus_graph=None) -> WidgetPlan:
        """Select widgets using 8B LLM, with RL reranking and rule-based fallback."""
        # Try LLM first
        try:
            result = self._select_with_llm(intent, data_summary, user_context,
                                            widget_context=widget_context,
                                            focus_graph=focus_graph)
            if result is not None and result.widgets:
                # Apply RL score adjustments from low-rank scorer
                self._apply_rl_reranking(result, intent)
                return result
        except Exception as e:
            logger.warning(f"LLM widget selection failed: {e}")

        # Fallback to rule-based
        logger.info("Falling back to rule-based widget selection")
        return self._select_with_rules(intent)

    def _apply_rl_reranking(self, plan: "WidgetPlan", intent: ParsedIntent):
        """
        Apply RL-learned score adjustments to rerank widgets.

        The low-rank scorer outputs a [-1, 1] adjustment per scenario.
        We combine this with the LLM's relevance score to produce the
        final ordering. The first widget (hero) is never demoted.
        """
        scorer = self.rl_scorer
        if scorer is None or not plan.widgets:
            return

        try:
            transcript = intent.raw_text
            scenarios = [w.scenario for w in plan.widgets]

            # Build enriched query context from ParsedIntent
            from rl.lora_scorer import QueryContext
            ctx = QueryContext.from_parsed_intent(intent)
            ctx.num_candidate_widgets = len(scenarios)

            adjustments = scorer.score_widgets(transcript, scenarios, ctx=ctx)

            # Apply adjustments to relevance scores (keep hero in place)
            for i, widget in enumerate(plan.widgets):
                adj = adjustments.get(widget.scenario, 0.0)
                # Blend: 50% LLM relevance + 50% RL adjustment
                # (RL needs enough influence to meaningfully rerank)
                new_relevance = max(0.0, min(1.0,
                    widget.relevance + 0.5 * adj
                ))

                # Enforce safety floor for critical widgets
                if widget.scenario in SAFETY_CRITICAL_SCENARIOS:
                    if new_relevance < SAFETY_RELEVANCE_FLOOR:
                        logger.warning(
                            f"[rl-safety] RL tried to downrank safety widget "
                            f"'{widget.scenario}' to {new_relevance:.2f} "
                            f"(adj={adj:.2f}), clamping to floor {SAFETY_RELEVANCE_FLOOR}"
                        )
                        new_relevance = max(new_relevance, SAFETY_RELEVANCE_FLOOR)

                widget.relevance = new_relevance

            # Re-sort by adjusted relevance (but keep hero/first widget fixed)
            if len(plan.widgets) > 2:
                hero = plan.widgets[0]
                rest = sorted(plan.widgets[1:], key=lambda w: w.relevance, reverse=True)

                # Enforce position constraint: safety widgets within top SAFETY_MAX_POSITION
                for idx, w in list(enumerate(rest)):
                    if (w.scenario in SAFETY_CRITICAL_SCENARIOS
                            and idx >= SAFETY_MAX_POSITION):
                        rest.remove(w)
                        insert_at = min(SAFETY_MAX_POSITION - 1, len(rest))
                        rest.insert(insert_at, w)
                        logger.warning(
                            f"[rl-safety] Promoted safety widget '{w.scenario}' "
                            f"from position {idx + 2} to {insert_at + 2}"
                        )

                plan.widgets = [hero] + rest

        except Exception as e:
            logger.debug(f"RL reranking skipped: {e}")

    def _compute_widget_count(self, intent: ParsedIntent) -> int:
        """Determine optimal widget count based on query complexity.

        Comprehensive dashboards: operators benefit from seeing all relevant
        data at once (KPIs, trends, alerts, distributions, etc.).
        """
        entities = intent.entities.get("devices", [])
        primary = intent.primary_characteristic
        secondary = intent.secondary_characteristics

        # Base count — generous to show comprehensive dashboards
        if len(entities) == 0:
            # No specific entities — broad overview: matrix, bars, alerts, KPIs, trends
            base = 10
        elif len(entities) == 1:
            # Single entity — show all relevant aspects: KPI, trend, alerts, edge panel, distribution, etc.
            base = 8
        elif len(entities) == 2:
            # Comparison — comparison + individual views + aggregates
            base = 10
        else:
            # Multiple entities — comprehensive multi-entity dashboard
            base = 12

        # Adjust for characteristics
        if primary == "comparison":
            base = max(base, 10)
        elif primary == "health_status" and len(entities) == 0:
            base = max(base, 12)  # Overview needs matrix + bars + alerts + KPIs + trends
        elif primary in ("alerts", "maintenance"):
            base = max(base, 10)  # alerts list + timeline + context + trends + KPIs

        # Add 1 for each relevant secondary characteristic (up to +3)
        base += min(len(secondary), 3)

        return min(base, MAX_WIDGETS)

    def _select_with_llm(self, intent: ParsedIntent, data_summary: str,
                          user_context: str, widget_context: dict = None,
                          focus_graph=None) -> Optional[WidgetPlan]:
        """Select widgets using LLM with context-aware prompt.

        Passes full intent context (entities, characteristics, domains) to the
        LLM so it can make informed, query-specific widget choices.

        The "why" field is populated from templates in post-processing.
        Uses quality (70B) model if WIDGET_SELECT_QUALITY=1.
        """
        use_quality = os.getenv("WIDGET_SELECT_QUALITY", "0") == "1"
        llm = self.pipeline.llm_quality if use_quality else self.pipeline.llm_fast
        logger.info(f"Widget selection using {'quality (70B)' if use_quality else 'fast (8B)'} model")
        catalog_text = get_catalog_prompt_text()

        # Compute adaptive widget count
        widget_count = self._compute_widget_count(intent)

        # In interactive mode, always provide a rich dashboard (min 8 widgets)
        if widget_context:
            widget_count = max(widget_count, 8)

        # Build entity and characteristic strings for the prompt
        entities = intent.entities.get("devices", [])
        entity_hint = entities[0] if entities else ""
        entities_str = json.dumps(entities) if entities else "none"
        primary_char = intent.primary_characteristic or "general query"
        secondary_chars = ", ".join(intent.secondary_characteristics[:3]) if intent.secondary_characteristics else "none"

        # Build the production prompt (shared with claude_teacher for distillation parity)
        _, prompt, prompt_version = build_production_prompt(
            query=intent.raw_text,
            intent_type=intent.type,
            domains=", ".join(intent.domains) if intent.domains else "industrial",
            entities=entities_str,
            entity_hint=entity_hint,
            primary_char=primary_char,
            secondary_chars=secondary_chars,
            data_summary=data_summary or "No pre-fetched data available.",
            widget_count=widget_count,
        )

        # Append interactive context so LLM builds a contextual dashboard
        if widget_context:
            prompt += self._build_interactive_prompt_section(widget_context)

        # Append focus graph context for entity-aware widget selection
        if focus_graph and hasattr(focus_graph, 'to_prompt_context'):
            graph_ctx = focus_graph.to_prompt_context()
            if graph_ctx:
                prompt += f"\n\n## SEMANTIC FOCUS\n{graph_ctx}"

        logger.info(f"Widget selection: {widget_count} widgets for '{intent.raw_text[:60]}' "
                     f"(primary={primary_char}, entities={entities})"
                     f"{' [INTERACTIVE]' if widget_context else ''}")

        data = llm.generate_json(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,  # Slight variance for diversity while staying focused
            max_tokens=2048,
            cache_key=f"widget_select:{intent.raw_text}",
        )

        if data is None:
            return None

        plan = self._validate_and_build_plan(data, method="llm")
        plan.prompt_version = prompt_version

        # Post-processing: ensure parsed entities flow into widget data_requests
        if entities:
            self._propagate_entities(plan, entities)

        return plan

    def _propagate_entities(self, plan: WidgetPlan, entities: list[str]):
        """Ensure parsed entities appear in widget data_requests.

        When the LLM omits entities in data_request, inject them from the
        parsed intent so downstream data_collector can resolve them.
        """
        for widget in plan.widgets:
            dr = widget.data_request
            if not dr:
                widget.data_request = {"entities": entities[:2], "query": "", "metric": ""}
                continue

            existing_entities = dr.get("entities", [])
            if not existing_entities:
                # Inject entities based on widget type
                if widget.scenario in ("comparison", "trend-multi-line"):
                    dr["entities"] = entities[:4]
                elif widget.scenario in ("kpi", "trend", "edgedevicepanel"):
                    dr["entities"] = entities[:1]
                else:
                    dr["entities"] = entities[:2]

    def _validate_and_build_plan(self, data: dict, method: str = "llm") -> WidgetPlan:
        """Validate LLM output and build a WidgetPlan."""
        heading = data.get("heading", "Dashboard")
        raw_widgets = data.get("widgets", [])

        plan = WidgetPlan(heading=heading, select_method=method)
        budget = MAX_HEIGHT_UNITS
        kpi_count = 0
        scenario_counts: dict[str, int] = {}

        for w in raw_widgets:
            # F6 Fix: Normalize scenario name to lowercase to handle LLM case variations
            # (e.g., "edgeDevicePanel" → "edgedevicepanel")
            scenario = w.get("scenario", "").lower()
            if scenario not in VALID_SCENARIOS:
                logger.warning(f"Widget selector returned unknown scenario: {scenario}")
                continue

            # Hard-ban certain scenarios
            if scenario in BANNED_SCENARIOS:
                logger.info(f"Skipping banned scenario: {scenario}")
                continue

            # Enforce same-scenario cap
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
            if scenario_counts[scenario] > MAX_SAME_SCENARIO:
                continue

            # Enforce KPI cap
            if scenario == "kpi":
                kpi_count += 1
                if kpi_count > MAX_KPIS:
                    continue

            catalog_entry = CATALOG_BY_SCENARIO[scenario]
            height = catalog_entry["height_units"]

            # Enforce height budget
            if budget - height < 0:
                continue
            budget -= height

            # Validate size against widget-specific allowed sizes
            size = w.get("size", "normal")
            allowed_sizes = set(catalog_entry.get("sizes", ["normal"]))
            if size not in allowed_sizes:
                # Use largest allowed size (prefer hero > expanded > normal > compact)
                size_priority = ["hero", "expanded", "normal", "compact"]
                size = next((s for s in size_priority if s in allowed_sizes), "normal")
                logger.debug(f"Size correction: {scenario} requested {w.get('size')} → {size}")

            # Use template-based "why" if LLM didn't provide one (fast prompt optimization)
            why = w.get("why") or WHY_TEMPLATES.get(scenario, "")

            item = WidgetPlanItem(
                scenario=scenario,
                size=size,
                relevance=min(1.0, max(0.0, w.get("relevance", 0.8))),
                why=why,
                data_request=w.get("data_request", {}),
                height_units=height,
            )
            plan.widgets.append(item)

            # Enforce max widgets
            if len(plan.widgets) >= MAX_WIDGETS:
                break

        plan.total_height_units = MAX_HEIGHT_UNITS - budget
        return plan

    # ── Rule-based fallback (simplified version of old if-elif chain) ──

    def _select_with_rules(self, intent: ParsedIntent) -> WidgetPlan:
        """Rule-based widget selection — fallback when LLM is unavailable."""
        primary = intent.primary_characteristic
        secondary = intent.secondary_characteristics
        entities = intent.entities
        has_devices = bool(entities.get("devices"))
        query = intent.raw_text.lower()

        heading = self._generate_heading_simple(intent)
        widgets: list[WidgetPlanItem] = []

        def add(scenario: str, size: str, relevance: float, why: str):
            if scenario in VALID_SCENARIOS:
                cat = CATALOG_BY_SCENARIO[scenario]
                widgets.append(WidgetPlanItem(
                    scenario=scenario,
                    size=size,
                    relevance=relevance,
                    why=why,
                    height_units=cat["height_units"],
                    data_request={"query": query, "entities": entities.get("devices", [])},
                ))

        # Hero widget based on primary characteristic
        if primary == "comparison":
            add("comparison", "hero", 0.95, "Direct comparison requested")
            add("trend-multi-line", "expanded", 0.80, "Supporting trend context")
        elif primary == "trend":
            add("trend", "hero", 0.95, "Trend/time series requested")
        elif primary == "distribution":
            add("distribution", "hero", 0.95, "Distribution/breakdown requested")
        elif primary == "flow_sankey":
            add("flow-sankey", "hero", 0.95, "Energy flow requested")
        elif primary == "cumulative":
            add("trends-cumulative", "hero", 0.95, "Cumulative totals requested")
        elif primary == "maintenance":
            add("timeline", "hero", 0.92, "Maintenance timeline")
            add("eventlogstream", "expanded", 0.80, "Maintenance event log")
        elif primary == "work_orders":
            add("eventlogstream", "hero", 0.92, "Work order log")
        elif primary == "shift":
            add("eventlogstream", "hero", 0.90, "Shift log events")
        elif primary == "energy":
            add("trend", "hero", 0.92, "Energy trend")
            add("distribution", "expanded", 0.80, "Energy breakdown")
        elif primary == "health_status":
            if has_devices:
                add("edgedevicepanel", "hero", 0.95, "Detailed device status")
            else:
                add("matrix-heatmap", "hero", 0.90, "Overall health overview")
        elif primary == "power_quality":
            add("trend-multi-line", "hero", 0.92, "PQ parameters over time")
        elif primary == "hvac":
            add("trend", "hero", 0.90, "HVAC trend")
            add("comparison", "expanded", 0.80, "HVAC comparison")
        elif primary == "ups_dg":
            add("trend", "hero", 0.90, "UPS/DG trend")
            add("alerts", "normal", 0.75, "Related alerts")
        elif primary == "top_consumers":
            add("category-bar", "hero", 0.95, "Top consumers ranking")
        elif primary == "alerts":
            add("alerts", "hero", 0.95, "Active alerts")
        elif primary == "people":
            add("peopleview", "hero", 0.90, "Workforce overview")
        elif primary == "supply_chain":
            add("supplychainglobe", "hero", 0.90, "Supply chain overview")
        else:
            # Default: adaptive dashboard based on entity count
            if has_devices and len(entities.get("devices", [])) == 1:
                # Single entity focus
                device = entities["devices"][0]
                add("trend", "hero", 0.95, f"Power consumption trend for {device}.")
                add("kpi", "compact", 0.88, f"Current reading for {device}.")
                add("alerts", "normal", 0.80, f"Active alerts for {device}.")
            elif has_devices and len(entities.get("devices", [])) >= 2:
                # Multi-entity comparison
                add("comparison", "hero", 0.95, "Side-by-side comparison of the requested equipment.")
                add("trend-multi-line", "expanded", 0.85, "Trend overlay of both entities over time.")
                for dev in entities["devices"][:2]:
                    add("kpi", "compact", 0.78, f"Current reading for {dev}.")
                add("alerts", "normal", 0.72, "Related alerts and warnings.")
            else:
                # General overview — no specific entities
                add("matrix-heatmap", "hero", 0.90, "Overview of equipment health across the facility.")
                add("category-bar", "expanded", 0.82, "Ranking of equipment by power consumption.")
                add("alerts", "normal", 0.78, "Active alerts and warnings requiring attention.")
                add("distribution", "normal", 0.72, "Breakdown of energy consumption by category.")
                add("kpi", "compact", 0.68, "Key operational metric at a glance.")
                add("kpi", "compact", 0.64, "Secondary metric for operational context.")

        # Add secondary enrichments
        for char in secondary[:3]:
            if char == "alerts" and not any(w.scenario == "alerts" for w in widgets):
                add("alerts", "normal", 0.70, "Related alerts")
            elif char == "trend" and not any(w.scenario in ("trend", "trend-multi-line") for w in widgets):
                add("trend", "expanded", 0.65, "Supporting trend")
            elif char == "energy" and not any(w.scenario == "distribution" for w in widgets):
                add("distribution", "normal", 0.60, "Energy breakdown")
            elif char == "maintenance" and not any(w.scenario == "timeline" for w in widgets):
                add("timeline", "expanded", 0.60, "Maintenance events")

        # Add KPIs for quick reference (if space allows)
        if has_devices and not any(w.scenario == "kpi" for w in widgets):
            for dev in entities.get("devices", [])[:2]:
                add("kpi", "compact", 0.70, f"Quick status for {dev}")

        # Apply height budget and enforce bans/diversity
        budget = MAX_HEIGHT_UNITS
        final_widgets = []
        kpi_count = 0
        scenario_counts: dict[str, int] = {}
        for w in widgets:
            if w.scenario in BANNED_SCENARIOS:
                continue
            scenario_counts[w.scenario] = scenario_counts.get(w.scenario, 0) + 1
            if scenario_counts[w.scenario] > MAX_SAME_SCENARIO:
                continue
            if w.scenario == "kpi":
                kpi_count += 1
                if kpi_count > MAX_KPIS:
                    continue
            if budget - w.height_units >= 0:
                final_widgets.append(w)
                budget -= w.height_units
            if len(final_widgets) >= MAX_WIDGETS:
                break

        return WidgetPlan(
            heading=heading,
            widgets=final_widgets,
            total_height_units=MAX_HEIGHT_UNITS - budget,
            select_method="fallback",
        )

    def _build_interactive_prompt_section(self, widget_context: dict) -> str:
        """Build the interactive context section appended to the widget selection prompt."""
        equipment = widget_context.get("equipment", "unknown")
        metric = widget_context.get("metric", "")
        history = widget_context.get("conversation_history", [])

        section = f"""

## INTERACTIVE CONTEXT
User is in interactive mode focused on {equipment}{f' ({metric})' if metric else ''}.
All queries relate to this equipment. Pronouns ("it", "this") refer to {equipment}.
Build a FULL dashboard with multiple widgets for this follow-up query:
- Include KPI, trend, maintenance, alerts, edge panel — whatever is relevant
- Do NOT limit to a single widget — provide a complete contextual dashboard
- The user expects rich, multi-widget responses scoped to {equipment}"""

        if history:
            section += "\n\n## CONVERSATION HISTORY\n"
            # Last 10 exchanges, ~2000 tokens max
            for turn in history[-10:]:
                role = "User" if turn.get("role") == "user" else "AI"
                text = turn.get("text", "")[:200]
                section += f'{role}: "{text}"\n'

        return section

    def _generate_heading_simple(self, intent: ParsedIntent) -> str:
        """Generate a simple heading from intent."""
        query = intent.raw_text
        # Truncate to first 60 chars, capitalize first letter
        if len(query) > 60:
            heading = query[:57] + "..."
        else:
            heading = query
        return heading[0].upper() + heading[1:] if heading else "Dashboard"
