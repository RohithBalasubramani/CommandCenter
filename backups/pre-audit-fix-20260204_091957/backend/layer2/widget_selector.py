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

logger = logging.getLogger(__name__)

MAX_HEIGHT_UNITS = 18  # scrollable dashboard budget
MAX_WIDGETS = 10
MAX_KPIS = 4

# Scenarios that should NEVER be selected by the LLM
BANNED_SCENARIOS = {"helpview", "pulseview"}
# Maximum instances of the same scenario type
MAX_SAME_SCENARIO = 2


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


class WidgetSelector:
    """LLM-based widget selector with rule-based fallback."""

    def __init__(self):
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = get_rag_pipeline()
        return self._pipeline

    def select(self, intent: ParsedIntent, data_summary: str = "",
               user_context: str = "") -> WidgetPlan:
        """Select widgets using 8B LLM, with fallback to rule-based selection."""
        # Try LLM first
        try:
            result = self._select_with_llm(intent, data_summary, user_context)
            if result is not None and result.widgets:
                return result
        except Exception as e:
            logger.warning(f"LLM widget selection failed: {e}")

        # Fallback to rule-based
        logger.info("Falling back to rule-based widget selection")
        return self._select_with_rules(intent)

    def _select_with_llm(self, intent: ParsedIntent, data_summary: str,
                          user_context: str) -> Optional[WidgetPlan]:
        """Select widgets using LLM. Uses quality (70B) model if WIDGET_SELECT_QUALITY=1."""
        use_quality = os.getenv("WIDGET_SELECT_QUALITY", "0") == "1"
        llm = self.pipeline.llm_quality if use_quality else self.pipeline.llm_fast
        logger.info(f"Widget selection using {'quality (70B)' if use_quality else 'fast (8B)'} model")
        catalog_text = get_catalog_prompt_text()

        prompt = SELECT_PROMPT_TEMPLATE.format(
            max_widgets=MAX_WIDGETS,
            max_height=MAX_HEIGHT_UNITS,
            max_kpis=MAX_KPIS,
            catalog=catalog_text,
            data_summary=data_summary or "No pre-fetch data available.",
            user_memory=user_context or "New user — no prior history.",
            query=intent.raw_text,
            intent_type=intent.type,
            domains=json.dumps(intent.domains),
            entities=json.dumps(intent.entities),
            primary_char=intent.primary_characteristic or "none",
            secondary_chars=json.dumps(intent.secondary_characteristics),
        )

        data = llm.generate_json(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=2048,
        )

        if data is None:
            return None

        return self._validate_and_build_plan(data, method="llm")

    def _validate_and_build_plan(self, data: dict, method: str = "llm") -> WidgetPlan:
        """Validate LLM output and build a WidgetPlan."""
        heading = data.get("heading", "Dashboard")
        raw_widgets = data.get("widgets", [])

        plan = WidgetPlan(heading=heading, select_method=method)
        budget = MAX_HEIGHT_UNITS
        kpi_count = 0
        scenario_counts: dict[str, int] = {}

        for w in raw_widgets:
            scenario = w.get("scenario", "")
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

            # Validate size
            size = w.get("size", "normal")
            valid_sizes = {"compact", "normal", "expanded", "hero"}
            if size not in valid_sizes:
                size = catalog_entry["sizes"][0] if catalog_entry["sizes"] else "normal"

            item = WidgetPlanItem(
                scenario=scenario,
                size=size,
                relevance=min(1.0, max(0.0, w.get("relevance", 0.8))),
                why=w.get("why", ""),
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
            # Default: general status dashboard with diverse widgets
            add("trend", "hero", 0.90, "Shows the most relevant metric trend over recent hours.")
            add("kpi", "compact", 0.85, "Key metric at a glance showing current value and status.")
            add("kpi", "compact", 0.83, "Secondary metric providing additional operational context.")
            add("alerts", "normal", 0.80, "Active alerts and warnings requiring attention.")
            add("distribution", "normal", 0.75, "Breakdown of key metrics by category or source.")
            add("category-bar", "expanded", 0.70, "Ranking of equipment or zones by performance.")

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

    def _generate_heading_simple(self, intent: ParsedIntent) -> str:
        """Generate a simple heading from intent."""
        query = intent.raw_text
        # Truncate to first 60 chars, capitalize first letter
        if len(query) > 60:
            heading = query[:57] + "..."
        else:
            heading = query
        return heading[0].upper() + heading[1:] if heading else "Dashboard"
