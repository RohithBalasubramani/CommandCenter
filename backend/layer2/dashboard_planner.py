"""
Dashboard Planner — Phase 2 of RAG & Widget Selection Redesign.

Replaces the single LLM widget selection call with a two-stage process:
1. Narrative Planning (LLM): What should this dashboard communicate?
2. Widget Allocation (deterministic): Which widgets fill which information slots?

The planner receives:
- ParsedIntent (what the user asked)
- RetrievalPlan (what data was sought)
- Data prefetch summary (what data actually exists)
- Optional retrieval assessment (if Phase 1 data available)

And produces:
- DashboardNarrative with typed InformationSlots
- WidgetPlan from deterministic slot → widget mapping
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from layer2.intent_parser import ParsedIntent
from layer2.query_decomposer import RetrievalPlan
from layer2.widget_selector import (
    WidgetPlan, WidgetPlanItem,
    MAX_HEIGHT_UNITS, MAX_WIDGETS, MAX_KPIS, MAX_SAME_SCENARIO,
    BANNED_SCENARIOS, SAFETY_CRITICAL_SCENARIOS,
    WHY_TEMPLATES,
)
from layer2.widget_catalog import CATALOG_BY_SCENARIO, VALID_SCENARIOS

logger = logging.getLogger(__name__)


# ── Slot-to-scenario mapping ──
# Maps information slot roles to preferred widget scenarios, with fallbacks

SLOT_SCENARIO_MAP = {
    "primary_metric": {
        "scenarios": ["kpi"],
        "default_size": "compact",
        "hero_eligible": False,
    },
    "primary_trend": {
        "scenarios": ["trend", "trend-multi-line"],
        "default_size": "hero",
        "hero_eligible": True,
    },
    "primary_comparison": {
        "scenarios": ["comparison", "trend-multi-line"],
        "default_size": "hero",
        "hero_eligible": True,
    },
    "primary_ranking": {
        "scenarios": ["category-bar", "distribution"],
        "default_size": "hero",
        "hero_eligible": True,
    },
    "primary_device": {
        "scenarios": ["edgedevicepanel"],
        "default_size": "hero",
        "hero_eligible": True,
    },
    "primary_health_overview": {
        "scenarios": ["matrix-heatmap"],
        "default_size": "hero",
        "hero_eligible": True,
    },
    "primary_flow": {
        "scenarios": ["flow-sankey"],
        "default_size": "hero",
        "hero_eligible": True,
    },
    "primary_events": {
        "scenarios": ["timeline", "eventlogstream"],
        "default_size": "hero",
        "hero_eligible": True,
    },
    "primary_alerts": {
        "scenarios": ["alerts"],
        "default_size": "hero",
        "hero_eligible": True,
    },
    "primary_people": {
        "scenarios": ["peopleview", "peoplehexgrid"],
        "default_size": "hero",
        "hero_eligible": True,
    },
    "primary_supply": {
        "scenarios": ["supplychainglobe"],
        "default_size": "hero",
        "hero_eligible": True,
    },
    "trend_context": {
        "scenarios": ["trend", "trend-multi-line", "trends-cumulative"],
        "default_size": "expanded",
        "hero_eligible": False,
    },
    "alert_surface": {
        "scenarios": ["alerts"],
        "default_size": "normal",
        "hero_eligible": False,
    },
    "breakdown": {
        "scenarios": ["distribution", "composition"],
        "default_size": "normal",
        "hero_eligible": False,
    },
    "event_history": {
        "scenarios": ["timeline", "eventlogstream"],
        "default_size": "expanded",
        "hero_eligible": False,
    },
    "kpi_supplement": {
        "scenarios": ["kpi"],
        "default_size": "compact",
        "hero_eligible": False,
    },
    "detail_panel": {
        "scenarios": ["edgedevicepanel"],
        "default_size": "expanded",
        "hero_eligible": False,
    },
    "cumulative_context": {
        "scenarios": ["trends-cumulative"],
        "default_size": "expanded",
        "hero_eligible": False,
    },
    "flow_context": {
        "scenarios": ["flow-sankey"],
        "default_size": "expanded",
        "hero_eligible": False,
    },
    "matrix_context": {
        "scenarios": ["matrix-heatmap"],
        "default_size": "expanded",
        "hero_eligible": False,
    },
}

# Primary characteristic → primary slot type + supporting slots
# Comprehensive dashboards: operators benefit from seeing KPIs, trends, alerts,
# distributions, events, and device panels together — not just 3-4 widgets.
CHARACTERISTIC_SLOT_TEMPLATES = {
    "comparison": {
        "primary": "primary_comparison",
        "supporting": ["kpi_supplement", "kpi_supplement", "kpi_supplement",
                        "trend_context", "alert_surface", "breakdown",
                        "event_history", "cumulative_context"],
    },
    "trend": {
        "primary": "primary_trend",
        "supporting": ["kpi_supplement", "kpi_supplement", "alert_surface",
                        "breakdown", "cumulative_context", "event_history",
                        "detail_panel"],
    },
    "distribution": {
        "primary": "breakdown",
        "supporting": ["kpi_supplement", "kpi_supplement", "trend_context",
                        "alert_surface", "cumulative_context", "event_history"],
    },
    "maintenance": {
        "primary": "primary_events",
        "supporting": ["alert_surface", "kpi_supplement", "kpi_supplement",
                        "trend_context", "detail_panel", "breakdown"],
    },
    "work_orders": {
        "primary": "primary_events",
        "supporting": ["alert_surface", "kpi_supplement", "kpi_supplement",
                        "trend_context", "breakdown"],
    },
    "shift": {
        "primary": "primary_events",
        "supporting": ["kpi_supplement", "kpi_supplement", "alert_surface",
                        "trend_context", "breakdown"],
    },
    "energy": {
        "primary": "primary_trend",
        "supporting": ["breakdown", "kpi_supplement", "kpi_supplement",
                        "cumulative_context", "alert_surface", "event_history",
                        "matrix_context"],
    },
    "health_status": {
        # Depends on whether we have specific entities
        "primary_with_entity": "primary_device",
        "primary_without_entity": "primary_health_overview",
        "supporting": ["alert_surface", "trend_context", "kpi_supplement",
                        "kpi_supplement", "breakdown", "event_history",
                        "detail_panel", "cumulative_context"],
    },
    "flow_sankey": {
        "primary": "primary_flow",
        "supporting": ["kpi_supplement", "kpi_supplement", "trend_context",
                        "alert_surface", "breakdown"],
    },
    "cumulative": {
        "primary": "cumulative_context",
        "supporting": ["kpi_supplement", "kpi_supplement", "trend_context",
                        "alert_surface", "breakdown"],
    },
    "power_quality": {
        "primary": "primary_trend",
        "supporting": ["kpi_supplement", "kpi_supplement", "alert_surface",
                        "breakdown", "event_history", "cumulative_context"],
    },
    "hvac": {
        "primary": "primary_trend",
        "supporting": ["kpi_supplement", "kpi_supplement", "kpi_supplement",
                        "alert_surface", "breakdown", "event_history",
                        "detail_panel"],
    },
    "ups_dg": {
        "primary": "primary_trend",
        "supporting": ["kpi_supplement", "kpi_supplement", "alert_surface",
                        "breakdown", "event_history"],
    },
    "top_consumers": {
        "primary": "primary_ranking",
        "supporting": ["kpi_supplement", "kpi_supplement", "trend_context",
                        "alert_surface", "cumulative_context"],
    },
    "alerts": {
        "primary": "primary_alerts",
        "supporting": ["kpi_supplement", "kpi_supplement", "trend_context",
                        "event_history", "breakdown", "detail_panel"],
    },
    "people": {
        "primary": "primary_people",
        "supporting": ["kpi_supplement", "kpi_supplement", "event_history"],
    },
    "supply_chain": {
        "primary": "primary_supply",
        "supporting": ["kpi_supplement", "kpi_supplement", "event_history"],
    },
}

# Narrative prompt for LLM (optional enrichment)
NARRATIVE_PROMPT = '''Given this industrial operations query and available data, plan what the dashboard should communicate.

## QUERY
"{transcript}"
Intent: {intent_type} | Entities: {entities} | Primary: {primary_char}

## DATA AVAILABLE
{data_summary}

## RETRIEVAL PLAN
Required metrics: {required_metrics}
Temporal scope: {temporal_hours}h ({granularity})
Causal hypothesis: {causal_hypothesis}

## INSTRUCTIONS
Think about:
1. What is the PRIMARY answer the operator needs?
2. What CONTEXT makes the primary answer more useful?
3. Are there RISKS or anomalies that should be surfaced even if not asked?
4. What data GAPS should the operator be made aware of?

## OUTPUT (JSON only)
{{"primary_answer": "<1 sentence: what the dashboard primarily shows>", "risk_note": "<risk assessment or null>", "data_gaps": ["<gap1>"], "information_priority": ["<what matters most>", "<second>", "<third>"]}}'''


@dataclass
class InformationSlot:
    """A semantic slot in the dashboard that needs filling."""
    role: str           # Key into SLOT_SCENARIO_MAP
    priority: float     # 0-1, determines order
    entities: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "priority": self.priority,
            "entities": self.entities,
            "metrics": self.metrics,
            "description": self.description,
        }


@dataclass
class DashboardNarrative:
    """What story should this dashboard tell?"""
    heading: str = "Dashboard"
    primary_answer: str = ""
    risk_note: Optional[str] = None
    data_gaps: list[str] = field(default_factory=list)
    information_slots: list[InformationSlot] = field(default_factory=list)
    plan_method: str = "rule"  # "rule" or "llm"

    def to_dict(self) -> dict:
        return {
            "heading": self.heading,
            "primary_answer": self.primary_answer,
            "risk_note": self.risk_note,
            "data_gaps": self.data_gaps,
            "slots": [s.to_dict() for s in self.information_slots],
            "plan_method": self.plan_method,
        }


class DashboardPlanner:
    """
    Plans dashboard narratives and allocates widgets from information slots.

    Strategy:
    1. Rule-based slot generation from intent characteristics (always)
    2. Optional LLM enrichment for risk notes and priority reordering
    3. Deterministic widget allocation from slots
    """

    def __init__(self):
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from layer2.rag_pipeline import get_rag_pipeline
            self._pipeline = get_rag_pipeline()
        return self._pipeline

    def plan(self, intent: ParsedIntent, retrieval_plan: RetrievalPlan,
             data_summary: str = "", user_context: str = "",
             widget_context: dict = None) -> DashboardNarrative:
        """
        Generate a dashboard narrative from intent and retrieval plan.

        Returns a DashboardNarrative with typed information slots.
        """
        narrative = DashboardNarrative()

        # Generate heading
        narrative.heading = self._generate_heading(intent)

        # Build information slots from characteristics
        entities = intent.entities.get("devices", [])
        primary = intent.primary_characteristic
        secondary = intent.secondary_characteristics

        # Primary slot
        self._add_primary_slot(narrative, primary, entities, retrieval_plan)

        # Supporting slots
        self._add_supporting_slots(narrative, primary, secondary, entities, retrieval_plan)

        # Interactive mode adjustments
        if widget_context:
            self._adjust_for_interactive(narrative, widget_context, entities)

        # Optional LLM enrichment for risk notes
        try:
            self._enrich_narrative(narrative, intent, retrieval_plan, data_summary)
        except Exception as e:
            logger.debug(f"Narrative LLM enrichment skipped: {e}")

        logger.info(
            f"[planner] Narrative: {len(narrative.information_slots)} slots, "
            f"primary='{narrative.primary_answer[:60] if narrative.primary_answer else 'none'}', "
            f"method={narrative.plan_method}"
        )

        return narrative

    def allocate_widgets(self, narrative: DashboardNarrative,
                          intent: ParsedIntent,
                          retrieval_plan: RetrievalPlan) -> WidgetPlan:
        """
        Deterministically allocate widgets from information slots.

        Maps each slot to a widget scenario using SLOT_SCENARIO_MAP,
        then applies budget constraints.
        """
        widgets = []
        budget = MAX_HEIGHT_UNITS
        kpi_count = 0
        scenario_counts: dict[str, int] = {}
        hero_assigned = False

        entities = intent.entities.get("devices", [])

        for slot in narrative.information_slots:
            mapping = SLOT_SCENARIO_MAP.get(slot.role)
            if not mapping:
                logger.warning(f"[planner] Unknown slot role: {slot.role}")
                continue

            # Pick the first valid scenario from the mapping
            scenario = None
            for candidate in mapping["scenarios"]:
                if candidate in BANNED_SCENARIOS:
                    continue
                if candidate not in VALID_SCENARIOS:
                    continue
                scenario_counts_for = scenario_counts.get(candidate, 0)
                if scenario_counts_for >= MAX_SAME_SCENARIO:
                    continue
                if candidate == "kpi":
                    if kpi_count >= MAX_KPIS:
                        continue
                scenario = candidate
                break

            if not scenario:
                continue

            # Determine size
            if not hero_assigned and mapping.get("hero_eligible", False):
                size = "hero"
                hero_assigned = True
            else:
                size = mapping["default_size"]
                if size == "hero":
                    size = "expanded"

            # Check height budget
            catalog_entry = CATALOG_BY_SCENARIO.get(scenario)
            if not catalog_entry:
                continue
            height = catalog_entry.get("height_units", 2)

            # Validate size against allowed sizes
            allowed_sizes = set(catalog_entry.get("sizes", ["normal"]))
            if size not in allowed_sizes:
                size_priority = ["hero", "expanded", "normal", "compact"]
                size = next((s for s in size_priority if s in allowed_sizes), "normal")

            if budget - height < 0:
                continue
            budget -= height

            # Build data_request
            slot_entities = slot.entities or entities[:2]
            slot_metrics = slot.metrics or retrieval_plan.required_metrics[:1]
            data_request = {
                "query": slot.description or intent.raw_text,
                "entities": slot_entities,
                "metric": slot_metrics[0] if slot_metrics else "",
            }

            # Build why description
            why = slot.description or WHY_TEMPLATES.get(scenario, "")

            item = WidgetPlanItem(
                scenario=scenario,
                size=size,
                relevance=slot.priority,
                why=why,
                data_request=data_request,
                height_units=height,
            )
            widgets.append(item)

            # Track counts
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
            if scenario == "kpi":
                kpi_count += 1

            if len(widgets) >= MAX_WIDGETS:
                break

        # Ensure we have a hero widget
        if widgets and widgets[0].size != "hero":
            # Promote the first hero-eligible widget
            for w in widgets:
                catalog = CATALOG_BY_SCENARIO.get(w.scenario, {})
                if "hero" in catalog.get("sizes", []):
                    w.size = "hero"
                    break

        # ── Fill remaining budget with additional relevant widget types ──
        # After slot-based allocation, pad with unused scenarios for a comprehensive dashboard
        used_scenarios = set(w.scenario for w in widgets)
        # Fill with DIVERSE widget types not already in the dashboard
        # Prioritize unused scenarios to maximize widget variety
        fill_candidates = [
            ("composition", "normal", 0.40, "Composition analysis"),
            ("category-bar", "expanded", 0.45, "Category comparison"),
            ("eventlogstream", "expanded", 0.40, "Recent event activity"),
            ("timeline", "expanded", 0.42, "Event timeline"),
            ("trend-multi-line", "expanded", 0.44, "Multi-metric trend comparison"),
            ("trends-cumulative", "expanded", 0.38, "Cumulative trend"),
            ("matrix-heatmap", "expanded", 0.43, "Health overview matrix"),
            ("flow-sankey", "expanded", 0.37, "Energy/resource flow"),
            ("peoplenetwork", "expanded", 0.35, "Personnel network"),
            ("diagnosticpanel", "expanded", 0.36, "Diagnostic analysis"),
            ("kpi", "compact", 0.45, "Key metric overview"),
            ("trend", "expanded", 0.50, "Historical trend analysis"),
            ("alerts", "normal", 0.50, "Active alerts monitoring"),
            ("distribution", "normal", 0.40, "Data distribution breakdown"),
            ("comparison", "expanded", 0.42, "Side-by-side comparison"),
            ("uncertaintypanel", "expanded", 0.34, "Confidence and uncertainty view"),
            ("edgedevicepanel", "expanded", 0.38, "Equipment detail panel"),
            ("peoplehexgrid", "expanded", 0.33, "Team overview grid"),
            ("peopleview", "expanded", 0.32, "Personnel details"),
            ("chatstream", "expanded", 0.31, "Activity feed"),
        ]
        for fill_scenario, fill_size, fill_rel, fill_why in fill_candidates:
            if len(widgets) >= MAX_WIDGETS or budget <= 0:
                break
            # Prefer scenarios not yet in dashboard (maximize diversity)
            sc_count = scenario_counts.get(fill_scenario, 0)
            if fill_scenario in used_scenarios and sc_count >= MAX_SAME_SCENARIO:
                continue
            if fill_scenario == "kpi" and kpi_count >= MAX_KPIS:
                continue
            cat = CATALOG_BY_SCENARIO.get(fill_scenario)
            if not cat:
                continue
            h = cat.get("height_units", 2)
            if budget - h < 0:
                continue
            # Validate size
            allowed = set(cat.get("sizes", ["normal"]))
            if fill_size not in allowed:
                fill_size = "normal" if "normal" in allowed else list(allowed)[0]
            budget -= h
            # Use retrieval plan's metrics when available for data collection
            fill_metric = ""
            if retrieval_plan and retrieval_plan.required_metrics:
                fill_metric = retrieval_plan.required_metrics[0]
            item = WidgetPlanItem(
                scenario=fill_scenario,
                size=fill_size,
                relevance=fill_rel,
                why=fill_why,
                data_request={
                    "query": intent.raw_text,
                    "entities": entities[:2],
                    "metric": fill_metric,
                },
                height_units=h,
            )
            widgets.append(item)
            scenario_counts[fill_scenario] = sc_count + 1
            if fill_scenario == "kpi":
                kpi_count += 1

        plan = WidgetPlan(
            heading=narrative.heading,
            widgets=widgets,
            total_height_units=MAX_HEIGHT_UNITS - budget,
            select_method="planner",
        )

        logger.info(
            f"[planner] Allocated {len(widgets)} widgets (incl. fill), "
            f"height={plan.total_height_units}/{MAX_HEIGHT_UNITS}"
        )

        return plan

    def _add_primary_slot(self, narrative: DashboardNarrative,
                           primary: Optional[str], entities: list[str],
                           plan: RetrievalPlan):
        """Add the primary information slot based on query characteristic."""
        template = CHARACTERISTIC_SLOT_TEMPLATES.get(primary, {})

        if primary == "health_status":
            # Choose between device detail and health overview
            if entities:
                role = template.get("primary_with_entity", "primary_device")
            else:
                role = template.get("primary_without_entity", "primary_health_overview")
        elif template:
            role = template.get("primary", "primary_trend")
        else:
            # Default: adaptive based on entity count
            if not entities:
                role = "primary_health_overview"
            elif len(entities) == 1:
                role = "primary_trend"
            elif len(entities) >= 2:
                role = "primary_comparison"
            else:
                role = "primary_trend"

        slot = InformationSlot(
            role=role,
            priority=0.95,
            entities=entities[:4],
            metrics=plan.required_metrics[:2],
            description=f"Primary answer for: {primary or 'general query'}",
        )
        narrative.information_slots.append(slot)

    def _add_supporting_slots(self, narrative: DashboardNarrative,
                               primary: Optional[str], secondary: list[str],
                               entities: list[str], plan: RetrievalPlan):
        """Add supporting information slots."""
        template = CHARACTERISTIC_SLOT_TEMPLATES.get(primary, {})
        supporting_roles = template.get("supporting", [])

        # Default supporting slots if no template
        if not supporting_roles:
            if entities:
                supporting_roles = ["kpi_supplement", "alert_surface", "trend_context"]
            else:
                supporting_roles = ["alert_surface", "breakdown", "kpi_supplement"]

        # Add supporting slots with decreasing priority
        base_priority = 0.80
        for i, role in enumerate(supporting_roles):
            priority = max(0.4, base_priority - i * 0.10)

            # For KPI supplements, assign different entities/metrics
            slot_entities = entities[:2]
            slot_metrics = plan.required_metrics[:1]
            description = ""

            if role == "kpi_supplement":
                # Alternate between different metrics for KPI diversity
                all_metrics = plan.required_metrics + plan.nice_to_have_metrics
                metric_idx = i % len(all_metrics) if all_metrics else 0
                slot_metrics = [all_metrics[metric_idx]] if all_metrics else []
                entity_idx = i % len(entities) if entities else 0
                slot_entities = [entities[entity_idx]] if entities else []
                if slot_metrics:
                    description = f"Current {slot_metrics[0].replace('_', ' ')} reading"

            elif role == "alert_surface":
                description = "Active alerts and warnings for monitored equipment"

            elif role == "trend_context":
                description = f"Historical trend over {plan.temporal_scope.hours}h"

            elif role == "breakdown":
                description = "Breakdown of values across categories"

            elif role == "event_history":
                description = "Timeline of recent maintenance and operational events"

            slot = InformationSlot(
                role=role,
                priority=priority,
                entities=slot_entities,
                metrics=slot_metrics,
                description=description,
            )
            narrative.information_slots.append(slot)

        # Add secondary characteristic enrichments
        for char in secondary[:2]:
            if char == "alerts" and not any(s.role == "alert_surface" for s in narrative.information_slots):
                narrative.information_slots.append(InformationSlot(
                    role="alert_surface", priority=0.65,
                    entities=entities[:2],
                    description="Related alerts and warnings",
                ))
            elif char == "maintenance" and not any(s.role == "event_history" for s in narrative.information_slots):
                narrative.information_slots.append(InformationSlot(
                    role="event_history", priority=0.55,
                    entities=entities[:2],
                    description="Maintenance history and events",
                ))

    def _adjust_for_interactive(self, narrative: DashboardNarrative,
                                 widget_context: dict, entities: list[str]):
        """Adjust narrative for interactive mode — ensure rich contextual dashboard."""
        equipment = widget_context.get("equipment", "")
        metric = widget_context.get("metric", "")

        # In interactive mode, ensure at least 5 slots and scope to equipment
        if len(narrative.information_slots) < 5:
            # Add detail panel if not present
            if not any(s.role == "detail_panel" for s in narrative.information_slots):
                narrative.information_slots.append(InformationSlot(
                    role="detail_panel", priority=0.7,
                    entities=[equipment] if equipment else entities[:1],
                    description=f"Detailed view of {equipment}",
                ))
            # Add trend context
            if not any(s.role == "trend_context" for s in narrative.information_slots):
                narrative.information_slots.append(InformationSlot(
                    role="trend_context", priority=0.65,
                    entities=[equipment] if equipment else entities[:1],
                    metrics=[metric] if metric else [],
                    description=f"Trend for {equipment} {metric}",
                ))

        # Scope all slots to the focused equipment
        for slot in narrative.information_slots:
            if equipment and not slot.entities:
                slot.entities = [equipment]

    def _enrich_narrative(self, narrative: DashboardNarrative,
                           intent: ParsedIntent, plan: RetrievalPlan,
                           data_summary: str):
        """Optionally enrich narrative with LLM for risk notes and primary answer."""
        llm = self.pipeline.llm_fast

        entities_str = json.dumps(intent.entities.get("devices", []))
        prompt = NARRATIVE_PROMPT.format(
            transcript=intent.raw_text,
            intent_type=intent.type,
            entities=entities_str,
            primary_char=intent.primary_characteristic or "general",
            data_summary=data_summary or "No specific data available.",
            required_metrics=", ".join(plan.required_metrics) or "default",
            temporal_hours=plan.temporal_scope.hours,
            granularity=plan.temporal_scope.granularity,
            causal_hypothesis=plan.causal_hypothesis or "none",
        )

        data = llm.generate_json(
            prompt=prompt,
            system_prompt="You plan industrial dashboards. Respond with JSON only.",
            temperature=0.1,
            max_tokens=512,
            cache_key=f"narrative:{intent.raw_text}",
        )

        if data is None:
            return

        if isinstance(data.get("primary_answer"), str):
            narrative.primary_answer = data["primary_answer"]
        if isinstance(data.get("risk_note"), str) and data["risk_note"].lower() != "null":
            narrative.risk_note = data["risk_note"]
        if isinstance(data.get("data_gaps"), list):
            narrative.data_gaps = [g for g in data["data_gaps"] if isinstance(g, str)]

        narrative.plan_method = "llm"

    def _generate_heading(self, intent: ParsedIntent) -> str:
        """Generate a dashboard heading from intent."""
        query = intent.raw_text
        if len(query) > 60:
            heading = query[:57] + "..."
        else:
            heading = query
        return heading[0].upper() + heading[1:] if heading else "Dashboard"
