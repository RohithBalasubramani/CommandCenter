"""
Query Decomposer — Phase 2 of RAG & Widget Selection Redesign.

Decomposes a user transcript into a structured RetrievalPlan BEFORE any
retrieval executes. Maps operator language to specific data needs:
- Which metrics are required vs nice-to-have
- What temporal scope is appropriate
- What related entities might be relevant
- What causal hypothesis might explain the user's question

Uses the 8B LLM with structured JSON output. Budget: 300-500ms.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from layer2.data_collector import (
    ENTITY_TABLE_PREFIX_MAP,
    EQUIPMENT_METRIC_MAP,
    resolve_entity_to_table,
    resolve_metric_column,
)
from layer2.intent_parser import ParsedIntent

logger = logging.getLogger(__name__)

# Temporal scope defaults by query type
TEMPORAL_DEFAULTS = {
    "trend": {"hours": 24, "granularity": "1 hour"},
    "comparison": {"hours": 24, "granularity": "1 hour"},
    "cumulative": {"hours": 24, "granularity": "1 hour"},
    "distribution": {"hours": 24, "granularity": "1 hour"},
    "alerts": {"hours": 168, "granularity": "1 day"},  # 7 days
    "maintenance": {"hours": 720, "granularity": "1 day"},  # 30 days
    "shift": {"hours": 12, "granularity": "15 minutes"},
    "health_status": {"hours": 1, "granularity": "5 minutes"},
    "default": {"hours": 24, "granularity": "1 hour"},
}

# Shift schedule for temporal resolution
SHIFT_HOURS = {
    "morning": (6, 14), "day": (6, 14),
    "afternoon": (14, 22), "evening": (14, 22),
    "night": (22, 6),
    "first": (6, 14), "second": (14, 22), "third": (22, 6),
}

DECOMPOSE_PROMPT = '''Decompose this industrial operations query into data retrieval needs.

## EQUIPMENT METRICS AVAILABLE
{available_metrics}

## QUERY
"{transcript}"

## PARSED INTENT
Type: {intent_type}
Entities: {entities}
Primary: {primary_char}
Domains: {domains}

## INSTRUCTIONS
Identify exactly what data is needed to answer this query.
- required_metrics: metrics the answer DEPENDS on (use exact column names from above)
- nice_to_have_metrics: metrics that add context but aren't essential
- temporal_scope_hours: how many hours of history are relevant (1-720)
- causal_hypothesis: if the user seems to be investigating a problem, state what they might be looking for (null if purely informational)
- supporting_questions: what additional data would help answer this comprehensively (max 3)
- related_entity_types: other equipment types that are operationally connected (e.g., chiller → cooling_tower)

## OUTPUT (JSON only)
{{"required_metrics": ["<column_name>"], "nice_to_have_metrics": ["<column_name>"], "temporal_scope_hours": <int>, "granularity": "<interval>", "causal_hypothesis": "<string or null>", "supporting_questions": ["<q1>", "<q2>"], "related_entity_types": ["<type>"]}}'''


@dataclass
class TemporalScope:
    """Time window for data retrieval."""
    hours: int = 24
    granularity: str = "1 hour"  # PG interval string
    rationale: str = ""
    shift_aware: bool = False
    shift_name: str = ""


@dataclass
class RetrievalPlan:
    """Structured plan for what data needs to be retrieved."""
    primary_question: str = ""
    supporting_questions: list[str] = field(default_factory=list)
    temporal_scope: TemporalScope = field(default_factory=TemporalScope)
    required_metrics: list[str] = field(default_factory=list)  # Resolved PG column names
    nice_to_have_metrics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    entity_tables: dict[str, str] = field(default_factory=dict)  # entity → table_name
    entity_prefixes: dict[str, str] = field(default_factory=dict)  # entity → prefix
    related_entity_types: list[str] = field(default_factory=list)
    causal_hypothesis: Optional[str] = None
    include_alerts: bool = True
    include_maintenance: bool = False
    include_work_orders: bool = False
    decompose_method: str = "llm"  # "llm" or "rule"

    def to_dict(self) -> dict:
        return {
            "primary_question": self.primary_question,
            "supporting_questions": self.supporting_questions,
            "temporal_scope": {
                "hours": self.temporal_scope.hours,
                "granularity": self.temporal_scope.granularity,
                "rationale": self.temporal_scope.rationale,
            },
            "required_metrics": self.required_metrics,
            "nice_to_have_metrics": self.nice_to_have_metrics,
            "entities": self.entities,
            "entity_tables": self.entity_tables,
            "related_entity_types": self.related_entity_types,
            "causal_hypothesis": self.causal_hypothesis,
            "include_alerts": self.include_alerts,
            "include_maintenance": self.include_maintenance,
            "decompose_method": self.decompose_method,
        }


class QueryDecomposer:
    """
    Decomposes user queries into structured retrieval plans.

    Strategy:
    1. Rule-based resolution first (entity → table, metric → column) — fast, deterministic
    2. LLM enrichment for causal hypothesis, supporting questions, temporal scope — when available
    3. Pure rule fallback if LLM is unavailable
    """

    def __init__(self):
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from layer2.rag_pipeline import get_rag_pipeline
            self._pipeline = get_rag_pipeline()
        return self._pipeline

    def decompose(self, intent: ParsedIntent) -> RetrievalPlan:
        """
        Decompose a parsed intent into a retrieval plan.

        Always starts with deterministic entity/metric resolution,
        then optionally enriches with LLM for causal reasoning.
        """
        plan = RetrievalPlan(
            primary_question=intent.raw_text,
            entities=intent.entities.get("devices", []),
        )

        # Step 1: Resolve entities to PG tables (deterministic)
        self._resolve_entities(plan)

        # Step 2: Resolve metrics from query text (deterministic)
        self._resolve_metrics(plan, intent)

        # Step 3: Determine temporal scope (deterministic + query parsing)
        self._resolve_temporal_scope(plan, intent)

        # Step 4: Determine which collections to search
        self._resolve_collections(plan, intent)

        # Step 5: Try LLM enrichment for causal reasoning
        try:
            self._enrich_with_llm(plan, intent)
        except Exception as e:
            logger.debug(f"LLM enrichment skipped: {e}")
            plan.decompose_method = "rule"

        logger.info(
            f"[decomposer] Plan: {len(plan.required_metrics)} required metrics, "
            f"{len(plan.nice_to_have_metrics)} nice-to-have, "
            f"temporal={plan.temporal_scope.hours}h/{plan.temporal_scope.granularity}, "
            f"entities={list(plan.entity_tables.keys())}, "
            f"causal={'yes' if plan.causal_hypothesis else 'no'}, "
            f"method={plan.decompose_method}"
        )

        return plan

    def _resolve_entities(self, plan: RetrievalPlan):
        """Resolve entity names to PG table names and prefixes."""
        for entity in plan.entities:
            resolved = resolve_entity_to_table(entity)
            if resolved:
                table_name, prefix, default_metric, default_unit = resolved
                plan.entity_tables[entity] = table_name
                plan.entity_prefixes[entity] = prefix

    def _resolve_metrics(self, plan: RetrievalPlan, intent: ParsedIntent):
        """Resolve metric names from query text to PG column names."""
        query_lower = intent.raw_text.lower()

        # Metric keywords in query → metric alias
        METRIC_KEYWORDS = {
            "temperature": "temperature", "temp": "temperature",
            "vibration": "vibration", "pressure": "pressure",
            "voltage": "voltage", "current": "current",
            "power factor": "power_factor", "pf": "power_factor",
            "power": "power", "load": "load", "energy": "power",
            "consumption": "power", "frequency": "frequency",
            "humidity": "humidity", "flow": "flow", "flow rate": "flow",
            "cop": "cop", "efficiency": "cop",
            "battery": "battery", "runtime": "runtime",
            "speed": "speed", "rpm": "speed",
            "oil temp": "oil_temp", "winding temp": "winding_temp",
            "coolant": "coolant", "fuel": "fuel",
        }

        # Extract metric from query — check longer phrases first
        detected_metric_alias = ""
        for kw in sorted(METRIC_KEYWORDS, key=len, reverse=True):
            if kw in query_lower:
                detected_metric_alias = METRIC_KEYWORDS[kw]
                break

        # Resolve to actual PG column names per entity
        for entity in plan.entities:
            prefix = plan.entity_prefixes.get(entity)
            if not prefix:
                continue

            metric_map = EQUIPMENT_METRIC_MAP.get(prefix, {})

            if detected_metric_alias:
                col, unit = resolve_metric_column(prefix, detected_metric_alias)
                if col not in plan.required_metrics:
                    plan.required_metrics.append(col)
            else:
                # No explicit metric — use default
                default_col = metric_map.get("default", ("active_power_kw", "kW"))
                if isinstance(default_col, tuple):
                    col = default_col[0]
                else:
                    col = default_col
                if col not in plan.required_metrics:
                    plan.required_metrics.append(col)

            # Add nice-to-have metrics based on equipment type
            nice_to_have = self._get_contextual_metrics(prefix, detected_metric_alias)
            for nh in nice_to_have:
                if nh not in plan.required_metrics and nh not in plan.nice_to_have_metrics:
                    plan.nice_to_have_metrics.append(nh)

    def _get_contextual_metrics(self, prefix: str, primary_metric: str) -> list[str]:
        """Get contextual metrics that complement the primary one."""
        context_map = {
            "trf": {
                "power": ["load_percent", "oil_temperature_top_c"],
                "temperature": ["active_power_kw", "load_percent"],
                "load": ["active_power_kw", "oil_temperature_top_c"],
                "default": ["active_power_kw", "load_percent"],
            },
            "dg": {
                "power": ["load_percent", "fuel_level_pct"],
                "fuel": ["active_power_kw", "load_percent"],
                "default": ["active_power_kw", "fuel_level_pct"],
            },
            "ups": {
                "power": ["load_percent", "battery_charge_pct"],
                "battery": ["output_power_kw", "battery_time_remaining_min"],
                "default": ["output_power_kw", "battery_charge_pct"],
            },
            "chiller": {
                "power": ["load_percent", "current_cop"],
                "temperature": ["power_consumption_kw", "current_cop"],
                "cop": ["power_consumption_kw", "chw_supply_temp_c"],
                "default": ["power_consumption_kw", "current_cop"],
            },
            "pump": {
                "power": ["flow_rate_m3h", "vibration_de_mm_s"],
                "vibration": ["motor_power_kw", "discharge_pressure_bar"],
                "pressure": ["motor_power_kw", "flow_rate_m3h"],
                "default": ["motor_power_kw", "flow_rate_m3h"],
            },
            "motor": {
                "power": ["load_percent", "winding_temp_r_c"],
                "vibration": ["active_power_kw", "winding_temp_r_c"],
                "temperature": ["active_power_kw", "vibration_de_h_mm_s"],
                "default": ["active_power_kw", "load_percent"],
            },
        }

        equip_context = context_map.get(prefix, {})
        return equip_context.get(primary_metric, equip_context.get("default", []))

    def _resolve_temporal_scope(self, plan: RetrievalPlan, intent: ParsedIntent):
        """Determine the appropriate time window from query text and intent."""
        query_lower = intent.raw_text.lower()
        primary = intent.primary_characteristic or "default"

        # Start with default for the query type
        defaults = TEMPORAL_DEFAULTS.get(primary, TEMPORAL_DEFAULTS["default"])
        scope = TemporalScope(
            hours=defaults["hours"],
            granularity=defaults["granularity"],
            rationale=f"Default for {primary} queries",
        )

        # Override with explicit temporal references in query
        # "last N hours/days/minutes"
        time_match = re.search(r'(?:last|past)\s+(\d+)\s+(hour|minute|day|week|month)', query_lower)
        if time_match:
            num = int(time_match.group(1))
            unit = time_match.group(2)
            multiplier = {"minute": 1/60, "hour": 1, "day": 24, "week": 168, "month": 720}
            scope.hours = int(num * multiplier.get(unit, 1))
            scope.granularity = self._pick_granularity(scope.hours)
            scope.rationale = f"Explicit: last {num} {unit}(s)"

        # "today" / "this morning"
        elif "today" in query_lower:
            scope.hours = 16  # Approx hours since midnight
            scope.granularity = "1 hour"
            scope.rationale = "Today (since midnight)"

        # "yesterday"
        elif "yesterday" in query_lower:
            scope.hours = 48  # Need yesterday's full day
            scope.granularity = "1 hour"
            scope.rationale = "Yesterday"

        # Shift-aware queries
        for shift_name, (start_h, end_h) in SHIFT_HOURS.items():
            if shift_name in query_lower or f"{shift_name} shift" in query_lower:
                if end_h > start_h:
                    scope.hours = end_h - start_h
                else:
                    scope.hours = (24 - start_h) + end_h
                scope.granularity = "15 minutes"
                scope.shift_aware = True
                scope.shift_name = shift_name
                scope.rationale = f"Shift-aware: {shift_name} shift ({start_h}:00-{end_h}:00)"
                break

        # "this week"
        if "this week" in query_lower:
            scope.hours = 168
            scope.granularity = "1 day"
            scope.rationale = "This week"

        # "this month"
        elif "this month" in query_lower:
            scope.hours = 720
            scope.granularity = "1 day"
            scope.rationale = "This month"

        # Time references from intent parser
        time_refs = intent.entities.get("time", [])
        for tref in time_refs:
            tref_lower = tref.lower()
            if "hour" in tref_lower:
                num_match = re.search(r'(\d+)', tref_lower)
                if num_match:
                    scope.hours = int(num_match.group(1))
                    scope.granularity = self._pick_granularity(scope.hours)
                    scope.rationale = f"From entity: {tref}"

        plan.temporal_scope = scope

    def _pick_granularity(self, hours: int) -> str:
        """Pick appropriate aggregation granularity for a given time window."""
        if hours <= 2:
            return "5 minutes"
        elif hours <= 6:
            return "15 minutes"
        elif hours <= 48:
            return "1 hour"
        elif hours <= 168:
            return "1 day"
        else:
            return "1 day"

    def _resolve_collections(self, plan: RetrievalPlan, intent: ParsedIntent):
        """Determine which auxiliary collections to search."""
        primary = intent.primary_characteristic or ""
        domains = intent.domains or []
        query_lower = intent.raw_text.lower()

        # Always include alerts for equipment queries
        plan.include_alerts = "industrial" in domains or "alerts" in domains

        # Include maintenance for relevant queries
        maintenance_signals = [
            "maintenance" in query_lower, "repair" in query_lower,
            "service" in query_lower, "replaced" in query_lower,
            "failure" in query_lower, "failing" in query_lower,
            "why" in query_lower,  # "Why is X failing?" → check maintenance history
            primary == "maintenance",
        ]
        plan.include_maintenance = any(maintenance_signals)

        # Include work orders
        wo_signals = [
            "work order" in query_lower, "task" in query_lower,
            "pending" in query_lower, "overdue" in query_lower,
            primary == "work_orders",
        ]
        plan.include_work_orders = any(wo_signals)

        # Causal queries should include maintenance history
        causal_signals = [
            "why" in query_lower, "cause" in query_lower,
            "reason" in query_lower, "failing" in query_lower,
            "problem" in query_lower, "issue" in query_lower,
            "wrong" in query_lower, "underperform" in query_lower,
        ]
        if any(causal_signals):
            plan.include_maintenance = True
            plan.include_alerts = True

    def _enrich_with_llm(self, plan: RetrievalPlan, intent: ParsedIntent):
        """Optionally enrich the plan with LLM reasoning."""
        llm = self.pipeline.llm_fast

        # Build available metrics text for the prompt
        metrics_text = self._build_metrics_text(plan)

        entities_str = json.dumps(intent.entities.get("devices", []))
        prompt = DECOMPOSE_PROMPT.format(
            available_metrics=metrics_text,
            transcript=intent.raw_text,
            intent_type=intent.type,
            entities=entities_str,
            primary_char=intent.primary_characteristic or "general",
            domains=", ".join(intent.domains) if intent.domains else "industrial",
        )

        data = llm.generate_json(
            prompt=prompt,
            system_prompt="You decompose industrial queries into data retrieval needs. Respond with JSON only.",
            temperature=0.0,
            max_tokens=512,
            cache_key=f"decompose:{intent.raw_text}",
        )

        if data is None:
            return

        # Enrich with LLM output (don't overwrite rule-based results, augment them)
        llm_required = data.get("required_metrics", [])
        for m in llm_required:
            if isinstance(m, str) and m not in plan.required_metrics:
                # Validate it's a real column name
                if self._is_valid_column(m, plan):
                    plan.required_metrics.append(m)

        llm_nice = data.get("nice_to_have_metrics", [])
        for m in llm_nice:
            if isinstance(m, str) and m not in plan.nice_to_have_metrics and m not in plan.required_metrics:
                if self._is_valid_column(m, plan):
                    plan.nice_to_have_metrics.append(m)

        # Causal hypothesis from LLM
        hyp = data.get("causal_hypothesis")
        if hyp and isinstance(hyp, str) and hyp.lower() != "null":
            plan.causal_hypothesis = hyp

        # Supporting questions from LLM
        sq = data.get("supporting_questions", [])
        if isinstance(sq, list):
            plan.supporting_questions = [q for q in sq if isinstance(q, str)][:3]

        # Related entity types
        related = data.get("related_entity_types", [])
        if isinstance(related, list):
            plan.related_entity_types = [r for r in related if isinstance(r, str)][:3]

        # Temporal scope override from LLM (only if different from defaults)
        llm_hours = data.get("temporal_scope_hours")
        if isinstance(llm_hours, (int, float)) and llm_hours > 0:
            llm_hours = int(llm_hours)
            # Only override if LLM suggests significantly different scope
            if abs(llm_hours - plan.temporal_scope.hours) > plan.temporal_scope.hours * 0.3:
                plan.temporal_scope.hours = llm_hours
                plan.temporal_scope.granularity = self._pick_granularity(llm_hours)
                plan.temporal_scope.rationale = f"LLM-adjusted: {llm_hours}h"

        plan.decompose_method = "llm"

    def _build_metrics_text(self, plan: RetrievalPlan) -> str:
        """Build a text description of available metrics for the LLM prompt."""
        lines = []
        seen_prefixes = set()
        for entity in plan.entities:
            prefix = plan.entity_prefixes.get(entity)
            if not prefix or prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)

            metric_map = EQUIPMENT_METRIC_MAP.get(prefix, {})
            if metric_map:
                metric_list = [f"{alias} → {col[0]} ({col[1]})"
                              for alias, col in metric_map.items()
                              if alias != "default"]
                lines.append(f"{prefix.upper()} metrics: {', '.join(metric_list)}")

        return "\n".join(lines) if lines else "Standard industrial metrics available."

    def _is_valid_column(self, column_name: str, plan: RetrievalPlan) -> bool:
        """Check if a column name is valid for any of the resolved entities."""
        for entity in plan.entities:
            prefix = plan.entity_prefixes.get(entity)
            if not prefix:
                continue
            metric_map = EQUIPMENT_METRIC_MAP.get(prefix, {})
            for alias, (col, unit) in metric_map.items():
                if col == column_name:
                    return True
        return False
