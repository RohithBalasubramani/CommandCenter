"""
LLM-based Fixture Selector — replaces rule-based keyword matching with
a single batched 8B LLM call that picks visual variants for all widgets
at once, enabling cross-widget diversity reasoning.

Falls back to the existing rule-based FixtureSelector on failure.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

FIXTURE_SELECT_LLM = int(os.getenv("FIXTURE_SELECT_LLM", "1"))

SYSTEM_PROMPT = """You are a visual fixture selector for an industrial dashboard.
You pick the best visual variant for each widget based on data context and dashboard story.
You MUST respond with ONLY valid JSON, no explanation or markdown."""

FIXTURE_SELECT_PROMPT = """Pick the best visual fixture variant for each widget in this dashboard.

DASHBOARD: "{heading}"
USER ASKED: "{query}"
STORY: {story}

RULES:
- Each widget has visual variants listed below — pick ONE per widget
- Pick the variant that best matches the DATA and the STORY
- Adjacent widgets should look DIFFERENT — visual diversity matters
- If unsure, pick the variant whose description best fits the data context
- For KPIs: match the alert state (critical→red, warning→amber, offline→grey) or the metric type

{widget_blocks}

Respond ONLY with JSON:
{{"fixtures": [{{"index": 0, "fixture": "<exact fixture slug>", "reason": "<brief reason>"}}, ...]}}

JSON:"""


# ── Fixture descriptions: slug → 1-line human-readable description ───────────
# Only multi-variant scenarios. Single-variant scenarios (edgedevicepanel, etc.)
# are handled directly without LLM.

FIXTURE_DESCRIPTIONS = {
    "kpi": {
        "kpi_alert-critical-state": "Red/urgent KPI card with pulsing alert — use when state is critical/fault/alarm",
        "kpi_alert-warning-state": "Amber warning KPI card — use when state is warning/caution",
        "kpi_status-offline": "Greyed-out offline indicator — use when device is stopped/disconnected",
        "kpi_live-high-contrast": "High-contrast live value card — best for electrical readings (voltage, power, kW)",
        "kpi_status-badge": "Status badge showing operational state — for status/mode/running indicators",
        "kpi_lifecycle-dark-mode-gauge": "Circular gauge dial — for load, utilization, capacity, efficiency",
        "kpi_accumulated-daily-total": "Counter-style daily total — for accumulated/daily/total counts",
        "kpi_lifecycle-progress-bar": "Progress bar — for lifecycle/completion/remaining percentage",
        "kpi_live-standard": "Default numeric display — general purpose fallback",
    },
    "alerts": {
        "modal-ups-battery-critical": "Full modal alert for critical/emergency situations",
        "toast-power-factor-critical-low": "Toast notification for warnings and threshold breaches",
        "card-dg-02-started-successfully": "Success card for resolved/completed events",
        "badge-ahu-01-high-temperature": "Badge-style alert for temperature/HVAC/info notices",
        "banner-energy-peak-threshold-exceeded": "Banner alert for energy peak/threshold events (default)",
    },
    "trend": {
        "trend_alert_context-line-threshold": "Line chart with threshold markers — when limits/breaches matter",
        "trend_live-area": "Filled area chart — best for energy/consumption/cumulative data",
        "trend_phased-rgb-phase-line": "RGB phase lines — for 3-phase electrical data",
        "trend_pattern-heatmap": "Heatmap pattern — for weekly/monthly cyclic patterns",
        "trend_standard-step-line": "Step line chart — for discrete states (on/off, running/stopped)",
        "trend_live-line": "Standard live line chart (default)",
    },
    "trend-multi-line": {
        "power-sources-stacked": "Stacked area showing power sources (solar, grid, DG)",
        "main-lt-phases-current": "Multi-line for phase currents or LT panel data",
        "ups-health-dual-axis": "Dual-axis chart for UPS/battery/backup metrics",
        "power-quality": "Multi-line for power quality (THD, harmonics, power factor)",
        "hvac-performance": "Multi-line for HVAC/chiller/cooling performance",
        "energy-demand": "Multi-line energy demand chart (default)",
    },
    "trends-cumulative": {
        "instantaneous-power": "Live instantaneous power reading with cumulative backdrop",
        "source-mix": "Cumulative by source (solar, grid, mix)",
        "performance-vs-baseline": "Actual vs target/baseline comparison over time",
        "cost-vs-budget": "Cost tracking against budget/tariff over time",
        "batch-production": "Batch/production output cumulative tracking",
        "energy-consumption": "Standard energy consumption cumulative (default)",
    },
    "distribution": {
        "dist_energy_source_share-donut": "Donut chart — energy source share/mix",
        "dist_energy_source_share-100-stacked-bar": "100% stacked bar — proportional breakdown",
        "dist_load_by_asset-horizontal-bar": "Horizontal bar — load by asset/equipment",
        "dist_consumption_by_category-pie": "Pie chart — consumption by category/type",
        "dist_consumption_by_shift-grouped-bar": "Grouped bar — consumption by shift/time-of-day",
        "dist_downtime_top_contributors-pareto-bar": "Pareto bar — top contributors/downtime (default)",
    },
    "comparison": {
        "waterfall_visual-loss-analysis": "Waterfall chart — loss/waste/gain analysis",
        "grouped_bar_visual-phase-comparison": "Grouped bars — 3-phase or multi-parameter comparison",
        "delta_bar_visual-deviation-bar": "Delta/deviation bar — highlighting differences/changes",
        "small_multiples_visual-temp-grid": "Small multiples grid — temperature zones/floors",
        "composition_split_visual-load-type": "Split composition — load type breakdown comparison",
        "side_by_side_visual-plain-values": "Side-by-side plain values (default)",
    },
    "composition": {
        "stacked_area": "Stacked area — composition over time/cumulative",
        "donut_pie": "Donut/pie — share/proportion snapshot",
        "waterfall": "Waterfall — gain/loss/bridge breakdown",
        "treemap": "Treemap — hierarchical/nested breakdown",
        "stacked_bar": "Stacked bar — category composition (default)",
    },
    "flow-sankey": {
        "flow_sankey_energy_balance-sankey-with-explicit-loss-branches-dropping-out": "Sankey with loss branches — energy balance with waste visible",
        "flow_sankey_multi_source-many-to-one-flow-diagram": "Many-to-one flow — multiple sources feeding single destination",
        "flow_sankey_layered-multi-stage-hierarchical-flow": "Multi-stage hierarchical — plant→subsystem→asset drill-down",
        "flow_sankey_time_sliced-sankey-with-time-scrubberplayer": "Time-sliced sankey with scrubber — flow over time periods",
        "flow_sankey_standard-classic-left-to-right-sankey": "Classic left-to-right sankey (default)",
    },
    "matrix-heatmap": {
        "correlation-matrix": "Correlation matrix — parameter relationships/vs comparisons",
        "calendar-heatmap": "Calendar heatmap — daily/weekly/monthly patterns",
        "status-matrix": "Status matrix — equipment health/online/offline grid",
        "density-matrix": "Density matrix — frequency/distribution visualization",
        "value-heatmap": "Value heatmap — equipment × parameter values (default)",
    },
    "timeline": {
        "machine-state-timeline": "Machine state bars — equipment running/stopped/idle over time",
        "multi-lane-shift-schedule": "Multi-lane shift schedule — crew/shift roster timeline",
        "forensic-annotated-view": "Forensic annotated — root cause/incident analysis with annotations",
        "log-density-burst-analysis": "Log density/burst — event frequency pattern analysis",
        "linear-incident-timeline": "Linear incident timeline (default)",
    },
    "eventlogstream": {
        "tabular-log-view": "Tabular log — maintenance/work order table view",
        "correlation-stack": "Correlation stack — related/clustered events grouped",
        "grouped-by-asset": "Grouped by asset — events organized by equipment",
        "compact-card-feed": "Compact card feed — brief summary cards",
        "chronological-timeline": "Chronological timeline stream (default)",
    },
    "category-bar": {
        "oee-by-machine": "OEE by machine — equipment availability/performance bars",
        "downtime-duration": "Downtime duration — stoppage/breakdown ranking",
        "production-states": "Production states — running/idle/output status bars",
        "shift-comparison": "Shift comparison — morning/evening/night shift bars",
        "efficiency-deviation": "Efficiency deviation — performance deviation bars (default)",
    },
}

# Scenarios with only one fixture variant — skip LLM
SINGLE_VARIANT_SCENARIOS = {
    "chatstream": "default-render",
    "edgedevicepanel": "default-render",
    "peoplehexgrid": "default-render",
    "peoplenetwork": "default-render",
    "peopleview": "default-render",
    "supplychainglobe": "default-render",
}

# Build a reverse lookup: fixture_slug → scenario
_VALID_FIXTURES: dict[str, set[str]] = {}
for scenario, variants in FIXTURE_DESCRIPTIONS.items():
    _VALID_FIXTURES[scenario] = set(variants.keys())


class LLMFixtureSelector:
    """Batched LLM-based fixture selection with rule-based fallback."""

    def __init__(self):
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from layer2.rag_pipeline import get_rag_pipeline
            self._pipeline = get_rag_pipeline()
        return self._pipeline

    def select_all(self, widgets: list[dict], story: str, query: str) -> list[dict]:
        """
        Select fixture variants for ALL widgets in one batched 8B LLM call.
        Modifies widgets in-place (sets w["fixture"]) and returns the list.
        Falls back to rule-based FixtureSelector per widget on failure.
        """
        if not FIXTURE_SELECT_LLM:
            return self._fallback_all(widgets)

        # Separate single-variant widgets (no LLM needed)
        multi_indices = []
        for i, w in enumerate(widgets):
            scenario = w.get("scenario", "")
            if scenario in SINGLE_VARIANT_SCENARIOS:
                w["fixture"] = SINGLE_VARIANT_SCENARIOS[scenario]
            else:
                multi_indices.append(i)

        if not multi_indices:
            return widgets

        # Build per-widget blocks for the prompt
        widget_blocks = []
        for block_idx, wi in enumerate(multi_indices):
            w = widgets[wi]
            scenario = w.get("scenario", "")
            size = w.get("size", "normal")
            dor = w.get("data_override") or {}
            demo = dor.get("demoData", {})

            # Extract compact data context
            # AUDIT FIX: Handle list-type demoData (e.g., alerts)
            if isinstance(demo, list):
                # For list data, extract info from first item or use widget-level context
                first_item = demo[0] if demo else {}
                label = first_item.get("title", "") if isinstance(first_item, dict) else ""
                state = first_item.get("severity", "") if isinstance(first_item, dict) else ""
                unit = ""
                ctx = dor.get("_query_context", "")[:120]
            else:
                label = demo.get("label", "")
                state = demo.get("state", "")
                unit = demo.get("unit", "")
                ctx = demo.get("_query_context", "")[:120]
            why = w.get("description", "")

            # Get available variants for this scenario
            variants = FIXTURE_DESCRIPTIONS.get(scenario, {})
            if not variants:
                # Unknown scenario — assign default
                w["fixture"] = "default-render"
                continue

            variant_lines = []
            for slug, desc in variants.items():
                variant_lines.append(f'    - "{slug}": {desc}')

            block = (
                f"Widget {block_idx}: {scenario} ({size})\n"
                f"  Purpose: {why}\n"
                f"  Data: label=\"{label}\", state=\"{state}\", unit=\"{unit}\"\n"
                f"  Context: {ctx}\n"
                f"  Variants:\n" + "\n".join(variant_lines)
            )
            widget_blocks.append(block)

        if not widget_blocks:
            return widgets

        # Build heading from first widget or query
        heading = query[:60] if query else "Dashboard"

        prompt = FIXTURE_SELECT_PROMPT.format(
            heading=heading,
            query=query[:100],
            story=story[:200],
            widget_blocks="\n\n".join(widget_blocks),
        )

        try:
            llm = self.pipeline.llm_fast
            data = llm.generate_json(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.0,  # AUDIT FIX: Deterministic fixture selection
                max_tokens=1024,
            )

            if data and "fixtures" in data:
                fixtures = data["fixtures"]
                for entry in fixtures:
                    block_idx = entry.get("index", -1)
                    fixture_slug = entry.get("fixture", "")
                    if 0 <= block_idx < len(multi_indices):
                        wi = multi_indices[block_idx]
                        scenario = widgets[wi].get("scenario", "")
                        # Validate the fixture exists for this scenario
                        valid = _VALID_FIXTURES.get(scenario, set())
                        if fixture_slug in valid:
                            widgets[wi]["fixture"] = fixture_slug
                            logger.info(f"LLM fixture: {scenario} → {fixture_slug} ({entry.get('reason', '')})")
                        else:
                            logger.warning(f"LLM returned invalid fixture '{fixture_slug}' for {scenario}")

        except Exception as e:
            logger.warning(f"LLM fixture selection failed: {e}")

        # Fill any widgets that still lack a fixture with rule-based fallback
        return self._fill_missing(widgets)

    def _fill_missing(self, widgets: list[dict]) -> list[dict]:
        """Use rule-based fallback for any widget without a fixture."""
        from layer2.fixture_selector import FixtureSelector
        fallback = FixtureSelector()
        for w in widgets:
            if not w.get("fixture"):
                w["fixture"] = fallback.select(w["scenario"], w.get("data_override") or {})
                logger.debug(f"Fallback fixture: {w['scenario']} → {w['fixture']}")
        return widgets

    def _fallback_all(self, widgets: list[dict]) -> list[dict]:
        """Use rule-based selection for ALL widgets (FIXTURE_SELECT_LLM=0)."""
        from layer2.fixture_selector import FixtureSelector
        fallback = FixtureSelector()
        for w in widgets:
            w["fixture"] = fallback.select(w["scenario"], w.get("data_override") or {})
        return widgets
