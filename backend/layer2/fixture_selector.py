"""
Fixture Selector — Context-aware fixture variant selection with diversity tracking.

Each scenario has multiple visual fixture variants. This module picks the best
variant based on the data context (e.g., critical alert → toast, percentage value
→ progress bar) and tracks usage within a layout to avoid repetition.
"""

import re
import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Per-scenario selection rules ──────────────────────────────────────────────
# Each rule is a (condition_fn, fixture_slug) tuple.
# condition_fn receives data_override dict, returns bool.
# First matching rule wins. Last entry is the fallback (always True).

def _has_key(data: dict, *keys: str) -> bool:
    """Check if any of the keys exist in data or data.demoData."""
    demo = data.get("demoData", {})
    if isinstance(demo, dict):
        for k in keys:
            if k in data or k in demo:
                return True
    return False


def _get_val(data: dict, key: str, default=None):
    """Get value from data or data.demoData."""
    demo = data.get("demoData", {})
    if isinstance(demo, dict):
        if key in demo:
            return demo[key]
    return data.get(key, default)


def _state_in(data: dict, *states: str) -> bool:
    state = _get_val(data, "state", "")
    return str(state).lower() in [s.lower() for s in states]


def _text_matches(data: dict, key: str, *patterns: str) -> bool:
    val = str(_get_val(data, key, "")).lower()
    return any(p in val for p in patterns)


# ── KPI Rules (9 variants) ───────────────────────────────────────────────────
# Ordered to prevent progress-bar from catching most % values.
# Specific matchers (alerts, electrical, status) come first.
KPI_RULES = [
    (lambda d: _state_in(d, "critical", "fault", "alarm"),
     "kpi_alert-critical-state"),
    (lambda d: _state_in(d, "warning", "caution"),
     "kpi_alert-warning-state"),
    (lambda d: _state_in(d, "offline", "stopped", "disconnected", "down"),
     "kpi_status-offline"),
    (lambda d: _text_matches(d, "label", "voltage", "current", "power", "frequency", "energy", "kw", "kva"),
     "kpi_live-high-contrast"),
    (lambda d: _text_matches(d, "label", "status", "state", "mode", "running", "online"),
     "kpi_status-badge"),
    (lambda d: _text_matches(d, "label", "utilization", "load", "capacity", "usage", "efficiency", "oee")
     or _text_matches(d, "unit", "gauge"),
     "kpi_lifecycle-dark-mode-gauge"),
    (lambda d: _text_matches(d, "label", "daily", "total", "count", "accumulated", "today", "sum", "aggregate"),
     "kpi_accumulated-daily-total"),
    # Progress-bar only for explicit progress/lifecycle scenarios (narrowed)
    (lambda d: _has_key(d, "progress")
     or (_text_matches(d, "label", "progress", "lifecycle", "remaining", "completion")
         and _text_matches(d, "unit", "%", "percent")),
     "kpi_lifecycle-progress-bar"),
    (lambda _: True,
     "kpi_live-standard"),
]

# ── Alerts Rules (5 variants) ────────────────────────────────────────────────
ALERTS_RULES = [
    (lambda d: _state_in(d, "critical", "emergency")
     or _text_matches(d, "_query_context", "critical", "emergency", "fault"),
     "modal-ups-battery-critical"),
    (lambda d: _state_in(d, "warning")
     or _text_matches(d, "_query_context", "warning", "caution", "threshold"),
     "toast-power-factor-critical-low"),
    (lambda d: _state_in(d, "success", "resolved", "ok")
     or _text_matches(d, "_query_context", "resolved", "completed", "success"),
     "card-dg-02-started-successfully"),
    (lambda d: _state_in(d, "info", "notice")
     or _text_matches(d, "_query_context", "temperature", "ahu", "hvac"),
     "badge-ahu-01-high-temperature"),
    (lambda _: True,
     "banner-energy-peak-threshold-exceeded"),
]

# ── Trend Rules (6 variants) ─────────────────────────────────────────────────
TREND_RULES = [
    (lambda d: _has_key(d, "threshold", "limit", "setpoint")
     or _text_matches(d, "_query_context", "threshold", "limit", "breach", "exceed"),
     "trend_alert_context-line-threshold"),
    (lambda d: _text_matches(d, "label", "cumulative", "energy", "consumption", "kwh")
     or _text_matches(d, "_query_context", "energy", "consumption", "kwh", "power"),
     "trend_live-area"),
    (lambda d: _text_matches(d, "label", "phase", "3-phase", "r-phase", "y-phase", "b-phase")
     or _text_matches(d, "_query_context", "phase", "3-phase", "voltage phase"),
     "trend_phased-rgb-phase-line"),
    (lambda d: _has_key(d, "heatmapData", "pattern")
     or _text_matches(d, "label", "pattern", "weekly", "monthly")
     or _text_matches(d, "_query_context", "pattern", "weekly", "monthly"),
     "trend_pattern-heatmap"),
    (lambda d: _text_matches(d, "label", "state", "status", "on/off", "binary", "discrete")
     or _text_matches(d, "_query_context", "state", "status", "on/off", "running"),
     "trend_standard-step-line"),
    (lambda _: True,
     "trend_live-line"),
]

# ── Trend-Multi-Line Rules (6 variants) ──────────────────────────────────────
TREND_MULTI_LINE_RULES = [
    (lambda d: _text_matches(d, "label", "power source", "solar", "grid", "dg", "source")
     or _text_matches(d, "_query_context", "power source", "solar", "grid supply", "dg"),
     "power-sources-stacked"),
    (lambda d: _text_matches(d, "label", "phase", "current", "lt panel", "main")
     or _text_matches(d, "_query_context", "phase", "current", "lt panel"),
     "main-lt-phases-current"),
    (lambda d: _text_matches(d, "label", "ups", "battery", "backup")
     or _text_matches(d, "_query_context", "ups", "battery", "backup"),
     "ups-health-dual-axis"),
    (lambda d: _text_matches(d, "label", "power quality", "harmonic", "thd", "pf")
     or _text_matches(d, "_query_context", "power quality", "harmonic", "thd", "power factor"),
     "power-quality"),
    (lambda d: _text_matches(d, "label", "hvac", "chiller", "ahu", "cooling", "temperature")
     or _text_matches(d, "_query_context", "hvac", "chiller", "ahu", "cooling", "temperature"),
     "hvac-performance"),
    (lambda _: True,
     "energy-demand"),
]

# ── Trends-Cumulative Rules (6 variants) ─────────────────────────────────────
TRENDS_CUMULATIVE_RULES = [
    (lambda d: _text_matches(d, "label", "instantaneous", "real-time", "live")
     or _text_matches(d, "_query_context", "instantaneous", "real-time", "live power"),
     "instantaneous-power"),
    (lambda d: _text_matches(d, "label", "source", "solar", "grid", "mix")
     or _text_matches(d, "_query_context", "source mix", "solar", "grid supply"),
     "source-mix"),
    (lambda d: _text_matches(d, "label", "baseline", "target", "benchmark", "performance")
     or _text_matches(d, "_query_context", "baseline", "target", "benchmark", "performance"),
     "performance-vs-baseline"),
    (lambda d: _text_matches(d, "label", "cost", "budget", "tariff", "expense")
     or _text_matches(d, "_query_context", "cost", "budget", "tariff", "expense"),
     "cost-vs-budget"),
    (lambda d: _text_matches(d, "label", "batch", "production", "output")
     or _text_matches(d, "_query_context", "batch", "production", "output"),
     "batch-production"),
    (lambda _: True,
     "energy-consumption"),
]

# ── Distribution Rules (6 variants) ──────────────────────────────────────────
DISTRIBUTION_RULES = [
    (lambda d: _text_matches(d, "variant", "donut", "DIST_ENERGY_SOURCE")
     or _text_matches(d, "label", "source", "share", "mix", "composition")
     or _text_matches(d, "_query_context", "energy source", "share", "mix"),
     "dist_energy_source_share-donut"),
    (lambda d: _text_matches(d, "variant", "stacked", "100")
     or _text_matches(d, "label", "proportion", "percentage", "breakdown")
     or _text_matches(d, "_query_context", "proportion", "breakdown", "percentage"),
     "dist_energy_source_share-100-stacked-bar"),
    (lambda d: _text_matches(d, "variant", "horizontal", "DIST_LOAD")
     or _text_matches(d, "label", "load", "asset", "equipment", "device")
     or _text_matches(d, "_query_context", "load", "by asset", "by equipment", "by device"),
     "dist_load_by_asset-horizontal-bar"),
    (lambda d: _text_matches(d, "variant", "pie", "DIST_CONSUMPTION_BY_CATEGORY")
     or _text_matches(d, "label", "category", "type", "classification")
     or _text_matches(d, "_query_context", "by category", "by type", "classification"),
     "dist_consumption_by_category-pie"),
    (lambda d: _text_matches(d, "variant", "grouped", "DIST_CONSUMPTION_BY_SHIFT")
     or _text_matches(d, "label", "shift", "time-of-day")
     or _text_matches(d, "_query_context", "by shift", "time-of-day", "shift-wise"),
     "dist_consumption_by_shift-grouped-bar"),
    (lambda _: True,
     "dist_downtime_top_contributors-pareto-bar"),
]

# ── Comparison Rules (6 variants) ────────────────────────────────────────────
COMPARISON_RULES = [
    (lambda d: _text_matches(d, "label", "loss", "waste", "waterfall")
     or _text_matches(d, "_query_context", "loss", "waste", "waterfall"),
     "waterfall_visual-loss-analysis"),
    (lambda d: _text_matches(d, "label", "phase", "3-phase", "r y b")
     or _text_matches(d, "_query_context", "phase", "3-phase"),
     "grouped_bar_visual-phase-comparison"),
    (lambda d: _text_matches(d, "label", "delta", "deviation", "change", "difference")
     or _text_matches(d, "_query_context", "delta", "deviation", "change", "difference"),
     "delta_bar_visual-deviation-bar"),
    (lambda d: _text_matches(d, "label", "temp", "temperature", "grid", "zone")
     or _text_matches(d, "_query_context", "temperature", "zone", "floor"),
     "small_multiples_visual-temp-grid"),
    (lambda d: _text_matches(d, "label", "load type", "composition", "split")
     or _text_matches(d, "_query_context", "load type", "composition", "split"),
     "composition_split_visual-load-type"),
    (lambda _: True,
     "side_by_side_visual-plain-values"),
]

# ── Composition Rules (5 variants) ───────────────────────────────────────────
COMPOSITION_RULES = [
    (lambda d: _text_matches(d, "label", "area", "time series", "over time", "cumulative")
     or _text_matches(d, "_query_context", "over time", "cumulative", "area"),
     "stacked_area"),
    (lambda d: _text_matches(d, "label", "donut", "pie", "share", "proportion")
     or _text_matches(d, "_query_context", "share", "proportion", "donut"),
     "donut_pie"),
    (lambda d: _text_matches(d, "label", "waterfall", "loss", "gain", "bridge")
     or _text_matches(d, "_query_context", "waterfall", "loss", "gain"),
     "waterfall"),
    (lambda d: _text_matches(d, "label", "tree", "hierarchy", "nested", "breakdown")
     or _text_matches(d, "_query_context", "hierarchy", "nested", "breakdown"),
     "treemap"),
    (lambda _: True,
     "stacked_bar"),
]

# ── Flow-Sankey Rules (5 variants) ───────────────────────────────────────────
FLOW_SANKEY_RULES = [
    # Match on label OR _query_context for contextual variant selection
    (lambda d: _text_matches(d, "label", "loss", "waste", "balance", "input vs output", "efficiency")
     or _text_matches(d, "_query_context", "loss", "waste", "balance", "input vs output", "efficiency"),
     "flow_sankey_energy_balance-sankey-with-explicit-loss-branches-dropping-out"),
    (lambda d: _text_matches(d, "label", "multi-source", "multiple source", "many", "feeding", "sources")
     or _text_matches(d, "_query_context", "multi-source", "multiple source", "sources", "feeding", "generation"),
     "flow_sankey_multi_source-many-to-one-flow-diagram"),
    (lambda d: _text_matches(d, "label", "stage", "layer", "hierarchy", "multi-stage", "drill", "plant to asset")
     or _text_matches(d, "_query_context", "stage", "layer", "hierarchy", "drill", "breakdown"),
     "flow_sankey_layered-multi-stage-hierarchical-flow"),
    (lambda d: _text_matches(d, "label", "time", "period", "daily", "hourly", "shift")
     or _text_matches(d, "_query_context", "time", "period", "daily", "hourly", "shift", "over time"),
     "flow_sankey_time_sliced-sankey-with-time-scrubberplayer"),
    (lambda _: True,
     "flow_sankey_standard-classic-left-to-right-sankey"),
]

# ── Matrix-Heatmap Rules (5 variants) ────────────────────────────────────────
MATRIX_HEATMAP_RULES = [
    (lambda d: _text_matches(d, "label", "correlation", "relationship", "vs")
     or _text_matches(d, "_query_context", "correlation", "relationship"),
     "correlation-matrix"),
    (lambda d: _text_matches(d, "label", "calendar", "daily", "monthly", "weekly")
     or _text_matches(d, "_query_context", "calendar", "daily pattern", "weekly pattern"),
     "calendar-heatmap"),
    (lambda d: _text_matches(d, "label", "status", "state", "online", "offline")
     or _text_matches(d, "_query_context", "status", "health", "online", "offline"),
     "status-matrix"),
    (lambda d: _text_matches(d, "label", "density", "distribution", "frequency")
     or _text_matches(d, "_query_context", "density", "frequency"),
     "density-matrix"),
    (lambda _: True,
     "value-heatmap"),
]

# ── Timeline Rules (5 variants) ──────────────────────────────────────────────
TIMELINE_RULES = [
    (lambda d: _text_matches(d, "label", "machine", "equipment", "state", "running", "stopped")
     or _text_matches(d, "_query_context", "machine state", "equipment state", "running", "stopped"),
     "machine-state-timeline"),
    (lambda d: _text_matches(d, "label", "shift", "schedule", "roster", "crew")
     or _text_matches(d, "_query_context", "shift", "schedule", "roster"),
     "multi-lane-shift-schedule"),
    (lambda d: _text_matches(d, "label", "forensic", "root cause", "annotated", "detailed")
     or _text_matches(d, "_query_context", "root cause", "forensic", "incident"),
     "forensic-annotated-view"),
    (lambda d: _text_matches(d, "label", "burst", "density", "frequency", "pattern")
     or _text_matches(d, "_query_context", "burst", "frequency", "pattern"),
     "log-density-burst-analysis"),
    (lambda _: True,
     "linear-incident-timeline"),
]

# ── EventLogStream Rules (5 variants) ────────────────────────────────────────
EVENTLOGSTREAM_RULES = [
    (lambda d: _text_matches(d, "label", "maintenance", "work order", "wo")
     or _text_matches(d, "_query_context", "maintenance", "work order"),
     "tabular-log-view"),
    (lambda d: _text_matches(d, "label", "correlated", "related", "cluster")
     or _text_matches(d, "_query_context", "correlated", "related", "cluster"),
     "correlation-stack"),
    (lambda d: _text_matches(d, "label", "equipment", "asset", "device", "grouped")
     or _text_matches(d, "_query_context", "by equipment", "by asset", "by device"),
     "grouped-by-asset"),
    (lambda d: _text_matches(d, "label", "compact", "brief", "summary")
     or _text_matches(d, "_query_context", "compact", "summary", "brief"),
     "compact-card-feed"),
    (lambda _: True,
     "chronological-timeline"),
]

# ── Category-Bar Rules (5 variants) ──────────────────────────────────────────
CATEGORY_BAR_RULES = [
    (lambda d: _text_matches(d, "label", "oee", "equipment", "machine", "availability")
     or _text_matches(d, "_query_context", "oee", "availability", "machine performance"),
     "oee-by-machine"),
    (lambda d: _text_matches(d, "label", "downtime", "stoppage", "breakdown")
     or _text_matches(d, "_query_context", "downtime", "stoppage", "breakdown"),
     "downtime-duration"),
    (lambda d: _text_matches(d, "label", "production", "state", "running", "idle")
     or _text_matches(d, "_query_context", "production", "output", "yield"),
     "production-states"),
    (lambda d: _text_matches(d, "label", "shift", "morning", "evening", "night")
     or _text_matches(d, "_query_context", "by shift", "shift-wise", "morning", "evening"),
     "shift-comparison"),
    (lambda _: True,
     "efficiency-deviation"),
]

# ── Master registry ──────────────────────────────────────────────────────────
# Maps scenario slug → list of (condition, fixture_slug) rules.
# Scenarios with only one variant (default-render) don't need rules.

SCENARIO_RULES = {
    "kpi":                KPI_RULES,
    "alerts":             ALERTS_RULES,
    "trend":              TREND_RULES,
    "trend-multi-line":   TREND_MULTI_LINE_RULES,
    "trends-cumulative":  TRENDS_CUMULATIVE_RULES,
    "distribution":       DISTRIBUTION_RULES,
    "comparison":         COMPARISON_RULES,
    "composition":        COMPOSITION_RULES,
    "flow-sankey":        FLOW_SANKEY_RULES,
    "matrix-heatmap":     MATRIX_HEATMAP_RULES,
    "timeline":           TIMELINE_RULES,
    "eventlogstream":     EVENTLOGSTREAM_RULES,
    "category-bar":       CATEGORY_BAR_RULES,
}

# Scenarios with only one fixture variant
SINGLE_VARIANT_SCENARIOS = {
    "chatstream":       "default-render",
    "edgedevicepanel":  "default-render",
    "peoplehexgrid":    "default-render",
    "peoplenetwork":    "default-render",
    "peopleview":       "default-render",
    "supplychainglobe": "default-render",
}


class FixtureSelector:
    """
    Selects the best fixture variant for a given scenario + data context.
    Tracks usage within a layout to penalize repetition (diversity score).

    Usage:
        selector = FixtureSelector()
        fixture = selector.select("kpi", data_override)
        fixture2 = selector.select("kpi", other_data)  # will prefer a different variant
    """

    def __init__(self):
        self._usage_counts: dict[str, int] = {}  # fixture_slug → count

    def select(self, scenario: str, data_override: Optional[dict] = None) -> str:
        """
        Select the best fixture variant for a scenario given data context.

        Args:
            scenario: Widget scenario slug (e.g., "kpi", "alerts")
            data_override: Data from RAG that will be passed to the widget

        Returns:
            fixture slug string
        """
        data = data_override or {}

        # Single-variant scenarios — no choice needed
        if scenario in SINGLE_VARIANT_SCENARIOS:
            slug = SINGLE_VARIANT_SCENARIOS[scenario]
            self._record(slug)
            return slug

        rules = SCENARIO_RULES.get(scenario)
        if not rules:
            logger.warning(f"No fixture rules for scenario '{scenario}', using fallback")
            return "default-render"

        # Collect ALL fixture slugs for this scenario (for diversity fallback)
        all_slugs = [fixture_slug for _, fixture_slug in rules]

        # Score each candidate: rule priority + diversity penalty
        candidates = []
        for priority, (condition_fn, fixture_slug) in enumerate(rules):
            try:
                matches = condition_fn(data)
            except Exception:
                matches = False

            if matches:
                # Base score: higher for earlier rules (more specific)
                base_score = len(rules) - priority
                # Diversity penalty: -5 per previous use (aggressive — forces variety fast)
                penalty = self._usage_counts.get(fixture_slug, 0) * 5
                # Freshness bonus: +2 for never-used variants (prefer new visuals)
                freshness = 2 if self._usage_counts.get(fixture_slug, 0) == 0 else 0
                final_score = base_score - penalty + freshness
                candidates.append((final_score, fixture_slug))

        if not candidates:
            # Should never happen (fallback rule always matches), but be safe
            slug = rules[-1][1]
            self._record(slug)
            return slug

        # Sort by score descending, pick the best
        candidates.sort(key=lambda c: c[0], reverse=True)
        best_score = candidates[0][0]

        # If the best score is very low, pick the least-used variant to force diversity
        if best_score <= 0:
            least_used = min(all_slugs, key=lambda s: self._usage_counts.get(s, 0))
            self._record(least_used)
            return least_used

        # Among top candidates (within 2 points of best), pick randomly for variety
        near_best = [c for c in candidates if c[0] >= best_score - 2]
        slug = random.choice(near_best)[1]

        self._record(slug)
        return slug

    def _record(self, slug: str):
        self._usage_counts[slug] = self._usage_counts.get(slug, 0) + 1

    def reset(self):
        """Reset usage tracking (call at start of each layout generation)."""
        self._usage_counts.clear()
