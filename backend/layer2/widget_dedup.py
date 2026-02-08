"""
Widget Deduplication — Phase 1 of RAG & Widget Selection Redesign.

Detects and eliminates redundant widgets in a dashboard layout by:
1. Hashing each widget's resolved (entity, metric, time_range) tuple
2. Detecting contradictions between widgets (e.g., alert says critical but KPI says normal)
3. Collapsing multiple similar KPIs into a single comparison or category-bar

This runs AFTER data collection, so it operates on resolved data — not LLM-generated strings.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Visualization categories — widgets in the SAME category showing the same entity+metric
# are considered duplicates. Widgets in DIFFERENT categories are complementary views.
VIZ_CATEGORY = {
    "kpi": "snapshot",
    "trend": "timeseries",
    "trend-multi-line": "timeseries",
    "trends-cumulative": "timeseries",
    "comparison": "comparison",
    "distribution": "distribution",
    "composition": "composition",
    "category-bar": "categorical",
    "alerts": "alerts",
    "timeline": "events",
    "eventlogstream": "events",
    "edgedevicepanel": "device",
    "matrix-heatmap": "matrix",
    "flow-sankey": "flow",
    "diagnosticpanel": "diagnostic",
    "uncertaintypanel": "uncertainty",
    "chatstream": "chat",
    "peopleview": "people",
    "peoplehexgrid": "people",
    "peoplenetwork": "people",
    "supplychainglobe": "globe",
}


@dataclass
class WidgetSignature:
    """Unique data signature for a widget, used for redundancy detection."""
    scenario: str
    entities: frozenset  # Set of entity identifiers
    metric_column: str   # Resolved PG column name
    data_source: str     # e.g., "pg_timeseries:trf_001"
    is_timeseries: bool  # Whether this shows temporal data

    @property
    def viz_category(self) -> str:
        return VIZ_CATEGORY.get(self.scenario, self.scenario)

    @property
    def dedup_key(self) -> str:
        """Key for deduplication — widgets with same key show the same data.

        Includes viz_category so different visualization types for the same
        entity/metric are kept (they show complementary views).
        """
        entities_str = ",".join(sorted(self.entities)) if self.entities else "none"
        return f"{self.viz_category}|{entities_str}|{self.metric_column}"


@dataclass
class DeduplicationResult:
    """Result of widget deduplication."""
    kept: list[dict] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)
    drop_reasons: list[str] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)


def extract_signature(widget: dict) -> WidgetSignature:
    """
    Extract the data signature from a collected widget.

    Examines data_override to determine what entity/metric/source this widget
    actually shows, regardless of what the LLM requested.
    """
    scenario = widget.get("scenario", "")
    data_override = widget.get("data_override", {})
    demo = data_override.get("demoData", data_override) if isinstance(data_override, dict) else {}

    entities = set()
    metric_column = ""
    data_source = ""
    is_timeseries = scenario in ("trend", "trend-multi-line", "trends-cumulative")

    if isinstance(data_override, dict):
        data_source = data_override.get("_data_source", "")
        if not data_source and isinstance(demo, dict):
            data_source = demo.get("_data_source", "")

    if isinstance(demo, dict):
        # Extract entities from various widget data shapes
        if scenario == "kpi":
            label = demo.get("label", "")
            if label and label != "N/A" and label != "Unknown":
                entities.add(_normalize_entity(label))
            # Extract metric from data source or value context
            table = demo.get("_table", "")
            if table:
                entities.add(table)
            metric_column = _extract_metric_from_source(data_source, demo)

        elif scenario == "comparison":
            for key in ("labelA", "labelB"):
                val = demo.get(key, "")
                if val:
                    entities.add(_normalize_entity(val))
            metric_column = demo.get("unit", "")

        elif scenario in ("trend", "trend-multi-line", "trends-cumulative"):
            label = demo.get("label", "")
            if label:
                entities.add(_normalize_entity(label))
            metric_column = _extract_metric_from_source(data_source, demo)
            # Check for series
            series = demo.get("series", [])
            for s in (series if isinstance(series, list) else []):
                if isinstance(s, dict) and s.get("label"):
                    entities.add(_normalize_entity(s["label"]))

        elif scenario == "alerts":
            # Alerts are scoped by entity
            if isinstance(demo, list):
                for alert in demo:
                    if isinstance(alert, dict):
                        src = alert.get("source", "")
                        if src:
                            entities.add(_normalize_entity(src))
            elif isinstance(demo, dict):
                alerts_list = demo.get("alerts", [])
                for alert in (alerts_list if isinstance(alerts_list, list) else []):
                    if isinstance(alert, dict):
                        src = alert.get("source", "")
                        if src:
                            entities.add(_normalize_entity(src))

        elif scenario == "edgedevicepanel":
            device = demo.get("device", {})
            if isinstance(device, dict):
                name = device.get("name", device.get("id", ""))
                if name:
                    entities.add(_normalize_entity(name))

        elif scenario in ("distribution", "composition", "category-bar"):
            label = demo.get("label", "")
            if label:
                entities.add(_normalize_entity(label))
            metric_column = demo.get("unit", "")

        elif scenario == "matrix-heatmap":
            label = demo.get("label", "")
            if label:
                entities.add(_normalize_entity(label))

        elif scenario == "flow-sankey":
            label = demo.get("label", "")
            if label:
                entities.add(_normalize_entity(label))

    return WidgetSignature(
        scenario=scenario,
        entities=frozenset(entities),
        metric_column=metric_column,
        data_source=data_source,
        is_timeseries=is_timeseries,
    )


def deduplicate_widgets(widget_data: list[dict]) -> DeduplicationResult:
    """
    Remove redundant widgets from the layout.

    Rules:
    1. If two widgets show the exact same (entities, metric, is_timeseries), keep higher relevance
    2. If a KPI and a trend show the same entity+metric, keep the trend (more informative)
    3. If two KPIs show the same entity but different metrics, keep both (they're complementary)
    4. Hero widgets are never dropped due to redundancy
    """
    result = DeduplicationResult()

    if not widget_data:
        return result

    signatures = [extract_signature(w) for w in widget_data]
    seen_keys: dict[str, int] = {}  # dedup_key → index of kept widget
    keep_indices = set()

    for i, (w, sig) in enumerate(zip(widget_data, signatures)):
        size = w.get("size", "normal")
        relevance = w.get("relevance", 0.5)

        # Hero widgets are always kept
        if size == "hero":
            keep_indices.add(i)
            seen_keys[sig.dedup_key] = i
            continue

        key = sig.dedup_key

        if key in seen_keys:
            prev_idx = seen_keys[key]
            prev_w = widget_data[prev_idx]
            prev_relevance = prev_w.get("relevance", 0.5)
            prev_size = prev_w.get("size", "normal")

            # Keep the more informative version
            should_replace = False

            # Trend beats KPI for same data
            if sig.scenario in ("trend", "trend-multi-line") and \
               widget_data[prev_idx].get("scenario") == "kpi":
                should_replace = True
            # Higher relevance wins if same scenario type
            elif sig.scenario == widget_data[prev_idx].get("scenario") and \
                 relevance > prev_relevance:
                should_replace = True

            if should_replace and prev_size != "hero":
                # Replace previous with current
                keep_indices.discard(prev_idx)
                keep_indices.add(i)
                seen_keys[key] = i
                result.dropped.append(prev_w)
                result.drop_reasons.append(
                    f"Redundant: {prev_w.get('scenario')} replaced by "
                    f"{w.get('scenario')} (same data: {key})"
                )
            else:
                # Current is redundant
                result.dropped.append(w)
                result.drop_reasons.append(
                    f"Redundant: {w.get('scenario')} duplicates "
                    f"{prev_w.get('scenario')} (same data: {key})"
                )
        else:
            keep_indices.add(i)
            seen_keys[key] = i

    # Build final list preserving original order
    result.kept = [widget_data[i] for i in sorted(keep_indices)]

    # Detect contradictions
    result.contradictions = _detect_contradictions(result.kept, signatures)

    if result.dropped:
        logger.info(
            f"[dedup] Dropped {len(result.dropped)} redundant widget(s): "
            f"{'; '.join(result.drop_reasons)}"
        )

    if result.contradictions:
        logger.warning(
            f"[dedup] Detected {len(result.contradictions)} contradiction(s) "
            f"between widgets"
        )

    return result


def _detect_contradictions(widgets: list[dict],
                            signatures: list[WidgetSignature]) -> list[dict]:
    """
    Detect contradictions between widgets showing the same entity.

    Example: Alert widget says "critical" for TRF-1, but KPI shows "normal" state.
    """
    contradictions = []

    # Collect entity states from different widget types
    entity_states: dict[str, list[dict]] = {}  # entity → [{scenario, state, source}]

    for w in widgets:
        scenario = w.get("scenario", "")
        data_override = w.get("data_override", {})
        demo = data_override.get("demoData", data_override) if isinstance(data_override, dict) else {}

        if scenario == "kpi" and isinstance(demo, dict):
            entity = _normalize_entity(demo.get("label", ""))
            state = demo.get("state", "normal")
            if entity:
                entity_states.setdefault(entity, []).append({
                    "scenario": "kpi",
                    "state": state,
                    "widget_idx": widgets.index(w),
                })

        elif scenario == "alerts":
            if isinstance(demo, list):
                alert_list = demo
            elif isinstance(demo, dict):
                alert_list = demo.get("alerts", [])
            else:
                alert_list = []

            for alert in (alert_list if isinstance(alert_list, list) else []):
                if isinstance(alert, dict):
                    entity = _normalize_entity(alert.get("source", ""))
                    severity = alert.get("severity", "info")
                    if entity:
                        alert_state = "critical" if severity in ("critical", "high") else \
                                     "warning" if severity == "warning" else "normal"
                        entity_states.setdefault(entity, []).append({
                            "scenario": "alerts",
                            "state": alert_state,
                            "severity": severity,
                            "widget_idx": widgets.index(w),
                        })

    # Check for contradictions
    for entity, states in entity_states.items():
        if len(states) < 2:
            continue

        # Look for KPI showing "normal" while alerts show "critical" or "warning"
        kpi_states = [s for s in states if s["scenario"] == "kpi"]
        alert_states = [s for s in states if s["scenario"] == "alerts"]

        for kpi in kpi_states:
            for alert in alert_states:
                if kpi["state"] == "normal" and alert["state"] in ("critical", "warning"):
                    contradictions.append({
                        "entity": entity,
                        "kpi_state": kpi["state"],
                        "alert_state": alert["state"],
                        "kpi_widget_idx": kpi["widget_idx"],
                        "alert_widget_idx": alert["widget_idx"],
                        "message": (
                            f"{entity}: KPI shows 'normal' but alerts show "
                            f"'{alert.get('severity', 'warning')}' severity"
                        ),
                    })

    return contradictions


def inject_contradiction_flags(widgets: list[dict],
                                contradictions: list[dict]) -> list[dict]:
    """
    Inject contradiction flags into widget data so the frontend can display them.
    """
    if not contradictions:
        return widgets

    # Build a set of widget indices involved in contradictions
    conflict_map: dict[int, str] = {}  # widget_idx → conflict message
    for c in contradictions:
        if "kpi_widget_idx" in c:
            conflict_map[c["kpi_widget_idx"]] = c["message"]
        if "alert_widget_idx" in c:
            conflict_map[c["alert_widget_idx"]] = c["message"]

    for i, w in enumerate(widgets):
        if i in conflict_map:
            w["_conflict_flag"] = conflict_map[i]

    return widgets


def _normalize_entity(name: str) -> str:
    """Normalize entity name for comparison."""
    if not name:
        return ""
    # Lowercase, strip whitespace, normalize separators
    normalized = name.lower().strip()
    # Remove common suffixes
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def _extract_metric_from_source(data_source: str, demo: dict) -> str:
    """Extract the metric column name from a data source identifier."""
    # From pg_timeseries source: "pg_timeseries:trf_001" — metric is in the data
    unit = ""
    if isinstance(demo, dict):
        unit = demo.get("unit", "")
    return unit or ""
