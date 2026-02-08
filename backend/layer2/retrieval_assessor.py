"""
Retrieval Assessor — Phase 1 of RAG & Widget Selection Redesign.

Evaluates the quality of collected widget data AFTER the data collector runs,
producing a per-widget and aggregate assessment that drives:
1. Widget dropping (no real data → widget removed from layout)
2. Data-aware sizing (few data points → smaller widget)
3. Staleness detection (old timestamps → visual indicator)
4. Confidence scoring (feeds into voice response caveats)

This is purely deterministic — no LLM calls. Runs in <50ms.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Freshness thresholds (seconds) — how old data can be before considered stale
FRESHNESS_THRESHOLDS = {
    "real_time": 300,       # 5 min — live sensors, SCADA
    "near_real_time": 900,  # 15 min — energy meters, aggregated readings
    "periodic": 3600,       # 1 hour — shift summaries, batch data
    "historical": 86400,    # 24 hours — maintenance records, work orders
}

# Equipment type → expected freshness category
EQUIPMENT_FRESHNESS = {
    "trf": "real_time", "dg": "real_time", "ups": "real_time",
    "chiller": "real_time", "ahu": "real_time", "ct": "real_time",
    "pump": "real_time", "compressor": "real_time", "motor": "real_time",
    "em": "near_real_time",
    "lt_mcc": "near_real_time", "lt_pcc": "near_real_time",
    "lt_apfc": "near_real_time", "lt_db": "near_real_time",
    "lt_vfd": "near_real_time", "lt_plc": "near_real_time",
    "lt_ats": "near_real_time", "lt_changeover": "near_real_time",
}

# Minimum data points for meaningful visualization
# Kept very low — user wants all relevant widgets shown even with sparse data.
# Widgets with low data still render (with quality badges) rather than being dropped.
MIN_TREND_POINTS = 0
MIN_COMPARISON_ENTITIES = 1
MIN_DISTRIBUTION_SEGMENTS = 1
MIN_MATRIX_CELLS = 1
MIN_ALERT_COUNT = 0
MIN_TIMELINE_EVENTS = 0

# Scenarios that absolutely require real data to be useful
# Only KPI truly needs a real value — other scenarios can still show structure/layout
REQUIRES_REAL_DATA = {
    "kpi",
}

# Scenarios that can still be useful with empty/demo data
TOLERATES_EMPTY = {
    "edgedevicepanel",  # Shows device metadata even without timeseries
    "chatstream",       # Always synthetic
    "peopleview", "peoplehexgrid", "peoplenetwork",  # Demo domain
    "supplychainglobe",  # Demo domain
}


@dataclass
class WidgetAssessment:
    """Quality assessment for a single widget's data."""
    scenario: str
    has_real_data: bool = False
    is_synthetic: bool = False
    is_stale: bool = False
    staleness_seconds: int = 0
    data_point_count: int = 0
    entity_count: int = 0
    freshness_category: str = "unknown"
    data_quality: str = "missing"  # "real_time" | "recent" | "stale" | "synthetic" | "missing"
    should_drop: bool = False
    drop_reason: str = ""
    recommended_size: Optional[str] = None  # Override size if data is sparse
    confidence: float = 0.0  # 0-1 per-widget confidence

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "has_real_data": self.has_real_data,
            "is_synthetic": self.is_synthetic,
            "is_stale": self.is_stale,
            "staleness_seconds": self.staleness_seconds,
            "data_point_count": self.data_point_count,
            "data_quality": self.data_quality,
            "should_drop": self.should_drop,
            "drop_reason": self.drop_reason,
            "recommended_size": self.recommended_size,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class RetrievalAssessment:
    """Aggregate assessment of all retrieved data for a dashboard."""
    widget_assessments: list[WidgetAssessment] = field(default_factory=list)
    total_widgets: int = 0
    real_data_widgets: int = 0
    synthetic_widgets: int = 0
    stale_widgets: int = 0
    widgets_to_drop: int = 0
    completeness: float = 0.0       # Fraction of widgets with real data
    freshness: float = 1.0          # Min freshness across all widgets (0-1)
    data_fill_quality: float = 0.0  # real_data_widgets / total_widgets
    gaps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_widgets": self.total_widgets,
            "real_data_widgets": self.real_data_widgets,
            "synthetic_widgets": self.synthetic_widgets,
            "stale_widgets": self.stale_widgets,
            "widgets_to_drop": self.widgets_to_drop,
            "completeness": round(self.completeness, 2),
            "freshness": round(self.freshness, 2),
            "data_fill_quality": round(self.data_fill_quality, 2),
            "gaps": self.gaps,
            "warnings": self.warnings,
        }


class RetrievalAssessor:
    """
    Evaluates quality of collected widget data after data collection.

    Runs deterministically — no LLM calls, no DB queries.
    Input: list of widget_data dicts (output of SchemaDataCollector.collect_all)
    Output: RetrievalAssessment with per-widget and aggregate quality scores.
    """

    def assess(self, widget_data: list[dict]) -> RetrievalAssessment:
        """
        Assess the quality of all collected widget data.

        Args:
            widget_data: Output of SchemaDataCollector.collect_all() — list of dicts with
                        scenario, size, data_override, relevance, why, schema_valid

        Returns:
            RetrievalAssessment with per-widget assessments and aggregate scores.
        """
        assessment = RetrievalAssessment(total_widgets=len(widget_data))
        freshness_scores = []

        for w in widget_data:
            wa = self._assess_widget(w)
            assessment.widget_assessments.append(wa)

            if wa.has_real_data and not wa.is_synthetic:
                assessment.real_data_widgets += 1
            if wa.is_synthetic:
                assessment.synthetic_widgets += 1
            if wa.is_stale:
                assessment.stale_widgets += 1
            if wa.should_drop:
                assessment.widgets_to_drop += 1
                assessment.gaps.append(wa.drop_reason)

            # Freshness score: 1.0 = real-time, decays toward 0 as staleness increases
            if wa.has_real_data:
                max_stale = FRESHNESS_THRESHOLDS.get(wa.freshness_category, 3600)
                f_score = max(0.0, 1.0 - (wa.staleness_seconds / (max_stale * 3)))
                freshness_scores.append(f_score)

        # Aggregate scores
        if assessment.total_widgets > 0:
            assessment.completeness = assessment.real_data_widgets / assessment.total_widgets
            assessment.data_fill_quality = assessment.real_data_widgets / assessment.total_widgets

        if freshness_scores:
            assessment.freshness = min(freshness_scores)
        else:
            assessment.freshness = 0.0

        # Generate warnings
        if assessment.stale_widgets > 0:
            assessment.warnings.append(
                f"{assessment.stale_widgets} widget(s) have stale data"
            )
        if assessment.synthetic_widgets > assessment.total_widgets * 0.5:
            assessment.warnings.append(
                "More than half of widgets have synthetic/demo data"
            )

        return assessment

    def _assess_widget(self, w: dict) -> WidgetAssessment:
        """Assess a single widget's data quality."""
        scenario = w.get("scenario", "")
        data_override = w.get("data_override", {})
        schema_valid = w.get("schema_valid", True)
        size = w.get("size", "normal")

        wa = WidgetAssessment(scenario=scenario)

        if not data_override or not isinstance(data_override, dict):
            wa.data_quality = "missing"
            wa.should_drop = scenario in REQUIRES_REAL_DATA
            wa.drop_reason = f"No data collected for {scenario} widget"
            wa.confidence = 0.0
            return wa

        # Check for synthetic/demo markers
        wa.is_synthetic = self._check_synthetic(data_override)

        # Check for real PG timeseries data source
        wa.has_real_data = self._check_real_data(data_override)

        # Extract and assess data density
        wa.data_point_count = self._count_data_points(scenario, data_override)
        wa.entity_count = self._count_entities(scenario, data_override)

        # Check freshness
        timestamp = self._extract_latest_timestamp(data_override)
        if timestamp:
            wa.staleness_seconds = self._compute_staleness(timestamp)
            wa.freshness_category = self._infer_freshness_category(data_override)
            threshold = FRESHNESS_THRESHOLDS.get(wa.freshness_category, 3600)
            wa.is_stale = wa.staleness_seconds > threshold

        # Determine data quality category
        if wa.is_synthetic and not wa.has_real_data:
            wa.data_quality = "synthetic"
        elif wa.is_stale:
            wa.data_quality = "stale"
        elif wa.has_real_data:
            wa.data_quality = "real_time" if wa.staleness_seconds < 300 else "recent"
        else:
            wa.data_quality = "missing"

        # Decide if widget should be dropped
        wa.should_drop, wa.drop_reason = self._should_drop(
            scenario, wa, data_override, schema_valid
        )

        # Compute recommended size based on data density
        wa.recommended_size = self._recommend_size(scenario, wa, size)

        # Compute per-widget confidence
        wa.confidence = self._compute_confidence(wa)

        return wa

    def _check_synthetic(self, data: dict) -> bool:
        """Check if data contains synthetic/demo markers at any level."""
        if data.get("_synthetic"):
            return True
        if data.get("_data_source") in ("no_data_fallback", "widget_demo_shape"):
            return True

        demo = data.get("demoData", {})
        if isinstance(demo, dict):
            if demo.get("_synthetic"):
                return True
            if demo.get("_data_source") in ("no_data_fallback", "widget_demo_shape"):
                return True
            # Check for "N/A" values which indicate no real data
            if demo.get("value") == "N/A":
                return True

        return False

    def _check_real_data(self, data: dict) -> bool:
        """Check if data comes from a real data source (PG timeseries, etc.)."""
        source = data.get("_data_source", "")
        if not source:
            demo = data.get("demoData", {})
            if isinstance(demo, dict):
                source = demo.get("_data_source", "")

        if not source:
            return False

        real_prefixes = ("pg_timeseries:", "django.industrial", "sqlite:")
        return any(source.startswith(p) for p in real_prefixes)

    def _count_data_points(self, scenario: str, data: dict) -> int:
        """Count the number of meaningful data points in widget data."""
        demo = data.get("demoData", data)
        if not isinstance(demo, dict) and not isinstance(demo, list):
            return 0

        if scenario in ("trend", "trend-multi-line", "trends-cumulative"):
            # Time series data
            ts = None
            if isinstance(demo, dict):
                ts = demo.get("timeSeries") or demo.get("data", [])
                if not ts:
                    # Check for series array (multi-line)
                    series = demo.get("series", [])
                    if series and isinstance(series, list):
                        return sum(
                            len(s.get("data", []))
                            for s in series
                            if isinstance(s, dict)
                        )
            if isinstance(ts, list):
                return len(ts)
            return 0

        elif scenario == "alerts":
            if isinstance(demo, list):
                return len(demo)
            alerts = demo.get("alerts", []) if isinstance(demo, dict) else []
            return len(alerts) if isinstance(alerts, list) else 0

        elif scenario == "comparison":
            # Count how many comparison pairs have real values
            if isinstance(demo, dict):
                has_a = demo.get("valueA") is not None
                has_b = demo.get("valueB") is not None
                return (1 if has_a else 0) + (1 if has_b else 0)
            return 0

        elif scenario in ("distribution", "composition", "category-bar"):
            if isinstance(demo, dict):
                series = demo.get("series", [])
                categories = demo.get("categories", [])
                values = demo.get("values", [])
                return max(len(series), len(categories), len(values))
            return 0

        elif scenario in ("timeline", "eventlogstream"):
            if isinstance(demo, dict):
                events = demo.get("events", [])
                return len(events) if isinstance(events, list) else 0
            return 0

        elif scenario == "matrix-heatmap":
            if isinstance(demo, dict):
                dataset = demo.get("dataset", [])
                return len(dataset) if isinstance(dataset, list) else 0
            return 0

        elif scenario == "flow-sankey":
            if isinstance(demo, dict):
                nodes = demo.get("nodes", [])
                return len(nodes) if isinstance(nodes, list) else 0
            return 0

        elif scenario == "kpi":
            if isinstance(demo, dict):
                val = demo.get("value")
                return 1 if val is not None and val != "N/A" else 0
            return 0

        elif scenario == "edgedevicepanel":
            if isinstance(demo, dict):
                device = demo.get("device", {})
                return 1 if device else 0
            return 0

        return 1 if demo else 0

    def _count_entities(self, scenario: str, data: dict) -> int:
        """Count distinct entities referenced in widget data."""
        demo = data.get("demoData", data)
        if not isinstance(demo, dict):
            return 0

        if scenario == "comparison":
            entities = set()
            if demo.get("labelA"):
                entities.add(demo["labelA"])
            if demo.get("labelB"):
                entities.add(demo["labelB"])
            return len(entities)

        elif scenario in ("trend-multi-line",):
            series = demo.get("series", [])
            return len(series) if isinstance(series, list) else 0

        elif scenario in ("category-bar", "distribution", "composition"):
            categories = demo.get("categories", demo.get("series", []))
            return len(categories) if isinstance(categories, list) else 0

        elif scenario == "matrix-heatmap":
            dataset = demo.get("dataset", [])
            labels = set()
            for row in (dataset if isinstance(dataset, list) else []):
                if isinstance(row, dict) and row.get("label"):
                    labels.add(row["label"])
            return len(labels)

        return 1

    def _extract_latest_timestamp(self, data: dict) -> Optional[str]:
        """Extract the most recent timestamp from widget data."""
        demo = data.get("demoData", data)

        # Direct timestamp field
        for key in ("timestamp", "_timestamp", "lastUpdated"):
            ts = data.get(key) or (demo.get(key) if isinstance(demo, dict) else None)
            if ts:
                return str(ts)

        # From timeseries data
        if isinstance(demo, dict):
            ts = demo.get("timeSeries", [])
            if isinstance(ts, list) and ts:
                last = ts[-1]
                if isinstance(last, dict):
                    return last.get("time") or last.get("timestamp")

        return None

    def _compute_staleness(self, timestamp_str: str) -> int:
        """Compute staleness in seconds from a timestamp string."""
        try:
            # Try ISO format
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = (now - ts).total_seconds()
            return max(0, int(delta))
        except (ValueError, TypeError):
            return 0

    def _infer_freshness_category(self, data: dict) -> str:
        """Infer the expected freshness category from data source."""
        source = data.get("_data_source", "")
        demo = data.get("demoData", {})
        if isinstance(demo, dict) and not source:
            source = demo.get("_data_source", "")

        if not source:
            return "periodic"

        # Extract equipment prefix from pg_timeseries source
        if source.startswith("pg_timeseries:"):
            table = source.split(":")[1]
            prefix = re.match(r'^([a-z_]+)_\d+$', table)
            if prefix:
                return EQUIPMENT_FRESHNESS.get(prefix.group(1), "near_real_time")

        return "periodic"

    def _should_drop(self, scenario: str, wa: WidgetAssessment,
                     data: dict, schema_valid: bool) -> tuple[bool, str]:
        """Decide if a widget should be dropped from the layout."""

        # Never drop tolerant scenarios
        if scenario in TOLERATES_EMPTY:
            return False, ""

        # Drop if schema validation failed AND data is synthetic
        if not schema_valid and wa.is_synthetic:
            return True, f"Schema validation failed for {scenario} with synthetic data"

        # Drop widgets requiring real data that only have synthetic
        if scenario in REQUIRES_REAL_DATA and wa.is_synthetic and not wa.has_real_data:
            return True, f"{scenario} requires real data but only has synthetic/demo data"

        # Drop empty time series widgets
        if scenario in ("trend", "trend-multi-line", "trends-cumulative"):
            if wa.data_point_count < MIN_TREND_POINTS:
                return True, f"{scenario} has only {wa.data_point_count} data points (minimum {MIN_TREND_POINTS})"

        # Drop empty comparison widgets
        if scenario == "comparison" and wa.data_point_count < MIN_COMPARISON_ENTITIES:
            return True, f"Comparison has data for only {wa.data_point_count} entities (need {MIN_COMPARISON_ENTITIES})"

        # Drop empty distribution/composition
        if scenario in ("distribution", "composition", "category-bar"):
            if wa.data_point_count < MIN_DISTRIBUTION_SEGMENTS:
                return True, f"{scenario} has only {wa.data_point_count} segments (minimum {MIN_DISTRIBUTION_SEGMENTS})"

        # Drop empty matrix
        if scenario == "matrix-heatmap" and wa.data_point_count < MIN_MATRIX_CELLS:
            return True, f"Matrix has only {wa.data_point_count} cells (minimum {MIN_MATRIX_CELLS})"

        # Drop alert widgets with 0 alerts
        if scenario == "alerts" and wa.data_point_count < MIN_ALERT_COUNT:
            return True, f"No active alerts to display"

        # Drop timeline/eventlog with no events
        if scenario in ("timeline", "eventlogstream") and wa.data_point_count < MIN_TIMELINE_EVENTS:
            return True, f"No events to display in {scenario}"

        # Drop KPI with N/A value
        if scenario == "kpi" and wa.data_point_count == 0:
            return True, f"KPI has no valid value"

        return False, ""

    def _recommend_size(self, scenario: str, wa: WidgetAssessment,
                        current_size: str) -> Optional[str]:
        """Recommend a size based on actual data density.

        Returns None if current size is appropriate, or a new size string.
        """
        if current_size == "hero":
            # Hero sizing: downsize if data is too sparse for hero treatment
            if scenario in ("trend", "trend-multi-line", "trends-cumulative"):
                if wa.data_point_count < 10:
                    return "expanded"
                if wa.data_point_count < 20:
                    return None  # Keep hero but it's borderline
            if scenario == "comparison" and wa.entity_count < 2:
                return "expanded"
            if scenario == "matrix-heatmap" and wa.data_point_count < 6:
                return "expanded"
            return None

        if current_size == "expanded":
            # Downsize expanded to normal if sparse
            if scenario in ("trend", "trend-multi-line"):
                if wa.data_point_count < 8:
                    return "normal"
            return None

        if current_size == "normal":
            # Downsize normal to compact if very sparse
            if scenario in ("distribution", "composition"):
                if wa.data_point_count <= 2:
                    return "compact"
            return None

        return None

    def _compute_confidence(self, wa: WidgetAssessment) -> float:
        """Compute per-widget confidence score."""
        if wa.should_drop:
            return 0.0

        score = 0.0

        # Data source quality
        if wa.has_real_data and not wa.is_synthetic:
            score += 0.5
        elif wa.has_real_data:
            score += 0.3
        elif wa.is_synthetic:
            score += 0.1

        # Data density
        if wa.data_point_count > 20:
            score += 0.3
        elif wa.data_point_count > 5:
            score += 0.2
        elif wa.data_point_count > 0:
            score += 0.1

        # Freshness
        if not wa.is_stale:
            score += 0.2
        elif wa.staleness_seconds < 7200:  # < 2 hours
            score += 0.1

        return min(1.0, score)


def apply_assessment_to_widgets(widget_data: list[dict],
                                 assessment: RetrievalAssessment) -> list[dict]:
    """
    Apply retrieval assessment to widget data:
    1. Drop widgets flagged for removal
    2. Resize widgets based on data density
    3. Inject quality metadata for frontend indicators

    Args:
        widget_data: List of widget dicts from data collection
        assessment: RetrievalAssessment from RetrievalAssessor

    Returns:
        Filtered and resized widget_data list
    """
    if len(widget_data) != len(assessment.widget_assessments):
        logger.warning(
            f"[assessor] Widget count mismatch: {len(widget_data)} widgets, "
            f"{len(assessment.widget_assessments)} assessments"
        )
        return widget_data

    result = []
    dropped = []
    resized = []

    for w, wa in zip(widget_data, assessment.widget_assessments):
        # Drop widgets with insufficient data
        if wa.should_drop:
            dropped.append(f"{wa.scenario} ({wa.drop_reason})")
            continue

        # Apply recommended size
        if wa.recommended_size and wa.recommended_size != w.get("size"):
            old_size = w.get("size", "normal")
            w["size"] = wa.recommended_size
            resized.append(f"{wa.scenario}: {old_size} → {wa.recommended_size}")

        # Inject quality metadata for frontend
        w["_data_quality"] = wa.data_quality
        w["_is_stale"] = wa.is_stale
        w["_staleness_seconds"] = wa.staleness_seconds
        w["_widget_confidence"] = wa.confidence
        w["_is_synthetic"] = wa.is_synthetic

        result.append(w)

    if dropped:
        logger.info(
            f"[assessor] Dropped {len(dropped)} widget(s): {'; '.join(dropped)}"
        )
    if resized:
        logger.info(
            f"[assessor] Resized {len(resized)} widget(s): {'; '.join(resized)}"
        )

    return result
