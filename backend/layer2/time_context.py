"""
Time-Travel + Comparison Focus — Upgrade 3

Resolves natural language time references to concrete TimeWindows,
captures temporal snapshots, and computes deltas between time periods.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TimeReference(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    EVENT_ANCHORED = "event_anchored"
    COMPARATIVE = "comparative"


@dataclass
class TimeWindow:
    start: datetime
    end: datetime
    reference_type: TimeReference
    label: str
    anchor_event: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "reference_type": self.reference_type.value,
            "label": self.label,
            "anchor_event": self.anchor_event,
            "confidence": self.confidence,
        }


@dataclass
class TimeComparison:
    """Two time windows being compared."""
    baseline: TimeWindow
    target: TimeWindow
    delta_type: str
    alignment: str

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline.to_dict(),
            "target": self.target.to_dict(),
            "delta_type": self.delta_type,
            "alignment": self.alignment,
        }


@dataclass
class TemporalSnapshot:
    """A captured state of equipment metrics at a specific time."""
    equipment_id: str
    timestamp: datetime
    metrics: dict[str, float] = field(default_factory=dict)
    alerts_active: list[str] = field(default_factory=list)
    status: str = "normal"

    def to_dict(self) -> dict:
        return {
            "equipment_id": self.equipment_id,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "alerts_active": self.alerts_active,
            "status": self.status,
        }


@dataclass
class TemporalDelta:
    """Computed difference between two snapshots."""
    equipment_id: str
    baseline_time: datetime
    target_time: datetime
    metric_deltas: dict[str, dict] = field(default_factory=dict)
    alert_changes: dict = field(default_factory=dict)
    status_change: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "equipment_id": self.equipment_id,
            "baseline_time": self.baseline_time.isoformat(),
            "target_time": self.target_time.isoformat(),
            "metric_deltas": self.metric_deltas,
            "alert_changes": self.alert_changes,
            "status_change": self.status_change,
        }


# Relative time patterns
RELATIVE_PATTERNS = [
    (r"last (\d+) hours?", lambda m: timedelta(hours=int(m.group(1)))),
    (r"last (\d+) minutes?", lambda m: timedelta(minutes=int(m.group(1)))),
    (r"last (\d+) days?", lambda m: timedelta(days=int(m.group(1)))),
    (r"past (\d+) hours?", lambda m: timedelta(hours=int(m.group(1)))),
    (r"past (\d+) days?", lambda m: timedelta(days=int(m.group(1)))),
    (r"past week", lambda m: timedelta(weeks=1)),
    (r"last week", lambda m: timedelta(weeks=1)),
    (r"yesterday", lambda m: timedelta(days=1)),
    (r"today", lambda m: timedelta(hours=0)),
    (r"this morning", lambda m: timedelta(hours=12)),
    (r"last shift", lambda m: timedelta(hours=8)),
    (r"past month", lambda m: timedelta(days=30)),
    (r"last month", lambda m: timedelta(days=30)),
]

# Comparison patterns
COMPARISON_PATTERNS = [
    (r"before (?:the |)(?:anomaly|alert|issue|incident|failure)", "pre_post_anomaly"),
    (r"after (?:the |)(?:maintenance|repair|fix|service)", "pre_post_maintenance"),
    (r"this week (?:vs|versus|compared to) last week", "week_over_week"),
    (r"today (?:vs|versus|compared to) yesterday", "day_over_day"),
    (r"morning (?:shift |)(?:vs|versus|compared to) (?:night|evening) (?:shift|)", "shift_comparison"),
    (r"(?:compare|comparing) (?:with|to) (?:last |)(\w+)", "period_comparison"),
]


class TimeResolver:
    """Resolves natural language time references to concrete TimeWindows."""

    def resolve(self, text: str, focus_graph=None) -> Optional[TimeWindow]:
        """
        Resolution cascade:
        1. Relative expressions ("last 24 hours")
        2. Event-anchored ("before the anomaly")
        3. Default (last 24h)
        """
        text_lower = text.lower()
        now = datetime.now()

        # 1. Relative expressions
        for pattern, delta_fn in RELATIVE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                delta = delta_fn(match)
                if delta.total_seconds() == 0:
                    # "today" → from midnight
                    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    start = now - delta
                label = match.group(0).strip()
                return TimeWindow(
                    start=start, end=now,
                    reference_type=TimeReference.RELATIVE,
                    label=label.title(),
                    confidence=0.95,
                )

        # 2. Event-anchored
        event_window = self._resolve_event_anchor(text_lower, focus_graph)
        if event_window:
            return event_window

        return None

    def _resolve_event_anchor(self, text: str, focus_graph=None) -> Optional[TimeWindow]:
        """Resolve event-anchored time references."""
        now = datetime.now()

        if "before" in text and any(w in text for w in ["anomaly", "alert", "issue", "incident"]):
            # Default: 6 hours before now (approximate anomaly time)
            event_time = now - timedelta(hours=2)
            if focus_graph and hasattr(focus_graph, 'get_active_anomalies'):
                anomalies = focus_graph.get_active_anomalies()
                if anomalies:
                    ts = anomalies[0].properties.get("timestamp")
                    if ts:
                        try:
                            event_time = datetime.fromisoformat(str(ts))
                        except (ValueError, TypeError):
                            pass
            return TimeWindow(
                start=event_time - timedelta(hours=6),
                end=event_time,
                reference_type=TimeReference.EVENT_ANCHORED,
                label="Pre-anomaly",
                anchor_event="anomaly",
                confidence=0.7,
            )

        if "after" in text and any(w in text for w in ["maintenance", "repair", "fix", "service"]):
            event_time = now - timedelta(hours=12)
            return TimeWindow(
                start=event_time,
                end=now,
                reference_type=TimeReference.EVENT_ANCHORED,
                label="Post-maintenance",
                anchor_event="maintenance",
                confidence=0.7,
            )

        return None

    def build_comparison(self, text: str, focus_graph=None) -> Optional[TimeComparison]:
        """Parse comparative time references."""
        text_lower = text.lower()
        now = datetime.now()

        for pattern, delta_type in COMPARISON_PATTERNS:
            if re.search(pattern, text_lower):
                if delta_type == "week_over_week":
                    baseline = TimeWindow(
                        start=now - timedelta(weeks=2), end=now - timedelta(weeks=1),
                        reference_type=TimeReference.COMPARATIVE, label="Last week",
                    )
                    target = TimeWindow(
                        start=now - timedelta(weeks=1), end=now,
                        reference_type=TimeReference.COMPARATIVE, label="This week",
                    )
                    return TimeComparison(
                        baseline=baseline, target=target,
                        delta_type=delta_type, alignment="time_of_day",
                    )
                elif delta_type == "day_over_day":
                    baseline = TimeWindow(
                        start=now - timedelta(days=2), end=now - timedelta(days=1),
                        reference_type=TimeReference.COMPARATIVE, label="Yesterday",
                    )
                    target = TimeWindow(
                        start=now - timedelta(days=1), end=now,
                        reference_type=TimeReference.COMPARATIVE, label="Today",
                    )
                    return TimeComparison(
                        baseline=baseline, target=target,
                        delta_type=delta_type, alignment="time_of_day",
                    )
                elif delta_type == "pre_post_anomaly":
                    event_time = now - timedelta(hours=2)
                    baseline = TimeWindow(
                        start=event_time - timedelta(hours=6), end=event_time,
                        reference_type=TimeReference.EVENT_ANCHORED,
                        label="Pre-anomaly", anchor_event="anomaly",
                    )
                    target = TimeWindow(
                        start=event_time, end=now,
                        reference_type=TimeReference.EVENT_ANCHORED,
                        label="Post-anomaly", anchor_event="anomaly",
                    )
                    return TimeComparison(
                        baseline=baseline, target=target,
                        delta_type=delta_type, alignment="event_relative",
                    )
                elif delta_type == "shift_comparison":
                    morning_start = now.replace(hour=6, minute=0, second=0)
                    morning_end = now.replace(hour=14, minute=0, second=0)
                    night_start = now.replace(hour=22, minute=0, second=0) - timedelta(days=1)
                    night_end = now.replace(hour=6, minute=0, second=0)
                    baseline = TimeWindow(
                        start=night_start, end=night_end,
                        reference_type=TimeReference.COMPARATIVE, label="Night shift",
                    )
                    target = TimeWindow(
                        start=morning_start, end=morning_end,
                        reference_type=TimeReference.COMPARATIVE, label="Morning shift",
                    )
                    return TimeComparison(
                        baseline=baseline, target=target,
                        delta_type=delta_type, alignment="shift",
                    )
                else:
                    # Generic period comparison
                    baseline = TimeWindow(
                        start=now - timedelta(days=14), end=now - timedelta(days=7),
                        reference_type=TimeReference.COMPARATIVE, label="Previous period",
                    )
                    target = TimeWindow(
                        start=now - timedelta(days=7), end=now,
                        reference_type=TimeReference.COMPARATIVE, label="Current period",
                    )
                    return TimeComparison(
                        baseline=baseline, target=target,
                        delta_type=delta_type, alignment="sequential",
                    )
        return None


class SnapshotEngine:
    """Captures and compares equipment state at different times."""

    def capture_snapshot(self, equipment_id: str, timestamp: datetime,
                          metrics: dict[str, float] = None,
                          alerts: list[str] = None) -> TemporalSnapshot:
        """Create a snapshot from provided data (or DB lookup in production)."""
        return TemporalSnapshot(
            equipment_id=equipment_id,
            timestamp=timestamp,
            metrics=metrics or {},
            alerts_active=alerts or [],
            status=self._compute_status(metrics or {}),
        )

    def compute_delta(self, baseline: TemporalSnapshot,
                       target: TemporalSnapshot) -> TemporalDelta:
        """Compute difference between two snapshots."""
        metric_deltas = {}

        all_metrics = set(list(baseline.metrics.keys()) + list(target.metrics.keys()))
        for metric in all_metrics:
            b_val = baseline.metrics.get(metric)
            t_val = target.metrics.get(metric)
            if b_val is not None and t_val is not None:
                delta = t_val - b_val
                pct_change = (delta / b_val * 100) if b_val != 0 else 0
                # Significance based on z-score approximation
                significance = "trivial"
                if abs(pct_change) > 50:
                    significance = "significant"
                elif abs(pct_change) > 10:
                    significance = "notable"
                metric_deltas[metric] = {
                    "baseline": b_val,
                    "target": t_val,
                    "delta": round(delta, 3),
                    "percent_change": round(pct_change, 1),
                    "significance": significance,
                }

        # Alert changes
        baseline_alerts = set(baseline.alerts_active)
        target_alerts = set(target.alerts_active)
        alert_changes = {
            "new": list(target_alerts - baseline_alerts),
            "resolved": list(baseline_alerts - target_alerts),
            "persistent": list(baseline_alerts & target_alerts),
        }

        # Status change
        status_change = None
        if baseline.status != target.status:
            status_change = f"{baseline.status}→{target.status}"

        return TemporalDelta(
            equipment_id=baseline.equipment_id,
            baseline_time=baseline.timestamp,
            target_time=target.timestamp,
            metric_deltas=metric_deltas,
            alert_changes=alert_changes,
            status_change=status_change,
        )

    def _compute_status(self, metrics: dict[str, float]) -> str:
        """Compute overall status from metrics."""
        # Simplified — in production would check thresholds per metric
        return "normal"
