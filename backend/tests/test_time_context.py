"""
Tests for Upgrade 3: Time-Travel + Comparison

Test IDs: TC-B01 through TC-B05
"""

import pytest
from datetime import datetime, timedelta
from layer2.time_context import (
    TimeResolver, SnapshotEngine, TimeWindow, TimeComparison,
    TemporalSnapshot, TemporalDelta, TimeReference,
)


class TestTimeResolver:
    """TC-B01/B02: Time resolution from natural language."""

    def test_resolve_relative_last_24_hours(self):
        resolver = TimeResolver()
        window = resolver.resolve("show data from last 24 hours")
        assert window is not None
        assert window.reference_type == TimeReference.RELATIVE
        assert (window.end - window.start).total_seconds() == pytest.approx(86400, abs=5)

    def test_resolve_relative_last_7_days(self):
        resolver = TimeResolver()
        window = resolver.resolve("trend over last 7 days")
        assert window is not None
        assert (window.end - window.start).days == 7

    def test_resolve_relative_yesterday(self):
        resolver = TimeResolver()
        window = resolver.resolve("what happened yesterday?")
        assert window is not None
        assert "Yesterday" in window.label

    def test_resolve_event_anchored_before_anomaly(self):
        resolver = TimeResolver()
        window = resolver.resolve("show data before the anomaly")
        assert window is not None
        assert window.reference_type == TimeReference.EVENT_ANCHORED
        assert window.anchor_event == "anomaly"
        assert "Pre-anomaly" in window.label

    def test_resolve_no_time_reference(self):
        resolver = TimeResolver()
        window = resolver.resolve("show pump 4 vibration")
        assert window is None


class TestTimeComparison:
    """TC-B03: Comparison time resolution."""

    def test_build_week_over_week(self):
        resolver = TimeResolver()
        comparison = resolver.build_comparison("this week vs last week")
        assert comparison is not None
        assert comparison.delta_type == "week_over_week"
        assert comparison.baseline.label == "Last week"
        assert comparison.target.label == "This week"

    def test_build_day_over_day(self):
        resolver = TimeResolver()
        comparison = resolver.build_comparison("today vs yesterday")
        assert comparison is not None
        assert comparison.delta_type == "day_over_day"

    def test_build_shift_comparison(self):
        resolver = TimeResolver()
        comparison = resolver.build_comparison("morning shift vs night shift")
        assert comparison is not None
        assert comparison.delta_type == "shift_comparison"

    def test_build_pre_post_anomaly(self):
        resolver = TimeResolver()
        comparison = resolver.build_comparison("before the anomaly vs after")
        assert comparison is not None
        assert comparison.delta_type == "pre_post_anomaly"

    def test_no_comparison_detected(self):
        resolver = TimeResolver()
        comparison = resolver.build_comparison("show pump 4 vibration")
        assert comparison is None


class TestSnapshotEngine:
    """TC-B04: Snapshot capture and delta computation."""

    def test_capture_snapshot(self):
        engine = SnapshotEngine()
        snapshot = engine.capture_snapshot(
            equipment_id="pump_004",
            timestamp=datetime.now(),
            metrics={"vibration": 3.2, "temperature": 42.0},
            alerts=["high_vibration"],
        )
        assert snapshot.equipment_id == "pump_004"
        assert snapshot.metrics["vibration"] == 3.2
        assert len(snapshot.alerts_active) == 1

    def test_compute_delta(self):
        engine = SnapshotEngine()
        now = datetime.now()
        baseline = TemporalSnapshot(
            equipment_id="pump_004",
            timestamp=now - timedelta(hours=6),
            metrics={"vibration": 1.8, "temperature": 41.0},
            alerts_active=[],
            status="normal",
        )
        target = TemporalSnapshot(
            equipment_id="pump_004",
            timestamp=now,
            metrics={"vibration": 3.2, "temperature": 42.0},
            alerts_active=["high_vibration"],
            status="critical",
        )
        delta = engine.compute_delta(baseline, target)
        assert delta.equipment_id == "pump_004"
        assert delta.metric_deltas["vibration"]["delta"] == pytest.approx(1.4, abs=0.01)
        assert delta.metric_deltas["vibration"]["significance"] == "significant"
        assert delta.metric_deltas["temperature"]["significance"] == "trivial"
        assert "high_vibration" in delta.alert_changes["new"]
        assert delta.status_change == "normal→critical"

    def test_delta_with_zero_baseline(self):
        """Handle zero baseline without division error."""
        engine = SnapshotEngine()
        now = datetime.now()
        baseline = TemporalSnapshot(
            equipment_id="pump_004", timestamp=now - timedelta(hours=1),
            metrics={"power": 0.0},
        )
        target = TemporalSnapshot(
            equipment_id="pump_004", timestamp=now,
            metrics={"power": 15.0},
        )
        delta = engine.compute_delta(baseline, target)
        assert delta.metric_deltas["power"]["delta"] == 15.0
        assert delta.metric_deltas["power"]["percent_change"] == 0  # division by zero safe


class TestTemporalSerialization:
    """TC-B05: Serialization."""

    def test_time_window_to_dict(self):
        now = datetime.now()
        tw = TimeWindow(
            start=now - timedelta(hours=24), end=now,
            reference_type=TimeReference.RELATIVE, label="Last 24 hours",
        )
        d = tw.to_dict()
        assert "start" in d
        assert "end" in d
        assert d["reference_type"] == "relative"
        assert d["label"] == "Last 24 hours"

    def test_time_comparison_to_dict(self):
        now = datetime.now()
        tc = TimeComparison(
            baseline=TimeWindow(start=now - timedelta(weeks=2), end=now - timedelta(weeks=1),
                                reference_type=TimeReference.COMPARATIVE, label="Last week"),
            target=TimeWindow(start=now - timedelta(weeks=1), end=now,
                              reference_type=TimeReference.COMPARATIVE, label="This week"),
            delta_type="week_over_week", alignment="time_of_day",
        )
        d = tc.to_dict()
        assert d["delta_type"] == "week_over_week"
        assert "baseline" in d
        assert "target" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
