"""
Tests for Upgrade 9: Voice as Control Surface

Test IDs: VC-B01 through VC-B10
"""

import time
import pytest
from layer2.voice_control import (
    InterruptDetector, InterruptEvent, InterruptType,
    PlanCancellationManager, ProsodySignals, ProsodyLevel,
)


class TestInterruptDetection:
    """VC-B01/B02: Interrupt keyword detection."""

    def test_stop_keyword_detected(self):
        """VC-B01: 'stop' triggers stop interrupt."""
        detector = InterruptDetector()
        result = detector.check("stop")
        assert result is not None
        assert result.type == InterruptType.STOP

    def test_cancel_keyword_detected(self):
        detector = InterruptDetector()
        result = detector.check("cancel")
        assert result is not None
        assert result.type == InterruptType.STOP

    def test_nevermind_detected(self):
        detector = InterruptDetector()
        result = detector.check("never mind")
        assert result is not None
        assert result.type == InterruptType.STOP

    def test_forget_it_detected(self):
        detector = InterruptDetector()
        result = detector.check("forget it")
        assert result is not None
        assert result.type == InterruptType.STOP

    def test_redirect_no_detected(self):
        """VC-B02: 'no,' triggers redirect."""
        detector = InterruptDetector()
        result = detector.check("no, I meant pump 5")
        assert result is not None
        assert result.type == InterruptType.REDIRECT

    def test_redirect_actually_detected(self):
        detector = InterruptDetector()
        result = detector.check("actually, show me motor 1")
        assert result is not None
        assert result.type == InterruptType.REDIRECT

    def test_redirect_instead_detected(self):
        detector = InterruptDetector()
        result = detector.check("instead, compare with pump 5")
        assert result is not None
        assert result.type == InterruptType.REDIRECT

    def test_redirect_wait_detected(self):
        detector = InterruptDetector()
        result = detector.check("wait, that's not right")
        assert result is not None
        assert result.type == InterruptType.REDIRECT

    def test_no_interrupt_on_normal_text(self):
        detector = InterruptDetector()
        result = detector.check("show me pump 4 vibration")
        assert result is None

    def test_empty_text_returns_none(self):
        detector = InterruptDetector()
        assert detector.check("") is None
        assert detector.check("  ") is None
        assert detector.check(None) is None


class TestFalsePositiveRejection:
    """VC-B03: False positive interrupt rejection."""

    def test_stop_in_context_not_interrupt(self):
        """'Stop the vibration from increasing' is NOT an interrupt."""
        detector = InterruptDetector()
        result = detector.check("stop the vibration from increasing")
        assert result is None

    def test_stop_pump_not_interrupt(self):
        detector = InterruptDetector()
        result = detector.check("stop pump 4")
        assert result is None

    def test_halt_production_not_interrupt(self):
        detector = InterruptDetector()
        result = detector.check("halt production line 2")
        assert result is None


class TestRapidFireDetection:
    """VC-B04: Rapid-fire interrupt handling."""

    def test_rapid_fire_detection(self):
        detector = InterruptDetector()
        detector.RAPID_FIRE_THRESHOLD = 3
        detector.RAPID_FIRE_WINDOW_S = 10.0

        # Fire 3 interrupts rapidly
        detector.check("stop")
        detector.check("cancel")
        detector.check("halt")

        assert detector.is_rapid_fire()

    def test_rapid_fire_returns_special_event(self):
        detector = InterruptDetector()
        detector.RAPID_FIRE_THRESHOLD = 3

        detector.check("stop")
        detector.check("cancel")
        detector.check("halt")

        # 4th interrupt should return rapid fire indicator
        result = detector.check("stop again")
        assert result is not None
        assert "[rapid_fire_detected]" in result.transcript

    def test_reset_clears_state(self):
        detector = InterruptDetector()
        detector.check("stop")
        detector.check("cancel")
        detector.reset()
        assert not detector.is_rapid_fire()


class TestPlanCancellation:
    """VC-B05/B06: Plan cancellation management."""

    def test_register_and_cancel(self):
        """VC-B05: Register plan then cancel it."""
        mgr = PlanCancellationManager()
        mgr.register_plan("plan-001")
        result = mgr.cancel("plan-001")
        assert result["cancelled"] is True
        assert result["plan_id"] == "plan-001"

    def test_cancel_nonexistent_plan(self):
        mgr = PlanCancellationManager()
        result = mgr.cancel("nonexistent")
        assert result["cancelled"] is False
        assert result["reason"] == "plan_not_found"

    def test_cancel_already_completed(self):
        """VC-B06: Cancel after completion returns already_completed."""
        mgr = PlanCancellationManager()
        mgr.register_plan("plan-001")
        mgr.mark_completed("plan-001")
        result = mgr.cancel("plan-001")
        assert result["cancelled"] is False
        assert result["reason"] == "already_completed"

    def test_cancel_already_cancelled(self):
        mgr = PlanCancellationManager()
        mgr.register_plan("plan-001")
        mgr.cancel("plan-001")
        result = mgr.cancel("plan-001")
        assert result["cancelled"] is False
        assert result["reason"] == "already_cancelled"

    def test_is_cancelled(self):
        mgr = PlanCancellationManager()
        mgr.register_plan("plan-001")
        assert not mgr.is_cancelled("plan-001")
        mgr.cancel("plan-001")
        assert mgr.is_cancelled("plan-001")

    def test_active_count(self):
        mgr = PlanCancellationManager()
        mgr.register_plan("p1")
        mgr.register_plan("p2")
        mgr.register_plan("p3")
        assert mgr.get_active_count() == 3
        mgr.cancel("p1")
        assert mgr.get_active_count() == 2
        mgr.mark_completed("p2")
        assert mgr.get_active_count() == 1

    def test_cleanup_old_plans(self):
        mgr = PlanCancellationManager()
        mgr.register_plan("old")
        mgr._active_plans["old"]["started_at"] = time.time() - 600
        mgr.register_plan("new")
        mgr.cleanup_old(max_age_s=300)
        assert "old" not in mgr._active_plans
        assert "new" in mgr._active_plans


class TestProsodySignals:
    """VC-B07: Prosody signal defaults."""

    def test_default_neutral(self):
        signals = ProsodySignals()
        assert signals.urgency == 0.5
        assert signals.hesitation == 0.5
        assert signals.confidence == 0.5
        assert signals.urgency_level() == ProsodyLevel.NEUTRAL

    def test_high_urgency(self):
        signals = ProsodySignals(urgency=0.9)
        assert signals.urgency_level() == ProsodyLevel.HIGH

    def test_low_urgency(self):
        signals = ProsodySignals(urgency=0.1)
        assert signals.urgency_level() == ProsodyLevel.LOW

    def test_serialization(self):
        signals = ProsodySignals(urgency=0.8, hesitation=0.2, confidence=0.9)
        d = signals.to_dict()
        assert d["urgency"] == 0.8
        assert d["urgency_level"] == "high"


class TestInterruptEventSerialization:
    """VC-B08: Event serialization."""

    def test_interrupt_event_to_dict(self):
        event = InterruptEvent(
            type=InterruptType.STOP,
            transcript="stop",
            plan_id="plan-001",
        )
        d = event.to_dict()
        assert d["type"] == "stop"
        assert d["plan_id"] == "plan-001"
        assert d["transcript"] == "stop"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
