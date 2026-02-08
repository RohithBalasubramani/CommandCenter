"""
Tests for Upgrade 5: Constraint-Aware Decision Making

Test IDs: CS-B01 through CS-B07
"""

import pytest
from layer2.constraints import (
    Constraint, ConstraintType, ViolationAction,
    ConstraintChecker, ConstraintCheckResult, ConstraintViolation,
)


class TestConstraintEvaluation:
    """CS-B01/B02: Individual constraint evaluation."""

    def test_freshness_constraint_satisfied(self):
        """CS-B01: Data within freshness threshold → satisfied."""
        c = Constraint(
            type=ConstraintType.DATA_FRESHNESS,
            description="Data < 5 min old",
            threshold=300.0,
            action_on_violation=ViolationAction.QUALIFY,
        )
        assert c.evaluate(120.0) is True
        assert c.satisfied is True

    def test_freshness_constraint_violated(self):
        """CS-B02: Data too old → violated with message."""
        c = Constraint(
            type=ConstraintType.DATA_FRESHNESS,
            description="Data < 5 min old",
            threshold=300.0,
            action_on_violation=ViolationAction.QUALIFY,
        )
        assert c.evaluate(600.0) is False
        assert c.satisfied is False
        assert "600" in c.message

    def test_confidence_constraint_satisfied(self):
        c = Constraint(
            type=ConstraintType.CONFIDENCE,
            description="Confidence > 30%",
            threshold=0.30,
        )
        assert c.evaluate(0.85) is True

    def test_confidence_constraint_violated(self):
        c = Constraint(
            type=ConstraintType.CONFIDENCE,
            description="Confidence > 30%",
            threshold=0.30,
        )
        assert c.evaluate(0.15) is False

    def test_latency_constraint(self):
        c = Constraint(
            type=ConstraintType.LATENCY,
            description="Under 8s",
            threshold=8000.0,
        )
        assert c.evaluate(5000.0) is True
        assert c.evaluate(12000.0) is False

    def test_coverage_constraint(self):
        c = Constraint(
            type=ConstraintType.COVERAGE,
            description="At least 50% coverage",
            threshold=0.50,
        )
        assert c.evaluate(0.80) is True
        assert c.evaluate(0.20) is False


class TestConstraintChecker:
    """CS-B03 through CS-B05: Checker evaluates multiple constraints."""

    def test_all_satisfied(self):
        """CS-B03: All constraints satisfied → can_proceed=True, no violations."""
        checker = ConstraintChecker()
        result = checker.check({
            ConstraintType.DATA_FRESHNESS: 60.0,
            ConstraintType.CONFIDENCE: 0.85,
            ConstraintType.LATENCY: 3000.0,
            ConstraintType.COVERAGE: 0.90,
        })
        assert result.can_proceed is True
        assert len(result.violations) == 0

    def test_freshness_violated_produces_qualifier(self):
        """CS-B04: Stale data produces voice qualifier."""
        checker = ConstraintChecker()
        result = checker.check({
            ConstraintType.DATA_FRESHNESS: 600.0,  # 10 min (limit 5 min)
            ConstraintType.CONFIDENCE: 0.85,
        })
        assert result.can_proceed is True  # QUALIFY doesn't block
        assert len(result.violations) == 1
        assert len(result.qualifiers) == 1
        assert "current" in result.qualifiers[0].lower() or "old" in result.qualifiers[0].lower()

    def test_refuse_blocks_proceeding(self):
        """CS-B05: REFUSE action blocks proceeding."""
        checker = ConstraintChecker(constraints=[
            Constraint(
                type=ConstraintType.SAFETY,
                description="Safety check",
                threshold=1.0,
                action_on_violation=ViolationAction.REFUSE,
            ),
        ])
        result = checker.check({ConstraintType.SAFETY: 0.0})
        assert result.can_proceed is False
        assert result.highest_action == ViolationAction.REFUSE


class TestConstraintSerialization:
    """CS-B06: Serialization."""

    def test_constraint_to_dict(self):
        c = Constraint(
            type=ConstraintType.DATA_FRESHNESS,
            description="Data < 5 min old",
            threshold=300.0,
        )
        c.evaluate(600.0)
        d = c.to_dict()
        assert d["type"] == "data_freshness"
        assert d["satisfied"] is False
        assert d["actual"] == 600.0

    def test_check_result_to_dict(self):
        checker = ConstraintChecker()
        result = checker.check({ConstraintType.DATA_FRESHNESS: 600.0})
        d = result.to_dict()
        assert "can_proceed" in d
        assert "violations" in d
        assert len(d["violations"]) >= 1


class TestVoiceQualifierBuilder:
    """CS-B07: Voice qualifier generation."""

    def test_build_voice_prefix_empty_when_no_violations(self):
        checker = ConstraintChecker()
        result = checker.check({ConstraintType.CONFIDENCE: 0.90})
        prefix = checker.build_voice_prefix(result)
        assert prefix == ""

    def test_build_voice_prefix_includes_qualifiers(self):
        checker = ConstraintChecker()
        result = checker.check({ConstraintType.DATA_FRESHNESS: 900.0})
        prefix = checker.build_voice_prefix(result)
        assert len(prefix) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
