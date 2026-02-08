"""
Tests for Upgrade 10: Explicit Failure-Mode UX

Test IDs: UA-B01 through UA-B12
"""

import pytest
from layer2.uncertainty import (
    UncertaintyAssessor, UncertaintyAssessment,
    KnownFact, UnknownFactor, NextStep, ConstraintViolationInfo,
)


@pytest.fixture
def assessor():
    return UncertaintyAssessor()


class TestUncertaintyAssessment:
    """UA-B01/B02: Assessment data structure."""

    def test_should_show_panel_low_confidence(self):
        """UA-B01: Low confidence → show panel."""
        assessment = UncertaintyAssessment(overall_confidence=0.4)
        assert assessment.should_show_panel() is True

    def test_should_show_panel_many_unknowns(self):
        """UA-B02: Multiple unknowns → show panel."""
        assessment = UncertaintyAssessment(
            overall_confidence=0.8,
            unknown_factors=[
                UnknownFactor(description="A"),
                UnknownFactor(description="B"),
            ],
        )
        assert assessment.should_show_panel() is True

    def test_should_show_panel_constraint_violation(self):
        assessment = UncertaintyAssessment(
            overall_confidence=0.8,
            constraint_violations=[
                ConstraintViolationInfo(type="freshness", message="Data stale"),
            ],
        )
        assert assessment.should_show_panel() is True

    def test_should_not_show_panel_high_confidence(self):
        """High confidence + no issues → hide panel."""
        assessment = UncertaintyAssessment(overall_confidence=0.9)
        assert assessment.should_show_panel() is False


class TestVoicePrefixSuffix:
    """UA-B03: Voice response qualifiers."""

    def test_low_confidence_prefix(self):
        assessment = UncertaintyAssessment(overall_confidence=0.3)
        prefix = assessment.get_voice_prefix()
        assert "not very confident" in prefix

    def test_moderate_confidence_prefix(self):
        assessment = UncertaintyAssessment(overall_confidence=0.5)
        prefix = assessment.get_voice_prefix()
        assert "gaps" in prefix

    def test_high_confidence_no_prefix(self):
        assessment = UncertaintyAssessment(overall_confidence=0.8)
        assert assessment.get_voice_prefix() == ""

    def test_unknown_suffix(self):
        assessment = UncertaintyAssessment(
            unknown_factors=[
                UnknownFactor(description="sensor data for pump 5"),
            ],
        )
        suffix = assessment.get_voice_suffix()
        assert "sensor data for pump 5" in suffix

    def test_no_unknowns_no_suffix(self):
        assessment = UncertaintyAssessment()
        assert assessment.get_voice_suffix() == ""


class TestAssessorFromPlanSteps:
    """UA-B04/B05: Assessor extracts from plan execution."""

    def test_completed_retrieve_becomes_known_fact(self, assessor):
        """UA-B04: Completed retrieves → known facts."""
        steps = [
            {
                "type": "RETRIEVE",
                "status": "completed",
                "outputs": {
                    "metric": "vibration",
                    "equipment": "pump_004",
                    "value": "3.2",
                    "unit": "mm/s",
                    "source": "timeseries",
                    "age_seconds": 30,
                    "confidence": 0.9,
                },
            },
        ]
        result = assessor.assess(plan_steps=steps)
        assert len(result.known_facts) == 1
        assert "vibration" in result.known_facts[0].statement
        assert result.known_facts[0].freshness == "live"

    def test_failed_retrieve_becomes_unknown(self, assessor):
        """UA-B05: Failed retrieves → unknowns."""
        steps = [
            {
                "type": "RETRIEVE",
                "status": "failed",
                "description": "Fetch data for pump_005",
                "error": "Sensor offline",
                "inputs": {"equipment": "pump_005"},
            },
        ]
        result = assessor.assess(plan_steps=steps)
        assert len(result.unknown_factors) == 1
        assert "pump_005" in result.unknown_factors[0].description


class TestAssessorFromReasoning:
    """UA-B06: Assessor extracts from reasoning results."""

    def test_reasoning_facts_and_unknowns(self, assessor):
        reasoning = {
            "known_facts": ["Vibration is above threshold", "Temperature is normal"],
            "unknown_factors": ["Bearing condition unknown"],
            "recommended_checks": ["Inspect bearing on pump 4"],
            "confidence": 0.7,
        }
        result = assessor.assess(reasoning_result=reasoning)
        assert len(result.known_facts) == 2
        assert len(result.unknown_factors) == 1
        assert len(result.next_steps) == 1
        assert "Inspect" in result.next_steps[0].action


class TestAssessorFromConstraints:
    """UA-B07: Assessor extracts constraint violations."""

    def test_constraint_violations_extracted(self, assessor):
        constraint_result = {
            "violations": [
                {"type": "data_freshness", "message": "Data is 600s old", "action": "qualify"},
                {"type": "safety", "message": "Safety limit exceeded", "action": "refuse"},
            ],
        }
        result = assessor.assess(constraint_result=constraint_result)
        assert len(result.constraint_violations) == 2
        assert result.constraint_violations[0].severity == "warning"
        assert result.constraint_violations[1].severity == "error"


class TestConfidenceComputation:
    """UA-B08/B09: Overall confidence scoring."""

    def test_high_confidence_all_known(self, assessor):
        """UA-B08: All data present → high confidence."""
        steps = [
            {
                "type": "RETRIEVE", "status": "completed",
                "outputs": {"confidence": 0.9, "age_seconds": 10},
            },
            {
                "type": "RETRIEVE", "status": "completed",
                "outputs": {"confidence": 0.85, "age_seconds": 20},
            },
        ]
        result = assessor.assess(plan_steps=steps)
        assert result.overall_confidence >= 0.7

    def test_low_confidence_all_facts_low(self, assessor):
        """UA-B09: All facts low confidence → overall ≤ 0.1."""
        steps = [
            {
                "type": "RETRIEVE", "status": "completed",
                "outputs": {"confidence": 0.2, "age_seconds": 10},
            },
            {
                "type": "RETRIEVE", "status": "completed",
                "outputs": {"confidence": 0.1, "age_seconds": 10},
            },
        ]
        result = assessor.assess(plan_steps=steps)
        assert result.overall_confidence <= 0.1

    def test_stale_data_reduces_confidence(self, assessor):
        """Stale data reduces overall confidence."""
        steps = [
            {
                "type": "RETRIEVE", "status": "completed",
                "outputs": {"confidence": 0.9, "age_seconds": 10},
            },
        ]
        fresh = assessor.assess(plan_steps=steps, data_freshness_s=30)
        stale = assessor.assess(plan_steps=steps, data_freshness_s=7200)
        assert fresh.overall_confidence > stale.overall_confidence


class TestUnknownsTruncation:
    """UA-B10: Truncation of too many unknowns."""

    def test_truncate_at_max(self, assessor):
        steps = [
            {"type": "RETRIEVE", "status": "failed",
             "description": f"Item {i}", "inputs": {"equipment": f"eq_{i}"}}
            for i in range(10)
        ]
        result = assessor.assess(plan_steps=steps)
        # Should be MAX_UNKNOWNS_DISPLAY + 1 (truncation notice)
        assert len(result.unknown_factors) <= assessor.MAX_UNKNOWNS_DISPLAY + 1
        assert "more" in result.unknown_factors[-1].description.lower()


class TestFreshnessLabels:
    """UA-B11: Freshness label computation."""

    def test_live(self, assessor):
        assert assessor._freshness_label(10) == "live"

    def test_5min(self, assessor):
        assert assessor._freshness_label(200) == "5min"

    def test_1hr(self, assessor):
        assert assessor._freshness_label(1800) == "1hr"

    def test_stale(self, assessor):
        assert assessor._freshness_label(7200) == "stale"

    def test_unknown(self, assessor):
        assert assessor._freshness_label(None) == "unknown"


class TestSerialization:
    """UA-B12: Dict serialization."""

    def test_assessment_to_dict(self):
        assessment = UncertaintyAssessment(
            overall_confidence=0.65,
            known_facts=[KnownFact(statement="Pump running", source="timeseries", freshness="live", confidence=0.9)],
            unknown_factors=[UnknownFactor(description="Bearing condition", why_unknown="no_sensor", impact="high")],
            next_steps=[NextStep(action="Inspect bearing", automated=False, priority="high")],
            constraint_violations=[ConstraintViolationInfo(type="freshness", message="Data stale", severity="warning")],
        )
        d = assessment.to_dict()
        assert d["overall_confidence"] == 0.65
        assert len(d["known_facts"]) == 1
        assert d["known_facts"][0]["source"] == "timeseries"
        assert len(d["unknown_factors"]) == 1
        assert len(d["next_steps"]) == 1
        assert len(d["constraint_violations"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
