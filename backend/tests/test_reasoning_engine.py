"""
Tests for Upgrade 2: Cross-Widget Reasoning Engine

Test IDs: RE-B01 through RE-B06
"""

import pytest
from layer2.reasoning_engine import (
    CrossWidgetReasoningEngine, DataPoint, ReasoningQuery,
    ReasoningType, ReasoningResult, Hypothesis,
)


class TestReasoningEnginePatternMatching:
    """RE-B01: Rule-based pattern matching."""

    def test_vibration_high_temperature_normal_pattern(self):
        """Classic bearing wear pattern detected."""
        engine = CrossWidgetReasoningEngine()
        observations = [
            DataPoint(equipment="pump_004", metric="vibration_de_mm_s",
                      value=3.2, unit="mm/s", timestamp=1000.0,
                      status="critical", threshold=2.5),
            DataPoint(equipment="pump_004", metric="oil_temperature_top_c",
                      value=42.0, unit="°C", timestamp=1000.0,
                      status="normal", threshold=85.0),
        ]
        query = ReasoningQuery(
            type=ReasoningType.COMPARATIVE,
            question="why is vibration high but temperature normal?",
            observations=observations,
        )
        result = engine.reason(query)
        assert len(result.hypotheses) >= 1
        assert "bearing" in result.hypotheses[0].statement.lower() or \
               "misalignment" in result.hypotheses[0].statement.lower()
        assert result.hypotheses[0].confidence > 0.5

    def test_temperature_high_vibration_normal_pattern(self):
        """Cooling system degradation pattern detected."""
        engine = CrossWidgetReasoningEngine()
        observations = [
            DataPoint(equipment="pump_004", metric="temperature_c",
                      value=95.0, unit="°C", timestamp=1000.0,
                      status="critical", threshold=85.0),
            DataPoint(equipment="pump_004", metric="vibration_mm_s",
                      value=1.0, unit="mm/s", timestamp=1000.0,
                      status="normal", threshold=2.5),
        ]
        query = ReasoningQuery(
            type=ReasoningType.COMPARATIVE,
            question="why is temperature high but vibration normal?",
            observations=observations,
        )
        result = engine.reason(query)
        assert len(result.hypotheses) >= 1
        assert "cooling" in result.hypotheses[0].statement.lower() or \
               "overload" in result.hypotheses[0].statement.lower()


class TestReasoningEngineInsufficientData:
    """RE-B02: Handling insufficient data."""

    def test_single_observation_returns_incomplete(self):
        """Single data point → insufficient data."""
        engine = CrossWidgetReasoningEngine()
        observations = [
            DataPoint(equipment="pump_004", metric="vibration",
                      value=3.2, unit="mm/s", timestamp=1000.0,
                      status="critical", threshold=2.5),
        ]
        query = ReasoningQuery(
            type=ReasoningType.CAUSAL,
            question="why is vibration high?",
            observations=observations,
        )
        result = engine.reason(query)
        assert len(result.hypotheses) == 0
        assert len(result.unknown_factors) > 0
        assert result.confidence < 0.5

    def test_no_observations_returns_empty(self):
        """No data points → insufficient data."""
        engine = CrossWidgetReasoningEngine()
        query = ReasoningQuery(
            type=ReasoningType.DIAGNOSTIC,
            question="what's wrong?",
            observations=[],
        )
        result = engine.reason(query)
        assert len(result.hypotheses) == 0


class TestReasoningEngineTypeDetection:
    """RE-B03: Query type detection."""

    def test_detect_causal_type(self):
        engine = CrossWidgetReasoningEngine()
        assert engine.detect_reasoning_type("why is vibration high?") == ReasoningType.CAUSAL

    def test_detect_comparative_type(self):
        engine = CrossWidgetReasoningEngine()
        assert engine.detect_reasoning_type("why is vibration high but temperature normal?") == ReasoningType.COMPARATIVE

    def test_detect_predictive_type(self):
        engine = CrossWidgetReasoningEngine()
        assert engine.detect_reasoning_type("what will happen if this continues?") == ReasoningType.PREDICTIVE

    def test_detect_correlative_type(self):
        engine = CrossWidgetReasoningEngine()
        assert engine.detect_reasoning_type("are these metrics related?") == ReasoningType.CORRELATIVE

    def test_detect_diagnostic_type(self):
        engine = CrossWidgetReasoningEngine()
        assert engine.detect_reasoning_type("help me troubleshoot the pump") == ReasoningType.DIAGNOSTIC

    def test_no_reasoning_needed(self):
        engine = CrossWidgetReasoningEngine()
        assert engine.detect_reasoning_type("show pump 4 vibration") is None


class TestReasoningResultSerialization:
    """RE-B04: Result serialization."""

    def test_to_dict_contains_all_fields(self):
        result = ReasoningResult(
            query_type=ReasoningType.CAUSAL,
            hypotheses=[
                Hypothesis(
                    id="h1", statement="Test hypothesis",
                    confidence=0.72,
                    supporting_evidence=["evidence 1"],
                    contradicting_evidence=[],
                    check_steps=["check 1"],
                ),
            ],
            known_facts=["Vibration is high"],
            unknown_factors=["Bearing condition unknown"],
            recommended_checks=["Check bearings"],
            confidence=0.72,
            reasoning_chain=["Step 1: collected data"],
            execution_time_ms=15,
        )
        d = result.to_dict()
        assert d["query_type"] == "causal"
        assert len(d["hypotheses"]) == 1
        assert d["hypotheses"][0]["confidence"] == 0.72
        assert len(d["known_facts"]) == 1
        assert len(d["unknown_factors"]) == 1
        assert d["execution_time_ms"] == 15


class TestReasoningEngineKnownFacts:
    """RE-B05: Known facts generation."""

    def test_known_facts_describe_observations(self):
        engine = CrossWidgetReasoningEngine()
        observations = [
            DataPoint(equipment="pump_004", metric="vibration",
                      value=3.2, unit="mm/s", timestamp=1000.0,
                      status="critical", threshold=2.5),
            DataPoint(equipment="pump_004", metric="temperature",
                      value=42.0, unit="°C", timestamp=1000.0,
                      status="normal", threshold=85.0),
        ]
        query = ReasoningQuery(
            type=ReasoningType.COMPARATIVE,
            question="why is vibration high but temperature normal?",
            observations=observations,
        )
        result = engine.reason(query)
        assert len(result.known_facts) == 2
        assert "vibration" in result.known_facts[0].lower()
        assert "temperature" in result.known_facts[1].lower()


class TestReasoningEngineCheckSteps:
    """RE-B06: Recommended check deduplication."""

    def test_check_steps_deduplicated(self):
        engine = CrossWidgetReasoningEngine()
        observations = [
            DataPoint(equipment="pump_004", metric="vibration",
                      value=3.2, unit="mm/s", timestamp=1000.0,
                      status="critical", threshold=2.5),
            DataPoint(equipment="pump_004", metric="temperature",
                      value=42.0, unit="°C", timestamp=1000.0,
                      status="normal", threshold=85.0),
        ]
        query = ReasoningQuery(
            type=ReasoningType.COMPARATIVE,
            question="why is vibration high but temperature normal?",
            observations=observations,
        )
        result = engine.reason(query)
        # No duplicate check steps
        assert len(result.recommended_checks) == len(set(result.recommended_checks))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
