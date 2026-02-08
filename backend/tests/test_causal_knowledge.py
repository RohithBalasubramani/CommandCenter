"""
Tests for Upgrade 8: Causal Knowledge Engine

Test IDs: CK-B01 through CK-B06
"""

import pytest
from layer2.causal_knowledge import (
    CausalKnowledgeEngine, FailureMode, SymptomMatch, KnowledgeGap,
)


class TestSymptomMatching:
    """CK-B01/B02: Symptom-based diagnosis."""

    def test_bearing_wear_diagnosis(self):
        """CK-B01: High vibration + normal temperature → bearing wear."""
        engine = CausalKnowledgeEngine()
        matches = engine.diagnose(
            equipment_type="pump",
            symptoms={"vibration": "high", "temperature": "normal"},
        )
        assert len(matches) > 0
        top = matches[0]
        assert "bearing" in top.failure_mode.root_cause.lower() or \
               "misalignment" in top.failure_mode.root_cause.lower()
        assert top.match_score > 0.3
        assert len(top.matched_symptoms) >= 1

    def test_motor_overheating_diagnosis(self):
        """CK-B02: High temp + normal vibration + high current → motor overload."""
        engine = CausalKnowledgeEngine()
        matches = engine.diagnose(
            equipment_type="motor",
            symptoms={"temperature": "high", "vibration": "normal", "current": "high"},
        )
        assert len(matches) > 0
        top = matches[0]
        assert "overload" in top.failure_mode.root_cause.lower() or \
               "insulation" in top.failure_mode.description.lower()


class TestEquipmentTypeMatching:
    """CK-B03: Fuzzy equipment type matching."""

    def test_partial_match(self):
        """CK-B03: 'pump' matches 'centrifugal_pump'."""
        engine = CausalKnowledgeEngine()
        matches = engine.diagnose(
            equipment_type="pump",
            symptoms={"vibration": "high"},
        )
        assert len(matches) > 0

    def test_exact_match(self):
        engine = CausalKnowledgeEngine()
        matches = engine.diagnose(
            equipment_type="centrifugal_pump",
            symptoms={"vibration": "high"},
        )
        assert len(matches) > 0


class TestContradictingSymptoms:
    """CK-B04: Contradicting symptoms reduce score."""

    def test_contradicting_reduces_score(self):
        engine = CausalKnowledgeEngine()
        # Bearing wear expects vibration=high, temp=normal
        # Providing temp=high contradicts
        matches_normal = engine.diagnose(
            "pump", {"vibration": "high", "temperature": "normal"}
        )
        matches_contra = engine.diagnose(
            "pump", {"vibration": "high", "temperature": "high"}
        )
        # Get bearing wear score from both
        bearing_normal = [m for m in matches_normal if "bearing" in m.failure_mode.root_cause]
        bearing_contra = [m for m in matches_contra if "bearing" in m.failure_mode.root_cause]
        if bearing_normal and bearing_contra:
            assert bearing_normal[0].match_score > bearing_contra[0].match_score


class TestKnowledgeGaps:
    """CK-B05: Knowledge gap detection."""

    def test_no_matches_logs_knowledge_gap(self):
        """CK-B05: Unknown symptoms create a knowledge gap."""
        engine = CausalKnowledgeEngine()
        matches = engine.diagnose(
            equipment_type="alien_device",
            symptoms={"quantum_flux": "high", "dark_matter": "low"},
        )
        assert len(matches) == 0
        gaps = engine.get_knowledge_gaps()
        assert len(gaps) == 1
        assert gaps[0]["equipment_type"] == "alien_device"


class TestFailureModeLoading:
    """CK-B06: Loading from JSON file."""

    def test_load_from_file(self, tmp_path):
        """CK-B06: Additional failure modes loaded from JSON."""
        fm_file = tmp_path / "test_modes.json"
        fm_file.write_text('''{
            "failure_modes": [{
                "id": "fm_test",
                "equipment_type": "test_device",
                "symptom_signature": {"smoke": "visible"},
                "root_cause": "on_fire",
                "description": "Test device is on fire"
            }]
        }''')
        engine = CausalKnowledgeEngine(failure_mode_db_path=str(fm_file))
        matches = engine.diagnose("test_device", {"smoke": "visible"})
        assert len(matches) > 0
        assert matches[0].failure_mode.root_cause == "on_fire"

    def test_builtin_modes_loaded(self):
        """Builtin failure modes are always available."""
        engine = CausalKnowledgeEngine()
        assert len(engine.failure_modes) >= 6  # At least 6 builtin modes

    def test_symptom_match_serialization(self):
        """SymptomMatch serializes to dict."""
        fm = FailureMode(
            id="test", equipment_type="pump",
            symptom_signature={"vibration": "high"},
            root_cause="test_cause", description="Test",
        )
        match = SymptomMatch(
            failure_mode=fm, match_score=0.72,
            matched_symptoms=["vibration"],
        )
        d = match.to_dict()
        assert d["match_score"] == 0.72
        assert d["failure_mode"]["root_cause"] == "test_cause"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
