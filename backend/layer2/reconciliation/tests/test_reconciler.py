"""
Tests for the Reconciler (mismatch classification).
"""
import pytest

from layer2.reconciliation.reconciler import (
    classify_mismatch,
    classify_value_type,
    check_security,
    check_nesting_depth,
    build_schema_from_widget,
)
from layer2.reconciliation.types import MismatchClass, WidgetSchema, FieldSchema
from layer2.reconciliation.errors import SecurityViolation


class TestValueClassification:
    """Tests for classify_value_type()."""

    def test_string_number_representational(self):
        """String numbers should be REPRESENTATIONAL_EQUIVALENCE."""
        mismatch_class, reason, rewritable = classify_value_type("42", "number")

        assert mismatch_class == MismatchClass.REPRESENTATIONAL_EQUIVALENCE
        assert rewritable is True

    def test_string_float_representational(self):
        """String floats should be REPRESENTATIONAL_EQUIVALENCE."""
        mismatch_class, reason, rewritable = classify_value_type("3.14", "number")

        assert mismatch_class == MismatchClass.REPRESENTATIONAL_EQUIVALENCE
        assert rewritable is True

    def test_string_with_unit_ambiguous(self):
        """String with unit should be UNKNOWN_AMBIGUOUS."""
        mismatch_class, reason, rewritable = classify_value_type("500 kW", "number")

        assert mismatch_class == MismatchClass.UNKNOWN_AMBIGUOUS
        assert rewritable is False

    def test_invalid_string_semantic(self):
        """Invalid string for number should be SEMANTIC_DIFFERENCE."""
        mismatch_class, reason, rewritable = classify_value_type("not a number", "number")

        assert mismatch_class == MismatchClass.SEMANTIC_DIFFERENCE
        assert rewritable is False

    def test_string_boolean_representational(self):
        """String booleans should be REPRESENTATIONAL_EQUIVALENCE."""
        for val in ["true", "True", "TRUE", "false", "False", "FALSE"]:
            mismatch_class, reason, rewritable = classify_value_type(val, "boolean")
            assert mismatch_class == MismatchClass.REPRESENTATIONAL_EQUIVALENCE

    def test_exact_type_match_none(self):
        """Exact type matches should be NONE."""
        mismatch_class, reason, rewritable = classify_value_type(42, "number")

        assert mismatch_class == MismatchClass.NONE
        assert rewritable is False


class TestSecurityChecks:
    """Tests for security pattern detection."""

    def test_sql_injection_detected(self):
        """SQL injection patterns should be detected."""
        violation = check_security("'; DROP TABLE users; --", "test_field")
        assert violation is not None
        assert "SQL injection" in violation

    def test_xss_script_detected(self):
        """XSS script tags should be detected."""
        violation = check_security("<script>alert('xss')</script>", "test_field")
        assert violation is not None
        assert "XSS" in violation

    def test_xss_onclick_detected(self):
        """XSS event handlers should be detected."""
        violation = check_security('<div onclick="evil()">click</div>', "test_field")
        assert violation is not None

    def test_clean_string_passes(self):
        """Clean strings should pass security check."""
        violation = check_security("Normal text with numbers 42", "test_field")
        assert violation is None


class TestNestingDepth:
    """Tests for nesting depth check (DoS protection)."""

    def test_shallow_nesting_ok(self):
        """Shallow nesting should pass."""
        data = {"a": {"b": {"c": 1}}}
        depth = check_nesting_depth(data)
        assert depth <= 10

    def test_deep_nesting_detected(self):
        """Deep nesting should be detected."""
        data = {"a": {"a": {"a": {"a": {"a": {"a": {"a": {"a": {"a": {"a": {"a": {"a": 1}}}}}}}}}}}}
        depth = check_nesting_depth(data)
        assert depth > 10


class TestClassifyMismatch:
    """Tests for classify_mismatch()."""

    def test_classify_valid_data(self, kpi_schema):
        """Valid data should have no mismatches."""
        data = {
            "label": "Power",
            "value": 500,
            "unit": "kW",
        }

        report = classify_mismatch(data, kpi_schema)

        assert report.overall_class == MismatchClass.NONE
        assert len(report.field_mismatches) == 0
        assert len(report.missing_required) == 0

    def test_classify_string_number(self, kpi_schema):
        """String numbers should be classified as rewritable."""
        data = {
            "label": "Power",
            "value": "500",
            "unit": "kW",
        }

        report = classify_mismatch(data, kpi_schema)

        assert report.rewritable_count >= 1
        value_mismatch = next(
            (m for m in report.field_mismatches if m.field_path == "value"),
            None
        )
        assert value_mismatch is not None
        assert value_mismatch.rewritable is True

    def test_classify_demodata_wrapper(self, kpi_schema):
        """demoData wrapper should be classified as structural."""
        data = {
            "demoData": {
                "label": "Power",
                "value": 500,
                "unit": "kW",
            }
        }

        report = classify_mismatch(data, kpi_schema)

        # Should detect structural wrapper
        root_mismatch = next(
            (m for m in report.field_mismatches if m.field_path == "root"),
            None
        )
        if root_mismatch:
            assert root_mismatch.mismatch_class == MismatchClass.STRUCTURAL_EQUIVALENCE

    def test_classify_missing_required(self, kpi_schema):
        """Missing required fields should be detected."""
        data = {
            "label": "Power",
            # Missing value and unit
        }

        report = classify_mismatch(data, kpi_schema)

        assert len(report.missing_required) >= 1

    def test_classify_security_violation(self, kpi_schema):
        """Security violations should raise immediately."""
        data = {
            "label": "<script>alert('xss')</script>",
            "value": 500,
            "unit": "kW",
        }

        with pytest.raises(SecurityViolation):
            classify_mismatch(data, kpi_schema)

    def test_classify_requires_resolution(self, kpi_schema):
        """Ambiguous data should require resolution."""
        data = {
            "label": "Power",
            "value": "500 kW",  # Value with embedded unit
            "unit": "",
        }

        report = classify_mismatch(data, kpi_schema)

        assert report.requires_resolution is True


class TestBuildSchema:
    """Tests for build_schema_from_widget()."""

    def test_build_kpi_schema(self):
        """Should build schema from widget registry."""
        schema = build_schema_from_widget("kpi")

        assert schema.scenario == "kpi"
        assert len(schema.required_fields) >= 1

    def test_build_unknown_scenario(self):
        """Should handle unknown scenarios gracefully."""
        schema = build_schema_from_widget("unknown_widget")

        assert schema.scenario == "unknown_widget"
        # Should have empty fields for unknown
