"""
Tests for the Widget Normalizer.

Verifies:
1. Lossless transformations work correctly
2. Ambiguous cases FAIL (not guess)
3. Validation remains strict after normalization
"""
import sys
import os
from pathlib import Path

# Setup Django
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "command_center.settings")

import django
django.setup()

from layer2.widget_normalizer import (
    normalize_widget_data,
    normalize_and_validate,
    NormalizationError,
)
from layer2.widget_schemas import ValidationError


def test_kw_to_kw_no_change():
    """Test: kW values stay as kW (no conversion needed)."""
    data = {
        "demoData": {
            "label": "Power",
            "value": 500,
            "unit": "kW"
        }
    }

    result = normalize_widget_data("kpi", data)

    assert result.data["demoData"]["value"] == 500
    assert result.data["demoData"]["unit"] == "kW"
    print("✓ PASS: kW → kW (no change)")


def test_w_to_kw_conversion():
    """Test: W → kW conversion (lossless, unambiguous)."""
    data = {
        "demoData": {
            "label": "Power",
            "value": 5000,  # 5000 W
            "unit": "W"
        }
    }

    result = normalize_widget_data("kpi", data)

    assert result.data["demoData"]["value"] == 5.0  # 5 kW
    assert result.data["demoData"]["unit"] == "kW"
    assert len(result.transformations) > 0
    print(f"✓ PASS: W → kW (5000 W → 5.0 kW)")
    for t in result.transformations:
        print(f"  └─ {t.action}: {t.original} → {t.normalized}")


def test_mw_to_kw_conversion():
    """Test: MW → kW conversion (lossless, unambiguous)."""
    data = {
        "demoData": {
            "label": "Power",
            "value": 0.5,  # 0.5 MW
            "unit": "MW"
        }
    }

    result = normalize_widget_data("kpi", data)

    assert result.data["demoData"]["value"] == 500.0  # 500 kW
    assert result.data["demoData"]["unit"] == "kW"
    print(f"✓ PASS: MW → kW (0.5 MW → 500.0 kW)")


def test_string_to_number_conversion():
    """Test: String numbers → numeric (lossless)."""
    data = {
        "demoData": {
            "label": "Power",
            "value": "42",  # String
            "unit": "kW"
        }
    }

    result = normalize_widget_data("kpi", data)

    assert result.data["demoData"]["value"] == 42  # Integer
    assert isinstance(result.data["demoData"]["value"], int)
    print("✓ PASS: String → Number ('42' → 42)")


def test_placeholder_preserved():
    """Test: N/A placeholders are preserved (not converted)."""
    data = {
        "demoData": {
            "label": "Power",
            "value": "N/A",
            "unit": "kW"
        }
    }

    result = normalize_widget_data("kpi", data)

    assert result.data["demoData"]["value"] == "N/A"
    print("✓ PASS: Placeholder preserved ('N/A' stays 'N/A')")


def test_comparison_same_unit():
    """Test: Comparison with same units passes."""
    data = {
        "demoData": {
            "label": "Power Comparison",
            "unit": "kW",
            "labelA": "Pump 1",
            "valueA": 500,
            "labelB": "Pump 2",
            "valueB": 450,
        }
    }

    result = normalize_widget_data("comparison", data)

    # Should pass normalization
    assert result.data["demoData"]["valueA"] == 500
    assert result.data["demoData"]["valueB"] == 450
    print("✓ PASS: Comparison with same units")


def test_comparison_validates_after_normalization():
    """Test: Normalized comparison passes validation (no false positive)."""
    data = {
        "demoData": {
            "label": "Power Comparison",
            "unit": "kW",
            "labelA": "Pump 1",
            "valueA": 500,
            "labelB": "Pump 2",
            "valueB": 450,
        }
    }

    # This should NOT trigger the "suspicious ratio" check
    # because values are in the same order of magnitude
    result = normalize_and_validate("comparison", data)
    print("✓ PASS: Comparison validates after normalization")


def test_null_data_fails():
    """Test: Null data fails normalization (not guessed)."""
    try:
        normalize_widget_data("kpi", None)
        assert False, "Should have raised NormalizationError"
    except NormalizationError as e:
        print(f"✓ PASS: Null data fails normalization ({e.reason})")


def test_missing_unit_passes_through():
    """Test: Missing unit passes through to validation (not guessed)."""
    data = {
        "demoData": {
            "label": "Value",
            "value": 42,
            "unit": ""  # Empty unit
        }
    }

    # Normalization should pass (nothing to normalize)
    result = normalize_widget_data("kpi", data)
    assert result.data["demoData"]["value"] == 42
    print("✓ PASS: Missing unit passes through (not guessed)")


def test_invalid_data_still_fails_validation():
    """Test: Invalid data fails validation even after normalization."""
    data = {
        "demoData": {
            "label": "Efficiency",
            "value": -50,  # Invalid: negative percentage
            "unit": "%"
        }
    }

    try:
        normalize_and_validate("kpi", data)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        print(f"✓ PASS: Invalid data still fails validation")
        print(f"  └─ Errors: {e.errors}")


def test_transformation_provenance():
    """Test: All transformations are tracked with provenance."""
    data = {
        "demoData": {
            "label": "Power",
            "value": "5000",  # String
            "unit": "W"  # Will be converted to kW
        }
    }

    result = normalize_widget_data("kpi", data)

    # Should have transformations logged
    assert len(result.transformations) >= 1
    print(f"✓ PASS: Transformation provenance tracked ({len(result.transformations)} transformations)")
    for t in result.transformations:
        print(f"  └─ [{t.field}] {t.action}: {t.original} → {t.normalized}")
        print(f"     Reason: {t.reason}")


def run_all_tests():
    """Run all normalization tests."""
    print("=" * 70)
    print("  WIDGET NORMALIZER TESTS")
    print("=" * 70)
    print()

    tests = [
        test_kw_to_kw_no_change,
        test_w_to_kw_conversion,
        test_mw_to_kw_conversion,
        test_string_to_number_conversion,
        test_placeholder_preserved,
        test_comparison_same_unit,
        test_comparison_validates_after_normalization,
        test_null_data_fails,
        test_missing_unit_passes_through,
        test_invalid_data_still_fails_validation,
        test_transformation_provenance,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAIL: {test.__name__}")
            print(f"  └─ {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {test.__name__}")
            print(f"  └─ {type(e).__name__}: {e}")
            failed += 1
        print()

    print("=" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
