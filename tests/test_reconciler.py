"""
Reconciler Tests — Proving domain-independence.

This test suite proves the reconciler works based on FORMAL PROPERTIES,
not domain knowledge. Each category tests a different kind of data.

KEY PRINCIPLE: These tests require NO domain knowledge to understand.
The reconciler knows NOTHING about:
- Units (kW, MW, psi, bar)
- Labels ("Pump A", "pump-a")
- Ranges (valid power ranges, valid temperatures)
- Semantics (what "efficiency" means)

It ONLY knows:
- Type representations (string "42" → int 42)
- Structure patterns ({"demoData": X} → X)
- Format normalization (whitespace, dates)
"""
import sys
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "command_center.settings")

import django
django.setup()

from layer2.reconciler import (
    Reconciler,
    MismatchClass,
    ReconciliationResult,
    # Rules
    StringToIntRule,
    StringToFloatRule,
    StringToBoolRule,
    StringToNullRule,
    WhitespaceNormalizationRule,
    ISODateNormalizationRule,
    UnwrapDemoDataRule,
    UnwrapSingletonArrayRule,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 1: NUMBERS
# Representational equivalence: string ↔ number
# ═══════════════════════════════════════════════════════════════════════════════

def test_number_string_to_int():
    """String "42" → int 42 (lossless)"""
    rule = StringToIntRule()
    assert rule.applies("42")
    assert rule.transform("42") == 42
    assert rule.inverse(42) == "42"
    print("✓ NUMBER: '42' → 42 (reversible)")


def test_number_string_to_float():
    """String "3.14159" → float 3.14159 (lossless)"""
    rule = StringToFloatRule()
    assert rule.applies("3.14159")
    assert rule.transform("3.14159") == 3.14159
    print("✓ NUMBER: '3.14159' → 3.14159")


def test_number_invalid_string_fails():
    """String "not-a-number" → REFUSES to transform"""
    rule = StringToIntRule()
    assert not rule.applies("not-a-number")
    failure = rule.fails_when("not-a-number")
    assert failure is not None
    print(f"✓ NUMBER: 'not-a-number' correctly REFUSED ({failure})")


def test_number_nan_fails():
    """String "nan" → REFUSES (NaN is not valid data)"""
    rule = StringToFloatRule()
    failure = rule.fails_when("nan")
    assert failure is not None
    print(f"✓ NUMBER: 'nan' correctly REFUSED ({failure})")


def test_number_infinity_fails():
    """String "inf" → REFUSES (infinity is not valid data)"""
    rule = StringToFloatRule()
    failure = rule.fails_when("inf")
    assert failure is not None
    print(f"✓ NUMBER: 'inf' correctly REFUSED ({failure})")


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 2: STRINGS
# Representational equivalence: whitespace normalization
# ═══════════════════════════════════════════════════════════════════════════════

def test_string_whitespace_normalization():
    """'  hello   world  ' → 'hello world' (semantic whitespace irrelevant)"""
    rule = WhitespaceNormalizationRule()
    assert rule.applies("  hello   world  ")
    assert rule.transform("  hello   world  ") == "hello world"
    print("✓ STRING: '  hello   world  ' → 'hello world'")


def test_string_trim():
    """'  trimmed  ' → 'trimmed'"""
    rule = WhitespaceNormalizationRule()
    assert rule.transform("  trimmed  ") == "trimmed"
    print("✓ STRING: '  trimmed  ' → 'trimmed'")


def test_string_multiline_collapse():
    """'line1\\n  line2' → 'line1 line2' (newlines are whitespace)"""
    rule = WhitespaceNormalizationRule()
    assert rule.transform("line1\n  line2") == "line1 line2"
    print("✓ STRING: 'line1\\n  line2' → 'line1 line2'")


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 3: BOOLEANS
# Representational equivalence: string ↔ bool
# ═══════════════════════════════════════════════════════════════════════════════

def test_bool_true_representations():
    """Various true representations → true"""
    rule = StringToBoolRule()
    for val in ["true", "True", "TRUE", "1", "yes", "Yes", "YES"]:
        assert rule.applies(val)
        assert rule.transform(val) is True
    print("✓ BOOLEAN: 'true', 'True', '1', 'yes', etc. → true")


def test_bool_false_representations():
    """Various false representations → false"""
    rule = StringToBoolRule()
    for val in ["false", "False", "FALSE", "0", "no", "No", "NO"]:
        assert rule.applies(val)
        assert rule.transform(val) is False
    print("✓ BOOLEAN: 'false', 'False', '0', 'no', etc. → false")


def test_bool_invalid_fails():
    """'maybe' → REFUSES (not a boolean)"""
    rule = StringToBoolRule()
    assert not rule.applies("maybe")
    failure = rule.fails_when("maybe")
    assert failure is not None
    print(f"✓ BOOLEAN: 'maybe' correctly REFUSED ({failure})")


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 4: NULL / EMPTY
# Representational equivalence: string ↔ null
# ═══════════════════════════════════════════════════════════════════════════════

def test_null_representations():
    """Various null representations → null"""
    rule = StringToNullRule()
    for val in ["null", "Null", "NULL", "None", "none", ""]:
        assert rule.applies(val)
        assert rule.transform(val) is None
    print("✓ NULL: 'null', 'None', '' → null")


def test_null_inverse():
    """null → 'null' (canonical form)"""
    rule = StringToNullRule()
    assert rule.inverse(None) == "null"
    print("✓ NULL: null → 'null' (inverse)")


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 5: TIME / DATES
# Representational equivalence: ISO date formats
# ═══════════════════════════════════════════════════════════════════════════════

def test_date_only_to_datetime():
    """'2024-01-15' → '2024-01-15T00:00:00'"""
    rule = ISODateNormalizationRule()
    assert rule.applies("2024-01-15")
    assert rule.transform("2024-01-15") == "2024-01-15T00:00:00"
    print("✓ TIME: '2024-01-15' → '2024-01-15T00:00:00'")


def test_datetime_z_suffix():
    """'2024-01-15T10:30:00Z' → '2024-01-15T10:30:00'"""
    rule = ISODateNormalizationRule()
    assert rule.applies("2024-01-15T10:30:00Z")
    result = rule.transform("2024-01-15T10:30:00Z")
    assert result == "2024-01-15T10:30:00"
    print("✓ TIME: '2024-01-15T10:30:00Z' → '2024-01-15T10:30:00'")


def test_datetime_no_seconds():
    """'2024-01-15T10:30' → '2024-01-15T10:30:00'"""
    rule = ISODateNormalizationRule()
    assert rule.applies("2024-01-15T10:30")
    result = rule.transform("2024-01-15T10:30")
    assert result == "2024-01-15T10:30:00"
    print("✓ TIME: '2024-01-15T10:30' → '2024-01-15T10:30:00'")


def test_invalid_date_fails():
    """'not-a-date' → REFUSES"""
    rule = ISODateNormalizationRule()
    assert not rule.applies("not-a-date")
    print("✓ TIME: 'not-a-date' correctly REFUSED")


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 6: STRUCTURES
# Structural equivalence: wrapper patterns
# ═══════════════════════════════════════════════════════════════════════════════

def test_structure_unwrap_demodata():
    """{"demoData": X} → X"""
    rule = UnwrapDemoDataRule()
    data = {"demoData": {"value": 42, "label": "Test"}}
    assert rule.applies(data)
    assert rule.transform(data) == {"value": 42, "label": "Test"}
    print("✓ STRUCTURE: {'demoData': X} → X")


def test_structure_unwrap_demodata_reversible():
    """X → {"demoData": X} (inverse)"""
    rule = UnwrapDemoDataRule()
    data = {"value": 42}
    assert rule.inverse(data) == {"demoData": {"value": 42}}
    print("✓ STRUCTURE: X → {'demoData': X} (inverse)")


def test_structure_singleton_unwrap():
    """[X] → X (singleton array)"""
    rule = UnwrapSingletonArrayRule()
    assert rule.applies([{"value": 42}])
    assert rule.transform([{"value": 42}]) == {"value": 42}
    print("✓ STRUCTURE: [X] → X (singleton)")


def test_structure_non_singleton_fails():
    """[X, Y] → REFUSES (not singleton)"""
    rule = UnwrapSingletonArrayRule()
    assert not rule.applies([1, 2])
    failure = rule.fails_when([1, 2])
    assert failure is not None
    print(f"✓ STRUCTURE: [X, Y] correctly REFUSED ({failure})")


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 7: UNITS — PROVING WE DON'T TOUCH THEM
# Units require domain knowledge → UNKNOWN/AMBIGUOUS → REFUSED
# ═══════════════════════════════════════════════════════════════════════════════

def test_units_not_transformed():
    """
    500 kW vs 0.5 MW — reconciler does NOT touch this.

    This requires domain knowledge to know they're equivalent.
    The reconciler is SOUND: it refuses rather than guess.
    """
    reconciler = Reconciler()

    # Data with unit information — reconciler doesn't know what to do
    data = {"value": 500, "unit": "kW"}
    result = reconciler.reconcile(data)

    # The reconciler should NOT change the value
    assert result.data["value"] == 500
    assert result.data["unit"] == "kW"

    # No transforms applied to numeric values (they're already numbers)
    value_transforms = [t for t in result.transforms if "value" in t.field_path]
    assert len(value_transforms) == 0

    print("✓ UNITS: 500 kW NOT transformed (requires domain knowledge)")


def test_units_comparison_untouched():
    """
    Comparison of 500 kW vs 0.5 MW — reconciler leaves both values alone.

    The normalization layer (dimensions.py) handles this.
    The reconciler handles only syntactic transformations.
    """
    reconciler = Reconciler()

    data = {
        "valueA": 500,
        "valueB": 0.5,
        "unitA": "kW",
        "unitB": "MW",
    }
    result = reconciler.reconcile(data)

    # Values unchanged — reconciler doesn't do unit math
    assert result.data["valueA"] == 500
    assert result.data["valueB"] == 0.5
    assert result.data["unitA"] == "kW"
    assert result.data["unitB"] == "MW"

    print("✓ UNITS: 500 kW vs 0.5 MW — both values preserved (no domain guessing)")


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 8: LABELS — PROVING WE DON'T TOUCH THEM
# Labels require domain knowledge → REFUSED
# ═══════════════════════════════════════════════════════════════════════════════

def test_labels_not_transformed():
    """
    "Pump A" vs "pump-a" — reconciler does NOT make this equivalence.

    This requires domain knowledge (are these the same pump?).
    The reconciler refuses rather than guess.
    """
    reconciler = Reconciler()

    data = {"label": "Pump A", "id": "pump-a"}
    result = reconciler.reconcile(data)

    # Labels unchanged — reconciler doesn't guess equivalences
    assert result.data["label"] == "Pump A"
    assert result.data["id"] == "pump-a"

    print("✓ LABELS: 'Pump A' vs 'pump-a' — NOT equated (no domain guessing)")


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_full_pipeline_mixed_data():
    """
    Full reconciliation of LLM-style output with various type mismatches.
    """
    reconciler = Reconciler()

    # Typical LLM output with string numbers, date strings, whitespace issues
    # Note: We don't wrap in demoData since the unwrap rule will extract it
    llm_output = {
        "value": "42",           # String number → should become 42
        "enabled": "true",       # String bool → should become true
        "timestamp": "2024-01-15", # Date without time → should normalize
        "label": "  Test Label  ",  # Extra whitespace → should trim
        "nested": {
            "count": "100",      # Nested string number
        },
    }

    result = reconciler.reconcile(llm_output)

    # Check transformations applied
    assert result.data["value"] == 42
    assert result.data["enabled"] is True
    assert result.data["timestamp"] == "2024-01-15T00:00:00"
    assert result.data["label"] == "Test Label"
    assert result.data["nested"]["count"] == 100

    print("✓ PIPELINE: Full reconciliation with mixed type issues")
    print(f"  Applied {len(result.transforms)} transformations")
    for t in result.transforms:
        print(f"    [{t.field_path}] {t.rule_name}: {t.original!r} → {t.transformed!r}")


def test_full_pipeline_soundness():
    """
    Prove SOUNDNESS: reconciler never lies.

    Even with ambiguous data, it either:
    1. Transforms correctly
    2. Refuses and reports why

    It never guesses wrong.
    """
    reconciler = Reconciler()

    # Data with ambiguous intent
    data = {
        "value": "maybe",      # Not a valid number, bool, or null
        "ratio": "1.5x",       # Looks numeric but has suffix
        "date": "yesterday",   # Not a valid ISO date
    }

    result = reconciler.reconcile(data)

    # These values should be UNCHANGED (not guessed)
    assert result.data["value"] == "maybe"
    assert result.data["ratio"] == "1.5x"
    assert result.data["date"] == "yesterday"

    print("✓ SOUNDNESS: Ambiguous values NOT transformed (no guessing)")


# ═══════════════════════════════════════════════════════════════════════════════
# MISMATCH CLASSIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_classification_representational():
    """String "42" → int classified as REPRESENTATIONAL_EQUIVALENCE"""
    reconciler = Reconciler()
    classification = reconciler.classify_mismatch("42", int)

    assert classification.mismatch_class == MismatchClass.REPRESENTATIONAL_EQUIVALENCE
    assert classification.transformable is True
    print("✓ CLASSIFICATION: '42' → int is REPRESENTATIONAL_EQUIVALENCE")


def test_classification_semantic():
    """String "hello" → int classified as SEMANTIC_DIFFERENCE"""
    reconciler = Reconciler()
    classification = reconciler.classify_mismatch("hello", int)

    assert classification.mismatch_class == MismatchClass.SEMANTIC_DIFFERENCE
    assert classification.transformable is False
    print("✓ CLASSIFICATION: 'hello' → int is SEMANTIC_DIFFERENCE")


def test_classification_structural():
    """
    Wrapped dict: types already match (dict → dict).

    Structural unwrapping (demoData extraction) happens in reconcile(),
    not in classify_mismatch(). Classification is for type mismatches.
    """
    reconciler = Reconciler()
    classification = reconciler.classify_mismatch({"demoData": {"value": 1}}, dict)

    # Types match (both dict), so it's already structurally equivalent
    assert classification.mismatch_class == MismatchClass.STRUCTURAL_EQUIVALENCE
    assert classification.transformable is False  # No type transform needed
    print("✓ CLASSIFICATION: dict → dict types match (unwrapping in reconcile())")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    print("=" * 70)
    print("  RECONCILER TESTS — Domain-Independent Transformation")
    print("=" * 70)
    print()

    categories = [
        ("NUMBERS", [
            test_number_string_to_int,
            test_number_string_to_float,
            test_number_invalid_string_fails,
            test_number_nan_fails,
            test_number_infinity_fails,
        ]),
        ("STRINGS", [
            test_string_whitespace_normalization,
            test_string_trim,
            test_string_multiline_collapse,
        ]),
        ("BOOLEANS", [
            test_bool_true_representations,
            test_bool_false_representations,
            test_bool_invalid_fails,
        ]),
        ("NULL/EMPTY", [
            test_null_representations,
            test_null_inverse,
        ]),
        ("TIME/DATES", [
            test_date_only_to_datetime,
            test_datetime_z_suffix,
            test_datetime_no_seconds,
            test_invalid_date_fails,
        ]),
        ("STRUCTURES", [
            test_structure_unwrap_demodata,
            test_structure_unwrap_demodata_reversible,
            test_structure_singleton_unwrap,
            test_structure_non_singleton_fails,
        ]),
        ("UNITS (DOMAIN-INDEPENDENT)", [
            test_units_not_transformed,
            test_units_comparison_untouched,
        ]),
        ("LABELS (DOMAIN-INDEPENDENT)", [
            test_labels_not_transformed,
        ]),
        ("FULL PIPELINE", [
            test_full_pipeline_mixed_data,
            test_full_pipeline_soundness,
        ]),
        ("CLASSIFICATION", [
            test_classification_representational,
            test_classification_semantic,
            test_classification_structural,
        ]),
    ]

    passed = 0
    failed = 0

    for category_name, tests in categories:
        print(f"\n── {category_name} ──")
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
    print()
    print("  KEY INSIGHT: The reconciler knows NOTHING about:")
    print("  - Units (kW, MW, psi, bar)")
    print("  - Labels (what 'Pump A' means)")
    print("  - Ranges (valid power values)")
    print("  - Semantics (what 'efficiency' means)")
    print()
    print("  It ONLY knows:")
    print("  - String '42' represents integer 42")
    print("  - String 'true' represents boolean true")
    print("  - ISO date formats are equivalent")
    print("  - Whitespace is not semantic content")
    print("  - {'demoData': X} and X are structurally equivalent")
    print()
    print("  This scales to MILLIONS of cases because the rules are")
    print("  SYNTACTIC (about representation) not SEMANTIC (about meaning).")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
