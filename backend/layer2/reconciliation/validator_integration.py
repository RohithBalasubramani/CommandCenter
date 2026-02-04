"""
Validator Integration — The Final Gate.

This module enforces the INVARIANT:
  Final output MUST pass validate_widget_data() or be explicit refusal.

NO BYPASS IS ALLOWED.

The validate_final() function is the ONLY exit point for reconciled data.
It either:
1. Returns validated data (success)
2. Raises ValidationGateError (failure with structured info)
"""
import logging
from typing import Optional

from layer2.reconciliation.types import Provenance
from layer2.reconciliation.errors import ValidationGateError

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION GATE
# ═══════════════════════════════════════════════════════════════════════════════

# Flag to track if validation was invoked (for CI gating tests)
_validation_invoked = False


def reset_validation_flag() -> None:
    """Reset the validation invocation flag (for testing)."""
    global _validation_invoked
    _validation_invoked = False


def was_validation_invoked() -> bool:
    """Check if validation was invoked (for CI gating tests)."""
    return _validation_invoked


def validate_final(
    scenario: str,
    data: dict,
    provenance: list[Provenance],
) -> dict:
    """
    FINAL VALIDATION GATE.

    This is the ONLY way reconciled data can exit the pipeline.

    INVARIANT: Either returns validated data or raises ValidationGateError.
    There is NO bypass. This function MUST be called before rendering.

    Args:
        scenario: Widget scenario name
        data: Reconciled data to validate
        provenance: Transformation provenance for audit

    Returns:
        Validated data (unchanged if valid)

    Raises:
        ValidationGateError: If validation fails (HARD FAILURE)
    """
    global _validation_invoked
    _validation_invoked = True

    logger.info(f"VALIDATION GATE: Validating {scenario} with {len(provenance)} transforms")

    # Import the actual validator
    try:
        from layer2.widget_schemas import validate_widget_data, ValidationError
    except ImportError as e:
        raise ValidationGateError(
            message="Validator not available - cannot proceed",
            validation_errors=[f"Import error: {e}"],
            scenario=scenario,
        )

    # Call the strict validator
    try:
        validate_widget_data(scenario, data)
        logger.info(f"VALIDATION GATE: {scenario} PASSED")
        return data

    except ValidationError as e:
        logger.error(f"VALIDATION GATE: {scenario} FAILED - {e.errors}")

        # Create structured error with full context
        raise ValidationGateError(
            message=f"Validation failed for {scenario}: {e.errors}",
            validation_errors=e.errors,
            scenario=scenario,
        )


def validate_or_refuse(
    scenario: str,
    data: Optional[dict],
    provenance: list[Provenance],
) -> tuple[bool, Optional[dict], Optional[list[str]]]:
    """
    Validate data, returning success status instead of raising.

    Use this when you want to handle validation failure gracefully
    (e.g., to produce a structured refusal).

    Args:
        scenario: Widget scenario name
        data: Data to validate (may be None)
        provenance: Transformation provenance

    Returns:
        (success, validated_data, errors)
    """
    if data is None:
        return False, None, ["Data is None - cannot validate"]

    try:
        validated = validate_final(scenario, data, provenance)
        return True, validated, None
    except ValidationGateError as e:
        return False, None, e.validation_errors


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-VALIDATION CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def pre_validate_structure(scenario: str, data: dict) -> list[str]:
    """
    Quick structural check before full validation.

    This is a fast-path check to catch obvious issues early.
    It does NOT replace the full validation gate.

    Returns list of issues found (empty if OK).
    """
    issues = []

    if data is None:
        issues.append("Data is None")
        return issues

    if not isinstance(data, dict):
        issues.append(f"Data must be dict, got {type(data).__name__}")
        return issues

    # Check for required wrapper structure
    if scenario != "trends-cumulative":
        if "demoData" in data:
            demo = data["demoData"]
            if not isinstance(demo, (dict, list)):
                issues.append(f"demoData must be dict or list, got {type(demo).__name__}")

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# ASSERTION HELPERS (for CI)
# ═══════════════════════════════════════════════════════════════════════════════

def assert_validation_invoked() -> None:
    """
    Assert that validation was invoked.

    Call this in CI tests to verify the validation gate was hit.

    Raises:
        AssertionError: If validation was not invoked
    """
    if not _validation_invoked:
        raise AssertionError(
            "VALIDATION GATE BYPASS DETECTED: validate_final() was not called. "
            "This is a CRITICAL security violation. All data MUST pass through "
            "the validation gate before rendering."
        )


def create_validation_bypass_test():
    """
    Create a test that fails if validation is bypassed.

    Returns a test function for pytest.
    """
    def test_validation_not_bypassed():
        """Verify that validation gate was invoked."""
        assert_validation_invoked()

    return test_validation_not_bypassed
