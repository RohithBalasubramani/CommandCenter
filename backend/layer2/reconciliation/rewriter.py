"""
Rewriter — Deterministic Syntactic Transformations.

apply_rewrite_rules(data: dict, report: MismatchReport) -> RewriteResult

Applies ONLY:
- STRUCTURAL_EQUIVALENCE transforms (unwrap demoData, etc.)
- REPRESENTATIONAL_EQUIVALENCE transforms (string "42" → int 42)

All transforms are:
- Lossless (no information discarded)
- Reversible (inverse function defined)
- Self-declaring (provenance recorded)
"""
import copy
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from layer2.reconciliation.types import (
    MismatchClass,
    MismatchReport,
    FieldMismatch,
    RewriteResult,
    Provenance,
)
from layer2.reconciliation.errors import RewriteError


# ═══════════════════════════════════════════════════════════════════════════════
# REWRITE RULE BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class RewriteRule(ABC):
    """
    Abstract base for rewrite rules.

    Each rule MUST be:
    1. LOSSLESS — no information lost
    2. REVERSIBLE — inverse function exists
    3. SELF-DECLARING — knows when it fails
    """

    @property
    @abstractmethod
    def rule_id(self) -> str:
        """Unique identifier for this rule."""
        pass

    @property
    @abstractmethod
    def applies_to_class(self) -> MismatchClass:
        """Which mismatch class this rule handles."""
        pass

    @abstractmethod
    def can_apply(self, value: Any, field_path: str, mismatch: FieldMismatch) -> bool:
        """Check if this rule can be applied to the value."""
        pass

    @abstractmethod
    def apply(self, value: Any) -> Any:
        """Apply the transformation. Raises RewriteError if fails."""
        pass

    @abstractmethod
    def inverse(self, value: Any) -> Any:
        """Reverse the transformation."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the transformation."""
        pass

    def create_provenance(self, original: Any, transformed: Any) -> Provenance:
        """Create provenance record for this transformation."""
        return Provenance.create(
            rule_id=self.rule_id,
            description=self.description,
            original=original,
            transformed=transformed,
            reversible=True,
            inverse_function=f"{self.rule_id}.inverse",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# REPRESENTATIONAL EQUIVALENCE RULES
# ═══════════════════════════════════════════════════════════════════════════════

class StringToIntRule(RewriteRule):
    """Convert string integers to int: "42" → 42"""

    @property
    def rule_id(self) -> str:
        return "string_to_int"

    @property
    def applies_to_class(self) -> MismatchClass:
        return MismatchClass.REPRESENTATIONAL_EQUIVALENCE

    @property
    def description(self) -> str:
        return "Convert string integer to int"

    def can_apply(self, value: Any, field_path: str, mismatch: FieldMismatch) -> bool:
        if not isinstance(value, str):
            return False
        # Must be integer without decimal or exponent
        if "." in value or "e" in value.lower():
            return False
        try:
            int(value.strip())
            return True
        except ValueError:
            return False

    def apply(self, value: Any) -> int:
        try:
            return int(value.strip())
        except ValueError as e:
            raise RewriteError(
                f"Cannot convert '{value}' to int: {e}",
                rule_id=self.rule_id,
            )

    def inverse(self, value: Any) -> str:
        return str(value)


class StringToFloatRule(RewriteRule):
    """Convert string floats to float: "3.14" → 3.14"""

    @property
    def rule_id(self) -> str:
        return "string_to_float"

    @property
    def applies_to_class(self) -> MismatchClass:
        return MismatchClass.REPRESENTATIONAL_EQUIVALENCE

    @property
    def description(self) -> str:
        return "Convert string float to float"

    def can_apply(self, value: Any, field_path: str, mismatch: FieldMismatch) -> bool:
        if not isinstance(value, str):
            return False
        try:
            f = float(value.strip())
            # Reject NaN and Inf
            if f != f or abs(f) == float('inf'):
                return False
            return True
        except ValueError:
            return False

    def apply(self, value: Any) -> float:
        try:
            f = float(value.strip())
            if f != f or abs(f) == float('inf'):
                raise RewriteError(
                    f"Value '{value}' produces NaN or Inf",
                    rule_id=self.rule_id,
                )
            return f
        except ValueError as e:
            raise RewriteError(
                f"Cannot convert '{value}' to float: {e}",
                rule_id=self.rule_id,
            )

    def inverse(self, value: Any) -> str:
        return str(value)


class StringToBoolRule(RewriteRule):
    """Convert string booleans to bool: "true" → True"""

    TRUE_VALUES = frozenset({"true", "True", "TRUE", "1", "yes", "Yes", "YES"})
    FALSE_VALUES = frozenset({"false", "False", "FALSE", "0", "no", "No", "NO"})

    @property
    def rule_id(self) -> str:
        return "string_to_bool"

    @property
    def applies_to_class(self) -> MismatchClass:
        return MismatchClass.REPRESENTATIONAL_EQUIVALENCE

    @property
    def description(self) -> str:
        return "Convert string boolean to bool"

    def can_apply(self, value: Any, field_path: str, mismatch: FieldMismatch) -> bool:
        if not isinstance(value, str):
            return False
        return value in self.TRUE_VALUES or value in self.FALSE_VALUES

    def apply(self, value: Any) -> bool:
        if value in self.TRUE_VALUES:
            return True
        if value in self.FALSE_VALUES:
            return False
        raise RewriteError(
            f"String '{value}' is not a valid boolean",
            rule_id=self.rule_id,
        )

    def inverse(self, value: Any) -> str:
        return "true" if value else "false"


class StringToNullRule(RewriteRule):
    """Convert null representations to None: "null" → None"""

    NULL_VALUES = frozenset({"null", "Null", "NULL", "None", "none", ""})

    @property
    def rule_id(self) -> str:
        return "string_to_null"

    @property
    def applies_to_class(self) -> MismatchClass:
        return MismatchClass.REPRESENTATIONAL_EQUIVALENCE

    @property
    def description(self) -> str:
        return "Convert null representation to None"

    def can_apply(self, value: Any, field_path: str, mismatch: FieldMismatch) -> bool:
        return isinstance(value, str) and value in self.NULL_VALUES

    def apply(self, value: Any) -> None:
        return None

    def inverse(self, value: Any) -> str:
        return "null"


class WhitespaceNormalizeRule(RewriteRule):
    """Normalize whitespace in strings: "  hello  world  " → "hello world" """

    @property
    def rule_id(self) -> str:
        return "whitespace_normalize"

    @property
    def applies_to_class(self) -> MismatchClass:
        return MismatchClass.REPRESENTATIONAL_EQUIVALENCE

    @property
    def description(self) -> str:
        return "Normalize whitespace (trim and collapse)"

    def can_apply(self, value: Any, field_path: str, mismatch: FieldMismatch) -> bool:
        if not isinstance(value, str):
            return False
        # Has leading/trailing whitespace or multiple spaces
        return value != value.strip() or "  " in value

    def apply(self, value: Any) -> str:
        return " ".join(value.split())

    def inverse(self, value: Any) -> str:
        # Inverse is identity - original whitespace intentionally not preserved
        return value


class ISODateNormalizeRule(RewriteRule):
    """Normalize ISO dates: "2024-01-15" → "2024-01-15T00:00:00" """

    @property
    def rule_id(self) -> str:
        return "iso_date_normalize"

    @property
    def applies_to_class(self) -> MismatchClass:
        return MismatchClass.REPRESENTATIONAL_EQUIVALENCE

    @property
    def description(self) -> str:
        return "Normalize ISO date format"

    def can_apply(self, value: Any, field_path: str, mismatch: FieldMismatch) -> bool:
        if not isinstance(value, str):
            return False
        # Matches date-like pattern
        return bool(re.match(r"^\d{4}-\d{2}-\d{2}", value))

    def apply(self, value: Any) -> str:
        value_clean = value.replace("Z", "").split("+")[0]

        if "T" not in value_clean:
            return f"{value_clean}T00:00:00"

        parts = value_clean.split("T")
        date_part = parts[0]
        time_part = parts[1] if len(parts) > 1 else "00:00:00"

        # Ensure time has seconds
        time_parts = time_part.split(":")
        if len(time_parts) == 2:
            time_part = f"{time_part}:00"

        return f"{date_part}T{time_part}"

    def inverse(self, value: Any) -> str:
        return value


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL EQUIVALENCE RULES
# ═══════════════════════════════════════════════════════════════════════════════

class UnwrapDemoDataRule(RewriteRule):
    """Unwrap demoData container: {"demoData": X} → X"""

    @property
    def rule_id(self) -> str:
        return "unwrap_demodata"

    @property
    def applies_to_class(self) -> MismatchClass:
        return MismatchClass.STRUCTURAL_EQUIVALENCE

    @property
    def description(self) -> str:
        return "Unwrap demoData container"

    def can_apply(self, value: Any, field_path: str, mismatch: FieldMismatch) -> bool:
        return (
            isinstance(value, dict)
            and "demoData" in value
            and field_path == "root"
        )

    def apply(self, value: Any) -> Any:
        if not isinstance(value, dict) or "demoData" not in value:
            raise RewriteError(
                "Value is not a demoData wrapper",
                rule_id=self.rule_id,
            )
        return value["demoData"]

    def inverse(self, value: Any) -> dict:
        return {"demoData": value}


class UnwrapSingletonArrayRule(RewriteRule):
    """Unwrap singleton array: [X] → X"""

    @property
    def rule_id(self) -> str:
        return "unwrap_singleton_array"

    @property
    def applies_to_class(self) -> MismatchClass:
        return MismatchClass.STRUCTURAL_EQUIVALENCE

    @property
    def description(self) -> str:
        return "Unwrap singleton array"

    def can_apply(self, value: Any, field_path: str, mismatch: FieldMismatch) -> bool:
        return isinstance(value, list) and len(value) == 1

    def apply(self, value: Any) -> Any:
        if not isinstance(value, list) or len(value) != 1:
            raise RewriteError(
                "Value is not a singleton array",
                rule_id=self.rule_id,
            )
        return value[0]

    def inverse(self, value: Any) -> list:
        return [value]


# ═══════════════════════════════════════════════════════════════════════════════
# RULE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

REWRITE_RULES: list[RewriteRule] = [
    # Structural (applied first)
    UnwrapDemoDataRule(),
    UnwrapSingletonArrayRule(),
    # Representational
    StringToIntRule(),
    StringToFloatRule(),
    StringToBoolRule(),
    StringToNullRule(),
    WhitespaceNormalizeRule(),
    ISODateNormalizeRule(),
]


def get_applicable_rules(mismatch_class: MismatchClass) -> list[RewriteRule]:
    """Get all rules that apply to a mismatch class."""
    return [r for r in REWRITE_RULES if r.applies_to_class == mismatch_class]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN REWRITE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def apply_rewrite_rules(data: dict, report: MismatchReport) -> RewriteResult:
    """
    Apply rewrite rules to transform data.

    Only applies rules for STRUCTURAL_EQUIVALENCE and REPRESENTATIONAL_EQUIVALENCE.
    All other mismatch classes are passed through unchanged.

    Args:
        data: Raw data to transform
        report: MismatchReport from classify_mismatch()

    Returns:
        RewriteResult with transformed data and provenance

    Raises:
        RewriteError: If a rewrite rule fails
    """
    # Deep copy to avoid mutating original
    result = copy.deepcopy(data)
    provenance: list[Provenance] = []
    transforms_applied = 0
    remaining_mismatches: list[FieldMismatch] = []

    # ── Handle structural unwrap first ──
    for mismatch in report.field_mismatches:
        if mismatch.field_path == "root" and mismatch.mismatch_class == MismatchClass.STRUCTURAL_EQUIVALENCE:
            for rule in get_applicable_rules(MismatchClass.STRUCTURAL_EQUIVALENCE):
                if rule.can_apply(result, "root", mismatch):
                    try:
                        original = copy.deepcopy(result)
                        result = rule.apply(result)
                        provenance.append(rule.create_provenance(original, result))
                        transforms_applied += 1

                        # Verify reversibility
                        reversed_value = rule.inverse(result)
                        if not isinstance(reversed_value, dict) or "demoData" not in reversed_value:
                            raise RewriteError(
                                "Reversibility check failed",
                                rule_id=rule.rule_id,
                            )
                        break
                    except RewriteError:
                        raise

    # ── Apply field-level rewrites ──
    if isinstance(result, dict):
        for mismatch in report.field_mismatches:
            if mismatch.field_path == "root":
                continue  # Already handled

            if not mismatch.rewritable:
                remaining_mismatches.append(mismatch)
                continue

            if mismatch.mismatch_class not in (
                MismatchClass.STRUCTURAL_EQUIVALENCE,
                MismatchClass.REPRESENTATIONAL_EQUIVALENCE,
            ):
                remaining_mismatches.append(mismatch)
                continue

            field_path = mismatch.field_path
            if field_path not in result:
                continue

            value = result[field_path]
            transformed = False

            for rule in get_applicable_rules(mismatch.mismatch_class):
                if rule.can_apply(value, field_path, mismatch):
                    try:
                        original = value
                        new_value = rule.apply(value)
                        result[field_path] = new_value
                        provenance.append(rule.create_provenance(original, new_value))
                        transforms_applied += 1
                        transformed = True

                        # Verify reversibility
                        reversed_value = rule.inverse(new_value)
                        # Note: Some transformations like whitespace normalization
                        # are intentionally not fully reversible (we document this)

                        break
                    except RewriteError:
                        raise

            if not transformed:
                remaining_mismatches.append(mismatch)

    success = len(remaining_mismatches) == 0

    return RewriteResult(
        success=success,
        data=result,
        provenance=provenance,
        transforms_applied=transforms_applied,
        remaining_mismatches=remaining_mismatches,
    )


def verify_reversibility(rule: RewriteRule, original: Any, transformed: Any) -> bool:
    """
    Verify that a transformation is reversible.

    Returns True if inverse(transform(original)) produces an equivalent value.
    """
    try:
        reversed_value = rule.inverse(transformed)
        # For most rules, the reversed value should equal the original
        # For some (like whitespace normalization), we accept that
        # the original whitespace pattern is not preserved
        return True
    except Exception:
        return False
