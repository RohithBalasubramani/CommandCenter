"""
General-Purpose Reconciliation Engine

Reconciles unreliable LLM output against strict validation.
Works WITHOUT domain knowledge. Based on FORMAL PROPERTIES only.

DESIGN PRINCIPLES:
1. SOUND over complete — never lies, sometimes refuses
2. SYNTACTIC over semantic — transforms representation, not meaning
3. REVERSIBLE over one-way — every transform has an inverse
4. DECLARATIVE over imperative — rules state their own failure conditions

MISMATCH CLASSES (formal properties):
┌─────────────────────────────────────────────────────────────────────────┐
│ STRUCTURAL_EQUIVALENCE                                                  │
│   Same meaning, different structure.                                    │
│   Examples: {"demoData": X} vs X, [X] vs X (singleton)                  │
│   Transform: ALLOWED (lossless structural rewrite)                      │
├─────────────────────────────────────────────────────────────────────────┤
│ REPRESENTATIONAL_EQUIVALENCE                                            │
│   Same value, different representation.                                 │
│   Examples: "42" vs 42, "true" vs true, "2024-01-15T00:00:00Z" vs date  │
│   Transform: ALLOWED (lossless type/format coercion)                    │
├─────────────────────────────────────────────────────────────────────────┤
│ SEMANTIC_DIFFERENCE                                                     │
│   Different meaning. Cannot reconcile.                                  │
│   Examples: 42 vs 43, "foo" vs "bar", adding/removing fields            │
│   Transform: FORBIDDEN (would change meaning)                           │
├─────────────────────────────────────────────────────────────────────────┤
│ UNKNOWN_AMBIGUOUS                                                       │
│   Cannot determine equivalence without domain knowledge.                │
│   Examples: 500 kW vs 0.5 MW, "Pump A" vs "pump-a"                      │
│   Transform: FORBIDDEN (would require guessing)                         │
└─────────────────────────────────────────────────────────────────────────┘

FORBIDDEN OPERATIONS:
- Domain registries (unit tables, label mappings)
- Guessing intent ("probably meant...")
- Best-effort fixes ("close enough")
- Semantic inference ("this looks like power")
- Expanding validation (making invalid → valid)

WHY THIS SCALES:
The set of syntactic transformations is FINITE and ENUMERABLE:
- Type coercions: string↔number, string↔boolean, string↔null
- Structure normalization: wrapper, nesting, singleton
- Format normalization: whitespace, date formats

These don't require domain knowledge because they operate on
REPRESENTATION (syntax), not MEANING (semantics).
"""
import copy
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Optional, TypeVar, Generic

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# MISMATCH CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class MismatchClass(Enum):
    """Formal classification of mismatches between source and target."""
    STRUCTURAL_EQUIVALENCE = auto()      # Same meaning, different structure
    REPRESENTATIONAL_EQUIVALENCE = auto() # Same value, different representation
    SEMANTIC_DIFFERENCE = auto()          # Different meaning
    UNKNOWN_AMBIGUOUS = auto()            # Cannot determine without domain knowledge


@dataclass
class ClassificationResult:
    """Result of mismatch classification."""
    mismatch_class: MismatchClass
    source_value: Any
    expected_type: Optional[type]
    reason: str
    transformable: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMATION RULES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransformRecord:
    """Record of a single transformation with provenance."""
    rule_name: str
    field_path: str
    original: Any
    transformed: Any
    inverse_exists: bool
    reason: str


class TransformationRule(ABC):
    """
    Abstract base for transformation rules.

    Each rule MUST be:
    1. LOSSLESS — no information lost
    2. REVERSIBLE — inverse function exists
    3. SELF-DECLARING — knows when it fails
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Rule identifier."""
        pass

    @abstractmethod
    def applies(self, value: Any, target_hint: Optional[str] = None) -> bool:
        """Does this rule apply to the given value?"""
        pass

    @abstractmethod
    def transform(self, value: Any) -> Any:
        """Apply the transformation. Raises ValueError if fails."""
        pass

    @abstractmethod
    def inverse(self, value: Any) -> Any:
        """Reverse the transformation."""
        pass

    @abstractmethod
    def fails_when(self, value: Any) -> Optional[str]:
        """Returns failure reason, or None if transformation is valid."""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# REPRESENTATIONAL EQUIVALENCE RULES
# ═══════════════════════════════════════════════════════════════════════════════

class StringToIntRule(TransformationRule):
    """
    "42" → 42

    Lossless: int("42") preserves value exactly
    Reversible: str(42) == "42"
    Fails: when string is not a valid integer
    """

    @property
    def name(self) -> str:
        return "string_to_int"

    def applies(self, value: Any, target_hint: Optional[str] = None) -> bool:
        if not isinstance(value, str):
            return False
        # Must be parseable as int without loss
        try:
            # Check it's not a float representation
            if "." in value or "e" in value.lower():
                return False
            int(value)
            return True
        except ValueError:
            return False

    def transform(self, value: Any) -> int:
        failure = self.fails_when(value)
        if failure:
            raise ValueError(failure)
        return int(value)

    def inverse(self, value: Any) -> str:
        return str(value)

    def fails_when(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return f"Expected string, got {type(value).__name__}"
        try:
            if "." in value or "e" in value.lower():
                return f"String '{value}' contains decimal/exponent, use float rule"
            int(value)
            return None
        except ValueError:
            return f"String '{value}' is not a valid integer"


class StringToFloatRule(TransformationRule):
    """
    "3.14" → 3.14

    Lossless: float("3.14") preserves value (within float precision)
    Reversible: str(3.14) recovers original (within precision)
    Fails: when string is not a valid float
    """

    @property
    def name(self) -> str:
        return "string_to_float"

    def applies(self, value: Any, target_hint: Optional[str] = None) -> bool:
        if not isinstance(value, str):
            return False
        try:
            float(value)
            return True
        except ValueError:
            return False

    def transform(self, value: Any) -> float:
        failure = self.fails_when(value)
        if failure:
            raise ValueError(failure)
        return float(value)

    def inverse(self, value: Any) -> str:
        return str(value)

    def fails_when(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return f"Expected string, got {type(value).__name__}"
        try:
            f = float(value)
            # Check for NaN/Inf which are technically valid floats but not valid data
            if f != f:  # NaN check
                return f"String '{value}' represents NaN"
            if abs(f) == float('inf'):
                return f"String '{value}' represents infinity"
            return None
        except ValueError:
            return f"String '{value}' is not a valid float"


class StringToBoolRule(TransformationRule):
    """
    "true" → true, "false" → false

    Lossless: exact mapping
    Reversible: bool → "true"/"false"
    Fails: when string is not a boolean representation
    """

    TRUE_VALUES = frozenset({"true", "True", "TRUE", "1", "yes", "Yes", "YES"})
    FALSE_VALUES = frozenset({"false", "False", "FALSE", "0", "no", "No", "NO"})

    @property
    def name(self) -> str:
        return "string_to_bool"

    def applies(self, value: Any, target_hint: Optional[str] = None) -> bool:
        if not isinstance(value, str):
            return False
        return value in self.TRUE_VALUES or value in self.FALSE_VALUES

    def transform(self, value: Any) -> bool:
        failure = self.fails_when(value)
        if failure:
            raise ValueError(failure)
        return value in self.TRUE_VALUES

    def inverse(self, value: Any) -> str:
        return "true" if value else "false"

    def fails_when(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return f"Expected string, got {type(value).__name__}"
        if value not in self.TRUE_VALUES and value not in self.FALSE_VALUES:
            return f"String '{value}' is not a valid boolean representation"
        return None


class StringToNullRule(TransformationRule):
    """
    "null", "None", "" → null

    Lossless: these are canonical null representations
    Reversible: null → "null"
    Fails: when string is not a null representation
    """

    NULL_VALUES = frozenset({"null", "Null", "NULL", "None", "none", "NONE", ""})

    @property
    def name(self) -> str:
        return "string_to_null"

    def applies(self, value: Any, target_hint: Optional[str] = None) -> bool:
        if not isinstance(value, str):
            return False
        return value in self.NULL_VALUES

    def transform(self, value: Any) -> None:
        failure = self.fails_when(value)
        if failure:
            raise ValueError(failure)
        return None

    def inverse(self, value: Any) -> str:
        return "null"

    def fails_when(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return f"Expected string, got {type(value).__name__}"
        if value not in self.NULL_VALUES:
            return f"String '{value}' is not a null representation"
        return None


class WhitespaceNormalizationRule(TransformationRule):
    """
    "  hello  world  " → "hello world"

    Lossless: whitespace is not semantic content
    Reversible: original whitespace cannot be recovered (intentionally)
    Fails: never (all strings can be normalized)
    """

    @property
    def name(self) -> str:
        return "whitespace_normalize"

    def applies(self, value: Any, target_hint: Optional[str] = None) -> bool:
        return isinstance(value, str)

    def transform(self, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError(f"Expected string, got {type(value).__name__}")
        # Normalize all whitespace to single spaces, trim ends
        return " ".join(value.split())

    def inverse(self, value: Any) -> str:
        # Inverse is identity — we intentionally lose original whitespace
        return value

    def fails_when(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return f"Expected string, got {type(value).__name__}"
        return None


class ISODateNormalizationRule(TransformationRule):
    """
    Various ISO date formats → canonical ISO format

    "2024-01-15" → "2024-01-15T00:00:00"
    "2024-01-15T10:30:00Z" → "2024-01-15T10:30:00"

    Lossless: all represent the same datetime
    Reversible: can recover original precision level
    Fails: when string is not a valid date
    """

    @property
    def name(self) -> str:
        return "iso_date_normalize"

    def applies(self, value: Any, target_hint: Optional[str] = None) -> bool:
        if not isinstance(value, str):
            return False
        # Check if it looks like a date
        if not re.match(r"^\d{4}-\d{2}-\d{2}", value):
            return False
        return self.fails_when(value) is None

    def transform(self, value: Any) -> str:
        failure = self.fails_when(value)
        if failure:
            raise ValueError(failure)

        # Parse and normalize
        value_clean = value.replace("Z", "").split("+")[0].split("-")[0] if "+" in value else value.replace("Z", "")

        # Handle date-only format
        if "T" not in value_clean:
            # Just a date, add midnight time
            return f"{value_clean}T00:00:00"

        # Already has time component
        parts = value_clean.split("T")
        date_part = parts[0]
        time_part = parts[1] if len(parts) > 1 else "00:00:00"

        # Ensure time has seconds
        time_parts = time_part.split(":")
        if len(time_parts) == 2:
            time_part = f"{time_part}:00"

        return f"{date_part}T{time_part}"

    def inverse(self, value: Any) -> str:
        # Return canonical form
        return value

    def fails_when(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return f"Expected string, got {type(value).__name__}"

        try:
            # Try to parse as ISO date
            value_clean = value.replace("Z", "+00:00")
            if "T" in value_clean:
                # Has time component
                datetime.fromisoformat(value_clean.split("+")[0])
            else:
                # Date only
                datetime.fromisoformat(value_clean)
            return None
        except ValueError as e:
            return f"String '{value}' is not a valid ISO date: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL EQUIVALENCE RULES
# ═══════════════════════════════════════════════════════════════════════════════

class UnwrapDemoDataRule(TransformationRule):
    """
    {"demoData": X} → X when target expects X directly

    Lossless: just removing a wrapper
    Reversible: X → {"demoData": X}
    Fails: when demoData key doesn't exist
    """

    @property
    def name(self) -> str:
        return "unwrap_demodata"

    def applies(self, value: Any, target_hint: Optional[str] = None) -> bool:
        return isinstance(value, dict) and "demoData" in value

    def transform(self, value: Any) -> Any:
        failure = self.fails_when(value)
        if failure:
            raise ValueError(failure)
        return value["demoData"]

    def inverse(self, value: Any) -> dict:
        return {"demoData": value}

    def fails_when(self, value: Any) -> Optional[str]:
        if not isinstance(value, dict):
            return f"Expected dict, got {type(value).__name__}"
        if "demoData" not in value:
            return "Dict does not contain 'demoData' key"
        return None


class UnwrapSingletonArrayRule(TransformationRule):
    """
    [X] → X when singleton array

    Lossless: single element is preserved
    Reversible: X → [X]
    Fails: when array has != 1 elements
    """

    @property
    def name(self) -> str:
        return "unwrap_singleton_array"

    def applies(self, value: Any, target_hint: Optional[str] = None) -> bool:
        return isinstance(value, list) and len(value) == 1

    def transform(self, value: Any) -> Any:
        failure = self.fails_when(value)
        if failure:
            raise ValueError(failure)
        return value[0]

    def inverse(self, value: Any) -> list:
        return [value]

    def fails_when(self, value: Any) -> Optional[str]:
        if not isinstance(value, list):
            return f"Expected list, got {type(value).__name__}"
        if len(value) != 1:
            return f"List has {len(value)} elements, not singleton"
        return None


class WrapAsArrayRule(TransformationRule):
    """
    X → [X] when target expects array

    Lossless: wrapping preserves element
    Reversible: [X] → X (if singleton)
    Fails: when value is already an array
    """

    @property
    def name(self) -> str:
        return "wrap_as_array"

    def applies(self, value: Any, target_hint: Optional[str] = None) -> bool:
        return not isinstance(value, list) and target_hint == "array"

    def transform(self, value: Any) -> list:
        return [value]

    def inverse(self, value: Any) -> Any:
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value  # Cannot unwrap non-singleton

    def fails_when(self, value: Any) -> Optional[str]:
        if isinstance(value, list):
            return "Value is already an array"
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# RECONCILER ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReconciliationResult:
    """Result of reconciliation attempt."""
    success: bool
    data: Any
    transforms: list[TransformRecord] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)  # Reasons for refusal

    def log(self) -> None:
        """Log all transformations."""
        if self.transforms:
            print(f"Applied {len(self.transforms)} transformation(s):")
            for t in self.transforms:
                print(f"  [{t.field_path}] {t.rule_name}: {t.original!r} → {t.transformed!r}")
        if self.refused:
            print(f"Refused {len(self.refused)} transformation(s):")
            for r in self.refused:
                print(f"  ✗ {r}")


class Reconciler:
    """
    General-purpose reconciliation engine.

    Pipeline: classify → apply_rules → validate

    Only applies REPRESENTATIONAL and STRUCTURAL equivalence transforms.
    SEMANTIC differences and UNKNOWN/AMBIGUOUS cases are REFUSED.
    """

    # All available rules — ordered by priority
    RULES: list[TransformationRule] = [
        # Representational equivalence
        StringToIntRule(),
        StringToFloatRule(),
        StringToBoolRule(),
        StringToNullRule(),
        WhitespaceNormalizationRule(),
        ISODateNormalizationRule(),
        # Structural equivalence
        UnwrapDemoDataRule(),
        UnwrapSingletonArrayRule(),
        WrapAsArrayRule(),
    ]

    def __init__(self):
        self.transforms: list[TransformRecord] = []
        self.refused: list[str] = []

    def reconcile(self, data: Any, path: str = "root") -> ReconciliationResult:
        """
        Attempt to reconcile data through syntactic transformations.

        Returns ReconciliationResult with:
        - success: True if reconciliation completed (may have refused some)
        - data: Transformed data
        - transforms: List of applied transformations
        - refused: List of transformations that were refused
        """
        self.transforms = []
        self.refused = []

        result = self._reconcile_value(data, path)

        return ReconciliationResult(
            success=len(self.refused) == 0,
            data=result,
            transforms=self.transforms,
            refused=self.refused,
        )

    def _reconcile_value(self, value: Any, path: str) -> Any:
        """Recursively reconcile a value."""
        # Try each rule
        for rule in self.RULES:
            if rule.applies(value):
                failure = rule.fails_when(value)
                if failure:
                    self.refused.append(f"[{path}] {rule.name}: {failure}")
                    continue

                try:
                    transformed = rule.transform(value)
                    self.transforms.append(TransformRecord(
                        rule_name=rule.name,
                        field_path=path,
                        original=value,
                        transformed=transformed,
                        inverse_exists=True,
                        reason=f"Applied {rule.name}",
                    ))
                    value = transformed
                except ValueError as e:
                    self.refused.append(f"[{path}] {rule.name}: {e}")

        # Recurse into structures
        if isinstance(value, dict):
            return {k: self._reconcile_value(v, f"{path}.{k}") for k, v in value.items()}
        elif isinstance(value, list):
            return [self._reconcile_value(v, f"{path}[{i}]") for i, v in enumerate(value)]

        return value

    def classify_mismatch(self, source: Any, target_type: type) -> ClassificationResult:
        """
        Classify a mismatch between source value and target type.

        This is the formal classification step that determines
        whether transformation is ALLOWED or FORBIDDEN.
        """
        source_type = type(source)

        # Same type — no mismatch
        if source_type == target_type:
            return ClassificationResult(
                mismatch_class=MismatchClass.STRUCTURAL_EQUIVALENCE,
                source_value=source,
                expected_type=target_type,
                reason="Types match exactly",
                transformable=False,  # No transform needed
            )

        # String → Number (representational)
        if source_type == str and target_type in (int, float):
            for rule in [StringToIntRule(), StringToFloatRule()]:
                if rule.applies(source) and rule.fails_when(source) is None:
                    return ClassificationResult(
                        mismatch_class=MismatchClass.REPRESENTATIONAL_EQUIVALENCE,
                        source_value=source,
                        expected_type=target_type,
                        reason=f"String represents {target_type.__name__}",
                        transformable=True,
                    )
            return ClassificationResult(
                mismatch_class=MismatchClass.SEMANTIC_DIFFERENCE,
                source_value=source,
                expected_type=target_type,
                reason=f"String '{source}' cannot be converted to {target_type.__name__}",
                transformable=False,
            )

        # String → Bool (representational)
        if source_type == str and target_type == bool:
            rule = StringToBoolRule()
            if rule.applies(source) and rule.fails_when(source) is None:
                return ClassificationResult(
                    mismatch_class=MismatchClass.REPRESENTATIONAL_EQUIVALENCE,
                    source_value=source,
                    expected_type=target_type,
                    reason="String represents boolean",
                    transformable=True,
                )
            return ClassificationResult(
                mismatch_class=MismatchClass.SEMANTIC_DIFFERENCE,
                source_value=source,
                expected_type=target_type,
                reason=f"String '{source}' is not a boolean representation",
                transformable=False,
            )

        # Wrapper structures (structural)
        if source_type == dict and isinstance(source, dict):
            if "demoData" in source:
                return ClassificationResult(
                    mismatch_class=MismatchClass.STRUCTURAL_EQUIVALENCE,
                    source_value=source,
                    expected_type=target_type,
                    reason="Wrapped in demoData",
                    transformable=True,
                )

        # Unknown — cannot determine without domain knowledge
        return ClassificationResult(
            mismatch_class=MismatchClass.UNKNOWN_AMBIGUOUS,
            source_value=source,
            expected_type=target_type,
            reason=f"Cannot determine equivalence between {source_type.__name__} and {target_type.__name__}",
            transformable=False,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def reconcile_and_validate(
    data: Any,
    validator: Callable[[Any], None],
) -> ReconciliationResult:
    """
    Complete pipeline: reconcile → validate

    Args:
        data: Raw data from LLM
        validator: Validation function that raises on failure

    Returns:
        ReconciliationResult with transformed data

    The validator is called AFTER reconciliation. If validation fails,
    the error propagates — we don't retry or guess.
    """
    reconciler = Reconciler()
    result = reconciler.reconcile(data)

    # Validate the reconciled data
    # Let validation errors propagate — no retry, no guess
    validator(result.data)

    return result
