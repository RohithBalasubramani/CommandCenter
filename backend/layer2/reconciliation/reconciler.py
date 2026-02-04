"""
Reconciler — Mismatch Classification.

classify_mismatch(data: dict, schema: Schema) -> MismatchReport

Classifies each field into:
- STRUCTURAL_EQUIVALENCE: wrapper differences, key ordering
- REPRESENTATIONAL_EQUIVALENCE: type representations (string "42" vs int 42)
- UNKNOWN_AMBIGUOUS: cannot determine without context
- SEMANTIC_DIFFERENCE: different meaning, cannot reconcile
- SECURITY_VIOLATION: injection/XSS detected
"""
import re
from typing import Any, Optional

from layer2.reconciliation.types import (
    MismatchClass,
    MismatchReport,
    FieldMismatch,
    WidgetSchema,
    FieldSchema,
)
from layer2.reconciliation.errors import SecurityViolation


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY PATTERNS (checked first, before any transformation)
# ═══════════════════════════════════════════════════════════════════════════════

SQL_INJECTION_PATTERNS = [
    r";\s*DROP\s+",
    r";\s*DELETE\s+",
    r";\s*INSERT\s+",
    r";\s*UPDATE\s+",
    r";\s*SELECT\s+",
    r"--\s*$",
    r"'\s*OR\s+'1'\s*=\s*'1",
    r"'\s*OR\s+1\s*=\s*1",
    r"UNION\s+SELECT",
]

XSS_PATTERNS = [
    r"<script[^>]*>",
    r"</script>",
    r"javascript:",
    r"on\w+\s*=",
    r"<iframe",
    r"<object",
    r"<embed",
    r"<svg[^>]*onload",
]

MAX_NESTING_DEPTH = 10


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def check_security(value: str, field_path: str) -> Optional[str]:
    """
    Check a string value for security violations.

    Returns violation description or None if clean.
    """
    if not isinstance(value, str):
        return None

    # SQL injection
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            return f"SQL injection pattern '{pattern}' in {field_path}"

    # XSS
    for pattern in XSS_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            return f"XSS pattern '{pattern}' in {field_path}"

    return None


def check_nesting_depth(obj: Any, current_depth: int = 0) -> int:
    """Check object nesting depth for DoS protection."""
    if current_depth > MAX_NESTING_DEPTH:
        return current_depth

    if isinstance(obj, dict):
        if not obj:
            return current_depth
        return max(check_nesting_depth(v, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return current_depth
        return max(check_nesting_depth(v, current_depth + 1) for v in obj)
    else:
        return current_depth


def recursive_security_check(obj: Any, path: str, violations: list[str]) -> None:
    """Recursively check all string values for security issues."""
    if isinstance(obj, str):
        violation = check_security(obj, path)
        if violation:
            violations.append(violation)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            recursive_security_check(v, f"{path}.{k}", violations)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            recursive_security_check(v, f"{path}[{i}]", violations)


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

# Values that represent null/missing
NULL_REPRESENTATIONS = frozenset({"null", "Null", "NULL", "None", "none", "", "N/A", "—", "-"})

# Boolean representations
TRUE_VALUES = frozenset({"true", "True", "TRUE", "1", "yes", "Yes", "YES"})
FALSE_VALUES = frozenset({"false", "False", "FALSE", "0", "no", "No", "NO"})


def classify_value_type(value: Any, expected_type: str) -> tuple[MismatchClass, str, bool]:
    """
    Classify the mismatch between a value and expected type.

    Returns: (MismatchClass, reason, is_rewritable)
    """
    actual_type = type(value).__name__

    # Same type - no mismatch
    if expected_type == "string" and isinstance(value, str):
        return MismatchClass.NONE, "Type matches", False
    if expected_type == "number" and isinstance(value, (int, float)):
        return MismatchClass.NONE, "Type matches", False
    if expected_type == "boolean" and isinstance(value, bool):
        return MismatchClass.NONE, "Type matches", False
    if expected_type == "array" and isinstance(value, list):
        return MismatchClass.NONE, "Type matches", False
    if expected_type == "object" and isinstance(value, dict):
        return MismatchClass.NONE, "Type matches", False

    # String → Number (representational equivalence)
    if expected_type == "number" and isinstance(value, str):
        # Check if it's a valid numeric string
        try:
            if "." in value or "e" in value.lower():
                float(value)
            else:
                int(value)
            return (
                MismatchClass.REPRESENTATIONAL_EQUIVALENCE,
                f"String '{value}' represents number",
                True,
            )
        except ValueError:
            # Check for unit suffix (e.g., "500 kW")
            match = re.match(r"^([\d.]+)\s*([a-zA-Z°/³]+)$", value.strip())
            if match:
                return (
                    MismatchClass.UNKNOWN_AMBIGUOUS,
                    f"String '{value}' contains number with unit - requires resolution",
                    False,
                )
            return (
                MismatchClass.SEMANTIC_DIFFERENCE,
                f"String '{value}' is not a valid number",
                False,
            )

    # String → Boolean (representational equivalence)
    if expected_type == "boolean" and isinstance(value, str):
        if value in TRUE_VALUES or value in FALSE_VALUES:
            return (
                MismatchClass.REPRESENTATIONAL_EQUIVALENCE,
                f"String '{value}' represents boolean",
                True,
            )
        return (
            MismatchClass.SEMANTIC_DIFFERENCE,
            f"String '{value}' is not a valid boolean",
            False,
        )

    # Null representations
    if isinstance(value, str) and value in NULL_REPRESENTATIONS:
        return (
            MismatchClass.REPRESENTATIONAL_EQUIVALENCE,
            f"String '{value}' represents null",
            True,
        )

    # Number → String (representational equivalence, reversible)
    if expected_type == "string" and isinstance(value, (int, float)):
        return (
            MismatchClass.REPRESENTATIONAL_EQUIVALENCE,
            f"Number {value} can be represented as string",
            True,
        )

    # Object with demoData wrapper (structural equivalence)
    if expected_type == "object" and isinstance(value, dict):
        if "demoData" in value and len(value) == 1:
            return (
                MismatchClass.STRUCTURAL_EQUIVALENCE,
                "Wrapped in demoData container",
                True,
            )

    # Unknown type mismatch
    return (
        MismatchClass.UNKNOWN_AMBIGUOUS,
        f"Cannot determine equivalence between {actual_type} and {expected_type}",
        False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLASSIFICATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_mismatch(data: dict, schema: WidgetSchema) -> MismatchReport:
    """
    Classify all mismatches between data and schema.

    This is the FIRST step in the pipeline. It:
    1. Checks for security violations (BEFORE any transformation)
    2. Identifies structural mismatches (wrapper patterns)
    3. Identifies representational mismatches (type coercion)
    4. Identifies semantic differences (cannot reconcile)
    5. Identifies unknown/ambiguous cases (need resolution)

    Args:
        data: Raw data from LLM
        schema: Widget schema with field definitions

    Returns:
        MismatchReport with classification for each field

    Raises:
        SecurityViolation: If injection/XSS detected (immediate rejection)
    """
    # ── Security checks FIRST ──
    security_violations: list[str] = []

    # Check nesting depth (DoS protection)
    depth = check_nesting_depth(data)
    if depth > MAX_NESTING_DEPTH:
        raise SecurityViolation(
            f"Excessive nesting depth {depth} > {MAX_NESTING_DEPTH}",
            violation_type="dos_nesting",
            field_path="root",
        )

    # Check all strings for injection/XSS
    recursive_security_check(data, "data", security_violations)

    if security_violations:
        raise SecurityViolation(
            f"Security violations detected: {security_violations[0]}",
            violation_type="xss" if "XSS" in security_violations[0] else "sql_injection",
            pattern_matched=security_violations[0],
        )

    # ── Structural check: demoData wrapper ──
    check_dict = data
    structural_unwrap = False

    if "demoData" in data:
        demo_data = data["demoData"]
        if isinstance(demo_data, dict):
            check_dict = demo_data
            structural_unwrap = True
        elif isinstance(demo_data, list):
            # Array case - validate each item separately
            check_dict = data  # Keep as-is for now

    # ── Field-by-field classification ──
    field_mismatches: list[FieldMismatch] = []
    missing_required: list[str] = []
    rewritable_count = 0
    requires_resolution = False
    requires_escalation = False

    for field_schema in schema.fields:
        field_name = field_schema.name
        expected_type = field_schema.type

        if field_name not in check_dict:
            if field_schema.required:
                missing_required.append(field_name)
            continue

        value = check_dict[field_name]

        # Classify the value
        mismatch_class, reason, rewritable = classify_value_type(value, expected_type)

        if mismatch_class == MismatchClass.NONE:
            continue  # No mismatch

        fm = FieldMismatch(
            field_path=field_name,
            mismatch_class=mismatch_class,
            expected_type=expected_type,
            actual_type=type(value).__name__,
            actual_value=value,
            reason=reason,
            rewritable=rewritable,
        )
        field_mismatches.append(fm)

        if rewritable:
            rewritable_count += 1

        if mismatch_class == MismatchClass.UNKNOWN_AMBIGUOUS:
            requires_resolution = True

        if mismatch_class == MismatchClass.SEMANTIC_DIFFERENCE:
            requires_escalation = True

    # ── Add structural mismatch if wrapper detected ──
    if structural_unwrap:
        field_mismatches.insert(0, FieldMismatch(
            field_path="root",
            mismatch_class=MismatchClass.STRUCTURAL_EQUIVALENCE,
            expected_type="object",
            actual_type="wrapped_object",
            actual_value=None,
            reason="Data wrapped in demoData container",
            rewritable=True,
        ))
        rewritable_count += 1

    # ── Determine overall class ──
    if security_violations:
        overall_class = MismatchClass.SECURITY_VIOLATION
    elif requires_escalation:
        overall_class = MismatchClass.SEMANTIC_DIFFERENCE
    elif requires_resolution:
        overall_class = MismatchClass.UNKNOWN_AMBIGUOUS
    elif field_mismatches:
        # All mismatches are rewritable
        if all(fm.rewritable for fm in field_mismatches):
            overall_class = field_mismatches[0].mismatch_class
        else:
            overall_class = MismatchClass.UNKNOWN_AMBIGUOUS
            requires_resolution = True
    else:
        overall_class = MismatchClass.NONE

    return MismatchReport(
        scenario=schema.scenario,
        overall_class=overall_class,
        field_mismatches=field_mismatches,
        missing_required=missing_required,
        security_violations=security_violations,
        rewritable_count=rewritable_count,
        requires_resolution=requires_resolution,
        requires_escalation=requires_escalation,
    )


def build_schema_from_widget(scenario: str) -> WidgetSchema:
    """
    Build a WidgetSchema from the existing WIDGET_SCHEMAS registry.

    This bridges the reconciliation system with the existing validation system.
    """
    from layer2.widget_schemas import WIDGET_SCHEMAS

    schema_def = WIDGET_SCHEMAS.get(scenario, {})
    required = schema_def.get("required", [])
    optional = schema_def.get("optional", [])

    # Infer types from field names (heuristic)
    def infer_type(field_name: str) -> str:
        if field_name in ("value", "valueA", "valueB", "total", "count"):
            return "number"
        if field_name in ("enabled", "active", "visible"):
            return "boolean"
        if field_name in ("series", "timeSeries", "events", "data"):
            return "array"
        if field_name in ("config", "range", "nested"):
            return "object"
        return "string"

    # Infer unit dimension from field names
    def infer_dimension(field_name: str) -> Optional[str]:
        if "power" in field_name.lower():
            return "power"
        if "energy" in field_name.lower():
            return "energy"
        if "temp" in field_name.lower():
            return "temperature"
        return None

    fields = []
    for field_name in required:
        fields.append(FieldSchema(
            name=field_name,
            type=infer_type(field_name),
            required=True,
            unit_dimension=infer_dimension(field_name),
        ))

    for field_name in optional:
        fields.append(FieldSchema(
            name=field_name,
            type=infer_type(field_name),
            required=False,
            unit_dimension=infer_dimension(field_name),
        ))

    return WidgetSchema(
        scenario=scenario,
        fields=fields,
        required_fields=required,
    )
