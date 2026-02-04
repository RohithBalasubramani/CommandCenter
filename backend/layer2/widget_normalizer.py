"""
Widget Data Normalizer — Dimension-driven lossless transformations.

Pipeline: LLM → normalize_widget_data() → validate_widget_data() → render

DESIGN PRINCIPLE: This module is CONFIGURATION-DRIVEN, not example-driven.
All normalization rules flow from dimensions.py, NOT from hardcoded examples.

RULES:
- Only perform LOSSLESS, UNAMBIGUOUS transformations
- If intent is ambiguous → FAIL (raise NormalizationError)
- No guessing, no inventing, no silent fixes
- All transformations are logged with provenance
- Validation remains strict and unchanged

ALLOWED:
- Unit normalization within same DIMENSION (kW → MW when unambiguous)
- Canonical numeric formatting (string "42" → int 42)
- Safe structural normalization (flatten known patterns)

FORBIDDEN:
- Guessing missing units
- Inventing data
- Fixing logically invalid data
- Converting across DIMENSIONS (power → energy)
- Ignoring semantic mismatch (capacity vs usage)
"""
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from layer2.dimensions import (
    Dimension,
    Semantic,
    DIMENSION_REGISTRY,
    infer_dimension,
    normalize_to_base,
    convert_value,
    ConversionError,
    get_base_unit,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION ERROR
# ═══════════════════════════════════════════════════════════════════════════════

class NormalizationError(Exception):
    """Raised when normalization cannot proceed due to ambiguity."""

    def __init__(self, scenario: str, reason: str):
        self.scenario = scenario
        self.reason = reason
        super().__init__(f"Normalization failed for '{scenario}': {reason}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMATION RECORD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Transformation:
    """Record of a single normalization transformation."""
    field: str
    action: str
    original: Any
    normalized: Any
    reason: str
    dimension: Optional[str] = None  # Dimension name if unit conversion


@dataclass
class NormalizationResult:
    """Result of normalization with provenance tracking."""
    data: dict
    transformations: list[Transformation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def log_transformations(self, scenario: str) -> None:
        """Log all transformations for audit trail."""
        if self.transformations:
            logger.info(f"Normalized {scenario}: {len(self.transformations)} transformation(s)")
            for t in self.transformations:
                dim_info = f" [{t.dimension}]" if t.dimension else ""
                logger.info(f"  [{t.field}]{dim_info} {t.action}: {t.original!r} → {t.normalized!r} ({t.reason})")
        if self.warnings:
            for w in self.warnings:
                logger.warning(f"  ⚠ {w}")


# ═══════════════════════════════════════════════════════════════════════════════
# CORE NORMALIZATION FUNCTIONS — DIMENSION-DRIVEN
# ═══════════════════════════════════════════════════════════════════════════════

# Placeholder values that should NOT be converted
PLACEHOLDERS = frozenset({"N/A", "—", "-", "", "null", "None", "--"})


def _normalize_numeric(value: Any, field_name: str, result: NormalizationResult) -> Any:
    """
    Normalize numeric values to canonical form.

    - String numbers → float/int
    - Preserves N/A placeholders
    - Does NOT guess or invent
    """
    if value is None:
        return None

    # Preserve placeholder strings
    if isinstance(value, str):
        if value in PLACEHOLDERS:
            return value

        # Try to convert numeric strings
        try:
            value_clean = value.strip()
            # Prefer int if it's a whole number
            if "." not in value_clean and "e" not in value_clean.lower():
                normalized = int(value_clean)
            else:
                normalized = float(value_clean)

            result.transformations.append(Transformation(
                field=field_name,
                action="string_to_number",
                original=value,
                normalized=normalized,
                reason="Lossless string-to-number conversion"
            ))
            return normalized
        except ValueError:
            # Not a number - leave as is for validation to catch
            return value

    return value


def _normalize_unit_value(
    value: Any,
    unit: str,
    target_unit: Optional[str],
    field_name: str,
    result: NormalizationResult
) -> tuple[Any, str]:
    """
    Normalize a value and unit using the dimension registry.

    This is the KEY function that uses dimensions.py for ALL unit logic.
    No hardcoded unit tables here.

    Returns (normalized_value, normalized_unit).
    Raises NormalizationError if conversion is ambiguous.
    """
    if value is None or not unit:
        return value, unit

    # Normalize the numeric value first
    value = _normalize_numeric(value, field_name, result)

    # If value is a placeholder, skip unit conversion
    if isinstance(value, str) and value in PLACEHOLDERS:
        return value, unit

    # Skip non-numeric values
    if not isinstance(value, (int, float)):
        return value, unit

    unit_clean = unit.strip()

    # Use dimension registry to infer dimension
    dim_info = infer_dimension(unit_clean)

    if dim_info is None:
        # Unknown unit - cannot normalize, pass through
        # This is NOT an error - validation will handle if needed
        return value, unit

    dimension, unit_spec = dim_info
    base_unit = get_base_unit(dimension)

    # Determine target unit
    if target_unit:
        # Explicit target specified - convert to that
        final_unit = target_unit
    else:
        # Default to base unit of the dimension
        final_unit = base_unit

    # Skip if already in target unit
    if unit_clean == final_unit:
        return value, unit

    # Perform conversion using dimension registry
    try:
        new_value = convert_value(value, unit_clean, final_unit)

        result.transformations.append(Transformation(
            field=field_name,
            action="unit_normalization",
            original=f"{value} {unit_clean}",
            normalized=f"{new_value} {final_unit}",
            reason=f"Normalized to {dimension.name} base unit",
            dimension=dimension.name
        ))

        return new_value, final_unit

    except ConversionError as e:
        # Conversion failed - this is an error
        raise NormalizationError(field_name, str(e))


def _check_dimension_consistency(
    values_with_units: list[tuple[str, Any, str]],
    scenario: str
) -> None:
    """
    Check that all values use consistent dimensions.
    Raises NormalizationError if dimensions are mixed.

    Uses dimension registry to detect mismatches.
    """
    dimensions_found: dict[Dimension, list[tuple[str, str]]] = {}

    for field_name, value, unit in values_with_units:
        if not unit:
            continue

        dim_info = infer_dimension(unit.strip())
        if dim_info is None:
            continue

        dimension, _ = dim_info
        if dimension not in dimensions_found:
            dimensions_found[dimension] = []
        dimensions_found[dimension].append((field_name, unit.strip()))

    # Check for dimension conflicts
    # It's OK to have values in different dimensions (e.g., power and temperature)
    # But within a comparison, we should NOT mix incompatible units

    # Check for suspicious combinations
    if Dimension.POWER in dimensions_found and Dimension.ENERGY in dimensions_found:
        power_fields = dimensions_found[Dimension.POWER]
        energy_fields = dimensions_found[Dimension.ENERGY]
        raise NormalizationError(
            scenario,
            f"Mixed POWER and ENERGY dimensions. "
            f"Power: {power_fields}, Energy: {energy_fields}. "
            f"These are different physical quantities and cannot be compared."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO NORMALIZERS — STRUCTURE ONLY, NO HARDCODED UNITS
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_kpi(data: dict, result: NormalizationResult) -> dict:
    """Normalize KPI widget data."""
    if "value" in data:
        unit = data.get("unit", "")
        data["value"], new_unit = _normalize_unit_value(
            data["value"], unit, None, "value", result
        )
        if new_unit and new_unit != unit:
            data["unit"] = new_unit
    return data


def _normalize_comparison(data: dict, result: NormalizationResult, scenario: str) -> dict:
    """
    Normalize comparison widget data.

    KEY: Convert both values to the same base unit for fair comparison.
    """
    unit = data.get("unit", "")

    # Check dimension consistency BEFORE normalizing
    values_with_units = []
    if "valueA" in data:
        values_with_units.append(("valueA", data["valueA"], unit))
    if "valueB" in data:
        values_with_units.append(("valueB", data["valueB"], unit))

    _check_dimension_consistency(values_with_units, scenario)

    # Normalize each value to base unit
    if "valueA" in data:
        data["valueA"], _ = _normalize_unit_value(
            data["valueA"], unit, None, "valueA", result
        )

    if "valueB" in data:
        data["valueB"], _ = _normalize_unit_value(
            data["valueB"], unit, None, "valueB", result
        )

    # Update unit to base unit if we normalized
    if unit:
        dim_info = infer_dimension(unit)
        if dim_info:
            data["unit"] = get_base_unit(dim_info[0])

    return data


def _normalize_trend(data: dict, result: NormalizationResult) -> dict:
    """Normalize trend widget data."""
    unit = data.get("unit", "")
    target_unit = None

    # Determine target unit from dimension
    if unit:
        dim_info = infer_dimension(unit)
        if dim_info:
            target_unit = get_base_unit(dim_info[0])

    if "timeSeries" in data and isinstance(data["timeSeries"], list):
        for i, point in enumerate(data["timeSeries"]):
            if isinstance(point, dict) and "value" in point:
                point["value"], _ = _normalize_unit_value(
                    point["value"], unit, target_unit, f"timeSeries[{i}].value", result
                )

    # Update unit if normalized
    if target_unit and target_unit != unit:
        data["unit"] = target_unit

    return data


def _normalize_distribution(data: dict, result: NormalizationResult) -> dict:
    """Normalize distribution widget data."""
    unit = data.get("unit", "")
    target_unit = None

    # Determine target unit from dimension
    if unit:
        dim_info = infer_dimension(unit)
        if dim_info:
            target_unit = get_base_unit(dim_info[0])

    if "total" in data:
        data["total"], new_unit = _normalize_unit_value(
            data["total"], unit, target_unit, "total", result
        )
        if new_unit:
            data["unit"] = new_unit

    if "series" in data and isinstance(data["series"], list):
        for i, item in enumerate(data["series"]):
            if isinstance(item, dict) and "value" in item:
                item["value"], _ = _normalize_unit_value(
                    item["value"], data.get("unit", unit), target_unit, f"series[{i}].value", result
                )

    return data


def _normalize_generic(data: dict, result: NormalizationResult) -> dict:
    """
    Generic normalization for unknown widget types.

    Only performs safe transformations:
    - String to number conversion
    - Unit normalization if unit field exists
    """
    # Look for common patterns
    if "value" in data and "unit" in data:
        data["value"], new_unit = _normalize_unit_value(
            data["value"], data.get("unit", ""), None, "value", result
        )
        if new_unit:
            data["unit"] = new_unit

    # Normalize any numeric strings
    for key, value in list(data.items()):
        if isinstance(value, str) and key not in ("label", "title", "name", "unit", "id"):
            converted = _normalize_numeric(value, key, result)
            if converted != value:
                data[key] = converted

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN NORMALIZATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_widget_data(scenario: str, data: dict) -> NormalizationResult:
    """
    Normalize widget data before validation.

    Uses the DIMENSION REGISTRY for all unit normalization.
    No hardcoded unit tables - extends automatically when new dimensions
    are added to dimensions.py.

    Args:
        scenario: Widget scenario name
        data: Raw widget data from LLM/data collector

    Returns:
        NormalizationResult with normalized data and transformation log

    Raises:
        NormalizationError: If normalization cannot proceed (ambiguous/contradictory)
    """
    if data is None:
        raise NormalizationError(scenario, "Data is null - cannot normalize")

    if not isinstance(data, dict):
        raise NormalizationError(scenario, f"Data must be object, got {type(data).__name__}")

    # Deep copy to avoid mutating original
    normalized = copy.deepcopy(data)
    result = NormalizationResult(data=normalized)

    # Determine where the actual data is
    if scenario == "trends-cumulative":
        check_dict = normalized
    elif "demoData" in normalized:
        demo_data = normalized["demoData"]
        if isinstance(demo_data, list):
            # Array of items (e.g., alerts)
            for i, item in enumerate(demo_data):
                if isinstance(item, dict):
                    # Normalize each item
                    for key, value in list(item.items()):
                        if isinstance(value, str) and key not in ("label", "title", "name", "unit", "id"):
                            try:
                                converted = _normalize_numeric(value, f"demoData[{i}].{key}", result)
                                if converted != value:
                                    item[key] = converted
                            except Exception:
                                pass
            result.data = normalized
            result.log_transformations(scenario)
            return result
        elif isinstance(demo_data, dict):
            check_dict = demo_data
        else:
            raise NormalizationError(scenario, f"demoData has unexpected type: {type(demo_data).__name__}")
    else:
        check_dict = normalized

    # Apply scenario-specific normalization
    scenario_normalizers = {
        "kpi": _normalize_kpi,
        "comparison": lambda d, r: _normalize_comparison(d, r, scenario),
        "trend": _normalize_trend,
        "trend-multi-line": _normalize_trend,
        "distribution": _normalize_distribution,
    }

    normalizer = scenario_normalizers.get(scenario)
    if normalizer:
        check_dict = normalizer(check_dict, result)
    else:
        # Generic normalization for unknown scenarios
        check_dict = _normalize_generic(check_dict, result)

    # Update the result data
    if "demoData" in normalized and isinstance(normalized["demoData"], dict):
        normalized["demoData"] = check_dict
    elif scenario != "trends-cumulative":
        result.data = check_dict

    # Log transformations
    result.log_transformations(scenario)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_and_validate(scenario: str, data: dict) -> dict:
    """
    Complete pipeline: normalize → validate → return.

    This is the recommended entry point for the data pipeline.

    Args:
        scenario: Widget scenario name
        data: Raw widget data

    Returns:
        Normalized and validated data

    Raises:
        NormalizationError: If data is ambiguous
        ValidationError: If data is invalid
    """
    from layer2.widget_schemas import validate_widget_data

    # Step 1: Normalize using dimension registry
    result = normalize_widget_data(scenario, data)

    # Step 2: Validate (unchanged, strict)
    validate_widget_data(scenario, result.data)

    # Step 3: Return normalized data
    return result.data
