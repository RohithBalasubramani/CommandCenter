"""
Widget Data Schemas — per-widget data contracts for Pipeline v2.

Each widget scenario declares:
- required: fields that MUST be present in the data_override.demoData
- optional: fields that improve the widget but aren't mandatory
- rag_strategy: which data collection strategy to use
- default_collections: which ChromaDB collections to search

These schemas drive the data_collector.py — each widget gets exactly the data
it needs, formatted to match the frontend fixtureData.ts demoData shapes.

VALIDATION: All widget data MUST pass validate_widget_data() before reaching
the frontend. Invalid data raises ValidationError - no coercion, no defaults.
"""
import re
from datetime import datetime
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION ERROR
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationError(Exception):
    """Raised when widget data fails validation. No coercion, fail fast."""

    def __init__(self, scenario: str, errors: list[str]):
        self.scenario = scenario
        self.errors = errors
        super().__init__(f"Validation failed for '{scenario}': {'; '.join(errors)}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# SQL injection patterns
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

# XSS patterns
XSS_PATTERNS = [
    r"<script[^>]*>",
    r"</script>",
    r"javascript:",
    r"on\w+\s*=",  # onclick=, onerror=, etc.
    r"<iframe",
    r"<object",
    r"<embed",
    r"<svg[^>]*onload",
]

# Maximum nesting depth (DoS protection)
MAX_NESTING_DEPTH = 10

# Valid units for power/energy
VALID_POWER_UNITS = {"kW", "MW", "GW", "W", "kVA", "MVA", "VA"}
VALID_ENERGY_UNITS = {"kWh", "MWh", "GWh", "Wh", "J", "kJ", "MJ"}
VALID_PERCENTAGE_UNITS = {"%", "percent"}
VALID_TEMPERATURE_UNITS = {"°C", "°F", "K", "C", "F"}


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _check_nesting_depth(obj: Any, current_depth: int = 0) -> int:
    """Check object nesting depth. Returns max depth found."""
    if current_depth > MAX_NESTING_DEPTH:
        return current_depth

    if isinstance(obj, dict):
        if not obj:
            return current_depth
        return max(_check_nesting_depth(v, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return current_depth
        return max(_check_nesting_depth(v, current_depth + 1) for v in obj)
    else:
        return current_depth


def _check_security(value: str, field_name: str, errors: list[str]) -> None:
    """Check string for SQL injection and XSS patterns."""
    if not isinstance(value, str):
        return

    # SQL injection
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            errors.append(f"SQL injection detected in '{field_name}': pattern '{pattern}'")
            return

    # XSS
    for pattern in XSS_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            errors.append(f"XSS detected in '{field_name}': pattern '{pattern}'")
            return


def _check_timestamp(value: str, field_name: str, errors: list[str]) -> None:
    """Validate timestamp is not in the future and is a valid date."""
    if not isinstance(value, str):
        errors.append(f"Field '{field_name}' must be a string timestamp, got {type(value).__name__}")
        return

    # Try to parse ISO format
    try:
        # Handle various ISO formats
        value_clean = value.replace("Z", "+00:00")
        if "T" in value_clean:
            dt = datetime.fromisoformat(value_clean.split("+")[0])
        else:
            dt = datetime.fromisoformat(value_clean)

        # Check for future date (allow 1 day buffer for timezone issues)
        now = datetime.now()
        if dt.year > now.year + 1:
            errors.append(f"Field '{field_name}' has future timestamp: {value} (year {dt.year})")

        # Check for impossible dates (parser usually handles this, but be explicit)
        if dt.month > 12 or dt.day > 31:
            errors.append(f"Field '{field_name}' has invalid date: {value}")

    except ValueError as e:
        errors.append(f"Field '{field_name}' has invalid timestamp format: {value} ({e})")


def _check_numeric_value(value: Any, field_name: str, unit: Optional[str], errors: list[str]) -> None:
    """Validate numeric value is appropriate for its unit."""
    # Allow string numbers
    if isinstance(value, str):
        if value in ("N/A", "—", "-", ""):
            return  # Acceptable placeholder
        try:
            value = float(value)
        except ValueError:
            errors.append(f"Field '{field_name}' must be numeric, got string '{value}'")
            return

    if not isinstance(value, (int, float)):
        errors.append(f"Field '{field_name}' must be numeric, got {type(value).__name__}")
        return

    # Check for NaN/Inf
    if isinstance(value, float):
        if value != value:  # NaN check
            errors.append(f"Field '{field_name}' is NaN")
            return
        if abs(value) == float('inf'):
            errors.append(f"Field '{field_name}' is infinite")
            return

    # Range checks based on unit
    if unit:
        unit_clean = unit.strip()

        # Percentage checks
        if unit_clean in VALID_PERCENTAGE_UNITS:
            if value < 0:
                errors.append(f"Field '{field_name}' has negative percentage: {value}%")
            # Percentages > 200% are likely data corruption or unit mismatch
            # (100-200% can be legitimate: temporary overload, YoY growth, COP)
            elif value > 200:
                errors.append(
                    f"Field '{field_name}' has suspicious percentage: {value}%. "
                    f"Values >200% likely indicate data corruption or unit error."
                )

        # Power checks - must be non-negative (unless explicitly noted)
        if unit_clean in VALID_POWER_UNITS:
            if value < 0:
                errors.append(f"Field '{field_name}' has negative power: {value} {unit_clean}")

        # Energy checks - must be non-negative
        if unit_clean in VALID_ENERGY_UNITS:
            if value < 0:
                errors.append(f"Field '{field_name}' has negative energy: {value} {unit_clean}")


def _validate_string_field(value: Any, field_name: str, errors: list[str]) -> None:
    """Validate a string field."""
    if value is None:
        errors.append(f"Field '{field_name}' is null")
        return

    if not isinstance(value, str):
        errors.append(f"Field '{field_name}' must be string, got {type(value).__name__}")
        return

    _check_security(value, field_name, errors)


def _validate_array_field(value: Any, field_name: str, errors: list[str], min_length: int = 0) -> None:
    """Validate an array field."""
    if value is None:
        errors.append(f"Field '{field_name}' is null")
        return

    if not isinstance(value, list):
        errors.append(f"Field '{field_name}' must be array, got {type(value).__name__}")
        return

    if len(value) < min_length:
        errors.append(f"Field '{field_name}' requires at least {min_length} items, got {len(value)}")


def _validate_object_field(value: Any, field_name: str, errors: list[str]) -> None:
    """Validate an object field."""
    if value is None:
        errors.append(f"Field '{field_name}' is null")
        return

    if not isinstance(value, dict):
        errors.append(f"Field '{field_name}' must be object, got {type(value).__name__}")


def _validate_time_series(time_series: Any, field_name: str, errors: list[str]) -> None:
    """Validate time series array structure."""
    if not isinstance(time_series, list):
        errors.append(f"Field '{field_name}' must be array, got {type(time_series).__name__}")
        return

    for i, point in enumerate(time_series):
        if not isinstance(point, dict):
            errors.append(f"Field '{field_name}[{i}]' must be object, got {type(point).__name__}")
            continue

        if "time" not in point:
            errors.append(f"Field '{field_name}[{i}]' missing required 'time' field")
        else:
            _check_timestamp(point["time"], f"{field_name}[{i}].time", errors)

        if "value" not in point:
            errors.append(f"Field '{field_name}[{i}]' missing required 'value' field")
        elif point["value"] is None:
            errors.append(f"Field '{field_name}[{i}].value' is null")


def _recursive_security_check(obj: Any, path: str, errors: list[str]) -> None:
    """Recursively check all string values for security issues."""
    if isinstance(obj, str):
        _check_security(obj, path, errors)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _recursive_security_check(v, f"{path}.{k}", errors)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _recursive_security_check(v, f"{path}[{i}]", errors)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN VALIDATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_widget_data(scenario: str, data: dict, *, raise_on_error: bool = True) -> dict:
    """
    Validate widget data against schema. STRICT validation.

    Returns:
        dict: {"is_valid": bool, "errors": list[str]}

    Raises:
        ValidationError: If raise_on_error=True (default) and validation fails.

    Checks:
        - Required fields present and non-null
        - Type correctness (no silent coercion)
        - Range validity (no negative percentages, impossible values)
        - Temporal validity (no future timestamps, valid dates)
        - Security (no SQL injection, no XSS)
        - Structure (no excessive nesting - DoS protection)

    Args:
        scenario: Widget scenario name (e.g., "kpi", "trend", "alerts")
        data: Widget data dict (may contain demoData or be direct fields)
        raise_on_error: If True (default), raise ValidationError on failure.
                        If False, return {"is_valid": False, "errors": [...]} instead.
    """
    errors: list[str] = []

    # ── Basic structure checks ──
    if data is None:
        if raise_on_error:
            raise ValidationError(scenario, ["Data is null"])
        return {"is_valid": False, "errors": ["Data is null"]}

    if not isinstance(data, dict):
        err = f"Data must be object, got {type(data).__name__}"
        if raise_on_error:
            raise ValidationError(scenario, [err])
        return {"is_valid": False, "errors": [err]}

    # ── DoS protection: nesting depth ──
    depth = _check_nesting_depth(data)
    if depth > MAX_NESTING_DEPTH:
        err = f"Data nesting depth {depth} exceeds maximum {MAX_NESTING_DEPTH}"
        if raise_on_error:
            raise ValidationError(scenario, [err])
        return {"is_valid": False, "errors": [err]}

    # ── Get schema ──
    schema = WIDGET_SCHEMAS.get(scenario)
    if not schema:
        # Unknown scenario - still do security checks
        _recursive_security_check(data, "data", errors)
        if errors:
            if raise_on_error:
                raise ValidationError(scenario, errors)
            return {"is_valid": False, "errors": errors}
        return {"is_valid": True, "errors": []}

    # ── Determine data location ──
    if scenario == "trends-cumulative":
        # Special case: config and data at top level
        check_dict = data
        required = schema.get("required", [])
    elif "demoData" in data:
        demo_data = data["demoData"]

        # Handle array case (alerts widget)
        if isinstance(demo_data, list):
            # Validate each item in the array
            for i, item in enumerate(demo_data):
                if not isinstance(item, dict):
                    errors.append(f"demoData[{i}] must be object, got {type(item).__name__}")
                    continue

                # Check required fields in each item
                for field in schema.get("required", []):
                    if field not in item:
                        errors.append(f"demoData[{i}] missing required field '{field}'")
                    elif item[field] is None:
                        errors.append(f"demoData[{i}].{field} is null")

                # Security check each item
                _recursive_security_check(item, f"demoData[{i}]", errors)

            if errors:
                if raise_on_error:
                    raise ValidationError(scenario, errors)
                return {"is_valid": False, "errors": errors}
            return {"is_valid": True, "errors": []}

        elif not isinstance(demo_data, dict):
            err = f"demoData must be object or array, got {type(demo_data).__name__}"
            if raise_on_error:
                raise ValidationError(scenario, [err])
            return {"is_valid": False, "errors": [err]}

        check_dict = demo_data
        required = schema.get("required", [])
    else:
        check_dict = data
        required = schema.get("required", [])

    # ── Required field checks ──
    for field in required:
        if field not in check_dict:
            errors.append(f"Missing required field '{field}'")
        elif check_dict[field] is None:
            errors.append(f"Required field '{field}' is null")

    # ── Scenario-specific validation ──
    if scenario == "kpi":
        _validate_kpi(check_dict, errors)
    elif scenario == "comparison":
        _validate_comparison(check_dict, errors)
    elif scenario == "trend":
        _validate_trend(check_dict, errors)
    elif scenario == "trend-multi-line":
        _validate_trend_multi_line(check_dict, errors)
    elif scenario == "trends-cumulative":
        _validate_trends_cumulative(data, errors)
    elif scenario == "distribution":
        _validate_distribution(check_dict, errors)
    elif scenario == "timeline":
        _validate_timeline(check_dict, errors)
    elif scenario == "alerts":
        _validate_alerts(check_dict, errors)

    # ── Global security check ──
    _recursive_security_check(data, "data", errors)

    # ── Return or raise based on errors ──
    if errors:
        if raise_on_error:
            raise ValidationError(scenario, errors)
        return {"is_valid": False, "errors": errors}

    return {"is_valid": True, "errors": []}


def validate_widget_data_safe(scenario: str, data: dict) -> dict:
    """
    Safe wrapper around validate_widget_data that returns a dict instead of raising.

    Returns:
        dict: {"is_valid": bool, "errors": list[str]}
    """
    try:
        validate_widget_data(scenario, data)
        return {"is_valid": True, "errors": []}
    except ValidationError as e:
        return {"is_valid": False, "errors": e.errors}
    except Exception as e:
        return {"is_valid": False, "errors": [str(e)]}


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO-SPECIFIC VALIDATORS
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_kpi(data: dict, errors: list[str]) -> None:
    """Validate KPI widget data."""
    if "label" in data:
        _validate_string_field(data["label"], "label", errors)

    if "value" in data:
        unit = data.get("unit", "")
        _check_numeric_value(data["value"], "value", unit, errors)

    if "unit" in data:
        _validate_string_field(data["unit"], "unit", errors)

    if "state" in data and data["state"] is not None:
        if data["state"] not in ("normal", "warning", "critical"):
            errors.append(f"Field 'state' must be one of: normal, warning, critical. Got '{data['state']}'")


def _validate_comparison(data: dict, errors: list[str]) -> None:
    """Validate comparison widget data."""
    unit = data.get("unit", "")

    if "valueA" in data:
        _check_numeric_value(data["valueA"], "valueA", unit, errors)

    if "valueB" in data:
        _check_numeric_value(data["valueB"], "valueB", unit, errors)

    # Check for unit mixing: if one value is ~1000x different, likely kW/MW mismatch
    if "valueA" in data and "valueB" in data:
        try:
            val_a = float(data["valueA"]) if data["valueA"] not in ("N/A", "—", "-", "") else None
            val_b = float(data["valueB"]) if data["valueB"] not in ("N/A", "—", "-", "") else None

            if val_a is not None and val_b is not None and val_a != 0 and val_b != 0:
                ratio = max(val_a, val_b) / min(val_a, val_b)
                # If ratio is ~1000x, likely kW/MW mismatch
                if 500 <= ratio <= 2000:
                    errors.append(
                        f"Suspicious value ratio ({ratio:.0f}x) between valueA={val_a} and valueB={val_b}. "
                        f"Possible unit mismatch (kW vs MW?)"
                    )
        except (ValueError, TypeError):
            pass  # Non-numeric values already caught above

    for field in ["label", "unit", "labelA", "labelB"]:
        if field in data:
            _validate_string_field(data[field], field, errors)


def _validate_trend(data: dict, errors: list[str]) -> None:
    """Validate trend widget data."""
    if "label" in data:
        _validate_string_field(data["label"], "label", errors)

    if "unit" in data:
        _validate_string_field(data["unit"], "unit", errors)

    if "timeSeries" in data:
        _validate_time_series(data["timeSeries"], "timeSeries", errors)


def _validate_trend_multi_line(data: dict, errors: list[str]) -> None:
    """Validate trend-multi-line widget data."""
    if "label" in data:
        _validate_string_field(data["label"], "label", errors)

    if "series" in data:
        _validate_array_field(data["series"], "series", errors)

        if isinstance(data["series"], list):
            for i, s in enumerate(data["series"]):
                if not isinstance(s, dict):
                    errors.append(f"series[{i}] must be object, got {type(s).__name__}")
                    continue

                if "timeSeries" in s:
                    _validate_time_series(s["timeSeries"], f"series[{i}].timeSeries", errors)


def _validate_trends_cumulative(data: dict, errors: list[str]) -> None:
    """Validate trends-cumulative widget data."""
    if "config" in data:
        _validate_object_field(data["config"], "config", errors)

    if "data" in data:
        _validate_array_field(data["data"], "data", errors)

        if isinstance(data["data"], list):
            for i, point in enumerate(data["data"]):
                if not isinstance(point, dict):
                    errors.append(f"data[{i}] must be object, got {type(point).__name__}")
                    continue

                if "x" in point:
                    _check_timestamp(str(point["x"]), f"data[{i}].x", errors)


def _validate_distribution(data: dict, errors: list[str]) -> None:
    """Validate distribution widget data."""
    if "total" in data:
        _check_numeric_value(data["total"], "total", data.get("unit"), errors)

    if "series" in data:
        _validate_array_field(data["series"], "series", errors)

        if isinstance(data["series"], list):
            # Reject empty series when total > 0 (meaningless distribution)
            total = data.get("total", 0)
            try:
                total_num = float(total) if total not in ("N/A", "—", "-", "") else 0
                if total_num > 0 and len(data["series"]) == 0:
                    errors.append(
                        f"Empty series array with total={total_num}. "
                        f"Distribution must have items to sum to total."
                    )
            except (ValueError, TypeError):
                pass

            for i, item in enumerate(data["series"]):
                if not isinstance(item, dict):
                    errors.append(f"series[{i}] must be object, got {type(item).__name__}")
                    continue

                if "value" in item:
                    _check_numeric_value(item["value"], f"series[{i}].value", data.get("unit"), errors)


def _validate_timeline(data: dict, errors: list[str]) -> None:
    """Validate timeline widget data."""
    if "range" in data:
        r = data["range"]
        if isinstance(r, dict):
            if "start" in r:
                _check_timestamp(str(r["start"]), "range.start", errors)
            if "end" in r:
                _check_timestamp(str(r["end"]), "range.end", errors)

    if "events" in data:
        _validate_array_field(data["events"], "events", errors)


def _validate_alerts(data: dict, errors: list[str]) -> None:
    """Validate alerts widget data (single alert object)."""
    if "severity" in data and data["severity"] is not None:
        valid_severities = ("info", "low", "medium", "warning", "high", "critical")
        if data["severity"] not in valid_severities:
            errors.append(f"Field 'severity' must be one of: {', '.join(valid_severities)}. Got '{data['severity']}'")

    for field in ["id", "title", "message", "source"]:
        if field in data:
            _validate_string_field(data[field], field, errors)

WIDGET_SCHEMAS = {
    "kpi": {
        "description": "Single metric KPI",
        "required": ["label", "value", "unit"],
        "optional": ["state", "period", "max", "context"],
        "rag_strategy": "single_metric",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "label": "Metric Name",
                "value": "42",
                "unit": "kW",
                "state": "normal",  # normal | warning | critical
            }
        },
    },
    "alerts": {
        "description": "Alert notification panel",
        "required": ["id", "title", "message", "severity", "source"],
        # AUDIT FIX: Added missing optional fields used by frontend fixtures
        "optional": ["evidence", "threshold", "actions", "assignee", "timestamp", "state",
                     "category", "triggerCondition", "occurrenceCount"],
        "rag_strategy": "alert_query",
        "default_collections": ["alerts"],
        "demo_shape": {
            "demoData": {
                "id": "ALT-001",
                "title": "Parameter Name",
                "message": "Alert description",
                "severity": "warning",
                "category": "Equipment",
                "source": "Device Name",
                "state": "new",
                "timestamp": "2026-01-31T10:00:00Z",
                "evidence": {
                    "label": "Value",
                    "value": "95",
                    "unit": "%",
                    "trend": "up",
                },
                "threshold": "90%",
                "triggerCondition": "Value > 90%",
                "occurrenceCount": 1,
                "actions": [],
            }
        },
    },
    "comparison": {
        "description": "Side-by-side comparison",
        "required": ["label", "unit", "labelA", "valueA", "labelB", "valueB"],
        "optional": ["delta", "deltaPct"],
        "rag_strategy": "multi_entity_metric",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "label": "Metric Comparison",
                "unit": "%",
                "labelA": "Entity A",
                "valueA": 92,
                "labelB": "Entity B",
                "valueB": 87,
                "delta": 5,
                "deltaPct": 5.7,
            }
        },
    },
    "trend": {
        "description": "Time series line/area chart",
        "required": ["label", "unit", "timeSeries"],
        "optional": ["timeRange", "threshold"],
        "rag_strategy": "time_series",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "label": "Metric Trend",
                "unit": "kW",
                "timeSeries": [{"time": "2026-01-31T00:00:00Z", "value": 42}],
                "timeRange": "last_24h",
            }
        },
    },
    "trend-multi-line": {
        # AUDIT FIX: Enhanced documentation for multi-line trend structure
        # Each series contains a name and its own timeSeries array.
        # All series should share the same time axis for proper alignment.
        "description": "Multi-line time series chart — overlays 2-4 metrics on same time axis",
        "required": ["label", "unit", "series"],
        "optional": ["timeRange", "threshold"],
        "rag_strategy": "multi_time_series",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "label": "Multi-Metric Trend",
                "unit": "kW",
                "timeRange": "last_24h",  # Optional: last_24h, last_7d, last_30d
                "series": [
                    # Each series has: name (legend label), color (optional), timeSeries (data points)
                    {
                        "name": "Metric A",
                        "color": "#2563eb",  # Optional: hex color for the line
                        "timeSeries": [
                            {"time": "2026-01-31T00:00:00Z", "value": 42},
                            {"time": "2026-01-31T01:00:00Z", "value": 45},
                        ]
                    },
                    {
                        "name": "Metric B",
                        "color": "#16a34a",
                        "timeSeries": [
                            {"time": "2026-01-31T00:00:00Z", "value": 38},
                            {"time": "2026-01-31T01:00:00Z", "value": 41},
                        ]
                    },
                ],
            }
        },
    },
    "trends-cumulative": {
        "description": "Stacked area / cumulative chart",
        "required": ["config", "data"],
        "optional": [],
        "rag_strategy": "cumulative_time_series",
        "default_collections": ["equipment"],
        "demo_shape": {
            "config": {
                "title": "Cumulative Trend",
                "subtitle": "",
                "variant": "V1",
                "mode": "cumulative",
                "series": [{"id": "S1", "label": "Value", "unit": "kWh", "color": "#2563eb"}],
            },
            "data": [{"x": "2026-01-31T00:00:00Z", "S1_raw": 0, "S1_cumulative": 0}],
        },
    },
    "distribution": {
        "description": "Proportional breakdown chart",
        "required": ["total", "unit", "series"],
        "optional": [],
        "rag_strategy": "aggregation",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "total": 1000,
                "unit": "kW",
                "series": [
                    {"label": "Category A", "value": 400},
                    {"label": "Category B", "value": 350},
                    {"label": "Category C", "value": 250},
                ],
            }
        },
    },
    "composition": {
        "description": "Stacked bar / grouped composition",
        "required": ["label", "unit", "categories", "series"],
        "optional": [],
        "rag_strategy": "aggregation",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "label": "Composition",
                "unit": "kW",
                "categories": ["Cat A", "Cat B"],
                "series": [
                    {"name": "Group 1", "values": [100, 200]},
                    {"name": "Group 2", "values": [150, 180]},
                ],
            }
        },
    },
    "category-bar": {
        "description": "Bar chart across categories",
        "required": ["label", "unit", "categories", "values"],
        "optional": ["orientation"],
        "rag_strategy": "aggregation",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "label": "Category Comparison",
                "unit": "kW",
                "categories": ["Item 1", "Item 2", "Item 3"],
                "values": [100, 80, 60],
            }
        },
    },
    "timeline": {
        "description": "Horizontal timeline of events",
        "required": ["title", "range", "events"],
        "optional": ["lanes"],
        "rag_strategy": "events_in_range",
        "default_collections": ["maintenance", "shift_logs"],
        "demo_shape": {
            "demoData": {
                "title": "Event Timeline",
                "range": {"start": "2026-01-01", "end": "2026-01-31"},
                "events": [
                    {"time": "2026-01-15", "label": "Event 1", "type": "maintenance"},
                ],
            }
        },
    },
    "flow-sankey": {
        "description": "Sankey flow diagram",
        "required": ["label", "nodes", "links"],
        "optional": ["unit"],
        "rag_strategy": "flow_analysis",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "label": "Energy Flow",
                "unit": "kW",
                "nodes": [{"id": "source", "label": "Grid"}],
                "links": [{"source": "source", "target": "dest", "value": 100}],
            }
        },
    },
    "matrix-heatmap": {
        "description": "Matrix / heatmap visualization",
        "required": ["label", "dataset"],
        "optional": ["unit", "rows", "cols"],
        "rag_strategy": "cross_tabulation",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "label": "Health Matrix",
                "rows": ["Equipment 1", "Equipment 2"],
                "cols": ["Metric A", "Metric B"],
                "dataset": [[0.9, 0.85], [0.7, 0.92]],
            }
        },
    },
    "eventlogstream": {
        "description": "Scrollable event log / log stream",
        "required": ["events"],
        "optional": ["title", "filters"],
        "rag_strategy": "events_in_range",
        "default_collections": ["maintenance", "shift_logs", "work_orders"],
        "demo_shape": {
            "demoData": {
                "title": "Event Log",
                "events": [
                    {"timestamp": "2026-01-31T10:00:00Z", "type": "info", "message": "Event description", "source": "System"},
                ],
            }
        },
    },
    "edgedevicepanel": {
        "description": "Detailed single-device panel",
        "required": ["device"],
        "optional": ["readings", "alerts", "maintenance"],
        "rag_strategy": "single_entity_deep",
        "default_collections": ["equipment", "alerts", "maintenance"],
        "demo_shape": {
            "demoData": {
                "device": {
                    "name": "Transformer 1",
                    "id": "TF-001",
                    "type": "transformer",
                    "status": "running",
                    "health": 94,
                },
                "readings": [],
                "alerts": [],
            }
        },
    },
    "chatstream": {
        "description": "Conversational AI stream",
        "required": ["messages"],
        "optional": [],
        "rag_strategy": "none",
        "default_collections": [],
        "demo_shape": {
            "demoData": {
                "messages": [],
            }
        },
    },
    "helpview": {
        "description": "Help and capabilities display",
        "required": [],
        "optional": [],
        "rag_strategy": "none",
        "default_collections": [],
        "demo_shape": {"demoData": {}},
    },
    "peopleview": {
        "description": "Workforce overview",
        "required": ["roster"],
        "optional": ["shifts", "attendance"],
        "rag_strategy": "people_query",
        "default_collections": ["shift_logs"],
        "demo_shape": {
            "demoData": {
                "roster": [],
                "shifts": [],
            }
        },
    },
    "peoplehexgrid": {
        "description": "Personnel hex grid",
        "required": ["people"],
        "optional": ["zones"],
        "rag_strategy": "people_query",
        "default_collections": ["shift_logs"],
        "demo_shape": {"demoData": {"people": [], "zones": []}},
    },
    "peoplenetwork": {
        "description": "People network graph",
        "required": ["nodes", "edges"],
        "optional": [],
        "rag_strategy": "people_query",
        "default_collections": ["shift_logs"],
        "demo_shape": {"demoData": {"nodes": [], "edges": []}},
    },
    "supplychainglobe": {
        "description": "Supply chain globe",
        "required": ["locations", "routes"],
        "optional": [],
        "rag_strategy": "supply_query",
        "default_collections": ["work_orders"],
        "demo_shape": {"demoData": {"locations": [], "routes": []}},
    },
    "pulseview": {
        "description": "Real-time pulse view",
        "required": [],
        "optional": ["signals"],
        "rag_strategy": "none",
        "default_collections": [],
        "demo_shape": {"demoData": {}},
    },
}
