"""
Schema-Driven Data Collector for Pipeline v2.

For each widget in the WidgetPlan, executes a targeted RAG query and
transforms results into the exact data shape the widget expects (as
defined in widget_schemas.py).

Replaces the flat "collect everything then format" approach in orchestrator.py
with "select widgets → build targeted queries → collect in parallel → format to schema".
"""

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

from layer2.rag_pipeline import (
    get_rag_pipeline,
    IndustrialRAGPipeline,
    RAGSearchResult,
    EQUIPMENT_COLLECTION,
    ALERTS_COLLECTION,
    MAINTENANCE_COLLECTION,
    OPERATIONAL_DOCS_COLLECTION,
    SHIFT_LOGS_COLLECTION,
    WORK_ORDERS_COLLECTION,
)
from layer2.widget_schemas import WIDGET_SCHEMAS, validate_widget_data, ValidationError

# ── PostgreSQL Timeseries Entity Resolution ──
# Maps natural language equipment names to PostgreSQL table prefixes in command_center_data.
# Each entry: keyword → (table_prefix, primary_metric_column, unit)
ENTITY_TABLE_PREFIX_MAP = {
    "transformer": ("trf", "active_power_kw", "kW"),
    "trafo": ("trf", "active_power_kw", "kW"),
    "generator": ("dg", "active_power_kw", "kW"),
    "genset": ("dg", "active_power_kw", "kW"),
    "dg": ("dg", "active_power_kw", "kW"),
    "diesel": ("dg", "active_power_kw", "kW"),
    "ups": ("ups", "output_power_kw", "kW"),
    "chiller": ("chiller", "power_consumption_kw", "kW"),
    "ahu": ("ahu", "fan_motor_power_kw", "kW"),
    "air handling": ("ahu", "fan_motor_power_kw", "kW"),
    "cooling tower": ("ct", "fan_motor_power_kw", "kW"),
    "pump": ("pump", "motor_power_kw", "kW"),
    "compressor": ("compressor", "motor_power_kw", "kW"),
    "motor": ("motor", "active_power_kw", "kW"),
    "energy meter": ("em", "active_power_total_kw", "kW"),
    "meter": ("em", "active_power_total_kw", "kW"),
    "mcc": ("lt_mcc", "active_power_total_kw", "kW"),
    "motor control": ("lt_mcc", "active_power_total_kw", "kW"),
    "pcc": ("lt_pcc", "active_power_total_kw", "kW"),
    "power control": ("lt_pcc", "active_power_total_kw", "kW"),
    "apfc": ("lt_apfc", "active_power_total_kw", "kW"),
    "distribution board": ("lt_db", "active_power_total_kw", "kW"),
    "vfd": ("lt_vfd", "active_power_kw", "kW"),
    "variable frequency": ("lt_vfd", "active_power_kw", "kW"),
    "plc": ("lt_plc", "active_power_total_kw", "kW"),
    "ats": ("lt_ats", "active_power_total_kw", "kW"),
    "auto transfer": ("lt_ats", "active_power_total_kw", "kW"),
    "changeover": ("lt_changeover", "active_power_total_kw", "kW"),
    "mldb": ("lt_mldb", "active_power_total_kw", "kW"),
    "smdb": ("lt_smdb", "active_power_total_kw", "kW"),
}

# Metric aliases for each equipment type prefix → {metric_alias: (column_name, unit)}
EQUIPMENT_METRIC_MAP = {
    "trf": {
        "power": ("active_power_kw", "kW"), "load": ("load_percent", "%"),
        "power_factor": ("power_factor", "PF"), "pf": ("power_factor", "PF"),
        "voltage": ("secondary_voltage_r", "V"), "oil_temp": ("oil_temperature_top_c", "°C"),
        "winding_temp": ("winding_temperature_hv_c", "°C"), "temperature": ("oil_temperature_top_c", "°C"),
        "frequency": ("frequency_hz", "Hz"), "default": ("active_power_kw", "kW"),
    },
    "dg": {
        "power": ("active_power_kw", "kW"), "load": ("load_percent", "%"),
        "voltage": ("output_voltage_r", "V"), "frequency": ("frequency_hz", "Hz"),
        "coolant": ("coolant_temperature_c", "°C"), "temperature": ("coolant_temperature_c", "°C"),
        "fuel": ("fuel_level_pct", "%"), "rpm": ("engine_rpm", "RPM"),
        "default": ("active_power_kw", "kW"),
    },
    "ups": {
        "power": ("output_power_kw", "kW"), "load": ("load_percent", "%"),
        "voltage": ("output_voltage_r", "V"), "battery": ("battery_charge_pct", "%"),
        "battery_status": ("battery_charge_pct", "%"), "battery_health": ("battery_health_pct", "%"),
        "battery_voltage": ("battery_voltage_v", "V"), "runtime": ("battery_time_remaining_min", "min"),
        "battery_temperature": ("battery_temperature_c", "°C"),
        "temperature": ("battery_temperature_c", "°C"), "default": ("output_power_kw", "kW"),
    },
    "chiller": {
        "power": ("power_consumption_kw", "kW"), "load": ("load_percent", "%"),
        "cop": ("current_cop", "COP"), "capacity": ("cooling_capacity_kw", "kW"),
        "temperature": ("chw_supply_temp_c", "°C"), "flow": ("chw_flow_rate_m3h", "m³/h"),
        "default": ("power_consumption_kw", "kW"),
    },
    "ahu": {
        "power": ("fan_motor_power_kw", "kW"), "temperature": ("supply_air_temp_c", "°C"),
        "flow": ("supply_air_flow_cfm", "CFM"), "humidity": ("supply_air_humidity_pct", "%"),
        "default": ("fan_motor_power_kw", "kW"),
    },
    "ct": {
        "power": ("fan_motor_power_kw", "kW"), "temperature": ("inlet_water_temp_c", "°C"),
        "outlet_temp": ("outlet_water_temp_c", "°C"), "flow": ("water_flow_rate_m3h", "m³/h"),
        "vibration": ("fan_vibration_mm_s", "mm/s"),
        "default": ("fan_motor_power_kw", "kW"),
    },
    "pump": {
        "power": ("motor_power_kw", "kW"), "flow": ("flow_rate_m3h", "m³/h"),
        "pressure": ("discharge_pressure_bar", "bar"), "vibration": ("vibration_de_mm_s", "mm/s"),
        "temperature": ("fluid_temperature_c", "°C"),
        "default": ("motor_power_kw", "kW"),
    },
    "compressor": {
        "power": ("motor_power_kw", "kW"), "pressure": ("discharge_pressure_bar", "bar"),
        "temperature": ("discharge_temperature_c", "°C"), "vibration": ("vibration_mm_s", "mm/s"),
        "load": ("load_percent", "%"), "default": ("motor_power_kw", "kW"),
    },
    "motor": {
        "power": ("active_power_kw", "kW"), "load": ("load_percent", "%"),
        "temperature": ("winding_temp_r_c", "°C"), "vibration": ("vibration_de_h_mm_s", "mm/s"),
        "speed": ("speed_rpm", "RPM"), "default": ("active_power_kw", "kW"),
    },
    "em": {
        "power": ("active_power_total_kw", "kW"), "voltage": ("voltage_avg", "V"),
        "current": ("current_avg", "A"), "power_factor": ("power_factor", "PF"),
        "frequency": ("frequency_hz", "Hz"), "default": ("active_power_total_kw", "kW"),
    },
}

# Number words to digits
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "first": "1", "second": "2", "third": "3",
}


def resolve_entity_to_table(entity: str) -> Optional[tuple]:
    """
    Resolve a natural language entity name to a PostgreSQL timeseries table.

    Examples:
        "transformer 1"   → ("trf_001", "trf", "active_power_kw", "kW")
        "Transformer One"  → ("trf_001", "trf", "active_power_kw", "kW")
        "chiller 3"        → ("chiller_003", "chiller", "power_consumption_kw", "kW")
        "DG 2"             → ("dg_002", "dg", "active_power_kw", "kW")
        "trf_005"          → ("trf_005", "trf", "active_power_kw", "kW")

    Returns:
        (table_name, prefix, default_metric_col, default_unit) or None
    """
    if not entity:
        return None

    normalized = entity.lower().strip()

    # Replace number words with digits
    for word, digit in _NUMBER_WORDS.items():
        normalized = re.sub(rf'\b{word}\b', digit, normalized)

    # Check if entity is already a valid table name (e.g., "trf_001")
    table_match = re.match(r'^([a-z_]+)_(\d{3})$', normalized)
    if table_match:
        prefix = table_match.group(1)
        for kw, (pfx, metric, unit) in ENTITY_TABLE_PREFIX_MAP.items():
            if pfx == prefix:
                return (normalized, prefix, metric, unit)
        # If prefix not in map, still return it
        return (normalized, prefix, "active_power_kw", "kW")

    # Extract trailing number
    num_match = re.search(r'(\d+)\s*$', normalized)
    entity_number = int(num_match.group(1)) if num_match else None

    # Match against known equipment keywords
    for keyword, (prefix, default_metric, default_unit) in ENTITY_TABLE_PREFIX_MAP.items():
        if keyword in normalized:
            if entity_number is not None:
                table_name = f"{prefix}_{entity_number:03d}"
                return (table_name, prefix, default_metric, default_unit)
            else:
                # No number specified — default to _001
                table_name = f"{prefix}_001"
                return (table_name, prefix, default_metric, default_unit)

    return None


def resolve_metric_column(prefix: str, metric: str) -> tuple:
    """
    Resolve a metric alias to the actual PostgreSQL column name for an equipment type.

    Args:
        prefix: Equipment table prefix (e.g., "trf", "dg", "chiller")
        metric: Metric alias (e.g., "power", "load", "temperature")

    Returns:
        (column_name, unit)
    """
    metric_map = EQUIPMENT_METRIC_MAP.get(prefix)
    if not metric_map:
        # For LT panels and unknown types, try common column names
        metric_map = EQUIPMENT_METRIC_MAP.get("em", {})

    if not metric:
        return metric_map.get("default", ("active_power_kw", "kW"))

    metric_lower = metric.lower().strip().replace(" ", "_")

    # Direct match
    if metric_lower in metric_map:
        return metric_map[metric_lower]

    # Try common aliases
    aliases = {
        "power_kw": "power", "consumption": "power", "energy": "power",
        "power_consumption": "power", "power_output": "power",
        "demand": "power", "kw": "power", "watt": "power",
        "power_factor": "power_factor", "pf": "power_factor",
        "voltage_avg": "voltage", "volt": "voltage", "v": "voltage",
        "current_avg": "current", "amp": "current", "ampere": "current",
        "freq": "frequency", "hz": "frequency",
        "temp": "temperature", "thermal": "temperature",
        "water_temperature": "temperature", "water_temp": "temperature",
        "oil_temp": "oil_temp", "oil_temperature": "oil_temp",
        "vibration_level": "vibration", "vibration_levels": "vibration",
        "pressure_reading": "pressure", "pressure_level": "pressure",
        "flow_rate": "flow", "humidity_level": "humidity",
        "battery_level": "battery", "battery_charge": "battery",
        "battery_status": "battery_status", "charge": "battery",
        "current_load": "load", "load_percent": "load",
        "cop_value": "cop", "efficiency_value": "cop",
    }
    resolved = aliases.get(metric_lower, metric_lower)
    if resolved in metric_map:
        return metric_map[resolved]

    # Fuzzy: check if any metric_map key is contained in the metric string
    for key in metric_map:
        if key != "default" and key in metric_lower:
            return metric_map[key]

    return metric_map.get("default", ("active_power_kw", "kW"))


def _validate_widget_data(scenario: str, data: dict) -> tuple[bool, list[str]]:
    """
    Validate widget data against its schema.

    Returns:
        (is_valid, missing_field_names) tuple for backward-compatible test access.
    """
    result = validate_widget_data(scenario, data, raise_on_error=False)
    if result["is_valid"]:
        return (True, [])
    missing = []
    for error in result.get("errors", []):
        if "Missing required field" in error and "'" in error:
            missing.append(error.split("'")[1])
        elif "is null" in error and "'" in error:
            missing.append(error.split("'")[1])
    return (False, missing)
from layer2.widget_normalizer import normalize_widget_data, NormalizationError
from layer2.widget_selector import WidgetPlanItem

logger = logging.getLogger(__name__)


def _normalize_and_validate(scenario: str, data: dict) -> dict:
    """
    Pipeline: normalize → validate → return.

    1. NORMALIZE: Lossless, unambiguous transformations (unit conversion, etc.)
    2. VALIDATE: Strict validation (unchanged)

    Args:
        scenario: Widget scenario name (e.g., "kpi", "trend", "alerts")
        data: Collected data dict (may contain demoData, config, or direct fields)

    Returns:
        Normalized data (if validation passes)

    Raises:
        NormalizationError: If data is ambiguous (cannot determine intent)
        ValidationError: If data is invalid (fails strict validation)
    """
    # Step 1: Normalize (lossless transformations)
    result = normalize_widget_data(scenario, data)

    # Step 2: Validate (strict, unchanged)
    validate_widget_data(scenario, result.data)

    # Return normalized data
    return result.data


# Map collection short names → full names
COLLECTION_MAP = {
    "equipment": EQUIPMENT_COLLECTION,
    "alerts": ALERTS_COLLECTION,
    "maintenance": MAINTENANCE_COLLECTION,
    "operational_docs": OPERATIONAL_DOCS_COLLECTION,
    "shift_logs": SHIFT_LOGS_COLLECTION,
    "work_orders": WORK_ORDERS_COLLECTION,
}


class SchemaDataCollector:
    """
    Collects data for each widget in the plan using targeted RAG queries,
    formats results to match widget data schemas.
    """

    def __init__(self):
        self._pipeline: Optional[IndustrialRAGPipeline] = None
        self._kpi_lock = threading.Lock()

    @property
    def pipeline(self) -> IndustrialRAGPipeline:
        if self._pipeline is None:
            self._pipeline = get_rag_pipeline()
        return self._pipeline

    # ── PostgreSQL Timeseries Direct Queries ──

    def _query_pg_latest(self, table_name: str, columns: list[str] = None) -> Optional[dict]:
        """
        Query the latest reading from a PostgreSQL timeseries table.

        Args:
            table_name: Equipment table name (e.g., 'trf_001')
            columns: Specific columns to fetch (None = all)

        Returns:
            Dict with column values, or None if table doesn't exist
        """
        from django.db import connections
        try:
            col_str = ', '.join(columns) if columns else '*'
            sql = f"SELECT {col_str} FROM {table_name} ORDER BY timestamp DESC LIMIT 1"
            with connections['timeseries'].cursor() as cursor:
                cursor.execute(sql)
                cols = [desc[0] for desc in cursor.description]
                row = cursor.fetchone()
                if row:
                    return dict(zip(cols, row))
        except Exception as e:
            logger.debug(f"PG latest query failed for {table_name}: {e}")
        return None

    def _query_pg_timeseries(self, table_name: str, metric_col: str,
                              hours: int = 24, interval: str = '1 hour') -> list[dict]:
        """
        Query aggregated timeseries data from PostgreSQL.

        Uses the latest available data point as reference (not NOW()),
        so queries work even when data generation has stopped.

        Args:
            table_name: Equipment table name (e.g., 'trf_001')
            metric_col: Column to aggregate (e.g., 'active_power_kw')
            hours: Hours of history to fetch
            interval: Aggregation interval (e.g., '1 hour', '15 minutes')

        Returns:
            List of {timestamp, value} dicts
        """
        from django.db import connections
        try:
            # Map interval strings to date_trunc fields
            # date_trunc takes: 'minute', 'hour', 'day', etc. — not '1 hour'
            trunc_map = {
                '1 minute': 'minute', '5 minutes': 'minute', '10 minutes': 'minute',
                '15 minutes': 'minute', '30 minutes': 'minute',
                '1 hour': 'hour', 'hour': 'hour',
                '1 day': 'day', 'day': 'day',
            }
            trunc_field = trunc_map.get(interval, 'hour')

            # Use latest available timestamp as reference point
            sql = f"""
                WITH latest AS (
                    SELECT MAX(timestamp) AS max_ts FROM {table_name}
                )
                SELECT date_trunc('{trunc_field}', t.timestamp) AS ts,
                       AVG(t.{metric_col}) AS value
                FROM {table_name} t, latest l
                WHERE t.timestamp >= l.max_ts - INTERVAL '{hours} hours'
                  AND t.timestamp <= l.max_ts
                GROUP BY 1
                ORDER BY 1
            """
            with connections['timeseries'].cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                return [{"time": str(row[0]), "value": round(float(row[1]), 2) if row[1] else 0}
                        for row in rows]
        except Exception as e:
            logger.debug(f"PG timeseries query failed for {table_name}.{metric_col}: {e}")
        return []

    def _query_pg_stats(self, table_name: str, metric_col: str,
                         hours: int = 24) -> Optional[dict]:
        """
        Query aggregated statistics (min, max, avg, latest) from PostgreSQL.

        Uses the latest available data point as reference (not NOW()),
        so queries work even when data generation has stopped.

        Args:
            table_name: Equipment table name
            metric_col: Column to aggregate

        Returns:
            Dict with min, max, avg, latest values
        """
        from django.db import connections
        try:
            sql = f"""
                WITH latest AS (
                    SELECT MAX(timestamp) AS max_ts FROM {table_name}
                )
                SELECT MIN(t.{metric_col}), MAX(t.{metric_col}), AVG(t.{metric_col}),
                       (SELECT {metric_col} FROM {table_name} ORDER BY timestamp DESC LIMIT 1)
                FROM {table_name} t, latest l
                WHERE t.timestamp >= l.max_ts - INTERVAL '{hours} hours'
                  AND t.timestamp <= l.max_ts
            """
            with connections['timeseries'].cursor() as cursor:
                cursor.execute(sql)
                row = cursor.fetchone()
                if row and row[0] is not None:
                    return {
                        "min": round(float(row[0]), 2),
                        "max": round(float(row[1]), 2),
                        "avg": round(float(row[2]), 2),
                        "latest": round(float(row[3]), 2) if row[3] is not None else None,
                    }
        except Exception as e:
            logger.debug(f"PG stats query failed for {table_name}.{metric_col}: {e}")
        return None

    def _query_pg_multi_metric_latest(self, table_name: str, metrics: list[str]) -> Optional[dict]:
        """
        Query latest values for multiple metrics from a PostgreSQL table.

        Returns:
            Dict mapping metric_col → value
        """
        from django.db import connections
        try:
            cols = ', '.join(metrics)
            sql = f"SELECT {cols} FROM {table_name} ORDER BY timestamp DESC LIMIT 1"
            with connections['timeseries'].cursor() as cursor:
                cursor.execute(sql)
                row = cursor.fetchone()
                if row:
                    return {m: (round(float(v), 2) if v is not None else None)
                            for m, v in zip(metrics, row)}
        except Exception as e:
            logger.debug(f"PG multi-metric query failed for {table_name}: {e}")
        return None

    def _resolve_and_query_entity(self, entity: str, metric: str = "") -> Optional[dict]:
        """
        Resolve a natural language entity to a PG table and query latest value.

        Returns:
            Dict with: table_name, label, value, unit, metric_col, prefix
            or None if resolution fails.
        """
        resolved = resolve_entity_to_table(entity)
        if not resolved:
            return None

        table_name, prefix, default_metric, default_unit = resolved
        metric_col, unit = resolve_metric_column(prefix, metric) if metric else (default_metric, default_unit)

        latest = self._query_pg_latest(table_name, ['timestamp', metric_col])
        if not latest:
            return None

        value = latest.get(metric_col)
        if value is None:
            return None

        # Build a human-readable label
        prefix_labels = {
            "trf": "Transformer", "dg": "DG Set", "ups": "UPS", "chiller": "Chiller",
            "ahu": "AHU", "ct": "Cooling Tower", "pump": "Pump", "compressor": "Compressor",
            "motor": "Motor", "em": "Energy Meter", "lt_mcc": "MCC Panel",
            "lt_pcc": "PCC Panel", "lt_apfc": "APFC Panel", "lt_db": "Distribution Board",
            "lt_vfd": "VFD Panel", "lt_plc": "PLC Panel", "lt_ats": "ATS Panel",
            "lt_changeover": "Changeover", "lt_mldb": "Main LT DB", "lt_smdb": "Sub-Main DB",
        }
        num_match = re.search(r'(\d+)$', table_name)
        num = int(num_match.group(1)) if num_match else 0
        label = f"{prefix_labels.get(prefix, prefix.upper())} {num}"

        return {
            "table_name": table_name,
            "prefix": prefix,
            "label": label,
            "value": round(float(value), 2) if isinstance(value, (int, float)) else value,
            "unit": unit,
            "metric_col": metric_col,
            "timestamp": str(latest.get("timestamp", "")),
        }

    def collect_all(self, widgets: list[WidgetPlanItem], query: str) -> list[dict]:
        """
        Collect data for all widgets in parallel.

        Widget selection is prompt-driven — every widget the LLM selected is
        kept in the layout.  If data collection or validation fails, the widget
        is included with its demo_shape placeholder so the frontend can still
        render it.

        Returns list of dicts with:
            scenario, size, data_override, relevance, why
        """
        # Reset per-run dedup trackers
        self._kpi_seen_entities = set()
        self._kpi_seen_meters = set()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._collect_one, w, query): i
                for i, w in enumerate(widgets)
            }
            results = [None] * len(widgets)
            for future in as_completed(futures):
                idx = futures[future]
                widget = widgets[idx]
                try:
                    data_override = future.result()
                except Exception as e:
                    logger.warning(f"Data collection failed for {widget.scenario}: {e}")
                    data_override = {}

                schema_valid = True
                # Try normalize → validate; fall back to demo_shape placeholder
                try:
                    normalized_data = _normalize_and_validate(widget.scenario, data_override)
                except (NormalizationError, ValidationError) as exc:
                    reason = getattr(exc, 'reason', None) or getattr(exc, 'errors', str(exc))
                    logger.warning(
                        f"Validation fallback for {widget.scenario}: {reason}"
                    )
                    # Use demo_shape placeholder so the widget still appears
                    schema = WIDGET_SCHEMAS.get(widget.scenario, {})
                    normalized_data = schema.get("demo_shape", data_override or {})
                    schema_valid = False

                results[idx] = {
                    "scenario": widget.scenario,
                    "size": widget.size,
                    "relevance": widget.relevance,
                    "why": widget.why,
                    "data_override": normalized_data,
                    "schema_valid": schema_valid,
                }

        return [r for r in results if r is not None]

    # Pattern for extracting equipment entities from natural language queries
    # e.g., "Transformer One and Transformer Two" -> ["Transformer One", "Transformer Two"]
    _ENTITY_SPLIT_RE = re.compile(
        r'(?:^|(?<=\s))'                              # start or after space
        r'((?:main\s+|dry\s+type\s+)?'                # optional prefix
        r'(?:transformer|trafo|generator|genset|dg|ups|chiller|ahu|'  # equipment keywords
        r'cooling\s+tower|pump|compressor|motor|energy\s+meter|meter|'
        r'mcc|pcc|apfc|vfd|plc|ats|changeover|'
        r'trf|ct|em|lt_\w+)'                          # table prefix forms
        r'(?:[_\s]+(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+))?)',  # optional number
        re.IGNORECASE
    )

    # Metric keywords in query text → metric alias for resolve_metric_column()
    _QUERY_METRIC_KEYWORDS = {
        "temperature": "temperature", "temp": "temperature",
        "battery": "battery", "battery status": "battery",
        "battery health": "battery_health", "runtime": "runtime",
        "cop": "cop", "efficiency": "cop",
        "voltage": "voltage", "power factor": "power_factor",
        "frequency": "frequency", "vibration": "vibration",
        "pressure": "pressure", "flow": "flow", "flow rate": "flow",
        "humidity": "humidity", "load": "load",
    }

    def _expand_all_entities(self, query: str, entities: list) -> list:
        """Expand bare equipment keywords to numbered instances for 'all X' queries.

        When the query implies multiple units (e.g., 'all UPS systems', 'show every chiller')
        and entities contain a bare type keyword without a number, expand to
        the first N actual PG table instances.
        """
        q = query.lower()
        all_indicators = ("all ", "every ", "each ", "systems", "units")
        if not any(ind in q for ind in all_indicators):
            return entities

        expanded = []
        for ent in entities:
            ent_lower = ent.lower().strip()
            # Check if entity is a bare keyword (no trailing number)
            has_number = re.search(r'\d+$', ent_lower)
            if has_number:
                expanded.append(ent)
                continue

            # Try exact match first, then look for equipment keyword within the entity
            matched_key = None
            if ent_lower in ENTITY_TABLE_PREFIX_MAP:
                matched_key = ent_lower
            else:
                # Search for equipment keywords within the entity string
                for kw in sorted(ENTITY_TABLE_PREFIX_MAP.keys(), key=len, reverse=True):
                    if kw in ent_lower:
                        matched_key = kw
                        break

            if not matched_key:
                expanded.append(ent)
                continue

            # Expand: find all PG tables for this prefix
            prefix, _, _ = ENTITY_TABLE_PREFIX_MAP[matched_key]
            try:
                from django.db import connections
                with connections['timeseries'].cursor() as cursor:
                    cursor.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema='public' AND table_name LIKE %s "
                        "ORDER BY table_name LIMIT 8",
                        [f"{prefix}_%"]
                    )
                    tables = [row[0] for row in cursor.fetchall()]
                    if tables:
                        for tbl in tables:
                            num_match = re.search(r'(\d+)$', tbl)
                            if num_match:
                                expanded.append(f"{ent_lower} {int(num_match.group(1))}")
                        logger.info(f"Expanded '{ent}' to {len(tables)} instances: {expanded[-len(tables):]}")
                    else:
                        expanded.append(ent)
            except Exception as e:
                logger.debug(f"Entity expansion failed for {ent}: {e}")
                expanded.append(ent)

        return expanded if expanded else entities

    def _extract_metric_from_query(self, query: str) -> str:
        """Extract a metric alias from query text for resolve_metric_column()."""
        q = query.lower()
        # Check multi-word keys first (longer matches)
        for kw in sorted(self._QUERY_METRIC_KEYWORDS, key=len, reverse=True):
            if kw in q:
                return self._QUERY_METRIC_KEYWORDS[kw]
        return ""

    def _extract_entities_from_query(self, query: str) -> list[str]:
        """Extract equipment entity mentions from a natural language query."""
        matches = self._ENTITY_SPLIT_RE.findall(query)
        return [m.strip() for m in matches if m.strip()]

    def _collect_one(self, widget: WidgetPlanItem, query: str) -> dict:
        """Collect data for a single widget based on its schema and data_request."""
        schema = WIDGET_SCHEMAS.get(widget.scenario)
        if not schema:
            return {}

        strategy = schema["rag_strategy"]
        data_request = widget.data_request or {}
        search_query = data_request.get("query", query)
        entities = data_request.get("entities", [])
        # AUDIT FIX: Ensure metric is a string (LLM sometimes returns a list)
        raw_metric = data_request.get("metric", "")
        metric = raw_metric[0] if isinstance(raw_metric, list) and raw_metric else (raw_metric if isinstance(raw_metric, str) else "")

        # Extract metric from query text — always attempt, and prefer over LLM metric.
        # The user's explicit words ("vibration", "temperature", "voltage") are the
        # strongest signal for what metric they want.  The LLM widget selector often
        # defaults to power/load regardless of the user's actual request.
        query_metric = self._extract_metric_from_query(query)
        if not metric:
            metric = query_metric
        elif query_metric:
            # User's query explicitly mentions a metric — trust it over the LLM's choice
            # unless both agree (LLM metric resolves to same alias)
            llm_lower = metric.lower()
            if query_metric != llm_lower and query_metric not in llm_lower:
                logger.info(f"Override LLM metric '{metric}' with query-extracted '{query_metric}'")
                metric = query_metric

        # When LLM doesn't fill data_request, extract entities from query text
        if not entities:
            extracted = self._extract_entities_from_query(query)
            if extracted:
                entities = extracted
                logger.info(f"Extracted entities from query: {entities}")

        # Expand "all X" queries: bare equipment keyword → numbered instances
        # e.g., "show all UPS systems" with entity ["ups"] → ["ups 1", "ups 2", ...]
        entities = self._expand_all_entities(query, entities)

        # Choose collection to search
        collections = data_request.get("collections", schema.get("default_collections", ["equipment"]))
        collection_names = [COLLECTION_MAP.get(c, c) for c in collections if c in COLLECTION_MAP]

        # No-data widgets (helpview, pulseview, chatstream)
        if strategy == "none":
            shape = schema.get("demo_shape", {})
            if isinstance(shape, dict):
                shape["_synthetic"] = True
                shape["_data_source"] = "widget_demo_shape"
            return shape

        # Route to appropriate strategy
        if strategy == "single_metric":
            return self._collect_single_metric(search_query, entities, metric, collection_names)
        elif strategy == "alert_query":
            return self._collect_alerts(search_query, entities)
        elif strategy == "multi_entity_metric":
            return self._collect_comparison(search_query, entities, metric, collection_names)
        elif strategy == "time_series":
            return self._collect_time_series(search_query, entities, metric)
        elif strategy == "cumulative_time_series":
            return self._collect_cumulative_time_series(search_query, entities, metric)
        elif strategy == "multi_time_series":
            return self._collect_multi_time_series(search_query, entities, metric)
        elif strategy == "aggregation":
            return self._collect_aggregation(search_query, entities, metric, collection_names)
        elif strategy == "events_in_range":
            return self._collect_events(search_query, entities, collection_names)
        elif strategy == "single_entity_deep":
            return self._collect_device_detail(search_query, entities)
        elif strategy == "cross_tabulation":
            return self._collect_matrix(search_query, entities, collection_names)
        elif strategy == "flow_analysis":
            return self._collect_flow(search_query, entities)
        elif strategy == "people_query":
            return self._collect_people(search_query)
        elif strategy == "supply_query":
            return self._collect_supply(search_query)
        else:
            # Generic: search and return raw context
            return self._collect_generic(search_query, collection_names)

    # ── Strategy implementations ──

    # Track which entities we've already used for KPIs in this collection run
    # to avoid duplicate data across multiple KPI widgets
    _kpi_seen_entities: set = set()

    # Equipment type keywords for direct table lookup (bypass energy_readings)
    _EQUIPMENT_KEYWORDS = {
        "transformer", "trafo", "generator", "genset", "dg", "diesel",
        "ups", "chiller", "ahu", "air handling", "cooling tower",
        "pump", "compressor", "motor", "energy meter", "meter",
        "mcc", "pcc", "apfc", "vfd", "plc", "ats", "changeover",
    }

    def _collect_single_metric(self, query: str, entities: list, metric: str, collections: list) -> dict:
        """Collect a single metric for KPI widget.

        Uses entity/metric from data_request when available, and skips
        entities already shown in other KPIs to ensure diversity.
        """
        vs = self.pipeline.vector_store
        # Pick first entity not yet shown in another KPI widget (thread-safe)
        entity = ""
        with self._kpi_lock:
            for e in (entities or []):
                if e.lower() not in self._kpi_seen_entities:
                    entity = e
                    self._kpi_seen_entities.add(e.lower())
                    break
        if not entity:
            entity = entities[0] if entities else ""

        # If entity is a known equipment type (transformer, generator, etc.),
        # try equipment tables FIRST before energy_readings.
        # For equipment queries, pass the raw metric through to resolve_metric_column()
        # which handles per-equipment-type resolution (load → load_percent for UPS, etc.)
        entity_lower = entity.lower()
        query_lower = query.lower()
        is_equipment = any(kw in entity_lower or kw in query_lower for kw in self._EQUIPMENT_KEYWORDS)

        # Only apply energy_readings-specific aliases for NON-equipment queries
        if is_equipment:
            resolved_metric = metric or ""
        else:
            metric_aliases = {
                "power": "power_kw", "consumption": "power_kw", "energy": "power_kw",
                "load": "power_kw", "demand": "max_demand_kw",
                "power_factor": "power_factor", "pf": "power_factor",
                "voltage": "voltage_avg", "frequency": "frequency",
            }
            resolved_metric = metric_aliases.get((metric or "").lower().strip(), metric)
        if is_equipment:
            # Use the resolved/extracted metric; only default to power_kw if no metric at all
            if not resolved_metric:
                resolved_metric = "power_kw"
            equip_result = self._try_equipment_power_data(entity or query, resolved_metric)
            if equip_result:
                # Mark this entity as seen for KPI dedup
                label = equip_result.get("demoData", {}).get("label", "")
                if label:
                    self._kpi_seen_entities.add(label.lower())
                self._kpi_seen_entities.add(entity_lower)
                return equip_result

        # If a specific energy metric is requested, try SQL
        energy_metrics = {"power_kw", "power_factor", "voltage_avg", "frequency",
                         "energy_kwh", "max_demand_kw", "max_demand_kva"}
        if resolved_metric in energy_metrics:
            return self._collect_kpi_from_energy_sql(entity, resolved_metric)

        # If query context suggests energy, try SQL with power_kw
        energy_keywords = {"energy", "power", "consumption", "kw", "kwh", "load", "demand", "meter", "voltage"}
        query_words = set(query.lower().split())
        if query_words & energy_keywords and not metric:
            return self._collect_kpi_from_energy_sql(entity, "power_kw")

        search_q = f"{entity} {metric}".strip() or query
        coll = collections[0] if collections else EQUIPMENT_COLLECTION

        # Fetch multiple results to ensure diversity across KPI widgets
        results = vs.search(coll, search_q, n_results=5)

        if not results:
            return {"demoData": {"label": entity or "Unknown", "value": "N/A", "unit": "",
                                 "_synthetic": True, "_data_source": "no_data_fallback"}}

        # Pick first result not already used in another KPI
        doc = None
        for r in results:
            eid = r.metadata.get("equipment_id", r.metadata.get("name", ""))
            if eid not in self._kpi_seen_entities:
                doc = r
                self._kpi_seen_entities.add(eid)
                break
        if doc is None:
            doc = results[0]

        meta = doc.metadata
        value, unit, state = self._extract_metric_from_doc(doc, metric)

        return {
            "demoData": {
                "label": meta.get("name", entity or search_q[:30]),
                "value": str(value),
                "unit": unit,
                "state": state,
            }
        }

    # Track meters already used in KPI SQL queries for diversity
    _kpi_seen_meters: set = set()

    def _collect_kpi_from_energy_sql(self, entity: str, metric: str) -> dict:
        """Collect a KPI value directly from energy_readings SQL.

        Falls back to equipment tables (transformers, generators, etc.) when
        no matching meter is found in energy_readings.
        Ensures diversity by tracking which meters have been used.
        """
        from django.db import connection
        try:
            metric_col = metric if metric in (
                "power_kw", "power_factor", "voltage_avg", "frequency"
            ) else "power_kw"
            with connection.cursor() as c:
                if entity:
                    c.execute(f"""
                        SELECT meter_id, meter_name, {metric_col}
                        FROM energy_readings
                        WHERE meter_id = %s OR meter_name LIKE %s
                        ORDER BY timestamp DESC LIMIT 1
                    """, [entity, f"%{entity}%"])
                else:
                    # Get top meters, skip ones already shown
                    skip_clause = ""
                    params = []
                    if self._kpi_seen_meters:
                        placeholders = ",".join(["%s"] * len(self._kpi_seen_meters))
                        skip_clause = f"WHERE meter_id NOT IN ({placeholders})"
                        params = list(self._kpi_seen_meters)
                    c.execute(f"""
                        SELECT meter_id, meter_name, {metric_col}
                        FROM energy_readings
                        {skip_clause}
                        ORDER BY energy_kwh_cumulative DESC LIMIT 1
                    """, params)
                row = c.fetchone()
                if row:
                    self._kpi_seen_meters.add(row[0])  # row[0] = meter_id
                    unit_map = {
                        "power_kw": "kW", "power_factor": "PF",
                        "voltage_avg": "V", "frequency": "Hz",
                    }
                    val = round(float(row[2]), 1) if row[2] is not None else "N/A"
                    return {
                        "demoData": {
                            "label": row[1] or entity or "Meter",  # row[1] = meter_name
                            "value": str(val),
                            "unit": unit_map.get(metric_col, ""),
                            "state": "normal",
                        }
                    }
        except Exception as e:
            logger.warning(f"KPI energy SQL failed: {e}")

        # Fallback: try equipment tables when energy_readings has no match
        if entity:
            equip_result = self._try_equipment_power_data(entity, metric)
            if equip_result:
                return equip_result

        return {"demoData": {"label": entity or "Unknown", "value": "N/A", "unit": "",
                             "_synthetic": True, "_data_source": "no_data_fallback"}}

    def _try_equipment_power_data(self, entity: str, metric: str) -> Optional[dict]:
        """Try to find power/energy data from PostgreSQL timeseries tables.

        PRIMARY: Queries command_center_data PostgreSQL (276GB, 357 equipment tables)
        FALLBACK: Queries Django SQLite industrial_* metadata tables

        Handles fuzzy name matching: 'Transformer One' -> trf_001,
        'Transformer 2' -> trf_002, 'DG 3' -> dg_003, etc.
        """
        # ── PRIMARY: PostgreSQL timeseries (command_center_data) ──
        pg_result = self._resolve_and_query_entity(entity, metric)
        if pg_result:
            logger.info(
                f"PG Timeseries: {entity} → {pg_result['table_name']}.{pg_result['metric_col']}"
                f" = {pg_result['value']} {pg_result['unit']} (ts={pg_result['timestamp'][:19]})"
            )
            return {
                "demoData": {
                    "label": pg_result["label"],
                    "value": str(pg_result["value"]),
                    "unit": pg_result["unit"],
                    "state": "normal",
                    "_data_source": f"pg_timeseries:{pg_result['table_name']}",
                    "_table": pg_result["table_name"],
                }
            }

        # ── FALLBACK: Django SQLite metadata tables ──
        from django.db import connection

        number_words = {
            "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        }
        normalized = entity.lower().strip()
        for word, digit in number_words.items():
            normalized = re.sub(rf'\b{word}\b', digit, normalized)

        num_match = re.search(r'(\d+)\s*$', normalized)
        entity_number = num_match.group(1) if num_match else ""

        equipment_tables = [
            {
                "table": "industrial_transformer",
                "keywords": ["transformer"],
                "columns": {
                    "power_kw": ("current_load_kw", "kW"),
                    "voltage_avg": ("output_voltage", "V"),
                    "power_factor": ("load_percent", "%"),
                },
                "default_col": ("current_load_kw", "kW"),
            },
            {
                "table": "industrial_dieselgenerator",
                "keywords": ["generator", "dg", "genset"],
                "columns": {
                    "power_kw": ("current_load_kw", "kW"),
                    "voltage_avg": ("output_voltage", "V"),
                    "frequency": ("output_frequency", "Hz"),
                },
                "default_col": ("current_load_kw", "kW"),
            },
            {
                "table": "industrial_ups",
                "keywords": ["ups"],
                "columns": {
                    "power_kw": ("current_load_kw", "kW"),
                    "voltage_avg": ("output_voltage", "V"),
                },
                "default_col": ("current_load_kw", "kW"),
            },
        ]

        for tbl in equipment_tables:
            if not any(kw in normalized for kw in tbl["keywords"]):
                continue
            col_info = tbl["columns"].get(metric, tbl["default_col"])
            col_name, unit = col_info
            try:
                with connection.cursor() as c:
                    like_patterns = []
                    if entity_number:
                        like_patterns.append(f"% {entity_number}")
                        like_patterns.append(f"%{entity_number}")
                    like_patterns.append(f"%{normalized}%")

                    for pattern in like_patterns:
                        c.execute(f"""
                            SELECT name, {col_name}, health_score, status
                            FROM {tbl['table']}
                            WHERE LOWER(name) LIKE %s
                            ORDER BY
                                CASE WHEN LOWER(name) LIKE 'main %%' THEN 0 ELSE 1 END,
                                name
                            LIMIT 1
                        """, [pattern])
                        row = c.fetchone()
                        if row:
                            val = round(float(row[1]), 1) if row[1] is not None else "N/A"
                            state = "warning" if row[3] in ("fault", "offline", "stopped") else "normal"
                            logger.info(f"SQLite fallback: found {row[0]} in {tbl['table']} ({col_name}={val})")
                            return {
                                "demoData": {
                                    "label": row[0],
                                    "value": str(val),
                                    "unit": unit,
                                    "state": state,
                                    "_data_source": f"sqlite:{tbl['table']}",
                                }
                            }
            except Exception as e:
                logger.warning(f"Equipment power fallback failed for {tbl['table']}: {e}")

        return None

    def _collect_alerts(self, query: str, entities: list) -> dict:
        """Collect alerts for alerts widget."""
        vs = self.pipeline.vector_store
        search_q = query

        results = vs.search(ALERTS_COLLECTION, search_q, n_results=5)
        if not results:
            return {"demoData": []}

        alerts = []
        for r in results:
            meta = r.metadata
            severity = meta.get("severity", "info")
            content = r.content

            # Parse evidence from content
            evidence = self._parse_alert_evidence(content)

            alerts.append({
                "id": meta.get("equipment_id", "ALT-000"),
                "title": evidence.get("label", meta.get("equipment_name", "Alert")),
                "message": content.split("|")[0].replace("Alert:", "").strip()[:120] if "|" in content else content[:120],
                "severity": severity,
                "category": "Equipment",
                "source": meta.get("equipment_name", "Unknown"),
                "state": "acknowledged" if meta.get("acknowledged") else "new",
                "evidence": {
                    "label": evidence.get("label", "Value"),
                    "value": evidence.get("value", "—"),
                    "unit": evidence.get("unit", ""),
                    "trend": "up" if severity in ("critical", "high", "warning") else "stable",
                },
                "threshold": evidence.get("threshold", "N/A"),
                "actions": [
                    {"label": "Acknowledge", "intent": "ack", "type": "primary"},
                    {"label": "View Details", "intent": "open", "type": "secondary"},
                ],
            })

        return {"demoData": alerts}

    def _collect_comparison(self, query: str, entities: list, metric: str, collections: list) -> dict:
        """Collect comparison data for two entities from PostgreSQL timeseries.

        PRIMARY: Resolves entities to PG tables and queries real timeseries data.
        FALLBACK: Django SQLite metadata → ChromaDB vector search.
        """
        entity_a = entities[0] if len(entities) > 0 else ""
        entity_b = entities[1] if len(entities) > 1 else ""

        # ── PRIMARY: PostgreSQL timeseries comparison ──
        resolved_a = resolve_entity_to_table(entity_a) if entity_a else None
        resolved_b = resolve_entity_to_table(entity_b) if entity_b else None

        if resolved_a and resolved_b:
            table_a, prefix_a, default_metric_a, default_unit_a = resolved_a
            table_b, prefix_b, default_metric_b, default_unit_b = resolved_b

            # Resolve the metric column for both (use prefix_a for metric resolution)
            metric_col_a, unit_a = resolve_metric_column(prefix_a, metric) if metric else (default_metric_a, default_unit_a)
            metric_col_b, unit_b = resolve_metric_column(prefix_b, metric) if metric else (default_metric_b, default_unit_b)

            # Query 24h stats for both
            stats_a = self._query_pg_stats(table_a, metric_col_a, hours=24)
            stats_b = self._query_pg_stats(table_b, metric_col_b, hours=24)

            if stats_a and stats_b:
                val_a = stats_a["latest"] if stats_a["latest"] is not None else stats_a["avg"]
                val_b = stats_b["latest"] if stats_b["latest"] is not None else stats_b["avg"]

                try:
                    delta = float(val_a) - float(val_b)
                    delta_pct = (delta / float(val_b) * 100) if float(val_b) != 0 else 0
                except (ValueError, TypeError, ZeroDivisionError):
                    delta, delta_pct = 0, 0

                # Build labels
                prefix_labels = {
                    "trf": "Transformer", "dg": "DG Set", "ups": "UPS",
                    "chiller": "Chiller", "ahu": "AHU", "ct": "Cooling Tower",
                    "pump": "Pump", "compressor": "Compressor", "motor": "Motor",
                    "em": "Energy Meter",
                }
                num_a = re.search(r'(\d+)$', table_a)
                num_b = re.search(r'(\d+)$', table_b)
                label_a = f"{prefix_labels.get(prefix_a, prefix_a.upper())} {int(num_a.group(1))}" if num_a else table_a
                label_b = f"{prefix_labels.get(prefix_b, prefix_b.upper())} {int(num_b.group(1))}" if num_b else table_b

                metric_label = metric_col_a.replace("_", " ").replace("kw", "kW").title() if not metric else metric.replace("_", " ").title()

                logger.info(
                    f"PG Comparison: {table_a}={val_a} vs {table_b}={val_b} "
                    f"({metric_col_a}, delta={delta:.2f})"
                )

                return {
                    "demoData": {
                        "label": metric_label,
                        "unit": unit_a,
                        "labelA": label_a,
                        "valueA": str(val_a),
                        "labelB": label_b,
                        "valueB": str(val_b),
                        "delta": round(delta, 2),
                        "deltaPct": round(delta_pct, 1),
                        "avgA": stats_a["avg"],
                        "avgB": stats_b["avg"],
                        "minA": stats_a["min"],
                        "minB": stats_b["min"],
                        "maxA": stats_a["max"],
                        "maxB": stats_b["max"],
                        "_data_source": f"pg_timeseries:{table_a},{table_b}",
                    }
                }

        # ── FALLBACK: _try_equipment_power_data (tries PG first, then SQLite) ──
        query_lower = query.lower()
        is_equipment = any(kw in query_lower or kw in entity_a.lower() or kw in entity_b.lower()
                           for kw in self._EQUIPMENT_KEYWORDS)
        if is_equipment or resolved_a or resolved_b:
            equip_a = self._try_equipment_power_data(entity_a or query, metric) if entity_a else None
            equip_b = self._try_equipment_power_data(entity_b or query, metric) if entity_b else None

            if equip_a and equip_b:
                da = equip_a.get("demoData", {})
                db_data = equip_b.get("demoData", {})
                val_a = da.get("value", "N/A")
                val_b = db_data.get("value", "N/A")
                unit = da.get("unit", db_data.get("unit", ""))
                try:
                    delta = float(val_a) - float(val_b)
                    delta_pct = (delta / float(val_b) * 100) if float(val_b) != 0 else 0
                except (ValueError, TypeError, ZeroDivisionError):
                    delta, delta_pct = 0, 0
                return {
                    "demoData": {
                        "label": f"{metric or 'Power'}".replace("_", " ").title(),
                        "unit": unit,
                        "labelA": da.get("label", entity_a),
                        "valueA": val_a,
                        "labelB": db_data.get("label", entity_b),
                        "valueB": val_b,
                        "delta": round(delta, 2),
                        "deltaPct": round(delta_pct, 1),
                    }
                }

        # ── LAST RESORT: vector search ──
        vs = self.pipeline.vector_store
        coll = collections[0] if collections else EQUIPMENT_COLLECTION

        result_a = vs.search(coll, f"{entity_a} {metric}".strip() or query, n_results=1)
        result_b = vs.search(coll, f"{entity_b} {metric}".strip() or query, n_results=1) if entity_b else []

        val_a, unit_a, _ = self._extract_metric_from_doc(result_a[0], metric) if result_a else ("N/A", "", "normal")
        val_b, unit_b, _ = self._extract_metric_from_doc(result_b[0], metric) if result_b else ("N/A", "", "normal")

        try:
            delta = float(val_a) - float(val_b)
            delta_pct = (delta / float(val_b) * 100) if float(val_b) != 0 else 0
        except (ValueError, TypeError, ZeroDivisionError):
            delta, delta_pct = 0, 0

        label_a = result_a[0].metadata.get("name", entity_a) if result_a else entity_a
        label_b = result_b[0].metadata.get("name", entity_b) if result_b else entity_b

        return {
            "demoData": {
                "label": f"{metric or 'Comparison'}".replace("_", " ").title(),
                "unit": unit_a or unit_b or "",
                "labelA": label_a or "Entity A",
                "valueA": val_a,
                "labelB": label_b or "Entity B",
                "valueB": val_b,
                "delta": round(delta, 2),
                "deltaPct": round(delta_pct, 1),
            }
        }

    def _collect_time_series(self, query: str, entities: list, metric: str) -> dict:
        """Collect time series data — PostgreSQL timeseries first, then energy_readings, then synthetic."""
        entity = entities[0] if entities else None

        # ── PRIMARY: PostgreSQL timeseries (command_center_data) ──
        if entity:
            resolved = resolve_entity_to_table(entity)
            if resolved:
                table_name, prefix, default_metric, default_unit = resolved
                metric_col, unit = resolve_metric_column(prefix, metric) if metric else (default_metric, default_unit)
                ts_data = self._query_pg_timeseries(table_name, metric_col, hours=24, interval='1 hour')
                if ts_data:
                    num_match = re.search(r'(\d+)$', table_name)
                    num = int(num_match.group(1)) if num_match else 0
                    prefix_labels = {
                        "trf": "Transformer", "dg": "DG Set", "ups": "UPS",
                        "chiller": "Chiller", "ahu": "AHU", "ct": "Cooling Tower",
                        "pump": "Pump", "compressor": "Compressor", "motor": "Motor",
                        "em": "Energy Meter",
                    }
                    label = f"{prefix_labels.get(prefix, prefix.upper())} {num}"
                    logger.info(f"PG TimeSeries: {table_name}.{metric_col} → {len(ts_data)} points")
                    return {
                        "demoData": {
                            "label": label,
                            "unit": unit,
                            "timeSeries": ts_data,
                            "timeRange": "last_24h",
                            "_data_source": f"pg_timeseries:{table_name}",
                            "_table": table_name,
                            "_metric": metric_col,
                        }
                    }

        # ── FALLBACK 1: energy_readings SQL ──
        energy_data = self.pipeline.query_energy_sql(equipment_id=entity)
        if energy_data:
            points = sorted(energy_data, key=lambda p: p.get("timestamp", ""))
            time_series = [
                {"time": str(p.get("timestamp", "")), "value": p.get("power_kw", 0)}
                for p in points
            ]
            label = energy_data[0].get("meter_name", entity or "Energy") if energy_data else "Trend"
            return {
                "demoData": {
                    "label": label,
                    "unit": "kW",
                    "timeSeries": time_series,
                    "timeRange": "last_24h",
                }
            }

        # ── FALLBACK 2: synthetic from vector metadata ──
        vs = self.pipeline.vector_store
        results = vs.search(EQUIPMENT_COLLECTION, f"{entity or ''} {metric}".strip() or query, n_results=1)
        meta = results[0].metadata if results else {}

        base_value = float(meta.get("health_score", meta.get("power_kw", 50)))
        unit = meta.get("unit", "kW" if "power" in (metric or "").lower() else "%")
        label = meta.get("name", entity or query[:30])

        import random
        now = datetime.now()
        time_series = []
        for i in range(48):
            t = now - timedelta(hours=24) + timedelta(minutes=i * 30)
            noise = random.uniform(-base_value * 0.08, base_value * 0.08)
            time_series.append({
                "time": t.isoformat(),
                "value": round(base_value + noise, 2),
            })

        return {
            "demoData": {
                "label": label,
                "unit": unit,
                "timeSeries": time_series,
                "timeRange": "last_24h",
                "_synthetic": True,
                "_data_source": "synthetic_from_vector_metadata",
            }
        }

    def _collect_cumulative_time_series(self, query: str, entities: list, metric: str) -> dict:
        """
        Collect time series data formatted for the trends-cumulative widget.

        The trends-cumulative component expects:
        {
            config: { title, subtitle, variant, mode, series: [{id, label, unit, color}] },
            data: [ {x: ISO_string, {id}_raw: float, {id}_cumulative: float}, ... ]
        }
        """
        entity = entities[0] if entities else None
        series_id = "S1"
        unit = "kW"
        label = metric.replace("_", " ").title() if metric else "Energy"

        # Try energy SQL for time series (actual readings with timestamps)
        energy_data = self.pipeline.query_energy_sql(equipment_id=entity)
        if energy_data:
            points = sorted(energy_data, key=lambda p: p.get("timestamp", ""))
            cumulative = 0.0
            data_points = []
            for p in points:
                raw = float(p.get("power_kw", 0))
                cumulative += raw * 0.25  # 15-min intervals → kWh
                data_points.append({
                    "x": str(p.get("timestamp", "")),
                    f"{series_id}_raw": round(raw, 2),
                    f"{series_id}_cumulative": round(cumulative, 2),
                })
            label = energy_data[0].get("meter_name", entity or "Energy") if energy_data else label
            unit = "kWh"
        else:
            # Fallback: generate synthetic cumulative data from equipment metadata
            vs = self.pipeline.vector_store
            results = vs.search(EQUIPMENT_COLLECTION, f"{entity or ''} {metric}".strip() or query, n_results=1)
            meta = results[0].metadata if results else {}

            base_value = float(meta.get("health_score", meta.get("power_kw", 50)))
            unit = meta.get("unit", "kWh" if "energy" in (metric or "").lower() else "kW")
            label = meta.get("name", entity or query[:30])

            from datetime import datetime, timedelta
            import random
            now = datetime.now()
            cumulative = 0.0
            data_points = []
            for i in range(48):  # 30-min intervals over 24h
                t = now - timedelta(hours=24) + timedelta(minutes=i * 30)
                noise = random.uniform(-base_value * 0.08, base_value * 0.08)
                raw = round(base_value + noise, 2)
                cumulative += raw * 0.5  # 30-min interval
                data_points.append({
                    "x": t.isoformat(),
                    f"{series_id}_raw": raw,
                    f"{series_id}_cumulative": round(cumulative, 2),
                })

        return {
            "config": {
                "title": label,
                "subtitle": f"Cumulative {unit}",
                "variant": "V1",
                "mode": "cumulative",
                "series": [{"id": series_id, "label": label, "unit": unit, "color": "#2563eb"}],
            },
            "data": data_points,
        }

    def _collect_multi_time_series(self, query: str, entities: list, metric: str) -> dict:
        """Collect multiple time series for multi-line trend.

        When entities are provided (and they're distinct), fetches each entity's time series.
        Otherwise, gets top 3 meters from energy_readings SQL directly.
        """
        series = []

        # Deduplicate entities and check we have at least 2 unique ones
        unique_entities = list(dict.fromkeys(entities)) if entities else []
        detected_unit = "kW"  # default, overridden by actual data
        if len(unique_entities) >= 2:
            seen_labels = set()
            for entity in unique_entities[:4]:
                ts_data = self._collect_time_series(query, [entity], metric)
                demo = ts_data.get("demoData", {})
                label = demo.get("label", entity or "Series")
                # Pick up the unit from the first series that has one
                if demo.get("unit"):
                    detected_unit = demo["unit"]
                # Skip if we already got this meter (fuzzy match collision)
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                series.append({
                    "name": label,
                    "timeSeries": demo.get("timeSeries", []),
                })
            # If dedup reduced to <2, fall through to SQL
            if len(series) >= 2:
                return {
                    "demoData": {
                        "label": f"{metric or 'Multi-Metric'}".replace("_", " ").title(),
                        "unit": detected_unit,
                        "series": series,
                    }
                }
            series = []  # reset and use SQL path below

        if not series:
            # Single entity with multiple metrics from PG, or discover related entities
            entity = unique_entities[0] if unique_entities else None
            if entity:
                resolved = resolve_entity_to_table(entity)
                if resolved:
                    table_name, prefix, default_metric, default_unit = resolved
                    # Get the requested metric + 2 related metrics for a multi-line view
                    metric_map = EQUIPMENT_METRIC_MAP.get(prefix, {})
                    metrics_to_plot = []
                    if metric and metric.lower() in metric_map:
                        col, u = metric_map[metric.lower()]
                        metrics_to_plot.append((metric.replace("_", " ").title(), col, u))
                    for alias, (col, u) in metric_map.items():
                        if alias == "default" or any(col == m[1] for m in metrics_to_plot):
                            continue
                        metrics_to_plot.append((alias.replace("_", " ").title(), col, u))
                        if len(metrics_to_plot) >= 3:
                            break
                    num_match = re.search(r'(\d+)$', table_name)
                    num = int(num_match.group(1)) if num_match else 0
                    prefix_labels = {
                        "trf": "Transformer", "dg": "DG Set", "ups": "UPS",
                        "chiller": "Chiller", "ahu": "AHU", "ct": "Cooling Tower",
                        "pump": "Pump", "compressor": "Compressor", "motor": "Motor",
                    }
                    eq_label = f"{prefix_labels.get(prefix, prefix.upper())} {num}"
                    for name, col, u in metrics_to_plot:
                        ts_data = self._query_pg_timeseries(table_name, col, hours=24, interval='1 hour')
                        if ts_data:
                            detected_unit = u
                            series.append({
                                "name": f"{eq_label} {name}",
                                "timeSeries": ts_data,
                            })

            if not series:
                # Last resort: single series from time_series collector
                ts_data = self._collect_time_series(query, unique_entities or [], metric)
                demo = ts_data.get("demoData", {})
                if demo.get("unit"):
                    detected_unit = demo["unit"]
                series.append({
                    "name": demo.get("label", "Series"),
                    "timeSeries": demo.get("timeSeries", []),
                })

        return {
            "demoData": {
                "label": f"{metric or 'Multi-Metric'}".replace("_", " ").title(),
                "unit": detected_unit,
                "series": series,
            }
        }

    def _collect_aggregation(self, query: str, entities: list, metric: str, collections: list) -> dict:
        """Collect aggregation data for distribution/composition/category-bar.

        Tries SQL energy breakdown first (groups by meter type/building),
        then falls back to vector search grouping.
        """
        # Try SQL-based energy breakdown for richer aggregation
        energy_breakdown = self._collect_aggregation_energy_sql(metric)
        if energy_breakdown:
            return energy_breakdown

        vs = self.pipeline.vector_store
        coll = collections[0] if collections else EQUIPMENT_COLLECTION

        # Search broadly for diverse equipment types
        results = vs.search(coll, query, n_results=30)
        if not results:
            return {"demoData": {"total": 0, "unit": "", "series": []}}

        # Group by equipment type and sum power_kw where available
        groups = {}
        for r in results:
            meta = r.metadata
            key = meta.get("equipment_type", meta.get("name", "Other"))
            power = float(meta.get("power_kw", 0) or 0)
            if key not in groups:
                groups[key] = {"label": key.replace("_", " ").title(), "total_power": 0, "count": 0}
            groups[key]["total_power"] += power
            groups[key]["count"] += 1

        # Use actual power values if available, otherwise fall back to counts
        has_power = any(g["total_power"] > 0 for g in groups.values())
        total = sum(g["total_power"] if has_power else g["count"] for g in groups.values())

        series = []
        for g in groups.values():
            raw = g["total_power"] if has_power else g["count"]
            pct = round((raw / total * 100), 1) if total > 0 else 0
            series.append({"label": g["label"], "value": pct})

        sorted_series = sorted(series, key=lambda s: -s["value"])
        categories = [s["label"] for s in sorted_series]
        values = [s["value"] for s in sorted_series]

        return {
            "demoData": {
                "label": metric.replace("_", " ").title() if metric else "Distribution",
                "unit": "kW" if has_power else "%",
                "total": round(total, 1) if has_power else 100.0,
                "series": [{"name": s["label"], "values": [s["value"]]} for s in sorted_series],
                "categories": categories,
                "values": values,
            }
        }

    # Human-readable labels for equipment type prefixes
    _PREFIX_LABELS = {
        "trf": "Transformers", "dg": "DG Sets", "ups": "UPS Systems",
        "chiller": "Chillers", "ahu": "AHUs", "ct": "Cooling Towers",
        "pump": "Pumps", "compressor": "Compressors", "motor": "Motors",
        "em": "Energy Meters", "lt_mcc": "MCC Panels", "lt_pcc": "PCC Panels",
        "lt_apfc": "APFC Panels", "lt_db": "Distribution Boards",
        "lt_vfd": "VFD Panels", "lt_plc": "PLC Panels", "lt_ats": "ATS Panels",
        "lt_changeover": "Changeovers", "lt_mldb": "Main LT DBs",
        "lt_smdb": "Sub-Main DBs",
    }

    def _collect_aggregation_energy_sql(self, metric: str) -> dict | None:
        """Query real PG timeseries tables for energy breakdown by equipment type.

        Dynamically discovers equipment table prefixes from ENTITY_TABLE_PREFIX_MAP,
        queries one representative table per prefix for latest power, then multiplies
        by table count to estimate category totals.
        """
        from django.db import connections
        try:
            # Build unique prefix → (metric_col, unit) map from entity map
            prefix_info = {}
            for _kw, (prefix, metric_col, unit) in ENTITY_TABLE_PREFIX_MAP.items():
                if prefix not in prefix_info:
                    prefix_info[prefix] = (metric_col, unit)

            # Discover actual tables per prefix from PG
            with connections['timeseries'].cursor() as cursor:
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name ~ '_[0-9]+$'
                    ORDER BY table_name
                """)
                all_tables = [row[0] for row in cursor.fetchall()]

            # Group tables by prefix
            import re as _re
            prefix_tables = {}
            for tbl in all_tables:
                m = _re.match(r'^(.+?)_(\d+)$', tbl)
                if m:
                    pfx = m.group(1)
                    if pfx in prefix_info:
                        prefix_tables.setdefault(pfx, []).append(tbl)

            if not prefix_tables:
                return None

            # Query latest power from one representative table per prefix
            category_data = []
            with connections['timeseries'].cursor() as cursor:
                for pfx, tables in prefix_tables.items():
                    metric_col, unit = prefix_info[pfx]
                    rep_table = tables[0]  # representative table
                    try:
                        cursor.execute(
                            f"SELECT {metric_col} FROM {rep_table} ORDER BY timestamp DESC LIMIT 1"
                        )
                        row = cursor.fetchone()
                        if row and row[0] is not None:
                            per_unit_power = float(row[0])
                            total_power = per_unit_power * len(tables)
                            label = self._PREFIX_LABELS.get(pfx, pfx.upper())
                            category_data.append({
                                "label": label,
                                "total_power": round(total_power, 1),
                                "count": len(tables),
                                "per_unit": round(per_unit_power, 1),
                            })
                    except Exception:
                        continue  # skip tables with missing columns

            if not category_data or len(category_data) < 2:
                return None

            # Sort by total power descending and compute percentages
            category_data.sort(key=lambda x: -x["total_power"])
            grand_total = sum(c["total_power"] for c in category_data)
            if grand_total <= 0:
                return None

            sorted_data = []
            for c in category_data:
                pct = round(c["total_power"] / grand_total * 100, 1)
                if pct > 0:
                    sorted_data.append({
                        "label": f"{c['label']} ({c['count']}×{c['per_unit']}kW)",
                        "value": pct,
                    })

            categories = [s["label"] for s in sorted_data]
            values = [s["value"] for s in sorted_data]

            return {
                "demoData": {
                    "label": metric.replace("_", " ").title() if metric else "Energy Distribution",
                    "unit": "%",
                    "total": 100.0,
                    "series": [{"name": s["label"], "values": [s["value"]]} for s in sorted_data],
                    "categories": categories,
                    "values": values,
                    "_data_source": "pg_timeseries:energy_breakdown",
                }
            }
        except Exception as e:
            logger.debug(f"Energy aggregation PG failed: {e}")
            return None

    def _collect_events(self, query: str, entities: list, collections: list) -> dict:
        """Collect events for timeline or eventlogstream."""
        vs = self.pipeline.vector_store
        events = []

        for coll_name in collections:
            results = vs.search(coll_name, query, n_results=5)
            for r in results:
                meta = r.metadata
                events.append({
                    "timestamp": meta.get("shift_date", meta.get("created_at", "")),
                    "type": meta.get("maintenance_type", meta.get("work_type", "info")),
                    "message": r.content[:150],
                    "source": meta.get("equipment_name", meta.get("supervisor", "System")),
                })

        # Fill in missing timestamps with sequential times
        from datetime import datetime, timedelta
        now = datetime.now()
        for i, event in enumerate(events):
            if not event.get("timestamp"):
                event["timestamp"] = (now - timedelta(hours=len(events) - i)).isoformat()

        # AUDIT FIX: Calculate range for timeline schema compliance
        # timeline requires: title, range, events
        timestamps = [e.get("timestamp", "") for e in events if e.get("timestamp")]
        if timestamps:
            sorted_ts = sorted(timestamps)
            range_start = sorted_ts[0][:10] if sorted_ts else now.strftime("%Y-%m-%d")
            range_end = sorted_ts[-1][:10] if sorted_ts else now.strftime("%Y-%m-%d")
        else:
            range_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
            range_end = now.strftime("%Y-%m-%d")

        return {
            "demoData": {
                "title": "Event Log",
                "range": {"start": range_start, "end": range_end},
                "events": events[:20],
            }
        }

    def _collect_device_detail(self, query: str, entities: list) -> dict:
        """Collect detailed info for a single device (edgedevicepanel).

        Enriches ChromaDB metadata with live PostgreSQL timeseries readings.
        """
        vs = self.pipeline.vector_store
        entity = entities[0] if entities else ""

        # Equipment info from ChromaDB
        eq_results = vs.search(EQUIPMENT_COLLECTION, entity or query, n_results=1)
        alert_results = vs.search(ALERTS_COLLECTION, entity or query, n_results=3)
        maint_results = vs.search(MAINTENANCE_COLLECTION, entity or query, n_results=3)

        device = {}
        if eq_results:
            meta = eq_results[0].metadata
            device = {
                "name": meta.get("name", entity),
                "id": meta.get("equipment_id", ""),
                "type": meta.get("equipment_type", ""),
                "status": meta.get("status", "unknown"),
                "health": meta.get("health_score", 0),
                "location": meta.get("location", ""),
                "criticality": meta.get("criticality", ""),
            }

        # ── PostgreSQL timeseries live readings ──
        readings = []
        resolved = resolve_entity_to_table(entity)
        if resolved:
            table_name, prefix, default_metric, default_unit = resolved
            # Get key metrics for this equipment type
            metric_map = EQUIPMENT_METRIC_MAP.get(prefix, {})
            metric_cols = [col for key, (col, unit) in metric_map.items() if key != "default"][:8]
            if metric_cols:
                live_data = self._query_pg_multi_metric_latest(table_name, metric_cols)
                if live_data:
                    device["_data_source"] = f"pg_timeseries:{table_name}"
                    for col, val in live_data.items():
                        if val is not None:
                            # Find unit from metric map
                            unit = ""
                            for key, (mc, u) in metric_map.items():
                                if mc == col:
                                    unit = u
                                    break
                            readings.append({
                                "parameter": col.replace("_", " ").title(),
                                "value": val,
                                "unit": unit,
                            })
                    logger.info(f"PG DeviceDetail: {table_name} → {len(readings)} live readings")

        # Fallback to ChromaDB metadata readings if PG didn't work
        if not readings and eq_results:
            meta = eq_results[0].metadata
            reading_keys = {
                "power_kw": "kW", "voltage": "V", "current": "A",
                "temperature": "°C", "pressure": "bar", "health_score": "%",
                "load_percent": "%", "efficiency": "%",
            }
            for key, unit in reading_keys.items():
                if key in meta:
                    readings.append({
                        "parameter": key.replace("_", " ").title(),
                        "value": meta[key],
                        "unit": unit,
                    })

        demo = {
            "device": device,
            "readings": readings,
            "alerts": [{"message": r.content[:100], "severity": r.metadata.get("severity", "info")} for r in alert_results],
            "maintenance": [{"description": r.content[:100], "type": r.metadata.get("maintenance_type", "")} for r in maint_results],
        }
        # Propagate PG source to demoData level for provenance stamping
        if device.get("_data_source"):
            demo["_data_source"] = device["_data_source"]
        return {"demoData": demo}

    def _collect_matrix(self, query: str, entities: list, collections: list) -> dict:
        """Collect data for matrix-heatmap."""
        vs = self.pipeline.vector_store
        results = vs.search(EQUIPMENT_COLLECTION, query, n_results=20)

        if not results:
            return {"demoData": {"label": "Health Matrix", "rows": [], "cols": [], "dataset": []}}

        # Build a simple health matrix grouped by type
        by_type = {}
        for r in results:
            meta = r.metadata
            eq_type = meta.get("equipment_type", "other").replace("_", " ").title()
            name = meta.get("name", "")
            health = meta.get("health_score", 0)
            if eq_type not in by_type:
                by_type[eq_type] = []
            by_type[eq_type].append({"name": name, "health": health})

        rows = list(by_type.keys())[:8]
        # Use first N items of each type as columns
        max_cols = max((len(v) for v in by_type.values()), default=0)
        max_cols = min(max_cols, 6)
        cols = [f"Unit {i+1}" for i in range(max_cols)]

        dataset = []
        for row_type in rows:
            items = by_type[row_type]
            row_vals = [(items[i]["health"] / 100.0 if i < len(items) else 0) for i in range(max_cols)]
            dataset.append(row_vals)

        return {
            "demoData": {
                "label": "Equipment Health Matrix",
                "rows": rows,
                "cols": cols,
                "dataset": dataset,
            }
        }

    def _collect_flow(self, query: str, entities: list) -> dict:
        """Collect data for sankey flow diagram."""
        vs = self.pipeline.vector_store
        results = vs.search(EQUIPMENT_COLLECTION, query, n_results=15)

        # Build simplified flow: group equipment by type, create source→consumer links
        nodes = [{"id": "grid", "label": "Grid Supply"}]
        links = []
        seen_types = set()

        for r in results:
            meta = r.metadata
            eq_type = meta.get("equipment_type", "other")
            if eq_type not in seen_types:
                seen_types.add(eq_type)
                node_id = eq_type.replace("_", "-")
                nodes.append({"id": node_id, "label": eq_type.replace("_", " ").title()})
                # Estimate consumption from health score (placeholder)
                links.append({"source": "grid", "target": node_id, "value": meta.get("health_score", 50)})

        return {
            "demoData": {
                "label": "Energy Flow",
                "unit": "kW",
                "nodes": nodes,
                "links": links,
            }
        }

    def _collect_people(self, query: str) -> dict:
        """Collect people/workforce data."""
        vs = self.pipeline.vector_store
        try:
            results = vs.search(SHIFT_LOGS_COLLECTION, query, n_results=5)
        except Exception:
            results = []

        roster = []
        for r in results:
            meta = r.metadata
            roster.append({
                "name": meta.get("supervisor", ""),
                "shift": meta.get("shift_name", ""),
                "date": meta.get("shift_date", ""),
            })

        return {"demoData": {"roster": roster, "shifts": []}}

    def _collect_supply(self, query: str) -> dict:
        """Collect supply chain data."""
        vs = self.pipeline.vector_store
        try:
            results = vs.search(WORK_ORDERS_COLLECTION, query, n_results=5)
        except Exception:
            results = []

        locations = []
        for r in results:
            meta = r.metadata
            if meta.get("vendor"):
                locations.append({"name": meta["vendor"], "type": "vendor"})

        return {"demoData": {"locations": locations, "routes": []}}

    def _collect_generic(self, query: str, collections: list) -> dict:
        """Generic collection — search and return raw context."""
        vs = self.pipeline.vector_store
        all_results = vs.search_multiple_collections(
            query, collections, n_results_per=3,
        )
        context = "\n".join(r.content[:200] for r in all_results[:5])
        return {"demoData": {"context": context}}

    # ── Helpers ──

    def _extract_metric_from_doc(self, doc: RAGSearchResult, metric: str) -> tuple:
        """Extract a metric value, unit, and state from a RAG result."""
        meta = doc.metadata
        content = doc.content

        # Try metadata fields first
        health = meta.get("health_score")
        if health is not None:
            state = "critical" if health < 50 else "warning" if health < 75 else "normal"
            if not metric or "health" in metric:
                return (health, "%", state)

        # Try to parse from content: "Load: 75.3%" or "Capacity: 500 kVA"
        if metric:
            pattern = rf'{metric}[:\s]+(\d+(?:\.\d+)?)\s*(%|kW|kVA|kWh|°C|TR|A|V|Hz)?'
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                val = match.group(1)
                unit = match.group(2) or ""
                return (val, unit, "normal")

        # Fallback to health score
        if health is not None:
            return (health, "%", "normal")

        return ("N/A", "", "normal")

    def _parse_alert_evidence(self, content: str) -> dict:
        """Parse alert evidence fields from content string."""
        evidence = {}
        for segment in content.split("|"):
            segment = segment.strip()
            if segment.startswith("Value:"):
                parts = segment.replace("Value:", "").strip().split()
                if parts:
                    evidence["value"] = parts[0]
                    evidence["unit"] = parts[1] if len(parts) > 1 else ""
            elif segment.startswith("Parameter:"):
                evidence["label"] = segment.replace("Parameter:", "").strip()
            elif segment.startswith("Threshold:"):
                evidence["threshold"] = segment.replace("Threshold:", "").strip()
        return evidence
