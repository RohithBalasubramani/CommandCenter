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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    @property
    def pipeline(self) -> IndustrialRAGPipeline:
        if self._pipeline is None:
            self._pipeline = get_rag_pipeline()
        return self._pipeline

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

    def _collect_single_metric(self, query: str, entities: list, metric: str, collections: list) -> dict:
        """Collect a single metric for KPI widget.

        Uses entity/metric from data_request when available, and skips
        entities already shown in other KPIs to ensure diversity.
        """
        vs = self.pipeline.vector_store
        entity = entities[0] if entities else ""

        # Map common metric aliases to SQL column names
        metric_aliases = {
            "power": "power_kw", "consumption": "power_kw", "energy": "power_kw",
            "load": "power_kw", "demand": "max_demand_kw",
            "power_factor": "power_factor", "pf": "power_factor",
            "voltage": "voltage_avg", "frequency": "frequency",
        }
        resolved_metric = metric_aliases.get((metric or "").lower().strip(), metric)

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
        return {"demoData": {"label": entity or "Unknown", "value": "N/A", "unit": "",
                             "_synthetic": True, "_data_source": "no_data_fallback"}}

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
        """Collect comparison data for two entities."""
        vs = self.pipeline.vector_store
        coll = collections[0] if collections else EQUIPMENT_COLLECTION

        # Need at least 2 entities
        entity_a = entities[0] if len(entities) > 0 else ""
        entity_b = entities[1] if len(entities) > 1 else ""

        result_a = vs.search(coll, f"{entity_a} {metric}".strip() or query, n_results=1)
        result_b = vs.search(coll, f"{entity_b} {metric}".strip() or query, n_results=1) if entity_b else []

        val_a, unit_a, _ = self._extract_metric_from_doc(result_a[0], metric) if result_a else ("N/A", "", "normal")
        val_b, unit_b, _ = self._extract_metric_from_doc(result_b[0], metric) if result_b else ("N/A", "", "normal")

        # Calculate delta
        try:
            delta = float(val_a) - float(val_b)
            delta_pct = (delta / float(val_b) * 100) if float(val_b) != 0 else 0
        except (ValueError, TypeError):
            delta = 0
            delta_pct = 0

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
        """Collect time series data — first try SQL energy readings, then fallback to vector."""
        entity = entities[0] if entities else None

        # Try energy SQL for time series (actual readings with timestamps)
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

        # Fallback: search equipment docs for the entity and generate synthetic time series
        vs = self.pipeline.vector_store
        results = vs.search(EQUIPMENT_COLLECTION, f"{entity or ''} {metric}".strip() or query, n_results=1)
        meta = results[0].metadata if results else {}

        # Generate 24h of synthetic data from base value (health_score or power_kw)
        base_value = float(meta.get("health_score", meta.get("power_kw", 50)))
        unit = meta.get("unit", "kW" if "power" in (metric or "").lower() else "%")
        label = meta.get("name", entity or query[:30])

        from datetime import datetime, timedelta
        import random
        now = datetime.now()
        time_series = []
        for i in range(48):  # 30-min intervals over 24h
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
                "_synthetic": True,  # GROUNDING AUDIT: marks data as synthetically generated
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
        if len(unique_entities) >= 2:
            seen_labels = set()
            for entity in unique_entities[:4]:
                ts_data = self._collect_time_series(query, [entity], metric)
                demo = ts_data.get("demoData", {})
                label = demo.get("label", entity or "Series")
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
                        "unit": "kW",
                        "series": series,
                    }
                }
            series = []  # reset and use SQL path below

        if not series:
            # Fetch all data for top 3 meters in one query (avoids cursor issues)
            from django.db import connection
            try:
                with connection.cursor() as c:
                    c.execute("""
                        SELECT e.meter_id, e.meter_name, e.timestamp, e.power_kw
                        FROM energy_readings e
                        INNER JOIN (
                            SELECT meter_id, MAX(energy_kwh_cumulative) as max_kwh
                            FROM energy_readings
                            GROUP BY meter_id
                            ORDER BY max_kwh DESC
                            LIMIT 3
                        ) top ON e.meter_id = top.meter_id
                        ORDER BY e.meter_id, e.timestamp ASC
                    """)
                    rows = c.fetchall()

                    # Group by meter
                    from collections import OrderedDict
                    meter_data = OrderedDict()
                    for meter_id, meter_name, ts, power_kw in rows:
                        if meter_id not in meter_data:
                            meter_data[meter_id] = {"name": meter_name, "points": []}
                        meter_data[meter_id]["points"].append({
                            "time": str(ts), "value": power_kw or 0
                        })

                    for mid, mdata in meter_data.items():
                        series.append({
                            "name": mdata["name"] or mid,
                            "timeSeries": mdata["points"],
                        })
            except Exception as e:
                logger.warning(f"Multi time series SQL failed: {e}")

            if not series:
                # Fallback: single series from time_series collector
                ts_data = self._collect_time_series(query, [], metric)
                demo = ts_data.get("demoData", {})
                series.append({
                    "name": demo.get("label", "Series"),
                    "timeSeries": demo.get("timeSeries", []),
                })

        return {
            "demoData": {
                "label": f"{metric or 'Multi-Metric'}".replace("_", " ").title(),
                "unit": "kW",
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

    def _collect_aggregation_energy_sql(self, metric: str) -> dict | None:
        """Try SQL-based energy breakdown grouped by meter type or building."""
        from django.db import connection
        try:
            with connection.cursor() as c:
                # Get latest reading per meter, grouped by meter type prefix
                c.execute("""
                    SELECT
                        CASE
                            WHEN meter_id LIKE 'EM-INC%%' THEN 'Main Incomer'
                            WHEN meter_id LIKE 'EM-SUB%%' THEN 'Sub Meter'
                            WHEN meter_id LIKE 'EM-DG%%' THEN 'DG Set'
                            WHEN meter_id LIKE 'EM-SOL%%' THEN 'Solar'
                            WHEN meter_id LIKE 'EM-UPS%%' THEN 'UPS'
                            ELSE 'Other'
                        END as category,
                        ROUND(AVG(power_kw), 1) as avg_power,
                        COUNT(DISTINCT meter_id) as meter_count
                    FROM energy_readings
                    WHERE id IN (
                        SELECT MAX(id) FROM energy_readings GROUP BY meter_id
                    )
                    GROUP BY category
                    ORDER BY avg_power DESC
                """)
                rows = c.fetchall()
                if not rows or len(rows) < 2:
                    return None

                total = sum(float(r[1] or 0) for r in rows)
                if total <= 0:
                    return None

                sorted_data = [
                    {"label": r[0], "value": round(float(r[1] or 0) / total * 100, 1)}
                    for r in rows if float(r[1] or 0) > 0
                ]
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
                    }
                }
        except Exception as e:
            logger.debug(f"Energy aggregation SQL failed: {e}")
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
        """Collect detailed info for a single device (edgedevicepanel)."""
        vs = self.pipeline.vector_store
        entity = entities[0] if entities else ""

        # Equipment info
        eq_results = vs.search(EQUIPMENT_COLLECTION, entity or query, n_results=1)
        # Related alerts
        alert_results = vs.search(ALERTS_COLLECTION, entity or query, n_results=3)
        # Related maintenance
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

        # Extract numerical metadata as sensor readings
        readings = []
        if eq_results:
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

        return {
            "demoData": {
                "device": device,
                "readings": readings,
                "alerts": [{"message": r.content[:100], "severity": r.metadata.get("severity", "info")} for r in alert_results],
                "maintenance": [{"description": r.content[:100], "type": r.metadata.get("maintenance_type", "")} for r in maint_results],
            }
        }

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
