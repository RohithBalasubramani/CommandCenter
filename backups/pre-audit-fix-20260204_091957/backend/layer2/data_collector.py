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
from layer2.widget_schemas import WIDGET_SCHEMAS
from layer2.widget_selector import WidgetPlanItem

logger = logging.getLogger(__name__)

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

        Returns list of dicts with:
            scenario, size, data_override, relevance, why
        """
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

                results[idx] = {
                    "scenario": widget.scenario,
                    "size": widget.size,
                    "relevance": widget.relevance,
                    "why": widget.why,
                    "data_override": data_override,
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
        metric = data_request.get("metric", "")

        # Choose collection to search
        collections = data_request.get("collections", schema.get("default_collections", ["equipment"]))
        collection_names = [COLLECTION_MAP.get(c, c) for c in collections if c in COLLECTION_MAP]

        # No-data widgets (helpview, pulseview, chatstream)
        if strategy == "none":
            return schema.get("demo_shape", {})

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

    def _collect_single_metric(self, query: str, entities: list, metric: str, collections: list) -> dict:
        """Collect a single metric for KPI widget."""
        vs = self.pipeline.vector_store
        entity = entities[0] if entities else ""
        search_q = f"{entity} {metric}".strip() or query

        results = vs.search(
            collections[0] if collections else EQUIPMENT_COLLECTION,
            search_q,
            n_results=1,
        )

        if not results:
            return {"demoData": {"label": entity or "Unknown", "value": "N/A", "unit": ""}}

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
        """Collect multiple time series for multi-line trend."""
        series = []
        for entity in (entities or [""])[:4]:
            ts_data = self._collect_time_series(query, [entity] if entity else [], metric)
            demo = ts_data.get("demoData", {})
            series.append({
                "name": demo.get("label", entity or "Series"),
                "timeSeries": demo.get("timeSeries", []),
            })

        return {
            "demoData": {
                "label": f"{metric or 'Multi-Metric'}".replace("_", " ").title(),
                "unit": series[0].get("unit", "units") if series else "units",
                "series": series,
            }
        }

    def _collect_aggregation(self, query: str, entities: list, metric: str, collections: list) -> dict:
        """Collect aggregation data for distribution/composition/category-bar."""
        vs = self.pipeline.vector_store
        coll = collections[0] if collections else EQUIPMENT_COLLECTION

        results = vs.search(coll, query, n_results=10)
        if not results:
            return {"demoData": {"total": 0, "unit": "", "series": []}}

        # Group by equipment type or name and aggregate
        groups = {}
        for r in results:
            meta = r.metadata
            key = meta.get("equipment_type", meta.get("name", "Other"))
            if key not in groups:
                groups[key] = {"label": key.replace("_", " ").title(), "value": 0, "count": 0}
            # Try to extract a numeric value
            health = meta.get("health_score", 0)
            groups[key]["value"] += health
            groups[key]["count"] += 1

        series = []
        total = 0
        for g in groups.values():
            avg = round(g["value"] / g["count"], 1) if g["count"] else 0
            series.append({"label": g["label"], "value": avg})
            total += avg

        return {
            "demoData": {
                "total": round(total, 1),
                "unit": "%",
                "series": sorted(series, key=lambda s: -s["value"]),
            }
        }

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

        return {
            "demoData": {
                "title": "Event Log",
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
