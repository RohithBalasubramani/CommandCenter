"""
Focus Graph Builder — Builds and evolves the semantic focus graph.

Ingests parsed intents and RAG results to create/update nodes and edges.
Handles pronoun resolution and comparison target merging.
"""

import re
import logging
from typing import Optional

from layer2.focus_graph import (
    SemanticFocusGraph, FocusNode, FocusEdge,
    NodeType, EdgeType,
)

logger = logging.getLogger(__name__)

# Metric keywords → NodeType.METRIC label mapping
METRIC_KEYWORDS = {
    "vibration": "vibration",
    "temperature": "temperature",
    "temp": "temperature",
    "power": "power",
    "energy": "energy",
    "voltage": "voltage",
    "current": "current",
    "pressure": "pressure",
    "flow": "flow_rate",
    "load": "load",
    "efficiency": "efficiency",
    "speed": "speed",
    "rpm": "rpm",
    "oil": "oil_temperature",
    "winding": "winding_temperature",
    "bearing": "bearing_temperature",
}


class FocusGraphBuilder:
    """Builds and evolves the semantic focus graph from parsed intents and RAG results."""

    def __init__(self, graph: SemanticFocusGraph):
        self.graph = graph

    def ingest_intent(self, intent) -> list[str]:
        """
        Extract entities from ParsedIntent and add as nodes.
        Returns list of newly created node IDs.
        """
        new_ids = []

        # 1. Devices → EQUIPMENT nodes
        devices = intent.entities.get("devices", [])
        for device in devices:
            node_id = f"equipment:{device}"
            node = FocusNode(
                id=node_id,
                type=NodeType.EQUIPMENT,
                label=self._humanize_equipment(device),
                properties={"equipment_id": device},
                source="user_query",
            )
            self.graph.add_node(node)
            new_ids.append(node_id)

        # 2. Detect metric from query text and characteristics
        metric_label = self._detect_metric(intent.raw_text, intent.primary_characteristic)
        if metric_label:
            metric_id = f"metric:{metric_label}"
            metric_node = FocusNode(
                id=metric_id,
                type=NodeType.METRIC,
                label=metric_label,
                source="user_query",
            )
            self.graph.add_node(metric_node)
            new_ids.append(metric_id)

            # Link equipment → metric
            for device in devices:
                equip_id = f"equipment:{device}"
                try:
                    self.graph.add_edge(
                        equip_id, metric_id, EdgeType.MEASURED_BY,
                        confidence=0.9, evidence=f"User asked about {metric_label} for {device}",
                    )
                except ValueError:
                    pass

        # 3. Time references → TIME_RANGE nodes
        time_refs = intent.entities.get("time", [])
        for tref in time_refs[:2]:
            time_id = f"time_range:{tref.replace(' ', '_')}"
            time_node = FocusNode(
                id=time_id,
                type=NodeType.TIME_RANGE,
                label=tref,
                source="user_query",
            )
            self.graph.add_node(time_node)
            new_ids.append(time_id)

        # 4. Maintenance characteristic → TASK placeholder
        if intent.primary_characteristic == "maintenance":
            task_id = "task:maintenance_inquiry"
            task_node = FocusNode(
                id=task_id,
                type=NodeType.TASK,
                label="Maintenance Inquiry",
                source="user_query",
                confidence=0.7,
            )
            self.graph.add_node(task_node)
            new_ids.append(task_id)

        # 5. Alerts characteristic → ANOMALY placeholder
        if intent.primary_characteristic == "alerts":
            anomaly_id = "anomaly:alert_inquiry"
            anomaly_node = FocusNode(
                id=anomaly_id,
                type=NodeType.ANOMALY,
                label="Alert Investigation",
                source="user_query",
                confidence=0.6,
            )
            self.graph.add_node(anomaly_node)
            new_ids.append(anomaly_id)

        return new_ids

    def ingest_rag_results(self, rag_results: list) -> list[str]:
        """
        Extract relationships from RAG results.
        """
        new_ids = []

        for result in rag_results:
            if not isinstance(result, dict):
                # Handle dataclass RAG results
                if hasattr(result, '__dict__'):
                    result = result.__dict__
                else:
                    continue

            domain = result.get("domain", "")
            data = result.get("data", {})
            success = result.get("success", False)

            if not success or not data:
                continue

            # Alert data → ANOMALY nodes
            if domain == "alerts":
                alerts = data.get("alerts", [])
                if isinstance(alerts, list):
                    for alert in alerts[:5]:
                        if isinstance(alert, dict):
                            alert_id = f"anomaly:{alert.get('id', 'unknown')}"
                            alert_node = FocusNode(
                                id=alert_id,
                                type=NodeType.ANOMALY,
                                label=alert.get("description", alert.get("message", "Alert")),
                                properties=alert,
                                source="rag_inferred",
                                confidence=0.8,
                            )
                            self.graph.add_node(alert_node)
                            new_ids.append(alert_id)

                            # Link to equipment if available
                            equip = alert.get("equipment", alert.get("device", ""))
                            if equip:
                                equip_id = f"equipment:{equip}"
                                if equip_id in self.graph.nodes:
                                    try:
                                        self.graph.add_edge(
                                            alert_id, equip_id, EdgeType.AFFECTS,
                                            confidence=0.85, evidence=f"Alert on {equip}",
                                        )
                                    except ValueError:
                                        pass

            # Maintenance data → TASK nodes
            if domain == "maintenance" or "maintenance" in str(data.get("type", "")):
                records = data.get("records", data.get("work_orders", []))
                if isinstance(records, list):
                    for record in records[:3]:
                        if isinstance(record, dict):
                            task_id = f"task:{record.get('id', 'wo_unknown')}"
                            task_node = FocusNode(
                                id=task_id,
                                type=NodeType.TASK,
                                label=record.get("description", "Work Order"),
                                properties=record,
                                source="rag_inferred",
                                confidence=0.75,
                            )
                            self.graph.add_node(task_node)
                            new_ids.append(task_id)

        return new_ids

    def resolve_pronoun(self, text: str) -> Optional[FocusNode]:
        """
        Resolve pronouns ("it", "this", "the same one") against the graph.

        Resolution order:
        1. root_node (primary focus equipment)
        2. Most recently referenced equipment node
        3. Most recently referenced anomaly
        4. None (unresolvable)
        """
        pronoun_patterns = [
            r'\b(it|its|this|that|the same|the same one|this one|that one)\b',
            r'\b(the pump|the motor|the device|the equipment|the machine)\b',
        ]
        text_lower = text.lower()
        has_pronoun = any(re.search(p, text_lower) for p in pronoun_patterns)

        if not has_pronoun:
            return None

        # 1. Root node
        root = self.graph.get_root_equipment()
        if root:
            return root

        # 2. Most recently referenced equipment
        equipment = self.graph.get_all_equipment()
        if equipment:
            return equipment[0]  # Already sorted by reference_count desc

        # 3. Most recently referenced anomaly
        anomalies = self.graph.get_active_anomalies()
        if anomalies:
            return max(anomalies, key=lambda n: n.last_referenced)

        return None

    def merge_comparison_target(self, new_equipment: str) -> str:
        """
        When user says "compare with pump 5", add pump_005 node
        and COMPARED_WITH edge to current root.
        """
        new_id = f"equipment:{new_equipment}"
        new_node = FocusNode(
            id=new_id,
            type=NodeType.EQUIPMENT,
            label=self._humanize_equipment(new_equipment),
            properties={"equipment_id": new_equipment},
            source="user_query",
        )
        self.graph.add_node(new_node)

        # Link with COMPARED_WITH to root
        root = self.graph.get_root_equipment()
        if root:
            try:
                self.graph.add_edge(
                    root.id, new_id, EdgeType.COMPARED_WITH,
                    confidence=0.95, evidence="User requested comparison",
                )
            except ValueError:
                pass

        return new_id

    def _humanize_equipment(self, equipment_id: str) -> str:
        """Convert equipment_id like 'pump_004' to 'Pump 4'."""
        match = re.match(r'([a-zA-Z]+)[_\s]*0*(\d+)', equipment_id)
        if match:
            name = match.group(1).capitalize()
            number = match.group(2)
            return f"{name} {number}"
        return equipment_id.replace("_", " ").title()

    def _detect_metric(self, text: str, primary_characteristic: str = None) -> Optional[str]:
        """Detect metric name from query text."""
        text_lower = text.lower()

        # Check for explicit metric keywords
        for keyword, metric_label in METRIC_KEYWORDS.items():
            if keyword in text_lower:
                return metric_label

        # Infer from characteristic
        char_metric_map = {
            "energy": "power",
            "power_quality": "power_quality",
            "hvac": "temperature",
        }
        if primary_characteristic and primary_characteristic in char_metric_map:
            return char_metric_map[primary_characteristic]

        return None
