"""
Tests for Upgrade 1: Persistent Semantic Focus Graph

Test IDs: FG-B01 through FG-B08
"""

import pytest
import time
from layer2.focus_graph import (
    SemanticFocusGraph, FocusNode, FocusEdge,
    NodeType, EdgeType,
)
from layer2.focus_graph_builder import FocusGraphBuilder
from layer2.intent_parser import ParsedIntent


class TestFocusGraphDataStructure:
    """FG-B01: Graph CRUD operations."""

    def test_add_node_sets_root_equipment(self):
        """First equipment node becomes root."""
        graph = SemanticFocusGraph(session_id="test-1")
        node = FocusNode(id="equipment:pump_004", type=NodeType.EQUIPMENT, label="Pump 4")
        graph.add_node(node)
        assert graph.root_node_id == "equipment:pump_004"
        assert len(graph.nodes) == 1
        assert graph.version == 1

    def test_add_duplicate_node_increments_reference_count(self):
        """Adding same node again increments reference_count."""
        graph = SemanticFocusGraph(session_id="test-2")
        node = FocusNode(id="equipment:pump_004", type=NodeType.EQUIPMENT, label="Pump 4")
        graph.add_node(node)
        graph.add_node(node)
        assert graph.nodes["equipment:pump_004"].reference_count == 2
        assert graph.version == 1  # version only increments on new nodes

    def test_add_edge_between_existing_nodes(self):
        """Edge creation between two existing nodes."""
        graph = SemanticFocusGraph(session_id="test-3")
        graph.add_node(FocusNode(id="equipment:pump_004", type=NodeType.EQUIPMENT, label="Pump 4"))
        graph.add_node(FocusNode(id="metric:vibration", type=NodeType.METRIC, label="vibration"))
        edge_id = graph.add_edge(
            "equipment:pump_004", "metric:vibration", EdgeType.MEASURED_BY,
            confidence=0.9, evidence="User asked about vibration for pump 4",
        )
        assert edge_id != ""
        assert len(graph.edges) == 1
        assert graph.edges[0].type == EdgeType.MEASURED_BY

    def test_add_edge_nonexistent_node_raises(self):
        """Edge creation with nonexistent node raises ValueError."""
        graph = SemanticFocusGraph(session_id="test-4")
        graph.add_node(FocusNode(id="equipment:pump_004", type=NodeType.EQUIPMENT, label="Pump 4"))
        with pytest.raises(ValueError):
            graph.add_edge("equipment:pump_004", "metric:missing", EdgeType.MEASURED_BY)

    def test_add_edge_deduplicates(self):
        """Adding same edge twice updates confidence, doesn't duplicate."""
        graph = SemanticFocusGraph(session_id="test-5")
        graph.add_node(FocusNode(id="a", type=NodeType.EQUIPMENT, label="A"))
        graph.add_node(FocusNode(id="b", type=NodeType.METRIC, label="B"))
        graph.add_edge("a", "b", EdgeType.MEASURED_BY, confidence=0.5)
        graph.add_edge("a", "b", EdgeType.MEASURED_BY, confidence=0.9)
        assert len(graph.edges) == 1
        assert graph.edges[0].confidence == 0.9

    def test_get_neighbors(self):
        """Get neighbors returns connected nodes."""
        graph = SemanticFocusGraph(session_id="test-6")
        graph.add_node(FocusNode(id="a", type=NodeType.EQUIPMENT, label="A"))
        graph.add_node(FocusNode(id="b", type=NodeType.METRIC, label="B"))
        graph.add_node(FocusNode(id="c", type=NodeType.METRIC, label="C"))
        graph.add_edge("a", "b", EdgeType.MEASURED_BY)
        graph.add_edge("a", "c", EdgeType.MEASURED_BY)
        neighbors = graph.get_neighbors("a")
        assert len(neighbors) == 2
        # Filter by edge type
        neighbors_measured = graph.get_neighbors("a", EdgeType.MEASURED_BY)
        assert len(neighbors_measured) == 2

    def test_cycle_detection_for_causal_edges(self):
        """Reject edges that would create causal cycles."""
        graph = SemanticFocusGraph(session_id="test-7")
        graph.add_node(FocusNode(id="a", type=NodeType.ANOMALY, label="A"))
        graph.add_node(FocusNode(id="b", type=NodeType.ANOMALY, label="B"))
        graph.add_edge("a", "b", EdgeType.CAUSED_BY)
        # Adding b→a would create cycle
        edge_id = graph.add_edge("b", "a", EdgeType.CAUSED_BY)
        assert edge_id == ""  # Rejected


class TestFocusGraphSerialization:
    """FG-B02: Serialization and deserialization."""

    def test_to_dict_and_from_dict_roundtrip(self):
        """Graph survives JSON serialization roundtrip."""
        graph = SemanticFocusGraph(session_id="test-roundtrip")
        graph.add_node(FocusNode(id="equipment:pump_004", type=NodeType.EQUIPMENT,
                                  label="Pump 4", properties={"equipment_id": "pump_004"}))
        graph.add_node(FocusNode(id="metric:vibration", type=NodeType.METRIC, label="vibration"))
        graph.add_edge("equipment:pump_004", "metric:vibration", EdgeType.MEASURED_BY)

        data = graph.to_dict()
        restored = SemanticFocusGraph.from_dict(data)

        assert restored.session_id == "test-roundtrip"
        assert len(restored.nodes) == 2
        assert len(restored.edges) == 1
        assert restored.root_node_id == "equipment:pump_004"
        assert restored.nodes["equipment:pump_004"].label == "Pump 4"

    def test_from_dict_with_corrupt_data_raises(self):
        """Corrupt JSON raises during deserialization."""
        with pytest.raises(Exception):
            SemanticFocusGraph.from_dict({"session_id": "bad", "nodes": {"x": {"type": "invalid_type"}}})

    def test_to_prompt_context_produces_text(self):
        """to_prompt_context produces non-empty text for LLM injection."""
        graph = SemanticFocusGraph(session_id="test-prompt")
        graph.add_node(FocusNode(id="equipment:pump_004", type=NodeType.EQUIPMENT, label="Pump 4"))
        graph.add_node(FocusNode(id="metric:vibration", type=NodeType.METRIC, label="vibration"))
        graph.add_edge("equipment:pump_004", "metric:vibration", EdgeType.MEASURED_BY)
        ctx = graph.to_prompt_context()
        assert "Pump 4" in ctx
        assert "vibration" in ctx
        assert "measured_by" in ctx


class TestFocusGraphBuilder:
    """FG-B03 through FG-B06: Builder operations."""

    def test_ingest_intent_creates_equipment_nodes(self):
        """FG-B03: Intent with devices creates equipment nodes."""
        graph = SemanticFocusGraph(session_id="test-builder-1")
        builder = FocusGraphBuilder(graph)
        intent = ParsedIntent(
            type="query",
            domains=["industrial"],
            entities={"devices": ["pump_004"]},
            raw_text="show pump 4 vibration",
            primary_characteristic="energy",
        )
        new_ids = builder.ingest_intent(intent)
        assert "equipment:pump_004" in new_ids
        assert "equipment:pump_004" in graph.nodes
        assert graph.root_node_id == "equipment:pump_004"

    def test_ingest_intent_creates_metric_nodes(self):
        """FG-B04: Intent with metric keyword creates metric + edge."""
        graph = SemanticFocusGraph(session_id="test-builder-2")
        builder = FocusGraphBuilder(graph)
        intent = ParsedIntent(
            type="query",
            domains=["industrial"],
            entities={"devices": ["pump_004"]},
            raw_text="show pump 4 vibration trend",
        )
        new_ids = builder.ingest_intent(intent)
        assert "metric:vibration" in new_ids
        # Should have MEASURED_BY edge
        assert len(graph.edges) == 1
        assert graph.edges[0].type == EdgeType.MEASURED_BY

    def test_ingest_intent_creates_time_range_nodes(self):
        """FG-B05: Intent with time references creates time range nodes."""
        graph = SemanticFocusGraph(session_id="test-builder-3")
        builder = FocusGraphBuilder(graph)
        intent = ParsedIntent(
            type="query",
            entities={"time": ["last 24 hours"]},
            raw_text="show data from last 24 hours",
        )
        new_ids = builder.ingest_intent(intent)
        assert any("time_range:" in nid for nid in new_ids)

    def test_resolve_pronoun_returns_root(self):
        """FG-B06: Pronoun resolution returns root equipment."""
        graph = SemanticFocusGraph(session_id="test-resolve")
        graph.add_node(FocusNode(id="equipment:pump_004", type=NodeType.EQUIPMENT, label="Pump 4"))
        builder = FocusGraphBuilder(graph)

        resolved = builder.resolve_pronoun("is it running normally?")
        assert resolved is not None
        assert resolved.id == "equipment:pump_004"

    def test_resolve_pronoun_no_pronoun_returns_none(self):
        """No pronoun in text returns None."""
        graph = SemanticFocusGraph(session_id="test-no-pronoun")
        graph.add_node(FocusNode(id="equipment:pump_004", type=NodeType.EQUIPMENT, label="Pump 4"))
        builder = FocusGraphBuilder(graph)
        resolved = builder.resolve_pronoun("show pump 5 vibration")
        assert resolved is None

    def test_merge_comparison_target_adds_compared_with_edge(self):
        """FG-B07: Comparison target creates node + COMPARED_WITH edge."""
        graph = SemanticFocusGraph(session_id="test-compare")
        graph.add_node(FocusNode(id="equipment:pump_004", type=NodeType.EQUIPMENT, label="Pump 4"))
        builder = FocusGraphBuilder(graph)

        new_id = builder.merge_comparison_target("pump_005")
        assert new_id == "equipment:pump_005"
        assert "equipment:pump_005" in graph.nodes
        assert len(graph.edges) == 1
        assert graph.edges[0].type == EdgeType.COMPARED_WITH

    def test_humanize_equipment(self):
        """Equipment ID humanization works."""
        builder = FocusGraphBuilder(SemanticFocusGraph(session_id="x"))
        assert builder._humanize_equipment("pump_004") == "Pump 4"
        assert builder._humanize_equipment("motor_12") == "Motor 12"
        assert builder._humanize_equipment("transformer_001") == "Transformer 1"


class TestFocusGraphPruning:
    """FG-B08: Graph pruning under capacity pressure."""

    def test_prune_stale_nodes_on_capacity(self):
        """Graph prunes stale nodes when exceeding MAX_NODES."""
        graph = SemanticFocusGraph(session_id="test-prune")
        graph.MAX_NODES = 5  # Low limit for testing

        # Add 5 nodes
        for i in range(5):
            node = FocusNode(
                id=f"metric:m{i}", type=NodeType.METRIC, label=f"Metric {i}",
                reference_count=1, last_referenced=time.time() - 600,  # 10min ago (stale)
            )
            graph.nodes[node.id] = node
        graph.version = 5

        # Adding a 6th should trigger pruning
        new_node = FocusNode(id="equipment:pump_001", type=NodeType.EQUIPMENT, label="Pump 1")
        graph.add_node(new_node)

        # Should have fewer nodes (some were pruned)
        assert len(graph.nodes) <= 5
        assert "equipment:pump_001" in graph.nodes

    def test_get_all_equipment_sorted_by_reference_count(self):
        """Equipment nodes returned in reference_count desc order."""
        graph = SemanticFocusGraph(session_id="test-sort")
        n1 = FocusNode(id="equipment:p1", type=NodeType.EQUIPMENT, label="P1", reference_count=3)
        n2 = FocusNode(id="equipment:p2", type=NodeType.EQUIPMENT, label="P2", reference_count=7)
        n3 = FocusNode(id="equipment:p3", type=NodeType.EQUIPMENT, label="P3", reference_count=1)
        graph.nodes = {n.id: n for n in [n1, n2, n3]}

        equipment = graph.get_all_equipment()
        assert equipment[0].id == "equipment:p2"
        assert equipment[1].id == "equipment:p1"
        assert equipment[2].id == "equipment:p3"

    def test_get_active_anomalies_filters_low_confidence(self):
        """Only anomalies with confidence > 0.5 returned."""
        graph = SemanticFocusGraph(session_id="test-anomalies")
        graph.nodes["a1"] = FocusNode(id="a1", type=NodeType.ANOMALY, label="High", confidence=0.8)
        graph.nodes["a2"] = FocusNode(id="a2", type=NodeType.ANOMALY, label="Low", confidence=0.3)
        anomalies = graph.get_active_anomalies()
        assert len(anomalies) == 1
        assert anomalies[0].id == "a1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
