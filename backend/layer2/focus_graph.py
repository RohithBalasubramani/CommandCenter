"""
Persistent Semantic Focus Graph — Upgrade 1

Replaces the flat interactiveCtx {key, scenario, label, equipment, metric}
with a graph that can represent relationships between entities, track causal
chains, and survive complex multi-turn investigations.

When a user says "now compare that with pump 5", the graph knows "that" means
the vibration anomaly on pump 4 discussed 3 turns ago.
"""

import time
import hashlib
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class NodeType(Enum):
    EQUIPMENT = "equipment"
    METRIC = "metric"
    ANOMALY = "anomaly"
    TIME_RANGE = "time_range"
    TASK = "task"
    DOCUMENT = "document"


class EdgeType(Enum):
    CAUSED_BY = "caused_by"
    CORRELATED_WITH = "correlated_with"
    DERIVED_FROM = "derived_from"
    PRECEDES = "precedes"
    EXPLAINS = "explains"
    MEASURED_BY = "measured_by"
    PART_OF = "part_of"
    AFFECTS = "affects"
    COMPARED_WITH = "compared_with"


@dataclass
class FocusNode:
    id: str
    type: NodeType
    label: str
    properties: dict = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "user_query"
    created_at: float = field(default_factory=time.time)
    last_referenced: float = field(default_factory=time.time)
    reference_count: int = 1


@dataclass
class FocusEdge:
    id: str
    source: str
    target: str
    type: EdgeType
    confidence: float = 0.8
    evidence: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class SemanticFocusGraph:
    session_id: str
    nodes: dict[str, FocusNode] = field(default_factory=dict)
    edges: list[FocusEdge] = field(default_factory=list)
    root_node_id: Optional[str] = None
    version: int = 0
    created_at: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)

    MAX_NODES = 50

    def add_node(self, node: FocusNode) -> str:
        """Add node, return id. If node exists, increment reference_count."""
        if node.id in self.nodes:
            self.nodes[node.id].reference_count += 1
            self.nodes[node.id].last_referenced = time.time()
            return node.id
        # Prune if at capacity
        if len(self.nodes) >= self.MAX_NODES:
            self._prune_stale_nodes()
        self.nodes[node.id] = node
        if self.root_node_id is None and node.type == NodeType.EQUIPMENT:
            self.root_node_id = node.id
        self.version += 1
        self.last_modified = time.time()
        return node.id

    def add_edge(self, source: str, target: str, edge_type: EdgeType,
                 confidence: float = 0.8, evidence: str = "") -> str:
        """Add edge between existing nodes. Returns edge id."""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Both nodes must exist: {source}, {target}")
        # Cycle detection for causal edges
        if edge_type == EdgeType.CAUSED_BY and self._would_create_cycle(source, target, EdgeType.CAUSED_BY):
            logger.warning(f"Rejected edge {source}→{target} ({edge_type.value}): would create cycle")
            return ""
        edge_id = hashlib.sha256(f"{source}:{target}:{edge_type.value}".encode()).hexdigest()[:16]
        # Deduplicate
        for e in self.edges:
            if e.id == edge_id:
                e.confidence = max(e.confidence, confidence)
                return edge_id
        self.edges.append(FocusEdge(
            id=edge_id, source=source, target=target,
            type=edge_type, confidence=confidence, evidence=evidence,
        ))
        self.version += 1
        self.last_modified = time.time()
        return edge_id

    def _would_create_cycle(self, source: str, target: str, edge_type: EdgeType) -> bool:
        """Check if adding source→target would create a cycle for the given edge type."""
        visited = set()
        stack = [source]
        while stack:
            current = stack.pop()
            if current == target:
                # Check if target can reach source (would complete cycle)
                pass
            if current in visited:
                continue
            visited.add(current)
        # BFS from target to see if we can reach source
        reachable = set()
        queue = [target]
        while queue:
            node = queue.pop(0)
            if node == source:
                return True
            if node in reachable:
                continue
            reachable.add(node)
            for edge in self.edges:
                if edge.type == edge_type and edge.source == node:
                    queue.append(edge.target)
        return False

    def _prune_stale_nodes(self):
        """Remove nodes with reference_count == 1 and last_referenced > 5min ago."""
        cutoff = time.time() - 300  # 5 minutes
        to_remove = [
            nid for nid, node in self.nodes.items()
            if node.reference_count <= 1
            and node.last_referenced < cutoff
            and nid != self.root_node_id
        ]
        for nid in to_remove[:10]:  # Remove at most 10 at a time
            del self.nodes[nid]
            self.edges = [e for e in self.edges if e.source != nid and e.target != nid]
            logger.info(f"Pruned stale node: {nid}")

    def get_neighbors(self, node_id: str, edge_type: EdgeType = None) -> list[FocusNode]:
        """Get all nodes connected to node_id, optionally filtered by edge type."""
        neighbor_ids = set()
        for edge in self.edges:
            if edge_type and edge.type != edge_type:
                continue
            if edge.source == node_id:
                neighbor_ids.add(edge.target)
            elif edge.target == node_id:
                neighbor_ids.add(edge.source)
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]

    def get_root_equipment(self) -> Optional[FocusNode]:
        """Return the primary equipment node."""
        if self.root_node_id and self.root_node_id in self.nodes:
            return self.nodes[self.root_node_id]
        return None

    def get_all_equipment(self) -> list[FocusNode]:
        """Return all equipment nodes, ordered by reference_count desc."""
        equip = [n for n in self.nodes.values() if n.type == NodeType.EQUIPMENT]
        return sorted(equip, key=lambda n: n.reference_count, reverse=True)

    def get_active_anomalies(self) -> list[FocusNode]:
        """Return anomaly nodes with confidence > 0.5."""
        return [n for n in self.nodes.values()
                if n.type == NodeType.ANOMALY and n.confidence > 0.5]

    def get_time_context(self) -> Optional[FocusNode]:
        """Return the most recently referenced time_range node."""
        times = [n for n in self.nodes.values() if n.type == NodeType.TIME_RANGE]
        if not times:
            return None
        return max(times, key=lambda n: n.last_referenced)

    def to_prompt_context(self, max_nodes: int = 15) -> str:
        """Serialize graph to text for LLM prompt injection."""
        lines = []
        ranked = sorted(self.nodes.values(), key=lambda n: n.reference_count, reverse=True)[:max_nodes]
        for node in ranked:
            lines.append(f"- [{node.type.value}] {node.label} (refs={node.reference_count}, conf={node.confidence:.2f})")
        if self.edges:
            lines.append("Relationships:")
            for edge in self.edges[:20]:
                src = self.nodes.get(edge.source)
                tgt = self.nodes.get(edge.target)
                if src and tgt:
                    lines.append(f"  {src.label} --{edge.type.value}--> {tgt.label} (conf={edge.confidence:.2f})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for JSON transport."""
        return {
            "session_id": self.session_id,
            "version": self.version,
            "root_node_id": self.root_node_id,
            "nodes": {nid: {
                "id": n.id, "type": n.type.value, "label": n.label,
                "properties": n.properties, "confidence": n.confidence,
                "source": n.source, "reference_count": n.reference_count,
            } for nid, n in self.nodes.items()},
            "edges": [{
                "id": e.id, "source": e.source, "target": e.target,
                "type": e.type.value, "confidence": e.confidence,
                "evidence": e.evidence,
            } for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticFocusGraph":
        """Deserialize from JSON."""
        try:
            graph = cls(session_id=data["session_id"])
            graph.version = data.get("version", 0)
            graph.root_node_id = data.get("root_node_id")
            for nid, nd in data.get("nodes", {}).items():
                graph.nodes[nid] = FocusNode(
                    id=nd["id"], type=NodeType(nd["type"]), label=nd["label"],
                    properties=nd.get("properties", {}),
                    confidence=nd.get("confidence", 1.0),
                    source=nd.get("source", "unknown"),
                    reference_count=nd.get("reference_count", 1),
                )
            for ed in data.get("edges", []):
                graph.edges.append(FocusEdge(
                    id=ed["id"], source=ed["source"], target=ed["target"],
                    type=EdgeType(ed["type"]),
                    confidence=ed.get("confidence", 0.8),
                    evidence=ed.get("evidence", ""),
                ))
            return graph
        except Exception as e:
            logger.warning(f"Failed to deserialize focus graph: {e}")
            raise
