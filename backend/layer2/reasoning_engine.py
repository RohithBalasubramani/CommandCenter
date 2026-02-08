"""
Cross-Widget Reasoning Engine — Upgrade 2

Structured diagnostic reasoning across multiple widget data sources.
Uses rule-based pattern matching first, then statistical correlation,
then LLM synthesis for natural language output.
"""

import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    CORRELATIVE = "correlative"


@dataclass
class DataPoint:
    """A concrete data observation from a widget or RAG."""
    equipment: str
    metric: str
    value: float
    unit: str
    timestamp: float
    status: str  # "normal", "warning", "critical"
    threshold: Optional[float] = None
    source: str = ""


@dataclass
class ReasoningQuery:
    """Structured input to the reasoning engine."""
    type: ReasoningType
    question: str
    observations: list[DataPoint] = field(default_factory=list)
    focus_graph: Optional[dict] = None
    constraints: list = field(default_factory=list)


@dataclass
class Hypothesis:
    """A possible explanation with evidence."""
    id: str
    statement: str
    confidence: float
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    check_steps: list[str] = field(default_factory=list)
    source: str = "knowledge_base"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "check_steps": self.check_steps,
            "source": self.source,
        }


@dataclass
class ReasoningResult:
    """Output of the reasoning engine."""
    query_type: ReasoningType
    hypotheses: list[Hypothesis] = field(default_factory=list)
    known_facts: list[str] = field(default_factory=list)
    unknown_factors: list[str] = field(default_factory=list)
    recommended_checks: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_chain: list[str] = field(default_factory=list)
    execution_time_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "query_type": self.query_type.value,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "known_facts": self.known_facts,
            "unknown_factors": self.unknown_factors,
            "recommended_checks": self.recommended_checks,
            "confidence": self.confidence,
            "reasoning_chain": self.reasoning_chain,
            "execution_time_ms": self.execution_time_ms,
        }


class CrossWidgetReasoningEngine:
    """
    Structured diagnostic reasoning across multiple widget data sources.

    Pipeline:
    1. Rule-based pattern matching for known failure modes
    2. Statistical correlation from historical data
    3. LLM synthesis for natural language output
    """

    DIAGNOSTIC_PATTERNS = [
        {
            "pattern": "vibration_high_temperature_normal",
            "conditions": {
                "vibration": {"status": "critical", "threshold_exceeded": True},
                "temperature": {"status": "normal", "threshold_exceeded": False},
            },
            "hypotheses": [
                {
                    "statement": "Bearing wear or misalignment — mechanical vibration "
                                 "increases before thermal effects manifest",
                    "confidence": 0.72,
                    "check_steps": [
                        "Check vibration frequency spectrum for bearing defect frequencies",
                        "Review last alignment report",
                        "Check bearing replacement date",
                    ],
                },
                {
                    "statement": "Loose mounting or foundation issue — causes vibration "
                                 "without thermal impact",
                    "confidence": 0.45,
                    "check_steps": [
                        "Inspect mounting bolts",
                        "Check foundation for cracks",
                    ],
                },
            ],
        },
        {
            "pattern": "temperature_high_vibration_normal",
            "conditions": {
                "temperature": {"status": "critical", "threshold_exceeded": True},
                "vibration": {"status": "normal", "threshold_exceeded": False},
            },
            "hypotheses": [
                {
                    "statement": "Cooling system degradation — insufficient heat dissipation "
                                 "while mechanical components operate normally",
                    "confidence": 0.68,
                    "check_steps": [
                        "Check coolant flow rate and level",
                        "Inspect heat exchanger for fouling",
                        "Verify fan operation",
                    ],
                },
                {
                    "statement": "Overload condition — equipment drawing more power than rated, "
                                 "generating excess heat without mechanical wear patterns",
                    "confidence": 0.55,
                    "check_steps": [
                        "Check current draw vs rated current",
                        "Verify load conditions",
                        "Check process parameters",
                    ],
                },
            ],
        },
        {
            "pattern": "power_high_efficiency_low",
            "conditions": {
                "power": {"status": "warning", "threshold_exceeded": True},
                "efficiency": {"status": "warning", "threshold_exceeded": True},
            },
            "hypotheses": [
                {
                    "statement": "Mechanical friction increase — wear on bearings, seals, "
                                 "or impeller causing energy waste as heat",
                    "confidence": 0.65,
                    "check_steps": [
                        "Compare current efficiency with commissioning baseline",
                        "Check bearing temperatures",
                        "Inspect seal condition",
                    ],
                },
            ],
        },
        {
            "pattern": "multiple_metrics_degraded",
            "conditions": {
                "_all_degraded": True,
            },
            "hypotheses": [
                {
                    "statement": "Systemic degradation — multiple parameters degrading "
                                 "simultaneously suggests overall equipment health decline",
                    "confidence": 0.60,
                    "check_steps": [
                        "Review overall equipment health score trend",
                        "Check time since last comprehensive maintenance",
                        "Schedule condition assessment",
                    ],
                },
            ],
        },
        {
            "pattern": "vibration_high_power_high",
            "conditions": {
                "vibration": {"status": "critical", "threshold_exceeded": True},
                "power": {"status": "warning", "threshold_exceeded": True},
            },
            "hypotheses": [
                {
                    "statement": "Impeller damage or cavitation — causes both increased "
                                 "vibration and higher power consumption",
                    "confidence": 0.70,
                    "check_steps": [
                        "Check for cavitation indicators (noise, pressure fluctuation)",
                        "Review suction pressure",
                        "Inspect impeller condition during next outage",
                    ],
                },
            ],
        },
    ]

    def reason(self, query: ReasoningQuery) -> ReasoningResult:
        """Execute reasoning pipeline."""
        start = time.time()

        # Insufficient data check
        if len(query.observations) < 2:
            return ReasoningResult(
                query_type=query.type,
                known_facts=[self._describe_observation(obs) for obs in query.observations],
                unknown_factors=["Only one metric available — need at least two for cross-widget reasoning"],
                confidence=0.2,
                reasoning_chain=["Insufficient data for cross-widget reasoning"],
                execution_time_ms=int((time.time() - start) * 1000),
            )

        # Step 1: Collect known facts
        known_facts = [self._describe_observation(obs) for obs in query.observations]

        # Step 2: Pattern matching
        pattern_hypotheses = self._match_diagnostic_patterns(query.observations)

        # Step 3: Build reasoning chain
        reasoning_chain = [
            f"Step 1: Collected {len(query.observations)} data points",
            f"Step 2: Pattern matching found {len(pattern_hypotheses)} hypotheses",
        ]

        # Step 4: Determine unknowns
        unknown_factors = self._identify_unknowns(query.observations)

        # Step 5: Recommended checks from all hypotheses
        recommended_checks = []
        for h in pattern_hypotheses:
            recommended_checks.extend(h.check_steps)
        recommended_checks = list(dict.fromkeys(recommended_checks))  # dedupe

        # Adjust confidence if contradicting evidence found
        for h in pattern_hypotheses:
            if len(h.contradicting_evidence) > len(h.supporting_evidence):
                h.confidence *= 0.5
                reasoning_chain.append(
                    f"Step 3: Demoted '{h.statement[:50]}...' — "
                    f"contradicting ({len(h.contradicting_evidence)}) > supporting ({len(h.supporting_evidence)})"
                )

        # Overall confidence
        overall_confidence = max([h.confidence for h in pattern_hypotheses], default=0.0)

        execution_time = int((time.time() - start) * 1000)

        return ReasoningResult(
            query_type=query.type,
            hypotheses=pattern_hypotheses,
            known_facts=known_facts,
            unknown_factors=unknown_factors,
            recommended_checks=recommended_checks[:10],
            confidence=overall_confidence,
            reasoning_chain=reasoning_chain,
            execution_time_ms=execution_time,
        )

    def detect_reasoning_type(self, text: str) -> Optional[ReasoningType]:
        """Detect if a query needs reasoning and what type."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["why", "cause", "reason", "explain"]):
            if "but" in text_lower or "although" in text_lower or "despite" in text_lower:
                return ReasoningType.COMPARATIVE
            return ReasoningType.CAUSAL
        if any(w in text_lower for w in ["what if", "will", "predict", "expect"]):
            return ReasoningType.PREDICTIVE
        if any(w in text_lower for w in ["related", "correlat", "connect"]):
            return ReasoningType.CORRELATIVE
        if any(w in text_lower for w in ["diagnos", "troubleshoot", "investigate"]):
            return ReasoningType.DIAGNOSTIC
        return None

    def _match_diagnostic_patterns(self, observations: list[DataPoint]) -> list[Hypothesis]:
        """Rule-based pattern matching against DIAGNOSTIC_PATTERNS."""
        hypotheses = []

        # Build observation map: metric_keyword → DataPoint
        obs_map = {}
        for obs in observations:
            metric_lower = obs.metric.lower()
            for keyword in ["vibration", "temperature", "power", "efficiency",
                            "pressure", "flow", "current", "voltage"]:
                if keyword in metric_lower:
                    obs_map[keyword] = obs
                    break

        for pattern in self.DIAGNOSTIC_PATTERNS:
            conditions = pattern["conditions"]

            # Special case: all degraded
            if conditions.get("_all_degraded"):
                non_normal = [o for o in observations if o.status != "normal"]
                if len(non_normal) >= 2 and len(non_normal) == len(observations):
                    for h_data in pattern["hypotheses"]:
                        hypothesis = self._build_hypothesis(h_data, pattern["pattern"], observations)
                        hypotheses.append(hypothesis)
                continue

            # Normal pattern matching
            matched = True
            for metric_key, expected in conditions.items():
                obs = obs_map.get(metric_key)
                if not obs:
                    matched = False
                    break
                if expected.get("status") and obs.status != expected["status"]:
                    matched = False
                    break
                if expected.get("threshold_exceeded"):
                    if obs.threshold and obs.value <= obs.threshold:
                        matched = False
                        break

            if matched:
                for h_data in pattern["hypotheses"]:
                    hypothesis = self._build_hypothesis(h_data, pattern["pattern"], observations)
                    hypotheses.append(hypothesis)

        return hypotheses

    def _build_hypothesis(self, h_data: dict, pattern_name: str,
                           observations: list[DataPoint]) -> Hypothesis:
        """Build a Hypothesis from pattern match data."""
        h_id = hashlib.sha256(
            f"{pattern_name}:{h_data['statement'][:30]}".encode()
        ).hexdigest()[:12]

        supporting = []
        for obs in observations:
            if obs.status in ("critical", "warning"):
                supporting.append(
                    f"{obs.metric} at {obs.value}{obs.unit} — {obs.status}"
                    + (f" (threshold: {obs.threshold})" if obs.threshold else "")
                )

        return Hypothesis(
            id=h_id,
            statement=h_data["statement"],
            confidence=h_data["confidence"],
            supporting_evidence=supporting,
            contradicting_evidence=[],
            check_steps=h_data.get("check_steps", []),
            source="knowledge_base",
        )

    def _describe_observation(self, obs: DataPoint) -> str:
        """Human-readable description of a data point."""
        desc = f"{obs.metric} at {obs.value}{obs.unit} — {obs.status}"
        if obs.threshold:
            if obs.value > obs.threshold:
                desc += f" (ABOVE threshold of {obs.threshold}{obs.unit})"
            else:
                desc += f" (within threshold of {obs.threshold}{obs.unit})"
        return desc

    def _identify_unknowns(self, observations: list[DataPoint]) -> list[str]:
        """Identify what we don't know from the available data."""
        unknowns = []
        metrics_seen = {obs.metric.lower() for obs in observations}

        # Common unknowns based on what metrics we have
        if any("vibration" in m for m in metrics_seen):
            unknowns.append("Vibration frequency spectrum not available from current sensors")
        if not any("bearing" in m for m in metrics_seen):
            unknowns.append("Bearing temperature not being monitored")
        if not any("maintenance" in str(obs.source) for obs in observations):
            unknowns.append("Last maintenance date not available in current data")

        return unknowns
