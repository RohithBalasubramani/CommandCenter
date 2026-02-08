"""
Uncertainty Assessment — Upgrade 10

Builds explicit uncertainty assessments from plan execution results,
constraint evaluations, and reasoning outputs. Determines when to
show the Uncertainty Panel widget.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KnownFact:
    statement: str
    source: str = "unknown"      # timeseries, rag, site_memory, reasoning
    freshness: str = "unknown"   # live, 5min, 1hr, stale
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "statement": self.statement,
            "source": self.source,
            "freshness": self.freshness,
            "confidence": self.confidence,
        }


@dataclass
class UnknownFactor:
    description: str
    why_unknown: str = "no_data"   # no_sensor, data_stale, no_knowledge
    impact: str = "medium"         # low, medium, high
    check_action: str = ""

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "why_unknown": self.why_unknown,
            "impact": self.impact,
            "check_action": self.check_action,
        }


@dataclass
class NextStep:
    action: str
    automated: bool = False
    priority: str = "medium"   # low, medium, high

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "automated": self.automated,
            "priority": self.priority,
        }


@dataclass
class ConstraintViolationInfo:
    type: str
    message: str
    severity: str = "warning"   # warning, error

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class UncertaintyAssessment:
    """Complete uncertainty assessment for a response."""
    overall_confidence: float = 0.5
    known_facts: list[KnownFact] = field(default_factory=list)
    unknown_factors: list[UnknownFactor] = field(default_factory=list)
    next_steps: list[NextStep] = field(default_factory=list)
    constraint_violations: list[ConstraintViolationInfo] = field(default_factory=list)

    def should_show_panel(self) -> bool:
        """Determine if the uncertainty panel should be displayed."""
        if self.overall_confidence < 0.6:
            return True
        if len(self.unknown_factors) >= 2:
            return True
        if len(self.constraint_violations) >= 1:
            return True
        return False

    def get_voice_prefix(self) -> str:
        """Generate voice prefix based on confidence level."""
        if self.overall_confidence < 0.4:
            return "I'm not very confident in this answer. "
        elif self.overall_confidence < 0.6:
            return "Based on what I can see, but with some gaps — "
        return ""

    def get_voice_suffix(self) -> str:
        """Generate voice suffix about unknowns."""
        if self.unknown_factors:
            descriptions = [u.description for u in self.unknown_factors[:2]]
            return f" I should note that I don't have data on: {', '.join(descriptions)}."
        return ""

    def to_dict(self) -> dict:
        return {
            "overall_confidence": self.overall_confidence,
            "known_facts": [f.to_dict() for f in self.known_facts],
            "unknown_factors": [u.to_dict() for u in self.unknown_factors],
            "next_steps": [s.to_dict() for s in self.next_steps],
            "constraint_violations": [v.to_dict() for v in self.constraint_violations],
        }


class UncertaintyAssessor:
    """
    Builds uncertainty assessments from execution outputs.

    Analyzes plan results, constraint evaluations, and reasoning
    to build a complete picture of what's known and unknown.
    """

    MAX_UNKNOWNS_DISPLAY = 5

    def assess(
        self,
        plan_steps: list[dict] = None,
        constraint_result: dict = None,
        reasoning_result: dict = None,
        data_freshness_s: float = None,
    ) -> UncertaintyAssessment:
        """
        Build uncertainty assessment from available data.

        Args:
            plan_steps: List of plan step results with status/outputs
            constraint_result: Constraint checker result dict
            reasoning_result: Reasoning engine result dict
            data_freshness_s: Data age in seconds
        """
        assessment = UncertaintyAssessment()
        plan_steps = plan_steps or []
        confidence_components = []

        # Extract known facts from completed retrieve steps
        for step in plan_steps:
            if step.get("type") == "RETRIEVE" and step.get("status") == "completed":
                outputs = step.get("outputs", {})
                fact = KnownFact(
                    statement=self._format_fact(outputs),
                    source=outputs.get("source", "timeseries"),
                    freshness=self._freshness_label(outputs.get("age_seconds")),
                    confidence=outputs.get("confidence", 0.7),
                )
                assessment.known_facts.append(fact)
                confidence_components.append(fact.confidence)

        # Extract unknowns from failed retrieve steps
        for step in plan_steps:
            if step.get("type") == "RETRIEVE" and step.get("status") == "failed":
                equipment = step.get("inputs", {}).get("equipment", "equipment")
                assessment.unknown_factors.append(UnknownFactor(
                    description=step.get("description", f"Data for {equipment}"),
                    why_unknown=step.get("error", "Data retrieval failed"),
                    impact="medium",
                    check_action=f"Verify sensor connectivity for {equipment}",
                ))

        # Extract from reasoning result
        if reasoning_result:
            for fact in reasoning_result.get("known_facts", []):
                if isinstance(fact, str):
                    assessment.known_facts.append(KnownFact(
                        statement=fact,
                        source="reasoning",
                        freshness="computed",
                        confidence=reasoning_result.get("confidence", 0.5),
                    ))

            for unknown in reasoning_result.get("unknown_factors", []):
                if isinstance(unknown, str):
                    assessment.unknown_factors.append(UnknownFactor(
                        description=unknown,
                        why_unknown="Not enough data to determine",
                        impact="high",
                    ))

            for check in reasoning_result.get("recommended_checks", []):
                if isinstance(check, str):
                    assessment.next_steps.append(NextStep(
                        action=check,
                        automated=False,
                        priority="medium",
                    ))

            if reasoning_result.get("confidence"):
                confidence_components.append(reasoning_result["confidence"])

        # Extract constraint violations
        if constraint_result:
            for v in constraint_result.get("violations", []):
                severity = "error" if v.get("action") == "refuse" else "warning"
                assessment.constraint_violations.append(ConstraintViolationInfo(
                    type=v.get("type", "unknown"),
                    message=v.get("message", "Constraint violated"),
                    severity=severity,
                ))

        # Truncate unknowns if too many
        if len(assessment.unknown_factors) > self.MAX_UNKNOWNS_DISPLAY:
            extra = len(assessment.unknown_factors) - self.MAX_UNKNOWNS_DISPLAY
            assessment.unknown_factors = assessment.unknown_factors[:self.MAX_UNKNOWNS_DISPLAY]
            assessment.unknown_factors.append(UnknownFactor(
                description=f"...and {extra} more unknown factors",
                why_unknown="truncated",
                impact="low",
            ))

        # Compute overall confidence
        assessment.overall_confidence = self._compute_confidence(
            confidence_components,
            len(assessment.unknown_factors),
            len(assessment.constraint_violations),
            data_freshness_s,
        )

        return assessment

    def _format_fact(self, outputs: dict) -> str:
        """Format a retrieve step output as a fact statement."""
        metric = outputs.get("metric", "Value")
        equipment = outputs.get("equipment", "equipment")
        value = outputs.get("value", "N/A")
        unit = outputs.get("unit", "")
        return f"{metric} for {equipment}: {value} {unit}".strip()

    def _freshness_label(self, age_seconds: float = None) -> str:
        """Convert age in seconds to a freshness label."""
        if age_seconds is None:
            return "unknown"
        if age_seconds < 60:
            return "live"
        elif age_seconds < 300:
            return "5min"
        elif age_seconds < 3600:
            return "1hr"
        return "stale"

    def _compute_confidence(
        self,
        components: list[float],
        unknowns_count: int,
        violations_count: int,
        data_freshness_s: float = None,
    ) -> float:
        """Compute overall confidence score."""
        if not components and unknowns_count == 0:
            return 0.5  # No data at all

        # Base confidence from component average
        if components:
            base = sum(components) / len(components)
        else:
            base = 0.3

        # Penalize for unknowns
        unknown_penalty = min(unknowns_count * 0.1, 0.4)
        base -= unknown_penalty

        # Penalize for constraint violations
        violation_penalty = min(violations_count * 0.15, 0.3)
        base -= violation_penalty

        # Penalize for stale data
        if data_freshness_s is not None and data_freshness_s > 300:
            staleness_penalty = min((data_freshness_s - 300) / 3600 * 0.2, 0.2)
            base -= staleness_penalty

        # All facts low confidence → floor at 0.1
        if components and all(c < 0.3 for c in components):
            base = min(base, 0.1)

        return max(0.0, min(1.0, base))
