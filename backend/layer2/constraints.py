"""
Constraint-Aware Decision Making — Upgrade 5

Treats data quality as first-class: freshness, confidence, latency,
coverage. Violations trigger REFUSE > QUALIFY > WARN > DEGRADE actions.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    SAFETY = "safety"
    DATA_FRESHNESS = "data_freshness"
    CONFIDENCE = "confidence"
    LATENCY = "latency"
    COVERAGE = "coverage"
    AUTHORITY = "authority"


class ViolationAction(Enum):
    REFUSE = "refuse"
    QUALIFY = "qualify"
    WARN = "warn"
    DEGRADE = "degrade"


@dataclass
class Constraint:
    type: ConstraintType
    description: str
    threshold: float
    actual: Optional[float] = None
    satisfied: bool = True
    action_on_violation: ViolationAction = ViolationAction.QUALIFY
    message: str = ""

    def evaluate(self, actual: float) -> bool:
        self.actual = actual
        if self.type == ConstraintType.DATA_FRESHNESS:
            self.satisfied = actual <= self.threshold
        elif self.type == ConstraintType.CONFIDENCE:
            self.satisfied = actual >= self.threshold
        elif self.type == ConstraintType.LATENCY:
            self.satisfied = actual <= self.threshold
        elif self.type == ConstraintType.COVERAGE:
            self.satisfied = actual >= self.threshold
        else:
            self.satisfied = actual >= self.threshold

        if not self.satisfied:
            self.message = self._build_violation_message()
        return self.satisfied

    def _build_violation_message(self) -> str:
        messages = {
            ConstraintType.DATA_FRESHNESS: f"Data is {self.actual:.0f}s old (limit: {self.threshold:.0f}s)",
            ConstraintType.CONFIDENCE: f"Confidence is {self.actual:.0%} (minimum: {self.threshold:.0%})",
            ConstraintType.LATENCY: f"Response took {self.actual:.0f}ms (budget: {self.threshold:.0f}ms)",
            ConstraintType.COVERAGE: f"Only {self.actual:.0%} of requested data available (need: {self.threshold:.0%})",
            ConstraintType.SAFETY: f"Safety constraint violated: threshold={self.threshold}, actual={self.actual}",
            ConstraintType.AUTHORITY: f"Insufficient authorization level: {self.actual} < {self.threshold}",
        }
        return messages.get(self.type, f"Constraint violated: {self.type.value}")

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "description": self.description,
            "threshold": self.threshold,
            "actual": self.actual,
            "satisfied": self.satisfied,
            "action": self.action_on_violation.value,
            "message": self.message,
        }


@dataclass
class ConstraintViolation:
    """A recorded violation of a constraint."""
    constraint: Constraint
    action_taken: ViolationAction
    voice_qualifier: str = ""
    widget_qualifier: str = ""

    def to_dict(self) -> dict:
        return {
            "constraint": self.constraint.to_dict(),
            "action_taken": self.action_taken.value,
            "voice_qualifier": self.voice_qualifier,
            "widget_qualifier": self.widget_qualifier,
        }


@dataclass
class ConstraintCheckResult:
    """Result of checking all constraints."""
    can_proceed: bool = True
    violations: list[ConstraintViolation] = field(default_factory=list)
    qualifiers: list[str] = field(default_factory=list)
    highest_action: ViolationAction = ViolationAction.DEGRADE

    def to_dict(self) -> dict:
        return {
            "can_proceed": self.can_proceed,
            "violations": [v.to_dict() for v in self.violations],
            "qualifiers": self.qualifiers,
            "highest_action": self.highest_action.value,
        }


# Action severity ordering
ACTION_SEVERITY = {
    ViolationAction.REFUSE: 4,
    ViolationAction.QUALIFY: 3,
    ViolationAction.WARN: 2,
    ViolationAction.DEGRADE: 1,
}


# Default constraint templates
DEFAULT_CONSTRAINTS = [
    Constraint(
        type=ConstraintType.DATA_FRESHNESS,
        description="Sensor data must be less than 5 minutes old",
        threshold=300.0,  # 300 seconds = 5 minutes
        action_on_violation=ViolationAction.QUALIFY,
    ),
    Constraint(
        type=ConstraintType.CONFIDENCE,
        description="AI confidence must be above 30%",
        threshold=0.30,
        action_on_violation=ViolationAction.WARN,
    ),
    Constraint(
        type=ConstraintType.LATENCY,
        description="Total response under 8 seconds",
        threshold=8000.0,
        action_on_violation=ViolationAction.DEGRADE,
    ),
    Constraint(
        type=ConstraintType.COVERAGE,
        description="At least 50% of requested data available",
        threshold=0.50,
        action_on_violation=ViolationAction.QUALIFY,
    ),
]

# Voice qualifier templates
VOICE_QUALIFIERS = {
    ConstraintType.DATA_FRESHNESS: "Note: this data may not be current — it's {age} old.",
    ConstraintType.CONFIDENCE: "I'm not fully confident in this answer.",
    ConstraintType.COVERAGE: "I could only find partial data for your request.",
    ConstraintType.LATENCY: "This response was delayed due to system load.",
}


class ConstraintChecker:
    """Evaluates constraints and determines appropriate action."""

    def __init__(self, constraints: list[Constraint] = None):
        self.constraints = constraints or [
            Constraint(
                type=c.type, description=c.description,
                threshold=c.threshold, action_on_violation=c.action_on_violation,
            )
            for c in DEFAULT_CONSTRAINTS
        ]

    def check(self, actuals: dict[ConstraintType, float]) -> ConstraintCheckResult:
        """
        Evaluate all constraints against actual values.

        Args:
            actuals: Dict mapping constraint type to actual measured value
                     e.g., {ConstraintType.DATA_FRESHNESS: 600, ConstraintType.CONFIDENCE: 0.85}
        """
        result = ConstraintCheckResult()
        max_severity = 0

        for constraint in self.constraints:
            actual = actuals.get(constraint.type)
            if actual is None:
                continue

            satisfied = constraint.evaluate(actual)
            if not satisfied:
                action = constraint.action_on_violation
                severity = ACTION_SEVERITY.get(action, 0)
                if severity > max_severity:
                    max_severity = severity
                    result.highest_action = action

                # Build voice qualifier
                qualifier_template = VOICE_QUALIFIERS.get(constraint.type, "")
                voice_qualifier = ""
                if qualifier_template:
                    if constraint.type == ConstraintType.DATA_FRESHNESS:
                        age_mins = int(constraint.actual / 60)
                        voice_qualifier = qualifier_template.format(
                            age=f"{age_mins} minutes" if age_mins > 1 else "over a minute"
                        )
                    else:
                        voice_qualifier = qualifier_template

                violation = ConstraintViolation(
                    constraint=constraint,
                    action_taken=action,
                    voice_qualifier=voice_qualifier,
                    widget_qualifier=constraint.message,
                )
                result.violations.append(violation)

                if voice_qualifier:
                    result.qualifiers.append(voice_qualifier)

                logger.info(
                    f"[Constraint] {constraint.type.value} violated: "
                    f"{constraint.message} → {action.value}"
                )

        # REFUSE prevents proceeding
        if any(v.action_taken == ViolationAction.REFUSE for v in result.violations):
            result.can_proceed = False

        return result

    def build_voice_prefix(self, check_result: ConstraintCheckResult) -> str:
        """Build a voice qualifier prefix from constraint violations."""
        if not check_result.qualifiers:
            return ""
        return " ".join(check_result.qualifiers) + " "
