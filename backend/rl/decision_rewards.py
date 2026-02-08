"""
Decision-Level Reward System — Upgrade 6

Extends RL from widget-level rewards to full decision-chain tracking.
Captures plan execution quality, constraint compliance, reasoning effectiveness,
and negative signals (ignored widgets, quick exits, re-asks).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .experience_buffer import Experience
from .reward_signals import RewardSignalAggregator

logger = logging.getLogger(__name__)


@dataclass
class DecisionExperience(Experience):
    """Extends Experience with full decision chain tracking."""

    # Decision chain
    plan_steps_executed: list = field(default_factory=list)
    plan_steps_failed: list = field(default_factory=list)
    constraints_evaluated: list = field(default_factory=list)
    constraints_violated: list = field(default_factory=list)
    reasoning_hypotheses: list = field(default_factory=list)
    focus_graph_version: int = 0
    focus_graph_node_count: int = 0

    # Negative signals
    widgets_ignored: list = field(default_factory=list)
    time_to_exit_ms: Optional[int] = None
    re_ask_within_30s: bool = False
    user_interrupted: bool = False
    plan_was_cancelled: bool = False
    clarification_was_needed: bool = False

    # Operator action tracking
    operator_created_work_order: bool = False
    operator_escalated: bool = False
    operator_overrode_suggestion: bool = False
    operator_acknowledged_alert: bool = False

    def has_decision_signals(self) -> bool:
        """Check if this experience has decision-chain data."""
        return (
            len(self.plan_steps_executed) > 0
            or len(self.constraints_evaluated) > 0
            or len(self.reasoning_hypotheses) > 0
        )

    def has_negative_signals(self) -> bool:
        """Check if this experience has any negative signals."""
        return (
            len(self.widgets_ignored) > 0
            or (self.time_to_exit_ms is not None and self.time_to_exit_ms < 5000)
            or self.re_ask_within_30s
            or self.user_interrupted
            or self.plan_was_cancelled
        )

    def has_operator_action(self) -> bool:
        """Check if operator took any real-world action."""
        return (
            self.operator_created_work_order
            or self.operator_escalated
            or self.operator_acknowledged_alert
        )


class DecisionRewardAggregator(RewardSignalAggregator):
    """Rewards the full decision chain, not just widget clicks."""

    DECISION_WEIGHTS = {
        # Positive signals
        "explicit_rating": 1.0,
        "widget_engagement": 0.3,
        "intent_confidence": 0.1,
        "plan_completion": 0.3,
        "constraint_compliance": 0.2,
        "reasoning_used": 0.15,
        "operator_action_taken": 0.5,

        # Negative signals
        "widgets_ignored_penalty": -0.2,
        "quick_exit_penalty": -0.4,
        "re_ask_penalty": -0.6,
        "interruption_penalty": -0.3,
        "override_penalty": -0.2,
        "constraint_violation_penalty": -0.5,
    }

    def __init__(self, weights: dict = None):
        super().__init__(weights)
        # Merge decision weights into base weights
        for k, v in self.DECISION_WEIGHTS.items():
            self.weights.setdefault(k, v)

    def compute_reward(self, experience: Experience) -> float:
        """Compute reward including decision-chain signals."""
        # Start with base reward from parent
        reward = super().compute_reward(experience)

        # Add decision-chain rewards if this is a DecisionExperience
        if isinstance(experience, DecisionExperience):
            reward += self._plan_completion_reward(experience)
            reward += self._constraint_compliance_reward(experience)
            reward += self._reasoning_reward(experience)
            reward += self._operator_action_reward(experience)
            reward += self._negative_signal_penalties(experience)

        return max(self.min_reward, min(self.max_reward, reward))

    def _plan_completion_reward(self, exp: DecisionExperience) -> float:
        """Reward for plan step completion rate."""
        if not exp.plan_steps_executed and not exp.plan_steps_failed:
            return 0.0

        total = len(exp.plan_steps_executed) + len(exp.plan_steps_failed)
        completion_rate = len(exp.plan_steps_executed) / max(total, 1)
        return completion_rate * self.weights.get("plan_completion", 0.3)

    def _constraint_compliance_reward(self, exp: DecisionExperience) -> float:
        """Reward for respecting constraints."""
        if not exp.constraints_evaluated:
            return 0.0

        violated = len(exp.constraints_violated)
        total = len(exp.constraints_evaluated)
        compliance = 1.0 - (violated / total)

        reward = compliance * self.weights.get("constraint_compliance", 0.2)

        # Additional penalty for constraint violations
        if violated > 0:
            reward += violated * self.weights.get("constraint_violation_penalty", -0.5)

        return reward

    def _reasoning_reward(self, exp: DecisionExperience) -> float:
        """Reward for using reasoning when appropriate."""
        if not exp.reasoning_hypotheses:
            return 0.0

        # Having hypotheses with reasonable confidence is a positive signal
        high_confidence = sum(
            1 for h in exp.reasoning_hypotheses
            if isinstance(h, dict) and h.get("confidence", 0) > 0.5
        )

        if high_confidence > 0:
            return self.weights.get("reasoning_used", 0.15)
        return self.weights.get("reasoning_used", 0.15) * 0.5

    def _operator_action_reward(self, exp: DecisionExperience) -> float:
        """Strong positive signal when operator acts on information."""
        reward = 0.0
        weight = self.weights.get("operator_action_taken", 0.5)

        if exp.operator_created_work_order:
            reward += weight
        if exp.operator_acknowledged_alert:
            reward += weight * 0.8
        if exp.operator_escalated:
            reward += weight * 0.6

        # Operator override is a mixed signal — they used the info but disagreed
        if exp.operator_overrode_suggestion:
            reward += self.weights.get("override_penalty", -0.2)

        return reward

    def _negative_signal_penalties(self, exp: DecisionExperience) -> float:
        """Compute penalties from negative behavioral signals."""
        penalty = 0.0

        # Widgets ignored
        if exp.widgets_ignored:
            # Exempt single-widget queries from this penalty
            total_widgets = len(exp.widget_plan.get("widgets", []))
            if total_widgets > 1:
                per_widget = self.weights.get("widgets_ignored_penalty", -0.2)
                penalty += max(len(exp.widgets_ignored) * per_widget, -0.8)

        # Quick exit (but exempt single-widget/KPI queries)
        if exp.time_to_exit_ms is not None and exp.time_to_exit_ms < 5000:
            total_widgets = len(exp.widget_plan.get("widgets", []))
            if total_widgets > 1:
                penalty += self.weights.get("quick_exit_penalty", -0.4)

        # Re-ask within 30s
        if exp.re_ask_within_30s:
            penalty += self.weights.get("re_ask_penalty", -0.6)

        # User interrupted
        if exp.user_interrupted:
            penalty += self.weights.get("interruption_penalty", -0.3)

        return penalty

    def get_decision_breakdown(self, exp: DecisionExperience) -> dict:
        """Get detailed breakdown of decision-chain reward components."""
        base = self.get_reward_breakdown(exp)
        base.update({
            "plan_completion": self._plan_completion_reward(exp),
            "constraint_compliance": self._constraint_compliance_reward(exp),
            "reasoning": self._reasoning_reward(exp),
            "operator_action": self._operator_action_reward(exp),
            "negative_signals": self._negative_signal_penalties(exp),
            "decision_total": self.compute_reward(exp),
        })
        return base
