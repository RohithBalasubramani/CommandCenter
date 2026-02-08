"""
Tests for Upgrade 6: RL Decision Rewards

Test IDs: DR-B01 through DR-B10
"""

import pytest
from rl.decision_rewards import DecisionExperience, DecisionRewardAggregator


@pytest.fixture
def aggregator():
    return DecisionRewardAggregator()


def _make_exp(**kwargs) -> DecisionExperience:
    """Helper to create a DecisionExperience with defaults."""
    defaults = {
        "query_id": "test-001",
        "transcript": "show pump 4 vibration",
        "widget_plan": {"widgets": [{"scenario": "kpi"}, {"scenario": "trend"}]},
        "intent_confidence": 0.85,
        "processing_time_ms": 2000,
    }
    defaults.update(kwargs)
    return DecisionExperience(**defaults)


class TestDecisionExperienceStructure:
    """DR-B01: DecisionExperience extends Experience correctly."""

    def test_inherits_base_fields(self):
        exp = _make_exp()
        assert exp.query_id == "test-001"
        assert exp.transcript == "show pump 4 vibration"
        assert exp.intent_confidence == 0.85

    def test_has_decision_chain_fields(self):
        exp = _make_exp(
            plan_steps_executed=[{"type": "RETRIEVE"}, {"type": "SELECT_WIDGETS"}],
            constraints_evaluated=[{"type": "data_freshness"}],
        )
        assert len(exp.plan_steps_executed) == 2
        assert len(exp.constraints_evaluated) == 1
        assert exp.has_decision_signals()

    def test_has_negative_signals(self):
        exp = _make_exp(
            widgets_ignored=["alerts"],
            time_to_exit_ms=3000,
        )
        assert exp.has_negative_signals()

    def test_no_negative_signals(self):
        exp = _make_exp()
        assert not exp.has_negative_signals()

    def test_has_operator_action(self):
        exp = _make_exp(operator_created_work_order=True)
        assert exp.has_operator_action()

    def test_serializes_to_dict(self):
        exp = _make_exp(
            plan_steps_executed=[{"type": "RETRIEVE"}],
            widgets_ignored=["alerts"],
        )
        d = exp.to_dict()
        assert d["plan_steps_executed"] == [{"type": "RETRIEVE"}]
        assert d["widgets_ignored"] == ["alerts"]


class TestPlanCompletionReward:
    """DR-B02: Plan completion rate rewards."""

    def test_full_completion(self, aggregator):
        exp = _make_exp(
            plan_steps_executed=[
                {"type": "RETRIEVE"},
                {"type": "SELECT_WIDGETS"},
                {"type": "GENERATE_RESPONSE"},
            ],
            plan_steps_failed=[],
        )
        reward = aggregator._plan_completion_reward(exp)
        assert reward == pytest.approx(0.3, abs=0.01)

    def test_partial_completion(self, aggregator):
        exp = _make_exp(
            plan_steps_executed=[{"type": "RETRIEVE"}],
            plan_steps_failed=[{"type": "REASON"}, {"type": "SELECT_WIDGETS"}],
        )
        reward = aggregator._plan_completion_reward(exp)
        expected = (1/3) * 0.3
        assert reward == pytest.approx(expected, abs=0.01)

    def test_no_plan(self, aggregator):
        exp = _make_exp()
        reward = aggregator._plan_completion_reward(exp)
        assert reward == 0.0


class TestConstraintComplianceReward:
    """DR-B03: Constraint compliance rewards."""

    def test_all_constraints_satisfied(self, aggregator):
        exp = _make_exp(
            constraints_evaluated=[
                {"type": "data_freshness", "satisfied": True},
                {"type": "confidence", "satisfied": True},
            ],
            constraints_violated=[],
        )
        reward = aggregator._constraint_compliance_reward(exp)
        assert reward == pytest.approx(0.2, abs=0.01)

    def test_one_constraint_violated(self, aggregator):
        exp = _make_exp(
            constraints_evaluated=[
                {"type": "data_freshness"},
                {"type": "confidence"},
            ],
            constraints_violated=[
                {"type": "data_freshness"},
            ],
        )
        reward = aggregator._constraint_compliance_reward(exp)
        # compliance = 0.5, reward = 0.5 * 0.2 + 1 * (-0.5) = 0.1 - 0.5 = -0.4
        assert reward < 0

    def test_no_constraints(self, aggregator):
        exp = _make_exp()
        assert aggregator._constraint_compliance_reward(exp) == 0.0


class TestReasoningReward:
    """DR-B04: Reasoning quality rewards."""

    def test_high_confidence_hypotheses(self, aggregator):
        exp = _make_exp(
            reasoning_hypotheses=[
                {"description": "Bearing wear", "confidence": 0.8},
                {"description": "Misalignment", "confidence": 0.6},
            ],
        )
        reward = aggregator._reasoning_reward(exp)
        assert reward == pytest.approx(0.15, abs=0.01)

    def test_low_confidence_hypotheses(self, aggregator):
        exp = _make_exp(
            reasoning_hypotheses=[
                {"description": "Unknown", "confidence": 0.3},
            ],
        )
        reward = aggregator._reasoning_reward(exp)
        assert reward == pytest.approx(0.075, abs=0.01)

    def test_no_reasoning(self, aggregator):
        exp = _make_exp()
        assert aggregator._reasoning_reward(exp) == 0.0


class TestOperatorActionReward:
    """DR-B05: Operator action rewards."""

    def test_work_order_created(self, aggregator):
        exp = _make_exp(operator_created_work_order=True)
        reward = aggregator._operator_action_reward(exp)
        assert reward == pytest.approx(0.5, abs=0.01)

    def test_alert_acknowledged(self, aggregator):
        exp = _make_exp(operator_acknowledged_alert=True)
        reward = aggregator._operator_action_reward(exp)
        assert reward == pytest.approx(0.4, abs=0.01)

    def test_override_penalty(self, aggregator):
        exp = _make_exp(operator_overrode_suggestion=True)
        reward = aggregator._operator_action_reward(exp)
        assert reward == pytest.approx(-0.2, abs=0.01)

    def test_multiple_actions(self, aggregator):
        exp = _make_exp(
            operator_created_work_order=True,
            operator_acknowledged_alert=True,
        )
        reward = aggregator._operator_action_reward(exp)
        assert reward > 0.5


class TestNegativeSignalPenalties:
    """DR-B06/B07: Negative signal penalties."""

    def test_widgets_ignored_penalty(self, aggregator):
        exp = _make_exp(
            widgets_ignored=["alerts", "trend"],
            widget_plan={"widgets": [{"scenario": "kpi"}, {"scenario": "trend"}, {"scenario": "alerts"}]},
        )
        penalty = aggregator._negative_signal_penalties(exp)
        assert penalty < 0

    def test_single_widget_exempt_from_ignore_penalty(self, aggregator):
        """Single-widget queries should not penalize quick exit."""
        exp = _make_exp(
            widgets_ignored=["kpi"],
            widget_plan={"widgets": [{"scenario": "kpi"}]},
        )
        penalty = aggregator._negative_signal_penalties(exp)
        assert penalty == 0.0

    def test_quick_exit_penalty(self, aggregator):
        exp = _make_exp(
            time_to_exit_ms=2000,
            widget_plan={"widgets": [{"scenario": "kpi"}, {"scenario": "trend"}]},
        )
        penalty = aggregator._negative_signal_penalties(exp)
        assert penalty == pytest.approx(-0.4, abs=0.01)

    def test_quick_exit_exempt_single_widget(self, aggregator):
        """Single-widget queries exempt from quick exit."""
        exp = _make_exp(
            time_to_exit_ms=2000,
            widget_plan={"widgets": [{"scenario": "kpi"}]},
        )
        penalty = aggregator._negative_signal_penalties(exp)
        assert penalty == 0.0

    def test_re_ask_penalty(self, aggregator):
        exp = _make_exp(re_ask_within_30s=True)
        penalty = aggregator._negative_signal_penalties(exp)
        assert penalty == pytest.approx(-0.6, abs=0.01)

    def test_interruption_penalty(self, aggregator):
        exp = _make_exp(user_interrupted=True)
        penalty = aggregator._negative_signal_penalties(exp)
        assert penalty == pytest.approx(-0.3, abs=0.01)


class TestFullRewardComputation:
    """DR-B08: End-to-end reward computation."""

    def test_positive_decision(self, aggregator):
        """Good experience: thumbs up + full plan + no violations."""
        exp = _make_exp(
            user_rating="up",
            plan_steps_executed=[{"type": "RETRIEVE"}, {"type": "SELECT_WIDGETS"}],
            constraints_evaluated=[{"type": "freshness"}],
            constraints_violated=[],
            operator_created_work_order=True,
        )
        reward = aggregator.compute_reward(exp)
        assert reward > 1.0  # Strongly positive

    def test_negative_decision(self, aggregator):
        """Bad experience: thumbs down + failed plan + re-ask."""
        exp = _make_exp(
            user_rating="down",
            plan_steps_executed=[],
            plan_steps_failed=[{"type": "RETRIEVE"}],
            re_ask_within_30s=True,
            user_interrupted=True,
        )
        reward = aggregator.compute_reward(exp)
        assert reward < -0.5  # Strongly negative

    def test_reward_clipping(self, aggregator):
        """Reward stays within [-2.0, 2.0]."""
        exp = _make_exp(
            user_rating="up",
            operator_created_work_order=True,
            operator_acknowledged_alert=True,
            operator_escalated=True,
            plan_steps_executed=[{"type": "A"}, {"type": "B"}, {"type": "C"}],
            reasoning_hypotheses=[{"confidence": 0.9}],
            constraints_evaluated=[{"type": "x"}],
        )
        reward = aggregator.compute_reward(exp)
        assert -2.0 <= reward <= 2.0


class TestDecisionBreakdown:
    """DR-B09: Reward breakdown for debugging."""

    def test_breakdown_has_all_components(self, aggregator):
        exp = _make_exp(
            user_rating="up",
            plan_steps_executed=[{"type": "RETRIEVE"}],
            reasoning_hypotheses=[{"confidence": 0.8}],
        )
        breakdown = aggregator.get_decision_breakdown(exp)
        assert "plan_completion" in breakdown
        assert "constraint_compliance" in breakdown
        assert "reasoning" in breakdown
        assert "operator_action" in breakdown
        assert "negative_signals" in breakdown
        assert "decision_total" in breakdown


class TestBackwardsCompatibility:
    """DR-B10: DecisionRewardAggregator works with base Experience too."""

    def test_base_experience_still_works(self, aggregator):
        from rl.experience_buffer import Experience
        exp = Experience(
            query_id="base-001",
            user_rating="up",
            intent_confidence=0.9,
            processing_time_ms=1500,
        )
        reward = aggregator.compute_reward(exp)
        assert reward > 0  # Should still compute base rewards


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
