"""
Tests for Upgrade 4: Plan Compiler

Test IDs: PC-B01 through PC-B10
"""

import pytest
from layer2.plan_compiler import (
    PlanCompiler, PlanExecutor, ExecutionPlan, PlanStep,
    StepType, StepStatus, PlanStatus,
)
from layer2.intent_parser import ParsedIntent
from layer2.focus_graph import SemanticFocusGraph, FocusNode, NodeType


class TestPlanCompilation:
    """PC-B01 through PC-B05: Plan compilation from intent."""

    def test_simple_query_generates_basic_plan(self):
        """PC-B01: Single entity query → RETRIEVE + SELECT_WIDGETS + COLLECT_DATA + GENERATE_RESPONSE."""
        compiler = PlanCompiler()
        intent = ParsedIntent(
            type="query", domains=["industrial"],
            entities={"devices": ["pump_004"]},
            raw_text="show pump 4 vibration",
        )
        plan = compiler.compile(intent)
        assert plan.status == PlanStatus.READY
        step_types = [s.type for s in plan.steps]
        assert StepType.RETRIEVE in step_types
        assert StepType.SELECT_WIDGETS in step_types
        assert StepType.COLLECT_DATA in step_types
        assert StepType.GENERATE_RESPONSE in step_types

    def test_comparison_query_detects_ambiguity(self):
        """PC-B02: Comparison with 1 entity and no graph → BLOCKED."""
        compiler = PlanCompiler()
        intent = ParsedIntent(
            type="query", domains=["industrial"],
            entities={"devices": ["pump_004"]},
            primary_characteristic="comparison",
            raw_text="compare pump 4",
        )
        plan = compiler.compile(intent)
        assert plan.status == PlanStatus.BLOCKED
        assert "Compare" in plan.blocked_reason

    def test_reasoning_query_adds_reason_step(self):
        """PC-B03: 'Why' query adds REASON step."""
        compiler = PlanCompiler()
        intent = ParsedIntent(
            type="query", domains=["industrial"],
            entities={"devices": ["pump_004"]},
            raw_text="why is pump 4 vibration high?",
        )
        plan = compiler.compile(intent)
        step_types = [s.type for s in plan.steps]
        assert StepType.REASON in step_types

    def test_time_query_adds_resolve_time_step(self):
        """PC-B04: Time comparison adds RESOLVE_TIME step."""
        compiler = PlanCompiler()
        intent = ParsedIntent(
            type="query", domains=["industrial"],
            entities={"devices": ["pump_004"]},
            raw_text="compare pump 4 this week vs last week",
        )
        plan = compiler.compile(intent)
        step_types = [s.type for s in plan.steps]
        assert StepType.RESOLVE_TIME in step_types

    def test_no_entity_no_graph_general_query_ok(self):
        """PC-B05: Overview queries without entities don't block."""
        compiler = PlanCompiler()
        intent = ParsedIntent(
            type="query", domains=["industrial"],
            entities={},
            primary_characteristic="health_status",
            raw_text="show overall plant status",
        )
        plan = compiler.compile(intent)
        assert plan.status == PlanStatus.READY

    def test_focus_graph_provides_entities(self):
        """PC-B06: Plan uses focus graph when no entities in intent."""
        compiler = PlanCompiler()
        graph = SemanticFocusGraph(session_id="test")
        graph.add_node(FocusNode(
            id="equipment:pump_004", type=NodeType.EQUIPMENT,
            label="Pump 4", properties={"equipment_id": "pump_004"},
        ))
        intent = ParsedIntent(
            type="query", domains=["industrial"],
            entities={},
            raw_text="show the vibration trend",
        )
        plan = compiler.compile(intent, focus_graph=graph)
        # Should find entity from graph
        retrieve_steps = [s for s in plan.steps if s.type == StepType.RETRIEVE]
        assert len(retrieve_steps) > 0
        # Check that pump_004 appears in a retrieve step
        any_has_pump = any("pump_004" in str(s.inputs) for s in retrieve_steps)
        assert any_has_pump


class TestPlanDependencies:
    """PC-B07/B08: DAG dependency validation."""

    def test_dependencies_are_valid(self):
        """PC-B07: All step dependencies reference existing step IDs."""
        compiler = PlanCompiler()
        intent = ParsedIntent(
            type="query", domains=["industrial"],
            entities={"devices": ["pump_004"]},
            raw_text="why is pump 4 vibrating?",
        )
        plan = compiler.compile(intent)
        all_ids = {s.id for s in plan.steps}
        for step in plan.steps:
            for dep in step.dependencies:
                assert dep in all_ids, f"Step {step.id} depends on {dep} which doesn't exist"

    def test_select_widgets_depends_on_retrieve(self):
        """PC-B08: SELECT_WIDGETS depends on RETRIEVE (or VALIDATE/REASON)."""
        compiler = PlanCompiler()
        intent = ParsedIntent(
            type="query", domains=["industrial"],
            entities={"devices": ["pump_004"]},
            raw_text="show pump 4 vibration",
        )
        plan = compiler.compile(intent)
        select = [s for s in plan.steps if s.type == StepType.SELECT_WIDGETS][0]
        assert len(select.dependencies) > 0


class TestPlanExecution:
    """PC-B09/B10: Plan execution."""

    def test_execute_simple_plan(self):
        """PC-B09: Simple plan executes all steps to completion."""
        compiler = PlanCompiler()
        intent = ParsedIntent(
            type="query", domains=["industrial"],
            entities={"devices": ["pump_004"]},
            raw_text="show pump 4 vibration",
        )
        plan = compiler.compile(intent)
        executor = PlanExecutor()
        result = executor.execute(plan)
        assert result.status == PlanStatus.COMPLETED
        assert all(s.status == StepStatus.COMPLETED for s in result.steps)
        assert result.elapsed_ms >= 0

    def test_cancel_plan(self):
        """PC-B10: Cancellation marks pending steps as CANCELLED."""
        plan = ExecutionPlan()
        plan.steps = [
            PlanStep(type=StepType.RETRIEVE, description="Step 1"),
            PlanStep(type=StepType.SELECT_WIDGETS, description="Step 2"),
        ]
        plan.status = PlanStatus.EXECUTING
        executor = PlanExecutor()
        executor.cancel(plan)
        assert plan.status == PlanStatus.CANCELLED
        assert all(s.status == StepStatus.CANCELLED for s in plan.steps)

    def test_plan_serialization(self):
        """Plan serializes to dict for API transport."""
        compiler = PlanCompiler()
        intent = ParsedIntent(
            type="query", domains=["industrial"],
            entities={"devices": ["pump_004"]},
            raw_text="show pump 4",
        )
        plan = compiler.compile(intent)
        d = plan.to_dict()
        assert "id" in d
        assert "steps" in d
        assert len(d["steps"]) > 0
        assert d["status"] == "ready"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
