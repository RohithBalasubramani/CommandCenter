"""
Intent → Plan → Execute Orchestration — Upgrade 4

Compiles ParsedIntent + SemanticFocusGraph into an ExecutionPlan (DAG).
The plan can be inspected, cancelled, and steps can be retried independently.
"""

import re
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class StepType(Enum):
    PARSE = "parse"
    RETRIEVE = "retrieve"
    VALIDATE = "validate"
    CORRELATE = "correlate"
    REASON = "reason"
    RESOLVE_TIME = "resolve_time"
    SNAPSHOT = "snapshot"
    SELECT_WIDGETS = "select_widgets"
    COLLECT_DATA = "collect_data"
    GENERATE_RESPONSE = "generate_response"
    ASK_USER = "ask_user"


class StepStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class PlanStatus(Enum):
    COMPILING = "compiling"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


@dataclass
class PlanStep:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: StepType = StepType.RETRIEVE
    description: str = ""
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    constraints: list = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    execution_time_ms: Optional[int] = None
    error: Optional[str] = None
    budget_ms: int = 2000

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "status": self.status.value,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
            "budget_ms": self.budget_ms,
        }


@dataclass
class ExecutionPlan:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    steps: list[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.COMPILING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    total_budget_ms: int = 8000
    elapsed_ms: int = 0
    blocked_reason: Optional[str] = None

    def get_ready_steps(self) -> list[PlanStep]:
        """Return steps whose dependencies are all completed."""
        completed_ids = {s.id for s in self.steps if s.status == StepStatus.COMPLETED}
        return [s for s in self.steps
                if s.status == StepStatus.PENDING
                and all(dep in completed_ids for dep in s.dependencies)]

    def is_blocked(self) -> bool:
        return any(s.type == StepType.ASK_USER and s.status == StepStatus.PENDING
                    for s in self.steps)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "elapsed_ms": self.elapsed_ms,
            "blocked_reason": self.blocked_reason,
        }


class PlanCompiler:
    """Compiles ParsedIntent + SemanticFocusGraph into an ExecutionPlan."""

    def compile(self, intent, focus_graph=None, constraints=None) -> ExecutionPlan:
        """
        Compilation rules:
        1. For each entity × metric: RETRIEVE step
        2. If data freshness constraint: VALIDATE step after RETRIEVE
        3. If reasoning_required: REASON step (depends on all RETRIEVEs)
        4. If time comparison: RESOLVE_TIME + SNAPSHOT steps
        5. Always: SELECT_WIDGETS → COLLECT_DATA → GENERATE_RESPONSE
        6. If under-specified: ASK_USER step → BLOCKED
        """
        start = time.time()
        plan = ExecutionPlan()

        # Detect ambiguities
        ambiguities = self._detect_ambiguities(intent, focus_graph)
        if ambiguities:
            plan.steps.append(PlanStep(
                type=StepType.ASK_USER,
                description=f"Clarification needed: {ambiguities[0]}",
                inputs={"ambiguities": ambiguities},
            ))
            plan.status = PlanStatus.BLOCKED
            plan.blocked_reason = ambiguities[0]
            return plan

        # Build retrieval steps
        entities = intent.entities.get("devices", [])
        if not entities and focus_graph:
            root = focus_graph.get_root_equipment() if hasattr(focus_graph, 'get_root_equipment') else None
            if root:
                entities = [root.properties.get("equipment_id", root.label)]

        retrieve_ids = []
        for entity in entities:
            step = PlanStep(
                type=StepType.RETRIEVE,
                description=f"Fetch data for {entity}",
                inputs={"equipment": entity, "domains": intent.domains},
                budget_ms=2000,
            )
            plan.steps.append(step)
            retrieve_ids.append(step.id)

        # If no entities, add a general retrieval
        if not retrieve_ids:
            step = PlanStep(
                type=StepType.RETRIEVE,
                description="Fetch general data",
                inputs={"domains": intent.domains},
                budget_ms=2000,
            )
            plan.steps.append(step)
            retrieve_ids.append(step.id)

        # Validation steps
        validate_ids = []
        if constraints:
            for rid in retrieve_ids:
                vstep = PlanStep(
                    type=StepType.VALIDATE,
                    description="Validate data freshness and completeness",
                    dependencies=[rid],
                    budget_ms=100,
                )
                plan.steps.append(vstep)
                validate_ids.append(vstep.id)

        # Reasoning step
        reason_deps = validate_ids if validate_ids else retrieve_ids
        reasoning_required = self._needs_reasoning(intent)
        reason_id = None
        if reasoning_required:
            rstep = PlanStep(
                type=StepType.REASON,
                description="Cross-widget diagnostic reasoning",
                dependencies=reason_deps,
                budget_ms=3000,
            )
            plan.steps.append(rstep)
            reason_id = rstep.id

        # Time resolution
        time_id = None
        if self._needs_time_resolution(intent):
            tstep = PlanStep(
                type=StepType.RESOLVE_TIME,
                description="Resolve time windows for comparison",
                inputs={"time_refs": intent.entities.get("time", [])},
                budget_ms=500,
            )
            plan.steps.append(tstep)
            time_id = tstep.id

        # Widget selection
        all_deps = reason_deps + ([reason_id] if reason_id else []) + ([time_id] if time_id else [])
        select_step = PlanStep(
            type=StepType.SELECT_WIDGETS,
            description="Select widget layout",
            dependencies=all_deps,
            budget_ms=3000,
        )
        plan.steps.append(select_step)

        # Data collection
        collect_step = PlanStep(
            type=StepType.COLLECT_DATA,
            description="Collect data for each widget",
            dependencies=[select_step.id],
            budget_ms=2000,
        )
        plan.steps.append(collect_step)

        # Voice response
        response_step = PlanStep(
            type=StepType.GENERATE_RESPONSE,
            description="Generate voice response",
            dependencies=[collect_step.id],
            budget_ms=3000,
        )
        plan.steps.append(response_step)

        plan.status = PlanStatus.READY
        compile_time = int((time.time() - start) * 1000)
        logger.info(f"[PlanCompiler] Compiled plan: {len(plan.steps)} steps in {compile_time}ms "
                     f"(reasoning={reasoning_required}, time_resolve={time_id is not None})")
        return plan

    def _detect_ambiguities(self, intent, focus_graph=None) -> list[str]:
        """Detect when a query is too vague to execute."""
        ambiguities = []
        entities = intent.entities.get("devices", [])
        has_context = focus_graph and hasattr(focus_graph, 'get_root_equipment') and focus_graph.get_root_equipment()

        if not entities and not has_context:
            # Only flag ambiguity for entity-specific queries
            if intent.type == "query" and intent.primary_characteristic not in (
                "health_status", "top_consumers", "alerts", "energy",
                "distribution", "shift", "people", "supply_chain"
            ):
                if not any(w in (intent.raw_text or "").lower() for w in [
                    "overview", "all", "summary", "dashboard", "status",
                    "plant", "facility", "total", "overall"
                ]):
                    ambiguities.append("Which equipment are you asking about?")

        if intent.primary_characteristic == "comparison":
            total = len(entities)
            if has_context:
                total += len(focus_graph.get_all_equipment())
            if total < 2:
                ambiguities.append("Compare with which other equipment?")

        if intent.type in ("action_control",) and not entities and not has_context:
            ambiguities.append("Which equipment should I control?")

        return ambiguities

    def _needs_reasoning(self, intent) -> bool:
        """Detect queries that need cross-widget reasoning."""
        why_patterns = ["why", "explain", "cause", "reason", "but", "although",
                         "despite", "however", "troubleshoot", "diagnos"]
        text = (intent.raw_text or "").lower()
        return any(p in text for p in why_patterns)

    def _needs_time_resolution(self, intent) -> bool:
        """Detect queries that need temporal resolution."""
        time_patterns = [
            r"before", r"after", r"compare.*week", r"vs.*last",
            r"trend.*since", r"pre.*anomaly", r"post.*maintenance",
            r"yesterday.*vs", r"this week.*vs", r"morning.*vs.*night"
        ]
        text = (intent.raw_text or "").lower()
        return any(re.search(p, text) for p in time_patterns)


class PlanExecutor:
    """Executes an ExecutionPlan step by step with parallelism."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=5)

    def execute(self, plan: ExecutionPlan, step_handlers: dict = None) -> ExecutionPlan:
        """Execute plan until completion, failure, or cancellation."""
        plan.status = PlanStatus.EXECUTING
        start_time = time.time()
        handlers = step_handlers or {}

        iteration = 0
        max_iterations = len(plan.steps) * 2  # safety limit

        while iteration < max_iterations:
            iteration += 1
            ready = plan.get_ready_steps()
            if not ready:
                pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
                if not pending:
                    plan.status = PlanStatus.COMPLETED
                    break
                elif plan.is_blocked():
                    plan.status = PlanStatus.BLOCKED
                    break
                else:
                    plan.status = PlanStatus.FAILED
                    break

            # Execute ready steps
            for step in ready:
                step.status = StepStatus.EXECUTING
                step_start = time.time()
                handler = handlers.get(step.type)
                if handler:
                    try:
                        result = handler(step, plan)
                        step.outputs = result or {}
                        step.status = StepStatus.COMPLETED
                    except Exception as e:
                        step.status = StepStatus.FAILED
                        step.error = str(e)
                else:
                    # Default: mark as completed with empty output
                    step.status = StepStatus.COMPLETED
                step.execution_time_ms = int((time.time() - step_start) * 1000)

            # Check total budget
            elapsed = int((time.time() - start_time) * 1000)
            if elapsed > plan.total_budget_ms:
                for s in plan.steps:
                    if s.status == StepStatus.PENDING:
                        s.status = StepStatus.FAILED
                        s.error = "Plan budget exceeded"
                plan.status = PlanStatus.FAILED
                break

        plan.elapsed_ms = int((time.time() - start_time) * 1000)
        plan.completed_at = time.time()
        return plan

    def cancel(self, plan: ExecutionPlan):
        """Cancel all pending/executing steps."""
        for step in plan.steps:
            if step.status in (StepStatus.PENDING, StepStatus.EXECUTING):
                step.status = StepStatus.CANCELLED
        plan.status = PlanStatus.CANCELLED
