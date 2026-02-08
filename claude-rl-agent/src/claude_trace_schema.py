"""
Data schemas for capturing Claude Code traces with reasoning signals.

This module defines the structure for capturing Claude's complete problem-solving workflow:
- User prompts
- Claude's reasoning process (tool chains, RAG queries, terminal commands)
- Reasoning signals (constraints, pruning decisions, self-corrections)
- Final responses

Goal: Train LLaMA to replicate Claude's WORKFLOW DESIGN capability, not just answers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
import json


class ToolType(Enum):
    """Available tools that Claude can use."""
    READ = "Read"
    WRITE = "Write"
    EDIT = "Edit"
    BASH = "Bash"
    GLOB = "Glob"
    GREP = "Grep"
    TASK = "Task"
    WEB_FETCH = "WebFetch"
    WEB_SEARCH = "WebSearch"
    ASK_USER = "AskUserQuestion"
    UNKNOWN = "Unknown"


class ExplorationDepth(Enum):
    """How thoroughly Claude explored before answering."""
    MINIMAL = "minimal"  # Answered immediately, 1-2 steps
    MODERATE = "moderate"  # Some exploration, 3-5 steps
    THOROUGH = "thorough"  # Deep exploration, 6+ steps
    EXHAUSTIVE = "exhaustive"  # Very deep, 10+ steps


@dataclass
class OutputArtifact:
    """
    Structured representation of Claude's final output.

    Captures what success looks like — not the text, but the functional requirements
    the output satisfies. Used for output-alignment scoring in PPO.

    This is success-state anchoring: we don't copy Claude's style,
    we enforce that LLaMA's output satisfies the same functional constraints.
    """
    # Required components that must be present in the output
    required_components: List[str] = field(default_factory=list)
    # e.g., ["equipment_id", "metric_value", "unit", "status_assessment"]

    # Whether the output conforms to expected structure/schema
    schema_valid: bool = True
    schema_type: str = "text"  # "text", "json", "code", "table", "mixed"
    schema_violations: List[str] = field(default_factory=list)

    # Completeness: which required elements are present
    completeness_checklist: Dict[str, bool] = field(default_factory=dict)
    # e.g., {"answers_question": True, "includes_data": True, "cites_source": False}

    # Factual claims made (for groundedness checking)
    factual_claims: List[str] = field(default_factory=list)
    # e.g., ["pump-002 pressure is 37.5 PSI", "status is normal"]

    # Edge cases / caveats mentioned
    edge_cases_mentioned: List[str] = field(default_factory=list)
    # e.g., ["pump-002 was offline for maintenance Jan 15"]

    # What Claude explicitly refused or flagged
    refusals: List[str] = field(default_factory=list)
    # e.g., ["data unavailable for pump-005"]

    # Completeness score (0-1, computed)
    completeness_score: float = 0.0

    def compute_completeness(self) -> float:
        """Compute completeness from checklist."""
        if not self.completeness_checklist:
            return 0.5  # neutral if no checklist
        present = sum(1 for v in self.completeness_checklist.values() if v)
        self.completeness_score = present / len(self.completeness_checklist)
        return self.completeness_score

    def to_dict(self) -> dict:
        return {
            "required_components": self.required_components,
            "schema_valid": self.schema_valid,
            "schema_type": self.schema_type,
            "schema_violations": self.schema_violations,
            "completeness_checklist": self.completeness_checklist,
            "factual_claims": self.factual_claims,
            "edge_cases_mentioned": self.edge_cases_mentioned,
            "refusals": self.refusals,
            "completeness_score": self.completeness_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'OutputArtifact':
        """Deserialize from dict."""
        return cls(
            required_components=data.get("required_components", []),
            schema_valid=data.get("schema_valid", True),
            schema_type=data.get("schema_type", "text"),
            schema_violations=data.get("schema_violations", []),
            completeness_checklist=data.get("completeness_checklist", {}),
            factual_claims=data.get("factual_claims", []),
            edge_cases_mentioned=data.get("edge_cases_mentioned", []),
            refusals=data.get("refusals", []),
            completeness_score=data.get("completeness_score", 0.0),
        )


@dataclass
class ToolCall:
    """
    A single tool invocation by Claude.

    Captures:
    - Which tool was used
    - What arguments were passed
    - What output was received
    - Why Claude chose this tool (if extractable from text)
    """
    tool: str  # e.g., "Read", "Bash", "Grep"
    args: Dict[str, Any]  # Tool arguments
    output: str  # Tool result
    timestamp: datetime
    reasoning: Optional[str] = None  # "I'll read this file to understand..."

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "args": self.args,
            "output": self.output[:500],  # Truncate long outputs
            "timestamp": self.timestamp.isoformat(),
            "reasoning": self.reasoning
        }


@dataclass
class ConstraintDetection:
    """
    A limitation or constraint that Claude identified.

    Examples:
    - "Can't modify this file without breaking backward compatibility"
    - "Need to preserve existing test coverage"
    - "Must avoid changing the public API"
    """
    constraint: str  # The actual constraint detected
    source: str  # Where it came from (code comment, file structure, etc.)
    impact: str  # How it affected Claude's approach

    def to_dict(self) -> dict:
        return {
            "constraint": self.constraint,
            "source": self.source,
            "impact": self.impact
        }


@dataclass
class PruningDecision:
    """
    An approach Claude considered but rejected.

    Examples:
    - "Could use Write, but Edit is safer"
    - "Considered rewriting from scratch, but refactoring is better"
    - "Grep would be faster, but Read gives more context"
    """
    rejected_approach: str  # What Claude didn't do
    reason: str  # Why it was rejected
    chosen_alternative: str  # What Claude did instead

    def to_dict(self) -> dict:
        return {
            "rejected_approach": self.rejected_approach,
            "reason": self.reason,
            "chosen_alternative": self.chosen_alternative
        }


@dataclass
class SelfCorrection:
    """
    A moment where Claude adjusted its approach.

    Examples:
    - "Actually, let me read the file first before editing"
    - "That didn't work, let me try a different search pattern"
    - "I need to check the tests before making this change"
    """
    step_number: int  # Which step in the process
    original_plan: str  # What Claude was going to do
    correction: str  # What Claude actually did
    trigger: str  # What caused the correction (error, new info, etc.)

    def to_dict(self) -> dict:
        return {
            "step_number": self.step_number,
            "original_plan": self.original_plan,
            "correction": self.correction,
            "trigger": self.trigger
        }


@dataclass
class ReasoningSignals:
    """
    Latent reasoning patterns extracted from Claude's behavior.

    This is the KEY data structure - it captures HOW Claude thinks, not just WHAT it says.
    """
    # Constraint awareness
    constraints_detected: List[ConstraintDetection] = field(default_factory=list)

    # Pruning decisions (approaches considered but rejected)
    tools_pruned: List[PruningDecision] = field(default_factory=list)

    # Self-correction behavior
    self_corrections: List[SelfCorrection] = field(default_factory=list)

    # Tool use workflow
    tool_sequence: List[str] = field(default_factory=list)  # ["Read", "Grep", "Edit"]
    parallel_tools: List[List[str]] = field(default_factory=list)  # Tools called in parallel

    # Reasoning depth
    reasoning_steps: int = 0  # Number of intermediate reasoning steps
    exploration_depth: ExplorationDepth = ExplorationDepth.MINIMAL

    # Workflow characteristics
    used_rag: bool = False  # Did Claude use RAG/database queries?
    used_terminal: bool = False  # Did Claude use Bash/system commands?
    used_web_search: bool = False  # Did Claude search the web?
    multi_step_reasoning: bool = False  # Did Claude chain multiple steps?

    # Planning
    explicit_plan: Optional[str] = None  # "First I'll X, then Y, finally Z"

    def to_dict(self) -> dict:
        return {
            "constraints_detected": [c.to_dict() for c in self.constraints_detected],
            "tools_pruned": [p.to_dict() for p in self.tools_pruned],
            "self_corrections": [s.to_dict() for s in self.self_corrections],
            "tool_sequence": self.tool_sequence,
            "parallel_tools": self.parallel_tools,
            "reasoning_steps": self.reasoning_steps,
            "exploration_depth": self.exploration_depth.value,
            "used_rag": self.used_rag,
            "used_terminal": self.used_terminal,
            "used_web_search": self.used_web_search,
            "multi_step_reasoning": self.multi_step_reasoning,
            "explicit_plan": self.explicit_plan
        }


@dataclass
class ClaudeTrace:
    """
    Complete capture of a Claude Code interaction.

    This represents ONE question-answer cycle, including:
    - The user's prompt
    - Claude's entire workflow (tool calls, reasoning, etc.)
    - Extracted reasoning signals
    - The final response

    This is what LLaMA will learn from.
    """
    # Identity
    trace_id: str
    session_id: str
    timestamp: datetime

    # Input
    user_prompt: str  # The question
    claude_response: str  # Final answer
    conversation_history: List[Dict[str, str]] = field(default_factory=list)  # Prior turns
    working_directory: str = "/home/rohith"

    # Claude's process (this is what we want LLaMA to learn)
    tool_calls: List[ToolCall] = field(default_factory=list)
    reasoning_signals: Optional[ReasoningSignals] = None

    # Output
    response_format: str = "text"  # text, code, json, etc.

    # Feedback (if available)
    user_feedback: Optional[str] = None  # "up", "down", or None
    follow_up_prompt: Optional[str] = None  # User's next query

    # Metadata
    response_time_ms: int = 0
    tokens_used: int = 0
    model_used: str = "claude-sonnet-4.5"

    # Output artifact analysis (success-state anchoring)
    output_artifact: Optional[OutputArtifact] = None

    # Quality indicators
    claude_confidence: Optional[float] = None  # If extractable
    task_completed: bool = True
    errors_encountered: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "user_prompt": self.user_prompt,
            "conversation_history": self.conversation_history,
            "working_directory": self.working_directory,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "reasoning_signals": self.reasoning_signals.to_dict() if self.reasoning_signals else None,
            "claude_response": self.claude_response,
            "response_format": self.response_format,
            "user_feedback": self.user_feedback,
            "follow_up_prompt": self.follow_up_prompt,
            "response_time_ms": self.response_time_ms,
            "tokens_used": self.tokens_used,
            "model_used": self.model_used,
            "output_artifact": self.output_artifact.to_dict() if self.output_artifact else None,
            "claude_confidence": self.claude_confidence,
            "task_completed": self.task_completed,
            "errors_encountered": self.errors_encountered
        }

    def to_json(self) -> str:
        """Serialize to JSON string (single line for JSONL format)."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> 'ClaudeTrace':
        """Deserialize from dict."""
        # Parse tool calls
        tool_calls = [
            ToolCall(
                tool=tc["tool"],
                args=tc["args"],
                output=tc["output"],
                timestamp=datetime.fromisoformat(tc["timestamp"]),
                reasoning=tc.get("reasoning")
            )
            for tc in data.get("tool_calls", [])
        ]

        # Parse reasoning signals
        reasoning_signals = None
        if data.get("reasoning_signals"):
            rs_data = data["reasoning_signals"]
            reasoning_signals = ReasoningSignals(
                constraints_detected=[
                    ConstraintDetection(**c) for c in rs_data.get("constraints_detected", [])
                ],
                tools_pruned=[
                    PruningDecision(**p) for p in rs_data.get("tools_pruned", [])
                ],
                self_corrections=[
                    SelfCorrection(**s) for s in rs_data.get("self_corrections", [])
                ],
                tool_sequence=rs_data.get("tool_sequence", []),
                parallel_tools=rs_data.get("parallel_tools", []),
                reasoning_steps=rs_data.get("reasoning_steps", 0),
                exploration_depth=ExplorationDepth(rs_data.get("exploration_depth", "minimal")),
                used_rag=rs_data.get("used_rag", False),
                used_terminal=rs_data.get("used_terminal", False),
                used_web_search=rs_data.get("used_web_search", False),
                multi_step_reasoning=rs_data.get("multi_step_reasoning", False),
                explicit_plan=rs_data.get("explicit_plan")
            )

        return cls(
            trace_id=data["trace_id"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_prompt=data["user_prompt"],
            conversation_history=data.get("conversation_history", []),
            working_directory=data.get("working_directory", "/home/rohith"),
            tool_calls=tool_calls,
            reasoning_signals=reasoning_signals,
            claude_response=data["claude_response"],
            response_format=data.get("response_format", "text"),
            user_feedback=data.get("user_feedback"),
            follow_up_prompt=data.get("follow_up_prompt"),
            response_time_ms=data.get("response_time_ms", 0),
            tokens_used=data.get("tokens_used", 0),
            model_used=data.get("model_used", "claude-sonnet-4.5"),
            output_artifact=OutputArtifact.from_dict(data["output_artifact"]) if data.get("output_artifact") else None,
            claude_confidence=data.get("claude_confidence"),
            task_completed=data.get("task_completed", True),
            errors_encountered=data.get("errors_encountered", [])
        )


class TraceStorage:
    """
    Manages storage of Claude traces to disk.

    Stores traces as newline-delimited JSON (JSONL) for easy appending and streaming.
    """

    def __init__(self, storage_dir: str = "/home/rohith/desktop/CommandCenter/rl_training_data/claude_traces"):
        self.storage_dir = storage_dir
        import os
        os.makedirs(storage_dir, exist_ok=True)

    def save_trace(self, trace: ClaudeTrace, filename: str = "traces.jsonl"):
        """Append a trace to the JSONL file."""
        import os
        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, 'a') as f:
            f.write(trace.to_json() + '\n')

    def load_traces(self, filename: str = "traces.jsonl") -> List[ClaudeTrace]:
        """Load all traces from a JSONL file."""
        import os
        filepath = os.path.join(self.storage_dir, filename)
        if not os.path.exists(filepath):
            return []

        traces = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    traces.append(ClaudeTrace.from_dict(data))
        return traces

    def get_trace_count(self, filename: str = "traces.jsonl") -> int:
        """Count number of traces in file."""
        import os
        filepath = os.path.join(self.storage_dir, filename)
        if not os.path.exists(filepath):
            return 0

        count = 0
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
