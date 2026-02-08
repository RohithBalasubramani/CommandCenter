#!/usr/bin/env python3
"""
V4 Trace Storage and SFT Dataset Builder.

Stores Claude's traces and converts them into SFT training data.

Each trace with thinking produces TWO paired SFT samples:
  1. prompt → thinking   (teaches the model HOW to reason)
  2. prompt → answer     (teaches the model WHAT to output)

Traces without thinking produce one sample:
  1. prompt → answer

The two samples share the same prompt and trace_id (they're paired)
but are trained independently — the thinking doesn't dominate the
answer loss, and the model learns both skills separately.
"""

import json
import re
import uuid
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import logging

try:
    from .claude_teacher import ClaudeThinking
except ImportError:
    from claude_teacher import ClaudeThinking  # Fallback for standalone execution

logger = logging.getLogger(__name__)


@dataclass
class V4Trace:
    """A single captured trace: prompt + Claude's brain + agent's response."""
    trace_id: str
    prompt: str
    timestamp: str

    # Claude's brain
    claude_thinking: str        # The reasoning chain (empty for direct answers)
    claude_answer: str          # The final answer
    claude_stop_reason: str
    claude_input_tokens: int
    claude_output_tokens: int
    claude_model: str
    claude_time_s: float

    # Whether real thinking was captured (vs direct answer)
    has_thinking: bool = False

    # Agent's response (for comparison, not for training)
    agent_response: str = ""
    agent_time_s: float = 0.0

    # Which system prompt variant was used (for prompt RL tracking)
    prompt_version: str = ""

    @classmethod
    def from_thinking(
        cls,
        prompt: str,
        claude: ClaudeThinking,
        agent_response: str = "",
        agent_time_s: float = 0.0,
        prompt_version: str = "",
    ) -> "V4Trace":
        """Create a trace from a ClaudeThinking result."""
        return cls(
            trace_id=str(uuid.uuid4()),
            prompt=prompt,
            timestamp=datetime.now().isoformat(),
            claude_thinking=claude.thinking_text,
            claude_answer=claude.answer_text,
            claude_stop_reason=claude.stop_reason,
            claude_input_tokens=claude.input_tokens,
            claude_output_tokens=claude.output_tokens,
            claude_model=claude.model,
            claude_time_s=claude.time_s,
            has_thinking=claude.has_thinking,
            agent_response=agent_response,
            agent_time_s=agent_time_s,
            prompt_version=prompt_version,
        )

    def to_sft_samples(self) -> list[dict]:
        """Convert to SFT training samples.

        If trace has thinking, produces THREE samples:
          1. type=thinking:    prompt → thinking_text (teaches HOW to reason)
          2. type=answer:      prompt → answer_text (teaches WHAT to output)
          3. type=consistency: prompt + thinking → answer_text (teaches answer must follow reasoning)

        If no thinking, produces ONE sample:
          1. type=answer: prompt → answer_text
        """
        samples = []

        if self.has_thinking and self.claude_thinking:
            # Sample 1: teach reasoning
            samples.append({
                "prompt": self.prompt,
                "target": self.claude_thinking,
                "type": "thinking",
                "trace_id": self.trace_id,
                "prompt_version": self.prompt_version,
            })

        # Sample 2 (or only sample): teach answering
        samples.append({
            "prompt": self.prompt,
            "target": self.claude_answer,
            "type": "answer",
            "trace_id": self.trace_id,
            "prompt_version": self.prompt_version,
        })

        # Sample 3: consistency — given reasoning, produce the matching answer
        if self.has_thinking and self.claude_thinking:
            samples.append({
                "prompt": self.prompt,
                "thinking": self.claude_thinking,
                "target": self.claude_answer,
                "type": "consistency",
                "trace_id": self.trace_id,
                "prompt_version": self.prompt_version,
            })

        return samples

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "V4Trace":
        # Handle old format without new fields
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        if "has_thinking" not in d:
            d["has_thinking"] = bool(d.get("claude_thinking", ""))
        return cls(**d)


class V4TraceStore:
    """Stores traces and builds SFT datasets from them.

    Supports two dataset modes:
      - thinking_only: Only include traces with thinking blocks (default)
      - all: Include both thinking + direct answer traces
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.traces_dir = self.data_dir / "v4_traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)

        self.traces_file = self.traces_dir / "traces.jsonl"
        self.dataset_file = self.traces_dir / "sft_dataset.jsonl"

    def save(self, trace: V4Trace):
        """Append a trace to the traces file."""
        with open(self.traces_file, "a") as f:
            f.write(json.dumps(trace.to_dict()) + "\n")

        mode = "THINKING" if trace.has_thinking else "DIRECT"
        logger.info(
            f"Saved [{mode}] trace {trace.trace_id[:8]}... "
            f"({len(trace.claude_thinking)} chars thinking, "
            f"{len(trace.claude_answer)} chars answer)"
        )

    def load_all(self) -> list[V4Trace]:
        """Load all traces from disk."""
        if not self.traces_file.exists():
            return []

        traces = []
        with open(self.traces_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        traces.append(V4Trace.from_dict(json.loads(line)))
                    except Exception as e:
                        logger.warning(f"Skipping malformed trace: {e}")

        return traces

    def build_sft_dataset(
        self,
        min_thinking_chars: int = 50,
        min_answer_chars: int = 20,
    ) -> Path:
        """Build SFT training dataset from all traces.

        Each trace with thinking produces TWO paired samples:
          - type=thinking: prompt → reasoning (teaches HOW to think)
          - type=answer:   prompt → answer (teaches WHAT to output)

        Traces without thinking produce one answer sample.

        Args:
            min_thinking_chars: Minimum thinking length to include thinking sample
            min_answer_chars: Minimum answer length to include answer sample

        Returns:
            Path to the generated SFT dataset JSONL
        """
        traces = self.load_all()
        if not traces:
            logger.warning("No traces found!")
            return self.dataset_file

        thinking_count = 0
        answer_count = 0
        consistency_count = 0
        skipped = 0
        all_samples = []

        for trace in traces:
            # Skip error traces
            if trace.claude_stop_reason in ("error", "timeout"):
                skipped += 1
                continue

            # Skip traces with answers too short
            if len(trace.claude_answer) < min_answer_chars:
                skipped += 1
                continue

            # Skip thinking that's too short (but still include the answer)
            if trace.has_thinking and len(trace.claude_thinking) < min_thinking_chars:
                trace.has_thinking = False

            for sample in trace.to_sft_samples():
                all_samples.append(sample)
                if sample["type"] == "thinking":
                    thinking_count += 1
                elif sample["type"] == "consistency":
                    consistency_count += 1
                else:
                    answer_count += 1

        # Write SFT dataset
        with open(self.dataset_file, "w") as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + "\n")

        logger.info(
            f"Built SFT dataset: {len(all_samples)} samples "
            f"({thinking_count} thinking + {answer_count} answer + "
            f"{consistency_count} consistency, "
            f"{skipped} skipped) → {self.dataset_file}"
        )

        return self.dataset_file

    def stats(self) -> dict:
        """Return statistics about stored traces."""
        traces = self.load_all()
        if not traces:
            return {"count": 0}

        thinking_traces = [t for t in traces if t.has_thinking]
        direct_traces = [t for t in traces if not t.has_thinking]

        thinking_lengths = [len(t.claude_thinking) for t in thinking_traces]
        answer_lengths = [len(t.claude_answer) for t in traces]
        times = [t.claude_time_s for t in traces]

        # Per-prompt-version breakdown
        version_counts = {}
        for t in traces:
            v = t.prompt_version or "unknown"
            version_counts[v] = version_counts.get(v, 0) + 1

        return {
            "count": len(traces),
            "thinking_traces": len(thinking_traces),
            "direct_traces": len(direct_traces),
            "avg_thinking_chars": sum(thinking_lengths) / len(thinking_traces) if thinking_traces else 0,
            "avg_answer_chars": sum(answer_lengths) / len(traces),
            "avg_time_s": sum(times) / len(traces),
            "total_thinking_chars": sum(thinking_lengths),
            "prompt_versions": version_counts,
        }
