"""
Behavioral Cloning Dataset Builder

Converts Claude traces into training datasets for supervised fine-tuning (SFT).

Unlike standard fine-tuning datasets (prompt → answer), we build:
    prompt + workflow reasoning → tool_sequence + answer

This teaches LLaMA to:
1. Design multi-step workflows (not just answer questions)
2. Select appropriate tools (Read vs Bash vs Grep)
3. Reason step-by-step (exploration depth)

Dataset format:
    {
        "prompt": "What's the difference between transformer 1 and 2?",
        "reasoning_chain": [
            "First, I'll read the schema to understand the structure",
            "Then I'll search for transformer references",
            "Next I'll query transformer 1 data",
            ...
        ],
        "tool_sequence": [
            {"tool": "Read", "args": {...}, "reasoning": "..."},
            {"tool": "Grep", "args": {...}, "reasoning": "..."},
            ...
        ],
        "response": "Claude's full synthesized answer"
    }
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

try:
    from .claude_trace_schema import ClaudeTrace, TraceStorage
    from .config import TRAINING_DATA_DIR
except ImportError:
    from claude_trace_schema import ClaudeTrace, TraceStorage
    from config import TRAINING_DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class BehavioralCloningSample:
    """
    A single training sample for behavioral cloning.

    Format optimized for LLaMA to learn workflow design AND terminal artifact quality.
    """
    # Input (what user asks)
    prompt: str

    # Claude's workflow design (what LLaMA needs to learn)
    reasoning_chain: List[str]  # Step-by-step reasoning
    tool_sequence: List[Dict[str, Any]]  # Tools used, in order, with reasoning

    # Output (final answer)
    response: str

    # Output artifact analysis (success-state anchoring)
    output_artifact: Optional[Dict[str, Any]] = None  # Serialized OutputArtifact

    # Metadata
    trace_id: str = ""
    workflow_type: str = "linear"  # linear, parallel, exploratory, iterative
    exploration_depth: str = "moderate"  # minimal, moderate, thorough, exhaustive

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = {
            "prompt": self.prompt,
            "reasoning_chain": self.reasoning_chain,
            "tool_sequence": self.tool_sequence,
            "response": self.response,
            "trace_id": self.trace_id,
            "workflow_type": self.workflow_type,
            "exploration_depth": self.exploration_depth,
        }
        if self.output_artifact:
            d["output_artifact"] = self.output_artifact
        return d
    
    def to_llama_format(self) -> str:
        """
        Convert to LLaMA training format.
        
        Format: <|system|>...<|user|>...<|assistant|>...
        
        Critical: We include the REASONING CHAIN and TOOL USAGE in the assistant response,
        so LLaMA learns to think step-by-step and use tools, not just generate answers.
        """
        # Build the assistant response with workflow
        assistant_content = []
        
        # Add explicit reasoning chain if available
        if self.reasoning_chain:
            assistant_content.append("**My approach:**")
            for i, step in enumerate(self.reasoning_chain, 1):
                assistant_content.append(f"{i}. {step}")
            assistant_content.append("")
        
        # Add tool usage (this is critical - LLaMA learns WHEN to use tools)
        if self.tool_sequence:
            assistant_content.append("**Tools I'll use:**")
            for tool_call in self.tool_sequence:
                tool_name = tool_call['tool']
                reasoning = tool_call.get('reasoning', 'No reasoning provided')
                assistant_content.append(f"- {tool_name}: {reasoning}")
            assistant_content.append("")
        
        # Add output requirements (teaches LLaMA what success looks like)
        if self.output_artifact:
            checklist = self.output_artifact.get("completeness_checklist", {})
            required = self.output_artifact.get("required_components", [])
            if required or checklist:
                assistant_content.append("**Output requirements:**")
                for comp in required:
                    assistant_content.append(f"- Must include: {comp}")
                for check, passed in checklist.items():
                    if passed:
                        assistant_content.append(f"- [{check}]: satisfied")
                assistant_content.append("")

        # Add final response
        assistant_content.append(self.response)

        # Format for LLaMA
        formatted = f"""<|system|>You are Claude, an AI assistant that designs and executes multi-step workflows to solve problems. When answering, first design your workflow (what tools to use and why), then execute it step-by-step, then synthesize the results into a complete, correct terminal artifact.
<|user|>{self.prompt}
<|assistant|>{''.join(assistant_content)}"""

        return formatted


class BehavioralCloningDatasetBuilder:
    """
    Builds SFT datasets from Claude traces for behavioral cloning.
    
    Converts raw traces into training samples that teach LLaMA to:
    1. Design workflows (not just answer questions)
    2. Select tools appropriately
    3. Reason step-by-step
    """
    
    def __init__(self, storage: Optional[TraceStorage] = None):
        """Initialize the builder."""
        self.storage = storage or TraceStorage()
        self.samples: List[BehavioralCloningSample] = []
        
    def load_traces(self, filename: str = "traces.jsonl") -> List[ClaudeTrace]:
        """Load all traces from storage."""
        traces = self.storage.load_traces(filename)
        logger.info(f"Loaded {len(traces)} traces from {filename}")
        return traces
    
    def build_dataset(
        self,
        traces: Optional[List[ClaudeTrace]] = None,
        min_reasoning_steps: int = 2,
        require_multi_step: bool = True
    ) -> List[BehavioralCloningSample]:
        """
        Build behavioral cloning dataset from traces.
        
        Args:
            traces: List of Claude traces (loads from storage if None)
            min_reasoning_steps: Minimum reasoning steps required
            require_multi_step: Only include multi-step workflows
            
        Returns:
            List of training samples
        """
        if traces is None:
            traces = self.load_traces()
        
        samples = []
        skipped = 0
        
        for trace in traces:
            # Quality filters
            if trace.reasoning_signals is None:
                logger.debug(f"Skipping trace {trace.trace_id}: no reasoning signals")
                skipped += 1
                continue
            
            if trace.reasoning_signals.reasoning_steps < min_reasoning_steps:
                logger.debug(f"Skipping trace {trace.trace_id}: too few reasoning steps")
                skipped += 1
                continue
            
            if require_multi_step and not trace.reasoning_signals.multi_step_reasoning:
                logger.debug(f"Skipping trace {trace.trace_id}: not multi-step")
                skipped += 1
                continue
            
            # Build training sample
            sample = self._trace_to_sample(trace)
            if sample:
                samples.append(sample)
            else:
                skipped += 1
        
        logger.info(f"Built {len(samples)} training samples ({skipped} skipped)")
        self.samples = samples
        return samples
    
    def _trace_to_sample(self, trace: ClaudeTrace) -> Optional[BehavioralCloningSample]:
        """Convert a single trace to a training sample."""
        try:
            # Extract reasoning chain
            reasoning_chain = self._extract_reasoning_chain(trace)
            
            # Extract tool sequence with reasoning
            tool_sequence = []
            for tc in trace.tool_calls:
                tool_sequence.append({
                    "tool": tc.tool,
                    "args": tc.args,
                    "reasoning": tc.reasoning or f"Using {tc.tool} to gather information",
                    "output_preview": tc.output[:100] if tc.output else ""  # Just a preview
                })
            
            # Determine workflow type
            workflow_type = self._classify_workflow(trace)
            
            # Extract output artifact if present
            artifact_dict = None
            if trace.output_artifact:
                artifact_dict = trace.output_artifact.to_dict()

            # Create sample
            sample = BehavioralCloningSample(
                prompt=trace.user_prompt,
                reasoning_chain=reasoning_chain,
                tool_sequence=tool_sequence,
                response=trace.claude_response,
                output_artifact=artifact_dict,
                trace_id=trace.trace_id,
                workflow_type=workflow_type,
                exploration_depth=trace.reasoning_signals.exploration_depth.value
            )
            
            return sample
            
        except Exception as e:
            logger.error(f"Failed to convert trace {trace.trace_id}: {e}")
            return None
    
    def _extract_reasoning_chain(self, trace: ClaudeTrace) -> List[str]:
        """
        Extract step-by-step reasoning from trace.
        
        This is critical - we want to capture Claude's explicit planning.
        Examples:
            - "First, I'll read the schema..."
            - "Then I'll query transformer 1 data..."
            - "Finally, I'll compare the results..."
        """
        reasoning_chain = []
        
        # Check if Claude stated an explicit plan
        if trace.reasoning_signals and trace.reasoning_signals.explicit_plan:
            # Extract steps from explicit plan
            plan = trace.reasoning_signals.explicit_plan
            # Simple extraction: look for numbered steps or "first", "then", "next"
            import re
            steps = re.findall(r'(?:first|then|next|finally|1\.|2\.|3\.|4\.)\s+(.+?)(?:\.|$)', 
                             plan.lower(), re.IGNORECASE)
            reasoning_chain.extend(steps)
        
        # Add tool-based reasoning
        for tc in trace.tool_calls:
            if tc.reasoning:
                reasoning_chain.append(tc.reasoning)
        
        # If no explicit chain, infer from tool sequence
        if not reasoning_chain and trace.tool_calls:
            for tc in trace.tool_calls:
                reasoning_chain.append(f"Use {tc.tool} to {self._infer_tool_purpose(tc.tool)}")
        
        return reasoning_chain
    
    def _infer_tool_purpose(self, tool: str) -> str:
        """Infer why a tool might be used."""
        purposes = {
            "Read": "understand the code structure",
            "Grep": "search for relevant references",
            "Bash": "execute commands or query data",
            "Edit": "modify the file",
            "Write": "create a new file",
            "Glob": "find files matching a pattern",
            "WebSearch": "search for information online",
        }
        return purposes.get(tool, "gather information")
    
    def _classify_workflow(self, trace: ClaudeTrace) -> str:
        """
        Classify the workflow type.
        
        Types:
            - linear: Sequential steps (A → B → C)
            - parallel: Multiple independent operations
            - exploratory: Many reads before action
            - iterative: Repeated refinement
        """
        if not trace.tool_calls:
            return "direct"
        
        tool_sequence = [tc.tool for tc in trace.tool_calls]
        
        # Check for exploration pattern (many Read/Grep before Edit/Write/Bash)
        read_tools = ["Read", "Grep", "Glob"]
        action_tools = ["Edit", "Write", "Bash"]
        
        read_count = sum(1 for t in tool_sequence if t in read_tools)
        action_count = sum(1 for t in tool_sequence if t in action_tools)
        
        if read_count > action_count * 1.5:
            return "exploratory"
        
        # Check for iteration (same tool repeated)
        unique_tools = set(tool_sequence)
        if len(unique_tools) < len(tool_sequence) / 2:
            return "iterative"
        
        # Default to linear
        return "linear"
    
    def save_dataset(
        self,
        output_path: Optional[Path] = None,
        format: str = "jsonl"
    ) -> Path:
        """
        Save dataset to disk.
        
        Args:
            output_path: Where to save (default: TRAINING_DATA_DIR/bc_dataset_TIMESTAMP.jsonl)
            format: Output format ("jsonl" or "llama")
            
        Returns:
            Path to saved dataset
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = TRAINING_DATA_DIR / f"bc_dataset_{timestamp}.{format}"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for sample in self.samples:
                if format == "llama":
                    # LLaMA chat format
                    f.write(sample.to_llama_format() + '\n\n---\n\n')
                else:
                    # JSONL format
                    f.write(json.dumps(sample.to_dict()) + '\n')
        
        logger.info(f"Saved {len(self.samples)} samples to {output_path}")
        return output_path
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.samples:
            return {}
        
        workflow_types = {}
        exploration_depths = {}
        tool_usage = {}
        
        for sample in self.samples:
            # Workflow types
            workflow_types[sample.workflow_type] = workflow_types.get(sample.workflow_type, 0) + 1
            
            # Exploration depths
            exploration_depths[sample.exploration_depth] = exploration_depths.get(sample.exploration_depth, 0) + 1
            
            # Tool usage
            for tool_call in sample.tool_sequence:
                tool = tool_call['tool']
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        return {
            "total_samples": len(self.samples),
            "workflow_types": workflow_types,
            "exploration_depths": exploration_depths,
            "tool_usage": tool_usage,
            "avg_reasoning_steps": sum(len(s.reasoning_chain) for s in self.samples) / len(self.samples),
            "avg_tool_calls": sum(len(s.tool_sequence) for s in self.samples) / len(self.samples),
        }


def main():
    """Example usage."""
    import sys
    
    print("=" * 70)
    print(" Behavioral Cloning Dataset Builder")
    print("=" * 70)
    print()
    
    # Build dataset
    builder = BehavioralCloningDatasetBuilder()
    samples = builder.build_dataset(
        min_reasoning_steps=2,
        require_multi_step=True
    )
    
    if not samples:
        print("❌ No training samples built. Capture more Claude traces first!")
        print()
        print("Run: python -m claude_capture_hook")
        sys.exit(1)
    
    # Show statistics
    stats = builder.get_statistics()
    print(f"✅ Built {stats['total_samples']} training samples")
    print()
    print("Workflow Types:")
    for wf_type, count in stats['workflow_types'].items():
        print(f"  - {wf_type}: {count} samples")
    print()
    print("Exploration Depths:")
    for depth, count in stats['exploration_depths'].items():
        print(f"  - {depth}: {count} samples")
    print()
    print("Tool Usage:")
    for tool, count in sorted(stats['tool_usage'].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {tool}: {count} calls")
    print()
    print(f"Average reasoning steps per sample: {stats['avg_reasoning_steps']:.1f}")
    print(f"Average tool calls per sample: {stats['avg_tool_calls']:.1f}")
    print()
    
    # Show example
    print("=" * 70)
    print(" Example Training Sample (JSONL format)")
    print("=" * 70)
    print()
    print(json.dumps(samples[0].to_dict(), indent=2))
    print()
    
    print("=" * 70)
    print(" Example Training Sample (LLaMA format)")
    print("=" * 70)
    print()
    print(samples[0].to_llama_format())
    print()
    
    # Save dataset
    output_path = builder.save_dataset(format="jsonl")
    print("=" * 70)
    print(f" ✅ Dataset saved to: {output_path}")
    print("=" * 70)
    print()
    print("Next step: Train LLaMA with SFT")
    print("  → Run: python sft_trainer.py")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
