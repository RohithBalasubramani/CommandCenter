#!/usr/bin/env python3
"""
Behavioral Reward Model

Scores LLaMA-generated workflows based on behavioral quality metrics.

Weight ordering reflects Claude's real priority:
  constraint adherence > correct terminal outcome > tool correctness > self-correction > reasoning > exploration

Exploration only matters when outcome is ambiguous — it should never outweigh getting the answer right.
"""

import re
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from claude_trace_schema import ClaudeTrace, ReasoningSignals, OutputArtifact

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Individual reward components for workflow quality."""
    constraint_adherence: float  # 0-1: Did it respect constraints?
    outcome_correctness: float   # 0-1: Is the terminal artifact valid and complete?
    tool_efficiency: float       # 0-1: Were tools used appropriately?
    self_correction: float       # 0-1: Did it self-correct when needed?
    reasoning_depth: float       # 0-1: How thorough was the reasoning?
    exploration_fit: float       # 0-1: Was exploration depth appropriate?

    total_reward: float  # Weighted sum

    def to_dict(self) -> Dict[str, float]:
        return {
            "constraint_adherence": self.constraint_adherence,
            "outcome_correctness": self.outcome_correctness,
            "tool_efficiency": self.tool_efficiency,
            "self_correction": self.self_correction,
            "reasoning_depth": self.reasoning_depth,
            "exploration_fit": self.exploration_fit,
            "total_reward": self.total_reward,
        }


class BehavioralRewardModel:
    """
    Lightweight reward model for scoring workflow quality.

    Unlike typical RM that needs a neural network, this uses rule-based
    heuristics combined with learned weights to score behavioral patterns.
    """

    def __init__(self, weights: Dict[str, float] = None):
        # Default weights — ordered by Claude's real priority
        self.weights = weights or {
            "constraint_adherence": 0.30,   # Highest: did it respect constraints?
            "outcome_correctness": 0.25,    # Second: is the final artifact correct/complete?
            "tool_efficiency": 0.15,        # Third: were tools used correctly?
            "self_correction": 0.15,        # Fourth: did it recover from mistakes?
            "reasoning_depth": 0.10,        # Fifth: appropriate reasoning depth?
            "exploration_fit": 0.05,        # Last: exploration only matters when outcome is ambiguous
        }

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

    def compute_reward(
        self,
        reasoning_signals: ReasoningSignals,
        task_complexity: str = "medium",
        output_artifact: Optional[OutputArtifact] = None,
        response_text: str = "",
    ) -> RewardComponents:
        """Compute reward for a generated workflow."""

        # 1. Constraint adherence (highest priority)
        constraint_score = self._score_constraints(reasoning_signals)

        # 2. Outcome correctness (new — success-state anchoring)
        outcome_score = self._score_outcome_correctness(
            output_artifact, response_text, reasoning_signals
        )

        # 3. Tool efficiency
        tool_score = self._score_tool_efficiency(reasoning_signals)

        # 4. Self-correction
        correction_score = self._score_self_correction(reasoning_signals)

        # 5. Reasoning depth
        reasoning_score = self._score_reasoning_depth(
            reasoning_signals, task_complexity
        )

        # 6. Exploration fit (lowest priority)
        exploration_score = self._score_exploration_fit(
            reasoning_signals, task_complexity
        )

        # Compute weighted total
        total_reward = (
            self.weights["constraint_adherence"] * constraint_score +
            self.weights["outcome_correctness"] * outcome_score +
            self.weights["tool_efficiency"] * tool_score +
            self.weights["self_correction"] * correction_score +
            self.weights["reasoning_depth"] * reasoning_score +
            self.weights["exploration_fit"] * exploration_score
        )

        return RewardComponents(
            constraint_adherence=constraint_score,
            outcome_correctness=outcome_score,
            tool_efficiency=tool_score,
            self_correction=correction_score,
            reasoning_depth=reasoning_score,
            exploration_fit=exploration_score,
            total_reward=total_reward,
        )

    def _score_outcome_correctness(
        self,
        artifact: Optional[OutputArtifact],
        response_text: str,
        signals: ReasoningSignals,
    ) -> float:
        """
        Score the terminal output artifact for correctness and completeness.

        This is success-state anchoring: does the output satisfy functional requirements?
        Not style — structure, completeness, validity.
        """
        score = 0.0
        checks = 0

        # --- If we have a structured OutputArtifact, use it ---
        if artifact is not None:
            # Schema validity (hard gate — invalid schema is a strong negative)
            if artifact.schema_valid:
                score += 1.0
            else:
                score += 0.1  # heavy penalty
            checks += 1

            # Completeness from checklist
            if artifact.completeness_checklist:
                completeness = artifact.compute_completeness()
                score += completeness
                checks += 1

            # Required components present
            if artifact.required_components and response_text:
                present = sum(
                    1 for comp in artifact.required_components
                    if comp.lower() in response_text.lower()
                )
                component_ratio = present / len(artifact.required_components)
                score += component_ratio
                checks += 1

            # Penalize missing edge cases when task is complex
            if artifact.edge_cases_mentioned:
                score += min(len(artifact.edge_cases_mentioned) * 0.2, 1.0)
                checks += 1

            # Penalize schema violations
            if artifact.schema_violations:
                violation_penalty = min(len(artifact.schema_violations) * 0.25, 0.8)
                score += (1.0 - violation_penalty)
                checks += 1

            # Reward appropriate refusals (Claude says "data unavailable" instead of hallucinating)
            if artifact.refusals:
                score += 0.8  # refusals are almost always correct
                checks += 1

        # --- Heuristic fallback when no artifact ---
        if checks == 0:
            score, checks = self._heuristic_outcome_score(response_text, signals)

        return score / max(checks, 1)

    def _heuristic_outcome_score(
        self, response_text: str, signals: ReasoningSignals
    ) -> tuple:
        """
        Fallback heuristic scoring when no OutputArtifact is available.
        Checks structural markers that indicate a correct terminal output.
        """
        score = 0.0
        checks = 0

        if not response_text:
            return 0.3, 1  # empty response = bad

        # 1. Does the response actually answer something? (not just meta-commentary)
        has_substantive_content = len(response_text.strip()) > 50
        filler_ratio = len(re.findall(
            r'\b(let me|I would|I can|I\'ll|here\'s what|based on)\b',
            response_text, re.IGNORECASE
        )) / max(len(response_text.split()), 1)

        if has_substantive_content and filler_ratio < 0.15:
            score += 1.0
        elif has_substantive_content:
            score += 0.6
        else:
            score += 0.3
        checks += 1

        # 2. Does it contain specific data points (not vague claims)?
        has_numbers = bool(re.search(r'\d+\.?\d*\s*(?:PSI|kW|MW|rpm|°?C|bar|%|mm|Hz)', response_text))
        has_ids = bool(re.search(r'(?:pump|chiller|transformer|CT|DG|AHU)[-_]?\d+', response_text, re.IGNORECASE))
        specificity = (0.5 if has_numbers else 0.0) + (0.5 if has_ids else 0.0)
        score += specificity
        checks += 1

        # 3. Does it address the question directly (first sentence)?
        sentences = re.split(r'[.!?]\s+', response_text.strip())
        if sentences:
            first = sentences[0].lower()
            is_direct = not any(first.startswith(p) for p in [
                "i ", "let me", "sure", "of course", "great question",
                "i'd be happy", "absolutely", "certainly"
            ])
            score += 1.0 if is_direct else 0.4
            checks += 1

        # 4. Task completion signal from reasoning
        if signals and signals.tool_sequence:
            # If tools were used and a response was generated, likely completed
            score += 0.8
            checks += 1

        return score, checks

    def _score_constraints(self, signals: ReasoningSignals) -> float:
        """Score constraint detection and adherence."""
        if not signals.constraints_detected:
            return 0.5  # Neutral - no constraints detected

        # More constraints detected = better awareness
        num_constraints = len(signals.constraints_detected)

        # Check for explicit constraint handling (impact field describes how it was handled)
        has_explicit_handling = any(
            c.impact is not None and len(c.impact) > 0
            for c in signals.constraints_detected
        )

        # Score: 0.6 base + 0.2 for multiple constraints + 0.2 for explicit handling
        score = 0.6
        if num_constraints > 1:
            score += 0.2
        if has_explicit_handling:
            score += 0.2

        return min(score, 1.0)

    def _score_reasoning_depth(
        self,
        signals: ReasoningSignals,
        task_complexity: str
    ) -> float:
        """Score reasoning depth appropriateness."""
        steps = signals.reasoning_steps

        # Expected steps by complexity
        complexity_thresholds = {
            "simple": (1, 3),    # 1-3 steps
            "medium": (3, 6),    # 3-6 steps
            "complex": (6, 12),  # 6-12 steps
        }

        min_steps, max_steps = complexity_thresholds.get(
            task_complexity,
            (3, 6)
        )

        # Score based on step count fit
        if steps < min_steps:
            return 0.3  # Too shallow
        elif steps > max_steps * 1.5:
            return 0.6  # Too verbose
        elif min_steps <= steps <= max_steps:
            return 1.0  # Perfect fit
        else:
            return 0.8  # Slightly over but acceptable

    def _score_tool_efficiency(self, signals: ReasoningSignals) -> float:
        """Score tool usage efficiency."""
        tool_sequence = signals.tool_sequence

        if not tool_sequence:
            return 0.4  # No tools used (might be appropriate)

        # Check for tool variety (not just repeating same tool)
        unique_tools = len(set(tool_sequence))
        tool_diversity = unique_tools / len(tool_sequence)

        # Check for logical tool ordering
        has_logical_flow = self._check_tool_flow(tool_sequence)

        # Score: 0.5 base + 0.25 for diversity + 0.25 for logical flow
        score = 0.5 + (tool_diversity * 0.25)
        if has_logical_flow:
            score += 0.25

        return min(score, 1.0)

    def _check_tool_flow(self, tools: List[str]) -> bool:
        """Check if tool sequence follows logical patterns."""
        # Good patterns
        good_patterns = [
            ["Read", "Grep"],      # Read then search
            ["Grep", "Read"],      # Search then read matches
            ["Read", "Edit"],      # Read then modify
            ["Bash", "Read"],      # Run command then check result
        ]

        # Check for any good pattern
        for i in range(len(tools) - 1):
            pair = [tools[i], tools[i+1]]
            if pair in good_patterns:
                return True

        # Check for bad patterns (immediate repetition)
        for i in range(len(tools) - 1):
            if tools[i] == tools[i+1]:
                return False  # Redundant tool calls

        return True  # Neutral

    def _score_self_correction(self, signals: ReasoningSignals) -> float:
        """Score self-correction behavior."""
        corrections = signals.self_corrections

        if not corrections:
            return 0.7  # No corrections needed (good or bad?)

        # Check correction triggers for quality signals
        has_approach_correction = any(
            any(kw in c.trigger.lower() for kw in ["approach", "revis", "rethink", "better", "instead"])
            for c in corrections
        )
        has_error_correction = any(
            any(kw in c.trigger.lower() for kw in ["error", "fail", "didn't work", "wrong", "bug"])
            for c in corrections
        )

        # Score: 0.7 base + 0.15 for approach + 0.15 for error
        score = 0.7
        if has_approach_correction:
            score += 0.15
        if has_error_correction:
            score += 0.15

        return min(score, 1.0)

    def _score_exploration_fit(
        self,
        signals: ReasoningSignals,
        task_complexity: str
    ) -> float:
        """
        Score exploration depth appropriateness.

        Downweighted to 5% — exploration should only matter when outcome is ambiguous.
        Model should commit when Claude would commit, not explore endlessly.
        """
        exploration = signals.exploration_depth

        # Expected exploration by complexity
        expected = {
            "simple": "minimal",
            "medium": "moderate",
            "complex": "thorough",
        }

        expected_depth = expected.get(task_complexity, "moderate")

        # Score based on match
        if exploration == expected_depth:
            return 1.0  # Perfect match

        # Exploration levels: minimal < moderate < thorough < exhaustive
        levels = ["minimal", "moderate", "thorough", "exhaustive"]

        try:
            actual_idx = levels.index(exploration)
            expected_idx = levels.index(expected_depth)
            diff = abs(actual_idx - expected_idx)

            if diff == 1:
                return 0.7  # One level off
            elif diff == 2:
                return 0.4  # Two levels off
            else:
                return 0.2  # Far off
        except ValueError:
            return 0.5  # Unknown level

    def compare_traces(
        self,
        claude_trace: ClaudeTrace,
        llama_trace: ClaudeTrace,
    ) -> Tuple[RewardComponents, RewardComponents, float]:
        """
        Compare Claude vs LLaMA traces.

        Returns:
            (claude_reward, llama_reward, similarity_score)
        """
        # Compute rewards for both
        claude_reward = self.compute_reward(
            claude_trace.reasoning_signals,
            output_artifact=claude_trace.output_artifact,
            response_text=claude_trace.claude_response,
        )
        llama_reward = self.compute_reward(
            llama_trace.reasoning_signals,
            output_artifact=llama_trace.output_artifact,
            response_text=llama_trace.claude_response,
        )

        # Compute similarity (0-1, higher = more similar)
        similarity = self._compute_similarity(claude_reward, llama_reward)

        return claude_reward, llama_reward, similarity

    def _compute_similarity(
        self,
        reward1: RewardComponents,
        reward2: RewardComponents
    ) -> float:
        """Compute similarity between two reward profiles."""
        components = [
            "constraint_adherence",
            "outcome_correctness",
            "tool_efficiency",
            "self_correction",
            "reasoning_depth",
            "exploration_fit",
        ]

        similarities = []
        for comp in components:
            val1 = getattr(reward1, comp)
            val2 = getattr(reward2, comp)
            sim = 1.0 - abs(val1 - val2)
            similarities.append(sim)

        return sum(similarities) / len(similarities)


class LearnableRewardModel(nn.Module):
    """
    Neural network-based reward model (optional, for future use).

    Can be trained on human preferences if we collect pairwise comparisons.
    For now, we use the rule-based BehavioralRewardModel.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        # Input: encoded reasoning signals (fixed-size vector)
        # Output: scalar reward

        self.encoder = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),  # Scalar reward
        )

    def forward(self, reasoning_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reasoning_features: (batch_size, 128) encoded reasoning signals

        Returns:
            rewards: (batch_size, 1) scalar rewards
        """
        return self.encoder(reasoning_features)


def main():
    """Demo reward model scoring."""
    from claude_trace_schema import (
        ReasoningSignals,
        ConstraintDetection,
        SelfCorrection,
        OutputArtifact,
    )

    # Example signals
    signals = ReasoningSignals(
        tool_sequence=["Read", "Grep", "Bash", "Edit"],
        reasoning_steps=5,
        exploration_depth="moderate",
        constraints_detected=[
            ConstraintDetection(
                constraint="Must check if file exists before editing",
                source="file_structure",
                impact="Read first, then Edit"
            )
        ],
        tools_pruned=[],
        self_corrections=[
            SelfCorrection(
                step_number=2,
                original_plan="Direct edit",
                correction="Read first to verify content",
                trigger="Realized need to check current state — rethink approach"
            )
        ],
    )

    # Example output artifact
    artifact = OutputArtifact(
        required_components=["equipment_id", "metric_value", "status"],
        schema_valid=True,
        schema_type="text",
        completeness_checklist={
            "answers_question": True,
            "includes_data": True,
            "cites_source": True,
            "mentions_caveats": False,
        },
        factual_claims=["pump-002 pressure is 37.5 PSI", "status is normal"],
        edge_cases_mentioned=["pump was offline for maintenance Jan 15"],
    )

    # Score it
    rm = BehavioralRewardModel()
    reward = rm.compute_reward(
        signals,
        task_complexity="medium",
        output_artifact=artifact,
        response_text="Pump-002 is running at 37.5 PSI, status normal. Note: was offline for maintenance Jan 15.",
    )

    print("=" * 70)
    print(" Behavioral Reward Model Demo")
    print("=" * 70)
    print()
    print("Reward Weights (Claude's priority order):")
    for k, v in rm.weights.items():
        print(f"  {k}: {v:.2f}")
    print()
    print("Workflow Analysis:")
    print(f"  Tools: {' -> '.join(signals.tool_sequence)}")
    print(f"  Reasoning steps: {signals.reasoning_steps}")
    print(f"  Exploration: {signals.exploration_depth}")
    print(f"  Constraints detected: {len(signals.constraints_detected)}")
    print(f"  Self-corrections: {len(signals.self_corrections)}")
    print()
    print("Reward Breakdown:")
    print(f"  Constraint adherence:  {reward.constraint_adherence:.3f}  (weight: {rm.weights['constraint_adherence']:.2f})")
    print(f"  Outcome correctness:   {reward.outcome_correctness:.3f}  (weight: {rm.weights['outcome_correctness']:.2f})")
    print(f"  Tool efficiency:       {reward.tool_efficiency:.3f}  (weight: {rm.weights['tool_efficiency']:.2f})")
    print(f"  Self-correction:       {reward.self_correction:.3f}  (weight: {rm.weights['self_correction']:.2f})")
    print(f"  Reasoning depth:       {reward.reasoning_depth:.3f}  (weight: {rm.weights['reasoning_depth']:.2f})")
    print(f"  Exploration fit:       {reward.exploration_fit:.3f}  (weight: {rm.weights['exploration_fit']:.2f})")
    print()
    print(f"  TOTAL REWARD: {reward.total_reward:.3f}")
    print()


if __name__ == "__main__":
    main()
