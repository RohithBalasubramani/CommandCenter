#!/usr/bin/env python3
"""
Automated Parallel Runner

Runs prompts through BOTH Claude CLI and LLaMA simultaneously:
1. Execute prompt in Claude Code CLI (automated)
2. Execute same prompt in LLaMA (via Ollama)
3. Capture both responses
4. Compare and identify differences
5. Train LLaMA to match Claude

This creates a continuous improvement loop.
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import logging

from claude_trace_schema import ClaudeTrace, ToolCall, TraceStorage
from reasoning_extractor import ReasoningSignalExtractor, ReasoningSignals
from enhanced_extractor import EnhancedSignalExtractor, enhance_trace_with_maximum_extraction
from enhanced_extraction import (
    EnhancedReasoningSignals, ReasoningVector,
    AssumptionStatement, ErrorValidationCheck, CounterfactualPath,
    ProvenanceRecord, SafetySignal, SelfCritique, PreferenceRanking
)
import numpy as np
from scipy.spatial.distance import cosine, euclidean

logger = logging.getLogger(__name__)


class AutomatedRunner:
    """
    Automated parallel execution and comparison system.

    Runs prompts through both Claude and LLaMA, captures responses,
    compares them, and uses differences for training.
    """

    def __init__(
        self,
        claude_cli_path: str = "claude",
        llama_model: str = "cc-claude-agent:latest",
        storage_dir: str = "/home/rohith/desktop/CommandCenter/claude-rl-agent/data"
    ):
        self.claude_cli = claude_cli_path
        self.llama_model = llama_model
        self.storage = TraceStorage(storage_dir)
        # Use ENHANCED extractor for maximum signal extraction (35 dimensions)
        self.extractor = EnhancedSignalExtractor()
        # Keep base extractor for fallback
        self.base_extractor = ReasoningSignalExtractor()

        # Comparison results storage
        self.comparison_log = Path(storage_dir) / "comparison_log.jsonl"
        self.dpo_pairs = Path(storage_dir) / "dpo_pairs.jsonl"

    def run_claude_cli(self, prompt: str) -> Tuple[str, float]:
        """
        Run prompt through Claude Code CLI.

        Returns:
            (response, duration_seconds)
        """
        logger.info(f"Running Claude CLI: {prompt[:50]}...")

        start_time = time.time()

        try:
            result = subprocess.run(
                [self.claude_cli, prompt],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            duration = time.time() - start_time
            response = result.stdout

            logger.info(f"Claude completed in {duration:.2f}s")
            return response, duration

        except subprocess.TimeoutExpired:
            logger.error(f"Claude CLI timeout after 300s")
            return "[ERROR: Timeout]", 300.0
        except Exception as e:
            logger.error(f"Claude CLI error: {e}")
            return f"[ERROR: {e}]", 0.0

    def run_llama(self, prompt: str) -> Tuple[str, float]:
        """
        Run prompt through LLaMA via Ollama.

        Returns:
            (response, duration_seconds)
        """
        logger.info(f"Running LLaMA: {prompt[:50]}...")

        start_time = time.time()

        try:
            result = subprocess.run(
                ["ollama", "run", self.llama_model, prompt],
                capture_output=True,
                text=True,
                timeout=300,
            )

            duration = time.time() - start_time
            response = result.stdout

            logger.info(f"LLaMA completed in {duration:.2f}s")
            return response, duration

        except subprocess.TimeoutExpired:
            logger.error(f"LLaMA timeout after 300s")
            return "[ERROR: Timeout]", 300.0
        except Exception as e:
            logger.error(f"LLaMA error: {e}")
            return f"[ERROR: {e}]", 0.0

    def extract_reasoning_signals(self, response: str) -> Tuple[ReasoningSignals, EnhancedReasoningSignals]:
        """
        Extract ALL reasoning signals from a response using MAXIMUM extraction.

        Captures 35+ dimensions:
        - Tool calls and sequences
        - Reasoning steps
        - Constraint detection
        - Self-corrections
        - Exploration depth
        - **ENHANCED**: Assumptions, validations, counterfactuals, provenance, safety, critique, 35-dim vectors
        """
        # Create a mock trace for the extractor
        mock_trace = ClaudeTrace(
            trace_id="temp",
            session_id="comparison",
            timestamp=datetime.now(),
            user_prompt="",
            claude_response=response,
            tool_calls=[],
            working_directory="/tmp",
            response_time_ms=0,
            task_completed=True
        )

        # Extract base signals (backwards compatibility)
        base_signals = self.base_extractor.extract_signals(mock_trace)

        # Extract ENHANCED signals (MAXIMUM extraction - 35 dimensions)
        enhanced_signals = self.extractor.extract_all(mock_trace)

        return base_signals, enhanced_signals

    def compare_behavioral_patterns(
        self,
        claude_signals: ReasoningSignals,
        llama_signals: ReasoningSignals
    ) -> Dict:
        """
        Deep comparison of behavioral patterns - not just text similarity.

        Compares:
        - Tool sequences (Claude: Bash→Read→Bash vs LLaMA: ???)
        - Reasoning depth (step counts)
        - Constraint detection (did both identify constraints?)
        - Self-correction (did both self-correct?)
        - Exploration appropriateness
        - All cognitive patterns
        """
        behavioral_comparison = {}

        # 1. Tool Sequence Comparison
        claude_tools = claude_signals.tool_sequence
        llama_tools = llama_signals.tool_sequence

        tool_match = (claude_tools == llama_tools)
        tool_overlap = len(set(claude_tools) & set(llama_tools)) / max(len(set(claude_tools) | set(llama_tools)), 1)

        behavioral_comparison["tool_sequence"] = {
            "claude": claude_tools,
            "llama": llama_tools,
            "exact_match": tool_match,
            "overlap": tool_overlap,
            "divergence": "DIFFERENT" if not tool_match else "SAME"
        }

        # 2. Reasoning Depth Comparison
        claude_steps = claude_signals.reasoning_steps
        llama_steps = llama_signals.reasoning_steps

        step_ratio = min(claude_steps, llama_steps) / max(claude_steps, llama_steps, 1)

        behavioral_comparison["reasoning_depth"] = {
            "claude_steps": claude_steps,
            "llama_steps": llama_steps,
            "similarity": step_ratio,
            "divergence": "DIFFERENT" if step_ratio < 0.8 else "SIMILAR"
        }

        # 3. Constraint Detection Comparison
        claude_constraints = len(claude_signals.constraints_detected)
        llama_constraints = len(llama_signals.constraints_detected)

        behavioral_comparison["constraint_detection"] = {
            "claude_count": claude_constraints,
            "llama_count": llama_constraints,
            "both_detected": claude_constraints > 0 and llama_constraints > 0,
            "divergence": "MISSING" if claude_constraints > llama_constraints else "SIMILAR"
        }

        # 4. Self-Correction Comparison
        claude_corrections = len(claude_signals.self_corrections)
        llama_corrections = len(llama_signals.self_corrections)

        behavioral_comparison["self_correction"] = {
            "claude_count": claude_corrections,
            "llama_count": llama_corrections,
            "both_corrected": claude_corrections > 0 and llama_corrections > 0,
            "divergence": "MISSING" if claude_corrections > llama_corrections else "SIMILAR"
        }

        # 5. Exploration Depth Comparison
        claude_exploration = claude_signals.exploration_depth.value if hasattr(claude_signals.exploration_depth, 'value') else str(claude_signals.exploration_depth)
        llama_exploration = llama_signals.exploration_depth.value if hasattr(llama_signals.exploration_depth, 'value') else str(llama_signals.exploration_depth)

        behavioral_comparison["exploration_depth"] = {
            "claude": claude_exploration,
            "llama": llama_exploration,
            "match": claude_exploration == llama_exploration,
            "divergence": "DIFFERENT" if claude_exploration != llama_exploration else "SAME"
        }

        # 6. Tool Pruning Comparison (approaches considered but rejected)
        claude_pruned = len(claude_signals.tools_pruned)
        llama_pruned = len(llama_signals.tools_pruned)

        behavioral_comparison["tool_pruning"] = {
            "claude_count": claude_pruned,
            "llama_count": llama_pruned,
            "divergence": "MISSING" if claude_pruned > llama_pruned else "SIMILAR"
        }

        # 7. Overall Behavioral Similarity Score (0-1)
        similarity_scores = [
            1.0 if tool_match else tool_overlap,
            step_ratio,
            1.0 if claude_exploration == llama_exploration else 0.5,
            1.0 if claude_constraints == llama_constraints else 0.7,
            1.0 if claude_corrections == llama_corrections else 0.8,
        ]

        behavioral_similarity = sum(similarity_scores) / len(similarity_scores)
        behavioral_comparison["overall_similarity"] = behavioral_similarity

        # 8. Determine if training is needed
        # Train if behavioral patterns differ significantly
        needs_training = (
            not tool_match or
            step_ratio < 0.8 or
            claude_exploration != llama_exploration or
            (claude_constraints > 0 and llama_constraints == 0) or
            (claude_corrections > 0 and llama_corrections == 0) or
            behavioral_similarity < 0.7
        )

        behavioral_comparison["should_train"] = needs_training
        behavioral_comparison["training_reason"] = []

        if not tool_match:
            behavioral_comparison["training_reason"].append("Tool sequence mismatch")
        if step_ratio < 0.8:
            behavioral_comparison["training_reason"].append("Reasoning depth differs")
        if claude_exploration != llama_exploration:
            behavioral_comparison["training_reason"].append("Exploration depth mismatch")
        if claude_constraints > 0 and llama_constraints == 0:
            behavioral_comparison["training_reason"].append("LLaMA missing constraint detection")
        if claude_corrections > 0 and llama_corrections == 0:
            behavioral_comparison["training_reason"].append("LLaMA missing self-correction")

        return behavioral_comparison

    def compare_enhanced_behavioral_patterns(
        self,
        claude_base: ReasoningSignals,
        llama_base: ReasoningSignals,
        claude_enhanced: EnhancedReasoningSignals,
        llama_enhanced: EnhancedReasoningSignals
    ) -> Dict:
        """
        MAXIMUM EXTRACTION COMPARISON - All 35+ dimensions.

        Compares:
        BASE SIGNALS:
        - Tool sequences, reasoning steps, constraints, self-corrections, exploration, pruning

        ENHANCED SIGNALS:
        - Assumptions (count, clarity)
        - Validation checks (completeness)
        - Counterfactual paths (alternative consideration)
        - Provenance (source citations)
        - Safety signals (refusal alignment)
        - Self-critique (confidence)
        - Reasoning vectors (35-dimensional behavioral profile)

        Returns comprehensive comparison with fine-grained similarity scoring.
        """
        comparison = {}

        # === BASE SIGNALS COMPARISON (backwards compatibility) ===
        base_comparison = self.compare_behavioral_patterns(claude_base, llama_base)
        comparison["base_signals"] = base_comparison

        # === ENHANCED SIGNALS COMPARISON ===
        enhanced = {}

        # 1. ASSUMPTIONS COMPARISON
        claude_assumptions = len(claude_enhanced.assumptions) if claude_enhanced.assumptions else 0
        llama_assumptions = len(llama_enhanced.assumptions) if llama_enhanced.assumptions else 0

        assumptions_clarity_score = 0.0
        if claude_assumptions > 0 and llama_assumptions > 0:
            # Compare assumption types and confidence
            claude_types = set(a.assumption_type.value if hasattr(a.assumption_type, 'value') else str(a.assumption_type)
                             for a in claude_enhanced.assumptions)
            llama_types = set(a.assumption_type.value if hasattr(a.assumption_type, 'value') else str(a.assumption_type)
                            for a in llama_enhanced.assumptions)
            type_overlap = len(claude_types & llama_types) / max(len(claude_types | llama_types), 1)
            assumptions_clarity_score = type_overlap

        enhanced["assumptions"] = {
            "claude_count": claude_assumptions,
            "llama_count": llama_assumptions,
            "clarity_score": assumptions_clarity_score,
            "divergence": "MISSING" if claude_assumptions > llama_assumptions else ("SIMILAR" if abs(claude_assumptions - llama_assumptions) <= 1 else "DIFFERENT")
        }

        # 2. VALIDATION CHECKS COMPARISON
        claude_validations = len(claude_enhanced.validation_checks) if claude_enhanced.validation_checks else 0
        llama_validations = len(llama_enhanced.validation_checks) if llama_enhanced.validation_checks else 0

        validation_completeness = 0.0
        if claude_validations > 0 and llama_validations > 0:
            validation_completeness = min(llama_validations, claude_validations) / max(claude_validations, llama_validations)

        enhanced["validation_checks"] = {
            "claude_count": claude_validations,
            "llama_count": llama_validations,
            "completeness_score": validation_completeness,
            "divergence": "MISSING" if claude_validations > llama_validations else ("SIMILAR" if abs(claude_validations - llama_validations) <= 1 else "DIFFERENT")
        }

        # 3. COUNTERFACTUAL PATHS COMPARISON
        claude_counterfactuals = len(claude_enhanced.counterfactual_paths) if claude_enhanced.counterfactual_paths else 0
        llama_counterfactuals = len(llama_enhanced.counterfactual_paths) if llama_enhanced.counterfactual_paths else 0

        counterfactual_consideration = 0.0
        if claude_counterfactuals > 0 and llama_counterfactuals > 0:
            counterfactual_consideration = min(llama_counterfactuals, claude_counterfactuals) / max(claude_counterfactuals, llama_counterfactuals)

        enhanced["counterfactual_paths"] = {
            "claude_count": claude_counterfactuals,
            "llama_count": llama_counterfactuals,
            "consideration_score": counterfactual_consideration,
            "divergence": "MISSING" if claude_counterfactuals > llama_counterfactuals else ("SIMILAR" if claude_counterfactuals == llama_counterfactuals else "DIFFERENT")
        }

        # 4. PROVENANCE TRACKING COMPARISON
        claude_provenance = len(claude_enhanced.provenance) if claude_enhanced.provenance else 0
        llama_provenance = len(llama_enhanced.provenance) if llama_enhanced.provenance else 0

        provenance_score = 0.0
        if claude_provenance > 0 and llama_provenance > 0:
            provenance_score = min(llama_provenance, claude_provenance) / max(claude_provenance, llama_provenance)

        enhanced["provenance"] = {
            "claude_count": claude_provenance,
            "llama_count": llama_provenance,
            "citation_score": provenance_score,
            "divergence": "MISSING" if claude_provenance > llama_provenance else ("SIMILAR" if abs(claude_provenance - llama_provenance) <= 1 else "DIFFERENT")
        }

        # 5. SAFETY SIGNALS COMPARISON
        claude_safety = len(claude_enhanced.safety_signals) if claude_enhanced.safety_signals else 0
        llama_safety = len(llama_enhanced.safety_signals) if llama_enhanced.safety_signals else 0

        safety_alignment = 1.0  # Default: both safe
        if claude_safety > 0 or llama_safety > 0:
            if claude_safety == llama_safety:
                safety_alignment = 1.0
            else:
                safety_alignment = min(llama_safety, claude_safety) / max(claude_safety, llama_safety, 1)

        enhanced["safety_signals"] = {
            "claude_count": claude_safety,
            "llama_count": llama_safety,
            "alignment_score": safety_alignment,
            "divergence": "MISALIGNED" if abs(claude_safety - llama_safety) > 0 else "ALIGNED"
        }

        # 6. SELF-CRITIQUE COMPARISON
        claude_confidence = claude_enhanced.self_critique.confidence_level if claude_enhanced.self_critique else 0.5
        llama_confidence = llama_enhanced.self_critique.confidence_level if llama_enhanced.self_critique else 0.5

        confidence_alignment = 1.0 - abs(claude_confidence - llama_confidence)

        enhanced["self_critique"] = {
            "claude_confidence": claude_confidence,
            "llama_confidence": llama_confidence,
            "alignment_score": confidence_alignment,
            "divergence": "SIMILAR" if abs(claude_confidence - llama_confidence) < 0.2 else "DIFFERENT"
        }

        # 7. REASONING VECTOR COMPARISON (35 dimensions)
        vector_comparison = {}
        if claude_enhanced.reasoning_vector and llama_enhanced.reasoning_vector:
            claude_vec = claude_enhanced.reasoning_vector.to_numpy()
            llama_vec = llama_enhanced.reasoning_vector.to_numpy()

            # Cosine similarity (1 = identical, 0 = orthogonal)
            cosine_sim = 1.0 - cosine(claude_vec, llama_vec)

            # Euclidean distance (normalized)
            euclidean_dist = euclidean(claude_vec, llama_vec)
            max_dist = np.linalg.norm(np.ones(35))  # Max possible distance
            normalized_euclidean = 1.0 - (euclidean_dist / max_dist)

            # Per-dimension comparison
            dimension_diffs = np.abs(claude_vec - llama_vec)
            critical_divergences = []

            dim_names = [
                "reasoning_steps", "exploration_depth", "tool_calls", "constraints", "self_corrections",
                "tools_pruned", "assumptions", "validations", "counterfactuals", "multi_step",
                "used_rag", "used_terminal", "used_web", "parallel_tools", "explicit_planning",
                "constraint_adherence", "reasoning_depth_score", "tool_efficiency", "self_correction_score",
                "exploration_fit", "assumption_clarity", "validation_completeness", "counterfactual_consideration",
                "overall_confidence", "task_success", "response_time", "response_length", "code_blocks",
                "markdown", "json_output", "error", "user_feedback", "safety_concerns",
                "provenance_citations", "edit_history"
            ]

            for i, (diff, name) in enumerate(zip(dimension_diffs, dim_names)):
                if diff > 0.3:  # Significant divergence threshold
                    critical_divergences.append({
                        "dimension": name,
                        "claude_value": float(claude_vec[i]),
                        "llama_value": float(llama_vec[i]),
                        "difference": float(diff)
                    })

            vector_comparison = {
                "cosine_similarity": float(cosine_sim),
                "normalized_euclidean_similarity": float(normalized_euclidean),
                "average_dimension_difference": float(np.mean(dimension_diffs)),
                "max_dimension_difference": float(np.max(dimension_diffs)),
                "critical_divergences": critical_divergences,
                "divergence": "SIMILAR" if cosine_sim > 0.85 else ("MODERATE" if cosine_sim > 0.7 else "DIFFERENT")
            }
        else:
            vector_comparison = {
                "cosine_similarity": 0.0,
                "divergence": "NO_VECTORS",
                "error": "Reasoning vectors not available"
            }

        enhanced["reasoning_vector"] = vector_comparison

        comparison["enhanced_signals"] = enhanced

        # === OVERALL SIMILARITY SCORE (weighted) ===
        # Combine base (40%) + enhanced (60%) for comprehensive score
        base_similarity = base_comparison["overall_similarity"]

        enhanced_similarity_scores = [
            assumptions_clarity_score if claude_assumptions > 0 else 1.0,
            validation_completeness if claude_validations > 0 else 1.0,
            counterfactual_consideration if claude_counterfactuals > 0 else 1.0,
            provenance_score if claude_provenance > 0 else 1.0,
            safety_alignment,
            confidence_alignment,
            vector_comparison.get("cosine_similarity", 0.0)
        ]

        enhanced_similarity = sum(enhanced_similarity_scores) / len(enhanced_similarity_scores)

        overall_similarity = (base_similarity * 0.4) + (enhanced_similarity * 0.6)
        comparison["overall_similarity"] = overall_similarity
        comparison["base_similarity_weighted"] = base_similarity * 0.4
        comparison["enhanced_similarity_weighted"] = enhanced_similarity * 0.6

        # === TRAINING DECISION (enhanced criteria) ===
        needs_training = (
            base_comparison["should_train"] or  # Base signals diverge
            assumptions_clarity_score < 0.7 or  # Assumptions mismatch
            validation_completeness < 0.7 or    # Validation incomplete
            counterfactual_consideration < 0.5 or  # Missing alternatives
            safety_alignment < 0.9 or            # Safety misalignment
            vector_comparison.get("cosine_similarity", 0.0) < 0.75 or  # Vector divergence
            overall_similarity < 0.75            # Overall behavioral divergence
        )

        comparison["should_train"] = needs_training
        comparison["training_reason"] = base_comparison.get("training_reason", []).copy()

        if assumptions_clarity_score < 0.7 and claude_assumptions > 0:
            comparison["training_reason"].append("Assumption clarity differs")
        if validation_completeness < 0.7 and claude_validations > 0:
            comparison["training_reason"].append("Validation completeness differs")
        if counterfactual_consideration < 0.5 and claude_counterfactuals > 0:
            comparison["training_reason"].append("Counterfactual consideration missing")
        if provenance_score < 0.7 and claude_provenance > 0:
            comparison["training_reason"].append("Provenance tracking differs")
        if safety_alignment < 0.9:
            comparison["training_reason"].append("Safety signal misalignment")
        if vector_comparison.get("cosine_similarity", 0.0) < 0.75:
            comparison["training_reason"].append(f"Reasoning vector divergence (similarity: {vector_comparison.get('cosine_similarity', 0.0):.2f})")

        return comparison

    def compare_responses(
        self,
        prompt: str,
        claude_response: str,
        llama_response: str
    ) -> Dict:
        """
        Deep comparison of Claude vs LLaMA - EVERYTHING, not just text.

        Extracts and compares:
        - All tool calls
        - Complete reasoning chains
        - Constraint detection
        - Self-corrections
        - Exploration patterns
        - Cognitive workflow
        """
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "claude_response": claude_response,
            "llama_response": llama_response,
        }

        # Extract reasoning signals from both responses (BASE + ENHANCED)
        logger.info("Extracting Claude's reasoning signals (MAXIMUM extraction - 35 dimensions)...")
        claude_base, claude_enhanced = self.extract_reasoning_signals(claude_response)

        logger.info("Extracting LLaMA's reasoning signals (MAXIMUM extraction - 35 dimensions)...")
        llama_base, llama_enhanced = self.extract_reasoning_signals(llama_response)

        # COMPREHENSIVE behavioral comparison (BASE + ENHANCED signals)
        behavioral_comparison = self.compare_enhanced_behavioral_patterns(
            claude_base,
            llama_base,
            claude_enhanced,
            llama_enhanced
        )

        comparison["behavioral_comparison"] = behavioral_comparison
        comparison["should_train"] = behavioral_comparison["should_train"]
        comparison["training_reason"] = behavioral_comparison.get("training_reason", [])

        # Also do basic text similarity for reference
        claude_len = len(claude_response)
        llama_len = len(llama_response)
        len_ratio = min(claude_len, llama_len) / max(claude_len, llama_len, 1)

        claude_words = set(claude_response.lower().split())
        llama_words = set(llama_response.lower().split())
        if claude_words or llama_words:
            jaccard = len(claude_words & llama_words) / len(claude_words | llama_words)
        else:
            jaccard = 0

        comparison["text_similarity"] = {
            "length_ratio": len_ratio,
            "word_overlap": jaccard,
            "claude_length": claude_len,
            "llama_length": llama_len,
        }

        if behavioral_comparison["should_train"]:
            logger.info(f"Behavioral divergence detected!")
            logger.info(f"  Reasons: {', '.join(behavioral_comparison['training_reason'])}")
            logger.info(f"  Overall similarity: {behavioral_comparison['overall_similarity']:.2%}")

        return comparison

    def create_dpo_pair(self, comparison: Dict) -> Dict:
        """
        Create DPO training pair from comparison.

        DPO format:
        - prompt: The question
        - chosen: Claude's response (better)
        - rejected: LLaMA's response (worse)
        """
        return {
            "prompt": comparison["prompt"],
            "chosen": comparison["claude_response"],
            "rejected": comparison["llama_response"],
            "timestamp": comparison["timestamp"],
            "metrics": comparison["metrics"],
        }

    def save_comparison(self, comparison: Dict):
        """Save comparison result to log."""
        with open(self.comparison_log, 'a') as f:
            f.write(json.dumps(comparison) + '\n')

    def save_dpo_pair(self, dpo_pair: Dict):
        """Save DPO training pair."""
        with open(self.dpo_pairs, 'a') as f:
            f.write(json.dumps(dpo_pair) + '\n')

    def run_parallel_comparison(self, prompt: str) -> Dict:
        """
        Run prompt through both Claude and LLaMA, compare results.

        Returns comparison dict.
        """
        print(f"\n{'='*70}")
        print(f"Running parallel comparison:")
        print(f"Prompt: {prompt[:60]}...")
        print(f"{'='*70}\n")

        # Run both in parallel (could use threading, but sequential for simplicity)
        claude_response, claude_time = self.run_claude_cli(prompt)
        llama_response, llama_time = self.run_llama(prompt)

        # Compare
        comparison = self.compare_responses(prompt, claude_response, llama_response)
        comparison["claude_time"] = claude_time
        comparison["llama_time"] = llama_time

        # Save
        self.save_comparison(comparison)

        # Create DPO pair if divergent
        if comparison["should_train"]:
            dpo_pair = self.create_dpo_pair(comparison)
            self.save_dpo_pair(dpo_pair)
            print(f"✅ DPO pair saved for training")
        else:
            print(f"✅ Responses similar - no training needed")

        # Print detailed behavioral comparison
        behavioral = comparison["behavioral_comparison"]

        print(f"\n{'─'*70}")
        print(f"BEHAVIORAL COMPARISON RESULTS")
        print(f"{'─'*70}")

        print(f"\n⏱  Execution Time:")
        print(f"  Claude: {claude_time:.2f}s | LLaMA: {llama_time:.2f}s")

        print(f"\n🔧 Tool Sequence:")
        print(f"  Claude: {' → '.join(behavioral['tool_sequence']['claude']) if behavioral['tool_sequence']['claude'] else 'None'}")
        print(f"  LLaMA:  {' → '.join(behavioral['tool_sequence']['llama']) if behavioral['tool_sequence']['llama'] else 'None'}")
        print(f"  Match:  {behavioral['tool_sequence']['divergence']}")

        print(f"\n🧠 Reasoning Depth:")
        print(f"  Claude: {behavioral['reasoning_depth']['claude_steps']} steps")
        print(f"  LLaMA:  {behavioral['reasoning_depth']['llama_steps']} steps")
        print(f"  Status: {behavioral['reasoning_depth']['divergence']}")

        print(f"\n🚧 Constraint Detection:")
        print(f"  Claude: {behavioral['constraint_detection']['claude_count']} constraints")
        print(f"  LLaMA:  {behavioral['constraint_detection']['llama_count']} constraints")
        print(f"  Status: {behavioral['constraint_detection']['divergence']}")

        print(f"\n🔄 Self-Correction:")
        print(f"  Claude: {behavioral['self_correction']['claude_count']} corrections")
        print(f"  LLaMA:  {behavioral['self_correction']['llama_count']} corrections")

        print(f"\n🔍 Exploration Depth:")
        print(f"  Claude: {behavioral['exploration_depth']['claude']}")
        print(f"  LLaMA:  {behavioral['exploration_depth']['llama']}")
        print(f"  Status: {behavioral['exploration_depth']['divergence']}")

        print(f"\n📊 Overall Behavioral Similarity: {behavioral['overall_similarity']:.1%}")
        print(f"📝 Text Overlap: {comparison['text_similarity']['word_overlap']:.1%}")

        print(f"\n{'─'*70}")
        if comparison['should_train']:
            print(f"🎯 TRAINING NEEDED")
            print(f"   Reasons:")
            for reason in comparison['training_reason']:
                print(f"   • {reason}")
        else:
            print(f"✅ PATTERNS MATCH - No training needed")
        print(f"{'─'*70}")

        return comparison

    def run_batch(self, prompts: List[str], max_prompts: int = None):
        """
        Run batch of prompts through parallel comparison.

        Args:
            prompts: List of prompts to run
            max_prompts: Optional limit on number of prompts
        """
        if max_prompts:
            prompts = prompts[:max_prompts]

        print(f"\n{'='*70}")
        print(f" Automated Parallel Runner - Batch Mode")
        print(f"{'='*70}")
        print(f"Running {len(prompts)} prompts through Claude + LLaMA\n")

        results = []
        divergent_count = 0

        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] Processing...")

            try:
                comparison = self.run_parallel_comparison(prompt)
                results.append(comparison)

                if comparison["should_train"]:
                    divergent_count += 1

            except Exception as e:
                logger.error(f"Error processing prompt: {e}")
                continue

        # Print summary
        print(f"\n{'='*70}")
        print(f" Batch Complete")
        print(f"{'='*70}")
        print(f"Total prompts:     {len(results)}")
        print(f"Divergent:         {divergent_count}")
        print(f"DPO pairs created: {divergent_count}")
        print(f"Training ready:    {divergent_count > 0}")
        print()

        if divergent_count > 0:
            print(f"Next step: Train LLaMA on DPO pairs:")
            print(f"  ./run.sh train --phase ppo")

        return results


# Command Center specific prompt generator
def generate_command_center_prompts(count: int = 50) -> List[str]:
    """Generate Command Center database prompts."""

    prompts = [
        # Equipment queries
        "What's the average power consumption of chiller_001 in the last 24 hours?",
        "Show me the current status of all 6 DG sets",
        "Compare efficiency of transformer trf_001 vs trf_002 today",
        "List all equipment in fault state right now",
        "What's the total energy consumption for July 2024?",

        # Anomaly detection
        "Find temperature anomalies in chiller_003 for last week",
        "Detect power spikes in transformer trf_002 yesterday",
        "Show motor_005 vibration trends for the past month",
        "Identify unusual runtime patterns for DG sets",

        # Maintenance
        "When was the last maintenance for chiller_001?",
        "Show maintenance history for all transformers",
        "Which equipment needs maintenance in the next 30 days?",
        "Calculate MTBF for motor_001 to motor_015",

        # Real-time monitoring
        "What's the current power factor across all LT panels?",
        "Display real-time cooling tower temperatures",
        "Show UPS battery status for all 8 units",
        "What alerts have been triggered in the last hour?",

        # Data aggregation
        "Calculate hourly averages for chiller_001 on June 15, 2024",
        "Show daily peak demand for all transformers in Q2 2024",
        "Aggregate monthly energy by building",
        "Create a weekly summary for all AHUs",

        # Optimization
        "What's the optimal chiller staging to minimize cost?",
        "When should we switch from grid to DG?",
        "Show the load curve for all transformers combined",
        "Calculate ROI of different chiller setpoints",

        # Schema queries
        "Show me the schema of lt_mcc_001 table",
        "List all 357 equipment tables",
        "What columns are in the alerts table?",
        "Show relationships between buildings and equipment",

        # Predictive maintenance
        "Predict next chiller filter change based on pressure drop trend",
        "When will motor_003 bearings need replacement?",
        "Forecast transformer oil testing schedule",
        "Show equipment health scores across all tables",
    ]

    # Repeat to reach desired count
    while len(prompts) < count:
        prompts.extend(prompts[:count - len(prompts)])

    return prompts[:count]


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated parallel Claude vs LLaMA runner"
    )
    parser.add_argument(
        "--prompt",
        help="Single prompt to run"
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="Run batch of N prompts"
    )
    parser.add_argument(
        "--llama-model",
        default="cc-claude-agent:latest",
        help="LLaMA model name in Ollama"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create runner
    runner = AutomatedRunner(llama_model=args.llama_model)

    if args.prompt:
        # Single prompt
        runner.run_parallel_comparison(args.prompt)

    elif args.batch:
        # Batch mode
        prompts = generate_command_center_prompts(args.batch)
        runner.run_batch(prompts, max_prompts=args.batch)

    else:
        print("Usage:")
        print("  Single: python automated_runner.py --prompt 'your prompt'")
        print("  Batch:  python automated_runner.py --batch 50")


if __name__ == "__main__":
    main()
