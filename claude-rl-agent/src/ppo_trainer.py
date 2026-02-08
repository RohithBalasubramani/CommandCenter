#!/usr/bin/env python3
"""
PPO Trainer for Behavioral Alignment

Refines the SFT model using PPO with behavioral rewards.
Optimizes for workflow design quality rather than just text similarity.
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
import logging

try:
    from unsloth import FastLanguageModel
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from transformers import AutoTokenizer
    from datasets import Dataset
except ImportError as e:
    print(f"⚠️  Missing dependencies. Install with:")
    print(f"   pip install unsloth trl transformers datasets")
    raise e

from config import MODELS_DIR, DATASETS_DIR
from reward_model import BehavioralRewardModel, RewardComponents
from reasoning_extractor import ReasoningSignalExtractor

logger = logging.getLogger(__name__)


@dataclass
class PPOTrainingConfig:
    """Configuration for PPO training."""

    # Model settings
    sft_model_path: str  # Path to SFT checkpoint
    max_seq_length: int = 4096

    # PPO settings
    learning_rate: float = 1.4e-5
    batch_size: int = 1  # PPO is memory-intensive
    mini_batch_size: int = 1
    ppo_epochs: int = 4
    kl_penalty: str = "kl"  # or "abs"
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95

    # Training control
    num_episodes: int = 100  # Number of training episodes
    max_steps: int = -1  # -1 = run all episodes

    # Reward model
    reward_weights: Dict[str, float] = None  # Will use defaults

    # Output
    output_dir: str = str(MODELS_DIR / "ppo_checkpoints")
    save_freq: int = 10  # Save every N episodes

    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {
                "constraint_adherence": 0.30,
                "outcome_correctness": 0.25,
                "tool_efficiency": 0.15,
                "self_correction": 0.15,
                "reasoning_depth": 0.10,
                "exploration_fit": 0.05,
            }


class ClaudePPOTrainer:
    """PPO trainer for behavioral alignment."""

    def __init__(self, config: Optional[PPOTrainingConfig] = None):
        self.config = config or PPOTrainingConfig()
        self.model = None
        self.ref_model = None  # Reference model for KL penalty
        self.tokenizer = None
        self.ppo_trainer = None
        self.reward_model = BehavioralRewardModel(self.config.reward_weights)
        self.extractor = ReasoningSignalExtractor()

    def load_model(self):
        """Load SFT model and prepare for PPO."""
        logger.info(f"Loading SFT model: {self.config.sft_model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.sft_model_path
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with value head for PPO
        logger.info("Loading model with value head for PPO")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.sft_model_path
        )

        # Load reference model (frozen, for KL penalty)
        logger.info("Loading reference model for KL penalty")
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.sft_model_path
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        logger.info("✅ Models loaded for PPO training")

    def load_prompts(self, prompt_file: str) -> List[str]:
        """Load training prompts."""
        logger.info(f"Loading prompts: {prompt_file}")

        prompts = []
        with open(prompt_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data['prompt'])

        logger.info(f"Loaded {len(prompts)} prompts")
        return prompts

    def generate_response(self, prompt: str) -> tuple[str, torch.Tensor]:
        """
        Generate response from current policy model.

        Returns:
            (response_text, response_tensors)
        """
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length // 2,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response, outputs

    def compute_reward(self, prompt: str, response: str) -> float:
        """
        Compute behavioral reward for a generated response.

        This extracts reasoning signals AND scores the terminal output artifact.
        """
        from claude_trace_schema import ClaudeTrace, ToolCall
        from datetime import datetime

        # Extract tool calls from response (simple regex)
        tool_calls = self._extract_tool_calls(response)

        trace = ClaudeTrace(
            trace_id=f"ppo_{datetime.now().timestamp()}",
            session_id="ppo_training",
            timestamp=datetime.now(),
            user_prompt=prompt,
            claude_response=response,
            tool_calls=tool_calls,
            working_directory="/tmp",
            response_time_ms=0,
            task_completed=True,
        )

        # Extract reasoning signals
        trace.reasoning_signals = self.extractor.extract_signals(trace)

        # Build output artifact from response (heuristic extraction)
        trace.output_artifact = self._extract_output_artifact(prompt, response)

        # Compute reward — now includes outcome_correctness
        reward_components = self.reward_model.compute_reward(
            trace.reasoning_signals,
            task_complexity="medium",
            output_artifact=trace.output_artifact,
            response_text=response,
        )

        # Return total reward (scaled to [-1, 1] for PPO)
        reward = (reward_components.total_reward * 2) - 1

        return reward

    def _extract_output_artifact(self, prompt: str, response: str) -> 'OutputArtifact':
        """
        Build an OutputArtifact from the response using heuristics.

        Extracts required components, schema type, factual claims, etc.
        """
        from claude_trace_schema import OutputArtifact
        import re

        artifact = OutputArtifact()

        # Detect schema type
        if '```json' in response or response.strip().startswith('{'):
            artifact.schema_type = "json"
            try:
                import json
                json.loads(response.strip().split('```json')[-1].split('```')[0].strip()
                           if '```json' in response else response.strip())
                artifact.schema_valid = True
            except (json.JSONDecodeError, IndexError):
                artifact.schema_valid = False
                artifact.schema_violations.append("invalid_json")
        elif '```' in response:
            artifact.schema_type = "code"
        elif '|' in response and '---' in response:
            artifact.schema_type = "table"

        # Extract factual claims (numbers with units, equipment IDs)
        claims = re.findall(
            r'(?:[\w-]+\s+(?:is|are|was|running at|operating at)\s+[\d.]+\s*\w+)',
            response, re.IGNORECASE
        )
        artifact.factual_claims = claims[:10]  # cap at 10

        # Extract equipment IDs mentioned
        equipment_ids = re.findall(
            r'(?:pump|chiller|transformer|CT|DG|AHU)[-_]?\d+',
            response, re.IGNORECASE
        )

        # Extract refusals / uncertainty markers
        refusal_patterns = [
            r'data (?:is |)(?:not |un)available',
            r'no data (?:found|exists)',
            r'cannot determine',
            r'insufficient (?:data|information)',
        ]
        for pattern in refusal_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                artifact.refusals.append(pattern)

        # Extract edge case mentions
        edge_patterns = [
            r'(?:note|caveat|warning|however|exception):\s*(.{20,100})',
            r'(?:was offline|maintenance|outage|anomaly)\s+(.{10,80})',
        ]
        for pattern in edge_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            artifact.edge_cases_mentioned.extend(matches[:5])

        # Build completeness checklist based on what the prompt asks
        prompt_lower = prompt.lower()
        checklist = {}
        checklist["answers_question"] = len(response.strip()) > 30
        checklist["includes_specific_data"] = bool(
            re.search(r'\d+\.?\d*\s*(?:PSI|kW|MW|rpm|°?C|bar|%|mm|Hz)', response)
        )
        if equipment_ids:
            checklist["references_equipment"] = True
        if "compare" in prompt_lower or "difference" in prompt_lower:
            checklist["makes_comparison"] = any(
                w in response.lower() for w in ["higher", "lower", "compared", "versus", "difference", "while"]
            )
        if "status" in prompt_lower:
            checklist["reports_status"] = any(
                w in response.lower() for w in ["running", "normal", "alarm", "offline", "operating", "idle"]
            )

        artifact.completeness_checklist = checklist
        artifact.compute_completeness()

        return artifact

    def _extract_tool_calls(self, response: str) -> list:
        """Extract tool calls from response text (simple heuristic)."""
        import re
        from claude_trace_schema import ToolCall
        from datetime import datetime

        tool_patterns = {
            'Read': r'(?:read|reading)\s+(?:file\s+)?`([^`]+)`',
            'Edit': r'(?:edit|editing)\s+(?:file\s+)?`([^`]+)`',
            'Bash': r'(?:run|running)\s+`([^`]+)`',
            'Grep': r'(?:search|grep)\s+for\s+["\']([^"\']+)["\']',
        }

        tool_calls = []
        for tool, pattern in tool_patterns.items():
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                tool_calls.append(ToolCall(
                    tool=tool,
                    args={'target': match.group(1)},
                    output="",
                    timestamp=datetime.now(),
                    reasoning=f"Extracted: {match.group(0)}"
                ))

        return tool_calls

    def train(self, prompt_file: str, output_name: Optional[str] = None):
        """Train model with PPO."""

        # Load model
        if self.model is None:
            self.load_model()

        # Load prompts
        prompts = self.load_prompts(prompt_file)

        # Setup output directory
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"claude-ppo-{timestamp}"

        output_dir = Path(self.config.output_dir) / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # PPO configuration
        ppo_config = PPOConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            ppo_epochs=self.config.ppo_epochs,
            kl_penalty=self.config.kl_penalty,
            cliprange=self.config.cliprange,
            cliprange_value=self.config.cliprange_value,
            vf_coef=self.config.vf_coef,
            gamma=self.config.gamma,
            lam=self.config.lam,
        )

        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

        logger.info("=" * 70)
        logger.info("🚀 Starting PPO Behavioral Alignment")
        logger.info("=" * 70)
        logger.info(f"Base model: {self.config.sft_model_path}")
        logger.info(f"Training prompts: {len(prompts)}")
        logger.info(f"Episodes: {self.config.num_episodes}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 70)

        # Training loop
        for episode in range(self.config.num_episodes):
            # Sample random prompt
            import random
            prompt = random.choice(prompts)

            # Generate response
            response, response_tensors = self.generate_response(prompt)

            # Compute reward
            reward = self.compute_reward(prompt, response)
            rewards = [torch.tensor(reward)]

            # PPO update
            query_tensors = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            self.ppo_trainer.step(
                queries=[query_tensors],
                responses=[response_tensors],
                scores=rewards
            )

            # Logging
            if episode % 10 == 0:
                logger.info(f"Episode {episode}/{self.config.num_episodes}")
                logger.info(f"  Reward: {reward:.3f}")
                logger.info(f"  Response preview: {response[:100]}...")

            # Save checkpoint
            if episode % self.config.save_freq == 0 and episode > 0:
                checkpoint_dir = output_dir / f"checkpoint-{episode}"
                self.model.save_pretrained(checkpoint_dir)
                logger.info(f"  💾 Saved checkpoint: {checkpoint_dir}")

        # Save final model
        final_dir = output_dir / "final"
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        logger.info(f"✅ Training complete! Saved: {final_dir}")

        return str(final_dir)


def main():
    """CLI entry point for PPO training."""
    import argparse

    parser = argparse.ArgumentParser(description="PPO training for behavioral alignment")
    parser.add_argument("--sft-model", required=True, help="Path to SFT checkpoint")
    parser.add_argument("--prompts", required=True, help="Path to prompts file (JSONL)")
    parser.add_argument("--output-name", help="Name for output model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--learning-rate", type=float, default=1.4e-5, help="Learning rate")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create config
    config = PPOTrainingConfig(
        sft_model_path=args.sft_model,
        num_episodes=args.episodes,
        learning_rate=args.learning_rate,
    )

    # Train
    trainer = ClaudePPOTrainer(config)
    model_path = trainer.train(args.prompts, args.output_name)

    print()
    print("=" * 70)
    print("✅ PPO TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Model saved: {model_path}")
    print()


if __name__ == "__main__":
    main()
