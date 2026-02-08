#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) Trainer for Behavioral Cloning

Trains LLaMA 3.1 8B to replicate Claude's workflow design patterns using QLoRA.

Key training distinctions:
  - Completion-only loss: only train on assistant response tokens (not system/user)
  - Thinking vs Direct: thinking traces can have thinking tokens masked or downweighted
  - Upsampling: thinking traces are upsampled to balance the dataset
  - Two-phase option: Phase 1 trains on everything, Phase 2 masks thinking tokens
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

try:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    from transformers import TrainingArguments
    from datasets import Dataset
except ImportError as e:
    print(f"Missing dependencies. Install with:")
    print(f"   pip install unsloth trl transformers datasets")
    raise e

try:
    from .config import AGENT_ROOT, MODELS_DIR, DATASETS_DIR
except ImportError:
    from config import AGENT_ROOT, MODELS_DIR, DATASETS_DIR

logger = logging.getLogger(__name__)

# The assistant header token sequence — everything before this is masked from loss
RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n"


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    # Model settings
    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct"
    max_seq_length: int = 4096
    load_in_4bit: bool = True

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = None

    # Training settings
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 10
    max_steps: int = -1  # -1 means train for full epochs

    # Output settings
    output_dir: str = str(MODELS_DIR / "sft_checkpoints")
    save_steps: int = 50
    logging_steps: int = 10

    # Whether to use completion-only loss (mask system+user prompt tokens)
    # Should almost always be True — only train on assistant response
    completion_only: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class ClaudeSFTTrainer:
    """Trains LLaMA to replicate Claude's behavioral patterns.

    Each trace produces up to three independent SFT samples:
    - type=thinking:    prompt → reasoning (teaches HOW to think)
    - type=answer:      prompt → answer JSON (teaches WHAT to output)
    - type=consistency: prompt + reasoning → answer JSON (teaches answer follows reasoning)

    All use completion-only loss (only train on assistant response tokens).
    """

    def __init__(self, config: Optional[SFTConfig] = None):
        self.config = config or SFTConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model(self):
        """Load base model with LoRA adapters."""
        logger.info(f"Loading base model: {self.config.base_model}")

        import torch
        device_map = {"": 0} if torch.cuda.is_available() else "cpu"

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.load_in_4bit,
            device_map=device_map,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add LoRA adapters
        logger.info(f"Adding LoRA adapters (r={self.config.lora_r})")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=self.config.target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        logger.info("Model loaded with LoRA adapters")

    def load_dataset_phases(self, dataset_path: str):
        """Load dataset and split into curriculum phases.

        Returns three datasets in training order:
          Phase 1: answer samples (schema discipline)
          Phase 2: thinking samples (reasoning skill)
          Phase 3: consistency samples (bridge reasoning → answer)

        Each phase is shuffled internally.
        """
        logger.info(f"Loading dataset: {dataset_path}")

        samples = []
        with open(dataset_path, 'r') as f:
            for line in f:
                samples.append(json.loads(line))

        # Split by type
        answer_samples = [s for s in samples if s.get("type") == "answer"]
        thinking_samples = [s for s in samples if s.get("type") == "thinking"]
        consistency_samples = [s for s in samples if s.get("type") == "consistency"]

        logger.info(
            f"Loaded {len(samples)} samples "
            f"({len(thinking_samples)} thinking, {len(answer_samples)} answer, "
            f"{len(consistency_samples)} consistency)"
        )

        import random
        random.seed(42)

        phases = []
        phase_names = []
        for name, group in [("answer", answer_samples), ("thinking", thinking_samples), ("consistency", consistency_samples)]:
            if not group:
                continue
            random.shuffle(group)
            formatted = [{"text": self._format_training_sample(s)} for s in group]
            phases.append(Dataset.from_list(formatted))
            phase_names.append(name)

        logger.info(f"Curriculum phases: {' → '.join(f'{n}({len(p)})' for n, p in zip(phase_names, phases))}")

        return phases, phase_names

    def _format_training_sample(self, sample: Dict) -> str:
        """Format a sample into LLaMA 3.1 instruction-tuning format.

        Three sample types from the same prompt, trained separately:

          type=thinking → teaches HOW to reason about the query
            system: "Think through this step by step..."
            target: Claude's raw reasoning chain

          type=answer → teaches the EXACT output to produce
            system: "respond with ONLY valid JSON"
            target: Claude's final JSON answer

          type=consistency → teaches answer must follow from reasoning
            system: "produce JSON that reflects your analysis"
            user: original prompt + thinking (as context)
            target: Claude's final JSON answer
        """
        sample_type = sample.get("type", "answer")

        if sample_type == "thinking":
            system_prompt = (
                "You are a dashboard widget selector for an industrial operations command center. "
                "Think through the user's query step by step. Identify the relevant equipment, "
                "metrics, and time ranges. Determine which widgets best answer the query and why."
            )
            user_content = sample["prompt"]

        elif sample_type == "consistency":
            system_prompt = (
                "You are a dashboard widget selector for an industrial operations command center. "
                "Based on your reasoning below, produce the final JSON answer that reflects "
                "your analysis. You MUST respond with ONLY valid JSON, no explanation or markdown."
            )
            # User content = original prompt + the thinking as context
            user_content = (
                f"{sample['prompt']}\n\n"
                f"Your reasoning:\n{sample['thinking']}"
            )

        else:  # answer
            system_prompt = (
                "You are a dashboard widget selector for an industrial operations command center. "
                "You select the best combination of widgets to build an informative dashboard for the user's query. "
                "You MUST respond with ONLY valid JSON, no explanation or markdown."
            )
            user_content = sample["prompt"]

        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}\n"
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_content}\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            f"{sample['target']}\n"
            f"<|eot_id|>"
        )

    def _build_data_collator(self):
        """Build completion-only data collator (masks system+user tokens)."""
        if not self.config.completion_only:
            return None
        try:
            response_ids = self.tokenizer.encode(
                RESPONSE_TEMPLATE, add_special_tokens=False
            )
            collator = DataCollatorForCompletionOnlyLM(
                response_template=response_ids,
                tokenizer=self.tokenizer,
            )
            logger.info("Completion-only loss: training on assistant response tokens only")
            return collator
        except Exception as e:
            logger.warning(f"Failed to create completion-only collator: {e}")
            return None

    def _train_phase(self, dataset: Dataset, phase_name: str, output_dir: Path, epochs: int):
        """Train one curriculum phase. Model weights carry over between phases."""
        phase_dir = output_dir / phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(phase_dir),
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            num_train_epochs=epochs,
            learning_rate=self.config.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=2,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",
        )

        data_collator = self._build_data_collator()

        trainer_kwargs = dict(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            args=training_args,
            packing=False,
        )
        if data_collator is not None:
            trainer_kwargs["data_collator"] = data_collator

        trainer = SFTTrainer(**trainer_kwargs)

        logger.info(f"Phase [{phase_name}]: {len(dataset)} samples, {epochs} epochs")
        trainer.train()
        logger.info(f"Phase [{phase_name}] complete")

    def train(self, dataset_path: str, output_name: Optional[str] = None):
        """Train the model with curriculum phases.

        Three phases, each building on the previous one's weights:
          Phase 1 (answer):      Learn correct JSON output format
          Phase 2 (thinking):    Learn how to reason about queries
          Phase 3 (consistency): Learn that answers must follow reasoning
        """
        if self.model is None:
            self.load_model()

        phases, phase_names = self.load_dataset_phases(dataset_path)

        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"claude-bc-{timestamp}"

        output_dir = Path(self.config.output_dir) / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Distribute epochs across phases
        total_epochs = self.config.num_epochs
        epochs_per_phase = max(1, total_epochs // len(phases)) if phases else 1

        logger.info("=" * 70)
        logger.info("Starting Curriculum SFT Training")
        logger.info("=" * 70)
        logger.info(f"Model: {self.config.base_model}")
        logger.info(f"Phases: {len(phases)} ({' → '.join(phase_names)})")
        logger.info(f"Epochs per phase: {epochs_per_phase}")
        logger.info(f"Effective batch: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 70)

        # Train each phase sequentially — same model, weights accumulate
        for i, (dataset, name) in enumerate(zip(phases, phase_names)):
            logger.info(f"\n{'='*70}")
            logger.info(f"PHASE {i+1}/{len(phases)}: {name}")
            logger.info(f"{'='*70}")
            self._train_phase(dataset, name, output_dir, epochs_per_phase)

        logger.info("All phases complete!")

        final_dir = output_dir / "final"
        logger.info(f"Saving final model to: {final_dir}")
        # Save using the last trainer's model (which has all accumulated weights)
        self.model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))

        return str(final_dir)

    def export_to_gguf(self, model_path: str, output_name: Optional[str] = None):
        """Export trained model to GGUF format for Ollama."""
        logger.info("=" * 70)
        logger.info("Exporting to GGUF Format")
        logger.info("=" * 70)

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if output_name is None:
            output_name = f"claude-bc-{datetime.now().strftime('%Y%m%d')}"

        export_dir = MODELS_DIR / "gguf"
        export_dir.mkdir(parents=True, exist_ok=True)
        output_file = export_dir / f"{output_name}.gguf"

        logger.info("Loading model for export...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )

        logger.info(f"Exporting to: {output_file}")
        model.save_pretrained_gguf(
            str(export_dir),
            tokenizer,
            quantization_method="q4_k_m",
        )

        exported_files = list(export_dir.glob("*.gguf"))
        if exported_files:
            exported_files[0].rename(output_file)
            logger.info(f"Exported to: {output_file}")
            return str(output_file)
        else:
            raise RuntimeError("Export failed - no GGUF file created")


def main():
    """CLI entry point for SFT training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train LLaMA with behavioral cloning")
    parser.add_argument("--dataset", required=True, help="Path to training dataset (JSONL)")
    parser.add_argument("--output-name", help="Name for output model")
    parser.add_argument("--export-gguf", action="store_true", help="Export to GGUF after training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    config = SFTConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    trainer = ClaudeSFTTrainer(config)
    model_path = trainer.train(args.dataset, args.output_name)

    if args.export_gguf:
        trainer.export_to_gguf(model_path, args.output_name)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Model saved: {model_path}")
    print()


if __name__ == "__main__":
    main()
