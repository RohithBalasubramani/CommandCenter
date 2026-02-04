"""
DPO Trainer for Command Center

Implements Direct Preference Optimization training with QLoRA for memory efficiency.
Uses HuggingFace TRL library for the training loop.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import DPO_CONFIG, QLORA_CONFIG, CHECKPOINTS_DIR, get_config

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result from a training run."""
    success: bool
    checkpoint_path: Optional[str] = None
    final_loss: Optional[float] = None
    train_samples: int = 0
    eval_samples: int = 0
    epochs_completed: int = 0
    error_message: Optional[str] = None
    metrics: dict = field(default_factory=dict)


class CommandCenterDPOTrainer:
    """
    DPO trainer for widget and fixture selection.

    Uses QLoRA for memory-efficient fine-tuning on consumer GPUs.
    """

    def __init__(self, config: Optional[dict] = None, config_name: str = "default"):
        """
        Initialize the trainer.

        Args:
            config: Custom configuration dict (overrides config_name)
            config_name: Name of preset config ("default", "small_gpu", "high_quality")
        """
        if config is None:
            config = get_config(config_name)

        self.config = config
        self.dpo_config = config.get("dpo", DPO_CONFIG)
        self.qlora_config = config.get("qlora", QLORA_CONFIG)
        self.base_model = config.get("base_model", DPO_CONFIG.get("base_model"))

        self.model = None
        self.tokenizer = None
        self.ref_model = None

    def _check_dependencies(self):
        """Check that required packages are installed."""
        missing = []
        try:
            import torch
        except ImportError:
            missing.append("torch")

        try:
            import transformers
        except ImportError:
            missing.append("transformers")

        try:
            import trl
        except ImportError:
            missing.append("trl")

        try:
            import peft
        except ImportError:
            missing.append("peft")

        try:
            import bitsandbytes
        except ImportError:
            missing.append("bitsandbytes")

        if missing:
            raise ImportError(
                f"Missing required packages: {', '.join(missing)}. "
                f"Install with: pip install {' '.join(missing)}"
            )

    def load_base_model(self):
        """Load the base model with QLoRA quantization."""
        self._check_dependencies()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        logger.info(f"Loading base model: {self.base_model}")

        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.qlora_config.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=getattr(torch, self.qlora_config.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_quant_type=self.qlora_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=self.qlora_config.get("bnb_4bit_use_double_quant", True),
        )

        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.dpo_config.get("lora_r", 16),
            lora_alpha=self.dpo_config.get("lora_alpha", 32),
            lora_dropout=self.dpo_config.get("lora_dropout", 0.05),
            target_modules=self.dpo_config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        logger.info("Model loaded successfully")

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        output_dir: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> TrainingResult:
        """
        Run DPO training.

        Args:
            train_dataset: HuggingFace Dataset with prompt/chosen/rejected columns
            eval_dataset: Optional validation dataset
            output_dir: Directory to save checkpoints
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            TrainingResult with training metrics
        """
        if self.model is None:
            self.load_base_model()

        from trl import DPOConfig, DPOTrainer

        if output_dir is None:
            output_dir = str(CHECKPOINTS_DIR)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting DPO training with {len(train_dataset)} samples")
        if eval_dataset:
            logger.info(f"Validation set: {len(eval_dataset)} samples")

        # Configure training arguments
        training_args = DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=self.dpo_config.get("batch_size", 4),
            per_device_eval_batch_size=self.dpo_config.get("batch_size", 4),
            gradient_accumulation_steps=self.dpo_config.get("gradient_accumulation_steps", 4),
            learning_rate=self.dpo_config.get("learning_rate", 5e-5),
            num_train_epochs=self.dpo_config.get("num_epochs", 3),
            warmup_ratio=self.dpo_config.get("warmup_ratio", 0.1),
            beta=self.dpo_config.get("beta", 0.1),
            max_length=self.dpo_config.get("max_length", 2048),
            max_prompt_length=self.dpo_config.get("max_prompt_length", 1024),
            optim=self.dpo_config.get("optim", "paged_adamw_8bit"),
            weight_decay=self.dpo_config.get("weight_decay", 0.01),
            max_grad_norm=self.dpo_config.get("max_grad_norm", 1.0),
            logging_steps=self.dpo_config.get("logging_steps", 10),
            save_steps=self.dpo_config.get("save_steps", 100),
            save_total_limit=self.dpo_config.get("save_total_limit", 3),
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=self.dpo_config.get("eval_steps", 50) if eval_dataset else None,
            bf16=True,
            remove_unused_columns=False,
            report_to="none",  # Disable wandb/tensorboard by default
        )

        # Create DPO trainer
        trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        try:
            # Run training
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            # Save final model
            final_path = Path(output_dir) / "final"
            trainer.save_model(str(final_path))
            self.tokenizer.save_pretrained(str(final_path))

            # Get metrics
            metrics = train_result.metrics
            final_loss = metrics.get("train_loss", None)

            logger.info(f"Training complete. Final loss: {final_loss}")
            logger.info(f"Model saved to: {final_path}")

            return TrainingResult(
                success=True,
                checkpoint_path=str(final_path),
                final_loss=final_loss,
                train_samples=len(train_dataset),
                eval_samples=len(eval_dataset) if eval_dataset else 0,
                epochs_completed=int(self.dpo_config.get("num_epochs", 3)),
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                train_samples=len(train_dataset),
            )

    def evaluate(self, eval_dataset) -> dict:
        """Evaluate the model on a dataset."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_base_model() first.")

        from trl import DPOConfig, DPOTrainer

        training_args = DPOConfig(
            output_dir="/tmp/eval",
            per_device_eval_batch_size=self.dpo_config.get("batch_size", 4),
            bf16=True,
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        return trainer.evaluate()

    def save_checkpoint(self, path: str):
        """Save a checkpoint."""
        if self.model is None:
            raise RuntimeError("No model to save.")

        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str):
        """Load from a checkpoint."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Load base model first (without quantization for inference)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load LoRA weights
        self.model = PeftModel.from_pretrained(self.model, path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        logger.info(f"Checkpoint loaded from: {path}")


def train_from_config(
    dataset,
    config_name: str = "default",
    output_dir: Optional[str] = None,
) -> TrainingResult:
    """
    Convenience function to train with a preset configuration.

    Args:
        dataset: HuggingFace DatasetDict with train/validation splits
        config_name: Preset config name
        output_dir: Output directory for checkpoints

    Returns:
        TrainingResult
    """
    trainer = CommandCenterDPOTrainer(config_name=config_name)
    trainer.load_base_model()

    train_data = dataset["train"] if "train" in dataset else dataset
    eval_data = dataset.get("validation") if hasattr(dataset, "get") else None

    return trainer.train(
        train_dataset=train_data,
        eval_dataset=eval_data,
        output_dir=output_dir,
    )
