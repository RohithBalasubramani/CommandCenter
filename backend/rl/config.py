"""
Training Configuration for Command Center RL

Hyperparameters optimized for:
- llama3.1:8b as base model (matching OLLAMA_MODEL_FAST)
- QLoRA for memory-efficient training on consumer GPUs
- DPO for stable preference learning
"""

import os
from pathlib import Path

# Base paths
RL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = RL_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent
CHECKPOINTS_DIR = PROJECT_DIR / "rl_checkpoints"
TRAINING_DATA_DIR = PROJECT_DIR / "rl_training_data"

# Model configuration
MODEL_CONFIG = {
    # Base model - should match what Ollama uses
    "base_model": os.getenv("RL_BASE_MODEL", "unsloth/Meta-Llama-3.1-8B-Instruct"),
    # Alternative for smaller GPUs
    "base_model_small": "meta-llama/Llama-3.2-3B-Instruct",
    # Output model name for Ollama
    "output_model_name": os.getenv("RL_OUTPUT_MODEL", "cc-widget-selector"),
}

# DPO Training Configuration
DPO_CONFIG = {
    # LoRA parameters
    "lora_r": 16,  # Rank - higher = more capacity but more memory
    "lora_alpha": 32,  # Scaling factor, typically 2x lora_r
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    # Training parameters
    "learning_rate": 5e-5,
    "beta": 0.1,  # DPO temperature - lower = more aggressive preference learning
    "max_length": 2048,  # Max sequence length
    "max_prompt_length": 1024,  # Max prompt length (rest for response)
    "batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch = 16
    "num_epochs": 3,
    "warmup_ratio": 0.1,

    # Optimization
    "optim": "paged_adamw_8bit",  # Memory-efficient optimizer
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,

    # Logging and saving
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 50,
    "save_total_limit": 3,  # Keep only last 3 checkpoints
}

# QLoRA Quantization Configuration
QLORA_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",  # Normalized float 4-bit
    "bnb_4bit_use_double_quant": True,  # Nested quantization for more memory savings
}

# GGUF Export Configuration
GGUF_CONFIG = {
    # Quantization levels (trade-off: quality vs size)
    "quantization": os.getenv("RL_GGUF_QUANT", "q4_k_m"),
    # Options: q2_k, q3_k_s, q3_k_m, q4_0, q4_k_s, q4_k_m, q5_0, q5_k_s, q5_k_m, q6_k, q8_0, f16
    "available_quantizations": {
        "q4_k_m": {"desc": "Balanced quality/size (recommended)", "bits": 4.5},
        "q5_k_m": {"desc": "Higher quality", "bits": 5.5},
        "q8_0": {"desc": "Highest quality", "bits": 8.0},
        "q3_k_m": {"desc": "Smaller size", "bits": 3.5},
    },
}

# Online Learning Configuration (batch mode - legacy)
ONLINE_LEARNING_CONFIG = {
    "min_samples_to_retrain": 100,  # Minimum feedback samples before retraining
    "max_buffer_size": 10000,  # Maximum samples to hold in memory
    "retrain_interval_hours": 24,  # Maximum time between retraining
    "auto_export": True,  # Automatically export to GGUF after training
    "auto_deploy": False,  # Automatically swap Ollama model (dangerous in prod)
}

# Continuous RL Configuration (real-time learning)
CONTINUOUS_RL_CONFIG = {
    # Experience buffer
    "buffer_size": 10000,           # Max experiences in memory
    "min_batch_size": 16,           # Min experiences needed to train
    "redis_key_prefix": "cc_rl:",   # Redis key prefix for persistence

    # Background training
    "train_interval": 60,           # Seconds between training steps
    "poll_interval": 5,             # Seconds between buffer checks
    "train_widget_selector": True,  # Train widget selection policy
    "train_fixture_selector": True, # Train fixture selection weights

    # Reward weights (sum to ~2.0 for scale)
    "reward_weights": {
        "explicit_rating": 1.0,     # User thumbs up/down (strongest signal)
        "follow_up_type": 0.5,      # Did user need to repeat/refine?
        "widget_engagement": 0.3,   # Which widgets did user interact with?
        "response_latency": 0.1,    # Faster is better (within bounds)
        "intent_confidence": 0.1,   # Higher confidence = more reliable
        "text_quality": 0.4,        # Voice response quality
    },

    # Model updates
    "checkpoint_interval": 100,     # Training steps between checkpoints
    "deploy_interval": 500,         # Training steps between Ollama deploys
    "keep_versions": 2,             # Number of old model versions to keep

    # Safety
    "max_reward": 2.0,              # Clip reward to prevent outliers
    "min_reward": -2.0,
    "baseline_model_name": None,    # Keep baseline for A/B (set via env)

    # Tier 1: Low-rank scorer
    "scorer_rank": 32,              # Low-rank factorization rank (32 for query-specificity)
    "scorer_lr": 1e-3,              # Learning rate for scorer
    "scorer_checkpoint_every": 50,  # Checkpoint frequency (feedback events)

    # Tier 2: Unified LoRA DPO fine-tuning (widget + voice pairs)
    "lora_min_pairs": 80,            # Min DPO pairs before triggering training
    "lora_train_cooldown": 1800,    # 30 min cooldown between training runs
    "lora_auto_deploy": True,       # Auto-deploy to Ollama after training
}

# Training data configuration
TRAINING_DATA_CONFIG = {
    # Minimum confidence for including a pair
    "min_rating_confidence": 0.5,
    # Maximum pairs per question (to avoid bias)
    "max_pairs_per_question": 5,
    # Train/validation split
    "val_split": 0.1,
    # Seed for reproducibility
    "seed": 42,
}


def get_config(config_name: str = "default") -> dict:
    """Get merged configuration by name."""
    configs = {
        "default": {
            **MODEL_CONFIG,
            "dpo": DPO_CONFIG,
            "qlora": QLORA_CONFIG,
            "gguf": GGUF_CONFIG,
            "online": ONLINE_LEARNING_CONFIG,
            "data": TRAINING_DATA_CONFIG,
        },
        "small_gpu": {
            **MODEL_CONFIG,
            "base_model": MODEL_CONFIG["base_model_small"],
            "dpo": {**DPO_CONFIG, "batch_size": 2, "gradient_accumulation_steps": 8},
            "qlora": QLORA_CONFIG,
            "gguf": {**GGUF_CONFIG, "quantization": "q3_k_m"},
            "online": ONLINE_LEARNING_CONFIG,
            "data": TRAINING_DATA_CONFIG,
        },
        "high_quality": {
            **MODEL_CONFIG,
            "dpo": {**DPO_CONFIG, "num_epochs": 5, "lora_r": 32, "lora_alpha": 64},
            "qlora": QLORA_CONFIG,
            "gguf": {**GGUF_CONFIG, "quantization": "q5_k_m"},
            "online": ONLINE_LEARNING_CONFIG,
            "data": TRAINING_DATA_CONFIG,
        },
    }
    return configs.get(config_name, configs["default"])
