"""
Command Center Reinforcement Learning Package

Implements continuous RL for widget and fixture selection improvement.

Core Components (Continuous RL):
- continuous: Main ContinuousRL coordinator
- experience_buffer: Thread-safe experience storage
- reward_signals: Reward computation from feedback
- background_trainer: Async training loop

Batch Training Components (DPO):
- config: Training hyperparameters and model configuration
- data_formatter: Convert feedback ratings to DPO training pairs
- dataset_builder: Build HuggingFace datasets from formatted data
- trainer: DPOTrainer wrapper with QLoRA support
- export: GGUF conversion for Ollama deployment
- online_learner: Batch retraining from production feedback
"""

__version__ = "0.2.0"

# Main continuous RL exports
from .continuous import (
    ContinuousRL,
    get_rl_system,
    init_rl_system,
    shutdown_rl_system,
)
from .experience_buffer import Experience, ExperienceBuffer, get_experience_buffer
from .reward_signals import RewardSignalAggregator, ImplicitSignalDetector
from .background_trainer import BackgroundTrainer

__all__ = [
    # Continuous RL
    "ContinuousRL",
    "get_rl_system",
    "init_rl_system",
    "shutdown_rl_system",
    "Experience",
    "ExperienceBuffer",
    "get_experience_buffer",
    "RewardSignalAggregator",
    "ImplicitSignalDetector",
    "BackgroundTrainer",
]
