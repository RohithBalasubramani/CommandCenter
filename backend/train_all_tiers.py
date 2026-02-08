#!/usr/bin/env python3
"""
Manual Training Script for All Three RL Tiers

This script manually triggers training for:
- Tier 1: Low-rank scorer (online learning)
- Tier 2: Unified DPO (widget + voice pairs)
- Tier 3: Reasoning distillation SFT (Claude thinking traces)
"""

import logging
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training_all_tiers.log')
    ]
)
logger = logging.getLogger(__name__)

from rl.experience_buffer import ExperienceBuffer
from rl.background_trainer import BackgroundTrainer
from rl.config import CONTINUOUS_RL_CONFIG


def train_tier1(trainer: BackgroundTrainer, buffer: ExperienceBuffer):
    """
    Train Tier 1: Low-rank scorer on all available experiences.
    """
    logger.info("=" * 60)
    logger.info("TIER 1: LOW-RANK SCORER TRAINING")
    logger.info("=" * 60)

    # Get all experiences with feedback directly from buffer
    all_experiences = [e for e in buffer.buffer if e.has_feedback()]

    logger.info(f"Found {len(all_experiences)} experiences with feedback")

    if len(all_experiences) == 0:
        logger.warning("No experiences with feedback available for Tier 1 training")
        return

    # Process in batches for Tier 1 training
    batch_size = 32
    total_trained = 0

    for i in range(0, len(all_experiences), batch_size):
        batch = all_experiences[i:i+batch_size]

        # Compute rewards
        from rl.reward_signals import RewardSignalAggregator
        aggregator = RewardSignalAggregator()
        rewards = [aggregator.compute_reward(e) for e in batch]

        # Train Tier 1
        trainer._tier1_update(batch, rewards)
        trainer._tier1b_composition_update(batch, rewards)

        total_trained += len(batch)
        logger.info(f"Trained batch {i//batch_size + 1}: {len(batch)} samples, "
                   f"progress: {total_trained}/{len(all_experiences)}")

    # Save checkpoint
    if trainer._scorer:
        trainer._scorer._save_checkpoint()
        logger.info("Tier 1 scorer checkpoint saved")

    if trainer._composition_scorer:
        trainer._composition_scorer._save_checkpoint()
        logger.info("Tier 1 composition scorer checkpoint saved")

    logger.info(f"Tier 1 training complete: {total_trained} samples trained")
    logger.info(f"Scorer steps: {trainer._scorer_steps}")
    logger.info("")


def train_tier2(trainer: BackgroundTrainer, buffer: ExperienceBuffer):
    """
    Train Tier 2: Unified DPO from accumulated widget + voice pairs.
    """
    logger.info("=" * 60)
    logger.info("TIER 2: UNIFIED DPO TRAINING")
    logger.info("=" * 60)

    # Get all experiences with feedback directly from buffer
    all_experiences = [e for e in buffer.buffer if e.has_feedback()]

    logger.info(f"Accumulating DPO pairs from {len(all_experiences)} experiences...")

    # Accumulate pairs in batches
    batch_size = 32
    for i in range(0, len(all_experiences), batch_size):
        batch = all_experiences[i:i+batch_size]

        # Compute rewards
        from rl.reward_signals import RewardSignalAggregator
        aggregator = RewardSignalAggregator()
        rewards = [aggregator.compute_reward(e) for e in batch]

        # Accumulate widget and voice pairs
        trainer._accumulate_widget_pairs(batch, rewards)
        trainer._accumulate_voice_pairs(batch, rewards)

    # Check how many pairs we have
    widget_count = sum(1 for p in trainer._dpo_pairs if p.get("pair_type") == "widget")
    voice_count = sum(1 for p in trainer._dpo_pairs if p.get("pair_type") == "voice")
    total_pairs = len(trainer._dpo_pairs)

    logger.info(f"Accumulated DPO pairs: {total_pairs} total "
               f"(widget={widget_count}, voice={voice_count})")

    # Check if we have enough pairs to train
    from rl.background_trainer import DPO_MIN_PAIRS
    if total_pairs < DPO_MIN_PAIRS:
        logger.warning(f"Not enough DPO pairs for training: {total_pairs} < {DPO_MIN_PAIRS}")
        logger.warning("Skipping Tier 2 training")
        return

    # Force DPO training by calling the method directly
    logger.info(f"Starting DPO training with {total_pairs} pairs...")

    try:
        trainer._run_dpo_training()
        logger.info("Tier 2 DPO training complete!")
        logger.info(f"DPO stats: {trainer._dpo_stats}")
    except Exception as e:
        logger.error(f"DPO training failed: {e}", exc_info=True)

    logger.info("")


def train_tier3(trainer: BackgroundTrainer):
    """
    Train Tier 3: Reasoning distillation SFT from Claude thinking traces.
    """
    logger.info("=" * 60)
    logger.info("TIER 3: REASONING DISTILLATION SFT")
    logger.info("=" * 60)

    # Check for available traces
    trace_dir = Path('../claude-rl-agent/data/v4_traces')
    if not trace_dir.exists():
        logger.warning("Trace directory not found")
        logger.warning("Skipping Tier 3 training (no traces available)")
        return

    # Read traces from traces.jsonl file
    trace_file = trace_dir / 'traces.jsonl'
    traces = []
    if trace_file.exists():
        import json
        with open(trace_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        traces.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line in traces.jsonl")

    logger.info(f"Found {len(traces)} thinking traces")

    # Check minimum threshold
    from rl.config import CONTINUOUS_RL_CONFIG
    min_traces = CONTINUOUS_RL_CONFIG.get('tier3_min_traces', 100)

    if len(traces) < min_traces:
        logger.warning(f"Not enough traces for training: {len(traces)} < {min_traces}")
        logger.warning("Skipping Tier 3 training")
        return

    # Trigger SFT training
    logger.info(f"Starting SFT training with {len(traces)} traces...")

    try:
        trainer._maybe_train_sft()
        logger.info("Tier 3 SFT training triggered!")
    except Exception as e:
        logger.error(f"SFT training failed: {e}", exc_info=True)

    logger.info("")


def main():
    logger.info("=" * 60)
    logger.info("MANUAL RL TRAINING - ALL THREE TIERS")
    logger.info("=" * 60)
    logger.info("")

    # Initialize components
    logger.info("Initializing components...")
    buffer = ExperienceBuffer()
    trainer = BackgroundTrainer(buffer, config=CONTINUOUS_RL_CONFIG)

    logger.info(f"Experience buffer: {buffer.size()} total, {buffer.feedback_count()} with feedback")
    logger.info("")

    # Train each tier
    try:
        # Tier 1: Low-rank scorer
        train_tier1(trainer, buffer)

        # Tier 2: Unified DPO
        train_tier2(trainer, buffer)

        # Tier 3: Reasoning distillation
        train_tier3(trainer)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
