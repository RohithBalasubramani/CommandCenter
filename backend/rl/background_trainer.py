"""
Background Trainer for Continuous RL — Two-Tier Architecture

Tier 1 — Low-Rank Scorer (milliseconds):
    Trains on every feedback event. A tiny PyTorch network (6,872 params)
    that learns score adjustments for widget selection. Runs on CPU.

Tier 2 — LoRA DPO Fine-Tuning (periodic):
    When enough preference pairs accumulate (≥50), triggers full LoRA
    adapter training on llama3.1:8b. Exports to GGUF, hot-swaps in Ollama.
    Runs on GPU.

Both tiers run in a background daemon thread, never blocking inference.
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from .config import CONTINUOUS_RL_CONFIG, CHECKPOINTS_DIR, TRAINING_DATA_DIR
from .reward_signals import RewardSignalAggregator

if TYPE_CHECKING:
    from .experience_buffer import Experience, ExperienceBuffer

logger = logging.getLogger(__name__)

# Tier 2 thresholds
LORA_MIN_PAIRS = int(CONTINUOUS_RL_CONFIG.get("lora_min_pairs", 50))
LORA_TRAIN_COOLDOWN = int(CONTINUOUS_RL_CONFIG.get("lora_train_cooldown", 3600))  # 1hr


class BackgroundTrainer:
    """
    Two-tier background trainer for continuous RL.

    Tier 1: Low-rank scorer — instant online learning from every feedback event
    Tier 2: LoRA DPO — periodic deep fine-tuning of the LLM itself
    """

    def __init__(
        self,
        buffer: "ExperienceBuffer",
        config: dict = None,
        on_training_step: Optional[Callable] = None,
    ):
        self.buffer = buffer
        self.config = config or CONTINUOUS_RL_CONFIG
        self.on_training_step = on_training_step

        self.reward_aggregator = RewardSignalAggregator()

        # Training state
        self.running = False
        self.training_steps = 0
        self.total_samples_trained = 0
        self.avg_reward_history = []

        # Tier 1: Low-rank scorer (lazy loaded to avoid import on worker init)
        self._scorer = None
        self._scorer_steps = 0

        # Tier 2: LoRA DPO
        self._dpo_pairs: list[dict] = []
        self._dpo_pairs_lock = threading.Lock()
        self._last_lora_train_time = 0
        self._lora_training = False
        self._lora_version = 0
        self._lora_stats = {
            "total_trainings": 0,
            "total_pairs_trained": 0,
            "last_loss": None,
            "current_version": 0,
            "last_training_time": None,
        }

        # Widget selector reference (set by ContinuousRL)
        self.widget_selector = None
        self.fixture_selector = None

        # Thread
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def scorer(self):
        """Lazy-load the low-rank scorer."""
        if self._scorer is None:
            from .lora_scorer import get_scorer
            self._scorer = get_scorer()
        return self._scorer

    def set_selectors(self, widget_selector, fixture_selector):
        """Set references to selectors for updating."""
        self.widget_selector = widget_selector
        self.fixture_selector = fixture_selector

    def start(self):
        """Start background training thread."""
        if self.running:
            logger.warning("Trainer already running")
            return

        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._training_loop,
            daemon=True,
            name="rl-background-trainer",
        )
        self._thread.start()
        logger.info("Background RL trainer started (Tier 1: scorer + Tier 2: LoRA DPO)")

    def stop(self, timeout: float = 5.0):
        """Stop training thread gracefully."""
        if not self.running:
            return

        logger.info("Stopping background trainer...")
        self._stop_event.set()
        self.running = False

        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Trainer thread did not stop cleanly")

        # Save scorer checkpoint
        if self._scorer:
            self._scorer._save_checkpoint()

        # Save pending DPO pairs
        self._save_dpo_pairs()

        logger.info("Background trainer stopped")

    # ================================================================
    # Main Training Loop
    # ================================================================

    def _training_loop(self):
        """Main training loop — runs both tiers."""
        poll_interval = self.config.get("poll_interval", 5)
        train_interval = self.config.get("train_interval", 60)
        min_batch_size = self.config.get("min_batch_size", 16)
        batch_size = self.config.get("batch_size", 32) if "batch_size" in self.config else min_batch_size * 2

        last_train_time = 0

        # Load any saved DPO pairs from disk
        self._load_dpo_pairs()

        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                # Check if enough time has passed for Tier 1 batch
                if current_time - last_train_time < train_interval:
                    time.sleep(poll_interval)
                    continue

                # Get batch of experiences with feedback
                batch = self.buffer.sample_batch(batch_size, require_feedback=True)
                if len(batch) < min_batch_size:
                    logger.debug(f"Not enough feedback: {len(batch)} < {min_batch_size}")
                    time.sleep(poll_interval)
                    continue

                # Compute rewards
                rewards = [self.reward_aggregator.compute_reward(e) for e in batch]
                avg_reward = sum(rewards) / len(rewards)
                self.avg_reward_history.append(avg_reward)
                if len(self.avg_reward_history) > 100:
                    self.avg_reward_history = self.avg_reward_history[-100:]

                # ── Tier 1: Update low-rank scorer ──
                self._tier1_update(batch, rewards)

                # ── Tier 2: Accumulate DPO pairs & trigger LoRA when ready ──
                self._tier2_accumulate(batch, rewards)
                self._tier2_maybe_train()

                # Update fixture selector (rule-based, lightweight)
                if self.config.get("train_fixture_selector", True) and self.fixture_selector:
                    self._update_fixture_selector(batch, rewards)

                # Tracking
                self.training_steps += 1
                self.total_samples_trained += len(batch)
                last_train_time = current_time

                logger.info(
                    f"Training step {self.training_steps}: batch={len(batch)}, "
                    f"avg_reward={avg_reward:.3f}, scorer_steps={self._scorer_steps}, "
                    f"dpo_pairs={len(self._dpo_pairs)}"
                )

                if self.on_training_step:
                    self.on_training_step({
                        "step": self.training_steps,
                        "batch_size": len(batch),
                        "avg_reward": avg_reward,
                        "total_samples": self.total_samples_trained,
                        "scorer_steps": self._scorer_steps,
                        "dpo_pairs_pending": len(self._dpo_pairs),
                    })

            except Exception as e:
                logger.error(f"Training error: {e}", exc_info=True)
                time.sleep(10)

    # ================================================================
    # Tier 1: Low-Rank Scorer (online learning)
    # ================================================================

    def _tier1_update(self, batch: list["Experience"], rewards: list[float]):
        """
        Train low-rank scorer on each experience in the batch.

        Each (transcript, scenario, reward) triggers one gradient step.
        """
        for exp, reward in zip(batch, rewards):
            transcript = exp.transcript or ""
            widget_plan = exp.widget_plan or {}

            # Train on each widget that was selected
            for widget in widget_plan.get("widgets", []):
                scenario = widget.get("scenario", "")
                if scenario:
                    self.scorer.train_step(transcript, scenario, reward)
                    self._scorer_steps += 1

    # ================================================================
    # Tier 2: LoRA DPO (batch fine-tuning)
    # ================================================================

    def _tier2_accumulate(self, batch: list["Experience"], rewards: list[float]):
        """
        Build DPO preference pairs from positive/negative experiences.

        A pair is (prompt, chosen_response, rejected_response) where
        "chosen" had higher reward than "rejected".
        """
        positive = [(e, r) for e, r in zip(batch, rewards) if r > 0]
        negative = [(e, r) for e, r in zip(batch, rewards) if r < 0]

        if not positive or not negative:
            return

        with self._dpo_pairs_lock:
            for pos_e, pos_r in positive[:10]:
                for neg_e, neg_r in negative[:10]:
                    if self._intents_similar(pos_e.parsed_intent, neg_e.parsed_intent):
                        self._dpo_pairs.append({
                            "prompt": self._format_widget_prompt(pos_e),
                            "chosen": json.dumps(pos_e.widget_plan or {}),
                            "rejected": json.dumps(neg_e.widget_plan or {}),
                            "pos_reward": pos_r,
                            "neg_reward": neg_r,
                            "timestamp": time.time(),
                        })

            # Cap stored pairs to prevent unbounded memory growth
            if len(self._dpo_pairs) > 5000:
                self._dpo_pairs = self._dpo_pairs[-5000:]

    def _tier2_maybe_train(self):
        """Check if we have enough pairs to trigger LoRA DPO training."""
        if self._lora_training:
            return  # Already training

        if len(self._dpo_pairs) < LORA_MIN_PAIRS:
            return  # Not enough pairs yet

        if time.time() - self._last_lora_train_time < LORA_TRAIN_COOLDOWN:
            return  # Cool down between training runs

        # Human approval gate: require flag file to proceed
        approval_file = TRAINING_DATA_DIR / "approve_lora_training"
        if not approval_file.exists():
            logger.info(
                f"Tier 2 LoRA training ready ({len(self._dpo_pairs)} pairs) but waiting "
                f"for approval. Create {approval_file} to approve, or POST to "
                f"/api/layer2/approve-training/"
            )
            return

        # Consume approval (one-shot)
        try:
            approval_file.unlink()
        except OSError:
            pass

        # Trigger LoRA DPO training in a separate thread
        logger.info(f"Triggering Tier 2 LoRA DPO training with {len(self._dpo_pairs)} pairs (approved)")
        train_thread = threading.Thread(
            target=self._run_lora_training,
            daemon=True,
            name="rl-lora-trainer",
        )
        train_thread.start()

    def _run_lora_training(self):
        """
        Run full LoRA DPO training and deploy to Ollama.

        This runs in a separate thread and can take 10-30 minutes.
        """
        self._lora_training = True
        self._last_lora_train_time = time.time()
        start_time = time.time()

        try:
            # Snapshot pairs for training
            with self._dpo_pairs_lock:
                training_pairs = list(self._dpo_pairs)
                self._dpo_pairs.clear()

            logger.info(f"Starting LoRA DPO training with {len(training_pairs)} pairs")

            # Build HuggingFace dataset from pairs
            dataset = self._pairs_to_dataset(training_pairs)
            if dataset is None or len(dataset) < 10:
                logger.warning("Too few valid training pairs, skipping")
                with self._dpo_pairs_lock:
                    self._dpo_pairs.extend(training_pairs)  # Put back
                return

            # Split train/eval
            split = dataset.train_test_split(test_size=0.1, seed=42)
            train_ds = split["train"]
            eval_ds = split["test"]

            # Configure for incremental training
            version = self._lora_version + 1
            output_dir = str(CHECKPOINTS_DIR / f"lora_v{version}")

            # Run DPO training
            from .trainer import CommandCenterDPOTrainer

            trainer = CommandCenterDPOTrainer()
            result = trainer.train(
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                output_dir=output_dir,
            )

            if result.success:
                logger.info(
                    f"LoRA training complete: loss={result.final_loss:.4f}, "
                    f"samples={result.train_samples}, "
                    f"duration={time.time() - start_time:.0f}s"
                )

                # Evaluation gate: reject models with poor convergence
                MAX_ACCEPTABLE_LOSS = 0.7
                if result.final_loss is not None and result.final_loss > MAX_ACCEPTABLE_LOSS:
                    logger.warning(
                        f"LoRA eval gate FAILED: loss {result.final_loss:.4f} > "
                        f"threshold {MAX_ACCEPTABLE_LOSS}. Skipping deployment."
                    )
                    with self._dpo_pairs_lock:
                        self._dpo_pairs.extend(training_pairs)
                    self._lora_training = False
                    return

                # Deploy to Ollama
                self._deploy_lora(output_dir, version)

                # Update stats
                self._lora_version = version
                self._lora_stats["total_trainings"] += 1
                self._lora_stats["total_pairs_trained"] += len(training_pairs)
                self._lora_stats["last_loss"] = result.final_loss
                self._lora_stats["current_version"] = version
                self._lora_stats["last_training_time"] = time.time()

                # Clean old versions
                self._cleanup_old_versions(keep=2)

            else:
                logger.error(f"LoRA training failed: {result.error_message}")
                # Put pairs back for retry
                with self._dpo_pairs_lock:
                    self._dpo_pairs.extend(training_pairs)

        except Exception as e:
            logger.error(f"LoRA training error: {e}", exc_info=True)
        finally:
            self._lora_training = False

    def _pairs_to_dataset(self, pairs: list[dict]):
        """Convert DPO pairs to HuggingFace dataset."""
        try:
            from datasets import Dataset

            records = []
            for pair in pairs:
                records.append({
                    "prompt": pair["prompt"],
                    "chosen": pair["chosen"],
                    "rejected": pair["rejected"],
                })

            if not records:
                return None

            return Dataset.from_list(records)
        except ImportError:
            logger.error("datasets package not installed")
            return None

    def _deploy_lora(self, checkpoint_dir: str, version: int):
        """Deploy trained LoRA to Ollama via GGUF export."""
        try:
            from .export import export_to_ollama

            model_name = f"cc-widget-selector-v{version}"

            result = export_to_ollama(
                checkpoint_path=str(Path(checkpoint_dir) / "final"),
                model_name=model_name,
                register=True,
            )

            if result.success:
                logger.info(f"Deployed LoRA v{version} to Ollama as '{model_name}'")

                # Update the model reference for the widget selector
                import os
                os.environ["OLLAMA_MODEL_FAST"] = model_name
                logger.info(f"Updated OLLAMA_MODEL_FAST → {model_name}")
            else:
                logger.error(f"LoRA deployment failed: {result.error_message}")

        except Exception as e:
            logger.error(f"LoRA deployment error: {e}", exc_info=True)

    def _cleanup_old_versions(self, keep: int = 2):
        """Remove old LoRA checkpoints, keeping the most recent N."""
        import shutil
        versions = sorted(
            [d for d in CHECKPOINTS_DIR.glob("lora_v*") if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
        )
        for old in versions[:-keep]:
            logger.info(f"Cleaning up old checkpoint: {old.name}")
            shutil.rmtree(old, ignore_errors=True)

    def _save_dpo_pairs(self):
        """Persist pending DPO pairs to disk for recovery."""
        if not self._dpo_pairs:
            return
        TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = TRAINING_DATA_DIR / "pending_dpo_pairs.json"
        with self._dpo_pairs_lock:
            with open(path, "w") as f:
                json.dump(self._dpo_pairs, f)
        logger.info(f"Saved {len(self._dpo_pairs)} pending DPO pairs to disk")

    def _load_dpo_pairs(self):
        """Load pending DPO pairs from disk."""
        path = TRAINING_DATA_DIR / "pending_dpo_pairs.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                pairs = json.load(f)
            with self._dpo_pairs_lock:
                self._dpo_pairs.extend(pairs)
            logger.info(f"Loaded {len(pairs)} pending DPO pairs from disk")
        except Exception as e:
            logger.warning(f"Failed to load DPO pairs: {e}")

    # ================================================================
    # Fixture Selector (rule-based)
    # ================================================================

    def _update_fixture_selector(self, batch: list["Experience"], rewards: list[float]):
        """Update fixture selector scoring weights."""
        fixture_rewards: dict[tuple[str, str], list[float]] = {}

        for exp, reward in zip(batch, rewards):
            for scenario, fixture in exp.fixtures.items():
                key = (scenario, fixture)
                if key not in fixture_rewards:
                    fixture_rewards[key] = []
                fixture_rewards[key].append(reward)

        if hasattr(self.fixture_selector, "update_preferences"):
            for (scenario, fixture), rewards_list in fixture_rewards.items():
                avg_reward = sum(rewards_list) / len(rewards_list)
                self.fixture_selector.update_preferences(scenario, fixture, avg_reward)
            logger.debug(f"Updated fixture preferences for {len(fixture_rewards)} fixtures")

    # ================================================================
    # Helpers
    # ================================================================

    def _intents_similar(self, intent1: dict, intent2: dict) -> bool:
        """Check if two intents are similar enough to compare."""
        domains1 = set(intent1.get("domains", []))
        domains2 = set(intent2.get("domains", []))
        if not domains1 & domains2:
            return False
        if intent1.get("type") != intent2.get("type"):
            return False
        return True

    def _format_widget_prompt(self, experience: "Experience") -> str:
        """
        Format experience into widget selection prompt.

        Enhanced with rich evaluation data from Claude Sonnet 4.5 when available.
        """
        intent = experience.parsed_intent or {}
        lines = [
            f"User query: {experience.transcript}",
            f"Domains: {', '.join(intent.get('domains', []))}",
        ]
        entities = intent.get("entities", {})
        if entities:
            lines.append(f"Entities: {', '.join(f'{k}={v}' for k, v in entities.items())}")

        # Add rich evaluation context from Claude Sonnet 4.5
        if experience.query_understanding:
            lines.append(f"Goal: {experience.query_understanding}")
        if experience.missing_widgets:
            missing = ", ".join(str(w) for w in experience.missing_widgets)
            lines.append(f"Consider adding: {missing}")
        if experience.suggested_improvements:
            # Include top 2 suggestions
            suggestions = experience.suggested_improvements[:2]
            if suggestions:
                lines.append(f"Improvements: {'; '.join(str(s) for s in suggestions)}")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get training statistics for both tiers."""
        scorer_stats = self._scorer.get_stats() if self._scorer else {}

        return {
            "running": self.running,
            "training_steps": self.training_steps,
            "total_samples_trained": self.total_samples_trained,
            "avg_reward_trend": (
                sum(self.avg_reward_history[-10:]) / max(len(self.avg_reward_history[-10:]), 1)
                if self.avg_reward_history else 0
            ),
            "recent_rewards": self.avg_reward_history[-10:] if self.avg_reward_history else [],
            # Tier 1 stats
            "tier1_scorer": scorer_stats,
            # Tier 2 stats
            "tier2_lora": {
                "training_in_progress": self._lora_training,
                "pending_pairs": len(self._dpo_pairs),
                "min_pairs_for_training": LORA_MIN_PAIRS,
                **self._lora_stats,
            },
        }
