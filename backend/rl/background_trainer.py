"""
Background Trainer for Continuous RL

Runs in a separate thread, continuously training from the experience buffer.
"""

import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional

from .config import CONTINUOUS_RL_CONFIG
from .reward_signals import RewardSignalAggregator

if TYPE_CHECKING:
    from .experience_buffer import Experience, ExperienceBuffer

logger = logging.getLogger(__name__)


class BackgroundTrainer:
    """
    Runs in separate thread, continuously improves widget/fixture selection.

    Training happens in parallel with serving - no blocking on the main thread.
    """

    def __init__(
        self,
        buffer: "ExperienceBuffer",
        config: dict = None,
        on_training_step: Optional[Callable] = None,
    ):
        """
        Initialize the background trainer.

        Args:
            buffer: Experience buffer to sample from
            config: Training configuration (defaults to CONTINUOUS_RL_CONFIG)
            on_training_step: Optional callback after each training step
        """
        self.buffer = buffer
        self.config = config or CONTINUOUS_RL_CONFIG
        self.on_training_step = on_training_step

        self.reward_aggregator = RewardSignalAggregator()

        # Training state
        self.running = False
        self.training_steps = 0
        self.total_samples_trained = 0
        self.avg_reward_history = []

        # Widget selector reference (set by ContinuousRL)
        self.widget_selector = None
        self.fixture_selector = None

        # Thread
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

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
        logger.info("Background RL trainer started")

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

        logger.info("Background trainer stopped")

    def _training_loop(self):
        """Main training loop - runs continuously."""
        poll_interval = self.config.get("poll_interval", 5)
        train_interval = self.config.get("train_interval", 60)
        min_batch_size = self.config.get("min_batch_size", 16)
        batch_size = self.config.get("batch_size", 32) if "batch_size" in self.config else min_batch_size * 2

        last_train_time = 0

        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                # Check if enough time has passed
                if current_time - last_train_time < train_interval:
                    time.sleep(poll_interval)
                    continue

                # Check if we have enough experiences
                batch = self.buffer.sample_batch(batch_size, require_feedback=True)
                if len(batch) < min_batch_size:
                    logger.debug(f"Not enough experiences with feedback: {len(batch)} < {min_batch_size}")
                    time.sleep(poll_interval)
                    continue

                # Compute rewards
                rewards = [self.reward_aggregator.compute_reward(e) for e in batch]

                # Log average reward
                avg_reward = sum(rewards) / len(rewards)
                self.avg_reward_history.append(avg_reward)

                # Keep only last 100 reward averages
                if len(self.avg_reward_history) > 100:
                    self.avg_reward_history = self.avg_reward_history[-100:]

                # Update widget selector
                if self.config.get("train_widget_selector", True) and self.widget_selector:
                    self._update_widget_selector(batch, rewards)

                # Update fixture selector
                if self.config.get("train_fixture_selector", True) and self.fixture_selector:
                    self._update_fixture_selector(batch, rewards)

                # Update tracking
                self.training_steps += 1
                self.total_samples_trained += len(batch)
                last_train_time = current_time

                logger.info(
                    f"Training step {self.training_steps}: "
                    f"batch={len(batch)}, avg_reward={avg_reward:.3f}"
                )

                # Callback
                if self.on_training_step:
                    self.on_training_step({
                        "step": self.training_steps,
                        "batch_size": len(batch),
                        "avg_reward": avg_reward,
                        "total_samples": self.total_samples_trained,
                    })

            except Exception as e:
                logger.error(f"Training error: {e}", exc_info=True)
                time.sleep(10)  # Back off on error

    def _update_widget_selector(self, batch: list["Experience"], rewards: list[float]):
        """
        Update widget selector using preference learning.

        Creates DPO-style pairs from positive and negative experiences.
        """
        # Split into positive and negative
        positive = [(e, r) for e, r in zip(batch, rewards) if r > 0]
        negative = [(e, r) for e, r in zip(batch, rewards) if r < 0]

        if not positive or not negative:
            logger.debug("No positive/negative pairs for widget selector update")
            return

        # Create preference pairs (positive vs negative for similar queries)
        pairs = []
        for pos_e, pos_r in positive[:10]:
            for neg_e, neg_r in negative[:10]:
                # Only pair experiences with similar intents
                if self._intents_similar(pos_e.parsed_intent, neg_e.parsed_intent):
                    pairs.append({
                        "prompt": self._format_widget_prompt(pos_e),
                        "chosen": json.dumps(pos_e.widget_plan),
                        "rejected": json.dumps(neg_e.widget_plan),
                        "pos_reward": pos_r,
                        "neg_reward": neg_r,
                    })

        if not pairs:
            return

        # Update widget selector if it supports preference learning
        if hasattr(self.widget_selector, "update_from_preferences"):
            self.widget_selector.update_from_preferences(pairs)
            logger.debug(f"Updated widget selector with {len(pairs)} preference pairs")
        else:
            # Fallback: store pairs for later batch training
            self._store_preference_pairs("widget", pairs)

    def _update_fixture_selector(self, batch: list["Experience"], rewards: list[float]):
        """
        Update fixture selector scoring weights.

        The fixture selector uses rule-based scoring, so we update the weights
        based on which fixtures led to positive/negative outcomes.
        """
        # Group rewards by (scenario, fixture)
        fixture_rewards: dict[tuple[str, str], list[float]] = {}

        for exp, reward in zip(batch, rewards):
            for scenario, fixture in exp.fixtures.items():
                key = (scenario, fixture)
                if key not in fixture_rewards:
                    fixture_rewards[key] = []
                fixture_rewards[key].append(reward)

        # Update fixture selector if it supports preference updates
        if hasattr(self.fixture_selector, "update_preferences"):
            for (scenario, fixture), rewards_list in fixture_rewards.items():
                avg_reward = sum(rewards_list) / len(rewards_list)
                self.fixture_selector.update_preferences(scenario, fixture, avg_reward)
            logger.debug(f"Updated fixture preferences for {len(fixture_rewards)} fixtures")

    def _intents_similar(self, intent1: dict, intent2: dict) -> bool:
        """Check if two intents are similar enough to compare."""
        # Same domains
        domains1 = set(intent1.get("domains", []))
        domains2 = set(intent2.get("domains", []))
        if not domains1 & domains2:
            return False

        # Same intent type
        if intent1.get("type") != intent2.get("type"):
            return False

        return True

    def _format_widget_prompt(self, experience: "Experience") -> str:
        """Format experience into widget selection prompt."""
        intent = experience.parsed_intent
        lines = [
            f"User query: {experience.transcript}",
            f"Domains: {', '.join(intent.get('domains', []))}",
        ]

        entities = intent.get("entities", {})
        if entities:
            lines.append(f"Entities: {', '.join(f'{k}={v}' for k, v in entities.items())}")

        if experience.user_history:
            lines.append(f"Recent queries: {len(experience.user_history)}")

        return "\n".join(lines)

    def _store_preference_pairs(self, pair_type: str, pairs: list[dict]):
        """Store preference pairs for later batch training."""
        # This could write to a file or database for later DPO training
        pass

    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            "running": self.running,
            "training_steps": self.training_steps,
            "total_samples_trained": self.total_samples_trained,
            "avg_reward_trend": (
                sum(self.avg_reward_history[-10:]) / max(len(self.avg_reward_history[-10:]), 1)
                if self.avg_reward_history else 0
            ),
            "recent_rewards": self.avg_reward_history[-10:] if self.avg_reward_history else [],
        }
