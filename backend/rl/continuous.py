"""
Continuous RL Coordinator for Command Center

Main entry point for the continuous reinforcement learning system.
Coordinates experience collection, reward computation, and background training.
"""

import logging
import os
import threading
from dataclasses import asdict
from datetime import datetime
from typing import Any, Optional

from .config import CONTINUOUS_RL_CONFIG
from .experience_buffer import Experience, ExperienceBuffer, get_experience_buffer
from .reward_signals import ImplicitSignalDetector, RewardSignalAggregator
from .background_trainer import BackgroundTrainer

logger = logging.getLogger(__name__)


class ContinuousRL:
    """
    Main coordinator for continuous reinforcement learning.

    Integrates with the orchestrator to:
    1. Record every interaction as an experience
    2. Process feedback (explicit and implicit)
    3. Run background training to improve selection
    """

    _instance: Optional["ContinuousRL"] = None
    _lock = threading.Lock()

    def __init__(self, config: dict = None):
        """
        Initialize the ContinuousRL system.

        Args:
            config: Configuration dict (defaults to CONTINUOUS_RL_CONFIG)
        """
        self.config = config or CONTINUOUS_RL_CONFIG

        # Core components
        self.buffer = get_experience_buffer()
        self.reward_aggregator = RewardSignalAggregator()
        self.implicit_detector = ImplicitSignalDetector()
        self.trainer: Optional[BackgroundTrainer] = None

        # State tracking
        self.running = False
        self._last_query_by_user: dict[str, tuple[str, dict]] = {}  # user_id -> (transcript, intent)

        # References to selectors (set when started)
        self.widget_selector = None
        self.fixture_selector = None

        logger.info("ContinuousRL initialized")

    @classmethod
    def get_instance(cls) -> "ContinuousRL":
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start(self, widget_selector=None, fixture_selector=None):
        """
        Start the continuous RL system.

        Args:
            widget_selector: Reference to the WidgetSelector for updates
            fixture_selector: Reference to the FixtureSelector for updates
        """
        if self.running:
            logger.warning("ContinuousRL already running")
            return

        self.widget_selector = widget_selector
        self.fixture_selector = fixture_selector

        # Create and start background trainer
        self.trainer = BackgroundTrainer(
            buffer=self.buffer,
            config=self.config,
            on_training_step=self._on_training_step,
        )
        self.trainer.set_selectors(widget_selector, fixture_selector)
        self.trainer.start()

        self.running = True
        logger.info("ContinuousRL started")

    def stop(self):
        """Stop the continuous RL system gracefully."""
        if not self.running:
            return

        logger.info("Stopping ContinuousRL...")

        if self.trainer:
            self.trainer.stop()

        # Save buffer to disk
        self.buffer.save()

        self.running = False
        logger.info("ContinuousRL stopped")

    def record_experience(
        self,
        query_id: str,
        transcript: str,
        user_id: str = "",
        parsed_intent: dict = None,
        widget_plan: dict = None,
        fixtures: dict = None,
        processing_time_ms: int = 0,
        available_data_summary: dict = None,
        user_history: list = None,
    ):
        """
        Record a new experience from the orchestrator.

        Called after each query is processed. Non-blocking.

        Args:
            query_id: Unique ID for this query (returned to frontend for feedback)
            transcript: Raw user query
            user_id: User identifier
            parsed_intent: Dict from IntentParser
            widget_plan: Dict from WidgetSelector
            fixtures: Dict of scenario -> fixture selections
            processing_time_ms: Total processing time
            available_data_summary: Summary of available data
            user_history: User's recent query history
        """
        # Detect implicit feedback from follow-up
        follow_up_type = None
        if user_id and user_id in self._last_query_by_user:
            prev_transcript, prev_intent = self._last_query_by_user[user_id]
            follow_up_type = self.implicit_detector.classify_follow_up(
                prev_intent,
                parsed_intent or {},
                prev_transcript,
                transcript,
            )

            # Update previous experience with follow-up type
            # (Find most recent experience for this user)
            recent = self.buffer.get_recent(20)
            for exp in reversed(recent):
                if exp.user_id == user_id and exp.query_id != query_id:
                    exp.follow_up_type = follow_up_type
                    break

        # Create new experience
        experience = Experience(
            query_id=query_id,
            timestamp=datetime.now(),
            user_id=user_id,
            transcript=transcript,
            parsed_intent=parsed_intent or {},
            widget_plan=widget_plan or {},
            fixtures=fixtures or {},
            processing_time_ms=processing_time_ms,
            available_data_summary=available_data_summary or {},
            user_history=user_history or [],
            intent_confidence=(parsed_intent or {}).get("confidence", 0.0),
        )

        # Add to buffer
        self.buffer.add(experience)

        # Track for follow-up detection
        self._last_query_by_user[user_id] = (transcript, parsed_intent or {})

        logger.debug(f"Recorded experience {query_id} for user {user_id}")

    def update_feedback(
        self,
        query_id: str,
        rating: str = None,
        interactions: list = None,
        correction: str = None,
    ) -> bool:
        """
        Update an experience with user feedback.

        Called when user provides explicit feedback via the UI.

        Args:
            query_id: The query ID from the original response
            rating: "up" or "down"
            interactions: List of widget interactions
            correction: User's correction text

        Returns:
            True if experience was found and updated
        """
        feedback = {}
        if rating:
            feedback["rating"] = rating
        if interactions:
            feedback["interactions"] = interactions
        if correction:
            feedback["correction"] = correction

        success = self.buffer.update_feedback(query_id, feedback)

        if success:
            logger.info(f"Updated feedback for {query_id}: rating={rating}")

            # Compute and store reward immediately
            exp = self.buffer.get_by_query_id(query_id)
            if exp:
                exp.computed_reward = self.reward_aggregator.compute_reward(exp)

        return success

    def get_status(self) -> dict:
        """Get current status of the RL system."""
        buffer_stats = self.buffer.get_stats()
        trainer_stats = self.trainer.get_stats() if self.trainer else {}

        return {
            "running": self.running,
            "buffer": buffer_stats,
            "trainer": trainer_stats,
            "config": {
                "train_widget_selector": self.config.get("train_widget_selector", True),
                "train_fixture_selector": self.config.get("train_fixture_selector", True),
                "train_interval": self.config.get("train_interval", 60),
                "min_batch_size": self.config.get("min_batch_size", 16),
            },
        }

    def get_recent_experiences(self, n: int = 10) -> list[dict]:
        """Get recent experiences (for debugging)."""
        experiences = self.buffer.get_recent(n)
        return [e.to_dict() for e in experiences]

    def _on_training_step(self, metrics: dict):
        """Callback after each training step."""
        logger.debug(f"Training step completed: {metrics}")

        # Check if we should checkpoint
        checkpoint_interval = self.config.get("checkpoint_interval", 100)
        if metrics.get("step", 0) % checkpoint_interval == 0:
            self.buffer.save()


# Global access functions

_rl_system: Optional[ContinuousRL] = None


def get_rl_system() -> ContinuousRL:
    """Get the global ContinuousRL instance."""
    return ContinuousRL.get_instance()


def init_rl_system(widget_selector=None, fixture_selector=None) -> ContinuousRL:
    """Initialize and start the RL system."""
    rl = get_rl_system()
    if not rl.running:
        rl.start(widget_selector, fixture_selector)
    return rl


def shutdown_rl_system():
    """Shutdown the RL system gracefully."""
    rl = ContinuousRL.get_instance()
    if rl.running:
        rl.stop()
