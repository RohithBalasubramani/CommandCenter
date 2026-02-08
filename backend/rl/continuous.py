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
        voice_response: str = "",
        prompt_version: str = "",
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
            voice_response: The generated voice response text
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

        # Debug: Log what we're receiving
        logger.info(f"[record_experience] Creating experience with parsed_intent keys: {list((parsed_intent or {}).keys())[:10]}")
        logger.info(f"[record_experience] parsed_intent type: {type(parsed_intent)}, len: {len(parsed_intent or {})}")

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
            voice_response=voice_response or "",
            prompt_version=prompt_version or "",
        )

        # Debug: Verify intent in created experience
        logger.info(f"[record_experience] Experience created, intent in object: {len(experience.parsed_intent)} fields")

        # Validate widget_plan structure before saving
        if not experience.widget_plan or not experience.widget_plan.get("widgets"):
            logger.warning(
                f"Experience {query_id} has no valid widget_plan "
                f"(plan={bool(experience.widget_plan)}, "
                f"widgets={bool(experience.widget_plan.get('widgets') if experience.widget_plan else False)}). "
                f"This will limit its usefulness for training."
            )
            # Ensure consistent structure: always have "widgets" key even if empty
            if experience.widget_plan is None or not isinstance(experience.widget_plan, dict):
                experience.widget_plan = {"widgets": [], "heading": "No Response"}
            elif "widgets" not in experience.widget_plan:
                experience.widget_plan["widgets"] = []

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
        # Rich evaluation fields from Claude Sonnet 4.5
        evaluation_confidence: float = None,
        evaluation_reasoning: str = None,
        query_understanding: str = None,
        per_widget_feedback: list = None,
        missing_widgets: list = None,
        suggested_improvements: list = None,
        # Claude voice evaluation fields
        voice_evaluation_confidence: float = None,
        voice_evaluation_reasoning: str = None,
        voice_dimension_scores_claude: dict = None,
    ) -> bool:
        """
        Update an experience with user feedback.

        Called when user provides explicit feedback via the UI or auto-evaluator.

        Args:
            query_id: The query ID from the original response
            rating: "up" or "down"
            interactions: List of widget interactions
            correction: User's correction text
            evaluation_confidence: Confidence score (0.0-1.0) from evaluator
            evaluation_reasoning: Detailed reasoning for the rating
            query_understanding: What the user is trying to accomplish
            per_widget_feedback: List of per-widget evaluation dicts
            missing_widgets: List of widget types that should have been included
            suggested_improvements: List of actionable improvement suggestions

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

        # Add rich evaluation fields
        if evaluation_confidence is not None:
            feedback["evaluation_confidence"] = evaluation_confidence
        if evaluation_reasoning:
            feedback["evaluation_reasoning"] = evaluation_reasoning
        if query_understanding:
            feedback["query_understanding"] = query_understanding
        if per_widget_feedback is not None:  # Fixed: save even empty list for consistency
            feedback["per_widget_feedback"] = per_widget_feedback
        if missing_widgets is not None:
            feedback["missing_widgets"] = missing_widgets
        if suggested_improvements is not None:
            feedback["suggested_improvements"] = suggested_improvements

        # Add Claude voice evaluation fields
        if voice_evaluation_confidence is not None:
            feedback["voice_evaluation_confidence"] = voice_evaluation_confidence
        if voice_evaluation_reasoning:
            feedback["voice_evaluation_reasoning"] = voice_evaluation_reasoning
        if voice_dimension_scores_claude:
            feedback["voice_dimension_scores_claude"] = voice_dimension_scores_claude

        success = self.buffer.update_feedback(query_id, feedback)

        if success:
            logger.info(f"Updated feedback for {query_id}: rating={rating}")

            # Compute and store reward immediately
            exp = self.buffer.get_by_query_id(query_id)
            if exp:
                exp.computed_reward = self.reward_aggregator.compute_reward(exp)

                # Derive voice_response_rating
                # Prefer Claude's voice-specific judgment over blunt user_rating proxy
                if exp.voice_response:
                    if exp.voice_evaluation_confidence is not None:
                        # Claude evaluated voice text directly — use its judgment
                        exp.voice_response_rating = "good" if exp.voice_evaluation_confidence >= 0.5 else "bad"
                    elif rating:
                        # Fallback: derive from overall user rating (blunt proxy)
                        exp.voice_response_rating = "good" if rating == "up" else "bad"

            # Update prompt evolver reward signal (Tier 1)
            # Uses Claude's evaluation_confidence as the reward
            if exp and exp.prompt_version and exp.evaluation_confidence is not None:
                try:
                    from rl.prompt_evolver import get_prompt_evolver
                    evolver = get_prompt_evolver()
                    # Convert 0.0-1.0 confidence to -1.0 to 1.0 reward range
                    reward = (exp.evaluation_confidence - 0.5) * 2.0
                    evolver.update_reward(exp.prompt_version, reward, thumbs=rating)
                    logger.debug(f"Updated prompt evolver: variant={exp.prompt_version}, confidence={exp.evaluation_confidence:.2f}, reward={reward:.2f}")
                except Exception as e:
                    logger.debug(f"Prompt evolver reward update failed: {e}")

            # Tier 3: Capture trace for SFT training (async, non-blocking)
            try:
                from rl.tier3_integration import should_capture_trace, capture_trace_async
                if exp and should_capture_trace(exp):
                    # Use transcript (the actual query text)
                    query_text = exp.transcript if hasattr(exp, 'transcript') and exp.transcript else None
                    if query_text:
                        capture_trace_async(query_text, query_id=query_id)
            except Exception as e:
                logger.debug(f"Tier 3 trace capture failed: {e}")

            # Also save rating to database for DPO training
            if rating:
                try:
                    from feedback.models import WidgetRating
                    from django.utils import timezone

                    WidgetRating.objects.update_or_create(
                        entry_id=query_id,
                        defaults={
                            "rating": rating,
                            "rated_at": timezone.now(),
                            "device_id": "rl-system",
                            "notes": f"Rating: {rating}" + (f", Correction: {correction}" if correction else ""),
                        }
                    )
                    logger.debug(f"Saved rating to database for DPO training")
                except Exception as e:
                    logger.warning(f"Failed to save rating to database: {e}")

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
