"""
Experience Buffer for Continuous RL

Thread-safe storage for (state, action, reward) experiences with optional Redis persistence.
"""

import collections
import json
import logging
import os
import random
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import CONTINUOUS_RL_CONFIG, TRAINING_DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """A single interaction experience for RL training."""

    # Identity
    query_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str = ""

    # State (inputs to the system)
    transcript: str = ""
    parsed_intent: dict = field(default_factory=dict)
    available_data_summary: dict = field(default_factory=dict)
    user_history: list = field(default_factory=list)

    # Action (what the system selected)
    widget_plan: dict = field(default_factory=dict)
    fixtures: dict = field(default_factory=dict)  # scenario -> fixture slug

    # Immediate signals (captured during request)
    intent_confidence: float = 0.0
    processing_time_ms: int = 0

    # Delayed signals (updated via feedback)
    user_rating: Optional[str] = None  # "up" / "down" / None
    follow_up_type: Optional[str] = None  # "satisfied" / "refinement" / "repeat" / "correction"
    widget_interactions: list = field(default_factory=list)
    correction_text: Optional[str] = None

    # Rich evaluation feedback (from Claude Sonnet 4.5)
    evaluation_confidence: Optional[float] = None  # 0.0-1.0 confidence in rating
    evaluation_reasoning: Optional[str] = None  # Why Claude rated it this way
    query_understanding: Optional[str] = None  # What Claude thinks user wants
    per_widget_feedback: list = field(default_factory=list)  # Detailed per-widget analysis
    missing_widgets: list = field(default_factory=list)  # Widget types that should be added
    suggested_improvements: list = field(default_factory=list)  # Actionable suggestions

    # Computed reward (set by RewardSignalAggregator)
    computed_reward: Optional[float] = None

    def has_feedback(self) -> bool:
        """Check if this experience has any feedback."""
        return (
            self.user_rating is not None
            or self.follow_up_type is not None
            or len(self.widget_interactions) > 0
        )

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Experience":
        """Create from dict."""
        d = d.copy()
        if isinstance(d.get("timestamp"), str):
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d)


class ExperienceBuffer:
    """
    Thread-safe experience buffer with optional Redis persistence.

    Stores experiences in memory with overflow to Redis.
    Supports async feedback updates by query_id.
    """

    def __init__(
        self,
        max_size: int = None,
        redis_url: str = None,
        persist_to_disk: bool = True,
    ):
        config = CONTINUOUS_RL_CONFIG
        self.max_size = max_size or config["buffer_size"]
        self.buffer: collections.deque[Experience] = collections.deque(maxlen=self.max_size)
        self.lock = threading.RLock()
        self.persist_to_disk = persist_to_disk

        # Query ID -> Experience index for fast lookups
        self._query_index: dict[str, Experience] = {}

        # Redis connection (optional)
        self.redis = None
        self.redis_key_prefix = config.get("redis_key_prefix", "cc_rl:")
        if redis_url:
            try:
                import redis as redis_lib
                self.redis = redis_lib.from_url(redis_url)
                logger.info(f"Connected to Redis: {redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")

        # Load persisted data
        self._load_from_disk()

    def _get_buffer_path(self) -> Path:
        """Get path for disk persistence."""
        TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return TRAINING_DATA_DIR / "experience_buffer.json"

    def _load_from_disk(self):
        """Load buffer from disk on startup."""
        buffer_path = self._get_buffer_path()
        if not buffer_path.exists():
            return

        try:
            with open(buffer_path) as f:
                data = json.load(f)

            for item in data.get("experiences", []):
                exp = Experience.from_dict(item)
                self.buffer.append(exp)
                self._query_index[exp.query_id] = exp

            logger.info(f"Loaded {len(self.buffer)} experiences from disk")

        except Exception as e:
            logger.warning(f"Failed to load buffer from disk: {e}")

    def _save_to_disk(self):
        """Save buffer to disk."""
        if not self.persist_to_disk:
            logger.debug("Persist to disk disabled, skipping save")
            return

        try:
            buffer_path = self._get_buffer_path()
            data = {
                "experiences": [exp.to_dict() for exp in self.buffer],
                "saved_at": datetime.now().isoformat(),
                "count": len(self.buffer),
            }

            with open(buffer_path, "w") as f:
                json.dump(data, f)

            logger.info(f"Saved {len(self.buffer)} experiences to {buffer_path}")

        except Exception as e:
            import traceback
            logger.error(f"Failed to save buffer to disk: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def add(self, experience: Experience):
        """
        Add a new experience (non-blocking).

        Called by orchestrator after each query.

        NOTE: With multiple gunicorn workers, we reload from disk before adding
        to prevent race conditions where workers overwrite each other's saves.
        """
        with self.lock:
            # Reload from disk to get latest state from all workers
            if self.persist_to_disk and not self.redis:
                try:
                    buffer_path = self._get_buffer_path()
                    if buffer_path.exists():
                        with open(buffer_path) as f:
                            data = json.load(f)
                        # Rebuild buffer from disk
                        self.buffer = [Experience.from_dict(exp_dict) for exp_dict in data.get("experiences", [])]
                        self._query_index = {exp.query_id: exp for exp in self.buffer}
                        logger.debug(f"Reloaded {len(self.buffer)} experiences from disk before add")
                except Exception as e:
                    logger.warning(f"Failed to reload buffer before add: {e}")

            # Check if we're at capacity
            if len(self.buffer) >= self.max_size:
                # Remove oldest from index
                oldest = self.buffer[0]
                self._query_index.pop(oldest.query_id, None)
                self.buffer.pop(0)

            self.buffer.append(experience)
            self._query_index[experience.query_id] = experience

            # Save to disk immediately after adding
            self._save_to_disk()

        # Async persist to Redis if available
        if self.redis:
            self._persist_to_redis_async(experience)

        logger.info(f"Added experience {experience.query_id}, buffer now has {len(self.buffer)} total")

    def update_feedback(self, query_id: str, feedback: dict) -> bool:
        """
        Update an experience with delayed feedback.

        Args:
            query_id: The query ID from the original response
            feedback: Dict with rating, interactions, follow_up_type, correction

        Returns:
            True if experience was found and updated
        """
        with self.lock:
            # CRITICAL FIX: Reload from disk to get experiences created by other workers
            if self.persist_to_disk and not self.redis:
                try:
                    buffer_path = self._get_buffer_path()
                    if buffer_path.exists():
                        with open(buffer_path) as f:
                            data = json.load(f)
                        # Rebuild buffer from disk
                        self.buffer = [Experience.from_dict(exp_dict) for exp_dict in data.get("experiences", [])]
                        self._query_index = {exp.query_id: exp for exp in self.buffer}
                        logger.debug(f"Reloaded {len(self.buffer)} experiences from disk before update_feedback")
                except Exception as e:
                    logger.warning(f"Failed to reload buffer before update_feedback: {e}")

            exp = self._query_index.get(query_id)
            if not exp:
                logger.warning(f"Experience not found for feedback: {query_id}")
                return False

            # Update basic fields
            if "rating" in feedback:
                exp.user_rating = feedback["rating"]
            if "follow_up_type" in feedback:
                exp.follow_up_type = feedback["follow_up_type"]
            if "interactions" in feedback:
                exp.widget_interactions = feedback["interactions"]
            if "correction" in feedback:
                exp.correction_text = feedback["correction"]

            # Update rich evaluation fields (from Claude Sonnet 4.5)
            if "evaluation_confidence" in feedback:
                exp.evaluation_confidence = feedback["evaluation_confidence"]
            if "evaluation_reasoning" in feedback:
                exp.evaluation_reasoning = feedback["evaluation_reasoning"]
            if "query_understanding" in feedback:
                exp.query_understanding = feedback["query_understanding"]
            if "per_widget_feedback" in feedback:
                exp.per_widget_feedback = feedback["per_widget_feedback"]
            if "missing_widgets" in feedback:
                exp.missing_widgets = feedback["missing_widgets"]
            if "suggested_improvements" in feedback:
                exp.suggested_improvements = feedback["suggested_improvements"]

            logger.debug(f"Updated feedback for {query_id}: rating={exp.user_rating}, confidence={exp.evaluation_confidence}")

            # CRITICAL FIX: Persist feedback to disk immediately
            self._save_to_disk()

            return True

    def get_by_query_id(self, query_id: str) -> Optional[Experience]:
        """Get experience by query ID."""
        with self.lock:
            return self._query_index.get(query_id)

    def sample_batch(self, batch_size: int, require_feedback: bool = True) -> list[Experience]:
        """
        Sample a random batch of experiences for training.

        Args:
            batch_size: Number of experiences to sample
            require_feedback: Only sample experiences with feedback

        Returns:
            List of sampled experiences
        """
        with self.lock:
            if require_feedback:
                candidates = [e for e in self.buffer if e.has_feedback()]
            else:
                candidates = list(self.buffer)

            if len(candidates) <= batch_size:
                return candidates

            return random.sample(candidates, batch_size)

    def get_recent(self, n: int = 100) -> list[Experience]:
        """Get most recent N experiences."""
        with self.lock:
            return list(self.buffer)[-n:]

    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)

    def feedback_count(self) -> int:
        """Get count of experiences with feedback."""
        with self.lock:
            return sum(1 for e in self.buffer if e.has_feedback())

    def clear(self):
        """Clear all experiences."""
        with self.lock:
            self.buffer.clear()
            self._query_index.clear()
        logger.info("Experience buffer cleared")

    def save(self):
        """Force save to disk."""
        with self.lock:
            self._save_to_disk()
        logger.info(f"Saved {len(self.buffer)} experiences to disk")

    def _persist_to_redis_async(self, experience: Experience):
        """Persist experience to Redis (fire and forget)."""
        if not self.redis:
            return

        try:
            key = f"{self.redis_key_prefix}exp:{experience.query_id}"
            self.redis.setex(
                key,
                86400 * 7,  # 7 day TTL
                json.dumps(experience.to_dict()),
            )
        except Exception as e:
            logger.warning(f"Failed to persist to Redis: {e}")

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        with self.lock:
            feedback_count = sum(1 for e in self.buffer if e.has_feedback())
            rating_counts = {"up": 0, "down": 0, "none": 0}
            for e in self.buffer:
                if e.user_rating == "up":
                    rating_counts["up"] += 1
                elif e.user_rating == "down":
                    rating_counts["down"] += 1
                else:
                    rating_counts["none"] += 1

            return {
                "total_experiences": len(self.buffer),
                "with_feedback": feedback_count,
                "without_feedback": len(self.buffer) - feedback_count,
                "ratings": rating_counts,
                "max_size": self.max_size,
                "redis_connected": self.redis is not None,
            }


# Global instance
_buffer_instance: Optional[ExperienceBuffer] = None


def get_experience_buffer() -> ExperienceBuffer:
    """Get or create the global experience buffer."""
    global _buffer_instance
    if _buffer_instance is None:
        _buffer_instance = ExperienceBuffer(
            redis_url=os.getenv("REDIS_URL"),
        )
    return _buffer_instance
