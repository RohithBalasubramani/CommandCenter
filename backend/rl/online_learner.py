"""
Online Learning System for Command Center

Implements continuous learning from production feedback.
Accumulates ratings and triggers retraining when threshold is reached.
"""

import json
import logging
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from .config import ONLINE_LEARNING_CONFIG, CHECKPOINTS_DIR, TRAINING_DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSample:
    """A single feedback sample for training."""
    entry_id: str
    rating: str  # "up" or "down"
    tags: list[str]
    notes: str
    timestamp: datetime
    # Optional metadata
    question_id: Optional[str] = None
    scenario: Optional[str] = None
    fixture: Optional[str] = None
    size: Optional[str] = None
    query: Optional[str] = None


class OnlineLearner:
    """
    Accumulates feedback and triggers retraining when conditions are met.

    Thread-safe for use in Django request handling.
    """

    def __init__(
        self,
        min_samples: int = None,
        max_buffer_size: int = None,
        retrain_interval_hours: int = None,
        checkpoint_dir: str = None,
        auto_export: bool = None,
        auto_deploy: bool = None,
        on_retrain_complete: Optional[Callable] = None,
    ):
        """
        Initialize the online learner.

        Args:
            min_samples: Minimum feedback samples before retraining
            max_buffer_size: Maximum samples to hold in memory
            retrain_interval_hours: Maximum time between retraining
            checkpoint_dir: Directory for saving checkpoints
            auto_export: Automatically export to GGUF after training
            auto_deploy: Automatically swap Ollama model
            on_retrain_complete: Callback when retraining finishes
        """
        config = ONLINE_LEARNING_CONFIG

        self.min_samples = min_samples or config["min_samples_to_retrain"]
        self.max_buffer_size = max_buffer_size or config["max_buffer_size"]
        self.retrain_interval_hours = retrain_interval_hours or config["retrain_interval_hours"]
        self.checkpoint_dir = checkpoint_dir or str(CHECKPOINTS_DIR)
        self.auto_export = auto_export if auto_export is not None else config["auto_export"]
        self.auto_deploy = auto_deploy if auto_deploy is not None else config["auto_deploy"]
        self.on_retrain_complete = on_retrain_complete

        # Thread-safe buffer
        self.feedback_buffer = deque(maxlen=self.max_buffer_size)
        self.lock = threading.Lock()

        # Tracking
        self.last_train_time: Optional[datetime] = None
        self.train_count = 0
        self.is_training = False
        self._training_thread: Optional[threading.Thread] = None

        # Load persisted buffer on startup
        self._load_buffer()

    def _get_buffer_path(self) -> Path:
        """Get path to persisted buffer file."""
        return Path(TRAINING_DATA_DIR) / "feedback_buffer.json"

    def _load_buffer(self):
        """Load persisted buffer from disk."""
        buffer_path = self._get_buffer_path()
        if buffer_path.exists():
            try:
                with open(buffer_path) as f:
                    data = json.load(f)
                for item in data.get("samples", []):
                    sample = FeedbackSample(
                        entry_id=item["entry_id"],
                        rating=item["rating"],
                        tags=item.get("tags", []),
                        notes=item.get("notes", ""),
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        question_id=item.get("question_id"),
                        scenario=item.get("scenario"),
                        fixture=item.get("fixture"),
                        size=item.get("size"),
                        query=item.get("query"),
                    )
                    self.feedback_buffer.append(sample)

                if data.get("last_train_time"):
                    self.last_train_time = datetime.fromisoformat(data["last_train_time"])

                logger.info(f"Loaded {len(self.feedback_buffer)} samples from buffer")
            except Exception as e:
                logger.warning(f"Failed to load buffer: {e}")

    def _save_buffer(self):
        """Persist buffer to disk."""
        buffer_path = self._get_buffer_path()
        buffer_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "samples": [
                {
                    "entry_id": s.entry_id,
                    "rating": s.rating,
                    "tags": s.tags,
                    "notes": s.notes,
                    "timestamp": s.timestamp.isoformat(),
                    "question_id": s.question_id,
                    "scenario": s.scenario,
                    "fixture": s.fixture,
                    "size": s.size,
                    "query": s.query,
                }
                for s in self.feedback_buffer
            ],
            "last_train_time": self.last_train_time.isoformat() if self.last_train_time else None,
            "saved_at": datetime.now().isoformat(),
        }

        with open(buffer_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_feedback(self, feedback: dict) -> bool:
        """
        Add new feedback to the buffer.

        Args:
            feedback: Dict with entry_id, rating, tags, notes, etc.

        Returns:
            True if retraining should be triggered
        """
        sample = FeedbackSample(
            entry_id=feedback["entry_id"],
            rating=feedback["rating"],
            tags=feedback.get("tags", []),
            notes=feedback.get("notes", ""),
            timestamp=datetime.now(),
            question_id=feedback.get("question_id"),
            scenario=feedback.get("scenario"),
            fixture=feedback.get("fixture"),
            size=feedback.get("size"),
            query=feedback.get("query"),
        )

        with self.lock:
            self.feedback_buffer.append(sample)
            self._save_buffer()

        logger.debug(f"Added feedback for {sample.entry_id}, buffer size: {len(self.feedback_buffer)}")

        return self.should_retrain()

    def should_retrain(self) -> bool:
        """
        Check if retraining conditions are met.

        Triggers retraining if:
        1. Buffer has minimum samples, OR
        2. Time since last training exceeds interval

        Returns:
            True if retraining should be triggered
        """
        if self.is_training:
            return False

        buffer_size = len(self.feedback_buffer)
        if buffer_size < self.min_samples:
            return False

        # Check time since last training
        if self.last_train_time:
            elapsed = datetime.now() - self.last_train_time
            if elapsed < timedelta(hours=self.retrain_interval_hours):
                # Not enough time has passed, but check if we have many more samples
                if buffer_size < self.min_samples * 2:
                    return False

        return True

    def trigger_retrain(self, async_mode: bool = True) -> bool:
        """
        Trigger a retraining run.

        Args:
            async_mode: Run training in background thread

        Returns:
            True if training was started
        """
        if self.is_training:
            logger.warning("Training already in progress")
            return False

        if async_mode:
            self._training_thread = threading.Thread(
                target=self._run_training,
                daemon=True,
            )
            self._training_thread.start()
            return True
        else:
            return self._run_training()

    def _run_training(self) -> bool:
        """Execute the training pipeline."""
        self.is_training = True
        logger.info(f"Starting training with {len(self.feedback_buffer)} samples")

        try:
            # Snapshot and clear buffer
            with self.lock:
                samples = list(self.feedback_buffer)
                self.feedback_buffer.clear()
                self._save_buffer()

            # Build dataset from samples
            from .dataset_builder import (
                get_all_scenarios,
                get_fixture_descriptions,
                pairs_to_hf_dataset,
            )
            from .data_formatter import (
                build_widget_dpo_pairs,
                build_fixture_dpo_pairs,
            )

            # Convert samples to entries format
            entries = []
            for s in samples:
                entries.append({
                    "entry_id": s.entry_id,
                    "rating": s.rating,
                    "tags": s.tags,
                    "notes": s.notes,
                    "question_id": s.question_id,
                    "scenario": s.scenario,
                    "fixture": s.fixture,
                    "size": s.size,
                    "question": s.query,
                })

            all_scenarios = get_all_scenarios()
            fixture_descriptions = get_fixture_descriptions()

            # Build DPO pairs
            widget_pairs = build_widget_dpo_pairs(entries, all_scenarios)
            fixture_pairs = build_fixture_dpo_pairs(entries, fixture_descriptions)
            all_pairs = widget_pairs + fixture_pairs

            if not all_pairs:
                logger.warning("No valid training pairs generated")
                return False

            # Convert to HF dataset
            dataset = pairs_to_hf_dataset(all_pairs)

            # Run training
            from .trainer import CommandCenterDPOTrainer
            trainer = CommandCenterDPOTrainer()
            trainer.load_base_model()

            train_data = dataset["train"] if "train" in dataset else dataset
            eval_data = dataset.get("validation")

            result = trainer.train(
                train_dataset=train_data,
                eval_dataset=eval_data,
                output_dir=self.checkpoint_dir,
            )

            if not result.success:
                logger.error(f"Training failed: {result.error_message}")
                return False

            # Export to GGUF if enabled
            if self.auto_export and result.checkpoint_path:
                from .export import export_to_ollama
                export_result = export_to_ollama(
                    checkpoint_path=result.checkpoint_path,
                    register=self.auto_deploy,
                )
                if not export_result.success:
                    logger.warning(f"Export failed: {export_result.error_message}")

            # Update tracking
            self.last_train_time = datetime.now()
            self.train_count += 1
            self._save_buffer()

            # Callback
            if self.on_retrain_complete:
                self.on_retrain_complete(result)

            logger.info(f"Training complete. Total trains: {self.train_count}")
            return True

        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            return False

        finally:
            self.is_training = False

    def get_status(self) -> dict:
        """Get current status of the online learner."""
        return {
            "buffer_size": len(self.feedback_buffer),
            "min_samples": self.min_samples,
            "is_training": self.is_training,
            "last_train_time": self.last_train_time.isoformat() if self.last_train_time else None,
            "train_count": self.train_count,
            "should_retrain": self.should_retrain(),
            "auto_export": self.auto_export,
            "auto_deploy": self.auto_deploy,
        }

    def force_save(self):
        """Force save the buffer to disk."""
        with self.lock:
            self._save_buffer()

    def clear_buffer(self):
        """Clear the feedback buffer."""
        with self.lock:
            self.feedback_buffer.clear()
            self._save_buffer()


# Global instance - initialized in Django AppConfig
_online_learner: Optional[OnlineLearner] = None


def get_online_learner() -> Optional[OnlineLearner]:
    """Get the global online learner instance."""
    return _online_learner


def init_online_learner(**kwargs) -> OnlineLearner:
    """Initialize the global online learner."""
    global _online_learner
    _online_learner = OnlineLearner(**kwargs)
    return _online_learner


def shutdown_online_learner():
    """Shutdown the online learner and save state."""
    global _online_learner
    if _online_learner:
        _online_learner.force_save()
        _online_learner = None
