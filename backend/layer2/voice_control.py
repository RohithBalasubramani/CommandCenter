"""
Voice Control Surface — Upgrade 9

Interrupt detection, plan cancellation, and prosody signals.
Backend components for voice-as-control-surface.
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class InterruptType(Enum):
    STOP = "stop"
    REDIRECT = "redirect"
    CANCEL = "cancel"


class ProsodyLevel(Enum):
    LOW = "low"
    NEUTRAL = "neutral"
    HIGH = "high"


@dataclass
class ProsodySignals:
    """Future-compatible prosody extraction from voice."""
    urgency: float = 0.5       # 0.0-1.0
    hesitation: float = 0.5    # 0.0-1.0
    confidence: float = 0.5    # 0.0-1.0

    def urgency_level(self) -> ProsodyLevel:
        if self.urgency > 0.7:
            return ProsodyLevel.HIGH
        elif self.urgency < 0.3:
            return ProsodyLevel.LOW
        return ProsodyLevel.NEUTRAL

    def to_dict(self) -> dict:
        return {
            "urgency": self.urgency,
            "hesitation": self.hesitation,
            "confidence": self.confidence,
            "urgency_level": self.urgency_level().value,
        }


@dataclass
class InterruptEvent:
    """Represents a detected voice interrupt."""
    type: InterruptType
    transcript: str = ""
    timestamp: float = field(default_factory=time.time)
    plan_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "transcript": self.transcript,
            "timestamp": self.timestamp,
            "plan_id": self.plan_id,
        }


class InterruptDetector:
    """
    Detects voice interrupts from partial transcripts.

    Uses keyword patterns with boundary context to reduce false positives.
    """

    STOP_PATTERNS = [
        re.compile(r'\b(stop|halt|never\s?mind|cancel|forget\s+it)\b', re.IGNORECASE),
    ]

    REDIRECT_PATTERNS = [
        re.compile(r'\b(?:no|wait|actually|instead)\s*,', re.IGNORECASE),
        re.compile(r'\b(?:not\s+that|I\s+meant)\b', re.IGNORECASE),
    ]

    # Context patterns that negate interrupt detection
    # e.g., "stop the vibration from increasing" is NOT an interrupt
    FALSE_POSITIVE_PATTERNS = [
        re.compile(r'\bstop\s+the\s+\w+\s+from\b', re.IGNORECASE),
        re.compile(r'\bstop\s+(?:vibration|leak|alarm|motor|pump)\b', re.IGNORECASE),
        re.compile(r'\bhalt\s+(?:production|line|process)\b', re.IGNORECASE),
    ]

    def __init__(self):
        self._last_interrupt_time = 0.0
        self._interrupt_count_window: list[float] = []
        self.RAPID_FIRE_THRESHOLD = 3
        self.RAPID_FIRE_WINDOW_S = 10.0

    def check(self, partial_transcript: str) -> Optional[InterruptEvent]:
        """
        Check a partial transcript for interrupt keywords.

        Returns an InterruptEvent if detected, None otherwise.
        """
        if not partial_transcript or len(partial_transcript.strip()) < 2:
            return None

        text = partial_transcript.strip()

        # Check for false positives first
        for fp in self.FALSE_POSITIVE_PATTERNS:
            if fp.search(text):
                return None

        # Check for rapid-fire interrupts
        now = time.time()
        self._interrupt_count_window = [
            t for t in self._interrupt_count_window
            if now - t < self.RAPID_FIRE_WINDOW_S
        ]
        if len(self._interrupt_count_window) >= self.RAPID_FIRE_THRESHOLD:
            return InterruptEvent(
                type=InterruptType.STOP,
                transcript="[rapid_fire_detected]",
            )

        # Check stop patterns
        for pattern in self.STOP_PATTERNS:
            if pattern.search(text):
                self._interrupt_count_window.append(now)
                self._last_interrupt_time = now
                return InterruptEvent(
                    type=InterruptType.STOP,
                    transcript=text,
                )

        # Check redirect patterns
        for pattern in self.REDIRECT_PATTERNS:
            if pattern.search(text):
                self._interrupt_count_window.append(now)
                self._last_interrupt_time = now
                return InterruptEvent(
                    type=InterruptType.REDIRECT,
                    transcript=text,
                )

        return None

    def is_rapid_fire(self) -> bool:
        """Check if we're in a rapid-fire interrupt situation."""
        now = time.time()
        recent = [t for t in self._interrupt_count_window
                  if now - t < self.RAPID_FIRE_WINDOW_S]
        return len(recent) >= self.RAPID_FIRE_THRESHOLD

    def reset(self):
        """Reset interrupt state."""
        self._interrupt_count_window.clear()
        self._last_interrupt_time = 0.0


class PlanCancellationManager:
    """
    Manages plan cancellation state.

    Tracks active plans and allows cancellation by ID.
    """

    def __init__(self):
        self._active_plans: dict[str, dict] = {}

    def register_plan(self, plan_id: str, plan_data: dict = None):
        """Register a plan as active."""
        self._active_plans[plan_id] = {
            "started_at": time.time(),
            "cancelled": False,
            "data": plan_data or {},
        }

    def cancel(self, plan_id: str) -> dict:
        """
        Cancel a plan by ID.

        Returns status dict.
        """
        plan = self._active_plans.get(plan_id)
        if not plan:
            return {
                "cancelled": False,
                "reason": "plan_not_found",
            }

        if plan.get("completed"):
            return {
                "cancelled": False,
                "reason": "already_completed",
            }

        if plan.get("cancelled"):
            return {
                "cancelled": False,
                "reason": "already_cancelled",
            }

        plan["cancelled"] = True
        plan["cancelled_at"] = time.time()

        return {
            "cancelled": True,
            "plan_id": plan_id,
            "elapsed_ms": int((time.time() - plan["started_at"]) * 1000),
        }

    def mark_completed(self, plan_id: str):
        """Mark a plan as completed."""
        plan = self._active_plans.get(plan_id)
        if plan:
            plan["completed"] = True
            plan["completed_at"] = time.time()

    def is_cancelled(self, plan_id: str) -> bool:
        """Check if a plan has been cancelled."""
        plan = self._active_plans.get(plan_id)
        return bool(plan and plan.get("cancelled"))

    def cleanup_old(self, max_age_s: float = 300):
        """Remove plans older than max_age_s."""
        now = time.time()
        to_remove = [
            pid for pid, p in self._active_plans.items()
            if now - p["started_at"] > max_age_s
        ]
        for pid in to_remove:
            del self._active_plans[pid]

    def get_active_count(self) -> int:
        """Get count of non-completed, non-cancelled plans."""
        return sum(
            1 for p in self._active_plans.values()
            if not p.get("completed") and not p.get("cancelled")
        )
