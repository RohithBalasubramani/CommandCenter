"""
Confidence Envelope — Phase 2 of RAG & Widget Selection Redesign.

Threads graduated confidence through the entire pipeline, enabling:
- Voice response caveats proportional to uncertainty
- Dashboard degradation when data is insufficient
- Frontend confidence indicators
- Decision not to render a dashboard when confidence is too low

The envelope is computed deterministically from stage outputs — no LLM calls.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Action thresholds
THRESHOLD_FULL = 0.75          # Full dashboard, no caveats
THRESHOLD_PARTIAL = 0.55       # Dashboard with voice caveats
THRESHOLD_REDUCED = 0.35       # Fewer widgets, explicit gaps
THRESHOLD_CLARIFY = 0.20       # Don't guess — ask the operator


@dataclass
class ConfidenceEnvelope:
    """
    Graduated confidence that flows through every pipeline stage.

    Each dimension is 0.0-1.0. The overall score uses weighted harmonic
    mean — dominated by the weakest link, because a dashboard is only
    as trustworthy as its weakest component.
    """
    # Per-stage confidence (0-1)
    intent_confidence: float = 0.5       # How sure are we about what the user wants?
    retrieval_completeness: float = 0.5  # Did we find everything we need?
    data_freshness: float = 1.0          # How current is the data?
    widget_fit: float = 0.5              # How well do widgets match information needs?
    data_fill_quality: float = 0.5       # How much widget data is real vs synthetic?

    # Metadata
    caveats: list[str] = None            # Specific caveats to include in voice response
    action: str = ""                     # Computed from overall score

    def __post_init__(self):
        if self.caveats is None:
            self.caveats = []
        self.action = self._compute_action()

    @property
    def overall(self) -> float:
        """Weighted harmonic mean — dominated by the weakest link."""
        weights = [0.15, 0.30, 0.20, 0.15, 0.20]
        values = [
            self.intent_confidence,
            self.retrieval_completeness,
            self.data_freshness,
            self.widget_fit,
            self.data_fill_quality,
        ]
        # Harmonic mean: n / sum(w_i / v_i)
        weighted_sum = 0.0
        for w, v in zip(weights, values):
            # Clamp to avoid division by zero
            v_safe = max(v, 0.01)
            weighted_sum += w / v_safe
        return min(1.0, sum(weights) / weighted_sum) if weighted_sum > 0 else 0.0

    def _compute_action(self) -> str:
        """Determine system action based on overall confidence."""
        score = self.overall
        if score >= THRESHOLD_FULL:
            return "full_dashboard"
        elif score >= THRESHOLD_PARTIAL:
            return "partial_with_caveats"
        elif score >= THRESHOLD_REDUCED:
            return "reduced_dashboard"
        elif score >= THRESHOLD_CLARIFY:
            return "reduced_with_warning"
        else:
            return "ask_clarification"

    def to_dict(self) -> dict:
        return {
            "intent_confidence": round(self.intent_confidence, 3),
            "retrieval_completeness": round(self.retrieval_completeness, 3),
            "data_freshness": round(self.data_freshness, 3),
            "widget_fit": round(self.widget_fit, 3),
            "data_fill_quality": round(self.data_fill_quality, 3),
            "overall": round(self.overall, 3),
            "action": self.action,
            "caveats": self.caveats,
        }


class ConfidenceComputer:
    """
    Computes a ConfidenceEnvelope from pipeline stage outputs.

    Call methods as each stage completes to build up the envelope.
    """

    def __init__(self):
        self._envelope = ConfidenceEnvelope()

    @property
    def envelope(self) -> ConfidenceEnvelope:
        """Get the current confidence envelope (recomputes action)."""
        self._envelope.action = self._envelope._compute_action()
        return self._envelope

    def set_intent_confidence(self, parsed_confidence: float):
        """Set from IntentParser output."""
        self._envelope.intent_confidence = max(0.0, min(1.0, parsed_confidence))

    def set_retrieval_completeness(self, completeness: float,
                                     gaps: list[str] = None):
        """Set from RetrievalAssessment."""
        self._envelope.retrieval_completeness = max(0.0, min(1.0, completeness))
        if gaps:
            for gap in gaps[:3]:
                self._envelope.caveats.append(gap)

    def set_data_freshness(self, freshness: float, stale_count: int = 0):
        """Set from RetrievalAssessment freshness score."""
        self._envelope.data_freshness = max(0.0, min(1.0, freshness))
        if stale_count > 0:
            self._envelope.caveats.append(
                f"{stale_count} widget(s) show data that may not be current"
            )

    def set_widget_fit(self, total_planned: int, total_rendered: int):
        """Set from the ratio of planned vs actually rendered widgets."""
        if total_planned > 0:
            ratio = total_rendered / total_planned
            self._envelope.widget_fit = max(0.0, min(1.0, ratio))
        else:
            self._envelope.widget_fit = 0.0

        if total_planned > 0 and total_rendered < total_planned * 0.5:
            dropped = total_planned - total_rendered
            self._envelope.caveats.append(
                f"{dropped} widget(s) were removed due to insufficient data"
            )

    def set_data_fill_quality(self, real_count: int, total_count: int):
        """Set from the ratio of real-data widgets vs total."""
        if total_count > 0:
            self._envelope.data_fill_quality = real_count / total_count
        else:
            self._envelope.data_fill_quality = 0.0

        if total_count > 0 and real_count < total_count * 0.5:
            synthetic = total_count - real_count
            self._envelope.caveats.append(
                f"{synthetic} widget(s) use estimated or demo data"
            )

    def build_voice_caveats(self) -> str:
        """
        Build a natural language caveat string for the voice response.

        Returns empty string if no caveats are needed.
        """
        envelope = self.envelope

        if envelope.action == "full_dashboard":
            return ""

        if envelope.action == "partial_with_caveats":
            if envelope.caveats:
                caveat_text = ". ".join(envelope.caveats[:2])
                return f" Note: {caveat_text}."
            return " Note: some data may not be fully current."

        if envelope.action == "reduced_dashboard":
            if envelope.caveats:
                caveat_text = ". ".join(envelope.caveats[:3])
                return (
                    f" I have partial data for this query. {caveat_text}. "
                    f"You may want to check the relevant systems directly."
                )
            return (
                " I have limited data for this query. "
                "The dashboard shows what I could find, but may not be complete."
            )

        if envelope.action in ("reduced_with_warning", "ask_clarification"):
            return (
                " I don't have enough data to build a comprehensive dashboard. "
                "Could you be more specific about what you'd like to see?"
            )

        return ""
