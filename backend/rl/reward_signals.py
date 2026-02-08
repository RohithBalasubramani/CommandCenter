"""
Reward Signal Aggregation for Continuous RL

Combines multiple feedback signals (explicit and implicit) into a single reward value.
"""

import logging
from typing import TYPE_CHECKING

from .config import CONTINUOUS_RL_CONFIG

if TYPE_CHECKING:
    from .experience_buffer import Experience

logger = logging.getLogger(__name__)


class RewardSignalAggregator:
    """
    Combines multiple feedback signals into a single reward for RL training.

    Reward components:
    1. Explicit rating (user thumbs up/down) - strongest signal
    2. Follow-up type (did user need to repeat/refine?) - implicit signal
    3. Widget engagement (which widgets did user interact with?) - implicit signal
    4. Response latency (faster is better) - system metric
    5. Intent confidence (higher = more reliable prediction) - system metric
    """

    def __init__(self, weights: dict = None):
        """
        Initialize with reward weights.

        Args:
            weights: Dict of signal_name -> weight. Defaults to config.
        """
        config_weights = CONTINUOUS_RL_CONFIG.get("reward_weights", {})
        self.weights = weights or config_weights

        # Ensure all weights exist
        self.weights.setdefault("explicit_rating", 1.0)
        self.weights.setdefault("follow_up_type", 0.5)
        self.weights.setdefault("widget_engagement", 0.3)
        self.weights.setdefault("response_latency", 0.1)
        self.weights.setdefault("intent_confidence", 0.1)

        # Rich evaluation weights (from Claude Sonnet 4.5)
        self.weights.setdefault("evaluation_confidence_boost", 0.2)
        self.weights.setdefault("per_widget_appropriateness", 0.4)
        self.weights.setdefault("missing_widget_penalty", -0.3)
        self.weights.setdefault("size_appropriateness", 0.2)

        # Reward bounds
        self.max_reward = CONTINUOUS_RL_CONFIG.get("max_reward", 2.0)
        self.min_reward = CONTINUOUS_RL_CONFIG.get("min_reward", -2.0)

    def compute_reward(self, experience: "Experience") -> float:
        """
        Compute total reward from all available signals.

        Args:
            experience: The Experience to compute reward for

        Returns:
            Float reward value (clipped to [min_reward, max_reward])
        """
        reward = 0.0

        # 1. Explicit rating (strongest signal)
        reward += self._explicit_rating_reward(experience)

        # 2. Follow-up type
        reward += self._follow_up_reward(experience)

        # 3. Widget engagement
        reward += self._engagement_reward(experience)

        # 4. Response latency
        reward += self._latency_reward(experience)

        # 5. Intent confidence
        reward += self._confidence_reward(experience)

        # 6. Rich evaluation signals (from Claude Sonnet 4.5)
        reward += self._evaluation_confidence_boost(experience)
        reward += self._per_widget_appropriateness_reward(experience)
        reward += self._missing_widget_penalty_reward(experience)
        reward += self._size_appropriateness_reward(experience)

        # Clip to bounds
        reward = max(self.min_reward, min(self.max_reward, reward))

        return reward

    def _explicit_rating_reward(self, experience: "Experience") -> float:
        """Compute reward from explicit user rating."""
        if not experience.user_rating:
            return 0.0

        weight = self.weights["explicit_rating"]

        if experience.user_rating == "up":
            return weight * 1.0
        elif experience.user_rating == "down":
            return weight * -1.0
        else:
            return 0.0

    def _follow_up_reward(self, experience: "Experience") -> float:
        """
        Compute reward from follow-up query analysis.

        - satisfied: User moved to new topic (success)
        - new_topic: User switched context (neutral)
        - refinement: User needed to clarify (slight negative)
        - repeat: User had to repeat (failure)
        - correction: User corrected the system (failure)
        """
        if not experience.follow_up_type:
            return 0.0

        weight = self.weights["follow_up_type"]

        follow_up_rewards = {
            "satisfied": 1.0,      # Best outcome
            "new_topic": 0.3,      # Neutral to slightly positive
            "refinement": -0.3,    # User needed to narrow down
            "repeat": -1.0,        # User had to repeat (bad)
            "correction": -0.8,    # User had to correct (bad)
        }

        return weight * follow_up_rewards.get(experience.follow_up_type, 0.0)

    def _engagement_reward(self, experience: "Experience") -> float:
        """
        Compute reward from widget engagement.

        More engagement with selected widgets = positive signal.
        """
        if not experience.widget_interactions:
            return 0.0

        weight = self.weights["widget_engagement"]

        # Count widgets selected
        num_widgets = len(experience.widget_plan.get("widgets", []))
        if num_widgets == 0:
            return 0.0

        # Count unique widgets engaged with
        engaged_widgets = set()
        for interaction in experience.widget_interactions:
            widget_idx = interaction.get("widget_index")
            if widget_idx is not None:
                engaged_widgets.add(widget_idx)

        # Engagement ratio (0 to 1)
        engagement_ratio = len(engaged_widgets) / num_widgets

        # Bonus for deep engagement (expand, scroll, click)
        deep_actions = {"expand", "scroll", "click", "drill_down"}
        has_deep_engagement = any(
            i.get("action") in deep_actions for i in experience.widget_interactions
        )

        if has_deep_engagement:
            engagement_ratio = min(1.0, engagement_ratio + 0.2)

        return weight * engagement_ratio

    def _latency_reward(self, experience: "Experience") -> float:
        """
        Compute reward from response latency.

        Faster responses (under 3s) get a small bonus.
        Very slow responses (over 10s) get a penalty.
        """
        if experience.processing_time_ms <= 0:
            return 0.0

        weight = self.weights["response_latency"]

        latency_ms = experience.processing_time_ms

        # Target: < 3000ms is great, 3000-6000ms is okay, > 6000ms is bad
        if latency_ms < 3000:
            # Bonus for fast responses (0 to 1 scale)
            bonus = 1.0 - (latency_ms / 3000)
            return weight * bonus
        elif latency_ms < 6000:
            # Neutral zone
            return 0.0
        else:
            # Penalty for slow responses
            penalty = min(1.0, (latency_ms - 6000) / 4000)
            return weight * -penalty

    def _confidence_reward(self, experience: "Experience") -> float:
        """
        Compute reward from intent confidence.

        Higher confidence on correct predictions = slight bonus.
        This encourages the model to be more certain when it's right.
        """
        weight = self.weights["intent_confidence"]

        confidence = experience.intent_confidence
        if confidence <= 0:
            return 0.0

        # Only give bonus if we have positive feedback
        if experience.user_rating == "up":
            # High confidence + positive feedback = good
            return weight * confidence
        elif experience.user_rating == "down":
            # High confidence + negative feedback = bad (overconfident)
            return weight * -confidence
        else:
            # No feedback yet, slight positive for high confidence
            return weight * confidence * 0.3

    def _evaluation_confidence_boost(self, experience: "Experience") -> float:
        """
        Boost reward based on Claude's confidence in the evaluation.

        High-confidence evaluations are more reliable, so we amplify them.
        Low-confidence evaluations are uncertain, so we dampen them.
        """
        if experience.evaluation_confidence is None:
            return 0.0

        weight = self.weights["evaluation_confidence_boost"]
        confidence = experience.evaluation_confidence

        # Get base rating reward
        if not experience.user_rating:
            return 0.0

        base_reward = 1.0 if experience.user_rating == "up" else -1.0

        # Amplify if confident (>0.8), dampen if uncertain (<0.6)
        if confidence > 0.8:
            boost = (confidence - 0.8) * 2.0  # 0.0 to 0.4 boost
            return weight * base_reward * boost
        elif confidence < 0.6:
            dampen = (0.6 - confidence) * 1.5  # 0.0 to 0.9 reduction
            return weight * base_reward * -dampen
        else:
            return 0.0

    def _per_widget_appropriateness_reward(self, experience: "Experience") -> float:
        """
        Compute reward from per-widget appropriateness scores.

        Uses Claude's detailed analysis of each widget's suitability.
        Averages scores across all widgets, weighted by position.
        """
        if not experience.per_widget_feedback:
            return 0.0

        weight = self.weights["per_widget_appropriateness"]

        total_score = 0.0
        total_weight = 0.0

        for widget_feedback in experience.per_widget_feedback:
            appropriateness = widget_feedback.get("appropriateness_score", 0.5)
            widget_index = widget_feedback.get("widget_index", 0)

            # Weight earlier widgets more heavily (hero > expanded > normal)
            # Add 1 to prevent division by zero for index 0
            position_weight = 1.0 / ((widget_index + 1) ** 0.5)  # sqrt decay

            total_score += appropriateness * position_weight
            total_weight += position_weight

        if total_weight == 0:
            return 0.0

        # Average score (0.0 to 1.0) -> reward (-1.0 to 1.0)
        avg_score = total_score / total_weight
        normalized_reward = (avg_score - 0.5) * 2.0  # Map [0,1] to [-1,1]

        return weight * normalized_reward

    def _missing_widget_penalty_reward(self, experience: "Experience") -> float:
        """
        Penalty for missing widgets that Claude identified as needed.

        Each missing widget type indicates a gap in the response.
        """
        if not experience.missing_widgets:
            return 0.0

        weight = self.weights["missing_widget_penalty"]

        # Count distinct missing widget types
        num_missing = len(experience.missing_widgets)

        # Penalty scales with number of missing widgets (capped at 3)
        penalty = min(num_missing / 3.0, 1.0)

        return weight * penalty

    def _size_appropriateness_reward(self, experience: "Experience") -> float:
        """
        Reward/penalty based on widget size appropriateness.

        Claude evaluates if each widget's size (hero/expanded/normal/compact)
        is appropriate for its content and importance.
        """
        if not experience.per_widget_feedback:
            return 0.0

        weight = self.weights["size_appropriateness"]

        total_widgets = len(experience.per_widget_feedback)
        size_appropriate_count = 0

        for widget_feedback in experience.per_widget_feedback:
            if widget_feedback.get("size_appropriate", False):
                size_appropriate_count += 1

        # Ratio of correctly-sized widgets (0.0 to 1.0)
        size_ratio = size_appropriate_count / total_widgets if total_widgets > 0 else 0.0

        # Map to reward (-1.0 to 1.0)
        # 100% appropriate = +1.0, 0% appropriate = -1.0, 50% = 0.0
        normalized_reward = (size_ratio - 0.5) * 2.0

        return weight * normalized_reward

    def compute_batch_rewards(self, experiences: list["Experience"]) -> list[float]:
        """Compute rewards for a batch of experiences."""
        return [self.compute_reward(e) for e in experiences]

    def get_reward_breakdown(self, experience: "Experience") -> dict:
        """Get detailed breakdown of reward components (for debugging)."""
        return {
            "explicit_rating": self._explicit_rating_reward(experience),
            "follow_up_type": self._follow_up_reward(experience),
            "widget_engagement": self._engagement_reward(experience),
            "response_latency": self._latency_reward(experience),
            "intent_confidence": self._confidence_reward(experience),
            "total": self.compute_reward(experience),
        }


class ImplicitSignalDetector:
    """
    Detects implicit feedback signals from user behavior.

    Analyzes follow-up queries to determine if the previous response was satisfactory.
    """

    # Keywords that indicate correction
    CORRECTION_KEYWORDS = {
        "no", "not", "wrong", "meant", "actually", "instead", "i said",
        "that's not", "thats not", "incorrect", "different",
    }

    # Keywords that indicate repetition
    REPEAT_KEYWORDS = {
        "again", "repeat", "what", "huh", "sorry", "didn't get", "didnt get",
    }

    def classify_follow_up(
        self,
        previous_intent: dict,
        current_intent: dict,
        previous_transcript: str,
        current_transcript: str,
    ) -> str:
        """
        Classify a follow-up query to determine feedback type.

        Args:
            previous_intent: ParsedIntent dict from previous query
            current_intent: ParsedIntent dict from current query
            previous_transcript: Raw text of previous query
            current_transcript: Raw text of current query

        Returns:
            One of: "satisfied", "new_topic", "refinement", "repeat", "correction"
        """
        current_lower = current_transcript.lower()

        # Check for explicit correction
        if any(kw in current_lower for kw in self.CORRECTION_KEYWORDS):
            return "correction"

        # Check for repetition
        if any(kw in current_lower for kw in self.REPEAT_KEYWORDS):
            return "repeat"

        # Check if queries are very similar (repeat)
        if self._is_similar(previous_transcript, current_transcript):
            return "repeat"

        # Check entity overlap
        prev_entities = set(str(v) for v in previous_intent.get("entities", {}).values())
        curr_entities = set(str(v) for v in current_intent.get("entities", {}).values())

        entity_overlap = 0
        if prev_entities:
            entity_overlap = len(prev_entities & curr_entities) / len(prev_entities)

        # Check domain overlap
        prev_domains = set(previous_intent.get("domains", []))
        curr_domains = set(current_intent.get("domains", []))
        domain_overlap = len(prev_domains & curr_domains) > 0

        # High entity overlap + same domain = refinement
        if entity_overlap > 0.5 and domain_overlap:
            return "refinement"

        # Low overlap = new topic (satisfied with previous)
        if entity_overlap < 0.3 and not domain_overlap:
            return "satisfied"

        # Medium overlap = moved to new topic
        return "new_topic"

    def _is_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are very similar (possible repeat)."""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold

    def extract_correction(self, transcript: str) -> str | None:
        """
        Extract the correction from a correction query.

        E.g., "No, I meant pump-002" -> "pump-002"
        """
        lower = transcript.lower()

        # Look for "I meant X" pattern
        for pattern in ["i meant", "meant", "i said", "actually"]:
            if pattern in lower:
                idx = lower.find(pattern)
                correction = transcript[idx + len(pattern):].strip()
                # Remove leading punctuation
                correction = correction.lstrip(",.:;")
                if correction:
                    return correction.strip()

        return None
