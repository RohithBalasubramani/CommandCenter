"""
Text Quality Scorer for Tier 2 DPO (Voice Quality)

Rule-based scorer that evaluates voice response quality on 5 dimensions.
Runs on CPU in <1ms per response. No ML model needed.

Dimensions:
1. Groundedness   — references specific data (equipment IDs, metrics, units)
2. Conciseness    — 2-3 sentences (TTS spec)
3. Directness     — first sentence answers the question (no filler)
4. Specificity    — contains numeric values and units
5. TTS-Friendliness — natural spoken language (no markdown, code, URLs)
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Filler phrases that indicate non-direct responses
FILLER_PATTERNS = [
    r"^i'?d be happy to",
    r"^let me",
    r"^sure[,!.]",
    r"^of course",
    r"^certainly",
    r"^absolutely",
    r"^great question",
    r"^that'?s a great",
    r"^based on (?:the|my|our)",
    r"^according to",
    r"^i can help",
    r"^here'?s what i",
    r"^well[,.]",
]

# Markdown/formatting patterns that break TTS
TTS_PENALTY_PATTERNS = [
    r'\*\*[^*]+\*\*',         # **bold**
    r'\*[^*]+\*',             # *italic*
    r'^#{1,6}\s',             # # headings
    r'^\s*[-*]\s',            # bullet lists
    r'^\s*\d+\.\s',          # numbered lists (at line start)
    r'```',                    # code fences
    r'`[^`]+`',               # inline code
    r'https?://\S+',          # URLs
    r'\([^)]{50,}\)',          # long parentheticals (>50 chars)
    r'\|.*\|.*\|',            # table rows
]

# Patterns indicating specific data references
SPECIFICITY_PATTERNS = [
    r'\d+\.?\d*\s*(?:PSI|psi|bar|kPa|MPa)',          # pressure
    r'\d+\.?\d*\s*(?:°[CF]|degrees?|celsius|fahrenheit)',  # temperature
    r'\d+\.?\d*\s*(?:kW|MW|GW|watts?|hp)',            # power
    r'\d+\.?\d*\s*(?:kWh|MWh|GWh)',                   # energy
    r'\d+\.?\d*\s*(?:RPM|rpm|Hz|hz)',                  # frequency
    r'\d+\.?\d*\s*(?:mm/s|m/s|in/s)',                  # vibration
    r'\d+\.?\d*\s*(?:amps?|volts?|kV|mA)',             # electrical
    r'\d+\.?\d*\s*(?:%|percent)',                       # percentage
    r'\d+\.?\d*\s*(?:GPM|gpm|L/min|m³/h)',             # flow
    r'\d+\.?\d*\s*(?:hours?|hrs?|minutes?|mins?|seconds?)',  # time
]

# Equipment ID patterns
EQUIPMENT_PATTERNS = [
    r'(?:pump|chiller|cooling tower|transformer|compressor|UPS|AHU|CT|TRF|generator|boiler|motor)[\s\-_]*\d+',
    r'[A-Z]{2,4}[\s\-_]*\d{1,4}',  # CT-01, AHU-03, etc.
]


class TextQualityScorer:
    """Scores voice response quality on 5 dimensions."""

    def __init__(self, weights: dict = None):
        self.weights = weights or {
            "groundedness": 0.25,
            "conciseness": 0.20,
            "directness": 0.20,
            "specificity": 0.20,
            "tts_friendliness": 0.15,
        }

    def score(self, voice_response: str, transcript: str = "") -> dict:
        """
        Score a voice response on all 5 dimensions.

        Args:
            voice_response: The generated voice response text
            transcript: The original user query (for grounding context)

        Returns:
            Dict with per-dimension scores (0.0-1.0) and weighted total
        """
        if not voice_response or not voice_response.strip():
            return {
                "groundedness": 0.0,
                "conciseness": 0.0,
                "directness": 0.0,
                "specificity": 0.0,
                "tts_friendliness": 0.0,
                "total": 0.0,
            }

        scores = {
            "groundedness": self._score_groundedness(voice_response, transcript),
            "conciseness": self._score_conciseness(voice_response),
            "directness": self._score_directness(voice_response),
            "specificity": self._score_specificity(voice_response),
            "tts_friendliness": self._score_tts_friendliness(voice_response),
        }

        # Weighted total
        total = sum(scores[k] * self.weights.get(k, 0.2) for k in scores)
        scores["total"] = round(total, 4)

        return scores

    def _score_groundedness(self, response: str, transcript: str) -> float:
        """Does the response reference specific data from the query context?"""
        score = 0.0
        resp_lower = response.lower()
        trans_lower = transcript.lower()

        # Check for equipment ID references
        equip_matches = 0
        for pattern in EQUIPMENT_PATTERNS:
            matches = re.findall(pattern, response, re.IGNORECASE)
            equip_matches += len(matches)

        if equip_matches > 0:
            score += min(0.4, equip_matches * 0.2)

        # Check for metric values with units
        metric_matches = 0
        for pattern in SPECIFICITY_PATTERNS:
            matches = re.findall(pattern, response, re.IGNORECASE)
            metric_matches += len(matches)

        if metric_matches > 0:
            score += min(0.3, metric_matches * 0.15)

        # Check if response references entities from the transcript
        # Extract key nouns from transcript
        transcript_words = set(re.findall(r'\b[a-z]{3,}\b', trans_lower))
        response_words = set(re.findall(r'\b[a-z]{3,}\b', resp_lower))
        equipment_words = {"pump", "chiller", "cooling", "tower", "transformer", "compressor",
                          "ups", "ahu", "generator", "boiler", "motor", "condenser", "valve"}
        shared_equip = transcript_words & response_words & equipment_words
        if shared_equip:
            score += 0.3

        return min(1.0, score)

    def _score_conciseness(self, response: str) -> float:
        """Is it 2-3 sentences as the TTS spec requires?"""
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s*', response.strip())
        sentences = [s for s in sentences if s.strip()]
        n = len(sentences)

        if n == 2 or n == 3:
            return 1.0
        elif n == 1 or n == 4:
            return 0.7
        elif n == 5:
            return 0.4
        elif n == 0:
            return 0.1
        else:  # 6+
            return 0.2

    def _score_directness(self, response: str) -> float:
        """Does the first sentence directly answer the question?"""
        first_sentence = response.strip().split('.')[0].lower().strip()

        # Check for filler phrases
        for pattern in FILLER_PATTERNS:
            if re.match(pattern, first_sentence, re.IGNORECASE):
                return 0.3  # Heavy penalty for filler

        # Check if first sentence contains data (numbers, equipment)
        has_number = bool(re.search(r'\d', first_sentence))
        has_equipment = bool(re.search(
            r'(?:pump|chiller|cooling|transformer|compressor|ups|ahu|generator|boiler|motor)',
            first_sentence, re.IGNORECASE
        ))

        if has_number and has_equipment:
            return 1.0  # Perfect: leads with data about specific equipment
        elif has_number or has_equipment:
            return 0.8  # Good: has some specificity
        else:
            return 0.6  # Acceptable: no filler, but not data-rich

    def _score_specificity(self, response: str) -> float:
        """Contains numeric values and units rather than vague claims?"""
        # Count numeric values with units
        specific_matches = 0
        for pattern in SPECIFICITY_PATTERNS:
            matches = re.findall(pattern, response, re.IGNORECASE)
            specific_matches += len(matches)

        # Count bare numbers (less specific but still concrete)
        bare_numbers = len(re.findall(r'\b\d+\.?\d*\b', response))

        if specific_matches >= 3:
            return 1.0
        elif specific_matches >= 2:
            return 0.85
        elif specific_matches >= 1:
            return 0.7
        elif bare_numbers >= 2:
            return 0.5
        elif bare_numbers >= 1:
            return 0.35
        else:
            return 0.15  # No specific data at all

    def _score_tts_friendliness(self, response: str) -> float:
        """Is it natural spoken language?"""
        score = 1.0
        lines = response.split('\n')

        for line in lines:
            for pattern in TTS_PENALTY_PATTERNS:
                if re.search(pattern, line, re.MULTILINE):
                    score -= 0.15

        # Bonus for short, simple sentences
        words = response.split()
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        if avg_word_len < 5.5:  # Short words = spoken-friendly
            score += 0.1

        # Penalty for very long response (TTS should be brief)
        if len(response) > 500:
            score -= 0.2
        elif len(response) > 300:
            score -= 0.1

        return max(0.0, min(1.0, score))
