"""
Causal Knowledge Engine — Upgrade 8

Replaces simple RAG with symptom-based failure mode matching.
Uses a structured failure mode database, then falls back to
document embeddings, then LLM general knowledge.
"""

import os
import json
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FailureMode:
    id: str
    equipment_type: str
    symptom_signature: dict  # {"vibration": "high", "temperature": "normal"}
    root_cause: str
    description: str
    confidence_base: float = 0.7
    resolution_steps: list[str] = field(default_factory=list)
    prevention_steps: list[str] = field(default_factory=list)
    mtbf_hours: Optional[float] = None
    severity: str = "medium"
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "equipment_type": self.equipment_type,
            "symptom_signature": self.symptom_signature,
            "root_cause": self.root_cause,
            "description": self.description,
            "confidence_base": self.confidence_base,
            "resolution_steps": self.resolution_steps,
            "prevention_steps": self.prevention_steps,
            "severity": self.severity,
        }


@dataclass
class SymptomMatch:
    failure_mode: FailureMode
    match_score: float = 0.0
    matched_symptoms: list[str] = field(default_factory=list)
    missing_symptoms: list[str] = field(default_factory=list)
    contradicting_symptoms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "failure_mode": self.failure_mode.to_dict(),
            "match_score": self.match_score,
            "matched_symptoms": self.matched_symptoms,
            "missing_symptoms": self.missing_symptoms,
            "contradicting_symptoms": self.contradicting_symptoms,
        }


@dataclass
class KnowledgeGap:
    """A piece of missing knowledge the system detected."""
    id: str = ""
    equipment_type: str = ""
    query: str = ""
    expected_knowledge_type: str = ""
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.sha256(
                f"{self.equipment_type}:{self.query}".encode()
            ).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "equipment_type": self.equipment_type,
            "query": self.query,
            "expected_knowledge_type": self.expected_knowledge_type,
            "resolved": self.resolved,
        }


# Built-in failure modes (industrial domain knowledge)
BUILTIN_FAILURE_MODES = [
    FailureMode(
        id="fm_pump_bearing_wear",
        equipment_type="centrifugal_pump",
        symptom_signature={"vibration": "high", "temperature": "normal", "noise": "increased"},
        root_cause="bearing_wear",
        description="Bearing wear in centrifugal pump — vibration increases before thermal effects",
        confidence_base=0.72,
        resolution_steps=[
            "Check vibration frequency spectrum for BPFO/BPFI frequencies",
            "Inspect bearing housing for discoloration",
            "Measure bearing clearance",
            "Replace bearings if defect frequencies confirmed",
        ],
        prevention_steps=[
            "Implement condition-based monitoring",
            "Follow lubrication schedule",
            "Track vibration trending monthly",
        ],
        mtbf_hours=25000,
        severity="high",
        sources=["domain_expert", "ISO_10816"],
    ),
    FailureMode(
        id="fm_pump_misalignment",
        equipment_type="centrifugal_pump",
        symptom_signature={"vibration": "high", "temperature": "slightly_elevated"},
        root_cause="shaft_misalignment",
        description="Shaft misalignment — causes 2x vibration frequency and axial vibration",
        confidence_base=0.65,
        resolution_steps=[
            "Perform laser alignment measurement",
            "Check coupling condition",
            "Verify soft foot",
            "Realign if offset > 0.05mm",
        ],
        prevention_steps=["Laser alignment after every motor replacement"],
        severity="medium",
        sources=["domain_expert"],
    ),
    FailureMode(
        id="fm_pump_cavitation",
        equipment_type="centrifugal_pump",
        symptom_signature={"vibration": "high", "power": "high", "flow": "fluctuating"},
        root_cause="cavitation",
        description="Cavitation — vapor bubbles collapsing inside pump casing",
        confidence_base=0.70,
        resolution_steps=[
            "Check NPSH available vs required",
            "Verify suction strainer not blocked",
            "Check suction valve fully open",
            "Reduce speed if VFD available",
        ],
        prevention_steps=["Maintain adequate suction pressure"],
        severity="high",
        sources=["domain_expert", "pump_manual"],
    ),
    FailureMode(
        id="fm_motor_overheating",
        equipment_type="motor",
        symptom_signature={"temperature": "high", "vibration": "normal", "current": "high"},
        root_cause="overload",
        description="Motor overload — drawing excess current, generating heat",
        confidence_base=0.68,
        resolution_steps=[
            "Check current draw vs rated current",
            "Verify load conditions",
            "Check for mechanical binding downstream",
            "Verify voltage at motor terminals",
        ],
        prevention_steps=["Set up overcurrent protection relays"],
        severity="high",
        sources=["domain_expert"],
    ),
    FailureMode(
        id="fm_motor_insulation",
        equipment_type="motor",
        symptom_signature={"temperature": "high", "vibration": "normal"},
        root_cause="insulation_degradation",
        description="Winding insulation degradation — hotspots in stator",
        confidence_base=0.55,
        resolution_steps=[
            "Perform insulation resistance test (megger)",
            "Check winding temperature distribution",
            "Perform surge test if accessible",
        ],
        prevention_steps=["Annual insulation resistance testing"],
        severity="critical",
        sources=["domain_expert"],
    ),
    FailureMode(
        id="fm_transformer_overload",
        equipment_type="transformer",
        symptom_signature={"temperature": "high", "load": "high"},
        root_cause="transformer_overload",
        description="Transformer operating above rated capacity",
        confidence_base=0.75,
        resolution_steps=[
            "Check load vs rating",
            "Verify cooling fans operational",
            "Check oil level and quality",
            "Consider load redistribution",
        ],
        prevention_steps=["Monitor loading factor trend"],
        severity="high",
        sources=["domain_expert"],
    ),
]


class CausalKnowledgeEngine:
    """
    Symptom-based failure mode matching with self-healing knowledge gaps.
    """

    def __init__(self, failure_modes: list[FailureMode] = None,
                  failure_mode_db_path: str = None):
        self.failure_modes = failure_modes or list(BUILTIN_FAILURE_MODES)
        if failure_mode_db_path and os.path.exists(failure_mode_db_path):
            self._load_from_file(failure_mode_db_path)
        self.knowledge_gaps: list[KnowledgeGap] = []

    def _load_from_file(self, path: str):
        """Load additional failure modes from JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
            for fm_data in data.get("failure_modes", []):
                fm = FailureMode(
                    id=fm_data["id"],
                    equipment_type=fm_data["equipment_type"],
                    symptom_signature=fm_data["symptom_signature"],
                    root_cause=fm_data["root_cause"],
                    description=fm_data["description"],
                    confidence_base=fm_data.get("confidence_base", 0.6),
                    resolution_steps=fm_data.get("resolution_steps", []),
                    prevention_steps=fm_data.get("prevention_steps", []),
                    severity=fm_data.get("severity", "medium"),
                    sources=fm_data.get("sources", []),
                )
                self.failure_modes.append(fm)
            logger.info(f"Loaded {len(data.get('failure_modes', []))} failure modes from {path}")
        except Exception as e:
            logger.warning(f"Failed to load failure modes from {path}: {e}")

    def diagnose(self, equipment_type: str, symptoms: dict,
                  focus_graph=None) -> list[SymptomMatch]:
        """
        Match observed symptoms against known failure modes.

        Args:
            equipment_type: e.g., "centrifugal_pump", "motor"
            symptoms: e.g., {"vibration": "high", "temperature": "normal"}
            focus_graph: Optional focus graph for additional context

        Returns:
            List of SymptomMatch sorted by match_score descending
        """
        matches = []

        for fm in self.failure_modes:
            # Filter by equipment type (fuzzy match)
            if not self._equipment_type_matches(fm.equipment_type, equipment_type):
                continue

            match = self._score_symptom_match(fm, symptoms)
            if match.match_score > 0:
                matches.append(match)

        # Sort by match score
        matches.sort(key=lambda m: m.match_score, reverse=True)

        # Log knowledge gap if no matches
        if not matches and symptoms:
            gap = KnowledgeGap(
                equipment_type=equipment_type,
                query=f"Symptoms: {symptoms}",
                expected_knowledge_type="failure_mode",
            )
            self.knowledge_gaps.append(gap)
            logger.info(f"Knowledge gap detected: {gap.id} — no failure modes for "
                        f"{equipment_type} with symptoms {symptoms}")

        return matches

    def get_knowledge_gaps(self) -> list[dict]:
        """Return all unresolved knowledge gaps."""
        return [g.to_dict() for g in self.knowledge_gaps if not g.resolved]

    def _equipment_type_matches(self, fm_type: str, query_type: str) -> bool:
        """Fuzzy match equipment types."""
        fm_lower = fm_type.lower()
        query_lower = query_type.lower()
        # Exact match
        if fm_lower == query_lower:
            return True
        # Partial match (e.g., "pump" matches "centrifugal_pump")
        if query_lower in fm_lower or fm_lower in query_lower:
            return True
        return False

    def _score_symptom_match(self, fm: FailureMode, symptoms: dict) -> SymptomMatch:
        """Score how well observed symptoms match a failure mode."""
        matched = []
        missing = []
        contradicting = []

        for symptom_key, expected_value in fm.symptom_signature.items():
            observed = symptoms.get(symptom_key)
            if observed is None:
                missing.append(symptom_key)
            elif self._symptom_values_match(observed, expected_value):
                matched.append(symptom_key)
            else:
                contradicting.append(symptom_key)

        total_symptoms = len(fm.symptom_signature)
        if total_symptoms == 0:
            score = 0.0
        else:
            # Score: matched / total, penalized by contradictions
            score = (len(matched) / total_symptoms) * fm.confidence_base
            if contradicting:
                score *= max(0.1, 1.0 - 0.3 * len(contradicting))

        return SymptomMatch(
            failure_mode=fm,
            match_score=round(score, 3),
            matched_symptoms=matched,
            missing_symptoms=missing,
            contradicting_symptoms=contradicting,
        )

    def _symptom_values_match(self, observed: str, expected: str) -> bool:
        """Check if observed symptom matches expected (fuzzy)."""
        observed_lower = observed.lower()
        expected_lower = expected.lower()
        if observed_lower == expected_lower:
            return True
        # Severity equivalences
        high_words = {"high", "elevated", "increased", "above_threshold", "critical", "warning"}
        normal_words = {"normal", "ok", "within_range", "nominal", "stable"}
        low_words = {"low", "decreased", "below_threshold", "reduced"}
        if observed_lower in high_words and expected_lower in high_words:
            return True
        if observed_lower in normal_words and expected_lower in normal_words:
            return True
        if observed_lower in low_words and expected_lower in low_words:
            return True
        return False
