"""
Type definitions for the Reconciliation Pipeline.

All structured types, JSON schemas, and provenance models.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional, Callable
import json
import hashlib
import uuid


# ═══════════════════════════════════════════════════════════════════════════════
# MISMATCH CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class MismatchClass(Enum):
    """
    Formal classification of mismatches between LLM output and schema.

    Determines what operations are ALLOWED:
    - NONE: No mismatch, pass through
    - STRUCTURAL_EQUIVALENCE: Different structure, same meaning → rewrite allowed
    - REPRESENTATIONAL_EQUIVALENCE: Different representation, same value → rewrite allowed
    - UNKNOWN_AMBIGUOUS: Cannot determine without context → resolver attempt allowed
    - SEMANTIC_DIFFERENCE: Different meaning → refuse or escalate
    - SECURITY_VIOLATION: Injection/XSS detected → immediate reject
    """
    NONE = auto()
    STRUCTURAL_EQUIVALENCE = auto()
    REPRESENTATIONAL_EQUIVALENCE = auto()
    UNKNOWN_AMBIGUOUS = auto()
    SEMANTIC_DIFFERENCE = auto()
    SECURITY_VIOLATION = auto()


class DecisionType(Enum):
    """Type of decision made by the pipeline."""
    PASSTHROUGH = auto()      # No changes needed
    TRANSFORM = auto()        # Syntactic rewrite applied
    NORMALIZE = auto()        # Domain normalization applied
    RESOLVE = auto()          # LLM resolver provided fix
    REFUSE = auto()           # Cannot reconcile, structured refusal
    ESCALATE = auto()         # Requires human intervention


# ═══════════════════════════════════════════════════════════════════════════════
# PROVENANCE & AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Provenance:
    """
    Provenance record for a single transformation.

    Attached to every change for full audit trail.
    """
    timestamp: str
    rule_id: str
    transform_description: str
    original_value_snippet: str
    transformed_value_snippet: str
    reversible: bool
    proof_token: Optional[str] = None  # Hash proving reversibility
    inverse_function: Optional[str] = None  # Name of inverse operation

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "rule_id": self.rule_id,
            "transform_description": self.transform_description,
            "original_value_snippet": self.original_value_snippet,
            "transformed_value_snippet": self.transformed_value_snippet,
            "reversible": self.reversible,
            "proof_token": self.proof_token,
            "inverse_function": self.inverse_function,
        }

    @staticmethod
    def create(
        rule_id: str,
        description: str,
        original: Any,
        transformed: Any,
        reversible: bool = True,
        inverse_function: Optional[str] = None,
    ) -> "Provenance":
        """Factory method with auto-generated timestamp and proof token."""
        original_str = str(original)[:100]
        transformed_str = str(transformed)[:100]

        # Generate proof token: hash of original + transformed
        proof_data = f"{original_str}|{transformed_str}|{rule_id}"
        proof_token = hashlib.sha256(proof_data.encode()).hexdigest()[:16]

        return Provenance(
            timestamp=datetime.utcnow().isoformat() + "Z",
            rule_id=rule_id,
            transform_description=description,
            original_value_snippet=original_str,
            transformed_value_snippet=transformed_str,
            reversible=reversible,
            proof_token=proof_token,
            inverse_function=inverse_function,
        )


@dataclass
class ReconcileEvent:
    """
    Audit event for reconciliation decisions.

    Stored in append-only audit sink for compliance and debugging.
    """
    event_id: str
    timestamp: str
    scenario: str
    decision: DecisionType
    mismatch_class: MismatchClass
    input_hash: str
    output_hash: Optional[str]
    provenance: list[Provenance]
    attempts: int
    success: bool
    error_message: Optional[str] = None
    escalation_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "scenario": self.scenario,
            "decision": self.decision.name,
            "mismatch_class": self.mismatch_class.name,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "provenance": [p.to_dict() for p in self.provenance],
            "attempts": self.attempts,
            "success": self.success,
            "error_message": self.error_message,
            "escalation_reason": self.escalation_reason,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def create(
        scenario: str,
        decision: DecisionType,
        mismatch_class: MismatchClass,
        input_data: Any,
        output_data: Optional[Any] = None,
        provenance: Optional[list[Provenance]] = None,
        attempts: int = 1,
        success: bool = True,
        error_message: Optional[str] = None,
        escalation_reason: Optional[str] = None,
    ) -> "ReconcileEvent":
        """Factory method with auto-generated IDs and hashes."""
        input_str = json.dumps(input_data, sort_keys=True, default=str)
        input_hash = hashlib.sha256(input_str.encode()).hexdigest()[:16]

        output_hash = None
        if output_data is not None:
            output_str = json.dumps(output_data, sort_keys=True, default=str)
            output_hash = hashlib.sha256(output_str.encode()).hexdigest()[:16]

        return ReconcileEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
            scenario=scenario,
            decision=decision,
            mismatch_class=mismatch_class,
            input_hash=input_hash,
            output_hash=output_hash,
            provenance=provenance or [],
            attempts=attempts,
            success=success,
            error_message=error_message,
            escalation_reason=escalation_reason,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MISMATCH REPORT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FieldMismatch:
    """Mismatch information for a single field."""
    field_path: str
    mismatch_class: MismatchClass
    expected_type: Optional[str]
    actual_type: str
    actual_value: Any
    reason: str
    rewritable: bool = False
    security_issue: Optional[str] = None


@dataclass
class MismatchReport:
    """
    Complete report of mismatches found during classification.

    Returned by classify_mismatch() to guide rewriting and resolution.
    """
    scenario: str
    overall_class: MismatchClass
    field_mismatches: list[FieldMismatch]
    missing_required: list[str]
    security_violations: list[str]
    rewritable_count: int
    requires_resolution: bool
    requires_escalation: bool

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "overall_class": self.overall_class.name,
            "field_mismatches": [
                {
                    "field_path": fm.field_path,
                    "mismatch_class": fm.mismatch_class.name,
                    "expected_type": fm.expected_type,
                    "actual_type": fm.actual_type,
                    "actual_value": str(fm.actual_value)[:50],
                    "reason": fm.reason,
                    "rewritable": fm.rewritable,
                    "security_issue": fm.security_issue,
                }
                for fm in self.field_mismatches
            ],
            "missing_required": self.missing_required,
            "security_violations": self.security_violations,
            "rewritable_count": self.rewritable_count,
            "requires_resolution": self.requires_resolution,
            "requires_escalation": self.requires_escalation,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REWRITE RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RewriteResult:
    """
    Result of applying rewrite rules.

    Contains transformed data and provenance for all changes.
    """
    success: bool
    data: dict
    provenance: list[Provenance]
    transforms_applied: int
    remaining_mismatches: list[FieldMismatch]

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "transforms_applied": self.transforms_applied,
            "provenance": [p.to_dict() for p in self.provenance],
            "remaining_mismatches": len(self.remaining_mismatches),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RESOLVER TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Assumption:
    """
    Explicit assumption made during resolution.

    MUST be attached to output and visible to downstream systems.
    """
    field: str
    assumed_value: Any
    basis: str  # Why this assumption was made
    confidence: float  # 0.0 to 1.0
    source: str  # "llm", "default", "inference"


@dataclass
class ResolveCandidate:
    """
    LLM-provided repair candidate.

    JSON Schema for LLM output:
    {
        "value": <any>,
        "unit": <string|null>,
        "metric_id": <string|null>,
        "frame": <string|null>,  # "instant", "hourly", "daily", "monthly", etc.
        "assumptions": [
            {"field": <string>, "assumed_value": <any>, "basis": <string>}
        ],
        "confidence": <float 0-1>,
        "reasoning": <string>
    }
    """
    value: Any
    unit: Optional[str]
    metric_id: Optional[str]
    frame: Optional[str]
    assumptions: list[Assumption]
    confidence: float
    reasoning: str

    @classmethod
    def from_dict(cls, d: dict) -> "ResolveCandidate":
        """Parse LLM JSON response into ResolveCandidate."""
        assumptions = [
            Assumption(
                field=a.get("field", ""),
                assumed_value=a.get("assumed_value"),
                basis=a.get("basis", ""),
                confidence=a.get("confidence", 0.5),
                source="llm",
            )
            for a in d.get("assumptions", [])
        ]
        return cls(
            value=d.get("value"),
            unit=d.get("unit"),
            metric_id=d.get("metric_id"),
            frame=d.get("frame"),
            assumptions=assumptions,
            confidence=d.get("confidence", 0.0),
            reasoning=d.get("reasoning", ""),
        )

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "metric_id": self.metric_id,
            "frame": self.frame,
            "assumptions": [
                {
                    "field": a.field,
                    "assumed_value": a.assumed_value,
                    "basis": a.basis,
                    "confidence": a.confidence,
                    "source": a.source,
                }
                for a in self.assumptions
            ],
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class ResolveAttempt:
    """Record of a single resolution attempt."""
    attempt_number: int
    prompt_used: str
    llm_response_raw: str
    candidate: Optional[ResolveCandidate]
    parse_success: bool
    validation_passed: bool
    error: Optional[str] = None


@dataclass
class ResolveResult:
    """
    Result of LLM-based resolution.

    Contains resolved data, all attempts, and explicit assumptions.
    """
    success: bool
    data: Optional[dict]
    candidate: Optional[ResolveCandidate]
    attempts: list[ResolveAttempt]
    assumptions: list[Assumption]
    requires_escalation: bool
    escalation_reason: Optional[str]
    provenance: list[Provenance]

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "attempts_count": len(self.attempts),
            "assumptions": [
                {
                    "field": a.field,
                    "assumed_value": a.assumed_value,
                    "basis": a.basis,
                    "confidence": a.confidence,
                }
                for a in self.assumptions
            ],
            "requires_escalation": self.requires_escalation,
            "escalation_reason": self.escalation_reason,
            "provenance": [p.to_dict() for p in self.provenance],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NormalizationResult:
    """
    Result of domain-aware normalization.

    Separate from syntactic rewriting - uses domain configuration.
    """
    success: bool
    data: dict
    transforms_applied: int
    provenance: list[Provenance]
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "transforms_applied": self.transforms_applied,
            "provenance": [p.to_dict() for p in self.provenance],
            "warnings": self.warnings,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RefusalDetail:
    """Structured refusal with actionable information."""
    reason: str
    missing_fields: list[str]
    attempted_repairs: list[str]
    recommendations: list[str]
    escalation_contact: Optional[str] = None


@dataclass
class PipelineResult:
    """
    Final result of the reconciliation pipeline.

    INVARIANT: Either success=True and data passes validation,
    or success=False with structured refusal.
    """
    success: bool
    decision: DecisionType
    data: Optional[dict]
    validated: bool
    assumptions: list[Assumption]
    provenance: list[Provenance]
    audit_event: ReconcileEvent
    refusal: Optional[RefusalDetail] = None

    def to_dict(self) -> dict:
        result = {
            "success": self.success,
            "decision": self.decision.name,
            "validated": self.validated,
            "assumptions": [
                {
                    "field": a.field,
                    "assumed_value": a.assumed_value,
                    "basis": a.basis,
                    "confidence": a.confidence,
                }
                for a in self.assumptions
            ],
            "provenance": [p.to_dict() for p in self.provenance],
            "audit_event_id": self.audit_event.event_id,
        }
        if self.refusal:
            result["refusal"] = {
                "reason": self.refusal.reason,
                "missing_fields": self.refusal.missing_fields,
                "attempted_repairs": self.refusal.attempted_repairs,
                "recommendations": self.refusal.recommendations,
            }
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FieldSchema:
    """Schema for a single field."""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    required: bool = False
    unit_dimension: Optional[str] = None  # "power", "energy", "temperature", etc.
    valid_values: Optional[list[str]] = None


@dataclass
class WidgetSchema:
    """Schema for a widget scenario."""
    scenario: str
    fields: list[FieldSchema]
    required_fields: list[str]

    def get_field(self, name: str) -> Optional[FieldSchema]:
        for f in self.fields:
            if f.name == name:
                return f
        return None
