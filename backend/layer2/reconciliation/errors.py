"""
Error types for the Reconciliation Pipeline.

All errors are structured and contain actionable information.
"""
from dataclasses import dataclass, field
from typing import Optional, Any


class ReconcileError(Exception):
    """
    Base error for reconciliation failures.

    All errors in this hierarchy contain structured data for logging and debugging.
    """

    def __init__(
        self,
        message: str,
        scenario: Optional[str] = None,
        field_path: Optional[str] = None,
        original_value: Optional[Any] = None,
        attempted_fixes: Optional[list[str]] = None,
    ):
        self.message = message
        self.scenario = scenario
        self.field_path = field_path
        self.original_value = original_value
        self.attempted_fixes = attempted_fixes or []
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "scenario": self.scenario,
            "field_path": self.field_path,
            "original_value": str(self.original_value)[:100] if self.original_value else None,
            "attempted_fixes": self.attempted_fixes,
        }


class ClassificationError(ReconcileError):
    """Error during mismatch classification phase."""
    pass


class RewriteError(ReconcileError):
    """
    Error during rewrite phase.

    Raised when a rewrite rule fails or produces invalid output.
    """

    def __init__(
        self,
        message: str,
        rule_id: str,
        scenario: Optional[str] = None,
        field_path: Optional[str] = None,
        original_value: Optional[Any] = None,
    ):
        super().__init__(message, scenario, field_path, original_value)
        self.rule_id = rule_id

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["rule_id"] = self.rule_id
        return d


class ResolutionError(ReconcileError):
    """
    Error during LLM resolution phase.

    Contains all attempts made before failure.
    """

    def __init__(
        self,
        message: str,
        attempts: int,
        last_llm_response: Optional[str] = None,
        scenario: Optional[str] = None,
        field_path: Optional[str] = None,
    ):
        super().__init__(message, scenario, field_path)
        self.attempts = attempts
        self.last_llm_response = last_llm_response

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["attempts"] = self.attempts
        d["last_llm_response"] = self.last_llm_response[:200] if self.last_llm_response else None
        return d


class NormalizationError(ReconcileError):
    """
    Error during domain normalization phase.

    Raised when domain-specific normalization cannot proceed.
    """

    def __init__(
        self,
        message: str,
        dimension: Optional[str] = None,
        from_unit: Optional[str] = None,
        to_unit: Optional[str] = None,
        scenario: Optional[str] = None,
        field_path: Optional[str] = None,
    ):
        super().__init__(message, scenario, field_path)
        self.dimension = dimension
        self.from_unit = from_unit
        self.to_unit = to_unit

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["dimension"] = self.dimension
        d["from_unit"] = self.from_unit
        d["to_unit"] = self.to_unit
        return d


class EscalationRequired(ReconcileError):
    """
    Escalation to human operator required.

    Raised when:
    - Semantic difference detected
    - LLM confidence below threshold
    - Conflicting constraints
    - Max resolution attempts exhausted with ambiguity
    """

    def __init__(
        self,
        message: str,
        reason: str,
        missing_fields: Optional[list[str]] = None,
        conflicting_values: Optional[dict[str, Any]] = None,
        recommendations: Optional[list[str]] = None,
        scenario: Optional[str] = None,
    ):
        super().__init__(message, scenario)
        self.reason = reason
        self.missing_fields = missing_fields or []
        self.conflicting_values = conflicting_values or {}
        self.recommendations = recommendations or []

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["reason"] = self.reason
        d["missing_fields"] = self.missing_fields
        d["conflicting_values"] = {k: str(v)[:50] for k, v in self.conflicting_values.items()}
        d["recommendations"] = self.recommendations
        return d

    def to_escalation_ticket(self) -> dict:
        """Format for human escalation system."""
        return {
            "type": "RECONCILIATION_ESCALATION",
            "severity": "HIGH",
            "reason": self.reason,
            "scenario": self.scenario,
            "missing_fields": self.missing_fields,
            "conflicting_values": self.conflicting_values,
            "recommendations": self.recommendations,
            "action_required": "Manual review and data correction required",
        }


class SecurityViolation(ReconcileError):
    """
    Security violation detected in input.

    Raised BEFORE any transformation for:
    - SQL injection patterns
    - XSS patterns
    - Excessive nesting (DoS protection)
    """

    def __init__(
        self,
        message: str,
        violation_type: str,  # "sql_injection", "xss", "dos_nesting"
        pattern_matched: Optional[str] = None,
        field_path: Optional[str] = None,
    ):
        super().__init__(message, field_path=field_path)
        self.violation_type = violation_type
        self.pattern_matched = pattern_matched

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["violation_type"] = self.violation_type
        d["pattern_matched"] = self.pattern_matched
        return d


class ValidationGateError(ReconcileError):
    """
    Final validation gate failed.

    This is a HARD FAILURE - data cannot be presented to users.
    """

    def __init__(
        self,
        message: str,
        validation_errors: list[str],
        scenario: Optional[str] = None,
    ):
        super().__init__(message, scenario)
        self.validation_errors = validation_errors

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["validation_errors"] = self.validation_errors
        return d
