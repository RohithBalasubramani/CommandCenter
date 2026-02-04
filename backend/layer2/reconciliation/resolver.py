"""
Resolver — LLM-Based Ambiguity Resolution.

resolve_ambiguity(data: dict, report: MismatchReport, max_attempts=3) -> ResolveResult

Uses the LLM to resolve UNKNOWN_AMBIGUOUS mismatches with:
- Strict schema-driven prompts
- JSON-only output format
- Explicit assumptions tracking
- Confidence thresholds
- Escalation when uncertain

ALLOWED operations:
- Ask LLM to restate value in explicit schema form
- Convert representations if lossless
- Provide assumptions as explicit key-value pairs

NOT ALLOWED:
- Invent metric identity
- Assume semantics from labels
- Silently coerce values
"""
import copy
import json
import logging
from typing import Any, Optional, Callable
from dataclasses import dataclass

from layer2.reconciliation.types import (
    MismatchClass,
    MismatchReport,
    FieldMismatch,
    ResolveResult,
    ResolveCandidate,
    ResolveAttempt,
    Assumption,
    Provenance,
)
from layer2.reconciliation.errors import ResolutionError, EscalationRequired
from layer2.reconciliation.prompts import (
    build_resolution_prompt,
    RESOLVE_CANDIDATE_SCHEMA,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum confidence for accepting LLM semantic claims
SEMANTIC_CONFIDENCE_THRESHOLD = 0.9

# Maximum attempts before giving up
DEFAULT_MAX_ATTEMPTS = 3

# Backoff multiplier between attempts (not time-based, but prompt complexity)
PROMPT_ESCALATION_LEVELS = ["basic", "detailed", "canonical"]


# ═══════════════════════════════════════════════════════════════════════════════
# LLM INTERFACE (Abstract - to be implemented with actual LLM client)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMResponse:
    """Response from LLM call."""
    content: str
    success: bool
    error: Optional[str] = None


# Type for LLM caller function
LLMCaller = Callable[[str], LLMResponse]


def default_llm_caller(prompt: str) -> LLMResponse:
    """
    Default LLM caller - raises NotImplementedError.

    In production, replace with actual LLM client.
    In tests, mock this function.
    """
    raise NotImplementedError(
        "LLM caller not configured. Provide an LLMCaller function."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE PARSING & VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def parse_llm_response(response_text: str) -> tuple[Optional[ResolveCandidate], Optional[str]]:
    """
    Parse LLM JSON response into ResolveCandidate.

    Strictly validates against schema. Returns (candidate, error).

    SECURITY: Uses json.loads(), not eval(). Validates types strictly.
    """
    try:
        # Strip markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Parse JSON
        data = json.loads(text)

        if not isinstance(data, dict):
            return None, f"Response must be object, got {type(data).__name__}"

        # Validate required fields
        if "value" not in data:
            return None, "Missing required field 'value'"

        if "confidence" not in data:
            return None, "Missing required field 'confidence'"

        confidence = data.get("confidence")
        if not isinstance(confidence, (int, float)):
            return None, f"Confidence must be number, got {type(confidence).__name__}"

        if not (0.0 <= confidence <= 1.0):
            return None, f"Confidence must be 0-1, got {confidence}"

        # Validate assumptions array
        assumptions = data.get("assumptions", [])
        if not isinstance(assumptions, list):
            return None, f"Assumptions must be array, got {type(assumptions).__name__}"

        for i, a in enumerate(assumptions):
            if not isinstance(a, dict):
                return None, f"assumptions[{i}] must be object"
            if "field" not in a:
                return None, f"assumptions[{i}] missing 'field'"
            if "basis" not in a:
                return None, f"assumptions[{i}] missing 'basis'"

        # Validate optional string fields
        for field in ["unit", "metric_id", "frame", "reasoning"]:
            if field in data and data[field] is not None:
                if not isinstance(data[field], str):
                    return None, f"{field} must be string or null, got {type(data[field]).__name__}"

        # Parse into ResolveCandidate
        candidate = ResolveCandidate.from_dict(data)
        return candidate, None

    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    except Exception as e:
        return None, f"Parse error: {e}"


def validate_candidate_safety(candidate: ResolveCandidate) -> tuple[bool, Optional[str]]:
    """
    Validate that a candidate is safe to apply.

    Checks:
    - No semantic claims with low confidence
    - Assumptions are explicit and reasonable
    - No obvious injection patterns in values
    """
    # Check semantic claims (metric_id) require high confidence
    if candidate.metric_id and candidate.confidence < SEMANTIC_CONFIDENCE_THRESHOLD:
        return False, (
            f"Semantic claim (metric_id='{candidate.metric_id}') with confidence "
            f"{candidate.confidence} < threshold {SEMANTIC_CONFIDENCE_THRESHOLD}"
        )

    # Check frame claims require high confidence
    if candidate.frame and candidate.confidence < SEMANTIC_CONFIDENCE_THRESHOLD:
        return False, (
            f"Frame claim (frame='{candidate.frame}') with confidence "
            f"{candidate.confidence} < threshold {SEMANTIC_CONFIDENCE_THRESHOLD}"
        )

    # Check assumptions have reasonable confidence
    for assumption in candidate.assumptions:
        if assumption.confidence < 0.5:
            return False, (
                f"Assumption for '{assumption.field}' has very low confidence "
                f"{assumption.confidence}"
            )

    return True, None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RESOLUTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_ambiguity(
    data: dict,
    report: MismatchReport,
    llm_caller: LLMCaller = default_llm_caller,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> ResolveResult:
    """
    Attempt to resolve ambiguous mismatches using LLM.

    Algorithm:
    1. For each UNKNOWN_AMBIGUOUS mismatch:
       a. Build schema-driven prompt with examples and constraints
       b. Call LLM with strict JSON output requirement
       c. Parse and validate response syntactically
       d. Check semantic claims against confidence threshold
       e. If valid, apply fix; if not, retry with more explicit prompt
    2. Retry up to max_attempts with increasing explicitness
    3. If still ambiguous after max_attempts, return structured refusal

    Args:
        data: Data to resolve
        report: MismatchReport from classification
        llm_caller: Function to call LLM (for dependency injection)
        max_attempts: Maximum resolution attempts

    Returns:
        ResolveResult with resolved data or structured refusal

    Raises:
        EscalationRequired: If semantic difference detected or confidence too low
    """
    # Filter to only UNKNOWN_AMBIGUOUS mismatches
    ambiguous = [
        m for m in report.field_mismatches
        if m.mismatch_class == MismatchClass.UNKNOWN_AMBIGUOUS
    ]

    if not ambiguous:
        # No ambiguous mismatches - nothing to resolve
        return ResolveResult(
            success=True,
            data=data,
            candidate=None,
            attempts=[],
            assumptions=[],
            requires_escalation=False,
            escalation_reason=None,
            provenance=[],
        )

    # Work on a copy
    result = copy.deepcopy(data)
    all_attempts: list[ResolveAttempt] = []
    all_assumptions: list[Assumption] = []
    all_provenance: list[Provenance] = []

    # Try to resolve each ambiguous mismatch
    for mismatch in ambiguous:
        field_resolved = False
        last_error: Optional[str] = None

        for attempt_num in range(1, max_attempts + 1):
            # Build prompt with escalating detail
            prompt_level = PROMPT_ESCALATION_LEVELS[
                min(attempt_num - 1, len(PROMPT_ESCALATION_LEVELS) - 1)
            ]

            prompt = build_resolution_prompt(
                field_path=mismatch.field_path,
                current_value=mismatch.actual_value,
                expected_type=mismatch.expected_type,
                scenario=report.scenario,
                level=prompt_level,
                previous_error=last_error if attempt_num > 1 else None,
            )

            # Call LLM
            try:
                llm_response = llm_caller(prompt)
            except NotImplementedError:
                # LLM not configured - cannot resolve
                return ResolveResult(
                    success=False,
                    data=None,
                    candidate=None,
                    attempts=all_attempts,
                    assumptions=[],
                    requires_escalation=True,
                    escalation_reason="LLM not configured for resolution",
                    provenance=[],
                )
            except Exception as e:
                last_error = f"LLM call failed: {e}"
                all_attempts.append(ResolveAttempt(
                    attempt_number=attempt_num,
                    prompt_used=prompt,
                    llm_response_raw="",
                    candidate=None,
                    parse_success=False,
                    validation_passed=False,
                    error=last_error,
                ))
                continue

            if not llm_response.success:
                last_error = f"LLM error: {llm_response.error}"
                all_attempts.append(ResolveAttempt(
                    attempt_number=attempt_num,
                    prompt_used=prompt,
                    llm_response_raw=llm_response.content or "",
                    candidate=None,
                    parse_success=False,
                    validation_passed=False,
                    error=last_error,
                ))
                continue

            # Parse response
            candidate, parse_error = parse_llm_response(llm_response.content)

            if parse_error:
                last_error = parse_error
                all_attempts.append(ResolveAttempt(
                    attempt_number=attempt_num,
                    prompt_used=prompt,
                    llm_response_raw=llm_response.content,
                    candidate=None,
                    parse_success=False,
                    validation_passed=False,
                    error=parse_error,
                ))
                continue

            # Validate candidate safety
            is_safe, safety_error = validate_candidate_safety(candidate)

            if not is_safe:
                last_error = safety_error
                all_attempts.append(ResolveAttempt(
                    attempt_number=attempt_num,
                    prompt_used=prompt,
                    llm_response_raw=llm_response.content,
                    candidate=candidate,
                    parse_success=True,
                    validation_passed=False,
                    error=safety_error,
                ))

                # If semantic claim with low confidence, escalate
                if "metric_id" in str(safety_error) or "frame" in str(safety_error):
                    raise EscalationRequired(
                        message=f"LLM semantic claim requires human verification: {safety_error}",
                        reason=safety_error,
                        scenario=report.scenario,
                        recommendations=[
                            f"Verify the metric identity for field '{mismatch.field_path}'",
                            "Provide explicit metric_id in upstream data",
                        ],
                    )
                continue

            # Success! Apply the resolved value
            all_attempts.append(ResolveAttempt(
                attempt_number=attempt_num,
                prompt_used=prompt,
                llm_response_raw=llm_response.content,
                candidate=candidate,
                parse_success=True,
                validation_passed=True,
            ))

            # Apply to result
            if isinstance(result, dict) and mismatch.field_path in result:
                original = result[mismatch.field_path]
                result[mismatch.field_path] = candidate.value

                # Record provenance
                all_provenance.append(Provenance.create(
                    rule_id="llm_resolve",
                    description=f"LLM resolved '{mismatch.field_path}' with confidence {candidate.confidence}",
                    original=original,
                    transformed=candidate.value,
                    reversible=True,
                    inverse_function="manual_revert",
                ))

            # Record assumptions
            all_assumptions.extend(candidate.assumptions)

            field_resolved = True
            break

        if not field_resolved:
            # Failed to resolve this field after max_attempts
            logger.warning(
                f"Failed to resolve {mismatch.field_path} after {max_attempts} attempts. "
                f"Last error: {last_error}"
            )

    # Check if we resolved all ambiguous fields
    resolved_fields = {
        a.candidate.value
        for a in all_attempts
        if a.candidate and a.validation_passed
    }

    success = all(
        any(
            a.candidate and a.validation_passed
            for a in all_attempts
        )
        for m in ambiguous
    )

    return ResolveResult(
        success=success,
        data=result if success else None,
        candidate=all_attempts[-1].candidate if all_attempts and all_attempts[-1].validation_passed else None,
        attempts=all_attempts,
        assumptions=all_assumptions,
        requires_escalation=not success,
        escalation_reason=f"Failed to resolve after {max_attempts} attempts" if not success else None,
        provenance=all_provenance,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK LLM FOR TESTING
# ═══════════════════════════════════════════════════════════════════════════════

class MockLLMCaller:
    """
    Mock LLM caller for deterministic testing.

    Provides canned responses based on input patterns.
    """

    def __init__(self, responses: Optional[dict[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> LLMResponse:
        self.call_count += 1
        self.calls.append(prompt)

        # Check for exact match
        if prompt in self.responses:
            return LLMResponse(content=self.responses[prompt], success=True)

        # Check for pattern match
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return LLMResponse(content=response, success=True)

        # Default response for testing
        if "value" in prompt.lower():
            return LLMResponse(
                content=json.dumps({
                    "value": 42,
                    "unit": "kW",
                    "metric_id": None,
                    "frame": None,
                    "assumptions": [],
                    "confidence": 0.95,
                    "reasoning": "Extracted numeric value from string",
                }),
                success=True,
            )

        return LLMResponse(
            content="",
            success=False,
            error="No mock response configured",
        )

    def add_response(self, pattern: str, response: str) -> None:
        """Add a response for a prompt pattern."""
        self.responses[pattern] = response
