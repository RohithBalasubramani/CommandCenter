"""
Pipeline — Complete Reconciliation Orchestration.

LLM → classify → rewrite → resolve → normalize → validate → render

This is the main entry point for the reconciliation system.

INVARIANTS:
1. Final output MUST pass validate_widget_data() or be explicit refusal
2. All transformations are lossless, reversible, self-declaring
3. Full provenance and audit trail for all decisions
"""
import copy
import logging
from typing import Optional, Callable

from layer2.reconciliation.types import (
    MismatchClass,
    MismatchReport,
    RewriteResult,
    ResolveResult,
    NormalizationResult,
    PipelineResult,
    DecisionType,
    Provenance,
    RefusalDetail,
    ReconcileEvent,
)
from layer2.reconciliation.errors import (
    ReconcileError,
    SecurityViolation,
    EscalationRequired,
    ValidationGateError,
)
from layer2.reconciliation.reconciler import (
    classify_mismatch,
    build_schema_from_widget,
)
from layer2.reconciliation.rewriter import apply_rewrite_rules
from layer2.reconciliation.resolver import (
    resolve_ambiguity,
    LLMCaller,
    default_llm_caller,
)
from layer2.reconciliation.normalizer import normalize_domain
from layer2.reconciliation.validator_integration import (
    validate_final,
    validate_or_refuse,
)
from layer2.reconciliation.audit import (
    audit_event,
    audit_transform,
    audit_refuse,
    audit_escalate,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ReconciliationPipeline:
    """
    Complete reconciliation pipeline.

    Usage:
        pipeline = ReconciliationPipeline()
        result = pipeline.process(scenario="kpi", data=llm_output)

        if result.success:
            # Use result.data
            pass
        else:
            # Handle refusal with result.refusal
            pass
    """

    def __init__(
        self,
        llm_caller: Optional[LLMCaller] = None,
        max_resolve_attempts: int = 3,
        enable_domain_normalization: bool = True,
    ):
        """
        Initialize the pipeline.

        Args:
            llm_caller: Function to call LLM for resolution (optional)
            max_resolve_attempts: Maximum resolution attempts
            enable_domain_normalization: Enable domain-aware unit normalization
        """
        self.llm_caller = llm_caller or default_llm_caller
        self.max_resolve_attempts = max_resolve_attempts
        self.enable_domain_normalization = enable_domain_normalization

    def process(
        self,
        scenario: str,
        data: dict,
    ) -> PipelineResult:
        """
        Process data through the complete pipeline.

        Pipeline stages:
        1. CLASSIFY - Identify mismatches
        2. REWRITE - Apply syntactic transformations
        3. RESOLVE - LLM-based ambiguity resolution (if needed)
        4. NORMALIZE - Domain-aware unit normalization (if enabled)
        5. VALIDATE - Final validation gate

        Args:
            scenario: Widget scenario name
            data: Raw data from LLM

        Returns:
            PipelineResult with either validated data or structured refusal
        """
        all_provenance: list[Provenance] = []
        all_assumptions = []
        current_data = copy.deepcopy(data)
        attempts = 0

        try:
            # ═══════════════════════════════════════════════════════════════
            # STAGE 1: CLASSIFY
            # ═══════════════════════════════════════════════════════════════
            logger.info(f"PIPELINE [{scenario}]: Stage 1 - Classify")

            schema = build_schema_from_widget(scenario)
            report = classify_mismatch(current_data, schema)

            logger.info(
                f"PIPELINE [{scenario}]: Classification complete - "
                f"class={report.overall_class.name}, "
                f"rewritable={report.rewritable_count}, "
                f"missing={report.missing_required}"
            )

            # Check for immediate failures
            if report.security_violations:
                return self._create_refusal(
                    scenario=scenario,
                    data=data,
                    reason="Security violation detected",
                    recommendations=["Review input for injection patterns"],
                    provenance=all_provenance,
                    mismatch_class=MismatchClass.SECURITY_VIOLATION,
                )

            if report.requires_escalation:
                return self._create_escalation(
                    scenario=scenario,
                    data=data,
                    reason="Semantic difference detected - cannot reconcile",
                    missing=report.missing_required,
                    provenance=all_provenance,
                )

            # No mismatches - pass through
            if report.overall_class == MismatchClass.NONE:
                logger.info(f"PIPELINE [{scenario}]: No mismatches - passthrough")
                return self._validate_and_complete(
                    scenario=scenario,
                    data=current_data,
                    decision=DecisionType.PASSTHROUGH,
                    provenance=all_provenance,
                    assumptions=[],
                    attempts=1,
                )

            # ═══════════════════════════════════════════════════════════════
            # STAGE 2: REWRITE
            # ═══════════════════════════════════════════════════════════════
            logger.info(f"PIPELINE [{scenario}]: Stage 2 - Rewrite")

            rewrite_result = apply_rewrite_rules(current_data, report)
            current_data = rewrite_result.data
            all_provenance.extend(rewrite_result.provenance)
            attempts += 1

            logger.info(
                f"PIPELINE [{scenario}]: Rewrite complete - "
                f"transforms={rewrite_result.transforms_applied}, "
                f"remaining={len(rewrite_result.remaining_mismatches)}"
            )

            # If all mismatches resolved, try validation
            if rewrite_result.success:
                return self._validate_and_complete(
                    scenario=scenario,
                    data=current_data,
                    decision=DecisionType.TRANSFORM,
                    provenance=all_provenance,
                    assumptions=[],
                    attempts=attempts,
                )

            # ═══════════════════════════════════════════════════════════════
            # STAGE 3: RESOLVE (if needed)
            # ═══════════════════════════════════════════════════════════════
            if report.requires_resolution or rewrite_result.remaining_mismatches:
                logger.info(f"PIPELINE [{scenario}]: Stage 3 - Resolve")

                try:
                    resolve_result = resolve_ambiguity(
                        data=current_data,
                        report=report,
                        llm_caller=self.llm_caller,
                        max_attempts=self.max_resolve_attempts,
                    )
                    attempts += len(resolve_result.attempts)
                    all_provenance.extend(resolve_result.provenance)
                    all_assumptions.extend(resolve_result.assumptions)

                    if resolve_result.success and resolve_result.data:
                        current_data = resolve_result.data
                        logger.info(
                            f"PIPELINE [{scenario}]: Resolution successful - "
                            f"attempts={len(resolve_result.attempts)}"
                        )
                    elif resolve_result.requires_escalation:
                        return self._create_escalation(
                            scenario=scenario,
                            data=data,
                            reason=resolve_result.escalation_reason or "Resolution failed",
                            missing=report.missing_required,
                            provenance=all_provenance,
                        )
                    else:
                        return self._create_refusal(
                            scenario=scenario,
                            data=data,
                            reason="Resolution failed after max attempts",
                            recommendations=[
                                "Provide more explicit data format",
                                "Include unit and metric_id in source data",
                            ],
                            provenance=all_provenance,
                            mismatch_class=MismatchClass.UNKNOWN_AMBIGUOUS,
                        )

                except EscalationRequired as e:
                    return self._create_escalation(
                        scenario=scenario,
                        data=data,
                        reason=e.reason,
                        missing=e.missing_fields,
                        provenance=all_provenance,
                    )

            # ═══════════════════════════════════════════════════════════════
            # STAGE 4: NORMALIZE (domain-aware)
            # ═══════════════════════════════════════════════════════════════
            if self.enable_domain_normalization:
                logger.info(f"PIPELINE [{scenario}]: Stage 4 - Normalize")

                norm_result = normalize_domain(current_data, scenario)
                current_data = norm_result.data
                all_provenance.extend(norm_result.provenance)

                logger.info(
                    f"PIPELINE [{scenario}]: Normalization complete - "
                    f"transforms={norm_result.transforms_applied}"
                )

            # ═══════════════════════════════════════════════════════════════
            # STAGE 5: VALIDATE
            # ═══════════════════════════════════════════════════════════════
            decision = DecisionType.RESOLVE if all_assumptions else DecisionType.TRANSFORM
            return self._validate_and_complete(
                scenario=scenario,
                data=current_data,
                decision=decision,
                provenance=all_provenance,
                assumptions=all_assumptions,
                attempts=attempts,
            )

        except SecurityViolation as e:
            logger.error(f"PIPELINE [{scenario}]: Security violation - {e}")
            return self._create_refusal(
                scenario=scenario,
                data=data,
                reason=f"Security violation: {e.message}",
                recommendations=["Remove injection patterns from input"],
                provenance=all_provenance,
                mismatch_class=MismatchClass.SECURITY_VIOLATION,
            )

        except ReconcileError as e:
            logger.error(f"PIPELINE [{scenario}]: Reconcile error - {e}")
            return self._create_refusal(
                scenario=scenario,
                data=data,
                reason=str(e),
                recommendations=["Review data format", "Check required fields"],
                provenance=all_provenance,
                mismatch_class=MismatchClass.UNKNOWN_AMBIGUOUS,
            )

        except Exception as e:
            logger.exception(f"PIPELINE [{scenario}]: Unexpected error")
            return self._create_refusal(
                scenario=scenario,
                data=data,
                reason=f"Unexpected error: {e}",
                recommendations=["Contact support"],
                provenance=all_provenance,
                mismatch_class=MismatchClass.UNKNOWN_AMBIGUOUS,
            )

    def _validate_and_complete(
        self,
        scenario: str,
        data: dict,
        decision: DecisionType,
        provenance: list[Provenance],
        assumptions: list,
        attempts: int,
    ) -> PipelineResult:
        """Validate and create success result."""
        logger.info(f"PIPELINE [{scenario}]: Stage 5 - Validate")

        success, validated, errors = validate_or_refuse(scenario, data, provenance)

        if success:
            audit_event = audit_transform(scenario, data, validated, provenance)
            return PipelineResult(
                success=True,
                decision=decision,
                data=validated,
                validated=True,
                assumptions=assumptions,
                provenance=provenance,
                audit_event=audit_event,
            )
        else:
            return self._create_refusal(
                scenario=scenario,
                data=data,
                reason=f"Validation failed: {errors}",
                recommendations=["Check required fields", "Verify data types"],
                provenance=provenance,
                mismatch_class=MismatchClass.UNKNOWN_AMBIGUOUS,
            )

    def _create_refusal(
        self,
        scenario: str,
        data: dict,
        reason: str,
        recommendations: list[str],
        provenance: list[Provenance],
        mismatch_class: MismatchClass,
    ) -> PipelineResult:
        """Create a structured refusal result."""
        audit_event = audit_refuse(scenario, data, reason)

        refusal = RefusalDetail(
            reason=reason,
            missing_fields=[],
            attempted_repairs=[p.rule_id for p in provenance],
            recommendations=recommendations,
        )

        return PipelineResult(
            success=False,
            decision=DecisionType.REFUSE,
            data=None,
            validated=False,
            assumptions=[],
            provenance=provenance,
            audit_event=audit_event,
            refusal=refusal,
        )

    def _create_escalation(
        self,
        scenario: str,
        data: dict,
        reason: str,
        missing: list[str],
        provenance: list[Provenance],
    ) -> PipelineResult:
        """Create an escalation result."""
        audit_event = audit_escalate(scenario, data, reason)

        refusal = RefusalDetail(
            reason=reason,
            missing_fields=missing,
            attempted_repairs=[p.rule_id for p in provenance],
            recommendations=[
                "Escalate to human operator for review",
                "Verify metric identity and time frame",
                "Provide explicit metadata in source data",
            ],
            escalation_contact="ops-escalation@example.com",
        )

        return PipelineResult(
            success=False,
            decision=DecisionType.ESCALATE,
            data=None,
            validated=False,
            assumptions=[],
            provenance=provenance,
            audit_event=audit_event,
            refusal=refusal,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def reconcile(
    scenario: str,
    data: dict,
    llm_caller: Optional[LLMCaller] = None,
) -> PipelineResult:
    """
    Convenience function to run the reconciliation pipeline.

    Args:
        scenario: Widget scenario name
        data: Raw data from LLM
        llm_caller: Optional LLM caller for resolution

    Returns:
        PipelineResult
    """
    pipeline = ReconciliationPipeline(llm_caller=llm_caller)
    return pipeline.process(scenario, data)
