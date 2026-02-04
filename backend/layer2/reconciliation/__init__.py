"""
Reconciliation Pipeline for LLM Output → Validated Widget Data.

Pipeline: LLM → classify → rewrite → resolve → normalize → validate → render

INVARIANTS (non-negotiable):
1. Final output MUST pass validate_widget_data() or be explicit refusal
2. No silent semantic guessing
3. All transforms are lossless, reversible, self-declaring
4. Full provenance and audit trail for all decisions
"""
from layer2.reconciliation.types import (
    MismatchClass,
    MismatchReport,
    RewriteResult,
    ResolveResult,
    NormalizationResult,
    Provenance,
    ReconcileEvent,
    PipelineResult,
)
from layer2.reconciliation.errors import (
    ReconcileError,
    ResolutionError,
    NormalizationError,
    EscalationRequired,
)
from layer2.reconciliation.pipeline import ReconciliationPipeline

__all__ = [
    "ReconciliationPipeline",
    "MismatchClass",
    "MismatchReport",
    "RewriteResult",
    "ResolveResult",
    "NormalizationResult",
    "Provenance",
    "ReconcileEvent",
    "PipelineResult",
    "ReconcileError",
    "ResolutionError",
    "NormalizationError",
    "EscalationRequired",
]
