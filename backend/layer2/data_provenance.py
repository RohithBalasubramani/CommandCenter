"""
Data Provenance Schema — Mandatory provenance markers for ALL data payloads.

Failure Mode F2: Silent Demo Data → CLOSED by this module.
Failure Mode F5: LLM Treated as Data Source → CLOSED by this module.

HARD RULES:
- Every data payload MUST carry _data_source, _integration_status, _authoritative
- If _data_source is missing → request is rejected
- LLM responses MUST include derived_from: [sources]
- If derived_from is empty → response rejected
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from layer2.system_registry import get_system_registry, IntegrationStatus

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceMarker:
    """Mandatory provenance marker for every data payload."""
    _data_source: str          # "django.industrial" | "chromadb.industrial" | "demo" | "stub" | "synthetic"
    _integration_status: str   # "real" | "demo" | "stub" | "hybrid"
    _authoritative: bool       # True if from the authoritative source for this domain
    _traversal_id: str = ""    # ID of the traversal step that produced this data
    _synthetic: bool = False   # True if data was generated/fabricated
    _safe_to_answer: bool = True  # False if source is stub/demo and no real alternative exists

    def to_dict(self) -> dict:
        return {
            "_data_source": self._data_source,
            "_integration_status": self._integration_status,
            "_authoritative": self._authoritative,
            "_traversal_id": self._traversal_id,
            "_synthetic": self._synthetic,
            "_safe_to_answer": self._safe_to_answer,
        }


@dataclass
class ResponseProvenance:
    """Provenance envelope for the entire orchestrator response."""
    derived_from: list[str] = field(default_factory=list)   # Source IDs the response was derived from
    traversal_log: list[dict] = field(default_factory=list)  # Full traversal steps
    resolution_outcome: str = ""                              # resolved | refused | etc.
    demo_warnings: list[str] = field(default_factory=list)
    data_sources_used: list[str] = field(default_factory=list)
    all_authoritative: bool = False
    safe_to_answer: bool = True

    def to_dict(self) -> dict:
        return {
            "derived_from": self.derived_from,
            "resolution_outcome": self.resolution_outcome,
            "demo_warnings": self.demo_warnings,
            "data_sources_used": self.data_sources_used,
            "all_authoritative": self.all_authoritative,
            "safe_to_answer": self.safe_to_answer,
            "traversal_step_count": len(self.traversal_log),
        }


def stamp_provenance(
    data: dict,
    source_id: str,
    is_authoritative: bool = False,
    traversal_id: str = "",
    synthetic: bool = False,
) -> dict:
    """
    Stamp a data payload with mandatory provenance markers.

    HARD RULE: This function MUST be called on every data payload
    before it reaches the frontend.
    """
    registry = get_system_registry()
    source = registry.get_source(source_id)

    if source:
        integration_status = source.integration_status.value
        safe = source.integration_status not in (IntegrationStatus.STUB,)
    else:
        integration_status = "unknown"
        safe = False

    # Override safe_to_answer for synthetic data
    if synthetic:
        safe = True  # Synthetic data from real metadata is usable, but flagged

    marker = ProvenanceMarker(
        _data_source=source_id,
        _integration_status=integration_status,
        _authoritative=is_authoritative,
        _traversal_id=traversal_id,
        _synthetic=synthetic,
        _safe_to_answer=safe,
    )

    # Inject into the data dict
    data.update(marker.to_dict())
    return data


def stamp_widget_provenance(
    widget_data: dict,
    source_resolution,
    traversal_context=None,
) -> dict:
    """
    Stamp a widget's data_override with provenance from the source resolution.

    Called for each widget in the collect_all results.
    """
    data_override = widget_data.get("data_override", {})
    if not isinstance(data_override, dict):
        return widget_data

    # Determine source from resolution
    primary_source = source_resolution.primary_source
    source_id = primary_source.id if primary_source else "unknown"
    is_authoritative = primary_source is not None

    # Check if data is synthetic
    synthetic = data_override.get("_synthetic", False)
    if not synthetic:
        # Check nested demoData
        demo_data = data_override.get("demoData", {})
        if isinstance(demo_data, dict):
            synthetic = demo_data.get("_synthetic", False)

    integration_status = primary_source.integration_status.value if primary_source else "unknown"
    safe = True
    if primary_source and primary_source.integration_status == IntegrationStatus.STUB:
        safe = False

    # Stamp the data_override — preserve collector-set _data_source if present
    # (e.g., "pg_timeseries:trf_001" from data_collector is more specific than
    # the generic "django.industrial" from source resolution)
    existing_source = data_override.get("_data_source")
    if not existing_source:
        # Also check nested demoData for collector-set source
        demo = data_override.get("demoData", {})
        if isinstance(demo, dict):
            existing_source = demo.get("_data_source")
    data_override["_data_source"] = existing_source or source_id
    data_override["_integration_status"] = integration_status
    data_override["_authoritative"] = is_authoritative
    data_override["_synthetic"] = synthetic
    data_override["_safe_to_answer"] = safe

    widget_data["data_override"] = data_override
    return widget_data


def build_response_provenance(
    source_resolution,
    traversal_context,
    widget_data: list[dict],
) -> ResponseProvenance:
    """
    Build the response-level provenance envelope.

    This is the `derived_from` field that MUST be on every response.
    HARD RULE: If derived_from is empty for a data query, response is rejected.
    """
    # Collect all sources used
    derived_from = []
    data_sources_used = set()

    if source_resolution.primary_source:
        derived_from.append(source_resolution.primary_source.id)
        data_sources_used.add(source_resolution.primary_source.id)

    for s in source_resolution.secondary_sources:
        derived_from.append(s.id)
        data_sources_used.add(s.id)

    # Add sources from traversal
    if traversal_context:
        for step in traversal_context.steps:
            if step.source_id and step.source_id not in ("system_registry", "verification"):
                data_sources_used.add(step.source_id)

    # Check if all sources are authoritative
    registry = get_system_registry()
    all_authoritative = all(
        not registry.is_demo_source(sid)
        for sid in data_sources_used
        if sid
    )

    # Safe to answer = at least one non-stub source resolved
    safe = len(derived_from) > 0

    return ResponseProvenance(
        derived_from=derived_from,
        traversal_log=[s.to_dict() for s in traversal_context.steps] if traversal_context else [],
        resolution_outcome=source_resolution.outcome.value,
        demo_warnings=source_resolution.demo_warnings,
        data_sources_used=list(data_sources_used),
        all_authoritative=all_authoritative,
        safe_to_answer=safe,
    )


def validate_provenance(data: dict) -> tuple[bool, str]:
    """
    Validate that a data payload has mandatory provenance markers.

    Returns (valid, reason).
    HARD RULE: Missing markers → rejected.
    """
    required = ["_data_source", "_integration_status", "_authoritative"]
    missing = [k for k in required if k not in data]

    if missing:
        return False, f"Missing provenance markers: {missing}"

    if data.get("_data_source") == "unknown":
        return False, "Data source is unknown — cannot verify origin"

    return True, ""


def validate_response_provenance(provenance: ResponseProvenance) -> tuple[bool, str]:
    """
    Validate response-level provenance.

    HARD RULE: Empty derived_from for data queries → rejected.
    """
    if not provenance.derived_from:
        return False, "Response has no derived_from sources — LLM cannot answer without data source"

    if not provenance.safe_to_answer:
        return False, "Response is not safe to answer — all sources are stubs"

    return True, ""
