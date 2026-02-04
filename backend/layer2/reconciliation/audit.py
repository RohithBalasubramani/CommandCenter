"""
Audit — Observability and Logging.

All reconciliation decisions are logged to an append-only audit sink
with structured JSON for compliance and debugging.

Supports:
- File-based audit log
- Structured JSON format
- Deterministic event IDs
- Full provenance chain
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO
import threading

from layer2.reconciliation.types import (
    ReconcileEvent,
    DecisionType,
    MismatchClass,
    Provenance,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT SINK INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class AuditSink:
    """Abstract interface for audit sinks."""

    def write_event(self, event: ReconcileEvent) -> None:
        """Write an audit event."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the sink."""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# FILE-BASED AUDIT SINK
# ═══════════════════════════════════════════════════════════════════════════════

class FileAuditSink(AuditSink):
    """
    Append-only file-based audit sink.

    Writes one JSON object per line (JSON Lines format).
    Thread-safe for concurrent writes.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file: Optional[TextIO] = None

    def _get_file(self) -> TextIO:
        if self._file is None or self._file.closed:
            self._file = open(self.path, "a", encoding="utf-8")
        return self._file

    def write_event(self, event: ReconcileEvent) -> None:
        """Write event as single JSON line."""
        with self._lock:
            try:
                f = self._get_file()
                json_line = json.dumps(event.to_dict(), separators=(",", ":"))
                f.write(json_line + "\n")
                f.flush()
            except Exception as e:
                logger.error(f"Failed to write audit event: {e}")

    def close(self) -> None:
        with self._lock:
            if self._file and not self._file.closed:
                self._file.close()
                self._file = None


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY AUDIT SINK (for testing)
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryAuditSink(AuditSink):
    """In-memory audit sink for testing."""

    def __init__(self):
        self.events: list[ReconcileEvent] = []
        self._lock = threading.Lock()

    def write_event(self, event: ReconcileEvent) -> None:
        with self._lock:
            self.events.append(event)

    def clear(self) -> None:
        with self._lock:
            self.events.clear()

    def get_events(self) -> list[ReconcileEvent]:
        with self._lock:
            return list(self.events)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING AUDIT SINK
# ═══════════════════════════════════════════════════════════════════════════════

class LoggingAuditSink(AuditSink):
    """Audit sink that writes to Python logging."""

    def __init__(self, logger_name: str = "reconciliation.audit"):
        self.logger = logging.getLogger(logger_name)

    def write_event(self, event: ReconcileEvent) -> None:
        log_level = logging.INFO if event.success else logging.WARNING

        self.logger.log(
            log_level,
            "AUDIT: %s %s %s (attempts=%d, success=%s)",
            event.event_id,
            event.decision.name,
            event.scenario,
            event.attempts,
            event.success,
        )

        # Log provenance at DEBUG level
        for p in event.provenance:
            self.logger.debug(
                "  TRANSFORM: %s: %s -> %s",
                p.rule_id,
                p.original_value_snippet,
                p.transformed_value_snippet,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL AUDIT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class AuditManager:
    """
    Manages audit sinks and event routing.

    Supports multiple sinks (e.g., file + logging).
    """

    def __init__(self):
        self.sinks: list[AuditSink] = []
        self._lock = threading.Lock()

    def add_sink(self, sink: AuditSink) -> None:
        """Add an audit sink."""
        with self._lock:
            self.sinks.append(sink)

    def remove_sink(self, sink: AuditSink) -> None:
        """Remove an audit sink."""
        with self._lock:
            if sink in self.sinks:
                self.sinks.remove(sink)

    def write_event(self, event: ReconcileEvent) -> None:
        """Write event to all sinks."""
        with self._lock:
            for sink in self.sinks:
                try:
                    sink.write_event(event)
                except Exception as e:
                    logger.error(f"Audit sink {sink} failed: {e}")

    def close(self) -> None:
        """Close all sinks."""
        with self._lock:
            for sink in self.sinks:
                try:
                    sink.close()
                except Exception as e:
                    logger.error(f"Failed to close audit sink: {e}")


# Global audit manager
_audit_manager: Optional[AuditManager] = None


def get_audit_manager() -> AuditManager:
    """Get the global audit manager."""
    global _audit_manager
    if _audit_manager is None:
        _audit_manager = AuditManager()
        # Add default logging sink
        _audit_manager.add_sink(LoggingAuditSink())
    return _audit_manager


def configure_audit(
    file_path: Optional[str] = None,
    enable_logging: bool = True,
) -> AuditManager:
    """
    Configure audit sinks.

    Args:
        file_path: Path to audit log file (optional)
        enable_logging: Enable logging sink

    Returns:
        Configured AuditManager
    """
    global _audit_manager
    _audit_manager = AuditManager()

    if enable_logging:
        _audit_manager.add_sink(LoggingAuditSink())

    if file_path:
        _audit_manager.add_sink(FileAuditSink(file_path))

    return _audit_manager


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def audit_event(
    scenario: str,
    decision: DecisionType,
    mismatch_class: MismatchClass,
    input_data: dict,
    output_data: Optional[dict] = None,
    provenance: Optional[list[Provenance]] = None,
    attempts: int = 1,
    success: bool = True,
    error_message: Optional[str] = None,
    escalation_reason: Optional[str] = None,
) -> ReconcileEvent:
    """
    Create and log an audit event.

    Returns the created event for reference.
    """
    event = ReconcileEvent.create(
        scenario=scenario,
        decision=decision,
        mismatch_class=mismatch_class,
        input_data=input_data,
        output_data=output_data,
        provenance=provenance,
        attempts=attempts,
        success=success,
        error_message=error_message,
        escalation_reason=escalation_reason,
    )

    get_audit_manager().write_event(event)
    return event


def audit_transform(
    scenario: str,
    input_data: dict,
    output_data: dict,
    provenance: list[Provenance],
) -> ReconcileEvent:
    """Audit a successful transformation."""
    return audit_event(
        scenario=scenario,
        decision=DecisionType.TRANSFORM,
        mismatch_class=MismatchClass.REPRESENTATIONAL_EQUIVALENCE,
        input_data=input_data,
        output_data=output_data,
        provenance=provenance,
        success=True,
    )


def audit_refuse(
    scenario: str,
    input_data: dict,
    reason: str,
    attempts: int = 1,
) -> ReconcileEvent:
    """Audit a refusal."""
    return audit_event(
        scenario=scenario,
        decision=DecisionType.REFUSE,
        mismatch_class=MismatchClass.UNKNOWN_AMBIGUOUS,
        input_data=input_data,
        attempts=attempts,
        success=False,
        error_message=reason,
    )


def audit_escalate(
    scenario: str,
    input_data: dict,
    reason: str,
    attempts: int = 1,
) -> ReconcileEvent:
    """Audit an escalation."""
    return audit_event(
        scenario=scenario,
        decision=DecisionType.ESCALATE,
        mismatch_class=MismatchClass.SEMANTIC_DIFFERENCE,
        input_data=input_data,
        attempts=attempts,
        success=False,
        escalation_reason=reason,
    )
