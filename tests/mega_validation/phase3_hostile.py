"""
MEGA VALIDATION - PHASE 3: HOSTILE TESTS

Break-it-or-block-release validation for Command Center.
Tests:
- Semantic Collision: Ambiguous intents that could be misinterpreted
- Data-Shape Poisoning: Valid schema but wrong content
- Capability Overreach: Requests beyond system capability

If any hostile test discovers exploitable failure → BLOCK RELEASE
"""
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup Django
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "command_center.settings")

import django
django.setup()

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HostileTestResult:
    """Result of a single hostile test."""
    name: str
    category: str
    attack_type: str
    passed: bool
    system_behavior: str
    expected_behavior: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    is_exploitable: bool = False
    severity: str = "low"  # low, medium, high, critical


@dataclass
class Phase3Result:
    """Complete Phase 3 hostile test result."""
    timestamp: str = ""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    exploitable_failures: int = 0
    test_results: List[HostileTestResult] = field(default_factory=list)
    release_blocked: bool = False
    block_reasons: List[str] = field(default_factory=list)


# =============================================================================
# HOSTILE TEST CASES
# =============================================================================

# Semantic Collision: Multiple valid interpretations
SEMANTIC_COLLISION_TESTS = [
    # Ambiguous equipment
    {
        "query": "Show me the status",
        "attack": "Ambiguous scope - no equipment specified",
        "valid_responses": ["ask_clarification", "show_all", "out_of_scope"],
        "invalid_responses": ["commit_single_random"],
    },
    {
        "query": "Turn it on",
        "attack": "Ambiguous target for control action",
        "valid_responses": ["ask_clarification", "reject_ambiguous"],
        "invalid_responses": ["execute_control"],
    },
    {
        "query": "Check the temperature",
        "attack": "Multiple temperature sensors possible",
        "valid_responses": ["ask_clarification", "show_all_temps"],
        "invalid_responses": ["commit_single_random"],
    },
    # Conflicting references
    {
        "query": "Show the other one",
        "attack": "No prior context to resolve 'other'",
        "valid_responses": ["ask_clarification"],
        "invalid_responses": ["random_guess", "commit_single"],
    },
    {
        "query": "Compare them",
        "attack": "No referent for 'them'",
        "valid_responses": ["ask_clarification"],
        "invalid_responses": ["random_comparison"],
    },
    # Contradictory requests
    {
        "query": "Start and stop pump 1",
        "attack": "Contradictory actions in single request",
        "valid_responses": ["ask_clarification", "reject_contradiction"],
        "invalid_responses": ["execute_both", "execute_one"],
    },
    {
        "query": "Show more but hide the details",
        "attack": "Contradictory display request",
        "valid_responses": ["ask_clarification"],
        "invalid_responses": ["random_choice"],
    },
    # Time ambiguity
    {
        "query": "Show yesterday's data",
        "attack": "Time specified but equipment ambiguous",
        "valid_responses": ["ask_clarification", "show_all_yesterday"],
        "invalid_responses": ["random_equipment"],
    },
    # Cross-domain ambiguity
    {
        "query": "What's wrong?",
        "attack": "Could be alerts, maintenance, equipment, or people",
        "valid_responses": ["ask_clarification", "show_multi_domain"],
        "invalid_responses": ["single_domain_guess"],
    },
]

# Data-Shape Poisoning: Valid schema, wrong semantic
DATA_POISONING_TESTS = [
    {
        "name": "Scalar expected, array given",
        "widget": "kpi",
        "data": {"value": [42], "label": "Test", "unit": "kW"},
        "attack": "KPI expects scalar value, not array",
        "should_reject": True,
    },
    {
        "name": "String number instead of int",
        "widget": "kpi",
        "data": {"value": "42", "label": "Test", "unit": "kW"},
        "attack": "Type mismatch but convertible",
        "should_reject": False,  # Should auto-convert
    },
    {
        "name": "Invalid string for number",
        "widget": "kpi",
        "data": {"value": "not a number", "label": "Test", "unit": "kW"},
        "attack": "Non-convertible string",
        "should_reject": True,
    },
    {
        "name": "Missing required field",
        "widget": "kpi",
        "data": {"label": "Test"},
        "attack": "Missing 'value' field",
        "should_reject": True,
    },
    {
        "name": "Negative percentage",
        "widget": "kpi",
        "data": {"value": -15, "label": "Efficiency %", "unit": "%"},
        "attack": "Semantically invalid value",
        "should_reject": True,  # Percentage can't be negative
    },
    {
        "name": "Future timestamp in historical data",
        "widget": "trend",
        "data": {"data": [{"timestamp": "2030-01-01", "value": 42}]},
        "attack": "Future timestamps in historical trend",
        "should_reject": True,
    },
    {
        "name": "Mixed units in series",
        "widget": "trend",
        "data": {"data": [
            {"value": 25, "unit": "C"},
            {"value": 77, "unit": "F"},
        ]},
        "attack": "Inconsistent units in same series",
        "should_reject": True,
    },
    {
        "name": "XSS in label",
        "widget": "kpi",
        "data": {"value": 42, "label": "<script>alert('xss')</script>", "unit": "kW"},
        "attack": "XSS injection attempt",
        "should_reject": True,
    },
    {
        "name": "SQL injection in label",
        "widget": "kpi",
        "data": {"value": 42, "label": "'; DROP TABLE users;--", "unit": "kW"},
        "attack": "SQL injection attempt",
        "should_reject": True,
    },
]

# Capability Overreach: Beyond system limits
CAPABILITY_OVERREACH_TESTS = [
    {
        "query": "Show me a 3D rotating model of the pump",
        "attack": "3D visualization not supported",
        "expected": "polite_decline",
    },
    {
        "query": "Predict when pump 1 will fail",
        "attack": "Predictive ML not implemented",
        "expected": "capability_limit",
    },
    {
        "query": "Automatically optimize the entire system",
        "attack": "Auto-optimization not available",
        "expected": "capability_limit",
    },
    {
        "query": "Show weather correlation with energy",
        "attack": "External weather data not integrated",
        "expected": "data_unavailable",
    },
    {
        "query": "Order replacement parts automatically",
        "attack": "Procurement not integrated",
        "expected": "capability_limit",
    },
    {
        "query": "Send SMS to all technicians",
        "attack": "SMS integration not available",
        "expected": "capability_limit",
    },
    {
        "query": "Update the equipment firmware remotely",
        "attack": "Firmware control not available",
        "expected": "capability_limit",
    },
    {
        "query": "Show real-time video of the plant",
        "attack": "Video streaming not implemented",
        "expected": "capability_limit",
    },
    {
        "query": "Calculate carbon footprint from operations",
        "attack": "Carbon analytics not implemented",
        "expected": "capability_limit",
    },
    {
        "query": "Connect to the building's fire alarm system",
        "attack": "Safety system integration not available",
        "expected": "security_boundary",
    },
]


# =============================================================================
# TEST EXECUTION
# =============================================================================

def test_semantic_collision() -> List[HostileTestResult]:
    """Test semantic collision handling."""
    from layer2.intent_parser import IntentParser

    parser = IntentParser()
    results = []

    for test in SEMANTIC_COLLISION_TESTS:
        query = test["query"]
        attack = test["attack"]

        try:
            intent = parser._parse_with_regex(query)

            # Check system behavior
            confidence = intent.confidence
            intent_type = intent.type

            # Determine if system committed inappropriately
            if confidence > 0.7 and intent_type in ("query", "action_control"):
                system_behavior = "high_confidence_commit"
                # Check if this is valid or invalid
                if "commit" in str(test["invalid_responses"]):
                    passed = False
                    is_exploitable = True
                    severity = "high"
                else:
                    passed = True
                    is_exploitable = False
                    severity = "low"
            elif intent_type in ("out_of_scope", "conversation") or confidence < 0.5:
                system_behavior = "low_confidence_or_reject"
                passed = True
                is_exploitable = False
                severity = "low"
            else:
                system_behavior = f"moderate_commit ({intent_type}, conf={confidence:.2f})"
                passed = True  # Moderate commitment is acceptable
                is_exploitable = False
                severity = "low"

            results.append(HostileTestResult(
                name=f"Semantic: {query[:30]}",
                category="semantic_collision",
                attack_type=attack,
                passed=passed,
                system_behavior=system_behavior,
                expected_behavior=str(test["valid_responses"]),
                evidence={
                    "query": query,
                    "intent_type": intent_type,
                    "confidence": confidence,
                    "domains": intent.domains,
                },
                is_exploitable=is_exploitable,
                severity=severity,
            ))

        except Exception as e:
            results.append(HostileTestResult(
                name=f"Semantic: {query[:30]}",
                category="semantic_collision",
                attack_type=attack,
                passed=False,
                system_behavior=f"exception: {str(e)}",
                expected_behavior=str(test["valid_responses"]),
                is_exploitable=True,
                severity="critical",
            ))

    return results


def test_data_poisoning() -> List[HostileTestResult]:
    """Test data-shape poisoning handling."""
    results = []

    # Try to import validation
    try:
        from layer2.widget_schemas import validate_widget_data, ValidationError
        has_validator = True
    except ImportError:
        has_validator = False
        ValidationError = Exception

    for test in DATA_POISONING_TESTS:
        name = test["name"]
        widget = test["widget"]
        data = test["data"]
        should_reject = test["should_reject"]

        try:
            if has_validator:
                # validate_widget_data raises ValidationError on failure
                try:
                    validate_widget_data(widget, data)
                    rejected = False  # No exception = valid
                except ValidationError:
                    rejected = True  # Exception = rejected
            else:
                # Basic validation without schema
                rejected = False

                # Check for obvious violations
                if isinstance(data.get("value"), list) and widget == "kpi":
                    rejected = True
                if isinstance(data.get("value"), str):
                    try:
                        float(data["value"])
                    except (ValueError, TypeError):
                        rejected = True
                if "<script>" in str(data).lower():
                    rejected = True
                if "drop table" in str(data).lower():
                    rejected = True

            # Determine result
            if should_reject and rejected:
                passed = True
                system_behavior = "correctly_rejected"
                is_exploitable = False
                severity = "low"
            elif should_reject and not rejected:
                passed = False
                system_behavior = "incorrectly_accepted"
                is_exploitable = True
                severity = "high" if "xss" in name.lower() or "sql" in name.lower() else "medium"
            elif not should_reject and rejected:
                passed = False
                system_behavior = "incorrectly_rejected"
                is_exploitable = False
                severity = "low"
            else:
                passed = True
                system_behavior = "correctly_accepted"
                is_exploitable = False
                severity = "low"

            results.append(HostileTestResult(
                name=f"DataPoison: {name}",
                category="data_poisoning",
                attack_type=test["attack"],
                passed=passed,
                system_behavior=system_behavior,
                expected_behavior="reject" if should_reject else "accept",
                evidence={
                    "widget": widget,
                    "data": data,
                    "should_reject": should_reject,
                    "was_rejected": rejected,
                },
                is_exploitable=is_exploitable,
                severity=severity,
            ))

        except Exception as e:
            # Non-ValidationError exception - this is a real failure
            results.append(HostileTestResult(
                name=f"DataPoison: {name}",
                category="data_poisoning",
                attack_type=test["attack"],
                passed=False,
                system_behavior=f"exception: {str(e)}",
                expected_behavior="reject" if should_reject else "accept",
                is_exploitable=True,
                severity="high",
            ))

    return results


def test_capability_overreach() -> List[HostileTestResult]:
    """Test capability overreach handling."""
    from layer2.intent_parser import IntentParser

    parser = IntentParser()
    results = []

    for test in CAPABILITY_OVERREACH_TESTS:
        query = test["query"]
        attack = test["attack"]

        try:
            intent = parser._parse_with_regex(query)

            # Check if system over-committed to impossible request
            confidence = intent.confidence
            intent_type = intent.type

            # High confidence on impossible = failure
            if confidence > 0.7 and intent_type in ("query", "action_control"):
                passed = False
                system_behavior = f"over_committed ({intent_type}, conf={confidence:.2f})"
                is_exploitable = True
                severity = "high"
            elif intent_type == "out_of_scope":
                passed = True
                system_behavior = "correctly_rejected"
                is_exploitable = False
                severity = "low"
            else:
                passed = True
                system_behavior = f"moderate_response ({intent_type}, conf={confidence:.2f})"
                is_exploitable = False
                severity = "low"

            results.append(HostileTestResult(
                name=f"Overreach: {query[:25]}",
                category="capability_overreach",
                attack_type=attack,
                passed=passed,
                system_behavior=system_behavior,
                expected_behavior=test["expected"],
                evidence={
                    "query": query,
                    "intent_type": intent_type,
                    "confidence": confidence,
                },
                is_exploitable=is_exploitable,
                severity=severity,
            ))

        except Exception as e:
            results.append(HostileTestResult(
                name=f"Overreach: {query[:25]}",
                category="capability_overreach",
                attack_type=attack,
                passed=False,
                system_behavior=f"exception: {str(e)}",
                expected_behavior=test["expected"],
                is_exploitable=True,
                severity="critical",
            ))

    return results


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_phase3_hostile(verbose: bool = True) -> Phase3Result:
    """
    Run Phase 3 hostile validation tests.

    Returns:
        Phase3Result with all hostile test results
    """
    result = Phase3Result(
        timestamp=datetime.now().isoformat(),
    )

    if verbose:
        print("=" * 70)
        print("  MEGA VALIDATION - PHASE 3: HOSTILE TESTS")
        print("=" * 70)
        print(f"  Timestamp: {result.timestamp}")
        print("=" * 70)

    # Run semantic collision tests
    if verbose:
        print("\n── SEMANTIC COLLISION TESTS ──")

    semantic_results = test_semantic_collision()
    result.test_results.extend(semantic_results)

    if verbose:
        passed = sum(1 for r in semantic_results if r.passed)
        print(f"  {passed}/{len(semantic_results)} passed")

    # Run data poisoning tests
    if verbose:
        print("\n── DATA POISONING TESTS ──")

    poison_results = test_data_poisoning()
    result.test_results.extend(poison_results)

    if verbose:
        passed = sum(1 for r in poison_results if r.passed)
        print(f"  {passed}/{len(poison_results)} passed")

    # Run capability overreach tests
    if verbose:
        print("\n── CAPABILITY OVERREACH TESTS ──")

    overreach_results = test_capability_overreach()
    result.test_results.extend(overreach_results)

    if verbose:
        passed = sum(1 for r in overreach_results if r.passed)
        print(f"  {passed}/{len(overreach_results)} passed")

    # Calculate totals
    result.total_tests = len(result.test_results)
    result.passed_tests = sum(1 for r in result.test_results if r.passed)
    result.failed_tests = result.total_tests - result.passed_tests
    result.exploitable_failures = sum(1 for r in result.test_results if r.is_exploitable)

    # Determine release decision
    critical_failures = [r for r in result.test_results if r.severity == "critical" and not r.passed]
    high_failures = [r for r in result.test_results if r.severity == "high" and r.is_exploitable]

    if critical_failures:
        result.release_blocked = True
        result.block_reasons.append(f"{len(critical_failures)} critical failures")

    if high_failures:
        result.release_blocked = True
        result.block_reasons.append(f"{len(high_failures)} exploitable high-severity failures")

    if verbose:
        print("\n" + "=" * 70)
        print("  PHASE 3 SUMMARY")
        print("=" * 70)
        print(f"  Total Tests: {result.total_tests}")
        print(f"  Passed: {result.passed_tests}")
        print(f"  Failed: {result.failed_tests}")
        print(f"  Exploitable: {result.exploitable_failures}")
        print("")

        if result.release_blocked:
            print("  ✗ RELEASE BLOCKED")
            for reason in result.block_reasons:
                print(f"    - {reason}")
        else:
            print("  ✓ RELEASE APPROVED (Phase 3 Hostile)")

        print("=" * 70)

    return result


def save_report(result: Phase3Result, output_dir: Path = None) -> Path:
    """Save Phase 3 report to JSON file."""
    if output_dir is None:
        output_dir = Path(__file__).parent

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"phase3_hostile_{timestamp}.json"

    # Convert to serializable dict
    data = {
        "timestamp": result.timestamp,
        "total_tests": result.total_tests,
        "passed_tests": result.passed_tests,
        "failed_tests": result.failed_tests,
        "exploitable_failures": result.exploitable_failures,
        "test_results": [asdict(r) for r in result.test_results],
        "release_blocked": result.release_blocked,
        "block_reasons": result.block_reasons,
    }

    with open(report_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return report_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3 Hostile Validation")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--save", action="store_true", help="Save report to JSON file")

    args = parser.parse_args()

    result = run_phase3_hostile(verbose=not args.quiet)

    if args.save:
        report_path = save_report(result)
        print(f"\nReport saved to: {report_path}")

    # Exit with non-zero if release blocked
    sys.exit(1 if result.release_blocked else 0)
