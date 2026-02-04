"""
Command Center AI Implementation Audit — Test Harness

Engineering-grade validation of:
- Intent parsing accuracy and determinism
- Widget selection correctness
- Layout schema compliance
- Latency measurement
- Widget registry consistency

Run: python manage.py shell < layer2/audit_tests.py
Or:  pytest layer2/audit_tests.py -v
"""

import json
import time
import statistics
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

# ============================================================
# Test Configuration
# ============================================================

DETERMINISM_RUNS = 5  # Number of runs per test case for determinism check

# Ground truth test cases for intent parsing
INTENT_TEST_CASES = [
    # (input, expected_type, expected_domains, expected_primary_char)
    ("What's the status of the pumps?", "query", ["industrial"], "health_status"),
    ("Show me transformer temperatures", "query", ["industrial"], "trend"),
    ("Compare pump 1 vs pump 2", "query", ["industrial"], "comparison"),
    ("What alerts are active?", "query", ["alerts"], "alerts"),
    ("Show energy consumption trend", "query", ["industrial"], "energy"),
    ("Who is on shift today?", "query", ["people"], "people"),
    ("What's the inventory level?", "query", ["supply"], "supply_chain"),
    ("Hello", "greeting", [], None),
    ("Thank you", "conversation", [], None),
    ("What's the weather like?", "out_of_scope", [], None),
    ("Show me the energy flow from grid to loads", "query", ["industrial"], "flow_sankey"),
    ("What's the cumulative energy today?", "query", ["industrial"], "cumulative"),
    ("Show power quality metrics", "query", ["industrial"], "power_quality"),
    ("What maintenance is pending?", "query", ["industrial"], "maintenance"),
    ("Show top energy consumers", "query", ["industrial"], "top_consumers"),
    ("Tell me about UPS battery status", "query", ["industrial"], "ups_dg"),
]

# Valid widget scenarios (canonical list)
VALID_WIDGET_SCENARIOS = {
    "kpi", "alerts", "comparison", "trend", "trend-multi-line",
    "trends-cumulative", "distribution", "composition", "category-bar",
    "timeline", "flow-sankey", "matrix-heatmap", "eventlogstream",
    "edgedevicepanel", "chatstream", "peopleview", "peoplehexgrid",
    "peoplenetwork", "supplychainglobe",
}

# Banned scenarios that should never appear in layout
BANNED_SCENARIOS = {"helpview", "pulseview"}

# Widget data shape requirements
WIDGET_REQUIRED_FIELDS = {
    "kpi": ["label", "value", "unit"],
    "alerts": ["id", "title", "message", "severity", "source"],
    "comparison": ["label", "unit", "labelA", "valueA", "labelB", "valueB"],
    "trend": ["label", "unit", "timeSeries"],
    "trend-multi-line": ["label", "unit", "series"],
    "trends-cumulative": ["config", "data"],
    "distribution": ["total", "unit", "series"],
    "flow-sankey": ["label", "nodes", "links"],
    "matrix-heatmap": ["label", "dataset"],
    "eventlogstream": ["events"],
    "edgedevicepanel": ["device"],
}


# ============================================================
# Data Classes for Results
# ============================================================

@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    stage: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    passed: bool
    failure_type: Optional[str] = None  # Classification from FAILURE_TYPES
    details: str = ""
    latencies: list = field(default_factory=list)


@dataclass
class AuditReport:
    """Complete audit report."""
    timestamp: str
    results: list = field(default_factory=list)
    latency_summary: dict = field(default_factory=dict)
    accuracy_matrix: dict = field(default_factory=dict)
    determinism_results: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)


FAILURE_TYPES = [
    "Intent Misparse",
    "RAG Retrieval Error",
    "LLM Hallucination",
    "Layout Contract Violation",
    "Widget/Data Mismatch",
    "Latency Regression",
    "Voice Pipeline Degradation",
]


# ============================================================
# Test Functions
# ============================================================

def test_widget_registry_consistency():
    """Validate frontend and backend widget registries match."""
    results = []

    try:
        from layer2.widget_catalog import VALID_SCENARIOS as backend_scenarios
        from layer2.widget_schemas import WIDGET_SCHEMAS

        # Check backend catalog vs schemas
        catalog_scenarios = backend_scenarios
        schema_scenarios = set(WIDGET_SCHEMAS.keys())

        # Missing in schemas
        missing_schemas = catalog_scenarios - schema_scenarios
        if missing_schemas:
            results.append(TestResult(
                test_name="widget_schema_coverage",
                passed=False,
                failure_type="Layout Contract Violation",
                details=f"Widgets in catalog but missing schemas: {missing_schemas}",
            ))
        else:
            results.append(TestResult(
                test_name="widget_schema_coverage",
                passed=True,
                details=f"All {len(catalog_scenarios)} widgets have schemas",
            ))

        # Check against canonical list
        extra_backend = catalog_scenarios - VALID_WIDGET_SCENARIOS
        missing_backend = VALID_WIDGET_SCENARIOS - catalog_scenarios

        if extra_backend or missing_backend:
            results.append(TestResult(
                test_name="widget_registry_canonical",
                passed=False,
                failure_type="Layout Contract Violation",
                details=f"Extra: {extra_backend}, Missing: {missing_backend}",
            ))
        else:
            results.append(TestResult(
                test_name="widget_registry_canonical",
                passed=True,
                details=f"Registry matches canonical {len(VALID_WIDGET_SCENARIOS)} widgets",
            ))

    except ImportError as e:
        results.append(TestResult(
            test_name="widget_registry_import",
            passed=False,
            failure_type="Layout Contract Violation",
            details=f"Import error: {e}",
        ))

    return results


def test_intent_parser_accuracy():
    """Test intent parser against ground truth cases."""
    results = []

    try:
        from layer2.intent_parser import IntentParser
        parser = IntentParser()

        correct = 0
        total = len(INTENT_TEST_CASES)
        errors = []

        for transcript, expected_type, expected_domains, expected_char in INTENT_TEST_CASES:
            parsed = parser.parse(transcript)

            type_match = parsed.type == expected_type
            domain_match = set(parsed.domains) == set(expected_domains) if expected_domains else not parsed.domains
            char_match = parsed.primary_characteristic == expected_char if expected_char else True

            if type_match and domain_match and char_match:
                correct += 1
            else:
                errors.append({
                    "input": transcript,
                    "expected": {"type": expected_type, "domains": expected_domains, "char": expected_char},
                    "got": {"type": parsed.type, "domains": parsed.domains, "char": parsed.primary_characteristic},
                })

        accuracy = correct / total * 100
        passed = accuracy >= 80  # 80% threshold

        results.append(TestResult(
            test_name="intent_parser_accuracy",
            passed=passed,
            failure_type="Intent Misparse" if not passed else None,
            details=f"Accuracy: {accuracy:.1f}% ({correct}/{total}). Errors: {len(errors)}",
        ))

        if errors:
            for err in errors[:3]:  # First 3 errors
                results.append(TestResult(
                    test_name=f"intent_case_{err['input'][:20]}",
                    passed=False,
                    failure_type="Intent Misparse",
                    details=f"Expected {err['expected']}, got {err['got']}",
                ))

    except Exception as e:
        results.append(TestResult(
            test_name="intent_parser_accuracy",
            passed=False,
            failure_type="Intent Misparse",
            details=f"Exception: {e}",
        ))

    return results


def test_intent_parser_determinism():
    """Test that same input produces same output across multiple runs."""
    results = []

    try:
        from layer2.intent_parser import IntentParser

        test_inputs = [
            "What's the status of pump 1?",
            "Show me energy consumption",
            "Compare transformer 1 vs transformer 2",
        ]

        for transcript in test_inputs:
            outputs = []
            for _ in range(DETERMINISM_RUNS):
                parser = IntentParser()  # Fresh instance each time
                parsed = parser.parse(transcript)
                output_key = (parsed.type, tuple(sorted(parsed.domains)), parsed.primary_characteristic)
                outputs.append(output_key)

            unique_outputs = set(outputs)
            is_deterministic = len(unique_outputs) == 1

            results.append(TestResult(
                test_name=f"intent_determinism_{transcript[:20]}",
                passed=is_deterministic,
                failure_type="Intent Misparse" if not is_deterministic else None,
                details=f"{DETERMINISM_RUNS} runs, {len(unique_outputs)} unique outputs: {unique_outputs}",
            ))

    except Exception as e:
        results.append(TestResult(
            test_name="intent_determinism",
            passed=False,
            failure_type="Intent Misparse",
            details=f"Exception: {e}",
        ))

    return results


def test_widget_selector_banned_scenarios():
    """Verify banned scenarios are never selected."""
    results = []

    try:
        from layer2.widget_selector import WidgetSelector
        from layer2.intent_parser import IntentParser, ParsedIntent

        selector = WidgetSelector()

        # Test queries that might trigger banned widgets
        test_queries = [
            "Help me understand the system",
            "Show system pulse",
            "What can you do?",
        ]

        for query in test_queries:
            parser = IntentParser()
            parsed = parser.parse(query)

            # Force query type for testing
            if parsed.type in ("out_of_scope", "conversation", "greeting"):
                parsed = ParsedIntent(
                    type="query",
                    domains=["industrial"],
                    entities={},
                    raw_text=query,
                    parse_method="test",
                )

            plan = selector.select(parsed)

            banned_found = [w.scenario for w in plan.widgets if w.scenario in BANNED_SCENARIOS]

            results.append(TestResult(
                test_name=f"banned_widget_{query[:20]}",
                passed=len(banned_found) == 0,
                failure_type="LLM Hallucination" if banned_found else None,
                details=f"Banned widgets found: {banned_found}" if banned_found else "No banned widgets",
            ))

    except Exception as e:
        results.append(TestResult(
            test_name="banned_widget_check",
            passed=False,
            failure_type="LLM Hallucination",
            details=f"Exception: {e}",
        ))

    return results


def test_layout_schema_compliance():
    """Test that layout_json adheres to schema requirements."""
    results = []

    try:
        from layer2.orchestrator import Layer2Orchestrator

        test_queries = [
            "What's the status of the pumps?",
            "Show me alerts",
            "Compare energy consumption",
        ]

        for query in test_queries:
            orchestrator = Layer2Orchestrator()
            response = orchestrator.process_transcript(query, user_id="test_user")

            layout = response.layout_json
            if layout is None:
                results.append(TestResult(
                    test_name=f"layout_schema_{query[:20]}",
                    passed=True,
                    details="No layout generated (non-query intent)",
                ))
                continue

            # Check required fields
            has_heading = "heading" in layout
            has_widgets = "widgets" in layout and isinstance(layout["widgets"], list)

            if not has_heading or not has_widgets:
                results.append(TestResult(
                    test_name=f"layout_schema_{query[:20]}",
                    passed=False,
                    failure_type="Layout Contract Violation",
                    details=f"Missing heading={not has_heading}, widgets={not has_widgets}",
                ))
                continue

            # Validate each widget
            widget_errors = []
            for i, w in enumerate(layout["widgets"]):
                scenario = w.get("scenario", "")

                # Check scenario is valid
                if scenario not in VALID_WIDGET_SCENARIOS:
                    widget_errors.append(f"Widget {i}: unknown scenario '{scenario}'")

                # Check size is valid
                size = w.get("size", "")
                if size not in {"compact", "normal", "expanded", "hero"}:
                    widget_errors.append(f"Widget {i}: invalid size '{size}'")

                # Check required data fields
                data_override = w.get("data_override", {})
                demo_data = data_override.get("demoData", data_override)

                required = WIDGET_REQUIRED_FIELDS.get(scenario, [])
                missing = [f for f in required if f not in demo_data]
                if missing and scenario in WIDGET_REQUIRED_FIELDS:
                    widget_errors.append(f"Widget {i} ({scenario}): missing fields {missing}")

            results.append(TestResult(
                test_name=f"layout_schema_{query[:20]}",
                passed=len(widget_errors) == 0,
                failure_type="Widget/Data Mismatch" if widget_errors else None,
                details=f"{len(layout['widgets'])} widgets, errors: {widget_errors[:3]}" if widget_errors else f"{len(layout['widgets'])} widgets valid",
            ))

    except Exception as e:
        results.append(TestResult(
            test_name="layout_schema_compliance",
            passed=False,
            failure_type="Layout Contract Violation",
            details=f"Exception: {e}",
        ))

    return results


def test_latency_breakdown():
    """Measure latency at each pipeline stage."""
    results = []
    latencies = defaultdict(list)

    try:
        from layer2.orchestrator import Layer2Orchestrator
        from layer2.intent_parser import IntentParser
        from layer2.widget_selector import WidgetSelector
        from layer2.rag_pipeline import get_rag_pipeline

        test_query = "What's the status of the pumps?"

        for run in range(3):  # 3 runs for variance
            # Stage 1: Intent parsing
            t0 = time.time()
            parser = IntentParser()
            parsed = parser.parse(test_query)
            t_intent = (time.time() - t0) * 1000
            latencies["intent_parsing"].append(t_intent)

            # Stage 2: RAG query (if available)
            t0 = time.time()
            try:
                pipeline = get_rag_pipeline()
                rag_result = pipeline.query(test_query, n_results=5)
                t_rag = (time.time() - t0) * 1000
                latencies["rag_query"].append(t_rag)
            except Exception:
                latencies["rag_query"].append(-1)  # Not available

            # Stage 3: Widget selection
            t0 = time.time()
            selector = WidgetSelector()
            plan = selector.select(parsed)
            t_widget = (time.time() - t0) * 1000
            latencies["widget_selection"].append(t_widget)

            # Stage 4: Full orchestrator
            t0 = time.time()
            orchestrator = Layer2Orchestrator()
            response = orchestrator.process_transcript(test_query, user_id="test_user")
            t_total = (time.time() - t0) * 1000
            latencies["total_orchestrator"].append(t_total)

            # Reported processing time
            latencies["reported_processing"].append(response.processing_time_ms)

        # Calculate stats and check budgets
        LATENCY_BUDGETS = {
            "intent_parsing": 500,    # 500ms max
            "rag_query": 2000,        # 2s max
            "widget_selection": 3000, # 3s max (includes LLM)
            "total_orchestrator": 8000, # 8s max
        }

        for stage, times in latencies.items():
            valid_times = [t for t in times if t >= 0]
            if not valid_times:
                continue

            avg = statistics.mean(valid_times)
            std = statistics.stdev(valid_times) if len(valid_times) > 1 else 0
            budget = LATENCY_BUDGETS.get(stage, float("inf"))

            passed = avg <= budget
            results.append(TestResult(
                test_name=f"latency_{stage}",
                passed=passed,
                failure_type="Latency Regression" if not passed else None,
                details=f"Avg: {avg:.0f}ms (±{std:.0f}), Budget: {budget}ms",
                latencies=[LatencyMeasurement(stage=stage, duration_ms=t) for t in valid_times],
            ))

    except Exception as e:
        results.append(TestResult(
            test_name="latency_breakdown",
            passed=False,
            failure_type="Latency Regression",
            details=f"Exception: {e}",
        ))

    return results


def test_widget_data_shape_matching():
    """Validate widget data shapes match expected schemas."""
    results = []

    try:
        from layer2.widget_schemas import WIDGET_SCHEMAS
        from layer2.data_collector import SchemaDataCollector
        from layer2.widget_selector import WidgetPlanItem

        collector = SchemaDataCollector()

        # Test each widget type
        test_widgets = [
            WidgetPlanItem(scenario="kpi", data_request={"query": "pump status", "metric": "health"}),
            WidgetPlanItem(scenario="alerts", data_request={"query": "active alerts"}),
            WidgetPlanItem(scenario="trend", data_request={"query": "energy consumption", "metric": "power_kw"}),
        ]

        for widget in test_widgets:
            schema = WIDGET_SCHEMAS.get(widget.scenario, {})
            required = schema.get("required", [])

            try:
                data = collector._collect_one(widget, "test query")
                demo_data = data.get("demoData", data.get("config", data))

                # Check for required fields (handle nested structures)
                if isinstance(demo_data, dict):
                    missing = [f for f in required if f not in demo_data]
                else:
                    missing = required  # Wrong shape entirely

                results.append(TestResult(
                    test_name=f"data_shape_{widget.scenario}",
                    passed=len(missing) == 0,
                    failure_type="Widget/Data Mismatch" if missing else None,
                    details=f"Missing fields: {missing}" if missing else "All required fields present",
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=f"data_shape_{widget.scenario}",
                    passed=False,
                    failure_type="Widget/Data Mismatch",
                    details=f"Collection error: {e}",
                ))

    except Exception as e:
        results.append(TestResult(
            test_name="data_shape_validation",
            passed=False,
            failure_type="Widget/Data Mismatch",
            details=f"Exception: {e}",
        ))

    return results


# ============================================================
# Main Audit Runner
# ============================================================

def run_full_audit() -> AuditReport:
    """Execute all audit tests and generate report."""
    from datetime import datetime

    report = AuditReport(timestamp=datetime.now().isoformat())

    print("=" * 70)
    print("COMMAND CENTER AI IMPLEMENTATION AUDIT")
    print("=" * 70)
    print()

    # Run all tests
    test_suites = [
        ("Widget Registry Consistency", test_widget_registry_consistency),
        ("Intent Parser Accuracy", test_intent_parser_accuracy),
        ("Intent Parser Determinism", test_intent_parser_determinism),
        ("Banned Widget Enforcement", test_widget_selector_banned_scenarios),
        ("Layout Schema Compliance", test_layout_schema_compliance),
        ("Widget Data Shape Matching", test_widget_data_shape_matching),
        ("Latency Breakdown", test_latency_breakdown),
    ]

    for suite_name, test_fn in test_suites:
        print(f"\n▶ {suite_name}")
        print("-" * 50)
        try:
            results = test_fn()
            report.results.extend(results)

            for r in results:
                status = "✅ PASS" if r.passed else "❌ FAIL"
                print(f"  {status} {r.test_name}")
                if not r.passed:
                    print(f"       Type: {r.failure_type}")
                    print(f"       Details: {r.details[:100]}")
                    report.failures.append({
                        "test": r.test_name,
                        "type": r.failure_type,
                        "details": r.details,
                    })
        except Exception as e:
            print(f"  ❌ SUITE ERROR: {e}")
            report.failures.append({
                "test": suite_name,
                "type": "Suite Error",
                "details": str(e),
            })

    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    total = len(report.results)
    passed = sum(1 for r in report.results if r.passed)
    failed = total - passed

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"Failed: {failed}")

    if report.failures:
        print("\n▶ FAILURES BY TYPE:")
        by_type = defaultdict(list)
        for f in report.failures:
            by_type[f["type"]].append(f["test"])
        for ftype, tests in by_type.items():
            print(f"  {ftype}: {len(tests)} failures")
            for t in tests[:3]:
                print(f"    - {t}")

    # Latency summary
    latency_results = [r for r in report.results if r.latencies]
    if latency_results:
        print("\n▶ LATENCY SUMMARY:")
        for r in latency_results:
            times = [l.duration_ms for l in r.latencies]
            avg = statistics.mean(times)
            print(f"  {r.test_name.replace('latency_', '')}: {avg:.0f}ms avg")

    print("\n" + "=" * 70)

    return report


if __name__ == "__main__":
    report = run_full_audit()

    # Output JSON report
    import json
    with open("/tmp/audit_report.json", "w") as f:
        json.dump({
            "timestamp": report.timestamp,
            "total_tests": len(report.results),
            "passed": sum(1 for r in report.results if r.passed),
            "failed": sum(1 for r in report.results if not r.passed),
            "failures": report.failures,
        }, f, indent=2)
    print(f"\nJSON report written to /tmp/audit_report.json")
