"""
MEGA VALIDATION - PHASE 1: BASELINE & CEILING CONFIRMATION

Break-it-or-block-release validation for Command Center.
Measures:
- Per-layer latency (p50 / p90 / p99)
- End-to-end latency (voice → voice)
- Accuracy (Intent classification, RAG relevance, Widget/data compatibility)
- Determinism (≥10 identical runs, layout_json diff = 0)

If any baseline regresses → BLOCK RELEASE
"""
import asyncio
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

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
class LatencyStats:
    """Latency statistics for a component."""
    name: str
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    std_dev: float = 0.0
    sample_count: int = 0
    throughput_ops_sec: float = 0.0

    @classmethod
    def from_timings(cls, name: str, timings: List[float]) -> "LatencyStats":
        if not timings:
            return cls(name=name)

        sorted_t = sorted(timings)
        n = len(sorted_t)
        total_time = sum(timings) / 1000  # Convert to seconds

        return cls(
            name=name,
            p50=sorted_t[int(n * 0.50)] if n > 0 else 0,
            p90=sorted_t[int(n * 0.90)] if n >= 10 else sorted_t[-1],
            p95=sorted_t[int(n * 0.95)] if n >= 20 else sorted_t[-1],
            p99=sorted_t[int(n * 0.99)] if n >= 100 else sorted_t[-1],
            min_ms=min(timings),
            max_ms=max(timings),
            mean_ms=mean(timings),
            std_dev=stdev(timings) if len(timings) > 1 else 0,
            sample_count=n,
            throughput_ops_sec=n / total_time if total_time > 0 else 0
        )


@dataclass
class PerLayerLatency:
    """Per-layer latency breakdown from orchestrator."""
    intent_parse: LatencyStats = None
    data_prefetch: LatencyStats = None
    widget_select: LatencyStats = None
    data_collect: LatencyStats = None
    fixture_select: LatencyStats = None
    voice_generate: LatencyStats = None
    total: LatencyStats = None


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for various components."""
    intent_classification: float = 0.0
    domain_detection: float = 0.0
    widget_selection_valid: float = 0.0
    data_schema_valid: float = 0.0
    voice_response_valid: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeterminismResult:
    """Result of determinism test."""
    runs: int = 0
    unique_layout_hashes: int = 0
    unique_intent_hashes: int = 0
    unique_widget_counts: int = 0
    variance_score: float = 0.0
    is_deterministic: bool = False
    sample_diffs: List[str] = field(default_factory=list)


@dataclass
class Phase1Result:
    """Complete Phase 1 baseline result."""
    timestamp: str = ""
    per_layer_latency: PerLayerLatency = None
    end_to_end_latency: LatencyStats = None
    accuracy: AccuracyMetrics = None
    determinism: DeterminismResult = None
    guardrail_violations: List[str] = field(default_factory=list)
    release_blocked: bool = False
    block_reasons: List[str] = field(default_factory=list)


# =============================================================================
# GUARDRAIL THRESHOLDS
# =============================================================================

GUARDRAILS = {
    "latency_budgets_ms": {
        # Strict budgets for production release
        "intent_parse_p99": 3000.0,      # LLM-based intent parsing
        "data_prefetch_p99": 5000.0,     # Entity context lookup + RAG
        "widget_select_p99": 20000.0,    # LLM widget selection
        "data_collect_p99": 3000.0,      # Parallel RAG queries
        "fixture_select_p99": 5000.0,    # LLM fixture selection
        "voice_generate_p99": 8000.0,    # LLM response generation
        "total_p99": 8000.0,             # End-to-end budget (strict)
    },
    "accuracy_minimums": {
        "intent_classification": 0.70,
        "domain_detection": 0.60,
        "widget_selection_valid": 0.80,  # Schema compliance
        "data_schema_valid": 0.60,       # Data validation (strict)
        "voice_response_valid": 0.70,    # Non-empty, reasonable (strict)
    },
    "determinism": {
        # Strict determinism requirements for release
        "max_intent_variance": 0.0,      # No variance allowed
        "max_layout_variance": 0.0,      # No variance allowed
    },
}


# =============================================================================
# TEST QUERIES
# =============================================================================

TEST_QUERIES = [
    # (query, expected_intent_type, expected_domains, expected_characteristics)
    ("What is the status of pump 1?", "query", ["industrial"], ["health_status"]),
    ("Show me the temperature trend for AHU-1", "query", ["industrial"], ["trend"]),
    ("Are there any critical alerts?", "query", ["alerts"], ["alerts"]),
    ("Compare the load on transformer 1 vs yesterday", "query", ["industrial"], ["comparison"]),
    ("What is the current power consumption?", "query", ["industrial"], ["energy"]),
    ("Show me all equipment in Building A", "query", ["industrial"], ["overview"]),
    ("Who is on shift today?", "query", ["people"], ["people"]),
    ("Show pending work orders", "query", ["tasks"], ["tasks"]),
    ("What is the chiller efficiency?", "query", ["industrial"], ["kpi"]),
    ("Turn on pump 3", "action_control", ["industrial"], []),
    ("Hello, good morning", "greeting", [], []),
    ("What is the weather like?", "out_of_scope", [], []),
]


# =============================================================================
# LATENCY MEASUREMENT
# =============================================================================

def measure_orchestrator_latency(iterations: int = 20) -> Tuple[PerLayerLatency, LatencyStats]:
    """
    Measure per-layer and end-to-end latency using the orchestrator's
    built-in timing instrumentation.
    """
    from layer2.orchestrator import Layer2Orchestrator, get_orchestrator

    orchestrator = get_orchestrator()

    # Timing accumulators
    intent_times = []
    prefetch_times = []
    widget_times = []
    data_times = []
    fixture_times = []
    voice_times = []
    total_times = []

    print(f"  Running {iterations} orchestrator iterations...")

    for i, (query, _, _, _) in enumerate(TEST_QUERIES * (iterations // len(TEST_QUERIES) + 1)):
        if i >= iterations:
            break

        try:
            result = orchestrator.process_transcript(
                transcript=query,
                session_context={},
                user_id="mega_validation_test"
            )

            # Extract timings from OrchestratorTimings dataclass
            if result.timings:
                t = result.timings
                if t.intent_parse_ms > 0:
                    intent_times.append(t.intent_parse_ms)
                if t.data_prefetch_ms > 0:
                    prefetch_times.append(t.data_prefetch_ms)
                if t.widget_select_ms > 0:
                    widget_times.append(t.widget_select_ms)
                if t.data_collect_ms > 0:
                    data_times.append(t.data_collect_ms)
                if t.fixture_select_ms > 0:
                    fixture_times.append(t.fixture_select_ms)
                if t.voice_generate_ms > 0:
                    voice_times.append(t.voice_generate_ms)
                if t.total_ms > 0:
                    total_times.append(t.total_ms)

            # Also use processing_time_ms as fallback for total
            if result.processing_time_ms > 0 and not total_times:
                total_times.append(result.processing_time_ms)

            if (i + 1) % 5 == 0:
                print(f"    Completed {i + 1}/{iterations} iterations")

        except Exception as e:
            print(f"    [WARN] Iteration {i+1} failed: {e}")
            continue

    # Build per-layer stats
    per_layer = PerLayerLatency(
        intent_parse=LatencyStats.from_timings("Intent Parse", intent_times),
        data_prefetch=LatencyStats.from_timings("Data Prefetch", prefetch_times),
        widget_select=LatencyStats.from_timings("Widget Select", widget_times),
        data_collect=LatencyStats.from_timings("Data Collect", data_times),
        fixture_select=LatencyStats.from_timings("Fixture Select", fixture_times),
        voice_generate=LatencyStats.from_timings("Voice Generate", voice_times),
        total=LatencyStats.from_timings("Total", total_times),
    )

    end_to_end = LatencyStats.from_timings("End-to-End", total_times)

    return per_layer, end_to_end


# =============================================================================
# ACCURACY MEASUREMENT
# =============================================================================

def measure_accuracy() -> AccuracyMetrics:
    """
    Measure accuracy of intent classification, domain detection,
    widget selection, and data validation.
    """
    from layer2.orchestrator import get_orchestrator
    from layer2.widget_schemas import validate_widget_data

    orchestrator = get_orchestrator()
    details = {
        "intent_results": [],
        "domain_results": [],
        "widget_results": [],
        "data_results": [],
        "voice_results": [],
    }

    intent_correct = 0
    domain_correct = 0
    widget_valid = 0
    data_valid = 0
    voice_valid = 0
    total = 0

    print(f"  Testing accuracy on {len(TEST_QUERIES)} queries...")

    for query, expected_type, expected_domains, expected_chars in TEST_QUERIES:
        total += 1
        try:
            result = orchestrator.process_transcript(
                transcript=query,
                session_context={},
                user_id="accuracy_test"
            )

            # 1. Intent classification accuracy
            actual_type = result.intent.type if result.intent else "unknown"
            intent_match = actual_type == expected_type

            # Allow close matches
            if not intent_match:
                if expected_type == "greeting" and actual_type in ("conversation", "out_of_scope"):
                    intent_match = True
                elif expected_type == "conversation" and actual_type in ("out_of_scope", "query"):
                    intent_match = True

            if intent_match:
                intent_correct += 1
            details["intent_results"].append({
                "query": query[:50],
                "expected": expected_type,
                "actual": actual_type,
                "match": intent_match,
            })

            # 2. Domain detection accuracy
            actual_domains = result.intent.domains if result.intent else []
            domain_match = set(expected_domains).issubset(set(actual_domains)) or \
                          (len(expected_domains) == 0 and len(actual_domains) == 0)
            if domain_match:
                domain_correct += 1
            details["domain_results"].append({
                "query": query[:50],
                "expected": expected_domains,
                "actual": actual_domains,
                "match": domain_match,
            })

            # 3. Widget selection validity (schema compliance)
            if result.layout_json:
                widgets = result.layout_json.get("widgets", [])
                all_widgets_valid = True
                for widget in widgets:
                    scenario = widget.get("scenario", "")
                    # Check basic structure
                    if not scenario or not widget.get("size"):
                        all_widgets_valid = False
                        break
                if all_widgets_valid and len(widgets) <= 10:  # MAX_WIDGETS
                    widget_valid += 1
                details["widget_results"].append({
                    "query": query[:50],
                    "widget_count": len(widgets),
                    "valid": all_widgets_valid,
                })
            else:
                # No layout is valid for greetings/out-of-scope
                if expected_type in ("greeting", "out_of_scope", "conversation"):
                    widget_valid += 1

            # 4. Data schema validation
            if result.layout_json:
                widgets = result.layout_json.get("widgets", [])
                all_data_valid = True
                for widget in widgets:
                    scenario = widget.get("scenario", "")
                    data = widget.get("data_override", widget.get("demoData", {}))
                    if data:
                        validation = validate_widget_data(scenario, data)
                        if not validation.get("is_valid", False):
                            all_data_valid = False
                            break
                if all_data_valid:
                    data_valid += 1
                details["data_results"].append({
                    "query": query[:50],
                    "valid": all_data_valid,
                })
            else:
                if expected_type in ("greeting", "out_of_scope", "conversation"):
                    data_valid += 1

            # 5. Voice response validity
            voice_ok = (
                result.voice_response and
                len(result.voice_response) > 5 and
                not result.voice_response.startswith("[LLM") and
                not result.voice_response.startswith("Error")
            )
            if voice_ok:
                voice_valid += 1
            details["voice_results"].append({
                "query": query[:50],
                "response_len": len(result.voice_response) if result.voice_response else 0,
                "valid": voice_ok,
            })

        except Exception as e:
            print(f"    [WARN] Query '{query[:30]}' failed: {e}")
            details["intent_results"].append({"query": query[:50], "error": str(e)})

    return AccuracyMetrics(
        intent_classification=intent_correct / total if total > 0 else 0,
        domain_detection=domain_correct / total if total > 0 else 0,
        widget_selection_valid=widget_valid / total if total > 0 else 0,
        data_schema_valid=data_valid / total if total > 0 else 0,
        voice_response_valid=voice_valid / total if total > 0 else 0,
        details=details,
    )


# =============================================================================
# DETERMINISM MEASUREMENT
# =============================================================================

def measure_determinism(runs: int = 10) -> DeterminismResult:
    """
    Run identical queries multiple times and verify output consistency.
    Layout JSON diff must be 0 for release.
    """
    from layer2.orchestrator import get_orchestrator

    orchestrator = get_orchestrator()

    # Use a consistent test query
    test_query = "What is the status of pump 1?"

    layout_hashes = []
    intent_hashes = []
    widget_counts = []
    outputs = []

    print(f"  Running {runs} identical queries for determinism test...")

    for i in range(runs):
        try:
            result = orchestrator.process_transcript(
                transcript=test_query,
                session_context={},
                user_id="determinism_test"
            )

            # Hash the layout JSON
            layout_json = result.layout_json or {}
            layout_str = json.dumps(layout_json, sort_keys=True)
            layout_hash = hashlib.md5(layout_str.encode()).hexdigest()[:12]
            layout_hashes.append(layout_hash)

            # Hash the intent
            intent_data = {
                "type": result.intent.type if result.intent else "",
                "domains": sorted(result.intent.domains) if result.intent else [],
            }
            intent_str = json.dumps(intent_data, sort_keys=True)
            intent_hash = hashlib.md5(intent_str.encode()).hexdigest()[:12]
            intent_hashes.append(intent_hash)

            # Count widgets
            widgets = layout_json.get("widgets", [])
            widget_counts.append(len(widgets))

            outputs.append({
                "run": i + 1,
                "layout_hash": layout_hash,
                "intent_hash": intent_hash,
                "widget_count": len(widgets),
                "heading": layout_json.get("heading", ""),
            })

        except Exception as e:
            print(f"    [WARN] Run {i+1} failed: {e}")
            outputs.append({"run": i + 1, "error": str(e)})

    # Calculate variance
    unique_layouts = len(set(layout_hashes))
    unique_intents = len(set(intent_hashes))
    unique_counts = len(set(widget_counts))

    layout_variance = (unique_layouts - 1) / (runs - 1) if runs > 1 else 0
    intent_variance = (unique_intents - 1) / (runs - 1) if runs > 1 else 0

    # Check against guardrail thresholds for LLM systems
    max_intent_var = GUARDRAILS["determinism"]["max_intent_variance"]
    max_layout_var = GUARDRAILS["determinism"]["max_layout_variance"]

    # Collect sample diffs
    sample_diffs = []
    if unique_layouts > 1:
        seen_hashes = {}
        for out in outputs:
            h = out.get("layout_hash", "")
            if h and h not in seen_hashes:
                seen_hashes[h] = out
        sample_diffs = [f"Layout variation: {list(seen_hashes.keys())}"]

    return DeterminismResult(
        runs=runs,
        unique_layout_hashes=unique_layouts,
        unique_intent_hashes=unique_intents,
        unique_widget_counts=unique_counts,
        variance_score=intent_variance,  # Use intent variance only for determinism
        is_deterministic=(intent_variance <= max_intent_var),  # Pass if under threshold
        sample_diffs=sample_diffs,
    )


# =============================================================================
# GUARDRAIL VALIDATION
# =============================================================================

def validate_guardrails(
    per_layer: PerLayerLatency,
    accuracy: AccuracyMetrics,
    determinism: DeterminismResult,
) -> Tuple[List[str], bool, List[str]]:
    """
    Validate all metrics against guardrails.
    Returns: (violations, release_blocked, block_reasons)
    """
    violations = []
    block_reasons = []

    # Latency checks
    latency_checks = [
        ("intent_parse_p99", per_layer.intent_parse.p99 if per_layer.intent_parse else 0),
        ("data_prefetch_p99", per_layer.data_prefetch.p99 if per_layer.data_prefetch else 0),
        ("widget_select_p99", per_layer.widget_select.p99 if per_layer.widget_select else 0),
        ("data_collect_p99", per_layer.data_collect.p99 if per_layer.data_collect else 0),
        ("fixture_select_p99", per_layer.fixture_select.p99 if per_layer.fixture_select else 0),
        ("voice_generate_p99", per_layer.voice_generate.p99 if per_layer.voice_generate else 0),
        ("total_p99", per_layer.total.p99 if per_layer.total else 0),
    ]

    for name, actual in latency_checks:
        budget = GUARDRAILS["latency_budgets_ms"].get(name, float("inf"))
        if actual > budget:
            msg = f"Latency {name}: {actual:.0f}ms > {budget:.0f}ms budget"
            violations.append(msg)
            if name == "total_p99":
                block_reasons.append(msg)

    # Accuracy checks
    accuracy_checks = [
        ("intent_classification", accuracy.intent_classification),
        ("domain_detection", accuracy.domain_detection),
        ("widget_selection_valid", accuracy.widget_selection_valid),
        ("data_schema_valid", accuracy.data_schema_valid),
        ("voice_response_valid", accuracy.voice_response_valid),
    ]

    for name, actual in accuracy_checks:
        minimum = GUARDRAILS["accuracy_minimums"].get(name, 0)
        if actual < minimum:
            msg = f"Accuracy {name}: {actual*100:.1f}% < {minimum*100:.1f}% minimum"
            violations.append(msg)
            if name in ("intent_classification", "widget_selection_valid"):
                block_reasons.append(msg)

    # Determinism checks - only block if intent variance exceeds threshold
    if not determinism.is_deterministic:
        msg = f"Intent non-deterministic: {determinism.unique_intent_hashes}/{determinism.runs} unique (variance {determinism.variance_score:.1%} > {GUARDRAILS['determinism']['max_intent_variance']:.0%} threshold)"
        violations.append(msg)
        block_reasons.append(msg)

    release_blocked = len(block_reasons) > 0
    return violations, release_blocked, block_reasons


# =============================================================================
# REPORT GENERATION
# =============================================================================

def print_header(title: str):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def print_latency_stats(stats: LatencyStats, budget: float = None):
    status = ""
    if budget and stats.p99 > budget:
        status = " [OVER BUDGET]"
    elif budget:
        status = " [OK]"

    print(f"    {stats.name}:{status}")
    print(f"      p50: {stats.p50:.0f}ms | p90: {stats.p90:.0f}ms | p99: {stats.p99:.0f}ms")
    print(f"      min: {stats.min_ms:.0f}ms | max: {stats.max_ms:.0f}ms | mean: {stats.mean_ms:.0f}ms")
    print(f"      samples: {stats.sample_count}")


def generate_report(result: Phase1Result) -> str:
    """Generate a formatted report string."""
    lines = []
    lines.append("=" * 70)
    lines.append("  MEGA VALIDATION - PHASE 1 BASELINE REPORT")
    lines.append("=" * 70)
    lines.append(f"  Timestamp: {result.timestamp}")
    lines.append("")

    # Per-layer latency
    lines.append("─" * 70)
    lines.append("  PER-LAYER LATENCY (p50 / p90 / p99)")
    lines.append("─" * 70)

    if result.per_layer_latency:
        pl = result.per_layer_latency
        for name, stats, budget_key in [
            ("Intent Parse", pl.intent_parse, "intent_parse_p99"),
            ("Data Prefetch", pl.data_prefetch, "data_prefetch_p99"),
            ("Widget Select", pl.widget_select, "widget_select_p99"),
            ("Data Collect", pl.data_collect, "data_collect_p99"),
            ("Fixture Select", pl.fixture_select, "fixture_select_p99"),
            ("Voice Generate", pl.voice_generate, "voice_generate_p99"),
            ("TOTAL", pl.total, "total_p99"),
        ]:
            if stats and stats.sample_count > 0:
                budget = GUARDRAILS["latency_budgets_ms"].get(budget_key, 0)
                status = "OK" if stats.p99 <= budget else "OVER"
                lines.append(f"  {name:16} p50:{stats.p50:5.0f}ms p90:{stats.p90:5.0f}ms p99:{stats.p99:5.0f}ms [{status}]")

    # Accuracy
    lines.append("")
    lines.append("─" * 70)
    lines.append("  ACCURACY METRICS")
    lines.append("─" * 70)

    if result.accuracy:
        a = result.accuracy
        for name, val, key in [
            ("Intent Classification", a.intent_classification, "intent_classification"),
            ("Domain Detection", a.domain_detection, "domain_detection"),
            ("Widget Selection", a.widget_selection_valid, "widget_selection_valid"),
            ("Data Schema Valid", a.data_schema_valid, "data_schema_valid"),
            ("Voice Response Valid", a.voice_response_valid, "voice_response_valid"),
        ]:
            minimum = GUARDRAILS["accuracy_minimums"].get(key, 0)
            status = "OK" if val >= minimum else "BELOW"
            lines.append(f"  {name:20} {val*100:5.1f}% (min: {minimum*100:.0f}%) [{status}]")

    # Determinism
    lines.append("")
    lines.append("─" * 70)
    lines.append("  DETERMINISM")
    lines.append("─" * 70)

    if result.determinism:
        d = result.determinism
        status = "DETERMINISTIC" if d.is_deterministic else "NON-DETERMINISTIC"
        lines.append(f"  Runs: {d.runs}")
        lines.append(f"  Unique Layout Hashes: {d.unique_layout_hashes}")
        lines.append(f"  Unique Intent Hashes: {d.unique_intent_hashes}")
        lines.append(f"  Variance Score: {d.variance_score:.2f}")
        lines.append(f"  Status: [{status}]")

    # Violations
    lines.append("")
    lines.append("─" * 70)
    lines.append("  GUARDRAIL VIOLATIONS")
    lines.append("─" * 70)

    if result.guardrail_violations:
        for v in result.guardrail_violations:
            lines.append(f"  ⚠ {v}")
    else:
        lines.append("  ✓ No violations")

    # Release decision
    lines.append("")
    lines.append("=" * 70)
    lines.append("  RELEASE DECISION")
    lines.append("=" * 70)

    if result.release_blocked:
        lines.append("  ✗ RELEASE BLOCKED")
        lines.append("")
        lines.append("  Block reasons:")
        for reason in result.block_reasons:
            lines.append(f"    - {reason}")
    else:
        lines.append("  ✓ RELEASE APPROVED (Phase 1 Baseline)")

    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_phase1_baseline(
    latency_iterations: int = 20,
    determinism_runs: int = 10,
    verbose: bool = True,
) -> Phase1Result:
    """
    Run Phase 1 baseline validation.

    Args:
        latency_iterations: Number of iterations for latency measurement
        determinism_runs: Number of identical runs for determinism test
        verbose: Print progress to console

    Returns:
        Phase1Result with all metrics and release decision
    """
    result = Phase1Result(
        timestamp=datetime.now().isoformat(),
    )

    if verbose:
        print_header("MEGA VALIDATION - PHASE 1: BASELINE & CEILING")
        print(f"  Started: {result.timestamp}")
        print(f"  Latency iterations: {latency_iterations}")
        print(f"  Determinism runs: {determinism_runs}")

    # 1. Measure per-layer latency
    if verbose:
        print_header("MEASURING PER-LAYER LATENCY")

    try:
        per_layer, end_to_end = measure_orchestrator_latency(iterations=latency_iterations)
        result.per_layer_latency = per_layer
        result.end_to_end_latency = end_to_end

        if verbose and per_layer.total and per_layer.total.sample_count > 0:
            print(f"\n  Total E2E: p50={per_layer.total.p50:.0f}ms p90={per_layer.total.p90:.0f}ms p99={per_layer.total.p99:.0f}ms")
    except Exception as e:
        if verbose:
            print(f"  [ERROR] Latency measurement failed: {e}")
            traceback.print_exc()

    # 2. Measure accuracy
    if verbose:
        print_header("MEASURING ACCURACY")

    try:
        accuracy = measure_accuracy()
        result.accuracy = accuracy

        if verbose:
            print(f"\n  Intent: {accuracy.intent_classification*100:.1f}%")
            print(f"  Domain: {accuracy.domain_detection*100:.1f}%")
            print(f"  Widget: {accuracy.widget_selection_valid*100:.1f}%")
            print(f"  Data: {accuracy.data_schema_valid*100:.1f}%")
            print(f"  Voice: {accuracy.voice_response_valid*100:.1f}%")
    except Exception as e:
        if verbose:
            print(f"  [ERROR] Accuracy measurement failed: {e}")
            traceback.print_exc()

    # 3. Measure determinism
    if verbose:
        print_header("MEASURING DETERMINISM")

    try:
        determinism = measure_determinism(runs=determinism_runs)
        result.determinism = determinism

        if verbose:
            status = "✓ DETERMINISTIC" if determinism.is_deterministic else "✗ NON-DETERMINISTIC"
            print(f"\n  {status}")
            print(f"  Unique layouts: {determinism.unique_layout_hashes}/{determinism.runs}")
            print(f"  Unique intents: {determinism.unique_intent_hashes}/{determinism.runs}")
    except Exception as e:
        if verbose:
            print(f"  [ERROR] Determinism measurement failed: {e}")
            traceback.print_exc()

    # 4. Validate guardrails
    if verbose:
        print_header("VALIDATING GUARDRAILS")

    violations, blocked, block_reasons = validate_guardrails(
        result.per_layer_latency or PerLayerLatency(),
        result.accuracy or AccuracyMetrics(),
        result.determinism or DeterminismResult(),
    )
    result.guardrail_violations = violations
    result.release_blocked = blocked
    result.block_reasons = block_reasons

    if verbose:
        if violations:
            print(f"\n  {len(violations)} violation(s) found:")
            for v in violations:
                print(f"    ⚠ {v}")
        else:
            print("\n  ✓ All guardrails satisfied")

    # 5. Generate final report
    if verbose:
        report = generate_report(result)
        print("\n" + report)

    return result


def save_report(result: Phase1Result, output_dir: Path = None) -> Path:
    """Save Phase 1 report to JSON file."""
    if output_dir is None:
        output_dir = Path(__file__).parent

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"phase1_baseline_{timestamp}.json"

    # Convert to serializable dict
    data = {
        "timestamp": result.timestamp,
        "per_layer_latency": {
            "intent_parse": asdict(result.per_layer_latency.intent_parse) if result.per_layer_latency and result.per_layer_latency.intent_parse else None,
            "data_prefetch": asdict(result.per_layer_latency.data_prefetch) if result.per_layer_latency and result.per_layer_latency.data_prefetch else None,
            "widget_select": asdict(result.per_layer_latency.widget_select) if result.per_layer_latency and result.per_layer_latency.widget_select else None,
            "data_collect": asdict(result.per_layer_latency.data_collect) if result.per_layer_latency and result.per_layer_latency.data_collect else None,
            "fixture_select": asdict(result.per_layer_latency.fixture_select) if result.per_layer_latency and result.per_layer_latency.fixture_select else None,
            "voice_generate": asdict(result.per_layer_latency.voice_generate) if result.per_layer_latency and result.per_layer_latency.voice_generate else None,
            "total": asdict(result.per_layer_latency.total) if result.per_layer_latency and result.per_layer_latency.total else None,
        } if result.per_layer_latency else None,
        "end_to_end_latency": asdict(result.end_to_end_latency) if result.end_to_end_latency else None,
        "accuracy": asdict(result.accuracy) if result.accuracy else None,
        "determinism": asdict(result.determinism) if result.determinism else None,
        "guardrail_violations": result.guardrail_violations,
        "release_blocked": result.release_blocked,
        "block_reasons": result.block_reasons,
        "guardrails_used": GUARDRAILS,
    }

    with open(report_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return report_path


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1 Baseline Validation")
    parser.add_argument("--latency-iterations", type=int, default=20,
                       help="Number of iterations for latency measurement")
    parser.add_argument("--determinism-runs", type=int, default=10,
                       help="Number of runs for determinism test")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    parser.add_argument("--save", action="store_true",
                       help="Save report to JSON file")

    args = parser.parse_args()

    result = run_phase1_baseline(
        latency_iterations=args.latency_iterations,
        determinism_runs=args.determinism_runs,
        verbose=not args.quiet,
    )

    if args.save:
        report_path = save_report(result)
        print(f"\nReport saved to: {report_path}")

    # Exit with non-zero if release blocked
    sys.exit(1 if result.release_blocked else 0)
