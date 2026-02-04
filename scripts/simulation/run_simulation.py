#!/usr/bin/env python3
"""
Dashboard Simulation Runner

Sends questions from the question bank through the RAG backend pipeline,
captures full responses, and logs everything for review and feedback.

Usage:
    python run_simulation.py                        # Run all questions
    python run_simulation.py --category energy       # Run one category
    python run_simulation.py --question q001         # Run one question
    python run_simulation.py --tag baseline          # Tag this run
    python run_simulation.py --parallel 4            # Concurrent workers
    python run_simulation.py --backend http://host:port  # Custom backend URL
"""

import argparse
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' library required. Install with: pip install requests")
    sys.exit(1)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
QUESTION_BANK = SCRIPT_DIR / "question_bank.json"
RESULTS_DIR = SCRIPT_DIR / "results"
FRONTEND_PUBLIC = PROJECT_ROOT / "frontend" / "public" / "simulation"

DEFAULT_BACKEND = "http://localhost:8100"
ORCHESTRATE_ENDPOINT = "/api/layer2/orchestrate/"
HEALTH_ENDPOINT = "/api/layer2/rag/industrial/health/"

# Mapping from question_bank expected_characteristics to orchestrator detection flags
CHARACTERISTIC_MAP = {
    "comparison": "is_comparison",
    "trend": "is_trend_query",
    "distribution": "is_distribution_query",
    "maintenance": "is_maintenance_query",
    "shift": "is_shift_query",
    "work_order": "is_work_order_query",
    "energy": "is_energy_query",
    "health": "is_health_query",
    "flow": "is_flow_query",
    "cumulative": "is_cumulative_query",
    "multi_source": "is_multi_source_query",
    "pq": "is_pq_query",
    "hvac": "is_hvac_query",
    "ups_dg": "is_ups_dg_query",
    "top_consumers": "is_top_consumers_query",
}


def load_questions(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def filter_questions(questions: list[dict], category: str | None, question_id: str | None) -> list[dict]:
    if question_id:
        return [q for q in questions if q["id"] == question_id]
    if category:
        return [q for q in questions if q["category"] == category]
    return questions


def check_backend(backend_url: str) -> dict | None:
    """Check if the backend is reachable and return health info."""
    try:
        resp = requests.get(f"{backend_url}{HEALTH_ENDPOINT}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.ConnectionError:
        return None
    except Exception:
        return None
    return None


def run_question(question: dict, backend_url: str, timeout: int = 120) -> dict:
    """Send a single question to the orchestrate endpoint and capture the result."""
    session_id = str(uuid.uuid4())
    sent_at = datetime.now().isoformat()
    start = time.time()

    result = {
        "question_id": question["id"],
        "question": question["question"],
        "category": question["category"],
        "expected_characteristics": question.get("expected_characteristics", []),
        "expected_domains": question.get("expected_domains", []),
        "expected_scenarios": question.get("expected_scenarios", []),
        "difficulty": question.get("difficulty", "basic"),
        "request": {
            "transcript": question["question"],
            "session_id": session_id,
            "sent_at": sent_at,
        },
        "response": None,
        "analysis": None,
    }

    try:
        resp = requests.post(
            f"{backend_url}{ORCHESTRATE_ENDPOINT}",
            json={"transcript": question["question"], "session_id": session_id, "context": {}},
            timeout=timeout,
        )
        elapsed_ms = int((time.time() - start) * 1000)

        if resp.status_code != 200:
            result["analysis"] = {
                "error": f"HTTP {resp.status_code}: {resp.text[:500]}",
                "total_time_ms": elapsed_ms,
            }
            return result

        data = resp.json()
        result["response"] = {
            "voice_response": data.get("voice_response", ""),
            "layout_json": data.get("layout_json"),
            "intent": data.get("intent"),
            "rag_results": data.get("rag_results", []),
            "processing_time_ms": data.get("processing_time_ms", 0),
        }

        # Analyze the response
        layout = data.get("layout_json") or {}
        widgets = layout.get("widgets", [])
        intent = data.get("intent") or {}

        scenarios_used = [w.get("scenario", "") for w in widgets]
        fixtures_used = [w.get("fixture", "") for w in widgets]

        # Characteristic match analysis
        detected_domains = intent.get("domains", [])
        expected_chars = question.get("expected_characteristics", [])

        result["analysis"] = {
            "total_time_ms": elapsed_ms,
            "network_overhead_ms": max(0, elapsed_ms - data.get("processing_time_ms", 0)),
            "widget_count": len(widgets),
            "scenarios_used": scenarios_used,
            "fixtures_used": fixtures_used,
            "heading": layout.get("heading", ""),
            "has_layout": bool(widgets),
            "has_voice_response": bool(data.get("voice_response")),
            "domain_match": {
                "expected": question.get("expected_domains", []),
                "detected": detected_domains,
                "match": set(question.get("expected_domains", [])).issubset(set(detected_domains)),
            },
            "scenario_coverage": {
                "expected": question.get("expected_scenarios", []),
                "present": list(set(scenarios_used) & set(question.get("expected_scenarios", []))),
                "missing": list(set(question.get("expected_scenarios", [])) - set(scenarios_used)),
                "unexpected": list(set(scenarios_used) - set(question.get("expected_scenarios", []))),
            },
            "error": None,
        }

    except requests.Timeout:
        elapsed_ms = int((time.time() - start) * 1000)
        result["analysis"] = {"error": f"Timeout after {timeout}s", "total_time_ms": elapsed_ms}
    except requests.ConnectionError:
        elapsed_ms = int((time.time() - start) * 1000)
        result["analysis"] = {"error": "Connection refused — is the backend running?", "total_time_ms": elapsed_ms}
    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        result["analysis"] = {"error": str(e), "total_time_ms": elapsed_ms}

    return result


def generate_summary(results: list[dict], run_meta: dict) -> dict:
    """Generate aggregate statistics from simulation results."""
    successful = [r for r in results if r.get("analysis", {}).get("error") is None]
    failed = [r for r in results if r.get("analysis", {}).get("error") is not None]

    times = [r["analysis"]["total_time_ms"] for r in successful if r.get("analysis")]
    times.sort()

    all_scenarios = set()
    all_fixtures = set()
    widget_counts = []
    domain_matches = 0
    layout_count = 0

    for r in successful:
        a = r.get("analysis", {})
        all_scenarios.update(a.get("scenarios_used", []))
        all_fixtures.update(a.get("fixtures_used", []))
        widget_counts.append(a.get("widget_count", 0))
        if a.get("domain_match", {}).get("match"):
            domain_matches += 1
        if a.get("has_layout"):
            layout_count += 1

    # Per-category stats
    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"count": 0, "successful": 0, "times": [], "widget_counts": []}
        by_category[cat]["count"] += 1
        a = r.get("analysis", {})
        if a.get("error") is None:
            by_category[cat]["successful"] += 1
            by_category[cat]["times"].append(a.get("total_time_ms", 0))
            by_category[cat]["widget_counts"].append(a.get("widget_count", 0))

    category_summary = {}
    for cat, data in by_category.items():
        t = data["times"]
        category_summary[cat] = {
            "count": data["count"],
            "successful": data["successful"],
            "avg_time_ms": int(sum(t) / len(t)) if t else 0,
            "avg_widgets": round(sum(data["widget_counts"]) / len(data["widget_counts"]), 1) if data["widget_counts"] else 0,
        }

    return {
        "run_id": run_meta["run_id"],
        "tag": run_meta.get("tag", ""),
        "timing": {
            "avg_ms": int(sum(times) / len(times)) if times else 0,
            "p50_ms": times[len(times) // 2] if times else 0,
            "p95_ms": times[int(len(times) * 0.95)] if times else 0,
            "min_ms": min(times) if times else 0,
            "max_ms": max(times) if times else 0,
        },
        "counts": {
            "total": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "with_layout": layout_count,
        },
        "accuracy": {
            "domain_match_pct": round(domain_matches / len(successful), 2) if successful else 0,
            "layout_generated_pct": round(layout_count / len(successful), 2) if successful else 0,
        },
        "widget_coverage": {
            "scenarios_used": sorted(all_scenarios),
            "fixtures_used": sorted(all_fixtures),
            "avg_widgets_per_dashboard": round(sum(widget_counts) / len(widget_counts), 1) if widget_counts else 0,
        },
        "by_category": category_summary,
        "errors": [{"question_id": r["question_id"], "error": r["analysis"]["error"]} for r in failed],
    }


def main():
    parser = argparse.ArgumentParser(description="Dashboard Simulation Runner")
    parser.add_argument("--category", type=str, help="Run only questions in this category")
    parser.add_argument("--question", type=str, help="Run a single question by ID")
    parser.add_argument("--tag", type=str, default="", help="Tag this simulation run for comparison")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--backend", type=str, default=DEFAULT_BACKEND, help="Backend base URL")
    parser.add_argument("--timeout", type=int, default=120, help="Per-question timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="List questions without running them")
    args = parser.parse_args()

    # Load and filter questions
    if not QUESTION_BANK.exists():
        print(f"Error: Question bank not found at {QUESTION_BANK}")
        sys.exit(1)

    questions = load_questions(QUESTION_BANK)
    questions = filter_questions(questions, args.category, args.question)

    if not questions:
        print("No questions matched the filter criteria.")
        sys.exit(1)

    print(f"Simulation: {len(questions)} question(s) selected")
    if args.tag:
        print(f"Tag: {args.tag}")

    if args.dry_run:
        for q in questions:
            print(f"  [{q['id']}] ({q['category']}) {q['question']}")
        return

    # Check backend health
    print(f"Backend: {args.backend}")
    health = check_backend(args.backend)
    if health is None:
        print("Error: Backend not reachable. Is the server running?")
        print(f"  Tried: {args.backend}{HEALTH_ENDPOINT}")
        sys.exit(1)

    print(f"  Status: {health.get('status', 'unknown')}")
    print(f"  Equipment: {health.get('equipment_count', 0)} indexed")
    print(f"  LLM: {'available' if health.get('llm_available') else 'unavailable'} ({health.get('llm_model', '?')})")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "run_id": run_id,
        "tag": args.tag,
        "started_at": datetime.now().isoformat(),
        "backend_url": args.backend,
        "total_questions": len(questions),
        "parallel_workers": args.parallel,
    }

    # Run simulation
    results = []
    if args.parallel > 1:
        print(f"Running {len(questions)} questions with {args.parallel} workers...")
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_question, q, args.backend, args.timeout): q
                for q in questions
            }
            for i, future in enumerate(as_completed(futures), 1):
                q = futures[future]
                result = future.result()
                results.append(result)
                err = result.get("analysis", {}).get("error")
                status = "FAIL" if err else "OK"
                widgets = result.get("analysis", {}).get("widget_count", "?")
                ms = result.get("analysis", {}).get("total_time_ms", "?")
                print(f"  [{i}/{len(questions)}] {q['id']} {status} — {widgets} widgets, {ms}ms")
    else:
        print(f"Running {len(questions)} questions sequentially...")
        for i, q in enumerate(questions, 1):
            result = run_question(q, args.backend, args.timeout)
            results.append(result)
            err = result.get("analysis", {}).get("error")
            status = "FAIL" if err else "OK"
            widgets = result.get("analysis", {}).get("widget_count", "?")
            ms = result.get("analysis", {}).get("total_time_ms", "?")
            print(f"  [{i}/{len(questions)}] {q['id']} {status} — {widgets} widgets, {ms}ms")

    # Sort results by question ID
    results.sort(key=lambda r: r["question_id"])

    run_meta["completed_at"] = datetime.now().isoformat()
    run_meta["successful"] = sum(1 for r in results if r.get("analysis", {}).get("error") is None)
    run_meta["failed"] = sum(1 for r in results if r.get("analysis", {}).get("error") is not None)

    # Write simulation log
    simulation_log = {"run_meta": run_meta, "results": results}
    log_path = run_dir / "simulation_log.json"
    with open(log_path, "w") as f:
        json.dump(simulation_log, f, indent=2)
    print(f"\nLog written: {log_path}")

    # Write summary
    summary = generate_summary(results, run_meta)
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written: {summary_path}")

    # Copy to frontend public directory for the SimulationView
    FRONTEND_PUBLIC.mkdir(parents=True, exist_ok=True)
    frontend_log = FRONTEND_PUBLIC / "simulation_log.json"
    with open(frontend_log, "w") as f:
        json.dump(simulation_log, f, indent=2)
    print(f"Frontend copy: {frontend_log}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Run: {run_id}" + (f" (tag: {args.tag})" if args.tag else ""))
    print(f"Total: {len(results)} | OK: {run_meta['successful']} | Failed: {run_meta['failed']}")
    print(f"Timing: avg {summary['timing']['avg_ms']}ms, p50 {summary['timing']['p50_ms']}ms, p95 {summary['timing']['p95_ms']}ms")
    print(f"Widgets: avg {summary['widget_coverage']['avg_widgets_per_dashboard']} per dashboard")
    print(f"Scenarios used: {', '.join(summary['widget_coverage']['scenarios_used'][:10])}")
    if summary["errors"]:
        print(f"\nErrors:")
        for e in summary["errors"]:
            print(f"  {e['question_id']}: {e['error'][:80]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
