#!/usr/bin/env python3
"""
Simulation Results Analyzer

Joins simulation results with user feedback to produce actionable
improvement recommendations for the orchestrator and fixture selector.

Usage:
    python analyze_results.py                              # Analyze latest run
    python analyze_results.py --run run_2026-01-31_143000  # Specific run
    python analyze_results.py --compare baseline fix_1     # Compare two tagged runs
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = SCRIPT_DIR / "results"
FEEDBACK_FILE = PROJECT_ROOT / "ref" / "widget-feedback.json"


def find_run(run_id: str | None = None, tag: str | None = None) -> Path | None:
    """Find a simulation run directory by ID or tag."""
    if run_id:
        p = RESULTS_DIR / run_id
        return p if p.exists() else None

    # Find by tag
    if tag:
        for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
            log_path = d / "simulation_log.json"
            if log_path.exists():
                with open(log_path) as f:
                    data = json.load(f)
                if data.get("run_meta", {}).get("tag") == tag:
                    return d
        return None

    # Latest run
    runs = sorted(
        [d for d in RESULTS_DIR.iterdir() if d.is_dir() and (d / "simulation_log.json").exists()],
        reverse=True,
    )
    return runs[0] if runs else None


def load_simulation(run_dir: Path) -> dict:
    with open(run_dir / "simulation_log.json") as f:
        return json.load(f)


def load_feedback() -> dict:
    """Load feedback and filter for simulation entries (dashboardId starts with 'sim:')."""
    if not FEEDBACK_FILE.exists():
        return {"widgets": [], "dashboards": [], "pages": []}
    with open(FEEDBACK_FILE) as f:
        data = json.load(f)
    # Filter dashboard feedback to simulation entries only
    sim_dashboards = [
        fb for fb in data.get("dashboards", [])
        if fb.get("dashboardId", "").startswith("sim:")
    ]
    return {
        "dashboards": sim_dashboards,
        "widgets": data.get("widgets", []),
        "pages": data.get("pages", []),
    }


def analyze_run(sim_data: dict, feedback: dict) -> dict:
    """Analyze a single simulation run against feedback."""
    results = sim_data.get("results", [])
    run_meta = sim_data.get("run_meta", {})

    # Index feedback by question ID
    fb_by_question: dict[str, list[dict]] = defaultdict(list)
    for fb in feedback.get("dashboards", []):
        qid = fb.get("dashboardId", "").replace("sim:", "")
        fb_by_question[qid].append(fb)

    # Per-category analysis
    category_stats: dict[str, dict] = defaultdict(lambda: {
        "questions": [],
        "ratings": [],
        "tags": defaultdict(int),
        "widget_counts": [],
        "times": [],
        "errors": 0,
    })

    # Per-scenario+fixture analysis
    fixture_stats: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "in_questions": [],
        "associated_ratings": [],
        "associated_tags": defaultdict(int),
    })

    # Characteristic detection analysis
    characteristic_stats: dict[str, dict] = defaultdict(lambda: {
        "expected_count": 0,
        "detected_count": 0,
        "questions": [],
    })

    # Overall stats
    all_ratings = []
    questions_with_feedback = 0
    questions_without_feedback = 0

    for r in results:
        qid = r["question_id"]
        cat = r["category"]
        analysis = r.get("analysis") or {}
        cs = category_stats[cat]

        cs["questions"].append(qid)
        cs["times"].append(analysis.get("total_time_ms", 0))
        cs["widget_counts"].append(analysis.get("widget_count", 0))

        if analysis.get("error"):
            cs["errors"] += 1

        # Collect scenario+fixture usage
        for scenario, fixture in zip(
            analysis.get("scenarios_used", []),
            analysis.get("fixtures_used", []),
        ):
            key = f"{scenario}/{fixture}" if fixture else scenario
            fixture_stats[key]["count"] += 1
            fixture_stats[key]["in_questions"].append(qid)

        # Match feedback
        question_feedback = fb_by_question.get(qid, [])
        if question_feedback:
            questions_with_feedback += 1
            for fb in question_feedback:
                rating = fb.get("rating", 0)
                if rating > 0:
                    cs["ratings"].append(rating)
                    all_ratings.append(rating)
                    # Associate rating with fixtures used in this question
                    for scenario, fixture in zip(
                        analysis.get("scenarios_used", []),
                        analysis.get("fixtures_used", []),
                    ):
                        key = f"{scenario}/{fixture}" if fixture else scenario
                        fixture_stats[key]["associated_ratings"].append(rating)

                for tag in fb.get("tags", []):
                    cs["tags"][tag] += 1
                    for scenario, fixture in zip(
                        analysis.get("scenarios_used", []),
                        analysis.get("fixtures_used", []),
                    ):
                        key = f"{scenario}/{fixture}" if fixture else scenario
                        fixture_stats[key]["associated_tags"][tag] += 1
        else:
            questions_without_feedback += 1

        # Characteristic detection tracking
        expected_chars = r.get("expected_characteristics", [])
        scenario_coverage = analysis.get("scenario_coverage", {})
        for char in expected_chars:
            characteristic_stats[char]["expected_count"] += 1
            characteristic_stats[char]["questions"].append(qid)
            # We approximate detection by checking if expected scenarios are present
            if scenario_coverage.get("missing") is not None and len(scenario_coverage["missing"]) == 0:
                characteristic_stats[char]["detected_count"] += 1

    # Build report
    report = {
        "run_id": run_meta.get("run_id", ""),
        "tag": run_meta.get("tag", ""),
        "overview": {
            "total_questions": len(results),
            "questions_with_feedback": questions_with_feedback,
            "questions_without_feedback": questions_without_feedback,
            "avg_rating": round(sum(all_ratings) / len(all_ratings), 2) if all_ratings else None,
            "total_feedback_items": len(feedback.get("dashboards", [])),
        },
        "low_rated_categories": [],
        "fixture_issues": [],
        "characteristic_gaps": [],
        "category_breakdown": {},
    }

    # Low-rated categories (avg rating < 3)
    for cat, stats in category_stats.items():
        ratings = stats["ratings"]
        avg = round(sum(ratings) / len(ratings), 2) if ratings else None
        tags = dict(stats["tags"])
        entry = {
            "category": cat,
            "question_count": len(stats["questions"]),
            "avg_rating": avg,
            "feedback_count": len(ratings),
            "top_tags": sorted(tags.items(), key=lambda x: -x[1])[:5],
            "avg_widgets": round(sum(stats["widget_counts"]) / len(stats["widget_counts"]), 1) if stats["widget_counts"] else 0,
            "avg_time_ms": int(sum(stats["times"]) / len(stats["times"])) if stats["times"] else 0,
            "error_count": stats["errors"],
        }
        report["category_breakdown"][cat] = entry
        if avg is not None and avg < 3.0:
            entry["suggestion"] = f"Category '{cat}' has low avg rating ({avg}). Review layout logic for this query type."
            report["low_rated_categories"].append(entry)

    # Fixture issues (avg rating < 3 when used)
    for key, stats in fixture_stats.items():
        ratings = stats["associated_ratings"]
        if not ratings:
            continue
        avg = round(sum(ratings) / len(ratings), 2)
        if avg < 3.0:
            tags = dict(stats["associated_tags"])
            report["fixture_issues"].append({
                "fixture": key,
                "times_used": stats["count"],
                "avg_rating_when_used": avg,
                "top_tags": sorted(tags.items(), key=lambda x: -x[1])[:3],
                "suggestion": f"Fixture '{key}' gets avg {avg} rating across {stats['count']} uses. Review fixture selection rules.",
            })

    # Characteristic gaps (detection rate < 70%)
    for char, stats in characteristic_stats.items():
        if stats["expected_count"] == 0:
            continue
        rate = round(stats["detected_count"] / stats["expected_count"], 2)
        if rate < 0.7:
            report["characteristic_gaps"].append({
                "characteristic": char,
                "expected_count": stats["expected_count"],
                "detected_count": stats["detected_count"],
                "detection_rate": rate,
                "affected_questions": stats["questions"],
                "suggestion": f"'{char}' detection rate is {int(rate*100)}%. Add more regex patterns in orchestrator.py.",
            })

    # Sort by severity
    report["low_rated_categories"].sort(key=lambda x: x.get("avg_rating") or 999)
    report["fixture_issues"].sort(key=lambda x: x["avg_rating_when_used"])
    report["characteristic_gaps"].sort(key=lambda x: x["detection_rate"])

    return report


def compare_runs(run_a: dict, run_b: dict, tag_a: str, tag_b: str) -> dict:
    """Compare two simulation runs."""
    results_a = {r["question_id"]: r for r in run_a.get("results", [])}
    results_b = {r["question_id"]: r for r in run_b.get("results", [])}

    common_ids = set(results_a.keys()) & set(results_b.keys())

    comparison = {
        "run_a": {"id": run_a["run_meta"]["run_id"], "tag": tag_a},
        "run_b": {"id": run_b["run_meta"]["run_id"], "tag": tag_b},
        "common_questions": len(common_ids),
        "timing": {},
        "widget_changes": [],
        "scenario_diff": {},
    }

    times_a = []
    times_b = []
    widgets_a = []
    widgets_b = []

    for qid in sorted(common_ids):
        ra = results_a[qid]
        rb = results_b[qid]
        aa = ra.get("analysis") or {}
        ab = rb.get("analysis") or {}

        ta = aa.get("total_time_ms", 0)
        tb = ab.get("total_time_ms", 0)
        times_a.append(ta)
        times_b.append(tb)

        wa = aa.get("widget_count", 0)
        wb = ab.get("widget_count", 0)
        widgets_a.append(wa)
        widgets_b.append(wb)

        sa = set(aa.get("scenarios_used", []))
        sb = set(ab.get("scenarios_used", []))
        if sa != sb:
            comparison["widget_changes"].append({
                "question_id": qid,
                "question": ra["question"],
                "scenarios_added": list(sb - sa),
                "scenarios_removed": list(sa - sb),
                "widgets_before": wa,
                "widgets_after": wb,
            })

    comparison["timing"] = {
        f"{tag_a}_avg_ms": int(sum(times_a) / len(times_a)) if times_a else 0,
        f"{tag_b}_avg_ms": int(sum(times_b) / len(times_b)) if times_b else 0,
        "delta_ms": int((sum(times_b) - sum(times_a)) / len(times_a)) if times_a else 0,
    }

    # Scenario usage across all questions
    all_scenarios_a: dict[str, int] = defaultdict(int)
    all_scenarios_b: dict[str, int] = defaultdict(int)
    for qid in common_ids:
        for s in (results_a[qid].get("analysis") or {}).get("scenarios_used", []):
            all_scenarios_a[s] += 1
        for s in (results_b[qid].get("analysis") or {}).get("scenarios_used", []):
            all_scenarios_b[s] += 1

    all_scenarios = set(all_scenarios_a.keys()) | set(all_scenarios_b.keys())
    for s in sorted(all_scenarios):
        ca = all_scenarios_a.get(s, 0)
        cb = all_scenarios_b.get(s, 0)
        if ca != cb:
            comparison["scenario_diff"][s] = {
                f"{tag_a}_count": ca,
                f"{tag_b}_count": cb,
                "delta": cb - ca,
            }

    return comparison


def print_report(report: dict) -> None:
    """Pretty-print analysis report to terminal."""
    print(f"\n{'='*60}")
    print(f"Analysis: {report['run_id']}" + (f" [{report['tag']}]" if report['tag'] else ""))
    print(f"{'='*60}")

    ov = report["overview"]
    print(f"\nOverview:")
    print(f"  Questions: {ov['total_questions']}")
    print(f"  With feedback: {ov['questions_with_feedback']}")
    print(f"  Avg rating: {ov['avg_rating'] or 'N/A'}")

    if report["low_rated_categories"]:
        print(f"\nLow-Rated Categories:")
        for cat in report["low_rated_categories"]:
            print(f"  {cat['category']}: avg {cat['avg_rating']} ({cat['feedback_count']} reviews)")
            if cat.get("top_tags"):
                tags = ", ".join(f"{t[0]}({t[1]})" for t in cat["top_tags"][:3])
                print(f"    Tags: {tags}")
            if cat.get("suggestion"):
                print(f"    -> {cat['suggestion']}")

    if report["fixture_issues"]:
        print(f"\nFixture Issues:")
        for fi in report["fixture_issues"]:
            print(f"  {fi['fixture']}: avg {fi['avg_rating_when_used']} across {fi['times_used']} uses")
            if fi.get("suggestion"):
                print(f"    -> {fi['suggestion']}")

    if report["characteristic_gaps"]:
        print(f"\nCharacteristic Detection Gaps:")
        for cg in report["characteristic_gaps"]:
            print(f"  {cg['characteristic']}: {int(cg['detection_rate']*100)}% ({cg['detected_count']}/{cg['expected_count']})")
            if cg.get("suggestion"):
                print(f"    -> {cg['suggestion']}")

    if not report["low_rated_categories"] and not report["fixture_issues"] and not report["characteristic_gaps"]:
        if ov["questions_with_feedback"] == 0:
            print("\nNo feedback collected yet. Review dashboards in the frontend and give ratings.")
        else:
            print("\nNo issues found — all categories and fixtures above threshold.")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Simulation Results Analyzer")
    parser.add_argument("--run", type=str, help="Specific run ID to analyze")
    parser.add_argument("--tag", type=str, help="Find run by tag")
    parser.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"), help="Compare two tagged runs")
    parser.add_argument("--output", type=str, help="Output report to file (JSON)")
    args = parser.parse_args()

    if args.compare:
        tag_a, tag_b = args.compare
        dir_a = find_run(tag=tag_a)
        dir_b = find_run(tag=tag_b)
        if not dir_a:
            print(f"Error: No run found with tag '{tag_a}'")
            sys.exit(1)
        if not dir_b:
            print(f"Error: No run found with tag '{tag_b}'")
            sys.exit(1)

        sim_a = load_simulation(dir_a)
        sim_b = load_simulation(dir_b)
        comparison = compare_runs(sim_a, sim_b, tag_a, tag_b)

        print(f"\n{'='*60}")
        print(f"Comparison: [{tag_a}] vs [{tag_b}]")
        print(f"{'='*60}")
        print(f"Common questions: {comparison['common_questions']}")
        print(f"Timing: {comparison['timing']}")
        if comparison["widget_changes"]:
            print(f"\nWidget changes ({len(comparison['widget_changes'])}):")
            for wc in comparison["widget_changes"][:10]:
                added = ", ".join(wc["scenarios_added"]) or "none"
                removed = ", ".join(wc["scenarios_removed"]) or "none"
                print(f"  {wc['question_id']}: +[{added}] -[{removed}] ({wc['widgets_before']}->{wc['widgets_after']} widgets)")
        if comparison["scenario_diff"]:
            print(f"\nScenario usage delta:")
            for s, d in comparison["scenario_diff"].items():
                sign = "+" if d["delta"] > 0 else ""
                print(f"  {s}: {sign}{d['delta']}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(comparison, f, indent=2)
            print(f"\nComparison written to: {args.output}")
        return

    # Single run analysis
    run_dir = find_run(run_id=args.run, tag=args.tag)
    if not run_dir:
        print("Error: No simulation run found. Run the simulation first:")
        print("  python run_simulation.py --tag baseline")
        sys.exit(1)

    sim_data = load_simulation(run_dir)
    feedback = load_feedback()
    report = analyze_run(sim_data, feedback)

    print_report(report)

    # Write report JSON
    output_path = args.output or str(run_dir / "analysis_report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
