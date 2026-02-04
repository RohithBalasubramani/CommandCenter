#!/usr/bin/env python3
"""
Export Training Data from Rated Widget Entries

Converts rated exhaustive entries into JSONL training pairs for
LLM fine-tuning (SFT, DPO, RLHF).

Usage:
    python export_training_data.py --run exhaustive_2026-02-01_120000
    python export_training_data.py --run <run_id> --positive-only
    python export_training_data.py --run <run_id> --pairs
    python export_training_data.py --run <run_id> --stage fixture
    python export_training_data.py --run <run_id> --stage widget
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FRONTEND_PUBLIC = SCRIPT_DIR.parent.parent / "frontend" / "public" / "simulation"


def load_data(run_id: str) -> tuple[list[dict], dict]:
    """Load exhaustive entries and ratings."""
    # Try results dir first, then frontend public
    data_path = RESULTS_DIR / run_id / "exhaustive_data.json"
    if not data_path.exists():
        data_path = FRONTEND_PUBLIC / "exhaustive_data.json"
    if not data_path.exists():
        print(f"Error: Data not found for run '{run_id}'")
        sys.exit(1)

    with open(data_path) as f:
        data = json.load(f)
    entries = data.get("entries", [])

    # Load ratings (from exported file or frontend public)
    ratings = {}
    for ratings_path in [
        RESULTS_DIR / run_id / "ratings.json",
        FRONTEND_PUBLIC / "ratings.json",
    ]:
        if ratings_path.exists():
            with open(ratings_path) as f:
                rd = json.load(f)
            ratings = rd.get("ratings", rd)
            break

    return entries, ratings


def merge_ratings(entries: list[dict], ratings: dict) -> list[dict]:
    """Merge ratings into entries."""
    for e in entries:
        r = ratings.get(e["entry_id"])
        if r:
            e["rating"] = r.get("rating")
            e["tags"] = r.get("tags", [])
            e["notes"] = r.get("notes", "")
    return entries


def format_fixture_input(entry: dict, available_fixtures: list[str]) -> str:
    """Format input for fixture selection training."""
    lines = [
        f"Scenario: {entry['scenario']}",
        f"Query: {entry['question']}",
    ]
    # Extract key data fields
    dor = entry.get("data_override", {})
    demo = dor.get("demoData", {})
    if isinstance(demo, dict):
        for key in ["label", "value", "unit", "state", "labelA", "valueA", "labelB", "valueB"]:
            if key in demo:
                lines.append(f"Data.{key}: {demo[key]}")
    elif "config" in dor:
        lines.append(f"Title: {dor['config'].get('title', '')}")

    lines.append(f"Size: {entry['size']}")
    lines.append(f"Available fixtures: [{', '.join(available_fixtures)}]")
    return "\n".join(lines)


def format_widget_input(entry: dict, all_scenarios: list[str]) -> str:
    """Format input for widget selection training."""
    intent = entry.get("pipeline_meta", {}).get("intent", {})
    lines = [
        f"User query: {entry['question']}",
        f"Available scenarios: [{', '.join(all_scenarios)}]",
    ]
    if intent:
        domains = intent.get("domains", [])
        if domains:
            lines.append(f"Detected domains: {', '.join(domains)}")
    return "\n".join(lines)


def export_fixture_training(entries: list[dict], mode: str, output_path: Path):
    """Export fixture selection training data."""
    from run_exhaustive import FIXTURE_DESCRIPTIONS

    rated = [e for e in entries if e.get("rating")]
    if mode == "positive":
        rated = [e for e in rated if e["rating"] == "up"]

    count = 0
    with open(output_path, "w") as f:
        for e in rated:
            scenario = e["scenario"]
            fixtures = list(FIXTURE_DESCRIPTIONS.get(scenario, {}).keys())
            if not fixtures:
                continue
            inp = format_fixture_input(e, fixtures)
            line = {
                "input": inp,
                "output": e["fixture"],
                "rating": e["rating"],
                "question_id": e["question_id"],
                "scenario": scenario,
            }
            if e.get("tags"):
                line["tags"] = e["tags"]
            if e.get("notes"):
                line["notes"] = e["notes"]
            f.write(json.dumps(line) + "\n")
            count += 1

    print(f"  Wrote {count} fixture training examples to {output_path}")


def export_fixture_pairs(entries: list[dict], output_path: Path):
    """Export fixture selection pairs (positive + negative for DPO)."""
    from run_exhaustive import FIXTURE_DESCRIPTIONS

    # Group by (question_id, scenario)
    groups = defaultdict(list)
    for e in entries:
        if e.get("rating"):
            groups[(e["question_id"], e["scenario"])].append(e)

    count = 0
    with open(output_path, "w") as f:
        for key, group in groups.items():
            positives = [e for e in group if e["rating"] == "up"]
            negatives = [e for e in group if e["rating"] == "down"]

            if not positives or not negatives:
                continue

            scenario = key[1]
            fixtures = list(FIXTURE_DESCRIPTIONS.get(scenario, {}).keys())
            if not fixtures:
                continue

            for pos in positives:
                for neg in negatives:
                    inp = format_fixture_input(pos, fixtures)
                    line = {
                        "input": inp,
                        "chosen": pos["fixture"],
                        "rejected": neg["fixture"],
                        "question_id": key[0],
                        "scenario": scenario,
                    }
                    f.write(json.dumps(line) + "\n")
                    count += 1

    print(f"  Wrote {count} fixture DPO pairs to {output_path}")


def export_widget_training(entries: list[dict], mode: str, output_path: Path):
    """Export widget selection training data (dashboard-level)."""
    from run_exhaustive import FIXTURE_DESCRIPTIONS, SINGLE_VARIANT_SCENARIOS

    all_scenarios = sorted(set(list(FIXTURE_DESCRIPTIONS.keys()) + list(SINGLE_VARIANT_SCENARIOS.keys())))

    # Group by question — collect which scenarios were liked/disliked
    by_question = defaultdict(lambda: {"liked_scenarios": set(), "disliked_scenarios": set(), "entries": []})
    for e in entries:
        if e.get("rating") and e.get("natural"):
            qid = e["question_id"]
            by_question[qid]["entries"].append(e)
            if e["rating"] == "up":
                by_question[qid]["liked_scenarios"].add(e["scenario"])
            else:
                by_question[qid]["disliked_scenarios"].add(e["scenario"])

    count = 0
    with open(output_path, "w") as f:
        for qid, data in by_question.items():
            if not data["entries"]:
                continue
            sample = data["entries"][0]
            inp = format_widget_input(sample, all_scenarios)

            liked = sorted(data["liked_scenarios"])
            disliked = sorted(data["disliked_scenarios"])

            if mode == "positive" and liked:
                line = {
                    "input": inp,
                    "output": ", ".join(liked),
                    "rating": "up",
                    "question_id": qid,
                }
                f.write(json.dumps(line) + "\n")
                count += 1
            elif mode == "all":
                if liked:
                    line = {
                        "input": inp,
                        "output": ", ".join(liked),
                        "rating": "up",
                        "question_id": qid,
                    }
                    f.write(json.dumps(line) + "\n")
                    count += 1
                if disliked:
                    line = {
                        "input": inp,
                        "output": ", ".join(disliked),
                        "rating": "down",
                        "question_id": qid,
                    }
                    f.write(json.dumps(line) + "\n")
                    count += 1

    print(f"  Wrote {count} widget selection examples to {output_path}")


def export_widget_pairs(entries: list[dict], output_path: Path):
    """Export widget selection pairs for DPO."""
    from run_exhaustive import FIXTURE_DESCRIPTIONS, SINGLE_VARIANT_SCENARIOS

    all_scenarios = sorted(set(list(FIXTURE_DESCRIPTIONS.keys()) + list(SINGLE_VARIANT_SCENARIOS.keys())))

    by_question = defaultdict(lambda: {"liked_scenarios": set(), "disliked_scenarios": set(), "entries": []})
    for e in entries:
        if e.get("rating") and e.get("natural"):
            qid = e["question_id"]
            by_question[qid]["entries"].append(e)
            if e["rating"] == "up":
                by_question[qid]["liked_scenarios"].add(e["scenario"])
            else:
                by_question[qid]["disliked_scenarios"].add(e["scenario"])

    count = 0
    with open(output_path, "w") as f:
        for qid, data in by_question.items():
            liked = sorted(data["liked_scenarios"])
            disliked = sorted(data["disliked_scenarios"])
            if not liked or not disliked:
                continue
            sample = data["entries"][0]
            inp = format_widget_input(sample, all_scenarios)
            line = {
                "input": inp,
                "chosen": ", ".join(liked),
                "rejected": ", ".join(disliked),
                "question_id": qid,
            }
            f.write(json.dumps(line) + "\n")
            count += 1

    print(f"  Wrote {count} widget selection DPO pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Training Data from Rated Widgets")
    parser.add_argument("--run", type=str, required=True, help="Run ID to export from")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["fixture", "widget", "both"],
        default="both",
        help="Which stage to export training data for",
    )
    parser.add_argument("--positive-only", action="store_true", help="Export only positive examples (SFT)")
    parser.add_argument("--pairs", action="store_true", help="Export DPO pairs (positive + negative)")
    parser.add_argument("--output-dir", type=str, help="Custom output directory")
    args = parser.parse_args()

    entries, ratings = load_data(args.run)
    entries = merge_ratings(entries, ratings)

    rated_count = sum(1 for e in entries if e.get("rating"))
    up_count = sum(1 for e in entries if e.get("rating") == "up")
    down_count = sum(1 for e in entries if e.get("rating") == "down")
    print(f"Loaded {len(entries)} entries, {rated_count} rated ({up_count} up, {down_count} down)")

    if rated_count == 0:
        print("No rated entries found. Rate some widgets first!")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / args.run / "training"
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "positive" if args.positive_only else "all"

    if args.stage in ("fixture", "both"):
        if args.pairs:
            export_fixture_pairs(entries, out_dir / "fixture_pairs.jsonl")
        else:
            export_fixture_training(entries, mode, out_dir / "fixture_training.jsonl")

    if args.stage in ("widget", "both"):
        if args.pairs:
            export_widget_pairs(entries, out_dir / "widget_pairs.jsonl")
        else:
            export_widget_training(entries, mode, out_dir / "widget_training.jsonl")

    print(f"\nOutput directory: {out_dir}")


if __name__ == "__main__":
    main()
