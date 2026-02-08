"""
Data Formatter for DPO Training

Converts widget ratings and feedback into DPO-compatible training pairs.
DPO requires (prompt, chosen, rejected) triplets for preference learning.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .config import TRAINING_DATA_CONFIG


@dataclass
class DPOPair:
    """A single DPO training pair."""
    prompt: str
    chosen: str
    rejected: str
    question_id: str
    scenario: Optional[str] = None
    metadata: Optional[dict] = None


def format_widget_selection_prompt(
    query: str,
    available_scenarios: list[str],
    context: Optional[dict] = None,
    rich_evaluation: Optional[dict] = None,
) -> str:
    """
    Format the prompt for widget selection training.

    Mirrors the prompt structure used in widget_selector.py for consistency.

    Args:
        query: User query text
        available_scenarios: List of available widget types
        context: Intent context (domains, entities, etc.)
        rich_evaluation: Rich evaluation from Claude Sonnet 4.5 (optional)
    """
    lines = [
        f"User query: {query}",
        f"Available scenarios: [{', '.join(available_scenarios)}]",
    ]

    if context:
        if context.get("domains"):
            lines.append(f"Detected domains: {', '.join(context['domains'])}")
        if context.get("entities"):
            entities_str = ", ".join(f"{k}:{v}" for k, v in context["entities"].items())
            lines.append(f"Entities: {entities_str}")
        if context.get("time_range"):
            lines.append(f"Time range: {context['time_range']}")

    # Add rich evaluation context from Claude Sonnet 4.5
    if rich_evaluation:
        lines.append("")
        if rich_evaluation.get("query_understanding"):
            lines.append(f"Goal: {rich_evaluation['query_understanding']}")
        if rich_evaluation.get("missing_widgets"):
            missing = ", ".join(rich_evaluation["missing_widgets"])
            lines.append(f"Consider adding: {missing}")
        if rich_evaluation.get("suggested_improvements"):
            # Include top 2 suggestions
            suggestions = rich_evaluation["suggested_improvements"][:2]
            if suggestions:
                lines.append(f"Improvements: {'; '.join(suggestions)}")

    lines.append("")
    lines.append("Select the most appropriate widgets and their sizes.")
    lines.append("Return format: scenario (size), scenario (size), ...")

    return "\n".join(lines)


def format_widget_selection_response(
    selected_scenarios: list[str],
    sizes: Optional[dict[str, str]] = None,
) -> str:
    """Format the expected response for widget selection."""
    if sizes:
        parts = [f"{s} ({sizes.get(s, 'normal')})" for s in selected_scenarios]
    else:
        parts = [f"{s} (normal)" for s in selected_scenarios]
    return ", ".join(parts)


def format_fixture_selection_prompt(
    scenario: str,
    query: str,
    available_fixtures: list[str],
    data_context: Optional[dict] = None,
    size: str = "normal",
) -> str:
    """
    Format the prompt for fixture selection training.

    Mirrors the prompt structure used in fixture_selector.py.
    """
    lines = [
        f"Scenario: {scenario}",
        f"Query: {query}",
        f"Size: {size}",
        f"Available fixtures: [{', '.join(available_fixtures)}]",
    ]

    if data_context:
        # Add relevant data fields
        if isinstance(data_context, dict):
            demo = data_context.get("demoData", data_context)
            if isinstance(demo, dict):
                for key in ["label", "value", "unit", "state", "labelA", "valueA", "labelB", "valueB"]:
                    if key in demo:
                        lines.append(f"Data.{key}: {demo[key]}")

        if "config" in data_context:
            title = data_context["config"].get("title", "")
            if title:
                lines.append(f"Title: {title}")

    lines.append("")
    lines.append("Select the most appropriate fixture for this widget.")

    return "\n".join(lines)


def format_fixture_selection_response(fixture: str) -> str:
    """Format the expected response for fixture selection."""
    return fixture


def build_widget_dpo_pairs(
    entries: list[dict],
    all_scenarios: list[str],
    max_pairs_per_question: int = None,
) -> list[DPOPair]:
    """
    Build DPO pairs for widget selection from rated entries.

    Groups entries by question and creates pairs from liked vs disliked scenarios.
    """
    if max_pairs_per_question is None:
        max_pairs_per_question = TRAINING_DATA_CONFIG["max_pairs_per_question"]

    # Group by question — collect liked and disliked scenarios
    by_question = defaultdict(lambda: {
        "liked_scenarios": set(),
        "disliked_scenarios": set(),
        "entries": [],
    })

    for entry in entries:
        rating = entry.get("rating")
        if not rating:
            continue

        qid = entry.get("question_id")
        if not qid:
            continue

        by_question[qid]["entries"].append(entry)

        if rating == "up":
            by_question[qid]["liked_scenarios"].add(entry.get("scenario"))
        else:
            by_question[qid]["disliked_scenarios"].add(entry.get("scenario"))

    pairs = []
    for qid, data in by_question.items():
        liked = sorted(data["liked_scenarios"] - {None})
        disliked = sorted(data["disliked_scenarios"] - {None})

        if not liked or not disliked:
            continue

        # Get sample entry for context
        sample = data["entries"][0]
        query = sample.get("question", "")
        intent = sample.get("pipeline_meta", {}).get("intent", {})
        context = {
            "domains": intent.get("domains", []),
            "entities": intent.get("entities", {}),
        }

        prompt = format_widget_selection_prompt(query, all_scenarios, context)
        chosen = format_widget_selection_response(liked)
        rejected = format_widget_selection_response(disliked)

        pairs.append(DPOPair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            question_id=qid,
            metadata={"type": "widget_selection", "num_liked": len(liked), "num_disliked": len(disliked)},
        ))

    return pairs[:max_pairs_per_question * len(by_question)] if max_pairs_per_question else pairs


def build_fixture_dpo_pairs(
    entries: list[dict],
    fixture_descriptions: dict[str, dict[str, str]],
    max_pairs_per_question: int = None,
) -> list[DPOPair]:
    """
    Build DPO pairs for fixture selection from rated entries.

    Groups entries by (question_id, scenario) and creates pairs from liked vs disliked fixtures.
    """
    if max_pairs_per_question is None:
        max_pairs_per_question = TRAINING_DATA_CONFIG["max_pairs_per_question"]

    # Group by (question_id, scenario)
    groups = defaultdict(list)
    for entry in entries:
        if entry.get("rating"):
            key = (entry.get("question_id"), entry.get("scenario"))
            groups[key].append(entry)

    pairs = []
    for (qid, scenario), group in groups.items():
        if not qid or not scenario:
            continue

        positives = [e for e in group if e.get("rating") == "up"]
        negatives = [e for e in group if e.get("rating") == "down"]

        if not positives or not negatives:
            continue

        # Get available fixtures for this scenario
        fixtures = list(fixture_descriptions.get(scenario, {}).keys())
        if not fixtures:
            continue

        pair_count = 0
        for pos in positives:
            for neg in negatives:
                if max_pairs_per_question and pair_count >= max_pairs_per_question:
                    break

                prompt = format_fixture_selection_prompt(
                    scenario=scenario,
                    query=pos.get("question", ""),
                    available_fixtures=fixtures,
                    data_context=pos.get("data_override", {}),
                    size=pos.get("size", "normal"),
                )

                pairs.append(DPOPair(
                    prompt=prompt,
                    chosen=format_fixture_selection_response(pos.get("fixture", "")),
                    rejected=format_fixture_selection_response(neg.get("fixture", "")),
                    question_id=qid,
                    scenario=scenario,
                    metadata={"type": "fixture_selection"},
                ))
                pair_count += 1

    return pairs


def build_combined_dpo_pairs(
    entries: list[dict],
    all_scenarios: list[str],
    fixture_descriptions: dict[str, dict[str, str]],
) -> list[DPOPair]:
    """Build both widget and fixture selection DPO pairs."""
    widget_pairs = build_widget_dpo_pairs(entries, all_scenarios)
    fixture_pairs = build_fixture_dpo_pairs(entries, fixture_descriptions)
    return widget_pairs + fixture_pairs


def pairs_to_jsonl(pairs: list[DPOPair], output_path: str) -> int:
    """Write DPO pairs to JSONL file."""
    with open(output_path, "w") as f:
        for pair in pairs:
            record = {
                "prompt": pair.prompt,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
                "question_id": pair.question_id,
            }
            if pair.scenario:
                record["scenario"] = pair.scenario
            if pair.metadata:
                record["metadata"] = pair.metadata
            f.write(json.dumps(record) + "\n")
    return len(pairs)


def load_pairs_from_jsonl(input_path: str) -> list[DPOPair]:
    """Load DPO pairs from JSONL file."""
    pairs = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                pairs.append(DPOPair(
                    prompt=record["prompt"],
                    chosen=record["chosen"],
                    rejected=record["rejected"],
                    question_id=record.get("question_id", ""),
                    scenario=record.get("scenario"),
                    metadata=record.get("metadata"),
                ))
    return pairs
