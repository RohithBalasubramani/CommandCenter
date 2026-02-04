"""
Dataset Builder for DPO Training

Builds HuggingFace datasets from formatted DPO pairs and database feedback.
"""

import json
import random
from pathlib import Path
from typing import Optional

from .config import TRAINING_DATA_CONFIG, TRAINING_DATA_DIR
from .data_formatter import (
    DPOPair,
    build_combined_dpo_pairs,
    build_fixture_dpo_pairs,
    build_widget_dpo_pairs,
    load_pairs_from_jsonl,
)


def get_all_scenarios() -> list[str]:
    """Get all available scenarios from the widget catalog."""
    try:
        # Import from layer2 if available
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from layer2.widget_catalog import WIDGET_CATALOG
        return sorted(WIDGET_CATALOG.keys())
    except ImportError:
        # Fallback to known scenarios
        return [
            "kpi", "trend", "comparison", "table", "alert",
            "gauge", "status", "list", "chart", "map",
            "timeline", "progress", "meter", "indicator", "summary",
            "pie", "bar", "donut", "heatmap",
        ]


def get_fixture_descriptions() -> dict[str, dict[str, str]]:
    """Get fixture descriptions for all scenarios."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "simulation"))
        from run_exhaustive import FIXTURE_DESCRIPTIONS, SINGLE_VARIANT_SCENARIOS
        return {**FIXTURE_DESCRIPTIONS, **SINGLE_VARIANT_SCENARIOS}
    except ImportError:
        # Fallback to empty dict - will need to load from file
        return {}


def load_entries_from_file(data_path: str) -> list[dict]:
    """Load exhaustive simulation entries from JSON file."""
    with open(data_path) as f:
        data = json.load(f)
    return data.get("entries", [])


def load_ratings_from_file(ratings_path: str) -> dict:
    """Load ratings from JSON file."""
    with open(ratings_path) as f:
        data = json.load(f)
    return data.get("ratings", data)


def merge_entries_with_ratings(entries: list[dict], ratings: dict) -> list[dict]:
    """Merge ratings into entries."""
    for entry in entries:
        entry_id = entry.get("entry_id")
        if entry_id and entry_id in ratings:
            rating_data = ratings[entry_id]
            entry["rating"] = rating_data.get("rating")
            entry["tags"] = rating_data.get("tags", [])
            entry["notes"] = rating_data.get("notes", "")
    return entries


def build_dataset_from_files(
    data_path: str,
    ratings_path: str,
    pair_type: str = "both",
) -> list[DPOPair]:
    """
    Build DPO dataset from simulation data files.

    Args:
        data_path: Path to exhaustive_data.json
        ratings_path: Path to ratings.json
        pair_type: "widget", "fixture", or "both"

    Returns:
        List of DPO pairs
    """
    entries = load_entries_from_file(data_path)
    ratings = load_ratings_from_file(ratings_path)
    entries = merge_entries_with_ratings(entries, ratings)

    # Filter to rated entries only
    rated_entries = [e for e in entries if e.get("rating")]
    if not rated_entries:
        return []

    all_scenarios = get_all_scenarios()
    fixture_descriptions = get_fixture_descriptions()

    if pair_type == "widget":
        return build_widget_dpo_pairs(rated_entries, all_scenarios)
    elif pair_type == "fixture":
        return build_fixture_dpo_pairs(rated_entries, fixture_descriptions)
    else:
        return build_combined_dpo_pairs(rated_entries, all_scenarios, fixture_descriptions)


def build_dataset_from_db(
    min_samples: int = 50,
    pair_type: str = "both",
) -> list[DPOPair]:
    """
    Build DPO dataset from database feedback.

    Queries WidgetRating model and converts to DPO pairs.
    """
    try:
        import django
        django.setup()
        from feedback.models import WidgetRating
    except Exception:
        raise RuntimeError("Django not configured. Run this from manage.py context.")

    # Get all ratings
    ratings_qs = WidgetRating.objects.all().values(
        "entry_id", "rating", "tags", "notes", "rated_at"
    )

    if ratings_qs.count() < min_samples:
        return []

    # Convert to dict format expected by data_formatter
    ratings = {
        r["entry_id"]: {
            "rating": r["rating"],
            "tags": r["tags"],
            "notes": r["notes"],
        }
        for r in ratings_qs
    }

    # We need the original entries to build pairs
    # Check for cached exhaustive data
    data_paths = [
        TRAINING_DATA_DIR / "exhaustive_data.json",
        Path(__file__).resolve().parent.parent.parent / "frontend" / "public" / "simulation" / "exhaustive_data.json",
    ]

    entries = []
    for path in data_paths:
        if path.exists():
            entries = load_entries_from_file(str(path))
            break

    if not entries:
        raise FileNotFoundError(
            "No exhaustive_data.json found. Run exhaustive simulation first."
        )

    entries = merge_entries_with_ratings(entries, ratings)
    rated_entries = [e for e in entries if e.get("rating")]

    all_scenarios = get_all_scenarios()
    fixture_descriptions = get_fixture_descriptions()

    if pair_type == "widget":
        return build_widget_dpo_pairs(rated_entries, all_scenarios)
    elif pair_type == "fixture":
        return build_fixture_dpo_pairs(rated_entries, fixture_descriptions)
    else:
        return build_combined_dpo_pairs(rated_entries, all_scenarios, fixture_descriptions)


def pairs_to_hf_dataset(pairs: list[DPOPair], val_split: float = None):
    """
    Convert DPO pairs to HuggingFace Dataset format.

    Returns:
        If val_split: DatasetDict with train/validation splits
        Otherwise: Dataset
    """
    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    if val_split is None:
        val_split = TRAINING_DATA_CONFIG["val_split"]

    # Convert to dict format
    data = {
        "prompt": [p.prompt for p in pairs],
        "chosen": [p.chosen for p in pairs],
        "rejected": [p.rejected for p in pairs],
        "question_id": [p.question_id for p in pairs],
    }

    dataset = Dataset.from_dict(data)

    if val_split and val_split > 0:
        splits = dataset.train_test_split(
            test_size=val_split,
            seed=TRAINING_DATA_CONFIG["seed"],
        )
        return DatasetDict({
            "train": splits["train"],
            "validation": splits["test"],
        })

    return dataset


def prepare_training_dataset(
    source: str = "db",
    data_path: Optional[str] = None,
    ratings_path: Optional[str] = None,
    pair_type: str = "both",
    min_samples: int = 50,
    val_split: Optional[float] = None,
):
    """
    Main entry point for preparing training dataset.

    Args:
        source: "db" for database, "file" for JSON files
        data_path: Path to exhaustive_data.json (required if source="file")
        ratings_path: Path to ratings.json (required if source="file")
        pair_type: "widget", "fixture", or "both"
        min_samples: Minimum samples required
        val_split: Validation split ratio (None = use config default)

    Returns:
        HuggingFace DatasetDict with train/validation splits
    """
    if source == "file":
        if not data_path or not ratings_path:
            raise ValueError("data_path and ratings_path required for file source")
        pairs = build_dataset_from_files(data_path, ratings_path, pair_type)
    else:
        pairs = build_dataset_from_db(min_samples, pair_type)

    if len(pairs) < min_samples:
        raise ValueError(
            f"Insufficient training data: {len(pairs)} pairs < {min_samples} required. "
            "Collect more feedback ratings first."
        )

    # Shuffle pairs
    random.seed(TRAINING_DATA_CONFIG["seed"])
    random.shuffle(pairs)

    return pairs_to_hf_dataset(pairs, val_split)


def get_dataset_stats(pairs: list[DPOPair]) -> dict:
    """Get statistics about the DPO dataset."""
    widget_pairs = [p for p in pairs if p.metadata and p.metadata.get("type") == "widget_selection"]
    fixture_pairs = [p for p in pairs if p.metadata and p.metadata.get("type") == "fixture_selection"]

    scenarios = set()
    for p in pairs:
        if p.scenario:
            scenarios.add(p.scenario)

    return {
        "total_pairs": len(pairs),
        "widget_selection_pairs": len(widget_pairs),
        "fixture_selection_pairs": len(fixture_pairs),
        "unique_scenarios": len(scenarios),
        "unique_questions": len(set(p.question_id for p in pairs)),
        "avg_prompt_length": sum(len(p.prompt) for p in pairs) // max(len(pairs), 1),
        "avg_chosen_length": sum(len(p.chosen) for p in pairs) // max(len(pairs), 1),
        "avg_rejected_length": sum(len(p.rejected) for p in pairs) // max(len(pairs), 1),
    }
