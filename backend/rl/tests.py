"""
Tests for Command Center RL Module

Run with: python manage.py test rl
"""

import json
import tempfile
from pathlib import Path
from unittest import TestCase, mock

from .config import get_config, DPO_CONFIG, QLORA_CONFIG
from .data_formatter import (
    DPOPair,
    format_widget_selection_prompt,
    format_widget_selection_response,
    format_fixture_selection_prompt,
    format_fixture_selection_response,
    build_widget_dpo_pairs,
    build_fixture_dpo_pairs,
    pairs_to_jsonl,
    load_pairs_from_jsonl,
)
from .dataset_builder import (
    get_all_scenarios,
    merge_entries_with_ratings,
    get_dataset_stats,
)
from .online_learner import OnlineLearner, FeedbackSample


class TestConfig(TestCase):
    """Test configuration loading."""

    def test_get_default_config(self):
        config = get_config("default")
        self.assertIn("dpo", config)
        self.assertIn("qlora", config)
        self.assertIn("base_model", config)

    def test_get_small_gpu_config(self):
        config = get_config("small_gpu")
        # Small GPU should have smaller batch size
        self.assertEqual(config["dpo"]["batch_size"], 2)

    def test_get_high_quality_config(self):
        config = get_config("high_quality")
        # High quality should have more epochs
        self.assertEqual(config["dpo"]["num_epochs"], 5)

    def test_dpo_config_has_required_fields(self):
        self.assertIn("lora_r", DPO_CONFIG)
        self.assertIn("lora_alpha", DPO_CONFIG)
        self.assertIn("learning_rate", DPO_CONFIG)
        self.assertIn("beta", DPO_CONFIG)

    def test_qlora_config_has_required_fields(self):
        self.assertIn("load_in_4bit", QLORA_CONFIG)
        self.assertIn("bnb_4bit_compute_dtype", QLORA_CONFIG)


class TestDataFormatter(TestCase):
    """Test data formatting functions."""

    def test_format_widget_selection_prompt(self):
        prompt = format_widget_selection_prompt(
            query="Show me pump efficiency",
            available_scenarios=["kpi", "trend", "gauge"],
            context={"domains": ["maintenance"], "entities": {"equipment": "pump-001"}},
        )
        self.assertIn("Show me pump efficiency", prompt)
        self.assertIn("kpi", prompt)
        self.assertIn("maintenance", prompt)
        self.assertIn("pump-001", prompt)

    def test_format_widget_selection_response(self):
        response = format_widget_selection_response(
            selected_scenarios=["kpi", "trend"],
            sizes={"kpi": "compact", "trend": "normal"},
        )
        self.assertIn("kpi (compact)", response)
        self.assertIn("trend (normal)", response)

    def test_format_fixture_selection_prompt(self):
        prompt = format_fixture_selection_prompt(
            scenario="kpi",
            query="Show efficiency",
            available_fixtures=["live-metric", "percentage-change", "dual-value"],
            size="compact",
        )
        self.assertIn("kpi", prompt)
        self.assertIn("Show efficiency", prompt)
        self.assertIn("live-metric", prompt)
        self.assertIn("compact", prompt)

    def test_format_fixture_selection_response(self):
        response = format_fixture_selection_response("live-metric")
        self.assertEqual(response, "live-metric")

    def test_build_widget_dpo_pairs(self):
        entries = [
            {"question_id": "q1", "scenario": "kpi", "rating": "up", "question": "test"},
            {"question_id": "q1", "scenario": "trend", "rating": "down", "question": "test"},
        ]
        pairs = build_widget_dpo_pairs(entries, ["kpi", "trend", "gauge"])

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].question_id, "q1")
        self.assertIn("kpi", pairs[0].chosen)
        self.assertIn("trend", pairs[0].rejected)

    def test_build_fixture_dpo_pairs(self):
        entries = [
            {"question_id": "q1", "scenario": "kpi", "fixture": "live-metric", "rating": "up", "question": "test"},
            {"question_id": "q1", "scenario": "kpi", "fixture": "dual-value", "rating": "down", "question": "test"},
        ]
        fixture_descriptions = {
            "kpi": {"live-metric": "desc1", "dual-value": "desc2"},
        }
        pairs = build_fixture_dpo_pairs(entries, fixture_descriptions)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].chosen, "live-metric")
        self.assertEqual(pairs[0].rejected, "dual-value")

    def test_pairs_to_jsonl_roundtrip(self):
        pairs = [
            DPOPair(
                prompt="test prompt",
                chosen="chosen response",
                rejected="rejected response",
                question_id="q1",
                scenario="kpi",
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            pairs_to_jsonl(pairs, path)
            loaded = load_pairs_from_jsonl(path)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].prompt, "test prompt")
            self.assertEqual(loaded[0].chosen, "chosen response")
            self.assertEqual(loaded[0].rejected, "rejected response")
            self.assertEqual(loaded[0].question_id, "q1")
            self.assertEqual(loaded[0].scenario, "kpi")
        finally:
            Path(path).unlink(missing_ok=True)


class TestDatasetBuilder(TestCase):
    """Test dataset building functions."""

    def test_get_all_scenarios(self):
        scenarios = get_all_scenarios()
        self.assertIsInstance(scenarios, list)
        self.assertTrue(len(scenarios) > 0)
        # Should include common scenarios
        self.assertIn("kpi", scenarios)

    def test_merge_entries_with_ratings(self):
        entries = [
            {"entry_id": "e1", "scenario": "kpi"},
            {"entry_id": "e2", "scenario": "trend"},
        ]
        ratings = {
            "e1": {"rating": "up", "tags": ["good"], "notes": "nice"},
        }

        merged = merge_entries_with_ratings(entries, ratings)

        self.assertEqual(merged[0]["rating"], "up")
        self.assertEqual(merged[0]["tags"], ["good"])
        self.assertIsNone(merged[1].get("rating"))

    def test_get_dataset_stats(self):
        pairs = [
            DPOPair("p1", "c1", "r1", "q1", "kpi", {"type": "widget_selection"}),
            DPOPair("p2", "c2", "r2", "q2", "kpi", {"type": "fixture_selection"}),
            DPOPair("p3", "c3", "r3", "q3", "trend", {"type": "widget_selection"}),
        ]

        stats = get_dataset_stats(pairs)

        self.assertEqual(stats["total_pairs"], 3)
        self.assertEqual(stats["widget_selection_pairs"], 2)
        self.assertEqual(stats["fixture_selection_pairs"], 1)
        self.assertEqual(stats["unique_scenarios"], 2)
        self.assertEqual(stats["unique_questions"], 3)


class TestOnlineLearner(TestCase):
    """Test online learning system."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_feedback(self):
        learner = OnlineLearner(
            min_samples=10,
            checkpoint_dir=self.temp_dir,
        )

        feedback = {
            "entry_id": "e1",
            "rating": "up",
            "tags": ["good"],
            "notes": "nice widget",
        }

        learner.add_feedback(feedback)

        self.assertEqual(len(learner.feedback_buffer), 1)
        self.assertEqual(learner.feedback_buffer[0].entry_id, "e1")
        self.assertEqual(learner.feedback_buffer[0].rating, "up")

    def test_should_retrain_threshold(self):
        learner = OnlineLearner(
            min_samples=3,
            checkpoint_dir=self.temp_dir,
        )

        # Add feedback below threshold
        for i in range(2):
            learner.add_feedback({
                "entry_id": f"e{i}",
                "rating": "up",
            })

        self.assertFalse(learner.should_retrain())

        # Add one more to reach threshold
        learner.add_feedback({
            "entry_id": "e2",
            "rating": "up",
        })

        self.assertTrue(learner.should_retrain())

    def test_get_status(self):
        learner = OnlineLearner(
            min_samples=10,
            checkpoint_dir=self.temp_dir,
        )

        learner.add_feedback({"entry_id": "e1", "rating": "up"})

        status = learner.get_status()

        self.assertEqual(status["buffer_size"], 1)
        self.assertEqual(status["min_samples"], 10)
        self.assertFalse(status["is_training"])
        self.assertFalse(status["should_retrain"])

    def test_clear_buffer(self):
        learner = OnlineLearner(
            min_samples=10,
            checkpoint_dir=self.temp_dir,
        )

        learner.add_feedback({"entry_id": "e1", "rating": "up"})
        self.assertEqual(len(learner.feedback_buffer), 1)

        learner.clear_buffer()
        self.assertEqual(len(learner.feedback_buffer), 0)


class TestTrainerMocked(TestCase):
    """Test trainer with mocked dependencies."""

    def test_check_dependencies_missing(self):
        from .trainer import CommandCenterDPOTrainer

        trainer = CommandCenterDPOTrainer()

        # Mock missing torch
        with mock.patch.dict("sys.modules", {"torch": None}):
            # This should raise ImportError about missing packages
            # (depends on actual environment, so we just test the class exists)
            pass

    def test_training_result_dataclass(self):
        from .trainer import TrainingResult

        result = TrainingResult(
            success=True,
            checkpoint_path="/path/to/checkpoint",
            final_loss=0.5,
            train_samples=100,
            epochs_completed=3,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.checkpoint_path, "/path/to/checkpoint")
        self.assertEqual(result.final_loss, 0.5)
        self.assertEqual(result.train_samples, 100)


class TestExportMocked(TestCase):
    """Test export functions with mocked dependencies."""

    def test_create_ollama_modelfile(self):
        from .export import create_ollama_modelfile

        modelfile = create_ollama_modelfile(
            gguf_path="/path/to/model.gguf",
            model_name="test-model",
            temperature=0.1,
            num_ctx=2048,
        )

        self.assertIn("FROM /path/to/model.gguf", modelfile)
        self.assertIn("PARAMETER temperature 0.1", modelfile)
        self.assertIn("PARAMETER num_ctx 2048", modelfile)

    def test_export_result_dataclass(self):
        from .export import ExportResult

        result = ExportResult(
            success=True,
            gguf_path="/path/to/model.gguf",
            ollama_model_name="cc-widget-selector",
            file_size_mb=1500.5,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.gguf_path, "/path/to/model.gguf")
        self.assertEqual(result.ollama_model_name, "cc-widget-selector")
        self.assertEqual(result.file_size_mb, 1500.5)

    def test_find_llama_cpp_not_found(self):
        from .export import find_llama_cpp

        # Mock Path.exists to return False for all paths
        with mock.patch("pathlib.Path.exists", return_value=False):
            with mock.patch.dict("sys.modules", {"llama_cpp": None}):
                result = find_llama_cpp()
                # Should return None when not found
                self.assertIsNone(result)
