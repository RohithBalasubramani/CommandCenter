"""
Layer 2 Test Suite — Command Center AI Pipeline Validation

F7 Fix: Proper test coverage for regression protection.

Run with:
    python manage.py test layer2 -v 2

Or specific tests:
    python manage.py test layer2.tests.WidgetRegistryTests -v 2
"""

from django.test import TestCase
from unittest import skipIf
import os


# ============================================================
# Test Configuration
# ============================================================

VALID_WIDGET_SCENARIOS = {
    "kpi", "alerts", "comparison", "trend", "trend-multi-line",
    "trends-cumulative", "distribution", "composition", "category-bar",
    "timeline", "flow-sankey", "matrix-heatmap", "eventlogstream",
    "edgedevicepanel", "chatstream", "peopleview", "peoplehexgrid",
    "peoplenetwork", "supplychainglobe",
}

BANNED_SCENARIOS = {"helpview", "pulseview"}

INTENT_TEST_CASES = [
    # (input, expected_type, expected_domains, expected_primary_char)
    ("What's the status of the pumps?", "query", ["industrial"], "health_status"),
    ("Show me transformer temperatures", "query", ["industrial"], "trend"),
    ("Compare pump 1 vs pump 2", "query", ["industrial"], "comparison"),
    ("What alerts are active?", "query", ["alerts"], "alerts"),
    ("Hello", "greeting", [], None),
    ("Thank you", "conversation", [], None),
    ("What's the weather like?", "out_of_scope", [], None),
]


# ============================================================
# Widget Registry Tests
# ============================================================

class WidgetRegistryTests(TestCase):
    """Test widget registry consistency and schema coverage."""

    def test_widget_catalog_imports(self):
        """Widget catalog should be importable."""
        from layer2.widget_catalog import VALID_SCENARIOS, CATALOG_BY_SCENARIO
        self.assertIsNotNone(VALID_SCENARIOS)
        self.assertIsNotNone(CATALOG_BY_SCENARIO)

    def test_widget_schemas_imports(self):
        """Widget schemas should be importable."""
        from layer2.widget_schemas import WIDGET_SCHEMAS
        self.assertIsNotNone(WIDGET_SCHEMAS)

    def test_all_scenarios_have_schemas(self):
        """All widget scenarios should have corresponding schemas."""
        from layer2.widget_catalog import VALID_SCENARIOS
        from layer2.widget_schemas import WIDGET_SCHEMAS

        schema_scenarios = set(WIDGET_SCHEMAS.keys())
        missing = VALID_SCENARIOS - schema_scenarios

        self.assertEqual(
            missing, set(),
            f"Widgets in catalog but missing schemas: {missing}"
        )

    def test_canonical_widget_count(self):
        """Widget count should match canonical list (19 widgets)."""
        from layer2.widget_catalog import VALID_SCENARIOS
        # The canonical count is 19 valid scenarios
        # (excluding helpview and pulseview which are banned)
        valid_count = len(VALID_SCENARIOS - BANNED_SCENARIOS)
        self.assertEqual(valid_count, 17, f"Expected 17 active widgets, got {valid_count}")

    def test_banned_scenarios_defined(self):
        """Banned scenarios should be defined in widget selector."""
        from layer2.widget_selector import BANNED_SCENARIOS as selector_banned
        self.assertEqual(selector_banned, BANNED_SCENARIOS)


# ============================================================
# Intent Parser Tests
# ============================================================

class IntentParserTests(TestCase):
    """Test intent parsing accuracy and determinism."""

    def test_parser_imports(self):
        """Intent parser should be importable."""
        from layer2.intent_parser import IntentParser, ParsedIntent
        self.assertIsNotNone(IntentParser)
        self.assertIsNotNone(ParsedIntent)

    def test_greeting_intent(self):
        """Greetings should be parsed correctly."""
        from layer2.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse("Hello")
        self.assertEqual(result.type, "greeting")

    def test_out_of_scope_intent(self):
        """Out-of-scope queries should be rejected."""
        from layer2.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse("What's the weather like?")
        self.assertEqual(result.type, "out_of_scope")

    def test_industrial_query_intent(self):
        """Industrial queries should be classified correctly."""
        from layer2.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse("What's the status of the pumps?")
        self.assertEqual(result.type, "query")
        self.assertIn("industrial", result.domains)

    def test_alert_query_intent(self):
        """Alert queries should include alerts domain."""
        from layer2.intent_parser import IntentParser
        parser = IntentParser()
        result = parser.parse("What alerts are active?")
        self.assertEqual(result.type, "query")
        self.assertIn("alerts", result.domains)

    def test_parser_determinism(self):
        """Same input should produce same output."""
        from layer2.intent_parser import IntentParser

        transcript = "What's the status of pump 1?"
        outputs = []

        for _ in range(3):
            parser = IntentParser()
            parsed = parser.parse(transcript)
            output_key = (parsed.type, tuple(sorted(parsed.domains)))
            outputs.append(output_key)

        unique_outputs = set(outputs)
        self.assertEqual(
            len(unique_outputs), 1,
            f"Non-deterministic parsing: {unique_outputs}"
        )


# ============================================================
# Widget Selector Tests
# ============================================================

class WidgetSelectorTests(TestCase):
    """Test widget selection logic and constraints."""

    def test_selector_imports(self):
        """Widget selector should be importable."""
        from layer2.widget_selector import WidgetSelector, WidgetPlan
        self.assertIsNotNone(WidgetSelector)
        self.assertIsNotNone(WidgetPlan)

    def test_selector_returns_valid_plan(self):
        """Selector should return a valid WidgetPlan."""
        from layer2.widget_selector import WidgetSelector
        from layer2.intent_parser import ParsedIntent

        selector = WidgetSelector()
        intent = ParsedIntent(
            type="query",
            domains=["industrial"],
            entities={"devices": ["pump-1"]},
            raw_text="Show pump status",
            parse_method="test",
        )

        plan = selector._select_with_rules(intent)

        self.assertIsNotNone(plan)
        self.assertIsInstance(plan.widgets, list)
        self.assertGreater(len(plan.widgets), 0)

    def test_banned_scenarios_not_selected(self):
        """Banned scenarios should never appear in widget plan."""
        from layer2.widget_selector import WidgetSelector
        from layer2.intent_parser import ParsedIntent

        selector = WidgetSelector()
        intent = ParsedIntent(
            type="query",
            domains=["industrial"],
            entities={},
            raw_text="Help me understand the system",
            parse_method="test",
        )

        plan = selector._select_with_rules(intent)

        for widget in plan.widgets:
            self.assertNotIn(
                widget.scenario, BANNED_SCENARIOS,
                f"Banned scenario {widget.scenario} was selected"
            )

    def test_max_widgets_enforced(self):
        """Widget count should not exceed MAX_WIDGETS."""
        from layer2.widget_selector import WidgetSelector, MAX_WIDGETS
        from layer2.intent_parser import ParsedIntent

        selector = WidgetSelector()
        intent = ParsedIntent(
            type="query",
            domains=["industrial", "alerts"],
            entities={"devices": ["pump-1", "pump-2", "pump-3"]},
            raw_text="Show everything about all pumps",
            parse_method="test",
        )

        plan = selector._select_with_rules(intent)
        self.assertLessEqual(len(plan.widgets), MAX_WIDGETS)

    def test_valid_sizes_only(self):
        """All widgets should have valid sizes."""
        from layer2.widget_selector import WidgetSelector
        from layer2.intent_parser import ParsedIntent

        valid_sizes = {"compact", "normal", "expanded", "hero"}

        selector = WidgetSelector()
        intent = ParsedIntent(
            type="query",
            domains=["industrial"],
            entities={},
            raw_text="Show energy consumption",
            parse_method="test",
        )

        plan = selector._select_with_rules(intent)

        for widget in plan.widgets:
            self.assertIn(
                widget.size, valid_sizes,
                f"Invalid size {widget.size} for {widget.scenario}"
            )


# ============================================================
# Data Collector Tests
# ============================================================

class DataCollectorTests(TestCase):
    """Test schema-driven data collection."""

    def test_collector_imports(self):
        """Data collector should be importable."""
        from layer2.data_collector import SchemaDataCollector
        self.assertIsNotNone(SchemaDataCollector)

    def test_validation_function_exists(self):
        """Schema validation function should exist (F2 fix)."""
        from layer2.data_collector import _validate_widget_data
        self.assertIsNotNone(_validate_widget_data)

    def test_validation_accepts_valid_kpi_data(self):
        """Validation should accept valid KPI data."""
        from layer2.data_collector import _validate_widget_data

        valid_data = {
            "demoData": {
                "label": "Pump 1",
                "value": "95",
                "unit": "%",
            }
        }

        is_valid, missing = _validate_widget_data("kpi", valid_data)
        self.assertTrue(is_valid, f"Valid data rejected, missing: {missing}")

    def test_validation_rejects_incomplete_kpi_data(self):
        """Validation should reject KPI data missing required fields."""
        from layer2.data_collector import _validate_widget_data

        invalid_data = {
            "demoData": {
                "label": "Pump 1",
                # missing "value" and "unit"
            }
        }

        is_valid, missing = _validate_widget_data("kpi", invalid_data)
        self.assertFalse(is_valid)
        self.assertIn("value", missing)
        self.assertIn("unit", missing)


# ============================================================
# Orchestrator Tests
# ============================================================

class OrchestratorTests(TestCase):
    """Test orchestrator pipeline and response structure."""

    def test_orchestrator_imports(self):
        """Orchestrator should be importable."""
        from layer2.orchestrator import Layer2Orchestrator, OrchestratorResponse
        self.assertIsNotNone(Layer2Orchestrator)
        self.assertIsNotNone(OrchestratorResponse)

    def test_timings_dataclass_exists(self):
        """OrchestratorTimings should exist (F1 fix)."""
        from layer2.orchestrator import OrchestratorTimings
        timings = OrchestratorTimings()
        self.assertEqual(timings.total_ms, 0)
        self.assertEqual(timings.intent_parse_ms, 0)

    def test_timings_to_dict(self):
        """Timings should be convertible to dict."""
        from layer2.orchestrator import OrchestratorTimings
        timings = OrchestratorTimings(
            intent_parse_ms=100,
            widget_select_ms=500,
            total_ms=600,
        )
        d = timings.to_dict()
        self.assertEqual(d["intent_parse_ms"], 100)
        self.assertEqual(d["widget_select_ms"], 500)
        self.assertEqual(d["total_ms"], 600)


# ============================================================
# Schema Tests
# ============================================================

class WidgetSchemaTests(TestCase):
    """Test widget schema definitions."""

    def test_all_schemas_have_required_fields(self):
        """All schemas should define 'required' field."""
        from layer2.widget_schemas import WIDGET_SCHEMAS

        for scenario, schema in WIDGET_SCHEMAS.items():
            self.assertIn(
                "required", schema,
                f"Schema for {scenario} missing 'required' field"
            )

    def test_all_schemas_have_rag_strategy(self):
        """All schemas should define 'rag_strategy' field."""
        from layer2.widget_schemas import WIDGET_SCHEMAS

        for scenario, schema in WIDGET_SCHEMAS.items():
            self.assertIn(
                "rag_strategy", schema,
                f"Schema for {scenario} missing 'rag_strategy' field"
            )

    def test_known_rag_strategies(self):
        """All rag_strategy values should be known."""
        from layer2.widget_schemas import WIDGET_SCHEMAS

        known_strategies = {
            "single_metric", "alert_query", "multi_entity_metric",
            "time_series", "cumulative_time_series", "multi_time_series",
            "aggregation", "events_in_range", "single_entity_deep",
            "cross_tabulation", "flow_analysis", "people_query",
            "supply_query", "none",
        }

        for scenario, schema in WIDGET_SCHEMAS.items():
            strategy = schema.get("rag_strategy")
            self.assertIn(
                strategy, known_strategies,
                f"Unknown rag_strategy '{strategy}' for {scenario}"
            )


# ============================================================
# Integration Tests (require full pipeline)
# ============================================================

@skipIf(os.environ.get("SKIP_INTEGRATION_TESTS") == "1", "Skipping integration tests")
class IntegrationTests(TestCase):
    """Integration tests requiring full pipeline (LLM, RAG, etc.)."""

    def test_full_query_pipeline(self):
        """Full query should return valid response."""
        from layer2.orchestrator import Layer2Orchestrator

        orchestrator = Layer2Orchestrator()
        response = orchestrator.process_transcript(
            "What's the status of the pumps?",
            user_id="test_user"
        )

        self.assertIsNotNone(response.voice_response)
        self.assertGreater(len(response.voice_response), 0)

    def test_greeting_no_layout(self):
        """Greetings should not generate layout."""
        from layer2.orchestrator import Layer2Orchestrator

        orchestrator = Layer2Orchestrator()
        response = orchestrator.process_transcript("Hello", user_id="test_user")

        self.assertIsNone(response.layout_json)

    def test_out_of_scope_rejection(self):
        """Out-of-scope queries should be rejected gracefully."""
        from layer2.orchestrator import Layer2Orchestrator

        orchestrator = Layer2Orchestrator()
        response = orchestrator.process_transcript(
            "What's the weather like?",
            user_id="test_user"
        )

        self.assertIn("outside", response.voice_response.lower())
        self.assertIsNone(response.layout_json)
