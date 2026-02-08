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
        # (helpview and pulseview are banned but not in the catalog)
        valid_count = len(VALID_SCENARIOS - BANNED_SCENARIOS)
        self.assertEqual(valid_count, 19, f"Expected 19 active widgets, got {valid_count}")

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


# ============================================================
# Thread Safety Tests
# ============================================================

class OrchestratorThreadSafetyTests(TestCase):
    """Test thread-safe singleton pattern for get_orchestrator()."""

    def test_get_orchestrator_returns_singleton(self):
        """get_orchestrator() should return the same instance."""
        from layer2.orchestrator import get_orchestrator
        o1 = get_orchestrator()
        o2 = get_orchestrator()
        self.assertIs(o1, o2)

    def test_get_orchestrator_lock_exists(self):
        """Module-level lock must exist for orchestrator singleton."""
        from layer2 import orchestrator as mod
        self.assertTrue(hasattr(mod, '_orchestrator_lock'))
        import threading
        self.assertIsInstance(mod._orchestrator_lock, type(threading.Lock()))

    def test_get_orchestrator_concurrent_access(self):
        """Concurrent calls to get_orchestrator() must return same instance."""
        import threading
        from layer2 import orchestrator as mod

        # Reset singleton to force creation race
        mod._orchestrator = None

        results = []
        errors = []
        barrier = threading.Barrier(10)

        def get_instance():
            try:
                barrier.wait(timeout=5)
                instance = mod.get_orchestrator()
                results.append(id(instance))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        self.assertEqual(len(results), 10, "Not all threads completed")
        unique_ids = set(results)
        self.assertEqual(
            len(unique_ids), 1,
            f"Multiple instances created under concurrency: {len(unique_ids)} unique IDs"
        )


# ============================================================
# Demo Marker Tests
# ============================================================

class DemoMarkerTests(TestCase):
    """All stub data methods must carry _data_source and _integration_status markers."""

    STUB_METHODS = [
        "_get_industrial_stub_data",
        "_get_alerts_stub_data",
        "_get_supply_stub_data",
        "_get_people_stub_data",
        "_get_tasks_stub_data",
    ]

    def _get_stub_data(self, method_name):
        from layer2.orchestrator import Layer2Orchestrator
        o = Layer2Orchestrator()
        method = getattr(o, method_name)
        return method("test query", {})

    def test_all_stubs_have_data_source_marker(self):
        """Every stub method must include _data_source: 'demo'."""
        for method_name in self.STUB_METHODS:
            data = self._get_stub_data(method_name)
            self.assertIn(
                "_data_source", data,
                f"{method_name} missing _data_source marker"
            )
            self.assertEqual(
                data["_data_source"], "demo",
                f"{method_name} _data_source is '{data['_data_source']}', expected 'demo'"
            )

    def test_all_stubs_have_integration_status_marker(self):
        """Every stub method must include _integration_status: 'pending'."""
        for method_name in self.STUB_METHODS:
            data = self._get_stub_data(method_name)
            self.assertIn(
                "_integration_status", data,
                f"{method_name} missing _integration_status marker"
            )
            self.assertEqual(
                data["_integration_status"], "pending",
                f"{method_name} _integration_status is '{data['_integration_status']}', expected 'pending'"
            )


# ============================================================
# Fixture Selector Determinism Tests
# ============================================================

class FixtureSelectorDeterminismTests(TestCase):
    """Fixture selection must be deterministic: same input → same output."""

    SCENARIOS_TO_TEST = ["kpi", "alerts", "trend", "comparison", "distribution"]

    def test_no_random_import(self):
        """fixture_selector.py must not use random module."""
        import layer2.fixture_selector as mod
        import inspect
        source = inspect.getsource(mod)
        self.assertNotIn(
            "random.choice",
            source,
            "fixture_selector.py still uses random.choice — violates determinism spec"
        )

    def test_same_input_same_output_5_runs(self):
        """Same scenario + same data must produce identical fixture across 5 runs."""
        from layer2.fixture_selector import FixtureSelector

        for scenario in self.SCENARIOS_TO_TEST:
            results = []
            for _ in range(5):
                selector = FixtureSelector()
                slug = selector.select(scenario, {"demoData": {"value": "42", "unit": "%", "label": "Test"}})
                results.append(slug)
            unique = set(results)
            self.assertEqual(
                len(unique), 1,
                f"Non-deterministic: scenario={scenario} produced {unique}"
            )

    def test_diversity_is_deterministic_across_calls(self):
        """Multiple calls within same selector must follow deterministic diversity."""
        from layer2.fixture_selector import FixtureSelector

        for scenario in self.SCENARIOS_TO_TEST:
            run_a = []
            run_b = []
            for run_results in [run_a, run_b]:
                selector = FixtureSelector()
                for i in range(5):
                    slug = selector.select(scenario, {"demoData": {"value": str(i), "unit": "kW", "label": f"Device {i}"}})
                    run_results.append(slug)
            self.assertEqual(
                run_a, run_b,
                f"Non-deterministic diversity for scenario={scenario}: {run_a} vs {run_b}"
            )


# ============================================================
# System Triggers API Tests
# ============================================================

class SystemTriggersTests(TestCase):
    """Test /api/layer2/triggers/ endpoint."""

    def test_triggers_endpoint_returns_200(self):
        """Triggers endpoint should return 200 with triggers array."""
        from django.test import RequestFactory
        from layer2.views import system_triggers

        factory = RequestFactory()
        request = factory.get("/api/layer2/triggers/")
        response = system_triggers(request)
        self.assertEqual(response.status_code, 200)
        self.assertIn("triggers", response.data)
        self.assertIsInstance(response.data["triggers"], list)
        self.assertIn("timestamp", response.data)

    def test_triggers_with_since_param(self):
        """Triggers endpoint should accept 'since' query parameter."""
        from django.test import RequestFactory
        from layer2.views import system_triggers

        factory = RequestFactory()
        request = factory.get("/api/layer2/triggers/", {"since": "2026-01-01T00:00:00Z"})
        response = system_triggers(request)
        self.assertEqual(response.status_code, 200)

    def test_trigger_schema(self):
        """Each trigger must have kind, source, message, timestamp fields."""
        from django.test import RequestFactory
        from layer2.views import system_triggers

        factory = RequestFactory()
        request = factory.get("/api/layer2/triggers/")
        response = system_triggers(request)

        valid_kinds = {"alert_fired", "threshold_breach", "scheduled_event", "role_change", "time_of_day", "webhook"}
        for trigger in response.data["triggers"]:
            self.assertIn("kind", trigger)
            self.assertIn(trigger["kind"], valid_kinds, f"Unknown trigger kind: {trigger['kind']}")
            self.assertIn("source", trigger)
            self.assertIn("message", trigger)
            self.assertIn("timestamp", trigger)


# ============================================================
# Feature Flags Tests
# ============================================================

class WebhookTriggerTests(TestCase):
    """Test all 6 trigger types: alert_fired, threshold_breach, scheduled_event, role_change, time_of_day, webhook."""

    def test_webhook_trigger_endpoint_accepts_webhook(self):
        """POST /api/layer2/triggers/webhook/ should queue a webhook trigger."""
        from django.test import RequestFactory
        from layer2.views import webhook_trigger, _webhook_trigger_store

        factory = RequestFactory()
        request = factory.post(
            "/api/layer2/triggers/webhook/",
            data={"kind": "webhook", "source": "test-system", "message": "Test webhook", "payload": {"foo": "bar"}},
            content_type="application/json",
        )
        response = webhook_trigger(request)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["status"], "queued")
        self.assertEqual(response.data["trigger"]["kind"], "webhook")
        self.assertEqual(response.data["trigger"]["source"], "test-system")

        # Drain and verify
        items = _webhook_trigger_store.drain("webhook")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["source"], "test-system")

    def test_webhook_trigger_endpoint_accepts_role_change(self):
        """POST /api/layer2/triggers/webhook/ should queue a role_change trigger."""
        from django.test import RequestFactory
        from layer2.views import webhook_trigger, _webhook_trigger_store

        factory = RequestFactory()
        request = factory.post(
            "/api/layer2/triggers/webhook/",
            data={"kind": "role_change", "source": "keycloak", "message": "User promoted to supervisor"},
            content_type="application/json",
        )
        response = webhook_trigger(request)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["trigger"]["kind"], "role_change")

        # Drain and verify
        items = _webhook_trigger_store.drain("role_change")
        self.assertEqual(len(items), 1)

    def test_webhook_trigger_rejects_invalid_kind(self):
        """POST with invalid kind should return 400."""
        from django.test import RequestFactory
        from layer2.views import webhook_trigger

        factory = RequestFactory()
        request = factory.post(
            "/api/layer2/triggers/webhook/",
            data={"kind": "invalid_type", "source": "test", "message": "Bad"},
            content_type="application/json",
        )
        response = webhook_trigger(request)
        self.assertEqual(response.status_code, 400)

    def test_webhook_triggers_appear_in_poll(self):
        """Webhook triggers should appear in GET /api/layer2/triggers/ after being pushed."""
        from django.test import RequestFactory
        from layer2.views import webhook_trigger, system_triggers, _webhook_trigger_store

        factory = RequestFactory()

        # Push a webhook trigger
        push_req = factory.post(
            "/api/layer2/triggers/webhook/",
            data={"kind": "webhook", "source": "external-api", "message": "Data sync complete"},
            content_type="application/json",
        )
        webhook_trigger(push_req)

        # Poll triggers
        poll_req = factory.get("/api/layer2/triggers/")
        response = system_triggers(poll_req)
        self.assertEqual(response.status_code, 200)

        webhook_triggers = [t for t in response.data["triggers"] if t["kind"] == "webhook"]
        self.assertEqual(len(webhook_triggers), 1)
        self.assertEqual(webhook_triggers[0]["source"], "external-api")

    def test_role_change_triggers_appear_in_poll(self):
        """Role change triggers should appear in GET /api/layer2/triggers/ after being pushed."""
        from django.test import RequestFactory
        from layer2.views import webhook_trigger, system_triggers

        factory = RequestFactory()

        # Push a role_change trigger
        push_req = factory.post(
            "/api/layer2/triggers/webhook/",
            data={"kind": "role_change", "source": "keycloak", "message": "Role updated to admin"},
            content_type="application/json",
        )
        webhook_trigger(push_req)

        # Poll
        poll_req = factory.get("/api/layer2/triggers/")
        response = system_triggers(poll_req)

        role_triggers = [t for t in response.data["triggers"] if t["kind"] == "role_change"]
        self.assertEqual(len(role_triggers), 1)
        self.assertEqual(role_triggers[0]["source"], "keycloak")

    def test_all_six_trigger_kinds_recognized(self):
        """The valid trigger kinds should be exactly the 6 from the blueprint."""
        valid_kinds = {"alert_fired", "threshold_breach", "scheduled_event", "role_change", "time_of_day", "webhook"}
        # All 6 must be present in the view docstring
        from layer2.views import system_triggers
        docstring = system_triggers.__wrapped__.__doc__ if hasattr(system_triggers, "__wrapped__") else system_triggers.__doc__ or ""
        for kind in valid_kinds:
            self.assertIn(kind, docstring, f"Trigger kind '{kind}' not documented in system_triggers view")

    def test_trigger_store_thread_safety(self):
        """TriggerStore should handle concurrent push/drain without data loss."""
        from layer2.views import _TriggerStore
        import threading

        store = _TriggerStore()
        push_count = 50

        def push_batch():
            for i in range(push_count):
                store.push("test", {"kind": "test", "index": i})

        t1 = threading.Thread(target=push_batch)
        t2 = threading.Thread(target=push_batch)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        items = store.drain("test")
        self.assertEqual(len(items), push_count * 2)


class ReconciliationIntegrationTests(TestCase):
    """Reconciliation pipeline is wired into the orchestrator."""

    def test_orchestrator_imports_reconciliation(self):
        """orchestrator.py must import ReconciliationPipeline."""
        from layer2.orchestrator import ReconciliationPipeline
        self.assertIsNotNone(ReconciliationPipeline)

    def test_orchestrator_has_reconcile_method(self):
        """Layer2Orchestrator must have _reconcile_widget_data method."""
        from layer2.orchestrator import Layer2Orchestrator
        self.assertTrue(hasattr(Layer2Orchestrator, "_reconcile_widget_data"))

    def test_reconcile_passes_through_valid_data(self):
        """_reconcile_widget_data should pass through widgets without crashing."""
        from layer2.orchestrator import Layer2Orchestrator
        orch = Layer2Orchestrator()
        widgets = [
            {"scenario": "kpi", "size": "compact", "data_override": {"demoData": {"label": "Test", "value": 42}}},
            {"scenario": "trend", "size": "normal", "data_override": {"demoData": {"label": "Power"}}},
        ]
        result = orch._reconcile_widget_data(widgets)
        self.assertEqual(len(result), 2)
        # Widgets should still have their scenarios intact
        self.assertEqual(result[0]["scenario"], "kpi")
        self.assertEqual(result[1]["scenario"], "trend")

    def test_reconcile_handles_empty_data_override(self):
        """Widgets without data_override should pass through unchanged."""
        from layer2.orchestrator import Layer2Orchestrator
        orch = Layer2Orchestrator()
        widgets = [
            {"scenario": "kpi", "size": "compact"},
            {"scenario": "trend", "size": "normal", "data_override": None},
        ]
        result = orch._reconcile_widget_data(widgets)
        self.assertEqual(len(result), 2)

    def test_reconcile_ms_in_timings(self):
        """OrchestratorTimings must have reconcile_ms field."""
        from layer2.orchestrator import OrchestratorTimings
        t = OrchestratorTimings()
        self.assertEqual(t.reconcile_ms, 0)
        self.assertIn("reconcile_ms", t.to_dict())


class FeatureFlagTests(TestCase):
    """Feature flags from README must exist as enforced code variables."""

    def test_pipeline_v2_flag_exists(self):
        """PIPELINE_V2 flag must be defined."""
        from layer2.orchestrator import PIPELINE_V2
        self.assertIsInstance(PIPELINE_V2, bool)

    def test_enable_rag_flag_exists(self):
        """ENABLE_RAG flag must be defined."""
        from layer2.orchestrator import ENABLE_RAG
        self.assertIsInstance(ENABLE_RAG, bool)

    def test_rag_domains_enabled_flags_exist(self):
        """Per-domain RAG flags must exist for all 5 domains."""
        from layer2.orchestrator import RAG_DOMAINS_ENABLED
        expected_domains = {"industrial", "supply", "people", "tasks", "alerts"}
        self.assertEqual(set(RAG_DOMAINS_ENABLED.keys()), expected_domains)
        for domain, enabled in RAG_DOMAINS_ENABLED.items():
            self.assertIsInstance(enabled, bool, f"RAG_{domain}_ENABLED must be bool")


class PerformanceBudgetTests(TestCase):
    """Performance budget constants must exist, match blueprint, and enforce at runtime."""

    def test_budget_constants_exist(self):
        """All 4 budget constants must be defined in orchestrator."""
        from layer2.orchestrator import (
            BUDGET_INTENT_MS, BUDGET_RAG_MS,
            BUDGET_WIDGET_SELECT_MS, BUDGET_TOTAL_MS,
        )
        self.assertEqual(BUDGET_INTENT_MS, 30_000)
        self.assertEqual(BUDGET_RAG_MS, 60_000)
        self.assertEqual(BUDGET_WIDGET_SELECT_MS, 60_000)
        self.assertEqual(BUDGET_TOTAL_MS, 300_000)

    def test_timings_check_budget_records_warning(self):
        """OrchestratorTimings.check_budget() must record violations."""
        from layer2.orchestrator import OrchestratorTimings
        timings = OrchestratorTimings()
        # Under budget — no warning
        timings.check_budget("test_stage", 400, 500)
        self.assertEqual(len(timings.budget_warnings), 0)
        # Over budget — warning recorded
        timings.check_budget("test_stage", 600, 500)
        self.assertEqual(len(timings.budget_warnings), 1)
        w = timings.budget_warnings[0]
        self.assertEqual(w["stage"], "test_stage")
        self.assertEqual(w["elapsed_ms"], 600)
        self.assertEqual(w["budget_ms"], 500)

    def test_timings_check_budget_exact_boundary(self):
        """Exactly at budget should NOT trigger a warning (only > triggers)."""
        from layer2.orchestrator import OrchestratorTimings
        timings = OrchestratorTimings()
        timings.check_budget("boundary", 500, 500)
        self.assertEqual(len(timings.budget_warnings), 0)

    def test_budget_warnings_in_to_dict(self):
        """budget_warnings must be included in to_dict() output for API responses."""
        from layer2.orchestrator import OrchestratorTimings
        timings = OrchestratorTimings()
        timings.check_budget("slow_stage", 9999, 1000)
        d = timings.to_dict()
        self.assertIn("budget_warnings", d)
        self.assertEqual(len(d["budget_warnings"]), 1)


class MaxHeightUnitsConsistencyTests(TestCase):
    """MAX_HEIGHT_UNITS must be consistent across code and docs."""

    def test_code_value_is_24(self):
        """widget_selector.MAX_HEIGHT_UNITS must be 24."""
        from layer2.widget_selector import MAX_HEIGHT_UNITS
        self.assertEqual(MAX_HEIGHT_UNITS, 24)

    def test_orchestrator_imports_same_value(self):
        """orchestrator must import MAX_HEIGHT_UNITS from widget_selector (same value)."""
        from layer2.orchestrator import MAX_HEIGHT_UNITS as orch_max
        from layer2.widget_selector import MAX_HEIGHT_UNITS as ws_max
        self.assertEqual(orch_max, ws_max)

    def test_rag_pipeline_doc_matches_code(self):
        """RAG_PIPELINE.md must say 'max 24 height units' (not 18)."""
        import pathlib
        doc_path = pathlib.Path(__file__).resolve().parent.parent.parent / "docs" / "RAG_PIPELINE.md"
        if not doc_path.exists():
            self.skipTest("RAG_PIPELINE.md not found")
        content = doc_path.read_text()
        self.assertIn("max 24 height units", content)
        self.assertNotIn("max 18 height units", content)
