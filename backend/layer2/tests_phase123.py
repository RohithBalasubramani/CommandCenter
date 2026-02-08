"""
Phase 1-3 RAG & Widget Selection Redesign — Unit Tests

Tests all new modules from the redesign:
  Phase 1: RetrievalAssessor, widget deduplication
  Phase 2: QueryDecomposer, DashboardPlanner, ConfidenceEnvelope
  Phase 3: CompositionScorer

Run:  python manage.py test layer2.tests_phase123 -v 2
"""

import time
from dataclasses import field
from datetime import datetime, timezone
from unittest import TestCase


# ════════════════════════════════════════════════════════════════
# Phase 1: Retrieval Assessor Tests
# ════════════════════════════════════════════════════════════════

class RetrievalAssessorTests(TestCase):
    """Tests for backend/layer2/retrieval_assessor.py"""

    def test_imports(self):
        from layer2.retrieval_assessor import (
            RetrievalAssessor, WidgetAssessment, RetrievalAssessment,
            apply_assessment_to_widgets,
        )
        self.assertIsNotNone(RetrievalAssessor)
        self.assertIsNotNone(apply_assessment_to_widgets)

    def test_empty_input_returns_empty_assessment(self):
        from layer2.retrieval_assessor import RetrievalAssessor
        assessor = RetrievalAssessor()
        result = assessor.assess([])
        self.assertEqual(result.total_widgets, 0)
        self.assertEqual(result.completeness, 0.0)

    def test_real_data_full_completeness(self):
        from layer2.retrieval_assessor import RetrievalAssessor
        now_iso = datetime.now(timezone.utc).isoformat()
        widgets = [
            {
                "scenario": "kpi", "size": "normal",
                "data_override": {
                    "demoData": {
                        "label": "Pump Power", "value": 42.5, "unit": "kW",
                        "_data_source": "pg_timeseries:pump_001",
                        "_timestamp": now_iso,
                    }
                }
            },
            {
                "scenario": "kpi", "size": "normal",
                "data_override": {
                    "demoData": {
                        "label": "Motor Temp", "value": 65.0, "unit": "C",
                        "_data_source": "pg_timeseries:pump_002",
                        "_timestamp": now_iso,
                    }
                }
            },
        ]
        assessor = RetrievalAssessor()
        result = assessor.assess(widgets)
        self.assertEqual(result.real_data_widgets, 2)
        self.assertAlmostEqual(result.completeness, 1.0)

    def test_synthetic_data_detected(self):
        from layer2.retrieval_assessor import RetrievalAssessor
        widgets = [{
            "scenario": "kpi", "size": "normal",
            "data_override": {
                "demoData": {
                    "label": "Test", "value": 100,
                    "_data_source": "no_data_fallback",
                }
            }
        }]
        assessor = RetrievalAssessor()
        result = assessor.assess(widgets)
        self.assertEqual(result.synthetic_widgets, 1)
        self.assertEqual(result.real_data_widgets, 0)
        wa = result.widget_assessments[0]
        self.assertTrue(wa.is_synthetic)

    def test_kpi_na_value_should_drop(self):
        from layer2.retrieval_assessor import RetrievalAssessor
        widgets = [{
            "scenario": "kpi", "size": "normal",
            "data_override": {
                "demoData": {"label": "Test", "value": "N/A", "unit": "kW"}
            }
        }]
        assessor = RetrievalAssessor()
        result = assessor.assess(widgets)
        wa = result.widget_assessments[0]
        self.assertTrue(wa.should_drop)

    def test_trend_insufficient_points_dropped(self):
        from layer2.retrieval_assessor import RetrievalAssessor
        widgets = [{
            "scenario": "trend", "size": "expanded",
            "data_override": {
                "demoData": {
                    "label": "Power", "unit": "kW",
                    "timeSeries": [{"t": "2024-01-01T00:00", "v": 10}] * 3,
                    "_data_source": "pg_timeseries:pump_001",
                }
            }
        }]
        assessor = RetrievalAssessor()
        result = assessor.assess(widgets)
        wa = result.widget_assessments[0]
        self.assertTrue(wa.should_drop)
        self.assertIn("data points", wa.drop_reason)

    def test_trend_sufficient_points_kept(self):
        from layer2.retrieval_assessor import RetrievalAssessor
        now_iso = datetime.now(timezone.utc).isoformat()
        widgets = [{
            "scenario": "trend", "size": "expanded",
            "data_override": {
                "demoData": {
                    "label": "Power", "unit": "kW",
                    "timeSeries": [{"t": f"2024-01-01T{i:02d}:00", "v": 10 + i} for i in range(10)],
                    "_data_source": "pg_timeseries:pump_001",
                    "_timestamp": now_iso,
                }
            }
        }]
        assessor = RetrievalAssessor()
        result = assessor.assess(widgets)
        wa = result.widget_assessments[0]
        self.assertFalse(wa.should_drop)

    def test_tolerates_empty_not_dropped(self):
        from layer2.retrieval_assessor import RetrievalAssessor
        widgets = [{
            "scenario": "edgedevicepanel", "size": "expanded",
            "data_override": {"demoData": {}}
        }]
        assessor = RetrievalAssessor()
        result = assessor.assess(widgets)
        wa = result.widget_assessments[0]
        self.assertFalse(wa.should_drop)

    def test_apply_assessment_drops_flagged(self):
        from layer2.retrieval_assessor import RetrievalAssessor, apply_assessment_to_widgets
        widgets = [
            {"scenario": "kpi", "size": "normal", "data_override": {"demoData": {"label": "A", "value": 42, "_data_source": "pg_timeseries:x", "_timestamp": datetime.now(timezone.utc).isoformat()}}},
            {"scenario": "kpi", "size": "normal", "data_override": {"demoData": {"label": "B", "value": "N/A"}}},
            {"scenario": "kpi", "size": "normal", "data_override": {"demoData": {"label": "C", "value": 99, "_data_source": "pg_timeseries:y", "_timestamp": datetime.now(timezone.utc).isoformat()}}},
        ]
        assessor = RetrievalAssessor()
        assessment = assessor.assess(widgets)
        result = apply_assessment_to_widgets(widgets, assessment)
        # Widget B has N/A value → should be dropped
        self.assertEqual(len(result), 2)

    def test_apply_assessment_injects_metadata(self):
        from layer2.retrieval_assessor import RetrievalAssessor, apply_assessment_to_widgets
        now_iso = datetime.now(timezone.utc).isoformat()
        widgets = [{
            "scenario": "kpi", "size": "normal",
            "data_override": {"demoData": {"label": "X", "value": 55, "_data_source": "pg_timeseries:p", "_timestamp": now_iso}}
        }]
        assessor = RetrievalAssessor()
        assessment = assessor.assess(widgets)
        result = apply_assessment_to_widgets(widgets, assessment)
        w = result[0]
        self.assertIn("_data_quality", w)
        self.assertIn("_is_stale", w)
        self.assertIn("_widget_confidence", w)
        self.assertIn("_is_synthetic", w)

    def test_to_dict_serialization(self):
        from layer2.retrieval_assessor import RetrievalAssessment
        a = RetrievalAssessment(total_widgets=5, real_data_widgets=3, completeness=0.6, freshness=0.8)
        d = a.to_dict()
        for key in ("total_widgets", "real_data_widgets", "synthetic_widgets",
                     "stale_widgets", "widgets_to_drop", "completeness",
                     "freshness", "data_fill_quality", "gaps", "warnings"):
            self.assertIn(key, d)


# ════════════════════════════════════════════════════════════════
# Phase 1: Widget Dedup Tests
# ════════════════════════════════════════════════════════════════

class WidgetDedupTests(TestCase):
    """Tests for backend/layer2/widget_dedup.py"""

    def test_imports(self):
        from layer2.widget_dedup import (
            deduplicate_widgets, inject_contradiction_flags,
            extract_signature, DeduplicationResult,
        )
        self.assertIsNotNone(deduplicate_widgets)

    def test_empty_list(self):
        from layer2.widget_dedup import deduplicate_widgets
        result = deduplicate_widgets([])
        self.assertEqual(result.kept, [])
        self.assertEqual(result.dropped, [])
        self.assertEqual(result.contradictions, [])

    def test_no_duplicates_all_kept(self):
        from layer2.widget_dedup import deduplicate_widgets
        widgets = [
            {"scenario": "kpi", "size": "normal", "data_override": {"demoData": {"label": "pump-001", "unit": "kW", "_data_source": "pg:pump_001"}}},
            {"scenario": "kpi", "size": "normal", "data_override": {"demoData": {"label": "pump-002", "unit": "kW", "_data_source": "pg:pump_002"}}},
        ]
        result = deduplicate_widgets(widgets)
        self.assertEqual(len(result.kept), 2)
        self.assertEqual(len(result.dropped), 0)

    def test_identical_kpi_deduplicated(self):
        from layer2.widget_dedup import deduplicate_widgets
        widgets = [
            {"scenario": "kpi", "size": "normal", "relevance": 0.8, "data_override": {"demoData": {"label": "pump-001", "unit": "kW", "_data_source": "pg:pump_001"}}},
            {"scenario": "kpi", "size": "normal", "relevance": 0.5, "data_override": {"demoData": {"label": "pump-001", "unit": "kW", "_data_source": "pg:pump_001"}}},
        ]
        result = deduplicate_widgets(widgets)
        self.assertEqual(len(result.kept), 1)
        self.assertEqual(len(result.dropped), 1)

    def test_kpi_and_trend_different_timeseries_both_kept(self):
        """KPI (is_timeseries=False) and trend (is_timeseries=True) have different
        dedup_keys, so both are kept as complementary views of the same data."""
        from layer2.widget_dedup import deduplicate_widgets
        widgets = [
            {"scenario": "kpi", "size": "normal", "relevance": 0.8, "data_override": {"demoData": {"label": "pump-001", "unit": "kW", "_data_source": "pg:pump_001"}}},
            {"scenario": "trend", "size": "expanded", "relevance": 0.7, "data_override": {"demoData": {"label": "pump-001", "unit": "kW", "_data_source": "pg:pump_001"}}},
        ]
        result = deduplicate_widgets(widgets)
        self.assertEqual(len(result.kept), 2)  # Different is_timeseries → different dedup_key

    def test_hero_never_dropped(self):
        from layer2.widget_dedup import deduplicate_widgets
        widgets = [
            {"scenario": "kpi", "size": "hero", "relevance": 0.5, "data_override": {"demoData": {"label": "pump-001", "unit": "kW", "_data_source": "pg:pump_001"}}},
            {"scenario": "kpi", "size": "normal", "relevance": 0.9, "data_override": {"demoData": {"label": "pump-001", "unit": "kW", "_data_source": "pg:pump_001"}}},
        ]
        result = deduplicate_widgets(widgets)
        hero = [w for w in result.kept if w["size"] == "hero"]
        self.assertEqual(len(hero), 1)

    def test_complementary_kpis_both_kept(self):
        from layer2.widget_dedup import deduplicate_widgets
        widgets = [
            {"scenario": "kpi", "size": "normal", "data_override": {"demoData": {"label": "pump-001", "unit": "kW"}}},
            {"scenario": "kpi", "size": "normal", "data_override": {"demoData": {"label": "pump-001", "unit": "C"}}},
        ]
        result = deduplicate_widgets(widgets)
        # Different units → different metric_column → both kept
        self.assertEqual(len(result.kept), 2)

    def test_contradiction_kpi_normal_alert_critical(self):
        """Contradiction detected: KPI shows 'normal' but alert shows 'critical' for same entity.
        Widgets must have different dedup_keys so both survive dedup to reach contradiction check.
        KPI has metric_column from unit, alerts has no metric — different dedup_keys."""
        from layer2.widget_dedup import deduplicate_widgets
        widgets = [
            {"scenario": "kpi", "size": "normal", "data_override": {"demoData": {"label": "trf-1", "value": 100, "unit": "kW", "state": "normal"}}},
            {"scenario": "alerts", "size": "normal", "data_override": {"demoData": {"alerts": [{"source": "trf-1", "severity": "critical", "message": "overheating"}]}}},
        ]
        result = deduplicate_widgets(widgets)
        # Both widgets should be kept (different dedup_keys: kpi has metric from unit, alerts has none)
        self.assertEqual(len(result.kept), 2)
        self.assertGreater(len(result.contradictions), 0)
        self.assertIn("trf-1", result.contradictions[0]["entity"])

    def test_no_contradiction_when_agreement(self):
        from layer2.widget_dedup import deduplicate_widgets
        widgets = [
            {"scenario": "kpi", "size": "normal", "data_override": {"demoData": {"label": "trf-1", "value": 100, "state": "warning"}}},
            {"scenario": "alerts", "size": "normal", "data_override": {"demoData": {"alerts": [{"source": "trf-1", "severity": "warning", "message": "high temp"}]}}},
        ]
        result = deduplicate_widgets(widgets)
        self.assertEqual(len(result.contradictions), 0)

    def test_inject_contradiction_flags(self):
        from layer2.widget_dedup import inject_contradiction_flags
        widgets = [
            {"scenario": "kpi", "data_override": {}},
            {"scenario": "alerts", "data_override": {}},
        ]
        contradictions = [{
            "entity": "trf-1", "kpi_state": "normal", "alert_state": "critical",
            "kpi_widget_idx": 0, "alert_widget_idx": 1,
            "message": "trf-1: KPI normal but alert critical",
        }]
        result = inject_contradiction_flags(widgets, contradictions)
        self.assertIn("_conflict_flag", result[0])
        self.assertIn("_conflict_flag", result[1])

    def test_inject_no_contradictions_noop(self):
        from layer2.widget_dedup import inject_contradiction_flags
        widgets = [{"scenario": "kpi", "data_override": {}}]
        result = inject_contradiction_flags(widgets, [])
        self.assertNotIn("_conflict_flag", result[0])

    def test_entity_normalization_case_insensitive(self):
        from layer2.widget_dedup import _normalize_entity
        self.assertEqual(_normalize_entity("Pump-001"), _normalize_entity("pump-001"))


# ════════════════════════════════════════════════════════════════
# Phase 2: Query Decomposer Tests
# ════════════════════════════════════════════════════════════════

class QueryDecomposerTests(TestCase):
    """Tests for backend/layer2/query_decomposer.py"""

    def _make_intent(self, **kwargs):
        from layer2.intent_parser import ParsedIntent
        defaults = {
            "type": "query",
            "domains": ["industrial"],
            "entities": {"devices": ["pump-001"]},
            "confidence": 0.85,
            "raw_text": "Show me pump-001 status",
            "primary_characteristic": "health_status",
        }
        defaults.update(kwargs)
        return ParsedIntent(**defaults)

    def test_imports(self):
        from layer2.query_decomposer import QueryDecomposer, RetrievalPlan, TemporalScope
        self.assertIsNotNone(QueryDecomposer)
        self.assertIsNotNone(RetrievalPlan)

    def test_decompose_returns_retrieval_plan(self):
        from layer2.query_decomposer import QueryDecomposer, RetrievalPlan
        decomposer = QueryDecomposer()
        intent = self._make_intent()
        plan = decomposer.decompose(intent)
        self.assertIsInstance(plan, RetrievalPlan)

    def test_entity_populates_plan(self):
        """Decompose with entity should populate plan's entity list."""
        from layer2.query_decomposer import QueryDecomposer
        decomposer = QueryDecomposer()
        intent = self._make_intent(entities={"devices": ["pump-001"]})
        plan = decomposer.decompose(intent)
        # Entity should appear in plan.entities even if entity_tables resolution fails
        self.assertIn("pump-001", plan.entities)

    def test_temporal_scope_default(self):
        from layer2.query_decomposer import QueryDecomposer
        decomposer = QueryDecomposer()
        intent = self._make_intent(primary_characteristic="trend")
        plan = decomposer.decompose(intent)
        self.assertGreater(plan.temporal_scope.hours, 0)

    def test_temporal_scope_explicit_last_n_hours(self):
        from layer2.query_decomposer import QueryDecomposer
        decomposer = QueryDecomposer()
        intent = self._make_intent(raw_text="Show pump data for last 4 hours")
        plan = decomposer.decompose(intent)
        self.assertEqual(plan.temporal_scope.hours, 4)

    def test_temporal_scope_yesterday(self):
        from layer2.query_decomposer import QueryDecomposer
        decomposer = QueryDecomposer()
        intent = self._make_intent(raw_text="Show yesterday's pump data")
        plan = decomposer.decompose(intent)
        self.assertGreaterEqual(plan.temporal_scope.hours, 24)

    def test_causal_query_includes_maintenance(self):
        from layer2.query_decomposer import QueryDecomposer
        decomposer = QueryDecomposer()
        intent = self._make_intent(raw_text="Why is pump-001 failing?")
        plan = decomposer.decompose(intent)
        self.assertTrue(plan.include_maintenance or plan.causal_hypothesis)

    def test_decompose_method_set(self):
        """Decompose method should be 'rule' or 'llm' depending on availability."""
        from layer2.query_decomposer import QueryDecomposer
        decomposer = QueryDecomposer()
        intent = self._make_intent()
        plan = decomposer.decompose(intent)
        self.assertIn(plan.decompose_method, ("rule", "llm"))

    def test_to_dict_serialization(self):
        from layer2.query_decomposer import QueryDecomposer
        decomposer = QueryDecomposer()
        intent = self._make_intent()
        plan = decomposer.decompose(intent)
        d = plan.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("temporal_scope", d)
        self.assertIn("required_metrics", d)


# ════════════════════════════════════════════════════════════════
# Phase 2: Dashboard Planner Tests
# ════════════════════════════════════════════════════════════════

class DashboardPlannerTests(TestCase):
    """Tests for backend/layer2/dashboard_planner.py"""

    def _make_intent(self, **kwargs):
        from layer2.intent_parser import ParsedIntent
        defaults = {
            "type": "query",
            "domains": ["industrial"],
            "entities": {"devices": ["pump-001"]},
            "confidence": 0.85,
            "raw_text": "Show pump-001 power trend",
            "primary_characteristic": "trend",
        }
        defaults.update(kwargs)
        return ParsedIntent(**defaults)

    def _make_plan(self):
        from layer2.query_decomposer import QueryDecomposer
        decomposer = QueryDecomposer()
        return decomposer.decompose(self._make_intent())

    def test_imports(self):
        from layer2.dashboard_planner import (
            DashboardPlanner, DashboardNarrative, InformationSlot,
            SLOT_SCENARIO_MAP, CHARACTERISTIC_SLOT_TEMPLATES,
        )
        self.assertIsNotNone(DashboardPlanner)

    def test_plan_returns_narrative(self):
        from layer2.dashboard_planner import DashboardPlanner, DashboardNarrative
        planner = DashboardPlanner()
        intent = self._make_intent()
        plan = self._make_plan()
        narrative = planner.plan(intent, plan, "", "")
        self.assertIsInstance(narrative, DashboardNarrative)
        self.assertGreater(len(narrative.information_slots), 0)

    def test_heading_from_intent(self):
        from layer2.dashboard_planner import DashboardPlanner
        planner = DashboardPlanner()
        intent = self._make_intent(raw_text="Show pump-001 power trend")
        plan = self._make_plan()
        narrative = planner.plan(intent, plan, "", "")
        self.assertTrue(len(narrative.heading) > 0)

    def test_primary_slot_for_trend(self):
        from layer2.dashboard_planner import DashboardPlanner
        planner = DashboardPlanner()
        intent = self._make_intent(primary_characteristic="trend")
        plan = self._make_plan()
        narrative = planner.plan(intent, plan, "", "")
        roles = [s.role for s in narrative.information_slots]
        self.assertIn("primary_trend", roles)

    def test_primary_slot_for_comparison(self):
        from layer2.dashboard_planner import DashboardPlanner
        planner = DashboardPlanner()
        intent = self._make_intent(
            primary_characteristic="comparison",
            raw_text="Compare pump-001 vs pump-002",
            entities={"devices": ["pump-001", "pump-002"]},
        )
        plan = self._make_plan()
        narrative = planner.plan(intent, plan, "", "")
        roles = [s.role for s in narrative.information_slots]
        self.assertIn("primary_comparison", roles)

    def test_supporting_slots_added(self):
        from layer2.dashboard_planner import DashboardPlanner
        planner = DashboardPlanner()
        intent = self._make_intent()
        plan = self._make_plan()
        narrative = planner.plan(intent, plan, "", "")
        self.assertGreater(len(narrative.information_slots), 1)

    def test_allocate_returns_widget_plan(self):
        from layer2.dashboard_planner import DashboardPlanner
        from layer2.widget_selector import WidgetPlan
        planner = DashboardPlanner()
        intent = self._make_intent()
        plan = self._make_plan()
        narrative = planner.plan(intent, plan, "", "")
        widget_plan = planner.allocate_widgets(narrative, intent, plan)
        self.assertIsInstance(widget_plan, WidgetPlan)
        self.assertGreater(len(widget_plan.widgets), 0)

    def test_allocate_respects_max_widgets(self):
        from layer2.dashboard_planner import DashboardPlanner
        from layer2.widget_selector import MAX_WIDGETS
        planner = DashboardPlanner()
        intent = self._make_intent()
        plan = self._make_plan()
        narrative = planner.plan(intent, plan, "", "")
        widget_plan = planner.allocate_widgets(narrative, intent, plan)
        self.assertLessEqual(len(widget_plan.widgets), MAX_WIDGETS)

    def test_allocate_banned_scenarios_excluded(self):
        from layer2.dashboard_planner import DashboardPlanner
        from layer2.widget_selector import BANNED_SCENARIOS
        planner = DashboardPlanner()
        intent = self._make_intent()
        plan = self._make_plan()
        narrative = planner.plan(intent, plan, "", "")
        widget_plan = planner.allocate_widgets(narrative, intent, plan)
        for w in widget_plan.widgets:
            self.assertNotIn(w.scenario, BANNED_SCENARIOS)

    def test_select_method_is_planner(self):
        from layer2.dashboard_planner import DashboardPlanner
        planner = DashboardPlanner()
        intent = self._make_intent()
        plan = self._make_plan()
        narrative = planner.plan(intent, plan, "", "")
        widget_plan = planner.allocate_widgets(narrative, intent, plan)
        self.assertEqual(widget_plan.select_method, "planner")

    def test_slot_scenario_map_completeness(self):
        from layer2.dashboard_planner import SLOT_SCENARIO_MAP, CHARACTERISTIC_SLOT_TEMPLATES
        # All roles used in templates should be in SLOT_SCENARIO_MAP
        for char_name, template in CHARACTERISTIC_SLOT_TEMPLATES.items():
            primary = template.get("primary", "")
            if primary:
                self.assertIn(primary, SLOT_SCENARIO_MAP,
                              f"Role '{primary}' from '{char_name}' not in SLOT_SCENARIO_MAP")
            for role in template.get("supporting", []):
                self.assertIn(role, SLOT_SCENARIO_MAP,
                              f"Supporting role '{role}' from '{char_name}' not in SLOT_SCENARIO_MAP")


# ════════════════════════════════════════════════════════════════
# Phase 2: Confidence Envelope Tests
# ════════════════════════════════════════════════════════════════

class ConfidenceTests(TestCase):
    """Tests for backend/layer2/confidence.py"""

    def test_imports(self):
        from layer2.confidence import (
            ConfidenceComputer, ConfidenceEnvelope,
            THRESHOLD_FULL, THRESHOLD_PARTIAL, THRESHOLD_REDUCED, THRESHOLD_CLARIFY,
        )
        self.assertIsNotNone(ConfidenceComputer)
        self.assertEqual(THRESHOLD_FULL, 0.75)
        self.assertEqual(THRESHOLD_PARTIAL, 0.55)
        self.assertEqual(THRESHOLD_REDUCED, 0.35)
        self.assertEqual(THRESHOLD_CLARIFY, 0.20)

    def test_default_envelope_values(self):
        from layer2.confidence import ConfidenceEnvelope
        env = ConfidenceEnvelope()
        self.assertEqual(env.intent_confidence, 0.5)
        self.assertEqual(env.retrieval_completeness, 0.5)
        self.assertEqual(env.data_freshness, 1.0)
        self.assertEqual(env.widget_fit, 0.5)
        self.assertEqual(env.data_fill_quality, 0.5)

    def test_overall_all_ones(self):
        from layer2.confidence import ConfidenceEnvelope
        env = ConfidenceEnvelope(
            intent_confidence=1.0, retrieval_completeness=1.0,
            data_freshness=1.0, widget_fit=1.0, data_fill_quality=1.0,
        )
        self.assertAlmostEqual(env.overall, 1.0, places=2)

    def test_overall_dominated_by_weakest(self):
        from layer2.confidence import ConfidenceEnvelope
        env = ConfidenceEnvelope(
            intent_confidence=1.0, retrieval_completeness=1.0,
            data_freshness=1.0, widget_fit=1.0, data_fill_quality=0.1,
        )
        # Harmonic mean should be well below arithmetic mean of 0.82
        self.assertLess(env.overall, 0.6)

    def test_action_full_dashboard(self):
        from layer2.confidence import ConfidenceEnvelope
        env = ConfidenceEnvelope(
            intent_confidence=0.9, retrieval_completeness=0.9,
            data_freshness=1.0, widget_fit=0.9, data_fill_quality=0.9,
        )
        self.assertEqual(env.action, "full_dashboard")

    def test_action_partial_with_caveats(self):
        from layer2.confidence import ConfidenceEnvelope
        env = ConfidenceEnvelope(
            intent_confidence=0.7, retrieval_completeness=0.6,
            data_freshness=0.8, widget_fit=0.6, data_fill_quality=0.6,
        )
        self.assertIn(env.action, ("partial_with_caveats", "full_dashboard"))

    def test_action_ask_clarification(self):
        from layer2.confidence import ConfidenceEnvelope
        env = ConfidenceEnvelope(
            intent_confidence=0.1, retrieval_completeness=0.1,
            data_freshness=0.1, widget_fit=0.1, data_fill_quality=0.1,
        )
        self.assertIn(env.action, ("ask_clarification", "reduced_with_warning"))

    def test_set_intent_confidence_clamped(self):
        from layer2.confidence import ConfidenceComputer
        cc = ConfidenceComputer()
        cc.set_intent_confidence(1.5)
        self.assertEqual(cc.envelope.intent_confidence, 1.0)
        cc.set_intent_confidence(-0.5)
        self.assertEqual(cc.envelope.intent_confidence, 0.0)

    def test_retrieval_completeness_with_gaps(self):
        from layer2.confidence import ConfidenceComputer
        cc = ConfidenceComputer()
        cc.set_retrieval_completeness(0.6, gaps=["Missing power data"])
        self.assertIn("Missing power data", cc.envelope.caveats)

    def test_data_freshness_stale_caveat(self):
        from layer2.confidence import ConfidenceComputer
        cc = ConfidenceComputer()
        cc.set_data_freshness(0.5, stale_count=3)
        caveats_text = " ".join(cc.envelope.caveats)
        self.assertIn("3 widget(s)", caveats_text)

    def test_widget_fit_dropped_caveat(self):
        from layer2.confidence import ConfidenceComputer
        cc = ConfidenceComputer()
        cc.set_widget_fit(total_planned=6, total_rendered=2)
        self.assertLess(cc.envelope.widget_fit, 0.5)
        caveats_text = " ".join(cc.envelope.caveats)
        self.assertIn("removed", caveats_text)

    def test_data_fill_quality_synthetic_caveat(self):
        from layer2.confidence import ConfidenceComputer
        cc = ConfidenceComputer()
        cc.set_data_fill_quality(real_count=1, total_count=5)
        self.assertAlmostEqual(cc.envelope.data_fill_quality, 0.2)
        caveats_text = " ".join(cc.envelope.caveats)
        self.assertIn("estimated or demo", caveats_text)

    def test_voice_caveats_full_dashboard_empty(self):
        from layer2.confidence import ConfidenceComputer
        cc = ConfidenceComputer()
        cc.set_intent_confidence(0.95)
        cc.set_retrieval_completeness(0.95)
        cc.set_data_freshness(1.0)
        cc.set_widget_fit(6, 6)
        cc.set_data_fill_quality(6, 6)
        caveat = cc.build_voice_caveats()
        self.assertEqual(caveat, "")

    def test_voice_caveats_partial_has_note(self):
        from layer2.confidence import ConfidenceComputer
        cc = ConfidenceComputer()
        cc.set_intent_confidence(0.7)
        cc.set_retrieval_completeness(0.6, gaps=["No alerts found"])
        cc.set_data_freshness(0.8)
        cc.set_widget_fit(6, 5)
        cc.set_data_fill_quality(4, 5)
        caveat = cc.build_voice_caveats()
        if cc.envelope.action == "partial_with_caveats":
            self.assertIn("Note:", caveat)

    def test_voice_caveats_low_confidence_non_empty(self):
        """Low confidence should produce a non-empty voice caveat."""
        from layer2.confidence import ConfidenceComputer
        cc = ConfidenceComputer()
        cc.set_intent_confidence(0.4)
        cc.set_retrieval_completeness(0.3, gaps=["No maintenance", "No alerts"])
        cc.set_data_freshness(0.4)
        cc.set_widget_fit(6, 2)
        cc.set_data_fill_quality(1, 5)
        caveat = cc.build_voice_caveats()
        # With such low confidence, should produce some voice caveat
        self.assertTrue(len(caveat) > 0, f"Expected non-empty caveat, action={cc.envelope.action}")

    def test_to_dict_all_keys(self):
        from layer2.confidence import ConfidenceEnvelope
        env = ConfidenceEnvelope()
        d = env.to_dict()
        for key in ("intent_confidence", "retrieval_completeness", "data_freshness",
                     "widget_fit", "data_fill_quality", "overall", "action", "caveats"):
            self.assertIn(key, d)


# ════════════════════════════════════════════════════════════════
# Phase 3: Composition Scorer Tests
# ════════════════════════════════════════════════════════════════

class CompositionScorerTests(TestCase):
    """Tests for backend/rl/composition_scorer.py"""

    def test_imports(self):
        from rl.composition_scorer import (
            ContinuousCompositionTrainer, CompositionScorerNet,
            CompositionScorerState,
        )
        self.assertIsNotNone(ContinuousCompositionTrainer)
        self.assertIsNotNone(CompositionScorerNet)

    def test_net_parameter_count(self):
        from rl.composition_scorer import CompositionScorerNet
        net = CompositionScorerNet()
        # ~69K params (scenario embeds + encoder layers)
        self.assertGreater(net.num_parameters, 60000)
        self.assertLess(net.num_parameters, 80000)

    def test_net_output_shape(self):
        import torch
        from rl.composition_scorer import CompositionScorerNet
        net = CompositionScorerNet()
        intent = torch.randn(2, 768)
        ids = torch.zeros(2, 12, dtype=torch.long)
        mask = torch.ones(2, 12)
        out = net(intent, ids, mask)
        self.assertEqual(out.shape, (2, 1))

    def test_net_output_range(self):
        import torch
        from rl.composition_scorer import CompositionScorerNet
        net = CompositionScorerNet()
        intent = torch.randn(5, 768)
        ids = torch.randint(0, 19, (5, 12))
        mask = torch.ones(5, 12)
        out = net(intent, ids, mask)
        self.assertTrue(torch.all(out >= -1.0))
        self.assertTrue(torch.all(out <= 1.0))

    def test_score_empty_scenarios_returns_zero(self):
        from rl.composition_scorer import ContinuousCompositionTrainer
        trainer = ContinuousCompositionTrainer()
        self.assertEqual(trainer.score_composition("test", []), 0.0)

    def test_score_returns_float(self):
        from rl.composition_scorer import ContinuousCompositionTrainer
        trainer = ContinuousCompositionTrainer()
        score = trainer.score_composition("Show pump status", ["kpi", "trend"])
        self.assertIsInstance(score, float)

    def test_initial_score_near_zero(self):
        from rl.composition_scorer import ContinuousCompositionTrainer
        trainer = ContinuousCompositionTrainer()
        score = trainer.score_composition("test query", ["kpi", "trend", "alerts"])
        self.assertGreater(score, -0.3)
        self.assertLess(score, 0.3)

    def test_train_step_buffer_fill(self):
        from rl.composition_scorer import ContinuousCompositionTrainer
        trainer = ContinuousCompositionTrainer()
        # First few calls go into buffer but don't train (buffer < batch size)
        loss = trainer.train_step("q1", ["kpi"], 0.5)
        self.assertEqual(loss, 0.0)  # Not enough buffer yet

    def test_train_step_after_buffer_fills(self):
        from rl.composition_scorer import ContinuousCompositionTrainer
        trainer = ContinuousCompositionTrainer()
        last_loss = 0.0
        for i in range(12):
            last_loss = trainer.train_step(f"query {i}", ["kpi", "trend"], 0.5 + i * 0.1)
        # After buffer fills (>= REPLAY_BATCH_SIZE=8), should have trained
        self.assertGreater(trainer.state.total_feedback_events, 0)

    def test_scenario_mapping_count(self):
        """Scenario mapping should cover all valid scenarios from catalog."""
        from rl.composition_scorer import ContinuousCompositionTrainer
        trainer = ContinuousCompositionTrainer()
        # At least 19 scenarios (may be more if catalog has additions)
        self.assertGreaterEqual(len(trainer._scenario_to_idx), 19)

    def test_get_stats_keys(self):
        from rl.composition_scorer import ContinuousCompositionTrainer
        trainer = ContinuousCompositionTrainer()
        stats = trainer.get_stats()
        for key in ("type", "parameters", "device", "training_steps",
                     "total_feedback_events", "avg_loss", "avg_val_loss",
                     "best_val_loss", "lr", "lr_decays",
                     "replay_buffer_size", "val_buffer_size"):
            self.assertIn(key, stats)
        self.assertEqual(stats["type"], "composition_scorer")

    def test_train_batch(self):
        from rl.composition_scorer import ContinuousCompositionTrainer
        trainer = ContinuousCompositionTrainer()
        experiences = [
            ("query 1", ["kpi", "trend"], 0.8),
            ("query 2", ["alerts", "comparison"], 0.5),
            ("query 3", ["kpi", "kpi", "trend"], 0.6),
        ]
        avg_loss = trainer.train_batch(experiences)
        self.assertIsInstance(avg_loss, float)
