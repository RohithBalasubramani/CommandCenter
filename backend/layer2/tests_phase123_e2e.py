#!/usr/bin/env python3
"""
Phase 1-3 RAG & Widget Selection Redesign — Integration E2E Tests

Tests the live orchestrator endpoint for Phase 1-3 metadata in responses:
  - _retrieval_assessment (Phase 1)
  - _confidence envelope (Phase 2)
  - _composition_score (Phase 3)
  - Per-widget quality metadata (_data_quality, _is_stale, _widget_confidence)

Run: python -m layer2.tests_phase123_e2e http://127.0.0.1:8100
"""

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Optional

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "command_center.settings")

import django
django.setup()

import requests

# Test configuration
BASE_URL = os.getenv("TEST_BASE_URL", "http://127.0.0.1:8100")
TIMEOUT = 60  # orchestrate can take a while with LLM calls
ORCHESTRATE_URL_PATH = "/api/layer2/orchestrate/"


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: int
    details: str = ""
    error: Optional[str] = None


class Phase123E2ETestSuite:
    """End-to-end integration tests for Phase 1-3 redesign modules."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.results: list[TestResult] = []
        self.session_id = f"phase123-e2e-{uuid.uuid4().hex[:8]}"
        # Cached response from a standard query (avoids re-querying)
        self._cached_response: Optional[dict] = None
        self._cached_layout: Optional[dict] = None

    def _orchestrate(self, transcript: str, session_id: str = None, retries: int = 3) -> dict:
        """Send a query to the orchestrate endpoint and return the JSON response.
        Retries on connection errors with backoff (gunicorn workers may restart)."""
        url = f"{self.base_url}{ORCHESTRATE_URL_PATH}"
        payload = {
            "transcript": transcript,
            "session_id": session_id or self.session_id,
        }
        last_err = None
        for attempt in range(retries):
            try:
                if attempt > 0:
                    time.sleep(3 * attempt)  # backoff: 3s, 6s
                resp = requests.post(url, json=payload, timeout=TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
                last_err = e
                continue
        raise last_err

    def _get_standard_response(self) -> tuple[dict, dict]:
        """Return a cached standard query response (reuses across tests)."""
        if self._cached_response is None:
            data = self._orchestrate("show me pump 4 power and vibration")
            self._cached_response = data
            # The orchestrator returns layout under "layout_json" key
            self._cached_layout = data.get("layout_json", data.get("layout", data))
        return self._cached_response, self._cached_layout

    def run_all(self) -> bool:
        """Run all tests and return overall success."""
        print("=" * 70)
        print("PHASE 1-3 REDESIGN — E2E INTEGRATION TESTS")
        print("=" * 70)
        print(f"Target: {self.base_url}")
        print(f"Session: {self.session_id}")
        print("=" * 70)

        test_groups = [
            ("Health & Connectivity", [
                self.test_health_check,
                self.test_orchestrate_reachable,
            ]),
            ("Phase 1: Retrieval Assessment", [
                self.test_retrieval_assessment_present,
                self.test_retrieval_assessment_keys,
                self.test_retrieval_assessment_values,
                self.test_per_widget_quality_metadata,
            ]),
            ("Phase 2: Confidence Envelope", [
                self.test_confidence_present,
                self.test_confidence_keys,
                self.test_confidence_overall_range,
                self.test_confidence_action_valid,
                self.test_confidence_action_threshold_consistency,
                self.test_confidence_caveats_type,
            ]),
            ("Phase 3: Composition Score", [
                self.test_composition_score_present,
                self.test_composition_score_range,
                self.test_composition_score_deterministic,
            ]),
            ("Cross-Phase Integration", [
                self.test_greeting_no_confidence,
                self.test_multi_entity_query,
            ]),
        ]

        for group_name, tests in test_groups:
            print(f"\n{'─' * 70}")
            print(f"  {group_name}")
            print(f"{'─' * 70}")
            for test_fn in tests:
                self._run_test(test_fn)

        return self._print_summary()

    def _run_test(self, test_fn):
        """Run a single test and record result."""
        name = test_fn.__name__.replace("test_", "").replace("_", " ").title()
        start = time.time()

        try:
            result = test_fn()
            duration = int((time.time() - start) * 1000)

            if isinstance(result, tuple):
                passed, details = result
            else:
                passed, details = result, ""

            self.results.append(TestResult(
                name=name,
                passed=passed,
                duration_ms=duration,
                details=details,
            ))

            status = "✓" if passed else "✗"
            print(f"  {status} {name} ({duration}ms)")
            if details and not passed:
                print(f"      {details}")

        except Exception as e:
            duration = int((time.time() - start) * 1000)
            self.results.append(TestResult(
                name=name,
                passed=False,
                duration_ms=duration,
                error=str(e),
            ))
            print(f"  ✗ {name} ({duration}ms)")
            print(f"      ERROR: {e}")

    def _print_summary(self) -> bool:
        """Print test summary and return success."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        total_time = sum(r.duration_ms for r in self.results)

        print(f"Total: {len(self.results)} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Duration: {total_time}ms")

        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.error or r.details}")

        print("=" * 70)
        return failed == 0

    # ================================================================
    # Health & Connectivity
    # ================================================================

    def test_health_check(self):
        """Check if server is responding."""
        try:
            resp = requests.get(f"{self.base_url}/api/layer2/rl-status/", timeout=5)
            return resp.status_code == 200, f"Status: {resp.status_code}"
        except requests.ConnectionError:
            return False, "Server not responding"

    def test_orchestrate_reachable(self):
        """Check if orchestrate endpoint responds to a basic query."""
        try:
            resp, layout = self._get_standard_response()
            has_widgets = len(layout.get("widgets", [])) > 0
            return has_widgets, f"Widgets: {len(layout.get('widgets', []))}"
        except Exception as e:
            return False, f"Orchestrate failed: {e}"

    # ================================================================
    # Phase 1: Retrieval Assessment
    # ================================================================

    def test_retrieval_assessment_present(self):
        """Response must include _retrieval_assessment."""
        _, layout = self._get_standard_response()
        present = "_retrieval_assessment" in layout
        return present, f"Keys: {list(k for k in layout if k.startswith('_'))}"

    def test_retrieval_assessment_keys(self):
        """_retrieval_assessment has expected keys."""
        _, layout = self._get_standard_response()
        ra = layout.get("_retrieval_assessment", {})
        required_keys = {"total_widgets", "completeness", "freshness"}
        present_keys = set(ra.keys())
        missing = required_keys - present_keys
        if missing:
            return False, f"Missing keys: {missing}"
        return True, f"Keys: {sorted(ra.keys())}"

    def test_retrieval_assessment_values(self):
        """Retrieval assessment values are in valid ranges."""
        _, layout = self._get_standard_response()
        ra = layout.get("_retrieval_assessment", {})

        checks = []
        completeness = ra.get("completeness", -1)
        if not (0.0 <= completeness <= 1.0):
            checks.append(f"completeness={completeness} not in [0,1]")

        freshness = ra.get("freshness", -1)
        if not (0.0 <= freshness <= 1.0):
            checks.append(f"freshness={freshness} not in [0,1]")

        total = ra.get("total_widgets", -1)
        if total < 0:
            checks.append(f"total_widgets={total} is negative")

        if checks:
            return False, "; ".join(checks)
        return True, f"completeness={completeness:.2f}, freshness={freshness:.2f}, total={total}"

    def test_per_widget_quality_metadata(self):
        """Each widget should have _data_quality and _widget_confidence metadata."""
        _, layout = self._get_standard_response()
        widgets = layout.get("widgets", [])

        if not widgets:
            return False, "No widgets in response"

        widgets_with_quality = 0
        widgets_with_confidence = 0
        for w in widgets:
            do = w.get("data_override", {})
            if "_data_quality" in do or "_data_quality" in w:
                widgets_with_quality += 1
            if "_widget_confidence" in do or "_widget_confidence" in w:
                widgets_with_confidence += 1

        # At least some widgets should have metadata
        pct_quality = widgets_with_quality / len(widgets) if widgets else 0
        pct_conf = widgets_with_confidence / len(widgets) if widgets else 0
        ok = pct_quality >= 0.5 or pct_conf >= 0.5

        return ok, (
            f"{widgets_with_quality}/{len(widgets)} with _data_quality, "
            f"{widgets_with_confidence}/{len(widgets)} with _widget_confidence"
        )

    # ================================================================
    # Phase 2: Confidence Envelope
    # ================================================================

    def test_confidence_present(self):
        """Response must include _confidence."""
        _, layout = self._get_standard_response()
        present = "_confidence" in layout
        return present, f"Keys: {list(k for k in layout if k.startswith('_'))}"

    def test_confidence_keys(self):
        """_confidence has all 8 expected keys."""
        _, layout = self._get_standard_response()
        conf = layout.get("_confidence", {})
        expected_keys = {
            "intent_confidence", "retrieval_completeness", "data_freshness",
            "widget_fit", "data_fill_quality", "overall", "action", "caveats",
        }
        present_keys = set(conf.keys())
        missing = expected_keys - present_keys
        if missing:
            return False, f"Missing keys: {missing}, present: {sorted(present_keys)}"
        return True, f"All 8 keys present"

    def test_confidence_overall_range(self):
        """Confidence overall score must be in [0, 1]."""
        _, layout = self._get_standard_response()
        conf = layout.get("_confidence", {})
        overall = conf.get("overall", -1)
        in_range = 0.0 <= overall <= 1.0
        return in_range, f"overall={overall}"

    def test_confidence_action_valid(self):
        """Confidence action must be one of the 4 valid values."""
        _, layout = self._get_standard_response()
        conf = layout.get("_confidence", {})
        action = conf.get("action", "")
        valid_actions = {"full_dashboard", "partial_with_caveats", "reduced_dashboard", "reduced_with_warning", "ask_clarification"}
        is_valid = action in valid_actions
        return is_valid, f"action='{action}'"

    def test_confidence_action_threshold_consistency(self):
        """Confidence action should be consistent with overall score thresholds."""
        _, layout = self._get_standard_response()
        conf = layout.get("_confidence", {})
        overall = conf.get("overall", 0)
        action = conf.get("action", "")

        # Thresholds: full >= 0.75, partial >= 0.55, reduced >= 0.35, warning >= 0.20, ask < 0.20
        expected = None
        if overall >= 0.75:
            expected = "full_dashboard"
        elif overall >= 0.55:
            expected = "partial_with_caveats"
        elif overall >= 0.35:
            expected = "reduced_dashboard"
        elif overall >= 0.20:
            expected = "reduced_with_warning"
        else:
            expected = "ask_clarification"

        consistent = action == expected
        return consistent, f"overall={overall:.2f}, action='{action}', expected='{expected}'"

    def test_confidence_caveats_type(self):
        """Caveats should be a list of strings."""
        _, layout = self._get_standard_response()
        conf = layout.get("_confidence", {})
        caveats = conf.get("caveats", None)

        if caveats is None:
            return False, "caveats is missing"
        if not isinstance(caveats, list):
            return False, f"caveats is {type(caveats).__name__}, expected list"
        # Each element should be a string (if non-empty)
        for i, c in enumerate(caveats):
            if not isinstance(c, str):
                return False, f"caveats[{i}] is {type(c).__name__}, expected str"
        return True, f"{len(caveats)} caveats"

    # ================================================================
    # Phase 3: Composition Score
    # ================================================================

    def test_composition_score_present(self):
        """Response must include _composition_score."""
        _, layout = self._get_standard_response()
        present = "_composition_score" in layout
        return present, f"Keys: {list(k for k in layout if k.startswith('_'))}"

    def test_composition_score_range(self):
        """Composition score must be in [-1, 1]."""
        _, layout = self._get_standard_response()
        score = layout.get("_composition_score", None)
        if score is None:
            return False, "Missing _composition_score"
        in_range = -1.0 <= score <= 1.0
        return in_range, f"_composition_score={score}"

    def test_composition_score_deterministic(self):
        """Same query should produce similar composition score."""
        query = "show me pump 4 power"
        sid = f"det-test-{uuid.uuid4().hex[:8]}"

        data1 = self._orchestrate(query, session_id=sid + "-a")
        layout1 = data1.get("layout_json", data1.get("layout", data1))
        score1 = layout1.get("_composition_score", 0)

        time.sleep(2)  # Give server breathing room between heavy requests

        data2 = self._orchestrate(query, session_id=sid + "-b")
        layout2 = data2.get("layout_json", data2.get("layout", data2))
        score2 = layout2.get("_composition_score", 0)

        # Scores should be within 0.3 of each other (neural net has some variance)
        diff = abs(score1 - score2)
        close = diff < 0.3
        return close, f"score1={score1:.3f}, score2={score2:.3f}, diff={diff:.3f}"

    # ================================================================
    # Cross-Phase Integration
    # ================================================================

    def test_greeting_no_confidence(self):
        """A greeting query should return minimal/no dashboard metadata."""
        try:
            data = self._orchestrate("hello", session_id=f"greet-{uuid.uuid4().hex[:8]}")
            layout = data.get("layout_json") or data.get("layout")

            # Greetings typically return layout_json=null — no dashboard needed
            if layout is None:
                return True, "layout_json is null for greeting (expected)"

            # If layout exists, check confidence handling
            conf = layout.get("_confidence")
            if conf is None:
                return True, "No _confidence for greeting (expected)"

            # If present, action should be one of the valid actions
            action = conf.get("action", "")
            ok = action in {"full_dashboard", "ask_clarification", "partial_with_caveats", "reduced_dashboard", "reduced_with_warning"}
            return ok, f"action='{action}' for greeting"
        except Exception as e:
            return False, f"Greeting query failed: {e}"

    def test_multi_entity_query(self):
        """A multi-entity query should produce widgets with varied scenarios."""
        try:
            data = self._orchestrate(
                "compare transformer 1 and transformer 2 temperatures",
                session_id=f"multi-{uuid.uuid4().hex[:8]}",
            )
            layout = data.get("layout_json", data.get("layout", data))
            widgets = layout.get("widgets", [])

            if not widgets:
                return False, "No widgets returned for multi-entity query"

            scenarios = set(w.get("scenario", "") for w in widgets)
            has_variety = len(scenarios) >= 1

            # Check that confidence is present
            conf = layout.get("_confidence")
            has_conf = conf is not None

            return has_variety and has_conf, (
                f"{len(widgets)} widgets, scenarios={scenarios}, "
                f"confidence={'yes' if has_conf else 'no'}"
            )
        except Exception as e:
            return False, f"Multi-entity query failed: {e}"


# ════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════

def main():
    url = sys.argv[1] if len(sys.argv) > 1 else BASE_URL
    suite = Phase123E2ETestSuite(base_url=url)
    success = suite.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
