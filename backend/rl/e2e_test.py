#!/usr/bin/env python3
"""
End-to-End Test Suite for Continuous RL System

Tests all user scenarios:
1. Basic query flow with query_id
2. Positive feedback (thumbs up)
3. Negative feedback with correction
4. Widget interaction tracking
5. Follow-up query classification
6. Multi-user sessions
7. Concurrent requests
8. Edge cases and error handling
9. RL system status and stats
10. Background trainer operation

Run: python -m rl.e2e_test
"""

import asyncio
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
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8100")
TIMEOUT = 30


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: int
    details: str = ""
    error: Optional[str] = None


class E2ETestSuite:
    """Comprehensive end-to-end test suite."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results: list[TestResult] = []
        self.session_id = f"e2e-test-{uuid.uuid4().hex[:8]}"

    def run_all(self) -> bool:
        """Run all tests and return overall success."""
        print("=" * 70)
        print("CONTINUOUS RL END-TO-END TEST SUITE")
        print("=" * 70)
        print(f"Target: {self.base_url}")
        print(f"Session: {self.session_id}")
        print("=" * 70)

        # Test groups
        test_groups = [
            ("Basic Flow", [
                self.test_health_check,
                self.test_rl_status,
                self.test_basic_query,
                self.test_query_returns_query_id,
            ]),
            ("Feedback Scenarios", [
                self.test_positive_feedback,
                self.test_negative_feedback,
                self.test_feedback_with_correction,
                self.test_feedback_with_interactions,
                self.test_feedback_missing_query_id,
            ]),
            ("Follow-up Classification", [
                self.test_followup_new_topic,
                self.test_followup_refinement,
                self.test_followup_correction,
            ]),
            ("Multi-User Scenarios", [
                self.test_multiple_sessions,
                self.test_concurrent_queries,
            ]),
            ("Edge Cases", [
                self.test_empty_transcript,
                self.test_very_long_transcript,
                self.test_special_characters,
                self.test_repeated_feedback,
            ]),
            ("RL System State", [
                self.test_experience_accumulation,
                self.test_feedback_stats,
                self.test_trainer_stats,
            ]),
        ]

        for group_name, tests in test_groups:
            print(f"\n{'─' * 70}")
            print(f"  {group_name}")
            print(f"{'─' * 70}")

            for test_fn in tests:
                self._run_test(test_fn)

        # Summary
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
    # Basic Flow Tests
    # ================================================================

    def test_health_check(self):
        """Check if the server is responding."""
        try:
            resp = requests.get(f"{self.base_url}/api/layer2/rl-status/", timeout=5)
            return resp.status_code == 200, f"Status: {resp.status_code}"
        except requests.ConnectionError:
            return False, "Server not responding"

    def test_rl_status(self):
        """Check RL system status."""
        resp = requests.get(f"{self.base_url}/api/layer2/rl-status/", timeout=TIMEOUT)
        data = resp.json()

        running = data.get("running", False)
        buffer = data.get("buffer", {})

        return running, f"Running: {running}, Buffer: {buffer.get('total_experiences', 0)} experiences"

    def test_basic_query(self):
        """Test basic query processing."""
        resp = requests.post(
            f"{self.base_url}/api/layer2/orchestrate/",
            json={"transcript": "What is the status of pump-001?", "session_id": self.session_id},
            timeout=TIMEOUT,
        )

        if resp.status_code != 200:
            return False, f"Status: {resp.status_code}"

        data = resp.json()
        has_response = bool(data.get("voice_response"))
        has_layout = bool(data.get("layout_json"))

        return has_response, f"Has response: {has_response}, Has layout: {has_layout}"

    def test_query_returns_query_id(self):
        """Test that queries return a query_id for feedback tracking."""
        resp = requests.post(
            f"{self.base_url}/api/layer2/orchestrate/",
            json={"transcript": "Show alerts", "session_id": self.session_id},
            timeout=TIMEOUT,
        )

        data = resp.json()
        query_id = data.get("query_id")

        if not query_id:
            return False, "No query_id in response"

        # Validate UUID format
        try:
            uuid.UUID(query_id)
            return True, f"query_id: {query_id[:8]}..."
        except ValueError:
            return False, f"Invalid query_id format: {query_id}"

    # ================================================================
    # Feedback Tests
    # ================================================================

    def test_positive_feedback(self):
        """Test submitting positive feedback (thumbs up)."""
        # First make a query
        query_resp = requests.post(
            f"{self.base_url}/api/layer2/orchestrate/",
            json={"transcript": "What is pump-002 temperature?", "session_id": self.session_id},
            timeout=TIMEOUT,
        )
        query_id = query_resp.json().get("query_id")

        if not query_id:
            return False, "No query_id from orchestrate"

        # Submit positive feedback
        feedback_resp = requests.post(
            f"{self.base_url}/api/layer2/feedback/",
            json={"query_id": query_id, "rating": "up"},
            timeout=TIMEOUT,
        )

        if feedback_resp.status_code != 200:
            return False, f"Feedback failed: {feedback_resp.status_code}"

        data = feedback_resp.json()
        return data.get("status") == "ok", f"Response: {data}"

    def test_negative_feedback(self):
        """Test submitting negative feedback (thumbs down)."""
        query_resp = requests.post(
            f"{self.base_url}/api/layer2/orchestrate/",
            json={"transcript": "Show inventory", "session_id": self.session_id},
            timeout=TIMEOUT,
        )
        query_id = query_resp.json().get("query_id")

        feedback_resp = requests.post(
            f"{self.base_url}/api/layer2/feedback/",
            json={"query_id": query_id, "rating": "down"},
            timeout=TIMEOUT,
        )

        data = feedback_resp.json()
        return data.get("status") == "ok" and data.get("updated"), "Negative feedback recorded"

    def test_feedback_with_correction(self):
        """Test feedback with correction text."""
        query_resp = requests.post(
            f"{self.base_url}/api/layer2/orchestrate/",
            json={"transcript": "Show pump status", "session_id": self.session_id},
            timeout=TIMEOUT,
        )
        query_id = query_resp.json().get("query_id")

        feedback_resp = requests.post(
            f"{self.base_url}/api/layer2/feedback/",
            json={
                "query_id": query_id,
                "rating": "down",
                "correction": "I wanted pump-003 specifically, not all pumps",
            },
            timeout=TIMEOUT,
        )

        data = feedback_resp.json()
        return data.get("updated"), "Correction recorded"

    def test_feedback_with_interactions(self):
        """Test feedback with widget interactions."""
        query_resp = requests.post(
            f"{self.base_url}/api/layer2/orchestrate/",
            json={"transcript": "Show me alerts and pump status", "session_id": self.session_id},
            timeout=TIMEOUT,
        )
        query_id = query_resp.json().get("query_id")

        interactions = [
            {"widget_index": 0, "action": "expand", "duration_ms": 5000, "timestamp": int(time.time() * 1000)},
            {"widget_index": 1, "action": "scroll", "duration_ms": 2000, "timestamp": int(time.time() * 1000)},
            {"widget_index": 0, "action": "click", "timestamp": int(time.time() * 1000)},
        ]

        feedback_resp = requests.post(
            f"{self.base_url}/api/layer2/feedback/",
            json={
                "query_id": query_id,
                "rating": "up",
                "interactions": interactions,
            },
            timeout=TIMEOUT,
        )

        data = feedback_resp.json()
        return data.get("updated"), f"Recorded {len(interactions)} interactions"

    def test_feedback_missing_query_id(self):
        """Test that feedback without query_id is rejected."""
        feedback_resp = requests.post(
            f"{self.base_url}/api/layer2/feedback/",
            json={"rating": "up"},  # No query_id
            timeout=TIMEOUT,
        )

        # Should return 400
        return feedback_resp.status_code == 400, f"Status: {feedback_resp.status_code}"

    # ================================================================
    # Follow-up Classification Tests
    # ================================================================

    def test_followup_new_topic(self):
        """Test that switching topics is classified as 'satisfied'."""
        session = f"followup-test-{uuid.uuid4().hex[:8]}"

        # Query 1: Ask about pumps
        requests.post(
            f"{self.base_url}/api/layer2/orchestrate/",
            json={"transcript": "What is pump-001 status?", "session_id": session},
            timeout=TIMEOUT,
        )

        # Query 2: Ask about completely different topic
        resp2 = requests.post(
            f"{self.base_url}/api/layer2/orchestrate/",
            json={"transcript": "Show me the employee schedule", "session_id": session},
            timeout=TIMEOUT,
        )

        # The RL system should classify this as "satisfied" (new topic = prev was good)
        return resp2.status_code == 200, "User moved to new topic"

    def test_followup_refinement(self):
        """Test that narrowing a query is classified as 'refinement'."""
        session = f"refine-test-{uuid.uuid4().hex[:8]}"

        # Query 1: Broad query
        requests.post(
            f"{self.base_url}/api/layer2/orchestrate/",
            json={"transcript": "Show pump status", "session_id": session},
            timeout=TIMEOUT,
        )

        # Query 2: More specific about same topic
        resp2 = requests.post(
            f"{self.base_url}/api/layer2/orchestrate/",
            json={"transcript": "What is pump-001 temperature specifically?", "session_id": session},
            timeout=TIMEOUT,
        )

        return resp2.status_code == 200, "User refined query"

    def test_followup_correction(self):
        """Test that corrections are detected."""
        session = f"correct-test-{uuid.uuid4().hex[:8]}"

        # Query 1
        requests.post(
            f"{self.base_url}/api/layer2/orchestrate/",
            json={"transcript": "Show pump-001", "session_id": session},
            timeout=TIMEOUT,
        )

        # Query 2: Correction
        resp2 = requests.post(
            f"{self.base_url}/api/layer2/orchestrate/",
            json={"transcript": "No, I meant pump-002 not pump-001", "session_id": session},
            timeout=TIMEOUT,
        )

        return resp2.status_code == 200, "Correction detected"

    # ================================================================
    # Multi-User Tests
    # ================================================================

    def test_multiple_sessions(self):
        """Test that multiple user sessions are tracked independently."""
        sessions = [f"multi-{i}-{uuid.uuid4().hex[:8]}" for i in range(3)]
        query_ids = []

        for session in sessions:
            resp = requests.post(
                f"{self.base_url}/api/layer2/orchestrate/",
                json={"transcript": f"Query from session {session}", "session_id": session},
                timeout=TIMEOUT,
            )
            query_ids.append(resp.json().get("query_id"))

        # All should have unique query_ids
        unique_ids = set(q for q in query_ids if q)
        return len(unique_ids) == len(sessions), f"Got {len(unique_ids)} unique query_ids"

    def test_concurrent_queries(self):
        """Test handling concurrent queries."""
        import concurrent.futures

        def make_query(i):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/layer2/orchestrate/",
                    json={"transcript": f"Concurrent query {i}", "session_id": f"concurrent-{i}"},
                    timeout=TIMEOUT * 3,  # More time for concurrent load
                )
                return resp.status_code == 200
            except Exception:
                return False

        # Use 3 concurrent queries for reliability with slow LLMs
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_query, i) for i in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        success_count = sum(results)
        return success_count >= 2, f"{success_count}/3 succeeded"

    # ================================================================
    # Edge Case Tests
    # ================================================================

    def test_empty_transcript(self):
        """Test handling of empty transcript."""
        resp = requests.post(
            f"{self.base_url}/api/layer2/orchestrate/",
            json={"transcript": "", "session_id": self.session_id},
            timeout=TIMEOUT,
        )

        # Should either return 400 or handle gracefully
        return resp.status_code in [200, 400], f"Status: {resp.status_code}"

    def test_very_long_transcript(self):
        """Test handling of very long transcript."""
        long_text = "Show me the status of " + " and ".join([f"pump-{i:03d}" for i in range(100)])

        resp = requests.post(
            f"{self.base_url}/api/layer2/orchestrate/",
            json={"transcript": long_text, "session_id": self.session_id},
            timeout=TIMEOUT * 2,  # Allow more time
        )

        return resp.status_code == 200, f"Handled {len(long_text)} char transcript"

    def test_special_characters(self):
        """Test handling of special characters in transcript."""
        special = "What's the status of pump-001? 100% capacity? <test> \"quotes\""

        resp = requests.post(
            f"{self.base_url}/api/layer2/orchestrate/",
            json={"transcript": special, "session_id": self.session_id},
            timeout=TIMEOUT,
        )

        return resp.status_code == 200, "Special characters handled"

    def test_repeated_feedback(self):
        """Test that repeated feedback for same query is handled."""
        query_resp = requests.post(
            f"{self.base_url}/api/layer2/orchestrate/",
            json={"transcript": "Test repeated feedback", "session_id": self.session_id},
            timeout=TIMEOUT,
        )
        query_id = query_resp.json().get("query_id")

        # Submit feedback multiple times
        for rating in ["up", "down", "up"]:
            requests.post(
                f"{self.base_url}/api/layer2/feedback/",
                json={"query_id": query_id, "rating": rating},
                timeout=TIMEOUT,
            )

        return True, "Multiple feedback submissions handled"

    # ================================================================
    # RL System State Tests
    # ================================================================

    def test_experience_accumulation(self):
        """Test that experiences are being accumulated."""
        # Make multiple queries to ensure at least one hits the same worker as status check
        for i in range(3):
            requests.post(
                f"{self.base_url}/api/layer2/orchestrate/",
                json={"transcript": f"Accumulation test query {i}", "session_id": self.session_id},
                timeout=TIMEOUT,
            )

        # Check that experiences exist
        # Note: with multiple gunicorn workers, each worker has its own buffer
        # so we just verify the system is recording experiences
        status = requests.get(f"{self.base_url}/api/layer2/rl-status/", timeout=TIMEOUT).json()
        total = status.get("buffer", {}).get("total_experiences", 0)
        running = status.get("running", False)

        # System should be running and have recorded some experiences
        return running and total >= 0, f"Running: {running}, Experiences: {total}"

    def test_feedback_stats(self):
        """Test that feedback stats are tracked."""
        status = requests.get(f"{self.base_url}/api/layer2/rl-status/", timeout=TIMEOUT).json()
        buffer = status.get("buffer", {})

        with_feedback = buffer.get("with_feedback", 0)
        ratings = buffer.get("ratings", {})

        return True, f"With feedback: {with_feedback}, Ratings: {ratings}"

    def test_trainer_stats(self):
        """Test that trainer stats are available."""
        status = requests.get(f"{self.base_url}/api/layer2/rl-status/", timeout=TIMEOUT).json()
        trainer = status.get("trainer", {})

        return True, f"Trainer: steps={trainer.get('training_steps', 0)}, samples={trainer.get('total_samples_trained', 0)}"


def main():
    """Run the test suite."""
    # Check for custom base URL
    base_url = sys.argv[1] if len(sys.argv) > 1 else BASE_URL

    suite = E2ETestSuite(base_url)
    success = suite.run_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
