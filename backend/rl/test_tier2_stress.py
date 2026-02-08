#!/usr/bin/env python3
"""
Unified DPO Stress Tests — Edge Cases That Break in Production (Tier 2)

The kind of issues we found with Tier 1 AFTER deployment:
- Stale data from disk reload
- Wrong confidence values filtering everything out
- Score blending producing NaN/Inf
- Empty voice responses sneaking through
- Serialization losing fields across worker restarts
- DPO pairs with zero reward gap
- Memory growth from unbounded accumulation

Run: cd backend && ./venv/bin/python -m rl.test_tier2_stress
"""

import os
import sys
import time
import json
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "command_center.settings")

import django
django.setup()


class StressRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run(self, name, fn):
        try:
            fn()
            self.passed += 1
            print(f"  ✓ {name}")
        except Exception as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  ✗ {name}")
            print(f"      {e}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"STRESS RESULTS: {self.passed}/{total} passed, {self.failed} failed")
        print(f"{'='*70}")
        if self.errors:
            print("\nFailed:")
            for n, e in self.errors:
                print(f"  - {n}: {e}")
        return self.failed == 0

r = StressRunner()

# ================================================================
print(f"\n{'─'*70}")
print("  Edge Case: Empty / Whitespace Voice Responses")
print(f"{'─'*70}")

def test_empty_voice_no_crash():
    """Empty voice_response doesn't crash any tier."""
    from rl.continuous import ContinuousRL
    rl = ContinuousRL()
    rl.record_experience(query_id="stress-empty-voice", transcript="test", voice_response="")
    rl.update_feedback("stress-empty-voice", rating="up",
                       voice_evaluation_confidence=0.9)
    exp = rl.buffer.get_by_query_id("stress-empty-voice")
    assert exp.voice_response_rating is None, "Empty voice → no rating"

r.run("Empty voice_response no crash", test_empty_voice_no_crash)

def test_whitespace_voice_no_crash():
    """Whitespace-only voice_response treated as empty."""
    from rl.continuous import ContinuousRL
    rl = ContinuousRL()
    rl.record_experience(query_id="stress-ws-voice", transcript="test", voice_response="   \n\t  ")
    rl.update_feedback("stress-ws-voice", rating="up")
    exp = rl.buffer.get_by_query_id("stress-ws-voice")
    # Whitespace-only should still technically be falsy for voice_response check
    # The reward should handle it gracefully
    from rl.reward_signals import RewardSignalAggregator
    agg = RewardSignalAggregator()
    reward = agg._text_quality_reward(exp)
    assert isinstance(reward, float)

r.run("Whitespace-only voice no crash", test_whitespace_voice_no_crash)


# ================================================================
print(f"\n{'─'*70}")
print("  Edge Case: Extreme / Invalid Claude Scores")
print(f"{'─'*70}")

def test_claude_scores_out_of_range():
    """Claude returning scores > 1.0 or < 0.0 doesn't crash."""
    from rl.experience_buffer import Experience
    from rl.reward_signals import RewardSignalAggregator
    agg = RewardSignalAggregator()
    exp = Experience(
        query_id="stress-oob",
        voice_response="Test response.",
        transcript="test",
        voice_dimension_scores_claude={
            "groundedness": 1.5,  # Over 1.0
            "conciseness": -0.3,  # Negative
            "directness": 0.0,
            "specificity": 999.0,  # Way over
            "tts_friendliness": 0.5,
        },
    )
    reward = agg._text_quality_reward(exp)
    assert isinstance(reward, float)
    assert not (reward != reward), "NaN detected"  # NaN check

r.run("Claude scores out of range no crash", test_claude_scores_out_of_range)

def test_claude_scores_none_values():
    """Claude returning None for some dimensions."""
    from rl.experience_buffer import Experience
    from rl.reward_signals import RewardSignalAggregator
    agg = RewardSignalAggregator()
    exp = Experience(
        query_id="stress-none-dims",
        voice_response="Test response.",
        transcript="test",
        voice_dimension_scores_claude={
            "groundedness": None,
            "conciseness": 0.8,
        },
    )
    reward = agg._text_quality_reward(exp)
    assert isinstance(reward, float)

r.run("Claude None dimension values no crash", test_claude_scores_none_values)

def test_claude_empty_dict():
    """Empty Claude scores dict treated as no-op."""
    from rl.experience_buffer import Experience
    from rl.reward_signals import RewardSignalAggregator
    agg = RewardSignalAggregator()
    exp = Experience(
        query_id="stress-empty-dict",
        voice_response="Test response.",
        transcript="test",
        voice_dimension_scores_claude={},
    )
    reward = agg._text_quality_reward(exp)
    assert isinstance(reward, float)

r.run("Empty Claude scores dict handled", test_claude_empty_dict)


# ================================================================
print(f"\n{'─'*70}")
print("  Edge Case: Serialization Survival Across Workers")
print(f"{'─'*70}")

def test_full_experience_json_roundtrip():
    """All voice eval fields survive JSON serialization (worker restart)."""
    from rl.experience_buffer import Experience
    exp = Experience(
        query_id="stress-json-rt",
        transcript="pump status",
        voice_response="Pump 001 at 2400 RPM.",
        voice_response_rating="good",
        voice_quality_scores={"groundedness": 0.9, "total": 0.85},
        voice_evaluation_confidence=0.92,
        voice_evaluation_reasoning="Excellent data citation",
        voice_dimension_scores_claude={"groundedness": 0.95, "conciseness": 0.9},
        evaluation_confidence=0.88,
        user_rating="up",
        computed_reward=1.25,
    )

    # Simulate disk save → load (what happens across gunicorn workers)
    d = exp.to_dict()
    json_str = json.dumps(d)
    d2 = json.loads(json_str)
    exp2 = Experience.from_dict(d2)

    assert exp2.voice_evaluation_confidence == 0.92
    assert exp2.voice_evaluation_reasoning == "Excellent data citation"
    assert exp2.voice_dimension_scores_claude["groundedness"] == 0.95
    assert exp2.voice_response_rating == "good"
    assert exp2.voice_quality_scores["total"] == 0.85
    assert exp2.computed_reward == 1.25

r.run("Full JSON roundtrip all voice fields", test_full_experience_json_roundtrip)

def test_old_schema_experience_loads():
    """Experience from OLD schema (no voice fields) loads without error."""
    from rl.experience_buffer import Experience
    old_dict = {
        "query_id": "old-schema-test",
        "transcript": "show alerts",
        "user_rating": "up",
        "widget_plan": {"widgets": [{"scenario": "alerts_list"}]},
        "timestamp": "2025-01-15T10:00:00",
    }
    exp = Experience.from_dict(old_dict)
    assert exp.voice_evaluation_confidence is None
    assert exp.voice_dimension_scores_claude == {}
    assert exp.voice_response is None

r.run("Old schema experience loads (backward compat)", test_old_schema_experience_loads)


# ================================================================
print(f"\n{'─'*70}")
print("  Edge Case: DPO Pair Quality Gates")
print(f"{'─'*70}")

def test_voice_no_pairs_when_all_low_confidence():
    """No DPO pairs generated when all experiences have low confidence."""
    from rl.background_trainer import BackgroundTrainer
    from rl.experience_buffer import Experience, ExperienceBuffer

    buf = ExperienceBuffer(max_size=100, persist_to_disk=False)
    trainer = BackgroundTrainer(buffer=buf, config={"min_batch_size": 2})

    # All low confidence — should produce 0 positive examples
    batch = [
        Experience(query_id="lc1", voice_response="response 1",
                   voice_evaluation_confidence=0.2, parsed_intent={"type": "q"}),
        Experience(query_id="lc2", voice_response="response 2",
                   voice_evaluation_confidence=0.1, parsed_intent={"type": "q"}),
    ]
    rewards = [0.5, -0.5]

    initial_count = len(trainer._dpo_pairs)
    trainer._accumulate_voice_pairs(batch, rewards)

    with trainer._dpo_pairs_lock:
        new_count = len(trainer._dpo_pairs) - initial_count

    assert new_count == 0, f"Expected 0 new pairs from low-confidence, got {new_count}"

r.run("No pairs when all low confidence", test_voice_no_pairs_when_all_low_confidence)

def test_voice_no_pairs_when_tiny_reward_gap():
    """No DPO pairs when reward gap < 0.3."""
    from rl.background_trainer import BackgroundTrainer
    from rl.experience_buffer import Experience, ExperienceBuffer

    buf = ExperienceBuffer(max_size=100, persist_to_disk=False)
    trainer = BackgroundTrainer(buffer=buf, config={"min_batch_size": 2})

    batch = [
        Experience(query_id="tg1", voice_response="response 1",
                   voice_evaluation_confidence=0.9, parsed_intent={"type": "q"}),
        Experience(query_id="tg2", voice_response="response 2",
                   voice_evaluation_confidence=0.1, parsed_intent={"type": "q"}),
    ]
    rewards = [0.2, 0.1]  # Gap = 0.1 < 0.3 threshold

    initial_count = len(trainer._dpo_pairs)
    trainer._accumulate_voice_pairs(batch, rewards)

    with trainer._dpo_pairs_lock:
        new_count = len(trainer._dpo_pairs) - initial_count

    assert new_count == 0, f"Expected 0 pairs from small gap, got {new_count}"

r.run("No pairs when reward gap < 0.3", test_voice_no_pairs_when_tiny_reward_gap)

def test_dpo_memory_cap():
    """DPO pairs list stays under 5000."""
    from rl.background_trainer import BackgroundTrainer
    from rl.experience_buffer import Experience, ExperienceBuffer

    buf = ExperienceBuffer(max_size=100, persist_to_disk=False)
    trainer = BackgroundTrainer(buffer=buf, config={"min_batch_size": 2})

    # Directly inject 5500 pairs to test cap
    with trainer._dpo_pairs_lock:
        for i in range(5500):
            trainer._dpo_pairs.append({
                "prompt": f"prompt-{i}",
                "chosen": f"chosen-{i}",
                "rejected": f"rejected-{i}",
                "pos_reward": 0.5,
                "neg_reward": -0.5,
                "reward_gap": 1.0,
                "timestamp": time.time(),
                "pair_type": "voice",
            })

    # Now trigger accumulate which should trigger the cap
    batch = [
        Experience(query_id=f"cap-pos", voice_response="good response",
                   voice_evaluation_confidence=0.9, parsed_intent={"type": "q", "domains": ["industrial"]}),
        Experience(query_id=f"cap-neg", voice_response="bad response",
                   parsed_intent={"type": "q", "domains": ["industrial"]}),
    ]
    trainer._accumulate_voice_pairs(batch, [0.5, -0.5])

    with trainer._dpo_pairs_lock:
        count = len(trainer._dpo_pairs)

    assert count <= 5000, f"DPO pairs should be capped at 5000, got {count}"

r.run("DPO pairs memory cap at 5000", test_dpo_memory_cap)


# ================================================================
print(f"\n{'─'*70}")
print("  Edge Case: Buffer Deque Correctness")
print(f"{'─'*70}")

def test_buffer_stays_deque_after_all_operations():
    """Buffer stays a deque through add, reload, update_feedback."""
    from rl.experience_buffer import Experience, ExperienceBuffer

    buf = ExperienceBuffer(max_size=50, persist_to_disk=False)
    assert isinstance(buf.buffer, collections.deque), "Initial buffer should be deque"

    for i in range(60):  # Exceed max_size
        buf.add(Experience(query_id=f"deque-stress-{i}"))
    assert isinstance(buf.buffer, collections.deque), "After overflow, should still be deque"
    assert buf.size() == 50, "Should be at max_size"

    buf.update_feedback(f"deque-stress-55", {"rating": "up"})
    assert isinstance(buf.buffer, collections.deque), "After update_feedback, should still be deque"

r.run("Buffer stays deque through all operations", test_buffer_stays_deque_after_all_operations)

def test_buffer_query_index_correct_after_overflow():
    """Query index stays in sync after overflow."""
    from rl.experience_buffer import Experience, ExperienceBuffer

    buf = ExperienceBuffer(max_size=10, persist_to_disk=False)

    for i in range(20):
        buf.add(Experience(query_id=f"idx-{i}"))

    # First 10 should be gone
    for i in range(10):
        assert buf.get_by_query_id(f"idx-{i}") is None, f"idx-{i} should be evicted"

    # Last 10 should be present
    for i in range(10, 20):
        assert buf.get_by_query_id(f"idx-{i}") is not None, f"idx-{i} should be present"

r.run("Query index correct after overflow", test_buffer_query_index_correct_after_overflow)


# ================================================================
print(f"\n{'─'*70}")
print("  Edge Case: Concurrent-Like Scenarios")
print(f"{'─'*70}")

def test_voice_rating_not_overwritten_by_later_update():
    """Later update without voice eval doesn't erase Claude's voice judgment."""
    from rl.continuous import ContinuousRL
    rl = ContinuousRL()

    rl.record_experience(query_id="overwrite-test", transcript="test",
                         voice_response="Good response with data.")

    # First update: Claude says voice is good
    rl.update_feedback("overwrite-test",
                       voice_evaluation_confidence=0.9,
                       voice_evaluation_reasoning="Great voice response")

    exp = rl.buffer.get_by_query_id("overwrite-test")
    assert exp.voice_response_rating == "good"

    # Second update: user gives widget rating but NO voice eval
    rl.update_feedback("overwrite-test", rating="down")

    exp2 = rl.buffer.get_by_query_id("overwrite-test")
    # voice_response_rating should STILL be "good" from Claude's judgment
    # (Claude's voice eval confidence is 0.9, which was set in first update)
    assert exp2.voice_response_rating == "good", \
        f"Voice rating should still be 'good' from Claude, got '{exp2.voice_response_rating}'"

r.run("Voice rating preserved across updates", test_voice_rating_not_overwritten_by_later_update)

def test_reward_computation_idempotent():
    """Computing reward twice gives same result."""
    from rl.experience_buffer import Experience
    from rl.reward_signals import RewardSignalAggregator

    agg = RewardSignalAggregator()
    exp = Experience(
        query_id="idempotent-test",
        user_rating="up",
        voice_response="Pump 001 at 2400 RPM.",
        transcript="pump status",
        voice_dimension_scores_claude={"groundedness": 0.9, "conciseness": 0.8},
    )

    r1 = agg.compute_reward(exp)
    r2 = agg.compute_reward(exp)
    assert abs(r1 - r2) < 0.001, f"Rewards should be equal: {r1} vs {r2}"

r.run("Reward computation idempotent", test_reward_computation_idempotent)


# ================================================================
success = r.summary()
sys.exit(0 if success else 1)
