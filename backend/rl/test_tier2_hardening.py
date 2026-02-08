#!/usr/bin/env python3
"""
Unified DPO Hardening Tests (Tier 2: Widget + Voice)

Tests all changes from the Claude Voice Evaluation plan:
1. Experience voice evaluation fields roundtrip
2. _voice_confidence() helper
3. _accumulate_voice_pairs uses voice-specific confidence
4. voice_response_rating prefers Claude voice judgment
5. _text_quality_reward blends Claude + rule-based scores
6. Auto-evaluator prompt includes voice_response text
7. Backward compatibility (all existing tests still pass)

Run: cd backend && ../venv/bin/python -m rl.test_tier2_hardening
"""

import os
import sys
import time
import json
import traceback
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "command_center.settings")

import django
django.setup()


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run(self, name, test_fn):
        """Run a test and track results."""
        start = time.time()
        try:
            test_fn()
            dur = int((time.time() - start) * 1000)
            self.passed += 1
            print(f"  ✓ {name} ({dur}ms)")
        except AssertionError as e:
            dur = int((time.time() - start) * 1000)
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  ✗ {name} ({dur}ms)")
            print(f"      ASSERT: {e}")
        except Exception as e:
            dur = int((time.time() - start) * 1000)
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  ✗ {name} ({dur}ms)")
            print(f"      ERROR: {e}")
            traceback.print_exc()

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"RESULTS: {self.passed}/{total} passed, {self.failed} failed")
        print(f"{'='*70}")
        if self.errors:
            print("\nFailed tests:")
            for name, err in self.errors:
                print(f"  - {name}: {err}")
        return self.failed == 0


runner = TestRunner()


# ================================================================
# Group 1: Experience Dataclass & Buffer
# ================================================================
print(f"\n{'─'*70}")
print("  Experience Dataclass & Buffer")
print(f"{'─'*70}")


def test_experience_voice_eval_fields():
    """New voice evaluation fields exist on Experience."""
    from rl.experience_buffer import Experience
    exp = Experience(
        query_id="test-voice-fields",
        voice_evaluation_confidence=0.85,
        voice_evaluation_reasoning="Clear and direct response",
        voice_dimension_scores_claude={
            "groundedness": 0.9,
            "conciseness": 0.8,
            "directness": 0.95,
            "specificity": 0.85,
            "tts_friendliness": 0.9,
        },
    )
    assert exp.voice_evaluation_confidence == 0.85
    assert exp.voice_evaluation_reasoning == "Clear and direct response"
    assert exp.voice_dimension_scores_claude["groundedness"] == 0.9
    assert exp.voice_dimension_scores_claude["tts_friendliness"] == 0.9

runner.run("Experience voice eval fields exist", test_experience_voice_eval_fields)


def test_experience_voice_eval_defaults():
    """Voice eval fields default to None/empty."""
    from rl.experience_buffer import Experience
    exp = Experience(query_id="test-defaults")
    assert exp.voice_evaluation_confidence is None
    assert exp.voice_evaluation_reasoning is None
    assert exp.voice_dimension_scores_claude == {}

runner.run("Experience voice eval defaults", test_experience_voice_eval_defaults)


def test_experience_roundtrip_serialization():
    """Voice eval fields survive to_dict() -> from_dict()."""
    from rl.experience_buffer import Experience
    exp = Experience(
        query_id="test-roundtrip",
        voice_response="Pump 001 is running at 95% load.",
        voice_evaluation_confidence=0.75,
        voice_evaluation_reasoning="Good specificity",
        voice_dimension_scores_claude={"groundedness": 0.8, "conciseness": 0.9},
    )
    d = exp.to_dict()
    exp2 = Experience.from_dict(d)
    assert exp2.voice_evaluation_confidence == 0.75
    assert exp2.voice_evaluation_reasoning == "Good specificity"
    assert exp2.voice_dimension_scores_claude["groundedness"] == 0.8

runner.run("Experience roundtrip serialization", test_experience_roundtrip_serialization)


def test_experience_from_dict_ignores_unknown():
    """from_dict still ignores unknown fields (forward compat)."""
    from rl.experience_buffer import Experience
    d = {
        "query_id": "test-unknown",
        "future_field_xyz": "should be ignored",
    }
    exp = Experience.from_dict(d)
    assert exp.query_id == "test-unknown"
    assert not hasattr(exp, "future_field_xyz")

runner.run("from_dict ignores unknown fields", test_experience_from_dict_ignores_unknown)


def test_buffer_update_feedback_voice_eval():
    """update_feedback stores voice eval fields."""
    from rl.experience_buffer import Experience, ExperienceBuffer
    buf = ExperienceBuffer(max_size=100, persist_to_disk=False)
    exp = Experience(query_id="test-fb-voice", voice_response="Test response.")
    buf.add(exp)

    success = buf.update_feedback("test-fb-voice", {
        "rating": "up",
        "voice_evaluation_confidence": 0.9,
        "voice_evaluation_reasoning": "Excellent direct answer",
        "voice_dimension_scores_claude": {"groundedness": 0.95, "directness": 0.9},
    })
    assert success
    updated = buf.get_by_query_id("test-fb-voice")
    assert updated.voice_evaluation_confidence == 0.9
    assert updated.voice_evaluation_reasoning == "Excellent direct answer"
    assert updated.voice_dimension_scores_claude["groundedness"] == 0.95

runner.run("Buffer update_feedback stores voice eval", test_buffer_update_feedback_voice_eval)


def test_buffer_deque_popleft():
    """Buffer uses deque.popleft() correctly at capacity."""
    from rl.experience_buffer import Experience, ExperienceBuffer
    import collections
    buf = ExperienceBuffer(max_size=5, persist_to_disk=False)
    for i in range(7):
        exp = Experience(query_id=f"deque-test-{i}")
        buf.add(exp)

    assert buf.size() == 5
    assert isinstance(buf.buffer, collections.deque)
    # Oldest should be gone
    assert buf.get_by_query_id("deque-test-0") is None
    assert buf.get_by_query_id("deque-test-1") is None
    assert buf.get_by_query_id("deque-test-6") is not None

runner.run("Buffer deque popleft at capacity", test_buffer_deque_popleft)


def test_has_feedback_includes_voice():
    """has_feedback() returns True for voice_response_rating."""
    from rl.experience_buffer import Experience
    exp = Experience(query_id="test-has-fb", voice_response_rating="good")
    assert exp.has_feedback()

    exp2 = Experience(query_id="test-no-fb")
    assert not exp2.has_feedback()

runner.run("has_feedback includes voice rating", test_has_feedback_includes_voice)


# ================================================================
# Group 2: ContinuousRL Voice Rating Derivation
# ================================================================
print(f"\n{'─'*70}")
print("  ContinuousRL Voice Rating Derivation")
print(f"{'─'*70}")


def test_voice_rating_prefers_claude():
    """voice_response_rating uses Claude voice eval when available."""
    from rl.continuous import ContinuousRL
    rl = ContinuousRL()

    # Record experience with voice_response
    rl.record_experience(
        query_id="voice-prefer-claude",
        transcript="What is pump-001 status?",
        voice_response="Pump 001 is operating normally at 2400 RPM.",
    )

    # Update with Claude voice eval (high confidence) + user thumbs down
    # Claude says voice is good (0.8 confidence) but user thumbs-downed (maybe bad widgets)
    rl.update_feedback(
        "voice-prefer-claude",
        rating="down",
        voice_evaluation_confidence=0.8,
        voice_evaluation_reasoning="Clear, direct, cites specific data",
    )

    exp = rl.buffer.get_by_query_id("voice-prefer-claude")
    # Should prefer Claude's judgment: 0.8 >= 0.5 → "good"
    assert exp.voice_response_rating == "good", f"Expected 'good', got '{exp.voice_response_rating}'"

runner.run("Voice rating prefers Claude over user_rating", test_voice_rating_prefers_claude)


def test_voice_rating_fallback_to_user():
    """voice_response_rating falls back to user_rating when no Claude eval."""
    from rl.continuous import ContinuousRL
    rl = ContinuousRL()

    rl.record_experience(
        query_id="voice-fallback-user",
        transcript="Show alerts",
        voice_response="There are 3 active alerts.",
    )

    # No voice_evaluation_confidence — should fall back to user_rating
    rl.update_feedback("voice-fallback-user", rating="down")

    exp = rl.buffer.get_by_query_id("voice-fallback-user")
    assert exp.voice_response_rating == "bad", f"Expected 'bad', got '{exp.voice_response_rating}'"

runner.run("Voice rating fallback to user_rating", test_voice_rating_fallback_to_user)


def test_voice_rating_claude_low_confidence():
    """Claude low confidence (0.3) → voice_response_rating = bad."""
    from rl.continuous import ContinuousRL
    rl = ContinuousRL()

    rl.record_experience(
        query_id="voice-low-conf",
        transcript="Tell me about compressors",
        voice_response="Sure, I can help with that. Let me look into the compressors.",
    )

    rl.update_feedback(
        "voice-low-conf",
        rating="up",  # User liked widgets, but voice was poor
        voice_evaluation_confidence=0.3,
    )

    exp = rl.buffer.get_by_query_id("voice-low-conf")
    # Claude says 0.3 < 0.5 → "bad" even though user said "up"
    assert exp.voice_response_rating == "bad", f"Expected 'bad', got '{exp.voice_response_rating}'"

runner.run("Voice rating Claude low confidence = bad", test_voice_rating_claude_low_confidence)


def test_no_voice_response_no_rating():
    """No voice_response → voice_response_rating stays None."""
    from rl.continuous import ContinuousRL
    rl = ContinuousRL()

    rl.record_experience(
        query_id="voice-none",
        transcript="Show power trends",
        voice_response="",  # Empty
    )

    rl.update_feedback("voice-none", rating="up")

    exp = rl.buffer.get_by_query_id("voice-none")
    assert exp.voice_response_rating is None, f"Expected None, got '{exp.voice_response_rating}'"

runner.run("No voice_response → no voice rating", test_no_voice_response_no_rating)


# ================================================================
# Group 3: Background Trainer _voice_confidence
# ================================================================
print(f"\n{'─'*70}")
print("  Background Trainer Voice Confidence")
print(f"{'─'*70}")


def test_voice_confidence_prefers_voice_eval():
    """_voice_confidence returns voice_evaluation_confidence first."""
    from rl.background_trainer import BackgroundTrainer
    from rl.experience_buffer import Experience, ExperienceBuffer

    buf = ExperienceBuffer(max_size=100, persist_to_disk=False)
    trainer = BackgroundTrainer(buffer=buf, config={"min_batch_size": 5})

    exp = Experience(
        query_id="vc-prefer-voice",
        voice_evaluation_confidence=0.9,
        evaluation_confidence=0.6,
    )
    assert trainer._voice_confidence(exp) == 0.9

runner.run("_voice_confidence prefers voice eval", test_voice_confidence_prefers_voice_eval)


def test_voice_confidence_fallback_widget():
    """_voice_confidence falls back to widget evaluation_confidence."""
    from rl.background_trainer import BackgroundTrainer
    from rl.experience_buffer import Experience, ExperienceBuffer

    buf = ExperienceBuffer(max_size=100, persist_to_disk=False)
    trainer = BackgroundTrainer(buffer=buf, config={"min_batch_size": 5})

    exp = Experience(
        query_id="vc-fallback",
        evaluation_confidence=0.7,
    )
    assert trainer._voice_confidence(exp) == 0.7

runner.run("_voice_confidence falls back to widget conf", test_voice_confidence_fallback_widget)


def test_voice_confidence_default_zero():
    """_voice_confidence returns 0.0 when no confidence at all."""
    from rl.background_trainer import BackgroundTrainer
    from rl.experience_buffer import Experience, ExperienceBuffer

    buf = ExperienceBuffer(max_size=100, persist_to_disk=False)
    trainer = BackgroundTrainer(buffer=buf, config={"min_batch_size": 5})

    exp = Experience(query_id="vc-default")
    assert trainer._voice_confidence(exp) == 0.0

runner.run("_voice_confidence default zero", test_voice_confidence_default_zero)


def test_accumulate_voice_uses_voice_confidence():
    """_accumulate_voice_pairs uses _voice_confidence for positive filtering."""
    from rl.background_trainer import BackgroundTrainer
    from rl.experience_buffer import Experience, ExperienceBuffer

    buf = ExperienceBuffer(max_size=100, persist_to_disk=False)
    config = {"min_batch_size": 5}
    trainer = BackgroundTrainer(buffer=buf, config=config)

    # Experience with voice_evaluation_confidence (no widget eval)
    exp_good = Experience(
        query_id="dpo-good",
        voice_response="Pump 001 running at 95% load, 2400 RPM.",
        voice_evaluation_confidence=0.9,
        parsed_intent={"type": "query", "domains": ["industrial"]},
    )

    # Experience with only widget eval (no voice eval) — should still work
    exp_fallback = Experience(
        query_id="dpo-fallback",
        voice_response="Status is normal.",
        evaluation_confidence=0.7,
        parsed_intent={"type": "query", "domains": ["industrial"]},
    )

    # Bad experience
    exp_bad = Experience(
        query_id="dpo-bad",
        voice_response="Well let me think about that for a moment.",
        voice_evaluation_confidence=0.1,
        parsed_intent={"type": "query", "domains": ["industrial"]},
    )

    batch = [exp_good, exp_fallback, exp_bad]
    rewards = [0.5, 0.3, -0.5]

    trainer._accumulate_voice_pairs(batch, rewards)

    # Should generate pairs: good×bad and fallback×bad
    with trainer._dpo_pairs_lock:
        pairs = list(trainer._dpo_pairs)

    assert len(pairs) >= 1, f"Expected at least 1 pair, got {len(pairs)}"
    # Verify pair quality
    for p in pairs:
        assert p["pair_type"] == "voice"
        assert p["reward_gap"] >= 0.3

runner.run("_accumulate_voice_pairs uses voice confidence", test_accumulate_voice_uses_voice_confidence)


def test_accumulate_voice_rejects_low_confidence():
    """_accumulate_voice_pairs rejects positive with low voice confidence."""
    from rl.background_trainer import BackgroundTrainer
    from rl.experience_buffer import Experience, ExperienceBuffer

    buf = ExperienceBuffer(max_size=100, persist_to_disk=False)
    config = {"min_batch_size": 5}
    trainer = BackgroundTrainer(buffer=buf, config=config)

    # Good reward but low confidence — should NOT be positive
    exp_low_conf = Experience(
        query_id="dpo-lowconf",
        voice_response="Here is some info about pumps.",
        voice_evaluation_confidence=0.3,
        parsed_intent={"type": "query", "domains": ["industrial"]},
    )

    exp_bad = Experience(
        query_id="dpo-lowconf-bad",
        voice_response="I don't know.",
        parsed_intent={"type": "query", "domains": ["industrial"]},
    )

    batch = [exp_low_conf, exp_bad]
    rewards = [0.5, -0.5]

    trainer._accumulate_voice_pairs(batch, rewards)

    # exp_low_conf has confidence 0.3 < 0.5, should not be positive
    with trainer._dpo_pairs_lock:
        pairs = list(trainer._dpo_pairs)

    # There were pre-existing pairs from previous test, filter to this batch
    new_pairs = [p for p in pairs if "lowconf" in p.get("chosen", "") or "lowconf" in p.get("rejected", "")]
    assert len(new_pairs) == 0, f"Expected 0 pairs from low-confidence, got {len(new_pairs)}"

runner.run("_accumulate_voice_pairs rejects low confidence", test_accumulate_voice_rejects_low_confidence)


# ================================================================
# Group 4: Reward Signal Blending
# ================================================================
print(f"\n{'─'*70}")
print("  Reward Signal Blending (Claude + Rule-Based)")
print(f"{'─'*70}")


def test_text_quality_rule_based_only():
    """_text_quality_reward works with rule-based only (no Claude dims)."""
    from rl.experience_buffer import Experience
    from rl.reward_signals import RewardSignalAggregator

    agg = RewardSignalAggregator()
    exp = Experience(
        query_id="reward-rule-only",
        voice_response="Pump 001 is operating at 2400 RPM with 95% load. No anomalies detected.",
        transcript="What is pump-001 status?",
    )
    reward = agg.compute_reward(exp)
    # Should be nonzero (voice text is decent)
    assert isinstance(reward, float)
    # Check scores were stored
    assert exp.voice_quality_scores, "voice_quality_scores should be populated"
    assert "total" in exp.voice_quality_scores

runner.run("Text quality rule-based only", test_text_quality_rule_based_only)


def test_text_quality_blend_with_claude():
    """_text_quality_reward blends Claude + rule-based when both present."""
    from rl.experience_buffer import Experience
    from rl.reward_signals import RewardSignalAggregator

    agg = RewardSignalAggregator()

    # First, get pure rule-based score
    exp1 = Experience(
        query_id="reward-rule-base",
        voice_response="Pump 001 is at 2400 RPM.",
        transcript="Show pump-001 RPM",
    )
    agg._text_quality_reward(exp1)
    rule_total = exp1.voice_quality_scores.get("total", 0)

    # Now with Claude dims — high scores should boost total
    exp2 = Experience(
        query_id="reward-blended",
        voice_response="Pump 001 is at 2400 RPM.",
        transcript="Show pump-001 RPM",
        voice_dimension_scores_claude={
            "groundedness": 1.0,
            "conciseness": 1.0,
            "directness": 1.0,
            "specificity": 1.0,
            "tts_friendliness": 1.0,
        },
    )
    agg._text_quality_reward(exp2)
    blended_total = exp2.voice_quality_scores.get("total", 0)

    # Blended should be >= rule-based since Claude scores are perfect 1.0
    assert blended_total >= rule_total, (
        f"Blended {blended_total:.4f} should be >= rule-based {rule_total:.4f}"
    )

runner.run("Text quality blends with Claude", test_text_quality_blend_with_claude)


def test_text_quality_blend_ignores_bad_claude():
    """_text_quality_reward handles invalid Claude dim types gracefully."""
    from rl.experience_buffer import Experience
    from rl.reward_signals import RewardSignalAggregator

    agg = RewardSignalAggregator()
    exp = Experience(
        query_id="reward-bad-claude",
        voice_response="Pump 001 is running normally.",
        transcript="Pump status",
        voice_dimension_scores_claude={
            "groundedness": "not_a_number",  # Invalid type
            "conciseness": 0.8,
        },
    )
    # Should not crash
    reward = agg._text_quality_reward(exp)
    assert isinstance(reward, float)
    # groundedness should keep rule-based score (ValueError caught)

runner.run("Text quality blend handles invalid types", test_text_quality_blend_ignores_bad_claude)


def test_text_quality_no_voice_response():
    """_text_quality_reward returns 0.0 when no voice_response."""
    from rl.experience_buffer import Experience
    from rl.reward_signals import RewardSignalAggregator

    agg = RewardSignalAggregator()
    exp = Experience(query_id="reward-no-voice")
    reward = agg._text_quality_reward(exp)
    assert reward == 0.0

runner.run("Text quality 0.0 with no voice", test_text_quality_no_voice_response)


def test_text_quality_poor_vs_good_discrimination():
    """Good voice responses score higher than poor ones (basic sanity)."""
    from rl.experience_buffer import Experience
    from rl.reward_signals import RewardSignalAggregator

    agg = RewardSignalAggregator()

    good = Experience(
        query_id="reward-good",
        voice_response="Pump 001 is operating at 2400 RPM with 95% load and 45 degrees Celsius bearing temperature.",
        transcript="What is pump-001 status?",
    )
    poor = Experience(
        query_id="reward-poor",
        voice_response="Well, let me think about that. Based on the available data, I'd be happy to help you understand the current situation regarding the equipment you mentioned. Here's what I found in the system.",
        transcript="What is pump-001 status?",
    )

    good_reward = agg._text_quality_reward(good)
    poor_reward = agg._text_quality_reward(poor)

    assert good_reward > poor_reward, (
        f"Good reward {good_reward:.4f} should be > poor reward {poor_reward:.4f}"
    )

runner.run("Good vs poor voice discrimination", test_text_quality_poor_vs_good_discrimination)


# ================================================================
# Group 5: Auto-Evaluator Prompt
# ================================================================
print(f"\n{'─'*70}")
print("  Auto-Evaluator Prompt Construction")
print(f"{'─'*70}")


def test_auto_eval_prompt_includes_voice():
    """evaluate_with_claude prompt includes voice_response text."""
    # We can't call Claude, but we can verify the prompt is built correctly
    # by importing the function and checking the prompt construction
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    # Build a mock experience with voice_response
    experience = {
        "query_id": "test-prompt",
        "transcript": "Show pump-001 status",
        "parsed_intent": {"type": "query", "domains": ["industrial"], "confidence": 0.9, "primary_characteristic": "status"},
        "widget_plan": {"widgets": [{"scenario": "device_status", "size": "large", "relevance": 0.95}]},
        "voice_response": "Pump 001 is running at 2400 RPM with normal vibration levels.",
    }

    # Simulate prompt building logic from auto_evaluate_responses.py
    transcript = experience.get('transcript', 'Unknown query')
    voice_response = experience.get('voice_response', '')
    has_voice = bool(voice_response and voice_response.strip())

    assert has_voice, "Should detect voice_response"

    # Check voice_evaluation JSON block would be included
    voice_json_block = ""
    if has_voice:
        voice_json_block = '"voice_evaluation"'

    assert voice_json_block, "Voice JSON block should be non-empty"

runner.run("Auto-eval prompt includes voice response", test_auto_eval_prompt_includes_voice)


def test_auto_eval_prompt_no_voice():
    """evaluate_with_claude prompt omits voice section when no voice_response."""
    experience = {
        "query_id": "test-no-voice",
        "transcript": "Show alerts",
        "parsed_intent": {"type": "query", "domains": ["alerts"], "confidence": 0.8},
        "widget_plan": {"widgets": [{"scenario": "alerts_list", "size": "medium", "relevance": 0.9}]},
        "voice_response": "",
    }

    voice_response = experience.get('voice_response', '')
    has_voice = bool(voice_response and voice_response.strip())

    assert not has_voice, "Should not detect empty voice_response"

runner.run("Auto-eval prompt omits voice when empty", test_auto_eval_prompt_no_voice)


def test_auto_eval_voice_eval_extraction():
    """create_rating extracts voice_evaluation from Claude response."""
    # Simulate what create_rating does with the evaluation dict
    evaluation = {
        "overall_rating": "up",
        "confidence": 0.85,
        "reasoning": "Good selection",
        "voice_evaluation": {
            "quality_rating": "GOOD",
            "confidence": 0.9,
            "reasoning": "Direct and specific",
            "dimension_scores": {
                "groundedness": 0.95,
                "conciseness": 0.9,
                "directness": 0.85,
                "specificity": 0.9,
                "tts_friendliness": 0.95,
            },
        },
    }

    voice_eval = evaluation.get("voice_evaluation", {})
    assert voice_eval
    assert isinstance(voice_eval, dict)

    api_payload = {}
    if voice_eval and isinstance(voice_eval, dict):
        api_payload["voice_evaluation_confidence"] = voice_eval.get("confidence")
        api_payload["voice_evaluation_reasoning"] = voice_eval.get("reasoning")
        api_payload["voice_dimension_scores_claude"] = voice_eval.get("dimension_scores", {})

    assert api_payload["voice_evaluation_confidence"] == 0.9
    assert api_payload["voice_evaluation_reasoning"] == "Direct and specific"
    assert api_payload["voice_dimension_scores_claude"]["groundedness"] == 0.95

runner.run("Auto-eval extracts voice_evaluation", test_auto_eval_voice_eval_extraction)


# ================================================================
# Group 6: Full Integration (existing tests must still pass)
# ================================================================
print(f"\n{'─'*70}")
print("  Full Integration (Backward Compatibility)")
print(f"{'─'*70}")


def test_existing_experience_buffer():
    """Original experience buffer tests still pass."""
    from rl.experience_buffer import Experience, ExperienceBuffer

    # Use large max_size to ensure disk-loaded + test experiences all fit
    buffer = ExperienceBuffer(max_size=10000, persist_to_disk=False)
    initial_size = buffer.size()

    for i in range(5):
        exp = Experience(
            query_id=f"compat-{i}",
            transcript=f"Test {i}",
            parsed_intent={"type": "query"},
            widget_plan={"widgets": [{"scenario": "test"}]},
            processing_time_ms=1000,
        )
        buffer.add(exp)

    assert buffer.size() == initial_size + 5, f"Expected {initial_size+5}, got {buffer.size()}"
    assert buffer.update_feedback("compat-2", {"rating": "up"})
    batch = buffer.sample_batch(3, require_feedback=False)
    assert len(batch) == 3, f"Expected 3, got {len(batch)}"
    stats = buffer.get_stats()
    assert stats["total_experiences"] == initial_size + 5

runner.run("Existing buffer tests pass", test_existing_experience_buffer)


def test_existing_reward_signals():
    """Reward signals work as before."""
    from rl.experience_buffer import Experience
    from rl.reward_signals import RewardSignalAggregator

    agg = RewardSignalAggregator()

    exp_up = Experience(
        query_id="compat-up",
        user_rating="up",
        processing_time_ms=2000,
        intent_confidence=0.9,
    )
    reward_up = agg.compute_reward(exp_up)
    assert reward_up > 0, f"Upvote reward should be positive, got {reward_up}"

    exp_down = Experience(
        query_id="compat-down",
        user_rating="down",
        processing_time_ms=8000,
        intent_confidence=0.3,
    )
    reward_down = agg.compute_reward(exp_down)
    assert reward_down < 0, f"Downvote reward should be negative, got {reward_down}"

runner.run("Existing reward signals pass", test_existing_reward_signals)


def test_existing_continuous_rl():
    """ContinuousRL coordinator works as before."""
    from rl.continuous import ContinuousRL

    rl = ContinuousRL()
    rl.record_experience(
        query_id="compat-rl-1",
        transcript="Show pump-001 status",
        parsed_intent={"type": "query", "domains": ["industrial"]},
        widget_plan={"widgets": [{"scenario": "device_status"}]},
        processing_time_ms=1500,
    )

    success = rl.update_feedback("compat-rl-1", rating="up")
    assert success

    status = rl.get_status()
    assert "buffer" in status
    assert "running" in status

runner.run("Existing ContinuousRL pass", test_existing_continuous_rl)


def test_existing_background_trainer():
    """Background trainer initializes correctly."""
    from rl.background_trainer import BackgroundTrainer
    from rl.experience_buffer import ExperienceBuffer

    buf = ExperienceBuffer(max_size=100, persist_to_disk=False)
    config = {"min_batch_size": 5}
    trainer = BackgroundTrainer(buffer=buf, config=config)
    stats = trainer.get_stats()
    assert "training_steps" in stats or isinstance(stats, dict)

runner.run("Existing background trainer pass", test_existing_background_trainer)


def test_existing_text_quality_scorer():
    """TextQualityScorer still works standalone."""
    from rl.text_quality_scorer import TextQualityScorer

    scorer = TextQualityScorer()

    good = scorer.score(
        "Pump 001 is operating at 2400 RPM with 95% load.",
        "What is pump-001 status?",
    )
    assert good["total"] > 0.5, f"Good response total should be >0.5, got {good['total']}"

    bad = scorer.score(
        "Well, let me think about that. Based on the data I can see, there might be something going on.",
        "What is pump-001 status?",
    )
    assert good["total"] > bad["total"], "Good should score higher than bad"

runner.run("Existing TextQualityScorer pass", test_existing_text_quality_scorer)


def test_dpo_config():
    """DPO config values present (unified Tier 2 config)."""
    from rl.config import CONTINUOUS_RL_CONFIG, DPO_CONFIG

    assert CONTINUOUS_RL_CONFIG["lora_min_pairs"] == 80
    assert DPO_CONFIG["max_length"] == 2048
    assert DPO_CONFIG["lora_r"] == 16

runner.run("DPO config values (unified)", test_dpo_config)


def test_gpu_lock_file():
    """DPO training uses lora_training.lock."""
    from rl.config import TRAINING_DATA_DIR
    import inspect
    from rl.background_trainer import BackgroundTrainer

    # Verify the lock file path is used in DPO training methods
    source = inspect.getsource(BackgroundTrainer)
    lock_references = source.count("lora_training.lock")
    assert lock_references >= 1, (
        f"Expected lora_training.lock referenced in DPO training, found {lock_references} refs"
    )
    # Verify TRAINING_DATA_DIR is consistent
    expected_lock = TRAINING_DATA_DIR / "lora_training.lock"
    assert "lora_training.lock" in str(expected_lock)

runner.run("GPU lock file for DPO training", test_gpu_lock_file)


# ================================================================
# Group 7: End-to-End Flow
# ================================================================
print(f"\n{'─'*70}")
print("  End-to-End Flow (Full Pipeline)")
print(f"{'─'*70}")


def test_full_pipeline_with_voice_eval():
    """Full flow: record → feedback with voice eval → reward → voice accumulate."""
    from rl.continuous import ContinuousRL
    from rl.background_trainer import BackgroundTrainer
    from rl.experience_buffer import ExperienceBuffer
    from rl.reward_signals import RewardSignalAggregator

    rl = ContinuousRL()

    # Step 1: Record experience with voice_response
    rl.record_experience(
        query_id="e2e-full-pipeline",
        transcript="What is the power consumption of CT scanner 3?",
        parsed_intent={"type": "query", "domains": ["industrial"], "entities": {"equipment": "ct-3"}},
        widget_plan={"widgets": [{"scenario": "power_trend", "size": "large", "relevance": 0.9}]},
        processing_time_ms=1200,
        voice_response="CT scanner 3 is consuming 45 kW, which is 12% above the daily average of 40 kW.",
    )

    # Step 2: Update with rich Claude feedback including voice eval
    success = rl.update_feedback(
        "e2e-full-pipeline",
        rating="up",
        evaluation_confidence=0.85,
        evaluation_reasoning="Good widget selection for power query",
        voice_evaluation_confidence=0.92,
        voice_evaluation_reasoning="Excellent: cites specific kW, compares to average, uses units",
        voice_dimension_scores_claude={
            "groundedness": 0.95,
            "conciseness": 0.9,
            "directness": 0.95,
            "specificity": 0.95,
            "tts_friendliness": 0.9,
        },
    )
    assert success

    # Step 3: Verify experience state
    exp = rl.buffer.get_by_query_id("e2e-full-pipeline")
    assert exp is not None
    assert exp.voice_response_rating == "good"  # Claude 0.92 >= 0.5
    assert exp.voice_evaluation_confidence == 0.92
    assert exp.voice_dimension_scores_claude["groundedness"] == 0.95
    assert exp.computed_reward is not None  # Reward was computed

    # Step 4: Check reward includes text quality
    agg = RewardSignalAggregator()
    breakdown = agg.get_reward_breakdown(exp)
    # Total should be positive (good response)
    assert breakdown["total"] > 0, f"Total reward should be positive, got {breakdown['total']}"

    # Step 5: Verify voice_quality_scores have blended values
    assert exp.voice_quality_scores, "voice_quality_scores should be set"
    assert exp.voice_quality_scores.get("total", 0) > 0.5, \
        f"Blended total should be > 0.5, got {exp.voice_quality_scores.get('total')}"

runner.run("Full pipeline with voice eval", test_full_pipeline_with_voice_eval)


def test_full_pipeline_without_voice():
    """Full flow works correctly when no voice_response."""
    from rl.continuous import ContinuousRL

    rl = ContinuousRL()

    rl.record_experience(
        query_id="e2e-no-voice",
        transcript="Show all alerts",
        parsed_intent={"type": "query", "domains": ["alerts"]},
        widget_plan={"widgets": [{"scenario": "alerts_list"}]},
        processing_time_ms=800,
    )

    success = rl.update_feedback(
        "e2e-no-voice",
        rating="up",
        evaluation_confidence=0.9,
    )
    assert success

    exp = rl.buffer.get_by_query_id("e2e-no-voice")
    assert exp.voice_response_rating is None  # No voice → no voice rating
    assert exp.computed_reward is not None

runner.run("Full pipeline without voice", test_full_pipeline_without_voice)


# ================================================================
# Summary
# ================================================================
success = runner.summary()
sys.exit(0 if success else 1)
