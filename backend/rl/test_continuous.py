"""
Test script for the continuous RL system.

Run with: python -m rl.test_continuous
"""

import os
import sys
import time

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "command_center.settings")

import django
django.setup()


def test_experience_buffer():
    """Test the experience buffer."""
    print("\n=== Testing Experience Buffer ===")

    from rl.experience_buffer import Experience, ExperienceBuffer

    # Create buffer
    buffer = ExperienceBuffer(max_size=100, persist_to_disk=False)

    # Add experiences
    for i in range(5):
        exp = Experience(
            query_id=f"test-query-{i}",
            transcript=f"Test transcript {i}",
            parsed_intent={"type": "query", "domains": ["industrial"]},
            widget_plan={"widgets": [{"scenario": "test"}]},
            processing_time_ms=1000 + i * 100,
        )
        buffer.add(exp)
        print(f"  Added experience {exp.query_id}")

    print(f"  Buffer size: {buffer.size()}")

    # Update feedback
    success = buffer.update_feedback("test-query-2", {"rating": "up"})
    print(f"  Update feedback: {success}")

    # Sample batch
    batch = buffer.sample_batch(3, require_feedback=False)
    print(f"  Sampled batch of {len(batch)} experiences")

    # Get stats
    stats = buffer.get_stats()
    print(f"  Stats: {stats}")

    print("  ✓ Experience buffer tests passed")
    return True


def test_reward_signals():
    """Test reward computation."""
    print("\n=== Testing Reward Signals ===")

    from rl.experience_buffer import Experience
    from rl.reward_signals import RewardSignalAggregator, ImplicitSignalDetector

    aggregator = RewardSignalAggregator()
    detector = ImplicitSignalDetector()

    # Test with positive feedback
    exp1 = Experience(
        query_id="test-1",
        user_rating="up",
        processing_time_ms=2000,
        intent_confidence=0.9,
        widget_plan={"widgets": [{"scenario": "a"}, {"scenario": "b"}]},
        widget_interactions=[{"widget_index": 0, "action": "expand"}],
    )
    reward1 = aggregator.compute_reward(exp1)
    print(f"  Positive feedback reward: {reward1:.3f}")

    # Test with negative feedback
    exp2 = Experience(
        query_id="test-2",
        user_rating="down",
        processing_time_ms=8000,  # Slow
        intent_confidence=0.3,
    )
    reward2 = aggregator.compute_reward(exp2)
    print(f"  Negative feedback reward: {reward2:.3f}")

    # Test follow-up classification (different topics = satisfied)
    prev_intent = {"domains": ["industrial"], "entities": {"pump": "pump-001"}}
    curr_intent = {"domains": ["supply"], "entities": {"item": "widget-x"}}
    follow_up = detector.classify_follow_up(
        prev_intent, curr_intent,
        "show me pump-001 status",
        "show inventory levels"  # Different topic, no repeat keywords
    )
    print(f"  Follow-up type (different topic): {follow_up}")
    assert follow_up == "satisfied", f"Expected 'satisfied', got '{follow_up}'"

    # Test follow-up classification (same topic = refinement)
    prev_intent2 = {"domains": ["industrial"], "entities": {"pump": "pump-001"}}
    curr_intent2 = {"domains": ["industrial"], "entities": {"pump": "pump-001", "metric": "temperature"}}
    follow_up2 = detector.classify_follow_up(
        prev_intent2, curr_intent2,
        "show me pump-001 status",
        "show pump-001 temperature"  # Same device, more specific
    )
    print(f"  Follow-up type (refinement): {follow_up2}")

    # Test correction detection
    correction = detector.extract_correction("No, I meant pump-002 not pump-001")
    print(f"  Extracted correction: {correction}")

    print("  ✓ Reward signal tests passed")
    return True


def test_continuous_rl_coordinator():
    """Test the ContinuousRL coordinator."""
    print("\n=== Testing ContinuousRL Coordinator ===")

    from rl.continuous import ContinuousRL

    rl = ContinuousRL()

    # Record some experiences
    rl.record_experience(
        query_id="coord-test-1",
        transcript="What's the status of pump-001?",
        parsed_intent={"type": "query", "domains": ["industrial"]},
        widget_plan={"widgets": [{"scenario": "device_status"}]},
        processing_time_ms=1500,
    )
    print("  Recorded experience 1")

    rl.record_experience(
        query_id="coord-test-2",
        transcript="Show alerts",
        parsed_intent={"type": "query", "domains": ["alerts"]},
        widget_plan={"widgets": [{"scenario": "alerts_list"}]},
        processing_time_ms=1200,
    )
    print("  Recorded experience 2")

    # Update feedback
    success = rl.update_feedback("coord-test-1", rating="up")
    print(f"  Updated feedback: {success}")

    # Get status
    status = rl.get_status()
    print(f"  Status: running={status['running']}, buffer={status['buffer']}")

    # Get recent experiences
    recent = rl.get_recent_experiences(5)
    print(f"  Recent experiences: {len(recent)}")

    print("  ✓ ContinuousRL coordinator tests passed")
    return True


def test_background_trainer():
    """Test the background trainer (brief, without full training)."""
    print("\n=== Testing Background Trainer ===")

    from rl.background_trainer import BackgroundTrainer
    from rl.experience_buffer import Experience, ExperienceBuffer
    from rl.config import CONTINUOUS_RL_CONFIG

    buffer = ExperienceBuffer(max_size=100, persist_to_disk=False)

    # Add some experiences with feedback
    for i in range(20):
        exp = Experience(
            query_id=f"trainer-test-{i}",
            transcript=f"Query {i}",
            user_rating="up" if i % 2 == 0 else "down",
            processing_time_ms=1000 + i * 50,
            widget_plan={"widgets": [{"scenario": "test"}]},
        )
        buffer.add(exp)

    print(f"  Added {buffer.size()} experiences")

    # Create trainer (don't start it)
    config = CONTINUOUS_RL_CONFIG.copy()
    config["min_batch_size"] = 5

    trainer = BackgroundTrainer(buffer=buffer, config=config)
    stats = trainer.get_stats()
    print(f"  Trainer stats: {stats}")

    print("  ✓ Background trainer tests passed")
    return True


def main():
    print("=" * 60)
    print("CONTINUOUS RL SYSTEM TEST")
    print("=" * 60)

    tests = [
        test_experience_buffer,
        test_reward_signals,
        test_continuous_rl_coordinator,
        test_background_trainer,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
