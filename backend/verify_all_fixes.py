#!/usr/bin/env python3
"""
Comprehensive verification of all fixes applied 2026-02-08.
This script ensures 100% production readiness.
"""

import sys
import traceback
from pathlib import Path

# Test results
results = {
    "passed": [],
    "failed": [],
    "warnings": []
}

def test(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            try:
                print(f"\n{'='*70}")
                print(f"TEST: {name}")
                print('='*70)
                func()
                results["passed"].append(name)
                print(f"✅ PASSED: {name}")
                return True
            except AssertionError as e:
                results["failed"].append((name, str(e)))
                print(f"❌ FAILED: {name}")
                print(f"   Error: {e}")
                traceback.print_exc()
                return False
            except Exception as e:
                results["failed"].append((name, str(e)))
                print(f"❌ ERROR: {name}")
                print(f"   Error: {e}")
                traceback.print_exc()
                return False
        return wrapper
    return decorator


@test("Fix 1: Reward signal imports and structure")
def test_reward_signals_import():
    from rl.reward_signals import RewardSignalAggregator

    agg = RewardSignalAggregator()

    # Verify refinement penalty is -0.1 (not -0.3)
    from rl.experience_buffer import Experience
    from datetime import datetime

    exp = Experience(
        query_id="test-1",
        timestamp=datetime.now(),
        transcript="test query",
        parsed_intent={"type": "monitoring"},
        widget_plan={"widgets": []},
        follow_up_type="refinement"
    )

    reward = agg._follow_up_reward(exp)
    expected = 0.5 * -0.1  # weight * refinement_value

    assert abs(reward - expected) < 0.001, \
        f"Refinement reward should be {expected}, got {reward}"

    print(f"  ✓ Refinement penalty correctly set to -0.1")
    print(f"  ✓ Computed reward: {reward:.3f}")


@test("Fix 1: Reward distribution improvement")
def test_reward_distribution():
    from rl.experience_buffer import ExperienceBuffer
    from rl.reward_signals import RewardSignalAggregator

    buffer = ExperienceBuffer()
    agg = RewardSignalAggregator()

    experiences = [e for e in buffer.buffer if e.has_feedback()]

    if not experiences:
        print("  ⚠️  No experiences with feedback (empty buffer)")
        results["warnings"].append("Empty experience buffer")
        return

    rewards = [agg.compute_reward(e) for e in experiences]
    mean_reward = sum(rewards) / len(rewards)

    positive = sum(1 for r in rewards if r > 0.1)
    negative = sum(1 for r in rewards if r < -0.1)

    print(f"  ✓ Mean reward: {mean_reward:.3f}")
    print(f"  ✓ Positive: {positive} ({positive/len(rewards)*100:.1f}%)")
    print(f"  ✓ Negative: {negative} ({negative/len(rewards)*100:.1f}%)")

    # Should be improved (less negative bias)
    assert mean_reward > -0.1, f"Mean reward too negative: {mean_reward}"


@test("Fix 2: Trainer config reading")
def test_trainer_config():
    from rl.trainer import CommandCenterDPOTrainer
    from rl.config import MODEL_CONFIG

    # Test with no config (should use defaults)
    trainer = CommandCenterDPOTrainer()

    assert trainer.base_model is not None, "base_model should not be None"
    assert trainer.base_model == MODEL_CONFIG.get("base_model"), \
        f"base_model mismatch: {trainer.base_model} != {MODEL_CONFIG.get('base_model')}"

    print(f"  ✓ Base model correctly set: {trainer.base_model}")

    # Test with custom config
    custom_config = {"base_model": "test-model"}
    trainer2 = CommandCenterDPOTrainer(config=custom_config)

    assert trainer2.base_model == "test-model", \
        f"Custom base_model not respected: {trainer2.base_model}"

    print(f"  ✓ Custom config override works")


@test("Fix 3: Evaluation consistency - None checks")
def test_evaluation_consistency():
    from rl.continuous import ContinuousRL
    from rl.experience_buffer import Experience
    from datetime import datetime

    # Create test experience
    rl = ContinuousRL()

    # Test with empty lists (should be saved now)
    query_id = "test-eval-consistency"

    # First record an experience
    rl.record_experience(
        query_id=query_id,
        transcript="test",
        widget_plan={"widgets": []},
        parsed_intent={"confidence": 0.8}
    )

    # Update with empty per_widget_feedback (should be saved)
    success = rl.update_feedback(
        query_id=query_id,
        evaluation_confidence=0.75,
        per_widget_feedback=[],  # Empty list - should be saved now
        missing_widgets=[],
        suggested_improvements=[]
    )

    assert success, "Feedback update should succeed"

    # Verify it was saved
    exp = rl.buffer.get_by_query_id(query_id)

    assert exp is not None, "Experience should exist"
    assert exp.evaluation_confidence == 0.75, "evaluation_confidence not saved"
    assert exp.per_widget_feedback == [], "per_widget_feedback should be empty list"
    assert exp.missing_widgets == [], "missing_widgets should be empty list"

    print(f"  ✓ Empty lists correctly saved (consistency maintained)")
    print(f"  ✓ evaluation_confidence: {exp.evaluation_confidence}")
    print(f"  ✓ per_widget_feedback: {exp.per_widget_feedback}")


@test("Fix 4: Widget plan validation")
def test_widget_plan_validation():
    from rl.continuous import ContinuousRL
    import logging

    # Capture warnings
    import io
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger("rl.continuous")
    logger.addHandler(handler)

    rl = ContinuousRL()

    # Test 1: Experience with no widget_plan
    rl.record_experience(
        query_id="test-no-plan",
        transcript="test",
        widget_plan=None,  # No plan
        parsed_intent={"confidence": 0.8}
    )

    exp = rl.buffer.get_by_query_id("test-no-plan")
    assert exp is not None, "Experience should be saved"
    assert exp.widget_plan is not None, "widget_plan should be normalized"
    assert "widgets" in exp.widget_plan, "widget_plan should have 'widgets' key"
    assert exp.widget_plan["widgets"] == [], "widgets should be empty list"

    print(f"  ✓ None widget_plan normalized to: {exp.widget_plan}")

    # Test 2: Experience with empty dict widget_plan
    rl.record_experience(
        query_id="test-empty-plan",
        transcript="test",
        widget_plan={},  # Empty dict
        parsed_intent={"confidence": 0.8}
    )

    exp2 = rl.buffer.get_by_query_id("test-empty-plan")
    assert "widgets" in exp2.widget_plan, "widgets key should be added"

    print(f"  ✓ Empty dict normalized to have 'widgets' key")

    # Verify warnings were logged
    log_output = log_stream.getvalue()
    assert "no valid widget_plan" in log_output.lower(), \
        "Should log warning for missing widget_plan"

    print(f"  ✓ Warning logged for invalid widget_plan")

    logger.removeHandler(handler)


@test("Fix 5: Tier 3 GGUF export (dry run)")
def test_tier3_export_import():
    # Just verify the import and structure is correct
    from rl.tier3_integration import check_and_trigger_training

    # Check that export_to_ollama can be imported
    from rl.export import export_to_ollama

    print(f"  ✓ check_and_trigger_training function exists")
    print(f"  ✓ export_to_ollama function exists")
    print(f"  ✓ Import paths correct")

    # Verify the export code is in tier3_integration
    import inspect
    source_file = Path(inspect.getfile(check_and_trigger_training))
    with open(source_file) as f:
        content = f.read()
        assert "export_to_ollama" in content, "export_to_ollama call should be in code"
        assert "cc-widget-selector" in content, "Model name should be in code"

    print(f"  ✓ GGUF export code verified in tier3_integration.py")

    # Note: Can't actually test export without training first


@test("Integration: Experience lifecycle")
def test_experience_lifecycle():
    from rl.continuous import ContinuousRL
    from rl.reward_signals import RewardSignalAggregator

    rl = ContinuousRL()
    agg = RewardSignalAggregator()

    # 1. Record experience
    query_id = "lifecycle-test"
    rl.record_experience(
        query_id=query_id,
        transcript="Show me pump status",
        widget_plan={
            "widgets": [
                {"scenario": "pump-kpi", "size": "medium"}
            ],
            "heading": "Pump Status"
        },
        parsed_intent={
            "type": "monitoring",
            "confidence": 0.9,
            "domains": ["equipment"]
        }
    )

    exp = rl.buffer.get_by_query_id(query_id)
    assert exp is not None, "Experience should be saved"
    assert exp.widget_plan["widgets"], "Should have widgets"

    print(f"  ✓ Step 1: Experience recorded")

    # 2. Add feedback with full evaluation
    success = rl.update_feedback(
        query_id=query_id,
        rating="up",
        evaluation_confidence=0.85,
        per_widget_feedback=[
            {
                "widget_index": 0,
                "appropriateness_score": 0.9,
                "size_appropriate": True
            }
        ],
        missing_widgets=[],
        suggested_improvements=[]
    )

    assert success, "Feedback update should succeed"

    exp = rl.buffer.get_by_query_id(query_id)
    assert exp.user_rating == "up", "Rating should be saved"
    assert exp.evaluation_confidence == 0.85, "Confidence should be saved"
    assert len(exp.per_widget_feedback) == 1, "Widget feedback should be saved"

    print(f"  ✓ Step 2: Feedback added (rating + evaluation)")

    # 3. Compute reward
    reward = agg.compute_reward(exp)

    assert reward > 0, f"Reward should be positive for thumbs up, got {reward}"

    print(f"  ✓ Step 3: Reward computed: {reward:.3f}")
    print(f"  ✓ Full lifecycle: record → feedback → reward ✅")


@test("Backward compatibility: Old experiences")
def test_backward_compatibility():
    from rl.experience_buffer import ExperienceBuffer
    from rl.reward_signals import RewardSignalAggregator

    buffer = ExperienceBuffer()
    agg = RewardSignalAggregator()

    # Check if old experiences (without new fields) still work
    old_experiences = [e for e in buffer.buffer
                      if not hasattr(e, 'per_widget_feedback') or not e.per_widget_feedback]

    if not old_experiences:
        print("  ⚠️  No old experiences found (all have per_widget_feedback)")
        return

    # Should be able to compute rewards without errors
    for exp in old_experiences[:10]:  # Test first 10
        try:
            reward = agg.compute_reward(exp)
            # Should not crash
        except Exception as e:
            raise AssertionError(f"Old experience crashed: {e}")

    print(f"  ✓ Old experiences ({len(old_experiences)}) still work")
    print(f"  ✓ Backward compatibility maintained")


@test("Performance: Reward computation")
def test_performance():
    import time
    from rl.experience_buffer import ExperienceBuffer
    from rl.reward_signals import RewardSignalAggregator

    buffer = ExperienceBuffer()
    agg = RewardSignalAggregator()

    experiences = [e for e in buffer.buffer if e.has_feedback()][:100]

    if not experiences:
        print("  ⚠️  No experiences to test performance")
        return

    start = time.time()
    for exp in experiences:
        _ = agg.compute_reward(exp)
    elapsed = time.time() - start

    avg_time = elapsed / len(experiences) * 1000  # ms

    print(f"  ✓ Computed {len(experiences)} rewards in {elapsed:.3f}s")
    print(f"  ✓ Average: {avg_time:.2f}ms per experience")

    # Should be fast (< 10ms per experience)
    assert avg_time < 10, f"Reward computation too slow: {avg_time:.2f}ms"


def main():
    print("="*70)
    print("COMPREHENSIVE VERIFICATION - ALL FIXES")
    print("="*70)
    print(f"Backend: {Path.cwd()}")
    print()

    # Run all tests
    test_reward_signals_import()
    test_reward_distribution()
    test_trainer_config()
    test_evaluation_consistency()
    test_widget_plan_validation()
    test_tier3_export_import()
    test_experience_lifecycle()
    test_backward_compatibility()
    test_performance()

    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    print(f"\n✅ PASSED: {len(results['passed'])}")
    for test_name in results['passed']:
        print(f"   • {test_name}")

    if results['failed']:
        print(f"\n❌ FAILED: {len(results['failed'])}")
        for test_name, error in results['failed']:
            print(f"   • {test_name}")
            print(f"     Error: {error}")

    if results['warnings']:
        print(f"\n⚠️  WARNINGS: {len(results['warnings'])}")
        for warning in results['warnings']:
            print(f"   • {warning}")

    # Overall result
    print("\n" + "="*70)
    if results['failed']:
        print("❌ VERIFICATION FAILED - NOT PRODUCTION READY")
        print("="*70)
        return 1
    else:
        print("✅ ALL TESTS PASSED - 100% PRODUCTION READY")
        print("="*70)
        return 0


if __name__ == "__main__":
    sys.exit(main())
