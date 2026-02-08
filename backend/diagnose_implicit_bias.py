#!/usr/bin/env python3
"""Find which implicit signals are creating negative bias."""

from rl.experience_buffer import ExperienceBuffer
from rl.reward_signals import RewardSignalAggregator

def main():
    buffer = ExperienceBuffer()
    aggregator = RewardSignalAggregator()

    # Get experiences without explicit rating
    no_rating_exps = [e for e in buffer.buffer if e.has_feedback() and not e.user_rating]

    print(f"Analyzing {len(no_rating_exps)} experiences without explicit rating...\n")

    # Aggregate signal contributions
    signals = {
        'follow_up': [],
        'engagement': [],
        'latency': [],
        'confidence': [],
        'eval_boost': [],
        'per_widget': [],
        'missing_widget': [],
        'size': [],
        'text_quality': [],
    }

    for exp in no_rating_exps:
        signals['follow_up'].append(aggregator._follow_up_reward(exp))
        signals['engagement'].append(aggregator._engagement_reward(exp))
        signals['latency'].append(aggregator._latency_reward(exp))
        signals['confidence'].append(aggregator._confidence_reward(exp))
        signals['eval_boost'].append(aggregator._evaluation_confidence_boost(exp))
        signals['per_widget'].append(aggregator._per_widget_appropriateness_reward(exp))
        signals['missing_widget'].append(aggregator._missing_widget_penalty_reward(exp))
        signals['size'].append(aggregator._size_appropriateness_reward(exp))
        signals['text_quality'].append(aggregator._text_quality_reward(exp))

    print("=" * 70)
    print("IMPLICIT SIGNAL CONTRIBUTIONS (experiences without explicit rating)")
    print("=" * 70)
    print(f"{'Signal':<25} {'Mean':>10} {'Nonzero':>10} {'Contribution':>15}")
    print("-" * 70)

    for name, values in signals.items():
        nonzero = [v for v in values if abs(v) > 0.001]
        mean = sum(values) / len(values) if values else 0
        nonzero_pct = len(nonzero) / len(values) * 100 if values else 0
        total_contribution = sum(values)

        print(f"{name:<25} {mean:>10.4f} {nonzero_pct:>9.1f}% {total_contribution:>15.2f}")

    # Find most negative signals
    print("\n" + "=" * 70)
    print("MOST NEGATIVE CONTRIBUTORS")
    print("=" * 70)
    total_contributions = {name: sum(values) for name, values in signals.items()}
    sorted_signals = sorted(total_contributions.items(), key=lambda x: x[1])

    for name, total in sorted_signals:
        if total < -1.0:  # Only show significant negative contributors
            count_negative = sum(1 for v in signals[name] if v < 0)
            print(f"{name}: {total:.2f} total ({count_negative} negative instances)")

if __name__ == "__main__":
    main()
