#!/usr/bin/env python3
"""
Diagnostic script to find reward polarity bug.

Raw user ratings: 220 up, 148 down (60% positive)
But computed rewards: 27% positive, 48% negative (inverted!)
"""

import sys
from rl.experience_buffer import ExperienceBuffer
from rl.reward_signals import RewardSignalAggregator

def main():
    buffer = ExperienceBuffer()
    aggregator = RewardSignalAggregator()

    # Get ALL experiences with feedback (includes those without explicit rating)
    experiences_with_feedback = [e for e in buffer.buffer if e.has_feedback()]

    print(f"Total experiences with feedback: {len(experiences_with_feedback)}")
    print()

    # Compute all rewards
    rewards = []
    ratings_count = {'up': 0, 'down': 0, 'none': 0}
    reward_signs = {'positive': 0, 'negative': 0, 'neutral': 0}

    for exp in experiences_with_feedback:
        reward = aggregator.compute_reward(exp)
        rewards.append(reward)

        # Count ratings
        if exp.user_rating == 'up':
            ratings_count['up'] += 1
        elif exp.user_rating == 'down':
            ratings_count['down'] += 1
        else:
            ratings_count['none'] += 1

        # Count reward signs
        if reward > 0.1:
            reward_signs['positive'] += 1
        elif reward < -0.1:
            reward_signs['negative'] += 1
        else:
            reward_signs['neutral'] += 1

    print("=" * 60)
    print("USER RATINGS (Explicit Feedback)")
    print("=" * 60)
    print(f"Thumbs UP:   {ratings_count['up']:3d} ({ratings_count['up']/len(experiences_with_feedback)*100:.1f}%)")
    print(f"Thumbs DOWN: {ratings_count['down']:3d} ({ratings_count['down']/len(experiences_with_feedback)*100:.1f}%)")
    print(f"No rating:   {ratings_count['none']:3d} ({ratings_count['none']/len(experiences_with_feedback)*100:.1f}%)")
    print()

    print("=" * 60)
    print("COMPUTED REWARDS (After Aggregation)")
    print("=" * 60)
    print(f"Positive (>0.1):  {reward_signs['positive']:3d} ({reward_signs['positive']/len(experiences_with_feedback)*100:.1f}%)")
    print(f"Negative (<-0.1): {reward_signs['negative']:3d} ({reward_signs['negative']/len(experiences_with_feedback)*100:.1f}%)")
    print(f"Neutral:          {reward_signs['neutral']:3d} ({reward_signs['neutral']/len(experiences_with_feedback)*100:.1f}%)")
    print()
    print(f"Mean reward: {sum(rewards)/len(rewards):.3f}")
    print()

    # Now check: for experiences WITH explicit ratings, do rewards match?
    rated_experiences = [e for e in experiences_with_feedback if e.user_rating in ['up', 'down']]

    up_positive = 0
    up_negative = 0
    down_positive = 0
    down_negative = 0

    for exp in rated_experiences:
        reward = aggregator.compute_reward(exp)

        if exp.user_rating == 'up':
            if reward > 0:
                up_positive += 1
            else:
                up_negative += 1
        else:  # down
            if reward > 0:
                down_positive += 1
            else:
                down_negative += 1

    print("=" * 60)
    print("POLARITY CHECK (Explicit Ratings vs Computed Rewards)")
    print("=" * 60)
    print(f"Thumbs UP → Positive reward: {up_positive}/{ratings_count['up']} ({up_positive/ratings_count['up']*100:.1f}%)")
    print(f"Thumbs UP → Negative reward: {up_negative}/{ratings_count['up']} ({up_negative/ratings_count['up']*100:.1f}%) ⚠️  BUG!")
    print()
    print(f"Thumbs DOWN → Negative reward: {down_negative}/{ratings_count['down']} ({down_negative/ratings_count['down']*100:.1f}%)")
    print(f"Thumbs DOWN → Positive reward: {down_positive}/{ratings_count['down']} ({down_positive/ratings_count['down']*100:.1f}%) ⚠️  BUG!")
    print()

    # Find experiences where implicit signals are creating the bias
    no_rating_rewards = [aggregator.compute_reward(e) for e in experiences_with_feedback if not e.user_rating]

    if no_rating_rewards:
        no_rating_positive = sum(1 for r in no_rating_rewards if r > 0.1)
        no_rating_negative = sum(1 for r in no_rating_rewards if r < -0.1)
        no_rating_mean = sum(no_rating_rewards) / len(no_rating_rewards)

        print("=" * 60)
        print("EXPERIENCES WITHOUT EXPLICIT RATING (Implicit Signals Only)")
        print("=" * 60)
        print(f"Total: {len(no_rating_rewards)}")
        print(f"Positive: {no_rating_positive} ({no_rating_positive/len(no_rating_rewards)*100:.1f}%)")
        print(f"Negative: {no_rating_negative} ({no_rating_negative/len(no_rating_rewards)*100:.1f}%)")
        print(f"Mean reward: {no_rating_mean:.3f}")
        print()
        print("👉 This might be pulling the overall distribution negative!")

if __name__ == "__main__":
    main()
