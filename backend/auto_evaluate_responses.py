#!/usr/bin/env python
"""
Automated Response Evaluation using Claude Code CLI

This script uses Claude Code to evaluate AI widget selections and provide
thumbs up/down feedback automatically. Claude acts as the "human" supervisor,
creating a self-improving feedback loop.

Usage:
    python auto_evaluate_responses.py [--batch-size 10] [--continuous]

Features:
    - Claude evaluates widget selections for appropriateness
    - Automatic thumbs up/down ratings based on Claude's judgment
    - Stores ratings in database for RL training
    - Can run continuously or one-shot
"""

import argparse
import json
import logging
import os
import requests
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from django.utils import timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "command_center.settings")
import django
django.setup()

from feedback.models import WidgetRating

# API endpoint for feedback
FEEDBACK_API_URL = "http://127.0.0.1:8100/api/layer2/feedback/"


def load_experience_buffer() -> Dict:
    """Load the experience buffer from disk."""
    buffer_path = Path(__file__).parent.parent / 'rl_training_data' / 'experience_buffer.json'

    if not buffer_path.exists():
        logger.error(f"Experience buffer not found at {buffer_path}")
        return {"experiences": []}

    try:
        with open(buffer_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading experience buffer: {e}")
        return {"experiences": []}


def get_unrated_experiences(experiences: List[Dict], limit: int = 10) -> List[Dict]:
    """Get experiences that haven't been rated yet."""
    unrated = []

    for exp in experiences:
        query_id = exp.get('query_id')
        if not query_id:
            continue

        # Check if already rated
        existing = WidgetRating.objects.filter(entry_id=query_id).first()
        if existing:
            continue

        # Must have widget plan to evaluate
        if not exp.get('widget_plan') or not exp.get('widget_plan', {}).get('widgets'):
            continue

        unrated.append(exp)
        if len(unrated) >= limit:
            break

    return unrated


def evaluate_with_claude(experience: Dict) -> Optional[Dict]:
    """
    Use Claude Code CLI to perform detailed evaluation of widget selections.

    Returns:
        Dict with comprehensive feedback, or None if evaluation fails:
        {
            'overall_rating': 'up' or 'down',
            'confidence': 0.0-1.0,
            'reasoning': str,
            'widget_feedback': [
                {
                    'widget_index': int,
                    'widget_type': str,
                    'appropriateness_score': 0.0-1.0,
                    'size_appropriate': bool,
                    'issues': [str],
                    'strengths': [str]
                }
            ],
            'missing_widgets': [str],
            'suggested_improvements': [str],
            'query_understanding': str
        }
    """
    query_id = experience.get('query_id')
    transcript = experience.get('transcript', 'Unknown query')
    intent = experience.get('parsed_intent', {})
    widget_plan = experience.get('widget_plan', {})
    widgets = widget_plan.get('widgets', [])

    # Build context for Claude
    intent_type = intent.get('type', 'unknown')
    primary_char = intent.get('primary_characteristic', '')
    intent_desc = f"{intent_type}" + (f" ({primary_char})" if primary_char else "")
    domains = intent.get('domains', [])

    evaluation_prompt = f"""You are an expert evaluator for an industrial dashboard AI system. Provide comprehensive feedback on this widget selection.

**User Query**: "{transcript}"

**Intent Analysis**:
- Type: {intent_desc}
- Domains: {', '.join(domains) if domains else 'general'}
- Confidence: {intent.get('confidence', 0):.2f}

**Widgets Selected by AI**:
"""

    for i, widget in enumerate(widgets, 1):
        scenario = widget.get('scenario', 'unknown')
        size = widget.get('size', 'medium')
        relevance = widget.get('relevance', 0.0)
        evaluation_prompt += f"{i}. {scenario} (size: {size}, relevance: {relevance:.3f})\n"

    evaluation_prompt += """
**Available Widget Types**: KPI, Trend, Table, Gauge, Alert, Comparison, Chart, Distribution

**Your Task**: Provide detailed evaluation in JSON format:

```json
{
  "overall_rating": "GOOD" or "POOR",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence explanation of your rating",
  "query_understanding": "What the user is trying to accomplish",
  "widget_feedback": [
    {
      "widget_index": 1,
      "widget_type": "alert",
      "appropriateness_score": 0.0-1.0,
      "size_appropriate": true/false,
      "issues": ["specific issue 1", "issue 2"],
      "strengths": ["strength 1", "strength 2"]
    }
  ],
  "missing_widgets": ["widget type that should be included"],
  "suggested_improvements": ["specific actionable suggestion 1", "suggestion 2"]
}
```

**Evaluation Guidelines**:
- Be SPECIFIC: Point out exactly what works or doesn't work
- Be GENEROUS: Rate GOOD if the selection reasonably answers the query
- Be CONSTRUCTIVE: Always provide actionable improvement suggestions
- Consider context: Intent type and domains inform what's appropriate

Provide ONLY the JSON object, no other text:"""

    try:
        # Call Claude Code CLI
        logger.info(f"  Evaluating query_id={query_id}: {transcript[:50]}...")

        result = subprocess.run(
            ['claude'],
            input=evaluation_prompt,
            capture_output=True,
            text=True,
            timeout=45  # Increased timeout for detailed response
        )

        if result.returncode != 0:
            logger.warning(f"  Claude CLI failed: {result.stderr}")
            return None

        # Parse JSON response
        response_text = result.stdout.strip()

        # Extract JSON from markdown code blocks if present
        if '```json' in response_text:
            start = response_text.find('```json') + 7
            end = response_text.find('```', start)
            response_text = response_text[start:end].strip()
        elif '```' in response_text:
            start = response_text.find('```') + 3
            end = response_text.find('```', start)
            response_text = response_text[start:end].strip()

        try:
            evaluation = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.warning(f"  Failed to parse JSON: {e}")
            logger.debug(f"  Response text: {response_text[:200]}")
            return None

        # Normalize rating
        overall = evaluation.get('overall_rating', '').upper()
        if 'GOOD' in overall:
            evaluation['overall_rating'] = 'up'
            logger.info(f"  ✓ GOOD - Confidence: {evaluation.get('confidence', 0):.2f}")
        elif 'POOR' in overall:
            evaluation['overall_rating'] = 'down'
            logger.info(f"  ✗ POOR - Confidence: {evaluation.get('confidence', 0):.2f}")
        else:
            logger.warning(f"  ? Uncertain rating: {overall}")
            return None

        # Add metadata
        evaluation['query_id'] = query_id
        evaluation['evaluated_at'] = datetime.now().isoformat()

        logger.info(f"  Reasoning: {evaluation.get('reasoning', 'N/A')[:80]}...")

        return evaluation

    except subprocess.TimeoutExpired:
        logger.warning(f"  Timeout evaluating {query_id}")
        return None
    except Exception as e:
        logger.error(f"  Error evaluating {query_id}: {e}")
        return None


def create_rating(experience: Dict, evaluation: Dict):
    """
    Create a WidgetRating with rich feedback via API endpoint.

    This ensures BOTH the experience buffer (Tier1) and database (Tier2)
    get updated with detailed feedback for enhanced RL training.

    Args:
        experience: The experience dict
        evaluation: Rich evaluation dict from Claude with:
            - overall_rating: 'up' or 'down'
            - confidence: float
            - reasoning: str
            - widget_feedback: list
            - missing_widgets: list
            - suggested_improvements: list
    """
    query_id = experience.get('query_id')
    rating = evaluation['overall_rating']

    # Build comprehensive notes from evaluation
    notes_parts = [
        f"Auto-evaluated by Claude Sonnet 4.5",
        f"Confidence: {evaluation.get('confidence', 0):.2f}",
        f"Reasoning: {evaluation.get('reasoning', 'N/A')}",
        f"Query understanding: {evaluation.get('query_understanding', 'N/A')}",
    ]

    if evaluation.get('missing_widgets'):
        notes_parts.append(f"Missing: {', '.join(evaluation['missing_widgets'])}")

    if evaluation.get('suggested_improvements'):
        improvements = evaluation['suggested_improvements'][:2]  # First 2 suggestions
        notes_parts.append(f"Suggestions: {'; '.join(improvements)}")

    notes = " | ".join(notes_parts)

    # Build correction text with detailed feedback
    correction_parts = [evaluation.get('reasoning', '')]

    widget_feedback = evaluation.get('widget_feedback', [])
    if widget_feedback:
        correction_parts.append("\nPer-widget feedback:")
        for wf in widget_feedback[:3]:  # First 3 widgets
            idx = wf.get('widget_index', 0)
            wtype = wf.get('widget_type', 'unknown')
            score = wf.get('appropriateness_score', 0)
            correction_parts.append(f"  Widget {idx} ({wtype}): {score:.2f}/1.0")
            if wf.get('issues'):
                correction_parts.append(f"    Issues: {', '.join(wf['issues'][:2])}")

    correction = "\n".join(correction_parts)

    try:
        # Call API endpoint with rich feedback
        api_payload = {
            "query_id": query_id,
            "rating": rating,
            "interactions": [],  # Required field
            "correction": correction[:1000],  # Limit length for DB field
            # Rich evaluation fields from Claude Sonnet 4.5
            "evaluation_confidence": evaluation.get("confidence"),
            "evaluation_reasoning": evaluation.get("reasoning"),
            "query_understanding": evaluation.get("query_understanding"),
            "per_widget_feedback": evaluation.get("widget_feedback", []),
            "missing_widgets": evaluation.get("missing_widgets", []),
            "suggested_improvements": evaluation.get("suggested_improvements", []),
        }

        response = requests.post(
            FEEDBACK_API_URL,
            json=api_payload,
            timeout=5
        )

        if response.status_code == 200:
            logger.info(f"  ✓ Rich feedback saved via API: {rating}")

            # Also save the full evaluation to a separate JSON file for deep analysis
            save_detailed_evaluation(query_id, evaluation)
        else:
            # Fallback to direct DB write
            logger.warning(f"  API returned {response.status_code}, using direct DB write")
            now = timezone.now()
            WidgetRating.objects.create(
                entry_id=query_id,
                rating=rating,
                rated_at=now,
                device_id='claude-auto-evaluator',
                notes=notes[:500]  # Truncate to fit DB field
            )
            logger.info(f"  ✓ Rating saved to database: {rating}")
            save_detailed_evaluation(query_id, evaluation)

    except requests.exceptions.RequestException as e:
        # Fallback to direct DB write
        logger.warning(f"  API unreachable ({e}), using direct DB write")
        now = timezone.now()
        WidgetRating.objects.create(
            entry_id=query_id,
            rating=rating,
            rated_at=now,
            device_id='claude-auto-evaluator',
            notes=notes[:500]
        )
        logger.info(f"  ✓ Rating saved to database: {rating}")
        save_detailed_evaluation(query_id, evaluation)


def save_detailed_evaluation(query_id: str, evaluation: Dict):
    """Save detailed evaluation to JSON file for deep RL training."""
    try:
        evaluations_dir = Path(__file__).parent.parent / 'rl_training_data' / 'detailed_evaluations'
        evaluations_dir.mkdir(exist_ok=True)

        eval_file = evaluations_dir / f"{query_id}.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation, f, indent=2)

        logger.debug(f"  Saved detailed evaluation to {eval_file}")
    except Exception as e:
        logger.warning(f"  Failed to save detailed evaluation: {e}")


def evaluate_batch(batch_size: int = 10) -> Dict[str, int]:
    """Evaluate a batch of unrated experiences."""
    logger.info("=" * 70)
    logger.info("Automated Response Evaluation with Claude Code")
    logger.info("=" * 70)

    # Load experiences
    buffer_data = load_experience_buffer()
    experiences = buffer_data.get('experiences', [])

    logger.info(f"Total experiences in buffer: {len(experiences)}")

    # Get unrated
    unrated = get_unrated_experiences(experiences, limit=batch_size)

    if not unrated:
        logger.info("No unrated experiences found!")
        return {"evaluated": 0, "up": 0, "down": 0, "uncertain": 0}

    logger.info(f"Found {len(unrated)} unrated experiences")
    logger.info("")

    stats = {"evaluated": 0, "up": 0, "down": 0, "uncertain": 0}

    for i, exp in enumerate(unrated, 1):
        logger.info(f"[{i}/{len(unrated)}] Evaluating...")

        # Evaluate with Claude (returns rich evaluation dict)
        evaluation = evaluate_with_claude(exp)

        if evaluation and evaluation.get('overall_rating'):
            # Save to database with rich feedback
            create_rating(exp, evaluation)

            rating = evaluation['overall_rating']
            stats["evaluated"] += 1
            stats[rating] += 1

            # Log rich feedback summary
            confidence = evaluation.get('confidence', 0)
            missing = evaluation.get('missing_widgets', [])
            if missing:
                logger.info(f"  Missing widgets: {', '.join(missing[:3])}")
        else:
            stats["uncertain"] += 1

        logger.info("")

        # Small delay to avoid rate limits
        if i < len(unrated):
            time.sleep(2)

    return stats


def run_continuous(batch_size: int = 10, interval: int = 300):
    """Run continuous evaluation loop."""
    logger.info("Starting continuous evaluation mode")
    logger.info(f"Batch size: {batch_size}, Interval: {interval}s")
    logger.info("")

    iteration = 0

    while True:
        iteration += 1
        logger.info(f"\n{'='*70}")
        logger.info(f"Iteration {iteration}")
        logger.info(f"{'='*70}\n")

        try:
            stats = evaluate_batch(batch_size)

            logger.info("\n" + "=" * 70)
            logger.info(f"Batch Complete - Iteration {iteration}")
            logger.info("=" * 70)
            logger.info(f"  Evaluated: {stats['evaluated']}")
            logger.info(f"  Thumbs up: {stats['up']}")
            logger.info(f"  Thumbs down: {stats['down']}")
            logger.info(f"  Uncertain: {stats['uncertain']}")
            logger.info(f"\nNext evaluation in {interval}s...")

            time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("\n\nStopping continuous evaluation...")
            break
        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {e}")
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description='Automated response evaluation with Claude Code')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of experiences to evaluate per batch')
    parser.add_argument('--continuous', action='store_true', help='Run continuously')
    parser.add_argument('--interval', type=int, default=300, help='Seconds between evaluations (continuous mode)')
    args = parser.parse_args()

    if args.continuous:
        run_continuous(args.batch_size, args.interval)
    else:
        stats = evaluate_batch(args.batch_size)

        print("\n" + "=" * 70)
        print("Evaluation Complete!")
        print("=" * 70)
        print(f"  Evaluated: {stats['evaluated']}")
        print(f"  Thumbs up: {stats['up']}")
        print(f"  Thumbs down: {stats['down']}")
        print(f"  Uncertain: {stats['uncertain']}")
        print("")

        # Show current database stats
        total_ratings = WidgetRating.objects.count()
        up_count = WidgetRating.objects.filter(rating='up').count()
        down_count = WidgetRating.objects.filter(rating='down').count()

        print(f"Database Stats:")
        print(f"  Total ratings: {total_ratings}")
        print(f"  Up votes: {up_count}")
        print(f"  Down votes: {down_count}")
        print(f"  DPO pairs available: {up_count * down_count:,}")
        print("")

        return 0


if __name__ == '__main__':
    sys.exit(main())
