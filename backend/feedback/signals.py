"""
Django Signals for Feedback Collection

Connects feedback events to the online learning system.
"""

import logging

from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import WidgetRating

logger = logging.getLogger(__name__)


@receiver(post_save, sender=WidgetRating)
def on_rating_saved(sender, instance, created, **kwargs):
    """
    Notify online learner of new feedback.

    Called after a WidgetRating is saved. Adds the rating to the
    online learning buffer for potential retraining.
    """
    try:
        from rl.online_learner import get_online_learner

        online_learner = get_online_learner()
        if online_learner is None:
            return

        # Only process new ratings (not updates)
        if not created:
            return

        # Add to online learner buffer
        feedback = {
            "entry_id": instance.entry_id,
            "rating": instance.rating,
            "tags": instance.tags or [],
            "notes": instance.notes or "",
        }

        should_retrain = online_learner.add_feedback(feedback)

        if should_retrain:
            logger.info("Online learner threshold reached, triggering retraining")
            # Queue async retraining task
            try:
                from django_q.tasks import async_task
                async_task(
                    "rl.online_learner.get_online_learner().trigger_retrain",
                    async_mode=True,
                )
            except ImportError:
                # Django-Q not installed, trigger directly in background
                online_learner.trigger_retrain(async_mode=True)

    except ImportError:
        # RL module not available, skip
        pass
    except Exception as e:
        logger.warning(f"Error in rating signal handler: {e}")
