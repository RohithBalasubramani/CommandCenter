from django.apps import AppConfig


class FeedbackConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'feedback'

    def ready(self):
        """Initialize signals and online learner on app startup."""
        # Import signals to register them
        from . import signals  # noqa: F401

        # Initialize online learner (optional - can be disabled via env)
        import os
        if os.getenv("ENABLE_ONLINE_LEARNING", "false").lower() == "true":
            try:
                from rl.online_learner import init_online_learner
                init_online_learner()
            except ImportError:
                pass  # RL module not installed
