import os
import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class Layer2Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'layer2'

    def ready(self):
        """Initialize the RL system on app startup."""
        # Only run in the main process (not in management commands or migrations)
        if os.environ.get("RUN_MAIN") == "true" or os.environ.get("GUNICORN_WORKER"):
            self._init_rl_system()

    def _init_rl_system(self):
        """Start the continuous RL system if enabled."""
        if os.environ.get("ENABLE_CONTINUOUS_RL", "true").lower() != "true":
            logger.info("Continuous RL disabled via ENABLE_CONTINUOUS_RL env var")
            return

        try:
            from rl.continuous import init_rl_system
            from .orchestrator import get_orchestrator

            orchestrator = get_orchestrator()
            rl = init_rl_system(
                widget_selector=getattr(orchestrator, 'widget_selector', None),
                fixture_selector=getattr(orchestrator, 'fixture_selector', None),
            )
            logger.info("Continuous RL system initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize RL system: {e}")
