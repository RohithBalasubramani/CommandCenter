import os
import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class Layer2Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'layer2'

    def ready(self):
        """Initialize the RL system on app startup."""
        logger.info("Layer2Config.ready() called")
        # Only run in the main process (not in management commands or migrations)
        # Auto-detect gunicorn workers even without explicit GUNICORN_WORKER env var
        is_dev_server = os.environ.get("RUN_MAIN") == "true"
        is_gunicorn_env = os.environ.get("GUNICORN_WORKER")
        is_gunicorn_auto = "gunicorn" in os.environ.get("SERVER_SOFTWARE", "")
        if not is_gunicorn_auto:
            try:
                import sys
                is_gunicorn_auto = any("gunicorn" in arg for arg in sys.argv)
            except Exception:
                pass
        if is_dev_server or is_gunicorn_env or is_gunicorn_auto:
            logger.info("RL system initialization triggered (gunicorn detected or RUN_MAIN set)")
            self._init_rl_system()
        else:
            logger.info("Skipping RL init (not main process)")

    def _init_rl_system(self):
        """Start the continuous RL system if enabled."""
        logger.info("_init_rl_system() called")

        if os.environ.get("ENABLE_CONTINUOUS_RL", "true").lower() != "true":
            logger.info("Continuous RL disabled via ENABLE_CONTINUOUS_RL env var")
            return

        try:
            logger.info("Importing RL modules...")
            from rl.continuous import init_rl_system
            from .orchestrator import get_orchestrator

            logger.info("Getting orchestrator...")
            orchestrator = get_orchestrator()

            logger.info("Calling init_rl_system()...")
            rl = init_rl_system(
                widget_selector=getattr(orchestrator, 'widget_selector', None),
                fixture_selector=getattr(orchestrator, 'fixture_selector', None),
            )
            logger.info(f"Continuous RL system initialized successfully - Running: {rl.running}")

            # Preload embedding model to avoid cold-start latency on first request
            self._preload_embedding_model()
        except Exception as e:
            import traceback
            logger.error(f"Failed to initialize RL system: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _preload_embedding_model(self):
        """Preload the sentence-transformers embedding model used by the scorer."""
        import threading

        def _load():
            try:
                from rl.lora_scorer import get_scorer
                scorer = get_scorer()
                # Trigger lazy-load of the embedding model with a dummy encode
                scorer._get_embedding("warmup")
                logger.info("Embedding model preloaded for scorer")
            except Exception as e:
                logger.warning(f"Failed to preload embedding model: {e}")

        # Run in background thread so it doesn't block startup
        threading.Thread(target=_load, daemon=True, name="embedding-preload").start()
