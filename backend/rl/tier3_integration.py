"""
Tier 3 Integration - Connects V4 Reasoning Distillation to RL Feedback Loop.

Automatically captures traces from production queries and triggers SFT training.
"""

import logging
import os
import random
import subprocess
import sys
import threading
from pathlib import Path
from queue import Queue

logger = logging.getLogger(__name__)

# Add claude-rl-agent to path
AGENT_DIR = Path(__file__).resolve().parent.parent.parent / "claude-rl-agent"
AGENT_SRC = AGENT_DIR / "src"

if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

# Thread-safe queue for trace capture requests
_trace_queue: Queue = Queue()
_worker_thread = None
_worker_lock = threading.Lock()


def _check_prerequisites() -> dict:
    """Check if all required components are available."""
    issues = []

    # Check Claude CLI
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            issues.append("Claude CLI not working")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        issues.append("Claude CLI not found")

    # Check Ollama
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            issues.append("Ollama not working")
        elif b"cc-widget-selector" not in result.stdout:
            issues.append("cc-widget-selector model not found in Ollama")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        issues.append("Ollama not found")

    # Check agent files exist
    if not (AGENT_SRC / "automated_runner.py").exists():
        issues.append("automated_runner.py not found")
    if not (AGENT_SRC / "v4_trace.py").exists():
        issues.append("v4_trace.py not found")

    return {
        "ready": len(issues) == 0,
        "issues": issues
    }


def should_capture_trace(experience) -> bool:
    """
    Decide whether to capture a trace for this experience.

    Criteria:
    - Enabled via ENABLE_TIER3_CAPTURE=true
    - Random sampling (15% of queries)
    - OR high-quality queries (evaluation_confidence > 0.8)
    """
    # Only capture if enabled
    if os.getenv("ENABLE_TIER3_CAPTURE", "false").lower() != "true":
        return False

    # Validate experience
    if not experience:
        return False

    # Random 15% sampling
    if random.random() < 0.15:
        return True

    # OR high-confidence queries
    if hasattr(experience, 'evaluation_confidence') and experience.evaluation_confidence:
        if experience.evaluation_confidence > 0.8:
            return True

    return False


def _trace_worker():
    """Background worker that processes trace capture requests."""
    from queue import Empty

    while True:
        item = None
        try:
            item = _trace_queue.get(timeout=60)
            if item is None:  # Shutdown signal
                break

            query, query_id = item
            _capture_trace_sync(query, query_id)

        except Empty:
            # Timeout — no items in queue, continue waiting
            continue
        except Exception as e:
            logger.error(f"Trace worker error: {e}")
        finally:
            if item is not None:
                _trace_queue.task_done()


def _capture_trace_sync(query: str, query_id: str = None):
    """Synchronously capture a trace (runs in worker thread)."""
    try:
        # Validate query
        if not query or not query.strip():
            logger.warning(f"Tier 3: Empty query for query_id={query_id}, skipping")
            return

        # Import here to avoid loading heavy deps on startup
        from automated_runner import AutomatedRunner

        # Ensure trace directory exists (V4TraceStore will create v4_traces subdir)
        data_dir = AGENT_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize runner
        runner = AutomatedRunner(
            claude_cli_path="claude",
            llama_model="cc-widget-selector:latest",
            storage_dir=str(data_dir)
        )

        # Run trace capture
        logger.info(f"Tier 3: Starting trace capture for query_id={query_id}")
        result = runner.run_parallel_comparison(query)

        if result:
            logger.info(f"Tier 3: Trace captured successfully: {result.get('trace_id')}")
        else:
            logger.warning(f"Tier 3: Trace capture returned no result")

    except ImportError as e:
        logger.error(f"Tier 3: Import failed (check claude-rl-agent setup): {e}")
    except Exception as e:
        logger.error(f"Tier 3: Trace capture failed for query_id={query_id}: {e}")


def capture_trace_async(query: str, query_id: str = None):
    """
    Capture a trace asynchronously (non-blocking).

    Queues the request for background processing by worker thread.
    """
    global _worker_thread

    # Validate query
    if not query or not query.strip():
        logger.debug(f"Tier 3: Skipping empty query for query_id={query_id}")
        return

    # Start worker thread if not running
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(
                target=_trace_worker,
                daemon=True,
                name="Tier3TraceWorker"
            )
            _worker_thread.start()
            logger.info("Tier 3: Started trace worker thread")

    # Queue the request
    _trace_queue.put((query, query_id))
    logger.debug(f"Tier 3: Queued trace capture for query_id={query_id}")


def check_and_trigger_training():
    """
    Check if enough traces accumulated and trigger SFT training.

    Should be called periodically (e.g., daily cron or background task).

    Returns:
        bool: True if training completed successfully, False otherwise
    """
    try:
        # Import here to avoid loading heavy deps on startup
        from v4_trace import V4TraceStore
        from sft_trainer import ClaudeSFTTrainer, SFTConfig

        # Ensure trace directory exists
        data_dir = AGENT_DIR / "data"
        trace_dir = data_dir / "v4_traces"
        if not trace_dir.exists():
            logger.info("Tier 3: No trace directory found, skipping training")
            return False

        # Check trace count
        store = V4TraceStore(data_dir=str(data_dir))
        traces = store.load_all()

        MIN_TRACES = int(os.getenv("TIER3_MIN_TRACES", "100"))

        if len(traces) < MIN_TRACES:
            logger.info(f"Tier 3: Only {len(traces)} traces, need {MIN_TRACES} for training")
            return False

        logger.info(f"Tier 3: Found {len(traces)} traces, starting training...")

        # Build dataset (returns Path to sft_dataset.jsonl)
        dataset_path = store.build_sft_dataset()

        if not dataset_path.exists():
            logger.error(f"Tier 3: Dataset file not created at {dataset_path}")
            return False

        logger.info(f"Tier 3: Dataset built at {dataset_path}")

        # Ensure output directory exists
        output_dir = AGENT_DIR / "models" / "sft_checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Train
        config = SFTConfig(
            output_dir=str(output_dir),
            num_epochs=3,
            batch_size=2,
        )

        trainer = ClaudeSFTTrainer(config)

        logger.info("Tier 3: Starting SFT training...")
        trainer.train(dataset_path=str(dataset_path))

        logger.info("Tier 3: Training completed successfully!")

        # Export to GGUF and deploy to Ollama
        try:
            from rl.export import export_to_ollama
            checkpoint_path = output_dir / "final"
            logger.info(f"Tier 3: Exporting checkpoint to GGUF: {checkpoint_path}")
            export_to_ollama(str(checkpoint_path), "cc-widget-selector")
            logger.info("Tier 3: Successfully exported and deployed to Ollama!")
        except Exception as e:
            logger.error(f"Tier 3: GGUF export failed (non-fatal): {e}")
            # Don't fail the whole training if export fails

        return True

    except ImportError as e:
        logger.error(f"Tier 3: Import failed (check dependencies): {e}")
        return False
    except Exception as e:
        logger.error(f"Tier 3: Training failed: {e}", exc_info=True)
        return False


def shutdown():
    """Gracefully shutdown the trace worker thread."""
    global _worker_thread

    if _worker_thread and _worker_thread.is_alive():
        logger.info("Tier 3: Shutting down trace worker...")
        _trace_queue.put(None)  # Shutdown signal
        _worker_thread.join(timeout=10)

        if _worker_thread.is_alive():
            logger.warning("Tier 3: Worker thread did not shut down cleanly")
        else:
            logger.info("Tier 3: Worker thread shut down successfully")
