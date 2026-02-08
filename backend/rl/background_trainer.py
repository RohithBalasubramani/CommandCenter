"""
Background Trainer for Continuous RL — Three-Tier Architecture

Tier 1 — Low-Rank Scorer (milliseconds):
    Trains on every feedback event. A tiny PyTorch network (6,872 params)
    that learns score adjustments for widget selection. Runs on CPU.

Tier 2 — Unified LoRA DPO Fine-Tuning (periodic):
    Accumulates DPO preference pairs from BOTH widget selection quality
    AND voice response quality into a single pool. When enough pairs
    accumulate (>=150), triggers LoRA adapter training on llama3.1:8b.
    Exports to GGUF, hot-swaps in Ollama. Runs on GPU.

    Pair types:
    - "widget": chosen/rejected are JSON widget plans
    - "voice":  chosen/rejected are voice response text strings

Tier 3 — Reasoning Distillation SFT (periodic):
    Collects Claude thinking traces automatically (15% + high-confidence queries).
    When enough traces accumulate (>=100), triggers SFT training to teach
    LLaMA to think AND answer like Claude. Runs on GPU.

All tiers run in a background daemon thread, never blocking inference.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from .config import CONTINUOUS_RL_CONFIG, CHECKPOINTS_DIR, TRAINING_DATA_DIR
from .reward_signals import RewardSignalAggregator

if TYPE_CHECKING:
    from .experience_buffer import Experience, ExperienceBuffer

logger = logging.getLogger(__name__)

# DPO thresholds (unified)
DPO_MIN_PAIRS = int(CONTINUOUS_RL_CONFIG.get("lora_min_pairs", 150))
DPO_TRAIN_COOLDOWN = int(CONTINUOUS_RL_CONFIG.get("lora_train_cooldown", 1800))


class BackgroundTrainer:
    """
    Three-tier background trainer for continuous RL.

    Tier 1: Low-rank scorer — instant online learning from every feedback event
    Tier 2: Unified LoRA DPO — periodic deep fine-tuning of the LLM from both
            widget selection and voice response preference pairs
    Tier 3: Reasoning distillation SFT — periodic training on Claude thinking traces
            to teach LLaMA to think AND answer like Claude
    """

    def __init__(
        self,
        buffer: "ExperienceBuffer",
        config: dict = None,
        on_training_step: Optional[Callable] = None,
    ):
        self.buffer = buffer
        self.config = config or CONTINUOUS_RL_CONFIG
        self.on_training_step = on_training_step

        self.reward_aggregator = RewardSignalAggregator()

        # Training state
        self.running = False
        self.training_steps = 0
        self.total_samples_trained = 0
        self.avg_reward_history = []

        # Tier 1: Low-rank scorer (lazy loaded to avoid import on worker init)
        self._scorer = None
        self._scorer_steps = 0

        # Tier 1b: Composition scorer (lazy loaded)
        self._composition_scorer = None
        self._composition_steps = 0

        # Tier 2: Unified DPO (widget + voice pairs in one pool)
        self._dpo_pairs: list[dict] = []
        self._dpo_pairs_lock = threading.Lock()
        self._last_dpo_train_time = 0

        # Tier 3: Reasoning distillation SFT
        self._last_tier3_check_time = 0
        self._tier3_training = False
        self._dpo_training = False
        self._dpo_version = 0
        self._dpo_stats = {
            "total_trainings": 0,
            "total_pairs_trained": 0,
            "last_loss": None,
            "current_version": 0,
            "last_training_time": None,
        }
        self._training_log: list[dict] = []  # Audit trail

        # Widget selector reference (set by ContinuousRL)
        self.widget_selector = None
        self.fixture_selector = None

        # Thread
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def scorer(self):
        """Lazy-load the low-rank scorer."""
        if self._scorer is None:
            from .lora_scorer import get_scorer
            self._scorer = get_scorer()
        return self._scorer

    @property
    def composition_scorer(self):
        """Lazy-load the composition scorer."""
        if self._composition_scorer is None:
            from .composition_scorer import ContinuousCompositionTrainer
            self._composition_scorer = ContinuousCompositionTrainer()
        return self._composition_scorer

    def set_selectors(self, widget_selector, fixture_selector):
        """Set references to selectors for updating."""
        self.widget_selector = widget_selector
        self.fixture_selector = fixture_selector

    def start(self):
        """Start background training thread."""
        if self.running:
            logger.warning("Trainer already running")
            return

        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._training_loop,
            daemon=True,
            name="rl-background-trainer",
        )
        self._thread.start()
        logger.info("Background RL trainer started (Tier 1: scorer + Tier 2: unified DPO)")

    def stop(self, timeout: float = 5.0):
        """Stop training thread gracefully."""
        if not self.running:
            return

        logger.info("Stopping background trainer...")
        self._stop_event.set()
        self.running = False

        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Trainer thread did not stop cleanly")

        # Save scorer checkpoints
        if self._scorer:
            self._scorer._save_checkpoint()
        if self._composition_scorer:
            self._composition_scorer._save_checkpoint()

        # Save pending DPO pairs
        self._save_dpo_pairs()

        logger.info("Background trainer stopped")

    # ================================================================
    # Main Training Loop
    # ================================================================

    def _training_loop(self):
        """Main training loop — runs both tiers."""
        poll_interval = self.config.get("poll_interval", 5)
        train_interval = self.config.get("train_interval", 60)
        min_batch_size = self.config.get("min_batch_size", 16)
        batch_size = self.config.get("batch_size", 32) if "batch_size" in self.config else min_batch_size * 2

        last_train_time = 0

        # Load any saved DPO pairs from disk
        self._load_dpo_pairs()

        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                # Check if enough time has passed for Tier 1 batch
                if current_time - last_train_time < train_interval:
                    time.sleep(poll_interval)
                    continue

                # Get batch of experiences with feedback
                batch = self.buffer.sample_batch(batch_size, require_feedback=True)
                if len(batch) < min_batch_size:
                    logger.debug(f"Not enough feedback: {len(batch)} < {min_batch_size}")
                    time.sleep(poll_interval)
                    continue

                # Compute rewards
                rewards = [self.reward_aggregator.compute_reward(e) for e in batch]
                avg_reward = sum(rewards) / len(rewards)
                self.avg_reward_history.append(avg_reward)
                if len(self.avg_reward_history) > 100:
                    self.avg_reward_history = self.avg_reward_history[-100:]

                # ── Tier 1: Update low-rank scorer ──
                # Skip Tier 1 GPU-based updates while DPO is training
                # to avoid CUDA memory conflicts
                if not self._dpo_training:
                    self._tier1_update(batch, rewards)
                    self._tier1b_composition_update(batch, rewards)
                else:
                    logger.debug("Skipping Tier 1 update — DPO training in progress")

                # ── Tier 2: Accumulate DPO pairs (widget + voice) & trigger training ──
                self._accumulate_widget_pairs(batch, rewards)
                self._accumulate_voice_pairs(batch, rewards)
                self._maybe_train_dpo()

                # ── Tier 3: Check for SFT training trigger (every 30 min) ──
                self._maybe_train_sft()

                # Update fixture selector (rule-based, lightweight)
                if self.config.get("train_fixture_selector", True) and self.fixture_selector:
                    self._update_fixture_selector(batch, rewards)

                # Tracking
                self.training_steps += 1
                self.total_samples_trained += len(batch)
                last_train_time = current_time

                # Count pairs by type
                with self._dpo_pairs_lock:
                    widget_count = sum(1 for p in self._dpo_pairs if p.get("pair_type") == "widget")
                    voice_count = sum(1 for p in self._dpo_pairs if p.get("pair_type") == "voice")

                logger.info(
                    f"Training step {self.training_steps}: batch={len(batch)}, "
                    f"avg_reward={avg_reward:.3f}, scorer_steps={self._scorer_steps}, "
                    f"dpo_pairs={len(self._dpo_pairs)} (widget={widget_count}, voice={voice_count})"
                )

                # Periodically save DPO pairs to disk (crash protection)
                if self.training_steps % 10 == 0:
                    if self._dpo_pairs:
                        self._save_dpo_pairs()

                if self.on_training_step:
                    self.on_training_step({
                        "step": self.training_steps,
                        "batch_size": len(batch),
                        "avg_reward": avg_reward,
                        "total_samples": self.total_samples_trained,
                        "scorer_steps": self._scorer_steps,
                        "dpo_pairs_pending": len(self._dpo_pairs),
                    })

            except Exception as e:
                logger.error(f"Training error: {e}", exc_info=True)
                time.sleep(10)

    # ================================================================
    # Tier 1: Low-Rank Scorer (online learning)
    # ================================================================

    def _tier1_update(self, batch: list["Experience"], rewards: list[float]):
        """
        Train low-rank scorer on rich-evaluated experiences only.

        Enriched with:
        - QueryContext from parsed_intent (confidence, domains, entities, etc.)
        - Per-widget rewards from per_widget_feedback instead of flat reward
        - Widget position and candidate count for position-aware learning
        """
        from .lora_scorer import QueryContext

        for exp, reward in zip(batch, rewards):
            # Only train on rich-evaluated experiences
            if exp.evaluation_confidence is None:
                continue

            transcript = exp.transcript or ""
            widget_plan = exp.widget_plan or {}
            widgets = widget_plan.get("widgets", [])

            # Build enriched context from stored parsed_intent
            ctx = QueryContext.from_parsed_intent(exp.parsed_intent or {})
            ctx.num_candidate_widgets = len(widgets)

            # Per-widget reward refinement using Claude's per_widget_feedback
            per_widget_map = self._build_per_widget_reward_map(exp, reward)

            for position, widget in enumerate(widgets):
                scenario = widget.get("scenario", "")
                if not scenario:
                    continue

                # Use per-widget reward if available, else flat reward
                widget_reward = per_widget_map.get(position, reward)

                ctx.widget_position = position
                self.scorer.train_step(
                    transcript, scenario, widget_reward,
                    ctx=ctx,
                    num_candidates=len(widgets),
                    widget_position=position,
                )
                self._scorer_steps += 1

    def _build_per_widget_reward_map(
        self, exp: "Experience", flat_reward: float
    ) -> dict[int, float]:
        """
        Build per-widget reward using Claude's per_widget_feedback.

        Uses rank-normalization within each query so the scorer learns
        *relative* widget quality per-query rather than global popularity.
        Without this, uniformly-high appropriateness scores (e.g. mean=0.88
        for trend-multi-line) train the scorer to always rank the same widgets
        top regardless of query.

        Returns:
            Dict of widget_position (0-based) -> reward.
            Empty dict means use flat_reward for all widgets.
        """
        if not exp.per_widget_feedback:
            return {}

        # First pass: collect raw appropriateness per widget
        raw_scores = []  # list of (idx, appropriateness, size_ok)
        for wf in exp.per_widget_feedback:
            idx = wf.get("widget_index")
            if idx is None:
                continue
            appropriateness = wf.get("appropriateness_score", 0.5)
            size_ok = wf.get("size_appropriate", True)
            raw_scores.append((idx, appropriateness, size_ok))

        if not raw_scores:
            return {}

        # Rank-normalize: center on query mean so scorer learns relative quality.
        # A widget with appropriateness=0.85 in a query where all are 0.80+
        # gets near-zero reward (average). Same widget in a query with rest
        # at 0.30-0.50 gets strong positive (clearly the best).
        approp_values = [a for _, a, _ in raw_scores]
        query_mean = sum(approp_values) / len(approp_values)

        per_widget_map = {}
        for idx, appropriateness, size_ok in raw_scores:
            # Center on query mean, then scale to reward range [-2, 2]
            # Deviation from mean: e.g., 0.85 - 0.80 = 0.05 -> small reward
            # Deviation: 0.85 - 0.40 = 0.45 -> large reward
            centered = appropriateness - query_mean

            # Also blend in absolute quality so truly bad widgets (approp < 0.3)
            # still get penalized even if they're average for a bad query
            absolute = appropriateness - 0.5

            # 60% relative (query-specific) + 40% absolute (quality floor)
            widget_reward = (0.6 * centered + 0.4 * absolute) * 4.0

            # Penalize wrong sizing
            if not size_ok:
                widget_reward -= 0.5

            # Blend with flat reward (70/30 widget/flat)
            pure_signal = widget_reward
            widget_reward = 0.7 * widget_reward + 0.3 * flat_reward

            # Never let flat reward flip the per-widget signal direction
            if pure_signal != 0 and (widget_reward > 0) != (pure_signal > 0):
                widget_reward = pure_signal

            # Clip to reward bounds
            widget_reward = max(-2.0, min(2.0, widget_reward))

            # auto_evaluate_responses uses 1-based widget_index -> convert to 0-based
            per_widget_map[idx - 1] = widget_reward

        return per_widget_map

    def _tier1b_composition_update(self, batch: list["Experience"], rewards: list[float]):
        """
        Train composition scorer on whole-dashboard compositions.

        Each (transcript, [scenarios], reward) trains the composition model
        to learn which combinations of widgets work well for which queries.
        """
        for exp, reward in zip(batch, rewards):
            if exp.evaluation_confidence is None:
                continue

            transcript = exp.transcript or ""
            widget_plan = exp.widget_plan or {}
            scenarios = [
                w.get("scenario", "")
                for w in widget_plan.get("widgets", [])
                if w.get("scenario")
            ]
            if scenarios:
                self.composition_scorer.train_step(transcript, scenarios, reward)
                self._composition_steps += 1

    # ================================================================
    # Tier 2: Unified LoRA DPO (widget + voice pairs)
    # ================================================================

    def _accumulate_widget_pairs(self, batch: list["Experience"], rewards: list[float]):
        """
        Build DPO preference pairs from widget selection quality.

        Quality filters:
        - Chosen (positive): must have rich evaluation (evaluation_confidence >= 0.5)
          to confirm the response is genuinely high-quality
        - Rejected (negative): user downvote is sufficient signal; only requires
          a widget_plan and negative reward
        - Same-query pairs preferred; cross-query only when entities and domains
          match closely
        - Minimum reward gap of 0.3 between chosen and rejected
        - Deduplication by (prompt, chosen) to avoid training on repeats
        """
        if not batch:
            return

        # Chosen: require auto-evaluation to confirm quality
        positive = [
            (e, r) for e, r in zip(batch, rewards)
            if r > 0.1
            and e.evaluation_confidence is not None
            and e.evaluation_confidence >= 0.5
            and e.widget_plan
        ]
        # Rejected: user downvote is strong enough signal
        negative = [
            (e, r) for e, r in zip(batch, rewards)
            if r < -0.1
            and e.widget_plan
        ]

        if not positive or not negative:
            return

        # Build dedup set from existing pairs to avoid duplicates
        existing_keys = set()
        with self._dpo_pairs_lock:
            for p in self._dpo_pairs:
                existing_keys.add((p.get("prompt", "")[:200], p.get("chosen", "")[:200]))

        new_pairs = []
        for pos_e, pos_r in positive[:10]:
            for neg_e, neg_r in negative[:10]:
                # Require minimum reward gap for clear preference signal
                if (pos_r - neg_r) < 0.3:
                    continue

                if not self._intents_similar(pos_e.parsed_intent, neg_e.parsed_intent):
                    continue

                prompt = self._format_widget_prompt(pos_e)
                chosen = json.dumps(pos_e.widget_plan or {})

                # Dedup check
                key = (prompt[:200], chosen[:200])
                if key in existing_keys:
                    continue
                existing_keys.add(key)

                new_pairs.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": json.dumps(neg_e.widget_plan or {}),
                    "pos_reward": pos_r,
                    "neg_reward": neg_r,
                    "reward_gap": round(pos_r - neg_r, 3),
                    "timestamp": time.time(),
                    "rich_data": True,
                    "pair_type": "widget",
                })

        if new_pairs:
            with self._dpo_pairs_lock:
                self._dpo_pairs.extend(new_pairs)
                # Cap stored pairs to prevent unbounded memory growth
                if len(self._dpo_pairs) > 5000:
                    self._dpo_pairs = self._dpo_pairs[-5000:]
            logger.debug(f"Added {len(new_pairs)} widget DPO pairs (total: {len(self._dpo_pairs)})")

    def _voice_confidence(self, exp) -> float:
        """Get voice-specific confidence, falling back to widget confidence."""
        vc = getattr(exp, "voice_evaluation_confidence", None)
        if vc is not None:
            return vc
        if exp.evaluation_confidence is not None:
            return exp.evaluation_confidence
        return 0.0

    def _accumulate_voice_pairs(self, batch: list["Experience"], rewards: list[float]):
        """
        Build DPO preference pairs from voice response quality.

        Quality filters mirror widget pairs:
        - Chosen: high reward + rich eval + has voice_response
        - Rejected: low reward + has voice_response
        - Same intent matching, min reward gap, dedup
        """
        if not batch:
            return

        # Filter to experiences with voice responses
        positive = [
            (e, r) for e, r in zip(batch, rewards)
            if r > 0.1
            and getattr(e, "voice_response", None)
            and self._voice_confidence(e) >= 0.5
        ]
        negative = [
            (e, r) for e, r in zip(batch, rewards)
            if r < -0.1
            and getattr(e, "voice_response", None)
        ]

        if not positive or not negative:
            return

        # Build dedup set
        existing_keys = set()
        with self._dpo_pairs_lock:
            for p in self._dpo_pairs:
                existing_keys.add((p.get("prompt", "")[:200], p.get("chosen", "")[:200]))

        new_pairs = []
        for pos_e, pos_r in positive[:10]:
            for neg_e, neg_r in negative[:10]:
                if (pos_r - neg_r) < 0.3:
                    continue

                if not self._intents_similar(pos_e.parsed_intent, neg_e.parsed_intent):
                    continue

                prompt = self._format_voice_prompt(pos_e)
                chosen = pos_e.voice_response
                rejected = neg_e.voice_response

                # Dedup
                key = (prompt[:200], chosen[:200])
                if key in existing_keys:
                    continue
                existing_keys.add(key)

                new_pairs.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "pos_reward": pos_r,
                    "neg_reward": neg_r,
                    "reward_gap": round(pos_r - neg_r, 3),
                    "timestamp": time.time(),
                    "rich_data": True,
                    "pair_type": "voice",
                })

        if new_pairs:
            with self._dpo_pairs_lock:
                self._dpo_pairs.extend(new_pairs)
                if len(self._dpo_pairs) > 5000:
                    self._dpo_pairs = self._dpo_pairs[-5000:]
            logger.debug(f"Added {len(new_pairs)} voice DPO pairs (total: {len(self._dpo_pairs)})")

    def _maybe_train_dpo(self):
        """Check if we have enough pairs to trigger unified DPO training."""
        if self._dpo_training:
            return  # Already training

        if len(self._dpo_pairs) < DPO_MIN_PAIRS:
            return  # Not enough pairs yet

        if time.time() - self._last_dpo_train_time < DPO_TRAIN_COOLDOWN:
            return  # Cool down between training runs

        # Pre-flight: verify CUDA is available before acquiring lock
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("DPO pre-flight: CUDA not available, skipping training")
                return
            free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
            if free_mem < 4.0:
                logger.warning(f"DPO pre-flight: only {free_mem:.1f}GB GPU free, need 4GB+, skipping")
                return
        except Exception as e:
            logger.warning(f"DPO pre-flight: CUDA check failed ({e}), proceeding cautiously")

        # File-based lock so only one gunicorn worker trains at a time.
        # Includes stale lock detection: if the PID in the lock file is dead,
        # remove the lock and proceed (handles worker crashes/restarts).
        lock_file = TRAINING_DATA_DIR / "lora_training.lock"
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
        except FileExistsError:
            # Check if the lock holder is still alive
            try:
                lock_pid = int(lock_file.read_text().strip())
                os.kill(lock_pid, 0)  # Signal 0 = check if process exists
                logger.info(f"Another worker (PID {lock_pid}) holds GPU lock, skipping")
                return
            except (ValueError, ProcessLookupError, PermissionError, OSError):
                # Lock holder is dead — stale lock. Remove and acquire.
                logger.warning(f"Stale GPU lock detected (dead PID), reclaiming")
                try:
                    lock_file.unlink(missing_ok=True)
                    fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(os.getpid()).encode())
                    os.close(fd)
                except (FileExistsError, OSError):
                    logger.info("Lost lock race after stale cleanup, skipping")
                    return

        # Count pairs by type for logging
        with self._dpo_pairs_lock:
            widget_count = sum(1 for p in self._dpo_pairs if p.get("pair_type") == "widget")
            voice_count = sum(1 for p in self._dpo_pairs if p.get("pair_type") == "voice")

        is_first = self._dpo_stats["total_trainings"] == 0
        n_pairs = len(self._dpo_pairs)

        if is_first:
            logger.info(
                f"=== FIRST DPO TRAINING === "
                f"pairs={n_pairs} (widget={widget_count}, voice={voice_count}), "
                f"tier1_steps={self._scorer_steps}"
            )
        else:
            logger.info(
                f"Triggering DPO training with {n_pairs} pairs "
                f"(widget={widget_count}, voice={voice_count})"
            )

        train_thread = threading.Thread(
            target=self._run_dpo_training,
            daemon=True,
            name="rl-dpo-trainer",
        )
        train_thread.start()

    def _run_dpo_training(self):
        """
        Run unified LoRA DPO training and deploy to Ollama.

        Trains on both widget selection and voice response pairs.
        This runs in a separate thread and can take 10-30 minutes.
        """
        self._dpo_training = True
        self._last_dpo_train_time = time.time()
        start_time = time.time()
        training_pairs = []
        is_first = self._dpo_stats["total_trainings"] == 0
        audit_entry = {
            "timestamp": time.time(),
            "is_first_training": is_first,
            "status": "started",
        }

        try:
            # Snapshot pairs for training — keep backup on disk until training succeeds
            with self._dpo_pairs_lock:
                training_pairs = list(self._dpo_pairs)
                self._dpo_pairs.clear()

            audit_entry["num_pairs"] = len(training_pairs)
            audit_entry["widget_pairs"] = sum(1 for p in training_pairs if p.get("pair_type") == "widget")
            audit_entry["voice_pairs"] = sum(1 for p in training_pairs if p.get("pair_type") == "voice")

            # Save backup so pairs survive worker crashes during training
            backup_path = TRAINING_DATA_DIR / "dpo_pairs_training_backup.json"
            try:
                with open(backup_path, "w") as f:
                    json.dump(training_pairs, f)
            except Exception as e:
                logger.warning(f"Failed to save DPO backup: {e}")

            # Log reward distribution for diagnostics
            rewards = [p.get("pos_reward", 0) for p in training_pairs]
            neg_rewards = [p.get("neg_reward", 0) for p in training_pairs]
            gaps = [p.get("reward_gap", 0) for p in training_pairs]
            if rewards:
                logger.info(
                    f"DPO pair stats: "
                    f"pos_reward=[{min(rewards):.2f}, {max(rewards):.2f}], "
                    f"neg_reward=[{min(neg_rewards):.2f}, {max(neg_rewards):.2f}], "
                    f"reward_gap=[{min(gaps):.2f}, {max(gaps):.2f}], "
                    f"avg_gap={sum(gaps)/len(gaps):.2f}"
                )

            logger.info(f"Starting unified DPO training with {len(training_pairs)} pairs")

            # Build HuggingFace dataset from pairs
            dataset = self._pairs_to_dataset(training_pairs)
            if dataset is None or len(dataset) < 10:
                logger.warning("Too few valid training pairs, skipping")
                audit_entry["status"] = "skipped_too_few"
                with self._dpo_pairs_lock:
                    self._dpo_pairs.extend(training_pairs)  # Put back
                return

            # Split train/eval
            split = dataset.train_test_split(test_size=0.1, seed=42)
            train_ds = split["train"]
            eval_ds = split["test"]

            # Configure for incremental training
            version = self._dpo_version + 1
            output_dir = str(CHECKPOINTS_DIR / f"dpo_v{version}")

            # Run DPO training
            from .trainer import CommandCenterDPOTrainer

            trainer = CommandCenterDPOTrainer()
            result = trainer.train(
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                output_dir=output_dir,
            )

            if result.success:
                duration = time.time() - start_time
                logger.info(
                    f"DPO training complete: loss={result.final_loss:.4f}, "
                    f"samples={result.train_samples}, "
                    f"duration={duration:.0f}s"
                )

                # Evaluation gate: reject models with poor convergence
                MAX_ACCEPTABLE_LOSS = 0.7
                if result.final_loss is not None and result.final_loss > MAX_ACCEPTABLE_LOSS:
                    logger.warning(
                        f"DPO eval gate FAILED: loss {result.final_loss:.4f} > "
                        f"threshold {MAX_ACCEPTABLE_LOSS}. Skipping deployment."
                    )
                    audit_entry["status"] = "eval_gate_failed"
                    audit_entry["final_loss"] = result.final_loss
                    with self._dpo_pairs_lock:
                        self._dpo_pairs.extend(training_pairs)
                    return

                # Free GPU memory before export (training model still holds VRAM)
                del trainer
                import gc
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                    logger.info("Freed GPU memory before export")
                except Exception:
                    pass

                # Deploy to Ollama
                self._deploy_dpo(output_dir, version)

                # Update stats
                self._dpo_version = version
                self._dpo_stats["total_trainings"] += 1
                self._dpo_stats["total_pairs_trained"] += len(training_pairs)
                self._dpo_stats["last_loss"] = result.final_loss
                self._dpo_stats["current_version"] = version
                self._dpo_stats["last_training_time"] = time.time()

                audit_entry["status"] = "success"
                audit_entry["final_loss"] = result.final_loss
                audit_entry["duration_s"] = round(duration, 1)
                audit_entry["version"] = version

                if is_first:
                    logger.info(
                        f"=== FIRST DPO TRAINING SUCCEEDED === "
                        f"loss={result.final_loss:.4f}, v{version}, {duration:.0f}s"
                    )

                # Clean old versions and remove training backup
                self._cleanup_old_versions(keep=2)
                backup_path = TRAINING_DATA_DIR / "dpo_pairs_training_backup.json"
                backup_path.unlink(missing_ok=True)

            else:
                logger.error(f"DPO training failed: {result.error_message}")
                audit_entry["status"] = "train_failed"
                audit_entry["error"] = str(result.error_message)
                # Put pairs back for retry
                with self._dpo_pairs_lock:
                    self._dpo_pairs.extend(training_pairs)

        except Exception as e:
            logger.error(f"DPO training error: {e}", exc_info=True)
            audit_entry["status"] = "exception"
            audit_entry["error"] = str(e)
            # Put pairs back so they aren't lost on crash/OOM
            if training_pairs:
                with self._dpo_pairs_lock:
                    self._dpo_pairs.extend(training_pairs)
                logger.info(f"Restored {len(training_pairs)} DPO pairs after training failure")
        finally:
            self._dpo_training = False
            # Release file-based lock
            lock_file = TRAINING_DATA_DIR / "lora_training.lock"
            try:
                lock_file.unlink(missing_ok=True)
            except OSError:
                pass
            # Persist audit entry
            audit_entry["duration_s"] = audit_entry.get("duration_s", round(time.time() - start_time, 1))
            self._training_log.append(audit_entry)
            self._persist_audit_log(audit_entry)

    def _pairs_to_dataset(self, pairs: list[dict]):
        """Convert DPO pairs to HuggingFace dataset."""
        try:
            from datasets import Dataset

            records = []
            for pair in pairs:
                records.append({
                    "prompt": pair["prompt"],
                    "chosen": pair["chosen"],
                    "rejected": pair["rejected"],
                })

            if not records:
                return None

            return Dataset.from_list(records)
        except ImportError:
            logger.error("datasets package not installed")
            return None

    def _deploy_dpo(self, checkpoint_dir: str, version: int):
        """Deploy trained LoRA to Ollama via GGUF export."""
        try:
            from .export import export_to_ollama

            model_name = f"cc-widget-selector-v{version}"

            result = export_to_ollama(
                checkpoint_path=str(Path(checkpoint_dir) / "final"),
                model_name=model_name,
                register=True,
            )

            if result.success:
                logger.info(f"Deployed DPO v{version} to Ollama as '{model_name}'")

                # Update the model reference for the widget selector
                import os
                os.environ["OLLAMA_MODEL_FAST"] = model_name
                logger.info(f"Updated OLLAMA_MODEL_FAST → {model_name}")
            else:
                logger.error(f"DPO deployment failed: {result.error_message}")

        except Exception as e:
            logger.error(f"DPO deployment error: {e}", exc_info=True)

    def _cleanup_old_versions(self, keep: int = 2):
        """Remove old LoRA checkpoints, keeping the most recent N."""
        import shutil
        versions = sorted(
            [d for d in CHECKPOINTS_DIR.glob("dpo_v*") if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
        )
        for old in versions[:-keep]:
            logger.info(f"Cleaning up old checkpoint: {old.name}")
            shutil.rmtree(old, ignore_errors=True)

    def _persist_audit_log(self, entry: dict):
        """Append a training audit entry to persistent log on disk."""
        try:
            TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
            log_path = TRAINING_DATA_DIR / "dpo_training_audit.jsonl"
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to persist audit log: {e}")

    def _save_dpo_pairs(self):
        """Persist pending DPO pairs to disk for recovery."""
        if not self._dpo_pairs:
            return
        TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = TRAINING_DATA_DIR / "pending_dpo_pairs.json"
        with self._dpo_pairs_lock:
            with open(path, "w") as f:
                json.dump(self._dpo_pairs, f)
        logger.info(f"Saved {len(self._dpo_pairs)} pending DPO pairs to disk")

    def _load_dpo_pairs(self):
        """Load pending DPO pairs from disk, filtering to only rich-evaluated pairs.

        Also recovers from training backup if a previous training run crashed
        before completing (e.g., worker killed during training).
        """
        loaded = 0

        # First, check for crash-recovery backup from a failed training run
        backup_path = TRAINING_DATA_DIR / "dpo_pairs_training_backup.json"
        if backup_path.exists():
            try:
                with open(backup_path) as f:
                    backup_pairs = json.load(f)
                rich_backup = [p for p in backup_pairs if p.get("rich_data")]
                if rich_backup:
                    with self._dpo_pairs_lock:
                        self._dpo_pairs.extend(rich_backup)
                    loaded += len(rich_backup)
                    logger.info(f"Recovered {len(rich_backup)} DPO pairs from training backup (previous crash)")
                backup_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to load DPO backup: {e}")

        # Then load any pending pairs
        path = TRAINING_DATA_DIR / "pending_dpo_pairs.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                pairs = json.load(f)
            # Only load pairs that were built from rich evaluation data
            rich_pairs = [p for p in pairs if p.get("rich_data")]
            skipped = len(pairs) - len(rich_pairs)
            with self._dpo_pairs_lock:
                self._dpo_pairs.extend(rich_pairs)
            loaded += len(rich_pairs)
            logger.info(f"Loaded {len(rich_pairs)} rich DPO pairs from disk (skipped {skipped} non-rich)")
        except Exception as e:
            logger.warning(f"Failed to load DPO pairs: {e}")

        if loaded:
            logger.info(f"Total DPO pairs loaded: {loaded}")

    # ================================================================
    # Tier 3: Reasoning Distillation SFT (periodic)
    # ================================================================

    def _maybe_train_sft(self):
        """
        Check if enough Claude traces accumulated and trigger SFT training.

        Checks every 30 minutes. If >=100 traces found, triggers training
        in a background thread (non-blocking).
        """
        current_time = time.time()
        check_interval = 1800  # 30 minutes

        # Check at most once per interval
        if current_time - self._last_tier3_check_time < check_interval:
            return

        self._last_tier3_check_time = current_time

        # Skip if already training
        if self._tier3_training:
            logger.debug("Tier 3 SFT training already in progress, skipping check")
            return

        # Import here to avoid loading heavy deps at startup
        try:
            from rl.tier3_integration import check_and_trigger_training
        except ImportError:
            logger.debug("Tier 3 integration not available (tier3_integration.py missing)")
            return

        # Launch in background thread (SFT training takes ~5 min)
        def train_worker():
            try:
                self._tier3_training = True
                logger.info("Tier 3: Checking for SFT training trigger...")
                success = check_and_trigger_training()
                if success:
                    logger.info("Tier 3: SFT training completed successfully!")
                else:
                    logger.debug("Tier 3: Training not triggered (not enough traces or failed)")
            except Exception as e:
                logger.error(f"Tier 3: Training failed: {e}", exc_info=True)
            finally:
                self._tier3_training = False

        thread = threading.Thread(target=train_worker, daemon=True, name="Tier3SFTTrainer")
        thread.start()
        logger.info("Tier 3: Launched SFT training check in background")

    # ================================================================
    # Fixture Selector (rule-based)
    # ================================================================

    def _update_fixture_selector(self, batch: list["Experience"], rewards: list[float]):
        """Update fixture selector scoring weights."""
        fixture_rewards: dict[tuple[str, str], list[float]] = {}

        for exp, reward in zip(batch, rewards):
            for scenario, fixture in exp.fixtures.items():
                key = (scenario, fixture)
                if key not in fixture_rewards:
                    fixture_rewards[key] = []
                fixture_rewards[key].append(reward)

        if hasattr(self.fixture_selector, "update_preferences"):
            for (scenario, fixture), rewards_list in fixture_rewards.items():
                avg_reward = sum(rewards_list) / len(rewards_list)
                self.fixture_selector.update_preferences(scenario, fixture, avg_reward)
            logger.debug(f"Updated fixture preferences for {len(fixture_rewards)} fixtures")

    # ================================================================
    # Helpers
    # ================================================================

    def _intents_similar(self, intent1: dict, intent2: dict) -> bool:
        """
        Check if two intents are similar enough to form a valid DPO pair.

        Strict matching: requires same intent type, overlapping domains,
        AND similar entity types (e.g. both about pumps, or both about
        cooling towers). This prevents noisy cross-device pairs like
        "pump vibration chosen vs cooling tower status rejected".
        """
        # Must be same intent type
        if intent1.get("type") != intent2.get("type"):
            return False

        # Must share at least one domain
        domains1 = set(intent1.get("domains", []))
        domains2 = set(intent2.get("domains", []))
        if not domains1 & domains2:
            return False

        # Entity similarity: extract device types and check overlap
        entities1 = intent1.get("entities", {})
        entities2 = intent2.get("entities", {})

        devices1 = set(self._normalize_devices(entities1.get("devices", [])))
        devices2 = set(self._normalize_devices(entities2.get("devices", [])))

        # If both have devices, they must share at least one device type
        if devices1 and devices2:
            if not devices1 & devices2:
                return False

        # Check primary_characteristic similarity if available
        char1 = intent1.get("primary_characteristic", "")
        char2 = intent2.get("primary_characteristic", "")
        if char1 and char2 and char1 != char2:
            return False

        return True

    @staticmethod
    def _normalize_devices(devices: list) -> list[str]:
        """Normalize device names to canonical types for comparison.

        'pump-002' -> 'pump', 'cooling tower 1' -> 'cooling tower',
        'UPS systems' -> 'ups', 'compressors' -> 'compressor'
        """
        import re
        normalized = []
        for d in devices:
            d = str(d).lower().strip()
            # Strip trailing numbers, dashes, spaces
            d = re.sub(r'[\s\-_]*\d+$', '', d).strip()
            # Strip common suffixes: "systems", "units"
            d = re.sub(r'\s+(systems?|units?)$', '', d)
            # Depluralize simple cases (pumps->pump, compressors->compressor)
            # But not short words (ups, aps) or words ending in 'ss'
            if d.endswith('s') and not d.endswith('ss') and len(d) > 3:
                d = d[:-1]
            if d:
                normalized.append(d)
        return normalized

    def _format_widget_prompt(self, experience: "Experience") -> str:
        """
        Format experience into widget selection prompt.

        Enhanced with rich evaluation data from Claude Sonnet 4.5 when available.
        """
        intent = experience.parsed_intent or {}
        lines = [
            f"User query: {experience.transcript}",
            f"Domains: {', '.join(intent.get('domains', []))}",
        ]
        entities = intent.get("entities", {})
        if entities:
            lines.append(f"Entities: {', '.join(f'{k}={v}' for k, v in entities.items())}")

        # Add rich evaluation context from Claude Sonnet 4.5
        if experience.query_understanding:
            lines.append(f"Goal: {experience.query_understanding}")
        if experience.missing_widgets:
            missing = ", ".join(str(w) for w in experience.missing_widgets)
            lines.append(f"Consider adding: {missing}")
        if experience.suggested_improvements:
            # Include top 2 suggestions
            suggestions = experience.suggested_improvements[:2]
            if suggestions:
                lines.append(f"Improvements: {'; '.join(str(s) for s in suggestions)}")

        return "\n".join(lines)

    def _format_voice_prompt(self, experience: "Experience") -> str:
        """Format voice response generation prompt from experience."""
        widgets = (experience.widget_plan or {}).get("widgets", [])
        widget_summary = ", ".join(
            f"{w.get('scenario', '')} ({w.get('why', '')})" for w in widgets[:5]
        )
        heading = (experience.widget_plan or {}).get("heading", "Dashboard")

        return (
            f"You are Command Center, an industrial operations voice assistant.\n"
            f'Dashboard built: "{heading}" with {len(widgets)} widgets showing: {widget_summary}\n'
            f"User question: {experience.transcript}\n"
            f"Response:"
        )

    def get_stats(self) -> dict:
        """Get training statistics for all three tiers."""
        scorer_stats = self._scorer.get_stats() if self._scorer else {}
        composition_stats = self._composition_scorer.get_stats() if self._composition_scorer else {}

        # Count pairs by type
        with self._dpo_pairs_lock:
            widget_pairs = sum(1 for p in self._dpo_pairs if p.get("pair_type") == "widget")
            voice_pairs = sum(1 for p in self._dpo_pairs if p.get("pair_type") == "voice")

        return {
            "running": self.running,
            "training_steps": self.training_steps,
            "total_samples_trained": self.total_samples_trained,
            "avg_reward_trend": (
                sum(self.avg_reward_history[-10:]) / max(len(self.avg_reward_history[-10:]), 1)
                if self.avg_reward_history else 0
            ),
            "recent_rewards": self.avg_reward_history[-10:] if self.avg_reward_history else [],
            # Tier 1 stats
            "tier1_scorer": scorer_stats,
            # Tier 1b stats
            "tier1b_composition": composition_stats,
            # Tier 2: Unified DPO stats
            "dpo": {
                "training_in_progress": self._dpo_training,
                "pending_pairs": len(self._dpo_pairs),
                "pending_widget_pairs": widget_pairs,
                "pending_voice_pairs": voice_pairs,
                "min_pairs_for_training": DPO_MIN_PAIRS,
                "cooldown_remaining_s": max(0, int(
                    DPO_TRAIN_COOLDOWN - (time.time() - self._last_dpo_train_time)
                )) if self._last_dpo_train_time > 0 else 0,
                "ready_to_train": (
                    not self._dpo_training
                    and len(self._dpo_pairs) >= DPO_MIN_PAIRS
                    and (time.time() - self._last_dpo_train_time) >= DPO_TRAIN_COOLDOWN
                ),
                "training_history": self._training_log[-5:],
                **self._dpo_stats,
            },
            # Tier 3: SFT stats
            "tier3_sft": {
                "training_in_progress": self._tier3_training,
                "last_check_time": self._last_tier3_check_time,
                "check_interval_s": 1800,  # 30 min
                "next_check_in_s": max(0, int(
                    1800 - (time.time() - self._last_tier3_check_time)
                )) if self._last_tier3_check_time > 0 else 0,
            },
        }
