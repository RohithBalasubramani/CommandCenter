"""
Low-Rank Scorer — Lightweight neural reranker for widget selection.

Tier 1 of the continuous RL system. Trains in milliseconds from each
feedback event and re-ranks LLM widget outputs immediately.

Architecture:
    Input: intent embedding (768-dim)
         + widget one-hot (19-dim)
         + query features (9-dim): confidence, domain/entity counts, flags
         + query type one-hot (7-dim)
         + widget context (5-dim): position, candidate count, safety
         + primary characteristic one-hot (5-dim)
         = 813-dim total

    Hidden: Low-rank factorization  W = A @ B  where A: (813, r), B: (r, 64)
    Output: score adjustment in [-1, 1] via tanh

    Total parameters at rank=32: 813*32 + 32*64 + 64*1 = 28,160 (small)

Anti-overfitting measures:
    - Replay buffer: stores experiences, samples mini-batches instead of
      grinding on the same data every cycle
    - Train/val split: 80/20 split, tracks validation loss to detect overfit
    - LR decay: halves learning rate when val loss plateaus
    - Dropout: 0.1 dropout in hidden layer for regularization
    - Skip-if-seen: new experiences go into the buffer, training samples
      from the full buffer so each example is seen proportionally
"""

import logging
import os
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

# Paths
_RL_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _RL_DIR.parent.parent
SCORER_CHECKPOINT_DIR = _PROJECT_DIR / "rl_checkpoints" / "scorer"

# ============================================================
# Feature dimensions
# ============================================================
INTENT_EMBEDDING_DIM = 768   # BGE-base-en-v1.5 output dim
NUM_SCENARIOS = 19            # Number of widget scenarios in catalog
QUERY_FEATURE_DIM = 9        # Scalar query-level features
QUERY_TYPE_DIM = 7            # One-hot query type encoding
WIDGET_CTX_DIM = 5            # Widget context features (position, count, safety)
PRIMARY_CHAR_DIM = 5          # One-hot primary characteristic

INPUT_DIM = (INTENT_EMBEDDING_DIM + NUM_SCENARIOS + QUERY_FEATURE_DIM
             + QUERY_TYPE_DIM + WIDGET_CTX_DIM + PRIMARY_CHAR_DIM)  # 813

# Old dimension for checkpoint migration
_OLD_INPUT_DIM = 787

# Replay buffer config
REPLAY_BUFFER_SIZE = 2000
REPLAY_BATCH_SIZE = 8
VAL_SPLIT = 0.2
LR_PATIENCE = 200  # Steps without val improvement before LR decay
MIN_LR = 1e-5

# Encoding lookup tables
QUERY_TYPES = [
    "query", "action_reminder", "action_message", "action_control",
    "action_task", "conversation", "out_of_scope",
]
PRIMARY_CHARS = ["comparison", "trend", "energy", "health_status", "alerts"]
SAFETY_CRITICAL_SCENARIOS = {"alerts", "kpi"}


# ============================================================
# QueryContext — enriched features available at scoring + training
# ============================================================

@dataclass
class QueryContext:
    """
    Enriched query features for the scorer.

    Available at both scoring time (from ParsedIntent) and training time
    (from Experience.parsed_intent). Carries all the MAXIMUM_EXTRACTION.md
    signal categories that map to query-level numeric features:
      - Confidence signals (#8)
      - Domain knowledge (#27)
      - Constraint extraction / entity counts (#9)
      - Context management / time refs (#34)
      - Decision criteria / comparison flags (#26)
      - Error detection / alert flags (#10)
    """
    intent_confidence: float = 0.0
    domains: list = field(default_factory=list)
    entities: dict = field(default_factory=dict)
    query_type: str = "query"
    primary_characteristic: str = ""
    secondary_characteristics: list = field(default_factory=list)
    num_candidate_widgets: int = 0
    widget_position: int = 0

    @classmethod
    def from_parsed_intent(cls, intent) -> "QueryContext":
        """Build from a ParsedIntent object or dict."""
        if hasattr(intent, "confidence"):
            # ParsedIntent dataclass object
            return cls(
                intent_confidence=getattr(intent, "confidence", 0.0),
                domains=getattr(intent, "domains", []) or [],
                entities=getattr(intent, "entities", {}) or {},
                query_type=getattr(intent, "type", "query") or "query",
                primary_characteristic=getattr(intent, "primary_characteristic", "") or "",
                secondary_characteristics=getattr(intent, "secondary_characteristics", []) or [],
            )
        elif isinstance(intent, dict):
            # Dict form (from Experience.parsed_intent)
            return cls(
                intent_confidence=intent.get("confidence", 0.0),
                domains=intent.get("domains", []) or [],
                entities=intent.get("entities", {}) or {},
                query_type=intent.get("type", "query") or "query",
                primary_characteristic=intent.get("primary_characteristic", "") or "",
                secondary_characteristics=intent.get("secondary_characteristics", []) or [],
            )
        return cls()


class LowRankScorer(nn.Module):
    """
    Low-rank factorized scoring network with dropout.

    Uses W = A @ B factorization to keep parameter count tiny while
    still learning meaningful adjustments to widget scores.
    """

    def __init__(self, input_dim: int = INPUT_DIM, rank: int = 8,
                 hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()

        self.rank = rank
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Low-rank factorized layer: W = A @ B
        self.A = nn.Linear(input_dim, rank, bias=False)
        self.B = nn.Linear(rank, hidden_dim, bias=True)

        # Output head
        self.out = nn.Linear(hidden_dim, 1, bias=True)

        # Activation + regularization
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        # Initialize with small weights (start near zero = no adjustment)
        self._init_weights()

    def _init_weights(self):
        """Small initialization so scorer starts near-zero (no adjustment)."""
        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.normal_(self.B.weight, std=0.01)
        nn.init.zeros_(self.B.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.A(x)              # (batch, rank)
        h = self.relu(h)
        h = self.B(h)              # (batch, hidden_dim)
        h = self.relu(h)
        h = self.dropout(h)        # Regularization
        score = self.out(h)        # (batch, 1)
        return self.tanh(score)    # Clamp to [-1, 1]

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class ScorerState:
    """Tracked state of the scorer."""
    training_steps: int = 0
    total_feedback_events: int = 0
    avg_loss: float = 0.0
    avg_val_loss: float = 0.0
    best_val_loss: float = float("inf")
    steps_since_val_improve: int = 0
    lr_decays: int = 0
    recent_losses: list = field(default_factory=list)
    recent_val_losses: list = field(default_factory=list)


class ContinuousLowRankTrainer:
    """
    Continuously trains the low-rank scorer from feedback with replay buffer.

    Instead of training on every experience as it arrives (which causes
    overfitting), new experiences are added to a replay buffer. Training
    samples mini-batches from the buffer, ensuring each experience is
    seen proportionally rather than repeatedly.
    """

    def __init__(
        self,
        rank: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        checkpoint_every: int = 100,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.scorer = LowRankScorer(rank=rank).to(self.device)
        self.optimizer = optim.AdamW(
            self.scorer.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self._initial_lr = lr
        self.checkpoint_every = checkpoint_every
        self.state = ScorerState()

        # Replay buffer: stores pre-encoded (input_tensor, target) pairs
        self._replay_buffer: deque[tuple[torch.Tensor, float]] = deque(maxlen=REPLAY_BUFFER_SIZE)
        self._val_buffer: deque[tuple[torch.Tensor, float]] = deque(maxlen=int(REPLAY_BUFFER_SIZE * VAL_SPLIT))

        # Embedding model (lazy loaded)
        self._embed_model = None
        self._embed_lock = threading.Lock()

        # Scenario name -> index mapping
        self._scenario_to_idx: dict[str, int] = {}
        self._init_scenario_mapping()

        # Thread safety
        self._lock = threading.Lock()

        # Try to load existing checkpoint
        self._load_checkpoint()

        logger.info(
            f"LowRankScorer initialized: rank={rank}, input_dim={self.scorer.input_dim}, "
            f"params={self.scorer.num_parameters}, device={self.device}, "
            f"steps={self.state.training_steps}, replay_buffer={len(self._replay_buffer)}"
        )

    def _init_scenario_mapping(self):
        """Build scenario name to index mapping from widget catalog."""
        try:
            from layer2.widget_catalog import VALID_SCENARIOS
            for idx, name in enumerate(sorted(VALID_SCENARIOS)):
                self._scenario_to_idx[name] = idx
        except ImportError:
            scenarios = [
                "alerts", "category-bar", "chatstream", "comparison",
                "composition", "distribution", "edgedevicepanel",
                "eventlogstream", "flow-sankey", "kpi", "matrix-heatmap",
                "peoplehexgrid", "peoplenetwork", "peopleview",
                "supplychainglobe", "timeline", "trend",
                "trend-multi-line", "trends-cumulative",
            ]
            for idx, name in enumerate(scenarios):
                self._scenario_to_idx[name] = idx

    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get intent embedding from sentence-transformers."""
        with self._embed_lock:
            if self._embed_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    model_name = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
                    self._embed_model = SentenceTransformer(model_name, device="cpu")
                    logger.info(f"Loaded embedding model: {model_name} (forced CPU)")
                except ImportError:
                    logger.warning("sentence-transformers not available, using random embeddings")
                    return torch.randn(INTENT_EMBEDDING_DIM)

            embedding = self._embed_model.encode(text, convert_to_tensor=True)
            return embedding.cpu().float()

    def _encode_input(
        self,
        transcript: str,
        scenario: str,
        ctx: Optional[QueryContext] = None,
        num_candidates: int = 0,
        widget_position: int = 0,
    ) -> torch.Tensor:
        """
        Encode a (transcript, scenario, context) tuple into the scorer input vector.

        Feature vector layout (813 total):
          [0:768]     Intent embedding
          [768:787]   Scenario one-hot
          [787:796]   Query-level features (confidence, counts, flags)
          [796:803]   Query type one-hot
          [803:808]   Widget context (candidate count, position, safety)
          [808:813]   Primary characteristic one-hot
        """
        # Intent embedding (768-dim)
        intent_emb = self._get_embedding(transcript)

        # Scenario one-hot (19-dim)
        scenario_vec = torch.zeros(NUM_SCENARIOS)
        idx = self._scenario_to_idx.get(scenario.lower(), -1)
        if idx >= 0:
            scenario_vec[idx] = 1.0

        # Query-level scalar features (9-dim)
        query_feats = torch.zeros(QUERY_FEATURE_DIM)
        if ctx:
            devices = ctx.entities.get("devices", []) if isinstance(ctx.entities, dict) else []
            time_ref = ctx.entities.get("time", None) if isinstance(ctx.entities, dict) else None
            query_feats[0] = min(max(ctx.intent_confidence, 0.0), 1.0)
            query_feats[1] = min(len(ctx.domains) / 5.0, 1.0)
            query_feats[2] = min(len(devices) / 10.0, 1.0)
            query_feats[3] = min(len(transcript.split()) / 30.0, 1.0)
            query_feats[4] = 1.0 if devices else 0.0
            query_feats[5] = 1.0 if time_ref else 0.0
            query_feats[6] = 1.0 if ctx.primary_characteristic == "comparison" else 0.0
            query_feats[7] = 1.0 if ctx.primary_characteristic == "alerts" else 0.0
            query_feats[8] = min(len(ctx.secondary_characteristics) / 5.0, 1.0)

        # Query type one-hot (7-dim)
        type_vec = torch.zeros(QUERY_TYPE_DIM)
        if ctx and ctx.query_type in QUERY_TYPES:
            type_vec[QUERY_TYPES.index(ctx.query_type)] = 1.0

        # Widget context features (5-dim)
        widget_ctx = torch.zeros(WIDGET_CTX_DIM)
        n_cand = num_candidates or (ctx.num_candidate_widgets if ctx else 0)
        w_pos = widget_position or (ctx.widget_position if ctx else 0)
        widget_ctx[0] = min(n_cand / 24.0, 1.0)
        widget_ctx[1] = min(w_pos / 24.0, 1.0)
        widget_ctx[2] = 1.0 if scenario.lower() in SAFETY_CRITICAL_SCENARIOS else 0.0
        # [3] and [4] reserved for future: scenario_frequency, etc.

        # Primary characteristic one-hot (5-dim)
        char_vec = torch.zeros(PRIMARY_CHAR_DIM)
        if ctx and ctx.primary_characteristic in PRIMARY_CHARS:
            char_vec[PRIMARY_CHARS.index(ctx.primary_characteristic)] = 1.0

        # Scale features to balance embedding (768 dims, norm=1.0) vs structured
        # features (45 dims). Without scaling, the embedding's 768 dims overwhelm
        # the 45 discriminative dims, causing the scorer to learn global popularity
        # rather than query-specific widget preferences.
        # Scaling: embedding * 0.3 (reduce dominance), structured * 3.0 (amplify)
        intent_emb = intent_emb * 0.3
        scenario_vec = scenario_vec * 3.0
        query_feats = query_feats * 3.0
        type_vec = type_vec * 3.0
        char_vec = char_vec * 3.0

        # Concatenate: 768 + 19 + 9 + 7 + 5 + 5 = 813
        x = torch.cat([intent_emb, scenario_vec, query_feats, type_vec,
                        widget_ctx, char_vec], dim=0)
        return x.unsqueeze(0).to(self.device)

    def score_widgets(
        self,
        transcript: str,
        scenarios: list[str],
        ctx: Optional[QueryContext] = None,
    ) -> dict[str, float]:
        """
        Score a list of widget scenarios for a given transcript.

        Args:
            transcript: Raw query text
            scenarios: List of widget scenario names to score
            ctx: Optional enriched query context from ParsedIntent
        """
        self.scorer.eval()
        scores = {}
        num_candidates = len(scenarios)
        with torch.no_grad():
            for position, scenario in enumerate(scenarios):
                x = self._encode_input(
                    transcript, scenario,
                    ctx=ctx,
                    num_candidates=num_candidates,
                    widget_position=position,
                )
                adjustment = self.scorer(x).item()
                scores[scenario] = adjustment
        return scores

    def train_step(
        self,
        transcript: str,
        scenario: str,
        reward: float,
        ctx: Optional[QueryContext] = None,
        num_candidates: int = 0,
        widget_position: int = 0,
    ) -> float:
        """
        Add experience to replay buffer and train on a sampled mini-batch.

        Instead of doing SGD on this single example (which causes
        memorization), we add it to the buffer and sample a batch.
        """
        with self._lock:
            # Encode and add to buffer
            x = self._encode_input(
                transcript, scenario,
                ctx=ctx,
                num_candidates=num_candidates,
                widget_position=widget_position,
            )
            target = max(-1.0, min(1.0, reward / 2.0))

            # 80/20 train/val split
            if random.random() < VAL_SPLIT:
                self._val_buffer.append((x.detach(), target))
            else:
                self._replay_buffer.append((x.detach(), target))

            self.state.total_feedback_events += 1

            # Need minimum buffer size before training
            if len(self._replay_buffer) < REPLAY_BATCH_SIZE:
                return 0.0

            # Sample a mini-batch from replay buffer
            batch = random.sample(list(self._replay_buffer), REPLAY_BATCH_SIZE)
            batch_x = torch.cat([b[0] for b in batch], dim=0)
            batch_y = torch.tensor([[b[1]] for b in batch],
                                   device=self.device)

            # Train
            self.scorer.train()
            predicted = self.scorer(batch_x)
            loss = nn.functional.mse_loss(predicted, batch_y)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.scorer.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track training loss
            loss_val = loss.item()
            self.state.training_steps += 1
            self.state.recent_losses.append(loss_val)
            if len(self.state.recent_losses) > 100:
                self.state.recent_losses = self.state.recent_losses[-100:]
            self.state.avg_loss = sum(self.state.recent_losses) / len(self.state.recent_losses)

            # Periodic validation + LR decay
            if self.state.training_steps % 50 == 0:
                self._validate_and_maybe_decay_lr()

            # Checkpoint
            if self.state.training_steps % self.checkpoint_every == 0:
                self._save_checkpoint()

            return loss_val

    def _validate_and_maybe_decay_lr(self):
        """Compute validation loss and decay LR if plateauing."""
        if len(self._val_buffer) < 4:
            return

        self.scorer.eval()
        with torch.no_grad():
            val_samples = random.sample(list(self._val_buffer),
                                        min(len(self._val_buffer), REPLAY_BATCH_SIZE * 2))
            val_x = torch.cat([v[0] for v in val_samples], dim=0)
            val_y = torch.tensor([[v[1]] for v in val_samples], device=self.device)
            val_pred = self.scorer(val_x)
            val_loss = nn.functional.mse_loss(val_pred, val_y).item()

        self.state.recent_val_losses.append(val_loss)
        if len(self.state.recent_val_losses) > 50:
            self.state.recent_val_losses = self.state.recent_val_losses[-50:]
        self.state.avg_val_loss = sum(self.state.recent_val_losses) / len(self.state.recent_val_losses)

        # Check for improvement
        if val_loss < self.state.best_val_loss * 0.99:  # 1% improvement threshold
            self.state.best_val_loss = val_loss
            self.state.steps_since_val_improve = 0
        else:
            self.state.steps_since_val_improve += 50

        # Decay LR if plateauing
        if self.state.steps_since_val_improve >= LR_PATIENCE:
            current_lr = self.optimizer.param_groups[0]["lr"]
            new_lr = max(current_lr * 0.5, MIN_LR)
            if new_lr < current_lr:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = new_lr
                self.state.lr_decays += 1
                self.state.steps_since_val_improve = 0
                logger.info(f"Scorer LR decayed: {current_lr:.1e} -> {new_lr:.1e} "
                            f"(val_loss={val_loss:.4f}, decay #{self.state.lr_decays})")

    def train_batch(
        self,
        experiences: list[tuple[str, str, float]],
    ) -> float:
        """Train on a batch of (transcript, scenario, reward) tuples."""
        total_loss = 0.0
        for transcript, scenario, reward in experiences:
            loss = self.train_step(transcript, scenario, reward)
            total_loss += loss
        return total_loss / max(len(experiences), 1)

    def train_pairwise(
        self,
        transcript: str,
        better_scenario: str,
        worse_scenario: str,
        margin: float = 0.3,
        ctx: Optional[QueryContext] = None,
        num_candidates: int = 0,
    ) -> float:
        """
        Pairwise ranking training: teach scorer that better_scenario should
        score higher than worse_scenario for this query.

        Uses margin ranking loss which directly optimizes relative ordering
        rather than absolute targets (which lets global biases dominate).

        Args:
            transcript: Query text
            better_scenario: Widget that should rank higher
            worse_scenario: Widget that should rank lower
            margin: Minimum score gap to enforce (default 0.3)
            ctx: Query context
            num_candidates: Total candidate count

        Returns:
            Loss value (0.0 if buffer too small)
        """
        with self._lock:
            x_better = self._encode_input(
                transcript, better_scenario, ctx=ctx,
                num_candidates=num_candidates, widget_position=0,
            )
            x_worse = self._encode_input(
                transcript, worse_scenario, ctx=ctx,
                num_candidates=num_candidates, widget_position=1,
            )

            self.scorer.train()
            score_better = self.scorer(x_better).squeeze()
            score_worse = self.scorer(x_worse).squeeze()

            # MarginRankingLoss: loss = max(0, -y*(x1-x2) + margin)
            # y=1 means x1 should be ranked higher than x2
            target = torch.ones(1, device=self.device).squeeze()
            loss = nn.functional.margin_ranking_loss(
                score_better.unsqueeze(0), score_worse.unsqueeze(0),
                target.unsqueeze(0), margin=margin,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.scorer.parameters(), max_norm=1.0)
            self.optimizer.step()

            loss_val = loss.item()
            self.state.training_steps += 1
            self.state.recent_losses.append(loss_val)
            if len(self.state.recent_losses) > 100:
                self.state.recent_losses = self.state.recent_losses[-100:]
            self.state.avg_loss = sum(self.state.recent_losses) / len(self.state.recent_losses)

            return loss_val

    def _save_checkpoint(self):
        """Save scorer checkpoint to disk."""
        SCORER_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        path = SCORER_CHECKPOINT_DIR / "scorer_latest.pt"

        # Save replay buffer as list of (input_tensor, target) for recovery
        replay_data = [(x.cpu(), t) for x, t in self._replay_buffer]
        val_data = [(x.cpu(), t) for x, t in self._val_buffer]

        torch.save({
            "model_state_dict": self.scorer.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "input_dim": self.scorer.input_dim,
            "state": {
                "training_steps": self.state.training_steps,
                "total_feedback_events": self.state.total_feedback_events,
                "avg_loss": self.state.avg_loss,
                "avg_val_loss": self.state.avg_val_loss,
                "best_val_loss": self.state.best_val_loss,
                "steps_since_val_improve": self.state.steps_since_val_improve,
                "lr_decays": self.state.lr_decays,
            },
            "scenario_mapping": self._scenario_to_idx,
            "replay_buffer": replay_data,
            "val_buffer": val_data,
        }, str(path))

        logger.info(
            f"Scorer checkpoint saved: step={self.state.training_steps}, "
            f"replay={len(self._replay_buffer)}, val={len(self._val_buffer)}"
        )

    def _pad_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Pad an old-dimension tensor to the current INPUT_DIM with zeros."""
        if x.shape[-1] >= self.scorer.input_dim:
            return x
        pad_size = self.scorer.input_dim - x.shape[-1]
        padding = torch.zeros(*x.shape[:-1], pad_size, device=x.device)
        return torch.cat([x, padding], dim=-1)

    def _load_checkpoint(self):
        """Load scorer checkpoint from disk if available, with dimension migration."""
        path = SCORER_CHECKPOINT_DIR / "scorer_latest.pt"
        if not path.exists():
            return

        try:
            checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
            old_input_dim = checkpoint.get("input_dim", _OLD_INPUT_DIM)

            # Check for rank mismatch (e.g. old rank=8, new rank=32)
            old_a_shape = checkpoint["model_state_dict"].get("A.weight", torch.empty(0)).shape
            new_a_shape = self.scorer.state_dict()["A.weight"].shape
            rank_changed = len(old_a_shape) == 2 and old_a_shape[0] != new_a_shape[0]

            if old_input_dim == self.scorer.input_dim and not rank_changed:
                # Same dimension and rank — direct load
                self.scorer.load_state_dict(checkpoint["model_state_dict"], strict=False)
            elif rank_changed:
                # Rank changed — cannot warm-start, train fresh but log it
                logger.info(
                    f"Scorer rank changed: {old_a_shape[0]} -> {new_a_shape[0]}. "
                    f"Starting fresh (old weights incompatible)."
                )
                # Don't load weights — keep the fresh initialization
                # But still restore state counters and replay buffer below
            else:
                # Dimension mismatch — warm-start migration
                old_state = checkpoint["model_state_dict"]
                new_state = self.scorer.state_dict()

                for key in old_state:
                    if key not in new_state:
                        continue
                    old_shape = old_state[key].shape
                    new_shape = new_state[key].shape
                    if old_shape == new_shape:
                        new_state[key] = old_state[key]
                    elif key == "A.weight" and len(old_shape) == 2:
                        # Copy old columns into new (zero-padded) weight matrix
                        min_cols = min(old_shape[1], new_shape[1])
                        new_state[key][:, :min_cols] = old_state[key][:, :min_cols]

                self.scorer.load_state_dict(new_state)
                logger.info(
                    f"Migrated scorer checkpoint: {old_input_dim} -> "
                    f"{self.scorer.input_dim} dims (warm start)"
                )

            # Restore tracked state
            state = checkpoint.get("state", {})
            self.state.training_steps = state.get("training_steps", 0)
            self.state.total_feedback_events = state.get("total_feedback_events", 0)
            self.state.avg_loss = state.get("avg_loss", 0.0)
            self.state.avg_val_loss = state.get("avg_val_loss", 0.0)
            self.state.best_val_loss = state.get("best_val_loss", float("inf"))
            self.state.lr_decays = state.get("lr_decays", 0)

            # Restore replay buffer (pad old entries if needed)
            replay_data = checkpoint.get("replay_buffer", [])
            for x, t in replay_data:
                x = self._pad_tensor(x.to(self.device))
                self._replay_buffer.append((x, t))
            val_data = checkpoint.get("val_buffer", [])
            for x, t in val_data:
                x = self._pad_tensor(x.to(self.device))
                self._val_buffer.append((x, t))

            logger.info(
                f"Scorer checkpoint loaded: step={self.state.training_steps}, "
                f"replay={len(self._replay_buffer)}, val={len(self._val_buffer)}"
            )
        except Exception as e:
            logger.warning(f"Failed to load scorer checkpoint: {e}")

    def get_stats(self) -> dict:
        """Get scorer statistics."""
        return {
            "type": "low_rank_scorer",
            "rank": self.scorer.rank,
            "input_dim": self.scorer.input_dim,
            "parameters": self.scorer.num_parameters,
            "device": str(self.device),
            "training_steps": self.state.training_steps,
            "total_feedback_events": self.state.total_feedback_events,
            "avg_loss": round(self.state.avg_loss, 6),
            "avg_val_loss": round(self.state.avg_val_loss, 6),
            "best_val_loss": round(self.state.best_val_loss, 6) if self.state.best_val_loss < float("inf") else None,
            "lr": self.optimizer.param_groups[0]["lr"],
            "lr_decays": self.state.lr_decays,
            "replay_buffer_size": len(self._replay_buffer),
            "val_buffer_size": len(self._val_buffer),
            "recent_losses": [round(l, 6) for l in self.state.recent_losses[-10:]],
            "recent_val_losses": [round(l, 6) for l in self.state.recent_val_losses[-10:]],
        }


# ============================================================
# Singleton
# ============================================================

_scorer: Optional[ContinuousLowRankTrainer] = None
_scorer_lock = threading.Lock()


def get_scorer() -> ContinuousLowRankTrainer:
    """Get the global low-rank scorer instance."""
    global _scorer
    if _scorer is None:
        with _scorer_lock:
            if _scorer is None:
                from .config import CONTINUOUS_RL_CONFIG
                _scorer = ContinuousLowRankTrainer(
                    rank=CONTINUOUS_RL_CONFIG.get("scorer_rank", 8),
                    lr=CONTINUOUS_RL_CONFIG.get("scorer_lr", 1e-3),
                    checkpoint_every=CONTINUOUS_RL_CONFIG.get("scorer_checkpoint_every", 100),
                )
    return _scorer
