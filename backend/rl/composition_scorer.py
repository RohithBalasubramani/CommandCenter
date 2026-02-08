"""
Composition Scorer — Scores entire dashboard compositions, not just individual widgets.

Tier 1b of the continuous RL system. Complements the per-widget LowRankScorer
by learning which *combinations* of widgets form coherent, useful dashboards.

Architecture:
    Input: intent embedding (768-dim) + bag-of-scenarios (19 × 16 = 304-dim pooled)
    Encoder: Linear(768 + 304, 64) → ReLU → Dropout → Linear(64, 1) → Tanh
    Output: composition score ∈ [-1, 1]

    Total parameters: 19*16 + (1072*64 + 64) + (64*1 + 1) = 304 + 68,672 + 65 ≈ 69K

Training:
    Same replay buffer approach as LowRankScorer — mini-batch SGD with
    80/20 train/val split, LR decay on val plateau.

Key insight:
    Per-widget scorer answers "Is this widget relevant?"
    Composition scorer answers "Is this set of widgets a coherent dashboard?"
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
COMP_CHECKPOINT_DIR = _PROJECT_DIR / "rl_checkpoints" / "composition"

# Dimensions
INTENT_EMBEDDING_DIM = 768  # BGE-base-en-v1.5 output dim
NUM_SCENARIOS = 19           # Number of widget scenarios in catalog
SCENARIO_EMBED_DIM = 16      # Per-scenario embedding dimension
COMPOSITION_DIM = 64         # Hidden dimension for composition encoder

# Replay buffer config
REPLAY_BUFFER_SIZE = 2000
REPLAY_BATCH_SIZE = 8
VAL_SPLIT = 0.2
LR_PATIENCE = 200
MIN_LR = 1e-5


class CompositionScorerNet(nn.Module):
    """
    Scores an entire dashboard composition against an intent embedding.

    Creates a bag-of-scenarios representation via learned embeddings,
    concatenates with the intent embedding, and produces a single score.
    """

    def __init__(
        self,
        embed_dim: int = INTENT_EMBEDDING_DIM,
        n_scenarios: int = NUM_SCENARIOS,
        scenario_embed_dim: int = SCENARIO_EMBED_DIM,
        composition_dim: int = COMPOSITION_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_scenarios = n_scenarios
        self.scenario_embed_dim = scenario_embed_dim

        # Learnable scenario embeddings (19 × 16)
        self.scenario_embed = nn.Embedding(n_scenarios, scenario_embed_dim)

        # Composition encoder
        input_dim = embed_dim + n_scenarios * scenario_embed_dim  # 768 + 304 = 1072
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, composition_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(composition_dim, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        """Small initialization so scorer starts near-zero (no adjustment)."""
        nn.init.normal_(self.scenario_embed.weight, std=0.01)
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        intent_embedding: torch.Tensor,
        scenario_ids: torch.Tensor,
        scenario_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a dashboard composition.

        Args:
            intent_embedding: (batch, 768) intent embedding
            scenario_ids: (batch, max_widgets) scenario indices (0-18)
            scenario_mask: (batch, max_widgets) 1.0 for real widgets, 0.0 for padding

        Returns:
            (batch, 1) composition score in [-1, 1]
        """
        # Get scenario embeddings: (batch, max_widgets, 16)
        scenario_embeds = self.scenario_embed(scenario_ids)

        # Masked mean pooling: average only over real widgets
        # (batch, max_widgets, 1) mask for broadcasting
        mask_expanded = scenario_mask.unsqueeze(-1)
        pooled = (scenario_embeds * mask_expanded).sum(dim=1)  # (batch, 16)
        counts = scenario_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1)
        pooled = pooled / counts  # (batch, 16)

        # Also build a bag-of-scenarios count vector for explicit composition info
        # One-hot accumulate: (batch, n_scenarios * scenario_embed_dim)
        # Use the full flattened embedding space
        batch_size = intent_embedding.shape[0]
        bag_vec = torch.zeros(batch_size, self.n_scenarios * self.scenario_embed_dim,
                              device=intent_embedding.device)

        # Scatter pooled embeddings into scenario-specific slots
        for b in range(batch_size):
            n_widgets = int(scenario_mask[b].sum().item())
            for w in range(n_widgets):
                sid = scenario_ids[b, w].item()
                start = sid * self.scenario_embed_dim
                end = start + self.scenario_embed_dim
                bag_vec[b, start:end] += scenario_embeds[b, w]
            # Normalize by count
            if n_widgets > 0:
                bag_vec[b] /= n_widgets

        # Concatenate intent + bag-of-scenarios: (batch, 768 + 304)
        combined = torch.cat([intent_embedding, bag_vec], dim=1)

        return self.encoder(combined)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class CompositionScorerState:
    """Tracked state of the composition scorer."""
    training_steps: int = 0
    total_feedback_events: int = 0
    avg_loss: float = 0.0
    avg_val_loss: float = 0.0
    best_val_loss: float = float("inf")
    steps_since_val_improve: int = 0
    lr_decays: int = 0
    recent_losses: list = field(default_factory=list)
    recent_val_losses: list = field(default_factory=list)


class ContinuousCompositionTrainer:
    """
    Continuously trains the composition scorer from feedback with replay buffer.

    Follows the same replay + mini-batch approach as ContinuousLowRankTrainer
    but operates on dashboard compositions rather than individual widgets.
    """

    MAX_WIDGETS = 12  # Maximum widgets in a dashboard (for padding)

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        checkpoint_every: int = 100,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.scorer = CompositionScorerNet().to(self.device)
        self.optimizer = optim.AdamW(
            self.scorer.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self._initial_lr = lr
        self.checkpoint_every = checkpoint_every
        self.state = CompositionScorerState()

        # Replay buffer: stores (intent_emb, scenario_ids, scenario_mask, target)
        self._replay_buffer: deque = deque(maxlen=REPLAY_BUFFER_SIZE)
        self._val_buffer: deque = deque(maxlen=int(REPLAY_BUFFER_SIZE * VAL_SPLIT))

        # Embedding model (lazy loaded, shared with per-widget scorer)
        self._embed_model = None
        self._embed_lock = threading.Lock()

        # Scenario name → index mapping
        self._scenario_to_idx: dict[str, int] = {}
        self._init_scenario_mapping()

        # Thread safety
        self._lock = threading.Lock()

        # Try to load existing checkpoint
        self._load_checkpoint()

        logger.info(
            f"CompositionScorer initialized: params={self.scorer.num_parameters}, "
            f"device={self.device}, steps={self.state.training_steps}, "
            f"replay_buffer={len(self._replay_buffer)}"
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

    def _encode_composition(
        self,
        transcript: str,
        scenarios: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a (transcript, [scenarios]) pair into scorer input tensors.

        Returns: (intent_emb, scenario_ids, scenario_mask) each with batch dim.
        """
        intent_emb = self._get_embedding(transcript)  # (768,)

        # Build scenario IDs and mask (padded to MAX_WIDGETS)
        scenario_ids = torch.zeros(self.MAX_WIDGETS, dtype=torch.long)
        scenario_mask = torch.zeros(self.MAX_WIDGETS, dtype=torch.float32)

        for i, scenario in enumerate(scenarios[:self.MAX_WIDGETS]):
            idx = self._scenario_to_idx.get(scenario.lower(), 0)
            scenario_ids[i] = idx
            scenario_mask[i] = 1.0

        return (
            intent_emb.unsqueeze(0).to(self.device),      # (1, 768)
            scenario_ids.unsqueeze(0).to(self.device),     # (1, MAX_WIDGETS)
            scenario_mask.unsqueeze(0).to(self.device),    # (1, MAX_WIDGETS)
        )

    def score_composition(
        self,
        transcript: str,
        scenarios: list[str],
    ) -> float:
        """
        Score a dashboard composition for a given transcript.

        Returns a score in [-1, 1] indicating how well the set of
        scenarios forms a coherent dashboard for this query.
        """
        if not scenarios:
            return 0.0

        self.scorer.eval()
        with torch.no_grad():
            intent_emb, scenario_ids, scenario_mask = self._encode_composition(
                transcript, scenarios
            )
            score = self.scorer(intent_emb, scenario_ids, scenario_mask)
            return score.item()

    def train_step(
        self,
        transcript: str,
        scenarios: list[str],
        reward: float,
    ) -> float:
        """
        Add composition experience to replay buffer and train on a mini-batch.
        """
        if not scenarios:
            return 0.0

        with self._lock:
            # Encode and add to buffer
            intent_emb, scenario_ids, scenario_mask = self._encode_composition(
                transcript, scenarios
            )
            target = max(-1.0, min(1.0, reward / 2.0))

            experience = (
                intent_emb.detach(),
                scenario_ids.detach(),
                scenario_mask.detach(),
                target,
            )

            # 80/20 train/val split
            if random.random() < VAL_SPLIT:
                self._val_buffer.append(experience)
            else:
                self._replay_buffer.append(experience)

            self.state.total_feedback_events += 1

            # Need minimum buffer size before training
            if len(self._replay_buffer) < REPLAY_BATCH_SIZE:
                return 0.0

            # Sample a mini-batch from replay buffer
            batch = random.sample(list(self._replay_buffer), REPLAY_BATCH_SIZE)
            batch_intent = torch.cat([b[0] for b in batch], dim=0)
            batch_ids = torch.cat([b[1] for b in batch], dim=0)
            batch_mask = torch.cat([b[2] for b in batch], dim=0)
            batch_target = torch.tensor(
                [[b[3]] for b in batch], device=self.device
            )

            # Train
            self.scorer.train()
            predicted = self.scorer(batch_intent, batch_ids, batch_mask)
            loss = nn.functional.mse_loss(predicted, batch_target)

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
            self.state.avg_loss = (
                sum(self.state.recent_losses) / len(self.state.recent_losses)
            )

            # Periodic validation + LR decay
            if self.state.training_steps % 50 == 0:
                self._validate_and_maybe_decay_lr()

            # Checkpoint
            if self.state.training_steps % self.checkpoint_every == 0:
                self._save_checkpoint()

            return loss_val

    def train_batch(
        self,
        experiences: list[tuple[str, list[str], float]],
    ) -> float:
        """Train on a batch of (transcript, scenarios, reward) tuples."""
        total_loss = 0.0
        for transcript, scenarios, reward in experiences:
            loss = self.train_step(transcript, scenarios, reward)
            total_loss += loss
        return total_loss / max(len(experiences), 1)

    def _validate_and_maybe_decay_lr(self):
        """Compute validation loss and decay LR if plateauing."""
        if len(self._val_buffer) < 4:
            return

        self.scorer.eval()
        with torch.no_grad():
            val_samples = random.sample(
                list(self._val_buffer),
                min(len(self._val_buffer), REPLAY_BATCH_SIZE * 2),
            )
            val_intent = torch.cat([v[0] for v in val_samples], dim=0)
            val_ids = torch.cat([v[1] for v in val_samples], dim=0)
            val_mask = torch.cat([v[2] for v in val_samples], dim=0)
            val_target = torch.tensor(
                [[v[3]] for v in val_samples], device=self.device
            )
            val_pred = self.scorer(val_intent, val_ids, val_mask)
            val_loss = nn.functional.mse_loss(val_pred, val_target).item()

        self.state.recent_val_losses.append(val_loss)
        if len(self.state.recent_val_losses) > 50:
            self.state.recent_val_losses = self.state.recent_val_losses[-50:]
        self.state.avg_val_loss = (
            sum(self.state.recent_val_losses) / len(self.state.recent_val_losses)
        )

        # Check for improvement
        if val_loss < self.state.best_val_loss * 0.99:
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
                logger.info(
                    f"CompositionScorer LR decayed: {current_lr:.1e} → {new_lr:.1e} "
                    f"(val_loss={val_loss:.4f}, decay #{self.state.lr_decays})"
                )

    def _save_checkpoint(self):
        """Save composition scorer checkpoint to disk."""
        COMP_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        path = COMP_CHECKPOINT_DIR / "composition_latest.pt"

        replay_data = [
            (e[0].cpu(), e[1].cpu(), e[2].cpu(), e[3])
            for e in self._replay_buffer
        ]
        val_data = [
            (v[0].cpu(), v[1].cpu(), v[2].cpu(), v[3])
            for v in self._val_buffer
        ]

        torch.save({
            "model_state_dict": self.scorer.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
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
            f"CompositionScorer checkpoint saved: step={self.state.training_steps}, "
            f"replay={len(self._replay_buffer)}, val={len(self._val_buffer)}"
        )

    def _load_checkpoint(self):
        """Load composition scorer checkpoint from disk if available."""
        path = COMP_CHECKPOINT_DIR / "composition_latest.pt"
        if not path.exists():
            return

        try:
            checkpoint = torch.load(
                str(path), map_location=self.device, weights_only=False
            )
            self.scorer.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )

            state = checkpoint.get("state", {})
            self.state.training_steps = state.get("training_steps", 0)
            self.state.total_feedback_events = state.get("total_feedback_events", 0)
            self.state.avg_loss = state.get("avg_loss", 0.0)
            self.state.avg_val_loss = state.get("avg_val_loss", 0.0)
            self.state.best_val_loss = state.get("best_val_loss", float("inf"))
            self.state.lr_decays = state.get("lr_decays", 0)

            replay_data = checkpoint.get("replay_buffer", [])
            for e in replay_data:
                self._replay_buffer.append(
                    (e[0].to(self.device), e[1].to(self.device),
                     e[2].to(self.device), e[3])
                )
            val_data = checkpoint.get("val_buffer", [])
            for v in val_data:
                self._val_buffer.append(
                    (v[0].to(self.device), v[1].to(self.device),
                     v[2].to(self.device), v[3])
                )

            logger.info(
                f"CompositionScorer checkpoint loaded: step={self.state.training_steps}, "
                f"replay={len(self._replay_buffer)}, val={len(self._val_buffer)}"
            )
        except Exception as e:
            logger.warning(f"Failed to load composition scorer checkpoint: {e}")

    def get_stats(self) -> dict:
        """Get composition scorer statistics."""
        return {
            "type": "composition_scorer",
            "parameters": self.scorer.num_parameters,
            "device": str(self.device),
            "training_steps": self.state.training_steps,
            "total_feedback_events": self.state.total_feedback_events,
            "avg_loss": round(self.state.avg_loss, 6),
            "avg_val_loss": round(self.state.avg_val_loss, 6),
            "best_val_loss": (
                round(self.state.best_val_loss, 6)
                if self.state.best_val_loss < float("inf") else None
            ),
            "lr": self.optimizer.param_groups[0]["lr"],
            "lr_decays": self.state.lr_decays,
            "replay_buffer_size": len(self._replay_buffer),
            "val_buffer_size": len(self._val_buffer),
        }
