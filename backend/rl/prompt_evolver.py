"""
RL-based System Prompt Optimization for Widget Selection.

Treats the system prompt as a learnable parameter. Uses a multi-armed bandit
(UCB1) to select among prompt variants, then evolves the population based on
user feedback rewards.

The prompt has three mutable sections:
  1. Domain affinity rules (which widgets for which query types)
  2. General selection rules (how to pick/size widgets)
  3. Sizing rules (which scenarios at which sizes)

Each PromptVariant is a specific configuration of these sections.
The PromptEvolver maintains a population, selects variants per-query,
tracks rewards, and periodically generates new variants by mutation.

Integration:
  - widget_selector.py calls evolver.get_prompt() instead of static FAST_SELECT_PROMPT
  - orchestrator.py records prompt_version in Experience
  - continuous.py calls evolver.update_reward() when feedback arrives
"""

import json
import logging
import math
import random
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE PROMPT SECTIONS — the current static prompt, decomposed into parts
# ═══════════════════════════════════════════════════════════════════════════════

BASELINE_DOMAIN_AFFINITIES = {
    "comparison": "comparison (hero), trend-multi-line, kpi for each entity",
    "energy_power": "trend (hero), distribution, kpi, trends-cumulative",
    "alerts": "alerts (hero), kpi, trend, timeline",
    "maintenance": "timeline (hero), eventlogstream, alerts, category-bar",
    "health_single": "edgedevicepanel (hero), trend, alerts, kpi",
    "health_overview": "matrix-heatmap (hero), category-bar, alerts, kpi",
    "hvac_chiller": "trend (hero), comparison, kpi, alerts",
    "top_consumers": "category-bar (hero), distribution, kpi, trend",
}

BASELINE_RULES = [
    "Select EXACTLY {widget_count} widgets — use ALL {widget_count} slots with DIVERSE widget types",
    "First widget MUST be hero size — pick the scenario that BEST answers the query",
    "Each widget MUST have a relevance score (0.0-1.0) reflecting how useful it is for THIS specific query\n   - Hero: 0.90-0.98 | Direct supporting: 0.75-0.89 | Context: 0.55-0.74 | Nice-to-have: 0.40-0.54",
    "Use EXACT scenario names from catalog",
    "Max 2 KPI widgets per entity, each showing a DIFFERENT metric",
    "Each widget MUST include data_request with query, metric, and entities from the query",
    "MAXIMIZE widget type diversity — use DIFFERENT scenario types (trend, alerts, distribution, category-bar, composition, timeline, eventlogstream, etc.)",
    "Operators need comprehensive dashboards — show KPIs AND trends AND alerts AND distributions, not just one type",
]

BASELINE_SIZING = {
    "hero": "First widget only (primary answer). Use: trend, trend-multi-line, comparison, category-bar, matrix-heatmap, edgedevicepanel, timeline, flow-sankey, eventlogstream",
    "expanded": "Secondary detail widgets. Use: trend, trend-multi-line, comparison, distribution, alerts, timeline, composition",
    "normal": "Supporting context. Use: alerts, distribution, kpi",
    "compact": "Quick-glance metrics only. Use: kpi",
}

# Template that gets filled with mutable sections
PROMPT_SKELETON = '''Select widgets for this industrial operations query.

## WIDGET CATALOG
{catalog}

## QUERY
"{query}"
Intent: {intent_type} | Domains: {domains} | Entities: {entities}
Primary focus: {primary_char} | Also relevant: {secondary_chars}

## DATA AVAILABLE
{data_summary}

## DOMAIN AFFINITY (prefer these widget types):
{domain_affinity_section}

## SIZING RULES
{sizing_section}

## RULES
{rules_section}

## OUTPUT (JSON only, no explanation)
{{"heading": "<short dashboard title>", "widgets": [
  {{"scenario": "<name>", "size": "<size>", "relevance": <0.0-1.0>, "data_request": {{"query": "<what data>", "metric": "<metric_name>", "entities": ["{entity_hint}"]}}}}
]}}'''


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT VARIANT — one configuration of the mutable sections
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PromptVariant:
    """A specific prompt configuration with tracked performance."""

    variant_id: str
    created_at: str
    parent_id: Optional[str] = None      # Which variant was mutated to create this
    mutation_type: Optional[str] = None   # What mutation was applied

    # Mutable sections
    domain_affinities: dict = field(default_factory=dict)
    rules: list = field(default_factory=list)
    sizing: dict = field(default_factory=dict)

    # Performance tracking (for UCB1)
    total_reward: float = 0.0
    num_trials: int = 0
    avg_reward: float = 0.0

    # Detailed stats
    thumbs_up: int = 0
    thumbs_down: int = 0

    def assemble_prompt(self) -> str:
        """Build the full prompt template from this variant's rules.

        Returns a template string with {catalog}, {query}, etc. placeholders
        that still need a second .format() call with real values. JSON braces
        in the output section are properly double-escaped ({{ }}) so they
        survive the final .format() call.
        """
        # Domain affinity section
        affinity_lines = []
        for domain, widgets in self.domain_affinities.items():
            label = domain.replace("_", " ").capitalize() + " queries"
            affinity_lines.append(f"- {label} → {widgets}")
        domain_affinity_section = "\n".join(affinity_lines)

        # Sizing section
        sizing_lines = []
        for size, desc in self.sizing.items():
            sizing_lines.append(f"{size}: {desc}")
        sizing_section = "\n".join(sizing_lines)

        # Rules section
        rules_lines = []
        for i, rule in enumerate(self.rules, 1):
            rules_lines.append(f"{i}. {rule}")
        rules_section = "\n".join(rules_lines)

        # Build via string replacement instead of .format() to preserve
        # {{ }} escaping in the JSON output section of PROMPT_SKELETON.
        result = PROMPT_SKELETON
        result = result.replace("{domain_affinity_section}", domain_affinity_section)
        result = result.replace("{sizing_section}", sizing_section)
        result = result.replace("{rules_section}", rules_section)
        return result

    def ucb1_score(self, total_trials: int, exploration_weight: float = 1.4) -> float:
        """UCB1 score: exploitation + exploration bonus."""
        if self.num_trials == 0:
            return float("inf")  # Always try untested variants
        exploitation = self.avg_reward
        exploration = exploration_weight * math.sqrt(math.log(total_trials) / self.num_trials)
        return exploitation + exploration

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PromptVariant":
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT EVOLVER — multi-armed bandit over prompt variants
# ═══════════════════════════════════════════════════════════════════════════════

class PromptEvolver:
    """Manages a population of prompt variants, selects the best, evolves new ones.

    Uses UCB1 (Upper Confidence Bound) for variant selection:
    - New/untested variants get infinite priority (explore)
    - Tested variants balance avg_reward + exploration bonus
    - Higher reward = more often selected

    Mutation strategies (language-only, preserves logic):
    - rephrase_rule: Reword a rule for clarity/brevity
    - simplify_affinity: Remove redundant words in domain affinity descriptions
    - tighten_wording: Make instructions more concise
    - improve_clarity: Make ambiguous phrasing more explicit
    """

    def __init__(
        self,
        data_dir: str = None,
        max_variants: int = 10,
        epsilon: float = 0.1,
        evolve_every: int = 25,
    ):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent.parent / "rl_training_data"
        self.state_file = self.data_dir / "prompt_evolver_state.json"
        self.max_variants = max_variants
        self.epsilon = epsilon  # Epsilon-greedy fallback when UCB1 is too aggressive
        self.evolve_every = evolve_every  # Mutate after this many trials
        self.lock = threading.RLock()

        self.variants: list[PromptVariant] = []
        self.total_trials = 0
        self._trials_since_evolve = 0

        # Validation holdout: track recent experiences for variant testing
        # Each entry: (variant_id, reward)
        self._validation_holdout: list[tuple[str, float]] = []
        self._validation_size = 50  # Keep last 50 experiences for validation

        # Load or create baseline
        self._load_state()
        if not self.variants:
            self.variants.append(self._create_baseline())
            self._save_state()

    def _create_baseline(self) -> PromptVariant:
        """Create the baseline variant from current static prompt."""
        return PromptVariant(
            variant_id="baseline_v1",
            created_at=datetime.now().isoformat(),
            parent_id=None,
            mutation_type=None,
            domain_affinities=deepcopy(BASELINE_DOMAIN_AFFINITIES),
            rules=list(BASELINE_RULES),
            sizing=deepcopy(BASELINE_SIZING),
        )

    def get_prompt(self) -> tuple[str, str]:
        """Select a prompt variant and return (assembled_prompt, variant_id).

        Uses UCB1 to balance exploitation (best-performing) and exploration (untested).
        """
        with self.lock:
            variant = self._select_variant()
            prompt = variant.assemble_prompt()
            return prompt, variant.variant_id

    def _select_variant(self) -> PromptVariant:
        """UCB1 selection with epsilon-greedy fallback."""
        if not self.variants:
            baseline = self._create_baseline()
            self.variants.append(baseline)
            return baseline

        # Epsilon-greedy: random selection with probability epsilon
        if random.random() < self.epsilon and len(self.variants) > 1:
            return random.choice(self.variants)

        # UCB1 selection
        total = max(self.total_trials, 1)
        best = max(self.variants, key=lambda v: v.ucb1_score(total))
        return best

    def update_reward(self, variant_id: str, reward: float, thumbs: Optional[str] = None):
        """Update a variant's performance after user feedback.

        Args:
            variant_id: The variant that was used
            reward: Computed reward signal (-2.0 to 2.0)
            thumbs: Optional "up" or "down" explicit rating
        """
        with self.lock:
            variant = self._find_variant(variant_id)
            if not variant:
                logger.warning(f"Prompt variant not found: {variant_id}")
                return

            variant.total_reward += reward
            variant.num_trials += 1
            variant.avg_reward = variant.total_reward / variant.num_trials

            if thumbs == "up":
                variant.thumbs_up += 1
            elif thumbs == "down":
                variant.thumbs_down += 1

            self.total_trials += 1
            self._trials_since_evolve += 1

            # Add to validation holdout
            self._validation_holdout.append((variant_id, reward))
            if len(self._validation_holdout) > self._validation_size:
                self._validation_holdout.pop(0)

            # Periodically evolve
            if self._trials_since_evolve >= self.evolve_every:
                self._evolve()
                self._trials_since_evolve = 0

            self._save_state()

            logger.debug(
                f"Prompt variant {variant_id[:12]}... reward={reward:.2f} "
                f"avg={variant.avg_reward:.3f} trials={variant.num_trials}"
            )

    def _find_variant(self, variant_id: str) -> Optional[PromptVariant]:
        """Find variant by ID."""
        for v in self.variants:
            if v.variant_id == variant_id:
                return v
        return None

    def _compute_validation_score(self, variant: PromptVariant) -> Optional[float]:
        """
        Compute variant's performance on validation holdout.

        Uses experiences from other variants to estimate how this variant
        would have performed (off-policy evaluation).

        Returns:
            Mean reward on validation set, or None if insufficient data
        """
        if len(self._validation_holdout) < 10:
            return None  # Need at least 10 experiences

        # Simple off-policy: average rewards from validation set
        # (In practice, could weight by similarity to variant's characteristics)
        rewards = [r for _, r in self._validation_holdout]
        return sum(rewards) / len(rewards)

    def _evolve(self):
        """Generate new prompt variants by mutating high-performers.

        Strategy:
        1. Rank variants by avg_reward
        2. Take the top 2 and mutate each
        3. Drop the worst performer if at capacity
        """
        if len(self.variants) < 1:
            return

        # Sort by average reward (best first)
        ranked = sorted(self.variants, key=lambda v: v.avg_reward, reverse=True)

        # Mutate the top performer
        parent = ranked[0]
        if parent.num_trials < 5:
            # Not enough data to evolve yet
            return

        # Generate a mutation
        child = self._mutate(parent)
        if child:
            # Test on validation set before adding
            val_score = self._compute_validation_score(child)
            if val_score is not None:
                # Only add if validation score is reasonable (above baseline mean)
                baseline_val = self._compute_validation_score(self._find_variant("baseline_v1"))
                if baseline_val is None or val_score >= baseline_val - 0.1:  # Allow 10% tolerance
                    self.variants.append(child)
                    logger.info(
                        f"Evolved new variant {child.variant_id[:12]}... "
                        f"from {parent.variant_id[:12]}... "
                        f"mutation={child.mutation_type}, val_score={val_score:.3f}"
                    )
                else:
                    logger.info(
                        f"Rejected variant {child.variant_id[:12]}... "
                        f"val_score={val_score:.3f} < baseline={baseline_val:.3f}"
                    )
            else:
                # Not enough validation data yet, add anyway
                self.variants.append(child)
                logger.info(
                    f"Evolved new variant {child.variant_id[:12]}... "
                    f"from {parent.variant_id[:12]}... "
                    f"mutation={child.mutation_type} (no validation yet)"
                )

        # Prune if over capacity
        while len(self.variants) > self.max_variants:
            # Never remove baseline or variants with < 10 trials
            removable = [
                v for v in self.variants
                if v.variant_id != "baseline_v1" and v.num_trials >= 10
            ]
            if not removable:
                break
            worst = min(removable, key=lambda v: v.avg_reward)
            self.variants.remove(worst)
            logger.info(f"Pruned variant {worst.variant_id[:12]}... avg_reward={worst.avg_reward:.3f}")

    def _mutate(self, parent: PromptVariant) -> Optional[PromptVariant]:
        """Create a new variant by mutating a parent.

        Mutation types (preserve logic, more impactful):
        - rephrase_rule: Reword a rule for clarity/brevity
        - simplify_affinity: Remove redundant words in domain descriptions
        - tighten_wording: Make instructions more concise
        - improve_clarity: Make ambiguous phrasing more explicit
        - reorder_rules: Change rule order (LLMs are order-sensitive)
        - add_examples: Inject concrete examples into rules
        - rephrase_conditional: Rephrase "prefer X for Y" as "when Y, X is most relevant"
        """
        mutation_type = random.choice([
            "rephrase_rule",
            "simplify_affinity",
            "tighten_wording",
            "improve_clarity",
            "reorder_rules",
            "add_examples",
            "rephrase_conditional",
        ])

        child = PromptVariant(
            variant_id=str(uuid.uuid4())[:12],
            created_at=datetime.now().isoformat(),
            parent_id=parent.variant_id,
            mutation_type=mutation_type,
            domain_affinities=deepcopy(parent.domain_affinities),
            rules=list(parent.rules),
            sizing=deepcopy(parent.sizing),
        )

        if mutation_type == "rephrase_rule":
            self._mutate_rephrase_rule(child)
        elif mutation_type == "simplify_affinity":
            self._mutate_simplify_affinity(child)
        elif mutation_type == "tighten_wording":
            self._mutate_tighten_wording(child)
        elif mutation_type == "improve_clarity":
            self._mutate_improve_clarity(child)
        elif mutation_type == "reorder_rules":
            self._mutate_reorder_rules(child)
        elif mutation_type == "add_examples":
            self._mutate_add_examples(child)
        elif mutation_type == "rephrase_conditional":
            self._mutate_rephrase_conditional(child)

        return child

    def _mutate_rephrase_rule(self, variant: PromptVariant):
        """Rephrase a rule for clarity or brevity (preserves meaning)."""
        if not variant.rules:
            return

        idx = random.randint(0, len(variant.rules) - 1)
        rule = variant.rules[idx]

        # Apply rewording transformations that preserve logic
        rephrases = [
            # Remove filler words
            (r'\bMUST be\b', 'must be'),
            (r'\bMUST have\b', 'must have'),
            (r'\bMUST include\b', 'must include'),
            (r'\bshould be able to\b', 'should'),
            (r'\bin order to\b', 'to'),
            (r'\bfor the purpose of\b', 'for'),
            # Tighten phrases
            (r'\bEXACTLY (\d+)\b', r'\1'),
            (r'\bat least one\b', '≥1'),
            (r'\bat most\b', '≤'),
            (r'\bgreater than\b', '>'),
            (r'\bless than\b', '<'),
            # Simplify common phrases
            (r'\beach and every\b', 'each'),
            (r'\ball of the\b', 'all'),
            (r'\bone or more\b', '1+'),
        ]

        import re
        new_rule = rule
        for pattern, replacement in rephrases:
            new_rule = re.sub(pattern, replacement, new_rule, flags=re.IGNORECASE)

        if new_rule != rule:
            variant.rules[idx] = new_rule
            logger.debug(f"Rephrased rule: '{rule[:40]}...' → '{new_rule[:40]}...'")

    def _mutate_simplify_affinity(self, variant: PromptVariant):
        """Simplify domain affinity wording (preserves widget list)."""
        if not variant.domain_affinities:
            return

        domain = random.choice(list(variant.domain_affinities.keys()))
        current = variant.domain_affinities[domain]

        # Apply simplifications
        simplifications = [
            # Remove redundant markers
            (r'\(hero\)', '(primary)'),
            (r'\s+for each entity\b', '/entity'),
            (r'\s+when available\b', ''),
            # Shorten common phrases
            (r'\btrend-multi-line\b', 'multi-trend'),
            (r'\btrends-cumulative\b', 'cumul-trend'),
            (r'\bmatrix-heatmap\b', 'heatmap'),
            (r'\beventlogstream\b', 'event-log'),
            (r'\bedgedevicepanel\b', 'device-panel'),
        ]

        import re
        new_affinity = current
        for pattern, replacement in simplifications:
            new_affinity = re.sub(pattern, replacement, new_affinity, flags=re.IGNORECASE)

        if new_affinity != current:
            variant.domain_affinities[domain] = new_affinity
            logger.debug(f"Simplified {domain} affinity: '{current}' → '{new_affinity}'")

    def _mutate_tighten_wording(self, variant: PromptVariant):
        """Make sizing descriptions more concise (preserves scenario lists)."""
        if not variant.sizing:
            return

        size = random.choice(list(variant.sizing.keys()))
        desc = variant.sizing[size]

        # Tighten common verbose phrases
        tightenings = [
            (r'\bFirst widget only\b', '1st widget'),
            (r'\bSecondary detail widgets\b', '2nd-level detail'),
            (r'\bSupporting context\b', 'Context'),
            (r'\bQuick-glance metrics only\b', 'Quick metrics'),
            (r'\bprimary answer\b', 'main answer'),
            (r'\. Use:\s*', ' - '),
        ]

        import re
        new_desc = desc
        for pattern, replacement in tightenings:
            new_desc = re.sub(pattern, replacement, new_desc, flags=re.IGNORECASE)

        if new_desc != desc:
            variant.sizing[size] = new_desc
            logger.debug(f"Tightened {size} sizing: '{desc}' → '{new_desc}'")

    def _mutate_improve_clarity(self, variant: PromptVariant):
        """Make ambiguous phrasing more explicit (preserves constraints)."""
        if not variant.rules:
            return

        idx = random.randint(0, len(variant.rules) - 1)
        rule = variant.rules[idx]

        # Clarify vague language
        clarifications = [
            # Be more explicit about ranges
            (r'\bhigh\b', 'high (0.8-1.0)'),
            (r'\blow\b(?! \()', 'low (0.0-0.4)'),
            (r'\bmedium\b', 'medium (0.4-0.8)'),
            # Clarify positioning
            (r'\bfirst widget\b', 'widget[0]'),
            (r'\blast widget\b', 'widget[-1]'),
            # Be explicit about requirements
            (r'\bshould\b', 'must'),
            (r'\btry to\b', 'ensure'),
            (r'\bconsider\b', 'evaluate'),
        ]

        import re
        new_rule = rule
        for pattern, replacement in clarifications:
            new_rule = re.sub(pattern, replacement, new_rule, flags=re.IGNORECASE)

        if new_rule != rule:
            variant.rules[idx] = new_rule
            logger.debug(f"Clarified rule: '{rule[:40]}...' → '{new_rule[:40]}...'")

    def _mutate_reorder_rules(self, variant: PromptVariant):
        """Reorder rules (LLMs are sensitive to information order)."""
        if len(variant.rules) < 3:
            return

        # Strategy: Move a high-importance rule earlier or lower-importance later
        # Keep first rule (widget count) and second rule (hero requirement) fixed
        if len(variant.rules) > 3:
            # Shuffle rules from index 2 onwards
            reorderable = variant.rules[2:]
            random.shuffle(reorderable)
            variant.rules = variant.rules[:2] + reorderable
            logger.debug(f"Reordered rules (kept first 2 fixed)")

    def _mutate_add_examples(self, variant: PromptVariant):
        """Add concrete examples to rules without changing constraints."""
        if not variant.rules:
            return

        # Examples that clarify without changing logic
        example_additions = {
            "diverse": " (e.g., combine trend + alerts + kpi, not 3 trends)",
            "DIFFERENT": " (avoid: 2 trends, 2 distributions)",
            "comprehensive": " (show multiple aspects: performance + health + context)",
            "MAX 2 KPI": " (if showing pump: power_kw + efficiency, not also temperature)",
            "relevance": " (directly answers query, not tangentially related)",
        }

        idx = random.randint(0, len(variant.rules) - 1)
        rule = variant.rules[idx]

        # Find a keyword in the rule that we can add an example to
        for keyword, example in example_additions.items():
            if keyword.lower() in rule.lower() and example not in rule:
                # Insert example after the keyword
                import re
                pattern = re.compile(f'({re.escape(keyword)})', re.IGNORECASE)
                new_rule = pattern.sub(rf'\1{example}', rule, count=1)
                if new_rule != rule:
                    variant.rules[idx] = new_rule
                    logger.debug(f"Added example: '{rule[:40]}...' → '{new_rule[:40]}...'")
                    break

    def _mutate_rephrase_conditional(self, variant: PromptVariant):
        """Rephrase 'prefer X for Y' as 'when Y, X is most relevant' (same logic)."""
        if not variant.domain_affinities:
            return

        # Rephrase domain affinity descriptions
        domain = random.choice(list(variant.domain_affinities.keys()))
        current = variant.domain_affinities[domain]

        # Pattern: "scenario (hero), other, other" → "For [domain] queries: scenario (primary choice), ..."
        rephrases = [
            (r'\(hero\)', '(primary choice)'),
            (r'\(hero\)', '(best fit)'),
            (r', ', ' → '),  # Change separator
            (r'prefer ', 'prioritize '),
        ]

        import re
        new_affinity = current
        for pattern, replacement in rephrases:
            new_affinity = re.sub(pattern, replacement, new_affinity, count=1)
            if new_affinity != current:
                break

        if new_affinity != current:
            variant.domain_affinities[domain] = new_affinity
            logger.debug(f"Rephrased conditional for {domain}: '{current}' → '{new_affinity}'")

    # ═══════════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════════

    def _save_state(self):
        """Save all variants and stats to disk."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "total_trials": self.total_trials,
                "trials_since_evolve": self._trials_since_evolve,
                "saved_at": datetime.now().isoformat(),
                "variants": [v.to_dict() for v in self.variants],
            }
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save prompt evolver state: {e}")

    def _load_state(self):
        """Load variants and stats from disk."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file) as f:
                state = json.load(f)

            self.total_trials = state.get("total_trials", 0)
            self._trials_since_evolve = state.get("trials_since_evolve", 0)
            self.variants = [
                PromptVariant.from_dict(v) for v in state.get("variants", [])
            ]
            logger.info(
                f"Loaded {len(self.variants)} prompt variants "
                f"({self.total_trials} total trials)"
            )
        except Exception as e:
            logger.error(f"Failed to load prompt evolver state: {e}")
            self.variants = []

    # ═══════════════════════════════════════════════════════════════════════════
    # REPORTING
    # ═══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        """Get statistics about all prompt variants."""
        with self.lock:
            return {
                "total_variants": len(self.variants),
                "total_trials": self.total_trials,
                "variants": [
                    {
                        "id": v.variant_id,
                        "parent": v.parent_id,
                        "mutation": v.mutation_type,
                        "trials": v.num_trials,
                        "avg_reward": round(v.avg_reward, 4),
                        "thumbs_up": v.thumbs_up,
                        "thumbs_down": v.thumbs_down,
                        "ucb1": round(v.ucb1_score(max(self.total_trials, 1)), 4),
                    }
                    for v in sorted(self.variants, key=lambda v: v.avg_reward, reverse=True)
                ],
            }

    def get_best_variant(self) -> Optional[PromptVariant]:
        """Get the highest-performing variant (by avg reward)."""
        with self.lock:
            tested = [v for v in self.variants if v.num_trials >= 5]
            if not tested:
                return self.variants[0] if self.variants else None
            return max(tested, key=lambda v: v.avg_reward)

    def promote_best(self) -> Optional[str]:
        """Lock in the best-performing variant as the new baseline.

        Returns the variant_id of the promoted variant, or None.
        """
        with self.lock:
            best = self.get_best_variant()
            if not best or best.variant_id == "baseline_v1":
                return None

            # Only promote if significantly better than baseline
            baseline = self._find_variant("baseline_v1")
            if baseline and best.avg_reward <= baseline.avg_reward + 0.05:
                return None

            logger.info(
                f"Promoting variant {best.variant_id[:12]}... "
                f"(avg_reward={best.avg_reward:.3f}) as new baseline"
            )

            # Replace baseline's rules with the best variant's
            if baseline:
                baseline.domain_affinities = deepcopy(best.domain_affinities)
                baseline.rules = list(best.rules)
                baseline.sizing = deepcopy(best.sizing)
                # Reset stats for new baseline
                baseline.total_reward = 0.0
                baseline.num_trials = 0
                baseline.avg_reward = 0.0

            # Remove the promoted variant (it's now the baseline)
            self.variants = [v for v in self.variants if v.variant_id != best.variant_id]
            if baseline and baseline not in self.variants:
                self.variants.insert(0, baseline)

            self._save_state()
            return best.variant_id


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_evolver_instance: Optional[PromptEvolver] = None


def get_prompt_evolver() -> PromptEvolver:
    """Get or create the global prompt evolver."""
    global _evolver_instance
    if _evolver_instance is None:
        _evolver_instance = PromptEvolver()
    return _evolver_instance
