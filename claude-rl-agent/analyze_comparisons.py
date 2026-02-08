#!/usr/bin/env python3
"""
Comparison Analysis Tool

Analyzes Claude vs LLaMA comparison results to provide insights:
1. Overall behavioral similarity trends
2. Per-dimension divergence patterns
3. DPO pair quality assessment
4. Training recommendations
"""

import json
import glob
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

@dataclass
class ComparisonStats:
    """Statistics for a set of comparisons."""
    total_comparisons: int
    avg_overall_similarity: float
    avg_base_similarity: float
    avg_enhanced_similarity: float
    avg_cosine_similarity: float
    critical_divergences: List[str]
    dpo_pairs_generated: int
    most_divergent_dimensions: List[tuple]  # (dimension, avg_difference)

class ComparisonAnalyzer:
    """Analyzes comparison results from automated_runner."""

    def __init__(self, data_dir: str = "data/dpo_pairs"):
        self.data_dir = Path(data_dir)
        self.comparisons: List[Dict] = []

    def load_comparisons(self):
        """Load all comparison JSON files."""
        print(f"Loading comparisons from: {self.data_dir}")

        json_files = list(self.data_dir.glob("comparison_*.json"))
        if not json_files:
            print(f"❌ No comparison files found in {self.data_dir}")
            return

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                self.comparisons.append(data)

        print(f"✅ Loaded {len(self.comparisons)} comparisons")

    def compute_statistics(self) -> ComparisonStats:
        """Compute statistics across all comparisons."""
        if not self.comparisons:
            raise ValueError("No comparisons loaded")

        # Extract behavioral comparison data
        behavioral_data = []
        for comp in self.comparisons:
            if "behavioral_comparison" in comp:
                behavioral_data.append(comp["behavioral_comparison"])

        if not behavioral_data:
            raise ValueError("No behavioral comparison data found")

        # Overall similarities
        overall_sims = [bc.get("overall_similarity", 0) for bc in behavioral_data]
        base_sims = [bc.get("base_similarity_weighted", 0) for bc in behavioral_data]
        enhanced_sims = [bc.get("enhanced_similarity_weighted", 0) for bc in behavioral_data]

        # Cosine similarities from reasoning vectors
        cosine_sims = []
        for bc in behavioral_data:
            enhanced = bc.get("enhanced_signals", {})
            rv = enhanced.get("reasoning_vector", {})
            if "cosine_similarity" in rv:
                cosine_sims.append(rv["cosine_similarity"])

        avg_cosine = np.mean(cosine_sims) if cosine_sims else 0.0

        # Collect critical divergences
        all_divergences = []
        for bc in behavioral_data:
            enhanced = bc.get("enhanced_signals", {})
            rv = enhanced.get("reasoning_vector", {})
            if "critical_divergences" in rv:
                for div in rv["critical_divergences"]:
                    all_divergences.append(div["dimension"])

        # Count DPO pairs (should_train = true)
        dpo_count = sum(1 for bc in behavioral_data if bc.get("should_train", False))

        # Dimension-level analysis
        dimension_diffs = defaultdict(list)
        for bc in behavioral_data:
            enhanced = bc.get("enhanced_signals", {})
            rv = enhanced.get("reasoning_vector", {})
            if "critical_divergences" in rv:
                for div in rv["critical_divergences"]:
                    dimension_diffs[div["dimension"]].append(abs(div["difference"]))

        # Most divergent dimensions
        avg_dimension_diffs = [
            (dim, np.mean(diffs))
            for dim, diffs in dimension_diffs.items()
        ]
        avg_dimension_diffs.sort(key=lambda x: x[1], reverse=True)

        return ComparisonStats(
            total_comparisons=len(self.comparisons),
            avg_overall_similarity=np.mean(overall_sims),
            avg_base_similarity=np.mean(base_sims),
            avg_enhanced_similarity=np.mean(enhanced_sims),
            avg_cosine_similarity=avg_cosine,
            critical_divergences=list(set(all_divergences)),
            dpo_pairs_generated=dpo_count,
            most_divergent_dimensions=avg_dimension_diffs[:10]
        )

    def print_analysis(self):
        """Print comprehensive analysis."""
        stats = self.compute_statistics()

        print("\n" + "="*70)
        print("  📊 COMPARISON ANALYSIS")
        print("="*70)

        print(f"\nOverall Statistics:")
        print(f"  Total Comparisons: {stats.total_comparisons}")
        print(f"  DPO Pairs Generated: {stats.dpo_pairs_generated} ({stats.dpo_pairs_generated/stats.total_comparisons*100:.1f}%)")

        print(f"\nSimilarity Metrics:")
        print(f"  Average Overall Similarity: {stats.avg_overall_similarity:.3f}")
        print(f"  Average Base Similarity: {stats.avg_base_similarity:.3f}")
        print(f"  Average Enhanced Similarity: {stats.avg_enhanced_similarity:.3f}")
        print(f"  Average Cosine Similarity (35-dim): {stats.avg_cosine_similarity:.3f}")

        print(f"\nCritical Divergences Found:")
        for div in stats.critical_divergences[:10]:
            print(f"  - {div}")

        print(f"\nMost Divergent Dimensions:")
        for i, (dim, avg_diff) in enumerate(stats.most_divergent_dimensions, 1):
            print(f"  {i}. {dim}: {avg_diff:.3f} avg difference")

        # Training recommendations
        print(f"\n🎯 Training Recommendations:")

        if stats.avg_overall_similarity < 0.6:
            print("  ⚠️  LOW SIMILARITY (<60%)")
            print("     → High-priority DPO training needed")
            print("     → Consider reviewing SFT training data quality")
        elif stats.avg_overall_similarity < 0.75:
            print("  ⚡ MODERATE SIMILARITY (60-75%)")
            print("     → DPO training will improve alignment")
            print("     → Focus on divergent dimensions")
        else:
            print("  ✅ HIGH SIMILARITY (>75%)")
            print("     → Model is well-aligned with Claude")
            print("     → Fine-tune with DPO for remaining differences")

        if stats.dpo_pairs_generated > 50:
            print(f"\n  ✅ Sufficient DPO pairs ({stats.dpo_pairs_generated}) for training!")
            print("     → Ready to run DPO training")
        else:
            print(f"\n  ⏳ Need more DPO pairs ({stats.dpo_pairs_generated}/50)")
            print(f"     → Run {50 - stats.dpo_pairs_generated} more comparisons")

    def export_summary(self, output_file: str = "data/comparison_summary.json"):
        """Export summary to JSON."""
        stats = self.compute_statistics()

        summary = {
            "total_comparisons": stats.total_comparisons,
            "dpo_pairs_generated": stats.dpo_pairs_generated,
            "avg_overall_similarity": float(stats.avg_overall_similarity),
            "avg_base_similarity": float(stats.avg_base_similarity),
            "avg_enhanced_similarity": float(stats.avg_enhanced_similarity),
            "avg_cosine_similarity": float(stats.avg_cosine_similarity),
            "critical_divergences": stats.critical_divergences,
            "most_divergent_dimensions": [
                {"dimension": dim, "avg_difference": float(diff)}
                for dim, diff in stats.most_divergent_dimensions
            ]
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Summary exported to: {output_path}")

def main():
    print("="*70)
    print("  🔍 Comparison Analysis Tool")
    print("="*70)

    analyzer = ComparisonAnalyzer()

    # Load comparisons
    analyzer.load_comparisons()

    if not analyzer.comparisons:
        print("\n❌ No comparisons to analyze")
        print("   Run comparisons first: ./run_comparisons.sh 10")
        return 1

    # Analyze and print results
    analyzer.print_analysis()

    # Export summary
    analyzer.export_summary()

    return 0

if __name__ == "__main__":
    exit(main())
