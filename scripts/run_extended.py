#!/usr/bin/env python3
"""
Extended Analysis Pipeline for TENSOR-DEFI

Phase 6: Temporal, cross-sectional, and robustness analysis.
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


def main():
    base_path = Path(__file__).parent.parent

    print("=" * 60)
    print("TENSOR-DEFI: Extended Analysis Pipeline (Phase 6)")
    print("=" * 60)

    # Load required data
    print("\nLoading data...")

    claims_matrix = np.load(base_path / "outputs/nlp/claims_matrix.npy")
    stats_matrix = np.load(base_path / "outputs/market/stats_matrix.npy")
    tensor = np.load(base_path / "outputs/tensor/market_tensor.npy")

    with open(base_path / "outputs/nlp/claims_matrix_meta.json") as f:
        claims_meta = json.load(f)
    with open(base_path / "outputs/market/stats_matrix_meta.json") as f:
        stats_meta = json.load(f)

    claims_symbols = claims_meta['symbols']
    stats_symbols = stats_meta['symbols']
    claim_categories = claims_meta['categories']
    stat_features = stats_meta['statistics']

    # Find common symbols
    common_symbols = [s for s in claims_symbols if s in stats_symbols]
    print(f"Common symbols: {len(common_symbols)}")

    # Subset matrices to common symbols
    claims_idx = [claims_symbols.index(s) for s in common_symbols]
    stats_idx = [stats_symbols.index(s) for s in common_symbols]

    claims_common = claims_matrix[claims_idx]
    stats_common = stats_matrix[stats_idx]

    output_dir = base_path / "outputs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== 1. TEMPORAL ANALYSIS =====
    print("\n" + "=" * 60)
    print("1. TEMPORAL ANALYSIS")
    print("=" * 60)

    from analysis.temporal import TemporalAnalyzer

    temporal = TemporalAnalyzer(
        data_dir=base_path / "data",
        output_dir=output_dir
    )

    rolling_stats = temporal.compute_rolling_stats(window_months=6, step_months=3)
    temporal_results = temporal.analyze_alignment_stability(
        claims_common, rolling_stats, common_symbols
    )
    temporal_results['rolling_stats'] = rolling_stats
    temporal.save_results(temporal_results)

    # ===== 2. CROSS-SECTIONAL ANALYSIS =====
    print("\n" + "=" * 60)
    print("2. CROSS-SECTIONAL ANALYSIS")
    print("=" * 60)

    from analysis.cross_sectional import CrossSectionalAnalyzer

    cross_sectional = CrossSectionalAnalyzer(output_dir=output_dir)

    entity_results = cross_sectional.compute_entity_alignment(
        claims_common, stats_common, common_symbols
    )

    try:
        cluster_results = cross_sectional.cluster_entities(
            claims_common, stats_common, common_symbols, n_clusters=3
        )
        entity_results.update(cluster_results)
    except Exception as e:
        print(f"Clustering skipped: {e}")

    cross_sectional.save_results(entity_results)

    # ===== 3. ROBUSTNESS CHECKS =====
    print("\n" + "=" * 60)
    print("3. ROBUSTNESS CHECKS")
    print("=" * 60)

    from analysis.robustness import RobustnessChecker

    robustness = RobustnessChecker(output_dir=output_dir)

    # Subsample stability
    print("\nRunning subsample stability test...")
    subsample_results = robustness.subsample_stability(
        claims_common, stats_common, common_symbols,
        n_iterations=100, subsample_frac=0.8
    )

    # Feature importance
    print("\nRunning feature importance analysis...")
    importance_results = robustness.feature_importance(
        claims_common, stats_common,
        claim_categories, stat_features
    )

    # Rank sensitivity (skip if tensor too large)
    rank_results = {'rank_sensitivity': [], 'note': 'Skipped - using pre-computed factors'}

    robustness_results = {
        'subsample_stability': subsample_results,
        'feature_importance': importance_results,
        'rank_sensitivity': rank_results
    }
    robustness.save_results(robustness_results)

    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("EXTENDED ANALYSIS COMPLETE")
    print("=" * 60)

    print(f"\nOutput files:")
    for f in output_dir.glob("*.json"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
