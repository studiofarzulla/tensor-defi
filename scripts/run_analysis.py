#!/usr/bin/env python3
"""
TENSOR-DEFI Analysis Pipeline

Regenerates all analysis outputs from current claims_matrix and stats_matrix:
- Cross-sectional analysis (entity alignment, clustering)
- Temporal analysis (rolling window alignment)
- Robustness checks (subsample stability, feature importance)

This script ensures all outputs use consistent data (same N entities, same taxonomy).
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis import CrossSectionalAnalyzer, TemporalAnalyzer, RobustnessChecker
from alignment.procrustes import ProcrustesAlignment
from alignment.congruence import CongruenceCoefficient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_align_data(base_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict, dict]:
    """Load claims and stats matrices and align to common symbols."""
    # Load claims matrix
    claims = np.load(base_path / "outputs" / "nlp" / "claims_matrix.npy")
    with open(base_path / "outputs" / "nlp" / "claims_matrix_meta.json") as f:
        claims_meta = json.load(f)

    # Load stats matrix
    stats = np.load(base_path / "outputs" / "market" / "stats_matrix.npy")
    with open(base_path / "outputs" / "market" / "stats_matrix_meta.json") as f:
        stats_meta = json.load(f)

    # Find common symbols
    claims_symbols = claims_meta['symbols']
    stats_symbols = stats_meta['symbols']
    common_symbols = sorted(set(claims_symbols) & set(stats_symbols))

    logger.info(f"Claims entities: {len(claims_symbols)}")
    logger.info(f"Stats entities: {len(stats_symbols)}")
    logger.info(f"Common entities: {len(common_symbols)}")

    # Align matrices to common symbols
    claims_idx = [claims_symbols.index(s) for s in common_symbols]
    stats_idx = [stats_symbols.index(s) for s in common_symbols]

    claims_common = claims[claims_idx]
    stats_common = stats[stats_idx]

    return claims_common, stats_common, common_symbols, claims_meta, stats_meta


def run_cross_sectional(base_path: Path, claims: np.ndarray, stats: np.ndarray, symbols: list[str]) -> dict:
    """Run cross-sectional analysis."""
    print("\n" + "=" * 60)
    print("CROSS-SECTIONAL ANALYSIS")
    print("=" * 60)

    output_dir = base_path / "outputs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    cs = CrossSectionalAnalyzer(output_dir)

    # Compute entity alignment
    results = cs.compute_entity_alignment(claims, stats, symbols)
    logger.info(f"Full alignment φ: {results['phi_full']:.4f}")

    # Cluster entities
    cluster_results = cs.cluster_entities(claims, stats, symbols)
    results.update(cluster_results)

    # Save
    cs.save_results(results)

    return results


def run_temporal(base_path: Path, claims: np.ndarray, symbols: list[str]) -> dict:
    """Run temporal analysis."""
    print("\n" + "=" * 60)
    print("TEMPORAL ANALYSIS")
    print("=" * 60)

    data_dir = base_path / "data"
    output_dir = base_path / "outputs" / "analysis"

    ta = TemporalAnalyzer(data_dir, output_dir)

    # Compute rolling stats from market data
    rolling_stats = ta.compute_rolling_stats(window_months=6, step_months=3)

    if not rolling_stats.get('windows'):
        logger.warning("No temporal windows generated - check market data")
        return {}

    logger.info(f"Generated {len(rolling_stats['windows'])} rolling windows")

    # Analyze alignment stability
    results = ta.analyze_alignment_stability(claims, rolling_stats, symbols)
    results['rolling_stats'] = rolling_stats

    # Save
    ta.save_results(results)

    return results


def run_robustness(base_path: Path, claims: np.ndarray, stats: np.ndarray,
                   categories: list[str], statistics: list[str]) -> dict:
    """Run robustness checks."""
    print("\n" + "=" * 60)
    print("ROBUSTNESS ANALYSIS")
    print("=" * 60)

    output_dir = base_path / "outputs" / "analysis"

    rc = RobustnessChecker(output_dir)

    # Subsample stability (bootstrap)
    logger.info("Running subsample stability test...")
    subsample_results = rc.subsample_stability(claims, stats, list(range(len(claims))))

    # Feature importance (ablation)
    logger.info("Running feature importance analysis...")
    importance_results = rc.feature_importance(claims, stats, categories, statistics)

    results = {
        'subsample_stability': subsample_results,
        'feature_importance': importance_results,
        'rank_sensitivity': {
            'rank_sensitivity': [],
            'note': 'Run via run_expansion.py for rank sensitivity'
        }
    }

    # Save
    rc.save_results(results)

    return results


def run_alignment_verification(claims: np.ndarray, stats: np.ndarray, symbols: list[str]) -> dict:
    """Quick verification of alignment values."""
    print("\n" + "=" * 60)
    print("ALIGNMENT VERIFICATION")
    print("=" * 60)

    aligner = ProcrustesAlignment()
    congruence = CongruenceCoefficient()

    result = aligner.align_matrices(claims, stats)
    cong = congruence.matrix_congruence(result['source_rotated'], result['target_centered'])

    print(f"  N entities: {len(symbols)}")
    print(f"  Claims shape: {claims.shape}")
    print(f"  Stats shape: {stats.shape}")
    print(f"  Alignment φ: {cong['mean_phi']:.4f}")
    print(f"  Per-column φ: {[f'{p:.3f}' for p in cong['column_phis'][:7]]}")

    return {
        'n_entities': len(symbols),
        'phi': float(cong['mean_phi']),
        'column_phis': [float(p) for p in cong['column_phis']]
    }


def main():
    """Run full analysis pipeline."""
    base_path = Path(__file__).parent.parent

    print("=" * 70)
    print("TENSOR-DEFI: Analysis Pipeline")
    print("=" * 70)

    # Load and align data
    claims, stats, symbols, claims_meta, stats_meta = load_and_align_data(base_path)

    # Verification
    verification = run_alignment_verification(claims, stats, symbols)

    # Cross-sectional analysis
    cs_results = run_cross_sectional(base_path, claims, stats, symbols)

    # Temporal analysis (uses full claims matrix for symbol lookup)
    claims_full = np.load(base_path / "outputs" / "nlp" / "claims_matrix.npy")
    ta_results = run_temporal(base_path, claims_full, claims_meta['symbols'])

    # Robustness checks
    rob_results = run_robustness(
        base_path, claims, stats,
        claims_meta['categories'],
        stats_meta['statistics']
    )

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nData Summary:")
    print(f"  N entities: {len(symbols)}")
    print(f"  Claim categories: {len(claims_meta['categories'])}")
    print(f"  Statistics: {len(stats_meta['statistics'])}")

    print(f"\nKey Results:")
    print(f"  Cross-sectional φ_full: {cs_results.get('phi_full', 'N/A'):.4f}")
    print(f"  Temporal mean φ: {ta_results.get('mean_phi', 'N/A'):.4f}")
    print(f"  Robustness mean φ: {rob_results['subsample_stability']['mean_phi']:.4f}")

    print(f"\nOutput files:")
    output_dir = base_path / "outputs" / "analysis"
    for f in sorted(output_dir.glob("*.json")):
        print(f"  - {f.name}")

    print(f"\nCategories used for feature importance:")
    for cat in claims_meta['categories']:
        print(f"  - {cat}")


if __name__ == "__main__":
    main()
