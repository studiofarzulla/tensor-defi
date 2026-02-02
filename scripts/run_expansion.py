#!/usr/bin/env python3
"""
TENSOR-DEFI Paper Expansion: Additional Analyses

Runs Tucker decomposition comparison and rank sensitivity analysis.
Prepares data for expanded figures.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensor_ops.decomposition import TensorDecomposition
from alignment.procrustes import ProcrustesAlignment
from alignment.congruence import CongruenceCoefficient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_tucker_comparison(base_path: Path, output_dir: Path):
    """Run Tucker decomposition and compare to CP."""
    print("\n" + "=" * 60)
    print("TUCKER VS CP COMPARISON")
    print("=" * 60)

    decomp = TensorDecomposition(
        tensor_dir=base_path / "outputs" / "tensor",
        output_dir=output_dir
    )

    # Run Tucker with ranks [5, 2, 2] - time compressed, asset and feature at rank 2
    # Time mode needs higher rank to capture dynamics
    ranks = [5, 2, 2]
    core, factors, tucker_exp_var = decomp.tucker_decomposition(ranks)

    # Extract asset factors (mode 1)
    tucker_asset_factors = factors[1]

    # Save Tucker results
    np.save(output_dir / "tucker_asset_factors.npy", tucker_asset_factors)

    # Get CP results for comparison
    cp_factors, cp_exp_var = decomp.cp_decomposition(rank=2)
    cp_asset_factors = cp_factors[1]

    # Run alignment with both
    aligner = ProcrustesAlignment()
    congruence = CongruenceCoefficient()

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

    # CP alignment
    cp_symbols = decomp.metadata['symbols']
    common_cp = sorted(set(claims_symbols) & set(cp_symbols))
    claims_idx = [claims_symbols.index(s) for s in common_cp]
    cp_idx = [cp_symbols.index(s) for s in common_cp]

    claims_common = claims[claims_idx]
    cp_common = cp_asset_factors[cp_idx]

    result_cp = aligner.align_matrices(claims_common, cp_common)
    phi_cp = congruence.matrix_congruence(result_cp['source_rotated'], result_cp['target_centered'])['mean_phi']

    # Tucker alignment
    tucker_common = tucker_asset_factors[cp_idx]
    result_tucker = aligner.align_matrices(claims_common, tucker_common)
    phi_tucker = congruence.matrix_congruence(result_tucker['source_rotated'], result_tucker['target_centered'])['mean_phi']

    results = {
        'cp': {
            'rank': 2,
            'explained_variance': float(cp_exp_var),
            'alignment_phi': float(phi_cp),
            'factor_shape': list(cp_asset_factors.shape)
        },
        'tucker': {
            'ranks': ranks,
            'explained_variance': float(tucker_exp_var),
            'alignment_phi': float(phi_tucker),
            'factor_shape': list(tucker_asset_factors.shape),
            'core_shape': list(core.shape)
        }
    }

    with open(output_dir / "tucker_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nCP Decomposition (rank 2):")
    print(f"  Explained variance: {cp_exp_var:.4f}")
    print(f"  Alignment φ: {phi_cp:.4f}")

    print(f"\nTucker Decomposition (ranks {ranks}):")
    print(f"  Explained variance: {tucker_exp_var:.4f}")
    print(f"  Alignment φ: {phi_tucker:.4f}")

    return results


def run_rank_sensitivity(base_path: Path, output_dir: Path):
    """Test CP decomposition at multiple ranks."""
    print("\n" + "=" * 60)
    print("RANK SENSITIVITY ANALYSIS")
    print("=" * 60)

    decomp = TensorDecomposition(
        tensor_dir=base_path / "outputs" / "tensor",
        output_dir=output_dir
    )

    # Load claims for alignment testing
    claims = np.load(base_path / "outputs" / "nlp" / "claims_matrix.npy")
    with open(base_path / "outputs" / "nlp" / "claims_matrix_meta.json") as f:
        claims_meta = json.load(f)
    claims_symbols = claims_meta['symbols']

    aligner = ProcrustesAlignment()
    congruence = CongruenceCoefficient()

    ranks_to_test = [1, 2, 3, 4, 5]
    results = []

    for rank in ranks_to_test:
        print(f"\nTesting rank {rank}...")
        factors, exp_var = decomp.cp_decomposition(rank)
        asset_factors = factors[1]

        # Find common symbols
        cp_symbols = decomp.metadata['symbols']
        common = sorted(set(claims_symbols) & set(cp_symbols))
        claims_idx = [claims_symbols.index(s) for s in common]
        cp_idx = [cp_symbols.index(s) for s in common]

        claims_common = claims[claims_idx]
        factors_common = asset_factors[cp_idx]

        # Alignment test
        try:
            result = aligner.align_matrices(claims_common, factors_common)
            phi = congruence.matrix_congruence(result['source_rotated'], result['target_centered'])['mean_phi']
        except Exception as e:
            logger.warning(f"Rank {rank} alignment failed: {e}")
            phi = 0.0

        results.append({
            'rank': rank,
            'explained_variance': float(exp_var),
            'alignment_phi': float(phi)
        })

        print(f"  Explained variance: {exp_var:.4f}")
        print(f"  Alignment φ: {phi:.4f}")

    with open(output_dir / "rank_sensitivity.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def extract_per_dimension_phi(base_path: Path, output_dir: Path):
    """Extract and format per-dimension φ values from existing alignment results."""
    print("\n" + "=" * 60)
    print("PER-DIMENSION PHI EXTRACTION")
    print("=" * 60)

    with open(base_path / "outputs" / "alignment" / "alignment_results.json") as f:
        alignment = json.load(f)

    # Get category names
    with open(base_path / "outputs" / "nlp" / "claims_matrix_meta.json") as f:
        claims_meta = json.load(f)
    categories = claims_meta['categories']

    # Extract per-column phis for claims-stats (most relevant)
    claims_stats_phis = alignment['claims_stats']['column_phis']
    claims_factors_phis = alignment['claims_factors']['column_phis']

    # Create structured output
    # Note: column_phis has 10 values (padded to max dimension)
    # First 7 correspond to stats dimensions after padding
    results = {
        'claims_stats': {
            'column_phis': claims_stats_phis,
            'mean_phi': alignment['claims_stats']['mean_phi'],
            'dimensions': 10  # Padded dimension
        },
        'claims_factors': {
            'column_phis': claims_factors_phis,
            'mean_phi': alignment['claims_factors']['mean_phi'],
            'dimensions': 10
        },
        'categories': categories
    }

    with open(output_dir / "per_dimension_phi.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nClaims-Stats per-dimension φ:")
    for i, phi in enumerate(claims_stats_phis):
        if phi > 0:
            print(f"  Dim {i+1}: {phi:.4f}")

    return results


def prepare_temporal_data(base_path: Path, output_dir: Path):
    """Extract temporal phi evolution for plotting."""
    print("\n" + "=" * 60)
    print("TEMPORAL DATA PREPARATION")
    print("=" * 60)

    with open(base_path / "outputs" / "analysis" / "temporal_analysis.json") as f:
        temporal = json.load(f)

    # Extract window data
    windows = temporal['temporal_alignment']

    # Format for plotting
    plot_data = []
    for w in windows:
        plot_data.append({
            'window_id': w['window_id'],
            'start': w['start'][:10],  # Just date
            'end': w['end'][:10],
            'phi': w['phi'],
            'label': f"{w['start'][5:7]}/{w['start'][2:4]}-{w['end'][5:7]}/{w['end'][2:4]}"
        })

    results = {
        'windows': plot_data,
        'mean_phi': temporal['mean_phi'],
        'std_phi': temporal['std_phi'],
        'n_windows': temporal['n_windows']
    }

    with open(output_dir / "temporal_plot_data.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTemporal evolution ({len(plot_data)} windows):")
    for w in plot_data:
        print(f"  {w['label']}: φ = {w['phi']:.4f}")

    print(f"\nMean φ: {temporal['mean_phi']:.4f} ± {temporal['std_phi']:.4f}")

    return results


def extract_factor_loadings(base_path: Path, output_dir: Path):
    """Extract and format factor loadings for visualization."""
    print("\n" + "=" * 60)
    print("FACTOR LOADING EXTRACTION")
    print("=" * 60)

    # Load CP factors
    factors = np.load(base_path / "outputs" / "tensor" / "cp_asset_factors.npy")
    with open(base_path / "outputs" / "tensor" / "cp_factors_meta.json") as f:
        meta = json.load(f)

    symbols = meta['symbols']

    # Create DataFrame for analysis
    df = pd.DataFrame(factors, index=symbols, columns=['Factor_1', 'Factor_2'])

    # Sort by Factor 1 loading
    df_sorted = df.sort_values('Factor_1', ascending=False)

    # Identify outliers (> 2 std from mean)
    f1_mean, f1_std = df['Factor_1'].mean(), df['Factor_1'].std()
    f2_mean, f2_std = df['Factor_2'].mean(), df['Factor_2'].std()

    outliers = []
    for sym in symbols:
        f1, f2 = df.loc[sym]
        is_outlier = abs(f1 - f1_mean) > 2 * f1_std or abs(f2 - f2_mean) > 2 * f2_std
        if is_outlier:
            outliers.append(sym)

    results = {
        'factor_loadings': df.to_dict('index'),
        'top_10_factor1': df_sorted.head(10).index.tolist(),
        'bottom_10_factor1': df_sorted.tail(10).index.tolist(),
        'outliers': outliers,
        'factor1_stats': {'mean': float(f1_mean), 'std': float(f1_std)},
        'factor2_stats': {'mean': float(f2_mean), 'std': float(f2_std)}
    }

    with open(output_dir / "factor_loadings.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nFactor 1 - Top 5 loadings:")
    for sym in df_sorted.head(5).index:
        print(f"  {sym}: {df.loc[sym, 'Factor_1']:.4f}")

    print(f"\nOutliers (>2σ): {outliers}")

    return results


def prepare_feature_importance(base_path: Path, output_dir: Path):
    """Format feature importance for plotting."""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE PREPARATION")
    print("=" * 60)

    with open(base_path / "outputs" / "analysis" / "robustness_analysis.json") as f:
        robustness = json.load(f)

    importance = robustness['feature_importance']['claims_importance']

    # Sort by importance
    importance_sorted = sorted(importance, key=lambda x: x['importance'], reverse=True)

    results = {
        'importance': importance_sorted,
        'phi_full': robustness['feature_importance']['phi_full'],
        'most_important': robustness['feature_importance']['most_important']
    }

    with open(output_dir / "feature_importance_plot.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\nFeature importance (sorted):")
    for item in importance_sorted:
        sign = '+' if item['importance'] > 0 else ''
        print(f"  {item['feature']:20s}: {sign}{item['importance']:.4f}")

    return results


def prepare_entity_impact(base_path: Path, output_dir: Path):
    """Format entity impact for plotting."""
    print("\n" + "=" * 60)
    print("ENTITY IMPACT PREPARATION")
    print("=" * 60)

    with open(base_path / "outputs" / "analysis" / "cross_sectional_analysis.json") as f:
        cross_sectional = json.load(f)

    entities = cross_sectional['entity_analysis']

    # Already sorted by impact in original file
    results = {
        'entities': entities,
        'phi_full': cross_sectional['phi_full'],
        'best_aligned': cross_sectional['best_aligned'],
        'worst_aligned': cross_sectional['worst_aligned'],
        'clusters': cross_sectional['clusters']
    }

    with open(output_dir / "entity_impact_plot.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\nEntity impact on alignment:")
    for e in entities:
        sign = '+' if e['impact'] > 0 else ''
        print(f"  {e['symbol']:5s}: {sign}{e['impact']:.4f} ({e['interpretation']})")

    return results


def extract_tensor_slice(base_path: Path, output_dir: Path):
    """Extract a tensor slice for heatmap visualization."""
    print("\n" + "=" * 60)
    print("TENSOR SLICE EXTRACTION")
    print("=" * 60)

    tensor = np.load(base_path / "outputs" / "tensor" / "market_tensor.npy")
    with open(base_path / "outputs" / "tensor" / "tensor_meta.json") as f:
        meta = json.load(f)

    # Take middle timestamp
    mid_t = tensor.shape[0] // 2

    # Extract slice: assets × features at time t
    # Shape is (time, venue, asset, feature) -> squeeze venue
    slice_data = tensor[mid_t, 0, :, :]  # (49 assets, 5 features)

    # Normalize for visualization
    slice_normalized = (slice_data - slice_data.mean(axis=0)) / (slice_data.std(axis=0) + 1e-8)

    results = {
        'timestamp_index': mid_t,
        'timestamp': meta['time_start'],  # Approximate
        'assets': meta['symbols'],
        'features': meta['features'],
        'shape': list(slice_normalized.shape),
        'data': slice_normalized.tolist()
    }

    with open(output_dir / "tensor_slice.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nExtracted slice at t={mid_t}")
    print(f"Shape: {slice_normalized.shape}")
    print(f"Assets: {len(meta['symbols'])}")
    print(f"Features: {meta['features']}")

    return results


def main():
    """Run all expansion analyses."""
    base_path = Path(__file__).parent.parent
    output_dir = base_path / "outputs" / "expansion"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TENSOR-DEFI: Paper Expansion Analyses")
    print("=" * 70)

    # 1. Tucker vs CP comparison
    tucker_results = run_tucker_comparison(base_path, output_dir)

    # 2. Rank sensitivity
    rank_results = run_rank_sensitivity(base_path, output_dir)

    # 3. Per-dimension phi
    dim_results = extract_per_dimension_phi(base_path, output_dir)

    # 4. Temporal data
    temporal_results = prepare_temporal_data(base_path, output_dir)

    # 5. Factor loadings
    factor_results = extract_factor_loadings(base_path, output_dir)

    # 6. Feature importance
    importance_results = prepare_feature_importance(base_path, output_dir)

    # 7. Entity impact
    entity_results = prepare_entity_impact(base_path, output_dir)

    # 8. Tensor slice
    slice_results = extract_tensor_slice(base_path, output_dir)

    print("\n" + "=" * 70)
    print("EXPANSION ANALYSES COMPLETE")
    print("=" * 70)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.json")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
