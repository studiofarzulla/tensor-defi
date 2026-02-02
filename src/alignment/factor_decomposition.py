#!/usr/bin/env python3
"""
Factor-Statistic Decomposition Analysis.

After Procrustes rotation, decompose which statistics correlate
with which latent factors. Addresses reviewer question:
"Can you decompose which statistics align most with which factors?"
"""

import numpy as np
import json
from pathlib import Path
from scipy import stats as sp_stats
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
import seaborn as sns


def load_matrices(base_path: Path) -> tuple:
    """Load matrices and align to common assets."""
    # Load matrices
    claims = np.load(base_path / 'outputs/nlp/claims_matrix.npy')
    stats_matrix = np.load(base_path / 'outputs/market/stats_matrix.npy')
    factors = np.load(base_path / 'outputs/tensor/cp_asset_factors.npy')

    # Load metadata
    with open(base_path / 'outputs/nlp/claims_matrix_meta.json') as f:
        claims_meta = json.load(f)
    with open(base_path / 'outputs/market/stats_matrix_meta.json') as f:
        stats_meta = json.load(f)
    with open(base_path / 'outputs/tensor/cp_factors_meta.json') as f:
        factors_meta = json.load(f)

    claims_symbols = claims_meta['symbols']
    stats_symbols = stats_meta['symbols']
    factors_symbols = factors_meta['symbols']
    stat_names = stats_meta['statistics']
    category_names = claims_meta['categories']

    # Find common assets
    common = sorted(set(claims_symbols) & set(stats_symbols) & set(factors_symbols))

    # Align matrices
    claims_idx = [claims_symbols.index(s) for s in common]
    stats_idx = [stats_symbols.index(s) for s in common]
    factors_idx = [factors_symbols.index(s) for s in common]

    claims_aligned = claims[claims_idx]
    stats_aligned = stats_matrix[stats_idx]
    factors_aligned = factors[factors_idx]

    return (claims_aligned, stats_aligned, factors_aligned,
            common, stat_names, category_names)


def procrustes_rotation(X: np.ndarray, Y: np.ndarray) -> tuple:
    """Compute Procrustes rotation matrix R such that X @ R ≈ Y."""
    # Center and normalize
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # Find optimal rotation
    R, scale = orthogonal_procrustes(X_centered, Y_centered)

    return R, scale


def compute_correlations(stats: np.ndarray, factors: np.ndarray,
                         stat_names: list) -> dict:
    """Compute per-statistic correlations with each factor."""
    n_stats = stats.shape[1]
    n_factors = factors.shape[1]

    correlations = np.zeros((n_stats, n_factors))
    p_values = np.zeros((n_stats, n_factors))

    for i in range(n_stats):
        for j in range(n_factors):
            r, p = sp_stats.pearsonr(stats[:, i], factors[:, j])
            correlations[i, j] = r
            p_values[i, j] = p

    return {
        'correlations': correlations,
        'p_values': p_values,
        'stat_names': stat_names,
        'factor_names': [f'Factor {j+1}' for j in range(n_factors)]
    }


def compute_claims_factor_correlations(claims: np.ndarray, factors: np.ndarray,
                                        category_names: list) -> dict:
    """Compute per-category correlations with each factor."""
    n_cats = claims.shape[1]
    n_factors = factors.shape[1]

    correlations = np.zeros((n_cats, n_factors))
    p_values = np.zeros((n_cats, n_factors))

    for i in range(n_cats):
        for j in range(n_factors):
            r, p = sp_stats.pearsonr(claims[:, i], factors[:, j])
            correlations[i, j] = r
            p_values[i, j] = p

    return {
        'correlations': correlations,
        'p_values': p_values,
        'category_names': category_names,
        'factor_names': [f'Factor {j+1}' for j in range(n_factors)]
    }


def plot_heatmaps(stats_corr: dict, claims_corr: dict, output_path: Path):
    """Create heatmaps of correlations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Statistics vs Factors
    ax1 = axes[0]
    im1 = ax1.imshow(stats_corr['correlations'], cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax1.set_xticks(range(len(stats_corr['factor_names'])))
    ax1.set_xticklabels(stats_corr['factor_names'])
    ax1.set_yticks(range(len(stats_corr['stat_names'])))
    ax1.set_yticklabels([s.replace('_', ' ').title() for s in stats_corr['stat_names']])
    ax1.set_title('Statistics ↔ Factors\n(correlation matrix)', fontsize=12)

    # Add significance markers
    for i in range(len(stats_corr['stat_names'])):
        for j in range(len(stats_corr['factor_names'])):
            r = stats_corr['correlations'][i, j]
            p = stats_corr['p_values'][i, j]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            color = 'white' if abs(r) > 0.5 else 'black'
            ax1.text(j, i, f'{r:.2f}{sig}', ha='center', va='center',
                     fontsize=9, color=color)

    # Claims vs Factors
    ax2 = axes[1]
    im2 = ax2.imshow(claims_corr['correlations'], cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax2.set_xticks(range(len(claims_corr['factor_names'])))
    ax2.set_xticklabels(claims_corr['factor_names'])
    ax2.set_yticks(range(len(claims_corr['category_names'])))
    ax2.set_yticklabels([c.replace('_', ' ').title() for c in claims_corr['category_names']])
    ax2.set_title('Claims ↔ Factors\n(correlation matrix)', fontsize=12)

    # Add significance markers
    for i in range(len(claims_corr['category_names'])):
        for j in range(len(claims_corr['factor_names'])):
            r = claims_corr['correlations'][i, j]
            p = claims_corr['p_values'][i, j]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            color = 'white' if abs(r) > 0.5 else 'black'
            ax2.text(j, i, f'{r:.2f}{sig}', ha='center', va='center',
                     fontsize=9, color=color)

    # Colorbar
    cbar = fig.colorbar(im2, ax=axes, shrink=0.8, label='Pearson r')

    plt.suptitle('Factor Loading Decomposition', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(output_path / 'factor_decomposition_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / 'factor_decomposition_heatmap.png'}")


def identify_key_drivers(stats_corr: dict, claims_corr: dict) -> dict:
    """Identify which features drive each factor."""
    results = {}

    # Statistics drivers
    stat_drivers = {}
    for j, factor in enumerate(stats_corr['factor_names']):
        correlations = stats_corr['correlations'][:, j]
        p_values = stats_corr['p_values'][:, j]

        # Sort by absolute correlation
        sorted_idx = np.argsort(np.abs(correlations))[::-1]

        drivers = []
        for idx in sorted_idx:
            drivers.append({
                'statistic': stats_corr['stat_names'][idx],
                'correlation': float(correlations[idx]),
                'p_value': float(p_values[idx]),
                'significant': bool(p_values[idx] < 0.05)
            })
        stat_drivers[factor] = drivers

    results['stat_drivers'] = stat_drivers

    # Claims drivers (or lack thereof)
    claims_drivers = {}
    for j, factor in enumerate(claims_corr['factor_names']):
        correlations = claims_corr['correlations'][:, j]
        p_values = claims_corr['p_values'][:, j]

        sorted_idx = np.argsort(np.abs(correlations))[::-1]

        drivers = []
        for idx in sorted_idx:
            drivers.append({
                'category': claims_corr['category_names'][idx],
                'correlation': float(correlations[idx]),
                'p_value': float(p_values[idx]),
                'significant': bool(p_values[idx] < 0.05)
            })
        claims_drivers[factor] = drivers

    results['claims_drivers'] = claims_drivers

    return results


def main():
    base_path = Path(__file__).parent.parent.parent
    output_path = base_path / 'outputs' / 'alignment'
    fig_path = base_path / 'outputs' / 'figures'
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FACTOR-STATISTIC DECOMPOSITION")
    print("=" * 60)

    # Load data
    print("\nLoading matrices...")
    (claims, stats, factors,
     common_assets, stat_names, category_names) = load_matrices(base_path)

    print(f"  Assets: {len(common_assets)}")
    print(f"  Statistics: {stat_names}")
    print(f"  Categories: {category_names}")

    # Compute correlations
    print("\nComputing correlations...")
    stats_corr = compute_correlations(stats, factors, stat_names)
    claims_corr = compute_claims_factor_correlations(claims, factors, category_names)

    # Generate heatmaps
    print("\nGenerating heatmaps...")
    plot_heatmaps(stats_corr, claims_corr, fig_path)

    # Identify key drivers
    print("\nIdentifying key drivers...")
    drivers = identify_key_drivers(stats_corr, claims_corr)

    # Print summary
    print("\n" + "-" * 40)
    print("STATISTICS → FACTORS")
    print("-" * 40)
    for factor, driver_list in drivers['stat_drivers'].items():
        print(f"\n{factor}:")
        for d in driver_list[:3]:  # Top 3
            sig = "*" if d['significant'] else ""
            print(f"  {d['statistic']:15} r={d['correlation']:+.3f}{sig} (p={d['p_value']:.3f})")

    print("\n" + "-" * 40)
    print("CLAIMS → FACTORS (weak expected)")
    print("-" * 40)
    for factor, driver_list in drivers['claims_drivers'].items():
        print(f"\n{factor}:")
        for d in driver_list[:3]:  # Top 3
            sig = "*" if d['significant'] else ""
            print(f"  {d['category']:20} r={d['correlation']:+.3f}{sig} (p={d['p_value']:.3f})")

    # Count significant relationships
    n_sig_stats = sum(1 for r in stats_corr['p_values'].flatten() if r < 0.05)
    n_total_stats = stats_corr['p_values'].size
    n_sig_claims = sum(1 for r in claims_corr['p_values'].flatten() if r < 0.05)
    n_total_claims = claims_corr['p_values'].size

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Statistics-Factors: {n_sig_stats}/{n_total_stats} significant pairs ({100*n_sig_stats/n_total_stats:.1f}%)")
    print(f"Claims-Factors: {n_sig_claims}/{n_total_claims} significant pairs ({100*n_sig_claims/n_total_claims:.1f}%)")

    # Save results
    output = {
        'n_assets': len(common_assets),
        'statistics_factors': {
            'correlations': stats_corr['correlations'].tolist(),
            'p_values': stats_corr['p_values'].tolist(),
            'stat_names': stats_corr['stat_names'],
            'factor_names': stats_corr['factor_names'],
            'n_significant': n_sig_stats,
            'n_total': n_total_stats
        },
        'claims_factors': {
            'correlations': claims_corr['correlations'].tolist(),
            'p_values': claims_corr['p_values'].tolist(),
            'category_names': claims_corr['category_names'],
            'factor_names': claims_corr['factor_names'],
            'n_significant': n_sig_claims,
            'n_total': n_total_claims
        },
        'drivers': drivers
    }

    with open(output_path / 'factor_decomposition.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path / 'factor_decomposition.json'}")

    return output


if __name__ == '__main__':
    main()
