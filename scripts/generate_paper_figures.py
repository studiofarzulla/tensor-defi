#!/usr/bin/env python3
"""
Generate publication-quality figures for tensor-defi paper
Runs CP decomposition at rank 4 and creates proper visualizations
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import tensorly as tl
from tensorly.decomposition import parafac, tucker
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "tensors"
OUTPUT_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Colors
COLORS = {
    'btc': '#F7931A',  # Bitcoin orange
    'eth': '#627EEA',  # Ethereum blue
    'sol': '#00FFA3',  # Solana green
    'cp': '#2563EB',   # Blue
    'tucker': '#8B5CF6',  # Purple
    'pca': '#EF4444',  # Red
}

ASSET_NAMES = ['BTC', 'ETH', 'SOL']
FEATURE_NAMES = ['Open', 'High', 'Low', 'Close', 'Volume']


def load_tensor():
    """Load the normalized OHLCV tensor."""
    with open(DATA_DIR / "normalized_ohlcv_tensor.pkl", 'rb') as f:
        data = pickle.load(f)

    tensor = data['tensor']
    metadata = data['metadata']
    timestamps = metadata['timestamps']

    # Remove venue dimension (only 1 venue)
    tensor = tensor.squeeze(axis=1)  # (8761, 3, 5)

    print(f"Tensor shape: {tensor.shape}")
    print(f"Time range: {timestamps[0]} to {timestamps[-1]}")

    return tensor, timestamps


def run_decompositions(tensor, rank=4):
    """Run CP, Tucker, and PCA decompositions."""
    print(f"\n=== Running decompositions at rank {rank} ===")

    # CP decomposition
    print("Running CP decomposition...")
    weights, factors = parafac(tensor, rank=rank, init='random', random_state=42)
    cp_reconstruction = tl.cp_to_tensor((weights, factors))
    cp_error = np.linalg.norm(tensor - cp_reconstruction) / np.linalg.norm(tensor)
    cp_variance = 1 - cp_error**2
    print(f"  CP explained variance: {cp_variance:.4f}")

    # Tucker decomposition
    print("Running Tucker decomposition...")
    core, tucker_factors = tucker(tensor, rank=[rank, min(rank, 3), min(rank, 5)], init='random', random_state=42)
    tucker_reconstruction = tl.tucker_to_tensor((core, tucker_factors))
    tucker_error = np.linalg.norm(tensor - tucker_reconstruction) / np.linalg.norm(tensor)
    tucker_variance = 1 - tucker_error**2
    print(f"  Tucker explained variance: {tucker_variance:.4f}")

    # PCA baseline (flatten to matrix)
    print("Running PCA baseline...")
    T, A, F = tensor.shape
    matrix = tensor.reshape(T, A * F)  # (8761, 15)
    pca = PCA(n_components=rank)
    pca.fit(matrix)
    pca_reconstruction = pca.inverse_transform(pca.transform(matrix))
    pca_error = np.linalg.norm(matrix - pca_reconstruction) / np.linalg.norm(matrix)
    pca_variance = 1 - pca_error**2
    print(f"  PCA explained variance: {pca_variance:.4f}")

    return {
        'cp': {
            'weights': weights,
            'factors': factors,
            'variance': cp_variance,
            'error': cp_error
        },
        'tucker': {
            'core': core,
            'factors': tucker_factors,
            'variance': tucker_variance,
            'error': tucker_error
        },
        'pca': {
            'model': pca,
            'variance': pca_variance,
            'error': pca_error,
            'components': pca.components_
        }
    }


def plot_method_comparison(results, save_path):
    """Create bar chart comparing CP, Tucker, and PCA performance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods = ['CP', 'Tucker', 'PCA']
    variances = [results['cp']['variance'], results['tucker']['variance'], results['pca']['variance']]
    errors = [results['cp']['error'], results['tucker']['error'], results['pca']['error']]
    colors = [COLORS['cp'], COLORS['tucker'], COLORS['pca']]

    # Explained variance
    bars1 = axes[0].bar(methods, variances, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Explained Variance')
    axes[0].set_title('Reconstruction Performance by Method')
    axes[0].set_ylim([0.85, 1.0])
    axes[0].grid(axis='y', alpha=0.3)

    for bar, var in zip(bars1, variances):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{var:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Reconstruction error
    bars2 = axes[1].bar(methods, errors, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Relative Reconstruction Error')
    axes[1].set_title('Error Comparison (Lower is Better)')
    axes[1].grid(axis='y', alpha=0.3)

    for bar, err in zip(bars2, errors):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., height + 0.002,
                     f'{err:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def plot_cp_factors(weights, factors, timestamps, save_path):
    """Create comprehensive CP factor visualization."""
    temporal_factors = factors[0]  # (8761, 4)
    asset_factors = factors[1]     # (3, 4)
    feature_factors = factors[2]   # (5, 4)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.5, 1, 1])

    # Temporal factors (top, spanning both columns)
    ax_time = fig.add_subplot(gs[0, :])

    # Downsample for plotting (every 24 hours)
    step = 24
    time_idx = np.arange(0, len(timestamps), step)
    time_labels = [timestamps[i] for i in time_idx]

    for i in range(4):
        ax_time.plot(time_labels, temporal_factors[time_idx, i],
                     label=f'Factor {i+1}', linewidth=1.5, alpha=0.8)

    ax_time.set_xlabel('Time')
    ax_time.set_ylabel('Factor Loading')
    ax_time.set_title('Temporal Factor Evolution (Daily Sampling)')
    ax_time.legend(loc='upper right', ncol=4)
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_time.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax_time.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_time.grid(True, alpha=0.3)

    # Asset factors (bottom left)
    ax_asset = fig.add_subplot(gs[1, 0])

    x = np.arange(len(ASSET_NAMES))
    width = 0.2

    for i in range(4):
        ax_asset.bar(x + i * width, asset_factors[:, i], width,
                     label=f'Factor {i+1}', alpha=0.8)

    ax_asset.set_xlabel('Asset')
    ax_asset.set_ylabel('Factor Loading')
    ax_asset.set_title('Asset Factor Loadings')
    ax_asset.set_xticks(x + 1.5 * width)
    ax_asset.set_xticklabels(ASSET_NAMES)
    ax_asset.legend(loc='upper right', ncol=2)
    ax_asset.grid(axis='y', alpha=0.3)

    # Feature factors (bottom right)
    ax_feat = fig.add_subplot(gs[1, 1])

    x = np.arange(len(FEATURE_NAMES))

    for i in range(4):
        ax_feat.bar(x + i * width, feature_factors[:, i], width,
                    label=f'Factor {i+1}', alpha=0.8)

    ax_feat.set_xlabel('Feature')
    ax_feat.set_ylabel('Factor Loading')
    ax_feat.set_title('Feature Factor Loadings')
    ax_feat.set_xticks(x + 1.5 * width)
    ax_feat.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right')
    ax_feat.legend(loc='upper right', ncol=2)
    ax_feat.grid(axis='y', alpha=0.3)

    # Heatmap of asset-factor loadings (bottom row left)
    ax_heat_asset = fig.add_subplot(gs[2, 0])

    sns.heatmap(asset_factors, annot=True, fmt='.2f', cmap='RdBu_r',
                xticklabels=[f'F{i+1}' for i in range(4)],
                yticklabels=ASSET_NAMES,
                ax=ax_heat_asset, center=0, cbar_kws={'shrink': 0.8})
    ax_heat_asset.set_title('Asset Factor Matrix')
    ax_heat_asset.set_xlabel('Factor')
    ax_heat_asset.set_ylabel('Asset')

    # Heatmap of feature-factor loadings (bottom row right)
    ax_heat_feat = fig.add_subplot(gs[2, 1])

    sns.heatmap(feature_factors, annot=True, fmt='.2f', cmap='RdBu_r',
                xticklabels=[f'F{i+1}' for i in range(4)],
                yticklabels=FEATURE_NAMES,
                ax=ax_heat_feat, center=0, cbar_kws={'shrink': 0.8})
    ax_heat_feat.set_title('Feature Factor Matrix')
    ax_heat_feat.set_xlabel('Factor')
    ax_heat_feat.set_ylabel('Feature')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def plot_temporal_evolution(factors, timestamps, save_path):
    """Create detailed temporal factor evolution plot."""
    temporal_factors = factors[0]  # (8761, 4)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    factor_names = [
        'Factor 1: Market Regime / Systematic',
        'Factor 2: Altcoin Rotation / Risk Appetite',
        'Factor 3: Volatility Clustering',
        'Factor 4: Microstructure / Mean Reversion'
    ]

    colors = ['#2563EB', '#8B5CF6', '#F59E0B', '#10B981']

    # Convert timestamps to numpy array for easier slicing
    ts_array = np.array(timestamps)

    for i, (ax, name, color) in enumerate(zip(axes, factor_names, colors)):
        # Rolling mean for smoother visualization
        window = 168  # 1 week
        smoothed = np.convolve(temporal_factors[:, i], np.ones(window)/window, mode='same')

        ax.fill_between(ts_array, temporal_factors[:, i], alpha=0.3, color=color)
        ax.plot(ts_array, smoothed, color=color, linewidth=2, label='7-day MA')

        ax.set_ylabel('Loading')
        ax.set_title(name, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Time')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def plot_asset_factor_space(factors, save_path):
    """Create asset positioning in factor space."""
    asset_factors = factors[1]  # (3, 4)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    asset_colors = [COLORS['btc'], COLORS['eth'], COLORS['sol']]

    # Factor 1 vs Factor 2
    ax = axes[0]
    for i, (name, color) in enumerate(zip(ASSET_NAMES, asset_colors)):
        ax.scatter(asset_factors[i, 0], asset_factors[i, 1],
                   s=300, c=color, label=name, edgecolor='black', linewidth=2)
        ax.annotate(name, (asset_factors[i, 0], asset_factors[i, 1]),
                    xytext=(10, 10), textcoords='offset points', fontsize=12, fontweight='bold')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Factor 1 (Systematic)')
    ax.set_ylabel('Factor 2 (Rotation)')
    ax.set_title('Asset Positioning: F1 vs F2')
    ax.grid(True, alpha=0.3)

    # Factor 3 vs Factor 4
    ax = axes[1]
    for i, (name, color) in enumerate(zip(ASSET_NAMES, asset_colors)):
        ax.scatter(asset_factors[i, 2], asset_factors[i, 3],
                   s=300, c=color, label=name, edgecolor='black', linewidth=2)
        ax.annotate(name, (asset_factors[i, 2], asset_factors[i, 3]),
                    xytext=(10, 10), textcoords='offset points', fontsize=12, fontweight='bold')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Factor 3 (Volatility)')
    ax.set_ylabel('Factor 4 (Microstructure)')
    ax.set_title('Asset Positioning: F3 vs F4')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def plot_rank_selection(tensor, save_path):
    """Create rank selection analysis plot."""
    ranks = [1, 2, 3, 4, 5, 6]
    cp_variances = []
    pca_variances = []

    print("\n=== Rank Selection Analysis ===")

    T, A, F = tensor.shape
    matrix = tensor.reshape(T, A * F)

    for rank in ranks:
        # CP
        try:
            weights, factors = parafac(tensor, rank=rank, init='random', random_state=42)
            cp_recon = tl.cp_to_tensor((weights, factors))
            cp_error = np.linalg.norm(tensor - cp_recon) / np.linalg.norm(tensor)
            cp_var = 1 - cp_error**2
        except:
            cp_var = np.nan
        cp_variances.append(cp_var)

        # PCA
        pca = PCA(n_components=rank)
        pca.fit(matrix)
        pca_recon = pca.inverse_transform(pca.transform(matrix))
        pca_error = np.linalg.norm(matrix - pca_recon) / np.linalg.norm(matrix)
        pca_var = 1 - pca_error**2
        pca_variances.append(pca_var)

        print(f"  Rank {rank}: CP={cp_var:.4f}, PCA={pca_var:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ranks, cp_variances, 'o-', color=COLORS['cp'], linewidth=2,
            markersize=10, label='CP Decomposition', markeredgecolor='black')
    ax.plot(ranks, pca_variances, 's--', color=COLORS['pca'], linewidth=2,
            markersize=10, label='PCA Baseline', markeredgecolor='black')

    ax.axvline(x=4, color='gray', linestyle=':', alpha=0.7, label='Selected Rank (4)')

    ax.set_xlabel('Rank')
    ax.set_ylabel('Explained Variance')
    ax.set_title('Rank Selection: Explained Variance vs Rank')
    ax.set_xticks(ranks)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Generating Publication Figures for Tensor-DeFi Paper")
    print("=" * 60)

    # Load data
    tensor, timestamps = load_tensor()

    # Run decompositions at rank 4
    results = run_decompositions(tensor, rank=4)

    # Generate figures
    print("\n=== Generating Figures ===")

    # 1. Method comparison
    plot_method_comparison(results, OUTPUT_DIR / "method_comparison.png")

    # 2. CP factors comprehensive
    plot_cp_factors(
        results['cp']['weights'],
        results['cp']['factors'],
        timestamps,
        OUTPUT_DIR / "cp_factors.png"
    )

    # 3. Temporal evolution (detailed)
    plot_temporal_evolution(
        results['cp']['factors'],
        timestamps,
        OUTPUT_DIR / "temporal_evolution.png"
    )

    # 4. Asset factor space
    plot_asset_factor_space(
        results['cp']['factors'],
        OUTPUT_DIR / "asset_factor_space.png"
    )

    # 5. Rank selection
    plot_rank_selection(tensor, OUTPUT_DIR / "rank_selection.png")

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    # Print summary stats for paper
    print("\n=== Summary Statistics for Paper ===")
    print(f"CP Explained Variance: {results['cp']['variance']:.2%}")
    print(f"Tucker Explained Variance: {results['tucker']['variance']:.2%}")
    print(f"PCA Explained Variance: {results['pca']['variance']:.2%}")
    print(f"CP Improvement over PCA: +{(results['cp']['variance'] - results['pca']['variance'])*100:.2f}pp")


if __name__ == "__main__":
    main()
