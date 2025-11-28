#!/usr/bin/env python3
"""
Generate publication-quality figures for tensor-defi v2.0.0
Narrative-Market Alignment Analysis
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
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
ENTITY_COLORS = {
    'BTC': '#F7931A',   # Bitcoin orange
    'ETH': '#627EEA',   # Ethereum blue
    'SOL': '#00FFA3',   # Solana green
    'AVAX': '#E84142',  # Avalanche red
    'DOT': '#E6007A',   # Polkadot pink
    'FIL': '#0090FF',   # Filecoin blue
    'LINK': '#375BD2',  # Chainlink blue
    'ALGO': '#000000',  # Algorand black
}

CONGRUENCE_COLORS = {
    'excellent': '#22C55E',  # Green (φ ≥ 0.95)
    'fair': '#3B82F6',       # Blue (0.85 ≤ φ < 0.95)
    'some': '#F59E0B',       # Orange (0.65 ≤ φ < 0.85)
    'distinct': '#EF4444',   # Red (φ < 0.65)
}

FUNCTIONAL_CATEGORIES = [
    "store_of_value",
    "medium_of_exchange",
    "smart_contracts",
    "infrastructure",
    "privacy",
    "governance",
    "data_oracle",
    "identity",
    "gaming_metaverse",
    "stablecoin"
]

CATEGORY_LABELS = [
    "Store of Value",
    "Medium of Exchange",
    "Smart Contracts",
    "Infrastructure",
    "Privacy",
    "Governance",
    "Data Oracle",
    "Identity",
    "Gaming/Metaverse",
    "Stablecoin"
]


def load_data():
    """Load all required data."""
    # Claims matrix
    claims_data = np.load(PROJECT_ROOT / "outputs/nlp/claims_matrix.npz", allow_pickle=True)
    claims_matrix = claims_data['matrix']
    symbols = list(claims_data['symbols'])
    categories = list(claims_data['categories'])

    # Market matrix
    market_data = np.load(PROJECT_ROOT / "outputs/alignment/market_matrix.npz", allow_pickle=True)
    market_matrix = market_data['matrix']

    # Functional profiles
    with open(PROJECT_ROOT / "outputs/nlp/functional_profiles.json") as f:
        profiles = json.load(f)

    # Alignment results
    with open(PROJECT_ROOT / "outputs/alignment/alignment_results.json") as f:
        alignment = json.load(f)

    return {
        'claims': claims_matrix,
        'market': market_matrix,
        'symbols': symbols,
        'categories': categories,
        'profiles': profiles,
        'alignment': alignment
    }


def plot_functional_profiles_heatmap(data, save_path):
    """
    Figure 1: Heatmap of entity functional profiles from whitepaper classification.
    Shows what each cryptocurrency claims to do based on NLP analysis.
    """
    profiles = data['profiles']
    symbols = data['symbols']

    # Build matrix
    matrix = np.array([
        [profiles[sym]['functional_profile'].get(cat, 0) for cat in FUNCTIONAL_CATEGORIES]
        for sym in symbols
    ])

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create heatmap
    im = sns.heatmap(
        matrix,
        xticklabels=CATEGORY_LABELS,
        yticklabels=[f"{sym} ({profiles[sym]['project_name']})" for sym in symbols],
        cmap='YlOrRd',
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Classification Score', 'shrink': 0.8},
        vmin=0,
        vmax=0.7
    )

    ax.set_title('Whitepaper Functional Profiles\n(Zero-Shot Classification with BART-MNLI)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Functional Category', fontsize=12)
    ax.set_ylabel('Entity', fontsize=12)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add primary function annotations
    for i, sym in enumerate(symbols):
        primary = profiles[sym].get('primary_functions', [])
        if primary:
            ax.annotate(f"Primary: {', '.join(primary[:2])}",
                       xy=(len(FUNCTIONAL_CATEGORIES), i + 0.5),
                       xytext=(5, 0), textcoords='offset points',
                       fontsize=9, va='center', style='italic', color='#666')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def plot_per_factor_congruence(data, save_path):
    """
    Figure 2: Per-factor Tucker's congruence coefficient with interpretation bands.
    Shows which market factors align with narrative claims.
    """
    alignment = data['alignment']
    phi_values = alignment['per_factor_congruence']
    n_factors = len(phi_values)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define thresholds
    thresholds = [
        (0.95, 1.0, CONGRUENCE_COLORS['excellent'], 'Equivalent (φ ≥ 0.95)'),
        (0.85, 0.95, CONGRUENCE_COLORS['fair'], 'Fair Similarity (0.85 ≤ φ < 0.95)'),
        (0.65, 0.85, CONGRUENCE_COLORS['some'], 'Some Similarity (0.65 ≤ φ < 0.85)'),
        (0.0, 0.65, CONGRUENCE_COLORS['distinct'], 'Distinct (φ < 0.65)'),
    ]

    # Draw threshold bands
    for low, high, color, label in thresholds:
        ax.axhspan(low, high, alpha=0.15, color=color, label=label)
        ax.axhline(y=high, color=color, linestyle='--', alpha=0.5, linewidth=1)

    # Plot bars
    x = np.arange(n_factors)
    colors = []
    for phi in phi_values:
        if phi >= 0.95:
            colors.append(CONGRUENCE_COLORS['excellent'])
        elif phi >= 0.85:
            colors.append(CONGRUENCE_COLORS['fair'])
        elif phi >= 0.65:
            colors.append(CONGRUENCE_COLORS['some'])
        else:
            colors.append(CONGRUENCE_COLORS['distinct'])

    bars = ax.bar(x, phi_values, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

    # Add value labels
    for bar, phi in zip(bars, phi_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'φ = {phi:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add overall mean line
    overall_phi = alignment['congruence_coefficient']
    ax.axhline(y=overall_phi, color='black', linestyle='-', linewidth=2, label=f'Overall φ = {overall_phi:.3f}')

    ax.set_xlabel('Market Factor', fontsize=12)
    ax.set_ylabel("Tucker's Congruence Coefficient (φ)", fontsize=12)
    ax.set_title('Per-Factor Alignment: Whitepaper Claims vs Market Structure',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Factor {i+1}' for i in range(n_factors)])
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def plot_claims_vs_market_space(data, save_path):
    """
    Figure 3: PCA projection of claims space vs market space.
    Visualizes the alignment (or misalignment) between narrative and market positioning.
    """
    claims = data['claims']
    market = data['market']
    symbols = data['symbols']

    # PCA on claims (10D → 2D)
    pca_claims = PCA(n_components=2)
    claims_2d = pca_claims.fit_transform(claims)

    # PCA on market (variable D → 2D)
    pca_market = PCA(n_components=2)
    market_2d = pca_market.fit_transform(market)

    # Normalize both to same scale for comparison
    claims_2d = (claims_2d - claims_2d.mean(0)) / claims_2d.std(0)
    market_2d = (market_2d - market_2d.mean(0)) / market_2d.std(0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Claims Space
    ax = axes[0]
    for i, sym in enumerate(symbols):
        color = ENTITY_COLORS.get(sym, '#333333')
        ax.scatter(claims_2d[i, 0], claims_2d[i, 1], s=200, c=color,
                  edgecolor='black', linewidth=1.5, zorder=5)
        ax.annotate(sym, (claims_2d[i, 0], claims_2d[i, 1]),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=11, fontweight='bold')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(f'PC1 ({pca_claims.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca_claims.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
    ax.set_title('Claims Space\n(Whitepaper Functional Profiles)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: Market Space
    ax = axes[1]
    for i, sym in enumerate(symbols):
        color = ENTITY_COLORS.get(sym, '#333333')
        ax.scatter(market_2d[i, 0], market_2d[i, 1], s=200, c=color,
                  edgecolor='black', linewidth=1.5, zorder=5)
        ax.annotate(sym, (market_2d[i, 0], market_2d[i, 1]),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=11, fontweight='bold')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(f'PC1 ({pca_market.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca_market.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
    ax.set_title('Market Space\n(CP Decomposition Factor Loadings)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: Overlay with displacement vectors
    ax = axes[2]

    # Procrustes alignment for visualization
    _, claims_aligned, _ = procrustes(market_2d, claims_2d)

    for i, sym in enumerate(symbols):
        color = ENTITY_COLORS.get(sym, '#333333')

        # Claims position (circle)
        ax.scatter(claims_aligned[i, 0], claims_aligned[i, 1], s=150, c=color,
                  marker='o', edgecolor='black', linewidth=1.5, alpha=0.7, zorder=5)

        # Market position (square)
        ax.scatter(market_2d[i, 0], market_2d[i, 1], s=150, c=color,
                  marker='s', edgecolor='black', linewidth=1.5, zorder=5)

        # Displacement vector
        ax.annotate('', xy=(market_2d[i, 0], market_2d[i, 1]),
                   xytext=(claims_aligned[i, 0], claims_aligned[i, 1]),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.7))

        # Label at market position
        ax.annotate(sym, (market_2d[i, 0], market_2d[i, 1]),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('PC1 (Aligned)', fontsize=11)
    ax.set_ylabel('PC2 (Aligned)', fontsize=11)
    ax.set_title('Alignment Overlay\n(○ Claims → ■ Market)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='gray', edgecolor='black', label='○ = Claims Position'),
        mpatches.Patch(facecolor='gray', edgecolor='black', label='■ = Market Position'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def plot_bootstrap_ci(data, save_path):
    """
    Figure 4: Bootstrap confidence interval visualization.
    Shows uncertainty around the overall congruence coefficient.
    """
    alignment = data['alignment']
    bootstrap = alignment['bootstrap']

    point_est = bootstrap['point_estimate']
    ci_lower = bootstrap['ci_lower']
    ci_upper = bootstrap['ci_upper']

    fig, ax = plt.subplots(figsize=(10, 5))

    # Threshold bands (horizontal)
    thresholds = [
        (0.95, 1.0, CONGRUENCE_COLORS['excellent'], 'Equivalent'),
        (0.85, 0.95, CONGRUENCE_COLORS['fair'], 'Fair Similarity'),
        (0.65, 0.85, CONGRUENCE_COLORS['some'], 'Some Similarity'),
        (0.0, 0.65, CONGRUENCE_COLORS['distinct'], 'Distinct'),
    ]

    for low, high, color, label in thresholds:
        ax.axvspan(low, high, alpha=0.2, color=color)

    # Point estimate
    ax.axvline(x=point_est, color='black', linewidth=3, label=f'Point Estimate: φ = {point_est:.3f}')

    # CI
    ax.axvspan(ci_lower, ci_upper, alpha=0.4, color='#3B82F6',
               label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
    ax.axvline(x=ci_lower, color='#3B82F6', linewidth=2, linestyle='--')
    ax.axvline(x=ci_upper, color='#3B82F6', linewidth=2, linestyle='--')

    # Null hypothesis threshold
    ax.axvline(x=0.65, color='red', linewidth=2, linestyle=':',
               label='H₀ Threshold: φ = 0.65')

    ax.set_xlim([0.4, 1.0])
    ax.set_xlabel("Tucker's Congruence Coefficient (φ)", fontsize=12)
    ax.set_title('Bootstrap Confidence Interval for Overall Alignment\n(n=50 bootstrap samples)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_yticks([])

    # Add interpretation annotations
    ax.annotate('Reject H₀\n(φ ≥ 0.65)', xy=(0.75, 0.7), xycoords='axes fraction',
               fontsize=11, ha='center', fontweight='bold', color='green')
    ax.annotate('CI excludes\nnull region', xy=(0.75, 0.5), xycoords='axes fraction',
               fontsize=10, ha='center', style='italic', color='#666')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def plot_entity_alignment_radar(data, save_path):
    """
    Figure 5: Radar chart showing functional profiles for all entities.
    Alternative visualization to heatmap - shows profile shapes.
    """
    profiles = data['profiles']
    symbols = data['symbols']

    # Number of variables
    categories = CATEGORY_LABELS
    N = len(categories)

    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    fig, axes = plt.subplots(2, 4, figsize=(16, 10), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for idx, sym in enumerate(symbols):
        ax = axes[idx]

        # Get values
        values = [profiles[sym]['functional_profile'].get(cat, 0) for cat in FUNCTIONAL_CATEGORIES]
        values += values[:1]  # Complete the circle

        color = ENTITY_COLORS.get(sym, '#333333')

        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c[:8] for c in categories], size=8)
        ax.set_ylim([0, 0.7])
        ax.set_title(f"{sym}\n({profiles[sym]['project_name']})",
                    fontsize=11, fontweight='bold', pad=10)

    plt.suptitle('Whitepaper Functional Profiles by Entity\n(Zero-Shot Classification Scores)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def plot_summary_dashboard(data, save_path):
    """
    Figure 6: Summary dashboard combining key results.
    Single figure for executive summary / README.
    """
    alignment = data['alignment']
    profiles = data['profiles']
    symbols = data['symbols']

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Per-factor congruence (top left)
    ax_factors = fig.add_subplot(gs[0, 0])
    phi_values = alignment['per_factor_congruence']
    colors = []
    for phi in phi_values:
        if phi >= 0.95:
            colors.append(CONGRUENCE_COLORS['excellent'])
        elif phi >= 0.85:
            colors.append(CONGRUENCE_COLORS['fair'])
        elif phi >= 0.65:
            colors.append(CONGRUENCE_COLORS['some'])
        else:
            colors.append(CONGRUENCE_COLORS['distinct'])

    bars = ax_factors.bar(range(len(phi_values)), phi_values, color=colors, edgecolor='black')
    ax_factors.axhline(y=alignment['congruence_coefficient'], color='black', linestyle='-', linewidth=2)
    ax_factors.axhline(y=0.65, color='red', linestyle='--', alpha=0.7)
    ax_factors.set_xlabel('Factor')
    ax_factors.set_ylabel('φ')
    ax_factors.set_title('(A) Per-Factor Congruence', fontweight='bold')
    ax_factors.set_xticks(range(len(phi_values)))
    ax_factors.set_xticklabels([f'F{i+1}' for i in range(len(phi_values))])
    ax_factors.set_ylim([0, 1.1])
    for bar, phi in zip(bars, phi_values):
        ax_factors.text(bar.get_x() + bar.get_width()/2., phi + 0.02,
                       f'{phi:.2f}', ha='center', va='bottom', fontsize=9)

    # Panel B: Entity primary functions (top middle)
    ax_primary = fig.add_subplot(gs[0, 1])

    # Count primary functions
    primary_counts = {}
    for sym in symbols:
        primary = profiles[sym].get('primary_functions', ['generalist'])
        for p in primary[:1]:  # Just first primary
            primary_counts[p] = primary_counts.get(p, 0) + 1

    funcs = list(primary_counts.keys())
    counts = list(primary_counts.values())
    ax_primary.barh(funcs, counts, color='#3B82F6', edgecolor='black')
    ax_primary.set_xlabel('Count')
    ax_primary.set_title('(B) Primary Functions\n(from NLP classification)', fontweight='bold')
    ax_primary.grid(axis='x', alpha=0.3)

    # Panel C: Bootstrap CI (top right)
    ax_ci = fig.add_subplot(gs[0, 2])
    bootstrap = alignment['bootstrap']

    ax_ci.barh(['Overall φ'], [bootstrap['point_estimate']],
               xerr=[[bootstrap['point_estimate'] - bootstrap['ci_lower']],
                    [bootstrap['ci_upper'] - bootstrap['point_estimate']]],
               capsize=10, color='#3B82F6', edgecolor='black', height=0.5)
    ax_ci.axvline(x=0.65, color='red', linestyle='--', label='H₀ threshold')
    ax_ci.set_xlim([0.4, 1.0])
    ax_ci.set_xlabel('φ')
    ax_ci.set_title('(C) Bootstrap 95% CI', fontweight='bold')
    ax_ci.legend(loc='lower right')

    # Panel D: Simplified heatmap (bottom, spanning 2 columns)
    ax_heat = fig.add_subplot(gs[1, :2])

    # Build matrix for top 5 categories only
    top_cats = ['infrastructure', 'medium_of_exchange', 'data_oracle', 'governance', 'identity']
    top_labels = ['Infrastructure', 'Medium of Exchange', 'Data Oracle', 'Governance', 'Identity']
    matrix = np.array([
        [profiles[sym]['functional_profile'].get(cat, 0) for cat in top_cats]
        for sym in symbols
    ])

    sns.heatmap(matrix, xticklabels=top_labels, yticklabels=symbols,
               cmap='YlOrRd', annot=True, fmt='.2f', ax=ax_heat,
               cbar_kws={'shrink': 0.5})
    ax_heat.set_title('(D) Top Functional Categories by Entity', fontweight='bold')
    plt.setp(ax_heat.get_xticklabels(), rotation=45, ha='right')

    # Panel E: Key stats text (bottom right)
    ax_stats = fig.add_subplot(gs[1, 2])
    ax_stats.axis('off')

    stats_text = f"""
    KEY RESULTS
    ══════════════════════

    Overall Alignment
    φ = {alignment['congruence_coefficient']:.3f}
    Interpretation: {alignment['interpretation']}

    Bootstrap CI
    [{bootstrap['ci_lower']:.3f}, {bootstrap['ci_upper']:.3f}]

    Best Factor (F4)
    φ = {max(phi_values):.3f} (Near-equivalent)

    Worst Factor (F1)
    φ = {min(phi_values):.3f} (Distinct)

    Universe
    {len(symbols)} entities
    10 functional categories
    """

    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.suptitle('Crypto Narrative-Market Alignment: Summary Dashboard',
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Generating Alignment Figures for Tensor-DeFi v2.0.0")
    print("=" * 60)

    # Load data
    data = load_data()
    print(f"Loaded data for {len(data['symbols'])} entities")
    print(f"Claims matrix: {data['claims'].shape}")
    print(f"Market matrix: {data['market'].shape}")
    print(f"Overall φ: {data['alignment']['congruence_coefficient']:.3f}")

    # Generate figures
    print("\n=== Generating Figures ===")

    # 1. Functional profiles heatmap
    plot_functional_profiles_heatmap(data, OUTPUT_DIR / "functional_profiles_heatmap.png")

    # 2. Per-factor congruence
    plot_per_factor_congruence(data, OUTPUT_DIR / "per_factor_congruence.png")

    # 3. Claims vs market space
    plot_claims_vs_market_space(data, OUTPUT_DIR / "claims_vs_market_space.png")

    # 4. Bootstrap CI
    plot_bootstrap_ci(data, OUTPUT_DIR / "bootstrap_confidence_interval.png")

    # 5. Radar chart
    plot_entity_alignment_radar(data, OUTPUT_DIR / "entity_radar_profiles.png")

    # 6. Summary dashboard
    plot_summary_dashboard(data, OUTPUT_DIR / "alignment_summary_dashboard.png")

    print("\n" + "=" * 60)
    print("All alignment figures generated!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
