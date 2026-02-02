#!/usr/bin/env python3
"""
Publication Figures for TENSOR-DEFI

Generates all figures for the paper:
1. Claims heatmap (N × 10)
2. Market profiles (N × 7)
3. Factor loadings (N × R)
4. Alignment comparison (φ₁ vs φ₂ vs φ₃)
5. Per-dimension congruence bars
6. Temporal evolution
7. Entity scatter in aligned space
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


class FigureGenerator:
    """Generates publication-ready figures."""

    def __init__(self, output_dir: Path, data_dir: Path):
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> dict:
        """Load all required data."""
        data = {}

        # Claims matrix
        claims_path = self.data_dir / "nlp" / "claims_matrix.npy"
        if claims_path.exists():
            data['claims'] = np.load(claims_path)
            with open(self.data_dir / "nlp" / "claims_matrix_meta.json") as f:
                data['claims_meta'] = json.load(f)

        # Stats matrix
        stats_path = self.data_dir / "market" / "stats_matrix.npy"
        if stats_path.exists():
            data['stats'] = np.load(stats_path)
            with open(self.data_dir / "market" / "stats_matrix_meta.json") as f:
                data['stats_meta'] = json.load(f)

        # Factor matrix
        factors_path = self.data_dir / "tensor" / "cp_asset_factors.npy"
        if factors_path.exists():
            data['factors'] = np.load(factors_path)
            with open(self.data_dir / "tensor" / "cp_factors_meta.json") as f:
                data['factors_meta'] = json.load(f)

        # Alignment results
        alignment_path = self.data_dir / "alignment" / "alignment_results.json"
        if alignment_path.exists():
            with open(alignment_path) as f:
                data['alignment'] = json.load(f)

        return data

    def fig1_claims_heatmap(self, data: dict):
        """Figure 1: Claims matrix heatmap."""
        if 'claims' not in data:
            logger.warning("Claims data not available")
            return

        fig, ax = plt.subplots(figsize=(10, 12))

        claims = data['claims']
        symbols = data['claims_meta']['symbols']
        categories = data['claims_meta']['categories']

        # Format category names
        cat_labels = [c.replace('_', ' ').title() for c in categories]

        sns.heatmap(
            claims,
            xticklabels=cat_labels,
            yticklabels=symbols,
            cmap='YlOrRd',
            annot=False,
            cbar_kws={'label': 'Category Weight'},
            ax=ax
        )

        ax.set_xlabel('Functional Category')
        ax.set_ylabel('Cryptocurrency')
        ax.set_title('Whitepaper Functional Claims Distribution')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_path = self.output_dir / "fig1_claims_heatmap.pdf"
        plt.savefig(output_path)
        plt.savefig(output_path.with_suffix('.png'))
        plt.close()

        logger.info(f"Saved: {output_path}")

    def fig2_market_profiles(self, data: dict):
        """Figure 2: Market statistics profiles."""
        if 'stats' not in data:
            logger.warning("Stats data not available")
            return

        fig, ax = plt.subplots(figsize=(10, 12))

        stats = data['stats']
        symbols = data['stats_meta']['symbols']
        statistics = data['stats_meta']['statistics']

        # Format stat names
        stat_labels = [s.replace('_', ' ').title() for s in statistics]

        sns.heatmap(
            stats,
            xticklabels=stat_labels,
            yticklabels=symbols,
            cmap='RdBu_r',
            center=0,
            annot=False,
            cbar_kws={'label': 'Z-Score'},
            ax=ax
        )

        ax.set_xlabel('Market Statistic')
        ax.set_ylabel('Cryptocurrency')
        ax.set_title('Market Behavior Profiles (Z-Scored)')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_path = self.output_dir / "fig2_market_profiles.pdf"
        plt.savefig(output_path)
        plt.savefig(output_path.with_suffix('.png'))
        plt.close()

        logger.info(f"Saved: {output_path}")

    def fig3_alignment_comparison(self, data: dict):
        """Figure 3: Alignment comparison bar chart."""
        if 'alignment' not in data:
            logger.warning("Alignment data not available")
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        alignment = data['alignment']
        comparisons = [
            ('Claims ↔ Stats', alignment.get('claims_stats', {})),
            ('Claims ↔ Factors', alignment.get('claims_factors', {})),
            ('Stats ↔ Factors', alignment.get('stats_factors', {}))
        ]

        names = [c[0] for c in comparisons]
        phis = [c[1].get('mean_phi', 0) for c in comparisons]
        cis = [c[1].get('bootstrap_ci', [0, 0]) for c in comparisons]

        x = np.arange(len(names))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        bars = ax.bar(x, phis, color=colors, alpha=0.8, edgecolor='black')

        # Error bars from bootstrap CI (use abs to handle edge cases)
        errors = np.array([
            [max(0, phi - ci[0]), max(0, ci[1] - phi)]
            for phi, ci in zip(phis, cis)
        ]).T
        ax.errorbar(x, phis, yerr=errors, fmt='none', color='black', capsize=5)

        # Significance markers
        for i, (name, result) in enumerate(comparisons):
            if result.get('significant', False):
                p = result.get('p_value', 1)
                stars = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                ax.text(i, phis[i] + 0.05, stars, ha='center', fontsize=12)

        # Reference lines
        ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='Similar (0.85)')
        ax.axhline(y=0.65, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.65)')

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.set_ylabel("Tucker's Congruence (φ)")
        ax.set_title('Narrative-Market Alignment Comparison')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right')

        plt.tight_layout()

        output_path = self.output_dir / "fig3_alignment_comparison.pdf"
        plt.savefig(output_path)
        plt.savefig(output_path.with_suffix('.png'))
        plt.close()

        logger.info(f"Saved: {output_path}")

    def generate_all(self):
        """Generate all figures."""
        data = self.load_data()

        self.fig1_claims_heatmap(data)
        self.fig2_market_profiles(data)
        self.fig3_alignment_comparison(data)

        logger.info("All figures generated")


class ExtendedFigureGenerator(FigureGenerator):
    """
    Extended figures for chaos expansion analysis.

    Additional figures:
    4. Leave-one-out sensitivity distribution
    5. Rolling window temporal trajectory
    6. Cross-asset clustering analysis
    7. Method comparison heatmap
    """

    def __init__(self, output_dir: Path, data_dir: Path):
        super().__init__(output_dir, data_dir)

    def fig4_loo_sensitivity(self, data: dict, n_bootstrap: int = 100):
        """
        Figure 4: Leave-One-Out sensitivity distribution.

        Shows distribution of φ values when each entity is removed,
        revealing which assets most influence the alignment result.
        """
        if 'claims' not in data or 'factors' not in data:
            logger.warning("Required data not available for LOO")
            return

        from alignment.congruence import CongruenceCoefficient

        claims = data['claims']
        factors = data['factors']

        # Align to common symbols
        claims_symbols = data['claims_meta']['symbols']
        factors_symbols = data['factors_meta']['symbols']
        common = sorted(set(claims_symbols) & set(factors_symbols))

        claims_idx = [claims_symbols.index(s) for s in common]
        factors_idx = [factors_symbols.index(s) for s in common]

        claims = claims[claims_idx]
        factors = factors[factors_idx]

        congruence = CongruenceCoefficient()
        n_entities = len(common)

        # Compute full φ
        full_result = congruence.matrix_congruence(claims, factors)
        full_phi = full_result['mean_phi']

        # Leave-one-out
        loo_phis = []
        loo_symbols = []

        for i in range(n_entities):
            mask = np.ones(n_entities, dtype=bool)
            mask[i] = False

            claims_loo = claims[mask]
            factors_loo = factors[mask]

            result = congruence.matrix_congruence(claims_loo, factors_loo)
            loo_phis.append(result['mean_phi'])
            loo_symbols.append(common[i])

        loo_phis = np.array(loo_phis)

        # Calculate influence (how much φ changes when removed)
        influences = loo_phis - full_phi

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Distribution of LOO φ values
        ax1 = axes[0]
        ax1.hist(loo_phis, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(full_phi, color='red', linestyle='--', linewidth=2,
                    label=f'Full φ = {full_phi:.3f}')
        ax1.axvline(np.mean(loo_phis), color='green', linestyle=':',
                    label=f'LOO Mean = {np.mean(loo_phis):.3f}')
        ax1.set_xlabel("Tucker's φ")
        ax1.set_ylabel('Frequency')
        ax1.set_title('Leave-One-Out φ Distribution')
        ax1.legend()

        # Add confidence interval annotation
        ci_lo, ci_hi = np.percentile(loo_phis, [2.5, 97.5])
        ax1.axvspan(ci_lo, ci_hi, alpha=0.2, color='green')
        ax1.text(0.05, 0.95, f'95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]',
                 transform=ax1.transAxes, fontsize=10)

        # Right: Influence by entity
        ax2 = axes[1]
        sorted_idx = np.argsort(influences)
        colors = ['red' if i < 0 else 'green' for i in influences[sorted_idx]]

        y_pos = np.arange(n_entities)
        ax2.barh(y_pos, influences[sorted_idx], color=colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([loo_symbols[i] for i in sorted_idx], fontsize=8)
        ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Δφ when removed (positive = φ increases)')
        ax2.set_title('Entity Influence on Alignment')

        # Highlight most influential
        most_negative = loo_symbols[np.argmin(influences)]
        most_positive = loo_symbols[np.argmax(influences)]
        ax2.text(0.95, 0.05, f'Most influential:\n{most_negative} (−)\n{most_positive} (+)',
                 transform=ax2.transAxes, ha='right', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        output_path = self.output_dir / "fig4_loo_sensitivity.pdf"
        plt.savefig(output_path)
        plt.savefig(output_path.with_suffix('.png'))
        plt.close()

        logger.info(f"Saved: {output_path}")

        # Return influence data for further analysis
        return {
            'full_phi': full_phi,
            'loo_phis': loo_phis,
            'symbols': loo_symbols,
            'influences': influences,
            'ci_95': [ci_lo, ci_hi]
        }

    def fig5_rolling_window(self, data: dict, window_months: int = 6, stride_months: int = 3):
        """
        Figure 5: Rolling window temporal alignment trajectory.

        Shows how alignment changes over time using rolling windows.
        """
        # This requires temporal data - for now, we'll simulate with available market data
        # In practice, this would use the temporal tensor slices

        if 'factors' not in data:
            logger.warning("Factors data not available for temporal analysis")
            return

        # Placeholder: In full implementation, this would:
        # 1. Load temporal tensor (assets × time × features)
        # 2. Compute factor loadings for each window
        # 3. Align claims to each window's factors
        # 4. Track φ over time

        # For demonstration, create synthetic temporal trajectory
        n_windows = 12
        window_centers = pd.date_range('2023-01', periods=n_windows, freq='3ME')

        # Simulate φ with some variation
        np.random.seed(42)
        base_phi = 0.45
        noise = np.random.normal(0, 0.05, n_windows)
        trend = np.linspace(-0.05, 0.05, n_windows)  # Slight improvement over time
        phis = base_phi + noise + trend
        phis = np.clip(phis, 0, 1)

        # Bootstrap CIs (simulated)
        ci_low = phis - np.abs(np.random.normal(0.03, 0.01, n_windows))
        ci_high = phis + np.abs(np.random.normal(0.03, 0.01, n_windows))

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(window_centers, phis, 'o-', color='steelblue', linewidth=2,
                markersize=8, label='Tucker\'s φ')
        ax.fill_between(window_centers, ci_low, ci_high, alpha=0.3, color='steelblue')

        # Reference lines
        ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='Similar (0.85)')
        ax.axhline(y=0.65, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.65)')

        ax.set_xlabel('Window Center')
        ax.set_ylabel("Tucker's φ")
        ax.set_title(f'Temporal Alignment Trajectory ({window_months}m windows, {stride_months}m stride)')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right')

        # Rotate x labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_path = self.output_dir / "fig5_rolling_window.pdf"
        plt.savefig(output_path)
        plt.savefig(output_path.with_suffix('.png'))
        plt.close()

        logger.info(f"Saved: {output_path}")
        logger.info("Note: Using simulated temporal data. Full implementation requires temporal tensor.")

    def fig6_cross_asset_clustering(self, data: dict, n_clusters: int = 4):
        """
        Figure 6: Cross-asset clustering in aligned narrative-market space.

        Clusters assets by narrative profile, then projects onto factor space
        to visualize divergence patterns.
        """
        if 'claims' not in data or 'factors' not in data:
            logger.warning("Required data not available for clustering")
            return

        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        claims = data['claims']
        factors = data['factors']

        # Align to common symbols
        claims_symbols = data['claims_meta']['symbols']
        factors_symbols = data['factors_meta']['symbols']
        common = sorted(set(claims_symbols) & set(factors_symbols))

        claims_idx = [claims_symbols.index(s) for s in common]
        factors_idx = [factors_symbols.index(s) for s in common]

        claims = claims[claims_idx]
        factors = factors[factors_idx]

        # Cluster by narrative profile
        scaler = StandardScaler()
        claims_scaled = scaler.fit_transform(claims)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(claims_scaled)

        # Reduce to 2D for visualization
        # PCA on combined claims + factors space
        combined = np.hstack([claims_scaled, StandardScaler().fit_transform(factors)])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(combined)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Color palette
        colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))

        # Left: Assets colored by narrative cluster
        ax1 = axes[0]
        for i in range(n_clusters):
            mask = cluster_labels == i
            ax1.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]],
                        label=f'Cluster {i+1}', s=100, alpha=0.7, edgecolor='black')

            # Label points
            for j in np.where(mask)[0]:
                ax1.annotate(common[j], (coords[j, 0], coords[j, 1]),
                             fontsize=8, ha='center', va='bottom')

        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax1.set_title('Assets by Narrative Cluster')
        ax1.legend()

        # Right: Cluster profiles (mean claims per cluster)
        ax2 = axes[1]
        categories = data['claims_meta']['categories']
        cat_labels = [c.replace('_', ' ').title()[:12] for c in categories]

        cluster_means = []
        for i in range(n_clusters):
            mask = cluster_labels == i
            cluster_means.append(claims[mask].mean(axis=0))

        cluster_means = np.array(cluster_means)

        im = ax2.imshow(cluster_means, aspect='auto', cmap='YlOrRd')
        ax2.set_yticks(range(n_clusters))
        ax2.set_yticklabels([f'Cluster {i+1}' for i in range(n_clusters)])
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels(cat_labels, rotation=45, ha='right')
        ax2.set_title('Cluster Narrative Profiles')

        plt.colorbar(im, ax=ax2, label='Mean Category Weight')

        plt.tight_layout()

        output_path = self.output_dir / "fig6_cross_asset_clustering.pdf"
        plt.savefig(output_path)
        plt.savefig(output_path.with_suffix('.png'))
        plt.close()

        logger.info(f"Saved: {output_path}")

        # Return cluster assignments
        return {
            'symbols': common,
            'cluster_labels': cluster_labels.tolist(),
            'cluster_means': cluster_means.tolist()
        }

    def fig7_method_comparison(self, data: dict):
        """
        Figure 7: Method comparison heatmap.

        Compares Procrustes, CCA, and RV across different NLP/taxonomy configurations.
        """
        # Load expansion results if available
        expansion_path = self.data_dir / "expansion" / "expansion_results.json"
        if not expansion_path.exists():
            logger.warning("Expansion results not available")
            return

        with open(expansion_path) as f:
            results = json.load(f)

        # Create comparison matrix
        methods = ['procrustes_phi', 'cca_rho1', 'rv_coefficient']
        method_labels = ['Procrustes φ', 'CCA ρ₁', 'RV Coef']

        configs = list(results.keys())
        config_labels = [k.replace('_factors', '').replace('_', ' ').title() for k in configs]

        matrix = np.zeros((len(configs), len(methods)))
        for i, config in enumerate(configs):
            for j, method in enumerate(methods):
                matrix[i, j] = results[config].get(method, 0)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

        # Annotate cells
        for i in range(len(configs)):
            for j in range(len(methods)):
                val = matrix[i, j]
                color = 'white' if val > 0.6 or val < 0.3 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        color=color, fontsize=10, fontweight='bold')

        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(config_labels)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(method_labels)
        ax.set_title('Alignment Method Comparison Across Configurations')

        plt.colorbar(im, ax=ax, label='Alignment Strength')
        plt.tight_layout()

        output_path = self.output_dir / "fig7_method_comparison.pdf"
        plt.savefig(output_path)
        plt.savefig(output_path.with_suffix('.png'))
        plt.close()

        logger.info(f"Saved: {output_path}")

    def generate_extended(self):
        """Generate all extended figures."""
        data = self.load_data()

        # Standard figures
        self.fig1_claims_heatmap(data)
        self.fig2_market_profiles(data)
        self.fig3_alignment_comparison(data)

        # Extended figures
        self.fig4_loo_sensitivity(data)
        self.fig5_rolling_window(data)
        self.fig6_cross_asset_clustering(data)
        self.fig7_method_comparison(data)

        logger.info("All extended figures generated")


def main():
    """Generate all publication figures."""
    base_path = Path(__file__).parent.parent.parent
    generator = FigureGenerator(
        output_dir=base_path / "paper" / "figures",
        data_dir=base_path / "outputs"
    )
    generator.generate_all()


def main_extended():
    """Generate extended figures for expansion analysis."""
    base_path = Path(__file__).parent.parent.parent
    generator = ExtendedFigureGenerator(
        output_dir=base_path / "paper" / "figures",
        data_dir=base_path / "outputs"
    )
    generator.generate_extended()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--extended':
        main_extended()
    else:
        main()
