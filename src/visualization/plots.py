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


def main():
    """Generate all publication figures."""
    base_path = Path(__file__).parent.parent.parent
    generator = FigureGenerator(
        output_dir=base_path / "figures",
        data_dir=base_path / "outputs"
    )
    generator.generate_all()


if __name__ == "__main__":
    main()
