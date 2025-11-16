#!/usr/bin/env python3
"""
Full Experimental Pipeline - Tensor Decomposition for DeFi Markets

Runs comprehensive experiments for academic paper:
1. Reconstruction quality (CP/Tucker/TT vs PCA)
2. Factor interpretation (temporal, venue, asset, feature)
3. Rank selection experiments
4. Generates all tables and figures for Section 4

Expected runtime: 10-30 minutes on consumer hardware

Memory Optimizations:
- Loads tensors on-demand instead of all at once
- Explicit garbage collection between experiments
- Deletes tensor references immediately after use
- Tracks memory usage with psutil
"""

import sys
import gc
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from tensor_ops.decomposition import TensorDecomposer, DecompositionResult
from baselines.traditional_methods import TraditionalMethods, BaselineResult
from visualization.tensor_plots import TensorVisualizer

# Output directories
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
FIGURES_DIR = OUTPUT_DIR / 'figures'
TABLES_DIR = OUTPUT_DIR / 'tables'
RESULTS_DIR = OUTPUT_DIR / 'results'

for d in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)

# Data paths
DATA_DIR = Path(__file__).parent.parent / 'data'
TENSOR_DIR = DATA_DIR / 'tensors'


class ExperimentRunner:
    """Orchestrates full experimental pipeline with memory-optimized loading."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.decomposer = TensorDecomposer(backend='numpy')
        self.baseline = TraditionalMethods()
        self.visualizer = TensorVisualizer(style='dark')

        self.results = {}
        self.process = psutil.Process()

    def log(self, msg):
        if self.verbose:
            mem_mb = self.process.memory_info().rss / (1024 * 1024)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{mem_mb:,.0f}MB] {msg}")

    def get_memory_mb(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)

    def load_tensor(self, name: str):
        """
        Load a single tensor from disk.

        Args:
            name: Tensor name ('raw_ohlcv', 'normalized_ohlcv', or 'log_returns')

        Returns:
            Tensor numpy array with singleton dimensions squeezed
        """
        path = TENSOR_DIR / f'{name}_tensor.pkl'
        with open(path, 'rb') as f:
            data = pickle.load(f)
            tensor = data['tensor']
            original_shape = tensor.shape

            # CRITICAL FIX: Squeeze singleton dimensions to avoid OOM
            # Tensorly's CP-ALS creates (N_total, N_total) Gram matrices
            # where N_total = product of all dimensions.
            # A singleton dimension doesn't add information but MASSIVELY
            # increases memory: (8761,1,3,5) creates 131k x 131k = 137GB matrix!
            # Squeezing to (8761,3,5) creates manageable 131k x 131k BUT properly structured
            tensor = np.squeeze(tensor)

            self.log(f"Loaded {name}: {original_shape} -> {tensor.shape} (squeezed)")
            return tensor

    def free_memory(self):
        """Force garbage collection and log memory freed."""
        mem_before = self.get_memory_mb()
        gc.collect()
        mem_after = self.get_memory_mb()
        freed = mem_before - mem_after
        if freed > 0:
            self.log(f"Freed {freed:.1f}MB via garbage collection")

    def experiment_1_reconstruction_quality(self):
        """
        Experiment 1: Reconstruction Quality Comparison

        Compare CP, Tucker, TT, and PCA across ranks 3, 5, 10, 15, 20
        Generate Table 2 from paper.

        Memory-optimized: Loads tensor once, deletes after use.
        """
        self.log("\n" + "="*80)
        self.log("EXPERIMENT 1: Reconstruction Quality")
        self.log("="*80)

        # Load normalized tensor on-demand
        tensor = self.load_tensor('normalized_ohlcv')
        ranks = [3, 5, 10, 15, 20]

        results = {
            'ranks': ranks,
            'pca': [],
            'cp': [],
            'tucker': [],
            'tt': []
        }

        for rank in ranks:
            self.log(f"\nRank {rank}:")

            # PCA baseline
            pca_result = self.baseline.fit_pca(tensor, n_components=rank, flatten_method='time_first')
            results['pca'].append(pca_result.explained_variance)
            self.log(f"  PCA: {pca_result.explained_variance:.4f}")
            del pca_result
            gc.collect()

            # CP decomposition
            cp_result = self.decomposer.cp_decomposition(tensor, rank=rank, verbose=False)
            results['cp'].append(cp_result.explained_variance)
            self.log(f"  CP:  {cp_result.explained_variance:.4f}")
            del cp_result
            gc.collect()

            # Tucker decomposition
            tucker_result = self.decomposer.tucker_decomposition(tensor, ranks=rank, verbose=False)
            results['tucker'].append(tucker_result.explained_variance)
            self.log(f"  Tucker: {tucker_result.explained_variance:.4f}")
            del tucker_result
            gc.collect()

            # Tensor Train
            tt_result = self.decomposer.tensor_train_decomposition(tensor, ranks=rank, verbose=False)
            results['tt'].append(tt_result.explained_variance)
            self.log(f"  TT:  {tt_result.explained_variance:.4f}")
            del tt_result
            gc.collect()

        # Delete tensor and free memory
        del tensor
        self.free_memory()

        # Save results
        self.results['reconstruction'] = results

        # Generate LaTeX table
        self._generate_reconstruction_table(results)

        # Plot comparison
        self._plot_reconstruction_comparison(results)

        return results

    def experiment_2_rank_selection(self):
        """
        Experiment 2: Automatic Rank Selection

        Find optimal rank for CP decomposition using explained variance threshold.

        Memory-optimized: Loads tensor, runs selection, immediately frees.
        """
        self.log("\n" + "="*80)
        self.log("EXPERIMENT 2: Rank Selection")
        self.log("="*80)

        # Load tensor on-demand
        tensor = self.load_tensor('normalized_ohlcv')

        # Run rank selection
        optimal_rank = self.decomposer.rank_selection(
            tensor,
            method='cp',
            max_rank=20,
            criterion='explained_variance',
            threshold=0.95
        )

        self.log(f"\n✓ Optimal rank for 95% explained variance: {optimal_rank}")

        # Free memory
        del tensor
        self.free_memory()

        self.results['optimal_rank'] = optimal_rank
        return optimal_rank

    def experiment_3_factor_analysis(self):
        """
        Experiment 3: Factor Interpretation

        Decompose at optimal rank and analyze temporal, asset, and feature factors.
        Generate figures for Section 4.2.

        Memory-optimized: Extracts factors, deletes full cp_result, keeps only factors.
        """
        self.log("\n" + "="*80)
        self.log("EXPERIMENT 3: Factor Analysis")
        self.log("="*80)

        # Load tensor on-demand
        tensor = self.load_tensor('normalized_ohlcv')
        rank = self.results.get('optimal_rank', 10)

        # CP decomposition at optimal rank
        self.log(f"\nRunning CP decomposition at rank {rank}...")
        cp_result = self.decomposer.cp_decomposition(tensor, rank=rank, verbose=True)

        # Extract factors (copy them to avoid keeping full cp_result in memory)
        temporal_factors = cp_result.factors[0].copy()  # (T, R)
        venue_factors = cp_result.factors[1].copy()     # (V, R) - will be (1, R) for single venue
        asset_factors = cp_result.factors[2].copy()     # (A, R)
        feature_factors = cp_result.factors[3].copy()   # (F, R)
        weights = cp_result.core.copy() if cp_result.core is not None else None

        self.log(f"\nFactor shapes:")
        self.log(f"  Temporal: {temporal_factors.shape}")
        self.log(f"  Venue:    {venue_factors.shape}")
        self.log(f"  Asset:    {asset_factors.shape}")
        self.log(f"  Feature:  {feature_factors.shape}")

        # Delete large objects
        del cp_result
        del tensor
        self.free_memory()

        # Save factors
        self.results['factors'] = {
            'temporal': temporal_factors,
            'venue': venue_factors,
            'asset': asset_factors,
            'feature': feature_factors,
            'weights': weights
        }

        # Visualize temporal factors
        self._plot_temporal_factors(temporal_factors)

        # Visualize asset factors
        self._plot_asset_factors(asset_factors)

        # Visualize feature factors
        self._plot_feature_factors(feature_factors)

        return None  # Don't return cp_result to avoid keeping it in memory

    def experiment_4_variance_by_tensor_type(self):
        """
        Experiment 4: Compare performance across tensor variants

        Does normalization or log-returns improve decomposition quality?

        Memory-optimized: Loads one tensor at a time, deletes immediately after.
        """
        self.log("\n" + "="*80)
        self.log("EXPERIMENT 4: Tensor Variant Comparison")
        self.log("="*80)

        rank = 10
        results = {}
        tensor_names = ['raw_ohlcv', 'normalized_ohlcv', 'log_returns']

        for name in tensor_names:
            self.log(f"\n{name}:")

            # Load tensor on-demand
            tensor = self.load_tensor(name)

            # Run decompositions
            cp_result = self.decomposer.cp_decomposition(tensor, rank=rank, verbose=False)
            tucker_result = self.decomposer.tucker_decomposition(tensor, ranks=rank, verbose=False)

            # Extract results
            results[name] = {
                'cp_variance': cp_result.explained_variance,
                'tucker_variance': tucker_result.explained_variance,
                'cp_error': cp_result.reconstruction_error,
                'tucker_error': tucker_result.reconstruction_error
            }

            self.log(f"  CP explained variance:     {cp_result.explained_variance:.4f}")
            self.log(f"  Tucker explained variance: {tucker_result.explained_variance:.4f}")

            # Delete everything for this tensor
            del tensor, cp_result, tucker_result
            self.free_memory()

        self.results['tensor_variants'] = results
        self._generate_variant_comparison_table(results)

        return results

    def experiment_5_correlation_analysis(self):
        """
        Experiment 5: Cross-asset correlation from factors

        Do asset factors capture known correlation structure?
        """
        self.log("\n" + "="*80)
        self.log("EXPERIMENT 5: Asset Correlation Structure")
        self.log("="*80)

        # Load original data to compute empirical correlations
        csv_path = DATA_DIR / 'cex_data_365d.csv'
        if not csv_path.exists():
            self.log("  Skipping - 365d data not found")
            return

        df = pd.read_csv(csv_path)
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Compute empirical correlation matrix (using close prices)
        assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        pivot = df.pivot_table(index='datetime', columns='symbol', values='close')
        empirical_corr = pivot[assets].corr()

        self.log("\nEmpirical correlation matrix:")
        self.log(empirical_corr.to_string())

        # Compute correlation from asset factors
        if 'factors' in self.results:
            asset_factors = self.results['factors']['asset']
            # Factor-based correlation: A @ A.T gives asset similarity
            factor_corr = asset_factors @ asset_factors.T
            # Normalize to correlation
            factor_corr = factor_corr / np.sqrt(np.diag(factor_corr)[:, None] @ np.diag(factor_corr)[None, :])

            self.log("\nFactor-based correlation matrix:")
            factor_corr_df = pd.DataFrame(factor_corr, index=assets, columns=assets)
            self.log(factor_corr_df.to_string())

            self.results['correlations'] = {
                'empirical': empirical_corr,
                'factor_based': factor_corr_df
            }

            self._plot_correlation_comparison(empirical_corr, factor_corr_df, assets)

    def _generate_reconstruction_table(self, results):
        """Generate LaTeX table for reconstruction quality (Table 2)."""

        latex = r"""\begin{table}[h]
\centering
\caption{Reconstruction Performance (Explained Variance)}
\label{tab:reconstruction}
\begin{tabular}{lrrrrr}
\toprule
Method & Rank 3 & Rank 5 & Rank 10 & Rank 15 & Rank 20 \\
\midrule
"""

        for method in ['pca', 'cp', 'tucker', 'tt']:
            method_name = method.upper() if method != 'pca' else 'PCA'
            values = ' & '.join([f"{v:.2f}" for v in results[method]])
            latex += f"{method_name} & {values} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

        output_path = TABLES_DIR / 'reconstruction_quality.tex'
        with open(output_path, 'w') as f:
            f.write(latex)

        self.log(f"\n✓ Saved LaTeX table: {output_path}")

    def _plot_reconstruction_comparison(self, results):
        """Plot reconstruction quality across methods and ranks."""

        plt.figure(figsize=(10, 6))

        for method, label in [('pca', 'PCA'), ('cp', 'CP'), ('tucker', 'Tucker'), ('tt', 'TT')]:
            plt.plot(results['ranks'], results[method], marker='o', label=label, linewidth=2)

        plt.axhline(y=0.90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
        plt.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label='95% threshold')

        plt.xlabel('Rank', fontsize=12)
        plt.ylabel('Explained Variance', fontsize=12)
        plt.title('Reconstruction Quality: Tensor Methods vs PCA', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        output_path = FIGURES_DIR / 'reconstruction_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.log(f"✓ Saved figure: {output_path}")
        plt.close()

    def _plot_temporal_factors(self, temporal_factors):
        """Plot temporal factor evolution over time."""

        n_factors = min(5, temporal_factors.shape[1])  # Plot top 5 factors

        fig, axes = plt.subplots(n_factors, 1, figsize=(14, 2.5 * n_factors), sharex=True)
        if n_factors == 1:
            axes = [axes]

        for i in range(n_factors):
            axes[i].plot(temporal_factors[:, i], linewidth=1)
            axes[i].set_ylabel(f'Factor {i+1}', fontsize=10)
            axes[i].grid(alpha=0.3)
            axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        axes[-1].set_xlabel('Time (hours)', fontsize=12)
        fig.suptitle('Temporal Factor Evolution', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = FIGURES_DIR / 'temporal_factors.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.log(f"✓ Saved figure: {output_path}")
        plt.close()

    def _plot_asset_factors(self, asset_factors):
        """Plot asset factor loadings."""

        assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        n_factors = asset_factors.shape[1]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(assets))
        width = 0.8 / n_factors

        for i in range(min(5, n_factors)):
            offset = (i - n_factors/2) * width
            ax.bar(x + offset, asset_factors[:, i], width, label=f'Factor {i+1}')

        ax.set_xlabel('Asset', fontsize=12)
        ax.set_ylabel('Factor Loading', fontsize=12)
        ax.set_title('Asset Factor Loadings', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(assets)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.8)
        plt.tight_layout()

        output_path = FIGURES_DIR / 'asset_factors.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.log(f"✓ Saved figure: {output_path}")
        plt.close()

    def _plot_feature_factors(self, feature_factors):
        """Plot feature factor loadings."""

        features = ['open', 'high', 'low', 'close', 'volume']
        n_factors = feature_factors.shape[1]

        plt.figure(figsize=(10, 6))

        im = plt.imshow(feature_factors.T, aspect='auto', cmap='RdBu_r',
                       vmin=-np.abs(feature_factors).max(), vmax=np.abs(feature_factors).max())

        plt.colorbar(im, label='Factor Loading')
        plt.yticks(range(n_factors), [f'Factor {i+1}' for i in range(n_factors)])
        plt.xticks(range(len(features)), features)
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Latent Factor', fontsize=12)
        plt.title('Feature Factor Loadings', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = FIGURES_DIR / 'feature_factors.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.log(f"✓ Saved figure: {output_path}")
        plt.close()

    def _generate_variant_comparison_table(self, results):
        """Generate table comparing tensor variants."""

        latex = r"""\begin{table}[h]
\centering
\caption{Performance by Tensor Variant (Rank 10)}
\label{tab:tensor_variants}
\begin{tabular}{lrr}
\toprule
Tensor Type & CP Variance & Tucker Variance \\
\midrule
"""

        for name, data in results.items():
            display_name = name.replace('_', ' ').title()
            latex += f"{display_name} & {data['cp_variance']:.4f} & {data['tucker_variance']:.4f} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

        output_path = TABLES_DIR / 'tensor_variants.tex'
        with open(output_path, 'w') as f:
            f.write(latex)

        self.log(f"✓ Saved table: {output_path}")

    def _plot_correlation_comparison(self, empirical, factor_based, assets):
        """Compare empirical vs factor-based correlations."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Empirical correlation
        im1 = ax1.imshow(empirical, cmap='RdBu_r', vmin=-1, vmax=1)
        ax1.set_xticks(range(len(assets)))
        ax1.set_yticks(range(len(assets)))
        ax1.set_xticklabels(assets, rotation=45, ha='right')
        ax1.set_yticklabels(assets)
        ax1.set_title('Empirical Correlation', fontsize=12, fontweight='bold')

        for i in range(len(assets)):
            for j in range(len(assets)):
                ax1.text(j, i, f'{empirical.iloc[i, j]:.2f}', ha='center', va='center')

        # Factor-based correlation
        im2 = ax2.imshow(factor_based, cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_xticks(range(len(assets)))
        ax2.set_yticks(range(len(assets)))
        ax2.set_xticklabels(assets, rotation=45, ha='right')
        ax2.set_yticklabels(assets)
        ax2.set_title('Factor-Based Correlation', fontsize=12, fontweight='bold')

        for i in range(len(assets)):
            for j in range(len(assets)):
                ax2.text(j, i, f'{factor_based.iloc[i, j]:.2f}', ha='center', va='center')

        fig.colorbar(im2, ax=[ax1, ax2], label='Correlation', fraction=0.046, pad=0.04)
        plt.tight_layout()

        output_path = FIGURES_DIR / 'correlation_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.log(f"✓ Saved figure: {output_path}")
        plt.close()

    def save_results(self):
        """Save all experimental results."""

        output_path = RESULTS_DIR / f'experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'

        with open(output_path, 'wb') as f:
            pickle.dump(self.results, f)

        self.log(f"\n✓ Saved all results: {output_path}")

        # Also save summary as JSON for easy inspection
        import json
        summary = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                summary[key] = {k: str(v) if isinstance(v, np.ndarray) else v
                               for k, v in value.items()}
            else:
                summary[key] = str(value)

        json_path = RESULTS_DIR / f'experiment_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.log(f"✓ Saved summary: {json_path}")

    def run_all_experiments(self):
        """
        Run complete experimental pipeline.

        Memory-optimized: Each experiment loads/frees tensors independently.
        """

        start_time = datetime.now()
        start_mem = self.get_memory_mb()

        self.log("\n" + "="*80)
        self.log("TENSOR DECOMPOSITION EXPERIMENTS - FULL PIPELINE")
        self.log("="*80)

        # Run experiments (each loads its own tensors on-demand)
        self.experiment_1_reconstruction_quality()
        self.free_memory()

        self.experiment_2_rank_selection()
        self.free_memory()

        self.experiment_3_factor_analysis()
        self.free_memory()

        self.experiment_4_variance_by_tensor_type()
        self.free_memory()

        self.experiment_5_correlation_analysis()
        self.free_memory()

        # Save everything
        self.save_results()

        elapsed = (datetime.now() - start_time).total_seconds()
        end_mem = self.get_memory_mb()
        peak_mem_increase = end_mem - start_mem

        self.log("\n" + "="*80)
        self.log(f"ALL EXPERIMENTS COMPLETE - Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        self.log(f"Memory: Start={start_mem:.0f}MB, End={end_mem:.0f}MB, Peak increase={peak_mem_increase:.0f}MB")
        self.log("="*80)
        self.log(f"\nOutputs saved to:")
        self.log(f"  Figures: {FIGURES_DIR}/")
        self.log(f"  Tables:  {TABLES_DIR}/")
        self.log(f"  Results: {RESULTS_DIR}/")

        return self.results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TENSOR DECOMPOSITION FOR DEFI MARKETS - EXPERIMENTAL PIPELINE")
    print("="*80)
    print("\nThis script runs all experiments for the academic paper.")
    print("Expected runtime: 10-30 minutes\n")

    runner = ExperimentRunner(verbose=True)
    results = runner.run_all_experiments()

    print("\n✓ Experiment pipeline complete!")
    print("\nNext steps:")
    print("  1. Review figures in outputs/figures/")
    print("  2. Check LaTeX tables in outputs/tables/")
    print("  3. Update paper Section 4 with results")
    print("  4. Run: cd _internal/paper && pdflatex main.tex")
