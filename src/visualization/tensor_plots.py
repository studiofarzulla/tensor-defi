"""
Tensor Visualization - Make the high-dimensional structure visible

Visualizing tensors is hard - we live in 3D but these objects are 4D+.
This module creates insightful projections and slices that reveal patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)


class TensorVisualizer:
    """Visualize tensor decompositions and market structure."""

    def __init__(self, style: str = 'dark'):
        """
        Initialize visualizer.

        Args:
            style: Plot style ('dark', 'light', 'presentation')
        """
        self.style = style
        if style == 'dark':
            plt.style.use('dark_background')
        elif style == 'presentation':
            sns.set_context("talk")

    def plot_tensor_factors(
        self,
        result,  # DecompositionResult
        metadata,  # TensorMetadata
        save_path: Optional[str] = None
    ):
        """
        Plot factor matrices from tensor decomposition.

        Args:
            result: DecompositionResult from CP/Tucker
            metadata: TensorMetadata with dimension info
            save_path: Optional path to save figure
        """
        n_factors = len(result.factors)

        fig, axes = plt.subplots(1, n_factors, figsize=(5 * n_factors, 6))
        if n_factors == 1:
            axes = [axes]

        fig.suptitle(f'{result.method.upper()} Decomposition Factors (rank={result.rank})',
                     fontsize=16, fontweight='bold')

        for i, (factor, ax) in enumerate(zip(result.factors, axes)):
            dim_name = metadata.dimension_names[i]

            # Heatmap of factor matrix
            im = ax.imshow(factor, aspect='auto', cmap='RdBu_r', interpolation='nearest')
            ax.set_title(f'{dim_name.capitalize()} Factors')
            ax.set_xlabel('Components')
            ax.set_ylabel(dim_name.capitalize())

            # Colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")

        plt.show()

    def plot_temporal_evolution(
        self,
        temporal_factors: np.ndarray,
        timestamps: Optional[List] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot temporal evolution of latent factors.

        Shows how market regimes change over time.

        Args:
            temporal_factors: Time × Components matrix
            timestamps: Optional list of timestamps
            save_path: Save path
        """
        n_time, n_components = temporal_factors.shape

        if timestamps is None:
            timestamps = np.arange(n_time)

        fig, axes = plt.subplots(n_components, 1, figsize=(14, 3 * n_components), sharex=True)
        if n_components == 1:
            axes = [axes]

        fig.suptitle('Temporal Evolution of Latent Factors', fontsize=16, fontweight='bold')

        for i, ax in enumerate(axes):
            ax.plot(timestamps, temporal_factors[:, i], linewidth=2, label=f'Factor {i+1}')
            ax.set_ylabel(f'Factor {i+1}')
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel('Time')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")

        plt.show()

    def plot_reconstruction_quality(
        self,
        original: np.ndarray,
        reconstruction: np.ndarray,
        slice_dims: Tuple[int, int] = (0, 0),
        save_path: Optional[str] = None
    ):
        """
        Compare original vs reconstructed tensor (2D slice).

        Args:
            original: Original tensor
            reconstruction: Reconstructed tensor
            slice_dims: Which dimensions to slice on
            save_path: Save path
        """
        # Extract 2D slice
        orig_slice = original[slice_dims[0], slice_dims[1]]
        recon_slice = reconstruction[slice_dims[0], slice_dims[1]]
        diff_slice = orig_slice - recon_slice

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Original
        im1 = axes[0].imshow(orig_slice, aspect='auto', cmap='viridis')
        axes[0].set_title('Original')
        axes[0].set_xlabel('Features')
        axes[0].set_ylabel('Assets')
        plt.colorbar(im1, ax=axes[0])

        # Reconstruction
        im2 = axes[1].imshow(recon_slice, aspect='auto', cmap='viridis')
        axes[1].set_title('Reconstruction')
        axes[1].set_xlabel('Features')
        axes[1].set_ylabel('Assets')
        plt.colorbar(im2, ax=axes[1])

        # Difference
        im3 = axes[2].imshow(diff_slice, aspect='auto', cmap='RdBu_r')
        axes[2].set_title('Residual Error')
        axes[2].set_xlabel('Features')
        axes[2].set_ylabel('Assets')
        plt.colorbar(im3, ax=axes[2])

        error = np.linalg.norm(diff_slice)
        fig.suptitle(f'Reconstruction Quality (Error: {error:.4f})', fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_explained_variance(
        self,
        results: Dict,  # method -> DecompositionResult
        save_path: Optional[str] = None
    ):
        """
        Compare explained variance across methods.

        Args:
            results: Dict of decomposition results
            save_path: Save path
        """
        methods = list(results.keys())
        variances = [results[m].explained_variance for m in methods]
        errors = [results[m].reconstruction_error for m in methods]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Explained variance
        bars1 = ax1.bar(methods, variances, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_ylabel('Explained Variance')
        ax1.set_title('Explained Variance by Method')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)

        for bar, var in zip(bars1, variances):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{var:.3f}', ha='center', va='bottom')

        # Reconstruction error
        bars2 = ax2.bar(methods, errors, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_ylabel('Reconstruction Error')
        ax2.set_title('Reconstruction Error by Method')
        ax2.grid(axis='y', alpha=0.3)

        for bar, err in zip(bars2, errors):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{err:.2f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_rank_selection(
        self,
        ranks: List[int],
        errors: List[float],
        variances: List[float],
        optimal_rank: Optional[int] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize rank selection process.

        Args:
            ranks: List of tested ranks
            errors: Reconstruction errors
            variances: Explained variances
            optimal_rank: Highlight this rank
            save_path: Save path
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Error curve
        ax1.plot(ranks, errors, 'o-', linewidth=2, markersize=8, label='Error')
        if optimal_rank:
            ax1.axvline(optimal_rank, color='red', linestyle='--', label=f'Optimal (rank={optimal_rank})')
        ax1.set_ylabel('Reconstruction Error')
        ax1.set_title('Rank Selection: Error vs Rank')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Variance curve
        ax2.plot(ranks, variances, 'o-', linewidth=2, markersize=8, color='green', label='Variance')
        if optimal_rank:
            ax2.axvline(optimal_rank, color='red', linestyle='--', label=f'Optimal (rank={optimal_rank})')
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Explained Variance')
        ax2.set_title('Rank Selection: Variance vs Rank')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_3d_factor_space(
        self,
        factors: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Factor Space",
        save_path: Optional[str] = None
    ):
        """
        3D scatter plot of first 3 factor components.

        Args:
            factors: Factor matrix (N × Components)
            labels: Optional labels for points
            title: Plot title
            save_path: Save path
        """
        if factors.shape[1] < 3:
            print("⚠ Need at least 3 components for 3D plot")
            return

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            factors[:, 0],
            factors[:, 1],
            factors[:, 2],
            c=np.arange(len(factors)),
            cmap='viridis',
            s=100,
            alpha=0.6
        )

        ax.set_xlabel('Factor 1')
        ax.set_ylabel('Factor 2')
        ax.set_zlabel('Factor 3')
        ax.set_title(title)

        if labels:
            for i, label in enumerate(labels):
                ax.text(factors[i, 0], factors[i, 1], factors[i, 2], label, fontsize=8)

        plt.colorbar(scatter, ax=ax, pad=0.1)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_interactive_tensor_slice(
        self,
        tensor: np.ndarray,
        metadata,  # TensorMetadata
        save_path: Optional[str] = None
    ):
        """
        Interactive plotly visualization of tensor slices.

        Args:
            tensor: Input tensor
            metadata: TensorMetadata
            save_path: Save path (HTML)
        """
        # Create interactive sliders for each dimension
        # Simplified: show time × asset slice for each venue

        fig = make_subplots(
            rows=1, cols=len(metadata.venues),
            subplot_titles=[f'Venue: {v}' for v in metadata.venues]
        )

        for v_idx, venue in enumerate(metadata.venues):
            # Slice: time × assets × first feature
            slice_data = tensor[:, v_idx, :, 0]

            fig.add_trace(
                go.Heatmap(
                    z=slice_data.T,
                    colorscale='Viridis',
                    showscale=(v_idx == 0),
                ),
                row=1, col=v_idx + 1
            )

        fig.update_layout(
            title_text="Tensor Slices Across Venues",
            height=500,
            showlegend=False
        )

        if save_path:
            fig.write_html(save_path)
            print(f"✓ Saved interactive plot to {save_path}")

        fig.show()


if __name__ == "__main__":
    print("=== Testing Tensor Visualization ===\n")

    # Create synthetic results
    np.random.seed(42)
    time_dim, venue_dim, asset_dim, feature_dim = 100, 3, 5, 10
    rank = 3

    # Generate factors
    time_factor = np.random.randn(time_dim, rank)
    venue_factor = np.random.randn(venue_dim, rank)
    asset_factor = np.random.randn(asset_dim, rank)
    feature_factor = np.random.randn(feature_dim, rank)

    # Create mock decomposition result
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class MockResult:
        method: str = 'cp'
        rank: int = rank
        factors: list = None
        reconstruction_error: float = 10.5
        explained_variance: float = 0.92

    result = MockResult(factors=[time_factor, venue_factor, asset_factor, feature_factor])

    @dataclass
    class MockMetadata:
        dimension_names: list = None
        venues: list = None

    metadata = MockMetadata(
        dimension_names=['time', 'venue', 'asset', 'feature'],
        venues=['binance', 'coinbase', 'uniswap']
    )

    # Test visualizations
    viz = TensorVisualizer(style='dark')

    print("1. Testing factor plots...")
    viz.plot_tensor_factors(result, metadata)

    print("\n2. Testing temporal evolution...")
    viz.plot_temporal_evolution(time_factor)

    print("\n3. Testing 3D factor space...")
    viz.plot_3d_factor_space(venue_factor, labels=metadata.venues, title="Venue Factor Space")

    print("\n✓ Visualization tests complete!")
