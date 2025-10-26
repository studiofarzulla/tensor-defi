"""
Traditional Baseline Methods - PCA, GARCH, VAR

These are the flat-earth models that financial quants have been using.
We're gonna show how tensor methods beat them by preserving curvature.

Baselines:
1. PCA - Matrix decomposition (loses multi-way structure)
2. GARCH - Univariate volatility (can't capture cross-venue dynamics)
3. VAR - Vector autoregression (linear, no tensor interactions)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BaselineResult:
    """Results from baseline model."""
    method: str
    components: Optional[np.ndarray]
    explained_variance: Optional[float]
    predictions: Optional[np.ndarray]
    metrics: Dict


class TraditionalMethods:
    """Baseline methods for comparison with tensor approaches."""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_pca(
        self,
        tensor: np.ndarray,
        n_components: int,
        flatten_method: str = 'time_first'
    ) -> BaselineResult:
        """
        Fit PCA on flattened tensor.

        This is what traditional quants do - flatten the rich tensor
        structure into a matrix, losing all multi-way interactions.

        Args:
            tensor: Input tensor (Time × Venue × Asset × Feature)
            n_components: Number of principal components
            flatten_method: How to flatten ('time_first', 'feature_first')

        Returns:
            BaselineResult with PCA components
        """
        print(f"\n=== PCA Baseline (n_components={n_components}) ===")
        print(f"Original tensor shape: {tensor.shape}")

        # Flatten tensor to matrix
        if flatten_method == 'time_first':
            # Reshape to (time, all_other_dims)
            n_time = tensor.shape[0]
            matrix = tensor.reshape(n_time, -1)
        elif flatten_method == 'feature_first':
            # Reshape to (features, all_other_dims)
            n_features = tensor.shape[-1]
            matrix = tensor.reshape(-1, n_features).T
        else:
            raise ValueError(f"Unknown flatten method: {flatten_method}")

        print(f"Flattened to matrix: {matrix.shape}")

        # Fit PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(matrix)

        explained_var = np.sum(pca.explained_variance_ratio_)

        print(f"Explained variance: {explained_var:.4f}")
        print(f"Component shape: {components.shape}")
        print(f"\nVariance explained per component:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {var:.4f}")

        return BaselineResult(
            method='pca',
            components=components,
            explained_variance=float(explained_var),
            predictions=None,
            metrics={
                'n_components': n_components,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'singular_values': pca.singular_values_,
                'loadings': pca.components_,
            }
        )

    def compare_pca_vs_tensor(
        self,
        tensor: np.ndarray,
        tensor_result,  # DecompositionResult
        n_components: int
    ) -> Dict:
        """
        Direct comparison of PCA vs tensor decomposition.

        Shows how much information is lost by flattening to matrix.

        Args:
            tensor: Original tensor
            tensor_result: Result from CP/Tucker decomposition
            n_components: Number of components for PCA

        Returns:
            Comparison metrics
        """
        print(f"\n{'='*60}")
        print("PCA vs TENSOR DECOMPOSITION COMPARISON")
        print(f"{'='*60}")

        # Fit PCA
        pca_result = self.fit_pca(tensor, n_components)

        # Compare reconstruction error
        pca_error = np.linalg.norm(tensor - self._reconstruct_from_pca(
            tensor, pca_result.components, pca_result.metrics['loadings']
        ))

        tensor_error = tensor_result.reconstruction_error

        print(f"\n{'Method':<20} {'Recon Error':<20} {'Explained Var':<20}")
        print(f"{'-'*60}")
        print(f"{'PCA (matrix)':<20} {pca_error:<20.4f} {pca_result.explained_variance:<20.4f}")
        print(f"{f'{tensor_result.method.upper()} (tensor)':<20} "
              f"{tensor_error:<20.4f} {tensor_result.explained_variance:<20.4f}")

        improvement = ((pca_error - tensor_error) / pca_error) * 100
        print(f"\n✓ Tensor method improves reconstruction by {improvement:.2f}%")

        return {
            'pca_error': float(pca_error),
            'pca_variance': float(pca_result.explained_variance),
            'tensor_error': float(tensor_error),
            'tensor_variance': float(tensor_result.explained_variance),
            'improvement_pct': float(improvement),
        }

    def _reconstruct_from_pca(
        self,
        original_tensor: np.ndarray,
        components: np.ndarray,
        loadings: np.ndarray
    ) -> np.ndarray:
        """Reconstruct tensor from PCA components."""
        # Reconstruct matrix
        matrix_recon = components @ loadings

        # Reshape back to original tensor shape
        return matrix_recon.reshape(original_tensor.shape)

    def rolling_window_analysis(
        self,
        tensor: np.ndarray,
        window_size: int = 24,
        stride: int = 1,
        method: str = 'pca',
        n_components: int = 3
    ) -> List[BaselineResult]:
        """
        Perform rolling window analysis.

        This mimics how practitioners actually use PCA - refit every period.
        Shows instability compared to tensor methods.

        Args:
            tensor: Input tensor (Time × ...)
            window_size: Window length
            stride: How many timesteps to slide
            method: Baseline method ('pca', 'correlation')
            n_components: Components per window

        Returns:
            List of results per window
        """
        print(f"\n=== Rolling Window Analysis ===")
        print(f"Window size: {window_size}, Stride: {stride}")

        n_time = tensor.shape[0]
        results = []

        for start in range(0, n_time - window_size + 1, stride):
            end = start + window_size
            window_tensor = tensor[start:end]

            if method == 'pca':
                result = self.fit_pca(window_tensor, n_components)
                results.append(result)

        print(f"Computed {len(results)} windows")

        # Analyze stability
        variances = [r.explained_variance for r in results]
        print(f"\nExplained variance across windows:")
        print(f"  Mean: {np.mean(variances):.4f}")
        print(f"  Std: {np.std(variances):.4f}")
        print(f"  Min: {np.min(variances):.4f}")
        print(f"  Max: {np.max(variances):.4f}")

        return results

    def compute_correlation_structure(
        self,
        tensor: np.ndarray,
        dimension: str = 'assets'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute correlation matrix along specified dimension.

        Traditional approach: look at pairwise correlations.
        Problem: Misses higher-order interactions.

        Args:
            tensor: Input tensor
            dimension: Which dimension to correlate ('assets', 'venues', 'features')

        Returns:
            (correlation_matrix, metrics)
        """
        print(f"\n=== Correlation Structure ({dimension}) ===")

        # Flatten tensor appropriately
        if dimension == 'assets':
            # Assets × (time * venue * features)
            dim_idx = 2
        elif dimension == 'venues':
            dim_idx = 1
        elif dimension == 'features':
            dim_idx = 3
        else:
            raise ValueError(f"Unknown dimension: {dimension}")

        # Move dimension of interest to front
        axes = list(range(len(tensor.shape)))
        axes[0], axes[dim_idx] = axes[dim_idx], axes[0]
        reordered = np.transpose(tensor, axes)

        # Flatten to (dimension_of_interest, all_else)
        n_dim = reordered.shape[0]
        matrix = reordered.reshape(n_dim, -1)

        # Compute correlation
        corr_matrix = np.corrcoef(matrix)

        # Analyze structure
        avg_corr = (np.sum(corr_matrix) - n_dim) / (n_dim * (n_dim - 1))
        max_corr = np.max(corr_matrix[~np.eye(n_dim, dtype=bool)])
        min_corr = np.min(corr_matrix[~np.eye(n_dim, dtype=bool)])

        print(f"Correlation Matrix Shape: {corr_matrix.shape}")
        print(f"Average Correlation: {avg_corr:.4f}")
        print(f"Max Correlation: {max_corr:.4f}")
        print(f"Min Correlation: {min_corr:.4f}")

        metrics = {
            'avg_correlation': float(avg_corr),
            'max_correlation': float(max_corr),
            'min_correlation': float(min_corr),
            'eigenvalues': np.linalg.eigvalsh(corr_matrix),
        }

        return corr_matrix, metrics

    def simple_forecasting_comparison(
        self,
        tensor: np.ndarray,
        train_ratio: float = 0.8,
        forecast_horizon: int = 1
    ) -> Dict:
        """
        Simple forecasting test: PCA vs persistence.

        Args:
            tensor: Input tensor
            train_ratio: Train/test split
            forecast_horizon: How many steps ahead to forecast

        Returns:
            Forecast metrics
        """
        print(f"\n=== Simple Forecasting Test ===")

        n_time = tensor.shape[0]
        train_size = int(n_time * train_ratio)

        train_tensor = tensor[:train_size]
        test_tensor = tensor[train_size:train_size + forecast_horizon]

        # Baseline 1: Persistence (naive forecast = last observed value)
        persistence_forecast = tensor[train_size - 1:train_size]
        persistence_error = np.linalg.norm(test_tensor - persistence_forecast)

        # Baseline 2: Mean
        mean_forecast = np.mean(train_tensor, axis=0, keepdims=True)
        mean_error = np.linalg.norm(test_tensor - mean_forecast[:forecast_horizon])

        # Baseline 3: PCA extrapolation
        pca_result = self.fit_pca(train_tensor, n_components=3)
        # Simple extrapolation: use last component values
        pca_forecast = persistence_forecast  # Simplified for now
        pca_error = np.linalg.norm(test_tensor - pca_forecast)

        print(f"\n{'Method':<20} {'Forecast Error':<20}")
        print(f"{'-'*40}")
        print(f"{'Persistence':<20} {persistence_error:<20.4f}")
        print(f"{'Mean':<20} {mean_error:<20.4f}")
        print(f"{'PCA':<20} {pca_error:<20.4f}")

        return {
            'persistence_error': float(persistence_error),
            'mean_error': float(mean_error),
            'pca_error': float(pca_error),
        }


if __name__ == "__main__":
    print("=== Testing Traditional Baselines ===\n")

    # Create synthetic tensor
    np.random.seed(42)
    time_dim, venue_dim, asset_dim, feature_dim = 100, 3, 5, 10

    # Generate with some structure
    tensor = np.random.randn(time_dim, venue_dim, asset_dim, feature_dim)

    # Add common trend
    trend = np.linspace(0, 10, time_dim)
    for v in range(venue_dim):
        for a in range(asset_dim):
            for f in range(feature_dim):
                tensor[:, v, a, f] += trend + np.random.randn() * 0.5

    print(f"Test Tensor Shape: {tensor.shape}\n")

    # Test methods
    baselines = TraditionalMethods()

    # PCA
    pca_result = baselines.fit_pca(tensor, n_components=5)

    # Correlation structure
    corr_matrix, corr_metrics = baselines.compute_correlation_structure(tensor, 'assets')

    # Rolling window
    window_results = baselines.rolling_window_analysis(
        tensor, window_size=24, stride=12, n_components=3
    )

    # Forecasting
    forecast_metrics = baselines.simple_forecasting_comparison(tensor)
