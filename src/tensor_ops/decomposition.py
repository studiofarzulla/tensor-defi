"""
Tensor Decomposition - CP, Tucker, and Tensor Train

Decomposes market tensors into latent factor structures that reveal
hidden patterns invisible to matrix-based methods.

Key insight: Financial markets have low-rank structure in tensor space.
Multi-way interactions can be captured with far fewer parameters than
full tensor requires.
"""

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker, tensor_train
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

# Set tensorly backend
tl.set_backend('numpy')


@dataclass
class DecompositionResult:
    """Results from tensor decomposition."""
    method: str
    rank: Union[int, List[int]]
    factors: Union[List[np.ndarray], Tuple]
    core: Optional[np.ndarray]
    reconstruction: np.ndarray
    reconstruction_error: float
    explained_variance: float
    n_iterations: Optional[int]
    converged: bool


class TensorDecomposer:
    """Decompose market tensors using CP, Tucker, and TT methods."""

    def __init__(self, backend: str = 'numpy'):
        """
        Initialize decomposer.

        Args:
            backend: Tensorly backend ('numpy', 'pytorch', 'jax')
        """
        tl.set_backend(backend)
        self.backend = backend

    def cp_decomposition(
        self,
        tensor: np.ndarray,
        rank: int,
        init: str = 'random',
        n_iter_max: int = 100,
        tol: float = 1e-6,
        verbose: bool = False
    ) -> DecompositionResult:
        """
        CP (CANDECOMP/PARAFAC) decomposition.

        Decomposes tensor into sum of rank-1 tensors:
        X ≈ Σᵣ λᵣ (a₁^(r) ⊗ a₂^(r) ⊗ ... ⊗ aₙ^(r))

        This is the most compact representation, revealing core
        latent factors that operate across all dimensions.

        Args:
            tensor: Input tensor
            rank: Number of components
            init: Initialization method ('random', 'svd')
                  'random' is memory-efficient and recommended for large tensors
                  'svd' provides better initialization but requires O(N²) memory
                  for HOSVD where N is the product of dimensions
            n_iter_max: Maximum iterations
            tol: Convergence tolerance
            verbose: Print progress

        Returns:
            DecompositionResult with factors and metrics
        """
        if verbose:
            print(f"\n=== CP Decomposition (rank={rank}, init={init}) ===")

        # Memory optimization: warn if using SVD with large tensor
        if init == 'svd':
            tensor_size_gb = tensor.nbytes / (1024**3)
            # Estimate HOSVD memory: largest unfolding creates N×M matrix
            # where M = prod(dims) / N, and SVD needs M×M for V matrix
            max_unfold_dim = max(np.prod(tensor.shape) // s for s in tensor.shape)
            svd_memory_gb = (max_unfold_dim ** 2 * 8) / (1024**3)

            if svd_memory_gb > 10:
                if verbose:
                    print(f"  WARNING: SVD init may require {svd_memory_gb:.1f} GB memory")
                    print(f"  Consider using init='random' for large tensors")

        # Convert to tensorly tensor
        tl_tensor = tl.tensor(tensor)

        # Perform CP decomposition
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                weights, factors = parafac(
                    tl_tensor,
                    rank=rank,
                    init=init,
                    n_iter_max=n_iter_max,
                    tol=tol,
                    verbose=verbose
                )

            # Reconstruct tensor
            reconstruction = tl.cp_to_tensor((weights, factors))
            reconstruction = tl.to_numpy(reconstruction)

            # Compute error metrics
            error = np.linalg.norm(tensor - reconstruction)
            relative_error = error / np.linalg.norm(tensor)
            explained_var = 1 - (error ** 2) / np.sum(tensor ** 2)

            if verbose:
                print(f"Reconstruction Error: {error:.4f}")
                print(f"Relative Error: {relative_error:.4f}")
                print(f"Explained Variance: {explained_var:.4f}")
                print(f"\nFactor Shapes:")
                for i, factor in enumerate(factors):
                    print(f"  Factor {i}: {factor.shape}")

            return DecompositionResult(
                method='cp',
                rank=rank,
                factors=[tl.to_numpy(f) for f in factors],
                core=tl.to_numpy(weights),
                reconstruction=reconstruction,
                reconstruction_error=float(error),
                explained_variance=float(explained_var),
                n_iterations=None,
                converged=True
            )

        except Exception as e:
            print(f"✗ CP decomposition failed: {e}")
            raise

    def tucker_decomposition(
        self,
        tensor: np.ndarray,
        ranks: Union[int, List[int]],
        init: str = 'random',
        n_iter_max: int = 100,
        tol: float = 1e-6,
        verbose: bool = False
    ) -> DecompositionResult:
        """
        Tucker decomposition.

        Decomposes tensor into core tensor and factor matrices:
        X ≈ G ×₁ A₁ ×₂ A₂ ×₃ A₃ ...

        Tucker is more flexible than CP, allowing different ranks
        per dimension. Better for capturing asymmetric structure.

        Args:
            tensor: Input tensor
            ranks: Rank per dimension (int = same for all, list = per-dimension)
            init: Initialization method ('random', 'svd')
                  'random' is memory-efficient and recommended for large tensors
                  'svd' (HOSVD) provides better initialization but requires O(N²) memory
            n_iter_max: Maximum iterations
            tol: Convergence tolerance
            verbose: Print progress

        Returns:
            DecompositionResult with core, factors, and metrics
        """
        if verbose:
            print(f"\n=== Tucker Decomposition (ranks={ranks}, init={init}) ===")

        # Memory optimization: warn if using SVD with large tensor
        if init == 'svd':
            max_unfold_dim = max(np.prod(tensor.shape) // s for s in tensor.shape)
            svd_memory_gb = (max_unfold_dim ** 2 * 8) / (1024**3)

            if svd_memory_gb > 10:
                if verbose:
                    print(f"  WARNING: SVD init may require {svd_memory_gb:.1f} GB memory")
                    print(f"  Consider using init='random' for large tensors")

        tl_tensor = tl.tensor(tensor)

        # Handle rank specification
        if isinstance(ranks, int):
            ranks = [ranks] * len(tensor.shape)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                core, factors = tucker(
                    tl_tensor,
                    rank=ranks,
                    init=init,
                    n_iter_max=n_iter_max,
                    tol=tol,
                    verbose=verbose
                )

            # Reconstruct
            reconstruction = tl.tucker_to_tensor((core, factors))
            reconstruction = tl.to_numpy(reconstruction)

            # Metrics
            error = np.linalg.norm(tensor - reconstruction)
            relative_error = error / np.linalg.norm(tensor)
            explained_var = 1 - (error ** 2) / np.sum(tensor ** 2)

            if verbose:
                print(f"Core Tensor Shape: {core.shape}")
                print(f"Reconstruction Error: {error:.4f}")
                print(f"Explained Variance: {explained_var:.4f}")
                print(f"\nFactor Shapes:")
                for i, factor in enumerate(factors):
                    print(f"  Factor {i}: {factor.shape}")

            return DecompositionResult(
                method='tucker',
                rank=ranks,
                factors=[tl.to_numpy(f) for f in factors],
                core=tl.to_numpy(core),
                reconstruction=reconstruction,
                reconstruction_error=float(error),
                explained_variance=float(explained_var),
                n_iterations=None,
                converged=True
            )

        except Exception as e:
            print(f"✗ Tucker decomposition failed: {e}")
            raise

    def tensor_train_decomposition(
        self,
        tensor: np.ndarray,
        ranks: Union[int, List[int]],
        verbose: bool = False
    ) -> DecompositionResult:
        """
        Tensor Train (TT) decomposition.

        Decomposes into chain of 3-way cores:
        X ≈ G₁ ×₂ G₂ ×₂ G₃ ×₂ ...

        TT is extremely efficient for high-dimensional tensors,
        scales linearly with number of dimensions.

        Args:
            tensor: Input tensor
            ranks: TT-ranks (int or list)
            verbose: Print progress

        Returns:
            DecompositionResult
        """
        if verbose:
            print(f"\n=== Tensor Train Decomposition (ranks={ranks}) ===")

        tl_tensor = tl.tensor(tensor)

        if isinstance(ranks, int):
            ranks = [1] + [ranks] * (len(tensor.shape) - 1) + [1]

        try:
            factors = tensor_train(tl_tensor, rank=ranks)

            # Reconstruct
            reconstruction = tl.tt_to_tensor(factors)
            reconstruction = tl.to_numpy(reconstruction)

            # Metrics
            error = np.linalg.norm(tensor - reconstruction)
            explained_var = 1 - (error ** 2) / np.sum(tensor ** 2)

            if verbose:
                print(f"Reconstruction Error: {error:.4f}")
                print(f"Explained Variance: {explained_var:.4f}")
                print(f"\nTT-Core Shapes:")
                for i, core in enumerate(factors):
                    print(f"  Core {i}: {core.shape}")

            return DecompositionResult(
                method='tensor_train',
                rank=ranks,
                factors=[tl.to_numpy(f) for f in factors],
                core=None,
                reconstruction=reconstruction,
                reconstruction_error=float(error),
                explained_variance=float(explained_var),
                n_iterations=None,
                converged=True
            )

        except Exception as e:
            print(f"✗ Tensor Train decomposition failed: {e}")
            raise

    def rank_selection(
        self,
        tensor: np.ndarray,
        method: str = 'cp',
        max_rank: int = 20,
        criterion: str = 'explained_variance',
        threshold: float = 0.95
    ) -> int:
        """
        Automatically select optimal rank.

        Uses elbow method or variance threshold to find best rank.

        Args:
            tensor: Input tensor
            method: Decomposition method ('cp', 'tucker')
            max_rank: Maximum rank to test
            criterion: 'explained_variance' or 'aic'
            threshold: Variance threshold (if using variance criterion)

        Returns:
            Optimal rank
        """
        print(f"\n=== Rank Selection (method={method}, criterion={criterion}) ===")

        errors = []
        variances = []
        ranks = range(1, max_rank + 1)

        for rank in ranks:
            try:
                if method == 'cp':
                    result = self.cp_decomposition(tensor, rank, verbose=False)
                elif method == 'tucker':
                    result = self.tucker_decomposition(tensor, rank, verbose=False)
                else:
                    raise ValueError(f"Unknown method: {method}")

                errors.append(result.reconstruction_error)
                variances.append(result.explained_variance)

                print(f"Rank {rank:2d}: Error={result.reconstruction_error:.4f}, "
                      f"Var={result.explained_variance:.4f}")

                # Early stopping if threshold met
                if criterion == 'explained_variance' and result.explained_variance >= threshold:
                    print(f"\n✓ Threshold {threshold} reached at rank {rank}")
                    return rank

            except Exception as e:
                print(f"✗ Rank {rank} failed: {e}")
                break

        if criterion == 'explained_variance':
            # Find first rank above threshold
            for i, var in enumerate(variances):
                if var >= threshold:
                    return i + 1
            return len(variances)  # Return max if threshold not met

        elif criterion == 'elbow':
            # Find elbow point in error curve
            errors = np.array(errors)
            # Compute second derivative
            second_deriv = np.diff(np.diff(errors))
            elbow_idx = np.argmax(second_deriv) + 2
            print(f"\n✓ Elbow detected at rank {elbow_idx}")
            return elbow_idx

        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def compare_methods(
        self,
        tensor: np.ndarray,
        rank: int,
        verbose: bool = True
    ) -> Dict[str, DecompositionResult]:
        """
        Compare CP, Tucker, and TT decompositions at same rank.

        Args:
            tensor: Input tensor
            rank: Rank to use for all methods
            verbose: Print comparison

        Returns:
            Dict of results by method
        """
        results = {}

        print(f"\n{'='*60}")
        print(f"COMPARING DECOMPOSITION METHODS (rank={rank})")
        print(f"{'='*60}")

        # CP
        try:
            results['cp'] = self.cp_decomposition(tensor, rank, verbose=verbose)
        except Exception as e:
            print(f"CP failed: {e}")

        # Tucker
        try:
            results['tucker'] = self.tucker_decomposition(tensor, rank, verbose=verbose)
        except Exception as e:
            print(f"Tucker failed: {e}")

        # Tensor Train
        try:
            results['tt'] = self.tensor_train_decomposition(tensor, rank, verbose=verbose)
        except Exception as e:
            print(f"TT failed: {e}")

        # Print comparison
        if verbose and results:
            print(f"\n{'='*60}")
            print("COMPARISON SUMMARY")
            print(f"{'='*60}")
            print(f"{'Method':<15} {'Error':<15} {'Explained Var':<15}")
            print(f"{'-'*60}")
            for method, result in results.items():
                print(f"{method:<15} {result.reconstruction_error:<15.4f} "
                      f"{result.explained_variance:<15.4f}")

        return results

    def extract_temporal_factors(self, result: DecompositionResult) -> np.ndarray:
        """Extract time-dimension factors (assumes time is first dimension)."""
        return result.factors[0]

    def extract_venue_factors(self, result: DecompositionResult) -> np.ndarray:
        """Extract venue-dimension factors (assumes venue is second dimension)."""
        return result.factors[1]

    def extract_asset_factors(self, result: DecompositionResult) -> np.ndarray:
        """Extract asset-dimension factors (assumes asset is third dimension)."""
        return result.factors[2]


if __name__ == "__main__":
    print("=== Testing Tensor Decomposition ===\n")

    # Create synthetic tensor
    np.random.seed(42)
    true_rank = 3
    time_dim, venue_dim, asset_dim, feature_dim = 100, 3, 5, 10

    # Generate low-rank tensor
    time_factor = np.random.randn(time_dim, true_rank)
    venue_factor = np.random.randn(venue_dim, true_rank)
    asset_factor = np.random.randn(asset_dim, true_rank)
    feature_factor = np.random.randn(feature_dim, true_rank)

    # Construct tensor from factors
    tensor = np.zeros((time_dim, venue_dim, asset_dim, feature_dim))
    for r in range(true_rank):
        rank1 = np.outer(time_factor[:, r], venue_factor[:, r])
        rank1 = np.outer(rank1.ravel(), asset_factor[:, r])
        rank1 = np.outer(rank1.ravel(), feature_factor[:, r])
        tensor += rank1.reshape(time_dim, venue_dim, asset_dim, feature_dim)

    # Add noise
    tensor += np.random.randn(*tensor.shape) * 0.1

    print(f"Synthetic Tensor Shape: {tensor.shape}")
    print(f"True Rank: {true_rank}\n")

    # Test decomposer
    decomposer = TensorDecomposer()

    # Test rank selection
    optimal_rank = decomposer.rank_selection(tensor, method='cp', max_rank=10, threshold=0.95)
    print(f"\nOptimal Rank Selected: {optimal_rank}")

    # Compare methods at optimal rank
    results = decomposer.compare_methods(tensor, rank=optimal_rank, verbose=True)

    # Analyze factors
    print(f"\n{'='*60}")
    print("FACTOR ANALYSIS")
    print(f"{'='*60}")

    if 'cp' in results:
        cp_result = results['cp']
        time_factors = decomposer.extract_temporal_factors(cp_result)
        print(f"\nTemporal Factors Shape: {time_factors.shape}")
        print(f"First 5 timepoints of first 3 factors:")
        print(time_factors[:5, :3])
