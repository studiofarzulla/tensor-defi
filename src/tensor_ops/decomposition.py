#!/usr/bin/env python3
"""
Tensor Decomposition for TENSOR-DEFI

Implements CP and Tucker decomposition to extract latent factors.
Produces N×R factor loading matrix for alignment testing.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorDecomposition:
    """Performs CP and Tucker decomposition on market tensor."""

    def __init__(self, tensor_dir: Path, output_dir: Path):
        self.tensor_dir = Path(tensor_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load tensor and metadata
        self.tensor = np.load(tensor_dir / "market_tensor.npy")
        with open(tensor_dir / "tensor_meta.json") as f:
            self.metadata = json.load(f)

        logger.info(f"Loaded tensor: {self.tensor.shape}")

    def explained_variance(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute explained variance ratio."""
        ss_total = np.sum((original - original.mean()) ** 2)
        ss_residual = np.sum((original - reconstructed) ** 2)
        return 1 - (ss_residual / ss_total)

    def cp_decomposition(
        self,
        rank: int,
        n_iter_max: int = 100,
        tol: float = 1e-6
    ) -> tuple[list[np.ndarray], float]:
        """
        Perform CP (CANDECOMP/PARAFAC) decomposition.

        Returns factor matrices for each mode and explained variance.
        """
        logger.info(f"Running CP decomposition with rank {rank}...")

        # Reshape to 3D if needed (drop venue dimension)
        tensor = self.tensor.squeeze()  # Remove singleton venue dim
        if tensor.ndim == 4:
            tensor = tensor[:, 0, :, :]  # Take first venue

        weights, factors = parafac(
            tensor,
            rank=rank,
            n_iter_max=n_iter_max,
            tol=tol,
            init='random',
            random_state=42
        )

        # Reconstruct for explained variance
        reconstructed = tl.cp_to_tensor((weights, factors))
        exp_var = self.explained_variance(tensor, reconstructed)

        logger.info(f"CP rank={rank}: explained variance = {exp_var:.4f}")

        return factors, exp_var

    def tucker_decomposition(
        self,
        ranks: list[int],
        n_iter_max: int = 100,
        tol: float = 1e-6
    ) -> tuple[np.ndarray, list[np.ndarray], float]:
        """
        Perform Tucker decomposition.

        Returns core tensor, factor matrices, and explained variance.
        """
        logger.info(f"Running Tucker decomposition with ranks {ranks}...")

        tensor = self.tensor.squeeze()
        if tensor.ndim == 4:
            tensor = tensor[:, 0, :, :]

        core, factors = tucker(
            tensor,
            rank=ranks,
            n_iter_max=n_iter_max,
            tol=tol,
            init='random',
            random_state=42
        )

        # Reconstruct
        reconstructed = tl.tucker_to_tensor((core, factors))
        exp_var = self.explained_variance(tensor, reconstructed)

        logger.info(f"Tucker ranks={ranks}: explained variance = {exp_var:.4f}")

        return core, factors, exp_var

    def find_optimal_rank(
        self,
        max_rank: int = 20,
        target_variance: float = 0.90
    ) -> int:
        """Find optimal CP rank via explained variance."""
        logger.info(f"Finding optimal rank (target variance = {target_variance})...")

        for rank in range(1, max_rank + 1):
            _, exp_var = self.cp_decomposition(rank)
            if exp_var >= target_variance:
                logger.info(f"Optimal rank: {rank} (explains {exp_var:.1%})")
                return rank

        logger.warning(f"Max rank {max_rank} only explains {exp_var:.1%}")
        return max_rank

    def extract_asset_factors(
        self,
        method: str = 'cp',
        rank: Optional[int] = None,
        target_variance: float = 0.90
    ) -> tuple[np.ndarray, list[str], dict]:
        """
        Extract N×R asset factor loading matrix.

        This is the key output for alignment testing.
        """
        # Auto-select rank if not specified
        if rank is None:
            rank = self.find_optimal_rank(target_variance=target_variance)

        if method == 'cp':
            factors, exp_var = self.cp_decomposition(rank)
            # Asset factor is mode 1 (0=time, 1=asset, 2=feature)
            asset_factors = factors[1]  # Shape: (N, R)
        else:
            # Tucker: use similar ranks for each mode
            ranks = [min(rank, self.tensor.shape[i]) for i in range(3)]
            core, factors, exp_var = self.tucker_decomposition(ranks)
            asset_factors = factors[1]

        # Get symbols
        symbols = self.metadata['symbols']

        # Save results
        np.save(self.output_dir / f"{method}_asset_factors.npy", asset_factors)

        results = {
            'method': method,
            'rank': rank,
            'explained_variance': exp_var,
            'shape': list(asset_factors.shape),
            'symbols': symbols
        }

        with open(self.output_dir / f"{method}_factors_meta.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Save as CSV for inspection
        import pandas as pd
        df = pd.DataFrame(
            asset_factors,
            index=symbols,
            columns=[f"factor_{i+1}" for i in range(asset_factors.shape[1])]
        )
        df.to_csv(self.output_dir / f"{method}_asset_factors.csv")

        self._print_summary(asset_factors, results)

        return asset_factors, symbols, results

    def _print_summary(self, factors: np.ndarray, results: dict):
        """Print decomposition summary."""
        print(f"\n{'='*60}")
        print("TENSOR DECOMPOSITION SUMMARY")
        print(f"{'='*60}")
        print(f"Method: {results['method'].upper()}")
        print(f"Rank: {results['rank']}")
        print(f"Explained variance: {results['explained_variance']:.1%}")
        print(f"Factor matrix shape: {factors.shape}")

        # Factor statistics
        print(f"\nFactor loadings statistics:")
        for i in range(factors.shape[1]):
            col = factors[:, i]
            print(f"  Factor {i+1}: min={col.min():.3f}  max={col.max():.3f}  std={col.std():.3f}")

        # Inter-factor correlations
        if factors.shape[1] > 1:
            corr = np.corrcoef(factors.T)
            print(f"\nInter-factor correlations:")
            for i in range(factors.shape[1]):
                for j in range(i + 1, factors.shape[1]):
                    print(f"  F{i+1} ↔ F{j+1}: {corr[i,j]:.3f}")

        print(f"{'='*60}")


def main():
    """Run tensor decomposition."""
    base_path = Path(__file__).parent.parent.parent
    decomp = TensorDecomposition(
        tensor_dir=base_path / "outputs" / "tensor",
        output_dir=base_path / "outputs" / "tensor"
    )

    # Run both methods
    decomp.extract_asset_factors(method='cp', target_variance=0.90)


if __name__ == "__main__":
    main()
