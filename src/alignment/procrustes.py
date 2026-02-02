#!/usr/bin/env python3
"""
Procrustes Analysis for TENSOR-DEFI

Implements orthogonal Procrustes rotation to align heterogeneous spaces.
Handles dimension mismatch between claims (10), stats (7), and factors (R).
"""

import logging
from typing import Optional

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcrustesAlignment:
    """Aligns matrices via orthogonal Procrustes transformation."""

    def __init__(self):
        pass

    def pad_to_match(self, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Pad smaller matrix with zeros to match dimensions.

        For alignment, we need same number of columns.
        """
        n_cols_A = A.shape[1]
        n_cols_B = B.shape[1]

        if n_cols_A == n_cols_B:
            return A, B

        max_cols = max(n_cols_A, n_cols_B)

        if n_cols_A < max_cols:
            padding = np.zeros((A.shape[0], max_cols - n_cols_A))
            A = np.hstack([A, padding])

        if n_cols_B < max_cols:
            padding = np.zeros((B.shape[0], max_cols - n_cols_B))
            B = np.hstack([B, padding])

        return A, B

    def reduce_to_common(
        self,
        A: np.ndarray,
        B: np.ndarray,
        method: str = 'pca'
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reduce both matrices to common dimensionality via PCA or truncation.
        """
        min_cols = min(A.shape[1], B.shape[1])

        if method == 'truncate':
            return A[:, :min_cols], B[:, :min_cols]

        elif method == 'pca':
            from sklearn.decomposition import PCA

            # Fit PCA on concatenated data for common basis
            combined = np.vstack([A, B])
            pca = PCA(n_components=min_cols)
            pca.fit(combined)

            A_reduced = pca.transform(A)
            B_reduced = pca.transform(B)

            return A_reduced, B_reduced

        else:
            raise ValueError(f"Unknown method: {method}")

    def orthogonal_procrustes(
        self,
        A: np.ndarray,
        B: np.ndarray,
        scale: bool = True
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Find optimal orthogonal rotation R to align A to B.

        Minimizes ||A @ R - B||² subject to R^T @ R = I

        Returns:
            R: Rotation matrix
            A_rotated: A after rotation (A @ R)
            disparity: Residual sum of squares
        """
        # Center matrices
        A_centered = A - A.mean(axis=0)
        B_centered = B - B.mean(axis=0)

        if scale:
            # Scale to unit Frobenius norm
            A_scale = np.linalg.norm(A_centered, 'fro')
            B_scale = np.linalg.norm(B_centered, 'fro')
            A_centered = A_centered / A_scale if A_scale > 0 else A_centered
            B_centered = B_centered / B_scale if B_scale > 0 else B_centered

        # Compute optimal rotation
        R, scale_factor = orthogonal_procrustes(A_centered, B_centered)

        # Apply rotation
        A_rotated = A_centered @ R

        # Compute disparity (Procrustes distance)
        disparity = np.sum((A_rotated - B_centered) ** 2)

        return R, A_rotated, disparity

    def align_matrices(
        self,
        source: np.ndarray,
        target: np.ndarray,
        source_name: str = "source",
        target_name: str = "target",
        handle_dims: str = 'pad'
    ) -> dict:
        """
        Full alignment pipeline between two matrices.

        Args:
            source: Source matrix (N × D1)
            target: Target matrix (N × D2)
            handle_dims: 'pad' to zero-pad, 'reduce' to use PCA

        Returns:
            Dictionary with alignment results
        """
        logger.info(f"Aligning {source_name} {source.shape} → {target_name} {target.shape}")

        # Handle dimension mismatch
        if source.shape[1] != target.shape[1]:
            if handle_dims == 'pad':
                source_adj, target_adj = self.pad_to_match(source, target)
            else:
                source_adj, target_adj = self.reduce_to_common(source, target)
            logger.info(f"Adjusted to common shape: {source_adj.shape}")
        else:
            source_adj, target_adj = source, target

        # Ensure same number of rows (entities)
        if source_adj.shape[0] != target_adj.shape[0]:
            raise ValueError(
                f"Row mismatch: {source_name} has {source_adj.shape[0]}, "
                f"{target_name} has {target_adj.shape[0]}"
            )

        # Perform Procrustes alignment
        R, source_rotated, disparity = self.orthogonal_procrustes(source_adj, target_adj)

        results = {
            'source_name': source_name,
            'target_name': target_name,
            'source_shape': list(source.shape),
            'target_shape': list(target.shape),
            'aligned_shape': list(source_adj.shape),
            'rotation_matrix': R,
            'source_rotated': source_rotated,
            'target_centered': target_adj - target_adj.mean(axis=0),
            'disparity': float(disparity),
            'procrustes_distance': float(np.sqrt(disparity))
        }

        return results


def main():
    """Test Procrustes alignment."""
    # Create test matrices
    np.random.seed(42)
    A = np.random.randn(50, 10)  # Claims: 50 entities, 10 categories
    B = np.random.randn(50, 7)   # Stats: 50 entities, 7 features

    aligner = ProcrustesAlignment()
    results = aligner.align_matrices(A, B, "claims", "stats")

    print(f"Disparity: {results['disparity']:.4f}")
    print(f"Procrustes distance: {results['procrustes_distance']:.4f}")


if __name__ == "__main__":
    main()
