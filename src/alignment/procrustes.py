"""
Orthogonal Procrustes Rotation

Aligns two factor matrices by finding the optimal orthogonal transformation
that minimizes the distance between them. Essential for comparing factor
structures from different tensor decompositions.

Reference: Korth & Tucker (1976)
"""

import numpy as np
from typing import Tuple


def orthogonal_procrustes(
    A: np.ndarray,
    B: np.ndarray,
    check_finite: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Compute the orthogonal Procrustes transformation.
    
    Finds the orthogonal matrix R that minimizes ||A @ R - B||_F
    
    This is used to optimally rotate factor matrix A to align with B
    before computing congruence coefficients.
    
    Args:
        A: Source matrix (n x k) to be rotated
        B: Target matrix (n x k) to align with
        check_finite: Check for NaN/Inf values
    
    Returns:
        R: Orthogonal rotation matrix (k x k)
        scale: Optimal scaling factor (not used in orthogonal version)
    
    Mathematical formulation:
        R = V @ U.T
        where U, S, V = SVD(B.T @ A)
    """
    if check_finite:
        if not np.isfinite(A).all():
            raise ValueError("Input matrix A contains NaN or Inf")
        if not np.isfinite(B).all():
            raise ValueError("Input matrix B contains NaN or Inf")
    
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: A={A.shape}, B={B.shape}")
    
    # SVD of B.T @ A
    M = B.T @ A
    U, S, Vt = np.linalg.svd(M)
    
    # Optimal rotation matrix
    R = Vt.T @ U.T
    
    # Compute disparity (Frobenius norm of residual)
    aligned_A = A @ R
    disparity = np.linalg.norm(aligned_A - B, 'fro')
    
    return R, disparity


def align_factor_matrices(
    source: np.ndarray,
    target: np.ndarray,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Align source factor matrix to target using Procrustes rotation.
    
    Args:
        source: Factor matrix to be aligned (n x k)
        target: Target factor matrix (n x k)
        normalize: Whether to column-normalize before alignment
    
    Returns:
        aligned_source: Rotated source matrix
        rotation: The rotation matrix used
        disparity: Alignment error (lower is better)
    """
    # Optionally normalize columns to unit length
    if normalize:
        source_norm = source / (np.linalg.norm(source, axis=0, keepdims=True) + 1e-10)
        target_norm = target / (np.linalg.norm(target, axis=0, keepdims=True) + 1e-10)
    else:
        source_norm = source
        target_norm = target
    
    # Find optimal rotation
    R, disparity = orthogonal_procrustes(source_norm, target_norm)
    
    # Apply rotation to original (non-normalized) source
    aligned = source @ R
    
    return aligned, R, disparity


def procrustes_analysis(
    A: np.ndarray,
    B: np.ndarray
) -> dict:
    """
    Full Procrustes analysis between two matrices.
    
    Returns comprehensive alignment statistics.
    
    Args:
        A: First matrix (n x k)
        B: Second matrix (n x k)
    
    Returns:
        Dict with alignment statistics
    """
    # Align A to B
    aligned_A, R, disparity = align_factor_matrices(A, B)
    
    # Compute various metrics
    
    # Frobenius norm of difference
    frob_diff = np.linalg.norm(aligned_A - B, 'fro')
    
    # Relative difference
    rel_diff = frob_diff / (np.linalg.norm(B, 'fro') + 1e-10)
    
    # Column-wise cosine similarities
    col_similarities = []
    for i in range(A.shape[1]):
        a_col = aligned_A[:, i]
        b_col = B[:, i]
        
        norm_a = np.linalg.norm(a_col)
        norm_b = np.linalg.norm(b_col)
        
        if norm_a > 0 and norm_b > 0:
            sim = np.dot(a_col, b_col) / (norm_a * norm_b)
        else:
            sim = 0.0
        col_similarities.append(sim)
    
    return {
        "rotation_matrix": R,
        "aligned_source": aligned_A,
        "disparity": disparity,
        "frobenius_difference": frob_diff,
        "relative_difference": rel_diff,
        "column_similarities": col_similarities,
        "mean_column_similarity": np.mean(col_similarities),
    }


if __name__ == "__main__":
    print("=== Procrustes Rotation Test ===\n")
    
    np.random.seed(42)
    
    # Create test matrices
    n, k = 10, 3
    
    # Target matrix
    B = np.random.randn(n, k)
    
    # Source matrix (rotated + noise version of B)
    true_R = np.linalg.qr(np.random.randn(k, k))[0]  # Random orthogonal
    noise = np.random.randn(n, k) * 0.1
    A = B @ true_R.T + noise
    
    print(f"Matrix shape: {n} x {k}")
    print(f"True rotation applied: {true_R.shape}")
    
    # Test alignment
    aligned_A, R, disparity = align_factor_matrices(A, B)
    
    print(f"\nAlignment Results:")
    print(f"  Disparity: {disparity:.4f}")
    print(f"  R @ R.T (should be identity):\n{np.round(R @ R.T, 3)}")
    
    # Full analysis
    results = procrustes_analysis(A, B)
    print(f"\nFull Analysis:")
    print(f"  Frobenius difference: {results['frobenius_difference']:.4f}")
    print(f"  Relative difference: {results['relative_difference']:.4f}")
    print(f"  Mean column similarity: {results['mean_column_similarity']:.4f}")
    print(f"  Column similarities: {[f'{s:.3f}' for s in results['column_similarities']]}")
