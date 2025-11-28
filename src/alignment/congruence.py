"""
Tucker's Congruence Coefficient

Measures factor similarity between two factor matrices.
The gold standard for comparing factor structures in psychometrics
and applied to tensor decomposition alignment testing.

Reference: Tucker (1951), Lorenzo-Seva & ten Berge (2006)

Interpretation:
- φ ≥ 0.95: Factors considered equivalent
- φ = 0.85-0.94: Fair similarity
- φ = 0.65-0.84: Some similarity
- φ < 0.65: Factors distinct
"""

import numpy as np
from typing import List, Tuple, Dict


def tuckers_congruence(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute Tucker's congruence coefficient between two vectors.
    
    φ(x,y) = Σxᵢyᵢ / √(Σxᵢ² × Σyᵢ²)
    
    This is equivalent to the cosine similarity between vectors,
    but has specific interpretation in factor analysis context.
    
    Args:
        x: First vector
        y: Second vector (same length as x)
    
    Returns:
        Congruence coefficient in [-1, 1]
        
    Interpretation:
        |φ| ≥ 0.95: Factors equivalent
        |φ| = 0.85-0.94: Fair similarity
        |φ| < 0.85: Factors distinct
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if len(x) != len(y):
        raise ValueError(f"Vector length mismatch: {len(x)} vs {len(y)}")
    
    # Compute congruence
    numerator = np.sum(x * y)
    denominator = np.sqrt(np.sum(x**2) * np.sum(y**2))
    
    if denominator < 1e-10:
        return 0.0
    
    return float(numerator / denominator)


def factor_similarity_matrix(
    A: np.ndarray,
    B: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise congruence coefficients between columns of two matrices.
    
    Creates a k_A × k_B matrix where entry (i,j) is the congruence
    between column i of A and column j of B.
    
    Args:
        A: First factor matrix (n × k_A)
        B: Second factor matrix (n × k_B)
    
    Returns:
        Congruence matrix (k_A × k_B)
    """
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"Row count mismatch: {A.shape[0]} vs {B.shape[0]}")
    
    k_A = A.shape[1]
    k_B = B.shape[1]
    
    similarity = np.zeros((k_A, k_B))
    
    for i in range(k_A):
        for j in range(k_B):
            similarity[i, j] = tuckers_congruence(A[:, i], B[:, j])
    
    return similarity


def mean_congruence(
    A: np.ndarray,
    B: np.ndarray,
    method: str = "diagonal"
) -> float:
    """
    Compute mean congruence between two factor matrices.
    
    Args:
        A: First factor matrix (n × k)
        B: Second factor matrix (n × k)
        method: 
            "diagonal" - Mean of diagonal elements (assumes matched columns)
            "best_match" - Mean of best matching pairs
            "all" - Mean of all pairwise congruences
    
    Returns:
        Mean congruence coefficient
    """
    sim_matrix = factor_similarity_matrix(A, B)
    
    if method == "diagonal":
        # Assume columns are already aligned
        k = min(A.shape[1], B.shape[1])
        return float(np.mean(np.diag(sim_matrix)[:k]))
    
    elif method == "best_match":
        # Hungarian algorithm would be ideal, but simple greedy for now
        k = min(A.shape[1], B.shape[1])
        used_j = set()
        matches = []
        
        for i in range(k):
            best_j = -1
            best_sim = -2
            for j in range(sim_matrix.shape[1]):
                if j not in used_j and abs(sim_matrix[i, j]) > best_sim:
                    best_sim = abs(sim_matrix[i, j])
                    best_j = j
            if best_j >= 0:
                used_j.add(best_j)
                matches.append(sim_matrix[i, best_j])
        
        return float(np.mean(matches)) if matches else 0.0
    
    elif method == "all":
        return float(np.mean(np.abs(sim_matrix)))
    
    else:
        raise ValueError(f"Unknown method: {method}")


def interpret_congruence(phi: float) -> str:
    """
    Interpret a congruence coefficient.
    
    Based on Lorenzo-Seva & ten Berge (2006) guidelines.
    
    Args:
        phi: Congruence coefficient
    
    Returns:
        Human-readable interpretation
    """
    phi_abs = abs(phi)
    
    if phi_abs >= 0.95:
        return "Equivalent (φ ≥ 0.95)"
    elif phi_abs >= 0.85:
        return "Fair similarity (0.85 ≤ φ < 0.95)"
    elif phi_abs >= 0.65:
        return "Some similarity (0.65 ≤ φ < 0.85)"
    else:
        return "Distinct (φ < 0.65)"


def congruence_analysis(
    claims_factors: np.ndarray,
    market_factors: np.ndarray,
    apply_procrustes: bool = True
) -> Dict:
    """
    Full congruence analysis between claims and market factor matrices.
    
    Args:
        claims_factors: Entity loadings from claims tensor decomposition (n × k)
        market_factors: Entity loadings from market tensor decomposition (n × k)
        apply_procrustes: Whether to apply Procrustes rotation before comparison
    
    Returns:
        Dict with comprehensive congruence statistics
    """
    from .procrustes import align_factor_matrices
    
    # Optionally align using Procrustes
    if apply_procrustes:
        aligned_claims, rotation, disparity = align_factor_matrices(
            claims_factors, market_factors, normalize=True
        )
    else:
        aligned_claims = claims_factors
        rotation = np.eye(claims_factors.shape[1])
        disparity = np.linalg.norm(claims_factors - market_factors, 'fro')
    
    # Compute similarity matrix
    sim_matrix = factor_similarity_matrix(aligned_claims, market_factors)
    
    # Compute various congruence measures
    diagonal_congruence = mean_congruence(aligned_claims, market_factors, method="diagonal")
    best_match_congruence = mean_congruence(aligned_claims, market_factors, method="best_match")
    
    # Per-factor congruences
    k = min(aligned_claims.shape[1], market_factors.shape[1])
    per_factor = [tuckers_congruence(aligned_claims[:, i], market_factors[:, i]) 
                  for i in range(k)]
    
    return {
        "similarity_matrix": sim_matrix,
        "diagonal_congruence": diagonal_congruence,
        "best_match_congruence": best_match_congruence,
        "per_factor_congruence": per_factor,
        "interpretation": interpret_congruence(diagonal_congruence),
        "procrustes_disparity": disparity,
        "rotation_matrix": rotation,
        "aligned_claims_factors": aligned_claims,
    }


if __name__ == "__main__":
    print("=== Tucker's Congruence Coefficient Test ===\n")
    
    np.random.seed(42)
    
    # Test 1: Identical vectors
    x = np.random.randn(10)
    phi = tuckers_congruence(x, x)
    print(f"Identical vectors: φ = {phi:.4f} ({interpret_congruence(phi)})")
    
    # Test 2: Orthogonal vectors
    y = np.random.randn(10)
    y = y - (np.dot(x, y) / np.dot(x, x)) * x  # Gram-Schmidt
    phi = tuckers_congruence(x, y)
    print(f"Orthogonal vectors: φ = {phi:.4f} ({interpret_congruence(phi)})")
    
    # Test 3: Opposite vectors
    phi = tuckers_congruence(x, -x)
    print(f"Opposite vectors: φ = {phi:.4f}")
    
    # Test 4: Factor matrices
    print("\n=== Factor Matrix Similarity ===")
    
    n, k = 8, 4  # 8 entities, 4 factors
    
    # Simulated claims factors
    claims = np.random.randn(n, k)
    
    # Market factors with some relationship + noise
    market = claims @ np.random.randn(k, k) * 0.5 + np.random.randn(n, k) * 0.3
    
    sim_matrix = factor_similarity_matrix(claims, market)
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    print(f"Similarity matrix:\n{np.round(sim_matrix, 2)}")
    
    # Mean congruences
    print(f"\nDiagonal mean: {mean_congruence(claims, market, 'diagonal'):.3f}")
    print(f"Best match mean: {mean_congruence(claims, market, 'best_match'):.3f}")
    print(f"All pairs mean: {mean_congruence(claims, market, 'all'):.3f}")
    
    # Full analysis
    print("\n=== Full Congruence Analysis ===")
    results = congruence_analysis(claims, market, apply_procrustes=True)
    print(f"Diagonal congruence: {results['diagonal_congruence']:.3f}")
    print(f"Best match congruence: {results['best_match_congruence']:.3f}")
    print(f"Per-factor: {[f'{p:.3f}' for p in results['per_factor_congruence']]}")
    print(f"Interpretation: {results['interpretation']}")
