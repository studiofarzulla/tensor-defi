"""
Alignment Test Module

Main class for testing alignment between functional claims tensor
and market dynamics tensor using tensor decomposition.

This is the core research question:
"Do crypto projects' functional claims align with their market behavior?"

"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from .procrustes import align_factor_matrices, procrustes_analysis
from .congruence import (
    tuckers_congruence, 
    factor_similarity_matrix, 
    mean_congruence,
    interpret_congruence,
    congruence_analysis
)


@dataclass
class AlignmentResult:
    """Results from alignment testing."""
    
    # Core metrics
    congruence_coefficient: float
    interpretation: str
    procrustes_disparity: float
    
    # Per-factor details
    per_factor_congruence: List[float]
    similarity_matrix: np.ndarray
    
    # Factor information
    claims_factors: np.ndarray
    market_factors: np.ndarray
    aligned_claims_factors: np.ndarray
    rotation_matrix: np.ndarray
    
    # Metadata
    n_entities: int
    n_factors: int
    entity_labels: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "congruence_coefficient": float(self.congruence_coefficient),
            "interpretation": self.interpretation,
            "procrustes_disparity": float(self.procrustes_disparity),
            "per_factor_congruence": [float(x) for x in self.per_factor_congruence],
            "n_entities": self.n_entities,
            "n_factors": self.n_factors,
            "entity_labels": self.entity_labels,
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "ALIGNMENT TEST RESULTS",
            "=" * 60,
            f"",
            f"Tucker's Congruence Coefficient: {self.congruence_coefficient:.4f}",
            f"Interpretation: {self.interpretation}",
            f"",
            f"Procrustes Disparity: {self.procrustes_disparity:.4f}",
            f"Entities: {self.n_entities}",
            f"Factors: {self.n_factors}",
            f"",
            "Per-Factor Congruence:",
        ]
        
        for i, phi in enumerate(self.per_factor_congruence):
            interp = interpret_congruence(phi)
            lines.append(f"  Factor {i+1}: φ = {phi:.4f} ({interp})")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


class AlignmentTest:
    """
    Test alignment between functional claims and market dynamics.
    
    Uses tensor decomposition to extract factor structures from both
    claims and market tensors, then compares using Tucker's congruence.
    """
    
    def __init__(
        self,
        n_factors: int = 4,
        apply_procrustes: bool = True,
        normalize_factors: bool = True,
    ):
        """
        Initialize alignment test.
        
        Args:
            n_factors: Number of factors to extract (rank for decomposition)
            apply_procrustes: Apply Procrustes rotation before comparison
            normalize_factors: Normalize factor columns to unit length
        """
        self.n_factors = n_factors
        self.apply_procrustes = apply_procrustes
        self.normalize_factors = normalize_factors
    
    def test_matrix_alignment(
        self,
        claims_matrix: np.ndarray,
        market_matrix: np.ndarray,
        entity_labels: Optional[List[str]] = None,
    ) -> AlignmentResult:
        """
        Test alignment between claims and market feature matrices.
        
        For the PoC, we directly compare feature matrices without
        full tensor decomposition. This tests whether functional
        profiles correlate with market characteristics.
        
        Args:
            claims_matrix: Entity × Features matrix from NLP pipeline
            market_matrix: Entity × Features matrix from market data
            entity_labels: Labels for entities (e.g., ['BTC', 'ETH', ...])
        
        Returns:
            AlignmentResult with congruence statistics
        """
        n_entities = claims_matrix.shape[0]
        
        if entity_labels is None:
            entity_labels = [f"Entity_{i}" for i in range(n_entities)]
        
        # Ensure same number of entities
        if claims_matrix.shape[0] != market_matrix.shape[0]:
            raise ValueError(
                f"Entity count mismatch: {claims_matrix.shape[0]} vs {market_matrix.shape[0]}"
            )
        
        # Determine number of factors (min of feature dimensions)
        k = min(claims_matrix.shape[1], market_matrix.shape[1], self.n_factors)
        
        # Use PCA to reduce to same dimensionality if needed
        claims_reduced = self._reduce_dimensions(claims_matrix, k)
        market_reduced = self._reduce_dimensions(market_matrix, k)
        
        # Run congruence analysis
        results = congruence_analysis(
            claims_reduced, 
            market_reduced,
            apply_procrustes=self.apply_procrustes
        )
        
        return AlignmentResult(
            congruence_coefficient=results["diagonal_congruence"],
            interpretation=results["interpretation"],
            procrustes_disparity=results["procrustes_disparity"],
            per_factor_congruence=results["per_factor_congruence"],
            similarity_matrix=results["similarity_matrix"],
            claims_factors=claims_reduced,
            market_factors=market_reduced,
            aligned_claims_factors=results["aligned_claims_factors"],
            rotation_matrix=results["rotation_matrix"],
            n_entities=n_entities,
            n_factors=k,
            entity_labels=entity_labels,
        )
    
    def test_tensor_alignment(
        self,
        claims_tensor: np.ndarray,
        market_tensor: np.ndarray,
        entity_labels: Optional[List[str]] = None,
    ) -> AlignmentResult:
        """
        Test alignment between claims and market tensors using CP decomposition.
        
        Decomposes both tensors and compares entity factor loadings.
        
        Args:
            claims_tensor: Entities × Time × Features tensor from NLP
            market_tensor: Entities × Time × Features tensor from market
            entity_labels: Entity labels
        
        Returns:
            AlignmentResult with congruence statistics
        """
        # Import tensorly for decomposition
        try:
            import tensorly as tl
            from tensorly.decomposition import parafac
        except ImportError:
            raise ImportError("tensorly required for tensor decomposition")
        
        n_entities = claims_tensor.shape[0]
        
        if entity_labels is None:
            entity_labels = [f"Entity_{i}" for i in range(n_entities)]
        
        # Decompose claims tensor
        print(f"Decomposing claims tensor {claims_tensor.shape}...")
        claims_weights, claims_factors = parafac(
            tl.tensor(claims_tensor),
            rank=self.n_factors,
            init='random',
            n_iter_max=100
        )
        claims_entity_loadings = tl.to_numpy(claims_factors[0])  # Entity dimension
        
        # Decompose market tensor
        print(f"Decomposing market tensor {market_tensor.shape}...")
        market_weights, market_factors = parafac(
            tl.tensor(market_tensor),
            rank=self.n_factors,
            init='random',
            n_iter_max=100
        )
        market_entity_loadings = tl.to_numpy(market_factors[0])  # Entity dimension
        
        # Run congruence analysis on entity loadings
        results = congruence_analysis(
            claims_entity_loadings,
            market_entity_loadings,
            apply_procrustes=self.apply_procrustes
        )
        
        return AlignmentResult(
            congruence_coefficient=results["diagonal_congruence"],
            interpretation=results["interpretation"],
            procrustes_disparity=results["procrustes_disparity"],
            per_factor_congruence=results["per_factor_congruence"],
            similarity_matrix=results["similarity_matrix"],
            claims_factors=claims_entity_loadings,
            market_factors=market_entity_loadings,
            aligned_claims_factors=results["aligned_claims_factors"],
            rotation_matrix=results["rotation_matrix"],
            n_entities=n_entities,
            n_factors=self.n_factors,
            entity_labels=entity_labels,
        )
    
    def _reduce_dimensions(self, matrix: np.ndarray, k: int) -> np.ndarray:
        """Reduce matrix to k dimensions using PCA."""
        if matrix.shape[1] <= k:
            # Pad with zeros if needed
            if matrix.shape[1] < k:
                padding = np.zeros((matrix.shape[0], k - matrix.shape[1]))
                return np.hstack([matrix, padding])
            return matrix
        
        # Simple PCA via SVD
        centered = matrix - matrix.mean(axis=0)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # Take top k components
        return U[:, :k] * S[:k]
    
    def bootstrap_confidence_interval(
        self,
        claims_matrix: np.ndarray,
        market_matrix: np.ndarray,
        n_bootstrap: int = 100,
        confidence: float = 0.95,
        entity_labels: Optional[List[str]] = None,
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for congruence coefficient.
        
        Args:
            claims_matrix: Claims feature matrix
            market_matrix: Market feature matrix
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            entity_labels: Entity labels
        
        Returns:
            (point_estimate, lower_bound, upper_bound)
        """
        n_entities = claims_matrix.shape[0]
        congruences = []
        
        for _ in range(n_bootstrap):
            # Sample entities with replacement
            indices = np.random.choice(n_entities, size=n_entities, replace=True)
            
            claims_sample = claims_matrix[indices]
            market_sample = market_matrix[indices]
            
            # Compute congruence
            result = self.test_matrix_alignment(
                claims_sample, 
                market_sample,
                entity_labels=[f"Entity_{i}" for i in range(n_entities)]
            )
            congruences.append(result.congruence_coefficient)
        
        congruences = np.array(congruences)
        
        # Point estimate from full data
        full_result = self.test_matrix_alignment(claims_matrix, market_matrix, entity_labels)
        point_estimate = full_result.congruence_coefficient
        
        # Confidence interval
        alpha = 1 - confidence
        lower = np.percentile(congruences, 100 * alpha / 2)
        upper = np.percentile(congruences, 100 * (1 - alpha / 2))
        
        return point_estimate, lower, upper


def run_alignment_test_from_files(
    claims_path: str = "outputs/nlp/claims_matrix.npz",
    market_path: str = "outputs/market/market_matrix.npz",
    output_path: str = "outputs/alignment/results.json",
) -> AlignmentResult:
    """
    Run alignment test from saved matrix files.
    
    Args:
        claims_path: Path to claims matrix .npz file
        market_path: Path to market matrix .npz file
        output_path: Path to save results
    
    Returns:
        AlignmentResult
    """
    # Load claims matrix
    claims_data = np.load(claims_path, allow_pickle=True)
    claims_matrix = claims_data["matrix"]
    symbols = list(claims_data["symbols"])
    
    # Load market matrix
    market_data = np.load(market_path, allow_pickle=True)
    market_matrix = market_data["matrix"]
    
    # Run alignment test
    test = AlignmentTest(n_factors=4)
    result = test.test_matrix_alignment(
        claims_matrix,
        market_matrix,
        entity_labels=symbols
    )
    
    # Save results
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print(result.summary())
    
    return result


if __name__ == "__main__":
    print("=== Alignment Test Demo ===\n")
    
    np.random.seed(42)
    
    # Create synthetic data
    n_entities = 8
    n_claims_features = 10
    n_market_features = 7
    
    # Simulated claims matrix (from NLP pipeline)
    claims = np.random.randn(n_entities, n_claims_features)
    
    # Simulated market matrix with some relationship to claims
    # This simulates partial alignment
    transformation = np.random.randn(n_claims_features, n_market_features) * 0.5
    noise = np.random.randn(n_entities, n_market_features) * 0.5
    market = claims @ transformation + noise
    
    entity_labels = ["BTC", "ETH", "SOL", "AVAX", "DOT", "FIL", "LINK", "ALGO"]
    
    # Run alignment test
    test = AlignmentTest(n_factors=4)
    result = test.test_matrix_alignment(claims, market, entity_labels)
    
    print(result.summary())
    
    # Bootstrap confidence interval
    print("\n=== Bootstrap Confidence Interval ===")
    point, lower, upper = test.bootstrap_confidence_interval(
        claims, market, n_bootstrap=50, entity_labels=entity_labels
    )
    print(f"Point estimate: {point:.4f}")
    print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
