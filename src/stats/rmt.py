#!/usr/bin/env python3
"""
Random Matrix Theory (RMT) analysis for claims matrices.

Uses Marchenko-Pastur distribution and Tracy-Widom test to determine
how many "real" factors exist vs noise.

The idea: eigenvalues of a random matrix follow known distributions.
Eigenvalues that exceed these bounds indicate genuine signal structure.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def marchenko_pastur_bounds(n: int, p: int, sigma: float = 1.0) -> tuple[float, float]:
    """
    Compute Marchenko-Pastur distribution bounds.

    For a random matrix of shape (n, p), eigenvalues of X @ X.T / p
    lie within [lambda_minus, lambda_plus] with probability 1 as n,p -> inf.

    Args:
        n: Number of rows (entities)
        p: Number of columns (features)
        sigma: Standard deviation of elements (default 1 for standardized data)

    Returns:
        (lambda_minus, lambda_plus) - MP bounds
    """
    q = n / p  # Aspect ratio

    if q > 1:
        # More rows than columns: use p/n instead
        q = p / n

    lambda_plus = sigma**2 * (1 + np.sqrt(q))**2
    lambda_minus = sigma**2 * (1 - np.sqrt(q))**2

    return lambda_minus, lambda_plus


def marchenko_pastur_pdf(x: np.ndarray, n: int, p: int, sigma: float = 1.0) -> np.ndarray:
    """
    Compute Marchenko-Pastur probability density.

    Args:
        x: Eigenvalue values to evaluate
        n: Number of rows
        p: Number of columns
        sigma: Standard deviation

    Returns:
        PDF values at x
    """
    q = min(n / p, p / n)
    lambda_minus, lambda_plus = marchenko_pastur_bounds(n, p, sigma)

    pdf = np.zeros_like(x, dtype=float)
    mask = (x >= lambda_minus) & (x <= lambda_plus)

    pdf[mask] = (
        np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus))
        / (2 * np.pi * sigma**2 * q * x[mask])
    )

    return pdf


def test_against_mp(
    matrix: np.ndarray,
    standardize: bool = True
) -> dict:
    """
    Test matrix eigenvalues against Marchenko-Pastur null.

    Args:
        matrix: Data matrix (n_entities, n_features)
        standardize: Z-score standardize before analysis

    Returns:
        Dict with eigenvalues, bounds, n_signal, etc.
    """
    n, p = matrix.shape

    # Standardize if requested
    if standardize:
        matrix = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-10)

    # Compute sample covariance eigenvalues
    # For n < p, use X @ X.T; for n >= p, use X.T @ X
    if n <= p:
        cov = matrix @ matrix.T / p
    else:
        cov = matrix.T @ matrix / n

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

    # MP bounds
    lambda_minus, lambda_plus = marchenko_pastur_bounds(n, p)

    # Count signal eigenvalues (above upper bound)
    n_signal = np.sum(eigenvalues > lambda_plus)

    # Also check below lower bound (rare but possible)
    n_below = np.sum(eigenvalues < lambda_minus)

    # Variance explained by signal vs noise
    total_var = eigenvalues.sum()
    signal_var = eigenvalues[eigenvalues > lambda_plus].sum()
    noise_var = eigenvalues[eigenvalues <= lambda_plus].sum()

    return {
        'eigenvalues': eigenvalues,
        'lambda_plus': lambda_plus,
        'lambda_minus': lambda_minus,
        'n_signal': n_signal,
        'n_below': n_below,
        'n_total': len(eigenvalues),
        'signal_variance_ratio': signal_var / total_var if total_var > 0 else 0,
        'noise_variance_ratio': noise_var / total_var if total_var > 0 else 0,
        'matrix_shape': (n, p)
    }


def tracy_widom_test(
    largest_eigenvalue: float,
    n: int,
    p: int,
    sigma: float = 1.0,
    alpha: float = 0.05
) -> dict:
    """
    Tracy-Widom test for largest eigenvalue significance.

    Tests H0: largest eigenvalue is consistent with MP distribution
    vs H1: largest eigenvalue indicates genuine signal.

    Args:
        largest_eigenvalue: The largest observed eigenvalue
        n: Number of rows
        p: Number of columns
        sigma: Standard deviation of elements
        alpha: Significance level

    Returns:
        Dict with z-score, p-value, significant flag
    """
    # Tracy-Widom centering and scaling for largest eigenvalue
    # Following Johnstone (2001) for real symmetric case
    gamma = min(n, p) / max(n, p)

    # Expected value and std of largest eigenvalue under null
    mu = sigma**2 * (np.sqrt(n) + np.sqrt(p))**2 / max(n, p)
    scale = sigma**2 * (np.sqrt(n) + np.sqrt(p)) * (
        1/np.sqrt(n) + 1/np.sqrt(p)
    )**(1/3) / max(n, p)**(2/3)

    # Standardized statistic
    z = (largest_eigenvalue - mu) / scale

    # Tracy-Widom Type 1 distribution (GOE)
    # Approximation using Gumbel (common for extreme value)
    # More accurate would use scipy.stats.tracy_widom if available
    try:
        from scipy.stats import gumbel_r
        # TW1 is approximately shifted Gumbel
        # Mean ~ -1.2065, SD ~ 1.268
        tw_mean = -1.2065
        tw_std = 1.268
        p_value = 1 - gumbel_r.cdf((z - tw_mean) / tw_std)
    except ImportError:
        # Fallback: use normal approximation (less accurate)
        p_value = 1 - stats.norm.cdf(z)

    return {
        'z_score': z,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
        'mu': mu,
        'scale': scale
    }


def eigenvalue_ratio_test(eigenvalues: np.ndarray) -> dict:
    """
    Test eigenvalue ratios against random matrix null.

    Under null (random matrix), consecutive eigenvalue ratios
    follow a specific distribution. Large ratios indicate signal.

    Args:
        eigenvalues: Sorted eigenvalues (descending)

    Returns:
        Dict with ratios and statistics
    """
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Compute ratios lambda_i / lambda_{i+1}
    ratios = eigenvalues[:-1] / (eigenvalues[1:] + 1e-10)

    # For random matrices, ratios cluster near 1
    # Large ratio (>2-3) suggests a "gap" indicating signal
    max_ratio = ratios.max()
    max_ratio_idx = ratios.argmax()

    # Number of factors before biggest gap
    n_factors_gap = max_ratio_idx + 1

    return {
        'ratios': ratios,
        'max_ratio': max_ratio,
        'max_ratio_position': max_ratio_idx,
        'suggested_n_factors': n_factors_gap
    }


def full_rmt_analysis(
    matrix: np.ndarray,
    name: str = "claims",
    standardize: bool = True,
    alpha: float = 0.05
) -> dict:
    """
    Run full RMT analysis suite on a matrix.

    Args:
        matrix: Data matrix (n_entities, n_features)
        name: Name for reporting
        standardize: Z-score standardize
        alpha: Significance level

    Returns:
        Comprehensive RMT analysis results
    """
    n, p = matrix.shape

    # 1. Marchenko-Pastur test
    mp_result = test_against_mp(matrix, standardize)

    # 2. Tracy-Widom test on largest eigenvalue
    tw_result = tracy_widom_test(
        mp_result['eigenvalues'][0],
        n, p, alpha=alpha
    )

    # 3. Eigenvalue ratio analysis
    ratio_result = eigenvalue_ratio_test(mp_result['eigenvalues'])

    # 4. Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"RMT ANALYSIS: {name}")
    logger.info(f"{'='*60}")
    logger.info(f"Matrix shape: {n} x {p}")
    logger.info(f"MP bounds: [{mp_result['lambda_minus']:.4f}, {mp_result['lambda_plus']:.4f}]")
    logger.info(f"Eigenvalues above MP upper bound: {mp_result['n_signal']}")
    logger.info(f"Signal variance ratio: {mp_result['signal_variance_ratio']:.2%}")
    logger.info(f"Tracy-Widom p-value: {tw_result['p_value']:.4f} {'***' if tw_result['significant'] else ''}")
    logger.info(f"Largest eigenvalue ratio: {ratio_result['max_ratio']:.2f} at position {ratio_result['max_ratio_position']}")
    logger.info(f"Suggested factors (gap method): {ratio_result['suggested_n_factors']}")
    logger.info(f"{'='*60}")

    return {
        'name': name,
        'mp': mp_result,
        'tw': tw_result,
        'ratio': ratio_result,
        'n_signal_consensus': max(
            mp_result['n_signal'],
            1 if tw_result['significant'] else 0
        )
    }


def compare_matrices_rmt(matrices: dict[str, np.ndarray]) -> dict:
    """
    Compare RMT analysis across multiple matrices.

    Args:
        matrices: Dict mapping name -> matrix

    Returns:
        Comparison results
    """
    results = {}
    for name, matrix in matrices.items():
        results[name] = full_rmt_analysis(matrix, name)

    # Summary comparison
    print("\n" + "="*60)
    print("RMT COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Matrix':<20} {'N_signal':<10} {'TW p-val':<10} {'Signal Var%':<12}")
    print("-"*60)

    for name, res in results.items():
        print(
            f"{name:<20} "
            f"{res['mp']['n_signal']:<10} "
            f"{res['tw']['p_value']:<10.4f} "
            f"{res['mp']['signal_variance_ratio']*100:<12.1f}"
        )

    print("="*60)

    return results


def main():
    """Demo RMT analysis on existing matrices."""
    base_path = Path(__file__).parent.parent.parent
    nlp_dir = base_path / "outputs" / "nlp"

    matrices = {}

    # Load available matrices
    for fname in ["claims_matrix_llm.npy", "claims_matrix_embedding.npy"]:
        path = nlp_dir / fname
        if path.exists():
            name = fname.replace("claims_matrix_", "").replace(".npy", "")
            matrices[name] = np.load(path)
            print(f"Loaded {name}: {matrices[name].shape}")

    if matrices:
        compare_matrices_rmt(matrices)
    else:
        print("No matrices found. Run classifiers first.")


if __name__ == "__main__":
    main()
