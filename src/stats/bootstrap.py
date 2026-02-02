#!/usr/bin/env python3
"""
Bootstrap confidence intervals for claims matrices.

Resamples chunks with replacement to quantify classification uncertainty.
Tight CIs = stable estimates, wide CIs = high variance.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def bootstrap_document_profile(
    chunks: list[str],
    classify_fn: Callable[[list[str]], np.ndarray],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None
) -> dict:
    """
    Bootstrap CI for a single document's category profile.

    Args:
        chunks: List of text chunks
        classify_fn: Function that takes chunks and returns (n_chunks, n_categories) scores
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (0.05 = 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Dict with mean, lower, upper, std for each category
    """
    if seed is not None:
        np.random.seed(seed)

    n_chunks = len(chunks)
    if n_chunks == 0:
        return None

    bootstrap_profiles = []

    for _ in range(n_bootstrap):
        # Resample chunks with replacement
        indices = np.random.choice(n_chunks, size=n_chunks, replace=True)
        resampled = [chunks[i] for i in indices]

        # Classify resampled chunks
        scores = classify_fn(resampled)  # (n_resampled, n_categories)

        # Aggregate to profile
        profile = scores.mean(axis=0)
        if profile.sum() > 0:
            profile = profile / profile.sum()

        bootstrap_profiles.append(profile)

    stacked = np.stack(bootstrap_profiles)  # (n_bootstrap, n_categories)

    # Compute statistics
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)

    # Percentile CI
    lower = np.percentile(stacked, 100 * alpha / 2, axis=0)
    upper = np.percentile(stacked, 100 * (1 - alpha / 2), axis=0)

    # BCa (Bias-Corrected and Accelerated) - more sophisticated
    # For now, use percentile. BCa requires jackknife which is expensive.

    return {
        'mean': mean,
        'std': std,
        'lower': lower,
        'upper': upper,
        'n_bootstrap': n_bootstrap,
        'alpha': alpha
    }


def bootstrap_claims_matrix(
    chunks_data: dict,
    classifier,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = 42
) -> dict:
    """
    Bootstrap CI for entire claims matrix.

    Args:
        chunks_data: Dict mapping symbol -> {'chunks': [...]}
        classifier: Classifier with classify_chunks method
        n_bootstrap: Bootstrap samples per document
        alpha: Significance level
        seed: Random seed

    Returns:
        Dict with matrices for mean, lower, upper, std
    """
    if seed is not None:
        np.random.seed(seed)

    symbols = sorted(chunks_data.keys())
    n_entities = len(symbols)
    n_categories = len(classifier.categories)

    # Initialize output matrices
    mean_matrix = np.zeros((n_entities, n_categories))
    lower_matrix = np.zeros((n_entities, n_categories))
    upper_matrix = np.zeros((n_entities, n_categories))
    std_matrix = np.zeros((n_entities, n_categories))

    for i, sym in enumerate(tqdm(symbols, desc="Bootstrap CI")):
        chunks = chunks_data[sym]['chunks']

        if not chunks:
            continue

        result = bootstrap_document_profile(
            chunks,
            classifier.classify_chunks,
            n_bootstrap=n_bootstrap,
            alpha=alpha
        )

        if result:
            mean_matrix[i] = result['mean']
            lower_matrix[i] = result['lower']
            upper_matrix[i] = result['upper']
            std_matrix[i] = result['std']

    return {
        'mean': mean_matrix,
        'lower': lower_matrix,
        'upper': upper_matrix,
        'std': std_matrix,
        'symbols': symbols,
        'n_bootstrap': n_bootstrap,
        'alpha': alpha
    }


def bootstrap_alignment_statistic(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    alignment_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = 42
) -> dict:
    """
    Bootstrap CI for alignment statistic between two matrices.

    Resamples rows (entities) with replacement.

    Args:
        matrix1: First claims matrix (n_entities, n_categories)
        matrix2: Second claims matrix (same shape)
        alignment_fn: Function that computes alignment statistic
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        seed: Random seed

    Returns:
        Dict with observed, mean, lower, upper, std, p_value
    """
    if seed is not None:
        np.random.seed(seed)

    assert matrix1.shape[0] == matrix2.shape[0], "Matrices must have same number of entities"

    n_entities = matrix1.shape[0]

    # Observed statistic
    observed = alignment_fn(matrix1, matrix2)

    # Bootstrap distribution
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_entities, size=n_entities, replace=True)
        m1_boot = matrix1[indices]
        m2_boot = matrix2[indices]
        stat = alignment_fn(m1_boot, m2_boot)
        bootstrap_stats.append(stat)

    bootstrap_stats = np.array(bootstrap_stats)

    # Statistics
    mean = bootstrap_stats.mean()
    std = bootstrap_stats.std()
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    # P-value: proportion of bootstrap samples <= 0 (for positive statistics)
    # This tests H0: statistic = 0
    p_value = (bootstrap_stats <= 0).mean()

    return {
        'observed': observed,
        'mean': mean,
        'std': std,
        'lower': lower,
        'upper': upper,
        'p_value': p_value,
        'n_bootstrap': n_bootstrap,
        'alpha': alpha
    }


def save_bootstrap_results(results: dict, output_path: Path):
    """Save bootstrap results to files."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save matrices
    for key in ['mean', 'lower', 'upper', 'std']:
        if key in results and isinstance(results[key], np.ndarray):
            np.save(output_path / f"bootstrap_{key}.npy", results[key])

    # Save metadata
    metadata = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
    if 'symbols' in results:
        metadata['symbols'] = results['symbols']

    with open(output_path / "bootstrap_meta.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Bootstrap results saved to {output_path}")


def main():
    """Demo bootstrap on existing matrix."""
    from ..nlp.embedding_classifier import EmbeddingClassifier

    base_path = Path(__file__).parent.parent.parent
    chunks_path = base_path / "outputs" / "nlp" / "extracted_chunks.json"
    output_dir = base_path / "outputs" / "stats"

    with open(chunks_path) as f:
        chunks_data = json.load(f)

    classifier = EmbeddingClassifier()

    results = bootstrap_claims_matrix(
        chunks_data,
        classifier,
        n_bootstrap=100,  # Quick demo
        seed=42
    )

    save_bootstrap_results(results, output_dir / "bootstrap_embedding")
    print(f"Mean CI width: {(results['upper'] - results['lower']).mean():.4f}")


if __name__ == "__main__":
    main()
