#!/usr/bin/env python3
"""
Reduced Taxonomy for TENSOR-DEFI Expansion

Collapses 10-category taxonomy to 5 categories to test if coarser
granularity improves signal stability:

10 → 5 Mapping:
- monetary: store_of_value + medium_of_exchange
- compute: smart_contracts + scalability
- finance: defi + oracle
- infrastructure: interoperability + data_storage + governance
- privacy: privacy (kept distinct - orthogonal dimension)

Hypothesis: Less sparse matrix may yield more stable alignment.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Reduced taxonomy mapping
REDUCED_TAXONOMY = {
    'monetary': ['store_of_value', 'medium_of_exchange'],
    'compute': ['smart_contracts', 'scalability'],
    'finance': ['defi', 'oracle'],
    'infrastructure': ['interoperability', 'data_storage', 'governance'],
    'privacy': ['privacy']
}

# Category descriptions for documentation
REDUCED_CATEGORY_INFO = {
    'monetary': {
        'label': 'Monetary Functions',
        'description': 'Store of value and medium of exchange claims',
        'sources': ['store_of_value', 'medium_of_exchange']
    },
    'compute': {
        'label': 'Compute Layer',
        'description': 'Smart contract execution and scalability claims',
        'sources': ['smart_contracts', 'scalability']
    },
    'finance': {
        'label': 'Financial Services',
        'description': 'DeFi protocols and oracle data feeds',
        'sources': ['defi', 'oracle']
    },
    'infrastructure': {
        'label': 'Infrastructure',
        'description': 'Cross-chain, storage, and governance claims',
        'sources': ['interoperability', 'data_storage', 'governance']
    },
    'privacy': {
        'label': 'Privacy',
        'description': 'Privacy and anonymity features',
        'sources': ['privacy']
    }
}

# Ordered list of reduced categories
REDUCED_CATEGORIES = ['monetary', 'compute', 'finance', 'infrastructure', 'privacy']

# Original 10 categories in order
ORIGINAL_CATEGORIES = [
    'store_of_value',
    'medium_of_exchange',
    'smart_contracts',
    'defi',
    'governance',
    'scalability',
    'privacy',
    'interoperability',
    'data_storage',
    'oracle'
]


def get_reduction_matrix() -> np.ndarray:
    """
    Create reduction matrix R that maps 10-cat to 5-cat.

    R is 10 × 5 where R[i,j] = 1 if original category i maps to reduced category j.
    Normalized so each column sums to 1 (average of source categories).

    Usage: reduced = original @ R
    """
    R = np.zeros((len(ORIGINAL_CATEGORIES), len(REDUCED_CATEGORIES)))

    for j, reduced_cat in enumerate(REDUCED_CATEGORIES):
        source_cats = REDUCED_TAXONOMY[reduced_cat]
        for source_cat in source_cats:
            i = ORIGINAL_CATEGORIES.index(source_cat)
            R[i, j] = 1.0

        # Normalize column to average source categories
        col_sum = R[:, j].sum()
        if col_sum > 0:
            R[:, j] /= col_sum

    return R


def reduce_claims_matrix(
    claims_10: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Reduce 10-category claims matrix to 5-category.

    Args:
        claims_10: N × 10 claims matrix
        normalize: Whether to normalize output rows to sum to 1

    Returns:
        N × 5 reduced claims matrix
    """
    R = get_reduction_matrix()
    claims_5 = claims_10 @ R

    if normalize:
        row_sums = claims_5.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        claims_5 = claims_5 / row_sums

    return claims_5


class ReducedTaxonomyConverter:
    """Converts between 10-category and 5-category taxonomies."""

    def __init__(self):
        self.R = get_reduction_matrix()

    def convert_matrix(
        self,
        input_path: Path,
        output_dir: Path,
        meta_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Convert existing 10-category claims matrix to 5-category.

        Args:
            input_path: Path to 10-category claims_matrix.npy
            output_dir: Output directory for reduced matrix
            meta_path: Path to claims_matrix_meta.json (optional)
        """
        # Load original matrix
        claims_10 = np.load(input_path)
        logger.info(f"Loaded {claims_10.shape} claims matrix")

        # Reduce
        claims_5 = reduce_claims_matrix(claims_10)
        logger.info(f"Reduced to {claims_5.shape}")

        # Load metadata if available
        symbols = None
        if meta_path and meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                symbols = meta.get('symbols', [])

        # Save reduced matrix
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "claims_matrix_5cat.npy", claims_5)

        # Save metadata
        meta_5 = {
            'symbols': symbols or [],
            'categories': REDUCED_CATEGORIES,
            'category_info': REDUCED_CATEGORY_INFO,
            'shape': list(claims_5.shape),
            'source_categories': ORIGINAL_CATEGORIES,
            'reduction_mapping': REDUCED_TAXONOMY
        }
        with open(output_dir / "claims_matrix_5cat_meta.json", 'w') as f:
            json.dump(meta_5, f, indent=2)

        # Save as CSV
        if symbols:
            df = pd.DataFrame(
                claims_5,
                index=symbols,
                columns=REDUCED_CATEGORIES
            )
            df.to_csv(output_dir / "claims_matrix_5cat.csv")

        logger.info(f"Saved reduced matrix to {output_dir}")

        self._print_comparison(claims_10, claims_5, symbols)

        return claims_5

    def _print_comparison(
        self,
        claims_10: np.ndarray,
        claims_5: np.ndarray,
        symbols: Optional[list] = None
    ):
        """Print comparison between original and reduced taxonomies."""
        print(f"\n{'='*60}")
        print("TAXONOMY REDUCTION SUMMARY")
        print(f"{'='*60}")
        print(f"Original:  {claims_10.shape[0]} entities × {claims_10.shape[1]} categories")
        print(f"Reduced:   {claims_5.shape[0]} entities × {claims_5.shape[1]} categories")

        print(f"\nReduction mapping:")
        for reduced_cat in REDUCED_CATEGORIES:
            sources = REDUCED_TAXONOMY[reduced_cat]
            print(f"  {reduced_cat:15s} ← {', '.join(sources)}")

        print(f"\nCategory means (reduced):")
        mean_5 = claims_5.mean(axis=0)
        for i, cat in enumerate(REDUCED_CATEGORIES):
            bar = '#' * int(mean_5[i] * 40)
            print(f"  {cat:15s} {mean_5[i]:.3f} {bar}")

        # Compare sparsity
        sparsity_10 = (claims_10 < 0.1).sum() / claims_10.size
        sparsity_5 = (claims_5 < 0.1).sum() / claims_5.size
        print(f"\nSparsity (<0.1 threshold):")
        print(f"  10-category: {sparsity_10:.1%}")
        print(f"  5-category:  {sparsity_5:.1%}")

        # Top entity per reduced category
        if symbols:
            print(f"\nTop entity per reduced category:")
            for i, cat in enumerate(REDUCED_CATEGORIES):
                top_idx = np.argmax(claims_5[:, i])
                top_symbol = symbols[top_idx]
                top_score = claims_5[top_idx, i]
                print(f"  {cat:15s} → {top_symbol} ({top_score:.3f})")

        print(f"{'='*60}")


def convert_ensemble_matrices(base_path: Path):
    """
    Convert both soft and hard ensemble matrices to 5-category.
    """
    nlp_dir = base_path / "outputs" / "nlp"
    converter = ReducedTaxonomyConverter()

    # Convert soft voting matrix
    soft_path = nlp_dir / "claims_matrix_ensemble_soft.npy"
    if soft_path.exists():
        logger.info("Converting ensemble soft matrix...")
        claims_soft_10 = np.load(soft_path)
        claims_soft_5 = reduce_claims_matrix(claims_soft_10)
        np.save(nlp_dir / "claims_matrix_ensemble_soft_5cat.npy", claims_soft_5)
        logger.info(f"Saved soft 5-cat: {claims_soft_5.shape}")

    # Convert hard voting matrix
    hard_path = nlp_dir / "claims_matrix_ensemble_hard.npy"
    if hard_path.exists():
        logger.info("Converting ensemble hard matrix...")
        claims_hard_10 = np.load(hard_path)
        claims_hard_5 = reduce_claims_matrix(claims_hard_10)
        np.save(nlp_dir / "claims_matrix_ensemble_hard_5cat.npy", claims_hard_5)
        logger.info(f"Saved hard 5-cat: {claims_hard_5.shape}")

    return {
        'soft_5cat': nlp_dir / "claims_matrix_ensemble_soft_5cat.npy",
        'hard_5cat': nlp_dir / "claims_matrix_ensemble_hard_5cat.npy"
    }


def main():
    """Run taxonomy reduction."""
    base_path = Path(__file__).parent.parent.parent
    nlp_dir = base_path / "outputs" / "nlp"

    # Convert original claims matrix
    converter = ReducedTaxonomyConverter()
    converter.convert_matrix(
        input_path=nlp_dir / "claims_matrix.npy",
        output_dir=nlp_dir,
        meta_path=nlp_dir / "claims_matrix_meta.json"
    )

    # Convert ensemble matrices if they exist
    convert_ensemble_matrices(base_path)


if __name__ == "__main__":
    main()
