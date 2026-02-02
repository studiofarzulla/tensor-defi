#!/usr/bin/env python3
"""
Compare classification results across multiple methods.

Computes:
- Pairwise correlations (Pearson, Spearman)
- Fleiss' Kappa for inter-rater reliability
- Per-category agreement
- Per-asset agreement

Usage:
    python scripts/compare_methods.py
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from itertools import combinations


def load_matrices(nlp_dir: Path) -> dict[str, np.ndarray]:
    """Load all valid claims matrices."""
    matrices = {}

    # Name mappings
    name_map = {
        'claims_matrix.npy': 'bart_nli',
        'claims_matrix_embedding.npy': 'embedding',
        'claims_matrix_llm.npy': 'ministral3',  # Latest LLM run
    }

    for file, name in name_map.items():
        path = nlp_dir / file
        if path.exists():
            m = np.load(path)
            # Check if valid (has non-zero rows)
            nonzero = np.count_nonzero(m.sum(axis=1))
            if nonzero >= m.shape[0] * 0.5:  # At least 50% valid
                matrices[name] = m
                print(f"[OK] {name}: {nonzero}/{m.shape[0]} valid rows")
            else:
                print(f"[SKIP] {name}: only {nonzero}/{m.shape[0]} valid rows")

    return matrices


def pairwise_correlations(matrices: dict[str, np.ndarray]) -> dict:
    """Compute pairwise correlations between methods."""
    results = {}

    for (name1, m1), (name2, m2) in combinations(matrices.items(), 2):
        pair = f"{name1}_vs_{name2}"

        flat1, flat2 = m1.flatten(), m2.flatten()

        r, p_r = pearsonr(flat1, flat2)
        rho, p_rho = spearmanr(flat1, flat2)

        results[pair] = {
            'pearson': {'r': r, 'p': p_r},
            'spearman': {'rho': rho, 'p': p_rho}
        }

    return results


def per_category_agreement(matrices: dict[str, np.ndarray], categories: list[str]) -> dict:
    """Compute per-category correlations across methods."""
    n_methods = len(matrices)
    if n_methods < 2:
        return {}

    method_names = list(matrices.keys())
    results = {}

    for i, cat in enumerate(categories):
        cat_data = {name: m[:, i] for name, m in matrices.items()}

        # Average pairwise correlation
        corrs = []
        for (n1, d1), (n2, d2) in combinations(cat_data.items(), 2):
            r, _ = pearsonr(d1, d2)
            corrs.append(r)

        results[cat] = {
            'mean_correlation': np.mean(corrs),
            'std_correlation': np.std(corrs),
            'n_pairs': len(corrs)
        }

    return results


def per_asset_agreement(matrices: dict[str, np.ndarray]) -> np.ndarray:
    """Compute per-asset agreement (average pairwise correlation per row)."""
    n_assets = list(matrices.values())[0].shape[0]
    method_data = list(matrices.values())

    asset_agreement = np.zeros(n_assets)

    for j in range(n_assets):
        rows = [m[j] for m in method_data]
        corrs = []
        for r1, r2 in combinations(rows, 2):
            if np.std(r1) > 0 and np.std(r2) > 0:
                r, _ = pearsonr(r1, r2)
                corrs.append(r)

        asset_agreement[j] = np.mean(corrs) if corrs else 0

    return asset_agreement


def fleiss_kappa_from_continuous(matrices: dict[str, np.ndarray], n_bins: int = 4) -> float:
    """
    Compute Fleiss' Kappa by discretizing continuous scores into bins.

    Args:
        matrices: Dict of method name -> matrix
        n_bins: Number of bins for discretization

    Returns:
        Fleiss' Kappa coefficient
    """
    from collections import Counter

    method_data = list(matrices.values())
    n_methods = len(method_data)
    n_items = method_data[0].size

    # Flatten and bin each method's scores
    bins = np.linspace(0, 1.0001, n_bins + 1)  # [0, 0.25, 0.5, 0.75, 1.0]

    all_binned = []
    for m in method_data:
        binned = np.digitize(m.flatten(), bins) - 1  # 0 to n_bins-1
        binned = np.clip(binned, 0, n_bins - 1)
        all_binned.append(binned)

    # Count category assignments per item
    # n_ij = number of raters who assigned item i to category j
    counts = np.zeros((n_items, n_bins))
    for binned in all_binned:
        for i, b in enumerate(binned):
            counts[i, int(b)] += 1

    # Fleiss' Kappa calculation
    n = n_methods  # Number of raters
    N = n_items    # Number of items
    k = n_bins     # Number of categories

    # P_i = proportion of agreement for each item
    P_i = (1.0 / (n * (n - 1))) * (np.sum(counts**2, axis=1) - n)
    P_bar = np.mean(P_i)

    # p_j = proportion of all assignments to category j
    p_j = np.sum(counts, axis=0) / (N * n)

    # P_e_bar = expected agreement by chance
    P_e_bar = np.sum(p_j**2)

    # Kappa
    if P_e_bar == 1:
        return 1.0

    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    return kappa


def main():
    base_path = Path(__file__).parent.parent
    nlp_dir = base_path / "outputs" / "nlp"

    # Categories
    categories = [
        'store_of_value', 'medium_of_exchange', 'smart_contracts', 'defi',
        'governance', 'scalability', 'privacy', 'interoperability',
        'data_storage', 'oracle'
    ]

    print("=" * 70)
    print("MULTI-METHOD CLASSIFICATION AGREEMENT")
    print("=" * 70)

    # Load matrices
    print("\nLoading matrices...")
    matrices = load_matrices(nlp_dir)

    if len(matrices) < 2:
        print("\nNeed at least 2 valid matrices for comparison!")
        return

    # Pairwise correlations
    print("\n" + "-" * 70)
    print("PAIRWISE CORRELATIONS")
    print("-" * 70)

    pw_corrs = pairwise_correlations(matrices)
    for pair, stats in pw_corrs.items():
        print(f"\n{pair}:")
        print(f"  Pearson r:  {stats['pearson']['r']:.3f} (p={stats['pearson']['p']:.2e})")
        print(f"  Spearman ρ: {stats['spearman']['rho']:.3f} (p={stats['spearman']['p']:.2e})")

    # Per-category agreement
    print("\n" + "-" * 70)
    print("PER-CATEGORY AGREEMENT (mean pairwise correlation)")
    print("-" * 70)

    cat_agreement = per_category_agreement(matrices, categories)
    sorted_cats = sorted(cat_agreement.items(), key=lambda x: x[1]['mean_correlation'], reverse=True)

    for cat, stats in sorted_cats:
        r = stats['mean_correlation']
        bar = "#" * int(max(0, r) * 30)
        print(f"  {cat:20s}: {r:+.3f} {bar}")

    # Per-asset agreement
    print("\n" + "-" * 70)
    print("PER-ASSET AGREEMENT (mean pairwise correlation)")
    print("-" * 70)

    asset_agreement = per_asset_agreement(matrices)

    print(f"  Mean:   {np.mean(asset_agreement):.3f}")
    print(f"  Std:    {np.std(asset_agreement):.3f}")
    print(f"  Min:    {np.min(asset_agreement):.3f}")
    print(f"  Max:    {np.max(asset_agreement):.3f}")

    # Assets with worst agreement
    worst_5 = np.argsort(asset_agreement)[:5]
    print(f"\n  Lowest agreement assets: {list(worst_5)}")

    # Fleiss' Kappa
    print("\n" + "-" * 70)
    print("FLEISS' KAPPA (discretized to 4 bins)")
    print("-" * 70)

    kappa = fleiss_kappa_from_continuous(matrices, n_bins=4)
    interpretation = (
        "Almost perfect" if kappa > 0.80 else
        "Substantial" if kappa > 0.60 else
        "Moderate" if kappa > 0.40 else
        "Fair" if kappa > 0.20 else
        "Slight" if kappa > 0 else
        "Poor"
    )
    print(f"  κ = {kappa:.3f} ({interpretation} agreement)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Methods compared: {list(matrices.keys())}")
    print(f"Average pairwise Pearson r: {np.mean([s['pearson']['r'] for s in pw_corrs.values()]):.3f}")
    print(f"Average per-category agreement: {np.mean([s['mean_correlation'] for s in cat_agreement.values()]):.3f}")
    print(f"Fleiss' Kappa: {kappa:.3f}")
    print("=" * 70)

    # Save results
    results = {
        'methods': list(matrices.keys()),
        'pairwise_correlations': {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                                      for kk, vv in v.items()}
                                  for k, v in pw_corrs.items()},
        'category_agreement': {k: {kk: float(vv) for kk, vv in v.items()}
                               for k, v in cat_agreement.items()},
        'fleiss_kappa': float(kappa)
    }

    output_path = nlp_dir / "method_agreement.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
