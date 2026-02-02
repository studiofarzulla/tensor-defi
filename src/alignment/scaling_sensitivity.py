#!/usr/bin/env python3
"""
Scaling Sensitivity Analysis.

Tests whether CP factor structure is robust to different tensor constructions:
1. Original: OHLCV levels (normalized)
2. Returns: Log returns instead of levels
3. Z-scored: Standardized features per asset

Addresses reviewer question about sensitivity to scaling choices.
"""

import numpy as np
import json
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
import tensorly as tl
from tensorly.decomposition import parafac
import warnings
warnings.filterwarnings('ignore')


def tucker_phi(A: np.ndarray, B: np.ndarray) -> float:
    """Tucker's congruence coefficient after Procrustes rotation."""
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)

    R, _ = orthogonal_procrustes(A, B)
    A_rot = A @ R

    phis = []
    for j in range(A_rot.shape[1]):
        num = np.dot(A_rot[:, j], B[:, j])
        denom = np.sqrt(np.dot(A_rot[:, j], A_rot[:, j]) * np.dot(B[:, j], B[:, j]))
        if denom > 0:
            phis.append(num / denom)

    return np.mean(phis) if phis else 0.0


def construct_returns_tensor(tensor: np.ndarray) -> np.ndarray:
    """Convert OHLCV tensor to returns-based tensor."""
    # tensor shape: (time, asset, features) after squeezing venue
    # Compute log returns for price features (OHLC), keep volume as pct change
    returns_tensor = np.zeros_like(tensor)

    for i in range(tensor.shape[1]):  # Each asset
        for f in range(tensor.shape[2]):  # Each feature
            series = tensor[:, i, f]
            # Log returns (avoiding log(0))
            series_safe = np.where(series > 0, series, 1e-10)
            returns = np.diff(np.log(series_safe))
            # Pad first value with 0
            returns_tensor[1:, i, f] = returns
            returns_tensor[0, i, f] = 0

    return returns_tensor


def construct_zscore_tensor(tensor: np.ndarray) -> np.ndarray:
    """Z-score normalize each asset-feature combination."""
    # tensor shape: (time, asset, features) after squeezing venue
    zscore_tensor = np.zeros_like(tensor)

    for i in range(tensor.shape[1]):  # Each asset
        for f in range(tensor.shape[2]):  # Each feature
            series = tensor[:, i, f]
            mean = np.mean(series)
            std = np.std(series)
            if std > 0:
                zscore_tensor[:, i, f] = (series - mean) / std
            else:
                zscore_tensor[:, i, f] = 0

    return zscore_tensor


def fit_cp_decomposition(tensor: np.ndarray, rank: int = 2) -> np.ndarray:
    """Fit CP decomposition and return asset factors.

    Tensor shape: (time, asset, feature)
    Mode 0 = time factors
    Mode 1 = asset factors (what we want)
    Mode 2 = feature factors
    """
    # Normalize tensor for stability
    tensor_norm = tensor / (np.linalg.norm(tensor) + 1e-10)

    # Fit CP-ALS
    weights, factors = parafac(tensor_norm, rank=rank, n_iter_max=200,
                               init='random', random_state=42)

    # Extract asset mode factors (mode 1)
    asset_factors = factors[1]  # Mode 1 = assets

    return asset_factors


def main():
    base_path = Path(__file__).parent.parent.parent
    output_path = base_path / 'outputs' / 'alignment'
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SCALING SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Load original tensor
    print("\nLoading market tensor...")
    tensor_4d = np.load(base_path / 'outputs/tensor/market_tensor.npy')

    with open(base_path / 'outputs/tensor/tensor_meta.json') as f:
        tensor_meta = json.load(f)

    # Squeeze out venue dimension: (time, venue, asset, feature) -> (time, asset, feature)
    tensor = tensor_4d.squeeze(axis=1)
    print(f"  Original shape: {tensor_4d.shape} (time × venue × asset × feature)")
    print(f"  Squeezed shape: {tensor.shape} (time × asset × feature)")
    print(f"  Features: {tensor_meta.get('features', ['O', 'H', 'L', 'C', 'V'])}")

    # Load original CP factors for comparison
    original_factors = np.load(base_path / 'outputs/tensor/cp_asset_factors.npy')
    print(f"  Original factors shape: {original_factors.shape}")

    # Construct alternative tensors
    print("\nConstructing alternative tensor representations...")

    tensors = {
        'original': tensor,
        'returns': construct_returns_tensor(tensor),
        'zscore': construct_zscore_tensor(tensor)
    }

    # Handle NaN/Inf in returns
    tensors['returns'] = np.nan_to_num(tensors['returns'], nan=0.0, posinf=0.0, neginf=0.0)

    # Fit CP decomposition for each
    print("\nFitting CP decompositions (rank=2)...")
    factors = {}

    for name, t in tensors.items():
        print(f"  {name}...", end=' ')
        factors[name] = fit_cp_decomposition(t, rank=2)
        ev = np.var(factors[name]) / np.var(t.reshape(-1)) * 100
        print(f"done (factor var: {np.var(factors[name]):.4f})")

    # Compare factor structures via Tucker's phi
    print("\n" + "-" * 60)
    print("FACTOR STRUCTURE COMPARISON (Tucker's φ)")
    print("-" * 60)

    comparisons = [
        ('original', 'returns'),
        ('original', 'zscore'),
        ('returns', 'zscore')
    ]

    results = {
        'tensor_shape': list(tensor.shape),
        'rank': 2,
        'comparisons': []
    }

    print("\n{:<25} {:>12} {:>15}".format("Comparison", "φ", "Interpretation"))
    print("-" * 55)

    for a, b in comparisons:
        phi = tucker_phi(factors[a], factors[b])
        interp = "excellent" if phi >= 0.95 else "good" if phi >= 0.85 else "moderate" if phi >= 0.65 else "weak"

        print(f"{a} vs {b:<12} {phi:>12.4f} {interp:>15}")

        results['comparisons'].append({
            'tensor_a': a,
            'tensor_b': b,
            'phi': float(phi),
            'interpretation': interp
        })

    # Compare with original stored factors
    print("\n" + "-" * 60)
    print("COMPARISON WITH STORED ORIGINAL FACTORS")
    print("-" * 60)

    # Check shapes match
    print(f"\nRecomputed shape: {factors['original'].shape}")
    print(f"Stored shape: {original_factors.shape}")

    if factors['original'].shape == original_factors.shape:
        phi_stored = tucker_phi(factors['original'], original_factors)
        print(f"Recomputed vs Stored original: φ = {phi_stored:.4f}")

        if phi_stored < 0.90:
            print("  Note: Difference may be due to random initialization in CP-ALS")
    else:
        print("  Shapes don't match - skipping direct comparison")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    mean_phi = np.mean([c['phi'] for c in results['comparisons']])
    print(f"\nMean cross-construction φ: {mean_phi:.4f}")

    if mean_phi >= 0.85:
        print("→ Factor structure is ROBUST to scaling choices")
    elif mean_phi >= 0.65:
        print("→ Factor structure shows MODERATE sensitivity to scaling")
    else:
        print("→ Factor structure is SENSITIVE to scaling choices")

    # Save results
    with open(output_path / 'scaling_sensitivity.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path / 'scaling_sensitivity.json'}")

    return results


if __name__ == '__main__':
    main()
