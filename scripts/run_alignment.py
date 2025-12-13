#!/usr/bin/env python3
"""
Alignment Testing Pipeline for TENSOR-DEFI

Runs Procrustes alignment and Tucker's congruence coefficient tests.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alignment.congruence import AlignmentTester


def main():
    base_path = Path(__file__).parent.parent

    print("="*60)
    print("TENSOR-DEFI: Alignment Testing Pipeline")
    print("="*60)

    tester = AlignmentTester(output_dir=base_path / "outputs" / "alignment")

    claims_path = base_path / "outputs" / "nlp" / "claims_matrix.npy"
    stats_path = base_path / "outputs" / "market" / "stats_matrix.npy"
    factors_path = base_path / "outputs" / "tensor" / "cp_asset_factors.npy"

    # Check all required files exist
    for path, name in [(claims_path, "claims"), (stats_path, "stats"), (factors_path, "factors")]:
        if not path.exists():
            print(f"ERROR: {name} matrix not found at {path}")
            print("Run previous pipeline stages first")
            return

    claims, stats_matrix, factors, symbols = tester.load_matrices(
        claims_path=claims_path,
        stats_path=stats_path,
        factors_path=factors_path
    )

    results = tester.run_alignment_tests(claims, stats_matrix, factors, n_bootstrap=1000)
    tester.save_results(results, symbols)

    print("\n" + "="*60)
    print("ALIGNMENT TESTING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
