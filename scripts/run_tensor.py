#!/usr/bin/env python3
"""
Tensor Decomposition Pipeline for TENSOR-DEFI

Builds tensor and runs CP decomposition to extract factors.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensor_ops.tensor_builder import TensorBuilder
from tensor_ops.decomposition import TensorDecomposition


def main():
    base_path = Path(__file__).parent.parent

    print("="*60)
    print("TENSOR-DEFI: Tensor Decomposition Pipeline")
    print("="*60)

    # Step 1: Build tensor
    print("\n[1/2] Building market tensor...")
    builder = TensorBuilder(
        market_dir=base_path / "data" / "market",
        output_dir=base_path / "outputs" / "tensor"
    )

    try:
        builder.build_tensor()
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Run data collection first")
        return

    # Step 2: CP decomposition
    print("\n[2/2] Running CP decomposition...")
    decomp = TensorDecomposition(
        tensor_dir=base_path / "outputs" / "tensor",
        output_dir=base_path / "outputs" / "tensor"
    )

    decomp.extract_asset_factors(method='cp', target_variance=0.90)

    print("\n" + "="*60)
    print("TENSOR DECOMPOSITION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
