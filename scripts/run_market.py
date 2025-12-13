#!/usr/bin/env python3
"""
Market Statistics Pipeline for TENSOR-DEFI

Computes summary statistics matrix from OHLCV data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market.summary_statistics import SummaryStatistics


def main():
    base_path = Path(__file__).parent.parent

    print("="*60)
    print("TENSOR-DEFI: Market Statistics Pipeline")
    print("="*60)

    stats = SummaryStatistics(
        market_dir=base_path / "data" / "market",
        output_dir=base_path / "outputs" / "market"
    )

    try:
        stats.build_stats_matrix()
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Run data collection first")
        return

    print("\n" + "="*60)
    print("MARKET STATISTICS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
