#!/usr/bin/env python3
"""
Data Collection Pipeline for TENSOR-DEFI

Runs whitepaper and market data collection in parallel.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_pipeline.whitepaper_collector import WhitepaperCollector
from data_pipeline.cex_collector import CEXCollector


async def main():
    base_path = Path(__file__).parent.parent

    print("="*60)
    print("TENSOR-DEFI: Data Collection Pipeline")
    print("="*60)

    # Run whitepaper and market collection in parallel
    wp_collector = WhitepaperCollector(
        output_dir=base_path / "data" / "whitepapers",
        metadata_path=base_path / "data" / "metadata"
    )

    market_collector = CEXCollector(
        output_dir=base_path / "data" / "market",
        metadata_path=base_path / "data" / "metadata"
    )

    # Run whitepapers first (faster), then market data
    print("\n[1/2] Collecting whitepapers...")
    await wp_collector.collect_all()

    print("\n[2/2] Collecting market data...")
    await market_collector.collect_all()

    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
