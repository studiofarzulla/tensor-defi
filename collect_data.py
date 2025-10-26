#!/usr/bin/env python3
"""
Quick Data Collection Script

Run this to collect 1 week of data for testing before full year collection.
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_pipeline.cex_collector import CEXCollector
from data_pipeline.dex_collector import DEXCollector, COMMON_POOLS

def collect_sample_data():
    """Collect 1 year of data for full analysis."""

    print("="*60)
    print("TENSOR DEFI - DATA COLLECTION")
    print("="*60)
    print(f"Started: {datetime.now()}")
    print()

    # Configuration
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    exchanges = ['binance']
    lookback_days = 365  # Full year collection

    print(f"Collecting {lookback_days} days of data for {len(symbols)} symbols")
    print(f"Exchanges: {exchanges}")
    print()

    # Initialize CEX collector
    print("1. Initializing CEX Collector...")
    cex = CEXCollector(exchanges)

    # Collect CEX data
    print("\n2. Collecting CEX Data (this may take a few minutes)...")
    try:
        cex_data = cex.collect_multimarket_timeseries(
            symbols=symbols,
            timeframe='1h',
            lookback_days=lookback_days
        )

        print(f"✓ Collected {len(cex_data)} CEX data points")
        print(f"  Date range: {cex_data['datetime'].min()} to {cex_data['datetime'].max()}")
        print(f"  Symbols: {cex_data['symbol'].unique().tolist()}")
        print(f"  Exchanges: {cex_data['exchange'].unique().tolist()}")

        # Save CEX data
        cex_file = f'data/cex_data_{lookback_days}d.csv'
        os.makedirs('data', exist_ok=True)
        cex_data.to_csv(cex_file, index=False)
        print(f"  Saved to: {cex_file}")

    except Exception as e:
        print(f"✗ CEX collection failed: {e}")
        print("  This is likely a network/API issue. Try again or check your connection.")
        return False

    # Initialize DEX collector
    print("\n3. Initializing DEX Collector...")
    dex = DEXCollector(['uniswap_v3'])

    # Collect DEX data
    print("\n4. Collecting DEX Data (GraphQL queries)...")
    try:
        dex_pools = [
            {
                'dex': 'uniswap_v3',
                'address': COMMON_POOLS['uniswap_v3']['WETH_USDC_005'],
                'name': 'ETH/USDC'
            },
            {
                'dex': 'uniswap_v3',
                'address': COMMON_POOLS['uniswap_v3']['WETH_USDT_005'],
                'name': 'ETH/USDT'
            },
        ]

        dex_data = dex.collect_multimarket_dex_data(
            pools=dex_pools,
            lookback_hours=lookback_days * 24
        )

        print(f"✓ Collected {len(dex_data)} DEX data points")
        print(f"  Pools: {dex_data['pair_name'].unique().tolist()}")

        # Save DEX data
        dex_file = f'data/dex_data_{lookback_days}d.csv'
        dex_data.to_csv(dex_file, index=False)
        print(f"  Saved to: {dex_file}")

    except Exception as e:
        print(f"✗ DEX collection failed: {e}")
        print("  The Graph API may be rate-limited or down. CEX data still saved.")
        print("  You can continue with CEX-only analysis.")

    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE")
    print("="*60)
    print(f"Finished: {datetime.now()}")
    print()
    print("Next steps:")
    print(f"  1. Check data quality: pandas.read_csv('data/cex_data_{lookback_days}d.csv')")
    print("  2. Run tensor construction: python scripts/build_tensor.py")
    print("  3. Run decomposition analysis: python src/tensor_ops/decomposition.py")

    return True

if __name__ == "__main__":
    success = collect_sample_data()
    sys.exit(0 if success else 1)
