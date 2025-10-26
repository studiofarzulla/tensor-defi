#!/usr/bin/env python3
"""
Build tensor from CEX data for decomposition experiments
Converts CSV → 4D tensor (Time × Venue × Asset × Feature)
"""

import pandas as pd
import numpy as np
import tensorly as tl
from pathlib import Path
import pickle

def load_and_validate_data(csv_path):
    """Load CSV and perform basic validation"""
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['datetime', 'symbol']).reset_index(drop=True)

    print(f"Loaded {len(df)} records")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Symbols: {df['symbol'].unique().tolist()}")

    return df

def create_tensor(df, normalize=False, log_returns=False):
    """
    Create 4D tensor from dataframe

    Args:
        df: DataFrame with columns [datetime, symbol, open, high, low, close, volume]
        normalize: If True, z-score normalize each feature within each asset
        log_returns: If True, use log returns instead of raw prices

    Returns:
        tensor: 4D numpy array (Time × Venue × Asset × Feature)
        metadata: Dict with dimension info
    """
    symbols = sorted(df['symbol'].unique())
    timestamps = sorted(df['datetime'].unique())

    if log_returns:
        features = ['log_return', 'high_low_range', 'log_volume']
        n_feature = 3
    else:
        features = ['open', 'high', 'low', 'close', 'volume']
        n_feature = 5

    n_time = len(timestamps)
    n_venue = 1  # Single exchange for now
    n_asset = len(symbols)

    print(f"\nBuilding tensor with shape: ({n_time}, {n_venue}, {n_asset}, {n_feature})")

    tensor = np.zeros((n_time, n_venue, n_asset, n_feature))

    for i, symbol in enumerate(symbols):
        symbol_df = df[df['symbol'] == symbol].sort_values('datetime').reset_index(drop=True)

        if log_returns:
            # Calculate log returns
            log_ret = np.log(symbol_df['close'] / symbol_df['open'])
            high_low_range = (symbol_df['high'] - symbol_df['low']) / symbol_df['open']
            log_vol = np.log(symbol_df['volume'] + 1)  # Add 1 to avoid log(0)

            tensor[:, 0, i, 0] = log_ret.values
            tensor[:, 0, i, 1] = high_low_range.values
            tensor[:, 0, i, 2] = log_vol.values

        else:
            # Use raw OHLCV
            for j, feature in enumerate(features):
                tensor[:, 0, i, j] = symbol_df[feature].values

        # Normalize if requested
        if normalize and not log_returns:
            for j in range(n_feature):
                feature_data = tensor[:, 0, i, j]
                tensor[:, 0, i, j] = (feature_data - feature_data.mean()) / feature_data.std()

    metadata = {
        'shape': tensor.shape,
        'symbols': symbols,
        'timestamps': timestamps,
        'features': features,
        'n_time': n_time,
        'n_venue': n_venue,
        'n_asset': n_asset,
        'n_feature': n_feature,
        'normalized': normalize,
        'log_returns': log_returns,
    }

    return tensor, metadata

def save_tensor(tensor, metadata, output_path):
    """Save tensor and metadata to disk"""
    data = {
        'tensor': tensor,
        'metadata': metadata
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nTensor saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

def print_tensor_stats(tensor, metadata):
    """Print summary statistics"""
    print("\n" + "=" * 80)
    print("TENSOR STATISTICS")
    print("=" * 80)

    print(f"\nShape: {tensor.shape}")
    print(f"  Time:    {metadata['n_time']} hours")
    print(f"  Venue:   {metadata['n_venue']} exchange")
    print(f"  Asset:   {metadata['n_asset']} symbols - {metadata['symbols']}")
    print(f"  Feature: {metadata['n_feature']} - {metadata['features']}")

    print(f"\nTotal elements: {tensor.size:,}")
    print(f"Memory usage: {tensor.nbytes / (1024*1024):.2f} MB")

    print(f"\nValue ranges:")
    for i, symbol in enumerate(metadata['symbols']):
        print(f"\n  {symbol}:")
        for j, feature in enumerate(metadata['features']):
            data = tensor[:, 0, i, j]
            print(f"    {feature:12s}: min={data.min():12.4f}, max={data.max():12.4f}, "
                  f"mean={data.mean():12.4f}, std={data.std():12.4f}")

    # Correlation analysis
    print("\n" + "-" * 80)
    print("CROSS-ASSET CORRELATION (using close prices or log returns)")
    print("-" * 80)

    if metadata['log_returns']:
        # Use log returns (feature 0)
        corr_matrix = np.corrcoef([tensor[:, 0, i, 0] for i in range(metadata['n_asset'])])
    else:
        # Use close prices (feature 3)
        corr_matrix = np.corrcoef([tensor[:, 0, i, 3] for i in range(metadata['n_asset'])])

    # Print correlation matrix
    print("\n        ", end="")
    for symbol in metadata['symbols']:
        print(f"{symbol:>12s}", end="")
    print()

    for i, symbol in enumerate(metadata['symbols']):
        print(f"{symbol:8s}", end="")
        for j in range(len(metadata['symbols'])):
            print(f"{corr_matrix[i, j]:12.3f}", end="")
        print()

def main():
    """Main execution"""
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'cex_data_365d.csv'
    output_dir = project_root / 'data' / 'tensors'
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("TENSOR CONSTRUCTION FROM CEX DATA")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    df = load_and_validate_data(data_path)

    # Create tensors with different configurations
    configs = [
        {'normalize': False, 'log_returns': False, 'name': 'raw_ohlcv'},
        {'normalize': True, 'log_returns': False, 'name': 'normalized_ohlcv'},
        {'normalize': False, 'log_returns': True, 'name': 'log_returns'},
    ]

    for config in configs:
        print("\n" + "=" * 80)
        print(f"Building tensor: {config['name']}")
        print("=" * 80)

        tensor, metadata = create_tensor(
            df,
            normalize=config['normalize'],
            log_returns=config['log_returns']
        )

        print_tensor_stats(tensor, metadata)

        output_path = output_dir / f"{config['name']}_tensor.pkl"
        save_tensor(tensor, metadata, output_path)

    print("\n" + "=" * 80)
    print("TENSOR CONSTRUCTION COMPLETE")
    print("=" * 80)
    print(f"\nCreated {len(configs)} tensor files in: {output_dir}")
    print("\nReady for decomposition experiments:")
    print("  - raw_ohlcv_tensor.pkl        → Use for price-focused analysis")
    print("  - normalized_ohlcv_tensor.pkl → Use for cross-asset comparison")
    print("  - log_returns_tensor.pkl      → Use for returns-based modeling")

    print("\nNext step: Run tensor decomposition (CP/Tucker)")

if __name__ == '__main__':
    main()
