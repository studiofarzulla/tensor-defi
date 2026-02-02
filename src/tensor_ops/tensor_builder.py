#!/usr/bin/env python3
"""
Tensor Construction for TENSOR-DEFI

Builds 4D tensor from OHLCV data: (Time × Venue × Asset × Feature)
For single-venue analysis: (T × 1 × N × F)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


FEATURES = ['open', 'high', 'low', 'close', 'volume']


class TensorBuilder:
    """Constructs OHLCV tensor for decomposition."""

    def __init__(self, market_dir: Path, output_dir: Path):
        self.market_dir = Path(market_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_and_align_data(self) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
        """Load all market data and align timestamps."""
        parquet_files = sorted(self.market_dir.glob("*.parquet"))

        if not parquet_files:
            raise ValueError(f"No market data found in {self.market_dir}")

        # Load all data
        dfs = {}
        all_timestamps = set()

        for parquet_file in parquet_files:
            symbol = parquet_file.stem.replace('_ohlcv', '').upper()
            df = pd.read_parquet(parquet_file)
            df = df.set_index('timestamp')
            dfs[symbol] = df
            all_timestamps.update(df.index)

        # Create common timestamp index
        timestamps = pd.DatetimeIndex(sorted(all_timestamps))
        symbols = sorted(dfs.keys())

        logger.info(f"Loaded {len(symbols)} assets, {len(timestamps)} timestamps")

        # Build aligned 3D array (T × N × F)
        T = len(timestamps)
        N = len(symbols)
        F = len(FEATURES)

        tensor_3d = np.full((T, N, F), np.nan)

        for i, symbol in enumerate(symbols):
            df = dfs[symbol]
            # Reindex to common timestamps
            df_aligned = df.reindex(timestamps)

            for j, feature in enumerate(FEATURES):
                if feature in df_aligned.columns:
                    tensor_3d[:, i, j] = df_aligned[feature].values

        return tensor_3d, symbols, timestamps

    def preprocess_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """Preprocess tensor: handle NaN, normalize."""
        # Forward-fill NaN (common in financial data)
        for i in range(tensor.shape[1]):  # Per asset
            for j in range(tensor.shape[2]):  # Per feature
                series = tensor[:, i, j]
                mask = ~np.isnan(series)
                if mask.sum() > 0:
                    # Forward fill
                    indices = np.where(mask, np.arange(len(series)), 0)
                    np.maximum.accumulate(indices, out=indices)
                    tensor[:, i, j] = series[indices]

        # Z-score normalization per feature
        for j in range(tensor.shape[2]):
            feature_data = tensor[:, :, j]
            mean = np.nanmean(feature_data)
            std = np.nanstd(feature_data)
            if std > 0:
                tensor[:, :, j] = (feature_data - mean) / std

        # Replace remaining NaN with 0
        tensor = np.nan_to_num(tensor, nan=0.0)

        return tensor

    def build_tensor(self, add_venue_dim: bool = True) -> tuple[np.ndarray, dict]:
        """Build and preprocess the market tensor."""
        tensor_3d, symbols, timestamps = self.load_and_align_data()
        tensor_3d = self.preprocess_tensor(tensor_3d)

        # Add venue dimension for full 4D tensor: (T × V × N × F)
        if add_venue_dim:
            tensor_4d = tensor_3d[:, np.newaxis, :, :]  # (T × 1 × N × F)
        else:
            tensor_4d = tensor_3d

        # Metadata
        metadata = {
            'shape': list(tensor_4d.shape),
            'dimensions': ['time', 'venue', 'asset', 'feature'] if add_venue_dim else ['time', 'asset', 'feature'],
            'symbols': symbols,
            'features': FEATURES,
            'venues': ['binance'],
            'time_start': str(timestamps[0]),
            'time_end': str(timestamps[-1]),
            'n_timestamps': len(timestamps)
        }

        # Save tensor
        np.save(self.output_dir / "market_tensor.npy", tensor_4d)

        with open(self.output_dir / "tensor_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved tensor: {tensor_4d.shape}")
        self._print_summary(tensor_4d, metadata)

        return tensor_4d, metadata

    def _print_summary(self, tensor: np.ndarray, metadata: dict):
        """Print tensor summary."""
        print(f"\n{'='*60}")
        print("TENSOR CONSTRUCTION SUMMARY")
        print(f"{'='*60}")
        print(f"Shape: {tensor.shape}")
        print(f"Dimensions: {metadata['dimensions']}")
        print(f"Assets: {len(metadata['symbols'])}")
        print(f"Features: {metadata['features']}")
        print(f"Time range: {metadata['time_start'][:10]} to {metadata['time_end'][:10]}")
        print(f"Timestamps: {metadata['n_timestamps']:,}")
        print(f"Total elements: {tensor.size:,}")
        print(f"Memory: {tensor.nbytes / 1e6:.1f} MB")
        print(f"NaN/zero %: {100 * (tensor == 0).sum() / tensor.size:.1f}%")
        print(f"{'='*60}")


def main():
    """Run tensor construction."""
    base_path = Path(__file__).parent.parent.parent
    builder = TensorBuilder(
        market_dir=base_path / "data" / "market",
        output_dir=base_path / "outputs" / "tensor"
    )
    builder.build_tensor()


if __name__ == "__main__":
    main()
