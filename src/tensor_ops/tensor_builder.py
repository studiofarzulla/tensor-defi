"""
Tensor Builder - Construct multi-dimensional market tensors

Transforms flat market data into rich tensor structures that preserve
multi-way interactions invisible to matrix-based models.

Core tensor shapes:
- Time × Venue × Asset × Feature (basic microstructure)
- Time × Venue × Price_Level × Side (order book dynamics)
- Time × Protocol × Asset × Function (DeFi composability)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class TensorMode(Enum):
    """Tensor construction modes."""
    MICROSTRUCTURE = "microstructure"  # Time × Venue × Asset × Feature
    ORDERBOOK = "orderbook"            # Time × Venue × Level × Side × Feature
    DEFI_COMPOSITE = "defi_composite"  # Time × Protocol × Asset × Function
    MULTIMARKET = "multimarket"        # Time × Venue × Asset × Regime


@dataclass
class TensorMetadata:
    """Metadata for constructed tensor."""
    shape: Tuple[int, ...]
    mode: TensorMode
    dimension_names: List[str]
    dimension_indices: Dict[str, Dict]
    timestamp_range: Tuple[pd.Timestamp, pd.Timestamp]
    assets: List[str]
    venues: List[str]


class TensorBuilder:
    """Build market microstructure tensors from time-series data."""

    def __init__(self):
        self.metadata: Optional[TensorMetadata] = None

    def build_microstructure_tensor(
        self,
        data: pd.DataFrame,
        features: List[str],
        time_window: str = '1H',
        fill_method: str = 'ffill'
    ) -> Tuple[np.ndarray, TensorMetadata]:
        """
        Build basic microstructure tensor: Time × Venue × Asset × Feature

        Args:
            data: DataFrame with columns [datetime, venue, asset, feature1, feature2, ...]
            features: List of feature column names to include
            time_window: Resampling window for time dimension
            fill_method: How to handle missing data ('ffill', 'bfill', 'zero', 'mean')

        Returns:
            (tensor, metadata) tuple
        """
        # Ensure datetime index
        if 'datetime' in data.columns:
            data = data.set_index('datetime')

        # Resample to uniform time grid
        data = data.groupby(['venue', 'asset']).resample(time_window).mean()
        data = data.reset_index()

        # Get unique values for each dimension
        timestamps = sorted(data['datetime'].unique())
        venues = sorted(data['venue'].unique())
        assets = sorted(data['asset'].unique())

        # Create dimension indices
        time_idx = {ts: i for i, ts in enumerate(timestamps)}
        venue_idx = {v: i for i, v in enumerate(venues)}
        asset_idx = {a: i for i, a in enumerate(assets)}
        feature_idx = {f: i for i, f in enumerate(features)}

        # Initialize tensor
        shape = (len(timestamps), len(venues), len(assets), len(features))
        tensor = np.zeros(shape)

        # Fill tensor
        for _, row in data.iterrows():
            if row['asset'] not in asset_idx:
                continue

            t_i = time_idx[row['datetime']]
            v_i = venue_idx[row['venue']]
            a_i = asset_idx[row['asset']]

            for f_i, feature in enumerate(features):
                if feature in row and pd.notna(row[feature]):
                    tensor[t_i, v_i, a_i, f_i] = row[feature]

        # Handle missing data
        tensor = self._fill_missing(tensor, fill_method)

        # Create metadata
        metadata = TensorMetadata(
            shape=shape,
            mode=TensorMode.MICROSTRUCTURE,
            dimension_names=['time', 'venue', 'asset', 'feature'],
            dimension_indices={
                'time': time_idx,
                'venue': venue_idx,
                'asset': asset_idx,
                'feature': feature_idx
            },
            timestamp_range=(timestamps[0], timestamps[-1]),
            assets=assets,
            venues=venues
        )

        self.metadata = metadata
        return tensor, metadata

    def build_orderbook_tensor(
        self,
        orderbook_snapshots: List[Dict],
        max_levels: int = 20,
        time_window: Optional[str] = None
    ) -> Tuple[np.ndarray, TensorMetadata]:
        """
        Build order book tensor: Time × Venue × Level × Side × Feature

        Features include: price, size, cumulative_size, distance_from_mid

        Args:
            orderbook_snapshots: List of order book snapshots from CEXCollector
            max_levels: Maximum price levels to include
            time_window: Optional resampling window

        Returns:
            (tensor, metadata) tuple
        """
        # Convert to DataFrame for easier processing
        snapshot_list = []
        for snap in orderbook_snapshots:
            snapshot_list.append({
                'datetime': pd.to_datetime(snap['datetime']),
                'venue': snap['exchange'],
                'asset': snap['symbol'],
                'bids': snap['bids'][:max_levels],
                'asks': snap['asks'][:max_levels],
            })

        df = pd.DataFrame(snapshot_list)

        # Get dimensions
        timestamps = sorted(df['datetime'].unique())
        venues = sorted(df['venue'].unique())
        assets = sorted(df['asset'].unique())
        sides = ['bid', 'ask']
        features = ['price', 'size', 'cumulative_size', 'distance_from_mid']

        # Create indices
        time_idx = {ts: i for i, ts in enumerate(timestamps)}
        venue_idx = {v: i for i, v in enumerate(venues)}
        asset_idx = {a: i for i, a in enumerate(assets)}
        level_idx = {i: i for i in range(max_levels)}
        side_idx = {'bid': 0, 'ask': 1}
        feature_idx = {f: i for i, f in enumerate(features)}

        # Initialize tensor
        shape = (len(timestamps), len(venues), len(assets), max_levels, 2, len(features))
        tensor = np.zeros(shape)

        # Fill tensor
        for _, row in df.iterrows():
            t_i = time_idx[row['datetime']]
            v_i = venue_idx[row['venue']]
            a_i = asset_idx[row['asset']]

            # Compute mid price
            if len(row['bids']) > 0 and len(row['asks']) > 0:
                mid_price = (row['bids'][0][0] + row['asks'][0][0]) / 2
            else:
                continue

            # Process bids
            cumulative = 0
            for level, (price, size) in enumerate(row['bids'][:max_levels]):
                cumulative += size
                tensor[t_i, v_i, a_i, level, 0, 0] = price
                tensor[t_i, v_i, a_i, level, 0, 1] = size
                tensor[t_i, v_i, a_i, level, 0, 2] = cumulative
                tensor[t_i, v_i, a_i, level, 0, 3] = (mid_price - price) / mid_price

            # Process asks
            cumulative = 0
            for level, (price, size) in enumerate(row['asks'][:max_levels]):
                cumulative += size
                tensor[t_i, v_i, a_i, level, 1, 0] = price
                tensor[t_i, v_i, a_i, level, 1, 1] = size
                tensor[t_i, v_i, a_i, level, 1, 2] = cumulative
                tensor[t_i, v_i, a_i, level, 1, 3] = (price - mid_price) / mid_price

        metadata = TensorMetadata(
            shape=shape,
            mode=TensorMode.ORDERBOOK,
            dimension_names=['time', 'venue', 'asset', 'level', 'side', 'feature'],
            dimension_indices={
                'time': time_idx,
                'venue': venue_idx,
                'asset': asset_idx,
                'level': level_idx,
                'side': side_idx,
                'feature': feature_idx
            },
            timestamp_range=(timestamps[0], timestamps[-1]),
            assets=assets,
            venues=venues
        )

        self.metadata = metadata
        return tensor, metadata

    def build_defi_composite_tensor(
        self,
        cex_data: pd.DataFrame,
        dex_data: pd.DataFrame,
        features: List[str],
        time_window: str = '1H'
    ) -> Tuple[np.ndarray, TensorMetadata]:
        """
        Build composite CEX/DEX tensor showing same asset across venues.

        This captures the multi-dimensional nature of DeFi assets:
        - CEX: order book microstructure
        - DEX: AMM bonding curve dynamics

        Args:
            cex_data: CEX time-series data
            dex_data: DEX time-series data
            features: Common features across both
            time_window: Time resampling

        Returns:
            (tensor, metadata) tuple
        """
        # Tag data sources
        cex_data = cex_data.copy()
        dex_data = dex_data.copy()
        cex_data['venue_type'] = 'cex'
        dex_data['venue_type'] = 'dex'

        # Combine
        combined = pd.concat([cex_data, dex_data], ignore_index=True)

        # Use standard microstructure tensor builder
        return self.build_microstructure_tensor(combined, features, time_window)

    def _fill_missing(self, tensor: np.ndarray, method: str) -> np.ndarray:
        """
        Fill missing values (zeros) in tensor.

        Args:
            tensor: Input tensor with missing values
            method: Fill method ('ffill', 'bfill', 'zero', 'mean')

        Returns:
            Filled tensor
        """
        if method == 'zero':
            return tensor

        elif method == 'mean':
            # Fill with feature-wise mean
            for i in range(tensor.shape[-1]):
                feature_slice = tensor[..., i]
                mask = feature_slice != 0
                if mask.any():
                    mean_val = feature_slice[mask].mean()
                    tensor[..., i] = np.where(feature_slice == 0, mean_val, feature_slice)
            return tensor

        elif method in ['ffill', 'bfill']:
            # Forward/backward fill along time dimension
            filled = tensor.copy()
            for v in range(tensor.shape[1]):
                for a in range(tensor.shape[2]):
                    for f in range(tensor.shape[3]):
                        series = filled[:, v, a, f]
                        if method == 'ffill':
                            # Forward fill
                            mask = series != 0
                            idx = np.where(mask, np.arange(len(series)), 0)
                            np.maximum.accumulate(idx, out=idx)
                            filled[:, v, a, f] = series[idx]
                        else:
                            # Backward fill
                            mask = series != 0
                            idx = np.where(mask, np.arange(len(series)), len(series) - 1)
                            idx = np.minimum.accumulate(idx[::-1])[::-1]
                            filled[:, v, a, f] = series[idx]
            return filled

        else:
            raise ValueError(f"Unknown fill method: {method}")

    def unfold_tensor(self, tensor: np.ndarray, mode: int) -> np.ndarray:
        """
        Unfold (matricize) tensor along specified mode.

        This is needed for CP/Tucker decomposition.

        Args:
            tensor: Input tensor
            mode: Which dimension to unfold along (0-indexed)

        Returns:
            Unfolded matrix
        """
        n_dims = len(tensor.shape)
        dims = list(range(n_dims))
        dims.remove(mode)
        dims.insert(0, mode)

        unfolded = np.transpose(tensor, dims)
        return unfolded.reshape(tensor.shape[mode], -1)

    def normalize_tensor(
        self,
        tensor: np.ndarray,
        method: str = 'zscore',
        axis: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Normalize tensor values.

        Args:
            tensor: Input tensor
            method: 'zscore', 'minmax', or 'robust'
            axis: Axis along which to normalize (None = global)

        Returns:
            (normalized_tensor, normalization_params)
        """
        if method == 'zscore':
            mean = np.mean(tensor, axis=axis, keepdims=True)
            std = np.std(tensor, axis=axis, keepdims=True)
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            normalized = (tensor - mean) / std
            params = {'mean': mean, 'std': std}

        elif method == 'minmax':
            min_val = np.min(tensor, axis=axis, keepdims=True)
            max_val = np.max(tensor, axis=axis, keepdims=True)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            normalized = (tensor - min_val) / range_val
            params = {'min': min_val, 'max': max_val}

        elif method == 'robust':
            # Use median and IQR for robustness to outliers
            median = np.median(tensor, axis=axis, keepdims=True)
            q75, q25 = np.percentile(tensor, [75, 25], axis=axis, keepdims=True)
            iqr = q75 - q25
            iqr = np.where(iqr == 0, 1, iqr)
            normalized = (tensor - median) / iqr
            params = {'median': median, 'iqr': iqr}

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized, params

    def compute_tensor_statistics(self, tensor: np.ndarray) -> Dict:
        """
        Compute summary statistics for tensor.

        Args:
            tensor: Input tensor

        Returns:
            Dict of statistics
        """
        return {
            'shape': tensor.shape,
            'size': tensor.size,
            'sparsity': np.mean(tensor == 0),
            'mean': np.mean(tensor),
            'std': np.std(tensor),
            'min': np.min(tensor),
            'max': np.max(tensor),
            'n_dims': len(tensor.shape),
            'memory_mb': tensor.nbytes / (1024 ** 2),
        }


if __name__ == "__main__":
    # Test tensor builder with synthetic data
    print("=== Testing Tensor Builder ===\n")

    # Create synthetic market data
    np.random.seed(42)
    timestamps = pd.date_range('2024-01-01', periods=100, freq='1H')
    venues = ['binance', 'coinbase', 'uniswap_v3']
    assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    features = ['mid_price', 'spread_bps', 'volume_imbalance', 'liquidity']

    data_rows = []
    for ts in timestamps:
        for venue in venues:
            for asset in assets:
                row = {
                    'datetime': ts,
                    'venue': venue,
                    'asset': asset,
                    'mid_price': 50000 + np.random.randn() * 1000,
                    'spread_bps': abs(np.random.randn() * 5),
                    'volume_imbalance': np.random.randn() * 0.3,
                    'liquidity': abs(np.random.randn() * 1e6),
                }
                data_rows.append(row)

    df = pd.DataFrame(data_rows)

    # Build tensor
    builder = TensorBuilder()
    tensor, metadata = builder.build_microstructure_tensor(df, features, time_window='1H')

    print(f"Tensor Shape: {tensor.shape}")
    print(f"Dimensions: {metadata.dimension_names}")
    print(f"\nTensor Statistics:")
    stats = builder.compute_tensor_statistics(tensor)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test normalization
    print("\n=== Testing Normalization ===")
    normalized, params = builder.normalize_tensor(tensor, method='zscore')
    print(f"Original mean: {tensor.mean():.4f}, std: {tensor.std():.4f}")
    print(f"Normalized mean: {normalized.mean():.4f}, std: {normalized.std():.4f}")

    # Test unfolding
    print("\n=== Testing Tensor Unfolding ===")
    for mode in range(len(tensor.shape)):
        unfolded = builder.unfold_tensor(tensor, mode)
        print(f"Mode-{mode} unfolding: {tensor.shape} → {unfolded.shape}")
