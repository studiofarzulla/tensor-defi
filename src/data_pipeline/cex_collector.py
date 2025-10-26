"""
CEX Data Collector - Binance, Coinbase via CCXT

Fetches order book snapshots, trades, and OHLCV data from centralized exchanges.
Designed to build time-series suitable for tensor construction.
"""

import ccxt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time


class CEXCollector:
    """Collect multi-venue CEX data for tensor construction."""

    def __init__(self, exchanges: Optional[List[str]] = None):
        """
        Initialize CEX collector with specified exchanges.

        Args:
            exchanges: List of exchange names (e.g., ['binance', 'coinbase'])
                      Defaults to ['binance'] if not specified
        """
        self.exchanges = exchanges or ['binance']
        self.exchange_instances = {}

        for exchange_name in self.exchanges:
            try:
                exchange_class = getattr(ccxt, exchange_name)
                self.exchange_instances[exchange_name] = exchange_class({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                print(f"✓ Initialized {exchange_name}")
            except Exception as e:
                print(f"✗ Failed to initialize {exchange_name}: {e}")

    def fetch_orderbook_snapshot(
        self,
        symbol: str,
        exchange: str,
        limit: int = 20
    ) -> Dict:
        """
        Fetch current order book snapshot.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            exchange: Exchange name
            limit: Number of price levels per side

        Returns:
            Dict with bids, asks, timestamp
        """
        if exchange not in self.exchange_instances:
            raise ValueError(f"Exchange {exchange} not initialized")

        ex = self.exchange_instances[exchange]
        orderbook = ex.fetch_order_book(symbol, limit=limit)

        return {
            'symbol': symbol,
            'exchange': exchange,
            'timestamp': orderbook['timestamp'],
            'datetime': orderbook['datetime'],
            'bids': np.array(orderbook['bids']),  # [[price, size], ...]
            'asks': np.array(orderbook['asks']),
        }

    def fetch_ohlcv(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = '1h',
        since: Optional[int] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data.

        Args:
            symbol: Trading pair
            exchange: Exchange name
            timeframe: Candle interval ('1m', '5m', '1h', '1d')
            since: Start timestamp in milliseconds
            limit: Number of candles

        Returns:
            DataFrame with timestamp, open, high, low, close, volume
        """
        if exchange not in self.exchange_instances:
            raise ValueError(f"Exchange {exchange} not initialized")

        ex = self.exchange_instances[exchange]
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        df['exchange'] = exchange
        df['timeframe'] = timeframe

        return df

    def fetch_recent_trades(
        self,
        symbol: str,
        exchange: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch recent trades.

        Args:
            symbol: Trading pair
            exchange: Exchange name
            limit: Number of recent trades

        Returns:
            DataFrame with timestamp, price, amount, side
        """
        if exchange not in self.exchange_instances:
            raise ValueError(f"Exchange {exchange} not initialized")

        ex = self.exchange_instances[exchange]
        trades = ex.fetch_trades(symbol, limit=limit)

        df = pd.DataFrame(trades)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        df['exchange'] = exchange

        return df[['datetime', 'timestamp', 'price', 'amount', 'side', 'symbol', 'exchange']]

    def _fetch_ohlcv_chunked(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        since_ms: int,
        until_ms: int,
        chunk_size: int = 500
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data in chunks to handle API pagination limits.

        Binance and most exchanges limit responses to ~500 candles per request.
        This method automatically chunks the time period and concatenates results.

        Args:
            symbol: Trading pair
            exchange: Exchange name
            timeframe: Candle interval ('1h', '1d', etc.)
            since_ms: Start timestamp in milliseconds
            until_ms: End timestamp in milliseconds
            chunk_size: Max candles per API call (default 500 for safety)

        Returns:
            Combined DataFrame with all chunks
        """
        # Calculate timeframe duration in milliseconds
        timeframe_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }

        if timeframe not in timeframe_ms:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(timeframe_ms.keys())}")

        interval_ms = timeframe_ms[timeframe]
        chunk_duration_ms = chunk_size * interval_ms

        # Calculate total chunks needed
        total_duration_ms = until_ms - since_ms
        total_chunks = int(np.ceil(total_duration_ms / chunk_duration_ms))

        print(f"  Total duration: {total_duration_ms / (24*60*60*1000):.1f} days")
        print(f"  Chunks needed: {total_chunks} (max {chunk_size} candles each)")

        all_chunks = []
        current_since = since_ms
        retry_count = 0
        max_retries = 3

        for chunk_idx in range(total_chunks):
            # Calculate chunk end time
            current_until = min(current_since + chunk_duration_ms, until_ms)

            # Progress indication
            progress_pct = (chunk_idx + 1) / total_chunks * 100
            print(f"  Chunk {chunk_idx + 1}/{total_chunks} ({progress_pct:.1f}%): "
                  f"{datetime.fromtimestamp(current_since/1000).strftime('%Y-%m-%d %H:%M')} to "
                  f"{datetime.fromtimestamp(current_until/1000).strftime('%Y-%m-%d %H:%M')}")

            try:
                # Fetch chunk
                chunk_df = self.fetch_ohlcv(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    since=current_since,
                    limit=chunk_size
                )

                if chunk_df.empty:
                    print(f"    ⚠ Empty chunk received, skipping")
                    current_since = current_until
                    continue

                # Validate chunk
                chunk_start = chunk_df['timestamp'].min()
                chunk_end = chunk_df['timestamp'].max()
                chunk_count = len(chunk_df)

                print(f"    ✓ Received {chunk_count} candles "
                      f"({datetime.fromtimestamp(chunk_start/1000).strftime('%Y-%m-%d %H:%M')} to "
                      f"{datetime.fromtimestamp(chunk_end/1000).strftime('%Y-%m-%d %H:%M')})")

                all_chunks.append(chunk_df)

                # Move to next chunk (use last timestamp + interval to avoid gaps)
                current_since = chunk_end + interval_ms

                # Reset retry counter on success
                retry_count = 0

                # Rate limiting - exchange already has enableRateLimit=True,
                # but add small delay between chunks to be respectful
                time.sleep(0.1)

            except Exception as e:
                retry_count += 1
                print(f"    ✗ Error fetching chunk: {e}")

                if retry_count >= max_retries:
                    print(f"    ✗ Max retries ({max_retries}) exceeded, skipping this chunk")
                    current_since = current_until
                    retry_count = 0
                    continue

                # Exponential backoff: 2^retry * base_delay
                backoff_delay = (2 ** retry_count) * 1.0
                print(f"    ⟳ Retry {retry_count}/{max_retries} after {backoff_delay:.1f}s backoff...")
                time.sleep(backoff_delay)

                # Retry same chunk (don't increment current_since)
                chunk_idx -= 1  # Will be incremented by loop
                continue

            # Check if we've reached the end
            if current_since >= until_ms:
                break

        if not all_chunks:
            raise ValueError(f"No data collected for {symbol} on {exchange}")

        # Combine all chunks
        combined = pd.concat(all_chunks, ignore_index=True)

        # Remove duplicates (can happen at chunk boundaries)
        combined = combined.drop_duplicates(subset=['timestamp'], keep='first')

        # Sort by timestamp
        combined = combined.sort_values('timestamp').reset_index(drop=True)

        # Validate no gaps (for hourly data)
        if timeframe == '1h' and len(combined) > 1:
            timestamps = combined['timestamp'].values
            diffs = np.diff(timestamps)
            expected_diff = interval_ms

            # Allow some tolerance for exchange timestamp inconsistencies
            gaps = np.where(diffs > expected_diff * 1.5)[0]

            if len(gaps) > 0:
                print(f"    ⚠ Found {len(gaps)} gaps in hourly data")
                for gap_idx in gaps[:5]:  # Show first 5 gaps
                    gap_start = datetime.fromtimestamp(timestamps[gap_idx]/1000)
                    gap_end = datetime.fromtimestamp(timestamps[gap_idx + 1]/1000)
                    gap_hours = (timestamps[gap_idx + 1] - timestamps[gap_idx]) / (60*60*1000)
                    print(f"      Gap: {gap_start} to {gap_end} ({gap_hours:.1f} hours)")

        return combined

    def collect_multimarket_timeseries(
        self,
        symbols: List[str],
        exchanges: Optional[List[str]] = None,
        timeframe: str = '1h',
        lookback_days: int = 30
    ) -> pd.DataFrame:
        """
        Collect OHLCV data across multiple symbols and exchanges.

        This creates the time dimension for tensor construction.
        Automatically handles API pagination for large date ranges.

        Args:
            symbols: List of trading pairs
            exchanges: List of exchanges (defaults to all initialized)
            timeframe: Candle interval
            lookback_days: How many days of history

        Returns:
            Combined DataFrame with all market data

        Examples:
            >>> # Small request (7 days) - single API call per symbol
            >>> collector.collect_multimarket_timeseries(['BTC/USDT'], lookback_days=7)

            >>> # Large request (365 days) - ~18 chunks per symbol
            >>> collector.collect_multimarket_timeseries(
            ...     ['BTC/USDT', 'ETH/USDT'],
            ...     lookback_days=365
            ... )
        """
        exchanges = exchanges or list(self.exchange_instances.keys())

        # Calculate time window
        now = datetime.now()
        since = now - timedelta(days=lookback_days)
        since_ms = int(since.timestamp() * 1000)
        until_ms = int(now.timestamp() * 1000)

        print(f"\nCollecting {lookback_days} days of {timeframe} data")
        print(f"Time range: {since.strftime('%Y-%m-%d %H:%M')} to {now.strftime('%Y-%m-%d %H:%M')}")
        print(f"Symbols: {symbols}")
        print(f"Exchanges: {exchanges}\n")

        all_data = []
        total_symbols = len(symbols) * len(exchanges)
        current_symbol = 0

        for symbol in symbols:
            for exchange in exchanges:
                current_symbol += 1
                print(f"[{current_symbol}/{total_symbols}] Fetching {symbol} from {exchange}...")

                try:
                    # Use chunked fetching for all requests
                    # (automatically handles both small and large date ranges)
                    df = self._fetch_ohlcv_chunked(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        since_ms=since_ms,
                        until_ms=until_ms
                    )

                    all_data.append(df)

                    print(f"  ✓ Collected {len(df)} candles for {symbol}")
                    print()

                except Exception as e:
                    print(f"  ✗ Failed {symbol} on {exchange}: {e}")
                    print()
                    continue

        if not all_data:
            raise ValueError("No data collected from any symbol/exchange combination")

        # Combine all data
        print("Combining all market data...")
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(['datetime', 'exchange', 'symbol'])

        print(f"✓ Total data points: {len(combined)}")
        print(f"  Date range: {combined['datetime'].min()} to {combined['datetime'].max()}")
        print(f"  Symbols: {sorted(combined['symbol'].unique().tolist())}")
        print(f"  Exchanges: {sorted(combined['exchange'].unique().tolist())}")

        return combined

    def collect_orderbook_snapshots(
        self,
        symbols: List[str],
        exchanges: Optional[List[str]] = None,
        num_snapshots: int = 100,
        interval_seconds: int = 60,
        levels: int = 20
    ) -> List[Dict]:
        """
        Collect order book snapshots over time.

        Creates time-series of order book states for tensor construction.

        Args:
            symbols: Trading pairs to track
            exchanges: Exchanges to query
            num_snapshots: How many snapshots to collect
            interval_seconds: Time between snapshots
            levels: Price levels per side

        Returns:
            List of order book snapshots
        """
        exchanges = exchanges or list(self.exchange_instances.keys())
        snapshots = []

        for i in range(num_snapshots):
            snapshot_time = datetime.now()

            for symbol in symbols:
                for exchange in exchanges:
                    try:
                        ob = self.fetch_orderbook_snapshot(symbol, exchange, levels)
                        ob['snapshot_id'] = i
                        snapshots.append(ob)
                    except Exception as e:
                        print(f"✗ Snapshot {i} failed for {symbol} on {exchange}: {e}")
                        continue

            print(f"Collected snapshot {i+1}/{num_snapshots} at {snapshot_time}")

            if i < num_snapshots - 1:
                time.sleep(interval_seconds)

        return snapshots

    def compute_microstructure_features(self, orderbook_snapshot: Dict) -> Dict:
        """
        Extract microstructure features from order book snapshot.

        Features for tensor construction:
        - Bid-ask spread
        - Mid price
        - Volume imbalance
        - Depth at different levels
        - Price impact estimates

        Args:
            orderbook_snapshot: Order book data from fetch_orderbook_snapshot

        Returns:
            Dict of computed features
        """
        bids = orderbook_snapshot['bids']
        asks = orderbook_snapshot['asks']

        if len(bids) == 0 or len(asks) == 0:
            return {}

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price) * 10000

        # Volume imbalance (normalized)
        bid_volume = np.sum(bids[:, 1])
        ask_volume = np.sum(asks[:, 1])
        total_volume = bid_volume + ask_volume
        volume_imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0

        # Depth at different levels (cumulative)
        depth_levels = [5, 10, 20]
        bid_depth = {f'bid_depth_{n}': np.sum(bids[:n, 1]) if len(bids) >= n else 0
                     for n in depth_levels}
        ask_depth = {f'ask_depth_{n}': np.sum(asks[:n, 1]) if len(asks) >= n else 0
                     for n in depth_levels}

        # Price impact (cost to buy/sell a certain size)
        def compute_price_impact(side: np.ndarray, size_usd: float) -> float:
            """Compute price impact for market order of given USD size."""
            cumulative_volume = 0
            weighted_price = 0

            for price, volume in side:
                remaining = size_usd - cumulative_volume * price
                if remaining <= 0:
                    break

                volume_to_take = min(volume, remaining / price)
                weighted_price += price * volume_to_take
                cumulative_volume += volume_to_take

            avg_price = weighted_price / cumulative_volume if cumulative_volume > 0 else 0
            return (avg_price - mid_price) / mid_price if mid_price > 0 else 0

        impact_sizes = [1000, 10000, 100000]  # USD
        buy_impact = {f'buy_impact_{s}': compute_price_impact(asks, s)
                      for s in impact_sizes}
        sell_impact = {f'sell_impact_{s}': compute_price_impact(bids, s)
                       for s in impact_sizes}

        return {
            'mid_price': mid_price,
            'spread': spread,
            'spread_bps': spread_bps,
            'volume_imbalance': volume_imbalance,
            **bid_depth,
            **ask_depth,
            **buy_impact,
            **sell_impact,
        }


if __name__ == "__main__":
    # Test the collector
    collector = CEXCollector(['binance'])

    # Test order book fetch
    print("\n=== Testing Order Book Fetch ===")
    ob = collector.fetch_orderbook_snapshot('BTC/USDT', 'binance', limit=5)
    print(f"Timestamp: {ob['datetime']}")
    print(f"Best Bid: {ob['bids'][0]}")
    print(f"Best Ask: {ob['asks'][0]}")

    # Test microstructure features
    print("\n=== Testing Microstructure Features ===")
    features = collector.compute_microstructure_features(ob)
    for key, value in features.items():
        print(f"{key}: {value:.6f}")

    # Test OHLCV fetch
    print("\n=== Testing OHLCV Fetch ===")
    df = collector.fetch_ohlcv('BTC/USDT', 'binance', '1h', limit=10)
    print(df[['datetime', 'open', 'high', 'low', 'close', 'volume']].head())
