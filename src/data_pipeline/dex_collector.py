"""
DEX Data Collector - Uniswap V3, Curve via The Graph

Fetches pool state, swaps, and liquidity data from decentralized exchanges.
This captures the unique microstructure of AMMs vs order books.
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time


class DEXCollector:
    """Collect DEX data from The Graph subgraphs."""

    # The Graph API endpoints
    SUBGRAPH_URLS = {
        'uniswap_v3': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
        'curve': 'https://api.thegraph.com/subgraphs/name/messari/curve-finance-ethereum',
    }

    def __init__(self, dexes: Optional[List[str]] = None):
        """
        Initialize DEX collector.

        Args:
            dexes: List of DEX names (e.g., ['uniswap_v3', 'curve'])
        """
        self.dexes = dexes or ['uniswap_v3']
        self.session = requests.Session()

        for dex in self.dexes:
            if dex not in self.SUBGRAPH_URLS:
                print(f"⚠ Warning: {dex} not supported, skipping")
            else:
                print(f"✓ Initialized {dex} collector")

    def _query_graph(self, dex: str, query: str) -> Dict:
        """
        Execute GraphQL query against The Graph.

        Args:
            dex: DEX name
            query: GraphQL query string

        Returns:
            Response data
        """
        if dex not in self.SUBGRAPH_URLS:
            raise ValueError(f"DEX {dex} not supported")

        url = self.SUBGRAPH_URLS[dex]

        try:
            response = self.session.post(url, json={'query': query}, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'errors' in data:
                raise Exception(f"GraphQL errors: {data['errors']}")

            return data.get('data', {})

        except Exception as e:
            print(f"✗ Query failed for {dex}: {e}")
            raise

    def fetch_uniswap_pool_state(self, pool_address: str) -> Dict:
        """
        Fetch current Uniswap V3 pool state.

        Args:
            pool_address: Pool contract address

        Returns:
            Pool state including liquidity, tick, price, volume
        """
        query = f"""
        {{
          pool(id: "{pool_address.lower()}") {{
            id
            token0 {{
              symbol
              decimals
            }}
            token1 {{
              symbol
              decimals
            }}
            liquidity
            sqrtPrice
            tick
            token0Price
            token1Price
            volumeUSD
            feeTier
            txCount
          }}
        }}
        """

        data = self._query_graph('uniswap_v3', query)
        return data.get('pool', {})

    def fetch_uniswap_swaps(
        self,
        pool_address: str,
        num_swaps: int = 100,
        start_timestamp: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch recent swaps from Uniswap V3 pool.

        Args:
            pool_address: Pool contract address
            num_swaps: Number of swaps to fetch
            start_timestamp: Unix timestamp to start from

        Returns:
            DataFrame of swap events
        """
        timestamp_filter = f'timestamp_gte: {start_timestamp}' if start_timestamp else ''

        query = f"""
        {{
          swaps(
            first: {num_swaps}
            where: {{pool: "{pool_address.lower()}", {timestamp_filter}}}
            orderBy: timestamp
            orderDirection: desc
          ) {{
            id
            timestamp
            amount0
            amount1
            amountUSD
            sqrtPriceX96
            tick
            sender
            recipient
          }}
        }}
        """

        data = self._query_graph('uniswap_v3', query)
        swaps = data.get('swaps', [])

        df = pd.DataFrame(swaps)
        if not df.empty:
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['pool'] = pool_address

        return df

    def fetch_uniswap_pool_hourly_data(
        self,
        pool_address: str,
        num_hours: int = 168  # 1 week default
    ) -> pd.DataFrame:
        """
        Fetch hourly aggregated pool data.

        This is the DEX equivalent of CEX OHLCV data.

        Args:
            pool_address: Pool contract address
            num_hours: Number of hourly periods

        Returns:
            DataFrame with hourly pool metrics
        """
        query = f"""
        {{
          poolHourDatas(
            first: {num_hours}
            where: {{pool: "{pool_address.lower()}"}}
            orderBy: periodStartUnix
            orderDirection: desc
          ) {{
            id
            periodStartUnix
            liquidity
            sqrtPrice
            token0Price
            token1Price
            volumeUSD
            volumeToken0
            volumeToken1
            txCount
            open
            high
            low
            close
          }}
        }}
        """

        data = self._query_graph('uniswap_v3', query)
        hourly = data.get('poolHourDatas', [])

        df = pd.DataFrame(hourly)
        if not df.empty:
            df['periodStartUnix'] = pd.to_numeric(df['periodStartUnix'])
            df['datetime'] = pd.to_datetime(df['periodStartUnix'], unit='s')
            df['pool'] = pool_address
            df = df.sort_values('datetime')

        return df

    def fetch_curve_pool_state(self, pool_address: str) -> Dict:
        """
        Fetch Curve pool state.

        Curve has different mechanics (stable swaps, metapools).

        Args:
            pool_address: Pool contract address

        Returns:
            Pool state data
        """
        query = f"""
        {{
          liquidityPool(id: "{pool_address.lower()}") {{
            id
            name
            symbol
            inputTokens {{
              symbol
              decimals
            }}
            totalValueLockedUSD
            cumulativeVolumeUSD
            inputTokenBalances
            outputTokenSupply
          }}
        }}
        """

        data = self._query_graph('curve', query)
        return data.get('liquidityPool', {})

    def compute_amm_microstructure_features(self, pool_state: Dict, dex_type: str) -> Dict:
        """
        Extract microstructure features from AMM pool state.

        AMM features differ from order book features:
        - No discrete levels, continuous bonding curve
        - Liquidity depth vs price impact
        - Impermanent loss exposure
        - Fee tier effects

        Args:
            pool_state: Pool state from fetch_*_pool_state
            dex_type: 'uniswap_v3' or 'curve'

        Returns:
            Dict of computed features
        """
        if dex_type == 'uniswap_v3':
            return self._uniswap_features(pool_state)
        elif dex_type == 'curve':
            return self._curve_features(pool_state)
        else:
            return {}

    def _uniswap_features(self, pool: Dict) -> Dict:
        """Extract Uniswap V3 specific features."""
        if not pool:
            return {}

        features = {
            'liquidity': float(pool.get('liquidity', 0)),
            'sqrt_price': float(pool.get('sqrtPrice', 0)),
            'tick': int(pool.get('tick', 0)),
            'token0_price': float(pool.get('token0Price', 0)),
            'token1_price': float(pool.get('token1Price', 0)),
            'volume_usd': float(pool.get('volumeUSD', 0)),
            'fee_tier': int(pool.get('feeTier', 0)),
            'tx_count': int(pool.get('txCount', 0)),
        }

        # Compute price impact for different trade sizes
        # Using constant product formula: x * y = k
        if features['liquidity'] > 0 and features['token0_price'] > 0:
            for size_usd in [1000, 10000, 100000]:
                # Simplified price impact (full calculation requires tick math)
                size_in_token = size_usd / features['token0_price']
                impact_pct = (size_in_token / features['liquidity']) * 100
                features[f'price_impact_{size_usd}'] = impact_pct

        return features

    def _curve_features(self, pool: Dict) -> Dict:
        """Extract Curve specific features."""
        if not pool:
            return {}

        return {
            'tvl_usd': float(pool.get('totalValueLockedUSD', 0)),
            'cumulative_volume_usd': float(pool.get('cumulativeVolumeUSD', 0)),
            'num_tokens': len(pool.get('inputTokens', [])),
            'output_token_supply': float(pool.get('outputTokenSupply', 0)),
        }

    def collect_multimarket_dex_data(
        self,
        pools: List[Dict[str, str]],  # [{'dex': 'uniswap_v3', 'address': '0x...'}]
        lookback_hours: int = 168
    ) -> pd.DataFrame:
        """
        Collect hourly data across multiple DEX pools.

        Args:
            pools: List of pool configs with 'dex' and 'address'
            lookback_hours: Hours of history

        Returns:
            Combined DataFrame
        """
        all_data = []

        for pool_config in pools:
            dex = pool_config['dex']
            address = pool_config['address']
            pair_name = pool_config.get('name', address[:8])

            try:
                print(f"Fetching {pair_name} from {dex}...")

                if dex == 'uniswap_v3':
                    df = self.fetch_uniswap_pool_hourly_data(address, lookback_hours)
                    df['dex'] = dex
                    df['pair_name'] = pair_name
                    all_data.append(df)
                elif dex == 'curve':
                    # Curve doesn't have hourly data in standard subgraph
                    # Would need custom aggregation from swap events
                    print(f"⚠ Hourly data not available for {dex}, skipping")
                    continue

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"✗ Failed {pair_name} on {dex}: {e}")
                continue

        if not all_data:
            raise ValueError("No DEX data collected")

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values('datetime')

        return combined


# Common pool addresses for testing
COMMON_POOLS = {
    'uniswap_v3': {
        'WETH_USDC_005': '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640',  # 0.05% fee
        'WETH_USDT_005': '0x11b815efb8f581194ae79006d24e0d814b7697f6',  # 0.05% fee
        'WBTC_WETH_005': '0xcbcdf9626bc03e24f779434178a73a0b4bad62ed',  # 0.05% fee
        'WETH_USDC_03': '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8',   # 0.3% fee
    }
}


if __name__ == "__main__":
    # Test the collector
    collector = DEXCollector(['uniswap_v3'])

    print("\n=== Testing Uniswap V3 Pool State ===")
    pool_address = COMMON_POOLS['uniswap_v3']['WETH_USDC_005']
    pool_state = collector.fetch_uniswap_pool_state(pool_address)

    if pool_state:
        print(f"Pool: {pool_state.get('token0', {}).get('symbol')}/{pool_state.get('token1', {}).get('symbol')}")
        print(f"Liquidity: {pool_state.get('liquidity')}")
        print(f"Token0 Price: {pool_state.get('token0Price')}")
        print(f"Volume USD: {pool_state.get('volumeUSD')}")

        print("\n=== Computing AMM Microstructure Features ===")
        features = collector.compute_amm_microstructure_features(pool_state, 'uniswap_v3')
        for key, value in features.items():
            print(f"{key}: {value}")

    print("\n=== Testing Hourly Data Fetch ===")
    df = collector.fetch_uniswap_pool_hourly_data(pool_address, num_hours=24)
    if not df.empty:
        print(df[['datetime', 'token0Price', 'volumeUSD', 'liquidity']].head(10))
