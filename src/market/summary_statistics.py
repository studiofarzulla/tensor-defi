#!/usr/bin/env python3
"""
Market Summary Statistics for TENSOR-DEFI

Computes N×7 summary statistics matrix from OHLCV data.
Statistics: mean_return, volatility, sharpe, max_drawdown, avg_volume, vol_volatility, trend
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


STATISTICS = [
    'mean_return',
    'volatility',
    'sharpe',
    'max_drawdown',
    'avg_volume',
    'vol_volatility',
    'trend'
]


class SummaryStatistics:
    """Computes market summary statistics for each asset."""

    def __init__(self, market_dir: Path, output_dir: Path):
        self.market_dir = Path(market_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_returns(self, df: pd.DataFrame) -> pd.Series:
        """Compute log returns from close prices."""
        return np.log(df['close'] / df['close'].shift(1)).dropna()

    def compute_max_drawdown(self, prices: pd.Series) -> float:
        """Compute maximum drawdown."""
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return abs(drawdown.min())

    def compute_trend(self, prices: pd.Series) -> float:
        """Compute linear trend slope (normalized)."""
        x = np.arange(len(prices))
        slope, _, _, _, _ = stats.linregress(x, prices.values)
        # Normalize by mean price
        return slope / prices.mean()

    def compute_stats(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute all summary statistics for an asset."""
        returns = self.compute_returns(df)

        # Daily aggregation for cleaner stats
        df['date'] = df['timestamp'].dt.date
        daily = df.groupby('date').agg({
            'close': 'last',
            'volume': 'sum'
        }).reset_index()

        daily_returns = np.log(daily['close'] / daily['close'].shift(1)).dropna()

        return {
            'mean_return': daily_returns.mean() * 252,  # Annualized
            'volatility': daily_returns.std() * np.sqrt(252),  # Annualized
            'sharpe': (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0,
            'max_drawdown': self.compute_max_drawdown(daily['close']),
            'avg_volume': np.log1p(daily['volume'].mean()),  # Log-scaled
            'vol_volatility': daily['volume'].std() / daily['volume'].mean() if daily['volume'].mean() > 0 else 0,
            'trend': self.compute_trend(daily['close'])
        }

    def build_stats_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Build N×7 statistics matrix from all market data."""
        parquet_files = sorted(self.market_dir.glob("*.parquet"))

        if not parquet_files:
            raise ValueError(f"No market data found in {self.market_dir}")

        symbols = []
        stats_list = []

        for parquet_file in parquet_files:
            symbol = parquet_file.stem.replace('_ohlcv', '').upper()
            df = pd.read_parquet(parquet_file)

            if df.empty:
                logger.warning(f"{symbol}: Empty data, skipping")
                continue

            try:
                stats_dict = self.compute_stats(df)
                stats_vector = [stats_dict[stat] for stat in STATISTICS]
                symbols.append(symbol)
                stats_list.append(stats_vector)
                logger.info(f"{symbol}: stats computed")
            except Exception as e:
                logger.error(f"{symbol}: Failed - {e}")
                continue

        # Convert to matrix
        stats_matrix = np.array(stats_list)

        # Z-score normalization across entities
        stats_matrix = (stats_matrix - stats_matrix.mean(axis=0)) / (stats_matrix.std(axis=0) + 1e-8)

        # Save
        np.save(self.output_dir / "stats_matrix.npy", stats_matrix)

        metadata = {
            'symbols': symbols,
            'statistics': STATISTICS,
            'shape': list(stats_matrix.shape)
        }
        with open(self.output_dir / "stats_matrix_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save as CSV
        df_stats = pd.DataFrame(stats_matrix, index=symbols, columns=STATISTICS)
        df_stats.to_csv(self.output_dir / "stats_matrix.csv")

        logger.info(f"Saved stats matrix: {stats_matrix.shape}")
        self._print_summary(symbols, stats_matrix)

        return stats_matrix, symbols

    def _print_summary(self, symbols: list[str], matrix: np.ndarray):
        """Print statistics summary."""
        print(f"\n{'='*60}")
        print("MARKET STATISTICS SUMMARY")
        print(f"{'='*60}")
        print(f"Entities: {len(symbols)}")
        print(f"Statistics: {len(STATISTICS)}")

        print(f"\nStatistic ranges (z-scored):")
        for i, stat in enumerate(STATISTICS):
            col = matrix[:, i]
            print(f"  {stat:15s} min={col.min():.2f}  max={col.max():.2f}  std={col.std():.2f}")

        # Correlation matrix
        corr = np.corrcoef(matrix.T)
        print(f"\nStatistic correlations:")
        for i, stat1 in enumerate(STATISTICS):
            for j, stat2 in enumerate(STATISTICS):
                if j > i:
                    print(f"  {stat1} ↔ {stat2}: {corr[i,j]:.2f}")

        print(f"{'='*60}")


def main():
    """Run summary statistics computation."""
    base_path = Path(__file__).parent.parent.parent
    stats = SummaryStatistics(
        market_dir=base_path / "data" / "market",
        output_dir=base_path / "outputs" / "market"
    )
    stats.build_stats_matrix()


if __name__ == "__main__":
    main()
