"""Temporal dynamics analysis for TENSOR-DEFI."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """Analyze temporal dynamics of alignment."""

    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_rolling_stats(self, window_months: int = 6, step_months: int = 3) -> dict:
        """Compute market statistics in rolling windows."""
        market_dir = self.data_dir / "market"
        parquet_files = list(market_dir.glob("*.parquet"))

        if not parquet_files:
            return {}

        all_data = {}
        for f in parquet_files:
            symbol = f.stem.replace("_ohlcv", "").upper()
            df = pd.read_parquet(f)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            all_data[symbol] = df

        overall_min = min(df.index.min() for df in all_data.values())
        overall_max = max(df.index.max() for df in all_data.values())
        full_range = (overall_max - overall_min).days

        full_coverage = {sym: df for sym, df in all_data.items()
                        if (df.index.max() - df.index.min()).days / full_range > 0.9}

        if not full_coverage:
            return {}

        all_data = full_coverage
        min_date = max(df.index.min() for df in all_data.values())
        max_date = min(df.index.max() for df in all_data.values())

        logger.info(f"Assets with full coverage: {len(all_data)}")
        logger.info(f"Temporal range: {min_date} to {max_date}")

        windows = []
        current = min_date
        while current + pd.DateOffset(months=window_months) <= max_date:
            windows.append((current, current + pd.DateOffset(months=window_months)))
            current += pd.DateOffset(months=step_months)

        logger.info(f"Generated {len(windows)} rolling windows")

        results = []
        for i, (start, end) in enumerate(windows):
            window_stats = {}
            for symbol, df in all_data.items():
                window_df = df[(df.index >= start) & (df.index < end)]
                if len(window_df) < 100:
                    continue
                returns = window_df['close'].pct_change().dropna()
                if len(returns) > 0:
                    cummax = window_df['close'].cummax()
                    drawdown = ((window_df['close'] - cummax) / cummax).min()
                    window_stats[symbol] = {
                        'mean_return': float(returns.mean()),
                        'volatility': float(returns.std()),
                        'sharpe': float(returns.mean() / returns.std()) if returns.std() > 0 else 0,
                        'max_drawdown': float(drawdown),
                        'avg_volume': float(window_df['volume'].mean()) if 'volume' in window_df else 0,
                    }
            if window_stats:
                results.append({
                    'window_id': i,
                    'start': start.isoformat(),
                    'end': end.isoformat(),
                    'n_assets': len(window_stats),
                    'stats': window_stats
                })

        return {'windows': results, 'window_months': window_months, 'total_windows': len(results)}

    def analyze_alignment_stability(self, claims_matrix: np.ndarray, rolling_stats: dict, symbols: list[str]) -> dict:
        """Analyze alignment across time windows."""
        import sys
        sys.path.insert(0, str(self.data_dir.parent / "src"))
        from alignment.procrustes import ProcrustesAlignment
        from alignment.congruence import CongruenceCoefficient

        aligner = ProcrustesAlignment()
        congruence = CongruenceCoefficient()
        results = []

        for window in rolling_stats.get('windows', []):
            common = [s for s in window['stats'].keys() if s in symbols]
            if len(common) < 5:
                continue

            stats_list, claims_indices = [], []
            for sym in common:
                s = window['stats'][sym]
                stats_list.append([s['mean_return'], s['volatility'], s['sharpe'], s['max_drawdown'], s['avg_volume']])
                claims_indices.append(symbols.index(sym))

            if len(stats_list) < 5:
                continue

            window_claims = claims_matrix[claims_indices]
            window_stats = np.array(stats_list)
            window_stats = (window_stats - window_stats.mean(axis=0)) / (window_stats.std(axis=0) + 1e-8)

            try:
                result = aligner.align_matrices(window_claims, window_stats)
                aligned = result['source_rotated']
                target = result['target_centered']
                cong_result = congruence.matrix_congruence(aligned, target)
                phi = cong_result['mean_phi']
                results.append({
                    'window_id': window['window_id'],
                    'start': window['start'],
                    'end': window['end'],
                    'n_common': len(common),
                    'phi': float(phi),
                })
            except Exception as e:
                logger.warning(f"Window {window['window_id']} failed: {e}")

        return {
            'temporal_alignment': results,
            'mean_phi': float(np.mean([r['phi'] for r in results])) if results else 0,
            'std_phi': float(np.std([r['phi'] for r in results])) if results else 0,
            'n_windows': len(results)
        }

    def save_results(self, results: dict):
        import json
        output_path = self.output_dir / "temporal_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved: {output_path}")

        print(f"\n{'='*60}")
        print("TEMPORAL ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Windows analyzed: {results.get('n_windows', 0)}")
        print(f"Mean φ across time: {results.get('mean_phi', 0):.3f}")
        print(f"Std φ across time: {results.get('std_phi', 0):.3f}")
        if results.get('temporal_alignment'):
            print(f"\nPer-window φ:")
            for r in results['temporal_alignment']:
                print(f"  {r['start'][:7]} - {r['end'][:7]}: φ = {r['phi']:.3f}")
