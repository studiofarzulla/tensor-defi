"""
Full Alignment Analysis Pipeline

Collects market data for the 8 projects with functional profiles,
builds market feature matrix, and runs alignment testing.

Usage:
    python scripts/run_alignment_analysis.py

"""

import sys
from pathlib import Path
import json
import pickle
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.data_pipeline.cex_collector import CEXCollector
from src.alignment.alignment_test import AlignmentTest, AlignmentResult


def collect_market_data(symbols: list, lookback_days: int = 365) -> dict:
    """
    Collect market data for given symbols.
    
    Args:
        symbols: List of crypto symbols (e.g., ['BTC', 'ETH'])
        lookback_days: Days of history to collect
    
    Returns:
        Dict with market data per symbol
    """
    print("=" * 60)
    print("COLLECTING MARKET DATA")
    print("=" * 60)
    
    collector = CEXCollector(['binance'])
    
    # Map symbols to Binance trading pairs
    pairs = {s: f"{s}/USDT" for s in symbols}
    
    market_data = {}
    
    for symbol, pair in pairs.items():
        print(f"\n[{symbol}] Collecting {pair}...")
        
        try:
            df = collector.collect_multimarket_timeseries(
                symbols=[pair],
                timeframe='1d',
                lookback_days=lookback_days
            )
            
            if df is not None and len(df) > 0:
                market_data[symbol] = df
                print(f"  ✓ {len(df)} daily candles")
            else:
                print(f"  ✗ No data returned")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    return market_data


def compute_market_features(market_data: dict) -> np.ndarray:
    """
    Compute market features for each project.
    
    Features:
    1. Mean daily return
    2. Volatility (std of returns)
    3. Sharpe ratio (return / volatility)
    4. Max drawdown
    5. Average volume
    6. Volume volatility
    7. Price trend (linear regression slope)
    
    Args:
        market_data: Dict of DataFrames per symbol
    
    Returns:
        Feature matrix (n_symbols × n_features)
    """
    print("\n" + "=" * 60)
    print("COMPUTING MARKET FEATURES")
    print("=" * 60)
    
    features = []
    symbols = []
    
    for symbol, df in market_data.items():
        print(f"\n[{symbol}] Computing features...")
        
        try:
            # Calculate returns
            df = df.sort_values('datetime')
            close = df['close'].values
            volume = df['volume'].values
            
            # Log returns
            returns = np.diff(np.log(close))
            
            # Feature 1: Mean return (annualized)
            mean_return = np.mean(returns) * 365
            
            # Feature 2: Volatility (annualized)
            volatility = np.std(returns) * np.sqrt(365)
            
            # Feature 3: Sharpe ratio
            sharpe = mean_return / (volatility + 1e-10)
            
            # Feature 4: Max drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            # Feature 5: Average volume (log scale)
            avg_volume = np.log(np.mean(volume) + 1)
            
            # Feature 6: Volume volatility
            vol_vol = np.std(volume) / (np.mean(volume) + 1e-10)
            
            # Feature 7: Price trend (normalized slope)
            x = np.arange(len(close))
            slope, _ = np.polyfit(x, close / close[0], 1)
            trend = slope * 365  # Annualized
            
            feature_vec = [
                mean_return,
                volatility,
                sharpe,
                max_drawdown,
                avg_volume,
                vol_vol,
                trend,
            ]
            
            features.append(feature_vec)
            symbols.append(symbol)
            
            print(f"  Return: {mean_return:.2%}, Vol: {volatility:.2%}, Sharpe: {sharpe:.2f}")
            
        except Exception as e:
            print(f"  ✗ Error computing features: {e}")
    
    feature_matrix = np.array(features)
    
    # Normalize features (z-score)
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0) + 1e-10
    normalized = (feature_matrix - mean) / std
    
    return normalized, symbols


def run_alignment_analysis(
    claims_path: str = "outputs/nlp/claims_matrix.npz",
    output_dir: str = "outputs/alignment",
    lookback_days: int = 365
) -> AlignmentResult:
    """
    Run full alignment analysis pipeline.
    
    Args:
        claims_path: Path to claims matrix from NLP pipeline
        output_dir: Directory for output files
        lookback_days: Days of market data to collect
    
    Returns:
        AlignmentResult
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ALIGNMENT ANALYSIS: CLAIMS vs MARKET")
    print("=" * 60)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    
    # Step 1: Load claims matrix
    print("\n" + "=" * 60)
    print("STEP 1: LOADING CLAIMS MATRIX")
    print("=" * 60)
    
    claims_data = np.load(claims_path, allow_pickle=True)
    claims_matrix = claims_data["matrix"]
    symbols = list(claims_data["symbols"])
    categories = list(claims_data["categories"])
    
    print(f"\nClaims matrix shape: {claims_matrix.shape}")
    print(f"Symbols: {symbols}")
    print(f"Categories: {categories}")
    
    # Step 2: Collect market data
    market_data = collect_market_data(symbols, lookback_days=lookback_days)
    
    # Check which symbols we have data for
    available_symbols = [s for s in symbols if s in market_data]
    
    if len(available_symbols) < len(symbols):
        missing = set(symbols) - set(available_symbols)
        print(f"\n⚠ Missing market data for: {missing}")
    
    # Step 3: Compute market features
    market_matrix, market_symbols = compute_market_features(market_data)
    
    # Step 4: Align matrices (same symbols, same order)
    print("\n" + "=" * 60)
    print("STEP 4: ALIGNING MATRICES")
    print("=" * 60)
    
    # Find common symbols
    common_symbols = [s for s in symbols if s in market_symbols]
    
    # Reorder matrices to match
    claims_indices = [symbols.index(s) for s in common_symbols]
    market_indices = [market_symbols.index(s) for s in common_symbols]
    
    aligned_claims = claims_matrix[claims_indices]
    aligned_market = market_matrix[market_indices]
    
    print(f"\nCommon symbols: {common_symbols}")
    print(f"Aligned claims shape: {aligned_claims.shape}")
    print(f"Aligned market shape: {aligned_market.shape}")
    
    # Save market matrix
    market_path = output_path / "market_matrix.npz"
    np.savez(
        market_path,
        matrix=aligned_market,
        symbols=np.array(common_symbols),
        features=np.array([
            "mean_return", "volatility", "sharpe", 
            "max_drawdown", "avg_volume", "vol_volatility", "trend"
        ])
    )
    print(f"\n✓ Saved market matrix: {market_path}")
    
    # Step 5: Run alignment test
    print("\n" + "=" * 60)
    print("STEP 5: ALIGNMENT TESTING")
    print("=" * 60)
    
    test = AlignmentTest(n_factors=4, apply_procrustes=True)
    result = test.test_matrix_alignment(
        aligned_claims,
        aligned_market,
        entity_labels=common_symbols
    )
    
    print(result.summary())
    
    # Step 6: Bootstrap confidence interval
    print("\n" + "=" * 60)
    print("STEP 6: BOOTSTRAP CONFIDENCE INTERVAL")
    print("=" * 60)
    
    print("\nRunning 50 bootstrap samples...")
    point, lower, upper = test.bootstrap_confidence_interval(
        aligned_claims,
        aligned_market,
        n_bootstrap=50,
        entity_labels=common_symbols
    )
    
    print(f"\nPoint estimate: φ = {point:.4f}")
    print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
    
    # Step 7: Save results
    print("\n" + "=" * 60)
    print("STEP 7: SAVING RESULTS")
    print("=" * 60)
    
    results_dict = result.to_dict()
    results_dict["bootstrap"] = {
        "point_estimate": float(point),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": 0.95,
        "n_bootstrap": 50,
    }
    results_dict["timestamp"] = datetime.now().isoformat()
    
    results_path = output_path / "alignment_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"✓ Saved results: {results_path}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("ALIGNMENT ANALYSIS COMPLETE!")
    print("=" * 60)
    
    print(f"""
Key Finding:
    Tucker's Congruence Coefficient: φ = {point:.4f}
    95% Confidence Interval: [{lower:.4f}, {upper:.4f}]
    Interpretation: {result.interpretation}

This measures the alignment between functional claims in crypto
whitepapers and actual market behavior patterns.

{
    'φ ≥ 0.95: Strong evidence that claims predict market positioning' if point >= 0.95
    else 'φ = 0.85-0.94: Moderate evidence of claims-market alignment' if point >= 0.85
    else 'φ = 0.65-0.84: Weak evidence of claims-market alignment' if point >= 0.65
    else 'φ < 0.65: No significant alignment between claims and market'
}
""")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run alignment analysis")
    parser.add_argument("--lookback", type=int, default=365,
                        help="Days of market data (default: 365)")
    parser.add_argument("--claims", type=str, default="outputs/nlp/claims_matrix.npz",
                        help="Path to claims matrix")
    args = parser.parse_args()
    
    result = run_alignment_analysis(
        claims_path=args.claims,
        lookback_days=args.lookback
    )
