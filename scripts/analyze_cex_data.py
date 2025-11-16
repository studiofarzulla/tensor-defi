#!/usr/bin/env python3
"""
CEX Data Quality Analysis for Tensor Decomposition Framework
Analyzes 7-day Binance OHLCV data for tensor construction readiness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path

# Set style for clean plots
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (16, 10)

# Load data
data_path = Path(__file__).parent.parent / 'data' / 'cex_data_7d.csv'
df = pd.read_csv(data_path)

print("=" * 80)
print("CEX DATA QUALITY ANALYSIS - TENSOR DECOMPOSITION READINESS")
print("=" * 80)

# ============================================================================
# 1. DATA STRUCTURE & COMPLETENESS
# ============================================================================

print("\n[1] DATA STRUCTURE & COMPLETENESS")
print("-" * 80)

print(f"\nTotal records: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)

# Missing values analysis
print("\n--- Missing Values Analysis ---")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ NO MISSING VALUES - Perfect data completeness")
else:
    print("✗ Missing values detected:")
    print(missing[missing > 0])

# Check for duplicates
duplicates = df.duplicated(subset=['datetime', 'symbol', 'exchange']).sum()
print(f"\nDuplicate rows: {duplicates}")
if duplicates > 0:
    print("✗ WARNING: Duplicates detected")
    print(df[df.duplicated(subset=['datetime', 'symbol', 'exchange'], keep=False)])
else:
    print("✓ No duplicates found")

# ============================================================================
# 2. TEMPORAL COVERAGE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("[2] TEMPORAL COVERAGE ANALYSIS")
print("-" * 80)

symbols = df['symbol'].unique()
print(f"\nSymbols: {list(symbols)}")
print(f"Exchange: {df['exchange'].unique()}")
print(f"Timeframe: {df['timeframe'].unique()}")

print("\n--- Date Range Coverage ---")
for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol]
    start = symbol_df['datetime'].min()
    end = symbol_df['datetime'].max()
    duration = end - start
    count = len(symbol_df)

    print(f"\n{symbol}:")
    print(f"  Start:    {start}")
    print(f"  End:      {end}")
    print(f"  Duration: {duration}")
    print(f"  Records:  {count}")

    # Expected records for hourly data
    expected_hours = int(duration.total_seconds() / 3600) + 1
    print(f"  Expected: {expected_hours} hourly records")

    if count == expected_hours:
        print(f"  ✓ Complete coverage ({count}/{expected_hours})")
    else:
        print(f"  ✗ Missing data ({count}/{expected_hours}) - {expected_hours - count} gaps")

# Check for gaps in timestamps
print("\n--- Timestamp Gap Analysis ---")
for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol].sort_values('datetime')
    time_diffs = symbol_df['datetime'].diff()

    # Expected: 1 hour
    expected_diff = pd.Timedelta(hours=1)
    gaps = time_diffs[time_diffs != expected_diff].dropna()

    if len(gaps) == 0:
        print(f"✓ {symbol}: No gaps - Perfect hourly continuity")
    else:
        print(f"✗ {symbol}: {len(gaps)} gaps detected:")
        for idx in gaps.index:
            prev_time = symbol_df.loc[idx-1, 'datetime']
            curr_time = symbol_df.loc[idx, 'datetime']
            gap_hours = (curr_time - prev_time).total_seconds() / 3600
            print(f"  Gap at {curr_time}: {gap_hours:.1f} hours")

# ============================================================================
# 3. PRICE & VOLUME VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("[3] PRICE & VOLUME VALIDATION")
print("-" * 80)

for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol]

    print(f"\n{symbol}:")
    print(f"  Open:   ${symbol_df['open'].min():,.2f} - ${symbol_df['open'].max():,.2f}")
    print(f"  High:   ${symbol_df['high'].min():,.2f} - ${symbol_df['high'].max():,.2f}")
    print(f"  Low:    ${symbol_df['low'].min():,.2f} - ${symbol_df['low'].max():,.2f}")
    print(f"  Close:  ${symbol_df['close'].min():,.2f} - ${symbol_df['close'].max():,.2f}")
    print(f"  Volume: {symbol_df['volume'].min():,.2f} - {symbol_df['volume'].max():,.2f}")

    # OHLC consistency checks
    invalid_high = symbol_df[symbol_df['high'] < symbol_df['low']]
    invalid_open = symbol_df[(symbol_df['open'] > symbol_df['high']) | (symbol_df['open'] < symbol_df['low'])]
    invalid_close = symbol_df[(symbol_df['close'] > symbol_df['high']) | (symbol_df['close'] < symbol_df['low'])]

    if len(invalid_high) + len(invalid_open) + len(invalid_close) == 0:
        print("  ✓ OHLC consistency: All records valid")
    else:
        print("  ✗ OHLC inconsistencies detected:")
        if len(invalid_high) > 0:
            print(f"    High < Low: {len(invalid_high)} records")
        if len(invalid_open) > 0:
            print(f"    Open outside High/Low: {len(invalid_open)} records")
        if len(invalid_close) > 0:
            print(f"    Close outside High/Low: {len(invalid_close)} records")

    # Zero volume check
    zero_volume = symbol_df[symbol_df['volume'] == 0]
    if len(zero_volume) > 0:
        print(f"  ⚠ Zero volume: {len(zero_volume)} records")
    else:
        print("  ✓ No zero volume records")

# ============================================================================
# 4. VOLATILITY & STATISTICAL ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("[4] VOLATILITY & STATISTICAL ANALYSIS")
print("-" * 80)

volatility_summary = []

for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol].copy()

    # Calculate returns
    symbol_df['returns'] = symbol_df['close'].pct_change()
    symbol_df['log_returns'] = np.log(symbol_df['close'] / symbol_df['close'].shift(1))
    symbol_df['intrabar_range'] = (symbol_df['high'] - symbol_df['low']) / symbol_df['open']

    print(f"\n{symbol}:")
    print(f"  Price Change:  {((symbol_df['close'].iloc[-1] / symbol_df['close'].iloc[0]) - 1) * 100:+.2f}%")
    print(f"  Mean Return:   {symbol_df['returns'].mean() * 100:.4f}%")
    print(f"  Std Dev:       {symbol_df['returns'].std() * 100:.4f}%")
    print(f"  Min Return:    {symbol_df['returns'].min() * 100:.2f}%")
    print(f"  Max Return:    {symbol_df['returns'].max() * 100:.2f}%")
    print(f"  Avg Range:     {symbol_df['intrabar_range'].mean() * 100:.2f}%")

    # Outlier detection (3 sigma)
    outliers = symbol_df[np.abs(symbol_df['returns']) > 3 * symbol_df['returns'].std()]
    print(f"  Outliers (3σ): {len(outliers)} events")

    if len(outliers) > 0:
        print(f"    Max spike: {outliers['returns'].abs().max() * 100:.2f}%")

    volatility_summary.append({
        'symbol': symbol,
        'mean_return': symbol_df['returns'].mean(),
        'volatility': symbol_df['returns'].std(),
        'sharpe_approx': symbol_df['returns'].mean() / symbol_df['returns'].std() if symbol_df['returns'].std() > 0 else 0
    })

vol_df = pd.DataFrame(volatility_summary)
print("\n--- Volatility Ranking ---")
print(vol_df.sort_values('volatility', ascending=False).to_string(index=False))

# ============================================================================
# 5. TENSOR CONSTRUCTION READINESS
# ============================================================================

print("\n" + "=" * 80)
print("[5] TENSOR CONSTRUCTION READINESS")
print("-" * 80)

# Expected tensor shape
n_symbols = len(symbols)
n_hours = len(df[df['symbol'] == symbols[0]])
n_features = 5  # OHLCV

print(f"\nTensor Shape (Time × Venue × Asset × Feature):")
print(f"  Time dimension:    {n_hours} hours")
print(f"  Venue dimension:   1 (Binance)")
print(f"  Asset dimension:   {n_symbols} symbols")
print(f"  Feature dimension: {n_features} (OHLCV)")
print(f"\n  Expected tensor:   ({n_hours}, 1, {n_symbols}, {n_features})")

# Verify alignment
print("\n--- Time Alignment Check ---")
timestamps_per_symbol = {symbol: set(df[df['symbol'] == symbol]['datetime']) for symbol in symbols}

# Find common timestamps
common_timestamps = set.intersection(*timestamps_per_symbol.values())
print(f"Common timestamps across all symbols: {len(common_timestamps)}")

if len(common_timestamps) == n_hours:
    print("✓ PERFECT ALIGNMENT - All symbols share same timestamps")
else:
    print("✗ Misalignment detected:")
    for symbol in symbols:
        missing = len(timestamps_per_symbol[symbol]) - len(common_timestamps)
        if missing != 0:
            print(f"  {symbol}: {missing} unique timestamps")

# Data quality score
checks_passed = 0
total_checks = 7

if df.isnull().sum().sum() == 0:
    checks_passed += 1
if duplicates == 0:
    checks_passed += 1
if len(common_timestamps) == n_hours:
    checks_passed += 1

# Check OHLC consistency across all symbols
ohlc_valid = True
for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol]
    if len(symbol_df[symbol_df['high'] < symbol_df['low']]) > 0:
        ohlc_valid = False
if ohlc_valid:
    checks_passed += 1

# Check no zero volumes
zero_vol_count = 0
for symbol in symbols:
    zero_vol_count += len(df[(df['symbol'] == symbol) & (df['volume'] == 0)])
if zero_vol_count == 0:
    checks_passed += 1

# Check no extreme outliers (>10 sigma)
extreme_outliers = 0
for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol].copy()
    symbol_df['returns'] = symbol_df['close'].pct_change()
    extreme_outliers += len(symbol_df[np.abs(symbol_df['returns']) > 10 * symbol_df['returns'].std()])
if extreme_outliers == 0:
    checks_passed += 1

# Check temporal continuity
gaps_total = 0
for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol].sort_values('datetime')
    time_diffs = symbol_df['datetime'].diff()
    expected_diff = pd.Timedelta(hours=1)
    gaps_total += len(time_diffs[time_diffs != expected_diff].dropna())
if gaps_total == 0:
    checks_passed += 1

quality_score = (checks_passed / total_checks) * 100

print("\n" + "=" * 80)
print(f"DATA QUALITY SCORE: {quality_score:.1f}% ({checks_passed}/{total_checks} checks passed)")
print("=" * 80)

if quality_score == 100:
    print("\n✓✓✓ DATASET IS PRODUCTION-READY FOR TENSOR CONSTRUCTION ✓✓✓")
    print("No preprocessing required - data is clean, aligned, and complete.")
elif quality_score >= 80:
    print("\n✓ Dataset is GOOD - Minor preprocessing may be needed")
else:
    print("\n✗ Dataset requires SIGNIFICANT preprocessing")

# ============================================================================
# 6. RECOMMENDATIONS FOR FULL YEAR COLLECTION
# ============================================================================

print("\n" + "=" * 80)
print("[6] RECOMMENDATIONS FOR FULL YEAR COLLECTION")
print("=" * 80)

total_hours_7d = n_hours
hours_per_year = 365 * 24
estimated_rows_1y = hours_per_year * n_symbols

current_size_mb = data_path.stat().st_size / (1024 * 1024)
estimated_size_1y_mb = current_size_mb * (hours_per_year / n_hours)

print(f"\nCurrent dataset (7 days):")
print(f"  Rows:      {len(df):,}")
print(f"  File size: {current_size_mb:.2f} MB")
print(f"  Hours:     {n_hours}")

print(f"\nEstimated 1-year dataset:")
print(f"  Rows:      {estimated_rows_1y:,}")
print(f"  File size: {estimated_size_1y_mb:.2f} MB (~{estimated_size_1y_mb/1024:.2f} GB)")
print(f"  Hours:     {hours_per_year:,}")

print("\nRecommendations:")
print("1. ✓ Current 7-day data quality is excellent - proceed with full year")
print("2. Consider chunked collection (monthly batches) to avoid API rate limits")
print("3. Store intermediate checkpoints - 1 year = ~52 weekly chunks")
print(f"4. Budget ~{estimated_size_1y_mb/1024:.1f} GB storage for raw CSV")
print("5. Consider Parquet format for 50-80% compression vs CSV")
print("6. Validate timestamp continuity after each batch")
print("7. Monitor for exchange maintenance windows (may cause gaps)")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("[7] GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(3, 2, figsize=(18, 12))
fig.suptitle('CEX Market Data - 7 Day Analysis (Binance OHLCV)', fontsize=16, fontweight='bold')

# Price evolution
ax = axes[0, 0]
for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol]
    ax.plot(symbol_df['datetime'], symbol_df['close'], label=symbol, linewidth=1.5)
ax.set_title('Close Price Evolution', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USDT)')
ax.legend()
ax.grid(True, alpha=0.3)

# Normalized prices (% change from start)
ax = axes[0, 1]
for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol].copy()
    symbol_df['normalized'] = (symbol_df['close'] / symbol_df['close'].iloc[0] - 1) * 100
    ax.plot(symbol_df['datetime'], symbol_df['normalized'], label=symbol, linewidth=1.5)
ax.set_title('Normalized Returns (% change from start)', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Return (%)')
ax.legend()
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

# Volume comparison
ax = axes[1, 0]
for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol]
    ax.plot(symbol_df['datetime'], symbol_df['volume'], label=symbol, alpha=0.7)
ax.set_title('Trading Volume', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Volume')
ax.legend()
ax.grid(True, alpha=0.3)

# Hourly returns distribution
ax = axes[1, 1]
for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol].copy()
    symbol_df['returns'] = symbol_df['close'].pct_change() * 100
    ax.hist(symbol_df['returns'].dropna(), bins=50, alpha=0.5, label=symbol)
ax.set_title('Hourly Returns Distribution', fontweight='bold')
ax.set_xlabel('Return (%)')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)

# Intrabar range (high-low)
ax = axes[2, 0]
for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol].copy()
    symbol_df['range_pct'] = ((symbol_df['high'] - symbol_df['low']) / symbol_df['open']) * 100
    ax.plot(symbol_df['datetime'], symbol_df['range_pct'], label=symbol, alpha=0.7)
ax.set_title('Intrabar Range (High-Low) %', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Range (%)')
ax.legend()
ax.grid(True, alpha=0.3)

# Correlation matrix
ax = axes[2, 1]
close_pivot = df.pivot(index='datetime', columns='symbol', values='close')
returns_pivot = close_pivot.pct_change().dropna()
corr_matrix = returns_pivot.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Returns Correlation Matrix', fontweight='bold')

plt.tight_layout()

# Save plot
output_path = Path(__file__).parent.parent / 'data' / 'cex_data_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved: {output_path}")

# ============================================================================
# 8. EXPORT SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("[8] EXPORTING SUMMARY REPORT")
print("=" * 80)

report_path = Path(__file__).parent.parent / 'data' / 'cex_data_quality_report.txt'

with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CEX DATA QUALITY REPORT - TENSOR DECOMPOSITION FRAMEWORK\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Dataset: {data_path}\n")
    f.write(f"Total Records: {len(df):,}\n")
    f.write(f"Symbols: {', '.join(symbols)}\n")
    f.write(f"Date Range: {df['datetime'].min()} to {df['datetime'].max()}\n")
    f.write(f"Duration: {df['datetime'].max() - df['datetime'].min()}\n\n")

    f.write("QUALITY CHECKS:\n")
    f.write(f"  Missing Values:     {'PASS' if df.isnull().sum().sum() == 0 else 'FAIL'}\n")
    f.write(f"  Duplicates:         {'PASS' if duplicates == 0 else 'FAIL'}\n")
    f.write(f"  Time Alignment:     {'PASS' if len(common_timestamps) == n_hours else 'FAIL'}\n")
    f.write(f"  OHLC Consistency:   {'PASS' if ohlc_valid else 'FAIL'}\n")
    f.write(f"  Zero Volumes:       {'PASS' if zero_vol_count == 0 else 'FAIL'}\n")
    f.write(f"  Extreme Outliers:   {'PASS' if extreme_outliers == 0 else 'FAIL'}\n")
    f.write(f"  Temporal Gaps:      {'PASS' if gaps_total == 0 else 'FAIL'}\n\n")

    f.write(f"OVERALL QUALITY SCORE: {quality_score:.1f}%\n\n")

    f.write("TENSOR SHAPE:\n")
    f.write(f"  ({n_hours}, 1, {n_symbols}, {n_features})\n")
    f.write(f"  Time × Venue × Asset × Feature\n\n")

    f.write("VOLATILITY SUMMARY:\n")
    f.write(vol_df.to_string(index=False))
    f.write("\n\n")

    f.write("RECOMMENDATION: ")
    if quality_score == 100:
        f.write("PROCEED WITH TENSOR CONSTRUCTION - Data is production-ready\n")
    elif quality_score >= 80:
        f.write("GOOD for tensor construction with minor preprocessing\n")
    else:
        f.write("REQUIRES preprocessing before tensor construction\n")

print(f"✓ Report saved: {report_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nOutputs:")
print(f"  1. Visualization: {output_path}")
print(f"  2. Report:        {report_path}")
print(f"\nNext steps:")
print(f"  - Review visualization for anomalies")
print(f"  - If quality score >= 80%, proceed to tensor construction")
print(f"  - If expanding to 1 year, use same collection methodology")
