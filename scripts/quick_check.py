#!/usr/bin/env python3
"""Quick data quality check - minimal dependencies"""

import pandas as pd
import numpy as np
from pathlib import Path

data_path = Path(__file__).parent.parent / 'data' / 'cex_data_7d.csv'
df = pd.read_csv(data_path)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)

print("CEX DATA QUICK CHECK")
print("=" * 60)

# Basic stats
print(f"\nTotal rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Duration: {df['datetime'].max() - df['datetime'].min()}")

# Per symbol
symbols = df['symbol'].unique()
print(f"\nSymbols: {list(symbols)}")

for symbol in symbols:
    sdf = df[df['symbol'] == symbol]
    print(f"\n{symbol}:")
    print(f"  Records: {len(sdf)}")
    print(f"  Price range: ${sdf['close'].min():.2f} - ${sdf['close'].max():.2f}")
    print(f"  Price change: {((sdf['close'].iloc[-1] / sdf['close'].iloc[0]) - 1) * 100:+.2f}%")
    print(f"  Avg volume: {sdf['volume'].mean():,.2f}")

# Data quality
print("\n" + "=" * 60)
print("DATA QUALITY:")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Duplicates: {df.duplicated(subset=['datetime', 'symbol', 'exchange']).sum()}")

# Check alignment
timestamps_per_symbol = {s: set(df[df['symbol'] == s]['datetime']) for s in symbols}
common_timestamps = set.intersection(*timestamps_per_symbol.values())
print(f"  Common timestamps: {len(common_timestamps)}/{len(df[df['symbol'] == symbols[0]])}")

# OHLC validation
ohlc_issues = 0
for symbol in symbols:
    sdf = df[df['symbol'] == symbol]
    ohlc_issues += len(sdf[sdf['high'] < sdf['low']])
    ohlc_issues += len(sdf[(sdf['open'] > sdf['high']) | (sdf['open'] < sdf['low'])])
    ohlc_issues += len(sdf[(sdf['close'] > sdf['high']) | (sdf['close'] < sdf['low'])])

print(f"  OHLC inconsistencies: {ohlc_issues}")

# Gap check
gaps_total = 0
for symbol in symbols:
    sdf = df[df['symbol'] == symbol].sort_values('datetime')
    time_diffs = sdf['datetime'].diff()
    expected_diff = pd.Timedelta(hours=1)
    gaps = time_diffs[time_diffs != expected_diff].dropna()
    gaps_total += len(gaps)

print(f"  Temporal gaps: {gaps_total}")

# Overall assessment
checks_passed = sum([
    df.isnull().sum().sum() == 0,
    df.duplicated(subset=['datetime', 'symbol', 'exchange']).sum() == 0,
    len(common_timestamps) == len(df[df['symbol'] == symbols[0]]),
    ohlc_issues == 0,
    gaps_total == 0
])

quality_score = (checks_passed / 5) * 100

print("\n" + "=" * 60)
print(f"QUALITY SCORE: {quality_score:.0f}% ({checks_passed}/5 checks passed)")

if quality_score == 100:
    print("✓ READY FOR TENSOR CONSTRUCTION")
elif quality_score >= 80:
    print("✓ Good quality - minor preprocessing may be needed")
else:
    print("✗ Requires preprocessing")

print("\nRun 'python scripts/analyze_cex_data.py' for detailed analysis")
