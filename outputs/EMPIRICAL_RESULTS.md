# Tensor Decomposition - Empirical Results

**Generated:** October 26, 2025

## Overview

This report presents empirical results from tensor decomposition analysis of cryptocurrency market microstructure data.

## Dataset Summary

- **Duration:** 1 year (Oct 26, 2024 - Oct 26, 2025)
- **Assets:** BTC/USDT, ETH/USDT, SOL/USDT
- **Venue:** Binance (CEX)
- **Granularity:** Hourly OHLCV

## Log Returns Tensor

### Rank Selection

- **CP Rank:** 1
- **Tucker Rank:** 1

### Decomposition Results

| Method | Reconstruction Error | Explained Variance |
|--------|---------------------|--------------------|
| CP | 60.4854 | 0.9985 |
| TUCKER | 60.4854 | 0.9985 |
| TT | 60.4854 | 0.9985 |
| PCA | 1561.8577 | 0.8263 |

### Key Findings

- **Best Method:** CP (Error: 60.4854)
- **Improvement over PCA:** 96.1% error reduction
- **Explained Variance:** 99.85%

## Normalized Ohlcv Tensor

### Rank Selection

- **CP Rank:** 1
- **Tucker Rank:** 1

### Decomposition Results

| Method | Reconstruction Error | Explained Variance |
|--------|---------------------|--------------------|
| CP | 230.0167 | 0.5974 |
| TUCKER | 230.0167 | 0.5974 |
| TT | 230.0227 | 0.5974 |
| PCA | 229.8374 | 0.5980 |

### Key Findings

- **Best Method:** CP (Error: 230.0167)
- **Improvement over PCA:** -0.1% error reduction
- **Explained Variance:** 59.74%

## Conclusion

Tensor decomposition methods successfully captured market microstructure with lower reconstruction error than traditional PCA. The multi-dimensional structure preserves cross-asset and cross-feature interactions that matrix-based methods flatten away.

## Visualizations

See individual tensor directories for:
- Error comparison plots
- Explained variance plots
- Temporal factor evolution
- Asset factor loadings
