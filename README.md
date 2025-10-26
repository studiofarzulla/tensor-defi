# Tensor Decomposition for DeFi Market Microstructure

**Academic Research Framework** | Phase 1: Proof of Concept

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()

## Overview

This repository implements a tensor decomposition framework for analyzing cryptocurrency and DeFi market microstructure. The core thesis: **markets don't live on flat planes—they evolve on curved, low-dimensional manifolds induced by microstructure. Tensor methods respect that curvature; traditional linear models erase it.**

### Key Innovation

Traditional models (PCA, VAR, GARCH) flatten multi-dimensional market data into matrices, losing critical interaction effects. This framework preserves the full tensor structure to capture:

- **Multi-dimensional assets**: Crypto assets simultaneously function as currency, collateral, gas, and governance tokens
- **Cross-venue dynamics**: Assets behave differently on centralized exchanges vs decentralized AMMs
- **Temporal regimes**: Market conditions create discrete behavioral modes (bull/bear, high/low volatility)
- **Functional superposition**: DeFi assets exist in multiple functional states across venues and protocols

## Current Status

**Phase 1 (Proof of Concept):** 60% Complete

- ✅ Data pipeline (CEX via CCXT)
- ✅ Tensor construction (Time × Venue × Asset × Feature)
- ✅ CP/Tucker/TT decomposition implementations
- ✅ PCA baseline comparison framework
- ✅ Visualization suite
- ✅ **Full year dataset collected** (8,761 hours, 3 assets)
- 🚧 Empirical results generation (in progress)
- 📝 Academic paper (Introduction + Methodology complete)

## Quick Start

### Installation (Arch Linux / venv)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/tensor-defi.git
cd tensor-defi

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Collect Market Data

```bash
# Test with 7 days
python collect_data.py

# Full year (takes ~30-60 minutes)
# Edit collect_data.py: lookback_days = 365
python collect_data.py
```

### Build Tensor

```bash
# Creates 3 tensor variants (raw, normalized, log-returns)
python scripts/build_tensor.py
```

### Run Decomposition

```bash
# CP and Tucker decomposition with rank selection
python src/tensor_ops/decomposition.py
```

## Project Structure

```
tensor-defi/
├── src/
│   ├── data_pipeline/          # Multi-venue data collection
│   │   ├── cex_collector.py    # Binance, Coinbase (CCXT)
│   │   └── dex_collector.py    # Uniswap, Curve (The Graph)
│   ├── tensor_ops/             # Tensor operations
│   │   ├── tensor_builder.py   # Construct 4D tensors
│   │   └── decomposition.py    # CP, Tucker, TT methods
│   ├── baselines/              # Traditional methods
│   │   └── traditional_methods.py  # PCA, VAR comparison
│   └── visualization/          # Plotting tools
│       └── tensor_plots.py     # Factor evolution, 3D projections
├── scripts/                    # Analysis scripts
│   ├── analyze_cex_data.py     # Data quality assessment
│   └── build_tensor.py         # Tensor construction
├── data/                       # Market data (not in repo)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Mathematical Framework

### Tensor Representation

Market microstructure is represented as a 4-mode tensor:

```
𝓧 ∈ ℝ^(T × V × A × F)

where:
  T = Time dimension (hourly timestamps)
  V = Venue dimension (exchanges/AMMs)
  A = Asset dimension (BTC, ETH, SOL, etc.)
  F = Feature dimension (price, volume, spread, liquidity)
```

### Decomposition Methods

1. **CP (CANDECOMP/PARAFAC)**: Most compact representation
   - Decomposes tensor as sum of rank-1 components
   - Economically interpretable factors
   - Best for discovering fundamental market drivers

2. **Tucker**: Most flexible representation
   - Core tensor captures factor interactions
   - Allows different ranks per dimension
   - Best for modeling complex cross-effects

3. **Tensor Train (TT)**: Memory-efficient sequential decomposition
   - Scales linearly with number of dimensions
   - Best for high-dimensional extensions (MEV, gas, block position)

### Why Tensor > Matrix?

**Matrix (PCA) Assumption:**
```
ℓ ≈ α_venue + β_asset + γ_time
```
(Additive model - misses interactions)

**Tensor (CP/Tucker) Reality:**
```
ℓ = g_venue(·) × h_asset(·) × φ_time(·)
```
(Multiplicative model - captures venue×asset×time interactions)

**Example**: ETH liquidity on Uniswap during high gas periods involves a 3-way interaction between:
- AMM bonding curve mechanics (venue)
- Gas payment demand (asset function)
- Network congestion (time-specific event)

Traditional models miss this by assuming additivity.

## Dataset

**Current Collection:**
- **Duration**: 1 year (Oct 26, 2024 - Oct 26, 2025)
- **Granularity**: Hourly OHLCV data
- **Assets**: BTC/USDT, ETH/USDT, SOL/USDT
- **Venue**: Binance (CEX)
- **Total observations**: 26,283 rows (8,761 hours × 3 assets)
- **Data quality**: 100% completeness, zero gaps

**Market Coverage:**
- BTC: $66,712 - $126,011 (+88.9% range)
- ETH: $1,419 - $4,935 (+247.8% range)
- SOL: $97 - $286 (+194.8% range)

## Research Questions

1. **Reconstruction**: Can tensor decomposition outperform PCA for market data reconstruction?
   - Hypothesis: 20-50% error reduction

2. **Regime Detection**: Do temporal factors reveal discrete market regimes?
   - Hypothesis: Sharp transitions during known events (crashes, regulatory changes)

3. **Cross-Asset Structure**: Can asset factors identify correlation patterns?
   - Hypothesis: Factors map to fundamental market drivers (risk-on/risk-off, DeFi vs BTC)

4. **Economic Interpretation**: Are extracted factors economically meaningful?
   - Hypothesis: Factors correspond to bull/bear regimes, liquidity conditions, volatility states

## Results (Preliminary)

**7-Day Test Dataset:**
- Tensor shape: (169, 1, 3, 5) = 2,535 elements
- Cross-asset correlations:
  - BTC-ETH: 0.637 (moderate)
  - BTC-SOL: 0.862 (strong)
  - ETH-SOL: 0.724 (strong)
- After log-returns transformation:
  - BTC-ETH: 0.878 ↑
  - Indicates strong shared latent structure

**Full Year Dataset:**
- Tensor shape: (8761, 1, 3, 5) = 131,415 elements
- Memory: 1 MB (tensor), 1.18 MB (pickle file)
- Correlations reveal distinct market structure:
  - BTC-ETH: 0.688 (crypto majors move together)
  - ETH-SOL: 0.762 (altcoins highly correlated)
  - BTC-SOL: 0.393 (BTC dominance effect)

## Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for detailed technical documentation including:
- Data collection pipeline
- Tensor construction algorithms
- Decomposition methods (CP-ALS, Tucker-HOOI)
- Baseline comparison framework
- Evaluation metrics

## Roadmap

### Phase 1: Proof of Concept (Current - 60% complete)
- [x] Data pipeline
- [x] Tensor operations
- [x] Baseline comparisons
- [x] Full year data collection
- [ ] Empirical results generation
- [ ] Academic paper completion

### Phase 2: Multi-Venue Extension (Future)
- [ ] Add Coinbase, Kraken (CEX)
- [ ] Fix DEX collector (The Graph API migration)
- [ ] Cross-venue arbitrage detection
- [ ] MEV-aware microstructure

### Phase 3: Manifold Geometry (Future)
- [ ] Riemannian metrics on factor space
- [ ] Ricci curvature for regime stress
- [ ] Geodesic distances for asset similarity
- [ ] Real-time deployment on homelab K8s

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{farzulla2025tensordefi,
  author = {Farzulla, Murad},
  title = {Tensor Decomposition for DeFi Market Microstructure},
  year = {2025},
  url = {https://github.com/studiofarzulla/tensor-defi},
  note = {Research implementation, Phase 1}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

This is an active research project. Contributions welcome:
- Bug reports and feature requests via Issues
- Code contributions via Pull Requests
- Academic collaboration inquiries: murad@farzulla.org

## Acknowledgments

- **Data Sources**: Binance (CCXT library), The Graph (DeFi protocols)
- **Libraries**: TensorLy, PyTorch, NumPy, Pandas
- **Infrastructure**: Developed on Arch Linux with Claude Code

---

**Status**: Research Framework | **Phase**: 1 (Proof of Concept) | **Updated**: October 2025

