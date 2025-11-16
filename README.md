# Information Geometry of Markets

**Cryptocurrency Exchanges as Multiplicative Tensor Processes — Evidence from High-Frequency Decomposition**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper: CC-BY-4.0](https://img.shields.io/badge/Paper-CC--BY--4.0-green.svg)](arxiv-submission/Farzulla_2025_Tensor_Decomposition.pdf)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)]()

**Version 1.0.0** | November 2025 | [Murad Farzulla](https://farzulla.org)

## Overview

This repository implements the **first tensor decomposition framework for multi-asset cryptocurrency market microstructure**. The core thesis: **markets don't live on flat planes—they evolve on curved, low-dimensional manifolds induced by microstructure dynamics. Tensor methods respect this geometry; traditional matrix methods erase it.**

### Research Paper

📄 **[Information Geometry of Markets: Cryptocurrency Exchanges as Multiplicative Tensor Processes](arxiv-submission/Farzulla_2025_Tensor_Decomposition.pdf)**

**Abstract:** We propose tensor decomposition as a proof-of-concept framework for modeling cryptocurrency market microstructure, preserving multi-dimensional interaction effects lost by traditional matrix methods. Using one year of hourly OHLCV data (8,761 hours) for BTC, ETH, and SOL from Binance, rank-4 CP and Tucker decompositions achieve 96.55% and 96.56% explained variance respectively, outperforming PCA (92.31%) by 4.6 percentage points.

**Key Findings:**
- Four economically interpretable factors: (1) Bitcoin dominance, (2) altcoin rotation, (3) volatility regimes, (4) intraday microstructure
- Tensor methods preserve multiplicative interactions between asset and temporal dimensions
- Negligible CP-Tucker difference (0.01%) suggests true rank-4 structure

**Version 1.0.0:** Proof-of-concept on single-venue data. Future versions will expand to multi-venue analysis.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/resurrexi/tensor-defi.git
cd tensor-defi

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Reproduce Paper Results

```bash
# 1. Collect market data (takes ~30-60 minutes)
python collect_data.py

# 2. Build tensor (creates 3 normalization variants)
python scripts/build_tensor.py

# 3. Run decompositions and baselines
python scripts/run_full_experiments.py

# Results saved to outputs/
```

### Dataset

**Included in repo:** Processed tensors (outputs/tensors/)
**Reproducible:** Raw OHLCV data can be re-collected via `collect_data.py`

**Specifications:**
- Duration: 1 year (Oct 26, 2024 - Oct 26, 2025)
- Granularity: Hourly OHLCV
- Assets: BTC/USDT, ETH/USDT, SOL/USDT
- Venue: Binance (v1.0.0), multi-venue in v2.0
- Total observations: 26,283 rows (8,761 hours × 3 assets)
- Data quality: 100% completeness, zero gaps

## Project Structure

```
tensor-defi/
├── arxiv-submission/              # LaTeX paper + figures
│   ├── Farzulla_2025_Tensor_Decomposition.tex
│   ├── Farzulla_2025_Tensor_Decomposition.pdf
│   ├── references.bib
│   └── figures/
├── src/
│   ├── data_pipeline/             # Data collection
│   │   └── cex_collector.py       # CCXT-based CEX collector
│   ├── tensor_ops/                # Tensor operations
│   │   ├── tensor_builder.py      # Construct 4D tensors
│   │   └── decomposition.py       # CP, Tucker implementations
│   ├── baselines/                 # Traditional methods
│   │   └── traditional_methods.py # PCA comparison
│   └── visualization/             # Plotting tools
│       └── tensor_plots.py
├── scripts/                       # Analysis scripts
│   ├── build_tensor.py            # Tensor construction
│   └── run_full_experiments.py    # Full experimental pipeline
├── outputs/                       # Results (generated)
│   ├── tensors/                   # Processed tensor files
│   ├── decomposition_results/     # CP/Tucker outputs
│   └── figures/                   # Generated plots
├── data/                          # Raw market data (not in repo, regenerate via collect_data.py)
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Mathematical Framework

### Tensor Representation

Market microstructure as a 3-way tensor (v1.0.0):

```
𝓧 ∈ ℝ^(T × A × F)

where:
  T = Time dimension (8,761 hourly timestamps)
  A = Asset dimension (3 cryptocurrencies)
  F = Feature dimension (5: open, high, low, close, volume)
```

**Note:** v1.0.0 uses single venue (Binance). v2.0 will add venue dimension for full 4-way tensor.

### Decomposition Methods

**CP (CANDECOMP/PARAFAC):**
```
𝓧 ≈ Σ_{r=1}^R λ_r · (a_r ⊗ b_r ⊗ c_r)
```
- Most compact representation
- Economically interpretable factors
- Rank-4 optimal for our dataset

**Tucker:**
```
𝓧 ≈ 𝓖 ×₁ A ×₂ B ×₃ C
```
- Core tensor captures interactions
- Flexible rank per dimension
- 96.56% explained variance (rank 4×1×4×4)

### Why Tensor > Matrix?

**Matrix (PCA):** Flattens tensor, assumes additive effects
**Tensor (CP/Tucker):** Preserves structure, models multiplicative interactions

**Example:** ETH liquidity = venue mechanics × asset function × time regime (3-way interaction lost by PCA)

## Results Summary

**Reconstruction Performance:**
- CP (rank-4): 96.55% explained variance
- Tucker (4×1×4×4): 96.56% explained variance
- PCA baseline: 92.31% explained variance
- **Improvement:** 4.6 percentage points

**Extracted Factors:**
1. Bitcoin dominance and systematic risk
2. Altcoin rotation and risk appetite
3. Volatility regimes and liquidity shocks
4. Intraday microstructure and mean reversion

See [paper](arxiv-submission/Farzulla_2025_Tensor_Decomposition.pdf) for full analysis.

## Versioning Roadmap

**v1.0.0 (Current):** Proof-of-concept on single-venue, 3-asset data
**v2.0 (Planned):** Multi-venue expansion (Coinbase, Kraken, Uniswap, Curve)
**v2.1 (Planned):** Robustness validation (bootstrap CIs, out-of-sample tests, 10+ assets)
**v3.0 (Planned):** Higher-dimensional features (MEV, gas prices, order book depth)
**v3.1 (Planned):** Information geometry (Riemannian metrics, Ricci curvature)

## Citation

If you use this work, please cite:

**Paper:**
```bibtex
@techreport{Farzulla2025InformationGeometry,
  author = {Farzulla, Murad},
  title = {Information Geometry of Markets: Cryptocurrency Exchanges as Multiplicative Tensor Processes},
  institution = {Farzulla Research},
  year = {2025},
  type = {Working Paper},
  note = {Version 1.0.0},
  url = {https://github.com/resurrexi/tensor-defi}
}
```

**Software:**
```bibtex
@software{Farzulla2025TensorDefi,
  author = {Farzulla, Murad},
  title = {tensor-defi: Tensor Decomposition for Cryptocurrency Market Microstructure},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/resurrexi/tensor-defi},
  doi = {pending}
}
```

## License

- **Code:** MIT License (see [LICENSE](LICENSE))
- **Paper:** CC-BY-4.0 (see [arxiv-submission/](arxiv-submission/))
- **Data:** Public market data from Binance (reproducible via CCXT)

## Reproducibility

**Software versions:**
- Python: 3.13+
- TensorLy: 0.8.1
- NumPy: 1.26.4
- CCXT: 4.4+

**Hardware:**
- Development: AMD Ryzen 9900X, 128GB RAM, Arch Linux
- Minimum: 8GB RAM, any modern CPU (experiments run in 30-90 seconds)

**Random seed:** 42 (set in all decomposition scripts)

**Convergence criteria:** ALS tolerance 10^-4, max 500 iterations

## Contributing

This is an active research project. Contributions welcome:
- **Bug reports:** Open an issue
- **Feature requests:** Open an issue with "enhancement" label
- **Code contributions:** Submit a pull request
- **Academic collaboration:** contact@farzulla.org

## Acknowledgments

**Data sources:** Binance (via CCXT library)
**Libraries:** TensorLy, NumPy, Pandas, Matplotlib
**AI assistance:** Anthropic Claude Sonnet 4.5 (literature review, code implementation, LaTeX preparation)
**Infrastructure:** Developed on Arch Linux with Claude Code

**Related work:** This complements [Farzulla (2025) Cryptocurrency Event Study](https://doi.org/10.5281/zenodo.17595207) analyzing cross-sectional heterogeneity in volatility responses.

---

**Author:** [Murad Farzulla](https://farzulla.org) | [Farzulla Research](https://farzulla.org)
**Contact:** contact@farzulla.org
**ORCID:** [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)

**Status:** Research Framework | **Version:** 1.0.0 | **Updated:** November 2025
