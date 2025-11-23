# Tensor Structure in Cryptocurrency Microstructure

**Low-Rank CP Decomposition from Binance OHLCV—Single-Venue Proof-of-Concept**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper: CC-BY-4.0](https://img.shields.io/badge/Paper-CC--BY--4.0-green.svg)](Farzulla_2025_Tensor_Structure_v1.0.0.pdf)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17626899-blue.svg)](https://doi.org/10.5281/zenodo.17626899)

**Version 1.0.0** | November 2025 | [Murad Farzulla](https://farzulla.org)

## Overview

This repository demonstrates tensor decomposition methods applied to cryptocurrency market microstructure, preserving multi-dimensional interaction effects lost by traditional matrix methods. This proof-of-concept uses single-venue (Binance) data to establish feasibility for future multi-venue extensions.

### Research Paper

📄 **[Tensor Structure in Cryptocurrency Microstructure: Low-Rank CP Decomposition from Binance OHLCV](Farzulla_2025_Tensor_Structure_v1.0.0.pdf)**

🌐 **[Interactive Dashboard](https://farzulla.org/research/tensor-defi/)** - Explore visualizations and methodology

**Abstract:** This proof-of-concept applies CP (CANDECOMP/PARAFAC) and Tucker decomposition to cryptocurrency market microstructure, preserving multi-dimensional interaction effects lost by traditional matrix methods. Using one year of hourly OHLCV data (8,761 hours) for three major cryptocurrencies (BTC/USDT, ETH/USDT, SOL/USDT) from Binance, rank-4 CP and Tucker decompositions achieve 96.55% and 96.56% explained variance respectively, outperforming Principal Component Analysis (92.31%) by 4.6 percentage points.

**Key Findings:**
- Binance OHLCV data exhibits low-rank tensor structure (rank-4 captures 96.56% variance)
- Four interpretable factors: (1) Bitcoin dominance, (2) altcoin rotation, (3) volatility regimes, (4) intraday microstructure
- Tensor methods preserve multiplicative interactions between asset and temporal dimensions
- Negligible CP-Tucker difference (0.01%) suggests true rank-4 structure

**Version 1.0.0:** Single-venue proof-of-concept on Binance data. Future work will expand to multi-venue analysis (CEX/DEX dynamics).

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

Expected runtime: **30-90 seconds** for decomposition on consumer-grade hardware.

### Dataset

**Included in repo:** Processed tensors (`outputs/tensors/`)
**Reproducible:** Raw OHLCV data can be re-collected via `collect_data.py`

**Specifications:**
- **Duration:** 1 year (Oct 26, 2024 - Oct 26, 2025)
- **Granularity:** Hourly OHLCV
- **Assets:** BTC/USDT, ETH/USDT, SOL/USDT
- **Venue:** Binance (single-venue proof-of-concept)
- **Total observations:** 26,283 rows (8,761 hours × 3 assets)
- **Data quality:** 100% completeness, zero gaps

## Project Structure

```
tensor-defi/
├── Farzulla_2025_Tensor_Structure_v1.0.0.tex    # LaTeX paper source
├── Farzulla_2025_Tensor_Structure_v1.0.0.pdf    # Compiled paper
├── dashboard.html                                # Interactive visualizations
├── references.bib                                # Bibliography
├── ZENODO_METADATA.txt                           # Zenodo upload metadata
├── src/
│   ├── data_pipeline/                # Data collection
│   │   └── cex_collector.py          # CCXT-based CEX collector
│   ├── tensor_ops/                   # Tensor operations
│   │   ├── tensor_builder.py         # Construct tensors
│   │   └── decomposition.py          # CP, Tucker implementations
│   ├── baselines/                    # Traditional methods
│   │   └── traditional_methods.py    # PCA comparison
│   └── visualization/                # Plotting tools
│       └── tensor_plots.py
├── scripts/                          # Analysis scripts
│   ├── build_tensor.py               # Tensor construction
│   └── run_full_experiments.py       # Full experimental pipeline
├── outputs/                          # Results (generated)
│   ├── tensors/                      # Processed tensor files
│   ├── decomposition_results/        # CP/Tucker outputs
│   └── figures/                      # Generated plots
├── data/                             # Raw market data (regenerate via collect_data.py)
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

## Mathematical Framework

### Tensor Representation

Market microstructure as a 3-way tensor:

```
𝓧 ∈ ℝ^(T × A × F)

where:
  T = Time dimension (8,761 hourly timestamps)
  A = Asset dimension (3 cryptocurrencies)
  F = Feature dimension (5: open, high, low, close, volume)
```

### Decomposition Methods

**CP (CANDECOMP/PARAFAC):**
```
𝓧 ≈ Σ_{r=1}^R λ_r · (a_r ⊗ b_r ⊗ c_r)
```
- Most compact representation
- Economically interpretable factors
- Rank-4 optimal for Binance dataset

**Tucker:**
```
𝓧 ≈ 𝓖 ×₁ A ×₂ B ×₃ C
```
- Core tensor captures interactions
- Flexible rank per dimension
- 96.56% explained variance (rank 4×1×4×4)

### Why Tensor Methods?

**Matrix (PCA):** Flattens tensor → loses multiplicative structure → 92.31% variance
**Tensor (CP/Tucker):** Preserves structure → captures interactions → 96.56% variance

**Improvement:** 4.6 percentage points on single-venue Binance data

## Results Summary

**Reconstruction Performance (Binance Data):**
- CP (rank-4): 96.55% explained variance
- Tucker (4×1×4×4): 96.56% explained variance
- PCA baseline: 92.31% explained variance
- **Improvement:** 4.6 percentage points

**Extracted Factors:**
1. Bitcoin dominance and systematic risk
2. Altcoin rotation and risk appetite
3. Volatility regimes and liquidity shocks (spikes during Nov 2024 election, March 2025 banking crisis)
4. Intraday microstructure and mean reversion (24-hour oscillations)

See [paper](Farzulla_2025_Tensor_Structure_v1.0.0.pdf) for full analysis.

## Future Work

**Planned extensions:**
- **Multi-venue expansion:** Adding Coinbase, Kraken (CEX) and Uniswap, Curve (DEX) to test cross-exchange arbitrage dynamics
- **Robustness validation:** Bootstrap confidence intervals for factor loadings, out-of-sample tests
- **Higher-dimensional features:** Incorporating MEV, gas prices, order book depth
- **Information geometry framework:** Computing Riemannian metrics and Ricci curvature for regime detection
- **Real-time deployment:** Implementing rolling window decomposition for live regime tracking

## Citation

If you use this work, please cite:

**Paper (APA):**
```
Farzulla, M. (2025). Tensor structure in cryptocurrency microstructure: Low-rank CP decomposition from Binance OHLCV—Single-venue proof-of-concept (Version 1.0.0). Farzulla Research. https://doi.org/10.5281/zenodo.17626899
```

**BibTeX:**
```bibtex
@techreport{Farzulla2025TensorStructure,
  author = {Farzulla, Murad},
  title = {Tensor Structure in Cryptocurrency Microstructure: Low-Rank CP Decomposition from Binance OHLCV—Single-Venue Proof-of-Concept},
  institution = {Farzulla Research},
  year = {2025},
  type = {Preprint},
  version = {1.0.0},
  doi = {10.5281/zenodo.17626899},
  url = {https://github.com/resurrexi/tensor-defi}
}
```

## License

- **Code:** MIT License (see [LICENSE](LICENSE))
- **Paper:** CC-BY-4.0
- **Data:** Public market data from Binance (reproducible via CCXT)

## Reproducibility

**Software versions:**
- Python: 3.13+
- TensorLy: 0.8.1
- NumPy: 1.26.4
- CCXT: 4.4+

**Hardware:**
- **Development:** Resurrexi Lab (8 nodes, 66 cores, 229GB RAM, 48GB VRAM) with professional Kubernetes orchestration
- **Minimum:** 8GB RAM, any modern CPU (experiments run in 30-90 seconds)

**Random seed:** 42 (set in all decomposition scripts)

**Convergence criteria:** ALS tolerance 10⁻⁴, max 500 iterations

## Data Availability

All data, analysis code, figure generation scripts, and comprehensive documentation are publicly available at: https://github.com/resurrexi/tensor-defi

Complete replication materials include:
- Python 3.13+ analysis code using TensorLy v0.8.1
- Binance OHLCV data collection via CCXT library (public API)
- Tensor construction and normalization scripts
- CP/Tucker decomposition implementations with convergence diagnostics
- PCA baseline comparisons
- Factor visualization code
- Publication-ready figure generation

## Research Context

This work forms part of the **Adversarial Systems Research** program, which investigates stability, alignment, and friction dynamics in complex systems where competing interests generate structural conflict. The program examines how agents with divergent preferences interact within institutional constraints across multiple domains: political governance, financial markets (cryptocurrency volatility and regulatory responses), human cognitive development (trauma as maladaptive learning), and artificial intelligence alignment (multi-agent systems with competing objectives).

In financial markets, adversarial dynamics manifest as the tension between different market participants (high-frequency traders vs. long-term investors, centralized exchanges vs. decentralized protocols) and between market innovation and regulatory oversight. This paper applies tensor decomposition to cryptocurrency microstructure to reveal how multi-dimensional market dynamics encode these competing forces.

**Related publications:**
- [Cryptocurrency Event Study](https://doi.org/10.5281/zenodo.17677682) - Volatility responses to infrastructure vs regulatory shocks
- [Doctrine of Consensual Sovereignty](https://doi.org/10.5281/zenodo.17684676) - Stakeholder consent and friction dynamics

Explore all research: [zenodo.org/communities/farzulla](https://zenodo.org/communities/farzulla/)

## Acknowledgments

This research benefited from Perplexity AI for efficient literature discovery and source verification, and Anthropic's Claude for assistance with analytical framework development, custom tensor decomposition implementation, literature synthesis, and technical writing. All conceptual innovations, theoretical claims, and interpretive judgments remain the author's sole responsibility.

All computational analysis was conducted at **Resurrexi Lab**, a distributed computing cluster built from consumer-grade hardware (8 nodes, 66 cores, 229GB RAM, 48GB VRAM), demonstrating that rigorous quantitative finance research is accessible without institutional supercomputing infrastructure.

---

**Author:** [Murad Farzulla](https://farzulla.org) | [Farzulla Research](https://farzulla.org)
**Contact:** murad@farzulla.org
**ORCID:** [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)

**Publication Status:** Preprint v1.0.0 (November 2025) | Farzulla Research
