# Crypto Narrative-Market Alignment

**Do Whitepaper Claims Predict Market Factor Structure?**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Version 2.0.0** | November 2025 | [Murad Farzulla](https://farzulla.org)

## Overview

This research tests whether **functional claims in cryptocurrency whitepapers align with market factor structures**. We extract functional taxonomies from project documentation using NLP, construct tensor representations of both narrative and market dynamics, and measure alignment using Procrustes rotation and Tucker's congruence coefficient.

**Key Finding:** Whitepaper functional profiles show statistically significant but imperfect alignment with market factors (φ = 0.72, 95% CI: [0.62, 0.95]), with one factor achieving near-perfect congruence (φ = 0.98).

## Research Question

> Do the functional purposes articulated in cryptocurrency whitepapers (store of value, smart contracts, infrastructure, etc.) correspond to how markets actually price these assets?

This addresses a genuine research gap: while plenty of work examines either textual analysis of crypto documentation OR market microstructure, nobody has systematically tested the alignment between these domains.

## Methodology

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     NARRATIVE TENSOR (X)                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Whitepaper  │───►│  Zero-Shot   │───►│  Functional  │          │
│  │  Extraction  │    │Classification│    │   Profile    │          │
│  │  (PyMuPDF)   │    │ (BART-MNLI)  │    │   Vectors    │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                   │                 │
│                                                   ▼                 │
│                                          ┌──────────────┐          │
│                                          │   Claims     │          │
│                                          │   Matrix     │          │
│                                          │ (8 × 10 dim) │          │
│                                          └──────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
                                          ┌──────────────┐
                                          │  Procrustes  │
                                          │  Alignment   │
                                          │  + Tucker's  │
                                          │     φ        │
                                          └──────────────┘
                                                   ▲
                                                   │
┌─────────────────────────────────────────────────────────────────────┐
│                      MARKET TENSOR (Y)                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Binance    │───►│    Tensor    │───►│     CP       │          │
│  │    OHLCV     │    │Construction  │    │Decomposition │          │
│  │   (CCXT)     │    │ (TensorLy)   │    │  (rank-4)    │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                   │                 │
│                                                   ▼                 │
│                                          ┌──────────────┐          │
│                                          │   Market     │          │
│                                          │   Factors    │          │
│                                          │ (8 × 4 dim)  │          │
│                                          └──────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

### Functional Taxonomy

Zero-shot classification against 10 functional categories:

| Category | Description | Example Claims |
|----------|-------------|----------------|
| `store_of_value` | Digital gold, inflation hedge | "digital scarcity", "deflationary" |
| `medium_of_exchange` | Payments, remittances | "micropayments", "peer-to-peer cash" |
| `smart_contracts` | DeFi, lending, derivatives | "programmable money", "composability" |
| `infrastructure` | Scaling, interoperability | "layer 1", "sharding", "rollups" |
| `privacy` | Anonymous transactions | "zero-knowledge", "confidential" |
| `governance` | DAOs, voting | "on-chain governance", "staking" |
| `data_oracle` | Off-chain data feeds | "price feeds", "external data" |
| `identity` | Decentralized identity | "DID", "verifiable credentials" |
| `gaming_metaverse` | NFTs, gaming | "play-to-earn", "metaverse" |
| `stablecoin` | Price stability | "pegged", "algorithmic stable" |

### Alignment Testing

1. **Extract functional profiles** from whitepapers using BART-MNLI zero-shot classification
2. **Construct claims matrix** X ∈ ℝ^(n_entities × n_functions)
3. **Construct market matrix** Y ∈ ℝ^(n_entities × n_factors) from CP decomposition
4. **Apply Procrustes rotation** to align factor spaces
5. **Compute Tucker's congruence coefficient** φ for factor similarity

**Interpretation thresholds:**
- φ ≥ 0.95: Factors equivalent
- φ = 0.85-0.94: Fair similarity
- φ = 0.65-0.84: Some similarity
- φ < 0.65: Factors distinct

## Results

### Entity Universe

| Asset | Project | Primary Function | Top Category Score |
|-------|---------|------------------|-------------------|
| BTC | Bitcoin | Medium of Exchange | 45.9% |
| ETH | Ethereum | Generalist | (evenly distributed) |
| SOL | Solana | Infrastructure | 52.0% |
| AVAX | Avalanche | Infrastructure | 64.1% |
| DOT | Polkadot | Governance | 63.3% |
| FIL | Filecoin | Infrastructure | 50.9% |
| LINK | Chainlink | Data Oracle | 54.2% |
| ALGO | Algorand | Identity | 45.4% |

### Alignment Results

**Overall Congruence:** φ = 0.719 (Some similarity)

**95% Bootstrap CI:** [0.623, 0.953]

**Per-Factor Congruence:**

| Factor | φ | Interpretation |
|--------|---|----------------|
| Factor 1 | 0.301 | Distinct (claims ≠ market) |
| Factor 2 | 0.867 | Fair similarity |
| Factor 3 | 0.728 | Some similarity |
| Factor 4 | 0.982 | Near-equivalent |

### Interpretation

- **Factor 4** shows near-perfect alignment (φ = 0.98): whatever this market factor captures, whitepapers describe it accurately
- **Factor 1** shows poor alignment (φ = 0.30): claims in this dimension diverge from market reality—potential marketing vs. fundamentals gap
- **Overall** alignment is significant but imperfect: whitepapers partially predict market positioning but aren't the whole story

## Quick Start

### Installation

```bash
git clone https://github.com/studiofarzulla/tensor-defi.git
cd tensor-defi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# 1. Collect whitepapers (or use included data/)
python scripts/collect_whitepapers.py

# 2. Run NLP extraction
python scripts/run_nlp_pipeline.py

# 3. Build market tensor and run decomposition
python scripts/run_full_experiments.py

# 4. Test alignment
python scripts/run_alignment_analysis.py
```

### Pre-computed Results

All results are included in `outputs/`:
- `outputs/nlp/functional_profiles.json` - Extracted functional profiles
- `outputs/alignment/alignment_results.json` - Congruence coefficients
- `figures/` - Publication-ready visualizations

## Project Structure

```
tensor-defi/
├── src/
│   ├── nlp/                      # NLP pipeline
│   │   ├── whitepaper_collector.py
│   │   ├── text_processor.py
│   │   ├── claim_extractor.py    # Zero-shot classification
│   │   └── taxonomy.py           # Functional categories
│   ├── tensor_ops/               # Tensor operations
│   │   ├── tensor_builder.py
│   │   ├── decomposition.py      # CP, Tucker
│   │   └── market_tensor.py
│   ├── alignment/                # Alignment testing
│   │   ├── procrustes.py
│   │   └── congruence.py
│   └── visualization/
├── scripts/
│   ├── run_nlp_pipeline.py
│   ├── run_full_experiments.py
│   └── run_alignment_analysis.py
├── data/                         # Whitepaper PDFs
├── outputs/
│   ├── nlp/                      # Functional profiles
│   ├── alignment/                # Congruence results
│   └── figures/
├── figures/                      # Publication figures
└── requirements.txt
```

## Mathematical Framework

### Tensor Decomposition (Market Side)

Market microstructure as 3-way tensor:
```
𝓧 ∈ ℝ^(T × A × F)

T = Time (hourly timestamps)
A = Assets (8 cryptocurrencies)
F = Features (OHLCV)
```

CP decomposition:
```
𝓧 ≈ Σ_{r=1}^R λ_r · (a_r ⊗ b_r ⊗ c_r)
```

Rank-4 captures 96.56% explained variance.

### Procrustes Alignment

Given claims matrix X and market factors Y:
```
R* = argmin_R ||Y - XR||_F  s.t. R^T R = I
```

Closed-form solution via SVD of Y^T X.

### Tucker's Congruence Coefficient

```
φ = Σ x_i y_i / √(Σ x_i² · Σ y_i²)
```

Measures factor similarity independent of scale.

## Dependencies

```txt
# Core
tensorly>=0.8.1
numpy>=1.26.4
scipy>=1.11.0

# NLP
transformers>=4.35.0
PyMuPDF>=1.23.0

# Data
ccxt>=4.4
pandas>=2.1.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0
```

## Citation

```bibtex
@techreport{Farzulla2025CryptoNarrative,
  author = {Farzulla, Murad},
  title = {Crypto Narrative-Market Alignment: Do Whitepaper Claims Predict Factor Structure?},
  institution = {Farzulla Research},
  year = {2025},
  type = {Preprint},
  version = {2.0.0},
  url = {https://github.com/studiofarzulla/tensor-defi}
}
```

## Related Work

- **v1.0.0** (archived): [Tensor Structure in Cryptocurrency Microstructure](https://doi.org/10.5281/zenodo.17688564) - Single-venue OHLCV proof-of-concept
- [Cryptocurrency Event Study](https://doi.org/10.5281/zenodo.17677682) - Infrastructure vs regulatory volatility
- [Doctrine of Consensual Sovereignty](https://doi.org/10.5281/zenodo.17684676) - Adversarial systems framework

## License

- **Code:** MIT License
- **Paper:** CC-BY-4.0
- **Data:** Public whitepapers + Binance market data

---

**Author:** [Murad Farzulla](https://farzulla.org) | [Farzulla Research](https://farzulla.org)
**Contact:** murad@farzulla.org
**ORCID:** [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)
