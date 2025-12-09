# Testing Narrative-Market Alignment in Cryptocurrency

**A Methodological Framework**

**Author:** Murad Farzulla
**Affiliations:** King's Business School, King's College London | Farzulla Research
**Status:** Preprint v2.0.1
**Date:** December 2025
[![Paper DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17772652.svg)](https://doi.org/10.5281/zenodo.17772652)
[![Code DOI](https://zenodo.org/badge/1083436148.svg)](https://doi.org/10.5281/zenodo.17688564)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Abstract

Do the functional purposes articulated in cryptocurrency whitepapers correspond to how markets actually price these assets? We propose a three-stage framework combining NLP with market characterization to test narrative-market alignment.

**Stage 1:** Extract functional profiles from whitepapers using zero-shot classification against a 10-category taxonomy
**Stage 2:** Construct market profiles from trading data (summary statistics)
**Stage 3:** Measure alignment using Procrustes rotation and Tucker's congruence coefficient

A pilot on 8 cryptocurrencies yields overall congruence of **φ = 0.719** (95% CI: [0.623, 0.953]). We emphasize the methodology over point estimates—the small sample precludes robust empirical inference. The framework enables systematic detection of narrative-market divergence, with applications in due diligence, market efficiency analysis, and regulatory classification.

## Research Question

> Do cryptocurrency whitepapers capture *static founding mandates* that markets subsequently reinterpret as *evolved utility*?

**Key insight:** Bitcoin's whitepaper emphasizes "peer-to-peer electronic cash" while markets price BTC as "digital gold"—illustrating the core problem. Whitepapers capture static founding mandates, while market profiles capture evolved market utility. The gap between them is what this framework measures.

## Methodology

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CLAIMS MATRIX (X)                              │
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
│                                          │  (8 × 10)    │          │
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
│                      MARKET MATRIX (Y)                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Binance    │───►│   Summary    │───►│    Market    │          │
│  │    OHLCV     │    │  Statistics  │    │   Profile    │          │
│  │   (CCXT)     │    │ Computation  │    │   Vectors    │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                   │                 │
│                                                   ▼                 │
│                                          ┌──────────────┐          │
│                                          │   Market     │          │
│                                          │   Matrix     │          │
│                                          │  (8 × 7)     │          │
│                                          └──────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

### Functional Taxonomy (10 Categories)

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

### Market Dimensions (7 Summary Statistics)

| Dimension | Description |
|-----------|-------------|
| `mean_return` | Average daily return |
| `volatility` | Standard deviation of returns |
| `sharpe` | Risk-adjusted return |
| `max_drawdown` | Maximum peak-to-trough decline |
| `avg_volume` | Average trading volume |
| `vol_volatility` | Volatility of volatility (volume variance) |
| `trend` | Linear trend coefficient |

### Alignment Measurement

1. **Extract functional profiles** from whitepapers using BART-MNLI zero-shot classification
2. **Construct claims matrix** X ∈ ℝ^(8 × 10) (entities × functional categories)
3. **Construct market matrix** Y ∈ ℝ^(8 × 7) (entities × summary statistics)
4. **Apply Procrustes rotation** to align heterogeneous spaces
5. **Compute Tucker's congruence coefficient** φ for dimension similarity

**Interpretation thresholds:**
- φ ≥ 0.95: Dimensions equivalent
- φ = 0.85–0.94: Fair similarity
- φ = 0.65–0.84: Some similarity
- φ < 0.65: Dimensions distinct

## Pilot Results (Illustrative Only)

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

**Per-Dimension Congruence:**

| Dimension | φ | Interpretation |
|-----------|---|----------------|
| Dim 1 | 0.301 | Distinct (claims ≠ market) |
| Dim 2 | 0.867 | Fair similarity |
| Dim 3 | 0.728 | Some similarity |
| Dim 4 | 0.982 | Near-equivalent |

**Interpretation:** Results suggest non-random correspondence but also systematic divergence in specific dimensions. The N=8 sample is illustrative—see paper for limitations discussion.

## Repository Contents

```
tensor-defi/
├── README.md                           # This file
├── CITATION.cff                        # Citation metadata
├── VERSION                             # Version tracking
├── LICENSE                             # MIT for code, CC-BY-4.0 for paper
├── ZENODO_METADATA.md                  # Zenodo upload metadata
├── Farzulla_2025_Narrative_Alignment_v2.0.1.pdf    # Compiled paper
├── Farzulla_2025_Narrative_Alignment_v2.0.1.tex    # LaTeX source
├── references.bib                      # Bibliography
├── src/
│   ├── nlp/                            # NLP pipeline
│   │   ├── whitepaper_collector.py
│   │   ├── text_processor.py
│   │   ├── claim_extractor.py          # Zero-shot classification
│   │   └── taxonomy.py                 # Functional categories
│   ├── market/                         # Market data processing
│   │   └── summary_statistics.py       # Market profile construction
│   ├── alignment/                      # Alignment testing
│   │   ├── procrustes.py
│   │   └── congruence.py
│   └── visualization/
├── scripts/
│   ├── run_nlp_pipeline.py
│   ├── run_market_analysis.py
│   └── run_alignment_analysis.py
├── data/                               # Whitepaper PDFs
├── outputs/
│   ├── nlp/                            # Functional profiles
│   ├── alignment/                      # Congruence results
│   └── figures/
├── figures/                            # Publication figures
│   ├── market_profiles.png             # Market summary statistics heatmap
│   ├── claims_vs_market_space.png      # Procrustes overlay visualization
│   └── per_dimension_congruence.png    # Per-dimension φ values
└── requirements.txt
```

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

# 3. Build market profiles and run alignment
python scripts/run_alignment_analysis.py
```

### Pre-computed Results

All results are included in `outputs/`:
- `outputs/nlp/functional_profiles.json` - Extracted functional profiles
- `outputs/alignment/alignment_results.json` - Congruence coefficients
- `figures/` - Publication-ready visualizations

## Mathematical Framework

### Procrustes Alignment

Given claims matrix X and market matrix Y with different dimensionalities:

```
R* = argmin_R ||Y - XR||_F  s.t. R^T R = I
```

Closed-form solution via SVD of Y^T X.

### Tucker's Congruence Coefficient

```
φ = Σ x_i y_i / √(Σ x_i² · Σ y_i²)
```

Measures dimension similarity independent of scale.

## Limitations

This is a **methodology proposal**, not robust empirical findings:

- **Small Sample (N=8):** Pilot illustrates framework; insufficient for parametric inference
- **OHLC Collinearity:** Summary statistics constructed from same price series introduce dependencies
- **Temporal Mismatch:** Static whitepaper claims vs. dynamic market profiles
- **Bootstrap Assumptions:** CI validity depends on assumptions that may not hold at N=8

See paper Section 6 for comprehensive limitations discussion.

## Related Work

- **v1.0.0** (archived): [Tensor Structure in Cryptocurrency Microstructure](https://doi.org/10.5281/zenodo.17688564) - Single-venue OHLCV proof-of-concept
- [Market Reaction Asymmetry](https://doi.org/10.5281/zenodo.17677682) - Infrastructure vs regulatory volatility (TARCH-X event study)
- [Doctrine of Consensual Sovereignty](https://doi.org/10.5281/zenodo.17684676) - Adversarial systems framework

## Citation

### Paper Citation

```bibtex
@techreport{Farzulla2025NarrativeAlignment,
  author = {Farzulla, Murad},
  title = {Testing Narrative-Market Alignment in Cryptocurrency: A Methodological Framework},
  institution = {Farzulla Research},
  year = {2025},
  type = {Preprint},
  version = {2.0.1},
  doi = {10.5281/zenodo.17772652},
  url = {https://github.com/studiofarzulla/tensor-defi}
}
```

### Repository Citation

See `CITATION.cff` for structured citation metadata (Zenodo/GitHub compatible).

## License

- **Code:** MIT License
- **Paper:** CC-BY-4.0
- **Data:** Public whitepapers + Binance market data

---

**Author:** [Murad Farzulla](https://farzulla.org) | [Farzulla Research](https://farzulla.org)
**Contact:** murad@farzulla.org
**ORCID:** [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)
