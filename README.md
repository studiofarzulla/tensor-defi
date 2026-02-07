# Do Whitepaper Claims Predict Market Behavior?

**Evidence from Cryptocurrency Factor Analysis**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17917922-blue.svg)](https://doi.org/10.5281/zenodo.17917922)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status](https://img.shields.io/badge/Status-With_Editor-yellow.svg)](https://doi.org/10.5281/zenodo.17917922)
[![arXiv](https://img.shields.io/badge/arXiv-2601.20336-b31b1b.svg)](https://arxiv.org/abs/2601.20336)

**Working Paper DAI-2508** | [Dissensus AI](https://dissensus.ai)

## Abstract

This study investigates whether cryptocurrency whitepaper narratives align with empirically observed market factor structure. We construct a pipeline combining zero-shot NLP classification of 38 whitepapers across 10 semantic categories with CP tensor decomposition of hourly market data (49 assets, 17,543 timestamps). Using Procrustes rotation and Tucker's congruence coefficient, we find weak alignment between claims and market statistics (phi = 0.246, p = 0.339) and between claims and latent factors (phi = 0.058, p = 0.751). A methodological validation comparison---statistics versus factors, both derived from market data---achieves significance (p < 0.001), confirming the pipeline detects real structure. The null result indicates whitepaper narratives do not meaningfully predict market factor structure, with implications for narrative economics and investor decision-making. Entity-level analysis reveals specialized tokens (XMR, CRV, YFI) show stronger narrative--market correspondence than broad infrastructure tokens.

## Key Findings

| Finding | Result |
|---------|--------|
| Claims-Statistics alignment | phi = 0.246 (weak, p = 0.339) |
| Claims-Factors alignment | phi = 0.058 (negligible, p = 0.751) |
| Pipeline validation (Stats vs Factors) | Significant (p < 0.001) |
| Variance explained by CP decomposition | 92.45% (rank-2) |
| Assets analyzed | 49 cryptocurrencies, 17,543 timestamps |

## Keywords

cryptocurrency, tensor decomposition, NLP, factor analysis, Procrustes rotation, Tucker's congruence coefficient, zero-shot classification

## Repository Structure

```
tensor-defi/
├── src/                      # Python modules
│   ├── alignment/            # Procrustes alignment methods
│   ├── nlp/                  # NLP classification pipeline
│   ├── tensor_ops/           # CP decomposition operations
│   ├── market/               # Market data processing
│   ├── visualization/        # Plotting utilities
│   └── stats/                # Statistical tests
├── scripts/                   # Analysis pipeline scripts
│   ├── run_full_pipeline.py  # Complete end-to-end pipeline
│   ├── run_nlp.py            # NLP classification
│   ├── run_tensor.py         # Tensor construction
│   ├── run_alignment.py      # Factor alignment
│   └── run_figures.py        # Figure generation
├── paper/                     # LaTeX source
│   ├── main-arxiv.tex        # Paper source
│   ├── references.bib        # Bibliography
│   └── figures/              # Paper figures
├── data/                      # Input data (included)
│   ├── whitepapers/          # PDF corpus
│   └── market/               # Parquet market data
├── outputs/                   # Pipeline outputs
├── figures/                   # Generated figures
├── CITATION.cff
├── requirements.txt
└── LICENSE
```

## Usage

### Full Pipeline

```bash
python scripts/run_full_pipeline.py
```

### Individual Steps

```bash
python scripts/run_nlp.py          # NLP classification of whitepapers
python scripts/run_tensor.py       # Build market tensor
python scripts/run_alignment.py    # Compute factor alignment
python scripts/run_figures.py      # Generate figures
```

### Hardware Requirements

- **RAM:** 16GB minimum (32GB recommended for full tensor operations)
- **GPU:** Optional but recommended for NLP inference (CUDA/ROCm supported)

## Citation

```bibtex
@article{farzulla2026whitepaper,
  author  = {Farzulla, Murad},
  title   = {Do Whitepaper Claims Predict Market Behavior? Evidence from Cryptocurrency Factor Analysis},
  year    = {2026},
  journal = {arXiv preprint arXiv:2601.20336},
  doi     = {10.5281/zenodo.17917922}
}
```

## Authors

- **Murad Farzulla** -- [Dissensus AI](https://dissensus.ai) & King's College London
  - ORCID: [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)
  - Email: murad@dissensus.ai

## License

Paper content: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
