# Tensor Decomposition for Cryptocurrency Market Microstructure

**Author:** Murad Farzulla
**Date:** November 2025
**Category:** Working Paper (Quantitative Finance, Computational Methods)

## Abstract

We propose a tensor decomposition framework for cryptocurrency market microstructure that captures multi-dimensional relationships beyond traditional matrix methods. Using one year of hourly OHLCV data (8,761 hours) for BTC, ETH, and SOL from Binance, we construct a 4-dimensional tensor (time × venue × asset × feature) and compare tensor methods against PCA baselines.

Tensor decomposition reveals remarkable low-dimensional structure: rank-4 CP and Tucker decompositions achieve 96.55% and 96.56% explained variance respectively, vastly outperforming traditional PCA (92.31%). Factor analysis reveals interpretable components corresponding to market-wide trends, asset-specific effects, volatility regimes, and microstructure noise.

Our findings validate that cryptocurrency markets evolve on curved, low-dimensional manifolds induced by microstructure dynamics. This framework enables efficient dimensionality reduction, improved forecasting, and principled multi-asset portfolio construction in high-dimensional cryptocurrency markets.

## Contents

**Main Paper:**
- `Farzulla_2025_Tensor_Decomposition.tex` - Pure LaTeX source (two-column academic format)
- `Farzulla_2025_Tensor_Decomposition.pdf` - Compiled PDF version
- `Farzulla_2025_Tensor_Decomposition.md` - Original Markdown source (archived)

**Figures:** (6 high-resolution PNG files in `figures/`)
- `rank_selection.png` - Rank selection via explained variance
- `cp_factors.png` - CP decomposition factor loadings
- `method_comparison.png` - Tensor vs. matrix method comparison
- `reconstruction_quality.png` - Reconstruction error analysis
- `temporal_evolution.png` - Temporal factor evolution over time
- `asset_factor_space.png` - Asset factor space visualization

**Documentation:**
- `LATEX_BUILD_GUIDE.md` - Comprehensive LaTeX build documentation (all quirks documented)
- `Makefile` - Build automation (4-pass pdflatex + bibtex workflow)
- `references.bib` - BibTeX bibliography (60+ citations)

## Key Findings

**Exceptional Variance Explained (Low-Dimensional Manifold):**
- Rank-4 CP: 96.55% explained variance
- Rank-4 Tucker: 96.56% explained variance
- Traditional PCA: 92.31% explained variance
- **4.24 percentage point improvement** over matrix methods

**Four Interpretable Factors:**
1. **Market-Wide Trend** (captures synchronized movements)
2. **Asset-Specific Dynamics** (BTC vs ETH vs SOL characteristics)
3. **Volatility Regime** (high/low volatility states)
4. **Microstructure Noise** (venue-specific effects, bid-ask spreads)

**Implications:**
- Cryptocurrency markets lie on curved, low-dimensional manifolds
- Multi-dimensional structure cannot be captured by matrix methods alone
- Tensor methods enable principled dimensionality reduction
- Improved forecasting through factor-based models
- Portfolio construction via interpretable components

## How to Cite

```bibtex
@techreport{farzulla2025tensor,
  author = {Farzulla, Murad},
  title = {Tensor Decomposition for Cryptocurrency Market Microstructure:
           Beyond Matrix Methods},
  year = {2025},
  type = {Working Paper},
  institution = {Farzulla Research},
  note = {Available at Zenodo: \url{https://doi.org/10.5281/zenodo.XXXXXX}}
}
```

## Keywords

tensor decomposition, cryptocurrency, market microstructure, CP decomposition, Tucker decomposition, CANDECOMP/PARAFAC, dimensionality reduction, OHLCV data, manifold learning, multi-way analysis

## License

CC-BY-4.0 (Creative Commons Attribution 4.0 International)
See LICENSE.txt for full license text

## Compilation Instructions

**Build System:** Pure LaTeX (two-column academic format)

**Prerequisites:**
```bash
# Arch Linux
sudo pacman -S texlive-core texlive-latexextra texlive-bibtexextra make

# Ubuntu/Debian
sudo apt install texlive-latex-base texlive-latex-extra texlive-bibtex-extra make
```

**Build Commands:**
```bash
make          # Full 4-pass build (pdflatex + bibtex + pdflatex + pdflatex)
make clean    # Remove auxiliary files (.aux, .log, etc.)
make distclean # Remove all generated files including PDF
make quick    # Single-pass compile (fast, for debugging LaTeX errors)
make help     # Show all available targets
```

**Why 4 passes?**
1. `pdflatex` - Generate .aux file with citations and references
2. `bibtex` - Process bibliography, create .bbl file
3. `pdflatex` - Incorporate bibliography into document
4. `pdflatex` - Resolve all cross-references and citations

**For detailed build documentation, quirks, and troubleshooting:** See `LATEX_BUILD_GUIDE.md`

## Data Sources

- **Exchange:** Binance (via `ccxt` library)
- **Assets:** BTC/USDT, ETH/USDT, SOL/USDT
- **Time Period:** December 2023 - December 2024 (1 year)
- **Frequency:** Hourly OHLCV data (8,761 observations)
- **Features:** Open, High, Low, Close, Volume, Returns, Log-Returns, Volatility

## Version History

- **v1.0 (November 2025):** Initial release
  - CP and Tucker decomposition implementation
  - Rank selection via explained variance
  - Factor interpretation and visualization
  - PCA baseline comparison
  - Code and data availability statements
