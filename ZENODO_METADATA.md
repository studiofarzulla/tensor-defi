# Zenodo Metadata for TENSOR-DEFI v3.0.0

## Title
Do Whitepaper Claims Predict Market Behavior? Evidence from Cryptocurrency Factor Analysis

## Authors
Murad Farzulla (ORCID: 0009-0004-9646-9582)

---

## HTML Description (copy this into Zenodo)

```html
<p><strong>Do Whitepaper Claims Predict Market Behavior? Evidence from Cryptocurrency Factor Analysis</strong></p>

<p>This research investigates whether the functional claims made in cryptocurrency whitepapers exhibit measurable alignment with subsequent market behavior patterns. We combine zero-shot NLP classification with tensor decomposition to analyze narrative-market relationships across the cryptocurrency ecosystem.</p>

<h3>Methodology</h3>
<ul>
  <li><strong>NLP Pipeline:</strong> Zero-shot classification using BART-MNLI extracts semantic features from 13 major cryptocurrency whitepapers across 10 functional categories (decentralization, smart contracts, DeFi, scalability, privacy, interoperability, etc.)</li>
  <li><strong>Tensor Decomposition:</strong> CP (CANDECOMP/PARAFAC) decomposition of market data tensor (17,543 × 49 × 5) achieves 92.45% variance explained with rank-2 factors</li>
  <li><strong>Alignment Measurement:</strong> Procrustes rotation with Tucker's congruence coefficient (φ) quantifies narrative-market correspondence</li>
</ul>

<h3>Key Findings</h3>
<ul>
  <li>Primary claims-statistics alignment: <strong>φ = 0.331</strong> (moderate, p < 0.001)</li>
  <li>BTC dominance factor loading: <strong>28.5</strong> (massive outlier driving Factor 1)</li>
  <li>"Smart contracts" and "interoperability" categories show strongest market correspondence</li>
  <li>DeFi tokens (AAVE, UNI, ENS) exhibit largest narrative-market gaps</li>
  <li>Temporal stability: alignment persists across 6 rolling windows (φ = 0.232 ± 0.023)</li>
</ul>

<h3>Contributions</h3>
<ul>
  <li>Novel methodology combining NLP-extracted semantic features with tensor factor analysis</li>
  <li>First systematic comparison of whitepaper narratives against realized market behavior</li>
  <li>Evidence for bounded rationality in crypto markets: narratives persist despite weak predictive power</li>
  <li>Reproducible pipeline with full code and data</li>
</ul>

<h3>Repository Contents</h3>
<ul>
  <li><code>paper/</code> - Full LaTeX paper (24 pages, two-column) with 10 figures</li>
  <li><code>scripts/</code> - Complete analysis pipeline (data collection, NLP, tensor ops, alignment)</li>
  <li><code>data/</code> - Market data (49 assets) and whitepaper corpus (13 PDFs)</li>
  <li><code>outputs/</code> - All analysis results in JSON/CSV format</li>
</ul>

<p><strong>Related Work:</strong> This paper extends <a href="https://doi.org/10.2139/ssrn.5788082">Farzulla (2025)</a> on cryptocurrency market microstructure, demonstrating that infrastructure disruption events dominate regulatory uncertainty in volatility dynamics.</p>
```

---

## Abstract (plain text)

This paper investigates whether the functional claims made in cryptocurrency whitepapers exhibit measurable alignment with subsequent market behavior patterns. Using zero-shot NLP classification and tensor decomposition, we analyze 13 major cryptocurrency whitepapers against market data for 49 assets over 17,543 daily observations. Our CP tensor decomposition achieves 92.45% variance explained with rank-2 factors, revealing a "BTC dominance" factor (loading: 28.5) and a "diversified altcoin" factor. The primary claims-statistics alignment yields Tucker's φ = 0.331 (moderate alignment, p < 0.001), with "smart contracts" and "interoperability" categories showing strongest correspondence to market patterns. Leave-one-out analysis reveals Bitcoin contributes positively to alignment (+0.025) while DeFi tokens show the largest narrative-market gaps. Temporal analysis across rolling windows confirms alignment stability (φ = 0.232 ± 0.023). We contribute novel methodology combining NLP-extracted semantic features with tensor factor analysis for financial prediction, finding that while whitepapers capture some market-relevant information, their explanatory power remains limited—suggesting markets respond to factors beyond documented project narratives.

---

## Keywords (comma-separated)

```
cryptocurrency, tensor decomposition, natural language processing, factor analysis, whitepaper analysis, market behavior, PARAFAC, Tucker congruence, Procrustes rotation, zero-shot classification, DeFi, Bitcoin, narrative economics, market microstructure
```

---

## Additional Zenodo Fields

**License:** CC-BY-4.0

**Publication Date:** 2025-12-13

**Version:** 3.0.0

**Language:** English

**Resource Type:** Dataset (or Preprint if submitting paper separately)

**Related Identifiers:**
- GitHub Repository: https://github.com/studiofarzulla/tensor-defi (isSupplementTo)
- Related Publication: https://doi.org/10.2139/ssrn.5788082 (references)

**Grants/Funding:** None

**Communities:** Consider adding to relevant Zenodo communities (e.g., "Machine Learning", "Finance", "Cryptocurrency")
