# Zenodo Metadata for tensor-defi v2.0.1

## Title

Testing Narrative-Market Alignment in Cryptocurrency: A Methodological Framework

## Authors

Farzulla, Murad (ORCID: 0009-0002-7164-8704)

## Description (copy-paste ready)

This paper proposes a methodological framework for testing whether cryptocurrency whitepaper claims align with market behavior. The framework combines natural language processing with market characterization to measure narrative-market correspondence.

The three-stage methodology:
1. Extract functional profiles from whitepapers using zero-shot classification (BART-MNLI) against a 10-category taxonomy
2. Construct market profiles from trading data (returns, volatility, Sharpe ratio, drawdown, volume metrics)
3. Measure alignment using Procrustes rotation and Tucker's congruence coefficient

Key insight: Bitcoin's whitepaper emphasizes "peer-to-peer electronic cash" while markets price BTC as "digital gold" - illustrating the core problem. Whitepapers capture static founding mandates, while market profiles capture evolved market utility. The gap between them is what this framework measures.

Pilot application on 8 cryptocurrencies yields overall congruence of 0.719 (95% CI: [0.623, 0.953]), interpreted as "some similarity." However, the N=8 sample is illustrative only - the contribution is the methodology, not the point estimates.

This work forms part of the Adversarial Systems Research program investigating alignment and friction dynamics in complex systems.

## Abstract (copy-paste ready)

Do the functional purposes articulated in cryptocurrency whitepapers correspond to how markets actually price these assets? We propose a three-stage framework combining NLP with market characterization to test narrative-market alignment. Stage 1: Extract functional profiles from whitepapers using zero-shot classification against a 10-category taxonomy. Stage 2: Construct market profiles from trading data. Stage 3: Measure alignment using Procrustes rotation and Tucker's congruence coefficient. A pilot on 8 cryptocurrencies yields overall congruence of 0.719 (95% CI: [0.623, 0.953]). We emphasize the methodology over point estimates - the small sample precludes robust empirical inference. The framework enables systematic detection of narrative-market divergence, with applications in due diligence, market efficiency analysis, and regulatory classification.

## Keywords (comma-separated, copy-paste ready)

cryptocurrency, narrative-market alignment, natural language processing, whitepaper analysis, Tucker congruence, Procrustes rotation, market microstructure, zero-shot classification, functional taxonomy, alignment testing

## License

Creative Commons Attribution 4.0 International (CC BY 4.0)

## Related Identifiers

- Is supplement to: https://github.com/studiofarzulla/tensor-defi (GitHub repository)
- References: https://doi.org/10.5281/zenodo.17677682 (Farzulla 2025, Market Reaction Asymmetry)

## Resource Type

Preprint

## Version

2.0.1

## Language

English

## Subjects (Zenodo categories)

- Finance
- Computer Science
- Quantitative Methods
