# Tensor-DeFi Pivot Plan: NLP + Tensor Alignment Research

**From:** OHLCV Tensor Decomposition (v1.0.0 - Published)  
**To:** Crypto Narrative-to-Market Alignment Testing

---

## Executive Summary

Transform tensor-defi from a single-venue OHLCV proof-of-concept into a novel research project testing whether **functional claims in crypto whitepapers align with market factor structures**. This addresses a genuine research gap: nobody has systematically extracted functional taxonomies from whitepapers and tested their alignment with market dynamics.

---

## Reusable Components (What We Keep)

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| **Tensor Decomposition** | `src/tensor_ops/decomposition.py` | ✅ Directly reusable | CP, Tucker, TT, rank selection |
| **Market Data Collector** | `src/data_pipeline/cex_collector.py` | ✅ Directly reusable | Binance OHLCV via CCXT |
| **Tensor Builder** | `src/tensor_ops/tensor_builder.py` | 🔧 Extensible | Add claims tensor mode |
| **Visualization** | `src/visualization/tensor_plots.py` | 🔧 Adapt | Factor comparison plots |
| **Baselines** | `src/baselines/traditional_methods.py` | ✅ Keep PCA | Add new baselines |

---

## New Components to Build

### Stage 1: NLP Pipeline (`src/nlp/`)

```
src/nlp/
├── __init__.py
├── whitepaper_collector.py    # PDF acquisition + text extraction
├── text_processor.py          # Chunking, cleaning, section detection
├── claim_extractor.py         # Zero-shot + fine-tuned classification
├── embedding_generator.py     # Sentence embeddings via Ollama
└── taxonomy.py                # Functional category definitions
```

**Key Design Decisions:**

1. **Model Stack:**
   - **Embeddings:** `nomic-embed-text` via Ollama (8192 context, 768 dims)
   - **Zero-shot Classification:** `facebook/bart-large-mnli` (HuggingFace)
   - **Optional Fine-tuning:** CryptoBERT for domain adaptation

2. **Functional Taxonomy (Initial):**
   ```python
   TAXONOMY = {
       "store_of_value": ["digital gold", "inflation hedge", "savings"],
       "medium_of_exchange": ["payments", "remittances", "micropayments"],
       "smart_contracts": ["defi", "lending", "derivatives", "nfts"],
       "infrastructure": ["scaling", "interoperability", "oracles"],
       "privacy": ["anonymous transactions", "confidential computing"],
       "governance": ["daos", "voting", "treasury management"],
   }
   ```

3. **Processing Pipeline:**
   ```
   PDF → PyMuPDF extraction → Section segmentation 
       → Semantic chunking (LangChain) → Multi-label classification
       → Sentence embeddings → Document-level aggregation
   ```

### Stage 2: Claims Tensor Builder (`src/tensor_ops/claims_tensor.py`)

**Tensor X (Claims):** `(Entities × Time × Embedding Features)`
- Entities: Crypto projects (BTC, ETH, SOL, etc.)
- Time: Weekly windows
- Features: Aggregated embedding dimensions or functional profile vectors

**Two Approaches:**

1. **Direct Embedding Approach:**
   - Generate sentence embeddings for all functional claim passages
   - Aggregate per entity-period via mean pooling
   - PCA reduce to ~50 dimensions for tractability

2. **Functional Profile Approach:**
   - Convert multi-label classification to soft vectors
   - Each dimension = presence/intensity of functional category
   - More interpretable, directly aligned with taxonomy

### Stage 3: Market Tensor Builder (`src/tensor_ops/market_tensor.py`)

**Tensor Y (Market):** `(Entities × Time × Market Features)`
- Entities: Same crypto projects as claims tensor
- Time: Same weekly windows
- Features: Market dynamics (from your spec)

**Market Features:**
```python
MARKET_FEATURES = [
    "log_returns",           # Weekly returns
    "realized_volatility",   # Intraweek std
    "volume_mcap_ratio",     # Turnover
    "amihud_illiquidity",    # Price impact
    "beta_crypto_market",    # Factor loading
    "momentum_4w",           # Trailing return
    "log_market_cap",        # Size
]
```

### Stage 4: Alignment Testing (`src/alignment/`)

```
src/alignment/
├── __init__.py
├── procrustes.py           # Orthogonal Procrustes rotation
├── congruence.py           # Tucker's congruence coefficient
├── granger.py              # VAR-based Granger causality
└── hypothesis_tests.py     # Liu et al. (2024) tensor alignment tests
```

**Core Algorithm:**
```python
def test_alignment(claims_tensor, market_tensor, rank=30):
    """Test whether claims factors predict market factors."""
    
    # 1. Decompose both tensors
    claims_factors = parafac(claims_tensor, rank=rank)
    market_factors = parafac(market_tensor, rank=rank)
    
    # 2. Extract entity loadings (what projects load on each factor)
    claims_entity = claims_factors.factors[0]  # (n_projects, rank)
    market_entity = market_factors.factors[0]  # (n_projects, rank)
    
    # 3. Apply Procrustes rotation to align factor spaces
    _, aligned_claims, disparity = procrustes(market_entity, claims_entity)
    
    # 4. Compute Tucker's Congruence Coefficient
    phi = tuckers_congruence(market_entity, aligned_claims)
    
    # φ ≥ 0.95: Factors equivalent
    # φ = 0.85-0.94: Fair similarity  
    # φ < 0.85: Factors distinct
    
    return AlignmentResult(
        congruence=phi,
        procrustes_disparity=disparity,
        claims_factors=claims_factors,
        market_factors=market_factors
    )
```

---

## Data Acquisition Strategy

### Whitepapers

| Source | Coverage | Method |
|--------|----------|--------|
| **AllCryptoWhitepapers.com** | 3,900+ | Web scraping |
| **Golden.com** | 15,267 | API (free tier) |
| **CryptoRating/whitepapers (GitHub)** | Organized repo | Git clone |

### Market Data

| Source | Coverage | Method |
|--------|----------|--------|
| **Binance** (existing) | Since 2017 | CCXT (already implemented) |
| **CoinGecko** | 10+ years | API (30 calls/min free) |
| **Binance bulk downloads** | Full history | data.binance.vision |

### Project Universe

Start with **top 50 by market cap** that have:
- Published whitepaper
- Sufficient trading history (1+ year)
- Active Binance listing

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Create `src/nlp/` module structure
- [ ] Implement `whitepaper_collector.py` with PDF extraction
- [ ] Set up Ollama with nomic-embed-text
- [ ] Define functional taxonomy in `taxonomy.py`
- [ ] Build basic `text_processor.py`

### Phase 2: NLP Pipeline (Week 2)
- [ ] Implement zero-shot classification with BART MNLI
- [ ] Build `claim_extractor.py` with multi-label output
- [ ] Implement `embedding_generator.py` with Ollama
- [ ] Test on 5-10 whitepapers manually

### Phase 3: Tensor Construction (Week 3)
- [ ] Build `claims_tensor.py` for functional embeddings
- [ ] Build `market_tensor.py` for expanded market features
- [ ] Extend `tensor_builder.py` with new modes
- [ ] Validate tensor alignment (same entities, same time windows)

### Phase 4: Alignment Analysis (Week 4)
- [ ] Implement Procrustes rotation
- [ ] Implement Tucker's congruence coefficient
- [ ] Build factor comparison visualizations
- [ ] Run initial alignment tests

### Phase 5: Causality & Validation (Week 5)
- [ ] Implement VAR-based Granger causality
- [ ] Rolling window analysis for regime detection
- [ ] Robustness checks (bootstrap, subsamples)
- [ ] Write up empirical results

### Phase 6: Paper & Release (Week 6)
- [ ] Update paper structure
- [ ] Generate publication-quality figures
- [ ] Update README and documentation
- [ ] Zenodo deposit (v2.0.0)

---

## Technical Requirements

### New Dependencies

```txt
# NLP
transformers>=4.35.0
sentence-transformers>=2.2.0
langchain>=0.1.0
PyMuPDF>=1.23.0
beautifulsoup4>=4.12.0

# Alignment Testing
scipy>=1.11.0  # Procrustes
statsmodels>=0.14.0  # VAR, Granger

# Optional (for GPU acceleration)
torch>=2.1.0
```

### Homelab Resource Allocation

| Component | Assignment |
|-----------|------------|
| **Embedding generation** | Ollama on PurrPower (7900 XTX) |
| **Tensor decomposition** | TensorLy with PyTorch backend |
| **Storage** | Local Zarr or pickle |
| **Document corpus** | ~50GB estimated |

---

## Key Research Questions

1. **Do functional claims cluster interpretably?**
   - BERTopic analysis of whitepaper corpus
   - Validate against manual taxonomy

2. **Do entity factors align between claims and market?**
   - Tucker's congruence ≥ 0.85 would be notable
   - Interpret which functional categories map to market factors

3. **Do claims *lead* market positioning?**
   - Granger causality: claims factors → market factors
   - Economic interpretation: forward-looking narrative

4. **Which claims are "priced" vs. marketing?**
   - Compare priced functional categories vs. pure rhetoric
   - Regime-dependent analysis

---

## Success Criteria

- [ ] Functional taxonomy discovered from unsupervised analysis
- [ ] Tucker's congruence coefficient computed with confidence intervals
- [ ] At least one statistically significant alignment result
- [ ] Granger causality tests completed (even if null)
- [ ] Paper structure updated with new methodology

---

## Questions Before Proceeding

1. **Project Universe:** Start with top 50 by market cap, or different selection?

2. **Whitepaper Sources:** Golden.com API vs. web scraping vs. manual collection?

3. **Embedding Model:** Ollama nomic-embed-text vs. OpenAI embeddings vs. other?

4. **GPU Acceleration:** Use PyTorch backend from start, or NumPy then optimize?

5. **Version Strategy:** Increment to v2.0.0 as major pivot, or new repo?

---

## File Cleanup (Post-Approval)

Files to **archive** (not delete - keep for v1.0.0 reference):
- `Farzulla_2025_Tensor_Structure_v1.0.0.tex/pdf` → `archive/v1/`
- `data/cex_data_*.csv` → May regenerate
- `outputs/` → Archive results

Files to **update**:
- `README.md` - New project description
- `METHODOLOGY.md` - New methodology
- `requirements.txt` - Add NLP deps
- `CITATION.cff` - New title/description

---

**Ready to proceed when you give the green light! 🚀**
