# Methodology: Crypto Narrative-Market Alignment

## Research Design

This study employs a mixed-methods approach combining natural language processing with tensor decomposition to test the alignment between narrative claims and market dynamics in cryptocurrency markets.

### Core Hypothesis

**H₀:** Whitepaper functional profiles are orthogonal to market factor loadings (φ < 0.65)

**H₁:** Whitepaper functional profiles show significant alignment with market factors (φ ≥ 0.65)

---

## Stage 1: NLP Pipeline

### 1.1 Whitepaper Collection

**Sources:**
- AllCryptoWhitepapers.com
- Project official documentation
- GitHub repositories

**Universe:** Top 8 cryptocurrencies by market cap with available whitepapers:
- BTC, ETH, SOL, AVAX, DOT, FIL, LINK, ALGO

### 1.2 Text Extraction

```python
# PyMuPDF-based extraction
import fitz  # PyMuPDF

def extract_whitepaper(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
```

### 1.3 Semantic Chunking

- Split documents into semantic chunks (512 tokens max)
- Preserve paragraph boundaries
- Remove headers, footers, page numbers

### 1.4 Zero-Shot Classification

**Model:** `facebook/bart-large-mnli`

**Taxonomy:**
```python
FUNCTIONAL_CATEGORIES = [
    "store_of_value",
    "medium_of_exchange",
    "smart_contracts",
    "infrastructure",
    "privacy",
    "governance",
    "data_oracle",
    "identity",
    "gaming_metaverse",
    "stablecoin"
]
```

**Classification:**
```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

def classify_chunk(chunk):
    result = classifier(chunk, FUNCTIONAL_CATEGORIES,
                       multi_label=True)
    return dict(zip(result['labels'], result['scores']))
```

### 1.5 Profile Aggregation

For each entity, aggregate chunk-level classifications:

```python
def aggregate_profile(chunk_results):
    profile = {cat: 0.0 for cat in FUNCTIONAL_CATEGORIES}
    for chunk in chunk_results:
        for cat, score in chunk.items():
            if score > 0.3:  # Threshold
                profile[cat] += score
    # Normalize
    total = sum(profile.values())
    return {k: v/total for k, v in profile.items()}
```

---

## Stage 2: Market Tensor Construction

### 2.1 Data Collection

**Source:** Binance via CCXT

**Specification:**
- Duration: 1 year (Oct 2024 - Oct 2025)
- Granularity: Hourly OHLCV
- Assets: 8 cryptocurrencies matching NLP universe

### 2.2 Tensor Construction

3-way tensor 𝓧 ∈ ℝ^(T × A × F):
- T = 8,761 hourly timestamps
- A = 8 assets
- F = 5 features (OHLCV)

**Normalization:** Z-score per asset-feature pair

### 2.3 CP Decomposition

```python
from tensorly.decomposition import parafac

# Rank selection via explained variance elbow
result = parafac(tensor, rank=4, random_state=42)

# Extract entity loadings
entity_factors = result.factors[1]  # Shape: (8, 4)
```

**Rank Selection:** Elbow method on explained variance curve; rank-4 optimal.

---

## Stage 3: Alignment Testing

### 3.1 Matrix Construction

**Claims Matrix X:** (n_entities × n_functions) = (8 × 10)
- Each row = entity's functional profile vector

**Market Matrix Y:** (n_entities × n_factors) = (8 × 4)
- Each row = entity's factor loadings from CP decomposition

### 3.2 Procrustes Rotation

Align claims space to market space:

```python
from scipy.spatial import procrustes

# Standardize
X_std = (X - X.mean(0)) / X.std(0)
Y_std = (Y - Y.mean(0)) / Y.std(0)

# Procrustes rotation
mtx1, mtx2, disparity = procrustes(Y_std, X_std)
```

### 3.3 Tucker's Congruence Coefficient

Per-factor congruence:

```python
def tuckers_phi(x, y):
    """Compute Tucker's congruence coefficient."""
    return np.sum(x * y) / np.sqrt(np.sum(x**2) * np.sum(y**2))

# Overall congruence
phi_values = [tuckers_phi(X_aligned[:, i], Y[:, i])
              for i in range(n_factors)]
phi_overall = np.mean(phi_values)
```

### 3.4 Bootstrap Confidence Intervals

```python
def bootstrap_congruence(X, Y, n_bootstrap=50):
    phi_samples = []
    n = len(X)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        phi = compute_alignment(X[idx], Y[idx])
        phi_samples.append(phi)
    return np.percentile(phi_samples, [2.5, 97.5])
```

---

## Interpretation Framework

### Congruence Thresholds

| Range | Interpretation |
|-------|----------------|
| φ ≥ 0.95 | Factors equivalent |
| 0.85 ≤ φ < 0.95 | Fair similarity |
| 0.65 ≤ φ < 0.85 | Some similarity |
| φ < 0.65 | Factors distinct |

### Economic Interpretation

- **High alignment (φ ≥ 0.85):** Markets price assets according to claimed functionality
- **Moderate alignment (0.65-0.85):** Partial correspondence; other factors dominate
- **Low alignment (φ < 0.65):** Whitepaper claims diverge from market reality

---

## Limitations

1. **Small sample size:** 8 entities limits statistical power
2. **Temporal mismatch:** Whitepapers static vs. market dynamics evolving
3. **Zero-shot accuracy:** Classification may miss domain-specific nuance
4. **Single venue:** Binance only; cross-exchange dynamics unexplored

## Extensions

1. **Temporal analysis:** Rolling window alignment to detect regime changes
2. **Granger causality:** Do claims *lead* market positioning?
3. **Expanded universe:** 50+ entities for robust inference
4. **Fine-tuned NLP:** Domain-specific models (CryptoBERT)
