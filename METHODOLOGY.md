# Methodology

**Tensor Decomposition for DeFi Market Microstructure**

## Overview

This document details the technical methodology for constructing and decomposing market microstructure tensors. The framework implements CP, Tucker, and Tensor Train decomposition methods with comprehensive baseline comparisons.

## Data Collection Pipeline

### Centralized Exchanges (CEX)

**Implementation**: `src/data_pipeline/cex_collector.py`

**API**: CCXT library (wrapper for 120+ exchange APIs)

**Methodology**:
1. Connect to Binance API (no authentication required for public OHLCV data)
2. Request hourly candles with chunked pagination (500-hour batches)
3. Fetch order book snapshots (top 20 levels)
4. Compute microstructure features:
   - Bid-ask spread
   - Volume imbalance (bid vs ask volume)
   - Liquidity depth (cumulative volume at 1% price impact)
   - Price impact (slippage for market orders)

**Chunking Strategy** (for large date ranges):
```python
# Binance API limits: 500 candles per request
# For 365 days (8,760 hours): need 18 chunks

since = start_timestamp
chunks = []
while since < end_timestamp:
    chunk = exchange.fetch_ohlcv(
        symbol,
        timeframe='1h',
        since=since,
        limit=500
    )
    chunks.append(chunk)
    since = chunk[-1][0] + 3600000  # Next hour
```

**Error Handling**:
- Exponential backoff for rate limits (max 3 retries)
- Data validation: detect duplicates, gaps, OHLC inconsistencies
- Completeness check: actual rows vs expected rows

### Decentralized Exchanges (DEX)

**Implementation**: `src/data_pipeline/dex_collector.py`

**API**: The Graph (GraphQL queries to Uniswap/Curve subgraphs)

**Status**: Currently disabled due to The Graph hosted service deprecation
- Migration to decentralized Graph Network in progress
- Requires API key (free tier available)
- Future work: Implement alternative via Dune Analytics or direct on-chain queries

## Tensor Construction

### Tensor Builder

**Implementation**: `src/tensor_ops/tensor_builder.py`

**Input**: Flat DataFrame with columns `[datetime, symbol, exchange, open, high, low, close, volume]`

**Output**: 4D NumPy array `𝓧 ∈ ℝ^(T × V × A × F)`

### Construction Algorithm

```python
# 1. Index creation
timestamps = sorted(df['datetime'].unique())  # T timesteps
venues = sorted(df['exchange'].unique())      # V venues
assets = sorted(df['symbol'].unique())        # A assets
features = ['open', 'high', 'low', 'close', 'volume']  # F features

# 2. Tensor initialization
tensor = np.zeros((len(timestamps), len(venues), len(assets), len(features)))

# 3. Fill tensor (vectorized operation)
for t, timestamp in enumerate(timestamps):
    for v, venue in enumerate(venues):
        for a, asset in enumerate(assets):
            row = df[(df['datetime'] == timestamp) &
                     (df['exchange'] == venue) &
                     (df['symbol'] == asset)]
            if not row.empty:
                tensor[t, v, a, :] = row[features].values

# 4. Missing data handling
# Forward fill: propagate last known value
# Alternative: mean imputation, interpolation
tensor = forward_fill(tensor, axis=0)
```

### Tensor Modes

**1. Microstructure Tensor** (Current implementation):
```
Shape: (T, V, A, F)
- T: Time (hourly timestamps)
- V: Venue (Binance, Coinbase, etc.)
- A: Asset (BTC/USDT, ETH/USDT, SOL/USDT)
- F: Feature (open, high, low, close, volume)
```

**2. Order Book Tensor** (Future work):
```
Shape: (T, V, L, S, F)
- L: Price level (1-20)
- S: Side (bid, ask)
- F: Feature (price, size, num_orders)
```

**3. Log Returns Tensor** (For stationarity):
```
Shape: (T, V, A, F)
Features: [log_return, high_low_range, log_volume]
```

### Normalization

Three variants created for robustness:

**1. Raw OHLCV** (`raw_ohlcv_tensor.pkl`):
- No normalization
- Absolute price levels preserved
- Use for: Price-focused analysis

**2. Z-Score Normalized** (`normalized_ohlcv_tensor.pkl`):
- Per-asset, per-feature normalization
- `x_norm = (x - μ) / σ`
- Use for: Cross-asset comparison

**3. Log Returns** (`log_returns_tensor.pkl`):
- Stationary time series
- Features: log returns, intraday range, log volume
- Use for: Returns-based modeling, forecasting

## Decomposition Methods

### CP Decomposition (CANDECOMP/PARAFAC)

**Implementation**: `src/tensor_ops/decomposition.py` → `cp_decomposition()`

**Method**: Alternating Least Squares (ALS)

**Mathematical Formulation**:
```
𝓧 ≈ Σᵣ λᵣ · (aᵣ ⊗ bᵣ ⊗ cᵣ ⊗ dᵣ)

where:
  λᵣ = component weight
  aᵣ = temporal factor (T × 1)
  bᵣ = venue factor (V × 1)
  cᵣ = asset factor (A × 1)
  dᵣ = feature factor (F × 1)
  ⊗ = outer product
```

**Algorithm** (CP-ALS):
```python
# Initialize factors randomly
A, B, C, D = random_init(T, V, A, F, rank)

for iteration in range(max_iter):
    # Update each factor matrix (fixing others)
    A = update_factor(X, B, C, D)  # Least squares solution
    B = update_factor(X, A, C, D)
    C = update_factor(X, A, B, D)
    D = update_factor(X, A, B, C)

    # Check convergence
    reconstruction = reconstruct_tensor(A, B, C, D)
    error = frobenius_norm(X - reconstruction)
    if error_change < tolerance:
        break
```

**Advantages**:
- Most compact representation
- Fully symmetric (all modes treated equally)
- Economically interpretable factors
- Reveals core latent structure

**Use Case**: Identifying fundamental market drivers

### Tucker Decomposition

**Implementation**: `src/tensor_ops/decomposition.py` → `tucker_decomposition()`

**Method**: Higher-Order Orthogonal Iteration (HOOI)

**Mathematical Formulation**:
```
𝓧 ≈ 𝓖 ×₁ A ×₂ B ×₃ C ×₄ D

where:
  𝓖 = core tensor (R₁ × R₂ × R₃ × R₄)
  A = temporal factors (T × R₁)
  B = venue factors (V × R₂)
  C = asset factors (A × R₃)
  D = feature factors (F × R₄)
  ×ₙ = mode-n product
```

**Algorithm** (Tucker-HOOI):
```python
# Initialize via HOSVD
A, B, C, D = hosvd(X, ranks)
G = compute_core(X, A, B, C, D)

for iteration in range(max_iter):
    # Update each factor (orthogonal)
    X_unfolded = unfold(X, mode=1)
    U, S, V = svd(X_unfolded @ khatri_rao(B, C, D))
    A = U[:, :rank_1]

    # Repeat for B, C, D
    # ...

    # Update core tensor
    G = X ×₁ A.T ×₂ B.T ×₃ C.T ×₄ D.T

    if converged:
        break
```

**Advantages**:
- More flexible than CP (different ranks per mode)
- Core tensor captures factor interactions
- Often lower reconstruction error
- Handles asymmetric structure

**Use Case**: Modeling complex cross-venue effects

### Tensor Train (TT)

**Implementation**: `src/tensor_ops/decomposition.py` → `tensor_train_decomposition()`

**Mathematical Formulation**:
```
𝓧(i₁, i₂, i₃, i₄) = G₁(i₁) · G₂(i₂) · G₃(i₃) · G₄(i₄)

where:
  Gₖ = TT-core (rₖ₋₁ × iₖ × rₖ)
  rₖ = TT-rank at position k
```

**Advantages**:
- Memory-efficient (linear in number of modes)
- Scales to high-dimensional tensors
- Best for future extensions (adding MEV, gas, block position dimensions)

**Use Case**: High-dimensional market microstructure

## Rank Selection

**Method**: Cross-validation with explained variance threshold

```python
def rank_selection(tensor, max_rank=20, threshold=0.90):
    """
    Select optimal rank via explained variance

    Returns: rank where explained_variance >= threshold
    """
    for rank in range(1, max_rank + 1):
        result = cp_decomposition(tensor, rank=rank)
        if result.explained_variance >= threshold:
            return rank
    return max_rank
```

**Metrics**:
- **Explained Variance**: `1 - (||X - X̂||² / ||X||²)`
- **Reconstruction Error**: Frobenius norm `||X - X̂||`
- **Relative Error**: `||X - X̂|| / ||X||`

**Heuristics**:
- Financial tensors typically low-rank (3-10 factors)
- Use cross-validation to prevent overfitting
- Compare multiple ranks, choose elbow point

## Baseline Comparisons

### PCA (Principal Component Analysis)

**Implementation**: `src/baselines/traditional_methods.py` → `fit_pca()`

**Methodology**:
1. Flatten tensor to matrix: `X ∈ ℝ^(T × VAF)`
2. Apply PCA: extract top-k components
3. Reconstruct: `X̂ = scores @ components.T`
4. Compute explained variance

**Key Insight**:
- PCA assumes additivity (loses tensor structure)
- Tensor methods preserve multiplicative interactions
- Expected: Tensor methods outperform PCA by 20-50%

### Comparison Framework

```python
def compare_pca_vs_tensor(tensor, decomposition_result, n_components):
    """
    Direct comparison: PCA vs Tensor decomposition

    Returns:
        - PCA explained variance
        - Tensor explained variance
        - Improvement percentage
        - Per-asset error comparison
    """
    # PCA baseline
    X_flat = tensor.reshape(T, V*A*F)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_flat)
    pca_variance = pca.explained_variance_ratio_.sum()

    # Tensor result
    tensor_variance = decomposition_result.explained_variance

    improvement = (tensor_variance - pca_variance) / pca_variance

    return {
        'pca': pca_variance,
        'tensor': tensor_variance,
        'improvement_pct': improvement * 100
    }
```

## Validation

### Data Quality Checks

**1. Completeness**:
```python
expected_rows = lookback_days * 24 * num_symbols
actual_rows = len(df)
completeness = actual_rows / expected_rows
assert completeness >= 0.99  # Allow 1% missing
```

**2. Temporal Gaps**:
```python
df['time_diff'] = df.groupby('symbol')['datetime'].diff()
gaps = df[df['time_diff'] > pd.Timedelta(hours=1.1)]
assert len(gaps) == 0  # No gaps > 1 hour
```

**3. OHLC Consistency**:
```python
assert (df['high'] >= df['low']).all()
assert (df['high'] >= df['open']).all()
assert (df['high'] >= df['close']).all()
assert (df['low'] <= df['open']).all()
assert (df['low'] <= df['close']).all()
```

### Decomposition Validation

**1. Convergence Check**:
- Monitor error decrease across iterations
- Ensure error change < tolerance before stopping
- Flag non-convergent decompositions

**2. Reconstruction Accuracy**:
```python
reconstruction_error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)
assert reconstruction_error < 0.3  # < 30% error for rank >= 5
```

**3. Factor Interpretability**:
- Inspect temporal factors for regime transitions
- Check asset factors for correlation structure
- Validate feature factors match economic intuition

## Performance Characteristics

**Data Collection** (Full year, 3 assets):
- Time: ~30-60 minutes
- API calls: ~54 requests (18 chunks × 3 symbols)
- Rate limit: Built-in (CCXT `enableRateLimit: True`)

**Tensor Construction** (8761 × 1 × 3 × 5):
- Time: < 1 second
- Memory: 1 MB tensor + 1.18 MB pickle overhead

**CP Decomposition** (rank 10):
- Time: ~30-60 seconds (NumPy backend, CPU)
- Iterations: 50-100 typical
- Memory: < 10 MB

**Tucker Decomposition** (rank [10, 5, 5, 5]):
- Time: ~45-90 seconds
- More memory-intensive than CP (core tensor)

## Reproducibility

All experiments use:
- Fixed random seeds for initialization
- Versioned dependencies (`requirements.txt`)
- Documented hyperparameters
- Git-tracked code

**To reproduce results**:
```bash
git clone <repo>
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python collect_data.py  # 365-day collection
python scripts/build_tensor.py
python src/tensor_ops/decomposition.py
```

## Future Extensions

**Multi-Venue Collection**:
- Add Coinbase, Kraken (CEX)
- Fix The Graph integration (DEX)
- Cross-venue arbitrage detection

**Advanced Features**:
- Rolling window decomposition (adaptive ranks)
- Online CP (incremental updates)
- Manifold geometry (Ricci curvature)

**GPU Acceleration**:
- Switch to PyTorch backend for TensorLy
- Expected speedup: 10-50× for large tensors

---

**Updated**: October 2025 | **Status**: Phase 1 Complete
