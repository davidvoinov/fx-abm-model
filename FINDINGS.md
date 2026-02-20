# Multi-Venue FX ABM — Simulation Findings

## Simulation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_iter` | 500 | Total simulation iterations |
| `price` | 100.0 | Initial mid-price (FX rate) |
| `clob_volume` | 1000 | Initial CLOB order book depth (orders) |
| `clob_std` | 2.0 | Std. deviation of initial book prices |
| `cpmm_reserves` | 500.0 | CPMM base reserves (x₀ = 500, y₀ = 50,000) |
| `hfmm_reserves` | 500.0 | HFMM base reserves (x₀ = 500, y₀ = 50,000) |
| `hfmm_A` | 100.0 | StableSwap amplification factor |
| `cpmm_fee` | 0.003 (30 bps) | CPMM LP fee |
| `hfmm_fee` | 0.001 (10 bps) | HFMM LP fee |
| `stress_start` | 200 | Stress regime onset (iteration) |
| `stress_end` | 350 | Stress regime end (iteration) |
| `σ_low` | 0.01 | Normal-regime volatility |
| `σ_high` | 0.05 | Stress-regime volatility |
| `c_low` | 0.001 | Normal-regime funding cost |
| `c_high` | 0.02 | Stress-regime funding cost |
| `β` | 1.0 | Venue-choice logit temperature |
| `tick_precision` | 0.01 | Order book tick size |

### Agent Population

| Agent Type | Count | Q range | Description |
|-----------|-------|---------|-------------|
| CLOB Noise | 10 | market orders | Keep the CLOB book alive |
| FX Noise Takers | 15 | 1–5 | Routed liquidity takers (prob=0.3) |
| FX Fundamentalists | 5 | 1–10 | Mean-reversion traders (γ=5e-3) |
| Retail | 10 | 1–2 | Small traders (prob=0.4) |
| Institutional | 3 | 15–50 | Large traders (prob=0.15) |
| Market Maker | 1 | — | σ/c-dependent spread (Brunnermeier–Pedersen) |
| AMM LP Providers | 2 | — | CPMM and HFMM liquidity adjusters |
| Arbitrageur | 1 | — | AMM ↔ CLOB price alignment |

### Market Maker (Brunnermeier–Pedersen) Coefficients

Quoted spread (bps): $s_t = \alpha_0 + \alpha_1 \sigma_t + \alpha_2 c_t$

| Coefficient | Value | Normal-regime contribution | Stress-regime contribution |
|------------|-------|---------------------------|---------------------------|
| α₀ | 3.0 | 3.0 bps (intercept) | 3.0 bps |
| α₁ | 300.0 | 3.0 bps (σ=0.01) | 15.0 bps (σ=0.05) |
| α₂ | 200.0 | 0.2 bps (c=0.001) | 4.0 bps (c=0.02) |
| **Total** | | **6.2 bps** | **22.0 bps** |

Depth: $d_t = d_0 - d_1 \sigma_t - d_2 c_t$  (floored at $d_{\min}=5$)

---

## Hypothesis 1: Execution Cost Comparison — CLOB vs AMM

### Research Question
Is AMM structurally more expensive than CLOB for trade execution, and does AMM add overall market liquidity despite higher costs?

### Results

**Average All-in Cost $C_v(Q)$ in bps:**

| Q | CLOB | CPMM | HFMM |
|---|------|------|------|
| 1 | 6.3 | 40.3 | 9.2 |
| 2 | 6.5 | 60.5 | 9.3 |
| 5 | 6.8 | 121.4 | 9.6 |
| 10 | 7.3 | 224.6 | 10.2 |
| 20 | 8.4 | 437.6 | 11.3 |
| 50 | 13.4 | 1133.3 | 14.8 |

**Average Flow Share:**

| Venue | Share |
|-------|-------|
| CLOB | 78.6% |
| HFMM | 21.4% |
| CPMM | ≈ 0% |

### Analysis

1. **CLOB is cheapest at all trade sizes.** CLOB cost ranges from 6.3 bps (Q=1) to 13.4 bps (Q=50), reflecting the spread plus price impact from walking the order book. The cost scaling is sub-linear — the deep order book absorbs moderate sizes efficiently.

2. **CPMM (x·y=k) is prohibitively expensive.** At Q=5, CPMM costs 121 bps — 18× more than CLOB. The constant-product invariant generates massive price impact, making CPMM non-competitive. It captures essentially 0% of flow.

3. **HFMM (StableSwap, A=100) is competitive.** With a 10 bps LP fee and flattened bonding curve near peg, HFMM costs only 9.2–14.8 bps — within 1.5× of CLOB. This makes it a viable alternative, and it captures 21.4% of volume.

4. **AMM adds a distinct liquidity layer.** Despite higher costs, the HFMM provides a permanent, executable venue. Unlike the CLOB where depth depends on active market-making and can evaporate, AMM liquidity is protocol-guaranteed by pool reserves. The 21.4% flow share confirms that traders rationally use AMM as a secondary liquidity source when CLOB costs rise.

5. **Cost decomposition (Q=5):** CLOB cost is entirely spread-based (no explicit fee). HFMM cost splits into ~5 bps slippage + ~5 bps fee. CPMM cost is dominated by slippage (~115 bps) with only ~6 bps fee.

### θ-Bin Analysis (Trade-Size Buckets)

| Bucket | CLOB avg cost | HFMM avg cost | CLOB volume | HFMM volume |
|--------|--------------|---------------|-------------|-------------|
| Small (θ<0.05) | 4.7 bps | 10.2 bps | 2,929 | 798 |
| Medium (0.05≤θ<0.2) | 3.9 bps | 10.9 bps | 1,290 | 319 |
| Large (θ≥0.2) | 5.0 bps | 14.2 bps | 2,872 | 965 |

CLOB maintains a cost advantage across all size buckets. HFMM usage increases for large trades (volume=965) — consistent with institutional traders seeking guaranteed execution when CLOB depth is uncertain.

---

## Hypothesis 2: Systemic Linkage Under Stress

### Research Question
During a stress event (σ₅× increase, c₂₀× increase), do execution costs co-move across venues? Does CLOB liquidity deteriorate, and does AMM become an alternative liquidity source absorbing flow from CLOB?

### Results

**Cost Correlation (CLOB ↔ AMM, Q=5):**

| Pair | Overall ρ | Before stress ρ | After stress ρ |
|------|----------|-----------------|----------------|
| CLOB ↔ CPMM | 0.222 | 0.553 | 0.323 |
| CLOB ↔ HFMM | 0.302 | 0.615 | 0.286 |

**CLOB Quoted Spread:**

| Regime | Avg Spread (bps) | Multiplier |
|--------|-----------------|------------|
| Normal (t<200) | 7.7 | 1.0× |
| Stress (t≥200) | 16.8 | 2.2× |

**AMM Volume Share — Normal vs Stress:**

| Venue | Normal | Stress | Δ |
|-------|--------|--------|---|
| HFMM | 2.8% | 33.8% | +31.0% |
| CPMM | 0.0% | 0.0% | — |

### Analysis

1. **Costs co-move across venues.** The positive correlation (ρ = 0.22–0.30 overall) confirms systemic linkage: when CLOB costs rise, AMM costs rise too. This is expected since both venues price the same underlying asset. The before-stress correlation is notably higher (0.55–0.62), reflecting stronger co-movement in a stable environment where price discovery dominates.

2. **CLOB liquidity deteriorates under stress.** The CLOB quoted spread widens from 7.7 to 16.8 bps (2.2× increase) during the stress regime. This is driven by the MarketMaker's σ/c-dependent spread function: higher volatility and funding costs force the MM to quote wider.

3. **AMM becomes an alternative liquidity destination.** This is the key finding: **HFMM's share of total volume jumps from 2.8% (normal) to 33.8% (stress) — a +31 percentage point increase.** When CLOB spreads widen, the logit venue-routing mechanism redirects flow to the AMM, which maintains relatively stable execution costs due to its protocol-based pricing.

4. **HFMM is the refuge, not CPMM.** Only the StableSwap pool (HFMM) absorbs flight-to-AMM volume. CPMM's constant-product pricing is too expensive even during stress. The StableSwap's flat bonding curve near peg (A=100) provides cost stability that attracts flow when CLOB deteriorates.

5. **Correlation structure shifts during stress.** The before/after split (e.g., CLOB↔HFMM: 0.615 → 0.286) shows that correlation *decreases* during and after stress. This may reflect divergent dynamics: CLOB costs spike on spread widening while AMM costs are primarily driven by pool composition changes and arbitrage pressure, which operate on a different timescale.

6. **AMM as crisis liquidity.** The results support the hypothesis that AMM venues act as a "safety valve" during periods of CLOB illiquidity. While AMM execution is more expensive in absolute terms, it provides guaranteed execution when CLOB depth evaporates — a market microstructure function analogous to "liquidity of last resort."

---

## Visualizations Produced

### H1 Plots
- **Execution Cost Curves** — $C_v(Q)$ per venue (confirms CLOB < HFMM << CPMM)
- **Cost Decomposition** — Slippage vs Fee stacked bar (Q=5)
- **Total Market Depth** — CLOB + AMM combined depth over time (stacked area)

### H2 Plots
- **Cost Time Series** — $C_v(Q=5)$ per venue with stress shading (10-iter MA)
- **Flow Allocation** — Stacked area of volume share with visible stress migration
- **CLOB Spread vs AMM Cost** — Dual-axis overlay highlighting divergence
- **Rolling Correlation** — 30-iter rolling Pearson ρ (CLOB ↔ AMM)
- **AMM Liquidity** — $L_t$ pool measure through stress
- **Stress Flow Migration** — Bar chart comparing AMM share (Normal vs Stress)

### Context Plots
- **Environment** — σ_t and c_t time series
- **FX Price** — CLOB mid-price (5-iter MA)

---

## Summary of Key Findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | CLOB is cheapest venue for all trade sizes | Cost 6.3–13.4 bps vs HFMM 9.2–14.8 bps |
| 2 | AMM adds liquidity layer (21.4% of flow) | HFMM captures significant volume despite higher cost |
| 3 | CPMM (x·y=k) is non-competitive | Cost 40–1133 bps, 0% flow share |
| 4 | Costs co-move (ρ > 0) across venues | Systemic linkage confirmed |
| 5 | CLOB spread widens 2.2× under stress | 7.7 → 16.8 bps |
| 6 | Volume migrates to AMM during stress | HFMM share: 2.8% → 33.8% (+31pp) |
| 7 | HFMM acts as crisis liquidity buffer | StableSwap absorbs CLOB overflow |
