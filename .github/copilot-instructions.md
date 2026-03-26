# Copilot Instructions — FX Agent-Based Model

## Project Overview
Multi-venue FX Agent-Based Model simulating market microstructure with heterogeneous traders routing orders across CLOB and AMM venues (CPMM + HFMM). Research focus: venue competition, liquidity dynamics, arbitrage, stress testing.

## Architecture
```
AgentBasedModel/
├── agents/agents.py        # Trader types: Noise, Fundamentalist, Retail, Institutional, MarketMaker, AMMProvider, AMMArbitrageur
├── venues/clob.py          # CLOBVenue (analytics over ExchangeAgent book), ShadowCLOB
├── venues/amm.py           # CPMMPool (x·y=k), HFMMPool (StableSwap)
├── environment/processes.py # MarketEnvironment: σ_t, c_t, S_t (GBM), regime switching
├── events/events.py        # Price/Liquidity/Information shocks, MM in/out
├── states/states.py        # Regime detection: trend, panic, disaster, mean-reversion, stable
├── metrics/logger.py       # Per-period metrics snapshots
├── simulator/simulator.py  # Orchestrator: classic and multi-venue modes
├── utils/math.py           # Custom math (exp, mean, std, rolling, difference)
├── utils/orders.py         # Order, OrderList for CLOB
└── visualization/          # Dashboards, market/trader/venue plots
```

## Entry Points
- `main.py` — CLI with presets (default, clob_only, amm_only, heavy_amm, heavy_clob, stress_test, shock_only, low_liquidity). 50+ params.
- `generate_dashboards.py` — Interactive dashboards from output
- `generate_shock.py` — Shock scenario analysis

## Simulation Loop (multi-venue mode, each iteration)
1. Price shock (if scheduled)
2. Environment step → σ_t, c_t
3. Pre-trade arbitrage (AMM→fair price)
4. CLOB MM quotes
5. CLOB noise traders refresh book
6. Dividends
7. AMM fees reset
8. FX takers route → CLOB or AMM
9. Post-trade arbitrage
10. LP providers adjust liquidity
11. AMM pools record state
12. Metrics snapshot

## Key Conventions
- Costs in **basis points (bps)**: cost_bps = slippage_bps + fee_bps + gas_cost_bps
- Trade size θ = Q/R: small ≤1%, medium ≤5%, large >5%
- Agent IDs: auto-incrementing class `id` variable
- Venue routing: Step 1 coin flip (amm_share_pct), Step 2 logit softmax CPMM vs HFMM
- Environment time: `step()` increments internal time before regime check, so stress_start=N activates on iteration N-1
- Python 3 with venv at `.venv/`

## Tech Stack
tqdm, matplotlib, pandas, numpy, statsmodels, scipy, seaborn

## Testing
- Unit: `tests/test_clob_unit.py`, `test_cpmm_unit.py`, `test_hfmm_unit.py`, `test_routing_unit.py`
- Stress: `tests/stress_sweep.py` (parametric sweeps)
- Regression: `tests/visualize_unit_tests.py`
- Output: `output/unit_tests/`, `output/stress/`, `output/shock/`, `output/fennell/`

## Language & Communication
- The developer (David Voinov) communicates in Russian. Respond in Russian when asked in Russian.
- Be concise and technical. Domain expertise: market microstructure, DeFi, quantitative finance.
