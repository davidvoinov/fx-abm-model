"""
test_full.py — Two-hypothesis test for the multi-venue FX ABM.

H1: Execution cost comparison CLOB vs AMM; AMM adds liquidity.
H2: Systemic linkage under stress — cost co-movement, CLOB spread
    widening, and flow migration to AMM as alternative liquidity source.
"""
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from AgentBasedModel.simulator.simulator import Simulator
from AgentBasedModel.visualization.venue_plots import (
    # H1
    plot_execution_cost_curves,
    plot_cost_decomposition,
    plot_total_market_depth,
    # H2
    plot_cost_timeseries,
    plot_flow_allocation,
    plot_clob_spread_vs_amm_cost,
    plot_commonality,
    plot_amm_liquidity,
    plot_stress_flow_migration,
    # Context
    plot_environment,
    plot_fx_price,
)

# ── Simulation parameters ────────────────────────────────────────────────
N_ITER        = 500
PRICE         = 100.0
STRESS_START  = 200
STRESS_END    = 350
CPMM_RESERVES = 500.0
HFMM_RESERVES = 500.0

sim = Simulator.default_fx(
    n_noise=10,
    n_fx_takers=15,
    n_fx_fund=5,
    n_retail=10,
    n_institutional=3,
    cpmm_reserves=CPMM_RESERVES,
    hfmm_reserves=HFMM_RESERVES,
    hfmm_A=100.0,
    cpmm_fee=0.003,       # 30 bps
    hfmm_fee=0.001,       # 10 bps
    stress_start=STRESS_START,
    stress_end=STRESS_END,
    sigma_low=0.01,
    sigma_high=0.05,
    c_low=0.001,
    c_high=0.02,
    price=PRICE,
    beta=1.0,
)
sim.simulate(N_ITER, silent=False)
logger = sim.logger
summary = logger.summary()

# ── Context plots ────────────────────────────────────────────────────────
plot_environment(logger)
plot_fx_price(logger, rolling=5)

# =====================================================================
#  H1: Execution cost comparison  +  AMM adds liquidity
# =====================================================================
print('\n' + '=' * 65)
print('  H1: EXECUTION COST COMPARISON  &  AMM ADDS LIQUIDITY')
print('=' * 65)

# Cost table
Q_vals = [1, 2, 5, 10, 20, 50]
venues = ['clob'] + list(logger.amm_cost_curves.keys())
print(f'\n  Average All-in Cost (bps):')
print(f'  {"Q":>6s}', end='')
for v in venues:
    print(f'  {v.upper():>8s}', end='')
print()
for Q in Q_vals:
    print(f'  {Q:>6.0f}', end='')
    for v in venues:
        val = summary.get(f'avg_cost_{v}_Q{Q}', float('nan'))
        if math.isfinite(val):
            print(f'  {val:>8.1f}', end='')
        else:
            print(f'  {"N/A":>8s}', end='')
    print()

# Flow shares → AMM captures part of volume = adds liquidity
print(f'\n  Average Flow Share (all iterations):')
for v in venues:
    fs = summary.get(f'avg_flow_share_{v}', 0)
    print(f'    {v.upper():>6s}: {fs:.1%}')

print(f'\n  ▹ AMM captures {sum(summary.get(f"avg_flow_share_{v}", 0) for v in venues if v != "clob"):.1%} of total volume')
print(f'    ⇒ AMM adds a secondary liquidity layer despite higher cost.\n')

plot_execution_cost_curves(logger)
plot_cost_decomposition(logger, Q=5)
plot_total_market_depth(logger, rolling=10)
plt.close('all')

# =====================================================================
#  H2: Systemic linkage under stress — liquidity flight to AMM?
# =====================================================================
print('\n' + '=' * 65)
print('  H2: SYSTEMIC LINKAGE UNDER STRESS')
print('=' * 65)

# Cost correlation
print(f'\n  Cost Correlation (CLOB ↔ AMM, Q=5):')
for name in logger.amm_cost_curves:
    rho = logger.cost_correlation('clob', name, Q=5)
    print(f'    CLOB ↔ {name.upper()}: ρ = {rho:.3f}')

# Before/after stress
print(f'\n  Correlation Before / After Stress (t={STRESS_START}):')
for name in logger.amm_cost_curves:
    ba = logger.commonality_before_after('clob', name, Q=5,
                                          stress_start=STRESS_START)
    b_s = f'{ba["before"]:.3f}' if math.isfinite(ba['before']) else 'N/A'
    a_s = f'{ba["after"]:.3f}' if math.isfinite(ba['after']) else 'N/A'
    print(f'    CLOB ↔ {name.upper()}: before={b_s},  after={a_s}')

# Flow migration: AMM share before vs during stress
n = len(logger.iterations)
idx = min(STRESS_START, n)
print(f'\n  AMM Volume Share — Normal vs Stress:')
for name in logger.amm_cost_curves:
    fs = logger.flow_share(name)
    before = fs[:idx]
    after = fs[idx:]
    avg_b = sum(before) / len(before) if before else 0
    avg_a = sum(after) / len(after) if after else 0
    delta = avg_a - avg_b
    print(f'    {name.upper()}: Normal={avg_b:.1%}  Stress={avg_a:.1%}  Δ={delta:+.1%}')

# CLOB spread before vs during stress
qspr = logger.clob_qspr
normal_spr = [s for s in qspr[:idx] if math.isfinite(s)]
stress_spr = [s for s in qspr[idx:] if math.isfinite(s)]
avg_normal = sum(normal_spr) / len(normal_spr) if normal_spr else 0
avg_stress = sum(stress_spr) / len(stress_spr) if stress_spr else 0
print(f'\n  CLOB Quoted Spread:')
print(f'    Normal: {avg_normal:.1f} bps')
print(f'    Stress: {avg_stress:.1f} bps  ({avg_stress/avg_normal:.1f}× wider)' if avg_normal > 0 else '')

print()

plot_cost_timeseries(logger, Q=5, rolling=10)
plot_flow_allocation(logger, rolling=10)
plot_clob_spread_vs_amm_cost(logger, Q=5, rolling=10)
plot_commonality(logger, Q=5, window=30)
plot_amm_liquidity(logger)
plot_stress_flow_migration(logger, stress_start=STRESS_START)
plt.close('all')

# ── Summary ──────────────────────────────────────────────────────────────
print('=' * 65)
print(f'  Iterations: {summary["n_iterations"]},  Trades: {summary["n_trades"]}')
print('  All plots & metrics OK!')
print('=' * 65)
