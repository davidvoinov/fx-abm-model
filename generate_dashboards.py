"""
generate_dashboards.py — Run simulation and save dashboard PNGs to output/.

Demonstrates venue configuration:
    enable_clob_mm  – toggle CLOB market maker
    enable_amm      – toggle AMM pools
    clob_liq        – scale CLOB liquidity  (book depth, noise, MM)
    amm_liq         – scale AMM liquidity   (reserves)
"""
import matplotlib
matplotlib.use('Agg')

from AgentBasedModel.simulator.simulator import Simulator
from AgentBasedModel.visualization.dashboards import (
    generate_all_dashboards,
    save_all_individual_plots,
)

# ── Shared agent parameters ──────────────────────────────────────────
COMMON = dict(
    n_noise=10,
    n_fx_takers=15,
    n_fx_fund=5,
    n_retail=10,
    n_institutional=3,
    stress_start=200,
    stress_end=350,
    price=100,
    amm_share_pct=25.0,
)

# ── Run simulation WITH AMM (mixed market, default proportions) ──────
print('Running simulation WITH AMM (mixed) ...')
sim = Simulator.default_fx(
    **COMMON,
    cpmm_reserves=500,
    hfmm_reserves=500,
    enable_clob_mm=True,
    enable_amm=True,
    clob_liq=1.0,
    amm_liq=1.0,
)
sim.simulate(500, silent=False)

# ── Run simulation WITHOUT AMM (CLOB-only counterfactual) ───────────
print('Running simulation WITHOUT AMM (CLOB-only) ...')
sim_no = Simulator.default_fx(
    **COMMON,
    enable_clob_mm=True,
    enable_amm=False,
)
sim_no.simulate(500, silent=False)

# ── Generate dashboards ─────────────────────────────────────────────
paths = generate_all_dashboards(
    sim.logger,
    out_dir='output',
    stress_start=200,
    Q=5,
    rolling=10,
    logger_no_amm=sim_no.logger,
)

# ── Save individual plots ───────────────────────────────────────────
indiv = save_all_individual_plots(
    sim.logger,
    out_dir='output',
    stress_start=200,
    Q=5,
    rolling=10,
    logger_no_amm=sim_no.logger,
)

print()
for p in paths + indiv:
    print(f'  ✓ {p}')
print(f'\nDone — {len(paths)} dashboards + {len(indiv)} individual plots saved to output/')
