"""
generate_shock.py — Run CLOB-only & AMM-only sims with a −20 % price shock,
then produce the comparison dashboard + individual plots.

Usage:
    python generate_shock.py
"""
import matplotlib
matplotlib.use('Agg')

from AgentBasedModel.simulator.simulator import Simulator
from AgentBasedModel.visualization.shock_dashboard import (
    dashboard_shock_comparison,
    save_shock_individual_plots,
    plot_price_recovery,
)

# ── Shared parameters ────────────────────────────────────────────────
N_ITER = 500
SHOCK_ITER = 200        # iteration at which the −20 % shock hits
SHOCK_PCT = -20.0       # −20 %

COMMON = dict(
    n_noise=30,
    n_fx_takers=15,
    n_fx_fund=5,
    n_retail=10,
    n_institutional=3,
    clob_volume=3000,
    stress_start=None,   # no MarketEnvironment stress — only the price shock
    stress_end=None,
    price=100,
    amm_share_pct=25.0,
    shock_iter=SHOCK_ITER,
    shock_pct=SHOCK_PCT,
)

# ── 1. CLOB-Only ─────────────────────────────────────────────────────
print(f'Running CLOB-Only simulation ({N_ITER} iterations, shock at {SHOCK_ITER}) ...')
sim_clob = Simulator.default_fx(
    **COMMON,
    enable_clob_mm=True,
    enable_amm=False,
    clob_liq=3.0,          # 3× deeper book to survive −20 % shock
)
sim_clob.simulate(N_ITER, silent=False)

# ── 2. AMM-Only ──────────────────────────────────────────────────────
print(f'Running AMM-Only simulation ({N_ITER} iterations, shock at {SHOCK_ITER}) ...')
sim_amm = Simulator.default_fx(
    **COMMON,
    cpmm_reserves=500,
    hfmm_reserves=500,
    enable_clob_mm=False,
    enable_amm=True,
    clob_liq=0.1,        # thin CLOB (reference price only)
    amm_liq=1.0,
)
sim_amm.simulate(N_ITER, silent=False)

# ── Generate ─────────────────────────────────────────────────────────
out = 'output/shock'
dash = dashboard_shock_comparison(
    sim_clob.logger, sim_amm.logger,
    shock_iter=SHOCK_ITER, out_dir=out, rolling=5, Q=5,
)
indiv = save_shock_individual_plots(
    sim_clob.logger, sim_amm.logger,
    shock_iter=SHOCK_ITER, out_dir=out, rolling=5, Q=5,
)
recovery = plot_price_recovery(
    sim_clob.logger, sim_amm.logger,
    shock_iter=SHOCK_ITER, out_dir=out, rolling=5,
)

print()
print(f'  ✓ {dash}')
for p in indiv:
    print(f'  ✓ {p}')
print(f'  ✓ {recovery}')
print(f'\nDone — 2 dashboards + {len(indiv)} individual plots → {out}/')
