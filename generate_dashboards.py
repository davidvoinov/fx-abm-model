"""
generate_dashboards.py — Run simulation and save dashboard PNGs to output/.
"""
import matplotlib
matplotlib.use('Agg')

from AgentBasedModel.simulator.simulator import Simulator
from AgentBasedModel.visualization.dashboards import (
    generate_all_dashboards,
    save_all_individual_plots,
)

# ── Run simulation WITH AMM ─────────────────────────────────────────
print('Running simulation WITH AMM ...')
sim = Simulator.default_fx(
    n_noise=10,
    n_fx_takers=15,
    n_fx_fund=5,
    n_retail=10,
    n_institutional=3,
    cpmm_reserves=500,
    hfmm_reserves=500,
    stress_start=200,
    stress_end=350,
    price=100,
    beta=1.0,
)
sim.simulate(500, silent=False)

# ── Run simulation WITHOUT AMM (counterfactual) ─────────────────────
print('Running simulation WITHOUT AMM ...')
sim_no = Simulator.default_fx_no_amm(
    n_noise=10,
    n_fx_takers=15,
    n_fx_fund=5,
    n_retail=10,
    n_institutional=3,
    stress_start=200,
    stress_end=350,
    price=100,
    beta=1.0,
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
