"""
generate_dashboards.py — Run simulation and save dashboard PNGs to output/.
"""
import matplotlib
matplotlib.use('Agg')

from AgentBasedModel.simulator.simulator import Simulator
from AgentBasedModel.visualization.dashboards import generate_all_dashboards

# ── Run simulation ───────────────────────────────────────────────────
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

# ── Generate dashboards ─────────────────────────────────────────────
paths = generate_all_dashboards(
    sim.logger,
    out_dir='output',
    stress_start=200,
    Q=5,
    rolling=10,
)

print()
for p in paths:
    print(f'  ✓ {p}')
print(f'\nDone — {len(paths)} dashboards saved to output/')
