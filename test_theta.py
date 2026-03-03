"""
test_theta.py — θ-normalization and bin-based metrics test.

Supplements H1 by showing that execution cost differences
(CLOB vs AMM) hold across trade-size buckets (small / medium / large).
"""
from AgentBasedModel.simulator.simulator import Simulator
import math

sim = Simulator.default_fx(
    n_noise=10, n_fx_takers=10, n_fx_fund=5,
    n_retail=10, n_institutional=3,
    cpmm_reserves=500, hfmm_reserves=500,
    stress_start=200, stress_end=350,
    price=100, amm_share_pct=25.0
)
sim.simulate(300, silent=False)

logger = sim.logger
summary = logger.summary()
print()
print('=' * 60)
print('  θ-BIN ANALYSIS (supplements H1)')
print('=' * 60)
print(f'  Iterations: {summary["n_iterations"]}')
print(f'  Trades:     {summary["n_trades"]}')

# Cost table by Q
print('\n  Average All-in Cost (bps):')
for Q in [1, 5, 20]:
    cc = summary.get(f'avg_cost_clob_Q{Q}', 0)
    cp = summary.get(f'avg_cost_cpmm_Q{Q}', 0)
    ch = summary.get(f'avg_cost_hfmm_Q{Q}', 0)
    print(f'    Q={Q:>3d}: CLOB={cc:.1f}, CPMM={cp:.1f}, HFMM={ch:.1f} bps')

# θ-bin summary
bs = logger.bin_summary()
print()
print('  θ-Bin Cost & Volume by Venue:')
for bucket in ('small', 'medium', 'large'):
    print(f'  [{bucket}]')
    for venue, stats in bs.get(bucket, {}).items():
        avg = stats['avg_cost_bps']
        vol = stats['total_volume']
        avg_s = f'{avg:.1f}' if math.isfinite(avg) else 'n/a'
        print(f'    {venue:>5s}: avg_cost={avg_s:>6s} bps, volume={vol:>8.0f}')

# Show sample trades with θ
sample = [t for t in logger.trade_log if 'theta' in t]
print()
print('  Sample Trades:')
for t in sample[:8]:
    print(f'    {t["trader_type"]:18s} Q={t["quantity"]:3d}  venue={t["venue"]:5s}  '
          f'θ={t["theta"]:.4f}  bucket={t["size_bucket"]:7s}  cost={t["cost_bps"]:.1f} bps')

print()
print('  OK!')
print('=' * 60)
