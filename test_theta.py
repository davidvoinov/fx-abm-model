"""Test θ-normalization and bin-based metrics."""
from AgentBasedModel.simulator.fx_simulator import FXSimulator
import math

sim = FXSimulator.default(
    n_noise=10, n_fx_takers=10, n_fx_fund=5,
    n_retail=10, n_institutional=3,
    cpmm_reserves=500, hfmm_reserves=500,
    stress_start=200, stress_end=350,
    price=100, beta=1.0
)
sim.simulate(300, silent=False)

logger = sim.logger
summary = logger.summary()
print()
print('=== KEY RESULTS ===')
print(f'Iterations: {summary["n_iterations"]}')
print(f'Trades: {summary["n_trades"]}')
for Q in [1, 5, 20]:
    cc = summary.get(f'avg_cost_clob_Q{Q}', 0)
    cp = summary.get(f'avg_cost_cpmm_Q{Q}', 0)
    ch = summary.get(f'avg_cost_hfmm_Q{Q}', 0)
    print(f'Q={Q}: CLOB={cc:.1f}, CPMM={cp:.1f}, HFMM={ch:.1f} bps')

# theta-bin summary
bs = logger.bin_summary()
print()
print('=== THETA-BIN SUMMARY ===')
for bucket in ('small', 'medium', 'large'):
    print(f'  [{bucket}]')
    for venue, stats in bs.get(bucket, {}).items():
        avg = stats['avg_cost_bps']
        vol = stats['total_volume']
        avg_s = f'{avg:.1f}' if math.isfinite(avg) else 'n/a'
        print(f'    {venue}: avg_cost={avg_s} bps, volume={vol:.0f}')

# Show sample trades with theta
sample = [t for t in logger.trade_log if 'theta' in t]
print()
print('=== SAMPLE TRADES ===')
for t in sample[:8]:
    print(f'  {t["trader_type"]:18s} Q={t["quantity"]:3d}  venue={t["venue"]:5s}  '
          f'theta={t["theta"]:.4f}  bucket={t["size_bucket"]:7s}  cost={t["cost_bps"]:.1f} bps')

print()
print('OK!')
