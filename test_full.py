import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from AgentBasedModel.simulator.fx_simulator import FXSimulator
from AgentBasedModel.metrics.logger import MetricsLogger

sim = FXSimulator.default(
    n_noise=10, n_fx_takers=15, n_fx_fund=5,
    cpmm_reserves=500, hfmm_reserves=500,
    stress_start=200, stress_end=350,
    price=100, beta=1.0
)
sim.simulate(500, silent=False)

from AgentBasedModel.visualization.venue_plots import (
    plot_execution_cost_curves, plot_cost_timeseries,
    plot_cost_decomposition, plot_flow_allocation,
    plot_amm_liquidity, plot_amm_reserves,
    plot_volume_slippage_profile, plot_environment,
    plot_clob_spread, plot_commonality, plot_fx_price,
)
logger = sim.logger

plot_execution_cost_curves(logger)
plot_cost_timeseries(logger, Q=5, rolling=10)
plot_cost_decomposition(logger, Q=5)
plot_flow_allocation(logger, rolling=10)
plot_amm_liquidity(logger)
plot_amm_reserves(logger, 'cpmm')
plot_volume_slippage_profile(logger, 'cpmm')
plot_environment(logger)
plot_clob_spread(logger, rolling=5)
plot_commonality(logger, Q=5, window=30)
plot_fx_price(logger, rolling=5)
plt.close('all')

summary = logger.summary()
print('\n=== KEY RESULTS ===')
print(f'Iterations: {summary["n_iterations"]}')
print(f'Trades: {summary["n_trades"]}')

for Q in [1, 5, 20]:
    cc = summary.get(f'avg_cost_clob_Q{Q}', 0)
    cp = summary.get(f'avg_cost_cpmm_Q{Q}', 0)
    ch = summary.get(f'avg_cost_hfmm_Q{Q}', 0)
    print(f'Q={Q}: CLOB={cc:.1f}, CPMM={cp:.1f}, HFMM={ch:.1f} bps')

for name in ['cpmm', 'hfmm']:
    rho = logger.cost_correlation('clob', name, Q=5)
    ba = logger.commonality_before_after('clob', name, Q=5, stress_start=200)
    print(f'CLOB-{name}: rho={rho:.3f}, before={ba["before"]:.3f}, after={ba["after"]:.3f}')

print('\nAll plots & metrics OK!')
