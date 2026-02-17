"""
main_fx.py — Entry point for the multi-venue FX ABM simulation.

Runs a dual-venue (CLOB + CPMM + HFMM) simulation with stress scenario
and produces all diagnostic plots needed for the coursework hypotheses:

    H1: AMM is structurally more expensive (higher C_v)
    H2: AMM adds overall liquidity but no cost parity
    H3: Systemic linkage — costs co-move across venues, especially in stress
    H4: Non-monotonic AMM liquidity response to volatility
"""

from AgentBasedModel.simulator.fx_simulator import FXSimulator
from AgentBasedModel.visualization.venue_plots import (
    plot_execution_cost_curves,
    plot_cost_timeseries,
    plot_cost_decomposition,
    plot_flow_allocation,
    plot_amm_liquidity,
    plot_amm_reserves,
    plot_volume_slippage_profile,
    plot_environment,
    plot_clob_spread,
    plot_commonality,
    plot_fx_price,
)
import json


def run_simulation(n_iter: int = 500, silent: bool = False, **kwargs):
    """Build and run the default FX simulation."""
    sim = FXSimulator.default(
        n_noise=10,
        n_fx_takers=15,
        n_fx_fund=5,
        cpmm_reserves=500.0,
        hfmm_reserves=500.0,
        hfmm_A=100.0,
        cpmm_fee=0.003,       # 30 bps
        hfmm_fee=0.001,       # 10 bps
        stress_start=200,
        stress_end=350,
        sigma_low=0.01,
        sigma_high=0.05,
        c_low=0.001,
        c_high=0.02,
        price=100.0,
        beta=1.0,
        deterministic=False,
        **kwargs,
    )
    sim.simulate(n_iter, silent=silent)
    return sim


def print_summary(sim: FXSimulator):
    """Print key metrics summary."""
    logger = sim.logger
    summary = logger.summary()

    print("\n" + "=" * 60)
    print("  FX ABM SIMULATION SUMMARY")
    print("=" * 60)
    print(f"  Iterations:  {summary.get('n_iterations', 0)}")
    print(f"  Total trades: {summary.get('n_trades', 0)}")
    print()

    # Cost comparison table
    Q_values = [1, 5, 10, 20]
    venues = ['clob'] + list(logger.amm_cost_curves.keys())
    print("  Average All-in Cost (bps):")
    print(f"  {'Q':>6s}", end='')
    for v in venues:
        print(f"  {v.upper():>8s}", end='')
    print()
    for Q in Q_values:
        print(f"  {Q:>6.0f}", end='')
        for v in venues:
            key = f'avg_cost_{v}_Q{Q}'
            val = summary.get(key, float('nan'))
            if val != val:  # nan
                print(f"  {'N/A':>8s}", end='')
            else:
                print(f"  {val:>8.1f}", end='')
        print()

    print()

    # Flow shares
    print("  Average Flow Share:")
    for v in venues:
        key = f'avg_flow_share_{v}'
        val = summary.get(key, 0)
        print(f"    {v.upper():>6s}: {val:.1%}")

    # Commonality
    print()
    print("  Cost Correlation (CLOB ↔ AMM, Q=5):")
    for name in logger.amm_cost_curves:
        rho = logger.cost_correlation('clob', name, Q=5)
        print(f"    CLOB ↔ {name.upper()}: ρ = {rho:.3f}")

    # Before/after stress
    stress_start = sim.env.stress_start
    if stress_start is not None:
        print()
        print(f"  Commonality Before/After Stress (t={stress_start}):")
        for name in logger.amm_cost_curves:
            ba = logger.commonality_before_after('clob', name, Q=5,
                                                  stress_start=stress_start)
            b = ba['before']
            a = ba['after']
            b_str = f"{b:.3f}" if b == b else "N/A"
            a_str = f"{a:.3f}" if a == a else "N/A"
            print(f"    CLOB ↔ {name.upper()}: before={b_str}, after={a_str}")

    print("=" * 60 + "\n")


def generate_all_plots(sim: FXSimulator):
    """Generate all diagnostic & hypothesis-testing plots."""
    logger = sim.logger

    # 1. Environment
    plot_environment(logger)

    # 2. FX price
    plot_fx_price(logger, rolling=5)

    # 3. CLOB spread
    plot_clob_spread(logger, rolling=5)

    # 4. Execution cost curves — H1: AMM structurally more expensive
    plot_execution_cost_curves(logger)

    # 5. Cost decomposition (slippage vs fee)
    plot_cost_decomposition(logger, Q=5)

    # 6. Cost time series — observe regime effects
    plot_cost_timeseries(logger, Q=5, rolling=10)

    # 7. AMM liquidity — H4: non-monotonic response to σ
    plot_amm_liquidity(logger)

    # 8. AMM reserves
    plot_amm_reserves(logger, 'cpmm')
    plot_amm_reserves(logger, 'hfmm')

    # 9. Volume-slippage profile — H2: AMM liquidity metric
    plot_volume_slippage_profile(logger, 'cpmm')
    plot_volume_slippage_profile(logger, 'hfmm')

    # 10. Flow allocation — where volume goes
    plot_flow_allocation(logger, rolling=10)

    # 11. Commonality — H3: systemic linkage
    plot_commonality(logger, Q=5, window=30)


# ---- main ----------------------------------------------------------------

if __name__ == '__main__':
    sim = run_simulation(n_iter=500, silent=False)
    print_summary(sim)
    generate_all_plots(sim)
