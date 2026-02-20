"""
main_fx.py — Entry point for the multi-venue FX ABM simulation.

Two hypotheses:

    H1: Execution cost comparison CLOB vs AMM.
        AMM is structurally more expensive but adds overall liquidity.
    H2: Systemic linkage under stress.
        Costs co-move; CLOB spread widens; AMM becomes an alternative
        liquidity source where volume migrates during stress.
"""

import math
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
    plot_clob_spread,
)


def run_simulation(n_iter: int = 500, silent: bool = False, **kwargs):
    """Build and run the default FX simulation."""
    sim = Simulator.default_fx(
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


def print_summary(sim: Simulator):
    """Print key metrics summary organised by hypothesis."""
    logger = sim.logger
    summary = logger.summary()

    print("\n" + "=" * 65)
    print("  FX ABM SIMULATION SUMMARY")
    print("=" * 65)
    print(f"  Iterations:   {summary.get('n_iterations', 0)}")
    print(f"  Total trades: {summary.get('n_trades', 0)}")

    # ── H1 ────────────────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("  H1: EXECUTION COST COMPARISON  &  AMM ADDS LIQUIDITY")
    print("-" * 65)

    Q_values = [1, 2, 5, 10, 20, 50]
    venues = ['clob'] + list(logger.amm_cost_curves.keys())

    print(f"\n  Average All-in Cost (bps):")
    print(f"  {'Q':>6s}", end='')
    for v in venues:
        print(f"  {v.upper():>8s}", end='')
    print()
    for Q in Q_values:
        print(f"  {Q:>6.0f}", end='')
        for v in venues:
            val = summary.get(f'avg_cost_{v}_Q{Q}', float('nan'))
            if math.isfinite(val):
                print(f"  {val:>8.1f}", end='')
            else:
                print(f"  {'N/A':>8s}", end='')
        print()

    print(f"\n  Average Flow Share:")
    for v in venues:
        val = summary.get(f'avg_flow_share_{v}', 0)
        print(f"    {v.upper():>6s}: {val:.1%}")
    amm_share = sum(summary.get(f'avg_flow_share_{v}', 0) for v in venues if v != 'clob')
    print(f"\n  ▹ AMM captures {amm_share:.1%} of total volume — adds liquidity layer.")

    # ── H2 ────────────────────────────────────────────────────────────
    stress_start = sim.env.stress_start
    print("\n" + "-" * 65)
    print("  H2: SYSTEMIC LINKAGE UNDER STRESS")
    print("-" * 65)

    print(f"\n  Cost Correlation (CLOB ↔ AMM, Q=5):")
    for name in logger.amm_cost_curves:
        rho = logger.cost_correlation('clob', name, Q=5)
        print(f"    CLOB ↔ {name.upper()}: ρ = {rho:.3f}")

    if stress_start is not None:
        print(f"\n  Correlation Before / After Stress (t={stress_start}):")
        for name in logger.amm_cost_curves:
            ba = logger.commonality_before_after('clob', name, Q=5,
                                                  stress_start=stress_start)
            b_s = f"{ba['before']:.3f}" if math.isfinite(ba['before']) else "N/A"
            a_s = f"{ba['after']:.3f}" if math.isfinite(ba['after']) else "N/A"
            print(f"    CLOB ↔ {name.upper()}: before={b_s}, after={a_s}")

        # Flow migration
        n = len(logger.iterations)
        idx = min(stress_start, n)
        print(f"\n  AMM Volume Share — Normal vs Stress:")
        for name in logger.amm_cost_curves:
            fs = logger.flow_share(name)
            before = fs[:idx]
            after = fs[idx:]
            avg_b = sum(before) / len(before) if before else 0
            avg_a = sum(after) / len(after) if after else 0
            delta = avg_a - avg_b
            print(f"    {name.upper()}: Normal={avg_b:.1%}  Stress={avg_a:.1%}  Δ={delta:+.1%}")

        # CLOB spread widening
        qspr = logger.clob_qspr
        normal_s = [s for s in qspr[:idx] if math.isfinite(s)]
        stress_s = [s for s in qspr[idx:] if math.isfinite(s)]
        avg_n = sum(normal_s) / len(normal_s) if normal_s else 0
        avg_st = sum(stress_s) / len(stress_s) if stress_s else 0
        print(f"\n  CLOB Quoted Spread:")
        print(f"    Normal: {avg_n:.1f} bps")
        if avg_n > 0:
            print(f"    Stress: {avg_st:.1f} bps  ({avg_st/avg_n:.1f}× wider)")

    print("=" * 65 + "\n")


def generate_all_plots(sim: Simulator):
    """Generate all diagnostic plots organised by hypothesis."""
    logger = sim.logger
    stress_start = sim.env.stress_start or 200

    # Context
    plot_environment(logger)
    plot_fx_price(logger, rolling=5)
    plot_clob_spread(logger, rolling=5)

    # H1: Cost comparison & AMM adds liquidity
    plot_execution_cost_curves(logger)
    plot_cost_decomposition(logger, Q=5)
    plot_total_market_depth(logger, rolling=10)

    # H2: Systemic linkage & liquidity flight
    plot_cost_timeseries(logger, Q=5, rolling=10)
    plot_flow_allocation(logger, rolling=10)
    plot_clob_spread_vs_amm_cost(logger, Q=5, rolling=10)
    plot_commonality(logger, Q=5, window=30)
    plot_amm_liquidity(logger)
    plot_stress_flow_migration(logger, stress_start=stress_start)


# ---- main ----------------------------------------------------------------

if __name__ == '__main__':
    sim = run_simulation(n_iter=500, silent=False)
    print_summary(sim)
    generate_all_plots(sim)
