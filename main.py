"""
main.py — Interactive demo of the multi-venue FX ABM.

Run:
    python3 main.py                          # default scenario
    python3 main.py --preset heavy_amm       # 60% AMM flow
    python3 main.py --preset clob_only       # no AMM pools
    python3 main.py --amm-share 30 --shock-iter 350 --shock-pct -20

All configurable parameters are exposed as CLI flags.
Use  python3 main.py --help  for the full list.
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

from AgentBasedModel.simulator.simulator import Simulator
from AgentBasedModel.visualization.dashboards import (
    generate_all_dashboards,
    save_all_individual_plots,
)


PRESETS = {
    "default": dict(),
    "clob_only": dict(
        enable_amm=0,
        amm_share_pct=0,
    ),
    "amm_only": dict(
        enable_clob_mm=0,
        clob_liq=0.1,
        amm_share_pct=100,
        amm_liq=3.0,
    ),
    "heavy_amm": dict(
        amm_share_pct=60,
        amm_liq=2.0,
    ),
    "heavy_clob": dict(
        amm_share_pct=10,
        clob_liq=2.0,
    ),
    "stress_test": dict(
        stress_start=150,
        stress_end=400,
        sigma_low=0.01,
        sigma_high=0.08,
        c_low=0.001,
        c_high=0.04,
    ),
    "shock_only": dict(
        stress_start=-1,
        stress_end=-1,
        shock_iter=250,
        shock_pct=-20.0,
    ),
    "low_liquidity": dict(
        clob_liq=0.3,
        amm_liq=0.3,
        clob_volume=300,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-venue FX Agent-Based Model — interactive demo",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    g = p.add_argument_group("General")
    g.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                   help="Named parameter bundle (overridden by explicit flags).\n"
                        "Available: " + ", ".join(PRESETS.keys()))
    g.add_argument("--n-iter", type=int, default=1000,
                   help="Number of simulation iterations (default: 1000)")
    g.add_argument("--price", type=float, default=100.0,
                   help="Initial FX mid-price (default: 100)")
    g.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    g.add_argument("--silent", action="store_true",
                   help="Suppress progress bar")
    g.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation")
    g.add_argument("--no-summary", action="store_true",
                   help="Skip text summary")

    g = p.add_argument_group(
        "CLOB agents",
        "Traders that populate the central limit order book.\n"
        "  Noise traders place random limit/market/cancel orders.\n"
        "  Market Maker quotes both sides with σ/c-dependent spread."
    )
    g.add_argument("--n-noise", type=int, default=30,
                   help="CLOB noise/liquidity traders (default: 30)")
    g.add_argument("--enable-clob-mm", type=int, choices=[0, 1], default=1,
                   help="Enable CLOB Market Maker: 1=yes, 0=no (default: 1)")
    g.add_argument("--clob-std", type=float, default=2.0,
                   help="Std of initial order-book price distribution (default: 2.0)")
    g.add_argument("--clob-volume", type=int, default=1000,
                   help="Initial number of orders in book (default: 1000)")

    g = p.add_argument_group(
        "FX liquidity takers",
        "Agents that route orders between CLOB and AMM pools.\n"
        "  Noise     — random direction, random size [q_min, q_max].\n"
        "  Fundament — trades when |mid − fair_rate| > γ.\n"
        "  Retail    — small orders, high frequency.\n"
        "  Institut. — large orders, low frequency."
    )
    g.add_argument("--n-fx-takers", type=int, default=15,
                   help="Noise takers with venue routing (default: 15)")
    g.add_argument("--n-fx-fund", type=int, default=5,
                   help="Fundamentalist traders (default: 5)")
    g.add_argument("--n-retail", type=int, default=10,
                   help="Retail noise traders (default: 10)")
    g.add_argument("--n-institutional", type=int, default=3,
                   help="Institutional traders (default: 3)")

    g = p.add_argument_group(
        "Flow allocation  (CLOB ↔ AMM split)",
        "Step 1 — AMM vs CLOB: coin flip with P(AMM) = amm-share/100.\n"
        "Step 2 — within AMM, CPMM vs HFMM: logit softmax with β_AMM,\n"
        "         adjusted by CPMM bias and cost noise."
    )
    g.add_argument("--amm-share", type=float, default=25.0,
                   dest="amm_share_pct",
                   help="Target AMM flow share in %% (0–100, default: 25)")
    g.add_argument("--deterministic", action="store_true",
                   help="Use deterministic argmin venue choice (no softmax)")
    g.add_argument("--beta-amm", type=float, default=0.05,
                   help="Logit sensitivity for CPMM vs HFMM (default: 0.05)")
    g.add_argument("--cpmm-bias", type=float, default=5.0,
                   dest="cpmm_bias_bps",
                   help="CPMM non-monetary cost discount in bps (default: 5)")
    g.add_argument("--cost-noise", type=float, default=1.5,
                   dest="cost_noise_std",
                   help="Std of cost estimation noise in bps (default: 1.5)")

    g = p.add_argument_group(
        "Liquidity levels",
        "Global multipliers that scale depth on each side.\n"
        "  clob-liq × → n_noise, clob_volume, MM depth.\n"
        "  amm-liq  × → CPMM / HFMM reserves."
    )
    g.add_argument("--clob-liq", type=float, default=1.0,
                   help="CLOB liquidity multiplier (default: 1.0)")
    g.add_argument("--amm-liq", type=float, default=1.0,
                   help="AMM liquidity multiplier (default: 1.0)")
    g.add_argument("--enable-amm", type=int, choices=[0, 1], default=1,
                   help="Enable AMM pools: 1=yes, 0=no (default: 1)")

    g = p.add_argument_group(
        "AMM pool parameters",
        "Configure CPMM (Uniswap-like) and HFMM (Curve-like) pools.\n"
        "  CPMM: x·y = k,  fee = cpmm-fee.\n"
        "  HFMM: StableSwap invariant,  fee = hfmm-fee, amplification = A."
    )
    g.add_argument("--cpmm-reserves", type=float, default=1500.0,
                   help="CPMM base-currency reserves (default: 1500)")
    g.add_argument("--hfmm-reserves", type=float, default=1500.0,
                   help="HFMM base-currency reserves (default: 1500)")
    g.add_argument("--cpmm-fee", type=float, default=0.003,
                   help="CPMM swap fee as fraction (default: 0.003 = 30 bps)")
    g.add_argument("--hfmm-fee", type=float, default=0.001,
                   help="HFMM swap fee as fraction (default: 0.001 = 10 bps)")
    g.add_argument("--hfmm-A", type=float, default=10.0,
                   help="HFMM amplification coefficient A (default: 10)")

    g = p.add_argument_group(
        "Stress regime",
        "Gradual volatility / funding-cost ramp over [stress-start, stress-end].\n"
        "  σ: sigma-low → sigma-high\n"
        "  c: c-low     → c-high\n"
        "Set --stress-start -1 to disable."
    )
    g.add_argument("--stress-start", type=int, default=-1,
                   help="Iteration when stress begins (default: -1 = off)")
    g.add_argument("--stress-end", type=int, default=-1,
                   help="Iteration when stress ends (default: -1 = off)")
    g.add_argument("--sigma-low", type=float, default=0.015,
                   help="Volatility in normal regime (default: 0.015)")
    g.add_argument("--sigma-high", type=float, default=0.025,
                   help="Volatility in stress regime (default: 0.025)")
    g.add_argument("--c-low", type=float, default=0.003,
                   help="Funding cost in normal regime (default: 0.003)")
    g.add_argument("--c-high", type=float, default=0.008,
                   help="Funding cost in stress regime (default: 0.008)")

    g = p.add_argument_group(
        "Exogenous price shock",
        "One-time instantaneous shift of all CLOB order prices.\n"
        "  e.g. --shock-iter 250 --shock-pct -20  → −20 %% crash at t=250."
    )
    g.add_argument("--shock-iter", type=int, default=None,
                   help="Iteration of exogenous price shock (default: off)")
    g.add_argument("--shock-pct", type=float, default=-20.0,
                   help="Shock magnitude in %% (default: -20)")

    return p


def build_sim(args: argparse.Namespace) -> Simulator:
    stress_start = args.stress_start if args.stress_start >= 0 else None
    stress_end = args.stress_end if stress_start is not None else None

    return Simulator.default_fx(
        n_noise=args.n_noise,
        n_fx_takers=args.n_fx_takers,
        n_fx_fund=args.n_fx_fund,
        n_retail=args.n_retail,
        n_institutional=args.n_institutional,
        clob_std=args.clob_std,
        clob_volume=args.clob_volume,
        enable_clob_mm=bool(args.enable_clob_mm),
        clob_liq=args.clob_liq,
        enable_amm=bool(args.enable_amm),
        amm_liq=args.amm_liq,
        cpmm_reserves=args.cpmm_reserves,
        hfmm_reserves=args.hfmm_reserves,
        cpmm_fee=args.cpmm_fee,
        hfmm_fee=args.hfmm_fee,
        hfmm_A=args.hfmm_A,
        amm_share_pct=args.amm_share_pct,
        deterministic=args.deterministic,
        beta_amm=args.beta_amm,
        cpmm_bias_bps=args.cpmm_bias_bps,
        cost_noise_std=args.cost_noise_std,
        price=args.price,
        stress_start=stress_start,
        stress_end=stress_end,
        sigma_low=args.sigma_low,
        sigma_high=args.sigma_high,
        c_low=args.c_low,
        c_high=args.c_high,
        shock_iter=args.shock_iter,
        shock_pct=args.shock_pct,
    )


def print_config(args: argparse.Namespace):
    W = 65
    print("\n" + "=" * W)
    print("  FX ABM — SIMULATION CONFIGURATION")
    print("=" * W)

    def row(label, value, unit=""):
        print(f"    {label:<32s} {str(value):>12s} {unit}")

    print("\n  General")
    row("Iterations", str(args.n_iter))
    row("Initial price", f"{args.price:.1f}")
    row("Random seed", str(args.seed) if args.seed else "random")
    if args.preset:
        row("Preset applied", args.preset)

    shadow = args.enable_amm and not args.enable_clob_mm
    print("\n  CLOB agents")
    row("Noise traders", str(args.n_noise))
    row("Market Maker", "ON" if args.enable_clob_mm else "OFF")
    row("CLOB mode", "Shadow (synthetic)" if shadow else "Live order-book")
    row("Order-book volume", str(args.clob_volume))
    row("Price std", f"{args.clob_std:.1f}")
    row("CLOB liquidity multiplier", f"{args.clob_liq:.1f}", "×")

    print("\n  FX liquidity takers")
    row("Noise takers", str(args.n_fx_takers))
    row("Fundamentalists", str(args.n_fx_fund))
    row("Retail", str(args.n_retail))
    row("Institutional", str(args.n_institutional))
    total = args.n_fx_takers + args.n_fx_fund + args.n_retail + args.n_institutional
    row("Total takers", str(total))

    print("\n  Flow allocation")
    row("AMM share target", f"{args.amm_share_pct:.0f}", "%")
    row("CLOB share target", f"{100 - args.amm_share_pct:.0f}", "%")
    row("Venue choice mode", "argmin" if args.deterministic else "softmax")
    row("beta_AMM (intra-AMM)", f"{args.beta_amm:.3f}")
    row("CPMM bias", f"{args.cpmm_bias_bps:.1f}", "bps")
    row("Cost noise sigma", f"{args.cost_noise_std:.1f}", "bps")

    print("\n  AMM pools")
    row("AMM enabled", "YES" if args.enable_amm else "NO")
    if args.enable_amm:
        row("CPMM reserves (base)", f"{args.cpmm_reserves:.0f}")
        row("HFMM reserves (base)", f"{args.hfmm_reserves:.0f}")
        row("CPMM fee", f"{args.cpmm_fee * 10_000:.0f}", "bps")
        row("HFMM fee", f"{args.hfmm_fee * 10_000:.0f}", "bps")
        row("HFMM amplification A", f"{args.hfmm_A:.0f}")
        row("AMM liquidity multiplier", f"{args.amm_liq:.1f}", "×")

    print("\n  Stress regime")
    if args.stress_start >= 0:
        row("Window", f"[{args.stress_start}, {args.stress_end}]")
        row("Volatility sigma", f"{args.sigma_low} -> {args.sigma_high}")
        row("Funding cost c", f"{args.c_low} -> {args.c_high}")
    else:
        row("Status", "OFF")

    print("\n  Exogenous price shock")
    if args.shock_iter is not None:
        row("Iteration", str(args.shock_iter))
        row("Magnitude", f"{args.shock_pct:+.0f}", "%")
    else:
        row("Status", "OFF")

    print("=" * W + "\n")


def print_summary(sim: Simulator):
    logger = sim.logger
    summary = logger.summary()

    W = 65
    print("\n" + "=" * W)
    print("  FX ABM — SIMULATION RESULTS")
    print("=" * W)
    print(f"  Iterations:   {summary.get('n_iterations', 0)}")
    print(f"  Total trades: {summary.get('n_trades', 0)}")

    print("\n" + "-" * W)
    print("  H1: EXECUTION COST COMPARISON  &  AMM ADDS LIQUIDITY")
    print("-" * W)

    Q_values = [1, 2, 5, 10, 20, 50]
    venues = ['clob'] + list(logger.amm_cost_curves.keys())

    print("\n  Average All-in Cost (bps):")
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

    print("\n  Average Flow Share:")
    for v in venues:
        val = summary.get(f'avg_flow_share_{v}', 0)
        print(f"    {v.upper():>6s}: {val:.1%}")
    amm_share = sum(summary.get(f'avg_flow_share_{v}', 0)
                    for v in venues if v != 'clob')
    print(f"\n  -> AMM captures {amm_share:.1%} of total volume.")

    stress_start = sim.env.stress_start if sim.env else None
    print("\n" + "-" * W)
    print("  H2: SYSTEMIC LINKAGE UNDER STRESS")
    print("-" * W)

    if not logger.amm_cost_curves:
        print("  (no AMM pools — skipped)")
        print("=" * W + "\n")
        return

    print("\n  Cost Correlation (CLOB <-> AMM, Q=5):")
    for name in logger.amm_cost_curves:
        rho = logger.cost_correlation('clob', name, Q=5)
        print(f"    CLOB <-> {name.upper()}: rho = {rho:.3f}")

    if stress_start is not None:
        print(f"\n  Correlation Before / After Stress (t={stress_start}):")
        for name in logger.amm_cost_curves:
            ba = logger.commonality_before_after('clob', name, Q=5,
                                                 stress_start=stress_start)
            b_s = f"{ba['before']:.3f}" if math.isfinite(ba['before']) else "N/A"
            a_s = f"{ba['after']:.3f}" if math.isfinite(ba['after']) else "N/A"
            print(f"    CLOB <-> {name.upper()}: before={b_s}, after={a_s}")

        n = len(logger.iterations)
        idx = min(stress_start, n)
        print("\n  AMM Volume Share — Normal vs Stress:")
        for name in logger.amm_cost_curves:
            fs = logger.flow_share(name)
            before = fs[:idx]
            after = fs[idx:]
            avg_b = sum(before) / len(before) if before else 0
            avg_a = sum(after) / len(after) if after else 0
            delta = avg_a - avg_b
            print(f"    {name.upper()}: Normal={avg_b:.1%}  "
                  f"Stress={avg_a:.1%}  Delta={delta:+.1%}")

        qspr = logger.clob_qspr
        normal_s = [s for s in qspr[:idx] if math.isfinite(s)]
        stress_s = [s for s in qspr[idx:] if math.isfinite(s)]
        avg_n = sum(normal_s) / len(normal_s) if normal_s else 0
        avg_st = sum(stress_s) / len(stress_s) if stress_s else 0
        print("\n  CLOB Quoted Spread:")
        print(f"    Normal: {avg_n:.1f} bps")
        if avg_n > 0:
            print(f"    Stress: {avg_st:.1f} bps  ({avg_st/avg_n:.1f}x wider)")

    # ---- Recovery time after shock ------------------------------------
    shock_iter = sim.shock_iter if hasattr(sim, 'shock_iter') else None
    if shock_iter is None:
        shock_iter = getattr(sim, '_shock_iter', None)

    if shock_iter is not None and logger.trade_log:
        print("\n" + "-" * W)
        print("  H3: POST-SHOCK RECOVERY TIME")
        print("-" * W)

        # Build trade DataFrame — use cost_bps directly (vs mid-price)
        trades = pd.DataFrame(logger.trade_log)
        # cost_bps is available in all trade records; fv_cost requires
        # exec_price which only stress_sweep captures.  cost_bps recovery
        # measures when execution quality returns to pre-shock levels.

        WINDOW = 20
        THRESHOLD = 20.0
        WARMUP = 50

        def _recovery(df, shock, window=WINDOW, threshold=THRESHOLD):
            fin = df[np.isfinite(df['cost_bps'])].copy()
            pre = fin[(fin['t'] >= WARMUP) & (fin['t'] < shock)]
            if len(pre) < window:
                return float('nan'), float('nan')
            baseline = pre['cost_bps'].median()
            post = fin[fin['t'] >= shock]
            if len(post) == 0:
                return float('nan'), baseline
            per_t = post.groupby('t')['cost_bps'].median().sort_index()
            roll = per_t.rolling(window, min_periods=window).median()
            within = (roll - baseline).abs() <= threshold
            recovered = within[within]
            if len(recovered) == 0:
                return float('inf'), baseline
            return max(0, recovered.index[0] - shock), baseline

        def _fmt(v):
            if v != v:
                return 'N/A'
            if v == float('inf'):
                return 'never'
            return f'{v:.0f}'

        rt_all, bl_all = _recovery(trades, shock_iter)

        clob_trades = trades[trades['venue'] == 'clob']
        amm_trades = trades[trades['venue'] != 'clob']

        rt_clob, bl_clob = _recovery(clob_trades, shock_iter) if len(clob_trades) else (float('nan'), float('nan'))
        rt_amm, bl_amm = _recovery(amm_trades, shock_iter) if len(amm_trades) else (float('nan'), float('nan'))

        print(f"\n  Metric: periods after t={shock_iter} until rolling({WINDOW})")
        print(f"          median fv_cost returns within ±{THRESHOLD:.0f} bps of baseline.")

        print(f"\n  {'Venue':<12s}  {'Baseline':>10s}  {'Recovery':>10s}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*10}")

        def _bl_fmt(v):
            return f'{v:.1f}' if np.isfinite(v) else 'N/A'

        print(f"  {'Combined':<12s}  {_bl_fmt(bl_all):>10s}  {_fmt(rt_all):>10s}")
        print(f"  {'CLOB':<12s}  {_bl_fmt(bl_clob):>10s}  {_fmt(rt_clob):>10s}")
        print(f"  {'AMM':<12s}  {_bl_fmt(bl_amm):>10s}  {_fmt(rt_amm):>10s}")

        # Per AMM pool recovery
        amm_pool_names = [v for v in trades['venue'].unique() if v != 'clob']
        if len(amm_pool_names) > 1:
            print()
            for pname in sorted(amm_pool_names):
                psub = trades[trades['venue'] == pname]
                rt_p, bl_p = _recovery(psub, shock_iter) if len(psub) else (float('nan'), float('nan'))
                print(f"  {'  ' + pname.upper():<12s}  {_bl_fmt(bl_p):>10s}  {_fmt(rt_p):>10s}")

    print("=" * W + "\n")


def generate_all_plots(sim: Simulator):
    import glob as _glob
    logger = sim.logger
    stress_start = (sim.env.stress_start or 200) if sim.env else 200

    # Remove stale plots from previous runs.
    out_dir = 'output/main'
    for old_png in _glob.glob(os.path.join(out_dir, '*.png')):
        os.remove(old_png)

    # Save dashboards + all individual plots to output/main/ (non-interactive).
    generate_all_dashboards(
        logger,
        out_dir='output/main',
        stress_start=stress_start,
        Q=5,
        rolling=10,
        logger_no_amm=None,
    )
    save_all_individual_plots(
        logger,
        out_dir='output/main',
        stress_start=stress_start,
        Q=5,
        rolling=10,
        logger_no_amm=None,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.preset:
        preset_vals = PRESETS[args.preset]
        for k, v in preset_vals.items():
            cli_key = k.replace("-", "_")
            if cli_key in vars(args):
                default_val = parser.get_default(cli_key)
                current_val = getattr(args, cli_key)
                if current_val == default_val:
                    setattr(args, cli_key, v)

    if args.seed is not None:
        import random as _rnd
        _rnd.seed(args.seed)

    print_config(args)

    sim = build_sim(args)
    sim.simulate(args.n_iter, silent=args.silent)

    if not args.no_summary:
        print_summary(sim)

    if not args.no_plots:
        generate_all_plots(sim)


if __name__ == '__main__':
    main()