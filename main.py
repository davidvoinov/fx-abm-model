"""
main.py — Interactive demo of the multi-venue FX ABM.

Base scenario (without any shocks):     
    python3 main.py --seed 42 --preset baseline   

Event simulations presets (with different shock types/mechanics):                                 
    python3 main.py --seed 42 --preset mm_withdrawal            
    python3 main.py --seed 42 --preset flash_crash              
    python3 main.py --seed 42 --preset dealer_liquidity_crisis  
    python3 main.py --seed 42 --preset funding_liquidity_shock   
    python3 main.py --seed 42 --preset high_vol_stress           

Shock scenarios:
    python3 main.py --seed 42 --shock-iter 350 --shock-mode realism --fundamental-shock-pct -12
    python3 main.py --seed 42 --shock-iter 350 --shock-mode realism --liquidity-shock-frac 0.45
    python3 main.py --seed 42 --shock-iter 350 --shock-mode realism --funding-vol-shock-intensity 1.0
    python3 main.py --seed 42 --shock-iter 350 --shock-mode realism --order-flow-shock-qty 250

Fixed liquidity share scenarios (no shocks, but different AMM/CLOB splits):
     python3 main.py --seed 42 --amm-share 30      

CLOB/AAMM only:
    python3 main.py --seed 42 --preset clob_only
    python3 main.py --seed 42 --preset amm_only              

Full list:
    python3 main.py --help                                       

All configurable parameters are exposed as CLI flags.
Use  python3 main.py --help  for the full list.
"""

import argparse
import math
import os
import sys
import statistics
from typing import Optional

import numpy as np
import pandas as pd

from AgentBasedModel.simulator.simulator import Simulator
from AgentBasedModel.visualization.dashboards import (
    generate_all_dashboards,
    save_all_individual_plots,
)


# ── Reproducibility: seed BOTH RNGs ─────────────────────────────
def _seed_all(seed: int):
    import random as _rnd
    _rnd.seed(seed)
    np.random.seed(seed)


LEGACY_PRESETS = {
    "default": dict(),
    "clob_only": dict(
        enable_amm=0,
        amm_share_pct=0,
    ),
    "amm_only": dict(
        n_mm=0,
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
        shock_iter=250,
        shock_pct=-20.0,
        # auto-stress kicks in unless --no-shock-stress
    ),
    "shock_stress": dict(
        shock_iter=250,
        shock_pct=-20.0,
        stress_start=230,
        stress_end=450,
        sigma_low=0.01,
        sigma_high=0.06,
        c_low=0.002,
        c_high=0.025,
    ),
    "low_liquidity": dict(
        clob_liq=0.3,
        amm_liq=0.3,
        clob_volume=300,
    ),
    "fx_calibrated": dict(
        sigma_low=0.01,
        sigma_high=0.04,
        c_low=0.002,
        c_high=0.015,
        stress_start=200,
        stress_end=350,
    ),
}


REALISM_PRESETS = {
    "baseline": dict(
        shock_mode="realism",
    ),
    "mm_withdrawal": dict(
        shock_iter=350,
        shock_mode="realism",
        # Narrow microstructure event: dealers pull quotes under one-sided flow,
        # but latent fair value does not fully re-anchor.
        # Keep the CLOB build identical to the default so cross-scenario
        # statistical tests differ by shock mechanics, not book composition.
        fundamental_shock_pct=0.0,
        order_flow_shock_qty=135.0,
        order_flow_shock_side="sell",
        liquidity_shock_frac=0.54,
        funding_vol_shock_intensity=0.35,
        arb_trade_fraction_cap=0.08,
        # Recovery is slower than the original soft preset, but materially
        # faster than the full dealer-liquidity-crisis scenario.
        reprice_prob_recovery=0.045,
        anchor_strength_recovery=0.022,
        bg_target_ratio_recovery=0.085,
        toxic_flow_decay=0.83,
        liquidity_shock_decay=0.87,
        mm_withdraw_threshold=1.70,
        mm_reentry_threshold=1.00,
        mm_loss_threshold_bps=19.0,
        mm_min_withdraw_ticks=4,
        mm_reentry_ticks=2,
    ),
    "flash_crash": dict(
        shock_iter=350,
        shock_mode="realism",
        # Pure microstructure dislocation: aggressive sell sweep + sudden
        # quote withdrawal, but without a large permanent fair-value shift.
        fundamental_shock_pct=0.0,
        order_flow_shock_qty=320.0,
        order_flow_shock_side="sell",
        liquidity_shock_frac=0.90,
        funding_vol_shock_intensity=0.55,
        # Flash crashes overshoot sharply, then partially mean-revert faster
        # than a dealer balance-sheet crisis once liquidity reappears.
        arb_trade_fraction_cap=0.12,
        reprice_prob_recovery=0.070,
        anchor_strength_recovery=0.040,
        bg_target_ratio_recovery=0.085,
        toxic_flow_decay=0.80,
        liquidity_shock_decay=0.84,
    ),
    "dealer_liquidity_crisis": dict(
        shock_iter=350,
        shock_mode="realism",
        # Fundamental dislocation: constrained dealers cannot absorb flow
        # → price discovery breaks (BIS WP 1073, Huang et al.)
        fundamental_shock_pct=-3.0,
        # Large directional sweep typical of institutional panic selling
        order_flow_shock_qty=250.0,
        order_flow_shock_side="sell",
        # Deep cancellation wave: dealers pull quotes (Mancini et al.)
        liquidity_shock_frac=0.80,
        # Volatility & funding spike (Bollerslev & Melvin)
        funding_vol_shock_intensity=0.80,
        # Restrict arbitrageur capacity during stress
        arb_trade_fraction_cap=0.05,
        # Slow recovery: dealer constraints persist (Lo & Hall resiliency)
        reprice_prob_recovery=0.015,       # ~37 ticks to restore (was 0.05 → 12 ticks)
        anchor_strength_recovery=0.008,    # ~28 ticks (was 0.03 → 8 ticks)
        bg_target_ratio_recovery=0.025,    # ~26 ticks (was 0.08 → 8 ticks)
        toxic_flow_decay=0.93,             # slower toxic flow fade (was 0.85)
        liquidity_shock_decay=0.95,        # slower systemic recovery (was 0.90)
    ),
    "funding_liquidity_shock": dict(
        shock_iter=350,
        shock_mode="realism",
        # Macro funding squeeze without a permanent fair-value repricing:
        # spreads widen and liquidity thins, but the latent anchor stays put.
        fundamental_shock_pct=0.0,
        order_flow_shock_qty=140.0,
        order_flow_shock_side="sell",
        liquidity_shock_frac=0.52,
        funding_vol_shock_intensity=1.00,
        arb_trade_fraction_cap=0.06,
        # Slower than a flash crash, faster than a dealer balance-sheet crisis.
        reprice_prob_recovery=0.040,
        anchor_strength_recovery=0.018,
        bg_target_ratio_recovery=0.060,
        toxic_flow_decay=0.90,
        liquidity_shock_decay=0.91,
    ),
    "high_vol_stress": dict(
        shock_mode="realism",
        stress_start=300,
        stress_end=500,
        sigma_low=0.01,
        sigma_high=0.08,
        c_low=0.002,
        c_high=0.020,
    ),
}


PRESETS = {
    **LEGACY_PRESETS,
    **REALISM_PRESETS,
}


def _preset_family(name: str) -> str:
    if name in REALISM_PRESETS:
        return 'realism'
    if name in LEGACY_PRESETS:
        return 'legacy'
    return 'custom'


def _format_preset_help() -> str:
    legacy = ', '.join(LEGACY_PRESETS.keys())
    realism = ', '.join(REALISM_PRESETS.keys())
    return (
        "Named parameter bundle (overridden by explicit flags).\n"
        f"Legacy/research presets: {legacy}\n"
        f"Realism presets: {realism}"
    )


# ── Auto-generate stress around shock (realism) ─────────────────
def _auto_stress_around_shock(args: argparse.Namespace):
    """
    If shock is set and the user explicitly asks for coupled
    regime-stress, auto-generate a realistic post-shock stress window.

    Pure shocks already trigger an endogenous microstructure aftermath
    inside MarketEnvironment.apply_shock().  This helper is only for a
    second, slower exogenous regime-stress layer.
    """
    if args.shock_iter is None:
        return
    if getattr(args, 'no_shock_stress', False):
        return
    if not getattr(args, 'shock_regime_stress', False):
        return
    if args.stress_start >= 0:
        return  # already configured

    shock_mag = abs(args.shock_pct)
    recovery_tail = max(100, int(shock_mag * 10))
    args.stress_start = max(0, args.shock_iter)
    args.stress_end = min(args.n_iter, args.shock_iter + recovery_tail)

    sigma_mult = max(3.0, 1.0 + shock_mag / 5.0)
    args.sigma_high = round(args.sigma_low * sigma_mult, 4)

    c_mult = max(2.0, 1.0 + shock_mag / 8.0)
    args.c_high = round(args.c_low * c_mult, 4)


def _shock_iter_from_sim(sim: Simulator):
    shock_iter = getattr(sim, 'shock_iter', None)
    if shock_iter is None:
        shock_iter = getattr(sim, '_shock_iter', None)
    return shock_iter


def _linkage_split_point(sim: Simulator):
    """Prefer explicit regime stress onset; otherwise split on the shock tick."""
    stress_start = getattr(sim.env, 'stress_start', None) if getattr(sim, 'env', None) else None
    if stress_start is not None and stress_start >= 0:
        return int(stress_start), 'Stress', 'Normal', 'Stress'

    shock_iter = _shock_iter_from_sim(sim)
    if shock_iter is not None and shock_iter >= 0:
        return int(shock_iter), 'Shock', 'Pre-shock', 'Post-shock'

    return None, None, None, None


def _finite_avg(values):
    finite = [x for x in values if math.isfinite(x)]
    return sum(finite) / len(finite) if finite else float('nan')


def _shock_window_slices(n: int, shock_iter: int):
    windows = [
        ('Pre', max(0, shock_iter - 50), max(0, shock_iter)),
        ('t0..t0+5', shock_iter, min(n, shock_iter + 5)),
        ('t0+5..t0+20', min(n, shock_iter + 5), min(n, shock_iter + 20)),
        ('t0+20..t0+100', min(n, shock_iter + 20), min(n, shock_iter + 100)),
    ]
    return [(label, start, end) for label, start, end in windows if end > start]


def _window_average(series, start: int, end: int):
    return _finite_avg(series[start:end])


def _rolling_normalization_time(series, *, shock_iter: int,
                                baseline: float,
                                direction: str,
                                rel_tol: float,
                                abs_tol: float = 0.0,
                                window: int = 5,
                                horizon: int = 100):
    if not math.isfinite(baseline):
        return float('nan')

    values = pd.Series(
        [x if math.isfinite(x) else np.nan for x in series],
        dtype='float64',
    )
    post = values.iloc[shock_iter:min(len(values), shock_iter + horizon)].reset_index(drop=True)
    rolled = post.rolling(window, min_periods=window).median()
    if direction == 'upper':
        target = max(baseline * (1.0 + rel_tol), baseline + abs_tol)
        recovered = rolled <= target
    else:
        target = baseline * (1.0 - rel_tol)
        recovered = rolled >= target

    hits = recovered[recovered].index.tolist()
    if not hits:
        return float('inf'), target
    return max(0, hits[0]), target


def _series_trough(series, *, shock_iter: int, direction: str, horizon: int = 100):
    end = min(len(series), shock_iter + horizon)
    post = [x for x in series[shock_iter:end]]
    if not post:
        return float('nan'), float('nan')

    best_idx = None
    best_val = None
    for idx, value in enumerate(post):
        if not math.isfinite(value):
            continue
        if best_val is None:
            best_val = value
            best_idx = idx
            continue
        if direction == 'upper' and value > best_val:
            best_val = value
            best_idx = idx
        if direction == 'lower' and value < best_val:
            best_val = value
            best_idx = idx
    if best_idx is None:
        return float('nan'), float('nan')
    return best_val, float(best_idx)


def _replenishment_speed(series, *, shock_iter: int, direction: str):
    start = shock_iter + 5
    end = shock_iter + 20
    if start >= len(series):
        return float('nan')
    start_val = _window_average(series, shock_iter, min(len(series), start))
    end_val = _window_average(series, min(len(series), start), min(len(series), end))
    if not (math.isfinite(start_val) and math.isfinite(end_val)):
        return float('nan')
    periods = max(1, min(len(series), end) - min(len(series), start))
    if direction == 'upper':
        return (start_val - end_val) / periods
    return (end_val - start_val) / periods


def _apply_preset_defaults(parser: argparse.ArgumentParser, args: argparse.Namespace):
    if not args.preset:
        return

    preset_vals = PRESETS[args.preset]
    for k, v in preset_vals.items():
        cli_key = k.replace("-", "_")
        if cli_key in vars(args):
            default_val = parser.get_default(cli_key)
            current_val = getattr(args, cli_key)
            if current_val == default_val:
                setattr(args, cli_key, v)
        else:
            setattr(args, cli_key, v)


def _median_finite(values):
    clean = [x for x in values if math.isfinite(x)]
    if not clean:
        return float('nan')
    return float(statistics.median(clean))


def _mean_finite(values):
    clean = [x for x in values if math.isfinite(x)]
    if not clean:
        return float('nan')
    return float(sum(clean) / len(clean))


def _std_finite(values):
    clean = [x for x in values if math.isfinite(x)]
    if len(clean) < 2:
        return 0.0 if clean else float('nan')
    return float(statistics.pstdev(clean))


def _window_bounds(n: int, start: int, end: int):
    lo = max(0, min(n, start))
    hi = max(lo, min(n, end))
    return lo, hi


def _shock_metric_snapshot(sim: Simulator):
    logger = sim.logger
    shock_iter = _shock_iter_from_sim(sim)
    if shock_iter is None:
        return None

    n = len(logger.iterations)
    pre_start, pre_end = _window_bounds(n, shock_iter - 30, shock_iter)
    shock_start, shock_end = _window_bounds(n, shock_iter, shock_iter + 5)
    post_start, post_end = _window_bounds(n, shock_iter + 5, shock_iter + 20)

    depth_series = [d.get('total', float('nan')) for d in logger.clob_depth]
    basis_series = logger.max_venue_basis_series()

    pre_spread = _median_finite(logger.clob_qspr[pre_start:pre_end])
    shock_spread = _median_finite(logger.clob_qspr[shock_start:shock_end])
    post_spread = _median_finite(logger.clob_qspr[post_start:post_end])
    pre_depth = _median_finite(depth_series[pre_start:pre_end])
    shock_depth = _median_finite(depth_series[shock_start:shock_end])
    post_depth = _median_finite(depth_series[post_start:post_end])
    shock_liq = _median_finite(logger.systemic_liquidity_series[shock_start:shock_end])
    post_liq = _median_finite(logger.systemic_liquidity_series[post_start:post_end])
    max_basis = _mean_finite(basis_series[shock_start:post_end])

    depth_ratio = float('nan')
    if math.isfinite(pre_depth) and pre_depth > 0 and math.isfinite(shock_depth):
        depth_ratio = shock_depth / pre_depth

    spread_ratio = float('nan')
    if math.isfinite(pre_spread) and pre_spread > 0 and math.isfinite(shock_spread):
        spread_ratio = shock_spread / pre_spread

    recovery_times = []
    metrics = [
        (logger.clob_qspr, 'upper', 0.50, 10.0),
        (depth_series, 'lower', 0.25, 0.0),
        (logger.systemic_liquidity_series, 'lower', 0.15, 0.0),
        (basis_series, 'upper', 0.25, 5.0),
    ]
    for series, direction, rel_tol, abs_tol in metrics:
        baseline = _median_finite(series[pre_start:pre_end])
        rt, _target = _rolling_normalization_time(
            series,
            shock_iter=shock_iter,
            baseline=baseline,
            direction=direction,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            window=5,
            horizon=100,
        )
        if math.isfinite(rt):
            recovery_times.append(rt)
        elif rt == float('inf'):
            recovery_times.append(rt)

    system_recovery = float('inf') if any(rt == float('inf') for rt in recovery_times) else (max(recovery_times) if recovery_times else float('nan'))

    return {
        'mode': 'shock',
        'pre_spread_bps': pre_spread,
        'shock_spread_bps': shock_spread,
        'post_spread_bps': post_spread,
        'spread_ratio': spread_ratio,
        'pre_depth': pre_depth,
        'shock_depth': shock_depth,
        'post_depth': post_depth,
        'depth_ratio': depth_ratio,
        'shock_systemic_liquidity': shock_liq,
        'post_systemic_liquidity': post_liq,
        'avg_basis_bps_0_20': max_basis,
        'system_recovery_ticks': system_recovery,
    }


def _stress_metric_snapshot(sim: Simulator):
    logger = sim.logger
    stress_start = getattr(sim.env, 'stress_start', None) if sim.env is not None else None
    if stress_start is None:
        return None

    n = len(logger.iterations)
    pre_start, pre_end = _window_bounds(n, stress_start - 50, stress_start)
    stress_start_i, stress_end_i = _window_bounds(n, stress_start, min(n, stress_start + 100))
    depth_series = [d.get('total', float('nan')) for d in logger.clob_depth]
    clob_share = [1.0 - sum(logger.flow_share(name)[i] for name in logger.amm_cost_curves)
                  for i in range(len(logger.iterations))]

    pre_spread = _mean_finite(logger.clob_qspr[pre_start:pre_end])
    stress_spread = _mean_finite(logger.clob_qspr[stress_start_i:stress_end_i])
    pre_depth = _mean_finite(depth_series[pre_start:pre_end])
    stress_depth = _mean_finite(depth_series[stress_start_i:stress_end_i])
    pre_liq = _mean_finite(logger.systemic_liquidity_series[pre_start:pre_end])
    stress_liq = _mean_finite(logger.systemic_liquidity_series[stress_start_i:stress_end_i])
    pre_clob_share = _mean_finite(clob_share[pre_start:pre_end])
    stress_clob_share = _mean_finite(clob_share[stress_start_i:stress_end_i])

    spread_ratio = float('nan')
    if math.isfinite(pre_spread) and pre_spread > 0 and math.isfinite(stress_spread):
        spread_ratio = stress_spread / pre_spread

    depth_ratio = float('nan')
    if math.isfinite(pre_depth) and pre_depth > 0 and math.isfinite(stress_depth):
        depth_ratio = stress_depth / pre_depth

    return {
        'mode': 'stress',
        'pre_spread_bps': pre_spread,
        'stress_spread_bps': stress_spread,
        'spread_ratio': spread_ratio,
        'pre_depth': pre_depth,
        'stress_depth': stress_depth,
        'depth_ratio': depth_ratio,
        'pre_systemic_liquidity': pre_liq,
        'stress_systemic_liquidity': stress_liq,
        'pre_clob_share': pre_clob_share,
        'stress_clob_share': stress_clob_share,
    }


def _baseline_metric_snapshot(sim: Simulator):
    logger = sim.logger
    n = len(logger.iterations)
    tail_start, tail_end = _window_bounds(n, max(0, n - 100), n)
    depth_series = [d.get('total', float('nan')) for d in logger.clob_depth]
    return {
        'mode': 'baseline',
        'avg_spread_bps': _mean_finite(logger.clob_qspr[tail_start:tail_end]),
        'avg_depth': _mean_finite(depth_series[tail_start:tail_end]),
        'avg_systemic_liquidity': _mean_finite(logger.systemic_liquidity_series[tail_start:tail_end]),
    }


def _robustness_snapshot(sim: Simulator):
    shock_snapshot = _shock_metric_snapshot(sim)
    if shock_snapshot is not None:
        return shock_snapshot
    stress_snapshot = _stress_metric_snapshot(sim)
    if stress_snapshot is not None:
        return stress_snapshot
    return _baseline_metric_snapshot(sim)


def _fmt_stat(mean_value: float, std_value: float, *, precision: int = 2, pct: bool = False):
    if not math.isfinite(mean_value):
        return 'N/A'
    suffix = '%' if pct else ''
    if not math.isfinite(std_value):
        return f'{mean_value:.{precision}f}{suffix}'
    return f'{mean_value:.{precision}f} ± {std_value:.{precision}f}{suffix}'


def print_robustness_summary(args: argparse.Namespace):
    seeds = max(2, int(args.robustness_seeds))
    base_seed = args.robustness_base_seed
    if base_seed is None:
        base_seed = args.seed if args.seed is not None else 42

    snapshots = []
    for offset in range(seeds):
        seed = base_seed + offset
        _seed_all(seed)
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = seed
        sim = build_sim(run_args)
        sim.simulate(run_args.n_iter, silent=True)
        snapshot = _robustness_snapshot(sim)
        snapshot['seed'] = seed
        snapshots.append(snapshot)

    mode = snapshots[0]['mode'] if snapshots else 'baseline'
    metrics = sorted(k for k in snapshots[0].keys() if k not in {'mode', 'seed'}) if snapshots else []

    W = 65
    print("\n" + "-" * W)
    print("  H4: ROBUSTNESS SANITY CHECK")
    print("-" * W)
    print(f"  Seeds: {base_seed}..{base_seed + seeds - 1}  (n={seeds})")
    print(f"  Scenario type: {mode}")
    print("\n  Metric means ± cross-seed std:")

    for metric in metrics:
        values = [snap[metric] for snap in snapshots]
        avg = _mean_finite(values)
        std = _std_finite(values)
        print(f"    {metric:<28s} {_fmt_stat(avg, std):>18s}")

    if mode == 'shock':
        stable = [snap for snap in snapshots if math.isfinite(snap['spread_ratio']) and math.isfinite(snap['depth_ratio'])]
        if stable:
            spread_ok = sum(snap['spread_ratio'] > 1.5 for snap in stable)
            depth_ok = sum(snap['depth_ratio'] < 0.8 for snap in stable)
            liq_ok = sum(snap['post_systemic_liquidity'] >= snap['shock_systemic_liquidity'] for snap in stable)
            print("\n  Quick checks:")
            print(f"    spread widens after shock: {spread_ok}/{len(stable)} seeds")
            print(f"    depth drops on impact:     {depth_ok}/{len(stable)} seeds")
            print(f"    liquidity rebounds by t+20:{liq_ok}/{len(stable)} seeds")
    elif mode == 'stress':
        stable = [snap for snap in snapshots if math.isfinite(snap['spread_ratio']) and math.isfinite(snap['depth_ratio'])]
        if stable:
            spread_ok = sum(snap['spread_ratio'] > 1.1 for snap in stable)
            depth_ok = sum(snap['depth_ratio'] < 0.95 for snap in stable)
            print("\n  Quick checks:")
            print(f"    spread wider in stress:    {spread_ok}/{len(stable)} seeds")
            print(f"    depth lower in stress:     {depth_ok}/{len(stable)} seeds")

    print("-" * W)


def build_parser(default_venue_choice_rule: str = "liquidity_aware") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-venue FX Agent-Based Model — interactive demo",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    g = p.add_argument_group("General")
    g.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                    help=_format_preset_help())
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
    g.add_argument("--comparison", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Run a second CLOB-only simulation and generate\n"
                        "With-AMM vs Without-AMM comparison plots (default: on)")
    g.add_argument("--robustness-check", action="store_true",
                   help="Run a short multi-seed sanity check after the main simulation")
    g.add_argument("--robustness-seeds", type=int, default=5,
                   help="Number of sequential seeds used in robustness check (default: 5)")
    g.add_argument("--robustness-base-seed", type=int, default=None,
                   help="First seed for robustness check (default: --seed or 42)")

    g = p.add_argument_group(
        "CLOB agents",
        "Traders that populate the central limit order book.\n"
        "  Noise traders place random limit/market/cancel orders.\n"
        "  FastRecyclerLP recycle short-lived near-mid liquidity.\n"
        "  LatentLP appear when spread/depth dislocate after stress.\n"
        "  Fundamentalist place DCF-based limit orders.\n"
        "  Chartist trade on sentiment (trend-following).\n"
        "  Universalist switch between Fund & Chart strategies.\n"
        "  Market Maker quotes both sides with σ/c-dependent spread."
    )
    g.add_argument("--n-noise", type=int, default=12,
                   help="CLOB noise/liquidity traders (default: 12)")
    g.add_argument("--n-mm", type=int, default=2,
                   help="Number of CLOB Market Makers (default: 2, 0=off)")
    g.add_argument("--n-fast-lp", type=int, default=10,
                   help="Fast replenishing LPs on the CLOB (default: 10)")
    g.add_argument("--n-latent-lp", type=int, default=6,
                   help="Latent/liquidity-backstop LPs on the CLOB (default: 6)")
    g.add_argument("--n-clob-fund", type=int, default=2,
                   help="Fundamentalist book agents on CLOB (default: 2)")
    g.add_argument("--n-clob-chart", type=int, default=0,
                   help="Chartist book agents on CLOB (default: 0)")
    g.add_argument("--n-clob-univ", type=int, default=1,
                   help="Universalist book agents on CLOB (default: 1)")
    g.add_argument("--enable-clob-mm", type=int, choices=[0, 1], default=1,
                   help="Enable CLOB Market Maker: 1=yes, 0=no (default: 1)")
    g.add_argument("--clob-std", type=float, default=2.0,
                   help="Std of initial order-book price distribution (default: 2.0)")
    g.add_argument("--clob-volume", type=int, default=1000,
                   help="Initial number of orders in book (default: 1000)")
    g.add_argument("--clob-anchor-strength", type=float, default=0.35,
                   help="Fraction of the fair-price gap closed by the CLOB per tick (default: 0.35)")
    g.add_argument("--clob-anchor-threshold-bps", type=float, default=5.0,
                   help="Ignore tiny fair-price gaps below this level when anchoring the CLOB (default: 5 bps)")
    g.add_argument("--clob-near-mid-target-ratio", type=float, default=1.5,
                   help="Multiplier for seeded near-mid background liquidity on the CLOB (default: 1.5x)")
    g.add_argument("--clob-amm-interaction", choices=["none", "competition", "toxicity"],
                   default="competition",
                   help="How AMM conditions affect CLOB MM risk-taking: none, competition, or toxicity (default: competition)")
    g.add_argument("--clob-amm-spread-impact-bps", type=float, default=3.0,
                   help="Sensitivity of MM spread-risk relief/penalty from AMM conditions (default: 3.0)")
    g.add_argument("--clob-amm-depth-impact", type=float, default=60.0,
                   help="Sensitivity of MM inventory/depth relief from AMM conditions (default: 60.0)")
    g.add_argument("--mm-withdraw-threshold", type=float, default=1.8,
                   help="Withdrawal score threshold for endogenous MM retreat (default: 1.8)")
    g.add_argument("--mm-reentry-threshold", type=float, default=1.05,
                   help="Re-entry score threshold for endogenous MM quoting to resume (default: 1.05)")
    g.add_argument("--mm-loss-threshold-bps", type=float, default=20.0,
                   help="EWMA mark-to-market loss in bps that maps to a unit withdrawal-loss score (default: 20)")
    g.add_argument("--mm-min-withdraw-ticks", type=int, default=4,
                   help="Minimum time endogenous MM stays withdrawn once it retreats (default: 4)")
    g.add_argument("--mm-reentry-ticks", type=int, default=3,
                   help="Ticks spent in the reentering state before returning to normal quoting (default: 3)")

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
        "         adjusted by CPMM bias and cost noise.\n"
        "Optional: liquidity_aware makes the top-level venue choice\n"
        "          respond to cost, depth, and venue price alignment."
    )
    g.add_argument("--amm-share", type=float, default=30.0,
                   dest="amm_share_pct",
                   help="Target AMM flow share in %% (0–100, default: 30). Explicit use opts into the legacy fixed_share venue rule")
    g.add_argument("--venue-choice-rule", choices=["fixed_share", "liquidity_aware"],
                   default=default_venue_choice_rule,
                   help="Top-level routing regime: liquidity_aware is the default; fixed_share preserves the legacy two-step AMM/CLOB split")
    g.add_argument("--deterministic", action="store_true",
                   help="Use deterministic argmin venue choice (overrides any stochastic routing rule)")
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
        "  clob-liq × → n_noise, n_fast_lp, n_latent_lp, clob_volume, MM depth.\n"
        "  amm-liq  × → CPMM / HFMM reserves."
    )
    g.add_argument("--clob-liq", type=float, default=1.0,
                   help="CLOB liquidity multiplier (default: 1.0)")
    g.add_argument("--amm-liq", type=float, default=1.0,
                   help="AMM liquidity multiplier (default: 1.0)")
    g.add_argument("--match-initial-depth", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Optional calibration aid: match initial live CLOB near-mid depth to aggregate AMM effective depth (default: off)")
    g.add_argument("--enable-amm", type=int, choices=[0, 1], default=1,
                   help="Enable AMM pools: 1=yes, 0=no (default: 1)")

    g = p.add_argument_group(
        "AMM pool parameters",
        "Configure CPMM (Uniswap-like) and HFMM (Curve-like) pools.\n"
        "  CPMM: x·y = k,  fee = cpmm-fee.\n"
        "  HFMM: StableSwap invariant,  fee = hfmm-fee, amplification = A."
    )
    g.add_argument("--cpmm-reserves", type=float, default=1000.0,
                   help="CPMM base-currency reserves (default: 1000)")
    g.add_argument("--hfmm-reserves", type=float, default=1000.0,
                   help="HFMM base-currency reserves (default: 1000)")
    g.add_argument("--cpmm-fee", type=float, default=0.003,
                   help="CPMM swap fee as fraction (default: 0.003 = 30 bps)")
    g.add_argument("--hfmm-fee", type=float, default=0.001,
                   help="HFMM swap fee as fraction (default: 0.001 = 10 bps)")
    g.add_argument("--hfmm-A", type=float, default=10.0,
                   help="HFMM amplification coefficient A (default: 10)")
    g.add_argument("--dynamic-fee", action="store_true",
                   help="Enable dynamic AMM fees that scale with volatility")

    g = p.add_argument_group(
        "Stress regime",
        "Gradual volatility / funding-cost ramp over [stress-start, stress-end].\n"
        "  σ: sigma-low → sigma-high\n"
        "  c: c-low     → c-high\n"
        "Set --stress-start -1 to disable.\n\n"
        "NOTE: a shock already includes an endogenous short-run liquidity\n"
        "      aftermath. Use --shock-regime-stress only if you also want\n"
        "      a slower exogenous regime-stress window."
    )
    g.add_argument("--stress-start", type=int, default=-1,
                   help="Iteration when stress begins (default: -1 = off)")
    g.add_argument("--stress-end", type=int, default=-1,
                   help="Iteration when stress ends (default: -1 = off)")
    g.add_argument("--sigma-low", type=float, default=0.01,
                   help="Volatility in normal regime (default: 0.01)")
    g.add_argument("--sigma-high", type=float, default=0.05,
                   help="Volatility in stress regime (default: 0.05)")
    g.add_argument("--c-low", type=float, default=0.002,
                   help="Funding cost in normal regime (default: 0.002)")
    g.add_argument("--c-high", type=float, default=0.020,
                   help="Funding cost in stress regime (default: 0.020)")

    g = p.add_argument_group(
        "Exogenous price shock",
        "Shock entry point shared by two implementations.\n"
        "  research: unified shock_pct with synchronous cross-venue reset.\n"
        "  realism:  decomposed event shock with separate fair-value, flow,\n"
        "            liquidity, and funding/volatility components.\n"
        "  Optional: --shock-regime-stress adds a slower regime-stress layer."
    )
    g.add_argument("--shock-iter", type=int, default=None,
                   help="Iteration of exogenous price shock (default: off)")
    g.add_argument("--shock-mode", choices=["research", "realism"], default="research",
                   help="Shock implementation: research keeps unified shock_pct; realism uses decomposed event shocks")
    g.add_argument("--shock-pct", type=float, default=-20.0,
                   help="Shock magnitude in %% (default: -20)")
    g.add_argument("--shock-regime-stress", action="store_true",
                   help="Couple the shock to an extended exogenous regime-stress window")
    g.add_argument("--no-shock-stress", action="store_true",
                   help="Deprecated alias. Pure shock is now the default.")

    g = p.add_argument_group(
        "Realism shock components",
        "Used only when --shock-mode realism.\n"
        "  fundamental: latent fair value jump\n"
        "  order-flow:  large CLOB sweep market order\n"
        "  liquidity:   cancellations + MM withdrawal + slower quote replenishment\n"
        "  funding-vol: σ and c spike with gradual decay"
    )
    g.add_argument("--fundamental-shock-pct", type=float, default=0.0,
                   help="Latent fair-value jump in %% for realism mode")
    g.add_argument("--order-flow-shock-qty", type=float, default=0.0,
                   help="Sweep market-order size on the CLOB in base units for realism mode")
    g.add_argument("--order-flow-shock-side", choices=["auto", "buy", "sell"], default="auto",
                   help="Direction of the realism-mode sweep order (default: auto from shock sign)")
    g.add_argument("--liquidity-shock-frac", type=float, default=0.0,
                   help="Fraction of resting orders cancelled in realism mode (0-1)")
    g.add_argument("--funding-vol-shock-intensity", type=float, default=0.0,
                   help="Intensity of sigma/funding jump in realism mode (1.0 ~= normal stress jump)")
    g.add_argument("--arb-max-correction-pct", type=float, default=10.0,
                   help="Max AMM price correction per arbitrage step in %% (default: 10)")
    g.add_argument("--arb-trade-fraction-cap", type=float, default=0.20,
                   help="Max fraction of AMM base reserves the arbitrageur can trade per step (default: 0.20)")

    return p


def build_sim(args: argparse.Namespace) -> Simulator:
    stress_start = args.stress_start if args.stress_start >= 0 else None
    stress_end = args.stress_end if stress_start is not None else None

    return Simulator.default_fx(
        n_noise=args.n_noise,
        n_mm=args.n_mm if args.enable_clob_mm else 0,
        n_fast_lp=args.n_fast_lp,
        n_latent_lp=args.n_latent_lp,
        n_clob_fund=args.n_clob_fund,
        n_clob_chart=args.n_clob_chart,
        n_clob_univ=args.n_clob_univ,
        n_fx_takers=args.n_fx_takers,
        n_fx_fund=args.n_fx_fund,
        n_retail=args.n_retail,
        n_institutional=args.n_institutional,
        clob_std=args.clob_std,
        clob_volume=args.clob_volume,
        clob_liq=args.clob_liq,
        clob_anchor_strength=args.clob_anchor_strength,
        clob_anchor_threshold_bps=args.clob_anchor_threshold_bps,
        clob_background_target_ratio=args.clob_near_mid_target_ratio,
        clob_amm_interaction=args.clob_amm_interaction,
        clob_amm_spread_impact_bps=args.clob_amm_spread_impact_bps,
        clob_amm_depth_impact=args.clob_amm_depth_impact,
        mm_withdraw_threshold=args.mm_withdraw_threshold,
        mm_reentry_threshold=args.mm_reentry_threshold,
        mm_loss_threshold_bps=args.mm_loss_threshold_bps,
        mm_min_withdraw_ticks=args.mm_min_withdraw_ticks,
        mm_reentry_ticks=args.mm_reentry_ticks,
        enable_amm=bool(args.enable_amm),
        amm_liq=args.amm_liq,
        match_initial_depth=args.match_initial_depth,
        cpmm_reserves=args.cpmm_reserves,
        hfmm_reserves=args.hfmm_reserves,
        cpmm_fee=args.cpmm_fee,
        hfmm_fee=args.hfmm_fee,
        hfmm_A=args.hfmm_A,
        amm_share_pct=args.amm_share_pct,
        venue_choice_rule=args.venue_choice_rule,
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
        shock_mode=args.shock_mode,
        fundamental_shock_pct=args.fundamental_shock_pct,
        order_flow_shock_qty=args.order_flow_shock_qty,
        order_flow_shock_side=args.order_flow_shock_side,
        liquidity_shock_frac=args.liquidity_shock_frac,
        funding_vol_shock_intensity=args.funding_vol_shock_intensity,
        arb_max_correction_pct=args.arb_max_correction_pct,
        arb_trade_fraction_cap=args.arb_trade_fraction_cap,
        dynamic_fee=args.dynamic_fee,
        reprice_prob_recovery=getattr(args, 'reprice_prob_recovery', None),
        anchor_strength_recovery=getattr(args, 'anchor_strength_recovery', None),
        bg_target_ratio_recovery=getattr(args, 'bg_target_ratio_recovery', None),
        toxic_flow_decay=getattr(args, 'toxic_flow_decay', None),
        liquidity_shock_decay=getattr(args, 'liquidity_shock_decay', None),
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
    row("Random seed", str(args.seed) if args.seed is not None else "random")
    row("Robustness check", "ON" if args.robustness_check else "OFF")
    if args.robustness_check:
        row("Robustness seeds", str(args.robustness_seeds))
        row("Robustness base seed",
            str(args.robustness_base_seed) if args.robustness_base_seed is not None else "auto")
    if args.preset:
        row("Preset applied", args.preset)
        row("Preset family", _preset_family(args.preset))

    eff_mm = args.n_mm if args.enable_clob_mm else 0
    shadow = bool(args.enable_amm) and eff_mm == 0
    print("\n  CLOB agents")
    row("Noise traders", str(args.n_noise))
    row("Market Makers", str(eff_mm))
    row("FastRecyclerLP", str(args.n_fast_lp))
    row("LatentLP", str(args.n_latent_lp))
    row("Fundamentalists (book)", str(args.n_clob_fund))
    row("Chartists (book)", str(args.n_clob_chart))
    row("Universalists (book)", str(args.n_clob_univ))
    row("CLOB mode", "Shadow (synthetic)" if shadow else "Live order-book")
    row("Order-book volume", str(args.clob_volume))
    row("Price std", f"{args.clob_std:.1f}")
    row("CLOB liquidity multiplier", f"{args.clob_liq:.1f}", "×")
    row("CLOB anchor strength", f"{args.clob_anchor_strength:.2f}")
    row("CLOB anchor threshold", f"{args.clob_anchor_threshold_bps:.1f}", "bps")
    row("Near-mid target", f"{args.clob_near_mid_target_ratio:.1f}", "×")
    row("AMM interaction", args.clob_amm_interaction)
    row("AMM spread impact", f"{args.clob_amm_spread_impact_bps:.1f}", "bps")
    row("AMM depth impact", f"{args.clob_amm_depth_impact:.1f}")

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
    route_mode = "argmin" if args.deterministic else args.venue_choice_rule
    row("Venue choice mode", route_mode)
    row("beta_AMM (intra-AMM)", f"{args.beta_amm:.3f}")
    row("CPMM bias", f"{args.cpmm_bias_bps:.1f}", "bps")
    row("Cost noise sigma", f"{args.cost_noise_std:.1f}", "bps")

    print("\n  AMM pools")
    row("AMM enabled", "YES" if args.enable_amm else "NO")
    if args.enable_amm:
        row("Initial depth match", "ON" if args.match_initial_depth else "OFF")
        row("CPMM reserves (base)", f"{args.cpmm_reserves:.0f}")
        row("HFMM reserves (base)", f"{args.hfmm_reserves:.0f}")
        row("CPMM fee", f"{args.cpmm_fee * 10_000:.0f}", "bps")
        row("HFMM fee", f"{args.hfmm_fee * 10_000:.0f}", "bps")
        row("HFMM amplification A", f"{args.hfmm_A:.0f}")
        row("Dynamic fees", "ON" if args.dynamic_fee else "OFF")
        row("AMM liquidity multiplier", f"{args.amm_liq:.1f}", "×")

    print("\n  Stress regime")
    if args.stress_start >= 0:
        auto_tag = ""
        if (args.shock_iter is not None
                and getattr(args, 'shock_regime_stress', False)
                and not getattr(args, 'no_shock_stress', False)):
            auto_tag = " (auto from shock)"
        row("Window", f"[{args.stress_start}, {args.stress_end}]{auto_tag}")
        row("Volatility σ", f"{args.sigma_low:.4f} → {args.sigma_high:.4f}")
        row("Funding cost c", f"{args.c_low:.4f} → {args.c_high:.4f}")
    else:
        row("Status", "OFF")

    print("\n  Exogenous price shock")
    if args.shock_iter is not None:
        row("Iteration", str(args.shock_iter))
        row("Shock mode", args.shock_mode)
        if args.shock_mode == 'research':
            row("Magnitude", f"{args.shock_pct:+.0f}", "%")
            row("Hits CLOB orders", "YES")
            row("Hits AMM reserves", "YES" if args.enable_amm else "N/A")
            row("Endogenous aftermath", "ON")
        else:
            row("Fundamental shock", f"{args.fundamental_shock_pct:+.1f}", "%")
            row("Order-flow shock", f"{args.order_flow_shock_qty:.0f}", "base")
            row("Order-flow side", args.order_flow_shock_side)
            row("Liquidity shock", f"{args.liquidity_shock_frac:.2f}")
            row("Funding/vol shock", f"{args.funding_vol_shock_intensity:.2f}")
            row("AMM reserve rebalance", "OFF")
            row("Arb max correction", f"{args.arb_max_correction_pct:.1f}", "%")
            row("Arb trade cap", f"{args.arb_trade_fraction_cap:.2f}")
        row("Regime stress layer", "ON"
            if (getattr(args, 'shock_regime_stress', False)
                and not getattr(args, 'no_shock_stress', False))
            else "OFF")
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
    print("  H1: EXECUTION COST COMPARISON  &  VENUE INTERACTION")
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

    split_iter, split_title, phase_before, phase_after = _linkage_split_point(sim)
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

    clob_depth_series = [d.get('total', float('nan')) for d in logger.clob_depth]
    print("\n  Liquidity Commonality (near-mid depth / shared factor):")
    clob_factor = logger.series_correlation(clob_depth_series, logger.systemic_liquidity_series)
    print(f"    CLOB depth <-> factor: rho = {clob_factor:.3f}")
    for name in logger.amm_cost_curves:
        amm_depth = logger.amm_depth_series.get(name, [])
        depth_rho = logger.series_correlation(clob_depth_series, amm_depth)
        factor_rho = logger.series_correlation(amm_depth, logger.systemic_liquidity_series)
        print(f"    CLOB depth <-> {name.upper()} depth: rho = {depth_rho:.3f}"
              f"  |  {name.upper()} depth <-> factor: rho = {factor_rho:.3f}")

    if split_iter is not None:
        n = len(logger.iterations)
        idx = min(split_iter, n)

        if split_title == 'Shock':
            windows = _shock_window_slices(n, idx)

            print(f"\n  Cost Correlation By Shock Window (Q=5, t={split_iter}):")
            for name in logger.amm_cost_curves:
                parts = []
                clob_cost = logger.cost_series('clob', 5)
                amm_cost = logger.cost_series(name, 5)
                for label, start, end in windows:
                    rho = logger.series_correlation(clob_cost[start:end], amm_cost[start:end])
                    rho_s = f"{rho:.3f}" if math.isfinite(rho) else "N/A"
                    parts.append(f"{label}={rho_s}")
                print(f"    CLOB <-> {name.upper()}: " + "  ".join(parts))

            print("\n  CLOB Quoted Spread By Local Window:")
            for label, start, end in windows:
                avg_qspr = _window_average(logger.clob_qspr, start, end)
                avg_s = f"{avg_qspr:.1f}" if math.isfinite(avg_qspr) else "N/A"
                print(f"    {label:<10s} {avg_s:>8s} bps")

            venues_all = ['clob'] + list(logger.amm_cost_curves.keys())
            print("\n  Execution Cost By Local Window (Q=5, bps):")
            print(f"  {'Window':<12s}", end='')
            for venue in venues_all:
                print(f"  {venue.upper():>8s}", end='')
            print()
            for label, start, end in windows:
                print(f"  {label:<12s}", end='')
                for venue in venues_all:
                    avg_cost = _window_average(logger.cost_series(venue, 5), start, end)
                    avg_s = f"{avg_cost:.1f}" if math.isfinite(avg_cost) else "N/A"
                    print(f"  {avg_s:>8s}", end='')
                print()

            print("\n  AMM Volume Share By Local Window:")
            for name in logger.amm_cost_curves:
                parts = []
                fs = logger.flow_share(name)
                for label, start, end in windows:
                    avg_share = _window_average(fs, start, end)
                    share_s = f"{avg_share:.1%}" if math.isfinite(avg_share) else "N/A"
                    parts.append(f"{label}={share_s}")
                print(f"    {name.upper()}: " + "  ".join(parts))

            print("\n  Venue Basis By Local Window (abs bps):")
            max_basis = logger.max_venue_basis_series()
            parts = []
            for label, start, end in windows:
                avg_basis = _window_average(max_basis, start, end)
                basis_s = f"{avg_basis:.1f}" if math.isfinite(avg_basis) else "N/A"
                parts.append(f"{label}={basis_s}")
            print(f"    MAX |AMM - CLOB|: " + "  ".join(parts))
            for name in logger.amm_cost_curves:
                basis = [abs(x) if math.isfinite(x) else float('nan') for x in logger.venue_basis_series(name)]
                parts = []
                for label, start, end in windows:
                    avg_basis = _window_average(basis, start, end)
                    basis_s = f"{avg_basis:.1f}" if math.isfinite(avg_basis) else "N/A"
                    parts.append(f"{label}={basis_s}")
                print(f"    {name.upper():>6s}: " + "  ".join(parts))
        else:
            print(f"\n  Correlation Before / After {split_title} (t={split_iter}):")
            for name in logger.amm_cost_curves:
                ba = logger.commonality_before_after('clob', name, Q=5,
                                                     stress_start=split_iter)
                b_s = f"{ba['before']:.3f}" if math.isfinite(ba['before']) else "N/A"
                a_s = f"{ba['after']:.3f}" if math.isfinite(ba['after']) else "N/A"
                print(f"    CLOB <-> {name.upper()}: before={b_s}, after={a_s}")

            print(f"\n  Depth Correlation Before / After {split_title}:")
            for name in logger.amm_cost_curves:
                ba = logger.series_commonality_before_after(
                    clob_depth_series,
                    logger.amm_depth_series.get(name, []),
                    split_iter,
                )
                b_s = f"{ba['before']:.3f}" if math.isfinite(ba['before']) else "N/A"
                a_s = f"{ba['after']:.3f}" if math.isfinite(ba['after']) else "N/A"
                print(f"    CLOB <-> {name.upper()}: before={b_s}, after={a_s}")

            print(f"\n  AMM Volume Share — {phase_before} vs {phase_after}:")
            for name in logger.amm_cost_curves:
                fs = logger.flow_share(name)
                before = fs[:idx]
                after = fs[idx:]
                avg_b = sum(before) / len(before) if before else 0
                avg_a = sum(after) / len(after) if after else 0
                delta = avg_a - avg_b
                print(f"    {name.upper()}: {phase_before}={avg_b:.1%}  "
                      f"{phase_after}={avg_a:.1%}  Delta={delta:+.1%}")

            qspr = logger.clob_qspr
            normal_s = [s for s in qspr[:idx] if math.isfinite(s)]
            stress_s = [s for s in qspr[idx:] if math.isfinite(s)]
            avg_n = sum(normal_s) / len(normal_s) if normal_s else 0
            avg_st = sum(stress_s) / len(stress_s) if stress_s else 0
            print("\n  CLOB Quoted Spread:")
            print(f"    {phase_before}: {avg_n:.1f} bps")
            if avg_n > 0:
                print(f"    {phase_after}: {avg_st:.1f} bps  ({avg_st/avg_n:.1f}x wider)")

            Q_rep = [1, 5, 10, 50]
            venues_all = ['clob'] + list(logger.amm_cost_curves.keys())
            print(f"\n  Execution Cost — {phase_before} vs {phase_after} (bps):")
            print(f"  {'Q':>4s}", end='')
            for v in venues_all:
                print(f"  {v.upper()+'-N':>8s} {v.upper()+'-S':>8s} {'×':>5s}", end='')
            print()
            for Q in Q_rep:
                print(f"  {Q:>4.0f}", end='')
                for v in venues_all:
                    cs = logger.cost_series(v, Q)
                    nrm = [x for x in cs[:idx] if math.isfinite(x)]
                    strs = [x for x in cs[idx:] if math.isfinite(x)]
                    avg_nrm = sum(nrm) / len(nrm) if nrm else float('nan')
                    avg_str = sum(strs) / len(strs) if strs else float('nan')
                    ratio = avg_str / avg_nrm if avg_nrm and avg_nrm > 0 else float('nan')
                    n_s = f"{avg_nrm:.1f}" if math.isfinite(avg_nrm) else "N/A"
                    s_s = f"{avg_str:.1f}" if math.isfinite(avg_str) else "N/A"
                    r_s = f"{ratio:.1f}" if math.isfinite(ratio) else "-"
                    print(f"  {n_s:>8s} {s_s:>8s} {r_s:>5s}", end='')
                print()

    # ---- Recovery time after shock ------------------------------------
    shock_iter = _shock_iter_from_sim(sim)

    if shock_iter is not None:
        print("\n" + "-" * W)
        print("  H3: POST-SHOCK RECOVERY TIME")
        print("-" * W)

        trades = pd.DataFrame(logger.trade_log) if logger.trade_log else pd.DataFrame()

        WINDOW = 5
        MAX_WARMUP = 50
        HORIZON = 100

        baseline_end = shock_iter
        if (sim.env is not None
                and getattr(sim.env, 'stress_start', None) is not None):
            baseline_end = min(baseline_end, sim.env.stress_start)
        warmup = min(MAX_WARMUP, max(0, baseline_end - 30))

        def _baseline(series):
            clean = [x for x in series[warmup:baseline_end] if np.isfinite(x)]
            if len(clean) < WINDOW:
                clean = [x for x in series[:baseline_end] if np.isfinite(x)]
            if len(clean) < WINDOW:
                return float('nan')
            return float(np.median(clean))

        def _rolling_recovery(series, *, direction: str,
                              rel_tol: float, abs_tol: float = 0.0,
                              window: int = WINDOW):
            baseline = _baseline(series)
            if not np.isfinite(baseline):
                return float('nan'), float('nan'), float('nan')

            values = pd.Series(
                [x if np.isfinite(x) else np.nan for x in series],
                dtype='float64',
            )
            post_values = values.iloc[shock_iter:].reset_index(drop=True)
            post = post_values.rolling(window, min_periods=window).median()
            if len(post) == 0:
                return float('nan'), baseline, float('nan')

            if direction == 'upper':
                target = max(baseline * (1.0 + rel_tol), baseline + abs_tol)
                recovered = post <= target
            else:
                target = baseline * (1.0 - rel_tol)
                recovered = post >= target

            hits = recovered[recovered].index.tolist()
            if not hits:
                return float('inf'), baseline, target
            return max(0, hits[0]), baseline, target

        def _trade_cost_series(venue: str = None, fallback=None):
            if fallback is None:
                fallback = []
            if trades.empty or 'cost_bps' not in trades.columns:
                return fallback

            subset = trades
            if venue is not None:
                subset = trades[trades['venue'] == venue]
            if subset.empty:
                return fallback

            med = subset.groupby('t')['cost_bps'].median()
            out = [float('nan')] * len(logger.iterations)
            for t, value in med.items():
                idx = int(t)
                if 0 <= idx < len(out):
                    out[idx] = float(value)

            finite = sum(np.isfinite(value) for value in out)
            return out if finite >= WINDOW else fallback

        def _avg_amm_cost_series(Q: float = 5):
            out = []
            for i in range(len(logger.iterations)):
                vals = []
                for name in logger.amm_cost_curves:
                    series = logger.amm_cost_curves[name].get(Q, [])
                    if i < len(series) and np.isfinite(series[i]):
                        vals.append(series[i])
                out.append(sum(vals) / len(vals) if vals else float('nan'))
            return out

        clob_depth_series = [d.get('total', float('nan')) for d in logger.clob_depth]
        clob_cost_series = _trade_cost_series(
            'clob',
            fallback=logger.cost_series('clob', 5),
        )
        max_basis_series = logger.max_venue_basis_series()
        metrics = [
            ('CLOB trade cost', clob_cost_series, 'upper', 0.50, 10.0),
            ('CLOB quoted spread', logger.clob_qspr, 'upper', 0.50, 10.0),
            ('CLOB near-mid depth', clob_depth_series, 'lower', 0.25, 0.0),
            ('Systemic liquidity', logger.systemic_liquidity_series, 'lower', 0.15, 0.0),
            ('Venue basis |AMM-CLOB|', max_basis_series, 'upper', 0.25, 5.0),
        ]

        if logger.amm_cost_curves:
            amm_depth_series = [
                sum(logger.amm_depth_series[name][i] for name in logger.amm_depth_series)
                for i in range(len(logger.iterations))
            ]
            metrics.extend([
                ('AMM quote cost (Q=5)', _avg_amm_cost_series(5), 'upper', 0.50, 10.0),
                ('AMM effective depth', amm_depth_series, 'lower', 0.25, 0.0),
            ])

        def _fmt(v):
            if v != v:
                return 'N/A'
            if v == float('inf'):
                return 'never'
            return f'{v:.0f}'

        baseline_mid = _window_average(logger.clob_mid_series, warmup, baseline_end)
        price_drawdown = []
        for mid in logger.clob_mid_series:
            if math.isfinite(mid) and math.isfinite(baseline_mid) and baseline_mid > 0:
                price_drawdown.append(10_000.0 * (mid - baseline_mid) / baseline_mid)
            else:
                price_drawdown.append(float('nan'))
        trough_bps, trough_time = _series_trough(
            price_drawdown,
            shock_iter=shock_iter,
            direction='lower',
            horizon=HORIZON,
        )

        print(f"\n  Event windows: [t0,t0+5), [t0+5,t0+20), [t0+20,t0+100)")
        print("  Recovery is evaluated locally around the event, not on the full post-shock tail.")
        print("\n  Impact trough:")
        trough_s = f"{trough_bps:.1f}" if math.isfinite(trough_bps) else "N/A"
        t_s = f"+{trough_time:.0f}" if math.isfinite(trough_time) else "N/A"
        print(f"    CLOB mid drawdown: {trough_s} bps at {t_s}")
        basis_trough, basis_time = _series_trough(
            max_basis_series,
            shock_iter=shock_iter,
            direction='upper',
            horizon=HORIZON,
        )
        basis_s = f"{basis_trough:.1f}" if math.isfinite(basis_trough) else "N/A"
        basis_t = f"+{basis_time:.0f}" if math.isfinite(basis_time) else "N/A"
        print(f"    Max venue basis:   {basis_s} bps at {basis_t}")

        print("\n  Replenishment speed (per tick, t0+5 -> t0+20):")
        clob_repl = _replenishment_speed(clob_depth_series, shock_iter=shock_iter, direction='lower')
        sysliq_repl = _replenishment_speed(logger.systemic_liquidity_series, shock_iter=shock_iter, direction='lower')
        basis_repl = _replenishment_speed(max_basis_series, shock_iter=shock_iter, direction='upper')
        print(f"    CLOB depth:        {clob_repl:.2f}" if math.isfinite(clob_repl) else "    CLOB depth:        N/A")
        print(f"    Systemic liquidity:{sysliq_repl:.4f}" if math.isfinite(sysliq_repl) else "    Systemic liquidity:N/A")
        print(f"    Basis compression: {basis_repl:.2f} bps" if math.isfinite(basis_repl) else "    Basis compression: N/A")

        def _bl_fmt(v):
            return f'{v:.1f}' if np.isfinite(v) else 'N/A'

        def _target_fmt(v):
            return f'{v:.1f}' if np.isfinite(v) else 'N/A'

        print(f"\n  {'Metric':<24s}  {'Baseline':>10s}  {'Target':>10s}  {'Normalize':>10s}")
        print(f"  {'-'*24}  {'-'*10}  {'-'*10}  {'-'*10}")

        recovery_times = []
        for label, series, direction, rel_tol, abs_tol in metrics:
            baseline = _baseline(series)
            rt, target = _rolling_normalization_time(
                series,
                shock_iter=shock_iter,
                baseline=baseline,
                direction=direction,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                window=WINDOW,
                horizon=HORIZON,
            )
            if np.isfinite(rt) or rt == float('inf'):
                recovery_times.append(rt)
            print(f"  {label:<24s}  {_bl_fmt(baseline):>10s}  "
                  f"{_target_fmt(target):>10s}  {_fmt(rt):>10s}")

        if recovery_times:
            system_recovery = float('inf') if any(rt == float('inf') for rt in recovery_times) else max(recovery_times)
            print(f"\n  {'System-wide max':<24s}  {'—':>10s}  {'—':>10s}  {_fmt(system_recovery):>10s}")

    print("=" * W + "\n")


def generate_all_plots(sim: Simulator, logger_no_amm=None,
                       out_dir: str = 'output/main_aware'):
    import glob as _glob
    logger = sim.logger
    stress_start = sim.env.stress_start if sim.env else None
    shock_iter = getattr(sim, 'shock_iter', None)

    # Remove stale plots from previous runs.
    for old_png in _glob.glob(os.path.join(out_dir, '*.png')):
        os.remove(old_png)

    # Save dashboards + all individual plots to the selected output directory.
    generate_all_dashboards(
        logger,
        out_dir=out_dir,
        stress_start=stress_start,
        Q=5,
        rolling=10,
        logger_no_amm=logger_no_amm,
        shock_iter=shock_iter,
    )
    save_all_individual_plots(
        logger,
        out_dir=out_dir,
        stress_start=stress_start,
        Q=5,
        rolling=10,
        logger_no_amm=logger_no_amm,
        shock_iter=shock_iter,
    )


def _cli_flag_present(flag: str, argv: list[str]) -> bool:
    return any(arg == flag or arg.startswith(f'{flag}=') for arg in argv)


def _resolve_main_routing(args: argparse.Namespace, argv: list[str]) -> str:
    if _cli_flag_present('--venue-choice-rule', argv):
        return args.venue_choice_rule
    if _cli_flag_present('--amm-share', argv):
        return 'fixed_share'
    if args.preset in LEGACY_PRESETS and 'amm_share_pct' in PRESETS.get(args.preset, {}):
        return 'fixed_share'
    return args.venue_choice_rule


def _main_output_dir(venue_choice_rule: str) -> str:
    if venue_choice_rule == 'fixed_share':
        return 'output/main_fixed'
    return 'output/main_aware'


def _run_main(argv: Optional[list[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    # 1. Apply preset defaults (CLI overrides preset)
    _apply_preset_defaults(parser, args)

    args.venue_choice_rule = _resolve_main_routing(args, argv)
    plot_out_dir = _main_output_dir(args.venue_choice_rule)

    # 2. Auto-generate stress around shock (AFTER preset applied)
    _auto_stress_around_shock(args)

    # 3. Seed both RNGs
    if args.seed is not None:
        _seed_all(args.seed)

    print_config(args)

    sim = build_sim(args)
    sim.simulate(args.n_iter, silent=args.silent)

    if not args.no_summary:
        print_summary(sim)

    if args.robustness_check:
        print_robustness_summary(args)

    logger_no_amm = None
    if args.comparison and bool(args.enable_amm):
        print('\nRunning counterfactual simulation WITHOUT AMM ...')
        # Re-seed for identical GBM trajectory
        if args.seed is not None:
            _seed_all(args.seed)
        else:
            print('  ⚠ WARNING: no --seed; counterfactual trajectory differs.')
            print('    Set --seed N for valid A/B comparison.')
        args_no = argparse.Namespace(**vars(args))
        args_no.enable_amm = 0
        args_no.amm_share_pct = 0
        sim_no = build_sim(args_no)
        sim_no.simulate(args.n_iter, silent=args.silent)
        logger_no_amm = sim_no.logger
        print('Counterfactual done.\n')

    if not args.no_plots:
        generate_all_plots(sim, logger_no_amm=logger_no_amm,
                           out_dir=plot_out_dir)


def main():
    _run_main()


if __name__ == '__main__':
    main()