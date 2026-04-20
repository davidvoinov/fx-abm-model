#!/usr/bin/env python3
"""Paired fixed_share vs liquidity_aware comparison on identical seeds."""

from __future__ import annotations

import argparse
import csv
import math
import os

from main import _apply_preset_defaults, _auto_stress_around_shock, _seed_all, build_parser, build_sim
from AgentBasedModel.metrics.resilience import (
    composite_resilience_metrics,
    price_resilience_metrics,
    series_resilience_metrics,
)
from AgentBasedModel.metrics.statistics import bootstrap_paired_diff_ci, paired_t_test


DEFAULT_PANELS = [
    ('mm_withdrawal', 'MM Withdrawal', 'pre_shock_baseline'),
    ('flash_crash', 'Flash Crash', 'pre_shock_baseline'),
    ('dealer_liquidity_crisis', 'Dealer Liquidity Crisis', 'fair_value_gap'),
    ('funding_liquidity_shock', 'Funding Liquidity Shock', 'pre_shock_baseline'),
]

DELTA_METRICS = [
    ('flow_share_clob', 'CLOB flow share'),
    ('flow_share_cpmm', 'CPMM flow share'),
    ('flow_share_hfmm', 'HFMM flow share'),
    ('flow_share_amm_total', 'Total AMM flow share'),
    ('mean_abs_basis_bps', 'Mean absolute max basis (bps)'),
    ('peak_abs_basis_bps', 'Peak absolute max basis (bps)'),
    ('post_shock_mean_qspr_bps', 'Post-shock quoted spread (bps)'),
    ('post_shock_mean_depth', 'Post-shock CLOB depth'),
    ('price_recovered_indicator', 'Price recovered indicator'),
    ('price_time_observed_steps', 'Price observed time'),
    ('price_normalized_auc_abs', 'Price normalized AUC'),
    ('spread_recovered_indicator', 'Spread recovered indicator'),
    ('spread_time_observed_steps', 'Spread observed time'),
    ('spread_normalized_auc_abs', 'Spread normalized AUC'),
    ('depth_recovered_indicator', 'Depth recovered indicator'),
    ('depth_time_observed_steps', 'Depth observed time'),
    ('depth_normalized_auc_abs', 'Depth normalized AUC'),
    ('execution_cost_recovered_indicator', 'Execution-cost recovered indicator'),
    ('execution_cost_time_observed_steps', 'Execution-cost observed time'),
    ('execution_cost_normalized_auc_abs', 'Execution-cost normalized AUC'),
    ('composite_recovered_indicator', 'Composite recovered indicator'),
    ('composite_time_observed_steps', 'Composite observed time'),
    ('composite_normalized_auc_abs', 'Composite normalized AUC'),
]


def _mean_finite(values) -> float:
    clean = [float(value) for value in values if value is not None and math.isfinite(value)]
    return sum(clean) / len(clean) if clean else float('nan')


def _median_finite(values) -> float:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(value))
    n = len(clean)
    if n == 0:
        return float('nan')
    mid = n // 2
    if n % 2 == 1:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def _base_args_for_preset(preset: str, n_iter: int, min_post_shock_window: int):
    parser = build_parser()
    args = parser.parse_args([])
    args.preset = preset
    args.n_iter = n_iter
    _apply_preset_defaults(parser, args)
    if args.shock_iter is not None:
        args.n_iter = max(args.n_iter, int(args.shock_iter) + max(50, min_post_shock_window))
    _auto_stress_around_shock(args)
    return args


def _resolve_cost_series(logger, cost_q: float):
    if not logger.clob_cost_curves:
        return [], float(cost_q)
    if cost_q in logger.clob_cost_curves:
        return logger.clob_cost_curves[cost_q], float(cost_q)
    available = sorted(logger.clob_cost_curves.keys(), key=lambda value: abs(value - cost_q))
    chosen_q = float(available[0])
    return logger.clob_cost_curves[chosen_q], chosen_q


def _resilience_bundle(price_metrics: dict, logger, shock_iter: int,
                       stable_window: int, avg_window: int,
                       tolerance_pct: float, retracement_fraction: float,
                       impact_window: int, horizon: int,
                       cost_q: float) -> dict:
    depth_series = [depth.get('total', float('nan')) for depth in logger.clob_depth]
    cost_series, resolved_cost_q = _resolve_cost_series(logger, cost_q)

    spread_metrics = series_resilience_metrics(
        logger.clob_qspr,
        shock_iter,
        avg_window=avg_window,
        tolerance_pct=tolerance_pct,
        stable_window=stable_window,
        horizon=horizon,
        retracement_fraction=retracement_fraction,
        impact_window=impact_window,
    )
    depth_metrics = series_resilience_metrics(
        depth_series,
        shock_iter,
        avg_window=avg_window,
        tolerance_pct=tolerance_pct,
        stable_window=stable_window,
        horizon=horizon,
        retracement_fraction=retracement_fraction,
        impact_window=impact_window,
    )
    cost_metrics = series_resilience_metrics(
        cost_series,
        shock_iter,
        avg_window=avg_window,
        tolerance_pct=tolerance_pct,
        stable_window=stable_window,
        horizon=horizon,
        retracement_fraction=retracement_fraction,
        impact_window=impact_window,
    )
    composite_metrics = composite_resilience_metrics(
        [price_metrics, spread_metrics, depth_metrics, cost_metrics],
        analysis_target_mode='joint_price_spread_depth_cost',
    )
    return {
        'spread': spread_metrics,
        'depth': depth_metrics,
        'execution_cost': cost_metrics,
        'composite': composite_metrics,
        'resolved_cost_q': resolved_cost_q,
    }


def _metric_fields(prefix: str, metrics: dict) -> dict:
    return {
        f'{prefix}_recovered_indicator': 1.0 if metrics.get('recovered', False) else 0.0,
        f'{prefix}_time_observed_steps': metrics.get('time_observed_steps', float('nan')),
        f'{prefix}_normalized_auc_abs': metrics.get('normalized_auc_abs', float('nan')),
    }


def _collect_seed_row(preset: str, label: str, seed: int, venue_choice_rule: str,
                      n_iter: int, tolerance_pct: float,
                      stable_window: int, avg_window: int,
                      retracement_fraction: float,
                      impact_window: int,
                      min_post_shock_window: int,
                      target_mode: str,
                      cost_q: float) -> dict | None:
    args = _base_args_for_preset(preset, n_iter, min_post_shock_window)
    args.seed = seed
    args.venue_choice_rule = venue_choice_rule
    _seed_all(seed)

    sim = build_sim(args)
    sim.simulate(args.n_iter, silent=True)
    shock_iter = getattr(sim, 'shock_iter', None)
    if shock_iter is None:
        return None

    target_series = getattr(sim.logger, 'fair_price_series', None) if target_mode == 'fair_value_gap' else None
    horizon = max(50, min(args.n_iter - shock_iter, min_post_shock_window))
    price_metrics = price_resilience_metrics(
        sim.logger.clob_mid_series,
        shock_iter=shock_iter,
        baseline_window=50,
        avg_window=avg_window,
        tolerance_pct=tolerance_pct,
        stable_window=stable_window,
        horizon=horizon,
        retracement_fraction=retracement_fraction,
        impact_window=impact_window,
        target_series=target_series,
        target_mode=target_mode,
    )
    resilience = _resilience_bundle(
        price_metrics,
        sim.logger,
        shock_iter,
        stable_window=stable_window,
        avg_window=avg_window,
        tolerance_pct=tolerance_pct,
        retracement_fraction=retracement_fraction,
        impact_window=impact_window,
        horizon=horizon,
        cost_q=cost_q,
    )
    post_slice = slice(shock_iter, min(len(sim.logger.clob_qspr), shock_iter + horizon))
    basis_series = sim.logger.max_venue_basis_series()[post_slice]
    depth_series = [depth.get('total', float('nan')) for depth in sim.logger.clob_depth][post_slice]
    qspr_series = sim.logger.clob_qspr[post_slice]
    row = {
        'preset': preset,
        'panel_label': label,
        'seed': seed,
        'venue_choice_rule': venue_choice_rule,
        'shock_iter': shock_iter,
        'analysis_target_mode': price_metrics.get('analysis_target_mode', target_mode),
        'resolved_cost_q': resilience['resolved_cost_q'],
        'flow_share_clob': _mean_finite(sim.logger.flow_share('clob')),
        'flow_share_cpmm': _mean_finite(sim.logger.flow_share('cpmm')),
        'flow_share_hfmm': _mean_finite(sim.logger.flow_share('hfmm')),
        'flow_share_amm_total': _mean_finite(
            clob_or_zero for clob_or_zero in [
                cpmm + hfmm
                for cpmm, hfmm in zip(sim.logger.flow_share('cpmm'), sim.logger.flow_share('hfmm'))
            ]
        ),
        'mean_abs_basis_bps': _mean_finite(abs(value) for value in basis_series),
        'peak_abs_basis_bps': max((abs(value) for value in basis_series if math.isfinite(value)), default=float('nan')),
        'post_shock_mean_qspr_bps': _mean_finite(qspr_series),
        'post_shock_mean_depth': _mean_finite(depth_series),
    }
    row.update(_metric_fields('price', price_metrics))
    row.update(_metric_fields('spread', resilience['spread']))
    row.update(_metric_fields('depth', resilience['depth']))
    row.update(_metric_fields('execution_cost', resilience['execution_cost']))
    row.update(_metric_fields('composite', resilience['composite']))
    return row


def _delta_seed(base_seed: int, *parts: str) -> int:
    state = int(base_seed) % (2 ** 32)
    for part in parts:
        for ch in str(part):
            state = (state * 131 + ord(ch)) % (2 ** 32)
    return state


def _paired_rows(seed_rows: list[dict], bootstrap_reps: int,
                 ci_level: float, base_seed: int) -> tuple[list[dict], list[dict]]:
    indexed = {}
    for row in seed_rows:
        indexed[(row['panel_label'], row['seed'], row['venue_choice_rule'])] = row

    pair_rows = []
    labels = sorted({row['panel_label'] for row in seed_rows})
    seeds = sorted({int(row['seed']) for row in seed_rows})
    for label in labels:
        for seed in seeds:
            fixed_row = indexed.get((label, seed, 'fixed_share'))
            liquidity_row = indexed.get((label, seed, 'liquidity_aware'))
            if fixed_row is None or liquidity_row is None:
                continue
            pair = {
                'panel_label': label,
                'seed': seed,
            }
            for metric_name, _ in DELTA_METRICS:
                fixed_value = fixed_row.get(metric_name, float('nan'))
                liquidity_value = liquidity_row.get(metric_name, float('nan'))
                pair[f'{metric_name}_fixed_share'] = fixed_value
                pair[f'{metric_name}_liquidity_aware'] = liquidity_value
                if math.isfinite(fixed_value) and math.isfinite(liquidity_value):
                    pair[f'{metric_name}_delta_liquidity_minus_fixed'] = liquidity_value - fixed_value
                else:
                    pair[f'{metric_name}_delta_liquidity_minus_fixed'] = float('nan')
            pair_rows.append(pair)

    summary_rows = []
    for label in labels:
        panel_pairs = [row for row in pair_rows if row['panel_label'] == label]
        for metric_name, metric_label in DELTA_METRICS:
            fixed_values = [row[f'{metric_name}_fixed_share'] for row in panel_pairs]
            liquidity_values = [row[f'{metric_name}_liquidity_aware'] for row in panel_pairs]
            paired = paired_t_test(liquidity_values, fixed_values)
            boot = bootstrap_paired_diff_ci(
                liquidity_values,
                fixed_values,
                n_boot=bootstrap_reps,
                ci_level=ci_level,
                seed=_delta_seed(base_seed, label, metric_name),
            )
            deltas = [
                row[f'{metric_name}_delta_liquidity_minus_fixed']
                for row in panel_pairs
                if math.isfinite(row[f'{metric_name}_delta_liquidity_minus_fixed'])
            ]
            summary_rows.append({
                'panel_label': label,
                'metric_name': metric_name,
                'metric_label': metric_label,
                'n_pairs': paired['n'],
                'mean_fixed_share': paired['mean_b'],
                'mean_liquidity_aware': paired['mean_a'],
                'mean_delta_liquidity_minus_fixed': paired['mean_diff'],
                'median_delta_liquidity_minus_fixed': _median_finite(deltas),
                'share_positive_delta': _mean_finite(1.0 if value > 0 else 0.0 for value in deltas),
                'paired_t_stat': paired['t_stat'],
                'paired_p_value': paired['p_value'],
                'ci_level': ci_level,
                'delta_ci_lo': boot['ci_lo'],
                'delta_ci_hi': boot['ci_hi'],
            })
    return pair_rows, summary_rows


def _save_csv(path: str, rows: list[dict], fieldnames: list[str]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def main():
    parser = argparse.ArgumentParser(
        description='Paired fixed_share vs liquidity_aware experiment on identical seeds.',
    )
    parser.add_argument('--seeds', type=int, default=40,
                        help='Number of sequential seeds per panel and route mode (default: 40)')
    parser.add_argument('--base-seed', type=int, default=42,
                        help='Starting seed (default: 42)')
    parser.add_argument('--n-iter', type=int, default=900,
                        help='Simulation length for each run (default: 900)')
    parser.add_argument('--avg-window', type=int, default=20,
                        help='Window used for average post-shock change (default: 20)')
    parser.add_argument('--tolerance-pct', type=float, default=1.0,
                        help='Legacy fixed recovery band retained for reference columns (default: 1.0)')
    parser.add_argument('--retracement-fraction', type=float, default=0.8,
                        help='Fraction of dislocation that must be closed to count as recovery (default: 0.8)')
    parser.add_argument('--impact-window', type=int, default=10,
                        help='Early post-shock window used to anchor peak dislocation (default: 10)')
    parser.add_argument('--min-post-shock-window', type=int, default=600,
                        help='Minimum post-shock steps observed for recovery analysis (default: 600)')
    parser.add_argument('--stable-window', type=int, default=10,
                        help='Consecutive periods required inside the recovery band (default: 10)')
    parser.add_argument('--cost-q', type=float, default=10.0,
                        help='Representative CLOB trade size used for execution-cost resilience (default: 10)')
    parser.add_argument('--bootstrap-reps', type=int, default=2000,
                        help='Bootstrap repetitions for paired delta confidence intervals (default: 2000)')
    parser.add_argument('--ci-level', type=float, default=0.95,
                        help='Confidence level for paired bootstrap intervals (default: 0.95)')
    parser.add_argument('--progress-every', type=int, default=10,
                        help='Print progress every N seeds per panel and route mode (default: 10)')
    parser.add_argument('--out-dir', default='output/routing_comparison',
                        help='Output directory for paired comparison CSVs')
    args = parser.parse_args()

    seed_rows = []
    route_modes = ('fixed_share', 'liquidity_aware')
    for preset, label, target_mode in DEFAULT_PANELS:
        for venue_choice_rule in route_modes:
            print(f'Starting {label} [{venue_choice_rule}]', flush=True)
            for offset in range(args.seeds):
                seed = args.base_seed + offset
                row = _collect_seed_row(
                    preset=preset,
                    label=label,
                    seed=seed,
                    venue_choice_rule=venue_choice_rule,
                    n_iter=args.n_iter,
                    tolerance_pct=args.tolerance_pct,
                    stable_window=args.stable_window,
                    avg_window=args.avg_window,
                    retracement_fraction=args.retracement_fraction,
                    impact_window=args.impact_window,
                    min_post_shock_window=args.min_post_shock_window,
                    target_mode=target_mode,
                    cost_q=args.cost_q,
                )
                if row is not None:
                    seed_rows.append(row)
                done = offset + 1
                if args.progress_every > 0 and (done == 1 or done % args.progress_every == 0 or done == args.seeds):
                    print(f'[{label} | {venue_choice_rule}] {done}/{args.seeds} seeds complete', flush=True)

    pair_rows, summary_rows = _paired_rows(
        seed_rows,
        bootstrap_reps=args.bootstrap_reps,
        ci_level=args.ci_level,
        base_seed=args.base_seed,
    )
    seed_fields = [
        'panel_label', 'preset', 'seed', 'venue_choice_rule', 'shock_iter', 'analysis_target_mode', 'resolved_cost_q',
        'flow_share_clob', 'flow_share_cpmm', 'flow_share_hfmm', 'flow_share_amm_total',
        'mean_abs_basis_bps', 'peak_abs_basis_bps', 'post_shock_mean_qspr_bps', 'post_shock_mean_depth',
    ] + [metric_name for metric_name, _ in DELTA_METRICS if metric_name not in {
        'flow_share_clob', 'flow_share_cpmm', 'flow_share_hfmm', 'flow_share_amm_total',
        'mean_abs_basis_bps', 'peak_abs_basis_bps', 'post_shock_mean_qspr_bps', 'post_shock_mean_depth',
    }]
    pair_fields = ['panel_label', 'seed']
    for metric_name, _ in DELTA_METRICS:
        pair_fields.extend([
            f'{metric_name}_fixed_share',
            f'{metric_name}_liquidity_aware',
            f'{metric_name}_delta_liquidity_minus_fixed',
        ])
    summary_fields = [
        'panel_label', 'metric_name', 'metric_label', 'n_pairs',
        'mean_fixed_share', 'mean_liquidity_aware', 'mean_delta_liquidity_minus_fixed',
        'median_delta_liquidity_minus_fixed', 'share_positive_delta',
        'paired_t_stat', 'paired_p_value', 'ci_level', 'delta_ci_lo', 'delta_ci_hi',
    ]

    seed_path = _save_csv(os.path.join(args.out_dir, 'routing_comparison_seed_metrics.csv'), seed_rows, seed_fields)
    pair_path = _save_csv(os.path.join(args.out_dir, 'routing_comparison_paired_deltas.csv'), pair_rows, pair_fields)
    summary_path = _save_csv(os.path.join(args.out_dir, 'routing_comparison_summary.csv'), summary_rows, summary_fields)
    print(f'Saved seed-level routing metrics to {seed_path}')
    print(f'Saved paired routing deltas to {pair_path}')
    print(f'Saved paired routing summary to {summary_path}')


if __name__ == '__main__':
    main()