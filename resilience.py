#!/usr/bin/env python3
"""Generate censor-aware multi-seed resilience study outputs."""

from __future__ import annotations

import argparse
import csv
import math
import os

from main import _apply_preset_defaults, _auto_stress_around_shock, _seed_all, build_parser, build_sim
from AgentBasedModel.metrics.resilience import (
    composite_resilience_metrics,
    kaplan_meier_curve,
    price_resilience_metrics,
    series_resilience_metrics,
)
from AgentBasedModel.visualization.resilience_plots import (
    plot_kaplan_meier_panels,
    plot_resilience_scatter_panels,
)


DEFAULT_PANELS = [
    ('mm_withdrawal', 'MM Withdrawal', '#1f77b4', 'pre_shock_baseline', 'Pre-shock baseline'),
    ('flash_crash', 'Flash Crash', '#2ca02c', 'pre_shock_baseline', 'Pre-shock baseline'),
    ('dealer_liquidity_crisis', 'Dealer Liquidity Crisis', '#ff7f0e', 'fair_value_gap', 'Fair-value gap'),
    ('funding_liquidity_shock', 'Funding Liquidity Shock', '#d62728', 'pre_shock_baseline', 'Pre-shock baseline'),
]

PRIORITY_METRICS = [
    ('spread_resilience', 'Quoted Spread Resilience'),
    ('depth_resilience', 'Depth Resilience'),
    ('execution_cost_resilience', 'Execution Cost Resilience'),
    ('composite_resilience', 'Composite Resilience'),
]


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


def _summary_row(label: str, target_mode: str, points: list, metric_name: str = '', metric_label: str = '') -> dict:
    km = kaplan_meier_curve(
        [point.get('time_observed_steps', float('nan')) for point in points],
        [bool(point.get('recovered', False)) for point in points],
    )
    row = {
        'panel_label': label,
        'target_mode': target_mode,
        'n_obs': len(points),
        'recovered_share': _mean_finite(
            1.0 if point.get('recovered', False) else 0.0 for point in points
        ),
        'km_median_steps': km['median_time'],
        'mean_observed_steps': _mean_finite(point.get('time_observed_steps', float('nan')) for point in points),
        'mean_recovery_steps': _mean_finite(point.get('recovery_steps', float('nan')) for point in points),
        'mean_normalized_avg_impact': _mean_finite(point.get('normalized_avg_impact', float('nan')) for point in points),
        'mean_normalized_auc_abs': _mean_finite(point.get('normalized_auc_abs', float('nan')) for point in points),
        'mean_initial_dislocation_pct': _mean_finite(point.get('initial_dislocation_pct', float('nan')) for point in points),
        'mean_reference_dislocation_step': _mean_finite(point.get('reference_dislocation_step', float('nan')) for point in points),
    }
    if metric_name:
        row['metric_name'] = metric_name
    if metric_label:
        row['metric_label'] = metric_label
    return row


def _resolve_cost_series(logger, cost_q: float):
    if not logger.clob_cost_curves:
        return [], float(cost_q)

    if cost_q in logger.clob_cost_curves:
        return logger.clob_cost_curves[cost_q], float(cost_q)

    available = sorted(logger.clob_cost_curves.keys(), key=lambda value: abs(value - cost_q))
    chosen_q = float(available[0])
    return logger.clob_cost_curves[chosen_q], chosen_q


def _priority_metric_points(price_metrics: dict, logger, shock_iter: int,
                            stable_window: int, avg_window: int,
                            tolerance_pct: float, retracement_fraction: float,
                            impact_window: int, horizon: int,
                            cost_q: float) -> list:
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

    rows = []
    metric_rows = [
        ('spread_resilience', 'Quoted Spread Resilience', spread_metrics, ''),
        ('depth_resilience', 'Depth Resilience', depth_metrics, ''),
        ('execution_cost_resilience', 'Execution Cost Resilience', cost_metrics, f'clob_cost_q={resolved_cost_q:g}'),
        ('composite_resilience', 'Composite Resilience', composite_metrics, 'price+spread+depth+cost'),
    ]
    for metric_name, metric_label, metrics, note in metric_rows:
        row = dict(metrics)
        row['metric_name'] = metric_name
        row['metric_label'] = metric_label
        row['metric_note'] = note
        rows.append(row)
    return rows


def _collect_points(preset: str, seeds: int, base_seed: int,
                    n_iter: int, tolerance_pct: float,
                    stable_window: int, avg_window: int,
                    retracement_fraction: float,
                    impact_window: int,
                    min_post_shock_window: int,
                    target_mode: str,
                    cost_q: float,
                    progress_every: int = 0) -> list:
    points = []
    priority_metric_points = []
    for offset in range(seeds):
        seed = base_seed + offset
        args = _base_args_for_preset(preset, n_iter, min_post_shock_window)
        args.seed = seed
        _seed_all(seed)

        sim = build_sim(args)
        sim.simulate(args.n_iter, silent=True)

        shock_iter = getattr(sim, 'shock_iter', None)
        if shock_iter is None:
            continue

        target_series = None
        if target_mode == 'fair_value_gap':
            target_series = getattr(sim.logger, 'fair_price_series', None)

        horizon = max(50, min(args.n_iter - shock_iter, min_post_shock_window))

        metrics = price_resilience_metrics(
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
        points.append({
            'preset': preset,
            'seed': seed,
            'target_mode': metrics['analysis_target_mode'],
            'recovery_steps': metrics['recovery_steps'],
            'time_observed_steps': metrics['time_observed_steps'],
            'recovered': metrics['recovered'],
            'is_censored': metrics['is_censored'],
            'half_life_steps': metrics['half_life_steps'],
            'legacy_recovery_steps': metrics['legacy_recovery_steps'],
            'initial_dislocation_pct': metrics['initial_dislocation_pct'],
            'reference_dislocation_pct': metrics['reference_dislocation_pct'],
            'reference_dislocation_step': metrics['reference_dislocation_step'],
            'target_abs_deviation_pct': metrics['target_abs_deviation_pct'],
            'avg_change_pct': metrics['avg_change_pct'],
            'normalized_avg_impact': metrics['normalized_avg_impact'],
            'normalized_auc_abs': metrics['normalized_auc_abs'],
            'trough_change_pct': metrics['trough_change_pct'],
            'peak_abs_change_pct': metrics['peak_abs_change_pct'],
        })
        for metric_point in _priority_metric_points(
            metrics,
            sim.logger,
            shock_iter,
            stable_window=stable_window,
            avg_window=avg_window,
            tolerance_pct=tolerance_pct,
            retracement_fraction=retracement_fraction,
            impact_window=impact_window,
            horizon=horizon,
            cost_q=cost_q,
        ):
            metric_row = dict(metric_point)
            metric_row['preset'] = preset
            metric_row['seed'] = seed
            priority_metric_points.append(metric_row)
        done = offset + 1
        if progress_every > 0 and (done == 1 or done % progress_every == 0 or done == seeds):
            print(f'[{preset}] {done}/{seeds} seeds complete', flush=True)
    return {
        'price_points': points,
        'priority_metric_points': priority_metric_points,
    }


def _save_csv(panel_specs, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'resilience_scatter_points.csv')
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'panel_label', 'preset', 'seed', 'target_mode', 'recovered', 'is_censored',
                'recovery_steps', 'time_observed_steps', 'half_life_steps',
                'legacy_recovery_steps', 'initial_dislocation_pct', 'reference_dislocation_pct',
                'reference_dislocation_step', 'target_abs_deviation_pct',
                'avg_change_pct', 'normalized_avg_impact', 'normalized_auc_abs',
                'trough_change_pct', 'peak_abs_change_pct',
            ],
        )
        writer.writeheader()
        for spec in panel_specs:
            for point in spec['points']:
                row = dict(point)
                row['panel_label'] = spec['label']
                writer.writerow(row)
    return path


def _save_priority_metric_csv(panel_specs, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'resilience_priority_metric_points.csv')
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'panel_label', 'preset', 'seed', 'metric_name', 'metric_label', 'metric_note',
                'analysis_target_mode', 'recovered', 'is_censored', 'recovery_steps',
                'time_observed_steps', 'half_life_steps', 'legacy_recovery_steps',
                'initial_dislocation_pct', 'reference_dislocation_pct', 'reference_dislocation_step',
                'target_abs_deviation_pct', 'avg_change_pct', 'normalized_avg_impact',
                'normalized_auc_abs', 'trough_change_pct', 'peak_abs_change_pct',
            ],
            extrasaction='ignore',
        )
        writer.writeheader()
        for spec in panel_specs:
            for point in spec['priority_metric_points']:
                row = dict(point)
                row['panel_label'] = spec['label']
                writer.writerow(row)
    return path


def _mean_finite(values) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else float('nan')


def _save_summary_csv(panel_specs, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'resilience_panel_summary.csv')
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'panel_label', 'target_mode', 'n_obs', 'recovered_share', 'km_median_steps',
                'mean_observed_steps', 'mean_recovery_steps', 'mean_normalized_avg_impact',
                'mean_normalized_auc_abs', 'mean_initial_dislocation_pct',
                'mean_reference_dislocation_step',
            ],
        )
        writer.writeheader()
        for spec in panel_specs:
            writer.writerow(_summary_row(spec['label'], spec.get('target_mode', 'pre_shock_baseline'), list(spec['points'])))
    return path


def _save_priority_metric_summary_csv(panel_specs, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'resilience_priority_metric_summary.csv')
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'panel_label', 'metric_name', 'metric_label', 'target_mode', 'n_obs',
                'recovered_share', 'km_median_steps', 'mean_observed_steps', 'mean_recovery_steps',
                'mean_normalized_avg_impact', 'mean_normalized_auc_abs',
                'mean_initial_dislocation_pct', 'mean_reference_dislocation_step',
            ],
        )
        writer.writeheader()
        for spec in panel_specs:
            for metric_name, metric_label in PRIORITY_METRICS:
                points = [
                    point for point in spec['priority_metric_points']
                    if point.get('metric_name') == metric_name
                ]
                if not points:
                    continue
                writer.writerow(
                    _summary_row(
                        spec['label'],
                        points[0].get('analysis_target_mode', 'pre_shock_baseline'),
                        points,
                        metric_name=metric_name,
                        metric_label=metric_label,
                    )
                )
    return path


def main():
    parser = argparse.ArgumentParser(
        description='Generate resilience scatter panels from multi-seed shock simulations.',
    )
    parser.add_argument('--seeds', type=int, default=300,
                        help='Number of sequential seeds per panel (default: 300)')
    parser.add_argument('--base-seed', type=int, default=42,
                        help='Starting seed (default: 42)')
    parser.add_argument('--n-iter', type=int, default=900,
                        help='Simulation length for each run (default: 900)')
    parser.add_argument('--avg-window', type=int, default=20,
                        help='Window used for average post-shock price change (default: 20)')
    parser.add_argument('--tolerance-pct', type=float, default=1.0,
                        help='Legacy fixed recovery band retained for reference columns (default: 1.0)')
    parser.add_argument('--retracement-fraction', type=float, default=0.8,
                        help='Fraction of initial dislocation that must be closed to count as recovery (default: 0.8)')
    parser.add_argument('--impact-window', type=int, default=10,
                        help='Early post-shock window used to anchor the peak dislocation (default: 10)')
    parser.add_argument('--min-post-shock-window', type=int, default=600,
                        help='Minimum number of post-shock steps observed for recovery analysis (default: 600)')
    parser.add_argument('--stable-window', type=int, default=10,
                        help='Consecutive periods required inside the recovery band (default: 10)')
    parser.add_argument('--progress-every', type=int, default=25,
                        help='Print progress every N seeds per panel (default: 25, 0 disables progress output)')
    parser.add_argument('--cost-q', type=float, default=10.0,
                        help='Representative CLOB trade size used for execution-cost resilience (default: 10)')
    parser.add_argument('--out-dir', default='output/resilience',
                        help='Output directory for PNG and CSV files')
    args = parser.parse_args()

    panel_specs = []
    for preset, label, color, target_mode, target_label in DEFAULT_PANELS:
        print(f'Starting panel: {label} [{target_label}]', flush=True)
        panel_data = _collect_points(
            preset=preset,
            seeds=args.seeds,
            base_seed=args.base_seed,
            n_iter=args.n_iter,
            tolerance_pct=args.tolerance_pct,
            stable_window=args.stable_window,
            avg_window=args.avg_window,
            retracement_fraction=args.retracement_fraction,
            impact_window=args.impact_window,
            min_post_shock_window=args.min_post_shock_window,
            target_mode=target_mode,
            cost_q=args.cost_q,
            progress_every=args.progress_every,
        )
        panel_specs.append({
            'label': label,
            'color': color,
            'target_mode': target_mode,
            'target_label': target_label,
            'points': panel_data['price_points'],
            'priority_metric_points': panel_data['priority_metric_points'],
        })

    png_path = plot_resilience_scatter_panels(panel_specs, out_dir=args.out_dir)
    km_path = plot_kaplan_meier_panels(panel_specs, out_dir=args.out_dir)
    csv_path = _save_csv(panel_specs, out_dir=args.out_dir)
    summary_path = _save_summary_csv(panel_specs, out_dir=args.out_dir)
    priority_csv_path = _save_priority_metric_csv(panel_specs, out_dir=args.out_dir)
    priority_summary_path = _save_priority_metric_summary_csv(panel_specs, out_dir=args.out_dir)

    print(f'Saved resilience scatter panels to {png_path}')
    print(f'Saved recovery survival panels to {km_path}')
    print(f'Saved resilience point data to {csv_path}')
    print(f'Saved resilience panel summary to {summary_path}')
    print(f'Saved priority resilience metric data to {priority_csv_path}')
    print(f'Saved priority resilience metric summary to {priority_summary_path}')


if __name__ == '__main__':
    main()