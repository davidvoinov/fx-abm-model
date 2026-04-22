#!/usr/bin/env python3
"""Generate research-question statistical tests for the FX ABM."""

from __future__ import annotations

import argparse
import csv
import math
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from main import _apply_preset_defaults, _auto_stress_around_shock, _seed_all, build_parser, build_sim
from AgentBasedModel.metrics.resilience import (
    composite_resilience_metrics,
    price_resilience_metrics,
    series_resilience_metrics,
)
from AgentBasedModel.metrics.statistics import (
    bootstrap_mean_ci,
    bootstrap_paired_diff_ci,
    paired_permutation_test,
)


DEFAULT_SCENARIOS = [
    ('default', 'Default'),
    ('mm_withdrawal', 'MM Withdrawal'),
    ('flash_crash', 'Flash Crash'),
    ('dealer_liquidity_crisis', 'Dealer Liquidity Crisis'),
    ('funding_liquidity_shock', 'Funding Liquidity Shock'),
]

MARKET_STRUCTURES = [
    ('with_amm', 'With AMM', True),
    ('without_amm', 'Without AMM', False),
]

PRICE_TARGET_MODE = {
    'mm_withdrawal': 'pre_shock_baseline',
    'flash_crash': 'pre_shock_baseline',
    'dealer_liquidity_crisis': 'fair_value_gap',
    'funding_liquidity_shock': 'pre_shock_baseline',
}

RQ_FIGURE_METRICS = {
    'RQ1': [
        'avg_clob_spread_bps',
        'avg_clob_depth',
        'avg_realized_volatility',
        'avg_cost_clob_q5',
        'price_recovery_steps',
        'spread_recovery_steps',
        'depth_recovery_steps',
        'liquidity_composite_recovery_steps',
    ],
    'RQ2': [
        'post_shock_avg_clob_spread_bps',
        'post_shock_avg_clob_depth',
        'post_shock_avg_realized_volatility',
        'post_shock_avg_systemic_liquidity',
        'post_shock_mm_share_withdrawn',
        'post_shock_mm_withdrawal_score',
        'avg_mm_share_withdrawn',
        'avg_mm_withdrawal_score',
    ],
}


def _metric_spec():
    return [
        ('n_trades', 'Total Trades', 'RQ1', False),
        ('avg_clob_spread_bps', 'Average CLOB Spread', 'RQ1', False),
        ('avg_clob_depth', 'Average CLOB Depth', 'RQ1', False),
        ('avg_systemic_liquidity', 'Average Systemic Liquidity', 'RQ1', False),
        ('avg_realized_volatility', 'Average Realized Volatility', 'RQ1', False),
        ('tail_realized_volatility', 'Tail Realized Volatility', 'RQ1', False),
        ('avg_cost_clob_q5', 'Average CLOB Cost Q=5', 'RQ1', False),
        ('avg_flow_share_amm_total', 'Average AMM Flow Share', 'RQ1', False),
        ('price_recovery_steps', 'Price Recovery Steps', 'RQ1', True),
        ('spread_recovery_steps', 'Spread Recovery Steps', 'RQ1', True),
        ('depth_recovery_steps', 'Depth Recovery Steps', 'RQ1', True),
        ('liquidity_composite_recovery_steps', 'Liquidity Composite Recovery Steps', 'RQ1', True),
        ('liquidity_composite_recovered', 'Liquidity Composite Recovered', 'RQ1', True),
        ('post_shock_avg_clob_spread_bps', 'Post-shock CLOB Spread', 'RQ2', True),
        ('post_shock_avg_clob_depth', 'Post-shock CLOB Depth', 'RQ2', True),
        ('post_shock_avg_systemic_liquidity', 'Post-shock Systemic Liquidity', 'RQ2', True),
        ('post_shock_avg_realized_volatility', 'Post-shock Realized Volatility', 'RQ2', True),
        ('post_shock_mm_share_withdrawn', 'Post-shock MM Withdrawn Share', 'RQ2', True),
        ('post_shock_mm_withdrawal_score', 'Post-shock MM Withdrawal Score', 'RQ2', True),
        ('avg_mm_share_withdrawn', 'Average MM Withdrawn Share', 'RQ2', False),
        ('avg_mm_withdrawal_score', 'Average MM Withdrawal Score', 'RQ2', False),
    ]


def _metric_label_map() -> dict[str, str]:
    return {
        metric_name: metric_label
        for metric_name, metric_label, _research_question, _shock_only in _metric_spec()
    }


def _metric_preference(metric_name: str) -> str | None:
    higher_is_better = {
        'avg_clob_depth',
        'avg_systemic_liquidity',
        'liquidity_composite_recovered',
        'post_shock_avg_clob_depth',
        'post_shock_avg_systemic_liquidity',
    }
    lower_is_better = {
        'avg_clob_spread_bps',
        'avg_realized_volatility',
        'tail_realized_volatility',
        'avg_cost_clob_q5',
        'price_recovery_steps',
        'spread_recovery_steps',
        'depth_recovery_steps',
        'liquidity_composite_recovery_steps',
        'post_shock_avg_clob_spread_bps',
        'post_shock_avg_realized_volatility',
        'post_shock_mm_share_withdrawn',
        'post_shock_mm_withdrawal_score',
        'avg_mm_share_withdrawn',
        'avg_mm_withdrawal_score',
    }
    if metric_name in higher_is_better:
        return 'higher'
    if metric_name in lower_is_better:
        return 'lower'
    return None


def _scenario_rank(scenario_key: str) -> int:
    ranks = {key: idx for idx, (key, _label) in enumerate(DEFAULT_SCENARIOS)}
    return ranks.get(scenario_key, len(ranks))


def _significance_marker(p_value: float) -> str:
    if not math.isfinite(p_value):
        return ''
    if p_value < 0.01:
        return ' **'
    if p_value < 0.05:
        return ' *'
    return ''


def _effect_color(metric_name: str, delta: float) -> str:
    if not math.isfinite(delta):
        return '#7f8c8d'
    preference = _metric_preference(metric_name)
    if preference == 'higher':
        return '#1f9d55' if delta > 0 else '#d1495b'
    if preference == 'lower':
        return '#1f9d55' if delta < 0 else '#d1495b'
    return '#4c78a8'


def _is_favorable(metric_name: str, delta: float) -> bool:
    preference = _metric_preference(metric_name)
    if preference == 'higher':
        return delta > 0
    if preference == 'lower':
        return delta < 0
    return delta > 0


def _heatmap_score(metric_name: str, delta: float, p_value: float) -> float:
    if not math.isfinite(delta):
        return 0.0
    magnitude = 0.35
    if math.isfinite(p_value):
        if p_value < 0.01:
            magnitude = 2.0
        elif p_value < 0.05:
            magnitude = 1.5
        elif p_value < 0.10:
            magnitude = 1.0
    return magnitude if _is_favorable(metric_name, delta) else -magnitude


def _format_delta(delta: float, p_value: float) -> str:
    if not math.isfinite(delta):
        return 'NA'
    abs_delta = abs(delta)
    if abs_delta >= 100:
        text = f'{delta:+.0f}'
    elif abs_delta >= 10:
        text = f'{delta:+.1f}'
    elif abs_delta >= 1:
        text = f'{delta:+.2f}'
    else:
        text = f'{delta:+.3f}'
    return text + _significance_marker(p_value).replace(' ', '')


def _stat_seed(base_seed: int, *parts: str) -> int:
    state = int(base_seed) % (2 ** 32)
    for part in parts:
        for ch in str(part):
            state = (state * 131 + ord(ch)) % (2 ** 32)
    return state


def _mean_finite(values) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else float('nan')


def _tail_mean(series, window: int = 100) -> float:
    if not series:
        return float('nan')
    return _mean_finite(series[-min(window, len(series)):])


def _window_mean(series, start: int, end: int) -> float:
    if not series:
        return float('nan')
    lo = max(0, min(len(series), start))
    hi = max(lo, min(len(series), end))
    return _mean_finite(series[lo:hi])


def _rolling_return_vol(mid_series: list[float], window: int = 20) -> list[float]:
    returns = []
    for idx in range(1, len(mid_series)):
        p0 = mid_series[idx - 1]
        p1 = mid_series[idx]
        if p0 > 0 and p1 > 0 and math.isfinite(p0) and math.isfinite(p1):
            returns.append(math.log(p1 / p0))
        else:
            returns.append(0.0)

    vol = []
    for idx in range(len(returns)):
        start = max(0, idx - window + 1)
        chunk = returns[start:idx + 1]
        if len(chunk) < 2:
            vol.append(0.0)
            continue
        mean_value = sum(chunk) / len(chunk)
        var = sum((value - mean_value) ** 2 for value in chunk) / (len(chunk) - 1)
        vol.append(var ** 0.5)
    return [0.0] + vol


def _shock_iter_from_sim(sim) -> int | None:
    shock_iter = getattr(sim, 'shock_iter', None)
    if shock_iter is None:
        shock_iter = getattr(sim, '_shock_iter', None)
    return shock_iter


def _base_args_for_scenario(preset: str, n_iter: int,
                            venue_choice_rule: str = 'liquidity_aware'):
    parser = build_parser(default_venue_choice_rule=venue_choice_rule)
    args = parser.parse_args([])
    args.preset = preset
    args.n_iter = n_iter
    _apply_preset_defaults(parser, args)
    args.venue_choice_rule = venue_choice_rule
    _auto_stress_around_shock(args)
    return args


def _scenario_metrics(sim, scenario_key: str,
                      resilience_horizon: int,
                      stable_window: int,
                      avg_window: int,
                      retracement_fraction: float,
                      impact_window: int,
                      vol_window: int) -> dict:
    logger = sim.logger
    summary = logger.summary()
    depth_series = [depth.get('total', float('nan')) for depth in logger.clob_depth]
    realized_vol = _rolling_return_vol(logger.clob_mid_series, window=vol_window)
    amm_flow_share = sum(
        float(summary.get(f'avg_flow_share_{venue}', 0.0))
        for venue in logger.amm_cost_curves
    )
    metrics = {
        'n_trades': float(summary.get('n_trades', 0)),
        'avg_clob_spread_bps': _mean_finite(logger.clob_qspr),
        'avg_clob_depth': _mean_finite(depth_series),
        'avg_systemic_liquidity': _mean_finite(logger.systemic_liquidity_series),
        'avg_realized_volatility': _mean_finite(realized_vol),
        'tail_realized_volatility': _tail_mean(realized_vol),
        'avg_cost_clob_q5': float(summary.get('avg_cost_clob_Q5', float('nan'))),
        'avg_flow_share_amm_total': amm_flow_share,
        'avg_mm_share_withdrawn': float(summary.get('avg_mm_share_withdrawn', float('nan'))),
        'avg_mm_withdrawal_score': float(summary.get('avg_mm_withdrawal_score', float('nan'))),
    }
    shock_iter = _shock_iter_from_sim(sim)
    if shock_iter is None:
        return metrics

    horizon_end = min(len(logger.iterations), shock_iter + max(50, resilience_horizon))
    target_mode = PRICE_TARGET_MODE.get(scenario_key, 'pre_shock_baseline')
    target_series = logger.fair_price_series if target_mode == 'fair_value_gap' else None

    price_metrics = price_resilience_metrics(
        logger.clob_mid_series,
        shock_iter=shock_iter,
        baseline_window=50,
        avg_window=avg_window,
        stable_window=stable_window,
        horizon=max(50, resilience_horizon),
        retracement_fraction=retracement_fraction,
        impact_window=impact_window,
        target_series=target_series,
        target_mode=target_mode,
    )
    spread_metrics = series_resilience_metrics(
        logger.clob_qspr,
        shock_iter=shock_iter,
        baseline_window=50,
        avg_window=avg_window,
        stable_window=stable_window,
        horizon=max(50, resilience_horizon),
        retracement_fraction=retracement_fraction,
        impact_window=impact_window,
        analysis_target_mode='pre_shock_baseline',
    )
    depth_metrics = series_resilience_metrics(
        depth_series,
        shock_iter=shock_iter,
        baseline_window=50,
        avg_window=avg_window,
        stable_window=stable_window,
        horizon=max(50, resilience_horizon),
        retracement_fraction=retracement_fraction,
        impact_window=impact_window,
        analysis_target_mode='pre_shock_baseline',
    )
    systemic_metrics = series_resilience_metrics(
        logger.systemic_liquidity_series,
        shock_iter=shock_iter,
        baseline_window=50,
        avg_window=avg_window,
        stable_window=stable_window,
        horizon=max(50, resilience_horizon),
        retracement_fraction=retracement_fraction,
        impact_window=impact_window,
        analysis_target_mode='pre_shock_baseline',
    )
    liquidity_metrics = composite_resilience_metrics(
        [spread_metrics, depth_metrics, systemic_metrics],
        analysis_target_mode='liquidity_composite',
    )

    metrics.update({
        'price_recovery_steps': price_metrics.get('recovery_steps', float('nan')),
        'spread_recovery_steps': spread_metrics.get('recovery_steps', float('nan')),
        'depth_recovery_steps': depth_metrics.get('recovery_steps', float('nan')),
        'liquidity_composite_recovery_steps': liquidity_metrics.get('recovery_steps', float('nan')),
        'liquidity_composite_recovered': 1.0 if liquidity_metrics.get('recovered', False) else 0.0,
        'post_shock_avg_clob_spread_bps': _window_mean(logger.clob_qspr, shock_iter, horizon_end),
        'post_shock_avg_clob_depth': _window_mean(depth_series, shock_iter, horizon_end),
        'post_shock_avg_systemic_liquidity': _window_mean(logger.systemic_liquidity_series, shock_iter, horizon_end),
        'post_shock_avg_realized_volatility': _window_mean(realized_vol, shock_iter, horizon_end),
        'post_shock_mm_share_withdrawn': _window_mean(logger.mm_state_shares['withdrawn'], shock_iter, horizon_end),
        'post_shock_mm_withdrawal_score': _window_mean(logger.mm_avg_withdrawal_score_series, shock_iter, horizon_end),
    })
    return metrics


def _collect_points(scenario_key: str, structure_key: str, enable_amm: bool,
                    seeds: int, base_seed: int,
                    n_iter: int, venue_choice_rule: str,
                    resilience_horizon: int,
                    stable_window: int,
                    avg_window: int,
                    retracement_fraction: float,
                    impact_window: int,
                    vol_window: int,
                    progress_every: int = 0) -> list[dict]:
    points = []
    for offset in range(seeds):
        seed = base_seed + offset
        args = _base_args_for_scenario(
            scenario_key,
            n_iter,
            venue_choice_rule=venue_choice_rule,
        )
        args.seed = seed
        args.enable_amm = 1 if enable_amm else 0
        if not enable_amm:
            args.amm_share_pct = 0
        _seed_all(seed)
        sim = build_sim(args)
        sim.simulate(args.n_iter, silent=True)
        metrics = _scenario_metrics(
            sim,
            scenario_key=scenario_key,
            resilience_horizon=resilience_horizon,
            stable_window=stable_window,
            avg_window=avg_window,
            retracement_fraction=retracement_fraction,
            impact_window=impact_window,
            vol_window=vol_window,
        )
        row = {
            'scenario_key': scenario_key,
            'structure_key': structure_key,
            'amm_enabled': 1 if enable_amm else 0,
            'seed': seed,
        }
        row.update(metrics)
        points.append(row)
        done = offset + 1
        if progress_every > 0 and (done == 1 or done % progress_every == 0 or done == seeds):
            print(f'[{scenario_key}:{structure_key}] {done}/{seeds} seeds complete', flush=True)
    return points


def _save_points_csv(rows: list[dict], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'rq_stat_points.csv')
    fieldnames = [
        'scenario_key', 'scenario_label', 'structure_key', 'structure_label', 'amm_enabled', 'seed'
    ] + [name for name, _label, _rq, _shock_only in _metric_spec()]
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _save_wide_summary_csv(rows: list[dict], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'rq_stat_summary.csv')
    fieldnames = [
        'scenario_key', 'scenario_label', 'structure_key', 'structure_label', 'amm_enabled', 'n_seeds'
    ] + [name for name, _label, _rq, _shock_only in _metric_spec()]
    grouped = {}
    for row in rows:
        grouped.setdefault(
            (
                row['scenario_key'],
                row['scenario_label'],
                row['structure_key'],
                row['structure_label'],
                row['amm_enabled'],
            ),
            [],
        ).append(row)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (scenario_key, scenario_label, structure_key, structure_label, amm_enabled), scenario_rows in grouped.items():
            out = {
                'scenario_key': scenario_key,
                'scenario_label': scenario_label,
                'structure_key': structure_key,
                'structure_label': structure_label,
                'amm_enabled': amm_enabled,
                'n_seeds': len(scenario_rows),
            }
            for metric_name, _metric_label, _rq, _shock_only in _metric_spec():
                out[metric_name] = _mean_finite(row.get(metric_name, float('nan')) for row in scenario_rows)
            writer.writerow(out)
    return path


def _save_statistical_summary_csv(rows: list[dict], out_dir: str,
                                  bootstrap_reps: int,
                                  ci_level: float,
                                  base_seed: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'rq_statistical_summary.csv')
    fieldnames = [
        'research_question', 'scenario_key', 'scenario_label', 'structure_key', 'structure_label', 'amm_enabled',
        'metric_name', 'metric_label',
        'n_used', 'mean', 'std', 'median', 'ci_level', 'ci_lo', 'ci_hi',
    ]
    grouped = {}
    for row in rows:
        grouped.setdefault(
            (
                row['scenario_key'],
                row['scenario_label'],
                row['structure_key'],
                row['structure_label'],
                row['amm_enabled'],
            ),
            [],
        ).append(row)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (scenario_key, scenario_label, structure_key, structure_label, amm_enabled), scenario_rows in grouped.items():
            for metric_name, metric_label, research_question, _shock_only in _metric_spec():
                stats = bootstrap_mean_ci(
                    [row.get(metric_name, float('nan')) for row in scenario_rows],
                    n_boot=bootstrap_reps,
                    ci_level=ci_level,
                    seed=_stat_seed(base_seed, scenario_key, metric_name),
                )
                writer.writerow({
                    'research_question': research_question,
                    'scenario_key': scenario_key,
                    'scenario_label': scenario_label,
                    'structure_key': structure_key,
                    'structure_label': structure_label,
                    'amm_enabled': amm_enabled,
                    'metric_name': metric_name,
                    'metric_label': metric_label,
                    'n_used': stats['n'],
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'median': stats['median'],
                    'ci_level': ci_level,
                    'ci_lo': stats['ci_lo'],
                    'ci_hi': stats['ci_hi'],
                })
    return path


def _extract_paired_samples(rows: list[dict], scenario_key: str,
                            metric_name: str) -> tuple[list[float], list[float]]:
    with_rows = {
        row['seed']: row for row in rows
        if row['scenario_key'] == scenario_key and row['structure_key'] == 'with_amm'
    }
    without_rows = {
        row['seed']: row for row in rows
        if row['scenario_key'] == scenario_key and row['structure_key'] == 'without_amm'
    }
    common_seeds = sorted(set(with_rows) & set(without_rows))
    sample_with = []
    sample_without = []
    for seed in common_seeds:
        value_with = with_rows[seed].get(metric_name, float('nan'))
        value_without = without_rows[seed].get(metric_name, float('nan'))
        if math.isfinite(value_with) and math.isfinite(value_without):
            sample_with.append(float(value_with))
            sample_without.append(float(value_without))
    return sample_with, sample_without


def _save_with_without_tests_csv(rows: list[dict], out_dir: str,
                                 bootstrap_reps: int,
                                 ci_level: float,
                                 base_seed: int,
                                 permutation_reps: int) -> tuple[str, list[dict]]:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'rq_with_without_amm_tests.csv')
    fieldnames = [
        'research_question', 'scenario_key', 'scenario_label', 'metric_name', 'metric_label',
        'n_pairs', 'mean_with_amm', 'mean_without_amm', 'mean_delta_with_minus_without',
        'permutation_stat', 'permutation_p_value', 'ci_level', 'delta_ci_lo', 'delta_ci_hi',
    ]
    scenario_labels = {
        row['scenario_key']: row['scenario_label']
        for row in rows
    }
    result_rows = []
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scenario_key, scenario_label in DEFAULT_SCENARIOS:
            for metric_name, metric_label, research_question, shock_only in _metric_spec():
                if shock_only and scenario_key not in PRICE_TARGET_MODE:
                    continue
                sample_with, sample_without = _extract_paired_samples(rows, scenario_key, metric_name)
                paired = paired_permutation_test(
                    sample_with,
                    sample_without,
                    n_perm=permutation_reps,
                    seed=_stat_seed(base_seed, 'perm', scenario_key, metric_name),
                )
                boot = bootstrap_paired_diff_ci(
                    sample_with,
                    sample_without,
                    n_boot=bootstrap_reps,
                    ci_level=ci_level,
                    seed=_stat_seed(base_seed, scenario_key, metric_name),
                )
                result = {
                    'research_question': research_question,
                    'scenario_key': scenario_key,
                    'scenario_label': scenario_labels.get(scenario_key, scenario_label),
                    'metric_name': metric_name,
                    'metric_label': metric_label,
                    'n_pairs': paired['n'],
                    'mean_with_amm': paired['mean_a'],
                    'mean_without_amm': paired['mean_b'],
                    'mean_delta_with_minus_without': paired['mean_diff'],
                    'permutation_stat': paired['stat'],
                    'permutation_p_value': paired['p_value'],
                    'ci_level': ci_level,
                    'delta_ci_lo': boot['ci_lo'],
                    'delta_ci_hi': boot['ci_hi'],
                }
                writer.writerow(result)
                result_rows.append(result)
    return path, result_rows


def _save_question_views(test_rows: list[dict], out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    fieldnames = [
        'research_question', 'scenario_key', 'scenario_label', 'metric_name', 'metric_label',
        'n_pairs', 'mean_with_amm', 'mean_without_amm', 'mean_delta_with_minus_without',
        'permutation_stat', 'permutation_p_value', 'ci_level', 'delta_ci_lo', 'delta_ci_hi',
    ]
    out_paths = []
    for question in ('RQ1', 'RQ2'):
        path = os.path.join(out_dir, f'{question.lower()}_tests.csv')
        with open(path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in test_rows:
                if row['research_question'] == question:
                    writer.writerow(row)
        out_paths.append(path)
    return out_paths


def _save_effect_dashboards(test_rows: list[dict], out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    label_map = _metric_label_map()
    out_paths = []

    for question, metric_names in RQ_FIGURE_METRICS.items():
        ncols = 2
        nrows = math.ceil(len(metric_names) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(7, 3.0 * nrows)), constrained_layout=True)
        axes_flat = list(axes.flat) if hasattr(axes, 'flat') else [axes]

        for ax, metric_name in zip(axes_flat, metric_names):
            metric_rows = [
                row for row in test_rows
                if row['research_question'] == question and row['metric_name'] == metric_name
            ]
            metric_rows.sort(key=lambda row: _scenario_rank(row['scenario_key']))
            metric_rows = [
                row for row in metric_rows
                if math.isfinite(row.get('mean_delta_with_minus_without', float('nan')))
            ]

            if not metric_rows:
                ax.set_axis_off()
                continue

            ax.axvline(0.0, color='#222222', lw=1.0, alpha=0.7)
            y_positions = list(range(len(metric_rows)))
            labels = [
                row['scenario_label'] + _significance_marker(row.get('permutation_p_value', float('nan')))
                for row in metric_rows
            ]
            deltas = [float(row['mean_delta_with_minus_without']) for row in metric_rows]

            x_extent = []
            for idx, row in enumerate(metric_rows):
                delta = deltas[idx]
                ci_lo = float(row.get('delta_ci_lo', float('nan')))
                ci_hi = float(row.get('delta_ci_hi', float('nan')))
                color = _effect_color(metric_name, delta)
                if math.isfinite(ci_lo) and math.isfinite(ci_hi):
                    err_lo = max(0.0, delta - ci_lo)
                    err_hi = max(0.0, ci_hi - delta)
                    ax.errorbar(
                        delta,
                        idx,
                        xerr=[[err_lo], [err_hi]],
                        fmt='o',
                        color=color,
                        ecolor=color,
                        elinewidth=2.0,
                        capsize=3,
                        markersize=6,
                    )
                    x_extent.extend([ci_lo, ci_hi])
                else:
                    ax.plot(delta, idx, 'o', color=color, markersize=6)
                x_extent.append(delta)

            finite_extent = [value for value in x_extent if math.isfinite(value)]
            if finite_extent:
                x_abs = max(abs(min(finite_extent)), abs(max(finite_extent)))
                x_pad = 0.12 * x_abs if x_abs > 0 else 1.0
                ax.set_xlim(-x_abs - x_pad, x_abs + x_pad)

            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels, fontsize=9)
            ax.invert_yaxis()
            ax.grid(axis='x', linestyle='--', alpha=0.25)
            ax.set_facecolor('#fbfbf8')
            ax.set_title(label_map.get(metric_name, metric_name), fontsize=11)
            ax.set_xlabel('Delta: with AMM - without AMM')

        for ax in axes_flat[len(metric_names):]:
            ax.set_axis_off()

        fig.suptitle(
            f'{question} Effect Dashboard\nGreen = favorable effect, red = adverse effect, * p < 0.05, ** p < 0.01',
            fontsize=15,
            fontweight='bold',
        )
        path = os.path.join(out_dir, f'{question.lower()}_effect_dashboard.png')
        fig.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        out_paths.append(path)

    return out_paths


def _save_heatmap_summary(test_rows: list[dict], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    label_map = _metric_label_map()
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    axes_flat = list(axes.flat) if hasattr(axes, 'flat') else [axes]
    image = None

    for ax, question in zip(axes_flat, ('RQ1', 'RQ2')):
        metric_names = RQ_FIGURE_METRICS[question]
        scenario_rows = [
            (scenario_key, scenario_label)
            for scenario_key, scenario_label in DEFAULT_SCENARIOS
            if any(
                row['research_question'] == question and row['scenario_key'] == scenario_key
                for row in test_rows
            )
        ]

        heatmap_values = []
        annotation_values = []
        for scenario_key, _scenario_label in scenario_rows:
            heatmap_row = []
            annotation_row = []
            for metric_name in metric_names:
                match = next(
                    (
                        row for row in test_rows
                        if row['research_question'] == question
                        and row['scenario_key'] == scenario_key
                        and row['metric_name'] == metric_name
                    ),
                    None,
                )
                if match is None:
                    heatmap_row.append(0.0)
                    annotation_row.append('')
                    continue
                delta = float(match.get('mean_delta_with_minus_without', float('nan')))
                p_value = float(match.get('permutation_p_value', float('nan')))
                heatmap_row.append(_heatmap_score(metric_name, delta, p_value))
                annotation_row.append(_format_delta(delta, p_value))
            heatmap_values.append(heatmap_row)
            annotation_values.append(annotation_row)

        image = ax.imshow(
            heatmap_values,
            aspect='auto',
            cmap='RdYlGn',
            vmin=-2.0,
            vmax=2.0,
        )
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(
            [label_map.get(metric_name, metric_name) for metric_name in metric_names],
            rotation=30,
            ha='right',
            fontsize=9,
        )
        ax.set_yticks(range(len(scenario_rows)))
        ax.set_yticklabels([scenario_label for _scenario_key, scenario_label in scenario_rows], fontsize=10)
        ax.set_title(f'{question} Compact Heatmap Summary', fontsize=13, fontweight='bold')

        for y_idx, row in enumerate(annotation_values):
            for x_idx, text in enumerate(row):
                value = heatmap_values[y_idx][x_idx]
                text_color = '#111111' if abs(value) < 1.25 else 'white'
                ax.text(x_idx, y_idx, text, ha='center', va='center', fontsize=8, color=text_color)

    if image is not None:
        cbar = fig.colorbar(image, ax=axes_flat, shrink=0.9, pad=0.02)
        cbar.set_label('Adverse  <  direction + statistical support  >  Favorable', fontsize=10)
        cbar.set_ticks([-2.0, -1.0, 0.0, 1.0, 2.0])
        cbar.set_ticklabels(['strong', 'moderate', 'neutral', 'moderate', 'strong'])

    fig.suptitle(
        'AMM Effect Heatmap Summary\nCell color = favorable/adverse effect weighted by p-value; text = delta with significance markers',
        fontsize=15,
        fontweight='bold',
    )
    path = os.path.join(out_dir, 'rq_effect_heatmap_summary.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate RQ-oriented multi-seed tests for AMM effects on liquidity, resilience, and volatility.',
    )
    parser.add_argument('--seeds', type=int, default=200,
                        help='Number of sequential seeds per scenario (default: 200)')
    parser.add_argument('--base-seed', type=int, default=42,
                        help='Starting seed (default: 42)')
    parser.add_argument('--n-iter', type=int, default=900,
                        help='Simulation length for each run (default: 900)')
    parser.add_argument('--bootstrap-reps', type=int, default=2000,
                        help='Bootstrap repetitions for confidence intervals (default: 2000)')
    parser.add_argument('--permutation-reps', type=int, default=10000,
                        help='Permutation/sign-flip repetitions for paired p-values (default: 10000)')
    parser.add_argument('--ci-level', type=float, default=0.95,
                        help='Confidence level for bootstrap intervals (default: 0.95)')
    parser.add_argument('--recovery-horizon', type=int, default=600,
                        help='Post-shock window used for resilience metrics (default: 600)')
    parser.add_argument('--stable-window', type=int, default=10,
                        help='Consecutive periods required inside the recovery band (default: 10)')
    parser.add_argument('--avg-window', type=int, default=20,
                        help='Window used for average post-shock impact metrics (default: 20)')
    parser.add_argument('--retracement-fraction', type=float, default=0.8,
                        help='Fraction of initial dislocation that must be closed to count as recovery (default: 0.8)')
    parser.add_argument('--impact-window', type=int, default=10,
                        help='Early post-shock window used to define the initial dislocation (default: 10)')
    parser.add_argument('--vol-window', type=int, default=20,
                        help='Rolling window for realized-volatility estimation (default: 20)')
    parser.add_argument('--progress-every', type=int, default=25,
                        help='Print progress every N seeds per scenario (default: 25, 0 disables progress output)')
    parser.add_argument('--venue-choice-rule', choices=['fixed_share', 'liquidity_aware'], default='liquidity_aware',
                        help='Use one routing regime consistently across all scenarios (default: liquidity_aware)')
    parser.add_argument('--out-dir', default='output/scenario_stats',
                        help='Output directory for CSV summaries (default: output/scenario_stats)')
    args = parser.parse_args()

    all_rows = []
    for scenario_key, scenario_label in DEFAULT_SCENARIOS:
        print(f'Starting scenario: {scenario_label}', flush=True)
        for structure_key, structure_label, enable_amm in MARKET_STRUCTURES:
            rows = _collect_points(
                scenario_key=scenario_key,
                structure_key=structure_key,
                enable_amm=enable_amm,
                seeds=args.seeds,
                base_seed=args.base_seed,
                n_iter=args.n_iter,
                venue_choice_rule=args.venue_choice_rule,
                resilience_horizon=args.recovery_horizon,
                stable_window=args.stable_window,
                avg_window=args.avg_window,
                retracement_fraction=args.retracement_fraction,
                impact_window=args.impact_window,
                vol_window=args.vol_window,
                progress_every=args.progress_every,
            )
            for row in rows:
                row['scenario_label'] = scenario_label
                row['structure_label'] = structure_label
            all_rows.extend(rows)

    points_path = _save_points_csv(all_rows, out_dir=args.out_dir)
    summary_path = _save_wide_summary_csv(all_rows, out_dir=args.out_dir)
    stats_path = _save_statistical_summary_csv(
        all_rows,
        out_dir=args.out_dir,
        bootstrap_reps=args.bootstrap_reps,
        ci_level=args.ci_level,
        base_seed=args.base_seed,
    )
    with_without_path, test_rows = _save_with_without_tests_csv(
        all_rows,
        out_dir=args.out_dir,
        bootstrap_reps=args.bootstrap_reps,
        ci_level=args.ci_level,
        base_seed=args.base_seed,
        permutation_reps=args.permutation_reps,
    )
    question_paths = _save_question_views(test_rows, out_dir=args.out_dir)
    dashboard_paths = _save_effect_dashboards(test_rows, out_dir=args.out_dir)
    heatmap_path = _save_heatmap_summary(test_rows, out_dir=args.out_dir)

    print(f'Saved raw RQ points to {points_path}')
    print(f'Saved RQ summary to {summary_path}')
    print(f'Saved RQ statistical summary to {stats_path}')
    print(f'Saved with/without AMM tests to {with_without_path}')
    for path in question_paths:
        print(f'Saved research-question view to {path}')
    for path in dashboard_paths:
        print(f'Saved effect dashboard to {path}')
    print(f'Saved compact heatmap summary to {heatmap_path}')


if __name__ == '__main__':
    main()