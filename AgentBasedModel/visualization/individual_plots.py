"""Per-scenario, per-metric individual plots used by the resilience and
stat_tests pipelines.

Every function in this module produces a SINGLE PNG focused on one quantity.
Composite "dashboards" are intentionally not used here; the regenerator and
the test runners call these functions individually so the user can browse
plots one at a time without having to crop them out of a 2x2 grid.
"""
from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from AgentBasedModel.metrics.resilience import kaplan_meier_curve

WITH_AMM_COLOR = '#1f77b4'
WITHOUT_AMM_COLOR = '#d62728'
WITH_AMM_LIGHT = '#9ecae1'
WITHOUT_AMM_LIGHT = '#fcae91'
NEUTRAL_COLOR = '#444444'
GRID_COLOR = '#cccccc'
DPI = 180
FIG_W = 10.0
FIG_H = 6.0

SCENARIOS = (
    ('mm_withdrawal', 'MM Withdrawal (Isolated)'),
    ('flash_crash', 'Flash Crash'),
    ('dealer_liquidity_crisis', 'Dealer Liquidity Crisis'),
    ('funding_liquidity_shock', 'Funding Liquidity Shock'),
    ('high_vol_stress', 'High-Vol Stress'),
)
PRIORITY_METRICS = (
    ('spread_resilience', 'Quoted Spread'),
    ('depth_resilience', 'CLOB Depth'),
    ('execution_cost_resilience', 'Execution Cost (Q=5)'),
    ('composite_resilience', 'Composite Liquidity'),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value) -> float:
    try:
        if value is None or value == '':
            return float('nan')
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _finite(values: Iterable[float]) -> List[float]:
    return [v for v in values if math.isfinite(v)]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _slug(text: str) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in text.lower()).strip('_')


def _save_and_close(fig, path: str) -> str:
    _ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def _scenario_label(scenario_key: str) -> str:
    for key, label in SCENARIOS:
        if key == scenario_key:
            return label
    return scenario_key


def _scenario_short(scenario_key: str) -> str:
    return {
        'mm_withdrawal': 'MM Wd',
        'flash_crash': 'Flash',
        'dealer_liquidity_crisis': 'Dealer Liq',
        'funding_liquidity_shock': 'Funding',
        'high_vol_stress': 'High Vol',
        'default': 'Default',
    }.get(scenario_key, scenario_key)


# ---------------------------------------------------------------------------
# Resilience: per-(scenario, metric) scatter
# ---------------------------------------------------------------------------

def plot_recovery_scatter(panel_label: str,
                          metric_label: str,
                          points_with: Sequence[Mapping],
                          points_without: Sequence[Mapping],
                          out_path: str) -> str:
    """Scatter of (time-to-80%-retracement, normalized average impact) per seed.

    One panel; with-AMM and without-AMM overlaid with distinct markers.
    Recovered seeds = filled circles; censored seeds = open triangles.
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    series = (
        ('With AMM', points_with, WITH_AMM_COLOR),
        ('Without AMM', points_without, WITHOUT_AMM_COLOR),
    )

    stats_lines = []
    all_y = []
    all_rec_x = []

    for label, pts, color in series:
        rec_x, rec_y, cen_x, cen_y = [], [], [], []
        for p in pts:
            x = _safe_float(p.get('time_observed_steps', p.get('recovery_steps')))
            y = _safe_float(p.get('normalized_avg_impact'))
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            recovered = bool(p.get('recovered', False))
            if isinstance(p.get('recovered'), str):
                recovered = p['recovered'].lower() in ('true', '1')
            if recovered:
                rec_x.append(x); rec_y.append(y)
            else:
                cen_x.append(x); cen_y.append(y)
        if rec_x:
            ax.scatter(rec_x, rec_y, s=28, alpha=0.85, color=color,
                       edgecolors='white', linewidths=0.4,
                       label=f'{label} (recovered, n={len(rec_x)})')
        if cen_x:
            ax.scatter(cen_x, cen_y, s=42, alpha=0.85, facecolors='none',
                       edgecolors=color, linewidths=1.2, marker='^',
                       label=f'{label} (censored, n={len(cen_x)})')
        all_y.extend(rec_y + cen_y)
        all_rec_x.extend(rec_x)
        n_total = len(rec_x) + len(cen_x)
        if n_total:
            rec_share = len(rec_x) / n_total
            med = float(np.median(rec_x)) if rec_x else float('nan')
            mean_y = float(np.mean(rec_y + cen_y)) if (rec_y + cen_y) else float('nan')
            stats_lines.append(
                f'{label}: n={n_total} | recovered={rec_share:.0%} | '
                f'med={med:.0f} | impact={mean_y:+.2f}'
            )

    ax.axhline(0.0, color='black', lw=0.8)
    if all_y:
        ax.axhline(float(np.mean(all_y)), color='crimson', ls='--', lw=0.8, alpha=0.6,
                   label=f'pooled mean impact = {np.mean(all_y):+.2f}')
    if all_rec_x:
        ax.axvline(float(np.median(all_rec_x)), color=NEUTRAL_COLOR, ls=':', lw=0.8, alpha=0.6,
                   label=f'pooled median recovery = {np.median(all_rec_x):.0f}')

    ax.set_xlabel('Observed time to 80 % retracement (ticks)', fontsize=11)
    ax.set_ylabel('Normalized average impact (avg / peak)', fontsize=11)
    ax.set_title(f'{panel_label} — {metric_label} resilience (per seed)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, color=GRID_COLOR, alpha=0.45, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=8, framealpha=0.85)
    if stats_lines:
        text = '\n'.join(stats_lines)
        ax.text(0.02, 0.02, text, transform=ax.transAxes, ha='left', va='bottom',
                fontsize=8, family='monospace',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.85,
                          edgecolor=GRID_COLOR))

    return _save_and_close(fig, out_path)


# ---------------------------------------------------------------------------
# Resilience: per-(scenario, metric) Kaplan-Meier
# ---------------------------------------------------------------------------

def plot_km_curve(panel_label: str,
                  metric_label: str,
                  points_with: Sequence[Mapping],
                  points_without: Sequence[Mapping],
                  out_path: str) -> str:
    """Kaplan-Meier survival curve for time-to-80%-retracement.

    Y axis = share of seeds that have NOT yet recovered.
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    legend_lines = []
    for label, pts, color in (('With AMM', points_with, WITH_AMM_COLOR),
                              ('Without AMM', points_without, WITHOUT_AMM_COLOR)):
        observed = []
        recovered = []
        for p in pts:
            t = _safe_float(p.get('time_observed_steps', p.get('recovery_steps')))
            r = p.get('recovered')
            if isinstance(r, str):
                r = r.lower() in ('true', '1')
            if math.isfinite(t):
                observed.append(t)
                recovered.append(bool(r))
        if not observed:
            continue
        km = kaplan_meier_curve(observed, recovered)
        times = km['times']
        survival = km['survival']
        ax.step(times, survival, where='post', color=color, lw=2.2, label=label)
        rec_share = sum(1 for r in recovered if r) / len(recovered)
        med = km['median_time']
        med_str = f'{med:.0f}' if math.isfinite(med) else 'NaN'
        legend_lines.append(f'{label}: n={len(observed)} | rec={rec_share:.0%} | KM med={med_str}')

    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Ticks since shock', fontsize=11)
    ax.set_ylabel('Share of seeds NOT yet recovered', fontsize=11)
    ax.set_title(f'{panel_label} — {metric_label} time-to-recovery (KM)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, color=GRID_COLOR, alpha=0.45, linestyle='--')
    ax.set_axisbelow(True)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc='upper right', fontsize=10, framealpha=0.85)

    if legend_lines:
        ax.text(0.02, 0.02, '\n'.join(legend_lines), transform=ax.transAxes,
                ha='left', va='bottom', fontsize=8, family='monospace',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.85,
                          edgecolor=GRID_COLOR))
    return _save_and_close(fig, out_path)


# ---------------------------------------------------------------------------
# Resilience: peak impact magnitude (across scenarios) per metric
# ---------------------------------------------------------------------------

def plot_peak_impact_bars(metric_label: str,
                          per_scenario: Sequence[Mapping],
                          out_path: str,
                          field: str = 'peak_abs_change_pct',
                          y_label_override: Optional[str] = None) -> str:
    """Grouped bar plot: peak |%-deviation| per scenario, with vs without AMM.

    `per_scenario` is a list of dicts:
        {'scenario_key', 'scenario_label', 'with_values' (list), 'without_values' (list)}
    """
    if not per_scenario:
        return ''

    fig, ax = plt.subplots(figsize=(FIG_W + 1, FIG_H))

    n = len(per_scenario)
    x = np.arange(n)
    bar_w = 0.36

    means_with = []
    means_without = []
    stds_with = []
    stds_without = []
    labels = []
    n_pairs = []

    for entry in per_scenario:
        with_vals = _finite(entry['with_values'])
        without_vals = _finite(entry['without_values'])
        means_with.append(float(np.mean(with_vals)) if with_vals else float('nan'))
        means_without.append(float(np.mean(without_vals)) if without_vals else float('nan'))
        stds_with.append(float(np.std(with_vals, ddof=1) / max(1, math.sqrt(len(with_vals)))) if len(with_vals) > 1 else 0.0)
        stds_without.append(float(np.std(without_vals, ddof=1) / max(1, math.sqrt(len(without_vals)))) if len(without_vals) > 1 else 0.0)
        labels.append(entry['scenario_label'])
        n_pairs.append(min(len(with_vals), len(without_vals)))

    bars1 = ax.bar(x - bar_w/2, means_with, bar_w, yerr=stds_with, capsize=4,
                   color=WITH_AMM_COLOR, edgecolor='white', label='With AMM',
                   error_kw=dict(elinewidth=1.2, ecolor='#444444'))
    bars2 = ax.bar(x + bar_w/2, means_without, bar_w, yerr=stds_without, capsize=4,
                   color=WITHOUT_AMM_COLOR, edgecolor='white', label='Without AMM',
                   error_kw=dict(elinewidth=1.2, ecolor='#444444'))

    for xi, mw, mo in zip(x, means_with, means_without):
        if math.isfinite(mw) and math.isfinite(mo):
            delta = mw - mo
            sign = '+' if delta >= 0 else '−'
            ax.annotate(f'Δ={sign}{abs(delta):.0f}',
                        xy=(xi, max(mw, mo)),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=8, color='#222222', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel(y_label_override or 'Mean peak |% deviation| from baseline', fontsize=11)
    ax.set_title(f'Peak shock magnitude — {metric_label}\n(error bars = SE; lower = better)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', color=GRID_COLOR, alpha=0.45, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=10, framealpha=0.85)

    return _save_and_close(fig, out_path)


# ---------------------------------------------------------------------------
# Resilience: paired-delta histogram for normalized_avg_impact
# ---------------------------------------------------------------------------

def plot_paired_delta_strip(metric_label: str,
                            per_scenario: Sequence[Mapping],
                            out_path: str,
                            field: str = 'normalized_avg_impact',
                            y_label: str = 'Δ (with-AMM − without-AMM) of normalized avg impact') -> str:
    """One row per scenario: scatter of paired deltas + mean ± 95 % CI.

    `per_scenario` rows must contain `paired_diffs` (list of (with, without))
    """
    if not per_scenario:
        return ''

    fig, ax = plt.subplots(figsize=(FIG_W + 1, max(FIG_H, 1.2 * len(per_scenario) + 2)))

    y_positions = list(range(len(per_scenario)))
    labels = []
    means = []
    cis = []
    counts = []

    rng = np.random.default_rng(0)

    for idx, entry in enumerate(per_scenario):
        diffs = [w - o for (w, o) in entry['paired_diffs']
                 if math.isfinite(w) and math.isfinite(o)]
        labels.append(entry['scenario_label'])
        if not diffs:
            means.append(float('nan'))
            cis.append((float('nan'), float('nan')))
            counts.append(0)
            continue
        diffs_np = np.array(diffs)
        # bootstrap CI
        boot = []
        for _ in range(2000):
            sample = rng.choice(diffs_np, size=len(diffs_np), replace=True)
            boot.append(sample.mean())
        ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        means.append(float(diffs_np.mean()))
        cis.append((ci_lo, ci_hi))
        counts.append(len(diffs))

        jitter = rng.uniform(-0.18, 0.18, size=len(diffs))
        ax.scatter(diffs, np.full(len(diffs), idx) + jitter,
                   s=10, alpha=0.35, color=WITH_AMM_COLOR if means[-1] < 0 else WITHOUT_AMM_COLOR,
                   edgecolors='none')

    ax.axvline(0.0, color='black', lw=1.0)
    for idx, (m, (lo, hi), n) in enumerate(zip(means, cis, counts)):
        if not math.isfinite(m):
            continue
        color = '#1a9641' if m < 0 else '#d7191c'
        ax.errorbar(m, idx, xerr=[[m - lo], [hi - m]],
                    fmt='o', color=color, ecolor=color,
                    markersize=8, elinewidth=2.2, capsize=5)
        ax.text(m, idx + 0.35, f'mean={m:+.3f} (n={n})',
                ha='center', va='bottom', fontsize=8, color='#222222')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(y_label, fontsize=11)
    ax.set_title(f'Paired with-vs-without AMM Δ — {metric_label}\n(left = AMM helps, dots = per-seed deltas, bars = mean ± 95 % CI)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', color=GRID_COLOR, alpha=0.45, linestyle='--')
    ax.set_axisbelow(True)

    return _save_and_close(fig, out_path)


# ---------------------------------------------------------------------------
# Resilience: recovery-time boxplot per metric
# ---------------------------------------------------------------------------

def plot_recovery_boxplot(metric_label: str,
                          per_scenario: Sequence[Mapping],
                          out_path: str,
                          field: str = 'recovery_steps') -> str:
    """Side-by-side boxplot per scenario: recovery times for with-AMM vs without."""
    if not per_scenario:
        return ''

    fig, ax = plt.subplots(figsize=(FIG_W + 1, FIG_H))

    n = len(per_scenario)
    positions_with = np.arange(n) - 0.18
    positions_without = np.arange(n) + 0.18
    labels = []
    box_with = []
    box_without = []

    for entry in per_scenario:
        labels.append(entry['scenario_label'])
        box_with.append(_finite(entry['with_values']))
        box_without.append(_finite(entry['without_values']))

    bp1 = ax.boxplot(box_with, positions=positions_with, widths=0.32, patch_artist=True,
                     showfliers=False, medianprops=dict(color='black', lw=1.2))
    bp2 = ax.boxplot(box_without, positions=positions_without, widths=0.32, patch_artist=True,
                     showfliers=False, medianprops=dict(color='black', lw=1.2))
    for box in bp1['boxes']:
        box.set_facecolor(WITH_AMM_LIGHT)
        box.set_edgecolor(WITH_AMM_COLOR)
        box.set_linewidth(1.4)
    for box in bp2['boxes']:
        box.set_facecolor(WITHOUT_AMM_LIGHT)
        box.set_edgecolor(WITHOUT_AMM_COLOR)
        box.set_linewidth(1.4)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel(f'Time to 80 % retracement (ticks)', fontsize=11)
    ax.set_title(f'Recovery time distribution — {metric_label}\n(box = IQR, whiskers = 1.5×IQR, line = median)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', color=GRID_COLOR, alpha=0.45, linestyle='--')
    ax.set_axisbelow(True)

    legend_handles = [plt.Rectangle((0,0),1,1, facecolor=WITH_AMM_LIGHT, edgecolor=WITH_AMM_COLOR, lw=1.4),
                      plt.Rectangle((0,0),1,1, facecolor=WITHOUT_AMM_LIGHT, edgecolor=WITHOUT_AMM_COLOR, lw=1.4)]
    ax.legend(legend_handles, ['With AMM', 'Without AMM'], loc='best', fontsize=10, framealpha=0.85)

    return _save_and_close(fig, out_path)


# ---------------------------------------------------------------------------
# Stat tests: per-(question, metric) effect plot
# ---------------------------------------------------------------------------

def plot_rq_effect(question: str,
                   metric_label: str,
                   metric_name: str,
                   rows: Sequence[Mapping],
                   out_path: str,
                   higher_is_better_metrics: Optional[set] = None,
                   lower_is_better_metrics: Optional[set] = None) -> str:
    """Single-metric paired-Δ dotplot across scenarios."""
    if not rows:
        return ''
    fig, ax = plt.subplots(figsize=(FIG_W, max(FIG_H, 0.7 * len(rows) + 2)))

    higher_is_better = higher_is_better_metrics or set()
    lower_is_better = lower_is_better_metrics or set()

    deltas = []
    cis = []
    labels = []
    sigs = []
    for row in rows:
        d = _safe_float(row.get('mean_delta_with_minus_without'))
        ci_lo = _safe_float(row.get('delta_ci_lo'))
        ci_hi = _safe_float(row.get('delta_ci_hi'))
        p = _safe_float(row.get('permutation_p_value'))
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        deltas.append(d)
        cis.append((ci_lo, ci_hi))
        labels.append(row.get('scenario_label', row.get('scenario_key', '?')))
        sigs.append(sig)

    y_positions = list(range(len(rows)))
    ax.axvline(0.0, color='black', lw=1.0)

    for idx, (d, (lo, hi), sig) in enumerate(zip(deltas, cis, sigs)):
        if not math.isfinite(d):
            continue
        if metric_name in higher_is_better:
            color = '#1a9641' if d > 0 else '#d7191c'
        elif metric_name in lower_is_better:
            color = '#1a9641' if d < 0 else '#d7191c'
        else:
            color = '#404040'
        if math.isfinite(lo) and math.isfinite(hi):
            ax.errorbar(d, idx, xerr=[[d - lo], [hi - d]],
                        fmt='o', color=color, ecolor=color,
                        markersize=8, elinewidth=2.2, capsize=5)
        else:
            ax.plot(d, idx, 'o', color=color, markersize=8)
        ax.text(d, idx + 0.30, f'{d:+.3g} ({sig})',
                ha='center', va='bottom', fontsize=8, color='#222222')

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'{lab}' for lab in labels])
    ax.invert_yaxis()
    ax.set_xlabel('Δ = mean(with AMM) − mean(without AMM)', fontsize=11)
    if metric_name in higher_is_better:
        legend_text = 'green = AMM increases (good), red = AMM decreases (bad)'
    elif metric_name in lower_is_better:
        legend_text = 'green = AMM decreases (good), red = AMM increases (bad)'
    else:
        legend_text = 'directional only — no a-priori "good" direction'
    ax.set_title(f'{question} — {metric_label}\n{legend_text}\n* p<0.05, ** p<0.01, *** p<0.001',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='x', color=GRID_COLOR, alpha=0.45, linestyle='--')
    ax.set_axisbelow(True)

    return _save_and_close(fig, out_path)


# ---------------------------------------------------------------------------
# Stat tests: per-metric H2 phase line plot
# ---------------------------------------------------------------------------

def plot_h2_phase(metric_name: str,
                  metric_label: str,
                  scenario_data: Sequence[Mapping],
                  out_path: str) -> str:
    """Line plot per scenario across before/during/after phases for a single metric."""
    if not scenario_data:
        return ''
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    palette = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    phases = ['before', 'during', 'after']
    x = np.arange(len(phases))

    is_share = 'share' in metric_name

    for idx, entry in enumerate(scenario_data):
        values = [entry.get(f'{metric_name}_{ph}', float('nan')) for ph in phases]
        values = [_safe_float(v) for v in values]
        if not any(math.isfinite(v) for v in values):
            continue
        if is_share:
            values = [100.0 * v if math.isfinite(v) else float('nan') for v in values]
        color = palette[idx % len(palette)]
        ax.plot(x, values, marker='o', lw=2.4, color=color,
                label=entry.get('scenario_label', entry.get('scenario_key', '?')))
        for xi, vi in zip(x, values):
            if math.isfinite(vi):
                fmt = f'{vi:.1f}{"%" if is_share else ""}'
                ax.annotate(fmt, xy=(xi, vi), xytext=(0, 6),
                            textcoords='offset points', ha='center',
                            fontsize=8, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['Before shock', 'During shock', 'After shock'])
    ax.set_ylabel('Percent' if is_share else 'Value', fontsize=11)
    ax.set_title(f'H2 phase trajectory — {metric_label}\n(per-scenario means, with-AMM arm)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, color=GRID_COLOR, alpha=0.45, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=9, framealpha=0.85, ncol=2)

    return _save_and_close(fig, out_path)


# ---------------------------------------------------------------------------
# Stat tests: MM behavior heatmap (kept compact, single PNG)
# ---------------------------------------------------------------------------

def plot_mm_behavior_heatmap(rows: Sequence[Mapping],
                             out_path: str) -> str:
    metric_names = [
        'post_shock_mm_share_active',
        'post_shock_mm_share_defensive',
        'post_shock_mm_share_withdrawn',
        'post_shock_mm_withdrawal_score',
    ]
    metric_labels = ['MM Active Share', 'MM Defensive Share', 'MM Withdrawn Share', 'MM Withdrawal Score']
    scenario_keys = [k for k, _ in SCENARIOS]
    scenario_labels = [_scenario_label(k) for k in scenario_keys]

    grid = np.full((len(scenario_keys), len(metric_names)), np.nan)
    annot = [['' for _ in metric_names] for _ in scenario_keys]
    for sc_idx, sc_key in enumerate(scenario_keys):
        for m_idx, m_name in enumerate(metric_names):
            match = next((r for r in rows
                          if r.get('scenario_key') == sc_key and r.get('metric_name') == m_name), None)
            if not match:
                continue
            d = _safe_float(match.get('mean_delta_with_minus_without'))
            p = _safe_float(match.get('permutation_p_value'))
            grid[sc_idx, m_idx] = d
            sig = ' ***' if p < 0.001 else ' **' if p < 0.01 else ' *' if p < 0.05 else ''
            annot[sc_idx][m_idx] = f'{d:+.3g}{sig}'

    fig, ax = plt.subplots(figsize=(FIG_W, max(FIG_H, 0.7 * len(scenario_keys) + 2)))
    abs_max = max(0.01, np.nanmax(np.abs(grid))) if not np.all(np.isnan(grid)) else 0.01
    im = ax.imshow(grid, aspect='auto', cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_labels, rotation=20, ha='right')
    ax.set_yticks(range(len(scenario_keys)))
    ax.set_yticklabels(scenario_labels)
    for i in range(len(scenario_keys)):
        for j in range(len(metric_names)):
            text = annot[i][j]
            if text:
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=8, color='black')
    ax.set_title('MM Behavior: with-AMM minus without-AMM\n(positive = AMM increases this share/score; * p<0.05, ** p<0.01, *** p<0.001)',
                 fontsize=12, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Δ (with AMM − without AMM)', fontsize=10)

    return _save_and_close(fig, out_path)
