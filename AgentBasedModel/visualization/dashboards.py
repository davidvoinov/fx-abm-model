"""
dashboards.py — Multi-panel dashboard figures for the FX ABM.

Each function produces a single high-resolution figure combining several
related plots. All dashboards are saved to *out_dir* as PNG files.

    dashboard_context()     — environment + price + spread overview
    dashboard_h1()          — H1: cost comparison & AMM adds liquidity
    dashboard_h2()          — H2: systemic linkage & liquidity flight
    generate_all_dashboards() — convenience wrapper
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

if TYPE_CHECKING:
    from AgentBasedModel.metrics.logger import MetricsLogger

from AgentBasedModel.visualization.venue_plots import (
    _avg_finite, _rolling_avg, _shade_stress,
)

# Shared palette
_C = {
    'clob': '#1f77b4',
    'cpmm': '#ff7f0e',
    'hfmm': '#2ca02c',
    'stress': '#c44e52',
    'normal': '#4c72b0',
    'slip': '#4c72b0',
    'fee': '#dd8452',
}

DPI = 200


# ===================================================================
#  Dashboard 0 — Context overview
# ===================================================================

def dashboard_context(logger: 'MetricsLogger',
                      out_dir: str = 'output',
                      rolling: int = 5) -> str:
    """
    3-panel context dashboard:
      top-left:  σ_t and c_t  (twin-axis)
      top-right: FX mid-price
      bottom:    CLOB quoted spread
    """
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

    fig.suptitle('Market Context Dashboard', fontsize=16, fontweight='bold',
                 y=0.97)

    # ── σ_t / c_t ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Volatility  $\\sigma_t$  &  Funding Cost  $c_t$')
    l1 = ax1.plot(logger.sigma_series, color='tab:red', label='$\\sigma_t$',
                  linewidth=1.2)
    ax1.set_ylabel('$\\sigma_t$', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1b = ax1.twinx()
    l2 = ax1b.plot(logger.c_series, color='tab:blue', label='$c_t$',
                   linewidth=1.2)
    ax1b.set_ylabel('$c_t$', color='tab:blue')
    ax1b.tick_params(axis='y', labelcolor='tab:blue')
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=10,
               loc='upper left')
    ax1.set_xlabel('Iteration')
    _shade_stress(ax1, logger)
    ax1.grid(True, alpha=0.25)

    # ── FX Price ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title(f'FX Mid-Price (CLOB)  [MA {rolling}]')
    s = _rolling_avg(logger.clob_mid_series, rolling)
    ax2.plot(s, color='black', linewidth=1)
    _shade_stress(ax2, logger)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Price')
    ax2.grid(True, alpha=0.25)

    # ── CLOB Spread ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_title(f'CLOB Quoted Spread  [MA {rolling}]')
    s = _rolling_avg(logger.clob_qspr, rolling)
    ax3.plot(s, color='black', linewidth=1)
    _shade_stress(ax3, logger)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Spread (bps)')
    ax3.grid(True, alpha=0.25)

    path = os.path.join(out_dir, 'dashboard_context.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path


# ===================================================================
#  Dashboard 1 — H1: cost comparison & AMM adds liquidity
# ===================================================================

def dashboard_h1(logger: 'MetricsLogger',
                 out_dir: str = 'output',
                 Q: float = 5,
                 rolling: int = 10) -> str:
    """
    4-panel H1 dashboard:
      top-left:   C_v(Q) cost curves
      top-right:  cost decomposition (slippage vs fee)
      bottom-left:  total market depth (CLOB + AMM stacked)
      bottom-right: summary KPI text
    """
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    fig.suptitle('H1 · Execution Cost Comparison  &  AMM Adds Liquidity',
                 fontsize=16, fontweight='bold', y=0.97)

    Q_grid = logger.Q_grid

    # ── Cost curves ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Average Execution Cost  $C_v(Q)$')
    means = _avg_finite(logger.clob_cost_curves, Q_grid)
    ax1.plot(Q_grid, means, 'o-', label='CLOB', lw=2, color=_C['clob'])
    for name in logger.amm_cost_curves:
        m = _avg_finite(logger.amm_cost_curves[name], Q_grid)
        ax1.plot(Q_grid, m, 's--', label=name.upper(), lw=2,
                 color=_C.get(name))
    ax1.set_xlabel('Trade Size  $Q$')
    ax1.set_ylabel('Cost (bps)')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.25)

    # ── Cost decomposition ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title(f'Cost Decomposition  (Q={Q})')

    venues, slippages, fees = [], [], []
    s = logger.clob_cost_curves.get(Q, [])
    fs = [x for x in s if math.isfinite(x)]
    if fs:
        venues.append('CLOB')
        slippages.append(sum(fs) / len(fs))
        fees.append(0)
    for name in logger.amm_slippage_curves:
        sl = logger.amm_slippage_curves[name].get(Q, [])
        fe = logger.amm_fee_curves[name].get(Q, [])
        fsl = [x for x in sl if math.isfinite(x)]
        ffe = [x for x in fe if math.isfinite(x)]
        if fsl:
            venues.append(name.upper())
            slippages.append(sum(fsl) / len(fsl))
            fees.append(sum(ffe) / len(ffe) if ffe else 0)

    xp = list(range(len(venues)))
    ax2.bar(xp, slippages, label='Slippage', alpha=0.85, color=_C['slip'])
    ax2.bar(xp, fees, bottom=slippages, label='LP Fee', alpha=0.85,
            color=_C['fee'])
    ax2.set_xticks(xp)
    ax2.set_xticklabels(venues)
    ax2.set_ylabel('Cost (bps)')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.25, axis='y')

    # ── Total market depth ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title(f'Total Market Depth: CLOB + AMM  [MA {rolling}]')
    n = len(logger.iterations)
    clob_d = []
    for d in logger.clob_depth:
        clob_d.append((d.get('bid', 0) + d.get('ask', 0))
                      if isinstance(d, dict) else 0)
    clob_r = _rolling_avg(clob_d, rolling)
    ax3.fill_between(range(n), 0, clob_r, alpha=0.4, label='CLOB',
                     color=_C['clob'])
    amm_total = [0.0] * n
    for name in logger.amm_L_series:
        v = logger.amm_L_series[name][:n]
        while len(v) < n:
            v.append(0)
        amm_total = [a + b for a, b in zip(amm_total, v)]
    amm_r = _rolling_avg(amm_total, rolling)
    combined = [c + a for c, a in zip(clob_r, amm_r)]
    ax3.fill_between(range(n), clob_r, combined, alpha=0.4,
                     label='AMM (added)', color=_C['hfmm'])
    _shade_stress(ax3, logger)
    ax3.legend(fontsize=10)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Depth (units)')
    ax3.grid(True, alpha=0.25)

    # ── KPI summary panel ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    summary = logger.summary()
    venues_all = ['clob'] + list(logger.amm_cost_curves.keys())

    lines = []
    lines.append(f'Iterations: {summary.get("n_iterations", 0)}')
    lines.append(f'Total trades: {summary.get("n_trades", 0)}')
    lines.append('')
    lines.append('Avg Cost (bps):')
    hdr = f'  {"Q":>4s}'
    for v in venues_all:
        hdr += f'  {v.upper():>7s}'
    lines.append(hdr)
    for q in Q_grid:
        row = f'  {q:>4.0f}'
        for v in venues_all:
            val = summary.get(f'avg_cost_{v}_Q{q}', float('nan'))
            row += f'  {val:>7.1f}' if math.isfinite(val) else f'  {"N/A":>7s}'
        lines.append(row)
    lines.append('')
    lines.append('Flow Share:')
    amm_share = 0.0
    for v in venues_all:
        fs = summary.get(f'avg_flow_share_{v}', 0)
        lines.append(f'  {v.upper():>7s}: {fs:.1%}')
        if v != 'clob':
            amm_share += fs
    lines.append(f'\n→ AMM captures {amm_share:.1%} of volume')

    ax4.text(0.05, 0.95, '\n'.join(lines), transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                       alpha=0.8))

    path = os.path.join(out_dir, 'dashboard_h1.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path


# ===================================================================
#  Dashboard 2 — H2: systemic linkage & liquidity flight
# ===================================================================

def dashboard_h2(logger: 'MetricsLogger',
                 out_dir: str = 'output',
                 Q: float = 5,
                 rolling: int = 10,
                 corr_window: int = 30,
                 stress_start: int = 200) -> str:
    """
    6-panel H2 dashboard:
      row 0: cost time series  |  CLOB spread vs AMM cost
      row 1: flow allocation   |  rolling correlation
      row 2: AMM liquidity     |  stress flow migration bars
    """
    fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.28)

    fig.suptitle('H2 · Systemic Linkage  &  Liquidity Flight Under Stress',
                 fontsize=16, fontweight='bold', y=0.97)

    def _roll(s, w=rolling):
        if w <= 1:
            return list(range(len(s))), s
        out = []
        for i in range(len(s) - w + 1):
            window = [x for x in s[i:i + w] if math.isfinite(x)]
            out.append(sum(window) / len(window) if window else float('nan'))
        return list(range(w - 1, len(s))), out

    # ── (0,0) Cost time series ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(f'Execution Cost Time Series  (Q={Q}, MA {rolling})')
    xs, ys = _roll(logger.cost_series('clob', Q))
    ax.plot(xs, ys, label='CLOB', lw=1.3, color=_C['clob'])
    for name in logger.amm_cost_curves:
        xs, ys = _roll(logger.cost_series(name, Q))
        ax.plot(xs, ys, label=name.upper(), lw=1.3, color=_C.get(name))
    _shade_stress(ax, logger)
    ax.legend(fontsize=10)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (bps)')
    ax.grid(True, alpha=0.25)

    # ── (0,1) CLOB spread vs AMM cost ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title(f'CLOB Spread vs AMM Cost  (Q={Q}, MA {rolling})')
    qspr = _rolling_avg(logger.clob_qspr, rolling)
    ax1.plot(qspr, label='CLOB qspr', color=_C['clob'], lw=1.3)
    ax1.set_ylabel('CLOB Spread (bps)', color=_C['clob'])
    ax1.tick_params(axis='y', labelcolor=_C['clob'])
    ax2 = ax1.twinx()
    for name in logger.amm_cost_curves:
        s = logger.cost_series(name, Q)
        sr = _rolling_avg(s, rolling)
        ax2.plot(sr, label=f'{name.upper()} cost', color=_C.get(name),
                 lw=1.3, ls='--')
    ax2.set_ylabel('AMM Cost (bps)')
    _shade_stress(ax1, logger)
    lines1, lb1 = ax1.get_legend_handles_labels()
    lines2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lb1 + lb2, fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.25)

    # ── (1,0) Flow allocation ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title(f'Flow Allocation by Venue  [MA {rolling}]')
    venues = list(logger.flow_volume.keys())
    n = len(logger.iterations)
    xs = list(range(n))
    bottoms = [0.0] * n
    for v in venues:
        vals = _rolling_avg(logger.flow_share(v), rolling)
        tops = [b + vi for b, vi in zip(bottoms, vals)]
        ax.fill_between(xs, bottoms, tops, label=v.upper(), alpha=0.6,
                         color=_C.get(v))
        bottoms = tops
    _shade_stress(ax, logger)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Volume Share')
    ax.grid(True, alpha=0.25)

    # ── (1,1) Rolling correlation ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title(f'Rolling Cost Correlation  (Q={Q}, w={corr_window})')
    clob_s = logger.cost_series('clob', Q)
    for name in logger.amm_cost_curves:
        amm_s = logger.cost_series(name, Q)
        nm = min(len(clob_s), len(amm_s))
        if nm < corr_window:
            continue
        rhos = []
        for i in range(corr_window, nm):
            w1 = clob_s[i - corr_window:i]
            w2 = amm_s[i - corr_window:i]
            pairs = [(a, b) for a, b in zip(w1, w2)
                     if math.isfinite(a) and math.isfinite(b)]
            if len(pairs) < 5:
                rhos.append(float('nan'))
                continue
            xv, yv = zip(*pairs)
            nn = len(xv)
            mx, my = sum(xv) / nn, sum(yv) / nn
            cov = sum((x - mx) * (y - my) for x, y in zip(xv, yv)) / nn
            sx = (sum((x - mx) ** 2 for x in xv) / nn) ** 0.5
            sy = (sum((y - my) ** 2 for y in yv) / nn) ** 0.5
            rhos.append(cov / (sx * sy) if sx > 0 and sy > 0 else 0)
        ax.plot(range(corr_window, nm), rhos, label=f'CLOB ↔ {name.upper()}',
                lw=1.3, color=_C.get(name))
    _shade_stress(ax, logger)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.legend(fontsize=10)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Pearson ρ')
    ax.grid(True, alpha=0.25)

    # ── (2,0) AMM liquidity ──────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title('AMM Liquidity  $L_t$  Under Stress')
    for name in logger.amm_L_series:
        ax.plot(logger.amm_L_series[name], label=name.upper(), lw=1.5,
                color=_C.get(name))
    _shade_stress(ax, logger)
    ax.legend(fontsize=10)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('$L_t$')
    ax.grid(True, alpha=0.25)

    # ── (2,1) Stress flow migration ──────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.set_title('AMM Share: Normal vs Stress')
    amm_venues = [v for v in logger.flow_volume if v != 'clob']
    idx = min(stress_start, n)
    before_sh, stress_sh = {}, {}
    for v in amm_venues:
        fs = logger.flow_share(v)
        bef = fs[:idx]
        aft = fs[idx:]
        before_sh[v] = sum(bef) / len(bef) if bef else 0
        stress_sh[v] = sum(aft) / len(aft) if aft else 0

    xp = list(range(len(amm_venues)))
    w = 0.35
    b1 = ax.bar([x - w / 2 for x in xp],
                [before_sh[v] for v in amm_venues],
                w, label='Normal', alpha=0.8, color=_C['normal'])
    b2 = ax.bar([x + w / 2 for x in xp],
                [stress_sh[v] for v in amm_venues],
                w, label='Stress', alpha=0.8, color=_C['stress'])
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f'{h:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    ha='center', va='bottom', fontsize=10)
    ax.set_xticks(xp)
    ax.set_xticklabels([v.upper() for v in amm_venues])
    ax.set_ylabel('Share of Total Volume')
    mx = max(max(before_sh.values(), default=0),
             max(stress_sh.values(), default=0))
    ax.set_ylim(0, mx * 1.45 if mx > 0 else 0.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, axis='y')

    # ── KPI text annotation ──────────────────────────────────────────
    kpi_lines = []
    for name in logger.amm_cost_curves:
        rho = logger.cost_correlation('clob', name, Q=Q)
        ba = logger.commonality_before_after('clob', name, Q=Q,
                                              stress_start=stress_start)
        b_s = f'{ba["before"]:.3f}' if math.isfinite(ba['before']) else 'N/A'
        a_s = f'{ba["after"]:.3f}' if math.isfinite(ba['after']) else 'N/A'
        kpi_lines.append(f'CLOB↔{name.upper()}: ρ={rho:.3f}  '
                         f'(before={b_s}, after={a_s})')
    # CLOB spread normal vs stress
    qspr_raw = logger.clob_qspr
    n_s = [s for s in qspr_raw[:idx] if math.isfinite(s)]
    s_s = [s for s in qspr_raw[idx:] if math.isfinite(s)]
    avg_n = sum(n_s) / len(n_s) if n_s else 0
    avg_s = sum(s_s) / len(s_s) if s_s else 0
    mult = f'{avg_s / avg_n:.1f}×' if avg_n > 0 else '—'
    kpi_lines.append(f'CLOB spread: {avg_n:.1f} → {avg_s:.1f} bps ({mult})')

    fig.text(0.50, 0.01, '   |   '.join(kpi_lines),
             ha='center', va='bottom', fontsize=10,
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7f7f7',
                       alpha=0.9))

    path = os.path.join(out_dir, 'dashboard_h2.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return path


# ===================================================================
#  Convenience wrapper
# ===================================================================

def generate_all_dashboards(logger: 'MetricsLogger',
                            out_dir: str = 'output',
                            stress_start: int = 200,
                            Q: float = 5,
                            rolling: int = 10) -> list:
    """Generate all dashboards, return list of file paths."""
    os.makedirs(out_dir, exist_ok=True)

    paths = [
        dashboard_context(logger, out_dir=out_dir, rolling=rolling),
        dashboard_h1(logger, out_dir=out_dir, Q=Q, rolling=rolling),
        dashboard_h2(logger, out_dir=out_dir, Q=Q, rolling=rolling,
                     stress_start=stress_start),
    ]
    return paths
