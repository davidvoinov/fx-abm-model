"""
Visualization functions for multi-venue FX ABM results.

All functions accept a MetricsLogger instance and produce matplotlib figures.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from AgentBasedModel.metrics.logger import MetricsLogger
    from AgentBasedModel.venues.amm import CPMMPool, HFMMPool


# ---------------------------------------------------------------------------
# 1.  Execution cost curves: C_v(Q) at a fixed iteration or averaged
# ---------------------------------------------------------------------------

def plot_execution_cost_curves(logger: 'MetricsLogger',
                               figsize=(10, 6)):
    """
    Average all-in cost C_v(Q) across the whole simulation, per venue.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('Average Execution Cost  $C_v(Q)$ by Venue')
    ax.set_xlabel('Trade Size  $Q$  (base)')
    ax.set_ylabel('All-in cost (bps)')

    Q_grid = logger.Q_grid

    # CLOB
    means = []
    for Q in Q_grid:
        s = logger.clob_cost_curves.get(Q, [])
        finite = [x for x in s if math.isfinite(x)]
        means.append(sum(finite) / len(finite) if finite else float('nan'))
    ax.plot(Q_grid, means, 'o-', label='CLOB', linewidth=2)

    # AMMs
    for name in logger.amm_cost_curves:
        means = []
        for Q in Q_grid:
            s = logger.amm_cost_curves[name].get(Q, [])
            finite = [x for x in s if math.isfinite(x)]
            means.append(sum(finite) / len(finite) if finite else float('nan'))
        ax.plot(Q_grid, means, 's--', label=name.upper(), linewidth=2)

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2.  Cost comparison time series for fixed Q
# ---------------------------------------------------------------------------

def plot_cost_timeseries(logger: 'MetricsLogger', Q: float = 5,
                         rolling: int = 1, figsize=(12, 5)):
    """
    Time series of C_v(Q) for each venue at a fixed trade size Q.
    """
    fig, ax = plt.subplots(figsize=figsize)
    title = f'Execution Cost Time Series  (Q={Q})'
    if rolling > 1:
        title += f'  [MA {rolling}]'
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (bps)')

    def _roll(s):
        if rolling <= 1:
            return list(range(len(s))), s
        out = []
        for i in range(len(s) - rolling + 1):
            window = [x for x in s[i:i + rolling] if math.isfinite(x)]
            out.append(sum(window) / len(window) if window else float('nan'))
        return list(range(rolling - 1, len(s))), out

    # CLOB
    xs, ys = _roll(logger.cost_series('clob', Q))
    ax.plot(xs, ys, label='CLOB', linewidth=1.2)

    # AMMs
    for name in logger.amm_cost_curves:
        xs, ys = _roll(logger.cost_series(name, Q))
        ax.plot(xs, ys, label=name.upper(), linewidth=1.2)

    # Shade stress regime
    _shade_stress(ax, logger)

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3.  Cost decomposition (slippage vs fee)
# ---------------------------------------------------------------------------

def plot_cost_decomposition(logger: 'MetricsLogger', Q: float = 5,
                            figsize=(10, 6)):
    """
    Stacked bar: slippage + fee for each venue (averaged).
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'Cost Decomposition (Q={Q}): Slippage vs Fee')

    venues = []
    slippages = []
    fees = []

    # CLOB: half_spread ≈ slippage, fee is separate
    s = logger.clob_cost_curves.get(Q, [])
    f_bps = logger.clob_espr_curves.get(Q, [])  # using espr as proxy
    finite_s = [x for x in s if math.isfinite(x)]
    finite_f = [x for x in f_bps if math.isfinite(x)]
    if finite_s:
        venues.append('CLOB')
        slippages.append(sum(finite_s) / len(finite_s))
        fees.append(0)  # fee already in cost for CLOB

    # AMMs
    for name in logger.amm_slippage_curves:
        sl = logger.amm_slippage_curves[name].get(Q, [])
        fe = logger.amm_fee_curves[name].get(Q, [])
        finite_sl = [x for x in sl if math.isfinite(x)]
        finite_fe = [x for x in fe if math.isfinite(x)]
        if finite_sl:
            venues.append(name.upper())
            slippages.append(sum(finite_sl) / len(finite_sl))
            fees.append(sum(finite_fe) / len(finite_fe) if finite_fe else 0)

    x_pos = range(len(venues))
    ax.bar(x_pos, slippages, label='Slippage', alpha=0.8)
    ax.bar(x_pos, fees, bottom=slippages, label='Fee', alpha=0.8)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(venues)
    ax.set_ylabel('Cost (bps)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4.  Flow allocation
# ---------------------------------------------------------------------------

def plot_flow_allocation(logger: 'MetricsLogger', rolling: int = 10,
                         figsize=(12, 5)):
    """
    Stacked area chart: share of volume to each venue over time.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('Flow Allocation by Venue')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Volume Share')

    venues = list(logger.flow_volume.keys())
    if not venues:
        return

    n = len(logger.iterations)
    shares = {}
    for v in venues:
        shares[v] = logger.flow_share(v)

    # Rolling average
    def _roll(s, w):
        if w <= 1:
            return s
        return [sum(s[max(0, i - w + 1):i + 1]) / min(w, i + 1)
                for i in range(len(s))]

    xs = list(range(n))
    bottoms = [0.0] * n
    for v in venues:
        vals = _roll(shares[v], rolling)
        ax.fill_between(xs, bottoms, [b + v_ for b, v_ in zip(bottoms, vals)],
                         label=v.upper(), alpha=0.6)
        bottoms = [b + v_ for b, v_ in zip(bottoms, vals)]

    _shade_stress(ax, logger)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5.  AMM reserves / liquidity over time
# ---------------------------------------------------------------------------

def plot_amm_liquidity(logger: 'MetricsLogger', figsize=(12, 5)):
    """
    Liquidity measure L_t for each AMM pool.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('AMM Liquidity Over Time')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Liquidity Measure  $L_t$')

    for name in logger.amm_L_series:
        ax.plot(logger.amm_L_series[name], label=name.upper(), linewidth=1.5)

    _shade_stress(ax, logger)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_amm_reserves(logger: 'MetricsLogger', pool_name: str = 'cpmm',
                      figsize=(12, 5)):
    """
    Base (x) and quote (y) reserves for a given AMM pool.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax1.set_title(f'{pool_name.upper()} Reserves')
    ax1.plot(logger.amm_x_series.get(pool_name, []), label='x (base)')
    ax1.set_ylabel('Base reserves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(logger.amm_y_series.get(pool_name, []), label='y (quote)',
             color='orange')
    ax2.set_ylabel('Quote reserves')
    ax2.set_xlabel('Iteration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    _shade_stress(ax1, logger)
    _shade_stress(ax2, logger)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6.  Volume-slippage profile
# ---------------------------------------------------------------------------

def plot_volume_slippage_profile(logger: 'MetricsLogger',
                                 pool_name: str = 'cpmm',
                                 figsize=(8, 5)):
    """
    Average max-Q at each slippage threshold for an AMM pool.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'Volume-Slippage Profile — {pool_name.upper()}')
    ax.set_xlabel('Slippage threshold (bps)')
    ax.set_ylabel('Max trade size  $Q$')

    if pool_name not in logger.amm_vol_slip:
        return

    thresholds = logger.slippage_thresholds
    avg_Qs = []
    for th in thresholds:
        s = logger.amm_vol_slip[pool_name][th]
        avg_Qs.append(sum(s) / len(s) if s else 0)

    ax.plot(thresholds, avg_Qs, 'o-', linewidth=2, markersize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 7.  Environment (σ, c) time series
# ---------------------------------------------------------------------------

def plot_environment(logger: 'MetricsLogger', figsize=(12, 4)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.set_title('Volatility  $\\sigma_t$')
    ax1.plot(logger.sigma_series, color='tab:red')
    ax1.set_xlabel('Iteration')
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Funding Cost  $c_t$')
    ax2.plot(logger.c_series, color='tab:blue')
    ax2.set_xlabel('Iteration')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 8.  CLOB spread / depth
# ---------------------------------------------------------------------------

def plot_clob_spread(logger: 'MetricsLogger', rolling: int = 5,
                     figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    title = 'CLOB Quoted Spread'
    if rolling > 1:
        title += f'  [MA {rolling}]'
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Spread (bps)')

    s = logger.clob_qspr
    if rolling > 1:
        s = [sum(s[max(0, i - rolling + 1):i + 1]) / min(rolling, i + 1)
             for i in range(len(s))]
    ax.plot(s, color='black', linewidth=1)
    _shade_stress(ax, logger)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 9.  Commonality / correlation analysis
# ---------------------------------------------------------------------------

def plot_commonality(logger: 'MetricsLogger', Q: float = 5,
                     window: int = 30, figsize=(12, 5)):
    """
    Rolling correlation of cost series between CLOB and each AMM.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'Rolling Cost Correlation (Q={Q}, window={window})')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Pearson ρ')

    clob_s = logger.cost_series('clob', Q)

    for name in logger.amm_cost_curves:
        amm_s = logger.cost_series(name, Q)
        n = min(len(clob_s), len(amm_s))
        if n < window:
            continue
        rhos = []
        for i in range(window, n):
            w1 = clob_s[i - window:i]
            w2 = amm_s[i - window:i]
            pairs = [(a, b) for a, b in zip(w1, w2)
                     if math.isfinite(a) and math.isfinite(b)]
            if len(pairs) < 5:
                rhos.append(float('nan'))
                continue
            xs, ys = zip(*pairs)
            n_ = len(xs)
            mx, my = sum(xs) / n_, sum(ys) / n_
            cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n_
            sx = (sum((x - mx) ** 2 for x in xs) / n_) ** 0.5
            sy = (sum((y - my) ** 2 for y in ys) / n_) ** 0.5
            rhos.append(cov / (sx * sy) if sx > 0 and sy > 0 else 0)

        ax.plot(range(window, n), rhos, label=f'CLOB ↔ {name.upper()}',
                linewidth=1.2)

    _shade_stress(ax, logger)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 10.  CLOB mid-price
# ---------------------------------------------------------------------------

def plot_fx_price(logger: 'MetricsLogger', rolling: int = 1,
                  figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    title = 'FX Mid-Price (CLOB)'
    if rolling > 1:
        title += f'  [MA {rolling}]'
    ax.set_title(title)
    s = logger.clob_mid_series
    if rolling > 1:
        s = [sum(s[max(0, i - rolling + 1):i + 1]) / min(rolling, i + 1)
             for i in range(len(s))]
    ax.plot(s, color='black', linewidth=1)
    _shade_stress(ax, logger)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _shade_stress(ax, logger: 'MetricsLogger', color='red', alpha=0.08):
    """Shade stress-regime intervals on an axis."""
    regimes = logger.regime_series
    in_stress = False
    start = 0
    for i, r in enumerate(regimes):
        if r == 'stress' and not in_stress:
            start = i
            in_stress = True
        elif r != 'stress' and in_stress:
            ax.axvspan(start, i, color=color, alpha=alpha)
            in_stress = False
    if in_stress:
        ax.axvspan(start, len(regimes), color=color, alpha=alpha)
