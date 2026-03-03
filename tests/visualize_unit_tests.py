#!/usr/bin/env python3
"""
Visual unit-test report — generates diagnostic plots for CLOB, CPMM, HFMM
and routing, saving them into  output/unit_tests/  as PNG dashboards.

Run:
    python tests/visualize_unit_tests.py
"""
import sys, os, math, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from AgentBasedModel.utils.orders import Order, OrderList
from AgentBasedModel.agents.agents import ExchangeAgent, Trader
from AgentBasedModel.venues.clob import CLOBVenue
from AgentBasedModel.venues.amm import (
    CPMMPool, HFMMPool,
    _hfmm_get_D, _hfmm_get_y, _hfmm_mid_price,
)
from collections import Counter

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'unit_tests')
os.makedirs(OUT_DIR, exist_ok=True)

ACCENT  = '#2196F3'
ACCENT2 = '#FF9800'
ACCENT3 = '#4CAF50'
ACCENT4 = '#E91E63'
GRAY    = '#9E9E9E'


# ─── helpers ──────────────────────────────────────────────────────────

def _make_exchange(bids, asks):
    ex = ExchangeAgent.__new__(ExchangeAgent)
    ex.name = "TestExchange"
    ex.order_book = {'bid': OrderList('bid'), 'ask': OrderList('ask')}
    ex.dividend_book = [100.0] * 10
    ex.risk_free = 0.0
    ex.transaction_cost = 0.0
    for p, q in reversed(bids):
        ex.order_book['bid'].push(Order(p, q, 'bid', None))
    for p, q in asks:
        ex.order_book['ask'].append(Order(p, q, 'ask', None))
    return ex

BIDS = [(99.0, 10), (98.0, 20), (97.0, 30)]
ASKS = [(101.0, 10), (102.0, 20), (103.0, 30)]


def _savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ saved  {path}")


# =====================================================================
#  1.  CLOB DASHBOARD
# =====================================================================

def plot_clob():
    print("\n[CLOB] generating dashboard …")
    ex = _make_exchange(BIDS, ASKS)
    clob = CLOBVenue(ex, f_broker=0.0, f_venue=0.0)
    clob_fee = CLOBVenue(_make_exchange(BIDS, ASKS), f_broker=2.0, f_venue=1.0)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # --- (a) Order book depth ---
    ax = fig.add_subplot(gs[0, 0])
    d = clob.depth(n_levels=3)
    bid_p = [p for p, _ in d['bid']]
    bid_v = [v for _, v in d['bid']]
    ask_p = [p for p, _ in d['ask']]
    ask_v = [v for _, v in d['ask']]
    ax.barh(range(len(bid_p)), bid_v, color=ACCENT3, alpha=0.8, label='Bids')
    ax.set_yticks(range(len(bid_p)))
    ax.set_yticklabels([f'{p:.0f}' for p in bid_p])
    ax2 = ax.twinx()
    ax2.barh(range(len(ask_p)), [-v for v in ask_v], color=ACCENT4, alpha=0.8, label='Asks')
    ax2.set_yticks(range(len(ask_p)))
    ax2.set_yticklabels([f'{p:.0f}' for p in ask_p])
    ax.set_xlabel('Volume')
    ax.set_title('(a) Order Book Depth')
    ax.axvline(0, color='k', lw=0.5)
    mid = clob.mid_price()
    ax.annotate(f'mid = {mid:.1f}', xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', fontsize=9, color=GRAY)

    # --- (b) VWAP curve (buy side) ---
    ax = fig.add_subplot(gs[0, 1])
    Qs = np.arange(1, 61)
    vwaps = []
    for Q in Qs:
        q = clob.quote_buy(int(Q))
        vwaps.append(q['exec_price'] if q['cost_bps'] < float('inf') else np.nan)
    ax.plot(Qs, vwaps, color=ACCENT, lw=2)
    ax.axhline(mid, ls='--', color=GRAY, lw=1, label=f'mid = {mid:.0f}')
    ax.set_xlabel('Buy quantity Q')
    ax.set_ylabel('VWAP (exec price)')
    ax.set_title('(b) VWAP vs Quantity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (c) Half-spread & impact ---
    ax = fig.add_subplot(gs[0, 2])
    Qs_c = np.arange(1, 61)
    half_sp, impact = [], []
    for Q in Qs_c:
        q = clob.quote_buy(int(Q))
        if q['cost_bps'] < float('inf'):
            half_sp.append(q['half_spread_bps'])
            impact.append(q['impact_bps'])
        else:
            half_sp.append(np.nan)
            impact.append(np.nan)
    ax.plot(Qs_c, half_sp, color=ACCENT, lw=2, label='half-spread (bps)')
    ax.plot(Qs_c, impact, color=ACCENT2, lw=2, label='impact (bps)')
    ax.set_xlabel('Buy quantity Q')
    ax.set_ylabel('bps')
    ax.set_title('(c) Half-Spread & Impact')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (d) All-in cost decomposition (zero fee vs 3 bps fee) ---
    ax = fig.add_subplot(gs[1, 0])
    Qs_d = [1, 5, 10, 25, 50]
    hs_nf, hs_f, fee_v = [], [], []
    for Q in Qs_d:
        q0 = clob.quote_buy(Q)
        q1 = clob_fee.quote_buy(Q)
        h = q0['half_spread_bps'] if q0['cost_bps'] < float('inf') else 0
        hs_nf.append(h)
        h1 = q1['half_spread_bps'] if q1['cost_bps'] < float('inf') else 0
        hs_f.append(h1)
        fee_v.append(q1['fee_bps'] if q1['cost_bps'] < float('inf') else 0)
    x = np.arange(len(Qs_d))
    w = 0.35
    ax.bar(x - w/2, hs_nf, w, label='half-spread (no fee)', color=ACCENT, alpha=0.8)
    ax.bar(x + w/2, hs_f, w, label='half-spread', color=ACCENT3, alpha=0.8)
    ax.bar(x + w/2, fee_v, w, bottom=hs_f, label='fee (3 bps)', color=ACCENT2, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in Qs_d])
    ax.set_xlabel('Quantity')
    ax.set_ylabel('bps')
    ax.set_title('(d) Cost Decomposition')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # --- (e) Cumulative depth corridor ---
    ax = fig.add_subplot(gs[1, 1])
    bps_range = np.arange(50, 510, 50)
    bid_depth, ask_depth = [], []
    for b in bps_range:
        td = clob.total_depth(bps_from_mid=b)
        bid_depth.append(td['bid'])
        ask_depth.append(td['ask'])
    ax.step(bps_range, bid_depth, color=ACCENT3, lw=2, where='mid', label='bid depth')
    ax.step(bps_range, ask_depth, color=ACCENT4, lw=2, where='mid', label='ask depth')
    ax.set_xlabel('Corridor (bps from mid)')
    ax.set_ylabel('Cumul. depth (units)')
    ax.set_title('(e) Depth by Corridor')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (f) Summary stats table ---
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    rows = [
        ['Mid-price', f'{mid:.2f}'],
        ['Quoted spread', f'{clob.quoted_spread_bps():.0f} bps'],
        ['Best bid / ask', f'{BIDS[0][0]:.0f} / {ASKS[0][0]:.0f}'],
        ['Total bid depth', f'{sum(v for _,v in BIDS):.0f}'],
        ['Total ask depth', f'{sum(v for _,v in ASKS):.0f}'],
        ['Fee (broker+venue)', '0 / 3 bps'],
    ]
    tbl = ax.table(cellText=rows, colLabels=['Metric', 'Value'],
                   loc='center', cellLoc='left')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#E3F2FD')
            cell.set_text_props(weight='bold')
    ax.set_title('(f) Summary', pad=20)

    fig.suptitle('CLOB Venue — Unit Test Diagnostics', fontsize=14, weight='bold', y=1.01)
    _savefig(fig, 'clob_diagnostics.png')


# =====================================================================
#  2.  CPMM DASHBOARD
# =====================================================================

def plot_cpmm():
    print("\n[CPMM] generating dashboard …")
    p = CPMMPool(x=1000.0, y=100_000.0, fee=0.003)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # --- (a) Bonding curve x·y = k ---
    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(200, 3000, 300)
    ys = p.k / xs
    ax.plot(xs, ys, color=ACCENT, lw=2)
    ax.plot(p.x, p.y, 'o', color=ACCENT4, ms=8, zorder=5, label=f'current ({p.x:.0f}, {p.y:.0f})')
    ax.set_xlabel('x (base)')
    ax.set_ylabel('y (quote)')
    ax.set_title(f'(a) Bonding Curve  x·y = {p.k:.0e}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (b) Slippage curve ---
    ax = fig.add_subplot(gs[0, 1])
    Qs = np.linspace(0.5, 200, 80)
    slip_buy, slip_sell = [], []
    for Q in Qs:
        qb = p.quote_buy(Q, S_t=100.0)
        qs = p.quote_sell(Q, S_t=100.0)
        slip_buy.append(qb['slippage_bps'] if qb['cost_bps'] < float('inf') else np.nan)
        slip_sell.append(qs['slippage_bps'] if qs['cost_bps'] < float('inf') else np.nan)
    ax.plot(Qs, slip_buy, color=ACCENT, lw=2, label='buy slippage')
    ax.plot(Qs, np.abs(slip_sell), color=ACCENT2, lw=2, ls='--', label='|sell slippage|')
    ax.set_xlabel('Quantity Q')
    ax.set_ylabel('Slippage (bps)')
    ax.set_title('(b) Slippage vs Quantity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (c) Exec price curve ---
    ax = fig.add_subplot(gs[0, 2])
    exec_buy, exec_sell = [], []
    for Q in Qs:
        qb = p.quote_buy(Q, S_t=100.0)
        qs = p.quote_sell(Q, S_t=100.0)
        exec_buy.append(qb['exec_price'] if qb['cost_bps'] < float('inf') else np.nan)
        exec_sell.append(qs['exec_price'] if qs['cost_bps'] < float('inf') else np.nan)
    ax.plot(Qs, exec_buy, color=ACCENT, lw=2, label='buy exec price')
    ax.plot(Qs, exec_sell, color=ACCENT2, lw=2, ls='--', label='sell exec price')
    ax.axhline(100.0, ls=':', color=GRAY, lw=1, label='mid = 100')
    ax.set_xlabel('Quantity Q')
    ax.set_ylabel('Execution Price')
    ax.set_title('(c) Exec Price vs Quantity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (d) Cost decomposition ---
    ax = fig.add_subplot(gs[1, 0])
    Qs_d = [1, 5, 10, 50, 100, 200]
    slips, fees = [], []
    for Q in Qs_d:
        q = p.quote_buy(Q, S_t=100.0)
        if q['cost_bps'] < float('inf'):
            slips.append(q['slippage_bps'])
            fees.append(q['fee_bps'])
        else:
            slips.append(0)
            fees.append(0)
    x_pos = np.arange(len(Qs_d))
    ax.bar(x_pos, slips, color=ACCENT, alpha=0.8, label='slippage')
    ax.bar(x_pos, fees, bottom=slips, color=ACCENT2, alpha=0.8, label='fee (30 bps)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(q) for q in Qs_d])
    ax.set_xlabel('Quantity')
    ax.set_ylabel('bps')
    ax.set_title('(d) Buy Cost = Slippage + Fee')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- (e) Invariant preservation (sequential trades) ---
    ax = fig.add_subplot(gs[1, 1])
    p2 = CPMMPool(x=1000.0, y=100_000.0, fee=0.003)
    k0 = p2.k
    ks = [k0]
    mids = [p2.mid_price()]
    labels = ['init']
    for i in range(1, 11):
        p2.execute_buy(2.0)
        ks.append(p2.x * p2.y)
        mids.append(p2.mid_price())
        labels.append(f'buy {i}')
    for i in range(1, 11):
        p2.execute_sell(2.0)
        ks.append(p2.x * p2.y)
        mids.append(p2.mid_price())
        labels.append(f'sell {i}')
    steps = range(len(ks))
    k_pct = [(k / k0 - 1) * 1e6 for k in ks]  # in ppm
    ax.plot(steps, k_pct, 'o-', color=ACCENT, ms=4, lw=1.5)
    ax.axhline(0, ls='--', color=GRAY, lw=1)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Δk / k₀  (ppm)')
    ax.set_title('(e) k-Invariant Drift')
    ax.grid(True, alpha=0.3)

    # --- (f) Volume-slippage max_Q ---
    ax = fig.add_subplot(gs[1, 2])
    thresholds = np.arange(20, 520, 20)
    max_Qs = [p.volume_slippage_max_Q(t, S_t=100.0) for t in thresholds]
    ax.plot(thresholds, max_Qs, color=ACCENT, lw=2)
    ax.set_xlabel('Cost threshold (bps)')
    ax.set_ylabel('Max tradeable Q')
    ax.set_title('(f) Max Q at Cost Budget')
    ax.grid(True, alpha=0.3)

    fig.suptitle('CPMM Pool — Unit Test Diagnostics', fontsize=14, weight='bold', y=1.01)
    _savefig(fig, 'cpmm_diagnostics.png')


# =====================================================================
#  3.  HFMM DASHBOARD
# =====================================================================

def plot_hfmm():
    print("\n[HFMM] generating dashboard …")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # --- (a) HFMM vs CPMM slippage ---
    ax = fig.add_subplot(gs[0, 0])
    cpmm = CPMMPool(x=1000, y=100_000, fee=0.0)
    hfmm = HFMMPool(x=1000, y=100_000, A=100, fee=0.0, rate=100.0)
    Qs = np.linspace(0.5, 300, 120)
    s_c, s_h = [], []
    for Q in Qs:
        qc = cpmm.quote_buy(Q, S_t=100.0)
        qh = hfmm.quote_buy(Q, S_t=100.0)
        s_c.append(qc['slippage_bps'] if qc['cost_bps'] < float('inf') else np.nan)
        s_h.append(qh['slippage_bps'] if qh['cost_bps'] < float('inf') else np.nan)
    ax.plot(Qs, s_c, color=ACCENT2, lw=2, label='CPMM')
    ax.plot(Qs, s_h, color=ACCENT, lw=2, label='HFMM (A=100)')
    ax.set_xlabel('Quantity Q')
    ax.set_ylabel('Slippage (bps)')
    ax.set_title('(a) HFMM vs CPMM Slippage')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (b) Amplification parameter sweep ---
    ax = fig.add_subplot(gs[0, 1])
    A_vals = [1, 5, 10, 50, 100, 500]
    Q_test = np.linspace(1, 200, 60)
    for A in A_vals:
        h = HFMMPool(x=1000, y=100_000, A=A, fee=0.0, rate=100.0)
        sl = []
        for Q in Q_test:
            q = h.quote_buy(Q, S_t=100.0)
            sl.append(q['slippage_bps'] if q['cost_bps'] < float('inf') else np.nan)
        ax.plot(Q_test, sl, lw=1.5, label=f'A={A}')
    ax.set_xlabel('Quantity Q')
    ax.set_ylabel('Slippage (bps)')
    ax.set_title('(b) Slippage by Amplification A')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- (c) StableSwap bonding curve ---
    ax = fig.add_subplot(gs[0, 2])
    h100 = HFMMPool(x=1000, y=100_000, A=100, fee=0.0, rate=100.0)
    D = h100.D
    # Plot implicit curve: vary xn, solve for yn
    xns = np.linspace(10, h100.D * 0.99, 400)
    yns = []
    for xn in xns:
        try:
            yn = _hfmm_get_y(xn, D, 100)
            yns.append(yn)
        except Exception:
            yns.append(np.nan)
    ax.plot(xns, yns, color=ACCENT, lw=2, label='HFMM (A=100)')
    # CPMM reference
    k_norm = (D/2)**2  # at balanced point xn=yn=D/2
    yn_cpmm = k_norm / xns
    ax.plot(xns, yn_cpmm, color=ACCENT2, lw=1.5, ls='--', label='CPMM equiv.')
    # Constant sum
    ax.plot(xns, D - xns, color=ACCENT3, lw=1, ls=':', label='const-sum')
    ax.plot(h100._xn, h100._yn, 'o', color=ACCENT4, ms=8, zorder=5, label='current')
    ax.set_xlabel('xₙ (normalised)')
    ax.set_ylabel('yₙ (normalised)')
    ax.set_title('(c) StableSwap Curve')
    ax.legend(fontsize=7)
    ax.set_xlim(0, D * 1.05)
    ax.set_ylim(0, D * 1.05)
    ax.grid(True, alpha=0.3)

    # --- (d) D-solver across configs ---
    ax = fig.add_subplot(gs[1, 0])
    configs = [
        (100, 100, 1), (100, 100, 10), (100, 100, 100), (100, 100, 1000),
        (1, 1e6, 50), (1e6, 1, 50), (1e-3, 1e-3, 100), (1e8, 1e8, 500),
    ]
    labels_d = []
    Ds = []
    for x1, x2, A in configs:
        d = _hfmm_get_D(x1, x2, A)
        Ds.append(d)
        labels_d.append(f'({x1:.0e},{x2:.0e},A={A})')
    ax.barh(range(len(Ds)), np.log10(np.array(Ds) + 1e-30), color=ACCENT, alpha=0.8)
    ax.set_yticks(range(len(Ds)))
    ax.set_yticklabels(labels_d, fontsize=7)
    ax.set_xlabel('log₁₀(D)')
    ax.set_title('(d) D-solver: log₁₀(D) per Config')
    ax.grid(True, alpha=0.3, axis='x')

    # --- (e) Cost comparison table ---
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    h = HFMMPool(x=1000, y=100_000, A=100, fee=0.001, rate=100.0)
    c = CPMMPool(x=1000, y=100_000, fee=0.003)
    rows = []
    for Q in [1, 5, 10, 50, 100]:
        qh = h.quote_buy(Q, S_t=100.0)
        qc = c.quote_buy(Q, S_t=100.0)
        rows.append([
            f'{Q}',
            f'{qh["slippage_bps"]:.2f}',
            f'{qh["cost_bps"]:.2f}',
            f'{qc["slippage_bps"]:.2f}',
            f'{qc["cost_bps"]:.2f}',
        ])
    tbl = ax.table(cellText=rows,
                   colLabels=['Q', 'HFMM slip', 'HFMM cost', 'CPMM slip', 'CPMM cost'],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)
    for (r, col), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#E3F2FD')
            cell.set_text_props(weight='bold')
    ax.set_title('(e) HFMM vs CPMM Cost (bps)', pad=20)

    # --- (f) D invariant drift under sequential trades ---
    ax = fig.add_subplot(gs[1, 2])
    h2 = HFMMPool(x=1000, y=100_000, A=100, fee=0.001, rate=100.0)
    D0 = h2.D
    d_drift = [0.0]
    for i in range(15):
        h2.execute_buy(3.0)
        h2._sync_norm()
        d_re = _hfmm_get_D(h2._xn, h2._yn, h2.A)
        d_drift.append((d_re / D0 - 1) * 1e6)
    for i in range(15):
        h2.execute_sell(3.0)
        h2._sync_norm()
        d_re = _hfmm_get_D(h2._xn, h2._yn, h2.A)
        d_drift.append((d_re / D0 - 1) * 1e6)
    ax.plot(range(len(d_drift)), d_drift, 'o-', color=ACCENT, ms=3, lw=1.5)
    ax.axhline(0, ls='--', color=GRAY, lw=1)
    ax.axvline(15, ls=':', color=ACCENT4, lw=1, alpha=0.5, label='switch sell')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('ΔD / D₀  (ppm)')
    ax.set_title('(f) D-Invariant Drift')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('HFMM Pool — Unit Test Diagnostics', fontsize=14, weight='bold', y=1.01)
    _savefig(fig, 'hfmm_diagnostics.png')


# =====================================================================
#  4.  ROUTING DASHBOARD
# =====================================================================

def plot_routing():
    print("\n[ROUTING] generating dashboard …")

    def _make_trader_r(amm_share_pct=50, beta_amm=0.05, bias=0.0, noise=0.0, det=False):
        ex = _make_exchange([(99, 50), (98, 50)], [(101, 50), (102, 50)])
        clob = CLOBVenue(ex)
        cpmm = CPMMPool(x=1000, y=100_000, fee=0.003)
        hfmm = HFMMPool(x=1000, y=100_000, A=100, fee=0.001, rate=100.0)
        return Trader(market=ex, cash=1e6, assets=100, clob=clob,
                      amm_pools={'cpmm': cpmm, 'hfmm': hfmm},
                      amm_share_pct=amm_share_pct, deterministic_venue=det,
                      beta_amm=beta_amm, cpmm_bias_bps=bias, cost_noise_std=noise)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # --- (a) Softmax probability distribution ---
    ax = fig.add_subplot(gs[0, 0])
    costs = {'A': 10.0, 'B': 20.0, 'Z': 15.0}
    betas = np.linspace(0.01, 0.5, 50)
    for venue in ['A', 'B', 'Z']:
        probs = []
        for beta in betas:
            min_c = min(costs.values())
            w = {k: math.exp(-beta * (c - min_c)) for k, c in costs.items()}
            total = sum(w.values())
            probs.append(w[venue] / total)
        ax.plot(betas, probs, lw=2, label=f'{venue} (cost={costs[venue]})')
    ax.set_xlabel('β (concentration)')
    ax.set_ylabel('Selection probability')
    ax.set_title('(a) Softmax P(venue) vs β')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (b) Empirical vs theoretical softmax ---
    ax = fig.add_subplot(gs[0, 1])
    random.seed(42)
    beta = 0.1
    N = 50_000
    min_c = min(costs.values())
    w = {k: math.exp(-beta * (c - min_c)) for k, c in costs.items()}
    total = sum(w.values())
    theo = {k: w[k] / total for k in costs}
    counts = Counter(Trader._softmax_pick(costs, beta) for _ in range(N))
    obs = {k: counts[k] / N for k in costs}
    x_pos = np.arange(len(costs))
    w_bar = 0.35
    ax.bar(x_pos - w_bar/2, [theo[k] for k in costs], w_bar, color=ACCENT, alpha=0.8, label='theoretical')
    ax.bar(x_pos + w_bar/2, [obs[k] for k in costs], w_bar, color=ACCENT2, alpha=0.8, label=f'observed (N={N})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(costs.keys()))
    ax.set_ylabel('Probability')
    ax.set_title(f'(b) Softmax β={beta}: Theory vs Obs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- (c) Venue share vs amm_share_pct ---
    ax = fig.add_subplot(gs[0, 2])
    amm_pcts = [0, 10, 25, 50, 75, 90, 100]
    clob_shares, cpmm_shares, hfmm_shares = [], [], []
    N = 2000
    for pct in amm_pcts:
        random.seed(42)
        tr = _make_trader_r(amm_share_pct=pct)
        picks = [tr.choose_venue(1.0) for _ in range(N)]
        c = Counter(picks)
        clob_shares.append(c.get('clob', 0) / N * 100)
        cpmm_shares.append(c.get('cpmm', 0) / N * 100)
        hfmm_shares.append(c.get('hfmm', 0) / N * 100)
    ax.stackplot(amm_pcts, clob_shares, cpmm_shares, hfmm_shares,
                 labels=['CLOB', 'CPMM', 'HFMM'],
                 colors=[ACCENT, ACCENT2, ACCENT3], alpha=0.8)
    ax.set_xlabel('amm_share_pct')
    ax.set_ylabel('Venue share (%)')
    ax.set_title('(c) Venue Flow vs amm_share_pct')
    ax.legend(fontsize=8, loc='center left')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # --- (d) CPMM bias effect ---
    ax = fig.add_subplot(gs[1, 0])
    biases = np.arange(0, 110, 10)
    cpmm_frac = []
    N = 3000
    for bias in biases:
        random.seed(42)
        tr = _make_trader_r(amm_share_pct=100, bias=bias, beta_amm=0.05)
        picks = Counter(tr.choose_venue(1.0) for _ in range(N))
        cpmm_frac.append(picks.get('cpmm', 0) / N * 100)
    ax.plot(biases, cpmm_frac, 'o-', color=ACCENT, lw=2, ms=5)
    ax.set_xlabel('cpmm_bias_bps')
    ax.set_ylabel('CPMM share (%)')
    ax.set_title('(d) CPMM Bias → CPMM Share')
    ax.grid(True, alpha=0.3)

    # --- (e) Noise effect ---
    ax = fig.add_subplot(gs[1, 1])
    noises = [0, 0.5, 1, 2, 3, 5, 10]
    hfmm_frac_n = []
    cpmm_frac_n = []
    N = 3000
    for ns in noises:
        random.seed(42)
        tr = _make_trader_r(amm_share_pct=100, noise=ns, beta_amm=0.05)
        picks = Counter(tr.choose_venue(1.0) for _ in range(N))
        total_n = picks.get('cpmm', 0) + picks.get('hfmm', 0)
        if total_n > 0:
            hfmm_frac_n.append(picks.get('hfmm', 0) / total_n * 100)
            cpmm_frac_n.append(picks.get('cpmm', 0) / total_n * 100)
        else:
            hfmm_frac_n.append(50)
            cpmm_frac_n.append(50)
    ax.plot(noises, hfmm_frac_n, 'o-', color=ACCENT, lw=2, ms=5, label='HFMM')
    ax.plot(noises, cpmm_frac_n, 's--', color=ACCENT2, lw=2, ms=5, label='CPMM')
    ax.set_xlabel('cost_noise_std')
    ax.set_ylabel('Share among AMMs (%)')
    ax.set_title('(e) Noise → AMM Split')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (f) Cost comparison ---
    ax = fig.add_subplot(gs[1, 2])
    tr = _make_trader_r()
    Qs = [0.5, 1, 5, 10, 50]
    c_clob, c_cpmm, c_hfmm = [], [], []
    for Q in Qs:
        costs_e = tr.estimate_costs(Q, 'buy')
        c_clob.append(costs_e.get('clob', float('nan')))
        c_cpmm.append(costs_e.get('cpmm', float('nan')))
        c_hfmm.append(costs_e.get('hfmm', float('nan')))
    x_pos = np.arange(len(Qs))
    w_b = 0.25
    ax.bar(x_pos - w_b, c_clob, w_b, color=ACCENT, alpha=0.8, label='CLOB')
    ax.bar(x_pos, c_cpmm, w_b, color=ACCENT2, alpha=0.8, label='CPMM')
    ax.bar(x_pos + w_b, c_hfmm, w_b, color=ACCENT3, alpha=0.8, label='HFMM')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(q) for q in Qs])
    ax.set_xlabel('Quantity')
    ax.set_ylabel('All-in cost (bps)')
    ax.set_title('(f) Venue Cost Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Venue Routing — Unit Test Diagnostics', fontsize=14, weight='bold', y=1.01)
    _savefig(fig, 'routing_diagnostics.png')


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Unit Test Visualisation Report")
    print("=" * 60)
    print(f"  Output → {os.path.abspath(OUT_DIR)}/")

    plot_clob()
    plot_cpmm()
    plot_hfmm()
    plot_routing()

    print(f"\n{'=' * 60}")
    print(f"  Done — 4 dashboards saved to output/unit_tests/")
    print(f"{'=' * 60}")
