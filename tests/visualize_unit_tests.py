#!/usr/bin/env python3
"""
QA-notebook — diagnostic plots for each unit-test block.

For every block (CLOB, CPMM, HFMM, Routing) the script:
  1. Runs the same scenarios as run_unit_report.py
  2. Collects results into pandas DataFrames / CSV
  3. Builds 1-2 focused diagnostic charts per sub-topic

Run:
    python tests/visualize_unit_tests.py
"""

import sys, os, math, random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# --- helpers --------------------------------------------------------------

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
    print(f"  saved  {path}")


def _save_csv(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"  csv    {path}")


# =====================================================================
#  1.  CLOB DIAGNOSTICS
# =====================================================================

def plot_clob():
    print("\n[CLOB] collecting data & generating plots")

    mid_manual = (99.0 + 101.0) / 2.0   # 100.0
    qspr_manual = 10_000 * (101.0 - 99.0) / mid_manual  # 200 bps

    # -- 1.1  mid / spread / effective-spread error table ---------------
    ex = _make_exchange(BIDS, ASKS)
    clob = CLOBVenue(ex, f_broker=0.0, f_venue=0.0)

    scenarios_11 = []

    # Scenario 0: basic mid & qspr
    scenarios_11.append(dict(
        scenario_id='mid+qspr',
        mid_code=clob.mid_price(),
        mid_manual=mid_manual,
        qspr_code=clob.quoted_spread_bps(),
        qspr_manual=qspr_manual,
        espr_code=0.0,
        espr_manual=0.0,
    ))

    # Effective spread scenarios: small (Q=5), medium (Q=25), large (Q=60)
    for label, Q in [('Q=5 (1 lvl)', 5), ('Q=25 (2 lvl)', 25), ('Q=60 (3 lvl)', 60)]:
        q = clob.quote_buy(Q)
        if q['cost_bps'] < float('inf'):
            remaining = Q
            cost = 0.0
            for p, v in ASKS:
                fill = min(remaining, v)
                cost += fill * p
                remaining -= fill
                if remaining <= 0:
                    break
            vwap_manual = cost / Q
            espr_manual = 2 * 10_000 * (vwap_manual - mid_manual) / mid_manual
            scenarios_11.append(dict(
                scenario_id=label,
                mid_code=clob.mid_price(),
                mid_manual=mid_manual,
                qspr_code=clob.quoted_spread_bps(),
                qspr_manual=qspr_manual,
                espr_code=q['espr_bps'],
                espr_manual=espr_manual,
            ))

    df11 = pd.DataFrame(scenarios_11)
    df11['mid_err'] = df11['mid_code'] - df11['mid_manual']
    df11['qspr_err'] = df11['qspr_code'] - df11['qspr_manual']
    df11['espr_err'] = df11['espr_code'] - df11['espr_manual']
    _save_csv(df11, 'clob_mid_spread.csv')

    # -- 1.2  impact & all-in cost vs Q ---------------------------------
    clob_fee = CLOBVenue(_make_exchange(BIDS, ASKS), f_broker=2.0, f_venue=1.0)

    rows_12 = []
    for Q in list(range(1, 61)):
        q0 = clob.quote_buy(Q)
        qf = clob_fee.quote_buy(Q)

        # manual impact
        consumed = 0
        new_best_ask = None
        for idx_a, (p, v) in enumerate(ASKS):
            consumed += v
            if consumed > Q:
                new_best_ask = p
                break
            elif consumed == Q:
                new_best_ask = ASKS[idx_a + 1][0] if idx_a + 1 < len(ASKS) else None
                break
        if new_best_ask is None and consumed <= Q:
            impact_manual = float('nan')
        elif new_best_ask is None:
            impact_manual = 0.0
        else:
            new_mid = (BIDS[0][0] + new_best_ask) / 2.0
            impact_manual = 10_000 * (new_mid - mid_manual) / mid_manual

        # manual cost (half_spread + fee)
        if q0['cost_bps'] < float('inf'):
            rem = Q
            total_c = 0.0
            for p, v in ASKS:
                fill = min(rem, v)
                total_c += fill * p
                rem -= fill
                if rem <= 0:
                    break
            vwap_m = total_c / Q
            hs_manual = 10_000 * (vwap_m - mid_manual) / mid_manual
            cost_manual_f = hs_manual + 3.0  # broker 2 + venue 1
        else:
            cost_manual_f = float('nan')

        rows_12.append(dict(
            Q=Q,
            impact_code=q0['impact_bps'] if q0['cost_bps'] < float('inf') else float('nan'),
            impact_manual=impact_manual,
            cost_code=qf['cost_bps'] if qf['cost_bps'] < float('inf') else float('nan'),
            cost_manual=cost_manual_f if qf['cost_bps'] < float('inf') else float('nan'),
        ))

    df12 = pd.DataFrame(rows_12)
    _save_csv(df12, 'clob_impact_cost.csv')

    # -- Charts ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('1. CLOB  -  Diagnostic Plots', fontsize=14, weight='bold')

    # (a) mid & qspr errors
    ax = axes[0, 0]
    x = np.arange(len(df11))
    w = 0.35
    ax.bar(x - w/2, df11['mid_err'], w, color=ACCENT, label='mid error')
    ax.bar(x + w/2, df11['qspr_err'], w, color=ACCENT2, label='qspr error (bps)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df11['scenario_id'], fontsize=8, rotation=15)
    ax.set_ylabel('Error')
    ax.set_title('1.1a  Mid & Quoted-Spread Error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # (b) effective spread error
    ax = axes[0, 1]
    df_espr = df11[df11['espr_manual'] != 0].copy()
    if len(df_espr) > 0:
        x2 = np.arange(len(df_espr))
        ax.bar(x2, df_espr['espr_err'].values, color=ACCENT4, alpha=0.8)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(x2)
        ax.set_xticklabels(df_espr['scenario_id'].values, fontsize=8)
    ax.set_ylabel('espr_code - espr_manual (bps)')
    ax.set_title('1.1b  Effective Spread Error by Scenario')
    ax.grid(True, alpha=0.3, axis='y')

    # (c) Impact vs Q
    ax = axes[1, 0]
    valid = df12.dropna(subset=['impact_code', 'impact_manual'])
    ax.plot(valid['Q'], valid['impact_code'], '-', color=ACCENT, lw=2, label='impact (code)')
    ax.plot(valid['Q'], valid['impact_manual'], '--', color=ACCENT2, lw=2, label='impact (manual)')
    ax.set_xlabel('Quantity Q')
    ax.set_ylabel('Impact (bps)')
    ax.set_title('1.2a  Price Impact vs Q')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) All-in cost vs Q
    ax = axes[1, 1]
    valid2 = df12.dropna(subset=['cost_code', 'cost_manual'])
    ax.plot(valid2['Q'], valid2['cost_code'], '-', color=ACCENT, lw=2, label='cost (code, 3 bps fee)')
    ax.plot(valid2['Q'], valid2['cost_manual'], '--', color=ACCENT2, lw=2, label='cost (manual)')
    ax.set_xlabel('Quantity Q')
    ax.set_ylabel('All-in cost (bps)')
    ax.set_title('1.2b  All-in Cost vs Q')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, 'clob_diagnostics.png')


# =====================================================================
#  2.  CPMM DIAGNOSTICS
# =====================================================================

def plot_cpmm():
    print("\n[CPMM] collecting data & generating plots")

    # -- 2.1  Invariant k & buy->sell roundtrip -------------------------
    p = CPMMPool(x=1000.0, y=100_000.0, fee=0.003)
    k0 = p.k

    rows_inv = []
    rows_inv.append(dict(step=0, x=p.x, y=p.y, k_code=p.x * p.y, k0=k0, cycle_id=0))

    # 10 buys then 10 sells
    for i in range(1, 11):
        p.execute_buy(2.0)
        rows_inv.append(dict(step=i, x=p.x, y=p.y, k_code=p.x * p.y, k0=k0, cycle_id=0))
    for i in range(11, 21):
        p.execute_sell(2.0)
        rows_inv.append(dict(step=i, x=p.x, y=p.y, k_code=p.x * p.y, k0=k0, cycle_id=0))

    # Multiple round-trip cycles for boxplot
    cycle_deltas = []
    for c in range(1, 31):
        pc = CPMMPool(x=1000.0, y=100_000.0, fee=0.003)
        kc0 = pc.k
        Q = 0.5 + c * 0.3
        pc.execute_buy(Q)
        pc.execute_sell(Q)
        cycle_deltas.append(dict(cycle_id=c, k_delta=pc.x * pc.y - kc0))

    df_inv = pd.DataFrame(rows_inv)
    df_cyc = pd.DataFrame(cycle_deltas)
    _save_csv(df_inv, 'cpmm_invariant.csv')
    _save_csv(df_cyc, 'cpmm_roundtrip.csv')

    # -- 2.2  Slippage & cost vs Q -------------------------------------
    p2 = CPMMPool(x=1000.0, y=100_000.0, fee=0.003)
    S = p2.mid_price()

    rows_slip = []
    for Q in np.concatenate([np.linspace(0.5, 10, 20), np.linspace(15, 300, 40)]):
        qb = p2.quote_buy(Q, S_t=S)
        if qb['cost_bps'] >= float('inf'):
            continue
        x_new = p2.x - Q
        y_new = p2.k / x_new
        dy_eff = y_new - p2.y
        p_exec_nf = dy_eff / Q
        slip_manual = 10_000 * (p_exec_nf - S) / S
        cost_manual = slip_manual + 10_000 * p2.fee + p2.gas_cost_bps

        rows_slip.append(dict(
            Q=Q,
            price_code=qb['exec_price'],
            price_expected=S,
            slip_code=qb['slippage_bps'],
            slip_manual=slip_manual,
            cost_code=qb['cost_bps'],
            cost_manual=cost_manual,
            fee_bps=qb['fee_bps'],
        ))

    df_slip = pd.DataFrame(rows_slip)
    _save_csv(df_slip, 'cpmm_slippage.csv')

    # -- Charts ---------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('2. CPMM  -  Diagnostic Plots', fontsize=14, weight='bold')

    # (a) Invariant k vs step
    ax = axes[0, 0]
    ax.plot(df_inv['step'], df_inv['k_code'], 'o-', color=ACCENT, ms=4, lw=1.5)
    ax.axhline(k0, ls='--', color=ACCENT4, lw=1, label=f'k0 = {k0:.2e}')
    ax.set_xlabel('Trade step')
    ax.set_ylabel('k = x*y')
    ax.set_title('2.1a  Invariant k')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Round-trip k deviation boxplot
    ax = axes[0, 1]
    ax.boxplot(df_cyc['k_delta'].values, vert=True, patch_artist=True,
               boxprops=dict(facecolor=ACCENT, alpha=0.5))
    ax.axhline(0, ls='--', color=GRAY, lw=1)
    ax.set_ylabel('k_after_cycle - k0')
    ax.set_title('2.1b  Round-trip k Deviation (30 cycles)')
    ax.grid(True, alpha=0.3, axis='y')

    # (c) Slippage vs Q
    ax = axes[0, 2]
    ax.plot(df_slip['Q'], df_slip['slip_code'], '-', color=ACCENT, lw=2, label='slip (code)')
    ax.plot(df_slip['Q'], df_slip['slip_manual'], '--', color=ACCENT2, lw=2, label='slip (manual)')
    ax.set_xlabel('Q')
    ax.set_ylabel('Slippage (bps)')
    ax.set_title('2.2a  Slippage vs Q')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if df_slip['Q'].max() > 50:
        ax.set_xscale('log')

    # (d) Slippage error
    ax = axes[1, 0]
    ax.plot(df_slip['Q'], df_slip['slip_code'] - df_slip['slip_manual'],
            '-', color=ACCENT4, lw=1.5)
    ax.axhline(0, ls='--', color=GRAY, lw=1)
    ax.set_xlabel('Q')
    ax.set_ylabel('slip_code - slip_manual (bps)')
    ax.set_title('2.2b  Slippage Error')
    ax.grid(True, alpha=0.3)
    if df_slip['Q'].max() > 50:
        ax.set_xscale('log')

    # (e) Fee vs Q  (should be horizontal line)
    ax = axes[1, 1]
    ax.plot(df_slip['Q'], df_slip['fee_bps'], '-', color=ACCENT3, lw=2)
    ax.axhline(10_000 * 0.003, ls='--', color=GRAY, lw=1, label='30 bps')
    ax.set_xlabel('Q')
    ax.set_ylabel('Fee (bps)')
    ax.set_title('2.2c  Fee vs Q (sanity: horizontal)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (f) All-in cost vs Q
    ax = axes[1, 2]
    ax.plot(df_slip['Q'], df_slip['cost_code'], '-', color=ACCENT, lw=2, label='cost (code)')
    ax.plot(df_slip['Q'], df_slip['cost_manual'], '--', color=ACCENT2, lw=2, label='cost (manual)')
    ax.set_xlabel('Q')
    ax.set_ylabel('All-in cost (bps)')
    ax.set_title('2.2d  All-in Cost vs Q')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if df_slip['Q'].max() > 50:
        ax.set_xscale('log')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, 'cpmm_diagnostics.png')


# =====================================================================
#  3.  HFMM DIAGNOSTICS
# =====================================================================

def plot_hfmm():
    print("\n[HFMM] collecting data & generating plots")

    # -- 3.1  Invariant D & solver stability ----------------------------
    h = HFMMPool(x=1000.0, y=100_000.0, A=100.0, fee=0.001, rate=100.0)
    D0 = h.D

    rows_d = []
    rows_d.append(dict(step=0, x1=h.x, x2=h.y, D_code=D0, D0=D0,
                        solver_iters=0, solver_error=0.0))

    for i in range(1, 16):
        h.execute_buy(3.0)
        h._sync_norm()
        D_re = _hfmm_get_D(h._xn, h._yn, h.A)
        rows_d.append(dict(step=i, x1=h.x, x2=h.y, D_code=D_re, D0=D0,
                            solver_iters=0, solver_error=abs(D_re - D0)))
    for i in range(16, 31):
        h.execute_sell(3.0)
        h._sync_norm()
        D_re = _hfmm_get_D(h._xn, h._yn, h.A)
        rows_d.append(dict(step=i, x1=h.x, x2=h.y, D_code=D_re, D0=D0,
                            solver_iters=0, solver_error=abs(D_re - D0)))

    # Solver convergence across diverse configs
    configs = [
        (100, 100, 1), (100, 100, 10), (100, 100, 100), (100, 100, 1000),
        (1, 1_000_000, 50), (1_000_000, 1, 50),
        (0.001, 0.001, 100), (1e8, 1e8, 500),
    ]
    solver_rows = []
    for idx, (x1, x2, A) in enumerate(configs):
        S = x1 + x2
        Ann = 4.0 * A
        D_val = S
        iters = 0
        err = float('inf')
        for it in range(512):
            P = 4.0 * x1 * x2
            if P < 1e-30:
                break
            D_P = D_val ** 3 / P
            D_prev = D_val
            denom = (Ann - 1.0) * D_val + 3.0 * D_P
            if abs(denom) < 1e-30:
                break
            D_val = (Ann * S + 2.0 * D_P) * D_val / denom
            if D_val < 0:
                D_val = D_prev * 0.5
            err = abs(D_val - D_prev)
            iters = it + 1
            if err < max(1e-12, abs(D_val) * 1e-14):
                break
        solver_rows.append(dict(
            config_id=idx, x1=x1, x2=x2, A=A,
            D=D_val, solver_iters=iters, solver_error=err,
        ))

    df_d = pd.DataFrame(rows_d)
    df_solver = pd.DataFrame(solver_rows)
    _save_csv(df_d, 'hfmm_invariant.csv')
    _save_csv(df_solver, 'hfmm_solver.csv')

    # -- 3.2  Comparison HFMM vs CPMM ----------------------------------
    cpmm_ref = CPMMPool(x=1000, y=100_000, fee=0.0)
    hfmm_ref = HFMMPool(x=1000, y=100_000, A=100, fee=0.0, rate=100.0)

    rows_cmp = []
    for Q in np.concatenate([np.linspace(0.5, 10, 20), np.linspace(15, 300, 40)]):
        qc = cpmm_ref.quote_buy(Q, S_t=100.0)
        qh = hfmm_ref.quote_buy(Q, S_t=100.0)
        if qc['cost_bps'] >= float('inf') or qh['cost_bps'] >= float('inf'):
            continue
        rows_cmp.append(dict(
            Q=Q,
            price_cpmm=qc['exec_price'],
            price_hfmm=qh['exec_price'],
            slip_cpmm=qc['slippage_bps'],
            slip_hfmm=qh['slippage_bps'],
        ))

    df_cmp = pd.DataFrame(rows_cmp)
    _save_csv(df_cmp, 'hfmm_vs_cpmm.csv')

    # -- Charts ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('3. HFMM  -  Diagnostic Plots', fontsize=14, weight='bold')

    # (a) D vs step
    ax = axes[0, 0]
    ax.plot(df_d['step'], df_d['D_code'], 'o-', color=ACCENT, ms=4, lw=1.5)
    ax.axhline(D0, ls='--', color=ACCENT4, lw=1, label=f'D0 = {D0:.2f}')
    ax.axvline(15, ls=':', color=GRAY, lw=1, alpha=0.5, label='switch -> sell')
    ax.set_xlabel('Trade step')
    ax.set_ylabel('D (recomputed)')
    ax.set_title('3.1a  D-invariant vs Step')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Solver iterations per config
    ax = axes[0, 1]
    ax.bar(range(len(df_solver)), df_solver['solver_iters'], color=ACCENT, alpha=0.8)
    labels_s = [f'({r.x1:.0e},{r.x2:.0e},A={int(r.A)})' for _, r in df_solver.iterrows()]
    ax.set_xticks(range(len(df_solver)))
    ax.set_xticklabels(labels_s, fontsize=6, rotation=45, ha='right')
    ax.set_ylabel('Newton iterations')
    ax.set_title('3.1b  Solver Convergence')
    ax.grid(True, alpha=0.3, axis='y')
    for idx_s, row in df_solver.iterrows():
        if row['solver_error'] > 1e-8:
            ax.plot(idx_s, row['solver_iters'], 'x', color=ACCENT4, ms=10, mew=2)

    # (c) Execution price vs Q (HFMM vs CPMM)
    ax = axes[1, 0]
    ax.plot(df_cmp['Q'], df_cmp['price_hfmm'], '-', color=ACCENT, lw=2, label='HFMM (A=100)')
    ax.plot(df_cmp['Q'], df_cmp['price_cpmm'], '--', color=ACCENT2, lw=2, label='CPMM')
    ax.axhline(100.0, ls=':', color=GRAY, lw=1, label='mid = 100')
    ax.set_xlabel('Quantity Q')
    ax.set_ylabel('Execution Price')
    ax.set_title('3.2a  Exec Price: HFMM vs CPMM')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Slippage HFMM vs CPMM
    ax = axes[1, 1]
    ax.plot(df_cmp['Q'], df_cmp['slip_hfmm'], '-', color=ACCENT, lw=2, label='HFMM (A=100)')
    ax.plot(df_cmp['Q'], df_cmp['slip_cpmm'], '--', color=ACCENT2, lw=2, label='CPMM')
    ax.set_xlabel('Quantity Q')
    ax.set_ylabel('Slippage (bps)')
    ax.set_title('3.2b  Slippage: HFMM vs CPMM')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, 'hfmm_diagnostics.png')


# =====================================================================
#  4.  ROUTING DIAGNOSTICS
# =====================================================================

def plot_routing():
    print("\n[ROUTING] collecting data & generating plots")

    def _make_trader_r(amm_share_pct=50, beta_amm=0.05, bias=0.0, noise=0.0, det=False):
        ex = _make_exchange([(99, 50), (98, 50)], [(101, 50), (102, 50)])
        clob = CLOBVenue(ex)
        cpmm = CPMMPool(x=1000, y=100_000, fee=0.003)
        hfmm = HFMMPool(x=1000, y=100_000, A=100, fee=0.001, rate=100.0)
        return Trader(market=ex, cash=1e6, assets=100, clob=clob,
                      amm_pools={'cpmm': cpmm, 'hfmm': hfmm},
                      amm_share_pct=amm_share_pct, deterministic_venue=det,
                      beta_amm=beta_amm, cpmm_bias_bps=bias, cost_noise_std=noise)

    # -- 4.1  Softmax: theory vs empirical ------------------------------
    costs = {'CLOB': 10.0, 'CPMM': 20.0, 'HFMM': 15.0}
    beta = 0.1
    N = 50_000
    min_c = min(costs.values())
    weights = {k: math.exp(-beta * (c - min_c)) for k, c in costs.items()}
    total_w = sum(weights.values())
    theo = {k: w / total_w for k, w in weights.items()}

    random.seed(42)
    counts = Counter(Trader._softmax_pick(
        {k: v for k, v in costs.items()}, beta) for _ in range(N))
    emp = {k: counts.get(k, 0) / N for k in costs}

    rows_sm = []
    for v in costs:
        rows_sm.append(dict(venue=v, p_theoretical=theo[v], p_empirical=emp[v]))
    df_sm = pd.DataFrame(rows_sm)
    _save_csv(df_sm, 'routing_softmax.csv')

    # -- 4.2  Regime transitions ----------------------------------------
    regime_scenarios = {
        'A: CLOB cheap':  {'CLOB': 5.0,  'CPMM': 25.0, 'HFMM': 20.0},
        'B: equal costs':  {'CLOB': 15.0, 'CPMM': 15.0, 'HFMM': 15.0},
        'C: AMM cheap':    {'CLOB': 30.0, 'CPMM': 12.0, 'HFMM': 8.0},
    }
    N_reg = 20_000
    regime_rows = []
    for scenario_name, sc_costs in regime_scenarios.items():
        random.seed(42)
        counts_r = Counter(
            Trader._softmax_pick(sc_costs, beta) for _ in range(N_reg))
        for v in sc_costs:
            regime_rows.append(dict(
                scenario=scenario_name,
                venue=v,
                empirical_share=counts_r.get(v, 0) / N_reg,
            ))
    df_reg = pd.DataFrame(regime_rows)
    _save_csv(df_reg, 'routing_regimes.csv')

    # -- 4.2+  Real two-step routing with amm_share_pct ----------------
    real_scenarios = {
        'amm_share=5%':  dict(amm_share_pct=5),
        'amm_share=50%': dict(amm_share_pct=50),
        'amm_share=95%': dict(amm_share_pct=95),
    }
    N_real = 5_000
    real_rows = []
    for sc_name, kwargs in real_scenarios.items():
        random.seed(42)
        tr = _make_trader_r(**kwargs)
        picks = [tr.choose_venue(1.0) for _ in range(N_real)]
        c = Counter(picks)
        for v in ['clob', 'cpmm', 'hfmm']:
            real_rows.append(dict(
                scenario=sc_name, venue=v,
                empirical_share=c.get(v, 0) / N_real,
            ))
    df_real = pd.DataFrame(real_rows)
    _save_csv(df_real, 'routing_real_twostep.csv')

    # -- Charts ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('4. Routing  -  Diagnostic Plots', fontsize=14, weight='bold')

    # (a) Softmax theory vs empirical
    ax = axes[0, 0]
    venues = df_sm['venue'].tolist()
    x = np.arange(len(venues))
    w = 0.35
    ax.bar(x - w/2, df_sm['p_theoretical'], w, color=ACCENT, alpha=0.8, label='theoretical')
    ax.bar(x + w/2, df_sm['p_empirical'], w, color=ACCENT2, alpha=0.8, label=f'empirical (N={N})')
    ax.set_xticks(x)
    ax.set_xticklabels(venues)
    ax.set_ylabel('Probability')
    ax.set_title(f'4.1  Softmax beta={beta}: Theory vs Empirical')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # (b) Regime transitions (grouped bar)
    ax = axes[0, 1]
    scenario_names = list(regime_scenarios.keys())
    venue_names = list(list(regime_scenarios.values())[0].keys())
    n_sc = len(scenario_names)
    n_v = len(venue_names)
    bar_w = 0.8 / n_v
    colors = [ACCENT, ACCENT2, ACCENT3]
    for j, vn in enumerate(venue_names):
        vals = [df_reg[(df_reg['scenario'] == sc) & (df_reg['venue'] == vn)]['empirical_share'].values[0]
                for sc in scenario_names]
        ax.bar(np.arange(n_sc) + j * bar_w - 0.4 + bar_w/2,
               vals, bar_w, color=colors[j], alpha=0.8, label=vn)
    ax.set_xticks(np.arange(n_sc))
    ax.set_xticklabels(scenario_names, fontsize=8)
    ax.set_ylabel('Empirical share')
    ax.set_title('4.2a  Regime Transitions (softmax)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # (c) Real two-step routing
    ax = axes[1, 0]
    real_sc_names = list(real_scenarios.keys())
    real_venues = ['clob', 'cpmm', 'hfmm']
    n_rsc = len(real_sc_names)
    n_rv = len(real_venues)
    bar_w2 = 0.8 / n_rv
    for j, vn in enumerate(real_venues):
        vals = [df_real[(df_real['scenario'] == sc) & (df_real['venue'] == vn)]['empirical_share'].values[0]
                for sc in real_sc_names]
        ax.bar(np.arange(n_rsc) + j * bar_w2 - 0.4 + bar_w2/2,
               vals, bar_w2, color=colors[j], alpha=0.8, label=vn)
    ax.set_xticks(np.arange(n_rsc))
    ax.set_xticklabels(real_sc_names, fontsize=8)
    ax.set_ylabel('Empirical share')
    ax.set_title('4.2b  Two-step Routing by amm_share_pct')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # (d) Cost per venue vs Q
    ax = axes[1, 1]
    tr = _make_trader_r()
    Qs = [0.5, 1, 5, 10, 25, 50]
    c_clob, c_cpmm, c_hfmm = [], [], []
    for Q in Qs:
        ce = tr.estimate_costs(Q, 'buy')
        c_clob.append(ce.get('clob', float('nan')))
        c_cpmm.append(ce.get('cpmm', float('nan')))
        c_hfmm.append(ce.get('hfmm', float('nan')))
    ax.plot(Qs, c_clob, 'o-', color=ACCENT, lw=2, label='CLOB')
    ax.plot(Qs, c_cpmm, 's--', color=ACCENT2, lw=2, label='CPMM')
    ax.plot(Qs, c_hfmm, '^:', color=ACCENT3, lw=2, label='HFMM')
    ax.set_xlabel('Quantity Q')
    ax.set_ylabel('All-in cost (bps)')
    ax.set_title('4.2c  Venue Cost Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, 'routing_diagnostics.png')


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  QA Diagnostic Visualisation")
    print("=" * 60)
    print(f"  Output -> {os.path.abspath(OUT_DIR)}/")

    plot_clob()
    plot_cpmm()
    plot_hfmm()
    plot_routing()

    print("\n" + "=" * 60)
    print("  Done -- 4 dashboards + CSV data saved to output/unit_tests/")
    print("=" * 60)
