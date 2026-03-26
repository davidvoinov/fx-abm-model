#!/usr/bin/env python3
"""
Fennell Tests — Transaction Cost Analysis: AMM vs CLOB
=======================================================

Replicates the methodology from James Fennell's honours thesis (UTS, 2024):
    "FX Transaction Costs: AMM vs CLOB"

For a fixed FX price path (GBM) and a sequence of orders, simultaneously
execute each trade on CLOB, CPMM, and HFMM, then compare the average
execution cost in bps by volume buckets Q.

Metric per trade:
    cost = side × (P_exec − P_mid) / P_mid × 10 000
where P_mid is the exogenous fair price at the moment of the trade.

Output:
    - Console tables (avg cost per bucket per venue, by fee regime)
    - CSV to output/fennell/
    - Plots: cost curves C_v(Q) for three venues, slippage vs fee decomposition

Run:
    python tests/fennel.py
"""

import sys, os, math, random, copy
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from AgentBasedModel.utils.orders import Order, OrderList
from AgentBasedModel.agents.agents import ExchangeAgent
from AgentBasedModel.venues.clob import CLOBVenue
from AgentBasedModel.venues.amm import CPMMPool, HFMMPool

# ── Output directory ─────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'fennell')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colors ────────────────────────────────────────────────────────────
C_CLOB = '#2196F3'
C_CPMM = '#FF9800'
C_HFMM = '#4CAF50'
C_GRAY = '#9E9E9E'


# =====================================================================
#  1. PRICE PATH GENERATION (GBM)
# =====================================================================

def generate_gbm_path(S0: float, mu: float, sigma: float,
                      n_steps: int, seed: int = 42) -> np.ndarray:
    """Generate a GBM price path with fixed seed for reproducibility."""
    rng = np.random.RandomState(seed)
    eps = rng.randn(n_steps)
    log_ret = (mu - 0.5 * sigma ** 2) + sigma * eps
    log_prices = np.concatenate([[np.log(S0)], np.cumsum(log_ret) + np.log(S0)])
    return np.exp(log_prices)


# =====================================================================
#  2. ORDER FLOW GENERATION
# =====================================================================

def generate_order_flow(n_trades: int, Q_buckets: list,
                        seed: int = 42) -> list:
    """
    Generate a deterministic sequence of (side, quantity) orders.

    For each bucket in Q_buckets, generates n_trades // len(Q_buckets)
    trades at that exact Q (balanced buys/sells), then shuffles
    deterministically so the flow is mixed.
    """
    rng = random.Random(seed)
    orders = []
    per_bucket = max(1, n_trades // len(Q_buckets))
    for Q in Q_buckets:
        for i in range(per_bucket):
            side = 'buy' if i % 2 == 0 else 'sell'
            orders.append({'side': side, 'quantity': float(Q)})
    rng.shuffle(orders)
    return orders


# =====================================================================
#  3. CLOB BUILDER (live order book replenished each step)
# =====================================================================

def _build_clob(mid: float, half_spread_bps: float,
                depth_per_level: float, n_levels: int = 10) -> CLOBVenue:
    """
    Construct a CLOBVenue with a synthetic order book centered at *mid*.

    Levels are evenly spaced from half_spread out to n_levels × tick.
    Book is non-depleting: rebuilt from scratch at each time step.
    """
    ex = ExchangeAgent.__new__(ExchangeAgent)
    ex.name = "FennellCLOB"
    ex.order_book = {'bid': OrderList('bid'), 'ask': OrderList('ask')}
    ex.dividend_book = [mid] * 5
    ex.risk_free = 0.0
    ex.transaction_cost = 0.0

    tick = mid * half_spread_bps / 10_000.0
    for i in range(n_levels):
        ask_px = round(mid + tick * (i + 1), 6)
        bid_px = round(mid - tick * (i + 1), 6)
        # Deeper levels get more liquidity
        qty = depth_per_level * (1.0 + 0.3 * i)
        ex.order_book['ask'].append(Order(ask_px, qty, 'ask', None))
        ex.order_book['bid'].push(Order(bid_px, qty, 'bid', None))

    return CLOBVenue(ex, f_broker=0.0, f_venue=0.0)


# =====================================================================
#  4. SINGLE TRADE EXECUTION ON EACH VENUE
# =====================================================================

def _execute_trade_clob(clob: CLOBVenue, side: str, Q: float,
                        P_mid: float, fee_bps: float) -> dict:
    """
    Execute a trade on CLOB (read-only quote) and compute Fennell cost.

    Fennell cost = side × (P_exec − P_mid) / P_mid × 10 000 + fee
    """
    if side == 'buy':
        quote = clob.quote_buy(Q)
    else:
        quote = clob.quote_sell(Q)

    P_exec = quote['exec_price']
    if P_exec == float('inf'):
        return {'cost_bps': float('inf'), 'slippage_bps': float('inf'),
                'fee_bps': fee_bps, 'P_exec': float('inf')}

    sign = 1.0 if side == 'buy' else -1.0
    slippage_bps = sign * (P_exec - P_mid) / P_mid * 10_000
    cost_bps = slippage_bps + fee_bps
    return {'cost_bps': cost_bps, 'slippage_bps': slippage_bps,
            'fee_bps': fee_bps, 'P_exec': P_exec}


def _execute_trade_amm(pool, side: str, Q: float,
                       P_mid: float) -> dict:
    """
    Execute a trade on an AMM pool and compute Fennell cost.

    The pool is mutated (reserves change).  Cost is measured
    against the exogenous P_mid, not the pool's own mid.
    """
    if side == 'buy':
        quote = pool.quote_buy(Q, S_t=P_mid)
    else:
        quote = pool.quote_sell(Q, S_t=P_mid)

    P_exec = quote['exec_price']
    if P_exec == float('inf'):
        return {'cost_bps': float('inf'), 'slippage_bps': float('inf'),
                'fee_bps': quote['fee_bps'], 'P_exec': float('inf')}

    sign = 1.0 if side == 'buy' else -1.0
    slippage_bps = sign * (P_exec - P_mid) / P_mid * 10_000 - quote['fee_bps']
    slippage_bps = max(slippage_bps, 0.0)
    cost_bps = slippage_bps + quote['fee_bps']
    return {'cost_bps': cost_bps, 'slippage_bps': slippage_bps,
            'fee_bps': quote['fee_bps'], 'P_exec': P_exec}


# =====================================================================
#  5. MAIN FENNELL TEST RUNNER
# =====================================================================

class FennellConfig:
    """All parameters for a single Fennell test run."""

    def __init__(self,
                 # Price process
                 S0: float = 100.0,
                 mu: float = 0.0,
                 sigma: float = 0.01,
                 n_steps: int = 500,
                 seed: int = 42,
                 # Order flow
                 Q_buckets: list = None,
                 trades_per_bucket: int = 80,
                 # CLOB
                 clob_half_spread_bps: float = 1.5,
                 clob_depth_per_level: float = 100.0,
                 clob_n_levels: int = 10,
                 clob_fee_bps: float = 0.0,
                 # CPMM
                 cpmm_x: float = 5000.0,
                 cpmm_fee: float = 0.0,
                 # HFMM
                 hfmm_x: float = 5000.0,
                 hfmm_A: float = 50.0,
                 hfmm_fee: float = 0.0,
                 # Arbitrage & update_rate
                 arb_every: int = 1,
                 update_rate_every: int = 10,
                 # Label
                 label: str = 'default'):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.n_steps = n_steps
        self.seed = seed
        self.Q_buckets = Q_buckets or [1, 2, 5, 10, 20, 50]
        self.trades_per_bucket = trades_per_bucket
        self.clob_half_spread_bps = clob_half_spread_bps
        self.clob_depth_per_level = clob_depth_per_level
        self.clob_n_levels = clob_n_levels
        self.clob_fee_bps = clob_fee_bps
        self.cpmm_x = cpmm_x
        self.cpmm_fee = cpmm_fee
        self.hfmm_x = hfmm_x
        self.hfmm_A = hfmm_A
        self.hfmm_fee = hfmm_fee
        self.arb_every = arb_every
        self.update_rate_every = update_rate_every
        self.label = label


def run_fennell(cfg: FennellConfig) -> pd.DataFrame:
    """
    Run a single Fennell test.

    Returns a DataFrame with one row per trade:
        step, side, quantity, bucket, venue, P_mid, P_exec,
        cost_bps, slippage_bps, fee_bps
    """
    # 1. Generate price path
    prices = generate_gbm_path(cfg.S0, cfg.mu, cfg.sigma,
                               cfg.n_steps, cfg.seed)

    # 2. Generate order flow
    n_trades = len(cfg.Q_buckets) * cfg.trades_per_bucket
    orders = generate_order_flow(n_trades, cfg.Q_buckets, cfg.seed)

    # 3. Initialise venues
    S0 = prices[0]
    cpmm = CPMMPool(x=cfg.cpmm_x, y=cfg.cpmm_x * S0, fee=cfg.cpmm_fee)
    hfmm = HFMMPool(x=cfg.hfmm_x, y=cfg.hfmm_x * S0,
                     A=cfg.hfmm_A, fee=cfg.hfmm_fee, rate=S0)

    # 4. Run trades — distribute evenly across time steps
    records = []
    trades_per_step = max(1, len(orders) // cfg.n_steps)
    order_idx = 0

    for t in range(cfg.n_steps):
        P_mid = prices[t]

        # Arb: align AMM pools to fair price
        if t > 0 and t % cfg.arb_every == 0:
            cpmm.arbitrage_to_target(P_mid)
            hfmm.arbitrage_to_target(P_mid)

        # HFMM: periodically recenter rate
        if t > 0 and t % cfg.update_rate_every == 0:
            hfmm.update_rate(threshold=0.005)

        # Rebuild CLOB around current mid each step
        clob = _build_clob(P_mid, cfg.clob_half_spread_bps,
                           cfg.clob_depth_per_level, cfg.clob_n_levels)

        # Execute batch of trades at this step
        batch_end = min(order_idx + trades_per_step, len(orders))
        for i in range(order_idx, batch_end):
            order = orders[i]
            side = order['side']
            Q = order['quantity']

            # --- CLOB ---
            res_clob = _execute_trade_clob(clob, side, Q, P_mid,
                                           cfg.clob_fee_bps)
            records.append(dict(
                step=t, side=side, quantity=Q, bucket=Q,
                venue='CLOB', P_mid=P_mid,
                P_exec=res_clob['P_exec'],
                cost_bps=res_clob['cost_bps'],
                slippage_bps=res_clob['slippage_bps'],
                fee_bps=res_clob['fee_bps'],
            ))

            # --- CPMM ---
            # Use fresh copy for quoting to keep venues independent
            res_cpmm = _execute_trade_amm(cpmm, side, Q, P_mid)
            records.append(dict(
                step=t, side=side, quantity=Q, bucket=Q,
                venue='CPMM', P_mid=P_mid,
                P_exec=res_cpmm['P_exec'],
                cost_bps=res_cpmm['cost_bps'],
                slippage_bps=res_cpmm['slippage_bps'],
                fee_bps=res_cpmm['fee_bps'],
            ))

            # Execute on CPMM (mutate reserves)
            if side == 'buy':
                cpmm.execute_buy(Q)
            else:
                cpmm.execute_sell(Q)

            # --- HFMM ---
            res_hfmm = _execute_trade_amm(hfmm, side, Q, P_mid)
            records.append(dict(
                step=t, side=side, quantity=Q, bucket=Q,
                venue='HFMM', P_mid=P_mid,
                P_exec=res_hfmm['P_exec'],
                cost_bps=res_hfmm['cost_bps'],
                slippage_bps=res_hfmm['slippage_bps'],
                fee_bps=res_hfmm['fee_bps'],
            ))

            # Execute on HFMM (mutate reserves)
            if side == 'buy':
                hfmm.execute_buy(Q)
            else:
                hfmm.execute_sell(Q)

        order_idx = batch_end

    # Process any remaining orders at the last price
    if order_idx < len(orders):
        t = cfg.n_steps - 1
        P_mid = prices[t]
        clob = _build_clob(P_mid, cfg.clob_half_spread_bps,
                           cfg.clob_depth_per_level, cfg.clob_n_levels)
        for i in range(order_idx, len(orders)):
            order = orders[i]
            side = order['side']
            Q = order['quantity']

            res_clob = _execute_trade_clob(clob, side, Q, P_mid,
                                           cfg.clob_fee_bps)
            records.append(dict(
                step=t, side=side, quantity=Q, bucket=Q,
                venue='CLOB', P_mid=P_mid,
                P_exec=res_clob['P_exec'],
                cost_bps=res_clob['cost_bps'],
                slippage_bps=res_clob['slippage_bps'],
                fee_bps=res_clob['fee_bps'],
            ))

            res_cpmm = _execute_trade_amm(cpmm, side, Q, P_mid)
            records.append(dict(
                step=t, side=side, quantity=Q, bucket=Q,
                venue='CPMM', P_mid=P_mid,
                P_exec=res_cpmm['P_exec'],
                cost_bps=res_cpmm['cost_bps'],
                slippage_bps=res_cpmm['slippage_bps'],
                fee_bps=res_cpmm['fee_bps'],
            ))
            if side == 'buy':
                cpmm.execute_buy(Q)
            else:
                cpmm.execute_sell(Q)

            res_hfmm = _execute_trade_amm(hfmm, side, Q, P_mid)
            records.append(dict(
                step=t, side=side, quantity=Q, bucket=Q,
                venue='HFMM', P_mid=P_mid,
                P_exec=res_hfmm['P_exec'],
                cost_bps=res_hfmm['cost_bps'],
                slippage_bps=res_hfmm['slippage_bps'],
                fee_bps=res_hfmm['fee_bps'],
            ))
            if side == 'buy':
                hfmm.execute_buy(Q)
            else:
                hfmm.execute_sell(Q)

    df = pd.DataFrame(records)
    # Drop infinite costs (book exhaustion)
    df = df[df['cost_bps'] < 1e6].copy()
    return df


# =====================================================================
#  6. AGGREGATION — Fennell Tables
# =====================================================================

def aggregate_fennell(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trade-level data into Fennell summary table:
    avg cost, count, p25, median, p75 per venue × bucket.
    """
    grouped = df.groupby(['venue', 'bucket']).agg(
        avg_cost_bps=('cost_bps', 'mean'),
        avg_slippage_bps=('slippage_bps', 'mean'),
        avg_fee_bps=('fee_bps', 'mean'),
        count=('cost_bps', 'count'),
        p25=('cost_bps', lambda x: np.percentile(x, 25)),
        p50=('cost_bps', 'median'),
        p75=('cost_bps', lambda x: np.percentile(x, 75)),
    ).reset_index()
    return grouped


def print_fennell_table(agg: pd.DataFrame, label: str):
    """Pretty-print the Fennell summary table."""
    print(f"\n{'=' * 80}")
    print(f"  Fennell Table — {label}")
    print(f"{'=' * 80}")

    venues = ['CLOB', 'CPMM', 'HFMM']
    buckets = sorted(agg['bucket'].unique())

    # Header
    header = f"{'Q':>6}"
    for v in venues:
        header += f" | {v:>12} (slip+fee)"
    header += f" | {'Count':>6}"
    print(header)
    print('-' * len(header))

    for Q in buckets:
        row = f"{Q:>6.0f}"
        for v in venues:
            sub = agg[(agg['venue'] == v) & (agg['bucket'] == Q)]
            if len(sub) > 0:
                avg = sub['avg_cost_bps'].values[0]
                slip = sub['avg_slippage_bps'].values[0]
                fee = sub['avg_fee_bps'].values[0]
                row += f" | {avg:>6.2f} ({slip:>5.2f}+{fee:>4.1f})"
            else:
                row += f" | {'N/A':>12}          "
        # Count (same for all venues at same Q)
        sub_any = agg[(agg['venue'] == venues[0]) & (agg['bucket'] == Q)]
        cnt = int(sub_any['count'].values[0]) if len(sub_any) > 0 else 0
        row += f" | {cnt:>6}"
        print(row)

    print()


# =====================================================================
#  7. PLOTTING
# =====================================================================

def plot_cost_curves(agg: pd.DataFrame, label: str, filename: str):
    """
    Plot Average Execution Cost C_v(Q) for CLOB, CPMM, HFMM.
    X-axis: trade size Q (log-scale), Y-axis: avg cost (bps).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for venue, color, marker in [('CLOB', C_CLOB, 'o'),
                                  ('CPMM', C_CPMM, 's'),
                                  ('HFMM', C_HFMM, '^')]:
        sub = agg[agg['venue'] == venue].sort_values('bucket')
        if len(sub) == 0:
            continue
        ax.plot(sub['bucket'], sub['avg_cost_bps'],
                color=color, marker=marker, linewidth=2,
                markersize=7, label=venue)

    ax.set_xlabel('Trade Size Q (base units)', fontsize=12)
    ax.set_ylabel('Average Execution Cost (bps)', fontsize=12)
    ax.set_title(f'Fennell Test: Execution Cost $C_v(Q)$ — {label}', fontsize=13)
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  saved  {path}")


def plot_decomposition(agg: pd.DataFrame, label: str, filename: str):
    """
    Stacked bar chart: slippage vs fee for each venue × bucket.
    """
    venues = ['CLOB', 'CPMM', 'HFMM']
    buckets = sorted(agg['bucket'].unique())
    n_v = len(venues)
    n_b = len(buckets)

    fig, axes = plt.subplots(1, n_v, figsize=(4 * n_v, 5), sharey=True)
    if n_v == 1:
        axes = [axes]
    colors_slip = [C_CLOB, C_CPMM, C_HFMM]
    colors_fee = ['#90CAF9', '#FFE0B2', '#C8E6C9']

    for idx, (venue, ax) in enumerate(zip(venues, axes)):
        sub = agg[agg['venue'] == venue].sort_values('bucket')
        x = np.arange(len(sub))
        slip = sub['avg_slippage_bps'].values
        fee = sub['avg_fee_bps'].values

        ax.bar(x, slip, color=colors_slip[idx], label='Slippage', alpha=0.85)
        ax.bar(x, fee, bottom=slip, color=colors_fee[idx], label='Fee', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(b)}' for b in sub['bucket'].values], fontsize=9)
        ax.set_xlabel('Q')
        ax.set_title(venue, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Cost (bps)', fontsize=11)
    fig.suptitle(f'Cost Decomposition: Slippage vs Fee — {label}', fontsize=13)
    fig.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  saved  {path}")


def plot_percentiles(agg: pd.DataFrame, label: str, filename: str):
    """
    Plot p25 / p50 / p75 cost bands per venue.
    """
    venues = ['CLOB', 'CPMM', 'HFMM']
    colors = [C_CLOB, C_CPMM, C_HFMM]

    fig, ax = plt.subplots(figsize=(8, 5))
    for venue, color in zip(venues, colors):
        sub = agg[agg['venue'] == venue].sort_values('bucket')
        if len(sub) == 0:
            continue
        Q = sub['bucket'].values
        ax.fill_between(Q, sub['p25'].values, sub['p75'].values,
                        alpha=0.15, color=color)
        ax.plot(Q, sub['p50'].values, color=color, linewidth=2,
                marker='o', markersize=5, label=f'{venue} (median)')

    ax.set_xlabel('Trade Size Q', fontsize=12)
    ax.set_ylabel('Cost (bps)', fontsize=12)
    ax.set_title(f'Cost Percentiles (p25–p75) — {label}', fontsize=13)
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  saved  {path}")


# =====================================================================
#  8. FEE OPTIMIZATION (grid search)
# =====================================================================

def optimize_fee(cfg_base: FennellConfig, venue: str,
                 fee_grid: np.ndarray,
                 min_fee: float = 0.0001) -> tuple:
    """
    Grid search: find the fee ≥ min_fee that minimizes average cost across
    all buckets for a given venue ('cpmm' or 'hfmm').

    min_fee ensures the pool collects *some* revenue (a zero-fee pool is
    uneconomic for LPs).

    Returns (best_fee, best_avg_cost, results_df).
    """
    results = []
    for fee_val in fee_grid:
        cfg = copy.copy(cfg_base)
        if venue == 'cpmm':
            cfg.cpmm_fee = float(fee_val)
        else:
            cfg.hfmm_fee = float(fee_val)
        cfg.label = f'{venue}_opt_{fee_val:.5f}'

        df = run_fennell(cfg)
        sub = df[df['venue'] == venue.upper()]
        avg_cost = sub['cost_bps'].mean() if len(sub) > 0 else float('inf')
        results.append({'fee': float(fee_val), 'avg_cost_bps': avg_cost})

    res_df = pd.DataFrame(results)
    # Only consider fees above min_fee (LP viability constraint)
    viable = res_df[res_df['fee'] >= min_fee]
    if len(viable) == 0:
        viable = res_df
    best = viable.loc[viable['avg_cost_bps'].idxmin()]
    return float(best['fee']), float(best['avg_cost_bps']), res_df


# =====================================================================
#  9. FEE REGIME DEFINITIONS
# =====================================================================

def get_fee_regimes(optimized_cpmm_fee: float = 0.0,
                    optimized_hfmm_fee: float = 0.0) -> list:
    """
    Three fee regimes per Fennell:
      1. fee=0 (pure slippage)
      2. Realistic FX fees
      3. Optimized fees
    """
    return [
        {
            'label': 'fee=0 (pure slippage)',
            'clob_fee_bps': 0.0,
            'cpmm_fee': 0.0,
            'hfmm_fee': 0.0,
        },
        {
            'label': 'Realistic FX fees',
            'clob_fee_bps': 0.5,     # ~0.5 bps for CLOB
            'cpmm_fee': 0.003,       # 30 bps
            'hfmm_fee': 0.001,       # 10 bps
        },
        {
            'label': 'Optimized fees',
            'clob_fee_bps': 0.5,
            'cpmm_fee': optimized_cpmm_fee,
            'hfmm_fee': optimized_hfmm_fee,
        },
    ]


# =====================================================================
#  10. MAIN
# =====================================================================

def main():
    print("=" * 80)
    print("  FENNELL TESTS — Transaction Cost Analysis: AMM vs CLOB")
    print("  Methodology: James Fennell, UTS Honours Thesis 2024")
    print("=" * 80)

    # ── Base configuration ────────────────────────────────────────────
    base = FennellConfig(
        S0=100.0,
        mu=0.0,
        sigma=0.01,
        n_steps=500,
        seed=42,
        Q_buckets=[1, 2, 5, 10, 20, 50],
        trades_per_bucket=80,
        clob_half_spread_bps=1.5,
        clob_depth_per_level=100.0,
        clob_n_levels=10,
        cpmm_x=5000.0,
        hfmm_x=5000.0,
        hfmm_A=50.0,
        arb_every=1,
        update_rate_every=10,
    )

    # ── Step A: Fee optimization (run with fee=0 as base) ─────────────
    print("\n[1/4] Optimizing CPMM fee …")
    opt_cpmm_fee, opt_cpmm_cost, opt_cpmm_df = optimize_fee(
        base, 'cpmm',
        fee_grid=np.arange(0.0, 0.006, 0.0005),
    )
    print(f"       Best CPMM fee = {opt_cpmm_fee:.4f} "
          f"(avg cost = {opt_cpmm_cost:.2f} bps)")

    print("[2/4] Optimizing HFMM fee …")
    opt_hfmm_fee, opt_hfmm_cost, opt_hfmm_df = optimize_fee(
        base, 'hfmm',
        fee_grid=np.arange(0.0, 0.004, 0.0002),
    )
    print(f"       Best HFMM fee = {opt_hfmm_fee:.4f} "
          f"(avg cost = {opt_hfmm_cost:.2f} bps)")

    # Plot optimization curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(opt_cpmm_df['fee'] * 10_000, opt_cpmm_df['avg_cost_bps'],
             color=C_CPMM, marker='o', markersize=4)
    ax1.axvline(opt_cpmm_fee * 10_000, color='red', linestyle='--', alpha=0.6)
    ax1.set_xlabel('CPMM Fee (bps)')
    ax1.set_ylabel('Avg Cost (bps)')
    ax1.set_title('CPMM Fee Optimization')
    ax1.grid(True, alpha=0.3)

    ax2.plot(opt_hfmm_df['fee'] * 10_000, opt_hfmm_df['avg_cost_bps'],
             color=C_HFMM, marker='o', markersize=4)
    ax2.axvline(opt_hfmm_fee * 10_000, color='red', linestyle='--', alpha=0.6)
    ax2.set_xlabel('HFMM Fee (bps)')
    ax2.set_ylabel('Avg Cost (bps)')
    ax2.set_title('HFMM Fee Optimization')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fee_optimization.png')
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  saved  {path}")

    # ── Step B: Run three fee regimes ─────────────────────────────────
    regimes = get_fee_regimes(opt_cpmm_fee, opt_hfmm_fee)
    all_results = []

    print(f"\n[3/4] Running {len(regimes)} fee regimes …")
    for i, regime in enumerate(regimes):
        label = regime['label']
        print(f"\n  Regime {i+1}/{len(regimes)}: {label}")

        cfg = copy.copy(base)
        cfg.clob_fee_bps = regime['clob_fee_bps']
        cfg.cpmm_fee = regime['cpmm_fee']
        cfg.hfmm_fee = regime['hfmm_fee']
        cfg.label = label

        df = run_fennell(cfg)
        df['regime'] = label
        all_results.append(df)

        agg = aggregate_fennell(df)
        print_fennell_table(agg, label)

        # Plots
        safe_label = label.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '')
        plot_cost_curves(agg, label, f'cost_curve_{safe_label}.png')
        plot_decomposition(agg, label, f'decomposition_{safe_label}.png')
        plot_percentiles(agg, label, f'percentiles_{safe_label}.png')

        # CSV
        agg.to_csv(os.path.join(OUT_DIR, f'fennell_{safe_label}.csv'), index=False)

    # ── Step C: Combined comparison chart ─────────────────────────────
    print(f"\n[4/4] Building combined comparison …")

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, 'fennell_all_trades.csv'), index=False)

    # Combined cost curves (one subplot per regime)
    n_regimes = len(regimes)
    fig, axes = plt.subplots(1, n_regimes, figsize=(6 * n_regimes, 5), sharey=True)
    if n_regimes == 1:
        axes = [axes]

    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        label = regime['label']
        sub = combined[combined['regime'] == label]
        agg = aggregate_fennell(sub)

        for venue, color, marker in [('CLOB', C_CLOB, 'o'),
                                      ('CPMM', C_CPMM, 's'),
                                      ('HFMM', C_HFMM, '^')]:
            v_sub = agg[agg['venue'] == venue].sort_values('bucket')
            if len(v_sub) > 0:
                ax.plot(v_sub['bucket'], v_sub['avg_cost_bps'],
                        color=color, marker=marker, linewidth=2,
                        markersize=6, label=venue)

        ax.set_xlabel('Trade Size Q')
        ax.set_title(label, fontsize=11)
        ax.set_xscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Avg Execution Cost (bps)')
    fig.suptitle('Fennell Test: AMM vs CLOB across Fee Regimes',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    path = os.path.join(OUT_DIR, 'fennell_combined.png')
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  saved  {path}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  DONE — Results in output/fennell/")
    print("=" * 80)
    print(f"  Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"    {f}")
    print()


if __name__ == '__main__':
    main()
