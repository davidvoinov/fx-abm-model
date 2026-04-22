#!/usr/bin/env python3
"""
Unified Unit Test — single executable report covering all venues and routing.

Modes
-----
  python tests/unit_test.py           # text report only (no extra deps)
  python tests/unit_test.py --plots   # report + diagnostic plots & CSV
                                      # (requires numpy, pandas, matplotlib)

Sections
--------
1. CLOB VENUE
   - Mid-price & quoted spread from a 3-level book
   - Execution price (single-level, multi-level sweep, full sweep, beyond book)
   - Effective spread & half-spread
   - Price impact: partial fill, full level consumption, sell side
   - All-in cost = half_spread + fee (with and without venue/broker fees)
   - Depth queries (n_levels, corridor filtering)
   - Edge cases: Q=0, Q<0, empty book

2. CPMM VENUE
   - Invariant k = x·y preservation after single buy
   - Round-trip (buy + sell) reserve drift
   - Mid-price = y/x, small-Q execution ≈ mid
   - Slippage monotonicity across increasing Q
   - Fee decomposition: fee_bps = 10 000 × fee, cost = slippage + fee + gas
   - High-fee pool: delta_y = dy_eff / (1 − f)
   - Boundary volumes: Q=0, Q<0, Q≥x (infinity), failed buy preserves reserves
   - Volume–slippage profile: max_Q at threshold, monotonicity
   - Liquidity: L = sqrt(k), effective depth R = sqrt(x·y)/S

3. HFMM VENUE (StableSwap)
   - Invariant D preservation: after buy, sell, multi-swap round-trip
   - Local linearity near parity: tiny Q → exec ≈ mid, |slippage| < 1 bps
   - HFMM vs CPMM slippage comparison (HFMM should be lower for same reserves)
   - CPMM-like transition under large Q and low amplification (A=1)
   - No negative reserves even at 95% depletion
   - Cost & fee decomposition: fee_bps, cost = slippage + fee + gas, exec = Δy/Q
   - D-solver numerical stability across varied (x1, x2, A) configurations
   - get_y round-trip: solve D then recover x2
   - Balanced mid-price: x=y → marginal price ≈ 1.0
   - Rate update: no-drift no-op; large trade triggers re-centering

4. VENUE ROUTING
   - Softmax mechanics: lower cost → higher weight, β concentration,
     equal-cost uniformity, inf cost exclusion, theoretical probability match
   - Two-step selection: amm_share_pct controls CLOB/AMM split,
     intra-AMM CPMM vs HFMM distribution, deterministic mode
   - Bias & noise: cpmm_bias_bps shifts flow, noise preserves diversity
   - estimate_costs() returns correct keys

Diagnostic Plots (--plots)
--------------------------
Generates PNG dashboards and CSV tables into output/unit_tests/:
  - clob_diagnostics.png  — mid/spread/espr errors, impact & cost vs Q
  - cpmm_diagnostics.png  — invariant k, round-trip drift, slippage & fees vs Q
  - hfmm_diagnostics.png  — D-invariant vs step, solver convergence,
                            exec price & slippage HFMM vs CPMM
  - routing_diagnostics.png — softmax theory vs empirical, regime transitions,
                              two-step routing, venue cost comparison
"""
import sys, os, math, random, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from AgentBasedModel.utils.orders import Order, OrderList
from AgentBasedModel.agents.agents import ExchangeAgent, Trader, Fundamentalist, MarketMaker, AMMArbitrageur, AMMProvider
from AgentBasedModel.venues.clob import CLOBVenue
from AgentBasedModel.venues.amm import (
    CPMMPool, HFMMPool,
    _hfmm_get_D, _hfmm_get_y, _hfmm_mid_price,
)
from AgentBasedModel.environment.processes import MarketEnvironment
from AgentBasedModel.metrics.resilience import (
    composite_resilience_metrics,
    kaplan_meier_curve,
    price_resilience_metrics,
    series_resilience_metrics,
)
from AgentBasedModel.metrics.statistics import (
    bootstrap_diff_ci,
    bootstrap_mean_ci,
    bootstrap_paired_diff_ci,
    independent_permutation_test,
    paired_permutation_test,
)
import main as main_module
from collections import Counter

W = 72
PASS = "✓ PASS"
FAIL = "✗ FAIL"
total_pass = 0
total_fail = 0


# ── Reporting helpers ────────────────────────────────────────────────

def check(label, actual, expected, rel=1e-6, abs_tol=None):
    global total_pass, total_fail
    if expected == float('inf'):
        ok = actual == float('inf')
    elif abs_tol is not None:
        ok = abs(actual - expected) <= abs_tol
    elif expected == 0:
        ok = abs(actual) < 1e-9
    else:
        ok = abs(actual - expected) / abs(expected) < rel
    tag = PASS if ok else FAIL
    if ok:
        total_pass += 1
    else:
        total_fail += 1
    print(f"    {tag}  {label}")
    print(f"           expected = {expected}")
    print(f"           actual   = {actual}")
    return ok


def check_bool(label, condition, detail=""):
    global total_pass, total_fail
    tag = PASS if condition else FAIL
    if condition:
        total_pass += 1
    else:
        total_fail += 1
    print(f"    {tag}  {label}")
    if detail:
        print(f"           {detail}")
    return condition


def section(title):
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")


def subsection(title):
    print(f"\n  ── {title} {'─' * max(0, W - len(title) - 6)}")


# ── Exchange / book helpers ──────────────────────────────────────────

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


# =====================================================================
#  1. CLOB VENUE
# =====================================================================

def test_clob():
    """CLOB venue: mid/spread, execution, impact, cost, depth, edge cases."""
    section("CLOB VENUE — Unit Tests")

    ex = _make_exchange(BIDS, ASKS)
    clob = CLOBVenue(ex, f_broker=0.0, f_venue=0.0)
    clob_fee = CLOBVenue(_make_exchange(BIDS, ASKS), f_broker=2.0, f_venue=1.0)

    # ── 1. Mid & spread ─────────────────────────────────────────────
    # Verifies mid = (best_bid + best_ask) / 2 and quoted spread
    # in basis points from a symmetric 3-level book.
    subsection("1. Mid-price & Quoted Spread")
    mid = clob.mid_price()
    check("mid_price = (99+101)/2 = 100.0", mid, 100.0)
    qspr = clob.quoted_spread_bps()
    check("qspr = 10000×(101−99)/100 = 200 bps", qspr, 200.0)

    # ── 2. Execution price ───────────────────────────────────────────
    # Tests VWAP calculation for small (single-level), multi-level,
    # full-sweep and beyond-book order sizes.
    subsection("2. Execution Price & Effective Spread")

    q5 = clob.quote_buy(5)
    check("Buy Q=5: exec_price = best_ask = 101.0", q5['exec_price'], 101.0)
    half_5 = 10_000 * (101.0 - 100.0) / 100.0
    check(f"Buy Q=5: half_spread = {half_5} bps", q5['half_spread_bps'], half_5)
    check(f"Buy Q=5: espr = {2*half_5} bps", q5['espr_bps'], 2 * half_5)

    q25 = clob.quote_buy(25)
    vwap25 = (10 * 101.0 + 15 * 102.0) / 25.0
    check(f"Buy Q=25: VWAP = {vwap25:.4f} (sweeps 2 levels)", q25['exec_price'], vwap25)
    half_25 = 10_000 * (vwap25 - 100.0) / 100.0
    check(f"Buy Q=25: half_spread = {half_25:.1f} bps", q25['half_spread_bps'], half_25)

    q60 = clob.quote_buy(60)
    vwap60 = (10*101 + 20*102 + 30*103) / 60.0
    check(f"Buy Q=60: VWAP = {vwap60:.4f} (all 3 levels)", q60['exec_price'], vwap60)

    q100 = clob.quote_buy(100)
    check("Buy Q=100: cost = inf (exceeds book)", q100['cost_bps'], float('inf'))

    # ── 3. Price impact ──────────────────────────────────────────────
    # Measures new-mid shift after hypothetical fill.
    # Q=5 doesn't exhaust best ask → 0 impact.
    # Q=10 consumes best ask → new mid shifts by 50 bps.
    subsection("3. Price Impact")

    q5 = clob.quote_buy(5)
    check("Buy Q=5: impact = 0 (best ask survives)", q5['impact_bps'], 0.0)

    q10 = clob.quote_buy(10)
    check("Buy Q=10: impact = 10000×(100.5−100)/100 = 50 bps", q10['impact_bps'], 50.0)

    qs10 = clob.quote_sell(10)
    check("Sell Q=10: impact = 10000×(99.5−100)/100 = −50 bps", qs10['impact_bps'], -50.0)

    # ── 4. All-in cost ───────────────────────────────────────────────
    # cost_bps = half_spread_bps + fee_bps.
    # Tested with zero fees and with broker=2 + venue=1 = 3 bps.
    subsection("4. All-in Cost = half_spread + fee")

    q5_nf = clob.quote_buy(5)
    check("Zero fee: cost = half_spread", q5_nf['cost_bps'], q5_nf['half_spread_bps'])

    q5_f = clob_fee.quote_buy(5)
    check("Fee = 2+1 = 3 bps", q5_f['fee_bps'], 3.0)
    check("Cost = half_spread + 3", q5_f['cost_bps'], q5_f['half_spread_bps'] + 3.0)

    for Q in [1, 5, 10, 25]:
        q = clob_fee.quote_buy(Q)
        if q['cost_bps'] < float('inf'):
            check_bool(
                f"Component sum Q={Q}: cost == half_spread + fee",
                abs(q['cost_bps'] - (q['half_spread_bps'] + q['fee_bps'])) < 1e-9,
                f"cost={q['cost_bps']:.4f}, half={q['half_spread_bps']:.4f}, fee={q['fee_bps']:.4f}"
            )

    # ── 5. Depth ─────────────────────────────────────────────────────
    # depth(n_levels) returns price-volume pairs.
    # total_depth(bps_from_mid) filters levels within ± corridor.
    subsection("5. Depth")

    d3 = clob.depth(n_levels=3)
    check("Ask levels count = 3", len(d3['ask']), 3)
    ask_vols = [v for _, v in d3['ask']]
    check_bool("Ask volumes = [10, 20, 30]",
               ask_vols == [10, 20, 30],
               f"actual = {ask_vols}")

    td_wide = clob.total_depth(bps_from_mid=10_000)
    check("Wide corridor: bid depth = 60", td_wide['bid'], 60.0)
    check("Wide corridor: ask depth = 60", td_wide['ask'], 60.0)

    td_narrow = clob.total_depth(bps_from_mid=100)
    check("±100 bps: bid depth = 10 (only level 99)", td_narrow['bid'], 10.0)
    check("±100 bps: ask depth = 10 (only level 101)", td_narrow['ask'], 10.0)

    # ── Edge cases ───────────────────────────────────────────────────
    # Boundary inputs: zero/negative Q, empty order book.
    subsection("Edge Cases")
    check("Q=0: cost = 0", clob.quote_buy(0)['cost_bps'], 0.0)
    check("Q<0: cost = 0", clob.quote_buy(-5)['cost_bps'], 0.0)

    ex_empty = ExchangeAgent.__new__(ExchangeAgent)
    ex_empty.name = "Empty"
    ex_empty.order_book = {'bid': OrderList('bid'), 'ask': OrderList('ask')}
    ex_empty.dividend_book = [100.0]
    ex_empty.risk_free = 0.0
    ex_empty.transaction_cost = 0.0
    c_empty = CLOBVenue(ex_empty)
    check("Empty book: mid = 100 (fallback)", c_empty.mid_price(), 100.0)
    check("Empty book: qspr = inf", c_empty.quoted_spread_bps(), float('inf'))

    inserted = OrderList('bid')
    inserted.insert(Order(99.0, 10, 'bid', None))
    check_bool(
        "Empty insert keeps single isolated node",
        inserted.first is inserted.last and inserted.first is not None
        and inserted.first.left is None and inserted.first.right is None,
        (
            f"first={inserted.first}, last={inserted.last}, "
            f"left={getattr(inserted.first, 'left', None)}, "
            f"right={getattr(inserted.first, 'right', None)}"
        ),
    )

    subsection("Partial Anchor")
    ex_anchor = _make_exchange(BIDS, ASKS)
    ex_anchor._anchor_strength = 0.25
    ex_anchor._anchor_threshold_bps = 0.0
    ex_anchor.rebalance_background_liquidity = lambda *args, **kwargs: None
    ex_anchor.recenter_book(110.0, reprice_prob=1.0,
                            reprice_noise_bps=0.0,
                            anchor_strength=0.25,
                            min_reprice_gap_bps=0.0)
    check("Anchor moves only 25% toward fair price",
          ex_anchor.price(), 102.5, abs_tol=1e-9)


# =====================================================================
#  2. CPMM VENUE
# =====================================================================

def test_cpmm():
    """CPMM (x·y=k): invariant, price, slippage, fees, boundary, volume profile."""
    section("CPMM VENUE — Unit Tests")

    # ── 1. Invariant ─────────────────────────────────────────────────
    # k = x·y must hold after trades. Round-trip should approximately
    # restore reserves (small drift due to fee extraction).
    subsection("1. Invariant k = x·y & Round-trip")

    p = CPMMPool(x=1000.0, y=100_000.0, fee=0.003)
    check("Initial k = 1000 × 100000 = 1e8", p.k, 1e8)

    k0 = p.k
    p.execute_buy(1.0)
    check_bool("After buy(1): x·y ≈ k₀",
               abs(p.x * p.y - k0) / k0 < 1e-6,
               f"x·y = {p.x*p.y:.2f}, k₀ = {k0:.2f}, Δ = {abs(p.x*p.y - k0):.4f}")

    p2 = CPMMPool(x=1000.0, y=100_000.0, fee=0.003)
    x0, y0 = p2.x, p2.y
    p2.execute_buy(0.1)
    p2.execute_sell(0.1)
    check_bool(f"Round-trip buy(0.1)+sell(0.1): x ≈ x₀",
               abs(p2.x - x0) / x0 < 0.01,
               f"x: {x0:.4f} → {p2.x:.4f} (Δ={abs(p2.x-x0):.6f})")
    check_bool(f"Round-trip buy(0.1)+sell(0.1): y ≈ y₀",
               abs(p2.y - y0) / y0 < 0.01,
               f"y: {y0:.4f} → {p2.y:.4f} (Δ={abs(p2.y-y0):.6f})")

    # ── 2. Price & slippage ──────────────────────────────────────────
    # mid_price = y/x. Small order → exec ≈ mid. Larger orders push
    # price along the x·y=k curve. Slippage increases monotonically.
    subsection("2. Price & Slippage")

    p = CPMMPool(x=1000.0, y=100_000.0, fee=0.003)
    check("mid_price = y/x = 100.0", p.mid_price(), 100.0)

    q_small = p.quote_buy(0.01, S_t=100.0)
    check_bool(f"Buy Q=0.01: exec ≈ mid",
               abs(q_small['exec_price'] - 100.0) / 100.0 < 0.01,
               f"exec = {q_small['exec_price']:.6f}")
    check_bool(f"Buy Q=0.01: |slippage| < 1 bps",
               abs(q_small['slippage_bps']) < 1.0,
               f"slippage = {q_small['slippage_bps']:.4f} bps")

    q_big = p.quote_buy(100.0, S_t=100.0)
    check_bool(f"Buy Q=100: exec > 110 (material slippage)",
               q_big['exec_price'] > 110.0,
               f"exec = {q_big['exec_price']:.4f}")

    print("\n    Slippage monotonicity table:")
    print(f"    {'Q':>8s}  {'slippage (bps)':>16s}  {'cost (bps)':>12s}  {'exec_price':>12s}")
    prev_slip = -1
    mono_ok = True
    for Q in [1, 5, 10, 50, 100]:
        q = p.quote_buy(Q, S_t=100.0)
        if q['slippage_bps'] <= prev_slip:
            mono_ok = False
        prev_slip = q['slippage_bps']
        print(f"    {Q:>8.0f}  {q['slippage_bps']:>16.2f}  {q['cost_bps']:>12.2f}  {q['exec_price']:>12.4f}")
    check_bool("Slippage monotonically increasing", mono_ok)

    # ── 3. Fees ──────────────────────────────────────────────────────
    # fee_bps = 10 000 × pool.fee. All-in cost = slippage + fee + gas.
    # High-fee pool: effective delta_y includes fee gross-up.
    subsection("3. Fees & All-in Cost")

    q5 = p.quote_buy(5.0, S_t=100.0)
    check("fee_bps = 10000×0.003 = 30", q5['fee_bps'], 30.0)
    check("cost = slippage + fee + gas",
          q5['cost_bps'], q5['slippage_bps'] + q5['fee_bps'] + p.gas_cost_bps)

    p10 = CPMMPool(x=1000, y=100_000, fee=0.10)
    Q = 10.0
    x_new = p10.x - Q
    y_new = p10.k / x_new
    dy_eff = y_new - p10.y
    dy_with_fee = dy_eff / (1 - 0.10)
    q10 = p10.quote_buy(Q, S_t=p10.mid_price())
    check(f"Fee=10%: delta_y = dy_eff/(1−f) = {dy_with_fee:.4f}", q10['delta_y'], dy_with_fee)

    # ── 4. Boundary ──────────────────────────────────────────────────
    # Edge sizes: Q=0, Q<0 → zero cost. Q ≥ x → inf (drains pool).
    # Failed buy must not alter reserves.
    subsection("4. Boundary Volumes")

    p = CPMMPool(x=1000.0, y=100_000.0, fee=0.003)
    check("Q=0: cost = 0", p.quote_buy(0)['cost_bps'], 0.0)
    check("Q<0: cost = 0", p.quote_buy(-5)['cost_bps'], 0.0)
    check("Q≥x: cost = inf", p.quote_buy(p.x)['cost_bps'], float('inf'))

    x_before, y_before = p.x, p.y
    p.execute_buy(p.x + 10)
    check_bool("Failed buy: reserves unchanged",
               p.x == x_before and p.y == y_before,
               f"x: {x_before} → {p.x}, y: {y_before} → {p.y}")

    p3 = CPMMPool(x=1000.0, y=100_000.0, fee=0.003)
    p3.execute_sell(500.0)
    check_bool("Big sell: x > 0 and y > 0",
               p3.x > 0 and p3.y > 0,
               f"x = {p3.x:.4f}, y = {p3.y:.4f}")

    # ── 5. Volume–slippage profile ───────────────────────────────────
    # volume_slippage_max_Q finds the largest Q whose cost stays under
    # a threshold. Higher thresholds allow bigger Q.
    subsection("5. Volume–Slippage Profile")

    p = CPMMPool(x=1000.0, y=100_000.0, fee=0.003)
    threshold = 100.0
    max_q = p.volume_slippage_max_Q(threshold, S_t=100.0)
    cost_at_max = p.quote_buy(max_q, S_t=100.0)['cost_bps']
    print(f"    threshold = {threshold} bps → max_Q = {max_q:.4f}")
    print(f"    cost at max_Q = {cost_at_max:.4f} bps")
    check_bool(f"cost ≤ threshold + 1 bps",
               cost_at_max <= threshold + 1.0,
               f"{cost_at_max:.4f} ≤ {threshold + 1.0}")

    q50 = p.volume_slippage_max_Q(50, S_t=100.0)
    q200 = p.volume_slippage_max_Q(200, S_t=100.0)
    check_bool(f"Higher threshold → bigger Q: {q200:.2f} > {q50:.2f}",
               q200 > q50)

    # ── Liquidity ────────────────────────────────────────────────────
    # L = sqrt(k), effective depth R = sqrt(x·y) / S_t.
    subsection("Liquidity Management")
    p = CPMMPool(x=1000.0, y=100_000.0, fee=0.003)
    check("L = sqrt(k)", p.liquidity_measure(), math.sqrt(p.k))
    check("R = sqrt(x·y)/S", p.effective_depth(100.0),
          math.sqrt(p.x * p.y) / 100.0)


# =====================================================================
#  3. HFMM VENUE (StableSwap)
# =====================================================================

def test_hfmm():
    """HFMM (StableSwap): invariant D, linearity, transition, fees, stability."""
    section("HFMM VENUE — Unit Tests")

    # ── 1. Invariant D ───────────────────────────────────────────────
    # The StableSwap invariant D must be preserved after any swap.
    # Re-solving for D from updated normalised reserves should match
    # the original D within tolerance.
    subsection("1. Invariant D Preservation")

    p = HFMMPool(x=1000.0, y=100_000.0, A=100.0, fee=0.001, rate=100.0)
    D0 = p.D
    print(f"    Initial D = {D0:.6f}")
    check_bool("D > 0", D0 > 0)

    p.execute_buy(1.0)
    p._sync_norm()
    D_re = _hfmm_get_D(p._xn, p._yn, p.A)
    check(f"After buy(1): D recomputed ≈ D₀", D_re, D0, rel=1e-6)

    p2 = HFMMPool(x=1000.0, y=100_000.0, A=100.0, fee=0.001, rate=100.0)
    D0_2 = p2.D
    p2.execute_sell(1.0)
    p2._sync_norm()
    D_re2 = _hfmm_get_D(p2._xn, p2._yn, p2.A)
    check(f"After sell(1): D recomputed ≈ D₀", D_re2, D0_2, rel=1e-6)

    p3 = HFMMPool(x=1000.0, y=100_000.0, A=100.0, fee=0.001, rate=100.0)
    D0_3 = p3.D
    for _ in range(10):
        p3.execute_buy(0.5)
    for _ in range(10):
        p3.execute_sell(0.5)
    p3._sync_norm()
    D_re3 = _hfmm_get_D(p3._xn, p3._yn, p3.A)
    check(f"10×buy + 10×sell: D drift < 0.01%", D_re3, D0_3, rel=1e-4)

    # ── 2. Local linearity ───────────────────────────────────────────
    # Near parity the StableSwap curve is approximately linear, so
    # tiny orders should fill at ≈ mid price with negligible slippage.
    # HFMM must also beat CPMM on slippage at comparable reserves.
    subsection("2. Local Linearity Near Parity")

    p = HFMMPool(x=1000.0, y=100_000.0, A=100.0, fee=0.001, rate=100.0)
    S = p.mid_price()
    q_tiny = p.quote_buy(0.01, S_t=S)
    check_bool(f"Buy Q=0.01: exec ≈ mid (rel < 1%)",
               abs(q_tiny['exec_price'] - S) / S < 0.01,
               f"exec = {q_tiny['exec_price']:.6f}, mid = {S:.6f}")
    check_bool(f"|slippage| < 1 bps",
               abs(q_tiny['slippage_bps']) < 1.0,
               f"slippage = {q_tiny['slippage_bps']:.4f} bps")

    cpmm = CPMMPool(x=1000, y=100_000, fee=0.0)
    hfmm = HFMMPool(x=1000, y=100_000, A=100, fee=0.0, rate=100.0)
    Q = 10.0
    slip_c = cpmm.quote_buy(Q, 100.0)['slippage_bps']
    slip_h = hfmm.quote_buy(Q, 100.0)['slippage_bps']
    check_bool(f"HFMM slippage < CPMM at Q={Q}",
               slip_h < slip_c,
               f"HFMM={slip_h:.2f} bps, CPMM={slip_c:.2f} bps")

    # ── 3. Transition to CPMM-like ───────────────────────────────────
    # With large Q, even high-A pools exhibit material slippage.
    # With A=1 the StableSwap degenerates toward the x·y=k curve,
    # producing slippage comparable to CPMM (within 3×).
    subsection("3. CPMM-like Transition (large Q / low A)")

    p = HFMMPool(x=1000.0, y=100_000.0, A=100.0, fee=0.001, rate=100.0)
    q200 = p.quote_buy(200.0, S_t=p.mid_price())
    check_bool(f"Buy Q=200: slippage > 5 bps (material)",
               q200['slippage_bps'] > 5,
               f"slippage = {q200['slippage_bps']:.2f} bps")

    p_low = HFMMPool(x=1000, y=100_000, A=1.0, fee=0.0, rate=100.0)
    slip_low = p_low.quote_buy(50, 100.0)['slippage_bps']
    slip_cpmm = CPMMPool(x=1000, y=100_000, fee=0.0).quote_buy(50, 100.0)['slippage_bps']
    check_bool(f"A=1 slippage ≈ CPMM (within 3×)",
               abs(slip_low - slip_cpmm) / max(slip_cpmm, 1e-9) < 3.0,
               f"A=1: {slip_low:.2f} bps, CPMM: {slip_cpmm:.2f} bps")

    p_big = HFMMPool(x=1000.0, y=100_000.0, A=100.0, fee=0.001, rate=100.0)
    p_big.execute_buy(p_big.x * 0.95)
    check_bool("95% buy: x > 0 and y > 0",
               p_big.x > 0 and p_big.y > 0,
               f"x={p_big.x:.4f}, y={p_big.y:.4f}")

    # ── 4. Cost / fee ────────────────────────────────────────────────
    # fee_bps = 10 000 × pool.fee. All-in cost = slippage + fee + gas.
    # exec_price = delta_y / Q by definition.
    subsection("4. Cost & Fee Decomposition")

    p = HFMMPool(x=1000.0, y=100_000.0, A=100.0, fee=0.001, rate=100.0)
    q5 = p.quote_buy(5.0, S_t=p.mid_price())
    check("fee_bps = 10000 × 0.001 = 10", q5['fee_bps'], 10.0)
    check("cost = slippage + fee + gas",
          q5['cost_bps'], q5['slippage_bps'] + q5['fee_bps'] + p.gas_cost_bps)
    check("exec_price = delta_y / Q",
          q5['exec_price'], q5['delta_y'] / 5.0)

    print("\n    Cost table (buy):")
    print(f"    {'Q':>8s}  {'slippage':>10s}  {'fee':>8s}  {'cost':>10s}  {'exec_price':>12s}")
    for Q in [0.5, 1, 5, 10, 50]:
        q = p.quote_buy(Q, S_t=p.mid_price())
        if q['cost_bps'] < float('inf'):
            print(f"    {Q:>8.1f}  {q['slippage_bps']:>10.2f}  {q['fee_bps']:>8.1f}  {q['cost_bps']:>10.2f}  {q['exec_price']:>12.4f}")

    # ── 5. Numerical stability ───────────────────────────────────────
    # The Newton-Raphson D-solver must converge for extreme reserve
    # ratios and amplification parameters. get_y must recover the
    # second coordinate from D.
    subsection("5. D-solver Numerical Stability")

    configs = [
        (100, 100, 1), (100, 100, 10), (100, 100, 1000),
        (1, 1_000_000, 50), (1_000_000, 1, 50),
        (0.001, 0.001, 100), (1e8, 1e8, 500),
    ]
    for x1, x2, A in configs:
        D = _hfmm_get_D(x1, x2, A)
        ok = D > 0 and math.isfinite(D)
        check_bool(f"D({x1}, {x2}, A={A}) = {D:.6g}", ok)

    print("\n    get_y round-trip:")
    for x1, x2, A in [(100, 100, 1), (100, 100, 100), (50, 200, 50)]:
        D = _hfmm_get_D(x1, x2, A)
        y_rec = _hfmm_get_y(x1, D, A)
        check(f"  get_y({x1}, D, {A}) ≈ {x2}", y_rec, x2, rel=1e-8)

    x, y, A = 100.0, 100.0, 50.0
    D = _hfmm_get_D(x, y, A)
    mp = _hfmm_mid_price(x, y, A, D)
    check("Balanced (x=y=100): marginal price ≈ 1.0", mp, 1.0, rel=1e-6)

    # ── Rate update ──────────────────────────────────────────────────
    # If the pool mid is close to rate (within threshold), update_rate
    # is a no-op. After a large trade that drifts mid away, the rate
    # re-centers.
    subsection("Rate Update")
    p = HFMMPool(x=1000.0, y=100_000.0, A=100.0, fee=0.001, rate=100.0)
    r0 = p.rate
    p.update_rate(threshold=0.01)
    check_bool("No drift → rate unchanged", p.rate == r0)

    p_low = HFMMPool(x=1000.0, y=100_000.0, A=1.0, fee=0.001, rate=100.0)
    p_low.execute_buy(p_low.x * 0.40)
    mp = p_low.mid_price()
    drift = abs(mp - p_low.rate) / p_low.rate
    print(f"    After buy(40%) [A=1]: mid = {mp:.4f}, rate = {p_low.rate:.4f}, drift = {drift*100:.2f}%")
    check_bool(f"drift > 1%", drift > 0.01, f"drift = {drift*100:.2f}%")
    p_low.update_rate(threshold=0.01)
    check_bool(f"After update_rate: rate ≈ mid",
               abs(p_low.rate - mp) / mp < 1e-6,
               f"rate = {p_low.rate:.4f}, mid = {mp:.4f}")


# =====================================================================
#  4. VENUE ROUTING
# =====================================================================

def test_routing():
    """Routing: softmax, two-step CLOB/AMM split, bias, noise, estimate_costs."""
    section("VENUE ROUTING — Unit Tests")

    def _make_exchange_r():
        return _make_exchange([(99, 50), (98, 50)], [(101, 50), (102, 50)])

    def _make_trader(amm_share_pct=50, beta_amm=0.05, bias=0.0, noise=0.0,
                     det=False, rule='fixed_share', ex=None):
        ex = ex or _make_exchange_r()
        clob = CLOBVenue(ex)
        cpmm = CPMMPool(x=1000, y=100_000, fee=0.003)
        hfmm = HFMMPool(x=1000, y=100_000, A=100, fee=0.001, rate=100.0)
        return Trader(market=ex, cash=1e6, assets=100, clob=clob,
                      amm_pools={'cpmm': cpmm, 'hfmm': hfmm},
                      amm_share_pct=amm_share_pct, venue_choice_rule=rule,
                      deterministic_venue=det,
                      beta_amm=beta_amm, cpmm_bias_bps=bias, cost_noise_std=noise)

    # ── 1. Softmax ───────────────────────────────────────────────────
    # Verifies the softmax venue picker: lower cost ↦ higher probability,
    # higher β ↦ sharper concentration, equal costs ↦ uniform,
    # inf cost ↦ zero weight, observed frequencies ≈ analytic formula.
    subsection("1. Softmax Mechanics")

    costs = {'A': 10.0, 'B': 20.0, 'Z': 15.0}
    beta = 0.1
    min_c = min(costs.values())
    weights = {k: math.exp(-beta * (c - min_c)) for k, c in costs.items()}
    total = sum(weights.values())
    probs = {k: w / total for k, w in weights.items()}

    random.seed(42)
    N = 50000
    counts = Counter(Trader._softmax_pick(costs, beta) for _ in range(N))
    print(f"    Softmax(β={beta}) with costs = {costs}")
    print(f"    {'venue':>6s}  {'theoretical':>12s}  {'observed':>10s}  {'Δ':>8s}")
    for k in costs:
        obs = counts[k] / N
        print(f"    {k:>6s}  {probs[k]:>12.4f}  {obs:>10.4f}  {obs-probs[k]:>+8.4f}")
        check_bool(f"  P({k}): |obs − theory| < 0.02",
                   abs(obs - probs[k]) < 0.02,
                   f"theory={probs[k]:.4f}, obs={obs:.4f}")

    random.seed(42)
    inf_ok = all(Trader._softmax_pick({'A': 10, 'B': float('inf')}, 0.1) == 'A'
                 for _ in range(100))
    check_bool("inf cost never picked (100 trials)", inf_ok)

    # ── 2. Two-step ──────────────────────────────────────────────────
    # Step 1: coin flip with amm_share_pct probability for AMM vs CLOB.
    # Step 2 (if AMM): softmax between CPMM and HFMM pools.
    # Tests: low amm_share → CLOB > 90%, 50/50 split, intra-AMM dist,
    # deterministic mode always picks argmin.
    subsection("2. Two-step CLOB/AMM Selection")

    random.seed(42)
    N = 3000
    trader5 = _make_trader(amm_share_pct=5)
    clob_share = sum(1 for _ in range(N) if trader5.choose_venue(1.0) == 'clob') / N
    check_bool(f"amm_share=5%: CLOB share > 90%",
               clob_share > 0.90,
               f"CLOB share = {clob_share*100:.1f}%")

    random.seed(42)
    trader50 = _make_trader(amm_share_pct=50)
    amm_share = 1.0 - sum(1 for _ in range(N) if trader50.choose_venue(1.0) == 'clob') / N
    check_bool(f"amm_share=50%: AMM share ≈ 50%",
               abs(amm_share - 0.50) < 0.05,
               f"AMM share = {amm_share*100:.1f}%")

    random.seed(42)
    trader100 = _make_trader(amm_share_pct=100, beta_amm=0.05)
    picks = [trader100.choose_venue(1.0) for _ in range(N)]
    cn_c = picks.count('cpmm')
    cn_h = picks.count('hfmm')
    print(f"\n    amm_share=100%: CPMM={cn_c} ({cn_c/N*100:.1f}%), HFMM={cn_h} ({cn_h/N*100:.1f}%)")
    check_bool("Both CPMM and HFMM get flow", cn_c > 0 and cn_h > 0)
    check_bool("HFMM > CPMM (lower fee)", cn_h > cn_c,
               f"HFMM={cn_h}, CPMM={cn_c}")

    trader_det = _make_trader(det=True)
    costs_d = trader_det.estimate_costs(1.0, 'buy')
    cheapest = min(costs_d, key=costs_d.get)
    all_same = all(trader_det.choose_venue(1.0) == cheapest for _ in range(50))
    check_bool(f"Deterministic: always picks {cheapest}",
               all_same, f"costs = {costs_d}")

    # ── 3. Bias & noise ──────────────────────────────────────────────
    # cpmm_bias_bps subtracts from perceived CPMM cost, increasing its
    # selection frequency. Cost noise adds N(0,[σ]) jitter per quote,
    # ensuring non-zero flow to every viable venue.
    subsection("3. Bias & Noise")

    random.seed(42)
    N = 5000
    t_no = _make_trader(amm_share_pct=100, bias=0.0, noise=0.0)
    t_bi = _make_trader(amm_share_pct=100, bias=50.0, noise=0.0)
    no_cpmm = sum(1 for _ in range(N) if t_no.choose_venue(1.0) == 'cpmm')
    random.seed(42)
    bi_cpmm = sum(1 for _ in range(N) if t_bi.choose_venue(1.0) == 'cpmm')
    check_bool(f"cpmm_bias=50 bps increases CPMM share",
               bi_cpmm > no_cpmm,
               f"no_bias={no_cpmm}, bias={bi_cpmm}")

    random.seed(42)
    t_noise = _make_trader(amm_share_pct=100, noise=3.0, beta_amm=0.05)
    picks_n = Counter(t_noise.choose_venue(1.0) for _ in range(N))
    check_bool("Noise: both venues get flow",
               picks_n['cpmm'] > 0 and picks_n['hfmm'] > 0,
               f"CPMM={picks_n['cpmm']}, HFMM={picks_n['hfmm']}")

    random.seed(42)
    thin_ex = _make_exchange([(99, 5)], [(101, 5)])
    fixed_thin = _make_trader(amm_share_pct=5, rule='fixed_share', ex=thin_ex)
    liq_thin = _make_trader(amm_share_pct=5, rule='liquidity_aware', ex=_make_exchange([(99, 5)], [(101, 5)]))
    q_large = 20.0
    fixed_amm_share = sum(1 for _ in range(N) if fixed_thin.choose_venue(q_large) != 'clob') / N
    random.seed(42)
    liq_amm_share = sum(1 for _ in range(N) if liq_thin.choose_venue(q_large) != 'clob') / N
    check_bool("Liquidity-aware routing overrides the fixed-share split when CLOB is infeasible",
               fixed_amm_share < 0.15 and liq_amm_share > 0.95,
               f"fixed_amm_share={fixed_amm_share:.3f}, liquidity_aware_amm_share={liq_amm_share:.3f}")

    # ── estimate_costs() ─────────────────────────────────────────────
    # Must return a dict with keys clob/cpmm/hfmm, all float.
    subsection("estimate_costs()")
    costs_e = _make_trader().estimate_costs(5.0, 'buy')
    print(f"    costs = {costs_e}")
    check_bool("Keys: clob, cpmm, hfmm",
               set(costs_e.keys()) == {'clob', 'cpmm', 'hfmm'})

def test_mm_amm_interaction():
    """AMM interaction channel can tighten or worsen CLOB MM quotes."""
    section("MARKET MAKER / AMM INTERACTION — Unit Tests")

    class MockEnv:
        def __init__(self, sigma=0.01, c=0.002, sigma_low=0.01, c_low=0.002):
            self.sigma = sigma
            self.funding_cost = c
            self.sigma_low = sigma_low
            self.c_low = c_low
            self.fair_price = 100.0
            self.systemic_liquidity = 1.0
            self.order_flow_imbalance = 0.0

    ex = _make_exchange(BIDS, ASKS)
    env = MockEnv()
    pools = {
        'cpmm': CPMMPool(x=1000.0, y=100_000.0, fee=0.003),
        'hfmm': HFMMPool(x=1000.0, y=100_000.0, A=10.0, fee=0.001, rate=100.0),
    }
    mid = 100.0

    mm_none = MarketMaker(ex, cash=1e4, env=env, amm_pools=pools,
                          venue_interaction_mode='none')
    mm_comp = MarketMaker(ex, cash=1e4, env=env, amm_pools=pools,
                          venue_interaction_mode='competition',
                          amm_spread_impact_bps=3.0, amm_depth_impact=20.0)
    mm_tox = MarketMaker(ex, cash=1e4, env=env, amm_pools=pools,
                         venue_interaction_mode='toxicity',
                         amm_spread_impact_bps=3.0, amm_depth_impact=20.0)

    spread_none = mm_none._target_spread_bps(mid)
    spread_comp = mm_comp._target_spread_bps(mid)
    spread_tox = mm_tox._target_spread_bps(mid)
    depth_none = mm_none._target_depth(mid)
    depth_comp = mm_comp._target_depth(mid)
    depth_tox = mm_tox._target_depth(mid)

    check_bool("Competition mode tightens CLOB MM spread",
               spread_comp < spread_none,
               f"competition={spread_comp:.3f}, none={spread_none:.3f}")
    check_bool("Competition mode boosts CLOB MM depth",
               depth_comp > depth_none,
               f"competition={depth_comp:.3f}, none={depth_none:.3f}")
    check_bool("Toxicity mode widens CLOB MM spread",
               spread_tox > spread_none,
               f"toxicity={spread_tox:.3f}, none={spread_none:.3f}")
    check_bool("Toxicity mode reduces CLOB MM depth",
               depth_tox < depth_none,
               f"toxicity={depth_tox:.3f}, none={depth_none:.3f}")

    support_calm = mm_comp._amm_support_score(mid)
    env.sigma = 0.05
    support_stress = mm_comp._amm_support_score(mid)
    check_bool("AMM support fades in stress",
               support_stress < support_calm,
               f"calm={support_calm:.3f}, stress={support_stress:.3f}")


def test_endogenous_mm_withdrawal():
    """MarketMaker can retreat and re-enter based on endogenous stress signals."""
    section("ENDOGENOUS MM WITHDRAWAL — Unit Tests")

    class MockEnv:
        def __init__(self, sigma=0.01, c=0.002, sigma_low=0.01, c_low=0.002):
            self.sigma = sigma
            self.funding_cost = c
            self.sigma_low = sigma_low
            self.c_low = c_low
            self.fair_price = 100.0
            self.systemic_liquidity = 1.0
            self.order_flow_imbalance = 0.0

    env = MockEnv()
    ex = _make_exchange(BIDS, ASKS)
    mm = MarketMaker(
        ex,
        cash=1e4,
        env=env,
        venue_interaction_mode='none',
        withdrawal_threshold=1.0,
        reentry_threshold=0.35,
        loss_threshold_bps=10.0,
        min_withdraw_ticks=2,
        reentry_ticks=2,
    )

    mm._update_endogenous_state(100.0)
    check_bool("Calm MM starts active",
               mm.mm_state == 'active',
               f"state={mm.mm_state}, score={mm.mm_withdrawal_score:.3f}")

    env.sigma = 0.05
    env.funding_cost = 0.02
    env.systemic_liquidity = 0.45
    env.order_flow_imbalance = 0.9
    mm.inventory = 90.0
    mm._ofi_window = [0.9] * 5
    mm._loss_bps_ewma = 30.0
    mm._update_endogenous_state(100.0)
    check_bool("Large loss and stress trigger withdrawal",
               mm.mm_state == 'withdrawn',
               f"state={mm.mm_state}, score={mm.mm_withdrawal_score:.3f}")

    mm.call()
    check_bool("Withdrawn MM keeps no resting quotes",
               len(mm.orders) == 0,
               f"orders={len(mm.orders)}")

    env.sigma = 0.01
    env.funding_cost = 0.002
    env.systemic_liquidity = 1.0
    env.order_flow_imbalance = 0.0
    mm.inventory = 0.0
    mm._ofi_window = [0.0]
    mm._loss_bps_ewma = 0.0

    mm._update_endogenous_state(100.0)
    mm._update_endogenous_state(100.0)
    check_bool("Recovered conditions move MM into reentering",
               mm.mm_state == 'reentering',
               f"state={mm.mm_state}, timer={mm.mm_state_timer}")

    mm._update_endogenous_state(100.0)
    mm._update_endogenous_state(100.0)
    check_bool("Reentry completes back to active",
               mm.mm_state == 'active',
               f"state={mm.mm_state}, timer={mm.mm_state_timer}")

    mm.mm_state = 'defensive'
    mm.mm_withdrawal_score = 0.9
    defensive_adj = mm._state_quote_adjustments()
    check_bool("Defensive MM widens spreads and thins depth",
               defensive_adj['spread_mult'] > 1.0 and defensive_adj['depth_mult'] < 1.0,
               f"spread_mult={defensive_adj['spread_mult']:.3f}, depth_mult={defensive_adj['depth_mult']:.3f}")


# =====================================================================
#  5. FEE ACCOUNTING & FUNDAMENTALIST ANCHOR
# =====================================================================

def test_fee_accounting():
    """Sell-side fee revenue uses actual exec_price, not mid_price."""
    section("FEE ACCOUNTING — Unit Tests")

    # ── CPMM sell-side fee accuracy ──────────────────────────────────
    subsection("CPMM sell-side fee revenue")
    p = CPMMPool(x=1000.0, y=100_000.0, fee=0.01)
    for Q in [1, 10, 50, 200]:
        p2 = CPMMPool(x=1000.0, y=100_000.0, fee=0.01)
        quote = p2.execute_sell(Q)
        expected_fee = Q * 0.01 * quote['exec_price']
        check(f"CPMM sell Q={Q}: fee_revenue ≈ Q·fee·exec_price",
              p2.fee_revenue, expected_fee, rel=1e-9)

    # ── HFMM sell-side fee accuracy ──────────────────────────────────
    subsection("HFMM sell-side fee revenue")
    for Q in [1, 10, 50]:
        h = HFMMPool(x=1000.0, y=100_000.0, A=10.0, fee=0.01, rate=100.0)
        quote = h.execute_sell(Q)
        expected_fee = Q * 0.01 * quote['exec_price']
        check(f"HFMM sell Q={Q}: fee_revenue ≈ Q·fee·exec_price",
              h.fee_revenue, expected_fee, rel=1e-9)

    # ── Cumulative fee test (buy + sell) ─────────────────────────────
    subsection("Cumulative fee (buy→sell→buy sequence)")
    p3 = CPMMPool(x=1000.0, y=100_000.0, fee=0.003)
    cum = 0.0
    for _ in range(10):
        q_buy = p3.execute_buy(5)
        cum += q_buy['delta_y'] * 0.003
        q_sell = p3.execute_sell(5)
        cum += 5 * 0.003 * q_sell['exec_price']
    check(f"Cumulative fee matches fee_revenue after 20 trades",
          p3.fee_revenue, cum, rel=1e-6)


def test_fundamentalist_anchor():
    """FX Fundamentalist uses env.fair_price, not static fundamental_rate."""
    section("FUNDAMENTALIST DYNAMIC ANCHOR — Unit Tests")

    # Minimal env mock
    class MockEnv:
        def __init__(self, fp):
            self.fair_price = fp
            self.sigma = 0.02
            self.funding_cost = 0.005
        def is_stress(self):
            return False

    # Minimal CLOB mock that returns a fixed mid
    class MockCLOB:
        def __init__(self, mid):
            self._mid = mid
        def mid_price(self):
            return self._mid
        def quote_buy(self, Q):
            return dict(exec_price=self._mid, cost_bps=5.0,
                        half_spread_bps=5.0, espr_bps=10.0,
                        impact_bps=0.0, fee_bps=0.0)
        def quote_sell(self, Q):
            return self.quote_buy(Q)

    # Build a Fundamentalist with fundamental_rate=100 but env has fair_price=80
    ex = ExchangeAgent(price=100.0, std=2.0, volume=100)
    env = MockEnv(fp=80.0)
    clob = MockCLOB(mid=80.0)

    f = Fundamentalist(ex, cash=1e4, access=1,
                       clob=clob, amm_pools={}, env=env,
                       amm_share_pct=0,
                       fundamental_rate=100.0, fx_gamma=5e-3, fx_q_max=10)

    # With env.fair_price=80 and CLOB mid=80 → mispricing ≈ 0 → no trade
    subsection("env.fair_price = CLOB mid → no trade")
    result = f.call()
    check_bool("No trade when fair_price ≈ mid",
               result is None,
               f"result = {result}")

    # Now move env.fair_price to 90 → mispricing = (90-80)/80 = 12.5% → buy
    subsection("env.fair_price > CLOB mid → buy signal")
    env.fair_price = 90.0
    result2 = f.call()
    check_bool("Triggers buy when env.fair_price > mid",
               result2 is not None and result2.get('side') == 'buy',
               f"result = {result2}")

    # With static anchor (100) and mid=80 you'd always get a buy — but
    # with env.fair_price=75 and mid=80 you should get a sell
    subsection("env.fair_price < CLOB mid → sell signal")
    env.fair_price = 75.0
    result3 = f.call()
    check_bool("Triggers sell when env.fair_price < mid",
               result3 is not None and result3.get('side') == 'sell',
               f"result = {result3}")

def test_environment_price_process():
    """Shock tick should log the shocked level before stochastic recovery begins."""
    section("ENVIRONMENT PRICE PROCESS — Unit Tests")

    random.seed(42)
    env = MarketEnvironment(price=100.0,
                            sigma_low=0.01, sigma_high=0.05,
                            c_low=0.001, c_high=0.01)

    env.apply_shock(-20.0)
    check("Shock applies immediate -20% level shift",
          env.fair_price, 80.0, abs_tol=1e-9)

    env.step()
    check("First post-shock step preserves shocked level",
          env.fair_price, 80.0, abs_tol=1e-9)

    path = []
    for _ in range(40):
        env.step()
        path.append(env.fair_price)

    check_bool("Post-shock path stays in a sane range over 40 steps",
               min(path) > 60.0 and max(path) < 110.0,
               f"min={min(path):.3f}, max={max(path):.3f}")


def test_latent_anchor_reference():
    """AMM arb reference should treat env.fair_price as a latent anchor."""
    section("LATENT FAIR-PRICE ANCHOR — Unit Tests")

    class MockEnv:
        def __init__(self):
            self.fair_price = 110.0
            self.sigma = 0.05
            self.sigma_low = 0.01
            self.funding_cost = 0.01
            self.systemic_liquidity = 0.6

    class MockCLOB:
        def mid_price(self):
            return 100.0
        def quoted_spread_bps(self):
            return 20.0

    env = MockEnv()
    clob = MockCLOB()
    pools = {
        'cpmm': CPMMPool(x=1000.0, y=99_000.0, fee=0.003),
        'hfmm': HFMMPool(x=1000.0, y=100_000.0, A=10.0, fee=0.001, rate=100.0),
    }
    arb = AMMArbitrageur(clob, pools, env=env)
    ref = arb._reference_price()

    check_bool("Reference stays between observed market and fair anchor",
               100.0 < ref < 110.0,
               f"ref={ref:.4f}")
    check_bool("Reference leans toward observed market, not fair anchor",
               abs(ref - 100.0) < abs(ref - 110.0),
               f"ref={ref:.4f}")


def test_systemic_liquidity_factor():
    """Shared liquidity factor should fall on shock / flow stress and later recover."""
    section("SYSTEMIC LIQUIDITY FACTOR — Unit Tests")

    env = MarketEnvironment(price=100.0,
                            sigma_low=0.01, sigma_high=0.05,
                            c_low=0.001, c_high=0.01)
    base_liq = env.systemic_liquidity

    env.apply_shock(-20.0)
    shock_liq = env.systemic_liquidity
    check_bool("Shock lowers common liquidity factor",
               shock_liq < base_liq,
               f"base={base_liq:.3f}, shock={shock_liq:.3f}")

    env.observe_order_flow([
        {'side': 'sell', 'quantity': 20},
        {'side': 'sell', 'quantity': 10},
    ])
    flow_liq = env.systemic_liquidity
    check_bool("Sell wave creates negative order-flow imbalance",
               env.order_flow_imbalance < 0,
               f"imbalance={env.order_flow_imbalance:.3f}")
    check_bool("Order-flow pressure does not improve liquidity factor",
               flow_liq <= shock_liq,
               f"shock={shock_liq:.3f}, flow={flow_liq:.3f}")

    for _ in range(40):
        env.step()

    check_bool("Common liquidity factor partially recovers over time",
               env.systemic_liquidity > flow_liq,
               f"flow={flow_liq:.3f}, recovered={env.systemic_liquidity:.3f}")


def test_shock_stress_cli_semantics():
    """Pure shocks should no longer auto-enable a separate regime-stress layer."""
    section("SHOCK / REGIME-STRESS CLI SEMANTICS — Unit Tests")

    args = argparse.Namespace(
        shock_iter=250,
        shock_pct=-20.0,
        stress_start=-1,
        stress_end=-1,
        n_iter=1000,
        sigma_low=0.01,
        sigma_high=0.05,
        c_low=0.002,
        c_high=0.02,
        shock_regime_stress=False,
        no_shock_stress=False,
    )
    main_module._auto_stress_around_shock(args)
    check_bool("Pure shock leaves regime stress disabled",
               args.stress_start == -1 and args.stress_end == -1,
               f"stress_start={args.stress_start}, stress_end={args.stress_end}")

    sim = argparse.Namespace(shock_iter=args.shock_iter, env=argparse.Namespace(stress_start=-1))
    split_iter, split_title, phase_before, phase_after = main_module._linkage_split_point(sim)
    check_bool("Pure shock H2 split falls back to shock tick",
               split_iter == args.shock_iter and split_title == 'Shock',
               f"split_iter={split_iter}, split_title={split_title}")
    check_bool("Pure shock labels are pre/post shock",
               phase_before == 'Pre-shock' and phase_after == 'Post-shock',
               f"phase_before={phase_before}, phase_after={phase_after}")

    args2 = argparse.Namespace(**vars(args))
    args2.shock_regime_stress = True
    main_module._auto_stress_around_shock(args2)
    check_bool("Explicit shock_regime_stress starts stress at shock tick",
               args2.stress_start == args2.shock_iter and args2.stress_end > args2.shock_iter,
               f"stress_start={args2.stress_start}, stress_end={args2.stress_end}")

    sim2 = argparse.Namespace(shock_iter=args2.shock_iter,
                              env=argparse.Namespace(stress_start=args2.stress_start))
    split_iter2, split_title2, _, phase_after2 = main_module._linkage_split_point(sim2)
    check_bool("Explicit regime stress remains preferred split point",
               split_iter2 == args2.stress_start and split_title2 == 'Stress',
               f"split_iter={split_iter2}, split_title={split_title2}")
    check_bool("Explicit regime stress keeps stress label",
               phase_after2 == 'Stress',
               f"phase_after={phase_after2}")


def test_flash_crash_preset():
    """Flash-crash preset should be wired as a pure microstructure realism shock."""
    section("FLASH CRASH PRESET — Unit Tests")

    parser = main_module.build_parser()
    args = parser.parse_args(['--preset', 'flash_crash'])
    main_module._apply_preset_defaults(parser, args)

    check_bool("Flash crash is a realism preset",
               main_module._preset_family(args.preset) == 'realism',
               f"preset_family={main_module._preset_family(args.preset)}")
    check_bool("Flash crash does not impose a fundamental repricing",
               args.fundamental_shock_pct == 0.0,
               f"fundamental_shock_pct={args.fundamental_shock_pct}")
    check_bool("Flash crash uses an explicit sell-side sweep",
               args.order_flow_shock_side == 'sell' and args.order_flow_shock_qty > 0,
               f"side={args.order_flow_shock_side}, qty={args.order_flow_shock_qty}")
    check_bool("Flash crash includes an aggressive liquidity withdrawal",
               args.liquidity_shock_frac >= 0.85,
               f"liquidity_shock_frac={args.liquidity_shock_frac}")
    check_bool("Flash crash keeps recovery faster than a dealer liquidity crisis",
               args.reprice_prob_recovery > main_module.REALISM_PRESETS['dealer_liquidity_crisis']['reprice_prob_recovery'],
               f"flash={args.reprice_prob_recovery}, dealer={main_module.REALISM_PRESETS['dealer_liquidity_crisis']['reprice_prob_recovery']}")


def test_mm_withdrawal_preset():
    """MM withdrawal should stay persistent but recover faster via secondary liquidity support."""
    section("MM WITHDRAWAL PRESET — Unit Tests")

    parser = main_module.build_parser()
    args = parser.parse_args(['--preset', 'mm_withdrawal'])
    main_module._apply_preset_defaults(parser, args)

    check_bool("MM withdrawal remains a realism preset",
               main_module._preset_family(args.preset) == 'realism',
               f"preset_family={main_module._preset_family(args.preset)}")
    check_bool("MM withdrawal keeps no permanent fair-value shift",
               args.fundamental_shock_pct == 0.0,
               f"fundamental_shock_pct={args.fundamental_shock_pct}")
    check_bool("MM withdrawal stays harsher than funding-liquidity shock on quote loss",
               args.liquidity_shock_frac > main_module.REALISM_PRESETS['funding_liquidity_shock']['liquidity_shock_frac'],
               f"mm={args.liquidity_shock_frac}, funding={main_module.REALISM_PRESETS['funding_liquidity_shock']['liquidity_shock_frac']}")
    check_bool("MM withdrawal fades liquidity stress faster than dealer-liquidity crisis",
               args.liquidity_shock_decay < main_module.REALISM_PRESETS['dealer_liquidity_crisis']['liquidity_shock_decay'],
               f"mm={args.liquidity_shock_decay}, dealer={main_module.REALISM_PRESETS['dealer_liquidity_crisis']['liquidity_shock_decay']}")
    check_bool("MM withdrawal keeps the default CLOB build for apples-to-apples stats",
               args.n_fast_lp == parser.get_default('n_fast_lp')
               and args.n_latent_lp == parser.get_default('n_latent_lp')
               and args.clob_near_mid_target_ratio == parser.get_default('clob_near_mid_target_ratio'),
               f"n_fast_lp={args.n_fast_lp}, n_latent_lp={args.n_latent_lp}, near_mid={args.clob_near_mid_target_ratio}")
    check_bool("MM withdrawal accelerates near-mid background replenishment",
               args.bg_target_ratio_recovery > main_module.REALISM_PRESETS['funding_liquidity_shock']['bg_target_ratio_recovery'],
               f"mm={args.bg_target_ratio_recovery}, funding={main_module.REALISM_PRESETS['funding_liquidity_shock']['bg_target_ratio_recovery']}")


def test_funding_liquidity_shock_preset():
    """Funding-liquidity shock should remain a temporary stress event, not a legacy repricing panel."""
    section("FUNDING LIQUIDITY SHOCK PRESET — Unit Tests")

    parser = main_module.build_parser()
    args = parser.parse_args(['--preset', 'funding_liquidity_shock'])
    main_module._apply_preset_defaults(parser, args)

    check_bool("Funding liquidity shock is a realism preset",
               main_module._preset_family(args.preset) == 'realism',
               f"preset_family={main_module._preset_family(args.preset)}")
    check_bool("Funding liquidity shock does not impose a permanent fair-value shift",
               args.fundamental_shock_pct == 0.0,
               f"fundamental_shock_pct={args.fundamental_shock_pct}")
    check_bool("Funding liquidity shock includes a strong funding/vol spike",
               args.funding_vol_shock_intensity >= 0.9,
               f"funding_vol_shock_intensity={args.funding_vol_shock_intensity}")
    check_bool("Funding liquidity shock is milder than flash crash on quote withdrawal",
               args.liquidity_shock_frac < main_module.REALISM_PRESETS['flash_crash']['liquidity_shock_frac'],
               f"funding={args.liquidity_shock_frac}, flash={main_module.REALISM_PRESETS['flash_crash']['liquidity_shock_frac']}")


def test_initial_depth_matching():
    """Initial live CLOB depth should be matched to aggregate AMM depth."""
    section("INITIAL DEPTH MATCHING — Unit Tests")

    from AgentBasedModel.simulator.simulator import Simulator

    random.seed(42)
    sim = Simulator.default_fx(match_initial_depth=True)
    clob_depth = sim.clob.total_depth(25)['total']
    mid = sim.clob.mid_price()
    amm_depth = sum(pool.effective_depth(mid) for pool in sim.amm_pools.values())
    ratio = clob_depth / amm_depth if amm_depth > 0 else float('inf')
    check_bool("Initial CLOB near-mid depth matches aggregate AMM depth",
               0.90 <= ratio <= 1.10,
               f"clob={clob_depth:.1f}, amm={amm_depth:.1f}, ratio={ratio:.3f}")

    random.seed(42)
    sim_no_match = Simulator.default_fx(match_initial_depth=False)
    clob_no_match = sim_no_match.clob.total_depth(25)['total']
    mid_no_match = sim_no_match.clob.mid_price()
    amm_no_match = sum(pool.effective_depth(mid_no_match) for pool in sim_no_match.amm_pools.values())
    ratio_no_match = clob_no_match / amm_no_match if amm_no_match > 0 else float('inf')
    check_bool("Opt-out keeps the unmatched initial calibration",
               abs(ratio_no_match - ratio) > 0.05,
               f"matched={ratio:.3f}, unmatched={ratio_no_match:.3f}")


def test_amm_core_liquidity_floor():
    """AMM LP rule should preserve a core liquidity floor under stress."""
    section("AMM CORE LIQUIDITY FLOOR — Unit Tests")

    env = MarketEnvironment(price=100.0,
                            sigma_low=0.01, sigma_high=0.05,
                            c_low=0.001, c_high=0.01)
    pool = HFMMPool(x=1000.0, y=100000.0, A=10.0, fee=0.001, rate=100.0)
    provider = AMMProvider(pool, env,
                           core_liquidity_ratio=0.75,
                           stabilizing_bias=0.03)

    initial_L = pool.liquidity_measure()
    pool.remove_liquidity(0.40)
    stressed_L = pool.liquidity_measure()
    env.apply_shock(-20.0)
    provider.update_liquidity()
    updated_L = pool.liquidity_measure()

    check_bool("Core floor triggers refill when liquidity falls too far",
               updated_L > stressed_L,
               f"stressed={stressed_L:.1f}, updated={updated_L:.1f}")
    check_bool("Core floor preserves a large stable AMM base",
               updated_L >= 0.60 * initial_L,
               f"initial={initial_L:.1f}, updated={updated_L:.1f}")


def test_realism_shock_mode():
    """Realism mode should separate fair-value jumps from immediate AMM repricing."""
    section("REALISM SHOCK MODE — Unit Tests")

    from AgentBasedModel.simulator.simulator import Simulator

    random.seed(42)
    sim = Simulator.default_fx(
        n_noise=8, n_mm=1,
        n_fx_takers=0, n_fx_fund=0,
        n_retail=0, n_institutional=0,
        n_clob_fund=1, n_clob_chart=0,
        n_clob_univ=0, clob_volume=200,
        shock_mode='realism',
        fundamental_shock_pct=-15.0,
        order_flow_shock_qty=0.0,
        liquidity_shock_frac=0.40,
        funding_vol_shock_intensity=1.0,
    )
    cpmm_mid_before = sim.amm_pools['cpmm'].mid_price()
    fair_before = sim.env.fair_price

    sim._apply_realism_shock()

    check_bool("Fair price jumps in realism mode",
               sim.env.fair_price < fair_before,
               f"before={fair_before:.2f}, after={sim.env.fair_price:.2f}")
    check_bool("AMM price is not instantly rebalanced in realism mode",
               abs(sim.amm_pools['cpmm'].mid_price() - cpmm_mid_before) < 1e-9,
               f"before={cpmm_mid_before:.4f}, after={sim.amm_pools['cpmm'].mid_price():.4f}")
    check_bool("Liquidity shock pauses market makers",
               sim.env.mm_pause_ticks > 0,
               f"mm_pause_ticks={sim.env.mm_pause_ticks}")

    random.seed(42)
    sim2 = Simulator.default_fx(
        n_noise=8, n_mm=1,
        n_fx_takers=0, n_fx_fund=0,
        n_retail=0, n_institutional=0,
        n_clob_fund=1, n_clob_chart=0,
        n_clob_univ=0, clob_volume=200,
        shock_mode='realism',
        fundamental_shock_pct=0.0,
        order_flow_shock_qty=80.0,
        order_flow_shock_side='sell',
        liquidity_shock_frac=0.20,
        funding_vol_shock_intensity=0.0,
    )
    pre_mid = sim2.clob.mid_price()
    sim2._apply_realism_shock()
    post_mid = sim2.clob.mid_price()
    check_bool("Order-flow shock sweeps the live CLOB",
               post_mid < pre_mid,
               f"pre_mid={pre_mid:.2f}, post_mid={post_mid:.2f}")


def test_shock_event_window_helpers():
    """Event-window helpers should expose local post-shock diagnostics."""
    section("SHOCK EVENT WINDOWS — Unit Tests")

    windows = main_module._shock_window_slices(500, 350)
    check_bool("Shock windows contain four local segments",
               len(windows) == 4,
               f"windows={windows}")
    check_bool("First post-shock window starts at t0",
               windows[1] == ('t0..t0+5', 350, 355),
               f"window={windows[1]}")

    series = [10.0] * 50 + [25.0, 22.0, 18.0, 14.0, 12.0, 11.0, 10.5, 10.2, 10.1, 10.0]
    baseline = 10.0
    rt, target = main_module._rolling_normalization_time(
        series,
        shock_iter=50,
        baseline=baseline,
        direction='upper',
        rel_tol=0.1,
        abs_tol=0.5,
        window=3,
        horizon=20,
    )
    check_bool("Rolling normalization finds a finite recovery time",
               math.isfinite(rt) and rt >= 0,
               f"rt={rt}, target={target}")

    trough, t_trough = main_module._series_trough(
        [0.0] * 50 + [-10.0, -30.0, -25.0, -5.0],
        shock_iter=50,
        direction='lower',
        horizon=10,
    )
    check_bool("Series trough captures max local impact",
               trough == -30.0 and t_trough == 1.0,
               f"trough={trough}, t_trough={t_trough}")


def test_resilience_metrics_censoring():
    """Relative-recovery metrics should distinguish recovered and censored trajectories."""
    section("RESILIENCE METRICS — Unit Tests")

    recovered_series = (
        [100.0] * 50
        + [80.0, 82.0, 90.0, 95.0, 96.0, 96.5, 96.2, 96.1]
    )
    recovered = price_resilience_metrics(
        recovered_series,
        shock_iter=50,
        avg_window=4,
        stable_window=3,
        retracement_fraction=0.8,
        horizon=8,
    )
    check_bool("80% retracement detects recovery on sustained normalization",
               recovered['recovered'] and not recovered['is_censored'],
               f"recovered={recovered['recovered']}, censored={recovered['is_censored']}")
    check("Recovered trajectory reaches target in 5 steps",
          recovered['recovery_steps'], 5.0)
    check_bool("Normalized impact is finite for recovered path",
               math.isfinite(recovered['normalized_avg_impact']),
               f"normalized_avg_impact={recovered['normalized_avg_impact']}")

    censored_series = [100.0] * 50 + [80.0] * 20
    censored = price_resilience_metrics(
        censored_series,
        shock_iter=50,
        avg_window=5,
        stable_window=3,
        retracement_fraction=0.8,
        horizon=10,
    )
    check_bool("Flat distressed path stays right-censored",
               (not censored['recovered']) and censored['is_censored'],
               f"recovered={censored['recovered']}, censored={censored['is_censored']}")
    check("Censored path reports observed horizon instead of fake recovery",
          censored['time_observed_steps'], 10.0)

    km = kaplan_meier_curve([4.0, 10.0, 6.0], [True, False, True])
    check_bool("Kaplan-Meier returns a finite median when survival crosses 50%",
               math.isfinite(km['median_time']) and km['median_time'] == 6.0,
               f"median_time={km['median_time']}, survival={km['survival']}")

    repricing_series = [100.0] * 50 + [90.0, 91.0, 93.0, 95.0, 96.0, 97.0, 97.0, 97.0]
    fair_value_series = [100.0] * 50 + [97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0]
    repricing = price_resilience_metrics(
        repricing_series,
        shock_iter=50,
        avg_window=4,
        stable_window=3,
        retracement_fraction=0.8,
        horizon=8,
        target_series=fair_value_series,
        target_mode='fair_value_gap',
    )
    check_bool("Fair-value-gap mode recovers to the new equilibrium instead of the old price",
               repricing['recovered'] and repricing['analysis_target_mode'] == 'fair_value_gap',
               f"recovered={repricing['recovered']}, target_mode={repricing['analysis_target_mode']}")
    check("Fair-value-gap recovery reaches target in 4 steps",
          repricing['recovery_steps'], 4.0)


def test_priority_resilience_metrics():
    """Generic spread/depth/cost metrics and their composite should recover consistently."""
    section("PRIORITY RESILIENCE METRICS — Unit Tests")

    price_series = [100.0] * 50 + [80.0, 82.0, 90.0, 95.0, 96.0, 96.5, 96.2, 96.1]
    spread_series = [2.0] * 50 + [8.0, 6.0, 4.0, 2.6, 2.4, 2.3, 2.2, 2.1]
    depth_series = [100.0] * 50 + [60.0, 70.0, 82.0, 91.0, 95.0, 96.0, 97.0, 98.0]
    cost_series = [5.0] * 50 + [15.0, 12.0, 8.0, 6.0, 5.6, 5.4, 5.3, 5.2]

    price_metrics = price_resilience_metrics(
        price_series,
        shock_iter=50,
        avg_window=4,
        stable_window=3,
        retracement_fraction=0.8,
        horizon=8,
    )
    spread_metrics = series_resilience_metrics(
        spread_series,
        shock_iter=50,
        avg_window=4,
        stable_window=3,
        retracement_fraction=0.8,
        horizon=8,
    )
    depth_metrics = series_resilience_metrics(
        depth_series,
        shock_iter=50,
        avg_window=4,
        stable_window=3,
        retracement_fraction=0.8,
        horizon=8,
    )
    cost_metrics = series_resilience_metrics(
        cost_series,
        shock_iter=50,
        avg_window=4,
        stable_window=3,
        retracement_fraction=0.8,
        horizon=8,
    )
    composite = composite_resilience_metrics(
        [price_metrics, spread_metrics, depth_metrics, cost_metrics],
        analysis_target_mode='joint_price_spread_depth_cost',
    )

    check_bool("Spread resilience recovers after a sustained spread compression",
               spread_metrics['recovered'],
               f"recovery_steps={spread_metrics['recovery_steps']}")
    check("Spread resilience reaches target in 3 steps",
          spread_metrics['recovery_steps'], 3.0)
    check_bool("Depth resilience recovers after depth rebuild",
               depth_metrics['recovered'],
               f"recovery_steps={depth_metrics['recovery_steps']}")
    check("Execution-cost resilience reaches target in 3 steps",
          cost_metrics['recovery_steps'], 3.0)
    check_bool("Composite resilience requires all component recoveries",
               composite['recovered'] and composite['analysis_target_mode'] == 'joint_price_spread_depth_cost',
               f"recovered={composite['recovered']}, mode={composite['analysis_target_mode']}")
    check("Composite recovery waits for the slowest component",
          composite['recovery_steps'], 5.0)


def test_resilience_statistical_metrics():
    """Bootstrap CI and permutation tests should expose stable effect estimates."""
    section("RESILIENCE STATISTICS — Unit Tests")

    sample = [1.0, 2.0, 3.0, 4.0]
    boot_mean = bootstrap_mean_ci(sample, n_boot=1000, ci_level=0.95, seed=7)
    check("Bootstrap mean matches arithmetic mean",
       boot_mean['mean'], 2.5)
    check_bool("Bootstrap CI contains the sample mean",
         boot_mean['ci_lo'] <= boot_mean['mean'] <= boot_mean['ci_hi'],
         f"ci=({boot_mean['ci_lo']:.3f}, {boot_mean['ci_hi']:.3f}), mean={boot_mean['mean']:.3f}")

    sample_a = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    sample_b = [2.0, 3.0, 2.0, 3.0, 2.0, 3.0]
    permutation = independent_permutation_test(sample_a, sample_b, n_perm=2000, seed=11)
    boot_diff = bootstrap_diff_ci(sample_a, sample_b, n_boot=1000, ci_level=0.95, seed=11)

    check_bool("Independent permutation test detects lower mean in sample_a",
         math.isfinite(permutation['p_value']) and permutation['p_value'] < 0.05 and permutation['mean_diff'] < 0,
         f"p={permutation['p_value']}, diff={permutation['mean_diff']}")
    check_bool("Bootstrap diff CI stays below zero when sample_a < sample_b",
         boot_diff['ci_hi'] < 0.0,
         f"ci=({boot_diff['ci_lo']:.3f}, {boot_diff['ci_hi']:.3f})")

    paired_a = [10.0, 12.0, 11.0, 13.0, 14.0, 15.0]
    paired_b = [9.0, 10.0, 10.0, 11.0, 12.0, 13.0]
    paired_test = paired_permutation_test(paired_a, paired_b, n_perm=2000, seed=13)
    paired_boot = bootstrap_paired_diff_ci(paired_a, paired_b, n_boot=1000, ci_level=0.95, seed=13)

    check_bool("Paired permutation test detects positive within-seed delta",
         math.isfinite(paired_test['p_value']) and paired_test['p_value'] < 0.05 and paired_test['mean_diff'] > 0.0,
         f"p={paired_test['p_value']}, diff={paired_test['mean_diff']}")
    check_bool("Paired bootstrap CI stays above zero for positive deltas",
         paired_boot['ci_lo'] > 0.0,
         f"ci=({paired_boot['ci_lo']:.3f}, {paired_boot['ci_hi']:.3f})")


def test_arbitrage_trade_cap():
    """Arbitrageur should respect per-step reserve trade caps."""
    section("ARBITRAGE TRADE CAP — Unit Tests")

    class MockCLOB:
        def mid_price(self):
            return 100.0
        def quoted_spread_bps(self):
            return 10.0

    pool = CPMMPool(x=1000.0, y=100000.0, fee=0.003)
    arb = AMMArbitrageur(
        MockCLOB(),
        {'cpmm': pool},
        max_correction_pct=50.0,
        trade_fraction_cap=0.02,
    )
    before_x = pool.x
    arb._arb_one_pool(pool, 120.0)
    traded_x = pool.x - before_x

    check_bool("Arbitrage trade is capped by configured reserve fraction",
               traded_x <= before_x * 0.02 + 1e-6,
               f"before_x={before_x:.2f}, after_x={pool.x:.2f}, traded_x={traded_x:.2f}")


# =====================================================================
#  DIAGNOSTIC PLOTS  (activated with --plots)
# =====================================================================

def _init_plots():
    """Lazy-import heavy deps only when --plots is used."""
    global np, pd, plt, OUT_DIR
    import numpy as _np
    import pandas as _pd
    import matplotlib as _mpl
    _mpl.use('Agg')
    import matplotlib.pyplot as _plt
    np, pd, plt = _np, _pd, _plt
    OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'unit_tests')
    os.makedirs(OUT_DIR, exist_ok=True)

ACCENT  = '#2196F3'
ACCENT2 = '#FF9800'
ACCENT3 = '#4CAF50'
ACCENT4 = '#E91E63'
GRAY    = '#9E9E9E'


def _savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  saved  {path}")


def _save_csv(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"  csv    {path}")


# ── CLOB plots ───────────────────────────────────────────────────────

def plot_clob():
    print("\n[CLOB] collecting data & generating plots")

    mid_manual = (99.0 + 101.0) / 2.0
    qspr_manual = 10_000 * (101.0 - 99.0) / mid_manual

    ex = _make_exchange(BIDS, ASKS)
    clob = CLOBVenue(ex, f_broker=0.0, f_venue=0.0)

    scenarios_11 = []
    scenarios_11.append(dict(
        scenario_id='mid+qspr',
        mid_code=clob.mid_price(), mid_manual=mid_manual,
        qspr_code=clob.quoted_spread_bps(), qspr_manual=qspr_manual,
        espr_code=0.0, espr_manual=0.0,
    ))

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
                mid_code=clob.mid_price(), mid_manual=mid_manual,
                qspr_code=clob.quoted_spread_bps(), qspr_manual=qspr_manual,
                espr_code=q['espr_bps'], espr_manual=espr_manual,
            ))

    df11 = pd.DataFrame(scenarios_11)
    df11['mid_err'] = df11['mid_code'] - df11['mid_manual']
    df11['qspr_err'] = df11['qspr_code'] - df11['qspr_manual']
    df11['espr_err'] = df11['espr_code'] - df11['espr_manual']
    _save_csv(df11, 'clob_mid_spread.csv')

    clob_fee = CLOBVenue(_make_exchange(BIDS, ASKS), f_broker=2.0, f_venue=1.0)
    rows_12 = []
    for Q in list(range(1, 61)):
        q0 = clob.quote_buy(Q)
        qf = clob_fee.quote_buy(Q)
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
            cost_manual_f = hs_manual + 3.0
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('1. CLOB  -  Diagnostic Plots', fontsize=14, weight='bold')

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

    ax = axes[1, 0]
    valid = df12.dropna(subset=['impact_code', 'impact_manual'])
    ax.plot(valid['Q'], valid['impact_code'], '-', color=ACCENT, lw=2, label='impact (code)')
    ax.plot(valid['Q'], valid['impact_manual'], '--', color=ACCENT2, lw=2, label='impact (manual)')
    ax.set_xlabel('Quantity Q')
    ax.set_ylabel('Impact (bps)')
    ax.set_title('1.2a  Price Impact vs Q')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

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


# ── CPMM plots ───────────────────────────────────────────────────────

def plot_cpmm():
    print("\n[CPMM] collecting data & generating plots")

    p = CPMMPool(x=1000.0, y=100_000.0, fee=0.003)
    k0 = p.k
    rows_inv = []
    rows_inv.append(dict(step=0, x=p.x, y=p.y, k_code=p.x * p.y, k0=k0, cycle_id=0))
    for i in range(1, 11):
        p.execute_buy(2.0)
        rows_inv.append(dict(step=i, x=p.x, y=p.y, k_code=p.x * p.y, k0=k0, cycle_id=0))
    for i in range(11, 21):
        p.execute_sell(2.0)
        rows_inv.append(dict(step=i, x=p.x, y=p.y, k_code=p.x * p.y, k0=k0, cycle_id=0))

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
            Q=Q, price_code=qb['exec_price'], price_expected=S,
            slip_code=qb['slippage_bps'], slip_manual=slip_manual,
            cost_code=qb['cost_bps'], cost_manual=cost_manual,
            fee_bps=qb['fee_bps'],
        ))
    df_slip = pd.DataFrame(rows_slip)
    _save_csv(df_slip, 'cpmm_slippage.csv')

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('2. CPMM  -  Diagnostic Plots', fontsize=14, weight='bold')

    ax = axes[0, 0]
    ax.plot(df_inv['step'], df_inv['k_code'], 'o-', color=ACCENT, ms=4, lw=1.5)
    ax.axhline(k0, ls='--', color=ACCENT4, lw=1, label=f'k0 = {k0:.2e}')
    ax.set_xlabel('Trade step')
    ax.set_ylabel('k = x*y')
    ax.set_title('2.1a  Invariant k')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.boxplot(df_cyc['k_delta'].values, vert=True, patch_artist=True,
               boxprops=dict(facecolor=ACCENT, alpha=0.5))
    ax.axhline(0, ls='--', color=GRAY, lw=1)
    ax.set_ylabel('k_after_cycle - k0')
    ax.set_title('2.1b  Round-trip k Deviation (30 cycles)')
    ax.grid(True, alpha=0.3, axis='y')

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

    ax = axes[1, 1]
    ax.plot(df_slip['Q'], df_slip['fee_bps'], '-', color=ACCENT3, lw=2)
    ax.axhline(10_000 * 0.003, ls='--', color=GRAY, lw=1, label='30 bps')
    ax.set_xlabel('Q')
    ax.set_ylabel('Fee (bps)')
    ax.set_title('2.2c  Fee vs Q (sanity: horizontal)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

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


# ── HFMM plots ───────────────────────────────────────────────────────

def plot_hfmm():
    print("\n[HFMM] collecting data & generating plots")

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

    cpmm_ref = CPMMPool(x=1000, y=100_000, fee=0.0)
    hfmm_ref = HFMMPool(x=1000, y=100_000, A=100, fee=0.0, rate=100.0)
    rows_cmp = []
    for Q in np.concatenate([np.linspace(0.5, 10, 20), np.linspace(15, 300, 40)]):
        qc = cpmm_ref.quote_buy(Q, S_t=100.0)
        qh = hfmm_ref.quote_buy(Q, S_t=100.0)
        if qc['cost_bps'] >= float('inf') or qh['cost_bps'] >= float('inf'):
            continue
        rows_cmp.append(dict(
            Q=Q, price_cpmm=qc['exec_price'], price_hfmm=qh['exec_price'],
            slip_cpmm=qc['slippage_bps'], slip_hfmm=qh['slippage_bps'],
        ))
    df_cmp = pd.DataFrame(rows_cmp)
    _save_csv(df_cmp, 'hfmm_vs_cpmm.csv')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('3. HFMM  -  Diagnostic Plots', fontsize=14, weight='bold')

    ax = axes[0, 0]
    ax.plot(df_d['step'], df_d['D_code'], 'o-', color=ACCENT, ms=4, lw=1.5)
    ax.axhline(D0, ls='--', color=ACCENT4, lw=1, label=f'D0 = {D0:.2f}')
    ax.axvline(15, ls=':', color=GRAY, lw=1, alpha=0.5, label='switch -> sell')
    ax.set_xlabel('Trade step')
    ax.set_ylabel('D (recomputed)')
    ax.set_title('3.1a  D-invariant vs Step')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

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

    ax = axes[1, 0]
    ax.plot(df_cmp['Q'], df_cmp['price_hfmm'], '-', color=ACCENT, lw=2, label='HFMM (A=100)')
    ax.plot(df_cmp['Q'], df_cmp['price_cpmm'], '--', color=ACCENT2, lw=2, label='CPMM')
    ax.axhline(100.0, ls=':', color=GRAY, lw=1, label='mid = 100')
    ax.set_xlabel('Quantity Q')
    ax.set_ylabel('Execution Price')
    ax.set_title('3.2a  Exec Price: HFMM vs CPMM')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

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


# ── Routing plots ────────────────────────────────────────────────────

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
                scenario=scenario_name, venue=v,
                empirical_share=counts_r.get(v, 0) / N_reg,
            ))
    df_reg = pd.DataFrame(regime_rows)
    _save_csv(df_reg, 'routing_regimes.csv')

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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('4. Routing  -  Diagnostic Plots', fontsize=14, weight='bold')

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
    print("=" * W)
    print("  UNIT TEST DETAILED REPORT")
    print("=" * W)

    test_clob()
    test_cpmm()
    test_hfmm()
    test_routing()
    test_mm_amm_interaction()
    test_endogenous_mm_withdrawal()
    test_fee_accounting()
    test_fundamentalist_anchor()
    test_environment_price_process()
    test_latent_anchor_reference()
    test_systemic_liquidity_factor()
    test_shock_stress_cli_semantics()
    test_flash_crash_preset()
    test_mm_withdrawal_preset()
    test_funding_liquidity_shock_preset()
    test_initial_depth_matching()
    test_amm_core_liquidity_floor()
    test_realism_shock_mode()
    test_shock_event_window_helpers()
    test_resilience_metrics_censoring()
    test_priority_resilience_metrics()
    test_resilience_statistical_metrics()
    test_arbitrage_trade_cap()
    print(f"\n{'=' * W}")
    print(f"  SUMMARY: {total_pass} passed, {total_fail} failed")
    print(f"{'=' * W}")

    do_plots = '--plots' in sys.argv
    if do_plots:
        _init_plots()
        print(f"\n{'=' * W}")
        print(f"  QA Diagnostic Visualisation")
        print(f"{'=' * W}")
        print(f"  Output -> {os.path.abspath(OUT_DIR)}/")
        plot_clob()
        plot_cpmm()
        plot_hfmm()
        plot_routing()
        print(f"\n{'=' * W}")
        print(f"  Done -- 4 dashboards + CSV saved to output/unit_tests/")
        print(f"{'=' * W}")

    sys.exit(1 if total_fail > 0 else 0)
