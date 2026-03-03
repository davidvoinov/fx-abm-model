#!/usr/bin/env python3
"""
Detailed unit-test report — prints every scenario with expected vs actual
values so the full verification is visible in a single output.

Run:
    python tests/run_unit_report.py
"""
import sys, os, math, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from AgentBasedModel.utils.orders import Order, OrderList
from AgentBasedModel.agents.agents import ExchangeAgent, Trader
from AgentBasedModel.venues.clob import CLOBVenue
from AgentBasedModel.venues.amm import (
    CPMMPool, HFMMPool,
    _hfmm_get_D, _hfmm_get_y, _hfmm_mid_price,
)
from collections import Counter

W = 72
PASS = "✓ PASS"
FAIL = "✗ FAIL"
total_pass = 0
total_fail = 0


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


# =====================================================================
#  HELPERS
# =====================================================================

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
#  CLOB VENUE
# =====================================================================

def test_clob():
    section("CLOB VENUE — Unit Tests")

    ex = _make_exchange(BIDS, ASKS)
    clob = CLOBVenue(ex, f_broker=0.0, f_venue=0.0)
    clob_fee = CLOBVenue(_make_exchange(BIDS, ASKS), f_broker=2.0, f_venue=1.0)

    # 1. Mid & spread
    subsection("1. Mid-price & Quoted Spread")
    mid = clob.mid_price()
    check("mid_price = (99+101)/2 = 100.0", mid, 100.0)
    qspr = clob.quoted_spread_bps()
    check("qspr = 10000×(101−99)/100 = 200 bps", qspr, 200.0)

    # 2. Execution price & effective spread
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

    # 3. Price impact
    subsection("3. Price Impact")

    q5 = clob.quote_buy(5)
    check("Buy Q=5: impact = 0 (best ask survives)", q5['impact_bps'], 0.0)

    q10 = clob.quote_buy(10)
    # After consuming 10@101, new best ask = 102, new mid = (99+102)/2 = 100.5
    check("Buy Q=10: impact = 10000×(100.5−100)/100 = 50 bps", q10['impact_bps'], 50.0)

    qs10 = clob.quote_sell(10)
    check("Sell Q=10: impact = 10000×(99.5−100)/100 = −50 bps", qs10['impact_bps'], -50.0)

    # 4. All-in cost
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

    # 5. Depth
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

    # Edge cases
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


# =====================================================================
#  CPMM VENUE
# =====================================================================

def test_cpmm():
    section("CPMM VENUE — Unit Tests")

    # 1. Invariant
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

    # 2. Price & slippage
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

    # 3. Fees
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

    # 4. Boundary
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

    # 5. Volume-slippage profile
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

    # Liquidity
    subsection("Liquidity Management")
    p = CPMMPool(x=1000.0, y=100_000.0, fee=0.003)
    check("L = sqrt(k)", p.liquidity_measure(), math.sqrt(p.k))
    check("R = sqrt(x·y)/S", p.effective_depth(100.0),
          math.sqrt(p.x * p.y) / 100.0)


# =====================================================================
#  HFMM VENUE
# =====================================================================

def test_hfmm():
    section("HFMM VENUE — Unit Tests")

    # 1. Invariant D
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

    # 2. Local linearity
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

    # HFMM vs CPMM slippage
    cpmm = CPMMPool(x=1000, y=100_000, fee=0.0)
    hfmm = HFMMPool(x=1000, y=100_000, A=100, fee=0.0, rate=100.0)
    Q = 10.0
    slip_c = cpmm.quote_buy(Q, 100.0)['slippage_bps']
    slip_h = hfmm.quote_buy(Q, 100.0)['slippage_bps']
    check_bool(f"HFMM slippage < CPMM at Q={Q}",
               slip_h < slip_c,
               f"HFMM={slip_h:.2f} bps, CPMM={slip_c:.2f} bps")

    # 3. Transition to CPMM-like
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

    # 4. Cost / fee
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

    # 5. Numerical stability
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

    # Balanced mid-price
    x, y, A = 100.0, 100.0, 50.0
    D = _hfmm_get_D(x, y, A)
    mp = _hfmm_mid_price(x, y, A, D)
    check("Balanced (x=y=100): marginal price ≈ 1.0", mp, 1.0, rel=1e-6)

    # Rate update
    subsection("Rate Update")
    p = HFMMPool(x=1000.0, y=100_000.0, A=100.0, fee=0.001, rate=100.0)
    r0 = p.rate
    p.update_rate(threshold=0.01)
    check_bool("No drift → rate unchanged", p.rate == r0)

    # Use low-A pool for rate drift (A=1 → more curvature → bigger mid shift)
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
#  ROUTING
# =====================================================================

def test_routing():
    section("VENUE ROUTING — Unit Tests")

    def _make_exchange_r():
        return _make_exchange([(99, 50), (98, 50)], [(101, 50), (102, 50)])

    def _make_trader(amm_share_pct=50, beta_amm=0.05, bias=0.0, noise=0.0, det=False):
        ex = _make_exchange_r()
        clob = CLOBVenue(ex)
        cpmm = CPMMPool(x=1000, y=100_000, fee=0.003)
        hfmm = HFMMPool(x=1000, y=100_000, A=100, fee=0.001, rate=100.0)
        return Trader(market=ex, cash=1e6, assets=100, clob=clob,
                      amm_pools={'cpmm': cpmm, 'hfmm': hfmm},
                      amm_share_pct=amm_share_pct, deterministic_venue=det,
                      beta_amm=beta_amm, cpmm_bias_bps=bias, cost_noise_std=noise)

    # 1. Softmax
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

    # 2. Two-step
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

    # Deterministic
    trader_det = _make_trader(det=True)
    costs_d = trader_det.estimate_costs(1.0, 'buy')
    cheapest = min(costs_d, key=costs_d.get)
    all_same = all(trader_det.choose_venue(1.0) == cheapest for _ in range(50))
    check_bool(f"Deterministic: always picks {cheapest}",
               all_same, f"costs = {costs_d}")

    # 3. Bias & noise
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

    # Estimate costs
    subsection("estimate_costs()")
    costs_e = _make_trader().estimate_costs(5.0, 'buy')
    print(f"    costs = {costs_e}")
    check_bool("Keys: clob, cpmm, hfmm",
               set(costs_e.keys()) == {'clob', 'cpmm', 'hfmm'})


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

    print(f"\n{'=' * W}")
    print(f"  SUMMARY: {total_pass} passed, {total_fail} failed")
    print(f"{'=' * W}")

    sys.exit(1 if total_fail > 0 else 0)
