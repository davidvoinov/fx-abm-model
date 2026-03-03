"""
Unit tests for CLOBVenue — order-book analytics layer.

Scenarios
---------
1. Mid-price & quoted spread on a synthetic book.
2. Execution price & effective spread (single-level and multi-level).
3. Price impact estimation.
4. All-in cost = half_spread + fee (component sum).
5. Depth calculation (cumulative and within ±δ bps).
"""

import math
import pytest

from AgentBasedModel.utils.orders import Order, OrderList
from AgentBasedModel.agents.agents import ExchangeAgent
from AgentBasedModel.venues.clob import CLOBVenue


# =====================================================================
#  Helpers — build a tiny, fully-controlled order book
# =====================================================================

def _make_exchange_with_book(bids, asks):
    """
    Build an ExchangeAgent whose book contains exactly the given levels.

    bids : list of (price, qty) — sorted best-first (highest first)
    asks : list of (price, qty) — sorted best-first (lowest first)
    """
    ex = ExchangeAgent.__new__(ExchangeAgent)
    ex.name = "TestExchange"
    ex.order_book = {
        'bid': OrderList('bid'),
        'ask': OrderList('ask'),
    }
    ex.dividend_book = [100.0] * 10
    ex.risk_free = 0.0
    ex.transaction_cost = 0.0

    # Bids: best (highest price) first → push each so the linked list
    # keeps best-first ordering (push = prepend).
    for p, q in reversed(bids):
        ex.order_book['bid'].push(Order(p, q, 'bid', None))

    # Asks: best (lowest price) first → append each
    for p, q in asks:
        ex.order_book['ask'].append(Order(p, q, 'ask', None))

    return ex


# ── Canonical book used by most tests ────────────────────────────────
# Bids: 99.0 × 10,  98.0 × 20,  97.0 × 30
# Asks: 101.0 × 10, 102.0 × 20, 103.0 × 30
BIDS = [(99.0, 10), (98.0, 20), (97.0, 30)]
ASKS = [(101.0, 10), (102.0, 20), (103.0, 30)]


@pytest.fixture
def clob():
    ex = _make_exchange_with_book(BIDS, ASKS)
    return CLOBVenue(ex, f_broker=0.0, f_venue=0.0)


@pytest.fixture
def clob_with_fees():
    ex = _make_exchange_with_book(BIDS, ASKS)
    return CLOBVenue(ex, f_broker=2.0, f_venue=1.0)   # 3 bps total fee


# =====================================================================
#  1. Mid-price & quoted spread
# =====================================================================

class TestMidAndSpread:
    def test_mid_price(self, clob):
        """mid = (best_bid + best_ask) / 2 = (99 + 101) / 2 = 100."""
        assert clob.mid_price() == pytest.approx(100.0, abs=0.1)

    def test_quoted_spread_bps(self, clob):
        """qspr = 10 000 × (ask − bid) / mid = 10 000 × 2 / 100 = 200 bps."""
        assert clob.quoted_spread_bps() == pytest.approx(200.0, rel=1e-6)


# =====================================================================
#  2. Execution price & effective spread
# =====================================================================

class TestExecutionPrice:
    def test_small_buy_single_level(self, clob):
        """Buy Q=5 fills entirely at best ask = 101.  VWAP = 101."""
        q = clob.quote_buy(5)
        assert q['exec_price'] == pytest.approx(101.0, rel=1e-9)

    def test_small_buy_effective_spread(self, clob):
        """espr = 2 × half_spread.  half = 10 000 × (101−100)/100 = 100 bps."""
        q = clob.quote_buy(5)
        half = 10_000.0 * (101.0 - 100.0) / 100.0  # 100 bps
        assert q['half_spread_bps'] == pytest.approx(half, rel=1e-6)
        assert q['espr_bps'] == pytest.approx(2 * half, rel=1e-6)

    def test_multi_level_buy(self, clob):
        """
        Buy Q=25 sweeps:
            10 × 101 = 1010
            15 × 102 = 1530
        VWAP = 2540 / 25 = 101.6
        """
        q = clob.quote_buy(25)
        expected_vwap = (10 * 101.0 + 15 * 102.0) / 25.0
        assert q['exec_price'] == pytest.approx(expected_vwap, rel=1e-9)
        expected_half = 10_000.0 * (expected_vwap - 100.0) / 100.0
        assert q['half_spread_bps'] == pytest.approx(expected_half, rel=1e-6)

    def test_full_ask_sweep(self, clob):
        """Buy Q=60 consumes all 3 ask levels."""
        q = clob.quote_buy(60)
        expected_vwap = (10 * 101 + 20 * 102 + 30 * 103) / 60.0
        assert q['exec_price'] == pytest.approx(expected_vwap, rel=1e-9)

    def test_exceeds_book(self, clob):
        """Buy Q=100 — not enough liquidity → inf cost."""
        q = clob.quote_buy(100)
        assert q['cost_bps'] == float('inf')


# =====================================================================
#  3. Price impact
# =====================================================================

class TestPriceImpact:
    def test_impact_small_order(self, clob):
        """
        Buy Q=5 from ask-level 101 (qty 10).  After consuming 5 units the
        best ask is still 101.  New mid = (99 + 101)/2 = 100 → impact = 0.
        """
        q = clob.quote_buy(5)
        assert q['impact_bps'] == pytest.approx(0.0, abs=1e-6)

    def test_impact_consuming_first_level(self, clob):
        """
        Buy Q=10 exhausts best ask (101 × 10).
        New best ask = 102.  New mid = (99 + 102)/2 = 100.5.
        impact = 10 000 × (100.5 − 100) / 100 = 50 bps.
        """
        q = clob.quote_buy(10)
        assert q['impact_bps'] == pytest.approx(50.0, rel=1e-6)

    def test_impact_symmetric_sell(self, clob):
        """
        Sell Q=10 exhausts best bid (99 × 10).
        New best bid = 98.  New mid = (98 + 101)/2 = 99.5.
        impact = 10 000 × (99.5 − 100) / 100 = −50 bps.
        """
        q = clob.quote_sell(10)
        assert q['impact_bps'] == pytest.approx(-50.0, rel=1e-6)


# =====================================================================
#  4. All-in cost = half_spread + fee
# =====================================================================

class TestAllInCost:
    def test_cost_zero_fee(self, clob):
        """cost = half_spread + 0."""
        q = clob.quote_buy(5)
        assert q['cost_bps'] == pytest.approx(q['half_spread_bps'], rel=1e-9)

    def test_cost_with_fee(self, clob_with_fees):
        """cost = half_spread + (f_broker + f_venue)."""
        q = clob_with_fees.quote_buy(5)
        assert q['fee_bps'] == pytest.approx(3.0, rel=1e-9)
        assert q['cost_bps'] == pytest.approx(q['half_spread_bps'] + 3.0, rel=1e-9)

    def test_cost_components_sum(self, clob_with_fees):
        """Component accounting: cost_bps = half_spread_bps + fee_bps."""
        for Q in [1, 5, 10, 25]:
            q = clob_with_fees.quote_buy(Q)
            if q['cost_bps'] < float('inf'):
                assert q['cost_bps'] == pytest.approx(
                    q['half_spread_bps'] + q['fee_bps'], rel=1e-9
                )


# =====================================================================
#  5. Depth
# =====================================================================

class TestDepth:
    def test_depth_levels(self, clob):
        """depth(n_levels=3) returns all 3 price levels per side."""
        d = clob.depth(n_levels=3)
        assert len(d['bid']) == 3
        assert len(d['ask']) == 3

    def test_depth_volumes(self, clob):
        """Cumulative volumes on each level match the synthetic book."""
        d = clob.depth(n_levels=3)
        ask_vols = [vol for _, vol in d['ask']]
        assert ask_vols == pytest.approx([10, 20, 30], abs=0.01)

    def test_total_depth_wide_corridor(self, clob):
        """Wide corridor (10 000 bps = 100 %) encompasses the whole book."""
        td = clob.total_depth(bps_from_mid=10_000)
        assert td['bid'] == pytest.approx(60, abs=0.01)
        assert td['ask'] == pytest.approx(60, abs=0.01)

    def test_total_depth_narrow_corridor(self, clob):
        """±100 bps around mid = 100 → [99, 101].  Only best levels qualify."""
        td = clob.total_depth(bps_from_mid=100)
        # bid ≥ 99.0 → level 99 × 10 = 10
        assert td['bid'] == pytest.approx(10, abs=0.01)
        # ask ≤ 101.0 → level 101 × 10 = 10
        assert td['ask'] == pytest.approx(10, abs=0.01)


# =====================================================================
#  Edge cases
# =====================================================================

class TestEdgeCases:
    def test_zero_quantity(self, clob):
        q = clob.quote_buy(0)
        assert q['cost_bps'] == 0.0

    def test_negative_quantity(self, clob):
        q = clob.quote_buy(-5)
        assert q['cost_bps'] == 0.0

    def test_empty_book(self):
        """Completely empty book → mid falls back, spread is inf."""
        ex = ExchangeAgent.__new__(ExchangeAgent)
        ex.name = "EmptyExchange"
        ex.order_book = {'bid': OrderList('bid'), 'ask': OrderList('ask')}
        ex.dividend_book = [100.0]
        ex.risk_free = 0.0
        ex.transaction_cost = 0.0
        c = CLOBVenue(ex)
        assert c.mid_price() == 100.0             # fallback to dividend
        assert c.quoted_spread_bps() == float('inf')

    def test_cost_bps_convenience(self, clob):
        """cost_bps(Q, side) matches quote dict."""
        for Q in [5, 10, 25]:
            assert clob.cost_bps(Q, 'buy') == clob.quote_buy(Q)['cost_bps']
            assert clob.cost_bps(Q, 'sell') == clob.quote_sell(Q)['cost_bps']
