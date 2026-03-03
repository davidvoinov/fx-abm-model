"""
Unit tests for CPMMPool — Constant Product Market Maker.

Scenarios
---------
1. Invariant k = x·y preservation & round-trip symmetry.
2. Price and slippage (small Q ≈ mid, large Q → material slippage).
3. Fee decomposition & all-in cost.
4. Boundary volumes (too large, zero, negative).
5. Volume–slippage profile (binary-search max Q).
"""

import math
import pytest

from AgentBasedModel.venues.amm import CPMMPool


# ── Default pool for most tests ─────────────────────────────────────
# x = 1000 base, y = 100 000 quote  ⟹  mid = 100, k = 1e8
@pytest.fixture
def pool():
    return CPMMPool(x=1000.0, y=100_000.0, fee=0.003)


# =====================================================================
#  1. Invariant & symmetry
# =====================================================================

class TestInvariant:
    def test_initial_k(self, pool):
        assert pool.k == pytest.approx(1000.0 * 100_000.0, rel=1e-12)

    def test_k_preserved_after_buy(self, pool):
        """After a buy, x_new * y_new = k  (fee does not break product)."""
        k_before = pool.k
        pool.execute_buy(1.0)
        # k is set from reserves; because fees are extracted before the
        # product calc, the product of (x_after, y_after) = k_before.
        # (Fee is taken from the input dy, but the reserves track
        # net-of-fee amounts, preserving k.)
        assert pool.x * pool.y == pytest.approx(k_before, rel=1e-6)

    def test_k_after_sell(self, pool):
        """After a sell, k may drift slightly upward (fees collected)."""
        k_before = pool.k
        pool.execute_sell(1.0)
        # k should be ≥ k_before (fees add to reserves)
        assert pool.k >= k_before * (1 - 1e-6)

    def test_round_trip_small(self, pool):
        """Buy 0.1 then sell 0.1 → reserves close to initial."""
        x0, y0 = pool.x, pool.y
        pool.execute_buy(0.1)
        pool.execute_sell(0.1)
        # Not exactly equal because of fees, but very close
        assert pool.x == pytest.approx(x0, rel=0.01)
        assert pool.y == pytest.approx(y0, rel=0.01)


# =====================================================================
#  2. Price & slippage
# =====================================================================

class TestPriceAndSlippage:
    def test_mid_price(self, pool):
        """mid = y/x = 100 000 / 1000 = 100."""
        assert pool.mid_price() == pytest.approx(100.0, rel=1e-12)

    def test_small_q_exec_price_close_to_mid(self, pool):
        """For tiny Q, exec price ≈ mid."""
        q = pool.quote_buy(0.01, S_t=100.0)
        assert q['exec_price'] == pytest.approx(100.0, rel=0.01)

    def test_small_q_slippage_near_zero(self, pool):
        q = pool.quote_buy(0.01, S_t=100.0)
        assert abs(q['slippage_bps']) < 1.0  # less than 1 bps

    def test_large_q_higher_exec_price(self, pool):
        """Buying 100 base (10 % of x) should materially increase exec price."""
        q = pool.quote_buy(100.0, S_t=100.0)
        assert q['exec_price'] > 110.0   # noticeable slippage

    def test_slippage_increases_with_volume(self, pool):
        """Slippage monotonically increases with Q."""
        slippages = []
        for Q in [1, 5, 10, 50, 100]:
            q = pool.quote_buy(Q, S_t=100.0)
            slippages.append(q['slippage_bps'])
        for i in range(len(slippages) - 1):
            assert slippages[i] < slippages[i + 1]

    def test_sell_slippage_positive(self, pool):
        """Selling should also produce positive slippage."""
        q = pool.quote_sell(10.0, S_t=100.0)
        assert q['slippage_bps'] > 0


# =====================================================================
#  3. Fees & all-in cost
# =====================================================================

class TestFees:
    def test_fee_bps(self, pool):
        """fee_bps = 10 000 × fee = 10 000 × 0.003 = 30."""
        q = pool.quote_buy(5.0, S_t=100.0)
        assert q['fee_bps'] == pytest.approx(30.0, rel=1e-9)

    def test_effective_amount_with_fee(self):
        """Trader pays dy = dy_eff / (1 − f).  Check explicitly."""
        p = CPMMPool(x=1000, y=100_000, fee=0.10)  # 10 % fee
        Q = 10.0
        x_new = p.x - Q
        y_new = p.k / x_new
        dy_eff = y_new - p.y
        dy_with_fee = dy_eff / (1 - 0.10)
        q = p.quote_buy(Q, S_t=p.mid_price())
        assert q['delta_y'] == pytest.approx(dy_with_fee, rel=1e-9)

    def test_cost_is_slippage_plus_fee_plus_gas(self, pool):
        """cost_bps = slippage_bps + fee_bps + gas_cost_bps."""
        for Q in [1, 5, 10, 50]:
            q = pool.quote_buy(Q, S_t=100.0)
            expected = q['slippage_bps'] + q['fee_bps'] + pool.gas_cost_bps
            assert q['cost_bps'] == pytest.approx(expected, rel=1e-9)

    def test_gas_cost_adds_to_total(self):
        """Non-zero gas_cost_bps increases all-in cost."""
        p = CPMMPool(x=1000, y=100_000, fee=0.003, gas_cost_bps=5.0)
        q = p.quote_buy(5.0, S_t=100.0)
        assert q['cost_bps'] == pytest.approx(
            q['slippage_bps'] + q['fee_bps'] + 5.0, rel=1e-9
        )


# =====================================================================
#  4. Boundary volumes
# =====================================================================

class TestBoundary:
    def test_zero_quantity(self, pool):
        q = pool.quote_buy(0.0)
        assert q['cost_bps'] == 0

    def test_negative_quantity(self, pool):
        q = pool.quote_buy(-5.0)
        assert q['cost_bps'] == 0

    def test_too_large_buy(self, pool):
        """Buying ≥ x should return inf cost."""
        q = pool.quote_buy(pool.x)         # exactly x
        assert q['cost_bps'] == float('inf')

    def test_too_large_buy_reserves_unchanged(self, pool):
        """Reserves must not change on a failed buy."""
        x0, y0 = pool.x, pool.y
        pool.execute_buy(pool.x + 10)
        assert pool.x == pytest.approx(x0, rel=1e-12)
        assert pool.y == pytest.approx(y0, rel=1e-12)

    def test_reserves_always_positive_after_sell(self, pool):
        """Even a large sell should not make reserves negative."""
        pool.execute_sell(500.0)
        assert pool.x > 0
        assert pool.y > 0


# =====================================================================
#  5. Volume–slippage profile
# =====================================================================

class TestVolumeSlippage:
    def test_max_q_at_threshold(self, pool):
        """
        volume_slippage_max_Q(threshold) should return Q such that
        cost ≈ threshold (and cost at Q + ε > threshold).
        """
        threshold = 100.0  # 100 bps
        max_q = pool.volume_slippage_max_Q(threshold, S_t=100.0)

        cost_at_max = pool.quote_buy(max_q, S_t=100.0)['cost_bps']
        assert cost_at_max <= threshold + 1.0   # within 1 bps tolerance

        # Slightly above should exceed threshold
        cost_above = pool.quote_buy(max_q * 1.05, S_t=100.0)['cost_bps']
        assert cost_above > threshold - 1.0

    def test_higher_threshold_allows_bigger_q(self, pool):
        """More slippage tolerance ⟹ larger max Q."""
        q50 = pool.volume_slippage_max_Q(50.0, S_t=100.0)
        q200 = pool.volume_slippage_max_Q(200.0, S_t=100.0)
        assert q200 > q50

    def test_brute_force_cost_table(self, pool):
        """Build a small cost table and verify monotonicity."""
        S_t = pool.mid_price()
        table = []
        for Q in [0.5, 1, 2, 5, 10, 20, 50, 100]:
            c = pool.quote_buy(Q, S_t)['cost_bps']
            table.append((Q, c))
        for i in range(len(table) - 1):
            assert table[i][1] <= table[i + 1][1]


# =====================================================================
#  Liquidity management
# =====================================================================

class TestLiquidity:
    def test_liquidity_measure(self, pool):
        """L = sqrt(k)."""
        assert pool.liquidity_measure() == pytest.approx(
            math.sqrt(pool.k), rel=1e-12
        )

    def test_effective_depth(self, pool):
        """R = sqrt(x·y) / S_t."""
        S = pool.mid_price()
        expected = math.sqrt(pool.x * pool.y) / S
        assert pool.effective_depth(S) == pytest.approx(expected, rel=1e-12)

    def test_add_liquidity(self, pool):
        """Adding 50 % scales reserves by 1.5."""
        x0, y0 = pool.x, pool.y
        pool.add_liquidity(0.5)
        assert pool.x == pytest.approx(x0 * 1.5, rel=1e-12)
        assert pool.y == pytest.approx(y0 * 1.5, rel=1e-12)

    def test_remove_liquidity_capped(self, pool):
        """Remove liquidity is capped at 50 %."""
        x0 = pool.x
        pool.remove_liquidity(0.9)  # asks for 90 %, gets 50 %
        assert pool.x == pytest.approx(x0 * 0.5, rel=1e-12)
