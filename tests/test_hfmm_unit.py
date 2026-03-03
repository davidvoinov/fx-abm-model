"""
Unit tests for HFMMPool — Hybrid Function Market Maker (StableSwap).

Scenarios
---------
1. Invariant D preservation after swaps.
2. Local linearity near parity (tiny Q → exec ≈ S_t, low slippage).
3. Transition to CPMM-like behaviour under heavy imbalance.
4. Cost / fee decomposition.
5. Numerical stability of the D-solver for varied configurations.
"""

import math
import pytest

from AgentBasedModel.venues.amm import (
    HFMMPool,
    CPMMPool,
    _hfmm_get_D,
    _hfmm_get_y,
    _hfmm_mid_price,
)


# ── Balanced pool near parity (quote/base ≈ 100) ────────────────────
@pytest.fixture
def pool():
    return HFMMPool(x=1000.0, y=100_000.0, A=100.0, fee=0.001, rate=100.0)


@pytest.fixture
def pool_low_A():
    """Low amplification → more CPMM-like."""
    return HFMMPool(x=1000.0, y=100_000.0, A=1.0, fee=0.001, rate=100.0)


# =====================================================================
#  1. Invariant D preservation
# =====================================================================

class TestInvariantD:
    def test_D_positive(self, pool):
        assert pool.D > 0

    def test_D_preserved_after_buy(self, pool):
        """After a buy, re-solving for D matches the original."""
        D0 = pool.D
        pool.execute_buy(1.0)
        pool._sync_norm()
        D_recomputed = _hfmm_get_D(pool._xn, pool._yn, pool.A)
        assert D_recomputed == pytest.approx(D0, rel=1e-6)

    def test_D_preserved_after_sell(self, pool):
        D0 = pool.D
        pool.execute_sell(1.0)
        pool._sync_norm()
        D_recomputed = _hfmm_get_D(pool._xn, pool._yn, pool.A)
        assert D_recomputed == pytest.approx(D0, rel=1e-6)

    def test_D_preserved_multi_swaps(self, pool):
        """Several small swaps should not cause D to drift."""
        D0 = pool.D
        for _ in range(10):
            pool.execute_buy(0.5)
        for _ in range(10):
            pool.execute_sell(0.5)
        pool._sync_norm()
        D_recomputed = _hfmm_get_D(pool._xn, pool._yn, pool.A)
        assert D_recomputed == pytest.approx(D0, rel=1e-4)


# =====================================================================
#  2. Local linearity near parity
# =====================================================================

class TestLocalLinearity:
    def test_small_q_exec_price_close_to_mid(self, pool):
        """Tiny buy → exec price ≈ mid ≈ S_t."""
        S = pool.mid_price()
        q = pool.quote_buy(0.01, S_t=S)
        assert q['exec_price'] == pytest.approx(S, rel=0.01)

    def test_small_q_slippage_near_zero(self, pool):
        q = pool.quote_buy(0.01, S_t=pool.mid_price())
        assert abs(q['slippage_bps']) < 1.0

    def test_hfmm_lower_slippage_than_cpmm(self):
        """
        For the same reserves & Q, HFMM (high A) should have lower
        slippage than CPMM — that's the whole point of StableSwap.
        """
        cpmm = CPMMPool(x=1000, y=100_000, fee=0.0)
        hfmm = HFMMPool(x=1000, y=100_000, A=100, fee=0.0, rate=100.0)
        S = 100.0
        Q = 10.0
        slip_cpmm = cpmm.quote_buy(Q, S)['slippage_bps']
        slip_hfmm = hfmm.quote_buy(Q, S)['slippage_bps']
        assert slip_hfmm < slip_cpmm


# =====================================================================
#  3. Transition to CPMM-like behaviour
# =====================================================================

class TestCPMMTransition:
    def test_large_q_significant_slippage(self, pool):
        """
        At Q = 200 (20 % of reserves) with A=100, StableSwap is very
        flat near parity, so slippage is modest but non-trivial.
        """
        q = pool.quote_buy(200.0, S_t=pool.mid_price())
        assert q['slippage_bps'] > 5   # at least a few bps

    def test_low_A_behaves_like_cpmm(self, pool_low_A):
        """With A=1, slippage should be close to CPMM at same reserves."""
        cpmm = CPMMPool(x=1000, y=100_000, fee=0.0)
        S = 100.0
        Q = 50.0
        slip_low_a = pool_low_A.quote_buy(Q, S)['slippage_bps']
        slip_cpmm = cpmm.quote_buy(Q, S)['slippage_bps']
        # Should be in the same order of magnitude (within 3×)
        assert slip_low_a == pytest.approx(slip_cpmm, rel=2.0)

    def test_no_negative_reserves(self, pool):
        """Even a huge buy should not produce negative reserves."""
        big_Q = pool.x * 0.95  # 95 % of base
        pool.execute_buy(big_Q)
        assert pool.x > 0
        assert pool.y > 0

    def test_imbalanced_sell_reserves_positive(self, pool):
        """Large sell after large buy → reserves still positive."""
        pool.execute_buy(pool.x * 0.4)
        pool.execute_sell(200.0)
        assert pool.x > 0
        assert pool.y > 0


# =====================================================================
#  4. Cost / fee decomposition
# =====================================================================

class TestCostAndFees:
    def test_fee_bps(self, pool):
        """fee_bps = 10 000 × fee = 10."""
        q = pool.quote_buy(5.0, S_t=pool.mid_price())
        assert q['fee_bps'] == pytest.approx(10.0, rel=1e-9)

    def test_cost_equals_slippage_plus_fee_plus_gas(self, pool):
        for Q in [0.5, 1, 5, 10, 50]:
            q = pool.quote_buy(Q, S_t=pool.mid_price())
            if q['cost_bps'] < float('inf'):
                expected = q['slippage_bps'] + q['fee_bps'] + pool.gas_cost_bps
                assert q['cost_bps'] == pytest.approx(expected, rel=1e-9)

    def test_sell_cost_decomposition(self, pool):
        q = pool.quote_sell(5.0, S_t=pool.mid_price())
        expected = q['slippage_bps'] + q['fee_bps'] + pool.gas_cost_bps
        assert q['cost_bps'] == pytest.approx(expected, rel=1e-9)

    def test_exec_price_formula(self, pool):
        """exec_price = delta_y / Q."""
        Q = 5.0
        q = pool.quote_buy(Q, S_t=pool.mid_price())
        assert q['exec_price'] == pytest.approx(q['delta_y'] / Q, rel=1e-9)


# =====================================================================
#  5. Numerical stability of the D-solver
# =====================================================================

class TestNumericalStability:
    @pytest.mark.parametrize("x1,x2,A", [
        (100, 100, 1),
        (100, 100, 10),
        (100, 100, 1000),
        (1, 1_000_000, 50),
        (1_000_000, 1, 50),
        (0.001, 0.001, 100),
        (1e8, 1e8, 500),
    ])
    def test_D_converges(self, x1, x2, A):
        """D-solver should converge for diverse (x1, x2, A) combos."""
        D = _hfmm_get_D(x1, x2, A)
        assert D > 0
        assert math.isfinite(D)

    @pytest.mark.parametrize("x1,x2,A", [
        (100, 100, 1),
        (100, 100, 100),
        (50, 200, 50),
    ])
    def test_get_y_roundtrip(self, x1, x2, A):
        """get_y(x, D, A) should return the original x2."""
        D = _hfmm_get_D(x1, x2, A)
        y_recovered = _hfmm_get_y(x1, D, A)
        assert y_recovered == pytest.approx(x2, rel=1e-8)

    def test_too_large_buy_returns_inf(self, pool):
        """Buying ≥ x → inf cost."""
        q = pool.quote_buy(pool.x)
        assert q['cost_bps'] == float('inf')

    def test_mid_price_formula(self):
        """Verify mid_price against the analytic formula."""
        x, y, A = 100.0, 100.0, 50.0
        D = _hfmm_get_D(x, y, A)
        mp = _hfmm_mid_price(x, y, A, D)
        # At balanced reserves (x == y), price should be ≈ 1
        assert mp == pytest.approx(1.0, rel=1e-6)


# =====================================================================
#  Rate update
# =====================================================================

class TestRateUpdate:
    def test_update_rate_no_drift(self, pool):
        """If mid ≈ rate, update_rate is a no-op."""
        rate_before = pool.rate
        D_before = pool.D
        pool.update_rate(threshold=0.01)
        assert pool.rate == rate_before
        assert pool.D == D_before

    def test_update_rate_after_drift(self, pool_low_A):
        """After large trade on low-A pool, mid drifts; update_rate recenters."""
        pool_low_A.execute_buy(pool_low_A.x * 0.40)
        mp = pool_low_A.mid_price()
        # low-A pool has more curvature → mid shifts materially
        assert abs(mp - pool_low_A.rate) / pool_low_A.rate > 0.01
        pool_low_A.update_rate(threshold=0.01)
        assert pool_low_A.rate == pytest.approx(mp, rel=1e-6)


# =====================================================================
#  Volume-slippage profile
# =====================================================================

class TestVolumeSlippage:
    def test_max_q_at_threshold(self, pool):
        threshold = 50.0
        max_q = pool.volume_slippage_max_Q(threshold, S_t=pool.mid_price())
        cost = pool.quote_buy(max_q, S_t=pool.mid_price())['cost_bps']
        assert cost <= threshold + 1.0

    def test_higher_threshold_allows_bigger_q(self, pool):
        S = pool.mid_price()
        q30 = pool.volume_slippage_max_Q(30, S)
        q100 = pool.volume_slippage_max_Q(100, S)
        assert q100 > q30
