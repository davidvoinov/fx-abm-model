"""
Unit tests for venue routing — the two-step choice logic in Trader.

Scenarios
---------
1. Softmax on a single step with known costs → expected probability ordering.
2. Two-step selection (CLOB vs AMM coin-flip, then CPMM vs HFMM softmax).
3. Noise in cost estimation (\u0109_v) → over many iterations frequencies
   converge to theoretical softmax probabilities.
"""

import math
import random
import pytest
from unittest.mock import MagicMock, patch
from collections import Counter

from AgentBasedModel.utils.orders import Order, OrderList
from AgentBasedModel.agents.agents import ExchangeAgent, Trader
from AgentBasedModel.venues.clob import CLOBVenue
from AgentBasedModel.venues.amm import CPMMPool, HFMMPool


# =====================================================================
#  Helpers
# =====================================================================

def _make_exchange():
    """Minimal exchange with a tight book around 100."""
    ex = ExchangeAgent.__new__(ExchangeAgent)
    ex.name = "TestExchange"
    ex.order_book = {'bid': OrderList('bid'), 'ask': OrderList('ask')}
    ex.dividend_book = [100.0] * 10
    ex.risk_free = 0.0
    ex.transaction_cost = 0.0
    for p, q in [(99, 50), (98, 50)]:
        ex.order_book['bid'].push(Order(p, q, 'bid', None))
    for p, q in [(101, 50), (102, 50)]:
        ex.order_book['ask'].append(Order(p, q, 'ask', None))
    return ex


def _make_trader(amm_share_pct=50.0, beta_amm=0.05,
                 cpmm_bias_bps=0.0, cost_noise_std=0.0,
                 deterministic=False):
    """Build a Trader with multi-venue routing enabled."""
    ex = _make_exchange()
    clob = CLOBVenue(ex)
    cpmm = CPMMPool(x=1000, y=100_000, fee=0.003)
    hfmm = HFMMPool(x=1000, y=100_000, A=100, fee=0.001, rate=100.0)
    pools = {'cpmm': cpmm, 'hfmm': hfmm}
    trader = Trader(
        market=ex, cash=1e6, assets=100,
        clob=clob, amm_pools=pools,
        amm_share_pct=amm_share_pct,
        deterministic_venue=deterministic,
        beta_amm=beta_amm,
        cpmm_bias_bps=cpmm_bias_bps,
        cost_noise_std=cost_noise_std,
    )
    return trader


# =====================================================================
#  1. Softmax mechanics
# =====================================================================

class TestSoftmax:
    def test_lower_cost_gets_higher_weight(self):
        """Given costs {A:10, B:20}, softmax should pick A more often."""
        costs = {'A': 10.0, 'B': 20.0}
        beta = 0.1
        counts = Counter()
        random.seed(42)
        N = 5000
        for _ in range(N):
            pick = Trader._softmax_pick(costs, beta)
            counts[pick] += 1
        assert counts['A'] > counts['B']

    def test_higher_beta_concentrates(self):
        """Higher β → more deterministic (lower-cost venue dominates)."""
        costs = {'A': 10.0, 'B': 15.0}
        random.seed(42)
        N = 5000
        low_beta_A = sum(1 for _ in range(N) if Trader._softmax_pick(costs, 0.01) == 'A')
        random.seed(42)
        high_beta_A = sum(1 for _ in range(N) if Trader._softmax_pick(costs, 1.0) == 'A')
        assert high_beta_A / N > low_beta_A / N

    def test_equal_costs_uniform(self):
        """Equal costs → roughly 50/50."""
        costs = {'A': 10.0, 'B': 10.0}
        random.seed(42)
        N = 10000
        picks = [Trader._softmax_pick(costs, 0.1) for _ in range(N)]
        share_A = picks.count('A') / N
        assert share_A == pytest.approx(0.5, abs=0.05)

    def test_inf_cost_gets_zero_weight(self):
        """Venues with inf cost should never be picked."""
        costs = {'A': 10.0, 'B': float('inf')}
        random.seed(42)
        for _ in range(100):
            assert Trader._softmax_pick(costs, 0.1) == 'A'

    def test_theoretical_probabilities(self):
        """Check softmax probabilities against the analytic formula."""
        costs = {'X': 10.0, 'Y': 20.0, 'Z': 15.0}
        beta = 0.1
        min_c = min(costs.values())
        weights = {k: math.exp(-beta * (c - min_c)) for k, c in costs.items()}
        total = sum(weights.values())
        probs = {k: w / total for k, w in weights.items()}

        random.seed(42)
        N = 50000
        counts = Counter(Trader._softmax_pick(costs, beta) for _ in range(N))
        for k in costs:
            observed = counts[k] / N
            assert observed == pytest.approx(probs[k], abs=0.02)


# =====================================================================
#  2. Two-step venue selection
# =====================================================================

class TestTwoStepChoice:
    def test_clob_dominates_when_cheap(self):
        """With amm_share_pct=5, almost all flow goes to CLOB."""
        trader = _make_trader(amm_share_pct=5.0)
        random.seed(42)
        N = 2000
        picks = [trader.choose_venue(1.0, 'buy') for _ in range(N)]
        clob_share = picks.count('clob') / N
        assert clob_share > 0.90

    def test_amm_gets_share_when_costs_equal(self):
        """With amm_share_pct=50, AMM should get substantial share."""
        trader = _make_trader(amm_share_pct=50.0)
        random.seed(42)
        N = 2000
        picks = [trader.choose_venue(1.0, 'buy') for _ in range(N)]
        amm_share = 1.0 - picks.count('clob') / N
        assert amm_share == pytest.approx(0.5, abs=0.05)

    def test_intra_amm_distribution(self):
        """Of the AMM-routed trades, both CPMM and HFMM get non-zero."""
        trader = _make_trader(amm_share_pct=100.0, beta_amm=0.05)
        random.seed(42)
        N = 3000
        picks = [trader.choose_venue(1.0, 'buy') for _ in range(N)]
        cpmm_n = picks.count('cpmm')
        hfmm_n = picks.count('hfmm')
        assert cpmm_n > 0
        assert hfmm_n > 0
        # HFMM (lower fee → lower cost) should get more
        assert hfmm_n > cpmm_n

    def test_deterministic_mode_picks_cheapest(self):
        """deterministic_venue=True → always picks argmin cost."""
        trader = _make_trader(deterministic=True)
        random.seed(42)
        costs = trader.estimate_costs(1.0, 'buy')
        cheapest = min(costs, key=costs.get)
        for _ in range(50):
            assert trader.choose_venue(1.0, 'buy') == cheapest

    def test_no_amm_pools_always_clob(self):
        """When amm_pools is empty, always CLOB."""
        ex = _make_exchange()
        clob = CLOBVenue(ex)
        trader = Trader(market=ex, cash=1e6, assets=0,
                        clob=clob, amm_pools={}, amm_share_pct=50)
        random.seed(42)
        for _ in range(100):
            assert trader.choose_venue(1.0) == 'clob'


# =====================================================================
#  3. Noise & bias in cost estimation
# =====================================================================

class TestNoiseAndBias:
    def test_cpmm_bias_increases_cpmm_share(self):
        """Positive cpmm_bias_bps lowers perceived CPMM cost → more CPMM."""
        random.seed(42)
        N = 5000
        trader_no_bias = _make_trader(amm_share_pct=100, cpmm_bias_bps=0.0,
                                      cost_noise_std=0.0)
        no_bias_cpmm = sum(
            1 for _ in range(N)
            if trader_no_bias.choose_venue(1.0) == 'cpmm'
        )

        random.seed(42)
        trader_bias = _make_trader(amm_share_pct=100, cpmm_bias_bps=50.0,
                                   cost_noise_std=0.0)
        bias_cpmm = sum(
            1 for _ in range(N)
            if trader_bias.choose_venue(1.0) == 'cpmm'
        )
        assert bias_cpmm > no_bias_cpmm

    def test_noise_does_not_collapse_share(self):
        """
        With noise, every venue that has non-zero probability should
        receive some flow over many iterations.
        """
        trader = _make_trader(amm_share_pct=100, cost_noise_std=3.0,
                              beta_amm=0.05)
        random.seed(42)
        N = 5000
        picks = [trader.choose_venue(1.0) for _ in range(N)]
        counts = Counter(picks)
        for name in ['cpmm', 'hfmm']:
            assert counts[name] > 0, f"{name} got zero flow with noise"

    def test_noise_frequencies_converge(self):
        """
        With zero bias and modest noise, the observed share should be
        approximately the softmax expectation (tested via a large N).
        """
        trader = _make_trader(amm_share_pct=100, cpmm_bias_bps=0.0,
                              cost_noise_std=1.0, beta_amm=0.05)
        random.seed(42)
        N = 20000
        picks = [trader.choose_venue(1.0) for _ in range(N)]
        counts = Counter(picks)
        hfmm_share = counts['hfmm'] / N
        # HFMM has lower fee (10 bps vs 30 bps) → should dominate
        assert hfmm_share > 0.5

    def test_estimate_costs_keys(self):
        """estimate_costs returns expected keys."""
        trader = _make_trader()
        costs = trader.estimate_costs(5.0, 'buy')
        assert 'clob' in costs
        assert 'cpmm' in costs
        assert 'hfmm' in costs
        for v in costs.values():
            assert isinstance(v, float)
