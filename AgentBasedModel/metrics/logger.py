"""
Comprehensive metrics logger for multi-venue FX ABM.

Records per-iteration snapshots of:
    - Execution cost curves C_v(Q) for CLOB, CPMM, HFMM
    - CLOB liquidity: qspr, espr, impact, depth, Amihud
    - AMM liquidity: reserves, D, volume-slippage profiles
    - Flow allocation: share of volume to each venue
    - Environment: σ_t, c_t, regime
    - Individual trades
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Dict
import math

if TYPE_CHECKING:
    from AgentBasedModel.venues.clob import CLOBVenue
    from AgentBasedModel.venues.amm import CPMMPool, HFMMPool
    from AgentBasedModel.environment.processes import MarketEnvironment

# θ size buckets
SIZE_BUCKETS = ('small', 'medium', 'large')


class MetricsLogger:
    """
    Centralized logger. Call ``snapshot(t, ...)`` each iteration to record
    all metrics.  After simulation, retrieve via properties / methods.
    """

    def __init__(self, Q_grid: Optional[List[float]] = None,
                 slippage_thresholds: Optional[List[float]] = None):
        """
        Parameters
        ----------
        Q_grid : list of float
            Trade sizes at which to evaluate execution cost curves each period.
        slippage_thresholds : list of float
            Slippage thresholds (bps) for volume-slippage profile.
        """
        self.Q_grid = Q_grid or [1, 2, 5, 10, 20, 50]
        self.slippage_thresholds = slippage_thresholds or [5, 10, 25, 50]

        # --- per-iteration records ----------------------------------------
        self.iterations: List[int] = []

        # Environment
        self.sigma_series: List[float] = []
        self.c_series: List[float] = []
        self.regime_series: List[str] = []

        # Prices
        self.clob_mid_series: List[float] = []
        # {pool_name: [mid_price_per_iter]}
        self.amm_mid_series: Dict[str, List[float]] = {}

        # CLOB metrics
        self.clob_qspr: List[float] = []
        self.clob_depth: List[dict] = []
        # {Q: [cost_bps per iter]}
        self.clob_cost_curves: Dict[float, List[float]] = {q: [] for q in self.Q_grid}
        self.clob_espr_curves: Dict[float, List[float]] = {q: [] for q in self.Q_grid}
        self.clob_impact_curves: Dict[float, List[float]] = {q: [] for q in self.Q_grid}

        # AMM metrics — {pool_name: {Q: [cost_bps]}}
        self.amm_cost_curves: Dict[str, Dict[float, List[float]]] = {}
        self.amm_slippage_curves: Dict[str, Dict[float, List[float]]] = {}
        self.amm_fee_curves: Dict[str, Dict[float, List[float]]] = {}

        # AMM reserves
        self.amm_x_series: Dict[str, List[float]] = {}
        self.amm_y_series: Dict[str, List[float]] = {}
        self.amm_L_series: Dict[str, List[float]] = {}

        # Volume-slippage profiles for AMM
        # {pool_name: {threshold: [max_Q per iter]}}
        self.amm_vol_slip: Dict[str, Dict[float, List[float]]] = {}

        # Flow allocation: {venue: [volume per iter]}
        self.flow_volume: Dict[str, List[float]] = {}
        self.flow_count: Dict[str, List[int]] = {}

        # Individual trade log
        self.trade_log: List[dict] = []

        # θ-bin metrics:  {bucket: {venue: [avg_cost per iter]}}
        self.bin_costs: Dict[str, Dict[str, List[float]]] = {
            b: {} for b in SIZE_BUCKETS
        }
        # {bucket: {venue: [volume per iter]}}
        self.bin_flow: Dict[str, Dict[str, List[float]]] = {
            b: {} for b in SIZE_BUCKETS
        }

    # ---- initialization --------------------------------------------------

    def _ensure_pool(self, name: str):
        """Lazily initialize storage for a new pool name."""
        if name not in self.amm_mid_series:
            self.amm_mid_series[name] = []
            self.amm_cost_curves[name] = {q: [] for q in self.Q_grid}
            self.amm_slippage_curves[name] = {q: [] for q in self.Q_grid}
            self.amm_fee_curves[name] = {q: [] for q in self.Q_grid}
            self.amm_x_series[name] = []
            self.amm_y_series[name] = []
            self.amm_L_series[name] = []
            self.amm_vol_slip[name] = {th: [] for th in self.slippage_thresholds}

    def _ensure_venue(self, name: str):
        if name not in self.flow_volume:
            self.flow_volume[name] = []
            self.flow_count[name] = []

    # ---- main snapshot ---------------------------------------------------

    def snapshot(self, t: int,
                 clob: 'CLOBVenue',
                 amm_pools: Dict[str, 'CPMMPool | HFMMPool'],
                 env: 'MarketEnvironment',
                 period_trades: List[dict]):
        """
        Record all metrics for iteration *t*.
        """
        self.iterations.append(t)

        # Environment
        self.sigma_series.append(env.sigma)
        self.c_series.append(env.funding_cost)
        self.regime_series.append('stress' if env.is_stress() else 'normal')

        # CLOB basics
        try:
            clob_mid = clob.mid_price()
        except Exception:
            clob_mid = self.clob_mid_series[-1] if self.clob_mid_series else 100.0
        self.clob_mid_series.append(clob_mid)
        self.clob_qspr.append(clob.quoted_spread_bps())
        self.clob_depth.append(clob.total_depth())

        # CLOB cost curves
        for Q in self.Q_grid:
            q_info = clob.quote_buy(Q)
            self.clob_cost_curves[Q].append(q_info['cost_bps'])
            self.clob_espr_curves[Q].append(q_info.get('espr_bps', 0))
            self.clob_impact_curves[Q].append(q_info.get('impact_bps', 0))

        # AMM metrics
        S_t = clob_mid
        for name, pool in amm_pools.items():
            self._ensure_pool(name)
            self.amm_mid_series[name].append(pool.mid_price())
            self.amm_x_series[name].append(pool.x)
            self.amm_y_series[name].append(pool.y)
            self.amm_L_series[name].append(pool.liquidity_measure())

            for Q in self.Q_grid:
                q_info = pool.quote_buy(Q, S_t)
                self.amm_cost_curves[name][Q].append(q_info['cost_bps'])
                self.amm_slippage_curves[name][Q].append(q_info['slippage_bps'])
                self.amm_fee_curves[name][Q].append(q_info['fee_bps'])

            for th in self.slippage_thresholds:
                max_q = pool.volume_slippage_max_Q(th, S_t)
                self.amm_vol_slip[name][th].append(max_q)

        # Flow allocation
        venue_vol: Dict[str, float] = {}
        venue_cnt: Dict[str, int] = {}
        for tr in period_trades:
            v = tr.get('venue', 'clob')
            venue_vol[v] = venue_vol.get(v, 0) + tr.get('quantity', 0)
            venue_cnt[v] = venue_cnt.get(v, 0) + 1

        all_venues = set(list(amm_pools.keys()) + ['clob'])
        for v in all_venues:
            self._ensure_venue(v)
            self.flow_volume[v].append(venue_vol.get(v, 0))
            self.flow_count[v].append(venue_cnt.get(v, 0))

        # Add trades to log
        for tr in period_trades:
            tr['t'] = t
            self.trade_log.append(tr)

        # ---- θ-bin aggregation -------------------------------------------
        bin_cost_acc: Dict[str, Dict[str, List[float]]] = {
            b: {} for b in SIZE_BUCKETS
        }
        bin_vol_acc: Dict[str, Dict[str, float]] = {
            b: {} for b in SIZE_BUCKETS
        }
        for tr in period_trades:
            bucket = tr.get('size_bucket', 'medium')
            venue = tr.get('venue', 'clob')
            cost = tr.get('cost_bps', 0)
            qty = tr.get('quantity', 0)
            if bucket not in bin_cost_acc:
                continue
            bin_cost_acc[bucket].setdefault(venue, []).append(cost)
            bin_vol_acc[bucket][venue] = bin_vol_acc[bucket].get(venue, 0) + qty

        all_venues_set = set(list(amm_pools.keys()) + ['clob'])
        for b in SIZE_BUCKETS:
            for v in all_venues_set:
                if v not in self.bin_costs[b]:
                    self.bin_costs[b][v] = []
                if v not in self.bin_flow[b]:
                    self.bin_flow[b][v] = []
                costs_list = bin_cost_acc[b].get(v, [])
                avg_c = (sum(costs_list) / len(costs_list)) if costs_list else float('nan')
                self.bin_costs[b][v].append(avg_c)
                self.bin_flow[b][v].append(bin_vol_acc[b].get(v, 0))

    # ---- analysis helpers ------------------------------------------------

    def cost_series(self, venue: str, Q: float) -> List[float]:
        """Time series of all-in cost for *venue* at size *Q*."""
        if venue == 'clob':
            return self.clob_cost_curves.get(Q, [])
        return self.amm_cost_curves.get(venue, {}).get(Q, [])

    def flow_share(self, venue: str) -> List[float]:
        """Fraction of total volume routed to *venue* each period."""
        if venue not in self.flow_volume:
            return []
        total = []
        for i in range(len(self.iterations)):
            t_vol = sum(self.flow_volume[v][i]
                        for v in self.flow_volume if i < len(self.flow_volume[v]))
            if t_vol > 0:
                total.append(self.flow_volume[venue][i] / t_vol)
            else:
                total.append(0.0)
        return total

    def cost_correlation(self, v1: str, v2: str, Q: float) -> float:
        """Pearson correlation of cost series between two venues at size Q."""
        s1 = self.cost_series(v1, Q)
        s2 = self.cost_series(v2, Q)
        n = min(len(s1), len(s2))
        if n < 3:
            return 0.0

        s1, s2 = s1[:n], s2[:n]
        # Filter infs
        pairs = [(a, b) for a, b in zip(s1, s2)
                 if math.isfinite(a) and math.isfinite(b)]
        if len(pairs) < 3:
            return 0.0

        xs, ys = zip(*pairs)
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
        sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
        sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
        if sx == 0 or sy == 0:
            return 0.0
        return cov / (sx * sy)

    def commonality_before_after(self, v1: str, v2: str, Q: float,
                                 stress_start: int) -> dict:
        """
        Compare cost correlation before vs after stress onset.
        """
        s1 = self.cost_series(v1, Q)
        s2 = self.cost_series(v2, Q)
        n = min(len(s1), len(s2))
        idx = min(stress_start, n)

        def _corr(a, b):
            pairs = [(x, y) for x, y in zip(a, b)
                     if math.isfinite(x) and math.isfinite(y)]
            if len(pairs) < 3:
                return float('nan')
            xs, ys = zip(*pairs)
            n_ = len(xs)
            mx = sum(xs) / n_
            my = sum(ys) / n_
            cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n_
            sx = (sum((x - mx) ** 2 for x in xs) / n_) ** 0.5
            sy = (sum((y - my) ** 2 for y in ys) / n_) ** 0.5
            if sx == 0 or sy == 0:
                return float('nan')
            return cov / (sx * sy)

        return {
            'before': _corr(s1[:idx], s2[:idx]),
            'after': _corr(s1[idx:], s2[idx:])
        }

    def summary(self) -> dict:
        """Return a concise summary dict for printing."""
        n = len(self.iterations)
        if n == 0:
            return {}

        result = {
            'n_iterations': n,
            'n_trades': len(self.trade_log),
        }

        # Average costs
        for v in list(self.amm_cost_curves.keys()) + ['clob']:
            for Q in self.Q_grid:
                s = self.cost_series(v, Q)
                finite = [x for x in s if math.isfinite(x)]
                if finite:
                    result[f'avg_cost_{v}_Q{Q}'] = sum(finite) / len(finite)

        # Flow shares
        for v in self.flow_volume:
            fs = self.flow_share(v)
            if fs:
                result[f'avg_flow_share_{v}'] = sum(fs) / len(fs)

        return result

    def bin_summary(self) -> Dict[str, dict]:
        """
        Average cost and total flow by θ-size bucket and venue.

        Returns
        -------
        {bucket: {venue: {'avg_cost_bps': float, 'total_volume': float}}}
        """
        out: Dict[str, dict] = {}
        for b in SIZE_BUCKETS:
            out[b] = {}
            for v in self.bin_costs.get(b, {}):
                costs = [c for c in self.bin_costs[b][v] if math.isfinite(c)]
                vols = self.bin_flow[b].get(v, [])
                out[b][v] = {
                    'avg_cost_bps': (sum(costs) / len(costs)) if costs else float('nan'),
                    'total_volume': sum(vols),
                }
        return out
