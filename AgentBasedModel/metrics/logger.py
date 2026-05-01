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
import numpy as np

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
        self.session_series: List[str] = []
        self.systemic_liquidity_series: List[float] = []
        self.flow_imbalance_series: List[float] = []
        self.clob_flow_imbalance_series: List[float] = []

        # Prices
        self.clob_mid_series: List[float] = []
        self.fair_price_series: List[float] = []   # exogenous S_t (GBM)
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
        self.amm_depth_series: Dict[str, List[float]] = {}  # effective_depth (base units)

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

        self.mm_states = ('active', 'defensive', 'withdrawn', 'reentering')
        self.mm_state_counts: Dict[str, List[float]] = {state: [] for state in self.mm_states}
        self.mm_state_shares: Dict[str, List[float]] = {state: [] for state in self.mm_states}
        self.n_market_makers_series: List[float] = []
        self.mm_avg_withdrawal_score_series: List[float] = []
        self.mm_avg_loss_bps_series: List[float] = []
        self.mm_avg_inventory_ratio_series: List[float] = []

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
            self.amm_depth_series[name] = []
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
                 period_trades: List[dict],
                 mm_state_summary: Optional[dict] = None):
        """
        Record all metrics for iteration *t*.
        """
        self.iterations.append(t)

        # Environment
        self.sigma_series.append(env.sigma)
        self.c_series.append(env.funding_cost)
        self.regime_series.append('stress' if env.is_stress() else 'normal')
        self.session_series.append(getattr(env, 'session_name', 'Unknown'))
        self.systemic_liquidity_series.append(getattr(env, 'systemic_liquidity', 1.0))
        self.flow_imbalance_series.append(
            getattr(env, 'logged_order_flow_imbalance',
                    getattr(env, 'order_flow_imbalance', 0.0))
        )
        self.clob_flow_imbalance_series.append(
            getattr(env, 'logged_clob_order_flow_imbalance',
                getattr(env, 'clob_order_flow_imbalance',
                    getattr(env, 'logged_order_flow_imbalance',
                        getattr(env, 'order_flow_imbalance', 0.0))))
        )

        # CLOB basics
        try:
            clob_mid = clob.mid_price()
        except Exception:
            clob_mid = self.clob_mid_series[-1] if self.clob_mid_series else 100.0
        self.clob_mid_series.append(clob_mid)
        self.clob_qspr.append(clob.quoted_spread_bps())
        self.clob_depth.append(clob.total_depth(bps_from_mid=25))

        # Exogenous fair price (GBM) — authoritative when available
        fp = getattr(env, 'fair_price', None)
        self.fair_price_series.append(fp if fp is not None else clob_mid)

        # CLOB cost curves
        for Q in self.Q_grid:
            q_info = clob.quote_buy(Q)
            self.clob_cost_curves[Q].append(q_info['cost_bps'])
            self.clob_espr_curves[Q].append(q_info.get('espr_bps', 0))
            self.clob_impact_curves[Q].append(q_info.get('impact_bps', 0))

        # AMM metrics — internal TCA vs each pool's own marginal mid
        for name, pool in amm_pools.items():
            self._ensure_pool(name)
            self.amm_mid_series[name].append(pool.mid_price())
            self.amm_x_series[name].append(pool.x)
            self.amm_y_series[name].append(pool.y)
            self.amm_L_series[name].append(pool.liquidity_measure())
            self.amm_depth_series[name].append(pool.effective_depth())

            for Q in self.Q_grid:
                q_info = pool.quote_buy(Q)
                self.amm_cost_curves[name][Q].append(q_info['cost_bps'])
                self.amm_slippage_curves[name][Q].append(q_info['slippage_bps'])
                self.amm_fee_curves[name][Q].append(q_info['fee_bps'])

            for th in self.slippage_thresholds:
                max_q = pool.volume_slippage_max_Q(th)
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

        mm_state_summary = mm_state_summary or {}
        counts = mm_state_summary.get('counts', {})
        shares = mm_state_summary.get('shares', {})
        n_market_makers = float(mm_state_summary.get('n_market_makers', 0))
        self.n_market_makers_series.append(n_market_makers)
        for state in self.mm_states:
            self.mm_state_counts[state].append(float(counts.get(state, 0.0)))
            self.mm_state_shares[state].append(float(shares.get(state, 0.0)))
        self.mm_avg_withdrawal_score_series.append(
            float(mm_state_summary.get('avg_withdrawal_score', float('nan')))
        )
        self.mm_avg_loss_bps_series.append(
            float(mm_state_summary.get('avg_loss_bps_ewma', float('nan')))
        )
        self.mm_avg_inventory_ratio_series.append(
            float(mm_state_summary.get('avg_inventory_ratio', float('nan')))
        )

    # ---- analysis helpers ------------------------------------------------

    def cost_series(self, venue: str, Q: float) -> List[float]:
        """Time series of all-in cost for *venue* at size *Q*."""
        if venue == 'clob':
            return self.clob_cost_curves.get(Q, [])
        return self.amm_cost_curves.get(venue, {}).get(Q, [])

    def venue_basis_series(self, venue: str) -> List[float]:
        """AMM-vs-CLOB mid-price basis in bps."""
        if venue == 'clob' or venue not in self.amm_mid_series:
            return [0.0 for _ in self.clob_mid_series]

        basis = []
        for clob_mid, amm_mid in zip(self.clob_mid_series, self.amm_mid_series[venue]):
            if not (math.isfinite(clob_mid) and math.isfinite(amm_mid) and clob_mid > 0):
                basis.append(float('nan'))
                continue
            basis.append(10_000.0 * (amm_mid - clob_mid) / clob_mid)
        return basis

    def max_venue_basis_series(self) -> List[float]:
        """Max absolute AMM-vs-CLOB basis across all AMM venues."""
        if not self.amm_mid_series:
            return [0.0 for _ in self.clob_mid_series]

        out = []
        for idx, clob_mid in enumerate(self.clob_mid_series):
            vals = []
            for venue in self.amm_mid_series:
                if idx >= len(self.amm_mid_series[venue]):
                    continue
                amm_mid = self.amm_mid_series[venue][idx]
                if math.isfinite(clob_mid) and math.isfinite(amm_mid) and clob_mid > 0:
                    vals.append(abs(10_000.0 * (amm_mid - clob_mid) / clob_mid))
            out.append(max(vals) if vals else float('nan'))
        return out

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

    @staticmethod
    def _finite_correlation(series_1: List[float], series_2: List[float]) -> float:
        pairs = [(x, y) for x, y in zip(series_1, series_2)
                 if math.isfinite(x) and math.isfinite(y)]
        if len(pairs) < 3:
            return float('nan')

        xs, ys = zip(*pairs)
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
        sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
        sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
        if sx == 0 or sy == 0:
            return float('nan')
        return cov / (sx * sy)

    def series_correlation(self, series_1: List[float], series_2: List[float]) -> float:
        """Pearson correlation between two generic finite series."""
        n = min(len(series_1), len(series_2))
        if n < 3:
            return 0.0
        corr = self._finite_correlation(series_1[:n], series_2[:n])
        return corr if math.isfinite(corr) else 0.0

    def series_commonality_before_after(self, series_1: List[float], series_2: List[float],
                                        split_at: int) -> dict:
        """Correlation of two generic series before vs after *split_at*."""
        n = min(len(series_1), len(series_2))
        if n < 3:
            return {'before': float('nan'), 'after': float('nan')}

        idx = max(0, min(int(split_at), n))
        return {
            'before': self._finite_correlation(series_1[:idx], series_2[:idx]),
            'after': self._finite_correlation(series_1[idx:], series_2[idx:]),
        }

    def cost_correlation(self, v1: str, v2: str, Q: float) -> float:
        """Pearson correlation of cost series between two venues at size Q."""
        s1 = self.cost_series(v1, Q)
        s2 = self.cost_series(v2, Q)
        return self.series_correlation(s1, s2)

    def commonality_before_after(self, v1: str, v2: str, Q: float,
                                 stress_start: int) -> dict:
        """
        Compare cost correlation before vs after stress onset.
        """
        s1 = self.cost_series(v1, Q)
        s2 = self.cost_series(v2, Q)
        return self.series_commonality_before_after(s1, s2, stress_start)

    def clob_total_depth_series(self) -> List[float]:
        """CLOB total depth time series within the logged near-mid band."""
        out: List[float] = []
        for row in self.clob_depth:
            value = row.get('total', float('nan')) if isinstance(row, dict) else float('nan')
            out.append(float(value) if math.isfinite(value) else float('nan'))
        return out

    def clob_touch_depth_series(self) -> List[float]:
        """One-sided CLOB depth at the touch, proxied by min(best bid qty, best ask qty)."""
        out: List[float] = []
        for row in self.clob_depth:
            if not isinstance(row, dict):
                out.append(float('nan'))
                continue
            bid = float(row.get('bid', float('nan')))
            ask = float(row.get('ask', float('nan')))
            value = min(bid, ask) if math.isfinite(bid) and math.isfinite(ask) else float('nan')
            out.append(value)
        return out

    def amm_total_depth_series(self) -> List[float]:
        """Aggregate AMM effective depth across all pools per period."""
        n = len(self.iterations)
        if n == 0:
            return []
        if not self.amm_depth_series:
            return [0.0 for _ in range(n)]

        out: List[float] = []
        for idx in range(n):
            total = 0.0
            has_finite = False
            for series in self.amm_depth_series.values():
                if idx >= len(series):
                    continue
                value = series[idx]
                if math.isfinite(value):
                    total += float(value)
                    has_finite = True
            out.append(total if has_finite else float('nan'))
        return out

    @staticmethod
    def _log_diff(series: List[float], eps: float = 1e-9) -> List[float]:
        """Finite log-difference transform for level series."""
        if len(series) < 2:
            return []
        out: List[float] = []
        for prev, curr in zip(series[:-1], series[1:]):
            if not (math.isfinite(prev) and math.isfinite(curr)):
                out.append(float('nan'))
                continue
            out.append(math.log(max(curr, eps)) - math.log(max(prev, eps)))
        return out

    @staticmethod
    def _lag_regression(y: List[float], x: List[float], lag: int = 1) -> dict:
        """
        OLS with intercept: y_t = a + b x_{t-lag} + e_t.

        Returns beta/alpha plus both classic t-stat and non-parametric
        block-bootstrap confidence diagnostics.
        """

        def _bootstrap_beta_ci(xs_vals: List[float], ys_vals: List[float],
                               n_boot: int = 400, block_size: int = 8) -> dict:
            """Moving-block bootstrap for lag-regression beta without normality assumptions."""
            n_vals = len(xs_vals)
            if n_vals < 20:
                return {
                    'beta_boot_lo': float('nan'),
                    'beta_boot_hi': float('nan'),
                    'p_boot': float('nan'),
                    'boot_n': 0,
                }

            block = max(2, min(int(block_size), n_vals))
            n_blocks = int(math.ceil(n_vals / block))
            # Deterministic seed to keep runs reproducible.
            rng = np.random.default_rng(17_071 + n_vals * 31 + block * 7)

            betas: List[float] = []
            for _ in range(int(n_boot)):
                starts = rng.integers(0, n_vals, size=n_blocks)
                x_sample: List[float] = []
                y_sample: List[float] = []

                for start in starts:
                    for shift in range(block):
                        idx = int((start + shift) % n_vals)
                        x_sample.append(xs_vals[idx])
                        y_sample.append(ys_vals[idx])
                        if len(x_sample) >= n_vals:
                            break
                    if len(x_sample) >= n_vals:
                        break

                m_x = sum(x_sample) / n_vals
                m_y = sum(y_sample) / n_vals
                s_xx = sum((xi - m_x) ** 2 for xi in x_sample)
                if s_xx <= 0:
                    continue
                s_xy = sum((xi - m_x) * (yi - m_y) for xi, yi in zip(x_sample, y_sample))
                b_hat = s_xy / s_xx
                if math.isfinite(b_hat):
                    betas.append(float(b_hat))

            if len(betas) < max(50, int(n_boot) // 5):
                return {
                    'beta_boot_lo': float('nan'),
                    'beta_boot_hi': float('nan'),
                    'p_boot': float('nan'),
                    'boot_n': len(betas),
                }

            beta_arr = np.array(betas, dtype='float64')
            lo, hi = np.quantile(beta_arr, [0.025, 0.975])
            p_pos = float(np.mean(beta_arr >= 0.0))
            p_neg = float(np.mean(beta_arr <= 0.0))
            p_boot = min(1.0, 2.0 * min(p_pos, p_neg))

            return {
                'beta_boot_lo': float(lo),
                'beta_boot_hi': float(hi),
                'p_boot': float(p_boot),
                'boot_n': len(betas),
            }

        lag = max(1, int(lag))
        if len(y) <= lag or len(x) <= lag:
            return {
                'alpha': float('nan'),
                'beta': float('nan'),
                't_beta': float('nan'),
                'nobs': 0,
                'beta_boot_lo': float('nan'),
                'beta_boot_hi': float('nan'),
                'p_boot': float('nan'),
                'boot_n': 0,
            }

        ys = y[lag:]
        xs = x[:-lag]
        pairs = [(xi, yi) for xi, yi in zip(xs, ys) if math.isfinite(xi) and math.isfinite(yi)]
        n = len(pairs)
        if n < 3:
            return {
                'alpha': float('nan'),
                'beta': float('nan'),
                't_beta': float('nan'),
                'nobs': n,
                'beta_boot_lo': float('nan'),
                'beta_boot_hi': float('nan'),
                'p_boot': float('nan'),
                'boot_n': 0,
            }

        xs_f, ys_f = zip(*pairs)
        mx = sum(xs_f) / n
        my = sum(ys_f) / n
        sxx = sum((xi - mx) ** 2 for xi in xs_f)
        if sxx <= 0:
            return {
                'alpha': float('nan'),
                'beta': float('nan'),
                't_beta': float('nan'),
                'nobs': n,
                'beta_boot_lo': float('nan'),
                'beta_boot_hi': float('nan'),
                'p_boot': float('nan'),
                'boot_n': 0,
            }

        sxy = sum((xi - mx) * (yi - my) for xi, yi in pairs)
        beta = sxy / sxx
        alpha = my - beta * mx
        boot = _bootstrap_beta_ci(list(xs_f), list(ys_f))

        if n <= 2:
            return {
                'alpha': alpha,
                'beta': beta,
                't_beta': float('nan'),
                'nobs': n,
                **boot,
            }

        rss = sum((yi - (alpha + beta * xi)) ** 2 for xi, yi in pairs)
        sigma2 = rss / max(1, n - 2)
        se_beta = math.sqrt(sigma2 / sxx) if sxx > 0 else float('inf')
        t_beta = beta / se_beta if se_beta > 0 and math.isfinite(se_beta) else float('nan')
        return {
            'alpha': alpha,
            'beta': beta,
            't_beta': t_beta,
            'nobs': n,
            **boot,
        }

    def liquidity_spillover_metrics(self, lag: int = 1,
                                    split_at: Optional[int] = None) -> dict:
        """
        Directional spillovers between AMM and CLOB liquidity.

        Uses log changes of aggregate AMM depth and CLOB near-mid depth.
        """
        clob_depth = self.clob_total_depth_series()
        amm_depth = self.amm_total_depth_series()
        d_clob = self._log_diff(clob_depth)
        d_amm = self._log_diff(amm_depth)

        out = {
            'lag': max(1, int(lag)),
            'n_points': min(len(d_clob), len(d_amm)),
            'corr_dliq_clob_amm': self.series_correlation(d_clob, d_amm),
            'amm_to_clob': self._lag_regression(d_clob, d_amm, lag=lag),
            'clob_to_amm': self._lag_regression(d_amm, d_clob, lag=lag),
            'avg_flow_share_clob': float('nan'),
            'avg_flow_share_amm': float('nan'),
        }

        clob_share = self.flow_share('clob')
        if clob_share:
            finite_clob = [x for x in clob_share if math.isfinite(x)]
            if finite_clob:
                out['avg_flow_share_clob'] = sum(finite_clob) / len(finite_clob)
                out['avg_flow_share_amm'] = 1.0 - out['avg_flow_share_clob']

        if split_at is not None:
            idx = max(0, min(int(split_at) - 1, min(len(d_clob), len(d_amm))))
            out['before_after'] = {
                'before': {
                    'corr_dliq_clob_amm': self.series_correlation(d_clob[:idx], d_amm[:idx]),
                    'amm_to_clob': self._lag_regression(d_clob[:idx], d_amm[:idx], lag=lag),
                    'clob_to_amm': self._lag_regression(d_amm[:idx], d_clob[:idx], lag=lag),
                },
                'after': {
                    'corr_dliq_clob_amm': self.series_correlation(d_clob[idx:], d_amm[idx:]),
                    'amm_to_clob': self._lag_regression(d_clob[idx:], d_amm[idx:], lag=lag),
                    'clob_to_amm': self._lag_regression(d_amm[idx:], d_clob[idx:], lag=lag),
                },
            }

        return out

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

        for state in self.mm_states:
            shares = [x for x in self.mm_state_shares[state] if math.isfinite(x)]
            if shares:
                result[f'avg_mm_share_{state}'] = sum(shares) / len(shares)

        mm_scores = [x for x in self.mm_avg_withdrawal_score_series if math.isfinite(x)]
        if mm_scores:
            result['avg_mm_withdrawal_score'] = sum(mm_scores) / len(mm_scores)

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
