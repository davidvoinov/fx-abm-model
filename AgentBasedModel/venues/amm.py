"""
AMM Pool implementations for FX market simulation.

- CPMMPool: Constant Product Market Maker (Uniswap-like), invariant x*y = k
- HFMMPool: Hybrid Function Market Maker (Curve StableSwap-like),
  invariant 4A(x+y) + D = 4AD + D^3/(4xy)

All costs are reported in basis points (bps) relative to a reference mid-price S_t.
"""

import math as _math
from typing import Optional, List, Dict


# ---------------------------------------------------------------------------
# Trade size classification  (θ = Q / R)
# ---------------------------------------------------------------------------

def classify_trade_size(Q: float, pool, mid_price: float) -> dict:
    """
    Classify trade as small / medium / large relative to pool depth.

    θ = Q / R,  where R = effective pool depth in base units.

    Thresholds (consistent with FX & AMM literature):
        small:  θ ≤ 1%    — near-zero slippage on CPMM
        medium: 1% < θ ≤ 5%
        large:  θ > 5%     — material slippage

    Returns dict(theta, size_bucket, R).
    """
    R = pool.effective_depth(mid_price)
    theta = Q / R if R > 0 else float('inf')
    if theta <= 0.01:
        bucket = 'small'
    elif theta <= 0.05:
        bucket = 'medium'
    else:
        bucket = 'large'
    return dict(theta=theta, size_bucket=bucket, R=R)


# ---------------------------------------------------------------------------
# CPMM — Constant Product Market Maker
# ---------------------------------------------------------------------------

class CPMMPool:
    """
    Constant Product Market Maker: x * y = k

    Reserves:
        x  – base currency
        y  – quote currency
    Internal mid-price: y / x
    Pool fee: fraction f of the input amount (e.g. 0.003 = 30 bps).
    """

    def __init__(self, x: float, y: float, fee: float = 0.003,
                 gas_cost_bps: float = 0.0):
        assert x > 0 and y > 0, "Reserves must be positive"
        self.x = float(x)
        self.y = float(y)
        self.k = self.x * self.y
        self.fee = fee
        self.gas_cost_bps = gas_cost_bps

        # Accumulated fee revenue (in quote terms) for LP income tracking
        self.fee_revenue = 0.0
        self.period_fee_revenue = 0.0  # fee revenue in current period

        # History
        self.x_history: List[float] = [self.x]
        self.y_history: List[float] = [self.y]
        self.fee_revenue_history: List[float] = [0.0]

    # ---- prices ----------------------------------------------------------

    def mid_price(self) -> float:
        """Internal pool mid-price: quote per base."""
        return self.y / self.x

    # ---- quoting (read-only) ---------------------------------------------

    def quote_buy(self, Q: float, S_t: Optional[float] = None) -> dict:
        """
        Estimate cost of buying *Q* base currency (trader pays quote).

        Returns dict with:
            exec_price   – effective price paid (quote/base)
            delta_y      – total quote paid by trader
            slippage_bps – pure bonding-curve slippage vs S_t
            fee_bps      – 10 000 * f
            cost_bps     – all-in cost = slippage + fee + gas
        """
        if Q <= 0:
            return self._zero_quote(S_t)
        if Q >= self.x:
            return self._inf_quote()

        S_ref = S_t if S_t is not None else self.mid_price()

        x_new = self.x - Q
        y_new = self.k / x_new
        dy_eff = y_new - self.y          # quote absorbed by pool (after fee)
        dy = dy_eff / (1.0 - self.fee)   # trader pays this

        # Fee-free execution price (for slippage decomposition)
        p_exec_no_fee = dy_eff / Q
        p_exec = dy / Q

        slippage_bps = 10_000.0 * (p_exec_no_fee - S_ref) / S_ref
        fee_bps = 10_000.0 * self.fee
        cost_bps = slippage_bps + fee_bps + self.gas_cost_bps

        return dict(exec_price=p_exec, delta_y=dy,
                    slippage_bps=slippage_bps, fee_bps=fee_bps,
                    cost_bps=cost_bps)

    def quote_sell(self, Q: float, S_t: Optional[float] = None) -> dict:
        """
        Estimate cost of selling *Q* base currency (trader receives quote).
        """
        if Q <= 0:
            return self._zero_quote(S_t)

        S_ref = S_t if S_t is not None else self.mid_price()

        dx_eff = Q * (1.0 - self.fee)
        x_new = self.x + dx_eff
        y_new = self.k / x_new
        dy = self.y - y_new  # trader receives

        # Fee-free exec price
        x_new_nf = self.x + Q
        y_new_nf = self.k / x_new_nf
        dy_nf = self.y - y_new_nf
        p_exec_no_fee = dy_nf / Q

        p_exec = dy / Q
        slippage_bps = 10_000.0 * (S_ref - p_exec_no_fee) / S_ref
        fee_bps = 10_000.0 * self.fee
        cost_bps = slippage_bps + fee_bps + self.gas_cost_bps

        return dict(exec_price=p_exec, delta_y=dy,
                    slippage_bps=slippage_bps, fee_bps=fee_bps,
                    cost_bps=cost_bps)

    # ---- execution (mutates state) ---------------------------------------

    def execute_buy(self, Q: float) -> dict:
        """Execute buying Q base. Updates reserves & fee revenue."""
        quote = self.quote_buy(Q, self.mid_price())
        if quote['cost_bps'] == float('inf'):
            return quote

        fee_amount = quote['delta_y'] * self.fee
        self.fee_revenue += fee_amount
        self.period_fee_revenue += fee_amount

        self.y += quote['delta_y'] * (1.0 - self.fee)
        self.x -= Q
        # k is invariant by construction
        return quote

    def execute_sell(self, Q: float) -> dict:
        """Execute selling Q base. Updates reserves & fee revenue."""
        quote = self.quote_sell(Q, self.mid_price())

        fee_amount = Q * self.fee * self.mid_price()
        self.fee_revenue += fee_amount
        self.period_fee_revenue += fee_amount

        self.x += Q * (1.0 - self.fee)
        self.y -= quote['delta_y']
        # Update k (may drift slightly due to fees collected)
        self.k = self.x * self.y
        return quote

    # ---- liquidity management --------------------------------------------

    def liquidity_measure(self) -> float:
        """L = sqrt(k) — geometric mean of reserves."""
        return _math.sqrt(self.k)

    def effective_depth(self, mid_price: Optional[float] = None) -> float:
        """
        Effective pool depth in base units:  R = sqrt(x·y) / S_t.

        Represents how many base units the pool can absorb before
        significant price impact occurs.
        """
        S = mid_price if mid_price else self.mid_price()
        return _math.sqrt(self.x * self.y) / S if S > 0 else self.x

    def add_liquidity(self, fraction: float):
        """Add liquidity by scaling reserves up by *fraction*."""
        self.x *= (1.0 + fraction)
        self.y *= (1.0 + fraction)
        self.k = self.x * self.y

    def remove_liquidity(self, fraction: float):
        """Remove liquidity by scaling reserves down by *fraction*."""
        fraction = min(fraction, 0.5)
        self.x *= (1.0 - fraction)
        self.y *= (1.0 - fraction)
        self.x = max(self.x, 1e-6)
        self.y = max(self.y, 1e-6)
        self.k = self.x * self.y

    # ---- arbitrage -------------------------------------------------------

    def arbitrage_to_target(self, S_t: float) -> float:
        """
        Trade to align pool mid-price with external price S_t.
        Returns quantity of base traded (positive = bought base from pool).
        Only arbs if deviation exceeds fee band.
        """
        current = self.mid_price()
        deviation = abs(current - S_t) / S_t
        if deviation < self.fee * 2:
            return 0.0

        # Target reserves: y/x = S_t, x*y = k
        x_target = _math.sqrt(self.k / S_t)
        delta_x = x_target - self.x

        if delta_x < 0:
            # Pool has too much x → buy base from pool
            Q = min(abs(delta_x), self.x * 0.5)
            self.execute_buy(Q)
            return Q
        else:
            # Pool needs more x → sell base to pool
            Q = abs(delta_x)
            self.execute_sell(Q)
            return -Q

    # ---- volume–slippage profile -----------------------------------------

    def volume_slippage_max_Q(self, threshold_bps: float,
                              S_t: Optional[float] = None,
                              side: str = 'buy',
                              tol: float = 0.01) -> float:
        """
        Maximum Q such that all-in cost ≤ threshold_bps (binary search).
        """
        hi = self.x * 0.99 if side == 'buy' else self.y * 0.99 / self.mid_price()
        lo = 0.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if side == 'buy':
                c = self.quote_buy(mid, S_t)['cost_bps']
            else:
                c = self.quote_sell(mid, S_t)['cost_bps']
            if c <= threshold_bps:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return lo

    # ---- period management -----------------------------------------------

    def reset_period_fees(self):
        """Reset per-period fee accumulator."""
        self.period_fee_revenue = 0.0

    def record_state(self):
        """Snapshot current reserves & cumulative fees."""
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.fee_revenue_history.append(self.fee_revenue)

    # ---- helpers ---------------------------------------------------------

    def _zero_quote(self, S_t=None):
        p = S_t if S_t else self.mid_price()
        return dict(exec_price=p, delta_y=0, slippage_bps=0,
                    fee_bps=0, cost_bps=0)

    def _inf_quote(self):
        return dict(exec_price=float('inf'), delta_y=float('inf'),
                    slippage_bps=float('inf'),
                    fee_bps=10_000 * self.fee,
                    cost_bps=float('inf'))


# ---------------------------------------------------------------------------
# HFMM — Hybrid Function Market Maker  (StableSwap / Curve-like)
# ---------------------------------------------------------------------------

def _hfmm_get_D(x1: float, x2: float, A: float) -> float:
    """
    Solve for D in the 2-token StableSwap invariant (Newton's method):

        4A(x₁ + x₂) + D = 4AD + D³/(4 x₁ x₂)
    """
    S = x1 + x2
    if S == 0:
        return 0.0
    if x1 <= 0 or x2 <= 0:
        return 0.0

    Ann = 4.0 * A
    D = S  # initial guess

    for _ in range(512):
        P = 4.0 * x1 * x2
        if P < 1e-30:
            return S  # fallback
        D_P = D ** 3 / P
        D_prev = D
        denom = (Ann - 1.0) * D + 3.0 * D_P
        if abs(denom) < 1e-30:
            return D
        D = (Ann * S + 2.0 * D_P) * D / denom
        if D < 0:
            D = D_prev * 0.5  # safeguard
        if abs(D - D_prev) < max(1e-12, abs(D) * 1e-14):
            return D

    return D  # return best estimate instead of raising


def _hfmm_get_y(x_new: float, D: float, A: float) -> float:
    """
    Solve for y given new x, invariant D, and amplification A.

    Quadratic in y:
        4A y² + b y - c = 0
    where b = 4A x + D(1 − 4A),  c = D³/(4x).
    """
    Ann = 4.0 * A
    b = Ann * x_new + D * (1.0 - Ann)
    c = D ** 3 / (4.0 * x_new)

    discriminant = b * b + 4.0 * Ann * c
    if discriminant < 0:
        raise ValueError("_hfmm_get_y: negative discriminant")

    y = (-b + _math.sqrt(discriminant)) / (2.0 * Ann)
    return max(y, 0.0)


def _hfmm_mid_price(x: float, y: float, A: float, D: float) -> float:
    """
    Marginal price (quote per base) on the StableSwap curve:

        price = (4A + D³/(4 x² y)) / (4A + D³/(4 x y²))
    """
    Ann = 4.0 * A
    D3 = D ** 3
    dFdx = Ann + D3 / (4.0 * x * x * y)
    dFdy = Ann + D3 / (4.0 * x * y * y)
    return dFdx / dFdy


class HFMMPool:
    """
    Hybrid Function Market Maker (StableSwap / Curve v1).

    Invariant (2 tokens, operating on *normalised* reserves):
        4A(x_n + y_n) + D = 4A D + D³ / (4 x_n y_n)

    where x_n = x_base × rate,  y_n = y_quote.
    *rate* is the expected equilibrium price (quote per base) and ensures
    the two reserve dimensions are comparable so the StableSwap curve
    behaves properly.

    Parameter A controls curvature:
        A → ∞  ⟹  behaviour ≈ CSMM (constant sum, near parity)
        reserves far from balance ⟹  behaviour ≈ CPMM (constant product)
    """

    def __init__(self, x: float, y: float, A: float = 100.0,
                 fee: float = 0.001, gas_cost_bps: float = 0.0,
                 rate: Optional[float] = None):
        """
        Parameters
        ----------
        x : float       base reserves
        y : float       quote reserves
        A : float       amplification coefficient
        fee : float     pool fee (fraction)
        rate : float    equilibrium price (quote/base).  If None → y/x.
        """
        assert x > 0 and y > 0 and A > 0
        self.x = float(x)          # raw base reserves
        self.y = float(y)          # raw quote reserves
        self.A = A
        self.fee = fee
        self.gas_cost_bps = gas_cost_bps
        self.rate = rate if rate is not None else self.y / self.x

        # Normalised reserves for the invariant
        self._xn = self.x * self.rate
        self._yn = self.y
        self.D = _hfmm_get_D(self._xn, self._yn, self.A)

        self.fee_revenue = 0.0
        self.period_fee_revenue = 0.0

        self.x_history: List[float] = [self.x]
        self.y_history: List[float] = [self.y]
        self.D_history: List[float] = [self.D]
        self.fee_revenue_history: List[float] = [0.0]

    def _sync_norm(self):
        """Re-compute normalised reserves from raw."""
        self._xn = self.x * self.rate
        self._yn = self.y

    # ---- prices ----------------------------------------------------------

    def mid_price(self) -> float:
        """Quote per base, derived from the marginal rate on the curve."""
        mp_norm = _hfmm_mid_price(self._xn, self._yn, self.A, self.D)
        return mp_norm * self.rate

    # ---- quoting ---------------------------------------------------------

    def quote_buy(self, Q: float, S_t: Optional[float] = None) -> dict:
        """
        Cost of buying Q base (trader pays quote, receives base).
        """
        if Q <= 0:
            return self._zero_quote(S_t)
        if Q >= self.x:
            return self._inf_quote()

        S_ref = S_t if S_t is not None else self.mid_price()

        # Fee-free execution in normalised space
        xn_new = (self.x - Q) * self.rate
        yn_new = _hfmm_get_y(xn_new, self.D, self.A)
        dy_nf = yn_new - self._yn          # quote to pay (no fee)
        p_exec_no_fee = dy_nf / Q

        # With fee
        dy = dy_nf / (1.0 - self.fee)
        p_exec = dy / Q

        slippage_bps = 10_000.0 * (p_exec_no_fee - S_ref) / S_ref
        fee_bps = 10_000.0 * self.fee
        cost_bps = slippage_bps + fee_bps + self.gas_cost_bps

        return dict(exec_price=p_exec, delta_y=dy,
                    slippage_bps=slippage_bps, fee_bps=fee_bps,
                    cost_bps=cost_bps)

    def quote_sell(self, Q: float, S_t: Optional[float] = None) -> dict:
        """
        Cost of selling Q base (trader pays base, receives quote).
        """
        if Q <= 0:
            return self._zero_quote(S_t)

        S_ref = S_t if S_t is not None else self.mid_price()

        # Fee-free
        xn_new_nf = (self.x + Q) * self.rate
        yn_new_nf = _hfmm_get_y(xn_new_nf, self.D, self.A)
        dy_nf = self._yn - yn_new_nf
        p_exec_no_fee = dy_nf / Q

        # With fee
        xn_new = (self.x + Q * (1.0 - self.fee)) * self.rate
        yn_new = _hfmm_get_y(xn_new, self.D, self.A)
        dy = self._yn - yn_new
        p_exec = dy / Q

        slippage_bps = 10_000.0 * (S_ref - p_exec_no_fee) / S_ref
        fee_bps = 10_000.0 * self.fee
        cost_bps = slippage_bps + fee_bps + self.gas_cost_bps

        return dict(exec_price=p_exec, delta_y=dy,
                    slippage_bps=slippage_bps, fee_bps=fee_bps,
                    cost_bps=cost_bps)

    # ---- execution -------------------------------------------------------

    def execute_buy(self, Q: float) -> dict:
        quote = self.quote_buy(Q, self.mid_price())
        if quote['cost_bps'] == float('inf'):
            return quote

        fee_amount = quote['delta_y'] * self.fee
        self.fee_revenue += fee_amount
        self.period_fee_revenue += fee_amount

        self.y += quote['delta_y'] * (1.0 - self.fee)
        self.x -= Q
        self._sync_norm()
        return quote

    def execute_sell(self, Q: float) -> dict:
        quote = self.quote_sell(Q, self.mid_price())

        fee_amount = Q * self.fee * self.mid_price()
        self.fee_revenue += fee_amount
        self.period_fee_revenue += fee_amount

        self.x += Q * (1.0 - self.fee)
        self.y -= quote['delta_y']
        self._sync_norm()
        return quote

    # ---- liquidity management --------------------------------------------

    def liquidity_measure(self) -> float:
        """D — the StableSwap invariant parameter."""
        return self.D

    def effective_depth(self, mid_price: Optional[float] = None) -> float:
        """
        Effective pool depth in base units, analogous to CPMM.
        Uses normalised reserves: R = sqrt(x_n · y_n) / rate.
        """
        return _math.sqrt(self._xn * self._yn) / self.rate if self.rate > 0 else self.x

    def add_liquidity(self, fraction: float):
        """Scale reserves up proportionally; recompute D."""
        self.x *= (1.0 + fraction)
        self.y *= (1.0 + fraction)
        self._sync_norm()
        self.D = _hfmm_get_D(self._xn, self._yn, self.A)

    def remove_liquidity(self, fraction: float):
        """Scale reserves down proportionally; recompute D."""
        fraction = min(fraction, 0.5)  # never remove more than half
        self.x *= (1.0 - fraction)
        self.y *= (1.0 - fraction)
        self.x = max(self.x, 1e-6)
        self.y = max(self.y, 1e-6)
        self._sync_norm()
        self.D = _hfmm_get_D(self._xn, self._yn, self.A)

    # ---- arbitrage -------------------------------------------------------

    def arbitrage_to_target(self, S_t: float) -> float:
        """
        Binary-search for x_target such that mid_price ≈ S_t,
        then execute the implied trade.  Returns signed quantity traded.
        """
        current = self.mid_price()
        deviation = abs(current - S_t) / S_t
        if deviation < self.fee * 2:
            return 0.0

        # Binary search for x_target (raw base reserves)
        lo = self.x * 0.01
        hi = self.x * 5.0
        D = self.D
        for _ in range(200):
            xm = (lo + hi) / 2.0
            xm_n = xm * self.rate
            ym_n = _hfmm_get_y(xm_n, D, self.A)
            if ym_n <= 0:
                hi = xm
                continue
            mp = _hfmm_mid_price(xm_n, ym_n, self.A, D) * self.rate
            if mp > S_t:
                lo = xm  # need more x to lower price
            else:
                hi = xm
            if abs(mp - S_t) / S_t < 1e-8:
                break

        x_target = (lo + hi) / 2.0
        delta = x_target - self.x

        if delta < 0:
            Q = min(abs(delta), self.x * 0.5)
            self.execute_buy(Q)
            return Q
        else:
            Q = abs(delta)
            self.execute_sell(Q)
            return -Q

    # ---- volume–slippage profile -----------------------------------------

    def volume_slippage_max_Q(self, threshold_bps: float,
                              S_t: Optional[float] = None,
                              side: str = 'buy',
                              tol: float = 0.01) -> float:
        hi = self.x * 0.99 if side == 'buy' else self.y * 0.99 / max(self.mid_price(), 1e-9)
        lo = 0.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if side == 'buy':
                c = self.quote_buy(mid, S_t)['cost_bps']
            else:
                c = self.quote_sell(mid, S_t)['cost_bps']
            if c <= threshold_bps:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return lo

    # ---- period management -----------------------------------------------

    def reset_period_fees(self):
        self.period_fee_revenue = 0.0

    def record_state(self):
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.D_history.append(self.D)
        self.fee_revenue_history.append(self.fee_revenue)

    # ---- helpers ---------------------------------------------------------

    def _zero_quote(self, S_t=None):
        p = S_t if S_t else self.mid_price()
        return dict(exec_price=p, delta_y=0, slippage_bps=0,
                    fee_bps=0, cost_bps=0)

    def _inf_quote(self):
        return dict(exec_price=float('inf'), delta_y=float('inf'),
                    slippage_bps=float('inf'),
                    fee_bps=10_000 * self.fee,
                    cost_bps=float('inf'))
