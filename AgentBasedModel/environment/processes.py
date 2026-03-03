"""
Exogenous market environment: volatility σ_t, funding cost c_t,
and fair price S_t.

Supports piecewise-constant regimes (normal → stress → normal)
or a simple mean-reverting stochastic process for σ, c.

S_t follows a Geometric Brownian Motion (GBM) driven by σ_t so that
there is always an authoritative fair price available — even when the
CLOB order book is thin or absent (AMM-only scenarios).
"""

import random
import math
from typing import Optional, List


class MarketEnvironment:
    """
    Manages exogenous volatility (σ_t), funding liquidity cost (c_t),
    and fair price S_t.

    Two modes for σ / c:
    1. **Piecewise-constant** (default): σ and c jump between predefined
       normal / stress levels at specified times.
    2. **Stochastic**: Ornstein–Uhlenbeck around a base level, with
       optional regime-shift overlay.

    Fair price S_t
    --------------
    Always present.  Follows GBM:

        S_{t+1} = S_t · exp((μ − ½σ²)Δt + σ √Δt · ε)

    where σ = σ_t (the current regime volatility), μ = ``drift``,
    Δt = 1.  The resulting path is the *official* reference price
    used by arbitrageurs when the CLOB is unreliable or absent.

    These parameters feed into:
      - CLOB market-maker spread:  qspr_t = α₀ + α₁ σ_t + α₂ c_t
      - AMM LP liquidity rule:     L_{t+1} = L_t + φ₁ Π^fee − φ₂ σ − φ₃ c
      - Arbitrageur target:        arb → S_t
    """

    def __init__(self,
                 sigma_low: float = 0.01,
                 sigma_high: float = 0.05,
                 c_low: float = 0.001,
                 c_high: float = 0.02,
                 stress_start: Optional[int] = None,
                 stress_end: Optional[int] = None,
                 mode: str = 'piecewise',
                 # ── exogenous fair price ──────────────────────
                 price: Optional[float] = None,
                 drift: float = 0.0):
        """
        Parameters
        ----------
        sigma_low, sigma_high : float
            Volatility in normal / stress regimes.
        c_low, c_high : float
            Funding cost in normal / stress regimes.
        stress_start, stress_end : int or None
            Iteration range [start, end) during which the market is stressed.
            If None the market stays in normal regime.
        mode : str
            'piecewise' for regime-switch, 'stochastic' for OU-based.
        price : float or None
            Initial fair price S_0.  If None the fair price is not tracked
            (backward-compatible with legacy callers).
        drift : float
            Annualised drift μ for the GBM (default 0 = martingale).
        """
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.c_low = c_low
        self.c_high = c_high
        self.stress_start = stress_start
        self.stress_end = stress_end
        self.mode = mode

        # Current values
        self._sigma = sigma_low
        self._c = c_low
        self._t = 0

        # ── Fair price (GBM) ─────────────────────────────────────────
        self._price: Optional[float] = float(price) if price is not None else None
        self._drift = drift

        # History
        self.sigma_history: List[float] = []
        self.c_history: List[float] = []
        self.regime_history: List[str] = []
        self.price_history: List[float] = []  # S_t series

    # ---- public API ------------------------------------------------------

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def funding_cost(self) -> float:
        return self._c

    @property
    def fair_price(self) -> Optional[float]:
        """Exogenous fair price S_t (None if not configured)."""
        return self._price

    def is_stress(self) -> bool:
        if self.stress_start is None:
            return False
        if self.stress_end is None:
            return self._t >= self.stress_start
        return self.stress_start <= self._t < self.stress_end

    def step(self):
        """Advance to next period and update σ_t, c_t, S_t."""
        self._t += 1

        if self.mode == 'piecewise':
            self._step_piecewise()
        else:
            self._step_stochastic()

        # GBM step for fair price
        if self._price is not None:
            self._step_price()

        self.sigma_history.append(self._sigma)
        self.c_history.append(self._c)
        self.regime_history.append('stress' if self.is_stress() else 'normal')
        if self._price is not None:
            self.price_history.append(self._price)

    def apply_shock(self, pct: float):
        """Apply an instantaneous shock to the fair price.

        Parameters
        ----------
        pct : float
            Percentage change (e.g. -20 for a 20 % crash).
        """
        if self._price is not None:
            self._price *= (1.0 + pct / 100.0)
            self._price = max(self._price, 1e-6)

    # ---- private ---------------------------------------------------------

    def _step_piecewise(self):
        if self.is_stress():
            self._sigma = self.sigma_high
            self._c = self.c_high
        else:
            self._sigma = self.sigma_low
            self._c = self.c_low

    def _step_stochastic(self):
        """
        OU-like: σ_{t+1} = σ_t + κ_σ (μ_σ − σ_t) + η_σ ε
        with regime-dependent mean μ_σ.
        """
        kappa_s, kappa_c = 0.1, 0.1
        eta_s, eta_c = 0.002, 0.001

        mu_s = self.sigma_high if self.is_stress() else self.sigma_low
        mu_c = self.c_high if self.is_stress() else self.c_low

        self._sigma += kappa_s * (mu_s - self._sigma) + eta_s * random.gauss(0, 1)
        self._c += kappa_c * (mu_c - self._c) + eta_c * random.gauss(0, 1)

        self._sigma = max(self._sigma, 1e-6)
        self._c = max(self._c, 0.0)

    def _step_price(self):
        """
        GBM step: S_{t+1} = S_t · exp((μ − ½σ²) + σ ε), Δt = 1.
        """
        sigma = self._sigma
        mu = self._drift
        eps = random.gauss(0, 1)
        log_ret = (mu - 0.5 * sigma * sigma) + sigma * eps
        self._price *= math.exp(log_ret)
        self._price = max(self._price, 1e-6)
