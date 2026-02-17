"""
Exogenous market environment: volatility σ_t and funding cost c_t.

Supports piecewise-constant regimes (normal → stress → normal)
or a simple mean-reverting stochastic process.
"""

import random
import math
from typing import Optional, List


class MarketEnvironment:
    """
    Manages exogenous volatility (σ_t) and funding liquidity cost (c_t).

    Two modes:
    1. **Piecewise-constant** (default): σ and c jump between predefined
       normal / stress levels at specified times.
    2. **Stochastic**: Ornstein–Uhlenbeck around a base level, with
       optional regime-shift overlay.

    These parameters feed into:
      - CLOB market-maker spread:  qspr_t = α₀ + α₁ σ_t + α₂ c_t
      - AMM LP liquidity rule:     L_{t+1} = L_t + φ₁ Π^fee − φ₂ σ − φ₃ c
    """

    def __init__(self,
                 sigma_low: float = 0.01,
                 sigma_high: float = 0.05,
                 c_low: float = 0.001,
                 c_high: float = 0.02,
                 stress_start: Optional[int] = None,
                 stress_end: Optional[int] = None,
                 mode: str = 'piecewise'):
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

        # History
        self.sigma_history: List[float] = []
        self.c_history: List[float] = []
        self.regime_history: List[str] = []

    # ---- public API ------------------------------------------------------

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def funding_cost(self) -> float:
        return self._c

    def is_stress(self) -> bool:
        if self.stress_start is None:
            return False
        if self.stress_end is None:
            return self._t >= self.stress_start
        return self.stress_start <= self._t < self.stress_end

    def step(self):
        """Advance to next period and update σ_t, c_t."""
        self._t += 1

        if self.mode == 'piecewise':
            self._step_piecewise()
        else:
            self._step_stochastic()

        self.sigma_history.append(self._sigma)
        self.c_history.append(self._c)
        self.regime_history.append('stress' if self.is_stress() else 'normal')

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
