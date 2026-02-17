"""
Multi-venue FX simulator.

Orchestrates:
    - CLOB (ExchangeAgent) + CLOBVenue wrapper
    - AMM pools (CPMM, HFMM)
    - FX agents with venue routing
    - Market makers (CLOB) with σ/c-dependent spread
    - LP agents managing AMM liquidity
    - Arbitrageurs aligning AMM ↔ CLOB prices
    - MarketEnvironment (σ_t, c_t)
    - MetricsLogger
"""

from __future__ import annotations

import random
from typing import Optional, List, Dict, Any
from tqdm import tqdm

from AgentBasedModel.agents.agents import (
    ExchangeAgent, Random,
    FXTrader, FXNoiseTaker, FXFundamentalist,
    FXRetailTaker, FXInstitutional,
    FXMarketMaker, AMMProvider, AMMArbitrageur,
)
from AgentBasedModel.venues.amm import CPMMPool, HFMMPool
from AgentBasedModel.venues.clob import CLOBVenue
from AgentBasedModel.environment.processes import MarketEnvironment
from AgentBasedModel.metrics.logger import MetricsLogger


class FXSimulator:
    """
    Multi-venue FX market simulator.

    Simulation loop per iteration:
        1.  Environment step → update σ_t, c_t
        2.  CLOB market maker updates quotes (spread ∝ σ, c)
        3.  CLOB noise traders act (maintain book)
        4.  FX liquidity takers route orders (venue selection)
        5.  Arbitrageurs align AMM prices to CLOB
        6.  LP agents adjust AMM liquidity
        7.  Metrics snapshot
    """

    def __init__(
        self,
        # CLOB
        exchange: ExchangeAgent | None = None,
        clob_fee_broker: float = 0.0,
        clob_fee_venue: float = 0.0,
        # AMM pools
        cpmm: CPMMPool | None = None,
        hfmm: HFMMPool | None = None,
        # Environment
        env: MarketEnvironment | None = None,
        # Agents
        fx_traders: list[FXTrader] | None = None,
        clob_noise: list | None = None,       # existing Random/Fundamentalist traders
        fx_market_maker: FXMarketMaker | None = None,
        lp_providers: list[AMMProvider] | None = None,
        arbitrageur: AMMArbitrageur | None = None,
        # Metrics
        logger: MetricsLogger | None = None,
    ):
        # ---- defaults ----
        self.exchange = exchange or ExchangeAgent(price=100, std=25, volume=1000)
        self.clob = CLOBVenue(self.exchange, clob_fee_broker, clob_fee_venue)

        # Build AMM dict
        self.amm_pools = {}  # type: Dict[str, Any]
        if cpmm is not None:
            self.amm_pools['cpmm'] = cpmm
        if hfmm is not None:
            self.amm_pools['hfmm'] = hfmm

        self.env = env or MarketEnvironment()

        self.fx_traders = fx_traders or []
        self.clob_noise = clob_noise or []
        self.fx_mm = fx_market_maker
        self.lp_providers = lp_providers or []
        self.arbitrageur = arbitrageur or AMMArbitrageur(self.clob, self.amm_pools)
        self.logger = logger or MetricsLogger()

    # ---- simulation ------------------------------------------------------

    def simulate(self, n_iter: int, silent: bool = False) -> 'FXSimulator':
        for t in tqdm(range(n_iter), desc='FX Simulation', disable=silent):

            # 1. Environment: update σ_t, c_t
            self.env.step()

            # 2. CLOB market maker quotes
            if self.fx_mm is not None:
                self.fx_mm.call()

            # 3. CLOB noise traders (keep book alive)
            random.shuffle(self.clob_noise)
            for tr in self.clob_noise:
                tr.call()

            # 4. Dividends / interest (existing ExchangeAgent mechanics)
            self.exchange.generate_dividend()

            # 5. Reset AMM period fees
            for pool in self.amm_pools.values():
                pool.reset_period_fees()

            # 6. FX liquidity takers route orders
            period_trades: List[dict] = []
            random.shuffle(self.fx_traders)
            for tr in self.fx_traders:
                result = tr.call()
                if result is not None:
                    period_trades.append(result)

            # 7. Arbitrageurs align AMM prices
            self.arbitrageur.arbitrage()

            # 8. LP agents adjust AMM liquidity
            for lp in self.lp_providers:
                lp.update_liquidity()

            # 9. Record AMM state
            for pool in self.amm_pools.values():
                pool.record_state()

            # 10. Metrics snapshot
            self.logger.snapshot(t, self.clob, self.amm_pools,
                                self.env, period_trades)

        return self

    # ---- convenience factory ---------------------------------------------

    @classmethod
    def default(cls,
                n_noise: int = 30,
                n_fx_takers: int = 15,
                n_fx_fund: int = 5,
                n_retail: int = 10,
                n_institutional: int = 3,
                clob_std: float = 2.0,
                clob_volume: int = 5000,
                cpmm_reserves: float = 500.0,
                hfmm_reserves: float = 500.0,
                hfmm_A: float = 100.0,
                cpmm_fee: float = 0.003,
                hfmm_fee: float = 0.001,
                stress_start: Optional[int] = 200,
                stress_end: Optional[int] = 350,
                sigma_low: float = 0.01,
                sigma_high: float = 0.05,
                c_low: float = 0.001,
                c_high: float = 0.02,
                price: float = 100.0,
                beta: float = 1.0,
                deterministic: bool = False,
                ) -> 'FXSimulator':
        """
        Build a ready-to-run simulator with sensible defaults.
        """
        # Exchange (CLOB) — deep, tight book mimicking institutional FX
        exchange = ExchangeAgent(price=price, std=clob_std, volume=clob_volume)

        # AMM pools
        cpmm = CPMMPool(x=cpmm_reserves, y=cpmm_reserves * price,
                        fee=cpmm_fee)
        hfmm = HFMMPool(x=hfmm_reserves, y=hfmm_reserves * price,
                        A=hfmm_A, fee=hfmm_fee, rate=price)

        # Wrap CLOB
        clob = CLOBVenue(exchange)

        # Environment
        env = MarketEnvironment(
            sigma_low=sigma_low, sigma_high=sigma_high,
            c_low=c_low, c_high=c_high,
            stress_start=stress_start, stress_end=stress_end,
            mode='piecewise',
        )

        amm_pools = {'cpmm': cpmm, 'hfmm': hfmm}

        # CLOB noise traders (keep the book alive)
        clob_noise = [Random(exchange, cash=1e4) for _ in range(n_noise)]

        # FX market maker for CLOB — high base depth for deep book
        fx_mm = FXMarketMaker(exchange, env,
                              alpha0=1.0, alpha1=50.0, alpha2=25.0,
                              d0=200.0, d1=500.0, d2=200.0, d_min=20.0)

        # FX liquidity takers (noise + fundamentalist)
        fx_traders: list[FXTrader] = []
        for _ in range(n_fx_takers):
            fx_traders.append(FXNoiseTaker(
                clob, amm_pools, env, beta=beta,
                deterministic=deterministic,
                trade_prob=0.3, q_min=1, q_max=5,
            ))
        for _ in range(n_fx_fund):
            fx_traders.append(FXFundamentalist(
                clob, amm_pools, env, beta=beta,
                deterministic=deterministic,
                fundamental_rate=price, gamma=5e-3, q_max=10,
            ))
        # Retail takers (small θ)
        for _ in range(n_retail):
            fx_traders.append(FXRetailTaker(
                clob, amm_pools, env, beta=beta,
                deterministic=deterministic,
                trade_prob=0.4, q_min=1, q_max=2,
            ))
        # Institutional takers (large θ)
        for _ in range(n_institutional):
            fx_traders.append(FXInstitutional(
                clob, amm_pools, env, beta=beta,
                deterministic=deterministic,
                trade_prob=0.15, q_min=15, q_max=50,
            ))

        # LP providers
        lp_cpmm = AMMProvider(cpmm, env, phi1=0.5, phi2=2.0, phi3=1.0)
        lp_hfmm = AMMProvider(hfmm, env, phi1=0.5, phi2=2.0, phi3=1.0)

        # Arbitrageur
        arb = AMMArbitrageur(clob, amm_pools)

        # Logger
        logger = MetricsLogger(
            Q_grid=[1, 2, 5, 10, 20, 50],
            slippage_thresholds=[5, 10, 25, 50],
        )

        return cls(
            exchange=exchange,
            cpmm=cpmm,
            hfmm=hfmm,
            env=env,
            fx_traders=fx_traders,
            clob_noise=clob_noise,
            fx_market_maker=fx_mm,
            lp_providers=[lp_cpmm, lp_hfmm],
            arbitrageur=arb,
            logger=logger,
        )
