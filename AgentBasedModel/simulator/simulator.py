from __future__ import annotations

from AgentBasedModel.agents import ExchangeAgent, Universalist, Chartist, Fundamentalist, MarketMaker, AMMProvider, AMMArbitrageur, Random
from AgentBasedModel.utils.math import mean, std, difference, rolling
import random
from typing import Optional, List, Dict, Any
from tqdm import tqdm


class Simulator:
    """
    Simulator is responsible for launching agents' actions and executing scenarios.

    Operates in two modes determined automatically:

    **Classic mode** (no AMM / CLOB wrapper supplied):
        Legacy stock-market ABM — same behaviour as before.

    **Multi-venue mode** (``clob``, ``amm_pools`` supplied):
        CLOB + AMM FX simulation with venue routing, LP agents,
        arbitrageurs, MetricsLogger, and MarketEnvironment.
    """
    def __init__(self,
                 exchange: ExchangeAgent = None,
                 traders: list = None,
                 events: list = None,
                 # ---- multi-venue extras (all optional) ----
                 clob: Any = None,
                 amm_pools: Dict[str, Any] = None,
                 env: Any = None,
                 fx_traders: List = None,
                 clob_noise: List = None,
                 market_maker: Any = None,
                 lp_providers: List[AMMProvider] = None,
                 arbitrageur: AMMArbitrageur = None,
                 logger: Any = None,
                 shock_iter: Optional[int] = None,
                 shock_pct: float = -20.0,
                 ):
        self.exchange = exchange
        self.events = [event.link(self) for event in events] if events else None
        self.traders = traders  # used in classic mode (+ SimulatorInfo)

        # Price shock (multi-venue mode)
        self.shock_iter = shock_iter
        self.shock_pct = shock_pct

        # Multi-venue
        self.clob = clob
        self.amm_pools = amm_pools or {}
        self.env = env
        self.fx_traders = fx_traders or []
        self.clob_noise = clob_noise or []
        self.mm = market_maker
        self.lp_providers = lp_providers or []
        self.arbitrageur = arbitrageur
        self.logger = logger

        # SimulatorInfo — only created when classic traders list is given
        if self.traders:
            self.info = SimulatorInfo(self.exchange, self.traders)
        else:
            self.info = None

    @property
    def multi_venue(self) -> bool:
        return self.clob is not None and (bool(self.amm_pools) or bool(self.fx_traders))

    # ------------------------------------------------------------------
    # Classic helpers
    # ------------------------------------------------------------------

    def _payments(self):
        for trader in self.traders:
            trader.cash += trader.assets * self.exchange.dividend()
            trader.cash += trader.cash * self.exchange.risk_free

    # ------------------------------------------------------------------
    # Unified simulate
    # ------------------------------------------------------------------

    def simulate(self, n_iter: int, silent: bool = False) -> Simulator:
        if self.multi_venue:
            return self._simulate_multi(n_iter, silent)
        return self._simulate_classic(n_iter, silent)

    # ---- classic single-venue loop --------------------------------------

    def _simulate_classic(self, n_iter: int, silent: bool) -> Simulator:
        for it in tqdm(range(n_iter), desc='Simulation', disable=silent):
            # Scenario events
            if self.events:
                for event in self.events:
                    event.call(it)

            # Capture info
            if self.info is not None:
                self.info.capture()

            # Behaviour changes (Universalist / Chartist)
            for trader in self.traders:
                if type(trader) == Universalist:
                    trader.change_strategy(self.info)
                elif type(trader) == Chartist:
                    trader.change_sentiment(self.info)

            # Call traders
            random.shuffle(self.traders)
            for trader in self.traders:
                trader.call()

            # Payments & dividends
            self._payments()
            self.exchange.generate_dividend()

        return self

    # ---- multi-venue loop -----------------------------------------------

    def _simulate_multi(self, n_iter: int, silent: bool) -> Simulator:
        from itertools import chain as _chain

        for t in tqdm(range(n_iter), desc='Simulation', disable=silent):

            # 0. Price shock
            if self.shock_iter is not None and t == self.shock_iter:
                # Apply to exogenous fair price
                if self.env is not None:
                    self.env.apply_shock(self.shock_pct)

                # Also shift CLOB orders (live book only)
                if self.exchange is not None:
                    try:
                        mid = self.exchange.price()
                    except Exception:
                        mid = 100.0
                    dp = mid * (self.shock_pct / 100.0)
                    for order in _chain(*self.exchange.order_book.values()):
                        order.price = round(order.price + dp, 2)
                    if self.mm is not None:
                        self.mm.call()
                    for tr in self.clob_noise[:3]:
                        tr.call()

            # 1. Environment step → σ_t, c_t
            if self.env is not None:
                self.env.step()

            # 2. CLOB market maker quotes
            if self.mm is not None:
                self.mm.call()

            # 3. CLOB noise traders (keep book alive)
            random.shuffle(self.clob_noise)
            for tr in self.clob_noise:
                tr.call()

            # 4. Dividends
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
            if self.arbitrageur is not None:
                self.arbitrageur.arbitrage()

            # 8. LP agents adjust AMM liquidity
            for lp in self.lp_providers:
                lp.update_liquidity()

            # 9. Record AMM state
            for pool in self.amm_pools.values():
                pool.record_state()

            # 10. Metrics snapshot
            if self.logger is not None:
                self.logger.snapshot(t, self.clob, self.amm_pools,
                                    self.env, period_trades)

        return self

    # ------------------------------------------------------------------
    # Convenience factory for multi-venue mode
    # ------------------------------------------------------------------

    @classmethod
    def default_fx(cls,
                   n_noise: int = 30,
                   n_fx_takers: int = 15,
                   n_fx_fund: int = 5,
                   n_retail: int = 10,
                   n_institutional: int = 3,
                   clob_std: float = 2.0,
                   clob_volume: int = 1000,
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
                   amm_share_pct: float = 25.0,
                   deterministic: bool = False,
                   # ── intra-AMM venue selection ────────────────
                   beta_amm: float = 0.05,
                   cpmm_bias_bps: float = 5.0,
                   cost_noise_std: float = 1.5,
                   # ── venue configuration ──────────────────────
                   enable_clob_mm: bool = True,
                   enable_amm: bool = True,
                   clob_liq: float = 1.0,
                   amm_liq: float = 1.0,
                   shock_iter: Optional[int] = None,
                   shock_pct: float = -20.0,
                   ) -> Simulator:
        """
        Build a ready-to-run multi-venue FX simulator.

        Venue configuration
        -------------------
        enable_clob_mm : bool
            If False, no MarketMaker quotes on the CLOB.
        enable_amm : bool
            If False, no AMM pools / LP / arbitrageur are created.
        clob_liq : float  (0 … ∞, default 1.0)
            Multiplier that scales CLOB-side liquidity:
            ``clob_volume``, ``n_noise``, and MM depth params ``d0``
            are all multiplied by this factor.
        amm_liq : float  (0 … ∞, default 1.0)
            Multiplier that scales AMM-side liquidity:
            ``cpmm_reserves`` and ``hfmm_reserves`` are multiplied
            by this factor.

        Flow allocation
        ---------------
        amm_share_pct : float (0–100, default 25)
            Target probability (%) of routing a trade to AMM vs CLOB.
            E.g. 25 means ~25% AMM, ~75% CLOB.

        Intra-AMM venue selection
        -------------------------
        beta_amm : float
            Logit sensitivity for CPMM vs HFMM choice (lower → more
            uniform; higher → cost-driven).  Default 0.3.
        cpmm_bias_bps : float
            Non-monetary utility discount subtracted from perceived
            CPMM cost (bps).  Models convenience / accessibility.
        cost_noise_std : float
            Std of Gaussian noise added to AMM cost estimates (bps).
            Models imperfect information about pool costs.

        Convenience shortcuts
        ---------------------
        * **CLOB-only**:  ``enable_amm=False``
        * **AMM-only**:   ``enable_clob_mm=False, clob_liq=0.1``
          In this mode a **ShadowCLOB** replaces the live book: the
          CLOB never depletes and arb/metrics use the exogenous
          GBM fair price S_t.
        * **70/30 CLOB-heavy**: ``clob_liq=1.4, amm_liq=0.6``
        """
        from AgentBasedModel.venues.amm import CPMMPool, HFMMPool
        from AgentBasedModel.venues.clob import CLOBVenue, ShadowCLOB
        from AgentBasedModel.environment.processes import MarketEnvironment
        from AgentBasedModel.metrics.logger import MetricsLogger

        # ── Decide CLOB mode ────────────────────────────────────────
        # "Shadow" mode: no live MM, thin CLOB → use ShadowCLOB for
        # metrics and arb reference, keep tiny live book for backward
        # compatibility with agents that still need exchange.price().
        shadow_clob = (enable_amm and not enable_clob_mm)

        # ── Apply liquidity scaling ──────────────────────────────────
        eff_clob_volume = max(10, int(clob_volume * clob_liq))
        eff_n_noise = max(1, int(n_noise * clob_liq))
        eff_d0 = 50.0 * clob_liq
        eff_cpmm_res = cpmm_reserves * amm_liq
        eff_hfmm_res = hfmm_reserves * amm_liq

        # Exchange (CLOB) — always present (reference price + classic agents)
        exchange = ExchangeAgent(price=price, std=clob_std,
                                 volume=eff_clob_volume)

        # Environment — always pass price so GBM S_t is available
        env = MarketEnvironment(
            sigma_low=sigma_low, sigma_high=sigma_high,
            c_low=c_low, c_high=c_high,
            stress_start=stress_start, stress_end=stress_end,
            mode='piecewise',
            price=price,
        )

        # CLOB wrapper: live or shadow
        if shadow_clob:
            clob = ShadowCLOB(env)
        else:
            clob = CLOBVenue(exchange)

        # ── AMM pools (optional) ────────────────────────────────────
        amm_pools: Dict[str, Any] = {}
        lp_providers: List = []
        arb = None

        if enable_amm:
            cpmm = CPMMPool(x=eff_cpmm_res,
                            y=eff_cpmm_res * price, fee=cpmm_fee)
            hfmm = HFMMPool(x=eff_hfmm_res,
                            y=eff_hfmm_res * price,
                            A=hfmm_A, fee=hfmm_fee, rate=price)
            amm_pools = {'cpmm': cpmm, 'hfmm': hfmm}
            lp_providers = [
                AMMProvider(cpmm, env, phi1=0.5, phi2=2.0, phi3=1.0),
                AMMProvider(hfmm, env, phi1=0.5, phi2=2.0, phi3=1.0),
            ]
            arb = AMMArbitrageur(clob, amm_pools, env=env)

        # ── CLOB noise traders (keep the book alive) ────────────────
        clob_noise = [Random(exchange, cash=1e4)
                      for _ in range(eff_n_noise)]

        # ── Market Maker (optional) ─────────────────────────────────
        mm = None
        if enable_clob_mm:
            mm = MarketMaker(exchange, cash=1e4, env=env,
                             alpha0=3.0, alpha1=300.0, alpha2=200.0,
                             d0=eff_d0, d1=500.0, d2=250.0, d_min=5.0)

        # ── Liquidity takers with venue routing ─────────────────────
        fx_traders: List = []
        for _ in range(n_fx_takers):
            fx_traders.append(Random(
                exchange, cash=1e4,
                clob=clob, amm_pools=amm_pools, env=env,
                amm_share_pct=amm_share_pct,
                deterministic_venue=deterministic,
                beta_amm=beta_amm, cpmm_bias_bps=cpmm_bias_bps,
                cost_noise_std=cost_noise_std,
                trade_prob=0.3, q_min=1, q_max=5, label='Noise',
            ))
        for _ in range(n_fx_fund):
            fx_traders.append(Fundamentalist(
                exchange, cash=1e4,
                clob=clob, amm_pools=amm_pools, env=env,
                amm_share_pct=amm_share_pct,
                deterministic_venue=deterministic,
                beta_amm=beta_amm, cpmm_bias_bps=cpmm_bias_bps,
                cost_noise_std=cost_noise_std,
                fundamental_rate=price, fx_gamma=5e-3, fx_q_max=10,
            ))
        for _ in range(n_retail):
            fx_traders.append(Random(
                exchange, cash=1e4,
                clob=clob, amm_pools=amm_pools, env=env,
                amm_share_pct=amm_share_pct,
                deterministic_venue=deterministic,
                beta_amm=beta_amm, cpmm_bias_bps=cpmm_bias_bps,
                cost_noise_std=cost_noise_std,
                trade_prob=0.4, q_min=1, q_max=2, label='Retail',
            ))
        for _ in range(n_institutional):
            fx_traders.append(Random(
                exchange, cash=5e4,
                clob=clob, amm_pools=amm_pools, env=env,
                amm_share_pct=amm_share_pct,
                deterministic_venue=deterministic,
                beta_amm=beta_amm, cpmm_bias_bps=cpmm_bias_bps,
                cost_noise_std=cost_noise_std,
                trade_prob=0.15, q_min=15, q_max=50, label='Institutional',
            ))

        # Logger
        logger = MetricsLogger(
            Q_grid=[1, 2, 5, 10, 20, 50],
            slippage_thresholds=[5, 10, 25, 50],
        )

        return cls(
            exchange=exchange,
            clob=clob,
            amm_pools=amm_pools,
            env=env,
            fx_traders=fx_traders,
            clob_noise=clob_noise,
            market_maker=mm,
            lp_providers=lp_providers,
            arbitrageur=arb,
            logger=logger,
            shock_iter=shock_iter,
            shock_pct=shock_pct,
        )

    @classmethod
    def default_fx_no_amm(cls, **kwargs) -> Simulator:
        """Shortcut: CLOB-only FX simulator (no AMM)."""
        kwargs.setdefault('enable_amm', False)
        return cls.default_fx(**kwargs)


class SimulatorInfo:
    """
    SimulatorInfo is responsible for capturing data during simulating
    """

    def __init__(self, exchange: ExchangeAgent = None, traders: list = None):
        self.exchange = exchange
        self.traders = {t.id: t for t in traders}

        # Market Statistics
        self.prices = list()  # price at the end of iteration
        self.spreads = list()  # bid-ask spreads
        self.dividends = list()  # dividend paid at each iteration
        self.orders = list()  # order book statistics

        # Agent statistics
        self.equities = list()  # agent: equity
        self.cash = list()  # agent: cash
        self.assets = list()  # agent: number of assets
        self.types = list()  # agent: current type
        self.sentiments = list()  # agent: current sentiment
        self.returns = [{tr_id: 0 for tr_id in self.traders.keys()}]  # agent: iteration return

        """
        # Market Statistics
        self.prices = list()  # price at the end of iteration
        self.spreads = list()  # bid-ask spreads
        self.spread_sizes = list()  # bid-ask spread sizes
        self.dividends = list()
        self.orders_quantities = list()  # list -> (bid, ask)
        self.orders_volumes = list()  # list -> (bid, ask) -> (sum, mean, q1, q3, std)
        self.orders_prices = list()  # list -> (bid, ask) -> (mean, q1, q3, std)

        # Agent Statistics
        self.equity = list()  # sum of equity of agents
        self.cash = list()  # sum of cash of agents
        self.assets_qty = list()  # sum of number of assets of agents
        self.assets_value = list()  # sum of value of assets of agents
        """

    def capture(self):
        """
        Method called at the end of each iteration to capture basic info on simulation.

        **Attributes:**

        *Market Statistics*

        - :class:`list[float]` **prices** --> stock prices on each iteration
        - :class:`list[dict]` **spreads** --> order book spreads on each iteration
        - :class:`list[float]` **dividends** --> dividend paid on each iteration
        - :class:`list[dict[dict]]` **orders** --> order book price, volume, quantity stats on each iteration

        *Traders Statistics*

        - :class:`list[dict]` **equities** --> each agent's equity on each iteration
        - :class:`list[dict]` **cash** --> each agent's cash on each iteration
        - :class:`list[dict]` **assets** --> each agent's number of stocks on each iteration
        - :class:`list[dict]` **types** --> each agent's type on each iteration
        """
        # Market Statistics
        self.prices.append(self.exchange.price())
        self.spreads.append((self.exchange.spread()))
        self.dividends.append(self.exchange.dividend())
        self.orders.append({
            'quantity': {'bid': len(self.exchange.order_book['bid']), 'ask': len(self.exchange.order_book['ask'])},
            # 'price mean': {
            #     'bid': mean([order.price for order in self.exchange.order_book['bid']]),
            #     'ask': mean([order.price for order in self.exchange.order_book['ask']])},
            # 'price std': {
            #     'bid': std([order.price for order in self.exchange.order_book['bid']]),
            #     'ask': std([order.price for order in self.exchange.order_book['ask']])},
            # 'volume sum': {
            #     'bid': sum([order.qty for order in self.exchange.order_book['bid']]),
            #     'ask': sum([order.qty for order in self.exchange.order_book['ask']])},
            # 'volume mean': {
            #     'bid': mean([order.qty for order in self.exchange.order_book['bid']]),
            #     'ask': mean([order.qty for order in self.exchange.order_book['ask']])},
            # 'volume std': {
            #     'bid': std([order.qty for order in self.exchange.order_book['bid']]),
            #     'ask': std([order.qty for order in self.exchange.order_book['ask']])}
        })

        # Trader Statistics
        self.equities.append({t_id: t.equity() for t_id, t in self.traders.items()})
        self.cash.append({t_id: t.cash for t_id, t in self.traders.items()})
        self.assets.append({t_id: t.assets for t_id, t in self.traders.items()})
        self.types.append({t_id: t.type for t_id, t in self.traders.items()})
        self.sentiments.append({t_id: t.sentiment for t_id, t in self.traders.items() if t.type == 'Chartist'})
        self.returns.append({tr_id: (self.equities[-1][tr_id] - self.equities[-2][tr_id]) / self.equities[-2][tr_id]
                             for tr_id in self.traders.keys()}) if len(self.equities) > 1 else None

    def fundamental_value(self, access: int = 1) -> list:
        divs = self.dividends.copy()
        n = len(divs)  # number of iterations
        divs.extend(self.exchange.dividend(access)[1:access])  # add not recorded future divs
        r = self.exchange.risk_free

        return [Fundamentalist.evaluate(divs[i:i+access], r) for i in range(n)]

    def stock_returns(self, roll: int = None) -> list or float:
        p = self.prices
        div = self.dividends
        r = [(p[i+1] - p[i]) / p[i] + div[i] / p[i] for i in range(len(p) - 1)]
        return rolling(r, roll) if roll else mean(r)

    def abnormal_returns(self, roll: int = None) -> list:
        rf = self.exchange.risk_free
        r = [r - rf for r in self.stock_returns()]
        return rolling(r, roll) if roll else r

    def return_volatility(self, window: int = None) -> list or float:
        if window is None:
            return std(self.stock_returns())
        n = len(self.stock_returns(1))
        return [std(self.stock_returns(1)[i:i+window]) for i in range(n - window)]

    def price_volatility(self, window: int = None) -> list or float:
        if window is None:
            return std(self.prices)
        return [std(self.prices[i:i+window]) for i in range(len(self.prices) - window)]

    def liquidity(self, roll: int = None) -> list or float:
        n = len(self.prices)
        spreads = [el['ask'] - el['bid'] for el in self.spreads]
        prices = self.prices
        liq = [spreads[i] / prices[i] for i in range(n)]
        return rolling(liq, roll) if roll else mean(liq)
