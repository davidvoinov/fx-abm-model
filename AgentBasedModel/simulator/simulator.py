from __future__ import annotations

from AgentBasedModel.agents import ExchangeAgent, Universalist, Chartist, Fundamentalist, MarketMaker, AMMProvider, AMMArbitrageur, Random, FastRecyclerLP, LatentLP
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
                 book_agents: List = None,
                 market_maker: Any = None,
                 lp_providers: List[AMMProvider] = None,
                 arbitrageur: AMMArbitrageur = None,
                 logger: Any = None,
                 shock_iter: Optional[int] = None,
                 shock_pct: float = -20.0,
                 shock_mode: str = 'research',
                 realism_shock_config: Optional[Dict[str, Any]] = None,
                 ):
        self.exchange = exchange
        self.events = [event.link(self) for event in events] if events else None
        self.traders = traders  # used in classic mode (+ SimulatorInfo)

        # Price shock (multi-venue mode)
        self.shock_iter = shock_iter
        self.shock_pct = shock_pct
        self.shock_mode = shock_mode
        self.realism_shock_config = realism_shock_config or {}

        # Multi-venue
        self.clob = clob
        self.amm_pools = amm_pools or {}
        self.env = env
        self.fx_traders = fx_traders or []
        self.book_agents = book_agents or []
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

    def _estimate_recovery_support(self) -> float:
        fast_agents = [tr for tr in self.book_agents if isinstance(tr, FastRecyclerLP)]
        latent_agents = [tr for tr in self.book_agents if isinstance(tr, LatentLP)]

        fast_total = len(fast_agents)
        latent_total = len(latent_agents)
        fast_active = sum(1 for tr in fast_agents if getattr(tr, 'last_quoted', False))
        latent_active = sum(1 for tr in latent_agents if getattr(tr, 'last_quoted', False))
        mm_active = 1.0 if self.mm is not None and getattr(self.mm, 'orders', None) else 0.0

        static_support = 0.85 + 0.02 * fast_total + 0.03 * latent_total
        dynamic_support = 0.0
        if fast_total > 0:
            dynamic_support += 0.14 * (fast_active / fast_total)
        if latent_total > 0:
            dynamic_support += 0.18 * (latent_active / latent_total)
        dynamic_support += 0.05 * mm_active

        return max(0.75, min(1.65, static_support + dynamic_support))

    def _sync_recovery_support(self):
        if self.env is None or not hasattr(self.env, 'set_recovery_support'):
            return
        self.env.set_recovery_support(self._estimate_recovery_support())

    # ------------------------------------------------------------------
    # Classic helpers
    # ------------------------------------------------------------------

    def _payments(self):
        for trader in self.traders:
            trader.cash += trader.assets * self.exchange.dividend()
            trader.cash += trader.cash * self.exchange.risk_free

    @staticmethod
    def _rebalance_pool_to_multiplier(pool, multiplier: float):
        import math as _math
        from AgentBasedModel.venues.amm import _hfmm_get_D

        sqrt_m = _math.sqrt(max(multiplier, 1e-6))
        old_x = pool.x
        old_y = pool.y
        pool.x = old_x / sqrt_m
        pool.y = old_y * sqrt_m
        if hasattr(pool, 'k'):
            pool.k = pool.x * pool.y
        if hasattr(pool, '_sync_norm'):
            pool._sync_norm()
        if hasattr(pool, 'D') and hasattr(pool, '_xn'):
            pool.D = _hfmm_get_D(pool._xn, pool._yn, pool.A)
        if hasattr(pool, 'rate'):
            pool.rate = pool.y / pool.x

    def _apply_research_shock(self):
        from itertools import chain as _chain

        multiplier = 1.0 + self.shock_pct / 100.0

        if self.env is not None:
            self.env.apply_shock(self.shock_pct)

        if self.exchange is not None:
            try:
                mid = self.exchange.price()
            except Exception:
                mid = 100.0
            dp = mid * (self.shock_pct / 100.0)
            for order in _chain(*self.exchange.order_book.values()):
                order.price = round(order.price + dp, 2)
            self.exchange.rebuild_order_book()

            cancel_frac = self.env.cancel_wave_frac if self.env else 0.5
            self.exchange.cancel_wave(cancel_frac, near_touch=True)

            for tr in self.book_agents[:3]:
                try:
                    tr.call()
                except Exception:
                    pass

        for pool in self.amm_pools.values():
            self._rebalance_pool_to_multiplier(pool, multiplier)

    def _resolve_realism_shock_side(self, config: Dict[str, Any]) -> str:
        side = config.get('order_flow_side', 'auto')
        if side in ('buy', 'sell'):
            return side

        signed_move = config.get('fundamental_pct', 0.0)
        if signed_move == 0:
            signed_move = self.shock_pct
        return 'sell' if signed_move < 0 else 'buy'

    def _execute_order_flow_sweep(self, side: str, quantity: float):
        from AgentBasedModel.agents.agents import Trader
        from AgentBasedModel.utils.orders import Order

        if self.exchange is None or quantity <= 0:
            return
        if side == 'buy' and not self.exchange.order_book['ask']:
            return
        if side == 'sell' and not self.exchange.order_book['bid']:
            return

        pseudo_trader = Trader(self.exchange, cash=1e9, assets=int(1e6))
        if side == 'buy':
            price = self.exchange.order_book['ask'].last.price
            order = Order(price, round(quantity), 'bid', pseudo_trader)
        else:
            price = self.exchange.order_book['bid'].last.price
            order = Order(price, round(quantity), 'ask', pseudo_trader)
        self.exchange.market_order(order)

    def _apply_realism_shock(self):
        config = self.realism_shock_config or {}
        fundamental_pct = float(config.get('fundamental_pct', 0.0) or 0.0)
        order_flow_qty = float(config.get('order_flow_qty', 0.0) or 0.0)
        liquidity_frac = max(0.0, min(1.0, float(config.get('liquidity_frac', 0.0) or 0.0)))
        funding_vol_intensity = max(0.0, float(config.get('funding_vol_intensity', 0.0) or 0.0))
        order_flow_side = self._resolve_realism_shock_side(config)
        direction = 1.0 if order_flow_side == 'buy' else -1.0

        if self.env is not None:
            if abs(fundamental_pct) > 0:
                self.env.apply_fundamental_shock(fundamental_pct, anchor_weight=1.0)
            if funding_vol_intensity > 0:
                self.env.apply_funding_volatility_shock(funding_vol_intensity)

            flow_intensity = 0.0
            if order_flow_qty > 0 and self.exchange is not None:
                try:
                    near_mid_depth = self.clob.total_depth(25)['total']
                except Exception:
                    near_mid_depth = 0.0
                if near_mid_depth > 0:
                    flow_intensity = min(2.0, order_flow_qty / near_mid_depth)

            liquidity_intensity = max(liquidity_frac, flow_intensity)
            if liquidity_intensity > 0:
                cancel_frac = max(liquidity_frac, min(0.85, 0.10 + 0.25 * liquidity_intensity))
                # Optional slow-decay parameters from config
                decay_kwargs = {}
                for key in ('reprice_prob_recovery', 'anchor_strength_recovery',
                            'bg_target_ratio_recovery', 'toxic_flow_decay',
                            'liquidity_shock_decay'):
                    val = config.get(key)
                    if val is not None:
                        decay_kwargs[key] = float(val)
                self.env.apply_liquidity_shock(
                    cancel_frac=cancel_frac,
                    direction=direction,
                    intensity=liquidity_intensity,
                    **decay_kwargs,
                )

        if order_flow_qty > 0:
            self._execute_order_flow_sweep(order_flow_side, order_flow_qty)

        if self.exchange is not None:
            cancel_frac = self.env.cancel_wave_frac if self.env is not None else liquidity_frac
            if cancel_frac > 0:
                self.exchange.cancel_wave(cancel_frac, near_touch=True)
            self.exchange.rebuild_order_book()

            for tr in self.book_agents[:2]:
                try:
                    tr.call()
                except Exception:
                    pass

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
        for t in tqdm(range(n_iter), desc='Simulation', disable=silent):

            # 0. Price shock — hits ALL venues simultaneously
            if self.shock_iter is not None and t == self.shock_iter:
                if self.shock_mode == 'realism':
                    self._apply_realism_shock()
                else:
                    self._apply_research_shock()

            # 1. Environment step → σ_t, c_t, S_t
            _sigma_prev = self.env.sigma if self.env is not None else None
            if self.env is not None:
                self._sync_recovery_support()
                self.env.step()
                # Re-centre book on new S_t — use reduced probability
                # during shock aftermath for natural price discovery lag
                fp = self.env.fair_price
                if fp is not None:
                    rp = self.env.reprice_prob_override if self.env.reprice_prob_override is not None else 0.6
                    anchor_strength = None
                    background_target_ratio = self.env.systemic_liquidity
                    if self.shock_mode == 'realism' and self.env.shock_ticks_remaining > 0:
                        anchor_strength = self.env.anchor_strength_override
                        if self.env.background_target_ratio_override is not None:
                            background_target_ratio = self.env.background_target_ratio_override
                    self.exchange.recenter_book(
                        fp,
                        reprice_prob=rp,
                        anchor_strength=anchor_strength,
                        background_target_ratio=background_target_ratio,
                    )

            # 1a. Dynamic fees — scale AMM fees with lagged σ
            if self.env is not None:
                for pool in self.amm_pools.values():
                    if hasattr(pool, 'update_fee'):
                        pool.update_fee(self.env.sigma, self.env.sigma_low,
                                        sigma_prev=_sigma_prev)

            # 1b. Pre-trade arbitrage — align AMM prices *before* FX
            #     takers observe quotes.  This second pass (together with
            #     the post-trade pass in step 7) keeps pool mid-prices
            #     close to S_t even under high-σ GBM dynamics.
            skip_pretrade_arb = (
                self.shock_mode == 'realism'
                and self.shock_iter is not None
                and t == self.shock_iter
            )
            if self.arbitrageur is not None and not skip_pretrade_arb:
                self.arbitrageur.arbitrage()

            # 1c. SimulatorInfo capture + behaviour changes
            if self.info is not None:
                try:
                    self.info.capture()
                    for tr in self.book_agents:
                        if type(tr) == Universalist:
                            tr.change_strategy(self.info)
                        elif type(tr) == Chartist:
                            tr.change_sentiment(self.info)
                except Exception:
                    pass

            # 2. CLOB market maker quotes (paused during shock aftermath)
            if self.mm is not None:
                mm_paused = (self.env is not None and self.env.mm_pause_ticks > 0)
                if mm_paused:
                    # MM withdraws — cancel all quotes but don't requote
                    for order in self.mm.orders.copy():
                        try:
                            self.mm._cancel_order(order)
                        except Exception:
                            pass
                    self.mm.orders.clear()
                else:
                    self.mm.call()

            # 3. CLOB book agents (diverse limit-order providers)
            random.shuffle(self.book_agents)
            for tr in self.book_agents:
                try:
                    tr.call()
                except Exception:
                    pass

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

            if self.env is not None:
                self.env.observe_order_flow(period_trades)

            # 7. Arbitrageurs align AMM prices
            if self.arbitrageur is not None:
                self.arbitrageur.arbitrage()

            # 7a. Expire stale orders (TTL-based lifecycle)
            fp = self.env.fair_price if self.env is not None else None
            self.exchange.expire_orders(fair_price=fp)
            self._sync_recovery_support()

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
                   n_noise: int = 12,
                   n_mm: int = 2,
                   n_fast_lp: int = 10,
                   n_latent_lp: int = 6,
                   n_clob_fund: int = 2,
                   n_clob_chart: int = 0,
                   n_clob_univ: int = 1,
                   n_fx_takers: int = 15,
                   n_fx_fund: int = 5,
                   n_retail: int = 10,
                   n_institutional: int = 3,
                   clob_std: float = 2.0,
                   clob_volume: int = 1000,
                   cpmm_reserves: float = 1000.0,
                   hfmm_reserves: float = 1000.0,
                   hfmm_A: float = 10.0,
                   cpmm_fee: float = 0.003,
                   hfmm_fee: float = 0.001,
                   stress_start: Optional[int] = None,
                   stress_end: Optional[int] = None,
                   sigma_low: float = 0.01,
                   sigma_high: float = 0.05,
                   c_low: float = 0.001,
                   c_high: float = 0.02,
                   price: float = 100.0,
                   amm_share_pct: float = 30.0,
                   venue_choice_rule: str = 'fixed_share',
                   deterministic: bool = False,
                   # ── intra-AMM venue selection ────────────────
                   beta_amm: float = 0.05,
                   cpmm_bias_bps: float = 5.0,
                   cost_noise_std: float = 1.5,
                   # ── venue configuration ──────────────────────
                   enable_amm: bool = True,
                   clob_liq: float = 1.0,
                   clob_anchor_strength: float = 0.35,
                   clob_anchor_threshold_bps: float = 5.0,
                   clob_background_target_ratio: float = 1.5,
                   clob_amm_interaction: str = 'competition',
                   clob_amm_spread_impact_bps: float = 3.0,
                   clob_amm_depth_impact: float = 60.0,
                   amm_liq: float = 1.0,
                   match_initial_depth: bool = False,
                   shock_iter: Optional[int] = None,
                   shock_pct: float = -20.0,
                   shock_mode: str = 'research',
                   fundamental_shock_pct: float = 0.0,
                   order_flow_shock_qty: float = 0.0,
                   order_flow_shock_side: str = 'auto',
                   liquidity_shock_frac: float = 0.0,
                   funding_vol_shock_intensity: float = 0.0,
                   arb_max_correction_pct: float = 10.0,
                   arb_trade_fraction_cap: float = 0.20,
                   dynamic_fee: bool = False,
                   # ── liquidity shock decay rates ──────────────
                   reprice_prob_recovery: Optional[float] = None,
                   anchor_strength_recovery: Optional[float] = None,
                   bg_target_ratio_recovery: Optional[float] = None,
                   toxic_flow_decay: Optional[float] = None,
                   liquidity_shock_decay: Optional[float] = None,
                   ) -> Simulator:
        """
        Build a ready-to-run multi-venue FX simulator.

        Venue configuration
        -------------------
        n_mm : int
            Number of Market Makers on the CLOB (0 = no MM).
        enable_amm : bool
            If False, no AMM pools / LP / arbitrageur are created.
        clob_liq : float  (0 … ∞, default 1.0)
            Multiplier that scales CLOB-side liquidity:
            ``clob_volume``, ``n_noise``, ``n_fast_lp``, ``n_latent_lp``,
            and MM depth params ``d0`` are all multiplied by this factor.
        amm_liq : float  (0 … ∞, default 1.0)
            Multiplier that scales AMM-side liquidity:
            ``cpmm_reserves`` and ``hfmm_reserves`` are multiplied
            by this factor.
        match_initial_depth : bool, default False
            Optional calibration aid. If enabled, rebalance the live CLOB
            at t=0 so its near-mid depth is comparable to aggregate AMM
            effective depth. Disabled by default to avoid imposing a
            like-for-like market structure before trading starts.

        Flow allocation
        ---------------
        amm_share_pct : float (0–100, default 25)
            Target probability (%) of routing a trade to AMM vs CLOB.
            E.g. 25 means ~25% AMM, ~75% CLOB.
        venue_choice_rule : str
            Routing regime. ``fixed_share`` preserves the current two-step
            top-level AMM/CLOB split, while ``liquidity_aware`` makes
            routing sensitive to cost, depth, and venue price alignment.

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
        * **AMM-only**:   ``n_mm=0, clob_liq=0.1``
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
        shadow_clob = (enable_amm and n_mm == 0)

        # ── Apply liquidity scaling ──────────────────────────────────
        eff_clob_volume = max(10, int(clob_volume * clob_liq))
        eff_n_noise = max(1, int(n_noise * clob_liq))
        eff_n_fast_lp = max(0, int(n_fast_lp * clob_liq))
        eff_n_latent_lp = max(0, int(n_latent_lp * clob_liq))
        eff_d0 = 75.0 * clob_liq
        eff_cpmm_res = cpmm_reserves * amm_liq
        eff_hfmm_res = hfmm_reserves * amm_liq

        # Exchange (CLOB) — always present (reference price + classic agents)
        exchange = ExchangeAgent(price=price, std=clob_std,
                     volume=eff_clob_volume,
                     background_target_ratio=clob_background_target_ratio,
                     anchor_strength=clob_anchor_strength,
                     anchor_threshold_bps=clob_anchor_threshold_bps)

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
            clob = CLOBVenue(exchange, env=env)

        # ── AMM pools (optional) ────────────────────────────────────
        amm_pools: Dict[str, Any] = {}
        lp_providers: List = []
        arb = None

        if enable_amm:
            cpmm = CPMMPool(x=eff_cpmm_res,
                            y=eff_cpmm_res * price, fee=cpmm_fee,
                            dynamic_fee=dynamic_fee)
            hfmm = HFMMPool(x=eff_hfmm_res,
                            y=eff_hfmm_res * price,
                            A=hfmm_A, fee=hfmm_fee, rate=price,
                            dynamic_fee=dynamic_fee)
            amm_pools = {'cpmm': cpmm, 'hfmm': hfmm}
            lp_providers = [
                AMMProvider(cpmm, env,
                            phi1=0.45, phi2=1.6, phi3=0.8,
                            core_liquidity_ratio=0.45,
                            stabilizing_bias=0.01),
                AMMProvider(hfmm, env,
                            phi1=0.55, phi2=1.0, phi3=0.5,
                            core_liquidity_ratio=0.75,
                            stabilizing_bias=0.03),
            ]
            arb = AMMArbitrageur(
                clob,
                amm_pools,
                env=env,
                max_correction_pct=arb_max_correction_pct,
                trade_fraction_cap=arb_trade_fraction_cap,
            )

        if not shadow_clob:
            exchange.rebalance_background_liquidity(price, target_ratio=1.0)

        if enable_amm and not shadow_clob and match_initial_depth and amm_pools:
            ref_mid = clob.mid_price()
            aggregate_amm_depth = sum(
                max(0.0, pool.effective_depth(ref_mid))
                for pool in amm_pools.values()
            )
            if aggregate_amm_depth > 0:
                exchange.set_background_depth_target(ref_mid, aggregate_amm_depth)

        # ── CLOB book agents (diverse limit-order providers) ─────
        book_agents = [Random(exchange, cash=1e4, env=env)
                       for _ in range(eff_n_noise)]
        for _ in range(eff_n_fast_lp):
            book_agents.append(FastRecyclerLP(exchange, cash=1e4, env=env))
        for _ in range(eff_n_latent_lp):
            book_agents.append(LatentLP(exchange, cash=1e4, env=env))
        for _ in range(n_clob_fund):
            book_agents.append(Fundamentalist(exchange, cash=1e4, access=1, env=env))
        for _ in range(n_clob_chart):
            book_agents.append(Chartist(exchange, cash=1e4, env=env))
        for _ in range(n_clob_univ):
            book_agents.append(Universalist(exchange, cash=1e4, access=1, env=env))

        # ── Market Makers ────────────────────────────────────────────
        mm = None
        for i in range(n_mm):
            _mm = MarketMaker(exchange, cash=1e4, env=env,
                              amm_pools=amm_pools,
                              alpha0=1.5 + i * 0.3,
                              alpha1=300.0, alpha2=500.0,
                              d0=max(5.0, eff_d0 - i * 5.0),
                              d1=750.0, d2=500.0, d_min=3.0,
                              n_levels=5,
                              venue_interaction_mode=clob_amm_interaction,
                              amm_spread_impact_bps=clob_amm_spread_impact_bps,
                              amm_depth_impact=clob_amm_depth_impact)
            if mm is None:
                mm = _mm           # first MM → self.mm (step 2)
            else:
                book_agents.append(_mm)  # extra MMs → step 3 with book agents

        # ── Liquidity takers with venue routing ─────────────────────
        fx_traders: List = []
        for _ in range(n_fx_takers):
            fx_traders.append(Random(
                exchange, cash=1e4,
                clob=clob, amm_pools=amm_pools, env=env,
                amm_share_pct=amm_share_pct,
                venue_choice_rule=venue_choice_rule,
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
                venue_choice_rule=venue_choice_rule,
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
                venue_choice_rule=venue_choice_rule,
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
                venue_choice_rule=venue_choice_rule,
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

        # Full book-agent list for SimulatorInfo (Chartist / Universalist
        # need sentiment / strategy updates each iteration).
        all_book_agents = list(book_agents)
        if mm is not None:
            all_book_agents.append(mm)

        return cls(
            exchange=exchange,
            traders=all_book_agents,
            clob=clob,
            amm_pools=amm_pools,
            env=env,
            fx_traders=fx_traders,
            book_agents=book_agents,
            market_maker=mm,
            lp_providers=lp_providers,
            arbitrageur=arb,
            logger=logger,
            shock_iter=shock_iter,
            shock_pct=shock_pct,
            shock_mode=shock_mode,
            realism_shock_config={
                'fundamental_pct': fundamental_shock_pct,
                'order_flow_qty': order_flow_shock_qty,
                'order_flow_side': order_flow_shock_side,
                'liquidity_frac': liquidity_shock_frac,
                'funding_vol_intensity': funding_vol_shock_intensity,
                'reprice_prob_recovery': reprice_prob_recovery,
                'anchor_strength_recovery': anchor_strength_recovery,
                'bg_target_ratio_recovery': bg_target_ratio_recovery,
                'toxic_flow_decay': toxic_flow_decay,
                'liquidity_shock_decay': liquidity_shock_decay,
            },
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

    def stock_returns(self, roll: int = None) -> List[float] | float:
        p = self.prices
        div = self.dividends
        r = [(p[i+1] - p[i]) / p[i] + div[i] / p[i] for i in range(len(p) - 1)]
        return rolling(r, roll) if roll else mean(r)

    def abnormal_returns(self, roll: int = None) -> List[float]:
        rf = self.exchange.risk_free
        r = [r - rf for r in self.stock_returns()]
        return rolling(r, roll) if roll else r

    def return_volatility(self, window: int = None) -> List[float] | float:
        if window is None:
            return std(self.stock_returns())
        n = len(self.stock_returns(1))
        return [std(self.stock_returns(1)[i:i+window]) for i in range(n - window)]

    def price_volatility(self, window: int = None) -> List[float] | float:
        if window is None:
            return std(self.prices)
        return [std(self.prices[i:i+window]) for i in range(len(self.prices) - window)]

    def liquidity(self, roll: int = None) -> List[float] | float:
        n = len(self.prices)
        spreads = [el['ask'] - el['bid'] for el in self.spreads]
        prices = self.prices
        liq = [spreads[i] / prices[i] for i in range(n)]
        return rolling(liq, roll) if roll else mean(liq)
