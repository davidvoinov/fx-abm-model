from __future__ import annotations

from AgentBasedModel.utils import Order, OrderList
from AgentBasedModel.utils.math import exp, mean
import random
import math as _math
from typing import TYPE_CHECKING, Optional, List, Dict, Union

if TYPE_CHECKING:
    from AgentBasedModel.venues.amm import CPMMPool, HFMMPool
    from AgentBasedModel.venues.clob import CLOBVenue
    from AgentBasedModel.environment.processes import MarketEnvironment
    from AgentBasedModel.venues.amm import classify_trade_size as _classify_trade_size


class ExchangeAgent:
    """
    ExchangeAgent implements automatic orders handling within the order book. It supports limit orders,
    market orders, cancel orders, returns current spread prices and volumes.
    """
    id = 0

    def __init__(self, price: Union[float, int] = 100, std: Union[float, int] = 25, volume: int = 1000, rf: float = 5e-4,
                 transaction_cost: float = 0):
        """
        Initialization parameters
        :param price: stock initial price
        :param std: standard deviation of order prices in book
        :param volume: number of orders in book
        :param rf: risk-free rate (interest rate for cash holdings of agents)
        :param transaction_cost: cost that is paid on each successful deal
        """
        self.name = f'ExchangeAgent{self.id}'
        ExchangeAgent.id += 1

        self.order_book = {'bid': OrderList('bid'), 'ask': OrderList('ask')}
        self.dividend_book = list()  # list of future dividends
        self.risk_free = rf
        self.transaction_cost = transaction_cost
        self._fill_book(price, std, volume, rf * price)

    def generate_dividend(self):
        """
        Generate time series on future dividends.
        """
        # Generate future dividend
        d = self.dividend_book[-1] * self._next_dividend()
        self.dividend_book.append(max(d, 0))  # dividend > 0
        self.dividend_book.pop(0)

    def _fill_book(self, price, std, volume, div: float = 0.05):
        """
        Fill order book with random orders. Fill dividend book with n future dividends.
        """
        # Order book — qty range scales with volume for deeper books
        qty_lo = max(1, volume // 500)
        qty_hi = max(5, volume // 100)
        prices1 = [round(random.normalvariate(price - std, std), 2) for _ in range(volume // 2)]
        prices2 = [round(random.normalvariate(price + std, std), 2) for _ in range(volume // 2)]
        quantities = [random.randint(qty_lo, qty_hi) for _ in range(volume)]

        for (p, q) in zip(sorted(prices1 + prices2), quantities):
            if p > price:
                order = Order(round(p, 2), q, 'ask', None)
                self.order_book['ask'].append(order)
            else:
                order = Order(p, q, 'bid', None)
                self.order_book['bid'].push(order)

        # Dividend book
        for i in range(100):
            self.dividend_book.append(max(div, 0))  # dividend > 0
            div *= self._next_dividend()

    def _clear_book(self):
        """
        Clears glass from orders with 0 qty.

        complexity O(n)

        :return: void
        """
        self.order_book['bid'] = OrderList.from_list([order for order in self.order_book['bid'] if order.qty > 0])
        self.order_book['ask'] = OrderList.from_list([order for order in self.order_book['ask'] if order.qty > 0])

    def spread(self) -> Optional[dict]:
        """
        :return: {'bid': float, 'ask': float}
        """
        if self.order_book['bid'] and self.order_book['ask']:
            return {'bid': self.order_book['bid'].first.price, 'ask': self.order_book['ask'].first.price}
        return None

    def spread_volume(self) -> Optional[dict]:
        """
        :return: {'bid': float, 'ask': float}
        """
        if self.order_book['bid'] and self.order_book['ask']:
            return {'bid': self.order_book['bid'].first.qty, 'ask': self.order_book['ask'].first.qty}
        return None

    def price(self) -> Optional[float]:
        spread = self.spread()
        if spread:
            return round((spread['bid'] + spread['ask']) / 2, 1)
        raise Exception(f'Price cannot be determined, since no orders either bid or ask')

    def dividend(self, access: int = None) -> Union[list, float]:
        """
        Returns current dividend payment value. If called by a trader, returns n future dividends
        given information access.
        """
        if access is None:
            return self.dividend_book[0]
        return self.dividend_book[:access]

    @classmethod
    def _next_dividend(cls, std=5e-3):
        return exp(random.normalvariate(0, std))

    def limit_order(self, order: Order):
        """
        Executes limit order, fulfilling orders if on other side of spread

        :return: void
        """
        bid, ask = self.spread().values()
        t_cost = self.transaction_cost
        if not bid or not ask:
            return

        if order.order_type == 'bid':
            if order.price >= ask:
                order = self.order_book['ask'].fulfill(order, t_cost)
            if order.qty > 0:
                self.order_book['bid'].insert(order)
            return

        elif order.order_type == 'ask':
            if order.price <= bid:
                order = self.order_book['bid'].fulfill(order, t_cost)
            if order.qty > 0:
                self.order_book['ask'].insert(order)

    def market_order(self, order: Order) -> Order:
        """
        Executes market order, fulfilling orders on the other side of spread

        :return: Order
        """
        t_cost = self.transaction_cost
        if order.order_type == 'bid':
            order = self.order_book['ask'].fulfill(order, t_cost)
        elif order.order_type == 'ask':
            order = self.order_book['bid'].fulfill(order, t_cost)
        return order

    def cancel_order(self, order: Order):
        """
        Cancel order from order book

        :return: void
        """
        if order.order_type == 'bid':
            self.order_book['bid'].remove(order)
        elif order.order_type == 'ask':
            self.order_book['ask'].remove(order)


class Trader:
    id = 0

    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0,
                 clob: CLOBVenue = None,
                 amm_pools: Dict[str, CPMMPool | HFMMPool] = None,
                 env: MarketEnvironment = None,
                 amm_share_pct: float = 25.0,
                 deterministic_venue: bool = False,
                 beta_amm: float = 0.05,
                 cpmm_bias_bps: float = 5.0,
                 cost_noise_std: float = 1.5):
        """
        Trader that is activated on call to perform action.

        :param market: link to exchange agent
        :param cash: trader's cash available
        :param assets: trader's number of shares hold
        :param clob: CLOBVenue wrapper (enables multi-venue mode)
        :param amm_pools: dict of AMM pools (enables multi-venue mode)
        :param env: MarketEnvironment for σ_t, c_t
        :param amm_share_pct: target probability (0–100) of routing to AMM vs CLOB
        :param deterministic_venue: if True, use argmin instead of logit
        :param beta_amm: logit sensitivity for intra-AMM choice (CPMM vs HFMM)
        :param cpmm_bias_bps: non-monetary utility discount applied to CPMM cost (bps)
        :param cost_noise_std: std of Gaussian noise added to AMM cost estimates (bps)
        """
        self.type = 'Unknown'
        self.name = f'Trader{self.id}'
        self.id = Trader.id
        Trader.id += 1

        self.market = market
        self.orders = list()

        self.cash = cash
        self.assets = assets

        # Multi-venue (optional)
        self.clob = clob
        self.amm_pools = amm_pools or {}
        self.env = env
        self.amm_share_pct = max(0.0, min(100.0, amm_share_pct))
        self.deterministic_venue = deterministic_venue
        self.beta_amm = beta_amm
        self.cpmm_bias_bps = cpmm_bias_bps
        self.cost_noise_std = cost_noise_std
        self.trades: List[dict] = []

    @property
    def multi_venue(self) -> bool:
        """True if trader operates in FX multi-venue mode (even CLOB-only)."""
        return self.clob is not None

    def __str__(self) -> str:
        return f'{self.name} ({self.type})'

    def equity(self):
        price = self.market.price() if self.market.price() is not None else 0
        return self.cash + self.assets * price

    def _buy_limit(self, quantity, price):
        order = Order(round(price, 2), round(quantity), 'bid', self)
        self.orders.append(order)
        self.market.limit_order(order)

    def _sell_limit(self, quantity, price):
        order = Order(round(price, 2), round(quantity), 'ask', self)
        self.orders.append(order)
        self.market.limit_order(order)

    def _buy_market(self, quantity) -> int:
        """
        :return: quantity unfulfilled
        """
        if not self.market.order_book['ask']:
            return quantity
        order = Order(self.market.order_book['ask'].last.price, round(quantity), 'bid', self)
        return self.market.market_order(order).qty

    def _sell_market(self, quantity) -> int:
        """
        :return: quantity unfulfilled
        """
        if not self.market.order_book['bid']:
            return quantity
        order = Order(self.market.order_book['bid'].last.price, round(quantity), 'ask', self)
        return self.market.market_order(order).qty

    def _cancel_order(self, order: Order):
        self.market.cancel_order(order)
        self.orders.remove(order)

    # ---- multi-venue routing (active only when amm_pools is set) ---------

    def estimate_costs(self, Q: float, side: str = 'buy') -> Dict[str, float]:
        """Return {venue_name: internal execution cost_bps} for quantity Q."""
        costs: Dict[str, float] = {}
        try:
            costs['clob'] = self.clob.cost_bps(Q, side)
        except Exception:
            costs['clob'] = float('inf')
        for name, pool in self.amm_pools.items():
            try:
                # AMM internal TCA: slippage is measured vs local pool mid.
                q = pool.quote_buy(Q) if side == 'buy' else pool.quote_sell(Q)
                costs[name] = q['cost_bps']
            except Exception:
                costs[name] = float('inf')
        return costs

    def choose_venue(self, Q: float, side: str = 'buy') -> str:
        """Two-step venue selection.

        Step 1 — CLOB vs AMM with fixed probability ``amm_share_pct``.
        Step 2 — within AMM, CPMM vs HFMM (softmax with β_amm),
            using bias-adjusted and noise-perturbed costs.

        Falls back to CLOB when there are no AMM pools.
        """
        costs = self.estimate_costs(Q, side)

        # Deterministic shortcut
        if self.deterministic_venue:
            return min(costs, key=costs.get)

        # Identify AMM venues
        amm_names = [n for n in costs if n != 'clob']

        # If no AMM pools → always CLOB
        if not amm_names:
            return 'clob'

        # ── Step 1: CLOB vs AMM (direct probability) ─────────────────
        if random.random() * 100.0 >= self.amm_share_pct:
            return 'clob'

        # ── Step 2: within AMM — CPMM vs HFMM ────────────────────────
        amm_raw_costs = {n: costs[n] for n in amm_names}
        # Apply bias and noise
        adj_costs: Dict[str, float] = {}
        for n, c in amm_raw_costs.items():
            adj = c
            # Bias: reduce perceived CPMM cost
            if n == 'cpmm':
                adj -= self.cpmm_bias_bps
            # Noise: estimation error
            if self.cost_noise_std > 0:
                adj += random.gauss(0, self.cost_noise_std)
            adj_costs[n] = adj

        return self._softmax_pick(adj_costs, self.beta_amm)

    # ---- helper ----------------------------------------------------------

    @staticmethod
    def _softmax_pick(costs: Dict[str, float], beta: float) -> str:
        """Pick a key from *costs* dict via logit softmax with sensitivity *beta*."""
        items = list(costs.items())
        finite = [c for _, c in items if c != float('inf')]
        if not finite:
            return items[0][0]
        min_c = min(finite)
        weights = []
        for name, c in items:
            if c == float('inf'):
                weights.append(0.0)
            else:
                weights.append(_math.exp(-beta * (c - min_c)))
        total = sum(weights)
        if total == 0:
            return items[0][0]
        r = random.random() * total
        cumulative = 0.0
        for (name, _), w in zip(items, weights):
            cumulative += w
            if r <= cumulative:
                return name
        return items[-1][0]

    def _classify(self, Q: float) -> dict:
        """
        Compute θ = Q/R relative to max AMM effective depth.
        """
        from AgentBasedModel.venues.amm import classify_trade_size
        try:
            mid = self.clob.mid_price()
        except Exception:
            return dict(theta=0.0, size_bucket='medium', R=1.0)
        R = 1.0
        ref_pool = None
        for pool in self.amm_pools.values():
            d = pool.effective_depth(mid)
            if d > R:
                R = d
                ref_pool = pool
        if ref_pool is not None:
            return classify_trade_size(Q, ref_pool, mid)
        theta = Q / R
        bucket = 'small' if theta <= 0.01 else ('medium' if theta <= 0.05 else 'large')
        return dict(theta=theta, size_bucket=bucket, R=R)

    def _execute_on_venue(self, venue: str, Q: float, side: str) -> dict:
        """Execute trade on chosen venue. Returns cost dict."""
        if venue == 'clob':
            return self._execute_clob(Q, side)
        pool = self.amm_pools[venue]
        return pool.execute_buy(Q) if side == 'buy' else pool.execute_sell(Q)

    def _execute_clob(self, Q: float, side: str) -> dict:
        """Market order on CLOB with cost quoting."""
        try:
            if side == 'buy':
                quote_info = self.clob.quote_buy(Q)
                if quote_info['cost_bps'] == float('inf'):
                    return quote_info
                self._buy_market(round(Q))
                return quote_info
            else:
                quote_info = self.clob.quote_sell(Q)
                if quote_info['cost_bps'] == float('inf'):
                    return quote_info
                self._sell_market(round(Q))
                return quote_info
        except Exception:
            return self.clob._inf_quote()

    def _make_trade_record(self, venue: str, side: str, Q: float,
                           result: dict, cls_info: dict) -> dict:
        """Build a trade record dict for logging."""
        return dict(
            trader_id=self.id, trader_type=self.type,
            venue=venue, side=side, quantity=Q,
            cost_bps=result.get('cost_bps', 0),
            theta=cls_info.get('theta', 0),
            size_bucket=cls_info.get('size_bucket', 'medium'),
        )


class Random(Trader):
    """
    Random creates noisy orders to recreate trading in real environment.

    When *amm_pools* / *clob* are provided (multi-venue mode), the agent
    becomes a stochastic market-order taker that routes via cost estimation
    (replaces former FXNoiseTaker).  Use *label* to tag sub-populations
    (e.g. 'Retail', 'Institutional').
    """
    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0,
                 # multi-venue params (forwarded to Trader)
                 clob: CLOBVenue = None,
                 amm_pools: Dict[str, CPMMPool | HFMMPool] = None,
                 env: MarketEnvironment = None,
                 amm_share_pct: float = 25.0,
                 deterministic_venue: bool = False,
                 beta_amm: float = 0.05,
                 cpmm_bias_bps: float = 5.0,
                 cost_noise_std: float = 1.5,
                 # FX-noise-taker params (only used in multi-venue mode)
                 trade_prob: float = 0.3,
                 q_min: int = 1,
                 q_max: int = 5,
                 label: str = 'Random'):
        super().__init__(market, cash, assets,
                         clob=clob, amm_pools=amm_pools, env=env,
                         amm_share_pct=amm_share_pct, deterministic_venue=deterministic_venue,
                         beta_amm=beta_amm, cpmm_bias_bps=cpmm_bias_bps,
                         cost_noise_std=cost_noise_std)
        self.type = label if self.multi_venue else 'Random'
        self.trade_prob = trade_prob
        self.q_min = q_min
        self.q_max = q_max

    @staticmethod
    def draw_delta(std: Union[float, int] = 2.5):
        lamb = 1 / std
        return random.expovariate(lamb)

    @staticmethod
    def draw_price(order_type, spread: dict, std: Union[float, int] = 2.5) -> float:
        """
        Draw price for limit order of Noise Agent. The price is calculated as:
        1) 35% - within the spread - uniform distribution
        2) 65% - out of the spread - delta from best price is exponential distribution r.v.
        """
        random_state = random.random()  # Determines IN spread OR OUT of spread

        # Within the spread
        if random_state < .35:
            return random.uniform(spread['bid'], spread['ask'])

        # Out of spread
        else:
            delta = Random.draw_delta(std)
            if order_type == 'bid':
                return spread['bid'] - delta
            if order_type == 'ask':
                return spread['ask'] + delta

    @staticmethod
    def draw_quantity(a=1, b=5) -> float:
        """
        Draw random quantity to buy from uniform distribution.

        :param a: minimal quantity
        :param b: maximal quantity
        :return: quantity for order
        """
        return random.randint(a, b)

    def call(self):
        # ---------- multi-venue mode (FX noise taker) ---------------------
        if self.multi_venue:
            if random.random() > self.trade_prob:
                return None
            side = 'buy' if random.random() > 0.5 else 'sell'
            Q = random.randint(self.q_min, self.q_max)
            venue = self.choose_venue(Q, side)
            cls_info = self._classify(Q)
            result = self._execute_on_venue(venue, Q, side)
            rec = self._make_trade_record(venue, side, Q, result, cls_info)
            self.trades.append(rec)
            return rec

        # ---------- classic single-venue mode -----------------------------
        spread = self.market.spread()
        if spread is None:
            return

        random_state = random.random()

        if random_state > .5:
            order_type = 'bid'
        else:
            order_type = 'ask'

        random_state = random.random()
        # Market order
        if random_state > .85:
            quantity = self.draw_quantity()
            if order_type == 'bid':
                self._buy_market(quantity)
            elif order_type == 'ask':
                self._sell_market(quantity)

        # Limit order
        elif random_state > .5:
            price = self.draw_price(order_type, spread)
            quantity = self.draw_quantity()
            if order_type == 'bid':
                self._buy_limit(quantity, price)
            elif order_type == 'ask':
                self._sell_limit(quantity, price)

        # Cancellation order
        elif random_state < .35:
            if self.orders:
                order_n = random.randint(0, len(self.orders) - 1)
                self._cancel_order(self.orders[order_n])


class Fundamentalist(Trader):
    """
    Fundamentalist traders strictly believe in the information they receive. If they find an ask
    order with a price lower or a bid order with a price higher than their estimated present
    value, i.e. E(V|Ij,k), they accept the limit order, otherwise they put a new limit order
    between the former best bid and best ask prices.

    In multi-venue mode (amm_pools provided), the agent trades when the CLOB
    mid-price deviates from *fundamental_rate* — replaces former FXFundamentalist.
    """
    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0, access: int = 1,
                 # multi-venue params
                 clob: CLOBVenue = None,
                 amm_pools: Dict[str, CPMMPool | HFMMPool] = None,
                 env: MarketEnvironment = None,
                 amm_share_pct: float = 25.0,
                 deterministic_venue: bool = False,
                 beta_amm: float = 0.05,
                 cpmm_bias_bps: float = 5.0,
                 cost_noise_std: float = 1.5,
                 # FX-fundamentalist params (used only in multi-venue mode)
                 fundamental_rate: float = 100.0,
                 fx_gamma: float = 5e-3,
                 fx_q_max: int = 10):
        """
        :param market: exchange agent link
        :param cash: number of cash
        :param assets: number of assets
        :param access: number of future dividends informed
        :param fundamental_rate: exogenous fair FX rate (multi-venue mode)
        :param fx_gamma: min mispricing to trigger FX trade
        :param fx_q_max: max FX trade size
        """
        super().__init__(market, cash, assets,
                         clob=clob, amm_pools=amm_pools, env=env,
                         amm_share_pct=amm_share_pct, deterministic_venue=deterministic_venue,
                         beta_amm=beta_amm, cpmm_bias_bps=cpmm_bias_bps,
                         cost_noise_std=cost_noise_std)
        self.type = 'Fundamentalist'
        self.access = access
        self.fundamental_rate = fundamental_rate
        self.fx_gamma = fx_gamma
        self.fx_q_max = fx_q_max

    @staticmethod
    def evaluate(dividends: list, risk_free: float):
        """
        Evaluate stock using constant dividend model.
        """
        divs = dividends  # expected value of future dividends
        r = risk_free  # risk-free rate

        perp = divs[-1] / r / (1 + r)**(len(divs) - 1)  # perpetual payments
        known = sum([divs[i] / (1 + r)**(i + 1) for i in range(len(divs) - 1)]) if len(divs) > 1 else 0
        return known + perp

    @staticmethod
    def draw_quantity(pf, p, gamma: float = 5e-3):
        q = round(abs(pf - p) / p / gamma)
        return min(q, 5)

    def call(self):
        # ---------- multi-venue mode (FX fundamentalist) ------------------
        if self.multi_venue:
            try:
                mid = self.clob.mid_price()
            except Exception:
                return None
            mispricing = (self.fundamental_rate - mid) / mid
            if abs(mispricing) < self.fx_gamma:
                return None
            side = 'buy' if mispricing > 0 else 'sell'
            Q = min(round(abs(mispricing) / self.fx_gamma), self.fx_q_max)
            if Q <= 0:
                return None
            venue = self.choose_venue(Q, side)
            cls_info = self._classify(Q)
            result = self._execute_on_venue(venue, Q, side)
            rec = self._make_trade_record(venue, side, Q, result, cls_info)
            self.trades.append(rec)
            return rec

        # ---------- classic single-venue mode (dividend DCF) --------------
        pf = round(self.evaluate(self.market.dividend(self.access), self.market.risk_free), 1)  # fundamental price
        p = self.market.price()
        spread = self.market.spread()
        t_cost = self.market.transaction_cost

        if spread is None:
            return

        random_state = random.random()
        qty = Fundamentalist.draw_quantity(pf, p)  # quantity to buy
        if not qty:
            return

        # Limit or Market order
        if random_state > .45:
            random_state = random.random()

            ask_t = round(spread['ask'] * (1 + t_cost), 1)
            bid_t = round(spread['bid'] * (1 - t_cost), 1)

            if pf >= ask_t:
                if random_state > .5:
                    self._buy_market(qty)
                else:
                    self._sell_limit(qty, (pf + Random.draw_delta()) * (1 + t_cost))

            elif pf <= bid_t:
                if random_state > .5:
                    self._sell_market(qty)
                else:
                    self._buy_limit(qty, (pf - Random.draw_delta()) * (1 - t_cost))

            elif ask_t > pf > bid_t:
                if random_state > .5:
                    self._buy_limit(qty, (pf - Random.draw_delta()) * (1 - t_cost))
                else:
                    self._sell_limit(qty, (pf + Random.draw_delta()) * (1 + t_cost))

        # Cancel order
        else:
            if self.orders:
                self._cancel_order(self.orders[0])


class Chartist(Trader):
    """
    Chartist traders are searching for trends in the price movements. Each trader has sentiment - opinion
    about future price movement (either increasing, or decreasing). Based on sentiment trader either
    buys stock or sells. Sentiment revaluation happens at the end of each iteration based on opinion
    propagation among other chartists, current price changes.
    """
    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0, **kwargs):
        """
        :param market: exchange agent link
        :param cash: number of cash
        :param assets: number of assets
        """
        super().__init__(market, cash, assets, **kwargs)
        self.type = 'Chartist'
        self.sentiment = 'Optimistic' if random.random() > .5 else 'Pessimistic'

    def call(self):
        """
        If 'steps' consecutive steps of upward (downward) price movements -> buy (sell) market order. If there are no
        such trend, act as random trader placing only limit orders.
        """
        random_state = random.random()
        t_cost = self.market.transaction_cost
        spread = self.market.spread()

        if self.sentiment == 'Optimistic':
            # Market order
            if random_state > .85:
                self._buy_market(Random.draw_quantity())
            # Limit order
            elif random_state > .5:
                self._buy_limit(Random.draw_quantity(), Random.draw_price('bid', spread) * (1 - t_cost))
            # Cancel order
            elif random_state < .35:
                if self.orders:
                    self._cancel_order(self.orders[-1])
        elif self.sentiment == 'Pessimistic':
            # Market order
            if random_state > .85:
                self._sell_market(Random.draw_quantity())
            # Limit order
            elif random_state > .5:
                self._sell_limit(Random.draw_quantity(), Random.draw_price('ask', spread) * (1 + t_cost))
            # Cancel order
            elif random_state < .35:
                if self.orders:
                    self._cancel_order(self.orders[-1])

    def change_sentiment(self, info, a1=1, a2=1, v1=.1):
        """
        Change sentiment

        :param info: SimulatorInfo
        :param a1: importance of chartists opinion
        :param a2: importance of current price changes
        :param v1: frequency of revaluation of opinion for sentiment
        """
        n_traders = len(info.traders)  # number of all traders
        n_chartists = sum([tr_type == 'Chartist' for tr_type in info.types[-1].values()])
        n_optimistic = sum([tr_type == 'Optimistic' for tr_type in info.sentiments[-1].values()])
        n_pessimists = sum([tr_type == 'Pessimistic' for tr_type in info.sentiments[-1].values()])

        dp = info.prices[-1] - info.prices[-2] if len(info.prices) > 1 else 0  # price derivative
        p = self.market.price()  # market price
        x = (n_optimistic - n_pessimists) / n_chartists

        U = a1 * x + a2 / v1 * dp / p
        if self.sentiment == 'Optimistic':
            prob = v1 * n_chartists / n_traders * exp(U)
            if prob > random.random():
                self.sentiment = 'Pessimistic'

        elif self.sentiment == 'Pessimistic':
            prob = v1 * n_chartists / n_traders * exp(-U)
            if prob > random.random():
                self.sentiment = 'Optimistic'

        # print('sentiment', prob)


class Universalist(Fundamentalist, Chartist):
    """
    Universalist mixes Fundamentalist, Chartist trading strategies, and allows to change from
    one strategy to another.
    """
    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0, access: int = 1):
        """
        :param market: exchange agent link
        :param cash: number of cash
        :param assets: number of assets
        :param access: number of future dividends informed
        """
        super().__init__(market, cash, assets, access=access)
        self.type = 'Chartist' if random.random() > .5 else 'Fundamentalist'  # randomly decide type
        self.sentiment = 'Optimistic' if random.random() > .5 else 'Pessimistic'  # sentiment about trend (Chartist)
        self.access = access  # next n dividend payments known (Fundamentalist)

    def call(self):
        """
        Call one of parents' methods depending on what type it is currently set.
        """
        if self.type == 'Chartist':
            Chartist.call(self)
        elif self.type == 'Fundamentalist':
            Fundamentalist.call(self)

    def change_strategy(self, info, a1=1, a2=1, a3=1, v1=.1, v2=.1, s=.1):
        """
        Change strategy or sentiment

        :param info: SimulatorInfo
        :param a1: importance of chartists opinion
        :param a2: importance of current price changes
        :param a3: importance of fundamentalist profit
        :param v1: frequency of revaluation of opinion for sentiment
        :param v2: frequency of revaluation of opinion for strategy
        :param s: importance of fundamental value opportunities
        """
        # Gather variables
        n_traders = len(info.traders)  # number of all traders
        n_fundamentalists = sum([tr.type == 'Fundamentalist' for tr in info.traders.values()])
        n_optimistic = sum([tr.sentiment == 'Optimistic' for tr in info.traders.values() if tr.type == 'Chartist'])
        n_pessimists = sum([tr.sentiment == 'Pessimistic' for tr in info.traders.values() if tr.type == 'Chartist'])

        dp = info.prices[-1] - info.prices[-2] if len(info.prices) > 1 else 0  # price derivative
        p = self.market.price()  # market price
        pf = self.evaluate(self.market.dividend(self.access), self.market.risk_free)  # fundamental price
        r = pf * self.market.risk_free  # expected dividend return
        R = mean(info.returns[-1].values())  # average return in economy

        # Change sentiment
        if self.type == 'Chartist':
            Chartist.change_sentiment(self, info, a1, a2, v1)

        # Change strategy
        U1 = max(-100, min(100, a3 * ((r + 1 / v2 * dp) / p - R - s * abs((pf - p) / p))))
        U2 = max(-100, min(100, a3 * (R - (r + 1 / v2 * dp) / p - s * abs((pf - p) / p))))

        if self.type == 'Chartist':
            if self.sentiment == 'Optimistic':
                prob = v2 * n_optimistic / (n_traders * exp(U1))
                if prob > random.random():
                    self.type = 'Fundamentalist'
            elif self.sentiment == 'Pessimistic':
                prob = v2 * n_pessimists / (n_traders * exp(U2))
                if prob > random.random():
                    self.type = 'Fundamentalist'

        elif self.type == 'Fundamentalist':
            prob = v2 * n_fundamentalists / (n_traders * exp(-U1))
            if prob > random.random() and self.sentiment == 'Pessimistic':
                self.type = 'Chartist'
                self.sentiment = 'Optimistic'

            prob = v2 * n_fundamentalists / (n_traders * exp(-U2))
            if prob > random.random() and self.sentiment == 'Optimistic':
                self.type = 'Chartist'
                self.sentiment = 'Pessimistic'


class MarketMaker(Trader):
    """
    MarketMaker creates limit orders on both sides of the spread trying to gain on
    spread between bid and ask prices, and maintain its assets to cash ratio in balance.

    When *env* is provided (multi-venue mode), the quoted spread and depth become
    functions of exogenous σ_t and c_t (Brunnermeier–Pedersen model) — replaces
    former FXMarketMaker.
    """

    def __init__(self, market: ExchangeAgent, cash: float, assets: int = 0, softlimit: int = 100,
                 # multi-venue / FX params
                 clob: CLOBVenue = None,
                 amm_pools: Dict[str, CPMMPool | HFMMPool] = None,
                 env: MarketEnvironment = None,
                 amm_share_pct: float = 25.0,
                 deterministic_venue: bool = False,
                 beta_amm: float = 0.05,
                 cpmm_bias_bps: float = 5.0,
                 cost_noise_std: float = 1.5,
                 # Brunnermeier-Pedersen params (used when env is set)
                 alpha0: float = 2.0, alpha1: float = 100.0, alpha2: float = 50.0,
                 d0: float = 50.0, d1: float = 200.0, d2: float = 100.0,
                 d_min: float = 5.0, n_levels: int = 1):
        super().__init__(market, cash, assets,
                         clob=clob, amm_pools=amm_pools, env=env,
                         amm_share_pct=amm_share_pct, deterministic_venue=deterministic_venue,
                         beta_amm=beta_amm, cpmm_bias_bps=cpmm_bias_bps,
                         cost_noise_std=cost_noise_std)
        self.type = 'Market Maker'
        self.softlimit = softlimit
        self.ul = softlimit
        self.ll = -softlimit
        self.panic = False
        # Brunnermeier-Pedersen coefficients
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.d_min = d_min
        self.n_levels = max(1, n_levels)

    def _target_spread_bps(self) -> float:
        """Quoted spread in bps as f(σ, c)."""
        return self.alpha0 + self.alpha1 * self.env.sigma + self.alpha2 * self.env.funding_cost

    def _target_depth(self) -> float:
        """Depth as f(σ, c), floored at d_min."""
        d = self.d0 - self.d1 * self.env.sigma - self.d2 * self.env.funding_cost
        return max(d, self.d_min)

    def call(self):
        # ---------- multi-venue mode (σ/c-dependent MM) -------------------
        if self.env is not None:
            for order in self.orders.copy():
                try:
                    self._cancel_order(order)
                except Exception:
                    pass
            self.orders.clear()

            sp = self.market.spread()
            if sp is None:
                return

            mid = (sp['bid'] + sp['ask']) / 2.0
            half_spread = mid * self._target_spread_bps() / 10_000.0 / 2.0
            total_depth = max(1, int(self._target_depth()))

            # Inverse pyramid: most depth at tight spread, tapering out.
            # Mimics real MM behaviour — dense inside, thin at wide levels.
            weights = list(range(self.n_levels, 0, -1))
            w_sum = sum(weights)
            for i in range(self.n_levels):
                qty = max(1, int(total_depth * weights[i] / w_sum))
                offset = half_spread * (1.0 + i * 0.5)
                bid_price = round(mid - offset, 2)
                ask_price = round(mid + offset, 2)
                self._buy_limit(qty, bid_price)
                self._sell_limit(qty, ask_price)
            return

        # ---------- classic single-venue mode (inventory balance) ---------
        # Clear previous orders
        for order in self.orders.copy():
            self._cancel_order(order)

        spread = self.market.spread()

        # Calculate bid and ask volume
        bid_volume = max(0., self.ul - 1 - self.assets)
        ask_volume = max(0., self.assets - self.ll - 1)

        # If in panic state we only either sell or buy commodities
        if not bid_volume or not ask_volume:
            self.panic = True
            self._buy_market((self.ul + self.ll) / 2 - self.assets) if ask_volume is None else None
            self._sell_market(self.assets - (self.ul + self.ll) / 2) if bid_volume is None else None
        else:
            self.panic = False
            base_offset = -((spread['ask'] - spread['bid']) * (self.assets / self.softlimit))  # Price offset
            self._buy_limit(bid_volume, spread['bid'] - base_offset - .1)  # BID
            self._sell_limit(ask_volume, spread['ask'] + base_offset + .1)  # ASK


# ---------------------------------------------------------------------------
# AMM-only agents (genuinely new concepts with no CLOB-only analogue)
# ---------------------------------------------------------------------------

class AMMProvider:
    """
    Liquidity Provider for AMM pools.

    Rule (§5.3):
        L_{t+1} = L_t + φ₁ Π^fee_t − φ₂ σ_t − φ₃ c_t

    Produces non-monotonic response to volatility:
    at moderate σ, fee income dominates → L grows;
    at high σ, risk terms dominate → L shrinks.
    """

    def __init__(self, pool: CPMMPool | HFMMPool,
                 env: MarketEnvironment,
                 phi1: float = 0.5,
                 phi2: float = 2.0,
                 phi3: float = 1.0,
                 max_adj: float = 0.05):
        self.type = 'AMMProvider'
        self.pool = pool
        self.env = env
        self.phi1 = phi1
        self.phi2 = phi2
        self.phi3 = phi3
        self.max_adj = max_adj

    def update_liquidity(self):
        """Adjust pool liquidity according to LP rule.  Called once per period."""
        L = self.pool.liquidity_measure()
        if L <= 0:
            return
        fee_income = self.pool.period_fee_revenue
        sigma = self.env.sigma
        c = self.env.funding_cost
        delta_L = self.phi1 * fee_income - self.phi2 * sigma - self.phi3 * c
        frac = delta_L / L
        frac = max(-self.max_adj, min(self.max_adj, frac))
        if frac > 0:
            self.pool.add_liquidity(frac)
        elif frac < 0:
            self.pool.remove_liquidity(abs(frac))


class AMMArbitrageur:
    """
    Arbitrageur that trades on AMM pools to align their mid-price with
    a reference price S_t.

    Reference price cascade
    -----------------------
    1. **env.fair_price** — exogenous GBM price (always authoritative
       when available; used in AMM-only mode).
    2. **CLOB mid** — if fair_price is absent and book is healthy.
    3. **Blend** — weighted average of CLOB and AMM consensus when
       spread is moderately wide.
    4. **AMM consensus** — pure average of pool mid-prices as last
       resort.

    Routing modes
    -------------
    * **all** (default) — arbitrage every mispriced pool independently.
    * **best** — rank pools by profitability (deviation − fee), trade
      only the most profitable pool each iteration.

    Robustness features
    -------------------
    * **Max correction cap**: per-iteration price correction is capped
      at ``max_correction_pct`` to prevent reserve drainage from a
      single erratic tick.
    * **Reserve-health guard**: arbitrage is skipped for a pool whose
      reserve value-ratio is dangerously out of balance.
    * **HFMM rate update**: after arbitrage the HFMM ``rate`` is
      re-centered when the mid-price drifts > 1 % from the peg.
    """

    def __init__(self, clob,
                 amm_pools: Dict[str, CPMMPool | HFMMPool],
                 env: MarketEnvironment = None,
                 max_spread_bps: float = 500.0,
                 max_correction_pct: float = 10.0,
                 routing: str = 'all'):
        self.clob = clob
        self.amm_pools = amm_pools
        self.env = env
        self.type = 'AMMArbitrageur'
        self.max_spread_bps = max_spread_bps
        self.max_correction_pct = max_correction_pct / 100.0
        self.routing = routing

    # ---- helpers ---------------------------------------------------------

    def _amm_consensus(self) -> Optional[float]:
        """Average mid-price across all AMM pools."""
        mids = []
        for p in self.amm_pools.values():
            try:
                mids.append(p.mid_price())
            except Exception:
                pass
        return sum(mids) / len(mids) if mids else None

    def _reference_price(self) -> Optional[float]:
        """
        Determine the best available reference price.

        Priority:
        1. env.fair_price (exogenous GBM — always trustworthy)
        2. CLOB mid (if book is healthy)
        3. Blend of CLOB + AMM consensus
        4. Pure AMM consensus
        """
        # (1) Exogenous fair price — highest priority
        if self.env is not None:
            fp = self.env.fair_price
            if fp is not None:
                return fp

        # (2–4) CLOB / blend / AMM fallback
        S_clob = None
        try:
            S_clob = self.clob.mid_price()
        except Exception:
            pass

        S_amm = self._amm_consensus()

        if S_clob is None and S_amm is None:
            return None
        if S_clob is None:
            return S_amm
        if S_amm is None:
            return S_clob

        # Sanity: if CLOB is implausible relative to AMM (>50 % off)
        clob_dev = abs(S_clob - S_amm) / S_amm if S_amm > 0 else 0
        if clob_dev > 0.5:
            return S_amm

        try:
            spread_bps = self.clob.quoted_spread_bps()
        except Exception:
            spread_bps = float('inf')

        if not _math.isfinite(spread_bps) or spread_bps > 2000:
            return S_amm
        if spread_bps > self.max_spread_bps:
            w = max(0.0, 1.0 - (spread_bps - self.max_spread_bps)
                    / (2000 - self.max_spread_bps))
            return w * S_clob + (1.0 - w) * S_amm
        return S_clob

    @staticmethod
    def _reserve_healthy(pool, threshold: float = 0.1) -> bool:
        """Check that pool reserves are not dangerously one-sided."""
        try:
            mp = pool.mid_price()
            if mp <= 0 or pool.x <= 0:
                return False
            value_ratio = pool.y / (pool.x * mp)
            return threshold < value_ratio < (1.0 / threshold)
        except Exception:
            return False

    # ---- main entry point ------------------------------------------------

    def arbitrage(self):
        """For each AMM pool, align its price to S_t if profitable."""
        S_t = self._reference_price()
        if S_t is None:
            return

        if self.routing == 'best':
            self._arbitrage_best(S_t)
        else:
            self._arbitrage_all(S_t)

    def _arbitrage_all(self, S_t):
        """Original mode: trade every mispriced pool."""
        for name, pool in self.amm_pools.items():
            self._arb_one_pool(pool, S_t)

    def _arbitrage_best(self, S_t):
        """Route to the most profitable pool only."""
        candidates = []
        for name, pool in self.amm_pools.items():
            if not self._reserve_healthy(pool):
                if hasattr(pool, 'update_rate'):
                    pool.update_rate()
                continue
            try:
                current = pool.mid_price()
                if current <= 0:
                    continue
                dev_bps = abs(S_t - current) / current * 10_000
                fee_bps = pool.fee * 10_000
                profit_bps = dev_bps - fee_bps * 2
                if profit_bps > 0:
                    candidates.append((profit_bps, name, pool))
            except Exception:
                pass

        if not candidates:
            return
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, _, best_pool = candidates[0]
        self._arb_one_pool(best_pool, S_t)

    def _arb_one_pool(self, pool, S_t):
        """Arbitrage a single pool toward S_t."""
        try:
            if not self._reserve_healthy(pool):
                if hasattr(pool, 'update_rate'):
                    pool.update_rate()
                return

            current = pool.mid_price()
            if current <= 0:
                return
            dev = (S_t - current) / current
            if abs(dev) > self.max_correction_pct:
                S_capped = current * (1.0 + _math.copysign(
                    self.max_correction_pct, dev))
            else:
                S_capped = S_t

            pool.arbitrage_to_target(S_capped)

            if hasattr(pool, 'update_rate'):
                pool.update_rate()
        except Exception:
            pass
