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
        prices1 = [round(random.normalvariate(price - std, std), 1) for _ in range(volume // 2)]
        prices2 = [round(random.normalvariate(price + std, std), 1) for _ in range(volume // 2)]
        quantities = [random.randint(qty_lo, qty_hi) for _ in range(volume)]

        for (p, q) in zip(sorted(prices1 + prices2), quantities):
            if p > price:
                order = Order(round(p, 1), q, 'ask', None)
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

    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0):
        """
        Trader that is activated on call to perform action.

        :param market: link to exchange agent
        :param cash: trader's cash available
        :param assets: trader's number of shares hold
        """
        self.type = 'Unknown'
        self.name = f'Trader{self.id}'
        self.id = Trader.id
        Trader.id += 1

        self.market = market
        self.orders = list()

        self.cash = cash
        self.assets = assets

    def __str__(self) -> str:
        return f'{self.name} ({self.type})'

    def equity(self):
        price = self.market.price() if self.market.price() is not None else 0
        return self.cash + self.assets * price

    def _buy_limit(self, quantity, price):
        order = Order(round(price, 1), round(quantity), 'bid', self)
        self.orders.append(order)
        self.market.limit_order(order)

    def _sell_limit(self, quantity, price):
        order = Order(round(price, 1), round(quantity), 'ask', self)
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


class Random(Trader):
    """
    Random creates noisy orders to recreate trading in real environment.
    """
    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0):
        super().__init__(market, cash, assets)
        self.type = 'Random'

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
    """
    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0, access: int = 1):
        """
        :param market: exchange agent link
        :param cash: number of cash
        :param assets: number of assets
        :param access: number of future dividends informed
        """
        super().__init__(market, cash, assets)
        self.type = 'Fundamentalist'
        self.access = access

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
    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0):
        """
        :param market: exchange agent link
        :param cash: number of cash
        :param assets: number of assets
        """
        super().__init__(market, cash, assets)
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
        super().__init__(market, cash, assets)
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
    """

    def __init__(self, market: ExchangeAgent, cash: float, assets: int = 0, softlimit: int = 100):
        super().__init__(market, cash, assets)
        self.type = 'Market Maker'
        self.softlimit = softlimit
        self.ul = softlimit
        self.ll = -softlimit
        self.panic = False

    def call(self):
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
# FX-specific agents with multi-venue routing
# ---------------------------------------------------------------------------

class FXTrader(Trader):
    """
    Base FX liquidity-taker with multi-venue cost estimation and routing.
    Extends :class:`Trader` to inherit CLOB order management methods.

    Venue selection rule (§5.1):
        deterministic:  v*(Q) = argmin_v  Ĉ_v(Q)
        stochastic:     P(v=i) ∝ exp(−β Ĉ_i(Q))
    """

    def __init__(self,
                 clob: CLOBVenue,
                 amm_pools: Dict[str, CPMMPool | HFMMPool],
                 env: MarketEnvironment,
                 cash: float = 1e4,
                 beta: float = 1.0,
                 deterministic: bool = False):
        super().__init__(clob.exchange, cash, assets=0)
        self.type = 'FXTrader'
        self.clob = clob
        self.amm_pools = amm_pools   # {'cpmm': CPMMPool, 'hfmm': HFMMPool}
        self.env = env
        self.beta = beta              # logit sensitivity
        self.deterministic = deterministic
        self.base_holdings = 0.0
        self.trades: List[dict] = []

    # ---- cost estimation -------------------------------------------------

    def estimate_costs(self, Q: float, side: str = 'buy') -> Dict[str, float]:
        """Return dict {venue_name: cost_bps} for quantity Q."""
        S_t = self.clob.mid_price()
        costs: Dict[str, float] = {}
        costs['clob'] = self.clob.cost_bps(Q, side)
        for name, pool in self.amm_pools.items():
            try:
                if side == 'buy':
                    q = pool.quote_buy(Q, S_t)
                else:
                    q = pool.quote_sell(Q, S_t)
                costs[name] = q['cost_bps']
            except Exception:
                costs[name] = float('inf')
        return costs

    def choose_venue(self, Q: float, side: str = 'buy') -> str:
        """Select venue using deterministic argmin or softmax (logit) rule."""
        costs = self.estimate_costs(Q, side)
        if self.deterministic:
            return min(costs, key=costs.get)
        items = list(costs.items())
        min_c = min(c for _, c in items)
        weights = []
        for name, c in items:
            if c == float('inf'):
                weights.append(0.0)
            else:
                weights.append(_math.exp(-self.beta * (c - min_c)))
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

    # ---- execution -------------------------------------------------------

    def _classify(self, Q: float, venue: str) -> dict:
        """
        Compute relative trade size θ = Q/R and size bucket.

        R is always the **maximum AMM effective depth** across pools,
        so that "small/medium/large" reflects the trade's significance
        relative to AMM liquidity — the dimension we study.
        This makes θ comparable across venues and avoids the artefact
        where CLOB depth ≠ AMM depth produces different buckets for
        the same Q.
        """
        from AgentBasedModel.venues.amm import classify_trade_size
        try:
            mid = self.clob.mid_price()
        except Exception:
            return dict(theta=0.0, size_bucket='medium', R=1.0)

        # Reference depth R = max effective depth among AMM pools
        R = 1.0
        ref_pool = None
        for pool in self.amm_pools.values():
            d = pool.effective_depth(mid)
            if d > R:
                R = d
                ref_pool = pool

        if ref_pool is not None:
            return classify_trade_size(Q, ref_pool, mid)

        # Fallback if no AMM pools
        theta = Q / R
        if theta <= 0.01:
            bucket = 'small'
        elif theta <= 0.05:
            bucket = 'medium'
        else:
            bucket = 'large'
        return dict(theta=theta, size_bucket=bucket, R=R)

    def _execute_on_venue(self, venue: str, Q: float, side: str) -> dict:
        """Execute trade of *Q* base on *venue*.  Returns cost dict."""
        if venue == 'clob':
            return self._execute_clob(Q, side)
        pool = self.amm_pools[venue]
        if side == 'buy':
            return pool.execute_buy(Q)
        return pool.execute_sell(Q)

    def _execute_clob(self, Q: float, side: str) -> dict:
        """Execute a market order on the CLOB, reusing inherited helpers."""
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

    def call(self):
        pass


class FXNoiseTaker(FXTrader):
    """
    Random FX taker: each call, with some probability, generates a
    market order of random size and routes it to the cheapest venue.
    Extends :class:`FXTrader` (→ :class:`Trader`).
    """

    def __init__(self, clob, amm_pools, env,
                 cash=1e4, beta=1.0, deterministic=False,
                 trade_prob=0.3, q_min=1, q_max=5):
        super().__init__(clob, amm_pools, env, cash, beta, deterministic)
        self.type = 'FXNoiseTaker'
        self.trade_prob = trade_prob
        self.q_min = q_min
        self.q_max = q_max

    def call(self) -> Optional[dict]:
        if random.random() > self.trade_prob:
            return None
        side = 'buy' if random.random() > 0.5 else 'sell'
        Q = random.randint(self.q_min, self.q_max)
        venue = self.choose_venue(Q, side)
        cls_info = self._classify(Q, venue)
        result = self._execute_on_venue(venue, Q, side)
        trade_record = dict(
            trader_id=self.id, trader_type=self.type,
            venue=venue, side=side, quantity=Q,
            cost_bps=result.get('cost_bps', 0),
            theta=cls_info['theta'],
            size_bucket=cls_info['size_bucket'],
        )
        self.trades.append(trade_record)
        return trade_record


class FXFundamentalist(FXTrader):
    """
    Trades when CLOB price deviates from a fundamental exchange rate.
    Trade size proportional to mispricing.
    Extends :class:`FXTrader` (→ :class:`Trader`).
    """

    def __init__(self, clob, amm_pools, env,
                 cash=1e4, beta=1.0, deterministic=False,
                 fundamental_rate: float = 100.0,
                 gamma: float = 5e-3, q_max: int = 10):
        super().__init__(clob, amm_pools, env, cash, beta, deterministic)
        self.type = 'FXFundamentalist'
        self.fundamental_rate = fundamental_rate
        self.gamma = gamma
        self.q_max = q_max

    def call(self) -> Optional[dict]:
        try:
            mid = self.clob.mid_price()
        except Exception:
            return None
        mispricing = (self.fundamental_rate - mid) / mid
        if abs(mispricing) < self.gamma:
            return None
        side = 'buy' if mispricing > 0 else 'sell'
        Q = min(round(abs(mispricing) / self.gamma), self.q_max)
        if Q <= 0:
            return None
        venue = self.choose_venue(Q, side)
        cls_info = self._classify(Q, venue)
        result = self._execute_on_venue(venue, Q, side)
        trade_record = dict(
            trader_id=self.id, trader_type=self.type,
            venue=venue, side=side, quantity=Q,
            cost_bps=result.get('cost_bps', 0),
            theta=cls_info['theta'],
            size_bucket=cls_info['size_bucket'],
        )
        self.trades.append(trade_record)
        return trade_record


class FXRetailTaker(FXTrader):
    """
    Retail FX taker — generates small trades (θ ≤ 1% pool depth).
    Models retail flow that should find AMM adequate.
    """

    def __init__(self, clob, amm_pools, env,
                 cash=1e4, beta=1.0, deterministic=False,
                 trade_prob=0.4, q_min=1, q_max=2):
        super().__init__(clob, amm_pools, env, cash, beta, deterministic)
        self.type = 'FXRetailTaker'
        self.trade_prob = trade_prob
        self.q_min = q_min
        self.q_max = q_max

    def call(self) -> Optional[dict]:
        if random.random() > self.trade_prob:
            return None
        side = 'buy' if random.random() > 0.5 else 'sell'
        Q = random.randint(self.q_min, self.q_max)
        venue = self.choose_venue(Q, side)
        cls_info = self._classify(Q, venue)
        result = self._execute_on_venue(venue, Q, side)
        trade_record = dict(
            trader_id=self.id, trader_type=self.type,
            venue=venue, side=side, quantity=Q,
            cost_bps=result.get('cost_bps', 0),
            theta=cls_info['theta'],
            size_bucket=cls_info['size_bucket'],
        )
        self.trades.append(trade_record)
        return trade_record


class FXInstitutional(FXTrader):
    """
    Institutional/hedger FX taker — generates large trades (θ > 5% pool depth).
    Models institutional flow that should strongly prefer CLOB.
    """

    def __init__(self, clob, amm_pools, env,
                 cash=5e4, beta=1.0, deterministic=False,
                 trade_prob=0.15, q_min=15, q_max=50):
        super().__init__(clob, amm_pools, env, cash, beta, deterministic)
        self.type = 'FXInstitutional'
        self.trade_prob = trade_prob
        self.q_min = q_min
        self.q_max = q_max

    def call(self) -> Optional[dict]:
        if random.random() > self.trade_prob:
            return None
        side = 'buy' if random.random() > 0.5 else 'sell'
        Q = random.randint(self.q_min, self.q_max)
        venue = self.choose_venue(Q, side)
        cls_info = self._classify(Q, venue)
        result = self._execute_on_venue(venue, Q, side)
        trade_record = dict(
            trader_id=self.id, trader_type=self.type,
            venue=venue, side=side, quantity=Q,
            cost_bps=result.get('cost_bps', 0),
            theta=cls_info['theta'],
            size_bucket=cls_info['size_bucket'],
        )
        self.trades.append(trade_record)
        return trade_record


class FXMarketMaker(Trader):
    """
    Market maker for the CLOB.  Sets quoted spread and depth as functions
    of exogenous volatility σ_t and funding cost c_t (Brunnermeier–Pedersen).

        qspr_t = α₀ + α₁ σ_t + α₂ c_t
        depth_t = d₀ − d₁ σ_t − d₂ c_t   (floored at d_min)

    Extends :class:`Trader` for order lifecycle helpers.
    """

    def __init__(self, market: ExchangeAgent, env: MarketEnvironment,
                 cash: float = 1e4,
                 alpha0: float = 2.0, alpha1: float = 100.0,
                 alpha2: float = 50.0,
                 d0: float = 50.0, d1: float = 200.0, d2: float = 100.0,
                 d_min: float = 5.0):
        super().__init__(market, cash)
        self.type = 'FXMarketMaker'
        self.env = env
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.d_min = d_min

    def _target_spread_bps(self) -> float:
        return self.alpha0 + self.alpha1 * self.env.sigma + self.alpha2 * self.env.funding_cost

    def _target_depth(self) -> float:
        d = self.d0 - self.d1 * self.env.sigma - self.d2 * self.env.funding_cost
        return max(d, self.d_min)

    def call(self):
        """Cancel previous orders; post new bid/ask around mid."""
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
        depth = max(1, int(self._target_depth()))

        bid_price = round(mid - half_spread, 2)
        ask_price = round(mid + half_spread, 2)

        self._buy_limit(depth, bid_price)
        self._sell_limit(depth, ask_price)


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
    the CLOB reference price S_t (minus a fee-band tolerance).
    """

    def __init__(self, clob: CLOBVenue,
                 amm_pools: Dict[str, CPMMPool | HFMMPool]):
        self.clob = clob
        self.amm_pools = amm_pools
        self.type = 'AMMArbitrageur'

    def arbitrage(self):
        """For each AMM pool, align its price to S_t if profitable."""
        try:
            S_t = self.clob.mid_price()
        except Exception:
            return
        for name, pool in self.amm_pools.items():
            try:
                pool.arbitrage_to_target(S_t)
            except Exception:
                pass
