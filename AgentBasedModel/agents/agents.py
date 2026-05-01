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

    def __init__(self, price: Union[float, int] = 100, std: Union[float, int] = 25, volume: int = 1000,
                 rf: float = 5e-4, transaction_cost: float = 0,
                 price_tick: float = 0.005,
                 background_corridor_bps: float = 25.0,
                 background_target_ratio: float = 1.0,
                 background_max_share_of_trader_depth: float = 1.0,
                 anchor_strength: float = 1.0,
                 anchor_threshold_bps: float = 0.0):
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
        self._seed_price = float(price)
        self._seed_std = float(std)
        self._seed_volume = int(volume)
        self._price_tick = max(1e-4, float(price_tick))
        self._seed_qty_lo = max(1, volume // 500)
        self._seed_qty_hi = max(5, volume // 100)
        self._background_corridor_bps = max(5.0, float(background_corridor_bps))
        self._background_target_ratio = max(1.0, float(background_target_ratio))
        # Keep calm near-mid background support from collapsing with transient trader-depth decay.
        self._background_floor_ratio = 0.71
        self._background_max_share_of_trader_depth = max(0.0, float(background_max_share_of_trader_depth))
        self._anchor_strength = max(0.0, min(1.0, float(anchor_strength)))
        self._anchor_threshold_bps = max(0.0, float(anchor_threshold_bps))
        self._last_book_mid_price = float(price)
        self._fill_book(price, std, volume, rf * price)
        background_qty = self._background_qty_near_mid(
            float(price), self._background_corridor_bps
        )
        self._background_target_qty = {
            side: max(1.0, qty * self._background_target_ratio)
            for side, qty in background_qty.items()
        }

    def generate_dividend(self):
        """
        Generate time series on future dividends.
        """
        # Generate future dividend
        d = self.dividend_book[-1] * self._next_dividend()
        self.dividend_book.append(max(d, 0))  # dividend > 0
        self.dividend_book.pop(0)

    def expire_orders(self, fair_price: float = None):
        """Remove only truly expiring orders.

        TTL-tagged orders are short-lived by construction and should roll
        off the book naturally. The seeded anonymous book, however,
        represents persistent background liquidity; removing it simply
        because fair value drifted was collapsing the default CLOB depth.
        """

        for side in ('bid', 'ask'):
            expired = []
            for order in self.order_book[side]:
                # TTL-based expiry
                if order.ttl is not None:
                    if order.ttl <= 0:
                        expired.append(order)
                    else:
                        order.ttl -= 1
            for order in expired:
                if order.trader is not None and hasattr(order.trader, 'orders'):
                    try:
                        order.trader.orders.remove(order)
                    except ValueError:
                        pass
                    if hasattr(order.trader, 'release_order_reservation'):
                        order.trader.release_order_reservation(order)
                self.order_book[side].remove(order)

    def rebuild_order_book(self):
        for side in ('bid', 'ask'):
            orders = list(self.order_book[side])
            orders.sort(key=lambda order: order.price, reverse=(side == 'bid'))

            rebuilt = OrderList(side)
            for order in orders:
                order.left = None
                order.right = None
                rebuilt.append(order)

            self.order_book[side] = rebuilt

        if self.order_book['bid'] and self.order_book['ask']:
            best_bid = self.order_book['bid'].first.price
            best_ask = self.order_book['ask'].first.price

            if best_bid >= best_ask:
                tick = self._tick_size()
                mid = 0.5 * (best_bid + best_ask)
                target_bid = self.floor_price(mid - 0.5 * tick)
                target_ask = self.ceil_price(mid + 0.5 * tick)
                bid_shift = best_bid - target_bid
                ask_shift = target_ask - best_ask

                for order in self.order_book['bid']:
                    order.price = self.round_price(order.price - bid_shift)
                for order in self.order_book['ask']:
                    order.price = self.round_price(order.price + ask_shift)

    def _tick_size(self) -> float:
        return max(1e-4, float(getattr(self, '_price_tick', 0.01)))

    def round_price(self, price: float) -> float:
        tick = self._tick_size()
        return round(round(float(price) / tick) * tick, 3)

    def floor_price(self, price: float) -> float:
        tick = self._tick_size()
        return round(_math.floor(float(price) / tick + 1e-12) * tick, 3)

    def ceil_price(self, price: float) -> float:
        tick = self._tick_size()
        return round(_math.ceil(float(price) / tick - 1e-12) * tick, 3)

    def _background_qty_near_mid(self, reference_price: float, corridor_bps: float) -> dict:
        if reference_price <= 0:
            return {'bid': 0.0, 'ask': 0.0}

        bid_lo = reference_price * (1.0 - corridor_bps / 10_000.0)
        ask_hi = reference_price * (1.0 + corridor_bps / 10_000.0)
        return {
            'bid': sum(
                order.qty for order in self.order_book['bid']
                if order.trader is None and order.price >= bid_lo
            ),
            'ask': sum(
                order.qty for order in self.order_book['ask']
                if order.trader is None and order.price <= ask_hi
            ),
        }

    def _trader_qty_near_mid(self, reference_price: float, corridor_bps: float) -> dict:
        if reference_price <= 0:
            return {'bid': 0.0, 'ask': 0.0}

        bid_lo = reference_price * (1.0 - corridor_bps / 10_000.0)
        ask_hi = reference_price * (1.0 + corridor_bps / 10_000.0)
        return {
            'bid': sum(
                order.qty for order in self.order_book['bid']
                if order.trader is not None and order.price >= bid_lo
            ),
            'ask': sum(
                order.qty for order in self.order_book['ask']
                if order.trader is not None and order.price <= ask_hi
            ),
        }

    def _draw_background_price(self, side: str, reference_price: float,
                               corridor_bps: float,
                               aggressiveness: float = 1.0) -> float:
        tick = self._tick_size()
        max_offset = max(tick, reference_price * corridor_bps / 10_000.0)
        aggressiveness = max(0.25, min(1.0, float(aggressiveness)))
        widen_mult = 1.0 + 2.5 * (1.0 - aggressiveness)
        min_offset = tick * widen_mult
        scale = max(tick, max_offset / (3.0 * aggressiveness))
        offset = min(max_offset, min_offset + random.expovariate(1.0 / scale))

        if side == 'bid':
            return self.round_price(reference_price - offset)
        return self.round_price(reference_price + offset)

    def _draw_background_price_outside_corridor(self, side: str,
                                                reference_price: float,
                                                corridor_bps: float,
                                                aggressiveness: float = 1.0) -> float:
        tick = self._tick_size()
        aggressiveness = max(0.25, min(1.0, float(aggressiveness)))
        widen_mult = 1.0 + 1.5 * (1.0 - aggressiveness)
        min_offset = max(tick, reference_price * corridor_bps / 10_000.0 + tick * widen_mult)
        scale = max(tick, min_offset / (2.0 * aggressiveness))
        offset = min_offset + random.expovariate(1.0 / scale)

        if side == 'bid':
            return self.round_price(reference_price - offset)
        return self.round_price(reference_price + offset)

    def set_background_depth_target(self, reference_price: float,
                                    total_target_qty: float,
                                    corridor_bps: float = None,
                                    rebalance: bool = True,
                                    respect_trader_cap: bool = True):
        if reference_price is None or reference_price <= 0:
            return

        corridor_bps = corridor_bps or self._background_corridor_bps
        per_side = max(1.0, float(total_target_qty) / 2.0)
        self._background_target_qty = {'bid': per_side, 'ask': per_side}

        if rebalance:
            self.rebalance_background_liquidity(
                reference_price,
                corridor_bps=corridor_bps,
                target_ratio=1.0,
                respect_trader_cap=respect_trader_cap,
            )

    def rebalance_background_liquidity(self, reference_price: float,
                                       corridor_bps: float = None,
                                       target_ratio: float = 1.0,
                                       respect_trader_cap: bool = True):
        if reference_price is None or reference_price <= 0:
            return

        corridor_bps = corridor_bps or self._background_corridor_bps
        effective_aggressiveness = max(0.25, min(1.0, float(target_ratio)))
        trader_near_qty = self._trader_qty_near_mid(reference_price, corridor_bps)
        targets = {}
        for side in ('bid', 'ask'):
            scaled_target = max(1.0, self._background_target_qty.get(side, 0.0) * target_ratio)
            if respect_trader_cap:
                scaled_target = min(
                    scaled_target,
                    scaled_target * self._background_floor_ratio
                    + self._background_max_share_of_trader_depth * max(0.0, trader_near_qty.get(side, 0.0))
                )
            targets[side] = scaled_target
        changed = False

        bid_lo = reference_price * (1.0 - corridor_bps / 10_000.0)
        ask_hi = reference_price * (1.0 + corridor_bps / 10_000.0)

        for side in ('bid', 'ask'):
            near_qty = 0.0
            near_orders = []
            far_orders = []

            for order in self.order_book[side]:
                if order.trader is not None:
                    continue

                in_corridor = order.price >= bid_lo if side == 'bid' else order.price <= ask_hi
                if in_corridor:
                    near_qty += order.qty
                    near_orders.append(order)
                else:
                    far_orders.append(order)

            near_orders.sort(key=lambda order: abs(order.price - reference_price), reverse=True)
            far_orders.sort(key=lambda order: abs(order.price - reference_price), reverse=True)

            while near_qty > targets[side] and near_orders:
                order = near_orders.pop(0)
                excess = near_qty - targets[side]
                move_qty = min(order.qty, excess)
                if move_qty <= 0:
                    break
                if move_qty + 1e-9 < order.qty:
                    order.qty -= move_qty
                    moved = Order(
                        self._draw_background_price_outside_corridor(
                            side,
                            reference_price,
                            corridor_bps,
                            aggressiveness=effective_aggressiveness,
                        ),
                        move_qty,
                        side,
                        None,
                    )
                    self.order_book[side].append(moved)
                else:
                    order.price = self._draw_background_price_outside_corridor(
                        side,
                        reference_price,
                        corridor_bps,
                        aggressiveness=effective_aggressiveness,
                    )
                near_qty -= move_qty
                changed = True

            while near_qty < targets[side] and far_orders:
                order = far_orders.pop(0)
                gap = targets[side] - near_qty
                move_qty = min(order.qty, gap)
                if move_qty <= 0:
                    break
                if move_qty + 1e-9 < order.qty:
                    order.qty -= move_qty
                    moved = Order(
                        self._draw_background_price(
                            side,
                            reference_price,
                            corridor_bps,
                            aggressiveness=effective_aggressiveness,
                        ),
                        move_qty,
                        side,
                        None,
                    )
                    self.order_book[side].append(moved)
                else:
                    order.price = self._draw_background_price(
                        side,
                        reference_price,
                        corridor_bps,
                        aggressiveness=effective_aggressiveness,
                    )
                near_qty += move_qty
                changed = True

            while near_qty < targets[side]:
                gap = targets[side] - near_qty
                qty = min(random.randint(self._seed_qty_lo, self._seed_qty_hi), gap)
                if qty <= 0:
                    break
                price = self._draw_background_price(
                    side,
                    reference_price,
                    corridor_bps,
                    aggressiveness=effective_aggressiveness,
                )
                self.order_book[side].append(Order(price, qty, side, None))
                near_qty += qty
                changed = True

        if changed:
            self.rebuild_order_book()

    def recenter_book(self, fair_price: float, reprice_prob: float = 0.6,
                      reprice_noise_bps: float = 0.5,
                      anchor_strength: float = None,
                      min_reprice_gap_bps: float = None,
                      background_target_ratio: float = 1.0):
        """Probabilistically shift resting orders toward *fair_price*.

        The book closes only a fraction of the gap to the anchor on each
        tick, creating a realistic lag between latent fair value and the
        observed CLOB mid.

        Each trader-owned order reprices with probability *reprice_prob*,
        creating a natural lag between fair-price moves and book adjustment.
        Repriced orders receive a small noise term to prevent
        artificial clustering.

        Parameters
        ----------
        fair_price : float
            Target mid-price (from GBM or post-shock).
        reprice_prob : float
            Per-order probability of adjusting this tick (0–1).
        reprice_noise_bps : float
            Std of Gaussian noise added to repriced orders (in bps
            of *fair_price*).
        anchor_strength : float or None
            Fraction of the gap between the current book mid and
            *fair_price* closed this tick.  If None, uses the exchange
            default.
        min_reprice_gap_bps : float or None
            Ignore tiny deviations below this threshold.  If None,
            uses the exchange default.
        """
        if fair_price is None or fair_price <= 0:
            return
        sp = self.spread()
        if sp is None:
            # After a shock cancel-wave one side can be empty. Re-seed the
            # anonymous background around fair value so the book can recover.
            self.rebalance_background_liquidity(fair_price, target_ratio=background_target_ratio)
            return
        book_mid = (sp['bid'] + sp['ask']) / 2.0
        anchor_strength = self._anchor_strength if anchor_strength is None else max(0.0, min(1.0, anchor_strength))
        min_reprice_gap_bps = (
            self._anchor_threshold_bps
            if min_reprice_gap_bps is None else max(0.0, min_reprice_gap_bps)
        )
        gap_bps = abs(fair_price - book_mid) / max(book_mid, 1e-9) * 10_000.0
        if anchor_strength <= 0.0:
            self.rebalance_background_liquidity(book_mid, target_ratio=background_target_ratio)
            return
        if gap_bps < min_reprice_gap_bps:
            self.rebalance_background_liquidity(book_mid, target_ratio=background_target_ratio)
            return
        target_mid = book_mid + (fair_price - book_mid) * anchor_strength
        dp = target_mid - book_mid
        noise_std = max(target_mid, 1e-9) * reprice_noise_bps / 10_000.0
        repriced = False
        for side in ('bid', 'ask'):
            for order in self.order_book[side]:
                if order.trader is None:
                    order.price = self.round_price(order.price + dp)
                    repriced = True
                elif random.random() < reprice_prob:
                    noise = random.gauss(0, noise_std) if noise_std > 0 else 0.0
                    new_price = self.round_price(order.price + dp + noise)
                    if hasattr(order.trader, 'adjust_order_price'):
                        if not order.trader.adjust_order_price(order, new_price, self.transaction_cost):
                            continue
                    else:
                        order.price = new_price
                    repriced = True

        if repriced:
            self.rebuild_order_book()
            self.rebalance_background_liquidity(target_mid, target_ratio=background_target_ratio)

    def cancel_wave(self, cancel_frac: float = 0.5, near_touch: bool = True):
        """Cancel a fraction of resting orders (liquidity crisis).

        Parameters
        ----------
        cancel_frac : float
            Fraction of orders to cancel (0–1).
        near_touch : bool
            If True, priority is given to orders near the top of book
            (most aggressive), which is realistic — market-makers and
            aggressive limit orders withdraw first in a crisis.
        """
        for side in ('bid', 'ask'):
            orders = list(self.order_book[side])
            if not orders:
                continue
            n_cancel = max(1, int(len(orders) * cancel_frac))
            if near_touch:
                # Cancel from the top (most aggressive) first
                to_cancel = orders[:n_cancel]
            else:
                to_cancel = random.sample(orders, min(n_cancel, len(orders)))
            for order in to_cancel:
                if order.trader is not None and hasattr(order.trader, 'orders'):
                    try:
                        order.trader.orders.remove(order)
                    except ValueError:
                        pass
                    if hasattr(order.trader, 'release_order_reservation'):
                        order.trader.release_order_reservation(order)
                self.order_book[side].remove(order)

    def _fill_book(self, price, std, volume, div: float = 0.05):
        """
        Fill order book with random orders. Fill dividend book with n future dividends.
        """
        # Order book — qty range scales with volume for deeper books
        qty_lo = max(1, volume // 500)
        qty_hi = max(5, volume // 100)
        prices1 = [self.round_price(random.normalvariate(price - std, std)) for _ in range(volume // 2)]
        prices2 = [self.round_price(random.normalvariate(price + std, std)) for _ in range(volume // 2)]
        quantities = [random.randint(qty_lo, qty_hi) for _ in range(volume)]

        for (p, q) in zip(sorted(prices1 + prices2), quantities):
            if p > price:
                order = Order(self.round_price(p), q, 'ask', None)
                self.order_book['ask'].append(order)
            else:
                order = Order(self.round_price(p), q, 'bid', None)
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
            bid = self.order_book['bid'].first.price
            ask = self.order_book['ask'].first.price
            self._last_book_mid_price = 0.5 * (bid + ask)
            return {'bid': bid, 'ask': ask}
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
        if order.trader is not None and hasattr(order.trader, 'prepare_limit_order'):
            if not order.trader.prepare_limit_order(order, self.transaction_cost):
                return False

        sp = self.spread()
        if sp is None:
            # Book empty — insert directly (no matching possible)
            if order.order_type == 'bid':
                self.order_book['bid'].insert(order)
            elif order.order_type == 'ask':
                self.order_book['ask'].insert(order)
            return True

        bid, ask = sp.values()
        t_cost = self.transaction_cost
        if not bid or not ask:
            return False

        if order.order_type == 'bid':
            if order.price >= ask:
                order = self.order_book['ask'].fulfill(order, t_cost)
            if order.qty > 0:
                self.order_book['bid'].insert(order)
            return True

        elif order.order_type == 'ask':
            if order.price <= bid:
                order = self.order_book['bid'].fulfill(order, t_cost)
            if order.qty > 0:
                self.order_book['ask'].insert(order)
            return True
        return False

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
        if order.trader is not None and hasattr(order.trader, 'release_order_reservation'):
            order.trader.release_order_reservation(order)


class Trader:
    id = 0

    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0,
                 clob: CLOBVenue = None,
                 amm_pools: Dict[str, CPMMPool | HFMMPool] = None,
                 env: MarketEnvironment = None,
                 amm_share_pct: float = 25.0,
                 venue_choice_rule: str = 'fixed_share',
                 deterministic_venue: bool = False,
                 beta_amm: float = 0.05,
                 cpmm_bias_bps: float = 5.0,
                 cost_noise_std: float = 1.5,
                 max_cash_borrow: float = 0.0,
                 max_short_assets: float = 0.0):
        """
        Trader that is activated on call to perform action.

        :param market: link to exchange agent
        :param cash: trader's cash available
        :param assets: trader's number of shares hold
        :param clob: CLOBVenue wrapper (enables multi-venue mode)
        :param amm_pools: dict of AMM pools (enables multi-venue mode)
        :param env: MarketEnvironment for σ_t, c_t
        :param amm_share_pct: target probability (0–100) of routing to AMM vs CLOB
        :param venue_choice_rule: routing regime, either fixed_share or liquidity_aware
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
        self.max_cash_borrow = max(0.0, float(max_cash_borrow))
        self.max_short_assets = max(0.0, float(max_short_assets))
        self.reserved_cash = 0.0
        self.reserved_assets = 0.0
        self.maintenance_margin_ratio = 0.0
        self.liquidation_fraction = 0.5
        self.borrow_spread_multiplier = 1.0
        self.short_borrow_spread_multiplier = 1.25
        self.defaulted = False
        self.default_reason: Optional[str] = None
        self.last_financing_charge = 0.0
        self.cumulative_financing_charge = 0.0

        # Multi-venue (optional)
        self.clob = clob
        self.amm_pools = amm_pools or {}
        self.env = env
        self.amm_share_pct = max(0.0, min(100.0, amm_share_pct))
        self.venue_choice_rule = (
            venue_choice_rule if venue_choice_rule in {'fixed_share', 'liquidity_aware'}
            else 'fixed_share'
        )
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

    def equity(self, reference_price: Optional[float] = None):
        price = reference_price
        if price is None:
            price = self.market.price() if self.market.price() is not None else 0
        return self.cash + self.assets * price

    def gross_exposure(self, reference_price: Optional[float] = None) -> float:
        price = reference_price
        if price is None:
            price = self.market.price() if self.market.price() is not None else 0.0
        price = max(0.0, float(price))
        borrowed_cash = max(0.0, -self.cash)
        return borrowed_cash + abs(self.assets) * price

    def maintenance_margin_required(self, reference_price: Optional[float] = None) -> float:
        return max(0.0, self.maintenance_margin_ratio) * self.gross_exposure(reference_price)

    def has_margin_breach(self, reference_price: Optional[float] = None) -> bool:
        return self.equity(reference_price) + 1e-9 < self.maintenance_margin_required(reference_price)

    def cancel_all_resting_orders(self):
        for order in self.orders.copy():
            try:
                self._cancel_order(order)
            except Exception:
                pass

    def apply_financing_charge(self, reference_price: Optional[float], funding_cost: float) -> float:
        if self.defaulted:
            self.last_financing_charge = 0.0
            return 0.0

        ref_price = max(0.0, float(reference_price or 0.0))
        base_rate = max(0.0, float(funding_cost))
        borrowed_cash = max(0.0, -self.cash)
        short_notional = max(0.0, -self.assets) * ref_price
        charge = borrowed_cash * base_rate * max(0.0, self.borrow_spread_multiplier)
        charge += short_notional * base_rate * max(0.0, self.short_borrow_spread_multiplier)
        if charge > 0.0:
            self.cash -= charge
            self.cumulative_financing_charge += charge
        self.last_financing_charge = charge
        return charge

    def _mark_default(self, reason: str):
        self.cancel_all_resting_orders()
        self.defaulted = True
        self.default_reason = reason

    def enforce_balance_sheet_discipline(self, reference_price: Optional[float] = None) -> dict:
        if self.defaulted:
            return {'status': 'defaulted', 'reason': self.default_reason}

        ref_price = reference_price
        if ref_price is None:
            ref_price = self.market.price() if self.market.price() is not None else 0.0
        ref_price = max(0.0, float(ref_price))

        hard_cash_breach = self.cash < -self.max_cash_borrow - 1e-9
        hard_short_breach = self.assets < -self.max_short_assets - 1e-9
        margin_breach = self.has_margin_breach(ref_price)
        if not (hard_cash_breach or hard_short_breach or margin_breach):
            return {'status': 'ok'}

        self.cancel_all_resting_orders()

        best_bid = None
        best_ask = None
        try:
            if self.market.order_book['bid']:
                best_bid = self.market.order_book['bid'].first.price
            if self.market.order_book['ask']:
                best_ask = self.market.order_book['ask'].first.price
        except Exception:
            pass

        if self.assets > 0.0 and (hard_cash_breach or margin_breach):
            cash_target = -0.8 * self.max_cash_borrow
            cash_gap = max(0.0, cash_target - self.cash)
            bid_ref = max(1e-9, float(best_bid if best_bid is not None else ref_price or 1.0))
            qty_for_cash = int(_math.ceil(cash_gap / bid_ref)) if cash_gap > 0 else 0
            qty_for_margin = int(_math.ceil(max(0.0, self.assets) * max(0.0, self.liquidation_fraction))) if margin_breach else 0
            qty_to_sell = max(qty_for_cash, qty_for_margin)
            if qty_to_sell > 0:
                self._execute_clob(qty_to_sell, 'sell')

        if self.assets < 0.0 and (hard_short_breach or margin_breach):
            short_excess = max(0.0, abs(self.assets) - 0.8 * self.max_short_assets)
            qty_for_short = int(_math.ceil(short_excess)) if short_excess > 0 else 0
            qty_for_margin = int(_math.ceil(abs(self.assets) * max(0.0, self.liquidation_fraction))) if margin_breach else 0
            qty_to_buy = max(qty_for_short, qty_for_margin)
            if qty_to_buy > 0:
                self._execute_clob(qty_to_buy, 'buy')

        if self.cash < -self.max_cash_borrow - 1e-9:
            self._mark_default('cash_limit_breach')
        elif self.assets < -self.max_short_assets - 1e-9:
            self._mark_default('short_limit_breach')
        elif self.has_margin_breach(ref_price):
            self._mark_default('maintenance_margin_breach')

        return {
            'status': 'defaulted' if self.defaulted else 'stabilized',
            'reason': self.default_reason,
            'cash': self.cash,
            'assets': self.assets,
        }

    def available_cash(self) -> float:
        return self.cash + self.max_cash_borrow - self.reserved_cash

    def available_assets(self) -> float:
        return self.assets + self.max_short_assets - self.reserved_assets

    def prepare_limit_order(self, order: Order, t_cost: float) -> bool:
        if order.trader is not self:
            return False
        if getattr(order, 'reserved_cash', 0.0) > 0.0 or getattr(order, 'reserved_assets', 0.0) > 0.0:
            return True
        if order.qty <= 0:
            return False
        if order.order_type == 'bid':
            reserve = max(0.0, order.price * order.qty * (1.0 + t_cost))
            if reserve > self.available_cash() + 1e-9:
                return False
            order.reserved_cash = reserve
            self.reserved_cash += reserve
            return True
        reserve = float(order.qty)
        if reserve > self.available_assets() + 1e-9:
            return False
        order.reserved_assets = reserve
        self.reserved_assets += reserve
        return True

    def release_order_reservation(self, order: Order):
        cash_release = min(getattr(order, 'reserved_cash', 0.0), self.reserved_cash)
        asset_release = min(getattr(order, 'reserved_assets', 0.0), self.reserved_assets)
        if cash_release > 0.0:
            self.reserved_cash -= cash_release
        if asset_release > 0.0:
            self.reserved_assets -= asset_release
        order.reserved_cash = 0.0
        order.reserved_assets = 0.0

    def adjust_order_price(self, order: Order, new_price: float, t_cost: float) -> bool:
        if order.trader is not self:
            return False
        if order.order_type != 'bid':
            order.price = new_price
            return True
        new_reserve = max(0.0, new_price * order.qty * (1.0 + t_cost))
        delta = new_reserve - getattr(order, 'reserved_cash', 0.0)
        if delta > self.available_cash() + 1e-9:
            return False
        self.reserved_cash += delta
        order.reserved_cash = new_reserve
        order.price = new_price
        return True

    def apply_fill(self, order: Order, fill_qty: float, fill_price: float,
                   t_cost: float, is_buy: bool, qty_before: float):
        if qty_before > 0.0 and getattr(order, 'reserved_cash', 0.0) > 0.0:
            release = min(order.reserved_cash, order.reserved_cash * fill_qty / qty_before)
            order.reserved_cash -= release
            self.reserved_cash = max(0.0, self.reserved_cash - release)
        if qty_before > 0.0 and getattr(order, 'reserved_assets', 0.0) > 0.0:
            release = min(order.reserved_assets, order.reserved_assets * fill_qty / qty_before)
            order.reserved_assets -= release
            self.reserved_assets = max(0.0, self.reserved_assets - release)

        gross = fill_price * fill_qty
        if is_buy:
            self.cash -= gross * (1.0 + t_cost)
            self.assets += fill_qty
        else:
            self.cash += gross * (1.0 - t_cost)
            self.assets -= fill_qty

    def _clob_venue(self):
        if self.clob is not None:
            return self.clob
        from AgentBasedModel.venues.clob import CLOBVenue
        return CLOBVenue(self.market)

    def _affordable_limit_buy_qty(self, quantity: float, price: float) -> int:
        unit_cost = price * (1.0 + self.market.transaction_cost)
        if unit_cost <= 0:
            return 0
        return max(0, min(int(round(quantity)), int(self.available_cash() // unit_cost)))

    def _sellable_limit_qty(self, quantity: float) -> int:
        return max(0, min(int(round(quantity)), int(self.available_assets())))

    def _affordable_market_buy_qty(self, quantity: float) -> int:
        target = max(0, int(round(quantity)))
        if target <= 0:
            return 0
        venue = self._clob_venue()
        lo, hi, best = 0, target, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if mid <= 0:
                lo = 1
                continue
            quote = venue.quote_buy(mid)
            cost = quote.get('exec_price', float('inf')) * mid * (1.0 + self.market.transaction_cost)
            feasible = quote.get('cost_bps', float('inf')) != float('inf') and cost <= self.available_cash() + 1e-9
            if feasible:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _sellable_market_qty(self, quantity: float) -> int:
        target = max(0, int(round(quantity)))
        return max(0, min(target, int(self.available_assets())))

    def _clip_amm_buy_qty(self, pool, quantity: float) -> float:
        target = max(0.0, float(quantity))
        if target <= 0.0:
            return 0.0
        available_cash = self.available_cash()
        if available_cash <= 0.0:
            return 0.0
        hi = min(target, max(0.0, pool.x * 0.949))
        if hi <= 0.0:
            return 0.0
        hi_quote = pool.quote_buy(hi)
        if hi_quote.get('cost_bps', float('inf')) != float('inf') and hi_quote.get('delta_y', float('inf')) <= available_cash:
            return hi
        lo = 0.0
        best = 0.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if mid <= 0.0:
                break
            quote = pool.quote_buy(mid)
            feasible = quote.get('cost_bps', float('inf')) != float('inf') and quote.get('delta_y', float('inf')) <= available_cash
            if feasible:
                best = mid
                lo = mid
            else:
                hi = mid
        return best

    def _clip_amm_sell_qty(self, quantity: float) -> float:
        return max(0.0, min(float(quantity), self.available_assets()))

    def _buy_limit(self, quantity, price):
        if self.defaulted:
            return 0
        quote_price = self.market.round_price(price) if hasattr(self.market, 'round_price') else round(price, 2)
        quantity = self._affordable_limit_buy_qty(quantity, quote_price)
        if quantity <= 0:
            return 0
        order = Order(quote_price, quantity, 'bid', self)
        accepted = self.market.limit_order(order)
        if accepted and order.qty > 0:
            self.orders.append(order)
        return order.qty

    def _sell_limit(self, quantity, price):
        if self.defaulted:
            return 0
        quantity = self._sellable_limit_qty(quantity)
        if quantity <= 0:
            return 0
        quote_price = self.market.round_price(price) if hasattr(self.market, 'round_price') else round(price, 2)
        order = Order(quote_price, quantity, 'ask', self)
        accepted = self.market.limit_order(order)
        if accepted and order.qty > 0:
            self.orders.append(order)
        return order.qty

    def _buy_market(self, quantity) -> int:
        """
        :return: quantity unfulfilled
        """
        if self.defaulted:
            return 0
        quantity = self._affordable_market_buy_qty(quantity)
        if quantity <= 0:
            return 0
        if not self.market.order_book['ask']:
            return 0
        order = Order(self.market.order_book['ask'].last.price, round(quantity), 'bid', self)
        return self.market.market_order(order).qty

    def _sell_market(self, quantity) -> int:
        """
        :return: quantity unfulfilled
        """
        if self.defaulted:
            return 0
        quantity = self._sellable_market_qty(quantity)
        if quantity <= 0:
            return 0
        if not self.market.order_book['bid']:
            return 0
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

    def _perceived_costs(self, costs: Dict[str, float]) -> Dict[str, float]:
        """Apply venue-specific bias and estimation noise to routing costs."""
        perceived: Dict[str, float] = {}
        for name, cost in costs.items():
            adjusted = cost
            if name == 'cpmm' and cost != float('inf'):
                adjusted -= self.cpmm_bias_bps
            if name != 'clob' and self.cost_noise_std > 0 and cost != float('inf'):
                adjusted += random.gauss(0, self.cost_noise_std)
            perceived[name] = adjusted
        return perceived

    def _liquidity_aware_pick(self, costs: Dict[str, float], Q: float) -> str:
        """Route via an endogenous CLOB-vs-AMM group choice, then pick an AMM pool."""
        amm_names = [name for name in costs if name != 'clob']
        if not amm_names:
            return 'clob'

        perceived = self._perceived_costs(costs)
        try:
            ref_mid = self.clob.mid_price()
        except Exception:
            ref_mid = float('nan')

        try:
            depth_row = self.clob.total_depth(25)
            clob_depth = min(depth_row['bid'], depth_row['ask'])
        except Exception:
            clob_depth = 0.0

        pool_depths: Dict[str, float] = {}
        pool_alignment: Dict[str, float] = {}
        for name in amm_names:
            pool = self.amm_pools[name]
            try:
                depth = pool.effective_depth(ref_mid) if _math.isfinite(ref_mid) else pool.effective_depth()
            except Exception:
                depth = 0.0
            pool_depths[name] = max(0.0, depth)

            alignment_score = 1.0
            try:
                pool_mid = pool.mid_price()
                if _math.isfinite(ref_mid) and ref_mid > 0 and _math.isfinite(pool_mid):
                    basis_bps = abs(pool_mid - ref_mid) / ref_mid * 10_000.0
                    alignment_score = 1.0 / (1.0 + basis_bps / 20.0)
            except Exception:
                alignment_score = 1.0
            pool_alignment[name] = alignment_score

        finite_amm = [name for name in amm_names if _math.isfinite(perceived[name]) and perceived[name] != float('inf')]
        best_amm_name = min(finite_amm, key=lambda name: perceived[name]) if finite_amm else amm_names[0]

        group_costs = {
            'clob': perceived.get('clob', float('inf')),
            'amm': perceived.get(best_amm_name, float('inf')),
        }
        finite_group_costs = [cost for cost in group_costs.values() if _math.isfinite(cost) and cost != float('inf')]
        best_group_cost = min(finite_group_costs) if finite_group_costs else 0.0

        amm_target = max(0.0, min(1.0, self.amm_share_pct / 100.0))
        cost_span = (
            max(finite_group_costs) - min(finite_group_costs)
            if len(finite_group_costs) >= 2 else 0.0
        )
        liquidity_factor = getattr(self.env, 'systemic_liquidity', 1.0) if self.env is not None else 1.0
        venue_basis_bps = getattr(self.env, 'venue_basis_bps', 0.0) if self.env is not None else 0.0
        arbitrage_capacity = getattr(self.env, 'arbitrage_capacity', 1.0) if self.env is not None else 1.0
        prior_mix = 0.08 / (1.0 + cost_span / 10.0)
        prior_mix *= 0.75 + 0.25 * max(0.0, min(1.0, liquidity_factor))
        prior_mix /= 1.0 + max(0.0, venue_basis_bps) / 25.0
        prior_mix = max(0.0, min(0.08, prior_mix))

        neutral_priors = {'clob': 0.5, 'amm': 0.5}
        target_priors = {'clob': 1.0 - amm_target, 'amm': amm_target}
        prior_scores = {}
        for name in ('clob', 'amm'):
            eff_prior = (1.0 - prior_mix) * neutral_priors[name] + prior_mix * target_priors[name]
            prior_scores[name] = eff_prior / neutral_priors[name]

        amm_depth = sum(pool_depths.values())
        amm_alignment = pool_alignment.get(best_amm_name, 1.0)
        group_depths = {'clob': clob_depth, 'amm': amm_depth}
        group_alignment = {'clob': 1.0, 'amm': amm_alignment}
        amm_state_score = 1.0
        if self.env is not None:
            liquidity_component = 0.80 + 0.20 * max(0.0, min(1.0, liquidity_factor))
            arbitrage_component = 0.65 + 0.35 * max(0.0, min(1.0, arbitrage_capacity))
            basis_component = 1.0 / (1.0 + max(0.0, venue_basis_bps) / 30.0)
            amm_state_score = liquidity_component * arbitrage_component * basis_component
            amm_state_score = max(0.20, min(1.0, amm_state_score))
        group_state_scores = {'clob': 1.0, 'amm': amm_state_score}

        group_scores: Dict[str, float] = {}
        for name, cost in group_costs.items():
            if cost == float('inf') or not _math.isfinite(cost):
                group_scores[name] = 0.0
                continue
            rel_cost = max(0.0, cost - best_group_cost)
            cost_score = _math.exp(-rel_cost / 10.0)
            depth = group_depths[name]
            depth_score = min(max(depth, 0.0) / max(10.0 * Q, 1.0), 2.0)
            liquidity_score = 0.4 + 0.6 * min(depth_score, 1.0)
            group_scores[name] = (
                prior_scores[name]
                * cost_score
                * liquidity_score
                * group_alignment[name]
                * group_state_scores[name]
            )

        if self._weighted_pick(group_scores) == 'clob':
            return 'clob'

        amm_scores: Dict[str, float] = {}
        best_amm_cost = min(
            perceived[name] for name in amm_names
            if _math.isfinite(perceived[name]) and perceived[name] != float('inf')
        ) if finite_amm else 0.0
        for name in amm_names:
            cost = perceived[name]
            if cost == float('inf') or not _math.isfinite(cost):
                amm_scores[name] = 0.0
                continue
            rel_cost = max(0.0, cost - best_amm_cost)
            cost_score = _math.exp(-rel_cost / 10.0)
            depth_score = min(pool_depths.get(name, 0.0) / max(8.0 * Q, 1.0), 2.0)
            liquidity_score = 0.4 + 0.6 * min(depth_score, 1.0)
            amm_scores[name] = cost_score * liquidity_score * pool_alignment.get(name, 1.0)

        return self._weighted_pick(amm_scores)

    def choose_venue(self, Q: float, side: str = 'buy') -> str:
        """Venue selection under a fixed-share or liquidity-aware routing rule.

        fixed_share:
            Step 1 — CLOB vs AMM with fixed probability ``amm_share_pct``.
            Step 2 — within AMM, CPMM vs HFMM (softmax with β_amm),
                using bias-adjusted and noise-perturbed costs.

        liquidity_aware:
            Route probabilistically across all venues using perceived cost,
            executable depth, price alignment, and a soft prior derived from
            ``amm_share_pct``.

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

        if self.venue_choice_rule == 'liquidity_aware':
            return self._liquidity_aware_pick(costs, Q)

        # ── Step 1: CLOB vs AMM (direct probability) ─────────────────
        if random.random() * 100.0 >= self.amm_share_pct:
            return 'clob'

        # ── Step 2: within AMM — CPMM vs HFMM ────────────────────────
        amm_raw_costs = {n: costs[n] for n in amm_names}
        adj_costs = {n: self._perceived_costs({n: c})[n] for n, c in amm_raw_costs.items()}

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

    @staticmethod
    def _weighted_pick(scores: Dict[str, float]) -> str:
        """Pick a key from positive score weights."""
        items = list(scores.items())
        total = sum(max(0.0, score) for _, score in items)
        if total <= 0.0:
            return items[0][0]
        r = random.random() * total
        cumulative = 0.0
        for name, score in items:
            cumulative += max(0.0, score)
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
        if self.defaulted:
            venue_obj = self._clob_venue() if venue == 'clob' else self.amm_pools.get(venue)
            result = venue_obj._inf_quote() if venue_obj is not None and hasattr(venue_obj, '_inf_quote') else {'cost_bps': float('inf')}
            result['executed_qty'] = 0.0
            result['requested_qty'] = Q
            return result
        if venue == 'clob':
            return self._execute_clob(Q, side)
        pool = self.amm_pools[venue]
        if side == 'buy':
            exec_q = self._clip_amm_buy_qty(pool, Q)
            if exec_q <= 0.0:
                result = pool._inf_quote()
                result['executed_qty'] = 0.0
                result['requested_qty'] = Q
                return result
            result = pool.execute_buy(exec_q)
            self.cash -= result.get('delta_y', 0.0)
            self.assets += exec_q
        else:
            exec_q = self._clip_amm_sell_qty(Q)
            if exec_q <= 0.0:
                result = pool._inf_quote()
                result['executed_qty'] = 0.0
                result['requested_qty'] = Q
                return result
            result = pool.execute_sell(exec_q)
            self.cash += result.get('delta_y', 0.0)
            self.assets -= exec_q
        result['executed_qty'] = exec_q
        result['requested_qty'] = Q
        return result

    def _execute_clob(self, Q: float, side: str) -> dict:
        """Market order on CLOB with cost quoting."""
        venue = self._clob_venue()
        try:
            if self.defaulted:
                quote_info = venue._inf_quote()
                quote_info['executed_qty'] = 0
                quote_info['requested_qty'] = Q
                return quote_info
            if side == 'buy':
                exec_q = self._affordable_market_buy_qty(Q)
                if exec_q <= 0:
                    quote_info = venue._inf_quote()
                    quote_info['executed_qty'] = 0
                    quote_info['requested_qty'] = Q
                    return quote_info
                quote_info = venue.quote_buy(exec_q)
                if quote_info['cost_bps'] == float('inf'):
                    quote_info['executed_qty'] = 0
                    quote_info['requested_qty'] = Q
                    return quote_info
                unfilled = self._buy_market(exec_q)
                quote_info['executed_qty'] = max(0, exec_q - unfilled)
                quote_info['requested_qty'] = Q
                return quote_info
            else:
                exec_q = self._sellable_market_qty(Q)
                if exec_q <= 0:
                    quote_info = venue._inf_quote()
                    quote_info['executed_qty'] = 0
                    quote_info['requested_qty'] = Q
                    return quote_info
                quote_info = venue.quote_sell(exec_q)
                if quote_info['cost_bps'] == float('inf'):
                    quote_info['executed_qty'] = 0
                    quote_info['requested_qty'] = Q
                    return quote_info
                unfilled = self._sell_market(exec_q)
                quote_info['executed_qty'] = max(0, exec_q - unfilled)
                quote_info['requested_qty'] = Q
                return quote_info
        except Exception:
            quote_info = venue._inf_quote()
            quote_info['executed_qty'] = 0
            quote_info['requested_qty'] = Q
            return quote_info

    def _make_trade_record(self, venue: str, side: str, Q: float,
                           result: dict, cls_info: dict) -> dict:
        """Build a trade record dict for logging."""
        return dict(
            trader_id=self.id, trader_type=self.type,
            venue=venue, side=side, quantity=Q,
            requested_quantity=result.get('requested_qty', Q),
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
                 venue_choice_rule: str = 'fixed_share',
                 deterministic_venue: bool = False,
                 beta_amm: float = 0.05,
                 cpmm_bias_bps: float = 5.0,
                 cost_noise_std: float = 1.5,
                 # FX-noise-taker params (only used in multi-venue mode)
                 trade_prob: float = 0.3,
                 q_min: int = 1,
                 q_max: int = 5,
                 label: str = 'Random',
                 flow_role: Optional[str] = None,
                 flow_persistence: float = 0.25,
                 session_sensitivity: float = 1.0):
        super().__init__(market, cash, assets,
                         clob=clob, amm_pools=amm_pools, env=env,
                         amm_share_pct=amm_share_pct, venue_choice_rule=venue_choice_rule,
                         deterministic_venue=deterministic_venue,
                         beta_amm=beta_amm, cpmm_bias_bps=cpmm_bias_bps,
                         cost_noise_std=cost_noise_std)
        self.type = label if self.multi_venue else 'Random'
        self.trade_prob = trade_prob
        self.q_min = q_min
        self.q_max = q_max
        self.flow_role = flow_role or (label if self.multi_venue else 'Random')
        self.flow_persistence = max(0.0, min(1.0, float(flow_persistence)))
        self.session_sensitivity = max(0.0, float(session_sensitivity))
        self._last_fx_side: Optional[str] = None
        self._flow_program_side: Optional[str] = None
        self._flow_program_remaining: int = 0
        self._flow_program_qty_scale: float = 1.0
        self._flow_program_trade_prob_mult: float = 1.0
        self._flow_slice_qty_scale: float = 1.0

    def _clear_flow_program(self):
        self._flow_program_side = None
        self._flow_program_remaining = 0
        self._flow_program_qty_scale = 1.0
        self._flow_program_trade_prob_mult = 1.0

    def _flow_program_profile(self) -> Optional[dict]:
        profile = None
        if self.flow_role == 'RetailToxic':
            profile = dict(start_prob=0.02 + 0.10 * self.flow_persistence,
                           run_lo=1, run_hi=3,
                           qty_scale=0.95,
                           trade_prob_mult=1.08)
        elif self.flow_role == 'RealMoney':
            profile = dict(start_prob=0.03 + 0.12 * self.flow_persistence,
                           run_lo=2, run_hi=4,
                           qty_scale=0.72,
                           trade_prob_mult=1.15)
        elif self.flow_role == 'Hedger':
            profile = dict(start_prob=0.01 + 0.04 * self.flow_persistence,
                           run_lo=1, run_hi=2,
                           qty_scale=0.92,
                           trade_prob_mult=1.04)

        if profile is None or self.env is None:
            return profile

        session_name = getattr(self.env, 'session_name', '')
        if self.flow_role == 'RealMoney' and session_name in {'London', 'Overlap', 'NewYork'}:
            profile['start_prob'] *= 1.10
        elif self.flow_role == 'RetailToxic' and session_name == 'Overlap':
            profile['start_prob'] *= 1.08
        elif self.flow_role == 'Hedger' and session_name == 'Asia':
            profile['start_prob'] *= 1.05

        profile['start_prob'] = max(0.0, min(0.45, profile['start_prob']))
        return profile

    def _continue_flow_program(self) -> Optional[tuple[str, float]]:
        if self._flow_program_side is None or self._flow_program_remaining <= 0:
            self._clear_flow_program()
            return None

        side = self._flow_program_side
        qty_scale = self._flow_program_qty_scale
        self._flow_program_remaining -= 1
        self._last_fx_side = side
        if self._flow_program_remaining <= 0:
            self._clear_flow_program()
        return side, qty_scale

    def _maybe_start_flow_program(self, side: str) -> float:
        profile = self._flow_program_profile()
        if profile is None or random.random() >= profile['start_prob']:
            return 1.0

        total_slices = random.randint(profile['run_lo'], profile['run_hi'])
        if total_slices > 1:
            self._flow_program_side = side
            self._flow_program_remaining = total_slices - 1
            self._flow_program_qty_scale = profile['qty_scale']
            self._flow_program_trade_prob_mult = profile['trade_prob_mult']
        else:
            self._clear_flow_program()
        return profile['qty_scale']

    def _session_trade_probability(self) -> float:
        prob = self.trade_prob
        if self._flow_program_side is not None and self._flow_program_remaining > 0:
            prob *= self._flow_program_trade_prob_mult
        if self.env is not None:
            session_mult = getattr(self.env, 'session_flow_multiplier', 1.0)
            prob *= max(0.5, 1.0 + self.session_sensitivity * (session_mult - 1.0))
            session_name = getattr(self.env, 'session_name', '')
            if self.flow_role == 'RetailToxic' and session_name == 'Overlap':
                prob *= 1.10
            elif self.flow_role == 'RealMoney' and session_name in {'London', 'Overlap', 'NewYork'}:
                prob *= 1.12
            elif self.flow_role == 'Hedger' and session_name == 'Asia':
                prob *= 0.95
        return max(0.0, min(0.98, prob))

    def _session_quantity_bounds(self) -> tuple[int, int]:
        lo = self.q_min
        hi = self.q_max
        if self.env is not None:
            session_mult = getattr(self.env, 'session_flow_multiplier', 1.0)
            hi = max(lo, int(round(hi * max(0.75, session_mult))))
            if self.flow_role == 'RealMoney':
                hi = max(lo, int(round(hi * 1.15)))
            elif self.flow_role == 'RetailToxic':
                hi = max(lo, int(round(hi * 0.9)))
        slice_scale = max(0.60, min(1.25, float(getattr(self, '_flow_slice_qty_scale', 1.0))))
        lo = max(1, int(round(lo * max(0.70, slice_scale))))
        hi = max(lo, int(round(hi * slice_scale)))
        return max(1, lo), max(max(1, lo), hi)

    def _draw_flow_side(self) -> str:
        self._flow_slice_qty_scale = 1.0
        program_side = self._continue_flow_program()
        if program_side is not None:
            side, qty_scale = program_side
            self._flow_slice_qty_scale = qty_scale
            return side

        bias = self.env.toxic_flow_bias if self.env else 0.0
        p_buy = 0.5 + bias
        persist_shift = 0.0
        if self._last_fx_side is not None:
            role_scale = 1.0
            if self.flow_role == 'Hedger':
                role_scale = 0.35
            elif self.flow_role == 'RetailToxic':
                role_scale = 1.15
            elif self.flow_role == 'RealMoney':
                role_scale = 0.90
            persist_shift = 0.18 * self.flow_persistence * role_scale
            if self._last_fx_side == 'buy':
                p_buy += persist_shift
            else:
                p_buy -= persist_shift
        if self.flow_role == 'Hedger' and self._last_fx_side is not None:
            mean_reversion_shift = 0.35 * persist_shift
            if self._last_fx_side == 'buy':
                p_buy -= mean_reversion_shift
            else:
                p_buy += mean_reversion_shift
        p_buy = max(0.05, min(0.95, p_buy))
        side = 'buy' if random.random() < p_buy else 'sell'
        self._last_fx_side = side
        self._flow_slice_qty_scale = self._maybe_start_flow_program(side)
        return side

    @staticmethod
    def draw_delta(std: Union[float, int] = 2.5):
        lamb = 1 / std
        return random.expovariate(lamb)

    @staticmethod
    def draw_price(order_type, spread: dict, std: Union[float, int] = 2.5,
                   sigma: float = None, sigma_low: float = None) -> float:
        """
        Draw price for limit order of Noise Agent.

        The inside-spread probability scales inversely with σ/σ_low,
        modelling the empirical tendency of passive liquidity to
        withdraw from the top of book during volatile markets.

        1) p_inside (35 % in normal, lower in stress) — uniform inside spread
        2) (1 − p_inside) — out of spread with exponential delta
        """
        # Regime-aware inside-spread probability
        p_inside = 0.35
        eff_std = std
        if sigma is not None and sigma_low is not None and sigma_low > 0:
            stress_ratio = sigma / sigma_low  # 1.0 normal, 4.0+ stress
            p_inside = max(0.05, 0.35 / stress_ratio)
            # Widen draw_delta in stress: orders placed further from touch
            eff_std = std * max(1.0, stress_ratio * 0.7)

        random_state = random.random()

        # Within the spread
        if random_state < p_inside:
            return random.uniform(spread['bid'], spread['ask'])

        # Out of spread
        else:
            delta = Random.draw_delta(eff_std)
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
            if self.defaulted or random.random() > self._session_trade_probability():
                return None
            side = self._draw_flow_side()
            q_lo, q_hi = self._session_quantity_bounds()
            Q = random.randint(q_lo, q_hi)
            venue = self.choose_venue(Q, side)
            result = self._execute_on_venue(venue, Q, side)
            exec_q = result.get('executed_qty', 0)
            if exec_q <= 0:
                return None
            cls_info = self._classify(exec_q)
            rec = self._make_trade_record(venue, side, exec_q, result, cls_info)
            self.trades.append(rec)
            return rec

        # ---------- classic single-venue mode -----------------------------
        spread = self.market.spread()
        if spread is None:
            return

        mid = (spread['bid'] + spread['ask']) / 2.0

        # Use fair_price as reference for staleness when available
        ref_mid = mid
        if self.env is not None:
            fp = getattr(self.env, 'fair_price', None)
            if fp is not None and fp > 0:
                fair_weight = 0.25 * getattr(self.env, 'systemic_liquidity', 1.0)
                ref_mid = mid + fair_weight * (fp - mid)

        # ── Re-center stale orders ──────────────────────────────────
        # If any resting order is >200 bps from reference mid, cancel it
        # and immediately replace with a fresh limit near the spread.
        stale = None
        for order in self.orders:
            dist_bps = abs(order.price - ref_mid) / ref_mid * 10_000
            if dist_bps > 200:
                stale = order
                break
        if stale is not None:
            side = stale.order_type
            self._cancel_order(stale)
            _sig = self.env.sigma if self.env else None
            _sig_lo = self.env.sigma_low if self.env else None
            price = self.draw_price(side, spread, sigma=_sig, sigma_low=_sig_lo)
            quantity = self.draw_quantity()
            quote_price = self.market.round_price(price) if hasattr(self.market, 'round_price') else round(price, 2)
            order = Order(quote_price, round(quantity), side, self)
            accepted = self.market.limit_order(order)
            if accepted and order.qty > 0:
                self.orders.append(order)
            return  # one re-center per tick is enough

        # ── Normal action selection (Fennell distribution) ──────────
        order_type = 'bid' if random.random() > 0.5 else 'ask'

        random_state = random.random()
        # Market order (15%)
        if random_state > .85:
            quantity = self.draw_quantity()
            if order_type == 'bid':
                self._buy_market(quantity)
            elif order_type == 'ask':
                self._sell_market(quantity)

        # Limit order (35%)
        elif random_state > .50:
            _sig = self.env.sigma if self.env else None
            _sig_lo = self.env.sigma_low if self.env else None
            price = self.draw_price(order_type, spread, sigma=_sig, sigma_low=_sig_lo)
            quantity = self.draw_quantity()
            if order_type == 'bid':
                self._buy_limit(quantity, price)
            elif order_type == 'ask':
                self._sell_limit(quantity, price)

        # Cancellation order (35%)
        elif random_state < .35:
            if self.orders:
                order_n = random.randint(0, len(self.orders) - 1)
                self._cancel_order(self.orders[order_n])


class FastRecyclerLP(Random):
    """Fast electronic LP that rapidly replenishes near-mid depth.

    The class represents short-lived, low-inventory liquidity provision.
    Quotes are refreshed every tick, cluster close to the mid, and become
    less aggressive when toxicity or funding stress rises.
    """

    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0,
                 env: MarketEnvironment = None,
                 levels: int = 2,
                 ttl: int = 2,
                 base_qty: int = 2,
                 max_qty: int = 5,
                 **kwargs):
        super().__init__(market, cash, assets, env=env, label='FastRecyclerLP', **kwargs)
        self.type = 'FastRecyclerLP'
        self.levels = max(1, int(levels))
        self.ttl = max(1, int(ttl))
        self.base_qty = max(1, int(base_qty))
        self.max_qty = max(self.base_qty, int(max_qty))
        self.last_quoted = False

    def _cancel_all(self):
        for order in self.orders.copy():
            try:
                self._cancel_order(order)
            except Exception:
                pass
        self.orders.clear()

    def _reference_mid(self, spread: dict) -> float:
        book_mid = 0.5 * (spread['bid'] + spread['ask'])
        if self.env is None:
            return book_mid
        fair_price = getattr(self.env, 'fair_price', None)
        if fair_price is None or fair_price <= 0:
            return book_mid
        fair_weight = 0.20 + 0.25 * getattr(self.env, 'systemic_liquidity', 1.0)
        fair_weight = max(0.15, min(0.45, fair_weight))
        return book_mid + fair_weight * (fair_price - book_mid)

    def call(self):
        self._cancel_all()
        self.last_quoted = False

        spread = self.market.spread()
        if spread is None:
            return

        mid = self._reference_mid(spread)
        if mid <= 0:
            return

        sigma = getattr(self.env, 'sigma', 0.01) if self.env is not None else 0.01
        funding = getattr(self.env, 'funding_cost', 0.0) if self.env is not None else 0.0
        liquidity = getattr(self.env, 'systemic_liquidity', 1.0) if self.env is not None else 1.0
        toxic_bias = abs(getattr(self.env, 'toxic_flow_bias', 0.0)) if self.env is not None else 0.0

        withdraw_prob = max(0.0, 0.10 + 0.35 * toxic_bias + 0.25 * (1.0 - liquidity))
        if random.random() < min(0.85, withdraw_prob):
            return

        total_spread_bps = 2.0 + 140.0 * sigma + 120.0 * funding
        total_spread_bps *= 1.0 + 0.60 * toxic_bias + 0.35 * (1.0 - liquidity)
        half_spread = max(0.01, mid * total_spread_bps / 20_000.0)
        tick = self.market._tick_size() if hasattr(self.market, '_tick_size') else 0.01

        for level in range(self.levels):
            level_mult = 1.0 + 0.45 * level
            offset = half_spread * level_mult
            qty_scale = max(0.5, liquidity) * (1.0 - 0.25 * level)
            qty = max(1, min(self.max_qty, int(round(self.base_qty * qty_scale + random.random()))))

            bid_price = self.market.round_price(mid - offset) if hasattr(self.market, 'round_price') else round(mid - offset, 2)
            ask_price = self.market.round_price(mid + offset) if hasattr(self.market, 'round_price') else round(mid + offset, 2)

            if bid_price >= spread['ask']:
                bid_price = self.market.round_price(spread['ask'] - tick) if hasattr(self.market, 'round_price') else round(spread['ask'] - tick, 2)
            if ask_price <= spread['bid']:
                ask_price = self.market.round_price(spread['bid'] + tick) if hasattr(self.market, 'round_price') else round(spread['bid'] + tick, 2)
            if bid_price >= ask_price:
                continue

            bid_order = Order(bid_price, qty, 'bid', self, ttl=self.ttl)
            ask_order = Order(ask_price, qty, 'ask', self, ttl=self.ttl)
            bid_ok = self.market.limit_order(bid_order)
            ask_ok = self.market.limit_order(ask_order)
            if bid_ok and bid_order.qty > 0:
                self.orders.append(bid_order)
            if ask_ok and ask_order.qty > 0:
                self.orders.append(ask_order)

        self.last_quoted = bool(self.orders)


class LatentLP(Random):
    """Liquidity provider that enters only when spread/depth dislocate.

    This class models latent liquidity that is not constantly displayed in
    the book but appears when market making becomes attractive enough.
    """

    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0,
                 env: MarketEnvironment = None,
                 entry_spread_bps: float = 8.0,
                 top_depth_threshold: float = 8.0,
                 ttl: int = 4,
                 base_qty: int = 3,
                 max_qty: int = 8,
                 **kwargs):
        super().__init__(market, cash, assets, env=env, label='LatentLP', **kwargs)
        self.type = 'LatentLP'
        self.entry_spread_bps = max(1.0, float(entry_spread_bps))
        self.top_depth_threshold = max(1.0, float(top_depth_threshold))
        self.ttl = max(1, int(ttl))
        self.base_qty = max(1, int(base_qty))
        self.max_qty = max(self.base_qty, int(max_qty))
        self.last_quoted = False

    def _cancel_all(self):
        for order in self.orders.copy():
            try:
                self._cancel_order(order)
            except Exception:
                pass
        self.orders.clear()

    def call(self):
        self._cancel_all()
        self.last_quoted = False

        spread = self.market.spread()
        top = self.market.spread_volume()
        if spread is None or top is None:
            return

        mid = 0.5 * (spread['bid'] + spread['ask'])
        if mid <= 0:
            return

        spread_bps = (spread['ask'] - spread['bid']) / mid * 10_000.0
        near_touch_depth = min(top['bid'], top['ask'])
        liquidity = getattr(self.env, 'systemic_liquidity', 1.0) if self.env is not None else 1.0
        toxic_bias = abs(getattr(self.env, 'toxic_flow_bias', 0.0)) if self.env is not None else 0.0
        shock_ticks = getattr(self.env, 'shock_ticks_remaining', 0) if self.env is not None else 0

        active = (
            spread_bps >= self.entry_spread_bps
            or near_touch_depth <= self.top_depth_threshold
            or shock_ticks > 0
        )
        if not active:
            return

        entry_prob = 0.55 + 0.20 * min(1.0, max(0.0, (spread_bps - self.entry_spread_bps) / self.entry_spread_bps))
        entry_prob += 0.15 * min(1.0, max(0.0, (self.top_depth_threshold - near_touch_depth) / self.top_depth_threshold))
        entry_prob -= 0.30 * toxic_bias
        entry_prob *= 0.75 + 0.25 * liquidity
        if random.random() > max(0.05, min(0.95, entry_prob)):
            return

        fair_price = getattr(self.env, 'fair_price', None) if self.env is not None else None
        if fair_price is not None and fair_price > 0:
            mid = 0.75 * mid + 0.25 * fair_price

        half_spread_bps = max(2.0, min(12.0, 0.35 * spread_bps))
        half_spread = max(0.01, mid * half_spread_bps / 10_000.0)
        tick = self.market._tick_size() if hasattr(self.market, '_tick_size') else 0.01
        depth_gap = max(0.0, self.top_depth_threshold - near_touch_depth)
        qty = max(self.base_qty, int(round(self.base_qty + 0.75 * depth_gap)))
        qty = min(self.max_qty, qty)

        for level in range(2):
            offset = half_spread * (1.0 + 0.75 * level)
            bid_price = self.market.round_price(mid - offset) if hasattr(self.market, 'round_price') else round(mid - offset, 2)
            ask_price = self.market.round_price(mid + offset) if hasattr(self.market, 'round_price') else round(mid + offset, 2)

            if bid_price >= spread['ask']:
                bid_price = self.market.round_price(spread['ask'] - tick) if hasattr(self.market, 'round_price') else round(spread['ask'] - tick, 2)
            if ask_price <= spread['bid']:
                ask_price = self.market.round_price(spread['bid'] + tick) if hasattr(self.market, 'round_price') else round(spread['bid'] + tick, 2)
            if bid_price >= ask_price:
                continue

            bid_order = Order(bid_price, qty, 'bid', self, ttl=self.ttl)
            ask_order = Order(ask_price, qty, 'ask', self, ttl=self.ttl)
            bid_ok = self.market.limit_order(bid_order)
            ask_ok = self.market.limit_order(ask_order)
            if bid_ok and bid_order.qty > 0:
                self.orders.append(bid_order)
            if ask_ok and ask_order.qty > 0:
                self.orders.append(ask_order)

        self.last_quoted = bool(self.orders)


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
                 venue_choice_rule: str = 'fixed_share',
                 deterministic_venue: bool = False,
                 beta_amm: float = 0.05,
                 cpmm_bias_bps: float = 5.0,
                 cost_noise_std: float = 1.5,
                 # FX-fundamentalist params (used only in multi-venue mode)
                 fundamental_rate: float = 100.0,
                 fx_gamma: float = 5e-3,
                 fx_q_max: int = 10,
                 flow_role: str = 'LeveragedDirectional'):
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
                         amm_share_pct=amm_share_pct, venue_choice_rule=venue_choice_rule,
                         deterministic_venue=deterministic_venue,
                         beta_amm=beta_amm, cpmm_bias_bps=cpmm_bias_bps,
                         cost_noise_std=cost_noise_std)
        self.type = 'Fundamentalist'
        self.access = access
        self.fundamental_rate = fundamental_rate
        self.fx_gamma = fx_gamma
        self.fx_q_max = fx_q_max
        self.flow_role = flow_role

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
            if self.defaulted:
                return None
            try:
                mid = self.clob.mid_price()
            except Exception:
                return None
            fair = self.env.fair_price if (self.env is not None and hasattr(self.env, 'fair_price')) else self.fundamental_rate
            session_mult = getattr(self.env, 'session_flow_multiplier', 1.0) if self.env is not None else 1.0
            gamma = self.fx_gamma / max(0.75, session_mult)
            mispricing = (fair - mid) / mid
            if abs(mispricing) < gamma:
                return None
            side = 'buy' if mispricing > 0 else 'sell'
            Q = min(round(abs(mispricing) / gamma), self.fx_q_max)
            if self.flow_role == 'LeveragedDirectional':
                Q = min(self.fx_q_max, max(1, int(round(Q * max(1.0, session_mult)))))
            if Q <= 0:
                return None
            venue = self.choose_venue(Q, side)
            result = self._execute_on_venue(venue, Q, side)
            exec_q = result.get('executed_qty', 0)
            if exec_q <= 0:
                return None
            cls_info = self._classify(exec_q)
            rec = self._make_trade_record(venue, side, exec_q, result, cls_info)
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
        if spread is None:
            return

        if self.sentiment == 'Optimistic':
            # Market order
            if random_state > .85:
                self._buy_market(Random.draw_quantity())
            # Limit order
            elif random_state > .5:
                _sig = self.env.sigma if self.env else None
                _sig_lo = self.env.sigma_low if self.env else None
                self._buy_limit(Random.draw_quantity(), Random.draw_price('bid', spread, sigma=_sig, sigma_low=_sig_lo) * (1 - t_cost))
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
                _sig = self.env.sigma if self.env else None
                _sig_lo = self.env.sigma_low if self.env else None
                self._sell_limit(Random.draw_quantity(), Random.draw_price('ask', spread, sigma=_sig, sigma_low=_sig_lo) * (1 + t_cost))
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
    def __init__(self, market: ExchangeAgent, cash: Union[float, int], assets: int = 0, access: int = 1, **kwargs):
        """
        :param market: exchange agent link
        :param cash: number of cash
        :param assets: number of assets
        :param access: number of future dividends informed
        """
        super().__init__(market, cash, assets, access=access, **kwargs)
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
                 alpha0: float = 1.5, alpha1: float = 300.0, alpha2: float = 500.0,
                 alpha3: float = 50.0,
                 d0: float = 50.0, d1: float = 750.0, d2: float = 500.0,
                 d3: float = 30.0,
                 d_min: float = 3.0, n_levels: int = 5,
                 inv_skew_bps: float = 0.3,
                 venue_interaction_mode: str = 'competition',
                 amm_spread_impact_bps: float = 3.0,
                 amm_depth_impact: float = 60.0,
                 amm_reference_q: float = 5.0,
                 withdrawal_threshold: float = 1.8,
                 reentry_threshold: float = 1.05,
                 loss_threshold_bps: float = 20.0,
                 min_withdraw_ticks: int = 4,
                 reentry_ticks: int = 3):
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
        self.alpha3 = alpha3    # OFI-based spread widening
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3            # OFI-based depth reduction
        self.d_min = d_min
        self.n_levels = max(1, n_levels)
        self.inv_skew_bps = inv_skew_bps  # inventory skew coefficient
        self.venue_interaction_mode = venue_interaction_mode
        self.amm_spread_impact_bps = max(0.0, amm_spread_impact_bps)
        self.amm_depth_impact = max(0.0, amm_depth_impact)
        self.amm_reference_q = max(1.0, float(amm_reference_q))
        self.withdrawal_threshold = max(0.1, float(withdrawal_threshold))
        self.reentry_threshold = max(0.0, min(self.withdrawal_threshold, float(reentry_threshold)))
        self.loss_threshold_bps = max(1.0, float(loss_threshold_bps))
        self.min_withdraw_ticks = max(1, int(min_withdraw_ticks))
        self.reentry_ticks = max(1, int(reentry_ticks))

        # Inventory & order-flow tracking
        self.inventory: float = 0.0       # net position (+ = long)
        self._ofi_window: List[float] = []  # recent order-flow imbalance
        self._ofi_maxlen: int = 20
        self.mm_state: str = 'active'
        self.mm_state_timer: int = 0
        self.mm_withdrawal_score: float = 0.0
        self.mm_state_details: Dict[str, float] = {}
        self._mtm_wealth_prev: Optional[float] = None
        self._mtm_pnl_bps_ewma: float = 0.0
        self._loss_bps_ewma: float = 0.0
        self._signal_decay: float = 0.85

    def _recent_ofi(self) -> float:
        """Average recent order-flow imbalance (positive = buy pressure)."""
        if not self._ofi_window:
            return 0.0
        return sum(self._ofi_window) / len(self._ofi_window)

    def _update_ofi(self):
        """Blend book imbalance with realised taker flow imbalance."""
        sv = self.market.spread_volume()
        if sv is not None:
            book_ofi = (sv['bid'] - sv['ask']) / max(1, sv['bid'] + sv['ask'])
        else:
            book_ofi = 0.0
        flow_ofi = 0.0
        if self.env is not None:
            flow_ofi = getattr(
                self.env,
                'clob_order_flow_imbalance',
                getattr(self.env, 'order_flow_imbalance', 0.0),
            )
        ofi = 0.4 * book_ofi + 0.6 * flow_ofi
        self._ofi_window.append(ofi)
        if len(self._ofi_window) > self._ofi_maxlen:
            self._ofi_window.pop(0)

    def _liquidity_factor(self) -> float:
        if self.env is None:
            return 1.0
        return max(0.2, min(1.0, getattr(self.env, 'systemic_liquidity', 1.0)))

    def _reference_mid(self, spread: Optional[dict], fair_mid: Optional[float]) -> Optional[float]:
        book_mid = None
        if spread is not None:
            book_mid = (spread['bid'] + spread['ask']) / 2.0

        if book_mid is None:
            return fair_mid
        if fair_mid is None or fair_mid <= 0:
            return book_mid

        sigma_base = getattr(self.env, 'sigma_low', None)
        stress_scale = 1.0
        if sigma_base is not None and sigma_base > 0:
            stress_scale = max(1.0, self.env.sigma / sigma_base)

        liquidity_factor = self._liquidity_factor()
        gap_bps = abs(fair_mid - book_mid) / max(book_mid, 1e-9) * 10_000.0
        fair_weight = 0.35 * liquidity_factor / stress_scale
        fair_weight /= 1.0 + gap_bps / 75.0
        fair_weight = max(0.05, min(0.35, fair_weight))
        return book_mid + fair_weight * (fair_mid - book_mid)

    def _amm_support_score(self, mid: float) -> float:
        """Competitive quality of AMM liquidity as an outside option for the CLOB MM.

        The score is high only when AMM liquidity is simultaneously deep,
        cheap for a representative trade, and internally consistent in price.
        This avoids giving the CLOB an automatic bonus just because an AMM
        exists in the market.
        """
        if not self.amm_pools or mid <= 0:
            return 0.0

        scores = []
        pool_mids = []
        stress_scale = 1.0
        sigma_base = getattr(self.env, 'sigma_low', None)
        if sigma_base is not None and sigma_base > 0:
            stress_scale = max(1.0, self.env.sigma / sigma_base)

        for pool in self.amm_pools.values():
            try:
                depth = max(0.0, pool.effective_depth(mid))
                buy_cost = pool.quote_buy(self.amm_reference_q, S_t=mid)['cost_bps']
                sell_cost = pool.quote_sell(self.amm_reference_q, S_t=mid)['cost_bps']
                pool_mid = pool.mid_price()
                avg_cost = 0.5 * (buy_cost + sell_cost)
                if not (_math.isfinite(avg_cost) and _math.isfinite(pool_mid) and pool_mid > 0):
                    continue

                alignment_bps = abs(pool_mid - mid) / mid * 10_000.0
                depth_score = min(depth / max(20.0 * self.amm_reference_q, 1.0), 2.0)
                cost_score = 1.0 / (1.0 + avg_cost / 25.0)
                alignment_score = 1.0 / (1.0 + alignment_bps / 20.0)
                scores.append(depth_score * cost_score * alignment_score)
                pool_mids.append(pool_mid)
            except Exception:
                pass

        if not scores:
            return 0.0

        support = sum(scores) / len(scores)
        if len(pool_mids) > 1:
            dispersion_bps = (max(pool_mids) - min(pool_mids)) / mid * 10_000.0
            support *= 1.0 / (1.0 + dispersion_bps / 15.0)

        return max(0.0, min(1.0, support / stress_scale))

    def _venue_interaction_state(self, mid: float) -> dict:
        """Translate AMM presence into risk modifiers, not direct quote subsidies."""
        liquidity_factor = self._liquidity_factor()
        state = {
            'ofi_scale': 1.0,
            'inventory_scale': 1.0,
            'liquidity_factor': liquidity_factor,
        }
        if self.venue_interaction_mode == 'none':
            return state

        support = self._amm_support_score(mid)
        if support <= 0.0:
            return state

        spread_relief = min(0.35, self.amm_spread_impact_bps / 20.0) * support
        depth_relief = min(0.35, self.amm_depth_impact / 150.0) * support

        if self.venue_interaction_mode == 'toxicity':
            state['ofi_scale'] = 1.0 + 0.60 * spread_relief
            state['inventory_scale'] = 1.0 + 0.75 * depth_relief
            state['liquidity_factor'] = max(0.2, liquidity_factor * (1.0 - 0.40 * depth_relief))
            return state

        state['ofi_scale'] = max(0.65, 1.0 - 0.60 * spread_relief)
        state['inventory_scale'] = max(0.70, 1.0 - 0.75 * depth_relief)
        support_bonus = 0.10 * spread_relief + 0.20 * depth_relief
        state['liquidity_factor'] = min(
            1.15,
            liquidity_factor + depth_relief * (1.0 - liquidity_factor) + support_bonus,
        )
        return state

    def _target_spread_bps(self, mid: float) -> float:
        """Quoted spread in bps as f(σ, c, |OFI|)."""
        venue_state = self._venue_interaction_state(mid)
        effective_ofi = abs(self._recent_ofi()) * venue_state['ofi_scale']
        base = self.alpha0 + self.alpha1 * self.env.sigma + self.alpha2 * self.env.funding_cost
        base += self.alpha3 * effective_ofi
        base += 35.0 * (1.0 - venue_state['liquidity_factor'])
        return max(0.5, base)

    def _target_depth(self, mid: float) -> float:
        """Depth as f(σ, c, |OFI|, |inventory|), floored at d_min."""
        venue_state = self._venue_interaction_state(mid)
        effective_ofi = abs(self._recent_ofi()) * venue_state['ofi_scale']
        inventory_penalty = 0.1 * abs(self.inventory) * venue_state['inventory_scale']
        d = self.d0 - self.d1 * self.env.sigma - self.d2 * self.env.funding_cost
        d -= self.d3 * effective_ofi
        d -= inventory_penalty
        d *= venue_state['liquidity_factor']
        return max(d, self.d_min)

    def _inventory_skew(self, mid: float) -> float:
        """Skew mid-price away from inventory risk (bps → price offset)."""
        return -self.inv_skew_bps * self.inventory * mid / 10_000.0

    def cancel_all_quotes(self):
        for order in self.orders.copy():
            try:
                self._cancel_order(order)
            except Exception:
                pass
        self.orders.clear()

    def _mark_to_market(self, mid: float) -> float:
        return self.cash + self.assets * mid

    def _update_mtm_signals(self, mid: float):
        wealth = self._mark_to_market(mid)
        if self._mtm_wealth_prev is None:
            self._mtm_wealth_prev = wealth
            return

        base = max(abs(self._mtm_wealth_prev), 1e-9)
        pnl_bps = 10_000.0 * (wealth - self._mtm_wealth_prev) / base
        decay = self._signal_decay
        self._mtm_pnl_bps_ewma = decay * self._mtm_pnl_bps_ewma + (1.0 - decay) * pnl_bps
        self._loss_bps_ewma = decay * self._loss_bps_ewma + (1.0 - decay) * max(0.0, -pnl_bps)
        self._mtm_wealth_prev = wealth

    def _withdrawal_components(self, mid: float) -> Dict[str, float]:
        sigma_base = max(getattr(self.env, 'sigma_low', self.env.sigma), 1e-6)
        funding_base = max(getattr(self.env, 'c_low', self.env.funding_cost), 1e-6)
        liquidity_factor = self._liquidity_factor()

        sigma_score = min(2.5, max(0.0, self.env.sigma / sigma_base - 1.0))
        funding_score = min(2.5, max(0.0, self.env.funding_cost / funding_base - 1.0))
        inventory_abs = max(abs(self.inventory), abs(float(self.assets)))
        inventory_ratio = min(1.5, inventory_abs / max(1.5 * float(self.softlimit), 1.0))
        ofi_score = min(2.0, abs(self._recent_ofi()))
        liquidity_damage = min(1.5, max(0.0, 1.0 - liquidity_factor))
        outside_option_score = min(1.5, self._amm_support_score(mid))
        loss_score = min(3.0, self._loss_bps_ewma / self.loss_threshold_bps)
        shock_window = min(1.0, float(getattr(self.env, 'shock_ticks_remaining', 0)) / 12.0)
        liquidity_shock = max(0.0, float(getattr(self.env, 'liquidity_shock', 0.0)))
        toxic_flow = abs(float(getattr(self.env, 'toxic_flow_bias', 0.0)))
        acute_shock_score = 0.85 * liquidity_shock + 0.30 * toxic_flow + 0.25 * shock_window

        score = (
            0.40 * loss_score
            + 0.12 * sigma_score
            + 0.05 * funding_score
            + 0.22 * inventory_ratio
            + 0.22 * ofi_score
            + 0.32 * liquidity_damage
            + 0.10 * outside_option_score
        )
        score += acute_shock_score
        return {
            'score': score,
            'loss_score': loss_score,
            'sigma_score': sigma_score,
            'funding_score': funding_score,
            'inventory_ratio': inventory_ratio,
            'ofi_score': ofi_score,
            'liquidity_factor': liquidity_factor,
            'outside_option_score': outside_option_score,
            'acute_shock_score': acute_shock_score,
            'loss_bps_ewma': self._loss_bps_ewma,
            'mtm_pnl_bps_ewma': self._mtm_pnl_bps_ewma,
        }

    def _update_endogenous_state(self, mid: float) -> str:
        if self.env is None:
            self.mm_state = 'active'
            self.mm_state_timer = 0
            self.mm_withdrawal_score = 0.0
            self.mm_state_details = {'score': 0.0, 'liquidity_factor': self._liquidity_factor()}
            return self.mm_state

        self._update_mtm_signals(mid)
        details = self._withdrawal_components(mid)
        score = details['score']
        defensive_threshold = max(self.reentry_threshold, 0.60 * self.withdrawal_threshold)
        withdraw_release_threshold = max(defensive_threshold, 0.85 * self.withdrawal_threshold)
        liquidity_ok = details['liquidity_factor'] >= 0.55

        self.mm_withdrawal_score = score
        self.mm_state_details = details

        if self.mm_state == 'withdrawn':
            if self.mm_state_timer > 0:
                self.mm_state_timer -= 1
            if self.mm_state_timer <= 0 and score <= withdraw_release_threshold and liquidity_ok:
                self.mm_state = 'reentering'
                self.mm_state_timer = self.reentry_ticks
            return self.mm_state

        if self.mm_state == 'reentering':
            if score >= self.withdrawal_threshold:
                self.mm_state = 'withdrawn'
                self.mm_state_timer = self.min_withdraw_ticks
                return self.mm_state

            if self.mm_state_timer > 0:
                self.mm_state_timer -= 1
            if self.mm_state_timer <= 0:
                self.mm_state = 'active' if score < 0.5 * self.reentry_threshold else 'defensive'
            return self.mm_state

        if score >= self.withdrawal_threshold:
            self.mm_state = 'withdrawn'
            self.mm_state_timer = self.min_withdraw_ticks
        elif score >= defensive_threshold:
            self.mm_state = 'defensive'
            self.mm_state_timer = 0
        elif self.mm_state == 'defensive' and score > self.reentry_threshold:
            self.mm_state = 'defensive'
        else:
            self.mm_state = 'active'
            self.mm_state_timer = 0
        return self.mm_state

    def _state_quote_adjustments(self) -> Dict[str, float | bool]:
        if self.mm_state == 'withdrawn':
            return {'can_quote': False, 'spread_mult': float('inf'), 'depth_mult': 0.0}

        if self.mm_state == 'reentering':
            progress = 1.0 - self.mm_state_timer / max(1, self.reentry_ticks)
            return {
                'can_quote': True,
                'spread_mult': max(1.05, 1.30 - 0.20 * progress),
                'depth_mult': min(0.80, 0.35 + 0.35 * progress),
            }

        if self.mm_state == 'defensive':
            score_ratio = min(1.5, self.mm_withdrawal_score / max(self.withdrawal_threshold, 1e-9))
            return {
                'can_quote': True,
                'spread_mult': 1.0 + 0.30 * score_ratio,
                'depth_mult': max(0.25, 1.0 - 0.40 * score_ratio),
            }

        return {'can_quote': True, 'spread_mult': 1.0, 'depth_mult': 1.0}

    def call(self):
        # ---------- multi-venue mode (σ/c-dependent MM) -------------------
        if self.env is not None:
            # Update OFI tracking
            self._update_ofi()

            # Track inventory from fills on previous orders:
            # orders that were partially filled have reduced qty
            for order in self.orders:
                if hasattr(order, '_mm_init_qty'):
                    filled = order._mm_init_qty - order.qty
                    if filled > 0:
                        if order.order_type == 'bid':
                            self.inventory += filled
                        else:
                            self.inventory -= filled
            self.cancel_all_quotes()

            sp = self.market.spread()
            fair_mid = getattr(self.env, 'fair_price', None)

            if sp is None and (fair_mid is None or fair_mid <= 0):
                return

            mid = self._reference_mid(sp, fair_mid)
            if mid is None or mid <= 0:
                return

            self._update_endogenous_state(mid)
            quote_state = self._state_quote_adjustments()
            if not quote_state['can_quote']:
                return

            # Apply inventory skew: shift effective mid away from risk
            skew = self._inventory_skew(mid)
            eff_mid = mid + skew

            half_spread = (
                mid * self._target_spread_bps(mid) * quote_state['spread_mult'] / 10_000.0 / 2.0
            )
            total_depth = max(1, int(self._target_depth(mid) * quote_state['depth_mult']))

            # Inventory-based one-sided thinning: reduce depth on the
            # overexposed side to encourage inventory mean-reversion
            inv_ratio = min(1.0, abs(self.inventory) / max(1, self.softlimit))
            bid_depth_frac = 1.0 - 0.5 * inv_ratio if self.inventory > 0 else 1.0
            ask_depth_frac = 1.0 - 0.5 * inv_ratio if self.inventory < 0 else 1.0

            # Inverse pyramid: most depth at tight spread, tapering out
            weights = list(range(self.n_levels, 0, -1))
            w_sum = sum(weights)
            for i in range(self.n_levels):
                base_qty = max(1, int(total_depth * weights[i] / w_sum))
                offset = half_spread * (1.0 + i * 0.5)
                bid_qty = max(1, int(base_qty * bid_depth_frac))
                ask_qty = max(1, int(base_qty * ask_depth_frac))
                bid_price = round(eff_mid - offset, 2)
                ask_price = round(eff_mid + offset, 2)
                self._buy_limit(bid_qty, bid_price)
                self._sell_limit(ask_qty, ask_price)
            # Tag orders with initial qty for inventory tracking next tick
            for order in self.orders:
                order._mm_init_qty = order.qty
            return

        # ---------- classic single-venue mode (inventory balance) ---------
        # Clear previous orders
        self.cancel_all_quotes()

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
                 max_adj: float = 0.05,
                 core_liquidity_ratio: float = 0.65,
                 stabilizing_bias: float = 0.02,
                 wallet_cash: Optional[float] = None,
                 wallet_base: Optional[float] = None,
                 wallet_cash_buffer_ratio: float = 0.20,
                 wallet_base_buffer_ratio: float = 0.20):
        self.type = 'AMMProvider'
        self.pool = pool
        self.env = env
        self.phi1 = phi1
        self.phi2 = phi2
        self.phi3 = phi3
        self.max_adj = max_adj
        self.core_liquidity_ratio = max(0.0, min(1.0, core_liquidity_ratio))
        self.stabilizing_bias = max(0.0, stabilizing_bias)
        self._initial_liquidity = max(self.pool.liquidity_measure(), 1e-9)
        self.deployed_cash = max(0.0, float(self.pool.y))
        self.deployed_base = max(0.0, float(self.pool.x))
        self.wallet_cash = (
            max(0.0, float(wallet_cash))
            if wallet_cash is not None
            else self.deployed_cash * max(0.0, float(wallet_cash_buffer_ratio))
        )
        self.wallet_base = (
            max(0.0, float(wallet_base))
            if wallet_base is not None
            else self.deployed_base * max(0.0, float(wallet_base_buffer_ratio))
        )
        ref_price = self._reference_price()
        self._prev_pool_value = self._pool_mark_to_market(ref_price)
        self._pnl_bps_ewma = 0.0
        self._adverse_selection_ewma = 0.0

    def _reference_price(self) -> float:
        ref_price = getattr(self.env, 'fair_price', None)
        if ref_price is None or not _math.isfinite(ref_price) or ref_price <= 0:
            ref_price = self.pool.mid_price()
        return max(float(ref_price), 1e-9)

    def _pool_mark_to_market(self, reference_price: float) -> float:
        return max(1e-9, self.pool.x * reference_price + self.pool.y)

    def _adverse_selection_score(self, reference_price: float) -> float:
        pool_mid = self.pool.mid_price()
        basis_bps = 0.0
        if _math.isfinite(pool_mid) and pool_mid > 0 and reference_price > 0:
            basis_bps = abs(pool_mid - reference_price) / reference_price * 10_000.0
        reserve_imbalance = max(0.0, min(1.0, getattr(self.env, 'amm_reserve_imbalance', 0.0)))
        slippage_bps = max(0.0, getattr(self.env, 'amm_slippage_signal', 0.0))
        flow_pressure = abs(getattr(self.env, 'amm_order_flow_imbalance', 0.0))
        return min(3.0, basis_bps / 20.0 + reserve_imbalance + 0.5 * flow_pressure + slippage_bps / 50.0)

    def _max_affordable_add_fraction(self) -> float:
        base_capacity = self.wallet_base / max(self.pool.x, 1e-9)
        cash_capacity = self.wallet_cash / max(self.pool.y, 1e-9)
        return max(0.0, min(base_capacity, cash_capacity))

    def _deploy_liquidity(self, fraction: float) -> float:
        fraction = max(0.0, min(float(fraction), self._max_affordable_add_fraction()))
        if fraction <= 0:
            return 0.0
        base_added = self.pool.x * fraction
        cash_added = self.pool.y * fraction
        self.wallet_base -= base_added
        self.wallet_cash -= cash_added
        self.deployed_base += base_added
        self.deployed_cash += cash_added
        self.pool.add_liquidity(fraction)
        return fraction

    def _withdraw_liquidity(self, fraction: float) -> float:
        fraction = max(0.0, min(float(fraction), 0.5))
        if fraction <= 0:
            return 0.0
        base_removed = self.pool.x * fraction
        cash_removed = self.pool.y * fraction
        self.pool.remove_liquidity(fraction)
        self.deployed_base = max(0.0, self.deployed_base - base_removed)
        self.deployed_cash = max(0.0, self.deployed_cash - cash_removed)
        self.wallet_base += base_removed
        self.wallet_cash += cash_removed
        return fraction

    def update_liquidity(self):
        """Adjust pool liquidity according to LP rule.  Called once per period."""
        L = self.pool.liquidity_measure()
        if L <= 0:
            return

        ref_price = self._reference_price()
        pool_value = self._pool_mark_to_market(ref_price)
        prev_value = max(self._prev_pool_value, 1e-9)
        pnl_bps = 10_000.0 * (pool_value - prev_value) / prev_value
        self._prev_pool_value = pool_value
        self._pnl_bps_ewma = 0.80 * self._pnl_bps_ewma + 0.20 * pnl_bps
        self._adverse_selection_ewma = 0.75 * self._adverse_selection_ewma + 0.25 * self._adverse_selection_score(ref_price)

        fee_income = self.pool.period_fee_revenue
        sigma = self.env.sigma
        c = self.env.funding_cost
        liquidity_factor = getattr(self.env, 'systemic_liquidity', 1.0)
        flow_pressure = abs(getattr(self.env, 'amm_order_flow_imbalance', 0.0))
        sigma_base = max(getattr(self.env, 'sigma_low', sigma), 1e-9)
        stress_excess = max(0.0, sigma / sigma_base - 1.0)
        stressed_flow_pressure = flow_pressure * (stress_excess + (1.0 - liquidity_factor))
        loss_score = min(3.0, max(0.0, -self._pnl_bps_ewma) / 15.0)
        adverse_selection_score = min(3.0, self._adverse_selection_ewma)
        delta_L = self.phi1 * fee_income - self.phi2 * sigma - self.phi3 * c
        frac = delta_L / L
        frac -= 0.03 * (1.0 - liquidity_factor)
        frac -= 0.02 * stressed_flow_pressure
        frac -= 0.014 * loss_score
        frac -= 0.018 * adverse_selection_score
        frac += self.stabilizing_bias * max(0.0, 0.85 - liquidity_factor)

        core_floor = self._initial_liquidity * self.core_liquidity_ratio
        if L < core_floor:
            refill = min(self.max_adj, max(0.0, (core_floor - L) / max(L, 1e-9)))
            frac = max(frac, refill)
        elif frac < 0 and L * (1.0 + frac) < core_floor:
            frac = core_floor / max(L, 1e-9) - 1.0

        frac = max(-self.max_adj, min(self.max_adj, frac))
        if frac > 0:
            self._deploy_liquidity(frac)
        elif frac < 0:
            self._withdraw_liquidity(abs(frac))


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
                 routing: str = 'all',
                 trade_fraction_cap: float = 0.20,
                 cash: Optional[float] = None,
                 assets: Optional[float] = None,
                 cash_buffer_ratio: float = 0.15,
                 asset_buffer_ratio: float = 0.15):
        self.clob = clob
        self.amm_pools = amm_pools
        self.env = env
        self.type = 'AMMArbitrageur'
        self.max_spread_bps = max_spread_bps
        self.max_correction_pct = max_correction_pct / 100.0
        self.routing = routing
        self.trade_fraction_cap = max(0.0, min(1.0, trade_fraction_cap))
        aggregate_base = sum(max(0.0, float(pool.x)) for pool in amm_pools.values())
        aggregate_cash = sum(max(0.0, float(pool.y)) for pool in amm_pools.values())
        self.cash = (
            max(0.0, float(cash))
            if cash is not None
            else aggregate_cash * max(0.0, float(cash_buffer_ratio))
        )
        self.assets = (
            max(0.0, float(assets))
            if assets is not None
            else aggregate_base * max(0.0, float(asset_buffer_ratio))
        )
        self.cumulative_traded_base = 0.0
        self.cumulative_pnl_quote = 0.0

    # ---- helpers ---------------------------------------------------------

    def _amm_consensus(self) -> Optional[float]:
        """Average AMM mid-price only when pools are internally consistent."""
        mids = []
        for p in self.amm_pools.values():
            try:
                mid = p.mid_price()
                if _math.isfinite(mid) and mid > 0:
                    mids.append(mid)
            except Exception:
                pass
        if not mids:
            return None
        if len(mids) == 1:
            return mids[0]

        mids = sorted(mids)
        mid_idx = len(mids) // 2
        if len(mids) % 2 == 0:
            center = 0.5 * (mids[mid_idx - 1] + mids[mid_idx])
        else:
            center = mids[mid_idx]
        dispersion_bps = 10_000.0 * (mids[-1] - mids[0]) / max(center, 1e-9)
        if dispersion_bps > 250.0:
            return None
        return sum(mids) / len(mids)

    def _blend_with_fair(self, observed: Optional[float]) -> Optional[float]:
        """Treat env.fair_price as a latent anchor rather than a command."""
        fair = None
        if self.env is not None:
            fair = self.env.fair_price

        if fair is None:
            return observed
        if observed is None:
            return fair

        sigma_base = getattr(self.env, 'sigma_low', None) if self.env is not None else None
        stress_scale = 1.0
        if sigma_base is not None and sigma_base > 0 and self.env is not None:
            stress_scale = max(1.0, self.env.sigma / sigma_base)

        liquidity_factor = getattr(self.env, 'systemic_liquidity', 1.0)
        gap_bps = abs(fair - observed) / max(observed, 1e-9) * 10_000.0
        fair_weight = 0.35 * liquidity_factor / stress_scale
        fair_weight /= 1.0 + gap_bps / 75.0
        fair_weight = max(0.05, min(0.35, fair_weight))
        return observed + fair_weight * (fair - observed)

    def _reference_price(self) -> Optional[float]:
        """
        Determine the best available reference price.

        Priority:
        1. CLOB / AMM observed price discovery
        2. Blend with env.fair_price as a latent anchor
        3. Pure fair_price only if all venue prices are unavailable
        """
        fair = self.env.fair_price if self.env is not None else None

        # CLOB / blend / AMM fallback
        S_clob = None
        try:
            S_clob = self.clob.mid_price()
        except Exception:
            pass

        S_amm = self._amm_consensus()

        if S_clob is None and S_amm is None:
            return fair
        if S_clob is None:
            return self._blend_with_fair(S_amm)
        if S_amm is None:
            return self._blend_with_fair(S_clob)

        # Sanity: if CLOB is implausible relative to AMM (>50 % off)
        clob_dev = abs(S_clob - S_amm) / S_amm if S_amm > 0 else 0
        if clob_dev > 0.5:
            return self._blend_with_fair(S_amm)

        try:
            spread_bps = self.clob.quoted_spread_bps()
        except Exception:
            spread_bps = float('inf')

        if not _math.isfinite(spread_bps) or spread_bps > 2000:
            return self._blend_with_fair(S_amm)
        if spread_bps > self.max_spread_bps:
            w = max(0.0, 1.0 - (spread_bps - self.max_spread_bps)
                    / (2000 - self.max_spread_bps))
            market_ref = w * S_clob + (1.0 - w) * S_amm
            return self._blend_with_fair(market_ref)

        market_ref = 0.65 * S_clob + 0.35 * S_amm
        return self._blend_with_fair(market_ref)

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

    def _max_affordable_buy_qty(self, pool, max_trade_qty: float) -> float:
        if max_trade_qty <= 0 or self.cash <= 0:
            return 0.0

        lo = 0.0
        hi = max_trade_qty
        for _ in range(60):
            mid = (lo + hi) / 2.0
            quote = pool.quote_buy(mid)
            if quote['cost_bps'] == float('inf') or quote['delta_y'] > self.cash:
                hi = mid
            else:
                lo = mid
        return lo

    def _settle_trade(self, pool, traded_qty: float, before_y: float, reference_price: float):
        if traded_qty > 0:
            cash_spent = max(0.0, (pool.y - before_y) / max(1.0 - pool.fee, 1e-9))
            self.assets += traded_qty
            self.cash = max(0.0, self.cash - cash_spent)
            self.cumulative_pnl_quote += traded_qty * reference_price - cash_spent
            self.cumulative_traded_base += traded_qty
        elif traded_qty < 0:
            sell_qty = abs(traded_qty)
            cash_received = max(0.0, before_y - pool.y)
            self.assets = max(0.0, self.assets - sell_qty)
            self.cash += cash_received
            self.cumulative_pnl_quote += cash_received - sell_qty * reference_price
            self.cumulative_traded_base += sell_qty

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

            max_trade_qty = max(0.0, pool.x * self.trade_fraction_cap)
            if current < S_capped:
                max_trade_qty = self._max_affordable_buy_qty(pool, max_trade_qty)
            else:
                max_trade_qty = min(max_trade_qty, max(0.0, self.assets))

            if max_trade_qty <= 1e-9:
                return

            before_y = pool.y
            traded_qty = pool.arbitrage_to_target(S_capped, max_trade_qty=max_trade_qty)
            if abs(traded_qty) <= 1e-9:
                return
            self._settle_trade(pool, traded_qty, before_y, S_capped)

            if hasattr(pool, 'update_rate'):
                pool.update_rate()
        except Exception:
            pass
