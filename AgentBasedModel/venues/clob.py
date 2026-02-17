"""
CLOB venue wrapper — computes execution cost metrics over ExchangeAgent's
order book without modifying it (for quoting) and delegates execution.

Metrics:
    - quoted spread (qspr)
    - effective spread (espr)
    - price impact
    - all-in execution cost  C_CLOB(Q)
    - depth by price level
    - Amihud illiquidity measure
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Dict

if TYPE_CHECKING:
    from AgentBasedModel.agents.agents import ExchangeAgent


class CLOBVenue:
    """
    Read-only analytics layer on top of an :class:`ExchangeAgent` order book.

    Parameters
    ----------
    exchange : ExchangeAgent
        The underlying CLOB.
    f_broker : float
        Broker fee in **bps** (default 0).
    f_venue : float
        Exchange/venue fee in **bps** (default 0).
    """

    def __init__(self, exchange: 'ExchangeAgent',
                 f_broker: float = 0.0, f_venue: float = 0.0):
        self.exchange = exchange
        self.f_broker = f_broker
        self.f_venue = f_venue

    # ---- basic prices ----------------------------------------------------

    def mid_price(self) -> float:
        return self.exchange.price()

    def spread(self) -> Optional[dict]:
        return self.exchange.spread()

    def quoted_spread_bps(self) -> float:
        """qspr_t = 10 000 × (ask − bid) / mid"""
        sp = self.exchange.spread()
        if sp is None:
            return float('inf')
        mid = self.mid_price()
        return 10_000.0 * (sp['ask'] - sp['bid']) / mid

    # ---- walk the book (read-only) ---------------------------------------

    def _walk_book(self, Q: float, side: str = 'buy'):
        """
        Walk the order book for *Q* units (base) without executing.

        Returns (vwap, levels_consumed, remaining_qty).
        *levels_consumed* is a list of (price, fill_qty) tuples.
        """
        book_side = 'ask' if side == 'buy' else 'bid'
        total_cost = 0.0
        remaining = Q
        levels = []

        for order in self.exchange.order_book[book_side]:
            if remaining <= 0:
                break
            fill = min(remaining, order.qty)
            total_cost += fill * order.price
            remaining -= fill
            levels.append((order.price, fill))

        if remaining > 0:
            return None, levels, remaining

        vwap = total_cost / Q
        return vwap, levels, 0.0

    # ---- cost quoting ----------------------------------------------------

    def quote_buy(self, Q: float) -> dict:
        """
        Quote buying *Q* base from the ask side.  Returns dict:
            exec_price, half_spread_bps, espr_bps, impact_bps,
            fee_bps, cost_bps
        """
        return self._quote(Q, 'buy')

    def quote_sell(self, Q: float) -> dict:
        """Quote selling *Q* base into the bid side."""
        return self._quote(Q, 'sell')

    def _quote(self, Q: float, side: str) -> dict:
        if Q <= 0:
            return self._zero_quote()

        mid = self.mid_price()
        vwap, levels, remaining = self._walk_book(Q, side)

        if vwap is None:
            return self._inf_quote()

        # Half effective spread (one-sided cost vs mid)
        if side == 'buy':
            half_spr = 10_000.0 * (vwap - mid) / mid
        else:
            half_spr = 10_000.0 * (mid - vwap) / mid

        # Effective spread (two-sided, standard definition)
        espr = 2.0 * half_spr

        # Price impact: estimate new mid after hypothetical execution
        impact_bps = self._estimate_impact_bps(Q, side, levels)

        fee_bps = self.f_broker + self.f_venue

        # All-in cost for venue comparison: half-spread + fee
        # (impact is logged separately as an externality metric)
        cost_bps = half_spr + fee_bps

        return dict(exec_price=vwap, half_spread_bps=half_spr,
                    espr_bps=espr, impact_bps=impact_bps,
                    fee_bps=fee_bps, cost_bps=cost_bps)

    def _estimate_impact_bps(self, Q: float, side: str,
                             levels: list) -> float:
        """
        Estimate post-trade mid shift.
        impact = 10 000 × (S_{t+} − S_t) / S_t  (signed for buy).
        """
        mid = self.mid_price()
        sp = self.exchange.spread()
        if sp is None:
            return 0.0

        book_side = 'ask' if side == 'buy' else 'bid'
        consumed_qty = sum(qty for _, qty in levels)
        remaining_in_levels = 0.0

        # Walk the book again to find new best price after consumption
        cursor_qty = 0.0
        new_best = None
        for order in self.exchange.order_book[book_side]:
            cursor_qty += order.qty
            if cursor_qty > consumed_qty:
                # This order is partially (or fully) remaining
                new_best = order.price
                break

        if new_best is None:
            # Book exhausted on this side
            return 0.0

        if side == 'buy':
            new_mid = (sp['bid'] + new_best) / 2.0
        else:
            new_mid = (new_best + sp['ask']) / 2.0

        return 10_000.0 * (new_mid - mid) / mid

    # ---- depth -----------------------------------------------------------

    def depth(self, n_levels: int = 5) -> dict:
        """
        Aggregate volume at the top *n_levels* distinct price levels on
        each side of the book.
        """
        result = {'bid': [], 'ask': []}
        for book_side in ('bid', 'ask'):
            price_vol: Dict[float, float] = {}
            for order in self.exchange.order_book[book_side]:
                price_vol.setdefault(order.price, 0.0)
                price_vol[order.price] += order.qty
            items = list(price_vol.items())[:n_levels]
            result[book_side] = items
        return result

    def total_depth(self, bps_from_mid: float = 100) -> dict:
        """Total volume within *bps_from_mid* of mid."""
        mid = self.mid_price()
        lo = mid * (1 - bps_from_mid / 10_000)
        hi = mid * (1 + bps_from_mid / 10_000)
        bid_vol = sum(o.qty for o in self.exchange.order_book['bid']
                      if o.price >= lo)
        ask_vol = sum(o.qty for o in self.exchange.order_book['ask']
                      if o.price <= hi)
        return {'bid': bid_vol, 'ask': ask_vol, 'total': bid_vol + ask_vol}

    # ---- Amihud illiquidity ----------------------------------------------

    @staticmethod
    def amihud(returns: List[float], volumes: List[float]) -> float:
        """
        Amihud illiquidity = mean(|r_t| / Vol_t).
        """
        if not returns or not volumes:
            return 0.0
        vals = []
        for r, v in zip(returns, volumes):
            if v > 0:
                vals.append(abs(r) / v)
        return sum(vals) / len(vals) if vals else 0.0

    # ---- helpers ---------------------------------------------------------

    def cost_bps(self, Q: float, side: str = 'buy') -> float:
        """Convenience: return scalar all-in cost in bps."""
        if side == 'buy':
            return self.quote_buy(Q)['cost_bps']
        return self.quote_sell(Q)['cost_bps']

    def _zero_quote(self):
        return dict(exec_price=self.mid_price(), half_spread_bps=0,
                    espr_bps=0, impact_bps=0, fee_bps=0, cost_bps=0)

    def _inf_quote(self):
        return dict(exec_price=float('inf'), half_spread_bps=float('inf'),
                    espr_bps=float('inf'), impact_bps=float('inf'),
                    fee_bps=self.f_broker + self.f_venue,
                    cost_bps=float('inf'))
