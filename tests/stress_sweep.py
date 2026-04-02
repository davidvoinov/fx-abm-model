#!/usr/bin/env python3
"""
Stress Sweep — Simulation + Analysis (unified)
===============================================

Функционал
----------
Этот файл объединяет генерацию данных и их анализ в единый пайплайн.

**Генерация (Block A + Block B)**

  Block A:  Полный перебор CLOB-конфигураций из 5 агентов
            (MarketMaker, Random, Fundamentalist, Chartist, Universalist),
            где MM <= 1.  Всего 91 конфигурация.  Без AMM-пулов.

  Block B:  Полный перебор AMM-конфигураций из 1-5 пулов
            (CPMM + HFMM).  Всего 20 конфигураций.  Без CLOB-агентов.
            AMM полностью автономен: TCA по internal benchmark (mid-price
            пула), reference price берётся из ShadowCLOB (env.fair_price).

  Block C:  Гетерогенные HFMM-пулы.  3 варианта:
            tight (fee=5bp, A=50, reserves=500),
            balanced (fee=10bp, A=10, reserves=1000),
            deep (fee=20bp, A=5, reserves=2000).
            Полный перебор 1-5 пулов из 3 вариантов.  55 конфигураций.

  Block D:  Мультивенью CLOB+AMM с FX-тейкерами.
            CLOB: лучший состав из Block A (MM=1, Rnd=2, Chart=1, Univ=1).
            AMM:  лучший пул из Block C (1 tight HFMM, fee=5bp, A=50, R=500).
            Перебор amm_share_pct = 0, 10, 20, ..., 100.  11 конфигураций.
            FX-тейкеры маршрутизируют ордера между CLOB и AMM.

  Каждая конфигурация:
    * 1 000 итераций, фиксированный seed=42
    * Шок -20% fair price на t=350
    * Глубина DEPTH=1000: одинаковая для CLOB volume и AMM reserves
    * Без внешних FX-тейкеров (замкнутая экосистема)
    * Результат: CSV с посделочными записями (config_id, t, venue,
      side, quantity, fair_mid, exec_price, cost_bps)

**Анализ**

  После генерации автоматически запускается анализ:
    1. Обзор данных и качество (fill rate, inf-записи)
    2. Block A: влияние состава агентов на fv_cost
    3. Block B: влияние типа/числа пулов на cost_bps
    3c. Block C: гетерогенные HFMM-пулы (homo vs hetero, marginal effects)
    3d. Block D: мультивенью CLOB+AMM (sweep amm_share_pct 0-100%)
    4. Кросс-площадочное сравнение CLOB vs AMM
    5. Шоковая устойчивость (pre/post t=350)
    6. Лучшие конфигурации для CLOB, AMM, HFMM и мультивенью
    7. Выводы

  Recovery time (во всех блоках):
    Метрика: кол-во периодов после шока, пока rolling(20) median fv_cost
    не вернётся в пределы ±20 bps от pre-shock baseline.
    Блоки A-C: recovery per config. Блок D: combined + per-venue (CLOB/AMM).

  Метрика fv_cost (fair-value execution cost, bps):
    buy  -> 10000 * (exec_price - fair_mid) / fair_mid
    sell -> 10000 * (fair_mid - exec_price) / fair_mid
    Положительная = переплата, отрицательная = выгодная цена.

Выходные файлы
--------------
  output/stress/CLOB_config_mm*_rnd*_fund*_chart*_univ*.csv
  output/stress/AMM_config_cpmm{C}_hfmm{H}.csv
  output/stress/AMM_C_config_t{T}_b{B}_d{D}.csv
  output/stress/MV_config_amm{P}pct.csv

Запуск
------
    python tests/stress_sweep.py            # генерация + анализ
    python tests/stress_sweep.py --analyze  # только анализ (без перегенерации)
"""

import sys, os, random, math, re, glob, warnings, argparse
import numpy as np
import pandas as pd
from itertools import chain as _chain

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from AgentBasedModel.agents.agents import (
    ExchangeAgent, Random, Fundamentalist, Chartist, Universalist,
    MarketMaker, AMMProvider, AMMArbitrageur, Trader,
)
from AgentBasedModel.simulator.simulator import Simulator, SimulatorInfo
from AgentBasedModel.venues.clob import CLOBVenue, ShadowCLOB
from AgentBasedModel.venues.amm import CPMMPool, HFMMPool
from AgentBasedModel.environment.processes import MarketEnvironment
from AgentBasedModel.metrics.logger import MetricsLogger

# ======================================================================
#  Constants
# ======================================================================
N_ITER      = 1000
SHOCK_ITER  = 350
SHOCK_PCT   = -20.0       # -20 % crash
SEED        = 42
PRICE       = 100.0
DEPTH       = 1000        # comparable CLOB volume & AMM reserves per pool

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'stress')
os.makedirs(OUT_DIR, exist_ok=True)


# ######################################################################
#  PART 1 -- SIMULATION  (data generation)
# ######################################################################


# -- Monkey-patch: include exec_price in trade records -----------------
_orig_make_record = Trader._make_trade_record

def _patched_make_record(self, venue, side, Q, result, cls_info):
    rec = _orig_make_record(self, venue, side, Q, result, cls_info)
    rec['exec_price'] = result.get('exec_price', None)
    return rec

Trader._make_trade_record = _patched_make_record

# -- Monkey-patch: capture CLOB agent market orders --------------------
_captured_clob_trades: list = []

_orig_buy_market = Trader._buy_market
_orig_sell_market = Trader._sell_market


def _capturing_buy_market(self, quantity):
    q = round(quantity)
    if q <= 0 or not self.market.order_book['ask']:
        return _orig_buy_market(self, quantity)
    mid = self.market.price()
    total_cost, remaining = 0.0, q
    for o in self.market.order_book['ask']:
        if remaining <= 0:
            break
        fill = min(remaining, o.qty)
        total_cost += fill * o.price
        remaining -= fill
    filled = q - remaining
    unfilled = _orig_buy_market(self, quantity)
    if filled > 0 and mid and mid > 0:
        vwap = total_cost / filled
        _captured_clob_trades.append({
            'venue': 'clob', 'side': 'buy', 'quantity': filled,
            'exec_price': vwap,
            'cost_bps': 10_000.0 * (vwap - mid) / mid,
        })
    return unfilled


def _capturing_sell_market(self, quantity):
    q = round(quantity)
    if q <= 0 or not self.market.order_book['bid']:
        return _orig_sell_market(self, quantity)
    mid = self.market.price()
    total_cost, remaining = 0.0, q
    for o in self.market.order_book['bid']:
        if remaining <= 0:
            break
        fill = min(remaining, o.qty)
        total_cost += fill * o.price
        remaining -= fill
    filled = q - remaining
    unfilled = _orig_sell_market(self, quantity)
    if filled > 0 and mid and mid > 0:
        vwap = total_cost / filled
        _captured_clob_trades.append({
            'venue': 'clob', 'side': 'sell', 'quantity': filled,
            'exec_price': vwap,
            'cost_bps': 10_000.0 * (mid - vwap) / mid,
        })
    return unfilled


Trader._buy_market = _capturing_buy_market
Trader._sell_market = _capturing_sell_market

# -- Monkey-patch: capture AMM pool executions (arbitrageur trades) ----
_captured_amm_trades: list = []

_orig_cpmm_exec_buy = CPMMPool.execute_buy
_orig_cpmm_exec_sell = CPMMPool.execute_sell
_orig_hfmm_exec_buy = HFMMPool.execute_buy
_orig_hfmm_exec_sell = HFMMPool.execute_sell


def _make_amm_capture(orig_fn, side):
    def wrapper(self, Q):
        result = orig_fn(self, Q)
        if result.get('cost_bps', float('inf')) != float('inf'):
            name = getattr(self, 'pool_name', 'amm')
            _captured_amm_trades.append({
                'venue': name,
                'side': side,
                'quantity': round(Q, 6),
                'exec_price': result.get('exec_price', 0),
                'cost_bps': result.get('cost_bps', 0),
            })
        return result
    return wrapper


CPMMPool.execute_buy = _make_amm_capture(_orig_cpmm_exec_buy, 'buy')
CPMMPool.execute_sell = _make_amm_capture(_orig_cpmm_exec_sell, 'sell')
HFMMPool.execute_buy = _make_amm_capture(_orig_hfmm_exec_buy, 'buy')
HFMMPool.execute_sell = _make_amm_capture(_orig_hfmm_exec_sell, 'sell')


# -- Simulation helpers ------------------------------------------------

def _seed_all():
    random.seed(SEED)
    np.random.seed(SEED)


def _make_env():
    return MarketEnvironment(
        sigma_low=0.01, sigma_high=0.05,
        c_low=0.001, c_high=0.02,
        stress_start=-1, stress_end=-1,
        mode='piecewise',
        price=PRICE,
    )


# -- Custom simulation loop -------------------------------------------

def run_simulation(config_id, exchange, env, clob, amm_pools,
                   clob_agents, fx_traders,
                   lp_providers, arbitrageur,
                   sim_info=None):
    records = []

    for t in range(N_ITER):

        # 0. Price shock
        if t == SHOCK_ITER:
            env.apply_shock(SHOCK_PCT)
            try:
                mid = exchange.price()
            except Exception:
                mid = PRICE
            dp = mid * (SHOCK_PCT / 100.0)
            for order in _chain(*exchange.order_book.values()):
                order.price = round(order.price + dp, 2)
            for agent in clob_agents[:min(3, len(clob_agents))]:
                try:
                    agent.call()
                except Exception:
                    pass

        # 1. Environment step
        env.step()

        # 1b. Pre-trade arbitrage
        _captured_amm_trades.clear()
        if arbitrageur is not None:
            arbitrageur.arbitrage()

        # 2. SimulatorInfo capture + behaviour changes
        if sim_info is not None:
            try:
                sim_info.capture()
                for agent in clob_agents:
                    if type(agent) is Universalist:
                        agent.change_strategy(sim_info)
                    elif type(agent) is Chartist:
                        agent.change_sentiment(sim_info)
            except Exception:
                pass

        # 4. Shuffle & call all CLOB agents
        _captured_clob_trades.clear()
        random.shuffle(clob_agents)
        for agent in clob_agents:
            try:
                agent.call()
            except Exception:
                pass

        # 5. Dividends
        exchange.generate_dividend()

        # 6. Reset AMM period fees
        for pool in amm_pools.values():
            pool.reset_period_fees()

        # 6b. Drain captured CLOB-agent market orders + pre-trade arb
        period_trades = list(_captured_clob_trades)
        _captured_clob_trades.clear()
        period_trades.extend(_captured_amm_trades)
        _captured_amm_trades.clear()

        # 7. FX liquidity takers
        random.shuffle(fx_traders)
        for tr in fx_traders:
            result = tr.call()
            if result is not None:
                period_trades.append(result)

        # 8. Post-trade arbitrage
        if arbitrageur is not None:
            arbitrageur.arbitrage()

        # 8b. Drain captured AMM arb trades
        period_trades.extend(_captured_amm_trades)
        _captured_amm_trades.clear()

        # 9. LP adjust liquidity
        for lp in lp_providers:
            lp.update_liquidity()

        # 10. Record AMM state
        for pool in amm_pools.values():
            pool.record_state()

        # 11. Collect trade records
        fair_mid = env.fair_price if env.fair_price is not None else PRICE
        for tr in period_trades:
            exec_price = tr.get('exec_price')
            cost_bps = tr.get('cost_bps', 0.0)
            if exec_price is None or exec_price == float('inf'):
                sign = 1.0 if tr.get('side') == 'buy' else -1.0
                exec_price = fair_mid * (1.0 + sign * cost_bps / 10_000.0)
            records.append({
                'config_id': config_id,
                't':          t,
                'venue':      tr.get('venue', 'unknown'),
                'side':       tr.get('side', 'unknown'),
                'quantity':   tr.get('quantity', 0),
                'fair_mid':   round(fair_mid, 6),
                'exec_price': round(exec_price, 6),
                'cost_bps':   round(cost_bps, 4),
            })

    return pd.DataFrame(records)


# -- Block A helpers ---------------------------------------------------

def _enumerate_clob_configs():
    """Yield (n_mm, n_rnd, n_fund, n_chart, n_univ) with total=5, n_mm <= 1."""
    for n_mm in range(2):
        rest = 5 - n_mm
        for n_rnd in range(rest + 1):
            for n_fund in range(rest - n_rnd + 1):
                for n_chart in range(rest - n_rnd - n_fund + 1):
                    n_univ = rest - n_rnd - n_fund - n_chart
                    yield (n_mm, n_rnd, n_fund, n_chart, n_univ)


def _build_clob_agents(exchange, env, n_mm, n_rnd, n_fund, n_chart, n_univ):
    agents = []
    for i in range(n_mm):
        agents.append(MarketMaker(
            exchange, cash=1e4, env=env,
            alpha0=3.0 + i * 0.5, alpha1=300.0, alpha2=200.0,
            d0=50.0 - i * 5.0, d1=500.0, d2=250.0, d_min=5.0,
        ))
    for _ in range(n_rnd):
        agents.append(Random(exchange, cash=1e4))
    for _ in range(n_fund):
        agents.append(Fundamentalist(exchange, cash=1e4, access=1))
    for _ in range(n_chart):
        agents.append(Chartist(exchange, cash=1e4))
    for _ in range(n_univ):
        agents.append(Universalist(exchange, cash=1e4, access=1))
    return agents


def _config_id_a(n_mm, n_rnd, n_fund, n_chart, n_univ):
    return (f'CLOB_config_mm{n_mm}_rnd{n_rnd}'
            f'_fund{n_fund}_chart{n_chart}_univ{n_univ}')


# -- Block A runner ----------------------------------------------------

def run_block_a():
    configs = list(_enumerate_clob_configs())
    print('=' * 70)
    print(f'  BLOCK A -- CLOB agent compositions  ({len(configs)} configs, no AMM)')
    print(f'  5 agents total, <=1 MM.  Types: MM, Random, Fund, Chartist, Univ')
    print('=' * 70)

    for idx, (n_mm, n_rnd, n_fund, n_chart, n_univ) in enumerate(configs, 1):
        config_id = _config_id_a(n_mm, n_rnd, n_fund, n_chart, n_univ)
        total = n_mm + n_rnd + n_fund + n_chart + n_univ
        print(f'\n  [{idx}/{len(configs)}] {config_id} (total={total}) ...',
              end=' ', flush=True)

        _seed_all()

        env       = _make_env()
        exchange  = ExchangeAgent(price=PRICE, std=2.0, volume=DEPTH)
        clob      = CLOBVenue(exchange)
        amm_pools = {}

        clob_agents = _build_clob_agents(
            exchange, env, n_mm, n_rnd, n_fund, n_chart, n_univ)

        sim_info = None
        if n_chart > 0 or n_univ > 0:
            sim_info = SimulatorInfo(exchange, clob_agents)

        df = run_simulation(
            config_id, exchange, env, clob, amm_pools,
            clob_agents=clob_agents, fx_traders=[],
            lp_providers=[], arbitrageur=None,
            sim_info=sim_info,
        )

        path = os.path.join(OUT_DIR, f'{config_id}.csv')
        df.to_csv(path, index=False)
        print(f'{len(df)} trades  ->  {path}')


# -- Block B runner ----------------------------------------------------

def run_block_b():
    print('\n' + '=' * 70)
    print('  BLOCK B -- AMM configurations  (CPMM + HFMM <= 5, fully autonomous)')
    print('=' * 70)

    configs = []
    for total in range(1, 6):
        for n_cpmm in range(total + 1):
            configs.append((n_cpmm, total - n_cpmm))

    for n_cpmm, n_hfmm in configs:
        config_id = f'AMM_config_cpmm{n_cpmm}_hfmm{n_hfmm}'
        print(f'\n  [{config_id}] ...', end=' ', flush=True)

        _seed_all()

        env      = _make_env()
        exchange = ExchangeAgent(price=PRICE, std=2.0, volume=DEPTH)
        clob     = ShadowCLOB(env)

        amm_pools  = {}
        lp_providers = []

        for i in range(n_cpmm):
            name = f'cpmm_{i}' if n_cpmm > 1 else 'cpmm'
            pool = CPMMPool(x=float(DEPTH), y=float(DEPTH) * PRICE, fee=0.003)
            pool.pool_name = name
            amm_pools[name] = pool
            lp_providers.append(AMMProvider(pool, env))

        for i in range(n_hfmm):
            name = f'hfmm_{i}' if n_hfmm > 1 else 'hfmm'
            pool = HFMMPool(x=float(DEPTH), y=float(DEPTH) * PRICE,
                            A=10.0, fee=0.001, rate=PRICE)
            pool.pool_name = name
            amm_pools[name] = pool
            lp_providers.append(AMMProvider(pool, env))

        arb = AMMArbitrageur(clob, amm_pools, env=env) if amm_pools else None
        fx_traders = []

        df = run_simulation(
            config_id, exchange, env, clob, amm_pools,
            clob_agents=[],
            fx_traders=fx_traders,
            lp_providers=lp_providers, arbitrageur=arb,
        )

        path = os.path.join(OUT_DIR, f'{config_id}.csv')
        df.to_csv(path, index=False)
        print(f'{len(df)} trades  ->  {path}')


# -- Block C: heterogeneous HFMM pools ---------------------------------

HFMM_VARIANTS = {
    'tight':    {'fee': 0.0005, 'A': 50.0, 'reserves': 500},
    'balanced': {'fee': 0.001,  'A': 10.0, 'reserves': 1000},
    'deep':     {'fee': 0.002,  'A': 5.0,  'reserves': 2000},
}


def _enumerate_amm_c_configs():
    """Yield (n_tight, n_balanced, n_deep) with 1 <= total <= 5."""
    for total in range(1, 6):
        for n_tight in range(total + 1):
            for n_balanced in range(total - n_tight + 1):
                n_deep = total - n_tight - n_balanced
                yield (n_tight, n_balanced, n_deep)


def _config_id_c(n_tight, n_balanced, n_deep):
    return f'AMM_C_config_t{n_tight}_b{n_balanced}_d{n_deep}'


def run_block_c():
    configs = list(_enumerate_amm_c_configs())
    print('\n' + '=' * 70)
    print(f'  BLOCK C -- Heterogeneous HFMM  ({len(configs)} configs, fully autonomous)')
    print(f'  Variants: tight (fee=5bp A=50 R=500), balanced (fee=10bp A=10 R=1000),')
    print(f'            deep (fee=20bp A=5 R=2000)')
    print('=' * 70)

    for n_tight, n_balanced, n_deep in configs:
        config_id = _config_id_c(n_tight, n_balanced, n_deep)
        print(f'\n  [{config_id}] ...', end=' ', flush=True)

        _seed_all()

        env      = _make_env()
        exchange = ExchangeAgent(price=PRICE, std=2.0, volume=DEPTH)
        clob     = ShadowCLOB(env)

        amm_pools    = {}
        lp_providers = []

        pool_idx = 0
        for variant_name, count in [('tight', n_tight),
                                     ('balanced', n_balanced),
                                     ('deep', n_deep)]:
            v = HFMM_VARIANTS[variant_name]
            for i in range(count):
                name = f'hfmm_{variant_name}_{pool_idx}'
                pool = HFMMPool(
                    x=float(v['reserves']),
                    y=float(v['reserves']) * PRICE,
                    A=v['A'], fee=v['fee'], rate=PRICE,
                )
                pool.pool_name = name
                amm_pools[name] = pool
                lp_providers.append(AMMProvider(pool, env))
                pool_idx += 1

        arb = AMMArbitrageur(clob, amm_pools, env=env,
                             routing='best') if amm_pools else None

        df = run_simulation(
            config_id, exchange, env, clob, amm_pools,
            clob_agents=[],
            fx_traders=[],
            lp_providers=lp_providers, arbitrageur=arb,
        )

        path = os.path.join(OUT_DIR, f'{config_id}.csv')
        df.to_csv(path, index=False)
        print(f'{len(df)} trades  ->  {path}')


# -- Block D: Multi-venue via Simulator.default_fx() ------------------

AMM_SHARE_GRID = list(range(0, 101, 10))  # 0%, 10%, ..., 100%


def run_block_d():
    """Sweep amm_share_pct using full Simulator.default_fx() architecture.

    Identical to main.py: 30 CLOB noise + 1 MM + 33 FX takers
    (15 Noise + 5 Fund + 10 Retail + 3 Institutional) + CPMM + HFMM + arb.
    """
    print('\n' + '=' * 70)
    print(f'  BLOCK D -- Multi-venue via default_fx()  ({len(AMM_SHARE_GRID)} configs)')
    print(f'  Architecture: identical to main.py (30 noise + 1 MM + 33 FX takers)')
    print(f'  AMM: CPMM (fee=30bp) + HFMM (fee=10bp, A=10)')
    print(f'  Sweep: amm_share_pct = {AMM_SHARE_GRID}')
    print('=' * 70)

    for amm_pct in AMM_SHARE_GRID:
        config_id = f'MV_config_amm{amm_pct}pct'
        print(f'\n  [{config_id}] ...', end=' ', flush=True)

        _seed_all()

        sim = Simulator.default_fx(
            price=PRICE,
            clob_volume=DEPTH,
            amm_share_pct=amm_pct,
            stress_start=-1,    # no gradual stress, only shock
            stress_end=-1,
            shock_iter=None,    # handled by run_simulation loop
        )

        # Extract components for run_simulation
        clob_agents = list(sim.clob_noise)
        if sim.mm:
            clob_agents.append(sim.mm)

        df = run_simulation(
            config_id, sim.exchange, sim.env, sim.clob, sim.amm_pools,
            clob_agents=clob_agents,
            fx_traders=sim.fx_traders,
            lp_providers=sim.lp_providers, arbitrageur=sim.arbitrageur,
            sim_info=None,
        )

        path = os.path.join(OUT_DIR, f'{config_id}.csv')
        df.to_csv(path, index=False)
        print(f'{len(df)} trades  ->  {path}')


# ######################################################################
#  PART 2 -- ANALYSIS  (reads CSVs, prints report)
# ######################################################################

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 220)
pd.set_option('display.max_rows', 200)
pd.set_option('display.float_format', '{:.2f}'.format)


def _parse_clob_name(fname):
    m = re.match(r'CLOB_config_mm(\d+)_rnd(\d+)_fund(\d+)_chart(\d+)_univ(\d+)', fname)
    if not m:
        return {}
    return dict(n_mm=int(m[1]), n_rnd=int(m[2]), n_fund=int(m[3]),
                n_chart=int(m[4]), n_univ=int(m[5]))


def _parse_amm_name(fname):
    m = re.match(r'AMM_config_cpmm(\d+)_hfmm(\d+)', fname)
    if not m:
        return {}
    return dict(n_cpmm=int(m[1]), n_hfmm=int(m[2]))


def _parse_amm_c_name(fname):
    m = re.match(r'AMM_C_config_t(\d+)_b(\d+)_d(\d+)', fname)
    if not m:
        return {}
    return dict(n_tight=int(m[1]), n_balanced=int(m[2]), n_deep=int(m[3]))


def _parse_mv_name(fname):
    m = re.match(r'MV_config_amm(\d+)pct', fname)
    if not m:
        return {}
    return dict(amm_pct=int(m[1]))


def load_block(prefix):
    files = sorted(glob.glob(os.path.join(OUT_DIR, f'{prefix}_*.csv')))
    if prefix == 'AMM':
        files = [f for f in files if not os.path.basename(f).startswith('AMM_C_')]
    frames = []
    for f in files:
        df = pd.read_csv(f)
        fname = os.path.basename(f).replace('.csv', '')
        df['config_id'] = fname
        if prefix == 'CLOB':
            for k, v in _parse_clob_name(fname).items():
                df[k] = v
        elif prefix == 'AMM_C':
            for k, v in _parse_amm_c_name(fname).items():
                df[k] = v
        elif prefix == 'MV':
            for k, v in _parse_mv_name(fname).items():
                df[k] = v
        else:
            for k, v in _parse_amm_name(fname).items():
                df[k] = v
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def add_fv_cost(df):
    """
    Fair-value execution cost (bps):
      buy  -> 10000 * (exec_price - fair_mid) / fair_mid
      sell -> 10000 * (fair_mid - exec_price) / fair_mid
    Positive = trader paid more than fair value (real cost).
    """
    df = df.copy()
    df['fv_cost'] = np.where(
        df['side'] == 'buy',
        10000 * (df['exec_price'] - df['fair_mid']) / df['fair_mid'],
        10000 * (df['fair_mid'] - df['exec_price']) / df['fair_mid'],
    )
    return df


def _section(title):
    print('\n' + '=' * 80)
    print(f'  {title}')
    print('=' * 80)


# -- Recovery time computation -----------------------------------------

RECOVERY_WINDOW    = 20     # rolling window (periods)
RECOVERY_THRESHOLD = 20.0   # bps: recovered when within this of baseline
WARMUP_PERIODS     = 50     # skip initial periods for baseline


def compute_recovery_time(df, shock_iter=SHOCK_ITER, window=RECOVERY_WINDOW,
                          threshold=RECOVERY_THRESHOLD, warmup=WARMUP_PERIODS):
    """Periods after shock until rolling(window) median fv_cost returns
    to within ±threshold of pre-shock baseline.

    Returns (recovery_time, baseline_fv).
    recovery_time: int (periods), inf (never recovered), or nan (no data).
    """
    fin = df[np.isfinite(df['fv_cost'])].copy()
    pre = fin[(fin['t'] >= warmup) & (fin['t'] < shock_iter)]
    if len(pre) < window:
        return np.nan, np.nan
    baseline = pre['fv_cost'].median()

    post = fin[fin['t'] >= shock_iter]
    if len(post) == 0:
        return np.nan, baseline

    per_t = post.groupby('t')['fv_cost'].median().sort_index()
    roll = per_t.rolling(window, min_periods=window).median()
    within = (roll - baseline).abs() <= threshold
    recovered = within[within]
    if len(recovered) == 0:
        return float('inf'), baseline
    return max(0, recovered.index[0] - shock_iter), baseline


def _recovery_summary(rec_series, label):
    """Print one-line recovery summary for a series of recovery times."""
    finite = rec_series[np.isfinite(rec_series)]
    n_inf = (rec_series == float('inf')).sum()
    n_nan = rec_series.isna().sum()
    if len(finite):
        print(f'    {label}:  median={finite.median():.0f}  '
              f'mean={finite.mean():.0f}  min={finite.min():.0f}  '
              f'max={finite.max():.0f}  never={n_inf}  nodata={n_nan}')
    elif n_inf:
        print(f'    {label}:  never recovered ({n_inf} configs)')
    else:
        print(f'    {label}:  no data')


# -- 1. Data overview --------------------------------------------------

def report_overview(clob, amm):
    _section('1. DATA OVERVIEW')
    n_clob_cfg = clob['config_id'].nunique() if len(clob) else 0
    n_amm_cfg  = amm['config_id'].nunique() if len(amm) else 0
    print(f'  CLOB configs : {n_clob_cfg}   ({len(clob):,} trade records)')
    print(f'  AMM  configs : {n_amm_cfg}   ({len(amm):,} trade records)')
    print(f'  Iterations   : {N_ITER} per config,  shock at t={SHOCK_ITER}')

    if len(clob):
        n_inf = (~np.isfinite(clob['cost_bps'])).sum()
        print(f'\n  CLOB data quality:')
        print(f'    Total trades           : {len(clob):,}')
        print(f'    Inf cost_bps (unfilled): {n_inf:,}  ({100*n_inf/len(clob):.1f}%)')
        print(f'    Fillable trades        : {len(clob)-n_inf:,}')

    if len(amm):
        n_inf_a = (~np.isfinite(amm['cost_bps'])).sum()
        print(f'\n  AMM data quality:')
        print(f'    Total trades           : {len(amm):,}')
        print(f'    Inf cost_bps           : {n_inf_a:,}  ({100*n_inf_a/len(amm):.1f}%)')
        print(f'    -> AMMs always fill orders (inf should be 0)')


# -- 2. Block A -- CLOB analysis ---------------------------------------

def report_block_a(clob):
    _section('2. BLOCK A -- CLOB: Effect of Agent Composition')

    if not len(clob):
        print('  (no data)')
        return None

    grp = clob.groupby('config_id').agg(
        n_mm   = ('n_mm', 'first'),
        n_rnd  = ('n_rnd', 'first'),
        n_fund = ('n_fund', 'first'),
        n_chart= ('n_chart', 'first'),
        n_univ = ('n_univ', 'first'),
        total  = ('cost_bps', 'size'),
        inf    = ('cost_bps', lambda s: (~np.isfinite(s)).sum()),
    ).reset_index()
    grp['fill_rate'] = 100 * (1 - grp['inf'] / grp['total'])

    filled = clob[np.isfinite(clob['cost_bps']) & np.isfinite(clob['fv_cost'])].copy()
    filled = filled[filled['fv_cost'].between(-5000, 5000)]

    med_fv = filled.groupby('config_id')['fv_cost'].agg(
        fv_cost_med='median', fv_cost_mean='mean',
    ).reset_index()

    cost_agg = filled.groupby('config_id')['cost_bps'].agg(
        cost_med='median', cost_mean='mean',
    ).reset_index()

    grp = grp.merge(med_fv, on='config_id', how='left')
    grp = grp.merge(cost_agg, on='config_id', how='left')

    # MM presence
    print('\n  2a. Effect of Market Maker (MM=0 vs MM=1):')
    print('  ' + '-' * 60)
    for mm_val in [0, 1]:
        sub = grp[grp['n_mm'] == mm_val]
        print(f'    MM={mm_val}  ({len(sub)} configs):')
        print(f'      Fill rate       : {sub["fill_rate"].mean():.1f}% '
              f'(min {sub["fill_rate"].min():.1f}%, max {sub["fill_rate"].max():.1f}%)')
        print(f'      Median fv_cost  : {sub["fv_cost_med"].median():.1f} bps')
        print(f'      Mean   fv_cost  : {sub["fv_cost_mean"].mean():.1f} bps')

    # Top/Bottom by fill rate
    print('\n  2b. Top 10 configs by fill rate:')
    print('  ' + '-' * 60)
    top = grp.nlargest(10, 'fill_rate')
    for _, r in top.iterrows():
        print(f'    mm={int(r.n_mm)} rnd={int(r.n_rnd)} fund={int(r.n_fund)} '
              f'chart={int(r.n_chart)} univ={int(r.n_univ)}  '
              f'fill={r.fill_rate:.1f}%  fv_med={r.fv_cost_med:.0f} bps')

    print('\n  2c. Bottom 10 configs by fill rate:')
    print('  ' + '-' * 60)
    bot = grp.nsmallest(10, 'fill_rate')
    for _, r in bot.iterrows():
        print(f'    mm={int(r.n_mm)} rnd={int(r.n_rnd)} fund={int(r.n_fund)} '
              f'chart={int(r.n_chart)} univ={int(r.n_univ)}  '
              f'fill={r.fill_rate:.1f}%  fv_med={r.fv_cost_med:.0f} bps')

    # Top/Bottom by fv_cost
    print('\n  2d. Top 10 configs by LOWEST median fv_cost (cheapest execution):')
    print('  ' + '-' * 60)
    cheap = grp.dropna(subset=['fv_cost_med']).nsmallest(10, 'fv_cost_med')
    for _, r in cheap.iterrows():
        print(f'    mm={int(r.n_mm)} rnd={int(r.n_rnd)} fund={int(r.n_fund)} '
              f'chart={int(r.n_chart)} univ={int(r.n_univ)}  '
              f'fill={r.fill_rate:.1f}%  fv_med={r.fv_cost_med:.1f} bps')

    print('\n  2e. Top 10 configs by HIGHEST median fv_cost (most expensive):')
    print('  ' + '-' * 60)
    expen = grp.dropna(subset=['fv_cost_med']).nlargest(10, 'fv_cost_med')
    for _, r in expen.iterrows():
        print(f'    mm={int(r.n_mm)} rnd={int(r.n_rnd)} fund={int(r.n_fund)} '
              f'chart={int(r.n_chart)} univ={int(r.n_univ)}  '
              f'fill={r.fill_rate:.1f}%  fv_med={r.fv_cost_med:.1f} bps')

    # Per-agent-type marginal effect
    print('\n  2f. Marginal effect of each agent type (average across configs):')
    print('  ' + '-' * 60)
    for col, label in [('n_mm','MarketMaker'), ('n_rnd','Random'),
                       ('n_fund','Fundamentalist'), ('n_chart','Chartist'),
                       ('n_univ','Universalist')]:
        agg = grp.groupby(col).agg(
            fill_rate=('fill_rate', 'mean'),
            fv_cost_med=('fv_cost_med', 'median'),
            count=('config_id', 'count'),
        )
        print(f'\n    {label}:')
        for val, row in agg.iterrows():
            print(f'      n={int(val):d}  ({int(row["count"]):2d} cfgs)  '
                  f'fill={row["fill_rate"]:.1f}%  fv_med={row["fv_cost_med"]:.1f} bps')

    # 2g. Recovery time
    print('\n  2g. Post-shock recovery time (periods after t=%d):' % SHOCK_ITER)
    print('  ' + '-' * 60)

    rec_rows = []
    for cid in clob['config_id'].unique():
        sub = clob[clob['config_id'] == cid]
        rt, bl = compute_recovery_time(sub)
        meta = _parse_clob_name(cid)
        rec_rows.append({**meta, 'config_id': cid, 'recovery_t': rt})
    rec_df = pd.DataFrame(rec_rows)
    grp = grp.merge(rec_df[['config_id', 'recovery_t']], on='config_id', how='left')

    for mm_val in [0, 1]:
        sub = rec_df[rec_df['n_mm'] == mm_val]
        _recovery_summary(sub['recovery_t'], f'MM={mm_val} ({len(sub)} cfgs)')

    print('\n    Per-agent-type marginal effect on recovery:')
    for col, label in [('n_mm','MarketMaker'), ('n_rnd','Random'),
                       ('n_fund','Fundamentalist'), ('n_chart','Chartist'),
                       ('n_univ','Universalist')]:
        for val in sorted(rec_df[col].unique()):
            sub = rec_df[rec_df[col] == val]
            fin = sub['recovery_t'][np.isfinite(sub['recovery_t'])]
            med_str = f'{fin.median():.0f}' if len(fin) else 'n/a'
            print(f'      {label} n={int(val):d}  ({len(sub):2d} cfgs)  '
                  f'recovery_med={med_str} periods')

    # Top 5 fastest / slowest recovery
    ranked = rec_df[np.isfinite(rec_df['recovery_t'])].sort_values('recovery_t')
    if len(ranked):
        print('\n    Top 5 fastest recovery:')
        for _, r in ranked.head(5).iterrows():
            print(f'      mm={int(r.n_mm)} rnd={int(r.n_rnd)} fund={int(r.n_fund)} '
                  f'chart={int(r.n_chart)} univ={int(r.n_univ)}  '
                  f'recovery={r.recovery_t:.0f} periods')
        print('\n    Top 5 slowest recovery:')
        for _, r in ranked.tail(5).iterrows():
            print(f'      mm={int(r.n_mm)} rnd={int(r.n_rnd)} fund={int(r.n_fund)} '
                  f'chart={int(r.n_chart)} univ={int(r.n_univ)}  '
                  f'recovery={r.recovery_t:.0f} periods')

    return grp


# -- 3. Block B -- AMM analysis ----------------------------------------

def report_block_b(amm):
    _section('3. BLOCK B -- AMM: Effect of Pool Composition')

    if not len(amm):
        print('  (no data)')
        return None

    grp = amm.groupby('config_id').agg(
        n_cpmm  = ('n_cpmm', 'first'),
        n_hfmm  = ('n_hfmm', 'first'),
        total   = ('cost_bps', 'size'),
        cost_med= ('cost_bps', 'median'),
        cost_mean=('cost_bps', 'mean'),
        fv_med  = ('fv_cost', 'median'),
        fv_mean = ('fv_cost', 'mean'),
    ).reset_index()
    grp['n_pools'] = grp['n_cpmm'] + grp['n_hfmm']

    print('\n  3a. All AMM configs (sorted by median cost_bps):')
    print('  ' + '-' * 60)
    for _, r in grp.sort_values('cost_med').iterrows():
        print(f'    cpmm={int(r.n_cpmm)} hfmm={int(r.n_hfmm)}  '
              f'trades={int(r.total):5d}  cost_med={r.cost_med:.1f}  '
              f'cost_mean={r.cost_mean:.1f}  fv_med={r.fv_med:.1f} bps')

    # Effect of pool type
    print('\n  3b. CPMM vs HFMM (marginal effect):')
    print('  ' + '-' * 60)
    for col, label in [('n_cpmm','CPMM'), ('n_hfmm','HFMM')]:
        agg = grp.groupby(col).agg(
            cost_med=('cost_med', 'mean'),
            fv_med=('fv_med', 'mean'),
            count=('config_id', 'count'),
        )
        print(f'\n    {label}:')
        for val, row in agg.iterrows():
            print(f'      n={int(val):d}  ({int(row["count"]):2d} cfgs)  '
                  f'cost_med={row["cost_med"]:.1f}  fv_med={row["fv_med"]:.1f} bps')

    # Effect of total pool count
    print('\n  3c. Effect of total pool count:')
    print('  ' + '-' * 60)
    for n, sub in grp.groupby('n_pools'):
        print(f'    {int(n)} pools  ({len(sub)} cfgs):  '
              f'cost_med={sub["cost_med"].mean():.1f}  fv_med={sub["fv_med"].mean():.1f} bps')

    # 3d. Recovery time
    print('\n  3d. Post-shock recovery time (periods after t=%d):' % SHOCK_ITER)
    print('  ' + '-' * 60)

    rec_rows = []
    for cid in amm['config_id'].unique():
        sub = amm[amm['config_id'] == cid]
        rt, bl = compute_recovery_time(sub)
        meta = _parse_amm_name(cid)
        rec_rows.append({**meta, 'config_id': cid, 'recovery_t': rt})
    rec_df = pd.DataFrame(rec_rows)
    grp = grp.merge(rec_df[['config_id', 'recovery_t']], on='config_id', how='left')

    for _, r in rec_df.sort_values('recovery_t').iterrows():
        rt_str = f'{r.recovery_t:.0f}' if np.isfinite(r.recovery_t) else 'never'
        print(f'    cpmm={int(r.n_cpmm)} hfmm={int(r.n_hfmm)}  recovery={rt_str} periods')

    # CPMM vs HFMM effect
    for col, label in [('n_cpmm', 'CPMM'), ('n_hfmm', 'HFMM')]:
        print(f'\n    {label} marginal effect on recovery:')
        for val in sorted(rec_df[col].unique()):
            sub = rec_df[rec_df[col] == val]
            _recovery_summary(sub['recovery_t'], f'n={int(val)} ({len(sub)} cfgs)')

    return grp


# -- 3c. Block C -- heterogeneous HFMM --------------------------------

def report_block_c(amm_c):
    _section('3c. BLOCK C -- Heterogeneous HFMM Pools')

    if not len(amm_c):
        print('  (no data)')
        return None

    grp = amm_c.groupby('config_id').agg(
        n_tight   = ('n_tight', 'first'),
        n_balanced= ('n_balanced', 'first'),
        n_deep    = ('n_deep', 'first'),
        total     = ('cost_bps', 'size'),
        cost_med  = ('cost_bps', 'median'),
        cost_mean = ('cost_bps', 'mean'),
        fv_med    = ('fv_cost', 'median'),
        fv_mean   = ('fv_cost', 'mean'),
    ).reset_index()
    grp['n_pools'] = grp['n_tight'] + grp['n_balanced'] + grp['n_deep']
    grp['is_homo'] = (((grp['n_tight'] > 0).astype(int) +
                       (grp['n_balanced'] > 0).astype(int) +
                       (grp['n_deep'] > 0).astype(int)) == 1)

    print('\n  3c-1. All configs (sorted by median cost_bps):')
    print('  ' + '-' * 75)
    for _, r in grp.sort_values('cost_med').iterrows():
        tag = 'HOMO' if r.is_homo else 'MIX '
        print(f'    t={int(r.n_tight)} b={int(r.n_balanced)} d={int(r.n_deep)}'
              f'  [{tag}]  trades={int(r.total):5d}'
              f'  cost_med={r.cost_med:.1f}  cost_mean={r.cost_mean:.1f}'
              f'  fv_med={r.fv_med:.1f} bps')

    # Homogeneous vs heterogeneous
    homo = grp[grp['is_homo']]
    hetero = grp[~grp['is_homo']]
    print(f'\n  3c-2. Homogeneous vs Heterogeneous:')
    print('  ' + '-' * 60)
    if len(homo):
        print(f'    Homogeneous  ({len(homo):2d} cfgs):  '
              f'cost_med avg = {homo["cost_med"].mean():.1f} bps')
    if len(hetero):
        print(f'    Heterogeneous ({len(hetero):2d} cfgs):  '
              f'cost_med avg = {hetero["cost_med"].mean():.1f} bps')

    # Per-variant marginal effect
    print(f'\n  3c-3. Marginal effect of each variant:')
    print('  ' + '-' * 60)
    for col, label in [('n_tight', 'Tight (fee=5bp A=50 R=500)'),
                       ('n_balanced', 'Balanced (fee=10bp A=10 R=1000)'),
                       ('n_deep', 'Deep (fee=20bp A=5 R=2000)')]:
        agg = grp.groupby(col).agg(
            cost_med=('cost_med', 'mean'),
            fv_med=('fv_med', 'mean'),
            count=('config_id', 'count'),
        )
        print(f'\n    {label}:')
        for val, row in agg.iterrows():
            print(f'      n={int(val):d}  ({int(row["count"]):2d} cfgs)  '
                  f'cost_med={row["cost_med"]:.1f}  fv_med={row["fv_med"]:.1f} bps')

    # Best vs Block B reference
    best_c = grp.loc[grp['cost_med'].idxmin()]
    print(f'\n  3c-4. Best Block C vs Block B reference (HFMM=1, 63.0 bps):')
    print('  ' + '-' * 60)
    print(f'    Block B best : HFMM=1 (fee=10bp A=10 R=1000)  cost_med = 63.0 bps')
    print(f'    Block C best : t={int(best_c.n_tight)} b={int(best_c.n_balanced)} '
          f'd={int(best_c.n_deep)}  cost_med = {best_c.cost_med:.1f} bps')

    # 3c-5. Recovery time
    print(f'\n  3c-5. Post-shock recovery time (periods after t={SHOCK_ITER}):')
    print('  ' + '-' * 60)

    rec_rows = []
    for cid in amm_c['config_id'].unique():
        sub = amm_c[amm_c['config_id'] == cid]
        rt, bl = compute_recovery_time(sub)
        meta = _parse_amm_c_name(cid)
        rec_rows.append({**meta, 'config_id': cid, 'recovery_t': rt})
    rec_df = pd.DataFrame(rec_rows)
    grp = grp.merge(rec_df[['config_id', 'recovery_t']], on='config_id', how='left')

    # Homo vs hetero recovery
    rec_df['n_pools'] = rec_df['n_tight'] + rec_df['n_balanced'] + rec_df['n_deep']
    rec_df['is_homo'] = (((rec_df['n_tight'] > 0).astype(int) +
                          (rec_df['n_balanced'] > 0).astype(int) +
                          (rec_df['n_deep'] > 0).astype(int)) == 1)
    homo = rec_df[rec_df['is_homo']]
    hetero = rec_df[~rec_df['is_homo']]
    _recovery_summary(homo['recovery_t'], f'Homogeneous  ({len(homo)} cfgs)')
    _recovery_summary(hetero['recovery_t'], f'Heterogeneous ({len(hetero)} cfgs)')

    # Per-variant marginal
    print('\n    Per-variant marginal effect on recovery:')
    for col, label in [('n_tight', 'Tight'), ('n_balanced', 'Balanced'),
                       ('n_deep', 'Deep')]:
        for val in sorted(rec_df[col].unique()):
            sub = rec_df[rec_df[col] == val]
            fin = sub['recovery_t'][np.isfinite(sub['recovery_t'])]
            med_str = f'{fin.median():.0f}' if len(fin) else 'n/a'
            print(f'      {label} n={int(val):d}  ({len(sub):2d} cfgs)  '
                  f'recovery_med={med_str} periods')

    return grp


# -- 3d. Block D -- multi-venue CLOB+AMM -------------------------------

def report_block_d(mv):
    _section('3d. BLOCK D -- Multi-venue CLOB+AMM')

    if not len(mv):
        print('  (no data)')
        return None

    # Filter to finite cost_bps for aggregation
    mv_fin = mv[np.isfinite(mv['cost_bps']) & np.isfinite(mv['fv_cost'])].copy()
    n_inf = len(mv) - len(mv_fin)
    if n_inf:
        print(f'\n  Note: {n_inf} trades with inf cost_bps excluded ({100*n_inf/len(mv):.1f}%)')

    # Per-config aggregate
    grp = mv_fin.groupby('config_id').agg(
        amm_pct   = ('amm_pct', 'first'),
        total     = ('cost_bps', 'size'),
        cost_med  = ('cost_bps', 'median'),
        cost_mean = ('cost_bps', 'mean'),
        fv_med    = ('fv_cost', 'median'),
        fv_mean   = ('fv_cost', 'mean'),
    ).reset_index().sort_values('amm_pct')

    # 3d-1. Overview sorted by amm_share_pct
    print('\n  3d-1. All configs (sorted by amm_share_pct):')
    print('  ' + '-' * 75)
    for _, r in grp.iterrows():
        print(f'    amm_pct={int(r.amm_pct):3d}%  trades={int(r.total):5d}'
              f'  cost_med={r.cost_med:.1f}  cost_mean={r.cost_mean:.1f}'
              f'  fv_med={r.fv_med:.1f} bps')

    # 3d-2. Venue split per config
    print('\n  3d-2. Venue split (CLOB vs AMM trades per config):')
    print('  ' + '-' * 75)
    for amm_pct in sorted(mv['amm_pct'].unique()):
        sub = mv[mv['amm_pct'] == amm_pct]
        n_clob = (sub['venue'] == 'clob').sum()
        n_amm  = (sub['venue'] != 'clob').sum()
        total  = len(sub)
        pct_amm = 100 * n_amm / total if total else 0

        sub_f = sub[np.isfinite(sub['fv_cost'])]
        clob_sub = sub_f[sub_f['venue'] == 'clob']
        amm_sub  = sub_f[sub_f['venue'] != 'clob']

        clob_med = clob_sub['fv_cost'].median() if len(clob_sub) else float('nan')
        amm_med  = amm_sub['fv_cost'].median() if len(amm_sub) else float('nan')

        n_inf_cfg = (sub['cost_bps'] == float('inf')).sum()
        inf_tag = f'  [inf={n_inf_cfg}]' if n_inf_cfg else ''

        print(f'    amm_pct={amm_pct:3d}%:  '
              f'CLOB={n_clob:4d} ({100-pct_amm:.0f}%)  '
              f'AMM={n_amm:4d} ({pct_amm:.0f}%)  '
              f'fv_clob={clob_med:.1f}  fv_amm={amm_med:.1f} bps{inf_tag}')

    # 3d-3. Pre/post shock resilience
    print(f'\n  3d-3. Shock resilience (pre/post t={SHOCK_ITER}):')
    print('  ' + '-' * 75)
    for amm_pct in sorted(mv['amm_pct'].unique()):
        sub = mv[mv['amm_pct'] == amm_pct]
        pre  = sub[sub['t'] < SHOCK_ITER]
        post = sub[sub['t'] >= SHOCK_ITER]
        pre_f  = pre[np.isfinite(pre['fv_cost']) & pre['fv_cost'].between(-5000, 5000)]
        post_f = post[np.isfinite(post['fv_cost']) & post['fv_cost'].between(-5000, 5000)]
        pre_m  = pre_f['fv_cost'].median() if len(pre_f) else float('nan')
        post_m = post_f['fv_cost'].median() if len(post_f) else float('nan')
        delta  = post_m - pre_m if np.isfinite(pre_m) and np.isfinite(post_m) else float('nan')
        print(f'    amm_pct={amm_pct:3d}%:  pre_fv={pre_m:.1f}  post_fv={post_m:.1f}'
              f'  delta={delta:+.1f} bps')

    # 3d-4. Best config
    best = grp.loc[grp['fv_med'].abs().idxmin()]
    print(f'\n  3d-4. Best multi-venue config (closest fv_cost to 0):')
    print('  ' + '-' * 60)
    print(f'    amm_share_pct = {int(best.amm_pct)}%')
    print(f'    trades        = {int(best.total):,}')
    print(f'    cost_bps med  = {best.cost_med:.1f} bps')
    print(f'    fv_cost  med  = {best.fv_med:.1f} bps')
    print(f'    |fv_cost|     = {abs(best.fv_med):.1f} bps')

    # 3d-5. Recovery time — combined + per-venue
    print(f'\n  3d-5. Post-shock recovery time (periods after t={SHOCK_ITER}):')
    print('  ' + '-' * 75)
    print(f'    {"amm_pct":>8s}  {"combined":>10s}  {"CLOB":>10s}  {"AMM":>10s}')
    print('    ' + '-' * 46)

    rec_rows = []
    for amm_pct in sorted(mv['amm_pct'].unique()):
        sub = mv[mv['amm_pct'] == amm_pct]
        rt_all, _ = compute_recovery_time(sub)
        clob_sub = sub[sub['venue'] == 'clob']
        amm_sub  = sub[sub['venue'] != 'clob']
        rt_clob, _ = compute_recovery_time(clob_sub) if len(clob_sub) else (np.nan, np.nan)
        rt_amm, _  = compute_recovery_time(amm_sub) if len(amm_sub) else (np.nan, np.nan)

        def _fmt(v):
            if v != v:  return 'n/a'
            if v == float('inf'):  return 'never'
            return f'{v:.0f}'
        print(f'    {amm_pct:7d}%  {_fmt(rt_all):>10s}  {_fmt(rt_clob):>10s}  {_fmt(rt_amm):>10s}')
        rec_rows.append({'amm_pct': amm_pct, 'recovery_all': rt_all,
                         'recovery_clob': rt_clob, 'recovery_amm': rt_amm})

    rec_df = pd.DataFrame(rec_rows)
    grp = grp.merge(rec_df[['amm_pct', 'recovery_all']], on='amm_pct', how='left')

    # Best combined recovery
    fin_rec = rec_df[np.isfinite(rec_df['recovery_all'])]
    if len(fin_rec):
        best_rec = fin_rec.loc[fin_rec['recovery_all'].idxmin()]
        print(f'\n    Fastest combined recovery: amm_pct={int(best_rec.amm_pct)}%  '
              f'{best_rec.recovery_all:.0f} periods')

    return grp


# -- 4. Cross-venue comparison -----------------------------------------

def report_cross_venue(clob, amm):
    _section('4. CROSS-VENUE: CLOB vs AMM')

    clob_mm1 = clob[clob['n_mm'] == 1].copy()
    clob_mm0 = clob[clob['n_mm'] == 0].copy()

    def fv_stats(df, label):
        fin = df[np.isfinite(df['cost_bps']) & np.isfinite(df['fv_cost'])].copy()
        fin = fin[fin['fv_cost'].between(-5000, 5000)]
        n_inf = (~np.isfinite(df['cost_bps'])).sum()
        fill = 100 * (1 - n_inf / len(df)) if len(df) else 0
        print(f'  {label}:')
        print(f'    Trades: {len(df):,}   Filled: {len(fin):,}  Fill rate: {fill:.1f}%')
        if len(fin):
            print(f'    fv_cost: med={fin["fv_cost"].median():.1f}  '
                  f'mean={fin["fv_cost"].mean():.1f}  '
                  f'p25={fin["fv_cost"].quantile(0.25):.1f}  '
                  f'p75={fin["fv_cost"].quantile(0.75):.1f} bps')
        print()

    fv_stats(clob_mm1, 'CLOB with MM (best CLOB)')
    fv_stats(clob_mm0, 'CLOB without MM')
    fv_stats(amm, 'AMM (all configs)')


# -- 5. Shock resilience -----------------------------------------------

def report_shock(clob, amm):
    _section('5. SHOCK RESILIENCE (pre vs post shock)')

    def pre_post(df, label):
        pre  = df[df['t'] < SHOCK_ITER].copy()
        post = df[df['t'] >= SHOCK_ITER].copy()

        pre_f  = pre[np.isfinite(pre['cost_bps']) & np.isfinite(pre['fv_cost'])]
        post_f = post[np.isfinite(post['cost_bps']) & np.isfinite(post['fv_cost'])]
        pre_f  = pre_f[pre_f['fv_cost'].between(-5000, 5000)]
        post_f = post_f[post_f['fv_cost'].between(-5000, 5000)]

        pre_fill  = 100 * len(pre_f) / len(pre) if len(pre) else 0
        post_fill = 100 * len(post_f) / len(post) if len(post) else 0

        print(f'\n  {label}:')
        if len(pre_f):
            print(f'    PRE-shock  (t<{SHOCK_ITER}):  fill={pre_fill:.1f}%  '
                  f'fv_cost med={pre_f["fv_cost"].median():.1f}  '
                  f'mean={pre_f["fv_cost"].mean():.1f} bps')
        else:
            print(f'    PRE-shock: no filled trades')
        if len(post_f):
            print(f'    POST-shock (t>={SHOCK_ITER}):  fill={post_fill:.1f}%  '
                  f'fv_cost med={post_f["fv_cost"].median():.1f}  '
                  f'mean={post_f["fv_cost"].mean():.1f} bps')
        else:
            print(f'    POST-shock: no filled trades')

    clob_mm1 = clob[clob['n_mm'] == 1]
    clob_mm0 = clob[clob['n_mm'] == 0]
    pre_post(clob_mm1, 'CLOB with MM')
    pre_post(clob_mm0, 'CLOB without MM')
    pre_post(amm, 'AMM (all configs)')


# -- 6. Best configs ---------------------------------------------------

def report_best_configs(clob_grp, amm_grp, amm_c_grp=None, mv_grp=None):
    _section('6. BEST CONFIGURATIONS')

    # Best CLOB — closest to fair value (|fv_cost| → 0) with MM=1
    if clob_grp is not None and len(clob_grp):
        valid = clob_grp.dropna(subset=['fv_cost_med'])
        if len(valid):
            # Prefer configs WITH market maker (MM=1) — proper CLOB
            mm1 = valid[valid['n_mm'] == 1]
            pool = mm1 if len(mm1) else valid
            pool = pool.copy()
            pool['abs_fv'] = pool['fv_cost_med'].abs()
            best = pool.loc[pool['abs_fv'].idxmin()]
            print(f'\n  >>> BEST CLOB configuration (closest to fair value, MM=1):')
            print(f'      MM={int(best.n_mm)}  Random={int(best.n_rnd)}  '
                  f'Fund={int(best.n_fund)}  Chartist={int(best.n_chart)}  '
                  f'Univ={int(best.n_univ)}')
            print(f'      Fill rate    : {best.fill_rate:.1f}%')
            print(f'      cost_bps med : {best.cost_med:.1f} bps')
            print(f'      cost_bps mean: {best.cost_mean:.1f} bps')
            print(f'      fv_cost med  : {best.fv_cost_med:.1f} bps')
            print(f'      fv_cost mean : {best.fv_cost_mean:.1f} bps')
            print(f'      |fv_cost|    : {best.abs_fv:.1f} bps  (lower = better)')

    # Best AMM
    if amm_grp is not None and len(amm_grp):
        best_a = amm_grp.loc[amm_grp['cost_med'].idxmin()]
        print(f'\n  >>> BEST AMM configuration (lowest median cost_bps):')
        print(f'      CPMM pools={int(best_a.n_cpmm)}  HFMM pools={int(best_a.n_hfmm)}')
        print(f'      Trades       : {int(best_a.total):,}')
        print(f'      Fill rate    : 100.0%  (AMM always fills)')
        print(f'      cost_bps med : {best_a.cost_med:.1f} bps')
        print(f'      cost_bps mean: {best_a.cost_mean:.1f} bps')
        print(f'      fv_cost  med : {best_a.fv_med:.1f} bps')
        print(f'      fv_cost  mean: {best_a.fv_mean:.1f} bps')
        print(f'      |fv_cost|    : {abs(best_a.fv_med):.1f} bps  (lower = better)')

    # Best AMM_C (heterogeneous HFMM)
    if amm_c_grp is not None and len(amm_c_grp):
        best_c = amm_c_grp.loc[amm_c_grp['cost_med'].idxmin()]
        print(f'\n  >>> BEST heterogeneous HFMM (Block C, lowest median cost_bps):')
        print(f'      Tight={int(best_c.n_tight)}  Balanced={int(best_c.n_balanced)}  '
              f'Deep={int(best_c.n_deep)}')
        print(f'      Trades       : {int(best_c.total):,}')
        print(f'      cost_bps med : {best_c.cost_med:.1f} bps')
        print(f'      cost_bps mean: {best_c.cost_mean:.1f} bps')
        print(f'      fv_cost  med : {best_c.fv_med:.1f} bps')
        print(f'      |fv_cost|    : {abs(best_c.fv_med):.1f} bps')

    # Best multi-venue (Block D)
    if mv_grp is not None and len(mv_grp):
        best_mv = mv_grp.loc[mv_grp['fv_med'].abs().idxmin()]
        print(f'\n  >>> BEST multi-venue CLOB+AMM (Block D, closest fv_cost to 0):')
        print(f'      amm_share_pct = {int(best_mv.amm_pct)}%')
        print(f'      Trades       : {int(best_mv.total):,}')
        print(f'      cost_bps med : {best_mv.cost_med:.1f} bps')
        print(f'      cost_bps mean: {best_mv.cost_mean:.1f} bps')
        print(f'      fv_cost  med : {best_mv.fv_med:.1f} bps')
        print(f'      |fv_cost|    : {abs(best_mv.fv_med):.1f} bps')


# -- 7. Conclusions ----------------------------------------------------

def report_conclusions():
    _section('7. SUMMARY & CONCLUSIONS')
    print("""
  Blocks A-C: closed ecosystems (no external FX takers).
    A: 5 CLOB agents trade among themselves.
    B: AMM pools with only the arbitrageur trading.
    C: heterogeneous HFMM pools, routing='best'.
  Block D: open system via Simulator.default_fx().
    30 CLOB noise + 1 MM + 33 FX takers (Noise/Fund/Retail/Institutional)
    + CPMM + HFMM + arbitrageur. Sweep amm_share_pct 0-100%.

  COMPONENT INSIGHTS (Blocks A-C):
  1. MARKET MAKER is critical for CLOB stability.
     MM=1: median fv_cost ~ -112 bps (tight spread execution).
     MM=0: higher variance, wider spread.

  2. HFMM CHEAPER THAN CPMM.
     Pure HFMM ~ 63 bps, pure CPMM ~ 85 bps.
     Tight HFMM (fee=5bp, A=50) further reduces to 52 bps.

  3. POOL HETEROGENEITY does not help without LP migration.
     Smart routing worsens multi-pool configs (untouched pools drift).

  4. AMM IS STRUCTURALLY PREDICTABLE.
     IQR(fv_cost): AMM ~ 41 bps vs CLOB ~ 2200+ bps.
     AMM is 50-60x more predictable in execution cost variance.

  5. AMM IS SHOCK-RESILIENT.
     Post-shock delta: AMM ~ 3 bps, CLOB ~ 200+ bps.
     Arbitrageur realigns pool prices immediately.

  SYSTEM-LEVEL INSIGHTS (Block D — full architecture):
  6. OPTIMAL AMM SHARE ~ 30%.
     |fv_cost| minimized at amm_share=30%.
     CLOB provides deep, cheap execution; AMM stabilizes.

  7. AMM FV_COST GROWS WITH FLOW.
     More AMM traffic → more slippage (fee + price impact).
     AMM fv_cost: ~10 bps at amm=30% → ~15 bps at amm=100%.

  8. CLOB FV_COST IS VOLATILE BUT DEEP.
     30 noise traders + MM absorb institutional flow (q=15-50).
     Only 1 inf trade out of 270k (0.0% rejection rate).

  9. SHOCK RESILIENCE at system level.
     amm_share=50-60%: pre/post delta < 1 bps (most stable).
     CLOB-heavy configs show higher shock sensitivity.

  RECOVERY TIME INSIGHTS:
  10. RECOVERY TIME measures periods after shock until rolling(20)
      median fv_cost returns to within ±20 bps of pre-shock baseline.

  11. AMM RECOVERS FASTER THAN CLOB.
      Arbitrageur realigns AMM pool prices in few periods.
      CLOB needs book refill (new limit orders from MM + noise).

  12. AMM SHARE AFFECTS SYSTEM RECOVERY.
      Higher amm_share_pct → faster combined recovery
      (AMM component stabilizes system execution quality).
""")


# ######################################################################
#  Analysis runner
# ######################################################################

def run_analysis():
    print('\nLoading data for analysis...')
    clob  = load_block('CLOB')
    amm   = load_block('AMM')
    amm_c = load_block('AMM_C')
    mv    = load_block('MV')

    if len(clob):
        clob = add_fv_cost(clob)
    if len(amm):
        amm = add_fv_cost(amm)
    if len(amm_c):
        amm_c = add_fv_cost(amm_c)
    if len(mv):
        mv = add_fv_cost(mv)

    report_overview(clob, amm)
    clob_grp = report_block_a(clob)
    amm_grp  = report_block_b(amm)
    amm_c_grp = report_block_c(amm_c)
    mv_grp    = report_block_d(mv)
    report_cross_venue(clob, amm)
    report_shock(clob, amm)
    report_best_configs(clob_grp, amm_grp, amm_c_grp, mv_grp)
    report_conclusions()


# ######################################################################
#  Entry point
# ######################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stress Sweep -- simulation + analysis')
    parser.add_argument('--analyze', action='store_true',
                        help='Skip simulation, run analysis only on existing CSVs')
    parser.add_argument('--block-c', action='store_true',
                        help='Run only Block C (heterogeneous HFMM), then analyze all')
    parser.add_argument('--block-d', action='store_true',
                        help='Run only Block D (multi-venue CLOB+AMM), then analyze all')
    args = parser.parse_args()

    if args.block_c:
        run_block_c()
    elif args.block_d:
        run_block_d()
    elif not args.analyze:
        run_block_a()
        run_block_b()
        run_block_c()
        run_block_d()
        print('\n' + '=' * 70)
        print('  Stress sweep complete.  Results -> output/stress/')
        print('=' * 70)

    run_analysis()
    print('\nDone.')
