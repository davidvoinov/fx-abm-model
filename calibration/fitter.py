from __future__ import annotations

import math
from typing import Any, Optional

from AgentBasedModel.metrics.resilience import (
    baseline_level,
    half_life_steps_pct,
    pct_deviation_series,
)


def _finite_mean(series: list[float]) -> float:
    finite = [float(value) for value in series if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float('nan')


def _finite_median(series: list[float]) -> float:
    finite = sorted(float(value) for value in series if math.isfinite(value))
    if not finite:
        return float('nan')
    mid = len(finite) // 2
    if len(finite) % 2 == 1:
        return finite[mid]
    return 0.5 * (finite[mid - 1] + finite[mid])


def _window_mean(series: list[float], start: int, end: int) -> float:
    if not series:
        return float('nan')
    lo = max(0, int(start))
    hi = max(lo, min(len(series), int(end)))
    return _finite_mean(series[lo:hi])


def _lag1_autocorr(series: list[float]) -> float:
    pairs = [
        (float(prev), float(curr))
        for prev, curr in zip(series[:-1], series[1:])
        if math.isfinite(prev) and math.isfinite(curr)
    ]
    if len(pairs) < 3:
        return float('nan')

    xs, ys = zip(*pairs)
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x in xs) / len(xs)
    vy = sum((y - my) ** 2 for y in ys) / len(ys)
    if vx <= 0 or vy <= 0:
        return float('nan')
    return cov / math.sqrt(vx * vy)


def _signed_nonzero_series(series: list[float], zero_tol: float = 1e-12) -> list[float]:
    out: list[float] = []
    for value in series:
        if not math.isfinite(value):
            continue
        if abs(float(value)) <= zero_tol:
            continue
        out.append(1.0 if float(value) > 0 else -1.0)
    return out


def _run_lengths(sign_series: list[float]) -> list[float]:
    if not sign_series:
        return []

    runs: list[float] = []
    current = sign_series[0]
    run_length = 1
    for sign in sign_series[1:]:
        if sign == current:
            run_length += 1
            continue
        runs.append(float(run_length))
        current = sign
        run_length = 1
    runs.append(float(run_length))
    return runs


def _signed_flow_diagnostics(series: list[float]) -> dict[str, float]:
    sign_series = _signed_nonzero_series(series)
    run_lengths = _run_lengths(sign_series)
    return {
        'lag1_sign_autocorr': _lag1_autocorr(sign_series),
        'mean_run_length': _finite_mean(run_lengths),
    }


def _nearest_trade_size(grid: list[float], target_q: float) -> float:
    if not grid:
        return float(target_q)
    return min(grid, key=lambda candidate: abs(float(candidate) - float(target_q)))


class CalibrationFitter:
    """Scaffold for evaluating one simulation run against literature targets."""

    def __init__(self, target_payload: Optional[dict] = None):
        self.target_payload = target_payload or {}
        self.targets = list(self.target_payload.get('targets', []))

    @staticmethod
    def _applies_to_scenario(target: dict[str, Any], scenario_name: Optional[str]) -> bool:
        target_scenario = target.get('evaluation_scenario')
        if target_scenario is None:
            return True
        return target_scenario == scenario_name

    def target_metric_defaults(self, scenario_name: Optional[str] = None) -> dict[str, float]:
        defaults: dict[str, float] = {}
        for target in self.targets:
            if not self._applies_to_scenario(target, scenario_name):
                continue
            observable = target.get('observable')
            if not observable:
                continue
            if 'target_value' in target:
                defaults[observable] = float(target['target_value'])
                continue
            target_range = target.get('target_range') or {}
            if 'low' in target_range and 'high' in target_range:
                defaults[observable] = 0.5 * (float(target_range['low']) + float(target_range['high']))
                continue
            if target.get('qualitative_bounds'):
                defaults[observable] = 1.0
        return defaults

    @staticmethod
    def _shock_iter(sim) -> Optional[int]:
        shock_iter = getattr(sim, 'shock_iter', None)
        if shock_iter is not None:
            return int(shock_iter)

        env = getattr(sim, 'env', None)
        if env is None:
            return None
        stress_start = getattr(env, 'stress_start', None)
        return int(stress_start) if stress_start is not None else None

    @staticmethod
    def _funding_liquidity_propagation(logger) -> float:
        return abs(float(logger.series_correlation(logger.c_series, logger.clob_qspr)))

    @staticmethod
    def _amm_volume_share(logger) -> float:
        amm_shares = []
        for venue in logger.flow_volume:
            if venue == 'clob':
                continue
            amm_shares.append(_finite_mean(logger.flow_share(venue)))
        return sum(value for value in amm_shares if math.isfinite(value)) if amm_shares else 0.0

    @staticmethod
    def _parameter_sanity(sim) -> float:
        pools = getattr(sim, 'amm_pools', {}) or {}
        if not pools:
            return 1.0

        checks: list[bool] = []
        max_basis = _finite_mean([
            abs(value)
            for value in sim.logger.max_venue_basis_series()
            if math.isfinite(value)
        ])
        if math.isfinite(max_basis):
            checks.append(max_basis <= 25.0)

        for pool in pools.values():
            checks.append(100.0 <= float(pool.x) <= 50_000.0)
            checks.append(0.0005 <= float(pool.fee) <= 0.01)
            if hasattr(pool, 'A'):
                checks.append(2.0 <= float(pool.A) <= 100.0)
                checks.append(0.0001 <= float(pool.fee) <= 0.005)

        return 1.0 if checks and all(checks) else 0.0

    def realized_metrics(self, sim) -> dict[str, float]:
        logger = sim.logger
        shock_iter = self._shock_iter(sim)
        clob_flow_series = list(
            getattr(logger, 'clob_flow_imbalance_series', logger.flow_imbalance_series)
        )
        flow_diagnostics = _signed_flow_diagnostics(clob_flow_series)

        impact_trade_size = 10.0
        impact_series = logger.clob_impact_curves.get(
            _nearest_trade_size(list(logger.clob_impact_curves.keys()), impact_trade_size),
            [],
        )

        recovery_half_life = float('nan')
        dealer_withdrawal_share = float('nan')
        if shock_iter is not None and shock_iter < len(logger.iterations):
            qspr_baseline = baseline_level(list(logger.clob_qspr), shock_iter, lookback=50)
            qspr_dev = pct_deviation_series(list(logger.clob_qspr), qspr_baseline)
            recovery_half_life = half_life_steps_pct(qspr_dev, shock_iter, stable_window=3, horizon=120)
            dealer_withdrawal_share = _window_mean(
                logger.mm_state_shares.get('withdrawn', []),
                shock_iter,
                shock_iter + 60,
            )

        return {
            'quoted_spread_bps': _finite_median(list(logger.clob_qspr)),
            'near_touch_depth': _finite_mean(logger.clob_touch_depth_series()),
            'order_flow_autocorrelation': flow_diagnostics['lag1_sign_autocorr'],
            'order_flow_mean_run_length': flow_diagnostics['mean_run_length'],
            'impact_curve': _finite_mean(impact_series),
            'recovery_half_life_ticks': recovery_half_life,
            'dealer_withdrawal_share': dealer_withdrawal_share,
            'funding_liquidity_stress_propagation': self._funding_liquidity_propagation(logger),
            'cross_venue_basis_bps': _finite_mean([
                abs(value) for value in logger.max_venue_basis_series() if math.isfinite(value)
            ]),
            'amm_volume_share': self._amm_volume_share(logger),
            'fx_amm_parameter_sanity': self._parameter_sanity(sim),
        }

    @staticmethod
    def _band_threshold(target: dict[str, Any], reference_value: float) -> float:
        band = target.get('accepted_error_band') or {}
        if 'absolute' in band:
            return max(float(band['absolute']), 1e-9)
        if 'relative' in band:
            scale = abs(reference_value) if reference_value != 0 else 1.0
            return max(scale * float(band['relative']), 1e-9)
        return 0.0

    def evaluate_metrics(self, metrics: dict[str, float], run_label: Optional[str] = None,
                         scenario_name: Optional[str] = None) -> dict:
        report_targets = []
        evaluated_targets = 0
        passed_targets = 0
        gating_failures = 0
        objective_terms = []

        for target in self.targets:
            if not self._applies_to_scenario(target, scenario_name):
                continue
            observable = target.get('observable')
            if not observable:
                continue

            realized = metrics.get(observable, float('nan'))
            gating = bool(target.get('gating', True))
            target_range = target.get('target_range') or {}
            target_value = target.get('target_value')
            if 'low' in target_range and 'high' in target_range:
                low = float(target_range['low'])
                high = float(target_range['high'])
                reference_value = min(max(float(realized), low), high) if math.isfinite(realized) else 0.5 * (low + high)
                inside_range = math.isfinite(realized) and low <= float(realized) <= high
                target_display = f'[{low}, {high}]'
            elif target_value is not None:
                reference_value = float(target_value)
                inside_range = False
                target_display = reference_value
            elif target.get('qualitative_bounds'):
                reference_value = 1.0
                inside_range = bool(math.isfinite(realized) and float(realized) >= 1.0 - 1e-9)
                target_display = 'sanity_bounds'
            else:
                reference_value = float('nan')
                inside_range = False
                target_display = 'n/a'

            entry = {
                'observable': observable,
                'units': target.get('units'),
                'source': target.get('source'),
                'confidence': target.get('confidence'),
                'match_type': target.get('match_type'),
                'evaluation_scenario': target.get('evaluation_scenario'),
                'gating': gating,
                'realized_value': float(realized) if math.isfinite(realized) else float('nan'),
                'target': target_display,
                'accepted_error_band': target.get('accepted_error_band'),
                'source_excerpt': target.get('source_excerpt'),
            }

            if not math.isfinite(realized):
                entry['status'] = 'not_evaluable'
                report_targets.append(entry)
                continue

            threshold = self._band_threshold(target, reference_value)
            distance = 0.0 if inside_range else abs(float(realized) - float(reference_value))
            passed = inside_range or threshold == 0.0 and distance == 0.0 or distance <= threshold
            normalized_error = distance / threshold if threshold > 0 else 0.0

            entry['status'] = 'pass' if passed else 'fail'
            entry['distance_to_target'] = distance
            entry['normalized_error'] = normalized_error
            evaluated_targets += 1
            if passed:
                passed_targets += 1
            elif gating:
                gating_failures += 1
            weight = max(0.0, float(target.get('objective_weight', 1.0)))
            if weight > 0.0:
                objective_terms.append(weight * normalized_error ** 2)
            report_targets.append(entry)

        objective_score = sum(objective_terms) / len(objective_terms) if objective_terms else float('nan')
        summary = {
            'run_label': run_label,
            'scenario_name': scenario_name,
            'evaluated_targets': evaluated_targets,
            'passed_targets': passed_targets,
            'failed_targets': max(0, evaluated_targets - passed_targets),
            'gating_failures': gating_failures,
            'objective_score': objective_score,
            'status': 'pass' if evaluated_targets > 0 and gating_failures == 0 else 'fail',
        }
        return {'summary': summary, 'targets': report_targets}

    def evaluate_simulation(self, sim, run_label: Optional[str] = None,
                            scenario_name: Optional[str] = None) -> dict:
        metrics = self.realized_metrics(sim)
        report = self.evaluate_metrics(metrics, run_label=run_label, scenario_name=scenario_name)
        report['realized_metrics'] = metrics
        return report

    def evaluate_scenario_suite(self, suite_metrics: dict[str, dict[str, float]],
                                run_label: Optional[str] = None) -> dict:
        report_targets = []
        evaluated_targets = 0
        passed_targets = 0
        gating_failures = 0
        objective_terms = []

        for target in self.targets:
            scenario_name = target.get('evaluation_scenario')
            metrics = suite_metrics.get(scenario_name or 'baseline_primary')
            if metrics is None:
                continue

            observable = target.get('observable')
            if not observable:
                continue

            single = self.evaluate_metrics(
                {observable: metrics.get(observable, float('nan'))},
                run_label=run_label,
                scenario_name=scenario_name,
            )
            for item in single.get('targets', []):
                if item.get('observable') != observable:
                    continue
                report_targets.append(item)
                if item.get('status') == 'not_evaluable':
                    break
                evaluated_targets += 1
                if item.get('status') == 'pass':
                    passed_targets += 1
                elif item.get('gating', True):
                    gating_failures += 1
                weight = max(0.0, float(target.get('objective_weight', 1.0)))
                if weight > 0.0:
                    objective_terms.append(weight * item.get('normalized_error', 0.0) ** 2)
                break

        objective_score = sum(objective_terms) / len(objective_terms) if objective_terms else float('nan')
        return {
            'summary': {
                'run_label': run_label,
                'scenario_name': 'suite',
                'evaluated_targets': evaluated_targets,
                'passed_targets': passed_targets,
                'failed_targets': max(0, evaluated_targets - passed_targets),
                'gating_failures': gating_failures,
                'objective_score': objective_score,
                'status': 'pass' if evaluated_targets > 0 and gating_failures == 0 else 'fail',
            },
            'targets': report_targets,
            'scenario_metrics': suite_metrics,
        }


def build_acceptance_report(sim, target_payload: Optional[dict] = None,
                            run_label: Optional[str] = None,
                            scenario_name: Optional[str] = None) -> dict:
    fitter = CalibrationFitter(target_payload)
    return fitter.evaluate_simulation(sim, run_label=run_label, scenario_name=scenario_name)