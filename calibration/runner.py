from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as main_module
from calibration.fitter import CalibrationFitter


DEFAULT_SEARCH_GRID: dict[str, list[float]] = {
    'mm_alpha0_base': [1.8, 2.1, 2.4, 2.7],
    'mm_alpha2': [420.0, 560.0, 700.0],
    'mm_d0_base': [75.0, 90.0, 105.0],
    'hedger_flow_persistence': [0.05, 0.10, 0.15],
    'retail_flow_persistence': [0.18, 0.28, 0.38],
    'institutional_flow_persistence': [0.10, 0.18, 0.26],
    'amm_share_pct': [22.0, 30.0, 38.0],
    'cpmm_bias_bps': [0.0, 2.5, 5.0],
    'cost_noise_std': [0.5, 1.0, 1.5],
    'hfmm_reserves': [1000.0, 1500.0, 2000.0],
    'hfmm_fee': [0.0005, 0.0010, 0.0015],
    'cpmm_fee': [0.0020, 0.0030, 0.0040],
    'hfmm_A': [10.0, 18.0, 30.0],
    'clob_amm_spread_impact_bps': [1.0, 2.0, 3.0],
    'arb_trade_fraction_cap': [0.2, 0.35, 0.5],
}


def _objective_value(report: dict[str, Any]) -> float:
    summary = report.get('summary', {})
    objective = summary.get('objective_score')
    if objective is None or not math.isfinite(objective):
        return float('inf')
    penalty = 1000.0 * float(summary.get('gating_failures', 0))
    return float(objective) + penalty


def _make_scenario_args(scenario: dict[str, Any], overrides: dict[str, Any], seed: int) -> argparse.Namespace:
    parser = main_module.build_parser()
    args = parser.parse_args([])
    args.seed = seed
    args.no_plots = True
    args.no_comparison = True
    args.silent = True
    args.spillover_artifacts = False
    args.run_label = scenario['name']
    args.preset = scenario.get('preset')

    if args.preset:
        main_module._apply_preset_defaults(parser, args)

    for key, value in overrides.items():
        # Preserve scenario-preset values when the candidate simply repeats
        # the parser baseline default. Only explicit deviations from the
        # baseline should override the scenario-specific preset.
        if hasattr(args, key):
            default_value = parser.get_default(key)
            current_value = getattr(args, key)
            if (main_module._same_value(value, default_value)
                    and not main_module._same_value(current_value, default_value)):
                continue
        setattr(args, key, value)

    if scenario.get('n_iter') is not None:
        args.n_iter = int(scenario['n_iter'])
    if scenario.get('shock_iter') is not None:
        args.shock_iter = scenario['shock_iter']
    if scenario.get('stress_start') is not None:
        args.stress_start = scenario['stress_start']
    if scenario.get('stress_end') is not None:
        args.stress_end = scenario['stress_end']

    main_module._auto_stress_around_shock(args)
    args.venue_choice_rule = main_module._resolve_main_routing(args, [])
    return args


def evaluate_candidate(overrides: dict[str, Any], target_payload: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    fitter = CalibrationFitter(target_payload)
    scenario_metrics: dict[str, dict[str, float]] = {}
    scenario_reports: dict[str, dict[str, Any]] = {}

    for scenario in target_payload.get('calibration_scenarios', []):
        args = _make_scenario_args(scenario, overrides, seed=seed)
        main_module._seed_all(seed)
        sim = main_module.build_sim(args)
        sim.simulate(args.n_iter, silent=True)
        metrics = fitter.realized_metrics(sim)
        scenario_metrics[scenario['name']] = metrics
        scenario_reports[scenario['name']] = fitter.evaluate_metrics(
            metrics,
            run_label=args.run_label,
            scenario_name=scenario['name'],
        )

    suite_report = fitter.evaluate_scenario_suite(scenario_metrics, run_label='calibration_suite')
    suite_report['scenario_reports'] = scenario_reports
    suite_report['candidate_overrides'] = deepcopy(overrides)
    suite_report['score'] = _objective_value(suite_report)
    return suite_report


def coordinate_search(target_payload: dict[str, Any], base_overrides: dict[str, Any],
                      search_grid: dict[str, list[float]], passes: int = 1,
                      seed: int = 42) -> dict[str, Any]:
    current = deepcopy(base_overrides)
    best_report = evaluate_candidate(current, target_payload, seed=seed)

    for _ in range(max(1, passes)):
        improved = False
        for parameter, candidates in search_grid.items():
            local_best_value = current.get(parameter)
            local_best_report = best_report

            ordered_candidates = []
            if local_best_value is not None:
                ordered_candidates.append(local_best_value)
            ordered_candidates.extend(value for value in candidates if value != local_best_value)

            for value in ordered_candidates:
                candidate = deepcopy(current)
                candidate[parameter] = value
                report = evaluate_candidate(candidate, target_payload, seed=seed)
                if report['score'] + 1e-12 < local_best_report['score']:
                    local_best_value = value
                    local_best_report = report

            if local_best_report is not best_report:
                current[parameter] = local_best_value
                best_report = local_best_report
                improved = True

        if not improved:
            break

    best_report['best_overrides'] = deepcopy(current)
    return best_report


def write_back_primary_defaults(best_overrides: dict[str, Any], model_path: Path) -> None:
    payload = json.loads(model_path.read_text())
    cli_defaults = payload.setdefault('cli_defaults', {})
    cli_defaults.update(best_overrides)
    model_path.write_text(json.dumps(payload, indent=2) + '\n')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run scenario-aware calibration search for the primary FX model.')
    parser.add_argument('--passes', type=int, default=1,
                        help='Coordinate-descent passes over the search grid (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Deterministic seed for calibration runs (default: 42)')
    parser.add_argument('--current-only', action='store_true',
                        help='Evaluate the current primary defaults only, without running the coordinate search')
    parser.add_argument('--write-back', action='store_true',
                        help='Write the best parameter set back into calibration/primary_model.json')
    parser.add_argument('--output', type=Path, default=Path('output/main_aware/calibration_search_report.json'),
                        help='Path for the calibration report artifact')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target_payload = main_module.load_primary_model_targets()
    base_defaults = main_module.load_primary_model_defaults()
    search_grid = {
        key: values
        for key, values in DEFAULT_SEARCH_GRID.items()
        if key in base_defaults
    }
    base_overrides = {key: base_defaults[key] for key in search_grid}
    if args.current_only:
        report = evaluate_candidate(base_overrides, target_payload, seed=args.seed)
        report['best_overrides'] = deepcopy(base_overrides)
    else:
        report = coordinate_search(
            target_payload=target_payload,
            base_overrides=base_overrides,
            search_grid=search_grid,
            passes=args.passes,
            seed=args.seed,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')

    if args.write_back:
        write_back_primary_defaults(report['best_overrides'], Path('calibration/primary_model.json'))

    print(json.dumps({
        'status': report.get('summary', {}).get('status'),
        'objective_score': report.get('summary', {}).get('objective_score'),
        'gating_failures': report.get('summary', {}).get('gating_failures'),
        'best_overrides': report.get('best_overrides', {}),
        'output': str(args.output),
    }, indent=2))


if __name__ == '__main__':
    main()
