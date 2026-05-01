# FX ABM Model

Multi-venue FX agent-based model with one CLOB, one CPMM, and one HFMM. The codebase is centered on `main.py` for single-run simulation, `tests/resilience_test.py` for resilience event studies, `tests/stat_tests.py` for scenario-level statistical comparisons, and `calibration/runner.py` for primary-model acceptance checks.

## Structure

```text
AgentBasedModel/
  agents/           trader logic
  environment/      volatility, funding, fair-price, session state
  metrics/          logging, resilience, statistics
  simulator/        main simulation loop
  utils/            math and order-book primitives
  venues/           CLOB and AMM venues
  visualization/    dashboards and plots
calibration/        primary model manifest, targets, fitter, search runner
tests/              unit, resilience, scenario-stat, and sweep scripts
output/             generated artifacts
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main Run

```bash
python main.py
python main.py --help
python main.py --preset dealer_liquidity_crisis
```

Default routing in `main.py` is `liquidity_aware`. The primary runtime defaults are stored in `calibration/primary_model.json`. Acceptance targets are stored in `calibration/primary_model_targets.json`.

## Research Scripts

```bash
python tests/unit_test.py
python tests/resilience_test.py --seeds 300 --base-seed 42 --n-iter 900 --out-dir output/resilience_aware_300
python tests/stat_tests.py --seeds 300 --base-seed 42 --n-iter 900 --out-dir output/scenario_stats_300
python calibration/runner.py --current-only
```

`tests/resilience_test.py` writes nine files per run:

- `resilience_scatter_panels.png`
- `recovery_survival_panels.png`
- `resilience_scatter_points.csv`
- `resilience_panel_summary.csv`
- `resilience_priority_metric_points.csv`
- `resilience_priority_metric_summary.csv`
- `resilience_statistical_summary.csv`
- `resilience_pairwise_tests.csv`
- `resilience_comparison_summary.csv`

`tests/stat_tests.py` writes nine files per run:

- `rq1_effect_dashboard.png`
- `rq1_tests.csv`
- `rq2_effect_dashboard.png`
- `rq2_tests.csv`
- `rq_effect_heatmap_summary.png`
- `rq_stat_points.csv`
- `rq_stat_summary.csv`
- `rq_statistical_summary.csv`
- `rq_with_without_amm_tests.csv`

## Outputs

Versioned output directories in this repository:

- `output/resilience_aware_300/`
- `output/scenario_stats_300/`

All other local output folders remain ignored.
