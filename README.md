# Agent-Based FX Market Model

The model is an agent-based simulation of the foreign exchange (FX) market in which multiple execution venues - a classical stock market stack (CLOB) and two types of automated market makers (AMMs) - operate simultaneously, competing for the order flow of heterogeneous participants.



## Model structure

```text
AgentBasedModel/
  agents/         # different types of CLOB/AMM traders
  environment/    # stress periods, exogeneous volatility, funding luquidity
  events/         # market shocks
  metrics/        # metrics aggregation
  simulator/      # arbitary scenarios
  states/         # market conditions
  utils/          # math
  venues/         # CLOB/AMM architecrure
  visualization/  # graphs and dashboards
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
python main.py --amm-share 30
python generate_resilience_study_fixed.py --seeds 40
python generate_resilience_study_aware.py --seeds 40

# Multi-seed resilience scatter study
python generate_resilience_study.py --seeds 40

# Censor-aware resilience study with 80% retracement recovery
python generate_resilience_study.py

# Override sample size / observation window if needed
python generate_resilience_study.py --seeds 300 --min-post-shock-window 600

# Paired fixed_share vs liquidity_aware comparison on identical seeds
python generate_routing_comparison.py --seeds 30
```

The resilience study now writes these artifacts into `output/resilience/`:

- `resilience_scatter_panels.png` — normalized impact vs observed recovery time, overlaid for `With AMM` and `Without AMM`, with censored points marked separately
- `recovery_survival_panels.png` — Kaplan-Meier recovery curves by scenario
- `resilience_scatter_points.csv` — seed-level point data with `series_key`, `amm_enabled`, and `recovered` / `is_censored` flags
- `resilience_panel_summary.csv` — panel-level summary statistics split by `With AMM` vs `Without AMM`
- `resilience_priority_metric_points.csv` — seed-level resilience points for spread, depth, execution cost, and composite metrics
- `resilience_priority_metric_summary.csv` — panel summaries for those priority resilience metrics
- `resilience_statistical_summary.csv` — bootstrap mean/CI summaries for core resilience endpoints, split by `With AMM` vs `Without AMM`
- `resilience_pairwise_tests.csv` — pairwise scenario-level statistical comparisons across panels, within each AMM series separately
- `resilience_comparison_summary.csv` — paired within-seed `With AMM` vs `Without AMM` recovery comparison for each scenario and resilience metric

Dedicated resilience entrypoints:

- `generate_resilience_study_fixed.py` writes the same resilience artifacts into `output/resilience_fixed/`
- `generate_resilience_study_aware.py` writes the same resilience artifacts into `output/resilience_aware/`

Routing note:

- `main.py` now defaults to `liquidity_aware` top-level venue routing.
- Explicit `--amm-share ...` opts into the legacy `fixed_share` AMM/CLOB split.
- Explicit `--venue-choice-rule fixed_share` or `--venue-choice-rule liquidity_aware` still works for direct control.
- Dashboards and plots remain separated by effective routing mode.
- `main.py` writes into `output/main_aware/` when the effective venue rule is `liquidity_aware`.
- `main.py` writes into `output/main_fixed/` when the effective venue rule is `fixed_share`.

The paired routing comparison study writes three CSV artifacts into `output/routing_comparison/`:

- `routing_comparison_seed_metrics.csv` — seed-level metrics for each scenario and routing mode
- `routing_comparison_paired_deltas.csv` — same-seed deltas `liquidity_aware - fixed_share`
- `routing_comparison_summary.csv` — paired mean deltas, paired t-tests, and paired bootstrap confidence intervals
