# Agent-Based FX Market Model

The project simulates FX market with multiple execution venues (CLOB + AM), stress scenario and execution metrics.

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
python main_fx.py
```
