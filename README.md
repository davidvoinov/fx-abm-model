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
python main_fx.py
```
