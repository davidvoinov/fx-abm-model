# TODO: paper roadmap for FX ABM article

## 0. Decision lock before writing results

- [ ] Freeze the paper claim: this is primarily a model of liquidity allocation, venue interaction, dislocation, and recovery, not a fully endogenous FX price-discovery model.
- [ ] Decide whether to implement explicit endogenous MM withdrawal logic or weaken the text so that withdrawal remains scenario-driven via the shock/liquidity block.
- [ ] Freeze the paper baseline configuration and robustness ranges before drafting the calibration subsection.

## 1. P0 gaps for traditional MM logic

### 1.1 Core modeling decision

- [ ] Choose one of two paths and keep the paper text consistent with it:
  - Path A: keep the current environment-driven withdrawal architecture and describe MM stress behavior as scenario-triggered quote withdrawal and slow re-quoting.
  - Path B: add explicit endogenous MM withdrawal and re-entry rules inside `MarketMaker`.

### 1.2 If Path B is chosen, implement explicit MM logic

Files to touch:
- `AgentBasedModel/agents/agents.py`
- `AgentBasedModel/metrics/logger.py`
- `AgentBasedModel/simulator/simulator.py`
- `tests/unit_test.py`

Missing logic to add:
- [ ] Track MM mark-to-market PnL or rolling adverse-selection loss, not only inventory and OFI.
- [ ] Add a withdrawal score driven by:
  - realized or marked-to-market losses
  - volatility `sigma`
  - funding cost `c`
  - inventory stress
  - toxic order-flow pressure
  - AMM outside-option pressure
- [ ] Separate soft response from hard response:
  - soft: wider spread, lower depth, asymmetric quoting
  - hard: partial or full quote withdrawal for several ticks
- [ ] Add re-entry logic after stress subsides, conditional on liquidity recovery and PnL stabilization.
- [ ] Add explicit MM state labels for the paper and diagnostics:
  - `active`
  - `defensive`
  - `withdrawn`
  - `reentering`
- [ ] Export MM state counts / shares over time so the paper can show dealer retreat dynamics directly.

### 1.3 If Path A is chosen, tighten paper wording instead of code

- [ ] Rewrite the methodology text so MM withdrawal is described as part of the realism shock architecture rather than an endogenous dealer optimization rule.
- [ ] Make it explicit that the current model captures strategic MM stress behavior only partially through spread/depth response plus liquidity-shock quote removal.
- [ ] Remove any sentence claiming a pure `if loss > threshold then withdraw` rule unless it is actually implemented.

## 2. P0 calibration workstream

### 2.1 Parameter inventory and classification

- [ ] Build a parameter table with columns:
  - parameter
  - interpretation
  - baseline value
  - robustness range
  - calibrated vs scenario-imposed vs convenience default
  - source or motivation
- [ ] Classify parameters into three buckets:
  - stylized-fact calibrated
  - literature-anchored but not estimated
  - purely scenario / design parameters

Core parameters to classify first:
- [ ] `sigma_low`, `sigma_high`
- [ ] `c_low`, `c_high`
- [ ] `cpmm_fee`, `hfmm_fee`
- [ ] `hfmm_A`
- [ ] `amm_share_pct`
- [ ] `beta_amm`
- [ ] `clob_liq`, `amm_liq`
- [ ] MM coefficients `alpha0..alpha3`, `d0..d3`, `d_min`, `inv_skew_bps`
- [ ] realism shock parameters such as `fundamental_shock_pct`, `order_flow_shock_qty`, `liquidity_shock_frac`, `funding_vol_shock_intensity`

### 2.2 Use existing calibration anchors already in repo

Existing assets to leverage:
- `main.py` preset `fx_calibrated`
- `main.py` flag `match_initial_depth`
- `tests/fennel.py`
- `output/fennell/analytical_replication.csv`
- `output/fennell/competitive_summary.csv`
- `output/fennell/competitive_f1_joint.csv`
- `output/fennell/competitive_f2_dynamic.csv`
- `output/fennell/competitive_f3_joint.csv`
- `output/fennell/hfmm_extension.csv`

TODO:
- [ ] Summarize which parts of the model are already anchored by the Fennell-style AMM vs CLOB cost benchmarks.
- [ ] Separate AMM transaction-cost calibration from dynamic multi-agent FX calibration in the paper.
- [ ] Treat `match_initial_depth` as a calibration aid, not as a default market structure assumption.

### 2.3 Define calibration targets explicitly

Targets that calibration should try to match:
- [ ] baseline quoted spread level and dispersion
- [ ] post-shock spread widening
- [ ] post-shock depth drawdown
- [ ] basis spike magnitude and decay
- [ ] recovery speed / half-life
- [ ] venue flow substitution into AMM during stress
- [ ] trade-size cost ordering by venue

### 2.4 Missing calibration machinery

- [ ] Add a dedicated calibration runner instead of relying only on presets.
- [ ] Add a multi-seed calibration summary script that writes paper-facing CSV tables.
- [ ] Add a simple loss/objective function over moments, for example weighted squared distance to target moments.
- [ ] Add uncertainty bands around calibrated outcomes across seeds.
- [ ] Document which parameters are tuned manually versus matched mechanically.

### 2.5 External stylized facts still missing

- [ ] Add a short empirical appendix or data note for FX stylized facts:
  - spread levels
  - volatility regimes
  - stress drawdowns in depth
  - shock recovery magnitudes
- [ ] Cite the exact literature / benchmark source for each targeted moment.
- [ ] Avoid saying “calibrated” broadly unless the moment-matching target is written down explicitly.

## 3. P0 statistical and experimental design

### 3.1 Hypotheses must be formalized

- [ ] Rewrite RQ1-RQ3 into testable hypotheses with outcome variables and expected signs.
- [ ] For each headline claim, define:
  - treatment / control
  - metric
  - effect sign
  - statistical summary
  - figure or table that will support it

### 3.2 Baseline multi-seed inference

- [ ] Add a baseline multi-seed batch runner for the default paper configuration.
- [ ] For each key KPI, export mean, std, median, percentile band, and bootstrap CI.

### 3.3 Paired comparisons already partly available but need packaging

Existing scripts:
- `generate_resilience_study.py`
- `generate_routing_comparison.py`

Still needed:
- [ ] Convert these outputs into final paper tables, not just raw CSV artifacts.
- [ ] Add a consistent naming convention for effect sizes across all comparison outputs.
- [ ] Add multiple-comparison discipline for broad metric panels.

### 3.4 Regressions with lags are not implemented yet

- [ ] Either remove “lag regressions” from the article skeleton or add a small panel/event-window regression layer on exported seed-level outputs.
- [ ] If added, keep it simple:
  - outcome on treatment dummy
  - scenario fixed effects
  - optional lagged baseline controls

## 4. P1 missing metrics for resilience and commonality

### 4.1 Half-life expansion

Current state:
- resilience half-life exists inside `AgentBasedModel/metrics/resilience.py`

Missing exports for the paper:
- [ ] half-life of CLOB quoted spread
- [ ] half-life of near-mid CLOB depth
- [ ] half-life of systemic liquidity
- [ ] half-life of AMM-vs-CLOB basis
- [ ] half-life of representative AMM execution cost

### 4.2 Commonality packaging

Current state:
- correlation helpers already exist in `AgentBasedModel/metrics/logger.py`

Missing paper-facing layer:
- [ ] Export before/after stress commonality for the exact metric pairs used in the paper.
- [ ] Add rolling-window commonality between CLOB spread and AMM slippage / AMM cost.
- [ ] Test and report whether commonality rises under stress.
- [ ] Decide whether commonality is measured on levels, absolute changes, or normalized deviations from baseline.

## 5. P1 RQ3-specific AMM comparison design

- [ ] Create a clean CPMM-vs-HFMM experiment with matched reserves and matched fee scenarios.
- [ ] Sweep `hfmm_A` systematically and report:
  - small-trade cost
  - medium-trade cost
  - large-trade cost
  - basis stability
  - spread/depth resilience contribution
  - substitution share under stress
- [ ] Keep venue-comparison claims size-specific; avoid a single blanket statement that one AMM is always better.

## 6. P1 paper-writing integration

- [ ] Add a dedicated `Calibration` subsection to methodology.
- [ ] Add a dedicated `Experimental Design and Statistical Evaluation` subsection.
- [ ] In results, keep three separate blocks:
  - liquidity quality
  - venue substitution and basis
  - resilience / recovery
- [ ] In limitations, state clearly that long-run fair value is externally anchored by `MarketEnvironment.fair_price`.
- [ ] Every headline claim should be tagged mentally as one of:
  - multi-seed statistical evidence
  - deterministic benchmark evidence
  - qualitative simulation evidence

## 7. P2 implementation and workflow helpers

- [ ] Add a paper-facing batch runner that writes final tables for baseline, resilience, routing, and AMM comparison sections.
- [ ] Add regression-safe baseline snapshots for paper-facing metrics beyond the current regression suite.
- [ ] Add a manifest of scenario parameters and git-stamped run metadata for reproducibility.
- [ ] Add a lightweight script to export parameter tables into Markdown or CSV for the paper appendix.

## 8. Deliverables checklist before drafting final paper results

### Must-have before strong draft
- [ ] Paper claim narrowed and internally consistent
- [ ] MM logic decision resolved
- [ ] Calibration table written
- [ ] Multi-seed inference tables generated
- [ ] Commonality outputs generated
- [ ] RQ3 CPMM-vs-HFMM experiment finalized

### Nice-to-have
- [ ] Lag-regression appendix
- [ ] More formal calibration objective
- [ ] Extended sensitivity appendix
