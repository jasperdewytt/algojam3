# Liferaft Pass 4A protocol

Status: **frozen before the Pass 4A experiment**.

Pass 4A is one production-risk experiment around the already-consumed
validation leader `burnin1_markov`. It is not a new model search and it does
not consume, construct, or inspect the locked final suite. The experiment uses
only `validation_scenarios()` and treats each result as an endogenous game
counterfactual: a focal action or exposure gate can change later prices,
majorities, opponent actions, and budget feasibility.

## Candidate

The only new candidate is `risk50_burnin1_markov`. It constructs a pristine
`make_cold_start_strategy("burnin1_markov")` delegate for every run. The
delegate is unchanged and retains:

- Markov order 2;
- Laplace alpha 1.0;
- one genuine-observation burn-in;
- minimum expected P&L AUD 1,000;
- confidence margin 0.10;
- existing public unknown/reset handling.

The wrapper never reads vote counts, margins, pivotality, opponent actions,
simulator diagnostics, or future prices.

## Frozen wrapper controls

```text
MAX_LIFERAFT_LOSS_AUD = 50_000
HEALTH_WINDOW = 8
HEALTH_MIN_OBSERVATIONS = 8
HEALTH_MIN_HIT_RATE = 0.40
HEALTH_BAD_STREAK_REQUIRED = 2
PORTFOLIO_RESERVE_AUD = 10_000
GROSS_PORTFOLIO_BUDGET = 600_000
```

At the start of a new decision day, the wrapper first accounts for
`observation.own_position * observation.previous_price_change` exactly once.
Only intervals ending after the voting-start day and not marked as reset are
counted. The actual public price change is used, including a floor-clipped
change. Inactive-year, day-zero, reset, and duplicate-day observations add no
realised P&L.

The wrapper scores the forecast stored on the preceding decision only after a
genuine nonzero public movement is observable. A forecast is scoreable only if
it has positive support and a non-tied predicted majority. Eight scoreable
forecasts are required before judging health. A rolling hit rate below 0.40
for two consecutive evaluations permanently activates the health stop.

Once cumulative marked P&L is at or below `-50,000`, the loss stop permanently
activates. A crossing may overshoot only through the final already-observable
adverse movement; the run records that movement and the pre/post cumulative
values.

After the delegate updates and its new forecast is stored, the sticky loss and
health stops take precedence. Otherwise, a genuine zero movement causes one
flat decision, the exact price floor causes a flat decision, and a nonzero
delegate request is allowed only when:

```text
other_gross_exposure + abs(action) * current_price + 10,000 <= 600,000
```

Equality is permitted. Negative, non-finite, boolean, and otherwise invalid
exposure values are rejected clearly. The same shared exposure source is used
by the wrapper and the simulator; the runner caches one validated value per
public decision day so both see the identical value. The gate is a causal
gross-exposure check, not a forecast of the complete other-instrument
portfolio.

All diagnostics are wrapper-local and public-information causal. Duplicate
calls for one day return the cached action and do not repeat accounting,
forecast scoring, or gate counters. Both inactive execution modes remain
flat before live voting.

## Experiment matrix

The fixed comparison is:

```text
flat
burnin1_markov
online_drift
risk50_burnin1_markov
```

Each candidate runs at each of the four fixed focal other-exposure levels
`AUD 0`, `AUD 150,000`, `AUD 300,000`, and `AUD 450,000` over all 480
consumed validation scenarios: `480 * 4 * 4 = 7,680` scenario/strategy /
exposure cells. No final cases are used.

The report includes run-level and family-level downside/upside statistics,
drawdown, active days, turnover, flat comparisons, pivotal/non-pivotal P&L,
budget diagnostics, paired P&L differences, stop and gate diagnostics,
overshoot, and path-dependence warnings. Turnover is reported as a stability
diagnostic, not subtracted as a transaction cost.

No parameter, candidate, or scenario definition may be changed after the
experiment output is observed.
