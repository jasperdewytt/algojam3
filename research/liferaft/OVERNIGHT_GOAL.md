# Liferaft overnight correction and robustness goal

## Outcome

Make the clarified cold-start Liferaft research harness reliable enough to
support a later production-strategy decision. Correct the known simulator,
metric, reporting, and scenario-design problems; rerun development and
validation; and leave a concise, reproducible audit trail.

This is an infrastructure and validation-quality pass. It is **not** permission
to implement the production trader, repeatedly tune against validation, or run
the locked final suite.

## Authoritative competition assumption

Use the organiser clarification currently documented in this directory:

- Liferaft is fixed at `$100,000` during days `0..364`.
- Live voting starts from the `$100,000` observation on day `365`.
- The action selected on day 365 determines the first genuine movement into
  day 366.
- Year 1 contains no observable majority-vote information.
- Year 1 calibration and replay conclusions from Pass 2 are superseded.

Retain both pre-voting execution interpretations:

1. `observe_and_ignore_actions`: agents are called during Year 1 and their
   internal object state may evolve, but Liferaft requests are not executed.
2. `fully_inactive`: agents are first called for Liferaft on day 365.

The current best assumption is `observe_and_ignore_actions`, but results must
show sensitivity to both modes.

## Hard boundaries

- Work only inside `research/liferaft/`.
- Do not modify `trader_interface/algorithm.py`, the supplied portfolio
  simulator, notebooks, or research for other instruments.
- Preserve unrelated and pre-existing worktree changes.
- Do not run `python -m research.liferaft.pass3_experiments --final` or call
  `final_scenarios()` through another path.
- Do not inspect, calculate, or report locked-final outcomes.
- Do not use hidden votes, vote counts, margins, pivotality, other agents'
  actions, or future prices inside candidate strategies.
- Do not add high-capacity ML, reinforcement learning, neural networks, or a
  broad hyperparameter search.
- Freeze the existing candidate strategy parameters during this pass. Correct
  implementation errors, but do not tune windows, priors, smoothing,
  confidence thresholds, burn-ins, ensemble learning rates, or drift
  thresholds after viewing the new validation results.
- Do not use network access or require new dependencies.
- Do not perform destructive Git or filesystem operations.

## Phase 1: establish the baseline

Before editing:

1. Read `AGENTS.md`, the Liferaft README, simulator, cold-start strategies,
   Pass 3 scenarios, experiment runner, tests, and reports.
2. Run the existing Liferaft unit tests and record the baseline result.
3. Record the current file status without modifying unrelated files.
4. Create or update `OVERNIGHT_PROGRESS.md` with the baseline, planned phases,
   and later checkpoint results. Keep it concise.

## Phase 2: correct inactive-market position semantics

In `observe_and_ignore_actions` mode, pre-voting requests are currently able to
flow into `held_positions` and appear as `own_position` on day 365. Correct
this.

Required semantics:

- Agents are called on every pre-voting day and receive the growing flat price
  history.
- Their internal Python object state may evolve because `decide()` is called.
- Raw requested actions may be retained as explicit diagnostics.
- The effective Liferaft position is flat throughout the inactive period.
- Pre-voting requests do not become votes, holdings, P&L, or a carried market
  position.
- `own_position` is zero on every pre-voting observation and on the day-365
  voting-start observation.
- The action newly selected on day 365 becomes the first live held position
  and vote.
- Inactive actions cannot cause budget rejection records that imply a
  Liferaft trade was attempted in the live market. If retaining validation
  diagnostics for raw requests, distinguish them clearly from live rejected
  actions.
- `fully_inactive` remains cold and backward-compatible with its documented
  behavior.
- The legacy `continuous_reset` mode and all existing Pass 1 mechanics remain
  backward-compatible.

Add explicit regression tests using a stateful agent whose boundary action
depends on `observation.own_position`. Prove that internal call state persists
while ignored market positions do not.

## Phase 3: correct P&L diagnostic attribution

Pass 3 P&L on record day `t` belongs to the position and majority from
`pnl_source_day`, normally `t-1`.

Fix pivotal/non-pivotal P&L attribution so it checks:

```text
result.days[record.pnl_source_day].focal_pivotal
```

and never current-day pivotality. Handle `None` defensively.

Add a regression test in which the focal agent is pivotal on the source day,
changes action on the realization day, and earns or loses P&L. The metric must
assign that P&L to the source day's pivotal category. Confirm:

```text
pivotal_pnl + non_pivotal_pnl == marked_pnl
```

for every Pass 3 result.

## Phase 4: correct experiment separation and statistics

Development scenarios are correctness/design cases and must not influence the
headline strategy ranking.

Change the normal Pass 3 experiment so it:

1. Runs development and validation without running final.
2. Reports development separately as diagnostics.
3. Uses **validation only** for rankings and conclusions.
4. Clearly displays scenario count and strategy/scenario cell count for each
   suite.

Report two distinct kinds of aggregation without mixing their meanings:

### Actual validation distribution

Across all validation scenarios for each strategy, report the real:

- mean and median marked P&L;
- lower quartile;
- minimum/worst run;
- mean and maximum drawdown;
- mean P&L per marked day;
- active days, positive-P&L hit rate, turnover, breaches and rejections;
- fraction outperforming flat;
- fraction of scenarios where flat is tied for best;
- regret against the best predeclared candidate for that scenario.

The displayed `worst`, `lower quartile`, and `maximum drawdown` must be computed
from the actual validation runs. Never label an average of family-level worst
results as the worst run.

### Family-balanced comparison

Separately report the mean of each family's mean P&L or mean P&L/day so large
families do not dominate. Label this explicitly as a family-balanced mean.
Do not manufacture family-balanced medians, quantiles, worst runs, or maxima by
averaging those statistics across families.

Keep turnover as a diagnostic, not a transaction cost. Show no-turnover
ranking; do not introduce a new tuning penalty.

## Phase 5: make validation scenarios genuinely diverse and symmetric

The present nominal seed count overstates independent diversity. Replace or
extend consumed validation scenarios while retaining the realistic
`730/365` timeline.

Requirements:

- At least 20 genuinely distinct seeded cases for each important stochastic
  family.
- Add paired short-biased and long-biased random families with mirrored
  probabilities.
- Add balanced random families.
- Vary population sizes across a small predeclared set, for example
  `3, 5, 9, 15`, including odd/even populations where meaningful.
- Vary fixed and stochastic majority margins so the focal agent ranges from
  clearly non-pivotal to frequently pivotal.
- Include persistent-long and persistent-short validation cases, preferably
  with small seeded noise rather than twenty identical copies.
- Regime-change cases must vary pre/post proportions, switch dates, directions,
  and seeds. Do not merely rotate agent names within an identical aggregate
  path.
- Gradual-drift cases must include drift in both directions and different
  strengths.
- Periodic cases must produce genuinely different aggregate patterns/phases,
  not a tiny combination repeated across nominal seeds.
- Reactive cases should vary mixtures of followers, counters, fixed agents,
  random agents, and win-stay/lose-shift behavior.
- Add startup/flat-history agents whose behavior actually differs between the
  two execution interpretations. Verify the paired cases do not accidentally
  collapse to the same aggregate majority.
- Include floor, runaway-budget, tie/zero, and no-trade-friendly stress cases,
  but keep these stress cases separate from stochastic ranking evidence when
  they are single deterministic paths.

For execution-mode sensitivity, construct paired scenarios wherever possible:
same population definition and seed, differing only in pre-voting execution
mode. Report the paired difference by strategy. If RNG consumption is intended
to differ because agents were called during Year 1, label that as state/RNG
evolution rather than a controlled same-path comparison. Prefer day-indexed
random schedules for at least one controlled paired family.

Add a duplicate-signature audit. It should detect validation scenarios that
produce identical opponent-only or flat-focal live majority/price paths within
the same family and report the effective unique-path count. Do not fail merely
because an intentional deterministic stress pair matches, but make accidental
duplication visible.

## Phase 6: audit online strategy correctness

Do not tune parameters. Verify and test:

- Every cold-start strategy is flat before live voting unless it is an explicit
  fixed-action diagnostic whose inactive request is ignored by the market.
- Models append at most one label per newly observed genuine live interval.
- Reset/inactive intervals never become labels.
- Genuine zero/floor-clipped public moves remain unknown labels.
- Markov context does not bridge unknown observations.
- Ensemble weights update only from the previous decision's forecast after its
  outcome becomes visible.
- Ensemble weights remain finite, normalized, and within declared bounds.
- Drift quality does not count a forecast with no predicted majority or
  insufficient model support as a failed prediction. It must not evaluate an
  outcome before observation.
- Future price or opponent-action perturbations cannot change earlier focal
  actions for rolling, Markov, ensemble, drift, and burn-in strategies.
- All actions are exact integers in `{-1, 0, 1}`.
- Budget-forced flattening behaves safely at runaway prices.

Correct genuine implementation bugs found by these tests, but do not alter
predeclared model parameters in response to P&L.

## Phase 7: improve the locked-final protocol without executing it

Do not run or otherwise evaluate final scenarios.

It is permissible to improve their definitions before permanently locking
them, provided no outcomes are computed. The locked suite should cover more
than new seeds from the same four validation distributions. Include unseen
compositions for:

- symmetric random biases;
- reactive mixtures;
- periodic behavior;
- regime changes and drift;
- startup-mode sensitivity;
- pivotal and non-pivotal margins.

Keep its seed range distinct. Store a concise manifest of family names, seed
ranges, population-size ranges, and a hash of the locked definition source.
The normal runner must not import, instantiate, or execute the locked cases if
that can be avoided.

Running `--final` must:

- require the explicit flag;
- write to a separate `PASS3_FINAL_REPORT.md`, never overwrite validation;
- state truthfully that the final suite was executed;
- never claim that it remains unconsumed after execution.

Do not create `PASS3_FINAL_REPORT.md` during this goal.

## Phase 8: run validation once and report honestly

After correctness tests and scenario definitions are complete:

1. Freeze the strategy code and its parameters.
2. Run the consumed development and validation experiment once.
3. Do not change strategy parameters afterward.
4. If an implementation/reporting bug requires another run, document why in
   `OVERNIGHT_PROGRESS.md`; do not use the result to tune strategy behavior.
5. Generate `PASS3_REPORT.md` with development diagnostics, validation-only
   evidence, execution-mode sensitivity, unique-path counts, and explicit
   downside statistics.

The report must not call a strategy robust merely because it has the largest
mean. Discuss lower-tail loss, drawdown, pivotal behavior, budget rejection,
family dependence, and how often flat is preferable. If `burnin1_markov`
remains first, describe it as the consumed-validation leader rather than a
production recommendation.

## Required validation

Run from the repository root, using the bundled Python runtime if plain
`python` is unavailable:

```text
python -m unittest discover -s research/liferaft -p "test_*.py"
python -m compileall -q research/liferaft
python -m research.liferaft.demo
python -m research.liferaft.pass3_experiments
```

Do **not** run the `--final` form.

Also verify:

- all legacy and new tests pass;
- normal Pass 3 execution never constructs or runs locked-final scenarios;
- `PASS3_FINAL_REPORT.md` does not exist;
- no production or unrelated files were changed by this goal;
- generated validation results are deterministic on a second report-only or
  narrowly scoped reproducibility check. Avoid a full second validation run
  unless necessary.

## Definition of done

The goal is complete only when all of the following are true:

- Inactive-market position semantics match this document and have regression
  tests.
- Pivotal P&L attribution uses `pnl_source_day` and is tested.
- Development is excluded from validation ranking.
- Actual quantile, worst-run, and maximum-drawdown statistics are truthful.
- Validation contains symmetric, varied, meaningfully distinct populations.
- Effective unique-path counts are measured and reported.
- Execution modes have controlled paired sensitivity cases.
- Online ensemble and drift timing have explicit causality tests.
- All tests, compilation, demo, and non-final Pass 3 experiments succeed.
- Pass 2 remains clearly marked superseded.
- The locked final suite has not been run and no final report exists.
- `OVERNIGHT_PROGRESS.md` records baseline, changes, commands, results, any
  reruns, and remaining uncertainty.
- The final response reports files changed, exact commands/results, validation
  leader and downside, scenario uniqueness, execution-mode sensitivity,
  remaining organiser questions, and explicit confirmation that final was not
  executed.

If blocked by a genuine ambiguity, preserve both configurable interpretations,
document the blocker, and continue with all work that does not depend on it.
Do not silently choose new competition mechanics.
