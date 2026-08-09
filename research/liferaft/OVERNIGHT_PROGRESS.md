# Overnight correction progress

## Pass 3.2 bounded correction baseline

- Baseline immediately before this pass: `72` Liferaft tests passed (`3.277s`)
  with the bundled Python runtime. The compile, demo, and experiment commands
  for this pass had not yet been run.
- The worktree status was preserved: existing changes outside
  `research/liferaft/` remain untouched. `PASS3_FINAL_REPORT.md` was absent and
  no final scenario constructor was called.
- Baseline status also included modified `research/research.ipynb`,
  `research/signal_hypotheses.md`, and `research/strategy_notes.md`; untracked
  `research/boat_party/`, `research/fintech_research.py`,
  `research/fintech_token/`, `research/jeans_parameter_policy_audit.py`,
  `research/thrifted_jeans/`, and
  `trader_interface/2024_data_DONOTUSENORMALLY/`. None is in scope for this
  pass.
- Reason for this bounded correction: review found that the existing
  portfolio-exposure results were being described as if they held the market
  path fixed, although focal budget flattening can change the endogenous game.
  The pass also adds a separate mechanics audit and a development-only,
  deterministic public-cycle diagnostic without changing candidate parameters.

## Pass 3.1 correction baseline

- Baseline before Pass 3.1 edits: `66` Liferaft tests passed (`0.900s`) with
  the bundled Python runtime.
- Baseline worktree status preserved: existing changes in
  `research/research.ipynb`, `research/signal_hypotheses.md`,
  `research/strategy_notes.md`, other research directories/files, and
  `trader_interface/2024_data_DONOTUSENORMALLY/`; `research/liferaft/` is the
  scoped work area. No production algorithm/simulator path was modified.
- `PASS3_FINAL_REPORT.md` was absent before editing and the locked suite
  remains unconsumed.
- Reason for this correction: review found that gradual drift cancelled at the
  population level and that `_persistent_population()` overwrote its intended
  day-indexed random component. The same review requested explicit path
  Hamming diversity and a separate constant-other-exposure portfolio
  sensitivity analysis. These are validation-design/mechanics corrections,
  not strategy-parameter tuning.
- Pass 3.1 construction is now frozen before experiment reruns: drift uses
  mirrored whole-population directions and predeclared strengths
  `{0.08, 0.10, 0.12, 0.14, 0.16, 0.18}`; persistent populations retain one
  seeded aligned random component and seeded persistence noise without audit
  pulses; portfolio levels are `{0, 150000, 300000, 450000}` for the five
  serious candidates only.
- Corrected non-final primary rerun completed once after those construction
  fixes: `9` development scenarios / `126` cells and `480` validation
  scenarios / `6,720` cells. The validation report includes actual tails,
  paired candidate differences, exact uniqueness, and Hamming diversity.
- The separate portfolio-sensitivity run completed once over `9,600` cells
  (`480` scenarios × `4` exposures × `5` serious candidates) and wrote
  `PASS31_PORTFOLIO_SENSITIVITY.md`. Its stored values are retained, but the
  interpretation is corrected: focal-only exposure is an endogenous game
  counterfactual, so focal flattening can change votes, prices, and reactive
  opponent actions. The decline cannot be attributed solely to focal
  rejection.
- After the primary run, a presentation-only report wording correction made
  the path-diversity/non-independence label explicit. It changed no metrics,
  strategy code, or parameters, so no primary rerun was warranted. The final
  manifest was re-hashed after this report-generator change.

## Baseline at goal start

- Existing Liferaft suite: `57` tests passed (`0.315s`) using the bundled
  Python runtime.
- Existing Pass 3 report: one combined development/validation report covering
  `188` scenarios and `2,632` strategy/scenario cells. Its headline ranking
  was not treated as authoritative because development cases were mixed into
  the ranking and family-level statistics were presented as run-level tails.
- `PASS3_FINAL_REPORT.md`: absent. The locked final suite has not been run.
- Worktree status was recorded before edits. Existing unrelated changes were
  preserved; this goal is restricted to `research/liferaft/`.

## Planned checkpoints

1. Correct inactive effective-position/observation semantics and add stateful
   boundary regressions.
2. Correct source-day pivotal P&L attribution.
3. Separate development diagnostics from validation-only rankings and make
   run-level downside statistics truthful.
4. Replace nominally varied validation cases with paired, symmetric,
   population-size-varied, path-audited scenarios.
5. Add online timing/weight/drift causality tests and a locked-suite manifest
   without executing it.
6. Run the allowed tests, compilation, demo, and one corrected development /
   validation experiment; perform a final scope and no-final-report audit.

## Checkpoint log

- Baseline confirmed: `57` tests passed before this correction pass.
- Implemented inactive effective-position semantics: raw requests remain
  diagnostic, effective inactive actions/holdings are flat, and boundary
  observations carry `own_position=0`.
- Implemented source-day pivotal P&L attribution with an exhaustive partition
  assertion and regression test.
- Rebuilt validation definitions as 12 families × 20 base seeds × 2 modes =
  `480` cases. Population sizes are `{3, 4, 5, 8, 9, 15}`. Superseded Pass
  3.0 checkpoint (retained for audit history): it reported 20/20 unique
  flat-focal paths in every family and mode. After the Pass 3.1
  persistent-population correction, `persistent_short` has 16 unique paths and
  four duplicate cases per mode; that earlier 20/20 wording is not current.
  The controlled-pair and state/RNG-evolution counts remain recorded in the
  corrected report.
- Corrected the runner/report design so development is diagnostic-only,
  validation supplies rankings, actual run-level downside statistics are
  separate from family-balanced means, and final execution writes a separate
  report.
- Added the locked-final manifest and source hash without constructing final
  scenarios. `PASS3_FINAL_REPORT.md` remains absent.
- Frozen non-final experiment completed successfully: 9 development cases /
  126 cells and 480 validation cases / 6,720 cells. The corrected report now
  ranks validation only and records actual run-level tails, paired mode
  differences, and path-diversity diagnostics. The later persistent-population
  audit shows 16 unique live signatures (four duplicates) for persistent-short
  in each mode; the earlier blanket “20 unique” wording is superseded.
- This was the required experiment rerun after mechanics/reporting/scenario
  corrections, not strategy tuning. Runtime was approximately `407s` with
  the bundled Python runtime.
- Final allowed validation commands completed after the experiment:
  `unittest discover` = `66` passed; `compileall -q` succeeded; `demo`
  completed with the legacy mixed/stateful/runaway outputs plus the cold-start
  smoke case. A narrow two-case, three-strategy rerun produced identical
  metrics, path audit, and rendered report; no second full validation run was
  performed.
- Corrected a report-only wording ambiguity so the flat baseline is explicitly
  reported as tied for best where appropriate; no metrics or strategy
  parameters changed, so the 407-second experiment was not rerun for that
  presentation-only edit.
- Final protocol audit: `PASS3_FINAL_REPORT.md` is absent, the locked final
  function was not called, and the manifest hash matches the current locked
  definition source.

This file is updated after each material implementation or validation rerun.
Any rerun after the required experiment will state whether it was caused by a
mechanics/reporting bug or by strategy changes. Strategy parameters remain
frozen throughout this goal.

## Remaining uncertainty

The organiser has not confirmed whether participant algorithms are called
during the inactive year. Both `observe_and_ignore_actions` and
`fully_inactive` remain explicit simulator modes; the former is the current
best assumption.

## Pass 3.1 final validation checkpoint

- Required final test command: `72` tests passed (`3.315s`) with the bundled
  runtime.
- Required compilation command: `python -m compileall -q research/liferaft`
  succeeded.
- Required demo command succeeded, including deterministic mixed, stateful
  boundary, runaway-budget, and cold-start smoke outputs.
- Corrected primary command completed once: `9` development scenarios / `126`
  cells and `480` validation scenarios / `6,720` cells; it wrote
  `PASS3_REPORT.md`.
- Portfolio command completed once: `480` scenarios × `4` exposures × `5`
  candidates = `9,600` cells; it wrote
  `PASS31_PORTFOLIO_SENSITIVITY.md`.
- Narrow reproducibility check reran representative persistent-long,
  persistent-short, gradual-drift, and startup cases and obtained identical
  results on repeated fresh runs.
- Final audit: the combined locked-code hash matches the manifest,
  `PASS3_FINAL_REPORT.md` is absent, and the production algorithm and supplied
  simulator paths have no status changes. No final scenario constructor was
  invoked.

## Pass 3.2 bounded correction completion

- The existing `PASS3_REPORT.md` and stored P&L tables in
  `PASS31_PORTFOLIO_SENSITIVITY.md` were not recomputed or numerically changed.
  Only the portfolio report's interpretation was corrected from “fixed path”
  wording to an endogenous game-counterfactual description.
- The new burnin1-only path audit ran once with the consumed validation cases:
  `480` scenarios × `4` exposures = `1,920` bounded mechanics cells. The
  zero-exposure self-check was exactly zero. Overall different live price and
  majority paths were `38.5%` at AUD150,000, `50.6%` at AUD300,000, and
  `65.4%` at AUD450,000. At AUD450,000 the controlled group was `62.0%` and
  the state/RNG-sensitive group was `82.5%`; mean differing opponent action
  cells were `92.0` and `213.0`, respectively. Full family rows are in
  `PASS31_PATH_DIVERGENCE.md`.
- The cycle development experiment ran on `21` new fixtures × `4` comparison
  candidates and wrote `CYCLE_REPORT.md`. The fixed detector activated on
  `13/13` true-cycle fixtures (`100.0%`) but also on `6/8` non-cycle controls
  (`75.0%` false activation); mean detection delay was `19.6` genuine
  observations and mean post-activation accuracy was `48.4%`. P&L is retained
  only as a secondary development diagnostic. The cycle remains quarantined
  and is not a production recommendation.
- The cycle experiment was rerun after a fixture-construction bug was found:
  the pure-period-13 branch still used the old contiguous 8-long/5-short block,
  which legitimately triggered a shorter local period-2 pattern. It was
  replaced with a fixed primitive 8-long/5-short block; this was a mechanics
  fixture correction, not strategy or parameter tuning.
- Final required checks for this pass: `86` unit tests passed (`4.043s`),
  `compileall -q research/liferaft` succeeded, and `python -m
  research.liferaft.demo` completed successfully. The path-audit command and
  final corrected cycle command both completed successfully.
- No normal full Pass 3 validation run and no full portfolio-sensitivity run
  was repeated. No command containing `--final` was run, no final scenario was
  instantiated, `PASS3_FINAL_REPORT.md` remains absent, and production files
  remain untouched.
- The locked manifest was re-hashed after the Pass 3 source changes. Cycle
  files are excluded from the lock; adding them later would require a
  deliberate candidate-catalogue change and explicit re-lock.
