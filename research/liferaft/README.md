# Liferaft Ticket research - Pass 1, Pass 2 (superseded), and Pass 3

This directory contains a small multi-agent simulator, opponent archetypes,
population/scenario helpers, and tests. It is deliberately independent of the
production trader and supplied portfolio simulator.

## Organiser clarification and current assumption

Roger clarified that Liferaft Ticket was fixed at `$100,000` throughout Round
1 / Year 1. The earlier changing Year-1 stream was erroneous bot-testing
data. Year 1 therefore contains no public information about the live
majority vote and must not be used to train an opponent model.

The current competition model is
`LiferaftConfig(market_mode="inactive_until_marked")`: days `0..364` are a
flat public history, day `365` is observed at `$100,000`, and the action
selected on day 365 determines the first genuine movement into day 366.
Marked P&L starts with that movement. There is no meaningful reset jump in
this mode, and no pre-voting action creates price movement or P&L.

The organiser has not yet confirmed whether participant algorithms are called
through the inactive period. This is explicit rather than hidden in the
simulator:

- `observe_and_ignore_actions` (the current best assumption): agents are
  called daily, see the growing flat history, and may evolve internal state;
  their raw pre-voting requests are logged but ignored by Liferaft price,
  votes, holdings, budget rejection, and P&L. `action` is the effective
  Liferaft action (flat while inactive); `market_action_ignored` explains why
  the raw request was not executed.
- `fully_inactive`: agents are not called before the voting-start day and are
  first called cold on day 365 with the flat public history.

Both modes prevent pre-voting positions from earning the first live interval.
The historical `continuous_reset` mode remains the default for Pass 1 and
legacy tests, so earlier mechanics can still be audited without silently
changing their assumptions.

## Timing and reset semantics

The supplied `trader_interface/simulation.py` exposes prices through the
current day before calling `get_positions()`. It then calculates day `t`
P&L using the position held from day `t - 1` over the movement from
`price[t - 1]` to `price[t]`, and stores the new desired position for the next
interval. This package mirrors that timing:

1. Day 0 starts at the configured initial price, has zero P&L, and agents
   choose positions for the interval to day 1. If
   `marked_boundary_day == 0`, the boundary event takes precedence: day 0
   starts at `reset_price`, is marked as the reset row, and still has zero
   realised P&L.
2. On every day, all agents receive the same public history through that
   day's price and decide before any same-day vote is counted.
3. The effective votes on day `t` determine the market-generated price for
   day `t + 1`: long majority is `-5,000`, short majority is `+8,000`, and a
   tie/all-flat is zero, with a configurable floor.
4. `marked_boundary_day` is an index. With the competition default of 365,
   price `365` is overwritten with `$100,000`; the reset jump is logged and
   contributes zero P&L. The action chosen on day 365 first earns or loses
   P&L on the movement into day 366.
5. Calibration P&L is the sum of daily P&L for days `< marked_boundary_day`.
   Marked P&L is reset to zero at the boundary and contains only daily P&L
   for days `> marked_boundary_day` (the boundary row itself is zero).

`AgentDayRecord.action`, `majority`, and `status` are decision-time fields:
they describe the newly selected vote on the current day. `daily_pnl` is
realised from the position held over the preceding interval, so its matching
fields are `pnl_position`, `pnl_source_day`, `pnl_majority`, and `pnl_status`.
The realised fields point to the prior day's effective action and majority.
They are all `None` for day 0 and the artificial reset row. Genuine zero or
floor-clipped movements still retain the engine-known prior majority in these
fields, even when a competitor could not infer it from public prices.

In inactive-until-marked mode, `DayRecord.voting_active` identifies whether a
row is a live vote. Pre-voting `action` records are effective flat market
positions, while `requested_action` retains the raw request and
`market_action_ignored` identifies the inactive diagnostic; the day-365 action
is the first action that determines a future price. `own_position` is zero on
every inactive observation and on the voting-start observation.

The PDFs specify the Liferaft starting price, position limit, majority
mechanics, floor, and the Round 1/Round 2 scoring split. They do not specify a
single continuous 730-day Liferaft price stream, the exact reset tick, or the
P&L treatment of a reset jump. The continuous two-period model and indexed
reset above are therefore the explicit Pass 1 assumption, chosen to match the
supplied simulator's one-day holding interval and the project brief. Both the
boundary index and all mechanics are configurable so an organiser clarification
can be tested without changing agent code.

Agents see only `AgentObservation`: public price history, the latest public
price change, their own held position, and known competition configuration.
Public majority inference is sign-based for genuine moves: negative means
long majority, positive means short majority, and zero is ambiguous. Reset
moves are always excluded. Vote counts, other agents' actions, pivotality, and
budget diagnostics are engine-only records. Invalid actions are flattened and
logged. A non-zero action that would exceed the gross budget after optional
other-portfolio exposure is also flattened before it can enter the vote. If
other exposure alone exceeds the budget while the Liferaft request is already
flat, the engine records a budget breach but not a rejected Liferaft action.

`Scenario.run()` obtains fresh agents from its population factory on every
call. Direct agent tuples are copied from a pristine snapshot captured when
the scenario is constructed. In contrast, a `LiferaftSimulator` is a
single-use run object: calling `.run()` twice raises a clear `RuntimeError`
instead of silently reusing stateful agents or RNGs.

## Commands

From the repository root:

```text
python -m unittest discover -s research/liferaft -p "test_*.py"
python -m compileall -q research/liferaft
python -m research.liferaft.demo
python -m research.liferaft.pass3_experiments
python -m research.liferaft.pass3_experiments --portfolio-sensitivity
python -m research.liferaft.pass3_experiments --portfolio-path-audit
```

The demo runs a deterministic mixed population, a stateful population across
the reset, and a runaway short-majority case. Pass 1 intentionally does not
modify or optimise `trader_interface/algorithm.py`.

## Pass 2 causal research framework (historical and superseded)

Pass 2 lives in `strategies.py`, `calibration.py`, `pass2_scenarios.py`, and
`pass2_experiments.py`. It is a deliberately small research catalogue, not a
production trading strategy. The declared candidates are flat, fixed-long,
fixed-short, last-majority counter, rolling frequency, regularised Markov,
periodic replay, Year-1-selected, a small ensemble, and a drift-aware policy.

All candidate code receives only `AgentObservation`. Public labels are derived
from genuine price changes; reset moves are skipped and genuine zero moves are
kept as unknown/tie observations. A negative movement is long-majority evidence
and a positive movement is short-majority evidence, including a clipped move at
the floor. The asymmetric break-even threshold is
`P(short majority) = 5,000 / (5,000 + 8,000) = 5/13`; the fixed no-trade
margin and confidence guard are applied conservatively.

Year-1 selection uses chronological walk-forward prefixes after a fixed warm-up
and three contiguous validation blocks. A candidate must beat flat by the
fixed block margin and show stable block evidence; fixed turnover and complexity
penalties are applied; otherwise flat is selected.
Selected names, replay periods, and ensemble weights are frozen at the marked
boundary. The drift policy observes a realised label only on the next decision
after that movement has appeared and can switch once to its predeclared
ensemble fallback. It never reads hidden counts, focal pivotality, or marked
P&L diagnostics.

`Pass2Scenario` factories construct fresh stateful and seeded-random opponents
for every run. The experiment runner keeps development, validation, and new
held-out suites separate and reports P&L distribution, drawdown, active-day
hit rate, turnover, budget diagnostics, pivotal/non-pivotal results, flat
outperformance, and hindsight regret. Held-out results are not fed back into
selection. These results are now consumed historical research only. The
changing Year-1 price assumption was wrong, so Year-1 selection, replay, and
the old “Markov is best” conclusion are superseded and must not be used as
current recommendations. The old implementation is retained and replay is
quarantined for auditability.

## Pass 3 cold-start research

Pass 3 is in `cold_start_strategies.py`, `pass3_scenarios.py`, and
`pass3_experiments.py`. It uses only online marked-period information. The
predeclared candidates are flat, fixed long/short benchmarks, flat burn-ins
of 1/3/5/10 live observations, a last-observed-majority counter, rolling
frequency with windows 5/10/20 and smoothing, order-one/two regularised
Markov, a bounded flat/last/rolling/Markov ensemble, a one-way drift fallback,
and immediate-long versus flat-first asymmetric-prior benchmarks.

The first live action is chosen with no live labels. At each later decision a
strategy may update only from the immediately preceding genuine movement;
unknown zero movements are not invented as majority labels. Expected-P&L
decisions use the asymmetric payoff (`-5,000` for a long majority and `+8,000`
for a short majority), a conservative no-trade margin, support guards, and
fixed pivotality-agnostic public information. Pivotal and non-pivotal results
are measured by the engine but never exposed to candidates.

The normal Pass 3 command runs development and validation scenarios only.
Validation uses new paired seeded families, both inactive execution modes,
mirrored biases, varied population sizes, and day-indexed schedules where a
controlled same-path comparison is useful. `final_scenarios()` uses a separate
locked seed range and is not run unless explicitly requested:

```text
python -m research.liferaft.pass3_experiments --final
```

`PASS3_REPORT.md` is the current report. It separates development diagnostics
from validation-only rankings. Validation tables use actual run-level mean,
median, lower quartile, worst run, mean and maximum drawdown, active days, hit
rate, turnover, budget diagnostics, flat comparisons, regret, and
pivotal/non-pivotal results. A separate family-balanced table reports only
family means, without manufacturing family-averaged quantiles or worst runs.
The report also includes paired mode differences and effective unique-path
counts plus minimum and mean pairwise live-majority Hamming distances. These
path distances are diversity diagnostics, not claims of statistical
independence. Turnover is a stability preference rather than a transaction cost;
Pass 2's historical selection utility compares total P&L with total turnover
rather than mixing block means with total turnover.

`PASS3_FINAL_MANIFEST.md` records the locked final families, seed blocks,
population-size range, paired execution modes, and source hash. It is an audit
manifest only; the final cases are not instantiated during the normal runner.

Run the complete tests, demo, and current Pass 3 report from the repository root:

```text
python -m unittest discover -s research/liferaft -p "test_*.py"
python -m compileall -q research/liferaft
python -m research.liferaft.demo
python -m research.liferaft.pass3_experiments
```

## Pass 3.1 validation-design and portfolio sensitivity correction

Pass 3.1 corrects two consumed-validation construction bugs. Gradual-drift
populations now give every agent the same declared whole-population drift
direction, with mirrored positive/negative directions, predeclared strengths,
seeded randomness, and modest strength heterogeneity. Persistent populations
retain one seeded, mostly aligned day-indexed random component while the rest
remain noisy persistence agents; no one-day audit pulses are used to create
path uniqueness. Scenario metadata records drift direction and strength.

The normal report now includes exact unique-path counts and pairwise Hamming
distance diagnostics for flat-focal live majority paths. These describe
effective path variation within each family/mode and are not treated as
independent statistical samples. It also includes same-scenario paired P&L
comparisons for `burnin1_markov`, `burnin3_markov`, and `online_markov`.

The separate portfolio-sensitivity command reruns the consumed validation
scenarios with the fixed serious-candidate set `flat`, `burnin1_markov`,
`burnin3_markov`, `online_markov`, and `online_drift`, at constant other gross
exposures of AUD 0, 150,000, 300,000, and 450,000:

```text
python -m research.liferaft.pass3_experiments --portfolio-sensitivity
```

It writes `PASS31_PORTFOLIO_SENSITIVITY.md` and does not alter the primary
zero-exposure ranking in `PASS3_REPORT.md`. The constant other exposure is
applied to the focal agent, but the simulator remains endogenous: a forced
focal flattening can change the majority, price path, reactive opponent
actions, and their budget feasibility. The stored P&L values are valid under
that game-counterfactual interpretation; their decline cannot be attributed
solely to focal rejection. This is a sensitivity model, not a forecast of the
final portfolio allocation.

The bounded path-divergence audit is separate from that stored P&L sensitivity:

```text
python -m research.liferaft.pass3_experiments --portfolio-path-audit
```

It uses only `burnin1_markov` and compares the same consumed validation case at
AUD 0, 150,000, 300,000, and 450,000 focal other exposure. It reports changed
live price paths, majority paths, and opponent effective-action cells. Because
the focal action participates in the endogenous vote, these are mechanics
counterfactual diagnostics rather than fixed-path budget-capacity measures or
new strategy-validation results. The report is written to
`PASS31_PATH_DIVERGENCE.md`.

## Bounded deterministic-cycle research (development-only)

`cycle_strategies.py`, `cycle_scenarios.py`, and `cycle_experiments.py` are a
quarantined mechanics experiment. The detector watches only public genuine
`LONG`/`SHORT` labels, requires three identical consecutive blocks for a fixed
period in `2..20`, chooses the shortest qualifying period, and deactivates on
unknown/reset/clipped labels or a contradiction. It uses the existing asymmetric
expected-P&L action safeguards. It is not in `COLD_START_STRATEGY_NAMES`, is
not evaluated on consumed Pass 3 validation, and is not in the locked-final
catalogue.

Run its small development fixtures separately:

```text
python -m research.liferaft.cycle_experiments
```

The result is written to `CYCLE_REPORT.md`. These deterministic fixtures are
mechanics diagnostics only. The current report concludes that the detector is
**not worth pursuing in its present form**: its causal fixture behavior does
not overcome the observed control activation and multiple-testing problem.
Any future redesign or consideration would require a separately frozen,
unseen evaluation and a new final lock.

## Pass 4A production-risk experiment

Pass 4A is one bounded wrapper around the consumed-validation leader
`burnin1_markov`. `risk50_burnin1_markov` keeps the existing Markov model and
adds only the frozen AUD 50,000 sticky loss stop, fixed forecast-health stop,
one-decision zero-movement pause, exact-floor flat gate, and AUD 10,000
portfolio-headroom reserve. Its protocol is recorded in
`PASS4A_PROTOCOL.md`; results are written separately to
`PASS4A_REPORT.md`.

The runner uses only the consumed `validation_scenarios()` at fixed other
gross exposures of AUD 0, 150,000, 300,000, and 450,000. Exposure gates can
change the focal vote and therefore later prices, opponents, and budget
feasibility, so these are endogenous same-initial-scenario counterfactuals,
not fixed-path backtests. Pass 4A does not construct or execute final cases.

## Pass 4 final locked evaluation

The final catalogue now adds `risk50_burnin1_markov` to the unchanged 14
candidates, producing 15 candidates across 160 locked scenarios (2,400 cells).
The frozen conjunctive production gate is recorded in
`PASS4_FINAL_DECISION_PROTOCOL.md`. The final suite is intentionally one-way:
only the explicit command below may consume it, and it writes the final
Markdown report, machine-readable results, decision, and execution receipt:

```text
python -m research.liferaft.pass3_experiments --final
```

The final lock covers the simulator, existing candidate/scenario code, the
Pass 4 wrapper, and the final runner. Passing synthetic final cases is not
proof of real competition expected value. After consumption, no result-
affecting research source or frozen wrapper parameter may be changed.
