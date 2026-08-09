# Liferaft Research Pass 5A Protocol

Status: frozen before the full Pass 5A development experiment.

## Hypothesis

A causal order-two regularised Markov predictor can identify populations with
useful short-horizon economic structure, but its first trades are vulnerable to
startup noise.  A short genuine public-information paper-trading period, with
the same asymmetric dollar payoff used for real decisions, should filter some
of those environments while retaining a material share of the persistent
opportunity.  Shadow validation is a development hypothesis test only; it is
not a production acceptance decision or a fresh holdout.

## Frozen candidates

The new candidate is `ShadowValidatedMarkov`.  All three candidates use the
same implementation and differ only in `minimum_genuine_nonzero_observations`:

| name | order | alpha | warm-up | minimum scoreable virtual trades | health window | initial virtual P&L | recent virtual P&L | deactivation window | cooldown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shadow8_markov` | 2 | 1.0 | 8 | 6 | 12 | AUD 10,000 | AUD 5,000 | AUD -10,000 for 2 evaluations | 5 genuine observations |
| `shadow12_markov` | 2 | 1.0 | 12 | 6 | 12 | AUD 10,000 | AUD 5,000 | AUD -10,000 for 2 evaluations | 5 genuine observations |
| `shadow20_markov` | 2 | 1.0 | 20 | 6 | 12 | AUD 10,000 | AUD 5,000 | AUD -10,000 for 2 evaluations | 5 genuine observations |

The current forecast must independently pass `payoff_action` with minimum
expected P&L AUD 1,000 and confidence margin 0.10.  The Markov model uses
alpha-1 Laplace smoothing (`alpha=1.0`).  Only public, non-reset, non-zero,
non-floor-clipped movements become `LONG` or `SHORT` labels.  Zero, reset and
floor-clipped movements append an unknown context break and are not scoreable
shadow evidence.

Required comparators are `flat`, existing `burnin1_markov`, existing
`online_markov`, and existing `risk50_burnin1_markov`.  Comparators provide
context only and cannot be selected as the Pass 5A challenger.

## Causal timing and virtual P&L

At a live decision on day `t`, the strategy first processes the newly visible
movement from `t-1` to `t` exactly once.  It accounts actual realised P&L from
the effective prior position, scores the virtual position selected on day
`t-1` using the public price change, and only then appends the inferred public
label to the Markov history.  It estimates the next forecast from that updated
history and selects the virtual action for the interval `t` to `t+1`.

Virtual long P&L is the actual public price change and virtual short P&L is its
negative.  A non-zero virtual action earns a scoreable virtual trade only on a
genuine, non-zero, non-clipped public movement.  Unscoreable intervals are
recorded with their raw public movement and raw virtual P&L for audit, but do
not enter cumulative or recent shadow P&L, health streaks, or activation
evidence.  The qualification-causing interval is never traded retroactively.
Duplicate calls for an already processed day return the cached action without
mutating state.

## Activation, deactivation, and hysteresis

Before live voting, and at the voting-start observation, the real position is
flat.  Virtual forecasting and scoring continue whenever a live observation is
available.  Initial or reactivation qualification is evaluated only after a
newly scoreable virtual trade.  The latest available up-to-12 scoreable
virtual-trade P&L values are used as the recent shadow window; at least six
scoreable trades are required.  A qualifying evaluation requires:

1. the warm-up count of genuine non-zero, non-clipped observations;
2. at least six scoreable virtual trades;
3. cumulative virtual P&L of at least AUD 10,000;
4. recent-window virtual P&L of at least AUD 5,000; and
5. the current forecast independently clearing the economic edge and
   confidence gates.

Two consecutive newly scoreable qualifying evaluations set an activation
pending state.  The real position remains flat on that evaluation day and
starts at the next causally available decision.  `activation_day` is that
first real decision day.

While active, the strategy keeps using the current economic edge and all risk
gates, but does not require requalification after an isolated shadow error.
The latest shadow window is checked after each newly scoreable virtual trade.
If it is AUD -10,000 or worse for two consecutive such evaluations, the real
position is flattened on the current decision and the strategy deactivates.
It then spends five subsequent genuine live observations in cooldown.  A
zero or clipped movement is still an observed live interval for cooldown
timing, but never supplies a Markov label or scoreable health evidence.  After
cooldown, the same original activation criteria and two-evaluation rule apply.
The virtual paper process continues during cooldown.

An unknown/zero or clipped observation causes a one-decision real safety
pause and breaks the Markov context.  It does not by itself deactivate an
active strategy.  A weak current forecast produces a flat real position but
does not by itself deactivate the strategy.

## Risk and portfolio controls

The strategy retains the following fixed controls:

- sticky actual cumulative Liferaft loss stop at AUD 50,000;
- actual loss-stop overshoot is recorded after the adverse movement is
  observable;
- flat at the exact public price floor;
- a AUD 10,000 reserve/headroom gate, requiring
  `other_exposure + abs(action)*price + 10,000 <= 600,000`;
- integral positions in `{-1, 0, 1}` only;
- no focal budget breaches or rejected actions;
- actual P&L from the effective prior position and newly observed public
  movement;
- at most one underlying callable-exposure evaluation per live focal day,
  cached for duplicate and simulator-side use.

The strategy never reads hidden vote counts, vote margins, pivotality, hidden
majorities, simulator P&L, or any other engine-only diagnostic.

## Allowed evidence and quarantine

The full development experiment may use only the existing constructors
`development_scenarios()` and `validation_scenarios()` from
`research/liferaft/pass3_scenarios.py`, including both existing inactive-year
execution modes and the four fixed exposure levels AUD 0, 150,000, 300,000,
and 450,000.  The validation cases are consumed historical development
evidence, not a fresh holdout.  A small deterministic callable-exposure audit
uses an existing scenario and checks caching and endogenous path changes; it
does not create a favourable portfolio trace.

The prohibited consumed final artifacts are not read, parsed, imported,
executed, recreated, renamed, deleted, or overwritten:

- `PASS3_FINAL_REPORT.md`
- `PASS4_FINAL_RESULTS.json`
- `PASS4_FINAL_DECISION.md`
- `PASS4_FINAL_EXECUTION_RECEIPT.json`
- `PASS3_FINAL_MANIFEST.md`
- `pass4_final.py`

No command containing `--final` is permitted, `final_scenarios()` is not
called, and no production strategy, locked strategy, simulator, scenario
source, or final catalogue is modified.

## Metrics

Raw results will be recorded per run and summarised from run-level values:
marked P&L mean, median, lower quartile, worst run, mean and maximum drawdown,
active days, real non-zero days, turnover, focal breaches, rejected actions,
pivotal and non-pivotal P&L, activation rate and day, never-activated
fraction, deactivation/reactivation frequencies, active/paused/cooldown time,
virtual P&L before and after activation, real P&L after activation, current
edge and headroom/floor/unknown/loss-stop gate counts, paired P&L against flat
and `risk50_burnin1_markov`, positive-upside retention, downside avoided,
family summaries, lifecycle-mode summaries, and exposure sensitivity.

Any missed-opportunity or paired comparison that uses the path generated
after the focal strategy changes its vote will be labelled a realised-path
diagnostic, not an opponent-only counterfactual.  Pivotal and non-pivotal
partitions are engine-only reporting dimensions and are never exposed to the
strategy.

## Development-only interpretation and screening rule

Pass 5A is a development screening pass over consumed scenarios.  It may
identify one challenger to carry into a future blind Pass 5B, but it does not
establish production suitability or a new holdout result.  A shadow candidate
is eligible only if, across the development experiment, focal budget breaches
are zero, focal rejected actions are zero, worst P&L is at least AUD -60,000,
maximum drawdown is at most AUD 75,000, aggregate mean P&L is positive, and
actual loss-stop/overshoot diagnostics are internally consistent.

Among eligible shadow candidates, select the highest aggregate mean P&L.  If
means differ by less than 5%, prefer `shadow12_markov`; if it is not in that
tie, prefer the longer warm-up.  No comparator can be selected.  Parameters
and this protocol are frozen before the full experiment and cannot be changed
after observing results.  No Pass 5B scenarios are created or executed here.

