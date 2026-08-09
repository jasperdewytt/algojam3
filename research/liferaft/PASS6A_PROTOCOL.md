# Liferaft Research Pass 6A Protocol

Status: frozen before the valid result-producing Pass 6A calibration or simulator experiments.

Revision note: the first orchestration attempt completed the null paths but
incorrectly required both gates to pass before continuing. That decision was
invalidated because the protocol permits evaluation when at least one gate
passes. A second audit then found that the synthetic calibration harness used
an AUD 5,000,000 price, so its portfolio-headroom guard blocked every live
action and made synthetic real P&L uninformative. Those results were
invalidated. The calibration price level is now frozen at AUD 100,000, with
the exogenous movement supplied in `previous_price_change` and no floor
clipping; strategy logic, constants, gates, paths, and statistical tests are
otherwise unchanged. Null, development, and validation checks are rerun after
this correction.

This is a development-only architecture test. Existing Pass 3, Pass 4, and
Pass 5 results are consumed evidence and are not tuning targets. Pass 6A does
not modify the supplied simulator, production strategy, prior locked source, or
any final-suite artifact.

## 1. Hypothesis and research basis

The candidate is a small causal Fixed-Share ensemble of public-history
sequence predictors, surrounded by statistical participation gates. The
minority-game papers motivate treating participation and crowding as part of
the decision problem, not as ordinary 50% classification. Herbster and Warmuth
motivate share-based expert tracking that retains recoverable mass after an
expert becomes temporarily poor. Context-tree weighting motivates a genuine
mixture over suffix contexts rather than selecting a Markov order. The
anytime-valid inference papers motivate the e-process gate and Ville-valid
repeated inspection. The online-abstention paper motivates abstaining while
evidence or economic edge is insufficient. None of these sources proves that
this synthetic strategy is profitable.

The economic outcome bit is 1 for a public positive movement (short majority)
and 0 for a public negative movement (long majority). Unknown, reset, genuine
zero, and floor-clipped observations are context breaks, not binary outcomes.
The live strategy never receives the hidden majority, vote margin, pivotality,
or simulator diagnostics.

## 2. Frozen constants and economics

| quantity | frozen value |
|---|---:|
| long-majority movement | AUD -5,000 |
| short-majority movement | AUD +8,000 |
| economic neutral q* = P(short majority) | 5/13 = 0.3846153846 |
| reward normalisation bound | AUD 8,000 |
| live horizon T | 365 scoreable observations |
| expert count K | 7 |
| Fixed-Share eta | sqrt(2 log(7) / 365) |
| Fixed-Share share rate | 2 / (365 - 1) |
| primary pivotal probability | 10% |
| pivotal haircut | AUD 1,300 = 0.10 x AUD 13,000 |
| minimum residual economic edge | AUD 1,000 |
| checkpoint block size | 20 scoreable observations |
| possible full checkpoints | floor(365 / 20) = 18 |
| global one-sided error level | 5% |
| per-gate error allocation | 2.5% each |
| checkpoint alpha | 0.025 / 18 |
| anytime e-process bet fraction | 0.5 on reward / AUD 8,000 |
| loss stop | sticky cumulative loss at AUD 50,000 |
| trailing drawdown stop | sticky high-water drawdown at AUD 50,000 |
| portfolio headroom reserve | AUD 10,000 |
| gross portfolio budget | AUD 600,000 |

The preferred position for forecast q is +1 when q > 5/13, -1 when q < 5/13,
and zero at equality. A model without a valid context abstains. For position
p and realised public change d, counterfactual dollar reward is p*d. Flat's
reward is exactly zero. Rewards used in Fixed-Share are `p*d / 8000` in
[-1, 1]. Actual strategy P&L always uses the actual public price change,
including floor clipping.

Fixed-Share first forms the exponential posterior

`v_i = w_i exp(eta * reward_i / 8000)` and `q_i = v_i / sum_j v_j`,

then shares it with

`w'_i = (1 - share_rate) q_i + share_rate / K`.

Weights are normalised after the share step. They are updated only after the
corresponding outcome becomes visible and never on an unscoreable observation.
The additive share term keeps every expert recoverably positive and handles
regime change without an outcome-dependent reset.

The master forecast is the weighted average of expert short-majority
probabilities. Flat forecasts q* and proposes zero, so its weight reduces
conviction without adding a 50% long bias. The master position is the economic
preferred sign of the aggregate q and its predicted edge is the absolute
price-taking expected dollar reward for that position.

## 3. Exact expert set

The set is exactly:

1. `flat`: always proposes zero and has zero reward.
2. `order_zero_frequency`: Beta(1,1) estimate using only prior scoreable
   binary movements; context breaks are skipped.
3. `markov_order_1`: add-one conditional estimate; transitions never cross a
   reset, zero, unknown, or clipped break.
4. `markov_order_2`: the same rules at order two, with lower-order backoff only
   through valid within-segment counts.
5. `context_tree_weighting`: binary CTW with Krichevsky-Trofimov node
   estimates and frozen maximum suffix depth 6. Every observed bit updates the
   root and each suffix node. Each node mixes its KT probability with the
   product of its child mixtures; prediction uses the normalized ratio of
   hypothetical sequence probabilities. A context break resets the CTW tree.
6. `persistence`: repeats the most recent scoreable binary outcome in the
   current contiguous context; no prediction without context.
7. `reversal`: predicts the opposite of the most recent scoreable binary
   outcome in the current contiguous context; no prediction without context.

Every forecast is formed before the next outcome. The implementation records
each forecast, proposed position, current weight, master proposal, and edge in
its audit timeline.

## 4. Participation gates

### Candidate A: fixed checkpoint / block gate

The master is paper-traded during the first 20 scoreable observations. At each
of the 18 predeclared boundaries, the just-completed non-overlapping block is
tested. For block mean `m` and n=20 bounded rewards in [-8000,8000], the
one-sided Hoeffding lower bound is

`LCB = m - 8000 * sqrt(2 log(1 / (0.025/18)) / n)`.

The block passes only when `LCB > 0`. A pass authorises the following block and
a failure leaves the following block flat. The action that earned the
qualifying observation cannot be changed retroactively. Shadow forecasts and
rewards continue while the real position is flat. No block length, threshold,
or checkpoint subset is searched.

### Candidate B: anytime-valid gate

For each scoreable prequential master reward r, define x=r/8000 and update the
nonnegative e-process

`E_t = E_(t-1) * (1 + 0.5*x_t)`, with E_0=1.

Under the economic-null conditional mean assumption `E[x_t | public history]
<= 0`, each factor has conditional expectation at most one and is nonnegative.
Ville's inequality gives `P(sup_t E_t >= 1/0.025) <= 0.025`; the threshold is
40. No rolling t-test, repeated confidence interval, Sharpe threshold,
cumulative-P&L boundary, or outcome-dependent reset is used. Daily inspection
is valid through the e-process construction. A crossing authorises trading on
future decisions only.

For both gates, statistical authorisation is necessary but not sufficient.
The current master edge must satisfy

`predicted price-taking edge - 1300 >= 1000`.

The gate is applied to the frozen aggregate master, not to the best expert.

## 5. Risk and portfolio controls

- Liferaft remains flat before the marked voting start in both
  `observe_and_ignore_actions` and `fully_inactive` modes and does not learn
  from inactive Year 1 outcomes.
- Actual P&L is booked against the effective prior position and current public
  change. A one-movement adverse loss or drawdown stop can overshoot AUD
  50,000; the observed overshoot is recorded.
- Loss and trailing-drawdown stops are sticky. Exact price-floor observations
  flatten. A zero, unknown, reset, or clipped interval is a one-decision
  safety pause and a context break.
- Before a nonzero action, `other exposure + abs(action)*price + 10,000 <=
  600,000` is required. A callable exposure provider is cached and evaluated at
  most once per focal day, including duplicate-day calls.
- Positions are integer values in {-1,0,+1}. Invalid positions and budget
  breaches are not expected in safe configurations; tests assert zero.
- Duplicate calls for a processed day return the cached action without scoring,
  updating, or re-evaluating exposure.

## 6. Frozen scenarios and metrics

Null calibration runs at least 10,000 deterministic SHA-256-seeded exogenous
non-pivotal paths with P(+8000)=5/13 and P(-5000)=8/13. The focal action does
not affect these paths. It reports per-gate false activation, combined
either-gate activation, activation days/delays, real and shadow P&L, never
activation, and conservative 99% Hoeffding upper uncertainty for each rate.
For this gate-only calibration, the synthetic observation price is held at
AUD 100,000 so a permitted unit position is feasible under the AUD 600,000
budget and the AUD 20,000 floor cannot clip the path; the supplied public
movement remains the realised P&L increment. This isolates statistical gate
behaviour from an artificial price-level/headroom failure.
Power diagnostics are predeclared at IID q=0.50, IID q=0.25, a stationary
zero-unconditional-edge Markov process with P(short|long)=0.25 and
P(short|short)=0.60, and a q=0.25-to-q=0.50 regime switch. They are not tuning
inputs.

Only if at least one gate passes its per-gate null screen and the combined
either-gate upper bound remains within the global 5% design are the existing
nine development scenarios and existing 480 consumed validation scenarios run,
at exposures AUD 0, 150,000, 300,000, and 450,000, with both inactive execution
modes where present. Candidates are:

- new `fixed_checkpoint_fixed_share`;
- new `anytime_valid_fixed_share`;
- diagnostic ungated Fixed-Share master;
- flat;
- existing `burnin1_markov`;
- existing `risk50_burnin1_markov`;
- existing `shadow8_markov`.

Run-level records report mean/median/lower quartile/worst P&L, mean and maximum
drawdown, beat-flat rate, active days, turnover, activation/reactivation and
never-activation rates, stop and overshoot rates, headroom/rejection/budget
counts, pivotal/non-pivotal P&L, family/mode/exposure sensitivity, final expert
weights, regret against the best static expert and best sequence with at most
two hindsight switches, CTW versus Markov-2, and paper-to-live transfer.
Hidden pivotal/family/population labels are reporting-only diagnostics.

Pivotal-haircut sensitivity is a separate, non-selection diagnostic on the
same validation scenarios and exposures for both new candidates at frozen
assumptions 0%, 5%, 10%, and 20%. The primary candidate remains the 10%
setting; the sensitivity results cannot alter eligibility or candidate
priority and are written to `PASS6A_SENSITIVITY.json` by the parent process.

Paper transfer is explicitly split into shadow reward through the first
qualification decision, shadow reward strictly afterwards, actual P&L, and
actual-minus-shadow gap. Pivotal and non-pivotal transfer gaps are shown
separately.

## 7. Frozen eligibility and candidate priority

The primary screen is the consumed validation evaluation pooled across its
four exposures and both execution modes. A challenger must satisfy every
criterion below:

### Statistical validity

- its relevant gate's conservative 99% null upper bound is at most 2.5%;
- the combined either-gate conservative 99% upper bound is at most 5%;
- no look-ahead, retroactive qualification, invalid repeated testing, or
  outcome-dependent reset is present.

### Safety

- worst run >= -AUD 60,000;
- maximum drawdown <= AUD 75,000;
- zero focal budget breaches;
- zero focal rejected actions;
- mean pivotal P&L >= -AUD 25,000.

### Performance

- mean P&L > 0;
- median P&L >= 0;
- beat flat in at least 55% of runs;
- positive mean in at least 6 of the 12 principal validation families;
- either mean P&L retains at least 80% of the existing `risk50_burnin1_markov`
  positive aggregate mean, or the predeclared material trade-off holds:
  median is at least AUD 10,000 above the wrapper median and maximum drawdown
  is at most 80% of the wrapper maximum drawdown.

Candidate priority is deterministic and not mean-ranked: (1) anytime-valid if
fully valid and eligible, (2) fixed-checkpoint if eligible, (3) no challenger.
A passing challenger would be frozen for a separately authorised one-shot Pass
6B; no Pass 6B or blind/final suite is created here.

## 8. Execution and reproducibility commands

Planned commands from the repository root:

```text
python -m research.liferaft.pass6_experiments --null --workers auto
python -m research.liferaft.pass6_experiments --development --workers auto
python -m research.liferaft.pass6_experiments --validation --workers auto
```

The runner also accepts `--workers N`, `--workers auto`, `--serial`, and
`--quick`. Quick output is a mechanical smoke test and is never merged into
Pass 6A results. Before the full run, a representative development subset is
run with one worker and with `min(4, available CPUs)`; ordered result digests
must be bit-identical. Process batches are top-level picklable functions,
Windows-safe under the main guard, and reduced in deterministic task-key order.
The parent alone writes JSON/Markdown artifacts. No NumPy/BLAS worker pools are
used.

## 9. Permitted and prohibited inputs

Permitted live inputs are only `AgentObservation` public price/timing fields,
the focal held position, known public competition constants, the callable
other-exposure provider, and the strategy's own causal state. Experiment-only
hidden majority, vote counts/margins, pivotality, floor-clipping engine flags,
family, population size, execution mode, and scenario labels are used only in
post-run attribution.

The following remain quarantined and are not opened, parsed, imported,
executed, recreated, renamed, overwritten, or used: `pass4_final.py`, all
listed Pass 4 final results/decision/receipt artifacts, `PASS3_FINAL_REPORT.md`,
`final_scenarios()`, anything invoked by `--final`, and all locked final
scenario definitions/results. Production files and supplied competition files
are not modified.
