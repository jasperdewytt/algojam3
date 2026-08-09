# Liferaft Pass 6A report

> Development-only research. Existing Pass 3/4/5 evidence is consumed context; this is not a blind or final suite.

Audit note: an initial result set was invalidated after discovering that the
synthetic gate harness used a AUD 5,000,000 price and therefore blocked every
live action on headroom. The null, development, and validation phases below
are the complete rerun with the frozen AUD 100,000 calibration price level;
the strategy, paths, gates, and constants were unchanged.

## Mathematical and statistical validity

The master uses seven causal experts: flat, Beta(1,1) order-zero frequency, add-one Markov orders 1 and 2, binary CTW with KT nodes at maximum depth 6, persistence, and reversal. Each expert proposes the economically preferred position relative to q*=5/13. Rewards are dollar rewards divided by the frozen AUD 8,000 bound before the exponential Fixed-Share update.

Fixed-Share constants: eta=0.10325948559181693, share_rate=0.005494505494505495, block size=20, per-gate alpha=2.5%, checkpoint alpha=0.001388888888888889, primary pivotal haircut=AUD 1,300, minimum residual edge=AUD 1,000.

The fixed-checkpoint gate tests each complete non-overlapping block with a one-sided Hoeffding lower bound and authorizes only the next block. The anytime gate is a nonnegative e-process with factor 1+0.5*(reward/8000); Ville's inequality covers repeated daily inspection without an outcome-dependent reset.

Economic-null calibration: 10000 non-pivotal IID paths at q=5/13. The predeclared calibration result is **PASS**; combined either-gate activation rate=2.160%, conservative 99% upper bound=3.677%.

| gate | activations | rate | conservative 99% upper | mean real P&L | mean shadow P&L |
|---|---:|---:|---:|---:|---:|
| `fixed_checkpoint_fixed_share` | 0 | 0.000% | 1.517% | AUD 0 | AUD 323 |
| `anytime_valid_fixed_share` | 216 | 2.160% | 3.677% | AUD 51 | AUD 323 |

The calibration is a statistical-gate test only: public paths are exogenous, focal actions cannot change them, and hidden majority/pivotal labels never enter either strategy.

## Synthetic power diagnostics

These were predeclared diagnostics, not tuning inputs.

| process | gate | activation rate | median activation delay | mean real P&L |
|---|---|---:|---:|---:|
| `conditional_zero_edge` | `fixed_checkpoint_fixed_share` | 0.8% | 240.0 | AUD 229 |
| `conditional_zero_edge` | `anytime_valid_fixed_share` | 98.5% | 80.0 | AUD 455,698 |
| `iid_q_025` | `fixed_checkpoint_fixed_share` | 0.0% | 0.0 | AUD 0 |
| `iid_q_025` | `anytime_valid_fixed_share` | 62.1% | 150.0 | AUD 31,999 |
| `iid_q_050` | `fixed_checkpoint_fixed_share` | 0.4% | 300.0 | AUD 10 |
| `iid_q_050` | `anytime_valid_fixed_share` | 57.6% | 108.0 | AUD 24,698 |
| `regime_switch` | `fixed_checkpoint_fixed_share` | 0.0% | 0.0 | AUD 0 |
| `regime_switch` | `anytime_valid_fixed_share` | 38.1% | 86.0 | AUD 26,922 |

## Simulator performance and transfer audit

Development rows: 252. Consumed validation rows: 13440. Validation scenarios were the existing 480 cases with both inactive execution modes and exposures AUD 0/150,000/300,000/450,000.

Validation aggregate summaries for the two new candidates:

| candidate | mean | median | lower quartile | worst | mean DD | max DD | beat flat | active days | turnover | pivotal mean | non-pivotal mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_checkpoint_fixed_share` | AUD 52,131 | AUD 0 | AUD 0 | AUD -50,000 | AUD 338 | AUD 50,000 | 19.6% | 9.3 | 13.3 | AUD -2,115 | AUD 54,246 |
| `anytime_valid_fixed_share` | AUD 102,414 | AUD 0 | AUD 0 | AUD -54,000 | AUD 4,151 | AUD 54,000 | 41.6% | 29.2 | 39.8 | AUD -15,125 | AUD 117,539 |

Paper-to-live transfer for the new candidates is reported from the same run records: `shadow_pre_qualification_pnl` is the master paper reward through the qualification decision, `shadow_post_qualification_pnl` starts strictly after that decision, and `shadow_to_actual_gap` is actual realised Liferaft P&L minus scoreable shadow P&L. Pivotal/non-pivotal partitions use engine-only diagnostics after the run; the live strategy never sees those labels.

| candidate | shadow pre-qualification | shadow post-qualification | actual-shadow gap | non-pivotal actual | pivotal actual | CTW - Markov-2 |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_checkpoint_fixed_share` | AUD 316,062 | AUD 170,185 | AUD -434,117 | AUD 54,246 | AUD -2,115 | AUD -104,258 |
| `anytime_valid_fixed_share` | AUD 121,544 | AUD 339,280 | AUD -358,410 | AUD 117,539 | AUD -15,125 | AUD -103,532 |

Transfer stratified by the experiment-only pivotal label shows that paper
performance did not transfer cleanly even when non-pivotal. The live strategy
also lost on pivotal intervals, but the pivotal shadow gap was much smaller;
the principal failure was shadow reward being unavailable or filtered by
edge, headroom, stops, and participation timing.

| candidate | partition | shadow P&L | actual P&L | actual - shadow |
|---|---|---:|---:|---:|
| `fixed_checkpoint_fixed_share` | pivotal | AUD -1,690 | AUD -2,115 | AUD -425 |
| `fixed_checkpoint_fixed_share` | non-pivotal | AUD 487,938 | AUD 54,246 | AUD -433,692 |
| `anytime_valid_fixed_share` | pivotal | AUD -14,417 | AUD -15,125 | AUD -708 |
| `anytime_valid_fixed_share` | non-pivotal | AUD 475,241 | AUD 117,539 | AUD -357,702 |

## Portfolio and expert diagnostics

Across validation runs, first authorization days ranged from day 385 to 728
for fixed-checkpoint (median 388 among activated runs) and day 375 to 719 for
anytime-valid (median 400 among activated runs); the complete activation-day
lists and reactivation events are retained at run level in the JSON.

| candidate | activation | never | reactivation | first activation median day | active days | turnover | loss stop | drawdown stop | mean/max overshoot | headroom gates | edge gates | unknown/floor gates | breaches/rejections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_checkpoint_fixed_share` | 30.4% | 69.6% | 13.2% | 388 | 9.3 | 13.3 | 0.1% | 0.1% | AUD 0 / AUD 0 | 11.9 | 4.9 | 54.8 / 0.0 | 0 / 0 |
| `anytime_valid_fixed_share` | 79.8% | 20.2% | 0.0% | 400 | 29.2 | 39.8 | 2.3% | 4.4% | AUD 59 / AUD 4,000 | 24.3 | 102.2 | 76.3 / 0.01 | 0 / 0 |

Validation-average final expert weights and regret diagnostics are shown
below. “Static regret” is best static expert reward minus master shadow
reward; the low-switch column uses a hindsight diagnostic with at most two
switches and was never an input to trading.

| candidate | flat | order-zero | Markov-1 | Markov-2 | CTW | persistence | reversal | static regret | <=2-switch regret | CTW - Markov-2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_checkpoint_fixed_share` | 4.2% | 25.2% | 21.3% | 21.5% | 11.6% | 7.5% | 8.7% | AUD -10,166 | AUD 49,790 | AUD -104,258 |
| `anytime_valid_fixed_share` | 4.4% | 26.7% | 20.9% | 20.8% | 11.1% | 7.4% | 8.7% | AUD -12,493 | AUD 50,645 | AUD -103,532 |

CTW retained a positive, recoverable weight, but its mean reward was below
Markov-2 by about AUD 104k over these validation paths. It did not add value
beyond Markov-2 in this pass.

| candidate | exposure | mean P&L | beat flat | active days | headroom gates |
|---|---:|---:|---:|---:|---:|
| `fixed_checkpoint_fixed_share` | AUD 0 | AUD 112,323 | 27.9% | 19.6 | 0.6 |
| `fixed_checkpoint_fixed_share` | AUD 150,000 | AUD 68,896 | 26.7% | 12.3 | 8.3 |
| `fixed_checkpoint_fixed_share` | AUD 300,000 | AUD 26,065 | 22.9% | 4.9 | 16.7 |
| `fixed_checkpoint_fixed_share` | AUD 450,000 | AUD 1,242 | 0.8% | 0.6 | 22.1 |
| `anytime_valid_fixed_share` | AUD 0 | AUD 191,908 | 50.0% | 50.9 | 0.9 |
| `anytime_valid_fixed_share` | AUD 150,000 | AUD 133,890 | 49.2% | 37.9 | 13.4 |
| `anytime_valid_fixed_share` | AUD 300,000 | AUD 73,306 | 49.0% | 22.5 | 30.8 |
| `anytime_valid_fixed_share` | AUD 450,000 | AUD 10,552 | 18.1% | 5.4 | 52.2 |

| candidate | inactive mode | mean P&L | beat flat | active days |
|---|---|---:|---:|---:|
| `fixed_checkpoint_fixed_share` | fully inactive | AUD 52,152 | 19.6% | 9.3 |
| `fixed_checkpoint_fixed_share` | observe-and-ignore | AUD 52,110 | 19.6% | 9.4 |
| `anytime_valid_fixed_share` | fully inactive | AUD 101,804 | 41.4% | 28.7 |
| `anytime_valid_fixed_share` | observe-and-ignore | AUD 103,024 | 41.8% | 29.6 |

Worst-run P&L by principal family is included to expose failure modes hidden
by aggregate means.

| family | fixed-checkpoint worst | anytime worst |
|---|---:|---:|
| persistent_long | AUD 0 | AUD -7,000 |
| persistent_short | AUD 0 | AUD 0 |
| balanced_random | AUD 0 | AUD -52,000 |
| short_biased_random | AUD 0 | AUD 0 |
| long_biased_random | AUD 0 | AUD -30,000 |
| periodic | AUD -42,000 | AUD -22,000 |
| reactive_mixture | AUD 0 | AUD -53,000 |
| regime_change | AUD 0 | AUD -50,000 |
| gradual_drift | AUD 0 | AUD -52,000 |
| startup_zero_history | AUD 0 | AUD -52,000 |
| history_rules | AUD 0 | AUD -54,000 |
| margin_mixture | AUD -50,000 | AUD -50,000 |

Pivotal-haircut sensitivity is a diagnostic only; 10% remains primary and
was not selected after seeing these results.

| candidate | pivotal assumption | mean P&L | median | worst | max DD | beat flat | pivotal mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_checkpoint_fixed_share` | 0% | AUD 51,926 | AUD 0 | AUD -51,000 | AUD 51,000 | 19.5% | AUD -2,269 |
| `fixed_checkpoint_fixed_share` | 5% | AUD 52,103 | AUD 0 | AUD -51,000 | AUD 51,000 | 19.5% | AUD -2,200 |
| `fixed_checkpoint_fixed_share` | 10% | AUD 52,131 | AUD 0 | AUD -50,000 | AUD 50,000 | 19.6% | AUD -2,115 |
| `fixed_checkpoint_fixed_share` | 20% | AUD 51,472 | AUD 0 | AUD -45,000 | AUD 45,000 | 19.4% | AUD -1,747 |
| `anytime_valid_fixed_share` | 0% | AUD 103,414 | AUD 0 | AUD -57,000 | AUD 57,000 | 44.8% | AUD -22,043 |
| `anytime_valid_fixed_share` | 5% | AUD 102,238 | AUD 0 | AUD -54,000 | AUD 54,000 | 41.4% | AUD -18,853 |
| `anytime_valid_fixed_share` | 10% | AUD 102,414 | AUD 0 | AUD -54,000 | AUD 54,000 | 41.6% | AUD -15,125 |
| `anytime_valid_fixed_share` | 20% | AUD 96,493 | AUD 0 | AUD -50,000 | AUD 52,000 | 32.6% | AUD -9,089 |

The full sensitivity rows are in `PASS6A_SENSITIVITY.json`.

## Family, exposure, and execution-mode sensitivity

The machine-readable results contain every run-level row and grouped family/exposure/mode summaries. The families deliberately cover persistent, reversing/reactive, regime-change, drift, startup, near-tie, floor, and budget failure modes; positive deterministic families are not treated as proof of generalisation.

### `fixed_checkpoint_fixed_share` family means

| family | exposure 0 mean | exposure 150k mean | exposure 300k mean | exposure 450k mean |
|---|---:|---:|---:|---:|
| `persistent_long` | AUD 0 | AUD 0 | AUD 0 | AUD 0 |
| `persistent_short` | AUD 250,300 | AUD 146,600 | AUD 30,250 | AUD 0 |
| `balanced_random` | AUD 0 | AUD 0 | AUD 0 | AUD 0 |
| `short_biased_random` | AUD 148,150 | AUD 84,450 | AUD 22,500 | AUD 0 |
| `long_biased_random` | AUD 0 | AUD 0 | AUD 0 | AUD 0 |
| `periodic` | AUD 54,150 | AUD 34,550 | AUD 20,000 | AUD 14,900 |
| `reactive_mixture` | AUD 3,325 | AUD 1,400 | AUD 1,175 | AUD 0 |
| `regime_change` | AUD 9,900 | AUD 1,100 | AUD 0 | AUD 0 |
| `gradual_drift` | AUD 0 | AUD 0 | AUD 0 | AUD 0 |
| `startup_zero_history` | AUD 0 | AUD 0 | AUD 0 | AUD 0 |
| `history_rules` | AUD 801,400 | AUD 515,050 | AUD 230,150 | AUD 0 |
| `margin_mixture` | AUD 80,650 | AUD 43,600 | AUD 8,700 | AUD 0 |

### `anytime_valid_fixed_share` family means

| family | exposure 0 mean | exposure 150k mean | exposure 300k mean | exposure 450k mean |
|---|---:|---:|---:|---:|
| `persistent_long` | AUD 1,600 | AUD 1,600 | AUD 1,600 | AUD 1,600 |
| `persistent_short` | AUD 368,050 | AUD 236,400 | AUD 103,350 | AUD 0 |
| `balanced_random` | AUD -14,600 | AUD -13,300 | AUD -5,300 | AUD 0 |
| `short_biased_random` | AUD 299,950 | AUD 186,250 | AUD 77,400 | AUD 0 |
| `long_biased_random` | AUD 9,250 | AUD 9,250 | AUD 9,250 | AUD 9,250 |
| `periodic` | AUD 232,500 | AUD 193,950 | AUD 148,700 | AUD 74,900 |
| `reactive_mixture` | AUD 73,025 | AUD 46,475 | AUD 21,600 | AUD 1,875 |
| `regime_change` | AUD 47,350 | AUD 26,850 | AUD 9,450 | AUD 0 |
| `gradual_drift` | AUD 22,900 | AUD 3,600 | AUD -3,550 | AUD -2,000 |
| `startup_zero_history` | AUD -15,325 | AUD -14,250 | AUD -7,925 | AUD -100 |
| `history_rules` | AUD 1,163,900 | AUD 861,700 | AUD 498,850 | AUD 41,200 |
| `margin_mixture` | AUD 114,300 | AUD 68,150 | AUD 26,250 | AUD -100 |

## Mechanical screen and recommendation

All eligibility criteria are conjunctive, with only the predeclared risk50-wrapper retention/trade-off alternative using OR. Candidate priority is anytime-valid first, fixed-checkpoint second, otherwise no challenger.

| candidate | null valid | safety/performance eligible | positive families | wrapper retention | recommendation status |
|---|---:|---:|---:|---:|---|
| `fixed_checkpoint_fixed_share` | True | False | 7/12 | 46.7% | failed |
| `anytime_valid_fixed_share` | False | False | 10/12 | 91.7% | failed |

Failure reasons were explicit: the fixed-checkpoint candidate passed null
calibration but failed beat-flat (19.6% versus 55%) and wrapper retention
(46.7% versus the required 80%); the anytime candidate failed its per-gate
null screen and beat-flat (41.6%), although its wrapper retention was 91.7%.
Both had positive mean, nonnegative median, at least six positive families,
safe worst-run/drawdown/pivotal results, and zero focal budget breaches or
rejected actions.

Baseline diagnostics were: flat mean AUD 0; ungated Fixed-Share mean AUD
116,357 with 57.9% beat-flat; burnin1_markov mean AUD 118,696 but 200,423
budget breaches/rejections; risk50_burnin1_markov mean AUD 111,659 with 59.1%
beat-flat and AUD 65,000 maximum drawdown; and shadow8_markov mean AUD 103,688
with AUD 93,000 maximum drawdown. These are diagnostics, not promotion
targets.

Selected challenger for a later blind Pass 6B: **none; all challengers failed**.

## Production recommendation

This pass does not modify the production algorithm. A positive synthetic mean is not sufficient: the decision is controlled by null calibration, transfer gaps, pivotal exposure, budget mechanics, and the frozen conjunctive screen. If no candidate is selected, retain the existing production strategy and do not promote a Pass 6 challenger.

Quarantine confirmation: no final scenario suite, final strategy module, final result, final decision, final receipt, or production file was imported, executed, created, modified, or used.
