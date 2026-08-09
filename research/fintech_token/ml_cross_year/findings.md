# Fintech Token cross-year ML study

## Decision

Do not replace the frozen EWMA ensemble with an ML model. The nested cross-year study finds useful out-of-sample gains on two held-out years, but the selected model loses to EWMA on the third held-out year, loses its worst-year P&L comparison, and does not survive a one-day execution delay in every year. The primary promotion requirement is therefore not met.

Recommendation: retain the frozen EWMA ensemble in `trader_interface/algorithm.py`. No production all-years fit was created and no production or shared research file was modified.

## Data and isolation

The study used exactly these three files as separate sequences:

| Label | File | Prices | Changes | Local day labels |
|---|---|---:|---:|---|
| A | `trader_interface/data/Fintech Token_price_history.csv` | 365 | 364 | 0--364 |
| B | `trader_interface/2024_data_DONOTUSENORMALLY/Fintech Token_price_history.csv` | 365 | 364 | 0--364 |
| C | `trader_interface/2024_data_DONOTUSENORMALLY/full_data/Fintech Token_price_history.csv` | 365 | 364 | 365--729 in the CSV, reset to local 0--364 |

No year identifiers, full-year means, full-year standard deviations, day-number features, retrospective ranking files, `tmp/compare_fintech_years.py`, or `tmp/fintech_three_year_grid.csv` were used. Volatility, rolling continuation history, model state, and positions are rebuilt separately for each yearly file.

The simulator alignment is `position[j] * d[j]`, where a position in slot `j` is selected after observing `P[j]` and the latest change `d[j-1]`. Equivalently, a feature row `i` observes through `d[i]`, predicts `d[i+1]`, and earns `position[i] * d[i+1]`. The first 30 observed changes in every year use simple reversal at +/-100; zero or invalid latest changes are flat.

## Causal target and features

The direct target is:

```text
z[i+1] = sign(d[i]) * d[i+1]
```

The regression target used for scale stability is:

```text
z[i+1] / max(sigma[i], EPS)
```

where `sigma[i]` is the causal lambda=0.90 EWMA volatility through `d[i]`. Logistic labels use `z[i+1] > 0`. Weighted logistic observations use a clipped version of the raw `abs(z)`; its clip quantile and cap are computed from the fitting rows only.

Every feature's final observable timestamp is listed below. `c[j]` means the completed outcome `sign(d[j-1]) * d[j]`; it is available only after `d[j]` is observed.

| Feature | Final observable data |
|---|---|
| Latest absolute change / causal volatility | `d[i]` and EWMA volatility through `d[i]` |
| Current EWMA volatility percentile | current `d[i]`; percentile cutoff uses EWMA values through `d[i-1]` only |
| Fast/slow volatility ratio | causal lambda=.85 and lambda=.95 EWMAs through `d[i]` |
| Change in log volatility | `d[i-1]` and `d[i]` |
| Frozen EWMA ensemble vote | observations through `d[i]` |
| Number of consecutive days in current EWMA state | states calculated through `d[i]` |
| Rolling 5-day continuation | completed `c[j]` through `j=i`, last five values, expressed in volatility units |
| Rolling 20-day continuation | completed `c[j]` through `j=i`, last 20 values, expressed in volatility units |
| Rolling 20-day positive fraction | signs of completed `c[j]` through `j=i`, last 20 values |
| Causal volatility threshold gap | current EWMA through `d[i]` versus threshold from earlier EWMA values |
| Frozen EWMA member agreement | three member states through `d[i]` |
| Raw target | `d[i+1]` is used only as a training outcome, never as a live feature |
| Normalised target | same `d[i+1]` training outcome divided by volatility through `d[i]` |

The reusable causal implementation is in [`ml_models.py`](ml_models.py); validation and robustness logic is in [`ml_validation.py`](ml_validation.py).

## Candidate family and nested selection

The candidate family was frozen before reading any outer-test results:

- Ridge regression on the normalised continuation target, `alpha` in `{0.01, 0.1, 1, 10, 100}`.
- Equal-weight L2 logistic regression for `P(z > 0)`, using the fixed probability decision boundary 0.5, with the same alpha grid.
- Clipped-absolute-target weighted logistic regression, with the same alpha grid and training-only clip quantile in `{0.75, 0.90}`.
- One depth-one regression stump on the normalised target, with minimum leaf size in `{60, 90}`.
- Each base configuration was paired with a confidence fallback threshold in `{0, 0.25, 0.50}`. For regression, confidence is `min(abs(prediction), 1)` in normalised target units; for logistic, it is `abs(2p-1)`. A low-confidence prediction falls back to reversal. This is not a changed logistic probability boundary.

This is 66 configurations: 15 ridge, 15 equal logistic, 30 weighted logistic, and 6 tree-stump configurations. The optional two-state HMM was not included: the user made it secondary, and the prior short-sample MS-AR/HMM work is not used as model-selection evidence here.

For each outer fold, each candidate was fit on one training year and validated on the other, then the direction was reversed. Selection used only the two inner validation rows:

1. Maximise the worst inner-year incremental P&L over frozen EWMA.
2. Among candidates within AUD 1,000 of that score, choose lower declared complexity.
3. Break any remaining tie with lower mean drawdown depth and then higher mean incremental P&L.
4. Refit the selected configuration on both training years with training-only feature standardisation.
5. Evaluate once on the untouched outer year.

| Outer test | Training years | Selected configuration | Worst inner increment vs EWMA | Mean inner increment vs EWMA | Mean inner drawdown depth |
|---|---|---|---:|---:|---:|
| A | B + C | Ridge, alpha=100, confidence=0 | +$61,110 | +$73,118 | $22,980 |
| B | A + C | Weighted logistic, alpha=100, abs(z) clip q=.90, confidence=0 | -$32,196 | -$10,678 | $21,214 |
| C | A + B | Equal logistic, alpha=100, confidence=0 | +$6,724 | +$14,760 | $19,469 |

The negative inner score for the B fold is not hidden or retuned after seeing B; it is the best worst-year score available from the declared family.

## Independent benchmark reproduction

The frozen benchmark is exactly:

```text
(lambda=.85, percentile=.75, warm-up=30)
(lambda=.90, percentile=.80, warm-up=30)
(lambda=.95, percentile=.85, warm-up=30)
```

The new implementation independently reproduces the known Dataset A values: reversal $75,369, frozen EWMA $157,409, and +$82,040 incremental. It also reports B and C for the outer comparisons.

| Year | Simple reversal P&L | Frozen EWMA P&L | EWMA increment vs reversal | EWMA max drawdown | EWMA max capital |
|---|---:|---:|---:|---:|---:|
| A | $75,369 | $157,409 | +$82,040 | -$8,996 | $85,434 |
| B | $65,050 | $202,594 | +$137,544 | -$33,749 | $204,230 |
| C | $83,396 | $135,354 | +$51,958 | -$32,735 | $173,989 |

## Outer-test results

All three outer test paths use a model fitted without any observations from that test path.

| Test year | Selected ML P&L | Reversal P&L | EWMA P&L | Increment vs reversal | Increment vs EWMA | Max drawdown | Hit rate | Active days | Turnover units | Max capital |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | $121,311 | $75,369 | $157,409 | +$45,942 | -$36,098 | -$14,074 | 61.2% | 363 | 40,700 | $85,434 |
| B | $284,654 | $65,050 | $202,594 | +$219,604 | +$82,060 | -$21,828 | 65.6% | 363 | 43,100 | $204,230 |
| C | $159,882 | $83,396 | $135,354 | +$76,486 | +$24,528 | -$32,735 | 59.2% | 363 | 41,900 | $173,989 |

Quarterly P&L for the selected ML policy was:

| Test year | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| A | $38,100 | $31,106 | $36,108 | $15,997 |
| B | $91,860 | $76,185 | $80,452 | $36,157 |
| C | $37,960 | $47,706 | $79,786 | -$5,570 |

The selected ML decisions were momentum/reversal as follows: A 77/286, B 66/297, C 89/274, with no flat decisions on these finite, nonzero paths. Candidate P&L during frozen-EWMA volatile/calm states was A $35,964/$85,347, B $93,622/$191,032, and C $50,507/$109,375. The corresponding candidate-minus-EWMA increments in volatile/calm states were A -$5,056/-$31,042, B +$24,850/+ $57,210, and C +$24,528/$0.

Aggregate outer-test result:

| Aggregate | Selected ML | Frozen EWMA | Difference |
|---|---:|---:|---:|
| Total P&L across three held-out years | $565,847 | $495,357 | +$70,490 ML |
| Total P&L versus reversal | $342,032 incremental | $271,542 incremental | — |
| Mean ML increment vs EWMA | +$23,497 | — | — |
| Worst outer-year increment vs EWMA | -$36,098 | — | — |
| Outer years beating reversal | 3/3 | — | — |
| Outer years beating EWMA | 2/3 | — | — |

The combined total is attractive but is not sufficient: the worst selected ML P&L is $121,311 versus $135,354 for EWMA, so worst-year P&L is $14,043 worse.

## Confidence-filter report

The inner selection chose confidence threshold zero in all folds. Fixed threshold variants of the already selected base models were also evaluated only as a report, not used to revise outer selection:

| Test | Family | Confidence 0 | Confidence .25 | Confidence .50 |
|---|---|---:|---:|---:|
| A | Ridge | -$36,098 | -$73,078 | -$82,040 |
| B | Weighted logistic | +$82,060 | -$57,192 | -$137,544 |
| C | Equal logistic | +$24,528 | -$2,186 | -$51,958 |

Values are incremental P&L versus EWMA. Confidence filtering does not rescue the A fold; a .50 threshold effectively falls back to reversal on these paths.

## Delay, episode, and move exclusions

One-day delayed-execution incremental P&L versus one-day-delayed EWMA was A -$17,914, B +$5,702, and C +$10,894. The delay condition therefore also fails the all-years promotion rule.

The largest frozen-EWMA volatile episode contributed +$8,168 to A's ML-minus-EWMA difference, +$18,416 in B, and +$10,316 in C. Excluding the largest positive volatile episode leaves increments of -$44,266, +$63,644, and +$14,212 respectively. The B and C improvements are not arithmetically explained by one episode, although the sample still contains only three independent yearly paths.

After removing the largest absolute realised moves, the ML-minus-EWMA increments were:

| Test | Remove 1 | Remove 3 | Remove 5 | Remove 10 |
|---|---:|---:|---:|---:|
| A | -$36,098 | -$36,792 | -$36,792 | -$30,714 |
| B | +$82,060 | +$66,270 | +$66,270 | +$66,270 |
| C | +$24,528 | +$24,528 | +$24,528 | +$24,528 |

## Paired moving-block bootstrap

This bootstrap resamples paired daily candidate-minus-EWMA P&L after each outer model is frozen; it does not refit on the outer test data. There are 1,000 repetitions per block length and per outer year.

| Test | Block | Mean increment | P(increment > 0) | 2.5%--97.5% interval |
|---|---:|---:|---:|---:|
| A | 5 | -$37,076 | .050 | -$86,742 to $5,497 |
| A | 10 | -$36,786 | .046 | -$82,182 to $6,397 |
| A | 20 | -$39,118 | .053 | -$93,324 to $7,840 |
| A | 40 | -$44,060 | .035 | -$103,126 to $1,557 |
| B | 5 | +$82,330 | .999 | $18,998 to $156,834 |
| B | 10 | +$86,130 | .998 | $20,281 to $163,012 |
| B | 20 | +$90,595 | .999 | $26,935 to $168,253 |
| B | 40 | +$98,826 | 1.000 | $33,012 to $169,700 |
| C | 5 | +$25,114 | .986 | $1,440 to $55,858 |
| C | 10 | +$24,796 | .972 | -$172 to $52,828 |
| C | 20 | +$25,592 | .977 | $252 to $62,380 |
| C | 40 | +$27,006 | .981 | $909 to $61,775 |

The bootstrap confirms that the A loss is not a resampling accident and that B/C gains are serially persistent on their frozen test paths. It does not provide three independent replications of the model-selection procedure.

## Prediction and calibration diagnostics

| Test | Model sign accuracy | Actual positive | Predicted positive | TP / TN / FP / FN | Score-target correlation | Brier |
|---|---:|---:|---:|---|---:|---:|
| A ridge | 61.0% | 143 | 77 | 45 / 158 / 32 / 98 | .195 | — |
| B weighted logistic | 65.8% | 144 | 66 | 48 / 171 / 18 / 96 | .275 | .230 |
| C equal logistic | 58.0% | 143 | 89 | 46 / 147 / 43 / 97 | .189 | .240 |

The selected logistic probabilities were concentrated in the .2--.8 range. For B, the `.2--.4`, `.4--.6`, and `.6--.8` bins had counts 219, 85, and 29 with observed positive rates .352, .565, and .655. For C the corresponding counts were 171, 120, and 42 with observed rates .345, .508, and .548. These are useful diagnostics, not independent calibration validation.

## Coefficient and decision stability

The final outer fits use standardised features; the standardisation means and scales are saved in `results/selected_model_summaries.json`. A is ridge, B is weighted logistic, and C is equal logistic, so coefficient magnitudes are not directly comparable across families. Sign comparison is only a directional stability diagnostic.

| Feature | A | B | C | Sign agreement |
|---|---:|---:|---:|---:|
| Latest abs change / vol | -0.0319 | -0.0488 | -0.0795 | 3/3 negative |
| Current vol percentile | +0.00455 | +0.0567 | +0.0157 | 3/3 positive |
| Fast/slow vol ratio | +0.1242 | +0.1208 | +0.1029 | 3/3 positive |
| Change in log vol | +0.00477 | +0.0219 | +0.0221 | 3/3 positive |
| Frozen EWMA vote | -0.00069 | +0.1190 | +0.1978 | mixed |
| EWMA state run length | -0.00736 | +0.0201 | +0.0778 | mixed |
| Rolling 5 continuation | +0.1905 | +0.2325 | +0.1441 | 3/3 positive |
| Rolling 20 continuation | +0.00352 | -0.0213 | +0.00981 | mixed |
| Rolling 20 positive fraction | -0.0197 | +0.0181 | -0.00884 | mixed |
| Causal threshold gap | -0.00156 | +0.0477 | +0.0811 | mixed |
| EWMA member agreement | -0.0128 | +0.0198 | +0.0232 | mixed |

Five of eleven feature signs are stable and six are mixed; the intercept is separately stable negative. The stable feature signs are only a weak positive signal because the selected family changes by fold and the fitted paths are correlated. No tree stump was selected. The alpha choice was stable at the upper end of the declared grid (`alpha=100` in every fold), but the model family and weighted/unweighted logistic choice were not stable.

## Leakage and boundary checks

- Every one of the 66 candidates was attempted in each outer fold; the candidate accounting table has 198 rows and zero fit errors.
- Feature standardisation was fit on the fitting rows only. The outer test year never supplied means, scales, thresholds, clipping caps, hyperparameters, early stopping, or family selection.
- All 18 future-data perturbation checks passed: changing observations after cuts 60, 90, 120, 180, 240, and 300 did not change positions through the cut in any outer year.
- All 12 prefix-feature checks passed: recomputing features from truncated prefixes produced identical earlier feature rows.
- The year-boundary audit shows 365 prices, 364 changes, first valid feature row 30, and 333 valid post-warmup training rows for every year. No state or rolling history is carried between files.
- The target's `d[i+1]` timestamp is used only in training labels and post-hoc P&L/diagnostics; it is never available to the decision function.
- The shuffled-target placebo uses the selected configuration and the same training pipeline with 20 target permutations per outer fold. Mean incremental P&L versus EWMA was -$102,647 for A, -$124,311 for B, and -$52,728 for C; positive shares were 0%, 0%, and 5%. This is a diagnostic rather than a formal p-value.
- The study does not treat 999 post-warmup rows as independent observations. The independent units are still only three yearly paths, with strong within-year dependence.

## Deliverables, execution, and concerns

Created in this directory:

- [`fintech_ml_cross_year.ipynb`](fintech_ml_cross_year.ipynb), executed end to end with zero error outputs.
- [`ml_models.py`](ml_models.py) and [`ml_validation.py`](ml_validation.py), containing reusable causal features, models, nested selection, metrics, bootstrap, perturbation, and prefix audits.
- This [`findings.md`](findings.md).
- Compact CSV/JSON outputs under [`results/`](results/) and figures under [`figures/`](figures/), including benchmark reproduction, selected configurations, candidate accounting, quarterly P&L, delay/confidence checks, bootstrap, placebo, coefficients, and audits.

The required unchanged production simulation was also run from `trader_interface/` with the existing `algorithm.py`: exit code 0, total P&L $555,796.50, zero budget violations, and no invalid-position or runtime errors. The study itself used the repository virtual environment's NumPy, Pandas, Matplotlib, and Jupyter. Scikit-learn was unavailable but was not required: the small ridge, logistic, and stump implementations use NumPy only. Jupyter emitted only non-fatal Windows/runtime warnings; the executed notebook contains no error cells.

The unresolved concern is generalisation. B and C provide encouraging out-of-sample improvements, but A is a decisive loss versus EWMA, the selected family is not stable, and only three yearly paths are available. The evidence validates the frozen EWMA mechanism more strongly than it justifies an ML replacement.
