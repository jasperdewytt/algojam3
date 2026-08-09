# Boat Party EWMA overlay audit V2

## Verdict

**Decision E: keep frozen Candidate D unchanged for submission.** The adaptive
EWMA overlay is retained as research evidence, not added to production. The
summer rule also remains fixed at AUD 45.

The teammate's approximately AUD 190,000 result is reproduced only when the
365-day signal is constructed from the next day's realized return. That is a
direct target-encoding leakage control, not an admissible Round 2 strategy. Its
fixed component earns AUD 185,880 with a 100% Round 1 fixed-signal hit rate.
Adding the post-update EWMA reconstruction raises that leakage-control result
to AUD 196,220 at alpha .65 and AUD 195,810 at alpha .90, but this does not
make the EWMA effect valid.

On the valid frozen Candidate-D neutral days, the causal post-update EWMA
adds AUD 9,100 at alpha 0.65 and AUD 7,240 at alpha 0.90. The corrected
prior-state version adds AUD 8,330 and AUD 8,630 respectively. Those gains are
in-sample Round 1 diagnostics. They are not independent validation and do not
meet the standard required for changing the submitted strategy: the simple
one-day reversal control earns AUD 9,600 on the same neutral-day overlay,
incremental gains are concentrated in a small number of days, the predictive
sample contains only 31 eligible neutral observations, and the sign placebo
removes the incremental edge.

The recommended production logic is therefore exactly the existing frozen
Candidate D:

* use the fixed Round 1 5/7/11 seasonal majority signal on days 0--321;
* use AUD 45 mean reversion on days 322--363;
* use zero position on day 364;
* use no adaptive neutral-day overlay, calendar warp, RLS, OU model, Fourier
  model, or absolute seasonal-level forecast.

All findings below are labelled as in-sample Round 1 diagnostics or
generator-conditioned stress tests. Synthetic results are not confidence
intervals and are not a second year of evidence.

## Scope and provenance

The production Candidate-D semester signal was extracted read-only from
`trader_interface/algorithm.py` and was not regenerated, optimized, or
modified. It is treated as a complete Round 1 seasonal prior that would be
frozen before Round 2. The V2 implementation is
[`ewma_overlay_audit_v2.py`](D:/Documents/Algojam/research/boat_party/ewma_overlay_audit_v2.py).

The repository does not contain the teammate's original 365-character source
string or a separate submitted-code file. V2 reproduces the established
construction exactly:

```text
change[t] = price[t + 1] - price[t]
signal[t] = "+" if change[t] > 0.50
            "-" if change[t] < -0.50
            "0" otherwise
signal[364] = "0"
```

This construction uses `price[t+1]` and is consequently included only as an
intentional leakage control. It is excluded from every valid strategy table,
stress summary, and recommendation.

## Mechanics audit

The valid backtests use the following causal accounting identity:

```text
position[t] is selected using information through price[t]
daily_pnl[t] = position[t] * (price[t+1] - price[t])
```

There is no return for day 364. The final position is explicitly zero, and the
last `next_return` is not included in P&L. The EWMA is updated in the same
order as the decision rule:

```text
old_level       = fair value known before day t
innovation[t]   = price[t] - old_level
post-update     = old_level + alpha * innovation[t]
post deviation  = price[t] - post-update
prior deviation = innovation[t]
```

The post-update implementation uses the teammate-style persistence and
reversal rule: a positive deviation requests a short position, a negative
deviation requests a long position, neutral observations preserve the prior
overlay position where the submitted mechanics require persistence, and
reversals are desired-position changes rather than trade quantities. The
corrected prior-state variant makes the current decision from `innovation[t]`
and updates only for the next day. The rolling scale uses observations before
the current decision; the startup period remains flat until enough history is
available.

The audit passed 18/18 correctness checks, including:

* exact day-`t` to day-`t+1` P&L alignment;
* a deterministic toy-series alignment assertion;
* future-price perturbation causality checks;
* integral positions and the ±AUD 1,000 Boat limit;
* flat day 364;
* frozen production signal provenance;
* exclusion of the leakage control; and
* no standalone Boat budget violation.

The full correctness output is
[`ewma_v2_correctness_checks.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_correctness_checks.csv).

## Round 1 strategy comparison

These are in-sample reconstruction diagnostics, not unbiased Round 2
estimates. Candidate D's AUD 92,560 decomposes into AUD 80,010 from the fixed
semester signal and AUD 12,550 from fixed AUD 45 summer reversion.

| Strategy | Total P&L | Increment vs D | Neutral overlay | Summer | Max drawdown | Active days | Trades | Turnover | Max Boat notional |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidate D | 92,560 | 0 | 0 | 12,550 | -2,460 | 333 | 92 | 134,000 | 55,390 |
| D + post EWMA, alpha .65 | 101,660 | 9,100 | 9,100 | 12,550 | -3,080 | 357 | 72 | 134,000 | 55,390 |
| D + post EWMA, alpha .90 | 99,800 | 7,240 | 7,240 | 12,550 | -2,460 | 352 | 73 | 126,000 | 55,390 |
| D + prior-state EWMA, alpha .65 | 100,890 | 8,330 | 8,330 | 12,550 | -3,080 | 360 | 72 | 138,000 | 55,390 |
| D + prior-state EWMA, alpha .90 | 101,190 | 8,630 | 8,630 | 12,550 | -3,080 | 359 | 72 | 136,000 | 55,390 |
| D + one-day reversal | 89,610 | -2,950 full year | 9,600 | 0 | -3,080 | 321 | 47 | 92,000 | 55,390 |
| D + causal MA20 reversal | 76,920 | -15,640 full year | -3,090 | 0 | -4,980 | 318 | 55 | 104,000 | 55,390 |

The one-day reversal row is a control: its full-year total excludes the fixed
AUD 45 summer rule, while its neutral-semester overlay P&L is directly
comparable. EWMA does not improve on this simple neutral-day control. The
complete table, including hit rates, Sharpe, long/short P&L, best/worst days,
and position checks, is
[`ewma_v2_strategy_comparison.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_strategy_comparison.csv).

The post-alpha-.65 attribution is:

| Source | P&L | Active days |
|---|---:|---:|
| Fixed Candidate-D semester signal | 80,010 | 291 |
| Neutral EWMA overlay | 9,100 | 24 |
| Fixed AUD 45 summer rule | 12,550 | 42 |
| Total | 101,660 | 357 |

For this strategy, long P&L is AUD 50,580 and short P&L is AUD 51,080.
The best day is AUD 6,710 on day 181 and the worst day is AUD -1,360 on day
57. The detailed attribution file is
[`ewma_v2_pnl_attribution.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_pnl_attribution.csv).

## Why alpha comparisons are easily misleading

For the supplied post-update ordering,

```text
ewma_new = ewma_old + alpha * (price - ewma_old)
price - ewma_new = (1 - alpha) * (price - ewma_old)
```

Thus a raw post-update threshold of AUD 0.05 corresponds to a pre-update
innovation threshold of approximately AUD 0.1429 at alpha 0.65 but AUD 0.50
at alpha 0.90. Alpha changes both adaptation speed and signal scale. V2
therefore reports the raw equivalent threshold and separately evaluates a
coarse standardized causal-innovation grid rather than treating a fixed raw
threshold as an apples-to-apples alpha comparison.

With the historical innovation standardizer, post and prior-state deviations
are algebraically rescaled versions of the same innovation. Their standardized
signals are therefore nearly identical in this data; the prior-state version
is still retained as the cleaner causal formulation. At alpha .65, the
diagnostic denominator comparison gives incremental P&L of AUD 8,330 with
historical innovation volatility, AUD 8,630 with return volatility, AUD 9,100
with price-level volatility, and AUD 8,330 with MAD innovation scale. At alpha
.90 the corresponding post-update increments are AUD 8,630, AUD 7,080,
AUD 7,240, and AUD 8,330.

The coarse standardized parameter grid does not show a narrow alpha optimum:
for alpha 0.30--0.90, median incremental P&L across windows and thresholds is
positive, while alpha 0.10 is negative. This is still a same-year grid, not
parameter validation. The full grid and equivalent-threshold columns are in
[`ewma_v2_parameter_stability.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_parameter_stability.csv),
with denominator diagnostics in
[`ewma_v2_denominator_comparison.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_denominator_comparison.csv).

## Chronology and concentration

The post-update alpha-.65 overlay is positive in both coarse semesters and in
five of seven 60-day blocks:

| Segment | Increment vs Candidate D |
|---|---:|
| Semester 1 | 2,930 |
| Semester 2 | 4,420 |
| Block 0 | 590 |
| Block 1 | 3,050 |
| Block 2 | -710 |
| Block 3 | 2,690 |
| Block 4 | 720 |
| Block 5 | 2,760 |
| Block 6 | 0 |

Leave-one-block-out remains positive, but the weakest retained increment is
AUD 6,050 when block 1 is removed. This is encouraging as a Round 1
description, but it is not independent validation because every result uses
the same path and the seasonal template is itself a full-path prior. The
full chronology and leave-one-block-out files are
[`ewma_v2_chronological_splits.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_chronological_splits.csv)
and
[`ewma_v2_leave_one_block_out.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_leave_one_block_out.csv).

Incremental P&L concentration for post alpha .65 is:

| Best realized overlay days | Share of total incremental P&L |
|---:|---:|
| 1 | 13.6% |
| 5 | 58.8% |
| 10 | 100.3% |
| 20 | 125.1% |

The top five days are a majority of the gain, and the top ten exceed the total
because some other overlay days lose money. Alpha .90 is more concentrated
(17.1%, 68.9%, 112.7%, and 120.3%). The concentration diagnostics are in
[`ewma_v2_concentration.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_concentration.csv).

Shifting only the post-alpha-.65 overlay gives full-year P&L of AUD 92,690,
94,580, 89,250, **101,660**, 91,550, 91,720, and 94,210 for displacements
-3, -2, -1, 0, +1, +2, and +3 days respectively. This is a useful timing
diagnostic but not a rejection by itself: immediate mean reversion should
decay with delay. Randomizing overlay signs while preserving activity gives a
median incremental result of about AUD -220 for alpha .65 and AUD +120 for
alpha .90, with lower tails below AUD -10,000. The placebo output is
[`ewma_v2_timing_and_sign_placebos.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_timing_and_sign_placebos.csv).

## Predictive evidence

On Candidate-D neutral semester days, the post-alpha-.65 causal deviation has
31 usable observations. The regression is

```text
return[t+1] = intercept + beta * deviation[t] + error[t+1]
```

| Specification | n | Beta | HAC 95% interval | Rank correlation | Directional hit rate |
|---|---:|---:|---:|---:|---:|
| Post EWMA .65, all neutral | 31 | -2.614 | [-3.799, -1.429] | -0.633 | 64.5% |
| Post EWMA .65, Semester 1 | 18 | -2.574 | approximately [-4.267, -0.882] | — | — |
| Post EWMA .65, Semester 2 | 11 | -2.018 | approximately [-4.035, -0.002] | — | — |
| Post EWMA .65, remove best 1 | 30 | -2.387 | approximately [-3.582, -1.192] | — | — |
| Post EWMA .65, remove best 5 | 26 | -1.846 | approximately [-2.930, -0.762] | — | — |
| Post EWMA .65, remove best 10 | 21 | -1.083 | approximately [-2.279, 0.113] | -0.312 | 47.6% |
| Prior-state EWMA .65, all neutral | 31 | -0.915 | [-1.330, -0.500] | — | — |
| Prior-state EWMA .65, remove best 10 | 21 | -0.379 | approximately [-0.798, 0.040] | — | — |

The sign of beta is consistent with in-sample mean reversion, and the
latest-return-controlled post-alpha-.65 beta remains negative at about -6.01
(n=30; interval approximately [-10.04, -1.99]). However, the result is based
on very few eligible observations, weakens materially after removing the ten
best realized overlay days, and the simple one-day reversal makes at least as
much neutral-day P&L. That means the regression is evidence for a short-horizon
reversal pattern in this Round 1 path, not evidence that the EWMA fair value
adds independent information beyond latest-return reversal. Newey-West/HAC
inference was used for the reported intervals; p-values were not relied upon
where the small-sample fit returned non-finite values. Full regression rows,
including all chronological blocks and latest-return controls, are in
[`ewma_v2_predictive_regressions.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_predictive_regressions.csv).

## Seasonality-preserving stress tests

The primary synthetic tests retained the broad calendar shape and varied the
four seasonal wave amplitudes independently over 0.5, 0.75, 1.0, 1.25, and
1.5. They also included fixed-seed residual block resampling, volatility
scaling, local shocks, and a secondary timing-displacement diagnostic. There
were 625 deterministic amplitude combinations plus 150 residual paths, 75
shock paths, and 250 secondary timing-shift paths. These are
**generator-conditioned stress tests**, not confidence intervals or true
out-of-sample observations.

| Strategy | Median P&L | P10 P&L | Worst P&L | Positive-path rate | Median max drawdown |
|---|---:|---:|---:|---:|---:|
| Candidate D | 55,413.5 | 39,046.5 | 17,720.0 | 100.0% | -1,834.8 |
| D + post EWMA .65 | 57,786.9 | 40,834.7 | 18,379.8 | 100.0% | -2,000.8 |
| D + prior-state EWMA .65 | 57,786.9 | 40,834.7 | 18,379.8 | 100.0% | -2,000.8 |
| D + one-day reversal | 53,217.9 | 38,030.4 | 6,390.5 | 100.0% | -2,073.2 |

Against Candidate D, post/prior EWMA .65 have paired median improvement
AUD 2,114.3, P10 difference AUD -637.9, worst difference AUD -9,273.1, and
positive difference rate 81.6%. This is a mild stress-test improvement in the
middle, but the lower tail is worse. It is not sufficient to justify spending
shared portfolio budget on an in-sample overlay. Detail and summaries are in
[`ewma_v2_seasonality_stress_detail.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_seasonality_stress_detail.csv),
[`ewma_v2_seasonality_stress_summary.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_seasonality_stress_summary.csv),
and
[`ewma_v2_seasonality_paired_differences.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_seasonality_paired_differences.csv).

## Summer fair-value tests

Summer variants were evaluated separately from the semester overlay. The
fixed-seed family included abrupt, gradual, and temporary displacements with
targets AUD 42, 44, 45, 46, and 48; gradual transitions used 7, 14, and 28
days. Each candidate saw the same paired synthetic path.

Pooled across the summer scenarios, the fixed AUD 45 rule has median full-year
P&L AUD 60,126.7, P10 AUD 41,818.7, and worst AUD 21,650. The adaptive
alpha-.10 summer rule has median AUD 64,217.1, P10 AUD 47,822.1, and worst
AUD 26,280, but its paired difference versus fixed 45 is median +3,568.6,
P10 -3,760.4, and worst -18,486.7. The worst paired result is therefore a
substantial failure on one of the generator-conditioned scenarios.

The adaptive rule also underperforms when the true equilibrium is exactly
AUD 45: its median full-year P&L is AUD 66,877.9 versus AUD 69,109.8 for
fixed 45. It performs well for permanent shifts, but it loses that advantage
when a temporary displacement returns to the anchor. For example, with a
temporary AUD 42 displacement, fixed 45 has median P&L about AUD 64,868
versus AUD 60,479 for adaptive alpha .10; with temporary AUD 48, the figures
are about AUD 64,631 versus AUD 60,758. Shrunk and guarded variants reduce
some of the risk but do not produce a broad, reliable improvement: the
guarded rule does not activate on the observed Round 1 path, and the best
shrunk anchor result has only a small same-year benefit.

The summer conclusion is therefore to retain fixed AUD 45. It preserves the
known-anchor case, avoids chasing temporary noise, and does not require a
parameterized fair-value contingency. Summer detail, paired differences, and
the diagnostic grid are in
[`ewma_v2_summer_stress_detail.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_summer_stress_detail.csv),
[`ewma_v2_summer_stress_paired_differences.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_summer_stress_paired_differences.csv),
[`ewma_v2_summer_stress_summary.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_summer_stress_summary.csv),
and
[`ewma_v2_summer_parameter_grid.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_summer_parameter_grid.csv).

## Leakage control

The thresholded future-return control is intentionally disqualified:

| Control | Fixed +/- days | Fixed hit rate | P&L | Included in valid selection? |
|---|---:|---:|---:|---:|
| Future-return threshold string | 160 | 100% | 185,880 | No |
| Future-return string + post EWMA .65 | 160 | 100% | 196,220 | No |
| Future-return string + post EWMA .90 | 160 | 100% | 195,810 | No |

The result demonstrates why a high Round 1 P&L is not enough. The fixed signal
already knows the sign of the next return and accounts for virtually all of
the teammate's reported result. The leakage file is
[`ewma_v2_leakage_control.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_leakage_control.csv).

## Portfolio capital

The read-only current-production portfolio reference uses maximum other-
instrument capital AUD 528,631. Candidate D's maximum Boat notional is
AUD 55,390, so the maximum observed combined exposure is AUD 575,991, below
the AUD 600,000 cap. No candidate in the audit creates a combined budget
violation in this diagnostic. The largest possible additional Boat notional
relative to D on an overlay-active day is AUD 54,820; mean additional Boat
capital while active is approximately AUD 3,183 for post alpha .65, AUD 2,571
for post alpha .90, AUD 3,546 for prior alpha .65, and AUD 3,444 for prior
alpha .90. No other instrument loses allocation in this read-only comparison
because the combined cap is not breached; the allocator was not optimized.

These capital figures do not turn the in-sample increment into a production
edge. They show that the apparent improvement is not forced by an accidental
standalone portfolio-limit violation. Details are in
[`ewma_v2_portfolio_capital.csv`](D:/Documents/Algojam/research/boat_party/results/ewma_v2_portfolio_capital.csv).

## Reproducibility and validation status

The executed V2 notebook is
[`analysis_v2.ipynb`](D:/Documents/Algojam/research/boat_party/analysis_v2.ipynb).
It runs the module, displays the main tables, and completed with zero error
outputs. The module and notebook were compiled/executed using the repository's
bundled environment. The unchanged simulator regression remained clean:
total P&L AUD 638,056.50, including Boat Party Ticket Returns AUD 92,560,
with no reported errors or budget violations.

No production file was modified. In particular,
`trader_interface/algorithm.py`, `trader_interface/simulation.py`, supplied
price data, and PDFs remain unchanged.
