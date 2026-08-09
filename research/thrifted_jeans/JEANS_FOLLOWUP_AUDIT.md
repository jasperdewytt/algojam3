# Thrifted Jeans follow-up audit

Status: research-only. This report is a separated follow-up to `JEANS_STRATEGY_AUDIT.md`; the earlier audit files and outputs are preserved. The production algorithm and supplied simulator were not modified.

## Executive verdict

The omitted hybrid-Kalman-without-EMA replay is real, but it does not justify replacing K2 yet.

- Candidate C (the hybrid Kalman parameters with no EMA) makes **AUD 119,584** under strict flat-day-zero timing, or **AUD 19,568 more than K2**. The previously quoted AUD 119,960 arithmetic includes the day-zero return; that is inconsistent with the requested flat-day-zero rule and overstates the replay by AUD 376.
- The corrected hybrid makes **AUD 151,584**, but its extra AUD 32,000 over Candidate C is a weak-state reversal contribution whose statistical interval includes zero and whose paired advantage is materially generator-dependent.
- The original hybrid makes **AUD 162,496**, but its residual scale is knowingly wrong. It is diagnostic only; it is not a production candidate.
- K2 remains the recommended submission: it is the simplest causal member of the “stay long unless the negative slope is convincing” family. Candidate C is a reasonable runner-up/uncertain candidate if later evidence makes the additional short exposure worthwhile. The corrected hybrid is not promoted.

This is not evidence that EMA mean reversion has no effect. It is evidence that one year of Round 1 data is insufficient to establish that its estimated effect is stable enough to pay for the additional state and threshold decisions.

## 1. Data, structural prior and timing

The supplied Jeans history contains 365 daily prices, from AUD 40.00 to AUD 87.67. Daily price-change autocorrelation is -0.0505, so one-day return predictability is weak. The description suggests a positive-drift instrument with persistent directional episodes; that is a prior, not a guarantee that every unseen year will rally.

The supplied simulator convention is important: the desired position selected on day `t` earns only the change from `t` to `t+1`. Day zero is flat for all audited candidates and the final day has no future return. Positions are desired integer positions, not trade quantities. There are no transaction costs in the supplied simulator.

The requested Candidate A therefore makes AUD 37,760. The familiar AUD 38,136 is the full `+800` reference held from day zero. Comparisons against the full reference are labelled explicitly; paired comparisons between K2, C and the hybrids are unaffected by this day-zero convention.

For Candidate C, the attribution arithmetic `AUD 95,624 strong + AUD 24,336 weak-long reference = AUD 119,960` therefore does not reproduce the strict replay. The strict weak-long branch is AUD 23,960, giving the direct causal replay AUD 119,584.

The maximum standalone Jeans notional observed for every full-limit candidate is AUD 75,824, below the AUD 600,000 gross limit. This is not the total portfolio budget: the unchanged full-portfolio Round 1 replay previously peaked at AUD 555,931 with full-limit Jeans and had zero budget breaches. More turnover or more short days do not increase the Jeans maximum notional, but they can increase execution and shared-portfolio interaction risk.

## 2. Exact candidate replay

`MDD` is maximum drawdown of the strategy P&L curve. “Best 5 share” is the share of total P&L contributed by that strategy’s five best realised days; it is descriptive, not a significance measure.

| Candidate | P&L | Increment vs full +800 | Increment vs K2 | MDD | Long/short days | Long P&L | Short P&L | Turnover | Best 5 share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A Always long, flat day 0 | 37,760 | -376 | -62,256 | -44,856 | 364 / 0 | 37,760 | 0 | 800 | 73.4% |
| B Existing K2 | 100,016 | 61,880 | 0 | -26,672 | 291 / 73 | 68,888 | 31,128 | 29,600 | 31.4% |
| C Hybrid Kalman, no EMA | 119,584 | 81,448 | **19,568** | -35,512 | 254 / 110 | 78,672 | 40,912 | 32,800 | 26.3% |
| D Corrected causal hybrid | 151,584 | 113,448 | 51,568 | **-14,536** | 196 / 155 | 90,792 | 60,792 | 53,600 | 20.7% |
| E Original hybrid, diagnostic only | 162,496 | 124,360 | 62,480 | -14,536 | 199 / 154 | 94,880 | 67,616 | 58,400 | 19.3% |

K2 and C both earn positive P&L from their short positions, but C’s improvement is not exclusively short-side: its long and short P&L are each AUD 9,784 above K2. C also has 37 more short days and a larger drawdown. D improves drawdown in this sample, but at substantially higher turnover and with 155 short days.

## 3. What the omitted ablation shows

All of these use the same hybrid Kalman strong/weak state definition: strong if `abs(slope_z) >= 0.6`. Strong-state Kalman direction contributes AUD 95,624 for the ablations. The table isolates what happens in weak states.

| Weak-state rule | P&L | Increment vs K2 | Strong P&L | Weak P&L | Long/short P&L | MDD | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strong direction; flat weak | 95,624 | -4,392 | 95,624 | 0 | 54,712 / 40,912 | -29,864 | 38,400 |
| Strong direction; long weak | 119,584 | 19,568 | 95,624 | 23,960 | 78,672 / 40,912 | -35,512 | 32,800 |
| Strong direction; hold previous weak | 80,200 | -19,816 | 95,624 | -15,424 | 57,008 / 23,192 | -40,680 | 26,400 |
| Strong direction; corrected EMA weak (D) | 151,584 | 51,568 | 95,624 | 55,960 | 90,792 / 60,792 | -14,536 | 53,600 |
| Strong direction; reverse latest one-day return weak | 134,592 | 34,576 | 95,624 | 38,968 | 86,176 / 48,416 | -19,960 | 100,000 |
| Strong direction; causal EMA sign weak | 153,488 | 53,472 | 95,624 | 57,864 | 94,800 / 58,688 | -15,024 | 58,400 |
| Existing K2 | 100,016 | 0 | 76,056 | 23,960 | 68,888 / 31,128 | -26,672 | 29,600 |

This is the critical result. Candidate C reproduces “strong-state Kalman direction plus remain long when uncertain”; it does not need an EMA to reach AUD 119,584. The corrected hybrid’s weak branch adds AUD 32,000 over simply staying long, but the non-stateful EMA-sign ablation makes AUD 153,488—AUD 1,904 more than the stateful corrected hybrid. The apparent hybrid edge is therefore not evidence that the original stateful entry/hold/reversal machinery is valuable. It is mostly a conditional weak-state reversal signal, layered on top of the hybrid Kalman threshold.

The original bad-residual hybrid is AUD 10,912 above the corrected hybrid, entirely in the weak branch (AUD 66,872 versus AUD 55,960). That is material for model ranking, but it does not explain the whole hybrid result: the no-EMA Candidate C already beats K2 by AUD 19,568.

## 4. Corrected weak-state predictive analysis

The previous “best five days” check was invalid. It removed the five largest raw price changes and compared a weak-state sample with 86 observations against a different all-state sample. The follow-up keeps the same 86 eligible weak-state observations and ranks only the tested weak-state contribution.

For the simple reversal signal, contribution is `-sign(ema_deviation_z) * next_change`. For the actual stateful hybrid, contribution is the realised weak-state position times next change, divided by 800 so that it is in price-change units.

| Test within the same weak sample | N | Simple mean contribution | Simple hit rate | Simple beta | 95% HAC interval |
|---|---:|---:|---:|---:|---:|
| All weak observations | 86 | 0.841 | 61.6% | -0.817 | [-1.897, 0.262] |
| Exclude best 1 signal contribution | 85 | 0.765 | 61.2% | -0.622 | [-1.512, 0.268] |
| Exclude best 3 | 83 | 0.629 | 60.2% | -0.500 | [-1.352, 0.352] |
| Exclude best 5 | 81 | 0.509 | 59.3% | -0.440 | [-1.271, 0.392] |
| Exclude worst 5 | 81 | 1.158 | 65.4% | -0.951 | [-2.081, 0.178] |

The actual stateful contribution means are 0.813, 0.737, 0.600, 0.480 and 1.112 for the same rows; its full-sample reversal hit rate is 55.8%. Winsorising at absolute contributions of 2.5 and 4.0 leaves positive mean contributions of 0.535 and 0.710. These are directionally supportive, but every full-sample interval above includes zero. The result is suggestive rather than decisive.

The sign split is asymmetric: negative deviations, which call for a long reversal, have 38 observations, mean next change +1.319 and a 73.7% reversal hit rate; positive deviations, which call for a short reversal, have 48 observations, mean next change -0.463 and a 52.1% hit rate. Magnitude buckets show negative coefficients for `|z| < 0.5` and `0.5 <= |z| < 1`, but those are small, selected diagnostic buckets rather than independent validation samples.

The weak-entry sample has mean contribution 0.927; later weak-state observations have mean 0.806. The later-state regression is more negative, but the entry sample is too small for a reliable distinction. Leave-one-quarter results keep a negative coefficient in each quarter, but Q3 has only five observations and Q4 is close to zero in contribution. Leave-one-60-day results are similarly uneven.

The interaction regression over all usable observations is:

`next_change = intercept + beta1 * ema_z + beta2 * weak + beta3 * ema_z * weak + error`

| Term | Estimate | HAC SE | 95% interval | Normal p |
|---|---:|---:|---:|---:|
| Intercept | 0.017 | 0.168 | [-0.313, 0.346] | 0.921 |
| EMA z | 0.133 | 0.106 | [-0.074, 0.341] | 0.208 |
| Weak state | 0.369 | 0.305 | [-0.228, 0.967] | 0.226 |
| EMA z × weak | **-0.951** | 0.523 | [-1.975, 0.074] | 0.069 |

The negative interaction is economically plausible and close to conventional significance, but its interval still includes zero. A non-significant corrected EMA regression is inconclusive evidence, not proof of no edge; here the effect size and directional consistency warrant retaining it as a research hypothesis, not promoting it as a one-year production rule.

## 5. Chronological evidence and concentration

Candidate C beats K2 in every non-overlapping quarter:

| Segment | K2 P&L | C P&L | C minus K2 | D P&L | D minus K2 |
|---|---:|---:|---:|---:|---:|
| Q1 | -3,784 | 1,416 | 5,200 | 26,600 | 30,384 |
| Q2 | 26,496 | 28,688 | 2,192 | 39,592 | 13,096 |
| Q3 | 54,856 | 59,672 | 4,816 | 71,272 | 16,416 |
| Q4 | 22,448 | 29,808 | 7,360 | 14,120 | -8,328 |

By half-year, C’s advantages are AUD 7,392 and AUD 12,176; D’s are AUD 43,480 and AUD 8,088. Across the combined quarterly, 60-day, half-year and early/middle/late segment set, C wins 13 of 16 paired segments and D wins 12 of 16. C’s cumulative paired-advantage drawdown is AUD 29,680; D’s is AUD 27,208.

The 60-day pattern is less uniform. C versus K2 is +20,400, -18,912, +5,904, -2,608, +7,424, +7,360 and 0 across B1–B7. D versus K2 is +15,568, +21,672, +6,240, +8,992, +7,424, -6,504 and -1,824. Thus C is consistently positive at quarterly scale but not every shorter block; D loses the late positive segment in this sample.

Excluding the final rally does not remove the K2/C distinction:

| Prefix | Always long | K2 | C | C minus K2 | D | D minus K2 |
|---|---:|---:|---:|---:|---:|---:|
| Exclude final 30 days | 20,584 | 82,840 | 102,408 | 19,568 | 153,800 | 70,960 |
| Exclude final 60 days | 19,840 | 88,576 | 100,432 | 11,856 | 150,232 | 61,656 |
| Exclude final 91 days | 5,552 | 74,288 | 86,496 | 12,208 | 134,184 | 59,896 |

Always-long is disproportionately dependent on the final rally: about 85% of its strict flat-day-zero P&L arrives in the final 91 days. C and K2 are less dependent on that rally because their negative-slope positions earn money in earlier declines. Removing each strategy’s own best realised days gives another caution: C’s advantage over K2 stays AUD 19,568 after its best five days, falls to AUD 9,600 after its best ten, and is only AUD 288 after its best 20. D retains AUD 51,568, AUD 29,952 and AUD 12,128 after its best 5, 10 and 20 days respectively. These are per-strategy exclusions, not a paired common-day test, so they are reported as concentration diagnostics rather than independent validation.

## 6. Simple Kalman family sensitivity

Both K2 and C were selected after observing Round 1; neither exact parameter set is an honest untouched out-of-sample choice. The predeclared follow-up tested 40 simple-Kalman configurations, grouped into one-at-a-time changes in `R`, one-at-a-time threshold changes, and modest joint `q_level`/`q_slope` grids.

| Base family | Variation | Configurations | Median P&L | Lower decile | Worst | Median incremental vs K2 | Median positive quarters |
|---|---|---:|---:|---:|---:|---:|---:|
| K2 | R one-at-a-time | 3 | 81,808 | 78,365 | 77,504 | -18,208 | 3 |
| K2 | q-level/q-slope joint | 12 | 80,976 | 69,126 | 68,384 | -19,040 | 3 |
| K2 | threshold one-at-a-time | 5 | 86,432 | 76,973 | 73,824 | -13,584 | 3 |
| C | R one-at-a-time | 3 | 95,568 | 94,864 | 94,688 | -4,448 | 3 |
| C | q-level/q-slope joint | 12 | 95,392 | 76,507 | 57,472 | -4,624 | 3 |
| C | threshold one-at-a-time | 5 | 85,664 | 77,715 | 73,728 | -14,352 | 3 |

C’s exact AUD 119,584 is the maximum of its 12-cell joint group. Its nearby one-at-a-time P&Ls range from AUD 73,728 to AUD 115,504; only a subset of nearby cells beat exact K2. K2 itself is also not a broad maximum—one nearby cell reaches AUD 101,904—but K2 requires fewer special decisions and has lower turnover and drawdown than C. The family result therefore supports a stable qualitative rule, not the exact C cell.

## 7. Corrected price-path bootstrap and generator stress

The primary bootstrap resamples absolute daily price changes in circular moving blocks, reconstructs each path from AUD 40.00 without shifting it upward, and reruns every strategy causally. Block lengths are 5, 10, 20, 40 and 60; there are 1,000 repetitions per length. Paths leaving the predeclared `(1, 200)` price range are retained in the all-attempts result and separately identified; the plausible-path subset is not silently substituted for the main result. Plausible fractions are 79.7%, 72.7%, 65.9%, 72.7% and 78.1% respectively.

The most useful paired results are below. “Win” is the fraction of reconstructed paths where the first strategy’s total P&L exceeds the reference’s total P&L; it is not the probability that either strategy is profitable.

| Additive block | C minus K2 median | C P5 | C win | D minus K2 median | D P5 | D win | D minus C median | D win |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | -8,928 | -54,652 | 37.7% | -15,368 | -92,863 | 36.6% | -8,112 | 40.2% |
| 10 | 688 | -45,309 | 50.8% | -9,336 | -88,121 | 41.9% | -10,328 | 38.8% |
| 20 | 7,536 | -39,156 | 59.7% | -4,876 | -78,232 | 45.8% | -10,004 | 37.4% |
| 40 | 13,048 | -34,304 | 68.0% | 15,520 | -55,626 | 64.0% | 1,480 | 52.4% |
| 60 | 16,280 | -31,631 | 71.6% | 25,080 | -40,642 | 73.9% | 11,344 | 60.4% |

At block 20, for example, C has 97.1% positive P&L paths, median P&L AUD 80,920 and median increment over always-long AUD 36,920; D has 95.3% positive paths, median P&L AUD 70,660 and median increment AUD 26,696. Both can beat always-long while still losing paired comparisons to K2 on many paths. This is why separate probabilities of beating always-long are not a valid preference test.

The drift-plus-residual block design uses drift multipliers 1.0, 0.75, 0.5, 0, and 1.25. Across those assumptions, C minus K2 has median paired differences AUD 7,688, 8,913, 14,107, 12,815 and 2,427, with win fractions 61.7%, 60.7%, 67.0%, 67.7% and 53.0%. D minus K2 has medians -1,232, -2,670, 1,852, 8,533 and -8,389, with win fractions 47.3%, 47.0%, 52.7%, 56.7% and 42.3%. The C result is more stable than D under this design, but its lower-tail differences remain negative.

The old percentage-return bootstrap is retained only as a secondary diagnostic. Its compounded paths change the dollar price scale while keeping fixed-dollar Kalman variances unchanged. It produces C/K2 median paired differences of only AUD 576 and AUD 3,039 for blocks 10 and 20, with wins 50.3% and 55.7%; D/K2 is -AUD 12,152 and -AUD 11,925, with wins 39.3% and 42.3%. This material change in ranking and scale sensitivity is why the old percentage bootstrap cannot be the primary selection test.

The generator-conditioned regime stresses are deliberately not confidence intervals. C beats K2 on long-negative, gradual-transition, low-noise and frequent-reversal paths, but loses on persistent-positive, balanced 40-day, short-reversal and high-noise paths. In particular, C loses AUD 12,135 on the persistent-positive scenario and gains AUD 9,227 on the long-negative scenario. That is consistent with the structural trade-off: shorts help if negative regimes persist, and hurt if the positive drift persists.

## 8. Family-wise selection bias

The follow-up counted 10 base candidates, 40 simple-Kalman configurations, and 64 unique effective configurations in the corrected main family after duplicate position paths were removed. The knowingly incorrect original hybrid was excluded from the main family maximum. The observed family maximum is AUD 153,488, corresponding to the strong-direction/causal-EMA-sign weak ablation—not a promoted production candidate.

Against 1,000 circular absolute-change null paths using block length 10, the family-maximum null median is AUD 87,564 and its 95th percentile is AUD 150,194. The all-attempts exceedance count is 43/1,000, giving a Monte Carlo p estimate 0.0440 with SE 0.0065 and approximate interval [0.031, 0.057]. Restricting to the 733 paths that remain in the plausible positive range gives p 0.0286, SE 0.0062, interval [0.017, 0.041]. This is marginal family-wise evidence that the searched family found structure in Round 1; it is not a second-year validation result, and it does not establish that C or the corrected hybrid will win unseen regimes.

## 9. Correctness and production-budget checks

Every candidate passed the follow-up checks for integer positions, the ±800 Jeans limit, standalone notional, day-zero flatness, prior-position P&L, final-day handling, prefix replay and future perturbation. Audit days for future perturbations were 0, 30, 120, 240 and 300; changing every price after each audit day did not change the position selected on that day.

The research implementation and simulator convention agree on desired positions and prior-position P&L. No production file was changed. The full portfolio budget figure above is a separate unchanged portfolio replay; it should not be confused with the Jeans-only AUD 75,824 notional or with standalone strategy P&L.

## 10. Final choice

**Submit existing K2 when implementation is requested:**

1. Start flat on day zero.
2. Use the causal local-linear filter with `q_level=1.0`, `q_slope=0.05`, observation variance `R=9.0`, and initial slope variance `1.0`.
3. On each later day, update using prices through that day and calculate `slope_z = filtered_slope / slope_uncertainty`.
4. Desired position is `-800` if `slope_z < -1.0`; otherwise `+800`.
5. Do not add the EMA branch, stateful weak-state reversal, or Round 1 parameter refitting.

This is a symmetric long/short rule, but it remains structurally long-biased because it is long everywhere except when the negative trend is sufficiently convincing. Candidate C is the runner-up if Round 2 evidence supports longer negative regimes: it uses the hybrid Kalman settings `(0.5, 0.05, 5.0)` and the same `+800/-800` rule with threshold `-0.6`. Its advantage is visible across the four quarters and at medium/long bootstrap blocks, but its exact cell is not a broad plateau and its short-block downside is substantial.

Further Jeans research is unlikely to add much value by tuning more Round 1 thresholds. The useful next evidence would be genuine Round 2 observations or an externally justified prior about regime duration/volatility. Until then, the appropriate conclusion is: **Candidate C is an uncertain improvement; K2 is the defensible production choice; the corrected hybrid is not justified; the original hybrid is invalid.**

## Reproducibility map

- `jeans_followup_audit.py`: causal replay, ablations, corrected weak-state tests, sensitivity family, path bootstrap, generator stress, family-wise null and correctness checks.
- `thrifted_jeans_followup_audit.ipynb`: executed notebook; zero error cells.
- `followup_outputs/`: all CSV tables and the run manifest.
- `followup_figures/`: cumulative P&L, paired advantage, weak-state attribution, parameter stability, paired bootstrap differences and slope-z diagnostics.
