# Thrifted Jeans: third-stage one-sided EMA audit

Status: research-only. No production strategy, simulator, supplied data, or competition file was modified.

## Executive verdict

**Verdict A — keep Candidate C.** Do not add the one-sided EMA branch to the submitted Jeans rule.

The primary predeclared one-sided rule (alpha 0.06, 20 prior residuals, centred residual z entry threshold 0.5) earns **AUD 99,528**, versus **AUD 119,584** for Candidate C. Its paired disadvantage is **AUD -20,056**. The overlay’s weak-state dip trades earn AUD 3,904, but Candidate C earns AUD 23,960 by simply remaining long in all weak states. The flat weak-state days lose more than the selected dips recover.

The older 73.7% long-side asymmetry is real for its stated historical sample: 28/38 observations, mean next-day change AUD 1.318684. It does not, by itself, validate a new rolling-residual trading rule. With the required causal centred W20 residual, the negative weak-state sample is larger but weaker (40 observations, 67.5% positive, mean change AUD 0.5805, HAC 95% interval for the return-on-z slope [-0.723, 2.746]). The positive side is not a useful short signal (45 observations, 48.9% short hit rate, mean change AUD 0.0400).

The hysteresis variants also remain below Candidate C: AUD 102,584 with exit z=0 and AUD 106,296 with exit z=+0.25. The entirely no-short rule earns AUD 58,616. The diagnostic uncentred version earns AUD 115,672, but that scale choice is not the predeclared centred rule and is too close to a Round 1-specific residual convention to justify production use. The prior corrected two-sided hybrid remains a useful diagnostic at AUD 151,584, but its additional EMA state machine is not supported as the simplest unseen-year choice.

## 1. Data, structural prior, and timing

The audit uses the supplied 365-price Thrifted Jeans history. The price starts at AUD 40.00, ends at AUD 87.67, reaches a maximum of AUD 94.78 on day 228, and has a minimum of AUD 26.14. A strict day-zero-flat always-long replay earns AUD 37,760; the frequently quoted AUD 38,136 is the reference that holds +800 from day zero and therefore includes one extra day of exposure.

The instrument description implies a positive structural drift and potentially persistent trends, but not a guaranteed monotone path. That prior supports staying long unless a negative trend is convincing. It does not make every positive EMA deviation a short opportunity.

Every decision in this audit is causal. A desired position at day `t` uses prices through `t`; it earns only the observed change from `t` to `t+1`. Day zero is flat, the final desired position has no subsequent return, and there are no transaction costs. Positions are integral and bounded by 800 in absolute value. The research P&L array is simulator-aligned by placing the return from `t-1` to `t` at row `t`; its total is the same as the direct `position[:-1] * diff(price)` calculation.

## 2. Frozen candidates and exact attribution

| Candidate | P&L | Increment vs K2 | Increment vs C | MDD | Annualised Sharpe | Long / short / flat days | Turnover | Weak P&L | Weak dip P&L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always long | 37,760 | -62,256 | -81,824 | -44,856 | 0.894 | 364 / 0 / 1 | 800 | 23,960 | 0 |
| K2 | 100,016 | 0 | -19,568 | -26,672 | 2.384 | 291 / 73 / 1 | 29,600 | 23,960 | 0 |
| Candidate C | 119,584 | 19,568 | 0 | -35,512 | 2.860 | 254 / 110 / 1 | 32,800 | 23,960 | 0 |
| Corrected two-sided hybrid | 151,584 | 51,568 | 32,000 | -14,536 | 3.702 | 196 / 155 / 14 | 53,600 | 55,960 | 0 |
| One-sided stateless primary | 99,528 | -488 | -20,056 | -24,696 | 2.620 | 182 / 110 / 73 | 51,200 | 3,904 | 3,904 |
| Hysteresis, exit z=0 | 102,584 | 2,568 | -17,000 | -28,680 | 2.642 | 194 / 110 / 61 | 47,200 | 6,960 | 3,904 |
| Hysteresis, exit z=+0.25 | 106,296 | 6,280 | -13,288 | -25,880 | 2.722 | 197 / 110 / 58 | 45,600 | 10,672 | 3,904 |
| Entirely no-short long/flat | 58,616 | -41,400 | -60,968 | -13,456 | 1.987 | 182 / 0 / 183 | 35,200 | 3,904 | 3,904 |
| Reduced weak structural long +200 | 104,542 | 4,526 | -15,042 | -27,400 | 2.736 | 254 / 110 / 1 | 46,600 | 8,918 | 3,904 |
| Corrected two-sided stateless diagnostic | 96,256 | -3,760 | -23,328 | -18,024 | 2.440 | 182 / 137 / 46 | 70,400 | 632 | 3,904 |
| Uncentred one-sided diagnostic | 115,672 | 15,656 | -3,912 | -23,512 | 3.041 | 186 / 110 / 69 | 43,200 | 20,048 | 20,048 |

The primary one-sided rule has exactly the same strong-state P&L as Candidate C: AUD 95,624. The entire difference is weak-state handling:

| Weak-state attribution | AUD |
|---|---:|
| Candidate C: long throughout weak states | 23,960 |
| One-sided primary: selected dip entries | 3,904 |
| One-sided primary: other weak states | 0 |
| Primary weak overlay minus Candidate C weak P&L | **-20,056** |

Thus the proposed overlay does not improve the Kalman model. It changes only the uncertain-period exposure and gives up AUD 20,056 of realized weak-state P&L. Its best individual day is AUD 6,616 and its best-day share of total P&L is 6.65%; the problem is not one isolated bad outlier but the repeated opportunity cost of being flat.

The corrected two-sided hybrid’s AUD 151,584 and the older incorrectly standardised hybrid’s AUD 162,496 are retained as diagnostics only. The latter used a single final EMA value to standardise all historical prices, which is not a historical EMA residual series and must not be submitted.

## 3. Reproduction of the earlier asymmetry

The earlier 73.7% figure reproduces exactly when its original sample definition is preserved:

| Legacy sample | Definition | N | Mean next-day change | Hit rate |
|---|---|---:|---:|---:|
| Negative deviation, long reversal | `t=1..n-2`; `abs(C Kalman z)<0.6`; finite prior corrected EMA z; old minimum history 5 | 38 | +1.318684 | **28/38 = 73.684%** |
| Positive deviation, short reversal | Same sample, positive EMA deviation | 48 | -0.462917 | **25/48 = 52.083%** |

This is a useful descriptive asymmetry, not an out-of-sample result. The two sides have different sample sizes, the older volatility convention is not the new W20 rolling scale, and the sample was observed after Round 1.

## 4. Correct rolling-residual construction and predictive checks

For each day `t`, the new implementation forms `prior_ema[t]` from prices through `t-1`, computes `residual[t] = price[t] - prior_ema[t]`, and updates the EMA only after the signal is formed. The centred z-score subtracts the mean of the previous 20 residuals and divides by their sample standard deviation; today’s residual is excluded from both. The uncentred calculation is saved separately as a diagnostic.

With the primary centred W20 scale, eligible weak-state observations are:

| Weak-state bucket | N | Mean next change | Median | Directional hit |
|---|---:|---:|---:|---:|
| z < 0, long-side diagnostic | 40 | +0.5805 | +0.695 | 67.5% |
| z > 0, short-side diagnostic | 45 | +0.0400 | +0.040 | 48.9% short hit |
| z <= -1 | 12 | -0.1450 | +0.235 | 66.7% long hit |
| -1 < z <= -0.5 | 8 | +0.8275 | +0.660 | 62.5% long hit |
| -0.5 < z < 0 | 20 | +0.9170 | +1.310 | 70.0% long hit |
| 0 <= z < 0.5 | 18 | -0.1272 | +0.050 | 44.4% short hit |
| 0.5 <= z < 1 | 14 | -0.1043 | +0.480 | 50.0% short hit |
| z >= 1 | 13 | +0.4269 | -0.760 | 53.8% short hit |

The negative side remains directionally more promising, but the strongest negative bucket does not improve monotonically with magnitude. The large positive-side buckets are unstable and do not justify EMA shorts. The centred negative-side HAC regression has beta 1.0115, HAC SE 0.8847, 95% interval [-0.7225, 2.7456], normal p=0.253. The positive-side beta is 0.7820, HAC SE 0.9225, interval [-1.0261, 2.5901], p=0.397. These intervals are wide; the non-significance is inconclusive rather than proof of no effect, but it is not evidence strong enough to add a parameterised branch.

Weak-state entry versus later observations is also not a clean persistent effect. The weak-entry subset has N=24, mean change +0.2996, and a supporting regression interval [0.0343, 0.8367]; the later weak subset has N=61, mean +0.2923 and interval [-1.2511, 1.2117]. This is compatible with a small initial dip effect, but not with a stable rule that improves the full weak-state allocation.

## 5. Corrected best-trade robustness

The previous “best five days” test was invalid because it removed the five largest raw positive price changes and compared different samples. This audit ranks the exact one-sided strategy’s weak-state contribution, `position[t] * change[t+1]`, within the 20 eligible primary dip entries.

| Test | Remaining dip contribution | Remaining primary P&L | Primary minus C |
|---|---:|---:|---:|
| All 20 eligible dips | 3,904 | 99,528 | -20,056 |
| Remove best 1 actual dip trade | 672 | 96,296 | -20,056 |
| Remove best 3 | -5,496 | 90,128 | -20,056 |
| Remove best 5 | -8,560 | 87,064 | -20,056 |
| Remove best 10 | -11,400 | 84,224 | -20,056 |
| Winsorise contribution at AUD 2,000 | 4,384 | 99,528 | -20,056 |
| Winsorise contribution at AUD 4,000 | 5,728 | 99,528 | -20,056 |

The paired gap remains unchanged when the same observations are removed from both paths, as it should. The contribution analysis is more informative: a few successful dip trades account for the entire positive weak-branch contribution, and after removing three to five of them the branch is negative. Leave-one-quarter contributions are -1,552, +4,072, +9,704, and -512 AUD; leave-one-half contributions are -1,384 and +5,288 AUD. This is not a broad, stable edge.

## 6. Chronological evidence

Absolute P&L is positive for Candidate C and the one-sided primary in every fixed quarter, but the paired comparison is consistently worse outside the first segment.

| Segment | Candidate C | One-sided primary | Primary minus C |
|---|---:|---:|---:|
| Q1 | 1,416 | 2,760 | +1,344 |
| Q2 | 28,688 | 24,088 | -4,600 |
| Q3 | 59,672 | 52,896 | -6,776 |
| Q4 | 29,808 | 19,784 | -10,024 |
| First half | 30,104 | 26,848 | -3,256 |
| Second half | 89,480 | 72,680 | -16,800 |
| Early third | — | — | +6,872 paired |
| Middle third | — | — | -13,336 paired |
| Late third | — | — | -13,592 paired |

For non-overlapping approximately 60-day blocks, the primary beats Candidate C only in B2 (+20,256) and loses in B1, B3, B4, B5, B6, and B7. The paired advantage wins 3 of 16 tested quarter/60-day/half/third summaries, with a maximum drawdown of the daily paired curve of AUD -27,120.

Excluding the final 30, 60, and 91 days gives Candidate C / primary P&Ls of 102,408 / 88,816; 100,432 / 91,032; and 86,496 / 76,464. The overlay does not become preferable when the late rally is removed.

Always-long is disproportionately dependent on the later advance: first-half P&L is AUD 6,616 and second-half P&L is AUD 31,144; excluding the final 91 days leaves AUD 5,552. Candidate C, however, remains ahead of always-long in both halves and retains AUD 86,496 when the final 91 days are excluded. The conclusion is therefore not “the upward drift guarantees always-long”; it is that Candidate C’s convincing negative-slope shorts add more robust value than selectively going flat in weak states.

## 7. Parameter sensitivity and chronological selection

The diagnostic centred grid contains 80 predeclared `(alpha, window, threshold)` configurations and 77 unique position paths. It is not treated as 80 independent validation samples. Across all configurations, median P&L is AUD 100,056, the lower decile is AUD 95,269, and the worst is AUD 89,080. Six cells exceed Candidate C, with the maximum AUD 129,384 at alpha 0.03, window 45, threshold 0.25. That maximum is not promoted.

Grouped medians show a broad direction but not a production plateau above Candidate C:

| Group value | Median P&L | Median increment vs C | Lower decile | Worst |
|---|---:|---:|---:|---:|
| alpha 0.03 | 105,452 | -14,132 | 98,578 | 94,808 |
| alpha 0.06 | 103,124 | -16,460 | 95,817 | 94,232 |
| alpha 0.10 | 99,596 | -19,988 | 96,398 | 95,464 |
| alpha 0.20 | 97,108 | -22,476 | 93,874 | 89,080 |
| window 10 | 99,968 | -19,616 | 95,220 | 94,032 |
| window 20 | 98,080 | -21,504 | 94,556 | 94,200 |
| window 30 | 101,880 | -17,704 | 95,835 | 89,080 |
| window 45 | 107,072 | -12,512 | 97,274 | 92,448 |
| threshold 0 | 110,088 | -9,496 | 99,676 | 96,784 |
| threshold 0.5 | 101,352 | -18,232 | 94,260 | 89,080 |
| threshold 1.0 | 97,272 | -22,312 | 94,520 | 92,448 |

The higher-performing region is low alpha, long window, and low entry threshold; the primary alpha 0.06/W20/0.5 cell is not near the top of a stable improvement plateau. All cells have positive absolute P&L in each quarter, but that is a weak test because the shared positive path makes even inferior policies profitable.

Selecting the best configuration on the first half chose alpha 0.03/window 30/threshold 0.0; its second-half P&L was AUD 82,080 versus Candidate C’s AUD 89,480, a paired loss of AUD 7,400. The two expanding walk-forward selections lost AUD 5,752 and AUD 12,912 to Candidate C in their subsequent blocks. Parameter selection therefore adds estimation noise rather than demonstrated subsequent improvement.

## 8. Additive, drift/residual, and regime-conditioned stresses

The primary bootstrap reconstructs prices from absolute daily changes with circular blocks of length 5, 10, 20, 40, and 60. It does not compound percentage returns or shift the path upward. Paired results are calculated on the same synthetic path.

For one-sided primary minus Candidate C, median paired differences / win fractions were:

| Additive block | Median paired difference | P5 | Worst | Win fraction |
|---:|---:|---:|---:|---:|
| 5 | -6,884 | -31,337 | -53,536 | 34.0% |
| 10 | -8,876 | -32,050 | -47,248 | 25.7% |
| 20 | -8,924 | -30,178 | -44,168 | 26.7% |
| 40 | -14,856 | -40,364 | -55,400 | 20.3% |
| 60 | -18,228 | -41,782 | -56,080 | 18.7% |

In the drift-plus-residual block design, the median primary-minus-C difference is negative under every drift assumption: -9,568 at 100% drift, -9,551 at 75%, -10,307 at 50%, -8,090 at zero, and -8,017 at 125%. The primary’s median win fraction is only 21%–31%.

Regime-conditioned paths show why a one-sided dip overlay can look appealing in a particular sample: primary-minus-C is positive in the persistent-negative scenario (+5,163 median) and higher-noise scenario (+2,388), but negative in persistent-positive (-7,853), short-negative-reversal (-9,102), long-negative (-1,604), balanced (-3,541), gradual (-1,550), and lower-noise (-852) scenarios. These are generator-conditioned stresses, not confidence intervals; they do not support changing the default rule.

The entirely no-short variant improves drawdown in several stresses but sacrifices the negative-trend short edge and loses AUD 60,968 to Candidate C on Round 1. The corrected two-sided hybrid is better than the primary in several long-block stresses, but it has substantially more state complexity and remains a diagnostic rather than a recommendation.

## 9. Family-wise selection bias

The main family contains the 80 centred one-sided grid configurations plus serious core candidates, deduplicated to **85 unique position paths**. It excludes the uncentred diagnostic from the main family decision because that is a scale diagnostic, not the predeclared primary rule.

Using a block-10 circular-change family null with 500 reconstructed paths, the observed family maximum is the corrected two-sided hybrid at AUD 151,584. The null median family maximum is AUD 69,888 and the null 95th percentile is AUD 132,602. The exceedance p-value is 0.01597 with Monte Carlo SE 0.00561, interval [0.00498, 0.02696]. Restricting to plausible positive-price paths gives p=0.00872, SE 0.00502, interval [0, 0.01856]. This is evidence that the observed path contains selectable structure somewhere in the researched family; it is not evidence that the one-sided overlay generalises, and no resampling adjustment creates a second unseen year.

Candidate C and its Kalman parameters were also selected after observing Round 1. The fair conclusion is not that C is unbiased, but that the new EMA family does not supply a clear subsequent-block or paired-stress improvement that would justify its extra selection degrees of freedom.

## 10. Correctness, notional, and portfolio interaction

All saved correctness rows pass:

- integer positions: pass for every candidate;
- Jeans limit: pass, with absolute position never above 800;
- day zero flat: pass;
- P&L uses the prior desired position for the next observed change: pass;
- final-day position does not generate a nonexistent future return: pass;
- prefix replay matches the full replay on audit days 0, 30, 120, 240, and 300: pass;
- perturbing every price after each audit day leaves that day’s position unchanged: pass;
- the EMA residual scale excludes today’s residual and matches prefix replay: pass.

The largest standalone Jeans notional is AUD 75,824 (800 times the AUD 94.78 path maximum), or 12.64% of the AUD 600,000 gross portfolio limit. A prior unchanged full-portfolio replay with full-limit Jeans exposure peaks at AUD 555,931, leaving AUD 44,069 headroom and producing zero budget breaches. This is portfolio interaction evidence, not a claim that all other instruments or a different Round 2 path will have the same headroom. A strategy that goes flat in weak states uses less capital at times, but its observed incremental edge is negative; capital efficiency alone does not justify it.

## 11. Exact recommended Jeans rule

Submit Candidate C unchanged:

1. On day zero, return position 0.
2. Run the causal local-linear Kalman filter with `q_level=0.5`, `q_slope=0.05`, observation variance `R=5.0`, and initial slope variance 1.0.
3. Compute `slope_z = filtered_slope / filtered_slope_uncertainty`.
4. Return -800 when `slope_z <= -0.6`; otherwise return +800.
5. Do not use EMA residuals, EMA hysteresis, or EMA-based shorts.

This remains a symmetric long/short position rule at the Kalman level, but it is structurally long-biased because the default state is +800 and only a convincingly negative filtered slope causes a short. No-short long/flat is not supported by this audit.

## 12. Deliverables and reproducibility

The reproducible module is [jeans_one_sided_ema_audit.py](<D:/Documents/Algojam/research/thrifted_jeans/jeans_one_sided_ema_audit.py>). The executed zero-error notebook is [thrifted_jeans_one_sided_ema_audit.ipynb](<D:/Documents/Algojam/research/thrifted_jeans/thrifted_jeans_one_sided_ema_audit.ipynb>).

The full report outputs are in [one_sided_outputs](<D:/Documents/Algojam/research/thrifted_jeans/one_sided_outputs>) and figures are in [one_sided_figures](<D:/Documents/Algojam/research/thrifted_jeans/one_sided_figures>). Principal files include:

- [exact_candidate_comparison.csv](<D:/Documents/Algojam/research/thrifted_jeans/one_sided_outputs/exact_candidate_comparison.csv>)
- [weak_state_residual_buckets.csv](<D:/Documents/Algojam/research/thrifted_jeans/one_sided_outputs/weak_state_residual_buckets.csv>)
- [corrected_best_trade_exclusions.csv](<D:/Documents/Algojam/research/thrifted_jeans/one_sided_outputs/corrected_best_trade_exclusions.csv>)
- [parameter_sensitivity.csv](<D:/Documents/Algojam/research/thrifted_jeans/one_sided_outputs/parameter_sensitivity.csv>)
- [additive_bootstrap_paired.csv](<D:/Documents/Algojam/research/thrifted_jeans/one_sided_outputs/additive_bootstrap_paired.csv>)
- [drift_residual_paired.csv](<D:/Documents/Algojam/research/thrifted_jeans/one_sided_outputs/drift_residual_paired.csv>)
- [regime_preserving_multi_path.csv](<D:/Documents/Algojam/research/thrifted_jeans/one_sided_outputs/regime_preserving_multi_path.csv>)
- [familywise_diagnostics.csv](<D:/Documents/Algojam/research/thrifted_jeans/one_sided_outputs/familywise_diagnostics.csv>)
- [correctness_checks.csv](<D:/Documents/Algojam/research/thrifted_jeans/one_sided_outputs/correctness_checks.csv>)

The notebook was executed after the final deterministic and stress-output corrections; it completed with zero error cells. Production files remain unchanged.
