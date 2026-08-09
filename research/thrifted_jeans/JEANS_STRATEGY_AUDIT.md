# Thrifted Jeans strategy audit

Status: research-only. No production file was changed. The executed notebook and the Python module are the source of all tables and figures in this report.

## Executive verdict

The instrument description implies a noisy, positively drifting price: a pair of jeans bought near $40 may resell near $90 a year later. That is a prior for holding a long position, not a guarantee that every local move continues upward.

The original hybrid does reproduce its Round 1 result, but its extra P&L is not a safe basis for Round 2 selection. Its corrected causal version still has a large Round 1 result, yet:

- the corrected EMA-only component loses money;
- the corrected weak-trend EMA regression is negative but statistically uncertain and disappears after removing the five best next-change days;
- the corrected hybrid is less reliable than simple K2 on the reconstructed price-path bootstrap;
- its apparent advantage is materially reduced by the corrected residual calculation;
- all results are from one visible year, and the family search was conducted after seeing that year.

Recommended submission: retain the simple K2 Kalman strategy already present in production. Start flat on day 0; from day 1, use Q_LEVEL=1.0, Q_SLOPE=0.05, observation variance 9.0, initial slope variance 1.0, and short only when the filtered slope is below minus one filtered slope standard error. Otherwise hold +800. Do not add the EMA switch. This is a robust Round 1 choice, not a claim of true out-of-sample validation.

## 1. Data, structural prior, and timing

The audit uses the supplied Round 1 Thrifted Jeans CSV: 365 observations, starting at $40.00 and ending at $87.67. Daily price-change autocorrelation is -0.0505, so one-day continuation is weak. The description supports medium-horizon persistence and a positive long-run prior, but it does not promise that the upward drift repeats unchanged in Round 2.

The simulator convention is:

    pnl[t] = position[t - 1] * (price[t] - price[t - 1])

for t >= 1. A position selected on decision day t earns only the next day’s change. Day zero has no local P&L, and the final position has no subsequent Round 1 realization. Positions in this audit are desired integer positions, not trade quantities. Turnover is the absolute change in desired position, including the initial transition from zero.

There are no transaction costs in the supplied simulator. Every Jeans candidate is limited to ±800 units. The maximum standalone Jeans notional observed is AUD 75,824, which is 12.6% of the AUD 600,000 gross portfolio limit. This Jeans-only audit does not allocate the remaining portfolio budget; a previous unchanged full-portfolio replay peaked at AUD 555,931 with full-limit Jeans and had zero budget breaches.

The simulator rounds daily instrument P&L to cents. Because positions are integral and the supplied prices have cent precision, the research calculation agrees with that convention on all correctness checks.

## 2. Reproduction and candidate comparison

The exact original hybrid mechanics were taken from the research-only trader_interface/algorithm_juan file. In strong Kalman states it uses the trend direction. In weak states it uses the original EMA residual z-score and its stateful entry, holding, flatting, and reversal rules. A direct replay gives:

- simple K2 Kalman: $100,016;
- original hybrid: $162,496;
- always-long reference: $38,136.

The original hybrid P&L therefore reproduces exactly. The corrected hybrid is a separate implementation; it does not silently replace the original.

| Candidate | P&L | Incremental vs always-long | Sharpe | Active | Long / short / flat days | Turnover | Max drawdown | Hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Flat | $0 | -$38,136 | 0.00 | 0 | 0 / 0 / 365 | 0 | $0 | n/a |
| Always-long +800 | $38,136 | $0 | 0.90 | 365 | 365 / 0 / 0 | 800 | -$44,856 | 54.7% |
| Simple Kalman K2 uncertainty-short | $100,016 | $61,880 | 2.38 | 364 | 291 / 73 / 1 | 29,600 | -$26,672 | 55.9% |
| Kalman trend-only symmetric | $49,920 | $11,784 | 1.18 | 364 | 215 / 149 / 1 | 52,000 | -$64,152 | 50.4% |
| Original EMA mean reversion | $6,848 | -$31,288 | 0.17 | 343 | 140 / 203 / 22 | 50,400 | -$64,024 | 51.5% |
| Corrected EMA, prior volatility | -$3,488 | -$41,624 | -0.08 | 344 | 140 / 204 / 21 | 45,600 | -$67,832 | 50.4% |
| Corrected EMA, through-current volatility | -$928 | -$39,064 | -0.02 | 345 | 141 / 204 / 20 | 47,200 | -$67,832 | 50.6% |
| Original hybrid | $162,496 | $124,360 | 3.98 | 353 | 199 / 154 / 12 | 58,400 | -$14,536 | 57.7% |
| Corrected hybrid, prior volatility | $151,584 | $113,448 | 3.70 | 351 | 196 / 155 / 14 | 53,600 | -$14,536 | 56.6% |
| Corrected hybrid, through-current volatility | $154,144 | $116,008 | 3.76 | 352 | 197 / 155 / 13 | 55,200 | -$14,536 | 56.7% |
| 20-day momentum, symmetric | $35,352 | -$2,784 | 0.88 | 345 | 195 / 150 / 20 | 55,200 | -$36,616 | 52.9% |
| 20-day momentum, long/flat | $31,640 | -$6,496 | 1.07 | 195 | 195 / 0 / 170 | 28,000 | -$23,848 | 55.7% |

All serious candidates use the same maximum Jeans notional of AUD 75,824 when their position reaches ±800. The complete metric set, including best/worst individual days, losing streaks, long/short attribution, and concentration shares, is in candidate_comparison.csv.

The hybrid’s P&L is not simply a final-rally artifact. For the original hybrid, long exposure contributed $94,880 and short exposure contributed $67,616. For the corrected hybrid, the corresponding contributions were $90,792 and $60,792. Relative to always-long, the original hybrid’s incremental result decomposes into about $56,744 from better long/flat timing and $67,616 from short exposure; the corrected version decomposes into about $52,656 and $60,792. Shorting helped this Round 1 path, but it is not independently validated for Round 2.

## 3. Original versus corrected EMA calculation

The original calculation uses the final EMA value for every historical residual:

    residuals = every historical price - final EMA value

The corrected implementation instead compares price[t] with the EMA through t-1, stores that causal deviation, and uses only prior deviations for the preferred volatility estimate. A through-current version is also reported.

The correction materially changes the standalone signal: original EMA mean reversion makes $6,848, while corrected prior-volatility EMA mean reversion loses $3,488. In the hybrid, the correction reduces P&L from $162,496 to $151,584, a $10,912 reduction. The strong-trend days are identical in the original and corrected hybrids; the entire difference comes from the weak-state branch. Therefore the bad residual calculation inflated the hybrid result, but it did not create the whole apparent edge.

The corrected EMA-only candidates have negative P&L, negative or near-zero Sharpe, high turnover, and worse drawdowns than simple K2. This is not evidence for a standalone mean-reversion overlay.

## 4. Chronological and disjoint-period evidence

The delayed or segmented results are diagnostics within one year, not independent out-of-sample annual tests. The boundaries below were fixed chronological quarters, halves, and thirds; they were not selected to maximize P&L.

| Candidate | Q1 | Q2 | Q3 | Q4 | H1 | H2 | Excluding final 30 | Excluding final 60 | Excluding final 91 | After removing best 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always-long | $4,688 | $2,304 | $2,216 | $28,928 | $6,992 | $31,144 | $20,960 | $20,216 | $5,928 | $10,424 |
| Simple K2 | -$3,784 | $26,496 | $54,856 | $22,448 | $22,712 | $77,304 | $82,840 | $88,576 | $74,288 | $68,592 |
| Original hybrid | $33,128 | $39,592 | $71,272 | $18,504 | $72,720 | $89,776 | $161,240 | $156,760 | $140,712 | $131,072 |
| Corrected hybrid | $26,600 | $39,592 | $71,272 | $14,120 | $66,192 | $85,392 | $153,800 | $150,232 | $134,184 | $120,160 |

Always-long is disproportionately dependent on the final rally: Q4 supplies 75.8% of its full-year P&L, and excluding the final 91 days leaves only $5,928. Removing its five best days leaves $10,424. The hybrid is much less dependent on that final rally: the corrected hybrid still makes $134,184 before the final 91 days and retains $120,160 after its five best strategy days are removed. Simple K2 is also stable after the final rally and retains $68,592 after removing its five best days.

The hybrid is positive in all four quarters, while simple K2 loses Q1 but is positive in Q2-Q4. This is useful descriptive evidence, but the hybrid’s advantage remains a single-year observation and should not outweigh the corrected predictive and path-bootstrap weaknesses.

## 5. Hybrid state attribution

The fixed hybrid state definitions produce 272 strong-trend decision days and 92 weak-trend/mean-reversion decision days before the final next-day realization. Strong states average 11.3 days per run; weak states average 3.7 days per run.

| Candidate | State | Count | Mean duration | Mean next-day change | Always-long P&L | Strategy P&L | Long P&L | Short P&L |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original hybrid | Strong trend | 272 | 11.3 | $0.063 | $13,800 | $95,624 | $54,712 | $40,912 |
| Original hybrid | Weak / mean reversion | 92 | 3.7 | $0.331 | $24,336 | $66,872 | $40,168 | $26,704 |
| Original hybrid | State transition days | 48 | n/a | -$0.104 | -$4,008 | $35,368 | $15,928 | $19,440 |
| Corrected hybrid | Strong trend | 272 | 11.3 | $0.063 | $13,800 | $95,624 | $54,712 | $40,912 |
| Corrected hybrid | Weak / mean reversion | 92 | 3.7 | $0.331 | $24,336 | $55,960 | $36,080 | $19,880 |
| Corrected hybrid | State transition days | 48 | n/a | -$0.104 | -$4,008 | $35,368 | $15,928 | $19,440 |

The weak state has a positive unconditional next-day mean, not a negative one. Its apparent value comes from conditional deviations, stateful transitions, and short exposure rather than a broad weak-state drift. The corrected weak-state P&L is $10,912 lower than the original, exactly matching the hybrid correction gap.

## 6. Predictive tests

### Kalman slope

The K2 slope z-score buckets do not show a smooth monotonic forecast relationship:

| Slope z bucket | Observations | Mean next-day change | Positive fraction | Slope-following hit |
|---|---:|---:|---:|---:|
| Below -1.0 | 73 | -$0.533 | 46.6% | 53.4% |
| -1.0 to -0.6 | 25 | $0.458 | 64.0% | 36.0% |
| -0.6 to 0 | 51 | $0.389 | 62.7% | 37.3% |
| 0 to 0.6 | 74 | -$0.180 | 47.3% | 45.9% |
| 0.6 to 1.0 | 33 | $0.719 | 57.6% | 57.6% |
| Above 1.0 | 108 | $0.415 | 58.3% | 58.3% |

The strongest support for simple K2 is the negative mean in the below -1.0 bucket. Intermediate negative slope buckets reverse on average, and positive buckets are mixed. The z-score is a filtered-state confidence score, not an independently calibrated t-statistic.

### Causal EMA deviation

The regression is next-day price change on the causal EMA deviation z-score. A negative coefficient supports mean reversion.

| Signal / scope | N | Beta | HAC SE | 95% CI | Rank correlation | Reversal hit | Normal p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original EMA, all | 364 | $+0.028 | $0.147 | [-$0.261, $0.317] | 0.001 | 51.6% | 0.850 |
| Original EMA, weak trend | 92 | -$1.133 | $0.455 | [-$2.025, -$0.242] | -0.365 | 63.0% | 0.013 |
| Corrected EMA, all | 358 | $+0.069 | $0.108 | [-$0.142, $0.281] | 0.028 | 51.1% | 0.522 |
| Corrected EMA, weak trend | 86 | -$0.817 | $0.551 | [-$1.897, $0.262] | -0.287 | 61.6% | 0.138 |
| Corrected EMA, excluding best 5 next-change days | 353 | $+0.030 | $0.101 | [-$0.168, $0.228] | 0.018 | 51.6% | 0.766 |

The corrected weak-trend coefficient has the expected sign but its HAC interval includes zero. The negative relationship is not stable after removing the five most profitable next-change days. This fails the requirement for demonstrable robust weak-trend mean reversion.

## 7. Parameter sensitivity

The grids were predeclared for diagnosis and were not used to promote the maximum cell. The audit contains 12 base candidates and 69 sensitivity configurations across Kalman and original/corrected hybrid families.

| Family | Configurations | Median P&L | Lower decile | Worst | Best | Median incremental | Median positive quarters |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kalman q_level / q_slope | 12 | $80,976 | $69,126 | $68,384 | $101,904 | $42,840 | 3 |
| Kalman observation variance | 4 | $82,808 | $78,795 | $77,504 | $100,016 | $44,672 | 3 |
| Kalman confidence gate | 5 | $87,984 | $78,867 | $73,824 | $100,016 | $49,848 | 3 |
| Original hybrid alpha / reversion | 16 | $131,772 | $101,304 | $93,080 | $162,496 | $93,636 | 4 |
| Corrected hybrid alpha / reversion | 16 | $131,184 | $96,204 | $89,216 | $153,488 | $93,048 | 4 |
| Original hybrid trend threshold | 5 | $131,048 | $118,120 | $113,272 | $162,496 | $92,912 | 4 |
| Corrected hybrid trend threshold | 5 | $117,816 | $104,370 | $101,736 | $151,584 | $79,680 | 3 |
| Original hybrid short confidence | 3 | $128,576 | $125,210 | $124,368 | $162,496 | $90,440 | 4 |
| Corrected hybrid short confidence | 3 | $117,184 | $113,818 | $112,976 | $151,584 | $79,048 | 4 |

There is a broad positive Round 1 plateau for both hybrid families, but the corrected hybrid is consistently lower. The corrected grid does not collapse to one exact cell, so the result is not only a single-point numerical optimum. That stability is not enough to overcome the lack of robust corrected EMA predictive evidence and the weaker reconstructed-path performance.

## 8. Price-path bootstrap, placebo, and stress tests

The primary bootstrap is not the old fixed-P&L bootstrap. For each block length 5, 10, and 20, it resamples circular blocks of daily percentage returns, reconstructs a synthetic path from the original $40 start, reruns every candidate causally, and compares each candidate with always-long on the same path. The fixed realised-P&L block bootstrap is saved separately as a diagnostic only and is not used for the verdict.

| Block | Candidate | P(P&L > 0) | P(beats always-long) | Median P&L | 5th P&L | Median incremental | 5th incremental |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | Simple K2 | 62.4% | 34.4% | $14,636 | -$47,863 | -$11,200 | -$228,193 |
| 5 | Original hybrid | 52.6% | 33.0% | $2,612 | -$96,372 | -$24,207 | -$313,031 |
| 5 | Corrected hybrid | 54.0% | 33.6% | $3,662 | -$90,748 | -$23,439 | -$311,212 |
| 10 | Simple K2 | 74.2% | 51.6% | $32,441 | -$31,894 | $2,238 | -$176,521 |
| 10 | Original hybrid | 69.2% | 46.2% | $23,290 | -$60,683 | -$5,202 | -$257,665 |
| 10 | Corrected hybrid | 67.0% | 45.8% | $22,317 | -$60,004 | -$4,838 | -$262,744 |
| 20 | Simple K2 | 81.2% | 61.8% | $55,524 | -$24,049 | $11,327 | -$240,616 |
| 20 | Original hybrid | 76.6% | 52.2% | $41,353 | -$49,016 | $3,088 | -$352,073 |
| 20 | Corrected hybrid | 76.8% | 52.4% | $42,604 | -$59,336 | $3,404 | -$371,617 |

The path bootstrap favors simple K2 over both hybrids on positive-path probability and probability of beating always-long for all three block sizes. Its 5th-percentile incremental P&L is still very negative, so this is not a confidence interval and cannot recreate unseen structural regimes.

Signal timing placebos support causal alignment: delaying the original hybrid by 1, 2, and 3 days reduces P&L to $105,656, $84,384, and $80,864. The corrected hybrid falls to $104,520, $83,312, and $70,592. Shifting signals backward uses future information and is shown only as a placebo; its higher P&L is not evidence.

Generator-conditioned stress paths show model dependence. For example, with 1.5x noise, simple K2 loses $5,945 versus always-long, original hybrid gains $8,530, and corrected hybrid loses $1,157. With volatility halved, original hybrid gains $14,025 while corrected hybrid loses $944. With a positive drift shift, corrected hybrid loses $8,776 versus always-long while simple K2 gains $17,512. These are one-path synthetic scenarios, not statistical intervals.

## 9. Family-wise and overfitting checks

The base catalogue contains 12 candidates. The 69 sensitivity configurations are reported as exploratory surfaces; the family maximum test below is explicitly over the 12 base policies and does not pretend to adjust for every sensitivity cell. All parameters were evaluated after Round 1 was visible, so the full-year maximum is subject to selection bias.

| Family maximum construction | Observed best | Null median | Null 95th | Exceedances / 500 | Monte Carlo p | MC SE |
|---|---:|---:|---:|---:|---:|---:|
| Daily-change permutation, destroys serial structure | $162,496 | $45,452 | $90,540 | 0 | 0.0020 | 0.0020 |
| Circular 10-day return blocks, preserves short-range structure | $162,496 | $75,124 | $134,944 | 4 | 0.0100 | 0.0044 |

The easy permutation null makes the hybrid look unusually strong, but it destroys the persistent local structure the strategy is intended to exploit. The block-preserving family maximum is the more relevant diagnostic; it still puts the observed maximum above the simulated 95th percentile, but it is a resampling diagnostic rather than a true Round 2 probability. Neither result licenses treating the hybrid as robust out of sample.

The main overfitting risks are the short-confidence gate, trend/mean-reversion switch, EMA alpha, reversion threshold, stateful reversal mechanics, and multiple candidate comparisons. The original hybrid has a broad positive Round 1 surface and retains 80.7% of its P&L after its five best strategy days are removed; the corrected hybrid retains 79.2%. That is encouraging concentration-wise, but it does not validate the weak-state mechanism.

## 10. Correctness checks and production decision

For every candidate, the saved correctness table verifies:

- integer positions;
- maximum absolute position no greater than 800;
- maximum standalone gross value no greater than AUD 600,000;
- P&L uses the prior desired position;
- prefix replay matches the full causal run at days 0, 30, 120, 240, and 300;
- perturbing every price after those audit days leaves that day’s position unchanged;
- always-long reproduces $38,136.

No correctness check failed. The executed notebook has zero error outputs.

Exact recommended Jeans rule:

1. On day 0 return 0.
2. On each later day, use only the visible Jeans prices through that day.
3. Initialize the local-linear state at [price[0], 0] and covariance diag([9.0, 1.0]).
4. Use transition matrix [[1, 1], [0, 1]], process covariance diag([1.0, 0.05]), and observation variance 9.0.
5. Perform the causal predict/update recursion through today.
6. Let slope be the filtered slope and uncertainty be sqrt of the filtered slope covariance.
7. Return -800 if slope < -uncertainty; otherwise return +800.
8. Do not fit parameters, use EMA residuals, or use future prices. Keep transaction-cost assumption at zero because that is what the simulator specifies.

This is a symmetric long/short position rule in the sense that it can use ±800, but it is structurally long-biased: only convincingly negative slope evidence triggers the short. If production risk tolerance prefers no shorting, the long/flat variant should be tested separately; this audit does not promote it over the exact simple K2 baseline.

## 11. Deliverables

- Executed notebook: thrifted_jeans_audit.ipynb
- Reproducible validation module: jeans_audit.py
- This report: JEANS_STRATEGY_AUDIT.md
- Candidate table: outputs/candidate_comparison.csv
- Chronological splits: outputs/chronological_splits.csv
- Regime attribution: outputs/regime_attribution.csv
- Slope buckets: outputs/slope_bucket_returns.csv
- EMA regressions: outputs/ema_predictive_regressions.csv
- Parameter surfaces and summaries: outputs/parameter_sensitivity.csv and outputs/parameter_sensitivity_summary.csv
- Concentration: outputs/concentration_diagnostics.csv
- Correctness: outputs/correctness_checks.csv
- Price-path bootstrap: outputs/price_path_bootstrap.csv
- Fixed-P&L diagnostic: outputs/fixed_pnl_bootstrap_diagnostic.csv
- Stress and placebos: outputs/stress_placebo_tests.csv
- Family null: outputs/familywise_null.csv
- Figures: figures/cumulative_pnl_comparison.png, figures/positions_against_price.png, figures/pnl_by_chronological_segment.png, figures/kalman_q_sensitivity_heatmap.png, figures/hybrid_alpha_reversion_heatmap.png, and figures/next_day_returns_by_slope_bucket.png
