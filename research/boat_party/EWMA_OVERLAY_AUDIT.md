# Boat Party Ticket: adaptive EWMA overlay audit

## Verdict

**Conclusion A — keep frozen Candidate D. Do not add the adaptive EWMA overlay to the production strategy.**

The reported approximately **AUD 190,000** does not reproduce from the only Boat Party signal available in this repository under the described causal mechanics:

| Strategy | Round 1 P&L | Max drawdown | Active days | Trades | Turnover units |
|---|---:|---:|---:|---:|---:|
| Frozen Candidate D | AUD 92,560 | -AUD 2,460 | 333 | 92 | 134,000 |
| Adaptive EWMA, alpha 0.65 | AUD 101,830 | -AUD 3,080 | 354 | 74 | 132,000 |
| Adaptive EWMA, alpha 0.90 | AUD 97,050 | -AUD 2,460 | 343 | 77 | 122,000 |

The adaptive overlay is mechanically causal and has suggestive in-sample reversal evidence, but the incremental improvement is modest, timing-fragile, concentrated in a small overlay sample, and not independently validated. The safest unseen-Round-2 handoff is the already frozen Candidate D.

The repository does **not** contain a literal `BOAT_PARTY_SIGNALS`, `BOAT_VOL_WINDOW`, or teammate adaptive implementation. It contains `BOAT_PARTY_SEMESTER_SIGNALS` in `trader_interface/algorithm.py`. This audit extracts that string read-only; it is length 322 and exactly reproduces the frozen 5/7/11 majority signal through day 321. Therefore the AUD 190,000 claim cannot be reproduced *exactly as submitted* without the missing teammate source. The numbers above are the reproducible implementation of the supplied behavioural specification and available frozen signal, and this limitation is material.

## Exact audited mechanics

The audited adaptive configurations are predeclared alpha 0.65 and 0.90, `BOAT_VOL_WINDOW = 10`, price-level rolling standard deviation with `ddof=0`, and `BOAT_REVERT_THRESHOLD = 0.05`.

For each observed day `t`, the research implementation does the following causally:

```text
ewma_old = previous EWMA, initialised to price[0]
ewma_new = ewma_old + alpha * (price[t] - ewma_old)
deviation = price[t] - ewma_new
vol = std(price[t-9:t+1]) when ten observed prices exist
z = deviation / vol

if signal[t] == '+': desired_position = +1000
elif signal[t] == '-': desired_position = -1000
else:
    if previous_position == 0:
        z >= 0.05 -> -1000
        z <= -0.05 -> +1000
        otherwise -> 0
    elif previous_position * z > 0:
        abs(z) >= 0.05 -> reverse to -previous_position
        otherwise -> 0
    else:
        desired_position = previous_position

position[364] = 0
pnl[t] = position[t] * (price[t+1] - price[t])
```

The position is a desired holding, not a trade quantity. A reversal therefore has 2,000 units of turnover while the holding remains within ±1,000. The available fixed string ends before day 322; the adaptive implementation consequently applies its EWMA rule on zero/summer days. Candidate D instead uses its frozen AUD 45 rule from day 322.

The algebra audit confirms:

```text
price[t] - ewma_new = (1 - alpha) * (price[t] - ewma_old)
```

Thus alpha changes the measured post-update deviation and the effective threshold. A post-update threshold of 0.05 is equivalent to a pre-update threshold of 0.05/(1-alpha): 0.142857 for alpha 0.65 and 0.50 for alpha 0.90. The research output shows these equivalent formulations reproduce identical positions. Holding the numeric 0.05 threshold fixed on the pre-update deviation is not an apples-to-apples alpha comparison; it corresponds to effective post-update thresholds 0.0175 and 0.0050.

## Where the observed P&L comes from

The available fixed signal alone produces AUD 80,010. Candidate D is exactly that signal through day 321 plus AUD 12,550 from the AUD 45 summer rule.

| Attribution | Alpha 0.65 | Alpha 0.90 |
|---|---:|---:|
| Fixed `+/-` signal days | AUD 80,010 | AUD 80,010 |
| Neutral zero days traded by EWMA | AUD 9,100 | AUD 7,240 |
| Summer zero days traded by EWMA | AUD 12,720 | AUD 9,800 |
| Full-year adaptive total | **AUD 101,830** | **AUD 97,050** |
| Increment over noisy signal held flat | **AUD 21,820** | **AUD 17,040** |
| Increment over frozen Candidate D | **AUD 9,270** | **AUD 4,490** |

The incremental alpha-0.65 P&L is positive in Semester 1 (+AUD 2,930), Semester 2 (+AUD 4,420), and summer (+AUD 12,720), but the six 60-day blocks are +590, +3,050, -710, +2,690, +720, and +14,700 AUD. Alpha 0.90 has the same pattern with a -AUD 50 block and +AUD 11,580 in block 5. The overlay therefore does not depend on one best day, but a large part of its incremental edge arrives late in the year.

The best full-strategy day is day 181 at AUD 6,710 and the worst is day 57 at -AUD 1,360 for both alpha values. For the incremental overlay, the best single-day contribution is AUD 1,240 (day 355); the best 1/5/10/20 incremental days contribute:

| Strategy | Best 1 | Best 5 | Best 10 | Best 20 |
|---|---:|---:|---:|---:|
| Alpha 0.65 | 5.7% | 27.1% | 50.2% | 85.3% |
| Alpha 0.90 | 7.3% | 34.6% | 62.2% | 103.1% |

The best-20 alpha-0.90 share exceeding 100% means other incremental days are net negative. This is concentration risk even though the full strategy's best single day is not dominant.

The available signal has 291 nonzero days, 65 adjacent character changes, and 23 direction changes among nonzero characters. It is not a distinct noisy signal relative to the frozen majority in this repository: it matches Candidate B before summer exactly. The AUD 190,000 result therefore cannot be attributed here to a higher-frequency signal string that Candidate D removed.

## Causal predictive test

On the 31 eligible neutral semester observations, the regression is `return[t+1] = intercept + beta * post_update_deviation[t] + error[t+1]`, with HAC/Newey-West inference and no use of `return[t+1]` in the decision.

| Configuration/sample | Beta | HAC SE | 95% CI | Rank correlation | Directional hit rate |
|---|---:|---:|---:|---:|---:|
| Alpha 0.65, all neutral | -2.614 | 0.558 | [-3.707, -1.520] | -0.633 | 64.5% |
| Alpha 0.65, excluding best 10 neutral days | -1.083 | 0.454 | [-1.973, -0.193] | -0.312 | 47.6% |
| Alpha 0.90, all neutral | -7.258 | 1.809 | [-10.803, -3.713] | -0.607 | 61.3% |
| Alpha 0.90, excluding best 10 neutral days | -2.631 | 2.088 | [-6.723, 1.462] | -0.191 | 42.9% |

Semester-level beta estimates remain negative, but the sample is only 18 observations in Semester 1 and 11 in Semester 2. The sign is consistent with mean reversion in-sample, yet the alpha-0.90 inference loses significance after removing the ten best neutral strategy days, and alpha-0.65's directional hit rate falls below 50%. This is evidence of a Round 1 relationship, not a reliable independent validation result.

Simple comparisons do not support a large adaptive advantage: noisy signal flat neutral is AUD 80,010; reverse latest return on neutral is AUD 97,320; causal MA20 on neutral is AUD 82,900; EWMA-only is AUD 83,150 with an AUD 8,860 drawdown; and summer-only AUD 45 reversion is AUD 12,550.

## Timing, sensitivity, and denominator diagnostics

The displacement convention is `+d` = use the overlay position from day `t-d`, so positive displacement delays the signal relative to the price path. Fixed `+/-` signal days stay at their original indices; only the EWMA overlay is displaced.

| Overlay displacement | Alpha 0.65 P&L | Alpha 0.90 P&L |
|---:|---:|---:|
| -3 | AUD 86,310 | AUD 85,070 |
| -2 | AUD 81,490 | AUD 84,120 |
| -1 | AUD 58,640 | AUD 59,010 |
| 0 | **AUD 101,830** | **AUD 97,050** |
| +1 | AUD 80,580 | AUD 81,210 |
| +2 | AUD 73,800 | AUD 74,400 |
| +3 | AUD 86,250 | AUD 85,420 |

Worst P&L is the -1-day displacement for both configurations. Relative to zero shift, the loss is 42.4% for alpha 0.65 and 39.2% for alpha 0.90. The ±3-day median P&Ls are AUD 81,490 and AUD 84,120 respectively. This timing fragility is a major reason not to spend shared portfolio budget on the overlay.

The predeclared diagnostic grid contains 5 alphas × 4 windows × 5 thresholds = 100 combinations. Total P&L is positive across the grid because the fixed signal contributes AUD 80,010. Incremental P&L versus that fixed signal has the following alpha ranges across windows and thresholds:

| Alpha | Minimum incremental | Median incremental | Maximum incremental |
|---:|---:|---:|---:|
| 0.30 | AUD 13,760 | AUD 21,175 | AUD 22,310 |
| 0.50 | AUD 11,680 | AUD 18,710 | AUD 20,680 |
| 0.65 | AUD 8,390 | AUD 19,750 | AUD 21,820 |
| 0.75 | AUD 6,750 | AUD 16,380 | AUD 19,250 |
| 0.90 | AUD 4,220 | AUD 12,500 | AUD 18,170 |

This is a broad positive Round 1 region, but not evidence of out-of-sample stability: the whole grid is evaluated on the same year and the alpha-0.90 edge is materially smaller. No grid maximum was selected as a strategy, and no formal multiple-testing correction is claimed.

Alternative causal denominator diagnostics are close: alpha 0.65 gives AUD 101,830 with price-level volatility, AUD 101,360 with return volatility, and AUD 100,570 with EWMA-residual volatility. Alpha 0.90 gives AUD 97,050, AUD 97,000, and AUD 97,510 respectively. The denominator choice does not explain AUD 190,000.

## Resampling, placebo, and summer stress

The moving-block bootstrap resamples Round 1 returns in blocks of 5, 10, and 20 days and reconstructs synthetic price paths. The pooled 600-path results are generator-conditioned stress diagnostics, not confidence intervals:

| Strategy | Median P&L | P10 | Worst | Positive-path rate | Median max drawdown |
|---|---:|---:|---:|---:|---:|
| Frozen Candidate D | AUD 55 | -AUD 20,941 | -AUD 42,020 | 50.0% | -AUD 17,755 |
| Noisy signal flat neutral | AUD 1,120 | -AUD 19,419 | -AUD 45,960 | 51.8% | -AUD 16,220 |
| Adaptive alpha 0.65 | AUD 14,870 | -AUD 10,292 | -AUD 33,600 | 77.2% | -AUD 15,660 |
| Adaptive alpha 0.90 | AUD 13,320 | -AUD 10,423 | -AUD 34,850 | 76.3% | -AUD 15,425 |

When eligible neutral-day overlay signs are randomized while activity is preserved, the median results fall to AUD 80,030 (alpha 0.65) and AUD 79,640 (alpha 0.90), close to the AUD 80,010 fixed signal. This placebo removes the incremental advantage, but it is still an in-sample activity-preserving placebo, not a second-year test.

The fixed-seed gradual summer stress uses AUD 43, 45, and 47 equilibria, 7/14/28-day transitions, and 100 paired residual paths per scenario. Pooled full-year results are:

| Strategy | Median P&L | P10 | Worst | Median summer P&L |
|---|---:|---:|---:|---:|
| Frozen Candidate D | AUD 58,455 | AUD 46,210 | AUD 29,460 | AUD 7,913 |
| Adaptive alpha 0.65 | AUD 71,551 | AUD 56,196 | AUD 43,029 | AUD 12,946 |
| Adaptive alpha 0.90 | AUD 67,534 | AUD 53,767 | AUD 44,810 | AUD 11,034 |

These synthetic results favour the adaptive overlay in this generator, including under gradual transitions, but they are not empirical validation. The worst individual adaptive full-year result is AUD 43,029 for alpha 0.65 at target AUD 43 with a 7-day transition; alpha 0.90's worst is AUD 44,810 at target AUD 43 with a 28-day transition. Paired adaptive-minus-D differences are positive in most synthetic paths but have worst cases of about -AUD 15,270 for alpha 0.65 and -AUD 22,244 for alpha 0.90.

## Mechanics audit

All 16 research correctness checks pass. They cover:

- `position[t]` uses only `template/signal[t]` or prices observed through `t`;
- P&L is exactly `position[t] * (price[t+1] - price[t])`;
- no day-364 return is included and day 364 is flat;
- the EWMA update precedes the deviation test in the audited implementation;
- future-price perturbation after day 120 does not change earlier positions;
- the extracted string has length 322 and only `+`, `-`, and `0`;
- neutral EWMA startup is flat before ten observations;
- positions are integral and within ±1,000;
- standalone Boat notional is at most AUD 55,390, far below the AUD 600,000 portfolio cap;
- the centered seasonal object is explicitly labelled a Round 1 fixed prior, not a causal Round 2 fit;
- no calendar warp, RLS, Kalman, OU, Fourier, or absolute seasonal-level forecast is used.

The centered smoothing used to make the fixed template is intentional Round 1 look-ahead in the research construction, because the complete Round 1 path is frozen before a hypothetical Round 2 run. It is not a causal same-year trading signal.

## Production handoff recommendation

Keep the existing frozen Candidate D:

1. Days 0–321: fixed 5/7/11-day centered Round 1 majority template; each template uses one-day slope `template[t+1] - template[t]` and independent AUD 0.02 deadbands; trade ±1,000 when at least two voters agree, otherwise flat.
2. Days 322–363: current observed price below AUD 45 → +1,000; above AUD 45 → -1,000; exactly AUD 45 → flat.
3. Day 364: flat.
4. Position limit: ±1,000; maximum observed standalone Boat notional: AUD 55,390.

No calendar warp, RLS, OU, or absolute seasonal-level forecast is used. The adaptive overlay remains a research diagnostic only. No production file was changed.

## Principal outputs

- [Reproducible audit module](ewma_overlay_audit.py)
- [Executed notebook](analysis.ipynb)
- [Strategy comparison](results/ewma_strategy_comparison.csv)
- [P&L attribution](results/ewma_pnl_attribution.csv)
- [Chronological splits](results/ewma_chronological_splits.csv)
- [Parameter sensitivity](results/ewma_parameter_sensitivity.csv)
- [Predictive regressions](results/ewma_predictive_regressions.csv)
- [Concentration diagnostics](results/ewma_concentration.csv)
- [Bootstrap/placebo detail](results/ewma_bootstrap_placebo.csv)
- [Bootstrap/placebo summary](results/ewma_bootstrap_placebo_summary.csv)
- [Gradual summer stress detail](results/ewma_summer_stress.csv)
- [Gradual summer stress summary](results/ewma_summer_stress_summary.csv)
- [Correctness checks](results/ewma_correctness_checks.csv)
- [Alpha/threshold mechanics](results/ewma_alpha_threshold_mechanics.csv)
- [Volatility denominator comparison](results/ewma_denominator_comparison.csv)
- [Overfitting diagnostics](results/ewma_overfitting_diagnostics.csv)
- [Audit figures](figures/ewma_overlay_cumulative_pnl.png), [segment P&L](figures/ewma_segment_pnl.png), [sensitivity](figures/ewma_parameter_sensitivity.png), and [deviation/next-return plot](figures/ewma_deviation_next_return.png)
