# Boat Party Ticket — final fixed-template strategy test

## Verdict

**Select Candidate D for production implementation, subject to the shared portfolio-cap guard.** Candidate D is the frozen smoothing-majority template (5-, 7- and 11-day centred Round 1 templates, equal votes, AUD 0.02 deadband) through day 321, followed by the frozen AUD 45 mean-reversion rule from day 322. This is a fixed day-index seasonal prior, not a calendar-warped model.

The previous report called the third voter “10-day”. The existing helper increments even windows, so that request was actually an 11-point centred smoother. This correction explicitly requests 11, renames the voter `template_11d`, and verifies that the corrected labels reproduce the earlier Candidate B and D positions and P&Ls exactly: B AUD 84,040 and D AUD 92,560.

This selection follows the predeclared rule:

- Candidate A remains the default unless B materially improves timing/noise downside robustness while its zero-shift P&L is within 10% of A. B is 4.78% below A at zero shift, improves the timing-displacement loss and drawdown diagnostics, and improves B's noise-stress P10, worst path and median drawdown relative to A.
- The D summer overlay improves observed performance and the synthetic lower tails at equilibria AUD 43, 45 and 47 relative to B. In the deterministic amplitude family it preserves B's result and remains positive on every path.
- No candidate is rejected for a complete collapse under the tested +/-1-day displacement. This remains an in-sample fragility diagnostic, not independent validation.

The result is not an unbiased Round 2 estimate. The complete Round 1 path is used retrospectively to create an externally fixed Round 2 prior, and the Round 1 score is labelled an **in-sample reconstruction**. Synthetic paths are transparent stress scenarios, not confidence intervals or independent empirical evidence.

## Frozen strategy definition

The selected strategy is Candidate D:

```text
round1 = complete 365-day Boat Party Ticket Round 1 path
templates = centred moving_average(round1, windows=[5, 7, 11])

for day t in 0..363:
    if t >= 322:
        position[t] = +1000 if round2_price[t] < 45.00
                       -1000 if round2_price[t] > 45.00
                          0   if round2_price[t] == 45.00
    else:
        votes = []
        for template in templates:
            expected_change = template[t + 1] - template[t]
            votes.append(+1 if expected_change > +0.02
                         -1 if expected_change < -0.02
                          0 otherwise)
        position[t] = +1000 if at least two votes are +1
                       -1000 if at least two votes are -1
                          0 otherwise

position[364] = 0
daily_pnl[t] = position[t] * (round2_price[t + 1] - round2_price[t])
```

Frozen parameters:

| Parameter | Value |
|---|---:|
| Day alignment | Fixed competition-series indices; no academic-calendar warp |
| Smoothing windows | 5, 7 and 11 days, centred |
| Template forecast | One-day slope, `template[t+1] - template[t]` |
| Deadband | AUD 0.02 per day, applied independently to each voter |
| Voting | Equal-weight majority; at least two of three non-zero votes |
| Position outside deadband | +/-1,000 tickets |
| Summer switch | Day 322, inclusive |
| Summer equilibrium | AUD 45.00 |
| Day 364 | Flat; no access to day 365 |
| Position/capital checks | Integral, within +/-1,000; Boat-only notional below AUD 600,000 |

The selected model therefore does not forecast an absolute next price from the prior year's absolute level. It uses the prior year's fixed timing and only the current Round 2 price for the frozen summer level rule.

## Round 1 candidate comparison

These are same-year, in-sample reconstruction diagnostics. The template source is the complete Round 1 path, retained as the externally fixed Round 2 prior. P&L is AUD; maximum capital is absolute Boat notional.

| Model | P&L | Sharpe | Active-day hit rate | Active days | Max drawdown | Max capital | P&L / max capital |
|---|---:|---:|---:|---:|---:|---:|---:|
| Candidate A | 88,260 | 4.80 | 56.3% | 332 | -5,280 | 55,390 | 1.593 |
| Candidate B | 84,040 | 4.60 | 55.9% | 322 | -3,150 | 55,390 | 1.517 |
| Candidate C | 94,910 | 5.17 | 58.5% | 340 | -5,280 | 55,390 | 1.713 |
| Candidate D | 92,560 | 5.05 | 58.3% | 333 | -2,460 | 55,390 | 1.671 |
| Broad fixed-day schedule | 65,990 | 3.87 | 55.4% | 287 | -3,460 | 55,390 | 1.191 |
| Constant AUD 45 mean reversion | 79,060 | 4.20 | 57.3% | 363 | -10,260 | 55,390 | 1.427 |
| Flat Boat Party position | 0 | 0.00 | — | 0 | 0 | 0 | 0.000 |
| Fixed 7-day template, no deadband (diagnostic) | 87,510 | 4.68 | 56.3% | 364 | -5,280 | 55,390 | 1.580 |

Candidate B is not chosen because it has the highest visible P&L. It is retained because its downside behaviour is more robust in the predeclared timing and residual-noise tests, while its zero-shift P&L is only 4,220 AUD, or 4.78%, below A.

## Correctness and alignment audit

All checks below passed. The deterministic toy series uses template `[0.00, 0.04, 0.04]` and prices `[10.00, 12.00, 99.00]`; it produces positions `[+1000, 0, 0]` and P&L AUD 2,000. A one-day shift or an accidental two-day horizon would fail this assertion.

| Check | Result |
|---|---|
| Signal uses `template[t+1] - template[t]` | PASS |
| P&L uses `position[t] * (price[t+1] - price[t])` | PASS |
| No accidental two-day or three-day horizon | PASS |
| No initialization warm-up/off-by-one error | PASS; day 0 is active in the toy test |
| Day 364 does not access day 365 | PASS; final position is flat |
| Manual P&L equals backtest P&L | PASS; Candidate A AUD 88,260 |
| All positions integral | PASS |
| All positions within +/-1,000 | PASS |
| Boat-only portfolio-cap check | PASS; no AUD 600,000 violations |
| Effective smoothing windows are exactly 5, 7 and 11 | PASS |
| Corrected 5/7/11 labels reproduce legacy B/D positions and P&L | PASS; B AUD 84,040; D AUD 92,560 |
| Round 1 template provenance/evidence label | PASS; in-sample score, fixed Round 2 prior |

The exported 365-row semester schedule also passes exact checks: its majority position equals Candidate B's fixed schedule, it recreates the selected Candidate D position on days 0–321, and day 364 is zero. From day 322 onward Candidate D's position is intentionally runtime-dependent on the current Round 2 price relative to AUD 45 and cannot be fully precomputed.

## Smoothing comparison and signal agreement

Pairwise rates are calculated over days 0–363, the days for which a next-day return exists. `Same vote` includes both flat and active votes; `same non-zero direction` is conditional on both voters being active.

| Comparison | Same vote | Same non-zero direction | Days disagree |
|---|---:|---:|---:|
| 5-day vs 7-day | 65.38% | 74.92% | 126 |
| 5-day vs 11-day | 63.74% | 76.51% | 132 |
| 7-day vs 11-day | 71.15% | 85.91% | 105 |
| Candidate A vs Candidate B | 84.34% | 84.34% | 57 |

The majority vote reduces active days from 332 to 322. It does not materially reduce maximum Boat notional, because any active position remains the full +/-1,000 tickets.

## Fixed timing-displacement test

Convention: displacement `+d` delays the template by `d` days; at day `t`, the shifted template value is the original value at `t-d`, with edge values held. This is an in-sample timing-fragility diagnostic. It does not use a new template or claim validation.

| Template displacement | Candidate A P&L | Candidate B P&L |
|---:|---:|---:|
| -3 days | 119,630 | 77,260 |
| -2 days | 83,350 | 100,760 |
| -1 day | 76,990 | 79,350 |
| 0 days | 88,260 | 84,040 |
| +1 day | 76,680 | 76,680 |
| +2 days | 81,520 | 103,450 |
| +3 days | 120,050 | 87,880 |

| Model | Zero-shift P&L | Worst of seven | Median of seven | Worst max drawdown | Worst loss vs zero |
|---|---:|---:|---:|---:|---:|
| Candidate A | 88,260 | 76,680 | 83,350 | -8,040 | 13.12% |
| Candidate B | 84,040 | 76,680 | 84,040 | -6,510 | 8.76% |

Both candidates remain profitable on all seven in-sample shifts. B's worst P&L is equal to A's, while its worst drawdown and percentage loss relative to zero shift are better. This is sufficient for the predeclared B preference, but it is not evidence that Round 2 timing cannot move.

## Peak-amplitude stress

The deterministic family starts from a broad fixed 21-day seasonal path and independently scales the four fixed episodes (`S1_large`, `S1_small`, `S2_large`, `S2_small`) with every combination of multipliers in `{0.50, 0.75, 1.00, 1.25, 1.50}`: 625 paths. Turning-point days stay fixed, and the strategy always uses the frozen Round 1 templates rather than the evaluated synthetic path.

| Model | Paths | Median P&L | P10 P&L | Worst P&L | Positive-path rate | Median max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Candidate A | 625 | 51,135 | 38,064 | 25,568 | 100.0% | -1,192 |
| Candidate B | 625 | 49,604 | 36,486 | 24,802 | 100.0% | -1,103 |
| Candidate C | 625 | 51,135 | 38,064 | 25,568 | 100.0% | -1,192 |
| Candidate D | 625 | 49,604 | 36,486 | 24,802 | 100.0% | -1,103 |

Amplitude variation does not reverse the core timing signal in this transparent family. C equals A and D equals B here because the deterministic base is at the AUD 45 summer equilibrium after the seasonal episodes; noisy summer paths are tested separately below. These are deterministic stress results, not independent evidence.

## Residual-noise stress

Residuals are `price - centred_21_day_smooth(price)`, mean-centred, and resampled in moving blocks of length 7. Each of 400 paths independently draws the four amplitude multipliers from the same five frozen levels, then adds a fixed-seed (`20260809`) block-bootstrap residual path. Synthetic percentiles are not statistical confidence intervals.

| Model | Paths | Median P&L | P10 P&L | Worst P&L | Positive-path rate | Median max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Candidate A | 400 | 52,306 | 30,045 | 6,591 | 100.0% | -9,473 |
| Candidate B | 400 | 50,310 | 31,475 | 14,705 | 100.0% | -8,720 |
| Candidate C | 400 | 68,026 | 49,000 | 26,739 | 100.0% | -8,034 |
| Candidate D | 400 | 66,247 | 48,119 | 29,626 | 100.0% | -7,347 |
| Broad fixed-day schedule | 400 | 48,518 | 33,854 | 18,537 | 100.0% | -6,127 |
| Constant AUD 45 mean reversion | 400 | 63,810 | 47,496 | 29,170 | 100.0% | -11,097 |
| Flat Boat Party position | 400 | 0 | 0 | 0 | 0.0% | 0 |
| Fixed 7-day template, no deadband | 400 | 51,782 | 29,314 | -2,410 | 99.75% | -9,962 |

D has a lower median than C but a better worst path and drawdown than C in this particular fixed-seed family. Relative to B, D improves median, P10, worst path and median drawdown. This supports the predeclared overlay decision but remains model-dependent stress evidence.

## Summer ablation

Observed Round 1 summer-only means P&L/drawdown from day 322 onward; the full-year columns include the semester signal. No summer date or equilibrium was tuned.

| Model | Full-year P&L | Full-year max DD | Summer-only P&L | Summer-only max DD |
|---|---:|---:|---:|---:|
| Candidate A | 88,260 | -5,280 | 5,900 | -1,850 |
| Candidate B | 84,040 | -3,150 | 4,030 | -1,930 |
| Candidate C | 94,910 | -5,280 | 12,550 | -940 |
| Candidate D | 92,560 | -2,460 | 12,550 | -940 |

For the synthetic summer-only slices, the table reports the median, P10 and worst summer-only P&L across 100 paths per equilibrium. The overlay comparison is A→C and B→D.

| Equilibrium | Base | Overlay | Base median / P10 / worst | Overlay median / P10 / worst | Base median DD | Overlay median DD |
|---:|---|---|---:|---:|---:|---:|
| 43 | A | C | 1,606 / -7,611 / -15,936 | 3,004 / -1,359 / -3,192 | -5,395 | -3,082 |
| 45 | A | C | 172 / -8,752 / -17,194 | 17,190 / 10,725 / 5,635 | -5,383 | -1,657 |
| 47 | A | C | 751 / -6,868 / -14,510 | 995 / -1,440 / -3,838 | -5,781 | -3,676 |
| 43 | B | D | 1,935 / -5,936 / -15,190 | 3,004 / -1,359 / -3,192 | -4,508 | -3,082 |
| 45 | B | D | -578 / -6,280 / -11,438 | 17,190 / 10,725 / 5,635 | -4,930 | -1,657 |
| 47 | B | D | -113 / -6,878 / -16,652 | 995 / -1,440 / -3,838 | -5,074 | -3,676 |

The fixed AUD 45 overlay is therefore retained. It is especially valuable when the synthetic summer equilibrium is AUD 45, and it improves the lower tail at AUD 43 and AUD 47. The remaining risk is that a real Round 2 summer level can differ from all three tested values or move during the summer.

## Gradual summer-equilibrium transition stress

The earlier stationary summer test is retained as a historical ablation, but it did not expose Candidate D to the equilibrium move beginning on day 322. This correction uses paired paths with AUD 45 through day 321, a linear transition beginning on day 322, and the target held after the transition. Targets are AUD 43, 45 and 47; durations are 7, 14 and 28 days; there are 200 paths per target/duration combination. Each path uses the same fixed-seed residual construction as the prior summer audit: mean-centred residuals from the centred 21-day broad path, resampled in 7-day moving blocks. The same residual draw is reused across target/duration comparisons for each path ID.

All rows below are generator-conditioned stress diagnostics, not confidence intervals or empirical validation. Positive-path rate is 100% for every displayed B/D scenario.

| Target | Days | Model | Full median | Full P10 | Full worst | Summer median | Summer P10 | Summer worst | Median max DD |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 43 | 7 | B | 49,502 | 36,121 | 22,096 | 1,590 | -4,954 | -14,047 | -8,022 |
| 43 | 7 | D | 51,626 | 39,022 | 27,212 | 2,796 | -1,267 | -3,407 | -7,536 |
| 43 | 14 | B | 48,216 | 34,836 | 20,810 | 304 | -6,239 | -15,333 | -8,323 |
| 43 | 14 | D | 53,672 | 39,867 | 27,069 | 4,454 | -891 | -2,450 | -7,422 |
| 43 | 28 | B | 48,287 | 34,907 | 20,881 | 375 | -6,168 | -15,261 | -8,312 |
| 43 | 28 | D | 55,200 | 42,484 | 28,523 | 6,886 | 1,119 | -2,982 | -7,217 |
| 45 | 7 | B | 48,359 | 34,978 | 20,953 | 447 | -6,096 | -15,190 | -8,222 |
| 45 | 7 | D | 65,567 | 51,677 | 41,871 | 16,813 | 10,566 | 4,224 | -7,201 |
| 45 | 14 | B | 48,359 | 34,978 | 20,953 | 447 | -6,096 | -15,190 | -8,222 |
| 45 | 14 | D | 65,567 | 51,677 | 41,871 | 16,813 | 10,566 | 4,224 | -7,201 |
| 45 | 28 | B | 48,359 | 34,978 | 20,953 | 447 | -6,096 | -15,190 | -8,222 |
| 45 | 28 | D | 65,567 | 51,677 | 41,871 | 16,813 | 10,566 | 4,224 | -7,201 |
| 47 | 7 | B | 47,216 | 33,836 | 19,810 | -696 | -7,239 | -16,333 | -8,977 |
| 47 | 7 | D | 51,704 | 38,845 | 26,074 | 586 | -1,971 | -3,790 | -7,646 |
| 47 | 14 | B | 48,502 | 35,121 | 21,096 | 590 | -5,954 | -15,047 | -8,248 |
| 47 | 14 | D | 52,531 | 39,969 | 25,080 | 2,975 | -937 | -3,933 | -7,396 |
| 47 | 28 | B | 48,430 | 35,050 | 21,024 | 518 | -6,025 | -15,119 | -8,204 |
| 47 | 28 | D | 54,959 | 42,729 | 24,295 | 5,524 | 1,036 | -2,793 | -7,295 |

Pooled across all 1,800 paired paths:

| Model | Full median | Full P10 | Full worst | Summer median | Summer P10 | Summer worst | Median max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Candidate B | 48,390 | 35,003 | 19,810 | 444 | -6,235 | -16,333 | -8,277 |
| Candidate D | 56,769 | 42,523 | 24,295 | 7,614 | -430 | -3,933 | -7,331 |

Paired `D - B` full-year P&L results are:

| Target | Days | Median D-B | P10 D-B | Worst D-B |
|---:|---:|---:|---:|---:|
| 43 | 7 | 1,635 | -6,928 | -19,115 |
| 43 | 14 | 4,677 | -4,783 | -14,842 |
| 43 | 28 | 6,559 | -3,011 | -14,985 |
| 45 | 7 | 15,807 | 8,180 | -828 |
| 45 | 14 | 15,807 | 8,180 | -828 |
| 45 | 28 | 15,807 | 8,180 | -828 |
| 47 | 7 | 2,143 | -4,740 | -10,272 |
| 47 | 14 | 2,668 | -3,798 | -10,898 |
| 47 | 28 | 5,532 | -2,401 | -9,451 |
| **Pooled** | **all** | **8,170** | **-3,341** | **-19,115** |

The frozen selection rule passes: pooled D P10 and worst are both above B's; every individual paired median is at least AUD 1,635, above the AUD -2,000 threshold; the observed Round 1 D summer-only P&L is positive at AUD 12,550; and all correctness/budget checks pass. The worst individual D full-year result is AUD 24,295 at target AUD 47 over 28 days; the worst individual paired D-B outcome is AUD -19,115 within the AUD 43, 7-day scenario, while its paired median remains positive.

## Portfolio mechanics

| Model | Max Boat capital | 95th-percentile capital | Active days | Active days vs A | Active days vs C |
|---|---:|---:|---:|---:|---:|
| Candidate A | 55,390 | 54,202 | 332 | 0 | -8 |
| Candidate B | 55,390 | 54,126 | 322 | -10 | -18 |
| Candidate C | 55,390 | 54,202 | 340 | +8 | 0 |
| Candidate D | 55,390 | 54,126 | 333 | +1 | -7 |

The maximum Boat-only notional is AUD 55,390, leaving AUD 544,610 of nominal headroom under the AUD 600,000 shared cap before considering UQ Dollar or other instruments. B/D free approximately 10/7 active days relative to A/C, but they do not lower the maximum notional because all active positions are still +/-1,000. The primary agent must still enforce the shared absolute-value allocator when Boat Party is combined with the existing UQ Dollar strategy; this task does not optimize that allocator.

Day-level price, position and capital usage for all four candidates is recorded in [`fixed_template_daily_capital.csv`](results/fixed_template_daily_capital.csv); the summary above is its max/95th-percentile/active-day reduction.

## Production handoff

- **Final selected candidate:** Candidate D.
- **Exact frozen semester strategy:** fixed competition-series indices; centred 5-, 7- and 11-day Round 1 templates; one-day slope `template[t+1] - template[t]`; independent AUD 0.02 deadbands; +/-1,000 when at least two voters agree; flat otherwise; semester schedule through day 321.
- **5/7/11 clarification:** the prior “10-day” label was the existing helper's even-window input and produced an 11-point smoother. The final implementation explicitly requests 11 and preserves the earlier B/D positions and P&Ls exactly.
- **Day alignment:** position at day `t` earns `price[t+1] - price[t]`; day 364 is flat; no academic-calendar warp.
- **Semester schedule:** [`fixed_template_semester_schedule.csv`](results/fixed_template_semester_schedule.csv). It contains the fixed daily slopes/votes and the majority diagnostic for all days. Candidate D's summer position is not precomputed.
- **Summer rule:** from day 322, +1,000 below current AUD 45, -1,000 above current AUD 45, and zero at AUD 45; day 364 remains flat.
- **Position/capital:** limit +/-1,000; maximum observed Boat notional AUD 55,390.
- **Explicit exclusions:** no calendar warp, RLS, Kalman, OU, Fourier, or absolute seasonal-level forecast.

## Reproducibility and validation record

The focused pipeline is [`final_strategy_test.py`](final_strategy_test.py). It reuses the existing smoothing and backtest mechanics in `analysis.py` and writes the required outputs under `results/`:

- [`fixed_template_candidate_comparison.csv`](results/fixed_template_candidate_comparison.csv)
- [`fixed_template_timing_shifts.csv`](results/fixed_template_timing_shifts.csv) and [`fixed_template_timing_shift_summary.csv`](results/fixed_template_timing_shift_summary.csv)
- [`fixed_template_amplitude_stress.csv`](results/fixed_template_amplitude_stress.csv) and [`fixed_template_amplitude_stress_summary.csv`](results/fixed_template_amplitude_stress_summary.csv)
- [`fixed_template_noise_stress_detail.csv`](results/fixed_template_noise_stress_detail.csv) and [`fixed_template_noise_stress_summary.csv`](results/fixed_template_noise_stress_summary.csv)
- [`fixed_template_summer_ablation.csv`](results/fixed_template_summer_ablation.csv) and [`fixed_template_summer_ablation_summary.csv`](results/fixed_template_summer_ablation_summary.csv)
- [`fixed_template_gradual_summer_stress_detail.csv`](results/fixed_template_gradual_summer_stress_detail.csv)
- [`fixed_template_gradual_summer_stress_summary.csv`](results/fixed_template_gradual_summer_stress_summary.csv)
- [`fixed_template_gradual_summer_paired_comparison.csv`](results/fixed_template_gradual_summer_paired_comparison.csv)
- [`fixed_template_semester_schedule.csv`](results/fixed_template_semester_schedule.csv)
- [`fixed_template_signal_agreement.csv`](results/fixed_template_signal_agreement.csv)
- [`fixed_template_correctness_checks.csv`](results/fixed_template_correctness_checks.csv)
- [`fixed_template_portfolio_mechanics.csv`](results/fixed_template_portfolio_mechanics.csv)
- [`fixed_template_daily_capital.csv`](results/fixed_template_daily_capital.csv)

The executed notebook is [`analysis.ipynb`](analysis.ipynb), including the clearly labelled final fixed-template section. It contains 22 cells and zero error outputs. The research Python modules compile successfully. The figure [`fixed_template_final_strategy.png`](figures/fixed_template_final_strategy.png) shows the Round 1 price/fixed template, expected slope and Candidate A position, cumulative P&L for A–D, and timing-displacement sensitivity.

The unchanged simulator was run as a regression check after the research work. No production files were modified: `trader_interface/algorithm.py`, `trader_interface/simulation.py`, the supplied data, and PDFs remain unchanged.

## Remaining uncertainty

The evidence base contains one observed seasonal path. Same-year template scores are necessarily in-sample, and the fixed timing-displacement test is also same-year. The synthetic paths assume fixed turning-point timing, the selected residual family, and a limited summer-equilibrium grid. They cannot establish Round 2 generalisation. If Round 2 timing is materially warped or the summer equilibrium changes outside the tested range, Candidate D can fail; the fallback is a broader fixed timing schedule rather than a new adaptive model introduced after seeing Round 2.
