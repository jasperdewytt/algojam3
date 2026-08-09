# Boat Party SavGol regime audit

## Result

The SavGol regime idea is a credible improvement candidate, but the teammate's
supplied string cannot yet be reproduced exactly from the stated filter settings. Keep
the fixed AUD 45 summer rule and obtain the exact string-generation code before
freezing a submission.

| Strategy | Round 1 P&L | Max drawdown | Fixed semester | Neutral overlay | Summer |
|---|---:|---:|---:|---:|---:|
| Frozen Candidate D | 92,560 | -2,460 | 80,010 | 0 | 12,550 |
| Supplied regime string, neutral flat | 96,660 | -1,680 | 84,110 | 0 | 12,550 |
| Supplied regime string + one-day reversal | 126,200 | -3,810 | 84,110 | 29,540 | 12,550 |
| Supplied regime string + EWMA | **134,860** | **-1,680** | 84,110 | 38,200 | 12,550 |
| SavGol 21/2, threshold 0.5 + EWMA | 125,300 | -2,600 | 80,640 | 32,110 | 12,550 |

The supplied-string EWMA gain is spread across the year: every 60-day block
has positive total P&L, and the best five neutral trades provide 20.2% of the
neutral-overlay gain. EWMA everywhere without a calendar gate earns only
86,420, so the seasonal regime classification adds material value.

## Parameter robustness

Across 5 window lengths, four slope thresholds and low polynomial orders 2--4,
the EWMA hybrid has median Round 1 P&L of approximately 108,000--115,000. The
10th-percentile configuration result is approximately 97,000--99,000, above
Candidate D's 92,560. These are same-year configuration diagnostics, not
out-of-sample observations.

The simpler 21-day/order-2/0.5-threshold model earns 125,300 and is preferable
to selecting the maximum-P&L high-order configuration purely in-sample.

## Seasonality-preserving stress

On 800 generator-conditioned paths preserving a broad annual curve while
varying amplitude and block-resampled residual noise:

| Strategy | Median | P10 | Worst | Paired win rate vs D |
|---|---:|---:|---:|---:|
| Candidate D | 74,554 | 50,478 | 31,526 | -- |
| Supplied regime string + EWMA | 97,002 | 64,766 | 39,236 | 85.9% |
| SavGol 21/2/0.5 + EWMA | 91,829 | 63,601 | 42,643 | 84.0% |

Paired lower tails remain negative: the supplied-string hybrid has a P10
difference of -2,772 and a worst difference of -26,851 versus Candidate D.
These simulations are generator-conditioned stress tests, not confidence
intervals or independent validation.

## Provenance issue

The teammate subsequently confirmed a 21-day window and polynomial order 2,
which is a sensible low-complexity smoother. The closest tested conversion uses
a one-day slope threshold near AUD 0.19 (about 0.625 slope standard deviations)
and matches 318/365 characters. The exact edge treatment, slope calculation,
deadband and generation code must therefore still be obtained before treating
the supplied string as fully reproducible.

## Recommendation

1. Ask for and retain the exact script that generated the supplied string.
2. Verify that its parameters and neutral threshold were not chosen from
   next-day P&L or hand-edited against Round 1 outcomes.
3. Use fixed AUD 45 mean reversion on days 322--363 and flat day 364.
4. If provenance is clean, prefer the SavGol-gated EWMA hybrid over Candidate D.
5. If provenance cannot be reproduced, use the simpler auditable
   21-day/order-2/0.5-threshold hybrid or retain Candidate D.

Research code and tables:

- `savgol_regime_audit.py`
- `results/savgol_regime_comparison.csv`
- `results/savgol_regime_parameter_grid.csv`
- `results/savgol_regime_robustness_summary.csv`
- `results/savgol_regime_stress_summary.csv`
- `results/savgol_regime_stress_paired.csv`
