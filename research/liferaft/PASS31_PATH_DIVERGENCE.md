# Pass 3.1 portfolio path-divergence audit

This is a mechanics/path-divergence diagnostic, not a new strategy validation
or ranking. It reruns the consumed validation scenarios only for the frozen
`burnin1_markov` focal candidate at the predeclared other-exposure levels and
compares every nonzero run with the zero-exposure run for the same scenario.
The zero-exposure row is a deterministic self-comparison check.

The exposure is applied to the focal portfolio only, but the simulation is an
endogenous game counterfactual. A focal budget-forced flattening can change the
focal vote, the majority and price path, and then reactive opponents' future
effective actions or budget feasibility. The opponent metric below compares
effective engine actions, not merely their raw requests. These results must not
be read as a fixed-path capacity estimate or used to tune a strategy.

## Overall and execution-mode groups

| baseline → exposure | group | cases | different price path | different majority path | mean price days | max price days | mean majority days | max majority days | mean opponent action cells | max opponent action cells |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $0 → $0 | overall | 480 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $0 | controlled_day_indexed | 400 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $0 | state_or_rng_sensitive | 80 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $150,000 | overall | 480 | 38.5% | 38.5% | 62.9 | 319 | 6.4 | 64 | 24.2 | 685 |
| $0 → $150,000 | controlled_day_indexed | 400 | 39.0% | 39.0% | 68.3 | 319 | 6.2 | 63 | 21.0 | 316 |
| $0 → $150,000 | state_or_rng_sensitive | 80 | 36.2% | 36.2% | 35.8 | 218 | 7.3 | 64 | 40.5 | 685 |
| $0 → $300,000 | overall | 480 | 50.6% | 50.6% | 107.2 | 336 | 16.8 | 155 | 62.1 | 1441 |
| $0 → $300,000 | controlled_day_indexed | 400 | 48.0% | 48.0% | 106.9 | 336 | 15.3 | 155 | 48.8 | 622 |
| $0 → $300,000 | state_or_rng_sensitive | 80 | 63.7% | 63.7% | 108.6 | 291 | 23.9 | 152 | 128.8 | 1441 |
| $0 → $450,000 | overall | 480 | 65.4% | 65.4% | 191.9 | 357 | 31.3 | 216 | 112.2 | 2219 |
| $0 → $450,000 | controlled_day_indexed | 400 | 62.0% | 62.0% | 183.7 | 357 | 28.8 | 178 | 92.0 | 1206 |
| $0 → $450,000 | state_or_rng_sensitive | 80 | 82.5% | 82.5% | 232.7 | 356 | 43.7 | 216 | 213.0 | 2219 |

## Scenario families

| baseline → exposure | group | cases | different price path | different majority path | mean price days | max price days | mean majority days | max majority days | mean opponent action cells | max opponent action cells |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $0 → $0 | balanced_random | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $0 | gradual_drift | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $0 | history_rules | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $0 | long_biased_random | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $0 | margin_mixture | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $0 | periodic | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $0 | persistent_long | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $0 | persistent_short | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $0 | reactive_mixture | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $0 | regime_change | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $0 | short_biased_random | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $0 | startup_zero_history | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $150,000 | balanced_random | 40 | 30.0% | 30.0% | 24.9 | 142 | 1.9 | 20 | 14.7 | 272 |
| $0 → $150,000 | gradual_drift | 40 | 50.0% | 50.0% | 73.0 | 219 | 9.4 | 52 | 28.5 | 157 |
| $0 → $150,000 | history_rules | 40 | 35.0% | 35.0% | 19.8 | 85 | 5.9 | 45 | 15.8 | 176 |
| $0 → $150,000 | long_biased_random | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $150,000 | margin_mixture | 40 | 25.0% | 25.0% | 59.5 | 299 | 1.9 | 15 | 3.1 | 17 |
| $0 → $150,000 | periodic | 40 | 30.0% | 30.0% | 61.4 | 277 | 8.1 | 51 | 33.0 | 316 |
| $0 → $150,000 | persistent_long | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $150,000 | persistent_short | 40 | 75.0% | 75.0% | 166.0 | 319 | 7.3 | 32 | 14.8 | 71 |
| $0 → $150,000 | reactive_mixture | 40 | 32.5% | 32.5% | 39.3 | 218 | 12.8 | 64 | 75.5 | 685 |
| $0 → $150,000 | regime_change | 40 | 50.0% | 50.0% | 80.6 | 260 | 10.0 | 43 | 49.9 | 235 |
| $0 → $150,000 | short_biased_random | 40 | 95.0% | 95.0% | 197.8 | 305 | 17.6 | 63 | 50.0 | 166 |
| $0 → $150,000 | startup_zero_history | 40 | 40.0% | 40.0% | 32.4 | 164 | 1.8 | 13 | 5.4 | 112 |
| $0 → $300,000 | balanced_random | 40 | 70.0% | 70.0% | 99.8 | 246 | 7.5 | 48 | 31.2 | 285 |
| $0 → $300,000 | gradual_drift | 40 | 50.0% | 50.0% | 105.9 | 267 | 18.3 | 69 | 57.8 | 229 |
| $0 → $300,000 | history_rules | 40 | 40.0% | 40.0% | 76.3 | 228 | 25.1 | 155 | 65.0 | 616 |
| $0 → $300,000 | long_biased_random | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $300,000 | margin_mixture | 40 | 45.0% | 45.0% | 114.5 | 336 | 14.0 | 75 | 16.6 | 72 |
| $0 → $300,000 | periodic | 40 | 40.0% | 40.0% | 74.8 | 311 | 17.3 | 110 | 70.2 | 608 |
| $0 → $300,000 | persistent_long | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $300,000 | persistent_short | 40 | 85.0% | 85.0% | 230.2 | 335 | 13.8 | 42 | 30.4 | 101 |
| $0 → $300,000 | reactive_mixture | 40 | 47.5% | 47.5% | 97.9 | 291 | 39.4 | 152 | 229.4 | 1441 |
| $0 → $300,000 | regime_change | 40 | 55.0% | 55.0% | 115.2 | 297 | 22.4 | 86 | 116.0 | 622 |
| $0 → $300,000 | short_biased_random | 40 | 95.0% | 95.0% | 252.3 | 324 | 34.9 | 94 | 100.7 | 229 |
| $0 → $300,000 | startup_zero_history | 40 | 80.0% | 80.0% | 119.2 | 256 | 8.5 | 77 | 28.3 | 420 |
| $0 → $450,000 | balanced_random | 40 | 95.0% | 95.0% | 277.4 | 343 | 20.6 | 85 | 72.5 | 301 |
| $0 → $450,000 | gradual_drift | 40 | 80.0% | 80.0% | 198.4 | 343 | 31.4 | 106 | 86.6 | 341 |
| $0 → $450,000 | history_rules | 40 | 65.0% | 65.0% | 194.0 | 332 | 55.0 | 164 | 134.7 | 695 |
| $0 → $450,000 | long_biased_random | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $450,000 | margin_mixture | 40 | 50.0% | 50.0% | 167.8 | 356 | 30.1 | 164 | 46.0 | 222 |
| $0 → $450,000 | periodic | 40 | 55.0% | 55.0% | 159.2 | 353 | 35.6 | 178 | 141.2 | 924 |
| $0 → $450,000 | persistent_long | 40 | 0.0% | 0.0% | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 |
| $0 → $450,000 | persistent_short | 40 | 85.0% | 85.0% | 232.6 | 357 | 18.5 | 54 | 41.5 | 127 |
| $0 → $450,000 | reactive_mixture | 40 | 65.0% | 65.0% | 179.7 | 356 | 65.1 | 216 | 359.1 | 2219 |
| $0 → $450,000 | regime_change | 40 | 90.0% | 90.0% | 261.9 | 352 | 43.1 | 139 | 238.6 | 1206 |
| $0 → $450,000 | short_biased_random | 40 | 100.0% | 100.0% | 345.6 | 356 | 54.0 | 148 | 159.0 | 370 |
| $0 → $450,000 | startup_zero_history | 40 | 100.0% | 100.0% | 285.6 | 344 | 22.3 | 104 | 66.9 | 549 |

`controlled_day_indexed` means the opponent requests are intended to be
day-indexed and path-controlled; `state_or_rng_sensitive` includes reactive,
startup-state, and other populations whose calls or public path can affect
future behavior. Path differences are diagnostics of this counterfactual, not
claims of statistical independence.
