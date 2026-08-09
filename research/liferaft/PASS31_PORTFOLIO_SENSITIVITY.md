# Pass 3.1 Liferaft portfolio-exposure sensitivity

This report is a separate sensitivity analysis and is not mixed into the
primary zero-other-exposure ranking in `PASS3_REPORT.md`. It uses the same
consumed validation scenarios and causal simulator mechanics. The constant
gross exposure is applied to the focal agent only, but this is an **endogenous
game counterfactual**, not a fixed-path measurement of focal budget capacity:
when the focal request is flattened, its vote can change the majority and price
path, which can change reactive opponents' future actions and budget
feasibility.

Constant other exposure is a sensitivity model, not a forecast of the final
portfolio allocation. Turnover is not treated as a transaction cost, and no
candidate parameters were changed for this analysis.

| item | value |
|---|---:|
| validation scenarios per exposure | 480 |
| strategy/exposure cells | 9600 |
| exposure levels | $0, $150,000, $300,000, $450,000 |
| candidates | flat, burnin1_markov, burnin3_markov, online_markov, online_drift |

## Actual validation distributions

All statistics below are across actual runs at each exposure. Rejection and
breach counts refer to the focal Liferaft request during the marked period;
`any rejection` is the fraction of scenarios with at least one rejected
Liferaft action. The flat strategy is the zero-P&L comparator at each exposure.

| other gross exposure | strategy | runs | mean P&L | median P&L | lower quartile | worst run | mean DD | max DD | beat flat | mean rejected | mean breaches | any rejection | mean active days |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `burnin1_markov` | 480 | 218298 | 38000 | -47000 | -265000 | 46671 | 265000 | 60.0% | 66.37 | 66.37 | 33.8% | 147.3 |
| 0 | `burnin3_markov` | 480 | 216896 | 38000 | -48000 | -275000 | 46162 | 275000 | 59.8% | 66.51 | 66.51 | 33.8% | 146.4 |
| 0 | `flat` | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.00 | 0.00 | 0.0% | 0.0 |
| 0 | `online_drift` | 480 | 169496 | 0 | -17000 | -233000 | 30994 | 275000 | 47.1% | 47.09 | 47.09 | 20.0% | 92.1 |
| 0 | `online_markov` | 480 | 210731 | 32000 | -46000 | -280000 | 46125 | 280000 | 60.0% | 67.06 | 67.06 | 33.8% | 143.9 |
| 150000 | `burnin1_markov` | 480 | 153483 | 36500 | -46000 | -265000 | 45729 | 265000 | 60.2% | 85.15 | 85.15 | 47.1% | 131.6 |
| 150000 | `burnin3_markov` | 480 | 152562 | 35000 | -46000 | -275000 | 44985 | 275000 | 60.0% | 85.41 | 85.41 | 47.3% | 130.8 |
| 150000 | `flat` | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.00 | 0.00 | 0.0% | 0.0 |
| 150000 | `online_drift` | 480 | 121167 | 0 | -17000 | -233000 | 30952 | 275000 | 47.1% | 55.32 | 55.32 | 21.2% | 84.1 |
| 150000 | `online_markov` | 480 | 145954 | 31000 | -41000 | -280000 | 45156 | 280000 | 59.6% | 85.41 | 85.41 | 46.9% | 128.1 |
| 300000 | `burnin1_markov` | 480 | 85056 | 30000 | -45000 | -195000 | 42317 | 200000 | 58.5% | 111.68 | 111.68 | 59.8% | 111.2 |
| 300000 | `burnin3_markov` | 480 | 84517 | 30000 | -45000 | -195000 | 41346 | 200000 | 58.5% | 111.90 | 111.90 | 59.6% | 110.3 |
| 300000 | `flat` | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.00 | 0.00 | 0.0% | 0.0 |
| 300000 | `online_drift` | 480 | 67092 | 0 | -17000 | -233000 | 30719 | 275000 | 47.1% | 65.28 | 65.28 | 22.5% | 74.7 |
| 300000 | `online_markov` | 480 | 78285 | 26500 | -42000 | -190000 | 41473 | 194000 | 58.8% | 112.50 | 112.50 | 60.4% | 107.6 |
| 450000 | `burnin1_markov` | 480 | 17948 | 14000 | -28000 | -237000 | 31231 | 237000 | 57.9% | 154.35 | 154.35 | 71.0% | 84.2 |
| 450000 | `burnin3_markov` | 480 | 17312 | 14500 | -27000 | -237000 | 30240 | 237000 | 57.1% | 154.52 | 154.52 | 71.0% | 83.5 |
| 450000 | `flat` | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.00 | 0.00 | 0.0% | 0.0 |
| 450000 | `online_drift` | 480 | 7196 | 0 | -17000 | -233000 | 29440 | 275000 | 47.5% | 88.66 | 88.66 | 31.0% | 63.3 |
| 450000 | `online_markov` | 480 | 13238 | 8000 | -21000 | -167000 | 28306 | 194000 | 57.7% | 155.34 | 155.34 | 71.5% | 80.5 |

## Results by validation family

These family means make exposure-dependent budget effects visible without
presenting them as independent samples or replacing the actual-run tails.

| other gross exposure | family | strategy | mean P&L | mean rejected | mean breaches | any rejection |
|---:|---|---|---:|---:|---:|---:|
| 0 | balanced_random | `burnin1_markov` | -49650 | 0.00 | 0.00 | 0.0% |
| 0 | balanced_random | `burnin3_markov` | -50700 | 0.00 | 0.00 | 0.0% |
| 0 | balanced_random | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 0 | balanced_random | `online_drift` | -4850 | 0.00 | 0.00 | 0.0% |
| 0 | balanced_random | `online_markov` | -43150 | 0.00 | 0.00 | 0.0% |
| 0 | gradual_drift | `burnin1_markov` | 22550 | 55.85 | 55.85 | 40.0% |
| 0 | gradual_drift | `burnin3_markov` | 27550 | 56.30 | 56.30 | 40.0% |
| 0 | gradual_drift | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 0 | gradual_drift | `online_drift` | -9450 | 0.00 | 0.00 | 0.0% |
| 0 | gradual_drift | `online_markov` | 41600 | 56.45 | 56.45 | 45.0% |
| 0 | history_rules | `burnin1_markov` | 1196550 | 0.35 | 0.35 | 35.0% |
| 0 | history_rules | `burnin3_markov` | 1200200 | 0.35 | 0.35 | 35.0% |
| 0 | history_rules | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 0 | history_rules | `online_drift` | 1089250 | 0.35 | 0.35 | 35.0% |
| 0 | history_rules | `online_markov` | 1185700 | 0.35 | 0.35 | 35.0% |
| 0 | long_biased_random | `burnin1_markov` | -2800 | 0.00 | 0.00 | 0.0% |
| 0 | long_biased_random | `burnin3_markov` | -1800 | 0.00 | 0.00 | 0.0% |
| 0 | long_biased_random | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 0 | long_biased_random | `online_drift` | -80150 | 0.00 | 0.00 | 0.0% |
| 0 | long_biased_random | `online_markov` | -10650 | 0.00 | 0.00 | 0.0% |
| 0 | margin_mixture | `burnin1_markov` | 135700 | 98.35 | 98.35 | 35.0% |
| 0 | margin_mixture | `burnin3_markov` | 131600 | 98.45 | 98.45 | 35.0% |
| 0 | margin_mixture | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 0 | margin_mixture | `online_drift` | 116850 | 85.70 | 85.70 | 30.0% |
| 0 | margin_mixture | `online_markov` | 112050 | 98.90 | 98.90 | 35.0% |
| 0 | periodic | `burnin1_markov` | 305800 | 67.90 | 67.90 | 35.0% |
| 0 | periodic | `burnin3_markov` | 303150 | 68.10 | 68.10 | 35.0% |
| 0 | periodic | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 0 | periodic | `online_drift` | 217350 | 55.50 | 55.50 | 25.0% |
| 0 | periodic | `online_markov` | 294950 | 68.70 | 68.70 | 35.0% |
| 0 | persistent_long | `burnin1_markov` | 35750 | 0.00 | 0.00 | 0.0% |
| 0 | persistent_long | `burnin3_markov` | 36000 | 0.00 | 0.00 | 0.0% |
| 0 | persistent_long | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 0 | persistent_long | `online_drift` | -11850 | 0.00 | 0.00 | 0.0% |
| 0 | persistent_long | `online_markov` | 25700 | 0.00 | 0.00 | 0.0% |
| 0 | persistent_short | `burnin1_markov` | 482900 | 282.10 | 282.10 | 100.0% |
| 0 | persistent_short | `burnin3_markov` | 474150 | 282.50 | 282.50 | 100.0% |
| 0 | persistent_short | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 0 | persistent_short | `online_drift` | 441100 | 271.80 | 271.80 | 95.0% |
| 0 | persistent_short | `online_markov` | 441700 | 283.50 | 283.50 | 100.0% |
| 0 | reactive_mixture | `burnin1_markov` | 77100 | 17.65 | 17.65 | 22.5% |
| 0 | reactive_mixture | `burnin3_markov` | 76300 | 17.90 | 17.90 | 22.5% |
| 0 | reactive_mixture | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 0 | reactive_mixture | `online_drift` | 14450 | 0.00 | 0.00 | 0.0% |
| 0 | reactive_mixture | `online_markov` | 83300 | 17.35 | 17.35 | 20.0% |
| 0 | regime_change | `burnin1_markov` | 47450 | 41.95 | 41.95 | 35.0% |
| 0 | regime_change | `burnin3_markov` | 43200 | 41.95 | 41.95 | 35.0% |
| 0 | regime_change | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 0 | regime_change | `online_drift` | -1650 | 0.00 | 0.00 | 0.0% |
| 0 | regime_change | `online_markov` | 48050 | 42.25 | 42.25 | 30.0% |
| 0 | short_biased_random | `burnin1_markov` | 428300 | 230.00 | 230.00 | 100.0% |
| 0 | short_biased_random | `burnin3_markov` | 423950 | 230.25 | 230.25 | 100.0% |
| 0 | short_biased_random | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 0 | short_biased_random | `online_drift` | 271600 | 151.70 | 151.70 | 55.0% |
| 0 | short_biased_random | `online_markov` | 404350 | 232.35 | 232.35 | 100.0% |
| 0 | startup_zero_history | `burnin1_markov` | -60075 | 2.30 | 2.30 | 2.5% |
| 0 | startup_zero_history | `burnin3_markov` | -60850 | 2.30 | 2.30 | 2.5% |
| 0 | startup_zero_history | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 0 | startup_zero_history | `online_drift` | -8700 | 0.00 | 0.00 | 0.0% |
| 0 | startup_zero_history | `online_markov` | -54825 | 4.92 | 4.92 | 5.0% |
| 150000 | balanced_random | `burnin1_markov` | -46150 | 4.15 | 4.15 | 35.0% |
| 150000 | balanced_random | `burnin3_markov` | -45900 | 3.95 | 3.95 | 35.0% |
| 150000 | balanced_random | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 150000 | balanced_random | `online_drift` | -4850 | 0.00 | 0.00 | 0.0% |
| 150000 | balanced_random | `online_markov` | -39750 | 3.60 | 3.60 | 35.0% |
| 150000 | gradual_drift | `burnin1_markov` | -13700 | 76.80 | 76.80 | 50.0% |
| 150000 | gradual_drift | `burnin3_markov` | -9000 | 77.55 | 77.55 | 50.0% |
| 150000 | gradual_drift | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 150000 | gradual_drift | `online_drift` | -9450 | 0.00 | 0.00 | 0.0% |
| 150000 | gradual_drift | `online_markov` | 5050 | 79.90 | 79.90 | 50.0% |
| 150000 | history_rules | `burnin1_markov` | 900400 | 52.00 | 52.00 | 70.0% |
| 150000 | history_rules | `burnin3_markov` | 904050 | 52.00 | 52.00 | 70.0% |
| 150000 | history_rules | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 150000 | history_rules | `online_drift` | 815400 | 45.00 | 45.00 | 50.0% |
| 150000 | history_rules | `online_markov` | 886250 | 52.60 | 52.60 | 75.0% |
| 150000 | long_biased_random | `burnin1_markov` | -2800 | 0.00 | 0.00 | 0.0% |
| 150000 | long_biased_random | `burnin3_markov` | -1800 | 0.00 | 0.00 | 0.0% |
| 150000 | long_biased_random | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 150000 | long_biased_random | `online_drift` | -80150 | 0.00 | 0.00 | 0.0% |
| 150000 | long_biased_random | `online_markov` | -10650 | 0.00 | 0.00 | 0.0% |
| 150000 | margin_mixture | `burnin1_markov` | 82750 | 106.55 | 106.55 | 35.0% |
| 150000 | margin_mixture | `burnin3_markov` | 78650 | 106.70 | 106.70 | 35.0% |
| 150000 | margin_mixture | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 150000 | margin_mixture | `online_drift` | 71550 | 92.45 | 92.45 | 30.0% |
| 150000 | margin_mixture | `online_markov` | 59500 | 107.10 | 107.10 | 35.0% |
| 150000 | periodic | `burnin1_markov` | 254650 | 87.50 | 87.50 | 40.0% |
| 150000 | periodic | `burnin3_markov` | 256850 | 86.55 | 86.55 | 40.0% |
| 150000 | periodic | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 150000 | periodic | `online_drift` | 178250 | 65.80 | 65.80 | 25.0% |
| 150000 | periodic | `online_markov` | 247600 | 87.40 | 87.40 | 40.0% |
| 150000 | persistent_long | `burnin1_markov` | 35750 | 0.00 | 0.00 | 0.0% |
| 150000 | persistent_long | `burnin3_markov` | 36000 | 0.00 | 0.00 | 0.0% |
| 150000 | persistent_long | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 150000 | persistent_long | `online_drift` | -11850 | 0.00 | 0.00 | 0.0% |
| 150000 | persistent_long | `online_markov` | 25700 | 0.00 | 0.00 | 0.0% |
| 150000 | persistent_short | `burnin1_markov` | 334100 | 307.25 | 307.25 | 100.0% |
| 150000 | persistent_short | `burnin3_markov` | 324550 | 307.80 | 307.80 | 100.0% |
| 150000 | persistent_short | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 150000 | persistent_short | `online_drift` | 300800 | 294.85 | 294.85 | 95.0% |
| 150000 | persistent_short | `online_markov` | 298300 | 308.00 | 308.00 | 100.0% |
| 150000 | reactive_mixture | `burnin1_markov` | 45950 | 36.67 | 36.67 | 32.5% |
| 150000 | reactive_mixture | `burnin3_markov` | 45350 | 38.33 | 38.33 | 35.0% |
| 150000 | reactive_mixture | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 150000 | reactive_mixture | `online_drift` | 14450 | 0.00 | 0.00 | 0.0% |
| 150000 | reactive_mixture | `online_markov` | 49100 | 37.33 | 37.33 | 30.0% |
| 150000 | regime_change | `burnin1_markov` | 18300 | 71.60 | 71.60 | 55.0% |
| 150000 | regime_change | `burnin3_markov` | 14450 | 72.55 | 72.55 | 55.0% |
| 150000 | regime_change | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 150000 | regime_change | `online_drift` | -1650 | 0.00 | 0.00 | 0.0% |
| 150000 | regime_change | `online_markov` | 22250 | 67.50 | 67.50 | 55.0% |
| 150000 | short_biased_random | `burnin1_markov` | 289200 | 269.35 | 269.35 | 100.0% |
| 150000 | short_biased_random | `burnin3_markov` | 285500 | 269.65 | 269.65 | 100.0% |
| 150000 | short_biased_random | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 150000 | short_biased_random | `online_drift` | 190200 | 165.75 | 165.75 | 55.0% |
| 150000 | short_biased_random | `online_markov` | 261300 | 271.50 | 271.50 | 100.0% |
| 150000 | startup_zero_history | `burnin1_markov` | -56650 | 9.95 | 9.95 | 47.5% |
| 150000 | startup_zero_history | `burnin3_markov` | -57950 | 9.85 | 9.85 | 47.5% |
| 150000 | startup_zero_history | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 150000 | startup_zero_history | `online_drift` | -8700 | 0.00 | 0.00 | 0.0% |
| 150000 | startup_zero_history | `online_markov` | -53200 | 10.05 | 10.05 | 42.5% |
| 300000 | balanced_random | `burnin1_markov` | -42350 | 22.30 | 22.30 | 75.0% |
| 300000 | balanced_random | `burnin3_markov` | -41600 | 22.80 | 22.80 | 75.0% |
| 300000 | balanced_random | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 300000 | balanced_random | `online_drift` | -4850 | 0.00 | 0.00 | 0.0% |
| 300000 | balanced_random | `online_markov` | -35800 | 24.95 | 24.95 | 80.0% |
| 300000 | gradual_drift | `burnin1_markov` | -37500 | 101.05 | 101.05 | 50.0% |
| 300000 | gradual_drift | `burnin3_markov` | -32050 | 100.80 | 100.80 | 50.0% |
| 300000 | gradual_drift | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 300000 | gradual_drift | `online_drift` | -9450 | 0.00 | 0.00 | 0.0% |
| 300000 | gradual_drift | `online_markov` | -21950 | 101.10 | 101.10 | 50.0% |
| 300000 | history_rules | `burnin1_markov` | 532900 | 127.60 | 127.60 | 75.0% |
| 300000 | history_rules | `burnin3_markov` | 536550 | 127.50 | 127.50 | 75.0% |
| 300000 | history_rules | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 300000 | history_rules | `online_drift` | 470250 | 107.80 | 107.80 | 60.0% |
| 300000 | history_rules | `online_markov` | 521050 | 127.75 | 127.75 | 80.0% |
| 300000 | long_biased_random | `burnin1_markov` | -2800 | 0.00 | 0.00 | 0.0% |
| 300000 | long_biased_random | `burnin3_markov` | -1800 | 0.00 | 0.00 | 0.0% |
| 300000 | long_biased_random | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 300000 | long_biased_random | `online_drift` | -80150 | 0.00 | 0.00 | 0.0% |
| 300000 | long_biased_random | `online_markov` | -10650 | 0.00 | 0.00 | 0.0% |
| 300000 | margin_mixture | `burnin1_markov` | 41250 | 132.20 | 132.20 | 50.0% |
| 300000 | margin_mixture | `burnin3_markov` | 39550 | 133.00 | 133.00 | 50.0% |
| 300000 | margin_mixture | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 300000 | margin_mixture | `online_drift` | 27450 | 99.20 | 99.20 | 30.0% |
| 300000 | margin_mixture | `online_markov` | 22750 | 132.60 | 132.60 | 50.0% |
| 300000 | periodic | `burnin1_markov` | 194850 | 114.10 | 114.10 | 50.0% |
| 300000 | periodic | `burnin3_markov` | 193550 | 114.90 | 114.90 | 50.0% |
| 300000 | periodic | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 300000 | periodic | `online_drift` | 134750 | 77.75 | 77.75 | 30.0% |
| 300000 | periodic | `online_markov` | 184750 | 116.30 | 116.30 | 50.0% |
| 300000 | persistent_long | `burnin1_markov` | 35750 | 0.00 | 0.00 | 0.0% |
| 300000 | persistent_long | `burnin3_markov` | 36000 | 0.00 | 0.00 | 0.0% |
| 300000 | persistent_long | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 300000 | persistent_long | `online_drift` | -11850 | 0.00 | 0.00 | 0.0% |
| 300000 | persistent_long | `online_markov` | 25700 | 0.00 | 0.00 | 0.0% |
| 300000 | persistent_short | `burnin1_markov` | 187000 | 331.90 | 331.90 | 100.0% |
| 300000 | persistent_short | `burnin3_markov` | 177850 | 332.40 | 332.40 | 100.0% |
| 300000 | persistent_short | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 300000 | persistent_short | `online_drift` | 165800 | 317.40 | 317.40 | 95.0% |
| 300000 | persistent_short | `online_markov` | 155950 | 332.20 | 332.20 | 100.0% |
| 300000 | reactive_mixture | `burnin1_markov` | 14125 | 74.45 | 74.45 | 52.5% |
| 300000 | reactive_mixture | `burnin3_markov` | 15975 | 74.00 | 74.00 | 50.0% |
| 300000 | reactive_mixture | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 300000 | reactive_mixture | `online_drift` | 14450 | 0.00 | 0.00 | 0.0% |
| 300000 | reactive_mixture | `online_markov` | 13350 | 72.12 | 72.12 | 47.5% |
| 300000 | regime_change | `burnin1_markov` | -3850 | 99.80 | 99.80 | 75.0% |
| 300000 | regime_change | `burnin3_markov` | -6650 | 99.95 | 99.95 | 75.0% |
| 300000 | regime_change | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 300000 | regime_change | `online_drift` | -1650 | 0.00 | 0.00 | 0.0% |
| 300000 | regime_change | `online_markov` | 650 | 103.40 | 103.40 | 75.0% |
| 300000 | short_biased_random | `burnin1_markov` | 154350 | 308.90 | 308.90 | 100.0% |
| 300000 | short_biased_random | `burnin3_markov` | 149350 | 309.15 | 309.15 | 100.0% |
| 300000 | short_biased_random | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 300000 | short_biased_random | `online_drift` | 109050 | 181.15 | 181.15 | 55.0% |
| 300000 | short_biased_random | `online_markov` | 129900 | 312.45 | 312.45 | 100.0% |
| 300000 | startup_zero_history | `burnin1_markov` | -53050 | 27.85 | 27.85 | 90.0% |
| 300000 | startup_zero_history | `burnin3_markov` | -52525 | 28.32 | 28.32 | 90.0% |
| 300000 | startup_zero_history | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 300000 | startup_zero_history | `online_drift` | -8700 | 0.00 | 0.00 | 0.0% |
| 300000 | startup_zero_history | `online_markov` | -46275 | 27.10 | 27.10 | 92.5% |
| 450000 | balanced_random | `burnin1_markov` | -18100 | 76.80 | 76.80 | 95.0% |
| 450000 | balanced_random | `burnin3_markov` | -16850 | 76.95 | 76.95 | 95.0% |
| 450000 | balanced_random | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 450000 | balanced_random | `online_drift` | -3400 | 3.35 | 3.35 | 10.0% |
| 450000 | balanced_random | `online_markov` | -8750 | 79.55 | 79.55 | 100.0% |
| 450000 | gradual_drift | `burnin1_markov` | -22350 | 136.35 | 136.35 | 80.0% |
| 450000 | gradual_drift | `burnin3_markov` | -20700 | 138.50 | 138.50 | 80.0% |
| 450000 | gradual_drift | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 450000 | gradual_drift | `online_drift` | -7950 | 0.95 | 0.95 | 15.0% |
| 450000 | gradual_drift | `online_markov` | -11650 | 143.75 | 143.75 | 85.0% |
| 450000 | history_rules | `burnin1_markov` | 104950 | 240.65 | 240.65 | 100.0% |
| 450000 | history_rules | `burnin3_markov` | 108700 | 239.10 | 239.10 | 100.0% |
| 450000 | history_rules | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 450000 | history_rules | `online_drift` | 87250 | 190.40 | 190.40 | 65.0% |
| 450000 | history_rules | `online_markov` | 92750 | 239.70 | 239.70 | 100.0% |
| 450000 | long_biased_random | `burnin1_markov` | -2800 | 0.00 | 0.00 | 0.0% |
| 450000 | long_biased_random | `burnin3_markov` | -1800 | 0.00 | 0.00 | 0.0% |
| 450000 | long_biased_random | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 450000 | long_biased_random | `online_drift` | -80150 | 0.00 | 0.00 | 0.0% |
| 450000 | long_biased_random | `online_markov` | -10650 | 0.00 | 0.00 | 0.0% |
| 450000 | margin_mixture | `burnin1_markov` | 12800 | 171.40 | 171.40 | 50.0% |
| 450000 | margin_mixture | `burnin3_markov` | 10600 | 171.70 | 171.70 | 50.0% |
| 450000 | margin_mixture | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 450000 | margin_mixture | `online_drift` | -15000 | 124.80 | 124.80 | 35.0% |
| 450000 | margin_mixture | `online_markov` | -1450 | 172.50 | 172.50 | 50.0% |
| 450000 | periodic | `burnin1_markov` | 92350 | 164.15 | 164.15 | 65.0% |
| 450000 | periodic | `burnin3_markov` | 93250 | 163.15 | 163.15 | 65.0% |
| 450000 | periodic | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 450000 | periodic | `online_drift` | 71150 | 114.30 | 114.30 | 35.0% |
| 450000 | periodic | `online_markov` | 88150 | 161.95 | 161.95 | 60.0% |
| 450000 | persistent_long | `burnin1_markov` | 35750 | 0.00 | 0.00 | 0.0% |
| 450000 | persistent_long | `burnin3_markov` | 36000 | 0.00 | 0.00 | 0.0% |
| 450000 | persistent_long | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 450000 | persistent_long | `online_drift` | -11850 | 0.00 | 0.00 | 0.0% |
| 450000 | persistent_long | `online_markov` | 25700 | 0.00 | 0.00 | 0.0% |
| 450000 | persistent_short | `burnin1_markov` | 41050 | 355.65 | 355.65 | 100.0% |
| 450000 | persistent_short | `burnin3_markov` | 30450 | 356.15 | 356.15 | 100.0% |
| 450000 | persistent_short | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 450000 | persistent_short | `online_drift` | 25000 | 338.65 | 338.65 | 95.0% |
| 450000 | persistent_short | `online_markov` | 15650 | 355.10 | 355.10 | 100.0% |
| 450000 | reactive_mixture | `burnin1_markov` | -4600 | 114.80 | 114.80 | 67.5% |
| 450000 | reactive_mixture | `burnin3_markov` | -2550 | 114.10 | 114.10 | 67.5% |
| 450000 | reactive_mixture | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 450000 | reactive_mixture | `online_drift` | 9525 | 34.05 | 34.05 | 20.0% |
| 450000 | reactive_mixture | `online_markov` | -5900 | 112.25 | 112.25 | 67.5% |
| 450000 | regime_change | `burnin1_markov` | -19450 | 158.45 | 158.45 | 95.0% |
| 450000 | regime_change | `burnin3_markov` | -20350 | 159.45 | 159.45 | 95.0% |
| 450000 | regime_change | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 450000 | regime_change | `online_drift` | -1250 | 3.00 | 3.00 | 10.0% |
| 450000 | regime_change | `online_markov` | -15700 | 161.20 | 161.20 | 95.0% |
| 450000 | short_biased_random | `burnin1_markov` | 27000 | 349.35 | 349.35 | 100.0% |
| 450000 | short_biased_random | `burnin3_markov` | 21800 | 350.00 | 350.00 | 100.0% |
| 450000 | short_biased_random | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 450000 | short_biased_random | `online_drift` | 21650 | 253.85 | 253.85 | 80.0% |
| 450000 | short_biased_random | `online_markov` | 12200 | 348.25 | 348.25 | 100.0% |
| 450000 | startup_zero_history | `burnin1_markov` | -31225 | 84.55 | 84.55 | 100.0% |
| 450000 | startup_zero_history | `burnin3_markov` | -30800 | 85.12 | 85.12 | 100.0% |
| 450000 | startup_zero_history | `flat` | 0 | 0.00 | 0.00 | 0.0% |
| 450000 | startup_zero_history | `online_drift` | -8625 | 0.62 | 0.62 | 7.5% |
| 450000 | startup_zero_history | `online_markov` | -21500 | 89.83 | 89.83 | 100.0% |

## Interpretation

At higher other-instrument exposure, non-flat Liferaft requests can become
infeasible earlier as the price rises and are flattened by the simulator. A
budget breach is therefore not evidence that the candidate made an invalid
action; it records a valid Liferaft request that could not coexist with the
specified constant portfolio exposure. The stored numbers remain valid under
the endogenous simulation assumption, but their decline cannot be attributed
solely to focal rejection because the game path may also have changed. These
results should be combined with the primary ranking only as a risk/sizing
sensitivity, not as a fixed-market capacity curve.
