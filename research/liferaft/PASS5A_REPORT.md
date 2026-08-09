# Liferaft Pass 5A Development Report

Status: development screening evidence only.  This is not a production
acceptance result and not a fresh holdout result.

The frozen protocol was written before the full run.  Only existing
`development_scenarios()` and `validation_scenarios()` constructors were used;
the validation cases are consumed historical development evidence.

## Scope and execution

- development scenarios: **9**
- consumed validation scenarios: **480**
- total scenarios: **489**
- strategies per exposure: **7** (shadow8_markov, shadow12_markov, shadow20_markov, flat, burnin1_markov, online_markov, risk50_burnin1_markov)
- exposure levels: **AUD 0, AUD 150,000, AUD 300,000, AUD 450,000**
- run-level cells: **13692**
- runtime: **1258.26 seconds**
- aggregate run-level mean P&L: **AUD 89043**

Every strategy/scenario/exposure cell used fresh focal and opponent instances.
Exposure gates can change the focal vote, so later majorities, prices,
reactive actions, and budget feasibility can differ endogenously.  Paired and
undertrading comparisons using those paths are labelled realised-path
diagnostics, not opponent-only counterfactuals.

## Aggregate strategy diagnostics

| strategy | runs | mean P&L | median | lower quartile | worst | mean DD | max DD | activation rate | median activation day | never activated | deactivation frequency | reactivation frequency | active state days | paused days | cooldown days | virtual before | virtual after | real after | edge gates | edge frequency | unknown gates | floor gates | headroom gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shadow8_markov` | 1956 | 102633 | 8000 | -8000 | -57000 | 17277 | 93000 | 94.9% | 384.0 | 5.1% | 0.66 | 0.36 | 234.6 | 197.4 | 3.3 | 48301 | 260600 | 104651 | 51050 | 7.15% | 386195 | 8634 | 53163 |
| `shadow12_markov` | 1956 | 97374 | 5500 | -10000 | -57000 | 17001 | 93000 | 94.9% | 386.0 | 5.1% | 0.66 | 0.36 | 232.3 | 197.5 | 3.3 | 56442 | 255433 | 99740 | 49328 | 6.91% | 386365 | 8656 | 52460 |
| `shadow20_markov` | 1956 | 83675 | 0 | -8000 | -57000 | 15506 | 86000 | 88.3% | 414.0 | 11.7% | 0.62 | 0.35 | 198.0 | 198.1 | 3.1 | 76691 | 239444 | 86522 | 48083 | 6.73% | 387569 | 8752 | 51526 |
| `flat` | 1956 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0 | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0.00% | 0 | 0 | 0 |
| `burnin1_markov` | 1956 | 117690 | 25000 | -42000 | -265000 | 40975 | 265000 | 0.0% | 0.0 | 0.0% | 0.00 | 0.00 | 118.4 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0.00% | 0 | 0 | 0 |
| `online_markov` | 1956 | 111028 | 16000 | -34000 | -280000 | 39736 | 280000 | 0.0% | 0.0 | 0.0% | 0.00 | 0.00 | 114.8 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0.00% | 0 | 0 | 0 |
| `risk50_burnin1_markov` | 1956 | 110901 | 11000 | -12000 | -55000 | 13985 | 65000 | 0.0% | 0.0 | 0.0% | 0.00 | 0.00 | 31.4 | 121.2 | 0.0 | 0 | 0 | 0 | 0 | 0.00% | 237081 | 4916 | 30727 |

This table reports actual run-level means, medians, lower quartiles, worst
marked P&L, mean and maximum drawdown, activation and never-activation rates,
deactivation/reactivation frequency, active/paused/cooling time, virtual P&L
before and after activation, real P&L after activation, and gate counts.

## Headline metrics by exposure

| exposure | strategy | runs | mean P&L | median | lower quartile | worst | mean DD | max DD | active days | turnover | breaches | rejected | pivotal P&L | non-pivotal P&L |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `shadow8_markov` | 489 | 190350 | 29000 | -19000 | -54000 | 22673 | 93000 | 58.7 | 74.0 | 0 | 0 | -31535.787321063395 | 221885.48057259712 |
| 0 | `shadow12_markov` | 489 | 184286 | 25000 | -21000 | -57000 | 22346 | 93000 | 57.1 | 72.6 | 0 | 0 | -30875.25562372188 | 215161.55419222903 |
| 0 | `shadow20_markov` | 489 | 167569 | 5000 | -23000 | -55000 | 21323 | 86000 | 52.6 | 68.6 | 0 | 0 | -28623.721881390593 | 196192.2290388548 |
| 0 | `flat` | 489 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 |
| 0 | `burnin1_markov` | 489 | 216321 | 38000 | -47000 | -265000 | 46117 | 265000 | 146.7 | 95.3 | 32462 | 32462 | -74631.9018404908 | 290952.9652351738 |
| 0 | `online_markov` | 489 | 208753 | 32000 | -45000 | -280000 | 45550 | 280000 | 143.4 | 94.4 | 32795 | 32795 | -71979.55010224949 | 280732.10633946833 |
| 0 | `risk50_burnin1_markov` | 489 | 187603 | 13000 | -13000 | -55000 | 14620 | 65000 | 45.1 | 55.3 | 0 | 0 | -17505.11247443763 | 205108.38445807772 |
| 150,000 | `shadow8_markov` | 489 | 133051 | 27000 | -17000 | -54000 | 21284 | 93000 | 45.5 | 59.7 | 0 | 0 | -26462.167689161553 | 159513.29243353783 |
| 150,000 | `shadow12_markov` | 489 | 126924 | 19000 | -17000 | -57000 | 21008 | 93000 | 43.8 | 58.4 | 0 | 0 | -25576.68711656442 | 152501.02249488753 |
| 150,000 | `shadow20_markov` | 489 | 110211 | 3000 | -20000 | -55000 | 19828 | 85000 | 39.4 | 54.5 | 0 | 0 | -23689.16155419223 | 133899.79550102248 |
| 150,000 | `flat` | 489 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 |
| 150,000 | `burnin1_markov` | 489 | 152088 | 36000 | -46000 | -265000 | 45182 | 265000 | 131.3 | 82.5 | 41516 | 41516 | -67635.9918200409 | 219723.9263803681 |
| 150,000 | `online_markov` | 489 | 144566 | 31000 | -41000 | -280000 | 44579 | 280000 | 127.7 | 81.2 | 41643 | 41643 | -64938.65030674847 | 209505.11247443763 |
| 150,000 | `risk50_burnin1_markov` | 489 | 139851 | 13000 | -13000 | -55000 | 14558 | 65000 | 37.0 | 46.0 | 0 | 0 | -16098.159509202455 | 155948.87525562372 |
| 300,000 | `shadow8_markov` | 489 | 72593 | 20000 | -10000 | -56000 | 17468 | 93000 | 29.2 | 40.3 | 0 | 0 | -18918.200408997956 | 91511.24744376278 |
| 300,000 | `shadow12_markov` | 489 | 66360 | 13000 | -12000 | -57000 | 17059 | 93000 | 27.6 | 39.2 | 0 | 0 | -18032.719836400818 | 84392.63803680982 |
| 300,000 | `shadow20_markov` | 489 | 50708 | 1000 | -10000 | -55000 | 15092 | 71000 | 23.0 | 34.9 | 0 | 0 | -15791.411042944785 | 66498.97750511247 |
| 300,000 | `flat` | 489 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 |
| 300,000 | `burnin1_markov` | 489 | 84382 | 30000 | -44000 | -195000 | 41796 | 200000 | 111.1 | 63.4 | 54297 | 54297 | -57366.05316973415 | 141748.4662576687 |
| 300,000 | `online_markov` | 489 | 77605 | 26000 | -41000 | -190000 | 40916 | 194000 | 107.5 | 61.8 | 54684 | 54684 | -55024.539877300616 | 132629.85685071573 |
| 300,000 | `risk50_burnin1_markov` | 489 | 87912 | 13000 | -13000 | -55000 | 14256 | 65000 | 27.8 | 34.8 | 0 | 0 | -14040.899795501022 | 101952.96523517382 |
| 450,000 | `shadow8_markov` | 489 | 14540 | 0 | 0 | -57000 | 7681 | 93000 | 10.4 | 14.3 | 0 | 0 | -7850.715746421268 | 22390.593047034767 |
| 450,000 | `shadow12_markov` | 489 | 11926 | 0 | 0 | -57000 | 7591 | 93000 | 9.8 | 13.8 | 0 | 0 | -7584.867075664622 | 19511.247443762782 |
| 450,000 | `shadow20_markov` | 489 | 6213 | 0 | 0 | -57000 | 5779 | 68000 | 7.2 | 10.4 | 0 | 0 | -6081.799591002045 | 12294.478527607362 |
| 450,000 | `flat` | 489 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 |
| 450,000 | `burnin1_markov` | 489 | 17969 | 14000 | -28000 | -237000 | 30806 | 237000 | 84.4 | 33.8 | 74966 | 74966 | -41404.907975460126 | 59374.233128834356 |
| 450,000 | `online_markov` | 489 | 13188 | 8000 | -21000 | -167000 | 27898 | 194000 | 80.8 | 31.4 | 75447 | 75447 | -38503.06748466258 | 51691.20654396728 |
| 450,000 | `risk50_burnin1_markov` | 489 | 28239 | 11000 | -12000 | -55000 | 12505 | 65000 | 15.6 | 18.9 | 0 | 0 | -9903.885480572597 | 38143.14928425358 |

Quantiles and worst values are calculated from individual runs rather than
averages of averages.  `active days` counts non-zero real actions; the active
state can be longer because an economic-edge, floor, headroom, pause, or loss
gate can keep the real position flat.

## Family summaries

| family | strategy | runs | mean P&L | median | lower quartile | worst | mean DD | max DD | active days | activation rate | never activated | deactivations | reactivations | cooling days | paused days |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced_random | `burnin1_markov` | 160 | -39062 | -44000 | -56000 | -97000 | 53875 | 99000 | 35.4 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| balanced_random | `flat` | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| balanced_random | `online_markov` | 160 | -31862 | -37000 | -54250 | -109000 | 50900 | 109000 | 34.5 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| balanced_random | `risk50_burnin1_markov` | 160 | -13175 | -12000 | -25250 | -37000 | 20025 | 37000 | 7.5 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 4.2 |
| balanced_random | `shadow12_markov` | 160 | -12612 | -10000 | -30500 | -50000 | 22962 | 64000 | 13.4 | 95.0% | 5.0% | 0.99 | 0.44 | 4.9 | 74.4 |
| balanced_random | `shadow20_markov` | 160 | -12612 | -10000 | -31000 | -54000 | 22262 | 64000 | 14.8 | 95.0% | 5.0% | 0.99 | 0.47 | 4.9 | 73.0 |
| balanced_random | `shadow8_markov` | 160 | -12825 | -11000 | -28500 | -50000 | 23750 | 64000 | 14.1 | 95.0% | 5.0% | 0.97 | 0.41 | 4.9 | 74.4 |
| comfortable_nonpivotal | `burnin1_markov` | 4 | 60000 | 60000 | 60000 | 60000 | 0 | 0 | 361.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| comfortable_nonpivotal | `flat` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| comfortable_nonpivotal | `online_markov` | 4 | 55000 | 55000 | 55000 | 55000 | 0 | 0 | 360.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| comfortable_nonpivotal | `risk50_burnin1_markov` | 4 | 60000 | 60000 | 60000 | 60000 | 0 | 0 | 12.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 348.0 |
| comfortable_nonpivotal | `shadow12_markov` | 4 | 10000 | 10000 | 10000 | 10000 | 0 | 0 | 2.0 | 100.0% | 0.0% | 0.00 | 0.00 | 0.0 | 348.0 |
| comfortable_nonpivotal | `shadow20_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 100.0% | 0.00 | 0.00 | 0.0 | 348.0 |
| comfortable_nonpivotal | `shadow8_markov` | 4 | 20000 | 20000 | 20000 | 20000 | 0 | 0 | 4.0 | 100.0% | 0.0% | 0.00 | 0.00 | 0.0 | 348.0 |
| floor_clipping | `burnin1_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| floor_clipping | `flat` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| floor_clipping | `online_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| floor_clipping | `risk50_burnin1_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 364.0 |
| floor_clipping | `shadow12_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 100.0% | 0.00 | 0.00 | 0.0 | 364.0 |
| floor_clipping | `shadow20_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 100.0% | 0.00 | 0.00 | 0.0 | 364.0 |
| floor_clipping | `shadow8_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 100.0% | 0.00 | 0.00 | 0.0 | 364.0 |
| gradual_drift | `burnin1_markov` | 160 | -12750 | -39000 | -61250 | -100000 | 63000 | 128000 | 48.4 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| gradual_drift | `flat` | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| gradual_drift | `online_markov` | 160 | 3262 | -16000 | -48250 | -75000 | 48812 | 88000 | 45.7 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| gradual_drift | `risk50_burnin1_markov` | 160 | -8362 | -9500 | -15000 | -23000 | 15575 | 25000 | 6.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 3.7 |
| gradual_drift | `shadow12_markov` | 160 | 625 | 0 | -33000 | -57000 | 25012 | 64000 | 21.2 | 85.0% | 15.0% | 0.85 | 0.47 | 4.2 | 173.9 |
| gradual_drift | `shadow20_markov` | 160 | 12 | 0 | -33000 | -53000 | 24350 | 73000 | 20.6 | 85.0% | 15.0% | 0.79 | 0.41 | 3.9 | 174.5 |
| gradual_drift | `shadow8_markov` | 160 | 2550 | 0 | -30750 | -53000 | 25512 | 93000 | 22.1 | 85.0% | 15.0% | 0.79 | 0.44 | 3.9 | 174.2 |
| history_rules | `burnin1_markov` | 160 | 683700 | 536500 | 7000 | -89000 | 24038 | 95000 | 154.6 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| history_rules | `flat` | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| history_rules | `online_markov` | 160 | 671438 | 509500 | 26250 | -117000 | 23962 | 117000 | 149.6 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| history_rules | `risk50_burnin1_markov` | 160 | 610450 | 176500 | 1000 | -29000 | 12075 | 29000 | 113.3 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 24.9 |
| history_rules | `shadow12_markov` | 160 | 640200 | 491500 | 29000 | -57000 | 18888 | 68000 | 137.6 | 100.0% | 0.0% | 0.64 | 0.50 | 3.2 | 59.2 |
| history_rules | `shadow20_markov` | 160 | 609775 | 451500 | 10250 | -57000 | 18288 | 68000 | 131.9 | 100.0% | 0.0% | 0.64 | 0.50 | 3.2 | 59.4 |
| history_rules | `shadow8_markov` | 160 | 645825 | 504500 | 29000 | -57000 | 18700 | 68000 | 138.9 | 100.0% | 0.0% | 0.64 | 0.50 | 3.2 | 59.3 |
| long_biased_random | `burnin1_markov` | 160 | -2800 | 19500 | -52500 | -126000 | 71000 | 157000 | 225.7 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| long_biased_random | `flat` | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| long_biased_random | `online_markov` | 160 | -10650 | 12500 | -53750 | -161000 | 77100 | 194000 | 219.7 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| long_biased_random | `risk50_burnin1_markov` | 160 | 72500 | 96500 | -8000 | -55000 | 24400 | 65000 | 50.9 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 153.6 |
| long_biased_random | `shadow12_markov` | 160 | 26050 | 25000 | -17250 | -50000 | 23150 | 67000 | 32.2 | 100.0% | 0.0% | 1.40 | 0.75 | 7.0 | 252.2 |
| long_biased_random | `shadow20_markov` | 160 | 11250 | 11000 | -18500 | -51000 | 19550 | 61000 | 22.6 | 100.0% | 0.0% | 1.30 | 0.70 | 6.5 | 253.8 |
| long_biased_random | `shadow8_markov` | 160 | 32850 | 35500 | -9750 | -50000 | 21350 | 67000 | 33.6 | 100.0% | 0.0% | 1.35 | 0.75 | 6.7 | 252.2 |
| margin_mixture | `burnin1_markov` | 160 | 68125 | 41000 | 4250 | -265000 | 52488 | 265000 | 189.7 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| margin_mixture | `flat` | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| margin_mixture | `online_markov` | 160 | 48212 | 22000 | -11000 | -280000 | 59212 | 280000 | 184.8 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| margin_mixture | `risk50_burnin1_markov` | 160 | 94438 | 59000 | 9000 | -50000 | 12100 | 50000 | 28.2 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 244.5 |
| margin_mixture | `shadow12_markov` | 160 | 56512 | 20000 | 0 | -50000 | 11088 | 50000 | 18.5 | 100.0% | 0.0% | 0.26 | 0.05 | 1.3 | 300.8 |
| margin_mixture | `shadow20_markov` | 160 | 37475 | 0 | 0 | -50000 | 7588 | 50000 | 12.8 | 85.0% | 15.0% | 0.21 | 0.05 | 1.1 | 301.7 |
| margin_mixture | `shadow8_markov` | 160 | 61800 | 26000 | 0 | -50000 | 12550 | 50000 | 20.4 | 100.0% | 0.0% | 0.31 | 0.05 | 1.6 | 300.3 |
| near_tie_pivotal | `burnin1_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| near_tie_pivotal | `flat` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| near_tie_pivotal | `online_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| near_tie_pivotal | `risk50_burnin1_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 364.0 |
| near_tie_pivotal | `shadow12_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 100.0% | 0.00 | 0.00 | 0.0 | 364.0 |
| near_tie_pivotal | `shadow20_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 100.0% | 0.00 | 0.00 | 0.0 | 364.0 |
| near_tie_pivotal | `shadow8_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 100.0% | 0.00 | 0.00 | 0.0 | 364.0 |
| no_trade_stress | `burnin1_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| no_trade_stress | `flat` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| no_trade_stress | `online_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| no_trade_stress | `risk50_burnin1_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 364.0 |
| no_trade_stress | `shadow12_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 100.0% | 0.00 | 0.00 | 0.0 | 364.0 |
| no_trade_stress | `shadow20_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 100.0% | 0.00 | 0.00 | 0.0 | 364.0 |
| no_trade_stress | `shadow8_markov` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 100.0% | 0.00 | 0.00 | 0.0 | 364.0 |
| periodic | `burnin1_markov` | 160 | 211912 | 102000 | 28000 | -237000 | 32138 | 237000 | 130.2 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| periodic | `flat` | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| periodic | `online_markov` | 160 | 203862 | 92000 | 14750 | -167000 | 32275 | 175000 | 128.3 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| periodic | `risk50_burnin1_markov` | 160 | 154362 | 55000 | -14000 | -50000 | 14300 | 50000 | 54.4 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 156.7 |
| periodic | `shadow12_markov` | 160 | 137625 | 68000 | 0 | -52000 | 15362 | 52000 | 57.1 | 100.0% | 0.0% | 0.35 | 0.25 | 1.8 | 210.8 |
| periodic | `shadow20_markov` | 160 | 134962 | 40000 | 0 | -51000 | 13012 | 56000 | 53.7 | 90.0% | 10.0% | 0.40 | 0.30 | 2.0 | 211.6 |
| periodic | `shadow8_markov` | 160 | 141450 | 74500 | 0 | -52000 | 15362 | 52000 | 57.9 | 100.0% | 0.0% | 0.35 | 0.25 | 1.8 | 210.7 |
| persistent_long | `burnin1_markov` | 164 | 36341 | 45000 | 28000 | -16000 | 25659 | 75000 | 347.3 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| persistent_long | `flat` | 164 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| persistent_long | `online_markov` | 164 | 26415 | 36000 | 20000 | -27000 | 32634 | 84000 | 347.5 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| persistent_long | `risk50_burnin1_markov` | 164 | 72683 | 68000 | 60000 | -26000 | 8537 | 31000 | 24.6 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 297.6 |
| persistent_long | `shadow12_markov` | 164 | 22927 | 20000 | 10000 | 0 | 4780 | 16000 | 7.6 | 95.1% | 4.9% | 0.49 | 0.00 | 2.4 | 332.0 |
| persistent_long | `shadow20_markov` | 164 | 2439 | 0 | 0 | 0 | 1561 | 8000 | 1.2 | 43.9% | 56.1% | 0.24 | 0.00 | 1.2 | 332.8 |
| persistent_long | `shadow8_markov` | 164 | 31951 | 30000 | 20000 | 0 | 4780 | 16000 | 9.4 | 95.1% | 4.9% | 0.49 | 0.00 | 2.4 | 332.0 |
| persistent_short | `burnin1_markov` | 164 | 261524 | 235500 | 103500 | 12000 | 5073 | 25000 | 43.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| persistent_short | `flat` | 164 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| persistent_short | `online_markov` | 164 | 228195 | 215000 | 86000 | 3000 | 3976 | 25000 | 37.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| persistent_short | `risk50_burnin1_markov` | 164 | 219183 | 184000 | 40000 | 3000 | 4085 | 20000 | 34.4 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 278.1 |
| persistent_short | `shadow12_markov` | 164 | 158707 | 129500 | 47250 | 0 | 3427 | 20000 | 25.4 | 100.0% | 0.0% | 0.00 | 0.00 | 0.0 | 295.6 |
| persistent_short | `shadow20_markov` | 164 | 117402 | 94000 | 4500 | 0 | 2561 | 20000 | 18.9 | 100.0% | 0.0% | 0.00 | 0.00 | 0.0 | 296.2 |
| persistent_short | `shadow8_markov` | 164 | 179134 | 151000 | 55500 | 0 | 3415 | 20000 | 28.6 | 100.0% | 0.0% | 0.00 | 0.00 | 0.0 | 295.5 |
| reactive_mixture | `burnin1_markov` | 160 | 33144 | 500 | -46000 | -145000 | 47419 | 145000 | 104.2 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| reactive_mixture | `flat` | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| reactive_mixture | `online_markov` | 160 | 34962 | 0 | -48000 | -158000 | 44544 | 158000 | 101.9 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| reactive_mixture | `risk50_burnin1_markov` | 160 | 11644 | -1000 | -15500 | -50000 | 16100 | 50000 | 16.2 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 53.5 |
| reactive_mixture | `shadow12_markov` | 160 | 23538 | 0 | -25250 | -55000 | 26419 | 93000 | 38.2 | 95.0% | 5.0% | 0.91 | 0.60 | 4.6 | 168.9 |
| reactive_mixture | `shadow20_markov` | 160 | 17944 | 0 | -26750 | -55000 | 26381 | 85000 | 33.5 | 95.0% | 5.0% | 0.92 | 0.58 | 4.6 | 171.9 |
| reactive_mixture | `shadow8_markov` | 160 | 26044 | 0 | -25000 | -56000 | 27725 | 93000 | 40.4 | 95.0% | 5.0% | 0.99 | 0.69 | 5.0 | 169.1 |
| regime_change | `burnin1_markov` | 160 | 10612 | -29000 | -48000 | -92000 | 49188 | 93000 | 48.8 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| regime_change | `flat` | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| regime_change | `online_markov` | 160 | 13812 | -17500 | -31000 | -96000 | 43725 | 108000 | 45.1 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| regime_change | `risk50_burnin1_markov` | 160 | -1062 | -5000 | -16250 | -51000 | 15800 | 51000 | 8.6 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 7.3 |
| regime_change | `shadow12_markov` | 160 | 14675 | 0 | -17750 | -54000 | 19988 | 58000 | 19.8 | 80.0% | 20.0% | 0.84 | 0.55 | 4.2 | 123.8 |
| regime_change | `shadow20_markov` | 160 | 14538 | 0 | -15000 | -53000 | 19725 | 61000 | 20.0 | 80.0% | 20.0% | 0.76 | 0.53 | 3.8 | 123.3 |
| regime_change | `shadow8_markov` | 160 | 14862 | 0 | -15250 | -54000 | 20475 | 60000 | 20.2 | 80.0% | 20.0% | 0.76 | 0.53 | 3.8 | 123.4 |
| runaway_budget | `burnin1_markov` | 4 | 272000 | 272000 | 162000 | 48000 | 0 | 0 | 34.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| runaway_budget | `flat` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| runaway_budget | `online_markov` | 4 | 240000 | 240000 | 130000 | 16000 | 0 | 0 | 30.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| runaway_budget | `risk50_burnin1_markov` | 4 | 262000 | 260000 | 148000 | 40000 | 0 | 0 | 32.8 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 301.0 |
| runaway_budget | `shadow12_markov` | 4 | 174000 | 156000 | 60000 | 0 | 0 | 0 | 21.8 | 100.0% | 0.0% | 0.00 | 0.00 | 0.0 | 301.0 |
| runaway_budget | `shadow20_markov` | 4 | 126000 | 92000 | 12000 | 0 | 0 | 0 | 15.8 | 100.0% | 0.0% | 0.00 | 0.00 | 0.0 | 301.0 |
| runaway_budget | `shadow8_markov` | 4 | 198000 | 188000 | 84000 | 0 | 0 | 0 | 24.8 | 100.0% | 0.0% | 0.00 | 0.00 | 0.0 | 301.0 |
| short_biased_random | `burnin1_markov` | 160 | 224712 | 197000 | 50250 | -32000 | 12375 | 45000 | 64.7 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| short_biased_random | `flat` | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| short_biased_random | `online_markov` | 160 | 201938 | 169500 | 74250 | -10000 | 9775 | 30000 | 57.3 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| short_biased_random | `risk50_burnin1_markov` | 160 | 141275 | 58000 | 13250 | -7000 | 7388 | 22000 | 30.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 195.8 |
| short_biased_random | `shadow12_markov` | 160 | 133012 | 97000 | 0 | -10000 | 8425 | 30000 | 34.6 | 100.0% | 0.0% | 0.00 | 0.00 | 0.0 | 277.2 |
| short_biased_random | `shadow20_markov` | 160 | 103225 | 58500 | 0 | -10000 | 7588 | 30000 | 28.4 | 100.0% | 0.0% | 0.03 | 0.03 | 0.1 | 278.2 |
| short_biased_random | `shadow8_markov` | 160 | 140162 | 108000 | 0 | -10000 | 8650 | 30000 | 36.4 | 100.0% | 0.0% | 0.00 | 0.00 | 0.0 | 276.6 |
| startup_zero_history | `burnin1_markov` | 168 | -49946 | -48000 | -74500 | -133000 | 60857 | 133000 | 33.8 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| startup_zero_history | `flat` | 168 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| startup_zero_history | `online_markov` | 168 | -43815 | -44000 | -64000 | -119000 | 55179 | 119000 | 31.6 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| startup_zero_history | `risk50_burnin1_markov` | 168 | -12869 | -15000 | -20000 | -46000 | 19298 | 46000 | 6.8 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 4.0 |
| startup_zero_history | `shadow12_markov` | 168 | -19048 | -13000 | -36250 | -54000 | 26792 | 86000 | 14.9 | 95.2% | 4.8% | 1.26 | 0.76 | 6.3 | 82.5 |
| startup_zero_history | `shadow20_markov` | 168 | -18696 | -13000 | -36250 | -53000 | 25321 | 86000 | 13.1 | 95.2% | 4.8% | 1.18 | 0.65 | 5.9 | 82.6 |
| startup_zero_history | `shadow8_markov` | 168 | -18893 | -13000 | -37250 | -54000 | 27363 | 86000 | 15.2 | 95.2% | 4.8% | 1.30 | 0.79 | 6.5 | 82.5 |

## Execution-mode summaries

| execution mode | strategy | runs | mean P&L | median | lower quartile | worst | mean DD | max DD | active days | activation rate | never activated | deactivations | reactivations | cooling days | paused days |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fully_inactive | `burnin1_markov` | 972 | 117701 | 26500 | -44000 | -265000 | 41673 | 265000 | 119.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| fully_inactive | `flat` | 972 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| fully_inactive | `online_markov` | 972 | 111969 | 16000 | -34000 | -280000 | 40069 | 280000 | 115.5 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| fully_inactive | `risk50_burnin1_markov` | 972 | 112057 | 11000 | -12000 | -55000 | 13925 | 65000 | 31.5 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 119.5 |
| fully_inactive | `shadow12_markov` | 972 | 97506 | 6000 | -10000 | -57000 | 17020 | 93000 | 34.5 | 95.9% | 4.1% | 0.66 | 0.35 | 3.3 | 196.8 |
| fully_inactive | `shadow20_markov` | 972 | 83838 | 0 | -7250 | -57000 | 15526 | 86000 | 30.5 | 89.3% | 10.7% | 0.62 | 0.34 | 3.1 | 197.5 |
| fully_inactive | `shadow8_markov` | 972 | 102811 | 8000 | -7000 | -57000 | 17341 | 93000 | 35.9 | 95.9% | 4.1% | 0.67 | 0.36 | 3.3 | 196.7 |
| observe_and_ignore_actions | `burnin1_markov` | 984 | 117680 | 24000 | -40250 | -265000 | 40286 | 265000 | 117.7 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| observe_and_ignore_actions | `flat` | 984 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| observe_and_ignore_actions | `online_markov` | 984 | 110099 | 16000 | -34000 | -280000 | 39407 | 280000 | 114.2 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 0.0 |
| observe_and_ignore_actions | `risk50_burnin1_markov` | 984 | 109760 | 11000 | -13000 | -55000 | 14044 | 65000 | 31.2 | 0.0% | 0.0% | 0.00 | 0.00 | 0.0 | 122.9 |
| observe_and_ignore_actions | `shadow12_markov` | 984 | 97244 | 5000 | -10000 | -57000 | 16983 | 77000 | 34.6 | 93.9% | 6.1% | 0.66 | 0.37 | 3.3 | 198.2 |
| observe_and_ignore_actions | `shadow20_markov` | 984 | 83513 | 0 | -8250 | -57000 | 15486 | 85000 | 30.6 | 87.4% | 12.6% | 0.61 | 0.35 | 3.0 | 198.8 |
| observe_and_ignore_actions | `shadow8_markov` | 984 | 102458 | 6000 | -9000 | -57000 | 17213 | 93000 | 36.0 | 93.9% | 6.1% | 0.65 | 0.36 | 3.2 | 198.2 |

## Exposure sensitivity

| exposure | strategy | mean P&L | worst | max DD | mean active days | activation rate | virtual before | virtual after | real after | edge gates | unknown gates | floor gates | headroom gates |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `shadow8_markov` | 190350 | -54000 | 93000 | 58.7 | 94.9% | 47961 | 234965 | 193127 | 12936 | 94655 | 2161 | 406 |
| 0 | `shadow12_markov` | 184286 | -57000 | 93000 | 57.1 | 94.9% | 55865 | 230791 | 187468 | 12384 | 94682 | 2167 | 399 |
| 0 | `shadow20_markov` | 167569 | -55000 | 86000 | 52.6 | 88.3% | 76446 | 214746 | 171223 | 12211 | 95011 | 2192 | 412 |
| 0 | `flat` | 0 | 0 | 0 | 0.0 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | `burnin1_markov` | 216321 | -265000 | 265000 | 146.7 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | `online_markov` | 208753 | -280000 | 280000 | 143.4 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | `risk50_burnin1_markov` | 187603 | -55000 | 65000 | 45.1 | 0.0% | 0 | 0 | 0 | 0 | 57326 | 1229 | 301 |
| 150,000 | `shadow8_markov` | 133051 | -54000 | 93000 | 45.5 | 94.9% | 48153 | 243247 | 135791 | 12548 | 95771 | 2161 | 6828 |
| 150,000 | `shadow12_markov` | 126924 | -57000 | 93000 | 43.8 | 94.9% | 56057 | 238849 | 130070 | 12039 | 95847 | 2167 | 6729 |
| 150,000 | `shadow20_markov` | 110211 | -55000 | 85000 | 39.4 | 88.3% | 76115 | 223145 | 114061 | 11900 | 96087 | 2192 | 6690 |
| 150,000 | `flat` | 0 | 0 | 0 | 0.0 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 150,000 | `burnin1_markov` | 152088 | -265000 | 265000 | 131.3 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 150,000 | `online_markov` | 144566 | -280000 | 280000 | 127.7 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 150,000 | `risk50_burnin1_markov` | 139851 | -55000 | 65000 | 37.0 | 0.0% | 0 | 0 | 0 | 0 | 57337 | 1229 | 4108 |
| 300,000 | `shadow8_markov` | 72593 | -56000 | 93000 | 29.2 | 94.9% | 46479 | 266117 | 75247 | 12403 | 97339 | 2161 | 16299 |
| 300,000 | `shadow12_markov` | 66360 | -57000 | 93000 | 27.6 | 94.9% | 54712 | 261063 | 69327 | 12111 | 97399 | 2167 | 16130 |
| 300,000 | `shadow20_markov` | 50708 | -55000 | 71000 | 23.0 | 88.3% | 74753 | 246881 | 54051 | 11848 | 97781 | 2192 | 16411 |
| 300,000 | `flat` | 0 | 0 | 0 | 0.0 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 300,000 | `burnin1_markov` | 84382 | -195000 | 200000 | 111.1 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 300,000 | `online_markov` | 77605 | -190000 | 194000 | 107.5 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 300,000 | `risk50_burnin1_markov` | 87912 | -55000 | 65000 | 27.8 | 0.0% | 0 | 0 | 0 | 0 | 58768 | 1229 | 9051 |
| 450,000 | `shadow8_markov` | 14540 | -57000 | 93000 | 10.4 | 94.9% | 50609 | 298070 | 14438 | 13163 | 98430 | 2151 | 29630 |
| 450,000 | `shadow12_markov` | 11926 | -57000 | 93000 | 9.8 | 94.9% | 59135 | 291027 | 12096 | 12794 | 98437 | 2155 | 29202 |
| 450,000 | `shadow20_markov` | 6213 | -57000 | 68000 | 7.2 | 88.3% | 79450 | 273002 | 6753 | 12124 | 98690 | 2176 | 28013 |
| 450,000 | `flat` | 0 | 0 | 0 | 0.0 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 450,000 | `burnin1_markov` | 17969 | -237000 | 237000 | 84.4 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 450,000 | `online_markov` | 13188 | -167000 | 194000 | 80.8 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 450,000 | `risk50_burnin1_markov` | 28239 | -55000 | 65000 | 15.6 | 0.0% | 0 | 0 | 0 | 0 | 63650 | 1229 | 17267 |

These fixed other-exposure levels are portfolio-sensitivity counterfactuals,
not a forecast of complete portfolio allocation.

## Paired P&L against flat

| exposure | shadow | paired runs | mean shadow - flat | median | shadow wins | ties | shadow losses |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `shadow8_markov` | 489 | 190350 | 29000 | 297 | 31 | 161 |
| 0 | `shadow12_markov` | 489 | 184286 | 25000 | 294 | 31 | 164 |
| 0 | `shadow20_markov` | 489 | 167569 | 5000 | 250 | 73 | 166 |
| 150,000 | `shadow8_markov` | 489 | 133051 | 27000 | 290 | 34 | 165 |
| 150,000 | `shadow12_markov` | 489 | 126924 | 19000 | 291 | 34 | 164 |
| 150,000 | `shadow20_markov` | 489 | 110211 | 3000 | 250 | 76 | 163 |
| 300,000 | `shadow8_markov` | 489 | 72593 | 20000 | 285 | 51 | 153 |
| 300,000 | `shadow12_markov` | 489 | 66360 | 13000 | 285 | 52 | 152 |
| 300,000 | `shadow20_markov` | 489 | 50708 | 1000 | 245 | 98 | 146 |
| 450,000 | `shadow8_markov` | 489 | 14540 | 0 | 152 | 262 | 75 |
| 450,000 | `shadow12_markov` | 489 | 11926 | 0 | 143 | 271 | 75 |
| 450,000 | `shadow20_markov` | 489 | 6213 | 0 | 95 | 337 | 57 |

## Paired P&L against `risk50_burnin1_markov`

| exposure | shadow | paired runs | mean shadow - risk50_burnin1_markov | median | shadow wins | ties | shadow losses |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `shadow8_markov` | 489 | 2746 | -28000 | 140 | 15 | 334 |
| 0 | `shadow12_markov` | 489 | -3317 | -30000 | 136 | 11 | 342 |
| 0 | `shadow20_markov` | 489 | -20035 | -35000 | 139 | 9 | 341 |
| 150,000 | `shadow8_markov` | 489 | -6800 | -25000 | 143 | 13 | 333 |
| 150,000 | `shadow12_markov` | 489 | -12926 | -28000 | 140 | 10 | 339 |
| 150,000 | `shadow20_markov` | 489 | -29640 | -33000 | 143 | 8 | 338 |
| 300,000 | `shadow8_markov` | 489 | -15319 | -20000 | 143 | 10 | 336 |
| 300,000 | `shadow12_markov` | 489 | -21552 | -21000 | 136 | 9 | 344 |
| 300,000 | `shadow20_markov` | 489 | -37204 | -29000 | 143 | 9 | 337 |
| 450,000 | `shadow8_markov` | 489 | -13699 | -11000 | 172 | 15 | 302 |
| 450,000 | `shadow12_markov` | 489 | -16313 | -11000 | 166 | 9 | 314 |
| 450,000 | `shadow20_markov` | 489 | -22027 | -11000 | 170 | 10 | 309 |

## Positive-upside retention and downside avoided

These are realised-path diagnostics relative to the existing wrapper.  A
positive-wrapper case reports whether the shadow run remains positive and the
shadow/wrapper P&L ratio.  A negative-wrapper case reports the fraction with a
higher shadow P&L and its mean improvement.

| exposure | shadow | positive wrapper cases | positive shadow fraction | positive P&L retention ratio | negative wrapper cases | downside avoided fraction | mean improvement on negative cases |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `shadow8_markov` | 288 | 86.5% | 4.42 | 194 | 44.3% | 12876 |
| 0 | `shadow12_markov` | 288 | 86.1% | 3.66 | 194 | 45.4% | 13696 |
| 0 | `shadow20_markov` | 288 | 71.2% | 3.47 | 194 | 47.4% | 14830 |
| 150,000 | `shadow8_markov` | 288 | 85.8% | 2.09 | 194 | 47.4% | 9531 |
| 150,000 | `shadow12_markov` | 288 | 86.1% | 1.52 | 194 | 48.5% | 10201 |
| 150,000 | `shadow20_markov` | 288 | 71.2% | 1.36 | 194 | 52.1% | 10680 |
| 300,000 | `shadow8_markov` | 288 | 84.7% | -0.12 | 194 | 54.1% | 10253 |
| 300,000 | `shadow12_markov` | 288 | 85.1% | -0.71 | 194 | 54.6% | 10397 |
| 300,000 | `shadow20_markov` | 288 | 70.5% | -0.85 | 194 | 60.3% | 12361 |
| 450,000 | `shadow8_markov` | 288 | 44.8% | -0.35 | 194 | 80.9% | 12794 |
| 450,000 | `shadow12_markov` | 288 | 43.8% | -0.44 | 194 | 82.0% | 12088 |
| 450,000 | `shadow20_markov` | 288 | 26.0% | -0.36 | 194 | 85.6% | 12923 |

## Callable-exposure integration audit

The audit used an existing scenario and a deterministic schedule of AUD 0 on
even days and AUD 590,000 on odd days.  It checks one underlying evaluation per
live focal day, cache sharing, no focal rejection, and whether the exposure
gate changes the endogenous path.  It does not manufacture a favourable
dynamic portfolio trace.

```json
{
  "at_most_one_underlying_evaluation_per_live_day": true,
  "budget_breaches": 0,
  "cache_hit_count": 2,
  "callable_marked_pnl": 5000,
  "callable_path_digest": "eb49c3417d627041",
  "exposure_schedule": "AUD 0 on even days, AUD 590000 on odd days",
  "fixed_path_digest": "251d167e27e27d93",
  "fixed_zero_exposure_marked_pnl": 10000,
  "live_day_count": 365,
  "path_changed_endogenously": true,
  "provider_call_count": 365,
  "provider_calls": [
    365,
    366,
    367,
    368,
    369,
    370,
    371,
    372,
    373,
    374,
    375,
    376,
    377,
    378,
    379,
    380,
    381,
    382,
    383,
    384,
    385,
    386,
    387,
    388,
    389,
    390,
    391,
    392,
    393,
    394,
    395,
    396,
    397,
    398,
    399,
    400,
    401,
    402,
    403,
    404,
    405,
    406,
    407,
    408,
    409,
    410,
    411,
    412,
    413,
    414,
    415,
    416,
    417,
    418,
    419,
    420,
    421,
    422,
    423,
    424,
    425,
    426,
    427,
    428,
    429,
    430,
    431,
    432,
    433,
    434,
    435,
    436,
    437,
    438,
    439,
    440,
    441,
    442,
    443,
    444,
    445,
    446,
    447,
    448,
    449,
    450,
    451,
    452,
    453,
    454,
    455,
    456,
    457,
    458,
    459,
    460,
    461,
    462,
    463,
    464,
    465,
    466,
    467,
    468,
    469,
    470,
    471,
    472,
    473,
    474,
    475,
    476,
    477,
    478,
    479,
    480,
    481,
    482,
    483,
    484,
    485,
    486,
    487,
    488,
    489,
    490,
    491,
    492,
    493,
    494,
    495,
    496,
    497,
    498,
    499,
    500,
    501,
    502,
    503,
    504,
    505,
    506,
    507,
    508,
    509,
    510,
    511,
    512,
    513,
    514,
    515,
    516,
    517,
    518,
    519,
    520,
    521,
    522,
    523,
    524,
    525,
    526,
    527,
    528,
    529,
    530,
    531,
    532,
    533,
    534,
    535,
    536,
    537,
    538,
    539,
    540,
    541,
    542,
    543,
    544,
    545,
    546,
    547,
    548,
    549,
    550,
    551,
    552,
    553,
    554,
    555,
    556,
    557,
    558,
    559,
    560,
    561,
    562,
    563,
    564,
    565,
    566,
    567,
    568,
    569,
    570,
    571,
    572,
    573,
    574,
    575,
    576,
    577,
    578,
    579,
    580,
    581,
    582,
    583,
    584,
    585,
    586,
    587,
    588,
    589,
    590,
    591,
    592,
    593,
    594,
    595,
    596,
    597,
    598,
    599,
    600,
    601,
    602,
    603,
    604,
    605,
    606,
    607,
    608,
    609,
    610,
    611,
    612,
    613,
    614,
    615,
    616,
    617,
    618,
    619,
    620,
    621,
    622,
    623,
    624,
    625,
    626,
    627,
    628,
    629,
    630,
    631,
    632,
    633,
    634,
    635,
    636,
    637,
    638,
    639,
    640,
    641,
    642,
    643,
    644,
    645,
    646,
    647,
    648,
    649,
    650,
    651,
    652,
    653,
    654,
    655,
    656,
    657,
    658,
    659,
    660,
    661,
    662,
    663,
    664,
    665,
    666,
    667,
    668,
    669,
    670,
    671,
    672,
    673,
    674,
    675,
    676,
    677,
    678,
    679,
    680,
    681,
    682,
    683,
    684,
    685,
    686,
    687,
    688,
    689,
    690,
    691,
    692,
    693,
    694,
    695,
    696,
    697,
    698,
    699,
    700,
    701,
    702,
    703,
    704,
    705,
    706,
    707,
    708,
    709,
    710,
    711,
    712,
    713,
    714,
    715,
    716,
    717,
    718,
    719,
    720,
    721,
    722,
    723,
    724,
    725,
    726,
    727,
    728,
    729
  ],
  "rejected_actions": 0,
  "scenario": "dev-persistent-long",
  "strategy": "shadow12_markov",
  "strategy_exposure_evaluations": 2,
  "unique_provider_days": 365
}
```

## Activation, undertrading, pivotality, and weaknesses

Initial and reactivation eligibility requires the fixed genuine-observation
warm-up, six scoreable virtual trades, cumulative virtual P&L of at least AUD
10,000, recent-window P&L of at least AUD 5,000, and a current economic edge.
Two newly scoreable qualifying evaluations are required, and the movement
that completes qualification is never traded retroactively.  A weak current
forecast is a real flat decision without automatically deactivating the
strategy.  Unknown, zero, reset, and floor-clipped public movements do not
enter shadow health evidence.

Pivotal and non-pivotal P&L are engine-only reporting partitions.  The strategy
receives no hidden counts, margin, pivotality, or simulator diagnostics.
Pivotal populations, short opportunities during warm-up, floor/clipped paths,
and a small qualifying sample remain material sources of undertrading or
false qualification risk.  The family and mode tables show where activation
is rare, shadow gating is unprofitable, or the wrapper has a better trade-off.

## Mechanical screening and selection

| candidate | eligible | mean P&L | worst P&L | max DD | breaches | rejected | loss consistency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `shadow8_markov` | False | 102633 | -57000 | 93000 | 0 | 0 | True |
| `shadow12_markov` | False | 97374 | -57000 | 93000 | 0 | 0 | True |
| `shadow20_markov` | False | 83675 | -57000 | 86000 | 0 | 0 | True |

The screening rule is fixed: zero focal budget breaches, zero focal rejected
actions, worst P&L at least AUD -60,000, maximum drawdown at most AUD 75,000,
positive aggregate mean P&L, and internally consistent loss-stop/overshoot
diagnostics.  Only eligible shadow candidates are considered.  If means are
within five percent, `shadow12_markov` wins; otherwise the highest aggregate
mean wins, with the longer warm-up as the prescribed fallback tie-break.

Selected challenger: **`None`**.

This is only the challenger for a future blind Pass 5B.  It is not a
production acceptance decision.

## Quarantine and remaining uncertainty

The consumed final artifacts were not accessed, parsed, imported, executed,
recreated, renamed, deleted, or overwritten.  No final-suite command was run,
and no final scenario constructor was called.  Production trading files and
the existing final catalogue were not modified.

No blind Pass 5B suite was created or executed.  Real-competition uncertainty
remains around active-team population size, focal pivotality frequency, public
zero/floor frequency, endogenous path effects, and how representative these
consumed development families are of the unseen room.
