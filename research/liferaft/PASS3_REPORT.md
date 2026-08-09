# Pass 3 cold-start Liferaft report

> This report covers **development + validation** only. Development cases are diagnostics;
> all rankings and conclusions below use validation runs only. The locked final
> suite was not executed. The organiser clarification makes Year 1 a constant
> $100,000 observation, so all Year-1 calibration/replay conclusions in
> `PASS2_REPORT.md` are superseded.

## Method and suite separation

The simulator uses `market_mode=inactive_until_marked`, with voting starting
on day 365. Price is exactly $100,000 through the day-365 observation. The day
365 action determines the first genuine movement into day 366; marked P&L
begins on day 366 and flat pre-voting history never produces a majority label.
The current best execution assumption is `observe_and_ignore_actions`, with
`fully_inactive` retained as a paired sensitivity case.

| suite | scenarios | strategies/scenario cells | role |
|---|---:|---:|---|
| development | 9 | 126 | correctness/design diagnostics only |
| validation | 480 | 6720 | sole source of rankings/conclusions |

Turnover is a stability diagnostic, not a transaction cost. The displayed
ranking applies no turnover penalty and no strategy parameters were tuned
after viewing validation results.

## Development diagnostics (not ranked)

| strategy | runs | mean | median | lower quartile | worst run | mean/day | mean DD | max DD | active | hit | turnover | breaches | rejected | beat flat | flat tied | regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat` | 9 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 | 0.0% | 55.6% | 129778 |
| `always_long` | 9 | 64444 | -54000 | -80000 | -80000 | 177.0 | 54444 | 90000 | 297.1 | 28.2% | 1.2 | 67.1 | 67.1 | 22.2% | 55.6% | 65333 |
| `always_short` | 9 | -318333 | -504000 | -504000 | -507000 | -874.5 | 336333 | 509000 | 182.8 | 6.5% | 1.7 | 181.9 | 181.9 | 22.2% | 55.6% | 448111 |
| `burnin1_markov` | 9 | 110889 | 0 | 0 | -89000 | 304.6 | 16556 | 94000 | 103.9 | 29.0% | 16.4 | 67.1 | 67.1 | 44.4% | 55.6% | 18889 |
| `burnin3_markov` | 9 | 105556 | 0 | 0 | -89000 | 290.0 | 16333 | 94000 | 103.3 | 28.9% | 16.4 | 67.1 | 67.1 | 44.4% | 55.6% | 24222 |
| `burnin5_markov` | 9 | 104889 | 0 | 0 | -94000 | 288.2 | 15333 | 94000 | 102.9 | 29.1% | 16.4 | 67.1 | 67.1 | 44.4% | 55.6% | 24889 |
| `burnin10_markov` | 9 | 87667 | 0 | 0 | -91000 | 240.8 | 17111 | 91000 | 99.9 | 28.5% | 15.6 | 67.1 | 67.1 | 44.4% | 55.6% | 42111 |
| `online_last_counter` | 9 | -58333 | 0 | 0 | -875000 | -160.3 | 186111 | 878000 | 174.3 | 28.6% | 97.3 | 68.0 | 68.0 | 44.4% | 55.6% | 188111 |
| `online_rolling` | 9 | 92667 | 0 | 0 | -135000 | 254.6 | 32778 | 148000 | 114.4 | 28.8% | 18.0 | 67.1 | 67.1 | 44.4% | 55.6% | 37111 |
| `online_markov` | 9 | 103222 | 0 | 0 | -74000 | 283.6 | 14889 | 74000 | 103.0 | 28.9% | 16.9 | 67.1 | 67.1 | 44.4% | 55.6% | 26556 |
| `online_ensemble` | 9 | 105222 | 0 | 0 | -92000 | 289.1 | 18222 | 92000 | 108.3 | 28.7% | 18.3 | 67.1 | 67.1 | 44.4% | 55.6% | 24556 |
| `online_drift` | 9 | 117667 | 0 | 0 | -26000 | 323.3 | 6556 | 33000 | 96.1 | 27.3% | 4.4 | 67.1 | 67.1 | 44.4% | 55.6% | 12111 |
| `immediate_long_prior` | 9 | 64444 | -54000 | -80000 | -80000 | 177.0 | 54444 | 90000 | 297.1 | 28.2% | 1.2 | 67.1 | 67.1 | 22.2% | 55.6% | 65333 |
| `flat_first_long_prior` | 9 | 61778 | -65000 | -70000 | -80000 | 169.7 | 52333 | 111000 | 294.1 | 28.3% | 1.2 | 67.1 | 67.1 | 22.2% | 55.6% | 68000 |

## Actual validation distribution

The following are computed directly across actual validation runs. `worst run`,
lower quartile, and `max DD` are not averages of family-level statistics.

| strategy | runs | mean | median | lower quartile | worst run | mean/day | mean DD | max DD | active | hit | turnover | breaches | rejected | beat flat | flat tied | regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat` | 480 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 | 0.0% | 25.8% | 287340 |
| `always_long` | 480 | 121658 | -67000 | -80000 | -80000 | 334.2 | 62548 | 254000 | 304.7 | 30.3% | 1.3 | 59.6 | 59.6 | 39.6% | 25.8% | 165681 |
| `always_short` | 480 | -364077 | -503000 | -505000 | -508000 | -1000.2 | 392119 | 525000 | 195.9 | 19.4% | 1.7 | 168.8 | 168.8 | 22.1% | 25.8% | 651417 |
| `burnin1_markov` | 480 | 218298 | 38000 | -47000 | -265000 | 599.7 | 46671 | 265000 | 147.3 | 41.4% | 96.7 | 66.4 | 66.4 | 60.0% | 25.8% | 69042 |
| `burnin3_markov` | 480 | 216896 | 38000 | -48000 | -275000 | 595.9 | 46162 | 275000 | 146.4 | 41.4% | 96.0 | 66.5 | 66.5 | 59.8% | 25.8% | 70444 |
| `burnin5_markov` | 480 | 215588 | 39500 | -46000 | -270000 | 592.3 | 46223 | 270000 | 145.4 | 41.5% | 95.3 | 66.9 | 66.9 | 60.4% | 25.8% | 71752 |
| `burnin10_markov` | 480 | 205348 | 22000 | -45000 | -290000 | 564.1 | 46360 | 290000 | 143.9 | 41.3% | 94.2 | 66.9 | 66.9 | 57.1% | 25.8% | 81992 |
| `online_last_counter` | 480 | -543631 | -449000 | -948000 | -2174000 | -1493.5 | 637273 | 2174000 | 284.2 | 27.9% | 276.3 | 77.5 | 77.5 | 25.0% | 25.8% | 830971 |
| `online_rolling` | 480 | 51581 | -8000 | -130000 | -540000 | 141.7 | 80744 | 540000 | 97.9 | 32.3% | 38.0 | 66.5 | 66.5 | 44.6% | 25.8% | 235758 |
| `online_markov` | 480 | 210731 | 32000 | -46000 | -280000 | 578.9 | 46125 | 280000 | 143.9 | 41.7% | 95.8 | 67.1 | 67.1 | 60.0% | 25.8% | 76608 |
| `online_ensemble` | 480 | 185677 | -5000 | -76000 | -295000 | 510.1 | 64827 | 295000 | 136.8 | 39.1% | 97.4 | 69.3 | 69.3 | 46.7% | 25.8% | 101662 |
| `online_drift` | 480 | 169496 | 0 | -17000 | -233000 | 465.6 | 30994 | 275000 | 92.1 | 34.9% | 53.5 | 47.1 | 47.1 | 47.1% | 25.8% | 117844 |
| `immediate_long_prior` | 480 | 121658 | -67000 | -80000 | -80000 | 334.2 | 62548 | 254000 | 304.7 | 30.3% | 1.3 | 59.6 | 59.6 | 39.6% | 25.8% | 165681 |
| `flat_first_long_prior` | 480 | 118404 | -65000 | -75000 | -104000 | 325.3 | 61277 | 254000 | 301.3 | 30.4% | 1.3 | 60.0 | 60.0 | 39.0% | 25.8% | 168935 |

No-turnover validation ranking by actual mean P&L per marked day:
`burnin1_markov > burnin3_markov > burnin5_markov > online_markov > burnin10_markov > online_ensemble > online_drift > always_long > immediate_long_prior > flat_first_long_prior > online_rolling > flat > always_short > online_last_counter`.

## Family-balanced comparison

These rows average only each family's mean P&L and mean P&L/day. Quantiles,
worst runs, drawdowns, and fractions remain in the actual validation table.

| strategy | families | family-balanced mean marked P&L | family-balanced mean P&L/day |
|---|---:|---:|---:|
| `flat` | 12 | 0 | 0.0 |
| `always_long` | 12 | 121658 | 334.2 |
| `always_short` | 12 | -364077 | -1000.2 |
| `burnin1_markov` | 12 | 218298 | 599.7 |
| `burnin3_markov` | 12 | 216896 | 595.9 |
| `burnin5_markov` | 12 | 215588 | 592.3 |
| `burnin10_markov` | 12 | 205348 | 564.1 |
| `online_last_counter` | 12 | -543631 | -1493.5 |
| `online_rolling` | 12 | 51581 | 141.7 |
| `online_markov` | 12 | 210731 | 578.9 |
| `online_ensemble` | 12 | 185677 | 510.1 |
| `online_drift` | 12 | 169496 | 465.6 |
| `immediate_long_prior` | 12 | 121658 | 334.2 |
| `flat_first_long_prior` | 12 | 118404 | 325.3 |

Family-balanced mean P&L/day ranking:
`burnin1_markov > burnin3_markov > burnin5_markov > online_markov > burnin10_markov > online_ensemble > online_drift > always_long > immediate_long_prior > flat_first_long_prior > online_rolling > flat > always_short > online_last_counter`.

## Results by validation family

| family | strategy | runs | actual mean P&L | actual mean/day | actual worst run | beat flat | pivotal P&L | non-pivotal P&L |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| balanced_random | `always_long` | 40 | -45900 | -126.1 | -80000 | 10.0% | -218700 | 172800 |
| balanced_random | `always_short` | 40 | -503550 | -1383.4 | -508000 | 0.0% | -184400 | -319150 |
| balanced_random | `burnin10_markov` | 40 | -59350 | -163.0 | -145000 | 5.0% | -35250 | -24100 |
| balanced_random | `burnin1_markov` | 40 | -49650 | -136.4 | -97000 | 5.0% | -37350 | -12300 |
| balanced_random | `burnin3_markov` | 40 | -50700 | -139.3 | -109000 | 5.0% | -37500 | -13200 |
| balanced_random | `burnin5_markov` | 40 | -53250 | -146.3 | -120000 | 5.0% | -36850 | -16400 |
| balanced_random | `flat` | 40 | 0 | 0.0 | 0 | 0.0% | 0 | 0 |
| balanced_random | `flat_first_long_prior` | 40 | -48850 | -134.2 | -91000 | 10.0% | -217800 | 168950 |
| balanced_random | `immediate_long_prior` | 40 | -45900 | -126.1 | -80000 | 10.0% | -218700 | 172800 |
| balanced_random | `online_drift` | 40 | -4850 | -13.3 | -21000 | 30.0% | -5050 | 200 |
| balanced_random | `online_ensemble` | 40 | -59850 | -164.4 | -147000 | 5.0% | -51500 | -8350 |
| balanced_random | `online_last_counter` | 40 | -753500 | -2070.1 | -1108000 | 0.0% | -364700 | -388800 |
| balanced_random | `online_markov` | 40 | -43150 | -118.5 | -109000 | 10.0% | -35000 | -8150 |
| balanced_random | `online_rolling` | 40 | -119000 | -326.9 | -214000 | 0.0% | -80900 | -38100 |
| gradual_drift | `always_long` | 40 | 101750 | 279.5 | -80000 | 35.0% | -174950 | 276700 |
| gradual_drift | `always_short` | 40 | -360750 | -991.1 | -508000 | 25.0% | -214000 | -146750 |
| gradual_drift | `burnin10_markov` | 40 | 42950 | 118.0 | -76000 | 50.0% | -58750 | 101700 |
| gradual_drift | `burnin1_markov` | 40 | 22550 | 62.0 | -100000 | 45.0% | -63100 | 85650 |
| gradual_drift | `burnin3_markov` | 40 | 27550 | 75.7 | -91000 | 45.0% | -61600 | 89150 |
| gradual_drift | `burnin5_markov` | 40 | 33250 | 91.3 | -86000 | 50.0% | -60200 | 93450 |
| gradual_drift | `flat` | 40 | 0 | 0.0 | 0 | 0.0% | 0 | 0 |
| gradual_drift | `flat_first_long_prior` | 40 | 99250 | 272.7 | -104000 | 35.0% | -174350 | 273600 |
| gradual_drift | `immediate_long_prior` | 40 | 101750 | 279.5 | -80000 | 35.0% | -174950 | 276700 |
| gradual_drift | `online_drift` | 40 | -9450 | -26.0 | -20000 | 20.0% | -4500 | -4950 |
| gradual_drift | `online_ensemble` | 40 | 12500 | 34.3 | -130000 | 40.0% | -63900 | 76400 |
| gradual_drift | `online_last_counter` | 40 | -530250 | -1456.7 | -1118000 | 5.0% | -328200 | -202050 |
| gradual_drift | `online_markov` | 40 | 41600 | 114.3 | -75000 | 50.0% | -62050 | 103650 |
| gradual_drift | `online_rolling` | 40 | 25450 | 69.9 | -188000 | 40.0% | -97650 | 123100 |
| history_rules | `always_long` | 40 | 226100 | 621.2 | -80000 | 65.0% | -94100 | 320200 |
| history_rules | `always_short` | 40 | -499900 | -1373.4 | -507000 | 0.0% | -148400 | -351500 |
| history_rules | `burnin10_markov` | 40 | 1182300 | 3248.1 | -99000 | 75.0% | -85200 | 1267500 |
| history_rules | `burnin1_markov` | 40 | 1196550 | 3287.2 | -89000 | 75.0% | -87750 | 1284300 |
| history_rules | `burnin3_markov` | 40 | 1200200 | 3297.3 | -99000 | 75.0% | -87350 | 1287550 |
| history_rules | `burnin5_markov` | 40 | 1196500 | 3287.1 | -116000 | 75.0% | -86250 | 1282750 |
| history_rules | `flat` | 40 | 0 | 0.0 | 0 | 0.0% | 0 | 0 |
| history_rules | `flat_first_long_prior` | 40 | 222550 | 611.4 | -96000 | 65.0% | -92450 | 315000 |
| history_rules | `immediate_long_prior` | 40 | 226100 | 621.2 | -80000 | 65.0% | -94100 | 320200 |
| history_rules | `online_drift` | 40 | 1089250 | 2992.4 | -19000 | 75.0% | -31150 | 1120400 |
| history_rules | `online_ensemble` | 40 | 1155650 | 3174.9 | -116000 | 75.0% | -66450 | 1222100 |
| history_rules | `online_last_counter` | 40 | -1902150 | -5225.7 | -2174000 | 0.0% | -240250 | -1661900 |
| history_rules | `online_markov` | 40 | 1185700 | 3257.4 | -117000 | 80.0% | -84450 | 1270150 |
| history_rules | `online_rolling` | 40 | -79750 | -219.1 | -217000 | 0.0% | -23050 | -56700 |
| long_biased_random | `always_long` | 40 | -79600 | -218.7 | -80000 | 0.0% | -22750 | -56850 |
| long_biased_random | `always_short` | 40 | 11200 | 30.8 | -283000 | 75.0% | -296800 | 308000 |
| long_biased_random | `burnin10_markov` | 40 | -20750 | -57.0 | -158000 | 45.0% | -149700 | 128950 |
| long_biased_random | `burnin1_markov` | 40 | -2800 | -7.7 | -126000 | 60.0% | -145400 | 142600 |
| long_biased_random | `burnin3_markov` | 40 | -1800 | -4.9 | -126000 | 60.0% | -145400 | 143600 |
| long_biased_random | `burnin5_markov` | 40 | -250 | -0.7 | -126000 | 60.0% | -145550 | 145300 |
| long_biased_random | `flat` | 40 | 0 | 0.0 | 0 | 0.0% | 0 | 0 |
| long_biased_random | `flat_first_long_prior` | 40 | -70900 | -194.8 | -91000 | 0.0% | -21750 | -49150 |
| long_biased_random | `immediate_long_prior` | 40 | -79600 | -218.7 | -80000 | 0.0% | -22750 | -56850 |
| long_biased_random | `online_drift` | 40 | -80150 | -220.2 | -233000 | 0.0% | -97050 | 16900 |
| long_biased_random | `online_ensemble` | 40 | -126000 | -346.2 | -233000 | 0.0% | -132500 | 6500 |
| long_biased_random | `online_last_counter` | 40 | -471650 | -1295.7 | -735000 | 0.0% | -272400 | -199250 |
| long_biased_random | `online_markov` | 40 | -10650 | -29.3 | -161000 | 60.0% | -145450 | 134800 |
| long_biased_random | `online_rolling` | 40 | -17050 | -46.8 | -128000 | 45.0% | -51700 | 34650 |
| margin_mixture | `always_long` | 40 | 125050 | 343.5 | -80000 | 35.0% | -22850 | 147900 |
| margin_mixture | `always_short` | 40 | -246050 | -676.0 | -508000 | 45.0% | -137200 | -108850 |
| margin_mixture | `burnin10_markov` | 40 | 104500 | 287.1 | -290000 | 65.0% | -98650 | 203150 |
| margin_mixture | `burnin1_markov` | 40 | 135700 | 372.8 | -265000 | 80.0% | -99650 | 235350 |
| margin_mixture | `burnin3_markov` | 40 | 131600 | 361.5 | -275000 | 80.0% | -100900 | 232500 |
| margin_mixture | `burnin5_markov` | 40 | 125300 | 344.2 | -270000 | 80.0% | -98950 | 224250 |
| margin_mixture | `flat` | 40 | 0 | 0.0 | 0 | 0.0% | 0 | 0 |
| margin_mixture | `flat_first_long_prior` | 40 | 122250 | 335.9 | -104000 | 35.0% | -23650 | 145900 |
| margin_mixture | `immediate_long_prior` | 40 | 125050 | 343.5 | -80000 | 35.0% | -22850 | 147900 |
| margin_mixture | `online_drift` | 40 | 116850 | 321.0 | -131000 | 45.0% | -64350 | 181200 |
| margin_mixture | `online_ensemble` | 40 | 90650 | 249.0 | -295000 | 45.0% | -94000 | 184650 |
| margin_mixture | `online_last_counter` | 40 | -106600 | -292.9 | -806000 | 35.0% | -149700 | 43100 |
| margin_mixture | `online_markov` | 40 | 112050 | 307.8 | -280000 | 65.0% | -97900 | 209950 |
| margin_mixture | `online_rolling` | 40 | 132150 | 363.0 | -275000 | 75.0% | -53950 | 186100 |
| periodic | `always_long` | 40 | 92850 | 255.1 | -80000 | 35.0% | -135600 | 228450 |
| periodic | `always_short` | 40 | -417100 | -1145.9 | -508000 | 15.0% | -211600 | -205500 |
| periodic | `burnin10_markov` | 40 | 308800 | 848.4 | -185000 | 80.0% | -100550 | 409350 |
| periodic | `burnin1_markov` | 40 | 305800 | 840.1 | -195000 | 80.0% | -103700 | 409500 |
| periodic | `burnin3_markov` | 40 | 303150 | 832.8 | -195000 | 80.0% | -102750 | 405900 |
| periodic | `burnin5_markov` | 40 | 307400 | 844.5 | -208000 | 80.0% | -100850 | 408250 |
| periodic | `flat` | 40 | 0 | 0.0 | 0 | 0.0% | 0 | 0 |
| periodic | `flat_first_long_prior` | 40 | 91000 | 250.0 | -91000 | 35.0% | -134350 | 225350 |
| periodic | `immediate_long_prior` | 40 | 92850 | 255.1 | -80000 | 35.0% | -135600 | 228450 |
| periodic | `online_drift` | 40 | 217350 | 597.1 | -204000 | 65.0% | -48300 | 265650 |
| periodic | `online_ensemble` | 40 | 292050 | 802.3 | -204000 | 70.0% | -77400 | 369450 |
| periodic | `online_last_counter` | 40 | -1000500 | -2748.6 | -2166000 | 15.0% | -358150 | -642350 |
| periodic | `online_markov` | 40 | 294950 | 810.3 | -167000 | 80.0% | -97800 | 392750 |
| periodic | `online_rolling` | 40 | -15950 | -43.8 | -540000 | 30.0% | -64350 | 48400 |
| persistent_long | `always_long` | 40 | -80000 | -219.8 | -80000 | 0.0% | -3400 | -76600 |
| persistent_long | `always_short` | 40 | 78850 | 216.6 | 63000 | 100.0% | -100800 | 179650 |
| persistent_long | `burnin10_markov` | 40 | 18650 | 51.2 | -32000 | 85.0% | -88800 | 107450 |
| persistent_long | `burnin1_markov` | 40 | 35750 | 98.2 | -16000 | 90.0% | -90650 | 126400 |
| persistent_long | `burnin3_markov` | 40 | 36000 | 98.9 | -16000 | 90.0% | -90650 | 126650 |
| persistent_long | `burnin5_markov` | 40 | 34750 | 95.5 | -21000 | 90.0% | -90250 | 125000 |
| persistent_long | `flat` | 40 | 0 | 0.0 | 0 | 0.0% | 0 | 0 |
| persistent_long | `flat_first_long_prior` | 40 | -66550 | -182.8 | -78000 | 0.0% | -3150 | -63400 |
| persistent_long | `immediate_long_prior` | 40 | -80000 | -219.8 | -80000 | 0.0% | -3400 | -76600 |
| persistent_long | `online_drift` | 40 | -11850 | -32.6 | -137000 | 50.0% | -68800 | 56950 |
| persistent_long | `online_ensemble` | 40 | -12500 | -34.3 | -137000 | 50.0% | -70400 | 57900 |
| persistent_long | `online_last_counter` | 40 | -106250 | -291.9 | -415000 | 40.0% | -97050 | -9200 |
| persistent_long | `online_markov` | 40 | 25700 | 70.6 | -27000 | 80.0% | -91200 | 116900 |
| persistent_long | `online_rolling` | 40 | 35650 | 97.9 | -26000 | 95.0% | -28000 | 63650 |
| persistent_short | `always_long` | 40 | 503850 | 1384.2 | 501000 | 100.0% | -23000 | 526850 |
| persistent_short | `always_short` | 40 | -504000 | -1384.6 | -507000 | 0.0% | -26400 | -477600 |
| persistent_short | `burnin10_markov` | 40 | 426200 | 1170.9 | 413000 | 100.0% | -19500 | 445700 |
| persistent_short | `burnin1_markov` | 40 | 482900 | 1326.6 | 413000 | 100.0% | -21750 | 504650 |
| persistent_short | `burnin3_markov` | 40 | 474150 | 1302.6 | 439000 | 100.0% | -20750 | 494900 |
| persistent_short | `burnin5_markov` | 40 | 463100 | 1272.3 | 442000 | 100.0% | -20000 | 483100 |
| persistent_short | `flat` | 40 | 0 | 0.0 | 0 | 0.0% | 0 | 0 |
| persistent_short | `flat_first_long_prior` | 40 | 481800 | 1323.6 | 477000 | 100.0% | -21500 | 503300 |
| persistent_short | `immediate_long_prior` | 40 | 503850 | 1384.2 | 501000 | 100.0% | -23000 | 526850 |
| persistent_short | `online_drift` | 40 | 441100 | 1211.8 | 9000 | 100.0% | -18000 | 459100 |
| persistent_short | `online_ensemble` | 40 | 461000 | 1266.5 | 407000 | 100.0% | -20000 | 481000 |
| persistent_short | `online_last_counter` | 40 | 404050 | 1110.0 | 180000 | 100.0% | -23400 | 427450 |
| persistent_short | `online_markov` | 40 | 441700 | 1213.5 | 407000 | 100.0% | -18000 | 459700 |
| persistent_short | `online_rolling` | 40 | 475050 | 1305.1 | 415000 | 100.0% | -21250 | 496300 |
| reactive_mixture | `always_long` | 40 | 65750 | 180.6 | -80000 | 45.0% | -143550 | 209300 |
| reactive_mixture | `always_short` | 40 | -414375 | -1138.4 | -508000 | 5.0% | -285400 | -128975 |
| reactive_mixture | `burnin10_markov` | 40 | 74950 | 205.9 | -158000 | 47.5% | -93925 | 168875 |
| reactive_mixture | `burnin1_markov` | 40 | 77100 | 211.8 | -143000 | 52.5% | -92175 | 169275 |
| reactive_mixture | `burnin3_markov` | 40 | 76300 | 209.6 | -138000 | 50.0% | -92800 | 169100 |
| reactive_mixture | `burnin5_markov` | 40 | 81375 | 223.6 | -138000 | 52.5% | -91325 | 172700 |
| reactive_mixture | `flat` | 40 | 0 | 0.0 | 0 | 0.0% | 0 | 0 |
| reactive_mixture | `flat_first_long_prior` | 40 | 63800 | 175.3 | -104000 | 40.0% | -143000 | 206800 |
| reactive_mixture | `immediate_long_prior` | 40 | 65750 | 180.6 | -80000 | 45.0% | -143550 | 209300 |
| reactive_mixture | `online_drift` | 40 | 14450 | 39.7 | -122000 | 35.0% | -22800 | 37250 |
| reactive_mixture | `online_ensemble` | 40 | 50275 | 138.1 | -194000 | 40.0% | -84700 | 134975 |
| reactive_mixture | `online_last_counter` | 40 | -875450 | -2405.1 | -1963000 | 15.0% | -339400 | -536050 |
| reactive_mixture | `online_markov` | 40 | 83300 | 228.8 | -158000 | 55.0% | -88650 | 171950 |
| reactive_mixture | `online_rolling` | 40 | -111550 | -306.5 | -470000 | 10.0% | -98550 | -13000 |
| regime_change | `always_long` | 40 | 110550 | 303.7 | -80000 | 45.0% | -167550 | 278100 |
| regime_change | `always_short` | 40 | -503450 | -1383.1 | -508000 | 0.0% | -138000 | -365450 |
| regime_change | `burnin10_markov` | 40 | 42200 | 115.9 | -80000 | 30.0% | -52200 | 94400 |
| regime_change | `burnin1_markov` | 40 | 47450 | 130.4 | -82000 | 30.0% | -55650 | 103100 |
| regime_change | `burnin3_markov` | 40 | 43200 | 118.7 | -85000 | 30.0% | -54650 | 97850 |
| regime_change | `burnin5_markov` | 40 | 40100 | 110.2 | -80000 | 30.0% | -53850 | 93950 |
| regime_change | `flat` | 40 | 0 | 0.0 | 0 | 0.0% | 0 | 0 |
| regime_change | `flat_first_long_prior` | 40 | 104400 | 286.8 | -88000 | 45.0% | -166700 | 271100 |
| regime_change | `immediate_long_prior` | 40 | 110550 | 303.7 | -80000 | 45.0% | -167550 | 278100 |
| regime_change | `online_drift` | 40 | -1650 | -4.5 | -20000 | 40.0% | -4800 | 3150 |
| regime_change | `online_ensemble` | 40 | 33850 | 93.0 | -123000 | 35.0% | -47350 | 81200 |
| regime_change | `online_last_counter` | 40 | -531650 | -1460.6 | -1108000 | 10.0% | -242800 | -288850 |
| regime_change | `online_markov` | 40 | 48050 | 132.0 | -96000 | 35.0% | -51250 | 99300 |
| regime_change | `online_rolling` | 40 | -4800 | -13.2 | -217000 | 40.0% | -69650 | 64850 |
| short_biased_random | `always_long` | 40 | 499700 | 1372.8 | 432000 | 100.0% | -92000 | 591700 |
| short_biased_random | `always_short` | 40 | -505800 | -1389.6 | -508000 | 0.0% | -56800 | -449000 |
| short_biased_random | `burnin10_markov` | 40 | 400800 | 1101.1 | 231000 | 100.0% | -63000 | 463800 |
| short_biased_random | `burnin1_markov` | 40 | 428300 | 1176.6 | 204000 | 100.0% | -65750 | 494050 |
| short_biased_random | `burnin3_markov` | 40 | 423950 | 1164.7 | 204000 | 100.0% | -65250 | 489200 |
| short_biased_random | `burnin5_markov` | 40 | 421200 | 1157.1 | 204000 | 100.0% | -66500 | 487700 |
| short_biased_random | `flat` | 40 | 0 | 0.0 | 0 | 0.0% | 0 | 0 |
| short_biased_random | `flat_first_long_prior` | 40 | 485700 | 1334.3 | 442000 | 100.0% | -88500 | 574200 |
| short_biased_random | `immediate_long_prior` | 40 | 499700 | 1372.8 | 432000 | 100.0% | -92000 | 591700 |
| short_biased_random | `online_drift` | 40 | 271600 | 746.2 | -15000 | 90.0% | -19500 | 291100 |
| short_biased_random | `online_ensemble` | 40 | 396450 | 1089.1 | 146000 | 100.0% | -56000 | 452450 |
| short_biased_random | `online_last_counter` | 40 | 188550 | 518.0 | -283000 | 80.0% | -68400 | 256950 |
| short_biased_random | `online_markov` | 40 | 404350 | 1110.9 | 298000 | 100.0% | -61000 | 465350 |
| short_biased_random | `online_rolling` | 40 | 410850 | 1128.7 | 159000 | 100.0% | -59750 | 470600 |
| startup_zero_history | `always_long` | 40 | -60200 | -165.4 | -80000 | 5.0% | -232625 | 172425 |
| startup_zero_history | `always_short` | 40 | -504000 | -1384.6 | -508000 | 0.0% | -219000 | -285000 |
| startup_zero_history | `burnin10_markov` | 40 | -57075 | -156.8 | -123000 | 2.5% | -44550 | -12525 |
| startup_zero_history | `burnin1_markov` | 40 | -60075 | -165.0 | -133000 | 2.5% | -47075 | -13000 |
| startup_zero_history | `burnin3_markov` | 40 | -60850 | -167.2 | -133000 | 2.5% | -46425 | -14425 |
| startup_zero_history | `burnin5_markov` | 40 | -62425 | -171.5 | -128000 | 2.5% | -45975 | -16450 |
| startup_zero_history | `flat` | 40 | 0 | 0.0 | 0 | 0.0% | 0 | 0 |
| startup_zero_history | `flat_first_long_prior` | 40 | -63600 | -174.7 | -96000 | 2.5% | -233525 | 169925 |
| startup_zero_history | `immediate_long_prior` | 40 | -60200 | -165.4 | -80000 | 5.0% | -232625 | 172425 |
| startup_zero_history | `online_drift` | 40 | -8700 | -23.9 | -33000 | 15.0% | -6175 | -2525 |
| startup_zero_history | `online_ensemble` | 40 | -65950 | -181.2 | -148000 | 0.0% | -58100 | -7850 |
| startup_zero_history | `online_last_counter` | 40 | -838175 | -2302.7 | -1192000 | 0.0% | -442600 | -395575 |
| startup_zero_history | `online_markov` | 40 | -54825 | -150.6 | -112000 | 5.0% | -45025 | -9800 |
| startup_zero_history | `online_rolling` | 40 | -112075 | -307.9 | -191000 | 0.0% | -89075 | -23000 |

## Execution-mode sensitivity

| execution mode | strategy | runs | actual mean P&L | actual mean/day | beat flat | flat tied |
|---|---|---:|---:|---:|---:|---:|
| fully_inactive | `always_long` | 240 | 121858 | 334.8 | 40.0% | 25.8% |
| fully_inactive | `always_short` | 240 | -363979 | -999.9 | 22.1% | 25.8% |
| fully_inactive | `burnin10_markov` | 240 | 205608 | 564.9 | 57.1% | 25.8% |
| fully_inactive | `burnin1_markov` | 240 | 217771 | 598.3 | 59.6% | 25.8% |
| fully_inactive | `burnin3_markov` | 240 | 216588 | 595.0 | 59.6% | 25.8% |
| fully_inactive | `burnin5_markov` | 240 | 216175 | 593.9 | 60.8% | 25.8% |
| fully_inactive | `flat` | 240 | 0 | 0.0 | 0.0% | 25.8% |
| fully_inactive | `flat_first_long_prior` | 240 | 118271 | 324.9 | 39.2% | 25.8% |
| fully_inactive | `immediate_long_prior` | 240 | 121858 | 334.8 | 40.0% | 25.8% |
| fully_inactive | `online_drift` | 240 | 168742 | 463.6 | 47.5% | 25.8% |
| fully_inactive | `online_ensemble` | 240 | 183846 | 505.1 | 46.2% | 25.8% |
| fully_inactive | `online_last_counter` | 240 | -543196 | -1492.3 | 25.0% | 25.8% |
| fully_inactive | `online_markov` | 240 | 211129 | 580.0 | 60.0% | 25.8% |
| fully_inactive | `online_rolling` | 240 | 51767 | 142.2 | 44.6% | 25.8% |
| observe_and_ignore_actions | `always_long` | 240 | 121458 | 333.7 | 39.2% | 25.8% |
| observe_and_ignore_actions | `always_short` | 240 | -364175 | -1000.5 | 22.1% | 25.8% |
| observe_and_ignore_actions | `burnin10_markov` | 240 | 205088 | 563.4 | 57.1% | 25.8% |
| observe_and_ignore_actions | `burnin1_markov` | 240 | 218825 | 601.2 | 60.4% | 25.8% |
| observe_and_ignore_actions | `burnin3_markov` | 240 | 217204 | 596.7 | 60.0% | 25.8% |
| observe_and_ignore_actions | `burnin5_markov` | 240 | 215000 | 590.7 | 60.0% | 25.8% |
| observe_and_ignore_actions | `flat` | 240 | 0 | 0.0 | 0.0% | 25.8% |
| observe_and_ignore_actions | `flat_first_long_prior` | 240 | 118538 | 325.7 | 38.8% | 25.8% |
| observe_and_ignore_actions | `immediate_long_prior` | 240 | 121458 | 333.7 | 39.2% | 25.8% |
| observe_and_ignore_actions | `online_drift` | 240 | 170250 | 467.7 | 46.7% | 25.8% |
| observe_and_ignore_actions | `online_ensemble` | 240 | 187508 | 515.1 | 47.1% | 25.8% |
| observe_and_ignore_actions | `online_last_counter` | 240 | -544067 | -1494.7 | 25.0% | 25.8% |
| observe_and_ignore_actions | `online_markov` | 240 | 210333 | 577.8 | 60.0% | 25.8% |
| observe_and_ignore_actions | `online_rolling` | 240 | 51396 | 141.2 | 44.6% | 25.8% |

### Controlled paired differences

| strategy | paired cases | observe minus fully inactive mean P&L | min difference | max difference |
|---|---:|---:|---:|---:|
| `flat` | 240 | 0 | 0 | 0 |
| `always_long` | 240 | -400 | -71000 | 63000 |
| `always_short` | 240 | -196 | -83000 | 32000 |
| `burnin1_markov` | 240 | 1054 | -100000 | 102000 |
| `burnin3_markov` | 240 | 617 | -93000 | 102000 |
| `burnin5_markov` | 240 | -1175 | -166000 | 73000 |
| `burnin10_markov` | 240 | -521 | -149000 | 107000 |
| `online_last_counter` | 240 | -871 | -160000 | 110000 |
| `online_rolling` | 240 | -371 | -77000 | 70000 |
| `online_markov` | 240 | -796 | -145000 | 60000 |
| `online_ensemble` | 240 | 3662 | -62000 | 258000 |
| `online_drift` | 240 | 1508 | -57000 | 215000 |
| `immediate_long_prior` | 240 | -400 | -71000 | 63000 |
| `flat_first_long_prior` | 240 | 267 | -63000 | 58000 |

Positive/negative differences for controlled day-indexed populations measure
startup execution only. Reactive and startup-state pairs are labelled as
state/RNG evolution because their Year-1 calls intentionally change object
state or RNG consumption.

### Paired candidate comparison

These comparisons use the same validation scenario for both candidates. They
are not a separate holdout and do not apply a turnover cost.

| left strategy | right strategy | paired cases | mean left-minus-right P&L | median left-minus-right P&L | left wins | ties | left losses |
|---|---|---:|---:|---:|---:|---:|---:|
| `burnin1_markov` | `burnin3_markov` | 480 | 1402 | 0 | 124 | 242 | 114 |
| `burnin1_markov` | `online_markov` | 480 | 7567 | 9500 | 297 | 8 | 175 |
| `burnin3_markov` | `online_markov` | 480 | 6165 | 9000 | 297 | 13 | 170 |

## Unique-path audit

The table below is a path-diversity diagnostic over flat-focal live majority
paths. Exact uniqueness and pairwise Hamming distances show effective path
variation within each family/mode; they are not claims of statistical
independence. Naturally persistent families may have shallow distances.

| family | mode | cases | unique live path signatures | duplicate cases | min pairwise differing live days | mean pairwise differing live days |
|---|---|---:|---:|---:|---:|---:|
| balanced_random | fully_inactive | 20 | 20 | 0 | 157 | 232.2 |
| balanced_random | observe_and_ignore_actions | 20 | 20 | 0 | 157 | 232.2 |
| gradual_drift | fully_inactive | 20 | 20 | 0 | 64 | 224.8 |
| gradual_drift | observe_and_ignore_actions | 20 | 20 | 0 | 64 | 224.8 |
| history_rules | fully_inactive | 20 | 20 | 0 | 2 | 203.3 |
| history_rules | observe_and_ignore_actions | 20 | 20 | 0 | 2 | 203.3 |
| long_biased_random | fully_inactive | 20 | 20 | 0 | 43 | 128.4 |
| long_biased_random | observe_and_ignore_actions | 20 | 20 | 0 | 43 | 128.4 |
| margin_mixture | fully_inactive | 20 | 20 | 0 | 2 | 201.8 |
| margin_mixture | observe_and_ignore_actions | 20 | 20 | 0 | 2 | 201.8 |
| periodic | fully_inactive | 20 | 20 | 0 | 27 | 227.6 |
| periodic | observe_and_ignore_actions | 20 | 20 | 0 | 27 | 227.6 |
| persistent_long | fully_inactive | 20 | 20 | 0 | 1 | 38.5 |
| persistent_long | observe_and_ignore_actions | 20 | 20 | 0 | 1 | 38.5 |
| persistent_short | fully_inactive | 20 | 16 | 4 | 0 | 17.1 |
| persistent_short | observe_and_ignore_actions | 20 | 16 | 4 | 0 | 17.1 |
| reactive_mixture | fully_inactive | 20 | 20 | 0 | 58 | 232.3 |
| reactive_mixture | observe_and_ignore_actions | 20 | 20 | 0 | 56 | 232.9 |
| regime_change | fully_inactive | 20 | 20 | 0 | 66 | 222.1 |
| regime_change | observe_and_ignore_actions | 20 | 20 | 0 | 66 | 222.1 |
| short_biased_random | fully_inactive | 20 | 20 | 0 | 6 | 44.6 |
| short_biased_random | observe_and_ignore_actions | 20 | 20 | 0 | 6 | 44.6 |
| startup_zero_history | fully_inactive | 20 | 20 | 0 | 148 | 235.9 |
| startup_zero_history | observe_and_ignore_actions | 20 | 20 | 0 | 171 | 235.6 |

Controlled paired paths equal: 200; controlled pairs differing unexpectedly: 0.
State/RNG-evolution pairs differing as intended: 37.

## Interpretation

- The consumed-validation leader is `burnin1_markov` with actual mean P&L
  218298, mean P&L/day 599.7,
  lower quartile -47000, worst run
  -265000, and maximum drawdown 265000.
  This is a validation leader, not a production recommendation.
- Flat is the explicit no-trade fallback. It was tied for best in
  25.8% of validation scenarios; its own P&L is zero,
  and the candidate outperformance fractions show where trading was actually
  preferable.
- Burn-in, Markov, ensemble, and asymmetric-prior results must be read with
  lower-tail loss, pivotal P&L, budget rejection, and family dependence. The
  immediate-long prior is not assumed optimal merely because the upward move
  is $8,000 versus the $5,000 downward move.
- Online Markov's actual validation mean/day is 578.9;
  the ensemble is 510.1. Drift fallback is
  465.6 with mean drawdown 30994.
  These are validation comparisons, not held-out evidence.
- Periodic replay and Year-1 selection are excluded. The old Pass 2 “Markov is
  best” conclusion remains superseded and no replay value is claimed here.
- Stress cases such as floor clipping, runaway budget rejection, ties/zeros,
  and no-trade-friendly populations are kept in development diagnostics rather
  than allowed to dominate stochastic validation ranking.

## Locked final protocol

The final definition uses a distinct seed range beginning at 90,000 and unseen
families including symmetric biases, reactive mixtures, periodic behavior,
regime/drift, startup sensitivity, and pivotal margins. Its concise manifest is
stored separately. The normal runner does not instantiate it.

Only this explicit command may execute the final suite:

```text
python -m research.liferaft.pass3_experiments --final
```

When executed, it writes `PASS3_FINAL_REPORT.md` and states that final results
were executed; it does not overwrite this validation report or claim that the
final suite remains unconsumed.
