# Liferaft Pass 4 final locked-suite report

> The locked final suite was executed once under the frozen Pass 4 decision
> protocol. These outcomes are consumed and must not be used to retune the
> wrapper.

## Execution receipt

| item | value |
|---|---:|
| timestamp | `2026-08-09T03:28:42.816191+00:00` |
| command | `python -m research.liferaft.pass3_experiments --final` |
| locked combined hash | `443487A9E3332D72839936BCFEF66A9CEFE1C4F6D10F64CD8C1EED488254903E` |
| scenarios | 160 |
| candidates | 15 |
| candidate cells | 2400 |
| other exposure | AUD 0 |
| final families | 8 |
| automatic decision | **FAIL** |

The comparison uses fresh candidate and opponent instances for every scenario.
The wrapper is scored through the same simulator and P&L timing as the other
candidates. Because the focal vote can alter the market, these are endogenous
same-initial-scenario game paths, not fixed-path backtests.

## Overall final results

| strategy | runs | mean | median | lower quartile | worst run | mean/day | mean DD | max DD | active | hit | turnover | breaches | rejected | beat flat | flat tied | pivotal P&L | non-pivotal P&L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat` | 160 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 | 0.0% | 35.0% | 0 | 0 |
| `always_long` | 160 | 77644 | -70500 | -80000 | -80000 | 213.3 | 70262 | 251000 | 320.6 | 25.8% | 1.2 | 43.6 | 43.6 | 35.0% | 35.0% | -141681 | 219325 |
| `always_short` | 160 | -415825 | -503000 | -505000 | -508000 | -1142.4 | 432269 | 534000 | 180.9 | 18.4% | 1.8 | 183.9 | 183.9 | 11.9% | 35.0% | -196550 | -219275 |
| `burnin1_markov` | 160 | 99038 | -10000 | -69000 | -285000 | 272.1 | 53944 | 285000 | 116.8 | 33.1% | 79.5 | 49.4 | 49.4 | 46.9% | 35.0% | -66100 | 165138 |
| `burnin3_markov` | 160 | 97138 | -15000 | -69000 | -275000 | 266.9 | 53850 | 275000 | 115.9 | 32.9% | 78.3 | 49.7 | 49.7 | 45.6% | 35.0% | -66156 | 163294 |
| `burnin5_markov` | 160 | 95081 | -21000 | -72000 | -275000 | 261.2 | 54475 | 275000 | 115.7 | 32.6% | 78.1 | 48.6 | 48.6 | 45.0% | 35.0% | -64831 | 159912 |
| `burnin10_markov` | 160 | 89569 | -9000 | -67000 | -280000 | 246.1 | 52638 | 280000 | 113.4 | 32.8% | 76.3 | 50.8 | 50.8 | 44.4% | 35.0% | -64281 | 153850 |
| `online_last_counter` | 160 | -587750 | -662500 | -963000 | -2166000 | -1614.7 | 654338 | 2166000 | 299.9 | 25.1% | 302.4 | 63.3 | 63.3 | 19.4% | 35.0% | -311125 | -276625 |
| `online_rolling` | 160 | 4519 | -59000 | -150000 | -524000 | 12.4 | 100469 | 524000 | 107.4 | 28.5% | 48.0 | 49.6 | 49.6 | 34.4% | 35.0% | -80544 | 85062 |
| `online_markov` | 160 | 96075 | -7000 | -58000 | -275000 | 263.9 | 48844 | 275000 | 112.4 | 33.3% | 77.1 | 50.5 | 50.5 | 45.0% | 35.0% | -62806 | 158881 |
| `online_ensemble` | 160 | 81388 | -32500 | -81000 | -280000 | 223.6 | 64944 | 280000 | 120.9 | 32.1% | 79.6 | 51.1 | 51.1 | 40.6% | 35.0% | -75562 | 156950 |
| `online_drift` | 160 | 75731 | -5000 | -13000 | -159000 | 208.1 | 25150 | 200000 | 72.1 | 25.0% | 32.5 | 29.4 | 29.4 | 33.8% | 35.0% | -31975 | 107706 |
| `immediate_long_prior` | 160 | 77644 | -70500 | -80000 | -80000 | 213.3 | 70262 | 251000 | 320.6 | 25.8% | 1.2 | 43.6 | 43.6 | 35.0% | 35.0% | -141681 | 219325 |
| `flat_first_long_prior` | 160 | 75806 | -65000 | -75000 | -104000 | 208.3 | 68019 | 251000 | 317.5 | 25.8% | 1.2 | 43.7 | 43.7 | 35.0% | 35.0% | -140481 | 216288 |
| `risk50_burnin1_markov` | 160 | 91944 | -2000 | -17000 | -50000 | 252.6 | 15850 | 56000 | 30.8 | 38.9% | 36.6 | 0.0 | 0.0 | 45.6% | 35.0% | -16788 | 108731 |

## Family results

These are actual run-level family summaries; family rows are not treated as
independent samples and are not used to manufacture aggregate quantiles.

| family | strategy | runs | mean P&L | median | lower quartile | worst | mean DD | max DD | active | beat flat | pivotal P&L | non-pivotal P&L |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gradual_drift | `always_long` | 20 | 127200 | -10000 | -80000 | -80000 | 80500 | 114000 | 331.9 | 50.0% | -188000 | 315200 |
| gradual_drift | `always_short` | 20 | -450700 | -505500 | -508000 | -508000 | 479000 | 534000 | 166.0 | 10.0% | -242400 | -208300 |
| gradual_drift | `burnin10_markov` | 20 | 14300 | -37500 | -57000 | -86000 | 65200 | 94000 | 64.0 | 40.0% | -56300 | 70600 |
| gradual_drift | `burnin1_markov` | 20 | 24300 | -24500 | -53000 | -95000 | 58700 | 95000 | 68.3 | 40.0% | -56800 | 81100 |
| gradual_drift | `burnin3_markov` | 20 | 23400 | -24500 | -61000 | -95000 | 58800 | 95000 | 67.8 | 40.0% | -56000 | 79400 |
| gradual_drift | `burnin5_markov` | 20 | 25700 | -27000 | -61000 | -87000 | 58600 | 92000 | 70.3 | 40.0% | -56500 | 82200 |
| gradual_drift | `flat` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0 | 0 |
| gradual_drift | `flat_first_long_prior` | 20 | 127300 | -13000 | -83000 | -83000 | 79600 | 122000 | 328.8 | 50.0% | -185700 | 313000 |
| gradual_drift | `immediate_long_prior` | 20 | 127200 | -10000 | -80000 | -80000 | 80500 | 114000 | 331.9 | 50.0% | -188000 | 315200 |
| gradual_drift | `online_drift` | 20 | -12500 | -5500 | -18000 | -42000 | 17600 | 50000 | 6.2 | 0.0% | -6900 | -5600 |
| gradual_drift | `online_ensemble` | 20 | 14300 | -17500 | -81000 | -128000 | 67800 | 128000 | 73.8 | 40.0% | -65500 | 79800 |
| gradual_drift | `online_last_counter` | 20 | -578100 | -555500 | -807000 | -1071000 | 604300 | 1071000 | 290.0 | 0.0% | -359700 | -218400 |
| gradual_drift | `online_markov` | 20 | 23200 | -36000 | -54000 | -82000 | 54900 | 82000 | 64.2 | 40.0% | -57200 | 80400 |
| gradual_drift | `online_rolling` | 20 | -21400 | -55000 | -132000 | -178000 | 101500 | 186000 | 103.2 | 30.0% | -114100 | 92700 |
| gradual_drift | `risk50_burnin1_markov` | 20 | -16300 | -19000 | -23000 | -34000 | 20500 | 37000 | 7.7 | 20.0% | -8500 | -7800 |
| margin_mixture | `always_long` | 20 | 37500 | -80000 | -80000 | -80000 | 66000 | 80000 | 309.0 | 20.0% | -18000 | 55500 |
| margin_mixture | `always_short` | 20 | -96700 | 76500 | -504000 | -504000 | 165300 | 504000 | 274.4 | 70.0% | -134400 | 37700 |
| margin_mixture | `burnin10_markov` | 20 | 59100 | 14500 | -9000 | -280000 | 54200 | 280000 | 273.5 | 60.0% | -114800 | 173900 |
| margin_mixture | `burnin1_markov` | 20 | 82400 | 31000 | -10000 | -285000 | 54500 | 285000 | 276.4 | 70.0% | -114700 | 197100 |
| margin_mixture | `burnin3_markov` | 20 | 80200 | 31000 | -10000 | -275000 | 53500 | 275000 | 276.0 | 70.0% | -113700 | 193900 |
| margin_mixture | `burnin5_markov` | 20 | 73500 | 26000 | -15000 | -275000 | 53500 | 275000 | 274.8 | 70.0% | -113700 | 187200 |
| margin_mixture | `flat` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0 | 0 |
| margin_mixture | `flat_first_long_prior` | 20 | 42900 | -65000 | -65000 | -83000 | 55800 | 83000 | 305.9 | 20.0% | -18000 | 60900 |
| margin_mixture | `immediate_long_prior` | 20 | 37500 | -80000 | -80000 | -80000 | 66000 | 80000 | 309.0 | 20.0% | -18000 | 55500 |
| margin_mixture | `online_drift` | 20 | 20000 | -29000 | -61000 | -159000 | 76900 | 200000 | 231.4 | 40.0% | -98500 | 118500 |
| margin_mixture | `online_ensemble` | 20 | 30300 | -33500 | -92000 | -280000 | 102400 | 280000 | 264.0 | 40.0% | -115000 | 145300 |
| margin_mixture | `online_last_counter` | 20 | -124500 | -114000 | -275000 | -780000 | 237900 | 780000 | 307.2 | 30.0% | -138900 | 14400 |
| margin_mixture | `online_markov` | 20 | 71800 | 20500 | -6000 | -275000 | 57300 | 275000 | 273.8 | 70.0% | -113500 | 185300 |
| margin_mixture | `online_rolling` | 20 | 82100 | 27500 | -5000 | -285000 | 53000 | 285000 | 153.4 | 70.0% | -57100 | 139200 |
| margin_mixture | `risk50_burnin1_markov` | 20 | 116500 | 62500 | 39000 | -50000 | 15100 | 50000 | 37.5 | 90.0% | -27200 | 143700 |
| periodic | `always_long` | 20 | 52100 | -80000 | -80000 | -80000 | 59800 | 88000 | 345.2 | 30.0% | -116800 | 168900 |
| periodic | `always_short` | 20 | -469300 | -503500 | -504000 | -507000 | 471700 | 512000 | 211.3 | 0.0% | -275200 | -194100 |
| periodic | `burnin10_markov` | 20 | 338500 | 182500 | -53000 | -98000 | 32100 | 98000 | 146.6 | 70.0% | -75200 | 413700 |
| periodic | `burnin1_markov` | 20 | 338700 | 185000 | -58000 | -116000 | 33500 | 116000 | 148.2 | 70.0% | -79400 | 418100 |
| periodic | `burnin3_markov` | 20 | 338100 | 187500 | -58000 | -116000 | 33000 | 116000 | 147.8 | 70.0% | -79400 | 417500 |
| periodic | `burnin5_markov` | 20 | 341500 | 189000 | -53000 | -111000 | 31900 | 111000 | 147.6 | 70.0% | -79200 | 420700 |
| periodic | `flat` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0 | 0 |
| periodic | `flat_first_long_prior` | 20 | 51800 | -65000 | -78000 | -83000 | 56500 | 86000 | 342.2 | 30.0% | -116100 | 167900 |
| periodic | `immediate_long_prior` | 20 | 52100 | -80000 | -80000 | -80000 | 59800 | 88000 | 345.2 | 30.0% | -116800 | 168900 |
| periodic | `online_drift` | 20 | 284000 | 10500 | -10000 | -88000 | 30600 | 88000 | 115.7 | 50.0% | -59200 | 343200 |
| periodic | `online_ensemble` | 20 | 325300 | 170000 | -25000 | -88000 | 35100 | 88000 | 151.7 | 70.0% | -94700 | 420000 |
| periodic | `online_last_counter` | 20 | -1391900 | -1512000 | -1803000 | -2166000 | 1394500 | 2166000 | 317.0 | 0.0% | -382000 | -1009900 |
| periodic | `online_markov` | 20 | 349100 | 192000 | -48000 | -103000 | 32300 | 103000 | 148.2 | 70.0% | -79500 | 428600 |
| periodic | `online_rolling` | 20 | -91700 | -113500 | -150000 | -524000 | 128700 | 524000 | 39.9 | 10.0% | -53900 | -37800 |
| periodic | `risk50_burnin1_markov` | 20 | 339400 | 204500 | -19000 | -27000 | 13700 | 27000 | 100.6 | 70.0% | -43400 | 382800 |
| reactive_mixture | `always_long` | 20 | 43850 | -80000 | -80000 | -80000 | 63150 | 88000 | 364.0 | 30.0% | -113250 | 157100 |
| reactive_mixture | `always_short` | 20 | -296150 | -501000 | -504000 | -505000 | 317300 | 512000 | 235.9 | 15.0% | -202400 | -93750 |
| reactive_mixture | `burnin10_markov` | 20 | 56350 | 2000 | -25000 | -115000 | 45700 | 135000 | 196.3 | 50.0% | -112150 | 168500 |
| reactive_mixture | `burnin1_markov` | 20 | 74300 | 11500 | 0 | -115000 | 45900 | 134000 | 199.3 | 60.0% | -108800 | 183100 |
| reactive_mixture | `burnin3_markov` | 20 | 70650 | 10500 | 0 | -115000 | 45250 | 134000 | 196.3 | 60.0% | -110250 | 180900 |
| reactive_mixture | `burnin5_markov` | 20 | 72650 | 2000 | -3000 | -115000 | 46600 | 120000 | 192.8 | 55.0% | -99150 | 171800 |
| reactive_mixture | `flat` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0 | 0 |
| reactive_mixture | `flat_first_long_prior` | 20 | 47400 | -65000 | -65000 | -104000 | 56400 | 104000 | 361.0 | 30.0% | -112500 | 159900 |
| reactive_mixture | `immediate_long_prior` | 20 | 43850 | -80000 | -80000 | -80000 | 63150 | 88000 | 364.0 | 30.0% | -113250 | 157100 |
| reactive_mixture | `online_drift` | 20 | 9800 | 0 | -8000 | -37000 | 17500 | 91000 | 135.7 | 45.0% | -42900 | 52700 |
| reactive_mixture | `online_ensemble` | 20 | 55800 | 0 | -73000 | -131000 | 52250 | 154000 | 206.3 | 45.0% | -121950 | 177750 |
| reactive_mixture | `online_last_counter` | 20 | -421400 | -217000 | -774000 | -1710000 | 483500 | 1710000 | 332.5 | 25.0% | -282700 | -138700 |
| reactive_mixture | `online_markov` | 20 | 67800 | 29000 | 0 | -112000 | 39300 | 112000 | 191.9 | 65.0% | -96500 | 164300 |
| reactive_mixture | `online_rolling` | 20 | -31750 | 0 | -160000 | -232000 | 83600 | 232000 | 180.2 | 35.0% | -93600 | 61850 |
| reactive_mixture | `risk50_burnin1_markov` | 20 | 8350 | 8000 | -12000 | -25000 | 14300 | 36000 | 13.8 | 55.0% | -9200 | 17550 |
| regime_change | `always_long` | 20 | -33400 | -72000 | -80000 | -80000 | 105700 | 251000 | 364.0 | 30.0% | -186000 | 152600 |
| regime_change | `always_short` | 20 | -501600 | -502500 | -505000 | -508000 | 510000 | 521000 | 204.9 | 0.0% | -226400 | -275200 |
| regime_change | `burnin10_markov` | 20 | -45800 | -71000 | -93000 | -99000 | 70900 | 110000 | 58.4 | 20.0% | -41000 | -4800 |
| regime_change | `burnin1_markov` | 20 | -50300 | -66500 | -91000 | -96000 | 77900 | 107000 | 58.5 | 20.0% | -41300 | -9000 |
| regime_change | `burnin3_markov` | 20 | -53700 | -66500 | -91000 | -96000 | 78300 | 107000 | 57.7 | 10.0% | -42100 | -11600 |
| regime_change | `burnin5_markov` | 20 | -53000 | -68500 | -91000 | -93000 | 78600 | 110000 | 58.3 | 10.0% | -43100 | -9900 |
| regime_change | `flat` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0 | 0 |
| regime_change | `flat_first_long_prior` | 20 | -30800 | -67500 | -78000 | -96000 | 103200 | 251000 | 361.0 | 30.0% | -184800 | 154000 |
| regime_change | `immediate_long_prior` | 20 | -33400 | -72000 | -80000 | -80000 | 105700 | 251000 | 364.0 | 30.0% | -186000 | 152600 |
| regime_change | `online_drift` | 20 | -13800 | -6000 | -18000 | -58000 | 19300 | 58000 | 8.0 | 10.0% | -9700 | -4100 |
| regime_change | `online_ensemble` | 20 | -64500 | -77500 | -88000 | -176000 | 88900 | 176000 | 65.5 | 20.0% | -56700 | -7800 |
| regime_change | `online_last_counter` | 20 | -673900 | -701000 | -772000 | -885000 | 691700 | 917000 | 337.2 | 0.0% | -329900 | -344000 |
| regime_change | `online_markov` | 20 | -45600 | -47000 | -63000 | -91000 | 68200 | 104000 | 57.0 | 0.0% | -39900 | -5700 |
| regime_change | `online_rolling` | 20 | -76300 | -91000 | -159000 | -175000 | 123700 | 215000 | 106.7 | 20.0% | -86300 | 10000 |
| regime_change | `risk50_burnin1_markov` | 20 | -9300 | -7000 | -13000 | -34000 | 16700 | 34000 | 7.7 | 10.0% | -7300 | -2000 |
| short_biased_random | `always_long` | 20 | 504600 | 504500 | 502000 | 502000 | 12700 | 20000 | 122.7 | 100.0% | -59500 | 564100 |
| short_biased_random | `always_short` | 20 | -503700 | -504000 | -505000 | -507000 | 503700 | 507000 | 69.1 | 0.0% | -53600 | -450100 |
| short_biased_random | `burnin10_markov` | 20 | 415200 | 420500 | 398000 | 363000 | 13800 | 25000 | 100.5 | 100.0% | -45000 | 460200 |
| short_biased_random | `burnin1_markov` | 20 | 455200 | 455500 | 441000 | 388000 | 13000 | 25000 | 109.0 | 100.0% | -50500 | 505700 |
| short_biased_random | `burnin3_markov` | 20 | 449800 | 457500 | 431000 | 388000 | 13000 | 25000 | 107.8 | 100.0% | -50000 | 499800 |
| short_biased_random | `burnin5_markov` | 20 | 437000 | 446000 | 425000 | 346000 | 13500 | 25000 | 107.1 | 100.0% | -50000 | 487000 |
| short_biased_random | `flat` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0 | 0 |
| short_biased_random | `flat_first_long_prior` | 20 | 484400 | 480500 | 478000 | 478000 | 12700 | 20000 | 118.9 | 100.0% | -57500 | 541900 |
| short_biased_random | `immediate_long_prior` | 20 | 504600 | 504500 | 502000 | 502000 | 12700 | 20000 | 122.7 | 100.0% | -59500 | 564100 |
| short_biased_random | `online_drift` | 20 | 337300 | 446500 | 177000 | 7000 | 12300 | 20000 | 70.3 | 100.0% | -28500 | 365800 |
| short_biased_random | `online_ensemble` | 20 | 427300 | 446500 | 401000 | 309000 | 13500 | 27000 | 103.0 | 100.0% | -48500 | 475800 |
| short_biased_random | `online_last_counter` | 20 | 250900 | 270000 | 159000 | 36000 | 42000 | 94000 | 105.4 | 100.0% | -55600 | 306500 |
| short_biased_random | `online_markov` | 20 | 410800 | 415500 | 394000 | 347000 | 11700 | 20000 | 96.8 | 100.0% | -41500 | 452300 |
| short_biased_random | `online_rolling` | 20 | 451400 | 455500 | 445000 | 386000 | 11200 | 20000 | 108.6 | 100.0% | -52000 | 503400 |
| short_biased_random | `risk50_burnin1_markov` | 20 | 325500 | 368000 | 332000 | 9000 | 10800 | 23000 | 69.0 | 100.0% | -27500 | 353000 |
| startup_zero_history | `always_long` | 20 | -55300 | -67000 | -80000 | -80000 | 89050 | 137000 | 364.0 | 10.0% | -226400 | 171100 |
| startup_zero_history | `always_short` | 20 | -503950 | -503000 | -506000 | -508000 | 505550 | 516000 | 142.6 | 0.0% | -228400 | -275550 |
| startup_zero_history | `burnin10_markov` | 20 | -59700 | -61000 | -82000 | -109000 | 70100 | 109000 | 37.6 | 5.0% | -36800 | -22900 |
| startup_zero_history | `burnin1_markov` | 20 | -64300 | -62500 | -90000 | -116000 | 73550 | 119000 | 39.1 | 5.0% | -38600 | -25700 |
| startup_zero_history | `burnin3_markov` | 20 | -61950 | -62000 | -88000 | -112000 | 72750 | 119000 | 38.5 | 5.0% | -38600 | -23350 |
| startup_zero_history | `burnin5_markov` | 20 | -67500 | -73000 | -90000 | -119000 | 77100 | 119000 | 37.9 | 5.0% | -38000 | -29500 |
| startup_zero_history | `flat` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0 | 0 |
| startup_zero_history | `flat_first_long_prior` | 20 | -57550 | -65500 | -83000 | -91000 | 92350 | 153000 | 361.0 | 10.0% | -224150 | 166600 |
| startup_zero_history | `immediate_long_prior` | 20 | -55300 | -67000 | -80000 | -80000 | 89050 | 137000 | 364.0 | 10.0% | -226400 | 171100 |
| startup_zero_history | `online_drift` | 20 | -8050 | -10000 | -13000 | -35000 | 12000 | 43000 | 4.2 | 15.0% | -3900 | -4150 |
| startup_zero_history | `online_ensemble` | 20 | -81000 | -85000 | -107000 | -135000 | 91200 | 143000 | 56.0 | 0.0% | -57550 | -23450 |
| startup_zero_history | `online_last_counter` | 20 | -865000 | -953000 | -1086000 | -1213000 | 872300 | 1223000 | 356.4 | 0.0% | -474100 | -390900 |
| startup_zero_history | `online_markov` | 20 | -50900 | -55000 | -68000 | -102000 | 62050 | 102000 | 34.8 | 5.0% | -37250 | -13650 |
| startup_zero_history | `online_rolling` | 20 | -144900 | -150000 | -175000 | -214000 | 155250 | 217000 | 84.5 | 0.0% | -95650 | -49250 |
| startup_zero_history | `risk50_burnin1_markov` | 20 | -14300 | -13500 | -25000 | -29000 | 18700 | 29000 | 5.7 | 10.0% | -7100 | -7200 |
| symmetric_random | `always_long` | 20 | -55400 | -70500 | -77000 | -80000 | 85200 | 102000 | 364.0 | 10.0% | -225500 | 170100 |
| symmetric_random | `always_short` | 20 | -504500 | -505500 | -507000 | -508000 | 505600 | 508000 | 142.7 | 0.0% | -209600 | -294900 |
| symmetric_random | `burnin10_markov` | 20 | -61400 | -64000 | -82000 | -112000 | 69100 | 112000 | 30.5 | 10.0% | -33000 | -28400 |
| symmetric_random | `burnin1_markov` | 20 | -68000 | -73500 | -80000 | -142000 | 74500 | 142000 | 35.9 | 10.0% | -38700 | -29300 |
| symmetric_random | `burnin3_markov` | 20 | -69400 | -71000 | -78000 | -168000 | 76200 | 168000 | 35.6 | 10.0% | -39200 | -30200 |
| symmetric_random | `burnin5_markov` | 20 | -69200 | -68500 | -95000 | -125000 | 76000 | 125000 | 36.8 | 10.0% | -39000 | -30200 |
| symmetric_random | `flat` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0% | 0 | 0 |
| symmetric_random | `flat_first_long_prior` | 20 | -59000 | -70500 | -83000 | -96000 | 87600 | 102000 | 361.0 | 10.0% | -225100 | 166100 |
| symmetric_random | `immediate_long_prior` | 20 | -55400 | -70500 | -77000 | -80000 | 85200 | 102000 | 364.0 | 10.0% | -225500 | 170100 |
| symmetric_random | `online_drift` | 20 | -10900 | -11000 | -13000 | -18000 | 15000 | 24000 | 5.5 | 10.0% | -6200 | -4700 |
| symmetric_random | `online_ensemble` | 20 | -56400 | -63000 | -75000 | -97000 | 68400 | 108000 | 47.0 | 10.0% | -44600 | -11800 |
| symmetric_random | `online_last_counter` | 20 | -898100 | -960000 | -1061000 | -1176000 | 908500 | 1176000 | 353.8 | 0.0% | -466100 | -432000 |
| symmetric_random | `online_markov` | 20 | -57600 | -63000 | -74000 | -116000 | 65000 | 116000 | 32.3 | 10.0% | -37100 | -20500 |
| symmetric_random | `online_rolling` | 20 | -131300 | -144500 | -163000 | -204000 | 146800 | 206000 | 82.5 | 10.0% | -91700 | -39600 |
| symmetric_random | `risk50_burnin1_markov` | 20 | -14300 | -12500 | -18000 | -48000 | 17000 | 56000 | 4.6 | 10.0% | -4100 | -10200 |

## Wrapper diagnostics

| total wrapper runs | loss stops | health stops | both stops | both loss first | both health first | same observation | first loss (all) | first health (all) | mean/median loss day | mean/median health day | max overshoot | mean unknown pauses | mean floor gates | mean headroom gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| 160 | 2 (1.2%) | 99 (61.9%) | 0 (0.0%) | 0 | 0 | 0 | 2 | 99 | 474.0/474.0 | 391.0/384.0 | 0 | 99.6 | 3.8 | 0.4 |

### Loss-limit overshoot distribution

The distribution is conditional on wrapper loss-stop cases and includes zero
overshoot. Positive overshoot can only arise from the final newly observable
adverse movement that crosses AUD 50,000.

| overshoot amount | loss-stop cases | percentage of loss-stop cases |
|---:|---:|---:|
| 0 | 2 | 100.0% |

## Paired comparisons

| comparison | paired cases | mean wrapper minus comparison | median | wrapper wins | ties | wrapper losses |
|---|---:|---:|---:|---:|---:|---:|
| `flat` overall | 160 | 91944 | -2000 | 73 | 4 | 83 |
| `flat` / gradual_drift | 20 | -16300 | -19000 | 4 | 0 | 16 |
| `flat` / margin_mixture | 20 | 116500 | 62500 | 18 | 0 | 2 |
| `flat` / periodic | 20 | 339400 | 204500 | 14 | 0 | 6 |
| `flat` / reactive_mixture | 20 | 8350 | 8000 | 11 | 2 | 7 |
| `flat` / regime_change | 20 | -9300 | -7000 | 2 | 2 | 16 |
| `flat` / short_biased_random | 20 | 325500 | 368000 | 20 | 0 | 0 |
| `flat` / startup_zero_history | 20 | -14300 | -13500 | 2 | 0 | 18 |
| `flat` / symmetric_random | 20 | -14300 | -12500 | 2 | 0 | 18 |

| comparison | paired cases | mean wrapper minus comparison | median | wrapper wins | ties | wrapper losses |
|---|---:|---:|---:|---:|---:|---:|
| `burnin1_markov` overall | 160 | -7094 | 22500 | 96 | 6 | 58 |
| `burnin1_markov` / gradual_drift | 20 | -40600 | 6500 | 12 | 0 | 8 |
| `burnin1_markov` / margin_mixture | 20 | 34100 | 27000 | 14 | 2 | 4 |
| `burnin1_markov` / periodic | 20 | 700 | 19000 | 10 | 2 | 8 |
| `burnin1_markov` / reactive_mixture | 20 | -65950 | -17500 | 7 | 2 | 11 |
| `burnin1_markov` / regime_change | 20 | 41000 | 63500 | 16 | 0 | 4 |
| `burnin1_markov` / short_biased_random | 20 | -129700 | -92000 | 0 | 0 | 20 |
| `burnin1_markov` / startup_zero_history | 20 | 50000 | 52000 | 19 | 0 | 1 |
| `burnin1_markov` / symmetric_random | 20 | 53700 | 60500 | 18 | 0 | 2 |

| comparison | paired cases | mean wrapper minus comparison | median | wrapper wins | ties | wrapper losses |
|---|---:|---:|---:|---:|---:|---:|
| `online_drift` overall | 160 | 16212 | 0 | 76 | 5 | 79 |
| `online_drift` / gradual_drift | 20 | -3800 | -6500 | 8 | 0 | 12 |
| `online_drift` / margin_mixture | 20 | 96500 | 84500 | 16 | 0 | 4 |
| `online_drift` / periodic | 20 | 55400 | 53000 | 12 | 0 | 8 |
| `online_drift` / reactive_mixture | 20 | -1450 | -3500 | 7 | 2 | 11 |
| `online_drift` / regime_change | 20 | 4500 | 500 | 10 | 0 | 10 |
| `online_drift` / short_biased_random | 20 | -11800 | -42500 | 8 | 0 | 12 |
| `online_drift` / startup_zero_history | 20 | -6250 | -6500 | 5 | 1 | 14 |
| `online_drift` / symmetric_random | 20 | -3400 | 500 | 10 | 2 | 8 |

## Decision protocol outcome

See `PASS4_FINAL_DECISION.md` for every conjunctive criterion and its observed
PASS/FAIL status. Passing this synthetic suite is not proof of positive real
competition expected value.
