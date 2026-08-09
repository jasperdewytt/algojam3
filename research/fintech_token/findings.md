# Fintech Token research handoff

## Decision

**Promote the EWMA ensemble.** The selected policy is the predeclared majority vote of three causal volatility switches:

```text
(lambda=0.85, percentile=0.75, warm-up=30)
(lambda=0.90, percentile=0.80, warm-up=30)
(lambda=0.95, percentile=0.85, warm-up=30)
```

Each member reverses the latest observed change in the calm state and follows it in the volatile state. The ensemble takes the majority direction and uses `+/-100` units. This is a research recommendation only; `trader_interface/algorithm.py` was intentionally not changed.

## Data, timing and method

The file contains 365 prices and 364 one-day changes. The backtest uses the simulator convention explicitly: `changes[i] = P[i+1] - P[i]`, while `position[i]` is selected using observations with indices below `i` and earns `position[i] * changes[i]`. Thus a decision after observing `P[t]` can use `d[t] = P[t] - P[t-1]`, but earns against `d[t+1]` only.

The EWMA implementation updates the volatility estimate after the latest observed change and compares it with an expanding percentile of earlier volatility estimates only. It uses simple reversal during the 30-observation warm-up. All positions are integral and bounded by 100.

The audit included a fixed candidate family of 88 policies: 75 declared EWMA configurations, four baselines, four EWMA ensembles and five secondary probes. Two adaptive MS-AR challengers were tested separately. This candidate count is recorded in `results/model_counts.json`.

## Phase 1: behaviour

Price levels are nonstationary-looking and highly dispersed (mean 606.55, standard deviation 156.29, range 409.08–854.34). Changes have mean -0.77, standard deviation 13.26, median -1.04, 5th/95th percentiles -21.66/19.59 and range -43.17 to 38.91. Simple returns have mean -0.00090 and standard deviation 0.02379.

The lag-one signed-change ACF is -0.132, consistent with one-day reversal. Absolute-change and squared-change lag-one ACFs are 0.089 and 0.050; squared-change ACF at lag five is 0.198. EWMA volatility is persistent: for lambda 0.90 its ACF is 0.954 at lag one, 0.793 at lag five, 0.616 at lag ten and 0.421 at lag twenty. The main exploitable structure is therefore a weak signed reversal combined with persistent variance, not a persistent price-level trend.

Raw absolute-change terciles all reverse on average (low/middle/high next-change signed follow-up -1.29/-2.50/-2.80). The strategy's momentum state is not a claim that the largest individual move predicts continuation. It is conditional on a persistent EWMA-volatility state: the causal lambda=0.90, 80th-percentile diagnostic gives:

| Causal state | Observations | Mean EWMA volatility | Mean next change | P&L at 100 units |
| --- | ---: | ---: | ---: | ---: |
| Calm / reversal | 290 | 12.11 | -4.01 | -$116,389 as a signed-follow-up diagnostic |
| Volatile / momentum | 73 | 16.99 | +5.62 | +$41,020 as a signed-follow-up diagnostic |

The state relationship is not uniform across time. Volatile-state momentum is strong in quarter 2 (54 observations, mean follow-up +7.99), but there is only one volatile observation in quarter 3 and none in quarter 4. Calm-state reversal is negative in all four quarters, although it weakens in quarter 4. This is the principal transfer uncertainty.

The raw high-move indicator has 50 runs with mean duration 1.46 and maximum 4; no high runs last at least five observations. Low/middle runs have mean duration 5.71 and maximum 24. These durations do not provide strong evidence that a hidden semi-Markov model is warranted. The sample is better described as persistent EWMA memory with sparse high-state observations than as a reliably estimated non-geometric high-volatility regime.

Price-level reversal is only $8,881, level momentum is -$8,881, five-day reversal is $55,979 and five-day momentum is -$55,979. A jump-age probe reaches $85,343, but it was not frozen because it is a secondary feature tested after inspecting the same sample and lacks comparable family-level validation. No extra feature was established as a stable causal improvement over the volatility switch.

Useful plots are in `figures/`: price/changes, EWMA states, the causal state conditional relationship, raw volatility buckets and cumulative P&L.

## Phase 2: EWMA benchmark and parameter freeze

All 75 EWMA grid configurations were profitable on Round 1. Fifteen were within 10% of the retrospective grid maximum, forming a broad plateau rather than a single isolated peak. The retrospective best was `(0.97, 0.90, 30)` at approximately $163,069, but it is not recommended because it was selected after seeing the full path. The central `(0.90, 0.80, 30)` switch and the three-member ensemble have the same Round 1 path and P&L of $157,409. Warm-ups 20, 30 and 45 often produce the same positions for the plateau configurations; 30 is retained because it was the declared benchmark and gives a simple startup rule.

For the frozen ensemble:

- P&L: **$157,409**.
- Hit rate: 63.7% over all change slots, or 63.9% on 363 active slots.
- Quarterly P&L: **$30,774, $65,516, $45,138, $15,981**; all four quarters positive.
- Maximum drawdown: **-$8,996**.
- Turnover: **41,100 units across 206 position changes**.
- Maximum capital: **$85,434**, well below the shared $600,000 budget.
- P&L divided by maximum capital: **1.842**.
- Budget violations: **0**; all positions integral and within the instrument limit.

Holding the previous regime for one day changes P&L only from $157,409 to $157,465 and increases turnover. That is not enough evidence to introduce another frozen rule.

## Phase 3: Markov-switching challenger

The challenger is a two-state Gaussian Markov-switching AR(1) for changes, with state-specific mean, AR coefficient and variance, plus two free transition probabilities: eight dynamic parameters. States are labelled by fitted variance. Fits use multiple deterministic starts, expanding causal prefixes, filtered probabilities only, a minimum training prefix of 60 changes and scheduled refits every 30 observations. The trading forecast is a one-step expected change; the state-detector variant keeps the predeclared reversal/momentum directions.

The filtered expected-change strategy makes $56,637 with maximum drawdown -$37,605. It is positive in the full sample, but loses $4,867 from restart 90 and $25,133 from restart 120. The state-detector variant makes $39,431 with maximum drawdown -$44,923 and is negative from restarts 60, 90 and 120. The two-state parameter estimates are unstable in the short prefixes: the 90-change fit does not converge within 80 iterations, and the fitted high-state AR coefficient and persistence change substantially across prefixes. A Gaussian fit was retained for comparison because residual kurtosis was close to zero; a Student-t emission did not have a compelling diagnostic justification.

The MS-AR model is therefore rejected despite being causal and passing the future-perturbation check. It uses too many estimated quantities for 364 changes, its restart performance is unstable and its apparent benefit does not survive comparison with the simpler EWMA plateau.

## Robustness and overfitting controls

- **Chronological restarts:** the EWMA ensemble is positive from starts 0, 60, 90, 120, 180 and 240, with P&Ls $157.4k, $111.5k, $75.3k, $34.9k, $59.7k and $21.3k. Late performance is weaker and the start-120 drawdown reaches the simple-reversal drawdown, so this is evidence of resilience, not a guarantee.
- **Independent resets:** the 91-day blocks are all positive for the EWMA ensemble: $30.8k, $18.9k, $45.6k and $14.5k. The shorter 60-day blocks are mixed: five positive blocks, one -$22.3k block and one near-zero -$0.5k block.
- **Moving-block bootstrap:** positive shares for the EWMA policy are 0.98, 0.99 and 1.00 for block lengths 5, 10 and 20 over 100 resamples. The MS-AR bootstrap used only 10 resamples per length because fitting is expensive; its positive shares are 0.90, 0.80 and 0.80, with negative 5th percentiles at block lengths 10 and 20. Those MS-AR figures are low-power and do not rescue it.
- **Time-shuffle null:** 200 permutations of the fixed 88-policy family give selected and family-wise `p=0.00498`. This supports serial dependence in Round 1 and shows that the fixed family is not merely extracting an IID ordering. It does **not** prove transfer to Round 2. The adaptive MS-AR shuffle has only five permutations and gives family-wise `p=0.167`; treat it as a weak negative diagnostic.
- **Circular-shift null:** the selected-policy p-value is 0.0149, but the 88-policy family-wise p-value is 0.169. This weaker result is reported separately and is why the recommendation relies on the broad EWMA plateau and chronological tests rather than one null test.
- **Block permutations:** for block lengths 7, 14 and 21, selected and family-wise p-values are 0.00990. These permutations preserve local blocks, so they test dependence on block ordering rather than IID serial independence.
- **Large jumps:** excluding the largest 1, 3, 5 and 10 absolute changes leaves EWMA P&Ls of $153.1k, $153.4k, $153.5k and $138.4k. The result is not driven by one or a few jumps.
- **Future-data perturbation:** changing all observations after cuts 90, 180 and 300 leaves every EWMA and MS-AR position through the cut exactly unchanged. This directly audits the absence of lookahead in adaptive decisions.
- **Candidate accounting:** 88 fixed policies and two adaptive challengers were evaluated; the exact retrospective grid winner is explicitly separated from the frozen recommendation.

## Model-selection table

| Model | Causal? | Parameter count / complexity | Full-sample P&L | Restart / walk-forward results | Maximum drawdown | Robustness-test outcome | Overfitting risk | Recommendation |
| --- | --- | --- | ---: | --- | ---: | --- | --- | --- |
| EWMA diagonal ensemble | Yes; prior-volatility cutoffs only | 3 fixed EWMA switches; majority vote | $157,409 | Starts 0/60/90/120/180/240 all positive; 91-day blocks 4/4 positive; 60-day blocks 5/7 positive | -$8,996 | MBB positive share .98/.99/1.00; time-shuffle family p=.00498; block-permutation p=.00990; jump exclusions and lookahead audit pass | Moderate: one Round 1 path and 88 fixed candidates | **Promote** |
| Simple reversal | Yes | 1 sign rule | $75,369 | All tested restarts positive, but quarter 2 and one 91-day block are negative | -$34,766 | Strong baseline and bootstrap support; weaker return and drawdown | Low | Baseline / fallback |
| Retrospective EWMA grid best `(0.97, .90, 30)` | Yes | 1 switch, but selected after full-sample search | $163,069 | Not independently frozen; same broad family | -$9,599 | Profitable, but selection is contaminated by retrospective ranking | High | Do not freeze |
| Filtered two-state MS-AR(1) | Yes; filtered probabilities and scheduled causal refits | 8 dynamic parameters + EM/refit schedule | $56,637 | Negative from starts 90 and 120; 91-day block 2 negative | -$37,605 | Future audit passes; adaptive shuffle p=.167; low-rep bootstrap .80–.90; prefix-90 non-convergence | High for 364 changes | Reject |
| MS-AR state detector | Yes | 8 dynamic parameters + state rule | $39,431 | Negative from starts 60, 90 and 120 | -$44,923 | Worse restart profile and lower P&L | High | Reject |
| Always long / flat / simple momentum | Yes | Zero or 1 fixed rule | -$28,085 / $0 / -$75,369 | Consistent benchmarks | -$44,526 / $0 / -$82,718 | Useful nulls; no evidence for unconditional direction | Low | Reject |

## Round 2 operating specification

At startup, return flat on the first decision because no prior change is known. During the next 30 observed changes, use simple reversal at `+/-100` units. From decision index 31 onward, run the three frozen EWMA members and take their majority direction. If history is missing, non-finite or the latest observed change is exactly zero, return zero rather than inventing a direction. Keep all positions integral and in `[-100, 100]`; the single-instrument maximum capital observed in Round 1 was $85,434.

Copy-ready sketch, after this conclusion:

```python
# changes[i] is P[i+1] - P[i]; position[i] earns changes[i].
configs = [(0.85, 0.75), (0.90, 0.80), (0.95, 0.85)]
limit = 100
positions = np.zeros(len(changes), dtype=int)
variance_histories = [
    [float(changes[0] ** 2)] for _ in configs
] if len(changes) else [[] for _ in configs]

for t in range(1, len(changes)):
    latest = float(changes[t - 1])       # observed before deciding position[t]
    if not np.isfinite(latest):
        positions[t] = 0
        continue
    if t >= 2:                            # changes[0] is already initialized
        for (lam, _), history in zip(configs, variance_histories):
            history.append(lam * history[-1] + (1 - lam) * latest ** 2)
    if latest == 0:
        positions[t] = 0
        continue
    if t <= 30:
        positions[t] = -limit * int(np.sign(latest))
        continue

    votes = []
    for (_, percentile), history in zip(configs, variance_histories):
        v = history[-1] ** 0.5
        cutoff = np.quantile(np.sqrt(history[:-1]), percentile)
        regime = 1 if v >= cutoff else -1
        votes.append(regime * int(np.sign(latest)))
    positions[t] = limit * int(np.sign(sum(votes)))
```

The sketch is intentionally concise; use the tested implementations in `fintech_models.py` for the exact per-member EWMA histories and missing-data handling.

## Rejected or deferred avenues

Bayesian online changepoint detection, PELT, HSMM and SWARCH were not promoted. The observed high-move durations do not establish a stable non-geometric duration process; PELT is an offline diagnostic and cannot provide full-series labels to a causal backtest; changepoint detection does not predict jump direction; and the EWMA already captures the dominant volatility persistence. A jump-age probe was retained as a research lead, not as a frozen policy. Neural networks, tree models and broad feature searches were intentionally avoided because the sample has only 364 changes.

## Remaining uncertainties and follow-up

The conclusion is based on one 364-change Round 1 path. Volatile EWMA states are sparse and concentrated in quarter 2; late independent blocks are not uniformly positive. The 88-policy family is deliberately small but still creates selection risk, transaction costs/slippage are not represented in the supplied simulator, and Round 2 may alter regime persistence or the reversal/continuation relationship. The next useful evidence is live Round 2 prefix performance, especially the frequency and duration of volatile EWMA states.

The Hamilton Markov-switching paper, the Rydén–Teräsvirta–Åsbrink financial HMM paper and Hamilton–Susmel SWARCH paper are materially relevant but their publisher/full-text versions may be paywalled or inaccessible from the current environment. Accessible starting points are the [Hamilton paper mirror](https://citeseerx.ist.psu.edu/document?doi=de6046f58a05a769b5aa526d95a09c5fa5e5b42c&repid=rep1&type=pdf), [Rydén et al. abstract](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291099-1255%28199805/06%2913%3A3%3C217%3A%3AAID-JAE476%3E3.0.CO%3B2-V), [Adams–MacKay changepoint preprint](https://arxiv.org/abs/0710.3742), [Killick–Fearnhead–Eckley PELT preprint](https://arxiv.org/abs/1101.1438), [Killick et al. JASA article](https://doi.org/10.1080/01621459.2012.737745), and the [Hamilton–Susmel SWARCH DOI](https://doi.org/10.1016/0304-4076(94)90067-1). The current decision is not blocked by obtaining them; full Hamilton/Rydén/SWARCH texts would be useful only for a later, better-shrunk switching model.
