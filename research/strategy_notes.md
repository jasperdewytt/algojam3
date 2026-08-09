# Strategy Notes

## Portfolio rule

- Primary objective: maximise unseen Round 2 P&L within the AUD 600,000 gross-position budget.
- When budget is available, take any robust positive-expectation trade at its full limit.
- When budget binds, rank opportunities by expected P&L per dollar of capital and reduce the weakest first.
- Prefer stable, explainable signals over small in-sample improvements.

## Instruments

| Instrument | Current strategy/view | Status |
|---|---|---|
| UQ Dollar | Hold the full opposite position for every non-zero deviation from the $100 peg. | Confirmed |
| Boat Party Ticket | Trade broad semester calendar rises and declines; stay flat over summer and near uncertain turns. | Next candidate |
| Sausage Sizzle | Predict the next move from current changes in Bread, Sausage, and MenuDash using a causal rolling model. | Promising |
| Thrifted Jeans | Positive long-run trend; begin with a simple long bias. | Candidate |
| Sausage | Majority vote of causal EMA trend signals with `alpha = 0.10, 0.15, 0.20`; full position in the voted direction. | Promoted |
| Bread | Same three-EMA majority trend rule as Sausage. | Promoted |
| MenuDash | New causal Sizzle-implied labour-sensor ensemble is the leading candidate; keep out of production until implementation review. | Strong candidate |
| Fintech Token | Calm-regime one-day reversal; volatile-regime one-day momentum, detected with causal EWMA volatility. | Strong candidate; validate challenger |
| Liferaft Ticket | Live minority game with no valid local backtest; solve separately. | Deferred |

## Food/labour factor

- Sizzle is a **lagged cost composite**, not a conventional same-day PCA factor: current Bread, Sausage, and labour costs predict tomorrow's Sizzle.
- Current Bread and Sausage changes correlate `0.709` and `0.607` with next-day Sizzle changes; the causal change model explains about `85.6%` of their variance.
- After removing lagged Bread and Sausage costs, the remaining Sizzle level residual correlates `0.927` with MenuDash, strongly supporting a shared latent labour cost.
- Ordinary PCA is poorly matched because the inputs need not move together and the relationship is lagged. Small ridge regularisation may stabilise estimates but has not materially improved P&L.
- A causal latent-labour Kalman filter was tested with labour measured in Sizzle-dollar units, a fixed Sizzle loading of one, and training-only likelihood tuning of state and observation variances. Across evaluation starts 60, 90, 120, 180 and 240, fixed Kalman averaged RMSE ratio `0.693` and P&L `$36,180`, versus `0.345` and `$45,864` for the 20-day rolling OLS. Seeded-to-online and 50/50 blended variants did not close the gap. Reject Kalman for production; keep the simpler rolling change model.
- Round 2 startup recommendation: seed the causal change coefficients from Round 1 (intercept `0.005`, Bread `0.077`, Sausage `1.692`, MenuDash `0.360`), stay flat on day 0 when component changes are unavailable, use the seeded sign rule from day 1, then transition to a causal 20-day rolling estimate after 20 completed target outcomes. Keep the zero forecast threshold unless capital is scarce.

## UQ Dollar — locked baseline

```python
if price > 100:
    position = -limit
elif price < 100:
    position = limit
else:
    position = 0
```

- Round 1 P&L: **$64,551.50**; profitable in every quarter.
- Maximum capital: approximately **$66,000**.
- More complex z-score, regression, calendar, and cross-asset models did not improve walk-forward P&L.
- For portfolio allocation, use `abs(price - 100) / price` as the approximate expected edge per capital dollar. Keep large deviations before small ones if capital becomes constrained.

## Bread, Sausage and MenuDash robustness

- Frozen Bread/Sausage EMA-vote parameters: `0.10`, `0.15`, `0.20`. Round 1 P&L was `$22,430` and `$10,300`; every quarter was positive.
- Their focused-family permutation results were `p = 0.025` and `p = 0.008`; moving-block bootstrap profitability ranged from `84.9%` to `99.2%` depending on instrument and block length.
- MenuDash's frozen 5/7/10-day median ensemble made `$50,250` with four positive quarters, but failed the stricter family-wise shuffled test (`p = 0.172`). Do not promote it without unseen evidence.
- A second-pass causal labour-sensor model removes expanding Bread/Sausage loadings from lagged Sizzle, maps the residual to MenuDash using expanding/60/120-day regressions, and majority-votes the three fair-value signals. It made `$120,750`, with four positive quarters and independently reset 91-day block results of `$4,500`, `$36,000`, `$28,500`, and `$28,500` after fresh 30-day warm-ups.
- Breaking MenuDash's alignment with the labour sensor gave p-values `0.0001` for daily shuffles, `0.0002` for 7/14/21-day block permutations, and `0.0082` across every non-zero circular shift. This is substantially stronger than the rejected median-only rule, but the large research trial count still warrants a deliberate production decision.

## Fintech Token — regime research

- The useful distinction is behavioural, not simply low versus high price: low/middle-volatility moves tend to reverse the following day, while the highest-volatility observations show mild continuation. Plain one-day reversal made **$75,369**, but lost **$16,736** in Q2 and had a **-$34,766** maximum drawdown; fixed 5/10/20/40-day momentum rules all lost money.
- Leading benchmark: estimate variance causally with `v[t] = lambda*v[t-1] + (1-lambda)*change[t]^2`; classify volatility against an expanding historical percentile; reverse the latest change when calm and follow it when volatile. Use a 30-day warm-up and the full `100`-token limit.
- The central setting `(lambda=0.90, percentile=0.80)` made **$157,409**, with quarterly P&L of `$30,774/$65,516/$45,138/$15,981` and maximum drawdown **-$8,996**. Nearby settings were broadly profitable; the retrospective grid-best `(0.97, 0.90)` made `$163,069` but should not be selected merely because it won Round 1.
- A conservative candidate is the majority vote of `(0.85, 0.75)`, `(0.90, 0.80)`, and `(0.95, 0.85)`. It currently produces the same `$157,409` path. Independently resetting it every 91 days remained profitable in all four blocks. A 200-shuffle test correcting for the 25-setting search had no null result as large (`p ~= 0.005`), but this establishes Round 1 serial structure rather than Round 2 transferability; Q4 was noticeably weaker.
- Best challenger: a **two-state Markov-switching AR(1)** on price changes, with state-specific intercept, autoregressive coefficient, and variance. It can learn reversal versus continuation and persistent regime probabilities jointly. Trading decisions must use filtered probabilities through the current day, never smoothed/Viterbi states using the full series. With only 364 changes and roughly eight parameters, it must beat the EWMA rule across causal restarts to justify production.
- Secondary avenues: Bayesian online changepoint detection to reset stale state after jumps; PELT only for retrospective regime diagnostics; SWARCH if conditional volatility remains unexplained; a hidden semi-Markov model only if empirical regime durations are clearly non-geometric. These methods mainly improve state detection—they do not inherently predict jump direction.
- Further validation before promotion: freeze the EWMA ensemble; compare it with the two-state model using expanding fits and 60/90-day fresh starts; examine state-specific lag correlations and duration distributions; run moving-block bootstraps and family-wise nulls; perturb warm-up, percentile, and decay ranges; measure portfolio P&L per capital dollar alongside the other instruments.
- Useful primary references: [Hamilton's Markov-switching filter](https://citeseerx.ist.psu.edu/document?doi=de6046f58a05a769b5aa526d95a09c5fa5e5b42c&repid=rep1&type=pdf), [Rydén et al. on financial HMMs](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291099-1255%28199805/06%2913%3A3%3C217%3A%3AAID-JAE476%3E3.0.CO%3B2-V), [Bayesian online changepoints](https://arxiv.org/abs/0710.3742), [PELT](https://arxiv.org/abs/1101.1438), and [financial hidden semi-Markov models](https://mpra.ub.uni-muenchen.de/7675/1/MPRA_paper_7675.pdf).

## Thrifted Jeans audit conclusion

- The causal 57-candidate audit recommends the middle local-linear Kalman setting with an uncertainty-gated short: `q_level=1.00`, `q_slope=0.05`, observation variance `9.0`; flat on day 0, then `+800` unless estimated slope is below minus one estimated standard error, in which case `-800`. Round 1 P&L was **$100,016** (**+$61,880** versus always-long), with all seven overlapping pseudo-starts profitable and moving-block bootstrap positivity of `0.995/0.990/0.988` for blocks `5/10/20` days.
- This is an **uncertain improvement**, not a production-locked edge: the family-wise permuted-path and random-signal p-values after searching all 57 candidates were `0.096` and `0.068`. The CUSUM `k=0.25, h=2.5` long/short rule is the runner-up at **$100,312**, but nearby thresholds fall to `$54,696` and `-$10,472`; keep it as a diagnostic alternative. Do not put either Jeans rule into `algorithm.py` yet.
- The current portfolio peaks at **$426,919** gross; adding full-limit Jeans peaks at **$501,212** (Jeans alone **$75,824**), with zero Round 1 budget breaches.
- Targeted follow-up: the corrected 176-candidate price-path bootstrap and 2,000-path family-wise null weakened the apparent edge (combined p = **0.258**, random-signal p = **0.233**); positive neutral-state drift remains descriptive but uncertain. Use **always-long +800** if Jeans is allocated; retain CUSUM/K2 conservative-short **(+800/+200/-800)** only as an uncertain paper candidate. Leave algorithm.py unchanged.
- Focused causal parameter-policy audit: the new 89-policy search found no reason to replace the structural +800 Jeans prior. Fixed K1/K2/consensus remained plausible Round 1 candidates, but causal selection mostly stayed at K2, online MLE repeatedly hit variance bounds and underperformed, and scale normalization did not improve subsequent evidence. The new path reruns were runtime-bounded (100 fixed/selector/scale paths and 10 MLE paths per block); combined with the earlier 2,000-path corrected audit, the verdict remains always-long, with no production implementation. The final unchanged-algorithm replay peaked at AUD **$491,711.50** before Jeans and **$555,931.00** with full-limit Jeans, with zero gross-budget breaches.
- Third-stage one-sided EMA audit: the legacy 73.7% negative-deviation hit rate reproduces (28/38), but the properly centred causal rolling-residual buy-the-dip overlay makes AUD **99,528**, below Candidate C's **119,584**. Keep Candidate C's causal `q_level=0.5`, `q_slope=0.05`, `R=5`, `slope_z <= -0.6` short rule; do not add the EMA branch.
