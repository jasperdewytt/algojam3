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
| Fintech Token | Regime-switching between quiet plateaus and jumps; needs robust regime detection. | Later research |
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

## Thrifted Jeans audit conclusion

- The causal 57-candidate audit recommends the middle local-linear Kalman setting with an uncertainty-gated short: `q_level=1.00`, `q_slope=0.05`, observation variance `9.0`; flat on day 0, then `+800` unless estimated slope is below minus one estimated standard error, in which case `-800`. Round 1 P&L was **$100,016** (**+$61,880** versus always-long), with all seven overlapping pseudo-starts profitable and moving-block bootstrap positivity of `0.995/0.990/0.988` for blocks `5/10/20` days.
- This is an **uncertain improvement**, not a production-locked edge: the family-wise permuted-path and random-signal p-values after searching all 57 candidates were `0.096` and `0.068`. The CUSUM `k=0.25, h=2.5` long/short rule is the runner-up at **$100,312**, but nearby thresholds fall to `$54,696` and `-$10,472`; keep it as a diagnostic alternative. Do not put either Jeans rule into `algorithm.py` yet.
- The current portfolio peaks at **$426,919** gross; adding full-limit Jeans peaks at **$501,212** (Jeans alone **$75,824**), with zero Round 1 budget breaches.
