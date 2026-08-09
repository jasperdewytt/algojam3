# Frozen reproducible SavGol candidate

## Rule

- Build the Round 1 calendar template with a 21-day, order-2 local polynomial
  Savitzky--Golay equivalent.
- Calculate its one-day forward slope.
- Estimate the standard deviation of the template slopes.
- Use `+1` when slope score is at least `0.5`, `-1` when it is at most `-0.5`,
  and `0` otherwise.
- In Round 2, the resulting calendar string is frozen. On neutral semester
  days, use the causal EWMA overlay with alpha `0.65`, a 7-day rolling scale,
  and threshold `0.01`.
- From day 322 use fixed AUD 45 mean reversion. Day 364 is flat.

The slope threshold is a round half-standard-deviation confidence rule, not
the maximum-P&L threshold from the diagnostic grid.

## Frozen semester string

```text
0000000+++++++++++++++000000000+++++++++0000000000000------------------00-0-0000
-------0-00000-0000++0++++++++++000000----------------00---00++++++000----------
00++0++++++0+++++++++++++++++++-++000--0------------------0000--000----00000++++
+++++++0++000--0------------------00-0000+++++000-----------00++0++++++++++++0+0
0000-0000000000000000000000000000000000000000
```

The concatenated string has 365 characters; only days 0--321 are used for the
semester regime, and the summer rule replaces later characters.

## Diagnostics

- Round 1 P&L: AUD 125,300.
- Increment over Candidate D: AUD 32,740.
- Maximum drawdown: AUD -2,600.
- Maximum Boat notional: AUD 55,390.
- Fixed semester P&L: AUD 80,640.
- Neutral semester EWMA P&L: AUD 32,110.
- Fixed summer P&L: AUD 12,550.

Across 100 causal overlay settings spanning five alphas, four volatility
windows and five thresholds, median P&L is AUD 118,735, P10 is AUD 102,923,
and worst is AUD 99,650. All 100 configurations exceed Candidate D's observed
AUD 92,560. These remain same-year diagnostics.

On 800 seasonality-preserving generator-conditioned paths, median/P10/worst
P&L is AUD 91,829 / 63,601 / 42,643, with an 84.0% paired win rate against
Candidate D. The paired P10 difference is negative, so the candidate is not
uniformly superior on every generated path.

An additional adversarial residual-regime test varies residual AR(1)
autocorrelation from -0.6 to +0.9. The hybrid remains better in median through
phi `+0.6`; at unusually persistent momentum regimes (`+0.8` and `+0.9`) it
underperforms Candidate D by median AUD 4,732 and AUD 8,519 respectively, while
remaining profitable in median at AUD 59,710 and AUD 51,187. This is the main
identified Round 2 failure mode.

Implementation and full tables are in `savgol_regime_audit.py` and
`results/savgol_*.csv`. No production file was modified.
