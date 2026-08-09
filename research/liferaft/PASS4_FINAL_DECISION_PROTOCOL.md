# Liferaft Pass 4 final decision protocol

Status: **frozen before final-suite construction or execution**.

This protocol is a one-time, conjunctive risk-control gate for the selected
`risk50_burnin1_markov` candidate. It is not a parameter search, optimizer, or
claim about positive expected value in the real competition. The locked final
suite may be consumed once only. No post-final parameter or result-affecting
research-code changes are permitted.

## Evaluation scope

- Candidate under decision: `risk50_burnin1_markov`.
- Comparison exposure: AUD 0 other gross exposure.
- Final scenarios: the existing locked 160 paired cases, unchanged.
- Final catalogue: the existing 14 candidates plus the selected wrapper,
  for 15 candidates and 2,400 candidate cells.
- Flat is the automatic production fallback if any criterion fails.

All metrics are calculated from actual final run-level records. Family
conditions use the arithmetic mean over runs in each final family. The
wrapper's pivotal P&L is computed from the simulator's `pnl_source_day`, not
from the current decision row. Loss/health stop and overshoot diagnostics use
only observations causally available to the wrapper.

## Automatic PASS criteria

Every row below is required. There are no exceptions or post-result
reinterpretations.

### Safety conditions

| criterion | required result |
|---|---:|
| Lower-quartile marked P&L | at least AUD -25,000 |
| Worst marked P&L | at least AUD -60,000 |
| Maximum drawdown | at most AUD 75,000 |
| Focal budget breaches | exactly 0 |
| Focal rejected actions | exactly 0 |
| Mean pivotal marked P&L | at least AUD -25,000 |
| Loss-stop activation rate | at most 10% |
| Health-stop activation rate | at most 75% |
| Minimum final-family mean wrapper P&L | at least AUD -30,000 |

### Value conditions

| criterion | required result |
|---|---:|
| Overall mean marked P&L | strictly positive |
| Median marked P&L | non-negative |
| Runs beating flat | at least 55% |
| Final families with positive mean wrapper P&L | at least half of all final families |
| Wrapper mean relative to raw `burnin1_markov` | at least 50% when raw mean is positive |

The raw-mean retention condition is vacuously satisfied when raw
`burnin1_markov` has non-positive final mean P&L; this does not relax any other
criterion.

## Decision rule

The automatic decision is **PASS** if and only if every safety and value row
passes. Otherwise it is **FAIL**, and production Liferaft must remain flat.
Passing this synthetic final suite is not proof of positive real competition
expected value. It only determines whether this frozen risk wrapper clears the
predeclared research-to-production gate.
