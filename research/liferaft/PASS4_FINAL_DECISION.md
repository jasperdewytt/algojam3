# Liferaft Pass 4 final decision

Decision: **FAIL**

This result applies the frozen `PASS4_FINAL_DECISION_PROTOCOL.md`
mechanically to the zero-other-exposure locked final suite.

- Execution timestamp: `2026-08-09T03:28:42.816191+00:00`
- Locked combined hash: `443487A9E3332D72839936BCFEF66A9CEFE1C4F6D10F64CD8C1EED488254903E`
- Candidates: 15
- Scenarios: 160

| criterion | required threshold | observed result | decision |
|---|---|---:|---|
| `lower_quartile_pnl` | >= -25000 | -17000 | PASS |
| `worst_pnl` | >= -60000 | -50000 | PASS |
| `maximum_drawdown` | <= 75000 | 56000 | PASS |
| `focal_budget_breaches` | == 0 | 0 | PASS |
| `focal_rejected_actions` | == 0 | 0 | PASS |
| `mean_pivotal_pnl` | >= -25000 | -16787.50 | PASS |
| `loss_stop_rate` | <= 10% | 1.2% | PASS |
| `health_stop_rate` | <= 75% | 61.9% | PASS |
| `minimum_family_mean_pnl` | >= -30000 | -16300 | PASS |
| `overall_mean_pnl` | > 0 | 91943.75 | PASS |
| `median_pnl` | >= 0 | -2000.00 | FAIL |
| `beat_flat_fraction` | >= 55% | 45.6% | FAIL |
| `positive_family_fraction` | >= 4 of 8 families | 4 | PASS |
| `raw_mean_retention` | >= 50% | 0.93 | PASS |

A PASS is a risk-control gate, not proof of positive real competition
expected value. If this decision is FAIL, production Liferaft must
remain flat. No post-final parameter changes are permitted.
