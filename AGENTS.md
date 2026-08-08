# AGENTS.md

## Goal

Build a robust AlgoJam 3 trading strategy that performs well on unseen Round 2 data. We have tonight and the weekend, so prioritise a working, testable submission over elaborate infrastructure.

## Working rules

- Put strategy logic in `trader_interface/algorithm.py`; avoid changing the supplied simulator unless fixing a confirmed bug.
- Read `README.md` and the PDFs in `docs/` before making assumptions about instruments or rules.
- Keep every position integral, within its instrument limit, and the total absolute portfolio value at or below AUD 600,000.
- Avoid look-ahead bias: decisions may use only data available through `self.day`.
- Prefer simple strategies supported by evidence, with conservative sizing and sensible behaviour at startup or with missing history.
- Treat Round 1 as development data, not something to memorise; favour ideas likely to generalise to Round 2.
- After changes, run `python simulation.py` from `trader_interface/` and report P&L, budget violations, and any errors.
- Keep changes small and leave brief comments explaining non-obvious trading logic.

## Priorities

1. Establish a valid baseline and reliable backtest.
2. Analyse instrument behaviour and test strategies one instrument at a time.
3. Combine the strongest signals while respecting the shared budget.
4. Stress-test edge cases, simplify, and prepare the final submission.
