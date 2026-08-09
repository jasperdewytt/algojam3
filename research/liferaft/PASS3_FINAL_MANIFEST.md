# Pass 3 locked-final manifest

Status: **consumed exactly once under the Pass 4 final-catalogue lock**.

The one-time execution receipt is
`PASS4_FINAL_EXECUTION_RECEIPT.json`; the durable outcomes are
`PASS3_FINAL_REPORT.md`, `PASS4_FINAL_RESULTS.json`, and
`PASS4_FINAL_DECISION.md`. The locked source combined hash recorded below is
the hash printed before execution. No result-affecting source was changed
after that execution.

The normal experiment runner does not construct these cases. Do not compute
their paths or outcomes during validation work. The only execution path is the
explicit `python -m research.liferaft.pass3_experiments --final` command,
which runs the frozen Pass 4 final runner and writes durable final artifacts.

## Locked code hashes

The lock covers all code that can affect final construction or scoring,
including `archetypes.py` because locked populations use those archetypes.
Individual SHA-256 values:

| file | SHA-256 |
|---|---|
| `archetypes.py` | `ACDA007CF809E8EA76F9792BADEEDFF73ACBE4C46A4B59DE8522DD10939ACECB` |
| `simulator.py` | `5B61CA8C03E55444A656F5BD5C369BC687494CBED7B13D52B946ED45BBB55DBC` |
| `strategies.py` | `9BB4C0D5718D126B4725C2AC52C5AFBE774C4E5871F736CB6ACDB3C2018B7111` |
| `cold_start_strategies.py` | `321AE0C06DC168F5615F504AB4B4187D82E222A197BFF5880422096A6261EC34` |
| `pass3_scenarios.py` | `3AE0CC4DAA882078B3EA6AD8DB79145C4ECB5642385B746D5CEB5A9635C93983` |
| `pass3_experiments.py` | `FC0FAD5CA96D4E988A875C49F36DFF0C8AD79D8766654DFA0BC49B1B78F990BB` |
| `pass4_strategies.py` | `9D9DD76A3F22569F5D729277F2F23A3CCFC0660720D23F801C61C44E80E0C399` |
| `pass4_final.py` | `5975A66E56F690E949376FEF8C184511A73DBF6B9288489DA1839D8AA478A92C` |

Combined hash: `443487A9E3332D72839936BCFEF66A9CEFE1C4F6D10F64CD8C1EED488254903E`

The combined value is SHA-256 of the UTF-8 bytes of these exact newline-
terminated lines, in the table order above: `<filename> <individual hash>`.
Any later change to a locked file or frozen parameter invalidates this lock
and requires an explicit re-lock before final execution.

Decision protocol SHA-256: `2AEA526D327FC20B741B2AA3F4BA02D075818333E606A170A20E88C49F6220E3`

The development-only deterministic-cycle files are intentionally not part of
this catalogue or hash set. Adding a cycle strategy to a later final run would
be a deliberate candidate-catalogue change and would invalidate this lock,
requiring a new explicit re-lock before final execution.

## Frozen candidate catalogue

The final comparison uses the existing predeclared candidates without
parameter tuning in Pass 3.2:

- `flat`, `always_long`, `always_short`;
- `burnin1_markov`, `burnin3_markov`, `burnin5_markov`, and
  `burnin10_markov`, using Markov order 2, alpha 1.0, windows `(5, 10, 20)`,
  minimum expected P&L 1,000, and confidence margin 0.10;
- `online_last_counter`;
- `online_rolling`, using windows `(5, 10, 20)`, alpha 1.0, minimum support 3,
  minimum expected P&L 1,000, and confidence margin 0.10;
- `online_markov`, using order 2, alpha 1.0, minimum support 3, minimum
  expected P&L 1,000, and confidence margin 0.10;
- `online_ensemble`, using flat/last/rolling/Markov experts, learning rate
  0.5, weight bounds 0.05–0.70, and minimum support 3;
- `online_drift`, using the ensemble primary, flat fallback, quality minimum
  8, quality window 8, hit-rate threshold 0.40, and bad streak 2;
- `immediate_long_prior` and `flat_first_long_prior` (three-observation
  flat-first burn-in).
- `risk50_burnin1_markov`, wrapping the frozen `burnin1_markov` model with the
  Pass 4A risk controls recorded in `PASS4A_PROTOCOL.md`.

## Scenario manifest

- Families: `symmetric_random`, `short_biased_random`, `reactive_mixture`,
  `periodic`, `regime_change`, `gradual_drift`, `startup_zero_history`, and
  `margin_mixture`.
- Base seeds: ten offsets per family, beginning at `90,000`, with family
  blocks at `90,000`, `91,000`, ..., `97,000`.
- Each base seed has both `observe_and_ignore_actions` and `fully_inactive`
  members with the same population definition and seed.
- Population sizes cycle through the predeclared set `{3, 4, 5, 8, 9, 15}`.
- Timeline: `730` days, voting/marked boundary day `365`, pre-voting price
  `$100,000`.
- Locked scenario count: `160` paired mode cases.
- Candidate cell count if explicitly executed: `160 × 15 = 2,400`.

The final suite includes unseen symmetric and short-biased random mixtures,
reactive and periodic populations, regime changes, both drift directions,
startup-mode sensitivity, and pivotal/non-pivotal margin compositions. It is
not a validation holdout substitute and has not been executed.
