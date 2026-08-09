# Boat Party Ticket research report: revised recommendation

## Executive recommendation

The earlier dual-template RLS recommendation is **not validated strongly enough for production**. Its AUD 151,000 result was an in-sample reconstruction: the centered seven-day fixed and warped templates were built from the complete scored Round 1 path. On the genuinely disjoint chronological S1-to-S2 test, the same logic falls to AUD 22,390 for days 161:365, below a constant AUD 45 mean-reversion rule at AUD 59,030 and a calendar-plus-summer rule at AUD 49,070.

The current handoff should be:

- Primary candidate for human review: constant AUD 45 mean reversion, with no fitted seasonal template and no adaptive RLS.
- Simpler fallback: broad academic-calendar windows plus a summer policy beginning at the official summer landmark.
- Optional exploratory refinement: switch the constant-45 rule to a shrunk summer mean after the official summer date. This made AUD 59,490 on the chronological year-end window and had independent median/P10/worst P&L of AUD 60,953/AUD 22,165/AUD 4,985, but it was chosen after validation and is not preregistered.
- Do not promote dual-template RLS as primary. It does not improve the independent lower tail over fixed or warped RLS and adds complexity.

For planning only, the simple constant-45 candidate produced AUD 59,030 on the chronological S1-to-S2 plus summer test. Across 400 independent event-kernel and cross-semester scenarios it had median AUD 55,652, P10 AUD 19,062, worst AUD 2,139, and 100% positive paths. These are generator-conditioned scenarios, not a confidence interval. A cautious planning bracket is approximately AUD 20,000-AUD 60,000, with material uncertainty outside it.

## Evidence labels

The original evidence is preserved but relabelled:

| Result | Correct interpretation |
|---|---|
| Dual-template RLS AUD 151,000 | In-sample template reconstruction |
| Fixed-template RLS AUD 163,850 | In-sample template reconstruction |
| Fixed template plus residual AUD 164,290 | In-sample reconstruction / overlay benchmark |
| Original 96-path stress table | Template-conditioned sensitivity |
| Event-warped pseudo-2027 path | Model-conditioned timing sensitivity |

Those figures remain in the original [model results](results/model_results.csv), [generator comparison](results/generator_designs.csv), and [template-conditioned stress](results/stress_summary.csv) tables. They are not pooled with the new validation evidence.

## Disjoint chronological validation

The new S1-to-S2 templates use only Round 1 days 15:161, with local seven-day smoothing inside each source wave. No S2 prices, extrema, amplitudes, or normalization values enter template construction. RLS resets at day 161 to the frozen prior and updates only through each observed decision day.

| Model | S2 semester days 161:326 | S2 plus summer days 161:365 | Max DD, latter window |
|---|---:|---:|---:|
| Constant AUD 45 mean reversion | 46,610 | **59,030** | -3,770 |
| Calendar schedule plus summer | 36,650 | **49,070** | -3,330 |
| Broad calendar schedule | 36,650 | 36,650 | -3,330 |
| Fixed S1 transfer without RLS | 32,270 | 32,270 | -4,020 |
| Fixed S1 transfer RLS | 28,750 | 35,230 | -5,640 |
| Dual fixed/academic transfer agreement | 12,350 | 22,390 | -6,500 |
| Academic event-phase transfer RLS | -4,050 | 9,550 | -16,870 |
| Baseline-only RLS | 2,110 | 6,330 | -13,210 |

RLS does not beat the simple alternatives in the first 10, 10-30, 30-60, remaining-semester, or whole-semester windows. Its disjoint incremental value over the same fixed transfer without RLS is only AUD 2,960. Full details are in [heldout_rls.csv](results/heldout_rls.csv) and [rls_ablations.csv](results/rls_ablations.csv).

The reverse S2-to-S1 experiment is included only as a noncausal structural-transfer diagnostic. It uses later data to predict earlier data and is not deployable.

## Independent stress and timing uncertainty

The new 400-path stress set is separate from the old template-conditioned stress. It contains 200 low-parameter academic event-kernel paths and 200 cross-semester shape-transfer paths. Timing modes include fixed 2026 indices, official 2027 dates, and S1 Easter shifts of 8 and 11 days; local peak shifts from -14 to +14 days, asymmetric wave widths, missing secondary waves, amplitude rescaling, summer shifts, and independent AR(1) residuals are randomized.

| Model | Median | P10 | Worst | Positive rate |
|---|---:|---:|---:|---:|
| Dual-template agreement RLS | 71,768 | 22,790 | -9,728 | 99.0% |
| Fixed-template RLS | 71,697 | 22,866 | -9,358 | 99.3% |
| Warped-template RLS | 71,170 | 23,599 | -10,097 | 99.0% |
| Constant AUD 45 mean reversion | **55,652** | **19,062** | **2,139** | **100.0%** |
| Calendar schedule plus summer | 24,433 | 7,947 | -20,900 | 96.0% |
| Broad calendar schedule | 17,030 | 1,626 | -20,306 | 92.3% |

Agreement is not a meaningful lower-tail hedge: its P10 is slightly below fixed RLS, and its worst case is slightly worse than fixed RLS. It filters a median 11 of 364 decisions, so it usually leaves the two models trading together. Timing-specific results are in [rls_timing_stress.csv](results/rls_timing_stress.csv); all path-level results are in [independent_stress_detail.csv](results/independent_stress_detail.csv).

## Summer result

Summer mean reversion is independently more defensible than full-year seasonal RLS. Across the independent paths, the standalone shrunk mean (fixed AUD 45 prior with 14-day prior strength) had median AUD 11,263, P10 AUD 3,314, worst AUD -1,091, and 98.8% positive paths. The observed Round 1 official-summer diagnostic produced AUD 12,880. Use the official academic summer landmark rather than selecting a start date from the same summer's P&L.

See [summer_validation.csv](results/summer_validation.csv), [summer_independent_summary.csv](results/summer_independent_summary.csv), and [summer_shift_summary.csv](results/summer_shift_summary.csv).

## Final classification

| Strategy | Status |
|---|---|
| Constant AUD 45 mean reversion | Validated candidate, subject to the one-year limitation |
| Calendar schedule plus summer | Robust fallback |
| Broad calendar schedule | Robust fallback |
| Shrunk summer mean | Validated standalone overlay |
| Fixed seasonal template | Promising but unvalidated |
| Fixed-template RLS | Promising but unvalidated |
| Dual-template RLS agreement | Rejected as primary |

The detailed critique-by-critique audit is in [VALIDATION.md](VALIDATION.md). The executed [analysis notebook](analysis.ipynb) has zero error cells, and all research code remains under `research/boat_party/`. Production strategy and simulator files were not modified.
