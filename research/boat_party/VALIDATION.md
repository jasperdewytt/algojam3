# Boat Party Ticket second-stage validation audit

## Verdict first

The previous AUD 151,000 dual-template RLS recommendation does **not** survive disjoint validation strongly enough to implement as the primary strategy.

The full-path result was an in-sample template reconstruction: both centered seven-day templates were built from the complete scored Round 1 path. When the same RLS logic is given a template built only from Semester 1 and is scored on Semester 2, the dual transfer produces AUD 22,390 from days 161:365. A constant AUD 45 mean-reversion rule produces AUD 59,030 on the same decisions, and a broad calendar schedule plus official-date summer policy produces AUD 49,070.

The recommended direction is therefore a simpler candidate:

- Primary review candidate: constant AUD 45 mean reversion, full integer position only when the current price is on the opposite side of 45. This uses no fitted seasonal template and no adaptive RLS.
- Conservative fallback: broad academic-calendar windows plus an independently specified summer rule.
- Optional exploratory refinement: switch the constant-45 policy to a shrunk summer mean after the official summer landmark. This was tested after seeing the validation results, so it is not preregistered; it should not be treated as validated model selection.
- Do not promote dual-template RLS, full-year fixed-template RLS, or the centered seasonal template as the primary strategy without independent Round 2-like evidence.

This is still not a true Round 2 holdout: only one supplied year exists. The independent generators are transparent scenario families, not a confidence interval.

## What changed relative to the first investigation

The first investigation's key numbers remain in the original tables, but they are relabelled here:

| Earlier result | Correct label |
|---|---|
| Dual-template RLS AUD 151,000 | In-sample template reconstruction |
| Fixed-template RLS AUD 163,850 | In-sample template reconstruction |
| Fixed template plus residual AUD 164,290 | In-sample template reconstruction / overlay benchmark |
| Original 96-path stress table | Template-conditioned sensitivity |
| Event-warped pseudo-2027 path | Model-conditioned timing sensitivity |

The original stress paths were formed from the same seven-day template supplied to the strategy. They remain useful for parameter sensitivity, but they cannot support a general Round 2 P&L range.

## Chronological Semester 1 to Semester 2 test

The training template uses only Round 1 days 15:161, consisting of the two S1 wave episodes. Each source interval is smoothed locally, so centered smoothing is padded inside the source interval and cannot cross into S2. No S2 peak, trough, amplitude, normalization range, or turning point is read. RLS is reset at day 161 to the preregistered prior and then updates only with prices observed within S2.

The main S2 semester window is days 161:326, ending at the known 2026 summer landmark. The year-end window extends the same decision stream through day 364 so the summer overlay can be tested separately.

| Model | S2 semester P&L | S2 + summer P&L | Active hit, S2 + summer | Max DD |
|---|---:|---:|---:|---:|
| Constant AUD 45 mean reversion | 46,610 | **59,030** | 57.6% | -3,770 |
| Calendar schedule plus summer | 36,650 | **49,070** | 60.3% | -3,330 |
| Online baseline mean | 28,630 | 37,450 | 55.2% | -5,810 |
| Broad calendar schedule | 36,650 | 36,650 | 56.0% | -3,330 |
| Fixed S1-to-S2 transfer without RLS | 32,270 | 32,270 | 56.7% | -4,020 |
| Fixed S1-to-S2 transfer with RLS | 28,750 | 35,230 | 52.7% | -5,640 |
| Dual fixed/academic transfer agreement | 12,350 | 22,390 | 55.7% | -6,500 |
| Academic event-phase transfer with RLS | -4,050 | 9,550 | 54.7% | -16,870 |
| Baseline-only RLS | 2,110 | 6,330 | 51.2% | -13,210 |

RLS does not become useful after a short warm-up. In the first ten decisions fixed-transfer RLS loses AUD 1,190 versus AUD 1,350 for the constant and calendar rules. In days 10-30 it makes AUD 920 while the broad calendar makes AUD 12,500 and the dual rule loses AUD 5,790. In days 30-60 fixed RLS makes AUD 5,800, below the AUD 9,220 baseline-only RLS but close to the AUD 5,800 constant rule. Over the remaining semester, constant 45 makes AUD 31,040 versus AUD 23,220 for fixed-transfer RLS and AUD 13,850 for agreement.

The reverse S2-to-S1 test is a noncausal structural-transfer diagnostic because it uses later S2 data to predict the earlier S1 period. Its whole-period results were fixed transfer without RLS AUD 32,580, broad calendar AUD 30,690, academic RLS AUD 9,110, dual agreement AUD 8,870, and fixed RLS AUD 8,630. It does not support deployment, but it also gives no evidence that the sophisticated transfer is generally superior.

Full detail, including first 10, 10-30, 30-60, remaining, semester, and year-end windows, is in [heldout_rls.csv](results/heldout_rls.csv).

## What the AUD 151,000 depended on

The frozen full-year RLS score was AUD 163,850 for fixed-template RLS and AUD 151,000 for dual agreement. Rebuilding the shape from disjoint S1 prices drops the fixed transfer RLS to AUD 35,230 over the comparable year-end window and dual transfer to AUD 22,390. A fixed transferred template without RLS still makes AUD 32,270, so the incremental fixed-transfer RLS gain is only AUD 2,960.

The large apparent RLS benefit is therefore not established as an out-of-sample benefit. It is mostly the ability of an adaptive model to reconstruct the known Round 1 path around a template that already contains the scored year's future shape.

## Ablation results

All rows below use the identical chronological S1-to-S2 plus summer window.

| Ablation | P&L | Increment relevant to comparison |
|---|---:|---:|
| Constant AUD 45, no adaptation | **59,030** | Baseline |
| Online EWMA baseline | 37,450 | -21,580 vs constant 45 |
| Broad calendar schedule | 36,650 | -22,380 vs constant 45 |
| Fixed S1 transfer, no RLS | 32,270 | Shape transfer alone |
| Baseline-only RLS | 6,330 | -52,700 vs constant 45 |
| Fixed S1 transfer RLS | 35,230 | +2,960 vs same transfer without RLS |
| Academic S1 transfer RLS | 9,550 | -25,680 vs fixed transfer RLS |
| Dual transfer agreement | 22,390 | -12,840 vs fixed transfer RLS |
| Calendar schedule plus summer | 49,070 | +12,420 vs calendar alone |

This answers the main ablation question: RLS does not add stable value beyond a generic baseline or the transferred shape on the disjoint period. The constant prior is materially better than both online baseline adaptation and baseline-only RLS.

The complete table is [rls_ablations.csv](results/rls_ablations.csv).

## Leave-one-wave-out RLS

Each held wave receives a template made from other waves only. The actual recommended RLS update is reset at the held wave start. The prior leave-one-wave-out result was a median-shape slope rule; this table tests the actual RLS candidate.

| Held wave | Broad schedule | Transfer without RLS | Transfer RLS | Chronologically deployable? |
|---|---:|---:|---:|---|
| S1 large | **15,420** | 12,440 | -3,980 | No; source S2 is later |
| S1 small | 13,920 | **20,140** | 10,800 | No; source S2 is later |
| S2 large | **22,940** | 16,720 | 9,620 | Yes; source S1 is earlier |
| S2 small | **13,710** | 15,550 | -970 | Yes; source S1 is earlier |

The actual RLS candidate loses to the broad schedule on all four wave holdouts. Results are in [leave_one_wave_out_rls.csv](results/leave_one_wave_out_rls.csv).

## Placebo and timing findings

The full-path placebo table is an in-sample fragility diagnostic, not validation. The prior timing-displacement convention is reproduced below:

| Template displacement | Fixed RLS P&L | Dual P&L |
|---:|---:|---:|
| -14 | 61,670 | 56,910 |
| -7 | 95,510 | 90,480 |
| -3 | 132,350 | 119,770 |
| -1 | 164,430 | 145,910 |
| 0 | 163,850 | 151,000 |
| +1 | 161,050 | 156,090 |
| +3 | 128,790 | 132,120 |
| +7 | 36,210 | 39,370 |
| +14 | 21,490 | 16,130 |

The correct shape helps in-sample, but a 7-14 day error is enough to destroy much of the apparent edge. The constant, linear trend, reversed, phase-shuffled, secondary-wave-removed, and large/small-exchanged placebos also show that the full-year score is not an independent estimate of generalization. All placebo rows carry `uses_evaluated_prices = 1` and are explicitly excluded from validation claims. See [template_placebos.csv](results/template_placebos.csv).

## Independent generator results

The new stress set has 400 fixed-seed paths: 200 low-parameter academic event-kernel paths and 200 cross-semester shape-transfer paths. Event-kernel paths randomize broad opening/secondary amplitudes, asymmetric rise/decline speeds, peak timing, exam suppression, summer baseline and transition, residual volatility, and AR coefficient. Cross-semester paths use 14-day local source-wave shapes transferred across semesters, independently rescaled, warped, mixed between large and small roles, and sometimes omit a secondary wave. Both families include fixed 2026 timing, official 2027 timing, S1 Easter shifts of 8 and 11 days, local peak displacements from -14 to +14 days, wave-width variation, summer shifts, and residual AR(1) variation.

These paths are not constructed from the exact supplied seven-day template. They are nevertheless generator-conditioned scenarios, not an empirical confidence interval.

| Model | Median | P10 | Worst | Positive-path rate | Median DD |
|---|---:|---:|---:|---:|---:|
| Dual-template agreement RLS | 71,768 | 22,790 | -9,728 | 99.0% | -7,039 |
| Fixed-template RLS | 71,697 | 22,866 | -9,358 | 99.3% | -7,184 |
| Warped-template RLS | 71,170 | 23,599 | -10,097 | 99.0% | -7,349 |
| Constant AUD 45 mean reversion | **55,652** | **19,062** | **2,139** | **100.0%** | -6,348 |
| Online baseline mean | 54,577 | 13,078 | -8,330 | 98.5% | -6,367 |
| Baseline-only RLS | 53,756 | 8,903 | -14,882 | 95.5% | -7,199 |
| Calendar schedule plus summer | 24,433 | 7,947 | -20,900 | 96.0% | -7,178 |
| Broad calendar schedule | 17,030 | 1,626 | -20,306 | 92.3% | -7,178 |
| Fixed template without RLS | 9,949 | -11,968 | -38,001 | 72.3% | -12,885 |

The constant-45 result is encouraging but still depends on the generator families' use of a 45 AUD baseline. A cautious planning bracket is approximately AUD 20,000-AUD 60,000 for the simple candidate: the lower end is near its independent P10 and the upper end is near its independent median and the chronological S1-to-S2 year-end result. This is a planning range, not a claimed statistical interval; paths outside it remain plausible.

The complete path-level data are in [independent_stress_detail.csv](results/independent_stress_detail.csv), with summaries in [independent_stress_summary.csv](results/independent_stress_summary.csv). Timing-specific results are in [rls_timing_stress.csv](results/rls_timing_stress.csv).

## Does agreement protect against timing uncertainty?

No material lower-tail protection was found.

Across the 400 independent paths, dual agreement had P10 AUD 22,790 versus fixed RLS AUD 22,866 and warped RLS AUD 23,599. Its worst case AUD -9,728 was slightly worse than fixed RLS AUD -9,358 and better than warped RLS AUD -10,097, but not by a meaningful or consistent amount. Agreement was active for a median 353 of 364 decision days and filtered a median 11 days, so it rarely changed the exposure. It beat fixed RLS on 199/400 paired paths and warped RLS on 200/400, but was at least as good as both on only 1 path.

Under official 2027 event timing, median P&L was 89,432 for dual, 90,070 for fixed RLS, and 90,114 for warped RLS. Under an 8-day early S1 Easter cycle, dual median was 47,513 versus 45,699 fixed and 48,281 warped. The hedge sometimes sits between the two models, but it does not improve the independent lower tail enough to justify its complexity.

## Summer-only validation

Summer reversion has more independent support than the full-year seasonal RLS, but the policy should be simple and event-dated:

| Summer policy | Round 1 official-summer diagnostic | Independent median | Independent P10 | Independent worst | Positive rate |
|---|---:|---:|---:|---:|---:|
| Fixed AUD 45 | 12,420 | 6,056 | 297 | -3,577 | 93.5% |
| Online mean | 12,420 | 7,187 | 434 | -1,978 | 94.8% |
| Online median | 12,080 | 11,013 | 1,833 | -4,471 | 96.3% |
| Shrunk mean, prior AUD 45 with 14-day prior strength | **12,880** | **11,263** | **3,314** | **-1,091** | **98.8%** |

The shrunk mean was strongest in the independent generator set and remains positive across the observed official-summer diagnostic. It should be treated as an optional overlay, not as evidence for full-year RLS. Starting at day 302 or 322 was retained as a sensitivity check; the official 2026/2027 summer landmarks are the defensible deployment dates. See [summer_validation.csv](results/summer_validation.csv), [summer_independent_summary.csv](results/summer_independent_summary.csv), and [summer_shift_summary.csv](results/summer_shift_summary.csv).

## Decision classification

| Strategy | Classification | Reason |
|---|---|---|
| Constant AUD 45 mean reversion | **Validated candidate** | Best chronological S1-to-S2 simple rule and strong independent scenario lower tail; still only one real held-out semester |
| Calendar schedule plus summer | **Robust fallback** | Simpler event prior; lower P&L than constant 45 but useful if a fixed equilibrium is viewed as too strong |
| Broad calendar schedule | **Robust fallback** | Transparent and valid, but lower P&L in both disjoint and independent scenarios |
| Shrunk summer mean overlay | **Validated standalone overlay** | Better independent summer lower tail; use only from official summer date |
| Fixed seasonal template without RLS | **Promising but unvalidated** | Transfer works sometimes, but full-year template score is circular and independent no-RLS score is weak |
| Fixed-template RLS | **Promising but unvalidated** | Small disjoint gain over its transfer baseline and strong model-conditioned synthetic score; no clean Round 2 evidence |
| Dual-template RLS agreement | **Rejected as primary** | Fails disjoint S1-to-S2 comparison and does not improve independent lower tail |

The post-validation constant-45 plus shrunk-summer hybrid made AUD 59,490 on the chronological year-end window and had independent median/P10/worst AUD 60,953/AUD 22,165/AUD 4,985 across 400 paths. It is explicitly **exploratory post-validation**, not a preregistered result. The primary agent should decide whether its small improvement is worth adding complexity; the frozen constant-45 policy is the cleaner handoff.

## Leakage and validity audit

All seven leakage checks pass:

- Perturbing every S2 price by +AUD 17 leaves both S1-derived templates unchanged.
- Source indices are exactly days 15:161 and disjoint from S2 evaluation days 161:365.
- Centered smoothing is applied only inside source wave intervals.
- No held-out min, max, amplitude, or normalization range is used.
- Academic landmarks come from encoded official dates, not price extrema.
- A future-price perturbation after day 200 leaves all earlier RLS decisions unchanged.
- All validation positions are integral, within the 1,000-unit instrument limit, and have zero AUD 600,000 budget violations in the Boat-only test.

See [template_provenance.csv](results/template_provenance.csv) and [leakage_checks.csv](results/leakage_checks.csv).

## Files and status

- [analysis.py](analysis.py) now contains segment-local transfer templates, reset-window RLS, leakage checks, independent generators, and summer policies. It remains research-only.
- [analysis.ipynb](analysis.ipynb) was executed end to end with 20 cells and zero error outputs.
- [REPORT.md](REPORT.md) contains the revised high-level research narrative.
- New tables: [heldout_rls.csv](results/heldout_rls.csv), [rls_ablations.csv](results/rls_ablations.csv), [template_placebos.csv](results/template_placebos.csv), [rls_timing_stress.csv](results/rls_timing_stress.csv), [independent_stress_summary.csv](results/independent_stress_summary.csv), [independent_stress_detail.csv](results/independent_stress_detail.csv), [summer_validation.csv](results/summer_validation.csv), [template_provenance.csv](results/template_provenance.csv), [leave_one_wave_out_rls.csv](results/leave_one_wave_out_rls.csv), and the exploratory hybrid tables.
- New figures: [disjoint_validation_cumulative_pnl.png](figures/disjoint_validation_cumulative_pnl.png) and [independent_stress_pnl.png](figures/independent_stress_pnl.png).

The supplied production strategy, simulator, CSV, and PDFs were not modified.
