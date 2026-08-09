# Fintech Token paper follow-up

## Decision in one paragraph

Detection timing is adequate in this 364-change sample; the evidence does not support replacing the current detector with a faster change detector. The causal EWMA ensemble remains the simplest policy with a large and interpretable Round 1 advantage: $157,409 versus $75,369 for simple reversal, or +$82,040 paired incremental P&L. Its improvement is concentrated in Q2, so this is not a claim that the edge is permanent. The ratio and asymmetric alternatives do not beat reversal reliably, and BOCPD-reset EWMA is not stable enough to justify its additional online state. Recommendation: **retain the existing EWMA ensemble** and freeze its parameters for Round 2.

All P&L uses the simulator alignment position[t] * (P[t+1] - P[t]). A position at decision index t is built from changes through t-1 only. The full tables and figures are in results/ and figures/ beside this file.

## Reproduction and timing audit

The current ensemble is the three-member diagonal family:

    (lambda=0.85, percentile=0.75, warm-up=30)
    (lambda=0.90, percentile=0.80, warm-up=30)
    (lambda=0.95, percentile=0.85, warm-up=30)

The supplied figures reproduce exactly:

| Quantity | Reproduced result |
|---|---:|
| Prices / changes | 365 / 364 |
| Simple reversal P&L | $75,369 |
| EWMA ensemble P&L | $157,409 |
| Incremental P&L over reversal | $82,040 |
| Ensemble volatile/different days | 73 |
| Q1 / Q2 / Q3 / Q4 incremental P&L | -$1,788 / +$82,252 / +$1,576 / $0 |

The seven causal volatile runs are unchanged:

| Entry–exit | Days | Incremental P&L |
|---|---:|---:|
| 64–65 | 2 | -$86 |
| 69–78 | 10 | +$6,466 |
| 84–89 | 6 | -$8,168 |
| 115–129 | 15 | +$5,050 |
| 132–154 | 23 | +$54,754 |
| 157–163 | 7 | +$4,992 |
| 174–183 | 10 | +$19,032 |

There is no general gain from moving labels earlier in the offline audit. Relative to the causal shift 0, shifts of -1, -2, and -5 give +$68,174, +$44,526, and +$43,772, respectively, versus +$82,040 at shift 0. Negative shifts use future labels and are impossible trading policies. A one-day delayed state gives +$80,846 and a two-day delay gives +$57,682; the separate delayed-execution audit, which delays both the candidate and reversal benchmark, gives the ensemble +$53,852 over delayed reversal.

The event study shows a small average entry-day continuation (+$5.1 in signed units) and stronger mean continuation on several days after entry. The entry day itself is not a large, consistently missed opportunity. The exit-day continuation is negative on average (-$11.8), so the data do not show a systematic need to hold volatile mode longer. The two negative ensemble episodes are false/unhelpful entries, but the faster lambda settings create many more of them:

| EWMA lambda, q=.80 | Entries | False runs | Volatile days | Incremental P&L |
|---:|---:|---:|---:|---:|
| .80 | 15 | 8 | 80 | $43,652 |
| .85 | 13 | 7 | 81 | $51,816 |
| .90 | 7 | 2 | 73 | $82,040 |
| .95 | 4 | 2 | 71 | $55,718 |
| .97 | 1 | 0 | 72 | $60,110 |

The previously noted retrospective grid winner (0.97, .90, 30) earns $163,069 (+$87,700 over reversal), but it was selected after seeing Round 1 and is not frozen as the recommendation. The slower lambda helps mainly by avoiding false entries and stabilising the threshold; this is evidence against a simple “the detector must be faster” explanation. The single long .97, q=.80 run also shows that too much smoothing can miss useful episode boundaries, which is why the predeclared .90 ensemble is a better compromise.

The offline AR PELT diagnostic places boundaries near several EWMA entries (for example 63 versus 64, 83 versus 84, 112 versus 115, 132 versus 132, and 156 versus 157), but PELT is full-sample retrospective evidence. It does not convert those boundaries into a causal trading signal. Overall Phase 1 conclusion: **detection timing is adequate; false entries and episode concentration are more important concerns than demonstrated late entry or exit.**

## Paper-to-problem mapping

### Adams and MacKay: Bayesian online changepoint detection

The paper maintains a posterior over the current run length r_t, with a hazard transition:

    p(r_t=0 | r_{t-1})       = H(r_{t-1}+1)
    p(r_t=r_{t-1}+1 | ...)  = 1 - H(r_{t-1}+1)
    p(x_{t+1}|x_1:t)         = sum_r p(x_{t+1}|r_t=r, x_t^(r)) p(r_t=r|x_1:t)

For a constant hazard H=1/L, L is the expected geometric segment duration. The supplied paper uses conjugate updates and a predictive distribution for each run-length hypothesis. This follow-up uses a conjugate Normal–Inverse-Gamma update, whose marginal predictive is Student-t, with robust scale initialisation from the first 30 observations.

The observation is supplied causally. We tested raw changes, absolute changes, squared changes, and a lag-one reversal residual diagnostically. A decision never uses sign(d_t) * d_{t+1}; that outcome is only calculated in the event study after the future change exists. The traded BOCPD candidates use raw changes, expected durations 20/40/60/90, short-run posterior mass P(r_t <= 2) >= .20, and use BOCPD only to reset/accelerate EWMA. The direction rule remains the predeclared reversal/momentum switch.

A constant hazard has an important implementation detail: the posterior probability of the next reset is equal to the constant hazard at every time. Therefore the reset trigger uses the filtered short-run run-length posterior, not the raw one-step hazard probability. This is still causal and uses observations through the current decision only. BOCPD did identify some episode starts, but its duration choice materially changes the policy: full-sample incremental P&L ranged from -$5,548 (duration 20) to +$16,302 (duration 90), with no stable improvement over the EWMA ensemble.

BOCPD can improve online detection of a distributional change; it cannot infer the direction of the next move, guarantee a fast volatility estimate, or make a small sample large enough to estimate many regimes. Its filtered run-length posterior is causal; smoothed run lengths or Viterbi labels would be retrospective and were not used.

### Hamilton and Susmel: SWARCH

SWARCH models residuals with a conditional variance process whose scale is multiplied by a latent Markov state:

    y_t = x + phi*y_{t-1} + u_t
    u_t = sqrt(h_t) * epsilon_t
    h_t = a0 + a1*u^2_{t-1} + ... + aq*u^2_{t-q} + leverage term
    h_t(state) = gamma_state * h_t,  gamma_1 = 1

The state transition matrix creates persistent variance regimes. Hamilton and Susmel find that Student-t innovations and a small two-state model are useful in their much longer weekly sample, while additional regimes can be supported by only a few observations and lead to unstable curvature.

This structure addresses volatility persistence, not directional continuation. A SWARCH volatility forecast would still need the same separately specified reversal/momentum rule. The Fintech sample has only 364 changes, and the current EWMA already captures the useful state-conditioned directional effect. A minimal two-state SWARCH would add ARCH coefficients, a regime scale, transition probabilities, and possibly a tail parameter before any direction rule is estimated. That estimation risk is not justified by the timing audit, so SWARCH was mapped but not traded.

### Rydén, Teräsvirta and Åsbrink: financial-return HMMs

Their zero-mean Gaussian HMM represents returns as a mixture of state-specific variances and a Markov state. With d states, the transition probabilities and variances already give d(d-1)+d=d² parameters before adding a mean or AR structure. Their approximately 17,055-return study shows that a small HMM can reproduce mixture tails, volatility clustering, and higher-order dependence, but estimates vary considerably between subseries and HMM dependence decays geometrically.

The correct transfer here is an HMM used only as a volatility-state detector, with a predeclared directional rule. It does not support transplanting their state count or estimating state-specific AR direction from 364 changes. The previous custom MS-AR result rejects that tested implementation, not the entire HMM family. A minimal volatility HMM remains a possible Round 2 follow-up if another sustained episode supplies enough data, but it is not justified for this Round 1 handoff.

### Killick, Fearnhead and Eckley: PELT

PELT minimises a retrospective segmentation objective:

    F(s) = min_t { F(t) + C(y[t+1:s]) + beta }

with a BIC/SIC-style penalty beta = p log(n) or a declared multiple thereof. Its pruning theorem removes candidate last changepoints without changing the exact optimum under the stated cost condition and gives linear expected complexity under segment assumptions. The follow-up applied variance, absolute-change-mean, and AR(1) costs with penalty multiples 1, 2, 4, and 6 of log(n). PELT was used only to compare retrospective boundaries with EWMA transitions; no PELT label enters P&L.

The main sources were the supplied full PDFs: [Adams–MacKay](https://arxiv.org/abs/0710.3742), [Hamilton–Susmel](https://doi.org/10.1016/0304-4076(94)90067-1), [Rydén–Teräsvirta–Åsbrink](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291099-1255%28199805/06%2913%3A3%3C217%3A%3AAID-JAE476%3E3.0.CO%3B2-V), and [PELT](https://arxiv.org/abs/1101.1438). The arXiv PELT version was the main source; the second supplied PELT PDF was consulted only as a duplicate version. No inaccessible paper was needed to make the decision.

## Paired causal robustness

The central statistic is always candidate P&L minus simple-reversal P&L on the same path. The previous absolute-profit bootstrap was not used as the decision criterion.

### Paired moving-block bootstrap

Core candidates use 1,000 repetitions for each block length. BOCPD candidates use 250 repetitions because each path requires a full filtered run-length recursion; the actual count is recorded rather than presented as equal statistical power.

| Candidate | Block 5 mean / P(>0) | Block 10 mean / P(>0) | Block 20 mean / P(>0) | Block 40 mean / P(>0) |
|---|---:|---:|---:|---:|
| EWMA ensemble | -$3,702 / .441 | +$11,507 / .646 | +$28,154 / .723 | +$32,015 / .706 |
| Best ratio | +$1,107 / .577 | +$9,805 / .690 | +$13,990 / .714 | +$8,712 / .642 |
| Best asymmetric | -$20,498 / .302 | -$7,792 / .451 | +$2,799 / .545 | +$10,065 / .550 |
| BOCPD duration 90 | -$2,214 / .516 | +$9,959 / .620 | +$17,708 / .692 | +$18,434 / .708 |

Every bootstrap confidence interval for incremental P&L includes zero. The ensemble is the only candidate with a strong full-sample edge, a better drawdown profile, and a consistently positive medium/long-block paired result without adding a new detector. The stationary bootstrap gives the ensemble mean incremental P&L +$22,240 / +$31,180 with block means 10/20 and P(>0)=.717 for both. It gives no decisive support to a challenger.

### Episode and Q2 exclusions

Removing each EWMA episode one at a time leaves incremental P&L between +$27,286 (remove the large 132–154 run) and +$90,208 (remove the negative 84–89 run). Thus the total is not arithmetically equal to the one Q2 episode, although that episode contributes +$54,754 and is the dominant single run. Removing all Q2 observations leaves only -$212 incremental P&L. This is the principal overfitting concern and the reason the policy is a cautious Round 2 test, not a proven invariant.

The largest-change audit is less alarming: EWMA incremental P&L remains +$73,406, +$81,882, +$81,882, and +$76,026 after removing the largest 1, 3, 5, and 10 absolute changes. The raw P&L is therefore not explained by one isolated jump, but the regime edge is still concentrated in Q2’s serial episode.

### Restarts, delay, and lookahead

Fresh chronological starts at days 60 and 90 give ensemble incremental P&L +$59,350 and +$32,500; starts at 120 and 180 contain no later volatile state and give zero; start 240 gives -$7,646. Independent 91-day resets give -$1,788, +$36,538, $0, and $0 across the four blocks. Independent 60-day resets are mixed, with +$12,014 in the block containing the early Q2 episode and negative results in later blocks.

Every candidate and baseline passed the future-data perturbation test at cuts 60, 90, 120, 180, 240, and 300: positions before the cut were unchanged when all later changes were altered. This is an explicit test of causal prefix invariance, not a proof that the implementation is profitable. Delaying both candidate and reversal actions by one day leaves the ensemble ahead by +$53,852; this is a useful execution sensitivity result, though it does not remove the Q2 concentration concern.

The combined family-wise time-shuffle null covered 98 configurations: the previous fixed family of 88 plus the 10 new ratio/asymmetric/BOCPD configurations. The selected EWMA path has permutation p=.0099, and the maximum-over-family p=.0099 with 100 permutations. This supports serial dependence in Round 1 after accounting for the tested family. It does **not** prove transfer to Round 2. The circular-shift family result is weaker (p=.198); circular shifts preserve local serial structure and test start-location sensitivity, not an IID no-predictability null.

## Model-selection table

The challenger row for each family is its best full-sample member, but the family was frozen before results were inspected. Restart lists incremental P&L at starts 60/90/120/180/240.

| Model | Causal? | Parameter count / complexity | Full-sample P&L | Restart / walk-forward results | Max drawdown | Robustness-test outcome | Overfitting risk | Recommendation |
|---|---|---|---:|---|---:|---|---|---|
| Simple reversal | Yes | 0 estimated; one sign rule | $75,369; baseline inc. $0 | Baseline | -$34,766 | Reference for every paired test | Low | Keep as fallback |
| EWMA diagonal ensemble | Yes | 3 low-complexity members; 9 frozen values; scalar EWMA states | $157,409; inc. +$82,040 | +59,350 / +32,500 / 0 / 0 / -7,646 | -$8,996 | MBB 5/10/20/40: -3.7/+11.5/+28.2/+32.0k; delayed inc +$53.9k; Q2-excluded -$212 | Medium: Q2 concentration and frozen family choice | **Retain** |
| Fast/slow ratio family | Yes | 4 frozen values; one ratio and threshold | Best $61,449; inc -$13,920 | +2,154 / 0 / -20,948 / -27,192 / -4,238 | -$23,898 | All three full-sample incremental results negative; Q2-excluded -$33.4k | Medium/high; acceleration threshold is unstable | Reject |
| Asymmetric entry/exit family | Yes | 5 frozen values; hysteresis state | Best $122,211; inc +$46,842 | +69,732 / +64,750 / 0 / -810 / -19,342 | -$10,718 | MBB block 5/10 negative; Q2-excluded -$5,986; delayed inc +$56.6k | High; hand-tuned exit/entry asymmetry | Reject |
| BOCPD-reset EWMA | Yes | Online run-length posterior; duration, observation, trigger, EWMA settings; O(n) run states | Best $91,671; inc +$16,302 | +53,556 / +32,608 / -1,858 / 0 / -7,646 | -$25,976 | Duration 20 is negative; duration 90 is positive but all CIs cross zero; 250-rep MBB | High for 364 changes; lower bootstrap power and hazard sensitivity | Reject |
| Minimal variance HMM / SWARCH | Potentially | HMM d=2 already 4 variance/transition parameters; SWARCH adds ARCH and regime scale; direction remains separate | Not traded | Not evaluated as a new policy | — | Paper mapping says volatility-only detector is plausible, but timing audit found no clear missing volatility structure | High; state/local-max estimation from 364 changes | Defer |
| PELT | No | Offline segmentation and penalty; retrospective | Not applicable | Diagnostic boundaries only | — | Useful boundary comparison; forbidden in causal P&L | Certain lookahead if traded | Diagnostic only |

## Final Round 2 policy

Use the existing EWMA ensemble unchanged:

- parameters: (0.85, .75, 30), (0.90, .80, 30), (0.95, .85, 30);
- use each member’s EWMA volatility and expanding percentile threshold calculated from earlier volatility estimates only;
- during the 30-change warm-up, reverse the latest observed change;
- take the majority state: calm = -100 * sign(latest change), volatile = +100 * sign(latest change);
- if fewer than two prices exist, the latest change is missing/non-finite, or the latest change is exactly zero, use position 0; missing history then resumes the same warm-up rule;
- retain the simple-reversal fallback throughout warm-up and do not adapt parameters after seeing Round 2 outcomes.

The Fintech Token’s maximum required capital at the full ±100 limit is about $85,434 on this path, below the AUD $600,000 portfolio budget. This is a research handoff only; no production file was changed. A copy-ready sketch appears below for review and should not be pasted into algorithm.py without the repository interface checks.

    import numpy as np

    LIMIT = 100
    CONFIGS = ((0.85, 0.75), (0.90, 0.80), (0.95, 0.85))
    WARMUP = 30

    def fintech_position(price_history):
        prices = np.asarray(price_history, dtype=float)
        if len(prices) < 2 or not np.all(np.isfinite(prices[-2:])):
            return 0
        changes = np.diff(prices)
        latest = float(changes[-1])       # d_t; position earns d_{t+1}
        if latest == 0.0:
            return 0
        if len(changes) <= WARMUP:
            return int(-LIMIT * np.sign(latest))

        votes = []
        for lam, percentile in CONFIGS:
            variance = float(changes[0] ** 2)
            vols = [np.sqrt(max(variance, 0.0))]
            for move in changes[1:]:
                variance = lam * variance + (1.0 - lam) * float(move ** 2)
                vols.append(np.sqrt(max(variance, 0.0)))
            current = vols[-1]
            cutoff = float(np.quantile(np.asarray(vols[:-1]), percentile))
            votes.append(1 if current >= cutoff else -1)

        state = 1 if sum(votes) > 0 else -1
        return int(LIMIT * state * np.sign(latest))

## Remaining uncertainty and follow-up

Round 1 does not contain a genuinely independent volatile episode outside Q2 large enough to establish transfer. Round 2 should therefore be treated as the validation sample: log the frozen EWMA state, all three member states, latest change, volatility, threshold, and position before each decision. If a new sustained episode appears, compare the frozen ensemble with simple reversal on that episode before considering any online model change. Do not use PELT or smoothed HMM labels for live decisions. No additional inaccessible paper is material to the current decision; obtaining a larger independent return sample would be more valuable than adding another model family.
