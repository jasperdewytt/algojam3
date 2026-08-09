"""Small paper-driven Fintech Token follow-up models and diagnostics.

The existing causal EWMA implementation remains the benchmark.  This module
adds only latency-oriented alternatives justified by the supplied papers:

* fast/slow EWMA volatility acceleration;
* asymmetric entry/exit hysteresis;
* Adams--MacKay BOCPD with a Student-t Normal--Inverse-Gamma predictive;
* a BOCPD-triggered EWMA reset;
* retrospective PELT-style diagnostics for variance, magnitude and AR(1).

PELT functions are explicitly offline diagnostics.  They must not be passed
to a trading backtest or a production position builder.
"""

from dataclasses import dataclass
from math import lgamma, log, pi, sqrt
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from research.fintech_token.fintech_models import (
    LIMIT,
    as_float_array,
    ewma_regime,
    ewma_volatility,
    simple_reversal,
)


PRIMARY_CONFIGS: Tuple[Tuple[float, float, int], ...] = (
    (0.85, 0.75, 30),
    (0.90, 0.80, 30),
    (0.95, 0.85, 30),
)


def state_positions(
    changes: Sequence[float], state: Sequence[int], limit: int = LIMIT
) -> np.ndarray:
    """Convert a causal calm/volatile state path into tested positions.

    ``state[t]`` is the state used for the position earning ``changes[t]``.
    It is therefore based on observations through ``changes[t-1]`` only.
    ``-1`` means reverse and ``+1`` means follow.
    """

    changes = as_float_array(changes)
    state = np.asarray(state, dtype=int).reshape(-1)
    if len(state) != len(changes):
        raise ValueError("state and changes must have the same length")
    positions = np.zeros(len(changes), dtype=int)
    if len(changes) > 1:
        positions[1:] = (
            limit * state[1:] * np.sign(changes[:-1])
        ).astype(int)
    return positions


def causal_ensemble_state(
    changes: Sequence[float],
    configs: Iterable[Sequence[float]] = PRIMARY_CONFIGS,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Return majority state plus member states, volatilities and cutoffs."""

    changes = as_float_array(changes)
    member_states: List[np.ndarray] = []
    member_vols: List[np.ndarray] = []
    member_cutoffs: List[np.ndarray] = []
    for config in configs:
        state, vol, cutoff = ewma_regime(changes, *config)
        member_states.append(state)
        member_vols.append(vol)
        member_cutoffs.append(cutoff)
    if not member_states:
        return (
            np.zeros(len(changes), dtype=int),
            [],
            [],
            [],
        )
    majority = np.sign(np.sum(np.asarray(member_states), axis=0)).astype(int)
    if len(majority):
        majority[0] = 0
    return majority, member_states, member_vols, member_cutoffs


def causal_ensemble_positions(
    changes: Sequence[float],
    configs: Iterable[Sequence[float]] = PRIMARY_CONFIGS,
    limit: int = LIMIT,
) -> np.ndarray:
    state, _, _, _ = causal_ensemble_state(changes, configs)
    return state_positions(changes, state, limit=limit)


def regime_events(state: Sequence[int]) -> Dict[str, np.ndarray]:
    """Return entry and exit indices for the volatile state ``+1``."""

    state = np.asarray(state, dtype=int).reshape(-1)
    if len(state) < 2:
        return {"entries": np.array([], dtype=int), "exits": np.array([], dtype=int)}
    entries = np.where((state[1:] == 1) & (state[:-1] != 1))[0] + 1
    exits = np.where((state[1:] != 1) & (state[:-1] == 1))[0] + 1
    return {"entries": entries.astype(int), "exits": exits.astype(int)}


def shift_state_labels(state: Sequence[int], shift_days: int) -> np.ndarray:
    """Shift state labels for an offline timing diagnostic only.

    The convention is deliberately explicit: ``shift_days=-k`` moves labels
    ``k`` days earlier and therefore uses future/oracle information.  Positive
    shifts delay labels.  Missing edge labels are calm.  This function must
    never be used in a causal P&L.
    """

    state = np.asarray(state, dtype=int).reshape(-1)
    shifted = np.full(len(state), -1, dtype=int)
    for t in range(len(state)):
        source = t - int(shift_days)
        if 0 <= source < len(state):
            shifted[t] = state[source]
    if len(shifted):
        shifted[0] = 0
    return shifted


def shifted_state_positions(
    changes: Sequence[float], state: Sequence[int], shift_days: int
) -> np.ndarray:
    """Build an offline shifted-state path; never use for causal selection."""

    return state_positions(changes, shift_state_labels(state, shift_days))


def delayed_execution_positions(positions: Sequence[int]) -> np.ndarray:
    """Delay every action by one day for the execution-sensitivity audit."""

    positions = np.asarray(positions, dtype=int).reshape(-1)
    delayed = np.zeros(len(positions), dtype=int)
    if len(positions) > 1:
        delayed[1:] = positions[:-1]
    return delayed


def fast_slow_ratio_state(
    changes: Sequence[float],
    fast_lambda: float,
    slow_lambda: float,
    percentile: float,
    warmup: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal volatility-acceleration state using fast/slow EWMA ratio."""

    changes = as_float_array(changes)
    if not 0.0 < fast_lambda < 1.0 or not 0.0 < slow_lambda < 1.0:
        raise ValueError("EWMA lambdas must lie in (0, 1)")
    if fast_lambda >= slow_lambda:
        raise ValueError("fast lambda must be less than slow lambda")
    if not 0.0 < percentile < 1.0:
        raise ValueError("percentile must lie in (0, 1)")

    fast = ewma_volatility(changes, fast_lambda)
    slow = ewma_volatility(changes, slow_lambda)
    ratio = np.divide(
        fast,
        np.maximum(slow, 1e-12),
        out=np.ones(len(changes), dtype=float),
        where=np.maximum(slow, 1e-12) > 0,
    )
    state = np.zeros(len(changes), dtype=int)
    cutoffs = np.full(len(changes), np.nan, dtype=float)
    for t in range(1, len(changes)):
        if t > warmup and t - 1 >= warmup:
            prior = ratio[: t - 1]
            cutoff = float(np.quantile(prior, percentile))
            cutoffs[t] = cutoff
            state[t] = 1 if ratio[t - 1] >= cutoff else -1
        else:
            state[t] = -1
    return state, fast, slow, ratio, cutoffs


def fast_slow_ratio_positions(
    changes: Sequence[float],
    fast_lambda: float,
    slow_lambda: float,
    percentile: float,
    warmup: int = 30,
    limit: int = LIMIT,
) -> np.ndarray:
    state, *_ = fast_slow_ratio_state(
        changes, fast_lambda, slow_lambda, percentile, warmup
    )
    return state_positions(changes, state, limit=limit)


def asymmetric_entry_exit_state(
    changes: Sequence[float],
    entry_lambda: float,
    entry_percentile: float,
    exit_lambda: float,
    exit_percentile: float,
    warmup: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal hysteresis: fast entry, slower lower-threshold exit."""

    changes = as_float_array(changes)
    if not 0.0 < entry_lambda < 1.0 or not 0.0 < exit_lambda < 1.0:
        raise ValueError("EWMA lambdas must lie in (0, 1)")
    if not 0.0 < entry_percentile < 1.0 or not 0.0 < exit_percentile < 1.0:
        raise ValueError("percentiles must lie in (0, 1)")
    entry_vol = ewma_volatility(changes, entry_lambda)
    exit_vol = ewma_volatility(changes, exit_lambda)
    entry_cutoffs = np.full(len(changes), np.nan, dtype=float)
    exit_cutoffs = np.full(len(changes), np.nan, dtype=float)
    state = np.zeros(len(changes), dtype=int)
    volatile = False
    for t in range(1, len(changes)):
        if t <= warmup or t - 1 < warmup:
            volatile = False
            state[t] = -1
            continue
        entry_cutoffs[t] = float(
            np.quantile(entry_vol[: t - 1], entry_percentile)
        )
        exit_cutoffs[t] = float(
            np.quantile(exit_vol[: t - 1], exit_percentile)
        )
        if volatile:
            if exit_vol[t - 1] < exit_cutoffs[t]:
                volatile = False
        elif entry_vol[t - 1] >= entry_cutoffs[t]:
            volatile = True
        state[t] = 1 if volatile else -1
    return state, entry_vol, exit_vol, entry_cutoffs, exit_cutoffs


def asymmetric_entry_exit_positions(
    changes: Sequence[float],
    entry_lambda: float,
    entry_percentile: float,
    exit_lambda: float,
    exit_percentile: float,
    warmup: int = 30,
    limit: int = LIMIT,
) -> np.ndarray:
    state, *_ = asymmetric_entry_exit_state(
        changes,
        entry_lambda,
        entry_percentile,
        exit_lambda,
        exit_percentile,
        warmup,
    )
    return state_positions(changes, state, limit=limit)


def observation_transform(changes: Sequence[float], mode: str) -> np.ndarray:
    """Create a causal BOCPD observation stream."""

    changes = as_float_array(changes)
    if mode == "raw":
        return changes.copy()
    if mode == "absolute":
        return np.abs(changes)
    if mode == "squared":
        return changes**2
    if mode == "reversal_residual":
        residual = changes.copy()
        if len(changes) > 1:
            residual[1:] = changes[1:] + changes[:-1]
        return residual
    raise ValueError(f"unknown observation mode: {mode}")


def _student_t_logpdf(
    value: float, degrees: float, location: float, scale: float
) -> float:
    scale = max(float(scale), 1e-9)
    z = (float(value) - float(location)) / scale
    return (
        lgamma((degrees + 1.0) / 2.0)
        - lgamma(degrees / 2.0)
        - 0.5 * log(degrees * pi)
        - log(scale)
        - ((degrees + 1.0) / 2.0) * log1p(z * z / degrees)
    )


def log1p(value: float) -> float:
    """Small local wrapper kept separate for the scalar BOCPD loop."""

    return float(np.log1p(value))


def _nig_predictive(
    n: float,
    mean: float,
    m2: float,
    mu0: float,
    kappa0: float,
    alpha0: float,
    beta0: float,
) -> Tuple[float, float, float]:
    kappa = kappa0 + n
    mu = (kappa0 * mu0 + n * mean) / max(kappa, 1e-12)
    alpha = alpha0 + 0.5 * n
    beta = beta0 + 0.5 * m2
    if n > 0:
        beta += 0.5 * (kappa0 * n / max(kappa, 1e-12)) * (mean - mu0) ** 2
    degrees = 2.0 * alpha
    scale = sqrt(max(beta * (kappa + 1.0) / (alpha * kappa), 1e-12))
    return degrees, mu, scale


def _nig_update(
    n: float,
    mean: float,
    m2: float,
    value: float,
) -> Tuple[float, float, float]:
    new_n = n + 1.0
    delta = value - mean
    new_mean = mean + delta / new_n
    new_m2 = m2 + delta * (value - new_mean)
    return new_n, new_mean, new_m2


@dataclass
class BOCPDResult:
    """Causal run-length filtering output after each observed datum."""

    run_length_posterior: np.ndarray
    change_probability: np.ndarray
    predictive_mean: np.ndarray
    predictive_scale: np.ndarray
    observations: np.ndarray
    prior_mean: float
    prior_scale: float
    expected_duration: int
    observation_mode: str

    @property
    def short_run_probability(self) -> np.ndarray:
        """Causal posterior mass on a newly changed short run (r <= 2)."""

        if self.run_length_posterior.size == 0:
            return np.empty(0, dtype=float)
        return self.run_length_posterior[:, :3].sum(axis=1)

    @property
    def run_length_mean(self) -> np.ndarray:
        """Posterior mean run length after each observed datum."""

        if self.run_length_posterior.size == 0:
            return np.empty(0, dtype=float)
        run_lengths = np.arange(self.run_length_posterior.shape[1], dtype=float)
        return self.run_length_posterior @ run_lengths


def bocpd_student_t(
    changes: Sequence[float],
    expected_duration: int,
    observation_mode: str = "raw",
    prior_window: int = 30,
    max_run: Optional[int] = None,
) -> BOCPDResult:
    """Adams--MacKay BOCPD with a Student-t NIG predictive distribution.

    The run-length posterior after observation ``i`` uses observations through
    ``changes[i]`` only.  A constant hazard corresponds to the paper's
    geometric gap prior.  The startup prior is estimated only from the first
    ``prior_window`` observations, and callers use the output after warm-up.
    With a constant hazard, the posterior probability of the next reset is the
    hazard by construction; data-dependent timing is therefore read from the
    filtered run-length posterior, especially its short-run mass.
    """

    changes = as_float_array(changes)
    if expected_duration < 2:
        raise ValueError("expected_duration must be at least two")
    observations = observation_transform(changes, observation_mode)
    n_obs = len(observations)
    if n_obs == 0:
        return BOCPDResult(
            np.empty((0, 0)),
            np.empty(0),
            np.empty(0),
            np.empty(0),
            observations,
            0.0,
            1.0,
            expected_duration,
            observation_mode,
        )

    window = observations[: min(prior_window, n_obs)]
    prior_mean = float(np.median(window)) if len(window) else 0.0
    mad = float(np.median(np.abs(window - prior_mean))) if len(window) else 1.0
    prior_scale = max(1.4826 * mad, float(np.std(window)), 1e-3)
    kappa0 = 0.25
    alpha0 = 2.0
    beta0 = prior_scale**2 * (alpha0 - 1.0)
    hazard = 1.0 / float(expected_duration)
    max_run = n_obs if max_run is None else min(int(max_run), n_obs)

    posterior = np.zeros((n_obs, max_run + 1), dtype=float)
    change_probability = np.zeros(n_obs, dtype=float)
    predictive_mean = np.zeros(n_obs, dtype=float)
    predictive_scale = np.zeros(n_obs, dtype=float)

    # Each run-length hypothesis carries (n, mean, centered sum of squares).
    ns = np.zeros(max_run + 1, dtype=float)
    means = np.full(max_run + 1, prior_mean, dtype=float)
    m2s = np.zeros(max_run + 1, dtype=float)
    probs = np.zeros(max_run + 1, dtype=float)
    probs[0] = 1.0

    for t, value in enumerate(observations):
        # Vectorise the Normal--Inverse-Gamma sufficient-statistic update.  The
        # run-length loop is the conceptual algorithm from Adams--MacKay, but
        # keeping the arithmetic in NumPy makes the repeated bootstrap audit
        # practical on this small sample.
        kappa = kappa0 + ns
        locations = (kappa0 * prior_mean + ns * means) / np.maximum(kappa, 1e-12)
        alpha = alpha0 + 0.5 * ns
        beta = beta0 + 0.5 * m2s
        beta += 0.5 * (kappa0 * ns / np.maximum(kappa, 1e-12)) * (means - prior_mean) ** 2
        degrees = 2.0 * alpha
        scales = np.sqrt(np.maximum(beta * (kappa + 1.0) / (alpha * kappa), 1e-12))
        z = (float(value) - locations) / scales
        log_gamma = np.fromiter(
            (lgamma((df + 1.0) / 2.0) - lgamma(df / 2.0) for df in degrees),
            dtype=float,
            count=len(degrees),
        )
        log_predictive = (
            log_gamma
            - 0.5 * np.log(degrees * pi)
            - np.log(scales)
            - ((degrees + 1.0) / 2.0) * np.log1p(z * z / degrees)
        )

        max_log = float(np.max(log_predictive))
        likelihood = np.exp(log_predictive - max_log)
        weighted = probs * likelihood
        new_probs = np.zeros_like(probs)
        new_probs[0] = hazard * float(np.sum(weighted))
        if max_run > 0:
            new_probs[1:] = (1.0 - hazard) * weighted[:-1]
            # If the run-length state is truncated, retain its growth mass in
            # the final bucket rather than silently dropping probability.
            new_probs[max_run] += (1.0 - hazard) * weighted[max_run]
        total = float(np.sum(new_probs))
        if not np.isfinite(total) or total <= 0.0:
            new_probs[:] = 0.0
            new_probs[0] = 1.0
        else:
            new_probs /= total

        new_ns = np.zeros_like(ns)
        new_means = np.full_like(means, prior_mean)
        new_m2s = np.zeros_like(m2s)
        new_ns[0] = 1.0
        new_means[0] = float(value)
        if max_run > 0:
            new_ns[1:] = ns[:-1] + 1.0
            delta = float(value) - means[:-1]
            new_means[1:] = means[:-1] + delta / np.maximum(new_ns[1:], 1e-12)
            new_m2s[1:] = m2s[:-1] + delta * (float(value) - new_means[1:])
        posterior[t] = new_probs
        change_probability[t] = new_probs[0]
        predictive_mean[t] = float(np.sum(new_probs * new_means))
        # A compact mixture scale is sufficient for diagnostics; the exact
        # predictive distribution is represented by the run-length posterior.
        new_kappa = kappa0 + new_ns
        new_alpha = alpha0 + 0.5 * new_ns
        new_beta = beta0 + 0.5 * new_m2s
        new_beta += (
            0.5
            * (kappa0 * new_ns / np.maximum(new_kappa, 1e-12))
            * (new_means - prior_mean) ** 2
        )
        new_degrees = 2.0 * new_alpha
        new_scales = np.sqrt(
            np.maximum(
                new_beta * (new_kappa + 1.0) / (new_alpha * new_kappa), 1e-12
            )
        )
        predictive_variances = new_scales**2 * np.where(
            new_degrees > 2.0, new_degrees / (new_degrees - 2.0), 1.0
        )
        variance = float(
            np.sum(new_probs * (predictive_variances + (new_means - predictive_mean[t]) ** 2))
        )
        predictive_scale[t] = sqrt(max(variance, 1e-12))
        ns, means, m2s, probs = new_ns, new_means, new_m2s, new_probs

    return BOCPDResult(
        posterior,
        change_probability,
        predictive_mean,
        predictive_scale,
        observations,
        prior_mean,
        prior_scale,
        int(expected_duration),
        observation_mode,
    )


def bocpd_reset_ewma_state(
    changes: Sequence[float],
    expected_duration: int,
    observation_mode: str = "raw",
    cp_threshold: float = 0.20,
    lam: float = 0.90,
    percentile: float = 0.80,
    warmup: int = 30,
    max_run: Optional[int] = 180,
) -> Tuple[np.ndarray, BOCPDResult, np.ndarray, np.ndarray, np.ndarray]:
    """Use BOCPD only to reset/trigger a simple EWMA volatility state.

    Filtered short-run posterior mass above ``cp_threshold`` sets the current
    state to volatile and restarts the EWMA volatility history at the latest
    observed magnitude.  Direction remains the predeclared reversal/momentum
    rule; BOCPD never sees a future signed continuation outcome.  This uses
    the run-length posterior rather than the constant-hazard reset probability,
    which would otherwise equal the prior hazard at every time.
    """

    changes = as_float_array(changes)
    result = bocpd_student_t(
        changes,
        expected_duration,
        observation_mode=observation_mode,
        prior_window=warmup,
        max_run=max_run,
    )
    n = len(changes)
    state = np.zeros(n, dtype=int)
    vol = np.full(n, np.nan, dtype=float)
    cutoff = np.full(n, np.nan, dtype=float)
    reset = np.zeros(n, dtype=int)
    if n == 0:
        return state, result, vol, cutoff, reset

    variance = float(changes[0] ** 2)
    vol_history: List[float] = [sqrt(max(variance, 0.0))]
    for t in range(1, n):
        latest = float(changes[t - 1])
        if t >= 2:
            variance = lam * variance + (1.0 - lam) * latest**2
            vol_history.append(sqrt(max(variance, 0.0)))
        current_vol = sqrt(max(variance, 0.0))
        vol[t] = current_vol
        if t > warmup and result.short_run_probability[t - 1] >= cp_threshold:
            variance = max(latest**2, variance)
            current_vol = sqrt(max(variance, 0.0))
            vol[t] = current_vol
            vol_history = [current_vol]
            state[t] = 1
            reset[t] = 1
            continue
        if t > warmup and len(vol_history) > 1:
            cutoff[t] = float(np.quantile(vol_history[:-1], percentile))
            state[t] = 1 if current_vol >= cutoff[t] else -1
        else:
            state[t] = -1
    return state, result, vol, cutoff, reset


def bocpd_reset_ewma_positions(
    changes: Sequence[float],
    expected_duration: int,
    observation_mode: str = "raw",
    cp_threshold: float = 0.20,
    lam: float = 0.90,
    percentile: float = 0.80,
    warmup: int = 30,
    limit: int = LIMIT,
    max_run: Optional[int] = 180,
) -> np.ndarray:
    state, _, _, _, _ = bocpd_reset_ewma_state(
        changes,
        expected_duration,
        observation_mode,
        cp_threshold,
        lam,
        percentile,
        warmup,
        max_run,
    )
    return state_positions(changes, state, limit=limit)


def _variance_cost(values: np.ndarray, start: int, end: int) -> float:
    segment = values[start:end]
    if len(segment) < 2:
        return float("inf")
    variance = max(float(np.mean(segment**2)), 1e-12)
    return float(len(segment) * np.log(variance))


def _absolute_mean_cost(values: np.ndarray, start: int, end: int) -> float:
    segment = np.maximum(values[start:end], 1e-12)
    if len(segment) < 2:
        return float("inf")
    mean = max(float(np.mean(segment)), 1e-12)
    return float(len(segment) * (np.log(mean) + 1.0))


def _ar1_cost(values: np.ndarray, start: int, end: int) -> float:
    segment = values[start:end]
    if len(segment) < 4:
        return float("inf")
    design = np.column_stack((np.ones(len(segment) - 1), segment[:-1]))
    target = segment[1:]
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    variance = max(float(np.mean(residual**2)), 1e-12)
    return float(len(residual) * np.log(variance))


def pelt_segment(
    values: Sequence[float],
    cost_name: str,
    penalty: float,
    min_segment: int = 20,
) -> List[int]:
    """Offline PELT segmentation with a declared linear penalty.

    The recurrence and pruning rule follow Killick--Fearnhead--Eckley.  The
    small Fintech sample makes the worst-case quadratic path harmless, while
    the candidate set is still pruned using ``K=0`` for these likelihood-like
    costs.  Returned boundaries exclude 0 and n.
    """

    values = as_float_array(values)
    n = len(values)
    if min_segment < 2:
        raise ValueError("min_segment must be at least two")
    cost_map: Dict[str, Callable[[np.ndarray, int, int], float]] = {
        "variance": _variance_cost,
        "absolute_mean": _absolute_mean_cost,
        "ar1": _ar1_cost,
    }
    if cost_name not in cost_map:
        raise ValueError(f"unknown PELT cost: {cost_name}")
    cost = cost_map[cost_name]

    objective = np.full(n + 1, np.inf, dtype=float)
    objective[0] = -float(penalty)
    changepoints: List[List[int]] = [[] for _ in range(n + 1)]
    candidates: List[int] = [0]
    for end in range(1, n + 1):
        eligible = [
            start
            for start in candidates
            if end - start >= min_segment and np.isfinite(objective[start])
        ]
        if eligible:
            scores = np.asarray(
                [
                    objective[start] + cost(values, start, end) + penalty
                    for start in eligible
                ],
                dtype=float,
            )
            best_index = int(np.argmin(scores))
            best_start = eligible[best_index]
            objective[end] = float(scores[best_index])
            changepoints[end] = list(changepoints[best_start])
            if best_start > 0:
                changepoints[end].append(best_start)

            retained = []
            for start in candidates:
                if end - start < min_segment or not np.isfinite(objective[start]):
                    retained.append(start)
                elif objective[start] + cost(values, start, end) <= objective[end]:
                    retained.append(start)
            candidates = retained
        if np.isfinite(objective[end]):
            candidates.append(end)
    return changepoints[n]


def pelt_cost_inputs(changes: Sequence[float], cost_name: str) -> np.ndarray:
    """Prepare the offline series corresponding to a PELT cost."""

    changes = as_float_array(changes)
    if cost_name == "variance":
        return changes
    if cost_name == "absolute_mean":
        return np.abs(changes)
    if cost_name == "ar1":
        return changes
    raise ValueError(f"unknown PELT cost: {cost_name}")


__all__ = [
    "BOCPDResult",
    "PRIMARY_CONFIGS",
    "asymmetric_entry_exit_positions",
    "asymmetric_entry_exit_state",
    "bocpd_reset_ewma_positions",
    "bocpd_reset_ewma_state",
    "bocpd_student_t",
    "causal_ensemble_positions",
    "causal_ensemble_state",
    "delayed_execution_positions",
    "fast_slow_ratio_positions",
    "fast_slow_ratio_state",
    "observation_transform",
    "pelt_cost_inputs",
    "pelt_segment",
    "regime_events",
    "shift_state_labels",
    "shifted_state_positions",
    "simple_reversal",
    "state_positions",
]
