"""Causal Fintech Token strategy and model implementations.

The simulator sets a position after observing price ``P[t]`` and pays
``position[t] * (P[t+1] - P[t])``.  All position builders in this module use
that convention: position index ``t`` can inspect changes with indices below
``t`` only.

The module deliberately uses only NumPy and Pandas-compatible primitives.  In
particular, the two-state Markov-switching AR(1) fit has a small, explicit EM
implementation so that the research does not depend on a black-box smoother.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LIMIT = 100
EPS = 1e-12


def as_float_array(values: Sequence[float]) -> np.ndarray:
    """Return a one-dimensional finite float array."""

    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("Fintech inputs must be finite")
    return array


def price_to_changes(prices: Sequence[float]) -> np.ndarray:
    """Convert an observed price path to one-day changes."""

    prices = as_float_array(prices)
    if len(prices) < 2:
        return np.empty(0, dtype=float)
    return np.diff(prices)


def _direction_position(direction: float, limit: int = LIMIT) -> int:
    """Convert a signed forecast to an integral bounded position."""

    return int(limit * np.sign(direction))


def simple_reversal(changes: Sequence[float], limit: int = LIMIT) -> np.ndarray:
    """Reverse the latest observed change; flat on the first decision."""

    changes = as_float_array(changes)
    positions = np.zeros(len(changes), dtype=int)
    if len(changes) > 1:
        positions[1:] = (-limit * np.sign(changes[:-1])).astype(int)
    return positions


def simple_momentum(changes: Sequence[float], limit: int = LIMIT) -> np.ndarray:
    """Follow the latest observed change; flat on the first decision."""

    changes = as_float_array(changes)
    positions = np.zeros(len(changes), dtype=int)
    if len(changes) > 1:
        positions[1:] = (limit * np.sign(changes[:-1])).astype(int)
    return positions


def constant_position(
    changes: Sequence[float], direction: int, limit: int = LIMIT
) -> np.ndarray:
    """A predeclared always-long/always-short/flat benchmark."""

    changes = as_float_array(changes)
    return np.full(len(changes), int(np.clip(direction, -1, 1) * limit), dtype=int)


def ewma_volatility(changes: Sequence[float], lam: float) -> np.ndarray:
    """Causal EWMA standard deviation aligned with observed changes.

    ``vol[i]`` is the estimate after observing ``changes[i]``.  A trading
    decision at index ``t`` therefore compares ``vol[t-1]`` with
    ``vol[:t-1]``; the current estimate is never included in its own cutoff.
    """

    changes = as_float_array(changes)
    if not 0.0 < lam < 1.0:
        raise ValueError("lambda must lie strictly between zero and one")
    vol = np.zeros(len(changes), dtype=float)
    if len(changes) == 0:
        return vol
    variance = float(changes[0] ** 2)
    vol[0] = np.sqrt(max(variance, 0.0))
    for i in range(1, len(changes)):
        variance = lam * variance + (1.0 - lam) * float(changes[i] ** 2)
        vol[i] = np.sqrt(max(variance, 0.0))
    return vol


def _normalise_ewma_config(
    config: Sequence[float], default_warmup: int = 30
) -> Tuple[float, float, int]:
    if len(config) == 2:
        lam, percentile = config
        warmup = default_warmup
    elif len(config) == 3:
        lam, percentile, warmup = config
    else:
        raise ValueError("EWMA config must be (lambda, percentile[, warmup])")
    lam = float(lam)
    percentile = float(percentile)
    warmup = int(warmup)
    if not 0.0 < lam < 1.0:
        raise ValueError("lambda must lie strictly between zero and one")
    if not 0.0 < percentile < 1.0:
        raise ValueError("percentile must lie strictly between zero and one")
    if warmup < 1:
        raise ValueError("warmup must be positive")
    return lam, percentile, warmup


def ewma_regime(
    changes: Sequence[float],
    lam: float,
    percentile: float,
    warmup: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return causal regime labels, current volatility and old-data cutoffs.

    Regime ``-1`` is calm/reversal and ``+1`` is volatile/momentum.  The first
    decision is zero because no move has yet been observed.  The first
    percentile switch occurs after ``warmup`` *previous* volatility estimates
    are available, which is decision index ``warmup + 1`` for a 30-day warmup.
    """

    changes = as_float_array(changes)
    lam, percentile, warmup = _normalise_ewma_config((lam, percentile, warmup))
    n = len(changes)
    vol = ewma_volatility(changes, lam)
    regimes = np.zeros(n, dtype=int)
    cutoffs = np.full(n, np.nan, dtype=float)

    for t in range(1, n):
        latest = changes[t - 1]
        # The warm-up state is intentionally simple reversal.
        if t > warmup and t - 1 >= warmup:
            prior_vol = vol[: t - 1]
            cutoff = float(np.quantile(prior_vol, percentile))
            cutoffs[t] = cutoff
            regimes[t] = 1 if vol[t - 1] >= cutoff else -1
        else:
            regimes[t] = -1
    return regimes, vol, cutoffs


def ewma_switch_positions(
    changes: Sequence[float],
    lam: float,
    percentile: float,
    warmup: int = 30,
    limit: int = LIMIT,
) -> np.ndarray:
    """Causal EWMA reversal/momentum switch with integral positions."""

    changes = as_float_array(changes)
    regimes, _, _ = ewma_regime(changes, lam, percentile, warmup)
    positions = np.zeros(len(changes), dtype=int)
    if len(changes) > 1:
        latest = np.sign(changes[:-1])
        positions[1:] = (limit * regimes[1:] * latest).astype(int)
    return positions


def ewma_ensemble_positions(
    changes: Sequence[float],
    configs: Iterable[Sequence[float]],
    default_warmup: int = 30,
    limit: int = LIMIT,
) -> np.ndarray:
    """Majority vote across causal EWMA switch configurations."""

    changes = as_float_array(changes)
    signals = []
    for config in configs:
        lam, percentile, warmup = _normalise_ewma_config(
            config, default_warmup=default_warmup
        )
        signals.append(
            ewma_switch_positions(changes, lam, percentile, warmup, limit=limit)
        )
    if not signals:
        return np.zeros(len(changes), dtype=int)
    return (limit * np.sign(np.sum(np.asarray(signals), axis=0))).astype(int)


def ewma_grid_positions(
    changes: Sequence[float],
    lambdas: Sequence[float],
    percentiles: Sequence[float],
    warmups: Sequence[int] = (30,),
    limit: int = LIMIT,
) -> Dict[Tuple[float, float, int], np.ndarray]:
    """Compute a small EWMA family while sharing each volatility path."""

    changes = as_float_array(changes)
    n = len(changes)
    result: Dict[Tuple[float, float, int], np.ndarray] = {}
    for lam in lambdas:
        lam = float(lam)
        vol = ewma_volatility(changes, lam)
        for warmup in warmups:
            warmup = int(warmup)
            position_matrix = np.zeros((len(percentiles), n), dtype=int)
            for t in range(1, n):
                latest_sign = int(np.sign(changes[t - 1]))
                if t > warmup and t - 1 >= warmup:
                    cutoffs = np.quantile(vol[: t - 1], percentiles)
                    states = np.where(vol[t - 1] >= cutoffs, 1, -1)
                else:
                    states = -np.ones(len(percentiles), dtype=int)
                position_matrix[:, t] = limit * states * latest_sign
            for q_index, percentile in enumerate(percentiles):
                result[(lam, float(percentile), warmup)] = position_matrix[q_index]
    return result


def delayed_ewma_switch_positions(
    changes: Sequence[float],
    lam: float,
    percentile: float,
    warmup: int = 30,
    delay: int = 1,
    limit: int = LIMIT,
) -> np.ndarray:
    """Hold the previous regime for one day whenever a regime changes."""

    if delay < 1:
        return ewma_switch_positions(changes, lam, percentile, warmup, limit)
    changes = as_float_array(changes)
    regimes, _, _ = ewma_regime(changes, lam, percentile, warmup)
    used = regimes.copy()
    for t in range(1, len(regimes)):
        if regimes[t] != regimes[t - 1] and regimes[t - 1] != 0:
            used[t] = regimes[t - 1]
    positions = np.zeros(len(changes), dtype=int)
    if len(changes) > 1:
        positions[1:] = (
            limit * used[1:] * np.sign(changes[:-1])
        ).astype(int)
    return positions


def _logsumexp(values: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - maximum)
    total = np.sum(shifted, axis=axis, keepdims=True)
    answer = maximum + np.log(np.maximum(total, EPS))
    if axis is not None:
        answer = np.squeeze(answer, axis=axis)
    return answer


def _log_normal_density(
    observations: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    variances = np.maximum(np.asarray(variances, dtype=float), 1e-8)
    return -0.5 * (
        np.log(2.0 * np.pi * variances)
        + (observations - means) ** 2 / variances
    )


@dataclass
class MSARFit:
    """Filtered two-state Gaussian Markov-switching AR(1) fit."""

    means: np.ndarray
    phi: np.ndarray
    variances: np.ndarray
    transition: np.ndarray
    initial_probs: np.ndarray
    loglik: float
    converged: bool
    iterations: int
    n_starts: int
    failed_starts: int

    @property
    def sigma(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variances, 0.0))

    @property
    def parameter_count(self) -> int:
        # Two means + two AR terms + two variances + two free transition
        # probabilities.  Initial probabilities are estimated but constrained.
        return 8

    def sorted_by_variance(self) -> "MSARFit":
        """Return a variance-labelled copy to remove state-label ambiguity."""

        order = np.argsort(self.variances)
        return MSARFit(
            means=self.means[order].copy(),
            phi=self.phi[order].copy(),
            variances=self.variances[order].copy(),
            transition=self.transition[np.ix_(order, order)].copy(),
            initial_probs=self.initial_probs[order].copy(),
            loglik=float(self.loglik),
            converged=bool(self.converged),
            iterations=int(self.iterations),
            n_starts=int(self.n_starts),
            failed_starts=int(self.failed_starts),
        )

    def as_dict(self) -> Dict[str, object]:
        """JSON-friendly compact summary."""

        return {
            "means": [float(x) for x in self.means],
            "phi": [float(x) for x in self.phi],
            "sigma": [float(x) for x in self.sigma],
            "variance": [float(x) for x in self.variances],
            "transition": [[float(x) for x in row] for row in self.transition],
            "initial_probs": [float(x) for x in self.initial_probs],
            "loglik": float(self.loglik),
            "converged": bool(self.converged),
            "iterations": int(self.iterations),
            "n_starts": int(self.n_starts),
            "failed_starts": int(self.failed_starts),
            "parameter_count": self.parameter_count,
        }


def _forward_backward(
    log_emission: np.ndarray,
    transition: np.ndarray,
    initial_probs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Forward-backward probabilities in log space for two states."""

    n_obs, n_states = log_emission.shape
    log_transition = np.log(np.maximum(transition, EPS))
    log_initial = np.log(np.maximum(initial_probs, EPS))
    alpha = np.empty((n_obs, n_states), dtype=float)
    alpha[0] = log_initial + log_emission[0]
    for i in range(1, n_obs):
        alpha[i] = log_emission[i] + _logsumexp(
            alpha[i - 1][:, None] + log_transition, axis=0
        )
    loglik = float(_logsumexp(alpha[-1], axis=0))

    beta = np.zeros((n_obs, n_states), dtype=float)
    for i in range(n_obs - 2, -1, -1):
        beta[i] = _logsumexp(
            log_transition
            + log_emission[i + 1][None, :]
            + beta[i + 1][None, :],
            axis=1,
        )

    gamma = np.exp(alpha + beta - loglik)
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), EPS)
    xi = np.empty((max(n_obs - 1, 0), n_states, n_states), dtype=float)
    for i in range(n_obs - 1):
        log_xi = (
            alpha[i][:, None]
            + log_transition
            + log_emission[i + 1][None, :]
            + beta[i + 1][None, :]
            - loglik
        )
        xi[i] = np.exp(log_xi)
        xi[i] /= np.maximum(xi[i].sum(), EPS)
    return gamma, xi, loglik


def _initial_parameters(
    x: np.ndarray, y: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic EM start from a binary partition."""

    labels = np.asarray(labels, dtype=int).copy()
    if np.all(labels == labels[0]):
        order = np.argsort(np.abs(y))
        labels = np.zeros(len(y), dtype=int)
        labels[order[len(y) // 2 :]] = 1
    means = np.zeros(2, dtype=float)
    phi = np.zeros(2, dtype=float)
    variances = np.zeros(2, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    global_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    floor = max(float(np.var(y)) * 1e-4, 1e-5)
    for state in range(2):
        mask = labels == state
        if mask.sum() < 3:
            mask = np.ones(len(y), dtype=bool)
        beta = np.linalg.lstsq(design[mask], y[mask], rcond=None)[0]
        means[state] = float(beta[0])
        phi[state] = float(np.clip(beta[1], -0.99, 0.99))
        residual = y[mask] - design[mask] @ beta
        variances[state] = max(float(np.mean(residual**2)), floor)
    transition_counts = np.ones((2, 2), dtype=float)
    for left, right in zip(labels[:-1], labels[1:]):
        transition_counts[left, right] += 1.0
    transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    initial = np.bincount(labels[: min(20, len(labels))], minlength=2).astype(float)
    initial = (initial + 1.0) / (initial.sum() + 2.0)
    # A small residual-based nudge makes the alternative starts deterministic
    # without pretending that the initial labels are observed states.
    if np.isfinite(global_residual).all() and np.std(global_residual) > 0:
        variances = np.maximum(variances, floor)
    return means, phi, variances, transition, initial


def _fit_one_msar_start(
    x: np.ndarray,
    y: np.ndarray,
    initial: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    max_iter: int,
    tol: float,
) -> MSARFit:
    means, phi, variances, transition, initial_probs = [
        np.asarray(value, dtype=float).copy() for value in initial
    ]
    n_obs = len(y)
    design = np.column_stack([np.ones(n_obs), x])
    converged = False
    previous_loglik = -np.inf
    final_loglik = -np.inf
    iterations = 0
    for iteration in range(1, max_iter + 1):
        emission_means = means[None, :] + x[:, None] * phi[None, :]
        log_emission = _log_normal_density(
            y[:, None], emission_means, variances[None, :]
        )
        gamma, xi, loglik = _forward_backward(
            log_emission, transition, initial_probs
        )
        for state in range(2):
            weights = np.maximum(gamma[:, state], 0.0)
            sqrt_weights = np.sqrt(weights)
            weighted_design = design * sqrt_weights[:, None]
            weighted_y = y * sqrt_weights
            beta = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
            means[state] = float(beta[0])
            phi[state] = float(np.clip(beta[1], -0.99, 0.99))
            residual = y - design @ beta
            variances[state] = max(
                float(np.sum(weights * residual**2) / max(weights.sum(), EPS)),
                1e-6,
            )
        if len(xi):
            transition = xi.sum(axis=0) + 1e-4
            transition /= transition.sum(axis=1, keepdims=True)
        initial_probs = gamma[0] + 1e-4
        initial_probs /= initial_probs.sum()

        # Evaluate the updated parameters before testing convergence.
        updated_means = means[None, :] + x[:, None] * phi[None, :]
        updated_emission = _log_normal_density(
            y[:, None], updated_means, variances[None, :]
        )
        _, _, updated_loglik = _forward_backward(
            updated_emission, transition, initial_probs
        )
        final_loglik = float(updated_loglik)
        iterations = iteration
        if np.isfinite(previous_loglik) and abs(final_loglik - previous_loglik) <= tol * (
            1.0 + abs(previous_loglik)
        ):
            converged = True
            break
        previous_loglik = final_loglik

    if not np.isfinite(final_loglik):
        raise FloatingPointError("non-finite MS-AR likelihood")
    return MSARFit(
        means=means,
        phi=phi,
        variances=variances,
        transition=transition,
        initial_probs=initial_probs,
        loglik=final_loglik,
        converged=converged,
        iterations=iterations,
        n_starts=1,
        failed_starts=0,
    ).sorted_by_variance()


def fit_msar_gaussian(
    changes: Sequence[float],
    n_starts: int = 4,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> MSARFit:
    """Fit a two-state Gaussian MS-AR(1) by deterministic EM starts."""

    changes = as_float_array(changes)
    if len(changes) < 25:
        raise ValueError("at least 25 changes are needed for the MS-AR fit")
    x = changes[:-1]
    y = changes[1:]
    design = np.column_stack([np.ones(len(x)), x])
    global_beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ global_beta
    magnitude_y = np.abs(y)
    magnitude_residual = np.abs(residual)
    patterns = [
        magnitude_y >= np.quantile(magnitude_y, 0.50),
        magnitude_residual >= np.quantile(magnitude_residual, 0.50),
        magnitude_y >= np.quantile(magnitude_y, 0.70),
        magnitude_residual >= np.quantile(magnitude_residual, 0.70),
    ]
    patterns = patterns[: max(1, int(n_starts))]
    fits: List[MSARFit] = []
    failed = 0
    for labels in patterns:
        try:
            start = _initial_parameters(x, y, labels.astype(int))
            fit = _fit_one_msar_start(x, y, start, max_iter, tol)
            fits.append(fit)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            failed += 1
    if not fits:
        raise RuntimeError("all deterministic MS-AR starts failed")
    best = max(fits, key=lambda fit: fit.loglik)
    best = best.sorted_by_variance()
    best.n_starts = len(patterns)
    best.failed_starts = failed
    return best


def _update_filtered_posterior(
    fit: MSARFit,
    posterior: np.ndarray,
    lag: float,
    observation: float,
) -> np.ndarray:
    """Advance one filtered state probability using one new observation."""

    prior = posterior @ fit.transition
    emission_means = fit.means + fit.phi * float(lag)
    log_weight = np.log(np.maximum(prior, EPS)) + _log_normal_density(
        np.asarray([float(observation)]), emission_means, fit.variances
    )[0]
    log_weight -= float(_logsumexp(log_weight, axis=0))
    updated = np.exp(log_weight)
    return updated / np.maximum(updated.sum(), EPS)


def filtered_posterior(
    fit: MSARFit, observed_changes: Sequence[float]
) -> np.ndarray:
    """Return the filtered state probabilities for the last observed change."""

    history = as_float_array(observed_changes)
    if len(history) < 2:
        return fit.initial_probs.copy()
    posterior = fit.initial_probs.copy()
    for lag, observation in zip(history[:-1], history[1:]):
        posterior = _update_filtered_posterior(fit, posterior, lag, observation)
    return posterior


def next_forecast_from_posterior(
    fit: MSARFit, posterior: Sequence[float], latest_change: float
) -> Tuple[float, np.ndarray]:
    """Forecast one change from a posterior for the latest observed change."""

    next_state_prob = np.asarray(posterior, dtype=float) @ fit.transition
    next_means = fit.means + fit.phi * float(latest_change)
    forecast = float(next_state_prob @ next_means)
    return forecast, next_state_prob


def filtered_next_forecast(
    fit: MSARFit, observed_changes: Sequence[float]
) -> Tuple[float, np.ndarray]:
    """Filter observed changes and forecast the next change causally."""

    history = as_float_array(observed_changes)
    if len(history) < 2:
        return float("nan"), np.full(2, np.nan)
    posterior = filtered_posterior(fit, history)
    return next_forecast_from_posterior(fit, posterior, history[-1])


def msar_positions(
    changes: Sequence[float],
    min_train: int = 60,
    refit_every: int = 20,
    limit: int = LIMIT,
    fallback: str = "reversal",
    fit_starts: int = 2,
    fit_max_iter: int = 80,
    return_fits: bool = False,
) -> object:
    """Causal expanding MS-AR forecasts with scheduled refits.

    At decision ``t`` the fit, if due, sees ``changes[:t]`` only, and the
    filtered probability used for the forecast is recomputed from that prefix.
    The returned fit records make convergence and parameter stability auditable.
    """

    changes = as_float_array(changes)
    if min_train < 25:
        raise ValueError("min_train must be at least 25")
    if refit_every < 1:
        raise ValueError("refit_every must be positive")
    positions = np.zeros(len(changes), dtype=int)
    fit_records: List[Dict[str, object]] = []
    current_fit: Optional[MSARFit] = None
    posterior: Optional[np.ndarray] = None
    last_observed_index = -1
    last_fit_decision = -10**9

    for t in range(1, len(changes)):
        if t >= min_train and (
            current_fit is None or t - last_fit_decision >= refit_every
        ):
            try:
                current_fit = fit_msar_gaussian(
                    changes[:t], n_starts=fit_starts, max_iter=fit_max_iter
                )
                last_fit_decision = t
                posterior = filtered_posterior(current_fit, changes[:t])
                last_observed_index = t - 1
                fit_records.append(
                    {
                        "decision": int(t),
                        "converged": bool(current_fit.converged),
                        "iterations": int(current_fit.iterations),
                        "failed_starts": int(current_fit.failed_starts),
                        "loglik": float(current_fit.loglik),
                        "means": [float(x) for x in current_fit.means],
                        "phi": [float(x) for x in current_fit.phi],
                        "sigma": [float(x) for x in current_fit.sigma],
                        "transition": [
                            [float(x) for x in row]
                            for row in current_fit.transition
                        ],
                    }
                )
            except (FloatingPointError, np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
                fit_records.append(
                    {
                        "decision": int(t),
                        "converged": False,
                        "error": type(exc).__name__,
                    }
                )
                current_fit = None
                posterior = None
        if current_fit is None:
            if fallback == "reversal" and t > 0:
                positions[t] = _direction_position(-changes[t - 1], limit)
            elif fallback == "momentum" and t > 0:
                positions[t] = _direction_position(changes[t - 1], limit)
            else:
                positions[t] = 0
            continue
        if posterior is None:
            forecast, _ = filtered_next_forecast(current_fit, changes[:t])
        else:
            # Between scheduled refits, process only the newly observed
            # change.  This preserves the filtered recursion but avoids
            # replaying the entire prefix at every decision.
            for observation_index in range(last_observed_index + 1, t):
                posterior = _update_filtered_posterior(
                    current_fit,
                    posterior,
                    changes[observation_index - 1],
                    changes[observation_index],
                )
            last_observed_index = t - 1
            forecast, _ = next_forecast_from_posterior(
                current_fit, posterior, changes[t - 1]
            )
        positions[t] = _direction_position(forecast, limit)

    if return_fits:
        return positions, fit_records
    return positions


def msar_state_detector_positions(
    changes: Sequence[float],
    min_train: int = 60,
    refit_every: int = 20,
    warmup_fallback: str = "reversal",
    limit: int = LIMIT,
    fit_starts: int = 2,
    fit_max_iter: int = 80,
    return_fits: bool = False,
) -> object:
    """HMM state detector using predeclared reversal/momentum directions."""

    changes = as_float_array(changes)
    positions = np.zeros(len(changes), dtype=int)
    fit_records: List[Dict[str, object]] = []
    current_fit: Optional[MSARFit] = None
    posterior: Optional[np.ndarray] = None
    last_observed_index = -1
    last_fit_decision = -10**9
    for t in range(1, len(changes)):
        if t >= min_train and (
            current_fit is None or t - last_fit_decision >= refit_every
        ):
            try:
                current_fit = fit_msar_gaussian(
                    changes[:t], n_starts=fit_starts, max_iter=fit_max_iter
                )
                last_fit_decision = t
                posterior = filtered_posterior(current_fit, changes[:t])
                last_observed_index = t - 1
                fit_records.append(
                    {
                        "decision": int(t),
                        "converged": bool(current_fit.converged),
                        "failed_starts": int(current_fit.failed_starts),
                        "loglik": float(current_fit.loglik),
                        "sigma": [float(x) for x in current_fit.sigma],
                    }
                )
            except (FloatingPointError, np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
                current_fit = None
                posterior = None
                fit_records.append(
                    {
                        "decision": int(t),
                        "converged": False,
                        "error": type(exc).__name__,
                    }
                )
        if current_fit is None:
            if warmup_fallback == "reversal":
                positions[t] = _direction_position(-changes[t - 1], limit)
            elif warmup_fallback == "momentum":
                positions[t] = _direction_position(changes[t - 1], limit)
            continue
        if posterior is None:
            _, next_state_prob = filtered_next_forecast(current_fit, changes[:t])
        else:
            for observation_index in range(last_observed_index + 1, t):
                posterior = _update_filtered_posterior(
                    current_fit,
                    posterior,
                    changes[observation_index - 1],
                    changes[observation_index],
                )
            last_observed_index = t - 1
            _, next_state_prob = next_forecast_from_posterior(
                current_fit, posterior, changes[t - 1]
            )
        high_state = int(np.argmax(current_fit.variances))
        state_direction = 1 if next_state_prob[high_state] >= 0.5 else -1
        positions[t] = _direction_position(state_direction * changes[t - 1], limit)
    if return_fits:
        return positions, fit_records
    return positions


__all__ = [
    "LIMIT",
    "MSARFit",
    "as_float_array",
    "constant_position",
    "delayed_ewma_switch_positions",
    "ewma_ensemble_positions",
    "ewma_grid_positions",
    "ewma_regime",
    "ewma_switch_positions",
    "ewma_volatility",
    "filtered_next_forecast",
    "fit_msar_gaussian",
    "msar_positions",
    "msar_state_detector_positions",
    "price_to_changes",
    "simple_momentum",
    "simple_reversal",
]
