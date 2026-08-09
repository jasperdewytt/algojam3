"""Small causal Fintech Token models for the cross-year study.

The module is deliberately self-contained and uses only NumPy.  A row in the
feature matrix is a decision after observing ``d[i]`` and before earning
``d[i + 1]``.  Every helper accepts one yearly sequence at a time; there is no
stateful object that can silently carry volatility or rolling history across a
year boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


LIMIT = 100
EPS = 1e-12
WARMUP = 30
EWMA_CONFIGS: Tuple[Tuple[float, float], ...] = (
    (0.85, 0.75),
    (0.90, 0.80),
    (0.95, 0.85),
)
BASE_LAMBDA = 0.90
FAST_LAMBDA = 0.85
SLOW_LAMBDA = 0.95
BASE_THRESHOLD = 0.80
SHORT_ROLL = 5
LONG_ROLL = 20
ALPHAS: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
CONFIDENCE_THRESHOLDS: Tuple[float, ...] = (0.0, 0.25, 0.50)
WEIGHT_CLIP_QUANTILES: Tuple[float, ...] = (0.75, 0.90)
TREE_MIN_LEAVES: Tuple[int, ...] = (60, 90)

FEATURE_NAMES: Tuple[str, ...] = (
    "latest_abs_change_over_causal_vol",
    "current_ewma_vol_percentile",
    "fast_slow_vol_ratio",
    "change_in_log_causal_vol",
    "frozen_ewma_ensemble_vote",
    "ewma_state_run_length",
    "rolling_5_continuation_vol_units",
    "rolling_20_continuation_vol_units",
    "rolling_20_positive_continuation_fraction",
    "causal_vol_threshold_gap",
    "ewma_member_agreement_count",
)


def as_float_array(values: Sequence[float]) -> np.ndarray:
    """Return a finite one-dimensional float array."""

    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("Fintech inputs must be finite")
    return array


def price_to_changes(prices: Sequence[float]) -> np.ndarray:
    """Convert a single yearly price path into day-to-day changes."""

    prices = as_float_array(prices)
    return np.diff(prices) if len(prices) >= 2 else np.empty(0, dtype=float)


def ewma_volatility(changes: Sequence[float], lam: float) -> np.ndarray:
    """Causal EWMA standard deviation aligned with observed changes."""

    changes = as_float_array(changes)
    lam = float(lam)
    if not 0.0 < lam < 1.0:
        raise ValueError("lambda must lie strictly between zero and one")
    volatility = np.zeros(len(changes), dtype=float)
    if len(changes) == 0:
        return volatility
    variance = float(changes[0] ** 2)
    volatility[0] = np.sqrt(max(variance, 0.0))
    for index in range(1, len(changes)):
        variance = lam * variance + (1.0 - lam) * float(changes[index] ** 2)
        volatility[index] = np.sqrt(max(variance, 0.0))
    return volatility


def _decision_count(changes: np.ndarray) -> int:
    return max(len(changes) - 1, 0)


def ewma_member_states(
    changes: Sequence[float],
    lam: float,
    percentile: float,
    warmup: int = WARMUP,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return causal calm/reversal (-1) or volatile/momentum (+1) states.

    State row ``i`` is the state used after observing ``d[i]``.  The cutoff is
    computed from ``volatility[:i]`` and therefore excludes the current
    volatility observation ``volatility[i]`` from its own threshold.
    """

    changes = as_float_array(changes)
    volatility = ewma_volatility(changes, lam)
    n_decisions = _decision_count(changes)
    states = -np.ones(n_decisions, dtype=int)
    cutoffs = np.full(n_decisions, np.nan, dtype=float)
    percentile = float(percentile)
    warmup = int(warmup)
    if not 0.0 < percentile < 1.0:
        raise ValueError("percentile must lie strictly between zero and one")
    if warmup < 1:
        raise ValueError("warmup must be positive")
    for index in range(n_decisions):
        if index < warmup:
            continue
        prior_volatility = volatility[:index]
        cutoff = float(np.quantile(prior_volatility, percentile))
        cutoffs[index] = cutoff
        states[index] = int(volatility[index] >= cutoff)
        if states[index] == 0:
            states[index] = -1
    return states, volatility, cutoffs


def frozen_ewma_diagnostics(
    changes: Sequence[float],
    configs: Iterable[Sequence[float]] = EWMA_CONFIGS,
    warmup: int = WARMUP,
) -> Dict[str, np.ndarray]:
    """Build the frozen benchmark state without using future observations."""

    changes = as_float_array(changes)
    normalised_configs = tuple((float(c[0]), float(c[1])) for c in configs)
    member_states: List[np.ndarray] = []
    member_volatility: List[np.ndarray] = []
    member_cutoffs: List[np.ndarray] = []
    for lam, percentile in normalised_configs:
        state, volatility, cutoff = ewma_member_states(
            changes, lam, percentile, warmup=warmup
        )
        member_states.append(state)
        member_volatility.append(volatility)
        member_cutoffs.append(cutoff)
    if member_states:
        state_matrix = np.asarray(member_states, dtype=int)
        vote = np.sum(state_matrix, axis=0).astype(int)
        ensemble_state = np.sign(vote).astype(int)
        ensemble_state[ensemble_state == 0] = -1
        agreement = np.abs(vote) / float(len(member_states))
    else:
        state_matrix = np.empty((0, _decision_count(changes)), dtype=int)
        vote = np.zeros(_decision_count(changes), dtype=int)
        ensemble_state = -np.ones(_decision_count(changes), dtype=int)
        agreement = np.zeros(_decision_count(changes), dtype=float)
    return {
        "member_states": state_matrix,
        "member_volatility": np.asarray(member_volatility, dtype=float),
        "member_cutoffs": np.asarray(member_cutoffs, dtype=float),
        "vote": vote,
        "ensemble_state": ensemble_state,
        "agreement": agreement,
    }


def simple_reversal_decisions(
    changes: Sequence[float], limit: int = LIMIT
) -> np.ndarray:
    """Decision rows after each observed change, earning the next change."""

    changes = as_float_array(changes)
    return (-int(limit) * np.sign(changes[:-1])).astype(int)


def simple_momentum_decisions(
    changes: Sequence[float], limit: int = LIMIT
) -> np.ndarray:
    changes = as_float_array(changes)
    return (int(limit) * np.sign(changes[:-1])).astype(int)


def frozen_ewma_decisions(
    changes: Sequence[float],
    configs: Iterable[Sequence[float]] = EWMA_CONFIGS,
    warmup: int = WARMUP,
    limit: int = LIMIT,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return the frozen ensemble position rows and its causal diagnostics."""

    changes = as_float_array(changes)
    diagnostics = frozen_ewma_diagnostics(configs=configs, changes=changes, warmup=warmup)
    latest_direction = np.sign(changes[:-1]).astype(int)
    decisions = (int(limit) * diagnostics["ensemble_state"] * latest_direction).astype(int)
    decisions[latest_direction == 0] = 0
    return decisions, diagnostics


def _rolling_mean(values: np.ndarray, end: int, width: int) -> float:
    """Mean of values indexed by completed continuation day through ``end``."""

    if end < 1:
        return 0.0
    start = max(1, end - width + 1)
    window = values[start : end + 1]
    return float(np.mean(window)) if len(window) else 0.0


def _rolling_positive_fraction(values: np.ndarray, end: int, width: int) -> float:
    if end < 1:
        return 0.5
    start = max(1, end - width + 1)
    window = values[start : end + 1]
    return float(np.mean(window > 0.0)) if len(window) else 0.5


def _state_run_length(states: np.ndarray, index: int) -> int:
    if index < 0 or len(states) == 0:
        return 0
    current = int(states[index])
    run = 1
    cursor = index - 1
    while cursor >= 0 and int(states[cursor]) == current:
        run += 1
        cursor -= 1
    return run


def build_causal_features(
    changes: Sequence[float],
    warmup: int = WARMUP,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build the deliberately small, direction-invariant causal feature set.

    Invalid pre-warmup rows are represented by NaN and are not used for model
    fitting or model predictions.  At a valid row ``i`` every feature uses only
    ``changes[:i+1]``.  The returned metadata is used by the notebook's feature
    audit and documents each feature's final observable timestamp.
    """

    changes = as_float_array(changes)
    n_decisions = _decision_count(changes)
    frozen = frozen_ewma_diagnostics(changes, warmup=warmup)
    base_volatility = ewma_volatility(changes, BASE_LAMBDA)
    fast_volatility = ewma_volatility(changes, FAST_LAMBDA)
    slow_volatility = ewma_volatility(changes, SLOW_LAMBDA)

    # c[j] is available after d[j] is observed and uses only d[j-1], d[j].
    continuation = np.full(len(changes), np.nan, dtype=float)
    if len(changes) >= 2:
        continuation[1:] = np.sign(changes[:-1]) * changes[1:]
    continuation_vol_units = np.full(len(changes), np.nan, dtype=float)
    if len(changes) >= 2:
        continuation_vol_units[1:] = continuation[1:] / np.maximum(
            base_volatility[1:], EPS
        )

    features = np.full((n_decisions, len(FEATURE_NAMES)), np.nan, dtype=float)
    for index in range(n_decisions):
        # The model does not use rows before the frozen 30-change fallback.
        if index < warmup:
            continue
        prior_volatility = base_volatility[:index]
        threshold = float(np.quantile(prior_volatility, BASE_THRESHOLD))
        current_volatility = float(base_volatility[index])
        fast = float(fast_volatility[index])
        slow = float(slow_volatility[index])
        current_state = int(frozen["ensemble_state"][index])
        vote = float(frozen["vote"][index]) / len(EWMA_CONFIGS)
        valid_continuations = continuation_vol_units[1 : index + 1]
        last_short = valid_continuations[-SHORT_ROLL:]
        last_long = valid_continuations[-LONG_ROLL:]
        if len(last_short) == 0:
            short_mean = 0.0
        else:
            short_mean = float(np.mean(last_short))
        if len(last_long) == 0:
            long_mean = 0.0
            positive_fraction = 0.5
        else:
            long_mean = float(np.mean(last_long))
            positive_fraction = float(np.mean(last_long > 0.0))
        previous_volatility = max(float(base_volatility[index - 1]), EPS)
        features[index] = np.asarray(
            [
                abs(float(changes[index])) / max(current_volatility, EPS),
                float(np.mean(prior_volatility <= current_volatility)),
                fast / max(slow, EPS),
                float(np.log(max(current_volatility, EPS) / previous_volatility)),
                vote,
                float(_state_run_length(frozen["ensemble_state"], index)),
                short_mean,
                long_mean,
                positive_fraction,
                (current_volatility - threshold) / max(threshold, EPS),
                float(frozen["agreement"][index]),
            ],
            dtype=float,
        )

    audit = {
        "feature_names": list(FEATURE_NAMES),
        "feature_final_observation": {
            "latest_abs_change_over_causal_vol": "d[i] and causal EWMA volatility through d[i]",
            "current_ewma_vol_percentile": "d[i] and prior EWMA volatilities through d[i-1]",
            "fast_slow_vol_ratio": "d[i] and causal fast/slow EWMA volatilities through d[i]",
            "change_in_log_causal_vol": "d[i-1], d[i]",
            "frozen_ewma_ensemble_vote": "d[0:i+1]",
            "ewma_state_run_length": "d[0:i+1]",
            "rolling_5_continuation_vol_units": "c[j]=sign(d[j-1])*d[j] for j<=i, j in the last 5 completed outcomes",
            "rolling_20_continuation_vol_units": "c[j]=sign(d[j-1])*d[j] for j<=i, j in the last 20 completed outcomes",
            "rolling_20_positive_continuation_fraction": "sign(d[j-1])*d[j] for j<=i, last 20 completed outcomes",
            "causal_vol_threshold_gap": "d[i] and prior EWMA volatilities through d[i-1]",
            "ewma_member_agreement_count": "d[0:i+1]",
        },
        "target_final_observation": {
            "raw_target": "z[i+1]=sign(d[i])*d[i+1] (training label only)",
            "normalised_target": "z[i+1]/max(base causal EWMA volatility through d[i], EPS)",
        },
        "warmup": int(warmup),
        "base_lambda": BASE_LAMBDA,
        "base_threshold": BASE_THRESHOLD,
        "rolling_windows": [SHORT_ROLL, LONG_ROLL],
    }
    return features, audit


def continuation_targets(
    changes: Sequence[float],
    warmup: int = WARMUP,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw and causal-volatility-normalised continuation targets."""

    changes = as_float_array(changes)
    if len(changes) < 2:
        empty = np.empty(0, dtype=float)
        return empty, empty, empty
    raw = np.sign(changes[:-1]) * changes[1:]
    volatility = ewma_volatility(changes, BASE_LAMBDA)[:-1]
    normalised = raw / np.maximum(volatility, EPS)
    row_mask = np.arange(len(raw)) >= int(warmup)
    return raw, normalised, row_mask


@dataclass
class FeatureStandardizer:
    """Training-only feature transformation."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureStandardizer":
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or len(features) == 0:
            raise ValueError("standardizer requires a non-empty feature matrix")
        if not np.isfinite(features).all():
            raise ValueError("standardizer cannot fit non-finite features")
        mean = np.mean(features, axis=0)
        scale = np.std(features, axis=0, ddof=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if not np.isfinite(features).all():
            raise ValueError("cannot transform non-finite features")
        return (features - self.mean) / self.scale


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    out = np.empty_like(values)
    positive = values >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _logistic_objective(
    design: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    beta: np.ndarray,
    alpha: float,
) -> float:
    eta = design @ beta
    value = np.sum(weights * (np.logaddexp(0.0, eta) - labels * eta))
    value += 0.5 * float(alpha) * float(np.sum(beta[1:] ** 2))
    return float(value)


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    max_iter: int = 200,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    design = np.column_stack([np.ones(len(features)), features])
    beta = np.zeros(design.shape[1], dtype=float)
    regulariser = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    regulariser[1:, 1:] = np.eye(design.shape[1] - 1) * float(alpha)
    converged = False
    objective = _logistic_objective(design, labels, weights, beta, alpha)
    for iteration in range(1, max_iter + 1):
        probabilities = _sigmoid(design @ beta)
        gradient = design.T @ (weights * (probabilities - labels))
        gradient[1:] += float(alpha) * beta[1:]
        curvature = weights * probabilities * (1.0 - probabilities)
        hessian = design.T @ (curvature[:, None] * design) + regulariser
        hessian += np.eye(hessian.shape[0]) * 1e-9
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        if not np.isfinite(step).all():
            break
        step_scale = 1.0
        accepted = False
        for _ in range(20):
            candidate = beta - step_scale * step
            candidate_objective = _logistic_objective(
                design, labels, weights, candidate, alpha
            )
            if candidate_objective <= objective + 1e-10:
                beta = candidate
                objective = candidate_objective
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
        if np.max(np.abs(step_scale * step)) <= 1e-8:
            converged = True
            break
    return beta, {
        "iterations": int(iteration),
        "converged": bool(converged),
        "objective": float(objective),
    }


def _fit_regression_stump(
    features: np.ndarray,
    target: np.ndarray,
    min_leaf: int,
) -> Dict[str, Any]:
    """Fit a deterministic depth-one regression tree with large leaves."""

    n_rows, n_features = features.shape
    min_leaf = int(min_leaf)
    if n_rows < 2 * min_leaf:
        return {
            "feature": -1,
            "threshold": np.nan,
            "left_mean": float(np.mean(target)),
            "right_mean": float(np.mean(target)),
            "root_mean": float(np.mean(target)),
            "sse": float(np.sum((target - np.mean(target)) ** 2)),
        }
    total_sum = float(np.sum(target))
    total_sq = float(np.sum(target**2))
    best: Tuple[float, int, float, float, float] | None = None
    for feature_index in range(n_features):
        order = np.argsort(features[:, feature_index], kind="mergesort")
        values = features[order, feature_index]
        ordered_target = target[order]
        cumulative_sum = np.cumsum(ordered_target)
        cumulative_sq = np.cumsum(ordered_target**2)
        for split in range(min_leaf, n_rows - min_leaf + 1):
            if split >= n_rows or values[split - 1] >= values[split]:
                continue
            left_n = split
            right_n = n_rows - split
            left_sum = float(cumulative_sum[split - 1])
            right_sum = total_sum - left_sum
            left_sq = float(cumulative_sq[split - 1])
            right_sq = total_sq - left_sq
            sse = (
                left_sq - left_sum**2 / left_n
                + right_sq - right_sum**2 / right_n
            )
            threshold = 0.5 * (float(values[split - 1]) + float(values[split]))
            left_mean = left_sum / left_n
            right_mean = right_sum / right_n
            candidate = (float(sse), feature_index, threshold, left_mean, right_mean)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        root_mean = float(np.mean(target))
        return {
            "feature": -1,
            "threshold": np.nan,
            "left_mean": root_mean,
            "right_mean": root_mean,
            "root_mean": root_mean,
            "sse": float(np.sum((target - root_mean) ** 2)),
        }
    sse, feature_index, threshold, left_mean, right_mean = best
    return {
        "feature": int(feature_index),
        "threshold": float(threshold),
        "left_mean": float(left_mean),
        "right_mean": float(right_mean),
        "root_mean": float(np.mean(target)),
        "sse": float(sse),
    }


@dataclass
class FittedModel:
    """A fitted model plus its training-only preprocessing."""

    config: Dict[str, Any]
    standardizer: FeatureStandardizer
    family: str
    parameters: Dict[str, Any]
    feature_names: Tuple[str, ...] = FEATURE_NAMES
    fit_info: Dict[str, Any] | None = None

    def transformed(self, features: np.ndarray) -> np.ndarray:
        return self.standardizer.transform(features)

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        if self.family not in {"logistic_equal", "logistic_weighted"}:
            raise ValueError("probabilities are available only for logistic models")
        transformed = self.transformed(features)
        design = np.column_stack([np.ones(len(transformed)), transformed])
        return _sigmoid(design @ np.asarray(self.parameters["beta"], dtype=float))

    def predict_score(self, features: np.ndarray) -> np.ndarray:
        transformed = self.transformed(features)
        if self.family in {"logistic_equal", "logistic_weighted"}:
            probabilities = self.predict_probability(features)
            return 2.0 * probabilities - 1.0
        if self.family == "ridge":
            design = np.column_stack([np.ones(len(transformed)), transformed])
            beta = np.asarray(self.parameters["beta"], dtype=float)
            return design @ beta
        if self.family == "tree_stump":
            tree = self.parameters["tree"]
            feature_index = int(tree["feature"])
            if feature_index < 0:
                return np.full(len(transformed), float(tree["root_mean"]))
            return np.where(
                transformed[:, feature_index] <= float(tree["threshold"]),
                float(tree["left_mean"]),
                float(tree["right_mean"]),
            )
        raise ValueError(f"unknown model family: {self.family}")

    def confidence(self, features: np.ndarray) -> np.ndarray:
        if self.family in {"logistic_equal", "logistic_weighted"}:
            return np.abs(self.predict_score(features))
        # Regression confidence is measured in causal-volatility target units.
        # The cap makes the predeclared 0.25/0.50 grid comparable across folds.
        return np.minimum(np.abs(self.predict_score(features)), 1.0)

    def coefficient_vector(self) -> np.ndarray | None:
        if self.family in {"ridge", "logistic_equal", "logistic_weighted"}:
            return np.asarray(self.parameters["beta"], dtype=float).copy()
        return None

    def serialisable_summary(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "family": self.family,
            "config": dict(self.config),
            "feature_names": list(self.feature_names),
            "standardizer_mean": [float(x) for x in self.standardizer.mean],
            "standardizer_scale": [float(x) for x in self.standardizer.scale],
            "fit_info": dict(self.fit_info or {}),
        }
        coefficients = self.coefficient_vector()
        if coefficients is not None:
            result["coefficients_intercept_then_standardised_features"] = [
                float(x) for x in coefficients
            ]
        if self.family == "tree_stump":
            tree = self.parameters["tree"]
            result["tree"] = {
                "feature_index": int(tree["feature"]),
                "feature_name": (
                    None
                    if int(tree["feature"]) < 0
                    else self.feature_names[int(tree["feature"])]
                ),
                "threshold_standardised": float(tree["threshold"]),
                "left_mean": float(tree["left_mean"]),
                "right_mean": float(tree["right_mean"]),
            }
        return result


def _normalise_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(config)
    result["family"] = str(result["family"])
    result["confidence_threshold"] = float(result.get("confidence_threshold", 0.0))
    if result["family"] in {"ridge", "logistic_equal", "logistic_weighted"}:
        result["alpha"] = float(result["alpha"])
    if result["family"] == "logistic_weighted":
        result["weight_clip_quantile"] = float(result["weight_clip_quantile"])
    if result["family"] == "tree_stump":
        result["min_leaf"] = int(result["min_leaf"])
    return result


def fit_model(
    features: np.ndarray,
    normalised_target: np.ndarray,
    raw_target: np.ndarray,
    config: Mapping[str, Any],
) -> FittedModel:
    """Fit one declared candidate using training rows only."""

    features = np.asarray(features, dtype=float)
    normalised_target = np.asarray(normalised_target, dtype=float).reshape(-1)
    raw_target = np.asarray(raw_target, dtype=float).reshape(-1)
    if len(features) != len(normalised_target) or len(features) != len(raw_target):
        raise ValueError("features and targets must have equal rows")
    if len(features) == 0 or not np.isfinite(features).all():
        raise ValueError("model fit requires finite feature rows")
    if not np.isfinite(normalised_target).all() or not np.isfinite(raw_target).all():
        raise ValueError("model fit requires finite targets")
    config = _normalise_config(config)
    family = config["family"]
    standardizer = FeatureStandardizer.fit(features)
    transformed = standardizer.transform(features)
    fit_info: Dict[str, Any] = {"n_rows": int(len(features))}

    if family == "ridge":
        design = np.column_stack([np.ones(len(transformed)), transformed])
        penalty = np.zeros((design.shape[1], design.shape[1]), dtype=float)
        penalty[1:, 1:] = np.eye(design.shape[1] - 1) * config["alpha"]
        system = design.T @ design + penalty + np.eye(design.shape[1]) * 1e-10
        rhs = design.T @ normalised_target
        try:
            beta = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(system) @ rhs
        fit_info["target"] = "normalised_continuation"
        return FittedModel(
            config=config,
            standardizer=standardizer,
            family=family,
            parameters={"beta": beta},
            fit_info=fit_info,
        )

    if family in {"logistic_equal", "logistic_weighted"}:
        labels = (raw_target > 0.0).astype(float)
        weights = np.ones(len(labels), dtype=float)
        if family == "logistic_weighted":
            clip_quantile = config["weight_clip_quantile"]
            clip_value = float(np.quantile(np.abs(raw_target), clip_quantile))
            clip_value = max(clip_value, EPS)
            weights = np.minimum(np.abs(raw_target), clip_value)
            weights /= max(float(np.mean(weights)), EPS)
            fit_info["weight_clip_quantile"] = float(clip_quantile)
            fit_info["weight_clip_value_raw_target"] = float(clip_value)
        beta, logistic_info = _fit_logistic(
            transformed,
            labels,
            weights,
            alpha=config["alpha"],
        )
        fit_info.update(logistic_info)
        fit_info["target"] = "continuation_positive_probability"
        fit_info["weighting"] = "equal" if family == "logistic_equal" else "clipped_abs_raw_target"
        return FittedModel(
            config=config,
            standardizer=standardizer,
            family=family,
            parameters={"beta": beta},
            fit_info=fit_info,
        )

    if family == "tree_stump":
        tree = _fit_regression_stump(
            transformed, normalised_target, min_leaf=config["min_leaf"]
        )
        fit_info["target"] = "normalised_continuation"
        fit_info["min_leaf"] = int(config["min_leaf"])
        return FittedModel(
            config=config,
            standardizer=standardizer,
            family=family,
            parameters={"tree": tree},
            fit_info=fit_info,
        )

    raise ValueError(f"unknown candidate family: {family}")


def candidate_configs() -> List[Dict[str, Any]]:
    """Return the complete predeclared candidate grid.

    The confidence threshold is a confidence filter, not a different logistic
    probability cut: logistic decisions always use p>0.5 for momentum and
    p<=0.5 for reversal, with low-confidence predictions falling back to
    reversal.  All thresholds are selected only inside the two training years.
    """

    candidates: List[Dict[str, Any]] = []
    for alpha in ALPHAS:
        for threshold in CONFIDENCE_THRESHOLDS:
            candidates.append(
                {
                    "family": "ridge",
                    "alpha": alpha,
                    "confidence_threshold": threshold,
                }
            )
    for alpha in ALPHAS:
        for threshold in CONFIDENCE_THRESHOLDS:
            candidates.append(
                {
                    "family": "logistic_equal",
                    "alpha": alpha,
                    "confidence_threshold": threshold,
                }
            )
    for alpha in ALPHAS:
        for clip_quantile in WEIGHT_CLIP_QUANTILES:
            for threshold in CONFIDENCE_THRESHOLDS:
                candidates.append(
                    {
                        "family": "logistic_weighted",
                        "alpha": alpha,
                        "weight_clip_quantile": clip_quantile,
                        "confidence_threshold": threshold,
                    }
                )
    for min_leaf in TREE_MIN_LEAVES:
        for threshold in CONFIDENCE_THRESHOLDS:
            candidates.append(
                {
                    "family": "tree_stump",
                    "min_leaf": min_leaf,
                    "confidence_threshold": threshold,
                }
            )
    return candidates


def config_id(config: Mapping[str, Any]) -> str:
    config = _normalise_config(config)
    fields = [f"family={config['family']}"]
    for key in ("alpha", "weight_clip_quantile", "min_leaf", "confidence_threshold"):
        if key in config:
            value = config[key]
            if isinstance(value, float):
                fields.append(f"{key}={value:g}")
            else:
                fields.append(f"{key}={value}")
    return "|".join(fields)


def complexity_score(config: Mapping[str, Any]) -> float:
    config = _normalise_config(config)
    family_base = {
        "ridge": 1.0,
        "logistic_equal": 2.0,
        "logistic_weighted": 3.0,
        "tree_stump": 4.0,
    }
    score = family_base[config["family"]]
    if config.get("confidence_threshold", 0.0) > 0.0:
        score += 0.25
    if config["family"] == "logistic_weighted":
        score += 0.25
    if config["family"] == "tree_stump" and config.get("min_leaf", 90) == 60:
        score += 0.05
    return score


__all__ = [
    "ALPHAS",
    "BASE_LAMBDA",
    "BASE_THRESHOLD",
    "CONFIDENCE_THRESHOLDS",
    "EWMA_CONFIGS",
    "FEATURE_NAMES",
    "FittedModel",
    "LIMIT",
    "LONG_ROLL",
    "SHORT_ROLL",
    "TREE_MIN_LEAVES",
    "WARMUP",
    "WEIGHT_CLIP_QUANTILES",
    "as_float_array",
    "build_causal_features",
    "candidate_configs",
    "complexity_score",
    "config_id",
    "continuation_targets",
    "ewma_member_states",
    "ewma_volatility",
    "fit_model",
    "frozen_ewma_decisions",
    "frozen_ewma_diagnostics",
    "price_to_changes",
    "simple_momentum_decisions",
    "simple_reversal_decisions",
]
