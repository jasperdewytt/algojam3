"""Cross-year validation and causal robustness helpers for Fintech Token ML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ml_models import (
    EWMA_CONFIGS,
    FEATURE_NAMES,
    LIMIT,
    WARMUP,
    FittedModel,
    build_causal_features,
    candidate_configs,
    complexity_score,
    config_id,
    continuation_targets,
    fit_model,
    frozen_ewma_decisions,
    simple_reversal_decisions,
)


@dataclass
class YearData:
    """A single reset yearly sequence and its causal feature rows."""

    name: str
    path: str
    prices: np.ndarray
    changes: np.ndarray
    features: np.ndarray
    raw_target: np.ndarray
    normalised_target: np.ndarray
    valid_rows: np.ndarray
    feature_audit: Dict[str, Any]

    @property
    def n_prices(self) -> int:
        return int(len(self.prices))

    @property
    def n_changes(self) -> int:
        return int(len(self.changes))

    @property
    def n_decisions(self) -> int:
        return max(self.n_changes - 1, 0)


def load_year(path: str | Path, name: str, warmup: int = WARMUP) -> YearData:
    """Load and reset one CSV; no row from another year is consulted."""

    path = Path(path)
    frame = pd.read_csv(path)
    if "Price" not in frame.columns:
        raise ValueError(f"{path} has no Price column")
    prices = frame["Price"].to_numpy(dtype=float)
    if len(prices) != 365:
        raise ValueError(f"{path} has {len(prices)} prices, expected 365")
    if not np.isfinite(prices).all():
        raise ValueError(f"{path} contains non-finite prices")
    changes = np.diff(prices)
    features, feature_audit = build_causal_features(changes, warmup=warmup)
    raw_target, normalised_target, target_rows = continuation_targets(
        changes, warmup=warmup
    )
    valid_rows = (
        target_rows
        & np.isfinite(features).all(axis=1)
        & np.isfinite(raw_target)
        & np.isfinite(normalised_target)
    )
    return YearData(
        name=str(name),
        path=str(path),
        prices=prices,
        changes=changes,
        features=features,
        raw_target=raw_target,
        normalised_target=normalised_target,
        valid_rows=valid_rows,
        feature_audit=feature_audit,
    )


def training_matrix(
    years: Mapping[str, YearData], names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate rows from named years after each year's own reset."""

    if not names:
        raise ValueError("at least one training year is required")
    matrices = [years[name] for name in names]
    features = np.vstack([year.features[year.valid_rows] for year in matrices])
    normalised = np.concatenate(
        [year.normalised_target[year.valid_rows] for year in matrices]
    )
    raw = np.concatenate([year.raw_target[year.valid_rows] for year in matrices])
    row_year = np.concatenate(
        [np.full(int(year.valid_rows.sum()), name, dtype=object) for name, year in zip(names, matrices)]
    )
    return features, normalised, raw, row_year


def fit_on_years(
    years: Mapping[str, YearData], names: Sequence[str], config: Mapping[str, Any]
) -> FittedModel:
    features, normalised, raw, _ = training_matrix(years, names)
    return fit_model(features, normalised, raw, config)


def _full_positions(decision_positions: Sequence[int], n_changes: int) -> np.ndarray:
    full = np.zeros(int(n_changes), dtype=int)
    decision_positions = np.asarray(decision_positions, dtype=int)
    if len(full) >= 2:
        if len(decision_positions) != len(full) - 1:
            raise ValueError("decision position length does not match change path")
        full[1:] = decision_positions
    return full


def model_decisions_from_arrays(
    model: FittedModel,
    changes: Sequence[float],
    features: np.ndarray,
    warmup: int = WARMUP,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Apply a frozen model to causal features from one year."""

    changes = np.asarray(changes, dtype=float).reshape(-1)
    features = np.asarray(features, dtype=float)
    n_decisions = max(len(changes) - 1, 0)
    if features.shape != (n_decisions, len(FEATURE_NAMES)):
        raise ValueError("feature shape does not match changes")
    decisions = np.zeros(n_decisions, dtype=int)
    scores = np.full(n_decisions, np.nan, dtype=float)
    probabilities = np.full(n_decisions, np.nan, dtype=float)
    confidence = np.full(n_decisions, np.nan, dtype=float)
    available = np.zeros(n_decisions, dtype=bool)
    confidence_fallback = np.zeros(n_decisions, dtype=bool)
    latest_direction = np.sign(changes[:-1]).astype(int)

    for index in range(n_decisions):
        latest = float(changes[index])
        if not np.isfinite(latest) or latest == 0.0:
            decisions[index] = 0
            continue
        if index < int(warmup):
            decisions[index] = int(-LIMIT * np.sign(latest))
            continue
        if not np.isfinite(features[index]).all():
            # Invalid model inputs are flat after warm-up; no direction is invented.
            decisions[index] = 0
            continue
        row = features[index : index + 1]
        score = float(model.predict_score(row)[0])
        conf = float(model.confidence(row)[0])
        scores[index] = score
        confidence[index] = conf
        available[index] = True
        if model.family in {"logistic_equal", "logistic_weighted"}:
            probabilities[index] = float(model.predict_probability(row)[0])
        if conf < float(model.config.get("confidence_threshold", 0.0)):
            decisions[index] = int(-LIMIT * np.sign(latest))
            confidence_fallback[index] = True
            continue
        # Positive continuation prediction means momentum.  Zero is the
        # predeclared reversal fallback, so there is no accidental flat signal.
        direction = 1 if score > 0.0 else -1
        decisions[index] = int(LIMIT * np.sign(latest) * direction)

    return decisions, {
        "scores": scores,
        "probabilities": probabilities,
        "confidence": confidence,
        "available": available,
        "confidence_fallback": confidence_fallback,
        "latest_direction": latest_direction,
    }


def model_decisions(
    model: FittedModel, year: YearData, warmup: int = WARMUP
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    return model_decisions_from_arrays(model, year.changes, year.features, warmup=warmup)


def _max_drawdown(pnl: np.ndarray) -> float:
    cumulative = np.cumsum(np.asarray(pnl, dtype=float))
    curve = np.concatenate([[0.0], cumulative])
    drawdown = curve - np.maximum.accumulate(curve)
    return float(np.min(drawdown))


def _quarter_values(series: np.ndarray) -> List[float]:
    return [float(np.sum(series[indexes])) for indexes in np.array_split(np.arange(len(series)), 4)]


def _delayed_positions(full_positions: np.ndarray) -> np.ndarray:
    delayed = np.zeros_like(full_positions)
    if len(full_positions) >= 3:
        delayed[2:] = full_positions[1:-1]
    return delayed


def _episode_runs(states: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    index = 0
    while index < len(states):
        if int(states[index]) != 1:
            index += 1
            continue
        start = index
        while index + 1 < len(states) and int(states[index + 1]) == 1:
            index += 1
        runs.append((start, index))
        index += 1
    return runs


def prediction_diagnostics(
    year: YearData,
    prediction_details: Mapping[str, np.ndarray],
    warmup: int = WARMUP,
) -> Dict[str, Any]:
    available = np.asarray(prediction_details["available"], dtype=bool)
    score = np.asarray(prediction_details["scores"], dtype=float)
    probabilities = np.asarray(prediction_details["probabilities"], dtype=float)
    row_mask = available & (np.arange(year.n_decisions) >= int(warmup))
    labels = year.raw_target > 0.0
    predicted_positive = score > 0.0
    actual = labels[row_mask]
    predicted = predicted_positive[row_mask]
    tp = int(np.sum(predicted & actual))
    tn = int(np.sum((~predicted) & (~actual)))
    fp = int(np.sum(predicted & (~actual)))
    fn = int(np.sum((~predicted) & actual))
    result: Dict[str, Any] = {
        "rows": int(np.sum(row_mask)),
        "actual_positive": int(np.sum(actual)),
        "predicted_positive": int(np.sum(predicted)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sign_accuracy": float((tp + tn) / max(np.sum(row_mask), 1)),
        "score_mean": float(np.nanmean(score[row_mask])) if np.any(row_mask) else np.nan,
        "score_std": float(np.nanstd(score[row_mask])) if np.any(row_mask) else np.nan,
        "score_target_correlation": np.nan,
        "brier_score": np.nan,
        "calibration_bins": [],
        "model_rows_with_confidence_fallback": int(
            np.sum(np.asarray(prediction_details["confidence_fallback"], dtype=bool)[row_mask])
        ),
    }
    if np.sum(row_mask) >= 2:
        score_values = score[row_mask]
        target_values = year.normalised_target[row_mask]
        if np.std(score_values) > 0.0 and np.std(target_values) > 0.0:
            result["score_target_correlation"] = float(np.corrcoef(score_values, target_values)[0, 1])
    if np.any(np.isfinite(probabilities[row_mask])):
        probability_values = probabilities[row_mask]
        brier = np.mean((probability_values - actual.astype(float)) ** 2)
        result["brier_score"] = float(brier)
        bins = np.linspace(0.0, 1.0, 6)
        calibration: List[Dict[str, Any]] = []
        for left, right in zip(bins[:-1], bins[1:]):
            if right == 1.0:
                mask = (probability_values >= left) & (probability_values <= right)
            else:
                mask = (probability_values >= left) & (probability_values < right)
            calibration.append(
                {
                    "left": float(left),
                    "right": float(right),
                    "count": int(np.sum(mask)),
                    "mean_probability": float(np.mean(probability_values[mask])) if np.any(mask) else np.nan,
                    "observed_positive_rate": float(np.mean(actual[mask])) if np.any(mask) else np.nan,
                }
            )
        result["calibration_bins"] = calibration
    return result


def evaluate_strategy(
    year: YearData,
    decision_positions: Sequence[int],
    ewma_decisions: np.ndarray | None = None,
    ewma_diagnostics: Mapping[str, np.ndarray] | None = None,
    reversal_decisions: np.ndarray | None = None,
    prediction_details: Mapping[str, np.ndarray] | None = None,
    name: str = "candidate",
) -> Dict[str, Any]:
    """Calculate absolute, paired, state-conditioned and robustness metrics."""

    decisions = np.asarray(decision_positions, dtype=int)
    if len(decisions) != year.n_decisions:
        raise ValueError("decision positions do not match yearly sequence")
    if ewma_decisions is None:
        ewma_decisions, ewma_diagnostics_local = frozen_ewma_decisions(year.changes)
        ewma_diagnostics = ewma_diagnostics_local
    if reversal_decisions is None:
        reversal_decisions = simple_reversal_decisions(year.changes)
    ewma_decisions = np.asarray(ewma_decisions, dtype=int)
    reversal_decisions = np.asarray(reversal_decisions, dtype=int)
    ewma_full = _full_positions(ewma_decisions, year.n_changes)
    reversal_full = _full_positions(reversal_decisions, year.n_changes)
    candidate_full = _full_positions(decisions, year.n_changes)
    candidate_pnl = np.round(candidate_full * year.changes, 2)
    ewma_pnl = np.round(ewma_full * year.changes, 2)
    reversal_pnl = np.round(reversal_full * year.changes, 2)
    inc_ewma = candidate_pnl - ewma_pnl
    inc_reversal = candidate_pnl - reversal_pnl
    state = np.asarray((ewma_diagnostics or {})["ensemble_state"], dtype=int)

    active = decisions != 0
    signed_latest = decisions * year.changes[:-1]
    momentum_mask = signed_latest > 0
    reversal_mask = signed_latest < 0
    volatile_mask = state == 1
    calm_mask = state == -1
    quarter_indexes = np.array_split(np.arange(year.n_changes), 4)
    delayed_candidate = _delayed_positions(candidate_full)
    delayed_ewma = _delayed_positions(ewma_full)
    delayed_reversal = _delayed_positions(reversal_full)
    delayed_candidate_pnl = np.round(delayed_candidate * year.changes, 2)
    delayed_ewma_pnl = np.round(delayed_ewma * year.changes, 2)
    delayed_reversal_pnl = np.round(delayed_reversal * year.changes, 2)

    largest_moves: Dict[str, Dict[str, float]] = {}
    if year.n_changes > 1:
        ordered = np.argsort(np.abs(year.changes[1:]))[::-1] + 1
        for count in (1, 3, 5, 10):
            removed = set(int(x) for x in ordered[: min(count, len(ordered))])
            keep = np.asarray([index not in removed for index in range(year.n_changes)])
            largest_moves[str(count)] = {
                "candidate_pnl": float(np.sum(candidate_pnl[keep])),
                "ewma_pnl": float(np.sum(ewma_pnl[keep])),
                "incremental_vs_ewma": float(np.sum(inc_ewma[keep])),
                "reversal_pnl": float(np.sum(reversal_pnl[keep])),
            }

    episodes: List[Dict[str, Any]] = []
    for start, end in _episode_runs(state):
        episode_delta = inc_ewma[start + 1 : end + 2]
        episodes.append(
            {
                "decision_start": int(start),
                "decision_end": int(end),
                "realised_change_start": int(start + 1),
                "realised_change_end": int(end + 1),
                "days": int(end - start + 1),
                "incremental_vs_ewma": float(np.sum(episode_delta)),
            }
        )
    episode_contributions = [float(item["incremental_vs_ewma"]) for item in episodes]
    largest_episode = max(episode_contributions) if episode_contributions else 0.0

    metrics: Dict[str, Any] = {
        "year": year.name,
        "strategy": name,
        "pnl": float(np.sum(candidate_pnl)),
        "reversal_pnl": float(np.sum(reversal_pnl)),
        "ewma_pnl": float(np.sum(ewma_pnl)),
        "incremental_vs_reversal": float(np.sum(inc_reversal)),
        "incremental_vs_ewma": float(np.sum(inc_ewma)),
        "quarter_pnl": _quarter_values(candidate_pnl),
        "quarter_reversal_pnl": _quarter_values(reversal_pnl),
        "quarter_ewma_pnl": _quarter_values(ewma_pnl),
        "quarter_incremental_vs_ewma": _quarter_values(inc_ewma),
        "max_drawdown": _max_drawdown(candidate_pnl),
        "ewma_max_drawdown": _max_drawdown(ewma_pnl),
        "reversal_max_drawdown": _max_drawdown(reversal_pnl),
        "hit_rate": float(np.sum(candidate_pnl[1:][active] > 0.0) / max(np.sum(active), 1)),
        "active_days": int(np.sum(active)),
        "turnover_units": int(np.sum(np.abs(np.diff(candidate_full)))),
        "position_changes": int(np.sum(np.diff(candidate_full) != 0)),
        # Full position index j is held at price P[j] while earning d[j].
        # There are n_changes such tradable price slots; P[n_prices-1] has no
        # following local return and is therefore not a decision slot.
        "max_capital": float(np.max(np.abs(candidate_full * year.prices[:-1]))),
        "momentum_decisions": int(np.sum(momentum_mask)),
        "reversal_decisions": int(np.sum(reversal_mask)),
        "flat_decisions": int(np.sum(~active)),
        "ewma_volatile_pnl": float(np.sum(candidate_pnl[1:][volatile_mask])),
        "ewma_calm_pnl": float(np.sum(candidate_pnl[1:][calm_mask])),
        "ewma_volatile_incremental_vs_ewma": float(np.sum(inc_ewma[1:][volatile_mask])),
        "ewma_calm_incremental_vs_ewma": float(np.sum(inc_ewma[1:][calm_mask])),
        "largest_move_exclusions": largest_moves,
        "one_day_delayed_pnl": float(np.sum(delayed_candidate_pnl)),
        "one_day_delayed_ewma_pnl": float(np.sum(delayed_ewma_pnl)),
        "one_day_delayed_reversal_pnl": float(np.sum(delayed_reversal_pnl)),
        "one_day_delayed_incremental_vs_ewma": float(
            np.sum(delayed_candidate_pnl - delayed_ewma_pnl)
        ),
        "one_day_delayed_incremental_vs_reversal": float(
            np.sum(delayed_candidate_pnl - delayed_reversal_pnl)
        ),
        "volatile_episode_count": int(len(episodes)),
        "volatile_episodes": episodes,
        "largest_episode_incremental_vs_ewma": float(largest_episode),
        "incremental_excluding_largest_volatile_episode": float(
            np.sum(inc_ewma) - largest_episode
        ),
    }
    if prediction_details is not None:
        metrics["prediction_diagnostics"] = prediction_diagnostics(
            year, prediction_details
        )
    return {
        "metrics": metrics,
        "candidate_full_positions": candidate_full,
        "candidate_pnl": candidate_pnl,
        "ewma_full_positions": ewma_full,
        "ewma_pnl": ewma_pnl,
        "reversal_full_positions": reversal_full,
        "reversal_pnl": reversal_pnl,
        "incremental_vs_ewma_series": inc_ewma,
        "incremental_vs_reversal_series": inc_reversal,
        "ewma_state": state,
        "prediction_details": prediction_details,
    }


def evaluate_inner_candidate(
    years: Mapping[str, YearData],
    train_names: Sequence[str],
    config: Mapping[str, Any],
    warmup: int = WARMUP,
) -> List[Dict[str, Any]]:
    """Validate a candidate in both chronological directions inside two years."""

    if len(train_names) != 2:
        raise ValueError("exactly two years are required for the inner validation")
    rows: List[Dict[str, Any]] = []
    for fit_name, validation_name in ((train_names[0], train_names[1]), (train_names[1], train_names[0])):
        model = fit_on_years(years, [fit_name], config)
        decisions, details = model_decisions(model, years[validation_name], warmup=warmup)
        ewma_decisions, ewma_diag = frozen_ewma_decisions(years[validation_name].changes)
        evaluation = evaluate_strategy(
            years[validation_name],
            decisions,
            ewma_decisions=ewma_decisions,
            ewma_diagnostics=ewma_diag,
            prediction_details=details,
            name=config_id(config),
        )
        metrics = evaluation["metrics"]
        rows.append(
            {
                "fit_year": fit_name,
                "validation_year": validation_name,
                "pnl": float(metrics["pnl"]),
                "reversal_pnl": float(metrics["reversal_pnl"]),
                "ewma_pnl": float(metrics["ewma_pnl"]),
                "incremental_vs_reversal": float(metrics["incremental_vs_reversal"]),
                "incremental_vs_ewma": float(metrics["incremental_vs_ewma"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "active_days": int(metrics["active_days"]),
                "hit_rate": float(metrics["hit_rate"]),
                "one_day_delayed_incremental_vs_ewma": float(
                    metrics["one_day_delayed_incremental_vs_ewma"]
                ),
            }
        )
    return rows


def select_candidate(
    years: Mapping[str, YearData],
    train_names: Sequence[str],
    candidates: Sequence[Mapping[str, Any]] | None = None,
    warmup: int = WARMUP,
    close_tie_aud: float = 1000.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Select using worst inner-year EWMA increment, complexity, drawdown."""

    candidates = list(candidates or candidate_configs())
    accounting: List[Dict[str, Any]] = []
    detailed_inner: List[Dict[str, Any]] = []
    for candidate in candidates:
        candidate = dict(candidate)
        row: Dict[str, Any] = {
            "candidate_id": config_id(candidate),
            "family": candidate["family"],
            "complexity": float(complexity_score(candidate)),
            **candidate,
            "status": "ok",
        }
        try:
            inner_rows = evaluate_inner_candidate(years, train_names, candidate, warmup=warmup)
            worst_inc_ewma = min(x["incremental_vs_ewma"] for x in inner_rows)
            mean_inc_ewma = float(np.mean([x["incremental_vs_ewma"] for x in inner_rows]))
            worst_inc_reversal = min(x["incremental_vs_reversal"] for x in inner_rows)
            mean_drawdown_depth = float(np.mean([-x["max_drawdown"] for x in inner_rows]))
            row.update(
                {
                    "worst_inner_incremental_vs_ewma": float(worst_inc_ewma),
                    "mean_inner_incremental_vs_ewma": mean_inc_ewma,
                    "worst_inner_incremental_vs_reversal": float(worst_inc_reversal),
                    "mean_inner_drawdown_depth": mean_drawdown_depth,
                    "inner_delayed_incremental_vs_ewma": float(
                        np.mean([x["one_day_delayed_incremental_vs_ewma"] for x in inner_rows])
                    ),
                }
            )
            detailed_inner.append(
                {
                    "candidate_id": config_id(candidate),
                    "config": candidate,
                    "inner_rows": inner_rows,
                }
            )
        except Exception as error:  # keep an auditable record of every attempted candidate
            row.update(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "worst_inner_incremental_vs_ewma": -np.inf,
                    "mean_inner_incremental_vs_ewma": -np.inf,
                    "worst_inner_incremental_vs_reversal": -np.inf,
                    "mean_inner_drawdown_depth": np.inf,
                    "inner_delayed_incremental_vs_ewma": -np.inf,
                }
            )
        accounting.append(row)
    successful = [row for row in accounting if row["status"] == "ok"]
    if not successful:
        raise RuntimeError("all candidate configurations failed in inner validation")
    best_worst = max(row["worst_inner_incremental_vs_ewma"] for row in successful)
    eligible = [
        row
        for row in successful
        if best_worst - row["worst_inner_incremental_vs_ewma"] <= float(close_tie_aud)
    ]
    selected_row = min(
        eligible,
        key=lambda row: (
            float(row["complexity"]),
            float(row["mean_inner_drawdown_depth"]),
            -float(row["mean_inner_incremental_vs_ewma"]),
            str(row["candidate_id"]),
        ),
    )
    for row in accounting:
        row["best_worst_inner_incremental_vs_ewma"] = float(best_worst)
        row["within_close_tie_set"] = bool(row in eligible)
        row["selected"] = bool(row["candidate_id"] == selected_row["candidate_id"])
    selected_config = {
        key: value
        for key, value in selected_row.items()
        if key
        in {
            "family",
            "alpha",
            "weight_clip_quantile",
            "min_leaf",
            "confidence_threshold",
        }
    }
    return selected_config, accounting, detailed_inner


def fit_and_evaluate_outer(
    years: Mapping[str, YearData],
    train_names: Sequence[str],
    test_name: str,
    config: Mapping[str, Any],
    warmup: int = WARMUP,
) -> Tuple[FittedModel, Dict[str, Any]]:
    """Refit once on the two training years and evaluate the untouched year."""

    model = fit_on_years(years, train_names, config)
    test_decisions, details = model_decisions(model, years[test_name], warmup=warmup)
    ewma_decisions, ewma_diag = frozen_ewma_decisions(years[test_name].changes)
    evaluation = evaluate_strategy(
        years[test_name],
        test_decisions,
        ewma_decisions=ewma_decisions,
        ewma_diagnostics=ewma_diag,
        prediction_details=details,
        name=config_id(config),
    )
    return model, evaluation


def moving_block_bootstrap(
    incremental_series: Sequence[float],
    block_lengths: Sequence[int] = (5, 10, 20, 40),
    repetitions: int = 1000,
    seed: int = 20260808,
) -> List[Dict[str, Any]]:
    """Paired moving-block bootstrap of a frozen candidate-minus-EWMA series."""

    incremental_series = np.asarray(incremental_series, dtype=float).reshape(-1)
    if len(incremental_series) == 0:
        return []
    results: List[Dict[str, Any]] = []
    for offset, block_length in enumerate(block_lengths):
        block_length = int(block_length)
        if block_length < 1:
            raise ValueError("block lengths must be positive")
        rng = np.random.default_rng(int(seed) + offset)
        samples = np.empty(int(repetitions), dtype=float)
        max_start = max(len(incremental_series) - block_length, 0)
        for repeat in range(int(repetitions)):
            values: List[float] = []
            while len(values) < len(incremental_series):
                start = int(rng.integers(0, max_start + 1))
                values.extend(incremental_series[start : start + block_length].tolist())
            samples[repeat] = float(np.sum(values[: len(incremental_series)]))
        results.append(
            {
                "block_length": block_length,
                "repetitions": int(repetitions),
                "observed_incremental": float(np.sum(incremental_series)),
                "bootstrap_mean": float(np.mean(samples)),
                "probability_positive": float(np.mean(samples > 0.0)),
                "p025": float(np.quantile(samples, 0.025)),
                "p50": float(np.quantile(samples, 0.50)),
                "p975": float(np.quantile(samples, 0.975)),
            }
        )
    return results


def future_perturbation_audit(
    model: FittedModel,
    year: YearData,
    cuts: Sequence[int] = (60, 90, 120, 180, 240, 300),
    warmup: int = WARMUP,
) -> List[Dict[str, Any]]:
    """Verify future observations cannot alter earlier model positions."""

    base_decisions, _ = model_decisions(model, year, warmup=warmup)
    rows: List[Dict[str, Any]] = []
    for cut in cuts:
        cut = int(cut)
        if cut < 0 or cut >= len(year.changes) - 1:
            raise ValueError("future perturbation cut must leave a future suffix")
        perturbed_changes = year.changes.copy()
        perturbed_changes[cut + 1 :] = perturbed_changes[cut + 1 :] * -1.37 + 0.123
        perturbed_features, _ = build_causal_features(perturbed_changes, warmup=warmup)
        perturbed_decisions, _ = model_decisions_from_arrays(
            model, perturbed_changes, perturbed_features, warmup=warmup
        )
        unchanged = bool(np.array_equal(base_decisions[: cut + 1], perturbed_decisions[: cut + 1]))
        rows.append(
            {
                "cut_observed_change_index": cut,
                "positions_unchanged_through_cut": unchanged,
                "max_prefix_position_difference": int(
                    np.max(np.abs(base_decisions[: cut + 1] - perturbed_decisions[: cut + 1]))
                ),
            }
        )
    return rows


def prefix_feature_audit(
    year: YearData,
    cuts: Sequence[int] = (30, 60, 120, 240),
    warmup: int = WARMUP,
) -> List[Dict[str, Any]]:
    """Compare full-year features to features recomputed from truncated prefixes."""

    rows: List[Dict[str, Any]] = []
    for cut in cuts:
        cut = int(cut)
        if cut >= year.n_changes - 1:
            continue
        prefix_features, _ = build_causal_features(year.changes[: cut + 2], warmup=warmup)
        full_prefix = year.features[: cut + 1]
        prefix_rows = prefix_features[: cut + 1]
        same = bool(np.allclose(full_prefix, prefix_rows, equal_nan=True, atol=0.0, rtol=0.0))
        rows.append(
            {
                "last_compared_decision_index": cut,
                "features_unchanged": same,
                "max_absolute_difference": float(
                    np.nanmax(np.abs(full_prefix - prefix_rows))
                    if np.any(np.isfinite(full_prefix))
                    else 0.0
                ),
            }
        )
    return rows


__all__ = [
    "YearData",
    "evaluate_inner_candidate",
    "evaluate_strategy",
    "fit_and_evaluate_outer",
    "fit_on_years",
    "future_perturbation_audit",
    "load_year",
    "model_decisions",
    "model_decisions_from_arrays",
    "moving_block_bootstrap",
    "prefix_feature_audit",
    "prediction_diagnostics",
    "select_candidate",
    "training_matrix",
]
