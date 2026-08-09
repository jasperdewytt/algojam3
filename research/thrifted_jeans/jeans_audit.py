"""Causal Round 1 audit for the Thrifted Jeans strategies.

This module is deliberately self-contained.  It reads the supplied Jeans CSV
only as research data; it does not import or modify the production algorithm
or simulator.  All position arrays are desired integer positions indexed by
decision day.  The simulator convention is represented by

    pnl[t] = position[t - 1] * (price[t] - price[t - 1])

for t >= 1, with no local P&L on day zero.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "trader_interface" / "data" / "Thrifted Jeans_price_history.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_FIGURE_DIR = Path(__file__).resolve().parent / "figures"

LIMIT = 800
JEANS_BUDGET = 600_000.0
START_PRICE = 40.0
SEED = 20260809

K2 = (1.0, 0.05, 9.0)
HYBRID = (0.5, 0.05, 5.0)


def load_price_series(path: Path = DATA_PATH) -> np.ndarray:
    """Load only the visible Round 1 price history."""

    frame = pd.read_csv(path)
    if "Price" not in frame.columns:
        raise ValueError(f"Expected Price column in {path}")
    price = frame["Price"].astype(float).to_numpy()
    if len(price) != 365:
        raise ValueError(f"Expected 365 observations, got {len(price)}")
    if not np.isclose(price[0], START_PRICE):
        raise ValueError(f"Unexpected starting price {price[0]}")
    return price


def realized_pnl(price: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Return simulator-aligned daily P&L for a desired-position path."""

    price = np.asarray(price, dtype=float)
    position = np.asarray(position, dtype=int)
    if len(price) != len(position):
        raise ValueError("price and position lengths differ")
    out = np.zeros(len(price), dtype=float)
    if len(price) > 1:
        out[1:] = np.round(position[:-1] * np.diff(price), 2)
    return out


def _max_drawdown(pnl: np.ndarray) -> float:
    equity = np.cumsum(pnl)
    drawdown = equity - np.maximum.accumulate(np.r_[0.0, equity[:-1]])
    return float(np.min(drawdown)) if len(drawdown) else 0.0


def _longest_loss_streak(pnl: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in pnl:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _top_day_share(pnl: np.ndarray, total: float, k: int) -> float:
    if total == 0:
        return float("nan")
    return float(np.sort(np.asarray(pnl))[-k:].sum() / total)


def position_metrics(
    price: np.ndarray,
    position: np.ndarray,
    baseline_pnl: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute attribution and exposure metrics for one candidate."""

    price = np.asarray(price, dtype=float)
    position = np.asarray(position, dtype=int)
    pnl = realized_pnl(price, position)
    active_realized = position[:-1] != 0 if len(position) > 1 else np.array([], dtype=bool)
    active_pnl = pnl[1:][active_realized]
    total = float(np.sum(pnl))
    daily_std = float(np.std(pnl[1:], ddof=1)) if len(pnl) > 2 else 0.0
    sharpe = float(np.mean(pnl[1:]) / daily_std * math.sqrt(365.0)) if daily_std > 0 else 0.0
    long_mask = np.r_[False, position[:-1] > 0]
    short_mask = np.r_[False, position[:-1] < 0]
    flat_mask = np.r_[False, position[:-1] == 0]
    long_pnl = float(np.sum(pnl[long_mask]))
    short_pnl = float(np.sum(pnl[short_mask]))
    turnover = int(np.sum(np.abs(np.diff(np.r_[0, position]))))
    best = float(np.max(pnl)) if len(pnl) else 0.0
    worst = float(np.min(pnl)) if len(pnl) else 0.0
    row: dict[str, float | int] = {
        "P&L": total,
        "Active days": int(np.sum(position != 0)),
        "Long days": int(np.sum(position > 0)),
        "Short days": int(np.sum(position < 0)),
        "Flat days": int(np.sum(position == 0)),
        "Turnover": turnover,
        "Hit rate active": float(np.mean(active_pnl > 0)) if len(active_pnl) else float("nan"),
        "Sharpe annualized": sharpe,
        "Max drawdown": _max_drawdown(pnl),
        "Maximum Jeans exposure AUD": float(np.max(np.abs(position) * price)),
        "Longest losing streak": _longest_loss_streak(pnl),
        "Long P&L": long_pnl,
        "Short P&L": short_pnl,
        "Flat P&L": float(np.sum(pnl[flat_mask])),
        "Best day": best,
        "Worst day": worst,
        "Best 1 day share": _top_day_share(pnl, total, 1),
        "Best 5 day share": _top_day_share(pnl, total, 5),
        "Best 10 day share": _top_day_share(pnl, total, 10),
        "Best 20 day share": _top_day_share(pnl, total, 20),
    }
    if baseline_pnl is not None:
        row["Incremental vs always long"] = total - float(np.sum(baseline_pnl))
    return row


def _kalman_trace(
    price: np.ndarray,
    parameters: tuple[float, float, float],
    initial_slope_variance: float = 1.0,
) -> dict[str, np.ndarray]:
    """Causal local-linear Kalman trace, matching the supplied implementation."""

    price = np.asarray(price, dtype=float)
    q_level, q_slope, observation_variance = [float(x) for x in parameters]
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    process = np.diag([q_level, q_slope])
    observation = np.array([1.0, 0.0])
    state = np.array([price[0], 0.0], dtype=float)
    covariance = np.diag([observation_variance, initial_slope_variance]).astype(float)
    slopes = np.zeros(len(price), dtype=float)
    uncertainty = np.zeros(len(price), dtype=float)
    for day, observed in enumerate(price):
        if day > 0:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
        innovation = observed - float(observation @ state)
        innovation_variance = float(observation @ covariance @ observation) + observation_variance
        gain = covariance @ observation / max(innovation_variance, 1e-12)
        state = state + gain * innovation
        covariance = covariance - np.outer(gain, observation @ covariance)
        covariance = (covariance + covariance.T) / 2.0
        slopes[day] = state[1]
        uncertainty[day] = math.sqrt(max(float(covariance[1, 1]), 0.0))
    z = slopes / np.maximum(uncertainty, 1e-12)
    return {"slope": slopes, "uncertainty": uncertainty, "z": z}


def _position_from_slope(
    trace: dict[str, np.ndarray],
    mapping: str = "uncertainty_short",
    confidence: float = 1.0,
    limit: int = LIMIT,
) -> np.ndarray:
    slope = trace["slope"]
    uncertainty = trace["uncertainty"]
    if mapping == "uncertainty_short":
        signal = np.where(slope < -confidence * uncertainty, -1, 1)
    elif mapping == "symmetric":
        signal = np.sign(slope)
    elif mapping == "long_flat":
        signal = (slope > 0).astype(int)
    else:
        raise ValueError(mapping)
    position = (limit * signal).astype(int)
    position[0] = 0
    return position


def simple_kalman_position(
    price: np.ndarray,
    parameters: tuple[float, float, float] = K2,
    confidence: float = 1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    trace = _kalman_trace(price, parameters)
    return _position_from_slope(trace, "uncertainty_short", confidence), trace


def kalman_trend_only_position(
    price: np.ndarray,
    parameters: tuple[float, float, float] = K2,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    trace = _kalman_trace(price, parameters)
    return _position_from_slope(trace, "symmetric"), trace


def _original_ema_trace(price: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    """The original, non-causal residual calculation, evaluated prefix by prefix."""

    price = np.asarray(price, dtype=float)
    ema = np.zeros(len(price), dtype=float)
    z = np.zeros(len(price), dtype=float)
    fair = price[0]
    for day, observed in enumerate(price):
        if day > 0:
            fair += alpha * (observed - fair)
        ema[day] = fair
        residuals = price[: day + 1] - fair
        scale = float(np.std(residuals))
        z[day] = (observed - fair) / max(scale, 1e-12)
    return {"ema": ema, "deviation": price - ema, "z": z}


def _causal_ema_trace(
    price: np.ndarray,
    alpha: float,
    standardize_through_current: bool = False,
    minimum_history: int = 5,
) -> dict[str, np.ndarray]:
    """Causal EMA deviation trace.

    The decision at t compares price[t] with EMA[t-1].  The preferred
    standardization uses deviations available before t.  The alternate
    through-current variant includes today's deviation after it is formed;
    this is still causal but is reported separately.
    """

    price = np.asarray(price, dtype=float)
    ema = np.zeros(len(price), dtype=float)
    deviation = np.full(len(price), np.nan, dtype=float)
    z = np.full(len(price), np.nan, dtype=float)
    fair = price[0]
    ema[0] = fair
    historical: list[float] = []
    for day in range(1, len(price)):
        current_deviation = price[day] - fair
        deviation[day] = current_deviation
        previous_scale = (
            float(np.std(historical, ddof=1))
            if len(historical) >= minimum_history
            else float("nan")
        )
        fair = fair + alpha * current_deviation
        ema[day] = fair
        historical.append(float(current_deviation))
        current_scale = (
            float(np.std(historical, ddof=1))
            if len(historical) >= minimum_history
            else float("nan")
        )
        scale = current_scale if standardize_through_current else previous_scale
        if np.isfinite(scale) and scale > 1e-12:
            z[day] = current_deviation / scale
    return {"ema": ema, "deviation": deviation, "z": z}


def _mean_reversion_transition(
    current_position: int,
    z_score: float,
    threshold: float,
    limit: int = LIMIT,
) -> int:
    """Exact entry/holding/reversal mechanics from algorithm_juan."""

    if not np.isfinite(z_score):
        return 0 if current_position == 0 else current_position
    if current_position == 0:
        if z_score >= threshold:
            return -limit
        if z_score <= -threshold:
            return limit
        return 0
    if current_position * z_score > 0:
        if abs(z_score) >= threshold:
            return -current_position
        return 0
    return current_position


def original_ema_mean_reversion_position(
    price: np.ndarray,
    alpha: float = 0.06,
    revert_threshold: float = 0.2,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    trace = _original_ema_trace(price, alpha)
    position = np.zeros(len(price), dtype=int)
    for day in range(1, len(price)):
        previous = int(position[day - 1])
        position[day] = _mean_reversion_transition(
            previous, float(trace["z"][day]), revert_threshold
        )
    return position, trace


def corrected_ema_mean_reversion_position(
    price: np.ndarray,
    alpha: float = 0.06,
    revert_threshold: float = 0.2,
    standardize_through_current: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    trace = _causal_ema_trace(
        price,
        alpha,
        standardize_through_current=standardize_through_current,
    )
    position = np.zeros(len(price), dtype=int)
    for day in range(1, len(price)):
        position[day] = _mean_reversion_transition(
            int(position[day - 1]), float(trace["z"][day]), revert_threshold
        )
    return position, trace


def original_hybrid_position(
    price: np.ndarray,
    q_level: float = 0.5,
    q_slope: float = 0.05,
    observation_variance: float = 5.0,
    initial_slope_variance: float = 1.0,
    short_confidence: float = 0.5,
    alpha: float = 0.06,
    trend_threshold: float = 0.6,
    revert_threshold: float = 0.2,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Exact replay of the original hybrid in trader_interface/algorithm_juan."""

    kalman = _kalman_trace(
        price,
        (q_level, q_slope, observation_variance),
        initial_slope_variance=initial_slope_variance,
    )
    ema = _original_ema_trace(price, alpha)
    position = np.zeros(len(price), dtype=int)
    strong = np.abs(kalman["z"]) >= trend_threshold
    for day in range(1, len(price)):
        if strong[day]:
            position[day] = (
                -LIMIT
                if kalman["slope"][day] < -short_confidence * kalman["uncertainty"][day]
                else LIMIT
            )
        else:
            position[day] = _mean_reversion_transition(
                int(position[day - 1]), float(ema["z"][day]), revert_threshold
            )
    trace = {
        **kalman,
        "ema": ema["ema"],
        "deviation": ema["deviation"],
        "ema_z": ema["z"],
        "strong_trend": strong.astype(bool),
    }
    return position, trace


def corrected_hybrid_position(
    price: np.ndarray,
    q_level: float = 0.5,
    q_slope: float = 0.05,
    observation_variance: float = 5.0,
    initial_slope_variance: float = 1.0,
    short_confidence: float = 0.5,
    alpha: float = 0.06,
    trend_threshold: float = 0.6,
    revert_threshold: float = 0.2,
    standardize_through_current: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Hybrid with a causal EMA[t-1] deviation and causal volatility."""

    kalman = _kalman_trace(
        price,
        (q_level, q_slope, observation_variance),
        initial_slope_variance=initial_slope_variance,
    )
    ema = _causal_ema_trace(
        price,
        alpha,
        standardize_through_current=standardize_through_current,
    )
    position = np.zeros(len(price), dtype=int)
    strong = np.abs(kalman["z"]) >= trend_threshold
    for day in range(1, len(price)):
        if strong[day]:
            position[day] = (
                -LIMIT
                if kalman["slope"][day] < -short_confidence * kalman["uncertainty"][day]
                else LIMIT
            )
        else:
            position[day] = _mean_reversion_transition(
                int(position[day - 1]), float(ema["z"][day]), revert_threshold
            )
    trace = {
        **kalman,
        "ema": ema["ema"],
        "deviation": ema["deviation"],
        "ema_z": ema["z"],
        "strong_trend": strong.astype(bool),
    }
    return position, trace


def momentum_position(
    price: np.ndarray,
    window: int = 20,
    long_flat: bool = False,
) -> np.ndarray:
    price = np.asarray(price, dtype=float)
    position = np.zeros(len(price), dtype=int)
    for day in range(window, len(price)):
        move = price[day] - price[day - window]
        if long_flat:
            position[day] = LIMIT if move > 0 else 0
        else:
            position[day] = LIMIT * int(np.sign(move))
    return position


def candidate_positions(price: np.ndarray) -> dict[str, np.ndarray]:
    """Return the small predeclared candidate set used in the main audit."""

    positions: dict[str, np.ndarray] = {
        "always_long": np.full(len(price), LIMIT, dtype=int),
        "flat": np.zeros(len(price), dtype=int),
    }
    positions["simple_kalman_K2"] = simple_kalman_position(price)[0]
    positions["kalman_trend_only_K2"] = kalman_trend_only_position(price)[0]
    positions["original_ema_mean_reversion"] = original_ema_mean_reversion_position(price)[0]
    positions["corrected_ema_mean_reversion_prevstd"] = corrected_ema_mean_reversion_position(price)[0]
    positions["corrected_ema_mean_reversion_throughstd"] = corrected_ema_mean_reversion_position(
        price, standardize_through_current=True
    )[0]
    positions["original_hybrid"] = original_hybrid_position(price)[0]
    positions["corrected_hybrid_prevstd"] = corrected_hybrid_position(price)[0]
    positions["corrected_hybrid_throughstd"] = corrected_hybrid_position(
        price, standardize_through_current=True
    )[0]
    positions["momentum_20_symmetric"] = momentum_position(price, window=20)
    positions["momentum_20_long_flat"] = momentum_position(price, window=20, long_flat=True)
    return positions


def candidate_traces(price: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Return the diagnostics needed for slope/EMA/regime analysis."""

    simple, simple_trace = simple_kalman_position(price)
    trend, trend_trace = kalman_trend_only_position(price)
    original_ema, original_ema_trace = original_ema_mean_reversion_position(price)
    corrected_ema, corrected_ema_trace = corrected_ema_mean_reversion_position(price)
    original_hybrid, original_hybrid_trace = original_hybrid_position(price)
    corrected_hybrid, corrected_hybrid_trace = corrected_hybrid_position(price)
    del simple, trend, original_ema, corrected_ema, original_hybrid, corrected_hybrid
    return {
        "simple_kalman_K2": simple_trace,
        "kalman_trend_only_K2": trend_trace,
        "original_ema_mean_reversion": original_ema_trace,
        "corrected_ema_mean_reversion_prevstd": corrected_ema_trace,
        "original_hybrid": original_hybrid_trace,
        "corrected_hybrid_prevstd": corrected_hybrid_trace,
    }


def _segment_definitions(n: int) -> list[tuple[str, str, int, int]]:
    segments: list[tuple[str, str, int, int]] = []
    for i, start in enumerate(range(0, n, 60), 1):
        segments.append(("60_day", f"B{i}", start, min(start + 60, n)))
    quarters = [(0, 91), (91, 182), (182, 273), (273, n)]
    for i, (start, end) in enumerate(quarters, 1):
        segments.append(("quarter", f"Q{i}", start, end))
    halves = [(0, 182), (182, n)]
    for i, (start, end) in enumerate(halves, 1):
        segments.append(("half", f"H{i}", start, end))
    thirds = [(0, 120), (120, 243), (243, n)]
    for i, (start, end) in enumerate(thirds, 1):
        segments.append(("early_middle_late", f"T{i}", start, end))
    return segments


def chronological_table(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
    baseline_pnl: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, position in positions.items():
        pnl = realized_pnl(price, position)
        for kind, label, start, end in _segment_definitions(len(price)):
            rows.append(
                {
                    "Candidate": name,
                    "Split": kind,
                    "Segment": label,
                    "Start day": start,
                    "End day exclusive": end,
                    "P&L": float(np.sum(pnl[start:end])),
                    "Incremental vs always long": float(
                        np.sum(pnl[start:end]) - np.sum(baseline_pnl[start:end])
                    ),
                    "Long P&L": float(
                        np.sum(
                            pnl[start:end]
                            * (np.r_[False, position[:-1]][start:end] > 0)
                        )
                    ),
                    "Short P&L": float(
                        np.sum(
                            pnl[start:end]
                            * (np.r_[False, position[:-1]][start:end] < 0)
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _state_runs(state: np.ndarray) -> list[tuple[object, int]]:
    if len(state) == 0:
        return []
    runs: list[tuple[object, int]] = []
    start = 0
    for day in range(1, len(state)):
        if state[day] != state[start]:
            runs.append((state[start], day - start))
            start = day
    runs.append((state[start], len(state) - start))
    return runs


def regime_attribution_table(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
    traces: dict[str, dict[str, np.ndarray]],
    baseline_pnl: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in ("original_hybrid", "corrected_hybrid_prevstd"):
        trace = traces[name]
        state = np.where(trace["strong_trend"], "strong_trend", "weak_mean_reversion")
        position = positions[name]
        pnl = realized_pnl(price, position)
        realized_state = state[:-1]
        next_change = np.diff(price)
        for label in ("strong_trend", "weak_mean_reversion"):
            mask = realized_state == label
            runs = [duration for current, duration in _state_runs(state) if current == label]
            state_pos = position[:-1][mask]
            state_pnl = pnl[1:][mask]
            always_state_pnl = baseline_pnl[1:][mask]
            rows.append(
                {
                    "Candidate": name,
                    "Attribution": "state",
                    "State": label,
                    "Count": int(np.sum(mask)),
                    "Mean duration": float(np.mean(runs)) if runs else float("nan"),
                    "Total next-day price change": float(np.sum(next_change[mask])),
                    "Mean next-day price change": float(np.mean(next_change[mask])) if np.any(mask) else float("nan"),
                    "Always-long P&L": float(np.sum(always_state_pnl)),
                    "Strategy P&L": float(np.sum(state_pnl)),
                    "Long P&L": float(np.sum(state_pnl[state_pos > 0])),
                    "Short P&L": float(np.sum(state_pnl[state_pos < 0])),
                }
            )
        transition_mask = np.r_[False, state[1:] != state[:-1]][:-1]
        transition_pnl = pnl[1:][transition_mask]
        rows.append(
            {
                "Candidate": name,
                "Attribution": "state_transition",
                "State": "transition",
                "Count": int(np.sum(transition_mask)),
                "Mean duration": float("nan"),
                "Total next-day price change": float(np.sum(next_change[transition_mask])),
                "Mean next-day price change": float(np.mean(next_change[transition_mask]))
                if np.any(transition_mask)
                else float("nan"),
                "Always-long P&L": float(np.sum(baseline_pnl[1:][transition_mask])),
                "Strategy P&L": float(np.sum(transition_pnl)),
                "Long P&L": float(np.sum(transition_pnl[position[:-1][transition_mask] > 0])),
                "Short P&L": float(np.sum(transition_pnl[position[:-1][transition_mask] < 0])),
            }
        )
    return pd.DataFrame(rows)


def slope_bucket_table(price: np.ndarray, trace: dict[str, np.ndarray]) -> pd.DataFrame:
    z = trace["z"][:-1]
    next_change = np.diff(price)
    labels = np.full(len(z), "", dtype=object)
    labels[z < -1.0] = "below -1.0"
    labels[(z >= -1.0) & (z < -0.6)] = "-1.0 to -0.6"
    labels[(z >= -0.6) & (z < 0.0)] = "-0.6 to 0"
    labels[(z >= 0.0) & (z < 0.6)] = "0 to 0.6"
    labels[(z >= 0.6) & (z <= 1.0)] = "0.6 to 1.0"
    labels[z > 1.0] = "above 1.0"
    rows: list[dict[str, object]] = []
    order = ["below -1.0", "-1.0 to -0.6", "-0.6 to 0", "0 to 0.6", "0.6 to 1.0", "above 1.0"]
    for label in order:
        mask = labels == label
        values = next_change[mask]
        rows.append(
            {
                "Scope": "full",
                "Segment": "all",
                "Bucket": label,
                "Observations": int(np.sum(mask)),
                "Mean next-day change": float(np.mean(values)) if len(values) else float("nan"),
                "Total next-day change": float(np.sum(values)),
                "Volatility": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                "Positive next-day fraction": float(np.mean(values > 0)) if len(values) else float("nan"),
                "Slope-following hit rate": float(np.mean(z[mask] * values > 0)) if len(values) else float("nan"),
            }
        )
    for kind, label, start, end in _segment_definitions(len(price)):
        if kind != "quarter":
            continue
        segment_z = trace["z"][start : min(end, len(price) - 1)]
        segment_y = next_change[start : min(end, len(price) - 1)]
        segment_labels = labels[start : min(end, len(price) - 1)]
        for bucket in order:
            mask = segment_labels == bucket
            values = segment_y[mask]
            rows.append(
                {
                    "Scope": "quarter",
                    "Segment": label,
                    "Bucket": bucket,
                    "Observations": int(np.sum(mask)),
                    "Mean next-day change": float(np.mean(values)) if len(values) else float("nan"),
                    "Total next-day change": float(np.sum(values)),
                    "Volatility": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                    "Positive next-day fraction": float(np.mean(values > 0)) if len(values) else float("nan"),
                    "Slope-following hit rate": float(np.mean(segment_z[mask] * values > 0))
                    if len(values)
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def hac_regression(x: np.ndarray, y: np.ndarray, lags: int = 5) -> dict[str, float]:
    """OLS with a small Newey-West/HAC covariance, without scipy/statsmodels."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 8:
        return {
            "Observations": int(len(x)),
            "Intercept": float("nan"),
            "Beta": float("nan"),
            "Beta SE HAC": float("nan"),
            "Beta CI low": float("nan"),
            "Beta CI high": float("nan"),
            "Beta z": float("nan"),
            "Beta p normal": float("nan"),
        }
    design = np.column_stack([np.ones(len(x)), x])
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    residual = y - design @ beta
    score = design * residual[:, None]
    meat = score.T @ score
    max_lag = min(int(lags), max(1, len(x) // 4))
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        cross = score[lag:].T @ score[:-lag]
        meat += weight * (cross + cross.T)
    covariance = xtx_inv @ meat @ xtx_inv
    se = math.sqrt(max(float(covariance[1, 1]), 0.0))
    z_score = float(beta[1] / se) if se > 0 else float("nan")
    p_value = math.erfc(abs(z_score) / math.sqrt(2.0)) if np.isfinite(z_score) else float("nan")
    return {
        "Observations": int(len(x)),
        "Intercept": float(beta[0]),
        "Beta": float(beta[1]),
        "Beta SE HAC": se,
        "Beta CI low": float(beta[1] - 1.96 * se),
        "Beta CI high": float(beta[1] + 1.96 * se),
        "Beta z": z_score,
        "Beta p normal": p_value,
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def ema_predictive_table(price: np.ndarray, traces: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    next_change = np.diff(price)
    top5 = np.argsort(next_change)[-5:]
    excluded = np.ones(len(next_change), dtype=bool)
    excluded[top5] = False
    for name in ("original_ema_mean_reversion", "corrected_ema_mean_reversion_prevstd", "corrected_hybrid_prevstd"):
        trace = traces[name]
        z = trace["z"][:-1] if "z" in trace else trace["ema_z"][:-1]
        if name == "corrected_hybrid_prevstd":
            z = trace["ema_z"][:-1]
        regimes = {
            "all": np.ones(len(next_change), dtype=bool),
            "weak trend": np.abs(traces["corrected_hybrid_prevstd"]["z"][:-1]) < 0.6,
            "strong trend": np.abs(traces["corrected_hybrid_prevstd"]["z"][:-1]) >= 0.6,
            "excluding best 5 next-change days": excluded,
        }
        for scope, scope_mask in regimes.items():
            mask = scope_mask & np.isfinite(z)
            stats = hac_regression(z[mask], next_change[mask], lags=5)
            corr = float(np.corrcoef(_rankdata(z[mask]), _rankdata(next_change[mask]))[0, 1]) if np.sum(mask) > 2 else float("nan")
            reversal_hit = float(np.mean((-np.sign(z[mask])) * next_change[mask] > 0)) if np.any(mask) else float("nan")
            rows.append(
                {
                    "Signal": name,
                    "Scope": scope,
                    **stats,
                    "Rank correlation": corr,
                    "Reversal directional hit rate": reversal_hit,
                    "Mean deviation z": float(np.mean(z[mask])) if np.any(mask) else float("nan"),
                    "Mean next-day change": float(np.mean(next_change[mask])) if np.any(mask) else float("nan"),
                }
            )
        for kind, label, start, end in _segment_definitions(len(price)):
            if kind != "quarter":
                continue
            segment_z = z[start : min(end, len(next_change))]
            segment_y = next_change[start : min(end, len(next_change))]
            mask = np.isfinite(segment_z)
            stats = hac_regression(segment_z[mask], segment_y[mask], lags=5)
            rows.append(
                {
                    "Signal": name,
                    "Scope": "quarter",
                    "Segment": label,
                    **stats,
                    "Rank correlation": float(
                        np.corrcoef(_rankdata(segment_z[mask]), _rankdata(segment_y[mask]))[0, 1]
                    )
                    if np.sum(mask) > 2
                    else float("nan"),
                    "Reversal directional hit rate": float(
                        np.mean((-np.sign(segment_z[mask])) * segment_y[mask] > 0)
                    )
                    if np.any(mask)
                    else float("nan"),
                    "Mean deviation z": float(np.mean(segment_z[mask])) if np.any(mask) else float("nan"),
                    "Mean next-day change": float(np.mean(segment_y[mask])) if np.any(mask) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def sensitivity_table(price: np.ndarray, baseline_pnl: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(
        family: str,
        label: str,
        position: np.ndarray,
        parameters: dict[str, object],
    ) -> None:
        pnl = realized_pnl(price, position)
        row = {
            "Family": family,
            "Configuration": label,
            "P&L": float(np.sum(pnl)),
            "Incremental vs always long": float(np.sum(pnl) - np.sum(baseline_pnl)),
            "Positive quarters": int(
                sum(
                    np.sum(pnl[start:end]) > 0
                    for _, kind_label, start, end in _segment_definitions(len(price))
                    if kind_label.startswith("Q")
                )
            ),
        }
        row.update(parameters)
        rows.append(row)

    for q_level in (0.25, 0.5, 1.0, 2.0):
        for q_slope in (0.02, 0.05, 0.10):
            position = simple_kalman_position(price, (q_level, q_slope, 9.0), confidence=1.0)[0]
            add(
                "kalman_q_level_q_slope",
                f"qL={q_level:g},qS={q_slope:g},R=9,c=1",
                position,
                {"q_level": q_level, "q_slope": q_slope, "R": 9.0, "confidence": 1.0},
            )
    for observation_variance in (4.0, 5.0, 9.0, 16.0):
        position = simple_kalman_position(price, (1.0, 0.05, observation_variance), confidence=1.0)[0]
        add(
            "kalman_observation_variance",
            f"qL=1,qS=.05,R={observation_variance:g},c=1",
            position,
            {"q_level": 1.0, "q_slope": 0.05, "R": observation_variance, "confidence": 1.0},
        )
    for confidence in (0.5, 0.75, 1.0, 1.25, 1.5):
        position = simple_kalman_position(price, K2, confidence=confidence)[0]
        add(
            "kalman_confidence",
            f"K2,c={confidence:g}",
            position,
            {"q_level": 1.0, "q_slope": 0.05, "R": 9.0, "confidence": confidence},
        )
    hybrid_builders = (
        ("original", original_hybrid_position),
        ("corrected", corrected_hybrid_position),
    )
    for variant, builder in hybrid_builders:
        for trend_threshold in (0.4, 0.6, 0.8, 1.0, 1.5):
            position = builder(price, trend_threshold=trend_threshold)[0]
            add(
                f"{variant}_hybrid_trend_threshold",
                f"trend={trend_threshold:g},revert=.2,alpha=.06",
                position,
                {"variant": variant, "trend_threshold": trend_threshold, "revert_threshold": 0.2, "alpha": 0.06},
            )
        for confidence in (0.5, 1.0, 1.5):
            position = builder(price, short_confidence=confidence)[0]
            add(
                f"{variant}_hybrid_short_confidence",
                f"trend=.6,revert=.2,alpha=.06,short={confidence:g}",
                position,
                {"variant": variant, "trend_threshold": 0.6, "revert_threshold": 0.2, "alpha": 0.06, "short_confidence": confidence},
            )
        for alpha in (0.03, 0.06, 0.10, 0.20):
            for revert_threshold in (0.0, 0.2, 0.5, 1.0):
                position = builder(
                    price, alpha=alpha, revert_threshold=revert_threshold
                )[0]
                add(
                    f"{variant}_hybrid_alpha_revert_grid",
                    f"alpha={alpha:g},revert={revert_threshold:g}",
                    position,
                    {"variant": variant, "alpha": alpha, "revert_threshold": revert_threshold, "trend_threshold": 0.6},
                )
    return pd.DataFrame(rows)


def sensitivity_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, frame in table.groupby("Family"):
        rows.append(
            {
                "Family": family,
                "Configurations": int(len(frame)),
                "Median P&L": float(frame["P&L"].median()),
                "Lower decile P&L": float(frame["P&L"].quantile(0.10)),
                "Worst P&L": float(frame["P&L"].min()),
                "Best P&L": float(frame["P&L"].max()),
                "Median incremental": float(frame["Incremental vs always long"].median()),
                "Positive-quarter median": float(frame["Positive quarters"].median()),
            }
        )
    return pd.DataFrame(rows)


def _circular_block_changes(changes: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    pieces: list[np.ndarray] = []
    total = 0
    while total < len(changes):
        start = int(rng.integers(0, len(changes)))
        piece = changes[(start + np.arange(block_size)) % len(changes)]
        pieces.append(piece)
        total += len(piece)
    return np.concatenate(pieces)[: len(changes)]


def _path_from_changes(
    changes: np.ndarray,
    start_price: float,
    preserve_positive: bool = True,
) -> tuple[np.ndarray, bool]:
    path = np.r_[start_price, start_price + np.cumsum(changes)]
    guarded = False
    if preserve_positive and np.min(path) <= 1.0:
        path = path + (1.0 - float(np.min(path)) + 1e-6)
        guarded = True
    return path, guarded


def price_path_bootstrap(
    price: np.ndarray,
    candidates: Iterable[str],
    repetitions: int = 500,
    block_sizes: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Rebuild prices from moving/circular daily-return blocks and rerun models."""

    candidates = list(candidates)
    rng = np.random.default_rng(SEED + 11)
    rows: list[dict[str, object]] = []
    daily_returns = np.diff(price) / price[:-1]
    for block_size in block_sizes:
        totals = {name: np.empty(repetitions, dtype=float) for name in candidates}
        increments = {name: np.empty(repetitions, dtype=float) for name in candidates}
        drawdowns = {name: np.empty(repetitions, dtype=float) for name in candidates}
        always_totals = np.empty(repetitions, dtype=float)
        guard_count = 0
        for repetition in range(repetitions):
            sampled_returns = _circular_block_changes(daily_returns, block_size, rng)
            synthetic = np.r_[price[0], price[0] * np.cumprod(1.0 + sampled_returns)]
            synthetic_positions = candidate_positions(synthetic)
            always = realized_pnl(synthetic, synthetic_positions["always_long"])
            always_totals[repetition] = float(np.sum(always))
            for name in candidates:
                pnl = realized_pnl(synthetic, synthetic_positions[name])
                totals[name][repetition] = float(np.sum(pnl))
                increments[name][repetition] = float(np.sum(pnl) - np.sum(always))
                drawdowns[name][repetition] = _max_drawdown(pnl)
        for name in candidates:
            values = totals[name]
            inc = increments[name]
            rows.append(
                {
                    "Bootstrap": "price_path_circular_return_blocks",
                    "Block size": block_size,
                    "Candidate": name,
                    "Repetitions": repetitions,
                    "Positive path fraction": float(np.mean(values > 0)),
                    "Positive path MC SE": float(
                        math.sqrt(np.mean(values > 0) * (1 - np.mean(values > 0)) / repetitions)
                    ),
                    "Beats always-long fraction": float(np.mean(inc > 0)),
                    "Beats always-long MC SE": float(
                        math.sqrt(np.mean(inc > 0) * (1 - np.mean(inc > 0)) / repetitions)
                    ),
                    "Median P&L": float(np.median(values)),
                    "P5 P&L": float(np.quantile(values, 0.05)),
                    "P95 P&L": float(np.quantile(values, 0.95)),
                    "Median incremental": float(np.median(inc)),
                    "P5 incremental": float(np.quantile(inc, 0.05)),
                    "P95 incremental": float(np.quantile(inc, 0.95)),
                    "Median max drawdown": float(np.median(drawdowns[name])),
                    "P5 max drawdown": float(np.quantile(drawdowns[name], 0.05)),
                    "P95 max drawdown": float(np.quantile(drawdowns[name], 0.95)),
                    "Positive-price guard paths": guard_count,
                }
            )
    return pd.DataFrame(rows)


def fixed_realized_pnl_block_diagnostic(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
    repetitions: int = 500,
) -> pd.DataFrame:
    """A deliberately secondary diagnostic: resample fixed realized P&L blocks."""

    rng = np.random.default_rng(SEED + 12)
    rows: list[dict[str, object]] = []
    for name, position in positions.items():
        fixed = realized_pnl(price, position)[1:]
        for block_size in (5, 10, 20):
            values = np.empty(repetitions, dtype=float)
            for repetition in range(repetitions):
                sampled = _circular_block_changes(fixed, block_size, rng)
                values[repetition] = float(np.sum(sampled))
            rows.append(
                {
                    "Bootstrap": "fixed_realized_pnl_blocks_diagnostic_only",
                    "Block size": block_size,
                    "Candidate": name,
                    "Repetitions": repetitions,
                    "Positive path fraction": float(np.mean(values > 0)),
                    "Beats always-long fraction": float("nan"),
                    "Median P&L": float(np.median(values)),
                    "P5 P&L": float(np.quantile(values, 0.05)),
                    "P95 P&L": float(np.quantile(values, 0.95)),
                }
            )
    return pd.DataFrame(rows)


def _shift_position(position: np.ndarray, shift: int) -> np.ndarray:
    out = np.zeros_like(position)
    if shift > 0:
        out[shift:] = position[:-shift]
    elif shift < 0:
        out[:shift] = position[-shift:]
    else:
        out[:] = position
    return out.astype(int)


def stress_and_placebo_table(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
    baseline_pnl: np.ndarray,
    bootstrap_reps: int = 500,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    serious = [
        "always_long",
        "simple_kalman_K2",
        "original_ema_mean_reversion",
        "corrected_ema_mean_reversion_prevstd",
        "original_hybrid",
        "corrected_hybrid_prevstd",
    ]
    for name in serious:
        pnl = realized_pnl(price, positions[name])
        for shift in (-3, -2, -1, 0, 1, 2, 3):
            shifted = _shift_position(positions[name], shift)
            shifted_pnl = realized_pnl(price, shifted)
            rows.append(
                {
                    "Test": "signal_timing_shift",
                    "Scenario": f"shift_{shift:+d}_days",
                    "Candidate": name,
                    "P&L": float(np.sum(shifted_pnl)),
                    "Incremental vs always long": float(np.sum(shifted_pnl) - np.sum(baseline_pnl)),
                    "Parameter": shift,
                }
            )
        order = np.argsort(pnl)[::-1]
        for k in (1, 5, 10, 20):
            keep = np.ones(len(pnl), dtype=bool)
            keep[order[:k]] = False
            rows.append(
                {
                    "Test": "remove_best_strategy_days",
                    "Scenario": f"remove_best_{k}",
                    "Candidate": name,
                    "P&L": float(np.sum(pnl[keep])),
                    "Incremental vs always long": float(np.sum(pnl[keep]) - np.sum(baseline_pnl[keep])),
                    "Parameter": k,
                }
            )

    changes = np.diff(price)
    centered = changes - np.mean(changes)
    smooth = pd.Series(changes).rolling(7, min_periods=1).mean().to_numpy()
    stress_changes: dict[str, np.ndarray] = {
        "drift_plus_0.20": changes + 0.20,
        "drift_minus_0.20": changes - 0.20,
        "volatility_half": np.mean(changes) + 0.5 * centered,
        "volatility_1.5x": np.mean(changes) + 1.5 * centered,
        "noise_half": smooth + 0.5 * (changes - smooth),
        "noise_1.5x": smooth + 1.5 * (changes - smooth),
        "volatility_shift_up": np.r_[np.mean(changes) + 0.75 * centered[: len(changes) // 2], np.mean(changes) + 1.5 * centered[len(changes) // 2 :]],
        "gradual_trend": 0.5 * changes + 0.5 * smooth + 0.10,
    }
    for scenario, scenario_changes in stress_changes.items():
        scenario_price, guarded = _path_from_changes(scenario_changes, price[0])
        scenario_positions = candidate_positions(scenario_price)
        always = realized_pnl(scenario_price, scenario_positions["always_long"])
        for name in serious:
            scenario_pnl = realized_pnl(scenario_price, scenario_positions[name])
            rows.append(
                {
                    "Test": "synthetic_generator_conditioned_stress",
                    "Scenario": scenario,
                    "Candidate": name,
                    "P&L": float(np.sum(scenario_pnl)),
                    "Incremental vs always long": float(np.sum(scenario_pnl) - np.sum(always)),
                    "Parameter": int(guarded),
                }
            )
    for block_size in (5, 20, 40):
        rng = np.random.default_rng(SEED + block_size)
        sampled = _circular_block_changes(changes, block_size, rng)
        scenario_price, guarded = _path_from_changes(sampled, price[0])
        scenario_positions = candidate_positions(scenario_price)
        always = realized_pnl(scenario_price, scenario_positions["always_long"])
        for name in serious:
            scenario_pnl = realized_pnl(scenario_price, scenario_positions[name])
            rows.append(
                {
                    "Test": "single_circular_path_regime_duration_stress",
                    "Scenario": f"block_{block_size}",
                    "Candidate": name,
                    "P&L": float(np.sum(scenario_pnl)),
                    "Incremental vs always long": float(np.sum(scenario_pnl) - np.sum(always)),
                    "Parameter": int(guarded),
                }
            )
    return pd.DataFrame(rows)


def familywise_permuted_null(
    price: np.ndarray,
    repetitions: int = 500,
) -> pd.DataFrame:
    """Run both an easy permutation null and a structure-preserving block null."""

    observed_positions = candidate_positions(price)
    observed = {
        name: float(np.sum(realized_pnl(price, position)))
        for name, position in observed_positions.items()
    }
    observed_name = max(observed, key=observed.get)
    observed_best = observed[observed_name]
    rng = np.random.default_rng(SEED + 21)
    changes = np.diff(price)
    permutation_maxima = np.empty(repetitions, dtype=float)
    block_maxima = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        shuffled = rng.permutation(changes)
        synthetic, _ = _path_from_changes(shuffled, price[0])
        synthetic_positions = candidate_positions(synthetic)
        permutation_maxima[repetition] = max(
            float(np.sum(realized_pnl(synthetic, position)))
            for position in synthetic_positions.values()
        )
        blocked = _circular_block_changes(changes, 10, rng)
        synthetic, _ = _path_from_changes(blocked, price[0])
        synthetic_positions = candidate_positions(synthetic)
        block_maxima[repetition] = max(
            float(np.sum(realized_pnl(synthetic, position)))
            for position in synthetic_positions.values()
        )

    rows = []
    for null_name, maxima in (
        ("daily_change_permutation", permutation_maxima),
        ("circular_change_blocks_10_structure_preserving", block_maxima),
    ):
        hits = int(np.sum(maxima >= observed_best))
        p_value = (1.0 + hits) / (repetitions + 1.0)
        se = math.sqrt(p_value * (1.0 - p_value) / repetitions)
        rows.append(
            {
                "Null": null_name,
                "Candidates in maximum": len(observed_positions),
                "Observed best candidate": observed_name,
                "Observed best P&L": observed_best,
                "Repetitions": repetitions,
                "Null median family maximum": float(np.median(maxima)),
                "Null 95 family maximum": float(np.quantile(maxima, 0.95)),
                "Exceedances": hits,
                "Monte Carlo p-value": p_value,
                "Monte Carlo SE": se,
                "MC interval low": max(0.0, p_value - 1.96 * se),
                "MC interval high": min(1.0, p_value + 1.96 * se),
            }
        )
    return pd.DataFrame(rows)


def correctness_table(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
    baseline_pnl: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    audit_days = (0, 30, 120, 240, 300)
    for name, position in positions.items():
        rows.append(
            {
                "Candidate": name,
                "Check": "integer_positions",
                "Value": bool(np.issubdtype(position.dtype, np.integer)),
                "Details": str(position.dtype),
            }
        )
        rows.append(
            {
                "Candidate": name,
                "Check": "within_Jeans_limit",
                "Value": bool(np.max(np.abs(position)) <= LIMIT),
                "Details": int(np.max(np.abs(position))),
            }
        )
        rows.append(
            {
                "Candidate": name,
                "Check": "within_Jeans_gross_value",
                "Value": bool(np.max(np.abs(position) * price) <= JEANS_BUDGET),
                "Details": float(np.max(np.abs(position) * price)),
            }
        )
        full_pnl = realized_pnl(price, position)
        expected = np.r_[0.0, position[:-1] * np.diff(price)]
        rows.append(
            {
                "Candidate": name,
                "Check": "pnl_uses_prior_position",
                "Value": bool(np.allclose(full_pnl, expected, atol=0.011)),
                "Details": "pnl[t]=position[t-1]*(price[t]-price[t-1])",
            }
        )
        prefix_consistent = True
        future_unchanged = True
        for day in audit_days:
            prefix = candidate_positions(price[: day + 1])[name]
            if prefix[-1] != position[day]:
                prefix_consistent = False
            altered = price.copy()
            altered[day + 1 :] += 1000.0
            if candidate_positions(altered)[name][day] != position[day]:
                future_unchanged = False
        rows.append(
            {
                "Candidate": name,
                "Check": "prefix_replay_matches",
                "Value": prefix_consistent,
                "Details": str(audit_days),
            }
        )
        rows.append(
            {
                "Candidate": name,
                "Check": "future_perturbation_unchanged",
                "Value": future_unchanged,
                "Details": str(audit_days),
            }
        )
    rows.append(
        {
            "Candidate": "all",
            "Check": "always_long_round1_reference",
            "Value": bool(np.isclose(np.sum(baseline_pnl), 38136.0)),
            "Details": float(np.sum(baseline_pnl)),
        }
    )
    return pd.DataFrame(rows)


def concentration_table(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, position in positions.items():
        pnl = realized_pnl(price, position)
        order = np.argsort(pnl)[::-1]
        for k in (1, 5, 10, 20):
            keep = np.ones(len(pnl), dtype=bool)
            keep[order[:k]] = False
            rows.append(
                {
                    "Candidate": name,
                    "Diagnostic": f"exclude_best_{k}_strategy_days",
                    "Removed days": k,
                    "Removed P&L": float(np.sum(pnl[order[:k]])),
                    "Remaining P&L": float(np.sum(pnl[keep])),
                    "Remaining incremental vs always long": float(
                        np.sum(pnl[keep])
                        - np.sum(realized_pnl(price, positions["always_long"])[keep])
                    ),
                    "Best-day share": _top_day_share(pnl, np.sum(pnl), k),
                }
            )
        for end_name, end in (("full", len(price)), ("exclude_last_30", len(price) - 30), ("exclude_last_60", len(price) - 60), ("exclude_last_91", len(price) - 91)):
            rows.append(
                {
                    "Candidate": name,
                    "Diagnostic": end_name,
                    "Removed days": len(price) - end,
                    "Removed P&L": float(np.sum(pnl[end:])),
                    "Remaining P&L": float(np.sum(pnl[:end])),
                    "Remaining incremental vs always long": float(
                        np.sum(pnl[:end]) - np.sum(realized_pnl(price, positions["always_long"])[:end])
                    ),
                    "Best-day share": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _write_figures(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
    chronological: pd.DataFrame,
    sensitivity: pd.DataFrame,
    slope_buckets: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    key = ["always_long", "simple_kalman_K2", "original_hybrid", "corrected_hybrid_prevstd"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in key:
        ax.plot(np.cumsum(realized_pnl(price, positions[name])), label=name)
    ax.set_title("Thrifted Jeans causal cumulative P&L")
    ax.set_xlabel("Decision/realization day")
    ax.set_ylabel("AUD P&L")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_dir / "cumulative_pnl_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(price, color="black", linewidth=1.0)
    axes[0].set_title("Price and causal strategy positions")
    axes[0].set_ylabel("Price")
    axes[1].plot(positions["simple_kalman_K2"], label="simple K2", drawstyle="steps-post")
    axes[1].plot(positions["original_hybrid"], label="original hybrid", drawstyle="steps-post")
    axes[1].plot(positions["corrected_hybrid_prevstd"], label="corrected hybrid", drawstyle="steps-post")
    axes[1].set_ylabel("Desired Jeans units")
    axes[1].set_xlabel("Decision day")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_dir / "positions_against_price.png", dpi=160)
    plt.close(fig)

    quarter = chronological[chronological["Split"] == "quarter"].pivot(
        index="Segment", columns="Candidate", values="P&L"
    )
    names = [name for name in key if name in quarter.columns]
    fig, ax = plt.subplots(figsize=(10, 5))
    quarter[names].plot(kind="bar", ax=ax)
    ax.set_title("P&L by non-overlapping quarter")
    ax.set_ylabel("AUD P&L")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_dir / "pnl_by_chronological_segment.png", dpi=160)
    plt.close(fig)

    q_surface = sensitivity[sensitivity["Family"] == "kalman_q_level_q_slope"].copy()
    pivot = q_surface.pivot(index="q_level", columns="q_slope", values="P&L")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{x:g}" for x in pivot.index])
    ax.set_xlabel("q_slope")
    ax.set_ylabel("q_level")
    ax.set_title("Simple K2-style Kalman q surface: P&L")
    fig.colorbar(im, ax=ax, label="AUD P&L")
    fig.tight_layout()
    fig.savefig(figure_dir / "kalman_q_sensitivity_heatmap.png", dpi=160)
    plt.close(fig)

    hybrid_surface = sensitivity[sensitivity["Family"] == "hybrid_alpha_revert_grid"].copy()
    pivot = hybrid_surface.pivot(index="alpha", columns="revert_threshold", values="P&L")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{x:g}" for x in pivot.index])
    ax.set_xlabel("revert threshold")
    ax.set_ylabel("EMA alpha")
    ax.set_title("Original hybrid alpha/reversion surface: P&L")
    fig.colorbar(im, ax=ax, label="AUD P&L")
    fig.tight_layout()
    fig.savefig(figure_dir / "hybrid_alpha_reversion_heatmap.png", dpi=160)
    plt.close(fig)

    full_slope = slope_buckets[slope_buckets["Scope"] == "full"].copy()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(full_slope["Bucket"], full_slope["Mean next-day change"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Next-day Jeans price change by K2 slope z bucket")
    ax.set_ylabel("Mean next-day change")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_dir / "next_day_returns_by_slope_bucket.png", dpi=160)
    plt.close(fig)


def run_audit(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    bootstrap_repetitions: int | None = None,
    familywise_repetitions: int | None = None,
) -> dict[str, object]:
    """Run the complete reproducible audit and write all requested artifacts."""

    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if bootstrap_repetitions is None:
        bootstrap_repetitions = int(os.environ.get("JEANS_AUDIT_BOOTSTRAP_REPS", "500"))
    if familywise_repetitions is None:
        familywise_repetitions = int(os.environ.get("JEANS_AUDIT_FAMILYWISE_REPS", "500"))

    price = load_price_series()
    positions = candidate_positions(price)
    traces = candidate_traces(price)
    baseline_pnl = realized_pnl(price, positions["always_long"])

    comparison_rows: list[dict[str, object]] = []
    for name, position in positions.items():
        row = {"Candidate": name}
        row.update(position_metrics(price, position, baseline_pnl))
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows).set_index("Candidate")
    comparison.to_csv(output_dir / "candidate_comparison.csv")

    chronological = chronological_table(price, positions, baseline_pnl)
    chronological.to_csv(output_dir / "chronological_splits.csv", index=False)

    regime = regime_attribution_table(price, positions, traces, baseline_pnl)
    regime.to_csv(output_dir / "regime_attribution.csv", index=False)

    slopes = slope_bucket_table(price, traces["simple_kalman_K2"])
    slopes.to_csv(output_dir / "slope_bucket_returns.csv", index=False)

    ema_regressions = ema_predictive_table(price, traces)
    ema_regressions.to_csv(output_dir / "ema_predictive_regressions.csv", index=False)

    sensitivity = sensitivity_table(price, baseline_pnl)
    sensitivity_summary(sensitivity).to_csv(output_dir / "parameter_sensitivity_summary.csv", index=False)
    sensitivity.to_csv(output_dir / "parameter_sensitivity.csv", index=False)

    concentration = concentration_table(price, positions)
    concentration.to_csv(output_dir / "concentration_diagnostics.csv", index=False)

    correctness = correctness_table(price, positions, baseline_pnl)
    correctness.to_csv(output_dir / "correctness_checks.csv", index=False)

    bootstrap_candidates = list(positions.keys())
    path_bootstrap = price_path_bootstrap(
        price,
        bootstrap_candidates,
        repetitions=bootstrap_repetitions,
    )
    fixed_bootstrap = fixed_realized_pnl_block_diagnostic(
        price,
        positions,
        repetitions=bootstrap_repetitions,
    )
    path_bootstrap.to_csv(output_dir / "price_path_bootstrap.csv", index=False)
    fixed_bootstrap.to_csv(output_dir / "fixed_pnl_bootstrap_diagnostic.csv", index=False)

    stress = stress_and_placebo_table(
        price,
        positions,
        baseline_pnl,
        bootstrap_reps=bootstrap_repetitions,
    )
    familywise = familywise_permuted_null(price, repetitions=familywise_repetitions)
    stress.to_csv(output_dir / "stress_placebo_tests.csv", index=False)
    familywise.to_csv(output_dir / "familywise_null.csv", index=False)

    _write_figures(price, positions, chronological, sensitivity, slopes, figure_dir)

    manifest = {
        "data_path": str(DATA_PATH),
        "observations": int(len(price)),
        "start_price": float(price[0]),
        "end_price": float(price[-1]),
        "daily_change_autocorrelation": float(pd.Series(np.diff(price)).autocorr(1)),
        "base_candidate_count": int(len(positions)),
        "sensitivity_configuration_count": int(len(sensitivity)),
        "path_bootstrap_repetitions_per_block": int(bootstrap_repetitions),
        "path_bootstrap_block_sizes": [5, 10, 20],
        "familywise_repetitions": int(familywise_repetitions),
        "zero_transaction_costs_assumed": True,
        "production_files_modified": False,
        "reference_pnls": {
            "always_long": float(comparison.loc["always_long", "P&L"]),
            "simple_kalman_K2": float(comparison.loc["simple_kalman_K2", "P&L"]),
            "original_hybrid": float(comparison.loc["original_hybrid", "P&L"]),
        },
        "notes": [
            "Original hybrid uses the exact stateful mechanics from trader_interface/algorithm_juan.",
            "Corrected EMA uses price[t] minus EMA through t-1 and volatility from prior causal deviations.",
            "Price-path bootstrap resamples circular daily percentage-return blocks, preserves the original $40 start, and reruns positions; fixed-P&L bootstrap is diagnostic only.",
        ],
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Thrifted Jeans audit complete")
    print(comparison[["P&L", "Incremental vs always long", "Max drawdown", "Turnover"]].round(2).to_string())
    print(f"Price-path bootstrap repetitions per block: {bootstrap_repetitions}")
    print(f"Family-wise permutation repetitions: {familywise_repetitions}")
    return {
        "price": price,
        "positions": positions,
        "traces": traces,
        "comparison": comparison,
        "chronological": chronological,
        "regime": regime,
        "slopes": slopes,
        "ema_regressions": ema_regressions,
        "sensitivity": sensitivity,
        "concentration": concentration,
        "correctness": correctness,
        "price_path_bootstrap": path_bootstrap,
        "fixed_pnl_bootstrap": fixed_bootstrap,
        "stress": stress,
        "familywise": familywise,
        "manifest": manifest,
    }


if __name__ == "__main__":
    run_audit()
