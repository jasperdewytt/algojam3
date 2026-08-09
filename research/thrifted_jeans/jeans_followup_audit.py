"""Focused follow-up audit for Thrifted Jeans.

This file is intentionally separate from jeans_audit.py.  It imports the
prior research helpers, but writes all new tables and figures to
followup_outputs/ and followup_figures/.  Production code and supplied data
are never imported by a trading interface or modified.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jeans_audit import (
    DATA_PATH,
    JEANS_BUDGET,
    LIMIT,
    SEED,
    _causal_ema_trace,
    _circular_block_changes,
    _kalman_trace,
    _mean_reversion_transition,
    _max_drawdown,
    _segment_definitions,
    _state_runs,
    hac_regression,
    load_price_series,
    realized_pnl,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "followup_outputs"
DEFAULT_FIGURE_DIR = Path(__file__).resolve().parent / "followup_figures"

K2_PARAMS = (1.0, 0.05, 9.0)
HYBRID_PARAMS = (0.5, 0.05, 5.0)
HYBRID_THRESHOLD = 0.6
EMA_ALPHA = 0.06
REVERT_THRESHOLD = 0.2
PLAUSIBLE_MIN = 1.0
PLAUSIBLE_MAX = 200.0

BASE_NAMES = [
    "always_long",
    "simple_K2",
    "hybrid_Kalman_no_EMA",
    "corrected_hybrid",
    "original_hybrid",
    "strong_direction_flat_weak",
    "strong_direction_long_weak",
    "strong_direction_hold_weak",
    "strong_direction_latest_reversal_weak",
    "strong_direction_ema_sign_weak",
]


def _always_long_day1(price: np.ndarray) -> np.ndarray:
    position = np.zeros(len(price), dtype=int)
    if len(price) > 1:
        position[1:] = LIMIT
    return position


def _kalman_long_bias(
    price: np.ndarray,
    parameters: tuple[float, float, float],
    threshold: float,
    strict: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    trace = _kalman_trace(price, parameters)
    position = np.zeros(len(price), dtype=int)
    if strict:
        short_mask = trace["z"] <= -threshold
    else:
        short_mask = trace["z"] < -threshold
    if len(price) > 1:
        position[1:] = np.where(short_mask[1:], -LIMIT, LIMIT)
    return position, trace


def _hybrid_state_trace(price: np.ndarray) -> dict[str, np.ndarray]:
    kalman = _kalman_trace(price, HYBRID_PARAMS)
    ema = _causal_ema_trace(price, EMA_ALPHA, standardize_through_current=False)
    strong = np.abs(kalman["z"]) >= HYBRID_THRESHOLD
    return {
        **kalman,
        "ema": ema["ema"],
        "deviation": ema["deviation"],
        "ema_z": ema["z"],
        "strong": strong.astype(bool),
    }


def _strong_direction_position(trace: dict[str, np.ndarray]) -> np.ndarray:
    position = np.zeros(len(trace["z"]), dtype=int)
    if len(position) > 1:
        strong = trace["strong"][1:]
        position[1:] = np.where(
            strong,
            np.where(trace["z"][1:] <= -HYBRID_THRESHOLD, -LIMIT, LIMIT),
            0,
        )
    return position


def _corrected_hybrid_position_from_trace(
    trace: dict[str, np.ndarray],
) -> np.ndarray:
    position = np.zeros(len(trace["z"]), dtype=int)
    for day in range(1, len(position)):
        if trace["strong"][day]:
            position[day] = (
                -LIMIT if trace["z"][day] <= -HYBRID_THRESHOLD else LIMIT
            )
        else:
            position[day] = _mean_reversion_transition(
                int(position[day - 1]),
                float(trace["ema_z"][day]),
                REVERT_THRESHOLD,
            )
    return position


def _build_followup_candidates(
    price: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Build the exact candidates and critical weak-state ablations."""

    price = np.asarray(price, dtype=float)
    positions: dict[str, np.ndarray] = {}
    traces: dict[str, dict[str, np.ndarray]] = {}

    positions["always_long"] = _always_long_day1(price)

    k2_position, k2_trace = _kalman_long_bias(
        price, K2_PARAMS, threshold=1.0, strict=False
    )
    positions["simple_K2"] = k2_position
    traces["simple_K2"] = k2_trace

    hybrid_trace = _hybrid_state_trace(price)
    traces["hybrid"] = hybrid_trace
    strong_position = _strong_direction_position(hybrid_trace)

    # Candidate C: the hybrid Kalman parameters with no EMA branch at all.
    candidate_c = np.zeros(len(price), dtype=int)
    if len(price) > 1:
        candidate_c[1:] = np.where(
            hybrid_trace["z"][1:] <= -HYBRID_THRESHOLD,
            -LIMIT,
            LIMIT,
        )
    positions["hybrid_Kalman_no_EMA"] = candidate_c

    # Candidate D: the prior corrected causal hybrid, including exact mechanics.
    positions["corrected_hybrid"] = _corrected_hybrid_position_from_trace(
        hybrid_trace
    )

    # Candidate E is diagnostic only.  Reconstruct its old residual bug here:
    # the final EMA value is subtracted from every historical price.
    original_ema_level = price[0]
    original_ema = np.zeros(len(price), dtype=float)
    original_ema_z = np.zeros(len(price), dtype=float)
    for day, observed in enumerate(price):
        if day > 0:
            original_ema_level += EMA_ALPHA * (observed - original_ema_level)
        original_ema[day] = original_ema_level
        residual = price[: day + 1] - original_ema_level
        scale = max(float(np.std(residual)), 1e-12)
        original_ema_z[day] = (observed - original_ema_level) / scale
    original_position = np.zeros(len(price), dtype=int)
    for day in range(1, len(price)):
        if hybrid_trace["strong"][day]:
            original_position[day] = (
                -LIMIT
                if hybrid_trace["z"][day] <= -HYBRID_THRESHOLD
                else LIMIT
            )
        else:
            original_position[day] = _mean_reversion_transition(
                int(original_position[day - 1]),
                float(original_ema_z[day]),
                REVERT_THRESHOLD,
            )
    positions["original_hybrid"] = original_position
    traces["original_hybrid"] = {
        **hybrid_trace,
        "ema": original_ema,
        "ema_z": original_ema_z,
    }

    # Critical weak-state ablations.
    flat_weak = strong_position.copy()
    positions["strong_direction_flat_weak"] = flat_weak

    long_weak = strong_position.copy()
    if len(long_weak) > 1:
        long_weak[1:] = np.where(
            hybrid_trace["strong"][1:], strong_position[1:], LIMIT
        )
    positions["strong_direction_long_weak"] = long_weak

    hold_weak = np.zeros(len(price), dtype=int)
    for day in range(1, len(price)):
        hold_weak[day] = (
            int(strong_position[day])
            if hybrid_trace["strong"][day]
            else int(hold_weak[day - 1])
        )
    positions["strong_direction_hold_weak"] = hold_weak

    latest_reversal = strong_position.copy()
    if len(price) > 1:
        latest_reversal[1:] = np.where(
            hybrid_trace["strong"][1:],
            strong_position[1:],
            -np.sign(np.diff(price)).astype(int) * LIMIT,
        )
    positions["strong_direction_latest_reversal_weak"] = latest_reversal

    ema_sign = strong_position.copy()
    weak_ema_z = hybrid_trace["ema_z"]
    for day in range(1, len(price)):
        if not hybrid_trace["strong"][day]:
            if np.isfinite(weak_ema_z[day]):
                ema_sign[day] = -LIMIT * int(np.sign(weak_ema_z[day]))
            else:
                ema_sign[day] = 0
    positions["strong_direction_ema_sign_weak"] = ema_sign

    for name, position in positions.items():
        positions[name] = np.asarray(position, dtype=int)
    return positions, traces


def build_followup_candidates(price: np.ndarray) -> dict[str, np.ndarray]:
    return _build_followup_candidates(price)[0]


def build_followup_traces(price: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    return _build_followup_candidates(price)[1]


def _longest_loss_streak(pnl: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in pnl:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


def _top_share(pnl: np.ndarray, total: float, k: int) -> float:
    return float(np.sort(pnl)[-k:].sum() / total) if total else float("nan")


def _metrics(
    price: np.ndarray,
    position: np.ndarray,
    always_pnl: np.ndarray,
    k2_pnl: np.ndarray,
) -> dict[str, float | int]:
    pnl = realized_pnl(price, position)
    total = float(np.sum(pnl))
    daily_sd = float(np.std(pnl[1:], ddof=1))
    long_mask = np.r_[False, position[:-1] > 0]
    short_mask = np.r_[False, position[:-1] < 0]
    active_mask = position[:-1] != 0
    active_pnl = pnl[1:][active_mask]
    return {
        "P&L": total,
        "Incremental vs always long": total - float(np.sum(always_pnl)),
        "Incremental vs K2": total - float(np.sum(k2_pnl)),
        "Active days": int(np.sum(position != 0)),
        "Long days": int(np.sum(position > 0)),
        "Short days": int(np.sum(position < 0)),
        "Flat days": int(np.sum(position == 0)),
        "Turnover": int(np.sum(np.abs(np.diff(np.r_[0, position])))),
        "Hit rate active": float(np.mean(active_pnl > 0)) if len(active_pnl) else float("nan"),
        "Sharpe annualized": float(np.mean(pnl[1:]) / daily_sd * math.sqrt(365.0))
        if daily_sd > 0
        else 0.0,
        "Max drawdown": _max_drawdown(pnl),
        "Maximum notional AUD": float(np.max(np.abs(position) * price)),
        "Longest loss streak": _longest_loss_streak(pnl),
        "Long P&L": float(np.sum(pnl[long_mask])),
        "Short P&L": float(np.sum(pnl[short_mask])),
        "Best day": float(np.max(pnl)),
        "Worst day": float(np.min(pnl)),
        "Best 1 day share": _top_share(pnl, total, 1),
        "Best 3 day share": _top_share(pnl, total, 3),
        "Best 5 day share": _top_share(pnl, total, 5),
        "Best 10 day share": _top_share(pnl, total, 10),
        "Best 20 day share": _top_share(pnl, total, 20),
    }


def _state_attribution(
    price: np.ndarray,
    position: np.ndarray,
    strong: np.ndarray,
) -> dict[str, float | int]:
    pnl = realized_pnl(price, position)
    state = strong[:-1]
    realized = pnl[1:]
    long = position[:-1]
    return {
        "Strong days": int(np.sum(state)),
        "Weak days": int(np.sum(~state)),
        "Strong P&L": float(np.sum(realized[state])),
        "Weak P&L": float(np.sum(realized[~state])),
        "Strong long P&L": float(np.sum(realized[state & (long > 0)])),
        "Strong short P&L": float(np.sum(realized[state & (long < 0)])),
        "Weak long P&L": float(np.sum(realized[(~state) & (long > 0)])),
        "Weak short P&L": float(np.sum(realized[(~state) & (long < 0)])),
    }


def _chronological_table(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
    always_pnl: np.ndarray,
    k2_pnl: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, position in positions.items():
        pnl = realized_pnl(price, position)
        exposure = np.r_[False, position[:-1] != 0]
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
                        np.sum(pnl[start:end]) - np.sum(always_pnl[start:end])
                    ),
                    "Incremental vs K2": float(
                        np.sum(pnl[start:end]) - np.sum(k2_pnl[start:end])
                    ),
                    "Long P&L": float(
                        np.sum(pnl[start:end] * (np.r_[False, position[:-1] > 0][start:end]))
                    ),
                    "Short P&L": float(
                        np.sum(pnl[start:end] * (np.r_[False, position[:-1] < 0][start:end]))
                    ),
                    "Active realised days": int(np.sum(exposure[start:end])),
                }
            )
        for excluded in (30, 60, 91):
            end = len(price) - excluded
            rows.append(
                {
                    "Candidate": name,
                    "Split": "exclude_final_days",
                    "Segment": f"exclude_last_{excluded}",
                    "Start day": 0,
                    "End day exclusive": end,
                    "P&L": float(np.sum(pnl[:end])),
                    "Incremental vs always long": float(
                        np.sum(pnl[:end]) - np.sum(always_pnl[:end])
                    ),
                    "Incremental vs K2": float(
                        np.sum(pnl[:end]) - np.sum(k2_pnl[:end])
                    ),
                    "Long P&L": float(
                        np.sum(pnl[:end] * np.r_[False, position[:-1] > 0][:end])
                    ),
                    "Short P&L": float(
                        np.sum(pnl[:end] * np.r_[False, position[:-1] < 0][:end])
                    ),
                    "Active realised days": int(np.sum(exposure[:end])),
                }
            )
    return pd.DataFrame(rows)


def _best_day_exclusions(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
    always_pnl: np.ndarray,
    k2_pnl: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, position in positions.items():
        pnl = realized_pnl(price, position)
        for k in (1, 3, 5, 10, 20):
            best_idx = np.argsort(pnl)[-k:]
            keep = np.ones(len(pnl), dtype=bool)
            keep[best_idx] = False
            rows.append(
                {
                    "Candidate": name,
                    "Excluded": f"best_{k}_realised_days",
                    "Count excluded": k,
                    "Removed P&L": float(np.sum(pnl[best_idx])),
                    "Remaining P&L": float(np.sum(pnl[keep])),
                    "Remaining incremental vs always long": float(
                        np.sum(pnl[keep]) - np.sum(always_pnl[keep])
                    ),
                    "Remaining incremental vs K2": float(
                        np.sum(pnl[keep]) - np.sum(k2_pnl[keep])
                    ),
                }
            )
    return pd.DataFrame(rows)


def _hac_ols_matrix(
    design: np.ndarray,
    y: np.ndarray,
    lags: int = 5,
) -> dict[str, np.ndarray | int]:
    design = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    design = design[valid]
    y = y[valid]
    if len(y) < design.shape[1] + 5:
        nan = np.full(design.shape[1], np.nan)
        return {"n": int(len(y)), "beta": nan, "se": nan}
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    residual = y - design @ beta
    score = design * residual[:, None]
    meat = score.T @ score
    max_lag = min(lags, max(1, len(y) // 4))
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        cross = score[lag:].T @ score[:-lag]
        meat += weight * (cross + cross.T)
    covariance = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {"n": int(len(y)), "beta": beta, "se": se}


def _normal_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0)) if np.isfinite(z) else float("nan")


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return float("nan")
    xr = pd.Series(x[mask]).rank(method="average").to_numpy()
    yr = pd.Series(y[mask]).rank(method="average").to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])


def _weak_sample(
    price: np.ndarray,
    traces: dict[str, dict[str, np.ndarray]],
    positions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    trace = traces["hybrid"]
    z = trace["ema_z"][:-1]
    next_change = np.diff(price)
    weak = (~trace["strong"][:-1]) & np.isfinite(z)
    simple = -np.sign(z) * next_change
    actual = positions["corrected_hybrid"][:-1] * next_change
    return {
        "z": z,
        "next_change": next_change,
        "weak": weak,
        "simple_contribution": simple,
        "actual_contribution": actual,
        "actual_position": positions["corrected_hybrid"][:-1],
        "strong": trace["strong"][:-1],
    }


def _predictive_row(
    label: str,
    z: np.ndarray,
    next_change: np.ndarray,
    contribution: np.ndarray,
    mask: np.ndarray,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    mask = mask & np.isfinite(z) & np.isfinite(next_change) & np.isfinite(contribution)
    z_use = z[mask]
    y_use = next_change[mask]
    c_use = contribution[mask]
    regression = hac_regression(z_use, y_use, lags=5)
    return {
        "Signal": label,
        "Observations": int(np.sum(mask)),
        "Mean next change": float(np.mean(y_use)) if len(y_use) else float("nan"),
        "Median next change": float(np.median(y_use)) if len(y_use) else float("nan"),
        "Mean contribution": float(np.mean(c_use)) if len(c_use) else float("nan"),
        "Median contribution": float(np.median(c_use)) if len(c_use) else float("nan"),
        "Reversal hit rate": float(np.mean(c_use > 0)) if len(c_use) else float("nan"),
        "Rank correlation z/change": _rank_corr(z_use, y_use),
        "Beta": regression.get("Beta"),
        "Beta SE HAC": regression.get("Beta SE HAC"),
        "Beta CI low": regression.get("Beta CI low"),
        "Beta CI high": regression.get("Beta CI high"),
        "Beta p normal": regression.get("Beta p normal"),
        **(extra or {}),
    }


def weak_state_predictive_table(
    price: np.ndarray,
    traces: dict[str, dict[str, np.ndarray]],
    positions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = _weak_sample(price, traces, positions)
    z = sample["z"]
    y = sample["next_change"]
    weak = sample["weak"]
    simple = sample["simple_contribution"]
    actual = sample["actual_contribution"] / LIMIT
    rows: list[dict[str, object]] = []

    rows.append(_predictive_row("weak_all_simple_reversal", z, y, simple, weak))
    rows.append(_predictive_row("weak_all_actual_stateful", z, y, actual, weak))

    sign_groups = {
        "negative_deviation_long_reversal": weak & (z < 0),
        "positive_deviation_short_reversal": weak & (z > 0),
    }
    for label, mask in sign_groups.items():
        rows.append(_predictive_row(label, z, y, simple, mask))

    magnitude_groups = {
        "abs_z_0_to_0.5": weak & (np.abs(z) < 0.5),
        "abs_z_0.5_to_1": weak & (np.abs(z) >= 0.5) & (np.abs(z) < 1.0),
        "abs_z_1_to_1.5": weak & (np.abs(z) >= 1.0) & (np.abs(z) < 1.5),
        "abs_z_1.5_plus": weak & (np.abs(z) >= 1.5),
    }
    for label, mask in magnitude_groups.items():
        rows.append(_predictive_row(label, z, y, simple, mask))

    entry = weak.copy()
    entry[1:] = weak[1:] & (~weak[:-1])
    later = weak & ~entry
    rows.append(_predictive_row("weak_entry", z, y, simple, entry))
    rows.append(_predictive_row("weak_later", z, y, simple, later))

    for label, mask in (
        ("weak_long_reversal_signal", weak & (z < 0)),
        ("weak_short_reversal_signal", weak & (z > 0)),
    ):
        rows.append(_predictive_row(label, z, y, simple, mask))

    for kind, segment, start, end in _segment_definitions(len(price)):
        if kind not in ("quarter", "60_day"):
            continue
        segment_mask = weak.copy()
        segment_mask[:start] = False
        segment_mask[end:] = False
        rows.append(
            _predictive_row(
                "weak_simple_reversal",
                z,
                y,
                simple,
                segment_mask,
                {"Split": kind, "Segment": segment},
            )
        )
        rows.append(
            _predictive_row(
                "weak_actual_stateful",
                z,
                y,
                actual,
                segment_mask,
                {"Split": kind, "Segment": segment},
            )
        )

    # Interaction: next_change = a + b*z + c*weak + d*z*weak.
    valid = np.isfinite(z) & np.isfinite(y)
    weak_float = weak.astype(float)
    design = np.column_stack(
        [np.ones(len(z)), z, weak_float, z * weak_float]
    )
    interaction = _hac_ols_matrix(design[valid], y[valid], lags=5)
    interaction_rows = []
    labels = ["intercept", "ema_z_beta1", "weak_state_beta2", "interaction_beta3"]
    beta = interaction["beta"]
    se = interaction["se"]
    for label, b, s in zip(labels, beta, se):
        interaction_rows.append(
            {
                "Model": "next_change_on_ema_z_weak_interaction",
                "Coefficient": label,
                "Observations": interaction["n"],
                "Estimate": float(b),
                "HAC SE": float(s),
                "CI low": float(b - 1.96 * s),
                "CI high": float(b + 1.96 * s),
                "Normal p": _normal_p(float(b / s)) if s > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(interaction_rows)


def corrected_best_trade_exclusions(
    price: np.ndarray,
    traces: dict[str, dict[str, np.ndarray]],
    positions: dict[str, np.ndarray],
) -> pd.DataFrame:
    sample = _weak_sample(price, traces, positions)
    z = sample["z"]
    y = sample["next_change"]
    weak = sample["weak"]
    simple = sample["simple_contribution"]
    actual = sample["actual_contribution"] / LIMIT
    rows: list[dict[str, object]] = []

    for signal_name, contribution in (
        ("simple_reversal_contribution", simple),
        ("actual_stateful_position_contribution", actual),
    ):
        specifications: list[tuple[str, np.ndarray]] = [
            ("all_weak", np.ones(len(z), dtype=bool)),
        ]
        order = np.argsort(np.where(weak, contribution, -np.inf))
        for k in (1, 3, 5):
            keep = np.ones(len(z), dtype=bool)
            keep[order[-k:]] = False
            specifications.append((f"exclude_best_{k}_within_weak", keep))
        worst_order = np.argsort(np.where(weak, contribution, np.inf))
        keep_worst = np.ones(len(z), dtype=bool)
        keep_worst[worst_order[:5]] = False
        specifications.append(("exclude_worst_5_within_weak", keep_worst))
        for cap in (2.5, 4.0):
            clipped = np.clip(contribution, -cap, cap)
            mask = weak & np.isfinite(z)
            row = _predictive_row(
                signal_name,
                z,
                y,
                clipped,
                mask,
                {
                    "Test": f"winsorise_abs_{cap:g}",
                    "Contribution scale": "price-change units",
                    "Excluded observations": 0,
                },
            )
            rows.append(row)
        for label, keep in specifications:
            mask = weak & keep & np.isfinite(z)
            row = _predictive_row(
                signal_name,
                z,
                y,
                contribution,
                mask,
                {
                    "Test": label,
                    "Contribution scale": "price-change units",
                    "Excluded observations": int(np.sum(weak) - np.sum(mask)),
                },
            )
            rows.append(row)

        for kind, segment, start, end in _segment_definitions(len(price)):
            if kind not in ("quarter", "60_day"):
                continue
            mask = weak.copy()
            mask[start:end] = False
            rows.append(
                _predictive_row(
                    signal_name,
                    z,
                    y,
                    contribution,
                    mask,
                    {
                        "Test": "leave_one_chronological_block_out",
                        "Left_out_split": kind,
                        "Left_out_segment": segment,
                        "Contribution scale": "price-change units",
                        "Excluded observations": int(np.sum(weak & ~mask)),
                    },
                )
            )
    return pd.DataFrame(rows)


def _kalman_family_configs() -> list[dict[str, object]]:
    q_levels = (0.25, 0.5, 1.0, 2.0)
    q_slopes = (0.02, 0.05, 0.10)
    observations = (4.0, 5.0, 9.0, 16.0)
    thresholds = (0.5, 0.6, 0.75, 1.0, 1.25, 1.5)
    configs: list[dict[str, object]] = []
    bases = {
        "B_K2": (1.0, 0.05, 9.0, 1.0),
        "C_hybrid_params": (0.5, 0.05, 5.0, 0.6),
    }
    for base, (q_level, q_slope, observation, threshold) in bases.items():
        configs.append(
            {
                "Base": base,
                "Variation": "base",
                "q_level": q_level,
                "q_slope": q_slope,
                "R": observation,
                "threshold": threshold,
            }
        )
        for value in q_levels:
            configs.append(
                {
                    "Base": base,
                    "Variation": "q_level_ota",
                    "q_level": value,
                    "q_slope": q_slope,
                    "R": observation,
                    "threshold": threshold,
                }
            )
        for value in q_slopes:
            configs.append(
                {
                    "Base": base,
                    "Variation": "q_slope_ota",
                    "q_level": q_level,
                    "q_slope": value,
                    "R": observation,
                    "threshold": threshold,
                }
            )
        for value in observations:
            configs.append(
                {
                    "Base": base,
                    "Variation": "R_ota",
                    "q_level": q_level,
                    "q_slope": q_slope,
                    "R": value,
                    "threshold": threshold,
                }
            )
        for value in thresholds:
            configs.append(
                {
                    "Base": base,
                    "Variation": "threshold_ota",
                    "q_level": q_level,
                    "q_slope": q_slope,
                    "R": observation,
                    "threshold": value,
                }
            )
        for q_level_value in q_levels:
            for q_slope_value in q_slopes:
                configs.append(
                    {
                        "Base": base,
                        "Variation": "q_level_q_slope_joint",
                        "q_level": q_level_value,
                        "q_slope": q_slope_value,
                        "R": observation,
                        "threshold": threshold,
                    }
                )
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for config in configs:
        key = tuple(config[k] for k in ("Base", "q_level", "q_slope", "R", "threshold"))
        unique[key] = config
    return list(unique.values())


def kalman_family_table(
    price: np.ndarray,
    always_pnl: np.ndarray,
    k2_pnl: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for config in _kalman_family_configs():
        params = (float(config["q_level"]), float(config["q_slope"]), float(config["R"]))
        position, _ = _kalman_long_bias(
            price, params, float(config["threshold"]), strict=False
        )
        pnl = realized_pnl(price, position)
        quarters = [
            float(np.sum(pnl[start:end]))
            for kind, label, start, end in _segment_definitions(len(price))
            if kind == "quarter"
        ]
        halves = [
            float(np.sum(pnl[start:end]))
            for kind, label, start, end in _segment_definitions(len(price))
            if kind == "half"
        ]
        rows.append(
            {
                **config,
                "Configuration": (
                    f"qL={config['q_level']:g},qS={config['q_slope']:g},"
                    f"R={config['R']:g},threshold={config['threshold']:g}"
                ),
                "P&L": float(np.sum(pnl)),
                "Incremental vs always long": float(np.sum(pnl) - np.sum(always_pnl)),
                "Incremental vs K2": float(np.sum(pnl) - np.sum(k2_pnl)),
                "Positive quarters": int(sum(value > 0 for value in quarters)),
                "Q1": quarters[0],
                "Q2": quarters[1],
                "Q3": quarters[2],
                "Q4": quarters[3],
                "H1": halves[0],
                "H2": halves[1],
            }
        )
    table = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for group, frame in table.groupby(["Base", "Variation"]):
        summaries.append(
            {
                "Base": group[0],
                "Variation": group[1],
                "Configurations": int(len(frame)),
                "Median P&L": float(frame["P&L"].median()),
                "Lower decile P&L": float(frame["P&L"].quantile(0.10)),
                "Worst P&L": float(frame["P&L"].min()),
                "Best P&L": float(frame["P&L"].max()),
                "Median incremental vs K2": float(frame["Incremental vs K2"].median()),
                "Positive-quarter median": float(frame["Positive quarters"].median()),
                "Median H1": float(frame["H1"].median()),
                "Median H2": float(frame["H2"].median()),
            }
        )
    return table, pd.DataFrame(summaries)


def _effective_family(price: np.ndarray) -> dict[str, np.ndarray]:
    """Build a deduplicated main family, excluding the knowingly bad E."""

    positions, _ = _build_followup_candidates(price)
    family: dict[str, np.ndarray] = {
        name: positions[name]
        for name in positions
        if name != "original_hybrid"
    }
    for index, config in enumerate(_kalman_family_configs()):
        params = (float(config["q_level"]), float(config["q_slope"]), float(config["R"]))
        family[
            f"{config['Base']}_{config['Variation']}_{index}"
        ] = _kalman_long_bias(
            price, params, float(config["threshold"]), strict=False
        )[0]
    for alpha in (0.03, 0.06, 0.10, 0.20):
        for revert in (0.0, 0.2, 0.5, 1.0):
            trace = _hybrid_state_trace(price)
            ema = _causal_ema_trace(price, alpha, standardize_through_current=False)
            trace = {**trace, "ema_z": ema["z"]}
            pos = np.zeros(len(price), dtype=int)
            for day in range(1, len(price)):
                if trace["strong"][day]:
                    pos[day] = -LIMIT if trace["z"][day] <= -0.6 else LIMIT
                else:
                    pos[day] = _mean_reversion_transition(
                        int(pos[day - 1]), float(trace["ema_z"][day]), revert
                    )
            family[f"corrected_hybrid_alpha={alpha:g}_revert={revert:g}"] = pos
    for threshold in (0.4, 0.6, 0.8, 1.0, 1.5):
        trace = _hybrid_state_trace(price)
        pos = np.zeros(len(price), dtype=int)
        for day in range(1, len(price)):
            if abs(trace["z"][day]) >= threshold:
                pos[day] = -LIMIT if trace["z"][day] <= -threshold else LIMIT
            else:
                pos[day] = _mean_reversion_transition(
                    int(pos[day - 1]), float(trace["ema_z"][day]), 0.2
                )
        family[f"corrected_hybrid_threshold={threshold:g}"] = pos
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for position in family.values():
        unique.setdefault(tuple(position.tolist()), position)
    return {f"effective_{i:03d}": pos for i, pos in enumerate(unique.values())}


def _path_plausible(path: np.ndarray) -> bool:
    return bool(np.all(path > PLAUSIBLE_MIN) and np.all(path < PLAUSIBLE_MAX))


def _paired_summary(
    design: str,
    scenario: str,
    subset: str,
    candidate: str,
    reference: str,
    deltas: np.ndarray,
    mdd_deltas: np.ndarray,
    attempted: int,
    included: int,
    plausible_fraction: float,
) -> dict[str, object]:
    return {
        "Design": design,
        "Scenario": scenario,
        "Subset": subset,
        "Candidate": candidate,
        "Reference": reference,
        "Attempted paths": attempted,
        "Included paths": included,
        "Plausible path fraction": plausible_fraction,
        "Median paired difference": float(np.median(deltas)) if len(deltas) else float("nan"),
        "P10 paired difference": float(np.quantile(deltas, 0.10)) if len(deltas) else float("nan"),
        "P5 paired difference": float(np.quantile(deltas, 0.05)) if len(deltas) else float("nan"),
        "Worst paired difference": float(np.min(deltas)) if len(deltas) else float("nan"),
        "Win fraction": float(np.mean(deltas > 0)) if len(deltas) else float("nan"),
        "Win MC SE": (
            math.sqrt(np.mean(deltas > 0) * (1 - np.mean(deltas > 0)) / len(deltas))
            if len(deltas)
            else float("nan")
        ),
        "Median max-drawdown difference": float(np.median(mdd_deltas))
        if len(mdd_deltas)
        else float("nan"),
        "P5 max-drawdown difference": float(np.quantile(mdd_deltas, 0.05))
        if len(mdd_deltas)
        else float("nan"),
        "MDD improvement fraction": float(np.mean(mdd_deltas > 0))
        if len(mdd_deltas)
        else float("nan"),
    }


def _bootstrap_design(
    price: np.ndarray,
    candidates: dict[str, np.ndarray],
    design: str,
    scenarios: Iterable[tuple[str, np.ndarray]],
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_names = list(candidates)
    total_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for scenario, changes in scenarios:
        totals = {name: np.empty(repetitions, dtype=float) for name in candidate_names}
        mdds = {name: np.empty(repetitions, dtype=float) for name in candidate_names}
        plausible = np.zeros(repetitions, dtype=bool)
        for rep in range(repetitions):
            path = np.r_[price[0], price[0] + np.cumsum(changes[rep])]
            plausible[rep] = _path_plausible(path)
            path_positions = build_followup_candidates(path)
            for name in candidate_names:
                pnl = realized_pnl(path, path_positions[name])
                totals[name][rep] = float(np.sum(pnl))
                mdds[name][rep] = _max_drawdown(pnl)
        k2_values = totals["simple_K2"]
        k2_mdd = mdds["simple_K2"]
        always_values = totals["always_long"]
        for name in candidate_names:
            for subset, mask in (
                ("all_attempts", np.ones(repetitions, dtype=bool)),
                ("plausible_positive_range", plausible),
            ):
                values = totals[name][mask]
                if not len(values):
                    continue
                increments = values - always_values[mask]
                total_rows.append(
                    {
                        "Design": design,
                        "Scenario": scenario,
                        "Candidate": name,
                        "Subset": subset,
                        "Attempted paths": repetitions,
                        "Included paths": int(np.sum(mask)),
                        "Plausible path fraction": float(np.mean(plausible)),
                        "Positive P&L fraction": float(np.mean(values > 0)),
                        "Median P&L": float(np.median(values)),
                        "P5 P&L": float(np.quantile(values, 0.05)),
                        "P95 P&L": float(np.quantile(values, 0.95)),
                        "Median incremental vs always": float(np.median(increments)),
                        "P5 incremental vs always": float(np.quantile(increments, 0.05)),
                        "Median max drawdown": float(np.median(mdds[name][mask])),
                    }
                )
                for candidate, reference in (
                    ("hybrid_Kalman_no_EMA", "simple_K2"),
                    ("corrected_hybrid", "simple_K2"),
                    ("corrected_hybrid", "hybrid_Kalman_no_EMA"),
                ):
                    if name != candidate:
                        continue
                    deltas = totals[candidate][mask] - totals[reference][mask]
                    mdd_deltas = mdds[candidate][mask] - mdds[reference][mask]
                    paired_rows.append(
                        _paired_summary(
                            design,
                            scenario,
                            subset,
                            candidate,
                            reference,
                            deltas,
                            mdd_deltas,
                            repetitions,
                            int(np.sum(mask)),
                            float(np.mean(plausible)),
                        )
                    )
    return pd.DataFrame(total_rows), pd.DataFrame(paired_rows)


def additive_change_bootstrap(
    price: np.ndarray,
    candidates: dict[str, np.ndarray],
    repetitions: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 401)
    changes = np.diff(price)
    scenarios: list[tuple[str, np.ndarray]] = []
    for block in (5, 10, 20, 40, 60):
        sampled = np.asarray(
            [
                _circular_block_changes(changes, block, rng)
                for _ in range(repetitions)
            ]
        )
        scenarios.append((f"block_{block}", sampled))
    return _bootstrap_design(
        price,
        candidates,
        "additive_daily_change_circular_blocks",
        scenarios,
        repetitions,
    )


def drift_residual_bootstrap(
    price: np.ndarray,
    candidates: dict[str, np.ndarray],
    repetitions: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 402)
    changes = np.diff(price)
    drift = float(np.mean(changes))
    residual = changes - drift
    scenarios: list[tuple[str, np.ndarray]] = []
    for multiplier in (1.0, 0.75, 0.50, 0.0, 1.25):
        sampled = np.asarray(
            [
                _circular_block_changes(residual, 20, rng) * 1.0
                + multiplier * drift
                for _ in range(repetitions)
            ]
        )
        scenarios.append((f"drift_{multiplier:g}x", sampled))
    return _bootstrap_design(
        price,
        candidates,
        "drift_plus_residual_block_20",
        scenarios,
        repetitions,
    )


def percentage_bootstrap_secondary(
    price: np.ndarray,
    candidates: dict[str, np.ndarray],
    repetitions: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 403)
    returns = np.diff(price) / price[:-1]
    rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for block in (10, 20):
        totals = {name: np.empty(repetitions) for name in candidates}
        mdds = {name: np.empty(repetitions) for name in candidates}
        for rep in range(repetitions):
            sampled = _circular_block_changes(returns, block, rng)
            path = np.r_[price[0], price[0] * np.cumprod(1.0 + sampled)]
            path_positions = build_followup_candidates(path)
            for name in candidates:
                pnl = realized_pnl(path, path_positions[name])
                totals[name][rep] = np.sum(pnl)
                mdds[name][rep] = _max_drawdown(pnl)
        for candidate, reference in (
            ("hybrid_Kalman_no_EMA", "simple_K2"),
            ("corrected_hybrid", "simple_K2"),
            ("corrected_hybrid", "hybrid_Kalman_no_EMA"),
        ):
            delta = totals[candidate] - totals[reference]
            mdd_delta = mdds[candidate] - mdds[reference]
            pair_rows.append(
                _paired_summary(
                    "percentage_return_secondary",
                    f"block_{block}",
                    "all_attempts",
                    candidate,
                    reference,
                    delta,
                    mdd_delta,
                    repetitions,
                    repetitions,
                    1.0,
                )
            )
        for name in candidates:
            values = totals[name]
            rows.append(
                {
                    "Design": "percentage_return_secondary",
                    "Scenario": f"block_{block}",
                    "Candidate": name,
                    "Repetitions": repetitions,
                    "Median P&L": float(np.median(values)),
                    "P5 P&L": float(np.quantile(values, 0.05)),
                    "Positive fraction": float(np.mean(values > 0)),
                    "Median max drawdown": float(np.median(mdds[name])),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def _regime_path(
    n: int,
    positive_duration: int,
    negative_duration: int,
    drift_per_day: float,
    noise_scale: float,
    gradual: bool,
    seed: int,
    state_sequence: tuple[int, ...] = (1, -1),
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base_noise = float(np.std(np.diff(load_price_series()), ddof=1))
    changes: list[float] = []
    state_index = 0
    while len(changes) < n - 1:
        state = state_sequence[state_index % len(state_sequence)]
        duration = positive_duration if state > 0 else negative_duration
        duration = min(duration, n - 1 - len(changes))
        noise = rng.normal(0.0, base_noise * noise_scale, duration)
        target = state * drift_per_day
        if gradual and len(changes) > 0:
            previous_state = state_sequence[(state_index - 1) % len(state_sequence)]
            prior_target = previous_state * drift_per_day
            blend = np.linspace(0.0, 1.0, duration, endpoint=False)
            target_series = (1.0 - blend) * prior_target + blend * target
            changes.extend((target_series + noise).tolist())
        else:
            changes.extend((target + noise).tolist())
        state_index += 1
    return np.asarray(changes[: n - 1], dtype=float)


def regime_stress_table(
    price: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = {
        "persistent_positive": dict(positive_duration=80, negative_duration=30, drift=0.20, noise=0.70, gradual=False, seed=501),
        "balanced_40_day_regimes": dict(positive_duration=40, negative_duration=40, drift=0.18, noise=0.75, gradual=False, seed=502),
        "long_negative_regimes": dict(positive_duration=50, negative_duration=90, drift=0.18, noise=0.75, gradual=False, seed=503),
        "short_reversal_runs": dict(positive_duration=10, negative_duration=10, drift=0.20, noise=0.70, gradual=False, seed=504),
        "gradual_transitions": dict(positive_duration=60, negative_duration=60, drift=0.18, noise=0.70, gradual=True, seed=505),
        "high_noise": dict(positive_duration=60, negative_duration=60, drift=0.18, noise=1.50, gradual=False, seed=506),
        "low_noise": dict(positive_duration=60, negative_duration=60, drift=0.18, noise=0.50, gradual=False, seed=507),
        "more_frequent_reversals": dict(positive_duration=20, negative_duration=20, drift=0.14, noise=1.00, gradual=True, seed=508),
    }
    rows: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for scenario, spec in scenarios.items():
        changes = _regime_path(
            len(price),
            spec["positive_duration"],
            spec["negative_duration"],
            spec["drift"],
            spec["noise"],
            spec["gradual"],
            spec["seed"],
        )
        path = np.r_[price[0], price[0] + np.cumsum(changes)]
        path_positions = build_followup_candidates(path)
        plausible = _path_plausible(path)
        totals: dict[str, float] = {}
        mdds: dict[str, float] = {}
        for name in candidates:
            pnl = realized_pnl(path, path_positions[name])
            totals[name] = float(np.sum(pnl))
            mdds[name] = _max_drawdown(pnl)
        for name in candidates:
            rows.append(
                {
                    "Design": "regime_preserving_generator_conditioned",
                    "Scenario": scenario,
                    "Candidate": name,
                    "P&L": totals[name],
                    "Incremental vs always long": totals[name]
                    - totals["always_long"],
                    "Incremental vs K2": totals[name] - totals["simple_K2"],
                    "Maximum drawdown": mdds[name],
                    "Plausible positive price range": plausible,
                    "Positive duration": spec["positive_duration"],
                    "Negative duration": spec["negative_duration"],
                    "Drift per day": spec["drift"],
                    "Noise scale": spec["noise"],
                    "Gradual transition": spec["gradual"],
                }
            )
        for candidate, reference in (
            ("hybrid_Kalman_no_EMA", "simple_K2"),
            ("corrected_hybrid", "simple_K2"),
            ("corrected_hybrid", "hybrid_Kalman_no_EMA"),
        ):
            pairs.append(
                _paired_summary(
                    "regime_preserving_generator_conditioned",
                    scenario,
                    "single_path",
                    candidate,
                    reference,
                    np.asarray([totals[candidate] - totals[reference]]),
                    np.asarray([mdds[candidate] - mdds[reference]]),
                    1,
                    1,
                    float(plausible),
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def familywise_diagnostics(
    price: np.ndarray,
    repetitions: int = 200,
) -> pd.DataFrame:
    observed_family = _effective_family(price)
    observed_totals = {
        name: float(np.sum(realized_pnl(price, position)))
        for name, position in observed_family.items()
    }
    observed_name = max(observed_totals, key=observed_totals.get)
    observed_best = observed_totals[observed_name]
    rng = np.random.default_rng(SEED + 404)
    changes = np.diff(price)
    maxima = np.empty(repetitions, dtype=float)
    plausible = np.zeros(repetitions, dtype=bool)
    for rep in range(repetitions):
        sampled = _circular_block_changes(changes, 10, rng)
        path = np.r_[price[0], price[0] + np.cumsum(sampled)]
        plausible[rep] = _path_plausible(path)
        family = _effective_family(path)
        maxima[rep] = max(
            float(np.sum(realized_pnl(path, position)))
            for position in family.values()
        )
    rows: list[dict[str, object]] = []
    for subset, mask in (
        ("all_attempts", np.ones(repetitions, dtype=bool)),
        ("plausible_positive_range", plausible),
    ):
        values = maxima[mask]
        if not len(values):
            continue
        hits = int(np.sum(values >= observed_best))
        p_value = (1.0 + hits) / (len(values) + 1.0)
        se = math.sqrt(p_value * (1.0 - p_value) / len(values))
        rows.append(
            {
                "Family": "deduplicated_corrected_main_family",
                "Null": "circular_absolute_change_blocks_10",
                "Subset": subset,
                "Unique observed configurations": len(observed_family),
                "Observed best candidate": observed_name,
                "Observed best P&L": observed_best,
                "Attempted paths": repetitions,
                "Included paths": int(len(values)),
                "Plausible path fraction": float(np.mean(plausible)),
                "Null median family maximum": float(np.median(values)),
                "Null 95 family maximum": float(np.quantile(values, 0.95)),
                "Exceedances": hits,
                "Monte Carlo p-value": p_value,
                "Monte Carlo SE": se,
                "MC interval low": max(0.0, p_value - 1.96 * se),
                "MC interval high": min(1.0, p_value + 1.96 * se),
            }
        )
    return pd.DataFrame(rows)


def paired_daily_tables(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
    k2_pnl: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for name, position in positions.items():
        if name == "simple_K2":
            continue
        pnl = realized_pnl(price, position)
        delta = pnl - k2_pnl
        equity = np.cumsum(delta)
        daily_rows.extend(
            {
                "Candidate": name,
                "Day": day,
                "Candidate P&L": float(pnl[day]),
                "K2 P&L": float(k2_pnl[day]),
                "Paired difference": float(delta[day]),
                "Cumulative paired difference": float(equity[day]),
            }
            for day in range(len(price))
        )
        segment_values: list[float] = []
        for kind, label, start, end in _segment_definitions(len(price)):
            if kind in ("quarter", "60_day", "half", "early_middle_late"):
                segment_values.append(float(np.sum(delta[start:end])))
                summary_rows.append(
                    {
                        "Candidate": name,
                        "Comparison": "vs_K2",
                        "Split": kind,
                        "Segment": label,
                        "Paired advantage": float(np.sum(delta[start:end])),
                        "Segment won": bool(np.sum(delta[start:end]) > 0),
                    }
                )
        summary_rows.append(
            {
                "Candidate": name,
                "Comparison": "vs_K2",
                "Split": "full_summary",
                "Segment": "all",
                "Total paired advantage": float(np.sum(delta)),
                "Segments won": int(sum(value > 0 for value in segment_values)),
                "Segments tested": len(segment_values),
                "Percent segments won": float(np.mean(np.asarray(segment_values) > 0)),
                "Max drawdown paired curve": _max_drawdown(delta),
                "Best 1 paired-day share": _top_share(delta, float(np.sum(delta)), 1),
                "Best 3 paired-day share": _top_share(delta, float(np.sum(delta)), 3),
                "Best 5 paired-day share": _top_share(delta, float(np.sum(delta)), 5),
                "Best 10 paired-day share": _top_share(delta, float(np.sum(delta)), 10),
                "Best 20 paired-day share": _top_share(delta, float(np.sum(delta)), 20),
            }
        )
    return pd.DataFrame(daily_rows), pd.DataFrame(summary_rows)


def correctness_table(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    audit_days = (0, 30, 120, 240, 300)
    for name, position in positions.items():
        checks = {
            "integer_positions": bool(np.issubdtype(position.dtype, np.integer)),
            "within_Jeans_limit": bool(np.max(np.abs(position)) <= LIMIT),
            "within_standalone_notional": bool(
                np.max(np.abs(position) * price) <= JEANS_BUDGET
            ),
            "day_zero_flat": bool(position[0] == 0),
            "prior_position_P&L": bool(
                np.allclose(
                    realized_pnl(price, position),
                    np.r_[0.0, position[:-1] * np.diff(price)],
                    atol=0.011,
                )
            ),
            "final_day_has_no_future_return": bool(
                len(realized_pnl(price, position)) == len(price)
            ),
        }
        prefix_ok = True
        future_ok = True
        for day in audit_days:
            prefix = build_followup_candidates(price[: day + 1])[name]
            prefix_ok &= bool(prefix[-1] == position[day])
            altered = price.copy()
            altered[day + 1 :] += 1000.0
            altered_position = build_followup_candidates(altered)[name]
            future_ok &= bool(altered_position[day] == position[day])
        checks["prefix_replay_matches"] = prefix_ok
        checks["future_perturbation_unchanged"] = future_ok
        for check, value in checks.items():
            rows.append(
                {
                    "Candidate": name,
                    "Check": check,
                    "Value": bool(value),
                    "Details": str(audit_days) if check in ("prefix_replay_matches", "future_perturbation_unchanged") else "",
                }
            )
    return pd.DataFrame(rows)


def _write_figures(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
    paired_daily: pd.DataFrame,
    weak_predictive: pd.DataFrame,
    kalman_family: pd.DataFrame,
    paired_bootstrap: pd.DataFrame,
    traces: dict[str, dict[str, np.ndarray]],
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name in ("always_long", "simple_K2", "hybrid_Kalman_no_EMA", "corrected_hybrid"):
        ax.plot(np.cumsum(realized_pnl(price, positions[name])), label=name)
    ax.set_title("Thrifted Jeans follow-up: cumulative P&L")
    ax.set_xlabel("Day")
    ax.set_ylabel("AUD P&L")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "cumulative_followup_pnl.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name in ("hybrid_Kalman_no_EMA", "corrected_hybrid"):
        frame = paired_daily[paired_daily["Candidate"] == name]
        ax.plot(
            frame["Day"],
            frame["Cumulative paired difference"],
            label=f"{name} minus K2",
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Cumulative paired advantage over simple K2")
    ax.set_xlabel("Day")
    ax.set_ylabel("AUD paired P&L difference")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "paired_advantage_over_k2.png", dpi=160)
    plt.close(fig)

    weak = weak_predictive[
        (weak_predictive["Signal"].isin(["weak_simple_reversal", "weak_actual_stateful"]))
        & (weak_predictive.get("Split", pd.Series(dtype=object)) == "quarter")
    ]
    if len(weak):
        pivot = weak.pivot(index="Segment", columns="Signal", values="Mean contribution")
        fig, ax = plt.subplots(figsize=(9, 4))
        pivot.plot(kind="bar", ax=ax)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("Corrected weak-state contribution by quarter")
        ax.set_ylabel("Mean contribution (price-change units)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figure_dir / "weak_state_contribution_by_quarter.png", dpi=160)
        plt.close(fig)

    joint = kalman_family[
        (kalman_family["Variation"] == "q_level_q_slope_joint")
        & (kalman_family["Base"] == "B_K2")
    ]
    pivot = joint.pivot(index="q_level", columns="q_slope", values="P&L")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{x:g}" for x in pivot.index])
    ax.set_xlabel("q_slope")
    ax.set_ylabel("q_level")
    ax.set_title("Simple Kalman family P&L surfaces")
    fig.colorbar(im, ax=ax, label="AUD P&L")
    fig.tight_layout()
    fig.savefig(figure_dir / "simple_kalman_parameter_stability.png", dpi=160)
    plt.close(fig)

    pair = paired_bootstrap[
        (paired_bootstrap["Subset"] == "plausible_positive_range")
        & (paired_bootstrap["Design"] == "additive_daily_change_circular_blocks")
    ]
    if len(pair):
        fig, ax = plt.subplots(figsize=(9, 4))
        for candidate, color in (
            ("hybrid_Kalman_no_EMA", "#1f77b4"),
            ("corrected_hybrid", "#d62728"),
        ):
            frame = pair[
                (pair["Candidate"] == candidate)
                & (pair["Reference"] == "simple_K2")
            ]
            ax.plot(
                frame["Scenario"],
                frame["Median paired difference"],
                marker="o",
                label=f"{candidate} minus K2",
                color=color,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("Paired bootstrap differences on plausible additive paths")
        ax.set_xlabel("Block design")
        ax.set_ylabel("Median paired P&L difference")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / "paired_bootstrap_differences.png", dpi=160)
        plt.close(fig)

    trace = traces["hybrid"]
    z = trace["z"][:-1]
    y = np.diff(price)
    mask = np.isfinite(z)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(z[mask], y[mask], s=10, alpha=0.45)
    ax.axvline(-0.6, color="red", linestyle="--", linewidth=0.8)
    ax.axvline(0.6, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Hybrid Kalman slope z-score versus next-day Jeans change")
    ax.set_xlabel("Filtered slope z-score")
    ax.set_ylabel("Next-day price change")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "slope_z_against_next_day_change.png", dpi=160)
    plt.close(fig)


def run_followup(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    additive_repetitions: int | None = None,
    drift_repetitions: int | None = None,
    percentage_repetitions: int | None = None,
    familywise_repetitions: int | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    additive_repetitions = additive_repetitions or int(
        os.environ.get("JEANS_FOLLOWUP_ADDITIVE_REPS", "200")
    )
    drift_repetitions = drift_repetitions or int(
        os.environ.get("JEANS_FOLLOWUP_DRIFT_REPS", "120")
    )
    percentage_repetitions = percentage_repetitions or int(
        os.environ.get("JEANS_FOLLOWUP_PERCENTAGE_REPS", "120")
    )
    familywise_repetitions = familywise_repetitions or int(
        os.environ.get("JEANS_FOLLOWUP_FAMILYWISE_REPS", "200")
    )

    price = load_price_series()
    positions, traces = _build_followup_candidates(price)
    always_reference_position = np.full(len(price), LIMIT, dtype=int)
    always_reference_pnl = realized_pnl(price, always_reference_position)
    k2_pnl = realized_pnl(price, positions["simple_K2"])
    strong = traces["hybrid"]["strong"]

    comparison_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    for name, position in positions.items():
        row = {"Candidate": name}
        row.update(_metrics(price, position, always_reference_pnl, k2_pnl))
        row.update(_state_attribution(price, position, strong))
        comparison_rows.append(row)
        if name not in ("always_long", "simple_K2", "original_hybrid"):
            ablation_rows.append(row)
    comparison = pd.DataFrame(comparison_rows).set_index("Candidate")
    ablations = pd.DataFrame(ablation_rows)
    comparison.to_csv(output_dir / "exact_candidate_comparison.csv")
    ablations.to_csv(output_dir / "weak_state_ablations.csv", index=False)

    chronological = _chronological_table(
        price, positions, always_reference_pnl, k2_pnl
    )
    chronological.to_csv(output_dir / "chronological_segments.csv", index=False)
    exclusions = _best_day_exclusions(
        price, positions, always_reference_pnl, k2_pnl
    )
    exclusions.to_csv(output_dir / "best_realised_day_exclusions.csv", index=False)

    paired_daily, paired_segments = paired_daily_tables(price, positions, k2_pnl)
    paired_daily.to_csv(output_dir / "paired_daily_comparisons.csv", index=False)
    paired_segments.to_csv(output_dir / "paired_segment_comparisons.csv", index=False)

    weak_predictive, interaction = weak_state_predictive_table(
        price, traces, positions
    )
    weak_predictive.to_csv(output_dir / "corrected_weak_state_predictive.csv", index=False)
    interaction.to_csv(output_dir / "weak_state_interaction_regression.csv", index=False)
    corrected_exclusions = corrected_best_trade_exclusions(price, traces, positions)
    corrected_exclusions.to_csv(
        output_dir / "corrected_best_trade_exclusions.csv", index=False
    )

    kalman_family, kalman_summary = kalman_family_table(
        price, always_reference_pnl, k2_pnl
    )
    kalman_family.to_csv(output_dir / "simple_kalman_family.csv", index=False)
    kalman_summary.to_csv(
        output_dir / "simple_kalman_family_summary.csv", index=False
    )

    additive, additive_pairs = additive_change_bootstrap(
        price, positions, additive_repetitions
    )
    additive.to_csv(output_dir / "additive_change_bootstrap.csv", index=False)

    drift, drift_pairs = drift_residual_bootstrap(
        price, positions, drift_repetitions
    )
    drift.to_csv(output_dir / "drift_residual_bootstrap.csv", index=False)

    percentage, percentage_pairs = percentage_bootstrap_secondary(
        price, positions, percentage_repetitions
    )
    percentage.to_csv(
        output_dir / "percentage_return_bootstrap_secondary.csv", index=False
    )

    regime, regime_pairs = regime_stress_table(price, positions)
    regime.to_csv(output_dir / "regime_preserving_stress.csv", index=False)

    paired_bootstrap = pd.concat(
        [additive_pairs, drift_pairs, percentage_pairs, regime_pairs],
        ignore_index=True,
    )
    paired_bootstrap.to_csv(
        output_dir / "paired_bootstrap_differences.csv", index=False
    )

    familywise = familywise_diagnostics(price, familywise_repetitions)
    familywise.to_csv(output_dir / "familywise_diagnostics.csv", index=False)

    correctness = correctness_table(price, positions)
    correctness.to_csv(output_dir / "correctness_checks.csv", index=False)

    _write_figures(
        price,
        positions,
        paired_daily,
        weak_predictive,
        kalman_family,
        paired_bootstrap,
        traces,
        figure_dir,
    )

    manifest = {
        "data_path": str(DATA_PATH),
        "observations": int(len(price)),
        "start_price": float(price[0]),
        "end_price": float(price[-1]),
        "daily_change_autocorrelation": float(pd.Series(np.diff(price)).autocorr(1)),
        "base_candidate_count": len(positions),
        "kalman_family_configuration_count": len(kalman_family),
        "effective_main_family_unique_count": int(
            len(_effective_family(price))
        ),
        "additive_repetitions_per_block": int(additive_repetitions),
        "additive_blocks": [5, 10, 20, 40, 60],
        "drift_repetitions_per_assumption": int(drift_repetitions),
        "percentage_secondary_repetitions_per_block": int(percentage_repetitions),
        "familywise_repetitions": int(familywise_repetitions),
        "plausible_price_range": [PLAUSIBLE_MIN, PLAUSIBLE_MAX],
        "production_files_modified": False,
        "reference_pnls": {
            "always_long_strict_flat_day0": float(
                comparison.loc["always_long", "P&L"]
            ),
            "always_long_full_reference": float(np.sum(always_reference_pnl)),
            "simple_K2": float(comparison.loc["simple_K2", "P&L"]),
            "hybrid_Kalman_no_EMA": float(
                comparison.loc["hybrid_Kalman_no_EMA", "P&L"]
            ),
            "corrected_hybrid": float(comparison.loc["corrected_hybrid", "P&L"]),
            "original_hybrid": float(comparison.loc["original_hybrid", "P&L"]),
        },
        "notes": [
            "The stated flat-day0 Candidate A is AUD 37,760 under the simulator prior-position convention; AUD 38,136 is the full +800 day0 reference.",
            "Candidate C uses the hybrid Kalman parameters but no EMA or weak-state branch.",
            "The corrected weak sample is held fixed for all exclusion tests.",
            "Additive bootstrap paths are never shifted upward; implausible paths are reported separately.",
            "The percentage-return bootstrap is retained as a secondary scale-sensitive diagnostic.",
            "The main family excludes the knowingly incorrect original EMA residual variant.",
        ],
    }
    (output_dir / "followup_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Thrifted Jeans follow-up audit complete")
    print(
        comparison[
            ["P&L", "Incremental vs always long", "Incremental vs K2", "Max drawdown", "Turnover"]
        ].round(2).to_string()
    )
    print(f"Additive bootstrap repetitions per block: {additive_repetitions}")
    print(f"Family-wise repetitions: {familywise_repetitions}")
    return {
        "price": price,
        "positions": positions,
        "traces": traces,
        "comparison": comparison,
        "ablations": ablations,
        "chronological": chronological,
        "exclusions": exclusions,
        "paired_daily": paired_daily,
        "paired_segments": paired_segments,
        "weak_predictive": weak_predictive,
        "interaction": interaction,
        "corrected_exclusions": corrected_exclusions,
        "kalman_family": kalman_family,
        "kalman_summary": kalman_summary,
        "additive": additive,
        "drift": drift,
        "percentage": percentage,
        "regime": regime,
        "paired_bootstrap": paired_bootstrap,
        "familywise": familywise,
        "correctness": correctness,
        "manifest": manifest,
    }


if __name__ == "__main__":
    run_followup()
