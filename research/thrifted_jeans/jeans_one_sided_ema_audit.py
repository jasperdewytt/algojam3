"""Third-stage research audit for a one-sided Jeans EMA overlay.

This module is deliberately separate from the first two audits.  It imports
only research helpers and writes to one_sided_outputs/ and one_sided_figures/.
No production strategy, simulator, or supplied data file is modified.
"""

from __future__ import annotations

import hashlib
import json
import math
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
    _circular_block_changes,
    _kalman_trace,
    _max_drawdown,
    _segment_definitions,
    hac_regression,
    load_price_series,
    realized_pnl,
)
from jeans_followup_audit import build_followup_candidates, build_followup_traces


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "one_sided_outputs"
DEFAULT_FIGURE_DIR = Path(__file__).resolve().parent / "one_sided_figures"

K2_PARAMS = (1.0, 0.05, 9.0)
C_PARAMS = (0.5, 0.05, 5.0)
KALMAN_THRESHOLD = 0.6
PRIMARY_ALPHA = 0.06
PRIMARY_WINDOW = 20
PRIMARY_ENTRY_THRESHOLD = 0.5
EMA_ALPHAS = (0.03, 0.06, 0.10, 0.20)
RESIDUAL_WINDOWS = (10, 20, 30, 45)
ENTRY_THRESHOLDS = (0.0, 0.25, 0.5, 0.75, 1.0)
HYSTERESIS_EXITS = (0.0, 0.25)
PLAUSIBLE_MIN = 1.0
PLAUSIBLE_MAX = 200.0

CORE_NAMES = (
    "A_always_long",
    "B_K2",
    "C_candidate_C",
    "D_corrected_hybrid_frozen",
    "one_sided_primary",
    "one_sided_hysteresis_exit0",
    "one_sided_hysteresis_exit025",
    "no_short_long_flat",
    "reduced_weak_long_200",
    "two_sided_stateless",
    "uncentred_one_sided_diagnostic",
)

PAIR_SPECS = (
    ("one_sided_primary", "C_candidate_C"),
    ("one_sided_primary", "B_K2"),
    ("D_corrected_hybrid_frozen", "one_sided_primary"),
    ("no_short_long_flat", "one_sided_primary"),
)


def _path_id(position: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(position, dtype=np.int16).tobytes()).hexdigest()[:12]


def rolling_prediction_residual_trace(
    price: np.ndarray,
    alpha: float,
    window: int,
    min_history: int | None = None,
) -> dict[str, np.ndarray]:
    """Causal one-step EMA residuals and prior-residual rolling z-scores.

    At day t, prior_ema[t] is the EMA through t-1.  The residual at t is
    formed before today's EMA update, and today's scale uses only residuals
    strictly before t.  With min_history omitted, all `window` prior
    residuals are required.
    """

    price = np.asarray(price, dtype=float)
    n = len(price)
    prior_ema = np.full(n, np.nan, dtype=float)
    ema = np.full(n, np.nan, dtype=float)
    residual = np.full(n, np.nan, dtype=float)
    residual_mean = np.full(n, np.nan, dtype=float)
    residual_std = np.full(n, np.nan, dtype=float)
    z_centered = np.full(n, np.nan, dtype=float)
    z_uncentred = np.full(n, np.nan, dtype=float)
    eligible = np.zeros(n, dtype=bool)
    if n == 0:
        return {
            "prior_ema": prior_ema,
            "ema": ema,
            "residual": residual,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "z_centered": z_centered,
            "z_uncentred": z_uncentred,
            "eligible": eligible,
        }

    fair = float(price[0])
    ema[0] = fair
    prior_ema[0] = fair
    history: list[float] = []
    required = int(window if min_history is None else min_history)
    if required < 2:
        raise ValueError("min_history must be at least 2 for ddof=1 scale")

    for day in range(1, n):
        prior_ema[day] = fair
        current_residual = float(price[day] - fair)
        residual[day] = current_residual
        if len(history) >= required:
            scale_history = np.asarray(history[-window:], dtype=float)
            mean = float(np.mean(scale_history))
            scale = float(np.std(scale_history, ddof=1))
            residual_mean[day] = mean
            residual_std[day] = scale
            if np.isfinite(scale) and scale > 1e-12:
                eligible[day] = True
                z_centered[day] = (current_residual - mean) / scale
                z_uncentred[day] = current_residual / scale
        fair = fair + float(alpha) * current_residual
        ema[day] = fair
        history.append(current_residual)

    return {
        "prior_ema": prior_ema,
        "ema": ema,
        "residual": residual,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "z_centered": z_centered,
        "z_uncentred": z_uncentred,
        "eligible": eligible,
    }


def _kalman_state(price: np.ndarray) -> dict[str, np.ndarray]:
    trace = _kalman_trace(price, C_PARAMS)
    z = trace["z"]
    return {
        **trace,
        "strong_positive": z >= KALMAN_THRESHOLD,
        "strong_negative": z <= -KALMAN_THRESHOLD,
        "weak": np.abs(z) < KALMAN_THRESHOLD,
    }


def _stateless_one_sided_position(
    kalman: dict[str, np.ndarray],
    residual: dict[str, np.ndarray],
    entry_threshold: float,
    centered: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    z = residual["z_centered"] if centered else residual["z_uncentred"]
    position = np.zeros(len(z), dtype=int)
    weak_dip = np.zeros(len(z), dtype=bool)
    for day in range(1, len(z)):
        if kalman["strong_positive"][day]:
            position[day] = LIMIT
        elif kalman["strong_negative"][day]:
            position[day] = -LIMIT
        elif np.isfinite(z[day]) and z[day] <= entry_threshold * -1.0:
            position[day] = LIMIT
            weak_dip[day] = True
        else:
            position[day] = 0
    return position, weak_dip


def _hysteresis_one_sided_position(
    kalman: dict[str, np.ndarray],
    residual: dict[str, np.ndarray],
    entry_threshold: float,
    exit_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    z = residual["z_centered"]
    position = np.zeros(len(z), dtype=int)
    weak_dip = np.zeros(len(z), dtype=bool)
    for day in range(1, len(z)):
        if kalman["strong_positive"][day]:
            position[day] = LIMIT
            continue
        if kalman["strong_negative"][day]:
            position[day] = -LIMIT
            continue

        previous_long = position[day - 1] > 0
        if np.isfinite(z[day]) and z[day] <= -entry_threshold:
            position[day] = LIMIT
            weak_dip[day] = True
        elif np.isfinite(z[day]) and z[day] >= exit_threshold:
            position[day] = 0
        else:
            position[day] = LIMIT if previous_long else 0
    return position, weak_dip


def _no_short_position(
    kalman: dict[str, np.ndarray],
    residual: dict[str, np.ndarray],
    entry_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    z = residual["z_centered"]
    position = np.zeros(len(z), dtype=int)
    weak_dip = np.zeros(len(z), dtype=bool)
    for day in range(1, len(z)):
        if kalman["strong_positive"][day]:
            position[day] = LIMIT
        elif kalman["strong_negative"][day]:
            position[day] = 0
        elif np.isfinite(z[day]) and z[day] <= -entry_threshold:
            position[day] = LIMIT
            weak_dip[day] = True
    return position, weak_dip


def _reduced_weak_position(
    kalman: dict[str, np.ndarray],
    residual: dict[str, np.ndarray],
    entry_threshold: float,
    weak_long: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    z = residual["z_centered"]
    position = np.zeros(len(z), dtype=int)
    weak_dip = np.zeros(len(z), dtype=bool)
    for day in range(1, len(z)):
        if kalman["strong_positive"][day]:
            position[day] = LIMIT
        elif kalman["strong_negative"][day]:
            position[day] = -LIMIT
        elif np.isfinite(z[day]) and z[day] <= -entry_threshold:
            position[day] = LIMIT
            weak_dip[day] = True
        else:
            position[day] = int(weak_long)
    return position, weak_dip


def _two_sided_stateless_position(
    kalman: dict[str, np.ndarray],
    residual: dict[str, np.ndarray],
    entry_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    z = residual["z_centered"]
    position = np.zeros(len(z), dtype=int)
    weak_dip = np.zeros(len(z), dtype=bool)
    for day in range(1, len(z)):
        if kalman["strong_positive"][day]:
            position[day] = LIMIT
        elif kalman["strong_negative"][day]:
            position[day] = -LIMIT
        elif np.isfinite(z[day]) and z[day] <= -entry_threshold:
            position[day] = LIMIT
            weak_dip[day] = True
        elif np.isfinite(z[day]) and z[day] >= entry_threshold:
            position[day] = -LIMIT
        else:
            position[day] = 0
    return position, weak_dip


def _build_one_sided_candidates(
    price: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
]:
    """Return positions, causal traces, and weak-dip masks for core candidates."""

    price = np.asarray(price, dtype=float)
    old_positions = build_followup_candidates(price)
    old_traces = build_followup_traces(price)
    kalman = _kalman_state(price)
    primary_residual = rolling_prediction_residual_trace(
        price, PRIMARY_ALPHA, PRIMARY_WINDOW
    )
    positions: dict[str, np.ndarray] = {
        "A_always_long": old_positions["always_long"].copy(),
        "B_K2": old_positions["simple_K2"].copy(),
        "C_candidate_C": old_positions["hybrid_Kalman_no_EMA"].copy(),
        "D_corrected_hybrid_frozen": old_positions["corrected_hybrid"].copy(),
    }
    dip_masks: dict[str, np.ndarray] = {
        name: np.zeros(len(price), dtype=bool) for name in positions
    }

    primary, primary_dip = _stateless_one_sided_position(
        kalman, primary_residual, PRIMARY_ENTRY_THRESHOLD, centered=True
    )
    positions["one_sided_primary"] = primary
    dip_masks["one_sided_primary"] = primary_dip

    hysteresis0, hysteresis0_dip = _hysteresis_one_sided_position(
        kalman, primary_residual, PRIMARY_ENTRY_THRESHOLD, HYSTERESIS_EXITS[0]
    )
    positions["one_sided_hysteresis_exit0"] = hysteresis0
    dip_masks["one_sided_hysteresis_exit0"] = hysteresis0_dip

    hysteresis25, hysteresis25_dip = _hysteresis_one_sided_position(
        kalman, primary_residual, PRIMARY_ENTRY_THRESHOLD, HYSTERESIS_EXITS[1]
    )
    positions["one_sided_hysteresis_exit025"] = hysteresis25
    dip_masks["one_sided_hysteresis_exit025"] = hysteresis25_dip

    no_short, no_short_dip = _no_short_position(
        kalman, primary_residual, PRIMARY_ENTRY_THRESHOLD
    )
    positions["no_short_long_flat"] = no_short
    dip_masks["no_short_long_flat"] = no_short_dip

    reduced, reduced_dip = _reduced_weak_position(
        kalman, primary_residual, PRIMARY_ENTRY_THRESHOLD, weak_long=200
    )
    positions["reduced_weak_long_200"] = reduced
    dip_masks["reduced_weak_long_200"] = reduced_dip

    two_sided, two_sided_dip = _two_sided_stateless_position(
        kalman, primary_residual, PRIMARY_ENTRY_THRESHOLD
    )
    positions["two_sided_stateless"] = two_sided
    dip_masks["two_sided_stateless"] = two_sided_dip

    uncentred, uncentred_dip = _stateless_one_sided_position(
        kalman, primary_residual, PRIMARY_ENTRY_THRESHOLD, centered=False
    )
    positions["uncentred_one_sided_diagnostic"] = uncentred
    dip_masks["uncentred_one_sided_diagnostic"] = uncentred_dip

    traces: dict[str, dict[str, np.ndarray]] = {
        "kalman_C": kalman,
        "rolling_primary": primary_residual,
        "legacy_hybrid": old_traces["hybrid"],
    }
    for name, position in positions.items():
        positions[name] = np.asarray(position, dtype=int)
    return positions, traces, dip_masks


def build_one_sided_candidates(price: np.ndarray) -> dict[str, np.ndarray]:
    return _build_one_sided_candidates(price)[0]


def _loss_streak(pnl: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in pnl:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _top_share(pnl: np.ndarray, total: float, k: int) -> float:
    if not total:
        return float("nan")
    return float(np.sort(pnl)[-k:].sum() / total)


def _realized_state_mask(state: np.ndarray) -> np.ndarray:
    return np.r_[False, state[:-1]]


def _metrics(
    price: np.ndarray,
    position: np.ndarray,
    always_reference_pnl: np.ndarray,
    k2_pnl: np.ndarray,
    c_pnl: np.ndarray,
    strong: np.ndarray,
    weak: np.ndarray,
    dip_entry: np.ndarray,
) -> dict[str, object]:
    pnl = realized_pnl(price, position)
    daily = pnl[1:]
    sd = float(np.std(daily, ddof=1)) if len(daily) > 1 else 0.0
    total = float(np.sum(pnl))
    long_mask = np.r_[False, position[:-1] > 0]
    short_mask = np.r_[False, position[:-1] < 0]
    flat_mask = np.r_[False, position[:-1] == 0]
    strong_mask = _realized_state_mask(strong)
    weak_mask = _realized_state_mask(weak)
    dip_mask = _realized_state_mask(dip_entry) & weak_mask
    other_weak_mask = weak_mask & ~dip_mask
    return {
        "P&L": total,
        "Incremental vs full +800": total - float(np.sum(always_reference_pnl)),
        "Incremental vs K2": total - float(np.sum(k2_pnl)),
        "Incremental vs Candidate C": total - float(np.sum(c_pnl)),
        "Active days": int(np.sum(position != 0)),
        "Long days": int(np.sum(position > 0)),
        "Short days": int(np.sum(position < 0)),
        "Flat days": int(np.sum(position == 0)),
        "Turnover": int(np.sum(np.abs(np.diff(np.r_[0, position])))),
        "Hit rate active": float(np.mean(pnl[1:][position[:-1] != 0] > 0))
        if np.any(position[:-1] != 0)
        else float("nan"),
        "Sharpe annualized": float(np.mean(daily) / sd * math.sqrt(365.0))
        if sd > 0
        else 0.0,
        "Max drawdown": _max_drawdown(pnl),
        "Maximum notional AUD": float(np.max(np.abs(position) * price))
        if len(price)
        else 0.0,
        "Longest loss streak": _loss_streak(pnl),
        "Long P&L": float(np.sum(pnl[long_mask])),
        "Short P&L": float(np.sum(pnl[short_mask])),
        "Flat P&L": float(np.sum(pnl[flat_mask])),
        "Strong P&L": float(np.sum(pnl[strong_mask])),
        "Weak P&L": float(np.sum(pnl[weak_mask])),
        "Weak dip-entry P&L": float(np.sum(pnl[dip_mask])),
        "Weak non-entry/flat P&L": float(np.sum(pnl[other_weak_mask])),
        "Weak overlay minus C weak P&L": float(
            np.sum(pnl[weak_mask]) - np.sum(c_pnl[weak_mask])
        ),
        "Best day": float(np.max(pnl)) if len(pnl) else 0.0,
        "Worst day": float(np.min(pnl)) if len(pnl) else 0.0,
        "Best 1 day share": _top_share(pnl, total, 1),
        "Best 3 day share": _top_share(pnl, total, 3),
        "Best 5 day share": _top_share(pnl, total, 5),
        "Best 10 day share": _top_share(pnl, total, 10),
        "Best 20 day share": _top_share(pnl, total, 20),
    }


def legacy_asymmetry_reproduction(price: np.ndarray) -> pd.DataFrame:
    """Reproduce the old 38/48 observation asymmetry exactly."""

    old_positions = build_followup_candidates(price)
    old_traces = build_followup_traces(price)
    z = old_traces["hybrid"]["ema_z"][:-1]
    weak = (~old_traces["hybrid"]["strong"][:-1]) & np.isfinite(z)
    change = np.diff(price)
    rows: list[dict[str, object]] = []
    for label, mask, signal in (
        (
            "negative_deviation_long_reversal",
            weak & (z < 0),
            np.ones(len(z), dtype=float),
        ),
        (
            "positive_deviation_short_reversal",
            weak & (z > 0),
            -np.ones(len(z), dtype=float),
        ),
    ):
        y = change[mask]
        contribution = signal[mask] * y
        rows.append(
            {
                "Method": "legacy_corrected_prior_volatility_trace",
                "Sample definition": "t=1..n-2; abs(C-Kalman slope z)<0.6; finite old causal EMA z; old min-history=5",
                "Bucket": label,
                "Observations": int(np.sum(mask)),
                "Mean next-day change": float(np.mean(y)),
                "Median next-day change": float(np.median(y)),
                "Positive-return fraction": float(np.mean(y > 0)),
                "Long reversal hit rate": float(np.mean(y > 0))
                if label.startswith("negative")
                else float("nan"),
                "Short reversal hit rate": float(np.mean(y < 0))
                if label.startswith("positive")
                else float("nan"),
                "Mean reversal contribution": float(np.mean(contribution)),
                "Total reversal contribution": float(np.sum(contribution)),
                "Old corrected hybrid P&L sample": float(
                    np.sum(realized_pnl(price, old_positions["corrected_hybrid"])[1:][mask])
                ),
            }
        )
    return pd.DataFrame(rows)


def _bucket_masks(z: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("negative_centred", z < 0),
        ("positive_centred", z > 0),
        ("residual_z_le_minus_1", z <= -1.0),
        ("minus_1_to_minus_0.5", (z > -1.0) & (z <= -0.5)),
        ("minus_0.5_to_0", (z > -0.5) & (z < 0)),
        ("0_to_0.5", (z >= 0) & (z < 0.5)),
        ("0.5_to_1", (z >= 0.5) & (z < 1.0)),
        ("residual_z_ge_1", z >= 1.0),
    ]


def residual_bucket_table(
    price: np.ndarray,
    kalman: dict[str, np.ndarray],
    residual: dict[str, np.ndarray],
    primary_position: np.ndarray,
    method: str,
) -> pd.DataFrame:
    z_key = "z_uncentred" if "uncentred" in method else "z_centered"
    z = residual[z_key][:-1]
    eligible = residual["eligible"][:-1]
    weak = kalman["weak"][:-1]
    change = np.diff(price)
    position = primary_position[:-1]
    rows: list[dict[str, object]] = []
    for bucket, bucket_mask in _bucket_masks(z):
        mask = weak & eligible & bucket_mask & np.isfinite(change)
        y = change[mask]
        z_use = z[mask]
        reversal = -np.sign(z_use) * y
        long_contribution = LIMIT * y[z_use < 0]
        short_contribution = -LIMIT * y[z_use > 0]
        actual = position[mask] * y
        regression = hac_regression(z_use, y, lags=5)
        rows.append(
            {
                "Method": method,
                "Bucket": bucket,
                "Observations": int(len(y)),
                "Mean next-day price change": float(np.mean(y))
                if len(y)
                else float("nan"),
                "Median next-day price change": float(np.median(y))
                if len(y)
                else float("nan"),
                "Positive-return fraction": float(np.mean(y > 0))
                if len(y)
                else float("nan"),
                "Long-signal hit rate": float(np.mean(y[z_use < 0] > 0))
                if np.any(z_use < 0)
                else float("nan"),
                "Short-signal hit rate": float(np.mean(y[z_use > 0] < 0))
                if np.any(z_use > 0)
                else float("nan"),
                "Mean long contribution AUD": float(np.mean(long_contribution))
                if len(long_contribution)
                else float("nan"),
                "Mean short contribution AUD": float(np.mean(short_contribution))
                if len(short_contribution)
                else float("nan"),
                "Mean reversal contribution price units": float(np.mean(reversal))
                if len(reversal)
                else float("nan"),
                "Mean actual one-sided contribution AUD": float(np.mean(actual))
                if len(actual)
                else float("nan"),
                "Beta": regression.get("Beta"),
                "Beta SE HAC": regression.get("Beta SE HAC"),
                "Beta CI low": regression.get("Beta CI low"),
                "Beta CI high": regression.get("Beta CI high"),
                "Beta p normal": regression.get("Beta p normal"),
            }
        )

    entry = np.zeros(len(z), dtype=bool)
    entry[1:] = weak[1:] & ~weak[:-1]
    later = weak & ~entry
    for label, mask in (("weak_entry", entry), ("weak_later", later)):
        mask = mask & eligible & np.isfinite(z) & np.isfinite(change)
        y = change[mask]
        z_use = z[mask]
        contribution = -np.sign(z_use) * y
        regression = hac_regression(z_use, y, lags=5)
        rows.append(
            {
                "Method": method,
                "Bucket": label,
                "Observations": int(len(y)),
                "Mean next-day price change": float(np.mean(y)) if len(y) else float("nan"),
                "Median next-day price change": float(np.median(y)) if len(y) else float("nan"),
                "Positive-return fraction": float(np.mean(y > 0)) if len(y) else float("nan"),
                "Long-signal hit rate": float(np.mean(y[z_use < 0] > 0)) if np.any(z_use < 0) else float("nan"),
                "Short-signal hit rate": float(np.mean(y[z_use > 0] < 0)) if np.any(z_use > 0) else float("nan"),
                "Mean long contribution AUD": float(np.mean(LIMIT * y[z_use < 0])) if np.any(z_use < 0) else float("nan"),
                "Mean short contribution AUD": float(np.mean(-LIMIT * y[z_use > 0])) if np.any(z_use > 0) else float("nan"),
                "Mean reversal contribution price units": float(np.mean(contribution)) if len(contribution) else float("nan"),
                "Mean actual one-sided contribution AUD": float(np.mean(primary_position[:-1][mask] * y)) if len(y) else float("nan"),
                "Beta": regression.get("Beta"),
                "Beta SE HAC": regression.get("Beta SE HAC"),
                "Beta CI low": regression.get("Beta CI low"),
                "Beta CI high": regression.get("Beta CI high"),
                "Beta p normal": regression.get("Beta p normal"),
            }
        )
    return pd.DataFrame(rows)


def long_short_asymmetry_table(
    price: np.ndarray,
    kalman: dict[str, np.ndarray],
    residual: dict[str, np.ndarray],
) -> pd.DataFrame:
    z = residual["z_centered"][:-1]
    weak = kalman["weak"][:-1]
    eligible = residual["eligible"][:-1]
    change = np.diff(price)
    rows: list[dict[str, object]] = []
    for method, use_z, use_eligible in (
        ("rolling_primary_centered", z, eligible),
        ("rolling_primary_uncentred", residual["z_uncentred"][:-1], eligible),
    ):
        for side, mask, direction in (
            ("long_on_negative", weak & use_eligible & (use_z < 0), 1),
            ("short_on_positive", weak & use_eligible & (use_z > 0), -1),
        ):
            y = change[mask]
            contribution = direction * y
            regression = hac_regression(use_z[mask], y, lags=5)
            rows.append(
                {
                    "Method": method,
                    "Side": side,
                    "Observations": int(len(y)),
                    "Mean next-day change": float(np.mean(y)) if len(y) else float("nan"),
                    "Median next-day change": float(np.median(y)) if len(y) else float("nan"),
                    "Positive-return fraction": float(np.mean(y > 0)) if len(y) else float("nan"),
                    "Directional hit rate": float(np.mean(contribution > 0)) if len(y) else float("nan"),
                    "Mean reversal contribution": float(np.mean(contribution)) if len(y) else float("nan"),
                    "Median reversal contribution": float(np.median(contribution)) if len(y) else float("nan"),
                    "Beta": regression.get("Beta"),
                    "Beta SE HAC": regression.get("Beta SE HAC"),
                    "Beta CI low": regression.get("Beta CI low"),
                    "Beta CI high": regression.get("Beta CI high"),
                    "Beta p normal": regression.get("Beta p normal"),
                }
            )
    return pd.DataFrame(rows)


def _candidate_metrics_table(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
    dip_masks: dict[str, np.ndarray],
    kalman: dict[str, np.ndarray],
) -> pd.DataFrame:
    strict_always = realized_pnl(price, positions["A_always_long"])
    full_always = realized_pnl(price, np.full(len(price), LIMIT, dtype=int))
    k2_pnl = realized_pnl(price, positions["B_K2"])
    c_pnl = realized_pnl(price, positions["C_candidate_C"])
    rows: list[dict[str, object]] = []
    for name in positions:
        row = {"Candidate": name}
        row.update(
            _metrics(
                price,
                positions[name],
                full_always,
                k2_pnl,
                c_pnl,
                kalman["strong_positive"] | kalman["strong_negative"],
                kalman["weak"],
                dip_masks.get(name, np.zeros(len(price), dtype=bool)),
            )
        )
        row["Strict always-long P&L"] = float(np.sum(strict_always))
        rows.append(row)
    return pd.DataFrame(rows)


def _chronological_table(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
    kalman: dict[str, np.ndarray],
    dip_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    k2_pnl = realized_pnl(price, positions["B_K2"])
    c_pnl = realized_pnl(price, positions["C_candidate_C"])
    rows: list[dict[str, object]] = []
    segments = list(_segment_definitions(len(price)))
    segments.extend(
        (
            "exclude_final_days",
            f"exclude_last_{days}",
            0,
            len(price) - days,
        )
        for days in (30, 60, 91)
    )
    for name, position in positions.items():
        pnl = realized_pnl(price, position)
        long_mask = np.r_[False, position[:-1] > 0]
        short_mask = np.r_[False, position[:-1] < 0]
        strong_mask = _realized_state_mask(kalman["strong_positive"] | kalman["strong_negative"])
        weak_mask = _realized_state_mask(kalman["weak"])
        dip_mask = _realized_state_mask(dip_masks.get(name, np.zeros(len(price), dtype=bool))) & weak_mask
        for split, label, start, end in segments:
            end = min(end, len(price))
            rows.append(
                {
                    "Candidate": name,
                    "Split": split,
                    "Segment": label,
                    "Start day": start,
                    "End day exclusive": end,
                    "P&L": float(np.sum(pnl[start:end])),
                    "Incremental vs K2": float(np.sum(pnl[start:end]) - np.sum(k2_pnl[start:end])),
                    "Incremental vs Candidate C": float(np.sum(pnl[start:end]) - np.sum(c_pnl[start:end])),
                    "Long P&L": float(np.sum(pnl[start:end] * long_mask[start:end])),
                    "Short P&L": float(np.sum(pnl[start:end] * short_mask[start:end])),
                    "Strong P&L": float(np.sum(pnl[start:end] * strong_mask[start:end])),
                    "Weak P&L": float(np.sum(pnl[start:end] * weak_mask[start:end])),
                    "Weak dip-entry P&L": float(np.sum(pnl[start:end] * dip_mask[start:end])),
                    "Active realised days": int(np.sum(position[:-1][start : max(start, end - 1)] != 0))
                    if end > start
                    else 0,
                }
            )
    return pd.DataFrame(rows)


def paired_segment_comparisons(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
) -> pd.DataFrame:
    primary_pnl = realized_pnl(price, positions["one_sided_primary"])
    c_pnl = realized_pnl(price, positions["C_candidate_C"])
    rows: list[dict[str, object]] = []
    for candidate in (
        "one_sided_primary",
        "one_sided_hysteresis_exit0",
        "one_sided_hysteresis_exit025",
        "no_short_long_flat",
        "reduced_weak_long_200",
        "two_sided_stateless",
    ):
        delta = realized_pnl(price, positions[candidate]) - c_pnl
        segments = list(_segment_definitions(len(price)))
        for split, label, start, end in segments:
            rows.append(
                {
                    "Candidate": candidate,
                    "Reference": "C_candidate_C",
                    "Split": split,
                    "Segment": label,
                    "Paired advantage": float(np.sum(delta[start:end])),
                    "Segment won": bool(np.sum(delta[start:end]) > 0),
                }
            )
        segment_advantages = [float(np.sum(delta[s:e])) for _, _, s, e in segments]
        rows.append(
            {
                "Candidate": candidate,
                "Reference": "C_candidate_C",
                "Split": "full_summary",
                "Segment": "all",
                "Total paired advantage": float(np.sum(delta)),
                "Segments won": int(sum(x > 0 for x in segment_advantages)),
                "Segments tested": int(len(segment_advantages)),
                "Percent segments won": float(np.mean(np.asarray(segment_advantages) > 0)),
                "Max drawdown paired curve": _max_drawdown(delta),
                "Best 1 paired-day share": _top_share(delta, float(np.sum(delta)), 1),
                "Best 3 paired-day share": _top_share(delta, float(np.sum(delta)), 3),
                "Best 5 paired-day share": _top_share(delta, float(np.sum(delta)), 5),
                "Best 10 paired-day share": _top_share(delta, float(np.sum(delta)), 10),
                "Best 20 paired-day share": _top_share(delta, float(np.sum(delta)), 20),
            }
        )
    return pd.DataFrame(rows)


def corrected_best_trade_exclusions(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
    kalman: dict[str, np.ndarray],
    residual: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Exclude only actual one-sided weak dip contributions."""

    primary = positions["one_sided_primary"]
    c_pnl = realized_pnl(price, positions["C_candidate_C"])
    primary_pnl = realized_pnl(price, primary)
    change = np.diff(price)
    z = residual["z_centered"][:-1]
    weak = kalman["weak"][:-1]
    eligible = residual["eligible"][:-1]
    dip = weak & eligible & (z <= -PRIMARY_ENTRY_THRESHOLD)
    contribution_aud = primary[:-1] * change
    eligible_indices = np.flatnonzero(dip)
    order = eligible_indices[np.argsort(contribution_aud[eligible_indices])]
    rows: list[dict[str, object]] = []

    def append_row(test: str, excluded_indices: np.ndarray, contribution_values: np.ndarray | None = None) -> None:
        realized_indices = excluded_indices + 1
        keep = np.ones(len(primary_pnl), dtype=bool)
        keep[realized_indices] = False
        remaining = contribution_aud[dip].copy()
        if contribution_values is not None:
            remaining = contribution_values[dip]
        rows.append(
            {
                "Signal": "one_sided_primary",
                "Test": test,
                "Eligible dip observations": int(len(eligible_indices)),
                "Excluded observations": int(len(excluded_indices)),
                "Excluded contribution AUD": float(np.sum(contribution_aud[excluded_indices]))
                if len(excluded_indices)
                else 0.0,
                "Remaining dip contribution AUD": float(np.sum(remaining[~np.isin(eligible_indices, excluded_indices)]))
                if len(remaining)
                else 0.0,
                "Remaining mean dip contribution AUD": float(np.mean(remaining[~np.isin(eligible_indices, excluded_indices)]))
                if np.any(~np.isin(eligible_indices, excluded_indices))
                else float("nan"),
                "Remaining primary P&L": float(np.sum(primary_pnl[keep])),
                "Remaining Candidate C P&L": float(np.sum(c_pnl[keep])),
                "Remaining primary minus C": float(np.sum(primary_pnl[keep] - c_pnl[keep])),
            }
        )

    append_row("all_eligible_weak_dips", np.asarray([], dtype=int))
    for count in (1, 3, 5, 10):
        append_row(f"exclude_best_{count}_one_sided_trades", order[-count:])
    for cap in (2000.0, 4000.0, 6000.0):
        clipped = np.clip(contribution_aud, -cap, cap)
        append_row(f"winsorise_contribution_abs_{cap:g}", np.asarray([], dtype=int), clipped)

    for split, label, start, end in _segment_definitions(len(price)):
        excluded = eligible_indices[(eligible_indices >= start) & (eligible_indices < end)]
        append_row(f"leave_out_{split}_{label}", excluded)
    return pd.DataFrame(rows)


def _parameter_configs() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for alpha in EMA_ALPHAS:
        for window in RESIDUAL_WINDOWS:
            for threshold in ENTRY_THRESHOLDS:
                configs.append(
                    {
                        "Config": f"centered_a{alpha:g}_w{window}_t{threshold:g}",
                        "alpha": alpha,
                        "window": window,
                        "entry_threshold": threshold,
                        "centered": True,
                    }
                )
    return configs


def parameter_sensitivity(
    price: np.ndarray,
    kalman: dict[str, np.ndarray],
    c_pnl: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    positions: dict[str, np.ndarray] = {}
    full_always = realized_pnl(price, np.full(len(price), LIMIT, dtype=int))
    for config in _parameter_configs():
        residual = rolling_prediction_residual_trace(
            price, float(config["alpha"]), int(config["window"])
        )
        position, _ = _stateless_one_sided_position(
            kalman,
            residual,
            float(config["entry_threshold"]),
            centered=bool(config["centered"]),
        )
        positions[str(config["Config"])] = position
        pnl = realized_pnl(price, position)
        quarters = [
            float(np.sum(pnl[start:end]))
            for split, label, start, end in _segment_definitions(len(price))
            if split == "quarter"
        ]
        rows.append(
            {
                **config,
                "P&L": float(np.sum(pnl)),
                "Incremental vs always long": float(np.sum(pnl) - np.sum(full_always)),
                "Incremental vs Candidate C": float(np.sum(pnl) - np.sum(c_pnl)),
                "Positive quarters": int(sum(x > 0 for x in quarters)),
                "Q1": quarters[0],
                "Q2": quarters[1],
                "Q3": quarters[2],
                "Q4": quarters[3],
                "H1": float(np.sum(pnl[:182])),
                "H2": float(np.sum(pnl[182:])),
                "Long days": int(np.sum(position > 0)),
                "Short days": int(np.sum(position < 0)),
                "Flat days": int(np.sum(position == 0)),
                "Turnover": int(np.sum(np.abs(np.diff(np.r_[0, position])))),
                "Effective path ID": _path_id(position),
            }
        )
    table = pd.DataFrame(rows)
    return table, positions


def parameter_sensitivity_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for alpha in EMA_ALPHAS:
        frame = table[table["alpha"] == alpha]
        rows.append(
            {
                "Grouping": "alpha",
                "Value": alpha,
                "Configurations": int(len(frame)),
                "Unique position paths": int(frame["Effective path ID"].nunique()),
                "Median P&L": float(frame["P&L"].median()),
                "Lower decile P&L": float(frame["P&L"].quantile(0.10)),
                "Worst P&L": float(frame["P&L"].min()),
                "Median incremental vs C": float(frame["Incremental vs Candidate C"].median()),
                "Positive-quarter median": float(frame["Positive quarters"].median()),
            }
        )
    for window in RESIDUAL_WINDOWS:
        frame = table[table["window"] == window]
        rows.append(
            {
                "Grouping": "window",
                "Value": window,
                "Configurations": int(len(frame)),
                "Unique position paths": int(frame["Effective path ID"].nunique()),
                "Median P&L": float(frame["P&L"].median()),
                "Lower decile P&L": float(frame["P&L"].quantile(0.10)),
                "Worst P&L": float(frame["P&L"].min()),
                "Median incremental vs C": float(frame["Incremental vs Candidate C"].median()),
                "Positive-quarter median": float(frame["Positive quarters"].median()),
            }
        )
    for threshold in ENTRY_THRESHOLDS:
        frame = table[table["entry_threshold"] == threshold]
        rows.append(
            {
                "Grouping": "entry_threshold",
                "Value": threshold,
                "Configurations": int(len(frame)),
                "Unique position paths": int(frame["Effective path ID"].nunique()),
                "Median P&L": float(frame["P&L"].median()),
                "Lower decile P&L": float(frame["P&L"].quantile(0.10)),
                "Worst P&L": float(frame["P&L"].min()),
                "Median incremental vs C": float(frame["Incremental vs Candidate C"].median()),
                "Positive-quarter median": float(frame["Positive quarters"].median()),
            }
        )
    return pd.DataFrame(rows)


def chronological_parameter_selection(
    price: np.ndarray,
    table: pd.DataFrame,
    positions: dict[str, np.ndarray],
) -> pd.DataFrame:
    c_position = build_one_sided_candidates(price)["C_candidate_C"]
    c_pnl = realized_pnl(price, c_position)
    rows: list[dict[str, object]] = []
    first_scores: list[tuple[str, float, float]] = []
    for config, position in positions.items():
        pnl = realized_pnl(price, position)
        first_scores.append((config, float(np.sum(pnl[:182])), float(np.sum(pnl[182:]))))
    first_scores.sort(key=lambda x: x[1], reverse=True)
    for rank, (config, first_pnl, second_pnl) in enumerate(first_scores, 1):
        rows.append(
            {
                "Selection type": "first_half_rank_then_second_half",
                "Selection day": 182,
                "Block": "H1_to_H2",
                "Config": config,
                "First-half score P&L": first_pnl,
                "First-half rank": rank,
                "Second-half test P&L": second_pnl,
                "Second-half Candidate C P&L": float(np.sum(c_pnl[182:])),
                "Second-half paired advantage": second_pnl - float(np.sum(c_pnl[182:])),
                "Chosen": rank == 1,
            }
        )

    primary_name = "centered_a0.06_w20_t0.5"
    primary_position = positions[primary_name]
    primary_pnl = realized_pnl(price, primary_position)
    rows.extend(
        {
            "Selection type": "frozen_primary_diagnostic",
            "Selection day": 0,
            "Block": block,
            "Config": primary_name,
            "First-half score P&L": float("nan"),
            "First-half rank": float("nan"),
            "Second-half test P&L": float(np.sum(primary_pnl[start:end])),
            "Second-half Candidate C P&L": float(np.sum(c_pnl[start:end])),
            "Second-half paired advantage": float(np.sum(primary_pnl[start:end] - c_pnl[start:end])),
            "Chosen": True,
        }
        for block, start, end in (("H1", 0, 182), ("H2", 182, len(price)))
    )

    # A deliberately small expanding walk-forward diagnostic.  The selected
    # stateless path is causal; every future block is scored after selection.
    for boundary, end in ((120, 240), (240, len(price))):
        scored = []
        for config, position in positions.items():
            pnl = realized_pnl(price, position)
            scored.append((config, float(np.sum(pnl[:boundary])), pnl))
        scored.sort(key=lambda x: x[1], reverse=True)
        selected, score, selected_pnl = scored[0]
        rows.append(
            {
                "Selection type": "expanding_walk_forward",
                "Selection day": boundary,
                "Block": f"{boundary}_to_{end}",
                "Config": selected,
                "First-half score P&L": score,
                "First-half rank": 1,
                "Second-half test P&L": float(np.sum(selected_pnl[boundary:end])),
                "Second-half Candidate C P&L": float(np.sum(c_pnl[boundary:end])),
                "Second-half paired advantage": float(np.sum(selected_pnl[boundary:end] - c_pnl[boundary:end])),
                "Chosen": True,
            }
        )
    return pd.DataFrame(rows)


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
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    deltas = np.asarray(deltas, dtype=float)
    mdd_deltas = np.asarray(mdd_deltas, dtype=float)
    win = float(np.mean(deltas > 0)) if len(deltas) else float("nan")
    p = win
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
        "Win fraction": win,
        "Win MC SE": math.sqrt(p * (1.0 - p) / len(deltas)) if len(deltas) else float("nan"),
        "Median max-drawdown difference": float(np.median(mdd_deltas)) if len(mdd_deltas) else float("nan"),
        "P5 max-drawdown difference": float(np.quantile(mdd_deltas, 0.05)) if len(mdd_deltas) else float("nan"),
        "MDD improvement fraction": float(np.mean(mdd_deltas > 0)) if len(mdd_deltas) else float("nan"),
        **(extra or {}),
    }


def _evaluate_path(path: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    positions = build_one_sided_candidates(path)
    totals: dict[str, float] = {}
    mdds: dict[str, float] = {}
    for name, position in positions.items():
        pnl = realized_pnl(path, position)
        totals[name] = float(np.sum(pnl))
        mdds[name] = _max_drawdown(pnl)
    return totals, mdds


def _bootstrap_from_change_scenarios(
    price: np.ndarray,
    scenarios: Iterable[tuple[str, np.ndarray]],
    repetitions: int,
    design: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    scenario_list = list(scenarios)
    for scenario, changes_matrix in scenario_list:
        changes_matrix = np.asarray(changes_matrix, dtype=float)
        totals: dict[str, np.ndarray] = {name: np.empty(repetitions) for name in CORE_NAMES if name in {"A_always_long", "B_K2", "C_candidate_C", "D_corrected_hybrid_frozen", "one_sided_primary", "no_short_long_flat"}}
        mdds: dict[str, np.ndarray] = {name: np.empty(repetitions) for name in totals}
        plausible = np.zeros(repetitions, dtype=bool)
        for rep in range(repetitions):
            changes = changes_matrix[rep]
            path = np.r_[price[0], price[0] + np.cumsum(changes)]
            plausible[rep] = _path_plausible(path)
            path_totals, path_mdds = _evaluate_path(path)
            for name in totals:
                totals[name][rep] = path_totals[name]
                mdds[name][rep] = path_mdds[name]

        for name in totals:
            for subset, mask in (
                ("all_attempts", np.ones(repetitions, dtype=bool)),
                ("plausible_positive_range", plausible),
            ):
                values = totals[name][mask]
                if not len(values):
                    continue
                always = totals["A_always_long"][mask]
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
                        "Median incremental vs always-long": float(np.median(values - always)),
                        "P5 incremental vs always-long": float(np.quantile(values - always, 0.05)),
                        "Median max drawdown": float(np.median(mdds[name][mask])),
                    }
                )
        for candidate, reference in PAIR_SPECS:
            for subset, mask in (
                ("all_attempts", np.ones(repetitions, dtype=bool)),
                ("plausible_positive_range", plausible),
            ):
                if not np.any(mask):
                    continue
                pair_rows.append(
                    _paired_summary(
                        design,
                        scenario,
                        subset,
                        candidate,
                        reference,
                        totals[candidate][mask] - totals[reference][mask],
                        mdds[candidate][mask] - mdds[reference][mask],
                        repetitions,
                        int(np.sum(mask)),
                        float(np.mean(plausible)),
                    )
                )
    return pd.DataFrame(total_rows), pd.DataFrame(pair_rows)


def additive_change_bootstrap(
    price: np.ndarray,
    repetitions: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    changes = np.diff(price)
    rng = np.random.default_rng(SEED + 701)
    scenarios: list[tuple[str, np.ndarray]] = []
    for block in (5, 10, 20, 40, 60):
        matrix = np.vstack(
            [_circular_block_changes(changes, block, rng) for _ in range(repetitions)]
        )
        scenarios.append((f"block_{block}", matrix))
    return _bootstrap_from_change_scenarios(
        price, scenarios, repetitions, "additive_absolute_change_circular_blocks"
    )


def drift_residual_bootstrap(
    price: np.ndarray,
    repetitions: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    changes = np.diff(price)
    drift = float(np.mean(changes))
    residual = changes - drift
    rng = np.random.default_rng(SEED + 702)
    scenarios: list[tuple[str, np.ndarray]] = []
    for multiplier in (1.0, 0.75, 0.5, 0.0, 1.25):
        matrix = np.vstack(
            [
                multiplier * drift
                + _circular_block_changes(residual, 20, rng)
                for _ in range(repetitions)
            ]
        )
        scenarios.append((f"drift_{multiplier:g}x", matrix))
    return _bootstrap_from_change_scenarios(
        price, scenarios, repetitions, "drift_plus_residual_block_20"
    )


def _regime_changes(
    n_changes: int,
    positive_duration: int,
    negative_duration: int,
    drift: float,
    noise: float,
    gradual: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    base_noise = float(np.std(np.diff(load_price_series()), ddof=1))
    changes: list[float] = []
    state_index = 0
    previous_state = 1
    while len(changes) < n_changes:
        state = 1 if state_index % 2 == 0 else -1
        duration = positive_duration if state > 0 else negative_duration
        duration = min(duration, n_changes - len(changes))
        if gradual:
            blend = np.linspace(0.0, 1.0, duration, endpoint=False)
            prior_target = previous_state * drift
            target = (1.0 - blend) * prior_target + blend * state * drift
        else:
            target = np.full(duration, state * drift)
        changes.extend((target + rng.normal(0.0, base_noise * noise, duration)).tolist())
        previous_state = state
        state_index += 1
    return np.asarray(changes[:n_changes], dtype=float)


def regime_preserving_multi_path(
    price: np.ndarray,
    repetitions: int = 200,
) -> pd.DataFrame:
    scenarios = {
        "persistent_positive": dict(pos=100, neg=30, drift=0.18, noise=0.75, gradual=False),
        "persistent_negative": dict(pos=30, neg=100, drift=0.18, noise=0.75, gradual=False),
        "short_negative_reversals": dict(pos=60, neg=10, drift=0.18, noise=0.80, gradual=False),
        "long_negative_regimes": dict(pos=50, neg=90, drift=0.18, noise=0.75, gradual=False),
        "balanced_40_day": dict(pos=40, neg=40, drift=0.18, noise=0.75, gradual=False),
        "gradual_transitions": dict(pos=60, neg=60, drift=0.18, noise=0.70, gradual=True),
        "higher_noise": dict(pos=60, neg=60, drift=0.18, noise=1.50, gradual=False),
        "lower_noise": dict(pos=60, neg=60, drift=0.18, noise=0.50, gradual=False),
    }
    rng = np.random.default_rng(SEED + 703)
    rows: list[dict[str, object]] = []
    for scenario, spec in scenarios.items():
        totals: dict[tuple[str, str], list[float]] = {pair: [] for pair in PAIR_SPECS}
        mdd_deltas: dict[tuple[str, str], list[float]] = {pair: [] for pair in PAIR_SPECS}
        plausible_flags: list[bool] = []
        for _ in range(repetitions):
            changes = _regime_changes(
                len(price) - 1,
                spec["pos"],
                spec["neg"],
                spec["drift"],
                spec["noise"],
                spec["gradual"],
                rng,
            )
            path = np.r_[price[0], price[0] + np.cumsum(changes)]
            plausible_flags.append(_path_plausible(path))
            path_totals, path_mdds = _evaluate_path(path)
            for pair in PAIR_SPECS:
                candidate, reference = pair
                totals[pair].append(path_totals[candidate] - path_totals[reference])
                mdd_deltas[pair].append(path_mdds[candidate] - path_mdds[reference])
        plausible = np.asarray(plausible_flags, dtype=bool)
        for candidate, reference in PAIR_SPECS:
            pair = (candidate, reference)
            delta = np.asarray(totals[pair], dtype=float)
            mdd = np.asarray(mdd_deltas[pair], dtype=float)
            for subset, mask in (
                ("all_attempts", np.ones(repetitions, dtype=bool)),
                ("plausible_positive_range", plausible),
            ):
                if not np.any(mask):
                    continue
                rows.append(
                    _paired_summary(
                        "regime_preserving_generator_conditioned",
                        scenario,
                        subset,
                        candidate,
                        reference,
                        delta[mask],
                        mdd[mask],
                        repetitions,
                        int(np.sum(mask)),
                        float(np.mean(plausible)),
                        {
                            "Positive duration": spec["pos"],
                            "Negative duration": spec["neg"],
                            "Drift per day": spec["drift"],
                            "Noise scale": spec["noise"],
                            "Gradual transition": spec["gradual"],
                        },
                    )
                )
    return pd.DataFrame(rows)


def _effective_family(price: np.ndarray) -> tuple[dict[str, np.ndarray], str]:
    kalman = _kalman_state(price)
    family: dict[str, np.ndarray] = {}
    for config in _parameter_configs():
        residual = rolling_prediction_residual_trace(
            price, float(config["alpha"]), int(config["window"])
        )
        family[str(config["Config"])] = _stateless_one_sided_position(
            kalman,
            residual,
            float(config["entry_threshold"]),
            centered=True,
        )[0]
    core, _, _ = _build_one_sided_candidates(price)
    for name in (
        "B_K2",
        "C_candidate_C",
        "D_corrected_hybrid_frozen",
        "one_sided_primary",
        "one_sided_hysteresis_exit0",
        "one_sided_hysteresis_exit025",
        "no_short_long_flat",
        "reduced_weak_long_200",
        "two_sided_stateless",
    ):
        family[name] = core[name]
    unique: dict[tuple[int, ...], tuple[str, np.ndarray]] = {}
    for name, position in family.items():
        unique.setdefault(tuple(position.tolist()), (name, position))
    effective = {name: position for name, position in unique.values()}
    observed = max(
        effective,
        key=lambda name: float(np.sum(realized_pnl(price, effective[name]))),
    )
    return effective, observed


def familywise_diagnostics(price: np.ndarray, repetitions: int = 300) -> pd.DataFrame:
    family, observed_name = _effective_family(price)
    observed_best = float(np.sum(realized_pnl(price, family[observed_name])))
    rng = np.random.default_rng(SEED + 704)
    changes = np.diff(price)
    maxima = np.empty(repetitions, dtype=float)
    plausible = np.zeros(repetitions, dtype=bool)
    for rep in range(repetitions):
        sampled = _circular_block_changes(changes, 10, rng)
        path = np.r_[price[0], price[0] + np.cumsum(sampled)]
        plausible[rep] = _path_plausible(path)
        null_family, _ = _effective_family(path)
        maxima[rep] = max(
            float(np.sum(realized_pnl(path, position)))
            for position in null_family.values()
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
                "Family": "centered_one_sided_grid_plus_core",
                "Subset": subset,
                "Unique observed configurations": len(family),
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


def correctness_table(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
    kalman: dict[str, np.ndarray],
    residual: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    audit_days = (0, 30, 120, 240, 300)
    for name, position in positions.items():
        checks = {
            "integer_positions": bool(np.all(position == position.astype(int))),
            "within_Jeans_limit": bool(np.max(np.abs(position)) <= LIMIT),
            "day_zero_flat": bool(position[0] == 0),
            "prior_position_P&L": bool(
                np.allclose(
                    realized_pnl(price, position)[1:],
                    np.round(position[:-1] * np.diff(price), 2),
                )
            ),
            "final_day_has_no_future_return": bool(
                np.allclose(
                    realized_pnl(price, position),
                    realized_pnl(
                        price,
                        np.r_[
                            position[:-1],
                            0 if position[-1] != 0 else LIMIT,
                        ],
                    ),
                )
            ),
        }
        prefix_ok = True
        future_ok = True
        for day in audit_days:
            prefix_positions = build_one_sided_candidates(price[: day + 1])
            prefix_ok = prefix_ok and bool(prefix_positions[name][-1] == position[day])
            altered = price.copy()
            if day + 1 < len(price):
                altered[day + 1 :] += np.linspace(123.0, 456.0, len(price) - day - 1)
            altered_positions = build_one_sided_candidates(altered)
            future_ok = future_ok and bool(altered_positions[name][day] == position[day])
        checks["prefix_replay_matches"] = prefix_ok
        checks["future_price_perturbation_unchanged"] = future_ok
        for check, value in checks.items():
            rows.append(
                {
                    "Candidate": name,
                    "Check": check,
                    "Value": bool(value),
                    "Details": str(audit_days) if check in {"prefix_replay_matches", "future_price_perturbation_unchanged"} else "",
                }
            )

    for day in (21, 30, 50, 100, 200, 300):
        if day >= len(price):
            continue
        prefix = rolling_prediction_residual_trace(price[: day + 1], PRIMARY_ALPHA, PRIMARY_WINDOW)
        rows.append(
            {
                "Candidate": "rolling_primary_residual",
                "Check": "residual_scale_excludes_today_and_prefix_matches",
                "Value": bool(
                    np.allclose(
                        prefix["z_centered"][-1],
                        residual["z_centered"][day],
                        equal_nan=True,
                    )
                ),
                "Details": f"day={day}; prefix history ends at decision day",
            }
        )
    return pd.DataFrame(rows)


def capital_interaction_table(price: np.ndarray, positions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    combined_peak = 555931.0
    for name, position in positions.items():
        jeans_max = float(np.max(np.abs(position) * price))
        rows.append(
            {
                "Candidate": name,
                "Maximum standalone Jeans notional AUD": jeans_max,
                "Jeans share of AUD 600000 limit": jeans_max / JEANS_BUDGET,
                "Unchanged full-portfolio peak with full-limit Jeans AUD": combined_peak,
                "Remaining budget headroom AUD": JEANS_BUDGET - combined_peak,
                "Budget breaches in unchanged full-portfolio replay": 0,
                "Portfolio interaction note": "Full-portfolio peak is from the prior unchanged replay; Jeans-only edge and notional are reported separately.",
            }
        )
    return pd.DataFrame(rows)


def _write_figures(
    price: np.ndarray,
    positions: dict[str, np.ndarray],
    residual_buckets: pd.DataFrame,
    sensitivity: pd.DataFrame,
    chronological: pd.DataFrame,
    additive_pairs: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5))
    for name, label in (
        ("B_K2", "K2"),
        ("C_candidate_C", "Candidate C"),
        ("one_sided_primary", "One-sided primary"),
        ("D_corrected_hybrid_frozen", "Corrected hybrid"),
    ):
        plt.plot(np.cumsum(realized_pnl(price, positions[name])), label=label)
    plt.axhline(0, color="black", linewidth=0.7)
    plt.title("Thrifted Jeans cumulative P&L")
    plt.xlabel("Day")
    plt.ylabel("AUD")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "cumulative_one_sided_pnl.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 4))
    delta = realized_pnl(price, positions["one_sided_primary"]) - realized_pnl(price, positions["C_candidate_C"])
    plt.plot(np.cumsum(delta), color="tab:purple")
    plt.axhline(0, color="black", linewidth=0.7)
    plt.title("One-sided primary paired advantage over Candidate C")
    plt.xlabel("Day")
    plt.ylabel("AUD")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "one_sided_advantage_over_C.png", dpi=160)
    plt.close()

    bucket = residual_buckets[residual_buckets["Bucket"].isin(["negative_centred", "positive_centred", "residual_z_le_minus_1", "minus_1_to_minus_0.5", "minus_0.5_to_0", "0_to_0.5", "0.5_to_1", "residual_z_ge_1"])].copy()
    bucket = bucket[bucket["Method"] == "rolling_primary_centered"]
    if len(bucket):
        plt.figure(figsize=(12, 5))
        plt.bar(bucket["Bucket"], bucket["Mean next-day price change"])
        plt.axhline(0, color="black", linewidth=0.7)
        plt.xticks(rotation=35, ha="right")
        plt.title("Next-day change by weak-state rolling residual bucket")
        plt.ylabel("AUD price change")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / "weak_residual_bucket_returns.png", dpi=160)
        plt.close()

    pivot = sensitivity[
        (sensitivity["alpha"] == PRIMARY_ALPHA)
    ].pivot(index="window", columns="entry_threshold", values="P&L")
    if len(pivot):
        plt.figure(figsize=(8, 5))
        image = plt.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
        plt.colorbar(image, label="P&L AUD")
        plt.xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
        plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
        plt.xlabel("Entry threshold")
        plt.ylabel("Residual window")
        plt.title("One-sided P&L sensitivity at alpha=0.06")
        plt.tight_layout()
        plt.savefig(figure_dir / "one_sided_parameter_stability.png", dpi=160)
        plt.close()

    segment = chronological[
        (chronological["Split"] == "quarter")
        & chronological["Candidate"].isin(["C_candidate_C", "one_sided_primary", "D_corrected_hybrid_frozen"])
    ]
    if len(segment):
        pivot = segment.pivot(index="Segment", columns="Candidate", values="P&L")
        pivot.plot(kind="bar", figsize=(9, 5))
        plt.title("P&L by chronological quarter")
        plt.ylabel("AUD")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / "one_sided_pnl_by_quarter.png", dpi=160)
        plt.close()

    pair = additive_pairs[
        (additive_pairs["Subset"] == "all_attempts")
        & (additive_pairs["Candidate"] == "one_sided_primary")
        & (additive_pairs["Reference"] == "C_candidate_C")
    ]
    if len(pair):
        plt.figure(figsize=(9, 5))
        plt.bar(pair["Scenario"], pair["Median paired difference"])
        plt.axhline(0, color="black", linewidth=0.7)
        plt.xticks(rotation=30, ha="right")
        plt.title("Additive block bootstrap: one-sided minus Candidate C")
        plt.ylabel("Median paired AUD")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / "one_sided_paired_bootstrap.png", dpi=160)
        plt.close()


def run_audit(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    additive_repetitions: int = 300,
    drift_repetitions: int = 200,
    regime_repetitions: int = 300,
    familywise_repetitions: int = 500,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    price = load_price_series()
    positions, traces, dip_masks = _build_one_sided_candidates(price)
    kalman = traces["kalman_C"]
    primary_residual = traces["rolling_primary"]

    comparison = _candidate_metrics_table(price, positions, dip_masks, kalman)
    comparison.to_csv(output_dir / "exact_candidate_comparison.csv", index=False)
    comparison[comparison["Candidate"].isin(CORE_NAMES[2:])].to_csv(
        output_dir / "one_sided_mechanics_comparison.csv", index=False
    )

    legacy = legacy_asymmetry_reproduction(price)
    legacy.to_csv(output_dir / "legacy_asymmetry_reproduction.csv", index=False)
    buckets = pd.concat(
        [
            residual_bucket_table(
                price,
                kalman,
                primary_residual,
                positions["one_sided_primary"],
                "rolling_primary_centered",
            ),
            residual_bucket_table(
                price,
                kalman,
                primary_residual,
                positions["one_sided_primary"],
                "rolling_primary_uncentred_diagnostic",
            ),
        ],
        ignore_index=True,
    )
    buckets.to_csv(output_dir / "weak_state_residual_buckets.csv", index=False)
    asymmetry = pd.concat(
        [
            legacy.rename(columns={"Bucket": "Side"}),
            long_short_asymmetry_table(price, kalman, primary_residual),
        ],
        ignore_index=True,
        sort=False,
    )
    asymmetry.to_csv(output_dir / "long_short_asymmetry.csv", index=False)

    chronological = _chronological_table(price, positions, kalman, dip_masks)
    chronological.to_csv(output_dir / "chronological_segments.csv", index=False)
    paired_segments = paired_segment_comparisons(price, positions)
    paired_segments.to_csv(output_dir / "paired_segment_comparisons.csv", index=False)
    exclusions = corrected_best_trade_exclusions(
        price, positions, kalman, primary_residual
    )
    exclusions.to_csv(output_dir / "corrected_best_trade_exclusions.csv", index=False)

    c_pnl = realized_pnl(price, positions["C_candidate_C"])
    sensitivity, sensitivity_positions = parameter_sensitivity(price, kalman, c_pnl)
    sensitivity.to_csv(output_dir / "parameter_sensitivity.csv", index=False)
    sensitivity_summary = parameter_sensitivity_summary(sensitivity)
    sensitivity_summary.to_csv(output_dir / "parameter_sensitivity_summary.csv", index=False)
    selection = chronological_parameter_selection(price, sensitivity, sensitivity_positions)
    selection.to_csv(output_dir / "chronological_parameter_selection.csv", index=False)

    additive_total, additive_pairs = additive_change_bootstrap(
        price, additive_repetitions
    )
    additive_total.to_csv(output_dir / "additive_bootstrap_candidates.csv", index=False)
    additive_pairs.to_csv(output_dir / "additive_bootstrap_paired.csv", index=False)
    drift_total, drift_pairs = drift_residual_bootstrap(price, drift_repetitions)
    drift_total.to_csv(output_dir / "drift_residual_candidates.csv", index=False)
    drift_pairs.to_csv(output_dir / "drift_residual_paired.csv", index=False)
    regime = regime_preserving_multi_path(price, regime_repetitions)
    regime.to_csv(output_dir / "regime_preserving_multi_path.csv", index=False)
    familywise = familywise_diagnostics(price, familywise_repetitions)
    familywise.to_csv(output_dir / "familywise_diagnostics.csv", index=False)

    correctness = correctness_table(price, positions, kalman, primary_residual)
    correctness.to_csv(output_dir / "correctness_checks.csv", index=False)
    capital = capital_interaction_table(price, positions)
    capital.to_csv(output_dir / "full_portfolio_capital_interaction.csv", index=False)

    _write_figures(
        price,
        positions,
        buckets,
        sensitivity,
        chronological,
        additive_pairs,
        figure_dir,
    )

    manifest = {
        "data_path": str(DATA_PATH),
        "observations": int(len(price)),
        "start_price": float(price[0]),
        "end_price": float(price[-1]),
        "primary_alpha": PRIMARY_ALPHA,
        "primary_window": PRIMARY_WINDOW,
        "primary_entry_threshold": PRIMARY_ENTRY_THRESHOLD,
        "primary_residual_centered": True,
        "primary_residual_history_requirement": "all W prior residuals; current residual excluded",
        "core_candidate_count": int(len(positions)),
        "sensitivity_configuration_count": int(len(sensitivity)),
        "sensitivity_unique_position_paths": int(sensitivity["Effective path ID"].nunique()),
        "additive_repetitions_per_block": int(additive_repetitions),
        "drift_repetitions_per_assumption": int(drift_repetitions),
        "regime_repetitions_per_scenario": int(regime_repetitions),
        "familywise_repetitions": int(familywise_repetitions),
        "production_files_modified": False,
        "legacy_negative_observations": int(legacy.loc[legacy["Bucket"] == "negative_deviation_long_reversal", "Observations"].iloc[0]),
        "legacy_positive_observations": int(legacy.loc[legacy["Bucket"] == "positive_deviation_short_reversal", "Observations"].iloc[0]),
        "notes": [
            "Legacy 73.7%/52.1% asymmetry is reproduced separately using the prior corrected EMA trace and its exact weak-state sample.",
            "Primary rolling residual scale uses only the previous W residuals and is centered by their rolling mean.",
            "The primary one-sided EMA branch never shorts; strong negative Kalman states may still short.",
            "The no-short candidate is reported separately and sets strong negative Kalman states flat.",
            "Bootstrap paths are reconstructed from absolute daily changes without upward shifts; implausible paths are reported separately.",
            "Generator-conditioned regime results are stresses, not statistical confidence intervals.",
        ],
    }
    (output_dir / "one_sided_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Thrifted Jeans one-sided EMA audit complete")
    print(
        comparison.set_index("Candidate")[[
            "P&L",
            "Incremental vs K2",
            "Incremental vs Candidate C",
            "Weak P&L",
            "Weak dip-entry P&L",
            "Max drawdown",
            "Turnover",
        ]].round(2).to_string()
    )
    return {
        "price": price,
        "positions": positions,
        "traces": traces,
        "dip_masks": dip_masks,
        "comparison": comparison,
        "legacy": legacy,
        "buckets": buckets,
        "asymmetry": asymmetry,
        "chronological": chronological,
        "paired_segments": paired_segments,
        "exclusions": exclusions,
        "sensitivity": sensitivity,
        "selection": selection,
        "additive_total": additive_total,
        "additive_pairs": additive_pairs,
        "drift_pairs": drift_pairs,
        "regime": regime,
        "familywise": familywise,
        "correctness": correctness,
        "capital": capital,
        "manifest": manifest,
    }


if __name__ == "__main__":
    run_audit()
