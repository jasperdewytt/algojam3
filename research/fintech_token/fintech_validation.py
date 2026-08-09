"""Validation, diagnostics, figures, and result tables for the Fintech Token.

This file is the executable research audit.  It intentionally rebuilds the
preliminary result in ``tmp/fintech_validation.py`` instead of importing it.
All adaptive policies receive a change prefix ending before the change they
are asked to predict.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_PATH = ROOT / "trader_interface" / "data" / "Fintech Token_price_history.csv"
FIGURE_DIR = HERE / "figures"
RESULT_DIR = HERE / "results"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fintech_models import (  # noqa: E402
    LIMIT,
    constant_position,
    delayed_ewma_switch_positions,
    ewma_ensemble_positions,
    ewma_grid_positions,
    ewma_regime,
    ewma_switch_positions,
    fit_msar_gaussian,
    msar_positions,
    msar_state_detector_positions,
    price_to_changes,
    simple_momentum,
    simple_reversal,
)


EWMA_LAMBDAS = (0.80, 0.85, 0.90, 0.95, 0.97)
EWMA_PERCENTILES = (0.70, 0.75, 0.80, 0.85, 0.90)
EWMA_WARMUPS = (20, 30, 45)
DEFAULT_EWMA = (0.90, 0.80, 30)
DIAGONAL_ENSEMBLE = (
    (0.85, 0.75, 30),
    (0.90, 0.80, 30),
    (0.95, 0.85, 30),
)
Q80_ENSEMBLE = (
    (0.85, 0.80, 30),
    (0.90, 0.80, 30),
    (0.95, 0.80, 30),
)
BROAD5_ENSEMBLE = (
    (0.80, 0.75, 30),
    (0.85, 0.75, 30),
    (0.90, 0.80, 30),
    (0.95, 0.85, 30),
    (0.97, 0.85, 30),
)
NINE_ENSEMBLE = tuple(
    (lam, q, 30)
    for lam in (0.85, 0.90, 0.95)
    for q in (0.75, 0.80, 0.85)
)
MSAR_MIN_TRAIN = 60
MSAR_REFIT_EVERY = 30


def causal_msar_forecast_positions(
    changes: np.ndarray, return_fits: bool = False
) -> object:
    """Frozen causal challenger schedule used throughout the audit."""

    return msar_positions(
        changes,
        min_train=MSAR_MIN_TRAIN,
        refit_every=MSAR_REFIT_EVERY,
        fit_starts=2,
        fit_max_iter=80,
        return_fits=return_fits,
    )


def causal_msar_detector_positions(
    changes: np.ndarray, return_fits: bool = False
) -> object:
    """Frozen HMM state-detector schedule used throughout the audit."""

    return msar_state_detector_positions(
        changes,
        min_train=MSAR_MIN_TRAIN,
        refit_every=MSAR_REFIT_EVERY,
        fit_starts=2,
        fit_max_iter=80,
        return_fits=return_fits,
    )


def load_fintech_data(path: Path = DATA_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Load the supplied Round 1 price path and its one-day changes."""

    frame = pd.read_csv(path)
    if "Price" not in frame:
        raise ValueError("Fintech CSV must contain a Price column")
    prices = frame["Price"].to_numpy(float)
    changes = price_to_changes(prices)
    if len(prices) != 365 or len(changes) != 364:
        raise ValueError(f"expected 365 prices/364 changes, got {len(prices)}/{len(changes)}")
    return prices, changes


def max_drawdown(equity: Sequence[float]) -> float:
    """Maximum peak-to-trough drawdown, including the initial zero."""

    equity = np.asarray(equity, dtype=float)
    curve = np.r_[0.0, equity]
    drawdown = curve - np.maximum.accumulate(curve)
    return float(drawdown.min())


def _quarter_sums(values: np.ndarray) -> List[float]:
    return [float(chunk.sum()) for chunk in np.array_split(values, 4)]


def policy_metrics(
    positions: Sequence[int],
    changes: Sequence[float],
    prices: Optional[Sequence[float]] = None,
    limit: int = LIMIT,
) -> Dict[str, object]:
    """Compute simulator-aligned P&L, risk, capital and turnover metrics."""

    positions = np.asarray(positions)
    changes = np.asarray(changes, dtype=float)
    if len(positions) != len(changes):
        raise ValueError("positions and changes must have equal lengths")
    positions_float = positions.astype(float)
    pnl = positions_float * changes
    equity = np.cumsum(pnl)
    active = np.abs(positions_float) > 0
    active_hit_rate = float((pnl[active] > 0).mean()) if active.any() else 0.0
    # Keep the all-change-day convention used by the preliminary audit while
    # also exposing the more usual active-trade hit rate.
    hit_rate = float((pnl > 0).mean()) if len(pnl) else 0.0
    previous = np.r_[0.0, positions_float[:-1]]
    turnover_units = np.abs(positions_float - previous)
    if prices is None:
        capital = np.full(len(changes), np.nan)
        max_capital = float("nan")
    else:
        prices = np.asarray(prices, dtype=float)
        if len(prices) < len(changes):
            raise ValueError("prices must include the decision-day prices")
        capital = np.abs(positions_float) * prices[: len(changes)]
        max_capital = float(np.max(capital)) if len(capital) else 0.0
    valid_integral = bool(np.all(np.equal(positions_float, np.round(positions_float))))
    valid_limit = bool(np.max(np.abs(positions_float), initial=0) <= limit)
    budget_violations = bool(np.isfinite(max_capital) and max_capital > 600_000.0)
    result: Dict[str, object] = {
        "pnl": float(pnl.sum()),
        "hit_rate": hit_rate,
        "active_hit_rate": active_hit_rate,
        "active_days": int(active.sum()),
        "total_days": int(len(changes)),
        "max_drawdown": max_drawdown(equity),
        "turnover_units": float(turnover_units.sum()),
        "turnover_events": int((turnover_units > 0).sum()),
        "max_capital": max_capital,
        "avg_active_capital": float(np.mean(capital[active])) if active.any() and prices is not None else float("nan"),
        "pnl_per_max_capital": float(pnl.sum() / max_capital) if max_capital and np.isfinite(max_capital) else float("nan"),
        "budget_violations": int(budget_violations),
        "integral_positions": int(valid_integral),
        "within_limit": int(valid_limit),
        "quarter_1_pnl": _quarter_sums(pnl)[0],
        "quarter_2_pnl": _quarter_sums(pnl)[1],
        "quarter_3_pnl": _quarter_sums(pnl)[2],
        "quarter_4_pnl": _quarter_sums(pnl)[3],
        "positive_quarters": int(sum(x > 0 for x in _quarter_sums(pnl))),
        "largest_day_pnl": float(np.max(pnl)) if len(pnl) else 0.0,
        "smallest_day_pnl": float(np.min(pnl)) if len(pnl) else 0.0,
    }
    return result


def pnl_without_largest_jumps(
    positions: Sequence[int], changes: Sequence[float], counts: Iterable[int] = (1, 3, 5, 10)
) -> Dict[int, float]:
    """Set P&L to zero on the largest absolute changes, preserving chronology."""

    positions = np.asarray(positions, dtype=float)
    changes = np.asarray(changes, dtype=float)
    pnl = positions * changes
    order = np.argsort(np.abs(changes))[::-1]
    result = {0: float(pnl.sum())}
    for count in counts:
        keep = np.ones(len(pnl), dtype=bool)
        keep[order[:count]] = False
        result[int(count)] = float(pnl[keep].sum())
    return result


def acf(values: Sequence[float], max_lag: int = 20) -> np.ndarray:
    """Autocorrelation with a common zero-lag normalization."""

    values = np.asarray(values, dtype=float)
    values = values - values.mean()
    denominator = float(np.dot(values, values))
    result = np.full(max_lag + 1, np.nan, dtype=float)
    if denominator <= 0:
        return result
    result[0] = 1.0
    for lag in range(1, max_lag + 1):
        result[lag] = float(np.dot(values[:-lag], values[lag:]) / denominator)
    return result


def basic_statistics(prices: np.ndarray, changes: np.ndarray) -> pd.DataFrame:
    returns = changes / prices[:-1]
    rows = []
    for name, values in (
        ("price", prices),
        ("change", changes),
        ("simple_return", returns),
        ("absolute_change", np.abs(changes)),
    ):
        for statistic, value in (
            ("n", len(values)),
            ("mean", np.mean(values)),
            ("std", np.std(values, ddof=1)),
            ("min", np.min(values)),
            ("q05", np.quantile(values, 0.05)),
            ("q25", np.quantile(values, 0.25)),
            ("median", np.quantile(values, 0.50)),
            ("q75", np.quantile(values, 0.75)),
            ("q95", np.quantile(values, 0.95)),
            ("max", np.max(values)),
        ):
            rows.append({"series": name, "statistic": statistic, "value": float(value)})
    return pd.DataFrame(rows)


def autocorrelation_table(changes: np.ndarray, max_lag: int = 20) -> pd.DataFrame:
    rows = []
    for name, values in (
        ("signed_change", changes),
        ("absolute_change", np.abs(changes)),
        ("squared_change", changes**2),
    ):
        values_acf = acf(values, max_lag=max_lag)
        for lag, value in enumerate(values_acf):
            rows.append({"series": name, "lag": lag, "acf": float(value)})
    return pd.DataFrame(rows)


def volatility_persistence(changes: np.ndarray) -> pd.DataFrame:
    rows = []
    vol = np.sqrt(np.maximum(np.cumsum(changes**2) / np.arange(1, len(changes) + 1), 0))
    for name, values in (
        ("absolute_change", np.abs(changes)),
        ("squared_change", changes**2),
        ("expanding_rms", vol),
    ):
        values = np.asarray(values, dtype=float)
        for lag in (1, 5, 10, 20):
            rows.append(
                {
                    "series": name,
                    "lag": lag,
                    "acf": float(acf(values, lag)[lag]),
                }
            )
    for lam in (0.80, 0.90, 0.95):
        _, ewma_vol, _ = ewma_regime(changes, lam, 0.80, 30)
        for lag in (1, 5, 10, 20):
            rows.append(
                {
                    "series": f"ewma_vol_{lam:.2f}",
                    "lag": lag,
                    "acf": float(acf(ewma_vol, lag)[lag]),
                }
            )
    return pd.DataFrame(rows)


def _bucket_labels(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    return np.digitize(values, np.asarray(edges)[1:-1], right=True)


def conditional_volatility_table(
    changes: np.ndarray, quantile_edges: Sequence[float] = (0.0, 0.40, 0.80, 1.0)
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Descriptive lag-one relation conditional on the latest move size."""

    source = changes[:-1]
    target = changes[1:]
    absolute = np.abs(source)
    thresholds = np.quantile(absolute, quantile_edges)
    labels = _bucket_labels(absolute, thresholds)
    rows = []
    for bucket in range(len(thresholds) - 1):
        mask = labels == bucket
        signed_followup = np.sign(source[mask]) * target[mask]
        rows.append(
            {
                "bucket": bucket,
                "label": ["low", "middle", "high"][bucket] if len(thresholds) == 4 else str(bucket),
                "lower_abs_change": float(thresholds[bucket]),
                "upper_abs_change": float(thresholds[bucket + 1]),
                "n": int(mask.sum()),
                "mean_latest_abs_change": float(absolute[mask].mean()),
                "mean_next_change": float(target[mask].mean()),
                "mean_signed_followup": float(signed_followup.mean()),
                "pnl_per_100": float(100.0 * signed_followup.sum()),
                "hit_rate": float((signed_followup > 0).mean()),
            }
        )
    return pd.DataFrame(rows), thresholds


def conditional_quarter_table(
    changes: np.ndarray, thresholds: np.ndarray
) -> pd.DataFrame:
    source = changes[:-1]
    target = changes[1:]
    labels = _bucket_labels(np.abs(source), thresholds)
    rows = []
    for quarter, indices in enumerate(np.array_split(np.arange(len(source)), 4), 1):
        for bucket in range(len(thresholds) - 1):
            mask = labels[indices] == bucket
            signed = np.sign(source[indices[mask]]) * target[indices[mask]]
            rows.append(
                {
                    "quarter": quarter,
                    "bucket": bucket,
                    "n": int(mask.sum()),
                    "mean_signed_followup": float(signed.mean()) if len(signed) else float("nan"),
                    "pnl_per_100": float(100.0 * signed.sum()),
                    "hit_rate": float((signed > 0).mean()) if len(signed) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def regime_component_table(changes: np.ndarray) -> pd.DataFrame:
    source = changes[:-1]
    target = changes[1:]
    abs_source = np.abs(source)
    edges = np.quantile(abs_source, (0.0, 0.40, 0.80, 1.0))
    labels = _bucket_labels(abs_source, edges)
    rows = []
    for bucket, name in enumerate(("low", "middle", "high")):
        mask = labels == bucket
        signed = np.sign(source[mask]) * target[mask]
        rows.append(
            {
                "regime": name,
                "n": int(mask.sum()),
                "mean_change": float(source[mask].mean()),
                "std_change": float(source[mask].std(ddof=1)),
                "mean_next_change": float(target[mask].mean()),
                "signed_followup": float(signed.mean()),
                "followup_hit_rate": float((signed > 0).mean()),
                "jump_rate_q95": float((abs_source[mask] >= np.quantile(abs_source, 0.95)).mean()),
            }
        )
    return pd.DataFrame(rows)


def ewma_state_table(
    changes: np.ndarray,
    lam: float = DEFAULT_EWMA[0],
    percentile: float = DEFAULT_EWMA[1],
    warmup: int = DEFAULT_EWMA[2],
) -> pd.DataFrame:
    """Conditional next-change behaviour in the causal EWMA states."""

    regimes, vol, cutoffs = ewma_regime(changes, lam, percentile, warmup)
    latest = changes[:-1]
    target = changes[1:]
    state_at_decision = regimes[1:]
    rows = []
    for state, label in ((-1, "calm_reversal"), (1, "volatile_momentum")):
        mask = state_at_decision == state
        signed = np.sign(latest[mask]) * target[mask]
        rows.append(
            {
                "state": label,
                "n": int(mask.sum()),
                "mean_ewma_volatility": float(vol[:-1][mask].mean()) if mask.any() else float("nan"),
                "mean_cutoff": float(np.nanmean(cutoffs[1:][mask])) if mask.any() else float("nan"),
                "mean_signed_followup": float(signed.mean()) if mask.any() else float("nan"),
                "pnl_per_100": float(100.0 * signed.sum()),
                "hit_rate": float((signed > 0).mean()) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def ewma_state_quarter_table(
    changes: np.ndarray,
    lam: float = DEFAULT_EWMA[0],
    percentile: float = DEFAULT_EWMA[1],
    warmup: int = DEFAULT_EWMA[2],
) -> pd.DataFrame:
    regimes, _, _ = ewma_regime(changes, lam, percentile, warmup)
    latest = changes[:-1]
    target = changes[1:]
    rows = []
    for quarter, indices in enumerate(np.array_split(np.arange(len(latest)), 4), 1):
        for state, label in ((-1, "calm_reversal"), (1, "volatile_momentum")):
            mask = regimes[1:][indices] == state
            signed = np.sign(latest[indices[mask]]) * target[indices[mask]]
            rows.append(
                {
                    "quarter": quarter,
                    "state": label,
                    "n": int(mask.sum()),
                    "mean_signed_followup": float(signed.mean()) if mask.any() else float("nan"),
                    "pnl_per_100": float(100.0 * signed.sum()),
                    "hit_rate": float((signed > 0).mean()) if mask.any() else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def ewma_state_reset_table(
    changes: np.ndarray,
    starts: Sequence[int] = (0, 60, 90, 120, 180, 240),
    lam: float = DEFAULT_EWMA[0],
    percentile: float = DEFAULT_EWMA[1],
    warmup: int = DEFAULT_EWMA[2],
) -> pd.DataFrame:
    rows = []
    for start in starts:
        local = changes[int(start) :]
        if len(local) < 3:
            continue
        local_table = ewma_state_table(local, lam, percentile, warmup)
        for record in local_table.to_dict(orient="records"):
            record["start"] = int(start)
            record["n_changes"] = int(len(local))
            rows.append(record)
    return pd.DataFrame(rows)


def duration_table(changes: np.ndarray, threshold_quantile: float = 0.80) -> pd.DataFrame:
    """Empirical run lengths for a descriptive high-volatility indicator."""

    high = np.abs(changes) >= np.quantile(np.abs(changes), threshold_quantile)
    rows = []
    for state, name in ((False, "low_or_middle"), (True, "high")):
        selected = high == state
        lengths: List[int] = []
        i = 0
        while i < len(selected):
            if not selected[i]:
                i += 1
                continue
            j = i
            while j < len(selected) and selected[j]:
                j += 1
            lengths.append(j - i)
            i = j
        if not lengths:
            continue
        transitions = np.sum(selected[:-1] & selected[1:]) / max(np.sum(selected[:-1]), 1)
        geometric_mean = 1.0 / max(1.0 - float(transitions), 1e-6)
        rows.append(
            {
                "state": name,
                "n_observations": int(selected.sum()),
                "n_runs": len(lengths),
                "mean_duration": float(np.mean(lengths)),
                "median_duration": float(np.median(lengths)),
                "max_duration": int(np.max(lengths)),
                "duration_variance": float(np.var(lengths, ddof=1)) if len(lengths) > 1 else 0.0,
                "geometric_mean_from_stay": float(geometric_mean),
                "share_runs_at_least_5": float(np.mean(np.asarray(lengths) >= 5)),
                "run_lengths": ";".join(str(x) for x in lengths),
            }
        )
    return pd.DataFrame(rows)


def independent_reset_conditional(
    changes: np.ndarray, starts: Sequence[int] = (0, 60, 90, 120, 180, 240)
) -> pd.DataFrame:
    rows = []
    for start in starts:
        local = changes[int(start) :]
        if len(local) < 3:
            continue
        table, thresholds = conditional_volatility_table(local)
        high = table.iloc[-1]
        rows.append(
            {
                "start": int(start),
                "n_changes": int(len(local)),
                "low_signed_followup": float(table.iloc[0]["mean_signed_followup"]),
                "middle_signed_followup": float(table.iloc[1]["mean_signed_followup"]),
                "high_signed_followup": float(high["mean_signed_followup"]),
                "high_n": int(high["n"]),
                "threshold_high": float(thresholds[-2]),
            }
        )
    return pd.DataFrame(rows)


def probe_positions(
    changes: np.ndarray,
    prices: np.ndarray,
    name: str,
    limit: int = LIMIT,
) -> np.ndarray:
    """Small, predeclared causal probes for level/jump-duration information."""

    positions = np.zeros(len(changes), dtype=int)
    for t in range(1, len(changes)):
        history = changes[:t]
        latest = history[-1]
        if name == "recent_5_momentum":
            direction = np.sign(np.sum(history[-5:]))
        elif name == "recent_5_reversal":
            direction = -np.sign(np.sum(history[-5:]))
        elif name == "price_level_momentum":
            direction = np.sign(prices[t] - np.mean(prices[: t + 1]))
        elif name == "price_level_reversal":
            direction = -np.sign(prices[t] - np.mean(prices[: t + 1]))
        elif name == "jump_age":
            threshold = np.quantile(np.abs(history), 0.95) if len(history) >= 20 else np.inf
            jump_indices = np.flatnonzero(np.abs(history) >= threshold)
            age = t if len(jump_indices) == 0 else (t - 1 - jump_indices[-1])
            direction = np.sign(latest) if age < 3 else -np.sign(latest)
        else:
            raise ValueError(f"unknown causal probe {name}")
        positions[t] = int(limit * np.sign(direction))
    return positions


def model_policies(
    changes: np.ndarray,
    prices: np.ndarray,
    include_adaptive_challengers: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Dict[str, object]]]]:
    """Build the named policies used in the model-selection table."""

    policies: Dict[str, np.ndarray] = {
        "flat": constant_position(changes, 0),
        "simple_reversal": simple_reversal(changes),
        "simple_momentum": simple_momentum(changes),
        "always_long": constant_position(changes, 1),
        "ewma_0.90_q0.80": ewma_switch_positions(changes, *DEFAULT_EWMA),
        "ewma_diagonal_ensemble": ewma_ensemble_positions(changes, DIAGONAL_ENSEMBLE),
        "ewma_q80_ensemble": ewma_ensemble_positions(changes, Q80_ENSEMBLE),
        "ewma_broad5_ensemble": ewma_ensemble_positions(changes, BROAD5_ENSEMBLE),
        "ewma_nine_ensemble": ewma_ensemble_positions(changes, NINE_ENSEMBLE),
        "ewma_delayed_transition": delayed_ewma_switch_positions(changes, *DEFAULT_EWMA),
    }
    for probe in (
        "recent_5_momentum",
        "recent_5_reversal",
        "price_level_momentum",
        "price_level_reversal",
        "jump_age",
    ):
        policies[f"probe_{probe}"] = probe_positions(changes, prices, probe)
    fit_records: Dict[str, List[Dict[str, object]]] = {}
    if include_adaptive_challengers:
        policies["msar_filtered_forecast"], fit_records["msar_filtered_forecast"] = causal_msar_forecast_positions(
            changes, return_fits=True
        )
        policies["msar_state_detector"], fit_records["msar_state_detector"] = causal_msar_detector_positions(
            changes, return_fits=True
        )
    return policies, fit_records


def policy_table(
    policies: Mapping[str, np.ndarray], changes: np.ndarray, prices: np.ndarray
) -> pd.DataFrame:
    rows = []
    for name, positions in policies.items():
        row = {"model": name}
        row.update(policy_metrics(positions, changes, prices))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pnl", ascending=False).reset_index(drop=True)


def ewma_grid_table(
    changes: np.ndarray,
    prices: np.ndarray,
    lambdas: Sequence[float] = EWMA_LAMBDAS,
    percentiles: Sequence[float] = EWMA_PERCENTILES,
    warmups: Sequence[int] = EWMA_WARMUPS,
) -> Tuple[pd.DataFrame, Dict[Tuple[float, float, int], np.ndarray]]:
    grid = ewma_grid_positions(changes, lambdas, percentiles, warmups)
    rows = []
    for (lam, q, warmup), positions in grid.items():
        row = {"lambda": lam, "percentile": q, "warmup": warmup}
        row.update(policy_metrics(positions, changes, prices))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pnl", ascending=False).reset_index(drop=True), grid


def evaluate_restarts(
    policy_builder: Callable[[np.ndarray, np.ndarray], np.ndarray],
    name: str,
    changes: np.ndarray,
    prices: np.ndarray,
    starts: Sequence[int] = (0, 60, 90, 120, 180, 240),
    warmup: int = 0,
) -> pd.DataFrame:
    rows = []
    for start in starts:
        local_changes = changes[int(start) :]
        local_prices = prices[int(start) :]
        positions = policy_builder(local_changes, local_prices)
        row = {"model": name, "start": int(start), "n_changes": len(local_changes)}
        row.update(policy_metrics(positions, local_changes, local_prices))
        if warmup:
            first = min(len(local_changes), warmup + 1)
            post = policy_metrics(
                positions[first:], local_changes[first:], local_prices[first:]
            )
            row["post_warmup_pnl"] = post["pnl"]
            row["post_warmup_active_days"] = post["active_days"]
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_fixed_blocks(
    policy_builder: Callable[[np.ndarray, np.ndarray], np.ndarray],
    name: str,
    changes: np.ndarray,
    prices: np.ndarray,
    block_length: int,
) -> pd.DataFrame:
    rows = []
    for start in range(0, len(changes), block_length):
        stop = min(start + block_length, len(changes))
        local_changes = changes[start:stop]
        local_prices = prices[start : stop + 1]
        positions = policy_builder(local_changes, local_prices)
        row = {
            "model": name,
            "block_length": int(block_length),
            "start": int(start),
            "stop": int(stop),
        }
        row.update(policy_metrics(positions, local_changes, local_prices))
        rows.append(row)
    return pd.DataFrame(rows)


def moving_block_sample(
    values: np.ndarray, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Circular moving-block bootstrap preserving local order."""

    values = np.asarray(values)
    blocks = []
    while sum(len(block) for block in blocks) < len(values):
        start = int(rng.integers(0, len(values)))
        indices = (start + np.arange(block_length)) % len(values)
        blocks.append(values[indices])
    return np.concatenate(blocks)[: len(values)].copy()


def bootstrap_results(
    changes: np.ndarray,
    policy_builders: Mapping[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    block_lengths: Sequence[int] = (5, 10, 20),
    reps: int = 100,
    reps_by_model: Optional[Mapping[str, int]] = None,
    seed: int = 20260808,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for block_length in block_lengths:
        for name, builder in policy_builders.items():
            model_reps = int(reps_by_model.get(name, reps)) if reps_by_model else int(reps)
            values = []
            for _ in range(model_reps):
                sample = moving_block_sample(changes, block_length, rng)
                sample_prices = np.r_[803.0, 803.0 + np.cumsum(sample)]
                positions = builder(sample, sample_prices)
                values.append(policy_metrics(positions, sample, sample_prices)["pnl"])
            values = np.asarray(values, dtype=float)
            rows.append(
                {
                    "model": name,
                    "block_length": int(block_length),
                    "reps": int(model_reps),
                    "mean_pnl": float(values.mean()),
                    "q05_pnl": float(np.quantile(values, 0.05)),
                    "median_pnl": float(np.quantile(values, 0.50)),
                    "q95_pnl": float(np.quantile(values, 0.95)),
                    "positive_fraction": float(np.mean(values > 0)),
                }
            )
    return pd.DataFrame(rows)


def fixed_candidate_family(
    changes: np.ndarray, prices: np.ndarray
) -> Dict[str, np.ndarray]:
    """All fixed/grid/probe policies counted in the family-wise null."""

    candidates: Dict[str, np.ndarray] = {
        "flat": constant_position(changes, 0),
        "simple_reversal": simple_reversal(changes),
        "simple_momentum": simple_momentum(changes),
        "always_long": constant_position(changes, 1),
    }
    grid = ewma_grid_positions(changes, EWMA_LAMBDAS, EWMA_PERCENTILES, EWMA_WARMUPS)
    for (lam, q, warmup), positions in grid.items():
        candidates[f"ewma_lam{lam:.2f}_q{q:.2f}_w{warmup}"] = positions
    ensemble_defs = {
        "ewma_diagonal_ensemble": DIAGONAL_ENSEMBLE,
        "ewma_q80_ensemble": Q80_ENSEMBLE,
        "ewma_broad5_ensemble": BROAD5_ENSEMBLE,
        "ewma_nine_ensemble": NINE_ENSEMBLE,
    }
    for name, configs in ensemble_defs.items():
        candidates[name] = ewma_ensemble_positions(changes, configs)
    for probe in (
        "recent_5_momentum",
        "recent_5_reversal",
        "price_level_momentum",
        "price_level_reversal",
        "jump_age",
    ):
        candidates[f"probe_{probe}"] = probe_positions(changes, prices, probe)
    return candidates


def family_scores(
    candidates: Mapping[str, np.ndarray], changes: np.ndarray
) -> Dict[str, float]:
    return {name: float(np.sum(pos * changes)) for name, pos in candidates.items()}


def family_time_shuffle_null(
    changes: np.ndarray,
    prices: np.ndarray,
    selected_name: str = "ewma_diagonal_ensemble",
    n_perm: int = 200,
    seed: int = 20260808,
    include_msar: bool = True,
) -> Dict[str, object]:
    """Time-shuffle null with a maximum over every fixed candidate tested."""

    observed_candidates = fixed_candidate_family(changes, prices)
    if include_msar:
        observed_candidates["msar_filtered_forecast"] = causal_msar_forecast_positions(changes)
        observed_candidates["msar_state_detector"] = causal_msar_detector_positions(changes)
    observed = family_scores(observed_candidates, changes)
    rng = np.random.default_rng(seed)
    null_selected: List[float] = []
    null_best: List[float] = []
    null_msar_best: List[float] = []
    for _ in range(n_perm):
        sample = rng.permutation(changes)
        sample_prices = np.r_[prices[0], prices[0] + np.cumsum(sample)]
        candidates = fixed_candidate_family(sample, sample_prices)
        fixed_scores = family_scores(candidates, sample)
        if include_msar:
            msar_pos = causal_msar_forecast_positions(sample)
            detector_pos = causal_msar_detector_positions(sample)
            fixed_scores["msar_filtered_forecast"] = float(np.sum(msar_pos * sample))
            fixed_scores["msar_state_detector"] = float(np.sum(detector_pos * sample))
            null_msar_best.append(
                max(fixed_scores["msar_filtered_forecast"], fixed_scores["msar_state_detector"])
            )
        null_selected.append(float(fixed_scores[selected_name]))
        null_best.append(float(max(fixed_scores.values())))
    selected_observed = float(observed[selected_name])
    best_observed = float(max(observed.values()))
    p_selected = (1 + sum(value >= selected_observed for value in null_selected)) / (n_perm + 1)
    p_family = (1 + sum(value >= best_observed for value in null_best)) / (n_perm + 1)
    return {
        "type": "time_shuffle",
        "n_perm": int(n_perm),
        "seed": int(seed),
        "family_count": int(len(observed)),
        "selected_name": selected_name,
        "selected_observed": selected_observed,
        "selected_p_value": float(p_selected),
        "best_observed": best_observed,
        "family_wise_p_value": float(p_family),
        "null_selected_q05": float(np.quantile(null_selected, 0.05)),
        "null_selected_q50": float(np.quantile(null_selected, 0.50)),
        "null_selected_q95": float(np.quantile(null_selected, 0.95)),
        "null_best_q95": float(np.quantile(null_best, 0.95)),
        "msar_null_best_q95": float(np.quantile(null_msar_best, 0.95)) if null_msar_best else float("nan"),
        "observed_scores": observed,
    }


def adaptive_model_shuffle_null(
    changes: np.ndarray,
    selected_name: str = "msar_filtered_forecast",
    n_perm: int = 20,
    seed: int = 20260811,
) -> Dict[str, object]:
    """Small model-specific null for the two adaptive MS-AR challengers.

    The larger family-wise null below covers every fixed/grid/probe
    configuration.  Re-fitting an MS-AR on every shuffled path is much more
    expensive, so it is reported separately with its own declared sample size
    rather than silently mixing a lower-powered test into the fixed family.
    """

    observed_positions = {
        "msar_filtered_forecast": causal_msar_forecast_positions(changes),
        "msar_state_detector": causal_msar_detector_positions(changes),
    }
    observed = family_scores(observed_positions, changes)
    rng = np.random.default_rng(seed)
    selected_null = []
    family_null = []
    for _ in range(n_perm):
        sample = rng.permutation(changes)
        scores = {
            "msar_filtered_forecast": float(np.sum(causal_msar_forecast_positions(sample) * sample)),
            "msar_state_detector": float(np.sum(causal_msar_detector_positions(sample) * sample)),
        }
        selected_null.append(scores[selected_name])
        family_null.append(max(scores.values()))
    return {
        "type": "time_shuffle_adaptive_msar",
        "n_perm": int(n_perm),
        "seed": int(seed),
        "family_count": 2,
        "selected_name": selected_name,
        "selected_observed": float(observed[selected_name]),
        "selected_p_value": float(
            (1 + sum(x >= observed[selected_name] for x in selected_null)) / (n_perm + 1)
        ),
        "family_wise_p_value": float(
            (1 + sum(x >= max(observed.values()) for x in family_null)) / (n_perm + 1)
        ),
        "null_selected_q95": float(np.quantile(selected_null, 0.95)),
        "null_family_q95": float(np.quantile(family_null, 0.95)),
        "observed_scores": observed,
    }


def circular_shift_null(
    changes: np.ndarray,
    prices: np.ndarray,
    selected_name: str = "ewma_diagonal_ensemble",
    n_shifts: int = 200,
    seed: int = 20260809,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    offsets = rng.choice(np.arange(1, len(changes)), size=min(n_shifts, len(changes) - 1), replace=False)
    observed_candidates = fixed_candidate_family(changes, prices)
    observed = family_scores(observed_candidates, changes)
    selected_null = []
    best_null = []
    for offset in offsets:
        sample = np.roll(changes, int(offset))
        sample_prices = np.r_[prices[0], prices[0] + np.cumsum(sample)]
        scores = family_scores(fixed_candidate_family(sample, sample_prices), sample)
        selected_null.append(scores[selected_name])
        best_null.append(max(scores.values()))
    return {
        "type": "circular_shift",
        "n_shifts": int(len(offsets)),
        "selected_name": selected_name,
        "selected_observed": float(observed[selected_name]),
        "selected_p_value": float((1 + sum(x >= observed[selected_name] for x in selected_null)) / (len(selected_null) + 1)),
        "family_wise_p_value": float((1 + sum(x >= max(observed.values()) for x in best_null)) / (len(best_null) + 1)),
        "family_count": int(len(observed)),
    }


def block_permutation_null(
    changes: np.ndarray,
    prices: np.ndarray,
    block_lengths: Sequence[int] = (7, 14, 21),
    reps: int = 100,
    seed: int = 20260810,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    observed = fixed_candidate_family(changes, prices)
    observed_scores = family_scores(observed, changes)
    rows = []
    for block_length in block_lengths:
        blocks = [changes[i : i + block_length] for i in range(0, len(changes), block_length)]
        selected_null = []
        family_null = []
        for _ in range(reps):
            order = rng.permutation(len(blocks))
            sample = np.concatenate([blocks[i] for i in order])[: len(changes)]
            sample_prices = np.r_[prices[0], prices[0] + np.cumsum(sample)]
            scores = family_scores(fixed_candidate_family(sample, sample_prices), sample)
            selected_null.append(scores["ewma_diagonal_ensemble"])
            family_null.append(max(scores.values()))
        rows.append(
            {
                "block_length": int(block_length),
                "reps": int(reps),
                "selected_p_value": float((1 + sum(x >= observed_scores["ewma_diagonal_ensemble"] for x in selected_null)) / (reps + 1)),
                "family_wise_p_value": float((1 + sum(x >= max(observed_scores.values()) for x in family_null)) / (reps + 1)),
                "selected_null_q95": float(np.quantile(selected_null, 0.95)),
                "family_null_q95": float(np.quantile(family_null, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def future_perturbation_audit(
    changes: np.ndarray,
    policy_builders: Mapping[str, Callable[[np.ndarray], np.ndarray]],
    cut_points: Sequence[int] = (60, 90, 120, 180, 240, 300),
) -> pd.DataFrame:
    """Verify that changing later observations cannot change earlier positions."""

    rows = []
    for name, builder in policy_builders.items():
        original = np.asarray(builder(changes))
        for cut in cut_points:
            perturbed = changes.copy()
            if cut + 1 < len(perturbed):
                perturbed[cut + 1 :] = -perturbed[cut + 1 :] + 0.37
            altered = np.asarray(builder(perturbed))
            prefix_equal = bool(np.array_equal(original[: cut + 1], altered[: cut + 1]))
            rows.append(
                {
                    "model": name,
                    "cut": int(cut),
                    "prefix_equal": int(prefix_equal),
                    "max_prefix_difference": float(np.max(np.abs(original[: cut + 1] - altered[: cut + 1]))),
                }
            )
    return pd.DataFrame(rows)


def delayed_transition_table(
    changes: np.ndarray,
    prices: np.ndarray,
    names_and_builders: Mapping[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for name, builder in names_and_builders.items():
        positions = builder(changes, prices)
        row = {"model": name, "condition": "ordinary"}
        row.update(policy_metrics(positions, changes, prices))
        rows.append(row)
    return pd.DataFrame(rows)


def parameter_prefix_table(
    changes: np.ndarray,
    prefixes: Sequence[int] = (90, 120, 180, 240, 300, 364),
    fit_starts: int = 2,
    fit_max_iter: int = 80,
) -> pd.DataFrame:
    rows = []
    for prefix in prefixes:
        fit = fit_msar_gaussian(
            changes[:prefix], n_starts=fit_starts, max_iter=fit_max_iter
        )
        for state, label in enumerate(("low_variance", "high_variance")):
            rows.append(
                {
                    "prefix": int(prefix),
                    "state": label,
                    "mean": float(fit.means[state]),
                    "phi": float(fit.phi[state]),
                    "sigma": float(fit.sigma[state]),
                    "stay_probability": float(fit.transition[state, state]),
                    "loglik": float(fit.loglik),
                    "converged": int(fit.converged),
                    "iterations": int(fit.iterations),
                    "failed_starts": int(fit.failed_starts),
                    "parameter_count": fit.parameter_count,
                }
            )
    return pd.DataFrame(rows)


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    return value


def _save_figures(
    prices: np.ndarray,
    changes: np.ndarray,
    regimes: np.ndarray,
    vol: np.ndarray,
    conditional: pd.DataFrame,
    policies: Mapping[str, np.ndarray],
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.close("all")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(prices, color="#1565C0", linewidth=1.7)
    axes[0].set_title("Fintech Token price and one-day changes")
    axes[0].set_ylabel("Price (AUD)")
    axes[0].grid(alpha=0.25)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].plot(np.arange(1, len(prices)), changes, color="#455A64", linewidth=0.9)
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Change")
    axes[1].grid(alpha=0.25)
    fig.savefig(FIGURE_DIR / "price_changes.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(np.arange(1, len(prices)), prices[1:], color="#1565C0", linewidth=1.2)
    axes[0].set_title("Causal EWMA volatility state (lambda=0.90, expanding q=0.80)")
    axes[0].set_ylabel("Price (AUD)")
    axes[0].grid(alpha=0.25)
    decision_x = np.arange(1, len(changes))
    decision_vol = vol[:-1]
    axes[1].plot(decision_x, decision_vol, color="#6A1B9A", linewidth=1.2, label="EWMA volatility")
    high = regimes[1:] == 1
    axes[1].scatter(decision_x[high], decision_vol[high], s=12, color="#D84315", label="volatile / momentum")
    axes[1].set_xlabel("Decision day")
    axes[1].set_ylabel("EWMA std. dev.")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.savefig(FIGURE_DIR / "ewma_regimes.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(conditional["label"], conditional["mean_signed_followup"], color=["#43A047", "#78909C", "#D84315"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Next-change relation by latest move size")
    ax.set_ylabel("Mean sign(latest change) × next change")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIGURE_DIR / "conditional_volatility.png", dpi=180)
    plt.close(fig)

    state_conditional = ewma_state_table(changes)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(
        state_conditional["state"],
        state_conditional["mean_signed_followup"],
        color=["#43A047", "#D84315"],
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Next-change relation in causal EWMA states")
    ax.set_ylabel("Mean sign(latest change) × next change")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIGURE_DIR / "ewma_state_conditional.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    colors = {
        "simple_reversal": "#546E7A",
        "simple_momentum": "#90A4AE",
        "ewma_diagonal_ensemble": "#1565C0",
        "msar_filtered_forecast": "#D84315",
        "msar_state_detector": "#6A1B9A",
    }
    for name in colors:
        if name not in policies:
            continue
        pnl = np.cumsum(policies[name] * changes)
        ax.plot(np.arange(1, len(pnl) + 1), pnl, label=name, color=colors[name], linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("Causal cumulative P&L")
    ax.set_xlabel("Change index")
    ax.set_ylabel("AUD")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.savefig(FIGURE_DIR / "cumulative_pnl.png", dpi=180)
    plt.close(fig)


def run_all(
    bootstrap_reps: int = 100,
    permutation_reps: int = 200,
    block_permutation_reps: int = 100,
) -> Dict[str, object]:
    """Run the complete isolated audit and write compact results."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    def stage(label: str) -> None:
        print(f"[fintech] {label}", flush=True)

    stage("load data")
    prices, changes = load_fintech_data()

    stage("phase 1 diagnostics")
    stats = basic_statistics(prices, changes)
    acf_values = autocorrelation_table(changes)
    volatility = volatility_persistence(changes)
    conditional, thresholds = conditional_volatility_table(changes)
    conditional_quarters = conditional_quarter_table(changes, thresholds)
    components = regime_component_table(changes)
    ewma_states = ewma_state_table(changes)
    ewma_state_quarters = ewma_state_quarter_table(changes)
    ewma_state_resets = ewma_state_reset_table(changes)
    durations = duration_table(changes)
    reset_conditional = independent_reset_conditional(changes)

    stage("model comparison")
    policies, fit_records = model_policies(changes, prices, include_adaptive_challengers=True)
    model_results = policy_table(policies, changes, prices)
    grid_results, grid = ewma_grid_table(changes, prices)
    grid_best = grid_results.iloc[0].to_dict()
    grid_plateau = grid_results.assign(
        within_10pct_of_best=grid_results["pnl"] >= 0.90 * float(grid_results["pnl"].max()),
        profitable=grid_results["pnl"] > 0,
    )

    stage("walk-forward restarts")
    builders = {
        "ewma_diagonal_ensemble": lambda c, p: ewma_ensemble_positions(c, DIAGONAL_ENSEMBLE),
        "simple_reversal": lambda c, p: simple_reversal(c),
        "msar_filtered_forecast": lambda c, p: causal_msar_forecast_positions(c),
        "msar_state_detector": lambda c, p: causal_msar_detector_positions(c),
    }
    restart_frames = [
        evaluate_restarts(
            builder,
            name,
            changes,
            prices,
            warmup=30 if "ewma" in name else 0,
        )
        for name, builder in builders.items()
    ]
    restarts = pd.concat(restart_frames, ignore_index=True)
    independent_blocks = pd.concat(
        [
            evaluate_fixed_blocks(builder, name, changes, prices, block_length)
            for block_length in (60, 91)
            for name, builder in {
                "ewma_diagonal_ensemble": builders["ewma_diagonal_ensemble"],
                "simple_reversal": builders["simple_reversal"],
                "msar_filtered_forecast": builders["msar_filtered_forecast"],
            }.items()
        ],
        ignore_index=True,
    )

    stage("moving-block bootstrap")
    bootstrap = bootstrap_results(
        changes,
        {
            "simple_reversal": builders["simple_reversal"],
            "ewma_diagonal_ensemble": builders["ewma_diagonal_ensemble"],
            "msar_filtered_forecast": builders["msar_filtered_forecast"],
        },
        reps=bootstrap_reps,
        reps_by_model={"msar_filtered_forecast": max(1, bootstrap_reps // 10)},
    )
    stage("fixed-family time shuffle")
    shuffle = family_time_shuffle_null(
        changes, prices, n_perm=permutation_reps, include_msar=False
    )
    stage("adaptive challenger shuffle")
    adaptive_shuffle = adaptive_model_shuffle_null(
        # The fixed 88-policy family receives the full 200-shuffle test.  The
        # refit-every-path MS-AR null is reported separately at five paths to
        # keep the full audit reproducible on the available runtime.
        changes, n_perm=max(3, min(5, permutation_reps // 40))
    )
    stage("circular and block permutations")
    circular = circular_shift_null(changes, prices)
    block_permutation = block_permutation_null(
        changes, prices, reps=block_permutation_reps
    )
    stage("jump exclusions and delayed transitions")
    exclusion_rows = []
    for name in (
        "simple_reversal",
        "ewma_diagonal_ensemble",
        "msar_filtered_forecast",
    ):
        for count, pnl in pnl_without_largest_jumps(policies[name], changes).items():
            exclusion_rows.append({"model": name, "excluded_largest_jumps": count, "pnl": pnl})
    exclusions = pd.DataFrame(exclusion_rows)

    delay_builders = {
        "ewma_ordinary": lambda c, p: ewma_switch_positions(c, *DEFAULT_EWMA),
        "ewma_delayed": lambda c, p: delayed_ewma_switch_positions(c, *DEFAULT_EWMA),
        "ensemble_ordinary": lambda c, p: ewma_ensemble_positions(c, DIAGONAL_ENSEMBLE),
        "ensemble_delayed": lambda c, p: np.sign(
            delayed_ewma_switch_positions(c, *DIAGONAL_ENSEMBLE[0])
            + delayed_ewma_switch_positions(c, *DIAGONAL_ENSEMBLE[1])
            + delayed_ewma_switch_positions(c, *DIAGONAL_ENSEMBLE[2])
        ).astype(int)
        * LIMIT,
    }
    delay_results = pd.concat(
        [
            evaluate_restarts(builder, name, changes, prices, starts=(0,))
            for name, builder in delay_builders.items()
        ],
        ignore_index=True,
    )

    causal_builders = {
        "ewma_diagonal_ensemble": lambda c: ewma_ensemble_positions(c, DIAGONAL_ENSEMBLE),
        "msar_filtered_forecast": lambda c: causal_msar_forecast_positions(c),
        "msar_state_detector": lambda c: causal_msar_detector_positions(c),
    }
    stage("future perturbation and parameter prefixes")
    lookahead = future_perturbation_audit(
        changes, causal_builders, cut_points=(90, 180, 300)
    )
    parameter_prefixes = parameter_prefix_table(changes)
    regimes, ewma_vol, _ = ewma_regime(changes, *DEFAULT_EWMA)

    stage("figures")
    _save_figures(prices, changes, regimes, ewma_vol, conditional, policies)

    # The fixed family has 4 baselines + 75 grid settings + 4 ensembles + 5
    # secondary probes; the two adaptive challengers are added separately.
    model_count = {
        "ewma_grid_configurations": len(EWMA_LAMBDAS) * len(EWMA_PERCENTILES) * len(EWMA_WARMUPS),
        "fixed_candidate_family": 4 + 75 + 4 + 5,
        "adaptive_challengers": 2,
        "model_table_rows": int(len(model_results)),
    }

    stage("write result tables")
    stats.to_csv(RESULT_DIR / "basic_statistics.csv", index=False)
    acf_values.to_csv(RESULT_DIR / "acf.csv", index=False)
    volatility.to_csv(RESULT_DIR / "volatility_persistence.csv", index=False)
    conditional.to_csv(RESULT_DIR / "conditional_volatility.csv", index=False)
    conditional_quarters.to_csv(RESULT_DIR / "conditional_quarters.csv", index=False)
    components.to_csv(RESULT_DIR / "regime_components.csv", index=False)
    ewma_states.to_csv(RESULT_DIR / "ewma_states.csv", index=False)
    ewma_state_quarters.to_csv(RESULT_DIR / "ewma_state_quarters.csv", index=False)
    ewma_state_resets.to_csv(RESULT_DIR / "ewma_state_resets.csv", index=False)
    durations.to_csv(RESULT_DIR / "regime_durations.csv", index=False)
    reset_conditional.to_csv(RESULT_DIR / "conditional_resets.csv", index=False)
    model_results.to_csv(RESULT_DIR / "model_results.csv", index=False)
    grid_results.to_csv(RESULT_DIR / "ewma_grid.csv", index=False)
    grid_plateau.to_csv(RESULT_DIR / "ewma_grid_plateau.csv", index=False)
    restarts.to_csv(RESULT_DIR / "walk_forward_restarts.csv", index=False)
    independent_blocks.to_csv(RESULT_DIR / "independent_blocks.csv", index=False)
    bootstrap.to_csv(RESULT_DIR / "moving_block_bootstrap.csv", index=False)
    block_permutation.to_csv(RESULT_DIR / "block_permutations.csv", index=False)
    exclusions.to_csv(RESULT_DIR / "jump_exclusions.csv", index=False)
    delay_results.to_csv(RESULT_DIR / "transition_delay.csv", index=False)
    lookahead.to_csv(RESULT_DIR / "future_perturbation.csv", index=False)
    parameter_prefixes.to_csv(RESULT_DIR / "msar_prefix_stability.csv", index=False)

    summary = {
        "data": {
            "price_count": len(prices),
            "change_count": len(changes),
            "first_price": prices[0],
            "last_price": prices[-1],
        },
        "selected_model": "ewma_diagonal_ensemble",
        "default_ewma": list(DEFAULT_EWMA),
        "diagonal_ensemble": [list(x) for x in DIAGONAL_ENSEMBLE],
        "model_count": model_count,
        "grid_best": json_safe(grid_best),
        "grid_plateau": {
            "profitable_count": int(grid_plateau["profitable"].sum()),
            "within_10pct_count": int(grid_plateau["within_10pct_of_best"].sum()),
            "within_10pct_positive_count": int(
                (grid_plateau["within_10pct_of_best"] & grid_plateau["profitable"]).sum()
            ),
        },
        "model_results": json_safe(model_results.to_dict(orient="records")),
        "conditional": json_safe(conditional.to_dict(orient="records")),
        "regime_components": json_safe(components.to_dict(orient="records")),
        "ewma_states": json_safe(ewma_states.to_dict(orient="records")),
        "ewma_state_quarters": json_safe(ewma_state_quarters.to_dict(orient="records")),
        "ewma_state_resets": json_safe(ewma_state_resets.to_dict(orient="records")),
        "durations": json_safe(durations.to_dict(orient="records")),
        "walk_forward": json_safe(restarts.to_dict(orient="records")),
        "independent_blocks": json_safe(independent_blocks.to_dict(orient="records")),
        "bootstrap": json_safe(bootstrap.to_dict(orient="records")),
        "shuffle": json_safe(shuffle),
        "adaptive_shuffle": json_safe(adaptive_shuffle),
        "circular_shift": json_safe(circular),
        "block_permutation": json_safe(block_permutation.to_dict(orient="records")),
        "jump_exclusions": json_safe(exclusions.to_dict(orient="records")),
        "transition_delay": json_safe(delay_results.to_dict(orient="records")),
        "future_perturbation": json_safe(lookahead.to_dict(orient="records")),
        "msar_prefix_stability": json_safe(parameter_prefixes.to_dict(orient="records")),
        "msar_fit_records": json_safe(fit_records),
    }
    with (RESULT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    with (RESULT_DIR / "model_counts.json").open("w", encoding="utf-8") as handle:
        json.dump(model_count, handle, indent=2)

    print("Fintech Token validation complete")
    print(model_results[["model", "pnl", "max_drawdown", "hit_rate", "max_capital"]].to_string(index=False))
    print("Grid best:", grid_best)
    print("Family shuffle:", {key: shuffle[key] for key in ("family_count", "selected_p_value", "family_wise_p_value")})
    print("Future perturbation all equal:", bool(lookahead["prefix_equal"].all()))
    return summary


if __name__ == "__main__":
    run_all()
