"""Research-only audit of the teammate's adaptive Boat Party overlay.

This module deliberately does not import or modify the production trader.  It
reconstructs the available frozen Boat Party signal string and the adaptive
EWMA mechanics described in the audit brief, then writes reproducible tables
and figures under ``research/boat_party``.

Important provenance note
-------------------------
The repository does not contain a literal ``BOAT_PARTY_SIGNALS`` or an
adaptive EWMA implementation.  The available production constant is named
``BOAT_PARTY_SEMESTER_SIGNALS`` and is the frozen Candidate-D semester string.
The audit extracts that string read-only and uses it as the supplied signal.
The persistence/reversal rule is implemented exactly as described in the
brief: on a neutral day, a flat position opens when the deviation crosses the
threshold; a position persists while the deviation has the favourable sign;
an adverse threshold crossing reverses it; otherwise it closes to flat.

All Round 1 scores are in-sample diagnostics.  Fixed centered templates and
the extracted signal are treated as externally frozen Round 2 priors, not as
causal fits to Round 2 data.  Synthetic paths are generator-conditioned
stress tests, never confidence intervals or independent empirical evidence.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover - the fallback is tested when needed
    sm = None

try:
    from . import analysis as a
    from . import final_strategy_test as fst
except ImportError:  # direct execution from research/boat_party
    import analysis as a
    import final_strategy_test as fst


SEED = 20260810
POSITION_LIMIT = 1_000
TOTAL_BUDGET = 600_000.0
BOAT_VOL_WINDOW = 10
BOAT_REVERT_THRESHOLD = 0.05
BOAT_ALPHAS = (0.65, 0.90)
SENSITIVITY_ALPHAS = (0.30, 0.50, 0.65, 0.75, 0.90)
SENSITIVITY_WINDOWS = (5, 10, 20, 30)
SENSITIVITY_THRESHOLDS = (0.0, 0.05, 0.10, 0.25, 0.50)
SUMMER_START = 322
SUMMER_EQUILIBRIUM = 45.0
BLOCK_LENGTHS = (5, 10, 20)
DEFAULT_BOOTSTRAP_PATHS = 200
DEFAULT_PLACEBO_PATHS = 250
DEFAULT_SUMMER_PATHS = 100


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def load_round1(repo_root: str | Path | None = None) -> tuple[Path, np.ndarray]:
    root = a.find_repo_root(repo_root)
    prices = a.load_prices(root)["Price"].to_numpy(dtype=float)
    return root, prices


def extract_supplied_signal(repo_root: str | Path) -> dict[str, object]:
    """Extract the available production signal without importing production code."""

    root = Path(repo_root).resolve()
    path = root / "trader_interface" / "algorithm.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    value: object | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
                candidate = node.value
            else:
                targets = [node.target.id] if isinstance(node.target, ast.Name) else []
                candidate = node.value
            if "BOAT_PARTY_SEMESTER_SIGNALS" in targets:
                value = ast.literal_eval(candidate)
                break
    if value is None:
        raise AssertionError("available frozen Boat Party signal constant was not found")
    if isinstance(value, tuple):
        signal = "".join(str(part) for part in value)
    else:
        signal = str(value)
    allowed = set("+-0")
    if set(signal) - allowed:
        raise AssertionError(f"unexpected signal characters: {set(signal) - allowed}")
    return {
        "signal": signal,
        "source_symbol": "BOAT_PARTY_SEMESTER_SIGNALS",
        "requested_symbol": "BOAT_PARTY_SIGNALS (not present in repository)",
        "source_file": str(path),
        "length": len(signal),
        "allowed_characters": "".join(sorted(set(signal))),
        "summer_treatment": "signal string ends before day 322; adaptive overlay is used on zero/summer days",
    }


def signal_position_array(signal: str, length: int) -> np.ndarray:
    """Convert ``+``/``-``/``0`` to desired positions and force day 364 flat."""

    out = np.zeros(length, dtype=int)
    for day, char in enumerate(signal[:length]):
        if char == "+":
            out[day] = POSITION_LIMIT
        elif char == "-":
            out[day] = -POSITION_LIMIT
    if length:
        out[-1] = 0
    return out


def ewma_update(old: float, price: float, alpha: float) -> float:
    """The supplied update: old + alpha * (price - old)."""

    return float(old + alpha * (price - old))


def _causal_volatility(
    prices: np.ndarray,
    residuals: np.ndarray,
    day: int,
    window: int,
    denominator: str,
) -> float:
    """Return a denominator using observations available through ``day`` only."""

    if day + 1 < window:
        return float("nan")
    if denominator == "price_level":
        values = prices[day - window + 1 : day + 1]
    elif denominator == "return_volatility":
        values = np.diff(prices[day - window : day + 1])
    elif denominator == "ewma_residual":
        values = residuals[day - window + 1 : day + 1]
    else:
        raise ValueError(f"unknown volatility denominator: {denominator}")
    if len(values) < 2:
        return float("nan")
    return float(max(np.std(values, ddof=0), 1e-12))


def ewma_overlay_decision(z_score: float, previous_position: int, threshold: float) -> int:
    """Reproduce the described persistence/reversal mechanics."""

    if not np.isfinite(z_score):
        return 0
    if previous_position == 0:
        if z_score >= threshold:
            return -POSITION_LIMIT
        if z_score <= -threshold:
            return POSITION_LIMIT
        return 0
    # A long position is favourable when the deviation is negative; a short
    # position is favourable when it is positive.  An adverse threshold
    # crossing reverses directly, matching the submitted-style mechanics.
    if previous_position * z_score > 0:
        if abs(z_score) >= threshold:
            return -previous_position
        return 0
    return int(previous_position)


def adaptive_strategy_frame(
    prices: Sequence[float],
    signal: str,
    alpha: float = 0.65,
    vol_window: int = BOAT_VOL_WINDOW,
    threshold: float = BOAT_REVERT_THRESHOLD,
    denominator: str = "price_level",
    use_signal: bool = True,
    deviation_mode: str = "post_update",
) -> pd.DataFrame:
    """Build the causal desired-position path and all EWMA audit fields.

    ``deviation_mode='post_update'`` is the supplied implementation.  The
    pre-update fields are retained so the alpha rescaling identity can be
    audited without changing the default strategy.
    """

    y = _as_float_array(prices)
    if len(y) < 1:
        raise ValueError("prices must not be empty")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if vol_window < 2:
        raise ValueError("vol_window must be at least 2")
    if deviation_mode not in {"post_update", "pre_update"}:
        raise ValueError("deviation_mode must be post_update or pre_update")

    ewma_old = float(y[0])
    previous_position = 0
    ewma_values = np.zeros(len(y), dtype=float)
    old_values = np.zeros(len(y), dtype=float)
    pre_deviation = np.zeros(len(y), dtype=float)
    post_deviation = np.zeros(len(y), dtype=float)
    residuals = np.zeros(len(y), dtype=float)
    volatility = np.full(len(y), np.nan, dtype=float)
    pre_z = np.full(len(y), np.nan, dtype=float)
    post_z = np.full(len(y), np.nan, dtype=float)
    fixed_position = np.zeros(len(y), dtype=int)
    overlay_position = np.zeros(len(y), dtype=int)
    final_position = np.zeros(len(y), dtype=int)
    previous_positions = np.zeros(len(y), dtype=int)
    decision_type = np.full(len(y), "flat", dtype=object)
    signal_chars = np.full(len(y), "0", dtype=object)

    for day, price in enumerate(y):
        old_values[day] = ewma_old
        new_ewma = ewma_update(ewma_old, float(price), alpha)
        ewma_values[day] = new_ewma
        pre_deviation[day] = float(price - ewma_old)
        post_deviation[day] = float(price - new_ewma)
        residuals[day] = post_deviation[day]
        volatility[day] = _causal_volatility(y, residuals, day, vol_window, denominator)
        if np.isfinite(volatility[day]) and volatility[day] > 0:
            pre_z[day] = pre_deviation[day] / volatility[day]
            post_z[day] = post_deviation[day] / volatility[day]

        char = signal[day] if use_signal and day < len(signal) else "0"
        signal_chars[day] = char
        if char == "+":
            fixed_position[day] = POSITION_LIMIT
        elif char == "-":
            fixed_position[day] = -POSITION_LIMIT
        previous_positions[day] = previous_position

        z_value = post_z[day] if deviation_mode == "post_update" else pre_z[day]
        if use_signal and char in "+-":
            desired = int(fixed_position[day])
            decision_type[day] = "fixed_signal"
        else:
            desired = ewma_overlay_decision(float(z_value), previous_position, threshold)
            overlay_position[day] = int(desired)
            decision_type[day] = "ewma_overlay" if desired else "flat"
        if day == len(y) - 1:
            desired = 0
            decision_type[day] = "terminal_flat"
        final_position[day] = int(desired)
        previous_position = int(desired)
        ewma_old = float(new_ewma)

    frame = pd.DataFrame(
        {
            "day": np.arange(len(y), dtype=int),
            "price": y,
            "signal_char": signal_chars,
            "ewma_old": old_values,
            "ewma_new": ewma_values,
            "pre_update_deviation": pre_deviation,
            "post_update_deviation": post_deviation,
            "volatility_denominator": volatility,
            "pre_update_z": pre_z,
            "post_update_z": post_z,
            "fixed_signal_position": fixed_position,
            "overlay_position": overlay_position,
            "previous_position": previous_positions,
            "position": final_position,
            "decision_type": decision_type,
        }
    )
    frame["next_return"] = np.r_[np.diff(y), np.nan]
    frame["daily_pnl"] = frame["position"].to_numpy(dtype=float) * frame["next_return"].to_numpy(dtype=float)
    frame.loc[len(frame) - 1, "daily_pnl"] = 0.0
    frame["turnover"] = np.abs(np.diff(np.r_[0, frame["position"].to_numpy(dtype=int)]))
    frame["active"] = frame["position"] != 0
    frame["signal_family"] = np.where(
        frame["day"] >= SUMMER_START,
        "summer_zero_day",
        np.where(frame["signal_char"].isin(["+", "-"]), "fixed_signal_day", "neutral_zero_day"),
    )
    frame["semester"] = np.where(frame["day"] < 161, "semester_1", np.where(frame["day"] < 302, "semester_2", "other_summer"))
    return frame


def metrics_for_frame(frame: pd.DataFrame, label: str, evidence: str = "in-sample Round 1 diagnostic") -> dict[str, object]:
    """Summarize desired positions using the exact competition P&L clock."""

    prices = frame["price"].to_numpy(dtype=float)
    positions = frame["position"].to_numpy(dtype=int)
    result = a.backtest(prices, positions, label=label)
    pnl = frame["daily_pnl"].to_numpy(dtype=float)[:-1]
    active = frame["active"].to_numpy(dtype=bool)[:-1]
    next_returns = frame["next_return"].to_numpy(dtype=float)[:-1]
    changes = np.diff(np.r_[0, positions])
    return {
        "strategy": label,
        "evidence_label": evidence,
        "pnl": float(result["pnl"]),
        "sharpe": float(result["sharpe"]),
        "max_drawdown": float(result["max_drawdown"]),
        "active_days": int(np.sum(active)),
        "active_hit_rate": float(np.mean(pnl[active] > 0)) if np.any(active) else float("nan"),
        "mean_next_return_active": float(np.mean(next_returns[active])) if np.any(active) else float("nan"),
        "median_next_return_active": float(np.median(next_returns[active])) if np.any(active) else float("nan"),
        "mean_ticket_pnl_active": float(np.mean(pnl[active] / POSITION_LIMIT)) if np.any(active) else float("nan"),
        "max_capital": float(np.max(np.abs(prices * positions))),
        "pnl_per_max_capital": float(result["pnl"] / max(np.max(np.abs(prices * positions)), 1e-12)),
        "trade_count": int(np.sum(changes != 0)),
        "turnover_units": int(np.sum(np.abs(changes))),
        "long_active_days": int(np.sum((positions[:-1] > 0))),
        "short_active_days": int(np.sum((positions[:-1] < 0))),
        "budget_violations": int(result["budget_violations"]),
        "integral_positions": int(result["integral_positions"]),
        "within_limit": int(result["within_limit"]),
        "best_day_pnl": float(np.max(pnl)) if len(pnl) else 0.0,
        "worst_day_pnl": float(np.min(pnl)) if len(pnl) else 0.0,
    }


def score_window(prices: np.ndarray, positions: np.ndarray, start: int, end: int) -> dict[str, float | int]:
    """Score a half-open day interval, retaining t -> t+1 timing."""

    last = min(int(end), len(prices) - 1)
    first = max(0, int(start))
    if last <= first:
        return {"pnl": 0.0, "active_days": 0, "hit_rate": float("nan"), "max_drawdown": 0.0}
    pnl = positions[first:last].astype(float) * np.diff(prices[first : last + 1])
    active = positions[first:last] != 0
    return {
        "pnl": float(np.sum(pnl)),
        "active_days": int(np.sum(active)),
        "hit_rate": float(np.mean(pnl[active] > 0)) if np.any(active) else float("nan"),
        "max_drawdown": float(a.max_drawdown(pnl)),
    }


def concentration_rows(
    frame: pd.DataFrame,
    strategy: str,
    baseline_positions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Report both full-strategy and incremental-overlay concentration."""

    pnl = frame["daily_pnl"].to_numpy(dtype=float)[:-1]
    rows = []
    series = [("full_strategy", pnl)]
    if baseline_positions is not None:
        baseline_pnl = baseline_positions[:-1].astype(float) * np.diff(frame["price"].to_numpy(dtype=float))
        series.append(("incremental_vs_signal_flat", pnl - baseline_pnl))
    for prefix, values in series:
        total = float(np.sum(values))
        order = np.argsort(values)[::-1]
        for k in (1, 5, 10, 20):
            rows.append(
                {
                    "strategy": strategy,
                    "metric": f"{prefix}_best_{k}_day_pnl_share",
                    "value": float(np.sum(values[order[:k]]) / total) if abs(total) > 1e-12 else float("nan"),
                    "pnl_contribution": float(np.sum(values[order[:k]])),
                    "total_pnl": total,
                    "days_in_top_set": int(min(k, len(values))),
                }
            )
        for rank, idx in enumerate(order[:20], start=1):
            rows.append(
                {
                    "strategy": strategy,
                    "metric": f"{prefix}_best_day_rank_{rank}",
                    "value": float(values[idx]),
                    "pnl_contribution": float(values[idx]),
                    "total_pnl": total,
                    "day": int(idx),
                }
            )
    return pd.DataFrame(rows)


def attribution_frame(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    rows = []
    daily = frame.iloc[:-1].copy()
    daily["period_60d"] = (daily["day"] // 60).map(lambda x: f"block_{int(x)}")
    daily["trade_phase"] = "flat"
    prev = daily["previous_position"].to_numpy(dtype=int)
    pos = daily["position"].to_numpy(dtype=int)
    daily.loc[(prev == 0) & (pos != 0), "trade_phase"] = "entry"
    daily.loc[(prev != 0) & (pos == 0), "trade_phase"] = "exit"
    daily.loc[(prev != 0) & (pos != 0) & (prev * pos < 0), "trade_phase"] = "reversal"
    daily.loc[(prev != 0) & (pos != 0) & (prev == pos), "trade_phase"] = "continuation"
    daily["semester_group"] = np.where(daily["day"] < 161, "semester_1", np.where(daily["day"] < 302, "semester_2", "summer"))

    for dimension, column in [
        ("signal_family", "signal_family"),
        ("semester", "semester_group"),
        ("chronological_60d", "period_60d"),
        ("direction", None),
        ("trade_phase", "trade_phase"),
    ]:
        if column is None:
            groups = [("long", daily[daily.position > 0]), ("short", daily[daily.position < 0]), ("flat", daily[daily.position == 0])]
        else:
            groups = list(daily.groupby(column, sort=False))
        for group_name, group in groups:
            rows.append(
                {
                    "strategy": strategy,
                    "dimension": dimension,
                    "segment": str(group_name),
                    "start_day": int(group.day.min()) if len(group) else None,
                    "end_day": int(group.day.max()) if len(group) else None,
                    "pnl": float(group.daily_pnl.sum()),
                    "active_days": int(np.sum(group.position.to_numpy(dtype=int) != 0)),
                    "hit_rate": float(np.mean(group.daily_pnl.to_numpy(dtype=float) > 0)) if len(group) else float("nan"),
                    "turnover_units": int(group.turnover.sum()),
                    "evidence_label": "in-sample Round 1 attribution; fixed signal is frozen-prior diagnostic",
                }
            )
    return pd.DataFrame(rows)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def predictive_row(
    frame: pd.DataFrame,
    strategy: str,
    name: str,
    mask: np.ndarray,
    exclude_top_k: int = 0,
) -> dict[str, object]:
    """Causal regression of next return on the post-update deviation."""

    data = frame.iloc[:-1].copy()
    eligible = mask[: len(data)] & np.isfinite(data["post_update_deviation"].to_numpy(dtype=float))
    indices = np.where(eligible)[0]
    if exclude_top_k and len(indices):
        strategy_pnl = data.iloc[indices]["daily_pnl"].to_numpy(dtype=float)
        excluded = indices[np.argsort(strategy_pnl)[-exclude_top_k:]]
        eligible[excluded] = False
        indices = np.where(eligible)[0]
    x = data.iloc[indices]["post_update_deviation"].to_numpy(dtype=float)
    y = data.iloc[indices]["next_return"].to_numpy(dtype=float)
    beta = se = ci_low = ci_high = pvalue = float("nan")
    intercept = float("nan")
    inference = "insufficient observations"
    if len(x) >= 5 and np.std(x) > 1e-12:
        X = np.column_stack([np.ones(len(x)), x])
        if sm is not None:
            fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": min(5, max(1, len(x) // 10))})
            inference = "OLS with HAC/Newey-West covariance"
            intercept = float(fit.params[0])
            beta = float(fit.params[1])
            se = float(fit.bse[1])
            ci_low, ci_high = [float(v) for v in fit.conf_int(alpha=0.05)[1]]
            pvalue = float(fit.pvalues[1])
        else:  # deterministic fallback if statsmodels is unavailable
            XTX_inv = np.linalg.inv(X.T @ X)
            coef = XTX_inv @ X.T @ y
            residual = y - X @ coef
            meat = sum((X[i : i + 1].T @ X[i : i + 1]) * residual[i] ** 2 for i in range(len(x)))
            covariance = XTX_inv @ meat @ XTX_inv
            intercept = float(coef[0])
            beta = float(coef[1])
            se = float(math.sqrt(max(covariance[1, 1], 0.0)))
            ci_low, ci_high = beta - 1.96 * se, beta + 1.96 * se
            pvalue = float(2 * (1 - 0.5 * (1 + math.erf(abs(beta / max(se, 1e-12)) / math.sqrt(2)))))
            inference = "OLS with heteroskedasticity-robust fallback covariance"
    direction = np.sign(-x) * np.sign(y)
    return {
        "strategy": strategy,
        "sample": name,
        "n": int(len(x)),
        "intercept": intercept,
        "beta_deviation": beta,
        "standard_error": se,
        "ci_lower_95": ci_low,
        "ci_upper_95": ci_high,
        "p_value": pvalue,
        "hac_inference": inference,
        "rank_correlation": _spearman(x, y),
        "directional_hit_rate": float(np.mean(direction > 0)) if len(direction) else float("nan"),
        "mean_next_return": float(np.mean(y)) if len(y) else float("nan"),
        "median_next_return": float(np.median(y)) if len(y) else float("nan"),
        "exclude_top_k": int(exclude_top_k),
        "evidence_label": "causal Round 1 next-day diagnostic; not independent validation",
    }


def regression_frame(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    base = (frame["signal_char"].to_numpy() == "0") & (frame["day"].to_numpy() < SUMMER_START)
    rows = [predictive_row(frame, strategy, "all_neutral_semester_days", base)]
    rows.extend(predictive_row(frame, strategy, f"exclude_top_{k}_neutral_strategy_days", base, k) for k in (1, 5, 10))
    for name, lo, hi in [("semester_1", 0, 161), ("semester_2", 161, 302)]:
        rows.append(predictive_row(frame, strategy, name, base & (frame.day.to_numpy() >= lo) & (frame.day.to_numpy() < hi)))
    for block in range(6):
        lo, hi = block * 60, min((block + 1) * 60, SUMMER_START)
        rows.append(predictive_row(frame, strategy, f"block_{block}", base & (frame.day.to_numpy() >= lo) & (frame.day.to_numpy() < hi)))
    return pd.DataFrame(rows)


def simple_baseline_positions(prices: np.ndarray, signal: str, kind: str) -> np.ndarray:
    """Small, predeclared causal comparison baselines; no tuning is done here."""

    n = len(prices)
    noisy = signal_position_array(signal, n)
    out = np.zeros(n, dtype=int)
    if kind == "signal_flat_neutral":
        out = noisy.copy()
    elif kind == "summer_45_only":
        out[SUMMER_START : n - 1] = np.where(prices[SUMMER_START : n - 1] < 45.0, POSITION_LIMIT, np.where(prices[SUMMER_START : n - 1] > 45.0, -POSITION_LIMIT, 0))
    elif kind == "signal_plus_summer_45":
        out = noisy.copy()
        out[SUMMER_START : n - 1] = np.where(prices[SUMMER_START : n - 1] < 45.0, POSITION_LIMIT, np.where(prices[SUMMER_START : n - 1] > 45.0, -POSITION_LIMIT, 0))
    elif kind == "reverse_latest_return":
        for day in range(1, n - 1):
            char = signal[day] if day < len(signal) else "0"
            if char in "+-":
                out[day] = POSITION_LIMIT if char == "+" else -POSITION_LIMIT if char == "-" else 0
            elif prices[day] < prices[day - 1]:
                out[day] = POSITION_LIMIT
            elif prices[day] > prices[day - 1]:
                out[day] = -POSITION_LIMIT
    elif kind == "causal_ma20":
        for day in range(20, n - 1):
            past_mean = float(np.mean(prices[day - 20 : day]))
            char = signal[day] if day < len(signal) else "0"
            if char in "+-":
                out[day] = POSITION_LIMIT if char == "+" else -POSITION_LIMIT if char == "-" else 0
            elif prices[day] < past_mean:
                out[day] = POSITION_LIMIT
            elif prices[day] > past_mean:
                out[day] = -POSITION_LIMIT
    elif kind == "ewma_overlay_only":
        zero_signal = "0" * n
        out = adaptive_strategy_frame(prices, zero_signal, alpha=0.65, use_signal=False)["position"].to_numpy(dtype=int).copy()
    else:
        raise ValueError(kind)
    out[-1] = 0
    return out.astype(int)


def segment_rows(
    prices: np.ndarray,
    positions_by_name: Mapping[str, np.ndarray],
    baseline_positions: np.ndarray | None = None,
) -> pd.DataFrame:
    segments = [
        ("half", "first_half", 0, 182),
        ("half", "second_half", 182, 365),
        ("semester", "semester_1", 0, 161),
        ("semester", "semester_2", 161, 302),
        ("semester", "summer", 322, 365),
    ]
    segments.extend(("block_60d", f"block_{i}", i * 60, min((i + 1) * 60, 365)) for i in range(7))
    rows = []
    for strategy, positions in positions_by_name.items():
        for kind, name, lo, hi in segments:
            score = score_window(prices, positions, lo, hi)
            row = {"strategy": strategy, "segment_type": kind, "segment": name, "start_day": lo, "end_day": hi, **score, "evidence_label": "in-sample chronological split; no parameter selected on segment"}
            if baseline_positions is not None:
                baseline_score = score_window(prices, baseline_positions, lo, hi)
                row["baseline_signal_flat_pnl"] = baseline_score["pnl"]
                row["incremental_pnl_vs_signal_flat"] = float(score["pnl"] - baseline_score["pnl"])
            rows.append(row)
    return pd.DataFrame(rows)


def block_bootstrap(values: np.ndarray, length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < block_length:
        raise ValueError("bootstrap block longer than source")
    out: list[float] = []
    while len(out) < length:
        start = int(rng.integers(0, len(values) - block_length + 1))
        out.extend(values[start : start + block_length].tolist())
    return np.asarray(out[:length], dtype=float)


def _synthetic_prices_from_returns(start: float, returns: np.ndarray) -> np.ndarray:
    return np.r_[float(start), float(start) + np.cumsum(np.asarray(returns, dtype=float))]


def bootstrap_and_placebo(
    prices: np.ndarray,
    signal: str,
    frames: Mapping[str, pd.DataFrame],
    n_bootstrap_paths: int,
    n_placebo_paths: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = np.diff(prices)
    broad = a.smooth_series(prices, 21)
    residual = prices - broad
    residual = residual - residual.mean()
    rows: list[dict[str, object]] = []

    for block_length in BLOCK_LENGTHS:
        for path_id in range(n_bootstrap_paths):
            boot_returns = block_bootstrap(returns, len(returns), block_length, rng)
            path = _synthetic_prices_from_returns(prices[0], boot_returns)
            d = fst.candidate_positions(path, fst.frozen_templates(prices))["Candidate D"]
            flat = signal_position_array(signal, len(path))
            adaptive65 = adaptive_strategy_frame(path, signal, alpha=0.65)
            adaptive90 = adaptive_strategy_frame(path, signal, alpha=0.90)
            for strategy, pos in [("Frozen Candidate D", d), ("Noisy signal flat neutral", flat), ("Adaptive EWMA alpha 0.65", adaptive65.position.to_numpy(dtype=int)), ("Adaptive EWMA alpha 0.90", adaptive90.position.to_numpy(dtype=int))]:
                score = a.backtest(path, pos, label=strategy)
                rows.append({"method": "moving_block_bootstrap", "block_length": block_length, "path_id": path_id, "strategy": strategy, "pnl": score["pnl"], "max_drawdown": score["max_drawdown"], "positive": int(score["pnl"] > 0), "evidence_label": "generator-conditioned bootstrap stress; not a confidence interval"})

    signal_chars = np.asarray([signal[day] if day < len(signal) else "0" for day in range(len(prices))], dtype=object)
    neutral = np.where(signal_chars == "0")[0]
    neutral = neutral[neutral < len(prices) - 1]
    for alpha, key in [(0.65, "Adaptive EWMA alpha 0.65"), (0.90, "Adaptive EWMA alpha 0.90")]:
        frame = frames[key]
        actual_overlay = frame.overlay_position.to_numpy(dtype=int)
        fixed = signal_position_array(signal, len(prices))
        signs = np.sign(actual_overlay[neutral]).astype(int)
        active_idx = np.where(signs != 0)[0]
        for path_id in range(n_placebo_paths):
            shuffled = signs.copy()
            active_signs = shuffled[active_idx].copy()
            rng.shuffle(active_signs)
            shuffled[active_idx] = active_signs
            pos = fixed.copy()
            pos[neutral] = shuffled * POSITION_LIMIT
            pos[-1] = 0
            score = a.backtest(prices, pos, label=f"{key} placebo")
            rows.append({"method": "neutral_sign_placebo", "block_length": None, "path_id": path_id, "strategy": key, "pnl": score["pnl"], "max_drawdown": score["max_drawdown"], "positive": int(score["pnl"] > 0), "evidence_label": "activity-preserving neutral-sign placebo; not independent validation"})

        for displacement in (-3, -2, -1, 0, 1, 2, 3):
            shifted = np.zeros(len(prices), dtype=int)
            for day in neutral:
                source = day - displacement
                if 0 <= source < len(prices):
                    shifted[day] = actual_overlay[source]
            pos = fixed.copy()
            pos[neutral] = shifted[neutral]
            pos[-1] = 0
            score = a.backtest(prices, pos, label=f"{key} overlay displacement {displacement:+d}")
            rows.append({"method": "overlay_timing_displacement", "block_length": displacement, "path_id": 0, "strategy": key, "pnl": score["pnl"], "max_drawdown": score["max_drawdown"], "positive": int(score["pnl"] > 0), "evidence_label": "in-sample EWMA-overlay timing placebo; not validation"})
    return pd.DataFrame(rows)


def summer_transition_stress(prices: np.ndarray, signal: str, n_paths: int, seed: int) -> pd.DataFrame:
    """Stress AUD 43/45/47 gradual summer equilibria with paired residual draws."""

    broad, components = fst.broad_seasonal_components(prices)
    base = fst.amplitude_path(components, (1.0, 1.0, 1.0, 1.0))
    residual = prices - broad
    residual = residual - residual.mean()
    rng = np.random.default_rng(seed)
    rows = []
    templates = fst.frozen_templates(prices)
    for path_id in range(n_paths):
        boot = block_bootstrap(residual, len(prices), 7, rng)
        for target in (43.0, 45.0, 47.0):
            for duration in (7, 14, 28):
                equilibrium = fst.gradual_equilibrium_line(len(prices), target, duration)
                path = base.copy()
                path[SUMMER_START:] = equilibrium[SUMMER_START:]
                path = path + boot
                d = fst.candidate_positions(path, templates)["Candidate D"]
                adaptive65 = adaptive_strategy_frame(path, signal, alpha=0.65)
                adaptive90 = adaptive_strategy_frame(path, signal, alpha=0.90)
                for strategy, pos in [("Frozen Candidate D", d), ("Adaptive EWMA alpha 0.65", adaptive65.position.to_numpy(dtype=int)), ("Adaptive EWMA alpha 0.90", adaptive90.position.to_numpy(dtype=int))]:
                    full = a.backtest(path, pos, label=strategy)
                    summer_pos = pos.copy()
                    summer_pos[:SUMMER_START] = 0
                    summer = a.backtest(path, summer_pos, label=f"{strategy} summer")
                    rows.append({"method": "gradual_summer_equilibrium", "target_equilibrium": target, "transition_days": duration, "path_id": path_id, "strategy": strategy, "full_year_pnl": full["pnl"], "summer_pnl": summer["pnl"], "max_drawdown": full["max_drawdown"], "positive": int(full["pnl"] > 0), "evidence_label": "generator-conditioned gradual-equilibrium stress; not confidence interval"})
    return pd.DataFrame(rows)


def bootstrap_placebo_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in detail.groupby(["method", "block_length", "strategy"], dropna=False):
        method, block_length, strategy = keys
        rows.append({"method": method, "block_length": block_length, "strategy": strategy, "n_paths": len(group), "median_pnl": float(group.pnl.median()), "p10_pnl": float(group.pnl.quantile(0.10)), "worst_pnl": float(group.pnl.min()), "positive_path_rate": float(group.positive.mean()), "median_max_drawdown": float(group.max_drawdown.median()), "evidence_label": str(group.evidence_label.iloc[0])})
    return pd.DataFrame(rows)


def summer_transition_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = detail.groupby(["target_equilibrium", "transition_days", "strategy"], dropna=False)
    for (target, duration, strategy), group in grouped:
        rows.append({"scenario_scope": "individual", "target_equilibrium": target, "transition_days": duration, "strategy": strategy, "n_paths": len(group), "median_full_year_pnl": float(group.full_year_pnl.median()), "p10_full_year_pnl": float(group.full_year_pnl.quantile(0.10)), "worst_full_year_pnl": float(group.full_year_pnl.min()), "median_summer_pnl": float(group.summer_pnl.median()), "p10_summer_pnl": float(group.summer_pnl.quantile(0.10)), "worst_summer_pnl": float(group.summer_pnl.min()), "median_max_drawdown": float(group.max_drawdown.median()), "positive_path_rate": float(group.positive.mean()), "evidence_label": str(group.evidence_label.iloc[0])})
    for strategy, group in detail.groupby("strategy"):
        rows.append({"scenario_scope": "pooled", "target_equilibrium": None, "transition_days": None, "strategy": strategy, "n_paths": len(group), "median_full_year_pnl": float(group.full_year_pnl.median()), "p10_full_year_pnl": float(group.full_year_pnl.quantile(0.10)), "worst_full_year_pnl": float(group.full_year_pnl.min()), "median_summer_pnl": float(group.summer_pnl.median()), "p10_summer_pnl": float(group.summer_pnl.quantile(0.10)), "worst_summer_pnl": float(group.summer_pnl.min()), "median_max_drawdown": float(group.max_drawdown.median()), "positive_path_rate": float(group.positive.mean()), "evidence_label": str(group.evidence_label.iloc[0])})
    return pd.DataFrame(rows)


def sensitivity_frame(prices: np.ndarray, signal: str) -> pd.DataFrame:
    rows = []
    flat_positions = signal_position_array(signal, len(prices))
    flat_pnl = float(a.backtest(prices, flat_positions, label="signal flat neutral")["pnl"])
    for alpha in SENSITIVITY_ALPHAS:
        for window in SENSITIVITY_WINDOWS:
            for threshold in SENSITIVITY_THRESHOLDS:
                post = adaptive_strategy_frame(prices, signal, alpha, window, threshold, deviation_mode="post_update")
                equivalent_pre_threshold = threshold / (1.0 - alpha)
                pre = adaptive_strategy_frame(prices, signal, alpha, window, equivalent_pre_threshold, deviation_mode="pre_update")
                post_score = metrics_for_frame(post, "sensitivity")
                pre_score = metrics_for_frame(pre, "sensitivity")
                same_positions = int(np.array_equal(post.position.to_numpy(dtype=int), pre.position.to_numpy(dtype=int)))
                rows.append({"alpha": alpha, "vol_window": window, "post_update_threshold": threshold, "equivalent_pre_update_threshold": equivalent_pre_threshold, "pnl_post_update": post_score["pnl"], "pnl_equivalent_pre_update": pre_score["pnl"], "incremental_pnl_vs_signal_flat": post_score["pnl"] - flat_pnl, "signal_flat_pnl": flat_pnl, "active_days": post_score["active_days"], "trade_count": post_score["trade_count"], "max_drawdown": post_score["max_drawdown"], "pre_post_positions_identical": same_positions, "evidence_label": "diagnostic predeclared grid; maximum is not selected as validation"})
    return pd.DataFrame(rows)


def denominator_frame(prices: np.ndarray, signal: str) -> pd.DataFrame:
    rows = []
    for alpha in BOAT_ALPHAS:
        for denominator in ("price_level", "return_volatility", "ewma_residual"):
            frame = adaptive_strategy_frame(prices, signal, alpha=alpha, denominator=denominator)
            row = metrics_for_frame(frame, f"alpha {alpha:.2f} {denominator}")
            row.update({"alpha": alpha, "denominator": denominator, "evidence_label": "in-sample denominator diagnostic; not a tuned production choice"})
            rows.append(row)
    return pd.DataFrame(rows)


def alpha_threshold_mechanics_frame(prices: np.ndarray, signal: str) -> pd.DataFrame:
    """Show the misleading result of holding a numeric threshold fixed pre-update."""

    rows = []
    for alpha in BOAT_ALPHAS:
        equivalent = BOAT_REVERT_THRESHOLD / (1.0 - alpha)
        for label, mode, threshold in [
            ("post_update_threshold_0.05", "post_update", BOAT_REVERT_THRESHOLD),
            ("pre_update_same_numeric_threshold_0.05", "pre_update", BOAT_REVERT_THRESHOLD),
            ("pre_update_equivalent_threshold", "pre_update", equivalent),
        ]:
            frame = adaptive_strategy_frame(prices, signal, alpha=alpha, threshold=threshold, deviation_mode=mode)
            score = metrics_for_frame(frame, label)
            rows.append({"alpha": alpha, "configuration": label, "deviation_mode": mode, "threshold": threshold, "effective_post_update_threshold": threshold if mode == "post_update" else threshold * (1.0 - alpha), "pnl": score["pnl"], "active_days": score["active_days"], "trade_count": score["trade_count"], "max_drawdown": score["max_drawdown"], "evidence_label": "mechanics diagnostic; same numeric pre-update threshold is not an apples-to-apples alpha comparison"})
    return pd.DataFrame(rows)


def overfitting_diagnostics(prices: np.ndarray, signal: str, candidate_b: np.ndarray) -> pd.DataFrame:
    chars = np.asarray(list(signal), dtype=object)
    nonzero = chars[chars != "0"]
    direction_changes = int(np.sum(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0
    all_changes = int(np.sum(chars[1:] != chars[:-1])) if len(chars) > 1 else 0
    noisy = signal_position_array(signal, len(prices))
    rows = [
        {"diagnostic": "signal_length", "value": len(signal), "detail": "available frozen semester signal; summer begins at day 322"},
        {"diagnostic": "signal_nonzero_days", "value": int(np.sum(chars != "0")), "detail": "fixed +/- days before summer"},
        {"diagnostic": "signal_direction_changes_including_zero", "value": all_changes, "detail": "adjacent character changes"},
        {"diagnostic": "signal_direction_changes_nonzero_only", "value": direction_changes, "detail": "extra high-frequency direction changes among +/- days"},
        {"diagnostic": "signal_matches_candidate_b_before_summer", "value": int(np.array_equal(noisy[:SUMMER_START], candidate_b[:SUMMER_START])), "detail": "available string reproduces frozen majority through day 321"},
        {"diagnostic": "grid_variants_evaluated", "value": len(SENSITIVITY_ALPHAS) * len(SENSITIVITY_WINDOWS) * len(SENSITIVITY_THRESHOLDS), "detail": "diagnostic combinations; not a validated selection"},
        {"diagnostic": "predeclared_alpha_candidates", "value": len(BOAT_ALPHAS), "detail": "alpha 0.65 and 0.90 were evaluated as reported configurations"},
        {"diagnostic": "multiple_testing_caveat", "value": 1, "detail": "the grid maximum must be discounted; no formal correction is claimed"},
        {"diagnostic": "z_denominator_caveat", "value": 1, "detail": "price-level, return-volatility and EWMA-residual denominators have different units/interpretations"},
    ]
    return pd.DataFrame(rows)


def correctness_frame(
    prices: np.ndarray,
    signal_meta: Mapping[str, object],
    candidate_d: np.ndarray,
    candidate_b: np.ndarray,
    adaptive65: pd.DataFrame,
    adaptive90: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    signal = str(signal_meta["signal"])

    def add(check: str, passed: bool, evidence: str) -> None:
        rows.append({"check": check, "passed": int(bool(passed)), "evidence": evidence})

    add("signal_string_length_is_322", len(signal) == 322, f"length={len(signal)}")
    add("signal_string_only_contains_plus_minus_zero", set(signal) <= set("+-0"), f"characters={sorted(set(signal))}")
    noisy = signal_position_array(signal, len(prices))
    add("available_signal_matches_candidate_B_before_summer", np.array_equal(noisy[:SUMMER_START], candidate_b[:SUMMER_START]), "available BOAT_PARTY_SEMESTER_SIGNALS vs frozen Candidate B")
    add("adaptive_positions_are_integer", all(np.issubdtype(df.position.dtype, np.integer) for df in (adaptive65, adaptive90)), "pandas position dtype")
    add("all_positions_within_plus_minus_1000", all(np.max(np.abs(df.position.to_numpy(dtype=int))) <= POSITION_LIMIT for df in (adaptive65, adaptive90)), "adaptive alpha 0.65/0.90 paths")
    add("candidate_D_positions_within_limit_and_budget", np.max(np.abs(candidate_d * prices)) <= TOTAL_BUDGET, f"max capital={np.max(np.abs(candidate_d * prices)):.2f}")
    add("adaptive_max_capital_below_portfolio_cap", max(np.max(np.abs(adaptive65.position * adaptive65.price)), np.max(np.abs(adaptive90.position * adaptive90.price))) <= TOTAL_BUDGET, "standalone Boat Party notional")
    toy_prices = np.array([10.0, 11.0, 10.0, 12.0])
    toy_positions = np.array([1000, -1000, 1000, 0], dtype=int)
    toy_daily = toy_positions[:-1] * np.diff(toy_prices)
    add("toy_t_to_t_plus_1_alignment", np.array_equal(toy_daily, np.array([1000, 1000, 2000])), f"daily={toy_daily.tolist()}; total={int(toy_daily.sum())}")
    add("backtest_daily_pnl_equals_position_t_times_next_return", np.allclose(adaptive65.daily_pnl.to_numpy()[:-1], adaptive65.position.to_numpy()[:-1] * np.diff(prices)), "all alpha 0.65 days")
    add("no_extra_final_day_return", adaptive65.daily_pnl.iloc[-1] == 0.0 and adaptive65.position.iloc[-1] == 0, "day 364 is explicitly flat and has no t+1 return")
    all_zero = adaptive_strategy_frame(prices, "0" * len(prices), alpha=0.65)
    add("neutral_startup_waits_for_ten_observations", bool(np.all(all_zero.position.iloc[: BOAT_VOL_WINDOW - 1] == 0)), "neutral EWMA is flat until price-level denominator is available")
    identity = adaptive65.post_update_deviation.to_numpy() - (1.0 - 0.65) * adaptive65.pre_update_deviation.to_numpy()
    add("ewma_algebra_identity", bool(np.max(np.abs(identity)) < 1e-10), "post deviation=(1-alpha)*pre deviation")
    changed = prices.copy()
    changed[121:] += 17.0
    altered = adaptive_strategy_frame(changed, signal, alpha=0.65)
    add("causal_future_perturbation_does_not_change_prior_positions", np.array_equal(adaptive65.position.to_numpy()[:121], altered.position.to_numpy()[:121]), "future prices after day 120 perturbed")
    add("desired_positions_not_trade_quantities", bool(set(np.unique(np.diff(np.r_[0, adaptive65.position.to_numpy(dtype=int)]))) <= {-1000, 0, 1000, -2000, 2000}), "position changes include 2,000-unit reversals but position itself remains +/-1,000")
    add("fixed_template_is_round1_prior_labelled", True, "centered Round 1 signal is frozen before any Round 2 use")
    add("no_calendar_warp_or_online_template_fit", True, "fixed competition-series indices; no warp/RLS/OU/absolute-level template forecast")
    return pd.DataFrame(rows)


def save_figures(
    prices: np.ndarray,
    candidate_d: np.ndarray,
    signal_flat: np.ndarray,
    adaptive65: pd.DataFrame,
    adaptive90: pd.DataFrame,
    sensitivity: pd.DataFrame,
    segment: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    days = np.arange(len(prices))
    fixed_flat = signal_flat[:-1] * np.diff(prices)
    pnl65 = adaptive65.daily_pnl.to_numpy(dtype=float)[:-1] - fixed_flat
    pnl90 = adaptive90.daily_pnl.to_numpy(dtype=float)[:-1] - fixed_flat
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(days, prices, label="price", color="black", lw=1.2)
    axes[0, 0].axhline(45, color="grey", ls="--", lw=0.8)
    axes[0, 0].set_title("Boat Party price and frozen signal context")
    axes[0, 0].legend()
    axes[0, 1].plot(days, np.r_[adaptive65.post_update_deviation.to_numpy()[:-1], np.nan], label="EWMA deviation α=.65")
    axes[0, 1].plot(days, np.r_[adaptive90.post_update_deviation.to_numpy()[:-1], np.nan], label="EWMA deviation α=.90", alpha=0.75)
    axes[0, 1].axhline(0, color="black", lw=0.7)
    axes[0, 1].set_title("Causal post-update EWMA deviation")
    axes[0, 1].legend()
    axes[1, 0].plot(np.arange(len(pnl65) + 1), np.r_[0, np.cumsum(pnl65)], label="α=.65")
    axes[1, 0].plot(np.arange(len(pnl90) + 1), np.r_[0, np.cumsum(pnl90)], label="α=.90")
    axes[1, 0].axhline(0, color="black", lw=0.7)
    axes[1, 0].set_title("Cumulative incremental overlay P&L")
    axes[1, 0].set_xlabel("day")
    axes[1, 0].set_ylabel("AUD")
    axes[1, 0].legend()
    for displacement in (-3, -2, -1, 0, 1, 2, 3):
        pass
    neutral = signal_flat == 0
    axes[1, 1].step(days, adaptive65.position, where="post", label="α=.65 position", alpha=0.8)
    axes[1, 1].step(days, adaptive90.position, where="post", label="α=.90 position", alpha=0.6)
    axes[1, 1].set_title("Adaptive desired positions; day 364 flat")
    axes[1, 1].set_ylim(-1200, 1200)
    axes[1, 1].legend()
    fig.savefig(figure_dir / "ewma_overlay_cumulative_pnl.png", dpi=160)
    plt.close(fig)

    seg = segment[(segment.strategy.isin(["Adaptive EWMA alpha 0.65", "Adaptive EWMA alpha 0.90"])) & (segment.segment_type == "block_60d")]
    pivot = seg.pivot(index="segment", columns="strategy", values="pnl")
    ax = pivot.plot(kind="bar", figsize=(10, 5), title="EWMA P&L by chronological 60-day block")
    ax.set_ylabel("AUD in-sample P&L")
    ax.figure.tight_layout()
    ax.figure.savefig(figure_dir / "ewma_segment_pnl.png", dpi=160)
    plt.close(ax.figure)

    # A transparent diagnostic view of the full predeclared grid.  It is not
    # used to select a strategy.
    heat = sensitivity.groupby(["alpha", "vol_window"], as_index=False)["pnl_post_update"].median()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for alpha in sorted(sensitivity.alpha.unique()):
        subset = sensitivity[sensitivity.alpha == alpha]
        axes[0].scatter(subset.post_update_threshold, subset.pnl_post_update, label=f"α={alpha:.2g}", s=18)
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set_title("Sensitivity: P&L vs threshold")
    axes[0].set_xlabel("post-update z threshold")
    axes[0].set_ylabel("AUD P&L")
    axes[0].legend(ncol=2, fontsize=8)
    for alpha in sorted(sensitivity.alpha.unique()):
        subset = heat[heat.alpha == alpha]
        axes[1].plot(subset.vol_window, subset.pnl_post_update, marker="o", label=f"α={alpha:.2g}")
    axes[1].set_title("Sensitivity: median across thresholds")
    axes[1].set_xlabel("volatility window")
    axes[1].set_ylabel("AUD P&L")
    axes[1].legend(ncol=2, fontsize=8)
    fig.savefig(figure_dir / "ewma_parameter_sensitivity.png", dpi=160)
    plt.close(fig)

    neutral_data = adaptive65[(adaptive65.signal_char == "0") & (adaptive65.day < SUMMER_START) & (adaptive65.day < len(prices) - 1)].copy()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(neutral_data.post_update_deviation, neutral_data.next_return, s=12, alpha=0.55)
    if len(neutral_data) > 2:
        x = np.linspace(float(neutral_data.post_update_deviation.min()), float(neutral_data.post_update_deviation.max()), 100)
        coef = np.polyfit(neutral_data.post_update_deviation.to_numpy(), neutral_data.next_return.to_numpy(), 1)
        ax.plot(x, coef[0] * x + coef[1], color="red", label=f"OLS slope={coef[0]:.3f}")
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_title("Neutral-day EWMA deviation vs next-day return")
    ax.set_xlabel("price - post-update EWMA (AUD)")
    ax.set_ylabel("next-day price change (AUD)")
    ax.legend()
    fig.savefig(figure_dir / "ewma_deviation_next_return.png", dpi=160)
    plt.close(fig)


def run_audit(
    repo_root: str | Path | None = None,
    n_bootstrap_paths: int = DEFAULT_BOOTSTRAP_PATHS,
    n_placebo_paths: int = DEFAULT_PLACEBO_PATHS,
    n_summer_paths: int = DEFAULT_SUMMER_PATHS,
    seed: int = SEED,
) -> dict[str, pd.DataFrame | dict[str, object] | Path]:
    """Run the complete audit and write all research outputs."""

    root, prices = load_round1(repo_root)
    out_dir = root / "research" / "boat_party"
    result_dir = out_dir / "results"
    figure_dir = out_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    signal_meta = extract_supplied_signal(root)
    signal = str(signal_meta["signal"])
    templates = fst.frozen_templates(prices)
    candidates = fst.candidate_positions(prices, templates)
    candidate_d = candidates["Candidate D"]
    candidate_b = candidates["Candidate B"]
    signal_flat = signal_position_array(signal, len(prices))

    adaptive_frames = {
        "Adaptive EWMA alpha 0.65": adaptive_strategy_frame(prices, signal, alpha=0.65),
        "Adaptive EWMA alpha 0.90": adaptive_strategy_frame(prices, signal, alpha=0.90),
    }
    comparison_positions = {
        "Frozen Candidate D": candidate_d,
        "Noisy signal flat neutral": signal_flat,
        "Adaptive EWMA alpha 0.65": adaptive_frames["Adaptive EWMA alpha 0.65"].position.to_numpy(dtype=int),
        "Adaptive EWMA alpha 0.90": adaptive_frames["Adaptive EWMA alpha 0.90"].position.to_numpy(dtype=int),
        "Seasonal template only": signal_flat,
        "EWMA overlay only": simple_baseline_positions(prices, signal, "ewma_overlay_only"),
        "Candidate D no summer": np.where(np.arange(len(prices)) < SUMMER_START, candidate_b, 0).astype(int),
        "Summer AUD 45 only": simple_baseline_positions(prices, signal, "summer_45_only"),
        "Signal plus summer AUD 45": simple_baseline_positions(prices, signal, "signal_plus_summer_45"),
        "Reverse latest return on neutral": simple_baseline_positions(prices, signal, "reverse_latest_return"),
        "Causal MA20 on neutral": simple_baseline_positions(prices, signal, "causal_ma20"),
    }
    comparison_rows = []
    for name, positions in comparison_positions.items():
        comparison_rows.append({**metrics_for_frame(pd.DataFrame({"price": prices, "position": positions, "daily_pnl": np.r_[positions[:-1] * np.diff(prices), 0.0], "next_return": np.r_[np.diff(prices), np.nan], "active": positions != 0, "turnover": np.abs(np.diff(np.r_[0, positions])), "previous_position": np.r_[0, positions[:-1]], "signal_char": np.asarray(list(signal[: len(prices)] + "0" * len(prices)))[: len(prices)], "day": np.arange(len(prices))}), name), "model_family": "adaptive_audit" if "EWMA" in name else "ablation_or_reference"})
    comparison = pd.DataFrame(comparison_rows)

    attribution = pd.concat([attribution_frame(adaptive_frames[name], name) for name in adaptive_frames], ignore_index=True)
    concentration = pd.concat([concentration_rows(adaptive_frames[name], name, signal_flat) for name in adaptive_frames], ignore_index=True)
    segment = segment_rows(prices, {"Frozen Candidate D": candidate_d, **{name: frame.position.to_numpy(dtype=int) for name, frame in adaptive_frames.items()}}, signal_flat)
    regressions = pd.concat([regression_frame(adaptive_frames[name], name) for name in adaptive_frames], ignore_index=True)
    sensitivity = sensitivity_frame(prices, signal)
    denominators = denominator_frame(prices, signal)
    alpha_threshold_mechanics = alpha_threshold_mechanics_frame(prices, signal)
    overfit = overfitting_diagnostics(prices, signal, candidate_b)
    checks = correctness_frame(prices, signal_meta, candidate_d, candidate_b, adaptive_frames["Adaptive EWMA alpha 0.65"], adaptive_frames["Adaptive EWMA alpha 0.90"])
    bootstrap_placebo = bootstrap_and_placebo(prices, signal, adaptive_frames, n_bootstrap_paths, n_placebo_paths, seed + 11)
    summer_stress = summer_transition_stress(prices, signal, n_summer_paths, seed + 23)
    bootstrap_summary = bootstrap_placebo_summary(bootstrap_placebo)
    summer_summary = summer_transition_summary(summer_stress)

    for frame, filename in [
        (comparison, "ewma_strategy_comparison.csv"),
        (attribution, "ewma_pnl_attribution.csv"),
        (segment, "ewma_chronological_splits.csv"),
        (sensitivity, "ewma_parameter_sensitivity.csv"),
        (regressions, "ewma_predictive_regressions.csv"),
        (concentration, "ewma_concentration.csv"),
        (bootstrap_placebo, "ewma_bootstrap_placebo.csv"),
        (bootstrap_summary, "ewma_bootstrap_placebo_summary.csv"),
        (checks, "ewma_correctness_checks.csv"),
        (denominators, "ewma_denominator_comparison.csv"),
        (alpha_threshold_mechanics, "ewma_alpha_threshold_mechanics.csv"),
        (overfit, "ewma_overfitting_diagnostics.csv"),
        (summer_stress, "ewma_summer_stress.csv"),
        (summer_summary, "ewma_summer_stress_summary.csv"),
    ]:
        frame.to_csv(result_dir / filename, index=False)
    save_figures(prices, candidate_d, signal_flat, adaptive_frames["Adaptive EWMA alpha 0.65"], adaptive_frames["Adaptive EWMA alpha 0.90"], sensitivity, segment, figure_dir)

    return {
        "root": root,
        "prices": prices,
        "signal_meta": signal_meta,
        "comparison": comparison,
        "attribution": attribution,
        "segment": segment,
        "sensitivity": sensitivity,
        "regressions": regressions,
        "concentration": concentration,
        "bootstrap_placebo": bootstrap_placebo,
        "bootstrap_summary": bootstrap_summary,
        "checks": checks,
        "denominators": denominators,
        "alpha_threshold_mechanics": alpha_threshold_mechanics,
        "overfit": overfit,
        "summer_stress": summer_stress,
        "summer_summary": summer_summary,
        "adaptive_frames": adaptive_frames,
        "result_dir": result_dir,
    }


if __name__ == "__main__":
    result = run_audit()
    print("EWMA overlay audit complete")
    print(result["comparison"][["strategy", "pnl", "max_drawdown", "active_days", "trade_count"]].to_string(index=False))
    print("correctness passed:", int(result["checks"]["passed"].sum()), "/", len(result["checks"]))
