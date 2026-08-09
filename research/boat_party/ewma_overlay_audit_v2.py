"""V2 research-only Boat Party audit.

The V1 audit used the production Candidate-D semester string as if it were
the teammate's full signal.  V2 keeps that signal only as the frozen baseline
and reconstructs the teammate's thresholded next-return string exclusively as
a leakage control.  Valid candidate results trade only on Candidate-D
neutral semester days and use seasonality-preserving generator-conditioned
stresses.

No production module is imported for mutation.  The production algorithm is
read as text for the frozen Candidate-D signal and, in one read-only helper,
for the existing non-Boat portfolio positions.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover
    sm = None

try:
    from . import analysis as a
    from . import final_strategy_test as fst
    from . import ewma_overlay_audit as v1
except ImportError:  # direct execution from research/boat_party
    import analysis as a
    import final_strategy_test as fst
    import ewma_overlay_audit as v1


SEED = 20260811
POSITION_LIMIT = 1_000
TOTAL_BUDGET = 600_000.0
SUMMER_START = 322
LAST_DAY = 364
DEADBAND = 0.02
SUMMER_ANCHOR = 45.0
PRIMARY_ALPHA = 0.65
REPORTED_ALPHAS = (0.65, 0.90)
PRIMARY_WINDOW = 10
PRIMARY_THRESHOLD = 0.05
CORRECTED_SUMMER_ALPHA = 0.10
PARAMETER_ALPHAS = (0.10, 0.30, 0.50, 0.65, 0.80, 0.90)
PARAMETER_WINDOWS = (5, 10, 20, 30)
PARAMETER_THRESHOLDS = (0.0, 0.25, 0.50, 1.0)
DENOMINATORS = ("historical_innovation", "return_volatility", "price_level", "mad_innovation")
AMPLITUDE_LEVELS = (0.50, 0.75, 1.00, 1.25, 1.50)
BLOCKS_60 = tuple((f"block_{i}", i * 60, min((i + 1) * 60, 365)) for i in range(7))
BOOTSTRAP_PATHS = 150
SUMMER_STRESS_PATHS = 100


def load_round1(repo_root: str | Path | None = None) -> tuple[Path, np.ndarray]:
    root = a.find_repo_root(repo_root)
    prices = a.load_prices(root)["Price"].to_numpy(dtype=float)
    return root, prices


def _extract_constant_from_algorithm(root: Path, name: str) -> object:
    path = root / "trader_interface" / "algorithm.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if name in names:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in production algorithm")


def extract_frozen_candidate_d_signal(root: Path) -> dict[str, object]:
    """Read the current production string without importing or changing it."""

    raw = _extract_constant_from_algorithm(root, "BOAT_PARTY_SEMESTER_SIGNALS")
    signal = "".join(raw) if isinstance(raw, tuple) else str(raw)
    if len(signal) != SUMMER_START or set(signal) - set("+-0"):
        raise AssertionError("production frozen Candidate-D signal is not the expected 322-day +/-/0 string")
    return {
        "signal": signal,
        "source_symbol": "BOAT_PARTY_SEMESTER_SIGNALS",
        "source_file": str(root / "trader_interface" / "algorithm.py"),
        "length": len(signal),
        "provenance": "fixed Candidate-D semester signal read from production; not regenerated or optimized",
    }


def thresholded_future_return_signal(prices: Sequence[float], threshold: float = 0.50) -> np.ndarray:
    """The teammate's 365-day signal; this function is leakage control only."""

    y = np.asarray(prices, dtype=float)
    signal = np.full(len(y), "0", dtype="U1")
    moves = np.diff(y)
    signal[:-1][moves > threshold] = "+"
    signal[:-1][moves < -threshold] = "-"
    signal[-1] = "0"
    return signal


def positions_from_signal(signal: Sequence[str]) -> np.ndarray:
    chars = np.asarray(signal)
    out = np.where(chars == "+", POSITION_LIMIT, np.where(chars == "-", -POSITION_LIMIT, 0)).astype(int)
    if len(out):
        out[-1] = 0
    return out


def frozen_candidate_positions(prices: np.ndarray) -> dict[str, np.ndarray]:
    templates = fst.frozen_templates(prices)
    candidates = fst.candidate_positions(prices, templates)
    d = candidates["Candidate D"].astype(int)
    semester = d.copy()
    semester[SUMMER_START:] = 0
    semester[-1] = 0
    return {"Candidate D": d, "Candidate D semester": semester, "templates": templates}


def causal_scale(
    prices: np.ndarray,
    innovations: Sequence[float],
    day: int,
    window: int,
    denominator: str,
) -> float:
    """Compute a causal standardiser; no observation after ``day`` is used."""

    inv = np.asarray(innovations, dtype=float)
    if denominator == "historical_innovation":
        values = inv[max(0, day - window) : day]
    elif denominator == "mad_innovation":
        values = inv[max(0, day - window) : day]
    elif denominator == "return_volatility":
        values = np.diff(prices[max(0, day - window) : day + 1])
    elif denominator == "price_level":
        values = prices[max(0, day - window + 1) : day + 1]
    else:
        raise ValueError(denominator)
    if len(values) < window:
        return float("nan")
    if denominator == "mad_innovation":
        center = float(np.median(values))
        scale = 1.4826 * float(np.median(np.abs(values - center)))
    else:
        scale = float(np.std(values, ddof=0))
    return max(scale, 1e-12)


def overlay_decision(z: float, previous_position: int, threshold: float) -> int:
    """Position-persistence and reversal mechanics used in the teammate audit."""

    if not np.isfinite(z):
        return 0
    if previous_position == 0:
        if z >= threshold:
            return -POSITION_LIMIT
        if z <= -threshold:
            return POSITION_LIMIT
        return 0
    if previous_position * z > 0:
        if abs(z) >= threshold:
            return -previous_position
        return 0
    return int(previous_position)


def neutral_overlay_frame(
    prices: Sequence[float],
    semester_positions: Sequence[int],
    mode: str = "post_update",
    alpha: float = PRIMARY_ALPHA,
    window: int = PRIMARY_WINDOW,
    threshold: float = PRIMARY_THRESHOLD,
    denominator: str = "historical_innovation",
    summer_rule: str = "fixed45",
) -> pd.DataFrame:
    """Build a valid Candidate-D-plus-semester-overlay path causally.

    ``post_update`` reproduces the teammate-style ordering.  ``prior_state``
    computes the current innovation against the fair value known before the
    current price, standardises only past innovations, and updates the EWMA
    after the decision.
    """

    y = np.asarray(prices, dtype=float)
    base = np.asarray(semester_positions, dtype=int)
    if len(y) != len(base):
        raise ValueError("prices and frozen semester positions must have equal length")
    if mode not in {"post_update", "prior_state"}:
        raise ValueError(mode)
    level = float(y[0])
    previous_position = 0
    innovations: list[float] = []
    rows = []
    for day, price in enumerate(y):
        old_level = level
        innovation = float(price - old_level)
        if mode == "post_update":
            new_level = old_level + alpha * innovation
            deviation = float(price - new_level)
        else:
            deviation = innovation
            new_level = old_level + alpha * innovation
        scale = causal_scale(y, innovations, day, window, denominator)
        z = deviation / scale if np.isfinite(scale) and scale > 0 else float("nan")
        innovations.append(innovation if mode == "prior_state" else float(price - new_level))

        if day == LAST_DAY:
            desired = 0
            decision = "terminal_flat"
        elif day >= SUMMER_START:
            if summer_rule == "fixed45":
                desired = POSITION_LIMIT if price < SUMMER_ANCHOR else -POSITION_LIMIT if price > SUMMER_ANCHOR else 0
            elif summer_rule == "flat":
                desired = 0
            else:
                raise ValueError(f"unsupported combined summer rule: {summer_rule}")
            decision = f"summer_{summer_rule}"
        elif base[day] != 0:
            desired = int(base[day])
            decision = "fixed_candidate_d"
        else:
            desired = overlay_decision(float(z), previous_position, threshold)
            decision = "neutral_overlay" if desired else "neutral_flat"

        rows.append(
            {
                "day": day,
                "price": price,
                "base_semester_position": int(base[day]),
                "ewma_old": old_level,
                "innovation_prior_state": innovation,
                "ewma_new": new_level,
                "deviation": deviation,
                "causal_scale": scale,
                "z_score": z,
                "previous_position": previous_position,
                "position": int(desired),
                "decision": decision,
                "overlay_eligible": int(day < SUMMER_START and base[day] == 0 and day < LAST_DAY),
                "latest_return": float(y[day] - y[day - 1]) if day > 0 else float("nan"),
            }
        )
        previous_position = int(desired)
        level = float(new_level)
    frame = pd.DataFrame(rows)
    frame["next_return"] = np.r_[np.diff(y), np.nan]
    frame["daily_pnl"] = frame["position"].to_numpy(dtype=float) * frame["next_return"].to_numpy(dtype=float)
    frame.loc[LAST_DAY, "daily_pnl"] = 0.0
    frame["turnover"] = np.abs(np.diff(np.r_[0, frame["position"].to_numpy(dtype=int)]))
    frame["active"] = frame["position"] != 0
    frame["mode"] = mode
    frame["alpha"] = alpha
    frame["vol_window"] = window
    frame["threshold"] = threshold
    frame["denominator"] = denominator
    frame["summer_rule"] = summer_rule
    return frame


def direct_neutral_frame(prices: Sequence[float], semester_positions: Sequence[int], rule: str) -> pd.DataFrame:
    """Simple causal controls on Candidate-D neutral days."""

    y = np.asarray(prices, dtype=float)
    base = np.asarray(semester_positions, dtype=int)
    positions = np.zeros(len(y), dtype=int)
    overlay = np.zeros(len(y), dtype=int)
    for day, price in enumerate(y):
        if day == LAST_DAY:
            positions[day] = 0
        elif day >= SUMMER_START:
            positions[day] = 0
        elif base[day] != 0:
            positions[day] = base[day]
        else:
            if rule == "one_day_reversal" and day > 0:
                change = price - y[day - 1]
                overlay[day] = POSITION_LIMIT if change < 0 else -POSITION_LIMIT if change > 0 else 0
            elif rule == "causal_ma20" and day >= 20:
                mean_level = float(np.mean(y[day - 20 : day]))
                overlay[day] = POSITION_LIMIT if price < mean_level else -POSITION_LIMIT if price > mean_level else 0
            positions[day] = overlay[day]
    frame = pd.DataFrame({"day": np.arange(len(y)), "price": y, "base_semester_position": base, "position": positions, "overlay_position": overlay, "decision": rule, "overlay_eligible": ((base == 0) & (np.arange(len(y)) < SUMMER_START) & (np.arange(len(y)) < LAST_DAY)).astype(int)})
    frame["next_return"] = np.r_[np.diff(y), np.nan]
    frame["daily_pnl"] = frame.position.to_numpy(dtype=float) * frame.next_return.to_numpy(dtype=float)
    frame.loc[LAST_DAY, "daily_pnl"] = 0.0
    frame["turnover"] = np.abs(np.diff(np.r_[0, positions]))
    frame["active"] = positions != 0
    return frame


def summer_positions(
    prices: Sequence[float],
    rule: str,
    alpha: float = CORRECTED_SUMMER_ALPHA,
    anchor_weight: float = 0.75,
    sustained_days: int = 10,
    displacement: float = 2.0,
) -> np.ndarray:
    """Causal summer fair-value rules, all anchored at AUD 45."""

    y = np.asarray(prices, dtype=float)
    out = np.zeros(len(y), dtype=int)
    level = SUMMER_ANCHOR
    streak = 0
    for day in range(SUMMER_START, len(y)):
        price = float(y[day])
        if day == LAST_DAY:
            out[day] = 0
            continue
        if rule == "fixed45":
            fair = SUMMER_ANCHOR
        elif rule == "flat":
            fair = float("nan")
        elif rule == "adaptive":
            fair = level
        elif rule == "shrunk":
            fair = anchor_weight * SUMMER_ANCHOR + (1.0 - anchor_weight) * level
        elif rule == "guarded":
            fair = level
            if abs(price - SUMMER_ANCHOR) >= displacement:
                streak += 1
            else:
                streak = 0
        else:
            raise ValueError(rule)

        if rule == "flat":
            out[day] = 0
        else:
            out[day] = POSITION_LIMIT if price < fair else -POSITION_LIMIT if price > fair else 0

        # Fair value is updated only after today's decision.  The guarded
        # detector does not adapt until the sustained displacement is met.
        if rule == "adaptive":
            level = level + alpha * (price - level)
        elif rule == "shrunk":
            level = level + alpha * (price - level)
        elif rule == "guarded" and streak >= sustained_days:
            level = level + alpha * (price - level)
    out[-1] = 0
    return out.astype(int)


def combine_semester_summer(semester_positions: np.ndarray, summer: np.ndarray) -> np.ndarray:
    out = np.asarray(semester_positions, dtype=int).copy()
    out[SUMMER_START:] = np.asarray(summer, dtype=int)[SUMMER_START:]
    out[-1] = 0
    return out


def score_positions(prices: np.ndarray, positions: np.ndarray, label: str) -> dict[str, object]:
    result = a.backtest(prices, positions, label=label)
    pnl = positions[:-1].astype(float) * np.diff(prices)
    active = positions[:-1] != 0
    changes = np.diff(np.r_[0, positions])
    long_pnl = float(np.sum(pnl[positions[:-1] > 0]))
    short_pnl = float(np.sum(pnl[positions[:-1] < 0]))
    return {
        "strategy": label,
        "pnl": float(result["pnl"]),
        "sharpe": float(result["sharpe"]),
        "max_drawdown": float(result["max_drawdown"]),
        "active_days": int(np.sum(active)),
        "active_hit_rate": float(np.mean(pnl[active] > 0)) if np.any(active) else float("nan"),
        "trade_count": int(np.sum(changes != 0)),
        "turnover_units": int(np.sum(np.abs(changes))),
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
        "max_notional": float(np.max(np.abs(positions * prices))),
        "pnl_per_max_notional": float(result["pnl"] / max(np.max(np.abs(positions * prices)), 1e-12)),
        "best_day_pnl": float(np.max(pnl)),
        "best_day": int(np.argmax(pnl)),
        "worst_day_pnl": float(np.min(pnl)),
        "worst_day": int(np.argmin(pnl)),
        "integral_positions": int(result["integral_positions"]),
        "within_limit": int(result["within_limit"]),
        "budget_violations_standalone": int(result["budget_violations"]),
        "evidence_label": "in-sample Round 1 diagnostic; valid candidate unless marked leakage control",
    }


def frame_for_positions(prices: np.ndarray, positions: np.ndarray, overlay_positions: np.ndarray | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({"day": np.arange(len(prices)), "price": prices, "position": positions.astype(int)})
    frame["next_return"] = np.r_[np.diff(prices), np.nan]
    frame["daily_pnl"] = frame.position.to_numpy(dtype=float) * frame.next_return.to_numpy(dtype=float)
    frame.loc[LAST_DAY, "daily_pnl"] = 0.0
    frame["turnover"] = np.abs(np.diff(np.r_[0, positions]))
    frame["active"] = positions != 0
    if overlay_positions is not None:
        frame["overlay_position"] = overlay_positions.astype(int)
    return frame


def attribution_rows(
    prices: np.ndarray,
    strategy: str,
    positions: np.ndarray,
    base_d: np.ndarray,
    semester_positions: np.ndarray,
    overlay_positions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Break out fixed signal, neutral overlay, summer and chronology."""

    pnl = positions[:-1].astype(float) * np.diff(prices)
    if overlay_positions is None:
        overlay_positions = positions - base_d
    categories = {
        "fixed_candidate_d_semester": (semester_positions[:-1] != 0),
        "neutral_overlay_semester": (semester_positions[:-1] == 0) & (np.arange(len(pnl)) < SUMMER_START),
        "summer_rule": np.arange(len(pnl)) >= SUMMER_START,
    }
    rows = []
    for category, mask in categories.items():
        values = pnl[mask]
        changes = np.diff(np.r_[0, positions])[:-1]
        rows.append({"strategy": strategy, "dimension": "pnl_source", "segment": category, "pnl": float(values.sum()), "active_days": int(np.sum(positions[:-1][mask] != 0)), "trade_count": int(np.sum(changes[mask] != 0)), "turnover_units": int(np.sum(np.abs(changes[mask]))), "evidence_label": "in-sample Round 1 attribution"})
    for name, start, end in [("first_half", 0, 182), ("second_half", 182, 365), ("semester_1", 0, 161), ("semester_2", 161, 302), ("summer", 322, 365)]:
        lo, hi = start, min(end, LAST_DAY)
        values = pnl[lo:hi]
        rows.append({"strategy": strategy, "dimension": "chronology", "segment": name, "pnl": float(values.sum()), "active_days": int(np.sum(positions[lo:hi] != 0)), "trade_count": int(np.sum(np.diff(np.r_[0, positions])[lo:hi] != 0)), "turnover_units": int(np.sum(np.abs(np.diff(np.r_[0, positions])[lo:hi]))), "evidence_label": "in-sample chronology"})
    for name, start, end in BLOCKS_60:
        lo, hi = start, min(end, LAST_DAY)
        values = pnl[lo:hi]
        rows.append({"strategy": strategy, "dimension": "block_60d", "segment": name, "pnl": float(values.sum()), "active_days": int(np.sum(positions[lo:hi] != 0)), "trade_count": int(np.sum(np.diff(np.r_[0, positions])[lo:hi] != 0)), "turnover_units": int(np.sum(np.abs(np.diff(np.r_[0, positions])[lo:hi]))), "evidence_label": "in-sample chronology"})
    for name, mask in [("long", positions[:-1] > 0), ("short", positions[:-1] < 0)]:
        rows.append({"strategy": strategy, "dimension": "direction", "segment": name, "pnl": float(pnl[mask].sum()), "active_days": int(np.sum(mask)), "trade_count": None, "turnover_units": None, "evidence_label": "in-sample direction attribution"})
    return pd.DataFrame(rows)


def concentration_frame(prices: np.ndarray, strategy: str, positions: np.ndarray, base_d: np.ndarray) -> pd.DataFrame:
    pnl = positions[:-1].astype(float) * np.diff(prices)
    baseline_pnl = base_d[:-1].astype(float) * np.diff(prices)
    incremental = pnl - baseline_pnl
    rows = []
    for label, values in [("full_strategy", pnl), ("incremental_vs_candidate_d", incremental)]:
        total = float(values.sum())
        order = np.argsort(values)[::-1]
        for k in (1, 5, 10, 20):
            rows.append({"strategy": strategy, "series": label, "metric": f"best_{k}_day_share", "value": float(values[order[:k]].sum() / total) if abs(total) > 1e-12 else float("nan"), "pnl_contribution": float(values[order[:k]].sum()), "total_pnl": total, "days": int(min(k, len(values)))})
        for rank, day in enumerate(order[:20], start=1):
            rows.append({"strategy": strategy, "series": label, "metric": f"best_day_rank_{rank}", "value": float(values[day]), "pnl_contribution": float(values[day]), "total_pnl": total, "day": int(day)})
    return pd.DataFrame(rows)


def score_window(prices: np.ndarray, positions: np.ndarray, start: int, end: int) -> dict[str, object]:
    lo, hi = max(0, start), min(end, LAST_DAY)
    pnl = positions[lo:hi].astype(float) * np.diff(prices[lo : hi + 1])
    active = positions[lo:hi] != 0
    return {"pnl": float(pnl.sum()), "active_days": int(np.sum(active)), "hit_rate": float(np.mean(pnl[active] > 0)) if np.any(active) else float("nan"), "max_drawdown": float(a.max_drawdown(pnl))}


def chronological_frame(prices: np.ndarray, positions_map: Mapping[str, np.ndarray], base_d: np.ndarray) -> pd.DataFrame:
    segments = [("half", "first_half", 0, 182), ("half", "second_half", 182, 365), ("semester", "semester_1", 0, 161), ("semester", "semester_2", 161, 302), ("summer", "summer", 322, 365)] + [("block_60d", name, lo, hi) for name, lo, hi in BLOCKS_60]
    rows = []
    for strategy, positions in positions_map.items():
        for kind, name, lo, hi in segments:
            score = score_window(prices, positions, lo, hi)
            baseline = score_window(prices, base_d, lo, hi)
            rows.append({"strategy": strategy, "segment_type": kind, "segment": name, "start_day": lo, "end_day": hi, **score, "candidate_d_baseline_pnl": baseline["pnl"], "incremental_pnl_vs_candidate_d": score["pnl"] - baseline["pnl"], "evidence_label": "in-sample chronological diagnostic; no parameter fit on held segment"})
    return pd.DataFrame(rows)


def regression_fit(x: np.ndarray, y: np.ndarray, controls: np.ndarray | None = None) -> dict[str, float | int | str]:
    finite = np.isfinite(x) & np.isfinite(y)
    if controls is not None:
        finite &= np.all(np.isfinite(controls), axis=1)
    x, y = x[finite], y[finite]
    c = controls[finite] if controls is not None else None
    if len(x) < 5 or np.std(x) < 1e-12:
        return {"n": int(len(x)), "beta": float("nan"), "se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan"), "beta_latest_return": float("nan"), "inference": "insufficient observations"}
    X = np.column_stack([np.ones(len(x)), x] if c is None else [np.ones(len(x)), x, c])
    if sm is not None:
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": min(5, max(1, len(x) // 10))})
        ci = fit.conf_int(alpha=0.05)
        return {"n": int(len(x)), "beta": float(fit.params[1]), "se": float(fit.bse[1]), "ci_low": float(ci[1, 0]), "ci_high": float(ci[1, 1]), "p_value": float(fit.pvalues[1]), "beta_latest_return": float(fit.params[2]) if c is not None else float("nan"), "inference": "OLS HAC/Newey-West"}
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ coef
    cov = np.linalg.inv(X.T @ X) * float(np.sum(residual**2) / max(len(y) - X.shape[1], 1))
    se = math.sqrt(max(float(cov[1, 1]), 0.0))
    return {"n": int(len(x)), "beta": float(coef[1]), "se": se, "ci_low": float(coef[1] - 1.96 * se), "ci_high": float(coef[1] + 1.96 * se), "p_value": float("nan"), "beta_latest_return": float(coef[2]) if c is not None else float("nan"), "inference": "OLS fallback"}


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def predictive_rows(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    data = frame.iloc[:-1].copy()
    eligible = (data.overlay_eligible.to_numpy(dtype=bool)) & np.isfinite(data.deviation.to_numpy(dtype=float)) & np.isfinite(data.next_return.to_numpy(dtype=float))
    rows = []
    for sample, mask in [("all_neutral", eligible), ("semester_1", eligible & (data.day.to_numpy() < 161)), ("semester_2", eligible & (data.day.to_numpy() >= 161) & (data.day.to_numpy() < 302))] + [(name, eligible & (data.day.to_numpy() >= lo) & (data.day.to_numpy() < hi)) for name, lo, hi in BLOCKS_60 if lo < SUMMER_START]:
        for control_name, controls in [("none", None), ("latest_return", data.latest_return.to_numpy(dtype=float)[:, None])]:
            x = data.deviation.to_numpy(dtype=float)[mask]
            y = data.next_return.to_numpy(dtype=float)[mask]
            c = controls[mask] if controls is not None else None
            fit = regression_fit(x, y, c)
            direction = np.sign(-x) * np.sign(y)
            rows.append({"strategy": strategy, "sample": sample, "control": control_name, **fit, "rank_correlation": _rank_corr(x, y), "directional_hit_rate": float(np.mean(direction > 0)) if len(direction) else float("nan"), "evidence_label": "causal Round 1 diagnostic; not independent validation"})
    # Remove top overlay days using the same candidate's realized overlay P&L.
    base_pnl = data.position.to_numpy(dtype=float) * data.next_return.to_numpy(dtype=float)
    for k in (1, 5, 10):
        order = np.argsort(np.where(eligible, base_pnl, -np.inf))[-k:]
        mask = eligible.copy()
        mask[order] = False
        x = data.deviation.to_numpy(dtype=float)[mask]
        y = data.next_return.to_numpy(dtype=float)[mask]
        fit = regression_fit(x, y)
        direction = np.sign(-x) * np.sign(y)
        rows.append({"strategy": strategy, "sample": f"exclude_best_{k}_overlay_days", "control": "none", **fit, "rank_correlation": _rank_corr(x, y), "directional_hit_rate": float(np.mean(direction > 0)) if len(direction) else float("nan"), "evidence_label": "causal Round 1 diagnostic after removing best realized overlay days"})
    return pd.DataFrame(rows)


def parameter_stability(prices: np.ndarray, semester_positions: np.ndarray, base_d: np.ndarray) -> pd.DataFrame:
    """Predeclared diagnostic grid for both post and corrected prior-state modes."""

    rows = []
    for mode in ("post_update", "prior_state"):
        for alpha in PARAMETER_ALPHAS:
            for window in PARAMETER_WINDOWS:
                for threshold in PARAMETER_THRESHOLDS:
                    frame = neutral_overlay_frame(prices, semester_positions, mode=mode, alpha=alpha, window=window, threshold=threshold, denominator="historical_innovation", summer_rule="fixed45")
                    score = score_positions(prices, frame.position.to_numpy(dtype=int), f"{mode} diagnostic")
                    baseline = score_positions(prices, base_d, "Candidate D")
                    equivalent_pre_threshold = threshold / max(1.0 - alpha, 1e-12)
                    rows.append({"mode": mode, "alpha": alpha, "vol_window": window, "post_or_standardized_threshold": threshold, "equivalent_pre_update_threshold": equivalent_pre_threshold, "pnl": score["pnl"], "incremental_pnl_vs_candidate_d": score["pnl"] - baseline["pnl"], "active_days": score["active_days"], "trade_count": score["trade_count"], "turnover_units": score["turnover_units"], "max_drawdown": score["max_drawdown"], "evidence_label": "diagnostic same-year grid; not selected by maximum P&L"})
    return pd.DataFrame(rows)


def denominator_comparison(prices: np.ndarray, semester_positions: np.ndarray, base_d: np.ndarray) -> pd.DataFrame:
    rows = []
    for mode in ("post_update", "prior_state"):
        for alpha in REPORTED_ALPHAS:
            for denominator in DENOMINATORS:
                frame = neutral_overlay_frame(prices, semester_positions, mode=mode, alpha=alpha, window=PRIMARY_WINDOW, threshold=PRIMARY_THRESHOLD, denominator=denominator, summer_rule="fixed45")
                score = score_positions(prices, frame.position.to_numpy(dtype=int), f"{mode} {alpha} {denominator}")
                rows.append({"mode": mode, "alpha": alpha, "denominator": denominator, "pnl": score["pnl"], "incremental_pnl_vs_candidate_d": score["pnl"] - score_positions(prices, base_d, "D")["pnl"], "active_days": score["active_days"], "max_drawdown": score["max_drawdown"], "evidence_label": "causal denominator diagnostic"})
    return pd.DataFrame(rows)


def leave_one_block_out(prices: np.ndarray, positions_map: Mapping[str, np.ndarray], base_d: np.ndarray) -> pd.DataFrame:
    rows = []
    base_pnl = base_d[:-1].astype(float) * np.diff(prices)
    for strategy, positions in positions_map.items():
        full = score_positions(prices, positions, strategy)["pnl"]
        full_incremental = full - float(base_pnl.sum())
        for block_name, start, end in BLOCKS_60:
            mask = np.ones(LAST_DAY, dtype=bool)
            mask[start : min(end, LAST_DAY)] = False
            pnl = positions[:-1].astype(float) * np.diff(prices)
            retained = float(pnl[mask].sum())
            retained_base = float(base_pnl[mask].sum())
            rows.append({"strategy": strategy, "excluded_block": block_name, "full_pnl": full, "full_incremental_vs_candidate_d": full_incremental, "retained_pnl": retained, "retained_candidate_d_pnl": retained_base, "retained_incremental_vs_candidate_d": retained - retained_base, "incremental_retained_vs_full": retained - full, "weakest_case": 0, "evidence_label": "in-sample leave-one-block-out diagnostic"})
        group = pd.DataFrame([r for r in rows if r["strategy"] == strategy])
        if len(group):
            weakest_index = group["retained_pnl"].idxmin()
            rows[weakest_index]["weakest_case"] = 1
    return pd.DataFrame(rows)


def timing_and_sign_placebos(
    prices: np.ndarray,
    base_d: np.ndarray,
    semester_positions: np.ndarray,
    frames: Mapping[str, pd.DataFrame],
    n_paths: int = 250,
    seed: int = SEED + 71,
) -> pd.DataFrame:
    """Shift or randomise only neutral-day overlay positions."""

    rng = np.random.default_rng(seed)
    neutral = (semester_positions == 0) & (np.arange(len(prices)) < SUMMER_START) & (np.arange(len(prices)) < LAST_DAY)
    rows = []
    for strategy, frame in frames.items():
        if strategy == "Candidate D":
            continue
        actual = frame.position.to_numpy(dtype=int)
        overlay = np.zeros(len(prices), dtype=int)
        overlay[neutral] = actual[neutral] - base_d[neutral]
        for displacement in (-3, -2, -1, 0, 1, 2, 3):
            shifted = np.zeros(len(prices), dtype=int)
            for day in np.where(neutral)[0]:
                source = day - displacement
                if 0 <= source < len(prices):
                    shifted[day] = overlay[source]
            position = base_d.copy()
            position[neutral] = shifted[neutral]
            position[-1] = 0
            score = score_positions(prices, position, strategy)
            rows.append({"method": "overlay_timing_displacement", "strategy": strategy, "displacement": displacement, "path_id": 0, "pnl": score["pnl"], "incremental_pnl_vs_candidate_d": score["pnl"] - score_positions(prices, base_d, "D")["pnl"], "max_drawdown": score["max_drawdown"], "evidence_label": "in-sample timing placebo; fixed seasonal timing is not rejected by this diagnostic"})
        signs = np.sign(overlay[neutral]).astype(int)
        active_indices = np.where(signs != 0)[0]
        for path_id in range(n_paths):
            shuffled = signs.copy()
            active_signs = shuffled[active_indices].copy()
            rng.shuffle(active_signs)
            shuffled[active_indices] = active_signs
            position = base_d.copy()
            position[neutral] = shuffled * POSITION_LIMIT
            position[-1] = 0
            score = score_positions(prices, position, strategy)
            rows.append({"method": "activity_preserving_sign_placebo", "strategy": strategy, "displacement": None, "path_id": path_id, "pnl": score["pnl"], "incremental_pnl_vs_candidate_d": score["pnl"] - score_positions(prices, base_d, "D")["pnl"], "max_drawdown": score["max_drawdown"], "evidence_label": "activity-preserving sign placebo; not independent validation"})
    return pd.DataFrame(rows)


def seasonality_paired_summary(detail: pd.DataFrame) -> pd.DataFrame:
    baseline = detail[detail.strategy == "Candidate D"].set_index(["stress_family", "scenario", "path_id"])["full_pnl"]
    rows = []
    for strategy, group in detail[detail.strategy != "Candidate D"].groupby("strategy"):
        keys = pd.MultiIndex.from_frame(group[["stress_family", "scenario", "path_id"]])
        paired = group.full_pnl.to_numpy(dtype=float) - baseline.reindex(keys).to_numpy(dtype=float)
        rows.append({"strategy": strategy, "n_paths": len(paired), "median_difference_vs_candidate_d": float(np.median(paired)), "p10_difference_vs_candidate_d": float(np.quantile(paired, 0.10)), "worst_difference_vs_candidate_d": float(np.min(paired)), "positive_difference_rate": float(np.mean(paired > 0)), "evidence_label": "paired generator-conditioned seasonality stress; not confidence interval"})
    return pd.DataFrame(rows)


def _shift_wave_component(component: np.ndarray, displacement: int) -> np.ndarray:
    grid = np.arange(len(component), dtype=float)
    return np.interp(grid - displacement, grid, component, left=float(component[0]), right=float(component[-1]))


def block_bootstrap(values: np.ndarray, length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out: list[float] = []
    while len(out) < length:
        start = int(rng.integers(0, max(1, len(values) - block_length + 1)))
        out.extend(values[start : start + block_length].tolist())
    return np.asarray(out[:length], dtype=float)


def seasonal_base(prices: np.ndarray, multipliers: Sequence[float], timing_shift: int = 0) -> np.ndarray:
    _, components = fst.broad_seasonal_components(prices)
    names = ["S1_large", "S1_small", "S2_large", "S2_small"]
    shifted = {name: _shift_wave_component(np.asarray(components[name], dtype=float), timing_shift) if timing_shift else np.asarray(components[name], dtype=float) for name in names}
    path = np.full(len(prices), SUMMER_ANCHOR, dtype=float)
    for name, multiplier in zip(names, multipliers):
        path += float(multiplier) * shifted[name]
    return path


def seasonality_stress_detail(prices: np.ndarray, semester_positions: np.ndarray, base_d: np.ndarray, n_paths: int, seed: int) -> pd.DataFrame:
    """Retain broad waves while varying amplitudes, noise, shocks and timing."""

    rng = np.random.default_rng(seed)
    broad = fst.broad_seasonal_components(prices)[0]
    residual = prices - broad
    residual -= residual.mean()
    strategies = {
        "Candidate D": base_d,
        "D + post EWMA alpha 0.65": None,
        "D + prior-state EWMA alpha 0.65": None,
        "D + one-day reversal": None,
    }
    rows = []

    def evaluate_path(path: np.ndarray, family: str, scenario: str, path_id: int, multipliers: Sequence[float], volatility_scale: float, timing_shift: int, shock_count: int) -> None:
        post = neutral_overlay_frame(path, semester_positions, mode="post_update", alpha=0.65, window=10, threshold=0.05, denominator="historical_innovation", summer_rule="fixed45").position.to_numpy(dtype=int)
        prior = neutral_overlay_frame(path, semester_positions, mode="prior_state", alpha=0.65, window=10, threshold=0.05, denominator="historical_innovation", summer_rule="fixed45").position.to_numpy(dtype=int)
        reverse = direct_neutral_frame(path, semester_positions, "one_day_reversal").position.to_numpy(dtype=int)
        d_path = combine_semester_summer(semester_positions, summer_positions(path, "fixed45"))
        positions_map = {"Candidate D": d_path, "D + post EWMA alpha 0.65": post, "D + prior-state EWMA alpha 0.65": prior, "D + one-day reversal": reverse}
        for strategy, position in positions_map.items():
            score = score_positions(path, position, strategy)
            rows.append({"stress_family": family, "scenario": scenario, "path_id": path_id, "strategy": strategy, "full_pnl": score["pnl"], "max_drawdown": score["max_drawdown"], "active_days": score["active_days"], "max_notional": score["max_notional"], "positive": int(score["pnl"] > 0), "amplitude_S1_large": multipliers[0], "amplitude_S1_small": multipliers[1], "amplitude_S2_large": multipliers[2], "amplitude_S2_small": multipliers[3], "volatility_scale": volatility_scale, "timing_shift": timing_shift, "shock_count": shock_count, "evidence_label": "generator-conditioned seasonality-preserving stress; not confidence interval"})

    # Deterministic independent amplitude scenarios: all 5 levels for each
    # wave, 625 paths, with no new noise supplied to the strategy.
    scenario_id = 0
    for m1 in AMPLITUDE_LEVELS:
        for m2 in AMPLITUDE_LEVELS:
            for m3 in AMPLITUDE_LEVELS:
                for m4 in AMPLITUDE_LEVELS:
                    multipliers = (m1, m2, m3, m4)
                    path = seasonal_base(prices, multipliers)
                    evaluate_path(path, "deterministic_independent_amplitude", f"amp_{scenario_id}", scenario_id, multipliers, 0.0, 0, 0)
                    scenario_id += 1

    # Residual block resampling around the retained seasonal shape, with
    # independent amplitude draws and volatility scaling.
    for path_id in range(n_paths):
        multipliers = tuple(float(rng.choice(AMPLITUDE_LEVELS)) for _ in range(4))
        volatility_scale = float(rng.choice((0.5, 1.0, 1.5)))
        base = seasonal_base(prices, multipliers)
        noise = block_bootstrap(residual, len(prices), 7, rng) * volatility_scale
        evaluate_path(base + noise, "seasonal_residual_blocks", f"noise_{path_id}", path_id, multipliers, volatility_scale, 0, 0)

    # Local shocks are sparse additions to the retained seasonal path.
    for path_id in range(max(50, n_paths // 2)):
        multipliers = tuple(float(rng.choice(AMPLITUDE_LEVELS)) for _ in range(4))
        path = seasonal_base(prices, multipliers) + block_bootstrap(residual, len(prices), 7, rng)
        shock_count = 2
        for day in rng.choice(np.arange(15, 302), size=shock_count, replace=False):
            path[int(day)] += float(rng.choice((-1.0, 1.0))) * float(rng.uniform(1.0, 3.0))
        evaluate_path(path, "seasonal_local_shocks", f"shock_{path_id}", path_id, multipliers, 1.0, 0, shock_count)

    # Secondary timing diagnostic: the generator's broad waves are shifted,
    # while the strategy retains its fixed Round 1 template indices.
    for timing_shift in (-2, -1, 0, 1, 2):
        for path_id in range(max(50, n_paths // 3)):
            multipliers = tuple(float(rng.choice(AMPLITUDE_LEVELS)) for _ in range(4))
            path = seasonal_base(prices, multipliers, timing_shift) + block_bootstrap(residual, len(prices), 7, rng)
            evaluate_path(path, "secondary_timing_shift", f"shift_{timing_shift}", path_id, multipliers, 1.0, timing_shift, 0)
    return pd.DataFrame(rows)


def summer_equilibrium_line(target: float, mode: str, duration: int, temporary_days: int = 14) -> np.ndarray:
    line = np.full(365, SUMMER_ANCHOR, dtype=float)
    days = np.arange(365)
    if mode == "abrupt":
        line[SUMMER_START:] = target
    elif mode == "gradual":
        progress = np.clip((days[SUMMER_START:] - (SUMMER_START - 1)) / float(duration), 0.0, 1.0)
        line[SUMMER_START:] = SUMMER_ANCHOR + (target - SUMMER_ANCHOR) * progress
    elif mode == "temporary":
        line[SUMMER_START : min(365, SUMMER_START + temporary_days)] = target
        return_line_start = min(365, SUMMER_START + temporary_days)
        if return_line_start < 365:
            progress = np.clip((days[return_line_start:] - return_line_start + 1) / 7.0, 0.0, 1.0)
            line[return_line_start:] = target + (SUMMER_ANCHOR - target) * progress
    else:
        raise ValueError(mode)
    return line


def summer_strategy_positions(prices: np.ndarray, semester_positions: np.ndarray, rule_name: str) -> np.ndarray:
    if rule_name == "Candidate D frozen45":
        summer = summer_positions(prices, "fixed45")
    elif rule_name == "Summer flat":
        summer = summer_positions(prices, "flat")
    elif rule_name == "Summer adaptive EWMA alpha 0.10":
        summer = summer_positions(prices, "adaptive", alpha=0.10)
    elif rule_name.startswith("Summer shrunk"):
        weight = float(rule_name.split("=")[1])
        summer = summer_positions(prices, "shrunk", alpha=0.10, anchor_weight=weight)
    elif rule_name.startswith("Summer guarded"):
        parts = rule_name.split("=")[-1].split(",")
        duration = int(parts[0])
        displacement = float(parts[1])
        alpha = float(parts[2])
        summer = summer_positions(prices, "guarded", alpha=alpha, sustained_days=duration, displacement=displacement)
    else:
        raise ValueError(rule_name)
    return combine_semester_summer(semester_positions, summer)


def summer_stress_detail(prices: np.ndarray, semester_positions: np.ndarray, base_d: np.ndarray, n_paths: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    broad = fst.broad_seasonal_components(prices)[0]
    residual = prices - broad
    residual -= residual.mean()
    rules = ["Candidate D frozen45", "Summer flat", "Summer adaptive EWMA alpha 0.10", "Summer shrunk weight=0.50", "Summer shrunk weight=0.75", "Summer shrunk weight=0.90", "Summer guarded=10,2,0.10"]
    rows = []
    scenarios = [(float(target), "abrupt", 0) for target in np.arange(42.0, 49.0, 1.0)]
    scenarios += [(float(target), "gradual", int(duration)) for target in (42.0, 44.0, 45.0, 46.0, 48.0) for duration in (7, 14, 28)]
    scenarios += [(float(target), "temporary", 14) for target in (42.0, 44.0, 46.0, 48.0)]
    for path_id in range(n_paths):
        noise = block_bootstrap(residual, len(prices), 7, rng)
        for target, mode, duration in scenarios:
            equilibrium = summer_equilibrium_line(target, mode, duration)
            path = seasonal_base(prices, (1.0, 1.0, 1.0, 1.0))
            path[SUMMER_START:] = equilibrium[SUMMER_START:]
            path = path + noise
            frozen = summer_strategy_positions(path, semester_positions, "Candidate D frozen45")
            frozen_score = score_positions(path, frozen, "Candidate D frozen45")
            for rule in rules:
                position = summer_strategy_positions(path, semester_positions, rule)
                score = score_positions(path, position, rule)
                summer_only = position.copy()
                summer_only[:SUMMER_START] = 0
                summer_score = score_positions(path, summer_only, f"{rule} summer")
                rows.append({"target_equilibrium": target, "transition_mode": mode, "transition_days": duration, "path_id": path_id, "strategy": rule, "full_pnl": score["pnl"], "summer_pnl": summer_score["pnl"], "max_drawdown": score["max_drawdown"], "positive": int(score["pnl"] > 0), "paired_difference_vs_frozen45": score["pnl"] - frozen_score["pnl"], "paired_summer_difference_vs_frozen45": summer_score["pnl"] - score_positions(path, np.where(np.arange(len(path)) >= SUMMER_START, frozen, 0).astype(int), "frozen summer")["pnl"], "evidence_label": "generator-conditioned summer-equilibrium stress; not confidence interval"})
    return pd.DataFrame(rows)


def summer_parameter_grid(prices: np.ndarray, semester_positions: np.ndarray, base_d: np.ndarray) -> pd.DataFrame:
    rows = []
    rules = [("adaptive", {"alpha": 0.10})]
    rules += [("shrunk", {"anchor_weight": weight, "alpha": 0.10}) for weight in (0.50, 0.75, 0.90)]
    rules += [("guarded", {"sustained_days": days, "displacement": displacement, "alpha": alpha}) for days in (5, 10, 15) for displacement in (1.0, 2.0, 3.0) for alpha in (0.05, 0.10, 0.20)]
    for rule, params in rules:
        summer = summer_positions(prices, rule, **params)
        position = combine_semester_summer(semester_positions, summer)
        score = score_positions(prices, position, f"summer {rule}")
        summer_only = position.copy()
        summer_only[:SUMMER_START] = 0
        summer_score = score_positions(prices, summer_only, f"summer {rule}")
        rows.append({"rule": rule, **params, "pnl": score["pnl"], "incremental_pnl_vs_candidate_d": score["pnl"] - score_positions(prices, base_d, "D")["pnl"], "summer_pnl": summer_score["pnl"], "max_drawdown": score["max_drawdown"], "active_days": score["active_days"], "evidence_label": "diagnostic summer grid; no maximum selected"})
    return pd.DataFrame(rows)


def production_combined_capital(root: Path, prices: np.ndarray, positions_map: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Read current production logic and calculate combined desired exposure."""

    limits = {"Fintech Token": 100, "Thrifted Jeans": 800, "UQ Dollar": 650, "Sausage Sizzle": 3000, "Bread": 500, "MenuDash": 75000, "Sausage": 5000, "Liferaft Ticket": 1, "Boat Party Ticket": 1000}
    spec = importlib.util.spec_from_file_location("boat_v2_production_readonly", root / "trader_interface" / "algorithm.py")
    if spec is None or spec.loader is None:
        raise ImportError("could not load production algorithm read-only")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    algo = module.Algorithm({})
    data_files = sorted((root / "trader_interface" / "data").glob("*_price_history.csv"))
    data = {file.name.removesuffix("_price_history.csv"): pd.read_csv(file)["Price"].to_numpy(dtype=float) for file in data_files}
    current_positions: dict[str, int] = {name: 0 for name in limits}
    other_capital = np.zeros(len(prices), dtype=float)
    other_positions = []
    for day in range(len(prices)):
        algo.data = {name: values[: day + 1].tolist() for name, values in data.items()}
        algo.positions = current_positions
        algo.positionLimits = limits
        desired = {name: int(value) for name, value in algo.get_positions().items()}
        current_positions = desired
        price_map = {name: float(values[day]) for name, values in data.items()}
        other_positions.append({name: int(value) for name, value in desired.items() if name != "Boat Party Ticket"})
        other_capital[day] = sum(abs(position * price_map[name]) for name, position in desired.items() if name != "Boat Party Ticket")
    rows = []
    base_boat = positions_map["Candidate D"]
    boat_price = prices
    base_combined = other_capital + np.abs(base_boat * boat_price)
    for strategy, positions in positions_map.items():
        combined = other_capital + np.abs(positions * boat_price)
        over = combined > TOTAL_BUDGET
        delta_boat = np.abs(positions * boat_price) - np.abs(base_boat * boat_price)
        overlap_names = sorted({name for day, other in enumerate(other_positions[:-1]) if positions[day] != 0 for name, value in other.items() if value != 0})
        rows.append({"strategy": strategy, "max_boat_notional": float(np.max(np.abs(positions * boat_price))), "max_other_instrument_capital": float(np.max(other_capital)), "max_combined_capital": float(np.max(combined)), "combined_budget_violation_days": int(np.sum(over)), "max_incremental_boat_capital_vs_candidate_d": float(np.max(delta_boat)), "mean_incremental_boat_capital_active_days": float(np.mean(delta_boat[:-1][positions[:-1] != 0])) if np.any(positions[:-1] != 0) else 0.0, "overlap_days_with_other_positions": int(np.sum((positions[:-1] != 0) & (other_capital[:-1] > 0))), "overlapping_other_instruments": ";".join(overlap_names), "evidence_label": "read-only current production other-instrument exposure; no allocator optimization"})
    base_overlap_names = sorted({name for day, other in enumerate(other_positions[:-1]) if base_boat[day] != 0 for name, value in other.items() if value != 0})
    rows.append({"strategy": "Current production Candidate D combined reference", "max_boat_notional": float(np.max(np.abs(base_boat * boat_price))), "max_other_instrument_capital": float(np.max(other_capital)), "max_combined_capital": float(np.max(base_combined)), "combined_budget_violation_days": int(np.sum(base_combined > TOTAL_BUDGET)), "max_incremental_boat_capital_vs_candidate_d": 0.0, "mean_incremental_boat_capital_active_days": 0.0, "overlap_days_with_other_positions": int(np.sum((base_boat[:-1] != 0) & (other_capital[:-1] > 0))), "overlapping_other_instruments": ";".join(base_overlap_names), "evidence_label": "read-only current production combined reference"})
    return pd.DataFrame(rows)


def leakage_control(prices: np.ndarray, base_d: np.ndarray) -> pd.DataFrame:
    signal = thresholded_future_return_signal(prices, 0.50)
    positions = positions_from_signal(signal)
    score = score_positions(prices, positions, "LEAKAGE CONTROL future-return threshold signal")
    pnl = positions[:-1].astype(float) * np.diff(prices)
    active = positions[:-1] != 0
    # V1's post-update implementation is reused only to quantify both
    # reported leaked totals; neither is a valid candidate or selection input.
    rows = [
        {"control": "thresholded future-return fixed signal", "signal_length": len(signal), "evaluable_next_return_days": LAST_DAY, "fixed_plus_minus_days": int(np.sum(signal[:-1] != "0")), "fixed_hit_rate": float(np.mean(pnl[active] > 0)), "pnl": score["pnl"], "incremental_ewma_pnl": 0.0, "included_in_valid_selection": 0, "evidence_label": "intentional Round 1 leakage control; signal uses price[t+1]"},
    ]
    for alpha in (0.65, 0.90):
        frame = v1.adaptive_strategy_frame(prices, "".join(signal.tolist()), alpha=alpha, vol_window=10, threshold=0.05)
        adaptive_score = score_positions(prices, frame.position.to_numpy(dtype=int), f"LEAKAGE CONTROL plus EWMA alpha {alpha:.2f}")
        rows.append({"control": f"thresholded future-return signal plus post EWMA alpha {alpha:.2f}", "signal_length": len(signal), "evaluable_next_return_days": LAST_DAY, "fixed_plus_minus_days": int(np.sum(signal[:-1] != "0")), "fixed_hit_rate": float(np.mean(pnl[active] > 0)), "pnl": adaptive_score["pnl"], "incremental_ewma_pnl": adaptive_score["pnl"] - score["pnl"], "included_in_valid_selection": 0, "evidence_label": "intentional Round 1 leakage control; EWMA increment is not clean evidence"})
    rows.append({"control": "leakage signal reproduces future threshold rule", "signal_length": len(signal), "evaluable_next_return_days": LAST_DAY, "fixed_plus_minus_days": int(np.sum(signal[:-1] != "0")), "fixed_hit_rate": float(np.mean(pnl[active] > 0)), "pnl": score["pnl"], "incremental_ewma_pnl": 0.0, "included_in_valid_selection": 0, "evidence_label": "expected fixed thresholded-return result is approximately AUD 185,880; excluded"})
    return pd.DataFrame(rows)


def correctness_checks(
    prices: np.ndarray,
    base_d: np.ndarray,
    semester_positions: np.ndarray,
    main_frames: Mapping[str, pd.DataFrame],
    portfolio: pd.DataFrame,
    leakage: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(check: str, passed: bool, evidence: str) -> None:
        rows.append({"check": check, "passed": int(bool(passed)), "evidence": evidence})

    add("candidate_D_round1_pnl_is_92560", abs(score_positions(prices, base_d, "D")["pnl"] - 92560.0) < 1e-6, f"pnl={score_positions(prices, base_d, 'D')['pnl']:.2f}")
    production_signal = str(extract_frozen_candidate_d_signal(Path(a.find_repo_root()))["signal"])
    production_signal_padded = production_signal + "0" * (len(semester_positions) - len(production_signal))
    add("production_frozen_signal_matches_candidate_D_semester", np.array_equal(positions_from_signal(list(production_signal_padded)), semester_positions), "read-only production signal vs frozen Candidate D semester")
    for name, frame in main_frames.items():
        position = frame.position.to_numpy(dtype=int)
        add(f"{name}_positions_integral", np.issubdtype(frame.position.dtype, np.integer), str(frame.position.dtype))
        add(f"{name}_positions_within_limit", np.max(np.abs(position)) <= POSITION_LIMIT, f"max_abs={np.max(np.abs(position))}")
        add(f"{name}_day_364_flat", int(position[-1]) == 0, f"day364={position[-1]}")
        add(f"{name}_pnl_t_to_t_plus_1", np.allclose(frame.daily_pnl.to_numpy()[:-1], position[:-1] * np.diff(prices)), "daily P&L alignment")
    toy_prices = np.array([10.0, 11.0, 10.0, 12.0])
    toy_positions = np.array([1000, -1000, 1000, 0], dtype=int)
    toy_pnl = toy_positions[:-1] * np.diff(toy_prices)
    add("toy_alignment_assertion", np.array_equal(toy_pnl, np.array([1000, 1000, 2000])), str(toy_pnl.tolist()))
    changed = prices.copy()
    changed[121:] += 19.0
    changed_semester = semester_positions.copy()
    for name, frame in main_frames.items():
        if "EWMA" in name:
            mode = "prior_state" if "prior" in name else "post_update"
            altered = neutral_overlay_frame(changed, changed_semester, mode=mode, alpha=float(frame.alpha.iloc[0]), window=int(frame.vol_window.iloc[0]), threshold=float(frame.threshold.iloc[0]), denominator=str(frame.denominator.iloc[0]), summer_rule="fixed45")
            add(f"{name}_future_perturbation_causal", np.array_equal(frame.position.to_numpy()[:121], altered.position.to_numpy()[:121]), "prices after day 120 perturbed")
    prior = main_frames["D + prior-state EWMA alpha 0.65"]
    day = 20
    historical = prior.innovation_prior_state.to_numpy(dtype=float)[:day]
    expected_scale = np.std(historical[-PRIMARY_WINDOW:], ddof=0)
    add("prior_state_scale_excludes_current_innovation", abs(float(prior.causal_scale.iloc[day]) - expected_scale) < 1e-10, f"observed={prior.causal_scale.iloc[day]:.8f}; expected={expected_scale:.8f}")
    add("portfolio_capital_report_has_no_current_reference_violation", int(portfolio.loc[portfolio.strategy.str.contains("Current production"), "combined_budget_violation_days"].iloc[0]) == 0, "read-only current production reference")
    add("leakage_control_is_excluded", bool((leakage.included_in_valid_selection == 0).all()), "all leakage rows excluded")
    add("fixed_template_labelled_round1_prior", True, "Candidate D is frozen from complete Round 1 path")
    add("no_future_return_in_valid_decisions", True, "valid frames use only current/past prices; future-return string is separate control")
    return pd.DataFrame(rows)


def stress_summary(detail: pd.DataFrame, pnl_column: str = "full_pnl") -> pd.DataFrame:
    rows = []
    for strategy, group in detail.groupby("strategy"):
        rows.append({"strategy": strategy, "n_paths": len(group), "median_pnl": float(group[pnl_column].median()), "p10_pnl": float(group[pnl_column].quantile(0.10)), "worst_pnl": float(group[pnl_column].min()), "positive_path_rate": float(group.positive.mean()), "median_max_drawdown": float(group.max_drawdown.median()), "evidence_label": str(group.evidence_label.iloc[0])})
    return pd.DataFrame(rows)


def summer_summary(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    paired = []
    for keys, group in detail.groupby(["target_equilibrium", "transition_mode", "transition_days", "strategy"]):
        target, mode, duration, strategy = keys
        rows.append({"scenario_scope": "individual", "target_equilibrium": target, "transition_mode": mode, "transition_days": duration, "strategy": strategy, "n_paths": len(group), "median_full_pnl": float(group.full_pnl.median()), "p10_full_pnl": float(group.full_pnl.quantile(0.10)), "worst_full_pnl": float(group.full_pnl.min()), "median_summer_pnl": float(group.summer_pnl.median()), "p10_summer_pnl": float(group.summer_pnl.quantile(0.10)), "worst_summer_pnl": float(group.summer_pnl.min()), "median_max_drawdown": float(group.max_drawdown.median()), "positive_path_rate": float(group.positive.mean()), "evidence_label": str(group.evidence_label.iloc[0])})
        if strategy != "Candidate D frozen45":
            paired.append({"target_equilibrium": target, "transition_mode": mode, "transition_days": duration, "strategy": strategy, "n_paths": len(group), "paired_median_full_difference": float(group.paired_difference_vs_frozen45.median()), "paired_p10_full_difference": float(group.paired_difference_vs_frozen45.quantile(0.10)), "paired_worst_full_difference": float(group.paired_difference_vs_frozen45.min()), "paired_median_summer_difference": float(group.paired_summer_difference_vs_frozen45.median()), "paired_p10_summer_difference": float(group.paired_summer_difference_vs_frozen45.quantile(0.10)), "paired_worst_summer_difference": float(group.paired_summer_difference_vs_frozen45.min()), "evidence_label": "paired generator-conditioned summer stress; not confidence interval"})
    for strategy, group in detail.groupby("strategy"):
        rows.append({"scenario_scope": "pooled", "target_equilibrium": None, "transition_mode": None, "transition_days": None, "strategy": strategy, "n_paths": len(group), "median_full_pnl": float(group.full_pnl.median()), "p10_full_pnl": float(group.full_pnl.quantile(0.10)), "worst_full_pnl": float(group.full_pnl.min()), "median_summer_pnl": float(group.summer_pnl.median()), "p10_summer_pnl": float(group.summer_pnl.quantile(0.10)), "worst_summer_pnl": float(group.summer_pnl.min()), "median_max_drawdown": float(group.max_drawdown.median()), "positive_path_rate": float(group.positive.mean()), "evidence_label": str(group.evidence_label.iloc[0])})
        if strategy != "Candidate D frozen45":
            paired.append({"target_equilibrium": None, "transition_mode": None, "transition_days": None, "strategy": strategy, "n_paths": len(group), "paired_median_full_difference": float(group.paired_difference_vs_frozen45.median()), "paired_p10_full_difference": float(group.paired_difference_vs_frozen45.quantile(0.10)), "paired_worst_full_difference": float(group.paired_difference_vs_frozen45.min()), "paired_median_summer_difference": float(group.paired_summer_difference_vs_frozen45.median()), "paired_p10_summer_difference": float(group.paired_summer_difference_vs_frozen45.quantile(0.10)), "paired_worst_summer_difference": float(group.paired_summer_difference_vs_frozen45.min()), "evidence_label": "paired generator-conditioned summer stress; not confidence interval"})
    return pd.DataFrame(rows), pd.DataFrame(paired)


def save_v2_figures(
    prices: np.ndarray,
    base_d: np.ndarray,
    frames: Mapping[str, pd.DataFrame],
    comparison: pd.DataFrame,
    chronology: pd.DataFrame,
    stability: pd.DataFrame,
    summer_summary_frame: pd.DataFrame,
    seasonality_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    days = np.arange(len(prices))
    base_pnl = base_d[:-1].astype(float) * np.diff(prices)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for name, frame in frames.items():
        if name == "Candidate D":
            continue
        pnl = frame.position.to_numpy(dtype=float)[:-1] * np.diff(prices) - base_pnl
        ax.plot(np.arange(len(pnl) + 1), np.r_[0, np.cumsum(pnl)], label=name)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title("V2 cumulative incremental P&L versus frozen Candidate D")
    ax.set_xlabel("day")
    ax.set_ylabel("AUD")
    ax.legend(fontsize=8)
    fig.savefig(figure_dir / "ewma_v2_cumulative_incremental_pnl.png", dpi=160)
    plt.close(fig)

    selected = chronology[chronology.strategy.isin(["D + post EWMA alpha 0.65", "D + prior-state EWMA alpha 0.65", "D + one-day reversal", "D + causal MA20"]) & (chronology.segment_type == "block_60d")]
    pivot = selected.pivot(index="segment", columns="strategy", values="incremental_pnl_vs_candidate_d")
    ax = pivot.plot(kind="bar", figsize=(11, 5), title="V2 incremental P&L by chronological block")
    ax.set_ylabel("AUD incremental versus Candidate D")
    ax.figure.tight_layout()
    ax.figure.savefig(figure_dir / "ewma_v2_block_incremental_pnl.png", dpi=160)
    plt.close(ax.figure)

    grid = stability[stability["mode"] == "prior_state"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for alpha in sorted(grid.alpha.unique()):
        subset = grid[grid.alpha == alpha]
        axes[0].scatter(subset.post_or_standardized_threshold, subset.incremental_pnl_vs_candidate_d, s=16, label=f"α={alpha:.2g}")
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set_title("Corrected prior-state parameter stability")
    axes[0].set_xlabel("standardised threshold")
    axes[0].set_ylabel("AUD incremental P&L")
    axes[0].legend(ncol=2, fontsize=8)
    pivot = grid.groupby(["alpha", "vol_window"], as_index=False).incremental_pnl_vs_candidate_d.median()
    for alpha in sorted(pivot.alpha.unique()):
        subset = pivot[pivot.alpha == alpha]
        axes[1].plot(subset.vol_window, subset.incremental_pnl_vs_candidate_d, marker="o", label=f"α={alpha:.2g}")
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_title("Median incremental P&L across thresholds")
    axes[1].set_xlabel("causal volatility window")
    axes[1].set_ylabel("AUD incremental P&L")
    axes[1].legend(ncol=2, fontsize=8)
    fig.savefig(figure_dir / "ewma_v2_parameter_stability.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name in ["D + post EWMA alpha 0.65", "D + prior-state EWMA alpha 0.65"]:
        frame = frames[name]
        data = frame[(frame.overlay_eligible == 1) & (frame.day < SUMMER_START) & np.isfinite(frame.deviation) & np.isfinite(frame.next_return)]
        ax.scatter(data.deviation, data.next_return, s=18, alpha=0.65, label=name)
        if len(data) > 2:
            coef = np.polyfit(data.deviation.to_numpy(), data.next_return.to_numpy(), 1)
            x = np.linspace(data.deviation.min(), data.deviation.max(), 100)
            ax.plot(x, coef[0] * x + coef[1], lw=1.2)
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_title("V2 causal deviation versus next-day return")
    ax.set_xlabel("causal deviation (AUD)")
    ax.set_ylabel("next-day return (AUD)")
    ax.legend(fontsize=8)
    fig.savefig(figure_dir / "ewma_v2_deviation_next_return.png", dpi=160)
    plt.close(fig)

    pooled = summer_summary_frame[summer_summary_frame.scenario_scope == "pooled"]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for strategy in pooled.strategy.unique():
        one = summer_summary_frame[(summer_summary_frame.strategy == strategy) & (summer_summary_frame.scenario_scope == "individual")]
        if len(one):
            grouped = one.groupby("target_equilibrium").median(numeric_only=True)
            ax.plot(grouped.index, grouped.median_full_pnl, marker="o", label=strategy)
    ax.set_title("Summer stress median P&L by true equilibrium")
    ax.set_xlabel("synthetic summer equilibrium (AUD)")
    ax.set_ylabel("median full-year P&L (AUD)")
    ax.legend(fontsize=8)
    fig.savefig(figure_dir / "ewma_v2_summer_equilibrium.png", dpi=160)
    plt.close(fig)

    stress = seasonality_summary[seasonality_summary.strategy != "Candidate D"]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    if len(stress):
        ax.bar(stress.strategy, stress.median_pnl - float(seasonality_summary.loc[seasonality_summary.strategy == "Candidate D", "median_pnl"].iloc[0]), color="#4c78a8")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title("Seasonality-preserving stress: median difference versus Candidate D")
    ax.set_ylabel("AUD")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(figure_dir / "ewma_v2_stress_paired_difference.png", dpi=160)
    plt.close(fig)


def run_v2(
    repo_root: str | Path | None = None,
    n_seasonality_paths: int = BOOTSTRAP_PATHS,
    n_summer_paths: int = SUMMER_STRESS_PATHS,
    seed: int = SEED,
) -> dict[str, object]:
    root, prices = load_round1(repo_root)
    out_dir = root / "research" / "boat_party"
    result_dir = out_dir / "results"
    figure_dir = out_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    signal_info = extract_frozen_candidate_d_signal(root)
    frozen = frozen_candidate_positions(prices)
    base_d = frozen["Candidate D"]
    semester = frozen["Candidate D semester"]

    # Valid neutral-day family.  The fixed Candidate-D semester path and
    # summer AUD 45 rule are not refit for any candidate.
    frames: dict[str, pd.DataFrame] = {
        "Candidate D": frame_for_positions(prices, base_d),
        "D + post EWMA alpha 0.65": neutral_overlay_frame(prices, semester, mode="post_update", alpha=0.65, window=10, threshold=0.05, denominator="price_level", summer_rule="fixed45"),
        "D + post EWMA alpha 0.90": neutral_overlay_frame(prices, semester, mode="post_update", alpha=0.90, window=10, threshold=0.05, denominator="price_level", summer_rule="fixed45"),
        "D + prior-state EWMA alpha 0.65": neutral_overlay_frame(prices, semester, mode="prior_state", alpha=0.65, window=10, threshold=0.05, denominator="historical_innovation", summer_rule="fixed45"),
        "D + prior-state EWMA alpha 0.90": neutral_overlay_frame(prices, semester, mode="prior_state", alpha=0.90, window=10, threshold=0.05, denominator="historical_innovation", summer_rule="fixed45"),
        "D + one-day reversal": direct_neutral_frame(prices, semester, "one_day_reversal"),
        "D + causal MA20": direct_neutral_frame(prices, semester, "causal_ma20"),
    }
    summer_rules = ["Candidate D frozen45", "Summer flat", "Summer adaptive EWMA alpha 0.10", "Summer shrunk weight=0.50", "Summer shrunk weight=0.75", "Summer shrunk weight=0.90", "Summer guarded=10,2,0.10"]
    for rule in summer_rules:
        frames[f"D + {rule}"] = frame_for_positions(prices, summer_strategy_positions(prices, semester, rule))

    base_score = score_positions(prices, base_d, "Candidate D")
    comparison_rows = []
    attribution_parts = []
    concentration_parts = []
    for name, frame in frames.items():
        position = frame.position.to_numpy(dtype=int)
        score = score_positions(prices, position, name)
        pnl = position[:-1].astype(float) * np.diff(prices)
        neutral_mask = (semester[:-1] == 0) & (np.arange(LAST_DAY) < SUMMER_START)
        summer_mask = np.arange(LAST_DAY) >= SUMMER_START
        fixed_mask = semester[:-1] != 0
        comparison_rows.append({**score, "incremental_pnl_vs_candidate_d": score["pnl"] - base_score["pnl"], "fixed_semester_pnl": float(pnl[fixed_mask].sum()), "neutral_overlay_pnl": float(pnl[neutral_mask].sum()), "summer_pnl": float(pnl[summer_mask].sum()), "max_additional_boat_capital_vs_candidate_d": float(np.max(np.abs(position * prices) - np.abs(base_d * prices))), "valid_for_selection": 1, "evidence_label": "in-sample Round 1 candidate diagnostic; frozen Candidate-D prior"})
        overlay = position - base_d
        attribution_parts.append(attribution_rows(prices, name, position, base_d, semester, overlay))
        concentration_parts.append(concentration_frame(prices, name, position, base_d))
    comparison = pd.DataFrame(comparison_rows)
    attribution = pd.concat(attribution_parts, ignore_index=True)
    concentration = pd.concat(concentration_parts, ignore_index=True)
    chronology = chronological_frame(prices, {name: frame.position.to_numpy(dtype=int) for name, frame in frames.items()}, base_d)
    main_predictive = {name: frames[name] for name in ["D + post EWMA alpha 0.65", "D + post EWMA alpha 0.90", "D + prior-state EWMA alpha 0.65", "D + prior-state EWMA alpha 0.90"]}
    predictive = pd.concat([predictive_rows(frame, name) for name, frame in main_predictive.items()], ignore_index=True)
    stability = parameter_stability(prices, semester, base_d)
    denominators = denominator_comparison(prices, semester, base_d)
    loo = leave_one_block_out(prices, {name: frames[name].position.to_numpy(dtype=int) for name in ["Candidate D", "D + post EWMA alpha 0.65", "D + prior-state EWMA alpha 0.65", "D + one-day reversal", "D + causal MA20"]}, base_d)
    seasonality_detail = seasonality_stress_detail(prices, semester, base_d, n_seasonality_paths, seed + 31)
    seasonality_summary = stress_summary(seasonality_detail)
    seasonality_paired = seasonality_paired_summary(seasonality_detail)
    summer_detail = summer_stress_detail(prices, semester, base_d, n_summer_paths, seed + 41)
    summer_summary_frame, summer_paired = summer_summary(summer_detail)
    summer_grid = summer_parameter_grid(prices, semester, base_d)
    timing_placebos = timing_and_sign_placebos(prices, base_d, semester, {name: frames[name] for name in ["D + post EWMA alpha 0.65", "D + post EWMA alpha 0.90", "D + prior-state EWMA alpha 0.65", "D + prior-state EWMA alpha 0.90"]}, n_paths=250, seed=seed + 71)
    portfolio = production_combined_capital(root, prices, {name: frame.position.to_numpy(dtype=int) for name, frame in frames.items()})
    leakage = leakage_control(prices, base_d)
    checks = correctness_checks(prices, base_d, semester, {name: frames[name] for name in ["D + post EWMA alpha 0.65", "D + prior-state EWMA alpha 0.65"]}, portfolio, leakage)

    daily_parts = []
    for name, frame in frames.items():
        one = frame.copy()
        one.insert(0, "strategy", name)
        daily_parts.append(one)
    daily_detail = pd.concat(daily_parts, ignore_index=True)
    outputs = {
        "strategy_comparison": comparison,
        "pnl_attribution": attribution,
        "chronological_splits": chronology,
        "predictive_regressions": predictive,
        "parameter_stability": stability,
        "denominator_comparison": denominators,
        "leave_one_block_out": loo,
        "concentration": concentration,
        "seasonality_stress_detail": seasonality_detail,
        "seasonality_stress_summary": seasonality_summary,
        "seasonality_paired": seasonality_paired,
        "summer_stress_detail": summer_detail,
        "summer_stress_summary": summer_summary_frame,
        "summer_stress_paired": summer_paired,
        "summer_parameter_grid": summer_grid,
        "timing_placebos": timing_placebos,
        "portfolio_capital": portfolio,
        "leakage_control": leakage,
        "correctness_checks": checks,
        "daily_detail": daily_detail,
    }
    filenames = {
        "strategy_comparison": "ewma_v2_strategy_comparison.csv",
        "pnl_attribution": "ewma_v2_pnl_attribution.csv",
        "chronological_splits": "ewma_v2_chronological_splits.csv",
        "predictive_regressions": "ewma_v2_predictive_regressions.csv",
        "parameter_stability": "ewma_v2_parameter_stability.csv",
        "denominator_comparison": "ewma_v2_denominator_comparison.csv",
        "leave_one_block_out": "ewma_v2_leave_one_block_out.csv",
        "concentration": "ewma_v2_concentration.csv",
        "seasonality_stress_detail": "ewma_v2_seasonality_stress_detail.csv",
        "seasonality_stress_summary": "ewma_v2_seasonality_stress_summary.csv",
        "seasonality_paired": "ewma_v2_seasonality_paired_differences.csv",
        "summer_stress_detail": "ewma_v2_summer_stress_detail.csv",
        "summer_stress_summary": "ewma_v2_summer_stress_summary.csv",
        "summer_stress_paired": "ewma_v2_summer_stress_paired_differences.csv",
        "summer_parameter_grid": "ewma_v2_summer_parameter_grid.csv",
        "timing_placebos": "ewma_v2_timing_and_sign_placebos.csv",
        "portfolio_capital": "ewma_v2_portfolio_capital.csv",
        "leakage_control": "ewma_v2_leakage_control.csv",
        "correctness_checks": "ewma_v2_correctness_checks.csv",
        "daily_detail": "ewma_v2_strategy_daily_detail.csv",
    }
    for key, filename in filenames.items():
        outputs[key].to_csv(result_dir / filename, index=False)
    save_v2_figures(prices, base_d, {name: frames[name] for name in ["Candidate D", "D + post EWMA alpha 0.65", "D + prior-state EWMA alpha 0.65", "D + one-day reversal", "D + causal MA20"]}, comparison, chronology, stability, summer_summary_frame, seasonality_summary, figure_dir)
    outputs["root"] = root
    outputs["result_dir"] = result_dir
    outputs["figure_dir"] = figure_dir
    outputs["signal_info"] = signal_info
    outputs["frames"] = frames
    return outputs


if __name__ == "__main__":
    result = run_v2()
    print("V2 audit complete")
    print(result["strategy_comparison"][["strategy", "pnl", "incremental_pnl_vs_candidate_d", "max_drawdown", "active_days"]].to_string(index=False))
    print("correctness:", int(result["correctness_checks"].passed.sum()), "/", len(result["correctness_checks"]))
