"""Focused, research-only Savitzky-Golay Boat Party regime comparison.

This does not modify or import the production algorithm.  It reconstructs the
frozen Candidate D signal read-only, builds polynomial-smoothed seasonal
signals from Round 1, and evaluates a causal neutral-day overlay with the
fixed AUD 45 summer rule.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "trader_interface" / "data" / "Boat Party Ticket_price_history.csv"
ALGORITHM = ROOT / "trader_interface" / "algorithm.py"
RESULTS = Path(__file__).resolve().parent / "results"

LIMIT = 1_000
SUMMER_START = 322
SUMMER_FAIR_VALUE = 45.0
LAST_DAY = 364

TEAMMATE_SIGNAL = (
    "0000000+++++++++++++++000000000+++++++++0000000000000------------------"
    "000000000-------000000000000000++++++++++000000----------------0000000+"
    "+++++000----------00000++++++++++++++++++++++++++++000000000-----------"
    "-------000000000----00000+++++++++++000000000------------------"
    "0000000+++++000-----------00000++++++++++++0000000000000000000000000000"
    "000000000000000000"
)


def load_prices() -> np.ndarray:
    return pd.read_csv(DATA)["Price"].to_numpy(dtype=float)


def extract_candidate_d_signal() -> str:
    tree = ast.parse(ALGORITHM.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "BOAT_PARTY_SEMESTER_SIGNALS" in names:
                return str(ast.literal_eval(node.value))
    raise RuntimeError("Candidate D signal not found")


def local_polynomial_smooth(prices: np.ndarray, window: int, order: int) -> np.ndarray:
    """Savitzky-Golay equivalent using local least-squares polynomials.

    Interior points use centred windows.  Edge points use the first/last full
    window and evaluate its fitted polynomial at the requested point, matching
    the usual interpolation-style edge treatment.
    """

    if window % 2 != 1 or window <= order or window > len(prices):
        raise ValueError("window must be odd, greater than order, and fit the data")
    half = window // 2
    result = np.empty_like(prices, dtype=float)
    for day in range(len(prices)):
        start = min(max(day - half, 0), len(prices) - window)
        indices = np.arange(start, start + window)
        x = indices.astype(float) - float(day)
        design = np.vander(x, N=order + 1, increasing=True)
        coefficients, *_ = np.linalg.lstsq(design, prices[indices], rcond=None)
        result[day] = coefficients[0]
    return result


def seasonal_signal(prices: np.ndarray, window: int, order: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    trend = local_polynomial_smooth(prices, window, order)
    # Match the teammate implementation's appended wraparound difference when
    # calculating the slope scale, while leaving the terminal trading signal flat.
    slopes_for_scale = np.diff(trend, append=trend[0])
    scale = max(float(np.std(slopes_for_scale, ddof=0)), 1e-12)
    forward_slope = np.r_[np.diff(trend), 0.0]
    scores = forward_slope / scale
    signal = np.zeros(len(prices), dtype=int)
    signal[scores >= threshold] = 1
    signal[scores <= -threshold] = -1
    signal[LAST_DAY] = 0
    return signal, scores


def positions_from_string(signal: str, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=int)
    for day, char in enumerate(signal[:length]):
        out[day] = 1 if char == "+" else -1 if char == "-" else 0
    return out


def fixed_45_summer(prices: np.ndarray) -> np.ndarray:
    out = np.zeros(len(prices), dtype=int)
    out[SUMMER_START:LAST_DAY] = np.where(
        prices[SUMMER_START:LAST_DAY] < SUMMER_FAIR_VALUE,
        1,
        np.where(prices[SUMMER_START:LAST_DAY] > SUMMER_FAIR_VALUE, -1, 0),
    )
    return out


def candidate_d_positions(prices: np.ndarray) -> np.ndarray:
    signal = positions_from_string(extract_candidate_d_signal(), len(prices))
    signal[SUMMER_START:] = fixed_45_summer(prices)[SUMMER_START:]
    signal[LAST_DAY] = 0
    return LIMIT * signal


def hybrid_positions(
    prices: np.ndarray,
    signal: np.ndarray,
    overlay: str,
    alpha: float = 0.65,
    vol_window: int = 7,
    revert_threshold: float = 0.01,
) -> np.ndarray:
    """Combine fixed semester regimes, a causal neutral overlay, and summer 45."""

    positions = np.zeros(len(prices), dtype=int)
    fair_value = float(prices[0])
    current_position = 0
    summer = fixed_45_summer(prices)
    for day, price in enumerate(prices):
        if day > 0:
            fair_value += alpha * (float(price) - fair_value)
        if day == LAST_DAY:
            desired = 0
        elif day >= SUMMER_START:
            desired = int(summer[day] * LIMIT)
        elif signal[day] != 0:
            desired = int(signal[day] * LIMIT)
        elif overlay == "flat" or day < vol_window:
            desired = 0
        elif overlay == "one_day_reversal":
            change = float(prices[day] - prices[day - 1])
            desired = int(-np.sign(change) * LIMIT)
        elif overlay == "ewma":
            window = prices[day - vol_window + 1 : day + 1]
            scale = max(float(np.std(window, ddof=0)), 1e-12)
            z_score = (float(price) - fair_value) / scale
            if current_position == 0:
                desired = -LIMIT if z_score >= revert_threshold else LIMIT if z_score <= -revert_threshold else 0
            elif current_position * z_score > 0:
                desired = -current_position if abs(z_score) >= revert_threshold else 0
            else:
                desired = current_position
        else:
            raise ValueError(overlay)
        positions[day] = desired
        current_position = desired
    return positions


def score(prices: np.ndarray, positions: np.ndarray) -> dict[str, float | int]:
    daily = positions[:-1].astype(float) * np.diff(prices)
    curve = np.cumsum(daily)
    running_max = np.maximum.accumulate(np.r_[0.0, curve])
    drawdowns = np.r_[0.0, curve] - running_max
    active = positions[:-1] != 0
    return {
        "pnl": float(np.sum(daily)),
        "max_drawdown": float(np.min(drawdowns)),
        "active_days": int(np.sum(active)),
        "active_hit_rate": float(np.mean(daily[active] > 0)) if np.any(active) else np.nan,
        "max_notional": float(np.max(np.abs(positions * prices))),
    }


def attribution(prices: np.ndarray, positions: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    daily = positions[:-1].astype(float) * np.diff(prices)
    days = np.arange(len(daily))
    fixed = (days < SUMMER_START) & (signal[:-1] != 0)
    neutral = (days < SUMMER_START) & (signal[:-1] == 0)
    summer = days >= SUMMER_START
    return {
        "fixed_semester_pnl": float(np.sum(daily[fixed])),
        "neutral_semester_pnl": float(np.sum(daily[neutral])),
        "summer_pnl": float(np.sum(daily[summer])),
    }


def concentration(prices: np.ndarray, positions: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    daily = positions[:-1].astype(float) * np.diff(prices)
    mask = (np.arange(len(daily)) < SUMMER_START) & (signal[:-1] == 0)
    overlay = daily[mask]
    total = float(np.sum(overlay))
    ordered = np.sort(overlay)[::-1]
    return {
        "neutral_observations": int(len(overlay)),
        "best_1_share": float(np.sum(ordered[:1]) / total) if total else np.nan,
        "best_5_share": float(np.sum(ordered[:5]) / total) if total else np.nan,
        "best_10_share": float(np.sum(ordered[:10]) / total) if total else np.nan,
    }


def block_bootstrap(values: np.ndarray, length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    output: list[float] = []
    while len(output) < length:
        start = int(rng.integers(0, len(values) - block + 1))
        output.extend(values[start : start + block].tolist())
    return np.asarray(output[:length], dtype=float)


def seasonality_stress(prices: np.ndarray, teammate_signal: np.ndarray, paths: int = 800) -> pd.DataFrame:
    """Generator-conditioned stress retaining a broad annual price shape."""

    rng = np.random.default_rng(20260809)
    broad = local_polynomial_smooth(prices, 41, 3)
    residual = prices - broad
    residual -= float(np.mean(residual))
    rows: list[dict[str, object]] = []
    for path_id in range(paths):
        amplitude = float(rng.uniform(0.6, 1.4))
        volatility_scale = float(rng.choice((0.5, 0.75, 1.0, 1.25, 1.5)))
        noise = block_bootstrap(residual, len(prices), 7, rng) * volatility_scale
        path = SUMMER_FAIR_VALUE + amplitude * (broad - SUMMER_FAIR_VALUE) + noise
        strategies = {
            "Candidate D": candidate_d_positions(path),
            "SavGol gate + flat": hybrid_positions(path, teammate_signal, "flat"),
            "SavGol gate + one-day reversal": hybrid_positions(path, teammate_signal, "one_day_reversal"),
            "SavGol gate + EWMA": hybrid_positions(path, teammate_signal, "ewma"),
        }
        for name, positions in strategies.items():
            result = score(path, positions)
            rows.append({
                "path_id": path_id,
                "strategy": name,
                "amplitude": amplitude,
                "volatility_scale": volatility_scale,
                **result,
                "evidence_label": "seasonality-preserving generator-conditioned stress; not a confidence interval",
            })
    return pd.DataFrame(rows)


def residual_regime_stress(prices: np.ndarray, signal: np.ndarray, paths_per_phi: int = 500) -> pd.DataFrame:
    """Adversarial check when Round 2 residual autocorrelation changes."""

    rng = np.random.default_rng(77)
    broad = local_polynomial_smooth(prices, 41, 3)
    residual_scale = float(np.std(prices - broad, ddof=0))
    rows: list[dict[str, object]] = []
    for phi in (-0.6, -0.3, 0.0, 0.3, 0.6, 0.8, 0.9):
        innovation_scale = residual_scale * np.sqrt(max(1.0 - phi**2, 0.01))
        for path_id in range(paths_per_phi):
            residual = np.zeros(len(prices), dtype=float)
            shocks = rng.normal(0.0, innovation_scale, len(prices))
            for day in range(1, len(prices)):
                residual[day] = phi * residual[day - 1] + shocks[day]
            amplitude = float(rng.uniform(0.7, 1.3))
            path = SUMMER_FAIR_VALUE + amplitude * (broad - SUMMER_FAIR_VALUE) + residual
            d_pnl = float(score(path, candidate_d_positions(path))["pnl"])
            hybrid_pnl = float(score(path, hybrid_positions(path, signal, "ewma"))["pnl"])
            rows.append({
                "phi": phi,
                "path_id": path_id,
                "candidate_d_pnl": d_pnl,
                "hybrid_pnl": hybrid_pnl,
                "difference": hybrid_pnl - d_pnl,
                "evidence_label": "adversarial residual-regime generator stress; not a confidence interval",
            })
    return pd.DataFrame(rows)


def run() -> None:
    prices = load_prices()
    if len(prices) != 365 or len(TEAMMATE_SIGNAL) != 365:
        raise AssertionError("expected 365-day price and teammate signal paths")

    candidate_d = candidate_d_positions(prices)
    teammate_array = positions_from_string(TEAMMATE_SIGNAL, len(prices))
    # The teammate subsequently confirmed window 21 / polynomial order 2.
    # A standardized threshold near 0.625 is the closest reconstruction from
    # the available string, but the conversion rule is not yet known exactly.
    closest_signal, _ = seasonal_signal(prices, 21, 2, 0.625)
    closest_match = int(np.sum(closest_signal == teammate_array))

    rows: list[dict[str, object]] = []
    configurations = [("Candidate D", None, None, None, None, candidate_d)]
    for overlay in ("flat", "one_day_reversal", "ewma"):
        positions = hybrid_positions(prices, teammate_array, overlay)
        configurations.append((f"Supplied SavGol regime string + {overlay}", 21, 2, None, overlay, positions))
    for name, window, order, threshold, overlay, positions in configurations:
        base = score(prices, positions)
        attr = attribution(prices, positions, teammate_array if window else positions_from_string(extract_candidate_d_signal(), len(prices)))
        conc = concentration(prices, positions, teammate_array if window else positions_from_string(extract_candidate_d_signal(), len(prices)))
        rows.append({"strategy": name, "window": window, "order": order, "slope_threshold": threshold, "overlay": overlay, **base, **attr, **conc})

    grid_rows: list[dict[str, object]] = []
    for window in (15, 21, 31, 41, 61):
        for order in (2, 3, 4, 8):
            if order >= window:
                continue
            for threshold in (0.5, 0.75, 1.0, 1.25):
                signal, _ = seasonal_signal(prices, window, order, threshold)
                for overlay in ("flat", "one_day_reversal", "ewma"):
                    positions = hybrid_positions(prices, signal, overlay)
                    result = score(prices, positions)
                    attr = attribution(prices, positions, signal)
                    conc = concentration(prices, positions, signal)
                    grid_rows.append({
                        "window": window,
                        "order": order,
                        "slope_threshold": threshold,
                        "overlay": overlay,
                        "fixed_semester_days": int(np.sum(signal[:SUMMER_START] != 0)),
                        **result,
                        **attr,
                        **conc,
                    })

    RESULTS.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)
    grid = pd.DataFrame(grid_rows)
    comparison.to_csv(RESULTS / "savgol_regime_comparison.csv", index=False)
    grid.to_csv(RESULTS / "savgol_regime_parameter_grid.csv", index=False)

    robust = (
        grid.groupby(["overlay", "order"], as_index=False)
        .agg(
            configurations=("pnl", "size"),
            median_pnl=("pnl", "median"),
            p10_pnl=("pnl", lambda x: float(np.quantile(x, 0.10))),
            worst_pnl=("pnl", "min"),
            best_pnl=("pnl", "max"),
            median_drawdown=("max_drawdown", "median"),
            median_neutral_pnl=("neutral_semester_pnl", "median"),
        )
        .sort_values(["overlay", "order"])
    )
    robust.to_csv(RESULTS / "savgol_regime_robustness_summary.csv", index=False)

    # Sensitivity of the independently reproducible 21/2/0.5 candidate's
    # causal overlay. This is diagnostic; the reported baseline stays frozen.
    frozen_signal, _ = seasonal_signal(prices, 21, 2, 0.5)
    overlay_rows = []
    for alpha in (0.30, 0.50, 0.65, 0.80, 0.90):
        for vol_window in (5, 7, 10, 20):
            for revert_threshold in (0.01, 0.05, 0.10, 0.25, 0.50):
                positions = hybrid_positions(
                    prices,
                    frozen_signal,
                    "ewma",
                    alpha=alpha,
                    vol_window=vol_window,
                    revert_threshold=revert_threshold,
                )
                overlay_rows.append({
                    "alpha": alpha,
                    "vol_window": vol_window,
                    "revert_threshold": revert_threshold,
                    **score(prices, positions),
                })
    overlay_grid = pd.DataFrame(overlay_rows)
    overlay_grid.to_csv(RESULTS / "savgol_frozen_overlay_sensitivity.csv", index=False)

    stress = seasonality_stress(prices, teammate_array)
    stress.to_csv(RESULTS / "savgol_regime_stress_detail.csv", index=False)
    stress_summary = (
        stress.groupby("strategy", as_index=False)
        .agg(
            paths=("pnl", "size"),
            median_pnl=("pnl", "median"),
            p10_pnl=("pnl", lambda x: float(np.quantile(x, 0.10))),
            worst_pnl=("pnl", "min"),
            positive_rate=("pnl", lambda x: float(np.mean(x > 0))),
            median_drawdown=("max_drawdown", "median"),
        )
    )
    baseline = stress[stress.strategy == "Candidate D"].set_index("path_id")["pnl"]
    paired_rows = []
    for name, group in stress[stress.strategy != "Candidate D"].groupby("strategy"):
        differences = group.set_index("path_id")["pnl"] - baseline
        paired_rows.append({
            "strategy": name,
            "median_difference_vs_candidate_d": float(np.median(differences)),
            "p10_difference_vs_candidate_d": float(np.quantile(differences, 0.10)),
            "worst_difference_vs_candidate_d": float(np.min(differences)),
            "positive_difference_rate": float(np.mean(differences > 0)),
        })
    paired = pd.DataFrame(paired_rows)
    stress_summary.to_csv(RESULTS / "savgol_regime_stress_summary.csv", index=False)
    paired.to_csv(RESULTS / "savgol_regime_stress_paired.csv", index=False)

    residual_stress = residual_regime_stress(prices, frozen_signal)
    residual_stress.to_csv(RESULTS / "savgol_residual_regime_stress.csv", index=False)
    residual_summary = (
        residual_stress.groupby("phi", as_index=False)
        .agg(
            paths=("difference", "size"),
            candidate_d_median=("candidate_d_pnl", "median"),
            hybrid_median=("hybrid_pnl", "median"),
            median_difference=("difference", "median"),
            p10_difference=("difference", lambda x: float(np.quantile(x, 0.10))),
            worst_difference=("difference", "min"),
            hybrid_win_rate=("difference", lambda x: float(np.mean(x > 0))),
        )
    )
    residual_summary.to_csv(RESULTS / "savgol_residual_regime_stress_summary.csv", index=False)

    print(f"Closest 21/2 reconstruction match: {closest_match}/365")
    print(comparison.to_string(index=False))
    print("\nRobustness summary:\n", robust.to_string(index=False))
    best = grid.sort_values("pnl", ascending=False).head(10)
    print("\nTop diagnostic configurations (in-sample, not selections):\n", best.to_string(index=False))
    print("\nSeasonality-preserving stress:\n", stress_summary.to_string(index=False))
    print("\nPaired stress differences:\n", paired.to_string(index=False))
    print("\nAdversarial residual-regime stress:\n", residual_summary.to_string(index=False))


if __name__ == "__main__":
    run()
