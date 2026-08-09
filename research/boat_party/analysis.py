"""Research-only Boat Party Ticket analysis.

The module deliberately does not import or mutate trader_interface.algorithm.
It treats a decision at day t as earning price[t + 1] - price[t], matching the
competition simulator.  The fixed seasonal templates in this file are
retrospective Round 1 objects; they are useful as a proposed Round 2 prior,
but are labelled as in-sample wherever they are reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SEED = 20260808
POSITION_LIMIT = 1_000
TOTAL_BUDGET = 600_000.0
PRICE_FILE = Path("trader_interface/data/Boat Party Ticket_price_history.csv")


def find_repo_root(start: str | Path | None = None) -> Path:
    """Locate the repository without assuming the notebook's working folder."""

    origin = Path(start or Path.cwd()).resolve()
    candidates = [origin, *origin.parents]
    for candidate in candidates:
        if (candidate / PRICE_FILE).exists() and (candidate / "docs").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate repository from {origin}")


def load_prices(repo_root: str | Path | None = None) -> pd.DataFrame:
    root = find_repo_root(repo_root)
    frame = pd.read_csv(root / PRICE_FILE)
    if list(frame.columns) != ["Day", "Price"] or len(frame) != 365:
        raise ValueError("Boat Party history must contain 365 Day/Price rows")
    if not np.array_equal(frame["Day"].to_numpy(), np.arange(365)):
        raise ValueError("Boat Party Day column is not 0..364")
    return frame


# These dates are transcribed from the official UQ academic-calendar page.  The
# multiple in-semester exam dates are retained as text because they are not one
# continuous interval; landmark_days() uses the final-examination midpoints.
UQ_EVENTS = [
    # 2026, Semester 1
    (2026, "S1", "Orientation Week", "16-20 Feb 2026", "2026-02-16", "2026-02-20"),
    (2026, "S1", "Classes commence", "23 Feb 2026", "2026-02-23", "2026-02-23"),
    (2026, "S1", "In-semester examinations", "27-29 Mar; 17-19 Apr; 2 May 2026", "2026-03-27", "2026-05-02"),
    (2026, "S1", "In-semester break", "6-12 Apr 2026", "2026-04-06", "2026-04-12"),
    (2026, "S1", "Classes resume", "13 Apr 2026", "2026-04-13", "2026-04-13"),
    (2026, "S1", "Revision period", "1-5 Jun 2026", "2026-06-01", "2026-06-05"),
    (2026, "S1", "Final examination period", "6-20 Jun 2026", "2026-06-06", "2026-06-20"),
    (2026, "S1", "Semester ending", "20 Jun 2026", "2026-06-20", "2026-06-20"),
    # 2026, Semester 2
    (2026, "S2", "Orientation Week", "20-24 Jul 2026", "2026-07-20", "2026-07-24"),
    (2026, "S2", "Classes commence", "27 Jul 2026", "2026-07-27", "2026-07-27"),
    (2026, "S2", "In-semester examinations", "5 Sep; 11-13 Sep; 18-20 Sep 2026", "2026-09-05", "2026-09-20"),
    (2026, "S2", "In-semester break", "28 Sep-5 Oct 2026", "2026-09-28", "2026-10-05"),
    (2026, "S2", "Classes resume", "6 Oct 2026", "2026-10-06", "2026-10-06"),
    (2026, "S2", "Revision period", "2-6 Nov 2026", "2026-11-02", "2026-11-06"),
    (2026, "S2", "Final examination period", "7-21 Nov 2026", "2026-11-07", "2026-11-21"),
    (2026, "S2", "Semester ending", "21 Nov 2026", "2026-11-21", "2026-11-21"),
    (2026, "S2", "Summer break", "Starts 23 Nov 2026", "2026-11-23", "2026-11-23"),
    # 2027, Semester 1
    (2027, "S1", "Orientation Week", "15-19 Feb 2027", "2027-02-15", "2027-02-19"),
    (2027, "S1", "Classes commence", "22 Feb 2027", "2027-02-22", "2027-02-22"),
    (2027, "S1", "In-semester examinations", "3 Apr; 10 Apr; 17 Apr; 8 May 2027", "2027-04-03", "2027-05-08"),
    (2027, "S1", "In-semester break", "26 Mar-4 Apr 2027", "2027-03-26", "2027-04-04"),
    (2027, "S1", "Classes resume", "5 Apr 2027", "2027-04-05", "2027-04-05"),
    (2027, "S1", "Revision period", "31 May-4 Jun 2027", "2027-05-31", "2027-06-04"),
    (2027, "S1", "Final examination period", "5-19 Jun 2027", "2027-06-05", "2027-06-19"),
    (2027, "S1", "Semester ending", "19 Jun 2027", "2027-06-19", "2027-06-19"),
    # 2027, Semester 2
    (2027, "S2", "Orientation Week", "19-23 Jul 2027", "2027-07-19", "2027-07-23"),
    (2027, "S2", "Classes commence", "26 Jul 2027", "2027-07-26", "2027-07-26"),
    (2027, "S2", "In-semester examinations", "4 Sep; 11 Sep; 18 Sep 2027", "2027-09-04", "2027-09-18"),
    (2027, "S2", "In-semester break", "27 Sep-4 Oct 2027", "2027-09-27", "2027-10-04"),
    (2027, "S2", "Classes resume", "5 Oct 2027", "2027-10-05", "2027-10-05"),
    (2027, "S2", "Revision period", "1-5 Nov 2027", "2027-11-01", "2027-11-05"),
    (2027, "S2", "Final examination period", "6-20 Nov 2027", "2027-11-06", "2027-11-20"),
    (2027, "S2", "Semester ending", "20 Nov 2027", "2027-11-20", "2027-11-20"),
    (2027, "S2", "Summer break", "Starts 22 Nov 2027", "2027-11-22", "2027-11-22"),
]


def calendar_events_frame() -> pd.DataFrame:
    rows = []
    for year, semester, label, dates, start, end in UQ_EVENTS:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        jan1 = pd.Timestamp(f"{year}-01-01")
        rows.append(
            {
                "year": year,
                "semester": semester,
                "event": label,
                "dates": dates,
                "start_date": start,
                "end_date": end,
                "start_day": int((start_ts - jan1).days),
                "end_day": int((end_ts - jan1).days),
            }
        )
    return pd.DataFrame(rows)


def day_of_year(date: str, year: int) -> int:
    return int((pd.Timestamp(date) - pd.Timestamp(f"{year}-01-01")).days)


def landmark_days(year: int) -> dict[str, float]:
    """Comparable event landmarks for a piecewise-linear calendar warp."""

    if year == 2026:
        dates = {
            "start": "2026-01-01",
            "s1_orientation_mid": "2026-02-18",
            "s1_classes": "2026-02-23",
            "s1_break": "2026-04-06",
            "s1_resume": "2026-04-13",
            "s1_exam_mid": "2026-06-13",
            "s1_end": "2026-06-20",
            "s2_orientation_mid": "2026-07-22",
            "s2_classes": "2026-07-27",
            "s2_break": "2026-09-28",
            "s2_resume": "2026-10-06",
            "s2_exam_mid": "2026-11-14",
            "s2_end": "2026-11-21",
            "summer": "2026-11-23",
            "end": "2027-01-01",
        }
    elif year == 2027:
        dates = {
            "start": "2027-01-01",
            "s1_orientation_mid": "2027-02-17",
            "s1_classes": "2027-02-22",
            "s1_break": "2027-03-26",
            "s1_resume": "2027-04-05",
            "s1_exam_mid": "2027-06-12",
            "s1_end": "2027-06-19",
            "s2_orientation_mid": "2027-07-21",
            "s2_classes": "2027-07-26",
            "s2_break": "2027-09-27",
            "s2_resume": "2027-10-05",
            "s2_exam_mid": "2027-11-13",
            "s2_end": "2027-11-20",
            "summer": "2027-11-22",
            "end": "2028-01-01",
        }
    else:
        raise ValueError("Only 2026 and 2027 are encoded")
    jan1 = pd.Timestamp(f"{year}-01-01")
    return {name: float((pd.Timestamp(date) - jan1).days) for name, date in dates.items()}


def calendar_comparison() -> pd.DataFrame:
    a = landmark_days(2026)
    b = landmark_days(2027)
    keys = [k for k in a if k not in {"start", "end"}]
    return pd.DataFrame(
        {
            "landmark": keys,
            "day_2026": [a[k] for k in keys],
            "day_2027": [b[k] for k in keys],
            "2027_minus_2026": [b[k] - a[k] for k in keys],
        }
    )


def smooth_series(values: Sequence[float], window: int = 14) -> np.ndarray:
    """Centered edge-padded moving average used only for research templates."""

    x = np.asarray(values, dtype=float)
    if window < 1:
        raise ValueError("window must be positive")
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(x, (half, half), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def local_turning_points(values: Sequence[float], window: int = 14) -> pd.DataFrame:
    """Return broad extrema from a smoothed price curve."""

    x = np.asarray(values, dtype=float)
    s = smooth_series(x, window)
    candidates: list[tuple[int, str, float]] = []
    for i in range(1, len(s) - 1):
        if s[i] > s[i - 1] and s[i] >= s[i + 1]:
            candidates.append((i, "peak", s[i]))
        elif s[i] < s[i - 1] and s[i] <= s[i + 1]:
            candidates.append((i, "trough", s[i]))

    # Keep extrema separated so daily noise does not masquerade as a season.
    chosen: list[tuple[int, str, float]] = []
    for candidate in candidates:
        if not chosen or candidate[0] - chosen[-1][0] >= 8:
            chosen.append(candidate)
        elif candidate[1] == chosen[-1][1]:
            if (candidate[1] == "peak" and candidate[2] > chosen[-1][2]) or (
                candidate[1] == "trough" and candidate[2] < chosen[-1][2]
            ):
                chosen[-1] = candidate
    return pd.DataFrame(
        [
            {
                "day": day,
                "kind": kind,
                "price": float(x[day]),
                "smoothed_price": float(level),
                "date_if_2026_day0": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(day))).date().isoformat(),
            }
            for day, kind, level in chosen
        ]
    )


def phase_labels(n: int = 365, year: int = 2026) -> np.ndarray:
    """Broad calendar phases; used for diagnostics, not as a fitted signal."""

    if year != 2026:
        raise ValueError("phase_labels currently uses the Round 1/2026 phases")
    boundaries = [0, 46, 51, 95, 102, 148, 171, 200, 205, 270, 278, 303, 325, n]
    labels = [
        "summer_pre_s1",
        "s1_orientation",
        "s1_classes_pre_break",
        "s1_break",
        "s1_classes_post_break",
        "s1_revision_exam",
        "winter_break",
        "s2_orientation",
        "s2_classes_pre_break",
        "s2_break",
        "s2_classes_post_break",
        "s2_revision_exam",
        "summer_post_s2",
    ]
    out = np.full(n, "unlabelled", dtype=object)
    for start, end, label in zip(boundaries[:-1], boundaries[1:], labels):
        out[start:min(end, n)] = label
    return out


BASE_BOUNDARIES = np.array([0, 15, 54, 92, 115, 161, 196, 235, 252, 302, 365], dtype=float)
BASE_SIGNS = np.array([0, 1, -1, 1, -1, 1, -1, 1, -1, 0], dtype=int)


def positions_from_boundaries(
    boundaries: Sequence[float],
    signs: Sequence[int] = BASE_SIGNS,
    limit: int = POSITION_LIMIT,
    boundary_buffer: int = 0,
    reduced_boundary: bool = False,
) -> np.ndarray:
    b = np.rint(np.asarray(boundaries, dtype=float)).astype(int)
    if len(b) != len(signs) + 1 or np.any(np.diff(b) <= 0):
        raise ValueError("boundaries must be strictly increasing and match signs")
    out = np.zeros(b[-1], dtype=int)
    for i, sign in enumerate(signs):
        out[b[i] : b[i + 1]] = int(sign * limit)
    if boundary_buffer:
        for boundary in b[1:-1]:
            lo = max(0, boundary - boundary_buffer)
            hi = min(len(out), boundary + boundary_buffer + 1)
            if reduced_boundary:
                out[lo:hi] = np.rint(out[lo:hi] * 0.5).astype(int)
            else:
                out[lo:hi] = 0
    return out


def fixed_schedule_positions(
    shift: int = 0,
    boundary_buffer: int = 0,
    reduced_boundary: bool = False,
    local_easter_shift: int = 0,
) -> np.ndarray:
    b = BASE_BOUNDARIES.astype(float).copy()
    b[1:-1] += shift
    # The middle S1 boundaries bracket the Easter/in-semester segment.
    b[3:6] += local_easter_shift
    b[0], b[-1] = 0, 365
    b = np.maximum.accumulate(b)
    for i in range(1, len(b)):
        b[i] = max(b[i], b[i - 1] + 1)
    b[-1] = 365
    return positions_from_boundaries(b, boundary_buffer=boundary_buffer, reduced_boundary=reduced_boundary)


def calendar_warp_indices(source_year: int = 2026, target_year: int = 2027, n: int = 365) -> np.ndarray:
    source = landmark_days(source_year)
    target = landmark_days(target_year)
    names = [
        "start",
        "s1_orientation_mid",
        "s1_classes",
        "s1_break",
        "s1_resume",
        "s1_exam_mid",
        "s1_end",
        "s2_orientation_mid",
        "s2_classes",
        "s2_break",
        "s2_resume",
        "s2_exam_mid",
        "s2_end",
        "summer",
        "end",
    ]
    target_days = np.array([target[k] for k in names], dtype=float)
    source_days = np.array([source[k] for k in names], dtype=float)
    return np.interp(np.arange(n, dtype=float), target_days, source_days)


def map_boundaries_to_calendar(target_year: int = 2027) -> np.ndarray:
    source = landmark_days(2026)
    target = landmark_days(target_year)
    keys = [
        "start",
        "s1_orientation_mid",
        "s1_classes",
        "s1_break",
        "s1_resume",
        "s1_exam_mid",
        "s1_end",
        "s2_orientation_mid",
        "s2_classes",
        "s2_break",
        "s2_resume",
        "s2_exam_mid",
        "s2_end",
        "summer",
        "end",
    ]
    source_days = np.array([source[k] for k in keys], dtype=float)
    target_days = np.array([target[k] for k in keys], dtype=float)
    return np.interp(BASE_BOUNDARIES, source_days, target_days)


def calendar_schedule_positions(
    target_year: int = 2027,
    boundary_buffer: int = 0,
    reduced_boundary: bool = False,
) -> np.ndarray:
    return positions_from_boundaries(
        map_boundaries_to_calendar(target_year),
        boundary_buffer=boundary_buffer,
        reduced_boundary=reduced_boundary,
    )


def template_from_prices(prices: Sequence[float], window: int = 14) -> np.ndarray:
    return smooth_series(prices, window=window)


def sign_positions(expected_change: Sequence[float], limit: int = POSITION_LIMIT, deadband: float = 0.0) -> np.ndarray:
    change = np.asarray(expected_change, dtype=float)
    out = np.zeros(len(change), dtype=int)
    out[change > deadband] = limit
    out[change < -deadband] = -limit
    return out


def template_positions(template: Sequence[float], deadband: float = 0.0) -> np.ndarray:
    t = np.asarray(template, dtype=float)
    slope = np.r_[np.diff(t), 0.0]
    return sign_positions(slope, deadband=deadband)


def warped_template(prices: Sequence[float], window: int = 14, target_year: int = 2027) -> np.ndarray:
    source_template = template_from_prices(prices, window)
    source_index = calendar_warp_indices(2026, target_year, len(source_template))
    grid = np.arange(len(source_template), dtype=float)
    return np.interp(source_index, grid, source_template)


def blend_template_positions(
    fixed_template: Sequence[float],
    warped: Sequence[float],
    weight_warped: float = 0.5,
    flat_on_disagreement: bool = True,
    deadband: float = 0.0,
) -> np.ndarray:
    a = np.asarray(fixed_template, dtype=float)
    b = np.asarray(warped, dtype=float)
    fixed_slope = np.r_[np.diff(a), 0.0]
    warp_slope = np.r_[np.diff(b), 0.0]
    if flat_on_disagreement:
        agree = np.sign(fixed_slope) == np.sign(warp_slope)
        expected = (1 - weight_warped) * fixed_slope + weight_warped * warp_slope
        out = sign_positions(expected, deadband=deadband)
        out[~agree] = 0
        return out
    expected = (1 - weight_warped) * fixed_slope + weight_warped * warp_slope
    return sign_positions(expected, deadband=deadband)


def phase_regression_positions(
    prices: Sequence[float],
    phases: Sequence[str] | None = None,
    warmup: int = 30,
    ridge: float = 1.0,
) -> np.ndarray:
    y = np.asarray(prices, dtype=float)
    labels = np.asarray(phases if phases is not None else phase_labels(len(y)))
    unique = list(dict.fromkeys(labels.tolist()))
    x = np.column_stack([np.ones(len(y))] + [(labels == label).astype(float) for label in unique])
    out = np.zeros(len(y), dtype=int)
    for t in range(len(y) - 1):
        if t < warmup:
            continue
        xt = x[: t + 1]
        yt = y[: t + 1]
        beta = np.linalg.solve(xt.T @ xt + ridge * np.eye(xt.shape[1]), xt.T @ yt)
        forecast = float(x[t + 1] @ beta)
        out[t] = int(POSITION_LIMIT * np.sign(forecast - y[t]))
    return out


def fourier_design(days: Sequence[float], order: int, period: float = 365.0) -> np.ndarray:
    t = np.asarray(days, dtype=float)
    cols = [np.ones(len(t))]
    for k in range(1, order + 1):
        angle = 2 * np.pi * k * t / period
        cols.extend([np.sin(angle), np.cos(angle)])
    return np.column_stack(cols)


def fourier_positions(
    prices: Sequence[float],
    order: int = 2,
    warmup: int = 60,
    ridge: float = 1.0,
    residual_ar1: bool = False,
) -> np.ndarray:
    y = np.asarray(prices, dtype=float)
    x = fourier_design(np.arange(len(y)), order)
    out = np.zeros(len(y), dtype=int)
    for t in range(len(y) - 1):
        if t < warmup:
            continue
        xt = x[: t + 1]
        beta = np.linalg.solve(xt.T @ xt + ridge * np.eye(xt.shape[1]), xt.T @ y[: t + 1])
        forecast = float(x[t + 1] @ beta)
        if residual_ar1:
            residuals = y[: t + 1] - xt @ beta
            if len(residuals) >= 10:
                phi = ar1_phi(residuals)
                forecast += phi * residuals[-1]
        out[t] = int(POSITION_LIMIT * np.sign(forecast - y[t]))
    return out


def _rls_update(theta: np.ndarray, covariance: np.ndarray, feature: np.ndarray, observation: float, forgetting: float) -> tuple[np.ndarray, np.ndarray]:
    denom = forgetting + float(feature @ covariance @ feature)
    gain = covariance @ feature / max(denom, 1e-12)
    innovation = observation - float(feature @ theta)
    next_theta = theta + gain * innovation
    next_covariance = (covariance - np.outer(gain, feature @ covariance)) / forgetting
    next_covariance = (next_covariance + next_covariance.T) / 2.0
    return next_theta, next_covariance


def rls_positions(
    prices: Sequence[float],
    template: Sequence[float],
    forgetting: float = 0.995,
    mode: str = "global",
    prior_variance: float = 4.0,
) -> np.ndarray:
    """Causal baseline/amplitude RLS using an externally fixed template."""

    y = np.asarray(prices, dtype=float)
    t = np.asarray(template, dtype=float)
    if mode == "global":
        centered = t - np.mean(t)
        features = np.column_stack([np.ones(len(t)), centered])
        theta = np.array([np.mean(t), 1.0], dtype=float)
    elif mode == "separate_wave_amplitudes":
        baseline = 45.0
        seasonal = t - baseline
        large = np.zeros(len(t), dtype=float)
        small = np.zeros(len(t), dtype=float)
        large[np.r_[15:92, 161:235]] = seasonal[np.r_[15:92, 161:235]]
        small[np.r_[92:161, 235:302]] = seasonal[np.r_[92:161, 235:302]]
        features = np.column_stack([np.ones(len(t)), large, small])
        theta = np.array([baseline, 1.0, 1.0], dtype=float)
    else:
        raise ValueError(f"unknown RLS mode {mode}")
    out = np.zeros(len(y), dtype=int)
    covariance = np.eye(features.shape[1]) * prior_variance
    for day in range(len(y) - 1):
        theta, covariance = _rls_update(theta, covariance, features[day], y[day], forgetting)
        forecast = float(features[day + 1] @ theta)
        out[day] = int(POSITION_LIMIT * np.sign(forecast - y[day]))
    return out


def rls_dual_template_agreement_positions(
    prices: Sequence[float],
    fixed_template: Sequence[float],
    warped_template: Sequence[float],
    forgetting: float = 0.995,
) -> np.ndarray:
    """Use two causal amplitude filters and trade only on sign agreement."""

    fixed = rls_positions(prices, fixed_template, forgetting, "global")
    warped = rls_positions(prices, warped_template, forgetting, "global")
    return np.where(np.sign(fixed) == np.sign(warped), fixed, 0).astype(int)


def kalman_residual_positions(
    prices: Sequence[float],
    template: Sequence[float],
    process_variance: float = 0.05,
    observation_variance: float = 1.0,
) -> np.ndarray:
    """Compact causal local-level Kalman filter on template residuals."""

    y = np.asarray(prices, dtype=float)
    t = np.asarray(template, dtype=float)
    state = 0.0
    covariance = 4.0
    out = np.zeros(len(y), dtype=int)
    for day in range(len(y) - 1):
        covariance += process_variance
        gain = covariance / max(covariance + observation_variance, 1e-12)
        innovation = (y[day] - t[day]) - state
        state += gain * innovation
        covariance = (1 - gain) * covariance
        forecast = t[day + 1] + state
        out[day] = int(POSITION_LIMIT * np.sign(forecast - y[day]))
    return out


def ar1_phi(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    if len(x) < 3:
        return float("nan")
    x0 = x[:-1] - np.mean(x[:-1])
    x1 = x[1:] - np.mean(x[1:])
    denom = float(x0 @ x0)
    return float(x0 @ x1 / denom) if denom > 1e-12 else float("nan")


def half_life(phi: float) -> float:
    if not np.isfinite(phi) or not 0 < abs(phi) < 1:
        return float("nan")
    return float(np.log(0.5) / np.log(abs(phi)))


def residual_reversion_positions(
    prices: Sequence[float],
    template: Sequence[float],
    phi: float,
) -> np.ndarray:
    y = np.asarray(prices, dtype=float)
    t = np.asarray(template, dtype=float)
    residual = y - t
    expected = np.zeros(len(y), dtype=float)
    expected[:-1] = np.r_[np.diff(t), 0.0][:-1] + (phi - 1.0) * residual[:-1]
    return sign_positions(expected)


def summer_reversion_positions(
    prices: Sequence[float],
    start: int = 302,
    baseline: float = 45.0,
    online_baseline: bool = False,
    ramp_days: int = 0,
    limit: int = POSITION_LIMIT,
) -> np.ndarray:
    y = np.asarray(prices, dtype=float)
    out = np.zeros(len(y), dtype=int)
    observations: list[float] = []
    for day in range(start, len(y) - 1):
        observations.append(float(y[day]))
        target = float(np.mean(observations)) if online_baseline and observations else baseline
        size = limit if ramp_days <= 0 else int(round(limit * min(1.0, (day - start + 1) / ramp_days)))
        out[day] = int(size * np.sign(target - y[day]))
    return out


def _integer_position_check(positions: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(positions)) and np.all(positions == np.rint(positions)))


def max_drawdown(pnl: Sequence[float]) -> float:
    curve = np.cumsum(np.asarray(pnl, dtype=float))
    if len(curve) == 0:
        return 0.0
    return float(np.min(curve - np.maximum.accumulate(curve)))


def backtest(
    prices: Sequence[float],
    positions: Sequence[float],
    label: str = "",
    phases: Sequence[str] | None = None,
) -> dict[str, object]:
    """Evaluate positions with exact simulator timing and validity checks."""

    p = np.asarray(prices, dtype=float)
    pos = np.asarray(positions)
    if len(p) != len(pos):
        raise ValueError("prices and positions must have equal length")
    integer = _integer_position_check(pos)
    within_limit = bool(np.all(np.abs(pos) <= POSITION_LIMIT))
    capital = np.abs(pos.astype(float) * p.astype(float))
    budget_violations = int(np.sum(capital > TOTAL_BUDGET))
    pnl = pos[:-1].astype(float) * np.diff(p)
    active = pos[:-1] != 0
    sharpe = 0.0
    if len(pnl) > 1 and np.std(pnl, ddof=1) > 1e-12:
        sharpe = float(np.sqrt(365.0) * np.mean(pnl) / np.std(pnl, ddof=1))
    q_edges = [0, 91, 182, 274, len(p)]
    quarter_pnl = {f"Q{i + 1}": float(np.sum(pnl[q_edges[i] : min(q_edges[i + 1], len(pnl))])) for i in range(4)}
    phase_series = np.asarray(phases if phases is not None else phase_labels(len(p)))
    phase_pnl = {
        str(phase): float(np.sum(pnl[np.where(phase_series[:-1] == phase)[0]]))
        for phase in dict.fromkeys(phase_series[:-1].tolist())
    }
    curve = np.r_[np.cumsum(pnl), np.cumsum(pnl)[-1] if len(pnl) else 0.0]
    return {
        "model": label,
        "pnl": float(np.sum(pnl)),
        "sharpe": sharpe,
        "hit_rate": float(np.mean(pnl > 0)) if len(pnl) else 0.0,
        "active_hit_rate": float(np.mean(pnl[active] > 0)) if np.any(active) else float("nan"),
        "active_days": int(np.sum(active)),
        "total_days": int(len(pnl)),
        "max_drawdown": max_drawdown(pnl),
        "max_capital": float(np.max(capital)) if len(capital) else 0.0,
        "avg_active_capital": float(np.mean(capital[:-1][active])) if np.any(active) else 0.0,
        "pnl_per_max_capital": float(np.sum(pnl) / np.max(capital)) if np.max(capital) > 0 else 0.0,
        "budget_violations": budget_violations,
        "integral_positions": int(integer),
        "within_limit": int(within_limit),
        "quarter_pnl": quarter_pnl,
        "phase_pnl": phase_pnl,
        "positions": pos.astype(int) if integer else pos,
        "daily_pnl": pnl,
        "capital": capital,
        "curve": curve,
    }


def metrics_frame(results: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {k: v for k, v in result.items() if k not in {"positions", "daily_pnl", "capital", "curve", "phase_pnl", "quarter_pnl"}}
        quarters = result.get("quarter_pnl", {})
        for name, value in quarters.items():
            row[f"quarter_{name[-1]}_pnl"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def volatility_by_phase(prices: Sequence[float]) -> pd.DataFrame:
    y = np.asarray(prices, dtype=float)
    changes = np.diff(y)
    labels = phase_labels(len(y))[:-1]
    rows = []
    for phase in dict.fromkeys(labels.tolist()):
        x = changes[labels == phase]
        rows.append(
            {
                "phase": phase,
                "days": len(x),
                "std_change": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
                "mean_abs_change": float(np.mean(np.abs(x))) if len(x) else float("nan"),
                "mean_change": float(np.mean(x)) if len(x) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def level_next_move_correlations(prices: Sequence[float]) -> pd.DataFrame:
    y = np.asarray(prices, dtype=float)
    move = np.diff(y)
    level = y[:-1]
    rows = [{"regime": "all", "n": len(move), "corr": float(np.corrcoef(level, move)[0, 1])}]
    q = np.quantile(level, [0.25, 0.5, 0.75])
    buckets = [
        ("low_quartile", level <= q[0]),
        ("middle_half", (level > q[0]) & (level <= q[2])),
        ("high_quartile", level > q[2]),
    ]
    for name, mask in buckets:
        x, z = level[mask], move[mask]
        rows.append({"regime": name, "n": int(len(x)), "corr": float(np.corrcoef(x, z)[0, 1]) if len(x) > 2 else float("nan")})
    labels = phase_labels(len(y))[:-1]
    for phase in dict.fromkeys(labels.tolist()):
        mask = labels == phase
        x, z = level[mask], move[mask]
        rows.append({"regime": phase, "n": int(len(x)), "corr": float(np.corrcoef(x, z)[0, 1]) if len(x) > 2 else float("nan")})
    return pd.DataFrame(rows)


def residual_diagnostics(prices: Sequence[float], template: Sequence[float], summer_start: int = 302) -> pd.DataFrame:
    y = np.asarray(prices, dtype=float)
    t = np.asarray(template, dtype=float)
    raw = y - 45.0
    residual = y - t
    summer = y[summer_start:]
    summer_residual = summer - 45.0
    rows = []
    for name, x in [("raw_level_minus_45_all", raw), ("template_residual_all", residual), (f"raw_level_minus_45_d{summer_start}", summer), ("summer_level_minus_45", summer_residual)]:
        phi = ar1_phi(x)
        rows.append({"series": name, "n": len(x), "mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)), "phi_ar1": phi, "half_life_days": half_life(phi), "lag1_acf": float(np.corrcoef(x[:-1], x[1:])[0, 1])})
    for start in [302, 322]:
        x = y[start:]
        move = np.diff(x)
        corr = np.corrcoef(x[:-1], move)[0, 1]
        rows.append({"series": f"summer_level_next_move_d{start}", "n": len(move), "mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)), "phi_ar1": ar1_phi(x - 45.0), "half_life_days": half_life(ar1_phi(x - 45.0)), "lag1_acf": float(corr)})
    return pd.DataFrame(rows)


WAVE_SEGMENTS = [
    {"wave": "S1_large", "start": 15, "peak": 48, "end": 92, "kind": "large"},
    {"wave": "S1_small", "start": 92, "peak": 115, "end": 161, "kind": "small"},
    {"wave": "S2_large", "start": 161, "peak": 196, "end": 235, "kind": "large"},
    {"wave": "S2_small", "start": 235, "peak": 252, "end": 302, "kind": "small"},
]


def resample_segment(values: Sequence[float], start: int, end: int, points: int = 80) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    old = np.linspace(0.0, 1.0, end - start)
    new = np.linspace(0.0, 1.0, points)
    return np.interp(new, old, x[start:end])


def normalised_wave_shapes(prices: Sequence[float], points: int = 80) -> pd.DataFrame:
    y = np.asarray(prices, dtype=float)
    rows = []
    for wave in WAVE_SEGMENTS:
        shape = resample_segment(smooth_series(y, 7), wave["start"], wave["end"], points)
        shape = (shape - shape[0]) / max(np.ptp(shape), 1e-9)
        rows.append({"wave": wave["wave"], "kind": wave["kind"], "start": wave["start"], "peak": wave["peak"], "end": wave["end"], "amplitude": float(np.ptp(smooth_series(y, 7)[wave["start"] : wave["end"]])), "shape": shape})
    return pd.DataFrame(rows)


def semester_transfer(prices: Sequence[float]) -> pd.DataFrame:
    shapes = normalised_wave_shapes(prices)
    rows = []
    for kind in ["large", "small"]:
        a = shapes[shapes.kind == kind].iloc[0]["shape"]
        b = shapes[shapes.kind == kind].iloc[1]["shape"]
        rows.append({"comparison": f"S1_vs_S2_{kind}", "corr_normalised_shape": float(np.corrcoef(a, b)[0, 1]), "rmse_normalised_shape": float(np.sqrt(np.mean((a - b) ** 2))), "amplitude_S1": float(shapes[shapes.kind == kind].iloc[0].amplitude), "amplitude_S2": float(shapes[shapes.kind == kind].iloc[1].amplitude), "amplitude_ratio_S2_over_S1": float(shapes[shapes.kind == kind].iloc[1].amplitude / max(shapes[shapes.kind == kind].iloc[0].amplitude, 1e-9))})
    a = shapes[shapes.wave == "S1_large"].iloc[0]["shape"]
    b = shapes[shapes.wave == "S1_small"].iloc[0]["shape"]
    c = shapes[shapes.wave == "S2_large"].iloc[0]["shape"]
    d = shapes[shapes.wave == "S2_small"].iloc[0]["shape"]
    rows.extend([
        {"comparison": "S1_large_vs_small", "corr_normalised_shape": float(np.corrcoef(a, b)[0, 1]), "rmse_normalised_shape": float(np.sqrt(np.mean((a - b) ** 2))), "amplitude_S1": float(shapes[shapes.wave == "S1_large"].iloc[0].amplitude), "amplitude_S2": float(shapes[shapes.wave == "S1_small"].iloc[0].amplitude), "amplitude_ratio_S2_over_S1": float(shapes[shapes.wave == "S1_small"].iloc[0].amplitude / max(shapes[shapes.wave == "S1_large"].iloc[0].amplitude, 1e-9))},
        {"comparison": "S2_large_vs_small", "corr_normalised_shape": float(np.corrcoef(c, d)[0, 1]), "rmse_normalised_shape": float(np.sqrt(np.mean((c - d) ** 2))), "amplitude_S1": float(shapes[shapes.wave == "S2_large"].iloc[0].amplitude), "amplitude_S2": float(shapes[shapes.wave == "S2_small"].iloc[0].amplitude), "amplitude_ratio_S2_over_S1": float(shapes[shapes.wave == "S2_small"].iloc[0].amplitude / max(shapes[shapes.wave == "S2_large"].iloc[0].amplitude, 1e-9))},
    ])
    return pd.DataFrame(rows)


def leave_one_wave_out(prices: Sequence[float]) -> pd.DataFrame:
    y = np.asarray(prices, dtype=float)
    shapes = normalised_wave_shapes(y)
    rows = []
    for _, held in shapes.iterrows():
        train = shapes[shapes.wave != held.wave]
        median_shape = np.median(np.vstack(train["shape"].to_numpy()), axis=0)
        held_shape = held["shape"]
        predicted_direction = np.sign(np.gradient(np.interp(np.arange(len(held_shape)), np.arange(len(held_shape)), median_shape)))
        # Map the held-out wave's normalised shape direction to its daily index.
        day_grid = np.linspace(0, len(held_shape) - 1, held.end - held.start)
        direction = np.sign(np.interp(day_grid, np.arange(len(held_shape)), predicted_direction))
        direction[direction == 0] = 1
        pos = np.zeros(len(y), dtype=int)
        pos[int(held.start) : int(held.end)] = (direction * POSITION_LIMIT).astype(int)
        result = backtest(y, pos, label=f"LOO_{held.wave}")
        rows.append({"held_out_wave": held.wave, "fit_waves": ",".join(train.wave.tolist()), "pnl": result["pnl"], "active_hit_rate": result["active_hit_rate"], "max_drawdown": result["max_drawdown"], "active_days": result["active_days"]})
    return pd.DataFrame(rows)


def calendar_wave_intervals(year: int = 2026) -> dict[str, tuple[int, int]]:
    """Map preregistered wave boundaries to a known academic calendar.

    This uses dates only.  It never inspects prices or price extrema.
    """

    if year == 2026:
        mapped = BASE_BOUNDARIES.astype(float)
    else:
        mapped = map_boundaries_to_calendar(year)
    return {
        str(wave["wave"]): (
            int(round(np.interp(float(wave["start"]), BASE_BOUNDARIES, mapped))),
            int(round(np.interp(float(wave["end"]), BASE_BOUNDARIES, mapped))),
        )
        for wave in WAVE_SEGMENTS
    }


def academic_phase_intervals(year: int = 2026) -> dict[str, tuple[int, int]]:
    """Broad event-defined two-wave intervals using dates only."""

    landmarks = landmark_days(year)
    return {
        "S1_large": (int(round(landmarks["s1_orientation_mid"])), int(round(landmarks["s1_break"]))),
        "S1_small": (int(round(landmarks["s1_break"])), int(round(landmarks["s1_exam_mid"]))),
        "S2_large": (int(round(landmarks["s2_orientation_mid"])), int(round(landmarks["s2_break"]))),
        "S2_small": (int(round(landmarks["s2_break"])), int(round(landmarks["s2_exam_mid"]))),
    }


def _local_wave_curve(prices: Sequence[float], wave: Mapping[str, object], window: int = 7) -> np.ndarray:
    """Smooth one source wave locally, with no values outside its interval."""

    y = np.asarray(prices, dtype=float)
    start, end = int(wave["start"]), int(wave["end"])
    return smooth_series(y[start:end], window=window)


def transfer_template(
    prices: Sequence[float],
    source_wave_names: Sequence[str],
    target_wave_names: Sequence[str],
    target_year: int = 2026,
    target_mode: str = "fixed",
    window: int = 7,
    baseline: float = 45.0,
) -> tuple[np.ndarray, set[int]]:
    """Transfer source-wave levels into target phases without target prices.

    The source curves are smoothed inside each source interval only.  The
    returned set is the exact set of source price indices used to construct the
    template, which makes provenance and leakage checks explicit.
    """

    y = np.asarray(prices, dtype=float)
    if len(source_wave_names) != len(target_wave_names):
        raise ValueError("source and target wave lists must have equal length")
    if target_mode == "fixed":
        intervals = {wave["wave"]: (int(wave["start"]), int(wave["end"])) for wave in WAVE_SEGMENTS}
    elif target_mode == "calendar":
        intervals = calendar_wave_intervals(target_year)
    elif target_mode == "academic":
        intervals = academic_phase_intervals(target_year)
    else:
        raise ValueError("target_mode must be 'fixed', 'calendar', or 'academic'")

    template = np.full(len(y), float(baseline), dtype=float)
    source_indices: set[int] = set()
    for source_name, target_name in zip(source_wave_names, target_wave_names):
        source = next(wave for wave in WAVE_SEGMENTS if wave["wave"] == source_name)
        target_start, target_end = intervals[target_name]
        source_start, source_end = int(source["start"]), int(source["end"])
        source_indices.update(range(source_start, source_end))
        curve = _local_wave_curve(y, source, window=window)
        source_grid = np.linspace(0.0, 1.0, len(curve))
        target_grid = np.linspace(0.0, 1.0, max(2, target_end - target_start))
        target_curve = np.interp(target_grid, source_grid, curve)
        template[target_start:target_end] = target_curve[: max(0, target_end - target_start)]
    return template, source_indices


def held_out_wave_template(
    prices: Sequence[float],
    held_out_wave: str,
    window: int = 7,
    baseline: float = 45.0,
    target_mode: str = "fixed",
    target_year: int = 2026,
) -> tuple[np.ndarray, set[int], list[str]]:
    """Construct only the held wave from the other wave episodes.

    The same-kind training waves provide a more meaningful shape prior, while
    all three non-held waves remain eligible and are recorded in provenance.
    No price from the held interval is read.
    """

    held = next(wave for wave in WAVE_SEGMENTS if wave["wave"] == held_out_wave)
    train_waves = [wave for wave in WAVE_SEGMENTS if wave["wave"] != held_out_wave]
    same_kind = [wave for wave in train_waves if wave["kind"] == held["kind"]]
    source = same_kind[0] if same_kind else train_waves[0]
    template, source_indices = transfer_template(
        prices,
        [source["wave"]],
        [held_out_wave],
        target_year=target_year,
        target_mode=target_mode,
        window=window,
        baseline=baseline,
    )
    return template, source_indices, [wave["wave"] for wave in train_waves]


def shifted_template(template: Sequence[float], shift: int | float) -> np.ndarray:
    """Shift a template in time, using edge values outside the known range."""

    t = np.asarray(template, dtype=float)
    grid = np.arange(len(t), dtype=float)
    return np.interp(grid + float(shift), grid, t, left=t[0], right=t[-1])


def shifted_template_section(
    template: Sequence[float],
    start: int,
    end: int,
    shift: int | float,
) -> np.ndarray:
    """Shift only one event section while keeping the rest unchanged."""

    out = np.asarray(template, dtype=float).copy()
    section = shifted_template(out[start:end], shift)
    out[start:end] = section
    return out


def phase_shuffled_template(template: Sequence[float], seed: int = SEED) -> np.ndarray:
    """Shuffle broad phase blocks, preserving each block's internal shape."""

    t = np.asarray(template, dtype=float)
    blocks = [(int(BASE_BOUNDARIES[i]), int(BASE_BOUNDARIES[i + 1])) for i in range(len(BASE_BOUNDARIES) - 1)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blocks))
    out = np.empty_like(t)
    cursor = 0
    for index in order:
        start, end = blocks[index]
        block = t[start:end]
        out[cursor : cursor + len(block)] = block
        cursor += len(block)
    return out


def wave_role_exchanged_template(template: Sequence[float], baseline: float = 45.0) -> np.ndarray:
    """Swap large-wave and small-wave deviations using fixed phase intervals."""

    t = np.asarray(template, dtype=float)
    out = np.full(len(t), float(baseline), dtype=float)
    pairs = [(WAVE_SEGMENTS[0], WAVE_SEGMENTS[1]), (WAVE_SEGMENTS[2], WAVE_SEGMENTS[3])]
    for large, small in pairs:
        large_curve = t[int(large["start"]) : int(large["end"])] - baseline
        small_curve = t[int(small["start"]) : int(small["end"])] - baseline
        large_target = np.interp(
            np.linspace(0.0, 1.0, int(large["end"]) - int(large["start"])),
            np.linspace(0.0, 1.0, len(small_curve)),
            small_curve,
        )
        small_target = np.interp(
            np.linspace(0.0, 1.0, int(small["end"]) - int(small["start"])),
            np.linspace(0.0, 1.0, len(large_curve)),
            large_curve,
        )
        out[int(large["start"]) : int(large["end"])] = baseline + large_target
        out[int(small["start"]) : int(small["end"])] = baseline + small_target
    return out


def rls_positions_window(
    prices: Sequence[float],
    template: Sequence[float],
    start: int,
    end: int,
    forgetting: float = 0.995,
    mode: str = "global",
    prior_variance: float = 4.0,
) -> np.ndarray:
    """Run RLS only inside [start, end), resetting to the frozen prior."""

    y = np.asarray(prices, dtype=float)
    t = np.asarray(template, dtype=float)
    if len(y) != len(t):
        raise ValueError("prices and template must have equal length")
    if not 0 <= start < len(y) or end <= start:
        raise ValueError("invalid RLS evaluation window")
    centered = t - np.mean(t)
    features = np.column_stack([np.ones(len(t)), centered])
    theta = np.array([np.mean(t), 1.0], dtype=float)
    covariance = np.eye(2) * prior_variance
    out = np.zeros(len(y), dtype=int)
    for day in range(start, min(end, len(y) - 1)):
        theta, covariance = _rls_update(theta, covariance, features[day], y[day], forgetting)
        forecast = float(features[day + 1] @ theta)
        out[day] = int(POSITION_LIMIT * np.sign(forecast - y[day]))
    return out


def rls_dual_template_agreement_window(
    prices: Sequence[float],
    fixed_template: Sequence[float],
    warped_template: Sequence[float],
    start: int,
    end: int,
    forgetting: float = 0.995,
    prior_variance: float = 4.0,
) -> np.ndarray:
    """Reset two RLS filters at start and trade only on non-zero agreement."""

    fixed = rls_positions_window(prices, fixed_template, start, end, forgetting, "global", prior_variance)
    warped = rls_positions_window(prices, warped_template, start, end, forgetting, "global", prior_variance)
    agree = (np.sign(fixed) == np.sign(warped)) & (fixed != 0) & (warped != 0)
    return np.where(agree, fixed, 0).astype(int)


def backtest_window(
    prices: Sequence[float],
    positions: Sequence[float],
    start: int,
    end: int,
    label: str = "",
) -> dict[str, object]:
    """Score decision days [start, end), including price[end] for the last move."""

    y = np.asarray(prices, dtype=float)
    pos = np.asarray(positions)
    stop = min(len(y), end + 1)
    if start < 0 or start >= stop - 1:
        raise ValueError("invalid backtest window")
    result = backtest(y[start:stop], pos[start:stop], label=label)
    result["evaluated_start"] = int(start)
    result["evaluated_end"] = int(min(end, len(y) - 1))
    return result


def baseline_mean_reversion_positions(
    prices: Sequence[float],
    start: int = 0,
    end: int | None = None,
    baseline: float = 45.0,
    forgetting: float = 0.995,
    mode: str = "fixed",
    prior_strength: float = 14.0,
    ramp_days: int = 0,
) -> np.ndarray:
    """Generic baseline-only alternatives used for RLS ablations."""

    y = np.asarray(prices, dtype=float)
    end = len(y) - 1 if end is None else min(int(end), len(y) - 1)
    out = np.zeros(len(y), dtype=int)
    observations: list[float] = []
    ewma = float(baseline)
    for day in range(max(0, start), max(0, end)):
        observations.append(float(y[day]))
        if mode == "fixed":
            target = float(baseline)
        elif mode == "online_mean":
            ewma = forgetting * ewma + (1.0 - forgetting) * y[day]
            target = ewma
        elif mode == "online_median":
            target = float(np.median(observations))
        elif mode == "shrunk_mean":
            target = (prior_strength * baseline + len(observations) * float(np.mean(observations))) / (prior_strength + len(observations))
        else:
            raise ValueError("unknown baseline mode")
        size = POSITION_LIMIT if ramp_days <= 0 else int(round(POSITION_LIMIT * min(1.0, (day - start + 1) / ramp_days)))
        out[day] = int(size * np.sign(target - y[day]))
    return out


def summer_policy_positions(
    prices: Sequence[float],
    start: int,
    mode: str = "fixed",
    baseline: float = 45.0,
    forgetting: float = 0.995,
    prior_strength: float = 14.0,
    ramp_days: int = 0,
) -> np.ndarray:
    """Explicit summer policies with dates chosen from academic landmarks."""

    return baseline_mean_reversion_positions(
        prices,
        start=start,
        end=len(prices) - 1,
        baseline=baseline,
        forgetting=forgetting,
        mode=mode,
        prior_strength=prior_strength,
        ramp_days=ramp_days,
    )


def causal_rls_prefix_check(
    prices: Sequence[float],
    template: Sequence[float],
    start: int,
    end: int,
    cut_day: int,
) -> bool:
    """Verify future-price perturbations do not change earlier RLS decisions."""

    y = np.asarray(prices, dtype=float)
    changed = y.copy()
    changed[cut_day + 1 :] = changed[cut_day + 1 :] + 17.0
    before = rls_positions_window(y, template, start, end)
    after = rls_positions_window(changed, template, start, end)
    return bool(np.array_equal(before[start : cut_day + 1], after[start : cut_day + 1]))


def decompose_components(prices: Sequence[float], template: Sequence[float], summer_baseline: float = 45.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(prices, dtype=float)
    t = np.asarray(template, dtype=float)
    large_mask = np.zeros(len(y), dtype=bool)
    small_mask = np.zeros(len(y), dtype=bool)
    for wave in WAVE_SEGMENTS:
        if wave["kind"] == "large":
            large_mask[wave["start"] : wave["end"]] = True
        else:
            small_mask[wave["start"] : wave["end"]] = True
    large = np.where(large_mask, t - summer_baseline, 0.0)
    small = np.where(small_mask, t - summer_baseline, 0.0)
    base = summer_baseline + large + small
    residual = y - base
    return large, small, residual


def block_bootstrap(values: Sequence[float], length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    blocks = []
    while sum(len(block) for block in blocks) < length:
        start = int(rng.integers(0, max(1, len(x) - block_length + 1)))
        blocks.append(x[start : start + block_length])
    return np.concatenate(blocks)[:length]


def scenario_price(
    prices: Sequence[float],
    template: Sequence[float],
    amplitude_large: float = 1.0,
    amplitude_small: float = 1.0,
    summer_shift: float = 0.0,
    transition_days: int = 14,
    residual: Sequence[float] | None = None,
) -> np.ndarray:
    large, small, original_residual = decompose_components(prices, template)
    resid = original_residual if residual is None else np.asarray(residual, dtype=float)
    y = 45.0 + amplitude_large * large + amplitude_small * small + resid
    if summer_shift:
        transition_start = 302
        if transition_days <= 0:
            weight = (np.arange(len(y)) >= transition_start).astype(float)
        else:
            weight = np.clip((np.arange(len(y)) - transition_start + 1) / transition_days, 0.0, 1.0)
        y = y + summer_shift * weight
    return y


def stress_paths(prices: Sequence[float], template: Sequence[float], n_bootstrap: int = 80, seed: int = SEED) -> list[tuple[str, np.ndarray]]:
    y = np.asarray(prices, dtype=float)
    t = np.asarray(template, dtype=float)
    paths: list[tuple[str, np.ndarray]] = []
    for amp in [0.6, 0.8, 1.0, 1.2, 1.4]:
        paths.append((f"common_amp_{amp:.1f}", scenario_price(y, t, amp, amp)))
    for large in [0.8, 1.2]:
        for small in [0.8, 1.2]:
            paths.append((f"separate_amp_L{large:.1f}_S{small:.1f}", scenario_price(y, t, large, small)))
    for shift in [-2.0, -1.0, 1.0, 2.0]:
        paths.append((f"summer_shift_{shift:+.1f}", scenario_price(y, t, 1.0, 1.0, summer_shift=shift, transition_days=14)))
    for transition in [0, 7, 28]:
        paths.append((f"summer_transition_{transition}", scenario_price(y, t, 1.0, 1.0, summer_shift=1.0, transition_days=transition)))
    _, _, residual = decompose_components(y, t)
    rng = np.random.default_rng(seed)
    base = scenario_price(y, t)
    for i in range(n_bootstrap):
        boot_resid = block_bootstrap(residual, len(y), block_length=7, rng=rng)
        paths.append((f"residual_block_bootstrap_{i:03d}", base - residual + boot_resid))
    return paths


def run_stress(
    prices: Sequence[float],
    template: Sequence[float],
    rules: Mapping[str, Callable[[np.ndarray], np.ndarray]],
    n_bootstrap: int = 80,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = stress_paths(prices, template, n_bootstrap=n_bootstrap, seed=seed)
    rows = []
    scenario_rows = []
    for path_name, path in paths:
        scenario_kind = path_name.split("_")[0]
        for rule_name, rule in rules.items():
            result = backtest(path, rule(path), label=rule_name)
            rows.append({"scenario": path_name, "scenario_kind": scenario_kind, "model": rule_name, "pnl": result["pnl"], "max_drawdown": result["max_drawdown"], "positive": int(result["pnl"] > 0)})
    detail = pd.DataFrame(rows)
    for model, group in detail.groupby("model"):
        scenario_rows.append({"model": model, "n_scenarios": len(group), "median_pnl": float(group.pnl.median()), "p10_pnl": float(group.pnl.quantile(0.10)), "worst_pnl": float(group.pnl.min()), "positive_rate": float(group.positive.mean()), "median_max_drawdown": float(group.max_drawdown.median())})
    return pd.DataFrame(scenario_rows).sort_values("median_pnl", ascending=False), detail


def _ar1_residual_path(
    length: int,
    sigma: float,
    phi: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate residuals from a broad AR(1) calibration, not a template."""

    innovations = rng.normal(0.0, sigma, size=length)
    residual = np.zeros(length, dtype=float)
    for day in range(1, length):
        residual[day] = phi * residual[day - 1] + innovations[day]
    return residual


def _asymmetric_kernel(days: np.ndarray, center: float, rise: float, decline: float) -> np.ndarray:
    distance = days - float(center)
    scale = np.where(distance < 0.0, max(float(rise), 1.0), max(float(decline), 1.0))
    return np.exp(-0.5 * (distance / scale) ** 2)


def _independent_timing_landmarks(timing_mode: str) -> tuple[dict[str, float], int, int]:
    """Return known event dates plus an optional earlier S1 break shift."""

    if timing_mode == "fixed_2026":
        return landmark_days(2026), 2026, 0
    landmarks = landmark_days(2027)
    if timing_mode == "easter_early_8":
        shift = 8
    elif timing_mode == "easter_early_11":
        shift = 11
    else:
        shift = 0
    if shift:
        landmarks = dict(landmarks)
        landmarks["s1_break"] -= shift
        landmarks["s1_resume"] -= shift
    return landmarks, 2027, shift


def event_kernel_path(
    timing_mode: str,
    rng: np.random.Generator,
    length: int = 365,
    path_id: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Create a low-parameter event path independent of the 7-day template."""

    landmarks, calendar_year, easter_shift = _independent_timing_landmarks(timing_mode)
    days = np.arange(length, dtype=float)
    peak_shift = int(rng.choice([-14, -7, -3, 0, 3, 7, 14]))
    rise_days = int(rng.integers(10, 36))
    decline_days = int(rng.integers(12, 46))
    large_scale = float(rng.uniform(0.40, 1.45))
    small_scale = float(rng.uniform(0.35, 1.45))
    secondary_present = bool(rng.random() >= 0.18)
    opening_amp = rng.uniform(6.0, 10.5, size=2) * large_scale
    secondary_amp = rng.uniform(2.0, 6.0, size=2) * small_scale * float(secondary_present)
    centres = [
        landmarks["s1_classes"] + peak_shift,
        landmarks["s1_resume"] + 10.0 + peak_shift,
        landmarks["s2_classes"] - 12.0 + peak_shift,
        landmarks["s2_classes"] + 45.0 + peak_shift,
    ]
    price = np.full(length, 45.0, dtype=float)
    for amplitude, centre in zip([opening_amp[0], secondary_amp[0], opening_amp[1], secondary_amp[1]], centres):
        price += float(amplitude) * _asymmetric_kernel(days, centre, rise_days, decline_days)
    exam_depth = float(rng.uniform(1.5, 5.0))
    for exam in [landmarks["s1_exam_mid"], landmarks["s2_exam_mid"]]:
        price -= exam_depth * _asymmetric_kernel(days, exam, decline_days, rise_days)
    summer_shift = float(rng.uniform(-2.0, 2.0))
    transition_days = int(rng.choice([0, 7, 14, 28]))
    summer = float(landmarks["summer"])
    if transition_days <= 0:
        transition = (days >= summer).astype(float)
    else:
        transition = np.clip((days - summer + 1.0) / transition_days, 0.0, 1.0)
    price += summer_shift * transition
    residual_sigma = float(rng.uniform(0.45, 1.15))
    residual_phi = float(rng.uniform(-0.20, 0.55))
    price += _ar1_residual_path(length, residual_sigma, residual_phi, rng)
    metadata: dict[str, object] = {
        "path_id": path_id,
        "generator_family": "event_kernel",
        "generator_source": "broad academic-event kernels plus independent AR(1) residual",
        "timing_mode": timing_mode,
        "calendar_year": calendar_year,
        "easter_shift_days": easter_shift,
        "peak_shift_days": peak_shift,
        "rise_days": rise_days,
        "decline_days": decline_days,
        "large_scale": large_scale,
        "small_scale": small_scale,
        "secondary_present": int(secondary_present),
        "summer_shift": summer_shift,
        "transition_days": transition_days,
        "residual_sigma": residual_sigma,
        "residual_phi": residual_phi,
    }
    return price, metadata


def _cross_semester_target_intervals(timing_mode: str) -> dict[str, tuple[int, int]]:
    if timing_mode == "fixed_2026":
        return calendar_wave_intervals(2026)
    intervals = calendar_wave_intervals(2027)
    if timing_mode in {"easter_early_8", "easter_early_11"}:
        shift = 8 if timing_mode.endswith("8") else 11
        intervals = dict(intervals)
        for name in ["S1_large", "S1_small"]:
            start, end = intervals[name]
            intervals[name] = (start - shift, end - shift)
    return intervals


def cross_semester_shape_path(
    prices: Sequence[float],
    timing_mode: str,
    rng: np.random.Generator,
    length: int = 365,
    path_id: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Mix 14-day source-wave shapes across semesters and rescale them."""

    y = np.asarray(prices, dtype=float)
    intervals = _cross_semester_target_intervals(timing_mode)
    days = np.arange(length, dtype=float)
    peak_shift = int(rng.choice([-14, -7, -3, 0, 3, 7, 14]))
    secondary_present = bool(rng.random() >= 0.22)
    output = np.full(length, 45.0, dtype=float)
    source_assignments: dict[str, str] = {}
    for target in WAVE_SEGMENTS:
        target_name = str(target["wave"])
        if target["kind"] == "large":
            pool = WAVE_SEGMENTS[2:] if target_name.startswith("S1") else WAVE_SEGMENTS[:2]
        else:
            pool = WAVE_SEGMENTS[2:] if target_name.startswith("S1") else WAVE_SEGMENTS[:2]
        source = pool[int(rng.integers(0, len(pool)))]
        source_assignments[target_name] = str(source["wave"])
        target_start, target_end = intervals[target_name]
        if target["kind"] == "small" and not secondary_present:
            continue
        source_curve = _local_wave_curve(y, source, window=14)
        source_curve = (source_curve - float(np.mean(source_curve))) / max(float(np.ptp(source_curve)), 1e-9)
        target_grid = np.linspace(0.0, 1.0, max(2, target_end - target_start))
        source_grid = np.linspace(0.0, 1.0, len(source_curve))
        curve = np.interp(target_grid, source_grid, source_curve)
        local_shift = int(np.clip(peak_shift, -max(1, len(curve) // 3), max(1, len(curve) // 3)))
        if local_shift:
            curve = np.interp(
                np.arange(len(curve), dtype=float) + local_shift,
                np.arange(len(curve), dtype=float),
                curve,
                left=curve[0],
                right=curve[-1],
            )
        amplitude = float(rng.uniform(4.0, 10.5) if target["kind"] == "large" else rng.uniform(1.8, 5.5))
        output[target_start:target_end] += amplitude * curve[: max(0, target_end - target_start)]
    landmarks, _, easter_shift = _independent_timing_landmarks(timing_mode)
    exam_depth = float(rng.uniform(0.5, 4.0))
    for exam in [landmarks["s1_exam_mid"], landmarks["s2_exam_mid"]]:
        output -= exam_depth * _asymmetric_kernel(days, exam, 18.0, 24.0)
    summer_shift = float(rng.uniform(-2.0, 2.0))
    transition_days = int(rng.choice([0, 7, 14, 28]))
    summer = float(landmarks["summer"])
    transition = np.ones(length) if transition_days <= 0 else np.clip((days - summer + 1.0) / transition_days, 0.0, 1.0)
    transition[days < summer] = 0.0
    output += summer_shift * transition
    residual_sigma = float(rng.uniform(0.45, 1.10))
    residual_phi = float(rng.uniform(-0.20, 0.55))
    output += _ar1_residual_path(length, residual_sigma, residual_phi, rng)
    metadata: dict[str, object] = {
        "path_id": path_id,
        "generator_family": "cross_semester_shape",
        "generator_source": "14-day local source-wave shapes transferred across semesters and rescaled",
        "timing_mode": timing_mode,
        "easter_shift_days": easter_shift,
        "peak_shift_days": peak_shift,
        "secondary_present": int(secondary_present),
        "summer_shift": summer_shift,
        "transition_days": transition_days,
        "exam_depth": exam_depth,
        "residual_sigma": residual_sigma,
        "residual_phi": residual_phi,
        "source_assignments": str(source_assignments),
    }
    return output, metadata


def independent_generator_paths(
    prices: Sequence[float],
    n_paths: int = 400,
    seed: int = SEED,
) -> list[tuple[str, np.ndarray, dict[str, object]]]:
    """Generate balanced event-kernel and cross-semester independent paths."""

    if n_paths < 2:
        raise ValueError("n_paths must be at least two")
    rng = np.random.default_rng(seed)
    timing_modes = ["fixed_2026", "official_2027", "easter_early_8", "easter_early_11"]
    paths: list[tuple[str, np.ndarray, dict[str, object]]] = []
    for path_id in range(n_paths):
        timing_mode = timing_modes[path_id % len(timing_modes)]
        if path_id % 2 == 0:
            path, metadata = event_kernel_path(timing_mode, rng, path_id=path_id)
        else:
            path, metadata = cross_semester_shape_path(prices, timing_mode, rng, path_id=path_id)
        path_name = f"independent_{metadata['generator_family']}_{path_id:04d}"
        metadata["path_name"] = path_name
        paths.append((path_name, path, metadata))
    return paths


def save_figures(
    prices: Sequence[float],
    template: Sequence[float],
    results: Mapping[str, Mapping[str, object]],
    figure_dir: str | Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    out_dir = Path(figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(prices, dtype=float)
    t = np.asarray(template, dtype=float)
    paths: list[Path] = []
    events = landmark_days(2026)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(y, color="#177e89", linewidth=1.0, alpha=0.55, label="Round 1 price")
    ax.plot(t, color="#d1495b", linewidth=2.0, label="14-day centered template")
    for name in ["s1_classes", "s1_break", "s1_resume", "s1_end", "s2_classes", "s2_break", "s2_resume", "s2_end", "summer"]:
        ax.axvline(events[name], color="#555555", linewidth=0.6, alpha=0.35)
    ax.set(xlabel="Round 1 day (day 0 = 1 Jan 2026)", ylabel="AUD", title="Boat Party Ticket: Round 1 price and structural calendar landmarks")
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    path = out_dir / "price_calendar.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for name, result in results.items():
        ax.plot(result["curve"], linewidth=1.4, label=name)
    ax.axhline(0, color="#444444", linewidth=0.6)
    ax.set(xlabel="Day", ylabel="Cumulative P&L (AUD)", title="Round 1 cumulative P&L: shortlisted research rules")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = out_dir / "model_cumulative_pnl.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def write_result_tables(
    result_dir: str | Path,
    model_frame: pd.DataFrame,
    schedule_frame: pd.DataFrame,
    template_frame: pd.DataFrame,
    stress_frame: pd.DataFrame,
    stress_detail: pd.DataFrame,
    wave_frame: pd.DataFrame,
    transfer_frame: pd.DataFrame,
    diagnostics_frame: pd.DataFrame,
    calendar_frame: pd.DataFrame,
) -> None:
    out_dir = Path(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "model_results.csv": model_frame,
        "schedule_sensitivity.csv": schedule_frame,
        "template_sensitivity.csv": template_frame,
        "stress_summary.csv": stress_frame,
        "stress_detail.csv": stress_detail,
        "template_conditioned_stress_summary.csv": stress_frame,
        "template_conditioned_stress_detail.csv": stress_detail,
        "wave_leave_one_out.csv": wave_frame,
        "semester_transfer.csv": transfer_frame,
        "residual_diagnostics.csv": diagnostics_frame,
        "calendar_events.csv": calendar_frame,
    }.items():
        frame.to_csv(out_dir / name, index=False)
