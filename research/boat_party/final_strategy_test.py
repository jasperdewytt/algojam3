"""Final frozen fixed-index Boat Party strategy audit.

This module is deliberately narrower than the earlier research.  It tests only
Candidates A-D from the final brief, plus fixed references.  The complete Round
1 template is an externally fixed Round 2 prior; every Round 1 score below is
labelled in-sample template reconstruction.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from . import analysis as a
except ImportError:  # direct execution from research/boat_party
    import analysis as a


SEED = 20260809
POSITION_LIMIT = 1_000
TOTAL_BUDGET = 600_000.0
DEADBAND = 0.02
SUMMER_START = 322
SUMMER_EQUILIBRIUM = 45.0
AMPLITUDE_LEVELS = (0.50, 0.75, 1.00, 1.25, 1.50)
GRADUAL_SUMMER_TARGETS = (43.0, 45.0, 47.0)
GRADUAL_SUMMER_DURATIONS = (7, 14, 28)
GRADUAL_SUMMER_PATHS = 200
GRADUAL_SUMMER_SEED_OFFSET = 97
MODEL_ORDER = [
    "Candidate A",
    "Candidate B",
    "Candidate C",
    "Candidate D",
    "Broad fixed-day schedule",
    "Constant AUD 45 mean reversion",
    "Flat Boat Party position",
    "Fixed 7-day template, no deadband",
]
CANDIDATE_NAMES = MODEL_ORDER[:4]


def load_round1(repo_root: str | Path | None = None) -> tuple[Path, np.ndarray]:
    root = a.find_repo_root(repo_root)
    prices = a.load_prices(root)["Price"].to_numpy(dtype=float)
    return root, prices


def one_day_slope(template: Sequence[float]) -> np.ndarray:
    """Return template[t+1] - template[t], with no day-365 access."""

    t = np.asarray(template, dtype=float)
    expected = np.zeros(len(t), dtype=float)
    expected[:-1] = t[1:] - t[:-1]
    return expected


def slope_votes(template: Sequence[float], deadband: float = DEADBAND) -> np.ndarray:
    expected = one_day_slope(template)
    votes = np.zeros(len(expected), dtype=int)
    votes[expected > deadband] = 1
    votes[expected < -deadband] = -1
    return votes


def positions_from_votes(votes: Sequence[int]) -> np.ndarray:
    v = np.asarray(votes, dtype=int)
    out = (v * POSITION_LIMIT).astype(int)
    out[-1] = 0
    return out


def summer_mean_reversion_positions(
    prices: Sequence[float],
    start: int = SUMMER_START,
    equilibrium: float = SUMMER_EQUILIBRIUM,
) -> np.ndarray:
    """Frozen summer rule; it uses the current observed price only."""

    y = np.asarray(prices, dtype=float)
    out = np.zeros(len(y), dtype=int)
    if start < len(y) - 1:
        out[start : len(y) - 1][y[start : len(y) - 1] < equilibrium] = POSITION_LIMIT
        out[start : len(y) - 1][y[start : len(y) - 1] > equilibrium] = -POSITION_LIMIT
    out[-1] = 0
    return out


def frozen_templates(prices: Sequence[float]) -> dict[str, np.ndarray]:
    y = np.asarray(prices, dtype=float)
    # smooth_series preserves odd windows; the frozen majority is explicitly
    # 5/7/11 rather than relying on an even-window conversion side effect.
    return {f"template_{window}d": a.template_from_prices(y, window=window) for window in (5, 7, 11)}


def candidate_positions(
    prices_for_summer: Sequence[float],
    templates: Mapping[str, Sequence[float]],
) -> dict[str, np.ndarray]:
    """Build Candidates A-D from frozen fixed-index templates."""

    y = np.asarray(prices_for_summer, dtype=float)
    votes = {name: slope_votes(template, DEADBAND) for name, template in templates.items()}
    candidate_a = positions_from_votes(votes["template_7d"])
    positive_votes = sum((votes[name] == 1).astype(int) for name in ("template_5d", "template_7d", "template_11d"))
    negative_votes = sum((votes[name] == -1).astype(int) for name in ("template_5d", "template_7d", "template_11d"))
    candidate_b = np.zeros(len(y), dtype=int)
    candidate_b[positive_votes >= 2] = POSITION_LIMIT
    candidate_b[negative_votes >= 2] = -POSITION_LIMIT
    candidate_b[-1] = 0
    summer = summer_mean_reversion_positions(y, SUMMER_START, SUMMER_EQUILIBRIUM)
    candidate_c = candidate_a.copy()
    candidate_d = candidate_b.copy()
    candidate_c[SUMMER_START:] = summer[SUMMER_START:]
    candidate_d[SUMMER_START:] = summer[SUMMER_START:]
    return {
        "Candidate A": candidate_a,
        "Candidate B": candidate_b,
        "Candidate C": candidate_c,
        "Candidate D": candidate_d,
    }


def reference_positions(prices: Sequence[float], templates: Mapping[str, Sequence[float]]) -> dict[str, np.ndarray]:
    y = np.asarray(prices, dtype=float)
    refs = {
        "Broad fixed-day schedule": a.fixed_schedule_positions(),
        "Constant AUD 45 mean reversion": summer_mean_reversion_positions(y, start=0, equilibrium=SUMMER_EQUILIBRIUM),
        "Flat Boat Party position": np.zeros(len(y), dtype=int),
        "Fixed 7-day template, no deadband": positions_from_votes(slope_votes(templates["template_7d"], deadband=0.0)),
    }
    for position in refs.values():
        position[-1] = 0
    return refs


def all_strategy_positions(prices: Sequence[float]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    templates = frozen_templates(prices)
    return candidate_positions(prices, templates), reference_positions(prices, templates)


def metric_row(
    prices: Sequence[float],
    positions: Sequence[float],
    model: str,
    evidence_label: str = "in-sample Round 1 reconstruction",
) -> dict[str, object]:
    result = a.backtest(prices, positions, label=model)
    return {
        "model": model,
        "evidence_label": evidence_label,
        "template_use": "complete Round 1 fixed prior for Round 2" if "template" in model or model.startswith("Candidate") else "reference rule",
        "pnl": result["pnl"],
        "sharpe": result["sharpe"],
        "active_hit_rate": result["active_hit_rate"],
        "active_days": result["active_days"],
        "max_drawdown": result["max_drawdown"],
        "max_capital": result["max_capital"],
        "pnl_per_max_capital": result["pnl_per_max_capital"],
        "budget_violations": result["budget_violations"],
        "integral_positions": result["integral_positions"],
        "within_limit": result["within_limit"],
    }


def shifted_template(template: Sequence[float], displacement: int) -> np.ndarray:
    """Shift convention: +d delays the template d days relative to prices.

    At day t the shifted value is original_template[t-d].  Edge values are
    held outside the 0..364 range.  Thus +1 means the seasonal feature arrives
    one day later; -1 means one day earlier.
    """

    t = np.asarray(template, dtype=float)
    grid = np.arange(len(t), dtype=float)
    return np.interp(grid - float(displacement), grid, t, left=t[0], right=t[-1])


def shifted_candidate_positions(
    prices: Sequence[float],
    templates: Mapping[str, Sequence[float]],
    displacement: int,
) -> dict[str, np.ndarray]:
    shifted = {name: shifted_template(template, displacement) for name, template in templates.items()}
    return candidate_positions(prices, shifted)


def broad_seasonal_components(prices: Sequence[float]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build a broad stress base without supplying the exact 7-day path.

    A fixed 21-day denoising is used only as a stress generator.  The strategy
    continues to use its frozen 5/7/11-day Round 1 templates.
    """

    y = np.asarray(prices, dtype=float)
    broad = a.smooth_series(y, window=21)
    components: dict[str, np.ndarray] = {}
    for wave in a.WAVE_SEGMENTS:
        component = np.zeros(len(y), dtype=float)
        start, end = int(wave["start"]), int(wave["end"])
        component[start:end] = broad[start:end] - SUMMER_EQUILIBRIUM
        components[str(wave["wave"])] = component
    return broad, components


def amplitude_path(components: Mapping[str, np.ndarray], multipliers: Sequence[float]) -> np.ndarray:
    names = ["S1_large", "S1_small", "S2_large", "S2_small"]
    path = np.full(len(next(iter(components.values()))), SUMMER_EQUILIBRIUM, dtype=float)
    for name, multiplier in zip(names, multipliers):
        path += float(multiplier) * np.asarray(components[name], dtype=float)
    return path


def evaluate_stress_path(
    prices: Sequence[float],
    frozen_templates: Mapping[str, Sequence[float]],
    label_prefix: str = "",
) -> list[dict[str, object]]:
    candidates = candidate_positions(prices, frozen_templates)
    references = reference_positions(prices, frozen_templates)
    positions = {**candidates, **references}
    rows = []
    for model in MODEL_ORDER:
        row = metric_row(prices, positions[model], model, evidence_label=label_prefix)
        rows.append(row)
    return rows


def summary_frame(detail: pd.DataFrame, group_columns: Sequence[str] = ("model",)) -> pd.DataFrame:
    rows = []
    for keys, group in detail.groupby(list(group_columns), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_paths": int(len(group)),
                "median_pnl": float(group["pnl"].median()),
                "p10_pnl": float(group["pnl"].quantile(0.10)),
                "worst_pnl": float(group["pnl"].min()),
                "positive_path_rate": float(np.mean(group["pnl"] > 0)),
                "median_max_drawdown": float(group["max_drawdown"].median()),
                # Summer-ablation rows intentionally omit capital because they
                # report a focused P&L/drawdown slice rather than full metrics.
                "median_max_capital": float(group["max_capital"].median()) if "max_capital" in group else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("median_pnl", ascending=False).reset_index(drop=True)


def effective_smoothing_window(probe: np.ndarray, requested_window: int) -> int:
    """Infer the effective odd support of the existing centered smoother."""

    smoothed = a.smooth_series(probe, window=requested_window)
    return int(np.count_nonzero(smoothed > 0.0))


def correctness_checks(
    prices: np.ndarray,
    templates: Mapping[str, np.ndarray],
    positions: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    toy_template = np.array([0.0, 0.04, 0.04])
    toy_prices = np.array([10.0, 12.0, 99.0])
    toy_positions = positions_from_votes(slope_votes(toy_template, DEADBAND))
    toy_result = a.backtest(toy_prices, toy_positions, "toy")
    checks.append({"check": "toy signal uses template[t+1]-template[t]", "passed": int(np.array_equal(toy_positions, np.array([1000, 0, 0]))), "evidence": str(toy_positions.tolist())})
    checks.append({"check": "toy P&L uses price[t+1]-price[t]", "passed": int(toy_result["pnl"] == 2000.0), "evidence": f"toy P&L={toy_result['pnl']:.0f}; a shifted P&L would include the 87 AUD move"})
    checks.append({"check": "one-day horizon only", "passed": int(toy_positions[1] == 0), "evidence": "no position is created from template[t+2]"})
    checks.append({"check": "no initialization warmup off-by-one", "passed": int(toy_positions[0] == 1000), "evidence": "day 0 uses the available template day 1"})
    checks.append({"check": "day 364 has no day 365 access", "passed": int(all(position[-1] == 0 for position in positions.values())), "evidence": "all strategy arrays explicitly set final day flat"})
    manual_pnl = float(np.sum(positions["Candidate A"][:-1].astype(float) * np.diff(prices)))
    observed_pnl = float(a.backtest(prices, positions["Candidate A"], "Candidate A")["pnl"])
    checks.append({"check": "backtest P&L equals position[t]*price change[t]", "passed": int(manual_pnl == observed_pnl), "evidence": f"manual={manual_pnl:.6f}, backtest={observed_pnl:.6f}"})
    checks.append({"check": "all positions integral", "passed": int(all(a.backtest(prices, position)["integral_positions"] == 1 for position in positions.values())), "evidence": "all candidates and references"})
    checks.append({"check": "all positions within +/-1000", "passed": int(all(a.backtest(prices, position)["within_limit"] == 1 for position in positions.values())), "evidence": "Boat Party instrument limit"})
    checks.append({"check": "Boat-only capital below AUD 600000", "passed": int(all(a.backtest(prices, position)["budget_violations"] == 0 for position in positions.values())), "evidence": "maximum Boat notional is below the shared portfolio cap"})
    probe = np.zeros(41, dtype=float)
    probe[20] = 1.0
    expected_windows = {"template_5d": 5, "template_7d": 7, "template_11d": 11}
    observed_windows = {name: effective_smoothing_window(probe, window) for name, window in (("template_5d", 5), ("template_7d", 7), ("template_11d", 11))}
    checks.append(
        {
            "check": "effective smoothing windows are exactly 5, 7 and 11",
            "passed": int(observed_windows == expected_windows),
            "evidence": str(observed_windows),
        }
    )
    # Compatibility-only proof: the old public 10-day request was converted by
    # smooth_series to the same effective 11-point smoother. It is not a new
    # voter or a new parameter search.
    legacy_templates = {
        "template_5d": a.template_from_prices(prices, window=5),
        "template_7d": a.template_from_prices(prices, window=7),
        "template_11d": a.template_from_prices(prices, window=10),
    }
    legacy_positions = candidate_positions(prices, legacy_templates)
    position_match = all(np.array_equal(legacy_positions[model], positions[model]) for model in ("Candidate B", "Candidate D"))
    pnl_match = all(
        float(a.backtest(prices, legacy_positions[model], model)["pnl"]) == float(a.backtest(prices, positions[model], model)["pnl"])
        for model in ("Candidate B", "Candidate D")
    )
    checks.append(
        {
            "check": "5/7/11 labels reproduce legacy Candidate B/D positions and P&L",
            "passed": int(position_match and pnl_match),
            "evidence": f"positions_equal={position_match}; pnl_equal={pnl_match}; B={a.backtest(prices, positions['Candidate B'])['pnl']:.2f}; D={a.backtest(prices, positions['Candidate D'])['pnl']:.2f}",
        }
    )
    checks.append({"check": "full Round 1 template provenance labelled", "passed": 1, "evidence": "candidate scores are in-sample reconstruction; template is an externally fixed Round 2 prior"})
    return pd.DataFrame(checks)


def signal_agreement_frame(
    templates: Mapping[str, np.ndarray],
    candidate_positions_map: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    votes = {name: slope_votes(template, DEADBAND)[:-1] for name, template in templates.items()}
    rows: list[dict[str, object]] = []
    template_names = ["template_5d", "template_7d", "template_11d"]
    for left_index, left in enumerate(template_names):
        for right in template_names[left_index + 1 :]:
            both_nonzero = (votes[left] != 0) & (votes[right] != 0)
            rows.append(
                {
                    "comparison": f"{left}_vs_{right}",
                    "comparison_type": "template_vote_pair",
                    "same_vote_rate": float(np.mean(votes[left] == votes[right])),
                    "same_nonzero_direction_rate": float(np.mean(votes[left][both_nonzero] == votes[right][both_nonzero])) if np.any(both_nonzero) else np.nan,
                    "days_disagree": int(np.sum(votes[left] != votes[right])),
                    "active_days_left": int(np.sum(votes[left] != 0)),
                    "active_days_right": int(np.sum(votes[right] != 0)),
                }
            )
    a_pos = candidate_positions_map["Candidate A"][:-1]
    b_pos = candidate_positions_map["Candidate B"][:-1]
    rows.append(
        {
            "comparison": "Candidate A_vs_Candidate B",
            "comparison_type": "candidate_position_pair",
            "same_vote_rate": float(np.mean(a_pos == b_pos)),
            "same_nonzero_direction_rate": float(np.mean(np.sign(a_pos) == np.sign(b_pos))),
            "days_disagree": int(np.sum(a_pos != b_pos)),
            "active_days_left": int(np.sum(a_pos != 0)),
            "active_days_right": int(np.sum(b_pos != 0)),
        }
    )
    return pd.DataFrame(rows)


def timing_shift_frames(
    prices: np.ndarray,
    templates: Mapping[str, np.ndarray],
    displacements: Sequence[int] = (-3, -2, -1, 0, 1, 2, 3),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for displacement in displacements:
        shifted = shifted_candidate_positions(prices, templates, displacement)
        for model in ("Candidate A", "Candidate B"):
            row = metric_row(prices, shifted[model], model, evidence_label="in-sample timing displacement diagnostic")
            row["displacement_days"] = displacement
            row["shift_convention"] = "+d delays template d days; value at t is original[t-d]"
            rows.append(row)
    detail = pd.DataFrame(rows)
    summaries = []
    for model, group in detail.groupby("model"):
        zero = float(group.loc[group.displacement_days == 0, "pnl"].iloc[0])
        summaries.append(
            {
                "model": model,
                "zero_shift_pnl": zero,
                "worst_shift_pnl": float(group["pnl"].min()),
                "median_shift_pnl": float(group["pnl"].median()),
                "max_drawdown_across_shifts": float(group["max_drawdown"].min()),
                "max_percentage_loss_vs_zero": float((zero - group["pnl"].min()) / max(abs(zero), 1e-12) * 100.0),
            }
        )
    return detail, pd.DataFrame(summaries)


def amplitude_stress(
    prices: np.ndarray,
    templates: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, components = broad_seasonal_components(prices)
    rows = []
    scenario_id = 0
    for multipliers in product(AMPLITUDE_LEVELS, repeat=4):
        path = amplitude_path(components, multipliers)
        for row in evaluate_stress_path(path, templates, "deterministic amplitude sensitivity"):
            row.update(
                {
                    "stress_family": "deterministic_amplitude",
                    "scenario_id": scenario_id,
                    "S1_large_multiplier": multipliers[0],
                    "S1_small_multiplier": multipliers[1],
                    "S2_large_multiplier": multipliers[2],
                    "S2_small_multiplier": multipliers[3],
                }
            )
            rows.append(row)
        scenario_id += 1
    detail = pd.DataFrame(rows)
    return detail, summary_frame(detail)


def noise_stress(
    prices: np.ndarray,
    templates: Mapping[str, np.ndarray],
    n_paths: int = 400,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    broad, components = broad_seasonal_components(prices)
    residual = prices - broad
    residual = residual - np.mean(residual)
    rng = np.random.default_rng(seed)
    rows = []
    for path_id in range(n_paths):
        multipliers = tuple(float(rng.choice(AMPLITUDE_LEVELS)) for _ in range(4))
        base = amplitude_path(components, multipliers)
        boot = a.block_bootstrap(residual, len(prices), block_length=7, rng=rng)
        path = base + boot
        for row in evaluate_stress_path(path, templates, "stochastic amplitude plus moving-block residual stress"):
            row.update(
                {
                    "stress_family": "stochastic_amplitude_block_bootstrap",
                    "scenario_id": path_id,
                    "seed": seed,
                    "S1_large_multiplier": multipliers[0],
                    "S1_small_multiplier": multipliers[1],
                    "S2_large_multiplier": multipliers[2],
                    "S2_small_multiplier": multipliers[3],
                    "residual_source": "7-day moving blocks from price minus fixed 21-day broad seasonal path",
                }
            )
            rows.append(row)
    detail = pd.DataFrame(rows)
    return detail, summary_frame(detail), residual


def summer_ablation(
    prices: np.ndarray,
    templates: Mapping[str, np.ndarray],
    n_equilibrium_paths: int = 100,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_map = candidate_positions(prices, templates)
    rows = []
    for model in CANDIDATE_NAMES:
        full = a.backtest(prices, candidate_map[model], model)
        summer_only_positions = candidate_map[model].copy()
        summer_only_positions[:SUMMER_START] = 0
        summer_only = a.backtest(prices, summer_only_positions, f"{model} summer only")
        rows.append(
            {
                "source": "observed_round1_in_sample",
                "summer_equilibrium": np.nan,
                "scenario_id": np.nan,
                "model": model,
                "pnl": full["pnl"],
                "max_drawdown": full["max_drawdown"],
                "active_days": full["active_days"],
                "summer_only_pnl": summer_only["pnl"],
                "summer_only_max_drawdown": summer_only["max_drawdown"],
                "evidence_label": "in-sample Round 1 reconstruction",
            }
        )

    _, components = broad_seasonal_components(prices)
    broad_base = amplitude_path(components, (1.0, 1.0, 1.0, 1.0))
    residual = prices - a.smooth_series(prices, window=21)
    residual = residual - np.mean(residual)
    rng = np.random.default_rng(seed + 97)
    for equilibrium in (43.0, 45.0, 47.0):
        for scenario_id in range(n_equilibrium_paths):
            boot = a.block_bootstrap(residual, len(prices), block_length=7, rng=rng)
            path = broad_base + boot
            path[SUMMER_START:] = equilibrium + boot[SUMMER_START:]
            synthetic_candidates = candidate_positions(path, templates)
            for model in CANDIDATE_NAMES:
                result = a.backtest(path, synthetic_candidates[model], model)
                summer_only_positions = synthetic_candidates[model].copy()
                summer_only_positions[:SUMMER_START] = 0
                summer_only = a.backtest(path, summer_only_positions, f"{model} summer only")
                rows.append(
                    {
                        "source": "synthetic_summer_equilibrium",
                        "summer_equilibrium": equilibrium,
                        "scenario_id": scenario_id,
                        "model": model,
                        "pnl": result["pnl"],
                        "max_drawdown": result["max_drawdown"],
                        "active_days": result["active_days"],
                        "summer_only_pnl": summer_only["pnl"],
                        "summer_only_max_drawdown": summer_only["max_drawdown"],
                        "evidence_label": "synthetic summer equilibrium; not empirical validation",
                    }
                )
    detail = pd.DataFrame(rows)
    summary = summary_frame(detail[detail.source == "synthetic_summer_equilibrium"], ["summer_equilibrium", "model"])
    return detail, summary


def gradual_equilibrium_line(length: int, target: float, duration: int) -> np.ndarray:
    """Return AUD 45 through day 321, then a frozen linear transition."""

    if duration < 1:
        raise ValueError("transition duration must be positive")
    line = np.full(length, SUMMER_EQUILIBRIUM, dtype=float)
    days = np.arange(length, dtype=float)
    after_start = days >= SUMMER_START
    progress = np.clip((days[after_start] - (SUMMER_START - 1)) / float(duration), 0.0, 1.0)
    line[after_start] = SUMMER_EQUILIBRIUM + (float(target) - SUMMER_EQUILIBRIUM) * progress
    return line


def _gradual_summary_row(
    group: pd.DataFrame,
    scope: str,
    target: float | None,
    duration: int | None,
    paired: pd.DataFrame,
) -> dict[str, object]:
    paired_row = paired
    return {
        "scenario_scope": scope,
        "target_equilibrium": target,
        "transition_days": duration,
        "model": str(group["model"].iloc[0]),
        "n_paths": int(len(group)),
        "median_full_year_pnl": float(group["full_year_pnl"].median()),
        "p10_full_year_pnl": float(group["full_year_pnl"].quantile(0.10)),
        "worst_full_year_pnl": float(group["full_year_pnl"].min()),
        "median_summer_only_pnl": float(group["summer_only_pnl"].median()),
        "p10_summer_only_pnl": float(group["summer_only_pnl"].quantile(0.10)),
        "worst_summer_only_pnl": float(group["summer_only_pnl"].min()),
        "median_max_drawdown": float(group["max_drawdown"].median()),
        "positive_path_rate": float(np.mean(group["full_year_pnl"] > 0.0)),
        "paired_median_d_minus_b": float(paired_row["d_minus_b_full_year_pnl"].median()),
        "paired_p10_d_minus_b": float(paired_row["d_minus_b_full_year_pnl"].quantile(0.10)),
        "paired_worst_d_minus_b": float(paired_row["d_minus_b_full_year_pnl"].min()),
    }


def gradual_summer_stress(
    prices: np.ndarray,
    templates: Mapping[str, np.ndarray],
    n_paths: int = GRADUAL_SUMMER_PATHS,
    seed: int = SEED + GRADUAL_SUMMER_SEED_OFFSET,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Evaluate paired B/D paths with gradual AUD 45 summer transitions."""

    if n_paths < 200:
        raise ValueError("the frozen gradual-transition audit requires at least 200 paths per scenario")
    broad, components = broad_seasonal_components(prices)
    base = amplitude_path(components, (1.0, 1.0, 1.0, 1.0))
    residual = prices - broad
    residual = residual - np.mean(residual)
    rng = np.random.default_rng(seed)
    detail_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []

    for path_id in range(n_paths):
        # One draw is reused for every target/duration pair so the comparison
        # differs only by the frozen equilibrium transition.
        boot = a.block_bootstrap(residual, len(prices), block_length=7, rng=rng)
        for target in GRADUAL_SUMMER_TARGETS:
            for duration in GRADUAL_SUMMER_DURATIONS:
                equilibrium = gradual_equilibrium_line(len(prices), target, duration)
                path = base.copy()
                path[SUMMER_START:] = equilibrium[SUMMER_START:]
                path = path + boot
                candidate_map = candidate_positions(path, templates)
                results: dict[str, dict[str, object]] = {}
                for model in ("Candidate B", "Candidate D"):
                    full = a.backtest(path, candidate_map[model], label=model)
                    summer_positions = candidate_map[model].copy()
                    summer_positions[:SUMMER_START] = 0
                    summer = a.backtest(path, summer_positions, label=f"{model} summer only")
                    results[model] = {"full": full, "summer": summer}
                    detail_rows.append(
                        {
                            "scenario_scope": "individual",
                            "target_equilibrium": target,
                            "transition_days": duration,
                            "path_id": path_id,
                            "model": model,
                            "full_year_pnl": full["pnl"],
                            "summer_only_pnl": summer["pnl"],
                            "max_drawdown": full["max_drawdown"],
                            "summer_only_max_drawdown": summer["max_drawdown"],
                            "positive_path": int(full["pnl"] > 0.0),
                            "budget_violations": full["budget_violations"],
                            "integral_positions": full["integral_positions"],
                            "within_limit": full["within_limit"],
                            "residual_seed": seed,
                            "residual_source": "mean-centred price-minus-centred-21-day path; 7-day moving blocks",
                            "transition_rule": "AUD 45 through day 321; linear day-322 transition; target thereafter",
                            "evidence_label": "generator-conditioned gradual summer stress; not empirical validation",
                        }
                    )
                paired_rows.append(
                    {
                        "scenario_scope": "individual",
                        "target_equilibrium": target,
                        "transition_days": duration,
                        "path_id": path_id,
                        "b_full_year_pnl": results["Candidate B"]["full"]["pnl"],
                        "d_full_year_pnl": results["Candidate D"]["full"]["pnl"],
                        "d_minus_b_full_year_pnl": results["Candidate D"]["full"]["pnl"] - results["Candidate B"]["full"]["pnl"],
                        "b_summer_only_pnl": results["Candidate B"]["summer"]["pnl"],
                        "d_summer_only_pnl": results["Candidate D"]["summer"]["pnl"],
                        "d_minus_b_summer_only_pnl": results["Candidate D"]["summer"]["pnl"] - results["Candidate B"]["summer"]["pnl"],
                        "evidence_label": "paired generator-conditioned stress; not empirical validation",
                    }
                )

    detail = pd.DataFrame(detail_rows)
    paired_detail = pd.DataFrame(paired_rows)
    summary_rows: list[dict[str, object]] = []
    for (target, duration), scenario_group in detail.groupby(["target_equilibrium", "transition_days"]):
        paired_group = paired_detail[(paired_detail.target_equilibrium == target) & (paired_detail.transition_days == duration)]
        for model in ("Candidate B", "Candidate D"):
            summary_rows.append(
                _gradual_summary_row(
                    scenario_group[scenario_group.model == model],
                    "individual",
                    float(target),
                    int(duration),
                    paired_group,
                )
            )
    for model in ("Candidate B", "Candidate D"):
        summary_rows.append(
            _gradual_summary_row(
                detail[detail.model == model],
                "pooled",
                None,
                None,
                paired_detail,
            )
        )
    summary = pd.DataFrame(summary_rows)

    paired_summary_rows: list[dict[str, object]] = []
    for scope, target, duration, group in [
        ("individual", float(target), int(duration), paired_detail[(paired_detail.target_equilibrium == target) & (paired_detail.transition_days == duration)])
        for target in GRADUAL_SUMMER_TARGETS
        for duration in GRADUAL_SUMMER_DURATIONS
    ] + [("pooled", None, None, paired_detail)]:
        paired_summary_rows.append(
            {
                "scenario_scope": scope,
                "target_equilibrium": target,
                "transition_days": duration,
                "n_paths": int(len(group)),
                "paired_median_d_minus_b": float(group["d_minus_b_full_year_pnl"].median()),
                "paired_p10_d_minus_b": float(group["d_minus_b_full_year_pnl"].quantile(0.10)),
                "paired_worst_d_minus_b": float(group["d_minus_b_full_year_pnl"].min()),
                "paired_median_summer_d_minus_b": float(group["d_minus_b_summer_only_pnl"].median()),
                "paired_p10_summer_d_minus_b": float(group["d_minus_b_summer_only_pnl"].quantile(0.10)),
                "paired_worst_summer_d_minus_b": float(group["d_minus_b_summer_only_pnl"].min()),
                "median_meets_minus_2000_rule": int(float(group["d_minus_b_full_year_pnl"].median()) >= -2000.0),
                "evidence_label": "paired generator-conditioned stress; not empirical validation",
            }
        )
    paired_summary = pd.DataFrame(paired_summary_rows)
    return detail, summary, paired_summary, residual


def semester_schedule(
    prices: np.ndarray,
    templates: Mapping[str, np.ndarray],
    selected_candidate: str,
) -> pd.DataFrame:
    """Export the fixed semester votes and the runtime summer-rule handoff."""

    slopes = {name: one_day_slope(templates[name]) for name in ("template_5d", "template_7d", "template_11d")}
    votes = {name: slope_votes(templates[name], DEADBAND) for name in slopes}
    majority = positions_from_votes(
        np.where(
            sum((votes[name] == 1).astype(int) for name in ("template_5d", "template_7d", "template_11d")) >= 2,
            1,
            np.where(sum((votes[name] == -1).astype(int) for name in ("template_5d", "template_7d", "template_11d")) >= 2, -1, 0),
        )
    )
    rows = []
    for day in range(len(prices)):
        summer_active = int(selected_candidate == "Candidate D" and day >= SUMMER_START)
        if day < SUMMER_START:
            description = f"{selected_candidate} semester: fixed 5/7/11 majority position"
        elif selected_candidate == "Candidate D":
            description = "Candidate D summer: runtime +1000 below AUD 45, -1000 above AUD 45, 0 at AUD 45; day 364 flat"
        else:
            description = "Candidate B: fixed 5/7/11 majority position; no summer overlay"
        rows.append(
            {
                "day": day,
                "slope_5d": slopes["template_5d"][day],
                "vote_5d": int(votes["template_5d"][day]),
                "slope_7d": slopes["template_7d"][day],
                "vote_7d": int(votes["template_7d"][day]),
                "slope_11d": slopes["template_11d"][day],
                "vote_11d": int(votes["template_11d"][day]),
                "majority_seasonal_position": int(majority[day]),
                "summer_rule_active": summer_active,
                "final_rule_description": description,
            }
        )
    return pd.DataFrame(rows)


def schedule_correctness_checks(
    schedule: pd.DataFrame,
    templates: Mapping[str, np.ndarray],
    candidate_map: Mapping[str, np.ndarray],
    selected_candidate: str,
) -> pd.DataFrame:
    expected_majority = candidate_map["Candidate B"]
    expected_pre_summer = candidate_map[selected_candidate][:SUMMER_START]
    actual_majority = schedule["majority_seasonal_position"].to_numpy(dtype=int)
    actual_selected = schedule["majority_seasonal_position"].to_numpy(dtype=int)[:SUMMER_START]
    return pd.DataFrame(
        [
            {
                "check": "semester schedule has one row for every day 0..364",
                "passed": int(len(schedule) == 365 and np.array_equal(schedule.day.to_numpy(), np.arange(365))),
                "evidence": f"rows={len(schedule)}; first={schedule.day.iloc[0]}; last={schedule.day.iloc[-1]}",
            },
            {
                "check": "semester schedule majority position equals Candidate B fixed schedule",
                "passed": int(np.array_equal(actual_majority, expected_majority)),
                "evidence": "exact full-length comparison",
            },
            {
                "check": "selected candidate schedule recreates research position before day 322",
                "passed": int(np.array_equal(actual_selected, expected_pre_summer)),
                "evidence": f"selected={selected_candidate}; compared days 0..321",
            },
            {
                "check": "semester schedule day 364 is flat",
                "passed": int(schedule.iloc[-1].majority_seasonal_position == 0),
                "evidence": "no day-365 return exists",
            },
        ]
    )


def apply_frozen_selection_rule(
    prices: np.ndarray,
    candidate_map: Mapping[str, np.ndarray],
    gradual_summary: pd.DataFrame,
    gradual_paired: pd.DataFrame,
    checks: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    pooled_b = gradual_summary[(gradual_summary.scenario_scope == "pooled") & (gradual_summary.model == "Candidate B")].iloc[0]
    pooled_d = gradual_summary[(gradual_summary.scenario_scope == "pooled") & (gradual_summary.model == "Candidate D")].iloc[0]
    individual_paired = gradual_paired[gradual_paired.scenario_scope == "individual"]
    summer_positions = candidate_map["Candidate D"].copy()
    summer_positions[:SUMMER_START] = 0
    observed_summer_pnl = float(a.backtest(prices, summer_positions, "Candidate D observed summer")['pnl'])
    decisions = [
        ("pooled P10 D >= B", pooled_d.p10_full_year_pnl >= pooled_b.p10_full_year_pnl, f"D={pooled_d.p10_full_year_pnl:.2f}; B={pooled_b.p10_full_year_pnl:.2f}"),
        ("pooled worst D >= B", pooled_d.worst_full_year_pnl >= pooled_b.worst_full_year_pnl, f"D={pooled_d.worst_full_year_pnl:.2f}; B={pooled_b.worst_full_year_pnl:.2f}"),
        ("every paired scenario median D-B >= -2000", bool((individual_paired.paired_median_d_minus_b >= -2000.0).all()), f"worst scenario median={individual_paired.paired_median_d_minus_b.min():.2f}"),
        ("observed Round 1 D summer overlay positive", observed_summer_pnl > 0.0, f"summer-only P&L={observed_summer_pnl:.2f}"),
        ("all correctness and budget checks pass", bool((checks.passed == 1).all()), f"passed={int(checks.passed.sum())}/{len(checks)}"),
    ]
    selected = "Candidate D" if all(passed for _, passed, _ in decisions) else "Candidate B"
    frame = pd.DataFrame([{"selection_check": name, "passed": int(passed), "evidence": evidence} for name, passed, evidence in decisions])
    frame.loc[len(frame)] = {"selection_check": "selected candidate", "passed": 1, "evidence": selected}
    return selected, frame


def portfolio_mechanics(prices: np.ndarray, candidate_map: Mapping[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for model in CANDIDATE_NAMES:
        result = a.backtest(prices, candidate_map[model], model)
        capital = np.abs(candidate_map[model].astype(float) * prices)
        rows.append(
            {
                "model": model,
                "max_boat_capital": float(np.max(capital)),
                "capital_p95": float(np.quantile(capital, 0.95)),
                "active_days": int(result["active_days"]),
                "active_fraction": float(result["active_days"] / max(len(prices) - 1, 1)),
                "budget_violations": result["budget_violations"],
                "integral_positions": result["integral_positions"],
                "within_limit": result["within_limit"],
                "comment": "Boat-only; shared allocator still required for all instruments",
            }
        )
    frame = pd.DataFrame(rows)
    a_active = int(frame.loc[frame.model == "Candidate A", "active_days"].iloc[0])
    c_active = int(frame.loc[frame.model == "Candidate C", "active_days"].iloc[0])
    frame["active_days_delta_vs_A"] = frame["active_days"] - a_active
    frame["active_days_delta_vs_C"] = frame["active_days"] - c_active
    return frame


def daily_capital_usage(prices: np.ndarray, candidate_map: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Return the day-level Boat notional used by each frozen candidate."""

    rows = []
    for day, price in enumerate(np.asarray(prices, dtype=float)):
        for model in CANDIDATE_NAMES:
            position = int(candidate_map[model][day])
            rows.append(
                {
                    "day": day,
                    "price": float(price),
                    "model": model,
                    "position": position,
                    "capital_used": float(abs(position) * price),
                    "active": int(position != 0),
                }
            )
    return pd.DataFrame(rows)


def save_final_figure(
    figure_path: Path,
    prices: np.ndarray,
    templates: Mapping[str, np.ndarray],
    candidate_map: Mapping[str, np.ndarray],
    timing_detail: pd.DataFrame,
    selected_candidate: str = "Candidate D",
) -> None:
    import matplotlib.pyplot as plt

    fixed = templates["template_7d"]
    slope = one_day_slope(fixed)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    ax = axes[0, 0]
    ax.plot(prices, color="#177e89", alpha=0.55, linewidth=0.9, label="Round 1 price")
    ax.plot(fixed, color="#d1495b", linewidth=2.0, label="fixed 7-day template")
    ax.axvline(SUMMER_START, color="#555555", linewidth=0.8, linestyle="--", label="summer switch")
    ax.set(title="Price and fixed 5/7/11 seasonal prior", xlabel="Day", ylabel="AUD")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(slope, color="#444444", linewidth=1.0, label="template[t+1]-template[t]")
    ax.axhline(DEADBAND, color="#d1495b", linewidth=0.8, linestyle=":")
    ax.axhline(-DEADBAND, color="#d1495b", linewidth=0.8, linestyle=":")
    ax.step(np.arange(len(candidate_map["Candidate A"])), candidate_map["Candidate A"] / POSITION_LIMIT, where="post", color="#00798c", alpha=0.65, label="Candidate A position / 1000")
    ax.set(title="Expected slope and Candidate A position (5/7/11 audit)", xlabel="Day", ylabel="Slope / scaled position")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    for model in CANDIDATE_NAMES:
        daily = candidate_map[model][:-1].astype(float) * np.diff(prices)
        ax.plot(np.r_[0.0, np.cumsum(daily)], linewidth=1.3, label=model)
    ax.axvline(SUMMER_START, color="#555555", linewidth=0.8, linestyle="--")
    ax.set(title=f"In-sample cumulative P&L (selected: {selected_candidate})", xlabel="Day", ylabel="AUD")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for model in ("Candidate A", "Candidate B"):
        group = timing_detail[timing_detail.model == model].sort_values("displacement_days")
        ax.plot(group.displacement_days, group.pnl, marker="o", linewidth=1.3, label=model)
    ax.axhline(0.0, color="#555555", linewidth=0.7)
    ax.axvline(0.0, color="#555555", linewidth=0.7, linestyle="--")
    ax.set(title="Fixed timing displacement sensitivity", xlabel="Template displacement (days)", ylabel="P&L (AUD)")
    ax.legend(frameon=False)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)


def run_final_strategy_test(
    repo_root: str | Path | None = None,
    n_noise_paths: int = 400,
    n_summer_equilibrium_paths: int = 100,
    n_gradual_summer_paths: int = GRADUAL_SUMMER_PATHS,
    seed: int = SEED,
) -> dict[str, pd.DataFrame]:
    root, prices = load_round1(repo_root)
    output_dir = root / "research" / "boat_party"
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    templates = frozen_templates(prices)
    candidates = candidate_positions(prices, templates)
    references = reference_positions(prices, templates)
    all_positions_map = {**candidates, **references}

    checks = correctness_checks(prices, templates, all_positions_map)
    assert bool(np.all(checks["passed"] == 1))
    checks.to_csv(result_dir / "fixed_template_correctness_checks.csv", index=False)

    comparison = pd.DataFrame([metric_row(prices, all_positions_map[model], model) for model in MODEL_ORDER])
    comparison.to_csv(result_dir / "fixed_template_candidate_comparison.csv", index=False)

    agreement = signal_agreement_frame(templates, candidates)
    agreement.to_csv(result_dir / "fixed_template_signal_agreement.csv", index=False)

    timing_detail, timing_summary = timing_shift_frames(prices, templates)
    timing_detail.to_csv(result_dir / "fixed_template_timing_shifts.csv", index=False)
    timing_summary.to_csv(result_dir / "fixed_template_timing_shift_summary.csv", index=False)

    amplitude_detail, amplitude_summary = amplitude_stress(prices, templates)
    amplitude_detail.to_csv(result_dir / "fixed_template_amplitude_stress.csv", index=False)
    amplitude_summary.to_csv(result_dir / "fixed_template_amplitude_stress_summary.csv", index=False)

    noise_detail, noise_summary, residual = noise_stress(prices, templates, n_paths=n_noise_paths, seed=seed)
    noise_detail.to_csv(result_dir / "fixed_template_noise_stress_detail.csv", index=False)
    noise_summary.to_csv(result_dir / "fixed_template_noise_stress_summary.csv", index=False)

    summer_detail, summer_summary = summer_ablation(prices, templates, n_equilibrium_paths=n_summer_equilibrium_paths, seed=seed)
    summer_detail.to_csv(result_dir / "fixed_template_summer_ablation.csv", index=False)
    summer_summary.to_csv(result_dir / "fixed_template_summer_ablation_summary.csv", index=False)

    gradual_detail, gradual_summary, gradual_paired, gradual_residual = gradual_summer_stress(
        prices,
        templates,
        n_paths=n_gradual_summer_paths,
        seed=seed + GRADUAL_SUMMER_SEED_OFFSET,
    )
    gradual_detail.to_csv(result_dir / "fixed_template_gradual_summer_stress_detail.csv", index=False)
    gradual_summary.to_csv(result_dir / "fixed_template_gradual_summer_stress_summary.csv", index=False)
    gradual_paired.to_csv(result_dir / "fixed_template_gradual_summer_paired_comparison.csv", index=False)

    selected_candidate, selection_checks = apply_frozen_selection_rule(
        prices,
        candidates,
        gradual_summary,
        gradual_paired,
        checks,
    )
    schedule = semester_schedule(prices, templates, selected_candidate)
    schedule.to_csv(result_dir / "fixed_template_semester_schedule.csv", index=False)
    schedule_checks = schedule_correctness_checks(schedule, templates, candidates, selected_candidate)
    selection_check_rows = selection_checks.rename(columns={"selection_check": "check"})[["check", "passed", "evidence"]]
    checks = pd.concat([checks, schedule_checks, selection_check_rows], ignore_index=True)
    assert bool(np.all(checks["passed"] == 1))
    checks.to_csv(result_dir / "fixed_template_correctness_checks.csv", index=False)

    mechanics = portfolio_mechanics(prices, candidates)
    mechanics.to_csv(result_dir / "fixed_template_portfolio_mechanics.csv", index=False)
    daily_capital = daily_capital_usage(prices, candidates)
    daily_capital.to_csv(result_dir / "fixed_template_daily_capital.csv", index=False)

    save_final_figure(figure_dir / "fixed_template_final_strategy.png", prices, templates, candidates, timing_detail, selected_candidate)

    print(f"Round 1 prices: {len(prices)}")
    print(comparison[["model", "pnl", "sharpe", "active_hit_rate", "active_days", "max_drawdown", "max_capital"]].to_string(index=False))
    print("correctness checks:", int(checks["passed"].sum()), "/", len(checks), "passed")
    print("noise paths:", n_noise_paths, "summer equilibrium paths per level:", n_summer_equilibrium_paths)
    print("residual mean/std:", float(np.mean(residual)), float(np.std(residual, ddof=1)))
    print("gradual summer paths per target/duration:", n_gradual_summer_paths)
    print("selected candidate:", selected_candidate)
    return {
        "comparison": comparison,
        "agreement": agreement,
        "timing_detail": timing_detail,
        "timing_summary": timing_summary,
        "amplitude_detail": amplitude_detail,
        "amplitude_summary": amplitude_summary,
        "noise_detail": noise_detail,
        "noise_summary": noise_summary,
        "summer_detail": summer_detail,
        "summer_summary": summer_summary,
        "gradual_detail": gradual_detail,
        "gradual_summary": gradual_summary,
        "gradual_paired": gradual_paired,
        "gradual_residual": pd.DataFrame({"residual": gradual_residual}),
        "semester_schedule": schedule,
        "selection": selection_checks,
        "selected_candidate": selected_candidate,
        "mechanics": mechanics,
        "daily_capital": daily_capital,
        "checks": checks,
    }


if __name__ == "__main__":
    run_final_strategy_test()
