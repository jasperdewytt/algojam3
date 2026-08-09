"""Focused, causal validation for the Fintech paper follow-up.

This file deliberately starts from the timing question rather than reopening
the earlier EWMA grid.  The existing research files are imported for the
tested benchmark metrics and fixed candidate family; all new model logic is
in :mod:`paper_models`.

PELT and state-label shifts are offline diagnostics only.  Every position
builder used in a P&L calculation receives a prefix ending before the change
on which its position earns.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULT_DIR = HERE / "results"
FIGURE_DIR = HERE / "figures"
DATA_PATH = ROOT / "trader_interface" / "data" / "Fintech Token_price_history.csv"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.fintech_token.fintech_models import (  # noqa: E402
    LIMIT,
    constant_position,
    ewma_ensemble_positions,
    ewma_regime,
    simple_momentum,
    simple_reversal,
)
from research.fintech_token.fintech_validation import (  # noqa: E402
    DIAGONAL_ENSEMBLE,
    fixed_candidate_family,
    load_fintech_data,
    max_drawdown,
    moving_block_sample,
    policy_metrics,
    pnl_without_largest_jumps,
)
from research.fintech_token.paper_followup.paper_models import (  # noqa: E402
    PRIMARY_CONFIGS,
    asymmetric_entry_exit_positions,
    asymmetric_entry_exit_state,
    bocpd_reset_ewma_positions,
    bocpd_student_t,
    causal_ensemble_positions,
    causal_ensemble_state,
    delayed_execution_positions,
    fast_slow_ratio_positions,
    fast_slow_ratio_state,
    pelt_cost_inputs,
    pelt_segment,
    regime_events,
    shift_state_labels,
    shifted_state_positions,
)


# All additions were declared before inspecting their results.
RATIO_CONFIGS = (
    (0.80, 0.95, 0.80, 30),
    (0.80, 0.97, 0.80, 30),
    (0.85, 0.97, 0.85, 30),
)
ASYMMETRIC_CONFIGS = (
    (0.80, 0.80, 0.90, 0.70, 30),
    (0.80, 0.85, 0.95, 0.75, 30),
    (0.85, 0.80, 0.97, 0.75, 30),
)
BOCPD_CONFIGS = (
    (20, "raw", 0.20),
    (40, "raw", 0.20),
    (60, "raw", 0.20),
    (90, "raw", 0.20),
)
BOCPD_DIAGNOSTIC_MODES = ("raw", "absolute", "squared", "reversal_residual")
BOCPD_MAX_RUN = 120
TIMING_SHIFTS = tuple(range(-10, 11))
BOOTSTRAP_BLOCKS = (5, 10, 20, 40)
STARTS = (0, 60, 90, 120, 180, 240)
RESET_LENGTHS = (60, 91)
PELT_PENALTY_MULTIPLIERS = (1.0, 2.0, 4.0, 6.0)
SEED = 20260808


Builder = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _label_ratio(config: Sequence[float]) -> str:
    fast, slow, q, warmup = config
    return f"ratio_f{fast:.2f}_s{slow:.2f}_q{q:.2f}_w{int(warmup)}"


def _label_asymmetric(config: Sequence[float]) -> str:
    entry_lam, entry_q, exit_lam, exit_q, warmup = config
    return (
        f"asym_e{entry_lam:.2f}_q{entry_q:.2f}_x{exit_lam:.2f}"
        f"_q{exit_q:.2f}_w{int(warmup)}"
    )


def _label_bocpd(config: Sequence[object]) -> str:
    duration, mode, threshold = config
    return f"bocpd_{mode}_d{int(duration)}_r{float(threshold):.2f}"


def candidate_builders() -> "OrderedDict[str, Builder]":
    """Return the frozen comparison family, including simple benchmarks."""

    builders: "OrderedDict[str, Builder]" = OrderedDict()
    builders["flat"] = lambda changes, prices: constant_position(changes, 0)
    builders["simple_reversal"] = lambda changes, prices: simple_reversal(changes)
    builders["simple_momentum"] = lambda changes, prices: simple_momentum(changes)
    builders["always_long"] = lambda changes, prices: constant_position(changes, 1)
    builders["ewma_ensemble"] = lambda changes, prices: causal_ensemble_positions(
        changes, PRIMARY_CONFIGS
    )
    for config in RATIO_CONFIGS:
        name = _label_ratio(config)
        builders[name] = lambda changes, prices, config=config: fast_slow_ratio_positions(
            changes, *config
        )
    for config in ASYMMETRIC_CONFIGS:
        name = _label_asymmetric(config)
        builders[name] = (
            lambda changes, prices, config=config: asymmetric_entry_exit_positions(
                changes, *config
            )
        )
    for config in BOCPD_CONFIGS:
        name = _label_bocpd(config)
        builders[name] = lambda changes, prices, config=config: bocpd_reset_ewma_positions(
            changes,
            expected_duration=int(config[0]),
            observation_mode=str(config[1]),
            cp_threshold=float(config[2]),
            lam=0.90,
            percentile=0.80,
            warmup=30,
            max_run=BOCPD_MAX_RUN,
        )
    return builders


def new_candidate_names() -> List[str]:
    return [
        *[_label_ratio(config) for config in RATIO_CONFIGS],
        *[_label_asymmetric(config) for config in ASYMMETRIC_CONFIGS],
        *[_label_bocpd(config) for config in BOCPD_CONFIGS],
    ]


def serious_names() -> List[str]:
    return ["ewma_ensemble", *new_candidate_names()]


def quarter_indices(n: int) -> List[np.ndarray]:
    return [np.asarray(x, dtype=int) for x in np.array_split(np.arange(n), 4)]


def _quarter_sums(values: Sequence[float]) -> List[float]:
    array = np.asarray(values, dtype=float)
    return [float(chunk.sum()) for chunk in np.array_split(array, 4)]


def _finite(value: object) -> object:
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_finite(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    return value


def _save_frame(frame: pd.DataFrame, name: str) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULT_DIR / name, index=False)


def _price_path(changes: Sequence[float], starting_price: float) -> np.ndarray:
    return np.r_[float(starting_price), float(starting_price) + np.cumsum(changes)]


def incremental_metrics(
    name: str,
    positions: Sequence[int],
    changes: Sequence[float],
    prices: Sequence[float],
    reversal_positions: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Return absolute simulator metrics and paired P&L over reversal."""

    positions = np.asarray(positions, dtype=int)
    changes = np.asarray(changes, dtype=float)
    if reversal_positions is None:
        reversal_positions = simple_reversal(changes)
    reversal_positions = np.asarray(reversal_positions, dtype=int)
    if len(positions) != len(changes) or len(reversal_positions) != len(changes):
        raise ValueError("positions, changes and reversal must have equal lengths")
    row = {"model": name}
    row.update(policy_metrics(positions, changes, prices))
    candidate_pnl = positions.astype(float) * changes
    reversal_pnl = reversal_positions.astype(float) * changes
    incremental = candidate_pnl - reversal_pnl
    row["incremental_pnl"] = float(incremental.sum())
    row["incremental_hit_rate"] = float(np.mean(incremental > 0))
    row["incremental_positive_days"] = int(np.sum(incremental > 0))
    row["incremental_q1"] = _quarter_sums(incremental)[0]
    row["incremental_q2"] = _quarter_sums(incremental)[1]
    row["incremental_q3"] = _quarter_sums(incremental)[2]
    row["incremental_q4"] = _quarter_sums(incremental)[3]
    row["incremental_drawdown"] = max_drawdown(np.cumsum(incremental))
    row["causal_position_limit_ok"] = int(np.max(np.abs(positions), initial=0) <= LIMIT)
    row["causal_integral_ok"] = int(np.all(positions == np.round(positions)))
    row["candidate_minus_reversal_daily"] = incremental
    return row


def _strip_daily(row: Mapping[str, object]) -> Dict[str, object]:
    return {key: value for key, value in row.items() if key != "candidate_minus_reversal_daily"}


def timing_reproduction(changes: np.ndarray, prices: np.ndarray) -> Dict[str, object]:
    state, member_states, member_vols, member_cutoffs = causal_ensemble_state(
        changes, PRIMARY_CONFIGS
    )
    positions = causal_ensemble_positions(changes, PRIMARY_CONFIGS)
    reversal = simple_reversal(changes)
    candidate_pnl = float(np.sum(positions * changes))
    reversal_pnl = float(np.sum(reversal * changes))
    events = regime_events(state)
    runs = []
    for event_id, entry in enumerate(events["entries"], 1):
        possible_exits = events["exits"][events["exits"] > entry]
        exit_index = int(possible_exits[0] - 1) if len(possible_exits) else len(changes) - 1
        daily_increment = (positions - reversal) * changes
        runs.append(
            {
                "episode": event_id,
                "entry": int(entry),
                "exit": exit_index,
                "days": int(exit_index - entry + 1),
                "incremental_pnl": float(daily_increment[entry : exit_index + 1].sum()),
            }
        )
    q_inc = _quarter_sums((positions - reversal) * changes)
    return {
        "prices": int(len(prices)),
        "changes": int(len(changes)),
        "reversal_pnl": reversal_pnl,
        "ewma_pnl": candidate_pnl,
        "incremental_pnl": candidate_pnl - reversal_pnl,
        "volatile_days": int(np.sum(state == 1)),
        "differing_days": int(np.sum(positions != reversal)),
        "quarter_incremental": q_inc,
        "entries": [int(x) for x in events["entries"]],
        "exits": [int(x - 1) for x in events["exits"]],
        "episodes": runs,
        "state": state,
        "member_states": member_states,
        "member_vols": member_vols,
        "member_cutoffs": member_cutoffs,
    }


def event_study(
    changes: np.ndarray,
    prices: np.ndarray,
    state: np.ndarray,
    window: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Make event-level and grouped entry/exit studies."""

    positions = causal_ensemble_positions(changes, PRIMARY_CONFIGS)
    reversal = simple_reversal(changes)
    daily_increment = (positions - reversal) * changes
    events = regime_events(state)
    rows: List[Dict[str, object]] = []
    for event_type in ("entry", "exit"):
        indices = events["entries"] if event_type == "entry" else events["exits"]
        for event_id, event_index in enumerate(indices, 1):
            event_index = int(event_index)
            running = 0.0
            for relative in range(-window, window + 1):
                index = event_index + relative
                if 0 <= index < len(changes):
                    inc = float(daily_increment[index])
                    running += inc
                    continuation = (
                        float(np.sign(changes[index]) * changes[index + 1])
                        if index + 1 < len(changes)
                        else float("nan")
                    )
                    rows.append(
                        {
                            "event_type": event_type,
                            "event": event_id,
                            "event_index": event_index,
                            "relative_day": relative,
                            "index": index,
                            "price": float(prices[index]),
                            "change": float(changes[index]),
                            "continuation_outcome": continuation,
                            "incremental_pnl": inc,
                            "cumulative_incremental_from_window": running,
                            "state": int(state[index]),
                        }
                    )
    event_frame = pd.DataFrame(rows)
    grouped = (
        event_frame.groupby(["event_type", "relative_day"], as_index=False)
        .agg(
            events=("event", "nunique"),
            mean_change=("change", "mean"),
            mean_continuation=("continuation_outcome", "mean"),
            mean_incremental_pnl=("incremental_pnl", "mean"),
            mean_cumulative_incremental=(
                "cumulative_incremental_from_window",
                "mean",
            ),
        )
        .sort_values(["event_type", "relative_day"])
    )
    _save_frame(event_frame, "event_study.csv")
    _save_frame(grouped, "event_study_grouped.csv")
    return event_frame, grouped


def lambda_timing_table(changes: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    reversal = simple_reversal(changes)
    for lam in (0.80, 0.85, 0.90, 0.95, 0.97):
        state, _, _ = ewma_regime(changes, lam, 0.80, 30)
        positions = causal_ensemble_positions(changes, ((lam, 0.80, 30),))
        events = regime_events(state)
        inc = (positions - reversal) * changes
        false_runs = 0
        false_days = 0
        for entry in events["entries"]:
            exits = events["exits"][events["exits"] > entry]
            stop = int(exits[0] - 1) if len(exits) else len(changes) - 1
            run_inc = float(inc[entry : stop + 1].sum())
            if run_inc <= 0:
                false_runs += 1
                false_days += stop - int(entry) + 1
        rows.append(
            {
                "lambda": lam,
                "percentile": 0.80,
                "warmup": 30,
                "incremental_pnl": float(inc.sum()),
                "pnl": float(np.sum(positions * changes)),
                "volatile_days": int(np.sum(state == 1)),
                "entries": int(len(events["entries"])),
                "exits": int(len(events["exits"])),
                "false_runs": int(false_runs),
                "false_days": int(false_days),
                "entry_indices": ";".join(str(int(x)) for x in events["entries"]),
                "exit_indices": ";".join(str(int(x - 1)) for x in events["exits"]),
            }
        )
    frame = pd.DataFrame(rows)
    _save_frame(frame, "lambda_timing.csv")
    return frame


def timing_shift_table(changes: np.ndarray, state: np.ndarray) -> pd.DataFrame:
    reversal = simple_reversal(changes)
    rows = []
    for shift in TIMING_SHIFTS:
        shifted_state = np.asarray(shift_state_labels(state, shift), dtype=int)
        positions = shifted_state_positions(changes, state, shift)
        inc = (positions - reversal) * changes
        rows.append(
            {
                "shift_days": shift,
                "interpretation": (
                    "oracle_early_impossible" if shift < 0 else
                    "causal_delay_diagnostic" if shift > 0 else "causal_benchmark"
                ),
                "pnl": float(np.sum(positions * changes)),
                "incremental_pnl": float(inc.sum()),
                "q1_incremental": _quarter_sums(inc)[0],
                "q2_incremental": _quarter_sums(inc)[1],
                "q3_incremental": _quarter_sums(inc)[2],
                "q4_incremental": _quarter_sums(inc)[3],
                "volatile_days": int(np.sum(shifted_state == 1)),
            }
        )
    frame = pd.DataFrame(rows)
    _save_frame(frame, "state_shifts.csv")
    return frame


def pelt_diagnostics(changes: np.ndarray, ewma_state: np.ndarray) -> pd.DataFrame:
    rows = []
    n = len(changes)
    ewma_events = regime_events(ewma_state)
    ewma_boundaries = np.r_[ewma_events["entries"], ewma_events["exits"] - 1]
    for cost_name in ("variance", "absolute_mean", "ar1"):
        values = pelt_cost_inputs(changes, cost_name)
        base = math.log(max(len(values), 2))
        for multiplier in PELT_PENALTY_MULTIPLIERS:
            penalty = multiplier * base
            boundaries = pelt_segment(values, cost_name, penalty, min_segment=20)
            for boundary in boundaries:
                nearest_gap = (
                    int(np.min(np.abs(ewma_boundaries - boundary)))
                    if len(ewma_boundaries)
                    else None
                )
                rows.append(
                    {
                        "cost": cost_name,
                        "penalty_multiplier": multiplier,
                        "penalty": penalty,
                        "boundary": int(boundary),
                        "nearest_ewma_gap_days": nearest_gap,
                    }
                )
            if not boundaries:
                rows.append(
                    {
                        "cost": cost_name,
                        "penalty_multiplier": multiplier,
                        "penalty": penalty,
                        "boundary": None,
                        "nearest_ewma_gap_days": None,
                    }
                )
    frame = pd.DataFrame(rows)
    _save_frame(frame, "pelt_boundaries.csv")
    return frame


def bocpd_diagnostics(changes: np.ndarray) -> pd.DataFrame:
    rows = []
    for mode in BOCPD_DIAGNOSTIC_MODES:
        for duration in (20, 40, 60, 90):
            result = bocpd_student_t(
                changes,
                expected_duration=duration,
                observation_mode=mode,
                prior_window=30,
                max_run=BOCPD_MAX_RUN,
            )
            short = result.short_run_probability
            usable = short[31:]
            triggers = np.flatnonzero(short >= 0.20)
            triggers = triggers[triggers > 30]
            rows.append(
                {
                    "observation": mode,
                    "expected_duration": duration,
                    "hazard": 1.0 / duration,
                    "max_raw_reset_probability": float(np.max(result.change_probability)),
                    "short_run_q50": float(np.quantile(usable, 0.50)),
                    "short_run_q95": float(np.quantile(usable, 0.95)),
                    "short_run_max": float(np.max(usable)),
                    "short_run_trigger_count_at_020": int(len(triggers)),
                    "short_run_trigger_indices": ";".join(str(int(x)) for x in triggers),
                    "minimum_posterior_mean_run": float(np.min(result.run_length_mean[31:])),
                    "minimum_run_index": int(np.argmin(result.run_length_mean[31:]) + 31),
                }
            )
    frame = pd.DataFrame(rows)
    _save_frame(frame, "bocpd_diagnostics.csv")
    return frame


def candidate_result_table(
    builders: Mapping[str, Builder], changes: np.ndarray, prices: np.ndarray
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    reversal = simple_reversal(changes)
    positions: Dict[str, np.ndarray] = {}
    rows = []
    for name, builder in builders.items():
        path = np.asarray(builder(changes, prices), dtype=int)
        positions[name] = path
        rows.append(_strip_daily(incremental_metrics(name, path, changes, prices, reversal)))
    frame = pd.DataFrame(rows)
    frame = frame.sort_values("incremental_pnl", ascending=False).reset_index(drop=True)
    _save_frame(frame, "candidate_results.csv")
    return frame, positions


def episode_results(
    names: Iterable[str],
    positions: Mapping[str, np.ndarray],
    changes: np.ndarray,
    state: np.ndarray,
) -> pd.DataFrame:
    reversal = simple_reversal(changes)
    inc_by_name = {name: (positions[name] - reversal) * changes for name in names}
    events = regime_events(state)
    rows = []
    for episode, entry in enumerate(events["entries"], 1):
        exits = events["exits"][events["exits"] > entry]
        stop = int(exits[0] - 1) if len(exits) else len(changes) - 1
        for name, inc in inc_by_name.items():
            value = float(inc[int(entry) : stop + 1].sum())
            rows.append(
                {
                    "model": name,
                    "episode": episode,
                    "entry": int(entry),
                    "exit": stop,
                    "days": stop - int(entry) + 1,
                    "incremental_pnl": value,
                    "classification": "positive_sustained" if value > 0 else "false_or_unhelpful",
                }
            )
    frame = pd.DataFrame(rows)
    _save_frame(frame, "candidate_episode_results.csv")
    return frame


def exclusions(
    names: Iterable[str],
    positions: Mapping[str, np.ndarray],
    changes: np.ndarray,
    state: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reversal = simple_reversal(changes)
    quarter_two = quarter_indices(len(changes))[1]
    events = regime_events(state)
    episode_rows = []
    for episode, entry in enumerate(events["entries"], 1):
        exits = events["exits"][events["exits"] > entry]
        stop = int(exits[0] - 1) if len(exits) else len(changes) - 1
        for name in names:
            inc = (positions[name] - reversal) * changes
            episode_rows.append(
                {
                    "model": name,
                    "removed_episode": episode,
                    "entry": int(entry),
                    "exit": stop,
                    "full_incremental": float(inc.sum()),
                    "incremental_excluding_episode": float(
                        inc.sum() - inc[int(entry) : stop + 1].sum()
                    ),
                }
            )
    q2_rows = []
    for name in names:
        inc = (positions[name] - reversal) * changes
        pnl = positions[name] * changes
        q2_rows.append(
            {
                "model": name,
                "full_incremental": float(inc.sum()),
                "incremental_excluding_q2": float(inc[np.setdiff1d(np.arange(len(inc)), quarter_two)].sum()),
                "full_pnl": float(pnl.sum()),
                "pnl_excluding_q2": float(pnl[np.setdiff1d(np.arange(len(pnl)), quarter_two)].sum()),
            }
        )
    episodes = pd.DataFrame(episode_rows)
    q2 = pd.DataFrame(q2_rows)
    _save_frame(episodes, "episode_exclusions.csv")
    _save_frame(q2, "q2_exclusions.csv")
    return episodes, q2


def jump_exclusions(
    names: Iterable[str], positions: Mapping[str, np.ndarray], changes: np.ndarray
) -> pd.DataFrame:
    reversal = simple_reversal(changes)
    order = np.argsort(np.abs(changes))[::-1]
    rows = []
    for name in names:
        pos = positions[name]
        pnl = pos * changes
        inc = (pos - reversal) * changes
        row: Dict[str, object] = {"model": name}
        row["full_pnl"] = float(pnl.sum())
        row["full_incremental"] = float(inc.sum())
        for count in (1, 3, 5, 10):
            keep = np.ones(len(changes), dtype=bool)
            keep[order[:count]] = False
            row[f"pnl_excluding_{count}"] = float(pnl[keep].sum())
            row[f"incremental_excluding_{count}"] = float(inc[keep].sum())
        rows.append(row)
    frame = pd.DataFrame(rows)
    _save_frame(frame, "jump_exclusions.csv")
    return frame


def chronological_resets(
    names: Iterable[str],
    builders: Mapping[str, Builder],
    changes: np.ndarray,
    prices: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for name in names:
        for start in STARTS:
            local_changes = changes[start:]
            local_prices = prices[start:]
            pos = np.asarray(builders[name](local_changes, local_prices), dtype=int)
            rev = simple_reversal(local_changes)
            inc = (pos - rev) * local_changes
            metric = policy_metrics(pos, local_changes, local_prices)
            rows.append(
                {
                    "model": name,
                    "start": int(start),
                    "n_changes": len(local_changes),
                    "pnl": float(metric["pnl"]),
                    "incremental_pnl": float(inc.sum()),
                    "q1_incremental": _quarter_sums(inc)[0],
                    "q2_incremental": _quarter_sums(inc)[1],
                    "q3_incremental": _quarter_sums(inc)[2],
                    "q4_incremental": _quarter_sums(inc)[3],
                    "active_days": int(metric["active_days"]),
                }
            )
    starts = pd.DataFrame(rows)
    reset_rows = []
    for name in names:
        for length in RESET_LENGTHS:
            for start in range(0, len(changes), length):
                stop = min(start + length, len(changes))
                local_changes = changes[start:stop]
                local_prices = prices[start : stop + 1]
                if len(local_changes) < 2:
                    continue
                pos = np.asarray(builders[name](local_changes, local_prices), dtype=int)
                rev = simple_reversal(local_changes)
                inc = (pos - rev) * local_changes
                reset_rows.append(
                    {
                        "model": name,
                        "reset_length": length,
                        "start": int(start),
                        "stop": int(stop),
                        "n_changes": len(local_changes),
                        "pnl": float(np.sum(pos * local_changes)),
                        "incremental_pnl": float(inc.sum()),
                        "active_days": int(np.sum(pos != 0)),
                    }
                )
    resets = pd.DataFrame(reset_rows)
    _save_frame(starts, "chronological_starts.csv")
    _save_frame(resets, "independent_resets.csv")
    return starts, resets


def delayed_execution_table(
    names: Iterable[str], positions: Mapping[str, np.ndarray], changes: np.ndarray
) -> pd.DataFrame:
    reversal = simple_reversal(changes)
    delayed_reversal = delayed_execution_positions(reversal)
    rows = []
    for name in names:
        delayed = delayed_execution_positions(positions[name])
        # Delay the benchmark by the same day; otherwise the audit would
        # compare delayed candidates with an impossible immediate reversal.
        inc = (delayed - delayed_reversal) * changes
        metric = policy_metrics(delayed, changes)
        rows.append(
            {
                "model": name,
                "delayed_pnl": float(metric["pnl"]),
                "delayed_incremental_pnl": float(inc.sum()),
                "delayed_reversal_pnl": float(np.sum(delayed_reversal * changes)),
                "delayed_active_days": int(metric["active_days"]),
                "original_pnl": float(np.sum(positions[name] * changes)),
                "original_incremental_pnl": float(
                    np.sum((positions[name] - reversal) * changes)
                ),
            }
        )
    frame = pd.DataFrame(rows)
    _save_frame(frame, "delayed_execution.csv")
    return frame


def future_perturbation_table(
    names: Iterable[str],
    builders: Mapping[str, Builder],
    changes: np.ndarray,
    prices: np.ndarray,
    cuts: Sequence[int] = (60, 90, 120, 180, 240, 300),
) -> pd.DataFrame:
    rows = []
    for name in names:
        original = np.asarray(builders[name](changes, prices), dtype=int)
        for cut in cuts:
            if cut >= len(changes):
                continue
            perturbed = changes.copy()
            perturbed[cut:] = -perturbed[cut:] + 17.0
            perturbed_prices = _price_path(perturbed, prices[0])
            altered = np.asarray(builders[name](perturbed, perturbed_prices), dtype=int)
            same = bool(np.array_equal(original[:cut], altered[:cut]))
            rows.append(
                {
                    "model": name,
                    "cut": int(cut),
                    "prefix_equal": int(same),
                    "prefix_length": int(cut),
                    "max_prefix_difference": int(
                        np.max(np.abs(original[:cut] - altered[:cut]), initial=0)
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    _save_frame(frame, "future_perturbation.csv")
    return frame


def _stationary_sample(values: np.ndarray, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return values.copy()
    result = np.empty(n, dtype=values.dtype)
    index = int(rng.integers(0, n))
    for t in range(n):
        if t > 0 and rng.random() < 1.0 / mean_block:
            index = int(rng.integers(0, n))
        result[t] = values[index]
        index = (index + 1) % n
    return result


def paired_bootstrap(
    names: Sequence[str],
    builders: Mapping[str, Builder],
    changes: np.ndarray,
    prices: np.ndarray,
    block_lengths: Sequence[int],
    reps: int,
    bocpd_reps: int,
    seed: int,
    stationary: bool = False,
) -> pd.DataFrame:
    """Paired bootstrap: every candidate shares every reversal path."""

    rng = np.random.default_rng(seed)
    rows = []
    bocpd_names = {name for name in names if name.startswith("bocpd_")}
    for block_length in block_lengths:
        actual_reps = int(reps)
        if stationary:
            actual_reps = int(reps)
            sampling_label = f"stationary_mean_{block_length}"
        else:
            sampling_label = f"moving_block_{block_length}"
        if bocpd_names:
            # The BOCPD path is deliberately kept separate and reports its
            # actual repetition count instead of implying equal power.
            candidate_reps = {name: (bocpd_reps if name in bocpd_names else actual_reps) for name in names}
        else:
            candidate_reps = {name: actual_reps for name in names}
        values: Dict[str, List[float]] = {name: [] for name in names}
        absolute: Dict[str, List[float]] = {name: [] for name in names}
        reversal_values: Dict[str, List[float]] = {name: [] for name in names}
        for name in names:
            for _ in range(candidate_reps[name]):
                if stationary:
                    sample = _stationary_sample(changes, float(block_length), rng)
                else:
                    sample = moving_block_sample(changes, int(block_length), rng)
                sample_prices = _price_path(sample, prices[0])
                reversal = simple_reversal(sample)
                reversal_pnl = float(np.sum(reversal * sample))
                position = np.asarray(builders[name](sample, sample_prices), dtype=int)
                candidate_pnl = float(np.sum(position * sample))
                values[name].append(candidate_pnl - reversal_pnl)
                absolute[name].append(candidate_pnl)
                reversal_values[name].append(reversal_pnl)
        for name in names:
            inc_values = np.asarray(values[name], dtype=float)
            abs_values = np.asarray(absolute[name], dtype=float)
            rev_values = np.asarray(reversal_values[name], dtype=float)
            rows.append(
                {
                    "sampling": sampling_label,
                    "block_length_or_mean": int(block_length),
                    "model": name,
                    "reps": int(len(inc_values)),
                    "mean_incremental_pnl": float(inc_values.mean()),
                    "median_incremental_pnl": float(np.quantile(inc_values, 0.50)),
                    "ci025_incremental_pnl": float(np.quantile(inc_values, 0.025)),
                    "ci975_incremental_pnl": float(np.quantile(inc_values, 0.975)),
                    "prob_increment_gt_zero": float(np.mean(inc_values > 0)),
                    "mean_candidate_pnl": float(abs_values.mean()),
                    "mean_reversal_pnl_same_paths": float(rev_values.mean()),
                }
            )
    frame = pd.DataFrame(rows)
    filename = "paired_stationary_bootstrap.csv" if stationary else "paired_moving_block_bootstrap.csv"
    _save_frame(frame, filename)
    return frame


def family_time_shuffle(
    changes: np.ndarray,
    prices: np.ndarray,
    builders: Mapping[str, Builder],
    repetitions: int,
    seed: int,
) -> Dict[str, object]:
    """Max-over-family permutation null, including every new configuration."""

    observed = fixed_candidate_family(changes, prices)
    for name in new_candidate_names():
        observed[name] = np.asarray(builders[name](changes, prices), dtype=int)
    observed_scores = {name: float(np.sum(pos * changes)) for name, pos in observed.items()}
    rng = np.random.default_rng(seed)
    selected_null: List[float] = []
    best_null: List[float] = []
    new_null: Dict[str, List[float]] = {name: [] for name in new_candidate_names()}
    for iteration in range(repetitions):
        sample = rng.permutation(changes)
        sample_prices = _price_path(sample, prices[0])
        candidates = fixed_candidate_family(sample, sample_prices)
        for name in new_candidate_names():
            candidates[name] = np.asarray(builders[name](sample, sample_prices), dtype=int)
        scores = {name: float(np.sum(pos * sample)) for name, pos in candidates.items()}
        selected_null.append(scores["ewma_diagonal_ensemble"])
        best_null.append(max(scores.values()))
        for name in new_null:
            new_null[name].append(scores[name])
        if (iteration + 1) % 25 == 0:
            print(f"family shuffle {iteration + 1}/{repetitions}", flush=True)
    observed_best = max(observed_scores.values())
    result = {
        "type": "iid_time_shuffle_family_max",
        "repetitions": int(repetitions),
        "family_count": int(len(observed_scores)),
        "selected_name": "ewma_diagonal_ensemble",
        "selected_observed": observed_scores["ewma_diagonal_ensemble"],
        "selected_p_value": float(
            (1 + sum(x >= observed_scores["ewma_diagonal_ensemble"] for x in selected_null))
            / (repetitions + 1)
        ),
        "observed_best": float(observed_best),
        "family_wise_p_value": float(
            (1 + sum(x >= observed_best for x in best_null)) / (repetitions + 1)
        ),
        "selected_null_q95": float(np.quantile(selected_null, 0.95)),
        "family_best_null_q95": float(np.quantile(best_null, 0.95)),
        "new_candidate_p_values": {
            name: float(
                (1 + sum(x >= observed_scores[name] for x in new_null[name]))
                / (repetitions + 1)
            )
            for name in new_null
        },
        "observed_scores": observed_scores,
    }
    with (RESULT_DIR / "family_time_shuffle.json").open("w", encoding="utf-8") as handle:
        json.dump(_finite(result), handle, indent=2)
    return result


def family_circular_shift(
    changes: np.ndarray,
    prices: np.ndarray,
    builders: Mapping[str, Builder],
    repetitions: int,
    seed: int,
) -> Dict[str, object]:
    """Circular shifts preserve local serial structure; they are not IID nulls."""

    observed = fixed_candidate_family(changes, prices)
    for name in new_candidate_names():
        observed[name] = np.asarray(builders[name](changes, prices), dtype=int)
    observed_scores = {name: float(np.sum(pos * changes)) for name, pos in observed.items()}
    rng = np.random.default_rng(seed)
    offsets = rng.integers(1, len(changes), size=repetitions)
    selected_null = []
    best_null = []
    for offset in offsets:
        sample = np.roll(changes, int(offset))
        sample_prices = _price_path(sample, prices[0])
        candidates = fixed_candidate_family(sample, sample_prices)
        for name in new_candidate_names():
            candidates[name] = np.asarray(builders[name](sample, sample_prices), dtype=int)
        scores = [float(np.sum(pos * sample)) for pos in candidates.values()]
        selected_null.append(
            float(np.sum(candidates["ewma_diagonal_ensemble"] * sample))
        )
        best_null.append(max(scores))
    result = {
        "type": "circular_shift_start_location",
        "repetitions": int(repetitions),
        "family_count": int(len(observed_scores)),
        "selected_name": "ewma_diagonal_ensemble",
        "selected_observed": observed_scores["ewma_diagonal_ensemble"],
        "selected_p_value": float(
            (1 + sum(x >= observed_scores["ewma_diagonal_ensemble"] for x in selected_null))
            / (repetitions + 1)
        ),
        "observed_best": float(max(observed_scores.values())),
        "family_wise_p_value": float(
            (1 + sum(x >= max(observed_scores.values()) for x in best_null))
            / (repetitions + 1)
        ),
        "selected_null_q95": float(np.quantile(selected_null, 0.95)),
        "family_best_null_q95": float(np.quantile(best_null, 0.95)),
        "interpretation": "start-location sensitivity preserving local serial structure, not IID no-predictability",
    }
    with (RESULT_DIR / "family_circular_shift.json").open("w", encoding="utf-8") as handle:
        json.dump(_finite(result), handle, indent=2)
    return result


def make_figures(
    prices: np.ndarray,
    changes: np.ndarray,
    timing: Mapping[str, object],
    grouped_events: pd.DataFrame,
    shifts: pd.DataFrame,
    pelt: pd.DataFrame,
    result_frame: pd.DataFrame,
    positions: Mapping[str, np.ndarray],
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    state = np.asarray(timing["state"], dtype=int)
    vols = timing["member_vols"]
    cutoffs = timing["member_cutoffs"]
    fast = np.asarray(vols[0], dtype=float)
    slow = np.asarray(vols[-1], dtype=float)
    cutoff_array = np.asarray(cutoffs, dtype=float)
    cutoff_count = np.sum(np.isfinite(cutoff_array), axis=0)
    threshold = np.divide(
        np.nansum(cutoff_array, axis=0),
        cutoff_count,
        out=np.full(cutoff_count.shape, np.nan, dtype=float),
        where=cutoff_count > 0,
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(np.arange(len(prices)), prices, color="black", linewidth=1)
    axes[0].set_ylabel("price")
    axes[0].set_title("Fintech Token price and causal EWMA state")
    axes[1].plot(np.arange(len(changes)), changes, color="slateblue", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("change")
    axes[2].plot(fast, label="fast EWMA vol (lambda .85)", linewidth=1)
    axes[2].plot(slow, label="slow EWMA vol (lambda .95)", linewidth=1)
    axes[2].plot(threshold, label="mean member threshold", linewidth=1, linestyle="--")
    axes[2].fill_between(
        np.arange(len(state)),
        0,
        np.maximum(np.nanmax(np.vstack([fast, slow, threshold]), axis=0), 0),
        where=state == 1,
        color="tomato",
        alpha=0.14,
        label="volatile state",
    )
    axes[2].set_ylabel("volatility")
    axes[2].set_xlabel("change index / decision day")
    axes[2].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "timing_audit.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for event_type, color in (("entry", "tab:red"), ("exit", "tab:blue")):
        data = grouped_events[grouped_events.event_type == event_type]
        axes[0].plot(data.relative_day, data.mean_continuation, label=event_type, color=color)
        axes[1].plot(
            data.relative_day,
            data.mean_cumulative_incremental,
            label=event_type,
            color=color,
        )
    axes[0].axvline(0, color="black", linewidth=0.6)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("mean signed continuation")
    axes[1].axvline(0, color="black", linewidth=0.6)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("mean cumulative incremental P&L")
    axes[1].set_xlabel("days relative to transition")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Event study around causal EWMA entries and exits")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "event_study.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(shifts.shift_days, shifts.incremental_pnl, marker="o", color="tab:purple")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvspan(-10, -0.5, color="tomato", alpha=0.10, label="oracle / impossible early shift")
    ax.set_xlabel("state-label shift (negative = earlier using future labels)")
    ax.set_ylabel("incremental P&L vs reversal")
    ax.set_title("Offline timing diagnostic; negative shifts are not causal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "state_shift_audit.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(changes, color="slateblue", linewidth=0.8, label="change")
    colors = {"variance": "tab:red", "absolute_mean": "tab:green", "ar1": "tab:orange"}
    for cost, color in colors.items():
        selected = pelt[(pelt.cost == cost) & (pelt.penalty_multiplier == 2.0)]
        for boundary in selected.boundary.dropna().astype(int):
            ax.axvline(boundary, color=color, alpha=0.55, linewidth=1, label=f"PELT {cost}" if boundary == selected.boundary.dropna().iloc[0] else None)
    for entry in timing["entries"]:
        ax.axvline(entry, color="black", linestyle="--", alpha=0.7, label="EWMA entry" if entry == timing["entries"][0] else None)
    ax.set_title("Offline PELT boundaries versus causal EWMA entries")
    ax.set_xlabel("change index")
    ax.set_ylabel("change")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pelt_boundaries.png", dpi=150)
    plt.close(fig)

    plot_names = ["simple_reversal", "ewma_ensemble"]
    new_rows = result_frame[result_frame.model.isin(new_candidate_names())]
    if len(new_rows):
        best_new = str(new_rows.iloc[0].model)
        plot_names.append(best_new)
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in plot_names:
        ax.plot(np.cumsum(positions[name] * changes), label=name)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Cumulative P&L: reversal, EWMA ensemble and best new causal candidate")
    ax.set_xlabel("change index")
    ax.set_ylabel("P&L")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "candidate_pnl.png", dpi=150)
    plt.close(fig)


def run_all(
    bootstrap_reps: int = 1000,
    bocpd_bootstrap_reps: int = 250,
    stationary_reps: int = 300,
    family_reps: int = 100,
    seed: int = SEED,
) -> Dict[str, object]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    prices, changes = load_fintech_data(DATA_PATH)
    builders = candidate_builders()
    print("loaded Fintech Token: 365 prices / 364 changes", flush=True)

    timing = timing_reproduction(changes, prices)
    print(
        f"reproduction: reversal={timing['reversal_pnl']:.0f}, "
        f"ewma={timing['ewma_pnl']:.0f}, inc={timing['incremental_pnl']:.0f}, "
        f"episodes={len(timing['episodes'])}",
        flush=True,
    )
    state = np.asarray(timing["state"], dtype=int)
    _, grouped_events = event_study(changes, prices, state)
    lambda_table = lambda_timing_table(changes)
    shifts = timing_shift_table(changes, state)
    pelt = pelt_diagnostics(changes, state)
    bocpd = bocpd_diagnostics(changes)
    result_frame, positions = candidate_result_table(builders, changes, prices)
    serious = serious_names()
    episode_frame = episode_results(serious, positions, changes, state)
    episode_exclusions, q2_exclusions = exclusions(serious, positions, changes, state)
    jumps = jump_exclusions(serious, positions, changes)
    starts, resets = chronological_resets(serious, builders, changes, prices)
    delayed = delayed_execution_table(serious, positions, changes)
    future = future_perturbation_table(list(builders), builders, changes, prices)
    print("timing, PELT, BOCPD and deterministic candidate audits complete", flush=True)

    bootstrap_core = paired_bootstrap(
        serious,
        builders,
        changes,
        prices,
        BOOTSTRAP_BLOCKS,
        reps=bootstrap_reps,
        bocpd_reps=bocpd_bootstrap_reps,
        seed=seed,
        stationary=False,
    )
    print("paired moving-block bootstrap complete", flush=True)
    bootstrap_stationary = paired_bootstrap(
        serious,
        builders,
        changes,
        prices,
        (10, 20),
        reps=stationary_reps,
        bocpd_reps=max(100, stationary_reps // 2),
        seed=seed + 1,
        stationary=True,
    )
    print("paired stationary bootstrap complete", flush=True)
    family_shuffle = family_time_shuffle(changes, prices, builders, family_reps, seed + 2)
    family_circular = family_circular_shift(changes, prices, builders, family_reps, seed + 3)
    print("family-wise nulls complete", flush=True)

    make_figures(prices, changes, timing, grouped_events, shifts, pelt, result_frame, positions)

    model_counts = {
        "candidate_builder_count": len(builders),
        "serious_candidate_count": len(serious),
        "new_candidate_count": len(new_candidate_names()),
        "old_fixed_family_count": len(fixed_candidate_family(changes, prices)),
        "combined_time_shuffle_family_count": family_shuffle["family_count"],
        "ewma_grid_not_researched_again": True,
        "bootstrap_reps_core": bootstrap_reps,
        "bootstrap_reps_bocpd": bocpd_bootstrap_reps,
        "stationary_reps_core": stationary_reps,
        "family_shuffle_reps": family_reps,
    }
    with (RESULT_DIR / "model_counts.json").open("w", encoding="utf-8") as handle:
        json.dump(_finite(model_counts), handle, indent=2)

    summary = {
        "data": {"path": str(DATA_PATH), "prices": len(prices), "changes": len(changes)},
        "reproduction": _finite({key: value for key, value in timing.items() if key not in {"state", "member_states", "member_vols", "member_cutoffs"}}),
        "primary_configs": [list(config) for config in PRIMARY_CONFIGS],
        "ratio_configs": [list(config) for config in RATIO_CONFIGS],
        "asymmetric_configs": [list(config) for config in ASYMMETRIC_CONFIGS],
        "bocpd_configs": [list(config) for config in BOCPD_CONFIGS],
        "bocpd_max_run": BOCPD_MAX_RUN,
        "model_counts": model_counts,
        "top_candidate_rows": _finite(result_frame.head(12).to_dict(orient="records")),
        "family_time_shuffle": family_shuffle,
        "family_circular_shift": family_circular,
        "future_perturbation_failures": int(np.sum(future.prefix_equal == 0)),
        "event_episodes": timing["episodes"],
        "notebook_input_files": [
            "candidate_results.csv",
            "paired_moving_block_bootstrap.csv",
            "paired_stationary_bootstrap.csv",
            "event_study_grouped.csv",
            "state_shifts.csv",
            "pelt_boundaries.csv",
            "bocpd_diagnostics.csv",
        ],
    }
    with (RESULT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_finite(summary), handle, indent=2)
    print("all isolated paper-follow-up results and figures written", flush=True)
    return summary


def finalize_existing_outputs() -> Dict[str, object]:
    """Rebuild figures/metadata after a detached long bootstrap completes."""

    prices, changes = load_fintech_data(DATA_PATH)
    builders = candidate_builders()
    timing = timing_reproduction(changes, prices)
    _, grouped_events = event_study(changes, prices, np.asarray(timing["state"], dtype=int))
    shifts = pd.read_csv(RESULT_DIR / "state_shifts.csv")
    pelt = pd.read_csv(RESULT_DIR / "pelt_boundaries.csv")
    result_frame = pd.read_csv(RESULT_DIR / "candidate_results.csv")
    positions = {
        name: np.asarray(builders[name](changes, prices), dtype=int)
        for name in serious_names()
    }
    positions["simple_reversal"] = simple_reversal(changes)
    make_figures(prices, changes, timing, grouped_events, shifts, pelt, result_frame, positions)
    moving = pd.read_csv(RESULT_DIR / "paired_moving_block_bootstrap.csv")
    stationary = pd.read_csv(RESULT_DIR / "paired_stationary_bootstrap.csv")
    family_time = json.loads((RESULT_DIR / "family_time_shuffle.json").read_text(encoding="utf-8"))
    family_circular = json.loads((RESULT_DIR / "family_circular_shift.json").read_text(encoding="utf-8"))
    future = pd.read_csv(RESULT_DIR / "future_perturbation.csv")
    model_counts = {
        "candidate_builder_count": len(builders),
        "serious_candidate_count": len(serious_names()),
        "new_candidate_count": len(new_candidate_names()),
        "old_fixed_family_count": len(fixed_candidate_family(changes, prices)),
        "combined_time_shuffle_family_count": int(family_time["family_count"]),
        "ewma_grid_not_researched_again": True,
        "bootstrap_reps_core": int(moving[moving.model == "ewma_ensemble"].reps.max()),
        "bootstrap_reps_bocpd": int(moving[moving.model.str.startswith("bocpd_")].reps.max()),
        "stationary_reps_core": int(stationary[stationary.model == "ewma_ensemble"].reps.max()),
        "family_shuffle_reps": int(family_time["repetitions"]),
        "future_perturbation_failures": int(np.sum(future.prefix_equal == 0)),
    }
    with (RESULT_DIR / "model_counts.json").open("w", encoding="utf-8") as handle:
        json.dump(_finite(model_counts), handle, indent=2)
    summary = {
        "data": {"path": str(DATA_PATH), "prices": len(prices), "changes": len(changes)},
        "reproduction": _finite(
            {
                key: value
                for key, value in timing.items()
                if key not in {"state", "member_states", "member_vols", "member_cutoffs"}
            }
        ),
        "primary_configs": [list(config) for config in PRIMARY_CONFIGS],
        "ratio_configs": [list(config) for config in RATIO_CONFIGS],
        "asymmetric_configs": [list(config) for config in ASYMMETRIC_CONFIGS],
        "bocpd_configs": [list(config) for config in BOCPD_CONFIGS],
        "bocpd_max_run": BOCPD_MAX_RUN,
        "model_counts": model_counts,
        "top_candidate_rows": _finite(result_frame.head(12).to_dict(orient="records")),
        "family_time_shuffle": family_time,
        "family_circular_shift": family_circular,
        "paired_moving_rows": int(len(moving)),
        "paired_stationary_rows": int(len(stationary)),
        "notebook_input_files": sorted(path.name for path in RESULT_DIR.glob("*.csv")),
    }
    with (RESULT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_finite(summary), handle, indent=2)
    print("existing full tables finalized with figures and summary metadata", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--bocpd-bootstrap-reps", type=int, default=250)
    parser.add_argument("--stationary-reps", type=int, default=300)
    parser.add_argument("--family-reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.finalize_only:
        finalize_existing_outputs()
        return
    run_all(
        bootstrap_reps=max(1, args.bootstrap_reps),
        bocpd_bootstrap_reps=max(1, args.bocpd_bootstrap_reps),
        stationary_reps=max(1, args.stationary_reps),
        family_reps=max(1, args.family_reps),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
