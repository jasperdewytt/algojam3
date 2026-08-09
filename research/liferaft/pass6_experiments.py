"""Pass 6A experiment runner.

The runner has three explicit result-producing phases:

``--null``
    deterministic non-pivotal IID calibration plus predeclared power checks;
``--development``
    the nine existing development cases at the four fixed exposures;
``--validation``
    the 480 consumed validation cases at the same four exposures.

Only the parent process writes result/report/manifest files.  Worker functions
are top-level and process-pool safe on Windows.  No final-suite constructor or
quarantined final module is imported or called here.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .cold_start_strategies import make_cold_start_strategy
from .pass3_scenarios import development_scenarios, validation_scenarios
from .pass4_strategies import make_pass4_strategy
from .pass6_models import EXPERT_NAMES, REWARD_BOUND
from .pass6_strategies import (
    GLOBAL_ONE_SIDED_ALPHA,
    PASS6_STRATEGY_NAMES,
    PER_GATE_ALPHA,
    make_pass6_strategy,
)
from .shadow_strategies import make_shadow_strategy
from .simulator import AgentObservation, LiferaftSimulator, SimulationResult


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "PASS6A_RESULTS.json"
REPORT_PATH = ROOT / "PASS6A_REPORT.md"
MANIFEST_PATH = ROOT / "PASS6A_MANIFEST.md"

EXPOSURES: tuple[int, ...] = (0, 150_000, 300_000, 450_000)
EVALUATION_CANDIDATES: tuple[str, ...] = (
    "flat",
    "ungated_fixed_share",
    "burnin1_markov",
    "risk50_burnin1_markov",
    "shadow8_markov",
    "fixed_checkpoint_fixed_share",
    "anytime_valid_fixed_share",
)
NEW_CANDIDATES: tuple[str, ...] = (
    "fixed_checkpoint_fixed_share",
    "anytime_valid_fixed_share",
)
PRINCIPAL_FAMILIES: tuple[str, ...] = (
    "persistent_long",
    "persistent_short",
    "balanced_random",
    "short_biased_random",
    "long_biased_random",
    "periodic",
    "reactive_mixture",
    "regime_change",
    "gradual_drift",
    "startup_zero_history",
    "history_rules",
    "margin_mixture",
)
POWER_PATHS = 2_000
NULL_PATHS = 10_000
NULL_CONFIDENCE_DELTA = 0.01
EVAL_BATCH_SIZE = 32
NULL_BATCH_SIZE = 100


@dataclass(frozen=True)
class CaseTask:
    phase: str
    scenario_index: int
    candidate: str
    exposure: int

    @property
    def key(self) -> tuple[object, ...]:
        return (self.phase, self.scenario_index, self.candidate, self.exposure)


def stable_seed(*parts: object) -> int:
    """Derive a deterministic seed without Python's process-randomised hash."""

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


_SUITE_CACHE: dict[str, tuple[Any, ...]] = {}


def _scenario_suite(phase: str) -> tuple[Any, ...]:
    if phase not in {"development", "validation"}:
        raise ValueError(f"unsupported Pass 6 suite {phase!r}")
    if phase not in _SUITE_CACHE:
        # These are the existing non-final constructors only.  The locked
        # final constructor is intentionally absent from this runner.
        _SUITE_CACHE[phase] = (
            development_scenarios()
            if phase == "development"
            else validation_scenarios()
        )
    return _SUITE_CACHE[phase]


def _make_candidate(name: str, exposure: int) -> object:
    if name in PASS6_STRATEGY_NAMES:
        return make_pass6_strategy(name, other_portfolio_exposure=exposure)
    if name in {"flat", "burnin1_markov"}:
        return make_cold_start_strategy(name)
    if name == "risk50_burnin1_markov":
        return make_pass4_strategy(name, other_portfolio_exposure=exposure)
    if name == "shadow8_markov":
        return make_shadow_strategy(name, other_portfolio_exposure=exposure)
    raise KeyError(f"unknown evaluation candidate {name!r}")


def _quartile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(fraction * (len(ordered) - 1))
    return ordered[index]


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def _best_low_switch_reward(
    reward_history: Sequence[Mapping[str, object]],
    *,
    max_switches: int = 2,
) -> float:
    """Best expert sequence with at most two switches, computed causally.

    This is a hindsight diagnostic only.  It is not supplied to the master or
    used for selection.
    """

    if not reward_history:
        return 0.0
    dp: list[dict[str, float]] = [
        {name: float(reward_history[0]["expert_rewards"][name]) for name in EXPERT_NAMES}
    ] + [{} for _ in range(max_switches)]
    for row in reward_history[1:]:
        rewards = row["expert_rewards"]
        next_dp = [dict() for _ in range(max_switches + 1)]
        for switches in range(max_switches + 1):
            for name in EXPERT_NAMES:
                stay = dp[switches].get(name, float("-inf"))
                switch = max(
                    (value for other, value in dp[switches - 1].items() if other != name),
                    default=float("-inf"),
                ) if switches else float("-inf")
                next_dp[switches][name] = max(stay, switch) + float(rewards[name])
        dp = next_dp
    return max((value for state in dp for value in state.values()), default=0.0)


def _extract_stop_overshoots(diagnostics: Mapping[str, object]) -> list[int]:
    overshoots: list[int] = []
    for event in diagnostics.get("stop_events", []) or []:
        if isinstance(event, Mapping):
            overshoots.append(int(event.get("overshoot", 0)))
    loss_event = diagnostics.get("loss_stop_event")
    if loss_event is not None and not overshoots:
        value = getattr(loss_event, "loss_limit_overshoot", 0)
        overshoots.append(int(value or 0))
    return overshoots


def _activation_days_from_diagnostics(diagnostics: Mapping[str, object]) -> list[int]:
    gate = diagnostics.get("gate")
    if isinstance(gate, Mapping):
        days = gate.get("activation_days", [])
        if isinstance(days, Sequence) and not isinstance(days, (str, bytes)):
            return [int(day) for day in days]
    events = diagnostics.get("activation_events", [])
    if isinstance(events, Sequence):
        result: list[int] = []
        for event in events:
            if isinstance(event, Mapping) and event.get("day") is not None:
                result.append(int(event["day"]))
        return result
    return []


def _pivotal_for_interval(result: SimulationResult, day: int) -> bool:
    source_day = day - 1
    if source_day < 0 or source_day >= len(result.days):
        return False
    return bool(result.days[source_day].focal_pivotal)


def _run_statistics(
    result: SimulationResult,
    focal: object,
    *,
    phase: str,
    scenario_index: int,
    candidate: str,
    exposure: int,
) -> dict[str, object]:
    focal_name = result.focal_agent_name
    if focal_name is None:
        raise AssertionError("Pass 6 case is missing a focal agent")
    history = result.agent_history(focal_name)
    start = result.config.voting_start_day
    if start is None:
        start = result.config.marked_boundary_day
    live_days = [day for day in range(start, len(result.days))]
    realised_days = [day for day in range(start + 1, len(result.days))]
    daily_pnl = [history[day].daily_pnl for day in realised_days]
    cumulative = 0
    high_water = 0
    drawdowns: list[int] = []
    for increment in daily_pnl:
        cumulative += increment
        high_water = max(high_water, cumulative)
        drawdowns.append(high_water - cumulative)
    actions = [history[day].action for day in live_days]
    previous = 0
    turnover = 0
    for action in actions:
        turnover += abs(action - previous)
        previous = action

    pivotal_pnl = 0
    non_pivotal_pnl = 0
    pivotal_intervals = 0
    non_pivotal_intervals = 0
    for day, increment in zip(realised_days, daily_pnl):
        if _pivotal_for_interval(result, day):
            pivotal_pnl += increment
            pivotal_intervals += 1
        else:
            non_pivotal_pnl += increment
            non_pivotal_intervals += 1

    focal_breaches = sum(
        breach.agent_name == focal_name for breach in result.budget_breaches
    )
    focal_rejected = sum(
        rejection.agent_name == focal_name for rejection in result.rejected_actions
    )
    diagnostics: Mapping[str, object] = {}
    diagnostic_method = getattr(focal, "diagnostics", None)
    if callable(diagnostic_method):
        raw_diagnostics = diagnostic_method()
        if isinstance(raw_diagnostics, Mapping):
            diagnostics = raw_diagnostics
    activation_days = _activation_days_from_diagnostics(diagnostics)
    real_trade_days = diagnostics.get("real_trade_days", [])
    if not isinstance(real_trade_days, Sequence) or isinstance(real_trade_days, (str, bytes)):
        real_trade_days = []
    first_real_trade_day = min((int(day) for day in real_trade_days), default=None)
    first_activation_day = min(activation_days, default=None)

    master = diagnostics.get("master")
    if not isinstance(master, Mapping):
        master = {}
    shadow_events = diagnostics.get("shadow_events", [])
    if not isinstance(shadow_events, Sequence) or isinstance(shadow_events, (str, bytes)):
        shadow_events = []
    shadow_values = [
        int(event.get("reward", 0))
        for event in shadow_events
        if isinstance(event, Mapping)
    ]
    shadow_total = int(master.get("master_shadow_pnl", sum(shadow_values)) or 0)
    shadow_pre_qualification = sum(
        int(event.get("reward", 0))
        for event in shadow_events
        if isinstance(event, Mapping)
        and (first_activation_day is None or int(event.get("day", 0)) <= first_activation_day)
    )
    shadow_post_qualification = sum(
        int(event.get("reward", 0))
        for event in shadow_events
        if isinstance(event, Mapping)
        and first_activation_day is not None
        and int(event.get("day", 0)) > first_activation_day
    )
    pivotal_shadow = 0
    non_pivotal_shadow = 0
    for event in shadow_events:
        if not isinstance(event, Mapping):
            continue
        day = int(event.get("day", 0))
        reward = int(event.get("reward", 0))
        if _pivotal_for_interval(result, day):
            pivotal_shadow += reward
        else:
            non_pivotal_shadow += reward

    expert_rewards = master.get("expert_cumulative_rewards", {})
    if not isinstance(expert_rewards, Mapping):
        expert_rewards = {}
    expert_rewards = {str(name): int(value) for name, value in expert_rewards.items()}
    reward_history = master.get("reward_history", [])
    if not isinstance(reward_history, Sequence) or isinstance(reward_history, (str, bytes)):
        reward_history = []
    best_static = max(expert_rewards.values(), default=0)
    best_low_switch = _best_low_switch_reward(reward_history)
    gate = diagnostics.get("gate")
    if not isinstance(gate, Mapping):
        gate = {}
    stop_overshoots = _extract_stop_overshoots(diagnostics)
    drawdown_stop = bool(diagnostics.get("drawdown_stop_active", False))
    loss_stop = bool(diagnostics.get("loss_stop_active", False))
    if candidate == "risk50_burnin1_markov":
        loss_stop = bool(diagnostics.get("loss_stop_active", False))
    return {
        "phase": phase,
        "scenario_index": scenario_index,
        "scenario_name": result.scenario_name,
        "family": result.scenario_configuration.get("family"),
        "seed": result.scenario_configuration.get("seed"),
        "pair_id": result.scenario_configuration.get("pair_id"),
        "population_size": result.scenario_configuration.get("population_size"),
        "execution_mode": result.scenario_configuration.get("execution_mode"),
        "candidate": candidate,
        "exposure": exposure,
        "marked_pnl": int(result.marked_pnl[focal_name]),
        "calibration_pnl": int(result.calibration_pnl[focal_name]),
        "mean_drawdown": _safe_mean(drawdowns),
        "max_drawdown": max(drawdowns, default=0),
        "active_days": sum(action != 0 for action in actions),
        "turnover": turnover,
        "beat_flat": bool(result.marked_pnl[focal_name] > 0),
        "flat_tie": bool(result.marked_pnl[focal_name] == 0),
        "pivotal_pnl": pivotal_pnl,
        "non_pivotal_pnl": non_pivotal_pnl,
        "pivotal_intervals": pivotal_intervals,
        "non_pivotal_intervals": non_pivotal_intervals,
        "budget_breaches": focal_breaches,
        "rejected_actions": focal_rejected,
        "activation_count": len(activation_days),
        "reactivation_count": int(gate.get("reactivation_count", diagnostics.get("reactivation_count", 0)) or 0),
        "activation_days": activation_days,
        "first_activation_day": first_activation_day,
        "never_activated": not activation_days,
        "first_real_trade_day": first_real_trade_day,
        "loss_stop": loss_stop,
        "drawdown_stop": drawdown_stop,
        "stop_overshoots": stop_overshoots,
        "stop_overshoot_max": max(stop_overshoots, default=0),
        "headroom_gates": int(diagnostics.get("headroom_gate_count", 0) or 0),
        "edge_gates": int(diagnostics.get("edge_gate_count", 0) or 0),
        "unknown_gates": int(diagnostics.get("unknown_gate_count", diagnostics.get("unknown_pause_count", 0)) or 0),
        "floor_gates": int(diagnostics.get("floor_gate_count", 0) or 0),
        "shadow_total_pnl": shadow_total if master else None,
        "shadow_pre_qualification_pnl": shadow_pre_qualification if master else None,
        "shadow_post_qualification_pnl": shadow_post_qualification if master else None,
        "shadow_pivotal_pnl": pivotal_shadow if master else None,
        "shadow_non_pivotal_pnl": non_pivotal_shadow if master else None,
        "shadow_to_actual_gap": (
            int(result.marked_pnl[focal_name]) - shadow_total if master else None
        ),
        "expert_cumulative_rewards": expert_rewards if master else None,
        "best_static_expert_reward": best_static if master else None,
        "best_low_switch_reward": best_low_switch if master else None,
        "master_static_regret": (best_static - shadow_total) if master else None,
        "master_low_switch_regret": (best_low_switch - shadow_total) if master else None,
        "ctw_reward": expert_rewards.get("context_tree_weighting") if master else None,
        "markov2_reward": expert_rewards.get("markov_order_2") if master else None,
        "ctw_minus_markov2": (
            expert_rewards.get("context_tree_weighting", 0)
            - expert_rewards.get("markov_order_2", 0)
            if master
            else None
        ),
        "final_weights": master.get("weights") if master else None,
        "gate_state": dict(gate),
    }


def run_case(task: CaseTask) -> dict[str, object]:
    """Top-level picklable worker for one endogenous simulator case."""

    suite = _scenario_suite(task.phase)
    scenario = suite[task.scenario_index]
    focal = _make_candidate(task.candidate, task.exposure)
    result = LiferaftSimulator(
        (focal, *scenario.population_factory()),
        scenario.config,
        focal_agent_name=focal.name,
        scenario_name=scenario.name,
        scenario_configuration={
            "suite": task.phase,
            "family": scenario.family,
            "seed": scenario.seed,
            "execution_mode": scenario.execution_mode,
            "population": scenario.population_description,
            "population_size": scenario.population_size,
            "pair_id": scenario.pair_id,
            "path_controlled": scenario.path_controlled,
            "drift_direction": scenario.drift_direction,
            "drift_strength": scenario.drift_strength,
            "other_portfolio_exposure": task.exposure,
            "market_mode": scenario.config.market_mode,
            "voting_start_day": scenario.config.voting_start_day,
        },
        random_seeds={"scenario": scenario.seed},
        other_portfolio_exposure={focal.name: task.exposure},
    ).run()
    return _run_statistics(
        result,
        focal,
        phase=task.phase,
        scenario_index=task.scenario_index,
        candidate=task.candidate,
        exposure=task.exposure,
    )


def run_case_batch(tasks: Sequence[CaseTask]) -> list[dict[str, object]]:
    """Top-level process worker; it does not write files."""

    return [run_case(task) for task in tasks]


def _chunks(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def run_batches(
    tasks: Sequence[Any],
    worker_function: Any,
    *,
    workers: int,
    batch_size: int,
    label: str,
) -> tuple[list[Any], float]:
    """Run deterministic batches and reduce them in stable key order."""

    batches = _chunks(list(tasks), batch_size)
    started = time.perf_counter()
    outputs: list[Any] = []
    total = len(batches)
    completed = 0
    print(f"{label}: {len(tasks)} items in {total} batches, workers={workers}", flush=True)

    if workers <= 1:
        for batch in batches:
            outputs.extend(worker_function(batch))
            completed += 1
            if completed == 1 or completed == total or completed % max(1, total // 20) == 0:
                print(f"{label}: {completed}/{total} batches", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker_function, batch) for batch in batches]
            for future in concurrent.futures.as_completed(futures):
                # Calling result() here propagates the original worker
                # exception instead of silently discarding a failed batch.
                outputs.extend(future.result())
                completed += 1
                if completed == 1 or completed == total or completed % max(1, total // 20) == 0:
                    print(f"{label}: {completed}/{total} batches", flush=True)
    elapsed = time.perf_counter() - started
    return outputs, elapsed


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": _safe_mean(values),
        "median": statistics.median(values) if values else 0.0,
        "lower_quartile": _quartile(values, 0.25),
        "worst": min(values) if values else 0.0,
    }


def _group_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    pnl = [float(row["marked_pnl"]) for row in records]
    dd = [float(row["mean_drawdown"]) for row in records]
    max_dd = [float(row["max_drawdown"]) for row in records]
    return {
        "runs": len(records),
        "pnl": _summary(pnl),
        "mean_drawdown": _safe_mean(dd),
        "maximum_drawdown": max(max_dd, default=0.0),
        "beat_flat_rate": _safe_mean(float(bool(row["beat_flat"])) for row in records),
        "tie_flat_rate": _safe_mean(float(bool(row["flat_tie"])) for row in records),
        "active_days": _safe_mean(float(row["active_days"]) for row in records),
        "turnover": _safe_mean(float(row["turnover"]) for row in records),
        "activation_rate": _safe_mean(float(bool(row["activation_count"])) for row in records),
        "never_activated_rate": _safe_mean(float(bool(row["never_activated"])) for row in records),
        "reactivation_rate": _safe_mean(float(row["reactivation_count"] > 0) for row in records),
        "loss_stop_rate": _safe_mean(float(bool(row["loss_stop"])) for row in records),
        "drawdown_stop_rate": _safe_mean(float(bool(row["drawdown_stop"])) for row in records),
        "mean_activation_count": _safe_mean(float(row["activation_count"]) for row in records),
        "mean_reactivation_count": _safe_mean(float(row["reactivation_count"]) for row in records),
        "mean_pivotal_pnl": _safe_mean(float(row["pivotal_pnl"]) for row in records),
        "mean_non_pivotal_pnl": _safe_mean(float(row["non_pivotal_pnl"]) for row in records),
        "budget_breaches": sum(int(row["budget_breaches"]) for row in records),
        "rejected_actions": sum(int(row["rejected_actions"]) for row in records),
        "mean_headroom_gates": _safe_mean(float(row["headroom_gates"]) for row in records),
        "mean_edge_gates": _safe_mean(float(row["edge_gates"]) for row in records),
        "mean_unknown_gates": _safe_mean(float(row["unknown_gates"]) for row in records),
        "mean_floor_gates": _safe_mean(float(row["floor_gates"]) for row in records),
        "mean_stop_overshoot": _safe_mean(
            float(row["stop_overshoot_max"]) for row in records
        ),
        "shadow_total_mean": _safe_mean(
            float(row["shadow_total_pnl"])
            for row in records
            if row["shadow_total_pnl"] is not None
        ),
        "shadow_pre_qualification_mean": _safe_mean(
            float(row["shadow_pre_qualification_pnl"])
            for row in records
            if row["shadow_pre_qualification_pnl"] is not None
        ),
        "shadow_post_qualification_mean": _safe_mean(
            float(row["shadow_post_qualification_pnl"])
            for row in records
            if row["shadow_post_qualification_pnl"] is not None
        ),
        "shadow_to_actual_gap_mean": _safe_mean(
            float(row["shadow_to_actual_gap"])
            for row in records
            if row["shadow_to_actual_gap"] is not None
        ),
        "ctw_mean": _safe_mean(
            float(row["ctw_reward"]) for row in records if row["ctw_reward"] is not None
        ),
        "markov2_mean": _safe_mean(
            float(row["markov2_reward"])
            for row in records
            if row["markov2_reward"] is not None
        ),
        "ctw_minus_markov2_mean": _safe_mean(
            float(row["ctw_minus_markov2"])
            for row in records
            if row["ctw_minus_markov2"] is not None
        ),
    }


def aggregate_records(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_candidate_exposure: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_family: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_mode: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        by_candidate_exposure[
            f"{row['candidate']}|{row['exposure']}"
        ].append(row)
        by_family[
            f"{row['candidate']}|{row['exposure']}|{row['family']}"
        ].append(row)
        by_mode[
            f"{row['candidate']}|{row['exposure']}|{row['execution_mode']}"
        ].append(row)
    return {
        "candidate_exposure": {
            key: _group_summary(value)
            for key, value in sorted(by_candidate_exposure.items())
        },
        "family": {
            key: _group_summary(value) for key, value in sorted(by_family.items())
        },
        "execution_mode": {
            key: _group_summary(value) for key, value in sorted(by_mode.items())
        },
    }


def _conservative_binomial_upper(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    estimate = successes / trials
    margin = math.sqrt(math.log(1 / NULL_CONFIDENCE_DELTA) / (2 * trials))
    return min(1.0, estimate + margin)


def _synthetic_observation(
    *,
    day: int,
    price: int,
    previous_price: int | None,
    previous_change: int | None,
    own_position: int,
    start: int = 365,
) -> AgentObservation:
    history = (price,) if previous_price is None else (previous_price, price)
    return AgentObservation(
        day=day,
        price=price,
        price_history=history,
        previous_price=previous_price,
        previous_price_change=previous_change,
        previous_move_is_reset=False,
        is_reset_day=False,
        marked_boundary_day=start,
        price_floor=20_000,
        long_majority_move=-5_000,
        short_majority_move=8_000,
        position_limit=1,
        gross_portfolio_budget=600_000,
        own_position=own_position,
        voting_active=True,
        market_mode="inactive_until_marked",
        voting_start_day=start,
    )


def _changes_for_process(
    rng: random.Random,
    process: str,
    horizon: int = 365,
) -> list[int]:
    changes: list[int] = []
    last_bit: int | None = None
    for index in range(horizon):
        if process == "iid_null":
            q_short = 5 / 13
        elif process == "iid_q_050":
            q_short = 0.50
        elif process == "iid_q_025":
            q_short = 0.25
        elif process == "conditional_zero_edge":
            if last_bit is None:
                q_short = 5 / 13
            else:
                # P(short|long)=.25 and P(short|short)=.60 has stationary
                # q=5/13, but the transitions are exploitable.
                q_short = 0.25 if last_bit == 0 else 0.60
        elif process == "regime_switch":
            q_short = 0.25 if index < horizon // 2 else 0.50
        else:
            raise ValueError(f"unknown synthetic process {process!r}")
        last_bit = 1 if rng.random() < q_short else 0
        changes.append(8_000 if last_bit else -5_000)
    return changes


def _run_synthetic_candidate(candidate: str, changes: Sequence[int]) -> dict[str, object]:
    strategy = make_pass6_strategy(candidate)
    # Keep the synthetic instrument at a representative safe price so the
    # portfolio-headroom guard does not turn every activated path into a
    # zero-live-P&L path.  The exogenous public movement is supplied through
    # previous_price_change; no floor clipping is introduced in calibration.
    price = 100_000
    held = strategy.decide(
        _synthetic_observation(
            day=365,
            price=price,
            previous_price=None,
            previous_change=None,
            own_position=0,
        )
    )
    for index, change in enumerate(changes, start=1):
        next_price = price + change
        held = strategy.decide(
            _synthetic_observation(
                day=365 + index,
                price=next_price,
                previous_price=price,
                previous_change=change,
                own_position=held,
            )
        )
        price = next_price
    diagnostics = strategy.diagnostics()
    gate = diagnostics.get("gate", {})
    if not isinstance(gate, Mapping):
        gate = {}
    activation_days = gate.get("activation_days", [])
    if not isinstance(activation_days, Sequence) or isinstance(activation_days, (str, bytes)):
        activation_days = []
    return {
        "candidate": candidate,
        "activated": bool(activation_days),
        "activation_count": len(activation_days),
        "activation_days": [int(day) for day in activation_days],
        "activation_delay": (
            int(activation_days[0]) - 365 if activation_days else None
        ),
        "real_pnl": int(diagnostics.get("actual_realised_pnl", 0) or 0),
        "shadow_pnl": int(diagnostics.get("master", {}).get("master_shadow_pnl", 0) or 0),
        "scoreable_count": int(diagnostics.get("master", {}).get("scoreable_count", 0) or 0),
        "active_days": len(diagnostics.get("real_trade_days", [])),
        "never_activated": not bool(activation_days),
        "evidence": dict(gate),
    }


def run_null_batch(tasks: Sequence[tuple[str, int]]) -> list[dict[str, object]]:
    """Top-level worker for null or power path batches."""

    results: list[dict[str, object]] = []
    for process, index in tasks:
        rng = random.Random(stable_seed("pass6-null", process, index))
        changes = _changes_for_process(rng, process)
        candidate_results = {
            candidate: _run_synthetic_candidate(candidate, changes)
            for candidate in NEW_CANDIDATES
        }
        results.append(
            {
                "process": process,
                "path_index": index,
                "candidates": candidate_results,
                "either_activated": any(
                    bool(value["activated"]) for value in candidate_results.values()
                ),
            }
        )
    return results


def _activation_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    activations = [row for row in rows if bool(row["activated"])]
    delays = [int(row["activation_delay"]) for row in activations if row["activation_delay"] is not None]
    real_pnl = [float(row["real_pnl"]) for row in rows]
    shadow_pnl = [float(row["shadow_pnl"]) for row in rows]
    activation_days = [int(row["activation_days"][0]) for row in activations if row["activation_days"]]
    return {
        "paths": len(rows),
        "activation_count": len(activations),
        "activation_probability": (len(activations) / len(rows) if rows else 0.0),
        "activation_upper_99_conservative": _conservative_binomial_upper(len(activations), len(rows)),
        "never_activated_rate": (
            (len(rows) - len(activations)) / len(rows) if rows else 1.0
        ),
        "activation_day_distribution": _summary([float(value) for value in activation_days]),
        "activation_delay_distribution": _summary([float(value) for value in delays]),
        "mean_real_pnl": _safe_mean(real_pnl),
        "median_real_pnl": statistics.median(real_pnl) if real_pnl else 0.0,
        "mean_shadow_pnl": _safe_mean(shadow_pnl),
        "median_shadow_pnl": statistics.median(shadow_pnl) if shadow_pnl else 0.0,
        "mean_active_days": _safe_mean(float(row["active_days"]) for row in rows),
        "mean_activation_count": _safe_mean(float(row["activation_count"]) for row in rows),
    }


def run_null_calibration(*, workers: int, quick: bool = False) -> tuple[dict[str, object], float, dict[str, object]]:
    null_count = 64 if quick else NULL_PATHS
    power_count = 64 if quick else POWER_PATHS
    null_tasks = [("iid_null", index) for index in range(null_count)]
    null_rows, null_runtime = run_batches(
        null_tasks,
        run_null_batch,
        workers=workers,
        batch_size=8 if quick else NULL_BATCH_SIZE,
        label="economic-null calibration",
    )
    null_rows = sorted(null_rows, key=lambda row: int(row["path_index"]))
    per_gate: dict[str, object] = {}
    for candidate in NEW_CANDIDATES:
        rows = [row["candidates"][candidate] for row in null_rows]
        per_gate[candidate] = _activation_summary(rows)
    combined_count = sum(bool(row["either_activated"]) for row in null_rows)
    null_result = {
        "process": "iid_null_q_5_13",
        "path_count": null_count,
        "per_gate": per_gate,
        "combined_either_activation_count": combined_count,
        "combined_either_activation_probability": combined_count / null_count,
        "combined_either_upper_99_conservative": _conservative_binomial_upper(combined_count, null_count),
        "design_global_alpha": GLOBAL_ONE_SIDED_ALPHA,
        "design_per_gate_alpha": PER_GATE_ALPHA,
        "conservative_uncertainty": "99% Hoeffding upper bound for a binomial proportion",
        "rows": null_rows,
    }
    gate_passes = {
        candidate: bool(
            per_gate[candidate]["activation_upper_99_conservative"] <= PER_GATE_ALPHA
        )
        for candidate in NEW_CANDIDATES
    }
    null_pass = bool(
        null_result["combined_either_upper_99_conservative"] <= GLOBAL_ONE_SIDED_ALPHA
        and any(gate_passes.values())
    )
    null_result["gate_passes_predeclared_calibration"] = gate_passes
    null_result["passes_predeclared_calibration"] = null_pass

    processes = ("iid_q_050", "iid_q_025", "conditional_zero_edge", "regime_switch")
    power_tasks = [(process, index) for process in processes for index in range(power_count)]
    power_rows, power_runtime = run_batches(
        power_tasks,
        run_null_batch,
        workers=workers,
        batch_size=8 if quick else NULL_BATCH_SIZE,
        label="predeclared power diagnostics",
    )
    power_rows = sorted(power_rows, key=lambda row: (str(row["process"]), int(row["path_index"])))
    power_result: dict[str, object] = {
        "path_count_per_process": power_count,
        "processes": {},
        "rows": power_rows,
    }
    for process in processes:
        process_rows = [row for row in power_rows if row["process"] == process]
        power_result["processes"][process] = {
            candidate: _activation_summary(
                [row["candidates"][candidate] for row in process_rows]
            )
            for candidate in NEW_CANDIDATES
        }
    return (
        {
            "null": null_result,
            "power": power_result,
            "runtimes_seconds": {
                "null": null_runtime,
                "power": power_runtime,
                "total": null_runtime + power_runtime,
            },
        },
        null_runtime + power_runtime,
        {"null_pass": null_pass, "null_rows": null_rows, "power_rows": power_rows},
    )


def build_evaluation_tasks(phase: str, *, quick: bool = False) -> list[CaseTask]:
    scenarios = _scenario_suite(phase)
    scenario_indices = list(range(len(scenarios)))
    candidates = list(EVALUATION_CANDIDATES)
    exposures = list(EXPOSURES)
    if quick:
        scenario_indices = scenario_indices[:2]
        candidates = ["fixed_checkpoint_fixed_share", "anytime_valid_fixed_share", "flat"]
        exposures = [0, 450_000]
    return [
        CaseTask(phase, index, candidate, exposure)
        for index in scenario_indices
        for candidate in candidates
        for exposure in exposures
    ]


def run_evaluation_phase(
    phase: str,
    *,
    workers: int,
    quick: bool = False,
) -> tuple[dict[str, object], float]:
    tasks = build_evaluation_tasks(phase, quick=quick)
    rows, runtime = run_batches(
        tasks,
        run_case_batch,
        workers=workers,
        batch_size=4 if quick else EVAL_BATCH_SIZE,
        label=f"{phase} simulator evaluation",
    )
    rows = sorted(rows, key=lambda row: (
        str(row["phase"]),
        int(row["scenario_index"]),
        str(row["candidate"]),
        int(row["exposure"]),
    ))
    return {
        "phase": phase,
        "scenario_count": len(_scenario_suite(phase)) if not quick else min(2, len(_scenario_suite(phase))),
        "candidate_count": len(EVALUATION_CANDIDATES) if not quick else 3,
        "exposures": list(EXPOSURES) if not quick else [0, 450_000],
        "run_count": len(rows),
        "rows": rows,
        "aggregates": aggregate_records(rows),
    }, runtime


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_reproducibility_check(*, workers: int) -> dict[str, object]:
    tasks = [
        CaseTask("development", index, candidate, 0)
        for index in range(min(2, len(_scenario_suite("development"))))
        for candidate in ("fixed_checkpoint_fixed_share", "anytime_valid_fixed_share", "flat")
    ]
    serial_rows, serial_runtime = run_batches(
        tasks,
        run_case_batch,
        workers=1,
        batch_size=2,
        label="serial reproducibility reference",
    )
    parallel_workers = max(1, min(4, workers))
    parallel_rows, parallel_runtime = run_batches(
        tasks,
        run_case_batch,
        workers=parallel_workers,
        batch_size=2,
        label="parallel reproducibility check",
    )
    serial_rows = sorted(serial_rows, key=lambda row: (row["scenario_index"], row["candidate"], row["exposure"]))
    parallel_rows = sorted(parallel_rows, key=lambda row: (row["scenario_index"], row["candidate"], row["exposure"]))
    serial_digest = _canonical_digest(serial_rows)
    parallel_digest = _canonical_digest(parallel_rows)
    return {
        "case_count": len(tasks),
        "serial_workers": 1,
        "parallel_workers": parallel_workers,
        "serial_digest": serial_digest,
        "parallel_digest": parallel_digest,
        "bit_identical": serial_rows == parallel_rows,
        "serial_runtime_seconds": serial_runtime,
        "parallel_runtime_seconds": parallel_runtime,
    }


def _primary_validation_rows(
    evaluation: Mapping[str, object],
    candidate: str,
) -> list[Mapping[str, object]]:
    return [
        row
        for row in evaluation.get("rows", [])
        if row.get("candidate") == candidate
    ]


def mechanical_screen(
    state: Mapping[str, object],
) -> dict[str, object]:
    validation = state.get("evaluation", {}).get("validation", {})
    if not isinstance(validation, Mapping):
        return {candidate: {"eligible": False, "reason": "validation missing"} for candidate in NEW_CANDIDATES}
    null = state.get("null_calibration", {}).get("null", {})
    if not isinstance(null, Mapping):
        null = {}
    per_gate = null.get("per_gate", {})
    if not isinstance(per_gate, Mapping):
        per_gate = {}
    wrapper_rows = _primary_validation_rows(validation, "risk50_burnin1_markov")
    wrapper_summary = _group_summary(wrapper_rows)
    wrapper_mean = float(wrapper_summary["pnl"]["mean"])
    wrapper_median = float(wrapper_summary["pnl"]["median"])
    wrapper_max_dd = float(wrapper_summary["maximum_drawdown"])
    screen: dict[str, object] = {}
    for candidate in NEW_CANDIDATES:
        rows = _primary_validation_rows(validation, candidate)
        summary = _group_summary(rows)
        family_means: dict[str, float] = {}
        for family in PRINCIPAL_FAMILIES:
            family_rows = [row for row in rows if row.get("family") == family]
            family_means[family] = float(_group_summary(family_rows)["pnl"]["mean"])
        positive_families = sum(value > 0 for value in family_means.values())
        gate_calibration = per_gate.get(candidate, {})
        gate_ok = bool(
            isinstance(gate_calibration, Mapping)
            and gate_calibration.get("activation_upper_99_conservative", 1.0) <= PER_GATE_ALPHA
            and null.get("combined_either_upper_99_conservative", 1.0) <= GLOBAL_ONE_SIDED_ALPHA
        )
        candidate_mean = float(summary["pnl"]["mean"])
        candidate_median = float(summary["pnl"]["median"])
        candidate_max_dd = float(summary["maximum_drawdown"])
        wrapper_retention = (
            candidate_mean / wrapper_mean if wrapper_mean > 0 else 0.0
        )
        wrapper_tradeoff = bool(
            candidate_median >= wrapper_median + 10_000
            and candidate_max_dd <= 0.80 * wrapper_max_dd
        )
        criteria = {
            "gate_null_calibration": gate_ok,
            "combined_false_activation": bool(
                null.get("combined_either_upper_99_conservative", 1.0)
                <= GLOBAL_ONE_SIDED_ALPHA
            ),
            "worst_run_at_least_minus_60000": summary["pnl"]["worst"] >= -60_000,
            "maximum_drawdown_at_most_75000": candidate_max_dd <= 75_000,
            "zero_budget_breaches": summary["budget_breaches"] == 0,
            "zero_rejected_actions": summary["rejected_actions"] == 0,
            "mean_pivotal_pnl_at_least_minus_25000": summary["mean_pivotal_pnl"] >= -25_000,
            "mean_positive": candidate_mean > 0,
            "median_nonnegative": candidate_median >= 0,
            "beat_flat_at_least_55_percent": summary["beat_flat_rate"] >= 0.55,
            "positive_family_mean_at_least_half": positive_families >= math.ceil(len(PRINCIPAL_FAMILIES) / 2),
            "wrapper_retention_80_percent_or_tradeoff": bool(wrapper_retention >= 0.80 or wrapper_tradeoff),
        }
        screen[candidate] = {
            "eligible": all(criteria.values()),
            "criteria": criteria,
            "summary": summary,
            "family_means": family_means,
            "positive_family_count": positive_families,
            "wrapper_mean": wrapper_mean,
            "wrapper_median": wrapper_median,
            "wrapper_max_drawdown": wrapper_max_dd,
            "wrapper_retention": wrapper_retention,
            "wrapper_tradeoff": wrapper_tradeoff,
        }
    return screen


def candidate_priority(screen: Mapping[str, object]) -> str | None:
    for candidate in ("anytime_valid_fixed_share", "fixed_checkpoint_fixed_share"):
        details = screen.get(candidate, {})
        if isinstance(details, Mapping) and details.get("eligible"):
            return candidate
    return None


def _format_money(value: float) -> str:
    return f"AUD {value:,.0f}"


def write_report(state: Mapping[str, object], *, fail_fast: bool = False) -> None:
    null = state.get("null_calibration", {})
    null_summary = null.get("null", {}) if isinstance(null, Mapping) else {}
    power = null.get("power", {}) if isinstance(null, Mapping) else {}
    validation = state.get("evaluation", {}).get("validation", {})
    development = state.get("evaluation", {}).get("development", {})
    screen = state.get("mechanical_screen", {})
    selected = state.get("selected_challenger")
    lines = [
        "# Liferaft Pass 6A report",
        "",
        "> Development-only research. Existing Pass 3/4/5 evidence is consumed context; this is not a blind or final suite.",
        "",
        "## Mathematical and statistical validity",
        "",
        "The master uses seven causal experts: flat, Beta(1,1) order-zero frequency, add-one Markov orders 1 and 2, binary CTW with KT nodes at maximum depth 6, persistence, and reversal. Each expert proposes the economically preferred position relative to q*=5/13. Rewards are dollar rewards divided by the frozen AUD 8,000 bound before the exponential Fixed-Share update.",
        "",
        f"Fixed-Share constants: eta={state.get('constants', {}).get('eta')}, share_rate={state.get('constants', {}).get('share_rate')}, block size=20, per-gate alpha=2.5%, checkpoint alpha={state.get('constants', {}).get('checkpoint_alpha')}, primary pivotal haircut=AUD 1,300, minimum residual edge=AUD 1,000.",
        "",
        "The fixed-checkpoint gate tests each complete non-overlapping block with a one-sided Hoeffding lower bound and authorizes only the next block. The anytime gate is a nonnegative e-process with factor 1+0.5*(reward/8000); Ville's inequality covers repeated daily inspection without an outcome-dependent reset.",
        "",
        f"Economic-null calibration: {null_summary.get('path_count', 0)} non-pivotal IID paths at q=5/13. The predeclared calibration result is **{'PASS' if null_summary.get('passes_predeclared_calibration') else 'FAIL'}**; combined either-gate activation rate={float(null_summary.get('combined_either_activation_probability', 0))*100:.3f}%, conservative 99% upper bound={float(null_summary.get('combined_either_upper_99_conservative', 1))*100:.3f}%.",
        "",
        "| gate | activations | rate | conservative 99% upper | mean real P&L | mean shadow P&L |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in NEW_CANDIDATES:
        row = null_summary.get("per_gate", {}).get(candidate, {})
        lines.append(
            f"| `{candidate}` | {row.get('activation_count', 0)} | {float(row.get('activation_probability', 0))*100:.3f}% | {float(row.get('activation_upper_99_conservative', 1))*100:.3f}% | {_format_money(float(row.get('mean_real_pnl', 0)))} | {_format_money(float(row.get('mean_shadow_pnl', 0)))} |"
        )
    lines += [
        "",
        "The calibration is a statistical-gate test only: public paths are exogenous, focal actions cannot change them, and hidden majority/pivotal labels never enter either strategy.",
        "",
        "## Synthetic power diagnostics",
        "",
        "These were predeclared diagnostics, not tuning inputs.",
        "",
        "| process | gate | activation rate | median activation delay | mean real P&L |",
        "|---|---|---:|---:|---:|",
    ]
    for process, gate_rows in (power.get("processes", {}) if isinstance(power, Mapping) else {}).items():
        for candidate in NEW_CANDIDATES:
            row = gate_rows.get(candidate, {})
            delay = row.get("activation_delay_distribution", {}).get("median", 0)
            lines.append(
                f"| `{process}` | `{candidate}` | {float(row.get('activation_probability', 0))*100:.1f}% | {float(delay):.1f} | {_format_money(float(row.get('mean_real_pnl', 0)))} |"
            )
    lines += [
        "",
        "## Simulator performance and transfer audit",
        "",
        f"Development rows: {development.get('run_count', 0)}. Consumed validation rows: {validation.get('run_count', 0)}. Validation scenarios were the existing 480 cases with both inactive execution modes and exposures AUD 0/150,000/300,000/450,000.",
        "",
        "Validation aggregate summaries for the two new candidates:",
        "",
        "| candidate | mean | median | lower quartile | worst | mean DD | max DD | beat flat | active days | turnover | pivotal mean | non-pivotal mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in NEW_CANDIDATES:
        rows = _primary_validation_rows(validation, candidate)
        summary = _group_summary(rows)
        pnl = summary["pnl"]
        lines.append(
            f"| `{candidate}` | {_format_money(pnl['mean'])} | {_format_money(pnl['median'])} | {_format_money(pnl['lower_quartile'])} | {_format_money(pnl['worst'])} | {_format_money(summary['mean_drawdown'])} | {_format_money(summary['maximum_drawdown'])} | {summary['beat_flat_rate']*100:.1f}% | {summary['active_days']:.1f} | {summary['turnover']:.1f} | {_format_money(summary['mean_pivotal_pnl'])} | {_format_money(summary['mean_non_pivotal_pnl'])} |"
        )
    lines += [
        "",
        "Paper-to-live transfer for the new candidates is reported from the same run records: `shadow_pre_qualification_pnl` is the master paper reward through the qualification decision, `shadow_post_qualification_pnl` starts strictly after that decision, and `shadow_to_actual_gap` is actual realised Liferaft P&L minus scoreable shadow P&L. Pivotal/non-pivotal partitions use engine-only diagnostics after the run; the live strategy never sees those labels.",
        "",
        "| candidate | shadow pre-qualification | shadow post-qualification | actual-shadow gap | non-pivotal actual | pivotal actual | CTW - Markov-2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in NEW_CANDIDATES:
        rows = _primary_validation_rows(validation, candidate)
        summary = _group_summary(rows)
        lines.append(
            f"| `{candidate}` | {_format_money(summary['shadow_pre_qualification_mean'])} | {_format_money(summary['shadow_post_qualification_mean'])} | {_format_money(summary['shadow_to_actual_gap_mean'])} | {_format_money(summary['mean_non_pivotal_pnl'])} | {_format_money(summary['mean_pivotal_pnl'])} | {_format_money(summary['ctw_minus_markov2_mean'])} |"
        )
    lines += [
        "",
        "## Family, exposure, and execution-mode sensitivity",
        "",
        "The machine-readable results contain every run-level row and grouped family/exposure/mode summaries. The families deliberately cover persistent, reversing/reactive, regime-change, drift, startup, near-tie, floor, and budget failure modes; positive deterministic families are not treated as proof of generalisation.",
        "",
    ]
    for candidate in NEW_CANDIDATES:
        lines.append(f"### `{candidate}` family means")
        lines.append("")
        lines.append("| family | exposure 0 mean | exposure 150k mean | exposure 300k mean | exposure 450k mean |")
        lines.append("|---|---:|---:|---:|---:|")
        for family in PRINCIPAL_FAMILIES:
            means = []
            for exposure in EXPOSURES:
                key = f"{candidate}|{exposure}|{family}"
                value = validation.get("aggregates", {}).get("family", {}).get(key, {})
                means.append(float(value.get("pnl", {}).get("mean", 0)))
            lines.append(
                f"| `{family}` | {_format_money(means[0])} | {_format_money(means[1])} | {_format_money(means[2])} | {_format_money(means[3])} |"
            )
        lines.append("")
    lines += [
        "## Mechanical screen and recommendation",
        "",
        "All eligibility criteria are conjunctive, with only the predeclared risk50-wrapper retention/trade-off alternative using OR. Candidate priority is anytime-valid first, fixed-checkpoint second, otherwise no challenger.",
        "",
        "| candidate | null valid | safety/performance eligible | positive families | wrapper retention | recommendation status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for candidate in NEW_CANDIDATES:
        details = screen.get(candidate, {})
        criteria = details.get("criteria", {}) if isinstance(details, Mapping) else {}
        lines.append(
            f"| `{candidate}` | {criteria.get('gate_null_calibration', False)} | {details.get('eligible', False)} | {details.get('positive_family_count', 0)}/{len(PRINCIPAL_FAMILIES)} | {float(details.get('wrapper_retention', 0))*100:.1f}% | {'eligible' if details.get('eligible') else 'failed'} |"
        )
    lines += [
        "",
        f"Selected challenger for a later blind Pass 6B: **{selected or 'none; all challengers failed'}**.",
        "",
        "## Production recommendation",
        "",
        "This pass does not modify the production algorithm. A positive synthetic mean is not sufficient: the decision is controlled by null calibration, transfer gaps, pivotal exposure, budget mechanics, and the frozen conjunctive screen. If no candidate is selected, retain the existing production strategy and do not promote a Pass 6 challenger.",
        "",
        "Quarantine confirmation: no final scenario suite, final strategy module, final result, final decision, final receipt, or production file was imported, executed, created, modified, or used.",
        "",
    ]
    if fail_fast:
        lines.insert(4, "**Fail-fast:** both participation gates failed their predeclared null calibration screens, or the combined upper bound exceeded 5%, so no expensive simulator suite was run.")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(state: Mapping[str, object]) -> None:
    input_files = (
        "pass6_models.py",
        "pass6_strategies.py",
        "pass6_experiments.py",
        "test_pass6.py",
        "PASS6A_PROTOCOL.md",
        "simulator.py",
        "archetypes.py",
        "pass3_scenarios.py",
        "cold_start_strategies.py",
        "pass4_strategies.py",
        "shadow_strategies.py",
    )
    hashes = {
        filename: _sha256(ROOT / filename)
        for filename in input_files
        if (ROOT / filename).exists()
    }
    runtimes = state.get("runtimes_seconds", {})
    lines = [
        "# Liferaft Pass 6A manifest",
        "",
        "Status: frozen development-only evidence; no blind or final suite was created or executed.",
        "",
        f"- Python: `{sys.version.replace(chr(10), ' ')}`",
        f"- Platform: `{platform.platform()}`",
        f"- CPU count observed: `{os.cpu_count()}`",
        f"- Workers: `{state.get('workers')}`",
        f"- Commands: `{'; '.join(state.get('commands', []))}`",
        f"- Runtimes seconds: `{json.dumps(runtimes, sort_keys=True)}`",
        f"- Serial/parallel bit-identical: `{state.get('reproducibility', {}).get('bit_identical')}`",
        "",
        "## Result-affecting source hashes",
        "",
        "| file | SHA-256 |",
        "|---|---|",
    ]
    lines.extend(f"| `{filename}` | `{digest}` |" for filename, digest in hashes.items())
    lines += [
        "",
        "## Source inputs used",
        "",
        "- Existing non-final simulator, archetypes, scenario constructors, cold-start strategies, Pass 4A wrapper, and Pass 5 shadow wrapper.",
        "- Existing `development_scenarios()` and `validation_scenarios()` only; no `final_scenarios()` call.",
        "- No Pass 3/4/5 result was used for tuning; prior reports/results were consumed development evidence.",
        "",
        "## Quarantine confirmation",
        "",
        "The locked final artifacts remained untouched: no final strategy source, final result, final decision, final execution receipt, final report, final scenario constructor, or `--final` command was opened, parsed, imported, executed, recreated, renamed, overwritten, or used.",
        "",
        "Parent process was the sole writer of Pass 6A results, report, and manifest files; worker processes returned data only.",
    ]
    MANIFEST_PATH.write_text("\n".join(lines), encoding="utf-8")


def _load_state() -> dict[str, object]:
    if not RESULTS_PATH.exists():
        return {}
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def _write_state(state: Mapping[str, object]) -> None:
    RESULTS_PATH.write_text(
        json.dumps(state, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )


def _worker_count(args: argparse.Namespace) -> int:
    if args.serial:
        return 1
    raw = args.workers
    if raw == "auto":
        return max(1, (os.cpu_count() or 1) - 1)
    workers = int(raw)
    if workers <= 0:
        raise ValueError("--workers must be positive or auto")
    return workers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen Liferaft Pass 6A research pass")
    phases = parser.add_mutually_exclusive_group(required=True)
    phases.add_argument("--null", action="store_true", help="run null calibration and power diagnostics")
    phases.add_argument("--development", action="store_true", help="run the existing nine development scenarios")
    phases.add_argument("--validation", action="store_true", help="run the existing 480 consumed validation scenarios")
    parser.add_argument("--workers", default="auto", help="process workers: N or auto")
    parser.add_argument("--serial", action="store_true", help="force one worker")
    parser.add_argument("--quick", action="store_true", help="small mechanical smoke run; never merged into final results")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workers = _worker_count(args)
    if args.quick:
        if args.null:
            calibration, _, _ = run_null_calibration(workers=workers, quick=True)
            print(json.dumps({"quick": True, "null": calibration["null"], "power": calibration["power"]}, indent=2, sort_keys=True))
        else:
            phase = "development" if args.development else "validation"
            evaluation, runtime = run_evaluation_phase(phase, workers=workers, quick=True)
            print(json.dumps({"quick": True, "runtime_seconds": runtime, "aggregates": evaluation["aggregates"]}, indent=2, sort_keys=True))
        return 0

    state = _load_state()
    state.setdefault("constants", {
        "eta": __import__("research.liferaft.pass6_models", fromlist=["FIXED_SHARE_ETA"]).FIXED_SHARE_ETA,
        "share_rate": __import__("research.liferaft.pass6_models", fromlist=["FIXED_SHARE_RATE"]).FIXED_SHARE_RATE,
        "reward_bound": REWARD_BOUND,
        "checkpoint_alpha": __import__("research.liferaft.pass6_strategies", fromlist=["CHECKPOINT_ALPHA"]).CHECKPOINT_ALPHA,
    })
    state["workers"] = workers
    state.setdefault("commands", []).append("python -m research.liferaft.pass6_experiments " + " ".join(argv or sys.argv[1:]))
    state.setdefault("runtimes_seconds", {})

    if args.null:
        reproducibility = run_reproducibility_check(workers=workers)
        state["reproducibility"] = reproducibility
        calibration, runtime, _ = run_null_calibration(workers=workers)
        state["null_calibration"] = calibration
        state["runtimes_seconds"]["null_and_power"] = runtime
        state["fail_fast_permitted"] = not calibration["null"]["passes_predeclared_calibration"]
        _write_state(state)
        print(json.dumps({"null_pass": calibration["null"]["passes_predeclared_calibration"], "reproducibility": reproducibility}, indent=2, sort_keys=True))
        if state["fail_fast_permitted"]:
            state["selected_challenger"] = None
            state["mechanical_screen"] = {}
            write_report(state, fail_fast=True)
            write_manifest(state)
        return 0

    null_calibration = state.get("null_calibration")
    if not isinstance(null_calibration, Mapping):
        raise RuntimeError("run --null before simulator evaluation")
    null_pass = bool(null_calibration.get("null", {}).get("passes_predeclared_calibration", False))
    if not null_pass:
        print("Fail-fast: null calibration failed; simulator evaluation was not run.", flush=True)
        state["fail_fast_permitted"] = True
        state["selected_challenger"] = None
        write_report(state, fail_fast=True)
        write_manifest(state)
        _write_state(state)
        return 0

    phase = "development" if args.development else "validation"
    evaluation, runtime = run_evaluation_phase(phase, workers=workers)
    state.setdefault("evaluation", {})[phase] = evaluation
    state["runtimes_seconds"][phase] = runtime
    if phase == "validation":
        state["mechanical_screen"] = mechanical_screen(state)
        state["selected_challenger"] = candidate_priority(state["mechanical_screen"])
        write_report(state)
        write_manifest(state)
    _write_state(state)
    print(json.dumps({"phase": phase, "run_count": evaluation["run_count"], "runtime_seconds": runtime, "selected_challenger": state.get("selected_challenger")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
