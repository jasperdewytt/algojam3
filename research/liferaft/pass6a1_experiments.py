"""Fast Pass 6A.1 null calibration and consumed validation runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cold_start_strategies import make_cold_start_strategy
from .pass3_scenarios import validation_scenarios
from .pass4_strategies import make_pass4_strategy
from .pass6_strategies import make_pass6_strategy
from .pass6_models import FIXED_SHARE_ETA, FIXED_SHARE_RATE
from .pass6a1_strategy import (
    ADJUSTED_REWARD_BOUND,
    make_pass6a1_strategy,
    run_self_checks,
)
from .simulator import AgentObservation, LiferaftSimulator, MajorityOutcome, SimulationResult, majority_from_counts


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "PASS6A1_RESULTS.json"
EXPOSURES = (0, 150_000, 300_000, 450_000)
EVALUATION_CANDIDATES = (
    "pass6a1_corrected_anytime_fixed_share",
    "pass6_anytime_fixed_share",
    "risk50_burnin1_markov",
    "flat",
)
NEW_CANDIDATE = EVALUATION_CANDIDATES[0]
NULL_PATHS = 10_000
POWER_PATHS = 1_000
NULL_BATCH_SIZE = 50
EVAL_BATCH_SIZE = 32


@dataclass(frozen=True)
class CaseTask:
    scenario_index: int
    candidate: str
    exposure: int

    @property
    def key(self) -> tuple[object, ...]:
        return (self.scenario_index, self.candidate, self.exposure)


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


_SCENARIOS: tuple[Any, ...] | None = None


def _scenario_suite() -> tuple[Any, ...]:
    global _SCENARIOS
    if _SCENARIOS is None:
        _SCENARIOS = tuple(validation_scenarios())
    return _SCENARIOS


def _make_candidate(candidate: str, exposure: int) -> object:
    if candidate == NEW_CANDIDATE:
        return make_pass6a1_strategy(other_portfolio_exposure=exposure)
    if candidate == "pass6_anytime_fixed_share":
        return make_pass6_strategy(
            "anytime_valid_fixed_share",
            other_portfolio_exposure=exposure,
        )
    if candidate == "risk50_burnin1_markov":
        return make_pass4_strategy(
            "risk50_burnin1_markov",
            other_portfolio_exposure=exposure,
        )
    if candidate == "flat":
        return make_cold_start_strategy("flat")
    raise KeyError(candidate)


def _scenario_configuration(scenario: Any, exposure: int) -> dict[str, object]:
    return {
        "suite": "validation_pass6a1",
        "family": scenario.family,
        "seed": scenario.seed,
        "execution_mode": scenario.execution_mode,
        "population": scenario.population_description,
        "population_size": scenario.population_size,
        "pair_id": scenario.pair_id,
        "path_controlled": scenario.path_controlled,
        "drift_direction": scenario.drift_direction,
        "drift_strength": scenario.drift_strength,
        "other_portfolio_exposure": exposure,
        "market_mode": scenario.config.market_mode,
        "voting_start_day": scenario.config.voting_start_day,
    }


def _pnl_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"mean": 0.0, "median": 0.0, "lower_quartile": 0.0, "worst": 0.0}
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "lower_quartile": ordered[int(0.25 * (len(ordered) - 1))],
        "worst": ordered[0],
    }


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else 0.0


def _rate(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return (
        statistics.fmean(float(bool(row.get(key))) for row in rows)
        if rows
        else 0.0
    )


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    pnl = [float(row["pnl"]) for row in rows]
    dd = [float(row["mean_drawdown"]) for row in rows]
    max_dd = [float(row["max_drawdown"]) for row in rows]
    return {
        "runs": len(rows),
        "pnl": _pnl_summary(pnl),
        "mean_drawdown": statistics.fmean(dd) if dd else 0.0,
        "maximum_drawdown": max(max_dd, default=0.0),
        "beat_flat_rate": _rate(rows, "beat_flat"),
        "active_days": _mean(rows, "active_days"),
        "turnover": _mean(rows, "turnover"),
        "activation_rate": _rate(rows, "activated"),
        "median_activation_day": statistics.median(
            [float(row["activation_day"]) for row in rows if row.get("activation_day") is not None]
        ) if any(row.get("activation_day") is not None for row in rows) else None,
        "budget_breaches": sum(int(row["budget_breaches"]) for row in rows),
        "rejected_actions": sum(int(row["rejected_actions"]) for row in rows),
        "loss_stop_rate": _rate(rows, "loss_stop"),
        "drawdown_stop_rate": _rate(rows, "drawdown_stop"),
        "mean_pivotal_pnl": _mean(rows, "actual_pivotal_pnl"),
        "mean_non_pivotal_pnl": _mean(rows, "actual_non_pivotal_pnl"),
        "observable_shadow_mean": _mean(rows, "observable_shadow_pnl"),
        "observable_pivotal_shadow_mean": _mean(rows, "observable_pivotal_shadow_pnl"),
        "observable_non_pivotal_shadow_mean": _mean(rows, "observable_non_pivotal_shadow_pnl"),
        "adjusted_evidence_mean": _mean(rows, "adjusted_evidence_pnl"),
        "executable_shadow_mean": _mean(rows, "executable_shadow_pnl"),
        "executable_pivotal_shadow_mean": _mean(rows, "executable_pivotal_shadow_pnl"),
        "executable_non_pivotal_shadow_mean": _mean(rows, "executable_non_pivotal_shadow_pnl"),
        "foregone_opportunity_mean": _mean(rows, "foregone_paper_opportunity"),
        "observable_minus_executable_mean": _mean(rows, "observable_minus_executable_gap"),
        "counterfactual_pivotal_paper_rate": _mean(rows, "counterfactual_pivotal_paper_rate"),
        "matched_action_discrepancy_max": max(
            (float(row["matched_action_discrepancy_max"]) for row in rows if row.get("matched_action_discrepancy_max") is not None),
            default=0.0,
        ),
        "matched_action_discrepancy_total": sum(
            float(row["matched_action_discrepancy_total"])
            for row in rows
            if row.get("matched_action_discrepancy_total") is not None
        ),
    }


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_candidate: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_exposure: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_family: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_mode: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        candidate = str(row["candidate"])
        by_candidate[candidate].append(row)
        by_exposure[f"{candidate}|{row['exposure']}"] .append(row)
        by_family[f"{candidate}|{row['family']}"] .append(row)
        by_mode[f"{candidate}|{row['execution_mode']}"] .append(row)
    return {
        "overall": {key: _summary(value) for key, value in sorted(by_candidate.items())},
        "by_exposure": {key: _summary(value) for key, value in sorted(by_exposure.items())},
        "by_family": {key: _summary(value) for key, value in sorted(by_family.items())},
        "by_execution_mode": {key: _summary(value) for key, value in sorted(by_mode.items())},
    }


def _basic_metrics(result: SimulationResult, focal: object, candidate: str, exposure: int, scenario_index: int) -> dict[str, object]:
    focal_name = result.focal_agent_name
    assert focal_name is not None
    start = result.config.voting_start_day or result.config.marked_boundary_day
    live_days = [day for day in result.days if day.day >= start]
    actions = [int(day.agent_records[focal_name].action) for day in live_days]
    turnover = 0
    previous = 0
    for action in actions:
        turnover += abs(action - previous)
        previous = action
    daily = [int(day.agent_records[focal_name].daily_pnl) for day in result.days]
    running = 0
    high = 0
    dds: list[int] = []
    for value in daily:
        running += value
        high = max(high, running)
        dds.append(high - running)
    diagnostics = {}
    method = getattr(focal, "diagnostics", None)
    if callable(method):
        raw = method()
        if isinstance(raw, Mapping):
            diagnostics = dict(raw)
    gate = diagnostics.get("gate", {})
    if not isinstance(gate, Mapping):
        gate = {}
    activation_day = gate.get("activation_day")
    return {
        "scenario_index": scenario_index,
        "scenario_name": result.scenario_name,
        "family": result.scenario_configuration.get("family"),
        "execution_mode": result.scenario_configuration.get("execution_mode"),
        "population_size": result.scenario_configuration.get("population_size"),
        "candidate": candidate,
        "exposure": exposure,
        "pnl": int(result.marked_pnl[focal_name]),
        "mean_drawdown": statistics.fmean(dds) if dds else 0.0,
        "max_drawdown": max(dds, default=0),
        "beat_flat": int(result.marked_pnl[focal_name]) > 0,
        "active_days": sum(action != 0 for action in actions),
        "turnover": turnover,
        "activated": activation_day is not None,
        "activation_day": activation_day,
        "budget_breaches": sum(b.agent_name == focal_name for b in result.budget_breaches),
        "rejected_actions": sum(r.agent_name == focal_name for r in result.rejected_actions),
        "loss_stop": bool(diagnostics.get("loss_stop_active", False)),
        "drawdown_stop": bool(diagnostics.get("drawdown_stop_active", False)),
        "actual_pivotal_pnl": None,
        "actual_non_pivotal_pnl": None,
        "observable_shadow_pnl": None,
        "observable_pivotal_shadow_pnl": None,
        "observable_non_pivotal_shadow_pnl": None,
        "adjusted_evidence_pnl": None,
        "executable_shadow_pnl": None,
        "executable_pivotal_shadow_pnl": None,
        "executable_non_pivotal_shadow_pnl": None,
        "foregone_paper_opportunity": None,
        "observable_minus_executable_gap": None,
        "counterfactual_pivotal_paper_rate": None,
        "matched_action_discrepancy_max": None,
        "matched_action_discrepancy_total": None,
    }


def _hypothetical_majority(margin: int) -> MajorityOutcome:
    return majority_from_counts(max(margin, 0), max(-margin, 0))


def _majority_delta(majority: MajorityOutcome, config: object) -> int:
    if majority is MajorityOutcome.LONG:
        return int(config.long_majority_move)
    if majority is MajorityOutcome.SHORT:
        return int(config.short_majority_move)
    return 0


def _corrected_audit(result: SimulationResult, focal: object, row: dict[str, object]) -> dict[str, object]:
    focal_name = result.focal_agent_name
    assert focal_name is not None
    diagnostics = focal.diagnostics()
    paper_records = {
        int(record["day"]): record
        for record in diagnostics.get("paper_records", [])
        if isinstance(record, Mapping)
    }
    start = result.config.voting_start_day or result.config.marked_boundary_day
    observable = 0
    executable = 0
    actual_pivotal = 0
    actual_non_pivotal = 0
    observable_pivotal = 0
    observable_non_pivotal = 0
    executable_pivotal = 0
    executable_non_pivotal = 0
    pivotal_count = 0
    nonflat_count = 0
    pre_auth_observable = 0
    post_auth_observable = 0
    pre_auth_executable = 0
    post_auth_executable = 0
    matched_count = 0
    discrepancy_total = 0
    discrepancy_max = 0

    for source_day, source in enumerate(result.days[:-1]):
        if source.day < start:
            continue
        next_day = result.days[source_day + 1]
        record = paper_records.get(source.day, {"paper_action": 0, "stat_authorized": False})
        paper_action = int(record.get("paper_action", 0))
        actual_action = int(source.agent_records[focal_name].action)
        actual_increment = int(next_day.agent_records[focal_name].daily_pnl)
        observed_change = int(next_day.pnl_price_change or 0)
        usable_interval = bool(next_day.voting_active and not next_day.reset_applied)

        if usable_interval:
            observable += paper_action * observed_change
            if int(record.get("stat_authorized", False)):
                post_auth_observable += paper_action * observed_change
            else:
                pre_auth_observable += paper_action * observed_change
            if actual_action == paper_action:
                matched_count += 1
                discrepancy = actual_increment - paper_action * observed_change
                discrepancy_total += abs(discrepancy)
                discrepancy_max = max(discrepancy_max, abs(discrepancy))

        pivotal = False
        executable_reward = 0
        if usable_interval and paper_action != 0 and source.net_margin_before_focal is not None:
            nonflat_count += 1
            hypothetical_margin = int(source.net_margin_before_focal) + paper_action
            hypothetical_majority = _hypothetical_majority(hypothetical_margin)
            hypothetical_price = max(
                result.config.price_floor,
                source.price + _majority_delta(hypothetical_majority, result.config),
            )
            hypothetical_change = hypothetical_price - source.price
            pivotal = (
                hypothetical_majority is not source.majority
                or hypothetical_price != next_day.price
            )
            executable_reward = paper_action * hypothetical_change
            executable += executable_reward
            if int(record.get("stat_authorized", False)):
                post_auth_executable += executable_reward
            else:
                pre_auth_executable += executable_reward
            if pivotal:
                pivotal_count += 1

        if usable_interval:
            if pivotal:
                actual_pivotal += actual_increment
                observable_pivotal += paper_action * observed_change
                executable_pivotal += executable_reward
            else:
                actual_non_pivotal += actual_increment
                observable_non_pivotal += paper_action * observed_change
                executable_non_pivotal += executable_reward

    evidence_events = [
        event for event in diagnostics.get("evidence_events", [])
        if isinstance(event, Mapping)
    ]
    adjusted = sum(int(event.get("adjusted_gate_reward", 0)) for event in evidence_events)
    raw_evidence = sum(int(event.get("raw_observable_reward", 0)) for event in evidence_events)
    master_diag = diagnostics.get("master", {})
    if not isinstance(master_diag, Mapping):
        master_diag = {}
    actual_pnl = int(result.marked_pnl[focal_name])
    return {
        "actual_pivotal_pnl": actual_pivotal,
        "actual_non_pivotal_pnl": actual_non_pivotal,
        "observable_shadow_pnl": observable,
        "observable_pivotal_shadow_pnl": observable_pivotal,
        "observable_non_pivotal_shadow_pnl": observable_non_pivotal,
        "raw_evidence_pnl": raw_evidence,
        "adjusted_evidence_pnl": adjusted,
        "executable_shadow_pnl": executable,
        "executable_pivotal_shadow_pnl": executable_pivotal,
        "executable_non_pivotal_shadow_pnl": executable_non_pivotal,
        "foregone_paper_opportunity": executable - actual_pnl,
        "observable_minus_executable_gap": observable - executable,
        "counterfactual_pivotal_paper_rate": pivotal_count / nonflat_count if nonflat_count else 0.0,
        "counterfactual_pivotal_paper_count": pivotal_count,
        "nonflat_paper_count": nonflat_count,
        "matched_action_intervals": matched_count,
        "matched_action_discrepancy_total": discrepancy_total,
        "matched_action_discrepancy_max": discrepancy_max,
        "pre_authorization_observable_shadow_pnl": pre_auth_observable,
        "post_authorization_observable_shadow_pnl": post_auth_observable,
        "pre_authorization_executable_shadow_pnl": pre_auth_executable,
        "post_authorization_executable_shadow_pnl": post_auth_executable,
        "master_raw_observable_pnl": int(master_diag.get("raw_observable_pnl", raw_evidence)),
        "master_adjusted_evidence_pnl": int(master_diag.get("adjusted_evidence_pnl", adjusted)),
        "exposure_evaluation_count": int(diagnostics.get("exposure_evaluation_count", 0)),
        "headroom_gates": int(diagnostics.get("headroom_gate_count", 0)),
        "edge_gates": int(diagnostics.get("edge_gate_count", 0)),
        "unknown_gates": int(diagnostics.get("unknown_gate_count", 0)),
        "floor_gates": int(diagnostics.get("floor_gate_count", 0)),
        "stop_overshoot_max": max((int(v) for v in diagnostics.get("stop_overshoots", [])), default=0),
    }


def run_case(task: CaseTask) -> dict[str, object]:
    scenarios = _scenario_suite()
    scenario = scenarios[task.scenario_index]
    focal = _make_candidate(task.candidate, task.exposure)
    result = LiferaftSimulator(
        (focal, *scenario.population_factory()),
        scenario.config,
        focal_agent_name=focal.name,
        scenario_name=scenario.name,
        scenario_configuration=_scenario_configuration(scenario, task.exposure),
        random_seeds={"scenario": scenario.seed},
        other_portfolio_exposure={focal.name: task.exposure},
    ).run()
    row = _basic_metrics(result, focal, task.candidate, task.exposure, task.scenario_index)
    if task.candidate == NEW_CANDIDATE:
        row.update(_corrected_audit(result, focal, row))
    return row


def run_case_batch(tasks: Sequence[CaseTask]) -> list[dict[str, object]]:
    return [run_case(task) for task in tasks]


def _chunks(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def run_batches(
    tasks: Sequence[Any],
    worker: Any,
    *,
    workers: int,
    batch_size: int,
    label: str,
) -> tuple[list[Any], float]:
    batches = _chunks(list(tasks), batch_size)
    started = time.perf_counter()
    output: list[Any] = []
    print(f"{label}: {len(tasks)} items, {len(batches)} batches, workers={workers}", flush=True)
    if workers <= 1:
        for index, batch in enumerate(batches, 1):
            output.extend(worker(batch))
            if index == 1 or index == len(batches) or index % max(1, len(batches) // 20) == 0:
                print(f"{label}: {index}/{len(batches)}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, batch) for batch in batches]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                output.extend(future.result())
                if index == 1 or index == len(batches) or index % max(1, len(batches) // 20) == 0:
                    print(f"{label}: {index}/{len(batches)}", flush=True)
    return output, time.perf_counter() - started


def _synthetic_observation(
    *,
    day: int,
    change: int | None,
    own_position: int,
    base_price: int = 100_000,
) -> AgentObservation:
    if change is None:
        price = base_price
        previous_price = None
        history = (price,)
    else:
        previous_price = base_price
        price = base_price + change
        history = (previous_price, price)
    return AgentObservation(
        day=day,
        price=price,
        price_history=history,
        previous_price=previous_price,
        previous_price_change=change,
        previous_move_is_reset=False,
        is_reset_day=False,
        marked_boundary_day=365,
        price_floor=20_000,
        long_majority_move=-5_000,
        short_majority_move=8_000,
        position_limit=1,
        gross_portfolio_budget=600_000,
        own_position=own_position,
        voting_active=True,
        market_mode="inactive_until_marked",
        voting_start_day=365,
    )


def _synthetic_changes(rng: random.Random, process: str, horizon: int = 365) -> list[int]:
    result: list[int] = []
    last_bit: int | None = None
    for index in range(horizon):
        if process == "iid_null":
            q_short = 5 / 13
        elif process == "iid_q_050":
            q_short = 0.50
        elif process == "iid_q_025":
            q_short = 0.25
        elif process == "conditional_zero_edge":
            q_short = 5 / 13 if last_bit is None else (0.25 if last_bit == 0 else 0.60)
        elif process == "regime_switch":
            q_short = 0.25 if index < horizon // 2 else 0.50
        else:
            raise ValueError(process)
        last_bit = int(rng.random() < q_short)
        result.append(8_000 if last_bit else -5_000)
    return result


def _run_synthetic_path(process: str, index: int) -> dict[str, object]:
    rng = random.Random(stable_seed("pass6a1", process, index))
    changes = _synthetic_changes(rng, process)
    strategy = make_pass6a1_strategy(other_portfolio_exposure=0)
    held = strategy.decide(
        _synthetic_observation(day=365, change=None, own_position=0)
    )
    for offset, change in enumerate(changes, 1):
        held = strategy.decide(
            _synthetic_observation(day=365 + offset, change=change, own_position=held)
        )
    diagnostics = strategy.diagnostics()
    gate = diagnostics["gate"]
    master = diagnostics["master"]
    records = diagnostics["paper_records"]
    return {
        "process": process,
        "path_index": index,
        "activated": gate["activation_day"] is not None,
        "activation_day": gate["activation_day"],
        "observable_shadow_pnl": int(master["raw_observable_pnl"]),
        "adjusted_evidence_pnl": int(master["adjusted_evidence_pnl"]),
        "real_pnl": int(diagnostics["actual_realised_pnl"]),
        "active_days": sum(int(record["actual_action"] != 0) for record in records),
        "scoreable_count": int(gate["scoreable_count"]),
    }


def run_synthetic_batch(tasks: Sequence[tuple[str, int]]) -> list[dict[str, object]]:
    return [_run_synthetic_path(process, index) for process, index in tasks]


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _synthetic_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    activated = [row for row in rows if row["activated"]]
    return {
        "paths": len(rows),
        "activation_count": len(activated),
        "activation_rate": len(activated) / len(rows) if rows else 0.0,
        "activation_day_distribution": _distribution(
            [float(row["activation_day"]) for row in activated if row["activation_day"] is not None]
        ),
        "activation_delay_distribution": _distribution(
            [float(row["activation_day"]) - 365 for row in activated if row["activation_day"] is not None]
        ),
        "mean_observable_shadow_pnl": statistics.fmean(float(row["observable_shadow_pnl"]) for row in rows) if rows else 0.0,
        "mean_adjusted_evidence_pnl": statistics.fmean(float(row["adjusted_evidence_pnl"]) for row in rows) if rows else 0.0,
        "mean_real_pnl": statistics.fmean(float(row["real_pnl"]) for row in rows) if rows else 0.0,
        "mean_active_days": statistics.fmean(float(row["active_days"]) for row in rows) if rows else 0.0,
    }


def run_null(*, workers: int, quick: bool = False) -> tuple[dict[str, object], float]:
    null_count = 64 if quick else NULL_PATHS
    power_count = 64 if quick else POWER_PATHS
    null_tasks = [("iid_null", index) for index in range(null_count)]
    null_rows, null_runtime = run_batches(
        null_tasks,
        run_synthetic_batch,
        workers=workers,
        batch_size=8 if quick else NULL_BATCH_SIZE,
        label="Pass 6A.1 null",
    )
    null_rows.sort(key=lambda row: int(row["path_index"]))
    processes = ("iid_q_050", "iid_q_025", "conditional_zero_edge", "regime_switch")
    power_tasks = [(process, index) for process in processes for index in range(power_count)]
    power_rows, power_runtime = run_batches(
        power_tasks,
        run_synthetic_batch,
        workers=workers,
        batch_size=8 if quick else NULL_BATCH_SIZE,
        label="Pass 6A.1 power",
    )
    power_rows.sort(key=lambda row: (str(row["process"]), int(row["path_index"])))
    summaries = {"iid_null_q_5_13": _synthetic_summary(null_rows)}
    power_summaries = {
        process: _synthetic_summary([row for row in power_rows if row["process"] == process])
        for process in processes
    }
    null_summary = summaries["iid_null_q_5_13"]
    state = {
        "null": {
            "process": "iid_null_q_5_13",
            **null_summary,
            "observed_activation_exceeds_3pct": null_summary["activation_rate"] > 0.03,
            "sanity_pass": null_summary["activation_rate"] <= 0.03,
        },
        "power": power_summaries,
        "runtimes_seconds": {
            "null": null_runtime,
            "power": power_runtime,
            "total": null_runtime + power_runtime,
        },
    }
    return state, null_runtime + power_runtime


def build_tasks(quick: bool = False) -> list[CaseTask]:
    scenario_count = len(_scenario_suite())
    indices = list(range(scenario_count))
    candidates = list(EVALUATION_CANDIDATES)
    exposures = list(EXPOSURES)
    if quick:
        indices = indices[:2]
        candidates = [NEW_CANDIDATE, "flat"]
        exposures = [0, 450_000]
    return [
        CaseTask(index, candidate, exposure)
        for index in indices
        for candidate in candidates
        for exposure in exposures
    ]


def run_validation(*, workers: int, quick: bool = False) -> tuple[dict[str, object], float]:
    tasks = build_tasks(quick)
    rows, runtime = run_batches(
        tasks,
        run_case_batch,
        workers=workers,
        batch_size=4 if quick else EVAL_BATCH_SIZE,
        label="Pass 6A.1 validation",
    )
    rows.sort(key=lambda row: (int(row["scenario_index"]), str(row["candidate"]), int(row["exposure"])))
    return {
        "scenario_count": len(_scenario_suite()) if not quick else 2,
        "run_count": len(rows),
        "candidates": list(EVALUATION_CANDIDATES) if not quick else [NEW_CANDIDATE, "flat"],
        "exposures": list(EXPOSURES) if not quick else [0, 450_000],
        "rows": rows,
        "aggregates": _aggregate(rows),
    }, runtime


def run_reproducibility_check(workers: int) -> dict[str, object]:
    tasks = build_tasks(quick=True)[:4]
    serial, _ = run_batches(tasks, run_case_batch, workers=1, batch_size=2, label="Pass 6A.1 serial check")
    parallel, _ = run_batches(tasks, run_case_batch, workers=max(1, min(4, workers)), batch_size=2, label="Pass 6A.1 parallel check")
    serial.sort(key=lambda row: (row["scenario_index"], row["candidate"], row["exposure"]))
    parallel.sort(key=lambda row: (row["scenario_index"], row["candidate"], row["exposure"]))
    return {
        "bit_identical": serial == parallel,
        "serial_workers": 1,
        "parallel_workers": max(1, min(4, workers)),
        "serial_digest": hashlib.sha256(json.dumps(serial, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "parallel_digest": hashlib.sha256(json.dumps(parallel, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def _worker_count(raw: str, serial: bool) -> int:
    if serial:
        return 1
    if raw == "auto":
        return max(1, (os.cpu_count() or 1) - 1)
    value = int(raw)
    if value <= 0:
        raise ValueError("workers must be positive")
    return value


def _load_state() -> dict[str, object]:
    if not RESULTS_PATH.exists():
        return {}
    with RESULTS_PATH.open(encoding="utf-8") as stream:
        value = json.load(stream)
    return value if isinstance(value, dict) else {}


def _write_state(state: Mapping[str, object]) -> None:
    with RESULTS_PATH.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, indent=1, sort_keys=True, ensure_ascii=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pass 6A.1 corrected anytime evaluation")
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--null", action="store_true")
    phase.add_argument("--validation", action="store_true")
    phase.add_argument("--self-test", action="store_true")
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    workers = _worker_count(args.workers, args.serial)
    if args.self_test:
        run_self_checks()
        print("Pass 6A.1 self-checks: OK")
        return 0

    state = _load_state()
    state["workers"] = workers
    state.setdefault("commands", []).append(
        "python -m research.liferaft.pass6a1_experiments "
        + " ".join(argv or [])
    )
    state.setdefault("runtimes_seconds", {})
    state["constants"] = {
        "eta": FIXED_SHARE_ETA,
        "share_rate": FIXED_SHARE_RATE,
        "pivotal_probability": 0.10,
        "pivotal_haircut": 1_300,
        "minimum_residual_edge": 1_000,
        "adjusted_reward_bound": ADJUSTED_REWARD_BOUND,
        "anytime_alpha": 0.025,
        "bet_fraction": 0.5,
    }
    if args.null:
        state["reproducibility"] = run_reproducibility_check(workers)
        null_state, runtime = run_null(workers=workers, quick=args.quick)
        if not args.quick:
            state["null"] = null_state
            state["runtimes_seconds"]["null_and_power"] = runtime
            _write_state(state)
        print(json.dumps({"null": null_state["null"], "reproducibility": state["reproducibility"]}, indent=2, sort_keys=True))
        return 0

    validation_state, runtime = run_validation(workers=workers, quick=args.quick)
    if not args.quick:
        state["validation"] = validation_state
        state["runtimes_seconds"]["validation"] = runtime
        _write_state(state)
    print(json.dumps({"run_count": validation_state["run_count"], "runtime_seconds": runtime}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
