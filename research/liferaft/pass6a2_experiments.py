"""Rapid Pass 6A.2 mixture-gate calibration and consumed validation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .pass3_scenarios import validation_scenarios
from .pass6a1_experiments import (
    _basic_metrics,
    _scenario_configuration,
    _synthetic_changes,
    _synthetic_observation,
    run_batches,
    stable_seed,
)
from .pass6a1_strategy import ADJUSTED_REWARD_BOUND
from .pass6a2_strategy import LAMBDAS, make_pass6a2_strategy, run_self_checks
from .simulator import LiferaftSimulator, MajorityOutcome, majority_from_counts


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "PASS6A2_RESULTS.json"
EXPOSURES = (0, 150_000, 300_000, 450_000)
NULL_PATHS = 10_000
POWER_PATHS = 1_000
EVAL_BATCH_SIZE = 32


def _scenario_suite() -> tuple[Any, ...]:
    return tuple(validation_scenarios())


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
    return statistics.fmean(float(bool(row.get(key))) for row in rows) if rows else 0.0


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    nonflat = sum(int(row["nonflat_paper_count"]) for row in rows)
    pivotal = sum(int(row["market_impact_pivotal_count"]) for row in rows)
    comparisons = sum(int(row["paper_actual_comparison_count"]) for row in rows)
    divergences = sum(int(row["paper_actual_divergence_count"]) for row in rows)
    return {
        "runs": len(rows),
        "pnl": _pnl_summary([float(row["pnl"]) for row in rows]),
        "mean_drawdown": _mean(rows, "mean_drawdown"),
        "maximum_drawdown": max((float(row["max_drawdown"]) for row in rows), default=0.0),
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
        "actual_market_impact_pivotal_pnl": _mean(rows, "actual_market_impact_pivotal_pnl"),
        "actual_market_impact_non_pivotal_pnl": _mean(rows, "actual_market_impact_non_pivotal_pnl"),
        "observable_shadow_mean": _mean(rows, "observable_shadow_pnl"),
        "observable_pivotal_shadow_mean": _mean(rows, "observable_pivotal_shadow_pnl"),
        "observable_non_pivotal_shadow_mean": _mean(rows, "observable_non_pivotal_shadow_pnl"),
        "adjusted_evidence_mean": _mean(rows, "adjusted_evidence_pnl"),
        "one_step_executable_shadow_mean": _mean(rows, "one_step_executable_shadow_pnl"),
        "one_step_executable_pivotal_shadow_mean": _mean(rows, "one_step_executable_pivotal_shadow_pnl"),
        "one_step_executable_non_pivotal_shadow_mean": _mean(rows, "one_step_executable_non_pivotal_shadow_pnl"),
        "observable_minus_one_step_executable_mean": _mean(rows, "observable_minus_one_step_executable_gap"),
        "foregone_one_step_paper_opportunity_mean": _mean(rows, "foregone_one_step_paper_opportunity"),
        "pooled_market_impact_pivotal_rate": pivotal / nonflat if nonflat else 0.0,
        "mean_scenario_pivotal_rate": _mean(rows, "market_impact_pivotal_rate"),
        "market_impact_pivotal_count": pivotal,
        "nonflat_paper_count": nonflat,
        "paper_actual_divergence_rate": divergences / comparisons if comparisons else 0.0,
        "mean_scenario_paper_actual_divergence_rate": _mean(rows, "paper_actual_divergence_rate"),
        "paper_actual_divergence_count": divergences,
        "matched_action_discrepancy_max": max(
            (float(row["matched_action_discrepancy_max"]) for row in rows), default=0.0
        ),
        "matched_action_discrepancy_total": sum(
            float(row["matched_action_discrepancy_total"]) for row in rows
        ),
    }


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["family"])].append(row)
    return {key: _summary(value) for key, value in sorted(groups.items())}


def _majority_for_margin(margin: int) -> MajorityOutcome:
    return majority_from_counts(max(margin, 0), max(-margin, 0))


def _majority_delta(majority: MajorityOutcome, config: object) -> int:
    if majority is MajorityOutcome.LONG:
        return int(config.long_majority_move)
    if majority is MajorityOutcome.SHORT:
        return int(config.short_majority_move)
    return 0


def _next_price(source_price: int, majority: MajorityOutcome, config: object) -> int:
    return max(config.price_floor, source_price + _majority_delta(majority, config))


def _market_impact_audit(result: Any, focal: object) -> dict[str, object]:
    """Audit paper action against opponents-only baseline and actual path."""

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
    observable_pivotal = 0
    observable_non_pivotal = 0
    executable_pivotal = 0
    executable_non_pivotal = 0
    actual_pivotal = 0
    actual_non_pivotal = 0
    pivotal_count = 0
    nonflat_count = 0
    divergence_count = 0
    comparison_count = 0
    matched_count = 0
    discrepancy_total = 0
    discrepancy_max = 0

    for index, source in enumerate(result.days[:-1]):
        if source.day < start:
            continue
        next_day = result.days[index + 1]
        record = paper_records.get(source.day, {"paper_action": 0, "stat_authorized": False})
        paper_action = int(record.get("paper_action", 0))
        actual_action = int(source.agent_records[focal_name].action)
        actual_increment = int(next_day.agent_records[focal_name].daily_pnl)
        observed_change = int(next_day.pnl_price_change or 0)
        usable = bool(next_day.voting_active and not next_day.reset_applied)
        if not usable:
            continue

        pivotal = False
        executable_reward = 0
        observable_reward = paper_action * observed_change
        if paper_action != 0 and source.net_margin_before_focal is not None:
            nonflat_count += 1
            baseline_margin = int(source.net_margin_before_focal)
            baseline_majority = _majority_for_margin(baseline_margin)
            paper_majority = _majority_for_margin(baseline_margin + paper_action)
            baseline_price = _next_price(source.price, baseline_majority, result.config)
            paper_price = _next_price(source.price, paper_majority, result.config)
            pivotal = (
                baseline_majority is not paper_majority
                or baseline_price != paper_price
            )
            executable_reward = paper_action * (paper_price - source.price)
            observable += observable_reward
            executable += executable_reward
            if pivotal:
                pivotal_count += 1
                observable_pivotal += observable_reward
                executable_pivotal += executable_reward
                actual_pivotal += actual_increment
            else:
                observable_non_pivotal += observable_reward
                executable_non_pivotal += executable_reward

            # This is deliberately not called pivotality: it compares the
            # paper outcome with the actually realised action's outcome.
            comparison_count += 1
            if (
                paper_majority is not source.majority
                or paper_price != next_day.price
            ):
                divergence_count += 1
        else:
            actual_non_pivotal += actual_increment

        if not pivotal and paper_action != 0:
            actual_non_pivotal += actual_increment

        if actual_action == paper_action:
            matched_count += 1
            discrepancy = actual_increment - observable_reward
            discrepancy_total += abs(discrepancy)
            discrepancy_max = max(discrepancy_max, abs(discrepancy))

    evidence_events = [
        event for event in diagnostics.get("evidence_events", [])
        if isinstance(event, Mapping)
    ]
    adjusted = sum(int(event.get("adjusted_gate_reward", 0)) for event in evidence_events)
    master = diagnostics.get("master", {})
    if not isinstance(master, Mapping):
        master = {}
    gate = diagnostics.get("gate", {})
    if not isinstance(gate, Mapping):
        gate = {}
    actual_pnl = int(result.marked_pnl[focal_name])
    return {
        "actual_market_impact_pivotal_pnl": actual_pivotal,
        "actual_market_impact_non_pivotal_pnl": actual_non_pivotal,
        "observable_shadow_pnl": observable,
        "observable_pivotal_shadow_pnl": observable_pivotal,
        "observable_non_pivotal_shadow_pnl": observable_non_pivotal,
        "adjusted_evidence_pnl": adjusted,
        "one_step_executable_shadow_pnl": executable,
        "one_step_executable_pivotal_shadow_pnl": executable_pivotal,
        "one_step_executable_non_pivotal_shadow_pnl": executable_non_pivotal,
        "observable_minus_one_step_executable_gap": observable - executable,
        "foregone_one_step_paper_opportunity": executable - actual_pnl,
        "market_impact_pivotal_rate": pivotal_count / nonflat_count if nonflat_count else 0.0,
        "market_impact_pivotal_count": pivotal_count,
        "nonflat_paper_count": nonflat_count,
        "paper_actual_divergence_rate": divergence_count / comparison_count if comparison_count else 0.0,
        "paper_actual_divergence_count": divergence_count,
        "paper_actual_comparison_count": comparison_count,
        "matched_action_intervals": matched_count,
        "matched_action_discrepancy_total": discrepancy_total,
        "matched_action_discrepancy_max": discrepancy_max,
        "activation_lambda": gate.get("activation_lambda"),
        "final_mixture_e_value": gate.get("final_mixture_e_value"),
        "final_component_e_values": gate.get("final_component_e_values"),
        "master_raw_observable_pnl": int(master.get("raw_observable_pnl", 0)),
        "master_adjusted_evidence_pnl": int(master.get("adjusted_evidence_pnl", adjusted)),
        "headroom_gates": int(diagnostics.get("headroom_gate_count", 0)),
        "edge_gates": int(diagnostics.get("edge_gate_count", 0)),
        "unknown_gates": int(diagnostics.get("unknown_gate_count", 0)),
        "floor_gates": int(diagnostics.get("floor_gate_count", 0)),
        "stop_overshoot_max": max((int(v) for v in diagnostics.get("stop_overshoots", [])), default=0),
    }


def _compact_row(row: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "scenario_index", "family", "execution_mode", "population_size", "exposure",
        "pnl", "mean_drawdown", "max_drawdown", "beat_flat", "active_days", "turnover",
        "activated", "activation_day", "budget_breaches", "rejected_actions", "loss_stop",
        "drawdown_stop", "activation_lambda", "final_mixture_e_value",
        "actual_market_impact_pivotal_pnl", "actual_market_impact_non_pivotal_pnl",
        "observable_shadow_pnl", "observable_pivotal_shadow_pnl",
        "observable_non_pivotal_shadow_pnl", "adjusted_evidence_pnl",
        "one_step_executable_shadow_pnl", "one_step_executable_pivotal_shadow_pnl",
        "one_step_executable_non_pivotal_shadow_pnl", "observable_minus_one_step_executable_gap",
        "foregone_one_step_paper_opportunity", "market_impact_pivotal_rate",
        "market_impact_pivotal_count", "nonflat_paper_count", "paper_actual_divergence_rate",
        "paper_actual_divergence_count", "paper_actual_comparison_count",
        "matched_action_discrepancy_total", "matched_action_discrepancy_max",
    )
    return {key: row.get(key) for key in keys}


def run_case(task: tuple[int, int]) -> dict[str, object]:
    scenario_index, exposure = task
    scenario = _scenario_suite()[scenario_index]
    focal = make_pass6a2_strategy(other_portfolio_exposure=exposure)
    result = LiferaftSimulator(
        (focal, *scenario.population_factory()),
        scenario.config,
        focal_agent_name=focal.name,
        scenario_name=scenario.name,
        scenario_configuration=_scenario_configuration(scenario, exposure),
        random_seeds={"scenario": scenario.seed},
        other_portfolio_exposure={focal.name: exposure},
    ).run()
    row = _basic_metrics(
        result, focal, focal.name, exposure, scenario_index
    )
    row.update(_market_impact_audit(result, focal))
    return _compact_row(row)


def run_case_batch(tasks: Sequence[tuple[int, int]]) -> list[dict[str, object]]:
    return [run_case(task) for task in tasks]


def _synthetic_path(process: str, index: int) -> dict[str, object]:
    rng = random.Random(stable_seed("pass6a2", process, index))
    changes = _synthetic_changes(rng, process)
    strategy = make_pass6a2_strategy(other_portfolio_exposure=0)
    held = strategy.decide(_synthetic_observation(day=365, change=None, own_position=0))
    for offset, change in enumerate(changes, 1):
        held = strategy.decide(
            _synthetic_observation(day=365 + offset, change=change, own_position=held)
        )
    diagnostics = strategy.diagnostics()
    gate = diagnostics["gate"]
    master = diagnostics["master"]
    return {
        "activated": gate["activation_day"] is not None,
        "activation_day": gate["activation_day"],
        "activation_lambda": gate["activation_lambda"],
        "final_mixture_e_value": gate["final_mixture_e_value"],
        "final_component_e_values": gate["final_component_e_values"],
        "observable_shadow_pnl": int(master["raw_observable_pnl"]),
        "adjusted_evidence_pnl": int(master["adjusted_evidence_pnl"]),
        "real_pnl": int(diagnostics["actual_realised_pnl"]),
        "active_days": sum(int(record["actual_action"] != 0) for record in diagnostics["paper_records"]),
    }


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
    component_distributions = {
        str(lambda_value): _distribution(
            [float(row["final_component_e_values"][index]) for row in rows]
        )
        for index, lambda_value in enumerate(LAMBDAS)
    }
    return {
        "paths": len(rows),
        "activation_count": len(activated),
        "activation_rate": len(activated) / len(rows) if rows else 0.0,
        "median_activation_delay": statistics.median(
            [float(row["activation_day"]) - 365 for row in activated if row["activation_day"] is not None]
        ) if activated else None,
        "activation_day_distribution": _distribution(
            [float(row["activation_day"]) for row in activated if row["activation_day"] is not None]
        ),
        "final_mixture_e_value_distribution": _distribution(
            [float(row["final_mixture_e_value"]) for row in rows]
        ),
        "final_component_e_value_distributions": component_distributions,
        "activation_lambda_counts": {
            str(lambda_value): sum(row["activation_lambda"] == lambda_value for row in activated)
            for lambda_value in LAMBDAS
        },
        "mean_observable_shadow_pnl": statistics.fmean(float(row["observable_shadow_pnl"]) for row in rows) if rows else 0.0,
        "mean_adjusted_evidence_pnl": statistics.fmean(float(row["adjusted_evidence_pnl"]) for row in rows) if rows else 0.0,
        "mean_real_pnl": statistics.fmean(float(row["real_pnl"]) for row in rows) if rows else 0.0,
        "mean_active_days": statistics.fmean(float(row["active_days"]) for row in rows) if rows else 0.0,
    }


def run_null(*, workers: int) -> tuple[dict[str, object], float]:
    started = time.perf_counter()
    processes = ("iid_null", "iid_q_050", "iid_q_025", "conditional_zero_edge", "regime_switch")
    all_rows: dict[str, list[dict[str, object]]] = {}
    for process in processes:
        tasks = [(process, index) for index in range(NULL_PATHS if process == "iid_null" else POWER_PATHS)]
        batches = [tasks[index:index + 50] for index in range(0, len(tasks), 50)]
        print(f"Pass 6A.2 {process}: {len(tasks)} paths, workers={workers}", flush=True)

        def batch_worker(batch: Sequence[tuple[str, int]]) -> list[dict[str, object]]:
            return [_synthetic_path(name, index) for name, index in batch]

        # The worker must be top-level picklable on Windows; use the shared
        # top-level wrapper below rather than the local convenience closure.
        del batch_worker
        rows, _ = run_batches(
            tasks,
            run_synthetic_batch,
            workers=workers,
            batch_size=50,
            label=f"Pass 6A.2 {process}",
        )
        rows.sort(key=lambda row: int(row["path_index"]))
        all_rows[process] = rows
        if process == "iid_null" and _synthetic_summary(rows)["activation_rate"] > 0.03:
            break
    state = {
        "null": _synthetic_summary(all_rows["iid_null"]),
        "power": {
            process: _synthetic_summary(all_rows[process])
            for process in processes[1:]
            if process in all_rows
        },
        "fail_fast": _synthetic_summary(all_rows["iid_null"])["activation_rate"] > 0.03,
    }
    return state, time.perf_counter() - started


def run_synthetic_batch(tasks: Sequence[tuple[str, int]]) -> list[dict[str, object]]:
    output = []
    for process, index in tasks:
        row = _synthetic_path(process, index)
        row["process"] = process
        row["path_index"] = index
        output.append(row)
    return output


def _load_pass6a1_comparison() -> dict[str, object]:
    path = ROOT / "PASS6A1_RESULTS.json"
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    overall = data["validation"]["aggregates"]["overall"]
    fields = ("pnl", "beat_flat_rate", "maximum_drawdown", "active_days", "budget_breaches", "rejected_actions")
    return {
        name: {field: overall[name].get(field) for field in fields}
        for name in ("pass6a1_corrected_anytime_fixed_share", "pass6_anytime_fixed_share", "risk50_burnin1_markov")
    }


def _reproducibility_check(workers: int) -> dict[str, object]:
    tasks = [(0, 0), (0, 450_000), (1, 0), (1, 450_000)]
    serial, _ = run_batches(tasks, run_case_batch, workers=1, batch_size=2, label="Pass 6A.2 serial check")
    parallel, _ = run_batches(tasks, run_case_batch, workers=max(1, min(4, workers)), batch_size=2, label="Pass 6A.2 parallel check")
    serial.sort(key=lambda row: (row["scenario_index"], row["exposure"]))
    parallel.sort(key=lambda row: (row["scenario_index"], row["exposure"]))
    serial_json = json.dumps(serial, sort_keys=True, separators=(",", ":"))
    parallel_json = json.dumps(parallel, sort_keys=True, separators=(",", ":"))
    return {
        "bit_identical": serial == parallel,
        "serial_workers": 1,
        "parallel_workers": max(1, min(4, workers)),
        "serial_digest": hashlib.sha256(serial_json.encode()).hexdigest(),
        "parallel_digest": hashlib.sha256(parallel_json.encode()).hexdigest(),
    }


def run_validation(*, workers: int) -> tuple[dict[str, object], float]:
    tasks = [(scenario_index, exposure) for scenario_index in range(len(_scenario_suite())) for exposure in EXPOSURES]
    rows, runtime = run_batches(
        tasks,
        run_case_batch,
        workers=workers,
        batch_size=EVAL_BATCH_SIZE,
        label="Pass 6A.2 validation",
    )
    rows.sort(key=lambda row: (int(row["scenario_index"]), int(row["exposure"])))
    by_exposure: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_family: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_mode: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_exposure[str(row["exposure"])].append(row)
        by_family[str(row["family"])].append(row)
        by_mode[str(row["execution_mode"])].append(row)
    return {
        "scenario_count": len(_scenario_suite()),
        "run_count": len(rows),
        "exposures": list(EXPOSURES),
        "overall": _summary(rows),
        "by_exposure": {key: _summary(value) for key, value in sorted(by_exposure.items())},
        "by_family": {key: _summary(value) for key, value in sorted(by_family.items())},
        "by_execution_mode": {key: _summary(value) for key, value in sorted(by_mode.items())},
        "rows": rows,
    }, runtime


def _worker_count(raw: str, serial: bool) -> int:
    if serial:
        return 1
    if raw == "auto":
        return max(1, (os.cpu_count() or 1) - 1)
    value = int(raw)
    if value <= 0:
        raise ValueError("workers must be positive")
    return value


def _write(state: Mapping[str, object]) -> None:
    with RESULTS_PATH.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, indent=1, sort_keys=True, ensure_ascii=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pass 6A.2 mixture-gate evaluation")
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--null", action="store_true")
    phase.add_argument("--validation", action="store_true")
    phase.add_argument("--self-test", action="store_true")
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--serial", action="store_true")
    args = parser.parse_args(argv)
    workers = _worker_count(args.workers, args.serial)
    if args.self_test:
        run_self_checks()
        print("Pass 6A.2 self-checks: OK")
        return 0

    state: dict[str, object] = {
        "workers": workers,
        "constants": {
            "lambdas": list(LAMBDAS),
            "alpha": 0.025,
            "adjusted_reward_bound": ADJUSTED_REWARD_BOUND,
            "threshold": 40.0,
        },
    }
    if args.null:
        state["reproducibility"] = _reproducibility_check(workers)
        null_state, runtime = run_null(workers=workers)
        state["null_and_power"] = null_state
        state["runtimes_seconds"] = {"null_and_power": runtime}
        _write(state)
        print(json.dumps({"null": null_state["null"], "reproducibility": state["reproducibility"]}, indent=2, sort_keys=True))
        return 2 if null_state["fail_fast"] else 0

    existing = RESULTS_PATH
    if not existing.exists():
        raise RuntimeError("run --null before --validation")
    with existing.open(encoding="utf-8") as stream:
        state = json.load(stream)
    if state.get("null_and_power", {}).get("fail_fast"):
        print("Pass 6A.2 validation skipped: null activation exceeded 3%")
        return 2
    validation, runtime = run_validation(workers=workers)
    state["validation"] = validation
    state["comparison_pass6a1"] = _load_pass6a1_comparison()
    state.setdefault("runtimes_seconds", {})["validation"] = runtime
    _write(state)
    print(json.dumps({"run_count": validation["run_count"], "runtime_seconds": runtime}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
