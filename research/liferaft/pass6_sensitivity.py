"""Frozen pivotal-haircut sensitivity for the Pass 6A validation evidence.

This is an attribution-only diagnostic.  The 10% setting remains the sole
primary candidate and the sensitivity values are not used for selection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import pass6_experiments as experiments
from .pass6_strategies import make_pass6_strategy
from .simulator import LiferaftSimulator


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "PASS6A_SENSITIVITY.json"
CANDIDATES = ("fixed_checkpoint_fixed_share", "anytime_valid_fixed_share")
EXPOSURES = (0, 150_000, 300_000, 450_000)
PIVOTAL_PROBABILITIES = (0.0, 0.05, 0.10, 0.20)
BATCH_SIZE = 32


@dataclass(frozen=True)
class SensitivityTask:
    scenario_index: int
    candidate: str
    exposure: int
    pivotal_probability: float

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.scenario_index,
            self.candidate,
            self.exposure,
            self.pivotal_probability,
        )


def run_sensitivity_case(task: SensitivityTask) -> dict[str, object]:
    """Top-level Windows-picklable worker for one sensitivity case."""

    scenarios = experiments._scenario_suite("validation")
    scenario = scenarios[task.scenario_index]
    focal = make_pass6_strategy(
        task.candidate,
        other_portfolio_exposure=task.exposure,
        pivotal_probability=task.pivotal_probability,
    )
    result = LiferaftSimulator(
        (focal, *scenario.population_factory()),
        scenario.config,
        focal_agent_name=focal.name,
        scenario_name=scenario.name,
        scenario_configuration={
            "suite": "validation_pivotal_sensitivity",
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
            "pivotal_probability": task.pivotal_probability,
        },
        random_seeds={"scenario": scenario.seed},
        other_portfolio_exposure={focal.name: task.exposure},
    ).run()
    row = experiments._run_statistics(
        result,
        focal,
        phase="validation_pivotal_sensitivity",
        scenario_index=task.scenario_index,
        candidate=task.candidate,
        exposure=task.exposure,
    )
    row["pivotal_probability"] = task.pivotal_probability
    return row


def run_sensitivity_batch(tasks: Sequence[SensitivityTask]) -> list[dict[str, object]]:
    """Top-level worker batch; workers never write files."""

    return [run_sensitivity_case(task) for task in tasks]


def _workers(raw: str, serial: bool) -> int:
    if serial:
        return 1
    if raw == "auto":
        return max(1, (os.cpu_count() or 1) - 1)
    value = int(raw)
    if value <= 0:
        raise ValueError("--workers must be positive or auto")
    return value


def _tasks(quick: bool = False) -> list[SensitivityTask]:
    scenarios = experiments._scenario_suite("validation")
    indices = list(range(len(scenarios)))
    candidates = list(CANDIDATES)
    exposures = list(EXPOSURES)
    probabilities = list(PIVOTAL_PROBABILITIES)
    if quick:
        indices = indices[:2]
        exposures = [0, 450_000]
        probabilities = [0.0, 0.10, 0.20]
    return [
        SensitivityTask(index, candidate, exposure, probability)
        for index in indices
        for candidate in candidates
        for exposure in exposures
        for probability in probabilities
    ]


def _chunks(items: Sequence[SensitivityTask], size: int) -> list[Sequence[SensitivityTask]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _run(tasks: Sequence[SensitivityTask], workers: int) -> tuple[list[dict[str, object]], float]:
    batches = _chunks(tasks, BATCH_SIZE)
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    print(f"pivotal sensitivity: {len(tasks)} items in {len(batches)} batches, workers={workers}", flush=True)
    if workers == 1:
        for index, batch in enumerate(batches, start=1):
            rows.extend(run_sensitivity_batch(batch))
            if index == 1 or index == len(batches) or index % max(1, len(batches) // 20) == 0:
                print(f"pivotal sensitivity: {index}/{len(batches)} batches", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_sensitivity_batch, batch) for batch in batches]
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                rows.extend(future.result())
                if index == 1 or index == len(batches) or index % max(1, len(batches) // 20) == 0:
                    print(f"pivotal sensitivity: {index}/{len(batches)} batches", flush=True)
    rows.sort(key=lambda row: (
        int(row["scenario_index"]),
        str(row["candidate"]),
        int(row["exposure"]),
        float(row["pivotal_probability"]),
    ))
    return rows, time.perf_counter() - started


def _aggregate(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        key = f"{row['candidate']}|{row['exposure']}|{row['pivotal_probability']:.2f}"
        groups.setdefault(key, []).append(row)
    return {
        key: experiments._group_summary(group)
        for key, group in sorted(groups.items())
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen Pass 6A pivotal sensitivity")
    parser.add_argument("--workers", default="auto", help="process workers: N or auto")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    worker_count = _workers(args.workers, args.serial)
    tasks = _tasks(args.quick)
    rows, runtime = _run(tasks, worker_count)
    state = {
        "description": "Validation-only pivotal-haircut sensitivity; primary remains 10%.",
        "candidates": list(CANDIDATES),
        "exposures": list(EXPOSURES) if not args.quick else [0, 450_000],
        "pivotal_probabilities": list(PIVOTAL_PROBABILITIES) if not args.quick else [0.0, 0.10, 0.20],
        "scenario_count": len(experiments._scenario_suite("validation")) if not args.quick else 2,
        "run_count": len(rows),
        "workers": worker_count,
        "runtime_seconds": runtime,
        "quick": args.quick,
        "aggregates": _aggregate(rows),
        "rows": rows,
    }
    if not args.quick:
        RESULTS_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "run_count": len(rows),
        "runtime_seconds": runtime,
        "workers": worker_count,
        "quick": args.quick,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
