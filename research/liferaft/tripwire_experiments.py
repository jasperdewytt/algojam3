"""Frozen IID, deterministic-power, and consumed-validation tripwire run."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .pass3_scenarios import validation_scenarios
from .simulator import (
    AgentObservation,
    LiferaftSimulator,
    MajorityOutcome,
    majority_from_counts,
)
from .tripwire_strategy import (
    BLOCK_SIZE,
    CYCLE_SPECIALIST_NAME,
    IMPACT_HAIRCUT,
    MAX_ACTUAL_LOSS,
    MAX_TRAILING_DRAWDOWN,
    PORTFOLIO_BUDGET,
    PORTFOLIO_HEADROOM_RESERVE,
    VARIANTS,
    make_tripwire_strategy,
    run_self_checks,
)


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "TRIPWIRE_RESULTS.json"
EXPOSURES = (0, 150_000, 300_000, 450_000)
IID_NULL_PATHS = 5_000
SYNTHETIC_HORIZON = 365
VALIDATION_BATCH_SIZE = len(VARIANTS) * len(EXPOSURES)
IID_BATCH_SIZE = 50
START_DAY = 365
SERIOUS_FAMILY_MEAN_LOSS = -10_000
MAX_POSITIVE_FAMILY_CONCENTRATION = 0.80


_SCENARIOS: tuple[Any, ...] | None = None


def _scenario_suite() -> tuple[Any, ...]:
    global _SCENARIOS
    if _SCENARIOS is None:
        _SCENARIOS = tuple(validation_scenarios())
    return _SCENARIOS


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _synthetic_observation(
    *, day: int, change: int | None, own_position: int
) -> AgentObservation:
    previous = None if change is None else 100_000
    price = 100_000 if change is None else 100_000 + change
    return AgentObservation(
        day=day,
        price=price,
        price_history=(price,) if previous is None else (previous, price),
        previous_price=previous,
        previous_price_change=change,
        previous_move_is_reset=False,
        is_reset_day=False,
        marked_boundary_day=START_DAY,
        price_floor=20_000,
        long_majority_move=-5_000,
        short_majority_move=8_000,
        position_limit=1,
        gross_portfolio_budget=PORTFOLIO_BUDGET,
        own_position=own_position,
        voting_active=True,
        market_mode="inactive_until_marked",
        voting_start_day=START_DAY,
    )


def _activation_events(diagnostics: Mapping[str, object]) -> list[dict[str, object]]:
    compact = []
    raw_events = diagnostics.get("activation_events", [])
    if not isinstance(raw_events, Sequence):
        return compact
    for event in raw_events:
        if not isinstance(event, Mapping):
            continue
        compact.append(
            {
                "day": int(event["day"]),
                "block": int(event["block"]),
                "action": int(event["action"]),
                "qualified_specialists": list(event["qualified_specialists"]),
                "cause_specialists": list(event["available_agreeing_specialists"]),
                "cycle_period": event.get("cycle_period"),
            }
        )
    return compact


def _deactivation_reasons(diagnostics: Mapping[str, object]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    raw_events = diagnostics.get("deactivation_events", [])
    if isinstance(raw_events, Sequence):
        for event in raw_events:
            if isinstance(event, Mapping):
                counts.update(str(reason) for reason in event.get("reasons", []))
    return dict(sorted(counts.items()))


def _drawdowns(increments: Sequence[int]) -> tuple[float, int]:
    running = 0
    high_water = 0
    drawdowns: list[int] = []
    for increment in increments:
        running += increment
        high_water = max(high_water, running)
        drawdowns.append(high_water - running)
    return (
        statistics.fmean(drawdowns) if drawdowns else 0.0,
        max(drawdowns, default=0),
    )


def _base_row(
    *,
    variant: str,
    diagnostics: Mapping[str, object],
    increments: Sequence[int],
) -> dict[str, object]:
    events = _activation_events(diagnostics)
    mean_drawdown, max_drawdown = _drawdowns(increments)
    records = diagnostics.get("paper_records", [])
    active_days = 0
    if isinstance(records, Sequence):
        active_days = sum(
            int(isinstance(record, Mapping) and int(record.get("actual_action", 0)) != 0)
            for record in records
        )
    overshoots = diagnostics.get("stop_overshoots", [])
    overshoot_values = [
        int(event.get("overshoot", 0))
        for event in overshoots
        if isinstance(event, Mapping)
    ] if isinstance(overshoots, Sequence) else []
    return {
        "variant": variant,
        "pnl": int(diagnostics["actual_realised_pnl"]),
        "beat_flat": int(diagnostics["actual_realised_pnl"]) > 0,
        "mean_drawdown": mean_drawdown,
        "max_drawdown": max_drawdown,
        "activated": bool(events),
        "activation_day": None if not events else int(events[0]["day"]),
        "activation_count": len(events),
        "activation_events": events,
        "active_days": active_days,
        "loss_stop": bool(diagnostics.get("loss_stop_active", False)),
        "drawdown_stop": bool(diagnostics.get("drawdown_stop_active", False)),
        "stop_overshoot_max": max(overshoot_values, default=0),
        "headroom_gates": int(diagnostics.get("headroom_gate_count", 0)),
        "deactivation_reasons": _deactivation_reasons(diagnostics),
    }


def _run_synthetic_sequence(
    variant: str,
    changes: Sequence[int],
) -> tuple[dict[str, object], Mapping[str, object]]:
    strategy = make_tripwire_strategy(variant, other_portfolio_exposure=0)
    held = strategy.decide(
        _synthetic_observation(day=START_DAY, change=None, own_position=0)
    )
    for offset, change in enumerate(changes, 1):
        held = strategy.decide(
            _synthetic_observation(
                day=START_DAY + offset,
                change=change,
                own_position=held,
            )
        )
    diagnostics = strategy.diagnostics()
    records = diagnostics["paper_records"]
    increments = [
        int(record["actual_increment"])
        for record in records
        if isinstance(record, Mapping)
    ]
    return _base_row(
        variant=variant,
        diagnostics=diagnostics,
        increments=increments,
    ), diagnostics


def _iid_task(task: tuple[str, int]) -> dict[str, object]:
    variant, path_index = task
    rng = random.Random(stable_seed("tripwire_iid_economic_null", path_index))
    changes = [
        8_000 if rng.random() < 5 / 13 else -5_000
        for _ in range(SYNTHETIC_HORIZON)
    ]
    row, _ = _run_synthetic_sequence(variant, changes)
    row["path_index"] = path_index
    return row


def run_iid_batch(tasks: Sequence[tuple[str, int]]) -> list[dict[str, object]]:
    return [_iid_task(task) for task in tasks]


FIXTURE_PATTERNS: dict[str, tuple[int, ...]] = {
    "persistent_positive": (1,),
    "persistent_negative": (0,),
    "alternating": (1, 0),
    "period_2": (0, 1),
    "period_3": (1, 1, 0),
    "period_5": (1, 1, 0, 1, 0),
    "period_8": (1, 1, 0, 1, 0, 0, 1, 0),
}


def _fixture_changes(name: str) -> list[int]:
    if name == "regime_change":
        bits = [1] * 120 + [0] * (SYNTHETIC_HORIZON - 120)
    else:
        pattern = FIXTURE_PATTERNS[name]
        bits = [pattern[index % len(pattern)] for index in range(SYNTHETIC_HORIZON)]
    return [8_000 if bit else -5_000 for bit in bits]


def _fixture_row(variant: str, fixture: str) -> dict[str, object]:
    changes = _fixture_changes(fixture)
    row, diagnostics = _run_synthetic_sequence(variant, changes)
    row["fixture"] = fixture
    first_event = row["activation_events"][0] if row["activation_events"] else None
    delay = None if first_event is None else int(first_event["day"]) - START_DAY
    next_change = None
    first_authorized_correct = False
    if first_event is not None and delay is not None and delay < len(changes):
        next_change = changes[delay]
        first_authorized_correct = int(first_event["action"]) * next_change > 0
    row["activation_delay"] = delay
    row["first_authorized_next_change"] = next_change
    row["first_authorized_correct"] = first_authorized_correct
    row["causal_detection"] = bool(
        first_event is not None
        and delay is not None
        and delay >= 2 * BLOCK_SIZE
        and first_authorized_correct
        and int(row["active_days"]) > 0
    )
    blocks = diagnostics.get("specialist_blocks", {})
    row["cycle_qualified_blocks"] = []
    if isinstance(blocks, Mapping):
        cycle_blocks = blocks.get(CYCLE_SPECIALIST_NAME, [])
        if isinstance(cycle_blocks, Sequence):
            row["cycle_qualified_blocks"] = [
                int(block["block"])
                for block in cycle_blocks
                if isinstance(block, Mapping) and bool(block.get("qualifies"))
            ]
    return row


def _majority_for_margin(margin: int) -> MajorityOutcome:
    return majority_from_counts(max(margin, 0), max(-margin, 0))


def _next_price(source_price: int, majority: MajorityOutcome, config: object) -> int:
    if majority is MajorityOutcome.LONG:
        delta = int(config.long_majority_move)
    elif majority is MajorityOutcome.SHORT:
        delta = int(config.short_majority_move)
    else:
        delta = 0
    return max(int(config.price_floor), source_price + delta)


def _paper_impact_audit(result: Any, diagnostics: Mapping[str, object]) -> dict[str, int]:
    focal_name = result.focal_agent_name
    assert focal_name is not None
    paper_records = {
        int(record["day"]): record
        for record in diagnostics.get("paper_records", [])
        if isinstance(record, Mapping)
    }
    start = (
        result.config.voting_start_day
        if result.config.voting_start_day is not None
        else result.config.marked_boundary_day
    )
    observable = 0
    executable = 0
    adjusted_observable = 0
    adjusted_executable = 0
    actual_pivotal = 0
    actual_non_pivotal = 0
    pivotal_count = 0
    nonflat_count = 0

    for index, source in enumerate(result.days[:-1]):
        if source.day < start:
            continue
        next_day = result.days[index + 1]
        if not next_day.voting_active or next_day.reset_applied:
            continue
        record = paper_records.get(source.day, {})
        paper_action = int(record.get("paper_action", 0))
        observed_change = int(next_day.pnl_price_change or 0)
        actual_increment = int(next_day.agent_records[focal_name].daily_pnl)
        pivotal = False
        executable_reward = 0
        if paper_action != 0:
            if source.net_margin_before_focal is None:
                raise AssertionError("missing opponents-only margin")
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
            if pivotal:
                pivotal_count += 1
        observable_reward = paper_action * observed_change
        observable += observable_reward
        executable += executable_reward
        if paper_action != 0:
            adjusted_observable += observable_reward - IMPACT_HAIRCUT
            adjusted_executable += executable_reward - IMPACT_HAIRCUT
        if pivotal:
            actual_pivotal += actual_increment
        else:
            actual_non_pivotal += actual_increment

    if actual_pivotal + actual_non_pivotal != int(result.marked_pnl[focal_name]):
        raise AssertionError("paper-pivotal actual P&L partition is incomplete")
    return {
        "observable_paper_pnl": observable,
        "one_step_executable_paper_pnl": executable,
        "adjusted_observable_paper_pnl": adjusted_observable,
        "adjusted_one_step_executable_paper_pnl": adjusted_executable,
        "actual_paper_pivotal_pnl": actual_pivotal,
        "actual_paper_non_pivotal_pnl": actual_non_pivotal,
        "paper_pivotal_intervals": pivotal_count,
        "paper_nonflat_intervals": nonflat_count,
    }


def _scenario_configuration(scenario: Any, exposure: int) -> dict[str, object]:
    return {
        "suite": "consumed_validation_tripwire",
        "family": scenario.family,
        "seed": scenario.seed,
        "execution_mode": scenario.execution_mode,
        "population_size": scenario.population_size,
        "other_portfolio_exposure": exposure,
        "market_mode": scenario.config.market_mode,
        "voting_start_day": scenario.config.voting_start_day,
    }


def _validation_task(task: tuple[int, str, int]) -> dict[str, object]:
    scenario_index, variant, exposure = task
    scenario = _scenario_suite()[scenario_index]
    focal = make_tripwire_strategy(
        variant, other_portfolio_exposure=exposure
    )
    result = LiferaftSimulator(
        (focal, *scenario.population_factory()),
        scenario.config,
        focal_agent_name=focal.name,
        scenario_name=scenario.name,
        scenario_configuration=_scenario_configuration(scenario, exposure),
        random_seeds={"scenario": scenario.seed},
        other_portfolio_exposure={focal.name: exposure},
    ).run()
    diagnostics = focal.diagnostics()
    focal_name = focal.name
    pnl = int(result.marked_pnl[focal_name])
    if int(diagnostics["actual_realised_pnl"]) != pnl:
        raise AssertionError("strategy and simulator actual P&L differ")
    start = (
        result.config.voting_start_day
        if result.config.voting_start_day is not None
        else result.config.marked_boundary_day
    )
    increments = [
        int(day.agent_records[focal_name].daily_pnl)
        for day in result.days
        if day.day >= start
    ]
    row = _base_row(
        variant=variant,
        diagnostics=diagnostics,
        increments=increments,
    )
    row.update(
        {
            "scenario_index": scenario_index,
            "scenario_name": scenario.name,
            "family": scenario.family,
            "execution_mode": scenario.execution_mode,
            "population_size": scenario.population_size,
            "exposure": exposure,
            "budget_breaches": sum(
                breach.agent_name == focal_name for breach in result.budget_breaches
            ),
            "rejected_actions": sum(
                rejection.agent_name == focal_name
                for rejection in result.rejected_actions
            ),
        }
    )
    row.update(_paper_impact_audit(result, diagnostics))
    return row


def run_validation_batch(
    tasks: Sequence[tuple[int, str, int]],
) -> list[dict[str, object]]:
    return [_validation_task(task) for task in tasks]


def _chunks(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _run_batches(
    tasks: Sequence[Any],
    worker: Any,
    *,
    workers: int,
    batch_size: int,
    label: str,
) -> tuple[list[dict[str, object]], float]:
    batches = _chunks(tasks, batch_size)
    started = time.perf_counter()
    output: list[dict[str, object]] = []
    print(
        f"{label}: {len(tasks)} items, {len(batches)} ordered batches, workers={workers}",
        flush=True,
    )
    if workers == 1:
        iterator = map(worker, batches)
        pool = None
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        iterator = pool.map(worker, batches, chunksize=1)
    try:
        for index, rows in enumerate(iterator, 1):
            output.extend(rows)
            interval = max(1, len(batches) // 10)
            if index == 1 or index == len(batches) or index % interval == 0:
                print(f"{label}: {index}/{len(batches)}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    return output, time.perf_counter() - started


def _pnl_distribution(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"mean": 0.0, "median": 0.0, "lower_quartile": 0.0, "worst": 0.0}
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "lower_quartile": ordered[int(0.25 * (len(ordered) - 1))],
        "worst": ordered[0],
    }


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    activated = [row for row in rows if bool(row.get("activated"))]
    activation_days = [
        int(row["activation_day"])
        for row in activated
        if row.get("activation_day") is not None
    ]
    event_specialists: Counter[str] = Counter()
    run_specialists: Counter[str] = Counter()
    combinations: Counter[str] = Counter()
    cycle_periods: Counter[str] = Counter()
    deactivations: Counter[str] = Counter()
    for row in rows:
        seen: set[str] = set()
        events = row.get("activation_events", [])
        if isinstance(events, Sequence):
            for event in events:
                if not isinstance(event, Mapping):
                    continue
                causes = tuple(str(name) for name in event.get("cause_specialists", []))
                event_specialists.update(causes)
                seen.update(causes)
                combinations["+".join(causes)] += 1
                if event.get("cycle_period") is not None and CYCLE_SPECIALIST_NAME in causes:
                    cycle_periods[str(event["cycle_period"])] += 1
        run_specialists.update(seen)
        reasons = row.get("deactivation_reasons", {})
        if isinstance(reasons, Mapping):
            deactivations.update(
                {str(key): int(value) for key, value in reasons.items()}
            )

    summary: dict[str, object] = {
        "runs": len(rows),
        "pnl": _pnl_distribution([float(row["pnl"]) for row in rows]),
        "maximum_drawdown": max(
            (float(row["max_drawdown"]) for row in rows), default=0.0
        ),
        "mean_max_drawdown": statistics.fmean(
            float(row["max_drawdown"]) for row in rows
        ) if rows else 0.0,
        "mean_drawdown": statistics.fmean(
            float(row["mean_drawdown"]) for row in rows
        ) if rows else 0.0,
        "beat_flat_rate": statistics.fmean(
            float(bool(row["beat_flat"])) for row in rows
        ) if rows else 0.0,
        "exactly_flat_rate": statistics.fmean(
            float(int(row["pnl"]) == 0) for row in rows
        ) if rows else 0.0,
        "activation_rate": len(activated) / len(rows) if rows else 0.0,
        "conditional_on_activation": {
            "runs": len(activated),
            "pnl": _pnl_distribution(
                [float(row["pnl"]) for row in activated]
            ),
            "win_rate": statistics.fmean(
                float(int(row["pnl"]) > 0) for row in activated
            ) if activated else 0.0,
        },
        "median_activation_day": (
            statistics.median(activation_days) if activation_days else None
        ),
        "active_days": {
            "total": sum(int(row["active_days"]) for row in rows),
            "mean": statistics.fmean(
                float(row["active_days"]) for row in rows
            ) if rows else 0.0,
            "conditional_mean": statistics.fmean(
                float(row["active_days"]) for row in activated
            ) if activated else 0.0,
        },
        "loss_stop_activations": sum(bool(row["loss_stop"]) for row in rows),
        "drawdown_stop_activations": sum(
            bool(row["drawdown_stop"]) for row in rows
        ),
        "maximum_stop_overshoot": max(
            (int(row["stop_overshoot_max"]) for row in rows), default=0
        ),
        "headroom_gates": sum(int(row["headroom_gates"]) for row in rows),
        "budget_breaches": sum(int(row.get("budget_breaches", 0)) for row in rows),
        "rejected_actions": sum(int(row.get("rejected_actions", 0)) for row in rows),
        "activation_causes": {
            "event_specialist_counts": dict(sorted(event_specialists.items())),
            "activated_run_specialist_counts": dict(sorted(run_specialists.items())),
            "combination_counts": dict(sorted(combinations.items())),
            "cycle_period_counts": dict(sorted(cycle_periods.items())),
        },
        "deactivation_reason_counts": dict(sorted(deactivations.items())),
    }
    if any("paper_nonflat_intervals" in row for row in rows):
        paper_nonflat = sum(int(row["paper_nonflat_intervals"]) for row in rows)
        pivotal = sum(int(row["paper_pivotal_intervals"]) for row in rows)
        summary["true_opponents_only_market_impact_pivotal_rate"] = (
            pivotal / paper_nonflat if paper_nonflat else 0.0
        )
        summary["paper_intervals"] = {
            "nonflat": paper_nonflat,
            "pivotal": pivotal,
        }
        summary["observable_vs_one_step_executable_paper_pnl"] = {
            "observable_total": sum(int(row["observable_paper_pnl"]) for row in rows),
            "one_step_executable_total": sum(
                int(row["one_step_executable_paper_pnl"]) for row in rows
            ),
            "adjusted_observable_total": sum(
                int(row["adjusted_observable_paper_pnl"]) for row in rows
            ),
            "adjusted_one_step_executable_total": sum(
                int(row["adjusted_one_step_executable_paper_pnl"])
                for row in rows
            ),
            "observable_mean_per_run": statistics.fmean(
                float(row["observable_paper_pnl"]) for row in rows
            ) if rows else 0.0,
            "one_step_executable_mean_per_run": statistics.fmean(
                float(row["one_step_executable_paper_pnl"]) for row in rows
            ) if rows else 0.0,
        }
        summary["actual_pnl_by_paper_pivotality"] = {
            "paper_pivotal_total": sum(
                int(row["actual_paper_pivotal_pnl"]) for row in rows
            ),
            "paper_non_pivotal_total": sum(
                int(row["actual_paper_non_pivotal_pnl"]) for row in rows
            ),
            "paper_pivotal_mean_per_run": statistics.fmean(
                float(row["actual_paper_pivotal_pnl"]) for row in rows
            ) if rows else 0.0,
            "paper_non_pivotal_mean_per_run": statistics.fmean(
                float(row["actual_paper_non_pivotal_pnl"]) for row in rows
            ) if rows else 0.0,
        }
    return summary


def _digest(rows: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _family_concentration(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    positive: Counter[str] = Counter()
    net: Counter[str] = Counter()
    family_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        family = str(row["family"])
        pnl = int(row["pnl"])
        positive[family] += max(0, pnl)
        net[family] += pnl
        family_rows[family].append(row)
    positive_total = sum(positive.values())
    shares = {
        family: value / positive_total if positive_total else 0.0
        for family, value in sorted(positive.items())
    }
    means = {
        family: statistics.fmean(float(row["pnl"]) for row in grouped)
        for family, grouped in sorted(family_rows.items())
    }
    top_family = max(shares, key=shares.get) if shares else None
    outside_exceptions = {
        family: mean
        for family, mean in means.items()
        if family not in {"history_rules", "periodic"}
        and mean < SERIOUS_FAMILY_MEAN_LOSS
    }
    return {
        "positive_pnl_total": positive_total,
        "positive_pnl_by_family": dict(sorted(positive.items())),
        "positive_pnl_share_by_family": shares,
        "top_positive_family": top_family,
        "top_positive_family_share": 0.0 if top_family is None else shares[top_family],
        "net_pnl_by_family": dict(sorted(net.items())),
        "mean_pnl_by_family": means,
        "seriously_negative_outside_history_rules_periodic": outside_exceptions,
        "seriously_negative_family_mean_threshold": SERIOUS_FAMILY_MEAN_LOSS,
    }


def _aggregate_validation(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_variant: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_exposure: dict[str, dict[str, list[Mapping[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_family: dict[str, dict[str, list[Mapping[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        variant = str(row["variant"])
        by_variant[variant].append(row)
        by_exposure[variant][str(row["exposure"])].append(row)
        by_family[variant][str(row["family"])].append(row)
    return {
        "scenario_count": len(_scenario_suite()),
        "run_count": len(rows),
        "exposures": list(EXPOSURES),
        "overall": {
            variant: _summary(by_variant[variant]) for variant in VARIANTS
        },
        "by_exposure": {
            variant: {
                key: _summary(grouped)
                for key, grouped in sorted(by_exposure[variant].items())
            }
            for variant in VARIANTS
        },
        "by_family": {
            variant: {
                key: _summary(grouped)
                for key, grouped in sorted(by_family[variant].items())
            }
            for variant in VARIANTS
        },
        "positive_pnl_family_concentration": {
            variant: _family_concentration(by_variant[variant])
            for variant in VARIANTS
        },
    }


def _screen(
    iid: Mapping[str, Mapping[str, object]],
    fixtures: Mapping[str, Sequence[Mapping[str, object]]],
    validation: Mapping[str, object],
) -> dict[str, object]:
    outcomes: dict[str, object] = {}
    passing: list[str] = []
    overall = validation["overall"]
    concentration = validation["positive_pnl_family_concentration"]
    for variant in VARIANTS:
        candidate = overall[variant]
        conditional = candidate["conditional_on_activation"]
        fixture_rows = fixtures[variant]
        family = concentration[variant]
        checks = {
            "iid_null_activation_no_more_than_2pct": float(
                iid[variant]["activation_rate"]
            ) <= 0.02,
            "conditional_activation_median_pnl_positive": float(
                conditional["pnl"]["median"]
            ) > 0,
            "conditional_activation_win_rate_at_least_65pct": float(
                conditional["win_rate"]
            ) >= 0.65,
            "worst_actual_pnl_at_least_minus_30000": float(
                candidate["pnl"]["worst"]
            ) >= -30_000,
            "maximum_drawdown_no_more_than_30000": float(
                candidate["maximum_drawdown"]
            ) <= 30_000,
            "zero_budget_breaches_and_rejected_actions": int(
                candidate["budget_breaches"]
            ) == 0 and int(candidate["rejected_actions"]) == 0,
            "all_deterministic_fixtures_detected_causally": all(
                bool(row["causal_detection"]) for row in fixture_rows
            ),
            "positive_pnl_not_over_80pct_one_family": float(
                family["top_positive_family_share"]
            ) <= MAX_POSITIVE_FAMILY_CONCENTRATION,
            "no_family_mean_below_minus_10000_outside_history_rules_periodic": not bool(
                family["seriously_negative_outside_history_rules_periodic"]
            ),
        }
        passed = all(checks.values())
        if passed:
            passing.append(variant)
        outcomes[variant] = {"checks": checks, "passes": passed}

    recommendation = "flat"
    if passing:
        recommendation = max(
            passing,
            key=lambda variant: (
                float(overall[variant]["conditional_on_activation"]["pnl"]["median"]),
                float(overall[variant]["conditional_on_activation"]["win_rate"]),
                float(overall[variant]["pnl"]["mean"]),
            ),
        )
    return {
        "candidates": outcomes,
        "passing_candidates": passing,
        "recommendation": recommendation,
        "interpretive_thresholds": {
            "maximum_positive_pnl_family_share": MAX_POSITIVE_FAMILY_CONCENTRATION,
            "seriously_negative_family_mean": SERIOUS_FAMILY_MEAN_LOSS,
        },
    }


def _worker_count(raw: str) -> int:
    if raw == "auto":
        return max(1, (os.cpu_count() or 1) - 1)
    workers = int(raw)
    if workers <= 0:
        raise ValueError("workers must be positive")
    return workers


def _write(state: Mapping[str, object]) -> None:
    with RESULTS_PATH.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, indent=1, sort_keys=True, ensure_ascii=True)


def run_experiment(*, workers: int) -> dict[str, object]:
    run_self_checks()
    if len(_scenario_suite()) != 480:
        raise AssertionError("the consumed validation suite must contain 480 scenarios")

    iid_tasks = [
        (variant, path_index)
        for variant in VARIANTS
        for path_index in range(IID_NULL_PATHS)
    ]
    iid_rows, iid_runtime = _run_batches(
        iid_tasks,
        run_iid_batch,
        workers=workers,
        batch_size=IID_BATCH_SIZE,
        label="Tripwire IID economic null",
    )
    iid_rows.sort(key=lambda row: (str(row["variant"]), int(row["path_index"])))
    iid = {
        variant: _summary(
            [row for row in iid_rows if row["variant"] == variant]
        )
        for variant in VARIANTS
    }

    fixture_rows = [
        _fixture_row(variant, fixture)
        for variant in VARIANTS
        for fixture in (*FIXTURE_PATTERNS, "regime_change")
    ]
    fixtures = {
        variant: [
            row for row in fixture_rows if row["variant"] == variant
        ]
        for variant in VARIANTS
    }

    validation_tasks = [
        (scenario_index, variant, exposure)
        for scenario_index in range(len(_scenario_suite()))
        for variant in VARIANTS
        for exposure in EXPOSURES
    ]
    validation_rows, validation_runtime = _run_batches(
        validation_tasks,
        run_validation_batch,
        workers=workers,
        batch_size=VALIDATION_BATCH_SIZE,
        label="Tripwire consumed validation",
    )
    validation_rows.sort(
        key=lambda row: (
            int(row["scenario_index"]),
            str(row["variant"]),
            int(row["exposure"]),
        )
    )
    validation = _aggregate_validation(validation_rows)
    screen = _screen(iid, fixtures, validation)
    state: dict[str, object] = {
        "workers": workers,
        "constants": {
            "variants": list(VARIANTS),
            "iid_economic_null_paths_per_variant": IID_NULL_PATHS,
            "iid_short_probability": 5 / 13,
            "synthetic_horizon": SYNTHETIC_HORIZON,
            "scoreable_block_size": BLOCK_SIZE,
            "impact_haircut": IMPACT_HAIRCUT,
            "actual_loss_stop": MAX_ACTUAL_LOSS,
            "trailing_drawdown_stop": MAX_TRAILING_DRAWDOWN,
            "portfolio_budget": PORTFOLIO_BUDGET,
            "portfolio_headroom_reserve": PORTFOLIO_HEADROOM_RESERVE,
        },
        "iid_economic_null": iid,
        "deterministic_fixtures": fixtures,
        "validation": validation,
        "screen": screen,
        "stable_order_digests": {
            "iid_rows_sha256": _digest(iid_rows),
            "validation_rows_sha256": _digest(validation_rows),
        },
        "runtimes_seconds": {
            "iid_economic_null": iid_runtime,
            "validation": validation_runtime,
        },
    }
    _write(state)
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen Liferaft tripwire experiment")
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        run_self_checks()
        print("Tripwire focused assertions: OK")
        return 0
    state = run_experiment(workers=_worker_count(args.workers))
    concise = {
        "recommendation": state["screen"]["recommendation"],
        "passing_candidates": state["screen"]["passing_candidates"],
        "iid_activation_rates": {
            variant: state["iid_economic_null"][variant]["activation_rate"]
            for variant in VARIANTS
        },
        "validation_runs": state["validation"]["run_count"],
        "runtimes_seconds": state["runtimes_seconds"],
    }
    print(json.dumps(concise, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
