"""Pass 4A's bounded risk-wrapper experiment.

Only consumed ``validation_scenarios()`` are imported here.  In particular,
this module has no import or call path to the locked final scenario builder.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import mean, median
from typing import Callable, Iterable, Sequence

from .cold_start_strategies import make_cold_start_strategy
from .pass3_scenarios import Pass3Scenario, validation_scenarios
from .pass4_strategies import (
    ExposureSource,
    Risk50Burnin1Markov,
)
from .simulator import Agent, AgentDayRecord, LiferaftSimulator, SimulationResult


REPORT_PATH = Path(__file__).with_name("PASS4A_REPORT.md")
PASS4_STRATEGIES: tuple[str, ...] = (
    "flat",
    "burnin1_markov",
    "online_drift",
    Risk50Burnin1Markov.name,
)
PASS4_EXPOSURES: tuple[int, ...] = (0, 150_000, 300_000, 450_000)


@dataclass(frozen=True)
class Pass4Metric:
    scenario: str
    family: str
    execution_mode: str
    seed: int
    strategy: str
    other_portfolio_exposure: float
    marked_pnl: int
    marked_days: int
    pnl_per_marked_day: float
    maximum_drawdown: int
    active_days: int
    positive_pnl_hit_rate: float
    turnover: int
    budget_breaches: int
    rejected_actions: int
    pivotal_pnl: int
    non_pivotal_pnl: int
    flat_pnl: int
    best_pnl: int
    outperformed_flat: bool
    flat_tied_best: bool
    path_digest: str
    loss_stop_active: bool = False
    health_stop_active: bool = False
    first_stop_trigger: str | None = None
    loss_stop_day: int | None = None
    health_stop_day: int | None = None
    loss_pnl_before: int | None = None
    loss_pnl_after: int | None = None
    health_pnl_before: int | None = None
    health_pnl_after: int | None = None
    loss_limit_overshoot: int = 0
    unknown_pause_count: int = 0
    floor_gate_count: int = 0
    headroom_gate_count: int = 0
    raw_nonzero_requests: int = 0
    suppressed_by_loss: int = 0
    suppressed_by_health: int = 0
    suppressed_by_unknown: int = 0
    suppressed_by_floor: int = 0
    suppressed_by_headroom: int = 0


def _quartile(values: Sequence[float], fraction: float = 0.25) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def _marked_records(result: SimulationResult) -> tuple[AgentDayRecord, ...]:
    if result.focal_agent_name is None:
        raise ValueError("Pass 4 result must have a focal agent")
    start = result.config.voting_start_day
    assert start is not None
    return tuple(
        record
        for record in result.agent_history(result.focal_agent_name)
        if record.day > start
    )


def _path_digest(result: SimulationResult) -> str:
    start = result.config.voting_start_day
    assert start is not None
    values = (
        result.price_path[start:],
        tuple(day.majority.value for day in result.days[start:]),
    )
    return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()[:16]


class SharedExposure:
    """Use one fixed/callable exposure value for wrapper and simulator."""

    def __init__(self, source: ExposureSource, focal_name: str) -> None:
        self.source = source
        self.focal_name = focal_name
        self._cache: dict[int, float] = {}

    def __call__(self, observation) -> float:
        if observation.day not in self._cache:
            value = self.source(observation) if callable(self.source) else self.source
            # The wrapper performs the canonical validation.  Constructing a
            # temporary wrapper is unnecessary; the simulator repeats its own
            # public validation when it receives this cached value.
            if isinstance(value, bool):
                raise ValueError("other portfolio exposure must be numeric, not boolean")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid other portfolio exposure: {value!r}"
                ) from exc
            if not isfinite(numeric) or numeric < 0:
                raise ValueError(
                    "other portfolio exposure must be finite and non-negative: "
                    f"{value!r}"
                )
            self._cache[observation.day] = numeric
        return self._cache[observation.day]

    def simulator_callback(self, agent_name: str, observation) -> float:
        if agent_name == self.focal_name:
            return self(observation)
        return 0.0


def run_pass4_case(
    scenario: Pass3Scenario,
    strategy: str,
    *,
    other_portfolio_exposure: ExposureSource = 0.0,
) -> tuple[SimulationResult, Risk50Burnin1Markov | None]:
    """Run one fresh consumed scenario without changing Pass 3 sources."""

    if strategy != Risk50Burnin1Markov.name:
        if callable(other_portfolio_exposure):
            raise ValueError(
                "callable exposure sources are supported for the Pass 4 wrapper "
                "run; fixed candidates use numeric scenario exposure"
            )
        return (
            scenario.run(
                strategy,
                other_portfolio_exposure=other_portfolio_exposure,
            ),
            None,
        )

    focal = Risk50Burnin1Markov(
        other_portfolio_exposure=other_portfolio_exposure,
    )
    shared = SharedExposure(other_portfolio_exposure, focal.name)
    # Use the same shared provider object for the wrapper and simulator.  The
    # per-day cache prevents a callable from producing two values for one
    # public observation.
    focal.other_portfolio_exposure = shared
    opponents = scenario.population_factory()
    agents: tuple[Agent, ...] = (focal, *opponents)
    result = LiferaftSimulator(
        agents,
        scenario.config,
        other_portfolio_exposure=shared.simulator_callback,
        focal_agent_name=focal.name,
        scenario_name=scenario.name,
        scenario_configuration={
            "suite": "pass4a",
            "base_suite": "pass3_consumed_validation",
            "family": scenario.family,
            "seed": scenario.seed,
            "execution_mode": scenario.execution_mode,
            "population": scenario.population_description,
            "population_size": scenario.population_size,
            "pair_id": scenario.pair_id,
            "path_controlled": scenario.path_controlled,
            "other_portfolio_exposure": other_portfolio_exposure,
            "risk_wrapper": Risk50Burnin1Markov.name,
        },
        random_seeds={"scenario": scenario.seed},
    ).run()
    return result, focal


def _wrapper_event_values(
    agent: Risk50Burnin1Markov | None,
) -> dict[str, object]:
    if agent is None:
        return {}
    diagnostics = agent.diagnostics()
    loss_event = diagnostics["loss_stop_event"]
    health_event = diagnostics["health_stop_event"]
    return {
        "loss_stop_active": diagnostics["loss_stop_active"],
        "health_stop_active": diagnostics["health_stop_active"],
        "first_stop_trigger": diagnostics["first_stop_trigger"],
        "loss_stop_day": loss_event.day if loss_event is not None else None,
        "health_stop_day": health_event.day if health_event is not None else None,
        "loss_pnl_before": loss_event.pnl_before if loss_event is not None else None,
        "loss_pnl_after": loss_event.pnl_after if loss_event is not None else None,
        "health_pnl_before": health_event.pnl_before if health_event is not None else None,
        "health_pnl_after": health_event.pnl_after if health_event is not None else None,
        "loss_limit_overshoot": (
            loss_event.loss_limit_overshoot if loss_event is not None else 0
        ),
        "unknown_pause_count": diagnostics["unknown_pause_count"],
        "floor_gate_count": diagnostics["floor_gate_count"],
        "headroom_gate_count": diagnostics["headroom_gate_count"],
        "raw_nonzero_requests": diagnostics["raw_nonzero_requests"],
        "suppressed_by_loss": diagnostics["raw_nonzero_suppressed_by_gate"]["loss_stop"],
        "suppressed_by_health": diagnostics["raw_nonzero_suppressed_by_gate"]["health_stop"],
        "suppressed_by_unknown": diagnostics["raw_nonzero_suppressed_by_gate"]["unknown_pause"],
        "suppressed_by_floor": diagnostics["raw_nonzero_suppressed_by_gate"]["floor"],
        "suppressed_by_headroom": diagnostics["raw_nonzero_suppressed_by_gate"]["headroom"],
    }


def metric_without_comparison(
    result: SimulationResult,
    scenario: Pass3Scenario,
    strategy: str,
    exposure: float,
    wrapper: Risk50Burnin1Markov | None,
) -> Pass4Metric:
    if result.focal_agent_name is None:
        raise ValueError("Pass 4 result must have a focal agent")
    records = _marked_records(result)
    daily = [record.daily_pnl for record in records]
    cumulative = 0
    peak = 0
    drawdown = 0
    for value in daily:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)

    active_records = [record for record in records if record.pnl_position]
    active_days = len(active_records)
    hit_rate = (
        sum(record.daily_pnl > 0 for record in active_records) / active_days
        if active_days
        else 0.0
    )

    start = result.config.voting_start_day
    assert start is not None
    decision_records = tuple(
        record
        for record in result.agent_history(result.focal_agent_name)
        if record.day >= start
    )
    previous_action = 0
    turnover = 0
    for record in decision_records:
        turnover += abs(record.action - previous_action)
        previous_action = record.action

    decision_days = {record.day for record in decision_records}
    budget_breaches = sum(
        breach.agent_name == result.focal_agent_name and breach.day in decision_days
        for breach in result.budget_breaches
    )
    rejected_actions = sum(
        rejection.agent_name == result.focal_agent_name
        and rejection.day in decision_days
        for rejection in result.rejected_actions
    )

    pivotal_pnl = 0
    non_pivotal_pnl = 0
    for record in records:
        source_day = record.pnl_source_day
        pivotal = (
            source_day is not None
            and 0 <= source_day < len(result.days)
            and result.days[source_day].focal_pivotal
        )
        if pivotal:
            pivotal_pnl += record.daily_pnl
        else:
            non_pivotal_pnl += record.daily_pnl
    marked_pnl = result.marked_pnl[result.focal_agent_name]
    if pivotal_pnl + non_pivotal_pnl != marked_pnl:
        raise AssertionError(
            f"pivotal partition failed for {scenario.name}/{strategy}: "
            f"{pivotal_pnl}+{non_pivotal_pnl}!={marked_pnl}"
        )

    event_values = _wrapper_event_values(wrapper)
    return Pass4Metric(
        scenario=scenario.name,
        family=scenario.family,
        execution_mode=scenario.execution_mode,
        seed=scenario.seed,
        strategy=strategy,
        other_portfolio_exposure=exposure,
        marked_pnl=marked_pnl,
        marked_days=len(records),
        pnl_per_marked_day=marked_pnl / len(records) if records else 0.0,
        maximum_drawdown=drawdown,
        active_days=active_days,
        positive_pnl_hit_rate=hit_rate,
        turnover=turnover,
        budget_breaches=budget_breaches,
        rejected_actions=rejected_actions,
        pivotal_pnl=pivotal_pnl,
        non_pivotal_pnl=non_pivotal_pnl,
        flat_pnl=0,
        best_pnl=marked_pnl,
        outperformed_flat=False,
        flat_tied_best=False,
        path_digest=_path_digest(result),
        **event_values,
    )


def attach_comparisons(metrics: Sequence[Pass4Metric]) -> tuple[Pass4Metric, ...]:
    groups: dict[tuple[str, float], list[Pass4Metric]] = {}
    for metric in metrics:
        groups.setdefault((metric.scenario, metric.other_portfolio_exposure), []).append(metric)
    updated: list[Pass4Metric] = []
    for metric in metrics:
        values = groups[(metric.scenario, metric.other_portfolio_exposure)]
        flat = next(value.marked_pnl for value in values if value.strategy == "flat")
        best = max(value.marked_pnl for value in values)
        updated.append(
            Pass4Metric(
                **{
                    **metric.__dict__,
                    "flat_pnl": flat,
                    "best_pnl": best,
                    "outperformed_flat": metric.marked_pnl > flat,
                    "flat_tied_best": flat == best,
                }
            )
        )
    return tuple(updated)


def run_pass4_experiment(
    scenarios: Iterable[Pass3Scenario] | None = None,
    *,
    strategies: Sequence[str] = PASS4_STRATEGIES,
    exposures: Sequence[int] = PASS4_EXPOSURES,
    write_report: bool = True,
) -> tuple[Pass4Metric, ...]:
    """Run the frozen Pass 4A matrix over consumed validation cases only."""

    scenario_list = tuple(validation_scenarios() if scenarios is None else scenarios)
    invalid = set(strategies) - set(PASS4_STRATEGIES)
    if invalid:
        raise ValueError(f"unknown Pass 4A strategies: {sorted(invalid)}")
    if tuple(exposures) != tuple(PASS4_EXPOSURES):
        raise ValueError("Pass 4A exposures are frozen at the protocol levels")

    metrics: list[Pass4Metric] = []
    total = len(scenario_list) * len(exposures) * len(strategies)
    completed = 0
    for exposure in exposures:
        for scenario in scenario_list:
            for strategy in strategies:
                result, wrapper = run_pass4_case(
                    scenario,
                    strategy,
                    other_portfolio_exposure=exposure,
                )
                metrics.append(
                    metric_without_comparison(
                        result,
                        scenario,
                        strategy,
                        float(exposure),
                        wrapper,
                    )
                )
                completed += 1
            if completed % 480 == 0 or completed == total:
                print(f"[pass4a] completed {completed}/{total} cells")
    final_metrics = attach_comparisons(metrics)
    if write_report:
        REPORT_PATH.write_text(render_pass4_report(final_metrics), encoding="utf-8")
        print(f"wrote {REPORT_PATH}")
    return final_metrics


def _group(metrics: Sequence[Pass4Metric], *keys: str):
    grouped: dict[tuple[object, ...], list[Pass4Metric]] = {}
    for metric in metrics:
        grouped.setdefault(tuple(getattr(metric, key) for key in keys), []).append(metric)
    return grouped


def _summary(metrics: Sequence[Pass4Metric]) -> dict[str, float]:
    pnl = [metric.marked_pnl for metric in metrics]
    return {
        "runs": float(len(metrics)),
        "mean": mean(pnl) if pnl else 0.0,
        "median": median(pnl) if pnl else 0.0,
        "lower_quartile": _quartile(pnl),
        "worst": min(pnl) if pnl else 0.0,
        "mean_dd": mean(metric.maximum_drawdown for metric in metrics) if metrics else 0.0,
        "max_dd": max((metric.maximum_drawdown for metric in metrics), default=0),
        "active": mean(metric.active_days for metric in metrics) if metrics else 0.0,
        "hit": mean(metric.positive_pnl_hit_rate for metric in metrics) if metrics else 0.0,
        "turnover": mean(metric.turnover for metric in metrics) if metrics else 0.0,
        "breaches": mean(metric.budget_breaches for metric in metrics) if metrics else 0.0,
        "rejected": mean(metric.rejected_actions for metric in metrics) if metrics else 0.0,
        "beat_flat": mean(metric.outperformed_flat for metric in metrics) if metrics else 0.0,
        "flat_tied": mean(metric.flat_tied_best for metric in metrics) if metrics else 0.0,
        "pivotal": mean(metric.pivotal_pnl for metric in metrics) if metrics else 0.0,
        "non_pivotal": mean(metric.non_pivotal_pnl for metric in metrics) if metrics else 0.0,
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _overall_table(metrics: Sequence[Pass4Metric]) -> str:
    lines = [
        "| exposure | strategy | runs | mean P&L | median | lower quartile | worst | mean DD | max DD | active days | hit | turnover | breaches | rejected | beat flat | flat tied | pivotal P&L | non-pivotal P&L | unique paths |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for exposure in PASS4_EXPOSURES:
        for strategy in PASS4_STRATEGIES:
            values = [
                metric
                for metric in metrics
                if metric.other_portfolio_exposure == exposure
                and metric.strategy == strategy
            ]
            row = _summary(values)
            unique_paths = len({metric.path_digest for metric in values})
            lines.append(
                f"| {exposure:,} | `{strategy}` | {int(row['runs'])} | {row['mean']:.0f} | "
                f"{row['median']:.0f} | {row['lower_quartile']:.0f} | {row['worst']:.0f} | "
                f"{row['mean_dd']:.0f} | {row['max_dd']:.0f} | {row['active']:.1f} | "
                f"{_pct(row['hit'])} | {row['turnover']:.1f} | {row['breaches']:.2f} | "
                f"{row['rejected']:.2f} | {_pct(row['beat_flat'])} | "
                f"{_pct(row['flat_tied'])} | {row['pivotal']:.0f} | "
                f"{row['non_pivotal']:.0f} | {unique_paths}/{int(row['runs'])} |"
            )
    return "\n".join(lines)


def _family_table(metrics: Sequence[Pass4Metric]) -> str:
    lines = [
        "| exposure | family | strategy | runs | mean P&L | median | lower quartile | worst | mean DD | max DD | active days | hit | turnover | breaches | rejected | beat flat | flat tied | pivotal P&L | non-pivotal P&L |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (exposure, family, strategy), values in sorted(
        _group(metrics, "other_portfolio_exposure", "family", "strategy").items()
    ):
        row = _summary(values)
        lines.append(
            f"| {int(exposure):,} | {family} | `{strategy}` | {int(row['runs'])} | "
            f"{row['mean']:.0f} | {row['median']:.0f} | {row['lower_quartile']:.0f} | "
            f"{row['worst']:.0f} | {row['mean_dd']:.0f} | {row['max_dd']:.0f} | "
            f"{row['active']:.1f} | {_pct(row['hit'])} | {row['turnover']:.1f} | "
            f"{row['breaches']:.2f} | {row['rejected']:.2f} | "
            f"{_pct(row['beat_flat'])} | {_pct(row['flat_tied'])} | "
            f"{row['pivotal']:.0f} | {row['non_pivotal']:.0f} |"
        )
    return "\n".join(lines)


def _stop_table(metrics: Sequence[Pass4Metric]) -> str:
    lines = [
        "| exposure | runs | total loss stops | total health stops | both stops | both: loss first | both: health first | both: same observation | first loss (all stopped) | first health (all stopped) | mean loss stop day | median loss stop day | mean health stop day | median health stop day | mean loss P&L before/after | mean health P&L before/after | max overshoot | mean unknown pauses | mean floor gates | mean headroom gates | suppressed loss/health/unknown/floor/headroom |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for exposure in PASS4_EXPOSURES:
        values = [
            metric
            for metric in metrics
            if metric.strategy == Risk50Burnin1Markov.name
            and metric.other_portfolio_exposure == exposure
        ]
        loss = [metric for metric in values if metric.loss_stop_active]
        health = [metric for metric in values if metric.health_stop_active]
        both = [metric for metric in values if metric.loss_stop_active and metric.health_stop_active]
        loss_days = [metric.loss_stop_day for metric in loss if metric.loss_stop_day is not None]
        health_days = [metric.health_stop_day for metric in health if metric.health_stop_day is not None]
        # The requested ordering counts are specifically within the cases
        # where both sticky stops eventually activated.  The global first-
        # trigger counts remain separate so they cannot be mistaken for that
        # conditional population.
        both_loss_first = sum(
            metric.loss_stop_day is not None
            and metric.health_stop_day is not None
            and metric.loss_stop_day < metric.health_stop_day
            for metric in both
        )
        both_health_first = sum(
            metric.loss_stop_day is not None
            and metric.health_stop_day is not None
            and metric.health_stop_day < metric.loss_stop_day
            for metric in both
        )
        both_same_observation = sum(
            metric.loss_stop_day is not None
            and metric.loss_stop_day == metric.health_stop_day
            for metric in both
        )
        global_first_loss = sum(metric.first_stop_trigger == "loss_stop" for metric in values)
        global_first_health = sum(metric.first_stop_trigger == "health_stop" for metric in values)
        loss_before = [metric.loss_pnl_before for metric in loss if metric.loss_pnl_before is not None]
        loss_after = [metric.loss_pnl_after for metric in loss if metric.loss_pnl_after is not None]
        health_before = [metric.health_pnl_before for metric in health if metric.health_pnl_before is not None]
        health_after = [metric.health_pnl_after for metric in health if metric.health_pnl_after is not None]
        suppressed = [
            sum(metric.suppressed_by_loss for metric in values),
            sum(metric.suppressed_by_health for metric in values),
            sum(metric.suppressed_by_unknown for metric in values),
            sum(metric.suppressed_by_floor for metric in values),
            sum(metric.suppressed_by_headroom for metric in values),
        ]
        lines.append(
            f"| {exposure:,} | {len(values)} | {len(loss)} ({_pct(len(loss) / len(values))}) | "
            f"{len(health)} ({_pct(len(health) / len(values))}) | "
            f"{len(both)} ({_pct(len(both) / len(values))}) | "
            f"{both_loss_first} | {both_health_first} | {both_same_observation} | "
            f"{global_first_loss} | {global_first_health} | "
            f"{mean(loss_days) if loss_days else 0:.1f} | {median(loss_days) if loss_days else 0:.1f} | "
            f"{mean(health_days) if health_days else 0:.1f} | {median(health_days) if health_days else 0:.1f} | "
            f"{mean(loss_before) if loss_before else 0:.0f}/{mean(loss_after) if loss_after else 0:.0f} | "
            f"{mean(health_before) if health_before else 0:.0f}/{mean(health_after) if health_after else 0:.0f} | "
            f"{max((metric.loss_limit_overshoot for metric in values), default=0)} | "
            f"{mean(metric.unknown_pause_count for metric in values):.1f} | "
            f"{mean(metric.floor_gate_count for metric in values):.1f} | "
            f"{mean(metric.headroom_gate_count for metric in values):.1f} | "
            f"{'/'.join(str(item) for item in suppressed)} |"
        )
    return "\n".join(lines)


def _overshoot_table(metrics: Sequence[Pass4Metric]) -> str:
    """Render exact overshoot amounts conditional on loss-stop activation."""

    lines = [
        "## Loss-limit overshoot distribution",
        "",
        "The denominator is loss-stop cases at that exposure, and includes zero",
        "overshoot. A positive amount can only be created by the final newly",
        "observable adverse movement that crosses the sticky AUD 50,000 limit; no",
        "future or hidden information is used.",
        "",
        "| exposure | overshoot amount | loss-stop cases | percentage of loss-stop cases |",
        "|---:|---:|---:|---:|",
    ]
    for exposure in PASS4_EXPOSURES:
        loss = [
            metric
            for metric in metrics
            if metric.strategy == Risk50Burnin1Markov.name
            and metric.other_portfolio_exposure == exposure
            and metric.loss_stop_active
        ]
        counts = Counter(metric.loss_limit_overshoot for metric in loss)
        amounts = sorted({0, *counts})
        denominator = len(loss)
        for amount in amounts:
            count = counts.get(amount, 0)
            percentage = count / denominator if denominator else 0.0
            lines.append(
                f"| {exposure:,} | {amount:,} | {count} | {_pct(percentage)} |"
            )
    return "\n".join(lines)


def _paired_table(metrics: Sequence[Pass4Metric], right: str) -> str:
    rows = [
        "| exposure | family group | paired cases | mean wrapper minus comparison | median | wrapper wins | ties | wrapper losses |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    groups = _group(metrics, "other_portfolio_exposure", "family")
    for (exposure, family), values in sorted(groups.items()):
        by_case = {
            (metric.scenario, metric.other_portfolio_exposure): metric
            for metric in values
            if metric.strategy == Risk50Burnin1Markov.name
        }
        comparison = {
            (metric.scenario, metric.other_portfolio_exposure): metric
            for metric in values
            if metric.strategy == right
        }
        differences = [
            by_case[key].marked_pnl - comparison[key].marked_pnl
            for key in by_case.keys() & comparison.keys()
        ]
        if not differences:
            continue
        rows.append(
            f"| {int(exposure):,} | {family} | {len(differences)} | "
            f"{mean(differences):.0f} | {median(differences):.0f} | "
            f"{sum(value > 0 for value in differences)} | "
            f"{sum(value == 0 for value in differences)} | "
            f"{sum(value < 0 for value in differences)} |"
        )
    return "\n".join(rows)


def _paired_overall_table(metrics: Sequence[Pass4Metric], right: str) -> str:
    rows = [
        "| exposure | paired cases | mean wrapper minus comparison | median | wrapper wins | ties | wrapper losses |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for exposure in PASS4_EXPOSURES:
        values = [
            metric
            for metric in metrics
            if metric.other_portfolio_exposure == exposure
        ]
        left = {
            metric.scenario: metric
            for metric in values
            if metric.strategy == Risk50Burnin1Markov.name
        }
        comparison = {
            metric.scenario: metric
            for metric in values
            if metric.strategy == right
        }
        differences = [
            left[scenario].marked_pnl - comparison[scenario].marked_pnl
            for scenario in left.keys() & comparison.keys()
        ]
        if not differences:
            continue
        rows.append(
            f"| {exposure:,} | {len(differences)} | {mean(differences):.0f} | "
            f"{median(differences):.0f} | {sum(value > 0 for value in differences)} | "
            f"{sum(value == 0 for value in differences)} | "
            f"{sum(value < 0 for value in differences)} |"
        )
    return "\n".join(rows)


def _interpretation(metrics: Sequence[Pass4Metric]) -> str:
    lines = [
        "## Bounded interpretation",
        "",
        "The tables compare same-initial-scenario game counterfactuals. They are",
        "not fixed-path backtests: exposure gates can alter the focal vote, price",
        "path, reactive-opponent actions, and later budget feasibility.",
        "",
        "No strategy is selected from mean P&L alone. For auditability, the",
        "wrapper-versus-raw lower-tail and mean differences are shown below:",
        "",
        "| exposure | wrapper mean minus raw mean | wrapper lower quartile minus raw lower quartile | wrapper mean / raw mean | wrapper focal breaches |",
        "|---:|---:|---:|---:|---:|",
    ]
    for exposure in PASS4_EXPOSURES:
        wrapper = [
            metric
            for metric in metrics
            if metric.strategy == Risk50Burnin1Markov.name
            and metric.other_portfolio_exposure == exposure
        ]
        raw = [
            metric
            for metric in metrics
            if metric.strategy == "burnin1_markov"
            and metric.other_portfolio_exposure == exposure
        ]
        wrapper_summary = _summary(wrapper)
        raw_summary = _summary(raw)
        ratio = (
            wrapper_summary["mean"] / raw_summary["mean"]
            if raw_summary["mean"]
            else 0.0
        )
        lines.append(
            f"| {exposure:,} | {wrapper_summary['mean'] - raw_summary['mean']:.0f} | "
            f"{wrapper_summary['lower_quartile'] - raw_summary['lower_quartile']:.0f} | "
            f"{ratio:.2f} | {sum(metric.budget_breaches for metric in wrapper)} |"
        )
    lines.extend(
        [
            "",
            "A positive lower-quartile difference indicates improved consumed-",
            "validation downside relative to raw Markov at that exposure; it does",
            "not establish competition expected value. The wrapper is intended to",
            "prevent avoidable tail and budget risk, while its stop decisions may",
            "also change the endogenous game path. Pivotal losses remain separately",
            "visible in the overall and family tables.",
            "",
            "The risk candidate remains a research result, not a production",
            "recommendation. Inclusion in the locked final catalogue would require",
            "an explicit decision and a new lock; this experiment consumed no final",
            "cases.",
        ]
    )
    return "\n".join(lines)


def render_pass4_report(metrics: Sequence[Pass4Metric]) -> str:
    scenarios = len({metric.scenario for metric in metrics})
    cells = len(metrics)
    return f"""# Liferaft Pass 4A production-risk experiment

> This is a tightly bounded wrapper experiment over the consumed Pass 3
> validation suite. It did not construct or execute locked final scenarios.

## Protocol and scope

The only new candidate is `risk50_burnin1_markov`, which wraps a pristine
`make_cold_start_strategy("burnin1_markov")`. The frozen controls are the
AUD 50,000 sticky cumulative loss stop, the 8-observation/0.40/two-evaluation
forecast-health stop, one-decision unknown pause, exact-floor flat gate, and
AUD 10,000 full-portfolio reserve gate. Full details are in
`PASS4A_PROTOCOL.md`; no parameters were changed after observing results.

The experiment contains **{scenarios} consumed validation scenarios**, four
fixed candidates, and four fixed exposure levels: **{cells} cells**. The
exposure is a focal gross-exposure sensitivity, not a forecast of the complete
other-instrument portfolio.

## Overall validation results

{_overall_table(metrics)}

## Family results

Family rows are actual run-level summaries within each family; they are not
family-balanced quantile estimates.

{_family_table(metrics)}

## Wrapper stop and gate diagnostics

{_stop_table(metrics)}

Stop days and P&L before/after are conditional on the corresponding stop
activating. The `both: ...` columns condition trigger ordering on cases where
both stops eventually activated; `first ... (all stopped)` is the separate
global first-trigger count. Under the frozen update order, a same-observation
case would be reported separately rather than assigned to either strict order.

{_overshoot_table(metrics)}

## Paired P&L differences

### Wrapper minus raw `burnin1_markov`

#### Overall

{_paired_overall_table(metrics, "burnin1_markov")}

#### By family

{_paired_table(metrics, "burnin1_markov")}

### Wrapper minus `online_drift`

#### Overall

{_paired_overall_table(metrics, "online_drift")}

#### By family

{_paired_table(metrics, "online_drift")}

## Path interpretation

The `unique paths` column counts distinct live price/majority digests within
each strategy/exposure group. These are effective path-diversity diagnostics,
not independent samples. Because the focal action and the risk gates are part
of the vote, comparisons at different exposures or strategies can have
different subsequent price, majority, opponent, and budget paths. Treat all
comparisons as same-initial-scenario endogenous counterfactuals.

{_interpretation(metrics)}
"""


def main() -> None:
    run_pass4_experiment()


if __name__ == "__main__":
    main()
