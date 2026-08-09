"""Reproducible Pass 3 cold-start experiments.

Normal execution runs development diagnostics and validation evidence as
separate suites. The locked final suite is imported and instantiated only
after an explicit ``--final`` request and writes a separate report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

from .cold_start_strategies import COLD_START_STRATEGY_NAMES
from .pass3_scenarios import (
    Pass3Scenario,
    development_scenarios,
    validation_scenarios,
)
from .simulator import AgentDayRecord, SimulationResult


REPORT_PATH = Path(__file__).with_name("PASS3_REPORT.md")
FINAL_REPORT_PATH = Path(__file__).with_name("PASS3_FINAL_REPORT.md")
PORTFOLIO_REPORT_PATH = Path(__file__).with_name("PASS31_PORTFOLIO_SENSITIVITY.md")
PATH_DIVERGENCE_REPORT_PATH = Path(__file__).with_name("PASS31_PATH_DIVERGENCE.md")
PORTFOLIO_EXPOSURES: tuple[int, ...] = (0, 150_000, 300_000, 450_000)
PATH_DIVERGENCE_STRATEGY = "burnin1_markov"
PORTFOLIO_STRATEGIES: tuple[str, ...] = (
    "flat",
    "burnin1_markov",
    "burnin3_markov",
    "online_markov",
    "online_drift",
)


@dataclass(frozen=True)
class RunMetric:
    suite: str
    scenario: str
    family: str
    execution_mode: str
    seed: int
    strategy: str
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
    regret_against_best: int
    other_portfolio_exposure: float = 0.0


@dataclass(frozen=True)
class PathAuditRecord:
    suite: str
    family: str
    seed: int
    execution_mode: str
    pair_id: str | None
    path_controlled: bool
    signature: tuple[tuple[int, ...], tuple[str, ...]]


@dataclass(frozen=True)
class PathDiversitySummary:
    """Exact uniqueness plus simple within-family path-distance diagnostics."""

    suite: str
    family: str
    execution_mode: str
    cases: int
    unique_signatures: int
    duplicate_cases: int
    minimum_pairwise_hamming_days: int
    mean_pairwise_hamming_days: float


@dataclass(frozen=True)
class LivePathDifference:
    """Differences between two fresh runs of the same scenario.

    The counts cover the live segment from the voting-start observation. The
    opponent count deliberately excludes the focal agent and compares
    effective engine actions, so budget-forced flattening is visible even when
    an opponent's raw day-indexed request is unchanged.
    """

    differing_price_days: int
    differing_majority_days: int
    differing_opponent_action_cells: int


@dataclass(frozen=True)
class ExposurePathDivergenceSummary:
    """Path-divergence diagnostics, not a strategy-ranking statistic."""

    baseline_exposure: float
    exposure: float
    group: str
    cases: int
    different_price_path_fraction: float
    different_majority_path_fraction: float
    mean_differing_price_days: float
    maximum_differing_price_days: int
    mean_differing_majority_days: float
    maximum_differing_majority_days: int
    mean_differing_opponent_action_cells: float
    maximum_differing_opponent_action_cells: int


def _quartile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def _marked_records(result: SimulationResult) -> tuple[AgentDayRecord, ...]:
    start = result.config.voting_start_day
    assert start is not None
    if result.focal_agent_name is None:
        raise ValueError("Pass 3 result must have a focal agent")
    history = result.agent_history(result.focal_agent_name)
    return tuple(record for record in history if record.day > start)


def _metric_without_comparison(
    result: SimulationResult,
    *,
    suite: str,
    scenario: Pass3Scenario,
    strategy: str,
    other_portfolio_exposure: float = 0.0,
) -> RunMetric:
    if result.focal_agent_name is None:
        raise ValueError("Pass 3 result must have a focal agent")
    records = _marked_records(result)
    marked_days = len(records)
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

    marked_diag_days = {record.day for record in decision_records}
    budget_breaches = sum(
        breach.agent_name == result.focal_agent_name and breach.day in marked_diag_days
        for breach in result.budget_breaches
    )
    rejected_actions = sum(
        rejection.agent_name == result.focal_agent_name
        and rejection.day in marked_diag_days
        for rejection in result.rejected_actions
    )

    # P&L is categorized by the vote that generated the realized interval.
    # A missing/invalid source is conservatively treated as non-pivotal so the
    # two buckets remain an exhaustive partition of marked P&L.
    pivotal_pnl = 0
    non_pivotal_pnl = 0
    for record in records:
        source_day = record.pnl_source_day
        source_is_pivotal = (
            source_day is not None
            and 0 <= source_day < len(result.days)
            and result.days[source_day].focal_pivotal
        )
        if source_is_pivotal:
            pivotal_pnl += record.daily_pnl
        else:
            non_pivotal_pnl += record.daily_pnl

    marked_pnl = result.marked_pnl[result.focal_agent_name]
    if pivotal_pnl + non_pivotal_pnl != marked_pnl:
        raise AssertionError(
            f"pivotal P&L partition failed for {scenario.name}/{strategy}: "
            f"{pivotal_pnl}+{non_pivotal_pnl}!={marked_pnl}"
        )
    return RunMetric(
        suite=suite,
        scenario=scenario.name,
        family=scenario.family,
        execution_mode=scenario.execution_mode,
        seed=scenario.seed,
        strategy=strategy,
        marked_pnl=marked_pnl,
        marked_days=marked_days,
        pnl_per_marked_day=marked_pnl / marked_days if marked_days else 0.0,
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
        regret_against_best=0,
        other_portfolio_exposure=other_portfolio_exposure,
    )


def _attach_comparisons(metrics: Sequence[RunMetric]) -> tuple[RunMetric, ...]:
    by_scenario: dict[str, list[RunMetric]] = {}
    for metric in metrics:
        by_scenario.setdefault(metric.scenario, []).append(metric)
    flat_by_scenario = {
        scenario: next(
            metric.marked_pnl for metric in values if metric.strategy == "flat"
        )
        for scenario, values in by_scenario.items()
    }
    updated: list[RunMetric] = []
    for metric in metrics:
        values = by_scenario[metric.scenario]
        best = max(value.marked_pnl for value in values)
        flat = flat_by_scenario[metric.scenario]
        updated.append(
            RunMetric(
                **{
                    **metric.__dict__,
                    "flat_pnl": flat,
                    "best_pnl": best,
                    "outperformed_flat": metric.marked_pnl > flat,
                    "flat_tied_best": flat == best,
                    "regret_against_best": best - metric.marked_pnl,
                }
            )
        )
    return tuple(updated)


def run_suite(
    scenarios: Iterable[Pass3Scenario],
    *,
    suite: str,
    strategies: Sequence[str] = COLD_START_STRATEGY_NAMES,
    path_audit: list[PathAuditRecord] | None = None,
    other_portfolio_exposure: float = 0.0,
) -> tuple[RunMetric, ...]:
    """Run fresh focal agents and opponent populations for each case."""

    metrics: list[RunMetric] = []
    scenario_list = tuple(scenarios)
    for scenario_index, scenario in enumerate(scenario_list, start=1):
        print(
            f"[{suite}] scenario {scenario_index}/{len(scenario_list)} "
            f"{scenario.name} ({scenario.execution_mode})"
        )
        for strategy in strategies:
            result = scenario.run(
                strategy,
                other_portfolio_exposure=other_portfolio_exposure,
            )
            if path_audit is not None and strategy == "flat":
                path_audit.append(
                    PathAuditRecord(
                        suite=suite,
                        family=scenario.family,
                        seed=scenario.seed,
                        execution_mode=scenario.execution_mode,
                        pair_id=scenario.pair_id,
                        path_controlled=scenario.path_controlled,
                        signature=scenario.live_path_signature(result),
                    )
                )
            metrics.append(
                _metric_without_comparison(
                    result,
                    suite=suite,
                    scenario=scenario,
                    strategy=strategy,
                    other_portfolio_exposure=other_portfolio_exposure,
                )
            )
    return _attach_comparisons(metrics)


def _group(
    metrics: Sequence[RunMetric],
    *keys: str,
) -> dict[tuple[object, ...], list[RunMetric]]:
    grouped: dict[tuple[object, ...], list[RunMetric]] = {}
    for metric in metrics:
        grouped.setdefault(tuple(getattr(metric, key) for key in keys), []).append(metric)
    return grouped


def _actual_summary(metrics: Sequence[RunMetric]) -> dict[str, float]:
    """Statistics over actual runs, not averages of family statistics."""

    pnl = [metric.marked_pnl for metric in metrics]
    return {
        "runs": float(len(metrics)),
        "mean": mean(pnl) if pnl else 0.0,
        "median": median(pnl) if pnl else 0.0,
        "lower_quartile": _quartile(pnl, 0.25),
        "worst": min(pnl) if pnl else 0.0,
        "mean_per_day": mean(metric.pnl_per_marked_day for metric in metrics) if metrics else 0.0,
        "mean_drawdown": mean(metric.maximum_drawdown for metric in metrics) if metrics else 0.0,
        "max_drawdown": max((metric.maximum_drawdown for metric in metrics), default=0.0),
        "mean_active": mean(metric.active_days for metric in metrics) if metrics else 0.0,
        "hit_rate": mean(metric.positive_pnl_hit_rate for metric in metrics) if metrics else 0.0,
        "mean_turnover": mean(metric.turnover for metric in metrics) if metrics else 0.0,
        "breaches": mean(metric.budget_breaches for metric in metrics) if metrics else 0.0,
        "rejections": mean(metric.rejected_actions for metric in metrics) if metrics else 0.0,
        "beat_flat": mean(metric.outperformed_flat for metric in metrics) if metrics else 0.0,
        "flat_tied": mean(metric.flat_tied_best for metric in metrics) if metrics else 0.0,
        "mean_regret": mean(metric.regret_against_best for metric in metrics) if metrics else 0.0,
        "pivotal": mean(metric.pivotal_pnl for metric in metrics) if metrics else 0.0,
        "non_pivotal": mean(metric.non_pivotal_pnl for metric in metrics) if metrics else 0.0,
    }


def _family_balanced_means(
    metrics: Sequence[RunMetric],
    strategy: str,
) -> dict[str, float]:
    """Only mean P&L quantities are family-balanced.

    Quantiles, worst runs, drawdowns, and fractions are never averaged across
    family summaries; those remain in ``_actual_summary``.
    """

    family_groups = _group(
        [metric for metric in metrics if metric.strategy == strategy], "family"
    )
    family_pnl = [mean(metric.marked_pnl for metric in values) for values in family_groups.values()]
    family_per_day = [
        mean(metric.pnl_per_marked_day for metric in values)
        for values in family_groups.values()
    ]
    return {
        "families": float(len(family_groups)),
        "mean": mean(family_pnl) if family_pnl else 0.0,
        "mean_per_day": mean(family_per_day) if family_per_day else 0.0,
    }


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _strategy_table(metrics: Sequence[RunMetric]) -> str:
    lines = [
        "| strategy | runs | mean | median | lower quartile | worst run | mean/day | mean DD | max DD | active | hit | turnover | breaches | rejected | beat flat | flat tied | regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in COLD_START_STRATEGY_NAMES:
        row = _actual_summary([metric for metric in metrics if metric.strategy == strategy])
        lines.append(
            f"| `{strategy}` | {int(row['runs'])} | {row['mean']:.0f} | "
            f"{row['median']:.0f} | {row['lower_quartile']:.0f} | "
            f"{row['worst']:.0f} | {row['mean_per_day']:.1f} | "
            f"{row['mean_drawdown']:.0f} | {row['max_drawdown']:.0f} | "
            f"{row['mean_active']:.1f} | {_pct(row['hit_rate'])} | "
            f"{row['mean_turnover']:.1f} | {row['breaches']:.1f} | "
            f"{row['rejections']:.1f} | {_pct(row['beat_flat'])} | "
            f"{_pct(row['flat_tied'])} | {row['mean_regret']:.0f} |"
        )
    return "\n".join(lines)


def _family_balanced_table(metrics: Sequence[RunMetric]) -> str:
    lines = [
        "| strategy | families | family-balanced mean marked P&L | family-balanced mean P&L/day |",
        "|---|---:|---:|---:|",
    ]
    for strategy in COLD_START_STRATEGY_NAMES:
        row = _family_balanced_means(metrics, strategy)
        lines.append(
            f"| `{strategy}` | {int(row['families'])} | {row['mean']:.0f} | "
            f"{row['mean_per_day']:.1f} |"
        )
    return "\n".join(lines)


def _family_table(metrics: Sequence[RunMetric]) -> str:
    lines = [
        "| family | strategy | runs | actual mean P&L | actual mean/day | actual worst run | beat flat | pivotal P&L | non-pivotal P&L |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (family, strategy), values in sorted(_group(metrics, "family", "strategy").items()):
        row = _actual_summary(values)
        lines.append(
            f"| {family} | `{strategy}` | {int(row['runs'])} | {row['mean']:.0f} | "
            f"{row['mean_per_day']:.1f} | {row['worst']:.0f} | "
            f"{_pct(row['beat_flat'])} | {row['pivotal']:.0f} | "
            f"{row['non_pivotal']:.0f} |"
        )
    return "\n".join(lines)


def _mode_table(metrics: Sequence[RunMetric]) -> str:
    lines = [
        "| execution mode | strategy | runs | actual mean P&L | actual mean/day | beat flat | flat tied |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (mode, strategy), values in sorted(_group(metrics, "execution_mode", "strategy").items()):
        row = _actual_summary(values)
        lines.append(
            f"| {mode} | `{strategy}` | {int(row['runs'])} | {row['mean']:.0f} | "
            f"{row['mean_per_day']:.1f} | {_pct(row['beat_flat'])} | "
            f"{_pct(row['flat_tied'])} |"
        )
    return "\n".join(lines)


def _startup_table(metrics: Sequence[RunMetric]) -> str:
    names = (
        "flat",
        "burnin1_markov",
        "burnin3_markov",
        "burnin5_markov",
        "burnin10_markov",
        "immediate_long_prior",
        "flat_first_long_prior",
    )
    lines = [
        "| startup policy | actual mean P&L | actual mean/day | active days | beat flat | flat tied |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy in names:
        row = _actual_summary([metric for metric in metrics if metric.strategy == strategy])
        lines.append(
            f"| `{strategy}` | {row['mean']:.0f} | {row['mean_per_day']:.1f} | "
            f"{row['mean_active']:.1f} | {_pct(row['beat_flat'])} | "
            f"{_pct(row['flat_tied'])} |"
        )
    return "\n".join(lines)


def _paired_mode_table(metrics: Sequence[RunMetric]) -> str:
    groups: dict[tuple[str, int, str], dict[str, RunMetric]] = {}
    for metric in metrics:
        groups.setdefault((metric.family, metric.seed, metric.strategy), {})[
            metric.execution_mode
        ] = metric
    rows: list[tuple[str, int, float, float, float]] = []
    for strategy in COLD_START_STRATEGY_NAMES:
        differences = [
            values["observe_and_ignore_actions"].marked_pnl
            - values["fully_inactive"].marked_pnl
            for (family, _seed, candidate), values in groups.items()
            if candidate == strategy
            and "observe_and_ignore_actions" in values
            and "fully_inactive" in values
        ]
        if differences:
            rows.append(
                (
                    strategy,
                    len(differences),
                    mean(differences),
                    min(differences),
                    max(differences),
                )
            )
    lines = [
        "| strategy | paired cases | observe minus fully inactive mean P&L | min difference | max difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for strategy, count, average, minimum, maximum in rows:
        lines.append(
            f"| `{strategy}` | {count} | {average:.0f} | {minimum:.0f} | {maximum:.0f} |"
        )
    return "\n".join(lines)


def _paired_strategy_table(metrics: Sequence[RunMetric]) -> str:
    lines = [
        "| left strategy | right strategy | paired cases | mean left-minus-right P&L | median left-minus-right P&L | left wins | ties | left losses |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for left, right, count, average, middle, wins, ties, losses in paired_strategy_comparisons(metrics):
        lines.append(
            f"| `{left}` | `{right}` | {count} | {average:.0f} | {middle:.0f} | "
            f"{wins} | {ties} | {losses} |"
        )
    return "\n".join(lines)


def paired_strategy_comparisons(
    metrics: Sequence[RunMetric],
    strategies: Sequence[str] = (
        "burnin1_markov",
        "burnin3_markov",
        "online_markov",
    ),
) -> tuple[tuple[str, str, int, float, float, int, int, int], ...]:
    """Compare predeclared candidates on the same validation scenario."""

    validation = [metric for metric in metrics if metric.suite == "validation"]
    by_scenario: dict[str, dict[str, RunMetric]] = {}
    for metric in validation:
        by_scenario.setdefault(metric.scenario, {})[metric.strategy] = metric
    rows: list[tuple[str, str, int, float, float, int, int, int]] = []
    for left_index, left in enumerate(strategies):
        for right in strategies[left_index + 1 :]:
            differences = [
                values[left].marked_pnl - values[right].marked_pnl
                for values in by_scenario.values()
                if left in values and right in values
            ]
            if not differences:
                continue
            rows.append(
                (
                    left,
                    right,
                    len(differences),
                    mean(differences),
                    median(differences),
                    sum(value > 0 for value in differences),
                    sum(value == 0 for value in differences),
                    sum(value < 0 for value in differences),
                )
            )
    return tuple(rows)


def path_diversity_summaries(
    path_audit: Sequence[PathAuditRecord],
    *,
    suite: str = "validation",
) -> tuple[PathDiversitySummary, ...]:
    """Summarise exact uniqueness and pairwise majority-path Hamming distance.

    These distances describe variation in the observed flat-focal paths; they
    are diagnostics of effective diversity, not claims of independent samples.
    """

    entries = [entry for entry in path_audit if entry.suite == suite]
    groups = _group_path_audit(entries)
    summaries: list[PathDiversitySummary] = []
    for (family, execution_mode), values in sorted(groups.items()):
        paths = [entry.signature[1] for entry in values]
        distances = [
            sum(left != right for left, right in zip(first, second))
            for index, first in enumerate(paths)
            for second in paths[index + 1 :]
        ]
        summaries.append(
            PathDiversitySummary(
                suite=suite,
                family=family,
                execution_mode=execution_mode,
                cases=len(values),
                unique_signatures=len({entry.signature for entry in values}),
                duplicate_cases=len(values) - len({entry.signature for entry in values}),
                minimum_pairwise_hamming_days=min(distances, default=0),
                mean_pairwise_hamming_days=mean(distances) if distances else 0.0,
            )
        )
    return tuple(summaries)


def _path_audit_table(path_audit: Sequence[PathAuditRecord]) -> str:
    validation = [entry for entry in path_audit if entry.suite == "validation"]
    summaries = path_diversity_summaries(validation)
    lines = [
        "| family | mode | cases | unique live path signatures | duplicate cases | min pairwise differing live days | mean pairwise differing live days |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.family} | {summary.execution_mode} | {summary.cases} | "
            f"{summary.unique_signatures} | {summary.duplicate_cases} | "
            f"{summary.minimum_pairwise_hamming_days} | "
            f"{summary.mean_pairwise_hamming_days:.1f} |"
        )
    pair_groups: dict[tuple[str, int], list[PathAuditRecord]] = {}
    for entry in validation:
        pair_groups.setdefault((entry.family, entry.seed), []).append(entry)
    controlled_same = 0
    controlled_different = 0
    intended_stateful_different = 0
    for entries in pair_groups.values():
        by_mode = {entry.execution_mode: entry for entry in entries}
        if len(by_mode) != 2:
            continue
        if any(entry.path_controlled for entry in entries):
            if by_mode["observe_and_ignore_actions"].signature == by_mode["fully_inactive"].signature:
                controlled_same += 1
            else:
                controlled_different += 1
        else:
            if by_mode["observe_and_ignore_actions"].signature != by_mode["fully_inactive"].signature:
                intended_stateful_different += 1
    lines.extend(
        [
            "",
            f"Controlled paired paths equal: {controlled_same}; controlled pairs differing unexpectedly: {controlled_different}.",
            f"State/RNG-evolution pairs differing as intended: {intended_stateful_different}.",
        ]
    )
    return "\n".join(lines)


def _group_path_audit(
    path_audit: Sequence[PathAuditRecord],
) -> dict[tuple[str, str], list[PathAuditRecord]]:
    groups: dict[tuple[str, str], list[PathAuditRecord]] = {}
    for entry in path_audit:
        groups.setdefault((entry.family, entry.execution_mode), []).append(entry)
    return groups


def live_path_difference(
    baseline: SimulationResult,
    comparison: SimulationResult,
) -> LivePathDifference:
    """Compare two same-scenario runs over the live segment only.

    This helper is intentionally independent of P&L. It makes the endogenous
    counterfactual explicit: the focal exposure can change effective focal
    actions, which can change majority, prices, and later opponent actions.
    """

    baseline_start = baseline.config.voting_start_day
    comparison_start = comparison.config.voting_start_day
    if baseline_start != comparison_start:
        raise ValueError("path comparisons require the same voting-start day")
    if baseline.focal_agent_name != comparison.focal_agent_name:
        raise ValueError("path comparisons require the same focal agent name")
    start = baseline_start
    assert start is not None

    max_price_length = max(len(baseline.price_path), len(comparison.price_path))
    differing_price_days = sum(
        index >= len(baseline.price_path)
        or index >= len(comparison.price_path)
        or baseline.price_path[index] != comparison.price_path[index]
        for index in range(start, max_price_length)
    )

    max_day_length = max(len(baseline.days), len(comparison.days))
    differing_majority_days = 0
    differing_opponent_action_cells = 0
    focal_name = baseline.focal_agent_name
    for index in range(start, max_day_length):
        baseline_day = baseline.days[index] if index < len(baseline.days) else None
        comparison_day = comparison.days[index] if index < len(comparison.days) else None
        if baseline_day is None or comparison_day is None:
            differing_majority_days += 1
            differing_opponent_action_cells += len(
                set((baseline_day or comparison_day).actions) - {focal_name}
            )
            continue
        if baseline_day.majority != comparison_day.majority:
            differing_majority_days += 1
        baseline_opponents = set(baseline_day.actions) - {focal_name}
        comparison_opponents = set(comparison_day.actions) - {focal_name}
        for opponent_name in baseline_opponents | comparison_opponents:
            if baseline_day.actions.get(opponent_name) != comparison_day.actions.get(
                opponent_name
            ):
                differing_opponent_action_cells += 1

    return LivePathDifference(
        differing_price_days=differing_price_days,
        differing_majority_days=differing_majority_days,
        differing_opponent_action_cells=differing_opponent_action_cells,
    )


def _path_divergence_summary(
    differences: Sequence[LivePathDifference],
    *,
    baseline_exposure: float,
    exposure: float,
    group: str,
) -> ExposurePathDivergenceSummary:
    count = len(differences)
    return ExposurePathDivergenceSummary(
        baseline_exposure=baseline_exposure,
        exposure=exposure,
        group=group,
        cases=count,
        different_price_path_fraction=(
            mean(value.differing_price_days > 0 for value in differences)
            if differences
            else 0.0
        ),
        different_majority_path_fraction=(
            mean(value.differing_majority_days > 0 for value in differences)
            if differences
            else 0.0
        ),
        mean_differing_price_days=(
            mean(value.differing_price_days for value in differences)
            if differences
            else 0.0
        ),
        maximum_differing_price_days=max(
            (value.differing_price_days for value in differences), default=0
        ),
        mean_differing_majority_days=(
            mean(value.differing_majority_days for value in differences)
            if differences
            else 0.0
        ),
        maximum_differing_majority_days=max(
            (value.differing_majority_days for value in differences), default=0
        ),
        mean_differing_opponent_action_cells=(
            mean(value.differing_opponent_action_cells for value in differences)
            if differences
            else 0.0
        ),
        maximum_differing_opponent_action_cells=max(
            (value.differing_opponent_action_cells for value in differences),
            default=0,
        ),
    )


def portfolio_path_divergence_audit(
    scenarios: Iterable[Pass3Scenario],
    *,
    exposures: Sequence[int] = PORTFOLIO_EXPOSURES,
    strategy: str = PATH_DIVERGENCE_STRATEGY,
    progress: bool = False,
) -> tuple[ExposurePathDivergenceSummary, ...]:
    """Compare only the focal burn-in Markov path across exposure levels.

    The scenarios are consumed validation definitions, but this is a mechanics
    audit rather than another validation ranking. Exposure zero is run once as
    the common baseline; each nonzero level is compared with that same-scenario
    baseline. No final constructor is imported or called by this function.
    """

    if strategy != PATH_DIVERGENCE_STRATEGY:
        raise ValueError(
            "the path-divergence audit is intentionally limited to "
            f"{PATH_DIVERGENCE_STRATEGY!r}"
        )
    exposure_values = tuple(float(value) for value in exposures)
    if not exposure_values or exposure_values[0] != 0.0:
        raise ValueError("path-divergence exposures must start with zero")
    if len(set(exposure_values)) != len(exposure_values):
        raise ValueError("path-divergence exposures must be unique")

    scenario_list = tuple(scenarios)
    results_by_exposure: dict[float, tuple[SimulationResult, ...]] = {}
    for exposure in exposure_values:
        if progress:
            print(
                f"[portfolio-path-audit] exposure={exposure:.0f} "
                f"scenarios={len(scenario_list)}"
            )
        results_by_exposure[exposure] = tuple(
            scenario.run(
                strategy,
                other_portfolio_exposure=exposure,
            )
            for scenario in scenario_list
        )

    baseline_results = results_by_exposure[0.0]
    summaries: list[ExposurePathDivergenceSummary] = []
    family_names = sorted({scenario.family for scenario in scenario_list})
    for exposure in exposure_values:
        differences = tuple(
            live_path_difference(baseline, comparison)
            for baseline, comparison in zip(
                baseline_results,
                results_by_exposure[exposure],
                strict=True,
            )
        )
        summaries.append(
            _path_divergence_summary(
                differences,
                baseline_exposure=0.0,
                exposure=exposure,
                group="overall",
            )
        )
        for group, predicate in (
            (
                "controlled_day_indexed",
                lambda scenario: scenario.path_controlled,
            ),
            (
                "state_or_rng_sensitive",
                lambda scenario: not scenario.path_controlled,
            ),
        ):
            group_differences = tuple(
                difference
                for scenario, difference in zip(
                    scenario_list,
                    differences,
                    strict=True,
                )
                if predicate(scenario)
            )
            summaries.append(
                _path_divergence_summary(
                    group_differences,
                    baseline_exposure=0.0,
                    exposure=exposure,
                    group=group,
                )
            )
        for family in family_names:
            family_differences = tuple(
                difference
                for scenario, difference in zip(
                    scenario_list,
                    differences,
                    strict=True,
                )
                if scenario.family == family
            )
            summaries.append(
                _path_divergence_summary(
                    family_differences,
                    baseline_exposure=0.0,
                    exposure=exposure,
                    group=family,
                )
            )
    return tuple(summaries)


def render_path_divergence_report(
    summaries: Sequence[ExposurePathDivergenceSummary],
    *,
    strategy: str = PATH_DIVERGENCE_STRATEGY,
) -> str:
    """Render the mechanics-only exposure path-divergence report."""

    overall_groups = {
        "overall",
        "controlled_day_indexed",
        "state_or_rng_sensitive",
    }
    overall = [summary for summary in summaries if summary.group in overall_groups]
    families = [summary for summary in summaries if summary.group not in overall_groups]

    def table(values: Sequence[ExposurePathDivergenceSummary]) -> str:
        lines = [
            "| baseline → exposure | group | cases | different price path | different majority path | mean price days | max price days | mean majority days | max majority days | mean opponent action cells | max opponent action cells |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for summary in values:
            lines.append(
                f"| ${summary.baseline_exposure:,.0f} → ${summary.exposure:,.0f} | "
                f"{summary.group} | {summary.cases} | "
                f"{_pct(summary.different_price_path_fraction)} | "
                f"{_pct(summary.different_majority_path_fraction)} | "
                f"{summary.mean_differing_price_days:.1f} | "
                f"{summary.maximum_differing_price_days} | "
                f"{summary.mean_differing_majority_days:.1f} | "
                f"{summary.maximum_differing_majority_days} | "
                f"{summary.mean_differing_opponent_action_cells:.1f} | "
                f"{summary.maximum_differing_opponent_action_cells} |"
            )
        return "\n".join(lines)

    return f"""# Pass 3.1 portfolio path-divergence audit

This is a mechanics/path-divergence diagnostic, not a new strategy validation
or ranking. It reruns the consumed validation scenarios only for the frozen
`{strategy}` focal candidate at the predeclared other-exposure levels and
compares every nonzero run with the zero-exposure run for the same scenario.
The zero-exposure row is a deterministic self-comparison check.

The exposure is applied to the focal portfolio only, but the simulation is an
endogenous game counterfactual. A focal budget-forced flattening can change the
focal vote, the majority and price path, and then reactive opponents' future
effective actions or budget feasibility. The opponent metric below compares
effective engine actions, not merely their raw requests. These results must not
be read as a fixed-path capacity estimate or used to tune a strategy.

## Overall and execution-mode groups

{table(overall)}

## Scenario families

{table(families)}

`controlled_day_indexed` means the opponent requests are intended to be
day-indexed and path-controlled; `state_or_rng_sensitive` includes reactive,
startup-state, and other populations whose calls or public path can affect
future behavior. Path differences are diagnostics of this counterfactual, not
claims of statistical independence.
"""


def run_portfolio_path_audit(
    scenarios: Iterable[Pass3Scenario] | None = None,
    *,
    exposures: Sequence[int] = PORTFOLIO_EXPOSURES,
    write_report: bool = True,
) -> tuple[ExposurePathDivergenceSummary, ...]:
    """Run the bounded burn-in-only exposure path audit."""

    scenario_list = tuple(validation_scenarios() if scenarios is None else scenarios)
    summaries = portfolio_path_divergence_audit(
        scenario_list,
        exposures=exposures,
        strategy=PATH_DIVERGENCE_STRATEGY,
        progress=True,
    )
    if write_report:
        PATH_DIVERGENCE_REPORT_PATH.write_text(
            render_path_divergence_report(summaries),
            encoding="utf-8",
        )
        print(f"wrote {PATH_DIVERGENCE_REPORT_PATH}")
    return summaries


def _pct_or_zero(value: float) -> str:
    return _pct(value) if value else "0.0%"


def render_report(
    metrics: Sequence[RunMetric],
    *,
    suite_label: str,
    path_audit: Sequence[PathAuditRecord] = (),
) -> str:
    development = tuple(metric for metric in metrics if metric.suite == "development")
    validation = tuple(metric for metric in metrics if metric.suite == "validation")
    validation_rows = {
        strategy: _actual_summary([metric for metric in validation if metric.strategy == strategy])
        for strategy in COLD_START_STRATEGY_NAMES
    }
    ranked = sorted(
        COLD_START_STRATEGY_NAMES,
        key=lambda strategy: validation_rows[strategy]["mean_per_day"],
        reverse=True,
    )
    leader = ranked[0]
    flat_row = validation_rows["flat"]
    drift_row = validation_rows["online_drift"]
    ensemble_row = validation_rows["online_ensemble"]
    markov_row = validation_rows["online_markov"]
    leader_row = validation_rows[leader]
    family_balanced = sorted(
        COLD_START_STRATEGY_NAMES,
        key=lambda strategy: _family_balanced_means(validation, strategy)["mean_per_day"],
        reverse=True,
    )
    scenario_counts = {
        suite: len({metric.scenario for metric in metrics if metric.suite == suite})
        for suite in ("development", "validation")
    }
    cell_counts = {
        suite: sum(metric.suite == suite for metric in metrics)
        for suite in ("development", "validation")
    }
    return f"""# Pass 3 cold-start Liferaft report

> This report covers **{suite_label}** only. Development cases are diagnostics;
> all rankings and conclusions below use validation runs only. The locked final
> suite was not executed. The organiser clarification makes Year 1 a constant
> $100,000 observation, so all Year-1 calibration/replay conclusions in
> `PASS2_REPORT.md` are superseded.

## Method and suite separation

The simulator uses `market_mode=inactive_until_marked`, with voting starting
on day 365. Price is exactly $100,000 through the day-365 observation. The day
365 action determines the first genuine movement into day 366; marked P&L
begins on day 366 and flat pre-voting history never produces a majority label.
The current best execution assumption is `observe_and_ignore_actions`, with
`fully_inactive` retained as a paired sensitivity case.

| suite | scenarios | strategies/scenario cells | role |
|---|---:|---:|---|
| development | {scenario_counts['development']} | {cell_counts['development']} | correctness/design diagnostics only |
| validation | {scenario_counts['validation']} | {cell_counts['validation']} | sole source of rankings/conclusions |

Turnover is a stability diagnostic, not a transaction cost. The displayed
ranking applies no turnover penalty and no strategy parameters were tuned
after viewing validation results.

## Development diagnostics (not ranked)

{_strategy_table(development)}

## Actual validation distribution

The following are computed directly across actual validation runs. `worst run`,
lower quartile, and `max DD` are not averages of family-level statistics.

{_strategy_table(validation)}

No-turnover validation ranking by actual mean P&L per marked day:
`{' > '.join(ranked)}`.

## Family-balanced comparison

These rows average only each family's mean P&L and mean P&L/day. Quantiles,
worst runs, drawdowns, and fractions remain in the actual validation table.

{_family_balanced_table(validation)}

Family-balanced mean P&L/day ranking:
`{' > '.join(family_balanced)}`.

## Results by validation family

{_family_table(validation)}

## Execution-mode sensitivity

{_mode_table(validation)}

### Controlled paired differences

{_paired_mode_table(validation)}

Positive/negative differences for controlled day-indexed populations measure
startup execution only. Reactive and startup-state pairs are labelled as
state/RNG evolution because their Year-1 calls intentionally change object
state or RNG consumption.

### Paired candidate comparison

These comparisons use the same validation scenario for both candidates. They
are not a separate holdout and do not apply a turnover cost.

{_paired_strategy_table(validation)}

## Unique-path audit

The table below is a path-diversity diagnostic over flat-focal live majority
paths. Exact uniqueness and pairwise Hamming distances show effective path
variation within each family/mode; they are not claims of statistical
independence. Naturally persistent families may have shallow distances.

{_path_audit_table(path_audit)}

## Interpretation

- The consumed-validation leader is `{leader}` with actual mean P&L
  {leader_row['mean']:.0f}, mean P&L/day {leader_row['mean_per_day']:.1f},
  lower quartile {leader_row['lower_quartile']:.0f}, worst run
  {leader_row['worst']:.0f}, and maximum drawdown {leader_row['max_drawdown']:.0f}.
  This is a validation leader, not a production recommendation.
- Flat is the explicit no-trade fallback. It was tied for best in
  {_pct(flat_row['flat_tied'])} of validation scenarios; its own P&L is zero,
  and the candidate outperformance fractions show where trading was actually
  preferable.
- Burn-in, Markov, ensemble, and asymmetric-prior results must be read with
  lower-tail loss, pivotal P&L, budget rejection, and family dependence. The
  immediate-long prior is not assumed optimal merely because the upward move
  is $8,000 versus the $5,000 downward move.
- Online Markov's actual validation mean/day is {markov_row['mean_per_day']:.1f};
  the ensemble is {ensemble_row['mean_per_day']:.1f}. Drift fallback is
  {drift_row['mean_per_day']:.1f} with mean drawdown {drift_row['mean_drawdown']:.0f}.
  These are validation comparisons, not held-out evidence.
- Periodic replay and Year-1 selection are excluded. The old Pass 2 “Markov is
  best” conclusion remains superseded and no replay value is claimed here.
- Stress cases such as floor clipping, runaway budget rejection, ties/zeros,
  and no-trade-friendly populations are kept in development diagnostics rather
  than allowed to dominate stochastic validation ranking.

## Locked final protocol

The final definition uses a distinct seed range beginning at 90,000 and unseen
families including symmetric biases, reactive mixtures, periodic behavior,
regime/drift, startup sensitivity, and pivotal margins. Its concise manifest is
stored separately. The normal runner does not instantiate it.

Only this explicit command may execute the final suite:

```text
python -m research.liferaft.pass3_experiments --final
```

When executed, it writes `PASS3_FINAL_REPORT.md` and states that final results
were executed; it does not overwrite this validation report or claim that the
final suite remains unconsumed.
"""


def render_final_report(metrics: Sequence[RunMetric]) -> str:
    """Render truthful final output if a user explicitly executes --final."""

    return """# Pass 3 final Liferaft report

> The locked final suite was explicitly executed. These results are not
> validation evidence and must not be fed back into strategy selection.

The final suite contains the separate locked seed range and writes this file
instead of overwriting `PASS3_REPORT.md`.

## Final run summary

""" + _strategy_table(metrics) + "\n"


def _portfolio_table(metrics: Sequence[RunMetric]) -> str:
    lines = [
        "| other gross exposure | strategy | runs | mean P&L | median P&L | lower quartile | worst run | mean DD | max DD | beat flat | mean rejected | mean breaches | any rejection | mean active days |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    grouped = _group(metrics, "other_portfolio_exposure", "strategy")
    for (exposure, strategy), values in sorted(grouped.items()):
        row = _actual_summary(values)
        any_rejection = mean(metric.rejected_actions > 0 for metric in values) if values else 0.0
        lines.append(
            f"| {exposure:.0f} | `{strategy}` | {int(row['runs'])} | "
            f"{row['mean']:.0f} | {row['median']:.0f} | "
            f"{row['lower_quartile']:.0f} | {row['worst']:.0f} | "
            f"{row['mean_drawdown']:.0f} | {row['max_drawdown']:.0f} | "
            f"{_pct(row['beat_flat'])} | {row['rejections']:.2f} | "
            f"{row['breaches']:.2f} | {_pct(any_rejection)} | "
            f"{row['mean_active']:.1f} |"
        )
    return "\n".join(lines)


def _portfolio_family_table(metrics: Sequence[RunMetric]) -> str:
    lines = [
        "| other gross exposure | family | strategy | mean P&L | mean rejected | mean breaches | any rejection |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    grouped = _group(metrics, "other_portfolio_exposure", "family", "strategy")
    for (exposure, family, strategy), values in sorted(grouped.items()):
        row = _actual_summary(values)
        any_rejection = mean(metric.rejected_actions > 0 for metric in values) if values else 0.0
        lines.append(
            f"| {exposure:.0f} | {family} | `{strategy}` | {row['mean']:.0f} | "
            f"{row['rejections']:.2f} | {row['breaches']:.2f} | {_pct(any_rejection)} |"
        )
    return "\n".join(lines)


def render_portfolio_sensitivity(metrics: Sequence[RunMetric]) -> str:
    """Render the separate constant-other-exposure sensitivity report."""

    scenarios = len({metric.scenario for metric in metrics})
    cells = len(metrics)
    return f"""# Pass 3.1 Liferaft portfolio-exposure sensitivity

This report is a separate sensitivity analysis and is not mixed into the
primary zero-other-exposure ranking in `PASS3_REPORT.md`. It uses the same
consumed validation scenarios and causal simulator mechanics. The configured
constant gross exposure is applied to the focal agent only, but the resulting
comparison is an **endogenous game counterfactual**, not a fixed-path measure
of focal budget capacity: a budget-forced focal flattening can change the vote,
price path, and later reactive-opponent actions or budget feasibility.

Constant other exposure is a sensitivity model, not a forecast of the final
portfolio allocation. Turnover is not treated as a transaction cost, and no
candidate parameters were changed for this analysis.

| item | value |
|---|---:|
| validation scenarios per exposure | {scenarios} |
| strategy/exposure cells | {cells} |
| exposure levels | {', '.join(f'${value:,}' for value in PORTFOLIO_EXPOSURES)} |
| candidates | {', '.join(PORTFOLIO_STRATEGIES)} |

## Actual validation distributions

All statistics below are across actual runs at each exposure. Rejection and
breach counts refer to the focal Liferaft request during the marked period;
`any rejection` is the fraction of scenarios with at least one rejected
Liferaft action. The flat strategy is the zero-P&L comparator at each exposure.

{_portfolio_table(metrics)}

## Results by validation family

These family means make exposure-dependent budget effects visible without
presenting them as independent samples or replacing the actual-run tails.

{_portfolio_family_table(metrics)}

## Interpretation

At higher other-instrument exposure, non-flat Liferaft requests can become
infeasible earlier as the price rises and are flattened by the simulator. A
budget breach is therefore not evidence that the candidate made an invalid
action; it records a valid Liferaft request that could not coexist with the
specified constant portfolio exposure. Because that flattening can feed back
through majority, price, and reactive-opponent state, declines in these stored
P&L numbers cannot be attributed solely to focal rejection. They remain valid
under the endogenous simulation assumption and should be read only as a
risk/sizing sensitivity, not as a fixed-market capacity curve.
"""


def run_portfolio_sensitivity(
    scenarios: Iterable[Pass3Scenario] | None = None,
    *,
    exposures: Sequence[int] = PORTFOLIO_EXPOSURES,
    strategies: Sequence[str] = PORTFOLIO_STRATEGIES,
    write_report: bool = True,
) -> tuple[RunMetric, ...]:
    """Run the predeclared endogenous constant-exposure sensitivity suite."""

    scenario_list = tuple(validation_scenarios() if scenarios is None else scenarios)
    metrics: list[RunMetric] = []
    for exposure in exposures:
        metrics.extend(
            run_suite(
                scenario_list,
                suite=f"portfolio-{exposure:g}",
                strategies=strategies,
                other_portfolio_exposure=float(exposure),
            )
        )
    result = tuple(metrics)
    if write_report:
        PORTFOLIO_REPORT_PATH.write_text(
            render_portfolio_sensitivity(result),
            encoding="utf-8",
        )
        print(f"wrote {PORTFOLIO_REPORT_PATH}")
    return result


def run_experiments(*, final: bool = False) -> tuple[RunMetric, ...]:
    if final:
        # Keep the final runner behind the explicit flag. Normal development,
        # validation, and portfolio commands cannot instantiate locked cases.
        from .pass4_final import run_locked_final

        return run_locked_final()

    development_audit: list[PathAuditRecord] = []
    validation_audit: list[PathAuditRecord] = []
    development_metrics = run_suite(
        development_scenarios(),
        suite="development",
        path_audit=development_audit,
    )
    validation_metrics = run_suite(
        validation_scenarios(),
        suite="validation",
        path_audit=validation_audit,
    )
    metrics = (*development_metrics, *validation_metrics)
    REPORT_PATH.write_text(
        render_report(
            metrics,
            suite_label="development + validation",
            path_audit=(*development_audit, *validation_audit),
        ),
        encoding="utf-8",
    )
    print(
        f"development scenarios={len({metric.scenario for metric in development_metrics})} "
        f"cells={len(development_metrics)}"
    )
    print(
        f"validation scenarios={len({metric.scenario for metric in validation_metrics})} "
        f"cells={len(validation_metrics)}"
    )
    print(f"wrote {REPORT_PATH}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--final",
        action="store_true",
        help="run the separately locked final suite and write PASS3_FINAL_REPORT.md",
    )
    modes.add_argument(
        "--portfolio-sensitivity",
        action="store_true",
        help="run the separate constant-other-exposure validation sensitivity",
    )
    modes.add_argument(
        "--portfolio-path-audit",
        action="store_true",
        help="audit endogenous live-path divergence for burnin1_markov only",
    )
    args = parser.parse_args()
    if args.portfolio_sensitivity:
        run_portfolio_sensitivity()
    elif args.portfolio_path_audit:
        run_portfolio_path_audit()
    else:
        run_experiments(final=args.final)


if __name__ == "__main__":
    main()
