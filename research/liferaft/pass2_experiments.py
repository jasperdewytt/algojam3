"""Reproducible Pass 2 Liferaft experiments and compact reporting.

Run from the repository root with::

    python -m research.liferaft.pass2_experiments

The suite evaluates fixed candidate definitions on development and validation
scenarios, then runs the separately seeded held-out suite once.  No held-out
result is fed back into strategy construction or threshold selection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Callable, Iterable, Mapping, Sequence

from .pass2_scenarios import (
    Pass2Scenario,
    all_suites,
    strategy_factory_for_name,
)
from .simulator import AgentDayRecord, SimulationResult


STRATEGY_NAMES: tuple[str, ...] = (
    "flat",
    "always_long",
    "always_short",
    "last_majority_counter",
    "rolling_frequency",
    "markov",
    "periodic_replay",
    "year1_selected",
    "ensemble",
    "drift_aware",
)


@dataclass(frozen=True)
class RunMetrics:
    suite: str
    family: str
    scenario: str
    strategy: str
    marked_pnl: int
    max_drawdown: int
    hit_rate: float
    active_days: int
    turnover: int
    budget_breaches: int
    rejected_actions: int
    pivotal_marked_pnl: int
    non_pivotal_marked_pnl: int
    pivotal_days: int
    non_pivotal_days: int
    flat_marked_pnl: int = 0
    outperformed_flat: bool = False
    regret: int = 0


def _max_drawdown(records: Sequence[AgentDayRecord], boundary: int) -> int:
    curve = [
        record.marked_cumulative_pnl
        for record in records
        if record.day >= boundary
    ]
    peak = 0
    maximum = 0
    for value in curve:
        peak = max(peak, value)
        maximum = max(maximum, peak - value)
    return maximum


def _focal_metrics(
    result: SimulationResult,
    *,
    suite: str,
    strategy: str,
) -> RunMetrics:
    focal = result.focal_agent_name
    if focal is None:
        raise ValueError("Pass 2 result must identify a focal agent")
    records = result.agent_history(focal)
    boundary = result.config.marked_boundary_day
    realised = [
        record
        for record in records
        if record.day > boundary and record.pnl_source_day is not None
    ]
    active = [record for record in realised if record.pnl_position not in (None, 0)]
    hits = sum(record.daily_pnl > 0 for record in active)
    turnover = 0
    marked_decisions = [record for record in records if record.day >= boundary]
    for previous, current in zip(marked_decisions, marked_decisions[1:]):
        turnover += previous.action != current.action

    pivotal_pnl = 0
    non_pivotal_pnl = 0
    pivotal_days = 0
    non_pivotal_days = 0
    for record in realised:
        source_day = record.pnl_source_day
        if source_day is None:
            continue
        source = result.days[source_day]
        if source.focal_pivotal:
            pivotal_pnl += record.daily_pnl
            pivotal_days += 1
        else:
            non_pivotal_pnl += record.daily_pnl
            non_pivotal_days += 1

    focal_breaches = sum(
        breach.agent_name == focal for breach in result.budget_breaches
    )
    focal_rejections = sum(
        rejection.agent_name == focal for rejection in result.rejected_actions
    )
    return RunMetrics(
        suite=suite,
        family=str(result.scenario_configuration.get("family", "unknown")),
        scenario=result.scenario_name or "unnamed",
        strategy=strategy,
        marked_pnl=result.marked_pnl[focal],
        max_drawdown=_max_drawdown(records, boundary),
        hit_rate=hits / len(active) if active else 0.0,
        active_days=len(active),
        turnover=turnover,
        budget_breaches=focal_breaches,
        rejected_actions=focal_rejections,
        pivotal_marked_pnl=pivotal_pnl,
        non_pivotal_marked_pnl=non_pivotal_pnl,
        pivotal_days=pivotal_days,
        non_pivotal_days=non_pivotal_days,
    )


def _with_comparison_fields(
    metrics: Sequence[RunMetrics],
) -> tuple[RunMetrics, ...]:
    by_scenario: dict[str, list[RunMetrics]] = {}
    for metric in metrics:
        by_scenario.setdefault(f"{metric.suite}:{metric.scenario}", []).append(metric)
    result: list[RunMetrics] = []
    for metric in metrics:
        same_scenario = by_scenario[f"{metric.suite}:{metric.scenario}"]
        flat = next(item for item in same_scenario if item.strategy == "flat")
        best = max(item.marked_pnl for item in same_scenario)
        result.append(
            replace(
                metric,
                flat_marked_pnl=flat.marked_pnl,
                outperformed_flat=metric.marked_pnl > flat.marked_pnl,
                regret=best - metric.marked_pnl,
            )
        )
    return tuple(result)


def run_suite(
    suite_name: str,
    scenarios: Sequence[Pass2Scenario],
    *,
    strategy_names: Sequence[str] = STRATEGY_NAMES,
) -> tuple[RunMetrics, ...]:
    """Run one suite with fresh opponents and fresh focal state per cell."""

    metrics: list[RunMetrics] = []
    for scenario in scenarios:
        for strategy_name in strategy_names:
            result = scenario.run(strategy_factory_for_name(strategy_name))
            metrics.append(
                _focal_metrics(
                    result,
                    suite=suite_name,
                    strategy=strategy_name,
                )
            )
    return _with_comparison_fields(metrics)


def run_all_suites(
    *,
    strategy_names: Sequence[str] = STRATEGY_NAMES,
) -> tuple[RunMetrics, ...]:
    """Run development, validation, then untouched held-out scenarios once."""

    metrics: list[RunMetrics] = []
    for suite_name, scenarios in all_suites().items():
        metrics.extend(
            run_suite(
                suite_name,
                scenarios,
                strategy_names=strategy_names,
            )
        )
    return tuple(metrics)


def _lower_quartile(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[int((len(ordered) - 1) * 0.25)])


def aggregate_metrics(
    metrics: Sequence[RunMetrics],
    *,
    suite: str | None = None,
    family: str | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate all requested score fields by strategy."""

    selected = [
        metric
        for metric in metrics
        if (suite is None or metric.suite == suite)
        and (family is None or metric.family == family)
    ]
    grouped: dict[str, list[RunMetrics]] = {}
    for metric in selected:
        grouped.setdefault(metric.strategy, []).append(metric)
    summary: dict[str, dict[str, float]] = {}
    for strategy, rows in grouped.items():
        pnls = [row.marked_pnl for row in rows]
        active = sum(row.active_days for row in rows)
        pivotal_days = sum(row.pivotal_days for row in rows)
        non_pivotal_days = sum(row.non_pivotal_days for row in rows)
        summary[strategy] = {
            "runs": float(len(rows)),
            "mean_marked_pnl": sum(pnls) / len(pnls),
            "median_marked_pnl": float(median(pnls)),
            "lower_quartile_marked_pnl": _lower_quartile(pnls),
            "worst_marked_pnl": float(min(pnls)),
            "mean_max_drawdown": sum(row.max_drawdown for row in rows) / len(rows),
            "hit_rate": (
                sum(row.hit_rate * row.active_days for row in rows) / active
                if active
                else 0.0
            ),
            "active_days": float(active) / len(rows),
            "turnover": sum(row.turnover for row in rows) / len(rows),
            "budget_breaches": float(sum(row.budget_breaches for row in rows)),
            "rejected_actions": float(sum(row.rejected_actions for row in rows)),
            "outperformed_flat_fraction": sum(row.outperformed_flat for row in rows)
            / len(rows),
            "mean_regret": sum(row.regret for row in rows) / len(rows),
            "pivotal_marked_pnl": float(sum(row.pivotal_marked_pnl for row in rows)),
            "non_pivotal_marked_pnl": float(
                sum(row.non_pivotal_marked_pnl for row in rows)
            ),
            "pivotal_days": float(pivotal_days),
            "non_pivotal_days": float(non_pivotal_days),
        }
    return summary


def robust_strategy(metrics: Sequence[RunMetrics], suite: str = "held_out") -> str:
    """Choose the most robust held-out candidate by lower quartile then mean."""

    summary = aggregate_metrics(metrics, suite=suite)
    if not summary:
        raise ValueError(f"no metrics for suite {suite!r}")
    return max(
        summary,
        key=lambda name: (
            summary[name]["lower_quartile_marked_pnl"],
            summary[name]["mean_marked_pnl"],
            -summary[name]["mean_max_drawdown"],
        ),
    )


def _fmt_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.1f}"


def format_summary_table(summary: Mapping[str, Mapping[str, float]]) -> str:
    headers = (
        "strategy",
        "mean",
        "median",
        "LQ",
        "worst",
        "DD",
        "hit",
        "active",
        "turnover",
        "breaches",
        "rejects",
        "beat-flat",
        "regret",
        "pivotal/non-pivotal",
    )
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for strategy in sorted(summary):
        row = summary[strategy]
        lines.append(
            "| "
            + " | ".join(
                (
                    strategy,
                    _fmt_number(row["mean_marked_pnl"]),
                    _fmt_number(row["median_marked_pnl"]),
                    _fmt_number(row["lower_quartile_marked_pnl"]),
                    _fmt_number(row["worst_marked_pnl"]),
                    _fmt_number(row["mean_max_drawdown"]),
                    f"{row['hit_rate']:.1%}",
                    _fmt_number(row["active_days"]),
                    _fmt_number(row["turnover"]),
                    _fmt_number(row["budget_breaches"]),
                    _fmt_number(row["rejected_actions"]),
                    f"{row['outperformed_flat_fraction']:.1%}",
                    _fmt_number(row["mean_regret"]),
                    f"{_fmt_number(row['pivotal_marked_pnl'])} / "
                    f"{_fmt_number(row['non_pivotal_marked_pnl'])}",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _scenario_flatness_note(metrics: Sequence[RunMetrics]) -> str:
    groups: dict[tuple[str, str], list[RunMetrics]] = {}
    for metric in metrics:
        groups.setdefault((metric.suite, metric.scenario), []).append(metric)
    flat_best = sum(
        max(row.marked_pnl for row in rows if row.strategy == "flat")
        >= max(row.marked_pnl for row in rows)
        for rows in groups.values()
    )
    return f"{flat_best}/{len(groups)} scenario runs had flat at least tied for best."


def render_report(metrics: Sequence[RunMetrics]) -> str:
    lines = [
        "# SUPERSEDED — Liferaft Pass 2 experiment report",
        "",
        "> Roger's organiser clarification fixed Liferaft at $100,000 throughout",
        "> Year 1. The changing Year-1 price assumption and all Year-1 selection",
        "> and replay conclusions in this historical report are superseded. The",
        "> data remains for auditability; current cold-start evidence is in",
        "> `PASS3_REPORT.md`.",
        "",
        "This report is generated by `python -m research.liferaft.pass2_experiments`.",
        "The experiment uses public price prefixes only.  Candidate definitions,",
        "windows, Markov order, replay evidence, payoff margin, and drift thresholds",
        "are fixed before the final held-out suite is run.",
        "",
        "## Mechanics and decision policy",
        "",
        "- Year 1 is the indexed prefix before the marked boundary.  At the boundary",
        "  the price reset is excluded from P&L and public majority labels.",
        "- A decision on day `d` is scored on the genuine movement into day `d+1`.",
        "- The asymmetric zero-EV threshold for a long is `P(short majority) > 5/13`;",
        "  a fixed $1,000 expected-P&L margin makes uncertain forecasts flat.",
        "- Walk-forward calibration starts after a 30-day warm-up and uses three",
        "  contiguous validation blocks.  A candidate must beat flat by $5,000 per",
        "  mean block and be positive in at least half the blocks; otherwise flat wins.",
        "  Selection utility also applies fixed $1,000/switch and $500/complexity",
        "  penalties.",
        "- Ensemble weights are a fixed non-negative transform of Year-1 block",
        "  improvements and are frozen at the boundary.  Drift fallback requires 12",
        "  observed non-zero outcomes and three consecutive low-quality checks.",
        "",
        "## Candidate definitions",
        "",
        "`flat`, `always_long`, `always_short`, `last_majority_counter`,",
        "`rolling_frequency` (windows 5/10/20), `markov` (order 2 with Laplace",
        "smoothing), `periodic_replay` (periods 2/3/4/5 with a constant-frequency",
        "null test), `year1_selected`, `ensemble`, and `drift_aware`.",
        "",
        "## Results by suite",
    ]
    for suite in ("development", "validation", "held_out"):
        lines.extend(["", f"### {suite.replace('_', ' ').title()}", ""])
        lines.append(format_summary_table(aggregate_metrics(metrics, suite=suite)))
        families = sorted(
            {
                metric.family
                for metric in metrics
                if metric.suite == suite
            }
        )
        for family in families:
            lines.extend(["", f"#### {family}", ""])
            lines.append(
                format_summary_table(
                    aggregate_metrics(metrics, suite=suite, family=family)
                )
            )

    heldout = robust_strategy(metrics, "held_out")
    heldout_summary = aggregate_metrics(metrics, suite="held_out")
    replay_mean = heldout_summary.get("periodic_replay", {}).get("mean_marked_pnl", 0.0)
    markov_mean = heldout_summary.get("markov", {}).get("mean_marked_pnl", 0.0)
    drift_mean = heldout_summary.get("drift_aware", {}).get("mean_marked_pnl", 0.0)
    ensemble_mean = heldout_summary.get("ensemble", {}).get("mean_marked_pnl", 0.0)
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- Most robust held-out candidate by lower quartile/mean tie-break: **{heldout}**.",
            f"- Replay mean marked P&L minus Markov: {_fmt_number(replay_mean - markov_mean)}.",
            f"- Drift-aware mean marked P&L minus ensemble: {_fmt_number(drift_mean - ensemble_mean)}.",
            f"- {_scenario_flatness_note(metrics)}",
            "- Pivotal and non-pivotal P&L are reported separately; a focal vote can",
            "  change the market outcome in the near-balanced scenarios.",
            "- Budget breaches are diagnostics, not a price cap: the engine allows",
            "  price to exceed $600,000 and flattens an infeasible non-zero request.",
            "",
            "The held-out suite is not used to change any strategy parameter.  The",
            "replay and drift deltas are descriptive comparisons, not tuning targets;",
            "positive drift lift here is modest and should not be treated as a",
            "production-strategy conclusion.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    metrics = run_all_suites()
    report = render_report(metrics)
    # Keep the checked-in report synchronized with the deterministic runner.
    # The path is fixed inside this research package and never touches
    # production strategy or supplied simulator files.
    Path(__file__).with_name("PASS2_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    print(report)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a module
    raise SystemExit(main())


__all__ = [
    "RunMetrics",
    "STRATEGY_NAMES",
    "aggregate_metrics",
    "format_summary_table",
    "main",
    "render_report",
    "robust_strategy",
    "run_all_suites",
    "run_suite",
]
