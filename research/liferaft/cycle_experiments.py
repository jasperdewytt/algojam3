"""Executable development-only experiments for the public-cycle detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

from .cycle_scenarios import CycleScenario, development_cycle_scenarios
from .cycle_strategies import PublicCycleDetector
from .simulator import (
    MajorityOutcome,
    SimulationResult,
    infer_majority_from_price_change,
)


REPORT_PATH = Path(__file__).with_name("CYCLE_REPORT.md")
CYCLE_STRATEGIES: tuple[str, ...] = (
    "flat",
    "burnin1_markov",
    "online_markov",
    "cycle_detector",
)


@dataclass(frozen=True)
class CycleMetric:
    scenario: str
    family: str
    strategy: str
    true_cycle: bool
    expected_period: int | None
    pivotal_fixture: bool
    detected_period: int | None
    detection_count: int
    first_detection_delay: int | None
    forecast_hits: int
    scoreable_forecasts: int
    forecast_accuracy_after_activation: float | None
    cycle_breaks: int
    reactivations: int
    active_days: int
    marked_pnl: int
    maximum_drawdown: int


@dataclass(frozen=True)
class ForecastScore:
    """Causal forecast score with an explicit no-scoreable-observation state."""

    hits: int
    scoreable: int

    @property
    def accuracy(self) -> float | None:
        return self.hits / self.scoreable if self.scoreable else None


def _marked_history(result: SimulationResult):
    assert result.focal_agent_name is not None
    start = result.config.voting_start_day
    assert start is not None
    return tuple(
        record
        for record in result.agent_history(result.focal_agent_name)
        if record.day > start
    )


def _maximum_drawdown(result: SimulationResult) -> int:
    cumulative = 0
    peak = 0
    drawdown = 0
    for record in _marked_history(result):
        cumulative += record.daily_pnl
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return drawdown


def score_cycle_forecasts(
    result: SimulationResult,
    detector: PublicCycleDetector,
) -> ForecastScore:
    """Score day-t forecasts against the public movement into day t+1.

    ``result.days[t + 1].majority`` is the newly selected vote for the next
    decision and is deliberately not used here. The realised movement into
    that row is the public evidence for day ``t``'s vote.
    """

    start = result.config.voting_start_day
    assert start is not None
    first_detection_day = (
        detector.detections[0].day if detector.detections else None
    )
    if first_detection_day is None:
        return ForecastScore(0, 0)
    forecasts = detector.forecast_by_day
    hits = 0
    scored = 0
    for day, forecast in forecasts.items():
        if day < first_detection_day or forecast is None or day + 1 >= len(result.days):
            continue
        source_day = result.days[day]
        movement_day = result.days[day + 1]
        if (
            day < start
            or not source_day.voting_active
            or not movement_day.voting_active
        ):
            continue
        # The source row records whether its next price was floor-clipped. A
        # reset row and a zero/ambiguous movement do not provide a public label.
        if movement_day.reset_applied or source_day.floor_clipped:
            continue
        observed = infer_majority_from_price_change(
            movement_day.price_change,
            long_majority_move=result.config.long_majority_move,
            short_majority_move=result.config.short_majority_move,
            previous_move_is_reset=movement_day.reset_applied,
        )
        if observed is None:
            continue
        scored += 1
        hits += forecast is observed
    return ForecastScore(hits, scored)


def metric_for_run(
    scenario: CycleScenario,
    strategy: str,
    result: SimulationResult,
    agent: object,
) -> CycleMetric:
    detector = agent if isinstance(agent, PublicCycleDetector) else None
    detection_events = detector.detections if detector is not None else ()
    forecast_score = (
        score_cycle_forecasts(result, detector)
        if detector is not None
        else ForecastScore(0, 0)
    )
    history = _marked_history(result)
    return CycleMetric(
        scenario=scenario.name,
        family=scenario.family,
        strategy=strategy,
        true_cycle=scenario.true_cycle,
        expected_period=scenario.expected_period,
        pivotal_fixture=scenario.pivotal,
        detected_period=(detection_events[0].period if detection_events else None),
        detection_count=len(detection_events),
        first_detection_delay=(
            detection_events[0].observed_live_count if detection_events else None
        ),
        forecast_hits=forecast_score.hits,
        scoreable_forecasts=forecast_score.scoreable,
        forecast_accuracy_after_activation=forecast_score.accuracy,
        cycle_breaks=detector.cycle_breaks if detector is not None else 0,
        reactivations=detector.reactivation_count if detector is not None else 0,
        active_days=sum(record.action != 0 for record in history),
        marked_pnl=result.final_marked_pnl,
        maximum_drawdown=_maximum_drawdown(result),
    )


def run_cycle_experiments(
    scenarios: Iterable[CycleScenario] | None = None,
    *,
    strategies: Sequence[str] = CYCLE_STRATEGIES,
    write_report: bool = True,
) -> tuple[CycleMetric, ...]:
    """Run only the new development fixtures, never Pass 3 or final cases."""

    scenario_list = tuple(
        development_cycle_scenarios() if scenarios is None else scenarios
    )
    metrics: list[CycleMetric] = []
    for index, scenario in enumerate(scenario_list, start=1):
        print(f"[cycle-development] scenario {index}/{len(scenario_list)} {scenario.name}")
        for strategy in strategies:
            result, agent = scenario.run(strategy)
            metrics.append(metric_for_run(scenario, strategy, result, agent))
    result = tuple(metrics)
    if write_report:
        REPORT_PATH.write_text(render_cycle_report(result, scenario_list), encoding="utf-8")
        print(f"wrote {REPORT_PATH}")
    return result


def _summary(metrics: Sequence[CycleMetric], strategy: str) -> dict[str, float]:
    values = [metric for metric in metrics if metric.strategy == strategy]
    return {
        "runs": float(len(values)),
        "mean_pnl": mean(metric.marked_pnl for metric in values) if values else 0.0,
        "median_pnl": median(metric.marked_pnl for metric in values) if values else 0.0,
        "mean_dd": mean(metric.maximum_drawdown for metric in values) if values else 0.0,
        "max_dd": max((metric.maximum_drawdown for metric in values), default=0),
        "mean_active": mean(metric.active_days for metric in values) if values else 0.0,
    }


def _detector_summary(metrics: Sequence[CycleMetric]) -> dict[str, float | None]:
    detector = [metric for metric in metrics if metric.strategy == "cycle_detector"]
    true_cases = [metric for metric in detector if metric.true_cycle]
    controls = [metric for metric in detector if not metric.true_cycle]
    detected_true = [metric for metric in true_cases if metric.detection_count]
    activated_controls = [metric for metric in controls if metric.detection_count]
    delays = [metric.first_detection_delay for metric in detected_true if metric.first_detection_delay is not None]
    accuracy_metrics = [
        metric.forecast_accuracy_after_activation
        for metric in detected_true
        if metric.forecast_accuracy_after_activation is not None
    ]
    pooled_hits = sum(metric.forecast_hits for metric in detected_true)
    pooled_scoreable = sum(metric.scoreable_forecasts for metric in detected_true)
    return {
        "true_cases": float(len(true_cases)),
        "detection_rate": len(detected_true) / len(true_cases) if true_cases else 0.0,
        "control_cases": float(len(controls)),
        "control_activation_rate": len(activated_controls) / len(controls) if controls else 0.0,
        "mean_delay": mean(delays) if delays else 0.0,
        "pooled_hits": float(pooled_hits),
        "pooled_scoreable": float(pooled_scoreable),
        "pooled_accuracy": pooled_hits / pooled_scoreable if pooled_scoreable else None,
        "mean_accuracy": mean(accuracy_metrics) if accuracy_metrics else None,
        "breaks": float(sum(metric.cycle_breaks for metric in detector)),
        "reactivations": float(sum(metric.reactivations for metric in detector)),
    }


def _strategy_table(metrics: Sequence[CycleMetric]) -> str:
    lines = [
        "| strategy | runs | mean marked P&L | median marked P&L | mean DD | max DD | mean effective active days |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in CYCLE_STRATEGIES:
        row = _summary(metrics, strategy)
        lines.append(
            f"| `{strategy}` | {int(row['runs'])} | {row['mean_pnl']:.0f} | "
            f"{row['median_pnl']:.0f} | {row['mean_dd']:.0f} | {row['max_dd']:.0f} | "
            f"{row['mean_active']:.1f} |"
        )
    return "\n".join(lines)


def _detector_table(metrics: Sequence[CycleMetric]) -> str:
    rows = [metric for metric in metrics if metric.strategy == "cycle_detector"]
    lines = [
        "| scenario | family | true cycle | expected period | detected period | first delay | hits | scoreable | accuracy after activation | breaks | reactivations | active days | marked P&L | max DD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in rows:
        delay = "-" if metric.first_detection_delay is None else str(metric.first_detection_delay)
        period = "-" if metric.detected_period is None else str(metric.detected_period)
        expected = "-" if metric.expected_period is None else str(metric.expected_period)
        accuracy = (
            "N/A"
            if metric.forecast_accuracy_after_activation is None
            else f"{metric.forecast_accuracy_after_activation:.1%}"
        )
        lines.append(
            f"| {metric.scenario} | {metric.family} | {str(metric.true_cycle).lower()} | "
            f"{expected} | {period} | {delay} | {metric.forecast_hits} | "
            f"{metric.scoreable_forecasts} | {accuracy} | {metric.cycle_breaks} | "
            f"{metric.reactivations} | {metric.active_days} | {metric.marked_pnl} | "
            f"{metric.maximum_drawdown} |"
        )
    return "\n".join(lines)


def render_cycle_report(
    metrics: Sequence[CycleMetric],
    scenarios: Sequence[CycleScenario],
) -> str:
    detector = _detector_summary(metrics)
    return_cycle = next(
        (scenario for scenario in scenarios if scenario.name == "return-cycle-13-movements"),
        None,
    )
    return_path_note = "not run"
    if return_cycle is not None:
        # The fixture is non-pivotal, so the zero-net result is a direct public
        # path diagnostic rather than an inferred hidden-vote quantity.
        result, _agent = return_cycle.run("flat")
        start = result.config.voting_start_day
        assert start is not None
        if start + 13 < len(result.price_path):
            return_path_note = (
                f"flat-focal price change over first 13 live movements: "
                f"${result.price_path[start + 13] - result.price_path[start]:,}"
            )

    return f"""# Development-only deterministic Liferaft cycle report

> This report is an isolated mechanics experiment. It does not add a strategy
> to Pass 3, does not change the locked candidate catalogue, and is not a
> representative estimate of competition probabilities. It was not evaluated
> on consumed Pass 3 validation or on the locked final suite.

## Frozen detector specification

- Candidate periods are exactly integers `2..20`.
- Activation requires three complete, identical consecutive blocks.
- Every block label must be a genuine publicly inferred `LONG` or `SHORT`.
- Reset, zero/tie, and publicly identifiable floor-clipped observations clear
  the context.
- The shortest qualifying period is selected. The next label is forecast and
  converted through the existing asymmetric `payoff_action` safeguards.
- A contradictory observed label immediately deactivates the detector. It
  remains flat until another three-block context is causally established.
- The detector consumes no same-day unseen outcome and receives no hidden
  simulator fields.

## Fixture inventory

Development cases: **{len(scenarios)}**. They cover pure periods 2/3/4/5/7/13/20,
varied phases, the 13-movement 8-long/5-short zero-price-return cycle, multiple
periodic components, breaks/restarts, a period regime switch, corrupted and
non-periodic controls, random/Markov-like controls, unknown movements, floor
clipping, runaway budget behavior, pivotality, and both startup execution modes.

## Detection diagnostics

| true-cycle cases | detection rate | control cases | control activation rate | mean detection delay | pooled hits | pooled scoreable forecasts | pooled accuracy | mean per-scenario accuracy | total breaks | total reactivations |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| {int(detector['true_cases'])} | {detector['detection_rate']:.1%} | {int(detector['control_cases'])} | {detector['control_activation_rate']:.1%} | {detector['mean_delay']:.1f} | {int(detector['pooled_hits'])} | {int(detector['pooled_scoreable'])} | {('N/A' if detector['pooled_accuracy'] is None else f"{detector['pooled_accuracy']:.1%}")} | {('N/A' if detector['mean_accuracy'] is None else f"{detector['mean_accuracy']:.1%}")} | {int(detector['breaks'])} | {int(detector['reactivations'])} |

The previous report used the off-by-one comparison with the next decision's
`majority` field and displayed a mean accuracy of `48.4%`; it did not retain hit
or scoreable-observation counts. That historical figure is retained only to
make the reporting correction auditable. The corrected result above scores the
public movement generated by the forecast day and reports pooled hits divided
by pooled scoreable forecasts. `N/A` means that no public forecast movement
was scoreable.

Accuracy excludes reset movements, zero/ambiguous movements, and source rows
whose next movement was floor-clipped. A pivotal forecast that turns the
realised vote into a tie is therefore unscoreable, not an automatic miss.

{_detector_table(metrics)}

## Comparison candidates (secondary P&L diagnostics)

The only comparison candidates are flat, `burnin1_markov`, and
`online_markov`. P&L and drawdown below are secondary mechanics diagnostics;
they were not used to alter the detector's frozen specification.

{_strategy_table(metrics)}

## Focused observations

- {return_path_note}. The detector is expected to wait for 39 genuine labels
  before a period-13 activation, so a short fixture may show a long delay.
- The pivotal fixture is intentionally adversarial: a focal action can destroy
  or alter the apparent public cycle, unlike the clearly non-pivotal unanimous
  populations.
- A constant persistent control has primitive period 1, but it still satisfies
  the declared period-2 repeated-block equality. That is predictable
  persistence, not convincing evidence of a period-2 cycle.
- The detector scans overlapping windows across multiple candidate periods.
  Even an independent binary sequence gives a particular six-label period-2
  window probability `1/16`; repeated scanning creates a substantial
  multiple-testing problem with no false-discovery control. The `6/8` control
  activation result is material evidence of that limitation.

## Conclusion

The current detector is **not worth pursuing in its present form**. The causal
repeated-block mechanism works on constructed deterministic fixtures, but the
`6/8` control activation rate, primitive-period ambiguity, and uncorrected
multiple-testing problem outweigh that mechanical demonstration. Primitive-
period checks, confirmation holdouts, or statistical significance controls are
future theoretical proposals only; they were not applied here because doing so
would be post-result detector redesign. The detector remains quarantined from
the Pass 3 catalogue and locked-final manifest.
"""


def main() -> None:
    run_cycle_experiments()


if __name__ == "__main__":
    main()
