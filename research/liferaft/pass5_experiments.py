"""Development-only Liferaft Pass 5A screening experiment.

Only the existing development and consumed-validation scenario constructors
are used here.  The consumed final-suite builder and artifacts are deliberately
not imported or called.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .cold_start_strategies import make_cold_start_strategy
from .pass3_scenarios import (
    Pass3Scenario,
    development_scenarios,
    validation_scenarios,
)
from .pass4_strategies import Risk50Burnin1Markov
from .simulator import AgentObservation, LiferaftSimulator, SimulationResult
from .shadow_strategies import (
    SHADOW_PARAMETERS,
    SHADOW_STRATEGY_NAMES,
    ShadowValidatedMarkov,
    make_shadow_strategy,
)


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "PASS5A_RESULTS.json"
REPORT_PATH = ROOT / "PASS5A_REPORT.md"
MANIFEST_PATH = ROOT / "PASS5A_MANIFEST.md"
EXPOSURES = (0, 150_000, 300_000, 450_000)
COMPARATOR_NAMES = (
    "flat",
    "burnin1_markov",
    "online_markov",
    Risk50Burnin1Markov.name,
)
EXPERIMENT_STRATEGIES = SHADOW_STRATEGY_NAMES + COMPARATOR_NAMES
PASS5_RESULT_SOURCE_FILES = (
    "PASS5A_PROTOCOL.md",
    "archetypes.py",
    "simulator.py",
    "strategies.py",
    "cold_start_strategies.py",
    "pass3_scenarios.py",
    "pass4_strategies.py",
    "shadow_strategies.py",
    "pass5_experiments.py",
)


@dataclass
class ExposureResolver:
    """Share one cached callable exposure value between strategy and engine."""

    source: Callable[[AgentObservation], float]
    focal_name: str = "focal"

    def __post_init__(self) -> None:
        self._cache: dict[int, float] = {}
        self.provider_calls: list[int] = []
        self.cache_hits: list[int] = []

    def resolve(self, observation: AgentObservation) -> float:
        if observation.day in self._cache:
            self.cache_hits.append(observation.day)
            return self._cache[observation.day]
        value = float(self.source(observation))
        if value < 0:
            raise ValueError("callable exposure audit returned negative exposure")
        self._cache[observation.day] = value
        self.provider_calls.append(observation.day)
        return value

    def strategy_source(self, observation: AgentObservation) -> float:
        return self.resolve(observation)

    def simulator_source(
        self,
        agent_name: str,
        observation: AgentObservation,
    ) -> float:
        if agent_name != self.focal_name:
            return 0.0
        return self.resolve(observation)


def _json_safe(value: object) -> object:
    if hasattr(value, "value"):
        return getattr(value, "value")
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _json_safe(item)
            for key, item in asdict(value).items()  # type: ignore[arg-type]
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _make_focal(
    strategy_name: str,
    exposure: float | Callable[[AgentObservation], float],
) -> object:
    if strategy_name in SHADOW_STRATEGY_NAMES:
        return make_shadow_strategy(
            strategy_name,
            other_portfolio_exposure=exposure,
        )
    if strategy_name == Risk50Burnin1Markov.name:
        return Risk50Burnin1Markov(other_portfolio_exposure=exposure)
    return make_cold_start_strategy(strategy_name)


def run_cell(
    scenario: Pass3Scenario,
    strategy_name: str,
    exposure: float | Callable[[AgentObservation], float],
) -> tuple[SimulationResult, object, ExposureResolver | None]:
    """Run one fresh focal/opponent cell."""

    resolver: ExposureResolver | None = None
    if callable(exposure):
        if strategy_name not in SHADOW_STRATEGY_NAMES and strategy_name != Risk50Burnin1Markov.name:
            raise ValueError("callable exposure audit supports only shadow/risk cells")
        resolver = ExposureResolver(exposure)
        focal_exposure: float | Callable[[AgentObservation], float] = resolver.strategy_source
    else:
        focal_exposure = float(exposure)

    focal = _make_focal(strategy_name, focal_exposure)
    focal_name = getattr(focal, "name")
    opponents = scenario.population_factory()
    engine_exposure = (
        resolver.simulator_source
        if resolver is not None
        else {focal_name: float(exposure)}
    )
    result = LiferaftSimulator(
        (focal, *opponents),
        scenario.config,
        other_portfolio_exposure=engine_exposure,
        focal_agent_name=focal_name,
        scenario_name=scenario.name,
        scenario_configuration={
            "suite": "pass5a-development",
            "family": scenario.family,
            "seed": scenario.seed,
            "execution_mode": scenario.execution_mode,
            "population": scenario.population_description,
            "population_size": scenario.population_size,
            "pair_id": scenario.pair_id,
            "path_controlled": scenario.path_controlled,
            "other_portfolio_exposure": (
                "callable_audit" if resolver is not None else float(exposure)
            ),
        },
        random_seeds={"scenario": scenario.seed},
    ).run()
    return result, focal, resolver


def _history(result: SimulationResult) -> tuple:
    if result.focal_agent_name is None:
        raise ValueError("Pass 5A metrics require a focal agent")
    return result.agent_history(result.focal_agent_name)


def _drawdown(history: Sequence) -> int:
    peak = 0
    maximum = 0
    for record in history:
        peak = max(peak, record.marked_cumulative_pnl)
        maximum = max(maximum, peak - record.marked_cumulative_pnl)
    return maximum


def _turnover(history: Sequence, start_day: int) -> int:
    prior = 0
    turnover = 0
    for record in history:
        if record.day < start_day:
            continue
        turnover += abs(record.action - prior)
        prior = record.action
    return turnover


def _pivotal_pnl(result: SimulationResult, history: Sequence) -> tuple[int, int]:
    pivotal = 0
    non_pivotal = 0
    for record in history:
        if record.pnl_source_day is None or record.daily_pnl == 0:
            continue
        source = result.days[record.pnl_source_day]
        if source.focal_pivotal:
            pivotal += record.daily_pnl
        else:
            non_pivotal += record.daily_pnl
    return pivotal, non_pivotal


def _diagnostics_for(strategy: object) -> dict[str, object]:
    if isinstance(strategy, ShadowValidatedMarkov):
        diagnostics = strategy.diagnostics()
        diagnostics.pop("timeline", None)
        return _json_safe(diagnostics)  # type: ignore[return-value]
    if isinstance(strategy, Risk50Burnin1Markov):
        return _json_safe(strategy.diagnostics())  # type: ignore[return-value]
    return {}


def _loss_stop_values(strategy: object) -> tuple[int, bool, object | None]:
    if isinstance(strategy, ShadowValidatedMarkov):
        return (
            strategy.actual_cumulative_pnl,
            strategy.loss_stop_active,
            strategy.loss_stop_event,
        )
    if isinstance(strategy, Risk50Burnin1Markov):
        return (
            strategy.cumulative_marked_pnl,
            strategy.loss_stop_active,
            strategy.loss_stop_event,
        )
    return (0, False, None)


def _loss_consistent(actual: int, marked: int, event: object | None) -> bool:
    if actual != marked:
        return False
    if event is None:
        return True
    before = getattr(event, "pnl_before", None)
    after = getattr(event, "pnl_after", None)
    overshoot = getattr(event, "loss_limit_overshoot", None)
    return (
        after == actual
        and before is not None
        and overshoot == max(0, -after - 50_000)
    )


def _base_metric(
    result: SimulationResult,
    strategy_name: str,
    scenario: Pass3Scenario,
    exposure: float,
    strategy: object,
) -> dict[str, object]:
    focal = result.focal_agent_name
    if focal is None:
        raise ValueError("result has no focal agent")
    history = _history(result)
    start = result.config.voting_start_day or result.config.marked_boundary_day
    live = tuple(record for record in history if record.day >= start)
    pivotal, non_pivotal = _pivotal_pnl(result, history)
    actual_pnl, loss_stop_active, loss_event = _loss_stop_values(strategy)
    has_strategy_pnl_ledger = isinstance(
        strategy,
        (ShadowValidatedMarkov, Risk50Burnin1Markov),
    )
    if not has_strategy_pnl_ledger:
        actual_pnl = result.marked_pnl[focal]
    metric: dict[str, object] = {
        "strategy": strategy_name,
        "scenario": scenario.name,
        "family": scenario.family,
        "execution_mode": scenario.execution_mode,
        "pair_id": scenario.pair_id,
        "seed": scenario.seed,
        "population_size": scenario.population_size,
        "path_controlled": scenario.path_controlled,
        "exposure": exposure,
        "live_decisions": len(live),
        "marked_pnl": result.marked_pnl[focal],
        "maximum_drawdown": _drawdown(history),
        "active_days": sum(record.action != 0 for record in live),
        "real_nonzero_days": sum(record.action != 0 for record in live),
        "turnover": _turnover(history, start),
        "budget_breaches": sum(item.agent_name == focal for item in result.budget_breaches),
        "rejected_actions": sum(item.agent_name == focal for item in result.rejected_actions),
        "pivotal_pnl": pivotal,
        "non_pivotal_pnl": non_pivotal,
        "actual_realised_pnl": actual_pnl,
        "loss_stop_active": loss_stop_active,
        "loss_stop_event": _json_safe(loss_event),
        "loss_stop_consistent": _loss_consistent(
            actual_pnl,
            result.marked_pnl[focal],
            loss_event,
        ),
        "activation_count": 0,
        "activation_day": None,
        "reactivation_count": 0,
        "deactivation_count": 0,
        "active_state_days": sum(record.action != 0 for record in live),
        "paused_days": 0,
        "cooldown_days": 0,
        "virtual_pnl_before_activation": 0,
        "virtual_pnl_after_activation": 0,
        "real_pnl_after_activation": 0,
        "scoreable_virtual_trades": 0,
        "genuine_nonzero_observations": 0,
        "current_edge_gate_count": 0,
        "floor_gate_count": 0,
        "unknown_gate_count": 0,
        "headroom_gate_count": 0,
        "exposure_evaluation_count": 0,
        "activation_events": [],
        "deactivation_events": [],
        "pause_events": [],
        "path_digest": "",
        "strategy_diagnostics": _diagnostics_for(strategy),
    }

    if isinstance(strategy, ShadowValidatedMarkov):
        timeline = strategy.timeline
        activation_day = strategy.activation_day
        metric.update(
            {
                "activation_count": len(strategy.activation_events),
                "activation_day": activation_day,
                "reactivation_count": strategy.reactivation_count,
                "deactivation_count": len(strategy.deactivation_events),
                "active_state_days": sum(
                    item.real_active and item.live_decision for item in timeline
                ),
                "paused_days": sum(
                    item.pause_active and item.live_decision for item in timeline
                ),
                "cooldown_days": sum(
                    item.cooldown_active and item.live_decision for item in timeline
                ),
                "scoreable_virtual_trades": strategy.scoreable_virtual_trades,
                "genuine_nonzero_observations": strategy.genuine_nonzero_observations,
                "current_edge_gate_count": strategy.current_edge_gate_count,
                "floor_gate_count": strategy.floor_gate_count,
                "unknown_gate_count": strategy.unknown_gate_count,
                "headroom_gate_count": strategy.headroom_gate_count,
                "exposure_evaluation_count": strategy.exposure_evaluation_count,
                "activation_events": list(strategy.activation_events),
                "deactivation_events": list(strategy.deactivation_events),
                "pause_events": list(strategy.diagnostics()["pause_events"]),
            }
        )
        if activation_day is None:
            before = timeline
            after = ()
        else:
            before = tuple(item for item in timeline if item.day <= activation_day)
            after = tuple(item for item in timeline if item.day > activation_day)
        metric["virtual_pnl_before_activation"] = sum(
            item.virtual_interval_pnl for item in before
        )
        metric["virtual_pnl_after_activation"] = sum(
            item.virtual_interval_pnl for item in after
        )
        metric["real_pnl_after_activation"] = sum(
            item.actual_pnl_increment for item in after
        )
    elif isinstance(strategy, Risk50Burnin1Markov):
        diagnostics = strategy.diagnostics()
        metric.update(
            {
                "paused_days": int(diagnostics["unknown_pause_count"]),
                "floor_gate_count": int(diagnostics["floor_gate_count"]),
                "unknown_gate_count": int(diagnostics["unknown_pause_count"]),
                "headroom_gate_count": int(diagnostics["headroom_gate_count"]),
            }
        )

    metric["path_digest"] = _path_digest(result)
    return metric


def _path_digest(result: SimulationResult) -> str:
    payload = json.dumps(
        {
            "prices": result.price_path,
            "majorities": [day.majority.value for day in result.days],
            "actions": [day.actions for day in result.days],
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(metrics: Sequence[dict[str, object]]) -> dict[str, float | int]:
    """Aggregate directly from run-level metric rows."""

    pnl = [float(metric["marked_pnl"]) for metric in metrics]
    drawdowns = [float(metric["maximum_drawdown"]) for metric in metrics]

    def total(name: str) -> int:
        return sum(int(metric[name]) for metric in metrics)

    def mean_field(name: str) -> float:
        return _mean([float(metric[name]) for metric in metrics])

    shadow = [metric for metric in metrics if metric["strategy"] in SHADOW_STRATEGY_NAMES]
    activated = [metric for metric in shadow if int(metric["activation_count"]) > 0]
    activation_days = [
        float(metric["activation_day"])
        for metric in activated
        if metric["activation_day"] is not None
    ]
    return {
        "runs": len(metrics),
        "mean_pnl": _mean(pnl),
        "median_pnl": _quantile(pnl, 0.50),
        "lower_quartile_pnl": _quantile(pnl, 0.25),
        "worst_pnl": min(pnl) if pnl else 0.0,
        "mean_drawdown": _mean(drawdowns),
        "maximum_drawdown": max(drawdowns) if drawdowns else 0.0,
        "mean_active_days": mean_field("active_days"),
        "mean_turnover": mean_field("turnover"),
        "live_decisions": total("live_decisions"),
        "budget_breaches": total("budget_breaches"),
        "rejected_actions": total("rejected_actions"),
        "pivotal_pnl": mean_field("pivotal_pnl"),
        "non_pivotal_pnl": mean_field("non_pivotal_pnl"),
        "activation_rate": len(activated) / len(shadow) if shadow else 0.0,
        "mean_activation_day": _mean(activation_days),
        "median_activation_day": _quantile(activation_days, 0.50),
        "never_activated_fraction": (
            sum(metric["activation_day"] is None for metric in shadow) / len(shadow)
            if shadow
            else 0.0
        ),
        "deactivation_frequency": (
            _mean([float(metric["deactivation_count"]) for metric in shadow])
            if shadow
            else 0.0
        ),
        "reactivation_frequency": (
            _mean([float(metric["reactivation_count"]) for metric in shadow])
            if shadow
            else 0.0
        ),
        "mean_active_state_days": mean_field("active_state_days"),
        "mean_paused_days": mean_field("paused_days"),
        "mean_cooldown_days": mean_field("cooldown_days"),
        "virtual_pnl_before_activation": mean_field("virtual_pnl_before_activation"),
        "virtual_pnl_after_activation": mean_field("virtual_pnl_after_activation"),
        "real_pnl_after_activation": mean_field("real_pnl_after_activation"),
        "current_edge_gate_count": total("current_edge_gate_count"),
        "floor_gate_count": total("floor_gate_count"),
        "unknown_gate_count": total("unknown_gate_count"),
        "headroom_gate_count": total("headroom_gate_count"),
        "current_edge_gate_frequency": (
            total("current_edge_gate_count") / total("live_decisions")
            if total("live_decisions")
            else 0.0
        ),
        "unknown_gate_frequency": (
            total("unknown_gate_count") / total("live_decisions")
            if total("live_decisions")
            else 0.0
        ),
        "loss_stop_runs": sum(bool(metric["loss_stop_active"]) for metric in metrics),
        "loss_stop_inconsistencies": sum(
            not bool(metric["loss_stop_consistent"]) for metric in metrics
        ),
    }


def _group(
    metrics: Sequence[dict[str, object]],
    *keys: str,
) -> dict[tuple[object, ...], list[dict[str, object]]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for metric in metrics:
        grouped[tuple(metric[key] for key in keys)].append(metric)
    return grouped


def _paired_differences(
    metrics: Sequence[dict[str, object]],
    left: str,
    right: str,
    exposure: int | None = None,
) -> list[float]:
    grouped: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    for metric in metrics:
        if exposure is not None and metric["exposure"] != exposure:
            continue
        grouped[(metric["scenario"], metric["exposure"])][str(metric["strategy"])] = metric
    return [
        float(pair[left]["marked_pnl"]) - float(pair[right]["marked_pnl"])
        for pair in grouped.values()
        if left in pair and right in pair
    ]


def _retention_against_wrapper(
    metrics: Sequence[dict[str, object]],
    strategy: str,
    exposure: int,
) -> dict[str, float | int]:
    grouped: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    for metric in metrics:
        if metric["exposure"] != exposure:
            continue
        if metric["strategy"] not in {strategy, Risk50Burnin1Markov.name}:
            continue
        grouped[(metric["scenario"], metric["exposure"])][str(metric["strategy"])] = metric
    positive = [
        pair
        for pair in grouped.values()
        if Risk50Burnin1Markov.name in pair
        and float(pair[Risk50Burnin1Markov.name]["marked_pnl"]) > 0
    ]
    negative = [
        pair
        for pair in grouped.values()
        if Risk50Burnin1Markov.name in pair
        and float(pair[Risk50Burnin1Markov.name]["marked_pnl"]) < 0
    ]
    positive_available = [pair for pair in positive if strategy in pair]
    negative_available = [pair for pair in negative if strategy in pair]
    return {
        "positive_wrapper_cases": len(positive),
        "positive_shadow_fraction": _mean(
            [float(pair[strategy]["marked_pnl"]) > 0 for pair in positive_available]
        ),
        "positive_retention_ratio": _mean(
            [
                float(pair[strategy]["marked_pnl"])
                / float(pair[Risk50Burnin1Markov.name]["marked_pnl"])
                for pair in positive_available
            ]
        ),
        "negative_wrapper_cases": len(negative),
        "downside_avoided_fraction": _mean(
            [
                float(pair[strategy]["marked_pnl"])
                > float(pair[Risk50Burnin1Markov.name]["marked_pnl"])
                for pair in negative_available
            ]
        ),
        "mean_negative_case_improvement": _mean(
            [
                float(pair[strategy]["marked_pnl"])
                - float(pair[Risk50Burnin1Markov.name]["marked_pnl"])
                for pair in negative_available
            ]
        ),
    }


def _render_overall_table(metrics: Sequence[dict[str, object]]) -> str:
    lines = [
        "| exposure | strategy | runs | mean P&L | median | lower quartile | worst | mean DD | max DD | active days | turnover | breaches | rejected | pivotal P&L | non-pivotal P&L |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for exposure in EXPOSURES:
        for strategy in EXPERIMENT_STRATEGIES:
            rows = [
                metric
                for metric in metrics
                if metric["exposure"] == exposure and metric["strategy"] == strategy
            ]
            summary = summarize(rows)
            lines.append(
                "| {exposure:,} | `{strategy}` | {runs} | {mean_pnl:.0f} | {median_pnl:.0f} | {lower_quartile_pnl:.0f} | {worst_pnl:.0f} | {mean_drawdown:.0f} | {maximum_drawdown:.0f} | {mean_active_days:.1f} | {mean_turnover:.1f} | {budget_breaches} | {rejected_actions} | {pivotal_pnl} | {non_pivotal_pnl} |".format(
                    exposure=exposure,
                    strategy=strategy,
                    **summary,
                )
            )
    return "\n".join(lines)


def _render_strategy_summary_table(metrics: Sequence[dict[str, object]]) -> str:
    lines = [
        "| strategy | runs | mean P&L | median | lower quartile | worst | mean DD | max DD | activation rate | median activation day | never activated | deactivation frequency | reactivation frequency | active state days | paused days | cooldown days | virtual before | virtual after | real after | edge gates | edge frequency | unknown gates | floor gates | headroom gates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in EXPERIMENT_STRATEGIES:
        rows = [metric for metric in metrics if metric["strategy"] == strategy]
        summary = summarize(rows)
        lines.append(
            f"| `{strategy}` | {summary['runs']} | {summary['mean_pnl']:.0f} | {summary['median_pnl']:.0f} | {summary['lower_quartile_pnl']:.0f} | {summary['worst_pnl']:.0f} | {summary['mean_drawdown']:.0f} | {summary['maximum_drawdown']:.0f} | {summary['activation_rate']:.1%} | {summary['median_activation_day']:.1f} | {summary['never_activated_fraction']:.1%} | {summary['deactivation_frequency']:.2f} | {summary['reactivation_frequency']:.2f} | {summary['mean_active_state_days']:.1f} | {summary['mean_paused_days']:.1f} | {summary['mean_cooldown_days']:.1f} | {summary['virtual_pnl_before_activation']:.0f} | {summary['virtual_pnl_after_activation']:.0f} | {summary['real_pnl_after_activation']:.0f} | {summary['current_edge_gate_count']} | {summary['current_edge_gate_frequency']:.2%} | {summary['unknown_gate_count']} | {summary['floor_gate_count']} | {summary['headroom_gate_count']} |"
        )
    return "\n".join(lines)


def _render_group_table(
    metrics: Sequence[dict[str, object]],
    group_key: str,
) -> str:
    label = "family" if group_key == "family" else "execution mode"
    lines = [
        f"| {label} | strategy | runs | mean P&L | median | lower quartile | worst | mean DD | max DD | active days | activation rate | never activated | deactivations | reactivations | cooling days | paused days |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    pairs = sorted(
        {(metric[group_key], metric["strategy"]) for metric in metrics},
        key=lambda item: (str(item[0]), str(item[1])),
    )
    for group, strategy in pairs:
        rows = [
            metric
            for metric in metrics
            if metric[group_key] == group and metric["strategy"] == strategy
        ]
        summary = summarize(rows)
        lines.append(
            f"| {group} | `{strategy}` | {summary['runs']} | {summary['mean_pnl']:.0f} | {summary['median_pnl']:.0f} | {summary['lower_quartile_pnl']:.0f} | {summary['worst_pnl']:.0f} | {summary['mean_drawdown']:.0f} | {summary['maximum_drawdown']:.0f} | {summary['mean_active_days']:.1f} | {summary['activation_rate']:.1%} | {summary['never_activated_fraction']:.1%} | {summary['deactivation_frequency']:.2f} | {summary['reactivation_frequency']:.2f} | {summary['mean_cooldown_days']:.1f} | {summary['mean_paused_days']:.1f} |"
        )
    return "\n".join(lines)


def _render_exposure_table(metrics: Sequence[dict[str, object]]) -> str:
    lines = [
        "| exposure | strategy | mean P&L | worst | max DD | mean active days | activation rate | virtual before | virtual after | real after | edge gates | unknown gates | floor gates | headroom gates |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for exposure in EXPOSURES:
        for strategy in EXPERIMENT_STRATEGIES:
            rows = [
                metric
                for metric in metrics
                if metric["exposure"] == exposure and metric["strategy"] == strategy
            ]
            summary = summarize(rows)
            lines.append(
                f"| {exposure:,} | `{strategy}` | {summary['mean_pnl']:.0f} | {summary['worst_pnl']:.0f} | {summary['maximum_drawdown']:.0f} | {summary['mean_active_days']:.1f} | {summary['activation_rate']:.1%} | {summary['virtual_pnl_before_activation']:.0f} | {summary['virtual_pnl_after_activation']:.0f} | {summary['real_pnl_after_activation']:.0f} | {summary['current_edge_gate_count']} | {summary['unknown_gate_count']} | {summary['floor_gate_count']} | {summary['headroom_gate_count']} |"
            )
    return "\n".join(lines)


def _render_pair_table(metrics: Sequence[dict[str, object]], right: str) -> str:
    lines = [
        f"| exposure | shadow | paired runs | mean shadow - {right} | median | shadow wins | ties | shadow losses |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for exposure in EXPOSURES:
        for shadow in SHADOW_STRATEGY_NAMES:
            differences = _paired_differences(metrics, shadow, right, exposure)
            wins = sum(value > 0 for value in differences)
            ties = sum(value == 0 for value in differences)
            lines.append(
                f"| {exposure:,} | `{shadow}` | {len(differences)} | {_mean(differences):.0f} | {_quantile(differences, 0.5):.0f} | {wins} | {ties} | {len(differences) - wins - ties} |"
            )
    return "\n".join(lines)


def _select_challenger(metrics: Sequence[dict[str, object]]) -> dict[str, object]:
    screening: dict[str, dict[str, object]] = {}
    for name in SHADOW_STRATEGY_NAMES:
        rows = [metric for metric in metrics if metric["strategy"] == name]
        summary = summarize(rows)
        checks = {
            "zero_budget_breaches": summary["budget_breaches"] == 0,
            "zero_rejected_actions": summary["rejected_actions"] == 0,
            "worst_pnl_at_least_minus_60000": summary["worst_pnl"] >= -60_000,
            "maximum_drawdown_at_most_75000": summary["maximum_drawdown"] <= 75_000,
            "positive_aggregate_mean_pnl": summary["mean_pnl"] > 0,
            "loss_stop_overshoot_consistent": summary["loss_stop_inconsistencies"] == 0,
        }
        screening[name] = {
            "checks": checks,
            "eligible": all(checks.values()),
            "summary": summary,
        }
    eligible = [name for name, item in screening.items() if item["eligible"]]
    selected: str | None = None
    tie_group: list[str] = []
    if eligible:
        means = {
            name: float(screening[name]["summary"]["mean_pnl"])
            for name in eligible
        }
        best = max(means.values())
        tie_group = [
            name
            for name in eligible
            if abs(means[name] - best) / max(abs(best), 1.0) < 0.05
        ]
        if "shadow12_markov" in tie_group:
            selected = "shadow12_markov"
        else:
            selected = max(
                tie_group,
                key=lambda name: SHADOW_PARAMETERS[name].minimum_genuine_nonzero_observations,
            )
    return {
        "screening": screening,
        "eligible_shadow_candidates": eligible,
        "tie_group": tie_group,
        "selected_challenger": selected,
    }


def _hash_sources() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename in PASS5_RESULT_SOURCE_FILES:
        path = ROOT / filename
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _render_selection(selection: dict[str, object]) -> str:
    lines = [
        "| candidate | eligible | mean P&L | worst P&L | max DD | breaches | rejected | loss consistency |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    screening = selection["screening"]
    assert isinstance(screening, dict)
    for name in SHADOW_STRATEGY_NAMES:
        item = screening[name]
        assert isinstance(item, dict)
        summary = item["summary"]
        checks = item["checks"]
        assert isinstance(summary, dict) and isinstance(checks, dict)
        lines.append(
            f"| `{name}` | {item['eligible']} | {summary['mean_pnl']:.0f} | {summary['worst_pnl']:.0f} | {summary['maximum_drawdown']:.0f} | {summary['budget_breaches']} | {summary['rejected_actions']} | {checks['loss_stop_overshoot_consistent']} |"
        )
    return "\n".join(lines)


def callable_exposure_audit(scenario: Pass3Scenario) -> dict[str, object]:
    """Use one existing scenario to audit cached callable exposure."""

    def alternating_exposure(observation: AgentObservation) -> float:
        return 0.0 if observation.day % 2 == 0 else 590_000.0

    callable_result, callable_strategy, resolver = run_cell(
        scenario,
        "shadow12_markov",
        alternating_exposure,
    )
    fixed_result, _fixed_strategy, _ = run_cell(
        scenario,
        "shadow12_markov",
        0.0,
    )
    assert resolver is not None
    start = scenario.config.voting_start_day
    assert start is not None
    live_days = scenario.config.total_days - start
    return {
        "scenario": scenario.name,
        "strategy": "shadow12_markov",
        "exposure_schedule": "AUD 0 on even days, AUD 590000 on odd days",
        "provider_calls": list(resolver.provider_calls),
        "provider_call_count": len(resolver.provider_calls),
        "unique_provider_days": len(set(resolver.provider_calls)),
        "cache_hit_count": len(resolver.cache_hits),
        "live_day_count": live_days,
        "at_most_one_underlying_evaluation_per_live_day": len(set(resolver.provider_calls)) == len(resolver.provider_calls),
        "callable_marked_pnl": callable_result.final_marked_pnl,
        "fixed_zero_exposure_marked_pnl": fixed_result.final_marked_pnl,
        "callable_path_digest": _path_digest(callable_result),
        "fixed_path_digest": _path_digest(fixed_result),
        "path_changed_endogenously": _path_digest(callable_result) != _path_digest(fixed_result),
        "budget_breaches": sum(item.agent_name == callable_result.focal_agent_name for item in callable_result.budget_breaches),
        "rejected_actions": sum(item.agent_name == callable_result.focal_agent_name for item in callable_result.rejected_actions),
        "strategy_exposure_evaluations": callable_strategy.exposure_evaluation_count,
    }


def _render_positive_retention(metrics: Sequence[dict[str, object]]) -> str:
    lines = [
        "| exposure | shadow | positive wrapper cases | positive shadow fraction | positive P&L retention ratio | negative wrapper cases | downside avoided fraction | mean improvement on negative cases |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for exposure in EXPOSURES:
        for shadow in SHADOW_STRATEGY_NAMES:
            values = _retention_against_wrapper(metrics, shadow, exposure)
            lines.append(
                f"| {exposure:,} | `{shadow}` | {values['positive_wrapper_cases']} | {values['positive_shadow_fraction']:.1%} | {values['positive_retention_ratio']:.2f} | {values['negative_wrapper_cases']} | {values['downside_avoided_fraction']:.1%} | {values['mean_negative_case_improvement']:.0f} |"
            )
    return "\n".join(lines)


def render_report(
    *,
    metrics: Sequence[dict[str, object]],
    scenarios: Sequence[Pass3Scenario],
    selection: dict[str, object],
    callable_audit: dict[str, object],
    runtime_seconds: float,
) -> str:
    development_count = sum(scenario.name.startswith("dev-") for scenario in scenarios)
    validation_count = len(scenarios) - development_count
    overall = summarize(metrics)
    return f"""# Liferaft Pass 5A Development Report

Status: development screening evidence only.  This is not a production
acceptance result and not a fresh holdout result.

The frozen protocol was written before the full run.  Only existing
`development_scenarios()` and `validation_scenarios()` constructors were used;
the validation cases are consumed historical development evidence.

## Scope and execution

- development scenarios: **{development_count}**
- consumed validation scenarios: **{validation_count}**
- total scenarios: **{len(scenarios)}**
- strategies per exposure: **{len(EXPERIMENT_STRATEGIES)}** ({', '.join(EXPERIMENT_STRATEGIES)})
- exposure levels: **{', '.join(f'AUD {value:,}' for value in EXPOSURES)}**
- run-level cells: **{len(metrics)}**
- runtime: **{runtime_seconds:.2f} seconds**
- aggregate run-level mean P&L: **AUD {overall['mean_pnl']:.0f}**

Every strategy/scenario/exposure cell used fresh focal and opponent instances.
Exposure gates can change the focal vote, so later majorities, prices,
reactive actions, and budget feasibility can differ endogenously.  Paired and
undertrading comparisons using those paths are labelled realised-path
diagnostics, not opponent-only counterfactuals.

## Aggregate strategy diagnostics

{_render_strategy_summary_table(metrics)}

This table reports actual run-level means, medians, lower quartiles, worst
marked P&L, mean and maximum drawdown, activation and never-activation rates,
deactivation/reactivation frequency, active/paused/cooling time, virtual P&L
before and after activation, real P&L after activation, and gate counts.

## Headline metrics by exposure

{_render_overall_table(metrics)}

Quantiles and worst values are calculated from individual runs rather than
averages of averages.  `active days` counts non-zero real actions; the active
state can be longer because an economic-edge, floor, headroom, pause, or loss
gate can keep the real position flat.

## Family summaries

{_render_group_table(metrics, 'family')}

## Execution-mode summaries

{_render_group_table(metrics, 'execution_mode')}

## Exposure sensitivity

{_render_exposure_table(metrics)}

These fixed other-exposure levels are portfolio-sensitivity counterfactuals,
not a forecast of complete portfolio allocation.

## Paired P&L against flat

{_render_pair_table(metrics, 'flat')}

## Paired P&L against `risk50_burnin1_markov`

{_render_pair_table(metrics, Risk50Burnin1Markov.name)}

## Positive-upside retention and downside avoided

These are realised-path diagnostics relative to the existing wrapper.  A
positive-wrapper case reports whether the shadow run remains positive and the
shadow/wrapper P&L ratio.  A negative-wrapper case reports the fraction with a
higher shadow P&L and its mean improvement.

{_render_positive_retention(metrics)}

## Callable-exposure integration audit

The audit used an existing scenario and a deterministic schedule of AUD 0 on
even days and AUD 590,000 on odd days.  It checks one underlying evaluation per
live focal day, cache sharing, no focal rejection, and whether the exposure
gate changes the endogenous path.  It does not manufacture a favourable
dynamic portfolio trace.

```json
{json.dumps(callable_audit, indent=2, sort_keys=True)}
```

## Activation, undertrading, pivotality, and weaknesses

Initial and reactivation eligibility requires the fixed genuine-observation
warm-up, six scoreable virtual trades, cumulative virtual P&L of at least AUD
10,000, recent-window P&L of at least AUD 5,000, and a current economic edge.
Two newly scoreable qualifying evaluations are required, and the movement
that completes qualification is never traded retroactively.  A weak current
forecast is a real flat decision without automatically deactivating the
strategy.  Unknown, zero, reset, and floor-clipped public movements do not
enter shadow health evidence.

Pivotal and non-pivotal P&L are engine-only reporting partitions.  The strategy
receives no hidden counts, margin, pivotality, or simulator diagnostics.
Pivotal populations, short opportunities during warm-up, floor/clipped paths,
and a small qualifying sample remain material sources of undertrading or
false qualification risk.  The family and mode tables show where activation
is rare, shadow gating is unprofitable, or the wrapper has a better trade-off.

## Mechanical screening and selection

{_render_selection(selection)}

The screening rule is fixed: zero focal budget breaches, zero focal rejected
actions, worst P&L at least AUD -60,000, maximum drawdown at most AUD 75,000,
positive aggregate mean P&L, and internally consistent loss-stop/overshoot
diagnostics.  Only eligible shadow candidates are considered.  If means are
within five percent, `shadow12_markov` wins; otherwise the highest aggregate
mean wins, with the longer warm-up as the prescribed fallback tie-break.

Selected challenger: **`{selection['selected_challenger']}`**.

This is only the challenger for a future blind Pass 5B.  It is not a
production acceptance decision.

## Quarantine and remaining uncertainty

The consumed final artifacts were not accessed, parsed, imported, executed,
recreated, renamed, deleted, or overwritten.  No final-suite command was run,
and no final scenario constructor was called.  Production trading files and
the existing final catalogue were not modified.

No blind Pass 5B suite was created or executed.  Real-competition uncertainty
remains around active-team population size, focal pivotality frequency, public
zero/floor frequency, endogenous path effects, and how representative these
consumed development families are of the unseen room.
"""


def render_manifest(
    *,
    scenarios: Sequence[Pass3Scenario],
    metrics: Sequence[dict[str, object]],
    selection: dict[str, object],
    runtime_seconds: float,
) -> str:
    selected = selection["selected_challenger"]
    parameters = SHADOW_PARAMETERS.get(str(selected)) if selected else None
    exact_parameters = (
        json.dumps(asdict(parameters), sort_keys=True)
        if parameters is not None
        else "none; no shadow candidate passed the screening rule"
    )
    hashes = _hash_sources()
    frozen_parameter_lines = "\n".join(
        f"  - `{name}`: `{json.dumps(asdict(SHADOW_PARAMETERS[name]), sort_keys=True)}`"
        for name in SHADOW_STRATEGY_NAMES
    )
    hash_lines = "\n".join(
        f"| `{path}` | `{digest}` |" for path, digest in hashes.items()
    )
    return f"""# Liferaft Pass 5A Frozen Development Manifest

- selected challenger: **{selected}**
- exact frozen parameters: `{exact_parameters}`
- all frozen shadow candidate parameters:
{frozen_parameter_lines}
- experiment command: `python -m research.liferaft.pass5_experiments`
- development scenarios: {sum(scenario.name.startswith('dev-') for scenario in scenarios)}
- consumed validation scenarios: {sum(not scenario.name.startswith('dev-') for scenario in scenarios)}
- total scenarios: {len(scenarios)}
- candidates per exposure: {len(EXPERIMENT_STRATEGIES)}
- exposure levels: {len(EXPOSURES)}
- result cells: {len(metrics)}
- runtime seconds: {runtime_seconds:.2f}
- screening rule passed: {selected is not None}
- evidence status: development evidence only; not a production acceptance result
- fresh blind Pass 5B suite: not created or executed
- tuning prohibition: no tuning from any previously consumed final result

## Result-affecting source hashes

| source | SHA-256 |
| --- | --- |
{hash_lines}

The protocol was frozen before the full experiment.  Consumed final artifacts
remained quarantined, no final scenario constructor was called, and no
production file or final catalogue was modified.
"""


def regenerate_outputs_from_results() -> None:
    """Refresh report/manifest aggregation from an already completed run."""

    document = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    metrics = document["metrics"]
    scenarios = tuple(development_scenarios() + validation_scenarios())
    selection = _select_challenger(metrics)
    document["selection"] = selection
    document["source_hashes"] = _hash_sources()
    RESULTS_PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        render_report(
            metrics=metrics,
            scenarios=scenarios,
            selection=selection,
            callable_audit=document["callable_exposure_audit"],
            runtime_seconds=float(document["runtime_seconds"]),
        ),
        encoding="utf-8",
    )
    MANIFEST_PATH.write_text(
        render_manifest(
            scenarios=scenarios,
            metrics=metrics,
            selection=selection,
            runtime_seconds=float(document["runtime_seconds"]),
        ),
        encoding="utf-8",
    )


def run_experiment(
    *,
    scenarios: Iterable[Pass3Scenario] | None = None,
    exposures: Sequence[int] = EXPOSURES,
    write_outputs: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    """Run the frozen development screen and optionally write outputs."""

    if tuple(exposures) != EXPOSURES:
        raise ValueError("Pass 5A exposures are frozen at protocol levels")
    scenario_list = tuple(
        development_scenarios() + validation_scenarios()
        if scenarios is None
        else scenarios
    )
    started = time.perf_counter()
    metrics: list[dict[str, object]] = []
    expected = len(scenario_list) * len(EXPOSURES) * len(EXPERIMENT_STRATEGIES)
    completed = 0
    print(
        f"Pass 5A development run: scenarios={len(scenario_list)} cells={expected}",
        flush=True,
    )
    for exposure in EXPOSURES:
        for scenario in scenario_list:
            for strategy_name in EXPERIMENT_STRATEGIES:
                result, strategy, _resolver = run_cell(
                    scenario,
                    strategy_name,
                    float(exposure),
                )
                metrics.append(
                    _base_metric(
                        result,
                        strategy_name,
                        scenario,
                        float(exposure),
                        strategy,
                    )
                )
                completed += 1
                if completed % 500 == 0 or completed == expected:
                    print(
                        f"  completed={completed}/{expected} elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
    runtime_seconds = time.perf_counter() - started
    selection = _select_challenger(metrics)
    audit = callable_exposure_audit(scenario_list[0])
    result_document = {
        "protocol": "PASS5A_PROTOCOL.md",
        "evidence_status": "development-only-consumed-scenarios",
        "scenarios": {
            "development": sum(scenario.name.startswith("dev-") for scenario in scenario_list),
            "validation": sum(not scenario.name.startswith("dev-") for scenario in scenario_list),
            "total": len(scenario_list),
        },
        "strategies": list(EXPERIMENT_STRATEGIES),
        "shadow_candidates": list(SHADOW_STRATEGY_NAMES),
        "comparators": list(COMPARATOR_NAMES),
        "exposures": list(EXPOSURES),
        "cells": len(metrics),
        "runtime_seconds": runtime_seconds,
        "callable_exposure_audit": audit,
        "selection": selection,
        "source_hashes": _hash_sources(),
        "metrics": metrics,
    }
    report = render_report(
        metrics=metrics,
        scenarios=scenario_list,
        selection=selection,
        callable_audit=audit,
        runtime_seconds=runtime_seconds,
    )
    manifest = render_manifest(
        scenarios=scenario_list,
        metrics=metrics,
        selection=selection,
        runtime_seconds=runtime_seconds,
    )
    if write_outputs:
        RESULTS_PATH.write_text(
            json.dumps(result_document, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        REPORT_PATH.write_text(report, encoding="utf-8")
        MANIFEST_PATH.write_text(manifest, encoding="utf-8")
    return metrics, result_document, {"report": report, "manifest": manifest}


def main() -> None:
    metrics, result_document, _rendered = run_experiment()
    selection = result_document["selection"]
    print(
        f"Pass 5A complete: runs={len(metrics)} selected={selection['selected_challenger']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
