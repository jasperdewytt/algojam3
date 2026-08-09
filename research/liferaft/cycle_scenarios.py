"""Small development-only populations for the public-cycle experiment.

These are deliberately diagnostic fixtures, not a competition-probability
model.  They use a short marked period so the detector can be audited quickly;
none of these cases are imported by Pass 3 or by the locked-final catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable

from .cold_start_strategies import make_cold_start_strategy
from .cycle_strategies import PublicCycleDetector
from .simulator import (
    Agent,
    LiferaftConfig,
    LiferaftSimulator,
    MajorityOutcome,
    SimulationResult,
)


CYCLE_TOTAL_DAYS = 140
CYCLE_VOTING_START_DAY = 5
CYCLE_PRE_VOTING_PRICE = 100_000

CyclePopulationFactory = Callable[[], tuple[Agent, ...]]


class DayIndexedScheduleAgent:
    """A fresh, immutable-at-call-time schedule used by cycle fixtures."""

    def __init__(self, name: str, schedule: tuple[int, ...], *, seed: int | None = None) -> None:
        self.name = name
        self.seed = seed
        self._schedule = tuple(schedule)

    @property
    def schedule(self) -> tuple[int, ...]:
        return self._schedule

    def decide(self, observation) -> int:
        return self._schedule[observation.day] if observation.day < len(self._schedule) else 0


class StartupZeroHistoryAgent:
    """Stateful fixture whose startup action depends on pre-voting calls."""

    def __init__(self, name: str, *, live_action: int) -> None:
        self.name = name
        self.live_action = live_action
        self.calls = 0

    def decide(self, observation) -> int:
        if observation.day < observation.marked_boundary_day:
            self.calls += 1
            return 1
        # Observe-and-ignore receives five startup calls; fully-inactive does
        # not. The difference is intentional execution-mode sensitivity.
        return self.live_action if self.calls % 2 else -self.live_action


def _action_for_label(label: MajorityOutcome) -> int:
    return 1 if label is MajorityOutcome.LONG else -1


def _schedule_from_labels(
    labels: tuple[MajorityOutcome, ...],
    *,
    total_days: int = CYCLE_TOTAL_DAYS,
    boundary: int = CYCLE_VOTING_START_DAY,
    phase: int = 0,
) -> tuple[int, ...]:
    schedule = [0] * total_days
    for day in range(boundary, total_days):
        label = labels[(day - boundary + phase) % len(labels)]
        schedule[day] = _action_for_label(label)
    return tuple(schedule)


def _population_from_schedules(
    schedules: tuple[tuple[int, ...], ...],
    *,
    seed: int | None = None,
) -> CyclePopulationFactory:
    def factory() -> tuple[Agent, ...]:
        return tuple(
            DayIndexedScheduleAgent(
                f"opponent-{index}",
                schedule,
                seed=None if seed is None else seed + index,
            )
            for index, schedule in enumerate(schedules)
        )

    return factory


def _unanimous_periodic_population(
    labels: tuple[MajorityOutcome, ...],
    *,
    phase: int,
    count: int = 3,
    seed: int | None = None,
) -> CyclePopulationFactory:
    schedule = _schedule_from_labels(labels, phase=phase)
    return _population_from_schedules(tuple(schedule for _ in range(count)), seed=seed)


def _primitive_pattern(period: int, seed: int) -> tuple[MajorityOutcome, ...]:
    """Create a fixed seeded binary block with no shorter exact block period."""

    # These two blocks are explicit fixture constants: period 13 preserves the
    # requested eight-long/five-short price-return composition, while period 20
    # avoids an accidental local three-block period-2 run.
    if period == 13:
        return (
            MajorityOutcome.LONG,
        ) * 5 + (
            MajorityOutcome.SHORT,
        ) + (
            MajorityOutcome.LONG,
        ) * 3 + (
            MajorityOutcome.SHORT,
        ) * 4
    if period == 20:
        text = "LSLSSSLSLLLSSLLSLLLL"
        return tuple(
            MajorityOutcome.LONG if value == "L" else MajorityOutcome.SHORT
            for value in text
        )

    rng = Random(seed)
    pattern = [
        MajorityOutcome.LONG if rng.random() < 0.5 else MajorityOutcome.SHORT
        for _ in range(period)
    ]
    if len(set(pattern)) == 1:
        pattern[-1] = (
            MajorityOutcome.SHORT
            if pattern[-1] is MajorityOutcome.LONG
            else MajorityOutcome.LONG
        )
    for candidate in range(2, period):
        if all(pattern[index] is pattern[index % candidate] for index in range(period)):
            pattern[-1] = (
                MajorityOutcome.SHORT
                if pattern[-1] is MajorityOutcome.LONG
                else MajorityOutcome.LONG
            )
    return tuple(pattern)


def _random_schedule(seed: int, *, total_days: int = CYCLE_TOTAL_DAYS) -> tuple[int, ...]:
    rng = Random(seed)
    schedule = [0] * CYCLE_TOTAL_DAYS
    for day in range(CYCLE_VOTING_START_DAY, total_days):
        draw = rng.random()
        schedule[day] = 1 if draw < 0.45 else -1 if draw < 0.90 else 0
    return tuple(schedule)


def _markov_schedule(seed: int, *, total_days: int = CYCLE_TOTAL_DAYS) -> tuple[int, ...]:
    rng = Random(seed)
    current = 1 if seed % 2 else -1
    schedule = [0] * total_days
    for day in range(CYCLE_VOTING_START_DAY, total_days):
        if rng.random() < 0.23:
            current = -current
        schedule[day] = current
    return tuple(schedule)


def cycle_config(
    *,
    execution_mode: str = "fully_inactive",
    total_days: int = CYCLE_TOTAL_DAYS,
    voting_start_day: int = CYCLE_VOTING_START_DAY,
    pre_voting_price: int = CYCLE_PRE_VOTING_PRICE,
    price_floor: int = 20_000,
) -> LiferaftConfig:
    return LiferaftConfig(
        total_days=total_days,
        marked_boundary_day=voting_start_day,
        initial_price=pre_voting_price,
        reset_price=pre_voting_price,
        pre_voting_price=pre_voting_price,
        price_floor=price_floor,
        market_mode="inactive_until_marked",
        pre_voting_execution=execution_mode,
        voting_start_day=voting_start_day,
    )


@dataclass(frozen=True)
class CycleScenario:
    name: str
    family: str
    seed: int
    config: LiferaftConfig
    population_factory: CyclePopulationFactory
    population_description: str
    true_cycle: bool
    expected_period: int | None = None
    path_controlled: bool = True
    pivotal: bool = False

    def run(self, strategy_name: str) -> tuple[SimulationResult, object]:
        if strategy_name == "cycle_detector":
            focal = PublicCycleDetector()
        else:
            focal = make_cold_start_strategy(strategy_name)
        opponents = self.population_factory()
        result = LiferaftSimulator(
            (focal, *opponents),
            self.config,
            focal_agent_name=focal.name,
            scenario_name=self.name,
            scenario_configuration={
                "suite": "cycle-development-only",
                "family": self.family,
                "seed": self.seed,
                "true_cycle": self.true_cycle,
                "expected_period": self.expected_period,
                "path_controlled": self.path_controlled,
                "pivotal_fixture": self.pivotal,
            },
            random_seeds={"scenario": self.seed},
        ).run()
        return result, focal


def _scenario(
    name: str,
    family: str,
    seed: int,
    population_factory: CyclePopulationFactory,
    description: str,
    *,
    config: LiferaftConfig | None = None,
    true_cycle: bool,
    expected_period: int | None = None,
    path_controlled: bool = True,
    pivotal: bool = False,
) -> CycleScenario:
    return CycleScenario(
        name=name,
        family=family,
        seed=seed,
        config=config or cycle_config(),
        population_factory=population_factory,
        population_description=description,
        true_cycle=true_cycle,
        expected_period=expected_period,
        path_controlled=path_controlled,
        pivotal=pivotal,
    )


def development_cycle_scenarios() -> tuple[CycleScenario, ...]:
    scenarios: list[CycleScenario] = []

    # Pure fixed periods requested by the frozen development specification.
    for index, period in enumerate((2, 3, 4, 5, 7, 13, 20)):
        pattern = (
            _primitive_pattern(13, 4_000 + period)
            if period == 13
            else _primitive_pattern(period, 4_000 + period)
        )
        scenarios.append(
            _scenario(
                f"pure-period-{period}",
                "true_periodic",
                4_000 + index,
                _unanimous_periodic_population(
                    pattern,
                    phase=(index + 1) % period,
                    seed=4_000 + index,
                ),
                f"three non-pivotal opponents with primitive period {period}",
                true_cycle=True,
                expected_period=period,
            )
        )

    return_cycle_pattern = _primitive_pattern(13, 4_100)
    scenarios.append(
        _scenario(
            "return-cycle-13-movements",
            "price_return_cycle",
            4_100,
            _unanimous_periodic_population(
                return_cycle_pattern,
                phase=3,
                seed=4_100,
            ),
            "13 movements: eight long-majority down moves and five short-majority up moves",
            true_cycle=True,
            expected_period=13,
        )
    )

    # Several periodic components still produce a deterministic aggregate
    # pattern; only the effective majority path is public to the focal agent.
    aggregate = _primitive_pattern(4, 4_200)
    aggregate_schedule = _schedule_from_labels(aggregate, phase=1)
    scenarios.append(
        _scenario(
            "multiple-periodic-components",
            "multiple_periodic",
            4_200,
            _population_from_schedules(
                (
                    aggregate_schedule,
                    aggregate_schedule,
                    aggregate_schedule,
                    _schedule_from_labels((MajorityOutcome.LONG, MajorityOutcome.SHORT), phase=0),
                    _schedule_from_labels((MajorityOutcome.SHORT, MajorityOutcome.LONG), phase=1),
                ),
                seed=4_200,
            ),
            "three period-4 components plus two lower-period components",
            true_cycle=True,
            expected_period=4,
        )
    )

    base = _primitive_pattern(3, 4_300)
    broken = list(base)
    broken[1] = (
        MajorityOutcome.SHORT
        if broken[1] is MajorityOutcome.LONG
        else MajorityOutcome.LONG
    )
    broken_labels = base * 4
    break_offset = 22
    broken_schedule = list(_schedule_from_labels(broken_labels, phase=0))
    for day in range(CYCLE_VOTING_START_DAY, CYCLE_TOTAL_DAYS):
        if day == CYCLE_VOTING_START_DAY + break_offset:
            broken_schedule[day] = -broken_schedule[day]
    scenarios.append(
        _scenario(
            "cycle-break-and-restart",
            "cycle_break_restart",
            4_300,
            _population_from_schedules(
                (tuple(broken_schedule),) * 3,
                seed=4_300,
            ),
            "period-3 path with one corrupted movement and later restart",
            true_cycle=True,
            expected_period=3,
        )
    )

    first_regime = _primitive_pattern(3, 4_400)
    second_regime = _primitive_pattern(5, 4_401)
    switched_labels = first_regime * 4 + second_regime * 20
    scenarios.append(
        _scenario(
            "regime-switch-period-3-to-5",
            "cycle_regime_switch",
            4_400,
            _unanimous_periodic_population(switched_labels, phase=0, seed=4_400),
            "period-3 regime followed by period-5 regime",
            true_cycle=True,
            expected_period=5,
        )
    )

    corrupted_labels = list(_primitive_pattern(5, 4_500) * 20)
    corrupted_labels[31] = (
        MajorityOutcome.SHORT
        if corrupted_labels[31] is MajorityOutcome.LONG
        else MajorityOutcome.LONG
    )
    scenarios.append(
        _scenario(
            "one-corrupted-movement",
            "corrupted_cycle",
            4_500,
            _unanimous_periodic_population(tuple(corrupted_labels), phase=2, seed=4_500),
            "period-5 path with one known-label corruption",
            true_cycle=True,
            expected_period=5,
        )
    )

    scenarios.extend(
        (
            _scenario(
                "control-persistent-long",
                "non_periodic_control",
                4_600,
                _unanimous_periodic_population(
                    (MajorityOutcome.LONG,), phase=0, seed=4_600
                ),
                "constant long-majority control",
                true_cycle=False,
                expected_period=None,
            ),
            _scenario(
                "control-seeded-random",
                "non_periodic_control",
                4_601,
                _population_from_schedules(
                    tuple(_random_schedule(4_601 + index) for index in range(5)),
                    seed=4_601,
                ),
                "five seeded random schedules with occasional flats",
                true_cycle=False,
                path_controlled=True,
            ),
            _scenario(
                "control-markov-like",
                "non_periodic_control",
                4_602,
                _population_from_schedules(
                    tuple(_markov_schedule(4_602 + index) for index in range(5)),
                    seed=4_602,
                ),
                "five seeded low-order Markov-like schedules",
                true_cycle=False,
                path_controlled=True,
            ),
        )
    )

    tie_schedule = _schedule_from_labels(
        (MajorityOutcome.LONG, MajorityOutcome.SHORT), phase=0
    )
    opposite_schedule = tuple(-action for action in tie_schedule)
    scenarios.append(
        _scenario(
            "control-ties-and-zeros",
            "unknown_public_moves",
            4_700,
            _population_from_schedules(
                (tie_schedule, opposite_schedule, tuple(0 for _ in tie_schedule)),
                seed=4_700,
            ),
            "opposite schedules plus flat votes create ties and zero moves",
            true_cycle=False,
        )
    )

    scenarios.append(
        _scenario(
            "control-floor-clipping",
            "floor_clipping",
            4_701,
            _unanimous_periodic_population(
                (MajorityOutcome.LONG,), phase=0, seed=4_701
            ),
            "long-majority votes pinned at the price floor",
            config=cycle_config(pre_voting_price=20_000, price_floor=20_000),
            true_cycle=False,
        )
    )

    scenarios.append(
        _scenario(
            "control-runaway-budget",
            "runaway_budget",
            4_702,
            _unanimous_periodic_population(
                (MajorityOutcome.SHORT,), phase=0, seed=4_702
            ),
            "short-majority price runaway and eventual focal budget flattening",
            true_cycle=False,
        )
    )

    # A one-opponent population makes the focal vote pivotal; the three-agent
    # versions above are clearly non-pivotal for the same public pattern.
    pivotal_pattern = _primitive_pattern(3, 4_800)
    scenarios.append(
        _scenario(
            "pivotal-focal-cycle",
            "pivotality",
            4_800,
            _unanimous_periodic_population(
                pivotal_pattern,
                phase=0,
                count=1,
                seed=4_800,
            ),
            "one opponent; focal cycle action can alter majority or tie",
            true_cycle=True,
            expected_period=3,
            pivotal=True,
        )
    )

    startup_factory = lambda: (
        StartupZeroHistoryAgent("opponent-0", live_action=1),
        StartupZeroHistoryAgent("opponent-1", live_action=-1),
        StartupZeroHistoryAgent("opponent-2", live_action=1),
    )
    for execution_mode in ("observe_and_ignore_actions", "fully_inactive"):
        scenarios.append(
            _scenario(
                f"startup-rule-{execution_mode}",
                "startup_execution_mode",
                4_900,
                startup_factory,
                "opponents react to the number of inactive-period calls",
                config=cycle_config(execution_mode=execution_mode),
                true_cycle=False,
                path_controlled=False,
            )
        )

    return tuple(scenarios)


__all__ = [
    "CYCLE_PRE_VOTING_PRICE",
    "CYCLE_TOTAL_DAYS",
    "CYCLE_VOTING_START_DAY",
    "CycleScenario",
    "DayIndexedScheduleAgent",
    "development_cycle_scenarios",
    "cycle_config",
]
