"""Competition-correct cold-start scenario suites for Pass 3.

Development cases are correctness/design fixtures. Validation cases are new,
paired, seeded populations with a realistic 730/365 timeline. The locked
final definitions are separate and are only instantiated after an explicit
``--final`` request by the experiment runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable

from .archetypes import (
    AlwaysFlat,
    AlwaysLong,
    AlwaysShort,
    BoundaryAwareStrategy,
    BoundaryUnawareStrategy,
    PeriodicStrategy,
    PreviousMajorityFollower,
    PreviousMajorityPersistenceExploiter,
    PriceLevelFloorAware,
    SeededRandom,
    WinStayLoseShift,
)
from .cold_start_strategies import make_cold_start_strategy
from .simulator import Agent, LiferaftConfig, LiferaftSimulator, SimulationResult


TOTAL_DAYS = 730
MARKED_BOUNDARY_DAY = 365
PRE_VOTING_PRICE = 100_000
POPULATION_SIZES = (3, 4, 5, 8, 9, 15)
VALIDATION_OFFSETS = tuple(range(20))
DRIFT_STRENGTHS = (0.08, 0.10, 0.12, 0.14, 0.16, 0.18)
VALIDATION_FAMILIES = (
    "persistent_long",
    "persistent_short",
    "balanced_random",
    "short_biased_random",
    "long_biased_random",
    "periodic",
    "reactive_mixture",
    "regime_change",
    "gradual_drift",
    "startup_zero_history",
    "history_rules",
    "margin_mixture",
)
FINAL_FAMILIES = (
    "symmetric_random",
    "short_biased_random",
    "reactive_mixture",
    "periodic",
    "regime_change",
    "gradual_drift",
    "startup_zero_history",
    "margin_mixture",
)
FINAL_SEED_BASE = 90_000

PopulationFactory = Callable[[], tuple[Agent, ...]]


class DayIndexedRandomAgent:
    """A seeded schedule whose RNG is consumed at construction, not by calls."""

    def __init__(
        self,
        name: str,
        *,
        seed: int,
        p_long: float,
        p_short: float,
        total_days: int = TOTAL_DAYS,
    ) -> None:
        if min(p_long, p_short) < 0 or p_long + p_short > 1:
            raise ValueError("random probabilities must be non-negative and sum <= 1")
        self.name = name
        self.seed = seed
        rng = Random(seed)
        p_flat = 1.0 - p_long - p_short
        schedule: list[int] = []
        for _ in range(total_days):
            draw = rng.random()
            if draw < p_long:
                schedule.append(1)
            elif draw < p_long + p_short:
                schedule.append(-1)
            else:
                schedule.append(0)
        self._schedule = tuple(schedule)

    def decide(self, observation) -> int:
        return self._schedule[observation.day] if observation.day < len(self._schedule) else 0


class NoisyPersistenceAgent:
    """Mostly persistent side with seeded, day-indexed noise."""

    def __init__(
        self,
        name: str,
        *,
        seed: int,
        base_action: int,
        noise_probability: float,
        event_day: int | None = None,
        event_action: int = 0,
        total_days: int = TOTAL_DAYS,
    ) -> None:
        if base_action not in (-1, 1):
            raise ValueError("base_action must be -1 or 1")
        if not 0 <= noise_probability <= 1:
            raise ValueError("noise_probability must be in [0, 1]")
        if event_action not in (-1, 0, 1):
            raise ValueError("event_action must be -1, 0, or 1")
        self.name = name
        self.seed = seed
        rng = Random(seed)
        schedule: list[int] = []
        for day in range(total_days):
            if event_day is not None and day == event_day:
                schedule.append(event_action)
                continue
            if rng.random() >= noise_probability:
                schedule.append(base_action)
            else:
                schedule.append((-base_action, 0)[rng.randrange(2)])
        self._schedule = tuple(schedule)

    def decide(self, observation) -> int:
        day = observation.day
        return self._schedule[day] if day < len(self._schedule) else 0


class DayIndexedDriftAgent:
    """A reproducible, symmetric probability drift with no call-count dependence."""

    def __init__(
        self,
        name: str,
        *,
        seed: int,
        direction: int,
        strength: float,
        base_probability: float = 0.375,
        total_days: int = TOTAL_DAYS,
        boundary: int = MARKED_BOUNDARY_DAY,
    ) -> None:
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        if not 0 <= strength <= 0.35:
            raise ValueError("strength must be in [0, 0.35]")
        if not 0 < base_probability < 1:
            raise ValueError("base_probability must be in (0, 1)")
        if base_probability + strength > 1 or base_probability - strength < 0:
            raise ValueError("base probability and strength are incompatible")
        self.name = name
        self.seed = seed
        self.direction = direction
        self.strength = strength
        self.base_probability = base_probability
        rng = Random(seed)
        schedule: list[int] = []
        for day in range(total_days):
            progress = min(1.0, max(0, day - boundary) / 180.0)
            # Equal early long/short probabilities make opposite directions
            # true whole-population mirrors instead of opposite labels layered
            # on top of a fixed bias.
            p_long = base_probability + direction * strength * progress
            p_short = base_probability - direction * strength * progress
            draw = rng.random()
            if draw < p_long:
                schedule.append(1)
            elif draw < p_long + p_short:
                schedule.append(-1)
            else:
                schedule.append(0)
        self._schedule = tuple(schedule)

    def decide(self, observation) -> int:
        day = observation.day
        return self._schedule[day] if day < len(self._schedule) else 0


class DayIndexedRegimeAgent:
    """A seeded pre/post regime with noise and a fixed switch date."""

    def __init__(
        self,
        name: str,
        *,
        seed: int,
        switch_day: int,
        pre_long_probability: float,
        post_long_probability: float,
        noise_probability: float,
        total_days: int = TOTAL_DAYS,
    ) -> None:
        self.name = name
        self.seed = seed
        self.switch_day = switch_day
        if not 0 <= switch_day < total_days:
            raise ValueError("switch_day must be within the simulation")
        if not 0 <= pre_long_probability <= 1 or not 0 <= post_long_probability <= 1:
            raise ValueError("regime probabilities must be in [0, 1]")
        if not 0 <= noise_probability <= 1:
            raise ValueError("noise_probability must be in [0, 1]")
        rng = Random(seed)
        schedule: list[int] = []
        for day in range(total_days):
            long_probability = (
                pre_long_probability if day < switch_day else post_long_probability
            )
            if rng.random() < noise_probability:
                schedule.append(rng.choice((-1, 0, 1)))
            else:
                schedule.append(1 if rng.random() < long_probability else -1)
        self._schedule = tuple(schedule)

    def decide(self, observation) -> int:
        day = observation.day
        return self._schedule[day] if day < len(self._schedule) else 0


class HistoryPulseAgent:
    """Preserve a history policy while adding one seed-indexed audit pulse."""

    def __init__(self, delegate: Agent, *, event_day: int) -> None:
        self.name = delegate.name
        self._delegate = delegate
        self._event_day = event_day

    def decide(self, observation) -> int:
        action = self._delegate.decide(observation)
        return 0 if observation.day == self._event_day else action


class StartupZeroHistoryAgent:
    """Agent whose startup state exposes the execution-mode ambiguity."""

    def __init__(
        self,
        name: str,
        *,
        live_action: int = 1,
        seed: int = 0,
        total_days: int = TOTAL_DAYS,
    ) -> None:
        self.name = name
        self.live_action = live_action
        self.calls = 0
        rng = Random(seed)
        self._warm_schedule = tuple(rng.choice((-1, 0, 1)) for _ in range(total_days))
        self._cold_schedule = tuple(rng.choice((-1, 0, 1)) for _ in range(total_days))

    def decide(self, observation) -> int:
        self.calls += 1
        if observation.day < observation.marked_boundary_day:
            return 0
        # Observe mode reaches this branch with 365 prior calls; fully
        # inactive mode reaches it cold on the first voting-start call.
        if self.calls > 100:
            return self._warm_schedule[observation.day]
        if observation.day == observation.marked_boundary_day:
            return -self.live_action
        return self._cold_schedule[observation.day]


@dataclass(frozen=True)
class Pass3Scenario:
    """A reproducible focal-agent evaluation case."""

    name: str
    family: str
    config: LiferaftConfig
    population_factory: PopulationFactory
    seed: int
    execution_mode: str
    population_description: str
    population_size: int = 0
    pair_id: str | None = None
    path_controlled: bool = False
    drift_direction: int | None = None
    drift_strength: float | None = None

    def run(
        self,
        strategy_name: str,
        *,
        other_portfolio_exposure: float = 0.0,
    ) -> SimulationResult:
        focal = make_cold_start_strategy(strategy_name)
        opponents = self.population_factory()
        agents = (focal, *opponents)
        return LiferaftSimulator(
            agents,
            self.config,
            focal_agent_name=focal.name,
            scenario_name=self.name,
            scenario_configuration={
                "suite": "pass3",
                "family": self.family,
                "seed": self.seed,
                "execution_mode": self.execution_mode,
                "population": self.population_description,
                "population_size": self.population_size,
                "pair_id": self.pair_id,
                "path_controlled": self.path_controlled,
                "drift_direction": self.drift_direction,
                "drift_strength": self.drift_strength,
                "other_portfolio_exposure": other_portfolio_exposure,
                "market_mode": self.config.market_mode,
                "voting_start_day": self.config.voting_start_day,
            },
            random_seeds={"scenario": self.seed},
            # The exposure mapping controls the focal portfolio only, but this
            # is still an endogenous game counterfactual: flattening the focal
            # vote can change the majority and price, which can change future
            # reactive-opponent actions and their budget feasibility.
            other_portfolio_exposure={focal.name: other_portfolio_exposure},
        ).run()

    def live_path_signature(self, result: SimulationResult) -> tuple[tuple[int, ...], tuple[str, ...]]:
        start = result.config.voting_start_day
        assert start is not None
        return (
            tuple(result.price_path[start:]),
            tuple(day.majority.value for day in result.days[start:]),
        )


def cold_start_config(
    *,
    execution_mode: str = "observe_and_ignore_actions",
    total_days: int = TOTAL_DAYS,
    marked_boundary_day: int = MARKED_BOUNDARY_DAY,
    pre_voting_price: int = PRE_VOTING_PRICE,
    price_floor: int = 20_000,
) -> LiferaftConfig:
    """Build the clarified default configuration explicitly."""

    return LiferaftConfig(
        total_days=total_days,
        marked_boundary_day=marked_boundary_day,
        initial_price=pre_voting_price,
        reset_price=pre_voting_price,
        pre_voting_price=pre_voting_price,
        price_floor=price_floor,
        market_mode="inactive_until_marked",
        pre_voting_execution=execution_mode,
        voting_start_day=marked_boundary_day,
    )


def _population_size(seed: int, family_index: int = 0) -> int:
    return POPULATION_SIZES[(seed + family_index) % len(POPULATION_SIZES)]


def _fixed_population(action: int, count: int = 9) -> PopulationFactory:
    def factory() -> tuple[Agent, ...]:
        cls = {1: AlwaysLong, -1: AlwaysShort, 0: AlwaysFlat}[action]
        return tuple(cls(f"opponent-{index}") for index in range(count))

    return factory


def _day_random_population(
    seed: int,
    *,
    p_long: float,
    p_short: float,
    count: int,
) -> PopulationFactory:
    def factory() -> tuple[Agent, ...]:
        return tuple(
            DayIndexedRandomAgent(
                name=f"opponent-{index}",
                seed=seed + 1009 * (index + 1),
                p_long=p_long,
                p_short=p_short,
            )
            for index in range(count)
        )

    return factory


def _persistent_population(
    seed: int,
    *,
    base_action: int,
    count: int,
    noise_probability: float,
) -> PopulationFactory:
    def factory() -> tuple[Agent, ...]:
        if count <= 0:
            return ()
        # Keep one genuinely random but mostly aligned component. The other
        # agents remain persistent, so this is still a persistent family and
        # path variation comes from seeded schedules rather than pulses.
        agents: list[Agent] = [
            DayIndexedRandomAgent(
                "opponent-0",
                seed=seed + 1701,
                p_long=0.70 if base_action > 0 else 0.14,
                p_short=0.14 if base_action > 0 else 0.70,
            )
        ]
        for index in range(1, count):
            heterogeneous_noise = min(
                0.35,
                noise_probability + 0.01 * ((seed + index) % 3),
            )
            agents.append(
                NoisyPersistenceAgent(
                    name=f"opponent-{index}",
                    seed=seed + 701 * (index + 1),
                    base_action=base_action,
                    noise_probability=heterogeneous_noise,
                )
            )
        return tuple(agents)

    return factory


def _periodic_population(seed: int, count: int) -> PopulationFactory:
    def factory() -> tuple[Agent, ...]:
        rng = Random(seed)
        agents: list[Agent] = []
        for index in range(count):
            period = rng.choice((2, 3, 4))
            pattern = tuple(rng.choice((-1, 0, 1)) for _ in range(period))
            if all(action == 0 for action in pattern):
                pattern = (1,) + pattern[1:]
            agents.append(
                PeriodicStrategy(
                    name=f"opponent-{index}",
                    period=period,
                    pattern=pattern,
                    phase=rng.randrange(period),
                    reset_phase_at_boundary=bool(rng.randrange(2)),
                )
            )
        return tuple(agents)

    return factory


def _reactive_population(seed: int, count: int) -> PopulationFactory:
    def factory() -> tuple[Agent, ...]:
        rng = Random(seed)
        agents: list[Agent] = []
        for index in range(count):
            kind = rng.randrange(5)
            name = f"opponent-{index}"
            if kind == 0:
                agents.append(PreviousMajorityFollower(name))
            elif kind == 1:
                agents.append(PreviousMajorityPersistenceExploiter(name))
            elif kind == 2:
                agents.append(
                    WinStayLoseShift(
                        name,
                        switching_probability=0.20 + 0.05 * rng.randrange(6),
                        seed=seed + index * 17,
                        initial_action=rng.choice((-1, 1)),
                    )
                )
            elif kind == 3:
                agents.append(
                    SeededRandom(
                        name=name,
                        seed=seed + index * 31,
                        p_long=0.20 + 0.10 * rng.randrange(5),
                        p_short=0.20 + 0.10 * rng.randrange(5),
                        p_flat=0.20,
                    )
                )
            else:
                agents.append(rng.choice((AlwaysLong(name), AlwaysShort(name), AlwaysFlat(name))))
        return tuple(agents)

    return factory


def _regime_population(seed: int, *, switch_day: int, count: int) -> PopulationFactory:
    def factory() -> tuple[Agent, ...]:
        rng = Random(seed)
        agents: list[Agent] = []
        for index in range(count):
            pre = 0.25 + 0.10 * rng.randrange(6)
            post = 0.25 + 0.10 * rng.randrange(6)
            agents.append(
                DayIndexedRegimeAgent(
                    name=f"opponent-{index}",
                    seed=seed + 37 * (index + 1),
                    switch_day=switch_day,
                    pre_long_probability=pre,
                    post_long_probability=post,
                    noise_probability=0.05 + 0.02 * rng.randrange(5),
                )
            )
        return tuple(agents)

    return factory


def _drift_population(
    seed: int,
    *,
    direction: int,
    strength: float,
    count: int,
) -> PopulationFactory:
    def factory() -> tuple[Agent, ...]:
        return tuple(
            DayIndexedDriftAgent(
                name=f"opponent-{index}",
                seed=seed + 53 * (index + 1),
                direction=direction,
                strength=strength * (0.85 + 0.05 * ((seed + index) % 4)),
                base_probability=0.36 + 0.005 * ((seed + 3 * index) % 4),
            )
            for index in range(count)
        )

    return factory


def _startup_population(seed: int, count: int) -> PopulationFactory:
    def factory() -> tuple[Agent, ...]:
        return tuple(
            StartupZeroHistoryAgent(
                f"opponent-{index}",
                live_action=1 if (seed + 3 * index) % 4 < 2 else -1,
                seed=seed + 211 * (index + 1),
            )
            for index in range(count)
        )

    return factory


def _history_population(seed: int, count: int) -> PopulationFactory:
    def factory() -> tuple[Agent, ...]:
        rng = Random(seed)
        agents: list[Agent] = []
        for index in range(count):
            kind = rng.randrange(4)
            name = f"opponent-{index}"
            if kind == 0:
                agents.append(BoundaryAwareStrategy(name))
            elif kind == 1:
                agents.append(BoundaryUnawareStrategy(name))
            elif kind == 2:
                agents.append(PriceLevelFloorAware(name, near_floor_buffer=10_000))
            else:
                agents.append(PeriodicStrategy(name, period=rng.choice((2, 3, 4))))
        for index in range(min(2, len(agents))):
            # Day-indexed components make aggregate paths vary by seed even
            # when the public-history rules settle into the same regime.
            agents[index] = DayIndexedRandomAgent(
                f"opponent-{index}",
                seed=seed + 991 + index * 17,
                p_long=0.35,
                p_short=0.35,
            )
        event_day = 370 + (seed % 20)
        return tuple(
            HistoryPulseAgent(agent, event_day=event_day) for agent in agents
        )

    return factory


def _margin_population(seed: int, count: int) -> PopulationFactory:
    def factory() -> tuple[Agent, ...]:
        rng = Random(seed)
        long_count = rng.randrange(count + 1)
        short_count = rng.randrange(count - long_count + 1)
        flat_count = count - long_count - short_count
        agents: list[Agent] = []
        agents.extend(
            NoisyPersistenceAgent(
                f"long-{index}",
                seed=seed + 101 * (index + 1),
                base_action=1,
                noise_probability=0.20,
                event_day=380 + (seed % 20),
                event_action=0,
            )
            for index in range(long_count)
        )
        agents.extend(
            NoisyPersistenceAgent(
                f"short-{index}",
                seed=seed + 211 * (index + 1),
                base_action=-1,
                noise_probability=0.20,
                event_day=380 + (seed % 20),
                event_action=0,
            )
            for index in range(short_count)
        )
        agents.extend(AlwaysFlat(f"flat-{index}") for index in range(flat_count))
        return tuple(agents)

    return factory


def _near_tie_population(*, long_count: int, short_count: int) -> PopulationFactory:
    def factory() -> tuple[Agent, ...]:
        agents: list[Agent] = []
        agents.extend(AlwaysLong(f"long-{index}") for index in range(long_count))
        agents.extend(AlwaysShort(f"short-{index}") for index in range(short_count))
        return tuple(agents)

    return factory


def _scenario(
    *,
    name: str,
    family: str,
    seed: int,
    population_factory: PopulationFactory,
    population_description: str,
    execution_mode: str,
    population_size: int,
    pair_id: str | None = None,
    path_controlled: bool = False,
    drift_direction: int | None = None,
    drift_strength: float | None = None,
    config: LiferaftConfig | None = None,
) -> Pass3Scenario:
    return Pass3Scenario(
        name=name,
        family=family,
        config=config or cold_start_config(execution_mode=execution_mode),
        population_factory=population_factory,
        seed=seed,
        execution_mode=execution_mode,
        population_description=population_description,
        population_size=population_size,
        pair_id=pair_id,
        path_controlled=path_controlled,
        drift_direction=drift_direction,
        drift_strength=drift_strength,
    )


def _paired(
    *,
    base_name: str,
    family: str,
    seed: int,
    population_factory: PopulationFactory,
    population_description: str,
    population_size: int,
    path_controlled: bool,
    drift_direction: int | None = None,
    drift_strength: float | None = None,
    config_builder: Callable[..., LiferaftConfig] = cold_start_config,
) -> tuple[Pass3Scenario, Pass3Scenario]:
    pair_id = base_name
    return tuple(
        _scenario(
            name=f"{base_name}-{mode}",
            family=family,
            seed=seed,
            population_factory=population_factory,
            population_description=population_description,
            execution_mode=mode,
            population_size=population_size,
            pair_id=pair_id,
            path_controlled=path_controlled,
            drift_direction=drift_direction,
            drift_strength=drift_strength,
            config=config_builder(execution_mode=mode),
        )
        for mode in ("observe_and_ignore_actions", "fully_inactive")
    )  # type: ignore[return-value]


def development_scenarios() -> tuple[Pass3Scenario, ...]:
    """Small correctness/design and separate deterministic stress cases."""

    return (
        _scenario(
            name="dev-persistent-long",
            family="persistent_long",
            seed=10,
            population_factory=_fixed_population(1),
            population_description="nine always-long opponents",
            execution_mode="observe_and_ignore_actions",
            population_size=9,
        ),
        _scenario(
            name="dev-persistent-short",
            family="persistent_short",
            seed=11,
            population_factory=_fixed_population(-1),
            population_description="nine always-short opponents",
            execution_mode="fully_inactive",
            population_size=9,
        ),
        _scenario(
            name="dev-balanced-pivotal",
            family="near_tie_pivotal",
            seed=12,
            population_factory=_near_tie_population(long_count=4, short_count=4),
            population_description="equal fixed sides; focal action is pivotal",
            execution_mode="observe_and_ignore_actions",
            population_size=8,
        ),
        _scenario(
            name="dev-comfortable-nonpivotal",
            family="comfortable_nonpivotal",
            seed=13,
            population_factory=_near_tie_population(long_count=7, short_count=1),
            population_description="comfortable long majority; focal non-pivotal",
            execution_mode="fully_inactive",
            population_size=8,
        ),
        _scenario(
            name="dev-startup-state",
            family="startup_zero_history",
            seed=14,
            population_factory=_startup_population(14, 9),
            population_description="agents react to pre-voting call history",
            execution_mode="observe_and_ignore_actions",
            population_size=9,
        ),
        _scenario(
            name="dev-startup-cold",
            family="startup_zero_history",
            seed=14,
            population_factory=_startup_population(14, 9),
            population_description="same startup rule with no pre-voting calls",
            execution_mode="fully_inactive",
            population_size=9,
        ),
        _scenario(
            name="dev-floor-clipping",
            family="floor_clipping",
            seed=16,
            population_factory=_fixed_population(1),
            population_description="nine always-long opponents at the floor",
            execution_mode="observe_and_ignore_actions",
            population_size=9,
            config=cold_start_config(
                execution_mode="observe_and_ignore_actions",
                pre_voting_price=20_000,
                price_floor=20_000,
            ),
        ),
        _scenario(
            name="dev-runaway-budget",
            family="runaway_budget",
            seed=17,
            population_factory=_fixed_population(-1),
            population_description="nine always-short opponents; price exceeds budget",
            execution_mode="observe_and_ignore_actions",
            population_size=9,
        ),
        _scenario(
            name="dev-all-flat-no-trade",
            family="no_trade_stress",
            seed=18,
            population_factory=_fixed_population(0, 8),
            population_description="all-flat population with no live movement",
            execution_mode="observe_and_ignore_actions",
            population_size=8,
        ),
    )


def validation_scenarios() -> tuple[Pass3Scenario, ...]:
    """Build paired, consumed validation cases with 20 base seeds per family."""

    scenarios: list[Pass3Scenario] = []
    for family_index, family in enumerate(VALIDATION_FAMILIES):
        for offset in VALIDATION_OFFSETS:
            seed = 20_000 + family_index * 1_000 + offset
            count = _population_size(seed, family_index)
            drift_direction: int | None = None
            drift_strength: float | None = None
            if family == "persistent_long":
                population = _persistent_population(
                    seed,
                    base_action=1,
                    count=count,
                    noise_probability=0.18 + 0.02 * (offset % 8),
                )
                description = f"{count} mostly-long opponents with seeded noise"
                controlled = True
            elif family == "persistent_short":
                population = _persistent_population(
                    seed,
                    base_action=-1,
                    count=count,
                    noise_probability=0.18 + 0.02 * (offset % 8),
                )
                description = f"{count} mostly-short opponents with seeded noise"
                controlled = True
            elif family == "balanced_random":
                population = _day_random_population(
                    seed,
                    p_long=0.40,
                    p_short=0.40,
                    count=count,
                )
                description = f"{count} day-indexed balanced random opponents"
                controlled = True
            elif family == "short_biased_random":
                population = _day_random_population(
                    seed,
                    p_long=0.25,
                    p_short=0.60,
                    count=count,
                )
                description = f"{count} day-indexed short-biased opponents"
                controlled = True
            elif family == "long_biased_random":
                population = _day_random_population(
                    seed,
                    p_long=0.60,
                    p_short=0.25,
                    count=count,
                )
                description = f"{count} day-indexed long-biased opponents"
                controlled = True
            elif family == "periodic":
                population = _periodic_population(seed, count)
                description = f"{count} seeded period-2/3/4 opponents with varied patterns"
                controlled = True
            elif family == "reactive_mixture":
                population = _reactive_population(seed, count)
                description = f"{count} varied followers/counters/fixed/random/reactive opponents"
                controlled = False
            elif family == "regime_change":
                switch_day = 375 + ((seed * 7) % 220)
                population = _regime_population(
                    seed,
                    switch_day=switch_day,
                    count=count,
                )
                description = f"{count} seeded agents with varied pre/post regimes, switch={switch_day}"
                controlled = True
            elif family == "gradual_drift":
                # Directions are independent of the population-size cycle, so
                # both drift signs appear for odd and even populations.
                direction = 1 if offset % 4 in (0, 1) else -1
                strength = DRIFT_STRENGTHS[offset % len(DRIFT_STRENGTHS)]
                drift_direction = direction
                drift_strength = strength
                population = _drift_population(
                    seed,
                    direction=direction,
                    strength=strength,
                    count=count,
                )
                description = f"{count} day-indexed gradual drift agents, direction={direction}, strength={strength:.2f}"
                controlled = True
            elif family == "startup_zero_history":
                population = _startup_population(seed, count)
                description = f"{count} agents reacting to 365 flat-history calls"
                controlled = False
            elif family == "history_rules":
                population = _history_population(seed, count)
                description = f"{count} boundary-aware/unaware and price-history opponents"
                controlled = True
            else:
                population = _margin_population(seed, count)
                description = f"{count} seeded fixed-margin opponents including ties"
                controlled = True
            scenarios.extend(
                _paired(
                    base_name=f"val-{family}-{offset:02d}",
                    family=family,
                    seed=seed,
                    population_factory=population,
                    population_description=description,
                    population_size=count,
                    path_controlled=controlled,
                    drift_direction=drift_direction,
                    drift_strength=drift_strength,
                )
            )
    return tuple(scenarios)


def final_scenarios() -> tuple[Pass3Scenario, ...]:
    """Construct locked final cases only after an explicit final request."""

    scenarios: list[Pass3Scenario] = []
    for family_index, family in enumerate(FINAL_FAMILIES):
        for offset in range(10):
            seed = FINAL_SEED_BASE + family_index * 1_000 + offset
            count = POPULATION_SIZES[(seed + 2 * family_index) % len(POPULATION_SIZES)]
            if family == "symmetric_random":
                population = _day_random_population(seed, p_long=0.35, p_short=0.35, count=count)
                description = f"locked {count}-agent symmetric random composition"
                controlled = True
            elif family == "short_biased_random":
                population = _day_random_population(seed, p_long=0.20, p_short=0.65, count=count)
                description = f"locked {count}-agent short-biased composition"
                controlled = True
            elif family == "reactive_mixture":
                population = _reactive_population(seed + 101, count)
                description = f"locked {count}-agent unseen reactive composition"
                controlled = False
            elif family == "periodic":
                population = _periodic_population(seed + 303, count)
                description = f"locked {count}-agent unseen periodic composition"
                controlled = True
            elif family == "regime_change":
                population = _regime_population(seed, switch_day=390 + (offset * 29) % 250, count=count)
                description = f"locked {count}-agent unseen regime composition"
                controlled = True
            elif family == "gradual_drift":
                direction = -1 if offset % 2 else 1
                population = _drift_population(
                    seed,
                    direction=direction,
                    strength=0.12,
                    count=count,
                )
                description = f"locked {count}-agent unseen drift composition"
                controlled = True
            elif family == "startup_zero_history":
                population = _startup_population(seed, count)
                description = f"locked {count}-agent startup-mode composition"
                controlled = False
            else:
                population = _margin_population(seed, count)
                description = f"locked {count}-agent pivotal-margin composition"
                controlled = True
            scenarios.extend(
                _paired(
                    base_name=f"final-locked-{family}-{offset:02d}",
                    family=family,
                    seed=seed,
                    population_factory=population,
                    population_description=description,
                    population_size=count,
                    path_controlled=controlled,
                    drift_direction=direction if family == "gradual_drift" else None,
                    drift_strength=0.12 if family == "gradual_drift" else None,
                )
            )
    return tuple(scenarios)


__all__ = [
    "FINAL_FAMILIES",
    "FINAL_SEED_BASE",
    "DRIFT_STRENGTHS",
    "MARKED_BOUNDARY_DAY",
    "POPULATION_SIZES",
    "PRE_VOTING_PRICE",
    "TOTAL_DAYS",
    "VALIDATION_FAMILIES",
    "VALIDATION_OFFSETS",
    "Pass3Scenario",
    "cold_start_config",
    "development_scenarios",
    "final_scenarios",
    "validation_scenarios",
]
