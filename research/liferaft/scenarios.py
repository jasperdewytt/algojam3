"""Population and market-path helpers for Liferaft research experiments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, Mapping

from .archetypes import (
    AlwaysFlat,
    AlwaysLong,
    AlwaysShort,
    BoundaryAwareStrategy,
    BoundaryUnawareStrategy,
    PeriodicStrategy,
    PreviousMajorityFollower,
    PreviousMajorityPersistenceExploiter,
    SeededRandom,
    StrategySwitchingAgent,
    WinStayLoseShift,
)
from .simulator import (
    Agent,
    ExposureSpec,
    LiferaftConfig,
    LiferaftSimulator,
    SimulationResult,
)


AgentFactory = Callable[[str], Agent]
AgentPopulationFactory = Callable[[], tuple[Agent, ...]]


@dataclass(frozen=True)
class Scenario:
    """A named population, mechanics configuration, and optional exposure.

    ``agent_factory`` is expected to return brand-new agents on every call.
    When a caller supplies only ``agents``, a pristine deep copy is captured
    at construction time and copied for each run; the agents used by a prior
    run are never used as the source for a later run.
    """

    name: str
    agents: tuple[Agent, ...]
    config: LiferaftConfig
    other_portfolio_exposure: ExposureSpec = 0.0
    focal_agent_name: str | None = None
    random_seeds: Mapping[str, int | None] = field(default_factory=dict)
    configuration: Mapping[str, Any] = field(default_factory=dict)
    agent_factory: AgentPopulationFactory | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _pristine_agents: tuple[Agent, ...] | None = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        agents = tuple(self.agents)
        if not agents and self.agent_factory is None:
            raise ValueError("a scenario must contain at least one agent")
        object.__setattr__(self, "agents", agents)
        object.__setattr__(self, "_pristine_agents", deepcopy(agents))

    def fresh_agents(self) -> tuple[Agent, ...]:
        """Return agents that have never been used by a previous run."""

        if self.agent_factory is not None:
            agents = tuple(self.agent_factory())
        else:
            if self._pristine_agents is None:
                raise RuntimeError("scenario has no pristine agent population")
            agents = deepcopy(self._pristine_agents)
        if not agents:
            raise ValueError("scenario agent factory returned no agents")
        return agents

    def run(self) -> SimulationResult:
        return LiferaftSimulator(
            self.fresh_agents(),
            self.config,
            other_portfolio_exposure=self.other_portfolio_exposure,
            focal_agent_name=self.focal_agent_name,
            scenario_name=self.name,
            scenario_configuration=self.configuration,
            random_seeds=self.random_seeds,
        ).run()


def _require_count(count: int) -> None:
    if type(count) is not int or count < 0:
        raise ValueError("population count must be a non-negative integer")


def homogeneous_population(
    factory: AgentFactory,
    count: int,
    *,
    prefix: str = "agent",
) -> tuple[Agent, ...]:
    """Create a population from one archetype factory."""

    _require_count(count)
    return tuple(factory(f"{prefix}-{index}") for index in range(count))


def configurable_mixture(
    components: Mapping[str, tuple[AgentFactory, int]],
) -> tuple[Agent, ...]:
    """Create an insertion-ordered, explicitly sized mixture of factories."""

    population: list[Agent] = []
    for label, (factory, count) in components.items():
        _require_count(count)
        for index in range(count):
            population.append(factory(f"{label}-{index}"))
    return tuple(population)


def sampled_population_mixture(
    factories: Mapping[str, AgentFactory],
    count: int,
    *,
    weights: Mapping[str, float] | None = None,
    seed: int = 0,
    prefix: str = "sampled",
) -> tuple[Agent, ...]:
    """Sample archetype labels with a fixed seed, then instantiate agents."""

    _require_count(count)
    if not factories:
        raise ValueError("at least one factory is required")
    labels = tuple(factories)
    if weights is None:
        label_weights = [1.0] * len(labels)
    else:
        label_weights = [float(weights.get(label, 0.0)) for label in labels]
    if any(weight < 0 for weight in label_weights) or sum(label_weights) <= 0:
        raise ValueError("sample weights must be non-negative and not all zero")

    rng = Random(seed)
    selected = rng.choices(labels, weights=label_weights, k=count)
    return tuple(
        factories[label](f"{prefix}-{index}-{label}")
        for index, label in enumerate(selected)
    )


def stateful_population(
    count: int = 6,
    *,
    seed: int = 0,
    switching_probability: float = 0.5,
    prefix: str = "stateful",
) -> tuple[Agent, ...]:
    """Create independent stateful win-stay/lose-shift opponents."""

    _require_count(count)
    return tuple(
        WinStayLoseShift(
            name=f"{prefix}-{index}",
            switching_probability=switching_probability,
            seed=seed + index,
            initial_action=1 if index % 2 == 0 else -1,
        )
        for index in range(count)
    )


def boundary_switching_population(
    count: int = 4,
    *,
    prefix: str = "switching",
) -> tuple[Agent, ...]:
    """Create agents that change from flat to long at the marked boundary."""

    _require_count(count)
    return tuple(
        StrategySwitchingAgent(
            name=f"{prefix}-{index}",
            before_boundary=AlwaysFlat(name=f"{prefix}-{index}-pre"),
            after_boundary=AlwaysLong(name=f"{prefix}-{index}-post"),
        )
        for index in range(count)
    )


def stateful_vs_boundary_switching_population(
    *,
    stateful_count: int = 4,
    switching_count: int = 2,
    seed: int = 0,
) -> tuple[Agent, ...]:
    """Combine stateful reactive agents with explicit boundary switchers."""

    return stateful_population(
        stateful_count,
        seed=seed,
        prefix="stateful",
    ) + boundary_switching_population(
        switching_count,
        prefix="switching",
    )


def near_balanced_population(
    *,
    long_count: int = 1,
    short_count: int = 0,
    flat_count: int = 0,
    focal_action: int = -1,
    focal_name: str = "focal",
) -> tuple[Agent, ...]:
    """Build a near-tie population with a named focal vote.

    The defaults give one long vote before the focal short vote, so the focal
    agent converts a one-vote long majority into a tie.
    """

    _require_count(long_count)
    _require_count(short_count)
    _require_count(flat_count)
    if focal_action not in (-1, 0, 1):
        raise ValueError("focal_action must be -1, 0, or 1")

    population: list[Agent] = [
        AlwaysLong(name=f"long-{index}") for index in range(long_count)
    ]
    population.extend(
        AlwaysShort(name=f"short-{index}") for index in range(short_count)
    )
    population.extend(
        AlwaysFlat(name=f"flat-{index}") for index in range(flat_count)
    )
    if focal_action > 0:
        population.append(AlwaysLong(name=focal_name))
    elif focal_action < 0:
        population.append(AlwaysShort(name=focal_name))
    else:
        population.append(AlwaysFlat(name=focal_name))
    return tuple(population)


def deterministic_mixed_scenario(
    *,
    total_days: int = 12,
    marked_boundary_day: int = 6,
) -> Scenario:
    """A fixed mixture useful for smoke tests and demonstrations."""

    def agent_factory() -> tuple[Agent, ...]:
        return (
            AlwaysLong("long-0"),
            AlwaysShort("short-0"),
            AlwaysFlat("flat-0"),
            PreviousMajorityPersistenceExploiter("exploiter"),
            PreviousMajorityFollower("follower"),
            PeriodicStrategy("periodic-3", period=3),
        )

    config = LiferaftConfig(
        total_days=total_days,
        marked_boundary_day=marked_boundary_day,
    )
    return Scenario(
        name="deterministic-mixed",
        agents=agent_factory(),
        config=config,
        agent_factory=agent_factory,
        configuration={
            "population": "long/short/flat plus public-history archetypes",
            "total_days": total_days,
            "marked_boundary_day": marked_boundary_day,
        },
    )


def stateful_reset_scenario(
    *,
    total_days: int = 10,
    marked_boundary_day: int = 5,
) -> Scenario:
    """A compact stateful population whose state crosses the reset."""

    def agent_factory() -> tuple[Agent, ...]:
        return (
            WinStayLoseShift(
                "win-stay",
                switching_probability=1.0,
                seed=11,
                initial_action=1,
            ),
            PreviousMajorityPersistenceExploiter("exploiter"),
            PreviousMajorityFollower("follower"),
            BoundaryAwareStrategy("boundary-aware"),
            BoundaryUnawareStrategy("boundary-unaware"),
            StrategySwitchingAgent(
                "switcher",
                before_boundary=AlwaysFlat("switcher-pre"),
                after_boundary=AlwaysLong("switcher-post"),
            ),
        )

    config = LiferaftConfig(
        total_days=total_days,
        marked_boundary_day=marked_boundary_day,
    )
    return Scenario(
        name="stateful-across-reset",
        agents=agent_factory(),
        config=config,
        agent_factory=agent_factory,
        random_seeds={"win-stay": 11},
        configuration={
            "population": "stateful reactive plus boundary-aware/unaware and switcher",
            "total_days": total_days,
            "marked_boundary_day": marked_boundary_day,
        },
    )


def floor_lock_scenario(*, total_days: int = 8) -> Scenario:
    """All-long votes starting at the floor, with a late harmless reset."""

    boundary = total_days - 1
    config = LiferaftConfig(
        total_days=total_days,
        marked_boundary_day=boundary,
        initial_price=20_000,
    )

    def agent_factory() -> tuple[Agent, ...]:
        return homogeneous_population(AlwaysLong, 5, prefix="long")

    return Scenario(
        name="floor-lock",
        agents=agent_factory(),
        config=config,
        agent_factory=agent_factory,
        configuration={
            "population": "five always-long agents",
            "initial_price": 20_000,
            "marked_boundary_day": boundary,
        },
    )


def runaway_price_scenario(*, total_days: int = 80) -> Scenario:
    """All-short votes that run above $600,000 until budget rejection stops them."""

    boundary = total_days - 1
    config = LiferaftConfig(
        total_days=total_days,
        marked_boundary_day=boundary,
    )

    def agent_factory() -> tuple[Agent, ...]:
        return homogeneous_population(AlwaysShort, 3, prefix="short")

    return Scenario(
        name="runaway-short-majority",
        agents=agent_factory(),
        config=config,
        agent_factory=agent_factory,
        configuration={
            "population": "three always-short agents",
            "boundary_held_late": boundary,
            "other_portfolio_exposure": 0,
        },
    )


def seeded_random_scenario(
    *,
    total_days: int = 20,
    marked_boundary_day: int = 10,
    count: int = 4,
    seed: int = 101,
) -> Scenario:
    """A repeatable scenario containing fresh seeded-random agents."""

    _require_count(count)

    def agent_factory() -> tuple[Agent, ...]:
        return tuple(
            SeededRandom(
                name=f"random-{index}",
                p_long=0.25,
                p_short=0.5,
                p_flat=0.25,
                seed=seed + index,
            )
            for index in range(count)
        )

    config = LiferaftConfig(
        total_days=total_days,
        marked_boundary_day=marked_boundary_day,
    )
    return Scenario(
        name="seeded-random",
        agents=agent_factory(),
        config=config,
        agent_factory=agent_factory,
        configuration={
            "population": "seeded random",
            "count": count,
            "seed": seed,
            "total_days": total_days,
            "marked_boundary_day": marked_boundary_day,
        },
    )


__all__ = [
    "Scenario",
    "boundary_switching_population",
    "configurable_mixture",
    "deterministic_mixed_scenario",
    "floor_lock_scenario",
    "homogeneous_population",
    "near_balanced_population",
    "runaway_price_scenario",
    "sampled_population_mixture",
    "seeded_random_scenario",
    "stateful_population",
    "stateful_reset_scenario",
    "stateful_vs_boundary_switching_population",
]
