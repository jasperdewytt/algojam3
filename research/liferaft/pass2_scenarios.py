"""Development, validation, and held-out Pass 2 scenario suites.

Scenario factories create fresh opponent state for every run.  The suites are
small enough to run in unit tests and are intentionally diverse rather than
tuned to make one candidate look good.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .archetypes import (
    AlwaysFlat,
    AlwaysLong,
    AlwaysShort,
    BoundaryAwareStrategy,
    BoundaryUnawareStrategy,
    LastNMajorityRule,
    PeriodicStrategy,
    PriceLevelFloorAware,
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


StrategyFactory = Callable[[str, LiferaftConfig], Agent]
OpponentFactory = Callable[[], tuple[Agent, ...]]


@dataclass(frozen=True)
class Pass2Scenario:
    """A fresh-opponent scenario with one named focal strategy slot."""

    name: str
    family: str
    config: LiferaftConfig
    opponent_factory: OpponentFactory
    other_portfolio_exposure: ExposureSpec = 0.0
    random_seeds: Mapping[str, int | None] = field(default_factory=dict)
    configuration: Mapping[str, Any] = field(default_factory=dict)
    focal_name: str = "focal"

    def fresh_opponents(self) -> tuple[Agent, ...]:
        opponents = tuple(self.opponent_factory())
        if not opponents:
            raise ValueError("Pass 2 scenarios require at least one opponent")
        names = [agent.name for agent in opponents]
        if self.focal_name in names:
            raise ValueError("opponent name collides with focal agent")
        if len(set(names)) != len(names):
            raise ValueError("opponent names must be unique")
        return opponents

    def run(self, strategy_factory: StrategyFactory) -> SimulationResult:
        opponents = self.fresh_opponents()
        focal = strategy_factory(self.focal_name, self.config)
        if focal.name != self.focal_name:
            raise ValueError("strategy factory must preserve the requested focal name")
        return LiferaftSimulator(
            opponents + (focal,),
            self.config,
            other_portfolio_exposure=self.other_portfolio_exposure,
            focal_agent_name=self.focal_name,
            scenario_name=self.name,
            scenario_configuration={
                "family": self.family,
                **dict(self.configuration),
            },
            random_seeds=self.random_seeds,
        ).run()


def strategy_factory_for_name(strategy_name: str) -> StrategyFactory:
    """Return a factory that configures a Pass 2 strategy from scenario data."""

    from .strategies import strategy_from_name

    def factory(name: str, config: LiferaftConfig) -> Agent:
        return strategy_from_name(
            strategy_name,
            agent_name=name,
            marked_boundary_day=config.marked_boundary_day,
        )

    return factory


def _config(
    *,
    total_days: int = 80,
    boundary: int = 40,
    initial_price: int = 100_000,
    reset_price: int = 100_000,
) -> LiferaftConfig:
    return LiferaftConfig(
        total_days=total_days,
        marked_boundary_day=boundary,
        initial_price=initial_price,
        reset_price=reset_price,
    )


def _persistent(side: str, count: int, prefix: str) -> OpponentFactory:
    if side not in {"long", "short"}:
        raise ValueError("persistent side must be long or short")

    def factory() -> tuple[Agent, ...]:
        cls = AlwaysLong if side == "long" else AlwaysShort
        return tuple(cls(f"{prefix}-{index}") for index in range(count))

    return factory


def _alternating(count: int, period: int, phase: int = 0) -> OpponentFactory:
    def factory() -> tuple[Agent, ...]:
        return tuple(
            PeriodicStrategy(
                name=f"periodic-{index}",
                period=period,
                phase=phase + index,
                reset_phase_at_boundary=False,
            )
            for index in range(count)
        )

    return factory


def _random_mixture(count: int, seed: int) -> OpponentFactory:
    def factory() -> tuple[Agent, ...]:
        return tuple(
            SeededRandom(
                name=f"random-{index}",
                p_long=0.35,
                p_short=0.45,
                p_flat=0.20,
                seed=seed + index,
            )
            for index in range(count)
        )

    return factory


def _reactive_mixture(count: int, seed: int) -> OpponentFactory:
    def factory() -> tuple[Agent, ...]:
        agents: list[Agent] = []
        for index in range(count):
            if index % 4 == 0:
                agents.append(
                    WinStayLoseShift(
                        name=f"win-stay-{index}",
                        switching_probability=0.65,
                        seed=seed + index,
                        initial_action=1 if index % 2 == 0 else -1,
                    )
                )
            elif index % 4 == 1:
                agents.append(PreviousMajorityPersistenceExploiter(f"counter-{index}"))
            elif index % 4 == 2:
                agents.append(PreviousMajorityFollower(f"follower-{index}"))
            else:
                agents.append(AlwaysFlat(f"flat-{index}"))
        return tuple(agents)

    return factory


def _boundary_mix(count: int) -> OpponentFactory:
    def factory() -> tuple[Agent, ...]:
        agents: list[Agent] = []
        for index in range(count):
            if index % 3 == 0:
                agents.append(BoundaryAwareStrategy(f"aware-{index}"))
            elif index % 3 == 1:
                agents.append(BoundaryUnawareStrategy(f"unaware-{index}"))
            else:
                agents.append(
                    StrategySwitchingAgent(
                        f"switcher-{index}",
                        before_boundary=AlwaysFlat(f"switcher-pre-{index}"),
                        after_boundary=AlwaysShort(f"switcher-post-{index}"),
                    )
                )
        return tuple(agents)

    return factory


def _history_population(count: int) -> OpponentFactory:
    """Opponents that use only short rolling public price histories."""

    def factory() -> tuple[Agent, ...]:
        agents: list[Agent] = []
        for index in range(count):
            if index % 2 == 0:
                agents.append(
                    LastNMajorityRule(
                        name=f"history-{index}",
                        window=2 + index % 3,
                        follow=index % 4 == 0,
                        tie_action=0,
                    )
                )
            else:
                agents.append(
                    PriceLevelFloorAware(
                        name=f"floor-aware-{index}",
                        near_floor_buffer=5_000,
                        floor_action=1,
                        above_floor_action=0,
                    )
                )
        return tuple(agents)

    return factory


def _near_tie() -> OpponentFactory:
    def factory() -> tuple[Agent, ...]:
        # The focal strategy is the only possible counter-vote.  A short
        # focal action converts the one-vote long majority into a tie.
        return (AlwaysLong("near-long"), AlwaysFlat("near-flat"))

    return factory


def _regime_change(count: int) -> OpponentFactory:
    def factory() -> tuple[Agent, ...]:
        return tuple(
            StrategySwitchingAgent(
                f"regime-{index}",
                before_boundary=AlwaysLong(f"regime-pre-{index}"),
                after_boundary=AlwaysShort(f"regime-post-{index}"),
            )
            for index in range(count)
        )

    return factory


def _scenario(
    name: str,
    family: str,
    opponent_factory: OpponentFactory,
    *,
    config: LiferaftConfig | None = None,
    configuration: Mapping[str, Any] | None = None,
    random_seeds: Mapping[str, int | None] | None = None,
    other_portfolio_exposure: ExposureSpec = 0.0,
) -> Pass2Scenario:
    return Pass2Scenario(
        name=name,
        family=family,
        config=config or _config(),
        opponent_factory=opponent_factory,
        configuration=dict(configuration or {}),
        random_seeds=dict(random_seeds or {}),
        other_portfolio_exposure=other_portfolio_exposure,
    )


def development_scenarios() -> tuple[Pass2Scenario, ...]:
    """Small deterministic scenarios used for implementation development."""

    return (
        _scenario(
            "dev-persistent-short",
            "persistent-majority",
            _persistent("short", 5, "short"),
            configuration={"seed": None, "purpose": "directional baseline"},
        ),
        _scenario(
            "dev-alternating-period-2",
            "periodic",
            _alternating(5, 2),
            configuration={"period": 2, "phase": 0},
        ),
        _scenario(
            "dev-reactive-mixture",
            "reactive-mixture",
            _reactive_mixture(8, 13),
            random_seeds={"reactive-base": 13},
        ),
        _scenario(
            "dev-near-pivotal",
            "near-balanced-pivotal",
            _near_tie(),
            configuration={"near_tie": True},
        ),
        _scenario(
            "dev-boundary-mix",
            "boundary-reset",
            _boundary_mix(6),
        ),
    )


def validation_scenarios() -> tuple[Pass2Scenario, ...]:
    """Seeded mixtures used for limited predeclared rule validation."""

    return (
        _scenario(
            "val-random-101",
            "random-mixture",
            _random_mixture(9, 101),
            random_seeds={"population": 101},
        ),
        _scenario(
            "val-random-211",
            "random-mixture",
            _random_mixture(9, 211),
            random_seeds={"population": 211},
        ),
        _scenario(
            "val-period-3",
            "periodic",
            _alternating(7, 3, phase=1),
            configuration={"period": 3, "phase": 1},
        ),
        _scenario(
            "val-regime-change",
            "regime-change",
            _regime_change(5),
        ),
        _scenario(
            "val-history-opponents",
            "rolling-public-history",
            _history_population(7),
            configuration={"history_windows": (2, 3, 4)},
        ),
        _scenario(
            "val-runaway-budget",
            "runaway-budget",
            _persistent("short", 5, "runaway"),
            config=_config(total_days=100, boundary=20),
            configuration={"expected_budget_pressure": True},
        ),
        _scenario(
            "val-floor-lock",
            "floor-clipping",
            _persistent("long", 5, "floor"),
            config=_config(total_days=80, boundary=40, initial_price=20_000),
            configuration={"expected_floor_lock": True},
        ),
    )


def held_out_scenarios() -> tuple[Pass2Scenario, ...]:
    """Untouched final scenarios with new seeds and compositions.

    This function contains no strategy-specific branch.  The experiment
    runner evaluates it once after development/validation summaries are
    produced.
    """

    return (
        _scenario(
            "heldout-random-907",
            "random-mixture",
            _random_mixture(11, 907),
            random_seeds={"population": 907},
        ),
        _scenario(
            "heldout-period-4",
            "periodic",
            _alternating(9, 4, phase=2),
            configuration={"period": 4, "phase": 2},
        ),
        _scenario(
            "heldout-reactive-1201",
            "reactive-mixture",
            _reactive_mixture(11, 1201),
            random_seeds={"reactive-base": 1201},
        ),
        _scenario(
            "heldout-regime-change",
            "regime-change",
            _regime_change(7),
        ),
        _scenario(
            "heldout-near-pivotal",
            "near-balanced-pivotal",
            _near_tie(),
            configuration={"near_tie": True, "held_out": True},
        ),
        _scenario(
            "heldout-runaway-budget",
            "runaway-budget",
            _persistent("short", 7, "heldout-runaway"),
            config=_config(total_days=105, boundary=25),
            configuration={"expected_budget_pressure": True, "held_out": True},
        ),
    )


def all_suites() -> dict[str, tuple[Pass2Scenario, ...]]:
    return {
        "development": development_scenarios(),
        "validation": validation_scenarios(),
        "held_out": held_out_scenarios(),
    }


__all__ = [
    "Pass2Scenario",
    "StrategyFactory",
    "all_suites",
    "development_scenarios",
    "held_out_scenarios",
    "strategy_factory_for_name",
    "validation_scenarios",
]
