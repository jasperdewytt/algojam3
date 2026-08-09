"""Simple, stateful Liferaft opponent archetypes for Pass 1 research."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from .simulator import (
    AgentObservation,
    MajorityOutcome,
    _action_for_outcome,
)


@dataclass
class AlwaysFlat:
    name: str = "always-flat"

    def decide(self, observation: AgentObservation) -> int:
        return 0


@dataclass
class AlwaysLong:
    name: str = "always-long"

    def decide(self, observation: AgentObservation) -> int:
        return 1


@dataclass
class AlwaysShort:
    name: str = "always-short"

    def decide(self, observation: AgentObservation) -> int:
        return -1


@dataclass
class SeededRandom:
    """Draw independent, reproducible actions using probability weights."""

    name: str = "seeded-random"
    p_long: float = 1 / 3
    p_short: float = 1 / 3
    p_flat: float = 1 / 3
    seed: int = 0
    _rng: Random = field(init=False, repr=False)
    _long_cutoff: float = field(init=False, repr=False)
    _short_cutoff: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        probabilities = (self.p_long, self.p_short, self.p_flat)
        if any(value < 0 for value in probabilities):
            raise ValueError("SeededRandom probabilities must be non-negative")
        total = sum(probabilities)
        if total <= 0:
            raise ValueError("at least one SeededRandom probability must be positive")
        self._rng = Random(self.seed)
        self._long_cutoff = self.p_long / total
        self._short_cutoff = (self.p_long + self.p_short) / total

    def decide(self, observation: AgentObservation) -> int:
        draw = self._rng.random()
        if draw < self._long_cutoff:
            return 1
        if draw < self._short_cutoff:
            return -1
        return 0


class _PreviousMajorityBase:
    """Keep the last majority that was publicly inferable."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._last_majority: MajorityOutcome | None = None

    def _observe_last_majority(self, observation: AgentObservation) -> None:
        # AgentObservation owns reset filtering so every public caller gets
        # the same interpretation of a reset, including canonical-sized ones.
        inferred = observation.previous_inferred_majority
        if inferred is not None:
            self._last_majority = inferred


class PreviousMajorityPersistenceExploiter(_PreviousMajorityBase):
    """Take the opposite side of the last publicly inferred majority."""

    def __init__(self, name: str = "previous-majority-exploiter") -> None:
        super().__init__(name)

    def decide(self, observation: AgentObservation) -> int:
        self._observe_last_majority(observation)
        if self._last_majority is None:
            return 0
        return _action_for_outcome(self._last_majority, opposite=True)


class PreviousMajorityFollower(_PreviousMajorityBase):
    """Take the same side as the last publicly inferred majority."""

    def __init__(self, name: str = "previous-majority-follower") -> None:
        super().__init__(name)

    def decide(self, observation: AgentObservation) -> int:
        self._observe_last_majority(observation)
        if self._last_majority is None:
            return 0
        return _action_for_outcome(self._last_majority, opposite=False)


@dataclass
class WinStayLoseShift:
    """Stay after a profitable public move; switch after a loss probabilistically."""

    name: str = "win-stay-lose-shift"
    switching_probability: float = 0.5
    seed: int = 0
    initial_action: int = 0
    ignore_reset_jump: bool = True
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.switching_probability <= 1:
            raise ValueError("switching_probability must be in [0, 1]")
        if self.initial_action not in (-1, 0, 1):
            raise ValueError("initial_action must be -1, 0, or 1")
        self._rng = Random(self.seed)

    def decide(self, observation: AgentObservation) -> int:
        prior = observation.own_position
        if prior == 0:
            return self.initial_action
        if observation.previous_price_change is None:
            return prior
        if observation.previous_move_is_reset and self.ignore_reset_jump:
            return prior

        public_pnl = prior * observation.previous_price_change
        if public_pnl > 0:
            return prior
        if public_pnl < 0 and self._rng.random() < self.switching_probability:
            return -prior
        return prior


@dataclass
class PeriodicStrategy:
    """Repeat a configurable position pattern, including periods 2, 3, and 4."""

    name: str = "periodic"
    period: int = 2
    pattern: tuple[int, ...] | None = None
    phase: int = 0
    reset_phase_at_boundary: bool = False

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError("period must be positive")
        if self.pattern is None:
            defaults = {
                2: (1, -1),
                3: (1, -1, 0),
                4: (1, -1, 0, 1),
            }
            self.pattern = defaults.get(
                self.period,
                tuple(1 if index % 2 == 0 else -1 for index in range(self.period)),
            )
        else:
            self.pattern = tuple(self.pattern)
        if len(self.pattern) != self.period:
            raise ValueError("pattern length must equal period")
        if any(type(action) is not int or action not in (-1, 0, 1) for action in self.pattern):
            raise ValueError("periodic pattern actions must be -1, 0, or 1")

    def decide(self, observation: AgentObservation) -> int:
        if (
            self.reset_phase_at_boundary
            and observation.day >= observation.marked_boundary_day
        ):
            # Keep day 0 as the phase origin during calibration. The reset
            # starts a new pattern only at the boundary itself.
            pattern_day = observation.day - observation.marked_boundary_day
        else:
            pattern_day = observation.day
        index = (pattern_day + self.phase) % self.period
        return self.pattern[index]


@dataclass
class LastNMajorityRule:
    """Follow or counter the majority inferred from the last N public moves."""

    name: str = "last-n-majority"
    window: int = 3
    follow: bool = True
    tie_action: int = 0
    # Retained for compatibility; AgentObservation always excludes resets
    # from public inference now, regardless of this legacy setting.
    ignore_reset_jump: bool = True
    _history: list[MajorityOutcome] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.tie_action not in (-1, 0, 1):
            raise ValueError("tie_action must be -1, 0, or 1")
        self._history = []

    def decide(self, observation: AgentObservation) -> int:
        # Public inference always excludes reset jumps. Keep the legacy flag
        # for source compatibility, but do not let it re-enable reset leaks.
        inferred = observation.previous_inferred_majority
        if inferred is not None:
            self._history.append(inferred)

        recent = self._history[-self.window :]
        long_count = sum(item is MajorityOutcome.LONG for item in recent)
        short_count = sum(item is MajorityOutcome.SHORT for item in recent)
        outcome = majority_from_history_counts(long_count, short_count)
        if outcome is None:
            return self.tie_action
        return _action_for_outcome(outcome, opposite=not self.follow)


def majority_from_history_counts(
    long_count: int,
    short_count: int,
) -> MajorityOutcome | None:
    if long_count > short_count:
        return MajorityOutcome.LONG
    if short_count > long_count:
        return MajorityOutcome.SHORT
    return None


@dataclass
class PriceLevelFloorAware:
    """Use a conservative floor action only when the price is near the floor."""

    name: str = "price-level-floor-aware"
    floor_price: int | None = None
    near_floor_buffer: int = 0
    floor_action: int = 1
    above_floor_action: int = 0

    def __post_init__(self) -> None:
        if self.near_floor_buffer < 0:
            raise ValueError("near_floor_buffer must be non-negative")
        if self.floor_action not in (-1, 0, 1):
            raise ValueError("floor_action must be -1, 0, or 1")
        if self.above_floor_action not in (-1, 0, 1):
            raise ValueError("above_floor_action must be -1, 0, or 1")

    def decide(self, observation: AgentObservation) -> int:
        floor = observation.price_floor if self.floor_price is None else self.floor_price
        if observation.price <= floor + self.near_floor_buffer:
            return self.floor_action
        return self.above_floor_action


def _momentum_action(price_change: int | None) -> int:
    if price_change is None:
        return 0
    if price_change > 0:
        return 1
    if price_change < 0:
        return -1
    return 0


@dataclass
class BoundaryUnawareStrategy:
    """Treat the artificial reset jump as an ordinary latest price move."""

    name: str = "boundary-unaware"

    def decide(self, observation: AgentObservation) -> int:
        return _momentum_action(observation.previous_price_change)


@dataclass
class BoundaryAwareStrategy:
    """Use the latest genuine move, skipping the public reset jump."""

    name: str = "boundary-aware"

    def decide(self, observation: AgentObservation) -> int:
        history = observation.price_history
        latest_change_index = len(history) - 1
        if latest_change_index == observation.marked_boundary_day:
            latest_change_index -= 1
        if latest_change_index < 1:
            return 0
        genuine_change = (
            history[latest_change_index] - history[latest_change_index - 1]
        )
        return _momentum_action(genuine_change)


@dataclass
class StrategySwitchingAgent:
    """Delegate to one policy before the marked boundary and another after it."""

    name: str
    before_boundary: object
    after_boundary: object
    boundary_day: int | None = None

    def decide(self, observation: AgentObservation) -> object:
        boundary = (
            observation.marked_boundary_day
            if self.boundary_day is None
            else self.boundary_day
        )
        policy = self.after_boundary if observation.day >= boundary else self.before_boundary
        decide = getattr(policy, "decide", None)
        if not callable(decide):
            raise TypeError("switching policies must expose decide(observation)")
        return decide(observation)


# Friendly aliases keep the vocabulary used in research notes short.
PreviousMajorityExploiter = PreviousMajorityPersistenceExploiter
PreviousMajorityFollowerCounter = PreviousMajorityFollower
WinStayLoseShiftAgent = WinStayLoseShift


__all__ = [
    "AlwaysFlat",
    "AlwaysLong",
    "AlwaysShort",
    "BoundaryAwareStrategy",
    "BoundaryUnawareStrategy",
    "LastNMajorityRule",
    "PeriodicStrategy",
    "PriceLevelFloorAware",
    "PreviousMajorityExploiter",
    "PreviousMajorityFollower",
    "PreviousMajorityFollowerCounter",
    "PreviousMajorityPersistenceExploiter",
    "SeededRandom",
    "StrategySwitchingAgent",
    "WinStayLoseShift",
    "WinStayLoseShiftAgent",
]
