"""Development-only causal detector for deterministic public-price cycles.

This module is deliberately outside the Pass 3 candidate catalogue.  It is a
small mechanics experiment: the detector sees only ``AgentObservation`` and
consumes at most one newly observable live interval per decision.  It does not
receive vote counts, opponent actions, pivotality, or hidden engine records.
"""

from __future__ import annotations

from dataclasses import dataclass

from .simulator import (
    AgentObservation,
    MajorityOutcome,
    infer_majority_from_price_change,
)
from .strategies import Forecast, PublicLabel, payoff_action


CYCLE_CANDIDATE_PERIODS: tuple[int, ...] = tuple(range(2, 21))
CYCLE_REPEAT_BLOCKS = 3
CYCLE_MIN_EXPECTED_PNL = 1_000.0
CYCLE_MIN_CONFIDENCE = 0.10


def _voting_start_day(observation: AgentObservation) -> int:
    return (
        observation.voting_start_day
        if observation.voting_start_day is not None
        else observation.marked_boundary_day
    )


def _is_live_interval_observation(observation: AgentObservation) -> bool:
    """Accept a newly observable live interval, including a reset row.

    Reset rows are not genuine majority labels, but they must reach the state
    machine so an active detector can be cleared exactly once.
    """

    # Continuous-reset mode exposes the reset row on the boundary itself,
    # while inactive-until-marked mode has no reset row.  Let the explicit
    # reset flag through even when its day equals the voting start.
    return observation.previous_move_is_reset or (
        observation.day > _voting_start_day(observation)
    )


def _is_floor_clipped(observation: AgentObservation) -> bool:
    """Use public prices to reject a move whose downward magnitude was clipped."""

    return (
        observation.previous_price is not None
        and observation.previous_price + observation.long_majority_move
        < observation.price_floor
        and observation.price == observation.price_floor
        and (observation.previous_price_change or 0) < 0
    )


def _public_live_label(observation: AgentObservation) -> PublicLabel:
    """Infer a cycle label without treating reset, zero, or clipping as labels."""

    if _is_floor_clipped(observation):
        return None
    return infer_majority_from_price_change(
        observation.previous_price_change,
        long_majority_move=observation.long_majority_move,
        short_majority_move=observation.short_majority_move,
        previous_move_is_reset=observation.previous_move_is_reset,
    )


@dataclass(frozen=True)
class CycleDetection:
    """One immutable activation event for experiment reporting."""

    period: int
    pattern: tuple[MajorityOutcome, ...]
    observed_live_count: int
    day: int


class PublicCycleDetector:
    """Fixed-parameter online cycle detector.

    A cycle is activated only after the latest ``3 * period`` known labels are
    three identical consecutive blocks.  The shortest qualifying period is
    selected.  Unknown, zero, reset, and publicly identifiable clipped moves
    clear the context.  A known outcome that contradicts the stored forecast
    also clears the active detector before the next action is selected.
    """

    name = "development-cycle-detector"

    def __init__(self, name: str = name) -> None:
        self.name = name
        self._labels: list[MajorityOutcome] = []
        self._observed_live_count = 0
        self._last_observed_day: int | None = None
        self._active_pattern: tuple[MajorityOutcome, ...] | None = None
        self._next_pattern_index = 0
        self._expected_label: MajorityOutcome | None = None
        self._last_forecast = Forecast(0.5, 0.5, 0, "cycle-inactive")
        self._last_action = 0
        self._cycle_breaks = 0
        self._detections: list[CycleDetection] = []
        self._forecast_by_day: dict[int, MajorityOutcome | None] = {}

    @property
    def labels(self) -> tuple[MajorityOutcome, ...]:
        """The current uninterrupted known-label context."""

        return tuple(self._labels)

    @property
    def observed_live_count(self) -> int:
        return self._observed_live_count

    @property
    def last_forecast(self) -> Forecast:
        return self._last_forecast

    @property
    def last_action(self) -> int:
        return self._last_action

    @property
    def cycle_active(self) -> bool:
        return self._active_pattern is not None

    @property
    def detected_period(self) -> int | None:
        return len(self._active_pattern) if self._active_pattern else None

    @property
    def cycle_breaks(self) -> int:
        return self._cycle_breaks

    @property
    def detections(self) -> tuple[CycleDetection, ...]:
        return tuple(self._detections)

    @property
    def detection_count(self) -> int:
        return len(self._detections)

    @property
    def reactivation_count(self) -> int:
        return max(0, len(self._detections) - 1)

    @property
    def detection_delays(self) -> tuple[int, ...]:
        return tuple(event.observed_live_count for event in self._detections)

    @property
    def forecast_by_day(self) -> dict[int, MajorityOutcome | None]:
        """Publicly forecast labels for secondary, causal diagnostics."""

        return dict(self._forecast_by_day)

    def _deactivate(self) -> None:
        self._active_pattern = None
        self._next_pattern_index = 0
        self._expected_label = None

    def _observe_interval(
        self,
        observation: AgentObservation,
    ) -> tuple[bool, PublicLabel, bool]:
        """Return (new interval, label, contradiction) exactly once per day."""

        if (
            not _is_live_interval_observation(observation)
            or observation.day == self._last_observed_day
        ):
            return False, None, False

        self._last_observed_day = observation.day
        if observation.previous_move_is_reset:
            # A reset is observable state, not a genuine live label. It still
            # has to clear an active detector before this day's action is made.
            if self.cycle_active:
                self._cycle_breaks += 1
            self._labels.clear()
            self._deactivate()
            return True, None, False

        self._observed_live_count += 1
        label = _public_live_label(observation)
        if label is None:
            if self.cycle_active:
                self._cycle_breaks += 1
            self._labels.clear()
            self._deactivate()
            return True, None, False

        if self.cycle_active:
            if label is not self._expected_label:
                self._cycle_breaks += 1
                self._deactivate()
                # The contradictory observation is a valid first label for a
                # future context; it is not used to reactivate on this call.
                self._labels[:] = [label]
                return True, label, True
            self._labels.append(label)
            assert self._active_pattern is not None
            self._next_pattern_index = (
                self._next_pattern_index + 1
            ) % len(self._active_pattern)
            self._expected_label = self._active_pattern[self._next_pattern_index]
            return True, label, False

        self._labels.append(label)
        return True, label, False

    def _detect_shortest_cycle(self) -> tuple[MajorityOutcome, ...] | None:
        for period in CYCLE_CANDIDATE_PERIODS:
            required = CYCLE_REPEAT_BLOCKS * period
            if len(self._labels) < required:
                continue
            recent = tuple(self._labels[-required:])
            first = recent[:period]
            if all(recent[offset : offset + period] == first for offset in (period, 2 * period)):
                return first
        return None

    def _activate_if_qualified(self, observation: AgentObservation) -> None:
        pattern = self._detect_shortest_cycle()
        if pattern is None:
            return
        self._active_pattern = pattern
        self._next_pattern_index = 0
        self._expected_label = pattern[0]
        self._detections.append(
            CycleDetection(
                period=len(pattern),
                pattern=pattern,
                observed_live_count=self._observed_live_count,
                day=observation.day,
            )
        )

    def _choose_action(self, observation: AgentObservation) -> int:
        label = self._expected_label if self.cycle_active else None
        if label is None:
            forecast = Forecast(0.5, 0.5, 0, "cycle-inactive")
        elif label is MajorityOutcome.LONG:
            forecast = Forecast(
                1.0,
                0.0,
                len(self._labels),
                "cycle-detector",
            )
        else:
            forecast = Forecast(
                0.0,
                1.0,
                len(self._labels),
                "cycle-detector",
            )
        self._last_forecast = forecast
        action = payoff_action(
            forecast,
            long_majority_move=observation.long_majority_move,
            short_majority_move=observation.short_majority_move,
            min_expected_pnl=CYCLE_MIN_EXPECTED_PNL,
            min_confidence=CYCLE_MIN_CONFIDENCE,
        )
        self._last_action = action
        self._forecast_by_day[observation.day] = label
        return action

    def decide(self, observation: AgentObservation) -> int:
        if observation.day < _voting_start_day(observation):
            self._last_forecast = Forecast(0.5, 0.5, 0, "cycle-inactive")
            self._last_action = 0
            self._forecast_by_day[observation.day] = None
            return 0

        _new_interval, _label, contradiction = self._observe_interval(observation)
        if contradiction:
            self._last_forecast = Forecast(0.5, 0.5, 0, "cycle-break")
            self._last_action = 0
            self._forecast_by_day[observation.day] = None
            return 0
        if not self.cycle_active and _label is not None:
            self._activate_if_qualified(observation)
        return self._choose_action(observation)


__all__ = [
    "CYCLE_CANDIDATE_PERIODS",
    "CYCLE_MIN_CONFIDENCE",
    "CYCLE_MIN_EXPECTED_PNL",
    "CYCLE_REPEAT_BLOCKS",
    "CycleDetection",
    "PublicCycleDetector",
]
