"""Pass 4A's single production-risk wrapper.

This module deliberately does not alter the frozen Pass 3 candidate code.  The
wrapper delegates all prediction work to a pristine ``burnin1_markov``
candidate and adds only the risk controls frozen in ``PASS4A_PROTOCOL.md``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from typing import Callable, TypeAlias

from .cold_start_strategies import make_cold_start_strategy
from .simulator import AgentObservation, MajorityOutcome
from .strategies import Forecast


MAX_LIFERAFT_LOSS_AUD = 50_000
HEALTH_WINDOW = 8
HEALTH_MIN_OBSERVATIONS = 8
HEALTH_MIN_HIT_RATE = 0.40
HEALTH_BAD_STREAK_REQUIRED = 2
PORTFOLIO_RESERVE_AUD = 10_000
GROSS_PORTFOLIO_BUDGET = 600_000

ExposureSource: TypeAlias = float | int | Callable[[AgentObservation], float]


def _voting_start_day(observation: AgentObservation) -> int:
    return (
        observation.voting_start_day
        if observation.voting_start_day is not None
        else observation.marked_boundary_day
    )


def _is_genuine_live_interval(observation: AgentObservation) -> bool:
    """Return true only when the preceding public movement is live and real."""

    return (
        observation.day > _voting_start_day(observation)
        and not observation.previous_move_is_reset
        and not observation.is_reset_day
        and observation.previous_price_change is not None
    )


def _is_genuine_zero(observation: AgentObservation) -> bool:
    return _is_genuine_live_interval(observation) and observation.previous_price_change == 0


def _validate_exposure(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("other portfolio exposure must be numeric, not boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid other portfolio exposure: {value!r}"
        ) from exc
    if not isfinite(numeric) or numeric < 0:
        raise ValueError(
            "other portfolio exposure must be finite and non-negative: "
            f"{value!r}"
        )
    return numeric


@dataclass(frozen=True)
class StopEvent:
    """Causal snapshot when one sticky stop first becomes observable."""

    name: str
    day: int
    pnl_before: int
    pnl_after: int
    loss_limit_overshoot: int


class Risk50Burnin1Markov:
    """``burnin1_markov`` with one frozen, causal production-risk wrapper."""

    name = "risk50_burnin1_markov"

    def __init__(
        self,
        *,
        other_portfolio_exposure: ExposureSource = 0.0,
        name: str = name,
    ) -> None:
        self.name = name
        self.delegate = make_cold_start_strategy("burnin1_markov")
        self.other_portfolio_exposure = other_portfolio_exposure
        if not callable(other_portfolio_exposure):
            _validate_exposure(other_portfolio_exposure)

        self._last_decision_day: int | None = None
        self._last_action = 0
        self._last_raw_action = 0
        self._last_forecast: Forecast | None = None
        self._stored_forecast: Forecast | None = None
        self._last_exposure: float | None = None

        self._cumulative_marked_pnl = 0
        self._realised_increment_by_day: dict[int, int] = {}
        self._loss_stop_active = False
        self._health_stop_active = False
        self._first_stop_trigger: str | None = None
        self._loss_stop_event: StopEvent | None = None
        self._health_stop_event: StopEvent | None = None

        self._quality_hits: deque[bool] = deque(maxlen=HEALTH_WINDOW)
        self._quality_scoreable = 0
        self._quality_last_scored_day: int | None = None
        self._quality_bad_streak = 0

        self._loss_stop_suppressed = 0
        self._health_stop_suppressed = 0
        self._unknown_pause_count = 0
        self._floor_gate_count = 0
        self._headroom_gate_count = 0
        self._raw_nonzero_requests = 0
        self._raw_nonzero_suppressed_by_loss = 0
        self._raw_nonzero_suppressed_by_health = 0
        self._raw_nonzero_suppressed_by_unknown = 0
        self._raw_nonzero_suppressed_by_floor = 0
        self._raw_nonzero_suppressed_by_headroom = 0

    @property
    def cumulative_marked_pnl(self) -> int:
        return self._cumulative_marked_pnl

    @property
    def realised_increment_by_day(self) -> dict[int, int]:
        return dict(self._realised_increment_by_day)

    @property
    def loss_stop_active(self) -> bool:
        return self._loss_stop_active

    @property
    def health_stop_active(self) -> bool:
        return self._health_stop_active

    @property
    def first_stop_trigger(self) -> str | None:
        return self._first_stop_trigger

    @property
    def loss_stop_event(self) -> StopEvent | None:
        return self._loss_stop_event

    @property
    def health_stop_event(self) -> StopEvent | None:
        return self._health_stop_event

    @property
    def quality_history(self) -> tuple[bool, ...]:
        return tuple(self._quality_hits)

    @property
    def quality_scoreable(self) -> int:
        return self._quality_scoreable

    @property
    def last_forecast(self) -> Forecast | None:
        return self._last_forecast

    @property
    def last_action(self) -> int:
        return self._last_action

    @property
    def last_raw_action(self) -> int:
        return self._last_raw_action

    @property
    def unknown_pause_count(self) -> int:
        return self._unknown_pause_count

    @property
    def floor_gate_count(self) -> int:
        return self._floor_gate_count

    @property
    def headroom_gate_count(self) -> int:
        return self._headroom_gate_count

    @property
    def raw_nonzero_requests(self) -> int:
        return self._raw_nonzero_requests

    @property
    def raw_nonzero_suppressed_by_gate(self) -> dict[str, int]:
        return {
            "loss_stop": self._raw_nonzero_suppressed_by_loss,
            "health_stop": self._raw_nonzero_suppressed_by_health,
            "unknown_pause": self._raw_nonzero_suppressed_by_unknown,
            "floor": self._raw_nonzero_suppressed_by_floor,
            "headroom": self._raw_nonzero_suppressed_by_headroom,
        }

    def _resolve_exposure(self, observation: AgentObservation) -> float:
        source = self.other_portfolio_exposure
        value = source(observation) if callable(source) else source
        numeric = _validate_exposure(value)
        self._last_exposure = numeric
        return numeric

    def _record_stop(
        self,
        name: str,
        observation: AgentObservation,
        pnl_before: int,
        pnl_after: int,
    ) -> None:
        overshoot = max(0, -pnl_after - MAX_LIFERAFT_LOSS_AUD)
        event = StopEvent(name, observation.day, pnl_before, pnl_after, overshoot)
        if name == "loss_stop" and self._loss_stop_event is None:
            self._loss_stop_event = event
            self._loss_stop_active = True
        elif name == "health_stop" and self._health_stop_event is None:
            self._health_stop_event = event
            self._health_stop_active = True
        if self._first_stop_trigger is None:
            self._first_stop_trigger = name

    def _account_realised_pnl(self, observation: AgentObservation) -> None:
        if not _is_genuine_live_interval(observation):
            return
        assert observation.previous_price_change is not None
        increment = observation.own_position * observation.previous_price_change
        before = self._cumulative_marked_pnl
        after = before + increment
        self._realised_increment_by_day[observation.day] = increment
        self._cumulative_marked_pnl = after
        if after <= -MAX_LIFERAFT_LOSS_AUD and not self._loss_stop_active:
            self._record_stop("loss_stop", observation, before, after)

    def _score_prior_forecast(self, observation: AgentObservation) -> None:
        if (
            not _is_genuine_live_interval(observation)
            or observation.day == self._quality_last_scored_day
        ):
            return
        self._quality_last_scored_day = observation.day
        label = observation.previous_inferred_majority
        forecast = self._stored_forecast
        if label is None or forecast is None:
            return
        if forecast.support <= 0 or forecast.predicted_majority is None:
            return
        hit = forecast.predicted_majority is label
        self._quality_hits.append(hit)
        self._quality_scoreable += 1
        if len(self._quality_hits) < HEALTH_MIN_OBSERVATIONS:
            return
        hit_rate = sum(self._quality_hits) / len(self._quality_hits)
        if hit_rate < HEALTH_MIN_HIT_RATE:
            self._quality_bad_streak += 1
        else:
            self._quality_bad_streak = 0
        if (
            self._quality_bad_streak >= HEALTH_BAD_STREAK_REQUIRED
            and not self._health_stop_active
        ):
            self._record_stop(
                "health_stop",
                observation,
                self._cumulative_marked_pnl,
                self._cumulative_marked_pnl,
            )

    def _delegate_decision(self, observation: AgentObservation) -> int:
        raw_action = self.delegate.decide(observation)
        if type(raw_action) is not int or raw_action not in (-1, 0, 1):
            raise ValueError(
                "burnin1_markov returned an invalid action: "
                f"{raw_action!r}"
            )
        self._last_raw_action = raw_action
        self._last_forecast = getattr(self.delegate, "last_forecast", None)
        if self._last_forecast is not None and not isinstance(self._last_forecast, Forecast):
            raise TypeError("burnin1_markov exposed an invalid forecast object")
        self._stored_forecast = self._last_forecast
        if raw_action != 0:
            self._raw_nonzero_requests += 1
        return raw_action

    def decide(self, observation: AgentObservation) -> int:
        if self._last_decision_day is not None:
            if observation.day == self._last_decision_day:
                return self._last_action
            if observation.day < self._last_decision_day:
                raise ValueError("Pass 4 wrapper observations must be chronological")

        self._account_realised_pnl(observation)
        self._score_prior_forecast(observation)
        raw_action = self._delegate_decision(observation)

        # Keep the wrapper explicitly flat before live voting even if a future
        # delegate implementation ever changes its inactive-day default.
        action = 0 if observation.day < _voting_start_day(observation) else raw_action
        if observation.day < _voting_start_day(observation):
            pass
        elif self._loss_stop_active:
            if raw_action != 0:
                self._loss_stop_suppressed += 1
                self._raw_nonzero_suppressed_by_loss += 1
            action = 0
        elif self._health_stop_active:
            if raw_action != 0:
                self._health_stop_suppressed += 1
                self._raw_nonzero_suppressed_by_health += 1
            action = 0
        else:
            unknown_pause = _is_genuine_zero(observation)
            if unknown_pause:
                self._unknown_pause_count += 1
                if raw_action != 0:
                    self._raw_nonzero_suppressed_by_unknown += 1
                action = 0
            elif observation.voting_active and observation.price == observation.price_floor:
                self._floor_gate_count += 1
                if raw_action != 0:
                    self._raw_nonzero_suppressed_by_floor += 1
                action = 0
            elif raw_action != 0:
                exposure = self._resolve_exposure(observation)
                permitted = (
                    exposure
                    + abs(raw_action) * observation.price
                    + PORTFOLIO_RESERVE_AUD
                    <= GROSS_PORTFOLIO_BUDGET
                )
                if not permitted:
                    self._headroom_gate_count += 1
                    self._raw_nonzero_suppressed_by_headroom += 1
                    action = 0

        self._last_action = action
        self._last_decision_day = observation.day
        return action

    def diagnostics(self) -> dict[str, object]:
        """Return wrapper-only diagnostics for the Pass 4 report."""

        return {
            "cumulative_marked_pnl": self._cumulative_marked_pnl,
            "loss_stop_active": self._loss_stop_active,
            "health_stop_active": self._health_stop_active,
            "first_stop_trigger": self._first_stop_trigger,
            "loss_stop_event": self._loss_stop_event,
            "health_stop_event": self._health_stop_event,
            "quality_scoreable": self._quality_scoreable,
            "quality_history": tuple(self._quality_hits),
            "unknown_pause_count": self._unknown_pause_count,
            "floor_gate_count": self._floor_gate_count,
            "headroom_gate_count": self._headroom_gate_count,
            "raw_nonzero_requests": self._raw_nonzero_requests,
            "raw_nonzero_suppressed_by_gate": self.raw_nonzero_suppressed_by_gate,
            "realised_increment_by_day": self.realised_increment_by_day,
        }


def make_pass4_strategy(
    name: str,
    *,
    other_portfolio_exposure: ExposureSource = 0.0,
) -> Risk50Burnin1Markov:
    if name != Risk50Burnin1Markov.name:
        raise KeyError(f"unknown Pass 4A strategy {name!r}")
    return Risk50Burnin1Markov(
        other_portfolio_exposure=other_portfolio_exposure,
    )


__all__ = [
    "ExposureSource",
    "GROSS_PORTFOLIO_BUDGET",
    "HEALTH_BAD_STREAK_REQUIRED",
    "HEALTH_MIN_HIT_RATE",
    "HEALTH_MIN_OBSERVATIONS",
    "HEALTH_WINDOW",
    "MAX_LIFERAFT_LOSS_AUD",
    "PORTFOLIO_RESERVE_AUD",
    "Risk50Burnin1Markov",
    "StopEvent",
    "make_pass4_strategy",
]
