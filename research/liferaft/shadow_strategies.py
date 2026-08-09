"""Causal shadow-validated Markov strategies for Liferaft Pass 5A.

The strategy in this module is deliberately independent of the existing Pass
4 wrapper.  It consumes only ``AgentObservation`` and keeps a causal paper
ledger alongside the real-position state machine so that every activation and
deactivation decision can be audited after a simulator run.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Callable, TypeAlias

from .simulator import AgentObservation, MajorityOutcome
from .strategies import Forecast, RegularisedMarkovModel, payoff_action


MAX_LIFERAFT_LOSS_AUD = 50_000
PORTFOLIO_RESERVE_AUD = 10_000
GROSS_PORTFOLIO_BUDGET = 600_000
SHADOW_HEALTH_WINDOW = 12
MIN_SCOREABLE_VIRTUAL_TRADES = 6
INITIAL_VIRTUAL_PNL_MINIMUM = 10_000
RECENT_VIRTUAL_PNL_MINIMUM = 5_000
DEACTIVATION_VIRTUAL_PNL_LIMIT = -10_000
HEALTH_BAD_STREAK_REQUIRED = 2
COOLDOWN_GENUINE_OBSERVATIONS = 5
MIN_EXPECTED_PNL = 1_000.0
MIN_CONFIDENCE = 0.10

ExposureSource: TypeAlias = float | int | Callable[[AgentObservation], float]


@dataclass(frozen=True)
class ShadowParameters:
    """All result-affecting parameters for one predeclared candidate."""

    candidate_name: str
    minimum_genuine_nonzero_observations: int
    markov_order: int = 2
    alpha: float = 1.0
    minimum_scoreable_virtual_trades: int = MIN_SCOREABLE_VIRTUAL_TRADES
    shadow_health_window: int = SHADOW_HEALTH_WINDOW
    minimum_initial_virtual_pnl: int = INITIAL_VIRTUAL_PNL_MINIMUM
    minimum_recent_virtual_pnl: int = RECENT_VIRTUAL_PNL_MINIMUM
    deactivation_virtual_pnl_limit: int = DEACTIVATION_VIRTUAL_PNL_LIMIT
    health_bad_streak_required: int = HEALTH_BAD_STREAK_REQUIRED
    cooldown_genuine_observations: int = COOLDOWN_GENUINE_OBSERVATIONS
    minimum_expected_pnl: float = MIN_EXPECTED_PNL
    minimum_confidence: float = MIN_CONFIDENCE
    maximum_actual_loss: int = MAX_LIFERAFT_LOSS_AUD
    portfolio_reserve: int = PORTFOLIO_RESERVE_AUD

    def __post_init__(self) -> None:
        if self.minimum_genuine_nonzero_observations <= 0:
            raise ValueError("shadow warm-up must be positive")
        if self.markov_order != 2:
            raise ValueError("Pass 5A uses Markov order 2")
        if self.alpha != 1.0:
            raise ValueError("Pass 5A uses alpha 1.0")
        if self.minimum_scoreable_virtual_trades <= 0:
            raise ValueError("minimum scoreable trades must be positive")
        if self.shadow_health_window <= 0:
            raise ValueError("shadow health window must be positive")
        if self.health_bad_streak_required <= 0:
            raise ValueError("health bad streak must be positive")
        if self.cooldown_genuine_observations <= 0:
            raise ValueError("cooldown must be positive")


SHADOW_PARAMETERS: dict[str, ShadowParameters] = {
    "shadow8_markov": ShadowParameters(
        "shadow8_markov", minimum_genuine_nonzero_observations=8
    ),
    "shadow12_markov": ShadowParameters(
        "shadow12_markov", minimum_genuine_nonzero_observations=12
    ),
    "shadow20_markov": ShadowParameters(
        "shadow20_markov", minimum_genuine_nonzero_observations=20
    ),
}
SHADOW_STRATEGY_NAMES: tuple[str, ...] = tuple(SHADOW_PARAMETERS)


@dataclass(frozen=True)
class StopEvent:
    """Causal snapshot of the sticky actual loss stop."""

    name: str
    day: int
    pnl_before: int
    pnl_after: int
    loss_limit_overshoot: int


@dataclass(frozen=True)
class ShadowDecision:
    """One causal strategy decision and the evidence available at that time."""

    day: int
    live_decision: bool
    observed_live_interval: bool
    movement_kind: str
    observed_price_change: int | None
    observed_label: str | None
    actual_position: int
    actual_pnl_increment: int
    actual_cumulative_pnl: int
    prior_virtual_action: int
    raw_virtual_interval_pnl: int
    virtual_interval_pnl: int
    virtual_trade_scoreable: bool
    cumulative_virtual_pnl: int
    recent_virtual_pnl: int
    scoreable_virtual_trades: int
    genuine_observations: int
    genuine_nonzero_observations: int
    markov_context: tuple[str, ...]
    markov_context_support: int
    forecast_p_long: float
    forecast_p_short: float
    forecast_p_unknown: float
    forecast_confidence: float
    forecast_source: str
    virtual_action: int
    qualification_state: str
    qualification_streak: int
    real_active: bool
    real_action: int
    activation_day: int | None
    activation_pending: bool
    cooldown_active: bool
    cooldown_observations: int
    deactivated_this_day: bool
    deactivation_reason: str | None
    pause_active: bool
    pause_reason: str | None
    loss_stop_active: bool
    loss_stop_overshoot: int | None
    current_edge_gate: bool
    floor_gate: bool
    unknown_gate: bool
    headroom_gate: bool
    exposure: float | None


def _voting_start_day(observation: AgentObservation) -> int:
    return (
        observation.voting_start_day
        if observation.voting_start_day is not None
        else observation.marked_boundary_day
    )


def _is_live_decision(observation: AgentObservation) -> bool:
    """Return whether a current decision can select a live interval."""

    return observation.day >= _voting_start_day(observation) and (
        observation.voting_active
        or observation.market_mode == "continuous_reset"
    )


def _is_live_interval(observation: AgentObservation) -> bool:
    """Return whether the preceding movement is a public live interval."""

    start = _voting_start_day(observation)
    return (
        observation.day > start
        and observation.previous_price is not None
        and observation.previous_price_change is not None
        and not observation.previous_move_is_reset
        and not observation.is_reset_day
        and (
            observation.voting_active
            or observation.market_mode == "continuous_reset"
        )
    )


def _is_floor_clipped(observation: AgentObservation) -> bool:
    """Infer a floor clip from public prices without hidden simulator fields."""

    change = observation.previous_price_change
    previous = observation.previous_price
    if change is None or previous is None:
        return False
    return (
        change < 0
        and observation.price == observation.price_floor
        and previous + observation.long_majority_move < observation.price_floor
    )


def _movement_kind(observation: AgentObservation) -> str:
    if observation.previous_move_is_reset or observation.is_reset_day:
        return "reset"
    if not _is_live_interval(observation):
        return "inactive_or_startup"
    change = observation.previous_price_change
    if change is None:
        return "unknown"
    if change == 0:
        return "unknown_zero"
    if _is_floor_clipped(observation):
        return "floor_clipped"
    return "genuine_nonzero"


def _validate_exposure(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("other portfolio exposure must be numeric, not boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid other portfolio exposure: {value!r}") from exc
    if not isfinite(numeric) or numeric < 0:
        raise ValueError(
            "other portfolio exposure must be finite and non-negative: "
            f"{value!r}"
        )
    return numeric


class ShadowValidatedMarkov:
    """Causal order-two Markov paper validation with real-position hysteresis."""

    def __init__(
        self,
        *,
        parameters: ShadowParameters | None = None,
        candidate_name: str = "shadow12_markov",
        name: str = "focal",
        other_portfolio_exposure: ExposureSource = 0.0,
    ) -> None:
        if parameters is None:
            try:
                parameters = SHADOW_PARAMETERS[candidate_name]
            except KeyError as exc:
                raise KeyError(f"unknown Pass 5A shadow candidate {candidate_name!r}") from exc
        if parameters.candidate_name != candidate_name:
            raise ValueError("candidate_name and parameters do not agree")
        self.parameters = parameters
        self.candidate_name = candidate_name
        self.name = name
        self.other_portfolio_exposure = other_portfolio_exposure
        if not callable(other_portfolio_exposure):
            _validate_exposure(other_portfolio_exposure)

        self._model = RegularisedMarkovModel(
            order=parameters.markov_order,
            alpha=parameters.alpha,
        )
        self._labels: list[MajorityOutcome | None] = []
        self._last_observed_day: int | None = None
        self._last_decision_day: int | None = None
        self._last_action = 0
        self._virtual_action = 0
        self._last_forecast = Forecast(0.5, 0.5, 0, "uninitialised")

        self._observed_live_intervals = 0
        self._genuine_observations = 0
        self._genuine_nonzero_observations = 0
        self._unknown_observations = 0
        self._zero_observations = 0
        self._clipped_observations = 0
        self._reset_observations = 0

        self._virtual_trade_pnls: list[int] = []
        self._virtual_pnl_window: deque[int] = deque(
            maxlen=parameters.shadow_health_window
        )
        self._virtual_cumulative_pnl = 0
        self._scoreable_virtual_trades = 0
        self._qualification_streak = 0
        self._health_bad_streak = 0
        self._health_evaluation_count = 0
        self._last_scoreable_day: int | None = None

        self._real_active = False
        self._activation_pending = False
        self._activation_day: int | None = None
        self._activation_events: list[dict[str, object]] = []
        self._reactivation_count = 0
        self._ever_activated = False
        self._cooldown_active = False
        self._cooldown_observations = 0
        self._deactivation_events: list[dict[str, object]] = []
        self._deactivated_this_day = False
        self._deactivation_reason_today: str | None = None

        self._actual_cumulative_pnl = 0
        self._realised_increment_by_day: dict[int, int] = {}
        self._loss_stop_active = False
        self._loss_stop_event: StopEvent | None = None

        self._pause_events: list[dict[str, object]] = []
        self._pause_active_today = False
        self._pause_reason_today: str | None = None
        self._current_edge_gate_count = 0
        self._floor_gate_count = 0
        self._unknown_gate_count = 0
        self._headroom_gate_count = 0
        self._exposure_evaluation_count = 0
        self._exposure_cache_day: int | None = None
        self._exposure_cache_value: float | None = None

        self._timeline: list[ShadowDecision] = []

    # Public scalar properties make the strategy easy to audit in unit tests
    # without requiring callers to understand its private state layout.
    @property
    def labels(self) -> tuple[MajorityOutcome | None, ...]:
        return tuple(self._labels)

    @property
    def markov_context(self) -> tuple[MajorityOutcome, ...]:
        context: list[MajorityOutcome] = []
        for label in reversed(self._labels):
            if label is None:
                break
            context.append(label)
            if len(context) == self.parameters.markov_order:
                break
        return tuple(reversed(context))

    @property
    def last_forecast(self) -> Forecast:
        return self._last_forecast

    @property
    def last_action(self) -> int:
        return self._last_action

    @property
    def virtual_action(self) -> int:
        return self._virtual_action

    @property
    def genuine_observations(self) -> int:
        return self._genuine_observations

    @property
    def genuine_nonzero_observations(self) -> int:
        return self._genuine_nonzero_observations

    @property
    def observed_live_intervals(self) -> int:
        return self._observed_live_intervals

    @property
    def scoreable_virtual_trades(self) -> int:
        return self._scoreable_virtual_trades

    @property
    def cumulative_virtual_pnl(self) -> int:
        return self._virtual_cumulative_pnl

    @property
    def recent_virtual_pnl(self) -> int:
        return sum(self._virtual_pnl_window)

    @property
    def virtual_trade_pnls(self) -> tuple[int, ...]:
        return tuple(self._virtual_trade_pnls)

    @property
    def real_active(self) -> bool:
        return self._real_active

    @property
    def activation_pending(self) -> bool:
        return self._activation_pending

    @property
    def activation_day(self) -> int | None:
        return self._activation_day

    @property
    def activation_events(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(event) for event in self._activation_events)

    @property
    def deactivation_events(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(event) for event in self._deactivation_events)

    @property
    def reactivation_count(self) -> int:
        return self._reactivation_count

    @property
    def cooldown_active(self) -> bool:
        return self._cooldown_active

    @property
    def cooldown_observations(self) -> int:
        return self._cooldown_observations

    @property
    def actual_cumulative_pnl(self) -> int:
        return self._actual_cumulative_pnl

    @property
    def realised_increment_by_day(self) -> dict[int, int]:
        return dict(self._realised_increment_by_day)

    @property
    def loss_stop_active(self) -> bool:
        return self._loss_stop_active

    @property
    def loss_stop_event(self) -> StopEvent | None:
        return self._loss_stop_event

    @property
    def timeline(self) -> tuple[ShadowDecision, ...]:
        return tuple(self._timeline)

    @property
    def current_edge_gate_count(self) -> int:
        return self._current_edge_gate_count

    @property
    def floor_gate_count(self) -> int:
        return self._floor_gate_count

    @property
    def unknown_gate_count(self) -> int:
        return self._unknown_gate_count

    @property
    def headroom_gate_count(self) -> int:
        return self._headroom_gate_count

    @property
    def exposure_evaluation_count(self) -> int:
        return self._exposure_evaluation_count

    @property
    def qualification_state(self) -> str:
        if self._loss_stop_active:
            return "loss_stopped"
        if self._real_active:
            return "active"
        if self._activation_pending:
            return "pending_activation"
        if self._cooldown_active:
            return "cooldown"
        if self._qualification_streak:
            return f"qualifying_{self._qualification_streak}"
        if self._genuine_nonzero_observations < self.parameters.minimum_genuine_nonzero_observations:
            return "warming"
        return "paper"

    def _resolve_exposure(self, observation: AgentObservation) -> float:
        if self._exposure_cache_day == observation.day:
            assert self._exposure_cache_value is not None
            return self._exposure_cache_value
        source = self.other_portfolio_exposure
        value = source(observation) if callable(source) else source
        numeric = _validate_exposure(value)
        self._exposure_cache_day = observation.day
        self._exposure_cache_value = numeric
        self._exposure_evaluation_count += 1
        return numeric

    def _record_pause(self, observation: AgentObservation, reason: str) -> None:
        self._pause_active_today = True
        self._pause_reason_today = reason
        self._pause_events.append({"day": observation.day, "reason": reason})

    def _account_actual_pnl(
        self,
        observation: AgentObservation,
        *,
        live_interval: bool,
    ) -> int:
        if not live_interval:
            return 0
        change = observation.previous_price_change
        if change is None:
            return 0
        increment = observation.own_position * change
        before = self._actual_cumulative_pnl
        after = before + increment
        self._actual_cumulative_pnl = after
        self._realised_increment_by_day[observation.day] = increment
        if after <= -self.parameters.maximum_actual_loss and not self._loss_stop_active:
            overshoot = max(0, -after - self.parameters.maximum_actual_loss)
            self._loss_stop_event = StopEvent(
                "loss_stop",
                observation.day,
                before,
                after,
                overshoot,
            )
            self._loss_stop_active = True
        return increment

    def _append_observation(
        self,
        observation: AgentObservation,
        *,
        movement_kind: str,
        live_interval: bool,
    ) -> MajorityOutcome | None:
        """Append one public observation after virtual scoring."""

        if not live_interval:
            if movement_kind == "reset":
                self._reset_observations += 1
                # Reset is a hard public context boundary.  It is not a label.
                self._labels.clear()
            return None

        self._observed_live_intervals += 1
        self._genuine_observations += 1
        change = observation.previous_price_change
        if movement_kind == "genuine_nonzero":
            assert change is not None and change != 0
            label = MajorityOutcome.LONG if change < 0 else MajorityOutcome.SHORT
            self._genuine_nonzero_observations += 1
            self._labels.append(label)
            return label

        # Unknown, zero and clipped intervals are explicit context breaks.
        self._labels.append(None)
        if movement_kind == "unknown_zero":
            self._zero_observations += 1
            self._unknown_observations += 1
        elif movement_kind == "floor_clipped":
            self._clipped_observations += 1
            self._unknown_observations += 1
        else:
            self._unknown_observations += 1
        return None

    def _score_virtual_interval(
        self,
        observation: AgentObservation,
        *,
        movement_kind: str,
        live_interval: bool,
    ) -> tuple[int, int, bool]:
        """Score the prior virtual action before changing model history."""

        change = observation.previous_price_change
        raw_pnl = (
            self._virtual_action * change
            if live_interval and change is not None
            else 0
        )
        scoreable = (
            live_interval
            and movement_kind == "genuine_nonzero"
            and self._virtual_action != 0
        )
        if scoreable:
            self._virtual_trade_pnls.append(raw_pnl)
            self._virtual_pnl_window.append(raw_pnl)
            self._virtual_cumulative_pnl += raw_pnl
            self._scoreable_virtual_trades += 1
            self._last_scoreable_day = observation.day
            return raw_pnl, raw_pnl, True
        # Keep the raw economic result visible in the per-day audit, but do
        # not allow ambiguous/reset/clipped evidence into shadow health.
        return raw_pnl, 0, False

    def _update_cooldown(self, *, live_interval: bool) -> None:
        if not self._cooldown_active or not live_interval:
            return
        self._cooldown_observations += 1
        if self._cooldown_observations >= self.parameters.cooldown_genuine_observations:
            self._cooldown_active = False

    def _qualifying_evaluation(self, *, forecast_action: int) -> bool:
        return (
            self._genuine_nonzero_observations
            >= self.parameters.minimum_genuine_nonzero_observations
            and self._scoreable_virtual_trades
            >= self.parameters.minimum_scoreable_virtual_trades
            and self._virtual_cumulative_pnl
            >= self.parameters.minimum_initial_virtual_pnl
            and self.recent_virtual_pnl
            >= self.parameters.minimum_recent_virtual_pnl
            and forecast_action != 0
        )

    def _evaluate_shadow_health(
        self,
        *,
        scoreable_trade: bool,
        forecast_action: int,
    ) -> bool:
        """Update health only on a newly scoreable virtual trade."""

        if not scoreable_trade:
            return False
        self._health_evaluation_count += 1

        if self._real_active:
            if (
                self._scoreable_virtual_trades
                >= self.parameters.minimum_scoreable_virtual_trades
                and self.recent_virtual_pnl
                <= self.parameters.deactivation_virtual_pnl_limit
            ):
                self._health_bad_streak += 1
            else:
                self._health_bad_streak = 0
            if self._health_bad_streak >= self.parameters.health_bad_streak_required:
                return True
            return False

        if self._cooldown_active or self._activation_pending or self._loss_stop_active:
            return False
        if self._qualifying_evaluation(forecast_action=forecast_action):
            self._qualification_streak += 1
        else:
            self._qualification_streak = 0
        if self._qualification_streak >= 2:
            self._activation_pending = True
            return False
        return False

    def _deactivate(self, observation: AgentObservation, reason: str) -> None:
        self._real_active = False
        self._activation_pending = False
        self._qualification_streak = 0
        self._cooldown_active = True
        self._cooldown_observations = 0
        event = {
            "day": observation.day,
            "reason": reason,
            "recent_virtual_pnl": self.recent_virtual_pnl,
            "cumulative_virtual_pnl": self._virtual_cumulative_pnl,
        }
        self._deactivation_events.append(event)
        self._deactivated_this_day = True
        self._deactivation_reason_today = reason

    def _activate_if_pending(self, observation: AgentObservation) -> None:
        if not self._activation_pending:
            return
        self._activation_pending = False
        self._real_active = True
        self._activation_day = observation.day
        reactivation = self._ever_activated
        if reactivation:
            self._reactivation_count += 1
        self._ever_activated = True
        self._activation_events.append(
            {
                "day": observation.day,
                "reactivation": reactivation,
                "cumulative_virtual_pnl": self._virtual_cumulative_pnl,
                "recent_virtual_pnl": self.recent_virtual_pnl,
            }
        )

    def _real_action(
        self,
        observation: AgentObservation,
        *,
        movement_kind: str,
        live_decision: bool,
        virtual_action: int,
    ) -> tuple[int, bool, bool, bool, bool, float | None]:
        """Apply real-position gates to the already selected virtual action."""

        if not live_decision:
            return 0, False, False, False, False, None

        if movement_kind in {"reset", "inactive_or_startup"}:
            return 0, False, False, False, False, None

        if movement_kind in {"unknown_zero", "unknown"}:
            self._unknown_gate_count += 1
            self._record_pause(observation, movement_kind)
            return 0, False, False, True, False, None

        if movement_kind == "floor_clipped":
            self._unknown_gate_count += 1
            self._floor_gate_count += 1
            self._record_pause(observation, movement_kind)
            return 0, False, True, True, False, None

        if self._loss_stop_active:
            return 0, False, False, False, False, None

        floor_gate = observation.price == observation.price_floor
        if floor_gate:
            self._floor_gate_count += 1
            return 0, False, True, False, False, None

        if not self._real_active or self._activation_pending or self._cooldown_active:
            return 0, False, False, False, False, None

        if virtual_action == 0:
            self._current_edge_gate_count += 1
            return 0, True, False, False, False, None

        exposure = self._resolve_exposure(observation)
        permitted = (
            exposure
            + abs(virtual_action) * observation.price
            + self.parameters.portfolio_reserve
            <= GROSS_PORTFOLIO_BUDGET
        )
        if not permitted:
            self._headroom_gate_count += 1
            return 0, False, False, False, True, exposure
        return virtual_action, False, False, False, False, exposure

    def decide(self, observation: AgentObservation) -> int:
        """Process one causal observation and select the next real position."""

        if self._last_decision_day is not None:
            if observation.day == self._last_decision_day:
                return self._last_action
            if observation.day < self._last_decision_day:
                raise ValueError("shadow strategy observations must be chronological")

        self._pause_active_today = False
        self._pause_reason_today = None
        self._deactivated_this_day = False
        self._deactivation_reason_today = None

        # Qualification on day t sets pending activation; the first actual
        # position is selected on day t+1, never for the qualifying interval.
        self._activate_if_pending(observation)

        live_interval = _is_live_interval(observation)
        live_decision = _is_live_decision(observation)
        movement_kind = _movement_kind(observation)
        prior_virtual_action = self._virtual_action

        actual_increment = self._account_actual_pnl(
            observation,
            live_interval=live_interval,
        )
        raw_virtual_pnl, virtual_interval_pnl, scoreable_trade = self._score_virtual_interval(
            observation,
            movement_kind=movement_kind,
            live_interval=live_interval,
        )
        if live_interval and observation.day == self._last_observed_day:
            # This branch is unreachable because duplicate days return above;
            # keeping the assertion documents the one-observation invariant.
            raise RuntimeError("live interval was processed twice")
        if (
            live_interval
            or movement_kind == "reset"
        ):
            self._last_observed_day = observation.day

        label = self._append_observation(
            observation,
            movement_kind=movement_kind,
            live_interval=live_interval,
        )
        self._update_cooldown(live_interval=live_interval)

        if not live_decision:
            forecast = Forecast(0.5, 0.5, 0, movement_kind)
            virtual_action = 0
        else:
            forecast = self._model.estimate(tuple(self._labels))
            virtual_action = payoff_action(
                forecast,
                long_majority_move=observation.long_majority_move,
                short_majority_move=observation.short_majority_move,
                min_expected_pnl=self.parameters.minimum_expected_pnl,
                min_confidence=self.parameters.minimum_confidence,
            )
        self._last_forecast = forecast

        health_deactivation = self._evaluate_shadow_health(
            scoreable_trade=scoreable_trade,
            forecast_action=virtual_action,
        )
        if health_deactivation and self._real_active:
            self._deactivate(observation, "shadow_health")
        if self._loss_stop_active and self._real_active:
            self._deactivate(observation, "loss_stop")

        real_action, current_edge_gate, floor_gate, unknown_gate, headroom_gate, exposure = self._real_action(
            observation,
            movement_kind=movement_kind,
            live_decision=live_decision,
            virtual_action=virtual_action,
        )

        self._virtual_action = virtual_action
        self._last_action = real_action
        self._last_decision_day = observation.day

        loss_overshoot = (
            self._loss_stop_event.loss_limit_overshoot
            if self._loss_stop_event is not None
            and self._loss_stop_event.day == observation.day
            else None
        )
        context = tuple(label_item.value for label_item in self.markov_context)
        record = ShadowDecision(
            day=observation.day,
            live_decision=live_decision,
            observed_live_interval=live_interval,
            movement_kind=movement_kind,
            observed_price_change=observation.previous_price_change,
            observed_label=label.value if label is not None else None,
            actual_position=observation.own_position,
            actual_pnl_increment=actual_increment,
            actual_cumulative_pnl=self._actual_cumulative_pnl,
            prior_virtual_action=prior_virtual_action,
            raw_virtual_interval_pnl=raw_virtual_pnl,
            virtual_interval_pnl=virtual_interval_pnl,
            virtual_trade_scoreable=scoreable_trade,
            cumulative_virtual_pnl=self._virtual_cumulative_pnl,
            recent_virtual_pnl=self.recent_virtual_pnl,
            scoreable_virtual_trades=self._scoreable_virtual_trades,
            genuine_observations=self._genuine_observations,
            genuine_nonzero_observations=self._genuine_nonzero_observations,
            markov_context=context,
            markov_context_support=forecast.support,
            forecast_p_long=forecast.p_long,
            forecast_p_short=forecast.p_short,
            forecast_p_unknown=forecast.p_unknown,
            forecast_confidence=forecast.confidence,
            forecast_source=forecast.source,
            virtual_action=virtual_action,
            qualification_state=self.qualification_state,
            qualification_streak=self._qualification_streak,
            real_active=self._real_active,
            real_action=real_action,
            activation_day=self._activation_day,
            activation_pending=self._activation_pending,
            cooldown_active=self._cooldown_active,
            cooldown_observations=self._cooldown_observations,
            deactivated_this_day=self._deactivated_this_day,
            deactivation_reason=self._deactivation_reason_today,
            pause_active=self._pause_active_today,
            pause_reason=self._pause_reason_today,
            loss_stop_active=self._loss_stop_active,
            loss_stop_overshoot=loss_overshoot,
            current_edge_gate=current_edge_gate,
            floor_gate=floor_gate,
            unknown_gate=unknown_gate,
            headroom_gate=headroom_gate,
            exposure=exposure,
        )
        self._timeline.append(record)
        return real_action

    def diagnostics(self) -> dict[str, object]:
        """Return JSON-friendly scalar, event, and per-decision diagnostics."""

        return {
            "candidate_name": self.candidate_name,
            "agent_name": self.name,
            "parameters": asdict(self.parameters),
            "genuine_observations": self._genuine_observations,
            "genuine_nonzero_observations": self._genuine_nonzero_observations,
            "observed_live_intervals": self._observed_live_intervals,
            "unknown_observations": self._unknown_observations,
            "zero_observations": self._zero_observations,
            "clipped_observations": self._clipped_observations,
            "reset_observations": self._reset_observations,
            "labels": [label.value if label is not None else None for label in self._labels],
            "markov_context": [label.value for label in self.markov_context],
            "scoreable_virtual_trades": self._scoreable_virtual_trades,
            "virtual_trade_pnls": list(self._virtual_trade_pnls),
            "cumulative_virtual_pnl": self._virtual_cumulative_pnl,
            "recent_virtual_pnl": self.recent_virtual_pnl,
            "qualification_state": self.qualification_state,
            "qualification_streak": self._qualification_streak,
            "health_evaluation_count": self._health_evaluation_count,
            "health_bad_streak": self._health_bad_streak,
            "real_active": self._real_active,
            "activation_pending": self._activation_pending,
            "activation_day": self._activation_day,
            "activation_events": [dict(event) for event in self._activation_events],
            "deactivation_events": [dict(event) for event in self._deactivation_events],
            "reactivation_count": self._reactivation_count,
            "cooldown_active": self._cooldown_active,
            "cooldown_observations": self._cooldown_observations,
            "pause_events": [dict(event) for event in self._pause_events],
            "actual_realised_pnl": self._actual_cumulative_pnl,
            "realised_increment_by_day": dict(self._realised_increment_by_day),
            "loss_stop_active": self._loss_stop_active,
            "loss_stop_event": (
                asdict(self._loss_stop_event)
                if self._loss_stop_event is not None
                else None
            ),
            "current_edge_gate_count": self._current_edge_gate_count,
            "floor_gate_count": self._floor_gate_count,
            "unknown_gate_count": self._unknown_gate_count,
            "headroom_gate_count": self._headroom_gate_count,
            "exposure_evaluation_count": self._exposure_evaluation_count,
            "timeline": [asdict(record) for record in self._timeline],
        }


def make_shadow_strategy(
    name: str,
    *,
    agent_name: str = "focal",
    other_portfolio_exposure: ExposureSource = 0.0,
) -> ShadowValidatedMarkov:
    """Construct one of the three frozen Pass 5A shadow candidates."""

    try:
        parameters = SHADOW_PARAMETERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown Pass 5A shadow candidate {name!r}") from exc
    return ShadowValidatedMarkov(
        parameters=parameters,
        candidate_name=name,
        name=agent_name,
        other_portfolio_exposure=other_portfolio_exposure,
    )


__all__ = [
    "COOLDOWN_GENUINE_OBSERVATIONS",
    "DEACTIVATION_VIRTUAL_PNL_LIMIT",
    "GROSS_PORTFOLIO_BUDGET",
    "HEALTH_BAD_STREAK_REQUIRED",
    "INITIAL_VIRTUAL_PNL_MINIMUM",
    "MAX_LIFERAFT_LOSS_AUD",
    "MIN_CONFIDENCE",
    "MIN_EXPECTED_PNL",
    "MIN_SCOREABLE_VIRTUAL_TRADES",
    "PORTFOLIO_RESERVE_AUD",
    "RECENT_VIRTUAL_PNL_MINIMUM",
    "SHADOW_HEALTH_WINDOW",
    "SHADOW_PARAMETERS",
    "SHADOW_STRATEGY_NAMES",
    "ShadowDecision",
    "ShadowParameters",
    "ShadowValidatedMarkov",
    "StopEvent",
    "make_shadow_strategy",
]
