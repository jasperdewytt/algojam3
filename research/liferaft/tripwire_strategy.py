"""Flat-by-default prospective specialist tripwire for consumed Liferaft data."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Callable, Protocol, TypeAlias

from .pass6_strategies import movement_kind
from .simulator import AgentObservation


IMPACT_HAIRCUT = 1_300
BLOCK_SIZE = 20
MIN_BLOCK_FORECASTS = 10
MIN_DIRECTIONAL_ACCURACY = 0.80
MIN_ADJUSTED_BLOCK_PNL = 15_000
MAX_PAPER_BLOCK_DRAWDOWN = 10_000
MAX_ACTUAL_LOSS = 20_000
MAX_TRAILING_DRAWDOWN = 20_000
PORTFOLIO_BUDGET = 600_000
PORTFOLIO_HEADROOM_RESERVE = 10_000
ROLLING_BIAS_WINDOW = 20
CYCLE_PERIODS = tuple(range(2, 11))

VARIANTS = ("tripwire_any", "tripwire_consensus2", "tripwire_no_cycle")
BASE_SPECIALIST_NAMES = (
    "persistence",
    "reversal",
    "rolling_unconditional_bias_20",
    "markov_order_1",
    "markov_order_2",
    "markov_order_3",
)
CYCLE_SPECIALIST_NAME = "exact_cycle_2_10"

ExposureSource: TypeAlias = float | int | Callable[[AgentObservation], float]


def _position_for_bit(bit: int) -> int:
    if bit not in (0, 1):
        raise ValueError("movement bit must be zero or one")
    return 1 if bit else -1


@dataclass(frozen=True)
class SpecialistForecast:
    name: str
    position: int
    support: int
    detail: str
    cycle_period: int | None = None

    def __post_init__(self) -> None:
        if type(self.position) is not int or self.position not in (-1, 0, 1):
            raise ValueError("specialist position must be integral and in {-1,0,+1}")
        if self.support < 0:
            raise ValueError("specialist support cannot be negative")


class Specialist(Protocol):
    name: str

    def reset_context(self) -> None: ...

    def observe(self, bit: int) -> bool: ...

    def forecast(self) -> SpecialistForecast: ...


class PersistenceSpecialist:
    name = "persistence"

    def __init__(self) -> None:
        self._last: int | None = None

    def reset_context(self) -> None:
        self._last = None

    def observe(self, bit: int) -> bool:
        _position_for_bit(bit)
        self._last = bit
        return False

    def forecast(self) -> SpecialistForecast:
        return SpecialistForecast(
            self.name,
            0 if self._last is None else _position_for_bit(self._last),
            0 if self._last is None else 1,
            "previous_genuine_sign" if self._last is not None else "no_context",
        )


class ReversalSpecialist:
    name = "reversal"

    def __init__(self) -> None:
        self._last: int | None = None

    def reset_context(self) -> None:
        self._last = None

    def observe(self, bit: int) -> bool:
        _position_for_bit(bit)
        self._last = bit
        return False

    def forecast(self) -> SpecialistForecast:
        return SpecialistForecast(
            self.name,
            0 if self._last is None else _position_for_bit(1 - self._last),
            0 if self._last is None else 1,
            "opposite_previous_genuine_sign" if self._last is not None else "no_context",
        )


class RollingBiasSpecialist:
    name = "rolling_unconditional_bias_20"

    def __init__(self, window: int = ROLLING_BIAS_WINDOW) -> None:
        self.window = window
        self._history: deque[int] = deque(maxlen=window)

    def reset_context(self) -> None:
        self._history.clear()

    def observe(self, bit: int) -> bool:
        _position_for_bit(bit)
        self._history.append(bit)
        return False

    def forecast(self) -> SpecialistForecast:
        positives = sum(self._history)
        negatives = len(self._history) - positives
        position = 0
        if positives > negatives:
            position = 1
        elif negatives > positives:
            position = -1
        return SpecialistForecast(
            self.name,
            position,
            len(self._history),
            f"trailing_{self.window}_unconditional_sign_bias",
        )


class MarkovSpecialist:
    def __init__(self, order: int) -> None:
        if order not in (1, 2, 3):
            raise ValueError("tripwire Markov order must be 1, 2, or 3")
        self.order = order
        self.name = f"markov_order_{order}"
        self._tail: deque[int] = deque(maxlen=order)
        self._counts: dict[tuple[int, ...], list[int]] = {}

    def reset_context(self) -> None:
        self._tail.clear()
        self._counts.clear()

    def observe(self, bit: int) -> bool:
        _position_for_bit(bit)
        if len(self._tail) == self.order:
            context = tuple(self._tail)
            self._counts.setdefault(context, [0, 0])[bit] += 1
        self._tail.append(bit)
        return False

    def forecast(self) -> SpecialistForecast:
        if len(self._tail) < self.order:
            return SpecialistForecast(self.name, 0, 0, "incomplete_context")
        context = tuple(self._tail)
        counts = self._counts.get(context, [0, 0])
        position = 0
        if counts[1] > counts[0]:
            position = 1
        elif counts[0] > counts[1]:
            position = -1
        return SpecialistForecast(
            self.name,
            position,
            counts[0] + counts[1],
            f"exact_order_{self.order}_empirical_transition",
        )


class ExactCycleSpecialist:
    """Shortest exact period in 2..10 after two identical nonconstant blocks."""

    name = CYCLE_SPECIALIST_NAME

    def __init__(self) -> None:
        self._history: list[int] = []
        self._period: int | None = None

    @property
    def period(self) -> int | None:
        return self._period

    def reset_context(self) -> None:
        self._history.clear()
        self._period = None

    def _shortest_valid_period(self) -> int | None:
        for period in CYCLE_PERIODS:
            if len(self._history) < 2 * period:
                continue
            previous = self._history[-2 * period : -period]
            latest = self._history[-period:]
            if previous == latest and len(set(latest)) == 2:
                return period
        return None

    def observe(self, bit: int) -> bool:
        _position_for_bit(bit)
        if self._period is not None:
            expected = self._history[-self._period]
            if bit != expected:
                # Go flat immediately, but retain the causal observations so
                # a false short suffix cannot erase evidence for a longer
                # period. No replacement period is selected on this outcome.
                self._history.append(bit)
                self._period = None
                return True
            self._history.append(bit)
            return False
        self._history.append(bit)
        self._period = self._shortest_valid_period()
        return False

    def forecast(self) -> SpecialistForecast:
        if self._period is None:
            return SpecialistForecast(self.name, 0, len(self._history), "no_exact_cycle")
        predicted = self._history[-self._period]
        return SpecialistForecast(
            self.name,
            _position_for_bit(predicted),
            len(self._history),
            "two_complete_identical_nonconstant_blocks",
            cycle_period=self._period,
        )


@dataclass(frozen=True)
class SpecialistBlockResult:
    block: int
    scoreable_observations: int
    nonflat_forecasts: int
    correct_forecasts: int
    accuracy: float | None
    adjusted_pnl: int
    max_drawdown: int
    qualifies: bool


class _BlockAccumulator:
    def __init__(self) -> None:
        self.observations = 0
        self.forecasts = 0
        self.correct = 0
        self.pnl = 0
        self.high_water = 0
        self.max_drawdown = 0

    def score(self, position: int, price_change: int) -> None:
        self.observations += 1
        if position == 0:
            return
        self.forecasts += 1
        self.correct += int(position * price_change > 0)
        self.pnl += position * price_change - IMPACT_HAIRCUT
        self.high_water = max(self.high_water, self.pnl)
        self.max_drawdown = max(self.max_drawdown, self.high_water - self.pnl)

    def finish(self, block: int) -> SpecialistBlockResult:
        if self.observations != BLOCK_SIZE:
            raise AssertionError("tripwire blocks must contain exactly 20 scoreable outcomes")
        accuracy = self.correct / self.forecasts if self.forecasts else None
        qualifies = (
            self.forecasts >= MIN_BLOCK_FORECASTS
            and self.correct * 5 >= self.forecasts * 4
            and self.pnl >= MIN_ADJUSTED_BLOCK_PNL
            and self.max_drawdown <= MAX_PAPER_BLOCK_DRAWDOWN
        )
        return SpecialistBlockResult(
            block=block,
            scoreable_observations=self.observations,
            nonflat_forecasts=self.forecasts,
            correct_forecasts=self.correct,
            accuracy=accuracy,
            adjusted_pnl=self.pnl,
            max_drawdown=self.max_drawdown,
            qualifies=qualifies,
        )


def make_specialists(*, include_cycle: bool) -> tuple[Specialist, ...]:
    specialists: list[Specialist] = [
        PersistenceSpecialist(),
        ReversalSpecialist(),
        RollingBiasSpecialist(),
        MarkovSpecialist(1),
        MarkovSpecialist(2),
        MarkovSpecialist(3),
    ]
    if include_cycle:
        specialists.append(ExactCycleSpecialist())
    expected = BASE_SPECIALIST_NAMES + ((CYCLE_SPECIALIST_NAME,) if include_cycle else ())
    if tuple(specialist.name for specialist in specialists) != expected:
        raise AssertionError("tripwire specialist order changed")
    return tuple(specialists)


class TripwireStrategy:
    """Paper-test specialists independently and trade only checkpoint winners."""

    def __init__(
        self,
        variant: str,
        *,
        other_portfolio_exposure: ExposureSource = 0.0,
    ) -> None:
        if variant not in VARIANTS:
            raise KeyError(variant)
        self.variant = variant
        self.name = variant
        self.other_portfolio_exposure = other_portfolio_exposure
        self.specialists = make_specialists(include_cycle=variant != "tripwire_no_cycle")
        self._by_name = {specialist.name: specialist for specialist in self.specialists}
        self._blocks: dict[str, list[SpecialistBlockResult]] = {
            name: [] for name in self._by_name
        }
        self._block_accumulators = {
            name: _BlockAccumulator() for name in self._by_name
        }
        self._score_totals = {
            name: {"forecasts": 0, "correct": 0, "adjusted_pnl": 0}
            for name in self._by_name
        }
        self._pending_forecasts: dict[str, SpecialistForecast] = {}
        self._scoreable_count = 0
        self._qualified_names: tuple[str, ...] = ()
        self._active = False
        self._activation_day: int | None = None
        self._activation_events: list[dict[str, object]] = []
        self._deactivation_events: list[dict[str, object]] = []
        self._checkpoint_records: list[dict[str, object]] = []
        self._cooldown_until_block = 0
        self._actual_pnl = 0
        self._actual_high_water = 0
        self._actual_max_drawdown = 0
        self._loss_stop = False
        self._drawdown_stop = False
        self._stop_overshoots: list[dict[str, int | str]] = []
        self._last_day: int | None = None
        self._last_action = 0
        self._last_paper_action = 0
        self._exposure_day: int | None = None
        self._exposure_value: float | None = None
        self._exposure_evaluations = 0
        self._headroom_gates = 0
        self._context_breaks: dict[str, int] = {}
        self._paper_records: list[dict[str, object]] = []
        if not callable(other_portfolio_exposure):
            self._validate_exposure(other_portfolio_exposure)

    @staticmethod
    def _validate_exposure(value: object) -> float:
        if isinstance(value, bool):
            raise ValueError("other exposure must be numeric")
        numeric = float(value)
        if not isfinite(numeric) or numeric < 0:
            raise ValueError("other exposure must be finite and non-negative")
        return numeric

    def _resolve_exposure(self, observation: AgentObservation) -> float:
        if self._exposure_day == observation.day:
            assert self._exposure_value is not None
            return self._exposure_value
        source = self.other_portfolio_exposure
        value = source(observation) if callable(source) else source
        self._exposure_value = self._validate_exposure(value)
        self._exposure_day = observation.day
        self._exposure_evaluations += 1
        return self._exposure_value

    @staticmethod
    def _start_day(observation: AgentObservation) -> int:
        return (
            observation.voting_start_day
            if observation.voting_start_day is not None
            else observation.marked_boundary_day
        )

    def _account_actual(
        self, observation: AgentObservation, kind: str
    ) -> tuple[int, tuple[str, ...]]:
        if (
            observation.day <= self._start_day(observation)
            or observation.previous_price_change is None
            or kind in {"inactive_or_startup", "reset"}
        ):
            return 0, ()
        increment = observation.own_position * observation.previous_price_change
        self._actual_pnl += increment
        self._actual_high_water = max(self._actual_high_water, self._actual_pnl)
        drawdown = self._actual_high_water - self._actual_pnl
        self._actual_max_drawdown = max(self._actual_max_drawdown, drawdown)
        triggered: list[str] = []
        if self._actual_pnl <= -MAX_ACTUAL_LOSS and not self._loss_stop:
            self._loss_stop = True
            triggered.append("loss_stop")
            self._stop_overshoots.append(
                {
                    "day": observation.day,
                    "stop": "loss",
                    "overshoot": max(0, -self._actual_pnl - MAX_ACTUAL_LOSS),
                }
            )
        if drawdown >= MAX_TRAILING_DRAWDOWN and not self._drawdown_stop:
            self._drawdown_stop = True
            triggered.append("drawdown_stop")
            self._stop_overshoots.append(
                {
                    "day": observation.day,
                    "stop": "drawdown",
                    "overshoot": max(0, drawdown - MAX_TRAILING_DRAWDOWN),
                }
            )
        return increment, tuple(triggered)

    def _score_pending(self, price_change: int) -> bool:
        for name in self._by_name:
            position = self._pending_forecasts.get(
                name, SpecialistForecast(name, 0, 0, "no_prior_forecast")
            ).position
            self._block_accumulators[name].score(position, price_change)
            totals = self._score_totals[name]
            if position != 0:
                totals["forecasts"] += 1
                totals["correct"] += int(position * price_change > 0)
                totals["adjusted_pnl"] += position * price_change - IMPACT_HAIRCUT
        self._scoreable_count += 1
        if self._scoreable_count % BLOCK_SIZE:
            return False
        block = self._scoreable_count // BLOCK_SIZE
        for name in self._by_name:
            self._blocks[name].append(self._block_accumulators[name].finish(block))
            self._block_accumulators[name] = _BlockAccumulator()
        return True

    def _qualifying_specialists(self) -> tuple[str, ...]:
        qualified = []
        for name in self._by_name:
            blocks = self._blocks[name]
            if len(blocks) >= 2 and blocks[-1].qualifies and blocks[-2].qualifies:
                qualified.append(name)
        return tuple(qualified)

    def _next_eligible_block_after_deactivation(self) -> int:
        completed, partial = divmod(self._scoreable_count, BLOCK_SIZE)
        return completed + (1 if partial == 0 else 2)

    def _deactivate(self, *, day: int, reasons: tuple[str, ...]) -> None:
        if not self._active:
            return
        unique_reasons = tuple(dict.fromkeys(reasons))
        eligible = self._next_eligible_block_after_deactivation()
        self._cooldown_until_block = max(self._cooldown_until_block, eligible)
        self._deactivation_events.append(
            {
                "day": day,
                "reasons": list(unique_reasons),
                "scoreable_count": self._scoreable_count,
                "fresh_block_required_until": self._cooldown_until_block,
                "prior_qualified_specialists": list(self._qualified_names),
            }
        )
        self._active = False
        self._qualified_names = ()

    def _agreement(
        self, forecasts: dict[str, SpecialistForecast]
    ) -> tuple[int, str, tuple[str, ...]]:
        available = tuple(
            name
            for name in self._qualified_names
            if forecasts[name].position != 0
        )
        positions = {forecasts[name].position for name in available}
        if len(positions) > 1:
            return 0, "specialist_disagreement", available
        required = 2 if self.variant == "tripwire_consensus2" else 1
        if len(available) < required:
            return 0, "insufficient_available_qualifiers", available
        return next(iter(positions)), "agreed", available

    def _reset_context(self, kind: str) -> None:
        for specialist in self.specialists:
            specialist.reset_context()
        self._pending_forecasts = {}
        self._context_breaks[kind] = self._context_breaks.get(kind, 0) + 1

    def decide(self, observation: AgentObservation) -> int:
        if self._last_day is not None:
            if observation.day == self._last_day:
                return self._last_action
            if observation.day < self._last_day:
                raise ValueError("tripwire observations must be chronological")

        kind = movement_kind(observation)
        start = self._start_day(observation)
        actual_increment, stop_reasons = self._account_actual(observation, kind)
        checkpoint = False
        cycle_contradiction = False
        deactivation_reasons: list[str] = list(stop_reasons)

        if observation.day < start:
            paper_action = 0
            actual_action = 0
            decision_reason = "inactive_or_startup"
            current_forecasts = {
                name: specialist.forecast() for name, specialist in self._by_name.items()
            }
            self._pending_forecasts = current_forecasts
        else:
            if observation.day == start:
                self._reset_context("voting_start")
            elif kind == "genuine_nonzero":
                assert observation.previous_price_change is not None
                checkpoint = self._score_pending(observation.previous_price_change)
                bit = int(observation.previous_price_change > 0)
                for name, specialist in self._by_name.items():
                    contradicted = specialist.observe(bit)
                    if name == CYCLE_SPECIALIST_NAME and contradicted:
                        cycle_contradiction = True
                if self._active and actual_increment < 0:
                    deactivation_reasons.append("first_realised_losing_trade")
                if (
                    self._active
                    and self._last_paper_action != 0
                    and self._last_paper_action * observation.previous_price_change < 0
                ):
                    deactivation_reasons.append("prospective_prediction_contradiction")
                if (
                    self._active
                    and cycle_contradiction
                    and CYCLE_SPECIALIST_NAME in self._qualified_names
                ):
                    deactivation_reasons.append("cycle_contradiction")
            else:
                if self._active:
                    deactivation_reasons.append(kind)
                self._reset_context(kind)

            if deactivation_reasons:
                self._deactivate(day=observation.day, reasons=tuple(deactivation_reasons))

            block = self._scoreable_count // BLOCK_SIZE
            theoretical_qualified: tuple[str, ...] | None = None
            checkpoint_eligible = False
            if checkpoint:
                theoretical_qualified = self._qualifying_specialists()
                checkpoint_eligible = (
                    block >= self._cooldown_until_block
                    and not self._loss_stop
                    and not self._drawdown_stop
                )
                self._qualified_names = (
                    theoretical_qualified if checkpoint_eligible else ()
                )

            current_forecasts = {
                name: specialist.forecast() for name, specialist in self._by_name.items()
            }
            self._pending_forecasts = current_forecasts
            agreed_action, agreement_reason, available = self._agreement(current_forecasts)

            if self._active and agreement_reason == "specialist_disagreement":
                self._deactivate(
                    day=observation.day, reasons=("specialist_disagreement",)
                )
                agreed_action = 0
            elif self._active and checkpoint and agreed_action == 0:
                self._deactivate(
                    day=observation.day,
                    reasons=("checkpoint_no_agreed_qualification",),
                )

            activated_now = False
            if (
                not self._active
                and checkpoint
                and checkpoint_eligible
                and agreed_action != 0
                and agreement_reason == "agreed"
            ):
                self._active = True
                activated_now = True
                if self._activation_day is None:
                    self._activation_day = observation.day
                cycle_period = current_forecasts.get(
                    CYCLE_SPECIALIST_NAME,
                    SpecialistForecast(CYCLE_SPECIALIST_NAME, 0, 0, "excluded"),
                ).cycle_period
                self._activation_events.append(
                    {
                        "day": observation.day,
                        "block": block,
                        "scoreable_count": self._scoreable_count,
                        "qualified_specialists": list(self._qualified_names),
                        "available_agreeing_specialists": list(available),
                        "action": agreed_action,
                        "cycle_period": cycle_period,
                    }
                )

            paper_action = agreed_action if self._active else 0
            decision_reason = (
                "activated_at_checkpoint"
                if activated_now
                else agreement_reason if self._active else "flat_not_authorized"
            )
            actual_action = paper_action
            if actual_action != 0:
                exposure = self._resolve_exposure(observation)
                requested = (
                    exposure
                    + abs(actual_action) * observation.price
                    + PORTFOLIO_HEADROOM_RESERVE
                )
                if requested > PORTFOLIO_BUDGET:
                    actual_action = 0
                    self._headroom_gates += 1
                    decision_reason = "portfolio_headroom"

            if checkpoint:
                self._checkpoint_records.append(
                    {
                        "day": observation.day,
                        "block": block,
                        "theoretical_qualified_specialists": list(
                            theoretical_qualified or ()
                        ),
                        "eligible_after_cooldown": checkpoint_eligible,
                        "current_qualified_specialists": list(self._qualified_names),
                        "available_agreeing_specialists": list(available),
                        "forecast_positions": {
                            name: forecast.position
                            for name, forecast in current_forecasts.items()
                        },
                        "paper_action": paper_action,
                        "active_after_decision": self._active,
                    }
                )

        if type(paper_action) is not int or paper_action not in (-1, 0, 1):
            raise AssertionError("paper action is not integral")
        if type(actual_action) is not int or actual_action not in (-1, 0, 1):
            raise AssertionError("actual action is not integral")
        if observation.day >= start:
            self._paper_records.append(
                {
                    "day": observation.day,
                    "movement_kind": kind,
                    "paper_action": paper_action,
                    "actual_action": actual_action,
                    "own_position": observation.own_position,
                    "actual_increment": actual_increment,
                    "active_after_decision": self._active,
                    "qualified_specialists": list(self._qualified_names),
                    "decision_reason": decision_reason,
                    "checkpoint_block": (
                        self._scoreable_count // BLOCK_SIZE if checkpoint else None
                    ),
                }
            )
        self._last_paper_action = paper_action
        self._last_action = actual_action
        self._last_day = observation.day
        return actual_action

    def diagnostics(self) -> dict[str, object]:
        return {
            "candidate_name": self.name,
            "variant": self.variant,
            "scoreable_count": self._scoreable_count,
            "active": self._active,
            "activation_day": self._activation_day,
            "activation_events": list(self._activation_events),
            "deactivation_events": list(self._deactivation_events),
            "checkpoint_records": list(self._checkpoint_records),
            "qualified_specialists": list(self._qualified_names),
            "actual_realised_pnl": self._actual_pnl,
            "actual_high_water": self._actual_high_water,
            "actual_max_drawdown": self._actual_max_drawdown,
            "loss_stop_active": self._loss_stop,
            "drawdown_stop_active": self._drawdown_stop,
            "stop_overshoots": list(self._stop_overshoots),
            "headroom_gate_count": self._headroom_gates,
            "exposure_evaluation_count": self._exposure_evaluations,
            "context_breaks": dict(self._context_breaks),
            "paper_records": list(self._paper_records),
            "specialist_blocks": {
                name: [asdict(block) for block in blocks]
                for name, blocks in self._blocks.items()
            },
            "specialist_score_totals": {
                name: dict(values) for name, values in self._score_totals.items()
            },
        }


def make_tripwire_strategy(
    variant: str,
    *,
    other_portfolio_exposure: ExposureSource = 0.0,
) -> TripwireStrategy:
    return TripwireStrategy(
        variant,
        other_portfolio_exposure=other_portfolio_exposure,
    )


def _check_observation(
    *, day: int, change: int | None, own_position: int
) -> AgentObservation:
    previous = None if change is None else 100_000
    price = 100_000 if change is None else 100_000 + change
    return AgentObservation(
        day=day,
        price=price,
        price_history=(price,) if previous is None else (previous, price),
        previous_price=previous,
        previous_price_change=change,
        previous_move_is_reset=False,
        is_reset_day=False,
        marked_boundary_day=365,
        price_floor=20_000,
        long_majority_move=-5_000,
        short_majority_move=8_000,
        position_limit=1,
        gross_portfolio_budget=600_000,
        own_position=own_position,
        voting_active=True,
        market_mode="inactive_until_marked",
        voting_start_day=365,
    )


def run_self_checks() -> None:
    cycle = ExactCycleSpecialist()
    for _ in range(30):
        cycle.observe(1)
    assert cycle.period is None and cycle.forecast().position == 0
    cycle.reset_context()
    for bit in (1, 1, 0, 1, 1, 0):
        cycle.observe(bit)
    assert cycle.period == 3 and cycle.forecast().position == 1
    assert cycle.observe(0) is True
    assert cycle.period is None and cycle.forecast().position == 0
    cycle.reset_context()
    period_eight = (1, 1, 0, 1, 0, 0, 1, 0)
    for bit in period_eight * 3:
        cycle.observe(bit)
    assert cycle.period == 8

    exposure_calls = 0

    def exposure(_: AgentObservation) -> int:
        nonlocal exposure_calls
        exposure_calls += 1
        return 0

    strategy = make_tripwire_strategy(
        "tripwire_any", other_portfolio_exposure=exposure
    )
    held = strategy.decide(_check_observation(day=365, change=None, own_position=0))
    for offset in range(1, 40):
        held = strategy.decide(
            _check_observation(day=365 + offset, change=8_000, own_position=held)
        )
    assert held == 0 and strategy.diagnostics()["activation_day"] is None
    checkpoint_observation = _check_observation(
        day=405, change=8_000, own_position=held
    )
    held = strategy.decide(checkpoint_observation)
    assert held == 1 and strategy.diagnostics()["activation_day"] == 405
    assert strategy.diagnostics()["actual_realised_pnl"] == 0
    assert strategy.decide(checkpoint_observation) == 1
    assert exposure_calls == 1
    held = strategy.decide(
        _check_observation(day=406, change=8_000, own_position=held)
    )
    assert strategy.diagnostics()["actual_realised_pnl"] == 8_000
    held = strategy.decide(
        _check_observation(day=407, change=-5_000, own_position=held)
    )
    assert held == 0 and strategy.diagnostics()["active"] is False
    reasons = strategy.diagnostics()["deactivation_events"][-1]["reasons"]
    assert "first_realised_losing_trade" in reasons

    risk = make_tripwire_strategy("tripwire_any")
    risk.decide(_check_observation(day=365, change=None, own_position=0))
    for offset in range(1, 5):
        risk.decide(
            _check_observation(
                day=365 + offset,
                change=-5_000,
                own_position=1,
            )
        )
    risk_diagnostics = risk.diagnostics()
    assert risk_diagnostics["actual_realised_pnl"] == -MAX_ACTUAL_LOSS
    assert risk_diagnostics["loss_stop_active"] is True
    assert risk_diagnostics["drawdown_stop_active"] is True


__all__ = [
    "BASE_SPECIALIST_NAMES",
    "BLOCK_SIZE",
    "CYCLE_PERIODS",
    "CYCLE_SPECIALIST_NAME",
    "ExactCycleSpecialist",
    "IMPACT_HAIRCUT",
    "MAX_ACTUAL_LOSS",
    "MAX_TRAILING_DRAWDOWN",
    "PORTFOLIO_BUDGET",
    "PORTFOLIO_HEADROOM_RESERVE",
    "SpecialistBlockResult",
    "SpecialistForecast",
    "TripwireStrategy",
    "VARIANTS",
    "make_tripwire_strategy",
    "run_self_checks",
]
