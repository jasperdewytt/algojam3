"""Small causal strategies for the clarified cold-start Liferaft market.

These policies deliberately do not use Year-1 calibration.  In the
competition-correct simulator mode the pre-voting price is constant, so the
first useful public label can only arrive after the voting-start action has
already been chosen.  Every model below updates at the start of a decision
from the immediately preceding genuine live interval, then chooses the next
position.
"""

from __future__ import annotations

from collections import deque
from math import exp, log
from typing import Callable, Mapping, Sequence

from .simulator import (
    AgentObservation,
    MajorityOutcome,
    infer_majority_from_price_change,
)
from .strategies import (
    Forecast,
    FlatModel,
    LastMajorityModel,
    PublicLabel,
    RegularisedMarkovModel,
    RollingFrequencyModel,
    payoff_action,
)


def _voting_start_day(observation: AgentObservation) -> int:
    return (
        observation.voting_start_day
        if observation.voting_start_day is not None
        else observation.marked_boundary_day
    )


def _is_live_day(observation: AgentObservation) -> bool:
    return observation.day >= _voting_start_day(observation)


def _is_genuine_live_interval(observation: AgentObservation) -> bool:
    """Whether ``previous_price_change`` is an already observable live move."""

    return (
        observation.day > _voting_start_day(observation)
        and not observation.previous_move_is_reset
    )


def live_labels_from_observation(
    observation: AgentObservation,
) -> tuple[PublicLabel, ...]:
    """Return only genuine live labels in the public history.

    Intervals ending on or before the voting-start day are excluded.  This
    also excludes the historical reset interval when a strategy is pointed at
    the legacy continuous/reset simulator.
    """

    start = _voting_start_day(observation)
    labels: list[PublicLabel] = []
    for endpoint_day in range(start + 1, len(observation.price_history)):
        change = (
            observation.price_history[endpoint_day]
            - observation.price_history[endpoint_day - 1]
        )
        labels.append(
            infer_majority_from_price_change(
                change,
                previous_move_is_reset=False,
            )
        )
    return tuple(labels)


class ColdStartStrategy:
    """Shared online bookkeeping for all cold-start candidates."""

    name = "cold-start"

    def __init__(self) -> None:
        self._labels: list[PublicLabel] = []
        self._observed_live_count = 0
        self._last_observed_day: int | None = None
        self._last_forecast = Forecast(0.5, 0.5, 0, "uninitialised")
        self._last_action = 0

    @property
    def labels(self) -> tuple[PublicLabel, ...]:
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

    def _observe_live_interval(
        self,
        observation: AgentObservation,
    ) -> tuple[bool, PublicLabel | None]:
        """Append exactly one newly public live interval, including zeros."""

        if (
            not _is_genuine_live_interval(observation)
            or observation.day == self._last_observed_day
        ):
            return False, None
        label = infer_majority_from_price_change(
            observation.previous_price_change,
            previous_move_is_reset=observation.previous_move_is_reset,
        )
        self._labels.append(label)
        self._observed_live_count += 1
        self._last_observed_day = observation.day
        return True, label

    def _action_from_forecast(
        self,
        forecast: Forecast,
        observation: AgentObservation,
        *,
        min_expected_pnl: float = 1_000.0,
        min_confidence: float = 0.10,
    ) -> int:
        self._last_forecast = forecast
        action = payoff_action(
            forecast,
            long_majority_move=observation.long_majority_move,
            short_majority_move=observation.short_majority_move,
            min_expected_pnl=min_expected_pnl,
            min_confidence=min_confidence,
        )
        self._last_action = action
        return action

    def _flat_before_live(self, observation: AgentObservation) -> bool:
        if not _is_live_day(observation):
            self._last_forecast = Forecast(0.5, 0.5, 0, "inactive")
            self._last_action = 0
            return True
        return False


class ColdStartFlat:
    """Always-flat benchmark and explicit no-trade fallback."""

    def __init__(self, name: str = "flat") -> None:
        self.name = name

    def decide(self, observation: AgentObservation) -> int:
        del observation
        return 0


class ColdStartFixedAction:
    """Fixed-action benchmark used only for comparisons."""

    def __init__(self, action: int, name: str | None = None) -> None:
        if action not in (-1, 0, 1):
            raise ValueError("fixed action must be -1, 0, or 1")
        self.action = action
        self.name = name or {1: "always-long", -1: "always-short", 0: "flat"}[action]

    def decide(self, observation: AgentObservation) -> int:
        del observation
        return self.action


class FlatBurnInStrategy(ColdStartStrategy):
    """Remain flat for N genuine live observations, then use one model."""

    def __init__(
        self,
        burn_in_genuine_observations: int = 3,
        *,
        model: str = "markov",
        windows: Sequence[int] = (5, 10, 20),
        markov_order: int = 2,
        alpha: float = 1.0,
        min_expected_pnl: float = 1_000.0,
        min_confidence: float = 0.10,
        name: str | None = None,
    ) -> None:
        super().__init__()
        if burn_in_genuine_observations < 0:
            raise ValueError("burn-in must be non-negative")
        if model not in {"last", "rolling", "markov"}:
            raise ValueError("model must be 'last', 'rolling', or 'markov'")
        self.burn_in_genuine_observations = burn_in_genuine_observations
        self.model_name = model
        self.min_expected_pnl = min_expected_pnl
        self.min_confidence = min_confidence
        self._model = _make_model(
            model,
            windows=windows,
            markov_order=markov_order,
            alpha=alpha,
        )
        self.name = name or f"burnin-{burn_in_genuine_observations}-{model}"

    def decide(self, observation: AgentObservation) -> int:
        self._observe_live_interval(observation)
        if self._flat_before_live(observation):
            return 0
        forecast = self._model.estimate(self._labels)
        if self._observed_live_count < self.burn_in_genuine_observations:
            self._last_forecast = forecast
            self._last_action = 0
            return 0
        return self._action_from_forecast(
            forecast,
            observation,
            min_expected_pnl=self.min_expected_pnl,
            min_confidence=self.min_confidence,
        )


class OnlineLastMajorityCounter(ColdStartStrategy):
    """Counter the most recent non-zero live majority, with zero burn-in."""

    def __init__(self, *, name: str = "online-last-counter") -> None:
        super().__init__()
        self.name = name
        self._model = LastMajorityModel()

    def decide(self, observation: AgentObservation) -> int:
        self._observe_live_interval(observation)
        if self._flat_before_live(observation):
            return 0
        return self._action_from_forecast(self._model.estimate(self._labels), observation)


class OnlineRollingFrequency(ColdStartStrategy):
    """Smoothed rolling-frequency predictor using only live observations."""

    def __init__(
        self,
        *,
        windows: Sequence[int] = (5, 10, 20),
        alpha: float = 1.0,
        min_support: int = 3,
        min_expected_pnl: float = 1_000.0,
        min_confidence: float = 0.10,
        name: str = "online-rolling-frequency",
    ) -> None:
        super().__init__()
        if min_support < 0:
            raise ValueError("min_support must be non-negative")
        self.name = name
        self.min_support = min_support
        self.min_expected_pnl = min_expected_pnl
        self.min_confidence = min_confidence
        self._model = RollingFrequencyModel(windows, alpha=alpha)

    def decide(self, observation: AgentObservation) -> int:
        self._observe_live_interval(observation)
        if self._flat_before_live(observation):
            return 0
        forecast = self._model.estimate(self._labels)
        if forecast.support < self.min_support:
            self._last_forecast = forecast
            self._last_action = 0
            return 0
        return self._action_from_forecast(
            forecast,
            observation,
            min_expected_pnl=self.min_expected_pnl,
            min_confidence=self.min_confidence,
        )


class OnlineRegularisedMarkov(ColdStartStrategy):
    """Order-one/two smoothed Markov predictor with unknown-state breaks."""

    def __init__(
        self,
        *,
        order: int = 2,
        alpha: float = 1.0,
        min_support: int = 3,
        min_expected_pnl: float = 1_000.0,
        min_confidence: float = 0.10,
        name: str = "online-markov",
    ) -> None:
        super().__init__()
        if min_support < 0:
            raise ValueError("min_support must be non-negative")
        self.name = name
        self.min_support = min_support
        self.min_expected_pnl = min_expected_pnl
        self.min_confidence = min_confidence
        self._model = RegularisedMarkovModel(order=order, alpha=alpha)

    def decide(self, observation: AgentObservation) -> int:
        self._observe_live_interval(observation)
        if self._flat_before_live(observation):
            return 0
        forecast = self._model.estimate(self._labels)
        if forecast.support < self.min_support:
            self._last_forecast = forecast
            self._last_action = 0
            return 0
        return self._action_from_forecast(
            forecast,
            observation,
            min_expected_pnl=self.min_expected_pnl,
            min_confidence=self.min_confidence,
        )


def _bounded_normalize(
    weights: Mapping[str, float],
    *,
    minimum: float,
    maximum: float,
) -> dict[str, float]:
    """Normalise weights while keeping every expert in fixed bounds."""

    current = dict(weights)
    for _ in range(32):
        clipped = {
            key: min(maximum, max(minimum, value))
            for key, value in current.items()
        }
        total = sum(clipped.values())
        normalised = {key: value / total for key, value in clipped.items()}
        if all(minimum - 1e-9 <= value <= maximum + 1e-9 for value in normalised.values()):
            return normalised
        current = normalised
    return normalised


class OnlineExpertEnsemble(ColdStartStrategy):
    """Bounded Hedge-style ensemble of flat and small online experts."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.5,
        minimum_weight: float = 0.05,
        maximum_weight: float = 0.70,
        min_support: int = 3,
        min_expected_pnl: float = 1_000.0,
        min_confidence: float = 0.10,
        name: str = "online-ensemble",
    ) -> None:
        super().__init__()
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 < minimum_weight < maximum_weight < 1:
            raise ValueError("ensemble weight bounds are invalid")
        if min_support < 0:
            raise ValueError("min_support must be non-negative")
        self.name = name
        self.learning_rate = learning_rate
        self.minimum_weight = minimum_weight
        self.maximum_weight = maximum_weight
        self.min_support = min_support
        self.min_expected_pnl = min_expected_pnl
        self.min_confidence = min_confidence
        self._experts = {
            "flat": FlatModel(),
            "last": LastMajorityModel(),
            "rolling": RollingFrequencyModel((5, 10, 20), alpha=1.0),
            "markov": RegularisedMarkovModel(order=2, alpha=1.0),
        }
        self._weights = {
            key: 1.0 / len(self._experts) for key in self._experts
        }
        self._last_expert_forecasts: dict[str, Forecast] = {}

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def _update_weights(self, label: MajorityOutcome) -> None:
        if not self._last_expert_forecasts:
            return
        likelihoods = {
            name: max(0.05, min(0.95, forecast.probability(label)))
            for name, forecast in self._last_expert_forecasts.items()
        }
        posterior = {
            name: self._weights[name] * exp(self.learning_rate * log(likelihood))
            for name, likelihood in likelihoods.items()
        }
        self._weights = _bounded_normalize(
            posterior,
            minimum=self.minimum_weight,
            maximum=self.maximum_weight,
        )

    def decide(self, observation: AgentObservation) -> int:
        had_interval, label = self._observe_live_interval(observation)
        if had_interval and label is not None:
            # The stored forecasts were produced on the preceding decision;
            # this is the first point at which their outcome is observable.
            self._update_weights(label)
        if self._flat_before_live(observation):
            self._last_expert_forecasts = {}
            return 0

        forecasts = {
            name: expert.estimate(self._labels)
            for name, expert in self._experts.items()
        }
        self._last_expert_forecasts = forecasts
        support = max(forecast.support for forecast in forecasts.values())
        forecast = Forecast(
            sum(self._weights[name] * forecasts[name].p_long for name in forecasts),
            sum(self._weights[name] * forecasts[name].p_short for name in forecasts),
            support,
            self.name,
        )
        if forecast.support < self.min_support:
            self._last_forecast = forecast
            self._last_action = 0
            return 0
        return self._action_from_forecast(
            forecast,
            observation,
            min_expected_pnl=self.min_expected_pnl,
            min_confidence=self.min_confidence,
        )


class OnlineDriftAware(ColdStartStrategy):
    """One-way, hysteretic fallback after causal live degradation."""

    def __init__(
        self,
        *,
        primary_factory: Callable[[], ColdStartStrategy] = OnlineExpertEnsemble,
        fallback_factory: Callable[[], ColdStartStrategy] = ColdStartFlat,
        minimum_quality_observations: int = 8,
        quality_window: int = 8,
        minimum_hit_rate: float = 0.40,
        bad_streak_required: int = 2,
        name: str = "online-drift-aware",
    ) -> None:
        super().__init__()
        if minimum_quality_observations <= 0 or quality_window <= 0:
            raise ValueError("drift observation counts must be positive")
        if not 0 <= minimum_hit_rate <= 1:
            raise ValueError("minimum_hit_rate must be in [0, 1]")
        if bad_streak_required <= 0:
            raise ValueError("bad_streak_required must be positive")
        self.name = name
        self.primary = primary_factory()
        self.fallback = fallback_factory()
        self.minimum_quality_observations = minimum_quality_observations
        self.quality_window = quality_window
        self.minimum_hit_rate = minimum_hit_rate
        self.bad_streak_required = bad_streak_required
        self._quality_hits: deque[bool] = deque(maxlen=quality_window)
        self._bad_streak = 0
        self._fallback_active = False
        self._last_quality_day: int | None = None
        self._prior_primary_forecast: Forecast | None = None

    @property
    def fallback_active(self) -> bool:
        return self._fallback_active

    @property
    def quality_observations(self) -> int:
        """Count only outcomes scored against supported forecasts."""

        return len(self._quality_hits)

    def _update_quality(self, observation: AgentObservation) -> None:
        if (
            not _is_genuine_live_interval(observation)
            or observation.day == self._last_quality_day
            or self._prior_primary_forecast is None
        ):
            return
        self._last_quality_day = observation.day
        label = infer_majority_from_price_change(
            observation.previous_price_change,
            previous_move_is_reset=observation.previous_move_is_reset,
        )
        if label is None:
            return
        forecast = self._prior_primary_forecast
        minimum_support = getattr(self.primary, "min_support", 1)
        if forecast.support < minimum_support or forecast.predicted_majority is None:
            # A flat/uncertain or not-yet-supported forecast is not a failed
            # prediction. It is excluded from the degradation sample.
            return
        self._quality_hits.append(
            forecast.predicted_majority is label
        )
        if len(self._quality_hits) < self.minimum_quality_observations:
            return
        hit_rate = sum(self._quality_hits) / len(self._quality_hits)
        if hit_rate < self.minimum_hit_rate:
            self._bad_streak += 1
        else:
            self._bad_streak = 0
        if self._bad_streak >= self.bad_streak_required:
            # The fallback is deliberately sticky.  This is the hysteresis
            # rule: no daily strategy thrashing or hindsight re-selection.
            self._fallback_active = True

    def decide(self, observation: AgentObservation) -> int:
        self._update_quality(observation)
        primary_action = self.primary.decide(observation)
        fallback_action = self.fallback.decide(observation)
        self._prior_primary_forecast = self.primary.last_forecast
        self._observed_live_count = self.primary.observed_live_count
        self._labels = list(self.primary.labels)
        if self._fallback_active:
            self._last_forecast = self.fallback.last_forecast if isinstance(self.fallback, ColdStartStrategy) else Forecast(0.5, 0.5, 0, "flat")
            self._last_action = fallback_action
            return fallback_action
        self._last_forecast = self.primary.last_forecast
        self._last_action = primary_action
        return primary_action


class AsymmetricPriorStrategy(ColdStartStrategy):
    """Explicit prior benchmark for the 8k-up versus 5k-down asymmetry."""

    def __init__(
        self,
        *,
        burn_in_genuine_observations: int = 0,
        name: str = "immediate-long-prior",
    ) -> None:
        super().__init__()
        if burn_in_genuine_observations < 0:
            raise ValueError("burn-in must be non-negative")
        self.burn_in_genuine_observations = burn_in_genuine_observations
        self.name = name

    def decide(self, observation: AgentObservation) -> int:
        self._observe_live_interval(observation)
        if self._flat_before_live(observation):
            return 0
        self._last_forecast = Forecast(0.5, 0.5, self._observed_live_count, self.name)
        self._last_action = (
            1
            if self._observed_live_count >= self.burn_in_genuine_observations
            else 0
        )
        return self._last_action


def _make_model(
    model: str,
    *,
    windows: Sequence[int],
    markov_order: int,
    alpha: float,
) -> object:
    if model == "last":
        return LastMajorityModel()
    if model == "rolling":
        return RollingFrequencyModel(windows, alpha=alpha)
    if model == "markov":
        return RegularisedMarkovModel(order=markov_order, alpha=alpha)
    raise ValueError(f"unknown cold-start model: {model}")


COLD_START_STRATEGY_NAMES: tuple[str, ...] = (
    "flat",
    "always_long",
    "always_short",
    "burnin1_markov",
    "burnin3_markov",
    "burnin5_markov",
    "burnin10_markov",
    "online_last_counter",
    "online_rolling",
    "online_markov",
    "online_ensemble",
    "online_drift",
    "immediate_long_prior",
    "flat_first_long_prior",
)


def make_cold_start_strategy(name: str):
    """Construct one predeclared candidate with pristine state."""

    if name == "flat":
        return ColdStartFlat()
    if name == "always_long":
        return ColdStartFixedAction(1, "always-long")
    if name == "always_short":
        return ColdStartFixedAction(-1, "always-short")
    if name.startswith("burnin") and name.endswith("_markov"):
        burn_in = int(name[len("burnin") : -len("_markov")])
        return FlatBurnInStrategy(burn_in, model="markov")
    if name == "online_last_counter":
        return OnlineLastMajorityCounter()
    if name == "online_rolling":
        return OnlineRollingFrequency()
    if name == "online_markov":
        return OnlineRegularisedMarkov()
    if name == "online_ensemble":
        return OnlineExpertEnsemble()
    if name == "online_drift":
        return OnlineDriftAware()
    if name == "immediate_long_prior":
        return AsymmetricPriorStrategy()
    if name == "flat_first_long_prior":
        return AsymmetricPriorStrategy(
            burn_in_genuine_observations=3,
            name="flat-first-long-prior",
        )
    raise KeyError(f"unknown cold-start strategy {name!r}")


__all__ = [
    "AsymmetricPriorStrategy",
    "COLD_START_STRATEGY_NAMES",
    "ColdStartFixedAction",
    "ColdStartFlat",
    "FlatBurnInStrategy",
    "OnlineDriftAware",
    "OnlineExpertEnsemble",
    "OnlineLastMajorityCounter",
    "OnlineRegularisedMarkov",
    "OnlineRollingFrequency",
    "live_labels_from_observation",
    "make_cold_start_strategy",
]
