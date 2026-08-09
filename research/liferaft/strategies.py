"""Causal, low-capacity Liferaft research strategies for Pass 2.

The classes in this module are deliberately limited.  They consume only an
``AgentObservation`` and derive every majority label from the public price
history.  Calibration and model selection are imported lazily at the marked
boundary so that the strategy module remains usable without the experiment
runner.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from typing import Callable, Mapping, Protocol, Sequence, TypeAlias

from .simulator import (
    Agent,
    AgentObservation,
    LiferaftConfig,
    MajorityOutcome,
    infer_majority_from_price_change,
)


PublicLabel: TypeAlias = MajorityOutcome | None


@dataclass(frozen=True)
class Forecast:
    """A public-information forecast for the next non-flat majority.

    ``None`` labels are unknown/tie observations.  They are not assigned to a
    side, and ``support`` counts only observed non-zero movements.
    """

    p_long: float
    p_short: float
    support: int = 0
    source: str = ""

    def __post_init__(self) -> None:
        if not (isfinite(self.p_long) and isfinite(self.p_short)):
            raise ValueError("forecast probabilities must be finite")
        if self.p_long < 0 or self.p_short < 0:
            raise ValueError("forecast probabilities cannot be negative")
        if self.p_long + self.p_short > 1.0000001:
            raise ValueError("forecast probabilities cannot sum above one")
        if self.support < 0:
            raise ValueError("forecast support cannot be negative")

    @property
    def p_unknown(self) -> float:
        return max(0.0, 1.0 - self.p_long - self.p_short)

    @property
    def confidence(self) -> float:
        return abs(self.p_short - self.p_long)

    @property
    def predicted_majority(self) -> MajorityOutcome | None:
        if self.support <= 0 or self.p_long == self.p_short:
            return None
        return (
            MajorityOutcome.LONG
            if self.p_long > self.p_short
            else MajorityOutcome.SHORT
        )

    def probability(self, label: MajorityOutcome) -> float:
        if label is MajorityOutcome.LONG:
            return self.p_long
        if label is MajorityOutcome.SHORT:
            return self.p_short
        raise ValueError("probability is defined only for non-flat labels")


class PublicPredictor(Protocol):
    name: str

    def estimate(self, labels: Sequence[PublicLabel]) -> Forecast:
        """Estimate the next majority using only prior public labels."""


def labels_from_prices(
    prices: Sequence[int],
    *,
    marked_boundary_day: int | None = None,
) -> tuple[PublicLabel, ...]:
    """Convert public price intervals to labels, excluding the reset move.

    The returned sequence has one entry per genuine interval.  A genuine
    zero movement is retained as ``None`` so callers can distinguish an
    unknown observation from a missing/reset interval when needed.
    """

    labels: list[PublicLabel] = []
    for index in range(1, len(prices)):
        if marked_boundary_day is not None and index == marked_boundary_day:
            continue
        change = prices[index] - prices[index - 1]
        labels.append(
            infer_majority_from_price_change(
                change,
                previous_move_is_reset=False,
            )
        )
    return tuple(labels)


def labels_from_observation(observation: AgentObservation) -> tuple[PublicLabel, ...]:
    """Return all genuine public labels observable at this decision."""

    return labels_from_prices(
        observation.price_history,
        marked_boundary_day=observation.marked_boundary_day,
    )


def asymmetric_short_majority_threshold(
    *,
    long_majority_move: int = -5_000,
    short_majority_move: int = 8_000,
) -> float:
    """Return the zero-EV probability threshold for taking the long side.

    If ``q`` is the probability of a short majority, the expected P&L of a
    long is ``q*8,000 - (1-q)*5,000``.  The break-even value is therefore
    ``5,000 / (5,000 + 8,000) = 5/13`` under competition defaults.
    """

    if long_majority_move >= 0 or short_majority_move <= 0:
        raise ValueError("majority moves must have competition signs")
    return abs(long_majority_move) / (abs(long_majority_move) + short_majority_move)


def payoff_action(
    forecast: Forecast,
    *,
    long_majority_move: int = -5_000,
    short_majority_move: int = 8_000,
    min_expected_pnl: float = 1_000.0,
    min_confidence: float = 0.10,
) -> int:
    """Choose the position with the best conservative expected P&L.

    No action is taken without observed support, when the forecast is too
    uncertain, or when the best expected result does not clear the fixed
    no-trade margin.  This explicitly accounts for the asymmetric -5k/+8k
    payoff instead of optimizing classification accuracy alone.
    """

    if forecast.support <= 0 or forecast.confidence < min_confidence:
        return 0
    if long_majority_move >= 0 or short_majority_move <= 0:
        raise ValueError("majority moves must have competition signs")

    expected_long = (
        forecast.p_short * short_majority_move
        + forecast.p_long * long_majority_move
    )
    expected_short = (
        forecast.p_long * -long_majority_move
        + forecast.p_short * -short_majority_move
    )
    if max(expected_long, expected_short) < min_expected_pnl:
        return 0
    if expected_long > expected_short:
        return 1
    if expected_short > expected_long:
        return -1
    return 0


def _smoothed_forecast(
    long_count: int,
    short_count: int,
    *,
    alpha: float,
    support: int,
    source: str,
) -> Forecast:
    if alpha < 0:
        raise ValueError("smoothing alpha cannot be negative")
    if support <= 0 and long_count == 0 and short_count == 0:
        return Forecast(0.5, 0.5, 0, source)
    denominator = long_count + short_count + 2 * alpha
    if denominator == 0:
        return Forecast(0.5, 0.5, 0, source)
    return Forecast(
        (long_count + alpha) / denominator,
        (short_count + alpha) / denominator,
        support,
        source,
    )


class FlatModel:
    name = "flat"

    def estimate(self, labels: Sequence[PublicLabel]) -> Forecast:
        del labels
        return Forecast(0.5, 0.5, 0, self.name)


class LastMajorityModel:
    name = "last_majority_counter"

    def estimate(self, labels: Sequence[PublicLabel]) -> Forecast:
        for label in reversed(labels):
            if label is MajorityOutcome.LONG:
                return Forecast(1.0, 0.0, 1, self.name)
            if label is MajorityOutcome.SHORT:
                return Forecast(0.0, 1.0, 1, self.name)
        return Forecast(0.5, 0.5, 0, self.name)


class RollingFrequencyModel:
    name = "rolling_frequency"

    def __init__(
        self,
        windows: Sequence[int] = (5, 10, 20),
        *,
        alpha: float = 1.0,
    ) -> None:
        self.windows = tuple(windows)
        if not self.windows or any(window <= 0 for window in self.windows):
            raise ValueError("rolling windows must be positive and non-empty")
        self.alpha = alpha

    def estimate(self, labels: Sequence[PublicLabel]) -> Forecast:
        known = tuple(label for label in labels if label is not None)
        if not known:
            return Forecast(0.5, 0.5, 0, self.name)
        estimates: list[Forecast] = []
        for window in self.windows:
            sample = known[-window:]
            long_count = sample.count(MajorityOutcome.LONG)
            short_count = sample.count(MajorityOutcome.SHORT)
            estimates.append(
                _smoothed_forecast(
                    long_count,
                    short_count,
                    alpha=self.alpha,
                    support=len(sample),
                    source=self.name,
                )
            )
        return Forecast(
            sum(estimate.p_long for estimate in estimates) / len(estimates),
            sum(estimate.p_short for estimate in estimates) / len(estimates),
            len(known),
            self.name,
        )


class RegularisedMarkovModel:
    name = "markov"

    def __init__(self, order: int = 2, *, alpha: float = 1.0) -> None:
        if order not in (1, 2):
            raise ValueError("Pass 2 Markov order must be 1 or 2")
        if alpha <= 0:
            raise ValueError("Markov smoothing alpha must be positive")
        self.order = order
        self.alpha = alpha

    def estimate(self, labels: Sequence[PublicLabel]) -> Forecast:
        sequence = tuple(labels)
        known = tuple(label for label in sequence if label is not None)
        if not known:
            return Forecast(0.5, 0.5, 0, self.name)

        max_context = min(self.order, max(0, len(sequence) - 1))
        for context_length in range(max_context, -1, -1):
            context = sequence[-context_length:] if context_length else ()
            if context_length and any(label is None for label in context):
                # An unknown/tie observation breaks a public transition
                # context; do not bridge across it as if it never happened.
                continue
            long_count = 0
            short_count = 0
            support = 0
            for index in range(context_length, len(sequence)):
                prior = sequence[index - context_length : index]
                if sequence[index] is None or any(label is None for label in prior):
                    continue
                if prior == context:
                    support += 1
                    if sequence[index] is MajorityOutcome.LONG:
                        long_count += 1
                    else:
                        short_count += 1
            if support:
                return _smoothed_forecast(
                    long_count,
                    short_count,
                    alpha=self.alpha,
                    support=support,
                    source=self.name,
                )

        # With one observed label there is no transition yet.  A smoothed
        # marginal is safer than manufacturing a transition.
        return _smoothed_forecast(
            known.count(MajorityOutcome.LONG),
            known.count(MajorityOutcome.SHORT),
            alpha=self.alpha,
            support=len(known),
            source=self.name,
        )


def choose_replay_period(
    labels: Sequence[PublicLabel],
    *,
    periods: Sequence[int] = (2, 3, 4, 5),
    min_observations: int = 8,
    minimum_margin: float = 0.10,
) -> int | None:
    """Select a small periodic lag only when it beats a constant null."""

    known = tuple(label for label in labels if label is not None)
    if len(known) < min_observations:
        return None
    null_accuracy = max(
        known.count(MajorityOutcome.LONG), known.count(MajorityOutcome.SHORT)
    ) / len(known)
    best_period: int | None = None
    best_margin = minimum_margin
    for period in periods:
        if period <= 0:
            raise ValueError("replay periods must be positive")
        comparable = [
            index
            for index in range(period, len(labels))
            if labels[index] is not None and labels[index - period] is not None
        ]
        if len(comparable) < min_observations:
            continue
        accuracy = sum(
            labels[index] == labels[index - period] for index in comparable
        ) / len(comparable)
        margin = accuracy - null_accuracy
        if margin > best_margin:
            best_period = period
            best_margin = margin
    return best_period


class PeriodicReplayModel:
    name = "periodic_replay"

    def __init__(
        self,
        period: int | None = None,
        *,
        periods: Sequence[int] = (2, 3, 4, 5),
        evidence_margin: float = 0.10,
        min_observations: int = 8,
    ) -> None:
        self.period = period
        self.periods = tuple(periods)
        self.evidence_margin = evidence_margin
        self.min_observations = min_observations

    def estimate(self, labels: Sequence[PublicLabel]) -> Forecast:
        period = self.period
        if period is None:
            period = choose_replay_period(
                labels,
                periods=self.periods,
                min_observations=self.min_observations,
                minimum_margin=self.evidence_margin,
            )
        if period is not None and len(labels) > period:
            # The next unseen label has index len(labels); a period-p replay
            # therefore references the already observed label at len(labels)-p.
            reference = labels[-period]
            if reference is MajorityOutcome.LONG:
                return Forecast(0.90, 0.10, 1, self.name)
            if reference is MajorityOutcome.SHORT:
                return Forecast(0.10, 0.90, 1, self.name)
        # Replay has no credible target.  A smoothed marginal is useful for
        # diagnostics, but payoff_action will remain conservative on weak data.
        return RollingFrequencyModel((max(1, len(labels)),), alpha=1.0).estimate(labels)


@dataclass(frozen=True)
class CandidateSpec:
    """A predeclared calibration candidate and its action convention."""

    name: str
    model_factory: Callable[[], PublicPredictor]
    action_mode: str = "counter"
    fixed_action: int = 0
    complexity: int = 1

    def action(
        self,
        forecast: Forecast,
        *,
        long_majority_move: int,
        short_majority_move: int,
        min_expected_pnl: float,
        min_confidence: float,
    ) -> int:
        if self.action_mode == "fixed":
            return self.fixed_action
        return payoff_action(
            forecast,
            long_majority_move=long_majority_move,
            short_majority_move=short_majority_move,
            min_expected_pnl=min_expected_pnl,
            min_confidence=min_confidence,
        )


def candidate_specs() -> tuple[CandidateSpec, ...]:
    """Return the small fixed model list used by calibration."""

    return (
        CandidateSpec("flat", FlatModel, "fixed", 0, 0),
        CandidateSpec("always_long", FlatModel, "fixed", 1, 0),
        CandidateSpec("always_short", FlatModel, "fixed", -1, 0),
        CandidateSpec("last_majority_counter", LastMajorityModel, complexity=1),
        CandidateSpec(
            "rolling_frequency",
            RollingFrequencyModel,
            complexity=2,
        ),
        CandidateSpec("markov", RegularisedMarkovModel, complexity=3),
        CandidateSpec("periodic_replay", PeriodicReplayModel, complexity=4),
    )


def model_for_name(name: str, *, replay_period: int | None = None) -> PublicPredictor:
    """Construct one of the predeclared public-information models."""

    if name == "periodic_replay":
        return PeriodicReplayModel(period=replay_period)
    for spec in candidate_specs():
        if spec.name == name:
            return spec.model_factory()
    raise ValueError(f"unknown Liferaft Pass 2 model: {name}")


class FlatBenchmark:
    def __init__(self, name: str = "focal") -> None:
        self.name = name
        self.last_forecast: Forecast | None = None

    def decide(self, observation: AgentObservation) -> int:
        del observation
        self.last_forecast = None
        return 0


class FixedActionStrategy:
    def __init__(self, action: int, name: str = "focal") -> None:
        if action not in (-1, 0, 1):
            raise ValueError("fixed Liferaft actions must be -1, 0, or 1")
        self.name = name
        self.action = action
        self.last_forecast: Forecast | None = None

    def decide(self, observation: AgentObservation) -> int:
        del observation
        self.last_forecast = None
        return self.action


class _ModelStrategy:
    def __init__(
        self,
        model: PublicPredictor,
        *,
        name: str,
        min_expected_pnl: float = 1_000.0,
        min_confidence: float = 0.10,
    ) -> None:
        self.name = name
        self.model = model
        self.min_expected_pnl = min_expected_pnl
        self.min_confidence = min_confidence
        self.last_forecast: Forecast | None = None

    def _decide_from_model(self, observation: AgentObservation) -> int:
        forecast = self.model.estimate(labels_from_observation(observation))
        self.last_forecast = forecast
        return payoff_action(
            forecast,
            long_majority_move=observation.long_majority_move,
            short_majority_move=observation.short_majority_move,
            min_expected_pnl=self.min_expected_pnl,
            min_confidence=self.min_confidence,
        )

    def decide(self, observation: AgentObservation) -> int:
        return self._decide_from_model(observation)


class LastMajorityCounterStrategy(_ModelStrategy):
    def __init__(self, name: str = "focal", **kwargs: float) -> None:
        super().__init__(LastMajorityModel(), name=name, **kwargs)


class RollingFrequencyStrategy(_ModelStrategy):
    def __init__(
        self,
        name: str = "focal",
        *,
        windows: Sequence[int] = (5, 10, 20),
        **kwargs: float,
    ) -> None:
        super().__init__(
            RollingFrequencyModel(windows),
            name=name,
            **kwargs,
        )


class RegularisedMarkovStrategy(_ModelStrategy):
    def __init__(
        self,
        name: str = "focal",
        *,
        order: int = 2,
        **kwargs: float,
    ) -> None:
        super().__init__(RegularisedMarkovModel(order), name=name, **kwargs)


class PeriodicReplayStrategy:
    """Use a Year-1-selected replay lag, then causally gate it in Year 2."""

    def __init__(
        self,
        name: str = "focal",
        *,
        marked_boundary_day: int = 365,
        periods: Sequence[int] = (2, 3, 4, 5),
        evidence_margin: float = 0.10,
        min_observations: int = 8,
        min_replay_agreement: float = 0.60,
        min_expected_pnl: float = 1_000.0,
        min_confidence: float = 0.10,
    ) -> None:
        self.name = name
        self.marked_boundary_day = marked_boundary_day
        self.periods = tuple(periods)
        self.evidence_margin = evidence_margin
        self.min_observations = min_observations
        self.min_replay_agreement = min_replay_agreement
        self.min_expected_pnl = min_expected_pnl
        self.min_confidence = min_confidence
        self.selected_period: int | None = None
        self.replay_disabled = False
        self._frozen = False
        self._year1_labels: tuple[PublicLabel, ...] = ()
        self.last_forecast: Forecast | None = None

    def _freeze_at_boundary(self, observation: AgentObservation) -> None:
        boundary = observation.marked_boundary_day
        self._year1_labels = labels_from_prices(
            observation.price_history[:boundary],
            marked_boundary_day=boundary,
        )
        self.selected_period = choose_replay_period(
            self._year1_labels,
            periods=self.periods,
            min_observations=self.min_observations,
            minimum_margin=self.evidence_margin,
        )
        self._frozen = True

    def _replay_matches_prefix(self, year2_labels: Sequence[PublicLabel]) -> bool:
        if len(year2_labels) < self.min_observations or not self._year1_labels:
            return True
        comparisons = [
            index
            for index, label in enumerate(year2_labels)
            if label is not None
            and self._year1_labels[index % len(self._year1_labels)] is not None
        ]
        if not comparisons:
            return True
        agreement = sum(
            year2_labels[index]
            == self._year1_labels[index % len(self._year1_labels)]
            for index in comparisons
        ) / len(comparisons)
        return agreement >= self.min_replay_agreement

    def decide(self, observation: AgentObservation) -> int:
        if not self._frozen and observation.day >= observation.marked_boundary_day:
            self._freeze_at_boundary(observation)
        if observation.day < observation.marked_boundary_day:
            self.last_forecast = None
            return 0

        all_labels = labels_from_observation(observation)
        year2_start = len(self._year1_labels)
        year2_labels = all_labels[year2_start:]
        if not self._replay_matches_prefix(year2_labels):
            self.replay_disabled = True
        if self.replay_disabled or self.selected_period is None or not self._year1_labels:
            self.last_forecast = Forecast(0.5, 0.5, 0, "periodic_replay")
            return 0

        reference_index = len(year2_labels) % len(self._year1_labels)
        reference = self._year1_labels[reference_index]
        if reference is MajorityOutcome.LONG:
            forecast = Forecast(0.90, 0.10, 1, "periodic_replay")
        elif reference is MajorityOutcome.SHORT:
            forecast = Forecast(0.10, 0.90, 1, "periodic_replay")
        else:
            forecast = Forecast(0.5, 0.5, 0, "periodic_replay")
        self.last_forecast = forecast
        return payoff_action(
            forecast,
            long_majority_move=observation.long_majority_move,
            short_majority_move=observation.short_majority_move,
            min_expected_pnl=self.min_expected_pnl,
            min_confidence=self.min_confidence,
        )


def _config_from_observation(observation: AgentObservation) -> LiferaftConfig:
    """Build mechanics for calibration without exposing engine diagnostics."""

    initial = observation.price_history[0] if observation.price_history else observation.price
    return LiferaftConfig(
        total_days=max(observation.marked_boundary_day + 1, len(observation.price_history)),
        marked_boundary_day=observation.marked_boundary_day,
        initial_price=initial,
        reset_price=observation.price,
        price_floor=observation.price_floor,
        long_majority_move=observation.long_majority_move,
        short_majority_move=observation.short_majority_move,
        position_limit=observation.position_limit,
        gross_portfolio_budget=observation.gross_portfolio_budget,
    )


class Year1CalibratedStrategy:
    """Select one predeclared candidate from chronological Year-1 evidence."""

    def __init__(
        self,
        name: str = "focal",
        *,
        candidate_names: Sequence[str] | None = None,
        warmup_days: int = 30,
        min_improvement: float = 5_000.0,
    ) -> None:
        self.name = name
        self.candidate_names = tuple(candidate_names) if candidate_names else None
        self.warmup_days = warmup_days
        self.min_improvement = min_improvement
        self.selected_name: str | None = None
        self.selection_result = None
        self._delegate: Agent | None = None
        self._frozen = False
        self.last_forecast: Forecast | None = None

    def _freeze_at_boundary(self, observation: AgentObservation) -> None:
        from .calibration import select_candidate_from_year1

        config = _config_from_observation(observation)
        boundary = observation.marked_boundary_day
        year1_prices = observation.price_history[:boundary]
        self.selection_result = select_candidate_from_year1(
            year1_prices,
            boundary_day=boundary,
            config=config,
            candidate_names=self.candidate_names,
            warmup_days=self.warmup_days,
            minimum_improvement=self.min_improvement,
        )
        self.selected_name = self.selection_result.selected_name
        self._delegate = strategy_from_name(
            self.selected_name,
            agent_name=self.name,
            marked_boundary_day=boundary,
        )
        self._frozen = True

    def decide(self, observation: AgentObservation) -> int:
        if not self._frozen and observation.day >= observation.marked_boundary_day:
            self._freeze_at_boundary(observation)
        if observation.day < observation.marked_boundary_day or self._delegate is None:
            self.last_forecast = None
            return 0
        action = self._delegate.decide(observation)
        self.last_forecast = getattr(self._delegate, "last_forecast", None)
        return action


class SmallExpertEnsembleStrategy:
    """A fixed small ensemble whose Year-2 weights are frozen at the boundary."""

    DEFAULT_EXPERTS = (
        "last_majority_counter",
        "rolling_frequency",
        "markov",
        "periodic_replay",
    )

    def __init__(
        self,
        name: str = "focal",
        *,
        expert_names: Sequence[str] = DEFAULT_EXPERTS,
        warmup_days: int = 30,
        min_expected_pnl: float = 1_000.0,
        min_confidence: float = 0.10,
    ) -> None:
        self.name = name
        self.expert_names = tuple(expert_names)
        self.warmup_days = warmup_days
        self.min_expected_pnl = min_expected_pnl
        self.min_confidence = min_confidence
        self.weights: dict[str, float] = {}
        self.models: dict[str, PublicPredictor] = {}
        self.selection_result = None
        self._year1_labels: tuple[PublicLabel, ...] = ()
        self._frozen = False
        self.last_forecast: Forecast | None = None

    def _freeze_at_boundary(self, observation: AgentObservation) -> None:
        from .calibration import fit_ensemble_weights, select_candidate_from_year1

        config = _config_from_observation(observation)
        boundary = observation.marked_boundary_day
        year1_prices = observation.price_history[:boundary]
        self.selection_result = select_candidate_from_year1(
            year1_prices,
            boundary_day=boundary,
            config=config,
            candidate_names=("flat",) + self.expert_names,
            warmup_days=self.warmup_days,
        )
        self.weights = fit_ensemble_weights(
            self.selection_result.scores,
            expert_names=self.expert_names,
        )
        self._year1_labels = labels_from_prices(
            year1_prices,
            marked_boundary_day=boundary,
        )
        for expert_name in self.expert_names:
            period = (
                choose_replay_period(self._year1_labels)
                if expert_name == "periodic_replay"
                else None
            )
            self.models[expert_name] = model_for_name(
                expert_name,
                replay_period=period,
            )
        self.models["flat"] = FlatModel()
        self._frozen = True

    def _ensemble_forecast(
        self,
        labels: Sequence[PublicLabel],
    ) -> Forecast:
        if not self.weights:
            return Forecast(0.5, 0.5, 0, "ensemble")
        p_long = 0.0
        p_short = 0.0
        support = 0.0
        for expert_name, weight in self.weights.items():
            model = self.models.get(expert_name)
            if model is None:
                continue
            forecast = model.estimate(labels)
            p_long += weight * forecast.p_long
            p_short += weight * forecast.p_short
            support += weight * forecast.support
        return Forecast(
            min(1.0, max(0.0, p_long)),
            min(1.0, max(0.0, p_short)),
            int(round(support)),
            "ensemble",
        )

    def decide(self, observation: AgentObservation) -> int:
        if not self._frozen and observation.day >= observation.marked_boundary_day:
            self._freeze_at_boundary(observation)
        if observation.day < observation.marked_boundary_day:
            self.last_forecast = None
            return 0
        forecast = self._ensemble_forecast(labels_from_observation(observation))
        self.last_forecast = forecast
        return payoff_action(
            forecast,
            long_majority_move=observation.long_majority_move,
            short_majority_move=observation.short_majority_move,
            min_expected_pnl=self.min_expected_pnl,
            min_confidence=self.min_confidence,
        )


class DriftAwareStrategy:
    """Freeze a Year-1 choice, then fall back once causal quality degrades."""

    def __init__(
        self,
        name: str = "focal",
        *,
        candidate_names: Sequence[str] = (
            "last_majority_counter",
            "rolling_frequency",
            "markov",
            "periodic_replay",
        ),
        warmup_days: int = 30,
        quality_window: int = 12,
        minimum_quality_observations: int = 12,
        degradation_hit_rate: float = 0.40,
        degradation_streak: int = 3,
        min_expected_pnl: float = 1_000.0,
    ) -> None:
        if quality_window <= 0 or minimum_quality_observations <= 0:
            raise ValueError("drift quality windows must be positive")
        if not 0 <= degradation_hit_rate <= 1:
            raise ValueError("degradation_hit_rate must be in [0, 1]")
        self.name = name
        self.candidate_names = tuple(candidate_names)
        self.warmup_days = warmup_days
        self.quality_window = quality_window
        self.minimum_quality_observations = minimum_quality_observations
        self.degradation_hit_rate = degradation_hit_rate
        self.degradation_streak = degradation_streak
        self.min_expected_pnl = min_expected_pnl
        self.selected_name: str | None = None
        self.selection_result = None
        self.primary_model: PublicPredictor | None = None
        self.fallback = SmallExpertEnsembleStrategy(
            name=f"{name}-fallback",
            warmup_days=warmup_days,
            min_expected_pnl=min_expected_pnl,
        )
        self._quality: deque[bool] = deque(maxlen=quality_window)
        self._bad_streak = 0
        self._degraded = False
        self._frozen = False
        self.last_forecast: Forecast | None = None
        self.quality_observations = 0

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def quality_history(self) -> tuple[bool, ...]:
        return tuple(self._quality)

    def _freeze_at_boundary(self, observation: AgentObservation) -> None:
        from .calibration import select_candidate_from_year1

        config = _config_from_observation(observation)
        boundary = observation.marked_boundary_day
        year1_prices = observation.price_history[:boundary]
        self.selection_result = select_candidate_from_year1(
            year1_prices,
            boundary_day=boundary,
            config=config,
            candidate_names=self.candidate_names,
            warmup_days=self.warmup_days,
        )
        self.selected_name = self.selection_result.selected_name
        year1_labels = labels_from_prices(
            year1_prices,
            marked_boundary_day=boundary,
        )
        period = (
            choose_replay_period(year1_labels)
            if self.selected_name == "periodic_replay"
            else None
        )
        self.primary_model = model_for_name(
            self.selected_name,
            replay_period=period,
        )
        # Calling the fallback at the boundary initializes its own frozen
        # weights using exactly the same public Year-1 prefix.
        self.fallback.decide(observation)
        self._frozen = True

    def _observe_previous_outcome(self, observation: AgentObservation) -> None:
        if self.last_forecast is None:
            return
        if observation.day <= observation.marked_boundary_day:
            return
        if observation.previous_move_is_reset:
            return
        actual = observation.previous_inferred_majority
        if actual is None:
            return
        predicted = self.last_forecast.predicted_majority
        if predicted is None:
            return
        hit = predicted is actual
        self._quality.append(hit)
        self.quality_observations += 1
        if len(self._quality) >= self.minimum_quality_observations:
            hit_rate = sum(self._quality) / len(self._quality)
            if hit_rate < self.degradation_hit_rate:
                self._bad_streak += 1
            else:
                self._bad_streak = 0
            if self._bad_streak >= self.degradation_streak:
                # Hysteresis is intentionally one-way for this run: once a
                # frozen policy is judged degraded, it does not thrash daily.
                self._degraded = True

    def decide(self, observation: AgentObservation) -> int:
        if not self._frozen and observation.day >= observation.marked_boundary_day:
            self._freeze_at_boundary(observation)
        self._observe_previous_outcome(observation)
        if observation.day < observation.marked_boundary_day:
            self.last_forecast = None
            return 0

        fallback_action = self.fallback.decide(observation)
        if self._degraded or self.primary_model is None:
            self.last_forecast = self.fallback.last_forecast
            return fallback_action

        forecast = self.primary_model.estimate(labels_from_observation(observation))
        self.last_forecast = forecast
        return payoff_action(
            forecast,
            long_majority_move=observation.long_majority_move,
            short_majority_move=observation.short_majority_move,
            min_expected_pnl=self.min_expected_pnl,
            min_confidence=0.10,
        )


def strategy_from_name(
    name: str,
    *,
    agent_name: str = "focal",
    marked_boundary_day: int = 365,
    **kwargs: object,
) -> Agent:
    """Construct a named candidate with a stable public interface."""

    if name == "flat":
        return FlatBenchmark(agent_name)
    if name == "always_long":
        return FixedActionStrategy(1, agent_name)
    if name == "always_short":
        return FixedActionStrategy(-1, agent_name)
    if name == "last_majority_counter":
        return LastMajorityCounterStrategy(agent_name, **kwargs)
    if name == "rolling_frequency":
        return RollingFrequencyStrategy(agent_name, **kwargs)
    if name == "markov":
        return RegularisedMarkovStrategy(agent_name, **kwargs)
    if name == "periodic_replay":
        return PeriodicReplayStrategy(
            agent_name,
            marked_boundary_day=marked_boundary_day,
            **kwargs,
        )
    if name in {"ensemble", "small_expert_ensemble"}:
        return SmallExpertEnsembleStrategy(agent_name, **kwargs)
    if name in {"drift", "drift_aware"}:
        return DriftAwareStrategy(agent_name, **kwargs)
    if name in {"year1_selected", "calibrated"}:
        return Year1CalibratedStrategy(agent_name, **kwargs)
    raise ValueError(f"unknown Liferaft Pass 2 strategy: {name}")


# A few descriptive aliases make reports and downstream experiments easier to
# read without creating a second implementation.
AlwaysFlatStrategy = FlatBenchmark


class AlwaysLongStrategy(FixedActionStrategy):
    def __init__(self, name: str = "focal") -> None:
        super().__init__(1, name)


class AlwaysShortStrategy(FixedActionStrategy):
    def __init__(self, name: str = "focal") -> None:
        super().__init__(-1, name)


MarkovStrategy = RegularisedMarkovStrategy


__all__ = [
    "AlwaysFlatStrategy",
    "AlwaysLongStrategy",
    "AlwaysShortStrategy",
    "CandidateSpec",
    "DriftAwareStrategy",
    "FlatBenchmark",
    "FlatModel",
    "FixedActionStrategy",
    "Forecast",
    "LastMajorityCounterStrategy",
    "LastMajorityModel",
    "MarkovStrategy",
    "PeriodicReplayModel",
    "PeriodicReplayStrategy",
    "PublicLabel",
    "PublicPredictor",
    "RegularisedMarkovModel",
    "RegularisedMarkovStrategy",
    "RollingFrequencyModel",
    "RollingFrequencyStrategy",
    "SmallExpertEnsembleStrategy",
    "Year1CalibratedStrategy",
    "asymmetric_short_majority_threshold",
    "candidate_specs",
    "choose_replay_period",
    "labels_from_observation",
    "labels_from_prices",
    "model_for_name",
    "payoff_action",
    "strategy_from_name",
]
