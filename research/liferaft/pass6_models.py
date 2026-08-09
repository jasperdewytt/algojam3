"""Frozen causal sequence models and the Pass 6A Fixed-Share master.

This module is deliberately independent of the supplied competition files.  It
contains only public-history models: a binary outcome is ``0`` for a long
majority (a negative price move), ``1`` for a short majority (a positive move),
and ``None`` is a context break.  The master scores the position chosen before
an outcome is observed and updates its weights afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, lgamma, log, log1p
from typing import Iterable, Mapping, Protocol, Sequence, TypeAlias


# Competition economics and the frozen online-learning constants.
LONG_MOVE = -5_000
SHORT_MOVE = 8_000
NEUTRAL_SHORT_PROBABILITY = 5 / 13
REWARD_BOUND = max(abs(LONG_MOVE), abs(SHORT_MOVE))
LIVE_HORIZON = 365
FIXED_SHARE_ETA = (2 * log(7) / LIVE_HORIZON) ** 0.5
FIXED_SHARE_RATE = 2 / (LIVE_HORIZON - 1)
CTW_MAX_DEPTH = 6


OutcomeBit: TypeAlias = int
HistoryToken: TypeAlias = int | None


def _validate_bit(bit: int) -> None:
    if type(bit) is not int or bit not in (0, 1):
        raise ValueError(f"binary outcome must be 0 or 1, got {bit!r}")


def _validate_token(token: HistoryToken) -> None:
    if token is not None:
        _validate_bit(token)


def _preferred_position(q_short: float, *, valid: bool) -> int:
    """Trade the economically preferred side for a short-majority forecast.

    A long position earns the positive move and loses the negative move, so
    long is preferred exactly when ``q_short > 5/13``.  A model without a
    valid context abstains rather than manufacturing a direction.
    """

    if not valid:
        return 0
    if q_short > NEUTRAL_SHORT_PROBABILITY:
        return 1
    if q_short < NEUTRAL_SHORT_PROBABILITY:
        return -1
    return 0


@dataclass(frozen=True)
class ExpertForecast:
    """One expert's pre-outcome probability and economic recommendation."""

    name: str
    p_short: float
    support: int
    valid: bool
    source: str

    def __post_init__(self) -> None:
        if not isfinite(self.p_short) or not 0 <= self.p_short <= 1:
            raise ValueError("expert probability must lie in [0, 1]")
        if self.support < 0:
            raise ValueError("expert support cannot be negative")

    @property
    def p_long(self) -> float:
        return 1.0 - self.p_short

    @property
    def proposed_position(self) -> int:
        return _preferred_position(self.p_short, valid=self.valid)

    @property
    def expected_long_reward(self) -> float:
        return self.p_short * SHORT_MOVE + self.p_long * LONG_MOVE

    @property
    def expected_preferred_reward(self) -> float:
        position = self.proposed_position
        if position == 0:
            return 0.0
        return position * self.expected_long_reward

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "p_short": self.p_short,
            "p_long": self.p_long,
            "support": self.support,
            "valid": self.valid,
            "source": self.source,
            "proposed_position": self.proposed_position,
            "expected_long_reward": self.expected_long_reward,
            "expected_preferred_reward": self.expected_preferred_reward,
        }


class CausalExpert(Protocol):
    name: str

    def reset(self) -> None:
        """Clear current context without changing frozen parameters."""

    def observe(self, token: HistoryToken) -> None:
        """Consume one newly observable token, or a context break."""

    def forecast(self) -> ExpertForecast:
        """Forecast the next token using only already observed tokens."""


class FlatExpert:
    name = "flat"

    def reset(self) -> None:
        return None

    def observe(self, token: HistoryToken) -> None:
        _validate_token(token)

    def forecast(self) -> ExpertForecast:
        # Neutral q* makes flat a genuine zero-payoff expert and gives it no
        # artificial long bias inside the aggregate master forecast.
        return ExpertForecast(
            self.name,
            NEUTRAL_SHORT_PROBABILITY,
            0,
            False,
            "flat-zero-payoff",
        )


class OrderZeroFrequencyExpert:
    """Beta(1, 1) frequency estimate using all prior scoreable outcomes."""

    name = "order_zero_frequency"

    def __init__(self) -> None:
        self._short_count = 0
        self._long_count = 0

    def reset(self) -> None:
        # A reset is a context break, not missing Year-1 training.  The
        # order-zero expert is explicitly allowed to retain prior scoreable
        # frequencies, so this method intentionally leaves its counts intact.
        return None

    def observe(self, token: HistoryToken) -> None:
        _validate_token(token)
        if token == 1:
            self._short_count += 1
        elif token == 0:
            self._long_count += 1

    def forecast(self) -> ExpertForecast:
        support = self._short_count + self._long_count
        # Beta(1, 1) posterior mean for P(short-majority).
        p_short = (self._short_count + 1) / (support + 2)
        return ExpertForecast(
            self.name,
            p_short,
            support,
            support > 0,
            "beta_bernoulli_beta_1_1",
        )


class MarkovExpert:
    """Order-one or order-two add-one Markov expert with hard breaks."""

    def __init__(self, order: int) -> None:
        if order not in (1, 2):
            raise ValueError("Pass 6 Markov order must be 1 or 2")
        self.order = order
        self.name = f"markov_order_{order}"
        self._tail: list[int] = []
        # Each context maps to [long count, short count].  Counts never cross
        # a None break because only the current tail is used as a context.
        self._counts: dict[tuple[int, ...], list[int]] = {}
        self._total = 0

    def reset(self) -> None:
        self._tail.clear()

    def observe(self, token: HistoryToken) -> None:
        _validate_token(token)
        if token is None:
            self.reset()
            return
        for depth in range(min(self.order, len(self._tail)) + 1):
            context = tuple(self._tail[-depth:]) if depth else ()
            counts = self._counts.setdefault(context, [0, 0])
            counts[token] += 1
        self._total += 1
        self._tail.append(token)
        if len(self._tail) > self.order:
            del self._tail[0]

    def forecast(self) -> ExpertForecast:
        if not self._tail or self._total == 0:
            return ExpertForecast(
                self.name,
                0.5,
                0,
                False,
                f"markov_order_{self.order}_no_context",
            )

        maximum_depth = min(self.order, len(self._tail))
        # Back off only to counts collected within a valid contiguous
        # transition context.  No token before a break appears in ``_tail``.
        for depth in range(maximum_depth, -1, -1):
            context = tuple(self._tail[-depth:]) if depth else ()
            counts = self._counts.get(context)
            if counts is None:
                continue
            support = counts[0] + counts[1]
            if support <= 0:
                continue
            p_short = (counts[1] + 1) / (support + 2)
            return ExpertForecast(
                self.name,
                p_short,
                support,
                True,
                f"markov_order_{self.order}_add_one_depth_{depth}",
            )

        return ExpertForecast(
            self.name,
            0.5,
            0,
            False,
            f"markov_order_{self.order}_unsupported_context",
        )


def _log_kt_probability(counts: Sequence[int]) -> float:
    """Log Krichevsky-Trofimov probability for a binary count pair."""

    count_long, count_short = counts
    total = count_long + count_short
    if total == 0:
        return 0.0
    return (
        lgamma(count_long + 0.5)
        + lgamma(count_short + 0.5)
        - lgamma(total + 1.0)
        - 2.0 * lgamma(0.5)
    )


def _log_add_exp(left: float, right: float) -> float:
    if left == float("-inf"):
        return right
    if right == float("-inf"):
        return left
    largest = max(left, right)
    return largest + log1p(exp(min(left, right) - largest))


class BinaryCTW:
    """Incremental binary context-tree weighting predictor.

    Every observed bit updates the KT estimator at the root and at all
    suffix-context nodes up to ``max_depth``.  A node's weighted probability
    is the equal mixture of its KT probability and the product of its two
    child weighted probabilities.  Prediction is the ratio of the weighted
    sequence probability after a hypothetical bit to the current weighted
    sequence probability.  This is a genuine suffix-tree mixture, not a
    selected Markov order.
    """

    def __init__(self, max_depth: int = CTW_MAX_DEPTH) -> None:
        if max_depth < 0:
            raise ValueError("CTW maximum depth must be non-negative")
        self.max_depth = max_depth
        self._counts: dict[tuple[int, ...], list[int]] = {}
        self._log_weight: dict[tuple[int, ...], float] = {}
        self._history_tail: list[int] = []

    @property
    def support(self) -> int:
        root = self._counts.get(())
        return 0 if root is None else root[0] + root[1]

    @property
    def sequence_length(self) -> int:
        root = self._counts.get(())
        return 0 if root is None else root[0] + root[1]

    @property
    def context_tail(self) -> tuple[int, ...]:
        return tuple(self._history_tail)

    def reset(self) -> None:
        self._counts.clear()
        self._log_weight.clear()
        self._history_tail.clear()

    def _contexts_for_current_history(self) -> list[tuple[int, ...]]:
        contexts: list[tuple[int, ...]] = [()]
        for depth in range(1, min(self.max_depth, len(self._history_tail)) + 1):
            contexts.append(tuple(self._history_tail[-depth:]))
        return contexts

    def _cached_log_weight(self, context: tuple[int, ...]) -> float:
        return self._log_weight.get(context, 0.0)

    def _recompute_node(self, context: tuple[int, ...]) -> None:
        counts = self._counts.get(context, [0, 0])
        log_kt = _log_kt_probability(counts)
        if len(context) >= self.max_depth:
            self._log_weight[context] = log_kt
            return
        child_zero = (0,) + context
        child_one = (1,) + context
        child_product = self._cached_log_weight(child_zero) + self._cached_log_weight(child_one)
        self._log_weight[context] = -log(2.0) + _log_add_exp(log_kt, child_product)

    def _apply_count_change(self, token: int, amount: int) -> list[tuple[int, ...]]:
        _validate_bit(token)
        contexts = self._contexts_for_current_history()
        for context in contexts:
            counts = self._counts.get(context)
            if counts is None:
                if amount < 0:
                    raise RuntimeError("cannot undo a missing CTW node")
                counts = [0, 0]
                self._counts[context] = counts
            counts[token] += amount
            if counts[token] < 0:
                raise RuntimeError("CTW count underflow")
        # A child is recomputed before its parent.  The path's suffix nodes
        # are also exactly the ancestors in the suffix-tree representation.
        for context in reversed(contexts):
            self._recompute_node(context)
        if amount < 0:
            for context in contexts:
                counts = self._counts[context]
                if counts == [0, 0]:
                    del self._counts[context]
                    self._log_weight.pop(context, None)
        return contexts

    def observe(self, token: int) -> None:
        self._apply_count_change(token, 1)
        self._history_tail.append(token)
        if len(self._history_tail) > self.max_depth:
            del self._history_tail[0]

    def _hypothetical_log_probability(self, token: int) -> float:
        contexts = self._apply_count_change(token, 1)
        after = self._cached_log_weight(())
        self._apply_count_change(token, -1)
        # ``_apply_count_change`` uses the same current history and restores
        # all affected node probabilities.  Keep the returned path for an
        # explicit sanity check in debugging without exposing mutable state.
        del contexts
        return after

    def _raw_next_probabilities(self) -> tuple[float, float]:
        before = self._cached_log_weight(())
        raw = tuple(
            exp(self._hypothetical_log_probability(token) - before)
            for token in (0, 1)
        )
        if any(not isfinite(value) or value < 0 for value in raw):
            raise FloatingPointError("non-finite CTW next probability")
        total = raw[0] + raw[1]
        if not isfinite(total) or total <= 0:
            raise FloatingPointError("invalid CTW probability normalisation")
        return raw[0] / total, raw[1] / total

    def next_probability(self, token: int) -> float:
        _validate_bit(token)
        return self._raw_next_probabilities()[token]

    def predict_short_probability(self) -> float:
        # The raw CTW ratios can reflect the finite-context initialisation
        # convention and need not sum to one for the first few symbols.  The
        # normalized ratio is the predictive probability exposed to the
        # trading expert; it remains a genuine mixture of the KT suffix-tree
        # sequence probabilities.
        return self._raw_next_probabilities()[1]

    def root_kt_probability(self) -> float:
        return exp(_log_kt_probability(self._counts.get((), [0, 0])))

    def root_weighted_probability(self) -> float:
        return exp(self._cached_log_weight(()))


class ContextTreeWeightingExpert:
    name = "context_tree_weighting"

    def __init__(self, max_depth: int = CTW_MAX_DEPTH) -> None:
        self.max_depth = max_depth
        self._ctw = BinaryCTW(max_depth=max_depth)

    @property
    def ctw(self) -> BinaryCTW:
        return self._ctw

    def reset(self) -> None:
        self._ctw.reset()

    def observe(self, token: HistoryToken) -> None:
        _validate_token(token)
        if token is None:
            self.reset()
        else:
            self._ctw.observe(token)

    def forecast(self) -> ExpertForecast:
        support = self._ctw.sequence_length
        if support == 0:
            return ExpertForecast(
                self.name,
                0.5,
                0,
                False,
                f"binary_ctw_kt_depth_{self.max_depth}_no_context",
            )
        return ExpertForecast(
            self.name,
            self._ctw.predict_short_probability(),
            support,
            True,
            f"binary_ctw_kt_suffix_mix_depth_{self.max_depth}",
        )


class PersistenceExpert:
    name = "persistence"

    def __init__(self) -> None:
        self._last: int | None = None

    def reset(self) -> None:
        self._last = None

    def observe(self, token: HistoryToken) -> None:
        _validate_token(token)
        self._last = token

    def forecast(self) -> ExpertForecast:
        if self._last is None:
            return ExpertForecast(self.name, 0.5, 0, False, "persistence_no_context")
        return ExpertForecast(
            self.name,
            float(self._last),
            1,
            True,
            "persistence_last_binary_outcome",
        )


class ReversalExpert:
    name = "reversal"

    def __init__(self) -> None:
        self._last: int | None = None

    def reset(self) -> None:
        self._last = None

    def observe(self, token: HistoryToken) -> None:
        _validate_token(token)
        self._last = token

    def forecast(self) -> ExpertForecast:
        if self._last is None:
            return ExpertForecast(self.name, 0.5, 0, False, "reversal_no_context")
        return ExpertForecast(
            self.name,
            float(1 - self._last),
            1,
            True,
            "reversal_opposite_binary_outcome",
        )


EXPERT_NAMES: tuple[str, ...] = (
    "flat",
    "order_zero_frequency",
    "markov_order_1",
    "markov_order_2",
    "context_tree_weighting",
    "persistence",
    "reversal",
)


def make_experts() -> tuple[CausalExpert, ...]:
    experts: tuple[CausalExpert, ...] = (
        FlatExpert(),
        OrderZeroFrequencyExpert(),
        MarkovExpert(1),
        MarkovExpert(2),
        ContextTreeWeightingExpert(CTW_MAX_DEPTH),
        PersistenceExpert(),
        ReversalExpert(),
    )
    if tuple(expert.name for expert in experts) != EXPERT_NAMES:
        raise AssertionError("Pass 6 expert order changed")
    return experts


@dataclass(frozen=True)
class MasterProposal:
    """The complete deterministic pre-outcome master proposal."""

    weights: dict[str, float]
    forecasts: dict[str, ExpertForecast]
    expert_positions: dict[str, int]
    master_short_probability: float
    master_position: int
    predicted_economic_edge: float

    def as_dict(self) -> dict[str, object]:
        return {
            "weights": dict(self.weights),
            "forecasts": {
                name: forecast.as_dict()
                for name, forecast in self.forecasts.items()
            },
            "expert_positions": dict(self.expert_positions),
            "master_short_probability": self.master_short_probability,
            "master_position": self.master_position,
            "predicted_economic_edge": self.predicted_economic_edge,
        }


@dataclass(frozen=True)
class OutcomeScore:
    """Result of scoring a prior proposal after its outcome is visible."""

    scoreable: bool
    movement_kind: str
    price_change: int | None
    prior_master_position: int
    prior_expert_positions: dict[str, int]
    raw_master_reward: int
    master_reward: int
    expert_rewards: dict[str, int]
    weights_before: dict[str, float]
    weights_after: dict[str, float]


class FixedShareMaster:
    """One frozen reward-based Fixed-Share master over ``EXPERT_NAMES``."""

    def __init__(
        self,
        *,
        eta: float = FIXED_SHARE_ETA,
        share_rate: float = FIXED_SHARE_RATE,
        reward_bound: int = REWARD_BOUND,
    ) -> None:
        if eta <= 0 or not isfinite(eta):
            raise ValueError("Fixed-Share eta must be positive and finite")
        if not 0 <= share_rate <= 1:
            raise ValueError("Fixed-Share share rate must be in [0, 1]")
        if reward_bound <= 0:
            raise ValueError("Fixed-Share reward bound must be positive")
        self.eta = eta
        self.share_rate = share_rate
        self.reward_bound = reward_bound
        self.experts = make_experts()
        self.weights: dict[str, float] = {
            name: 1.0 / len(self.experts) for name in EXPERT_NAMES
        }
        self.last_proposal: MasterProposal | None = None
        self.expert_cumulative_rewards: dict[str, int] = {
            name: 0 for name in EXPERT_NAMES
        }
        self.master_shadow_pnl = 0
        self.master_raw_shadow_pnl = 0
        self.scoreable_count = 0
        self.reward_history: list[dict[str, object]] = []
        self.weight_history: list[dict[str, float]] = []

    def reset_context(self) -> None:
        for expert in self.experts:
            expert.reset()

    def _update_weights(self, rewards: Mapping[str, int]) -> dict[str, float]:
        pre = dict(self.weights)
        unnormalised = {
            name: pre[name] * exp(self.eta * rewards[name] / self.reward_bound)
            for name in EXPERT_NAMES
        }
        normaliser = sum(unnormalised.values())
        if not isfinite(normaliser) or normaliser <= 0:
            raise FloatingPointError("invalid Fixed-Share normaliser")
        posterior = {
            name: unnormalised[name] / normaliser for name in EXPERT_NAMES
        }
        shared = {
            name: (1.0 - self.share_rate) * posterior[name]
            + self.share_rate / len(EXPERT_NAMES)
            for name in EXPERT_NAMES
        }
        total = sum(shared.values())
        self.weights = {name: shared[name] / total for name in EXPERT_NAMES}
        if any(
            not isfinite(weight) or weight < 0 for weight in self.weights.values()
        ):
            raise FloatingPointError("invalid Fixed-Share weight")
        return pre

    def observe_outcome(
        self,
        *,
        movement_kind: str,
        price_change: int | None,
    ) -> OutcomeScore:
        """Score/update first, then append or reset public model context."""

        prior = self.last_proposal
        prior_master_position = 0 if prior is None else prior.master_position
        prior_positions = (
            {name: 0 for name in EXPERT_NAMES}
            if prior is None
            else dict(prior.expert_positions)
        )
        raw_master_reward = (
            0
            if price_change is None
            else prior_master_position * price_change
        )
        scoreable = (
            movement_kind == "genuine_nonzero"
            and price_change is not None
            and price_change != 0
            and prior is not None
        )
        expert_rewards = {
            name: prior_positions[name] * price_change if scoreable else 0
            for name in EXPERT_NAMES
        }
        master_reward = raw_master_reward if scoreable else 0
        weights_before = dict(self.weights)
        if scoreable:
            # This is the only learning update.  It occurs after the public
            # outcome is observable and before the next proposal is formed.
            weights_before = self._update_weights(expert_rewards)
            self.scoreable_count += 1
            self.master_shadow_pnl += master_reward
            for name, reward in expert_rewards.items():
                self.expert_cumulative_rewards[name] += reward
            self.reward_history.append(
                {
                    "master_reward": master_reward,
                    "expert_rewards": dict(expert_rewards),
                    "price_change": price_change,
                }
            )
        self.master_raw_shadow_pnl += raw_master_reward
        self.weight_history.append(dict(self.weights))

        if movement_kind == "genuine_nonzero" and price_change is not None:
            self._observe_bit(1 if price_change > 0 else 0)
        elif movement_kind == "reset":
            self.reset_context()
        elif movement_kind in {"unknown", "zero", "floor_clipped"}:
            for expert in self.experts:
                expert.observe(None)

        return OutcomeScore(
            scoreable=scoreable,
            movement_kind=movement_kind,
            price_change=price_change,
            prior_master_position=prior_master_position,
            prior_expert_positions=prior_positions,
            raw_master_reward=raw_master_reward,
            master_reward=master_reward,
            expert_rewards=expert_rewards,
            weights_before=weights_before,
            weights_after=dict(self.weights),
        )

    def _observe_bit(self, bit: int) -> None:
        _validate_bit(bit)
        for expert in self.experts:
            expert.observe(bit)

    def propose(self) -> MasterProposal:
        forecasts = {expert.name: expert.forecast() for expert in self.experts}
        weights = dict(self.weights)
        master_q = sum(weights[name] * forecasts[name].p_short for name in EXPERT_NAMES)
        if master_q > NEUTRAL_SHORT_PROBABILITY:
            master_position = 1
        elif master_q < NEUTRAL_SHORT_PROBABILITY:
            master_position = -1
        else:
            master_position = 0
        expected_long = master_q * SHORT_MOVE + (1.0 - master_q) * LONG_MOVE
        edge = abs(expected_long) if master_position else 0.0
        proposal = MasterProposal(
            weights=weights,
            forecasts=forecasts,
            expert_positions={
                name: forecasts[name].proposed_position for name in EXPERT_NAMES
            },
            master_short_probability=master_q,
            master_position=master_position,
            predicted_economic_edge=edge,
        )
        self.last_proposal = proposal
        return proposal

    def diagnostics(self) -> dict[str, object]:
        return {
            "expert_names": list(EXPERT_NAMES),
            "eta": self.eta,
            "share_rate": self.share_rate,
            "reward_bound": self.reward_bound,
            "neutral_short_probability": NEUTRAL_SHORT_PROBABILITY,
            "ctw_max_depth": CTW_MAX_DEPTH,
            "weights": dict(self.weights),
            "expert_cumulative_rewards": dict(self.expert_cumulative_rewards),
            "master_shadow_pnl": self.master_shadow_pnl,
            "master_raw_shadow_pnl": self.master_raw_shadow_pnl,
            "scoreable_count": self.scoreable_count,
            "reward_history": list(self.reward_history),
            "weight_history": list(self.weight_history),
            "last_proposal": (
                None if self.last_proposal is None else self.last_proposal.as_dict()
            ),
        }


__all__ = [
    "BinaryCTW",
    "CTW_MAX_DEPTH",
    "CausalExpert",
    "ContextTreeWeightingExpert",
    "EXPERT_NAMES",
    "ExpertForecast",
    "FIXED_SHARE_ETA",
    "FIXED_SHARE_RATE",
    "FlatExpert",
    "FixedShareMaster",
    "HistoryToken",
    "LONG_MOVE",
    "MarkovExpert",
    "MasterProposal",
    "NEUTRAL_SHORT_PROBABILITY",
    "OrderZeroFrequencyExpert",
    "OutcomeScore",
    "PersistenceExpert",
    "REWARD_BOUND",
    "ReversalExpert",
    "SHORT_MOVE",
    "make_experts",
]
