"""Pass 6A causal Fixed-Share strategies and statistically valid gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite, log, sqrt
from typing import Callable, TypeAlias

from .pass6_models import (
    EXPERT_NAMES,
    FIXED_SHARE_ETA,
    FIXED_SHARE_RATE,
    FixedShareMaster,
    MasterProposal,
    OutcomeScore,
    REWARD_BOUND,
)
from .simulator import AgentObservation


GROSS_PORTFOLIO_BUDGET = 600_000
PORTFOLIO_HEADROOM_RESERVE = 10_000
MAX_LIFERAFT_LOSS = 50_000
MAX_TRAILING_DRAWDOWN = 50_000
PRIMARY_PIVOTAL_PROBABILITY = 0.10
PIVOTAL_HAIRCUT = PRIMARY_PIVOTAL_PROBABILITY * 13_000
MIN_RESIDUAL_EDGE = 1_000.0
GLOBAL_ONE_SIDED_ALPHA = 0.05
PER_GATE_ALPHA = GLOBAL_ONE_SIDED_ALPHA / 2
CHECKPOINT_BLOCK_SIZE = 20
CHECKPOINT_COUNT = 365 // CHECKPOINT_BLOCK_SIZE
CHECKPOINT_ALPHA = PER_GATE_ALPHA / CHECKPOINT_COUNT
ANYTIME_BET_FRACTION = 0.5
ANYTIME_LOG_THRESHOLD = log(1 / PER_GATE_ALPHA)

ExposureSource: TypeAlias = float | int | Callable[[AgentObservation], float]


@dataclass(frozen=True)
class StopEvent:
    name: str
    day: int
    pnl_before: int
    pnl_after: int
    overshoot: int


class FixedCheckpointGate:
    """Bonferroni checkpoint gate using independent non-overlapping blocks.

    The reward samples are predictable bounded dollar rewards.  At each fixed
    block boundary, Hoeffding's one-sided lower confidence bound is compared
    with zero.  The result authorises only the following block; it cannot
    change the position that earned the qualifying reward.
    """

    name = "fixed_checkpoint"

    def __init__(
        self,
        *,
        block_size: int = CHECKPOINT_BLOCK_SIZE,
        max_checkpoints: int = CHECKPOINT_COUNT,
        alpha: float = PER_GATE_ALPHA,
        reward_bound: int = REWARD_BOUND,
    ) -> None:
        if block_size <= 0 or max_checkpoints <= 0:
            raise ValueError("checkpoint dimensions must be positive")
        if not 0 < alpha < 1:
            raise ValueError("checkpoint alpha must lie in (0, 1)")
        self.block_size = block_size
        self.max_checkpoints = max_checkpoints
        self.alpha = alpha
        self.checkpoint_alpha = alpha / max_checkpoints
        self.reward_bound = reward_bound
        self._block_rewards: list[int] = []
        self.scoreable_count = 0
        self.authorized = False
        self.ever_authorized = False
        self.first_authorization_day: int | None = None
        self.activation_days: list[int] = []
        self.reactivation_count = 0
        self.deactivation_days: list[int] = []
        self.evaluations: list[dict[str, object]] = []

    @property
    def active(self) -> bool:
        return self.authorized

    def observe(self, *, day: int, reward: int, scoreable: bool) -> None:
        if not scoreable:
            return
        if reward < -self.reward_bound or reward > self.reward_bound:
            raise ValueError("checkpoint reward is outside the frozen bound")
        self.scoreable_count += 1
        self._block_rewards.append(reward)
        if len(self._block_rewards) < self.block_size:
            return
        block_number = len(self.evaluations) + 1
        rewards = tuple(self._block_rewards)
        self._block_rewards.clear()
        if block_number > self.max_checkpoints:
            return
        block_mean = sum(rewards) / len(rewards)
        radius = self.reward_bound * sqrt(
            2.0 * log(1.0 / self.checkpoint_alpha) / len(rewards)
        )
        lower_bound = block_mean - radius
        passed = lower_bound > 0.0
        previous = self.authorized
        self.authorized = passed
        if passed and not previous:
            self.activation_days.append(day)
            if self.ever_authorized:
                self.reactivation_count += 1
            else:
                self.ever_authorized = True
                self.first_authorization_day = day
        elif previous and not passed:
            self.deactivation_days.append(day)
        self.evaluations.append(
            {
                "checkpoint": block_number,
                "day": day,
                "n": len(rewards),
                "block_sum": sum(rewards),
                "block_mean": block_mean,
                "hoeffding_radius": radius,
                "lower_bound": lower_bound,
                "passed": passed,
                "authorized_for_next_block": passed,
            }
        )

    def state(self) -> dict[str, object]:
        return {
            "name": self.name,
            "authorized": self.authorized,
            "scoreable_count": self.scoreable_count,
            "block_size": self.block_size,
            "max_checkpoints": self.max_checkpoints,
            "alpha": self.alpha,
            "checkpoint_alpha": self.checkpoint_alpha,
            "first_authorization_day": self.first_authorization_day,
            "activation_days": list(self.activation_days),
            "reactivation_count": self.reactivation_count,
            "deactivation_days": list(self.deactivation_days),
            "evaluations": list(self.evaluations),
        }


class AnytimeValidGate:
    """One-sided fixed-stake e-process with Ville-valid daily inspection.

    For normalized reward X in [-1, 1], the factor
    ``1 + ANYTIME_BET_FRACTION * X`` is nonnegative and has conditional mean
    at most one under the economic-null hypothesis E[X | history] <= 0.  The
    product is therefore a nonnegative supermartingale.  Crossing
    ``1 / PER_GATE_ALPHA`` has type-I probability at most PER_GATE_ALPHA by
    Ville's inequality, with no outcome-dependent reset.
    """

    name = "anytime_valid"

    def __init__(
        self,
        *,
        alpha: float = PER_GATE_ALPHA,
        reward_bound: int = REWARD_BOUND,
        bet_fraction: float = ANYTIME_BET_FRACTION,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("anytime alpha must lie in (0, 1)")
        if not 0 < bet_fraction <= 1:
            raise ValueError("anytime bet fraction must lie in (0, 1]")
        self.alpha = alpha
        self.reward_bound = reward_bound
        self.bet_fraction = bet_fraction
        self.log_e_value = 0.0
        self.scoreable_count = 0
        self.authorized = False
        self.ever_authorized = False
        self.first_authorization_day: int | None = None
        self.activation_days: list[int] = []

    @property
    def active(self) -> bool:
        return self.authorized

    @property
    def e_value(self) -> float:
        return exp(self.log_e_value)

    def observe(self, *, day: int, reward: int, scoreable: bool) -> None:
        if not scoreable:
            return
        if reward < -self.reward_bound or reward > self.reward_bound:
            raise ValueError("anytime reward is outside the frozen bound")
        self.scoreable_count += 1
        normalized = reward / self.reward_bound
        factor = 1.0 + self.bet_fraction * normalized
        if factor <= 0 or not isfinite(factor):
            raise ValueError("anytime e-process factor is not positive")
        self.log_e_value += log(factor)
        if not self.authorized and self.log_e_value >= ANYTIME_LOG_THRESHOLD:
            self.authorized = True
            self.ever_authorized = True
            self.first_authorization_day = day
            self.activation_days.append(day)

    def state(self) -> dict[str, object]:
        return {
            "name": self.name,
            "authorized": self.authorized,
            "scoreable_count": self.scoreable_count,
            "alpha": self.alpha,
            "bet_fraction": self.bet_fraction,
            "e_value": self.e_value,
            "log_e_value": self.log_e_value,
            "threshold": 1 / self.alpha,
            "first_authorization_day": self.first_authorization_day,
            "activation_days": list(self.activation_days),
            "reactivation_count": 0,
            "deactivation_days": [],
        }


def _voting_start_day(observation: AgentObservation) -> int:
    return (
        observation.voting_start_day
        if observation.voting_start_day is not None
        else observation.marked_boundary_day
    )


def _was_floor_clipped(observation: AgentObservation) -> bool:
    """Infer clipping from public prices and known canonical movement size."""

    if (
        observation.previous_price is None
        or observation.previous_price_change is None
        or observation.previous_price_change >= 0
        or observation.price != observation.price_floor
        or observation.previous_move_is_reset
    ):
        return False
    unclipped = observation.previous_price + observation.long_majority_move
    return unclipped < observation.price_floor


def movement_kind(observation: AgentObservation) -> str:
    """Classify the newly visible interval without hidden majority data."""

    start = _voting_start_day(observation)
    if observation.day <= start:
        return "inactive_or_startup"
    if observation.previous_move_is_reset or observation.is_reset_day:
        return "reset"
    change = observation.previous_price_change
    if change is None:
        return "unknown"
    if change == 0:
        return "zero"
    if _was_floor_clipped(observation):
        return "floor_clipped"
    return "genuine_nonzero"


def _is_live_interval(observation: AgentObservation, kind: str) -> bool:
    return (
        observation.day > _voting_start_day(observation)
        and kind in {"genuine_nonzero", "zero", "floor_clipped", "unknown"}
        and observation.previous_price_change is not None
    )


class Pass6FixedShareStrategy:
    """Shared implementation for the two gated candidates and ungated audit."""

    def __init__(
        self,
        *,
        candidate_name: str,
        gate: FixedCheckpointGate | AnytimeValidGate | None,
        other_portfolio_exposure: ExposureSource = 0.0,
        pivotal_probability: float = PRIMARY_PIVOTAL_PROBABILITY,
    ) -> None:
        if candidate_name not in {
            "fixed_checkpoint_fixed_share",
            "anytime_valid_fixed_share",
            "ungated_fixed_share",
        }:
            raise ValueError(f"unknown Pass 6 candidate {candidate_name!r}")
        if not 0 <= pivotal_probability <= 1:
            raise ValueError("pivotal probability must lie in [0, 1]")
        self.name = candidate_name
        self.gate = gate
        self.other_portfolio_exposure = other_portfolio_exposure
        self.pivotal_probability = pivotal_probability
        self.pivotal_haircut = pivotal_probability * 13_000
        self.master = FixedShareMaster()

        self._last_decision_day: int | None = None
        self._last_action = 0
        self._actual_cumulative_pnl = 0
        self._actual_high_water = 0
        self._loss_stop_active = False
        self._drawdown_stop_active = False
        self._stop_events: list[StopEvent] = []
        self._first_stop_trigger: str | None = None
        self._exposure_cache_day: int | None = None
        self._exposure_cache_value: float | None = None
        self._exposure_evaluation_count = 0
        self._headroom_gate_count = 0
        self._edge_gate_count = 0
        self._unknown_gate_count = 0
        self._floor_gate_count = 0
        self._real_trade_days: list[int] = []
        self._activation_days: list[int] = []
        self._timeline: list[dict[str, object]] = []
        self._shadow_events: list[dict[str, object]] = []

        if not callable(other_portfolio_exposure):
            self._validate_exposure(other_portfolio_exposure)

    @property
    def last_action(self) -> int:
        return self._last_action

    @property
    def cumulative_marked_pnl(self) -> int:
        return self._actual_cumulative_pnl

    @property
    def exposure_evaluation_count(self) -> int:
        return self._exposure_evaluation_count

    @property
    def loss_stop_active(self) -> bool:
        return self._loss_stop_active

    @property
    def drawdown_stop_active(self) -> bool:
        return self._drawdown_stop_active

    def _validate_exposure(self, value: object) -> float:
        if isinstance(value, bool):
            raise ValueError("other portfolio exposure must be numeric")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid other portfolio exposure {value!r}") from exc
        if not isfinite(numeric) or numeric < 0:
            raise ValueError("other portfolio exposure must be finite and non-negative")
        return numeric

    def _resolve_exposure(self, observation: AgentObservation) -> float:
        if self._exposure_cache_day == observation.day:
            assert self._exposure_cache_value is not None
            return self._exposure_cache_value
        source = self.other_portfolio_exposure
        value = source(observation) if callable(source) else source
        numeric = self._validate_exposure(value)
        self._exposure_cache_day = observation.day
        self._exposure_cache_value = numeric
        self._exposure_evaluation_count += 1
        return numeric

    def _record_stop(
        self,
        name: str,
        observation: AgentObservation,
        before: int,
        after: int,
        overshoot: int,
    ) -> None:
        event = StopEvent(name, observation.day, before, after, max(0, overshoot))
        self._stop_events.append(event)
        if name == "loss_stop":
            self._loss_stop_active = True
        elif name == "drawdown_stop":
            self._drawdown_stop_active = True
        if self._first_stop_trigger is None:
            self._first_stop_trigger = name

    def _account_actual_pnl(
        self,
        observation: AgentObservation,
        *,
        kind: str,
    ) -> int:
        if not _is_live_interval(observation, kind):
            return 0
        assert observation.previous_price_change is not None
        increment = observation.own_position * observation.previous_price_change
        before = self._actual_cumulative_pnl
        after = before + increment
        self._actual_cumulative_pnl = after
        self._actual_high_water = max(self._actual_high_water, after)
        if after <= -MAX_LIFERAFT_LOSS and not self._loss_stop_active:
            self._record_stop(
                "loss_stop",
                observation,
                before,
                after,
                -after - MAX_LIFERAFT_LOSS,
            )
        drawdown = self._actual_high_water - after
        if drawdown >= MAX_TRAILING_DRAWDOWN and not self._drawdown_stop_active:
            self._record_stop(
                "drawdown_stop",
                observation,
                before,
                after,
                drawdown - MAX_TRAILING_DRAWDOWN,
            )
        return increment

    def _gate_state(self) -> dict[str, object]:
        if self.gate is None:
            return {
                "name": "ungated",
                "authorized": True,
                "scoreable_count": self.master.scoreable_count,
                "activation_days": [],
                "reactivation_count": 0,
            }
        return self.gate.state()

    def _statistically_authorized(self) -> bool:
        return self.gate is None or self.gate.active

    def _select_action(
        self,
        observation: AgentObservation,
        *,
        kind: str,
        proposal: MasterProposal,
    ) -> tuple[int, dict[str, object]]:
        edge_after_haircut = proposal.predicted_economic_edge - self.pivotal_haircut
        edge_pass = edge_after_haircut >= MIN_RESIDUAL_EDGE
        statistical_pass = self._statistically_authorized()
        blocked_reason: str | None = None
        action = 0

        if not statistical_pass:
            blocked_reason = "statistical_gate"
        elif not edge_pass:
            self._edge_gate_count += 1
            blocked_reason = "economic_edge"
        elif kind in {"inactive_or_startup", "reset"}:
            blocked_reason = kind
        elif kind in {"unknown", "zero", "floor_clipped"}:
            self._unknown_gate_count += 1
            if kind == "floor_clipped":
                self._floor_gate_count += 1
            blocked_reason = kind
        elif observation.price == observation.price_floor:
            self._floor_gate_count += 1
            blocked_reason = "exact_price_floor"
        elif self._loss_stop_active:
            blocked_reason = "sticky_loss_stop"
        elif self._drawdown_stop_active:
            blocked_reason = "sticky_drawdown_stop"
        elif proposal.master_position == 0:
            blocked_reason = "flat_master"
        else:
            exposure = self._resolve_exposure(observation)
            permitted = (
                exposure
                + abs(proposal.master_position) * observation.price
                + PORTFOLIO_HEADROOM_RESERVE
                <= GROSS_PORTFOLIO_BUDGET
            )
            if not permitted:
                self._headroom_gate_count += 1
                blocked_reason = "portfolio_headroom"
            else:
                action = proposal.master_position

        return action, {
            "statistical_gate": statistical_pass,
            "edge_pass": edge_pass,
            "predicted_economic_edge": proposal.predicted_economic_edge,
            "pivotal_haircut": self.pivotal_haircut,
            "edge_after_haircut": edge_after_haircut,
            "minimum_residual_edge": MIN_RESIDUAL_EDGE,
            "blocked_reason": blocked_reason,
        }

    def decide(self, observation: AgentObservation) -> int:
        if self._last_decision_day is not None:
            if observation.day == self._last_decision_day:
                return self._last_action
            if observation.day < self._last_decision_day:
                raise ValueError("Pass 6 strategy observations must be chronological")

        kind = movement_kind(observation)
        actual_increment = 0
        outcome_score: OutcomeScore | None = None
        if observation.day >= _voting_start_day(observation):
            actual_increment = self._account_actual_pnl(observation, kind=kind)
            if observation.day > _voting_start_day(observation):
                outcome_score = self.master.observe_outcome(
                    movement_kind=kind,
                    price_change=observation.previous_price_change,
                )
                if self.gate is not None:
                    self.gate.observe(
                        day=observation.day,
                        reward=outcome_score.master_reward,
                        scoreable=outcome_score.scoreable,
                    )
                if outcome_score.scoreable:
                    self._shadow_events.append(
                        {
                            "day": observation.day,
                            "reward": outcome_score.master_reward,
                            "prior_master_position": outcome_score.prior_master_position,
                            "scoreable": True,
                        }
                    )
            proposal = self.master.propose()
            action, gate_details = self._select_action(
                observation,
                kind=kind,
                proposal=proposal,
            )
        else:
            # No model update or proposal is allowed before the live voting
            # start.  This covers both inactive lifecycle interpretations.
            proposal = None
            gate_details = {
                "statistical_gate": False,
                "edge_pass": False,
                "predicted_economic_edge": 0.0,
                "pivotal_haircut": self.pivotal_haircut,
                "edge_after_haircut": -self.pivotal_haircut,
                "minimum_residual_edge": MIN_RESIDUAL_EDGE,
                "blocked_reason": "inactive_or_startup",
            }
            action = 0

        if type(action) is not int or action not in (-1, 0, 1):
            raise AssertionError("Pass 6 produced a non-integral position")
        if action != 0:
            self._real_trade_days.append(observation.day)
        if (
            self.gate is not None
            and self.gate.first_authorization_day is not None
            and self.gate.first_authorization_day not in self._activation_days
        ):
            self._activation_days.append(self.gate.first_authorization_day)

        self._timeline.append(
            {
                "day": observation.day,
                "movement_kind": kind,
                "previous_price_change": observation.previous_price_change,
                "actual_increment": actual_increment,
                "actual_cumulative_pnl": self._actual_cumulative_pnl,
                "proposal": None if proposal is None else proposal.as_dict(),
                "outcome_score": (
                    None
                    if outcome_score is None
                    else {
                        "scoreable": outcome_score.scoreable,
                        "master_reward": outcome_score.master_reward,
                        "raw_master_reward": outcome_score.raw_master_reward,
                        "expert_rewards": dict(outcome_score.expert_rewards),
                        "weights_before": dict(outcome_score.weights_before),
                        "weights_after": dict(outcome_score.weights_after),
                    }
                ),
                "gate_state": self._gate_state(),
                "gate_details": gate_details,
                "action": action,
                "own_position": observation.own_position,
                "loss_stop_active": self._loss_stop_active,
                "drawdown_stop_active": self._drawdown_stop_active,
            }
        )
        self._last_action = action
        self._last_decision_day = observation.day
        return action

    def diagnostics(self) -> dict[str, object]:
        gate_state = self._gate_state()
        return {
            "candidate_name": self.name,
            "parameters": {
                "eta": FIXED_SHARE_ETA,
                "share_rate": FIXED_SHARE_RATE,
                "reward_bound": REWARD_BOUND,
                "block_size": CHECKPOINT_BLOCK_SIZE,
                "checkpoint_count": CHECKPOINT_COUNT,
                "global_one_sided_alpha": GLOBAL_ONE_SIDED_ALPHA,
                "per_gate_alpha": PER_GATE_ALPHA,
                "checkpoint_alpha": CHECKPOINT_ALPHA,
                "anytime_bet_fraction": ANYTIME_BET_FRACTION,
                "primary_pivotal_probability": self.pivotal_probability,
                "pivotal_haircut": self.pivotal_haircut,
                "minimum_residual_edge": MIN_RESIDUAL_EDGE,
                "loss_stop": MAX_LIFERAFT_LOSS,
                "trailing_drawdown_stop": MAX_TRAILING_DRAWDOWN,
                "portfolio_headroom_reserve": PORTFOLIO_HEADROOM_RESERVE,
            },
            "actual_realised_pnl": self._actual_cumulative_pnl,
            "actual_high_water": self._actual_high_water,
            "loss_stop_active": self._loss_stop_active,
            "drawdown_stop_active": self._drawdown_stop_active,
            "first_stop_trigger": self._first_stop_trigger,
            "stop_events": [asdict(event) for event in self._stop_events],
            "gate": gate_state,
            "activation_days": list(self._activation_days),
            "real_trade_days": list(self._real_trade_days),
            "exposure_evaluation_count": self._exposure_evaluation_count,
            "headroom_gate_count": self._headroom_gate_count,
            "edge_gate_count": self._edge_gate_count,
            "unknown_gate_count": self._unknown_gate_count,
            "floor_gate_count": self._floor_gate_count,
            "shadow_events": list(self._shadow_events),
            "master": self.master.diagnostics(),
            "timeline": list(self._timeline),
        }


def make_pass6_strategy(
    name: str,
    *,
    other_portfolio_exposure: ExposureSource = 0.0,
    pivotal_probability: float = PRIMARY_PIVOTAL_PROBABILITY,
) -> Pass6FixedShareStrategy:
    if name == "fixed_checkpoint_fixed_share":
        gate: FixedCheckpointGate | AnytimeValidGate | None = FixedCheckpointGate()
    elif name == "anytime_valid_fixed_share":
        gate = AnytimeValidGate()
    elif name == "ungated_fixed_share":
        gate = None
    else:
        raise KeyError(f"unknown Pass 6 strategy {name!r}")
    return Pass6FixedShareStrategy(
        candidate_name=name,
        gate=gate,
        other_portfolio_exposure=other_portfolio_exposure,
        pivotal_probability=pivotal_probability,
    )


PASS6_STRATEGY_NAMES: tuple[str, ...] = (
    "fixed_checkpoint_fixed_share",
    "anytime_valid_fixed_share",
    "ungated_fixed_share",
)


__all__ = [
    "ANYTIME_BET_FRACTION",
    "ANYTIME_LOG_THRESHOLD",
    "AnytimeValidGate",
    "CHECKPOINT_ALPHA",
    "CHECKPOINT_BLOCK_SIZE",
    "CHECKPOINT_COUNT",
    "FixedCheckpointGate",
    "GLOBAL_ONE_SIDED_ALPHA",
    "GROSS_PORTFOLIO_BUDGET",
    "MAX_LIFERAFT_LOSS",
    "MAX_TRAILING_DRAWDOWN",
    "MIN_RESIDUAL_EDGE",
    "PASS6_STRATEGY_NAMES",
    "PIVOTAL_HAIRCUT",
    "PORTFOLIO_HEADROOM_RESERVE",
    "PRIMARY_PIVOTAL_PROBABILITY",
    "Pass6FixedShareStrategy",
    "StopEvent",
    "make_pass6_strategy",
    "movement_kind",
]
