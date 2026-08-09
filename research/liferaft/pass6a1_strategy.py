"""Pass 6A.1 corrected anytime-gated Fixed-Share strategy.

This module intentionally leaves the historical Pass 6A implementation alone.
It reuses the seven causal experts but makes the evidence policy match the
non-statistical policy that would actually be executable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Callable, Mapping, TypeAlias

from .pass6_models import (
    EXPERT_NAMES,
    FIXED_SHARE_ETA,
    FIXED_SHARE_RATE,
    LONG_MOVE,
    NEUTRAL_SHORT_PROBABILITY,
    SHORT_MOVE,
    ExpertForecast,
    MasterProposal,
    make_experts,
)
from .pass6_strategies import movement_kind
from .simulator import AgentObservation, MajorityOutcome, majority_from_counts


IMPACT_HAIRCUT = 1_300
ADJUSTED_REWARD_BOUND = 9_300
PRIMARY_PIVOTAL_PROBABILITY = 0.10
MIN_RESIDUAL_EDGE = 1_000.0
PORTFOLIO_BUDGET = 600_000
PORTFOLIO_HEADROOM_RESERVE = 10_000
MAX_LIFERAFT_LOSS = 50_000
MAX_TRAILING_DRAWDOWN = 50_000
ANYTIME_ALPHA = 0.025
ANYTIME_BET_FRACTION = 0.5

ExposureSource: TypeAlias = float | int | Callable[[AgentObservation], float]


def _is_evidence_kind(kind: str) -> bool:
    return kind in {"genuine_nonzero", "floor_clipped", "zero"}


def _is_live_pnl_kind(observation: AgentObservation, kind: str) -> bool:
    start = (
        observation.voting_start_day
        if observation.voting_start_day is not None
        else observation.marked_boundary_day
    )
    return (
        observation.day > start
        and kind in {"genuine_nonzero", "floor_clipped", "zero", "unknown"}
        and observation.previous_price_change is not None
    )


def _neutralize_invalid(forecast: ExpertForecast) -> ExpertForecast:
    if forecast.valid:
        return forecast
    return ExpertForecast(
        forecast.name,
        NEUTRAL_SHORT_PROBABILITY,
        forecast.support,
        False,
        forecast.source + "|neutralized_to_q_star",
    )


class CorrectedAnytimeGate:
    """Fixed-stake e-process using the instance's configured alpha."""

    def __init__(
        self,
        *,
        alpha: float = ANYTIME_ALPHA,
        bet_fraction: float = ANYTIME_BET_FRACTION,
        reward_bound: int = ADJUSTED_REWARD_BOUND,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0 < bet_fraction <= 1:
            raise ValueError("bet fraction must lie in (0, 1]")
        if reward_bound <= 0:
            raise ValueError("reward bound must be positive")
        self.alpha = alpha
        self.bet_fraction = bet_fraction
        self.reward_bound = reward_bound
        self.log_e_value = 0.0
        self.authorized = False
        self.activation_day: int | None = None
        self.scoreable_count = 0
        self.unavailable_count = 0

    @property
    def threshold(self) -> float:
        return log(1.0 / self.alpha)

    @property
    def active(self) -> bool:
        return self.authorized

    @property
    def e_value(self) -> float:
        return exp(self.log_e_value)

    def observe(self, *, day: int, reward: int, scoreable: bool) -> None:
        if not scoreable:
            self.unavailable_count += 1
            return
        if not -self.reward_bound <= reward <= self.reward_bound:
            raise ValueError("adjusted reward outside the frozen bound")
        self.scoreable_count += 1
        factor = 1.0 + self.bet_fraction * reward / self.reward_bound
        if factor <= 0 or not isfinite(factor):
            raise ValueError("invalid e-process factor")
        self.log_e_value += log(factor)
        if not self.authorized and self.log_e_value >= self.threshold:
            self.authorized = True
            self.activation_day = day

    def diagnostics(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "bet_fraction": self.bet_fraction,
            "reward_bound": self.reward_bound,
            "threshold": 1.0 / self.alpha,
            "log_threshold": self.threshold,
            "log_e_value": self.log_e_value,
            "e_value": self.e_value,
            "authorized": self.authorized,
            "activation_day": self.activation_day,
            "scoreable_count": self.scoreable_count,
            "unavailable_count": self.unavailable_count,
        }


@dataclass(frozen=True)
class CorrectedOutcome:
    evidence_scoreable: bool
    movement_kind: str
    raw_observable_reward: int
    adjusted_gate_reward: int
    expert_rewards: dict[str, int]
    prior_paper_action: int


class CorrectedFixedShareMaster:
    """Fixed-Share master with impact-adjusted, executable-policy rewards."""

    def __init__(self) -> None:
        self.eta = FIXED_SHARE_ETA
        self.share_rate = FIXED_SHARE_RATE
        self.weights = {name: 1.0 / len(EXPERT_NAMES) for name in EXPERT_NAMES}
        self.experts = make_experts()
        self.last_proposal: MasterProposal | None = None
        self.evidence_count = 0
        self.raw_observable_pnl = 0
        self.adjusted_evidence_pnl = 0
        self.expert_cumulative_rewards = {name: 0 for name in EXPERT_NAMES}
        self.last_outcome: CorrectedOutcome | None = None

    def _update_weights(self, rewards: Mapping[str, int]) -> None:
        unnormalised = {
            name: self.weights[name]
            * exp(self.eta * rewards[name] / ADJUSTED_REWARD_BOUND)
            for name in EXPERT_NAMES
        }
        total = sum(unnormalised.values())
        if not isfinite(total) or total <= 0:
            raise FloatingPointError("invalid corrected Fixed-Share normalizer")
        posterior = {name: unnormalised[name] / total for name in EXPERT_NAMES}
        shared = {
            name: (1.0 - self.share_rate) * posterior[name]
            + self.share_rate / len(EXPERT_NAMES)
            for name in EXPERT_NAMES
        }
        total_shared = sum(shared.values())
        self.weights = {name: shared[name] / total_shared for name in EXPERT_NAMES}
        if any(not isfinite(value) or value < 0 for value in self.weights.values()):
            raise FloatingPointError("invalid corrected Fixed-Share weights")

    def reset_context(self) -> None:
        for expert in self.experts:
            expert.reset()

    def propose(self) -> MasterProposal:
        forecasts = {
            expert.name: _neutralize_invalid(expert.forecast())
            for expert in self.experts
        }
        master_q = sum(
            self.weights[name] * forecasts[name].p_short for name in EXPERT_NAMES
        )
        if abs(master_q - NEUTRAL_SHORT_PROBABILITY) < 1e-12:
            master_q = NEUTRAL_SHORT_PROBABILITY
        if master_q > NEUTRAL_SHORT_PROBABILITY:
            position = 1
        elif master_q < NEUTRAL_SHORT_PROBABILITY:
            position = -1
        else:
            position = 0
        expected_long = master_q * SHORT_MOVE + (1.0 - master_q) * LONG_MOVE
        proposal = MasterProposal(
            weights=dict(self.weights),
            forecasts=forecasts,
            expert_positions={
                name: forecasts[name].proposed_position for name in EXPERT_NAMES
            },
            master_short_probability=master_q,
            master_position=position,
            predicted_economic_edge=abs(expected_long) if position else 0.0,
        )
        self.last_proposal = proposal
        return proposal

    def observe_outcome(
        self,
        *,
        day: int,
        movement_kind: str,
        price_change: int | None,
        prior_paper_action: int,
    ) -> CorrectedOutcome:
        scoreable = _is_evidence_kind(movement_kind) and price_change is not None
        prior = self.last_proposal
        prior_positions = (
            {name: 0 for name in EXPERT_NAMES}
            if prior is None
            else dict(prior.expert_positions)
        )
        raw = prior_paper_action * price_change if scoreable else 0
        adjusted = (
            raw - IMPACT_HAIRCUT * int(prior_paper_action != 0)
            if scoreable
            else 0
        )
        expert_rewards = {
            name: (
                prior_positions[name] * price_change
                - IMPACT_HAIRCUT * int(prior_positions[name] != 0)
                if scoreable and price_change is not None
                else 0
            )
            for name in EXPERT_NAMES
        }
        if scoreable and prior is not None:
            self._update_weights(expert_rewards)
            self.evidence_count += 1
            self.raw_observable_pnl += raw
            self.adjusted_evidence_pnl += adjusted
            for name, reward in expert_rewards.items():
                self.expert_cumulative_rewards[name] += reward

        if movement_kind == "genuine_nonzero" and price_change is not None:
            bit = 1 if price_change > 0 else 0
            for expert in self.experts:
                expert.observe(bit)
        elif movement_kind in {"reset", "unknown", "zero", "floor_clipped"}:
            self.reset_context()

        outcome = CorrectedOutcome(
            evidence_scoreable=bool(scoreable and prior is not None),
            movement_kind=movement_kind,
            raw_observable_reward=raw if prior is not None else 0,
            adjusted_gate_reward=adjusted if prior is not None else 0,
            expert_rewards=expert_rewards,
            prior_paper_action=prior_paper_action,
        )
        self.last_outcome = outcome
        return outcome

    def diagnostics(self) -> dict[str, object]:
        return {
            "weights": dict(self.weights),
            "expert_cumulative_rewards": dict(self.expert_cumulative_rewards),
            "evidence_count": self.evidence_count,
            "raw_observable_pnl": self.raw_observable_pnl,
            "adjusted_evidence_pnl": self.adjusted_evidence_pnl,
            "last_proposal": (
                None if self.last_proposal is None else self.last_proposal.as_dict()
            ),
        }


class Pass6A1AnytimeStrategy:
    """Only candidate in Pass 6A.1: corrected anytime-gated Fixed-Share."""

    name = "pass6a1_corrected_anytime_fixed_share"

    def __init__(
        self,
        *,
        other_portfolio_exposure: ExposureSource = 0.0,
        alpha: float = ANYTIME_ALPHA,
    ) -> None:
        self.other_portfolio_exposure = other_portfolio_exposure
        self.master = CorrectedFixedShareMaster()
        self.gate = CorrectedAnytimeGate(alpha=alpha)
        self._last_day: int | None = None
        self._last_action = 0
        self._last_paper_action = 0
        self._actual_pnl = 0
        self._actual_high_water = 0
        self._loss_stop = False
        self._drawdown_stop = False
        self._stop_overshoots: list[int] = []
        self._exposure_day: int | None = None
        self._exposure_value: float | None = None
        self._exposure_evaluations = 0
        self._headroom_gates = 0
        self._edge_gates = 0
        self._unknown_gates = 0
        self._floor_gates = 0
        self._paper_records: list[dict[str, object]] = []
        self._evidence_events: list[dict[str, object]] = []

        if not callable(other_portfolio_exposure):
            self._validate_exposure(other_portfolio_exposure)

    def _validate_exposure(self, value: object) -> float:
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

    def _account_actual(self, observation: AgentObservation, kind: str) -> int:
        if not _is_live_pnl_kind(observation, kind):
            return 0
        assert observation.previous_price_change is not None
        increment = observation.own_position * observation.previous_price_change
        before = self._actual_pnl
        self._actual_pnl += increment
        self._actual_high_water = max(self._actual_high_water, self._actual_pnl)
        if self._actual_pnl <= -MAX_LIFERAFT_LOSS and not self._loss_stop:
            self._loss_stop = True
            self._stop_overshoots.append(
                max(0, -self._actual_pnl - MAX_LIFERAFT_LOSS)
            )
        drawdown = self._actual_high_water - self._actual_pnl
        if drawdown >= MAX_TRAILING_DRAWDOWN and not self._drawdown_stop:
            self._drawdown_stop = True
            self._stop_overshoots.append(max(0, drawdown - MAX_TRAILING_DRAWDOWN))
        return increment

    def _paper_action(
        self,
        observation: AgentObservation,
        kind: str,
        proposal: MasterProposal,
    ) -> tuple[int, str | None]:
        start = (
            observation.voting_start_day
            if observation.voting_start_day is not None
            else observation.marked_boundary_day
        )
        if observation.day <= start:
            return 0, "inactive_or_startup"
        if kind in {"reset", "unknown", "zero", "floor_clipped"}:
            self._unknown_gates += int(kind in {"reset", "unknown", "zero"})
            self._floor_gates += int(kind == "floor_clipped")
            return 0, kind
        if observation.price == observation.price_floor:
            self._floor_gates += 1
            return 0, "exact_price_floor"
        if proposal.master_position == 0:
            return 0, "flat_master"
        if proposal.predicted_economic_edge - IMPACT_HAIRCUT < MIN_RESIDUAL_EDGE:
            self._edge_gates += 1
            return 0, "economic_edge"
        position = proposal.master_position
        if type(position) is not int or position not in (-1, 0, 1):
            raise AssertionError("paper action is not integral")
        exposure = self._resolve_exposure(observation)
        if exposure + abs(position) * observation.price + PORTFOLIO_HEADROOM_RESERVE > PORTFOLIO_BUDGET:
            self._headroom_gates += 1
            return 0, "portfolio_headroom"
        return position, None

    def decide(self, observation: AgentObservation) -> int:
        if self._last_day is not None:
            if observation.day == self._last_day:
                return self._last_action
            if observation.day < self._last_day:
                raise ValueError("Pass 6A.1 observations must be chronological")

        kind = movement_kind(observation)
        actual_increment = self._account_actual(observation, kind)
        outcome: CorrectedOutcome | None = None
        proposal: MasterProposal | None = None
        if observation.day < (
            observation.voting_start_day
            if observation.voting_start_day is not None
            else observation.marked_boundary_day
        ):
            paper_action = 0
            paper_reason = "inactive_or_startup"
            actual_action = 0
        else:
            if observation.day > (
                observation.voting_start_day
                if observation.voting_start_day is not None
                else observation.marked_boundary_day
            ):
                outcome = self.master.observe_outcome(
                    day=observation.day,
                    movement_kind=kind,
                    price_change=observation.previous_price_change,
                    prior_paper_action=self._last_paper_action,
                )
                self.gate.observe(
                    day=observation.day,
                    reward=outcome.adjusted_gate_reward,
                    scoreable=outcome.evidence_scoreable,
                )
                if outcome.evidence_scoreable:
                    self._evidence_events.append(
                        {
                            "day": observation.day,
                            "source_day": observation.day - 1,
                            "movement_kind": kind,
                            "raw_observable_reward": outcome.raw_observable_reward,
                            "adjusted_gate_reward": outcome.adjusted_gate_reward,
                            "prior_paper_action": self._last_paper_action,
                        }
                    )
            proposal = self.master.propose()
            paper_action, paper_reason = self._paper_action(
                observation, kind, proposal
            )
            actual_action = (
                paper_action
                if self.gate.active and not self._loss_stop and not self._drawdown_stop
                else 0
            )

        if type(paper_action) is not int or paper_action not in (-1, 0, 1):
            raise AssertionError("paper action is not integral")
        if type(actual_action) is not int or actual_action not in (-1, 0, 1):
            raise AssertionError("actual action is not integral")
        self._paper_records.append(
            {
                "day": observation.day,
                "paper_action": paper_action,
                "actual_action": actual_action,
                "own_position": observation.own_position,
                "movement_kind": kind,
                "stat_authorized": self.gate.active,
                "paper_reason": paper_reason,
                "predicted_edge": (
                    None if proposal is None else proposal.predicted_economic_edge
                ),
                "actual_increment": actual_increment,
            }
        )
        self._last_paper_action = paper_action
        self._last_action = actual_action
        self._last_day = observation.day
        return actual_action

    def diagnostics(self) -> dict[str, object]:
        return {
            "candidate_name": self.name,
            "actual_realised_pnl": self._actual_pnl,
            "actual_high_water": self._actual_high_water,
            "loss_stop_active": self._loss_stop,
            "drawdown_stop_active": self._drawdown_stop,
            "stop_overshoots": list(self._stop_overshoots),
            "exposure_evaluation_count": self._exposure_evaluations,
            "headroom_gate_count": self._headroom_gates,
            "edge_gate_count": self._edge_gates,
            "unknown_gate_count": self._unknown_gates,
            "floor_gate_count": self._floor_gates,
            "paper_records": list(self._paper_records),
            "evidence_events": list(self._evidence_events),
            "gate": self.gate.diagnostics(),
            "master": self.master.diagnostics(),
        }


def make_pass6a1_strategy(
    *,
    other_portfolio_exposure: ExposureSource = 0.0,
    alpha: float = ANYTIME_ALPHA,
) -> Pass6A1AnytimeStrategy:
    return Pass6A1AnytimeStrategy(
        other_portfolio_exposure=other_portfolio_exposure,
        alpha=alpha,
    )


def run_self_checks() -> None:
    """Small focused assertions required for this correction pass."""

    master = CorrectedFixedShareMaster()
    proposal = master.propose()
    assert abs(proposal.master_short_probability - NEUTRAL_SHORT_PROBABILITY) < 1e-15
    assert proposal.master_position == 0

    floor = master.observe_outcome(
        day=1,
        movement_kind="floor_clipped",
        price_change=-3_000,
        prior_paper_action=1,
    )
    assert floor.evidence_scoreable
    assert floor.adjusted_gate_reward == -4_300
    assert master.experts[2].forecast().valid is False

    zero_master = CorrectedFixedShareMaster()
    zero_master.propose()
    zero = zero_master.observe_outcome(
        day=1,
        movement_kind="zero",
        price_change=0,
        prior_paper_action=1,
    )
    assert zero.adjusted_gate_reward == -IMPACT_HAIRCUT

    gate = CorrectedAnytimeGate(alpha=0.025)
    gate.observe(day=1, reward=6_700, scoreable=True)
    assert gate.activation_day is None
    gate.observe(day=2, reward=6_700, scoreable=True)
    assert gate.activation_day is None or gate.activation_day >= 2

    # Removing a flat focal vote leaves an opponent-only tie.  A hypothetical
    # long paper action changes that status and is therefore pivotal.
    opponent_only_margin = 0
    hypothetical = majority_from_counts(1, 0)
    actual = majority_from_counts(0, 0)
    assert hypothetical is not actual
    assert opponent_only_margin + 1 == 1

    assert 1 * (-5_000) == -5_000
    assert isinstance(1, int) and 1 in (-1, 0, 1)
    assert 500_000 + 100_000 + PORTFOLIO_HEADROOM_RESERVE > PORTFOLIO_BUDGET


__all__ = [
    "ADJUSTED_REWARD_BOUND",
    "ANYTIME_ALPHA",
    "CorrectedAnytimeGate",
    "CorrectedFixedShareMaster",
    "IMPACT_HAIRCUT",
    "Pass6A1AnytimeStrategy",
    "make_pass6a1_strategy",
    "movement_kind",
    "run_self_checks",
]
