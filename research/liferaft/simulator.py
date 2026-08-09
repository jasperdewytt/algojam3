"""A small, auditable simulator for the AlgoJam 3 Liferaft Ticket.

The simulator deliberately separates the public observation supplied to an
agent from the hidden vote and budget diagnostics recorded by the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, TypeAlias


class MajorityOutcome(str, Enum):
    """The side selected by the non-flat votes on a day."""

    LONG = "long"
    SHORT = "short"
    TIE = "tie"


class SideStatus(str, Enum):
    """An agent's relation to the day's vote outcome."""

    MAJORITY = "majority"
    MINORITY = "minority"
    FLAT = "flat"
    TIED = "tied"


Action: TypeAlias = int
ExposureCallback: TypeAlias = Callable[[str, "AgentObservation"], float]
ExposureSpec: TypeAlias = float | int | Mapping[str, float | int] | ExposureCallback
MarketMode: TypeAlias = Literal["continuous_reset", "inactive_until_marked"]
PreVotingExecution: TypeAlias = Literal[
    "observe_and_ignore_actions", "fully_inactive"
]


@dataclass(frozen=True)
class LiferaftConfig:
    """Competition mechanics and the configurable marked-period boundary.

    ``marked_boundary_day`` is an index, not a count.  With the competition
    default of 365, day 365 is the first day whose price is reset to
    ``reset_price``.  The action chosen on day 365 is held into the first
    scored movement, from day 365 to day 366.

    ``market_mode="continuous_reset"`` retains that historical indexed-reset
    model.  ``market_mode="inactive_until_marked"`` instead holds
    ``pre_voting_price`` through the voting-start observation; the day at
    ``voting_start_day`` chooses the first live movement into the next day.
    """

    total_days: int = 730
    marked_boundary_day: int = 365
    initial_price: int = 100_000
    reset_price: int = 100_000
    price_floor: int = 20_000
    long_majority_move: int = -5_000
    short_majority_move: int = 8_000
    position_limit: int = 1
    gross_portfolio_budget: int = 600_000
    # ``continuous_reset`` preserves the historical supplied-code model.
    # ``inactive_until_marked`` models the organiser's clarified timeline.
    market_mode: MarketMode = "continuous_reset"
    pre_voting_execution: PreVotingExecution = "observe_and_ignore_actions"
    pre_voting_price: int = 100_000
    voting_start_day: int | None = None

    def __post_init__(self) -> None:
        if self.total_days <= 0:
            raise ValueError("total_days must be positive")
        if not 0 <= self.marked_boundary_day < self.total_days:
            raise ValueError(
                "marked_boundary_day must be an index in [0, total_days)"
            )
        if self.price_floor <= 0:
            raise ValueError("price_floor must be positive")
        if self.initial_price < self.price_floor:
            raise ValueError("initial_price cannot be below price_floor")
        if self.reset_price < self.price_floor:
            raise ValueError("reset_price cannot be below price_floor")
        if self.long_majority_move >= 0:
            raise ValueError("long_majority_move must be downward")
        if self.short_majority_move <= 0:
            raise ValueError("short_majority_move must be upward")
        if self.position_limit <= 0:
            raise ValueError("position_limit must be positive")
        if self.gross_portfolio_budget < 0:
            raise ValueError("gross_portfolio_budget cannot be negative")
        if self.market_mode not in ("continuous_reset", "inactive_until_marked"):
            raise ValueError(
                "market_mode must be 'continuous_reset' or "
                "'inactive_until_marked'"
            )
        if self.pre_voting_execution not in (
            "observe_and_ignore_actions",
            "fully_inactive",
        ):
            raise ValueError(
                "pre_voting_execution must be 'observe_and_ignore_actions' "
                "or 'fully_inactive'"
            )
        if self.pre_voting_price < self.price_floor:
            raise ValueError("pre_voting_price cannot be below price_floor")
        if self.voting_start_day is None:
            object.__setattr__(
                self, "voting_start_day", self.marked_boundary_day
            )
        if not 0 <= self.voting_start_day < self.total_days:
            raise ValueError("voting_start_day must be an index in [0, total_days)")


@dataclass(frozen=True)
class AgentObservation:
    """Information available to one competitor at one simultaneous decision.

    The history includes the current price and never includes a future price.
    Vote counts, the true majority, other agents' actions, and budget
    diagnostics are intentionally absent.
    """

    day: int
    price: int
    price_history: tuple[int, ...]
    previous_price: int | None
    previous_price_change: int | None
    previous_move_is_reset: bool
    is_reset_day: bool
    marked_boundary_day: int
    price_floor: int
    long_majority_move: int
    short_majority_move: int
    position_limit: int
    gross_portfolio_budget: int
    own_position: int
    # These fields are defaults so direct construction by historical callers
    # remains source-compatible.  The simulator always supplies exact values.
    voting_active: bool = True
    market_mode: MarketMode = "continuous_reset"
    voting_start_day: int | None = None

    @property
    def previous_inferred_majority(self) -> MajorityOutcome | None:
        """Infer the last genuine majority from the public price move.

        Reset moves are explicitly excluded here so callers cannot
        accidentally treat a reset that happens to equal a canonical move as
        a vote outcome. Any genuine negative move is long-majority evidence,
        including a floor-clipped move; any genuine positive move is
        short-majority evidence. A genuine zero move remains ambiguous.
        """

        return infer_majority_from_price_change(
            self.previous_price_change,
            long_majority_move=self.long_majority_move,
            short_majority_move=self.short_majority_move,
            previous_move_is_reset=self.previous_move_is_reset,
        )


class Agent(Protocol):
    """Minimal Pass 2 interface for a candidate Liferaft strategy."""

    name: str

    def decide(self, observation: AgentObservation) -> object:
        """Return the desired integer position for the next price interval."""


def majority_from_counts(long_count: int, short_count: int) -> MajorityOutcome:
    """Return the majority outcome; flat votes are already excluded."""

    if long_count > short_count:
        return MajorityOutcome.LONG
    if short_count > long_count:
        return MajorityOutcome.SHORT
    return MajorityOutcome.TIE


def infer_majority_from_price_change(
    price_change: int | None,
    *,
    long_majority_move: int = -5_000,
    short_majority_move: int = 8_000,
    previous_move_is_reset: bool = False,
) -> MajorityOutcome | None:
    """Infer a majority from a genuine public price move.

    The canonical move parameters remain accepted for source compatibility,
    but inference intentionally uses the sign of a genuine move. A negative
    move implies a long majority even when the floor clipped its magnitude; a
    positive move implies a short majority. Genuine zero moves and every reset
    move are ambiguous and return ``None``. This is a public inference helper,
    not hidden engine information.
    """

    # Keep the named parameters in the public signature so custom scenario
    # callers remain source-compatible even though sign inference is now the
    # correct rule for clipped and custom-sized genuine moves.
    del long_majority_move, short_majority_move
    if previous_move_is_reset or price_change is None or price_change == 0:
        return None
    if price_change < 0:
        return MajorityOutcome.LONG
    if price_change > 0:
        return MajorityOutcome.SHORT
    return None


def _action_for_outcome(outcome: MajorityOutcome, *, opposite: bool) -> int:
    if outcome is MajorityOutcome.LONG:
        return -1 if opposite else 1
    if outcome is MajorityOutcome.SHORT:
        return 1 if opposite else -1
    return 0


@dataclass(frozen=True)
class RejectedAction:
    """A requested action that was flattened by validation or budget checks."""

    day: int
    agent_name: str
    requested_action: object
    effective_action: int
    current_price: int
    other_portfolio_exposure: float
    requested_gross_exposure: float
    reason: str


@dataclass(frozen=True)
class BudgetBreach:
    """A day on which a requested portfolio was over the gross budget."""

    day: int
    agent_name: str
    requested_action: object
    effective_action: int
    current_price: int
    other_portfolio_exposure: float
    requested_gross_exposure: float
    budget: int
    reason: str


@dataclass(frozen=True)
class AgentDayRecord:
    """Public action plus per-agent hidden diagnostics for one day.

    In inactive-until-marked mode, ``requested_action`` retains the raw agent
    request while ``action`` is the effective market action (zero). The
    ``market_action_ignored`` fields distinguish that diagnostic from a live
    rejected action or budget breach.
    """

    day: int
    agent_name: str
    price: int
    requested_action: object
    action: int
    action_rejected: bool
    rejection_reason: str | None
    other_portfolio_exposure: float
    requested_gross_exposure: float
    budget_usage: float
    budget_breach: bool
    daily_pnl: int
    cumulative_pnl: int
    calibration_cumulative_pnl: int
    marked_cumulative_pnl: int
    majority: MajorityOutcome
    status: SideStatus
    # Decision-time fields above describe the newly selected current-day vote.
    # The realised fields below describe the prior held position and movement.
    pnl_position: int | None = None
    pnl_source_day: int | None = None
    pnl_majority: MajorityOutcome | None = None
    pnl_status: SideStatus | None = None
    agent_called: bool = True
    # Inactive-mode requests are retained for audit but are not Liferaft
    # trades. These fields keep that distinction separate from live rejection
    # and budget diagnostics.
    market_action_ignored: bool = False
    market_action_ignore_reason: str | None = None

    @property
    def in_majority(self) -> bool:
        return self.status is SideStatus.MAJORITY

    @property
    def in_minority(self) -> bool:
        return self.status is SideStatus.MINORITY

    @property
    def flat(self) -> bool:
        return self.status is SideStatus.FLAT


@dataclass(frozen=True)
class DayRecord:
    """All engine diagnostics for one simulated day."""

    day: int
    price: int
    previous_price: int | None
    price_change: int | None
    pnl_price_change: int
    voting_active: bool
    reset_applied: bool
    reset_jump: int | None
    requested_actions: dict[str, object]
    actions: dict[str, int]
    long_count: int
    short_count: int
    flat_count: int
    net_vote_margin: int
    net_margin_before_focal: int | None
    net_margin_after_focal: int | None
    majority: MajorityOutcome
    focal_agent_name: str | None
    focal_pivotal: bool
    focal_converted_one_vote_majority_to_tie: bool
    unclipped_next_price: int | None
    next_price: int | None
    floor_clipped: bool
    agent_records: dict[str, AgentDayRecord]


@dataclass(frozen=True)
class SimulationResult:
    """Immutable-ish result object containing the full research audit trail."""

    config: LiferaftConfig
    days: tuple[DayRecord, ...]
    price_path: tuple[int, ...]
    reset_day: int
    reset_jump: int | None
    calibration_pnl: dict[str, int]
    marked_pnl: dict[str, int]
    cumulative_pnl: dict[str, int]
    rejected_actions: tuple[RejectedAction, ...]
    budget_breaches: tuple[BudgetBreach, ...]
    random_seeds: dict[str, int | None]
    scenario_name: str | None
    scenario_configuration: dict[str, Any]
    focal_agent_name: str | None

    @property
    def final_marked_pnl(self) -> int:
        """Marked-period score when the result contains one focal agent."""

        if self.focal_agent_name is not None:
            return self.marked_pnl[self.focal_agent_name]
        return sum(self.marked_pnl.values())

    def agent_history(self, agent_name: str) -> tuple[AgentDayRecord, ...]:
        """Return one agent's records in day order."""

        return tuple(day.agent_records[agent_name] for day in self.days)


class LiferaftSimulator:
    """Run simultaneous Liferaft decisions under one explicit market mode."""

    def __init__(
        self,
        agents: Sequence[Agent],
        config: LiferaftConfig | None = None,
        *,
        other_portfolio_exposure: ExposureSpec = 0.0,
        focal_agent_name: str | None = None,
        scenario_name: str | None = None,
        scenario_configuration: Mapping[str, Any] | None = None,
        random_seeds: Mapping[str, int | None] | None = None,
    ) -> None:
        self.config = config or LiferaftConfig()
        self.agents = tuple(agents)
        if not self.agents:
            raise ValueError("at least one agent is required")

        self.agent_names = tuple(agent.name for agent in self.agents)
        if any(not isinstance(name, str) or not name for name in self.agent_names):
            raise ValueError("every agent must have a non-empty string name")
        if len(set(self.agent_names)) != len(self.agent_names):
            raise ValueError("agent names must be unique")
        if focal_agent_name is not None and focal_agent_name not in self.agent_names:
            raise ValueError("focal_agent_name must identify one supplied agent")

        self.other_portfolio_exposure = other_portfolio_exposure
        self.focal_agent_name = focal_agent_name
        self.scenario_name = scenario_name
        self.scenario_configuration = dict(scenario_configuration or {})
        self.random_seeds = dict(random_seeds or {})
        self._has_run = False

    def run(self) -> SimulationResult:
        """Simulate all days and return the complete diagnostic record."""

        if self._has_run:
            raise RuntimeError(
                "LiferaftSimulator.run() may only be called once; construct a "
                "new simulator with fresh agent instances to rerun it"
            )
        self._has_run = True

        config = self.config
        inactive_mode = config.market_mode == "inactive_until_marked"
        voting_start_day = config.voting_start_day
        # LiferaftConfig normalises a missing voting_start_day in __post_init__.
        assert voting_start_day is not None
        held_positions = {name: 0 for name in self.agent_names}
        calibration_pnl = {name: 0 for name in self.agent_names}
        marked_pnl = {name: 0 for name in self.agent_names}
        scoring_cumulative_pnl = {name: 0 for name in self.agent_names}

        price_path: list[int] = []
        day_records: list[DayRecord] = []
        rejected_actions: list[RejectedAction] = []
        budget_breaches: list[BudgetBreach] = []
        pending_next_price: int | None = None
        reset_jump: int | None = None
        previous_vote_majority: MajorityOutcome | None = None

        for day in range(config.total_days):
            previous_price = price_path[-1] if price_path else None
            reset_applied = (
                not inactive_mode and day == config.marked_boundary_day
            )
            voting_active = not inactive_mode or day >= voting_start_day

            # In clarified competition mode the price is held constant
            # through the voting-start observation.  In particular, no
            # pre-voting pending move is allowed to cross into that row.
            if inactive_mode and day <= voting_start_day:
                price = config.pre_voting_price
            # The historical boundary event takes precedence over day-zero
            # initialization, so marked_boundary_day=0 starts at reset_price
            # and is represented as the reset row.
            elif reset_applied:
                # The reset overrides the market move that would otherwise
                # have followed the calibration-period action.
                price = config.reset_price
            elif day == 0:
                price = config.initial_price
            else:
                if pending_next_price is None:
                    raise RuntimeError("price schedule missing before a decision")
                price = pending_next_price

            price_path.append(price)
            price_change = (
                None if previous_price is None else price - previous_price
            )
            reset_jump_for_day = price_change if reset_applied else None
            if reset_applied:
                reset_jump = reset_jump_for_day

            # The supplied simulator marks P&L against the prior held
            # position.  The artificial boundary jump is explicitly excluded.
            # In inactive mode, day voting_start_day is an observation only;
            # the first genuine interval is the movement into the next day.
            realised_interval = (
                previous_price is not None
                and (
                    day > voting_start_day
                    if inactive_mode
                    else not reset_applied
                )
            )
            pnl_price_change = (
                0
                if not realised_interval or price_change is None
                else price_change
            )

            if reset_applied or (inactive_mode and day == voting_start_day):
                # Scoring P&L starts from zero at the boundary.  Calibration
                # P&L remains available in its separate ledger.
                for name in self.agent_names:
                    marked_pnl[name] = 0
                    scoring_cumulative_pnl[name] = 0

            daily_pnl: dict[str, int] = {}
            pnl_positions: dict[str, int | None] = {
                name: held_positions[name] if realised_interval else None
                for name in self.agent_names
            }
            pnl_source_day = day - 1 if realised_interval else None
            pnl_majority = previous_vote_majority if realised_interval else None
            for name in self.agent_names:
                daily = held_positions[name] * pnl_price_change
                daily_pnl[name] = daily
                if day < (voting_start_day if inactive_mode else config.marked_boundary_day):
                    calibration_pnl[name] += daily
                    scoring_cumulative_pnl[name] = calibration_pnl[name]
                elif day > (voting_start_day if inactive_mode else config.marked_boundary_day):
                    marked_pnl[name] += daily
                    scoring_cumulative_pnl[name] = marked_pnl[name]

            observations = {
                name: AgentObservation(
                    day=day,
                    price=price,
                    price_history=tuple(price_path),
                    previous_price=previous_price,
                    previous_price_change=price_change,
                    previous_move_is_reset=reset_applied,
                    is_reset_day=reset_applied,
                    marked_boundary_day=config.marked_boundary_day,
                    price_floor=config.price_floor,
                    long_majority_move=config.long_majority_move,
                    short_majority_move=config.short_majority_move,
                    position_limit=config.position_limit,
                    gross_portfolio_budget=config.gross_portfolio_budget,
                    own_position=held_positions[name],
                    voting_active=voting_active,
                    market_mode=config.market_mode,
                    voting_start_day=voting_start_day,
                )
                for name in self.agent_names
            }

            requested_actions: dict[str, object] = {}
            actions: dict[str, int] = {}
            decision_diagnostics: dict[
                str,
                tuple[
                    bool,
                    str | None,
                    float,
                    float,
                    float,
                    bool,
                    bool,
                    str | None,
                ],
            ] = {}
            agent_called_by_name: dict[str, bool] = {}

            # Every observation above is built before any same-day decision is
            # made.  Thus iteration order cannot leak a same-day vote.
            for agent in self.agents:
                name = agent.name
                observation = observations[name]
                agent_called = not (
                    inactive_mode
                    and config.pre_voting_execution == "fully_inactive"
                    and day < voting_start_day
                )
                agent_called_by_name[name] = agent_called
                if not agent_called:
                    # Fully inactive mode represents an unavailable Liferaft
                    # instrument.  A synthetic flat row keeps the audit
                    # schema rectangular without mutating the agent.
                    requested_actions[name] = 0
                    actions[name] = 0
                    decision_diagnostics[name] = (
                        False,
                        None,
                        0.0,
                        0.0,
                        0.0,
                        False,
                        True,
                        "agent was not called before the voting-start day",
                    )
                    continue
                try:
                    requested = agent.decide(observation)
                except Exception as exc:  # add useful context without hiding it
                    raise RuntimeError(
                        f"agent {name!r} failed on day {day}: {exc}"
                    ) from exc

                requested_actions[name] = requested
                if inactive_mode and not voting_active:
                    # Agents may evolve object state while the instrument is
                    # unavailable, but no raw request is a market action.
                    # Keep the request visible without running live validation,
                    # budget rejection, or vote accounting for it.
                    actions[name] = 0
                    decision_diagnostics[name] = (
                        False,
                        None,
                        0.0,
                        0.0,
                        0.0,
                        False,
                        True,
                        "inactive Liferaft market; request retained as diagnostic only",
                    )
                    continue
                other_exposure = self._resolve_other_exposure(name, observation)
                valid, validation_reason = self._validate_action(requested)
                requested_for_budget = requested if valid else 0
                requested_gross = other_exposure + abs(requested_for_budget) * price
                effective = int(requested_for_budget) if valid else 0
                rejection_reason = validation_reason
                action_rejected = validation_reason is not None
                budget_breach = requested_gross > config.gross_portfolio_budget

                if budget_breach:
                    budget_reason = (
                        f"requested gross exposure {requested_gross:g} exceeds "
                        f"budget {config.gross_portfolio_budget:g}"
                    )
                    budget_breaches.append(
                        BudgetBreach(
                            day=day,
                            agent_name=name,
                            requested_action=requested,
                            effective_action=0,
                            current_price=price,
                            other_portfolio_exposure=other_exposure,
                            requested_gross_exposure=requested_gross,
                            budget=config.gross_portfolio_budget,
                            reason=budget_reason,
                        )
                    )
                    if effective != 0:
                        effective = 0
                        action_rejected = True
                        rejection_reason = budget_reason
                    # If the Liferaft request was already flat, the breach is
                    # portfolio-level only. Keep action_rejected/reason tied
                    # to the Liferaft request and emit no rejection record.

                if action_rejected:
                    rejected_actions.append(
                        RejectedAction(
                            day=day,
                            agent_name=name,
                            requested_action=requested,
                            effective_action=effective,
                            current_price=price,
                            other_portfolio_exposure=other_exposure,
                            requested_gross_exposure=requested_gross,
                            reason=rejection_reason or "action rejected",
                        )
                    )

                actual_budget_usage = other_exposure + abs(effective) * price
                actions[name] = effective
                decision_diagnostics[name] = (
                    action_rejected,
                    rejection_reason,
                    other_exposure,
                    requested_gross,
                    actual_budget_usage,
                    budget_breach,
                    False,
                    None,
                )

            long_count = sum(action > 0 for action in actions.values())
            short_count = sum(action < 0 for action in actions.values())
            flat_count = sum(action == 0 for action in actions.values())
            majority = majority_from_counts(long_count, short_count)
            net_margin = long_count - short_count

            before_margin: int | None = None
            after_margin: int | None = None
            focal_pivotal = False
            focal_converted = False
            if self.focal_agent_name is not None and voting_active:
                focal_action = actions[self.focal_agent_name]
                before_margin = net_margin - (1 if focal_action > 0 else 0)
                before_margin += 1 if focal_action < 0 else 0
                after_margin = net_margin
                before_outcome = majority_from_counts(
                    max(before_margin, 0), max(-before_margin, 0)
                )
                after_outcome = majority
                focal_pivotal = before_outcome is not after_outcome
                focal_converted = (
                    focal_action != 0
                    and abs(before_margin) == 1
                    and after_margin == 0
                    and (
                        (focal_action < 0 and before_margin > 0)
                        or (focal_action > 0 and before_margin < 0)
                    )
                )

            if day < config.total_days - 1:
                if inactive_mode and day < voting_start_day:
                    # Do not create a hidden vote whose move appears on the
                    # voting-start observation.
                    unclipped_next_price = None
                    floor_clipped = False
                    next_price = config.pre_voting_price
                else:
                    if majority is MajorityOutcome.LONG:
                        delta = config.long_majority_move
                    elif majority is MajorityOutcome.SHORT:
                        delta = config.short_majority_move
                    else:
                        delta = 0
                    unclipped_next_price = price + delta
                    market_next_price = max(config.price_floor, unclipped_next_price)
                    floor_clipped = market_next_price != unclipped_next_price
                    # The reset is a separate boundary event and overrides the
                    # market-generated next price on the preceding day.
                    next_price = (
                        config.reset_price
                        if not inactive_mode
                        and day + 1 == config.marked_boundary_day
                        else market_next_price
                    )
                pending_next_price = next_price
            else:
                unclipped_next_price = None
                next_price = None
                floor_clipped = False
                pending_next_price = None

            agent_records: dict[str, AgentDayRecord] = {}
            for name in self.agent_names:
                (
                    action_rejected,
                    rejection_reason,
                    other_exposure,
                    requested_gross,
                    actual_budget_usage,
                    budget_breach,
                    market_action_ignored,
                    market_action_ignore_reason,
                ) = decision_diagnostics[name]
                action = actions[name]
                status = self._status_for(action, majority)
                agent_records[name] = AgentDayRecord(
                    day=day,
                    agent_name=name,
                    price=price,
                    requested_action=requested_actions[name],
                    action=action,
                    action_rejected=action_rejected,
                    rejection_reason=rejection_reason,
                    other_portfolio_exposure=other_exposure,
                    requested_gross_exposure=requested_gross,
                    budget_usage=actual_budget_usage,
                    budget_breach=budget_breach,
                    daily_pnl=daily_pnl[name],
                    cumulative_pnl=scoring_cumulative_pnl[name],
                    calibration_cumulative_pnl=calibration_pnl[name],
                    marked_cumulative_pnl=marked_pnl[name],
                    majority=majority,
                    status=status,
                    pnl_position=pnl_positions[name],
                    pnl_source_day=pnl_source_day,
                    pnl_majority=pnl_majority,
                    pnl_status=(
                        self._status_for(pnl_positions[name], pnl_majority)
                        if realised_interval
                        else None
                    ),
                    agent_called=agent_called_by_name[name],
                    market_action_ignored=market_action_ignored,
                    market_action_ignore_reason=market_action_ignore_reason,
                )

            day_records.append(
                DayRecord(
                    day=day,
                    price=price,
                    previous_price=previous_price,
                    price_change=price_change,
                    pnl_price_change=pnl_price_change,
                    voting_active=voting_active,
                    reset_applied=reset_applied,
                    reset_jump=reset_jump_for_day,
                    requested_actions=dict(requested_actions),
                    actions=dict(actions),
                    long_count=long_count,
                    short_count=short_count,
                    flat_count=flat_count,
                    net_vote_margin=net_margin,
                    net_margin_before_focal=before_margin,
                    net_margin_after_focal=after_margin,
                    majority=majority,
                    focal_agent_name=self.focal_agent_name,
                    focal_pivotal=focal_pivotal,
                    focal_converted_one_vote_majority_to_tie=focal_converted,
                    unclipped_next_price=unclipped_next_price,
                    next_price=next_price,
                    floor_clipped=floor_clipped,
                    agent_records=agent_records,
                )
            )

            # Only live actions become market holdings. Inactive requests are
            # intentionally absent from both the next observation and the
            # first live interval's prior position.
            held_positions = (
                dict(actions)
                if not inactive_mode or voting_active
                else {name: 0 for name in self.agent_names}
            )
            previous_vote_majority = majority if voting_active else None

        random_seeds = dict(self.random_seeds)
        for agent in self.agents:
            if agent.name not in random_seeds:
                seed = getattr(agent, "seed", None)
                if seed is None:
                    seed = getattr(agent, "random_seed", None)
                random_seeds[agent.name] = seed

        return SimulationResult(
            config=config,
            days=tuple(day_records),
            price_path=tuple(price_path),
            reset_day=config.marked_boundary_day,
            reset_jump=reset_jump,
            calibration_pnl=dict(calibration_pnl),
            marked_pnl=dict(marked_pnl),
            cumulative_pnl=dict(scoring_cumulative_pnl),
            rejected_actions=tuple(rejected_actions),
            budget_breaches=tuple(budget_breaches),
            random_seeds=random_seeds,
            scenario_name=self.scenario_name,
            scenario_configuration=dict(self.scenario_configuration),
            focal_agent_name=self.focal_agent_name,
        )

    def _resolve_other_exposure(
        self,
        agent_name: str,
        observation: AgentObservation,
    ) -> float:
        exposure_spec = self.other_portfolio_exposure
        if callable(exposure_spec):
            exposure = exposure_spec(agent_name, observation)
        elif isinstance(exposure_spec, Mapping):
            exposure = exposure_spec.get(agent_name, 0.0)
        else:
            exposure = exposure_spec

        if isinstance(exposure, bool):
            raise ValueError("other portfolio exposure must be numeric")
        try:
            numeric_exposure = float(exposure)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid other portfolio exposure for {agent_name!r}: {exposure!r}"
            ) from exc
        if not isfinite(numeric_exposure) or numeric_exposure < 0:
            raise ValueError(
                f"other portfolio exposure must be finite and non-negative: {exposure!r}"
            )
        return numeric_exposure

    def _validate_action(self, requested: object) -> tuple[bool, str | None]:
        if type(requested) is not int:
            return False, "invalid action: expected an integer position"
        limit = self.config.position_limit
        if not -limit <= requested <= limit:
            return False, (
                f"invalid action: {requested!r} outside position limit "
                f"[-{limit}, {limit}]"
            )
        return True, None

    @staticmethod
    def _status_for(
        action: int | None,
        majority: MajorityOutcome | None,
    ) -> SideStatus | None:
        if action is None or majority is None:
            return None
        if action == 0:
            return SideStatus.FLAT
        if majority is MajorityOutcome.TIE:
            return SideStatus.TIED
        is_majority = (
            (action > 0 and majority is MajorityOutcome.LONG)
            or (action < 0 and majority is MajorityOutcome.SHORT)
        )
        return SideStatus.MAJORITY if is_majority else SideStatus.MINORITY
