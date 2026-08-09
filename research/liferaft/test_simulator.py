"""Required mechanics and auditability tests for the Pass 1 simulator."""

from __future__ import annotations

import unittest

from research.liferaft.archetypes import (
    AlwaysFlat,
    AlwaysLong,
    AlwaysShort,
    BoundaryAwareStrategy,
    BoundaryUnawareStrategy,
    LastNMajorityRule,
    PeriodicStrategy,
    PriceLevelFloorAware,
    PreviousMajorityFollower,
    PreviousMajorityPersistenceExploiter,
    SeededRandom,
    StrategySwitchingAgent,
    WinStayLoseShift,
)
from research.liferaft.scenarios import (
    near_balanced_population,
    runaway_price_scenario,
    seeded_random_scenario,
    stateful_reset_scenario,
)
from research.liferaft.simulator import (
    AgentObservation,
    LiferaftConfig,
    LiferaftSimulator,
    MajorityOutcome,
    SideStatus,
    infer_majority_from_price_change,
)


class ScriptedAgent:
    def __init__(self, name: str, actions: list[object]) -> None:
        self.name = name
        self.actions = actions
        self.observations: list[AgentObservation] = []

    def decide(self, observation: AgentObservation) -> object:
        self.observations.append(observation)
        if observation.day < len(self.actions):
            return self.actions[observation.day]
        return self.actions[-1] if self.actions else 0


class CountingFlatAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0
        self.observations: list[AgentObservation] = []

    def decide(self, observation: AgentObservation) -> int:
        self.calls += 1
        self.observations.append(observation)
        return 0


def run_agents(
    agents,
    *,
    total_days: int = 3,
    boundary: int | None = None,
    initial_price: int = 100_000,
    reset_price: int = 100_000,
    other_portfolio_exposure=0.0,
    focal_agent_name: str | None = None,
):
    if boundary is None:
        boundary = total_days - 1
    config = LiferaftConfig(
        total_days=total_days,
        marked_boundary_day=boundary,
        initial_price=initial_price,
        reset_price=reset_price,
    )
    return LiferaftSimulator(
        tuple(agents),
        config,
        other_portfolio_exposure=other_portfolio_exposure,
        focal_agent_name=focal_agent_name,
    ).run()


def observation_for_day(day: int, boundary: int = 5) -> AgentObservation:
    return AgentObservation(
        day=day,
        price=100_000,
        price_history=(100_000,) * (day + 1),
        previous_price=None,
        previous_price_change=None,
        previous_move_is_reset=False,
        is_reset_day=day == boundary,
        marked_boundary_day=boundary,
        price_floor=20_000,
        long_majority_move=-5_000,
        short_majority_move=8_000,
        position_limit=1,
        gross_portfolio_budget=600_000,
        own_position=0,
    )


class LiferaftMechanicsTests(unittest.TestCase):
    def test_all_flat_leaves_price_unchanged(self) -> None:
        result = run_agents([AlwaysFlat("flat")], total_days=4, boundary=3)
        self.assertEqual(result.price_path, (100_000, 100_000, 100_000, 100_000))
        self.assertEqual(result.days[0].majority, MajorityOutcome.TIE)
        self.assertEqual(result.days[0].flat_count, 1)

    def test_long_majority_produces_exactly_minus_5000(self) -> None:
        result = run_agents(
            [AlwaysLong("long-0"), AlwaysLong("long-1"), AlwaysFlat("flat")],
            total_days=3,
            boundary=2,
        )
        self.assertEqual(result.price_path[1], 95_000)
        self.assertEqual(result.days[0].majority, MajorityOutcome.LONG)
        self.assertEqual(result.days[0].unclipped_next_price, 95_000)

    def test_short_majority_produces_exactly_plus_8000(self) -> None:
        result = run_agents(
            [AlwaysShort("short-0"), AlwaysShort("short-1"), AlwaysFlat("flat")],
            total_days=3,
            boundary=2,
        )
        self.assertEqual(result.price_path[1], 108_000)
        self.assertEqual(result.days[0].majority, MajorityOutcome.SHORT)

    def test_tie_leaves_price_unchanged(self) -> None:
        result = run_agents(
            [AlwaysLong("long"), AlwaysShort("short"), AlwaysFlat("flat")],
            total_days=3,
            boundary=2,
        )
        self.assertEqual(result.price_path[1], 100_000)
        self.assertEqual(result.days[0].majority, MajorityOutcome.TIE)

    def test_floor_clips_downward_move(self) -> None:
        result = run_agents(
            [AlwaysLong("long-0"), AlwaysLong("long-1")],
            total_days=3,
            boundary=2,
            initial_price=22_000,
        )
        self.assertEqual(result.price_path[1], 20_000)
        self.assertTrue(result.days[0].floor_clipped)
        self.assertEqual(result.days[0].unclipped_next_price, 17_000)

    def test_realised_pnl_keeps_hidden_majority_for_floor_clipped_move(self) -> None:
        result = run_agents(
            [AlwaysLong("long")],
            total_days=3,
            boundary=2,
            initial_price=22_000,
        )
        record = result.days[1].agent_records["long"]
        self.assertEqual(record.daily_pnl, -2_000)
        self.assertEqual(record.pnl_position, 1)
        self.assertEqual(record.pnl_source_day, 0)
        self.assertEqual(record.pnl_majority, MajorityOutcome.LONG)
        self.assertEqual(record.pnl_status, SideStatus.MAJORITY)

    def test_decisions_are_simultaneous_and_iteration_order_independent(self) -> None:
        def make_agents(reverse: bool):
            agents = [
                SeededRandom("random", p_long=0.4, p_short=0.4, p_flat=0.2, seed=17),
                PeriodicStrategy("periodic", period=4),
                PreviousMajorityPersistenceExploiter("exploiter"),
                PreviousMajorityFollower("follower"),
            ]
            return tuple(reversed(agents)) if reverse else tuple(agents)

        normal = run_agents(make_agents(False), total_days=16, boundary=8)
        reversed_result = run_agents(make_agents(True), total_days=16, boundary=8)
        self.assertEqual(normal.price_path, reversed_result.price_path)
        self.assertEqual(normal.days, reversed_result.days)

    def test_positions_are_integral_and_within_limit(self) -> None:
        result = run_agents(
            [
                ScriptedAgent("too-large", [2, 2]),
                ScriptedAgent("float", [0.5, 0.5]),
                AlwaysLong("valid"),
            ],
            total_days=3,
            boundary=2,
        )
        for day in result.days:
            for action in day.actions.values():
                self.assertIs(type(action), int)
                self.assertLessEqual(abs(action), 1)
        self.assertEqual(result.days[0].actions["too-large"], 0)
        self.assertEqual(result.days[0].actions["float"], 0)
        self.assertEqual(len(result.rejected_actions), 6)

    def test_boundary_resets_price_to_100000(self) -> None:
        result = run_agents(
            [AlwaysShort("short")],
            total_days=4,
            boundary=2,
        )
        self.assertEqual(result.price_path[:3], (100_000, 108_000, 100_000))
        self.assertTrue(result.days[2].reset_applied)
        self.assertEqual(result.days[2].reset_jump, -8_000)
        self.assertEqual(result.reset_day, 2)

    def test_boundary_zero_starts_at_reset_price(self) -> None:
        result = run_agents(
            [AlwaysFlat("flat")],
            total_days=2,
            boundary=0,
            initial_price=90_000,
            reset_price=110_000,
        )

        day_zero = result.days[0]
        self.assertEqual(result.price_path[0], 110_000)
        self.assertTrue(day_zero.reset_applied)
        self.assertEqual(day_zero.price, 110_000)
        record = day_zero.agent_records["flat"]
        self.assertEqual(record.daily_pnl, 0)
        self.assertIsNone(record.pnl_position)
        self.assertIsNone(record.pnl_source_day)
        self.assertIsNone(record.pnl_majority)
        self.assertIsNone(record.pnl_status)

    def test_reset_jump_does_not_earn_pnl(self) -> None:
        result = run_agents(
            [AlwaysShort("short")],
            total_days=5,
            boundary=2,
        )
        # Short earns a loss on the genuine +$8,000 move into day 1, but no
        # artificial profit is booked on the reset into day 2.
        self.assertEqual(result.days[1].agent_records["short"].daily_pnl, -8_000)
        self.assertEqual(result.days[2].agent_records["short"].daily_pnl, 0)
        self.assertEqual(result.days[2].agent_records["short"].marked_cumulative_pnl, 0)
        self.assertEqual(result.days[3].agent_records["short"].daily_pnl, -8_000)

    def test_decision_and_realised_pnl_fields_are_aligned(self) -> None:
        switcher = ScriptedAgent("switcher", [1, -1, -1, -1])
        result = run_agents(
            [AlwaysLong("driver"), switcher],
            total_days=4,
            boundary=3,
        )

        day_zero = result.days[0].agent_records["switcher"]
        self.assertEqual(day_zero.action, 1)
        self.assertEqual(day_zero.majority, MajorityOutcome.LONG)
        self.assertEqual(day_zero.status, SideStatus.MAJORITY)
        self.assertEqual(day_zero.daily_pnl, 0)
        self.assertIsNone(day_zero.pnl_position)
        self.assertIsNone(day_zero.pnl_source_day)
        self.assertIsNone(day_zero.pnl_majority)
        self.assertIsNone(day_zero.pnl_status)

        day_one = result.days[1].agent_records["switcher"]
        self.assertEqual(day_one.action, -1)
        self.assertEqual(day_one.majority, MajorityOutcome.TIE)
        self.assertEqual(day_one.status, SideStatus.TIED)
        self.assertEqual(day_one.daily_pnl, -5_000)
        self.assertEqual(day_one.pnl_position, 1)
        self.assertEqual(day_one.pnl_source_day, 0)
        self.assertEqual(day_one.pnl_majority, MajorityOutcome.LONG)
        self.assertEqual(day_one.pnl_status, SideStatus.MAJORITY)

        # The tie on day 1 creates a genuine zero movement into day 2. The
        # realised vote remains known even though public inference is not.
        day_two = result.days[2].agent_records["switcher"]
        self.assertEqual(day_two.daily_pnl, 0)
        self.assertEqual(day_two.pnl_position, -1)
        self.assertEqual(day_two.pnl_source_day, 1)
        self.assertEqual(day_two.pnl_majority, MajorityOutcome.TIE)
        self.assertEqual(day_two.pnl_status, SideStatus.TIED)

        reset_day = result.days[3].agent_records["switcher"]
        self.assertEqual(reset_day.daily_pnl, 0)
        self.assertIsNone(reset_day.pnl_position)
        self.assertIsNone(reset_day.pnl_source_day)
        self.assertIsNone(reset_day.pnl_majority)
        self.assertIsNone(reset_day.pnl_status)

    def test_marked_pnl_resets_and_contains_only_marked_movements(self) -> None:
        result = run_agents(
            [AlwaysShort("short")],
            total_days=4,
            boundary=2,
        )
        self.assertEqual(result.calibration_pnl["short"], -8_000)
        self.assertEqual(result.marked_pnl["short"], -8_000)
        self.assertEqual(result.days[2].agent_records["short"].cumulative_pnl, 0)
        self.assertEqual(result.days[3].agent_records["short"].cumulative_pnl, -8_000)

    def test_state_and_public_history_persist_across_boundary(self) -> None:
        driver = ScriptedAgent("driver", [-1, 0, 0, 0, 0])
        recorder = CountingFlatAgent("recorder")
        result = run_agents([driver, recorder], total_days=5, boundary=2)
        self.assertEqual(recorder.calls, 5)
        self.assertEqual(len(recorder.observations[2].price_history), 3)
        self.assertEqual(recorder.observations[2].price_history, (100_000, 108_000, 100_000))
        self.assertTrue(recorder.observations[2].is_reset_day)
        self.assertEqual(result.days[2].price, 100_000)

    def test_boundary_aware_and_unaware_differ_at_reset(self) -> None:
        driver = ScriptedAgent("driver", [-1, 0, 0, 0])
        aware = BoundaryAwareStrategy("aware")
        unaware = BoundaryUnawareStrategy("unaware")
        result = run_agents([driver, aware, unaware], total_days=4, boundary=2)
        self.assertEqual(result.days[2].actions["aware"], 1)
        self.assertEqual(result.days[2].actions["unaware"], -1)

    def test_public_majority_inference_handles_clips_zero_and_resets(self) -> None:
        self.assertEqual(
            infer_majority_from_price_change(-5_000),
            MajorityOutcome.LONG,
        )
        self.assertEqual(
            infer_majority_from_price_change(8_000),
            MajorityOutcome.SHORT,
        )
        self.assertEqual(
            infer_majority_from_price_change(20_000 - 22_000),
            MajorityOutcome.LONG,
        )
        self.assertIsNone(infer_majority_from_price_change(0))
        self.assertIsNone(
            infer_majority_from_price_change(-5_000, previous_move_is_reset=True)
        )
        self.assertIsNone(
            infer_majority_from_price_change(8_000, previous_move_is_reset=True)
        )

        reset_observation = AgentObservation(
            day=5,
            price=100_000,
            price_history=(108_000, 100_000),
            previous_price=108_000,
            previous_price_change=-8_000,
            previous_move_is_reset=True,
            is_reset_day=True,
            marked_boundary_day=5,
            price_floor=20_000,
            long_majority_move=-5_000,
            short_majority_move=8_000,
            position_limit=1,
            gross_portfolio_budget=600_000,
            own_position=1,
        )
        self.assertIsNone(reset_observation.previous_inferred_majority)

    def test_periodic_boundary_phase_restarts_only_at_boundary(self) -> None:
        reset_periodic = PeriodicStrategy(
            "reset-periodic",
            period=3,
            pattern=(1, -1, 0),
            reset_phase_at_boundary=True,
        )
        no_reset_periodic = PeriodicStrategy(
            "ordinary-periodic",
            period=3,
            pattern=(1, -1, 0),
            reset_phase_at_boundary=False,
        )
        reset_actions = [
            reset_periodic.decide(observation_for_day(day))
            for day in range(8)
        ]
        ordinary_actions = [
            no_reset_periodic.decide(observation_for_day(day))
            for day in range(8)
        ]
        self.assertEqual(reset_actions, [1, -1, 0, 1, -1, 1, -1, 0])
        self.assertEqual(ordinary_actions, [1, -1, 0, 1, -1, 0, 1, -1])

    def test_seeded_runs_are_exactly_reproducible(self) -> None:
        def make_agents():
            return [
                SeededRandom("random-a", p_long=0.25, p_short=0.5, p_flat=0.25, seed=3),
                SeededRandom("random-b", p_long=0.5, p_short=0.25, p_flat=0.25, seed=4),
            ]

        first = run_agents(make_agents(), total_days=25, boundary=12)
        second = run_agents(make_agents(), total_days=25, boundary=12)
        self.assertEqual(first, second)

    def test_reusing_stateful_scenario_creates_fresh_agents(self) -> None:
        scenario = stateful_reset_scenario(total_days=10, marked_boundary_day=5)
        first = scenario.run()
        second = scenario.run()
        self.assertEqual(first, second)
        self.assertEqual(first.random_seeds["win-stay"], 11)

    def test_reusing_seeded_random_scenario_reports_fresh_seeds(self) -> None:
        scenario = seeded_random_scenario(
            total_days=14,
            marked_boundary_day=7,
            count=3,
            seed=41,
        )
        first = scenario.run()
        second = scenario.run()
        self.assertEqual(first, second)
        self.assertEqual(
            first.random_seeds,
            {"random-0": 41, "random-1": 42, "random-2": 43},
        )

    def test_reusing_one_simulator_is_rejected_explicitly(self) -> None:
        simulator = LiferaftSimulator(
            [SeededRandom("random", seed=9)],
            LiferaftConfig(total_days=3, marked_boundary_day=2),
        )
        simulator.run()
        with self.assertRaisesRegex(RuntimeError, "may only be called once"):
            simulator.run()

    def test_margins_and_pivotality_diagnostics(self) -> None:
        agents = near_balanced_population()
        result = run_agents(
            agents,
            total_days=2,
            boundary=1,
            focal_agent_name="focal",
        )
        day = result.days[0]
        self.assertEqual(day.net_margin_before_focal, 1)
        self.assertEqual(day.net_margin_after_focal, 0)
        self.assertEqual(day.net_vote_margin, 0)
        self.assertEqual(day.majority, MajorityOutcome.TIE)
        self.assertTrue(day.focal_pivotal)
        self.assertTrue(day.focal_converted_one_vote_majority_to_tie)

    def test_budget_infeasible_action_is_flattened_and_logged(self) -> None:
        result = run_agents(
            [AlwaysLong("ticket")],
            total_days=3,
            boundary=2,
            other_portfolio_exposure={"ticket": 500_001},
        )
        day = result.days[0]
        self.assertEqual(day.requested_actions["ticket"], 1)
        self.assertEqual(day.actions["ticket"], 0)
        self.assertTrue(day.agent_records["ticket"].action_rejected)
        self.assertTrue(day.agent_records["ticket"].budget_breach)
        self.assertEqual(len(result.budget_breaches), 3)
        self.assertIn("exceeds budget", result.rejected_actions[0].reason)

    def test_flat_request_with_excess_other_exposure_is_not_rejected(self) -> None:
        result = run_agents(
            [AlwaysFlat("flat")],
            total_days=1,
            boundary=0,
            other_portfolio_exposure={"flat": 600_001},
        )
        record = result.days[0].agent_records["flat"]
        self.assertTrue(record.budget_breach)
        self.assertFalse(record.action_rejected)
        self.assertIsNone(record.rejection_reason)
        self.assertEqual(len(result.budget_breaches), 1)
        self.assertEqual(len(result.rejected_actions), 0)

    def test_nonzero_request_with_excess_combined_exposure_is_rejected(self) -> None:
        result = run_agents(
            [AlwaysLong("ticket")],
            total_days=1,
            boundary=0,
            other_portfolio_exposure={"ticket": 500_001},
        )
        record = result.days[0].agent_records["ticket"]
        self.assertTrue(record.budget_breach)
        self.assertTrue(record.action_rejected)
        self.assertIn("exceeds budget", record.rejection_reason or "")
        self.assertEqual(len(result.budget_breaches), 1)
        self.assertEqual(len(result.rejected_actions), 1)

    def test_invalid_action_and_excess_exposure_keep_both_diagnostics(self) -> None:
        result = run_agents(
            [ScriptedAgent("invalid", [2])],
            total_days=1,
            boundary=0,
            other_portfolio_exposure={"invalid": 600_001},
        )
        record = result.days[0].agent_records["invalid"]
        self.assertTrue(record.budget_breach)
        self.assertTrue(record.action_rejected)
        self.assertEqual(
            record.rejection_reason,
            "invalid action: 2 outside position limit [-1, 1]",
        )
        self.assertEqual(len(result.budget_breaches), 1)
        self.assertEqual(len(result.rejected_actions), 1)

    def test_exposure_exactly_equal_to_budget_is_allowed_for_flat_request(self) -> None:
        result = run_agents(
            [AlwaysFlat("flat")],
            total_days=1,
            boundary=0,
            other_portfolio_exposure={"flat": 600_000},
        )
        record = result.days[0].agent_records["flat"]
        self.assertFalse(record.budget_breach)
        self.assertFalse(record.action_rejected)
        self.assertIsNone(record.rejection_reason)
        self.assertEqual(result.budget_breaches, ())
        self.assertEqual(result.rejected_actions, ())

    def test_price_can_exceed_budget_but_ticket_then_becomes_infeasible(self) -> None:
        result = runaway_price_scenario(total_days=80).run()
        self.assertGreater(max(result.price_path), 600_000)
        first = result.budget_breaches[0]
        self.assertEqual(first.day, 63)
        self.assertEqual(first.current_price, 604_000)
        self.assertEqual(result.days[63].actions["short-0"], 0)
        self.assertEqual(result.days[62].actions["short-0"], -1)

    def test_long_and_short_pnl_signs_are_correct(self) -> None:
        long_majority = run_agents(
            [
                AlwaysLong("long-majority-0"),
                AlwaysLong("long-majority-1"),
                AlwaysShort("short-minority"),
            ],
            total_days=3,
            boundary=2,
        )
        self.assertEqual(
            long_majority.days[1].agent_records["long-majority-0"].daily_pnl,
            -5_000,
        )
        self.assertEqual(
            long_majority.days[1].agent_records["short-minority"].daily_pnl,
            5_000,
        )

        short_majority = run_agents(
            [
                AlwaysShort("short-majority-0"),
                AlwaysShort("short-majority-1"),
                AlwaysLong("long-minority"),
            ],
            total_days=3,
            boundary=2,
        )
        self.assertEqual(
            short_majority.days[1].agent_records["short-majority-0"].daily_pnl,
            -8_000,
        )
        self.assertEqual(
            short_majority.days[1].agent_records["long-minority"].daily_pnl,
            8_000,
        )

    def test_archetypes_have_expected_basic_behaviour(self) -> None:
        config = LiferaftConfig(total_days=4, marked_boundary_day=2)
        observation = AgentObservation(
            day=0,
            price=100_000,
            price_history=(100_000,),
            previous_price=None,
            previous_price_change=None,
            previous_move_is_reset=False,
            is_reset_day=False,
            marked_boundary_day=2,
            price_floor=20_000,
            long_majority_move=-5_000,
            short_majority_move=8_000,
            position_limit=1,
            gross_portfolio_budget=600_000,
            own_position=0,
        )
        agents = [
            AlwaysFlat("flat"),
            AlwaysLong("long"),
            AlwaysShort("short"),
            SeededRandom("random", seed=1),
            PreviousMajorityPersistenceExploiter("exploiter"),
            PreviousMajorityFollower("follower"),
            WinStayLoseShift("win", initial_action=1, seed=2),
            PeriodicStrategy("periodic-2", period=2),
            PeriodicStrategy("periodic-3", period=3),
            PeriodicStrategy("periodic-4", period=4),
            LastNMajorityRule("last-n"),
            PriceLevelFloorAware("floor"),
            BoundaryAwareStrategy("aware"),
            BoundaryUnawareStrategy("unaware"),
            StrategySwitchingAgent("switch", AlwaysFlat("pre"), AlwaysLong("post")),
        ]
        self.assertEqual(config.position_limit, 1)
        for agent in agents:
            action = agent.decide(observation)
            self.assertIs(type(action), int, msg=agent.name)
            self.assertIn(action, (-1, 0, 1), msg=agent.name)
        self.assertEqual(PriceLevelFloorAware("floor").decide(observation), 0)

    def test_status_records_majority_minority_flat_and_tied(self) -> None:
        result = run_agents(
            [AlwaysLong("long"), AlwaysShort("short"), AlwaysFlat("flat")],
            total_days=3,
            boundary=2,
        )
        records = result.days[0].agent_records
        self.assertEqual(records["long"].status, SideStatus.TIED)
        self.assertEqual(records["short"].status, SideStatus.TIED)
        self.assertEqual(records["flat"].status, SideStatus.FLAT)


if __name__ == "__main__":
    unittest.main()
