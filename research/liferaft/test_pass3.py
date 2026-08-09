"""Mechanics, causality, and cold-start strategy tests for Pass 3."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from research.liferaft.archetypes import AlwaysLong, AlwaysShort
from research.liferaft.cold_start_strategies import (
    COLD_START_STRATEGY_NAMES,
    OnlineDriftAware,
    OnlineExpertEnsemble,
    OnlineLastMajorityCounter,
    OnlineRegularisedMarkov,
    OnlineRollingFrequency,
    live_labels_from_observation,
)
from research.liferaft.scenarios import seeded_random_scenario, stateful_reset_scenario
import research.liferaft.pass3_scenarios as pass3_scenarios
from research.liferaft.pass3_scenarios import validation_scenarios
from research.liferaft.pass3_experiments import (
    PathAuditRecord,
    _metric_without_comparison,
    path_diversity_summaries,
    run_portfolio_sensitivity,
)
from research.liferaft.simulator import (
    AgentObservation,
    LiferaftConfig,
    LiferaftSimulator,
    MajorityOutcome,
    SideStatus,
    infer_majority_from_price_change,
)
from research.liferaft.strategies import RegularisedMarkovModel


class ScriptedAgent:
    def __init__(self, name: str, actions: list[object]) -> None:
        self.name = name
        self.actions = actions
        self.observations: list[AgentObservation] = []

    def decide(self, observation: AgentObservation) -> object:
        self.observations.append(observation)
        return self.actions[min(observation.day, len(self.actions) - 1)]


class CountingAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0
        self.observations: list[AgentObservation] = []

    def decide(self, observation: AgentObservation) -> int:
        self.calls += 1
        self.observations.append(observation)
        return 1 if observation.day >= observation.marked_boundary_day else 0


class OwnPositionBoundaryAgent:
    """Stateful regression agent for ignored-position semantics."""

    def __init__(self, name: str = "stateful") -> None:
        self.name = name
        self.calls = 0
        self.observations: list[AgentObservation] = []

    def decide(self, observation: AgentObservation) -> int:
        self.calls += 1
        self.observations.append(observation)
        if observation.day < observation.marked_boundary_day:
            return 1
        # The boundary request must see a flat effective position even though
        # the object has been called throughout the inactive period.
        return 1 if observation.own_position == 0 else -1


class RecordingEnsemble(OnlineExpertEnsemble):
    def __init__(self) -> None:
        super().__init__(name="recording-ensemble")
        self.weights_by_day: dict[int, dict[str, float]] = {}

    def decide(self, observation: AgentObservation) -> int:
        action = super().decide(observation)
        self.weights_by_day[observation.day] = self.weights
        return action


def population_majority_path(
    scenario: pass3_scenarios.Pass3Scenario,
    *,
    start: int = 365,
    end: int = 730,
) -> tuple[MajorityOutcome, ...]:
    """Inspect a fresh hidden population for scenario-construction tests."""

    agents = scenario.population_factory()
    labels: list[MajorityOutcome] = []
    for day in range(start, end):
        actions = [agent.decide(SimpleNamespace(day=day)) for agent in agents]
        long_count = sum(action > 0 for action in actions)
        short_count = sum(action < 0 for action in actions)
        labels.append(
            MajorityOutcome.LONG
            if long_count > short_count
            else MajorityOutcome.SHORT
            if short_count > long_count
            else MajorityOutcome.TIE
        )
    return tuple(labels)


def inactive_config(
    *,
    total_days: int = 8,
    boundary: int = 3,
    execution: str = "observe_and_ignore_actions",
    pre_voting_price: int = 100_000,
    price_floor: int = 20_000,
) -> LiferaftConfig:
    return LiferaftConfig(
        total_days=total_days,
        marked_boundary_day=boundary,
        initial_price=pre_voting_price,
        reset_price=pre_voting_price,
        pre_voting_price=pre_voting_price,
        price_floor=price_floor,
        market_mode="inactive_until_marked",
        pre_voting_execution=execution,
    )


class Pass3SimulatorTests(unittest.TestCase):
    def test_inactive_price_and_pnl_ignore_pre_voting_actions(self) -> None:
        agent = ScriptedAgent("focal", [1, -1, 1, -1, -1])
        result = LiferaftSimulator(
            [agent],
            inactive_config(),
            focal_agent_name="focal",
        ).run()

        self.assertEqual(result.price_path[:4], (100_000,) * 4)
        self.assertEqual(
            [day.agent_records["focal"].daily_pnl for day in result.days[:4]],
            [0, 0, 0, 0],
        )
        self.assertTrue(all(not day.voting_active for day in result.days[:3]))
        self.assertTrue(result.days[3].voting_active)
        self.assertTrue(all(not day.reset_applied for day in result.days))
        self.assertIsNone(result.reset_jump)
        self.assertEqual(result.days[2].next_price, 100_000)
        self.assertIsNone(result.days[2].unclipped_next_price)

    def test_voting_start_action_sets_first_live_movement(self) -> None:
        focal = ScriptedAgent("focal", [0, 0, 0, -1])
        result = LiferaftSimulator(
            [focal, AlwaysShort("opponent")],
            inactive_config(),
            focal_agent_name="focal",
        ).run()

        self.assertEqual(result.days[3].actions["focal"], -1)
        self.assertEqual(result.days[3].majority, MajorityOutcome.SHORT)
        self.assertEqual(result.days[3].next_price, 108_000)
        first_live = result.days[4].agent_records["focal"]
        self.assertEqual(first_live.daily_pnl, -8_000)
        self.assertEqual(first_live.pnl_position, -1)
        self.assertEqual(first_live.pnl_source_day, 3)
        self.assertEqual(first_live.pnl_majority, MajorityOutcome.SHORT)

    def test_fully_inactive_does_not_call_agents_before_boundary(self) -> None:
        agent = CountingAgent("counting")
        result = LiferaftSimulator(
            [agent],
            inactive_config(execution="fully_inactive"),
        ).run()

        self.assertEqual(agent.calls, 8 - 3)
        self.assertTrue(
            all(not day.agent_records["counting"].agent_called for day in result.days[:3])
        )
        self.assertTrue(result.days[3].agent_records["counting"].agent_called)
        self.assertEqual([day.actions["counting"] for day in result.days[:3]], [0, 0, 0])
        self.assertEqual(result.price_path[:4], (100_000,) * 4)

    def test_observe_mode_calls_agents_and_retains_flat_history(self) -> None:
        agent = CountingAgent("counting")
        result = LiferaftSimulator(
            [agent],
            inactive_config(execution="observe_and_ignore_actions"),
        ).run()

        self.assertEqual(agent.calls, 8)
        self.assertEqual(len(result.days[3].agent_records), 1)
        self.assertEqual(agent.observations[3].price_history, (100_000,) * 4)
        self.assertEqual(agent.observations[3].previous_inferred_majority, None)
        self.assertTrue(all(day.price == 100_000 for day in result.days[:4]))

    def test_ignored_requests_do_not_become_own_position_or_live_rejections(self) -> None:
        agent = OwnPositionBoundaryAgent()
        result = LiferaftSimulator(
            [agent],
            inactive_config(),
            focal_agent_name=agent.name,
        ).run()

        self.assertEqual(agent.calls, 8)
        self.assertTrue(all(observation.own_position == 0 for observation in agent.observations[:4]))
        self.assertEqual(result.days[3].requested_actions[agent.name], 1)
        self.assertEqual(result.days[3].actions[agent.name], 1)
        self.assertEqual(result.days[0].actions[agent.name], 0)
        self.assertTrue(result.days[0].agent_records[agent.name].market_action_ignored)
        self.assertFalse(result.days[0].agent_records[agent.name].action_rejected)
        self.assertFalse(result.days[0].agent_records[agent.name].budget_breach)
        self.assertEqual(result.rejected_actions, ())
        self.assertEqual(result.budget_breaches, ())

    def test_invalid_inactive_request_is_retained_without_live_rejection(self) -> None:
        agent = ScriptedAgent("invalid", ["not-an-action", "not-an-action", "not-an-action", 0])
        result = LiferaftSimulator([agent], inactive_config()).run()
        record = result.days[0].agent_records[agent.name]
        self.assertEqual(record.requested_action, "not-an-action")
        self.assertEqual(record.action, 0)
        self.assertFalse(record.action_rejected)
        self.assertTrue(record.market_action_ignored)
        self.assertEqual(result.rejected_actions, ())

    def test_record_fields_align_current_decision_and_realised_interval(self) -> None:
        focal = ScriptedAgent("focal", [1, 1, 1, 1, -1])
        result = LiferaftSimulator(
            [focal, AlwaysLong("opponent")],
            inactive_config(),
            focal_agent_name="focal",
        ).run()

        current = result.days[4].agent_records["focal"]
        # The day-4 row deliberately contains both the newly chosen day-4
        # action and the return realised from the day-3 holding.
        realised = current
        self.assertEqual(current.action, -1)
        self.assertEqual(current.majority, MajorityOutcome.TIE)
        self.assertEqual(current.status, SideStatus.TIED)
        self.assertEqual(realised.daily_pnl, -5_000)
        self.assertEqual(realised.pnl_position, 1)
        self.assertEqual(realised.pnl_source_day, 3)
        self.assertEqual(realised.pnl_majority, MajorityOutcome.LONG)
        self.assertEqual(realised.pnl_status, SideStatus.MAJORITY)

    def test_pnl_attribution_uses_source_day_pivotality(self) -> None:
        focal = ScriptedAgent("focal", [0, 0, 0, 1, -1, -1])
        opponents = [
            ScriptedAgent(f"opponent-{index}", [0, 0, 0, 0, 1, 1])
            for index in range(3)
        ]
        result = LiferaftSimulator(
            [focal, *opponents],
            inactive_config(total_days=5),
            focal_agent_name="focal",
        ).run()

        # Day 3's lone focal long vote is pivotal. On day 4 the focal action
        # changes, while three opponent longs make the current vote non-pivotal.
        self.assertTrue(result.days[3].focal_pivotal)
        self.assertFalse(result.days[4].focal_pivotal)
        record = result.days[4].agent_records["focal"]
        self.assertEqual(record.pnl_source_day, 3)
        metric = _metric_without_comparison(
            result,
            suite="test",
            scenario=validation_scenarios()[0],
            strategy="test",
        )
        self.assertEqual(metric.pivotal_pnl, -5_000)
        self.assertEqual(metric.non_pivotal_pnl, 0)
        self.assertEqual(
            metric.pivotal_pnl + metric.non_pivotal_pnl,
            metric.marked_pnl,
        )

    def test_floor_zero_movement_keeps_hidden_realised_majority(self) -> None:
        result = LiferaftSimulator(
            [AlwaysLong("focal"), AlwaysLong("opponent")],
            inactive_config(pre_voting_price=20_000, price_floor=20_000),
            focal_agent_name="focal",
        ).run()
        record = result.days[4].agent_records["focal"]
        self.assertEqual(result.days[3].next_price, 20_000)
        self.assertEqual(record.daily_pnl, 0)
        self.assertEqual(record.pnl_majority, MajorityOutcome.LONG)
        self.assertEqual(record.pnl_status, SideStatus.MAJORITY)

    def test_public_inference_handles_clipping_and_reset_canonical_sizes(self) -> None:
        self.assertIs(
            infer_majority_from_price_change(-5_000), MajorityOutcome.LONG
        )
        self.assertIs(
            infer_majority_from_price_change(8_000), MajorityOutcome.SHORT
        )
        self.assertIs(
            infer_majority_from_price_change(-2_000), MajorityOutcome.LONG
        )
        self.assertIsNone(infer_majority_from_price_change(0))
        self.assertIsNone(
            infer_majority_from_price_change(-5_000, previous_move_is_reset=True)
        )
        self.assertIsNone(
            infer_majority_from_price_change(8_000, previous_move_is_reset=True)
        )

    def test_flat_history_has_no_live_labels_or_inferred_majority(self) -> None:
        observation = AgentObservation(
            day=3,
            price=100_000,
            price_history=(100_000,) * 4,
            previous_price=100_000,
            previous_price_change=0,
            previous_move_is_reset=False,
            is_reset_day=False,
            marked_boundary_day=3,
            price_floor=20_000,
            long_majority_move=-5_000,
            short_majority_move=8_000,
            position_limit=1,
            gross_portfolio_budget=600_000,
            own_position=0,
            voting_active=True,
            market_mode="inactive_until_marked",
            voting_start_day=3,
        )
        self.assertIsNone(observation.previous_inferred_majority)
        self.assertEqual(live_labels_from_observation(observation), ())

    def test_online_strategy_updates_after_first_live_outcome(self) -> None:
        focal = OnlineLastMajorityCounter()
        result = LiferaftSimulator(
            [focal, AlwaysLong("opponent")],
            inactive_config(),
            focal_agent_name=focal.name,
        ).run()

        self.assertEqual(result.days[3].agent_records[focal.name].action, 0)
        self.assertEqual(result.days[4].agent_records[focal.name].action, -1)
        self.assertEqual(focal.observed_live_count, 4)
        self.assertEqual(focal.labels[0], MajorityOutcome.LONG)

    def test_ensemble_updates_after_observed_outcome_and_stays_bounded(self) -> None:
        focal = RecordingEnsemble()
        LiferaftSimulator(
            [focal, AlwaysLong("opponent")],
            inactive_config(total_days=8),
            focal_agent_name=focal.name,
        ).run()

        initial = focal.weights_by_day[3]
        # The first outcome is compared with an intentionally unsupported
        # 0.5/0.5 forecast. The next decision has one observed label and can
        # update the expert weights causally.
        after_supported_outcome = focal.weights_by_day[5]
        self.assertEqual(initial, {name: 0.25 for name in initial})
        self.assertNotEqual(after_supported_outcome, initial)
        self.assertAlmostEqual(sum(after_supported_outcome.values()), 1.0)
        self.assertTrue(all(0.05 - 1e-9 <= weight <= 0.70 + 1e-9 for weight in after_supported_outcome.values()))

    def test_drift_does_not_score_unsupported_or_undirected_forecasts(self) -> None:
        focal = OnlineDriftAware()
        LiferaftSimulator(
            [focal, AlwaysLong("opponent-0"), AlwaysLong("opponent-1"), AlwaysLong("opponent-2")],
            inactive_config(total_days=9),
            focal_agent_name=focal.name,
        ).run()

        # The first three live outcomes arrive before the ensemble has the
        # declared support of three; only the two later supported forecasts
        # enter the quality sample.
        self.assertEqual(focal.quality_observations, 2)
        self.assertFalse(focal.fallback_active)

    def test_gradual_drift_has_mirrored_directional_population_paths(self) -> None:
        scenarios = [
            scenario
            for scenario in validation_scenarios()
            if scenario.family == "gradual_drift"
            and scenario.execution_mode == "fully_inactive"
        ]
        self.assertEqual({scenario.drift_direction for scenario in scenarios}, {-1, 1})
        self.assertEqual(
            {scenario.drift_strength for scenario in scenarios},
            set(pass3_scenarios.DRIFT_STRENGTHS),
        )
        self.assertTrue(any(scenario.population_size % 2 == 0 for scenario in scenarios))
        self.assertTrue(any(scenario.population_size % 2 == 1 for scenario in scenarios))

        grouped: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for scenario in scenarios:
            path = population_majority_path(scenario)
            early = path[:60]
            late = path[-60:]
            early_long = sum(value is MajorityOutcome.LONG for value in early) / len(early)
            late_long = sum(value is MajorityOutcome.LONG for value in late) / len(late)
            early_short = sum(value is MajorityOutcome.SHORT for value in early) / len(early)
            late_short = sum(value is MajorityOutcome.SHORT for value in late) / len(late)
            grouped.setdefault(
                (scenario.drift_direction or 0, scenario.population_size % 2),
                [],
            ).append((late_long - early_long, late_short - early_short))

        for (direction, _parity), changes in grouped.items():
            long_change = sum(change[0] for change in changes) / len(changes)
            short_change = sum(change[1] for change in changes) / len(changes)
            if direction == 1:
                self.assertGreater(long_change, 0.05)
            else:
                self.assertGreater(short_change, 0.05)

    def test_gradual_drift_controlled_execution_pairs_have_equal_paths(self) -> None:
        scenarios = [
            scenario
            for scenario in validation_scenarios()
            if scenario.family == "gradual_drift"
        ]
        pairs: dict[str, dict[str, pass3_scenarios.Pass3Scenario]] = {}
        for scenario in scenarios:
            pairs.setdefault(scenario.pair_id or "", {})[scenario.execution_mode] = scenario
        self.assertEqual(len(pairs), len(pass3_scenarios.VALIDATION_OFFSETS))
        for pair in pairs.values():
            observed = pair["observe_and_ignore_actions"].run("flat")
            inactive = pair["fully_inactive"].run("flat")
            self.assertEqual(
                pair["observe_and_ignore_actions"].live_path_signature(observed),
                pair["fully_inactive"].live_path_signature(inactive),
            )

    def test_persistent_population_keeps_seeded_random_component_and_dominance(self) -> None:
        long_factory = pass3_scenarios._persistent_population(
            20_000,
            base_action=1,
            count=4,
            noise_probability=0.18,
        )
        short_factory = pass3_scenarios._persistent_population(
            20_000,
            base_action=-1,
            count=4,
            noise_probability=0.18,
        )
        long_agents = long_factory()
        long_again = long_factory()
        short_agents = short_factory()
        self.assertIsInstance(long_agents[0], pass3_scenarios.DayIndexedRandomAgent)
        self.assertTrue(
            all(
                isinstance(agent, pass3_scenarios.NoisyPersistenceAgent)
                for agent in long_agents[1:]
            )
        )
        self.assertTrue(
            all(getattr(agent, "_event_day", None) is None for agent in long_agents[1:])
        )
        self.assertEqual(
            [agent._schedule for agent in long_agents],
            [agent._schedule for agent in long_again],
        )

        other_seed_agents = pass3_scenarios._persistent_population(
            20_001,
            base_action=1,
            count=4,
            noise_probability=0.18,
        )()
        self.assertGreater(
            sum(
                left != right
                for left, right in zip(
                    long_agents[0]._schedule[365:],
                    other_seed_agents[0]._schedule[365:],
                )
            ),
            10,
        )

        long_path = population_majority_path(
            pass3_scenarios._scenario(
                name="test-persistent-long",
                family="persistent_long",
                seed=20_000,
                population_factory=long_factory,
                population_description="test",
                execution_mode="fully_inactive",
                population_size=4,
            )
        )
        short_path = population_majority_path(
            pass3_scenarios._scenario(
                name="test-persistent-short",
                family="persistent_short",
                seed=20_000,
                population_factory=short_factory,
                population_description="test",
                execution_mode="fully_inactive",
                population_size=4,
            )
        )
        self.assertGreater(
            sum(value is MajorityOutcome.LONG for value in long_path),
            sum(value is MajorityOutcome.SHORT for value in long_path),
        )
        self.assertGreater(
            sum(value is MajorityOutcome.SHORT for value in short_path),
            sum(value is MajorityOutcome.LONG for value in short_path),
        )
        self.assertNotEqual(long_path, short_path)

    def test_path_diversity_reports_unique_and_hamming_diagnostics(self) -> None:
        entries = tuple(
            PathAuditRecord(
                suite="validation",
                family="test-family",
                seed=seed,
                execution_mode="fully_inactive",
                pair_id=None,
                path_controlled=True,
                signature=(
                    (100_000,) * 4,
                    path,
                ),
            )
            for seed, path in enumerate(
                (
                    ("long", "long", "short", "tie"),
                    ("long", "short", "short", "tie"),
                    ("short", "short", "short", "tie"),
                )
            )
        )
        summary = path_diversity_summaries(entries)[0]
        self.assertEqual(summary.cases, 3)
        self.assertEqual(summary.unique_signatures, 3)
        self.assertEqual(summary.duplicate_cases, 0)
        self.assertEqual(summary.minimum_pairwise_hamming_days, 1)
        self.assertAlmostEqual(summary.mean_pairwise_hamming_days, 4 / 3)

    def test_pass3_scenario_exposure_is_optional_and_audited(self) -> None:
        scenario = validation_scenarios()[0]
        zero = scenario.run("flat")
        explicit_zero = scenario.run("flat", other_portfolio_exposure=0.0)
        exposed = scenario.run("burnin1_markov", other_portfolio_exposure=150_000)
        self.assertEqual(zero, explicit_zero)
        self.assertEqual(
            exposed.scenario_configuration["other_portfolio_exposure"],
            150_000,
        )
        focal_name = exposed.focal_agent_name
        self.assertIsNotNone(focal_name)
        assert focal_name is not None
        self.assertTrue(
            all(
                record.other_portfolio_exposure == 150_000
                for record in exposed.agent_history(focal_name)
                if record.day >= exposed.config.voting_start_day
            )
        )
        opponent_name = next(name for name in exposed.days[365].agent_records if name != focal_name)
        self.assertEqual(
            exposed.days[365].agent_records[opponent_name].other_portfolio_exposure,
            0.0,
        )

    def test_portfolio_sensitivity_supports_predeclared_subset_without_final(self) -> None:
        metrics = run_portfolio_sensitivity(
            validation_scenarios()[:1],
            exposures=(0, 450_000),
            strategies=("flat", "burnin1_markov"),
            write_report=False,
        )
        self.assertEqual(len(metrics), 4)
        self.assertEqual(
            {metric.other_portfolio_exposure for metric in metrics},
            {0.0, 450_000.0},
        )

    def test_validation_suite_has_paired_varied_seeded_cases(self) -> None:
        scenarios = validation_scenarios()
        self.assertEqual(len(scenarios), 480)
        self.assertEqual({scenario.population_size for scenario in scenarios}, {3, 4, 5, 8, 9, 15})
        by_pair: dict[str, list[object]] = {}
        for scenario in scenarios:
            self.assertIn(scenario.family, {item.family for item in scenarios})
            by_pair.setdefault(scenario.pair_id or "", []).append(scenario)
        self.assertTrue(by_pair)
        self.assertTrue(all(len(pair) == 2 for pair in by_pair.values()))
        self.assertEqual(
            {scenario.execution_mode for scenario in scenarios},
            {"observe_and_ignore_actions", "fully_inactive"},
        )

    def test_all_cold_start_candidates_use_integral_positions(self) -> None:
        for name in COLD_START_STRATEGY_NAMES:
            result = validation_scenarios()[0].run(name)
            focal_name = result.focal_agent_name
            self.assertIsNotNone(focal_name)
            assert focal_name is not None
            self.assertTrue(
                all(
                    type(day.agent_records[focal_name].action) is int
                    and day.agent_records[focal_name].action in (-1, 0, 1)
                    for day in result.days
                ),
                name,
            )

    def test_stateful_and_seeded_scenarios_are_reproducible(self) -> None:
        stateful = stateful_reset_scenario()
        self.assertEqual(stateful.run(), stateful.run())
        random_scenario = seeded_random_scenario(total_days=18, marked_boundary_day=9)
        self.assertEqual(random_scenario.run(), random_scenario.run())

    def test_pass3_scenario_factory_is_reproducible(self) -> None:
        scenario = validation_scenarios()[0]
        self.assertEqual(
            scenario.run("online_ensemble"),
            scenario.run("online_ensemble"),
        )

    def test_legacy_boundary_zero_starts_at_reset_price(self) -> None:
        config = LiferaftConfig(
            total_days=3,
            marked_boundary_day=0,
            initial_price=123_000,
            reset_price=97_000,
        )
        result = LiferaftSimulator([AlwaysShort("focal")], config).run()
        record = result.days[0].agent_records["focal"]
        self.assertEqual(result.days[0].price, 97_000)
        self.assertTrue(result.days[0].reset_applied)
        self.assertEqual(result.days[0].reset_jump, None)
        self.assertEqual(record.daily_pnl, 0)
        self.assertIsNone(record.pnl_source_day)
        self.assertIsNone(record.pnl_majority)

    def test_future_opponent_changes_do_not_change_earlier_online_actions(self) -> None:
        first_opponent = ScriptedAgent("opponent", [1, 1, 1, 1, 1, 1, 1, 1])
        second_opponent = ScriptedAgent("opponent", [1, 1, 1, 1, 1, 1, -1, -1])
        focal_one = OnlineRollingFrequency(name="focal")
        focal_two = OnlineRollingFrequency(name="focal")
        config = inactive_config(total_days=8)
        first = LiferaftSimulator(
            [focal_one, first_opponent], config, focal_agent_name="focal"
        ).run()
        second = LiferaftSimulator(
            [focal_two, second_opponent], config, focal_agent_name="focal"
        ).run()
        actions_one = [first.days[i].actions["focal"] for i in range(7)]
        actions_two = [second.days[i].actions["focal"] for i in range(7)]
        self.assertEqual(actions_one[:6], actions_two[:6])

    def test_future_perturbations_leave_all_online_candidate_prefixes_unchanged(self) -> None:
        candidate_names = (
            "burnin1_markov",
            "online_rolling",
            "online_markov",
            "online_ensemble",
            "online_drift",
        )
        for candidate_name in candidate_names:
            from research.liferaft.cold_start_strategies import make_cold_start_strategy

            first_focal = make_cold_start_strategy(candidate_name)
            second_focal = make_cold_start_strategy(candidate_name)
            first_opponent = ScriptedAgent("opponent", [1] * 10)
            second_opponent = ScriptedAgent("opponent", [1] * 7 + [-1] * 3)
            first = LiferaftSimulator(
                [first_focal, first_opponent],
                inactive_config(total_days=10),
                focal_agent_name=first_focal.name,
            ).run()
            second = LiferaftSimulator(
                [second_focal, second_opponent],
                inactive_config(total_days=10),
                focal_agent_name=second_focal.name,
            ).run()
            self.assertEqual(
                [first.days[index].actions[first_focal.name] for index in range(7)],
                [second.days[index].actions[second_focal.name] for index in range(7)],
                candidate_name,
            )

    def test_models_are_constructible_with_low_fixed_orders(self) -> None:
        self.assertEqual(OnlineRegularisedMarkov().name, "online-markov")
        self.assertEqual(OnlineRollingFrequency().name, "online-rolling-frequency")

    def test_markov_does_not_bridge_unknown_zero_observations(self) -> None:
        model = RegularisedMarkovModel(order=2, alpha=1.0)
        forecast = model.estimate(
            (MajorityOutcome.LONG, None, MajorityOutcome.SHORT)
        )
        self.assertEqual(forecast.support, 2)
        self.assertAlmostEqual(forecast.p_long, 0.5)
        self.assertAlmostEqual(forecast.p_short, 0.5)


if __name__ == "__main__":
    unittest.main()
