"""Causality, calibration, and Pass 2 strategy tests."""

from __future__ import annotations

import unittest

from research.liferaft.archetypes import AlwaysFlat
from research.liferaft.calibration import select_candidate_from_year1
from research.liferaft.pass2_scenarios import (
    development_scenarios,
    strategy_factory_for_name,
    validation_scenarios,
)
from research.liferaft.pass2_experiments import STRATEGY_NAMES, run_suite
from research.liferaft.simulator import (
    AgentObservation,
    LiferaftConfig,
    LiferaftSimulator,
    MajorityOutcome,
)
from research.liferaft.strategies import (
    DriftAwareStrategy,
    Forecast,
    PeriodicReplayStrategy,
    RegularisedMarkovModel,
    SmallExpertEnsembleStrategy,
    Year1CalibratedStrategy,
    asymmetric_short_majority_threshold,
    choose_replay_period,
    labels_from_prices,
    payoff_action,
)


def public_observation(
    prices: list[int],
    day: int,
    boundary: int,
    *,
    own_position: int = 0,
) -> AgentObservation:
    previous_price = prices[day - 1] if day else None
    return AgentObservation(
        day=day,
        price=prices[day],
        price_history=tuple(prices[: day + 1]),
        previous_price=previous_price,
        previous_price_change=(
            None if previous_price is None else prices[day] - previous_price
        ),
        previous_move_is_reset=day == boundary,
        is_reset_day=day == boundary,
        marked_boundary_day=boundary,
        price_floor=20_000,
        long_majority_move=-5_000,
        short_majority_move=8_000,
        position_limit=1,
        gross_portfolio_budget=600_000,
        own_position=own_position,
    )


def collect_actions(strategy, prices: list[int], boundary: int, through: int) -> list[int]:
    actions: list[int] = []
    position = 0
    for day in range(through + 1):
        action = strategy.decide(
            public_observation(prices, day, boundary, own_position=position)
        )
        actions.append(action)
        position = action if isinstance(action, int) else 0
    return actions


class Pass2CausalityTests(unittest.TestCase):
    def setUp(self) -> None:
        # The boundary jump is deliberately different from both canonical
        # market moves; tests below also cover canonical-sized resets directly.
        self.base_prices = [
            100_000,
            95_000,
            103_000,
            98_000,
            106_000,
            101_000,
            100_000,
            108_000,
            103_000,
            111_000,
            106_000,
            114_000,
            109_000,
            117_000,
            112_000,
            120_000,
        ]
        self.boundary = 6

    def test_asymmetric_threshold_and_no_trade_margin(self) -> None:
        self.assertAlmostEqual(
            asymmetric_short_majority_threshold(),
            5 / 13,
        )
        self.assertEqual(
            payoff_action(
                Forecast(0.5, 0.5, support=10),
                min_confidence=0.10,
            ),
            0,
        )

    def test_public_labels_skip_reset_but_keep_zero_ambiguous(self) -> None:
        self.assertEqual(
            labels_from_prices(
                [100_000, 95_000, 100_000, 100_000],
                marked_boundary_day=2,
            ),
            (MajorityOutcome.LONG, None),
        )
        self.assertIsNone(
            public_observation(
                [100_000, 95_000, 100_000],
                2,
                2,
            ).previous_inferred_majority
        )

    def test_markov_does_not_bridge_an_unknown_zero_observation(self) -> None:
        forecast = RegularisedMarkovModel(order=1).estimate(
            (MajorityOutcome.LONG, None, MajorityOutcome.SHORT)
        )
        self.assertAlmostEqual(forecast.p_long, 0.5)
        self.assertAlmostEqual(forecast.p_short, 0.5)

    def test_replay_period_requires_evidence_against_constant_null(self) -> None:
        alternating = tuple(
            MajorityOutcome.LONG if index % 2 == 0 else MajorityOutcome.SHORT
            for index in range(12)
        )
        self.assertEqual(
            choose_replay_period(
                alternating,
                periods=(2, 3, 4),
                min_observations=4,
                minimum_margin=0.10,
            ),
            2,
        )

    def test_future_price_and_opponent_perturbations_cannot_change_prefix(self) -> None:
        perturbed = list(self.base_prices)
        for index in range(10, len(perturbed)):
            perturbed[index] += 2_000 * (index - 9)
        factories = (
            lambda: Year1CalibratedStrategy(
                "focal", warmup_days=1, min_improvement=0
            ),
            lambda: SmallExpertEnsembleStrategy("focal", warmup_days=1),
            lambda: DriftAwareStrategy(
                "focal",
                warmup_days=1,
                minimum_quality_observations=2,
                degradation_streak=2,
            ),
            lambda: PeriodicReplayStrategy(
                "focal",
                marked_boundary_day=self.boundary,
                min_observations=2,
            ),
        )
        for factory in factories:
            base_actions = collect_actions(factory(), self.base_prices, self.boundary, 9)
            perturbed_actions = collect_actions(factory(), perturbed, self.boundary, 9)
            self.assertEqual(base_actions, perturbed_actions)

    def test_year1_selection_is_invariant_to_held_out_suffix(self) -> None:
        suffix_changed = list(self.base_prices)
        suffix_changed[8:] = [150_000 + 7_000 * index for index in range(8)]
        config = LiferaftConfig(
            total_days=len(self.base_prices),
            marked_boundary_day=self.boundary,
        )
        first = select_candidate_from_year1(
            self.base_prices,
            boundary_day=self.boundary,
            config=config,
            warmup_days=1,
            minimum_improvement=0,
        )
        second = select_candidate_from_year1(
            suffix_changed,
            boundary_day=self.boundary,
            config=config,
            warmup_days=1,
            minimum_improvement=0,
        )
        self.assertEqual(first, second)

    def test_online_quality_updates_only_after_a_genuine_movement(self) -> None:
        # A long-majority calibration prefix gives the counter a clear frozen
        # Year-1 choice.  Day 20 is a reset; day 21 is the first observed
        # genuine marked-period outcome.
        prices = [200_000 - 5_000 * index for index in range(20)]
        prices += [100_000, 108_000, 103_000, 111_000]
        strategy = DriftAwareStrategy(
            "focal",
            candidate_names=("last_majority_counter",),
            warmup_days=1,
            minimum_quality_observations=1,
            degradation_streak=1,
        )
        strategy.decide(public_observation(prices, 20, 20))
        self.assertEqual(strategy.quality_observations, 0)
        strategy.decide(public_observation(prices, 21, 20))
        self.assertEqual(strategy.quality_observations, 1)

    def test_hidden_engine_fields_are_not_in_public_observation(self) -> None:
        fields = set(AgentObservation.__dataclass_fields__)
        self.assertNotIn("long_count", fields)
        self.assertNotIn("short_count", fields)
        self.assertNotIn("majority", fields)
        self.assertNotIn("pivotal", fields)

    def test_boundary_zero_strategy_selection_has_no_year1_data(self) -> None:
        config = LiferaftConfig(
            total_days=2,
            marked_boundary_day=0,
            initial_price=90_000,
            reset_price=110_000,
        )
        strategy = Year1CalibratedStrategy("focal", warmup_days=1)
        result = LiferaftSimulator(
            (strategy,),
            config,
            focal_agent_name="focal",
        ).run()
        strategy_record = result.days[0].agent_records["focal"]
        self.assertEqual(result.price_path[0], 110_000)
        self.assertEqual(strategy_record.daily_pnl, 0)
        self.assertEqual(strategy.selected_name, "flat")


class Pass2ScenarioTests(unittest.TestCase):
    def test_seeded_pass2_scenario_is_reproducible(self) -> None:
        scenario = validation_scenarios()[0]
        first = scenario.run(strategy_factory_for_name("ensemble"))
        second = scenario.run(strategy_factory_for_name("ensemble"))
        self.assertEqual(first, second)
        self.assertEqual(first.random_seeds, second.random_seeds)

    def test_all_declared_strategies_keep_actions_integral_and_bounded(self) -> None:
        scenario = development_scenarios()[0]
        for name in STRATEGY_NAMES:
            result = scenario.run(strategy_factory_for_name(name))
            for day in result.days:
                action = day.actions["focal"]
                self.assertIs(type(action), int, name)
                self.assertLessEqual(abs(action), 1, name)

    def test_experiment_reports_pivotal_and_budget_dimensions(self) -> None:
        metrics = run_suite(
            "development",
            development_scenarios()[:1],
            strategy_names=("flat", "ensemble", "drift_aware"),
        )
        self.assertEqual(len(metrics), 3)
        self.assertTrue(all(metric.non_pivotal_days >= 0 for metric in metrics))
        self.assertTrue(all(metric.pivotal_days >= 0 for metric in metrics))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
