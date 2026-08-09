"""Causality and timing tests for the quarantined public-cycle experiment."""

from __future__ import annotations

from dataclasses import replace
import json
import unittest
from pathlib import Path

from research.liferaft.cold_start_strategies import COLD_START_STRATEGY_NAMES
from research.liferaft.cycle_scenarios import development_cycle_scenarios
from research.liferaft.cycle_experiments import metric_for_run, score_cycle_forecasts
from research.liferaft.cycle_strategies import PublicCycleDetector
from research.liferaft.simulator import AgentObservation, MajorityOutcome


def observation_for(
    day: int,
    label: MajorityOutcome | None,
    *,
    boundary: int = 0,
    previous_price: int = 100_000,
    floor: int = 20_000,
) -> AgentObservation:
    if label is MajorityOutcome.LONG:
        change = -5_000
    elif label is MajorityOutcome.SHORT:
        change = 8_000
    elif label is None:
        change = 0
    else:
        raise ValueError("test helper expects a label or None")
    price = previous_price + change
    return AgentObservation(
        day=day,
        price=price,
        price_history=(previous_price, price),
        previous_price=previous_price,
        previous_price_change=change,
        previous_move_is_reset=False,
        is_reset_day=False,
        marked_boundary_day=boundary,
        price_floor=floor,
        long_majority_move=-5_000,
        short_majority_move=8_000,
        position_limit=1,
        gross_portfolio_budget=600_000,
        own_position=0,
        voting_active=True,
        market_mode="inactive_until_marked",
        voting_start_day=boundary,
    )


def clipped_observation(day: int) -> AgentObservation:
    return AgentObservation(
        day=day,
        price=20_000,
        price_history=(22_000, 20_000),
        previous_price=22_000,
        previous_price_change=-2_000,
        previous_move_is_reset=False,
        is_reset_day=False,
        marked_boundary_day=0,
        price_floor=20_000,
        long_majority_move=-5_000,
        short_majority_move=8_000,
        position_limit=1,
        gross_portfolio_budget=600_000,
        own_position=0,
        voting_active=True,
        market_mode="inactive_until_marked",
        voting_start_day=0,
    )


def reset_observation(day: int, *, boundary: int = 0) -> AgentObservation:
    return AgentObservation(
        day=day,
        price=95_000,
        price_history=(100_000, 95_000),
        previous_price=100_000,
        previous_price_change=-5_000,
        previous_move_is_reset=True,
        is_reset_day=True,
        marked_boundary_day=boundary,
        price_floor=20_000,
        long_majority_move=-5_000,
        short_majority_move=8_000,
        position_limit=1,
        gross_portfolio_budget=600_000,
        own_position=0,
        voting_active=True,
        market_mode="continuous_reset",
        voting_start_day=boundary,
    )


def feed(
    detector: PublicCycleDetector,
    labels: tuple[MajorityOutcome | None, ...],
    *,
    start_day: int = 1,
) -> list[int]:
    return [
        detector.decide(observation_for(start_day + index, label))
        for index, label in enumerate(labels)
    ]


class TestPublicCycleDetector(unittest.TestCase):
    def test_no_action_before_live_voting_and_insufficient_history_stays_flat(self) -> None:
        detector = PublicCycleDetector()
        self.assertEqual(detector.decide(observation_for(0, None)), 0)
        actions = feed(
            detector,
            (
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
            ),
        )
        self.assertEqual(actions, [0, 0, 0, 0, 0])
        self.assertFalse(detector.cycle_active)

    def test_three_complete_blocks_activate_on_the_confirming_observation(self) -> None:
        detector = PublicCycleDetector()
        actions = feed(
            detector,
            (
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
            ),
        )
        self.assertEqual(actions[:5], [0, 0, 0, 0, 0])
        self.assertEqual(actions[5], -1)  # next predicted majority is LONG
        self.assertEqual(detector.detected_period, 2)
        self.assertEqual(detector.detections[0].day, 6)
        self.assertEqual(detector.detection_delays, (6,))

    def test_causal_accuracy_is_perfect_on_non_pivotal_pure_cycles(self) -> None:
        scenarios = development_cycle_scenarios()
        for name in ("pure-period-2", "pure-period-3", "pure-period-20"):
            scenario = next(item for item in scenarios if item.name == name)
            result, detector = scenario.run("cycle_detector")
            metric = metric_for_run(scenario, "cycle_detector", result, detector)
            self.assertGreater(metric.scoreable_forecasts, 0, name)
            self.assertEqual(metric.forecast_hits, metric.scoreable_forecasts, name)
            self.assertEqual(metric.forecast_accuracy_after_activation, 1.0, name)

    def test_accuracy_uses_realised_movement_not_next_decision_majority(self) -> None:
        scenario = next(
            item
            for item in development_cycle_scenarios()
            if item.name == "pure-period-3"
        )
        result, detector = scenario.run("cycle_detector")
        baseline = score_cycle_forecasts(result, detector)
        decision_day = detector.detections[0].day
        next_day = result.days[decision_day + 1]
        changed_majority = replace(
            next_day,
            majority=(
                MajorityOutcome.SHORT
                if next_day.majority is MajorityOutcome.LONG
                else MajorityOutcome.LONG
            ),
        )
        changed_result = replace(
            result,
            days=result.days[: decision_day + 1]
            + (changed_majority,)
            + result.days[decision_day + 2 :],
        )
        self.assertEqual(score_cycle_forecasts(changed_result, detector), baseline)

    def test_reset_zero_and_clipped_movements_are_not_scoreable(self) -> None:
        scenario = next(
            item
            for item in development_cycle_scenarios()
            if item.name == "pure-period-2"
        )
        result, detector = scenario.run("cycle_detector")
        baseline = score_cycle_forecasts(result, detector)
        decision_day = detector.detections[0].day
        movement_index = decision_day + 1

        reset_day = replace(result.days[movement_index], reset_applied=True)
        zero_day = replace(result.days[movement_index], price_change=0)
        clipped_source = replace(result.days[decision_day], floor_clipped=True)

        reset_result = replace(
            result,
            days=result.days[:movement_index] + (reset_day,) + result.days[movement_index + 1 :],
        )
        zero_result = replace(
            result,
            days=result.days[:movement_index] + (zero_day,) + result.days[movement_index + 1 :],
        )
        clipped_result = replace(
            result,
            days=result.days[:decision_day] + (clipped_source,) + result.days[decision_day + 1 :],
        )
        for changed in (reset_result, zero_result, clipped_result):
            score = score_cycle_forecasts(changed, detector)
            self.assertEqual(score.scoreable, baseline.scoreable - 1)

    def test_pivotal_tie_is_unscoreable_not_an_automatic_miss(self) -> None:
        scenario = next(
            item
            for item in development_cycle_scenarios()
            if item.name == "pivotal-focal-cycle"
        )
        result, detector = scenario.run("cycle_detector")
        baseline = score_cycle_forecasts(result, detector)
        decision_day = detector.detections[0].day
        movement_index = decision_day + 1
        source_day = result.days[decision_day]
        movement_day = result.days[movement_index]
        self.assertTrue(source_day.focal_pivotal)
        self.assertEqual(source_day.majority, MajorityOutcome.TIE)
        self.assertEqual(movement_day.price_change, 0)
        self.assertEqual(baseline.scoreable, 0)

        # The pivotal focal vote converts the underlying one-vote majority to
        # a tie.  Its zero movement is therefore unscoreable, not a forecast
        # miss, and changing the next decision's hidden label cannot alter it.
        changed_next_decision = replace(
            result.days[movement_index],
            majority=MajorityOutcome.LONG,
        )
        changed_result = replace(
            result,
            days=result.days[:movement_index]
            + (changed_next_decision,)
            + result.days[movement_index + 1 :],
        )
        changed_score = score_cycle_forecasts(changed_result, detector)
        self.assertEqual(changed_score.scoreable, baseline.scoreable)
        self.assertEqual(changed_score.hits, baseline.hits)

    def test_reset_while_inactive_is_processed_once_and_stays_flat(self) -> None:
        detector = PublicCycleDetector()
        self.assertEqual(detector.decide(reset_observation(1)), 0)
        self.assertEqual(detector.decide(reset_observation(1)), 0)
        self.assertFalse(detector.cycle_active)
        self.assertEqual(detector.labels, ())
        self.assertEqual(detector.observed_live_count, 0)
        self.assertEqual(detector.cycle_breaks, 0)

    def test_reset_while_active_clears_state_and_allows_clean_reactivation(self) -> None:
        detector = PublicCycleDetector()
        feed(
            detector,
            (
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
            ),
        )
        self.assertTrue(detector.cycle_active)
        self.assertEqual(detector.decide(reset_observation(7, boundary=7)), 0)
        self.assertEqual(detector.decide(reset_observation(7, boundary=7)), 0)
        self.assertFalse(detector.cycle_active)
        self.assertEqual(detector.labels, ())
        self.assertEqual(detector.cycle_breaks, 1)
        self.assertEqual(detector.observed_live_count, 6)

        actions = feed(
            detector,
            (
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
            ),
            start_day=8,
        )
        self.assertEqual(actions[:5], [0, 0, 0, 0, 0])
        self.assertEqual(actions[5], -1)
        self.assertEqual(detector.detection_count, 2)
        self.assertEqual(detector.reactivation_count, 1)

    def test_unknown_and_floor_clipped_observations_break_cycle_context(self) -> None:
        detector = PublicCycleDetector()
        feed(
            detector,
            (
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
            ),
        )
        self.assertTrue(detector.cycle_active)
        self.assertEqual(detector.decide(observation_for(7, None)), 0)
        self.assertFalse(detector.cycle_active)
        self.assertEqual(detector.labels, ())

        detector = PublicCycleDetector()
        feed(
            detector,
            (
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
            ),
        )
        self.assertEqual(detector.decide(clipped_observation(7)), 0)
        self.assertFalse(detector.cycle_active)

    def test_contradiction_deactivates_then_detector_can_reactivate(self) -> None:
        detector = PublicCycleDetector()
        feed(
            detector,
            (
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
            ),
        )
        # The active forecast is LONG, so a SHORT outcome is contradictory.
        self.assertEqual(detector.decide(observation_for(7, MajorityOutcome.SHORT)), 0)
        self.assertFalse(detector.cycle_active)
        self.assertEqual(detector.cycle_breaks, 1)

        actions = feed(
            detector,
            (
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
            ),
            start_day=8,
        )
        self.assertEqual(actions[-1], 1)  # context is S,L,S,L,S,L; next is SHORT
        self.assertEqual(detector.detection_count, 2)
        self.assertEqual(detector.reactivation_count, 1)

    def test_shortest_qualifying_period_is_selected(self) -> None:
        detector = PublicCycleDetector()
        feed(
            detector,
            (
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
                MajorityOutcome.SHORT,
                MajorityOutcome.LONG,
            ),
        )
        self.assertEqual(detector.detected_period, 2)

    def test_future_labels_do_not_change_prior_actions_and_actions_are_integral(self) -> None:
        first = PublicCycleDetector()
        second = PublicCycleDetector()
        prefix = (
            MajorityOutcome.LONG,
            MajorityOutcome.SHORT,
            MajorityOutcome.LONG,
            MajorityOutcome.SHORT,
            MajorityOutcome.LONG,
            MajorityOutcome.SHORT,
        )
        first_actions = feed(first, prefix)
        second_actions = feed(second, prefix)
        self.assertEqual(first_actions, second_actions)
        first_future = feed(
            first,
            (MajorityOutcome.LONG, MajorityOutcome.SHORT, MajorityOutcome.LONG),
            start_day=7,
        )
        second_future = feed(
            second,
            (MajorityOutcome.SHORT, MajorityOutcome.SHORT, MajorityOutcome.SHORT),
            start_day=7,
        )
        self.assertNotEqual(first_future, second_future)
        self.assertTrue(all(type(action) is int and action in (-1, 0, 1) for action in first_actions + first_future))

    def test_observation_contains_no_hidden_simulator_fields(self) -> None:
        observation = observation_for(1, MajorityOutcome.LONG)
        self.assertFalse(hasattr(observation, "long_count"))
        self.assertFalse(hasattr(observation, "focal_pivotal"))
        detector = PublicCycleDetector()
        self.assertIn(detector.decide(observation), (-1, 0, 1))

    def test_development_scenario_runs_are_fresh_and_deterministic(self) -> None:
        scenario = development_cycle_scenarios()[5]
        first, first_agent = scenario.run("cycle_detector")
        second, second_agent = scenario.run("cycle_detector")
        self.assertEqual(first, second)
        self.assertEqual(first_agent.detections, second_agent.detections)
        self.assertEqual(first_agent.forecast_by_day, second_agent.forecast_by_day)

    def test_cycle_detector_is_quarantined_from_pass3_and_locked_catalogue(self) -> None:
        self.assertNotIn("cycle_detector", COLD_START_STRATEGY_NAMES)
        manifest = Path(__file__).with_name("PASS3_FINAL_MANIFEST.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("cycle_strategies.py", manifest)
        final_report = Path(__file__).with_name("PASS3_FINAL_REPORT.md")
        receipt = Path(__file__).with_name("PASS4_FINAL_EXECUTION_RECEIPT.json")
        if final_report.exists():
            self.assertTrue(receipt.exists())
            self.assertEqual(
                json.loads(receipt.read_text(encoding="utf-8"))["status"],
                "consumed",
            )
        else:
            self.assertFalse(receipt.exists())

    def test_thirteen_movement_return_fixture_has_eight_long_and_five_short(self) -> None:
        scenario = next(
            item
            for item in development_cycle_scenarios()
            if item.name == "return-cycle-13-movements"
        )
        opponents = scenario.population_factory()
        actions = [
            opponent.schedule[scenario.config.voting_start_day + offset]
            for offset in range(13)
            for opponent in opponents[:1]
        ]
        self.assertEqual(actions.count(1), 8)
        self.assertEqual(actions.count(-1), 5)
        result, _agent = scenario.run("flat")
        start = scenario.config.voting_start_day
        assert start is not None
        self.assertEqual(result.price_path[start + 13], result.price_path[start])


if __name__ == "__main__":
    unittest.main()
