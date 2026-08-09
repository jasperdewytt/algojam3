"""Bounded Pass 3.1 correction and endogenous path-audit tests."""

from __future__ import annotations

import unittest

from research.liferaft.pass3_experiments import (
    LivePathDifference,
    live_path_difference,
    portfolio_path_divergence_audit,
)
from research.liferaft.pass3_scenarios import validation_scenarios
from research.liferaft.simulator import LiferaftConfig, LiferaftSimulator


class ScriptedAgent:
    def __init__(self, name: str, action: int) -> None:
        self.name = name
        self.action = action

    def decide(self, observation) -> int:
        del observation
        return self.action


class TestPass31PathAudit(unittest.TestCase):
    def test_zero_exposure_compared_with_itself_has_no_divergence(self) -> None:
        scenario = validation_scenarios()[0]
        result = scenario.run("burnin1_markov", other_portfolio_exposure=0)
        self.assertEqual(
            live_path_difference(result, result),
            LivePathDifference(0, 0, 0),
        )
        summaries = portfolio_path_divergence_audit(
            (scenario,), exposures=(0,), progress=False
        )
        overall = next(summary for summary in summaries if summary.group == "overall")
        self.assertEqual(overall.cases, 1)
        self.assertEqual(overall.different_price_path_fraction, 0.0)
        self.assertEqual(overall.different_majority_path_fraction, 0.0)
        self.assertEqual(overall.mean_differing_opponent_action_cells, 0.0)

    def test_pivotal_budget_flattening_changes_majority_and_price(self) -> None:
        config = LiferaftConfig(
            total_days=4,
            marked_boundary_day=1,
            initial_price=100_000,
            reset_price=100_000,
            market_mode="inactive_until_marked",
            pre_voting_price=100_000,
            voting_start_day=1,
            gross_portfolio_budget=100_000,
        )
        baseline = LiferaftSimulator(
            [ScriptedAgent("focal", 1), ScriptedAgent("opponent", -1)],
            config,
            focal_agent_name="focal",
            other_portfolio_exposure={"focal": 0},
        ).run()
        constrained = LiferaftSimulator(
            [ScriptedAgent("focal", 1), ScriptedAgent("opponent", -1)],
            config,
            focal_agent_name="focal",
            other_portfolio_exposure={"focal": 1},
        ).run()

        self.assertNotEqual(baseline.days[1].majority, constrained.days[1].majority)
        self.assertNotEqual(baseline.price_path[2:], constrained.price_path[2:])
        difference = live_path_difference(baseline, constrained)
        self.assertGreater(difference.differing_majority_days, 0)
        self.assertGreater(difference.differing_price_days, 0)

    def test_opponent_effective_actions_can_change_with_identical_raw_requests(self) -> None:
        config = LiferaftConfig(
            total_days=4,
            marked_boundary_day=1,
            initial_price=100_000,
            reset_price=100_000,
            market_mode="inactive_until_marked",
            pre_voting_price=100_000,
            voting_start_day=1,
            gross_portfolio_budget=100_000,
        )
        baseline = LiferaftSimulator(
            [ScriptedAgent("focal", 1), ScriptedAgent("opponent", -1)],
            config,
            focal_agent_name="focal",
            other_portfolio_exposure={"focal": 0, "opponent": 0},
        ).run()
        constrained = LiferaftSimulator(
            [ScriptedAgent("focal", 1), ScriptedAgent("opponent", -1)],
            config,
            focal_agent_name="focal",
            other_portfolio_exposure={"focal": 1, "opponent": 0},
        ).run()

        self.assertEqual(
            [day.requested_actions["opponent"] for day in baseline.days],
            [day.requested_actions["opponent"] for day in constrained.days],
        )
        self.assertNotEqual(
            [day.actions["opponent"] for day in baseline.days],
            [day.actions["opponent"] for day in constrained.days],
        )
        self.assertGreater(
            live_path_difference(baseline, constrained).differing_opponent_action_cells,
            0,
        )

    def test_path_audit_is_deterministic(self) -> None:
        scenarios = (validation_scenarios()[0], validation_scenarios()[1])
        first = portfolio_path_divergence_audit(
            scenarios,
            exposures=(0, 150_000, 450_000),
            progress=False,
        )
        second = portfolio_path_divergence_audit(
            scenarios,
            exposures=(0, 150_000, 450_000),
            progress=False,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
