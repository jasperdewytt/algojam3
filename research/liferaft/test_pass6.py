"""Focused non-quarantined tests for Liferaft research Pass 6A."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from research.liferaft.archetypes import AlwaysFlat, AlwaysShort
from research.liferaft.pass3_scenarios import cold_start_config, development_scenarios
from research.liferaft.pass6_models import (
    CTW_MAX_DEPTH,
    EXPERT_NAMES,
    FIXED_SHARE_ETA,
    FIXED_SHARE_RATE,
    BinaryCTW,
    ExpertForecast,
    FixedShareMaster,
    MarkovExpert,
    NEUTRAL_SHORT_PROBABILITY,
    make_experts,
)
from research.liferaft.pass6_strategies import (
    CHECKPOINT_ALPHA,
    CHECKPOINT_BLOCK_SIZE,
    CHECKPOINT_COUNT,
    MIN_RESIDUAL_EDGE,
    AnytimeValidGate,
    FixedCheckpointGate,
    make_pass6_strategy,
    movement_kind,
)
from research.liferaft.simulator import AgentObservation, LiferaftConfig, LiferaftSimulator


def observation(
    day: int,
    *,
    price: int,
    previous_price: int | None = None,
    previous_change: int | None = None,
    own_position: int = 0,
    start: int = 0,
    floor: int = 20_000,
    reset: bool = False,
) -> AgentObservation:
    history = (
        (price,)
        if previous_price is None
        else (previous_price, price)
    )
    return AgentObservation(
        day=day,
        price=price,
        price_history=history,
        previous_price=previous_price,
        previous_price_change=previous_change,
        previous_move_is_reset=reset,
        is_reset_day=reset,
        marked_boundary_day=start,
        price_floor=floor,
        long_majority_move=-5_000,
        short_majority_move=8_000,
        position_limit=1,
        gross_portfolio_budget=600_000,
        own_position=own_position,
        voting_active=day >= start,
        market_mode="inactive_until_marked",
        voting_start_day=start,
    )


def run_synthetic_strategy(strategy, changes: list[int], *, start: int = 0, base: int = 1_000_000):
    price = base
    held = strategy.decide(observation(start, price=price, start=start))
    last_observation = None
    for index, change in enumerate(changes, start=1):
        next_price = price + change
        last_observation = observation(
            start + index,
            price=next_price,
            previous_price=price,
            previous_change=change,
            own_position=held,
            start=start,
        )
        held = strategy.decide(last_observation)
        price = next_price
    return held, last_observation


class Pass6ModelTests(unittest.TestCase):
    def test_exact_expert_set_and_probability_action_bounds(self) -> None:
        experts = make_experts()
        self.assertEqual(tuple(expert.name for expert in experts), EXPERT_NAMES)
        for expert in experts:
            forecast = expert.forecast()
            self.assertGreaterEqual(forecast.p_short, 0.0)
            self.assertLessEqual(forecast.p_short, 1.0)
            self.assertIn(forecast.proposed_position, (-1, 0, 1))
            expert.observe(0)
            forecast = expert.forecast()
            self.assertGreaterEqual(forecast.p_short, 0.0)
            self.assertLessEqual(forecast.p_short, 1.0)
            self.assertIn(forecast.proposed_position, (-1, 0, 1))

    def test_future_perturbation_cannot_change_earlier_forecast(self) -> None:
        prefix = (0, 1, 1, 0)
        for expert in make_experts():
            for token in prefix:
                expert.observe(token)
            earlier = expert.forecast()
            expert.observe(0)
            expert.observe(0)
            self.assertEqual(earlier, earlier)
            replay = type(expert)() if expert.name != "markov_order_1" and expert.name != "markov_order_2" else MarkovExpert(expert.order)
            for token in prefix:
                replay.observe(token)
            self.assertEqual(earlier, replay.forecast())

    def test_order_zero_uses_economic_neutral_threshold(self) -> None:
        neutral = ExpertForecast("neutral", NEUTRAL_SHORT_PROBABILITY, 1, True, "test")
        self.assertEqual(neutral.proposed_position, 0)
        self.assertEqual(NEUTRAL_SHORT_PROBABILITY, 5 / 13)
        self.assertEqual(ExpertForecast("long", 0.50, 1, True, "test").proposed_position, 1)
        self.assertEqual(ExpertForecast("short", 0.20, 1, True, "test").proposed_position, -1)

    def test_fixed_share_update_is_normalised_and_recovers_bad_expert(self) -> None:
        master = FixedShareMaster()
        bad = {name: -8_000 for name in EXPERT_NAMES}
        bad["flat"] = 8_000
        master._update_weights(bad)  # frozen formula is directly unit-tested
        self.assertAlmostEqual(sum(master.weights.values()), 1.0)
        self.assertTrue(all(math.isfinite(weight) and weight >= 0 for weight in master.weights.values()))
        self.assertGreater(master.weights["order_zero_frequency"], 0.0)
        self.assertGreaterEqual(master.weights["order_zero_frequency"], FIXED_SHARE_RATE / len(EXPERT_NAMES) - 1e-12)
        self.assertAlmostEqual(master.eta, FIXED_SHARE_ETA)
        self.assertAlmostEqual(master.share_rate, FIXED_SHARE_RATE)

    def test_forecast_observe_score_update_order_and_flat_reward(self) -> None:
        master = FixedShareMaster()
        first = master.propose()
        score = master.observe_outcome(movement_kind="genuine_nonzero", price_change=8_000)
        self.assertEqual(score.prior_master_position, first.master_position)
        self.assertEqual(score.expert_rewards["flat"], 0)
        self.assertEqual(master.scoreable_count, 1)
        second = master.propose()
        self.assertGreaterEqual(second.forecasts["order_zero_frequency"].support, 1)
        self.assertAlmostEqual(sum(master.weights.values()), 1.0)

    def test_ctw_is_a_suffix_tree_mixture_with_bounded_probabilities(self) -> None:
        ctw = BinaryCTW(CTW_MAX_DEPTH)
        self.assertAlmostEqual(ctw.next_probability(0), 0.5)
        self.assertAlmostEqual(ctw.next_probability(1), 0.5)
        for token in (0, 0, 1, 0, 1, 1, 0):
            ctw.observe(token)
        p0 = ctw.next_probability(0)
        p1 = ctw.next_probability(1)
        self.assertGreaterEqual(p0, 0.0)
        self.assertLessEqual(p0, 1.0)
        self.assertGreaterEqual(p1, 0.0)
        self.assertLessEqual(p1, 1.0)
        self.assertAlmostEqual(p0 + p1, 1.0)
        self.assertNotAlmostEqual(ctw.root_weighted_probability(), ctw.root_kt_probability())

    def test_context_break_does_not_bridge_markov_or_ctw(self) -> None:
        markov = MarkovExpert(2)
        for token in (0, 1, None, 1):
            markov.observe(token)
        self.assertEqual(markov._tail, [1])
        ctw_master = FixedShareMaster()
        ctw_master.observe_outcome(movement_kind="genuine_nonzero", price_change=-5_000)
        ctw_master.propose()
        ctw_master.observe_outcome(movement_kind="zero", price_change=0)
        ctw_forecast = ctw_master.propose().forecasts["context_tree_weighting"]
        self.assertFalse(ctw_forecast.valid)
        self.assertEqual(ctw_forecast.proposed_position, 0)


class Pass6GateAndStrategyTests(unittest.TestCase):
    def test_checkpoint_error_allocation_is_frozen(self) -> None:
        self.assertEqual(CHECKPOINT_COUNT, 18)
        self.assertEqual(CHECKPOINT_BLOCK_SIZE, 20)
        self.assertAlmostEqual(CHECKPOINT_ALPHA, 0.025 / 18)
        gate = FixedCheckpointGate()
        for day in range(CHECKPOINT_BLOCK_SIZE - 1):
            gate.observe(day=day, reward=8_000, scoreable=True)
            self.assertFalse(gate.authorized)
        gate.observe(day=CHECKPOINT_BLOCK_SIZE - 1, reward=8_000, scoreable=True)
        self.assertTrue(gate.authorized)
        self.assertEqual(gate.activation_days, [CHECKPOINT_BLOCK_SIZE - 1])

    def test_anytime_gate_updates_once_and_does_not_reset(self) -> None:
        gate = AnytimeValidGate()
        for day in range(3):
            gate.observe(day=day, reward=8_000, scoreable=True)
        before = gate.log_e_value
        gate.observe(day=3, reward=8_000, scoreable=False)
        self.assertEqual(gate.log_e_value, before)
        self.assertFalse(gate.authorized)
        self.assertGreater(gate.e_value, 1.0)

    def test_qualifying_checkpoint_is_not_traded_retroactively(self) -> None:
        strategy = make_pass6_strategy("fixed_checkpoint_fixed_share")
        _, last = run_synthetic_strategy(strategy, [8_000] * 25)
        self.assertIsNotNone(last)
        timeline = strategy.diagnostics()["timeline"]
        # Day 19 selects the position for outcome 20 and must still be flat.
        # Day 20 observes outcome 20, so it is the first possible decision
        # for the subsequent block.
        self.assertEqual(timeline[19]["action"], 0)
        self.assertIsNone(timeline[19]["gate_state"]["first_authorization_day"])
        self.assertEqual(timeline[20]["gate_state"]["first_authorization_day"], 20)
        self.assertIn(timeline[20]["action"], (-1, 0, 1))

    def test_duplicate_day_is_idempotent_and_exposure_provider_is_cached(self) -> None:
        calls: list[int] = []

        def exposure(obs: AgentObservation) -> float:
            calls.append(obs.day)
            return 0.0

        strategy = make_pass6_strategy("ungated_fixed_share", other_portfolio_exposure=exposure)
        held, last = run_synthetic_strategy(strategy, [8_000] * 30)
        self.assertIsNotNone(last)
        assert last is not None
        before = dict(strategy.master.expert_cumulative_rewards)
        returned = strategy.decide(last)
        self.assertEqual(returned, held)
        self.assertEqual(before, strategy.master.expert_cumulative_rewards)
        self.assertEqual(len(calls), strategy.exposure_evaluation_count)
        self.assertLessEqual(len(calls), len(strategy.diagnostics()["timeline"]))
        self.assertGreaterEqual(strategy.exposure_evaluation_count, 0)

    def test_floor_clipping_and_reset_are_distinct(self) -> None:
        clipped = observation(
            2,
            price=20_000,
            previous_price=22_000,
            previous_change=-2_000,
            start=0,
        )
        exact = observation(
            2,
            price=20_000,
            previous_price=25_000,
            previous_change=-5_000,
            start=0,
        )
        reset = observation(
            0,
            price=100_000,
            previous_price=105_000,
            previous_change=-5_000,
            start=0,
            reset=True,
        )
        self.assertEqual(movement_kind(clipped), "floor_clipped")
        self.assertEqual(movement_kind(exact), "genuine_nonzero")
        self.assertEqual(movement_kind(reset), "inactive_or_startup")

    def test_stops_and_overshoot_use_actual_public_change(self) -> None:
        strategy = make_pass6_strategy("ungated_fixed_share")
        loss_obs = observation(1, price=950_000, previous_price=1_000_000, previous_change=-50_000, own_position=1, start=0)
        strategy._account_actual_pnl(loss_obs, kind="genuine_nonzero")
        self.assertTrue(strategy.loss_stop_active)
        self.assertGreaterEqual(strategy.diagnostics()["stop_events"][0]["overshoot"], 0)

        drawdown = make_pass6_strategy("ungated_fixed_share")
        gain = observation(1, price=1_010_000, previous_price=1_000_000, previous_change=10_000, own_position=1, start=0)
        loss = observation(2, price=950_000, previous_price=1_010_000, previous_change=-60_000, own_position=1, start=0)
        drawdown._account_actual_pnl(gain, kind="genuine_nonzero")
        drawdown._account_actual_pnl(loss, kind="genuine_nonzero")
        self.assertTrue(drawdown.drawdown_stop_active)

    def test_flat_before_live_in_both_execution_modes_and_safe_budget(self) -> None:
        for mode in ("observe_and_ignore_actions", "fully_inactive"):
            config = cold_start_config(total_days=12, marked_boundary_day=4, execution_mode=mode)
            focal = make_pass6_strategy("anytime_valid_fixed_share")
            result = LiferaftSimulator(
                (focal, *tuple(AlwaysFlat(f"flat-{index}") for index in range(5))),
                config,
                focal_agent_name=focal.name,
                other_portfolio_exposure={focal.name: 0},
            ).run()
            for day in result.days[:4]:
                self.assertEqual(day.agent_records[focal.name].action, 0)
            self.assertEqual(result.marked_pnl[focal.name], 0)
            self.assertEqual(len(result.budget_breaches), 0)
            self.assertEqual(len(result.rejected_actions), 0)

    def test_hidden_pivotal_information_is_not_an_input(self) -> None:
        source = Path(__file__).with_name("pass6_strategies.py").read_text(encoding="utf-8")
        self.assertNotIn("focal_pivotal", source)
        self.assertNotIn("net_margin", source)
        self.assertNotIn("long_count", source)


if __name__ == "__main__":
    unittest.main()
