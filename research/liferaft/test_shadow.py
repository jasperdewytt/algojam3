"""Mechanics and causal-state tests for Liferaft Pass 5A."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from research.liferaft.cold_start_strategies import COLD_START_STRATEGY_NAMES
from research.liferaft.pass3_scenarios import (
    development_scenarios,
    validation_scenarios,
)
from research.liferaft.simulator import (
    AgentObservation,
    LiferaftConfig,
    LiferaftSimulator,
)
from research.liferaft.strategies import Forecast
from research.liferaft.shadow_strategies import (
    GROSS_PORTFOLIO_BUDGET,
    PORTFOLIO_RESERVE_AUD,
    SHADOW_PARAMETERS,
    SHADOW_STRATEGY_NAMES,
    ShadowValidatedMarkov,
    make_shadow_strategy,
)


class ForecastSequenceModel:
    """Causal test model whose forecast is selected by the observed day."""

    name = "test-sequence"

    def __init__(self, default: Forecast, by_day: dict[int, Forecast] | None = None):
        self.default = default
        self.by_day = dict(by_day or {})
        self.calls: list[int] = []

    def estimate(self, labels):
        # The real strategy never passes time to a model.  Tests select by the
        # number of labels so the model remains a public-history-only double.
        key = len(labels)
        self.calls.append(key)
        return self.by_day.get(key, self.default)


def make_observation(
    day: int,
    *,
    previous_price: int | None = None,
    price: int = 100_000,
    change: int | None = None,
    own_position: int = 0,
    boundary: int = 0,
    reset: bool = False,
    floor: int = 20_000,
    market_mode: str = "inactive_until_marked",
    voting_active: bool | None = None,
    history: tuple[int, ...] | None = None,
) -> AgentObservation:
    if previous_price is None and change is not None:
        previous_price = price - change
    if history is None:
        history = (price,) if previous_price is None else (previous_price, price)
    return AgentObservation(
        day=day,
        price=price,
        price_history=history,
        previous_price=previous_price,
        previous_price_change=change,
        previous_move_is_reset=reset,
        is_reset_day=reset,
        marked_boundary_day=boundary,
        price_floor=floor,
        long_majority_move=-5_000,
        short_majority_move=8_000,
        position_limit=1,
        gross_portfolio_budget=600_000,
        own_position=own_position,
        voting_active=(
            day >= boundary if voting_active is None else voting_active
        ),
        market_mode=market_mode,
        voting_start_day=boundary,
    )


def positive_forecast() -> Forecast:
    # Predict a short majority, so payoff_action takes a long position and
    # earns the public positive movement.
    return Forecast(0.20, 0.80, 8, "test-positive-edge")


def weak_forecast() -> Forecast:
    return Forecast(0.50, 0.50, 8, "test-weak-edge")


def drive_shadow_to_activation(
    strategy: ShadowValidatedMarkov,
    *,
    days: int = 30,
    change: int = 8_000,
) -> list[int]:
    """Feed a deterministic positive public sequence and return real actions."""

    actions: list[int] = []
    price = 100_000
    prior_action = 0
    for day in range(days):
        if day == 0:
            current_change = None
            previous_price = None
        else:
            current_change = change
            previous_price = price
            price += change
        action = strategy.decide(
            make_observation(
                day,
                previous_price=previous_price,
                price=price,
                change=current_change,
                own_position=prior_action,
            )
        )
        actions.append(action)
        prior_action = action
    return actions


class ShadowStrategyMechanicsTests(unittest.TestCase):
    def test_no_lookahead_future_price_perturbation_does_not_change_prefix(self) -> None:
        first = make_shadow_strategy("shadow12_markov")
        second = make_shadow_strategy("shadow12_markov")
        price_first = 100_000
        price_second = 100_000
        prior_first = 0
        prior_second = 0
        actions_first: list[int] = []
        actions_second: list[int] = []
        for day in range(18):
            if day == 0:
                change_first = change_second = None
                previous_first = previous_second = None
            else:
                change_first = 8_000 if day % 3 else -5_000
                change_second = change_first
                previous_first = price_first
                previous_second = price_second
                price_first += change_first
                price_second += change_second
            actions_first.append(
                first.decide(
                    make_observation(
                        day,
                        previous_price=previous_first,
                        price=price_first,
                        change=change_first,
                        own_position=prior_first,
                    )
                )
            )
            actions_second.append(
                second.decide(
                    make_observation(
                        day,
                        previous_price=previous_second,
                        price=price_second,
                        change=change_second,
                        own_position=prior_second,
                        # This history is deliberately different in the
                        # suffix; the strategy must not read it as future data.
                        history=(100_000, price_second, 999_999_999),
                    )
                )
            )
            prior_first = actions_first[-1]
            prior_second = actions_second[-1]
        self.assertEqual(actions_first, actions_second)

    def test_one_day_virtual_pnl_alignment_and_causal_order(self) -> None:
        strategy = make_shadow_strategy("shadow8_markov")
        strategy._model = ForecastSequenceModel(positive_forecast())
        strategy.decide(make_observation(0))
        action_day_one = strategy.decide(
            make_observation(1, previous_price=100_000, price=108_000, change=8_000)
        )
        self.assertEqual(action_day_one, 0)
        record_day_one = strategy.timeline[-1]
        self.assertEqual(record_day_one.prior_virtual_action, 1)
        self.assertEqual(record_day_one.raw_virtual_interval_pnl, 8_000)
        self.assertEqual(record_day_one.virtual_interval_pnl, 8_000)
        self.assertTrue(record_day_one.virtual_trade_scoreable)
        self.assertEqual(strategy.cumulative_virtual_pnl, 8_000)

    def test_initial_real_position_stays_flat_and_qualification_is_not_retroactive(self) -> None:
        strategy = make_shadow_strategy("shadow8_markov")
        strategy._model = ForecastSequenceModel(positive_forecast())
        actions = drive_shadow_to_activation(strategy, days=16)
        self.assertEqual(actions[0], 0)
        activation_day = strategy.activation_day
        self.assertIsNotNone(activation_day)
        assert activation_day is not None
        self.assertTrue(all(action == 0 for action in actions[:activation_day]))
        self.assertEqual(actions[activation_day], 1)
        self.assertEqual(strategy.activation_events[0]["day"], activation_day)
        # The second qualifying evaluation is the prior decision; no real
        # P&L can include its movement because the position was still flat.
        qualifying = [record for record in strategy.timeline if record.activation_pending]
        self.assertTrue(qualifying)
        self.assertEqual(qualifying[-1].real_action, 0)

    def test_two_new_evaluations_activate_but_duplicate_calls_do_not(self) -> None:
        strategy = make_shadow_strategy("shadow8_markov")
        strategy._model = ForecastSequenceModel(positive_forecast())
        drive_shadow_to_activation(strategy, days=14)
        activation_events_before = strategy.activation_events
        last = strategy.timeline[-1]
        duplicate_action = strategy.decide(
            make_observation(
                last.day,
                previous_price=100_000,
                price=100_000,
                change=8_000,
            )
        )
        self.assertEqual(duplicate_action, last.real_action)
        self.assertEqual(strategy.activation_events, activation_events_before)
        self.assertEqual(len(strategy.timeline), 14)

    def test_weak_current_forecast_flattens_without_deactivation(self) -> None:
        strategy = make_shadow_strategy("shadow8_markov")
        model = ForecastSequenceModel(
            positive_forecast(),
            # Fourteen labels is the first post-drive decision in this
            # controlled fixture, after the two qualifying evaluations have
            # completed.
            by_day={14: weak_forecast()},
        )
        strategy._model = model
        drive_shadow_to_activation(strategy, days=14)
        self.assertTrue(strategy.real_active)
        prior = strategy.timeline[-1].real_action
        price = 100_000 + 8_000 * 14
        action = strategy.decide(
            make_observation(
                14,
                previous_price=price,
                price=price + 8_000,
                change=8_000,
                own_position=prior,
            )
        )
        self.assertEqual(action, 0)
        self.assertTrue(strategy.real_active)
        self.assertGreaterEqual(strategy.current_edge_gate_count, 1)

    def test_two_bad_shadow_health_evaluations_deactivate(self) -> None:
        strategy = make_shadow_strategy("shadow8_markov")
        strategy._model = ForecastSequenceModel(positive_forecast())
        drive_shadow_to_activation(strategy, days=16)
        self.assertTrue(strategy.real_active)
        price = 100_000 + 8_000 * 16
        prior = strategy.timeline[-1].real_action
        for day in range(16, 32):
            old_price = price
            price -= 5_000
            strategy.decide(
                make_observation(
                    day,
                    previous_price=old_price,
                    price=price,
                    change=-5_000,
                    own_position=prior,
                )
            )
            prior = strategy.timeline[-1].real_action
        self.assertFalse(strategy.real_active)
        self.assertEqual(strategy.deactivation_events[-1]["reason"], "shadow_health")
        self.assertEqual(strategy.deactivation_events[-1]["day"], 25)
        self.assertTrue(
            any(
                record.deactivated_this_day and record.cooldown_active
                for record in strategy.timeline
            )
        )

    def test_five_observation_cooldown_and_causal_reactivation(self) -> None:
        strategy = make_shadow_strategy("shadow8_markov")
        strategy._model = ForecastSequenceModel(positive_forecast())
        drive_shadow_to_activation(strategy, days=16)
        price = 100_000 + 8_000 * 16
        prior = strategy.timeline[-1].real_action
        saw_cooldown = False
        for day in range(16, 32):
            old_price = price
            price -= 5_000
            strategy.decide(
                make_observation(
                    day,
                    previous_price=old_price,
                    price=price,
                    change=-5_000,
                    own_position=prior,
                )
            )
            prior = strategy.timeline[-1].real_action
            saw_cooldown = saw_cooldown or strategy.cooldown_active
            if saw_cooldown and not strategy.cooldown_active:
                break
        self.assertTrue(saw_cooldown)
        self.assertFalse(strategy.real_active)
        self.assertFalse(strategy.cooldown_active)
        cooldown_days = [
            record.day
            for record in strategy.timeline
            if record.cooldown_active
        ]
        self.assertGreaterEqual(len(cooldown_days), 5)
        # The same positive virtual evidence is allowed to requalify after the
        # cooldown, but real activation still has the two-evaluation delay.
        for day in range(strategy.timeline[-1].day + 1, strategy.timeline[-1].day + 16):
            old_price = price
            price += 8_000
            strategy.decide(
                make_observation(
                    day,
                    previous_price=old_price,
                    price=price,
                    change=8_000,
                    own_position=prior,
                )
            )
            prior = strategy.timeline[-1].real_action
        self.assertGreaterEqual(strategy.reactivation_count, 1)
        self.assertTrue(strategy.real_active)

    def test_unknown_zero_reset_and_clipped_are_not_shadow_evidence(self) -> None:
        strategy = make_shadow_strategy("shadow8_markov")
        strategy._model = ForecastSequenceModel(positive_forecast())
        strategy.decide(make_observation(0))
        strategy.decide(
            make_observation(1, previous_price=100_000, price=108_000, change=8_000)
        )
        before = strategy.cumulative_virtual_pnl
        zero = strategy.decide(
            make_observation(2, previous_price=108_000, price=108_000, change=0)
        )
        self.assertEqual(zero, 0)
        self.assertEqual(strategy.cumulative_virtual_pnl, before)
        self.assertFalse(strategy.timeline[-1].virtual_trade_scoreable)
        reset = strategy.decide(
            make_observation(
                3,
                previous_price=108_000,
                price=100_000,
                change=-8_000,
                reset=True,
                market_mode="continuous_reset",
            )
        )
        self.assertEqual(reset, 0)
        self.assertEqual(strategy.timeline[-1].movement_kind, "reset")
        self.assertEqual(strategy.labels, ())
        clipped = strategy.decide(
            make_observation(
                4,
                previous_price=22_000,
                price=20_000,
                change=-2_000,
            )
        )
        self.assertEqual(clipped, 0)
        self.assertEqual(strategy.timeline[-1].movement_kind, "floor_clipped")
        self.assertEqual(strategy.cumulative_virtual_pnl, before)
        self.assertEqual(strategy.timeline[-1].virtual_interval_pnl, 0)

    def test_unknown_movement_breaks_context_and_pauses_one_decision(self) -> None:
        strategy = make_shadow_strategy("shadow8_markov")
        strategy._model = ForecastSequenceModel(positive_forecast())
        strategy.decide(make_observation(0))
        strategy._real_active = True
        strategy.decide(
            make_observation(1, previous_price=100_000, price=108_000, change=8_000)
        )
        paused = strategy.decide(
            make_observation(2, previous_price=108_000, price=108_000, change=0)
        )
        resumed = strategy.decide(
            make_observation(3, previous_price=108_000, price=116_000, change=8_000)
        )
        self.assertEqual(paused, 0)
        self.assertEqual(resumed, 1)
        self.assertTrue(strategy.timeline[-2].pause_active)
        self.assertEqual(strategy.timeline[-2].pause_reason, "unknown_zero")
        self.assertEqual(strategy.timeline[-2].markov_context, ())

    def test_exact_floor_flattening(self) -> None:
        strategy = make_shadow_strategy("shadow8_markov")
        strategy._model = ForecastSequenceModel(positive_forecast())
        strategy.decide(make_observation(0))
        strategy._real_active = True
        action = strategy.decide(
            make_observation(
                1,
                previous_price=25_000,
                price=20_000,
                change=-5_000,
                own_position=0,
            )
        )
        self.assertEqual(action, 0)
        self.assertGreaterEqual(strategy.floor_gate_count, 1)

    def test_sticky_actual_loss_stop_and_overshoot(self) -> None:
        strategy = make_shadow_strategy("shadow8_markov")
        strategy._model = ForecastSequenceModel(weak_forecast())
        strategy.decide(make_observation(0))
        price = 100_000
        for day in range(1, 8):
            old_price = price
            price -= 8_000
            strategy.decide(
                make_observation(
                    day,
                    previous_price=old_price,
                    price=price,
                    change=-8_000,
                    own_position=1,
                )
            )
        self.assertTrue(strategy.loss_stop_active)
        assert strategy.loss_stop_event is not None
        self.assertEqual(strategy.actual_cumulative_pnl, -56_000)
        self.assertEqual(strategy.loss_stop_event.loss_limit_overshoot, 6_000)
        self.assertEqual(
            strategy.decide(
                make_observation(
                    8,
                    previous_price=price,
                    price=price + 8_000,
                    change=8_000,
                    own_position=0,
                )
            ),
            0,
        )

    def test_callable_exposure_is_evaluated_once_per_day_and_cached(self) -> None:
        calls: list[int] = []

        def provider(observation: AgentObservation) -> float:
            calls.append(observation.day)
            return 590_001.0

        strategy = make_shadow_strategy(
            "shadow8_markov",
            other_portfolio_exposure=provider,
        )
        strategy._model = ForecastSequenceModel(positive_forecast())
        strategy.decide(make_observation(0))
        strategy._real_active = True
        first = strategy.decide(
            make_observation(1, previous_price=100_000, price=108_000, change=8_000)
        )
        second = strategy.decide(
            make_observation(1, previous_price=100_000, price=108_000, change=8_000)
        )
        self.assertEqual(first, 0)
        self.assertEqual(second, 0)
        self.assertEqual(calls, [1])
        self.assertEqual(strategy.exposure_evaluation_count, 1)
        self.assertGreaterEqual(strategy.headroom_gate_count, 1)

    def test_headroom_exact_boundary_and_simulator_has_no_focal_rejections(self) -> None:
        allowed = make_shadow_strategy("shadow8_markov", other_portfolio_exposure=10_000)
        allowed._model = ForecastSequenceModel(positive_forecast())
        allowed.decide(make_observation(0))
        allowed._real_active = True
        self.assertEqual(
            allowed.decide(
                make_observation(
                    1,
                    previous_price=100_000,
                    price=580_000,
                    change=480_000,
                )
            ),
            1,
        )

        scenario_agents = [
            make_shadow_strategy(
                "shadow12_markov",
                other_portfolio_exposure=450_000,
            ),
            _AlwaysShort("opponent"),
        ]
        result = LiferaftSimulator(
            scenario_agents,
            LiferaftConfig(total_days=24, marked_boundary_day=0, market_mode="inactive_until_marked"),
            focal_agent_name="focal",
            other_portfolio_exposure={"focal": 450_000},
        ).run()
        self.assertFalse(any(item.agent_name == "focal" for item in result.rejected_actions))
        self.assertFalse(any(item.agent_name == "focal" for item in result.budget_breaches))

    def test_fresh_instances_reproduce_and_both_inactive_modes_start_live(self) -> None:
        for mode in ("observe_and_ignore_actions", "fully_inactive"):
            config = LiferaftConfig(
                total_days=18,
                marked_boundary_day=6,
                market_mode="inactive_until_marked",
                pre_voting_execution=mode,
                voting_start_day=6,
            )

            def run_once():
                focal = make_shadow_strategy("shadow8_markov")
                return LiferaftSimulator(
                    [focal, _AlwaysShort("opponent")],
                    config,
                    focal_agent_name="focal",
                ).run()

            first = run_once()
            second = run_once()
            self.assertEqual(first, second)
            self.assertTrue(all(day.actions["focal"] == 0 for day in first.days[:6]))
            self.assertTrue(all(day.agent_records["focal"].agent_called == (mode == "observe_and_ignore_actions") for day in first.days[:6]))
            self.assertEqual(first.days[7].agent_records["focal"].pnl_position, 0)

    def test_public_information_only_and_no_production_catalogue_registration(self) -> None:
        fields = set(AgentObservation.__dataclass_fields__)
        self.assertNotIn("focal_pivotal", fields)
        self.assertNotIn("vote_margin", fields)
        self.assertTrue(all(name not in COLD_START_STRATEGY_NAMES for name in SHADOW_STRATEGY_NAMES))
        production = Path(__file__).parents[2] / "trader_interface" / "algorithm.py"
        production_source = production.read_text(encoding="utf-8")
        self.assertNotIn("shadow8_markov", production_source)
        self.assertNotIn("shadow12_markov", production_source)
        self.assertNotIn("shadow20_markov", production_source)

    def test_three_candidates_differ_only_in_warmup(self) -> None:
        baseline = SHADOW_PARAMETERS["shadow12_markov"]
        for name in SHADOW_STRATEGY_NAMES:
            candidate = SHADOW_PARAMETERS[name]
            baseline_values = baseline.__dict__.copy()
            candidate_values = candidate.__dict__.copy()
            baseline_values.pop("candidate_name")
            candidate_values.pop("candidate_name")
            if name == "shadow12_markov":
                self.assertEqual(candidate_values, baseline_values)
            else:
                self.assertEqual(
                    {key: value for key, value in candidate_values.items() if key != "minimum_genuine_nonzero_observations"},
                    {key: value for key, value in baseline_values.items() if key != "minimum_genuine_nonzero_observations"},
                )


class _AlwaysShort:
    def __init__(self, name: str):
        self.name = name

    def decide(self, observation: AgentObservation) -> int:
        del observation
        return -1


if __name__ == "__main__":
    unittest.main()
