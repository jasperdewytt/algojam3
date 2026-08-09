"""Focused tests for the frozen Pass 4A risk wrapper and runner."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

from research.liferaft.cold_start_strategies import make_cold_start_strategy
from research.liferaft.pass3_scenarios import validation_scenarios
from research.liferaft.pass3_experiments import _attach_comparisons, _metric_without_comparison
from research.liferaft.pass4_experiments import (
    PASS4_EXPOSURES,
    PASS4_STRATEGIES,
    render_pass4_report,
    run_pass4_case,
    run_pass4_experiment,
)
from research.liferaft.pass4_final import (
    FINAL_CANDIDATES,
    FINAL_HASH_FILES,
    FINAL_SCENARIO_COUNT,
    FINAL_WRAPPER_NAME,
    _wrapper_diagnostics,
    evaluate_decision,
    render_final_report,
    run_final_candidate,
)
from research.liferaft.pass4_strategies import (
    MAX_LIFERAFT_LOSS_AUD,
    PORTFOLIO_RESERVE_AUD,
    Risk50Burnin1Markov,
)
from research.liferaft.simulator import AgentObservation, MajorityOutcome
from research.liferaft.strategies import Forecast


def observation(
    day: int,
    *,
    change: int | None = None,
    own_position: int = 0,
    price: int = 100_000,
    boundary: int = 0,
    reset: bool = False,
    floor: int = 20_000,
    voting_active: bool | None = None,
) -> AgentObservation:
    previous_price = None if change is None else price - change
    return AgentObservation(
        day=day,
        price=price,
        price_history=(price,) if previous_price is None else (previous_price, price),
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
        voting_active=(day >= boundary if voting_active is None else voting_active),
        market_mode="inactive_until_marked",
        voting_start_day=boundary,
    )


class StubDelegate:
    """Test double used only to isolate wrapper gates from model forecasts."""

    name = "burnin-1-markov"

    def __init__(
        self,
        *,
        action: int = -1,
        forecast: Forecast | None = None,
        forecasts: dict[int, Forecast] | None = None,
    ) -> None:
        self.action = action
        self.forecast = forecast or Forecast(0.5, 0.5, 0, "stub")
        self.forecasts = forecasts or {}
        self.last_forecast: Forecast | None = None
        self.calls: list[int] = []

    def decide(self, current: AgentObservation) -> int:
        self.calls.append(current.day)
        self.last_forecast = self.forecasts.get(current.day, self.forecast)
        return self.action


def stub_wrapper(
    *,
    action: int = -1,
    forecast: Forecast | None = None,
    exposures=0.0,
) -> tuple[Risk50Burnin1Markov, StubDelegate]:
    wrapper = Risk50Burnin1Markov(other_portfolio_exposure=exposures)
    delegate = StubDelegate(action=action, forecast=forecast)
    wrapper.delegate = delegate
    return wrapper, delegate


class Pass4WrapperTests(unittest.TestCase):
    def test_wraps_pristine_frozen_burnin1_markov(self) -> None:
        wrapper = Risk50Burnin1Markov()
        self.assertEqual(wrapper.name, "risk50_burnin1_markov")
        self.assertEqual(wrapper.delegate.__class__.__name__, "FlatBurnInStrategy")
        self.assertEqual(wrapper.delegate.burn_in_genuine_observations, 1)
        self.assertEqual(wrapper.delegate._model.order, 2)
        self.assertEqual(wrapper.delegate._model.alpha, 1.0)

    def test_realised_pnl_uses_previous_effective_position_once(self) -> None:
        wrapper, delegate = stub_wrapper()
        wrapper.decide(observation(0))
        current = observation(1, change=-5_000, own_position=1)
        wrapper.decide(current)
        wrapper.decide(current)
        self.assertEqual(wrapper.realised_increment_by_day, {1: -5_000})
        self.assertEqual(wrapper.cumulative_marked_pnl, -5_000)
        self.assertEqual(delegate.calls, [0, 1])

    def test_inactive_and_reset_movements_do_not_create_pnl(self) -> None:
        wrapper, _delegate = stub_wrapper()
        for day in range(4):
            wrapper.decide(observation(day, boundary=3, own_position=1))
        wrapper.decide(
            observation(
                4,
                boundary=3,
                change=-5_000,
                own_position=1,
                reset=True,
            )
        )
        self.assertEqual(wrapper.cumulative_marked_pnl, 0)
        self.assertEqual(wrapper.realised_increment_by_day, {})

    def test_clipped_move_uses_actual_public_price_change(self) -> None:
        wrapper, _delegate = stub_wrapper()
        wrapper.decide(observation(0))
        wrapper.decide(
            observation(
                1,
                change=-2_000,
                price=20_000,
                own_position=-1,
            )
        )
        self.assertEqual(wrapper.cumulative_marked_pnl, 2_000)
        self.assertEqual(wrapper.realised_increment_by_day[1], 2_000)

    def test_loss_limit_exact_trigger_is_sticky(self) -> None:
        wrapper, _delegate = stub_wrapper()
        wrapper.decide(observation(0))
        for day in range(1, 11):
            action = wrapper.decide(
                observation(day, change=-5_000, own_position=1)
            )
        self.assertEqual(wrapper.cumulative_marked_pnl, -MAX_LIFERAFT_LOSS_AUD)
        self.assertTrue(wrapper.loss_stop_active)
        self.assertEqual(wrapper.loss_stop_event.day, 10)
        self.assertEqual(wrapper.loss_stop_event.loss_limit_overshoot, 0)
        self.assertEqual(action, 0)
        self.assertEqual(wrapper.decide(observation(11, own_position=0)), 0)

    def test_loss_overshoot_is_bounded_by_final_adverse_move(self) -> None:
        wrapper, _delegate = stub_wrapper()
        wrapper.decide(observation(0))
        for day in range(1, 10):
            wrapper.decide(observation(day, change=-5_000, own_position=1))
        wrapper.decide(observation(10, change=-8_000, own_position=1))
        event = wrapper.loss_stop_event
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.pnl_before, -45_000)
        self.assertEqual(event.pnl_after, -53_000)
        self.assertEqual(event.loss_limit_overshoot, 3_000)
        self.assertLessEqual(event.loss_limit_overshoot, 8_000)

    def test_health_scores_prior_forecast_not_same_day_forecast(self) -> None:
        wrapper = Risk50Burnin1Markov()
        delegate = StubDelegate(
            action=-1,
            forecasts={
                0: Forecast(1.0, 0.0, 8, "prior-long"),
                1: Forecast(0.0, 1.0, 8, "new-short"),
            },
        )
        wrapper.delegate = delegate
        wrapper.decide(observation(0))
        wrapper.decide(observation(1, change=-5_000))
        self.assertEqual(wrapper.quality_history, (True,))
        self.assertEqual(wrapper.quality_scoreable, 1)

    def test_unsupported_tied_zero_and_reset_outcomes_are_not_scored(self) -> None:
        cases = (
            (Forecast(0.5, 0.5, 0, "unsupported"), -5_000, False),
            (Forecast(0.5, 0.5, 8, "tied"), -5_000, False),
            (Forecast(0.8, 0.2, 8, "supported"), 0, False),
            (Forecast(0.8, 0.2, 8, "supported"), -5_000, True),
        )
        for forecast, change, reset in cases:
            wrapper, _delegate = stub_wrapper(forecast=forecast)
            wrapper.decide(observation(0))
            wrapper.decide(observation(1, change=change, reset=reset))
            self.assertEqual(wrapper.quality_scoreable, 0)

    def test_health_stop_is_sticky_after_two_unhealthy_evaluations(self) -> None:
        wrapper, _delegate = stub_wrapper(
            action=-1,
            forecast=Forecast(1.0, 0.0, 8, "always-long"),
        )
        wrapper.decide(observation(0))
        actions = [
            wrapper.decide(observation(day, change=8_000))
            for day in range(1, 10)
        ]
        self.assertTrue(wrapper.health_stop_active)
        self.assertEqual(wrapper.health_stop_event.day, 9)
        self.assertEqual(actions[-1], 0)
        self.assertEqual(wrapper.decide(observation(10, change=8_000)), 0)

    def test_unknown_movement_pauses_exactly_one_decision(self) -> None:
        wrapper, _delegate = stub_wrapper()
        wrapper.decide(observation(0))
        self.assertEqual(wrapper.decide(observation(1, change=0)), 0)
        self.assertEqual(wrapper.unknown_pause_count, 1)
        self.assertEqual(wrapper.decide(observation(2, change=8_000)), -1)

    def test_floor_gate_returns_flat_at_exact_floor(self) -> None:
        wrapper, _delegate = stub_wrapper()
        wrapper.decide(observation(0))
        self.assertEqual(
            wrapper.decide(observation(1, change=-5_000, price=20_000)),
            0,
        )
        self.assertEqual(wrapper.floor_gate_count, 1)
        self.assertEqual(wrapper.decide(observation(2, change=8_000, price=28_000)), -1)

    def test_exact_headroom_boundary_is_allowed_and_excess_is_flattened(self) -> None:
        wrapper, _delegate = stub_wrapper(exposures=10_000)
        wrapper.decide(observation(0))
        self.assertEqual(
            wrapper.decide(observation(1, change=8_000, price=580_000)),
            -1,
        )

        blocked, _delegate = stub_wrapper(exposures=0)
        blocked.decide(observation(0))
        self.assertEqual(
            blocked.decide(observation(1, change=8_000, price=590_001)),
            0,
        )
        self.assertEqual(blocked.headroom_gate_count, 1)
        self.assertEqual(
            0 + 590_000 + PORTFOLIO_RESERVE_AUD,
            600_000,
        )

    def test_constant_and_callable_exposure_sources_are_used(self) -> None:
        seen: list[int] = []

        def provider(current: AgentObservation) -> float:
            seen.append(current.day)
            return 0.0 if current.day == 1 else 500_000.0

        wrapper, _delegate = stub_wrapper(exposures=provider)
        wrapper.decide(observation(0))
        self.assertEqual(wrapper.decide(observation(1, change=8_000)), -1)
        self.assertEqual(wrapper.decide(observation(2, change=8_000)), 0)
        self.assertEqual(seen, [0, 1, 2])

    def test_invalid_exposure_values_are_rejected(self) -> None:
        for value in (-1, float("nan"), float("inf"), True, "bad"):
            with self.assertRaises(ValueError):
                Risk50Burnin1Markov(other_portfolio_exposure=value)
        wrapper, _delegate = stub_wrapper(
            exposures=lambda current: -1.0,
        )
        wrapper.decide(observation(0, boundary=1, voting_active=False))
        with self.assertRaises(ValueError):
            wrapper.decide(observation(1, change=8_000))

    def test_wrapper_exposes_no_hidden_vote_information(self) -> None:
        current = observation(0)
        self.assertFalse(hasattr(current, "long_count"))
        self.assertFalse(hasattr(current, "focal_pivotal"))
        wrapper, _delegate = stub_wrapper()
        self.assertIn(wrapper.decide(current), (-1, 0, 1))


class Pass4RunnerTests(unittest.TestCase):
    def test_both_execution_modes_are_flat_before_live_and_first_movement_is_timed(self) -> None:
        scenarios = validation_scenarios()
        pair_id = scenarios[0].pair_id
        pair = [scenario for scenario in scenarios if scenario.pair_id == pair_id]
        self.assertEqual({scenario.execution_mode for scenario in pair}, {
            "observe_and_ignore_actions",
            "fully_inactive",
        })
        for scenario in pair:
            result, wrapper = run_pass4_case(
                scenario,
                Risk50Burnin1Markov.name,
                other_portfolio_exposure=0.0,
            )
            assert wrapper is not None
            start = result.config.voting_start_day
            assert start is not None
            focal = result.focal_agent_name
            assert focal is not None
            self.assertTrue(all(day.price == 100_000 for day in result.days[: start + 1]))
            self.assertTrue(all(result.days[day].actions[focal] == 0 for day in range(start)))
            self.assertEqual(result.days[start].agent_records[focal].action, 0)
            self.assertEqual(result.days[start + 1].agent_records[focal].pnl_position, 0)
            if scenario.execution_mode == "observe_and_ignore_actions":
                self.assertTrue(all(result.days[day].agent_records[focal].agent_called for day in range(start)))
            else:
                self.assertTrue(all(not result.days[day].agent_records[focal].agent_called for day in range(start)))

    def test_pass4_case_is_reproducible_and_actions_are_integral(self) -> None:
        scenario = validation_scenarios()[0]
        first, first_wrapper = run_pass4_case(
            scenario,
            Risk50Burnin1Markov.name,
            other_portfolio_exposure=150_000,
        )
        second, second_wrapper = run_pass4_case(
            scenario,
            Risk50Burnin1Markov.name,
            other_portfolio_exposure=150_000,
        )
        self.assertEqual(first, second)
        assert first_wrapper is not None and second_wrapper is not None
        self.assertEqual(first_wrapper.diagnostics(), second_wrapper.diagnostics())
        focal = first.focal_agent_name
        assert focal is not None
        self.assertTrue(
            all(
                type(day.agent_records[focal].action) is int
                and day.agent_records[focal].action in (-1, 0, 1)
                for day in first.days
            )
        )

    def test_wrapper_pnl_matches_simulator_marked_ledger_and_has_no_focal_rejections(self) -> None:
        result, wrapper = run_pass4_case(
            validation_scenarios()[0],
            Risk50Burnin1Markov.name,
            other_portfolio_exposure=450_000,
        )
        assert wrapper is not None
        focal = result.focal_agent_name
        assert focal is not None
        self.assertEqual(wrapper.cumulative_marked_pnl, result.marked_pnl[focal])
        self.assertFalse(any(item.agent_name == focal for item in result.rejected_actions))
        self.assertFalse(any(item.agent_name == focal for item in result.budget_breaches))

    def test_callable_exposure_is_shared_cached_and_deterministic_through_runner(self) -> None:
        scenario = validation_scenarios()[0]

        def make_provider():
            calls: list[int] = []
            values: dict[int, float] = {}

            def provider(current: AgentObservation) -> float:
                calls.append(current.day)
                value = 0.0 if current.day % 2 == 0 else 590_000.0
                values[current.day] = value
                return value

            return provider, calls, values

        provider, calls, values = make_provider()
        first, first_wrapper = run_pass4_case(
            scenario,
            Risk50Burnin1Markov.name,
            other_portfolio_exposure=provider,
        )
        provider_again, calls_again, values_again = make_provider()
        second, second_wrapper = run_pass4_case(
            scenario,
            Risk50Burnin1Markov.name,
            other_portfolio_exposure=provider_again,
        )
        assert first_wrapper is not None and second_wrapper is not None
        focal = first.focal_agent_name
        assert focal is not None
        start = first.config.voting_start_day
        assert start is not None
        live = tuple(record for record in first.agent_history(focal) if record.day >= start)

        # Liferaft budget resolution happens once per live focal decision.  A
        # pre-voting request is ignored before the exposure callback is needed.
        self.assertEqual(calls, list(range(start, first.config.total_days)))
        self.assertEqual(calls_again, calls)
        self.assertEqual(values_again, values)
        self.assertTrue(
            all(record.other_portfolio_exposure == values[record.day] for record in live)
        )

        # Nonzero effective actions are allowed only at the provider's
        # low-exposure days; high-exposure days are stopped by the wrapper's
        # headroom gate before the simulator can reject them.
        allowed = [record for record in live if record.action != 0]
        self.assertTrue(allowed)
        self.assertTrue(
            all(
                record.other_portfolio_exposure
                + abs(record.action) * record.price
                + PORTFOLIO_RESERVE_AUD
                <= first.config.gross_portfolio_budget
                for record in allowed
            )
        )
        self.assertGreater(first_wrapper.headroom_gate_count, 0)
        self.assertFalse(any(item.agent_name == focal for item in first.budget_breaches))
        self.assertFalse(any(item.agent_name == focal for item in first.rejected_actions))
        self.assertEqual(first.days, second.days)
        self.assertEqual(first.price_path, second.price_path)
        self.assertEqual(first_wrapper.diagnostics(), second_wrapper.diagnostics())

    def test_report_separates_both_stop_ordering_and_overshoot_distribution(self) -> None:
        metrics = run_pass4_experiment(
            validation_scenarios()[:1],
            strategies=PASS4_STRATEGIES,
            exposures=PASS4_EXPOSURES,
            write_report=False,
        )
        report = render_pass4_report(metrics)
        self.assertIn("both: loss first", report)
        self.assertIn("first loss (all stopped)", report)
        self.assertIn("## Loss-limit overshoot distribution", report)
        self.assertIn("percentage of loss-stop cases", report)
        self.assertIn("| 0 | 0 | 0 | 0.0% |", report)

    def test_final_wrapper_constructs_and_scores_on_consumed_case_only(self) -> None:
        scenario = validation_scenarios()[0]
        result, wrapper = run_final_candidate(scenario, FINAL_WRAPPER_NAME)
        self.assertIsNotNone(wrapper)
        assert wrapper is not None
        self.assertEqual(wrapper.name, FINAL_WRAPPER_NAME)
        self.assertEqual(result.focal_agent_name, FINAL_WRAPPER_NAME)
        self.assertEqual(result.marked_pnl[FINAL_WRAPPER_NAME], wrapper.cumulative_marked_pnl)
        self.assertEqual(len(FINAL_CANDIDATES), 15)
        self.assertEqual(FINAL_SCENARIO_COUNT, 160)

    def test_final_report_and_decision_render_on_consumed_case(self) -> None:
        scenario = validation_scenarios()[0]
        metrics = []
        diagnostics = {}
        for strategy in FINAL_CANDIDATES:
            result, wrapper = run_final_candidate(scenario, strategy)
            metrics.append(
                _metric_without_comparison(
                    result,
                    suite="final",
                    scenario=scenario,
                    strategy=strategy,
                )
            )
            if wrapper is not None:
                diagnostics[(scenario.name, scenario.execution_mode)] = _wrapper_diagnostics(wrapper)
        metrics = _attach_comparisons(metrics)
        decision = evaluate_decision(metrics, diagnostics)
        report = render_final_report(
            metrics,
            diagnostics,
            combined_hash="A" * 64,
            timestamp="test",
            decision=decision,
        )
        self.assertIn("risk50_burnin1_markov", report)
        self.assertIn("Loss-limit overshoot distribution", report)
        self.assertIn(decision["decision"], {"PASS", "FAIL"})

    def test_failed_final_gate_keeps_production_liferaft_flat(self) -> None:
        from trader_interface.algorithm import Algorithm

        limits = {
            "Fintech Token": 100,
            "Thrifted Jeans": 800,
            "UQ Dollar": 650,
            "Sausage Sizzle": 3000,
            "Bread": 500,
            "MenuDash": 75_000,
            "Sausage": 5_000,
            "Liferaft Ticket": 1,
            "Boat Party Ticket": 1_000,
        }
        algorithm = Algorithm({name: 0 for name in limits})
        algorithm.positionLimits = limits
        algorithm.data = {name: [100.0, 100.0] for name in limits}
        algorithm.data["Liferaft Ticket"] = [100_000.0, 100_000.0]
        self.assertEqual(algorithm.get_positions()["Liferaft Ticket"], 0)

    def test_small_runner_uses_only_consumed_validation_cases(self) -> None:
        metrics = run_pass4_experiment(
            validation_scenarios()[:1],
            strategies=PASS4_STRATEGIES,
            exposures=PASS4_EXPOSURES,
            write_report=False,
        )
        self.assertEqual(len(metrics), 16)
        self.assertEqual({metric.other_portfolio_exposure for metric in metrics}, set(PASS4_EXPOSURES))

    def test_pass4_quarantine_and_locked_hashes(self) -> None:
        root = Path(__file__).parent
        manifest_path = root / "PASS3_FINAL_MANIFEST.md"
        manifest = manifest_path.read_text(encoding="utf-8")
        expected = dict(
            re.findall(r"\| `([^`]+)` \| `([A-F0-9]{64})` \|", manifest)
        )
        names = FINAL_HASH_FILES
        rows = []
        for name in names:
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest().upper()
            self.assertEqual(actual, expected[name], name)
            rows.append(f"{name} {actual}")
        combined = hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest().upper()
        manifest_combined = re.search(r"Combined hash: `([A-F0-9]{64})`", manifest)
        self.assertIsNotNone(manifest_combined)
        assert manifest_combined is not None
        self.assertEqual(combined, manifest_combined.group(1))
        final_report = root / "PASS3_FINAL_REPORT.md"
        final_results = root / "PASS4_FINAL_RESULTS.json"
        final_decision = root / "PASS4_FINAL_DECISION.md"
        receipt = root / "PASS4_FINAL_EXECUTION_RECEIPT.json"
        if final_report.exists():
            self.assertTrue(final_results.exists())
            self.assertTrue(final_decision.exists())
            self.assertTrue(receipt.exists())
            receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(receipt_data["status"], "consumed")
            self.assertEqual(receipt_data["locked_combined_hash"], manifest_combined.group(1))
        else:
            self.assertFalse(final_results.exists())
            self.assertFalse(final_decision.exists())
            self.assertFalse(receipt.exists())
        source = (root / "pass4_experiments.py").read_text(encoding="utf-8")
        self.assertNotIn("final_scenarios(", source)


if __name__ == "__main__":
    unittest.main()
