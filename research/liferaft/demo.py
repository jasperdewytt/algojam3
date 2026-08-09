"""Short reproducible Pass 1 and clarified cold-start demonstrations."""

from __future__ import annotations

from .scenarios import (
    deterministic_mixed_scenario,
    runaway_price_scenario,
    stateful_reset_scenario,
)
from .archetypes import AlwaysLong
from .cold_start_strategies import OnlineLastMajorityCounter
from .pass3_scenarios import cold_start_config
from .simulator import LiferaftSimulator


def _print_summary(result) -> None:
    print(f"scenario={result.scenario_name}")
    print(
        f"  days={len(result.days)} price_start={result.price_path[0]} "
        f"price_end={result.price_path[-1]} price_max={max(result.price_path)}"
    )
    print(f"  calibration_pnl={result.calibration_pnl}")
    print(f"  marked_pnl={result.marked_pnl}")
    print(
        f"  rejected_actions={len(result.rejected_actions)} "
        f"budget_breaches={len(result.budget_breaches)} "
        f"reset_day={result.reset_day} reset_jump={result.reset_jump}"
    )


def main() -> None:
    mixed = deterministic_mixed_scenario().run()
    _print_summary(mixed)

    stateful = stateful_reset_scenario().run()
    _print_summary(stateful)
    boundary = stateful.reset_day
    print("  actions_near_reset:")
    for record in stateful.days[max(0, boundary - 1) : boundary + 2]:
        selected = {
            name: record.actions[name]
            for name in ("win-stay", "boundary-aware", "boundary-unaware", "switcher")
        }
        print(
            f"    day={record.day} price={record.price} "
            f"majority={record.majority.value} actions={selected}"
        )

    runaway = runaway_price_scenario().run()
    _print_summary(runaway)
    first_breach = runaway.budget_breaches[0] if runaway.budget_breaches else None
    if first_breach is not None:
        print(
            "  first_budget_breach="
            f"day {first_breach.day}, price {first_breach.current_price}, "
            f"requested {first_breach.requested_action}, "
            f"effective {first_breach.effective_action}"
        )

    cold_start = OnlineLastMajorityCounter()
    cold_result = LiferaftSimulator(
        [cold_start, AlwaysLong("cold-opponent")],
        cold_start_config(total_days=8, marked_boundary_day=3),
        focal_agent_name=cold_start.name,
        scenario_name="cold-start-smoke",
    ).run()
    print(
        f"scenario={cold_result.scenario_name} "
        f"pre_voting_prices={cold_result.price_path[:4]} "
        f"first_live_price={cold_result.price_path[4]} "
        f"marked_pnl={cold_result.marked_pnl[cold_start.name]}"
    )


if __name__ == "__main__":
    main()
