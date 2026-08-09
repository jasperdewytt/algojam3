import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trader_interface"))

from algorithm import Algorithm


LIMITS = {
    "Fintech Token": 100,
    "Thrifted Jeans": 800,
    "UQ Dollar": 650,
    "Sausage Sizzle": 3000,
    "Bread": 500,
    "MenuDash": 75000,
    "Sausage": 5000,
    "Liferaft Ticket": 1,
    "Boat Party Ticket": 1000,
}

PRICES = {
    "Fintech Token": 600.0,
    "Thrifted Jeans": 75.0,
    "UQ Dollar": 100.0,
    "Sausage Sizzle": 40.0,
    "Bread": 125.0,
    "MenuDash": 2.0,
    "Sausage": 6.0,
    "Liferaft Ticket": 100_000.0,
    "Boat Party Ticket": 45.0,
}


class BudgetAllocatorTests(unittest.TestCase):
    def setUp(self):
        self.algorithm = Algorithm({name: 0 for name in LIMITS})
        self.algorithm.positionLimits = LIMITS
        self.algorithm.data = {name: [price] for name, price in PRICES.items()}

    @staticmethod
    def gross(positions, prices=PRICES):
        return sum(abs(position) * prices[name] for name, position in positions.items())

    def test_within_budget_positions_are_unchanged(self):
        desired = {name: limit for name, limit in LIMITS.items()}
        desired["Liferaft Ticket"] = 0

        adjusted = self.algorithm._apply_budget(desired)

        self.assertEqual(adjusted, desired)
        self.assertLessEqual(self.gross(adjusted), 600_000)

    def test_weakest_positions_are_trimmed_first(self):
        desired = {name: -limit for name, limit in LIMITS.items()}

        adjusted = self.algorithm._apply_budget(desired)

        self.assertEqual(adjusted["Bread"], 0)
        self.assertEqual(adjusted["Sausage"], 0)
        self.assertEqual(adjusted["Sausage Sizzle"], desired["Sausage Sizzle"])
        self.assertEqual(adjusted["Liferaft Ticket"], -1)
        self.assertTrue(all(type(position) is int for position in adjusted.values()))
        self.assertLessEqual(self.gross(adjusted), 600_000)

    def test_partial_trim_moves_position_toward_zero(self):
        desired = {name: limit for name, limit in LIMITS.items()}
        desired["Liferaft Ticket"] = 0
        self.algorithm.data["MenuDash"] = [2.2]
        prices = dict(PRICES, MenuDash=2.2)

        adjusted = self.algorithm._apply_budget(desired)

        self.assertGreater(adjusted["Bread"], 0)
        self.assertLess(adjusted["Bread"], desired["Bread"])
        self.assertLessEqual(self.gross(adjusted, prices), 600_000)

    def test_unaffordable_liferaft_is_flattened(self):
        desired = {name: 0 for name in LIMITS}
        desired["Liferaft Ticket"] = 1
        self.algorithm.data["Liferaft Ticket"] = [600_001.0]

        adjusted = self.algorithm._apply_budget(desired)

        self.assertEqual(adjusted["Liferaft Ticket"], 0)


class BoatCalendarTests(unittest.TestCase):
    def boat_position(self, history_day, final_price=45.0):
        positions = {name: 0 for name in LIMITS}
        algorithm = Algorithm(positions)
        algorithm.positionLimits = LIMITS
        prices = [45.0] * (history_day + 1)
        prices[-1] = final_price
        algorithm.data = {"Boat Party Ticket": prices}
        return algorithm._boat_party_position()

    def test_year_two_semester_signal_repeats(self):
        # Calendar day 7 is the first frozen positive-slope day.
        self.assertEqual(
            self.boat_position(365 + 7),
            LIMITS["Boat Party Ticket"],
        )

    def test_year_two_neutral_day_uses_existing_history(self):
        # Global day 365 is calendar day zero, but it is not a new warm-up.
        self.assertEqual(
            self.boat_position(365, final_price=60.0),
            -LIMITS["Boat Party Ticket"],
        )

    def test_year_two_summer_anchor_repeats(self):
        self.assertEqual(
            self.boat_position(365 + 322, final_price=44.0),
            LIMITS["Boat Party Ticket"],
        )

    def test_round_boundary_can_trade_normally(self):
        self.assertEqual(
            self.boat_position(364, final_price=44.0),
            LIMITS["Boat Party Ticket"],
        )

    def test_year_two_last_interval_trades_then_terminal_flattens(self):
        self.assertEqual(
            self.boat_position(729, final_price=44.0),
            LIMITS["Boat Party Ticket"],
        )
        self.assertEqual(self.boat_position(730, final_price=44.0), 0)


if __name__ == "__main__":
    unittest.main()
