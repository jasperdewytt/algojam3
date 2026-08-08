"""Causal MenuDash candidate research.

This file is exploratory only. It deliberately reports a fixed candidate set
rather than selecting and exporting the highest-P&L rule.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "trader_interface"
    / "data"
    / "MenuDash_price_history.csv"
)
LIMIT = 75_000
QUARTERS = ((0, 91), (91, 182), (182, 273), (273, 364))


def strategy_pnl(price, position):
    return np.asarray(position[:-1], dtype=float) * np.diff(price)


def median_residual_position(price, window):
    position = np.zeros(len(price), dtype=int)
    for day in range(1, len(price)):
        fair_value = np.median(price[max(0, day - window):day])
        position[day] = int(LIMIT * np.sign(fair_value - price[day]))
    return position


def median_ensemble_position(price):
    votes = np.sum(
        [np.sign(median_residual_position(price, window)) for window in (5, 7, 10)],
        axis=0,
    )
    return (LIMIT * np.sign(votes)).astype(int)


def robust_alpha_beta_position(
    price,
    alpha=0.20,
    beta=0.05,
    damping=0.90,
    clip_multiple=2.0,
):
    """Robust causal local-level/local-trend state filter."""
    position = np.zeros(len(price), dtype=int)
    level = float(price[0])
    trend = 0.0
    innovations = []

    for day in range(1, len(price)):
        predicted_level = level + damping * trend
        innovation = float(price[day] - predicted_level)

        if len(innovations) >= 10:
            recent = np.asarray(innovations[-30:])
            centre = np.median(recent)
            scale = 1.4826 * np.median(np.abs(recent - centre))
            bound = max(0.01, clip_multiple * scale)
            robust_innovation = float(np.clip(innovation, -bound, bound))
        else:
            robust_innovation = innovation

        level = predicted_level + alpha * robust_innovation
        trend = damping * trend + beta * robust_innovation
        next_forecast = level + damping * trend
        position[day] = int(LIMIT * np.sign(next_forecast - price[day]))
        innovations.append(innovation)

    return position


def residual_regression_position(price, fair_window=7, fit_window=60):
    """Rolling causal OLS: next change ~ intercept + current fair-value residual."""
    position = np.zeros(len(price), dtype=int)

    for day in range(max(20, fair_window + 2), len(price)):
        first_observation = max(fair_window, day - fit_window)
        rows = []
        targets = []

        # At decision day t, outcomes through t-1 are observable because
        # target[t-1] is price[t] - price[t-1].
        for observation_day in range(first_observation, day):
            fair_value = np.median(
                price[observation_day - fair_window:observation_day]
            )
            rows.append([1.0, price[observation_day] - fair_value])
            targets.append(
                price[observation_day + 1] - price[observation_day]
            )

        coefficients, *_ = np.linalg.lstsq(
            np.asarray(rows),
            np.asarray(targets),
            rcond=None,
        )

        current_fair_value = np.median(price[day - fair_window:day])
        current_residual = price[day] - current_fair_value
        prediction = coefficients[0] + coefficients[1] * current_residual
        position[day] = int(LIMIT * np.sign(prediction))

    return position


def theil_sen_position(price, window=10):
    """One-sided robust local linear extrapolation."""
    position = np.zeros(len(price), dtype=int)

    for day in range(window - 1, len(price)):
        values = price[day - window + 1:day + 1]
        slopes = []
        for left in range(window - 1):
            for right in range(left + 1, window):
                slopes.append((values[right] - values[left]) / (right - left))
        slope = float(np.median(slopes))
        time = np.arange(window, dtype=float)
        intercept = float(np.median(values - slope * time))
        forecast = intercept + slope * window
        position[day] = int(LIMIT * np.sign(forecast - price[day]))

    return position


def expert_aggregation_position(price, learning_rate=0.25):
    """Exponentially weight causal expert votes using realised sign accuracy."""
    expert_positions = np.asarray(
        [
            np.full(len(price), LIMIT, dtype=int),
            -np.r_[0, LIMIT * np.sign(np.diff(price))].astype(int),
            *(median_residual_position(price, window) for window in (3, 5, 7, 10, 15)),
            *(robust_alpha_beta_position(price, alpha=alpha) for alpha in (0.10, 0.20, 0.40)),
        ]
    )
    weights = np.ones(len(expert_positions), dtype=float)
    aggregate = np.zeros(len(price), dtype=int)

    for day in range(1, len(price)):
        realised_change = price[day] - price[day - 1]
        previous_votes = np.sign(expert_positions[:, day - 1])
        reward = previous_votes * np.sign(realised_change)
        weights *= np.exp(learning_rate * reward)
        weights /= weights.sum()
        vote = float(weights @ np.sign(expert_positions[:, day]))
        aggregate[day] = int(LIMIT * np.sign(vote))

    return aggregate


def labour_sensor_position(
    menu_price,
    sizzle_price,
    bread_price,
    sausage_price,
    warmup=30,
    mapping_windows=(None, 60, 120),
):
    """Trade MenuDash against a causal Sizzle-implied labour fair value.

    Sizzle at day t contains day t-1 ingredient and labour costs. After
    removing causally estimated Bread and Sausage loadings, its residual is
    an independent sensor for the latent labour value behind MenuDash.
    """
    n_days = len(menu_price)
    mapping_positions = []

    for mapping_window in mapping_windows:
        position = np.zeros(n_days, dtype=int)

        for day in range(warmup, n_days):
            observation_days = np.arange(1, day)
            change_design = np.column_stack(
                [
                    np.ones(len(observation_days)),
                    bread_price[observation_days]
                    - bread_price[observation_days - 1],
                    sausage_price[observation_days]
                    - sausage_price[observation_days - 1],
                ]
            )
            change_target = (
                sizzle_price[observation_days + 1]
                - sizzle_price[observation_days]
            )
            ingredient_coefficients, *_ = np.linalg.lstsq(
                change_design,
                change_target,
                rcond=None,
            )

            labour_sensor = (
                sizzle_price[1:day + 1]
                - ingredient_coefficients[1] * bread_price[:day]
                - ingredient_coefficients[2] * sausage_price[:day]
            )
            mapping_target = menu_price[:day]

            if mapping_window is not None:
                labour_sensor = labour_sensor[-mapping_window:]
                mapping_target = mapping_target[-mapping_window:]

            mapping_design = np.column_stack(
                [np.ones(len(labour_sensor)), labour_sensor]
            )
            mapping_coefficients, *_ = np.linalg.lstsq(
                mapping_design,
                mapping_target,
                rcond=None,
            )

            latest_sensor = (
                sizzle_price[day]
                - ingredient_coefficients[1] * bread_price[day - 1]
                - ingredient_coefficients[2] * sausage_price[day - 1]
            )
            fair_value = float(
                mapping_coefficients @ np.array([1.0, latest_sensor])
            )
            position[day] = int(
                LIMIT * np.sign(fair_value - menu_price[day])
            )

        mapping_positions.append(position)

    votes = np.sum(np.sign(mapping_positions), axis=0)
    return (LIMIT * np.sign(votes)).astype(int)


def candidate_positions(price):
    candidates = {
        "Always long": np.full(len(price), LIMIT, dtype=int),
        "One-day reversal": -np.r_[0, LIMIT * np.sign(np.diff(price))].astype(int),
        "Median ensemble 5/7/10": median_ensemble_position(price),
    }

    for alpha in (0.10, 0.20, 0.40):
        candidates[f"Robust state alpha={alpha}"] = robust_alpha_beta_position(
            price,
            alpha=alpha,
        )

    for fair_window in (5, 7, 10):
        for fit_window in (30, 60, 120):
            candidates[
                f"Residual OLS fair={fair_window} fit={fit_window}"
            ] = residual_regression_position(
                price,
                fair_window=fair_window,
                fit_window=fit_window,
            )

    for window in (7, 10, 14, 21):
        candidates[f"Theil-Sen {window}"] = theil_sen_position(price, window)

    for learning_rate in (0.10, 0.25, 0.50):
        candidates[
            f"Expert aggregation eta={learning_rate}"
        ] = expert_aggregation_position(price, learning_rate)

    return candidates


def summarise(price, candidates):
    rows = []
    for name, position in candidates.items():
        pnl = strategy_pnl(price, position)
        equity = np.r_[0.0, np.cumsum(pnl)]
        drawdown = equity - np.maximum.accumulate(equity)
        blocks = [pnl[start:end].sum() for start, end in QUARTERS]
        rows.append(
            {
                "Candidate": name,
                "P&L": pnl.sum(),
                "Max drawdown": drawdown.min(),
                "Positive quarters": sum(value > 0 for value in blocks),
                **{
                    f"Q{number}": value
                    for number, value in enumerate(blocks, 1)
                },
            }
        )
    return pd.DataFrame(rows).sort_values("P&L", ascending=False)


if __name__ == "__main__":
    menu_price = pd.read_csv(DATA_PATH)["Price"].to_numpy(dtype=float)
    data_directory = DATA_PATH.parent
    sizzle_price = pd.read_csv(
        data_directory / "Sausage Sizzle_price_history.csv"
    )["Price"].to_numpy(dtype=float)
    bread_price = pd.read_csv(
        data_directory / "Bread_price_history.csv"
    )["Price"].to_numpy(dtype=float)
    sausage_price = pd.read_csv(
        data_directory / "Sausage_price_history.csv"
    )["Price"].to_numpy(dtype=float)

    candidates = candidate_positions(menu_price)
    candidates["Sizzle labour-sensor ensemble"] = labour_sensor_position(
        menu_price,
        sizzle_price,
        bread_price,
        sausage_price,
    )
    results = summarise(menu_price, candidates)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
