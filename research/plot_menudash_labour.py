"""Plot the causal Sizzle-implied labour sensor against MenuDash."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "trader_interface" / "data"
OUTPUT = Path(__file__).with_name("menudash_labour_sensor.png")


def load_price(name):
    return pd.read_csv(DATA / f"{name}_price_history.csv")["Price"].to_numpy(float)


menu = load_price("MenuDash")
sizzle = load_price("Sausage Sizzle")
bread = load_price("Bread")
sausage = load_price("Sausage")

days = []
labour_estimate = []
menu_fair_value = []

for day in range(30, len(menu)):
    observation_days = np.arange(1, day)
    change_design = np.column_stack(
        [
            np.ones(len(observation_days)),
            bread[observation_days] - bread[observation_days - 1],
            sausage[observation_days] - sausage[observation_days - 1],
        ]
    )
    change_target = (
        sizzle[observation_days + 1] - sizzle[observation_days]
    )
    ingredient_coefficients, *_ = np.linalg.lstsq(
        change_design,
        change_target,
        rcond=None,
    )

    labour_history = (
        sizzle[1:day + 1]
        - ingredient_coefficients[1] * bread[:day]
        - ingredient_coefficients[2] * sausage[:day]
    )
    latest_labour = labour_history[-1]

    fair_values = []
    for window in (None, 60, 120):
        sensor = labour_history
        target = menu[:day]

        if window is not None:
            sensor = sensor[-window:]
            target = target[-window:]

        mapping_design = np.column_stack(
            [np.ones(len(sensor)), sensor]
        )
        mapping, *_ = np.linalg.lstsq(
            mapping_design,
            target,
            rcond=None,
        )
        fair_values.append(float(mapping @ [1.0, latest_labour]))

    days.append(day)
    labour_estimate.append(latest_labour)
    menu_fair_value.append(np.median(fair_values))

days = np.asarray(days)
labour_estimate = np.asarray(labour_estimate)
menu_fair_value = np.asarray(menu_fair_value)
observed_menu = menu[days]

menu_z = (observed_menu - observed_menu.mean()) / observed_menu.std()
labour_z = (
    (labour_estimate - labour_estimate.mean())
    / labour_estimate.std()
)

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(days, observed_menu, label="Observed MenuDash", linewidth=1.5)
axes[0].plot(
    days,
    menu_fair_value,
    label="Labour-implied fair price",
    linewidth=1.7,
)
axes[0].set_ylabel("Price (AUD)")
axes[0].set_title("MenuDash price versus Sizzle-implied labour fair value")
axes[0].legend(loc="upper left")

axes[1].plot(days, menu_z, label="MenuDash price (standardised)", linewidth=1.4)
axes[1].plot(
    days,
    labour_z,
    label="Hidden labour estimate (standardised)",
    linewidth=1.4,
)
axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.5)
axes[1].set_xlabel("Simulation day")
axes[1].set_ylabel("Standard deviations")
axes[1].set_title("Co-movement on comparable scales")
axes[1].legend(loc="upper left")

fig.tight_layout()
fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
print(OUTPUT)
