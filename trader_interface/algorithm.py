import numpy as np


# Custom trading Algorithm
class Algorithm:

    SIZZLE_SEED = np.array(
        [
            0.00530057,  # Intercept
            0.07690641,  # Bread change
            1.67799649,  # Sausage change
        ]
    )

    SIZZLE_WINDOW = 15
    UQ_PEG = 100.0
    FOOD_EMA_ALPHAS = (0.10, 0.15, 0.20)

    MENUDASH_WARMUP = 30
    MENUDASH_MAPPING_WINDOWS = (None, 60, 120)

    JEANS_Q_LEVEL = 1.0
    JEANS_Q_SLOPE = 0.05
    JEANS_OBSERVATION_VARIANCE = 9.0
    JEANS_INITIAL_SLOPE_VARIANCE = 1.0
    JEANS_SHORT_CONFIDENCE = 1.0

    # FUNCTION TO SETUP ALGORITHM CLASS
    def __init__(self, positions):
        self.data = {}  # Historical data of all instruments
        self.positionLimits = {}  # Initialise position limits
        self.day = 0  # Initialise the current day as 0
        self.positions = positions  # Initialise the current positions

    def get_current_price(self, instrument):
        """
        Helper function to fetch current price of an instrument.
        """
        return self.data[instrument][-1]

    # RETURN DESIRED POSITIONS IN DICT FORM
    def get_positions(self):
        desired_positions = {
            instrument: 0 for instrument in self.positionLimits
        }

        desired_positions["UQ Dollar"] = self._uq_dollar_position()
        desired_positions["Sausage Sizzle"] = self._sausage_sizzle_position()
        desired_positions["Bread"] = self._food_ema_position("Bread")
        desired_positions["Sausage"] = self._food_ema_position("Sausage")
        desired_positions["MenuDash"] = self._menudash_position()
        desired_positions["Thrifted Jeans"] = self._thrifted_jeans_position()

        return desired_positions

    def _uq_dollar_position(self):
        price = self.data["UQ Dollar"][-1]
        limit = self.positionLimits["UQ Dollar"]

        if price > self.UQ_PEG:
            return -limit

        if price < self.UQ_PEG:
            return limit

        return 0

    def _food_ema_position(self, instrument):
        prices = np.asarray(self.data[instrument], dtype=float)

        if len(prices) < 2:
            return 0

        votes = 0

        for alpha in self.FOOD_EMA_ALPHAS:
            level = prices[0]

            # Construct the EMA through yesterday.
            for historical_price in prices[1:-1]:
                level = alpha * historical_price + (1.0 - alpha) * level

            # Compare today's price with yesterday's EMA.
            votes += int(np.sign(prices[-1] - level))

        limit = self.positionLimits[instrument]

        if votes > 0:
            return limit

        if votes < 0:
            return -limit

        return 0

    def _sausage_sizzle_position(self):
        bread = np.asarray(self.data["Bread"], dtype=float)
        sausage = np.asarray(self.data["Sausage"], dtype=float)
        sizzle = np.asarray(
            self.data["Sausage Sizzle"],
            dtype=float,
        )

        day = len(sizzle) - 1
        limit = self.positionLimits["Sausage Sizzle"]

        # Component changes are unavailable on day zero.
        if day == 0:
            return 0

        current_features = np.array(
            [
                1.0,
                bread[day] - bread[day - 1],
                sausage[day] - sausage[day - 1],
            ]
        )

        if day <= self.SIZZLE_WINDOW:
            # Use coefficients learned from Round 1 during startup.
            coefficients = self.SIZZLE_SEED
        else:
            # At day t, outcomes through t-1 are fully observable:
            # target[t-1] = Sizzle[t] - Sizzle[t-1].
            start = day - self.SIZZLE_WINDOW
            feature_rows = []
            targets = []

            for observation_day in range(start, day):
                feature_rows.append(
                    [
                        1.0,
                        bread[observation_day] - bread[observation_day - 1],
                        sausage[observation_day] - sausage[observation_day - 1],
                    ]
                )

                targets.append(
                    sizzle[observation_day + 1] - sizzle[observation_day]
                )

            design = np.asarray(feature_rows, dtype=float)
            targets = np.asarray(targets, dtype=float)

            coefficients, *_ = np.linalg.lstsq(
                design,
                targets,
                rcond=None,
            )

        predicted_change = float(current_features @ coefficients)

        if predicted_change > 0:
            return limit

        if predicted_change < 0:
            return -limit

        return 0

    def _menudash_position(self):
        menu = np.asarray(self.data["MenuDash"], dtype=float)
        sizzle = np.asarray(
            self.data["Sausage Sizzle"],
            dtype=float,
        )
        bread = np.asarray(self.data["Bread"], dtype=float)
        sausage = np.asarray(self.data["Sausage"], dtype=float)

        day = len(menu) - 1
        limit = self.positionLimits["MenuDash"]

        # Allow enough completed observations to estimate both relationships.
        if day < self.MENUDASH_WARMUP:
            return 0

        # Estimate how today's Bread and Sausage changes affect tomorrow's
        # Sizzle change. At day t, targets through Sizzle[t] are available.
        observation_days = np.arange(1, day)

        change_design = np.column_stack(
            [
                np.ones(len(observation_days)),
                bread[observation_days] - bread[observation_days - 1],
                sausage[observation_days] - sausage[observation_days - 1],
            ]
        )

        change_targets = sizzle[observation_days + 1] - sizzle[observation_days]

        ingredient_coefficients, *_ = np.linalg.lstsq(
            change_design,
            change_targets,
            rcond=None,
        )

        bread_loading = ingredient_coefficients[1]
        sausage_loading = ingredient_coefficients[2]

        # Sizzle[t] reflects ingredient and labour costs from day t-1.
        # Removing the ingredients gives an independent labour-cost sensor
        # aligned with MenuDash[0:t].
        labour_sensor = (
            sizzle[1 : day + 1]
            - bread_loading * bread[:day]
            - sausage_loading * sausage[:day]
        )

        menu_targets = menu[:day]
        latest_sensor = labour_sensor[-1]

        votes = 0

        for mapping_window in self.MENUDASH_MAPPING_WINDOWS:
            sensor_history = labour_sensor
            target_history = menu_targets

            if mapping_window is not None:
                sensor_history = sensor_history[-mapping_window:]
                target_history = target_history[-mapping_window:]

            mapping_design = np.column_stack(
                [
                    np.ones(len(sensor_history)),
                    sensor_history,
                ]
            )

            mapping_coefficients, *_ = np.linalg.lstsq(
                mapping_design,
                target_history,
                rcond=None,
            )

            fair_value = float(
                mapping_coefficients @ np.array([1.0, latest_sensor])
            )

            votes += int(np.sign(fair_value - menu[day]))

        if votes > 0:
            return limit

        if votes < 0:
            return -limit

        return 0

    def _thrifted_jeans_position(self):
        prices = np.asarray(self.data["Thrifted Jeans"], dtype=float)
        limit = self.positionLimits["Thrifted Jeans"]

        # Stay flat on day zero while no price movement has been observed.
        if len(prices) < 2:
            return 0

        transition = np.array(
            [
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        process_covariance = np.diag(
            [
                self.JEANS_Q_LEVEL,
                self.JEANS_Q_SLOPE,
            ]
        )
        observation_vector = np.array([1.0, 0.0])

        # State contains the estimated price level and daily slope.
        state = np.array([prices[0], 0.0])
        covariance = np.diag(
            [
                self.JEANS_OBSERVATION_VARIANCE,
                self.JEANS_INITIAL_SLOPE_VARIANCE,
            ]
        )

        # Reconstruct the causal filter using prices available through today.
        for day, price in enumerate(prices):
            if day > 0:
                state = transition @ state
                covariance = (
                    transition @ covariance @ transition.T + process_covariance
                )

            innovation = price - float(observation_vector @ state)
            innovation_variance = (
                float(observation_vector @ covariance @ observation_vector)
                + self.JEANS_OBSERVATION_VARIANCE
            )

            gain = (
                covariance
                @ observation_vector
                / max(innovation_variance, 1e-12)
            )

            state = state + gain * innovation
            covariance = covariance - np.outer(
                gain,
                observation_vector @ covariance,
            )

            # Limit small numerical asymmetry.
            covariance = (covariance + covariance.T) / 2.0

        estimated_slope = float(state[1])
        slope_uncertainty = np.sqrt(max(float(covariance[1, 1]), 0.0))

        # Retain the positive-drift position unless there is sufficiently
        # strong evidence of a persistent negative regime.
        if estimated_slope < -self.JEANS_SHORT_CONFIDENCE * slope_uncertainty:
            return -limit

        return limit
