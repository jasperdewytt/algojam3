from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import copy

import numpy as np
import pandas as pd
from scipy.optimize import minimize

class FairValue(ABC):
    def __init__(self, fair_value: float | None = None) -> None:
        self.fair_value: float | None = fair_value
        self.history: list[float] = []

    @abstractmethod
    def update(self, price: float) -> None: ...

    def update_window(self, prices: np.ndarray) -> None:
        for price in prices:
            self.update(price)

class EWMAFairValue(FairValue):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def update(self, price: float) -> None:
        if self.fair_value is None:
            self.fair_value = price
        else:
            self.fair_value += self.alpha * (price - self.fair_value)

        self.history.append(self.fair_value)

@dataclass(slots=True, kw_only=True)
class KalmanFilter:
    x: np.ndarray
    F: np.ndarray
    H: np.ndarray
    b: np.ndarray | None = field(default=None, init=False)

    P: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    _last_S: np.ndarray | None = field(default=None, init=False, repr=False)

    def _predict(self) -> None:
        self.x = self.F @ self.x

        if self.b is not None:
            self.x += self.b

        self.P = self.F @ self.P @ self.F.T + self.Q

    def _update(self, observation: np.ndarray) -> np.ndarray:
        innovation = observation - self.H @ self.x

        S = self.H @ self.P @ self.H.T + self.R
        self._last_S = S

        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(len(S)))
        self.x += K @ innovation

        I = np.eye(len(self.x))

        I_KH = I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return innovation.squeeze()

    def step(self, observation: np.ndarray) -> np.ndarray:
        self._predict()
        innovation = self._update(observation)

        return innovation

@dataclass(slots=True, kw_only=True)
class OU(KalmanFilter):
    fair_value_model: FairValue

    @property
    def stat_std(self) -> float:
        q = self.Q.item()
        phi = self.F.item()

        return float(np.sqrt(q / (1 - phi**2)))

    @classmethod
    def from_prices(
            cls, prices: np.ndarray, ou_trust: float, 
            fair_value_model: FairValue) -> OU:

        prices = np.asarray(prices, dtype=float)
        fair_values = np.asarray(fair_value_model.history, dtype=float)

        residuals = prices - fair_values
        residuals = residuals[np.isfinite(residuals)]

        if residuals.size < 5:
            raise ValueError("Need at least 5 observations.")

        r_prev, r_curr = residuals[:-1], residuals[1:]

        X = np.column_stack([np.ones_like(r_prev), r_prev])
        c, phi = np.linalg.lstsq(X, r_curr, rcond=None)[0]

        phi = float(np.clip(phi, 0.01, 0.999))

        state_residuals = r_curr - (c + phi * r_prev)
        q = float(np.var(state_residuals, ddof=1))

        if not np.isfinite(q) or q <= 1e-12:
            q = max(float(np.var(residuals)) * 1e-4, 1e-8)

        stat_var = q / (1.0 - phi**2)

        p = np.clip(ou_trust, 0.0, 100.0) / 100.0
        trust = 0.1 + (10.0 - 0.1) * p
        r = q * trust

        return cls(
            fair_value_model=fair_value_model,
            x=np.array([residuals[-1]], dtype=float),
            F=np.array([[phi]], dtype=float),
            H=np.array([[1.0]], dtype=float),
            P=np.array([[stat_var]], dtype=float),
            Q=np.array([[q]], dtype=float),
            R=np.array([[r]], dtype=float),
        )

    def update(self, price: float) -> float:
        self.fair_value_model.update(price)
        fair = self.fair_value_model.fair_value or price
        
        residual = price - fair
        innovation = self.step(np.array([residual]))

        if self._last_S is None:
            raise ValueError("Innovation covariance not available.")

        return float(innovation) / self.stat_std

class TrendModel(ABC):
    @abstractmethod
    def update(self, price: float, forecast_horizon: int) -> float: ...

class LocalLinearTrend(KalmanFilter, TrendModel):
    def forecast(self, horizon: int) -> float:
        return float(self.x[0]) + horizon * float(self.x[1])

    def forecast_std(self, horizon: int) -> float:
        P = self.P.copy()

        for _ in range(horizon):
            P = self.F @ P @ self.F.T + self.Q

        variance = (self.H @ P @ self.H.T).item()
        
        return float(np.sqrt(variance))

    @classmethod
    def _build(
            cls, level: float, trend: float, q_level: float, q_trend: float,
            r: float, p_level: float, p_trend) -> LocalLinearTrend:
        
        return cls(
            x=np.array([level, trend]),
            F=np.array([[1.0, 1.0], [0.0, 1.0]]),
            H=np.array([[1.0, 0.0]]),
            P=np.array([[p_level, 0.0], [0.0, p_trend]]),
            Q=np.array([[q_level, 0.0], [0.0, q_trend]]),
            R=np.array([[r]]),
        )

    @classmethod
    def from_prices(
            cls, prices: np.ndarray, trend_trust: float) -> LocalLinearTrend:

        y = prices[np.isfinite(prices)].astype(float)
        y_var = float(np.var(y))

        dy = np.diff(y)
        dy_var = float(np.var(dy))

        level = float(y[0])
        trend = float(np.mean(dy[:5])) 

        log_bounds = [
            (np.log(y_var * 1e-5), np.log(y_var * 1e-1)),
            (np.log(y_var * 1e-4), np.log(y_var * 1e2)),
            (np.log(y_var * 1e-3), np.log(y_var * 1e2)),
        ]

        def negative_log_likelihood(log_params: np.ndarray) -> float:
            q_level, q_trend, r = np.exp(log_params)

            for val, (lower, upper) in zip(log_params, log_bounds):
                if val < lower or val > upper:
                    return 1e10

            kalman_filter = cls._build(
                level, trend, q_level, q_trend, r, y_var, dy_var
            )

            log_likelihood = 0.0

            for price in y:
                innovation = kalman_filter.step(np.array([price]))

                if kalman_filter._last_S is None:
                    raise ValueError("Kalman Filter S is None.")
            
                S = kalman_filter._last_S.item()

                if S <= 0 or not np.isfinite(S):
                    return 1e10

                log_likelihood -= 0.5 * (
                    np.log(2 * np.pi * S) + float(innovation)**2 / S
                )
                
            return -log_likelihood

        x0 = np.log([y_var * 0.01, y_var * 0.001, y_var * 0.1])

        result = minimize(
            negative_log_likelihood, x0, 
            method="Nelder-Mead", bounds=log_bounds
        )

        q_level, q_trend, r = np.exp(result.x)

        fraction = max(0, min(100, trend_trust)) / 100.0
        trust = 1e8 if fraction >= 1.0 else 0.1 + (10.0 - 0.1) * fraction

        p_level = y_var * trust
        p_trend = dy_var * trust

        kalman_filter = cls._build(
            level, trend, q_level, q_trend, r, p_level, p_trend
        )

        for price in y:
            kalman_filter.step(np.array([price]))

        return kalman_filter

    def update(self, price: float, forecast_horizon: int) -> float:
        self.step(np.array([price]))

        trend_std = float(np.sqrt(self.P[1, 1]))
        
        if trend_std <= 0:
            return 0.0
        
        return float(self.x[1]) / trend_std

class UnifiedModel:
    def __init__(
            self, trend_model: TrendModel | None, 
            revert_model: OU | None) -> None:
        
        self.trend_model = trend_model
        self.revert_model = revert_model

    def update(self, price: float, forecast_horizon: int) -> tuple[float, float]:
        trend_score = 0.0
        revert_z = 0.0

        if self.trend_model is not None:
            trend_score = self.trend_model.update(price, forecast_horizon)

        if self.revert_model is not None:
            revert_z = self.revert_model.update(price)

        return trend_score, revert_z
  
class Algorithm():
    def __init__(self, positions):
        self.data = {}  
        self.positionLimits = {}  
        self.day = 0     
        self.positions = positions 

        self.models = {}
        self.price_data = {}

    def trade(
            self, instrument: str, calib_window: int = 0, 
            trend: bool = False, trend_model: TrendModel | None = None,
            trend_threshold: float = 1.0, forecast_horizon: int = 1, 
            trend_trust: float = 50.0, mean_revert: bool = False, 
            revert_fair_value: FairValue | None = None, 
            revert_threshold: float = 1.0, ou_trust: float = 50.0) -> int:

        if not (trend or mean_revert):
            raise ValueError(
                "At least one of trend or mean_revert must be True."
            )

        if self.day < calib_window:
            return 0

        if instrument not in self.models:
            prices = self.get_seen_prices(instrument, calib_window)

            if trend and trend_model is None:
                trend_model = LocalLinearTrend.from_prices(
                    prices, trend_trust
                )

            revert_model = None

            if mean_revert:
                if revert_fair_value is None:
                    raise ValueError(
                        "fair_value_model required for mean reversion."
                    )

                fair_value_model = copy.deepcopy(revert_fair_value)
                fair_value_model.update_window(prices)

                revert_model = OU.from_prices(
                    prices, ou_trust, fair_value_model
                )

            self.models[instrument] = UnifiedModel(trend_model, revert_model)

            return 0

        price = self.get_current_price(instrument)

        model: UnifiedModel = self.models[instrument]
        trend_score, ou_z = model.update(price, forecast_horizon)

        current_pos = self.positions[instrument]
        max_pos = self.positionLimits[instrument]

        if trend and abs(trend_score) >= trend_threshold: 
            return max_pos if trend_score > 0 else -max_pos

        elif mean_revert:
            if current_pos == 0:
                if ou_z >= revert_threshold:
                    return -max_pos
                
                elif ou_z <= -revert_threshold:
                    return max_pos
                
            elif current_pos * ou_z > 0:
                if abs(ou_z) >= revert_threshold:
                    return -current_pos
                
                return 0

        return current_pos

    def load_data(self, instrument: str) -> np.ndarray:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base_dir, "data", f"{instrument}_price_history.csv")
        return pd.read_csv(filepath, index_col=0)["Price"].to_numpy()

    def get_seen_prices(self, instrument: str, window: int = 0) -> np.ndarray:
        if window > 0:
            return np.array(self.data[instrument][-window:])

        prices = self.price_data.get(instrument, self.load_data(instrument))
        
        if window == 0:
            return prices

        return prices[window:]

    def get_current_price(self, instrument: str) -> float:
        return self.data[instrument][-1]
    
    def get_positions(self):
        position_limits = self.positionLimits

        desired_positions = {}

        for instrument, _ in position_limits.items():
            desired_positions[instrument] = 0

        desired_positions["Thrifted Jeans"] = self.trade(
            instrument="Thrifted Jeans", calib_window=0, trend=True, 
            trend_threshold=0.8, trend_trust=50.0,
            mean_revert=True, revert_fair_value=EWMAFairValue(alpha=0.9),
            revert_threshold=0.1, ou_trust=50.0,
        )

        return desired_positions
