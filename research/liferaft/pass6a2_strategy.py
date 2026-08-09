"""Pass 6A.2: Pass 6A.1 with a fixed mixture anytime gate.

The trading policy and Fixed-Share master are inherited unchanged.  Only the
statistical gate differs: five fixed betting fractions are averaged in
e-value space, with all arithmetic maintained in log space.
"""

from __future__ import annotations

from math import exp, isfinite, log
from typing import Sequence

from .pass6a1_strategy import (
    ADJUSTED_REWARD_BOUND,
    ANYTIME_ALPHA,
    CorrectedFixedShareMaster,
    Pass6A1AnytimeStrategy,
    make_pass6a1_strategy,
    majority_from_counts,
)


LAMBDAS = (0.02, 0.05, 0.10, 0.20, 0.50)


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + log(sum(exp(value - maximum) for value in values))


class MixtureAnytimeGate:
    """Equal-weight mixture of frozen fixed-fraction e-processes."""

    def __init__(
        self,
        *,
        alpha: float = ANYTIME_ALPHA,
        lambdas: Sequence[float] = LAMBDAS,
        reward_bound: int = ADJUSTED_REWARD_BOUND,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if tuple(lambdas) != LAMBDAS:
            raise ValueError("Pass 6A.2 lambdas are frozen")
        if not lambdas or any(not 0 < value <= 1 for value in lambdas):
            raise ValueError("lambdas must lie in (0, 1]")
        if reward_bound <= 0:
            raise ValueError("reward bound must be positive")
        self.alpha = alpha
        self.lambdas = tuple(float(value) for value in lambdas)
        self.reward_bound = reward_bound
        self.log_component_e_values = [0.0 for _ in self.lambdas]
        self.authorized = False
        self.activation_day: int | None = None
        self.activation_lambda: float | None = None
        self.scoreable_count = 0
        self.unavailable_count = 0

    @property
    def log_threshold(self) -> float:
        return log(1.0 / self.alpha)

    @property
    def threshold(self) -> float:
        return 1.0 / self.alpha

    @property
    def log_mixture_e_value(self) -> float:
        return _logsumexp(self.log_component_e_values) - log(len(self.lambdas))

    @property
    def mixture_e_value(self) -> float:
        return exp(self.log_mixture_e_value)

    @property
    def active(self) -> bool:
        return self.authorized

    def observe(self, *, day: int, reward: int, scoreable: bool) -> None:
        if not scoreable:
            self.unavailable_count += 1
            return
        if not -self.reward_bound <= reward <= self.reward_bound:
            raise ValueError("adjusted reward outside the frozen bound")
        self.scoreable_count += 1
        for index, lambda_value in enumerate(self.lambdas):
            factor = 1.0 + lambda_value * reward / self.reward_bound
            if factor <= 0 or not isfinite(factor):
                raise ValueError("invalid mixture e-process factor")
            self.log_component_e_values[index] += log(factor)
        if not self.authorized and self.log_mixture_e_value >= self.log_threshold:
            self.authorized = True
            self.activation_day = day
            best_index = max(
                range(len(self.lambdas)),
                key=lambda index: self.log_component_e_values[index],
            )
            self.activation_lambda = self.lambdas[best_index]

    def diagnostics(self) -> dict[str, object]:
        component_values = [exp(value) for value in self.log_component_e_values]
        return {
            "alpha": self.alpha,
            "lambdas": list(self.lambdas),
            "reward_bound": self.reward_bound,
            "threshold": self.threshold,
            "log_threshold": self.log_threshold,
            "log_mixture_e_value": self.log_mixture_e_value,
            "mixture_e_value": self.mixture_e_value,
            "final_mixture_e_value": self.mixture_e_value,
            "final_component_e_values": component_values,
            "component_e_values": component_values,
            "authorized": self.authorized,
            "activation_day": self.activation_day,
            "activation_lambda": self.activation_lambda,
            "lambda_most_contributing_at_activation": self.activation_lambda,
            "scoreable_count": self.scoreable_count,
            "unavailable_count": self.unavailable_count,
        }


class Pass6A2AnytimeStrategy(Pass6A1AnytimeStrategy):
    """Pass 6A.1 strategy mechanics with only the mixture gate replaced."""

    name = "pass6a2_mixture_anytime_fixed_share"

    def __init__(self, *, other_portfolio_exposure=0.0, alpha: float = ANYTIME_ALPHA) -> None:
        # The superclass owns all predictor, paper-action, risk, exposure and
        # idempotence mechanics.  Replace its gate before the first decision.
        super().__init__(other_portfolio_exposure=other_portfolio_exposure, alpha=alpha)
        self.gate = MixtureAnytimeGate(alpha=alpha)


def make_pass6a2_strategy(*, other_portfolio_exposure=0.0, alpha: float = ANYTIME_ALPHA):
    return Pass6A2AnytimeStrategy(
        other_portfolio_exposure=other_portfolio_exposure,
        alpha=alpha,
    )


def run_self_checks() -> None:
    single = MixtureAnytimeGate(alpha=0.10, lambdas=(0.02, 0.05, 0.10, 0.20, 0.50))
    component = MixtureAnytimeGate(alpha=0.10, lambdas=(0.02, 0.05, 0.10, 0.20, 0.50))
    # A single-component instance is a useful mathematical fixture; the
    # production gate remains frozen to the five declared lambdas.
    single.lambdas = (0.10,)
    single.log_component_e_values = [0.0]
    component.observe(day=1, reward=1_000, scoreable=True)
    single.observe(day=1, reward=1_000, scoreable=True)
    assert abs(single.mixture_e_value - exp(single.log_component_e_values[0])) < 1e-15
    assert abs(MixtureAnytimeGate(alpha=0.10).threshold - 10.0) < 1e-15

    # Opponents tie; a long paper action changes the opponents-only status.
    baseline = majority_from_counts(0, 0)
    paper = majority_from_counts(1, 0)
    assert baseline is not paper

    def pooled(rows: list[tuple[int, int]]) -> float:
        return sum(pivotal for pivotal, _ in rows) / sum(total for _, total in rows)

    assert abs(pooled([(1, 2), (2, 3)]) - 3 / 5) < 1e-15


__all__ = [
    "LAMBDAS",
    "MixtureAnytimeGate",
    "Pass6A2AnytimeStrategy",
    "make_pass6a2_strategy",
    "run_self_checks",
]
