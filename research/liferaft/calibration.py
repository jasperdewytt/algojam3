"""Chronological Year-1 calibration and conservative model selection.

Calibration operates on the same public price sequence an agent receives.
Each row is generated from a prefix ending before the predicted movement; no
marked-period prices or hidden simulator diagnostics enter these functions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import log
from statistics import median
from typing import Iterable, Mapping, Sequence

from .simulator import LiferaftConfig, MajorityOutcome
from .strategies import (
    CandidateSpec,
    Forecast,
    PublicLabel,
    candidate_specs,
    labels_from_prices,
)


@dataclass(frozen=True)
class WalkForwardBlock:
    """Score for one contiguous chronological validation block."""

    block_index: int
    first_target_day: int | None
    last_target_day: int | None
    pnl: int
    active_days: int
    hit_rate: float
    log_loss: float | None
    turnover: int


@dataclass(frozen=True)
class CandidateScore:
    """Walk-forward metrics used for selection and reporting."""

    name: str
    total_pnl: int
    block_pnl: tuple[int, ...]
    mean_block_pnl: float
    median_block_pnl: float
    lower_quartile_pnl: float
    worst_block_pnl: int
    hit_rate: float
    active_days: int
    observations: int
    log_loss: float | None
    turnover: int
    complexity: int
    positive_blocks: int
    stable: bool

    @property
    def pnl_per_active_day(self) -> float:
        return self.total_pnl / self.active_days if self.active_days else 0.0


@dataclass(frozen=True)
class SelectionResult:
    """A frozen Year-1 selection decision and all candidate evidence."""

    selected_name: str
    scores: Mapping[str, CandidateScore]
    warmup_days: int
    validation_blocks: int
    minimum_improvement: float
    source_days: int

    def summary(self) -> dict[str, object]:
        return {
            "selected_name": self.selected_name,
            "warmup_days": self.warmup_days,
            "validation_blocks": self.validation_blocks,
            "minimum_improvement": self.minimum_improvement,
            "source_days": self.source_days,
            "scores": {
                name: {
                    "total_pnl": score.total_pnl,
                    "mean_block_pnl": score.mean_block_pnl,
                    "hit_rate": score.hit_rate,
                    "active_days": score.active_days,
                    "turnover": score.turnover,
                    "stable": score.stable,
                }
                for name, score in self.scores.items()
            },
        }


def _quartile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return float(ordered[index])


def _label_for_move(move: int) -> PublicLabel:
    if move < 0:
        return MajorityOutcome.LONG
    if move > 0:
        return MajorityOutcome.SHORT
    return None


def _partition_indices(length: int, block_count: int) -> tuple[tuple[int, ...], ...]:
    if length <= 0:
        return ()
    count = max(1, min(block_count, length))
    blocks: list[tuple[int, ...]] = []
    for block_index in range(count):
        start = (length * block_index) // count
        end = (length * (block_index + 1)) // count
        blocks.append(tuple(range(start, end)))
    return tuple(blocks)


def _score_rows(
    rows: Sequence[tuple[int, int, Forecast, int, int, PublicLabel]],
    *,
    block_count: int,
    complexity: int,
) -> tuple[WalkForwardBlock, ...]:
    """Score ``(decision_day, action, forecast, move, pnl, label)`` rows."""

    blocks: list[WalkForwardBlock] = []
    for block_index, indices in enumerate(_partition_indices(len(rows), block_count)):
        block_rows = [rows[index] for index in indices]
        if not block_rows:
            continue
        active = [row for row in block_rows if row[1] != 0]
        known_active = [
            row
            for row in active
            if row[5] in (MajorityOutcome.LONG, MajorityOutcome.SHORT)
        ]
        hits = sum(
            (row[1] == -1 and row[5] is MajorityOutcome.LONG)
            or (row[1] == 1 and row[5] is MajorityOutcome.SHORT)
            for row in known_active
        )
        log_losses = [
            -log(max(row[2].probability(row[5]), 1e-9))
            for row in block_rows
            if row[5] in (MajorityOutcome.LONG, MajorityOutcome.SHORT)
        ]
        turnover = sum(
            row[1] != block_rows[index - 1][1]
            for index, row in enumerate(block_rows)
            if index > 0
        )
        blocks.append(
            WalkForwardBlock(
                block_index=block_index,
                first_target_day=block_rows[0][0],
                last_target_day=block_rows[-1][0],
                pnl=sum(row[4] for row in block_rows),
                active_days=len(active),
                hit_rate=(hits / len(known_active) if known_active else 0.0),
                log_loss=(sum(log_losses) / len(log_losses) if log_losses else None),
                turnover=turnover,
            )
        )
    return tuple(blocks)


def walk_forward_evaluate(
    prices: Sequence[int],
    *,
    boundary_day: int,
    config: LiferaftConfig | None = None,
    candidate_names: Sequence[str] | None = None,
    warmup_days: int = 30,
    validation_blocks: int = 3,
    min_expected_pnl: float = 1_000.0,
    min_confidence: float = 0.10,
) -> dict[str, CandidateScore]:
    """Evaluate fixed candidates on contiguous, prefix-only Year-1 rows.

    A prediction made on decision day ``d`` sees ``prices[:d+1]`` and scores
    only against the genuine movement into ``d+1``.  The artificial boundary
    movement is never in the target range because targets are strictly below
    ``boundary_day``.
    """

    if boundary_day < 0:
        raise ValueError("boundary_day cannot be negative")
    config = config or LiferaftConfig(
        total_days=max(boundary_day + 1, len(prices)),
        marked_boundary_day=boundary_day,
    )
    available = {spec.name: spec for spec in candidate_specs()}
    selected = tuple(candidate_names) if candidate_names is not None else tuple(available)
    missing = [name for name in selected if name not in available]
    if missing:
        raise ValueError(f"unknown calibration candidates: {missing}")

    # The prefix is intentionally copied and clipped.  A caller cannot make
    # Year-2 prices influence Year-1 selection by passing a full path here.
    year1_prices = tuple(prices[:boundary_day])
    first_decision = max(0, warmup_days)
    last_decision = min(boundary_day - 2, len(year1_prices) - 2)
    rows_by_candidate: dict[str, list[tuple[int, int, Forecast, int, int, PublicLabel]]] = {
        name: [] for name in selected
    }
    models = {name: available[name].model_factory() for name in selected}

    if last_decision >= first_decision:
        for decision_day in range(first_decision, last_decision + 1):
            history = year1_prices[: decision_day + 1]
            labels = labels_from_prices(history, marked_boundary_day=boundary_day)
            move = year1_prices[decision_day + 1] - year1_prices[decision_day]
            label = _label_for_move(move)
            for name in selected:
                spec = available[name]
                forecast = models[name].estimate(labels)
                action = spec.action(
                    forecast,
                    long_majority_move=config.long_majority_move,
                    short_majority_move=config.short_majority_move,
                    min_expected_pnl=min_expected_pnl,
                    min_confidence=min_confidence,
                )
                rows_by_candidate[name].append(
                    (decision_day, action, forecast, move, action * move, label)
                )

    scores: dict[str, CandidateScore] = {}
    for name in selected:
        spec = available[name]
        rows = rows_by_candidate[name]
        blocks = _score_rows(
            rows,
            block_count=validation_blocks,
            complexity=spec.complexity,
        )
        block_pnl = tuple(block.pnl for block in blocks)
        active = sum(row[1] != 0 for row in rows)
        known_active = [
            row
            for row in rows
            if row[1] != 0 and row[5] in (MajorityOutcome.LONG, MajorityOutcome.SHORT)
        ]
        hits = sum(
            (row[1] == -1 and row[5] is MajorityOutcome.LONG)
            or (row[1] == 1 and row[5] is MajorityOutcome.SHORT)
            for row in known_active
        )
        log_losses = [
            -log(max(row[2].probability(row[5]), 1e-9))
            for row in rows
            if row[5] in (MajorityOutcome.LONG, MajorityOutcome.SHORT)
        ]
        turnover = sum(
            row[1] != rows[index - 1][1]
            for index, row in enumerate(rows)
            if index > 0
        )
        scores[name] = CandidateScore(
            name=name,
            total_pnl=sum(row[4] for row in rows),
            block_pnl=block_pnl,
            mean_block_pnl=(sum(block_pnl) / len(block_pnl) if block_pnl else 0.0),
            median_block_pnl=(float(median(block_pnl)) if block_pnl else 0.0),
            lower_quartile_pnl=_quartile(block_pnl, 0.25),
            worst_block_pnl=min(block_pnl) if block_pnl else 0,
            hit_rate=(hits / len(known_active) if known_active else 0.0),
            active_days=active,
            observations=len(rows),
            log_loss=(sum(log_losses) / len(log_losses) if log_losses else None),
            turnover=turnover,
            complexity=spec.complexity,
            positive_blocks=sum(value > 0 for value in block_pnl),
            stable=False,
        )

    # Stability is assigned only relative to the flat benchmark and only on
    # validation blocks; full-Year-1 aggregate P&L is retained for reporting,
    # not used as the sole selection criterion.
    flat = scores.get("flat")
    if flat is not None:
        for name, score in tuple(scores.items()):
            if name == "flat":
                scores[name] = replace(score, stable=True)
                continue
            positive_required = max(1, (len(score.block_pnl) + 1) // 2)
            stable = (
                score.mean_block_pnl > flat.mean_block_pnl
                and score.mean_block_pnl - flat.mean_block_pnl >= 0
                and score.positive_blocks >= positive_required
                and score.active_days > 0
            )
            scores[name] = replace(score, stable=stable)
    return scores


def select_candidate_from_year1(
    prices: Sequence[int],
    *,
    boundary_day: int,
    config: LiferaftConfig | None = None,
    candidate_names: Sequence[str] | None = None,
    warmup_days: int = 30,
    validation_blocks: int = 3,
    minimum_improvement: float = 5_000.0,
) -> SelectionResult:
    """Select a candidate only when block evidence beats flat meaningfully."""

    requested_names = tuple(candidate_names) if candidate_names is not None else None
    if requested_names is not None and "flat" not in requested_names:
        # Flat is always a valid fallback, including for a restricted primary
        # candidate list used by the drift policy.
        requested_names = ("flat",) + requested_names
    scores = walk_forward_evaluate(
        prices,
        boundary_day=boundary_day,
        config=config,
        candidate_names=requested_names,
        warmup_days=warmup_days,
        validation_blocks=validation_blocks,
    )
    flat = scores.get("flat")
    selected_name = "flat"
    if flat is not None:
        eligible = [
            score
            for name, score in scores.items()
            if name != "flat"
            and score.stable
            and score.mean_block_pnl >= flat.mean_block_pnl + minimum_improvement
        ]
        if eligible:
            selected_name = max(
                eligible,
                key=lambda score: (
                    # Keep units consistent: total P&L is compared with
                    # total turnover. Turnover is only a stability preference
                    # here, not a simulated transaction cost.
                    score.total_pnl
                    - 1_000 * score.turnover
                    - 500 * score.complexity,
                    score.lower_quartile_pnl,
                    -score.complexity,
                ),
            ).name
    elif scores:
        selected_name = min(scores)
    return SelectionResult(
        selected_name=selected_name,
        scores=scores,
        warmup_days=warmup_days,
        validation_blocks=validation_blocks,
        minimum_improvement=minimum_improvement,
        source_days=min(boundary_day, len(prices)),
    )


def fit_ensemble_weights(
    scores: Mapping[str, CandidateScore],
    *,
    expert_names: Sequence[str] = (
        "last_majority_counter",
        "rolling_frequency",
        "markov",
        "periodic_replay",
    ),
) -> dict[str, float]:
    """Map walk-forward block evidence to small frozen non-negative weights.

    This is a fixed scoring rule, not an optimizer: each expert receives a
    base weight plus its positive mean-block improvement over flat, capped by
    the fixed transform below.  If no expert has positive evidence, the
    ensemble is flat.
    """

    flat_mean = scores.get("flat").mean_block_pnl if scores.get("flat") else 0.0
    raw: dict[str, float] = {}
    for name in expert_names:
        score = scores.get(name)
        if score is None or not score.stable:
            continue
        excess = max(0.0, score.mean_block_pnl - flat_mean)
        raw[name] = min(3.0, 0.25 + excess / 20_000.0)
    if not raw or max(raw.values()) <= 0.25:
        return {"flat": 1.0}
    # Keep an explicit flat component for uncertainty and normalize once at
    # the boundary.  These weights never adapt from marked-period outcomes.
    raw["flat"] = 0.25
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()}


def calibrate_and_select(
    prices: Sequence[int],
    config: LiferaftConfig,
    **kwargs: object,
) -> SelectionResult:
    """Convenience wrapper used by demos and external research scripts."""

    return select_candidate_from_year1(
        prices,
        boundary_day=config.marked_boundary_day,
        config=config,
        **kwargs,
    )


__all__ = [
    "CandidateScore",
    "SelectionResult",
    "WalkForwardBlock",
    "calibrate_and_select",
    "fit_ensemble_weights",
    "select_candidate_from_year1",
    "walk_forward_evaluate",
]
