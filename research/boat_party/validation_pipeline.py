"""Second-stage, research-only validation pipeline for Boat Party Ticket.

This file deliberately keeps the preregistered candidate frozen.  It separates
same-path template reconstruction, disjoint semester transfer, and independent
synthetic generators so their P&L is never pooled into one score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

try:
    from . import analysis as a
except ImportError:  # notebook adds research/boat_party directly to sys.path
    import analysis as a


S2_START = 161
S2_SEMESTER_END = 326
YEAR_END = 365
S1_EVAL_START = 15
S1_EVAL_END = 171


def _metric_row(
    result: Mapping[str, object],
    *,
    validation_class: str,
    direction: str,
    window: str,
    template_source: str,
    evaluated_segment: str,
    uses_evaluated_prices: bool,
    chronologically_deployable: bool,
) -> dict[str, object]:
    return {
        "validation_class": validation_class,
        "direction": direction,
        "window": window,
        "model": result["model"],
        "template_source": template_source,
        "evaluated_segment": evaluated_segment,
        "uses_evaluated_prices": int(uses_evaluated_prices),
        "chronologically_deployable": int(chronologically_deployable),
        "pnl": result["pnl"],
        "sharpe": result["sharpe"],
        "active_hit_rate": result["active_hit_rate"],
        "active_days": result["active_days"],
        "max_drawdown": result["max_drawdown"],
        "max_capital": result["max_capital"],
        "pnl_per_max_capital": result["pnl_per_max_capital"],
        "budget_violations": result["budget_violations"],
        "integral_positions": result["integral_positions"],
        "within_limit": result["within_limit"],
    }


def _window_definitions(start: int, semester_end: int) -> dict[str, tuple[int, int]]:
    return {
        "first_10_days": (start, start + 10),
        "days_10_to_30": (start + 10, start + 30),
        "days_30_to_60": (start + 30, start + 60),
        "remaining_semester": (start + 60, semester_end),
        "whole_semester": (start, semester_end),
        "year_end_including_summer": (start, YEAR_END),
    }


def _warp_arbitrary_template(template: np.ndarray) -> np.ndarray:
    source_index = a.calendar_warp_indices(2026, 2027, len(template))
    grid = np.arange(len(template), dtype=float)
    return np.interp(source_index, grid, template)


def _assert_position_validity(prices: np.ndarray, positions: np.ndarray) -> None:
    result = a.backtest(prices, positions)
    assert result["integral_positions"] == 1
    assert result["within_limit"] == 1
    assert result["budget_violations"] == 0


def _s2_transfer_models(
    prices: np.ndarray,
    fixed_transfer: np.ndarray,
    academic_transfer: np.ndarray,
) -> dict[str, tuple[np.ndarray, str]]:
    calendar = a.calendar_schedule_positions(2026)
    summer = a.summer_policy_positions(prices, int(a.landmark_days(2026)["summer"]), mode="fixed")
    calendar_plus_summer = np.where(summer != 0, summer, calendar).astype(int)
    constant = a.baseline_mean_reversion_positions(prices, start=S2_START, end=YEAR_END, mode="fixed")
    online = a.baseline_mean_reversion_positions(prices, start=S2_START, end=YEAR_END, mode="online_mean")
    baseline_template = np.full(len(prices), 45.0, dtype=float)
    return {
        "constant_45_mean_reversion": (constant, "fixed AUD 45 equilibrium; no adaptation"),
        "online_baseline_mean": (online, "online EWMA baseline only; no seasonal template"),
        "broad_calendar_schedule": (calendar, "official 2026 academic landmarks and broad windows"),
        "fixed_transfer_no_rls": (a.template_positions(fixed_transfer), "S1 prices only; local 7-day transfer; no RLS"),
        "baseline_only_rls": (a.rls_positions_window(prices, baseline_template, S2_START, YEAR_END), "constant AUD 45 template; global RLS intercept only"),
        "fixed_transfer_rls": (a.rls_positions_window(prices, fixed_transfer, S2_START, YEAR_END), "S1 prices only; fixed-index local 7-day transfer; frozen RLS"),
        "academic_transfer_rls": (a.rls_positions_window(prices, academic_transfer, S2_START, YEAR_END), "S1 prices only; official event-phase local 7-day transfer; frozen RLS"),
        "dual_transfer_agreement": (a.rls_dual_template_agreement_window(prices, fixed_transfer, academic_transfer, S2_START, YEAR_END), "agreement of independent fixed-index and academic-phase S1 transfers"),
        "calendar_plus_summer": (calendar_plus_summer, "official broad calendar schedule plus fixed AUD 45 summer policy at official summer date"),
    }


def _reverse_models(
    prices: np.ndarray,
    fixed_transfer: np.ndarray,
    academic_transfer: np.ndarray,
) -> dict[str, tuple[np.ndarray, str]]:
    calendar = a.calendar_schedule_positions(2026)
    baseline_template = np.full(len(prices), 45.0, dtype=float)
    return {
        "broad_calendar_schedule": (calendar, "official 2026 academic landmarks and broad windows"),
        "fixed_transfer_no_rls": (a.template_positions(fixed_transfer), "S2 prices only; fixed-index transfer"),
        "baseline_only_rls": (a.rls_positions_window(prices, baseline_template, S1_EVAL_START, YEAR_END), "constant AUD 45 template; global RLS intercept only"),
        "fixed_transfer_rls": (a.rls_positions_window(prices, fixed_transfer, S1_EVAL_START, YEAR_END), "S2 prices only; fixed-index local 7-day transfer"),
        "academic_transfer_rls": (a.rls_positions_window(prices, academic_transfer, S1_EVAL_START, YEAR_END), "S2 prices only; official event-phase local 7-day transfer"),
        "dual_transfer_agreement": (a.rls_dual_template_agreement_window(prices, fixed_transfer, academic_transfer, S1_EVAL_START, YEAR_END), "agreement of independently transferred S2 shapes"),
    }


def _placebo_templates(fixed_template: np.ndarray) -> dict[str, np.ndarray]:
    broad_calendar = 45.0 + 0.55 * np.cumsum(a.fixed_schedule_positions().astype(float) / a.POSITION_LIMIT)
    constant = np.full(len(fixed_template), 45.0, dtype=float)
    secondary_removed = fixed_template.copy()
    for wave in a.WAVE_SEGMENTS:
        if wave["kind"] == "small":
            secondary_removed[int(wave["start"]) : int(wave["end"])] = 45.0
    local_easter = {
        f"easter_section_displacement_{displacement:+d}": a.shifted_template_section(fixed_template, 80, 125, -displacement)
        for displacement in [-14, -7, -3, -1, 1, 3, 7, 14]
    }
    templates: dict[str, np.ndarray] = {
        "correct_fixed_7day": fixed_template,
        "constant_45": constant,
        "linear_trend_prior": np.linspace(44.0, 46.0, len(fixed_template)),
        "broad_piecewise_calendar_shape": broad_calendar,
        "reversed_template": fixed_template[::-1].copy(),
        "phase_shuffled_template": a.phase_shuffled_template(fixed_template, seed=a.SEED + 17),
        "secondary_waves_removed": secondary_removed,
        "large_small_waves_exchanged": a.wave_role_exchanged_template(fixed_template),
    }
    for displacement in [-14, -7, -3, -1, 1, 3, 7, 14]:
        # The label is the displacement of the seasonal feature; the helper's
        # interpolation sign is reversed so this matches the prior audit.
        templates[f"global_displacement_{displacement:+d}"] = a.shifted_template(fixed_template, -displacement)
    templates.update(local_easter)
    return templates


def _agreement_row(
    path_name: str,
    metadata: Mapping[str, object],
    prices: np.ndarray,
    fixed_positions: np.ndarray,
    warped_positions: np.ndarray,
    dual_positions: np.ndarray,
) -> dict[str, object]:
    move = np.diff(prices)
    fixed_daily = fixed_positions[:-1].astype(float) * move
    warped_daily = warped_positions[:-1].astype(float) * move
    dual_daily = dual_positions[:-1].astype(float) * move
    disagreement = np.sign(fixed_positions[:-1]) != np.sign(warped_positions[:-1])
    return {
        "path_name": path_name,
        "generator_family": metadata["generator_family"],
        "timing_mode": metadata["timing_mode"],
        "disagreement_days": int(np.sum(disagreement)),
        "dual_active_days": int(np.sum(dual_positions[:-1] != 0)),
        "fixed_pnl_on_disagreement": float(np.sum(fixed_daily[disagreement])),
        "warped_pnl_on_disagreement": float(np.sum(warped_daily[disagreement])),
        "dual_pnl_on_disagreement": float(np.sum(dual_daily[disagreement])),
        "pnl_avoided_vs_fixed_on_disagreement": float(-np.sum(fixed_daily[disagreement])),
        "pnl_avoided_vs_warped_on_disagreement": float(-np.sum(warped_daily[disagreement])),
    }


def _summer_shift_bucket(value: float) -> str:
    if value <= -1.0:
        return "minus_1_to_minus_2"
    if value >= 1.0:
        return "plus_1_to_plus_2"
    return "near_zero"


def _summary_by_model(detail: pd.DataFrame, group_columns: list[str] | None = None) -> pd.DataFrame:
    group_columns = group_columns or ["model"]
    rows: list[dict[str, object]] = []
    for keys, group in detail.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_paths": int(len(group)),
                "median_pnl": float(group["pnl"].median()),
                "p10_pnl": float(group["pnl"].quantile(0.10)),
                "worst_pnl": float(group["pnl"].min()),
                "positive_path_rate": float(np.mean(group["pnl"] > 0)),
                "median_max_drawdown": float(group["max_drawdown"].median()),
                "median_active_days": float(group["active_days"].median()),
                "median_pnl_per_max_capital": float(group["pnl_per_max_capital"].median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("median_pnl", ascending=False).reset_index(drop=True)


def _save_stage2_figures(
    result_dir: Path,
    figure_dir: Path,
    prices: np.ndarray,
    holdout_positions: Mapping[str, np.ndarray],
    independent_detail: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for name, positions in holdout_positions.items():
        daily = positions[:-1].astype(float) * np.diff(prices)
        curve = np.cumsum(daily)
        ax.plot(np.arange(len(curve)), curve, linewidth=1.4, label=name)
    ax.axvline(S2_START, color="#555555", linewidth=0.7, alpha=0.5)
    ax.axvline(S2_SEMESTER_END, color="#555555", linewidth=0.7, alpha=0.5)
    ax.set(xlabel="Round 1 day", ylabel="Cumulative P&L (AUD)", title="Disjoint S1-to-S2 validation: cumulative P&L")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "disjoint_validation_cumulative_pnl.png", dpi=160)
    plt.close(fig)

    models = ["broad_calendar_schedule", "fixed_template_rls", "warped_template_rls", "dual_template_agreement", "calendar_plus_summer"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    data = [independent_detail.loc[independent_detail["model"] == name, "pnl"].to_numpy() for name in models]
    ax.boxplot(data, tick_labels=models, showfliers=False)
    ax.axhline(0.0, color="#555555", linewidth=0.7)
    ax.set_ylabel("Independent-path P&L (AUD)")
    ax.set_title("Independent generator P&L distributions")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figure_dir / "independent_stress_pnl.png", dpi=160)
    plt.close(fig)


def run_stage2_validation(
    repo_root: str | Path | None = None,
    n_paths: int = 400,
    seed: int = a.SEED,
) -> dict[str, pd.DataFrame]:
    """Run and persist the preregistered second-stage validation audit."""

    root = a.find_repo_root(repo_root)
    prices = a.load_prices(root)["Price"].to_numpy(dtype=float)
    output_root = root / "research" / "boat_party"
    result_dir = output_root / "results"
    figure_dir = output_root / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Frozen candidate objects.  These are the only full-path templates used
    # by the independent synthetic stress; no parameters are selected there.
    fixed_full = a.template_from_prices(prices, window=7)
    warped_full = a.warped_template(prices, window=7, target_year=2027)

    # Disjoint chronological S1 -> S2 templates.  Both read only days 15:161.
    fixed_s2, source_fixed_s2 = a.transfer_template(
        prices,
        ["S1_large", "S1_small"],
        ["S2_large", "S2_small"],
        target_year=2026,
        target_mode="fixed",
        window=7,
    )
    academic_s2, source_academic_s2 = a.transfer_template(
        prices,
        ["S1_large", "S1_small"],
        ["S2_large", "S2_small"],
        target_year=2026,
        target_mode="academic",
        window=7,
    )

    # Leakage assertions: alter every held-out price and ensure the templates
    # and source provenance remain unchanged.
    altered = prices.copy()
    altered[S2_START:] += 17.0
    altered_fixed_s2, altered_source_fixed = a.transfer_template(altered, ["S1_large", "S1_small"], ["S2_large", "S2_small"], target_year=2026, target_mode="fixed", window=7)
    altered_academic_s2, altered_source_academic = a.transfer_template(altered, ["S1_large", "S1_small"], ["S2_large", "S2_small"], target_year=2026, target_mode="academic", window=7)
    assert np.allclose(fixed_s2, altered_fixed_s2)
    assert np.allclose(academic_s2, altered_academic_s2)
    assert source_fixed_s2 == altered_source_fixed == set(range(15, 161))
    assert source_academic_s2 == altered_source_academic == set(range(15, 161))
    assert source_fixed_s2.isdisjoint(set(range(S2_START, YEAR_END)))
    assert source_academic_s2.isdisjoint(set(range(S2_START, YEAR_END)))
    assert a.causal_rls_prefix_check(prices, fixed_s2, S2_START, YEAR_END, cut_day=200)
    # Local smoothing is deliberately applied to each source interval, not to
    # a full-year array that could cross the held-out boundary.
    source_wave = next(wave for wave in a.WAVE_SEGMENTS if wave["wave"] == "S1_large")
    assert np.allclose(a._local_wave_curve(prices, source_wave, 7), a._local_wave_curve(altered, source_wave, 7))

    provenance_rows = [
        {
            "validation_class": "chronological_disjoint",
            "template_name": "s2_fixed_transfer",
            "template_source": "Round 1 S1 prices days 15:161; local 7-day smoothing; fixed S2 indices",
            "source_price_start": 15,
            "source_price_end_exclusive": 161,
            "evaluated_segment": "S2 days 161:365",
            "evaluated_start": S2_START,
            "evaluated_end_exclusive": YEAR_END,
            "uses_evaluated_prices": 0,
            "normalizes_with_evaluated_amplitude": 0,
            "centered_smoothing_crosses_evaluation_boundary": 0,
            "calendar_landmarks_price_dependent": 0,
            "chronologically_deployable": 1,
        },
        {
            "validation_class": "chronological_disjoint",
            "template_name": "s2_academic_transfer",
            "template_source": "Round 1 S1 prices days 15:161; local 7-day smoothing; official 2026 S2 event intervals",
            "source_price_start": 15,
            "source_price_end_exclusive": 161,
            "evaluated_segment": "S2 days 161:365",
            "evaluated_start": S2_START,
            "evaluated_end_exclusive": YEAR_END,
            "uses_evaluated_prices": 0,
            "normalizes_with_evaluated_amplitude": 0,
            "centered_smoothing_crosses_evaluation_boundary": 0,
            "calendar_landmarks_price_dependent": 0,
            "chronologically_deployable": 1,
        },
        {
            "validation_class": "noncausal_structural_transfer",
            "template_name": "s1_reverse_transfer",
            "template_source": "Round 1 S2 prices days 161:302; used to diagnose structural transfer to S1",
            "source_price_start": 161,
            "source_price_end_exclusive": 302,
            "evaluated_segment": "S1 days 15:171",
            "evaluated_start": S1_EVAL_START,
            "evaluated_end_exclusive": S1_EVAL_END,
            "uses_evaluated_prices": 0,
            "normalizes_with_evaluated_amplitude": 0,
            "centered_smoothing_crosses_evaluation_boundary": 0,
            "calendar_landmarks_price_dependent": 0,
            "chronologically_deployable": 0,
        },
    ]

    leakage_rows = [
        {"check": "held_out_prices_never_enter_s2_fixed_template", "passed": int(np.allclose(fixed_s2, altered_fixed_s2)), "evidence": "template unchanged after +17 AUD perturbation to all S2 prices"},
        {"check": "held_out_prices_never_enter_s2_academic_template", "passed": int(np.allclose(academic_s2, altered_academic_s2)), "evidence": "template unchanged after +17 AUD perturbation to all S2 prices"},
        {"check": "source_indices_disjoint_from_s2_evaluation", "passed": int(source_fixed_s2.isdisjoint(set(range(S2_START, YEAR_END)))), "evidence": "source index set is 15:161; evaluation is 161:365"},
        {"check": "local_centered_smoothing_does_not_cross_boundary", "passed": int(np.allclose(a._local_wave_curve(prices, source_wave, 7), a._local_wave_curve(altered, source_wave, 7))), "evidence": "source interval is smoothed as a standalone array"},
        {"check": "held_out_amplitude_not_used_for_normalisation", "passed": 1, "evidence": "transfer uses source curve levels; no held-out min/max or scale"},
        {"check": "calendar_landmarks_are_price_independent", "passed": 1, "evidence": "landmark_days() reads encoded official dates only"},
        {"check": "rls_uses_only_prices_through_decision_day", "passed": int(a.causal_rls_prefix_check(prices, fixed_s2, S2_START, YEAR_END, cut_day=200)), "evidence": "future prices after day 200 do not change earlier decisions"},
    ]
    leakage_frame = pd.DataFrame(leakage_rows)
    assert bool(np.all(leakage_frame["passed"] == 1))

    # Chronological S1 -> S2 performance, with all candidate parameters frozen.
    heldout_rows: list[dict[str, object]] = []
    s2_models = _s2_transfer_models(prices, fixed_s2, academic_s2)
    for window_name, (start, end) in _window_definitions(S2_START, S2_SEMESTER_END).items():
        for model_name, (positions, source) in s2_models.items():
            result = a.backtest_window(prices, positions, start, end, label=model_name)
            heldout_rows.append(
                _metric_row(
                    result,
                    validation_class="chronological_disjoint",
                    direction="S1_to_S2",
                    window=window_name,
                    template_source=source,
                    evaluated_segment=f"days {start}:{end}",
                    uses_evaluated_prices=False,
                    chronologically_deployable=True,
                )
            )

    # Reverse transfer is explicitly noncausal because it trains on later S2.
    fixed_s1, _ = a.transfer_template(prices, ["S2_large", "S2_small"], ["S1_large", "S1_small"], target_year=2026, target_mode="fixed", window=7)
    academic_s1, _ = a.transfer_template(prices, ["S2_large", "S2_small"], ["S1_large", "S1_small"], target_year=2026, target_mode="academic", window=7)
    reverse_rows: list[dict[str, object]] = []
    reverse_models = _reverse_models(prices, fixed_s1, academic_s1)
    for window_name, (start, end) in _window_definitions(S1_EVAL_START, S1_EVAL_END).items():
        for model_name, (positions, source) in reverse_models.items():
            result = a.backtest_window(prices, positions, start, end, label=model_name)
            reverse_rows.append(
                _metric_row(
                    result,
                    validation_class="noncausal_structural_transfer",
                    direction="S2_to_S1_reverse_diagnostic",
                    window=window_name,
                    template_source=source,
                    evaluated_segment=f"days {start}:{end}",
                    uses_evaluated_prices=False,
                    chronologically_deployable=False,
                )
            )
    heldout_frame = pd.DataFrame(heldout_rows + reverse_rows)

    # Actual recommended RLS logic on each held-out wave, not the previous
    # median-shape slope diagnostic.
    wave_rows: list[dict[str, object]] = []
    for held in a.WAVE_SEGMENTS:
        held_name = str(held["wave"])
        start, end = int(held["start"]), int(held["end"])
        template, source_indices, fit_waves = a.held_out_wave_template(prices, held_name, window=7, target_mode="fixed", target_year=2026)
        positions = {
            "broad_schedule": a.fixed_schedule_positions(),
            "held_wave_transfer_no_rls": a.template_positions(template),
            "held_wave_transfer_rls": a.rls_positions_window(prices, template, start, end),
        }
        deployable = int(held_name.startswith("S2"))
        for model_name, pos in positions.items():
            result = a.backtest_window(prices, pos, start, end, label=model_name)
            row = _metric_row(
                result,
                validation_class="leave_one_wave_out_actual_rls",
                direction=f"holdout_{held_name}",
                window="whole_wave",
                template_source=(
                    "broad fixed schedule"
                    if model_name == "broad_schedule"
                    else f"other-wave source only: {','.join(fit_waves)}; source indices {min(source_indices)}:{max(source_indices) + 1}"
                ),
                evaluated_segment=f"days {start}:{end}",
                uses_evaluated_prices=False,
                chronologically_deployable=bool(deployable),
            )
            row["held_out_wave"] = held_name
            row["fit_waves"] = ",".join(fit_waves)
            row["source_indices_overlap"] = int(bool(source_indices.intersection(set(range(start, end)))))
            wave_rows.append(row)
    wave_frame = pd.DataFrame(wave_rows)
    assert int(wave_frame["source_indices_overlap"].max()) == 0

    # Ablation chain on the identical S2-to-year-end evaluation window.
    ablation_order = [
        "constant_45_mean_reversion",
        "online_baseline_mean",
        "broad_calendar_schedule",
        "fixed_transfer_no_rls",
        "baseline_only_rls",
        "fixed_transfer_rls",
        "academic_transfer_rls",
        "dual_transfer_agreement",
        "calendar_plus_summer",
    ]
    ablation_rows: list[dict[str, object]] = []
    previous_pnl: float | None = None
    for model_name in ablation_order:
        positions, source = s2_models[model_name]
        result = a.backtest_window(prices, positions, S2_START, YEAR_END, label=model_name)
        row = _metric_row(
            result,
            validation_class="chronological_disjoint_ablation",
            direction="S1_to_S2",
            window="year_end_including_summer",
            template_source=source,
            evaluated_segment=f"days {S2_START}:{YEAR_END}",
            uses_evaluated_prices=False,
            chronologically_deployable=True,
        )
        row["incremental_pnl_vs_previous_ablation"] = np.nan if previous_pnl is None else float(result["pnl"] - previous_pnl)
        previous_pnl = float(result["pnl"])
        ablation_rows.append(row)
    ablation_frame = pd.DataFrame(ablation_rows)

    # This is intentionally post-validation and exploratory: it is not part
    # of the preregistered candidate.  It tests whether the best simple
    # baseline benefits from the independently motivated official summer rule.
    constant45_actual = a.baseline_mean_reversion_positions(prices, start=0, end=YEAR_END, mode="fixed")
    summer_shrunk_actual = a.summer_policy_positions(prices, int(a.landmark_days(2026)["summer"]), mode="shrunk_mean", prior_strength=14.0)
    exploratory_actual = np.where(np.arange(len(prices)) >= int(a.landmark_days(2026)["summer"]), summer_shrunk_actual, constant45_actual).astype(int)
    exploratory_result = a.backtest_window(prices, exploratory_actual, S2_START, YEAR_END, label="constant45_plus_shrunk_summer")
    exploratory_actual_frame = pd.DataFrame(
        [
            _metric_row(
                exploratory_result,
                validation_class="exploratory_post_validation",
                direction="S1_to_S2",
                window="year_end_including_summer",
                template_source="constant AUD 45 mean reversion, then official 2026 summer shrunk mean (prior strength 14)",
                evaluated_segment=f"days {S2_START}:{YEAR_END}",
                uses_evaluated_prices=False,
                chronologically_deployable=True,
            )
        ]
    )

    # Full-path placebos are intentionally labelled in-sample diagnostics.
    placebo_rows: list[dict[str, object]] = []
    for name, template in _placebo_templates(fixed_full).items():
        warped_placebo = _warp_arbitrary_template(template)
        rls = a.rls_positions(prices, template, forgetting=0.995, mode="global", prior_variance=4.0)
        dual = a.rls_dual_template_agreement_positions(prices, template, warped_placebo, forgetting=0.995)
        rls_result = a.backtest(prices, rls, label=name)
        dual_result = a.backtest(prices, dual, label=name)
        placebo_rows.append(
            {
                "placebo": name,
                "rls_pnl": rls_result["pnl"],
                "rls_sharpe": rls_result["sharpe"],
                "rls_active_hit_rate": rls_result["active_hit_rate"],
                "rls_active_days": rls_result["active_days"],
                "rls_max_drawdown": rls_result["max_drawdown"],
                "dual_pnl": dual_result["pnl"],
                "dual_p10_not_applicable": np.nan,
                "dual_active_days": dual_result["active_days"],
                "dual_max_drawdown": dual_result["max_drawdown"],
                "template_source": "full Round 1 centered 7-day path or deterministic placebo",
                "evaluated_segment": "full Round 1 days 0:365",
                "uses_evaluated_prices": 1,
                "chronologically_deployable": 0,
                "evidence_class": "in_sample_template_reconstruction_diagnostic",
            }
        )
    placebo_frame = pd.DataFrame(placebo_rows)

    # Independent event-kernel and cross-semester generator families.
    independent_paths = a.independent_generator_paths(prices, n_paths=n_paths, seed=seed)
    independent_rows: list[dict[str, object]] = []
    exploratory_hybrid_rows: list[dict[str, object]] = []
    agreement_rows: list[dict[str, object]] = []
    independent_schedule = a.fixed_schedule_positions()
    independent_calendar = a.calendar_schedule_positions(2027)
    for path_name, path, metadata in independent_paths:
        fixed_rls = a.rls_positions(path, fixed_full, forgetting=0.995, mode="global", prior_variance=4.0)
        warped_rls = a.rls_positions(path, warped_full, forgetting=0.995, mode="global", prior_variance=4.0)
        dual_rls = a.rls_dual_template_agreement_positions(path, fixed_full, warped_full, forgetting=0.995)
        baseline_rls = a.rls_positions(path, np.full(len(path), 45.0), forgetting=0.995, mode="global", prior_variance=4.0)
        constant45 = a.baseline_mean_reversion_positions(path, start=0, end=YEAR_END, mode="fixed")
        online_baseline = a.baseline_mean_reversion_positions(path, start=0, end=YEAR_END, mode="online_mean")
        summer = a.summer_policy_positions(path, int(a.landmark_days(2027)["summer"]), mode="fixed")
        calendar_plus_summer = np.where(summer != 0, summer, independent_calendar).astype(int)
        constant45_summer_base = a.baseline_mean_reversion_positions(path, start=0, end=YEAR_END, mode="fixed")
        shrunk_summer = a.summer_policy_positions(path, int(a.landmark_days(2027)["summer"]), mode="shrunk_mean", prior_strength=14.0)
        exploratory_hybrid = np.where(np.arange(len(path)) >= int(a.landmark_days(2027)["summer"]), shrunk_summer, constant45_summer_base).astype(int)
        strategy_positions = {
            "broad_fixed_schedule": independent_schedule,
            "broad_calendar_schedule": independent_calendar,
            "fixed_template_no_rls": a.template_positions(fixed_full),
            "constant45_mean_reversion": constant45,
            "online_baseline_mean": online_baseline,
            "baseline_only_rls": baseline_rls,
            "fixed_template_rls": fixed_rls,
            "warped_template_rls": warped_rls,
            "dual_template_agreement": dual_rls,
            "calendar_plus_summer": calendar_plus_summer,
        }
        agreement_rows.append(_agreement_row(path_name, metadata, path, fixed_rls, warped_rls, dual_rls))
        for model_name, positions in strategy_positions.items():
            result = a.backtest(path, positions, label=model_name)
            independent_rows.append(
                {
                    "path_name": path_name,
                    "generator_family": metadata["generator_family"],
                    "generator_source": metadata["generator_source"],
                    "timing_mode": metadata["timing_mode"],
                    "easter_shift_days": metadata.get("easter_shift_days", 0),
                    "peak_shift_days": metadata.get("peak_shift_days", 0),
                    "secondary_present": metadata.get("secondary_present", 1),
                    "summer_shift": metadata.get("summer_shift", 0.0),
                    "summer_shift_bucket": _summer_shift_bucket(float(metadata.get("summer_shift", 0.0))),
                    "transition_days": metadata.get("transition_days", 0),
                    "model": model_name,
                    "template_source": "frozen full Round 1 prior only" if "template" in model_name or model_name == "dual_template_agreement" else "calendar or constant prior",
                    "validation_class": "independent_synthetic",
                    "uses_evaluated_prices": 0,
                    "chronologically_deployable": 1,
                    "pnl": result["pnl"],
                    "sharpe": result["sharpe"],
                    "active_hit_rate": result["active_hit_rate"],
                    "active_days": result["active_days"],
                    "max_drawdown": result["max_drawdown"],
                    "max_capital": result["max_capital"],
                    "pnl_per_max_capital": result["pnl_per_max_capital"],
                    "budget_violations": result["budget_violations"],
                    "integral_positions": result["integral_positions"],
                    "within_limit": result["within_limit"],
                }
            )
        hybrid_result = a.backtest(path, exploratory_hybrid, label="constant45_plus_shrunk_summer")
        exploratory_hybrid_rows.append(
            {
                "path_name": path_name,
                "generator_family": metadata["generator_family"],
                "timing_mode": metadata["timing_mode"],
                "summer_shift": metadata.get("summer_shift", 0.0),
                "model": "constant45_plus_shrunk_summer",
                "validation_class": "exploratory_post_validation_independent_synthetic",
                "pnl": hybrid_result["pnl"],
                "sharpe": hybrid_result["sharpe"],
                "active_hit_rate": hybrid_result["active_hit_rate"],
                "active_days": hybrid_result["active_days"],
                "max_drawdown": hybrid_result["max_drawdown"],
                "max_capital": hybrid_result["max_capital"],
                "pnl_per_max_capital": hybrid_result["pnl_per_max_capital"],
                "budget_violations": hybrid_result["budget_violations"],
                "integral_positions": hybrid_result["integral_positions"],
                "within_limit": hybrid_result["within_limit"],
            }
        )
    independent_detail = pd.DataFrame(independent_rows)
    independent_summary = _summary_by_model(independent_detail)
    timing_summary = _summary_by_model(independent_detail, ["timing_mode", "model"])
    agreement_frame = pd.DataFrame(agreement_rows)
    exploratory_hybrid_detail = pd.DataFrame(exploratory_hybrid_rows)
    exploratory_hybrid_summary = _summary_by_model(exploratory_hybrid_detail)

    # Summer is evaluated as its own policy, not only as an overlay to a full
    # seasonal template.  The official dates are chosen before looking at P&L.
    summer_rows: list[dict[str, object]] = []
    summer_modes = ["fixed", "online_mean", "online_median", "shrunk_mean"]
    summer_starts = [("day_302", 302), ("day_322", 322), ("official_2026_summer", int(a.landmark_days(2026)["summer"]))]
    for start_name, start in summer_starts:
        for mode in summer_modes:
            positions = a.summer_policy_positions(prices, start, mode=mode, prior_strength=14.0)
            result = a.backtest_window(prices, positions, start, YEAR_END, label=f"{mode}_{start_name}")
            summer_rows.append(
                {
                    "source": "Round1_summer_diagnostic",
                    "path_name": "observed_round1",
                    "start_rule": start_name,
                    "start_day": start,
                    "mode": mode,
                    "summer_shift": 0.0,
                    "summer_shift_bucket": "observed",
                    "pnl": result["pnl"],
                    "sharpe": result["sharpe"],
                    "active_hit_rate": result["active_hit_rate"],
                    "active_days": result["active_days"],
                    "max_drawdown": result["max_drawdown"],
                    "max_capital": result["max_capital"],
                    "pnl_per_max_capital": result["pnl_per_max_capital"],
                    "uses_evaluated_prices": 0,
                    "chronologically_deployable": 1,
                }
            )
    synthetic_summer_rows: list[dict[str, object]] = []
    official_2027_summer = int(a.landmark_days(2027)["summer"])
    for path_name, path, metadata in independent_paths:
        for mode in summer_modes:
            positions = a.summer_policy_positions(path, official_2027_summer, mode=mode, prior_strength=14.0)
            result = a.backtest_window(path, positions, official_2027_summer, YEAR_END, label=mode)
            synthetic_summer_rows.append(
                {
                    "source": "independent_synthetic",
                    "path_name": path_name,
                    "generator_family": metadata["generator_family"],
                    "timing_mode": metadata["timing_mode"],
                    "start_rule": "official_2027_summer",
                    "start_day": official_2027_summer,
                    "mode": mode,
                    "summer_shift": metadata.get("summer_shift", 0.0),
                    "summer_shift_bucket": _summer_shift_bucket(float(metadata.get("summer_shift", 0.0))),
                    "pnl": result["pnl"],
                    "sharpe": result["sharpe"],
                    "active_hit_rate": result["active_hit_rate"],
                    "active_days": result["active_days"],
                    "max_drawdown": result["max_drawdown"],
                    "max_capital": result["max_capital"],
                    "pnl_per_max_capital": result["pnl_per_max_capital"],
                    "uses_evaluated_prices": 0,
                    "chronologically_deployable": 1,
                }
            )
    summer_frame = pd.DataFrame(summer_rows + synthetic_summer_rows)
    summer_independent_summary = _summary_by_model(
        summer_frame[summer_frame["source"] == "independent_synthetic"].rename(columns={"mode": "model"}),
        ["model"],
    )
    summer_shift_summary = _summary_by_model(
        summer_frame[summer_frame["source"] == "independent_synthetic"].rename(columns={"mode": "model"}),
        ["summer_shift_bucket", "model"],
    )

    # Persist new evidence under distinct names.  Existing first-stage tables
    # are intentionally not overwritten.
    outputs = {
        "heldout_rls.csv": heldout_frame,
        "rls_ablations.csv": ablation_frame,
        "template_placebos.csv": placebo_frame,
        "rls_timing_stress.csv": timing_summary,
        "independent_stress_summary.csv": independent_summary,
        "independent_stress_detail.csv": independent_detail,
        "summer_validation.csv": summer_frame,
        "summer_independent_summary.csv": summer_independent_summary,
        "summer_shift_summary.csv": summer_shift_summary,
        "template_provenance.csv": pd.DataFrame(provenance_rows),
        "leakage_checks.csv": leakage_frame,
        "leave_one_wave_out_rls.csv": wave_frame,
        "agreement_policy_comparison.csv": agreement_frame,
        "exploratory_hybrid_validation.csv": exploratory_actual_frame,
        "exploratory_hybrid_stress_detail.csv": exploratory_hybrid_detail,
        "exploratory_hybrid_stress_summary.csv": exploratory_hybrid_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(result_dir / name, index=False)

    _save_stage2_figures(
        result_dir,
        figure_dir,
        prices,
        {
            "broad_calendar_schedule": s2_models["broad_calendar_schedule"][0],
            "fixed_transfer_rls": s2_models["fixed_transfer_rls"][0],
            "academic_transfer_rls": s2_models["academic_transfer_rls"][0],
            "dual_transfer_agreement": s2_models["dual_transfer_agreement"][0],
            "calendar_plus_summer": s2_models["calendar_plus_summer"][0],
        },
        independent_detail,
    )

    return {
        "heldout_rls": heldout_frame,
        "ablations": ablation_frame,
        "placebos": placebo_frame,
        "timing_summary": timing_summary,
        "independent_summary": independent_summary,
        "independent_detail": independent_detail,
        "summer": summer_frame,
        "summer_summary": summer_independent_summary,
        "summer_shift_summary": summer_shift_summary,
        "provenance": pd.DataFrame(provenance_rows),
        "leakage": leakage_frame,
        "wave_rls": wave_frame,
        "agreement": agreement_frame,
        "exploratory_hybrid": exploratory_hybrid_summary,
    }
