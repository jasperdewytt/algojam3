"""One-time locked-final execution for the frozen Pass 4 wrapper.

The normal Pass 3 and Pass 4 runners never import this module.  The existing
``pass3_experiments --final`` entry point imports ``run_locked_final`` only
after the explicit final flag has been supplied and the manifest preflight has
passed.  The final scenario builder is consequently called in one place only.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

from .cold_start_strategies import COLD_START_STRATEGY_NAMES
from .pass3_experiments import (
    RunMetric,
    _actual_summary,
    _attach_comparisons,
    _group,
    _metric_without_comparison,
    _pct,
)
from .pass4_strategies import Risk50Burnin1Markov
from .simulator import Agent, LiferaftSimulator, SimulationResult


FINAL_WRAPPER_NAME = Risk50Burnin1Markov.name
FINAL_CANDIDATES: tuple[str, ...] = (*COLD_START_STRATEGY_NAMES, FINAL_WRAPPER_NAME)
FINAL_OTHER_EXPOSURE = 0.0
FINAL_SCENARIO_COUNT = 160
FINAL_REPORT_PATH = Path(__file__).with_name("PASS3_FINAL_REPORT.md")
FINAL_RESULTS_PATH = Path(__file__).with_name("PASS4_FINAL_RESULTS.json")
FINAL_DECISION_PATH = Path(__file__).with_name("PASS4_FINAL_DECISION.md")
FINAL_RECEIPT_PATH = Path(__file__).with_name("PASS4_FINAL_EXECUTION_RECEIPT.json")
MANIFEST_PATH = Path(__file__).with_name("PASS3_FINAL_MANIFEST.md")
PROTOCOL_PATH = Path(__file__).with_name("PASS4_FINAL_DECISION_PROTOCOL.md")
FINAL_HASH_FILES: tuple[str, ...] = (
    "archetypes.py",
    "simulator.py",
    "strategies.py",
    "cold_start_strategies.py",
    "pass3_scenarios.py",
    "pass3_experiments.py",
    "pass4_strategies.py",
    "pass4_final.py",
)
FINAL_COMMAND = "python -m research.liferaft.pass3_experiments --final"


def compute_locked_hashes() -> tuple[dict[str, str], str]:
    """Recompute the manifest's individual and combined source hashes."""

    manifest = MANIFEST_PATH.read_text(encoding="utf-8")
    expected = dict(re.findall(r"\| `([^`]+)` \| `([A-F0-9]{64})` \|", manifest))
    root = Path(__file__).parent
    actual: dict[str, str] = {}
    for name in FINAL_HASH_FILES:
        if name not in expected:
            raise RuntimeError(f"manifest is missing locked source {name!r}")
        actual[name] = hashlib.sha256((root / name).read_bytes()).hexdigest().upper()
        if actual[name] != expected[name]:
            raise RuntimeError(
                f"locked hash mismatch for {name}: expected {expected[name]}, "
                f"got {actual[name]}"
            )
    lines = "\n".join(f"{name} {actual[name]}" for name in FINAL_HASH_FILES) + "\n"
    combined = hashlib.sha256(lines.encode("utf-8")).hexdigest().upper()
    manifest_combined = re.search(r"Combined hash: `([A-F0-9]{64})`", manifest)
    if manifest_combined is None or combined != manifest_combined.group(1):
        raise RuntimeError(
            "locked combined hash mismatch: "
            f"expected {manifest_combined.group(1) if manifest_combined else None}, "
            f"got {combined}"
        )
    protocol_match = re.search(
        r"Decision protocol SHA-256: `([A-F0-9]{64})`", manifest
    )
    if protocol_match is None:
        raise RuntimeError("manifest is missing the frozen decision-protocol hash")
    protocol_hash = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest().upper()
    if protocol_hash != protocol_match.group(1):
        raise RuntimeError(
            f"decision protocol hash mismatch: expected {protocol_match.group(1)}, "
            f"got {protocol_hash}"
        )
    return actual, combined


def _preflight() -> tuple[dict[str, str], str]:
    """Refuse a second execution or a changed final definition."""

    if FINAL_REPORT_PATH.exists() or FINAL_RESULTS_PATH.exists() or FINAL_RECEIPT_PATH.exists():
        raise RuntimeError("the locked final suite has already been consumed")
    manifest = MANIFEST_PATH.read_text(encoding="utf-8")
    if "Status: **locked and unconsumed" not in manifest:
        raise RuntimeError("the final manifest is not in locked and unconsumed state")
    if FINAL_WRAPPER_NAME not in manifest:
        raise RuntimeError("the selected wrapper is absent from the final manifest")
    hashes, combined = compute_locked_hashes()
    return hashes, combined


def run_final_candidate(
    scenario: Any,
    strategy: str,
) -> tuple[SimulationResult, Risk50Burnin1Markov | None]:
    """Run one final candidate with a fresh focal and fresh opponents.

    This public helper is intentionally usable with a consumed validation
    scenario for pre-final construction/parity tests.  It does not call the
    locked scenario builder itself.
    """

    if strategy != FINAL_WRAPPER_NAME:
        return scenario.run(strategy, other_portfolio_exposure=FINAL_OTHER_EXPOSURE), None

    focal = Risk50Burnin1Markov(other_portfolio_exposure=FINAL_OTHER_EXPOSURE)
    opponents = scenario.population_factory()
    agents: tuple[Agent, ...] = (focal, *opponents)
    result = LiferaftSimulator(
        agents,
        scenario.config,
        other_portfolio_exposure=FINAL_OTHER_EXPOSURE,
        focal_agent_name=focal.name,
        scenario_name=scenario.name,
        scenario_configuration={
            "suite": "final",
            "candidate": FINAL_WRAPPER_NAME,
            "family": scenario.family,
            "seed": scenario.seed,
            "execution_mode": scenario.execution_mode,
            "population": scenario.population_description,
            "population_size": scenario.population_size,
            "pair_id": scenario.pair_id,
            "path_controlled": scenario.path_controlled,
            "other_portfolio_exposure": FINAL_OTHER_EXPOSURE,
        },
        random_seeds={"scenario": scenario.seed},
    ).run()
    return result, focal


def _wrapper_diagnostics(wrapper: Risk50Burnin1Markov | None) -> dict[str, Any]:
    if wrapper is None:
        return {}
    diagnostics = wrapper.diagnostics()
    loss_event = diagnostics["loss_stop_event"]
    health_event = diagnostics["health_stop_event"]

    def event_values(event: Any) -> dict[str, Any]:
        if event is None:
            return {"day": None, "pnl_before": None, "pnl_after": None, "overshoot": 0}
        return {
            "day": event.day,
            "pnl_before": event.pnl_before,
            "pnl_after": event.pnl_after,
            "overshoot": event.loss_limit_overshoot,
        }

    return {
        "loss_stop_active": bool(diagnostics["loss_stop_active"]),
        "health_stop_active": bool(diagnostics["health_stop_active"]),
        "first_stop_trigger": diagnostics["first_stop_trigger"],
        "loss_stop": event_values(loss_event),
        "health_stop": event_values(health_event),
        "quality_scoreable": diagnostics["quality_scoreable"],
        "unknown_pause_count": diagnostics["unknown_pause_count"],
        "floor_gate_count": diagnostics["floor_gate_count"],
        "headroom_gate_count": diagnostics["headroom_gate_count"],
        "raw_nonzero_requests": diagnostics["raw_nonzero_requests"],
        "suppressed_by_gate": diagnostics["raw_nonzero_suppressed_by_gate"],
        "cumulative_marked_pnl": diagnostics["cumulative_marked_pnl"],
    }


def _wrapper_records(
    metrics: Sequence[RunMetric],
    diagnostics: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for metric in metrics:
        if metric.strategy != FINAL_WRAPPER_NAME:
            continue
        values.append(
            {
                "scenario": metric.scenario,
                "family": metric.family,
                "execution_mode": metric.execution_mode,
                "seed": metric.seed,
                **diagnostics[(metric.scenario, metric.execution_mode)],
            }
        )
    return values


def _final_strategy_table(metrics: Sequence[RunMetric]) -> str:
    lines = [
        "| strategy | runs | mean | median | lower quartile | worst run | mean/day | mean DD | max DD | active | hit | turnover | breaches | rejected | beat flat | flat tied | pivotal P&L | non-pivotal P&L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in FINAL_CANDIDATES:
        row = _actual_summary([m for m in metrics if m.strategy == strategy])
        lines.append(
            f"| `{strategy}` | {int(row['runs'])} | {row['mean']:.0f} | "
            f"{row['median']:.0f} | {row['lower_quartile']:.0f} | {row['worst']:.0f} | "
            f"{row['mean_per_day']:.1f} | {row['mean_drawdown']:.0f} | "
            f"{row['max_drawdown']:.0f} | {row['mean_active']:.1f} | "
            f"{_pct(row['hit_rate'])} | {row['mean_turnover']:.1f} | "
            f"{row['breaches']:.1f} | {row['rejections']:.1f} | "
            f"{_pct(row['beat_flat'])} | {_pct(row['flat_tied'])} | "
            f"{row['pivotal']:.0f} | {row['non_pivotal']:.0f} |"
        )
    return "\n".join(lines)


def _final_family_table(metrics: Sequence[RunMetric]) -> str:
    lines = [
        "| family | strategy | runs | mean P&L | median | lower quartile | worst | mean DD | max DD | active | beat flat | pivotal P&L | non-pivotal P&L |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (family, strategy), values in sorted(_group(metrics, "family", "strategy").items()):
        row = _actual_summary(values)
        lines.append(
            f"| {family} | `{strategy}` | {int(row['runs'])} | {row['mean']:.0f} | "
            f"{row['median']:.0f} | {row['lower_quartile']:.0f} | {row['worst']:.0f} | "
            f"{row['mean_drawdown']:.0f} | {row['max_drawdown']:.0f} | "
            f"{row['mean_active']:.1f} | {_pct(row['beat_flat'])} | "
            f"{row['pivotal']:.0f} | {row['non_pivotal']:.0f} |"
        )
    return "\n".join(lines)


def _paired_table(metrics: Sequence[RunMetric], right: str) -> str:
    lines = [
        "| comparison | paired cases | mean wrapper minus comparison | median | wrapper wins | ties | wrapper losses |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    left = {metric.scenario: metric for metric in metrics if metric.strategy == FINAL_WRAPPER_NAME}
    other = {metric.scenario: metric for metric in metrics if metric.strategy == right}
    differences = [left[key].marked_pnl - other[key].marked_pnl for key in left.keys() & other.keys()]
    lines.append(
        f"| `{right}` overall | {len(differences)} | {mean(differences) if differences else 0:.0f} | "
        f"{median(differences) if differences else 0:.0f} | {sum(value > 0 for value in differences)} | "
        f"{sum(value == 0 for value in differences)} | {sum(value < 0 for value in differences)} |"
    )
    for family in sorted({metric.family for metric in metrics}):
        family_left = {key: value for key, value in left.items() if value.family == family}
        family_other = {key: value for key, value in other.items() if value.family == family}
        family_differences = [
            family_left[key].marked_pnl - family_other[key].marked_pnl
            for key in family_left.keys() & family_other.keys()
        ]
        lines.append(
            f"| `{right}` / {family} | {len(family_differences)} | "
            f"{mean(family_differences) if family_differences else 0:.0f} | "
            f"{median(family_differences) if family_differences else 0:.0f} | "
            f"{sum(value > 0 for value in family_differences)} | "
            f"{sum(value == 0 for value in family_differences)} | "
            f"{sum(value < 0 for value in family_differences)} |"
        )
    return "\n".join(lines)


def _wrapper_stop_table(
    metrics: Sequence[RunMetric],
    diagnostics: dict[tuple[str, str], dict[str, Any]],
) -> str:
    values = [metric for metric in metrics if metric.strategy == FINAL_WRAPPER_NAME]
    loss = [diagnostics[(metric.scenario, metric.execution_mode)] for metric in values if diagnostics[(metric.scenario, metric.execution_mode)]["loss_stop_active"]]
    health = [diagnostics[(metric.scenario, metric.execution_mode)] for metric in values if diagnostics[(metric.scenario, metric.execution_mode)]["health_stop_active"]]
    both = [item for item in diagnostics.values() if item["loss_stop_active"] and item["health_stop_active"]]
    both_loss_first = sum(
        item["loss_stop"]["day"] is not None
        and item["health_stop"]["day"] is not None
        and item["loss_stop"]["day"] < item["health_stop"]["day"]
        for item in both
    )
    both_health_first = sum(
        item["loss_stop"]["day"] is not None
        and item["health_stop"]["day"] is not None
        and item["health_stop"]["day"] < item["loss_stop"]["day"]
        for item in both
    )
    same_observation = sum(
        item["loss_stop"]["day"] is not None
        and item["loss_stop"]["day"] == item["health_stop"]["day"]
        for item in both
    )
    global_loss_first = sum(item["first_stop_trigger"] == "loss_stop" for item in diagnostics.values())
    global_health_first = sum(item["first_stop_trigger"] == "health_stop" for item in diagnostics.values())
    loss_days = [item["loss_stop"]["day"] for item in loss if item["loss_stop"]["day"] is not None]
    health_days = [item["health_stop"]["day"] for item in health if item["health_stop"]["day"] is not None]
    overshoots = [item["loss_stop"]["overshoot"] for item in loss]
    lines = [
        "| total wrapper runs | loss stops | health stops | both stops | both loss first | both health first | same observation | first loss (all) | first health (all) | mean/median loss day | mean/median health day | max overshoot | mean unknown pauses | mean floor gates | mean headroom gates |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
        f"| {len(values)} | {len(loss)} ({_pct(len(loss) / len(values))}) | "
        f"{len(health)} ({_pct(len(health) / len(values))}) | "
        f"{len(both)} ({_pct(len(both) / len(values))}) | {both_loss_first} | "
        f"{both_health_first} | {same_observation} | {global_loss_first} | "
        f"{global_health_first} | {mean(loss_days) if loss_days else 0:.1f}/"
        f"{median(loss_days) if loss_days else 0:.1f} | "
        f"{mean(health_days) if health_days else 0:.1f}/"
        f"{median(health_days) if health_days else 0:.1f} | "
        f"{max(overshoots, default=0)} | "
        f"{mean(item['unknown_pause_count'] for item in diagnostics.values()):.1f} | "
        f"{mean(item['floor_gate_count'] for item in diagnostics.values()):.1f} | "
        f"{mean(item['headroom_gate_count'] for item in diagnostics.values()):.1f} |",
    ]
    return "\n".join(lines)


def _overshoot_table(diagnostics: dict[tuple[str, str], dict[str, Any]]) -> str:
    loss = [item for item in diagnostics.values() if item["loss_stop_active"]]
    counts: dict[int, int] = {}
    for item in loss:
        amount = int(item["loss_stop"]["overshoot"])
        counts[amount] = counts.get(amount, 0) + 1
    lines = [
        "| overshoot amount | loss-stop cases | percentage of loss-stop cases |",
        "|---:|---:|---:|",
    ]
    for amount in sorted({0, *counts}):
        lines.append(
            f"| {amount:,} | {counts.get(amount, 0)} | "
            f"{_pct(counts.get(amount, 0) / len(loss)) if loss else '0.0%'} |"
        )
    return "\n".join(lines)


def evaluate_decision(
    metrics: Sequence[RunMetric],
    diagnostics: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    wrapper = [metric for metric in metrics if metric.strategy == FINAL_WRAPPER_NAME]
    raw = [metric for metric in metrics if metric.strategy == "burnin1_markov"]
    summary = _actual_summary(wrapper)
    family_means = {
        family: mean(metric.marked_pnl for metric in values)
        for (family,), values in _group(wrapper, "family").items()
    }
    loss_rate = mean(item["loss_stop_active"] for item in diagnostics.values())
    health_rate = mean(item["health_stop_active"] for item in diagnostics.values())
    total_breaches = sum(metric.budget_breaches for metric in wrapper)
    total_rejections = sum(metric.rejected_actions for metric in wrapper)
    raw_mean = mean(metric.marked_pnl for metric in raw) if raw else 0.0
    wrapper_mean = summary["mean"]
    retention = wrapper_mean / raw_mean if raw_mean > 0 else None
    criteria: list[dict[str, Any]] = []

    def add(name: str, required: str, observed: Any, passed: bool) -> None:
        criteria.append(
            {"criterion": name, "required": required, "observed": observed, "pass": bool(passed)}
        )

    add("lower_quartile_pnl", ">= -25000", summary["lower_quartile"], summary["lower_quartile"] >= -25_000)
    add("worst_pnl", ">= -60000", summary["worst"], summary["worst"] >= -60_000)
    add("maximum_drawdown", "<= 75000", summary["max_drawdown"], summary["max_drawdown"] <= 75_000)
    add("focal_budget_breaches", "== 0", total_breaches, total_breaches == 0)
    add("focal_rejected_actions", "== 0", total_rejections, total_rejections == 0)
    add("mean_pivotal_pnl", ">= -25000", summary["pivotal"], summary["pivotal"] >= -25_000)
    add("loss_stop_rate", "<= 10%", loss_rate, loss_rate <= 0.10)
    add("health_stop_rate", "<= 75%", health_rate, health_rate <= 0.75)
    add("minimum_family_mean_pnl", ">= -30000", min(family_means.values()), min(family_means.values()) >= -30_000)
    add("overall_mean_pnl", "> 0", summary["mean"], summary["mean"] > 0)
    add("median_pnl", ">= 0", summary["median"], summary["median"] >= 0)
    add("beat_flat_fraction", ">= 55%", summary["beat_flat"], summary["beat_flat"] >= 0.55)
    positive_families = sum(value > 0 for value in family_means.values())
    add(
        "positive_family_fraction",
        f">= {ceil(len(family_means) / 2)} of {len(family_means)} families",
        positive_families,
        positive_families >= ceil(len(family_means) / 2),
    )
    if raw_mean > 0:
        add("raw_mean_retention", ">= 50%", retention, retention >= 0.50)
    else:
        add("raw_mean_retention", "N/A when raw mean <= 0", None, True)
    return {
        "decision": "PASS" if all(item["pass"] for item in criteria) else "FAIL",
        "wrapper_summary": summary,
        "raw_mean": raw_mean,
        "family_means": family_means,
        "loss_stop_rate": loss_rate,
        "health_stop_rate": health_rate,
        "criteria": criteria,
    }


def render_decision(decision: dict[str, Any], *, combined_hash: str, timestamp: str) -> str:
    lines = [
        "# Liferaft Pass 4 final decision",
        "",
        f"Decision: **{decision['decision']}**",
        "",
        "This result applies the frozen `PASS4_FINAL_DECISION_PROTOCOL.md`",
        "mechanically to the zero-other-exposure locked final suite.",
        "",
        f"- Execution timestamp: `{timestamp}`",
        f"- Locked combined hash: `{combined_hash}`",
        f"- Candidates: {len(FINAL_CANDIDATES)}",
        f"- Scenarios: {FINAL_SCENARIO_COUNT}",
        "",
        "| criterion | required threshold | observed result | decision |",
        "|---|---|---:|---|",
    ]
    for item in decision["criteria"]:
        observed = item["observed"]
        if isinstance(observed, float):
            observed_text = f"{observed * 100:.1f}%" if "rate" in item["criterion"] or "fraction" in item["criterion"] else f"{observed:.2f}"
        elif observed is None:
            observed_text = "N/A"
        else:
            observed_text = str(observed)
        lines.append(
            f"| `{item['criterion']}` | {item['required']} | {observed_text} | "
            f"{'PASS' if item['pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "A PASS is a risk-control gate, not proof of positive real competition",
            "expected value. If this decision is FAIL, production Liferaft must",
            "remain flat. No post-final parameter changes are permitted.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_final_report(
    metrics: Sequence[RunMetric],
    diagnostics: dict[tuple[str, str], dict[str, Any]],
    *,
    combined_hash: str,
    timestamp: str,
    decision: dict[str, Any],
) -> str:
    families = len({metric.family for metric in metrics})
    return f"""# Liferaft Pass 4 final locked-suite report

> The locked final suite was executed once under the frozen Pass 4 decision
> protocol. These outcomes are consumed and must not be used to retune the
> wrapper.

## Execution receipt

| item | value |
|---|---:|
| timestamp | `{timestamp}` |
| command | `{FINAL_COMMAND}` |
| locked combined hash | `{combined_hash}` |
| scenarios | {FINAL_SCENARIO_COUNT} |
| candidates | {len(FINAL_CANDIDATES)} |
| candidate cells | {FINAL_SCENARIO_COUNT * len(FINAL_CANDIDATES)} |
| other exposure | AUD 0 |
| final families | {families} |
| automatic decision | **{decision['decision']}** |

The comparison uses fresh candidate and opponent instances for every scenario.
The wrapper is scored through the same simulator and P&L timing as the other
candidates. Because the focal vote can alter the market, these are endogenous
same-initial-scenario game paths, not fixed-path backtests.

## Overall final results

{_final_strategy_table(metrics)}

## Family results

These are actual run-level family summaries; family rows are not treated as
independent samples and are not used to manufacture aggregate quantiles.

{_final_family_table(metrics)}

## Wrapper diagnostics

{_wrapper_stop_table(metrics, diagnostics)}

### Loss-limit overshoot distribution

The distribution is conditional on wrapper loss-stop cases and includes zero
overshoot. Positive overshoot can only arise from the final newly observable
adverse movement that crosses AUD 50,000.

{_overshoot_table(diagnostics)}

## Paired comparisons

{_paired_table(metrics, "flat")}

{_paired_table(metrics, "burnin1_markov")}

{_paired_table(metrics, "online_drift")}

## Decision protocol outcome

See `PASS4_FINAL_DECISION.md` for every conjunctive criterion and its observed
PASS/FAIL status. Passing this synthetic suite is not proof of positive real
competition expected value.
"""


def _json_results(
    metrics: Sequence[RunMetric],
    diagnostics: dict[tuple[str, str], dict[str, Any]],
    *,
    combined_hash: str,
    timestamp: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "command": FINAL_COMMAND,
        "locked_combined_hash": combined_hash,
        "scenario_count": FINAL_SCENARIO_COUNT,
        "candidate_count": len(FINAL_CANDIDATES),
        "candidate_names": list(FINAL_CANDIDATES),
        "other_portfolio_exposure": FINAL_OTHER_EXPOSURE,
        "decision": decision,
        "metrics": [asdict(metric) for metric in metrics],
        "wrapper_diagnostics": _wrapper_records(metrics, diagnostics),
    }


def run_locked_final() -> tuple[RunMetric, ...]:
    """Execute and persist the one authorized final suite exactly once."""

    hashes, combined_hash = _preflight()
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"FINAL_EXECUTION_TIMESTAMP={timestamp}")
    print(f"FINAL_EXECUTION_COMMAND={FINAL_COMMAND}")
    print(f"FINAL_LOCKED_COMBINED_HASH={combined_hash}")
    print(f"FINAL_SCENARIOS={FINAL_SCENARIO_COUNT}")
    print(f"FINAL_CANDIDATES={len(FINAL_CANDIDATES)}")
    print("FINAL_CONFIRMATION=first-and-only-execution")
    del hashes

    # This is the sole call site that instantiates the locked scenarios.
    from .pass3_scenarios import final_scenarios

    scenarios = tuple(final_scenarios())
    if len(scenarios) != FINAL_SCENARIO_COUNT:
        raise RuntimeError(f"locked scenario count changed: {len(scenarios)}")

    metrics: list[RunMetric] = []
    diagnostics: dict[tuple[str, str], dict[str, Any]] = {}
    for scenario_index, scenario in enumerate(scenarios, start=1):
        print(
            f"[final] scenario {scenario_index}/{len(scenarios)} "
            f"{scenario.name} ({scenario.execution_mode})"
        )
        for strategy in FINAL_CANDIDATES:
            result, wrapper = run_final_candidate(scenario, strategy)
            metrics.append(
                _metric_without_comparison(
                    result,
                    suite="final",
                    scenario=scenario,
                    strategy=strategy,
                    other_portfolio_exposure=FINAL_OTHER_EXPOSURE,
                )
            )
            if wrapper is not None:
                diagnostics[(scenario.name, scenario.execution_mode)] = _wrapper_diagnostics(wrapper)

    final_metrics = _attach_comparisons(metrics)
    if len(final_metrics) != FINAL_SCENARIO_COUNT * len(FINAL_CANDIDATES):
        raise RuntimeError(f"unexpected final cell count: {len(final_metrics)}")
    decision = evaluate_decision(final_metrics, diagnostics)

    report = render_final_report(
        final_metrics,
        diagnostics,
        combined_hash=combined_hash,
        timestamp=timestamp,
        decision=decision,
    )
    results = _json_results(
        final_metrics,
        diagnostics,
        combined_hash=combined_hash,
        timestamp=timestamp,
        decision=decision,
    )
    FINAL_RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    FINAL_REPORT_PATH.write_text(report, encoding="utf-8")
    FINAL_DECISION_PATH.write_text(
        render_decision(decision, combined_hash=combined_hash, timestamp=timestamp),
        encoding="utf-8",
    )
    artifact_hashes = {
        "report": hashlib.sha256(FINAL_REPORT_PATH.read_bytes()).hexdigest().upper(),
        "results": hashlib.sha256(FINAL_RESULTS_PATH.read_bytes()).hexdigest().upper(),
        "decision": hashlib.sha256(FINAL_DECISION_PATH.read_bytes()).hexdigest().upper(),
    }
    receipt = {
        "status": "consumed",
        "timestamp": timestamp,
        "command": FINAL_COMMAND,
        "locked_combined_hash": combined_hash,
        "scenario_count": FINAL_SCENARIO_COUNT,
        "candidate_count": len(FINAL_CANDIDATES),
        "candidate_names": list(FINAL_CANDIDATES),
        "artifacts": artifact_hashes,
        "automatic_decision": decision["decision"],
    }
    FINAL_RECEIPT_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {FINAL_REPORT_PATH}")
    print(f"wrote {FINAL_RESULTS_PATH}")
    print(f"wrote {FINAL_DECISION_PATH}")
    print(f"wrote {FINAL_RECEIPT_PATH}")
    print(f"FINAL_DECISION={decision['decision']}")
    return final_metrics


__all__ = [
    "FINAL_CANDIDATES",
    "FINAL_COMMAND",
    "FINAL_OTHER_EXPOSURE",
    "FINAL_SCENARIO_COUNT",
    "FINAL_WRAPPER_NAME",
    "compute_locked_hashes",
    "evaluate_decision",
    "render_decision",
    "render_final_report",
    "run_final_candidate",
    "run_locked_final",
]
