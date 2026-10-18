"""Schedule a future AIPE study, never fit controllers or replay aggregate KPIs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "experiment/conf/aipe_sequential_protocol_v1.json"
OUT = ROOT / "experiment/data/audits/sequential_plan_2026-10-16"
STATE_KEYS = ("carry_inventory_between_reviews", "carry_outstanding_orders_between_reviews")
REQUIRED_ARTIFACTS = (
    "serialized_controllers_and_final_parameters",
    "serialized_selectors_and_training_data_provenance",
    "admissible_labels_available_at_each_update",
    "explicit_initial_inventory_and_outstanding_orders",
    "stateful_policy_switch_adapter",
    "per_period_actions_receipts_inventory_and_unfulfilled_demand",
    "disaggregated_costs_and_predecision_recommendations",
    "verified_record_availability_or_declared_latency_scenario",
    "frozen_primary_strategy_and_confirmatory_reserve_audit",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer in native periods")


def validate(config, dataset):
    spec = config["datasets"][dataset]
    expected = {"BA": "commercial_cycle", "M5": "21_days"}[dataset]
    if spec.get("resolution") != expected or spec.get("action_periods") != 1:
        raise ValueError("Only the existing native observation/action resolution is supported")
    if config["validation_periods"] != 7 or config["evaluation_periods"] != 7:
        raise ValueError("This plan is anchored to current seven-period validation/test panels")
    for field in ("review_cadences", "planning_horizons"):
        if not config[field]:
            raise ValueError(f"Empty {field}")
        for value in config[field]:
            positive_int(value, field)
    for field in ("record_latency_periods", "selector_update_every_periods",
                  "controller_recalibration_every_periods"):
        if type(config[field]) is not int or config[field] < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    if not config.get("availability_assumption"):
        raise ValueError("Record availability must be declared")
    if not config["history_windows"]:
        raise ValueError("Specify history windows")
    for window in config["history_windows"]:
        if window.get("mode") not in ("fixed", "expanding"):
            raise ValueError("History must be fixed or expanding")
        if window["mode"] == "fixed":
            positive_int(window.get("periods"), "history periods")
    state = config["state_contract"]
    if any(state.get(key) is not True for key in STATE_KEYS) or state.get("reset_eoq_at_review") is not False:
        raise ValueError("Inventory, outstanding orders and policy state must persist across reviews")
    if state.get("initial_state_common_across_strategies") is not True or not state.get("state_scope"):
        raise ValueError("Specify common initial conditions and separate strategy trajectories")
    missing = set(REQUIRED_ARTIFACTS) - set(config["required_execution_artifacts"])
    if missing:
        raise ValueError(f"Missing mandatory execution requirements: {sorted(missing)}")
    # Only review cadence changes here. Reaggregation needs another protocol.
    expected_costs = {"monetary_calibration": False,
        "holding_per_unit_end_of_native_period": 1.0,
        "shortage_per_unfulfilled_unit": 5.0, "fixed_per_positive_order": 50.0,
        "purchase_per_ordered_unit": 0.5, "lead_time_native_periods": 2,
        "lead_time_convention": "exact", "fill_rate_target": 0.7}
    if config["scenario_costs"] != expected_costs:
        raise ValueError("Costs/lead time must retain current native-period scenario units")


def decision_windows(origin, periods, cadence, horizon, window, latency=0):
    """All positions are zero-based; *_end_exclusive never includes decision demand."""
    for name, value in (("origin", origin), ("cadence", cadence), ("horizon", horizon)):
        positive_int(value, name)
    start, end = origin + 7, origin + 14
    if end > periods:
        raise ValueError("Incomplete evaluation block")
    for decision in range(start, end, cadence):
        available_end = decision - latency
        history_start = 0 if window["mode"] == "expanding" else available_end - window["periods"]
        if available_end <= 0 or history_start < 0:
            raise ValueError("Insufficient admissible history for configured window/latency")
        yield {"decision_position": decision,
               "history_start": history_start, "history_end_exclusive": available_end,
               "available_last_position": available_end - 1,
               "valid_until_exclusive": min(decision + cadence, end),
               "planning_end_exclusive": decision + horizon,
               "evaluation_end_exclusive": end,
               "scorable_horizon_end_exclusive": min(decision + horizon, end),
               "scoring_horizon_truncated_by_evaluation": decision + horizon > end}


def update_positions(start, end, interval):
    # Initial artifacts must exist before start. Updates are independent of PSE reviews.
    return [] if interval == 0 else list(range(start + interval, end, interval))


def plan_dataset(config, dataset, config_path=DEFAULT_CONFIG):
    import pandas as pd

    validate(config, dataset)
    run = ROOT / "experiment/data/08_reporting/research_v2" / config["datasets"][dataset]["run"]
    source_path = run / "plan.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("dataset") != dataset:
        raise ValueError("Source plan belongs to another dataset")
    target = OUT / dataset
    target.mkdir(parents=True, exist_ok=True)
    rows = cases = 0
    panel_info = []
    scenarios = list(itertools.product(config["review_cadences"], config["planning_horizons"], config["history_windows"]))
    schedule_path = target / "agenda.csv"
    with schedule_path.open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for meta in source["panel_metadata"]:
            path = run / meta["panel"]
            if sha(path) != meta["panel_sha256"]:
                raise ValueError(f"Panel checksum mismatch: {path}")
            # Index/columns only. No outcomes, active SQLite or scientific imports.
            panel = pd.read_parquet(path, columns=[], use_threads=False)
            import pyarrow.parquet as pq
            columns = [str(x) for x in pq.ParquetFile(path).schema_arrow.names
                       if x not in panel.index.names]
            origin = int(meta["origin"])
            if int(meta["validation_cycles"]) != 7 or int(meta["test_cycles"]) != 7:
                raise ValueError("Unexpected source validation/test horizon")
            if len(panel) != meta["eligible_series"] or not panel.index.is_unique:
                raise ValueError("Source cohort count or identity mismatch")
            cases += len(panel)
            start, end = origin + 7, origin + 14
            updates = {"selector": update_positions(start, end, config["selector_update_every_periods"]),
                       "controllers": update_positions(start, end, config["controller_recalibration_every_periods"])}
            for number, (cadence, horizon, window) in enumerate(scenarios, 1):
                windows = list(decision_windows(origin, len(columns), cadence, horizon, window,
                                                config["record_latency_periods"]))
                for key in panel.index:
                    key = key if isinstance(key, tuple) else (key,)
                    identity = dict(zip(panel.index.names, map(str, key)))
                    case = json.dumps([dataset, origin, *map(str, key)], ensure_ascii=False, separators=(",", ":"))
                    for slot in windows:
                        row = {"case_id": case, "dataset": dataset, "origin": origin,
                               "store_id": identity["store_id"], "item_id": identity["item_id"],
                               "scenario": number, "review_cadence": cadence,
                               "planning_horizon": horizon, "history_mode": window["mode"],
                               "history_periods": window.get("periods", ""), **slot,
                               "decision_period": columns[slot["decision_position"]],
                               "history_last_period": columns[slot["available_last_position"]],
                               "selector_update_positions": json.dumps(updates["selector"]),
                               "controller_update_positions": json.dumps(updates["controllers"]),
                               "state_action": "initialize_common_once" if slot["decision_position"] == start else "carry_strategy_state",
                               "scientific_execution_status": "NOT_READY"}
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        rows += 1
            panel_info.append({"origin": origin, "series": len(panel), "periods": len(columns),
                               "panel": str(path.relative_to(ROOT)), "sha256": meta["panel_sha256"],
                               "evaluation_start": start, "evaluation_end_exclusive": end,
                               "update_positions": updates})
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": dataset,
        "scope": "Chronological planning only; no training, simulation, policy recovery or queue mutation",
        "scientific_execution_status": "NOT_READY", "sequential_validation_performed": False,
        "config": config, "config_sha256": sha(config_path), "planner_sha256": sha(Path(__file__)),
        "source_plan": str(source_path.relative_to(ROOT)), "source_plan_sha256": sha(source_path),
        "panels": panel_info, "cases": cases, "scenarios": len(scenarios), "decision_rows": rows,
        "agenda_sha256": sha(schedule_path),
        "position_convention": "zero-based, exclusive interval ends; decision t sees completed records strictly before t minus declared latency",
        "horizon_rule": "Requested planning horizon remains full; future predictions required. Only realized outcomes inside the seven-period block can be scored.",
        "update_contract": "Any model update must run at its own scheduled position using only available records and matured labels. Scheduling is not evidence that these labels/artifacts exist.",
        "state_contract": config["state_contract"],
        "execution_blockers": config["required_execution_artifacts"],
        "limitations": ["Historical cohorts are not independent confirmatory reserves.",
            "Overlapping Bahia origins are separate episodes; their repeated periods are dependent.",
            "Each origin is a separate episode; no cross-origin state continuity is claimed.",
            "Seven-period aggregate KPIs cannot reconstruct sequential actions or trajectories.",
            "Record availability is assumed; profile/selector updates remain unevaluated.",
            "Commercial-cycle labels are not dates; M5 block labels are synthetic and native blocks span 21 days.",
            "Costs are scenario units and fixed-order cost never scales with review cadence."]}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"dataset": dataset, "cases": cases, "scenarios": len(scenarios),
            "decision_rows": rows, "scientific_execution_status": "NOT_READY", "output": str(target)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("BA", "M5", "both"), default="both")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    datasets = ("BA", "M5") if args.dataset == "both" else (args.dataset,)
    for dataset in datasets:
        print(json.dumps(plan_dataset(config, dataset, args.config), ensure_ascii=False))


if __name__ == "__main__":
    main()
