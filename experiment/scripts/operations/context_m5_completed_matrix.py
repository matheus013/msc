"""Explicit M5 controller-matrix frontend, independent from selector/oracle stage.

Does not alter canonical status or existing context source files. Dependency
injection is process-local: actual source status is preserved, never forged.
"""
from pathlib import Path
import argparse
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_rounds as rounds
import context_extension as context
import pandas as pd
from aipe_context.contracts import CASE, normalize_cases

STOCHASTIC = {"GA", "PPO", "DQN", "GA-DQN"}
DETERMINISTIC = {"EOQCycle", "sSCycle", "CappedBaseStock"}


def completed_matrix(run):
    """Require every frozen case × policy × seed × block, not just counts."""
    run = Path(run)
    before = {n: context.sha(run / n) for n in ("plan.json", "kpis.csv")}
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    if status.get("dataset") != "M5" or status.get("state") not in ("controllers_completed", "completed"):
        raise ValueError("M5 controllers have not finished")
    if status.get("planned_fits") != 79890 or status.get("completed_fits") != 79890 or plan.get("planned_fits") != 79890:
        raise ValueError("Expected frozen M5 target of 79,890 completed controller fits")
    if set(plan["policies"]) != STOCHASTIC | DETERMINISTIC or set(plan["seeds"]) != {11, 29, 47}:
        raise ValueError("Unexpected policy/seed contract")
    expected = set()
    for meta in plan["panel_metadata"]:
        path = run / meta["panel"]
        if context.sha(path) != meta["panel_sha256"]:
            raise ValueError("Changed frozen M5 panel")
        panel = pd.read_parquet(path)
        if list(panel.index.names) != CASE[1:] or panel.index.has_duplicates:
            raise ValueError("Invalid frozen panel identities")
        for warehouse, store, item in panel.index:
            for policy in plan["policies"]:
                seeds = plan["seeds"] if policy in STOCHASTIC else [0]
                for seed in seeds:
                    for block in ("validation", "test"):
                        expected.add((int(meta["origin"]), str(warehouse), str(store), str(item), policy, int(seed), block))
    seen = set()
    with (run / "kpis.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["origin"]), row["warehouse"], row["store_id"], row["item_id"], row["policy"], int(row["seed"]), row["block"])
            if row["dataset"] != "M5" or key not in expected or key in seen:
                raise ValueError("Unexpected or duplicate M5 controller outcome")
            seen.add(key)
    if seen != expected or len(seen) != 159780:
        raise ValueError("Incomplete controller matrix")
    if any(context.sha(run / name) != digest for name, digest in before.items()):
        raise ValueError("Controller matrix changed during audit")
    return status


def prepare_matrix(ds, destination):
    """Export immutable controller evidence, preserving real source phase."""
    run = context.resolve(ds["run"])
    status = completed_matrix(run)
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    status_digest = context.sha(run / "status.json")
    hashes = {str(run / name): context.sha(run / name) for name in ("plan.json", "kpis.csv")}
    dependency_audit = context.verify_feature_dependencies(plan)
    from simulation.pipelines.demand_profiling.nodes import _series_features
    from simulation.research.protocol import profile_of
    from simulation.pipelines.policy_selection.nodes import FEATURE_COLS
    features = []
    for meta in plan["panel_metadata"]:
        path = run / meta["panel"]
        if context.sha(path) != meta["panel_sha256"]:
            raise ValueError("Changed frozen M5 panel")
        hashes[str(path)] = context.sha(path)
        panel = pd.read_parquet(path)
        for key, values in zip(panel.index, panel.to_numpy(float)):
            prefix = values[:int(meta["origin"])]
            features.append(dict(zip(CASE, [meta["origin"], *key])) | _series_features(prefix, 17)
                            | {"operational_profile": profile_of(prefix)})
    features = normalize_cases(pd.DataFrame(features))
    export = {b: [] for b in ("validation", "test")}
    names = CASE + ["policy", "seed", "block", "TIC", "NS"]
    with (run / "kpis.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            export[row["block"]].append({name: row[name] for name in names})
    if any(context.sha(path) != digest for path, digest in hashes.items()):
        raise ValueError("Controller source changed during export")
    if context.verify_feature_dependencies(plan) != dependency_audit:
        raise ValueError("Feature dependencies changed during export")
    target = context.new_output(destination)
    features.to_csv(target / "features.csv", index=False)
    for block, rows in export.items():
        pd.DataFrame(rows).to_csv(target / (block + ".csv"), index=False)
    manifest = {"source_run": str(run), "source_sha256": hashes, "dependency_audit": dependency_audit,
        "base_numeric": FEATURE_COLS, "base_categorical": ["operational_profile"], "cases": len(features),
        "file_sha256": {n: context.sha(target / n) for n in ("features.csv", "validation.csv", "test.csv")},
        "source_stage": status["state"], "source_status_captured": status,
        "source_status_sha256_at_capture": status_digest,
        "immutable_dependency_note": "Canonical selector/oracle evaluation may change status.json; plan, panels and full controller matrix remain hash-guarded.",
        "matrix_audit": {"planned_fits": 79890, "completed_fits": 79890, "expected_outcome_rows": 159780,
                         "verified": "exact case × policy × expected seed × validation/test block"},
        "status": "prepared_not_evaluated"}
    context.write_json(target / "manifest.json", manifest)
    return {"prepared": str(target), "cases": len(features), "fitted_models": 0, "source_stage": status["state"]}


def install_process_local_frontend():
    # These audited guards affect this new command only, never source code on disk
    # or the BA process, and return the source's actual status without modification.
    context.prepare = prepare_matrix
    context.completed = completed_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--prepared")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Keep M5 within four workers while BA uses four")
    if sys.platform == "win32":
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    install_process_local_frontend()
    if args.command == "prepare":
        result = rounds.prepare("M5", args.output)
        target = context.resolve(args.output)
        protocol = json.loads((target / "context_protocol.json").read_text(encoding="utf-8"))
        protocol["context_code_sha256"][str(Path(__file__).resolve())] = context.sha(Path(__file__))
        context.write_json(target / "context_protocol.json", protocol)
    else:
        if not args.prepared:
            parser.error("--prepared required")
        result = rounds.run(args.prepared, args.output, args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
