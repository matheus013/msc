"""Optional context: plan now; prepare/evaluate completed matrices explicitly later.

Never imported by the live runner. No controller fitting or simulation entrypoint.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"  # Only this process; never changes the live runner.
sys.path.insert(0, str(ROOT / "experiment/src"))
DEFAULT_CONFIG = ROOT / "experiment/conf/aipe_context_extension_v1.json"
OUTPUT_ROOT = ROOT / "experiment/data/context_extension"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def resolve(path):
    p = Path(path)
    return (p if p.is_absolute() else ROOT / p).resolve()


def load_config(path, dataset):
    from aipe_context.contracts import validate_config
    config = json.loads(resolve(path).read_text(encoding="utf-8"))
    ds = config["datasets"][dataset]
    specs = validate_config({"enabled": config["enabled"], "selection": config["selection"], "features": ds["features"]})
    return config, ds, specs


def completed(run):
    status = json.loads((Path(run) / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed" or status.get("completed_fits") != status.get("planned_fits") or not status.get("planned_fits"):
        raise ValueError("Source simulation/evaluation must be completed; checkpoints alone are insufficient")
    return status


def validated_cutoffs(cutoffs, features):
    from aipe_context.contracts import CASE, normalize_cases, timestamps
    cutoffs = normalize_cases(cutoffs)
    if set(cutoffs[CASE].itertuples(index=False, name=None)) != set(features[CASE].itertuples(index=False, name=None)):
        raise ValueError("Cutoff and feature case sets differ")
    for field in ("decision_at", "validation_available_at"):
        if field not in cutoffs:
            raise ValueError("Explicit historical timing required: " + field)
        cutoffs[field] = timestamps(cutoffs[field], field)
    for _, cohort in cutoffs.groupby("origin"):
        if cohort.validation_available_at.max() > cohort.decision_at.min():
            raise ValueError("Training labels are not yet available to all decisions at this origin")
    return cutoffs


def plan(config, ds, specs):
    run = resolve(ds["run"])
    state = json.loads((run / "status.json").read_text(encoding="utf-8"))["state"] if (run / "status.json").exists() else "missing"
    selected = {mode: [s.name for s in specs if s.mode == mode] for mode in ("fixed", "candidate", "excluded")}
    blockers = []
    if not config["enabled"]:
        blockers.append("module_disabled")
    if state != "completed":
        blockers.append("source_run_not_completed")
    for name, other in config["datasets"].items():
        path = resolve(other["run"]) / "status.json"
        if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get("state") != "completed":
            blockers.append("waiting_for_run:" + name)
    for name in ("cutoffs_file", "records_file"):
        if not ds.get(name) or not resolve(ds[name]).is_file():
            blockers.append(name + "_required")
    for s in specs:
        if s.mode != "excluded" and not s.availability_verified:
            blockers.append("unverified_availability:" + s.name)
    return {"source_run": str(run), "source_state": state, "choices": selected,
            "ready_for_evaluation": not blockers, "blockers": blockers,
            "selection": config["selection"], "starts_training_or_simulation": False}


def new_output(path):
    target = resolve(path)
    if not target.is_relative_to(OUTPUT_ROOT.resolve()) or target == OUTPUT_ROOT.resolve():
        raise ValueError("Outputs must use a new subdirectory of experiment/data/context_extension")
    target.mkdir(parents=True, exist_ok=False)
    return target


def configure(path, dataset, choices, enabled=None):
    """Change only the optional registry; preserve an exact prior configuration."""
    from aipe_context.contracts import validate_config
    path = resolve(path)
    before_hash = sha(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    rows = config["datasets"][dataset]["features"]
    known = {r["name"] for r in rows}
    supplied = [name for names in choices.values() for name in names]
    if len(supplied) != len(set(supplied)) or not set(supplied) <= known:
        raise ValueError("Unknown feature or conflicting manager choices")
    if not supplied and enabled is None:
        raise ValueError("Specify feature choices or enable/disable")
    for mode, names in choices.items():
        for row in rows:
            if row["name"] in names:
                row["mode"] = mode
    if enabled is not None:
        config["enabled"] = enabled
    for ds in config["datasets"].values():
        validate_config({"enabled": config["enabled"], "selection": config["selection"], "features": ds["features"]})
    history = OUTPUT_ROOT / "config_history"
    history.mkdir(parents=True, exist_ok=True)
    backup = history / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + before_hash[:12] + ".json")
    backup.write_bytes(path.read_bytes())
    if sha(path) != before_hash:
        raise ValueError("Configuration changed concurrently")
    temporary = path.with_name(path.name + ".context.tmp")
    write_json(temporary, config)
    temporary.replace(path)
    return {"config": str(path), "backup": str(backup), "enabled": config["enabled"],
            "mode_changes": choices, "does_not_activate_live_runner": True}


FEATURE_DEPENDENCIES = (
    "experiment/src/simulation/pipelines/demand_profiling/nodes.py",
    "experiment/src/simulation/research/protocol.py",
    "experiment/src/simulation/pipelines/policy_selection/nodes.py",
)


def verify_feature_dependencies(source_plan):
    """Guard reconstructed features and disclose unrelated source revisions."""
    frozen = {str(key).replace("\\", "/"): value for key, value in source_plan["code_sha256"].items()}
    current = {rel: sha(ROOT / rel) if (ROOT / rel).is_file() else None for rel in frozen}
    for rel in FEATURE_DEPENDENCIES:
        if rel not in frozen or current[rel] != frozen[rel]:
            raise ValueError("Feature dependency changed or missing from frozen plan: " + rel)
    return {"required_feature_dependencies": {rel: current[rel] for rel in FEATURE_DEPENDENCIES},
            "other_source_differences": {rel: {"frozen": frozen[rel], "current": current[rel]}
                                         for rel in frozen if current[rel] != frozen[rel]}}


def prepare(ds, destination):
    """Export existing metrics and prefix features; do not infer historic context."""
    import pandas as pd
    from aipe_context.contracts import CASE, normalize_cases
    run = resolve(ds["run"])
    completed(run)
    source_files = [run / name for name in ("plan.json", "status.json", "kpis.csv")]
    hashes = {str(p): sha(p) for p in source_files}
    source_plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    dependency_audit = verify_feature_dependencies(source_plan)
    from simulation.pipelines.demand_profiling.nodes import _series_features
    from simulation.research.protocol import profile_of
    from simulation.pipelines.policy_selection.nodes import FEATURE_COLS
    rows = []
    for meta in source_plan["panel_metadata"]:
        panel_path = run / meta["panel"]
        if sha(panel_path) != meta["panel_sha256"]:
            raise ValueError("Panel differs from frozen source plan")
        hashes[str(panel_path)] = sha(panel_path)
        panel = pd.read_parquet(panel_path)
        if list(panel.index.names) != CASE[1:]:
            raise ValueError("Unexpected panel identity contract")
        for key, values in zip(panel.index, panel.to_numpy(float)):
            prefix = values[:meta["origin"]]
            rows.append(dict(zip(CASE, [meta["origin"], *key])) | _series_features(prefix, 17) | {"operational_profile": profile_of(prefix)})
    features = normalize_cases(pd.DataFrame(rows))
    # Read metrics once. Test rows are streamed to a separate file, not supplied to learners.
    exported = {block: [] for block in ("validation", "test")}
    with (run / "kpis.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = CASE + ["policy", "seed", "block", "TIC", "NS"]
        if not set(names) <= set(reader.fieldnames):
            raise ValueError("Missing source matrix fields")
        for row in reader:
            if row["block"] in exported:
                exported[row["block"]].append({name: row[name] for name in names})
    for rel, digest in hashes.items():
        if sha(rel) != digest:
            raise ValueError("Source changed during export; no prepared study produced")
    if verify_feature_dependencies(source_plan) != dependency_audit:
        raise ValueError("Source code changed during feature export")
    target = new_output(destination)
    features.to_csv(target / "features.csv", index=False)
    for block, data in exported.items():
        pd.DataFrame(data).to_csv(target / (block + ".csv"), index=False)
    cutoffs = features[CASE].copy()
    cutoffs["decision_at"] = ""
    cutoffs["validation_available_at"] = ""
    cutoffs.to_csv(target / "cutoffs_template.csv", index=False)
    pd.DataFrame(columns=["warehouse", "store_id", "item_id", "feature", "value", "available_at"]).to_csv(target / "context_template.csv", index=False)
    files = [target / n for n in ("features.csv", "validation.csv", "test.csv")]
    manifest = {"source_run": str(run), "source_sha256": hashes,
                "dependency_audit": dependency_audit,
                "base_numeric": FEATURE_COLS, "base_categorical": ["operational_profile"],
                "file_sha256": {p.name: sha(p) for p in files},
                "cases": len(features), "status": "prepared_not_evaluated",
                "cutoff_rule": "Supply documented recommendation timestamps; do not infer them from synthetic cycle labels or extraction date."}
    write_json(target / "manifest.json", manifest)
    return {"prepared": str(target), "cases": len(features), "fitted_models": 0}


def evaluate(config, ds, specs, prepared, destination):
    import joblib
    import pandas as pd
    from aipe_context.contracts import CASE, assemble_context, normalize_cases
    from aipe_context.selection import evaluate_context_pse, validation_matrix, measured, compare
    if not config["enabled"]:
        raise ValueError("Context module is disabled")
    # The user requested experiments only after the current simulations finish.
    for other in config["datasets"].values():
        completed(resolve(other["run"]))
    readiness = plan(config, ds, specs)
    if not readiness["ready_for_evaluation"]:
        raise ValueError("Unresolved context requirements: " + ", ".join(readiness["blockers"]))
    source = resolve(prepared)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if resolve(manifest["source_run"]) != resolve(ds["run"]):
        raise ValueError("Prepared matrix belongs to another run")
    for name, digest in manifest["file_sha256"].items():
        if sha(source / name) != digest:
            raise ValueError("Prepared input changed: " + name)
    for path, digest in manifest["source_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Prepared source snapshot changed; prepare again")
    types = {c: str for c in CASE[1:]}
    features = normalize_cases(pd.read_csv(source / "features.csv", dtype=types))
    validation = pd.read_csv(source / "validation.csv", dtype=types)
    cutoffs_path, records_path = resolve(ds["cutoffs_file"]), resolve(ds["records_file"])
    context_hashes = {str(p): sha(p) for p in (cutoffs_path, records_path)}
    cutoffs = validated_cutoffs(pd.read_csv(cutoffs_path, dtype=types), features)
    active = {s.name for s in specs if s.mode != "excluded"}
    chunks, retained = [], 0
    allowed = {*CASE[1:], "feature", "value", "available_at"}
    for chunk in pd.read_csv(records_path, dtype={c: str for c in [*CASE[1:], "feature", "value"]},
                             usecols=lambda c: c in allowed, chunksize=50000):
        if "feature" not in chunk:
            raise ValueError("Context feature column required")
        chunk = chunk[chunk.feature.isin(active)]
        retained += len(chunk)
        if retained > config["selection"]["max_source_records"]:
            raise ValueError("Selected context source exceeds the record budget; aggregate upstream")
        chunks.append(chunk)
    records = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=sorted(allowed))
    extra, availability = assemble_context(cutoffs, records, specs)
    features = features.merge(extra, on=CASE, validate="one_to_one")
    features, _ = validation_matrix(features, validation)
    for path, digest in context_hashes.items():
        if sha(path) != digest:
            raise ValueError("Context inputs changed during reading")
    target = new_output(destination)
    (target / "models").mkdir()
    write_json(target / "protocol_before_predictions.json", {
        "scope": "Development evaluation, not independent confirmation",
        "outer": "per-origin leave-one-store-out; all products of store excluded together",
        "inner": "grouped selection using validation outcomes of outer training stores only",
        "config": config, "dataset_config": ds, "context_sha256": context_hashes,
        "prepared_manifest_sha256": sha(source / "manifest.json"), "controller_fits": 0,
        "created_utc": datetime.now(timezone.utc).isoformat()})
    write_json(target / "availability.json", availability)

    def artifact(origin, warehouse, store, fit):
        name = hashlib.sha256(json.dumps([int(origin), warehouse, store]).encode()).hexdigest()[:20]
        joblib.dump(fit, target / "models" / (name + ".joblib"))

    predictions, audits = evaluate_context_pse(features, validation, specs, config["selection"],
        manifest["base_numeric"], manifest["base_categorical"], artifact_callback=artifact)
    predictions.to_csv(target / "predictions_frozen_before_test.csv", index=False)
    predictions_hash = sha(target / "predictions_frozen_before_test.csv")
    write_json(target / "feature_selection.json", audits)
    # Test consequences enter only after recommendations and selected attributes are frozen.
    test = pd.read_csv(source / "test.csv", dtype=types)
    _, test_matrix = validation_matrix(features, test, expected_block="test")
    scored = pd.concat([measured(g, test_matrix, config["selection"]["service_target"])
                        for _, g in predictions.groupby("strategy")], ignore_index=True)
    scored.to_csv(target / "case_metrics.csv", index=False)
    scored.groupby("strategy").agg(cases=("TIC", "size"), total_cost=("TIC", "sum"),
        mean_cost=("TIC", "mean"), mean_service=("NS", "mean"), defined_service=("NS", "count"),
        failed_service=("failed", "sum")).to_csv(target / "summary.csv")
    baseline = scored[scored.strategy.eq("PSE_base")]
    comparisons = {name: compare(g, baseline, config["selection"]["service_target"], config["selection"])
                   for name, g in scored.groupby("strategy")}
    if sha(target / "predictions_frozen_before_test.csv") != predictions_hash:
        raise ValueError("Frozen recommendations changed during scoring")
    for value in comparisons.values():
        value["meets_reported_thresholds"] = value.pop("accepted")
    paired = scored.merge(baseline[CASE + ["TIC", "NS", "policy"]], on=CASE,
                          suffixes=("", "_base"), validate="many_to_one")
    paired["cost_difference"] = paired.TIC - paired.TIC_base
    paired["service_difference"] = paired.NS - paired.NS_base
    paired["intervened"] = paired.policy.ne(paired.policy_base)
    paired.to_csv(target / "paired_base.csv", index=False)
    write_json(target / "comparison.json", {"comparisons": comparisons,
        "interpretation": "Test comparisons describe observed effects; they do not retroactively select features or activate context."})
    counts = pd.DataFrame([{"origin": a["origin"], "warehouse": a["warehouse"], "held_store": a["held_store"],
        "feature": name, "forced": name in a["fixed_features"], "context_recommended": a["context_recommended"]}
        for a in audits for name in a["selected_features"]],
        columns=["origin", "warehouse", "held_store", "feature", "forced", "context_recommended"])
    counts.to_csv(target / "selected_features_by_fold.csv", index=False)
    write_json(target / "completed.json", {"state": "completed", "controller_fits": 0,
        "predictions_sha256": predictions_hash, "cases": len(features), "outer_folds": len(audits),
        "outputs_sha256": {str(p.relative_to(target)): sha(p) for p in sorted(target.rglob("*")) if p.is_file()}})
    return {"output": str(target), "state": "completed", "controller_fits": 0}


def main():
    if sys.platform == "win32":
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "configure", "prepare", "evaluate"])
    parser.add_argument("--dataset", choices=["BA", "M5"], required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", help="New directory under experiment/data/context_extension")
    parser.add_argument("--prepared", help="Prepared matrix directory for evaluation")
    parser.add_argument("--fixed", nargs="*", default=[])
    parser.add_argument("--candidates", nargs="*", default=[])
    parser.add_argument("--excluded", nargs="*", default=[])
    switches = parser.add_mutually_exclusive_group()
    switches.add_argument("--enable", action="store_true")
    switches.add_argument("--disable", action="store_true")
    args = parser.parse_args()
    config, ds, specs = load_config(args.config, args.dataset)
    if args.command == "plan":
        result = plan(config, ds, specs)
    elif args.command == "configure":
        result = configure(args.config, args.dataset,
            {"fixed": args.fixed, "candidate": args.candidates, "excluded": args.excluded},
            enabled=True if args.enable else False if args.disable else None)
    elif not args.output:
        parser.error("--output required; existing study outputs are never overwritten")
    elif args.command == "prepare":
        result = prepare(ds, args.output)
    else:
        if not args.prepared:
            parser.error("--prepared required")
        result = evaluate(config, ds, specs, args.prepared, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
