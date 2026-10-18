"""Auditable additional-context rounds; reused controllers, grouped PSE fits only.

Independent from simulation. BA availability is author-declared at period level;
M5 uses completed historical days/weeks and invariant benchmark departments.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiment/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_extension as context
import numpy as np
import pandas as pd
from aipe_context.contracts import CASE, Feature
from aipe_context.selection import fit_context_pse, measured, compare, validation_matrix

BA_DECLARATION = 'Autor em 17/10/2026: "sim são do mesmo periodo, eu coletei direto do banco".'


def now():
    return datetime.now(timezone.utc).isoformat()


def feature(name, kind, source, evidence, level="series"):
    return dict(name=name, kind=kind, mode="candidate", level=level,
                description=name.replace("_", " "), source=source,
                availability_verified=True, availability_evidence=evidence)


def prepare(dataset, destination):
    """Build inputs only after its source has fully finished; no learning."""
    config, ds, _ = context.load_config(context.DEFAULT_CONFIG, dataset)
    result = context.prepare(ds, destination)
    target = context.resolve(destination)
    base = pd.read_csv(target / "features.csv", dtype={c: str for c in CASE[1:]})
    run = context.resolve(ds["run"])
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    source_hashes = {}
    timing = []
    if dataset == "BA":
        import duckdb
        files = sorted((ROOT / "data/source/vendas/uf=BA").glob("*.parquet"))
        source_hashes = {str(p): context.sha(p) for p in files}
        con = duckdb.connect()
        con.execute("SET threads=1")
        con.execute("SET memory_limit='512MB'")
        keys = base[["store_id", "item_id"]].drop_duplicates()
        con.register("cohort", keys)
        pattern = (ROOT / "data/source/vendas/uf=BA/*.parquet").as_posix()
        attrs = con.execute(f"""SELECT DISTINCT CAST(revendedor_cod AS VARCHAR) store_id,
            CAST(produto_cod AS VARCHAR) item_id, idade, segmento, filial
            FROM read_parquet('{pattern}', hive_partitioning=false)
            SEMI JOIN cohort ON CAST(revendedor_cod AS VARCHAR)=cohort.store_id
                AND CAST(produto_cod AS VARCHAR)=cohort.item_id""").df()
        con.close()
        if attrs.duplicated(["store_id", "item_id"]).any():
            raise ValueError("Conflicting BA metadata snapshots; cannot infer historical evolution")
        ready = base.merge(attrs, on=["store_id", "item_id"], how="left", validate="many_to_one")
        if ready[["idade", "segmento", "filial"]].isna().any().any():
            raise ValueError("Missing BA metadata")
        evidence = BA_DECLARATION + " Mesma fotografia do período nas fontes; sem datas civis inventadas ou histórico de alterações por campo."
        specs = [feature(n, "numeric" if n == "idade" else "categorical",
                         "data/source/vendas/uf=BA; cadastro anexado às vendas", evidence, "store")
                 for n in ("idade", "segmento", "filial")]
        for meta in plan["panel_metadata"]:
            panel = pd.read_parquet(run / meta["panel"])
            origin, valid = int(meta["origin"]), int(meta["validation_cycles"])
            timing.append(dict(origin=origin, prefix_last_cycle=str(panel.columns[origin - 1]),
                validation_last_cycle=str(panel.columns[origin + valid - 1]),
                decision_first_test_cycle=str(panel.columns[origin + valid]),
                availability_contract="Author-declared same-period database snapshot; no field-level change log"))
        limitation = "assumption_declared_snapshot_temporal_validity: BA contextual availability is author-declared for the period, not independently reconstructed field by field. Age as recorded, not reconstructed birth dates. No socioeconomic income interpretation."
    else:
        calendar_path = ROOT / "data/m5_walmart/calendar.csv"
        prices_path = ROOT / "data/m5_walmart/sell_prices.csv"
        sales_path = ROOT / "data/m5_walmart/sales_train_evaluation.csv"
        source_hashes = {str(p): context.sha(p) for p in (calendar_path, prices_path, sales_path)}
        frozen = {str(k).replace("\\", "/"): v for k, v in plan["panel_metadata"][0]["source_files"].items()}
        for path, digest in source_hashes.items():
            if frozen[Path(path).relative_to(ROOT).as_posix()] != digest:
                raise ValueError("M5 source differs from frozen controller source")
        cal = pd.read_csv(calendar_path, parse_dates=["date"])
        if not cal.date.diff().dropna().eq(pd.Timedelta(days=1)).all() or str(cal.date.iloc[0].date()) != "2011-01-29":
            raise ValueError("M5 calendar must map sequentially from d_1 on 2011-01-29")
        metadata = pd.read_csv(sales_path, usecols=["item_id", "store_id", "state_id", "dept_id"], dtype=str)
        if metadata.duplicated(["store_id", "item_id"]).any():
            raise ValueError("Duplicate M5 metadata")
        ready = base.merge(metadata, on=["store_id", "item_id"], how="left", validate="many_to_one")
        if ready.state_id.isna().any() or not ready.state_id.eq(ready.warehouse).all():
            raise ValueError("State/identity mismatch")
        ready = ready.rename(columns={"dept_id": "product_department"}).drop(columns="state_id")
        # Read only relevant series; the price source is not used in the simulator's costs.
        keys = pd.MultiIndex.from_frame(base[["store_id", "item_id"]].drop_duplicates())
        chunks = []
        for chunk in pd.read_csv(prices_path, dtype={"store_id": str, "item_id": str}, chunksize=150000):
            chunks.append(chunk[pd.MultiIndex.from_frame(chunk[["store_id", "item_id"]]).isin(keys)])
        prices = pd.concat(chunks, ignore_index=True)
        week_end = cal.groupby("wm_yr_wk").date.max()
        prices["week_end"] = prices.wm_yr_wk.map(week_end)
        if prices.week_end.isna().any():
            raise ValueError("Missing week-to-date mapping")
        for meta in plan["panel_metadata"]:
            origin, valid = int(meta["origin"]), int(meta["validation_cycles"])
            # Input window ends with calibration, deliberately before validation.
            end = cal.date.iloc[origin * 21 - 1]
            begin = cal.date.iloc[(origin - 1) * 21]
            decision = cal.date.iloc[(origin + valid) * 21]
            mask = ready.origin.eq(origin)
            prior = prices[prices.week_end.le(end) & prices.week_end.gt(end - pd.Timedelta(days=21))]
            means = prior.groupby(["store_id", "item_id"]).sell_price.mean()
            ready.loc[mask, "past_unit_price"] = [means.get((s, i), np.nan)
                for s, i in ready.loc[mask, ["store_id", "item_id"]].itertuples(index=False, name=None)]
            window = cal.iloc[(origin - 1) * 21:origin * 21]
            for state in ready.loc[mask, "warehouse"].unique():
                ready.loc[mask & ready.warehouse.eq(state), "past_snap_fraction"] = float(window["snap_" + state].mean())
            ready.loc[mask, "past_event_count"] = int(window[["event_name_1", "event_name_2"]].notna().sum().sum())
            timing.append(dict(origin=origin, past_window_start=str(begin.date()), past_window_end=str(end.date()),
                decision_at=decision.isoformat() + "Z", validation_available_at=decision.isoformat() + "Z",
                price_rule="Only weeks fully ended before calibration end; mean of last 21 days' completed weeks",
                calendar_rule="Only past completed 21-day calibration block, no future calendars/prices"))
        evidence = "Frozen M5 benchmark sources; d_1=2011-01-29, 21 consecutive daily observations per cycle. Inputs end at calibration prefix, validation results are available only at test start."
        specs = [feature("product_department", "categorical", str(sales_path), evidence + " Department is invariant benchmark item taxonomy."),
                 feature("past_unit_price", "numeric", str(prices_path), evidence + " Historical complete-week prices only; no monetary reinterpretation of simulation costs."),
                 feature("past_snap_fraction", "numeric", str(calendar_path), evidence + " State-matched past SNAP purchase eligibility indicators.", "warehouse"),
                 feature("past_event_count", "numeric", str(calendar_path), evidence + " Past calendar events, constant by origin; expected selection rejection.", "warehouse")]
        limitation = "Past calendar covariates and invariant product metadata; no forecast of future prices, no claim that simulated costs are actual financial savings."
    ready.to_csv(target / "context_features.csv", index=False)
    write = context.write_json
    code_paths = [Path(__file__), Path(context.__file__), ROOT / "experiment/src/aipe_context/selection.py", ROOT / "experiment/src/aipe_context/contracts.py"]
    protocol = {"dataset": dataset, "created_utc": now(), "feature_contracts": specs,
        "selection": config["selection"], "models": ["class", "consequences"],
        "comparisons": ["base", "fixed", "all_admissible", "selected", "guarded"],
        "availability": timing, "availability_limitations": limitation,
        "source_context_sha256": source_hashes, "controller_fits": 0,
        "context_code_sha256": {str(p.resolve()): context.sha(p) for p in code_paths},
        "scope": "Development evaluation; prior test results already observed, independent confirmation still required",
        "context_features_sha256": context.sha(target / "context_features.csv"),
        "prepared_manifest_sha256": context.sha(target / "manifest.json")}
    write(target / "context_protocol.json", protocol)
    return result | {"context_features": len(specs), "historical_contract": limitation, "models_fitted": 0}


_WORK = None


def initialize(prepared, output):
    global _WORK
    if sys.platform == "win32":
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    p = Path(prepared)
    types = {c: str for c in CASE[1:]}
    features = pd.read_csv(p / "context_features.csv", dtype=types)
    validation = pd.read_csv(p / "validation.csv", dtype=types)
    features, validation = validation_matrix(features, validation)
    protocol = json.loads((p / "context_protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
    _WORK = features, validation, protocol, manifest, Path(output)


def fold_job(task):
    import joblib
    model, origin, warehouse, store = task
    features, validation, protocol, manifest, out = _WORK
    name = hashlib.sha256(json.dumps(task).encode()).hexdigest()[:20]
    directory = out / "folds" / name
    directory.mkdir(parents=True, exist_ok=True)
    completion = directory / "completed.json"
    if completion.exists():
        previous = json.loads(completion.read_text())
        if all(context.sha(directory / f) == h for f, h in previous["files"].items()):
            return previous
        raise ValueError("Damaged checkpoint: " + name)
    started = time.monotonic()
    cohort = features[features.origin.eq(origin)]
    held_mask = cohort.warehouse.eq(warehouse) & cohort.store_id.eq(store)
    train, held = cohort[~held_mask], cohort[held_mask]
    train_keys = pd.MultiIndex.from_frame(train[CASE])
    outcomes = validation[pd.MultiIndex.from_frame(validation[CASE]).isin(train_keys)]
    options = protocol["selection"] | {"model": model}
    specs = [Feature(**s) for s in protocol["feature_contracts"]]
    selected = fit_context_pse(train, outcomes, specs, options, manifest["base_numeric"], manifest["base_categorical"])
    # Same learner/hyperparameters; only all attributes are mandatory in this comparator.
    all_specs = [Feature(**(s | {"mode": "fixed"})) for s in protocol["feature_contracts"] if s["mode"] != "excluded"]
    all_options = options | {"max_context_features": max(options["max_context_features"], len(all_specs))}
    all_fit = fit_context_pse(train, outcomes, all_specs, all_options, manifest["base_numeric"], manifest["base_categorical"])
    predictions = selected.predict(held)
    all_pred = all_fit.fixed.predict(held).assign(strategy="PSE_context_all")
    predictions = pd.concat([predictions, all_pred], ignore_index=True).assign(model=model)
    predictions.to_csv(directory / "predictions.csv", index=False)
    context.write_json(directory / "audit.json", {"model": model, "origin": origin, "warehouse": warehouse,
        "held_store": store, "selected": selected.audit, "all": all_fit.audit})
    joblib.dump({"selected": selected, "all": all_fit}, directory / "models.joblib")
    done = {"task": task, "fold": name, "seconds": time.monotonic() - started,
            "files": {n: context.sha(directory / n) for n in ("predictions.csv", "audit.json", "models.joblib")}}
    context.write_json(completion, done)
    return done


def run(prepared, destination, workers):
    prepared, target = context.resolve(prepared), context.resolve(destination)
    protocol = json.loads((prepared / "context_protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    context.completed(context.resolve(manifest["source_run"]))
    for name, digest in manifest["file_sha256"].items():
        if context.sha(prepared / name) != digest:
            raise ValueError("Changed prepared matrix")
    for path, digest in manifest["source_sha256"].items():
        if context.sha(path) != digest:
            raise ValueError("Changed source snapshot; prepare a new study")
    for path, digest in protocol["context_code_sha256"].items():
        if context.sha(path) != digest:
            raise ValueError("Changed contextual implementation; prepare a new study")
    for path, digest in protocol["source_context_sha256"].items():
        if context.sha(path) != digest:
            raise ValueError("Changed contextual source")
    if context.sha(prepared / "context_features.csv") != protocol["context_features_sha256"] or context.sha(prepared / "manifest.json") != protocol["prepared_manifest_sha256"]:
        raise ValueError("Changed prepared protocol")
    if not target.is_relative_to(context.OUTPUT_ROOT.resolve()) or target == context.OUTPUT_ROOT.resolve():
        raise ValueError("Output must be below isolated context root")
    target.mkdir(parents=True, exist_ok=True)
    frozen = target / "protocol_before_predictions.json"
    if frozen.exists():
        if json.loads(frozen.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Cannot resume with changed protocol")
    else:
        context.write_json(frozen, protocol)
    features = pd.read_csv(prepared / "context_features.csv", dtype={c: str for c in CASE[1:]})
    tasks = [(model, int(o), str(w), str(s)) for model in protocol["models"]
             for o, w, s in features[["origin", "warehouse", "store_id"]].drop_duplicates().itertuples(index=False, name=None)]
    status = dict(state="running", dataset=protocol["dataset"], pid=os.getpid(), workers=workers,
                  total_folds=len(tasks), completed_folds=0, controller_fits=0, started_utc=now())
    context.write_json(target / "status.json", status)
    results = []
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(str(prepared), str(target))) as pool:
            pending = {pool.submit(fold_job, task): task for task in tasks}
            for future in as_completed(pending):
                results.append(future.result())
                status.update(completed_folds=len(results), updated_utc=now(), last_fold_seconds=results[-1]["seconds"])
                context.write_json(target / "status.json", status)
        predictions = pd.concat([pd.read_csv(target / "folds" / r["fold"] / "predictions.csv",
            dtype={c: str for c in CASE[1:]}) for r in results], ignore_index=True)
        predictions = predictions.sort_values(["model", "strategy", *CASE]).reset_index(drop=True)
        predictions.to_csv(target / "predictions_frozen_before_test.csv", index=False)
        prediction_hash = context.sha(target / "predictions_frozen_before_test.csv")
        # First parse of test outcomes occurs only after every outer recommendation is frozen.
        test = pd.read_csv(prepared / "test.csv", dtype={c: str for c in CASE[1:]})
        _, test_matrix = validation_matrix(features, test, expected_block="test")
        scored = pd.concat([measured(group, test_matrix, protocol["selection"]["service_target"])
            for _, group in predictions.groupby(["model", "strategy"])], ignore_index=True)
        scored.to_csv(target / "case_metrics.csv", index=False)
        scored.groupby(["model", "strategy"]).agg(cases=("TIC", "size"), total_cost=("TIC", "sum"),
            mean_cost=("TIC", "mean"), mean_service=("NS", "mean"), defined_service=("NS", "count"),
            failed_service=("failed", "sum")).to_csv(target / "summary.csv")
        comparisons = {}
        for model, rows in scored.groupby("model"):
            baseline = rows[rows.strategy.eq("PSE_base")]
            comparisons[model] = {}
            for strategy, group in rows.groupby("strategy"):
                value = compare(group, baseline, protocol["selection"]["service_target"], protocol["selection"])
                value["meets_reported_thresholds"] = value.pop("accepted")
                comparisons[model][strategy] = value
        context.write_json(target / "comparison.json", comparisons)
        if context.sha(target / "predictions_frozen_before_test.csv") != prediction_hash:
            raise ValueError("Frozen predictions changed")
        status.update(state="completed", updated_utc=now(), cases=len(features), predictions_sha256=prediction_hash)
        context.write_json(target / "status.json", status)
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), updated_utc=now())
        context.write_json(target / "status.json", status)
        raise
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--dataset", choices=["BA", "M5"])
    parser.add_argument("--prepared")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8; keep combined datasets within CPU budget")
    if sys.platform == "win32":
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if args.command == "prepare":
        if not args.dataset:
            parser.error("--dataset required")
        result = prepare(args.dataset, args.output)
    else:
        if not args.prepared:
            parser.error("--prepared required")
        result = run(args.prepared, args.output, args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
