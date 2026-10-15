"""Reanalyse preserved KPI artifacts; never claim temporal validation.

Policy selection is cross-fitted across series/stores, while profiles and
historical simulator calibration remain retrospective. Bootstrap intervals
condition on the fitted selectors and resample stores, not deterministic
simulation repetitions. Run from any directory using the project Python.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiment/data/08_reporting/sca_article"
KEYS = ["warehouse", "store_id", "item_id"]


def choose(rows, alpha):
    means = rows.groupby("policy")[["TIC", "NS"]].mean()
    viable = means[means.NS >= alpha]
    if viable.empty:
        return means.sort_values(["NS", "TIC"], ascending=[False, True]).index[0], True
    return viable.sort_values(["TIC", "NS"], ascending=[True, False]).index[0], False


def analyse(df, alpha=0.7, bootstrap=5000):
    series = df[KEYS + ["operational_profile"]].drop_duplicates().sort_values(KEYS)
    global_policy, _ = choose(df, alpha)
    profile_policies = {g: choose(r, alpha)[0] for g, r in df.groupby("operational_profile")}
    selected = []
    for _, row in series.iterrows():
        match = np.logical_and.reduce([df[k].eq(row[k]) for k in KEYS])
        held = df[match].set_index("policy")
        train_series = df[~match]
        train_store = df[~df.store_id.eq(row.store_id)]
        record = {k: row[k] for k in KEYS + ["operational_profile"]}
        policies = {"EOQ": "EOQ", "SA": "SA", "global_in_sample": global_policy,
                    "profile_in_sample": profile_policies[row.operational_profile]}
        for tag, train in [("series", train_series), ("store", train_store)]:
            policies[f"global_leave_{tag}_out"], record[f"fallback_global_{tag}"] = choose(train, alpha)
            group_train = train[train.operational_profile.eq(row.operational_profile)]
            if group_train.empty:
                group_train = train
            policies[f"profile_leave_{tag}_out"], record[f"fallback_profile_{tag}"] = choose(group_train, alpha)
        for name, policy in policies.items():
            record[name + "_policy"] = policy
            for metric in ["TIC", "NS", "TR", "BE"]:
                record[name + "_" + metric] = float(held.loc[policy, metric])
        selected.append(record)
    selected = pd.DataFrame(selected)
    summary = []
    for name in policies:
        summary.append({"alpha": alpha, "strategy": name,
                        **{metric + "_mean": selected[name + "_" + metric].mean() for metric in ["TIC", "NS", "TR", "BE"]},
                        "TIC_total": selected[name + "_TIC"].sum(),
                        "series_below_alpha": int((selected[name + "_NS"] < alpha).sum()),
                        "policy_counts": selected[name + "_policy"].value_counts().to_dict()})
    contrasts = []
    rng = np.random.default_rng(20261014)
    stores = selected.store_id.unique()
    indices = {store: np.flatnonzero(selected.store_id.eq(store)) for store in stores}
    for treatment, baseline in [("profile_in_sample", "global_in_sample"),
                                ("profile_leave_series_out", "global_leave_series_out"),
                                ("profile_leave_store_out", "global_leave_store_out"),
                                ("profile_leave_store_out", "EOQ")]:
        a = selected[baseline + "_TIC"].to_numpy()
        b = selected[treatment + "_TIC"].to_numpy()
        service = (selected[treatment + "_NS"] - selected[baseline + "_NS"]).to_numpy()
        samples = []
        for _ in range(bootstrap):
            idx = np.concatenate([indices[s] for s in rng.choice(stores, len(stores), replace=True)])
            samples.append([100 * (a[idx].sum() - b[idx].sum()) / a[idx].sum(), service[idx].mean(), (a[idx] - b[idx]).mean()])
        ci = np.quantile(samples, [0.025, 0.975], axis=0)
        contrasts.append({"alpha": alpha, "treatment": treatment, "baseline": baseline,
                          "cost_reduction_percent": 100 * (a.sum() - b.sum()) / a.sum(),
                          "cost_reduction_percent_ci": ci[:, 0].tolist(),
                          "service_difference": service.mean(), "service_difference_ci": ci[:, 1].tolist(),
                          "mean_paired_cost_saving": (a - b).mean(), "mean_paired_cost_saving_ci": ci[:, 2].tolist(),
                          "bootstrap_unit": "store", "bootstrap_draws": bootstrap,
                          "qualification": "conditional on historical KPI matrix, profiles and fitted selectors; selectors not refitted in bootstrap"})
    return selected, summary, contrasts, profile_policies


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {"kpis": ROOT / "experiment/data/07_model_output/kpis.parquet",
             "profiles": ROOT / "experiment/data/04_feature/demand_profiles.parquet"}
    kpis = pd.read_parquet(paths["kpis"])
    profiles = pd.read_parquet(paths["profiles"])
    for frame in [kpis, profiles]:
        for key in KEYS:
            frame[key] = frame[key].astype(str)
    df = kpis.merge(profiles[KEYS + ["operational_profile"]], on=KEYS, validate="many_to_one")
    assert len(df) == len(kpis) and not df.duplicated(KEYS + ["policy"]).any()
    assert df.groupby(KEYS).policy.nunique().eq(df.policy.nunique()).all()
    summaries, contrasts, selections = [], [], {}
    for alpha in [0.70, 0.80, 0.85, 0.90, 0.95]:
        selected, rows, comparisons, groups = analyse(df, alpha)
        selected.to_csv(OUT / f"selection_alpha_{alpha:.2f}.csv", index=False)
        summaries.extend(rows)
        contrasts.extend(comparisons)
        selections[str(alpha)] = groups
    pd.DataFrame(summaries).to_csv(OUT / "strategy_summary.csv", index=False)
    df.groupby("policy")[["TIC", "NS", "TR", "BE"]].mean().to_csv(OUT / "policy_summary.csv")
    df.groupby(["operational_profile", "policy"])[["TIC", "NS", "TR", "BE"]].mean().to_csv(OUT / "profile_policy_summary.csv")
    payload = {"input_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
               "n_series": len(profiles), "n_stores": df.store_id.nunique(), "n_policies": df.policy.nunique(),
               "policies": sorted(df.policy.unique()), "profile_counts": profiles.operational_profile.value_counts().to_dict(),
               "maximum_TIC_std": float(df.TIC_std.max()), "profile_choices": selections,
               "contrasts": contrasts,
               "warnings": ["Existing June report uses a different policy portfolio and is stale.",
                            "Five deterministic evaluation repetitions are not independent training replications.",
                            "Cross-fitting policy choice does not remove retrospective profile or simulator scaling information.",
                            "Costs are simulation objective units unless original monetary calibration is independently documented."]}
    (OUT / "validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["n_series", "n_stores", "policies", "profile_choices"]}, indent=2))
    print(pd.DataFrame(summaries).query("alpha == 0.7").drop(columns="policy_counts").to_string(index=False))
    print(json.dumps(contrasts[:4], indent=2))


if __name__ == "__main__":
    main()
