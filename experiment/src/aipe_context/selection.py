"""Nested, group-disjoint context selection by operational validation outcomes.

No controller is trained. This module never receives test outcomes. Selection
scores are development evidence, not service guarantees or proof of deployment gain.
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from .contracts import CASE, normalize_cases, validate_config

EPS = 1e-9


def validation_matrix(features, validation, expected_block="validation"):
    features = normalize_cases(features)
    v = validation.copy()
    if "block" in v and not v.block.eq(expected_block).all():
        raise ValueError(f"Only {expected_block} outcomes allowed at this stage")
    if not set(CASE + ["policy", "TIC", "NS"]) <= set(v):
        raise ValueError("Missing validation matrix fields")
    for c in CASE[1:]:
        if v[c].isna().any():
            raise ValueError("Missing outcome case keys")
        v[c] = v[c].astype(str)
    if v[CASE + ["policy", "TIC"]].isna().any().any():
        raise ValueError("Incomplete validation keys/cost")
    v["origin"] = pd.to_numeric(v.origin, errors="raise")
    v["policy"] = v.policy.astype(str)
    v["TIC"] = pd.to_numeric(v.TIC, errors="raise")
    v["NS"] = pd.to_numeric(v.NS, errors="raise")
    if v.policy.str.strip().eq("").any() or ("seed" in v and v.seed.isna().any()):
        raise ValueError("Missing policy/seed")
    if not np.isfinite(v.TIC).all() or v.TIC.lt(0).any() or not v.NS.dropna().between(0, 1).all():
        raise ValueError("Invalid validation cost/service")
    unique = CASE + ["policy"] + (["seed"] if "seed" in v else [])
    if v.duplicated(unique).any():
        raise ValueError("Duplicate validation observations")
    if v.assign(defined=v.NS.notna()).groupby(CASE).defined.nunique().gt(1).any():
        raise ValueError("Defined service population must agree across policies and seeds")
    if "seed" in v:
        expected = v.groupby("policy").seed.apply(lambda x: frozenset(x))
        if any(frozenset(g.seed) != expected.loc[key[-1]] for key, g in v.groupby(CASE + ["policy"])):
            raise ValueError("Incomplete per-policy seed coverage")
    mean = v.groupby(CASE + ["policy"], as_index=False, dropna=False)[["TIC", "NS"]].mean()
    wanted = set(features[CASE].itertuples(index=False, name=None))
    actual = set(mean[CASE].itertuples(index=False, name=None))
    if wanted != actual or len(mean) != len(wanted) * mean.policy.nunique():
        raise ValueError("Incomplete policy matrix or mismatched feature cases")
    masks = mean.assign(defined=mean.NS.notna()).groupby(CASE).defined.nunique()
    if masks.gt(1).any():
        raise ValueError("Defined service population must agree across policies")
    return features, mean


def reference_policy(matrix, target):
    stats = matrix.groupby("policy")[["TIC", "NS"]].mean().reset_index()
    feasible = stats[stats.NS.ge(target - EPS)]
    if len(feasible):
        return feasible.sort_values(["TIC", "policy"]).iloc[0].policy
    return stats.sort_values(["NS", "TIC", "policy"], ascending=[False, True, True], na_position="last").iloc[0].policy


def group_ids(frame):
    return list(zip(frame.warehouse, frame.store_id))


class PolicyPredictor:
    """One matched learner, with or without extra context; fold-local transforms."""
    def __init__(self, numeric, categorical, options):
        self.numeric, self.categorical, self.options = list(numeric), list(categorical), dict(options)

    def fit(self, features, matrix):
        o = self.options
        self.reference = reference_policy(matrix, o["service_target"])
        transforms = []
        if self.numeric:
            transforms.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True), self.numeric))
        if self.categorical:
            transforms.append(("category", make_pipeline(
                SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True),
                OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False,
                              min_frequency=o["min_category_count"], max_categories=o["max_categories"])), self.categorical))
        if not transforms:
            raise ValueError("At least one baseline feature is required")
        self.preprocessor = ColumnTransformer(transforms)
        x = self.preprocessor.fit_transform(self._frame(features))
        opts = dict(n_estimators=o["trees"], max_depth=4, min_samples_leaf=o["min_samples_leaf"],
                    random_state=o["random_state"], n_jobs=1)
        index = pd.MultiIndex.from_frame(features[CASE])
        if o["model"] == "class":
            labels = {key: reference_policy(g, o["service_target"]) for key, g in matrix.groupby(CASE)}
            y = [labels[key] for key in features[CASE].itertuples(index=False, name=None)]
            self.classifier = ExtraTreesClassifier(**opts).fit(x, y)
        else:
            cost = matrix.pivot(index=CASE, columns="policy", values="TIC").reindex(index)
            service = matrix.pivot(index=CASE, columns="policy", values="NS").reindex(index).reindex(columns=cost.columns)
            self.policies = list(cost.columns)
            cost_target = np.log1p(cost.to_numpy())
            self.cost_model = ExtraTreesRegressor(**opts).fit(x, cost_target[:, 0] if len(self.policies) == 1 else cost_target)
            defined = service.notna().all(axis=1).to_numpy()
            service_target = service.to_numpy()[defined]
            self.service_model = ExtraTreesRegressor(**opts).fit(x[defined], service_target[:, 0] if len(self.policies) == 1 else service_target) if defined.any() else None
        return self

    def predict(self, features):
        x = self.preprocessor.transform(self._frame(features))
        out = features[CASE].copy()
        if self.options["model"] == "class":
            out["policy"] = self.classifier.predict(x)
        elif self.service_model is None:
            out["policy"] = self.reference
        else:
            cost = np.maximum(0, np.expm1(self.cost_model.predict(x))).reshape(len(features), -1)
            service = np.clip(self.service_model.predict(x), 0, 1).reshape(len(features), -1)
            ref = self.policies.index(self.reference)
            selected = []
            for costs, fill in zip(cost, service):
                feasible = np.flatnonzero(fill >= self.options["service_target"] - EPS)
                chosen = min(feasible, key=lambda j: (costs[j], self.policies[j])) if len(feasible) else ref
                # Empirical intervention rule, deliberately identical for all feature sets.
                override = (fill[chosen] >= self.options["service_target"] - EPS and
                            fill[chosen] >= fill[ref] - EPS and costs[chosen] < costs[ref] - EPS)
                selected.append(self.policies[chosen] if override else self.reference)
            out["policy"] = selected
        return out

    def _frame(self, features):
        frame = features.copy()
        for c in self.categorical:
            frame[c] = frame[c].astype(object).where(frame[c].notna(), np.nan)
        return frame


def measured(predictions, matrix, target):
    out = predictions.merge(matrix, on=CASE + ["policy"], validate="one_to_one")
    if len(out) != len(predictions) or out.TIC.isna().any():
        raise ValueError("Missing consequences for recommendations")
    return out.assign(failed=out.NS.lt(target - EPS) & out.NS.notna())


def compare(candidate, baseline, target, options):
    joined = candidate.merge(baseline, on=CASE, suffixes=("_context", "_base"), validate="one_to_one")
    if len(joined) != len(candidate) or len(joined) != len(baseline):
        raise ValueError("Different evaluation populations")
    cost_base = float(joined.TIC_base.sum())
    cost = float(joined.TIC_context.sum())
    gain = (cost_base - cost) / cost_base if cost_base > EPS else 0.0
    defined = joined.NS_base.notna() & joined.NS_context.notna()
    delta = float((joined.loc[defined, "NS_context"] - joined.loc[defined, "NS_base"]).mean()) if defined.any() else None
    failure_delta = float((joined.loc[defined, "NS_context"].lt(target - EPS).astype(int) - joined.loc[defined, "NS_base"].lt(target - EPS).astype(int)).mean()) if defined.any() else None
    positive = np.maximum(0, joined.TIC_base.to_numpy() - joined.TIC_context.to_numpy())
    accept = bool(delta is not None and gain > options["min_relative_cost_improvement"] + EPS and
                  delta >= -options["max_service_drop"] - EPS and
                  failure_delta <= options["max_failure_rate_increase"] + EPS)
    return {"cost_sum": cost, "baseline_cost_sum": cost_base, "relative_cost_improvement": gain,
            "service_difference": delta, "failure_rate_difference": failure_delta,
            "defined_service_cases": int(defined.sum()), "cases": len(joined),
            "interventions": int(joined.policy_context.ne(joined.policy_base).sum()),
            "new_service_failures": int((defined & joined.NS_base.ge(target - EPS) & joined.NS_context.lt(target - EPS)).sum()),
            "recovered_service_failures": int((defined & joined.NS_context.ge(target - EPS) & joined.NS_base.lt(target - EPS)).sum()),
            "largest_positive_saving_share": float(positive.max() / positive.sum()) if positive.sum() > EPS else None,
            "accepted": accept}


@dataclass
class FittedContextPSE:
    baseline: PolicyPredictor
    fixed: PolicyPredictor
    selected: PolicyPredictor
    audit: dict

    def recommend(self, base_features, cases, records, specs, *, enabled=True, strategy="PSE_context_guarded"):
        """Point-in-time inference entrypoint; no outcomes or fitting required."""
        from .contracts import assemble_context
        base = normalize_cases(base_features)
        models = {"PSE_base": self.baseline, "PSE_context_fixed": self.fixed,
                  "PSE_context_selected": self.selected,
                  "PSE_context_guarded": self.selected if self.audit["context_recommended"] else self.baseline}
        if strategy not in models:
            raise ValueError("Unknown context recommendation strategy")
        inactive = not enabled or strategy == "PSE_base" or (strategy == "PSE_context_guarded" and not self.audit["context_recommended"])
        if inactive:
            return self.baseline.predict(base).assign(strategy="PSE_base" if not enabled else strategy), {
                "enabled": enabled, "context_used": False, "reason": "disabled_or_baseline_recommended"}
        selected = set(self.audit["fixed_features"] if strategy == "PSE_context_fixed" else self.audit["selected_features"])
        if not selected:
            return models[strategy].predict(base).assign(strategy=strategy), {"enabled": enabled, "context_used": False}
        cutoffs = normalize_cases(cases)
        if set(base[CASE].itertuples(index=False, name=None)) != set(cutoffs[CASE].itertuples(index=False, name=None)):
            raise ValueError("Inference cases differ from baseline feature cases")
        # Unused candidate sources need not be present at recommendation time.
        needed = [s for s in specs if s.name in selected and s.mode != "excluded"]
        if {s.name for s in needed} != selected:
            raise ValueError("Missing selected context contracts")
        for s in needed:
            previous = self.audit["feature_contracts"][s.name]
            if s.kind != previous["kind"] or s.level != previous["level"]:
                raise ValueError("Context type/level changed; fit a new version")
        extra, availability = assemble_context(cutoffs, records, needed)
        ready = base.merge(extra, on=CASE, validate="one_to_one")
        availability["context_used"] = True
        availability["strategy"] = strategy
        return models[strategy].predict(ready).assign(strategy=strategy), availability

    def predict(self, features):
        """Return frozen recommendations only, with no realized cost or service."""
        outputs = []
        for name, model in (("PSE_base", self.baseline), ("PSE_context_fixed", self.fixed),
                            ("PSE_context_selected", self.selected),
                            ("PSE_context_guarded", self.selected if self.audit["context_recommended"] else self.baseline)):
            rows = model.predict(features)
            rows["strategy"] = name
            outputs.append(rows)
        return pd.concat(outputs, ignore_index=True)


def fit_context_pse(features, validation, specs, options, base_numeric, base_categorical=()):
    """Forward selection entirely inside the provided training population.

    fixed: always retained in the extended variants, subject to the declared cap.
    candidate: admitted only by nested operational comparisons. Guarded activation
    can still recommend baseline; mandatory attributes do not certify usefulness.
    """
    specs = validate_config({"enabled": True, "selection": options, "features": specs})
    features, matrix = validation_matrix(features, validation)
    if features.origin.nunique() != 1:
        raise ValueError("Fit each origin separately; future origins cannot train earlier ones")
    active = [s for s in specs if s.mode != "excluded"]
    if any(not s.availability_verified or not s.availability_evidence for s in active):
        raise ValueError("Only verified context contracts may be fitted")
    if set(s.name for s in active) & set([*base_numeric, *base_categorical]):
        raise ValueError("Context names collide with baseline features")
    fixed = sorted(s.name for s in active if s.mode == "fixed")
    if len(fixed) > options["max_context_features"]:
        raise ValueError("Fixed attributes exceed budget")
    group = pd.factorize(pd.Series(group_ids(features)), sort=True)[0]
    n_groups = len(set(group))
    if n_groups < 3:
        raise ValueError("At least three training groups required for context selection")
    splits = list(GroupKFold(min(options["inner_folds"], n_groups)).split(features, groups=group))
    kinds = {s.name: s.kind for s in active}
    quality, candidates = {}, []
    for s in active:
        if s.name not in features:
            raise ValueError(f"Missing assembled context: {s.name}")
        missing = float(features[s.name].isna().mean())
        reason = "constant_or_empty" if features[s.name].nunique(dropna=True) < 2 else "missing_fraction" if missing > options["max_missing_fraction"] else None
        quality[s.name] = {"missing_fraction": missing, "distinct_training_values": int(features[s.name].nunique()),
                           "reason": reason, "mode": s.mode}
        if s.mode == "fixed" and missing == 1:
            raise ValueError(f"Fixed context has no admissible training values: {s.name}")
        if s.mode == "candidate" and reason is None:
            candidates.append(s.name)
    cache, trials = {}, []

    def model(names, frame, outcomes):
        numeric = list(base_numeric) + [c for c in names if kinds[c] == "numeric"]
        categorical = list(base_categorical) + [c for c in names if kinds[c] == "categorical"]
        return PolicyPredictor(numeric, categorical, options).fit(frame, outcomes)

    def oof(names):
        key = tuple(sorted(names))
        if key not in cache:
            if len(cache) >= options["max_evaluations"]:
                raise ValueError("Internal evaluation budget exhausted")
            results = []
            for train_ix, held_ix in splits:
                train, held = features.iloc[train_ix], features.iloc[held_ix]
                train_keys = pd.MultiIndex.from_frame(train[CASE])
                held_keys = pd.MultiIndex.from_frame(held[CASE])
                train_m = matrix[pd.MultiIndex.from_frame(matrix[CASE]).isin(train_keys)]
                held_m = matrix[pd.MultiIndex.from_frame(matrix[CASE]).isin(held_keys)]
                pred = model(key, train, train_m).predict(held)
                results.append(measured(pred, held_m, options["service_target"]))
            cache[key] = pd.concat(results, ignore_index=True)
        return cache[key]

    baseline_oof = oof([])
    current = list(fixed)
    current_oof = oof(current)
    stopped = "no_candidates"
    while len(current) < options["max_context_features"]:
        remaining = sorted(set(candidates) - set(current))
        if not remaining:
            break
        needed = sum(tuple(sorted([*current, c])) not in cache for c in remaining)
        if len(cache) + needed > options["max_evaluations"]:
            stopped = "budget_insufficient_for_complete_step"
            break
        accepted = []
        for c in remaining:
            names = sorted([*current, c])
            pred = oof(names)
            result = compare(pred, current_oof, options["service_target"], options)
            trials.append({"added": c, "features": names, **result})
            if result["accepted"]:
                accepted.append((result["cost_sum"], c, names, pred))
        if not accepted:
            stopped = "no_operational_improvement"
            break
        _, _, current, current_oof = min(accepted, key=lambda r: (r[0], r[1]))
        stopped = "feature_cap" if len(current) == options["max_context_features"] else "no_candidates"
    comparison = compare(current_oof, baseline_oof, options["service_target"], options)
    audit = {"fixed_features": fixed, "selected_features": current, "automatic_features": sorted(set(current) - set(fixed)),
             "feature_contracts": {s.name: asdict(s) for s in active},
             "context_recommended": bool(current and comparison["accepted"]), "comparison_to_base": comparison,
             "fixed_comparison_to_base": compare(oof(fixed), baseline_oof, options["service_target"], options),
             "training_cases": len(features), "training_groups": n_groups, "inner_folds": len(splits),
             "evaluations": len(cache), "stop_reason": stopped, "quality": quality, "trials": trials,
             "claim_boundary": "Internal empirical selection; no service guarantee or independent confirmation"}
    fitted = {}
    def full(names):
        key = tuple(sorted(names))
        if key not in fitted:
            fitted[key] = model(key, features, matrix)
        return fitted[key]
    return FittedContextPSE(full([]), full(fixed), full(current), audit)


def evaluate_context_pse(features, validation, specs, options, base_numeric, base_categorical=(), artifact_callback=None):
    """Per-origin outer leave-one-store-out; held outcomes never select features."""
    features, matrix = validation_matrix(features, validation)
    predictions, audits = [], []
    for origin, cohort in features.groupby("origin", sort=True):
        groups = sorted(set(group_ids(cohort)))
        if len(groups) < 4:
            raise ValueError("Outer evaluation requires at least four independent stores")
        for warehouse, store in groups:
            is_held = cohort.warehouse.eq(warehouse) & cohort.store_id.eq(store)
            train, held = cohort.loc[~is_held], cohort.loc[is_held]
            keys = pd.MultiIndex.from_frame(train[CASE])
            outcomes = matrix[pd.MultiIndex.from_frame(matrix[CASE]).isin(keys)]
            fit = fit_context_pse(train, outcomes, specs, options, base_numeric, base_categorical)
            predictions.append(fit.predict(held))
            audits.append({"origin": int(origin), "warehouse": warehouse, "held_store": store, **fit.audit})
            if artifact_callback:
                artifact_callback(origin, warehouse, store, fit)
    return pd.concat(predictions, ignore_index=True), audits
