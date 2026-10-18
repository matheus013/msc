"""Manager choices and point-in-time context. No implicit snapshot backfilling."""
from dataclasses import dataclass, asdict
import re

import numpy as np
import pandas as pd

CASE = ["origin", "warehouse", "store_id", "item_id"]
LEVEL_KEYS = {"warehouse": ["warehouse"], "store": ["warehouse", "store_id"],
              "series": ["warehouse", "store_id", "item_id"]}
RESERVED = set(CASE + ["TIC", "NS", "policy", "seed", "block", "label", "strategy",
                       "decision_at", "available_at", "validation_available_at", "operational_profile"])


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str
    mode: str
    level: str
    description: str
    source: str
    availability_verified: bool = False
    availability_evidence: str = ""


def validate_config(config):
    """Validate choices without reading source data or fitting anything."""
    if type(config.get("enabled")) is not bool:
        raise ValueError("enabled must be an explicit boolean")
    opts = config["selection"]
    for name in ("max_context_features", "max_evaluations", "inner_folds", "trees", "max_categories", "min_category_count", "min_samples_leaf", "max_source_records"):
        if type(opts.get(name)) is not int or opts[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if opts["inner_folds"] < 2 or opts["max_evaluations"] < 2 or opts["max_categories"] < 2:
        raise ValueError("At least two inner folds/evaluations/categories required")
    for name, maximum in {"max_context_features": 32, "max_evaluations": 256, "inner_folds": 10,
                          "trees": 512, "max_categories": 128, "max_source_records": 1_000_000}.items():
        if opts[name] > maximum:
            raise ValueError(f"{name} exceeds the bounded extension limit ({maximum})")
    for name in ("max_missing_fraction", "min_relative_cost_improvement", "max_service_drop", "max_failure_rate_increase"):
        value = opts.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite fraction in [0,1]")
    if opts.get("model") not in ("class", "consequences"):
        raise ValueError("model must be class or consequences")
    if isinstance(opts.get("service_target"), bool) or not 0 < opts.get("service_target", 0) <= 1:
        raise ValueError("service_target must be in (0,1]")
    if type(opts.get("random_state")) is not int:
        raise ValueError("random_state must be an integer")
    features = validate_features(config["features"])
    if sum(f.mode == "fixed" for f in features) > opts["max_context_features"]:
        raise ValueError("Fixed features exceed the context feature budget")
    return features


def validate_features(rows):
    features = [row if isinstance(row, Feature) else Feature(**row) for row in rows]
    names = [f.name for f in features]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate feature definitions")
    for f in features:
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", f.name) or f.name.lower() in {s.lower() for s in RESERVED} or f.name.lower().startswith(("test_", "oracle_", "future_")):
            raise ValueError(f"Reserved/invalid context name: {f.name}")
        if f.kind not in ("numeric", "categorical") or f.mode not in ("fixed", "candidate", "excluded") or f.level not in LEVEL_KEYS:
            raise ValueError(f"Invalid feature contract: {f.name}")
        if not f.description.strip() or not f.source.strip():
            raise ValueError(f"Description/source required: {f.name}")
        if type(f.availability_verified) is not bool:
            raise ValueError("availability_verified must be boolean")
        if f.availability_verified and not f.availability_evidence.strip():
            raise ValueError(f"Historical availability evidence required: {f.name}")
    return features


def normalize_cases(frame):
    frame = frame.copy()
    if not set(CASE) <= set(frame) or frame[CASE].isna().any().any():
        raise ValueError("Complete case keys required")
    origin = pd.to_numeric(frame.origin, errors="raise")
    if not np.isfinite(origin).all() or not (origin > 0).all() or not origin.eq(origin.astype(int)).all():
        raise ValueError("origin must be a positive integer")
    frame["origin"] = origin.astype(int)
    for c in CASE[1:]:
        frame[c] = frame[c].astype(str)
        if frame[c].str.strip().eq("").any():
            raise ValueError("Blank case keys")
    if frame.duplicated(CASE).any():
        raise ValueError("Duplicate cases")
    return frame.reset_index(drop=True)


def timestamps(values, label):
    # Explicit offsets avoid interpreting local dates as silently available at midnight.
    raw = values.astype(str)
    if not raw.str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True).all():
        raise ValueError(f"{label} requires ISO timestamps with timezone offsets")
    parsed = pd.to_datetime(raw, utc=True, errors="raise", format="ISO8601")
    if parsed.isna().any():
        raise ValueError(f"Missing {label}")
    return parsed


def assemble_context(cases, records, features, *, enabled=True):
    """Latest admissible observation per feature/case; returns values and audit.

    available_at means when the value was actually knowable, not extraction date.
    Audited upstream provenance is still necessary; timestamps cannot establish it.
    """
    cases = normalize_cases(cases)
    out = cases[CASE].copy()
    audit = {"enabled": enabled, "cases": len(cases), "features": {}}
    if not enabled:
        return out, audit
    active = [f for f in validate_features(features) if f.mode != "excluded"]
    if not active:
        return out, audit
    for f in active:
        if not f.availability_verified:
            raise ValueError(f"Unverified historical availability: {f.name}")
    if "decision_at" not in cases:
        raise ValueError("Explicit per-case decision_at is required")
    cases["decision_at"] = timestamps(cases.decision_at, "decision_at")
    if not {"feature", "value", "available_at"} <= set(records):
        raise ValueError("Context needs feature, value, available_at")
    for f in active:
        keys = LEVEL_KEYS[f.level]
        if not set(keys) <= set(records):
            raise ValueError(f"Missing {f.level} keys for {f.name}")
        data = records.loc[records.feature.eq(f.name), keys + ["value", "available_at"]].copy()
        if data[keys].isna().any().any():
            raise ValueError(f"Missing keys for {f.name}")
        for k in keys:
            data[k] = data[k].astype(str)
        data["available_at"] = timestamps(data.available_at, "available_at")
        if f.kind == "numeric":
            data["value"] = pd.to_numeric(data.value, errors="raise")
            if np.isinf(data.value).any():
                raise ValueError(f"Infinite numeric context: {f.name}")
        else:
            data["value"] = data.value.map(lambda v: np.nan if pd.isna(v) or not str(v).strip() else str(v))
            if data.value.dropna().str.len().gt(256).any():
                raise ValueError("Categorical values exceed 256 characters")
        data = data.drop_duplicates()
        if data.duplicated(keys + ["available_at"]).any():
            raise ValueError(f"Conflicting values at the same availability timestamp: {f.name}")
        # An as-of merge avoids a cases x historical-observations cross product.
        joined = pd.merge_asof(cases[CASE + ["decision_at"]].sort_values("decision_at"),
            data.sort_values("available_at"), left_on="decision_at", right_on="available_at",
            by=keys, direction="backward", allow_exact_matches=True)
        timelines = {key if isinstance(key, tuple) else (key,): g.available_at.array.asi8
                     for key, g in data.sort_values("available_at").groupby(keys, sort=False)}
        future = 0
        for key, g in cases.groupby(keys, sort=False):
            times = timelines.get(key if isinstance(key, tuple) else (key,))
            if times is not None:
                future += int((len(times) - np.searchsorted(times, g.decision_at.array.asi8, side="right")).sum())
        values = joined[CASE + ["value"]].rename(columns={"value": f.name})
        out = out.merge(values, on=CASE, how="left", validate="one_to_one")
        audit["features"][f.name] = {**asdict(f), "source_observations": len(data),
            "matched_cases": int(out[f.name].notna().sum()),
            "future_case_observations_rejected": future}
    return out, audit
