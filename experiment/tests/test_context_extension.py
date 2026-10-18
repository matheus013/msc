"""Synthetic contracts only; no real matrices, controllers or running jobs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment/src"))
from aipe_context.contracts import CASE, Feature, assemble_context, validate_config
from aipe_context.selection import fit_context_pse, evaluate_context_pse, validation_matrix

SPEC = importlib.util.spec_from_file_location("context_command", ROOT / "experiment/scripts/operations/context_extension.py")
COMMAND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMAND)


def fixture():
    features, outcomes, records = [], [], []
    for store in range(6):
        for item in range(2):
            key = dict(origin=10, warehouse="BA", store_id=f"s{store}", item_id=f"i{item}")
            features.append(key | {"signal": 0.0, "useful": float(item), "noise": 1.0})
            records.append({"warehouse": "BA", "store_id": key["store_id"], "item_id": key["item_id"],
                            "feature": "useful", "value": item, "available_at": "2020-01-01T00:00:00Z"})
            for p, correct in (("A", item == 0), ("B", item == 1)):
                outcomes.append(key | {"policy": p, "TIC": 1.0 if correct else 10.0, "NS": 1.0, "block": "validation"})
    return pd.DataFrame(features), pd.DataFrame(outcomes), pd.DataFrame(records)


def feature(name="useful", mode="candidate", kind="numeric", level="series"):
    return Feature(name, kind, mode, level, "Synthetic admissible signal", "synthetic fixture", True, "generated before 2020-02-01")


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.options = json.loads((ROOT / "experiment/conf/aipe_context_extension_v1.json").read_text(encoding="utf-8"))["selection"]
        self.options.update(trees=8, min_samples_leaf=1, min_category_count=1, max_evaluations=8, max_context_features=2)
        self.x, self.y, self.records = fixture()
        self.cases = self.x[CASE].assign(decision_at="2020-02-01T00:00:00Z", validation_available_at="2020-01-31T00:00:00Z")

    def test_future_values_cannot_replace_known_context(self):
        late = self.records.assign(value=999, available_at="2021-01-01T00:00:00Z")
        a, _ = assemble_context(self.cases, self.records, [feature()])
        b, audit = assemble_context(self.cases, pd.concat([self.records, late]), [feature()])
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(audit["features"]["useful"]["future_case_observations_rejected"], len(self.x))

    def test_unverified_context_and_naive_dates_rejected(self):
        bad = Feature("useful", "numeric", "fixed", "series", "test", "test")
        with self.assertRaisesRegex(ValueError, "Unverified"):
            assemble_context(self.cases, self.records, [bad])
        with self.assertRaisesRegex(ValueError, "timezone"):
            assemble_context(self.cases.assign(decision_at="2020-02-01"), self.records, [feature()])

    def test_conflicting_snapshot_and_duplicate_cases_rejected(self):
        bad = pd.concat([self.records, self.records.assign(value=999)])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            assemble_context(self.cases, bad, [feature()])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            assemble_context(pd.concat([self.cases, self.cases]), self.records, [feature()])

    def test_disabled_module_never_reads_unverified_context(self):
        values, audit = assemble_context(self.cases, None, [Feature("x", "numeric", "fixed", "store", "test", "test")], enabled=False)
        self.assertEqual(list(values.columns), CASE)
        self.assertFalse(audit["enabled"])

    def test_fixed_features_and_automatic_operational_selection(self):
        fit = fit_context_pse(self.x, self.y, [feature(), feature("noise", mode="fixed")], self.options, ["signal"])
        self.assertEqual(fit.audit["fixed_features"], ["noise"])
        self.assertEqual(fit.audit["automatic_features"], ["useful"])
        self.assertTrue(fit.audit["context_recommended"])
        pred = fit.predict(self.x)
        selected = pred[pred.strategy.eq("PSE_context_selected")]
        self.assertTrue(selected.policy.eq(np.where(self.x.useful.eq(0), "A", "B")).all())

    def test_constant_candidate_rejected_and_baseline_preserved(self):
        fit = fit_context_pse(self.x, self.y, [feature("noise")], self.options, ["signal"])
        self.assertFalse(fit.audit["context_recommended"])
        self.assertEqual(fit.audit["selected_features"], [])
        p = fit.predict(self.x).pivot(index=CASE, columns="strategy", values="policy")
        self.assertTrue(p.PSE_base.eq(p.PSE_context_guarded).all())

    def test_outer_held_outcomes_cannot_change_own_predictions_or_selection(self):
        opts = self.options | {"model": "class"}
        p, audits = evaluate_context_pse(self.x, self.y, [feature()], opts, ["signal"])
        poisoned = self.y.copy()
        mask = poisoned.store_id.eq("s0")
        poisoned.loc[mask, "TIC"] = 100 - poisoned.loc[mask, "TIC"]
        q, audits2 = evaluate_context_pse(self.x, poisoned, [feature()], opts, ["signal"])
        pd.testing.assert_frame_equal(p[p.store_id.eq("s0")].reset_index(drop=True), q[q.store_id.eq("s0")].reset_index(drop=True))
        self.assertEqual(audits[0], audits2[0])

    def test_service_deterioration_blocks_automatic_context(self):
        from aipe_context.selection import compare, measured
        baseline = self.x[CASE].assign(policy="A")
        context = self.x[CASE].assign(policy="B")
        data = self.y.copy()
        data.loc[data.policy.eq("A"), ["TIC", "NS"]] = [10, 1]
        data.loc[data.policy.eq("B"), ["TIC", "NS"]] = [1, .5]
        base = measured(baseline, data, .7)
        chosen = measured(context, data, .7)
        self.assertFalse(compare(chosen, base, .7, self.options)["accepted"])

    def test_test_block_and_incomplete_matrices_rejected(self):
        with self.assertRaisesRegex(ValueError, "validation"):
            fit_context_pse(self.x, self.y.assign(block="test"), [feature()], self.options, ["signal"])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validation_matrix(self.x, self.y.iloc[1:])

    def test_group_categorical_encoding_handles_unseen_and_missing_values(self):
        train = self.x.copy();train["useful"] = np.where(train.useful.eq(0), "small", "large")
        fit = fit_context_pse(train, self.y, [feature(kind="categorical", mode="fixed")], self.options, ["signal"])
        held = train.iloc[:2].copy();held["useful"] = ["never_seen", np.nan]
        self.assertEqual(len(fit.predict(held)), 8)

    def test_configuration_budget_and_reserved_outcomes(self):
        base = {"enabled": True, "selection": self.options, "features": [feature().__dict__]}
        validate_config(base)
        bad = deepcopy(base);bad["features"][0]["name"] = "TIC"
        with self.assertRaises(ValueError):validate_config(bad)
        bad = deepcopy(base);bad["features"] *= 2
        with self.assertRaises(ValueError):validate_config(bad)
        bad = deepcopy(base);bad["selection"]["max_context_features"] = 1
        bad["features"] = [feature("a", "fixed").__dict__, feature("b", "fixed").__dict__]
        with self.assertRaisesRegex(ValueError, "budget"):validate_config(bad)

    def test_output_protection_and_completion_gate(self):
        with self.assertRaisesRegex(ValueError, "Outputs"):
            COMMAND.new_output(ROOT / "experiment/data/08_reporting/research_v2/forbidden")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "status.json").write_text(json.dumps({"state": "running", "completed_fits": 2, "planned_fits": 3}))
            with self.assertRaisesRegex(ValueError, "completed"):
                COMMAND.completed(p)

    def test_search_budget_does_not_choose_only_first_candidate_opportunistically(self):
        opts = self.options | {"max_evaluations": 2}
        data = self.x.assign(second=self.x.useful)
        fit = fit_context_pse(data, self.y, [feature(), feature("second")], opts, ["signal"])
        self.assertEqual(fit.audit["selected_features"], [])
        self.assertEqual(fit.audit["stop_reason"], "budget_insufficient_for_complete_step")

    def test_seed_coverage_cannot_shrink_silently(self):
        seeds = pd.concat([self.y.assign(seed=1), self.y.assign(seed=2)], ignore_index=True)
        validation_matrix(self.x, seeds)
        with self.assertRaisesRegex(ValueError, "seed coverage"):
            validation_matrix(self.x, seeds.iloc[1:])

    def test_service_cannot_be_silently_averaged_over_fewer_seeds(self):
        seeds = pd.concat([self.y.assign(seed=1), self.y.assign(seed=2)], ignore_index=True)
        seeds.loc[0, "NS"] = np.nan
        with self.assertRaisesRegex(ValueError, "Defined service"):
            validation_matrix(self.x, seeds)

    def test_no_observations_remains_missing_without_backfilling(self):
        empty = self.records.iloc[:0].copy()
        values, audit = assemble_context(self.cases, empty, [feature()])
        self.assertTrue(values.useful.isna().all())
        self.assertEqual(audit["features"]["useful"]["matched_cases"], 0)

    def test_live_other_run_blocks_evaluation_before_any_model_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, state in [("BA", "completed"), ("M5", "running")]:
                p = root / name;p.mkdir()
                (p / "status.json").write_text(json.dumps({"state": state, "completed_fits": 24, "planned_fits": 24}), encoding="utf-8")
            ds = {"run": str(root / "BA"), "features": [feature().__dict__]}
            config = {"enabled": True, "selection": self.options,
                      "datasets": {"BA": ds, "M5": {"run": str(root / "M5")}}}
            with patch("aipe_context.selection.evaluate_context_pse") as fit:
                with self.assertRaisesRegex(ValueError, "completed"):
                    COMMAND.evaluate(config, ds, [feature()], root / "not_read", root / "not_created")
                fit.assert_not_called()
            self.assertFalse((root / "not_created").exists())

    def test_single_policy_and_undefined_service_keep_reference(self):
        one = self.y[self.y.policy.eq("A")].copy()
        one["NS"] = np.nan
        fit = fit_context_pse(self.x, one, [feature()], self.options, ["signal"])
        self.assertFalse(fit.audit["context_recommended"])
        self.assertTrue(fit.predict(self.x).policy.eq("A").all())

    def test_all_missing_categorical_training_column_is_supported(self):
        from aipe_context.selection import PolicyPredictor
        model = PolicyPredictor(["signal"], ["category"], self.options)
        model.fit(self.x.assign(category=np.nan), self.y)
        self.assertEqual(len(model.predict(self.x.assign(category="unknown"))), len(self.x))

    def test_asof_warehouse_context_is_shared_without_cross_product(self):
        records = pd.DataFrame([dict(warehouse="BA", feature="useful", value=7,
                                    available_at="2020-01-01T00:00:00Z")])
        result, _ = assemble_context(self.cases, records, [feature(level="warehouse")])
        self.assertEqual(len(result), len(self.cases))
        self.assertTrue(result.useful.eq(7).all())

    def test_labels_must_be_available_before_prediction(self):
        COMMAND.validated_cutoffs(self.cases, self.x)
        future = self.cases.copy()
        future.loc[0, "validation_available_at"] = "2021-01-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "labels"):
            COMMAND.validated_cutoffs(future, self.x)

    def test_inference_uses_asof_context_and_can_be_disabled_without_sources(self):
        fit = fit_context_pse(self.x, self.y, [feature()], self.options, ["signal"])
        base = self.x[CASE + ["signal"]]
        expected = fit.predict(self.x)
        actual, audit = fit.recommend(base, self.cases, self.records, [feature()])
        pd.testing.assert_frame_equal(actual, expected[expected.strategy.eq("PSE_context_guarded")].reset_index(drop=True))
        self.assertTrue(audit["context_used"])
        disabled, _ = fit.recommend(base, None, None, None, enabled=False)
        pd.testing.assert_frame_equal(disabled, expected[expected.strategy.eq("PSE_base")].reset_index(drop=True))

    def test_inference_without_validated_gain_uses_baseline_without_extra_data(self):
        fit = fit_context_pse(self.x, self.y, [feature("noise")], self.options, ["signal"])
        rows, audit = fit.recommend(self.x[CASE + ["signal"]], None, None, None)
        self.assertFalse(audit["context_used"])
        self.assertEqual(len(rows), len(self.x))

    def test_changed_selected_contract_requires_new_fit(self):
        fit = fit_context_pse(self.x, self.y, [feature()], self.options, ["signal"])
        with self.assertRaisesRegex(ValueError, "type/level"):
            fit.recommend(self.x[CASE + ["signal"]], self.cases, self.records,
                          [feature(kind="categorical")])

    def test_feature_export_guards_dependencies_and_discloses_other_revisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = {}
            for rel in (*COMMAND.FEATURE_DEPENDENCIES, "unrelated_oracles.py"):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("frozen", encoding="utf-8")
                frozen[rel.replace("/", "\\")] = COMMAND.sha(path)
            (root / "unrelated_oracles.py").write_text("previous revision", encoding="utf-8")
            with patch.object(COMMAND, "ROOT", root):
                audit = COMMAND.verify_feature_dependencies({"code_sha256": frozen})
                self.assertEqual(list(audit["other_source_differences"]), ["unrelated_oracles.py"])
                (root / COMMAND.FEATURE_DEPENDENCIES[0]).write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Feature dependency"):
                    COMMAND.verify_feature_dependencies({"code_sha256": frozen})

    def test_manager_configuration_is_explicit_and_backed_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp);path = folder / "context.json"
            cfg = {"enabled": False, "selection": self.options,
                   "datasets": {"BA": {"features": [feature().__dict__]}}}
            path.write_text(json.dumps(cfg), encoding="utf-8")
            before = path.read_bytes()
            with patch.object(COMMAND, "OUTPUT_ROOT", folder / "outputs"):
                result = COMMAND.configure(path, "BA", {"fixed": ["useful"], "candidate": [], "excluded": []}, True)
            changed = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(changed["enabled"])
            self.assertEqual(changed["datasets"]["BA"]["features"][0]["mode"], "fixed")
            self.assertEqual(Path(result["backup"]).read_bytes(), before)

    def test_offline_evaluation_freezes_before_loading_test_and_saves_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);source = root / "completed_source";source.mkdir()
            status = source / "status.json"
            status.write_text(json.dumps({"state": "completed", "completed_fits": 24, "planned_fits": 24}), encoding="utf-8")
            prepared = root / "prepared";prepared.mkdir()
            self.x[CASE + ["signal"]].to_csv(prepared / "features.csv", index=False)
            self.y.to_csv(prepared / "validation.csv", index=False)
            self.y.assign(block="test").to_csv(prepared / "test.csv", index=False)
            self.cases.to_csv(root / "cutoffs.csv", index=False)
            self.records.to_csv(root / "context.csv", index=False)
            manifest = {"source_run": str(source), "source_sha256": {str(status): COMMAND.sha(status)},
                        "base_numeric": ["signal"], "base_categorical": [],
                        "file_sha256": {n: COMMAND.sha(prepared / n) for n in ("features.csv", "validation.csv", "test.csv")}}
            (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            ds = {"run": str(source), "cutoffs_file": str(root / "cutoffs.csv"),
                  "records_file": str(root / "context.csv"), "features": [feature().__dict__]}
            config = {"enabled": True, "selection": self.options, "datasets": {"BA": ds}}
            output = root / "outputs/study"
            real_reader = pd.read_csv
            test_reads = []
            def audited_reader(path, *args, **kwargs):
                if Path(path).name == "test.csv":
                    self.assertTrue((output / "predictions_frozen_before_test.csv").is_file())
                    self.assertTrue((output / "feature_selection.json").is_file())
                    test_reads.append(str(path))
                return real_reader(path, *args, **kwargs)
            with patch.object(COMMAND, "OUTPUT_ROOT", root / "outputs"), patch.object(pd, "read_csv", side_effect=audited_reader):
                result = COMMAND.evaluate(config, ds, [feature()], prepared, output)
            self.assertEqual(result["controller_fits"], 0)
            self.assertEqual(len(test_reads), 1)
            for name in ("completed.json", "comparison.json", "summary.csv", "paired_base.csv", "selected_features_by_fold.csv"):
                self.assertTrue((output / name).is_file())
            self.assertEqual(len(list((output / "models").glob("*.joblib"))), 6)


if __name__ == "__main__":
    unittest.main()
