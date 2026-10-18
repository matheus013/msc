"""Chronology and safety contracts for the planning-only sequential study."""
import copy
import csv
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "aipe_sequential_plan", ROOT / "experiment/scripts/operations/plan_aipe_sequential.py")
PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAN)


class SequentialPlanningTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(PLAN.DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_prefix_excludes_decision_and_future_with_latency(self):
        rows = list(PLAN.decision_windows(17, 38, 1, 7, {"mode": "expanding"}, latency=2))
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[0]["decision_position"], 24)
        for row in rows:
            self.assertEqual(row["history_end_exclusive"], row["decision_position"] - 2)
            self.assertLess(row["available_last_position"], row["decision_position"])
            self.assertEqual(row["history_start"], 0)

    def test_cadence_boundary_and_planning_horizon_are_distinct(self):
        rows = list(PLAN.decision_windows(17, 38, 2, 7, {"mode": "fixed", "periods": 12}))
        self.assertEqual([r["decision_position"] for r in rows], [24, 26, 28, 30])
        self.assertEqual([r["valid_until_exclusive"] for r in rows], [26, 28, 30, 31])
        self.assertEqual(rows[-1]["planning_end_exclusive"], 37)
        self.assertEqual(rows[-1]["scorable_horizon_end_exclusive"], 31)
        self.assertTrue(rows[-1]["scoring_horizon_truncated_by_evaluation"])
        self.assertEqual(rows[-1]["history_end_exclusive"] - rows[-1]["history_start"], 12)

    def test_missing_history_and_incomplete_block_rejected(self):
        with self.assertRaises(ValueError):
            list(PLAN.decision_windows(17, 30, 1, 7, {"mode": "expanding"}))
        with self.assertRaises(ValueError):
            list(PLAN.decision_windows(17, 38, 1, 7, {"mode": "fixed", "periods": 25}))

    def test_subperiod_resolution_and_unspecified_availability_rejected(self):
        for dataset in ("BA", "M5"):
            bad = copy.deepcopy(self.config)
            bad["datasets"][dataset]["resolution"] = "daily"
            with self.assertRaises(ValueError):
                PLAN.validate(bad, dataset)
        self.config["review_cadences"] = [0.5]
        with self.assertRaises(ValueError):
            PLAN.validate(self.config, "BA")
        self.setUp()
        del self.config["availability_assumption"]
        with self.assertRaises(ValueError):
            PLAN.validate(self.config, "BA")

    def test_state_and_artifact_contract_cannot_be_removed(self):
        for field in PLAN.STATE_KEYS:
            bad = copy.deepcopy(self.config)
            bad["state_contract"][field] = False
            with self.assertRaises(ValueError):
                PLAN.validate(bad, "BA")
        self.config["state_contract"]["reset_eoq_at_review"] = True
        with self.assertRaises(ValueError):
            PLAN.validate(self.config, "M5")
        self.setUp()
        self.config["required_execution_artifacts"].remove("stateful_policy_switch_adapter")
        with self.assertRaises(ValueError):
            PLAN.validate(self.config, "M5")

    def test_model_updates_have_their_own_clock(self):
        self.assertEqual(PLAN.update_positions(24, 31, 0), [])
        self.assertEqual(PLAN.update_positions(24, 31, 3), [27, 30])
        reviews = [r["decision_position"] for r in PLAN.decision_windows(17, 38, 2, 3, {"mode": "expanding"})]
        self.assertNotIn(27, reviews)

    def test_small_fixture_never_exposes_future_observations(self):
        # Two episodes with different origins and a sentinel at every future position.
        for origin in (3, 6):
            observed = list(range(origin + 14))
            for row in PLAN.decision_windows(origin, len(observed), 2, 7, {"mode": "fixed", "periods": 3}):
                prefix = observed[row["history_start"]:row["history_end_exclusive"]]
                poisoned = observed[:row["decision_position"]] + [-999] * (len(observed) - row["decision_position"])
                self.assertEqual(prefix, poisoned[row["history_start"]:row["history_end_exclusive"]])
                self.assertNotIn(-999, prefix)
                self.assertEqual(len(prefix), 3)

    def test_review_frequency_does_not_rescale_fixed_order_cost(self):
        for cadence in (1, 2, 5):
            self.config["review_cadences"] = [cadence]
            PLAN.validate(self.config, "BA")
            self.assertEqual(self.config["scenario_costs"]["fixed_per_positive_order"], 50)
        self.config["scenario_costs"]["fixed_per_positive_order"] = 100
        with self.assertRaises(ValueError):
            PLAN.validate(self.config, "BA")

    def test_generated_agendas_match_manifests_and_temporal_contract(self):
        if not all((PLAN.OUT / ds / "manifest.json").exists() for ds in ("BA", "M5")):
            self.skipTest("Run the planning-only CLI before validating generated artifacts")
        for dataset in ("BA", "M5"):
            manifest = json.loads((PLAN.OUT / dataset / "manifest.json").read_text(encoding="utf-8"))
            agenda = PLAN.OUT / dataset / "agenda.csv"
            self.assertEqual(manifest["scientific_execution_status"], "NOT_READY")
            self.assertFalse(manifest["sequential_validation_performed"])
            self.assertEqual(PLAN.sha(agenda), manifest["agenda_sha256"])
            count, cases, firsts = 0, set(), set()
            with agenda.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    count += 1
                    cases.add(row["case_id"])
                    decision = int(row["decision_position"])
                    self.assertLess(int(row["available_last_position"]), decision)
                    self.assertLessEqual(int(row["history_end_exclusive"]), decision)
                    self.assertLess(decision, int(row["evaluation_end_exclusive"]))
                    self.assertEqual(row["scientific_execution_status"], "NOT_READY")
                    if row["state_action"] == "initialize_common_once":
                        self.assertEqual(decision, int(row["origin"]) + 7)
                        firsts.add((row["case_id"], row["scenario"]))
                    else:
                        self.assertEqual(row["state_action"], "carry_strategy_state")
            self.assertEqual(count, manifest["decision_rows"])
            self.assertEqual(len(cases), manifest["cases"])
            self.assertEqual(len(firsts), manifest["cases"] * manifest["scenarios"])


if __name__ == "__main__":
    unittest.main()
