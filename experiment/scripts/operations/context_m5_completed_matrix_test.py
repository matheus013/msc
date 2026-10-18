"""Synthetic checks for the M5 completed-controller-only evidence gate."""
from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_m5_completed_matrix as front
import pandas as pd


class MatrixGateTest(unittest.TestCase):
    def test_partial_complete_duplicate_and_changed_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "panel.parquet").write_bytes(b"mock panel identity")
            policies = sorted(front.STOCHASTIC | front.DETERMINISTIC)
            plan = {"planned_fits": 79890, "policies": policies, "seeds": [11, 29, 47],
                "panel_metadata": [{"origin": 50, "panel": "panel.parquet", "panel_sha256": front.context.sha(run / "panel.parquet")}]}
            front.context.write_json(run / "plan.json", plan)
            index = pd.MultiIndex.from_tuples([("CA", f"store{i % 10}", f"item{i}") for i in range(5326)], names=front.CASE[1:])
            panel = pd.DataFrame(index=index)
            names = [*front.CASE, "policy", "seed", "block", "dataset"]
            with (run / "kpis.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=names)
                writer.writeheader()
                for w, s, item in index:
                    for policy in policies:
                        for seed in ([11, 29, 47] if policy in front.STOCHASTIC else [0]):
                            for block in ("validation", "test"):
                                writer.writerow(dict(origin=50, warehouse=w, store_id=s, item_id=item,
                                    policy=policy, seed=seed, block=block, dataset="M5"))
            status = dict(dataset="M5", state="controllers_completed", completed_fits=79890, planned_fits=79890)
            front.context.write_json(run / "status.json", status | {"completed_fits": 79889})
            with self.assertRaisesRegex(ValueError, "79,890"):
                front.completed_matrix(run)
            front.context.write_json(run / "status.json", status)
            with patch.object(front.pd, "read_parquet", return_value=panel):
                actual = front.completed_matrix(run)
                self.assertEqual(actual["state"], "controllers_completed")
                self.assertEqual(json.loads((run / "status.json").read_text()), status)
                original = front.context.sha
                calls = {}
                def changed(path):
                    key = str(path)
                    calls[key] = calls.get(key, 0) + 1
                    return "changed" if Path(path).name == "kpis.csv" and calls[key] > 1 else original(path)
                with patch.object(front.context, "sha", side_effect=changed):
                    with self.assertRaisesRegex(ValueError, "changed during audit"):
                        front.completed_matrix(run)
                with (run / "kpis.csv").open("a", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerow([50, "CA", "store0", "item0", policies[0], 0, "validation", "M5"])
                with self.assertRaisesRegex(ValueError, "Unexpected or duplicate"):
                    front.completed_matrix(run)


if __name__ == "__main__":
    unittest.main()
