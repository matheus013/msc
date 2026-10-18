"""Synthetic integration check of the isolated context runner."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_rounds as rounds


class ContextRoundsTest(unittest.TestCase):
    def test_paired_models_all_attributes_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepared, source = root / "prepared", root / "source"
            prepared.mkdir()
            source.mkdir()
            rounds.context.write_json(source / "status.json", {"state": "completed", "completed_fits": 1, "planned_fits": 1})
            rows, validation, test = [], [], []
            for store in range(6):
                for item in range(2):
                    key = dict(origin=10, warehouse="BA", store_id=f"s{store}", item_id=f"i{item}")
                    rows.append(key | {"signal": 1., "useful": float(item)})
                    for policy in ("A", "B"):
                        metric = key | {"policy": policy, "seed": 0, "TIC": 5. if (policy == "B" and item) or (policy == "A" and not item) else 20., "NS": .9}
                        validation.append(metric | {"block": "validation"})
                        test.append(metric | {"block": "test"})
            for name, data in (("context_features.csv", rows), ("validation.csv", validation), ("test.csv", test)):
                pd.DataFrame(data).to_csv(prepared / name, index=False)
            config, _, _ = rounds.context.load_config(rounds.context.DEFAULT_CONFIG, "BA")
            options = config["selection"] | {"trees": 4, "inner_folds": 2, "min_samples_leaf": 1, "min_category_count": 1}
            manifest = dict(source_run=str(source), file_sha256={n: rounds.context.sha(prepared / n) for n in ("validation.csv", "test.csv")},
                            source_sha256={}, base_numeric=["signal"], base_categorical=[])
            rounds.context.write_json(prepared / "manifest.json", manifest)
            protocol = dict(dataset="BA", models=["class", "consequences"], selection=options,
                feature_contracts=[rounds.feature("useful", "numeric", "synthetic", "Synthetic data")],
                source_context_sha256={}, context_code_sha256={},
                context_features_sha256=rounds.context.sha(prepared / "context_features.csv"),
                prepared_manifest_sha256=rounds.context.sha(prepared / "manifest.json"))
            rounds.context.write_json(prepared / "context_protocol.json", protocol)
            with patch.object(rounds.context, "OUTPUT_ROOT", root):
                status = rounds.run(prepared, root / "output", 2)
                self.assertEqual(status["completed_folds"], 12)
                self.assertEqual(status["controller_fits"], 0)
                metrics = pd.read_csv(root / "output/case_metrics.csv")
                self.assertEqual(len(metrics), 120)
                self.assertEqual(metrics.strategy.nunique(), 5)
                self.assertEqual(metrics.model.nunique(), 2)
                before = rounds.context.sha(root / "output/predictions_frozen_before_test.csv")
                rounds.run(prepared, root / "output", 1)
                self.assertEqual(before, rounds.context.sha(root / "output/predictions_frozen_before_test.csv"))
                summary = pd.read_csv(root / "output/summary.csv")
                self.assertTrue(summary.cases.eq(12).all())


if __name__ == "__main__":
    unittest.main()
