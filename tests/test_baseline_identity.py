"""Exercise the real loader with overlapping cell-line labels and tiny CSVs.

Run with: python -m unittest discover -s tests -p test_baseline_identity.py
No research dataset, checkpoint or model training is required.
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch


SOURCE = Path(__file__).resolve().parents[1] / "ProteinTalks" / "dataset.py"
SPEC = importlib.util.spec_from_file_location("proteintalks_dataset_test", SOURCE)
DATASET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATASET)
PREFIX = "fixture_"
ATCC = "MDA-MB-453(ATCC)"
PLAIN = "MDA-MB-453"


class BaselineIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def write_fixture(self, plain_baseline=(9, 0, 0)):
        # Put ATCC first: a substring/first-match lookup would borrow this row.
        labels = [[ATCC + "_no", 0], ["BT20_no", 0]]
        expression = [[0, 0, 9], [0, 9, 0]]
        if plain_baseline is not None:
            labels.append([PLAIN + "_no", 0])
            expression.append(list(plain_baseline))
        for cell in [ATCC, PLAIN, "BT20"]:
            for time in [6, 24, 48]:
                labels.append([cell + "_#1", time])
                expression.append([time, time + 1, time + 2])
        count = len(labels)
        tables = {
            "node_Index": [["P1"], ["P2"], ["P3"]],
            "loo_label": labels,
            "expr": expression,
            "pert": [[1, 0, 1]] * count,
            "pheno": [[0]] * count,
            "drug_fp_phychem_A": [["[0.0, 1.0, 2.0]"]] * count,
            "drug_fp_phychem_B": [["[2.0, 0.0, 1.0]"]] * count,
        }
        for suffix, rows in tables.items():
            pd.DataFrame(rows).to_csv(
                self.directory / (PREFIX + suffix + ".csv"),
                index=False, header=False,
            )

    def load(self):
        return DATASET.ProteomicsDataset(PREFIX, str(self.directory))

    def test_overlapping_labels_use_their_own_baselines(self):
        self.write_fixture()
        dataset = self.load()
        expected = {ATCC: [0., 0., 1.], PLAIN: [1., 0., 0.], "BT20": [0., 1., 0.]}
        self.assertEqual(len(dataset), 3)
        for index, experiment in enumerate(dataset.get_experiment_types()):
            cell = experiment.split("_")[0]
            self.assertTrue(torch.equal(dataset[index][0].flatten(), torch.tensor(expected[cell])))

    def test_zero_exact_baseline_is_not_admitted_using_another_context(self):
        self.write_fixture(plain_baseline=(0, 0, 0))
        dataset = self.load()
        self.assertEqual(set(dataset.get_experiment_types()), {ATCC + "_#1", "BT20_#1"})

    def test_missing_exact_baseline_does_not_borrow_from_a_prefix_match(self):
        self.write_fixture(plain_baseline=None)
        # Preserve the loader's existing missing-baseline error behavior.
        with self.assertRaises(IndexError):
            self.load()


if __name__ == "__main__":
    unittest.main()
