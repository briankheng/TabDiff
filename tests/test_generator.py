from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generator  # noqa: E402

# Tiny runs keep each CLI call to a few seconds on CPU.
FAST = ("--epochs", "2", "--target-n", "200", "--cpu")


def training_frame():
    # count and rate have more distinct values than MAX_DISCRETE_VALUES, so they
    # are continuous; binary, ttl and category are discrete. Attacks use values
    # no normal row takes: ttl 254, category "attack" and far larger counts.
    normal = pd.DataFrame({
        "count": range(10, 40),
        "rate": np.arange(1, 31) / 100,
        "binary": [0, 1] * 15,
        "ttl": [31] * 24 + [62] * 6,
        "category": [f"normal_{i}" for i in range(30)],
        "constant": [7] * 30,
    })
    attacks = pd.DataFrame({
        "count": [9999] * 5,
        "rate": [99.0] * 5,
        "binary": [1] * 5,
        "ttl": [254] * 5,
        "category": ["attack"] * 5,
        "constant": [7] * 5,
    })
    return pd.concat([normal.assign(is_anomaly=0), attacks.assign(is_anomaly=1)], ignore_index=True)


class TableCodecTests(unittest.TestCase):
    def setUp(self):
        self.train = training_frame()
        self.codec = generator.TableCodec(self.train)

    def test_columns_are_split_by_role_with_the_label_last(self):
        self.assertEqual(self.codec.constant, {"constant": 7})
        self.assertEqual(self.codec.continuous, ["count", "rate"])
        self.assertEqual(self.codec.discrete, ["binary", "ttl", "category", "is_anomaly"])
        self.assertEqual(self.codec.normal_categories["ttl"].tolist(), [True, True, False])
        self.assertEqual(self.codec.normal_categories["is_anomaly"].tolist(), [True, False])

    def test_encoding_then_decoding_returns_the_normal_rows(self):
        normal = self.train[self.train["is_anomaly"] == 0]
        numeric, codes = self.codec.encode(normal)
        decoded = self.codec.decode(numeric, codes)
        expected = normal.drop(columns="is_anomaly").reset_index(drop=True)
        pd.testing.assert_frame_equal(decoded, expected)

    def test_decoding_clips_to_the_normal_range_and_precision(self):
        normal = self.train[self.train["is_anomaly"] == 0].head(3)
        _, codes = self.codec.encode(normal)
        numeric = np.array([[5000.0, 0.12345], [-3.0, 7.0], [20.4, 0.1]])
        decoded = self.codec.decode(numeric, codes)
        self.assertEqual(decoded["count"].tolist(), [39, 10, 20])
        self.assertEqual(decoded["rate"].tolist(), [0.12, 0.3, 0.1])
        self.assertTrue(pd.api.types.is_integer_dtype(decoded["count"]))

    def test_decoding_rejects_categories_that_no_normal_row_uses(self):
        attacks = self.train[self.train["is_anomaly"] == 1]
        numeric, codes = self.codec.encode(attacks)
        with self.assertRaises(ValueError):
            self.codec.decode(numeric, codes)


class RecordingModel(torch.nn.Module):
    """Stands in for TabDiff's denoiser; its output depends on the label slot it is given."""

    def __init__(self, label_start):
        super().__init__()
        self.label_start = label_start
        self.inputs = []

    def forward(self, x_num, x_cat, t, sigma=None):
        self.inputs.append(x_cat.clone())
        is_attack = x_cat[:, self.label_start + generator.ATTACK:self.label_start + generator.ATTACK + 1]
        return x_num + 10 * is_attack, x_cat + 3 * is_attack


class ContrastiveGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.codec = generator.TableCodec(training_frame())
        slots = self.codec.category_sizes + 1
        self.label_start = int(slots[:-1].sum())
        self.x_num = torch.zeros(4, len(self.codec.continuous))
        self.x_cat = torch.zeros(4, int(slots.sum()))

    def guide(self, weight):
        model = RecordingModel(self.label_start)
        guidance = generator.ContrastiveGuidance(model, self.codec, weight)
        return model, *guidance(self.x_num, self.x_cat, torch.zeros(4))

    def test_zero_weight_conditions_on_the_normal_label_only(self):
        model, num, _ = self.guide(0.0)
        self.assertEqual(len(model.inputs), 1)
        label = model.inputs[0][:, self.label_start:self.label_start + 3]
        self.assertTrue(torch.equal(label, torch.tensor([[1.0, 0.0, 0.0]] * 4)))
        self.assertTrue(torch.equal(num, self.x_num))

    def test_positive_weight_moves_away_from_the_attack_prediction(self):
        model, num, _ = self.guide(2.0)
        self.assertEqual(len(model.inputs), 2)
        # normal predicts 0 and attack predicts 10, so 0 + 2 * (0 - 10) = -20.
        self.assertTrue(torch.equal(num, torch.full_like(self.x_num, -20.0)))

    def test_categories_no_normal_row_uses_are_blocked(self):
        _, _, cat = self.guide(1.0)
        blocked = torch.from_numpy(np.concatenate([
            np.append(~self.codec.normal_categories[c], False) for c in self.codec.discrete
        ]))
        self.assertTrue((cat[:, blocked] == generator.NEG_INF).all())
        self.assertTrue((cat[:, ~blocked] > generator.NEG_INF).all())


class GeneratorScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copyfile(ROOT / "generator.py", self.root / "generator.py")
        shutil.copytree(ROOT / "tabdiff", self.root / "tabdiff")
        self.cwd = self.root / "elsewhere"
        self.cwd.mkdir()
        self.train = training_frame()
        for dataset in ["nsl-kdd", "unsw-nb15"]:
            folder = self.root / dataset
            folder.mkdir()
            data = self.train if dataset == "nsl-kdd" else self.train.rename(columns={"count": "packets"})
            data.to_csv(folder / "train.csv", index=False)
            (folder / "sample_submission.csv").write_text("original sample\n")

    def generate(self, dataset, *options, output=None):
        command = [sys.executable, str(self.root / "generator.py"), dataset, *FAST, *options]
        if output is not None:
            command.extend(["--output", str(output)])
        result = subprocess.run(command, cwd=self.cwd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        path = output if output is not None else self.root / dataset / "submission.csv"
        self.assertTrue(path.is_file(), result.stdout)
        return pd.read_csv(path), result.stdout

    def test_required_row_counts_differ_by_dataset(self):
        self.assertEqual(generator.DATASET_SETTINGS["nsl-kdd"]["target_n"], 40000)
        self.assertEqual(generator.DATASET_SETTINGS["unsw-nb15"]["target_n"], 70000)

    def test_each_dataset_writes_its_own_schema_from_another_directory(self):
        for dataset, count_column in [("nsl-kdd", "count"), ("unsw-nb15", "packets")]:
            with self.subTest(dataset=dataset):
                result, _ = self.generate(dataset)
                self.assertEqual(
                    result.columns.tolist(),
                    ["id", count_column, "rate", "binary", "ttl", "category", "constant"],
                )
                self.assertEqual(result["id"].tolist(), list(range(200)))
                self.assertFalse(result.isna().any().any())
                self.assertEqual((self.root / dataset / "sample_submission.csv").read_text(), "original sample\n")

    def test_every_value_stays_within_the_normal_training_data(self):
        result, _ = self.generate("nsl-kdd")
        normal = self.train[self.train["is_anomaly"] == 0]
        for column in ["binary", "ttl", "category", "constant"]:
            with self.subTest(column=column):
                self.assertTrue(set(result[column]).issubset(set(normal[column])))
        for column in ["count", "rate"]:
            with self.subTest(column=column):
                self.assertGreaterEqual(result[column].min(), normal[column].min())
                self.assertLessEqual(result[column].max(), normal[column].max())
        self.assertTrue(pd.api.types.is_integer_dtype(result["count"]))
        self.assertTrue(np.allclose(result["rate"], result["rate"].round(2)))

    def test_trained_model_is_cached_and_sampling_is_reproducible(self):
        checkpoint = self.root / "model.pt"
        first, log = self.generate("nsl-kdd", "--checkpoint", str(checkpoint))
        self.assertIn("Training TabDiff", log)
        again, log = self.generate("nsl-kdd", "--checkpoint", str(checkpoint))
        self.assertIn("Loaded trained model", log)
        pd.testing.assert_frame_equal(first, again)
        # A fresh model trained with the same seed samples the same rows.
        retrained, log = self.generate("nsl-kdd")
        self.assertIn("Training TabDiff", log)
        pd.testing.assert_frame_equal(first, retrained)

        unguided, log = self.generate("nsl-kdd", "--checkpoint", str(checkpoint), "--guidance-weight", "0")
        self.assertIn("Loaded trained model", log)
        self.assertFalse(first.equals(unguided))
        other_seed, _ = self.generate("nsl-kdd", "--random-seed", "8")
        self.assertFalse(first.equals(other_seed))


if __name__ == "__main__":
    unittest.main()
