from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

import judge


class EvaluationTests(unittest.TestCase):
    def test_matching_ignores_numeric_storage_type_and_column_order(self):
        first = pd.DataFrame({"count": [1, 2], "category": ["tcp", "udp"]})
        second = pd.DataFrame({"category": ["tcp", "udp"], "count": [1.0, 2.0]})
        np.testing.assert_array_equal(
            judge.row_keys(first, ["count", "category"]),
            judge.row_keys(second, ["count", "category"]),
        )

    def test_evaluation_excludes_all_training_matches_and_preserves_class_ratio(self):
        reference = pd.DataFrame({
            "value": list(range(1, 11)) + list(range(101, 111)) + [1],
            "is_anomaly": [0] * 10 + [1] * 10 + [0],
        })
        coursework = reference.iloc[[0, 1, 10]].copy()
        evaluation, details = judge.make_evaluation(reference, coursework, 6, 42)
        repeated, _ = judge.make_evaluation(reference, coursework, 6, 42)
        self.assertEqual(evaluation["is_anomaly"].value_counts().to_dict(), {0: 4, 1: 2})
        self.assertTrue(set(evaluation["value"]).isdisjoint({1, 2, 101}))
        self.assertEqual(details["excluded_reference_rows"], 4)
        self.assertEqual(details["matched_coursework_rows"], 3)
        pd.testing.assert_frame_equal(evaluation, repeated)

    def test_category_encoder_learns_from_synthetic_samples_only(self):
        synthetic = pd.DataFrame({"value": [1.0, 2.0], "protocol": ["tcp", "udp"]})
        evaluation = pd.DataFrame({"protocol": ["icmp"], "value": [3.0]})
        encoder, train, test = judge.encode_features(synthetic, evaluation, ["protocol"])
        self.assertEqual(train.shape, (2, 3))
        self.assertEqual(test.shape, (1, 3))
        categories = encoder.named_transformers_["categorical"].categories_[0]
        self.assertEqual(categories.tolist(), ["tcp", "udp"])
        np.testing.assert_array_equal(test[0], [0.0, 0.0, 3.0])


class ScoringTests(unittest.TestCase):
    def test_successful_cli_run_without_model_saving_removes_stale_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copyfile(judge.ROOT / "judge.py", root / "judge.py")
            dataset = root / "nsl-kdd"
            dataset.mkdir()
            features = pd.DataFrame({
                "duration": np.linspace(0, 1, 20),
                "protocol_type": ["tcp", "udp"] * 10,
                "service": ["http", "dns"] * 10,
                "flag": ["SF", "REJ"] * 10,
            })
            features.assign(is_anomaly=0).to_csv(dataset / "train.csv", index=False)
            submission = features.copy()
            submission.insert(0, "id", range(20))
            submission.to_csv(dataset / "sample_submission.csv", index=False)
            evaluation = features.assign(is_anomaly=[0] * 10 + [1] * 10)
            evaluation.loc[10:, "duration"] += 10
            evaluation.to_csv(root / "valid.csv", index=False)
            output = root / "results"
            output.mkdir()
            model_path = output / "models.joblib"
            model_path.write_bytes(b"model from a previous run")
            result = subprocess.run([
                sys.executable, str(root / "judge.py"), "nsl-kdd",
                "--evaluation-csv", str(root / "valid.csv"),
                "--output-dir", str(output), "--trees", "5", "--seeds", "0", "--n-jobs", "1",
            ], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "predictions.csv").is_file())
            self.assertFalse(model_path.exists())

    def test_score_averages_detector_metrics_and_reports_both_forest_aggregations(self):
        labels = np.array([0, 0, 1, 1])
        ecod = np.array([0.0, 0.0, 2.0, 2.0])
        forests = np.column_stack([ecod, np.zeros(4)])
        metrics = judge.calculate_metrics(labels, ecod, forests)
        self.assertEqual(metrics["ecod_auprc"], 1.0)
        self.assertEqual(metrics["iforest_auprc_per_seed"], [1.0, 0.5])
        self.assertEqual(metrics["iforest_auprc"], 0.75)
        self.assertEqual(metrics["score"], 0.875)
        self.assertEqual(metrics["iforest_mean_score_auprc"], 1.0)
        self.assertEqual(metrics["score_using_mean_iforest_scores"], 1.0)

    def test_evaluation_requires_both_normal_and_anomalous_records(self):
        with self.assertRaisesRegex(ValueError, "both"):
            judge.calculate_metrics(np.zeros(4), np.arange(4), np.arange(4)[:, None])

    def test_trained_models_rank_anomalies_higher_and_can_be_reloaded(self):
        synthetic = pd.DataFrame({
            "value": np.linspace(-1, 1, 100), "protocol": ["tcp", "udp"] * 50,
        })
        evaluation = pd.DataFrame({
            "value": np.r_[np.linspace(-0.2, 0.2, 20), np.linspace(10, 20, 20)],
            "protocol": ["tcp", "udp"] * 20,
        })
        models, ecod, forests = judge.fit_judges(
            synthetic, evaluation, ["protocol"], seeds=[0, 7],
            trees=20, max_samples=64, n_jobs=1,
        )
        metrics = judge.calculate_metrics(np.r_[np.zeros(20), np.ones(20)], ecod, forests)
        self.assertGreater(metrics["ecod_auprc"], 0.99)
        self.assertGreater(metrics["iforest_auprc"], 0.99)
        self.assertEqual(forests.shape, (40, 2))
        self.assertEqual([len(model.estimators_) for model in models["iforest"]], [20, 20])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.joblib"
            joblib.dump(models, path)
            loaded = joblib.load(path)
            values = loaded["preprocessor"].transform(evaluation)
            np.testing.assert_allclose(loaded["ecod"].decision_function(values), ecod)
            for i, model in enumerate(loaded["iforest"]):
                np.testing.assert_allclose(-model.score_samples(values), forests[:, i])


if __name__ == "__main__":
    unittest.main()
