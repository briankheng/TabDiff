"""Local ECOD and Isolation Forest judge using the documented competition setup."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATASET_SETTINGS = {
    "nsl-kdd": {"categorical": ["protocol_type", "service", "flag"]},
    "unsw-nb15": {"categorical": ["proto", "service", "state"]},
}
JUDGE_SETTINGS = {
    "trees": 500,
    "seeds": list(range(10)),
    "max_samples": 256,
    "n_jobs": -1,
    "evaluation_size": 20000,
    "evaluation_seed": 42,
}


def row_keys(frame, features):
    values = frame[features].copy()
    # Match CSV values even when one file infers integers and another floats.
    numeric = values.select_dtypes(include="number").columns
    values[numeric] = values[numeric].astype("float64")
    return pd.util.hash_pandas_object(values, index=False)


def load_reference(dataset, features):
    folder = ROOT / "datasets" / dataset
    if dataset == "nsl-kdd":
        with (folder / "KDDTrain+.arff").open() as f:
            names = [
                line.split()[1].strip("'")
                for line in f if line.lower().startswith("@attribute")
            ][:-1]
        reference = pd.concat([
            pd.read_csv(
                folder / name, header=None,
                names=names + ["attack_type", "difficulty"],
                float_precision="round_trip",
            )
            for name in ["KDDTrain+.txt", "KDDTest+.txt"]
        ], ignore_index=True)
        reference["is_anomaly"] = (reference["attack_type"] != "normal").astype(int)
    else:
        # The published smaller splits contain rate and the 42 coursework features.
        reference = pd.concat([
            pd.read_csv(folder / name, float_precision="round_trip")
            for name in ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]
        ], ignore_index=True).rename(columns={"label": "is_anomaly"})
    return reference[features + ["is_anomaly"]]


def make_evaluation(reference, coursework, size, seed):
    features = coursework.columns.drop("is_anomaly").tolist()
    reference_keys = row_keys(reference, features)
    coursework_keys = row_keys(coursework, features)
    matched = coursework_keys.isin(reference_keys)
    if not matched.all():
        raise ValueError("Some coursework rows do not match the public reference data.")
    overlap = reference_keys.isin(coursework_keys)
    available = reference.loc[~overlap]
    anomaly_count = round(size * coursework["is_anomaly"].mean())
    if not 0 < anomaly_count < size:
        raise ValueError("The evaluation sample must contain both label classes.")
    evaluation = pd.concat([
        available.loc[available["is_anomaly"] == label].sample(n=count, random_state=seed)
        for label, count in [(0, size - anomaly_count), (1, anomaly_count)]
    ]).sample(frac=1, random_state=seed)
    details = {
        "source": "public reference rows excluding all coursework feature matches",
        "reference_rows": len(reference),
        "matched_coursework_rows": int(matched.sum()),
        "excluded_reference_rows": int(overlap.sum()),
        "remaining_reference_rows": len(available),
        "seed": seed,
        "anomaly_fraction_policy": "match coursework training label ratio",
    }
    return evaluation.reset_index(drop=True), details


def encode_features(synthetic, evaluation, categorical):
    numeric = [column for column in synthetic.columns if column not in categorical]
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numeric", "passthrough", numeric),
    ])
    train = preprocessor.fit_transform(synthetic)
    test = preprocessor.transform(evaluation)
    return preprocessor, train, test


def fit_judges(synthetic, evaluation, categorical, seeds, trees, max_samples, n_jobs):
    preprocessor, train, test = encode_features(synthetic, evaluation, categorical)
    print(f"ECOD: {len(train)} synthetic training rows, {train.shape[1]} encoded features", flush=True)
    ecod = ECOD(n_jobs=1)
    ecod.fit(train)
    # PyOD ECOD pools training features with the entire evaluation batch for ranks.
    ecod_scores = ecod.decision_function(test)
    forests = []
    forest_scores = []
    for seed in seeds:
        print(f"IForest: seed {seed}, {trees} trees", flush=True)
        model = IsolationForest(
            n_estimators=trees, max_samples=max_samples,
            contamination="auto", random_state=seed, n_jobs=n_jobs,
        )
        model.fit(train)
        # sklearn scores increase with normality; the judge needs anomaly scores.
        forest_scores.append(-model.score_samples(test))
        forests.append(model)
    models = {
        "preprocessor": preprocessor,
        "ecod": ecod,
        "iforest": forests,
        "seeds": list(seeds),
        "features": synthetic.columns.tolist(),
    }
    return models, ecod_scores, np.column_stack(forest_scores)


def calculate_metrics(labels, ecod_scores, forest_scores):
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires both normal (0) and anomalous (1) records.")
    ecod_ap = float(average_precision_score(labels, ecod_scores))
    forest_ap = [float(average_precision_score(labels, scores)) for scores in forest_scores.T]
    forest_mean_ap = float(np.mean(forest_ap))
    ensemble_ap = float(average_precision_score(labels, forest_scores.mean(axis=1)))
    trapezoid = []
    for scores in [ecod_scores, *forest_scores.T]:
        precision, recall, _ = precision_recall_curve(labels, scores)
        trapezoid.append(float(auc(recall, precision)))
    return {
        "metric": "average_precision",
        "ecod_auprc": ecod_ap,
        "iforest_auprc_per_seed": forest_ap,
        "iforest_auprc": forest_mean_ap,
        "iforest_auprc_std": float(np.std(forest_ap)),
        "score": (ecod_ap + forest_mean_ap) / 2,
        "iforest_mean_score_auprc": ensemble_ap,
        "score_using_mean_iforest_scores": (ecod_ap + ensemble_ap) / 2,
        "trapezoidal_pr_auc": {
            "ecod": trapezoid[0],
            "iforest_per_seed": trapezoid[1:],
            "iforest_mean": float(np.mean(trapezoid[1:])),
            "score": (trapezoid[0] + float(np.mean(trapezoid[1:]))) / 2,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASET_SETTINGS)
    parser.add_argument("--submission", type=Path, help="Defaults to <dataset>/sample_submission.csv.")
    parser.add_argument("--evaluation-csv", type=Path, help="Optional labeled CSV in the coursework schema.")
    parser.add_argument("--evaluation-size", type=int, default=JUDGE_SETTINGS["evaluation_size"])
    parser.add_argument("--evaluation-seed", type=int, default=JUDGE_SETTINGS["evaluation_seed"])
    parser.add_argument("--trees", type=int, default=JUDGE_SETTINGS["trees"])
    parser.add_argument("--seeds", type=int, nargs="+", default=JUDGE_SETTINGS["seeds"])
    parser.add_argument("--n-jobs", type=int, default=JUDGE_SETTINGS["n_jobs"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()

    folder = ROOT / args.dataset
    submission_path = args.submission or folder / "sample_submission.csv"
    output = args.output_dir or ROOT / "results" / args.dataset / submission_path.stem
    coursework = pd.read_csv(folder / "train.csv", float_precision="round_trip")
    features = coursework.columns.drop("is_anomaly").tolist()
    submission = pd.read_csv(submission_path, float_precision="round_trip")
    if submission.columns.tolist() != ["id"] + features:
        raise ValueError("Submission columns must be id followed by the coursework feature columns.")
    synthetic = submission[features]
    if args.evaluation_csv:
        evaluation = pd.read_csv(args.evaluation_csv, float_precision="round_trip")
        overlap = row_keys(evaluation, features).isin(row_keys(coursework, features))
        details = {
            "source": str(args.evaluation_csv.resolve()),
            "coursework_overlap_rows": int(overlap.sum()),
        }
    else:
        reference = load_reference(args.dataset, features)
        evaluation, details = make_evaluation(
            reference, coursework, args.evaluation_size, args.evaluation_seed,
        )
    labels = evaluation["is_anomaly"].to_numpy()
    output.mkdir(parents=True, exist_ok=True)
    evaluation_path = output / "evaluation.csv"
    evaluation[features + ["is_anomaly"]].to_csv(evaluation_path, index=False)
    print(f"Local evaluation: {len(evaluation)} rows, anomaly fraction {labels.mean():.4f}", flush=True)
    models, ecod_scores, forest_scores = fit_judges(
        synthetic, evaluation[features], DATASET_SETTINGS[args.dataset]["categorical"],
        seeds=args.seeds, trees=args.trees,
        max_samples=JUDGE_SETTINGS["max_samples"], n_jobs=args.n_jobs,
    )
    metrics = calculate_metrics(labels, ecod_scores, forest_scores)
    submission_hash = hashlib.sha256(submission_path.read_bytes()).hexdigest()
    report = {
        "dataset": args.dataset,
        "evaluation_kind": "local proxy, not the private Kaggle split",
        "submission": str(submission_path.resolve()),
        "submission_sha256": submission_hash,
        "synthetic_rows": len(synthetic),
        "evaluation": {
            **details, "rows": len(evaluation),
            "anomaly_fraction": float(labels.mean()),
            "csv_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        },
        "settings": {
            "trees": args.trees, "seeds": args.seeds,
            "max_samples": JUDGE_SETTINGS["max_samples"], "n_jobs": args.n_jobs,
            "categorical_encoding": "one-hot, fitted on synthetic data; unknown categories become zeros",
            "numeric_preprocessing": "unchanged",
            "iforest_aggregation": "mean of per-seed average precision",
            "ecod_implementation": "PyOD; ranks pool synthetic training rows and the full evaluation batch",
        },
        "assumptions": [
            "The PDFs specify ECOD, 10 fixed IForest seeds, 500 trees, and the two-detector mean.",
            "Seed values, preprocessing, max_samples, metric integration, and seed aggregation are unspecified.",
        ],
        "metrics": metrics,
        "versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "pyod", "scipy", "joblib"]},
    }
    predictions = pd.DataFrame({"is_anomaly": labels, "ecod_score": ecod_scores})
    for i, seed in enumerate(args.seeds):
        predictions[f"iforest_seed_{seed}"] = forest_scores[:, i]
    predictions["iforest_mean_score"] = forest_scores.mean(axis=1)
    predictions.to_csv(output / "predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.save_models:
        joblib.dump(models, output / "models.joblib", compress=3)
    else:
        (output / "models.joblib").unlink(missing_ok=True)
    print(f"ECOD AUPRC:          {metrics['ecod_auprc']:.5f}")
    print(f"IForest mean AUPRC:  {metrics['iforest_auprc']:.5f}")
    print(f"Local combined:     {metrics['score']:.5f}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
