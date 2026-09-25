# CS5344

`generator.py` generates synthetic normal records for either coursework dataset.
It trains TabDiff, a tabular diffusion model, on the selected folder's `train.csv`,
samples normal rows with contrastive guidance away from attacks, and writes
`submission.csv` there.

```text
generator.py              Generator, dataset settings and guidance
tabdiff/                  TabDiff model code from the authors (MIT licence)
judge.py                    Local ECOD and Isolation Forest evaluation
nsl-kdd/
    train.csv              NSL-KDD coursework training data
    sample_submission.csv  Provided sample submission
    submission.csv         Generated output
unsw-nb15/
    train.csv              UNSW-NB15 coursework training data
    sample_submission.csv  Provided sample submission
    submission.csv         Generated output
datasets/                  Full public datasets and source notes
tests/                     Generator and judge checks
results/                   Local evaluation results and cached trained models
```

Install dependencies in your virtual environment. Install torch first from the
wheel index that matches your machine; use `https://download.pytorch.org/whl/cpu`
without an NVIDIA GPU:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Run either dataset from the project directory:

```bash
python generator.py nsl-kdd
python generator.py unsw-nb15
```

Training uses TabDiff's default 8,000 epochs and takes hours on a laptop GPU.
The trained model is saved to `results/models/<dataset>_seed<seed>_epochs<epochs>.pt`
and reused while the training data, seed and epoch count match, so another
guidance weight only reruns sampling. Delete the file to retrain.

Edit `DATASET_SETTINGS` near the top of `generator.py` to change each dataset's
defaults independently:

| Setting | NSL-KDD | UNSW-NB15 | Meaning |
| --- | ---: | ---: | --- |
| `target_n` | 40000 | 70000 | Required submission row count |
| `random_seed` | 42 | 42 | Seed for training and sampling |
| `guidance_weight` | 1.0 | 1.0 | Contrastive guidance weight `w`; `0` samples plain label-conditional TabDiff |
| `epochs` | 8000 | 8000 | TabDiff training epochs, the authors' default |

`guidance_weight` has not been tuned on either dataset.

You can also override a setting for one run without editing the defaults:

```bash
python generator.py nsl-kdd --guidance-weight 0
python generator.py unsw-nb15 --guidance-weight 2 --random-seed 7
python generator.py nsl-kdd --epochs 50 --target-n 100 --output experiments/small.csv
```

Keep `target_n` at the required value when preparing a Kaggle submission.
`--output` paths are relative to your current working directory. Training inputs
and default output paths are resolved relative to the script, so it also works
when invoked from another directory. Re-running replaces the selected output.
`--checkpoint` picks another model cache and `--cpu` ignores the GPU.

## How the generator works

Both judged detectors are fitted on the submitted rows, so the submission is
really the detectors' definition of normal.

**Backbone.** TabDiff ([Shi et al., ICLR 2025](https://arxiv.org/abs/2410.20626))
diffuses continuous and categorical columns jointly, with a learnable noise
schedule per column. It is trained on every training row, attacks included,
with `is_anomaly` as one more categorical column. Continuous columns are
quantile-mapped to a normal distribution first, as in the authors' preprocessing.
Numeric columns with at most 20 distinct values, such as flags and TTLs, are
treated as categories, and constant columns are held fixed. Training follows
the authors' trainer and configuration: AdamW with plateau decay, an annealed
continuous-loss weight, and the EMA weights with the lowest training loss from
the second half of training.

**Contrastive guidance.** At every sampling step the denoiser runs twice, once
told the row is normal and once told it is an attack, and the two predictions
are combined:

```text
D = D(x | normal) + w * (D(x | normal) - D(x | attack))
```

The same combination is applied to the categorical logits. `w = 0` samples
TabDiff conditioned on the normal label. `w > 0` pushes every column, jointly,
away from what the model associates with attacks, so attack rows fall further
into the tails of the synthetic normal data the detectors are fitted on. The
previous generator did a per-column version of this by hand; guidance learns it
across columns. The guidance wraps the denoiser, so the TabDiff code is used
unchanged.

On NSL-KDD, with a model trained for only 300 epochs as a check, a
normal-versus-attack classifier fitted on `train.csv` gave the samples a mean
attack probability of 0.317 at `w = 0`, 0.201 at `w = 1` and 0.130 at `w = 2`.
The normal training rows, which the classifier was fitted on, score 0.0001.
That confirms the direction of the push, not the detection score, which the
local judge has not measured for this generator.

Every categorical value is one that normal training rows take: categories used
only by attacks are blocked while sampling. Continuous values are clipped to the
normal range and rounded to the column's precision, so dtypes, ranges and
categories stay valid. The generator reads only `<dataset>/train.csv`. The full
public datasets under `datasets/` have different schemas and are used by the
judge, not by the generator.

Run the checks with:

```bash
python -m unittest discover -s tests -v
```

## Choosing the backbone

TabDiff was chosen after a one-off benchmark, whose code is not kept here. Each
generator was trained on the normal training rows with its published settings
and scored with `judge.py` on two disjoint 20,000-row splits of the
public data that exclude the coursework rows and the judge's default split.
Scores are local combined AUPRC averaged over the two splits. Generative models
ran once. The bootstrap and the previous generator ran with five seeds each, with
standard deviations of 0.0003 to 0.0014.

| Generator | NSL-KDD | UNSW-NB15 |
| --- | ---: | ---: |
| Bootstrap of normal rows | 0.9334 | 0.5962 |
| Previous generator: bootstrap plus per-column reshaping | 0.9327 | **0.6515** |
| CTGAN | 0.9351 | 0.5513 |
| TVAE | 0.9266 | 0.5918 |
| TabDDPM, with predicted values clamped while sampling | 0.8910 | 0.5713 |
| TabDiff | **0.9356** | not finished |

- No generator beat the bootstrap by more than 0.0022. TabDiff came closest to
  the real normal data: on NSL-KDD its mean per-column KS distance to the normal
  training rows was 0.014, against 0.08 to 0.14 for the other generators.
- On UNSW-NB15, using the labels mattered far more than the generator. The
  reshaping step was worth +0.055 over the bootstrap. Applied to CTGAN, TVAE and
  TabDDPM samples it recovered most of the gap, reaching 0.629 to 0.643. That is
  the motivation for label-aware guidance.
- TabDDPM's published sampler diverged on NSL-KDD, scoring about 0.55, because
  heavy point masses at the quantile bounds blow up its first reverse steps.

## Local judge

The judge fits detectors on the submitted synthetic features and evaluates their
anomaly scores on labeled real records. It implements the setup in
[briefing slide 11](CS5344_project_briefing.pdf): ECOD, Isolation Forest with
500 trees for each of 10 fixed seeds, and the mean of their AUPRCs.

Run the original submitted baselines:

```bash
python judge.py nsl-kdd --save-models
python judge.py unsw-nb15 --save-models
```

The default input is `<dataset>/sample_submission.csv`. To evaluate a new output:

```bash
python judge.py nsl-kdd --submission nsl-kdd/submission.csv
python judge.py unsw-nb15 --submission unsw-nb15/submission.csv
```

Results go to `results/<dataset>/<submission filename without .csv>/`:

- `metrics.json`: detector scores, settings, data hashes, and package versions.
- `evaluation.csv`: the real records used for this run.
- `predictions.csv`: ECOD and individual Isolation Forest anomaly scores.
- `models.joblib`: fitted encoder and detectors, when `--save-models` is used.

Re-running the same filename replaces that result directory's output files.
If `--save-models` is omitted, any old `models.joblib` in that directory is removed.
Use `--output-dir results/my-experiment` to keep a separate experiment.

### Local evaluation data

The [problem formulation](CS5344_anomaly_detection_problem_formulation.pdf)
states that the validation and test sets are private and include unseen attack
types. Those partitions cannot be recovered from the two reported scores.

The local judge uses the downloaded original NSL-KDD train/test files and the
published UNSW-NB15 train/test tables. The latter include `rate` and all 42
coursework features, unlike the four large raw UNSW files. It excludes every
reference row whose features match a released coursework training row, including
duplicate matches. Numeric storage types are normalized before row matching.

The default evaluation sample has 20,000 records, uses seed 42, and matches the
coursework training anomaly fraction: about one third for NSL-KDD and 30% for
UNSW-NB15. This is a local evaluation choice. The private class ratios and attack
mixtures are unknown. The split is independent of the submitted synthetic data,
so different submissions use the same evaluation rows with the same settings.

You can supply your own evaluation CSV with the coursework feature columns and
`is_anomaly` using `--evaluation-csv path/to/valid.csv`. Its overlap with the
coursework training data is reported. A holdout drawn from training data should
also be excluded when generating synthetic samples.

### Assumptions and score interpretation

The organizers have not specified all implementation details in the PDFs. Local
choices are:

| Detail | Local choice |
| --- | --- |
| Isolation Forest seed values | 0 through 9 |
| Isolation Forest samples per tree | 256 |
| Categorical features | One-hot encoding fitted on synthetic samples only; unknown categories become zeros |
| Numeric features | Original values, without scaling |
| AUPRC definition | Scikit-learn average precision |
| Isolation Forest aggregation | Mean of the 10 per-seed average precision scores |
| Combined score | `(ECOD AUPRC + mean IForest AUPRC) / 2` |

The report also includes trapezoidal precision-recall area and the alternative
of averaging Isolation Forest anomaly scores before calculating average
precision. [Average precision and trapezoidal PR area differ](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

The ECOD implementation is [PyOD ECOD](https://pyod.readthedocs.io/en/latest/_modules/pyod/models/ecod.html).
Its `fit` call receives only synthetic samples, but `decision_function` pools
those features with the evaluation batch to compute ranks. No evaluation labels
enter the detector, but scores depend on the evaluation batch. Keep that batch
fixed when comparing submissions; scoring it in separate chunks changes ECOD's
results. The PDFs do not say whether the organizers use this implementation or
a version with empirical distributions fixed from training data.

Isolation Forest uses scikit-learn's
[IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).
Its `score_samples` values are negated so that larger scores mean more anomalous.
Threshold-based predictions are not used for either detector's AUPRC.

The reported Kaggle public scores below are comparison references, not targets
used to fit or calibrate the judge. Local scores are not estimates of the exact
Kaggle score.

The initial full judge runs on the submitted sample files gave:

| Dataset | Local ECOD | Local IForest mean | Local combined | Reported Kaggle public |
| --- | ---: | ---: | ---: | ---: |
| NSL-KDD | 0.87336 | 0.93179 | 0.90258 | 0.89919 |
| UNSW-NB15 | 0.62971 | 0.69878 | 0.66424 | 0.38281 |

The UNSW-NB15 gap is substantial. Similarity on NSL-KDD does not establish that
the local judge reproduces the organizers' pipeline or private evaluation data.

Edit `JUDGE_SETTINGS` in `judge.py` to change local defaults. For a quick smoke
test, use fewer trees and seeds:

```bash
python judge.py nsl-kdd --trees 20 --seeds 0 1 --evaluation-size 1000 --output-dir results/smoke
```
