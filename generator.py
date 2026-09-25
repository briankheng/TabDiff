"""
CS5344 synthetic normal generator: TabDiff with contrastive attack guidance.

Both judged detectors are fitted on the submission itself, so the rows we submit
are the detectors' definition of "normal". The generator has two parts.

1. Backbone. TabDiff (Shi et al., ICLR 2025), a mixed-type diffusion model, is
   trained on every training row, with the label as one more categorical column.
   The model code in tabdiff/ is the authors'; the training loop below follows
   their trainer and default configuration.
2. Contrastive guidance away from attacks. At every sampling step the denoiser
   runs twice, once told the row is normal and once told it is an attack, and
   the two predictions are combined as

       D = D(x | normal) + w * (D(x | normal) - D(x | attack))

   w = 0 samples plain label-conditional TabDiff. w > 0 pushes rows away from
   what the model associates with attacks, across all columns at once.

Samples keep to values normal traffic takes: categories that no normal training
row uses are blocked while sampling, and continuous values are clipped to the
normal range and rounded to the column's precision.
"""

import argparse
import copy
import hashlib
import math
from pathlib import Path
import random
import tomllib

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import torch

from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.modules.main_modules import Model, UniModMLP


ROOT = Path(__file__).resolve().parent
TABDIFF_CONFIG = ROOT / "tabdiff" / "configs" / "tabdiff_configs.toml"
LABEL_COLUMN = "is_anomaly"
NORMAL, ATTACK = 0, 1
# Numeric columns with at most this many distinct values (flags, TTLs, small
# counters) are modelled as categories, so no sample invents values like TTL 45.
MAX_DISCRETE_VALUES = 20
MAX_DECIMALS = 6
# TabDiff's own stand-in for minus infinity in logits.
NEG_INF = -1e6
# The denoiser, the learned numeric noise schedule and the learned categorical one.
PARTS = ["_denoise_fn", "num_schedule", "cat_schedule"]

# guidance_weight has not been tuned. epochs is TabDiff's default, which takes
# hours on a laptop GPU; the trained model is cached, so trying another weight
# only resamples.
DATASET_SETTINGS = {
    "nsl-kdd": {
        "target_n": 40000,
        "random_seed": 42,
        "guidance_weight": 1.0,
        "epochs": 8000,
    },
    "unsw-nb15": {
        "target_n": 70000,
        "random_seed": 42,
        "guidance_weight": 1.0,
        "epochs": 8000,
    },
}


def decimals(values):
    """Fewest decimals that write every value exactly, or None past MAX_DECIMALS."""
    for places in range(MAX_DECIMALS + 1):
        if np.allclose(values, np.round(values, places), rtol=0, atol=1e-9):
            return places
    return None


class TableCodec:
    """Encodes training rows for TabDiff and decodes samples into valid normal rows."""

    def __init__(self, train):
        self.features = [c for c in train.columns if c != LABEL_COLUMN]
        normal = train.loc[train[LABEL_COLUMN] == NORMAL]
        self.dtypes = normal[self.features].dtypes.to_dict()
        # Constant columns carry nothing to learn.
        self.constant = {c: train[c].iloc[0] for c in self.features if train[c].nunique() == 1}
        varying = [c for c in self.features if c not in self.constant]
        discrete = [
            c for c in varying
            if not pd.api.types.is_numeric_dtype(train[c]) or train[c].nunique() <= MAX_DISCRETE_VALUES
        ]
        self.continuous = [c for c in varying if c not in discrete]
        # The label goes last; it is the column guidance conditions on.
        self.discrete = discrete + [LABEL_COLUMN]
        self.categories = {c: np.sort(train[c].unique()) for c in self.discrete}
        if self.categories[LABEL_COLUMN].tolist() != [NORMAL, ATTACK]:
            raise ValueError(f"'{LABEL_COLUMN}' must contain both 0 and 1.")
        self.normal_categories = {c: np.isin(self.categories[c], normal[c].unique()) for c in self.discrete}
        self.low = normal[self.continuous].min()
        self.high = normal[self.continuous].max()
        self.decimals = {c: decimals(normal[c].to_numpy(dtype=float)) for c in self.continuous}

    @property
    def category_sizes(self):
        return np.array([len(self.categories[c]) for c in self.discrete])

    def encode(self, frame):
        """Continuous columns as floats and discrete columns as category codes."""
        numeric = frame[self.continuous].to_numpy(dtype=float)
        codes = np.column_stack([
            pd.Categorical(frame[c], categories=self.categories[c]).codes for c in self.discrete
        ]).astype(np.int64)
        return numeric, codes

    def decode(self, numeric, codes):
        rows = {}
        for c in self.features:
            if c in self.constant:
                values = np.repeat(self.constant[c], len(codes))
            elif c in self.continuous:
                values = np.clip(numeric[:, self.continuous.index(c)], self.low[c], self.high[c])
                if self.decimals[c] is not None:
                    values = np.round(values, self.decimals[c])
            else:
                column = codes[:, self.discrete.index(c)]
                if not self.normal_categories[c][column].all():
                    raise ValueError(f"{c}: sample has a category that no normal training row uses.")
                values = self.categories[c][column]
            rows[c] = values
        return pd.DataFrame(rows).astype(self.dtypes)


class ContrastiveGuidance(torch.nn.Module):
    """Wraps TabDiff's denoiser to steer sampling toward normal rows and away from attacks."""

    def __init__(self, model, codec, weight):
        super().__init__()
        self.model = model
        self.weight = weight
        # TabDiff one-hot encodes each category column with an extra mask slot.
        slots = codec.category_sizes + 1
        self.label_start = int(slots[:-1].sum())
        self.label_slots = int(slots[-1])
        # Block categories no normal row uses, including the attack label. The mask
        # slot is left to TabDiff, which excludes it itself.
        blocked = np.concatenate([np.append(~codec.normal_categories[c], False) for c in codec.discrete])
        self.register_buffer("blocked", torch.from_numpy(blocked))

    def with_label(self, x_cat, label):
        x_cat = x_cat.clone()
        x_cat[:, self.label_start:self.label_start + self.label_slots] = 0
        x_cat[:, self.label_start + label] = 1
        return x_cat

    def forward(self, x_num, x_cat, t, sigma=None):
        num, cat = self.model(x_num, self.with_label(x_cat, NORMAL), t, sigma=sigma)
        if self.weight:
            num_attack, cat_attack = self.model(x_num, self.with_label(x_cat, ATTACK), t, sigma=sigma)
            num = num + self.weight * (num - num_attack)
            cat = cat + self.weight * (cat - cat_attack)
        return num, cat.masked_fill(self.blocked, NEG_INF)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_diffusion(codec, config, device):
    params = config["diffusion_params"]
    sizes = codec.category_sizes
    backbone = UniModMLP(
        **config["unimodmlp_params"], d_numerical=len(codec.continuous),
        categories=(sizes + 1).tolist(),  # one extra slot for the mask category
    )
    model = Model(backbone, **params["edm_params"])
    # The authors' main.py uses per-column learnable noise schedules by default.
    return UnifiedCtimeDiffusion(
        num_classes=sizes, num_numerical_features=len(codec.continuous), denoise_fn=model,
        y_only_model=None, num_timesteps=params["num_timesteps"],
        scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column",
        noise_dist=params["noise_dist"], edm_params=params["edm_params"],
        noise_dist_params=params["noise_dist_params"],
        noise_schedule_params=params["noise_schedule_params"],
        sampler_params=params["sampler_params"], device=device,
    ).to(device)


def update_ema(target, source, rate):
    for t, s in zip(target.parameters(), source.parameters()):
        t.detach().mul_(rate).add_(s.detach(), alpha=1 - rate)


def train_tabdiff(diffusion, x, settings, epochs):
    """TabDiff's trainer: AdamW with plateau decay, an annealed continuous-loss
    weight and a per-epoch EMA. Returns the EMA weights with the lowest training
    loss from the second half of training, as the authors' checkpointing does."""
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=settings["factor"], patience=settings["reduce_lr_patience"],
    )
    ema = {name: copy.deepcopy(getattr(diffusion, name)) for name in PARTS}
    for module in ema.values():
        for p in module.parameters():
            p.detach_()

    def epoch_losses(training, closs_weight=1.0):
        order = torch.randperm(len(x), device=x.device) if training else torch.arange(len(x), device=x.device)
        totals = torch.zeros(2, device=x.device)
        for start in range(0, len(x), settings["batch_size"]):
            batch = x[order[start:start + settings["batch_size"]]]
            if training:
                diffusion.train()
                optimizer.zero_grad()
                dloss, closs = diffusion.mixed_loss(batch)
                (settings["d_lambda"] * dloss + closs_weight * closs).backward()
                optimizer.step()
            else:
                diffusion.eval()
                with torch.no_grad():
                    dloss, closs = diffusion.mixed_loss(batch)
            totals += torch.stack([dloss.detach(), closs.detach()]) * len(batch)
        dloss, closs = (totals / len(x)).tolist()
        return round(dloss, 4), round(closs, 4)

    best_loss, best_state = math.inf, None
    for epoch in range(epochs):
        dloss, closs = epoch_losses(True, settings["c_lambda"] * (1 - epoch / epochs))
        if math.isnan(closs):
            raise FloatingPointError("TabDiff's continuous loss became NaN.")
        plateau.step(dloss + closs)
        for name in PARTS:
            update_ema(ema[name], getattr(diffusion, name), settings["ema_decay"])
        if epoch + 1 > epochs // 2:
            live = {name: getattr(diffusion, name) for name in PARTS}
            for name in PARTS:
                setattr(diffusion, name, ema[name])
            ema_loss = sum(epoch_losses(False))
            for name in PARTS:
                setattr(diffusion, name, live[name])
            if ema_loss < best_loss:
                best_loss = ema_loss
                best_state = {name: copy.deepcopy(ema[name].state_dict()) for name in PARTS}
        if (epoch + 1) % 500 == 0 or epoch + 1 == epochs:
            print(f"Epoch {epoch + 1}/{epochs}: categorical loss {dloss}, continuous loss {closs}, "
                  f"best EMA loss {best_loss}", flush=True)
    return best_state


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic normal samples for a dataset.")
    parser.add_argument("dataset", choices=DATASET_SETTINGS)
    parser.add_argument("--target-n", type=int, help="Override the number of generated rows.")
    parser.add_argument("--random-seed", type=int, help="Override the random seed.")
    parser.add_argument("--guidance-weight", type=float, help="Contrastive guidance weight w; 0 disables guidance.")
    parser.add_argument("--epochs", type=int, help="Override the number of TabDiff training epochs.")
    parser.add_argument("--checkpoint", type=Path,
                        help="Trained model cache; defaults to results/models/<dataset>_seed<seed>_epochs<epochs>.pt.")
    parser.add_argument("--output", type=Path, help="Output CSV path; defaults to <dataset>/submission.csv.")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU even if CUDA is available.")
    args = parser.parse_args()

    settings = DATASET_SETTINGS[args.dataset].copy()
    for name in settings:
        value = getattr(args, name)
        if value is not None:
            settings[name] = value

    dataset_dir = ROOT / args.dataset
    TRAIN_CSV = dataset_dir / "train.csv"
    OUTPUT_CSV = args.output if args.output is not None else dataset_dir / "submission.csv"
    TARGET_N = settings["target_n"]
    RANDOM_SEED = settings["random_seed"]
    EPOCHS = settings["epochs"]
    if EPOCHS < 1:
        raise ValueError("--epochs must be at least 1.")
    CHECKPOINT = args.checkpoint or ROOT / "results" / "models" / f"{args.dataset}_seed{RANDOM_SEED}_epochs{EPOCHS}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")


    train = pd.read_csv(TRAIN_CSV)

    if LABEL_COLUMN not in train.columns:
        raise ValueError(f"Training data must contain '{LABEL_COLUMN}'.")

    if train.isna().any().any():
        raise ValueError("Training data contains missing values.")

    codec = TableCodec(train)

    print(f"Dataset: {args.dataset}")
    print(f"Settings: {settings}")
    print(f"Train shape: {train.shape}, normal rows: {int((train[LABEL_COLUMN] == NORMAL).sum())}")
    print(f"Columns: {len(codec.continuous)} continuous, {len(codec.discrete) - 1} discrete, "
          f"{len(codec.constant)} constant")
    print(f"Device: {device}")


    # TabDDPM and TabDiff both map continuous columns to a normal distribution first.
    seed_everything(RANDOM_SEED)
    numeric, codes = codec.encode(train)
    transformer = QuantileTransformer(
        output_distribution="normal", n_quantiles=max(min(len(train) // 30, 1000), 10),
        subsample=int(1e9), random_state=RANDOM_SEED,
    )
    x = torch.tensor(np.hstack([transformer.fit_transform(numeric), codes]), dtype=torch.float32, device=device)
    config = tomllib.loads(TABDIFF_CONFIG.read_text())
    diffusion = build_diffusion(codec, config, device)

    # The model depends on the data, seed and epochs, but not on the guidance weight.
    trained_on = {
        "train_sha256": hashlib.sha256(TRAIN_CSV.read_bytes()).hexdigest(),
        "random_seed": RANDOM_SEED,
        "epochs": EPOCHS,
    }
    cached = torch.load(CHECKPOINT, map_location=device, weights_only=True) if CHECKPOINT.exists() else None
    if cached is not None and cached["trained_on"] == trained_on:
        print(f"Loaded trained model from {CHECKPOINT}")
        state = cached["state"]
    else:
        print(f"Training TabDiff for {EPOCHS} epochs")
        state = train_tabdiff(diffusion, x, config["train"]["main"], EPOCHS)
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"trained_on": trained_on, "state": state}, CHECKPOINT)
        print(f"Saved trained model to {CHECKPOINT}")
    for name in PARTS:
        getattr(diffusion, name).load_state_dict(state[name])


    diffusion._denoise_fn = ContrastiveGuidance(diffusion._denoise_fn, codec, settings["guidance_weight"]).to(device)
    diffusion.eval()
    # Reseed so a loaded model samples exactly what a freshly trained one would.
    seed_everything(RANDOM_SEED)
    batch_size = min(config["sample"]["batch_size"], TARGET_N)
    sample = diffusion.sample_all(TARGET_N, batch_size, keep_nan_samples=False).cpu().numpy()
    width = len(codec.continuous)
    synthetic = codec.decode(
        transformer.inverse_transform(sample[:, :width]), sample[:, width:].round().astype(np.int64),
    )


    submission = synthetic[codec.features].copy()
    submission.insert(0, "id", np.arange(TARGET_N))


    if len(submission) != TARGET_N:
        raise ValueError(f"Submission must contain exactly {TARGET_N} rows.")

    if submission.columns.tolist() != ["id"] + codec.features:
        raise ValueError("Submission columns are incorrect.")

    if submission.isna().any().any():
        raise ValueError("Submission contains missing values.")

    numeric_values = submission.select_dtypes(include=[np.number]).to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError("Submission contains NaN or infinite values.")


    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_CSV, index=False)

    print("\nSubmission generated successfully.")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Shape: {submission.shape}")
    print(submission.head())


if __name__ == "__main__":
    main()
