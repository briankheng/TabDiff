"""Numbers and figure data for the EDA section of the proposal."""

from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

CATEGORICAL = {
    "nsl-kdd": ["protocol_type", "service", "flag"],
    "unsw-nb15": ["proto", "service", "state"],
}

for dataset, categorical in CATEGORICAL.items():
    df = pd.read_csv(ROOT / dataset / "train.csv")
    features = [c for c in df.columns if c != "is_anomaly"]
    numeric = [c for c in features if c not in categorical]
    normal = df[df["is_anomaly"] == 0]
    attack = df[df["is_anomaly"] == 1]

    print(f"\n{dataset}: {len(df)} rows, {len(normal)} normal, {len(attack)} attack "
          f"({len(attack) / len(df):.1%})")
    print(f"{len(numeric)} numeric, {len(categorical)} categorical")
    print("constant columns:", [c for c in features if df[c].nunique() == 1])
    print("numeric columns with at most 20 values:",
          sum(1 < df[c].nunique() <= 20 for c in numeric))
    print("numeric columns that are over 90% zero in normal rows:",
          int(((normal[numeric] == 0).mean() > 0.9).sum()))

    unseen_category = pd.Series(False, index=attack.index)
    for c in categorical:
        attack_only = set(attack[c]) - set(normal[c])
        unseen_category |= attack[c].isin(attack_only)
        print(f"{c}: {df[c].nunique()} values, {len(attack_only)} used only by attacks, "
              f"covering {attack[c].isin(attack_only).mean():.1%} of attacks")

    # An attack is an "odd value" if some numeric column leaves the normal min-max range.
    outside = ((attack[numeric] < normal[numeric].min()) | (attack[numeric] > normal[numeric].max())).any(axis=1)
    print(f"attacks with a numeric value outside the normal range: {outside.mean():.1%}")
    print(f"attacks with an attack-only category: {unseen_category.mean():.1%}")
    print(f"attacks inside the normal range on every column: {1 - (outside | unseen_category).mean():.1%}")

    # KS distance between the normal and attack values of each numeric column.
    ks = pd.Series({c: ks_2samp(normal[c], attack[c]).statistic for c in numeric if normal[c].nunique() > 1})
    ks = ks.sort_values(ascending=False)
    print(f"median KS {ks.median():.3f}, columns with KS > 0.5: {(ks > 0.5).sum()}")
    print(ks.head(10).round(3).to_string())

    top = ks.head(10)[::-1]
    lines = ["feature ks"] + [f"{name.replace('_', chr(92) + '_')} {value:.3f}" for name, value in top.items()]
    (FIGURES / f"ks_{dataset}.dat").write_text("\n".join(lines) + "\n")
