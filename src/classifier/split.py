import json
import os

import numpy as np

from src.classifier.data import CATEGORIES

SPLIT_PATH = "configs/split.json"


def make_split(df, seed=42, min_pos=3):
    """70/15/15 split by policy. Retry seeds until val and test
    each have at least min_pos positives for every category."""
    policies = sorted(df["policy_file"].unique())
    n = len(policies)
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    for s in range(seed, seed + 500):
        p = [str(x) for x in np.random.default_rng(s).permutation(policies)]
        split = {
            "seed": s,
            "train": p[:n_train],
            "val": p[n_train:n_train + n_val],
            "test": p[n_train + n_val:],
        }
        ok = all(
            df[df["policy_file"].isin(split[k])][c].sum() >= min_pos
            for k in ("val", "test") for c in CATEGORIES
        )
        if ok:
            return split
    raise RuntimeError("no valid split found, lower min_pos")


def save_split(split, path=SPLIT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path=SPLIT_PATH):
    with open(path) as f:
        return json.load(f)


def apply_split(df, split):
    return {k: df[df["policy_file"].isin(split[k])].reset_index(drop=True)
            for k in ("train", "val", "test")}