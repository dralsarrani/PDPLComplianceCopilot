import glob
import os
import re
import pandas as pd

CATEGORIES = [
    "First Party Collection/Use",
    "Third Party Sharing/Collection",
    "User Choice/Control",
    "User Access, Edit and Deletion",
    "Data Retention",
    "Data Security",
    "Policy Change",
    "Do Not Track",
    "International and Specific Audiences",
    "Other",
]
COLS = ["ann_id", "batch", "annotator", "policy_id", "segment_id",
        "category", "attrs", "date", "url"]


def load_annotations(root="data/opp115"):
    frames = []
    for f in sorted(glob.glob(f"{root}/annotations/*.csv")):
        df = pd.read_csv(f, header=None, names=COLS)
        df["policy_file"] = os.path.splitext(os.path.basename(f))[0]
        frames.append(df[["policy_file", "annotator", "segment_id", "category"]])
    return pd.concat(frames, ignore_index=True)


def load_segments(policy_file, root="data/opp115"):
    matches = glob.glob(f"{root}/sanitized_policies/{policy_file}.*")
    assert matches, f"no policy text for {policy_file}"
    with open(matches[0], encoding="utf-8") as fh:
        return [s.strip() for s in fh.read().split("|||")]


def build_table(root="data/opp115", min_votes=2):
    ann = load_annotations(root)
    unknown = set(ann["category"]) - set(CATEGORIES)
    assert not unknown, f"unexpected categories: {unknown}"
    rows = []
    for pf, g in ann.groupby("policy_file"):
        segs = load_segments(pf, root)
        votes = g.groupby(["segment_id", "category"])["annotator"].nunique().unstack(fill_value=0)
        for sid, text in enumerate(segs):
            row = {"policy_file": pf, "segment_id": sid, "text": text}
            for c in CATEGORIES:
                row[c] = int(sid in votes.index and c in votes.columns
                             and votes.loc[sid, c] >= min_votes)
            rows.append(row)
    return pd.DataFrame(rows)

def clean_text(t):
    t = re.sub(r"<[^>]+>", " ", str(t))   # strip <br> etc.
    return re.sub(r"\s+", " ", t).strip()