import pandas as pd
from pathlib import Path

# Locate the evaluation file
project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "evaluation_marketing.csv"

# Read the manually evaluated results
df = pd.read_csv(file_path, comment="#")

# Use only rows that have been evaluated
df = df.dropna(subset=["relevant"])
df["relevant"] = df["relevant"].astype(int)

# Calculate Precision@5
top5 = df[df["rank"] <= 5]

precision_at_5 = top5["relevant"].sum() / 5

print("Evaluation Results")
print("------------------")
print(f"Precision@5: {precision_at_5:.2%}")
 