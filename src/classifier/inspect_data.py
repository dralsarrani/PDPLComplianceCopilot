import glob
import pandas as pd

files = sorted(glob.glob("data/opp115/annotations/*.csv"))
print("policies:", len(files))

df = pd.read_csv(files[0], header=None)
print(df.shape)
print(df.head(5).to_string())
print(df[5].value_counts())   # column 5 should be the category name