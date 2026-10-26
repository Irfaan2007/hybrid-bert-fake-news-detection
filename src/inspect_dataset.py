import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Load combined dataset
df = pd.read_csv(BASE_DIR / "data" / "raw" / "news.csv")

print("========== DATASET INFO ==========")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nEmpty Titles:")
print((df["title"].fillna("").str.strip() == "").sum())

print("\nEmpty Text:")
print((df["text"].fillna("").str.strip() == "").sum())

print("\nText Length Statistics:")
print(df["text"].fillna("").str.len().describe())

print("\n========== SAMPLE DATA ==========")
print(df[["title", "text", "label"]].head(3))
