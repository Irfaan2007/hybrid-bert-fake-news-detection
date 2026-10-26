import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/news_clean.csv")

print("========== CLEAN DATASET ==========")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nSample Data:")
print(df[["title", "text", "label"]].head(5))