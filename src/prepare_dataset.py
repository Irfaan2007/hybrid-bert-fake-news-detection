import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"

# Load fake and real news
fake = pd.read_csv(RAW_DIR / "Fake.csv")
real = pd.read_csv(RAW_DIR / "True.csv")

required_columns = {"title", "text"}
for name, frame in (("Fake.csv", fake), ("True.csv", real)):
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing_columns)}"
        )

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine both datasets
df = pd.concat([fake, real], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save combined dataset
df.to_csv(RAW_DIR / "news.csv", index=False)

print("Dataset created successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nFirst 5 rows:")
print(df.head())
