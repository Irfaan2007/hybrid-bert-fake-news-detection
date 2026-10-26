import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "news.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "news_clean.csv"

# Load raw dataset
df = pd.read_csv(RAW_DATA_PATH)

required_columns = {"title", "text", "label"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(f"news.csv is missing required columns: {sorted(missing_columns)}")

labels = set(df["label"].dropna().unique())
if not labels.issubset({0, 1}):
    raise ValueError(f"Expected binary labels 0 and 1, found: {sorted(labels)}")

print("Original dataset:", df.shape)

# Remove rows where title or text is missing
df = df.dropna(subset=["title", "text"])

# Convert title and text to string
df["title"] = df["title"].astype(str)
df["text"] = df["text"].astype(str)

# Remove empty title/text
df = df[
    (df["title"].str.strip() != "") &
    (df["text"].str.strip() != "")
]

# Remove extremely short articles
df = df[df["text"].str.len() >= 50]

# Remove exact duplicate rows
df = df.drop_duplicates()

# Remove duplicate articles based on title + text
df = df.drop_duplicates(subset=["title", "text"])

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Create processed folder if it doesn't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Save cleaned dataset
df.to_csv(PROCESSED_PATH, index=False)

print("\nCleaning completed!")
print("Cleaned dataset:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSaved to:")
print(PROCESSED_PATH)
