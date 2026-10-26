import pandas as pd
import os

# Load raw dataset
df = pd.read_csv("data/raw/news.csv")

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
os.makedirs("data/processed", exist_ok=True)

# Save cleaned dataset
df.to_csv("data/processed/news_clean.csv", index=False)

print("\nCleaning completed!")
print("Cleaned dataset:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSaved to:")
print("data/processed/news_clean.csv")