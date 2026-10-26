import pandas as pd

# Load fake and real news
fake = pd.read_csv("data/raw/Fake.csv")
real = pd.read_csv("data/raw/True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine both datasets
df = pd.concat([fake, real], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save combined dataset
df.to_csv("data/raw/news.csv", index=False)

print("Dataset created successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nFirst 5 rows:")
print(df.head())