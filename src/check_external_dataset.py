import pandas as pd

files = {
"Fake": r"C:\Users\2007m\Downloads\Fake.csv",
"Real": r"C:\Users\2007m\Downloads\True.csv",
}

for name, path in files.items():
	print(f"\n{'=' * 50}")
	print(f"{name} dataset")
	print("=" * 50)

	df = pd.read_csv(path)

	print("Shape:", df.shape)
	print("Columns:", df.columns.tolist())
	print("\nFirst 2 rows:")
	print(df.head(2).to_string())
	print("\nMissing values:")
	print(df.isnull().sum())