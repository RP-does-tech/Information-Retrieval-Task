
import pandas as pd

# Load our CSV dataset
df = pd.read_csv("data/learning_resources.csv")

# 1. Display dataset dimensions
print("Dataset Shape:", df.shape)

# 2. Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# 3. Display first 5 records
print("\nFirst 5 Records:")
print(df.head())

# 4. Check data types and missing values
print("\nDataset Information:")
df.info()

# 5. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 6. Check duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# 7. Count resource types
print("\nResource Type Distribution:")
print(df["resource_type"].value_counts())
