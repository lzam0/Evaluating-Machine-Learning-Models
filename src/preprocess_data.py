# part 2b
import os
import pandas as pd
import matplotlib.pyplot as plt

# CSV
INPUT_CSV = "data/extracted_features/hand_landmarks.csv"
OUTPUT_CSV = "data/cleaned_features/hand_landmarks_cleaned.csv"

# pandas dataframe
df = pd.read_csv(INPUT_CSV)

print("Initial dataset shape:", df.shape)

# Check for missing values
missing_values = df.isna().sum().sum() 
print("\nMissing values per column:")
print(missing_values)

# Remove duplicated instance IDs
before = len(df)
df = df.drop_duplicates(subset="instance_id")
after = len(df)

print(f"Removed {before - after} duplicated instances")

# Analyse class distribution
class_counts = df["label"].value_counts().sort_index()

print("\nClass distribution:")
print(class_counts)

# Graph visualisation
plt.figure(figsize=(8, 5))
class_counts.plot(kind="bar")
plt.xlabel("ASL Class")
plt.ylabel("Number of Instances")
plt.title("Class Distribution After Cleaning")
plt.tight_layout()
plt.show()

# Feature Scaling - not really needed as MediaPipe has the coordinates normalised
# Separate features and labels
# feature_cols = [col for col in df.columns if col.startswith(("x", "y", "z"))]
# X = df[feature_cols]
# y = df["label"]

# print("Feature range check:")
# print(X.describe().loc[["min", "max"]])

# export the cleaned dataset
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print("Cleaned dataset saved to:", OUTPUT_CSV)
print("Final dataset shape:", df.shape)