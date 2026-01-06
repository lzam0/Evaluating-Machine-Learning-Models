import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
num_samples_per_class = 20  # Total 200 samples for letters A-J
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
num_features = 63  # 21 landmarks * (x, y, z)

data = []

for label in labels:
    for i in range(num_samples_per_class):
        # Create a unique instance ID
        instance_id = f"dummy_{label}_{i}"
        
        # Generate 63 random float values for coordinates
        features = np.random.rand(num_features).tolist()
        
        # Combine ID, features, and label
        row = [instance_id] + features + [label]
        data.append(row)

# Create column names: instance_id, x0, y0, z0, ..., x20, y20, z20, label
columns = ['instance_id']
for i in range(21):
    columns.extend([f'x{i}', f'y{i}', f'z{i}'])
columns.append('label')

# Convert to DataFrame
df_dummy = pd.DataFrame(data, columns=columns)

# Save to CSV as recommended in the spec
df_dummy.to_csv('dummy_asl_features.csv', index=False)

# This file generates a dummy dataset for ASL hand landmarks
# This dataset is NOT cleaned and will need to be preprocessed

# --------------------------------------------------------------------------------------------
# Split data into training and testing sets
from sklearn.model_selection import train_test_split

# Load the extracted features dataset
data = pd.read_csv('dummy_asl_features.csv')

# Separate features and labels
X = data.drop(columns=['instance_id', 'label']).values
y = data['label'].values

# Create training testing set - random state utilised so that the split is reproducible (seed)
# MUST SPLIT INTO 60-20-20 LATER FOR TRAIN-VAL-TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Create visualisations of the training data - scatter plot of first two features (x0, y0)
plt.figure(figsize=(10, 6))

unique_labels = np.unique(y_train)
for label in unique_labels:
    mask = y_train == label
    plt.scatter(X_train[mask, 0], X_train[mask, 1], label=f'Sign {label}', alpha=0.6)

plt.title('Scatter Plot of ASL Hand Landmarks (Feature x0 vs y0)')
plt.xlabel('Wrist X-coordinate (Feature 1)')
plt.ylabel('Wrist Y-coordinate (Feature 2)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside the plot
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()
