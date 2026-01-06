import math
from collections import Counter

import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

# Load the extracted features dataset
data = pd.read_csv('data/extracted_features/hand_landmarks.csv') # Replace with actual path to your CSV file

# Separate features and labels
X = data.drop(columns=['instance_id', 'label']).values
y = data['label'].values

# Create training testing set - random state utilised so that the split is reproducible (seed)
# First split off test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=2
)

# Then split the remaining data into training (60%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=2
)

print(f"Training size: {len(X_train)}") # ~60%
print(f"Validation size: {len(X_val)}") # ~20%
print(f"Testing size: {len(X_test)}") # ~20%
print(f"Total Dataset size: {len(X)}") # 100%