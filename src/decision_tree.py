import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from collections import Counter


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

#--------------------------------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Min sample split and max depth act as stopping conditions for the tree growth
param_grid = {
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Load the extracted features dataset 
dt_clf = DecisionTreeClassifier(random_state=2)

# Splits training data into 5 pieces
# Trains model on 4 pieces and tests on the 5th piece
# Repeats this for every combination for hyper parameters in the grid
grid_search = GridSearchCV(dt_clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Identify best parameters from the grid search
print(f"Best Parameters: {grid_search.best_params_}")
best_tree = grid_search.best_estimator_
# Explain why these parameters were chosen (Highest Mean cross validation accuracy)

# Evaluate the best model on the test set
y_pred = best_tree.predict(X_test)
print(classification_report(y_test, y_pred)) # This gives you Accuracy and Sensitivity (Recall)!

#--------------------------------------------------------------------------------------------
# Visualisation of the confusion matrix

# Create confusion matrix data based of y_test and predictions
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix using seaborn heatmaps
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Best Decision Tree: Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

plt.savefig('decision_tree_confusion_matrix.png')

#--------------------------------------------------------------------------------------------