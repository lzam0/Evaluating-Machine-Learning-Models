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

#--------------------------------------------------------------------------------------------
# Distances functions

# Euclidean distance - is a straight line distance between two points (like a ruler)
def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

# Manhattan distance - is the sum of the absolute difference of their coordinates (like navigating a grid of city blocks)
def manhattan_distance(point1, point2):
    return sum(abs(a - b) for a, b in zip(point1, point2))

#--------------------------------------------------------------------------------------------

# Class KNN
class KNN:
    def __init__(self, k, distance_metric):
        self.k = k
        self.distance_metric = distance_metric

    # Take training input X and output y
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # take in numpy array of new datapoints to predict
    def predict(self, new_points):
        # create a list comprehension to store predictions - iterates through each points in new_points
        predictions = [self.predict_class(point) for point in new_points]
        return predictions
    
    def predict_class(self, new_point):
        # calculate the distance between new point and every point in the data
        # either use euclidean or manhattan distance based of the input parameter
        distance = [self.distance_metric(point, new_point) for point in self.X_train]
        #distance = [euclidean_distance(point, new_point) for point in self.X_train]

        # find the indicies of k smallest distances - with the position of the list k smallest distances
        k_nearest_indices = sorted(
            range(len(distance)),
            key=lambda i: distance[i]
        )[:self.k]

        # retrieve the corresponding labels for the k nearest indices
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        most_common = Counter(k_nearest_labels).most_common(1)[0][0] # get the most common label

        return most_common

# --------------------------------------------------------------------------------------------
# Helper function to create folds for cross validation
def create_folds(X, y, k_folds=5, seed=2):
    # Set random seed for reproducibility (Also utilised inside of Decision Trees)
    random.seed(seed)

    # Shuffle the data indices
    indices = list(range(len(X)))
    random.shuffle(indices)

    # Split indices into k folds
    fold_size = len(X) // k_folds
    folds = []

    # Iterate through the k folds and create the fold indices
    for i in range(k_folds):
        start = i * fold_size
        end = start + fold_size if i != k_folds - 1 else len(X)
        fold_indices = indices[start:end]
        folds.append(fold_indices) # Append the fold indices to the folds list

    return folds


# --------------------------------------------------------------------------------------------
# Train and evaluate the KNN model

# Distance Calculations
distance_metrics = {
    "Euclidean": euclidean_distance,
    "Manhattan": manhattan_distance
}

# Different k values to test
k_values = [3,5,10,25,50] # testing different k values 

# We do this on the 80% temp data (Training + Validation)
folds = create_folds(X_train, y_train, k_folds=5)

results = []

for distance_name, distance_func in distance_metrics.items():
    print(f"\nDistance Metric: {distance_name}")

    # Iterate through different k values (hyperparameter)
    for k in k_values:
        fold_accuracies = []

        for i in range(len(folds)):
            # Validation fold
            val_indices = folds[i]

            # Training folds (all except 1 validation fold)
            train_indices = []
            for j in range(len(folds)):
                if j != i:
                    train_indices.extend(folds[j])

            # Build CV train/val sets
            X_cv_train = X_train[train_indices]
            y_cv_train = y_train[train_indices]
            X_cv_val = X_train[val_indices]
            y_cv_val = y_train[val_indices]

            # Train KNN on 4 folds
            knn = KNN(k=k, distance_metric=distance_func)
            knn.fit(X_cv_train, y_cv_train)

            # Validate on 1 fold
            predictions = knn.predict(X_cv_val)
            accuracy = np.mean(predictions == y_cv_val)
            fold_accuracies.append(accuracy)

        # CV results
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)

        # Store the results
        results.append({
            "Distance": distance_name,
            "k": k,
            "Mean Accuracy": mean_acc,
            "Std Dev": std_acc
        })

        print(f"k={k} | Mean Accuracy={mean_acc:.3f} | Std={std_acc:.3f}")


# --------------------------------------------------------------------------------------------

# Select best hyperparameters based on CV results

# Find the row with the highest mean accuracy
best_row = max(results, key=lambda x: x['Mean Accuracy'])

# Extract best k and distance metric
best_k = best_row['k']
best_distance_name = best_row['Distance']
best_distance_func = distance_metrics[best_distance_name]

print(f"\nBest Model from CV -> k={best_k}, Distance Metric={best_distance_name}, Mean Accuracy={best_row['Mean Accuracy']:.3f}")

# --------------------------------------------------------------------------------------------
# Retrain the model based on full training set with best hyperparameters

# Retrain KNN model
knn_best = KNN(k=best_k, distance_metric=best_distance_func)
knn_best.fit(X_train, y_train)

# Predict on the test set
predictions = knn_best.predict(X_test)

# Calculate overall test accuracy
test_accuracy = np.mean(predictions == y_test)
print(f"Test Set Accuracy: {test_accuracy:.3f}")

# --------------------------------------------------------------------------------------------

# Visualisation of the confusion matrix
from sklearn.metrics import confusion_matrix

# output path for visualisations
confusion_matrix_png = 'knn_confusion_matrix.png'


# Create confusion matrix data based of y_test and predictions
cm = confusion_matrix(y_test, predictions)

# Graph using Seaborn
plt.figure(figsize=(10, 8))

# Heat map
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix (k={best_k}, {best_distance_name} Distance)')
plt.tight_layout()
plt.savefig(confusion_matrix_png)
plt.show()

# Save the confusion matrix plot of best KNN model
plt.savefig('knn_confusion_matrix.png')
#--------------------------------------------------------------------------------------------

# Calculate the precision, recall, and F1-score for each class
classes = np.unique(y)
recall_per_class = {}

print("\nRecall per Class:")
for i, label in enumerate(classes):
    tp = cm[i, i]
    total_actual = np.sum(cm[i, :])
    recall = tp / total_actual if total_actual > 0 else 0
    recall_per_class[label] = recall
    print(f"Recall for Class {label}: {recall:.3f}")

avg_recall = np.mean(list(recall_per_class.values()))
print(f"\nAverage Recall across all classes: {avg_recall:.3f}")

f1_per_class = {}
print("\nF1-Score per Class:")
# --------------------------------------------------------------------------------------------