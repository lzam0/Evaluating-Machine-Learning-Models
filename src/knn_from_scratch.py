import math
from collections import Counter

import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

# output path for visualisations
confusion_matrix_png = 'knn_confusion_matrix.png'

# Eventually have it so that it gets parsed by a main script
# Load the extracted features dataset
data = pd.read_csv('data/extracted_features/hand_landmarks.csv') # Replace with actual path to your CSV file
#data = pd.read_csv('dummy_asl_features.csv')

# Separate features and labels
X = data.drop(columns=['instance_id', 'label']).values
y = data['label'].values

# Create training testing set - random state utilised so that the split is reproducible (seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

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
n_runs = 5

results = []

for distance_name, distance_func in distance_metrics.items():
    print(f"\nDistance Metric: {distance_name}")

    for k in k_values:
        accuracies = []

        for run in range(n_runs):
            # Different random_state each run
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=run
            )
            
            # Train the KNN model
            knn = KNN(k=k, distance_metric=distance_func)
            knn.fit(X_train, y_train)
            
            # Retrieve predictions for the test set
            predictions = knn.predict(X_test)

            # Calculate accuracy
            accuracy = np.mean(predictions == y_test)
            accuracies.append(accuracy)

        # Calculate mean and standard deviation of accuracy
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        # Store results
        results.append({
            "Distance": distance_name,
            "k": k,
            "Mean Accuracy": mean_accuracy,
            "Std Dev": std_accuracy
        })

        print(f"k={k} | Mean Accuracy={mean_accuracy:.3f} | Std={std_accuracy:.3f}")

# --------------------------------------------------------------------------------------------
# Single run with chosen parameters for confusion matrix and recall calculation 

# knn = KNN(k=5)

# # Train the KNN model
# knn.fit(X_train, y_train)

# # Retrieve predictions for the test set
# predictions = knn.predict(X_test)

# # To evaluate how accurate the model is, we can calculate the accuracy
# accuracy = np.mean(predictions == y_test) * 100
# print(f"KNN classification accuracy: {accuracy:.2f}%")

# --------------------------------------------------------------------------------------------

# Visualisation of the confusion matrix
from sklearn.metrics import confusion_matrix
def confusion_matrix():
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
    plt.title('Confusion Matrix for KNN ASL Classification')
    plt.tight_layout()
    plt.show()
    plt.savefig('knn_confusion_matrix.png')
#--------------------------------------------------------------------------------------------

# Calculate the precision, recall, and F1-score for each class

def calculate_recall_per_class():
        
    classes = np.unique(y)
    recall_per_class = {}

    # Calculate recall for each class - iterate through each class
    for i, label in enumerate(classes):

        # True Positives are on the diagonal
        tp = cm[i, i]

        # The sum of the row is the total actual samples for that class
        total_actual = np.sum(cm[i, :])
        recall = tp / total_actual if total_actual > 0 else 0
        recall_per_class[label] = recall
        print(f"Recall for Class {label}: {recall:.2f}")

    # Optional: Visualize Recall as a bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(recall_per_class.keys(), recall_per_class.values(), color='skyblue')
    plt.title('Recall per Class (KNN from Scratch)')
    plt.ylabel('Recall Score')
    plt.ylim(0, 1) # Recall is always between 0 and 1
    plt.show()
