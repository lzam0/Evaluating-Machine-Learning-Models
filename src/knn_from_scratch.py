import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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

# Creaete training testing set - radnom state utilised so that the split is reproducible (seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# --------------------------------------------------------------------------------------------
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

#--------------------------------------------------------------------------------------------
# Distances functions

# Euclidean distance - is a straight line distance between two points (like a ruler)
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Manhattan distance - is the sum of the absolute difference of their coordinates (like navigating a grid of city blocks)
def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

#--------------------------------------------------------------------------------------------

# Class KNN
class KNN:
    def __init__(self, k):
        self.k = k

    # Take training input X and output y
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # take in numpy array of new datapoints to predict
    def predict(self, new_points):
        # create a list comprehension to store predictions - iterates through each points in new_points
        predictions = [self.predict_class(point) for point in new_points]
        return np.array(predictions) # convert to np array
    
    def predict_class(self, new_point):
        # calculate the distance between new point and every point in the data
        distance = [euclidean_distance(point, new_point) for point in self.X_train]

        # find the indicies of k smallest distances - with the position of the list k smallest distances
        k_nearest_indices = np.argsort(distance)[:self.k]

        # retrieve the corresponding labels for the k nearest indices
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        most_common = Counter(k_nearest_labels).most_common(1)[0][0] # get the most common label

        return most_common
    
# Look at the 5 nearest neighbours
# In the future can look into more than just the 5 nearest neighbours (might be quite limiting!)
k_values = [5,10,25,50,100] # testing different k values 
knn = KNN(k=5)

# Train the KNN model
knn.fit(X_train, y_train)

# Retrieve predictions for the test set
predictions = knn.predict(X_test)

# To evaluate how accurate the model is, we can calculate the accuracy
accuracy = np.mean(predictions == y_test) * 100
print(f"KNN classification accuracy: {accuracy:.2f}%")

# --------------------------------------------------------------------------------------------

# Visualisation of the confusion matrix
from sklearn.metrics import confusion_matrix

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
