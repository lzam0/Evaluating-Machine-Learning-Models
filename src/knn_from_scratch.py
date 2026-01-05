import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

# Load the extracted features dataset
data = pd.read_csv('data/extracted_features/hand_landmarks.csv') # Replace with actual path to your CSV file

# Separate features and labels
X = data.drop(columns=['instance_id', 'label']).values
y = data['label'].values

# Creaete training testing set - radnom state utilised so that the split is reproducible (seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# --------------------------------------------------------------------------------------------
# Create visualisations of the training data
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

# Define the euclidean distance function
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

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
knn = KNN(k=5)
knn.fit(X_train, y_train)

# Retrieve predictions for the test set
predictions = knn.predict(X_test)

# To evaluate how accurate the model is, we can calculate the accuracy
accuracy = np.mean(predictions == y_test) * 100
print(f"KNN classification accuracy: {accuracy:.2f}%")