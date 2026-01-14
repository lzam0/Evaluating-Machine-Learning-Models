import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Classification Models
from knn_from_scratch import KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# SkLearn Analyse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Load the extracted features dataset
cleaned_csv = "data/extracted_features/hand_landmarks_sanitised.csv"
df = pd.read_csv(cleaned_csv)

# Separate features and labels
X = df.drop(columns=['label']).values
y = df['label'].values

# Create training testing set - random state utilised so that the split is reproducible (seed)
# 60% Train (to learn), 20% Val (to tune), 20% Test (final evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=2
)

print(f"Dataset Split: {len(X_train)} training samples, {len(X_test)} testing samples.")

#--------------------------------------------------------------------------------------------
# Train and evaluate the KNN model

print("Model: kNN Classifier")

# Use the best model selected (k=3 and distance = euclidean)
knn_model = KNN(k=3, distance_metric='euclidean')
knn_model.fit(X_train, y_train)

# Predict on the test set
y_pred = knn_model.predict(X_test)

# Calculate Accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Test Accuracy (k=3, Euclidean): {test_accuracy:.4%}")

# Use sklearn classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#--------------------------------------------------------------------------------------------
print("Model: Decision Tree Classifier")

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
dt_grid = GridSearchCV(dt_clf, param_grid, cv=5, scoring='accuracy')
dt_grid.fit(X_train, y_train)

# Identify best parameters from the grid search
print(f"Best Parameters: {dt_grid.best_params_}")
best_tree = dt_grid.best_estimator_

# Evaluate the best model on the test set
y_pred = best_tree.predict(X_test)
print(classification_report(y_test, y_pred)) # This gives you Accuracy and Sensitivity

#--------------------------------------------------------------------------------------------
# Random Forest Classifier

print("Model: Random Forest Classifier")

# Parameters for tuning
param_grid = {
    'n_estimators': [50, 75, 100, 200],
    'max_depth': [None, 10, 15, 20]
}

# Initialize Random Forest Classifier (random state = 2 for reproducibility)
rf_clf = RandomForestClassifier(random_state=2)

# Grid Search with Cross-Validation
# Trains model on 4 pieces and tests on the 5th piece
# Repeats this for every combination for hyper parameters in the grid
rf_grid = GridSearchCV(rf_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print (f"Best Parameters: {rf_grid.best_params_}")
best_forest = rf_grid.best_estimator_

# Evaluate the best model on the test set
y_pred = best_forest.predict(X_test)
print(classification_report(y_test, y_pred)) # This gives you Accuracy and Sensitivity
#--------------------------------------------------------------------------------------------

# Plot Confusion Matrix
def plot_cm(y_true, y_pred, model_name):

    # Calculate the accuracy
    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title(f'Confusion Matrix: {model_name}\nAccuracy: {acc:.2%}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'src/supervised learning/{model_name}.png')
    plt.show()

# kNN
y_pred_knn = knn_model.predict(X_test) 
plot_cm(y_test, y_pred_knn, "kNN")

# Decision Tree
y_pred_tree = best_tree.predict(X_test)
plot_cm(y_test, y_pred_tree, "Decision Tree")

# Random Forest
y_pred_forest = best_forest.predict(X_test)
plot_cm(y_test, y_pred_forest, "Random Forest")

#--------------------------------------------------------------------------------------------
# Identify the BEST classification model based off 'accuracy'

# Store accuracies for comparison
results = {
    "kNN": accuracy_score(y_test, y_pred_knn),
    "Decision Tree": accuracy_score(y_test, y_pred_tree),
    "Random Forest": accuracy_score(y_test, y_pred_forest)
}

# Present the best model
best_model_name = max(results, key=results.get)
print(f"\n--- Final Analysis ---")
for model, acc in results.items():
    print(f"{model} Accuracy: {acc:.4%}")

print(f"\nThe best classifier is: {best_model_name}")

#--------------------------------------------------------------------------------------------

# Identify the BEST classification model based off 'accuracy'

# Store accuracies for comparison
# results = {
#     "kNN": accuracy_score(y_test, y_pred_knn),
#     "Decision Tree": accuracy_score(y_test, y_pred_tree),
#     "Random Forest": accuracy_score(y_test, y_pred_forest)
# }

# # Present the best model
# best_model_name = max(results, key=results.get)
# print(f"\n--- Final Analysis ---")
# for model, acc in results.items():
#     print(f"{model} Accuracy: {acc:.4%}")

# print(f"\nThe best classifier is: {best_model_name}")

#--------------------------------------------------------------------------------------------
# Final Analysis: Consistency Check (CV Score vs Test Score)

# Note: For kNN, we don't have a GridSearchCV object, so we use the training accuracy 
# as a proxy for comparison if cross-validation wasn't performed on it.
def find_best_model():

    knn_train_acc = accuracy_score(y_train, knn_model.predict(X_train))

    # Store results in a dictionary for comparison
    comparison_data = {
        "kNN": {
            "cv_acc": knn_train_acc, # Proxy for consistency
            "test_acc": accuracy_score(y_test, y_pred_knn)
        },
        "Decision Tree": {
            "cv_acc": dt_grid.best_score_, # Mean accuracy from CV folds
            "test_acc": accuracy_score(y_test, y_pred_tree)
        },
        "Random Forest": {
            "cv_acc": rf_grid.best_score_, # Mean accuracy from CV folds
            "test_acc": accuracy_score(y_test, y_pred_forest)
        }
    }

    print(f"\n--- Model Consistency Analysis ---")
    print(f"{'Model':<15} | {'CV Acc':<10} | {'Test Acc':<10} | {'Gap (Diff)':<10}")
    print("-" * 55)

    consistency_results = {}

    for model, scores in comparison_data.items():
        cv = scores['cv_acc']
        test = scores['test_acc']
        gap = abs(cv - test)
        consistency_results[model] = gap
        print(f"{model:<15} | {cv:>8.2%} | {test:>9.2%} | {gap:>9.2%}")

    # The "Best" model based on consistency (smallest gap)
    most_consistent_model = min(consistency_results, key=consistency_results.get)
    # The "Best" model based on raw accuracy
    highest_accuracy_model = max(comparison_data, key=lambda x: comparison_data[x]['test_acc'])

    print(f"\nHighest Accuracy Model: {highest_accuracy_model}")
    print(f"Most Consistent Model: {most_consistent_model} (Difference of {consistency_results[most_consistent_model]:.2%})")

    # Final recommendation logic
    if most_consistent_model == highest_accuracy_model:
        print(f"Decision: {most_consistent_model} is the clear winner.")
    else:
        print(f"Decision: Consider {most_consistent_model} for stability or {highest_accuracy_model} for raw performance.")


from sklearn.metrics import precision_recall_fscore_support

def get_full_metrics(model_name, y_true, y_pred):
    # Extract macro average metrics (fair for multi-class)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    
    return {
        "Model": model_name,
        "Accuracy": f"{acc:.2%}",
        "Precision": f"{precision:.3f}",
        "Recall": f"{recall:.3f}",
        "F1-Score": f"{f1:.3f}"
    }

# Collect data for all 3 models
table_data = [
    get_full_metrics("kNN (Scratch)", y_test, y_pred_knn),
    get_full_metrics("Decision Tree", y_test, y_pred_tree),
    get_full_metrics("Random Forest", y_test, y_pred_forest)
]

# Create and display DataFrame
results_df = pd.DataFrame(table_data)
print("\n--- Poster Comparison Table ---")
print(results_df.to_string(index=False))

results_df.to_csv("supervised_results_table.csv", index=False)

find_best_model()